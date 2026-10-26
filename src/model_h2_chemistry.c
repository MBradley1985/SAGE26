/*
 * model_h2_chemistry.c -- Molecular gas fraction prescriptions.
 *
 * BR06 (Blitz & Rosolowsky 2006) pressure-based H2, KD12 (Krumholz &
 * Dekel 2012), K13 (Krumholz 2013), KMT09-style two-phase fits, GD14
 * (Gnedin & Draine 2014), the K13 depletion time, and the shared radial
 * integration over the exponential disk.  Surface densities in Msun/pc^2,
 * radii in pc; H2 masses returned in code units via the callers.
 *
 * Note: the BR06/KD12/KMT09 fit internals use float on purpose -- this is
 * frozen single-precision behaviour and part of the calibrated model output.
 * Do not promote to double (see docs/physics/units.md).
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "core_allvars.h"

#include "model_misc.h"
#include "model_starformation_and_feedback.h"  /* HYDROGEN_MASS_FRAC */

/* -------------------------------------------------------------------------
 * File-scope empirical constants (lifted per STYLE_C.md SS8).
 * -------------------------------------------------------------------------*/

/* Solar metallicity (Asplund et al. 2009): used to normalise Z' in K13.
 * Z' = Z_gas / Z_SOLAR_ASPLUND09.
 * Also defined as a file-scope constant in model_starformation_and_feedback.c
 * and model_mergers.c. */
static const double Z_SOLAR_ASPLUND09 = 0.014;

/* Solar metallicity from Grevesse & Sauval (1998): used in GD14 and KD12/KMT09.
 * D_MW = Z_gas / Z_SOLAR_GD14.
 * Also defined in model_starformation_and_feedback.c and model_mergers.c. */
static const double Z_SOLAR_GD14 = 0.02;

/* Gnedin & Draine (2014) UV-field s-parameter at U_MW = 1.0 (Milky Way ambient field).
 * s_param = pow(GD14_S_PARAM_UMW1, 0.7) from their eq. 11 / Table 1. */
static const double GD14_S_PARAM_UMW1 = 0.101;


/* Stellar disk scale height from disk scale length (Blitz & Rosolowsky 2006 eq. 9). */
static float calculate_stellar_scale_height_BR06(float disk_scale_length_pc)
{
    // BR06 equation (9): log h* = -0.23 - 0.8 log R*
    // where h* and R* are measured in parsecs
    if (disk_scale_length_pc <= 0.0) {
        return 0.0; // Default fallback value in pc
    }
    
    float log_h_star = -0.23 + 0.8 * log10(disk_scale_length_pc);
    float h_star_pc = pow(10.0, log_h_star);
    
    return h_star_pc;
}

/*
 * Core of the BR06 molecular fraction with the stellar scale height already
 * computed. The scale height depends only on the disk scale length, so the
 * radial-integration loop hoists it out and calls this variant per bin;
 * results are bit-identical to computing it in place.
 */
static float calculate_molecular_fraction_BR06_from_hstar(float gas_surface_density_msun_pc2, float stellar_surface_density_msun_pc2,
                                                          float h_star_pc)
{
    float pressure = 0.0;
    if (gas_surface_density_msun_pc2 > 0.0 && h_star_pc > 0.0) {
        float effective_sigma_stars = stellar_surface_density_msun_pc2;
        if (stellar_surface_density_msun_pc2 < 0.1) {
            effective_sigma_stars = 0.1;
        }
        const float v_g = 8.0;  // km/s, gas velocity dispersion (BR06)
        // BR06 Equation (5) - stellar-dominated approximation:
        // P_ext/k = 272 * Sigma_gas * sqrt(Sigma_*) * v_g * h_*^(-0.5)
        pressure = 272.0 * gas_surface_density_msun_pc2 * sqrt(effective_sigma_stars) * v_g / sqrt(h_star_pc);
    }
    
    if (pressure <= 0.0) {
        return 0.0;
    }

    // BR06 parameters from equation (13) for non-interacting galaxies
    // These are the exact values from the paper
    const float P0 = 4.3e4;    // Reference pressure, K cm-^3 (equation 13)
    const float alpha = 0.92;  // Power law index (equation 13)

    // BR06 Equation (11): R_mol = (P_ext/P0)^alpha
    float pressure_ratio = pressure / P0;
    float R_mol = pow(pressure_ratio, alpha);

    // Convert to molecular fraction: f_mol = R_mol / (1 + R_mol)
    // This is the standard conversion from molecular-to-atomic ratio to molecular fraction
    double f_mol = R_mol / (1.0 + R_mol);

    return f_mol;
}

/*
 * H2 molecular fraction from the Leroy et al. (2008, AJ 136, 2782) pressure law
 * (FixingGas 1).
 *
 * Their eq. 9 midplane pressure P_h = (pi/2) G Sigma_g (Sigma_g + (sigma_g/sigma_*,z) Sigma_*),
 * with sigma_g = 11 km/s and sigma_*,z = sqrt(2 pi G l_* / 7.3) Sigma_*^0.5 (their
 * eq. B3, stellar scale length over scale height = 7.3 from Kregel et al. 2002), and their Table 6 fit for
 * spirals (combined): R_mol = (P_h/P0)^alpha with log10 P0/k = 4.23, alpha = 0.80.
 * Surface densities include helium, as SAGE26's ColdGas does.
 * Returns f_H2 = R_mol / (1 + R_mol).
 */
static const double L08_LOG_P0_K        = 4.23;   /* log10 (P0/k) [K cm^-3] */
static const double L08_ALPHA           = 0.80;
static const double L08_SIGMA_GAS_CMS   = 11.0e5; /* gas velocity dispersion [cm/s] */
static const double L08_SCALE_RATIO     = 7.3;    /* stellar scale length / scale height */
static const double CM_PER_PC           = CM_PER_MPC * 1.0e-6;

float calculate_molecular_fraction_L08(float gas_surface_density_msun_pc2, float stellar_surface_density_msun_pc2,
                                       float disk_scale_length_pc)
{
    if(gas_surface_density_msun_pc2 <= 0.0f || disk_scale_length_pc <= 0.0f) {
        return 0.0f;
    }
    const double msun_pc2_to_cgs = SOLAR_MASS / (CM_PER_PC * CM_PER_PC);
    const double sg = gas_surface_density_msun_pc2 * msun_pc2_to_cgs;
    const double ss = (stellar_surface_density_msun_pc2 > 0.0f) ? stellar_surface_density_msun_pc2 * msun_pc2_to_cgs : 0.0;
    /* (sigma_g / sigma_*,z) Sigma_* = sigma_g Sigma_*^0.5 / sqrt(2 pi G l_* / 7.3) */
    const double stellar_term = (ss > 0.0)
        ? L08_SIGMA_GAS_CMS * sqrt(ss) / sqrt(2.0 * M_PI * GRAVITY * disk_scale_length_pc * CM_PER_PC / L08_SCALE_RATIO)
        : 0.0;
    const double p_over_k = 0.5 * M_PI * GRAVITY * sg * (sg + stellar_term) / BOLTZMANN;
    const double r_mol = pow(p_over_k / pow(10.0, L08_LOG_P0_K), L08_ALPHA);
    return (float)(r_mol / (1.0 + r_mol));
}

/*
 * H2 molecular fraction in Shark's BR06 model (Lagos et al. 2018, MNRAS 481, 3573,
 * Section 4.4.2 and Table 3 defaults): R_mol = (P/P0)^alpha_P (their eq. 8) with
 * P0/k = 34,673 K cm^-3 and alpha_P = 0.92; P = (pi/2) G Sigma_gas (Sigma_gas +
 * (sigma_gas/sigma_*) Sigma_*) (eq. 9, Elmegreen 1989) with sigma_gas = 10 km/s,
 * sigma_* = sqrt(pi G h_* Sigma_*) and h_* = r_star / 7.3, with r_star the half-stellar-mass
 * radius (Kregel et al. 2002).  Sigma_gas is total gas (atomic plus molecular).
 * FixingGas 2, 3.
 */
static const double SHARK_P0_K           = 34673.0;  /* K cm^-3 */
static const double SHARK_ALPHA_P        = 0.92;
static const double SHARK_SIGMA_GAS_CMS  = 10.0e5;   /* gas velocity dispersion [cm/s] */
static const double SHARK_HSTAR_RATIO    = 7.3;      /* half-stellar-mass radius / stellar scale height */
/* Lagos+18 integrate Sigma_SFR over 0-10 r_gas (half-mass) with 1 per cent accuracy;
 * 200 midpoint bins over that range are accurate to ~0.1 per cent. */
static const double SHARK_RMAX_HALF      = 10.0;
static const int    SHARK_NBINS          = 200;
static const double HALF_MASS_PER_SCALE  = 1.678346990;  /* exponential disk */

float calculate_molecular_fraction_shark(float gas_surface_density_msun_pc2, float stellar_surface_density_msun_pc2,
                                         float h_star_pc)
{
    if(gas_surface_density_msun_pc2 <= 0.0f) {
        return 0.0f;
    }
    const double msun_pc2_to_cgs = SOLAR_MASS / (CM_PER_PC * CM_PER_PC);
    const double sg = gas_surface_density_msun_pc2 * msun_pc2_to_cgs;
    const double ss = (stellar_surface_density_msun_pc2 > 0.0f) ? stellar_surface_density_msun_pc2 * msun_pc2_to_cgs : 0.0;
    const double veldisp_star = (h_star_pc > 0.0f) ? sqrt(M_PI * GRAVITY * h_star_pc * CM_PER_PC * ss) : 0.0;
    const double star_comp = (ss > 0.0 && veldisp_star > 0.0) ? (SHARK_SIGMA_GAS_CMS / veldisp_star) * ss : 0.0;
    const double p_over_k = 0.5 * M_PI * GRAVITY * sg * (sg + star_comp) / BOLTZMANN;
    const double r_mol = pow(p_over_k / SHARK_P0_K, SHARK_ALPHA_P);
    return (float)(r_mol / (1.0 + r_mol));
}

/*
 * Mass-weighted Shark molecular fraction of an exponential gas component of mass
 * m_gas [Msun] with half-mass radius r_half_gas_pc [pc], and stars of mass m_star
 * [Msun] in an exponential with half-mass radius r_half_star_pc [pc], integrated over
 * 0-10 r_half_gas (Lagos et al. 2018, Section 4.4.2).  For bulges (Section 4.4.3) the
 * caller passes the bulge size: the gas settles in an exponential of scale length
 * r_bulge/1.67.
 */
double shark_molecular_fraction(const double m_gas, const double re_gas_pc, const double m_star, const double re_star_pc)
{
    if(m_gas <= 0.0 || re_gas_pc <= 0.0) {
        return 0.0;
    }
    const double sg0 = m_gas / (2.0 * M_PI * re_gas_pc * re_gas_pc);
    const double ss0 = (m_star > 0.0 && re_star_pc > 0.0) ? m_star / (2.0 * M_PI * re_star_pc * re_star_pc) : 0.0;
    const double h_star = (re_star_pc > 0.0) ? HALF_MASS_PER_SCALE * re_star_pc / SHARK_HSTAR_RATIO : 0.0;
    const double dr = SHARK_RMAX_HALF * HALF_MASS_PER_SCALE * re_gas_pc / SHARK_NBINS;
    double mmol = 0.0, mtot = 0.0;
    for(int i = 0; i < SHARK_NBINS; i++) {
        const double r = (i + 0.5) * dr;
        const double sg = sg0 * exp(-r / re_gas_pc);
        const double ss = (ss0 > 0.0) ? ss0 * exp(-r / re_star_pc) : 0.0;
        const double dm = 2.0 * M_PI * r * sg * dr;
        mmol += dm * calculate_molecular_fraction_shark((float)sg, (float)ss, (float)h_star);
        mtot += dm;
    }
    return (mtot > 0.0) ? mmol / mtot : 0.0;
}

/*
 * Pressure-based molecular fraction for the BR06 family (SFprescription 1, 3):
 * Blitz & Rosolowsky (2006); Leroy et al. (2008) with FixingGas 1; Shark with
 * FixingGas 2, 3.
 */
float calculate_molecular_fraction_pressure(float gas_surface_density_msun_pc2, float stellar_surface_density_msun_pc2,
                                            float disk_scale_length_pc, const struct params *run_params)
{
    if(FG_SHARK_SF(run_params)) {
        return calculate_molecular_fraction_shark(gas_surface_density_msun_pc2, stellar_surface_density_msun_pc2,
                                                  HALF_MASS_PER_SCALE * disk_scale_length_pc / SHARK_HSTAR_RATIO);
    }
    if(FG_LEROY08(run_params)) {
        return calculate_molecular_fraction_L08(gas_surface_density_msun_pc2, stellar_surface_density_msun_pc2, disk_scale_length_pc);
    }
    return calculate_molecular_fraction_BR06(gas_surface_density_msun_pc2, stellar_surface_density_msun_pc2, disk_scale_length_pc);
}

/*
 * H2 molecular fraction from the Blitz & Rosolowsky (2006) midplane pressure
 * relation.
 *
 * Computes the midplane pressure from their eq. 5 (stellar-dominated
 * approximation, gas velocity dispersion 8 km/s), forms
 * R_mol = (P_ext/P_0)^alpha with P_0 = 4.3e4 K cm^-3 and alpha = 0.92
 * (their eq. 11 with the eq. 13 non-interacting fit), and returns
 * f_H2 = R_mol / (1 + R_mol) in [0, 1].
 *
 * gas_surface_density_msun_pc2     : Sigma_gas  [Msun/pc^2]
 * stellar_surface_density_msun_pc2 : Sigma_star [Msun/pc^2]
 * disk_scale_length_pc             : r_s [pc]; sets the stellar scale height
 */
float calculate_molecular_fraction_BR06(float gas_surface_density_msun_pc2, float stellar_surface_density_msun_pc2,
                                        float disk_scale_length_pc)
{
    if (disk_scale_length_pc <= 0.0) {
        return 0.0;
    }
    const float h_star_pc = calculate_stellar_scale_height_BR06(disk_scale_length_pc);
    return calculate_molecular_fraction_BR06_from_hstar(gas_surface_density_msun_pc2, stellar_surface_density_msun_pc2, h_star_pc);
}

/*
 * Radially integrated molecular fraction and SFR (used by SF prescriptions 2-6).
 *
 * Integrates f_H2(r) * Sigma_gas(r) over the exponential disk, using the
 * selected molecular fraction prescription per annulus. Optionally returns the
 * integrated SFR in code units via strdot_code_out (may be NULL).
 */
float calculate_molecular_fraction_radial_integration(const int gal, struct GALAXY *galaxies,
                                                      const struct params *run_params,
                                                      double *strdot_code_out)
{
    const float h = run_params->Hubble_h;  /* float on purpose: frozen single-precision behaviour, do not promote (see docs/physics/units.md) */
    const float rs_pc = CODE_LENGTH_TO_PC(galaxies[gal].DiskScaleRadius, h);  // Scale radius in pc

    if (rs_pc <= 0.0 || galaxies[gal].ColdGas <= 0.0) {
        return 0.0;
    }

    // Total masses in physical units (M_sun); stellar uses disk-only (no bulge)
    const float M_gas_total  = CODE_MASS_TO_MSUN(galaxies[gal].ColdGas, h);
    const float M_disk_star  = CODE_MASS_TO_MSUN(galaxies[gal].StellarMass - galaxies[gal].BulgeMass, h);

    // Central surface densities for exponential profiles: Sigma0 = M_total / (2pi r_s^2)
    const float sigma_gas_0  = M_gas_total / (2.0 * M_PI * rs_pc * rs_pc);
    const float sigma_star_0 = (M_disk_star > 0.0) ? M_disk_star / (2.0 * M_PI * rs_pc * rs_pc) : 0.0;

    // FixingGas 2, 3: Shark's integration (Lagos+18 Section 4.4.2), gas and stars sharing the SAGE disk scale length
    if(FG_SHARK_SF(run_params)) {
        galaxies[gal].H2gas = (float)(shark_molecular_fraction(M_gas_total, rs_pc, M_disk_star, rs_pc) * galaxies[gal].ColdGas * HYDROGEN_MASS_FRAC);
        return galaxies[gal].H2gas;
    }

    // Radial integration parameters from run_params (defaults: 25 bins, 5*r_s)
    const int   N_BINS = run_params->H2RadialNBins;
    const float R_MAX  = (float)(run_params->H2RadialRMaxFactor) * rs_pc;
    const float dr     = R_MAX / N_BINS;

    // Prescription-specific quantities that don't depend on radius (compute before loop)
    const int sfpres = run_params->SFprescription;

    // GD14 (prescription 7): precompute radially-invariant GD14 parameters
    float met7 = 0.0f, D_MW7 = 1e-4f, S7 = 0.0f, s_param7 = 0.0f;
    float D_star7 = 0.0f, g7 = 0.0f, Sigma_R1_7 = 1e10f, alpha7 = 1.0f;
    if(sfpres == 7) {
        met7 = (galaxies[gal].ColdGas > 0.0f) ? (float)(galaxies[gal].MetalsColdGas / galaxies[gal].ColdGas) : 0.0f;
        D_MW7 = met7 / 0.02f; if(D_MW7 < 1e-4f) D_MW7 = 1e-4f;
        S7 = 3.0f * rs_pc / 100.0f;
        s_param7 = powf(0.001f + 0.1f, 0.7f);  // U_MW=1.0
        D_star7 = 0.17f * (2.0f + powf(S7, 5.0f)) / (1.0f + powf(S7, 5.0f));
        g7 = sqrtf(D_MW7 * D_MW7 + D_star7 * D_star7);
        Sigma_R1_7 = (g7 > 0.0f) ? (40.0f / g7) * s_param7 / (1.0f + s_param7) : 1e10f;
        alpha7 = 1.0f + 0.7f * sqrtf(s_param7) / (1.0f + s_param7);
    }

    // BR06 (prescriptions 1/3): stellar scale height depends only on the disk
    // scale length -- compute once instead of once per radial bin.
    const float h_star_br06 = sf_prescription_is_br06(sfpres)
                              ? calculate_stellar_scale_height_BR06(rs_pc) : 0.0f;

    // KD12/KMT09/K13 (prescriptions 4/5/6): the disk metallicity and every
    // quantity derived from it alone are radius-independent -- hoisted out of
    // the loop. Values are identical to the previous per-bin evaluation.
    const float met_disk = (galaxies[gal].ColdGas > 0.0f)
                           ? (float)(galaxies[gal].MetalsColdGas / galaxies[gal].ColdGas) : 0.0f;
    float Zp5 = 0.0f, lognum5 = 0.0f;
    if(sfpres == 5) {
        Zp5 = (met_disk > 0.0f) ? met_disk / 0.02f : 0.0f;
        const float chi5 = 0.77f * (1.0f + 3.1f * powf(Zp5, 0.365f));
        lognum5 = logf(1.0f + 0.6f * chi5 + 0.01f * chi5 * chi5);
    }
    float Zp6 = 0.0f, lognum6 = 0.0f;
    if(sfpres == 6) {
        Zp6 = met_disk / 0.014f; if(Zp6 < 0.01f) Zp6 = 0.01f;
        const float chi6 = 3.1f * (1.0f + 3.1f * powf(Zp6, 0.365f)) / 4.1f;
        lognum6 = logf(1.0f + 0.6f * chi6 + 0.01f * chi6 * chi6);
    }

    // K13 radially-integrated SFR (only computed when caller requests it via strdot_code_out)
    double SFR_K13_total = 0.0;
    float Z_prime_k13 = 0.01f;
    if(strdot_code_out != NULL && galaxies[gal].ColdGas > 0.0f) {
        float Z_gas_k13 = (float)(galaxies[gal].MetalsColdGas / galaxies[gal].ColdGas);
        Z_prime_k13 = Z_gas_k13 / 0.014f;
        if(Z_prime_k13 < 0.01f) Z_prime_k13 = 0.01f;
    }

    // Integrate molecular gas mass
    float M_H2_total = 0.0;

    for (int i = 0; i < N_BINS; i++) {
        // Bin center radius
        const float r = (i + 0.5f) * dr;

        // Exponential surface density profiles: Sigma(r) = Sigma0 exp(-r/r_s)
        const float exp_factor = expf(-r / rs_pc);
        const float sigma_gas_r = sigma_gas_0 * exp_factor;
        const float sigma_star_r = sigma_star_0 * exp_factor;

        // Skip bins with negligible gas
        if (sigma_gas_r < 1e-3f) continue;

        // Calculate molecular fraction at this radius using the appropriate prescription
        float f_mol_r = 0.0f;

        if(sf_prescription_is_br06(sfpres)) {
            // BR06 (scale height precomputed above)
            f_mol_r = FG_LEROY08(run_params) ? calculate_molecular_fraction_L08(sigma_gas_r, sigma_star_r, rs_pc)
                    : calculate_molecular_fraction_BR06_from_hstar(sigma_gas_r, sigma_star_r, h_star_br06);

        } else if(sfpres == 4) {
            // KD12
            f_mol_r = calculate_H2_fraction_KD12(sigma_gas_r, met_disk, 5.0f);

        } else if(sfpres == 5) {
            // KMT09 (Zp5 and the chi log-numerator precomputed above)
            float fc5 = 3.0f;
            float tau_c5 = 0.066f * fc5 * Zp5 * sigma_gas_r;
            float s5 = (sigma_gas_r > 0.0f && tau_c5 > 1e-10f)
                       ? lognum5 / (0.6f * tau_c5) : 100.0f;
            f_mol_r = (s5 < 2.0f) ? 1.0f - (3.0f * s5) / (4.0f + s5) : 0.0f;
            if(f_mol_r < 0.0f) f_mol_r = 0.0f;

        } else if(sfpres == 6) {
            // K13 (Zp6 and the chi log-numerator precomputed above)
            float fc6 = 5.0f;
            float tau_c6 = 0.066f * fc6 * Zp6 * sigma_gas_r;
            float s6 = (tau_c6 > 0.0f) ? lognum6 / (0.6f * tau_c6) : 100.0f;
            f_mol_r = (s6 < 2.0f) ? 1.0f - (0.75f * s6) / (1.0f + 0.25f * s6) : 0.0f;
            if(f_mol_r < 0.0f) f_mol_r = 0.0f;

        } else if(sfpres == 7) {
            // GD14 -- galaxy-wide quantities precomputed above
            float q7 = (Sigma_R1_7 > 0.0f && sigma_gas_r > 0.0f)
                       ? powf(sigma_gas_r / Sigma_R1_7, alpha7) : 0.0f;
            f_mol_r = q7 / (1.0f + q7);
            if(f_mol_r > 1.0f) f_mol_r = 1.0f;

        } else {
            f_mol_r = 0.0f;
        }

        // dM_H2 = 2pi r * Sigma_gas * 0.74 * f_mol * dr: f_mol is fraction of H that is H2,
        // so multiply by HYDROGEN_MASS_FRAC to convert total gas surface density to hydrogen.
        const float dM_H2 = 2.0f * (float)M_PI * r * sigma_gas_r * 0.74f * f_mol_r * dr;

        M_H2_total += dM_H2;

        // K13 SFR: Sigma_gas(r) / t_dep(r) * 2pi r dr [Msun/Gyr], using local f_H2 from base prescription
        if(strdot_code_out != NULL) {
            double t_dep_r = calculate_tdep_K13_Gyr(sigma_gas_r, sigma_star_r, rs_pc, Z_prime_k13, f_mol_r);
            if(t_dep_r > 0.0)
                SFR_K13_total += (double)sigma_gas_r / t_dep_r * 2.0 * M_PI * r * dr;
        }
    }

    // Convert back to code units (10^10 M_sun/h)
    const float H2_code_units = MSUN_TO_CODE_MASS(M_H2_total, h);

    // Write K13 SFR and effective depletion time when requested
    if(strdot_code_out != NULL) {
        // SFR_K13_total [Msun/Gyr] -> code units [(10^10 Msun/h) / (UnitTime_in_Megayears Myr)]
        *strdot_code_out = MSUN_TO_CODE_MASS(SFR_K13_total, h) * run_params->UnitTime_in_Megayears / 1000.0;
        // Effective global depletion time [Gyr] = M_gas / SFR_K13
        galaxies[gal].H2DepletionTime_Gyr = (SFR_K13_total > 0.0)
                                            ? (float)(M_gas_total / SFR_K13_total) : -1.0f;
    }

    // Store and return
    galaxies[gal].H2gas = H2_code_units;
    return H2_code_units;
}

/*
 * Gas depletion time in Gyr from Krumholz (2013).
 *
 * Returns tau_dep = min(tau_2phase, tau_hydro_star, tau_hydro_gas) from K13 eq. 28.
 * Pass f_H2 from BR06 for the BR06+K13 hybrid, or K13's own f_H2 for pure K13.
 */
double calculate_tdep_K13_Gyr(float Sigma_gas_msun_pc2, float Sigma_star_msun_pc2, float rs_pc, float Z_prime, float f_H2)
{
    const double fc = 5.0;  // clumping factor for ~kpc scales (K13 Section 3.1)

    // Molecule-rich depletion time: t_dep_2p = 3.1 Gyr / (f_H2 * Sigma^0.25)
    double t_dep_2p = (f_H2 > 1e-6 && Sigma_gas_msun_pc2 > 1e-10)
                      ? 3.1 / (f_H2 * pow(Sigma_gas_msun_pc2, 0.25)) : 1.0e5;

    // Hydrostatic limits (K13 Eqs 21-22)
    double t_hydro_star = 1.0e10, t_hydro_gas = 1.0e10;
    if(Sigma_gas_msun_pc2 > 1e-10) {
        double h_z       = calculate_stellar_scale_height_BR06(rs_pc);
        double rho_sd_2  = (h_z > 0.0 && Sigma_star_msun_pc2 > 0.0)
                           ? (Sigma_star_msun_pc2 / (2.0 * h_z)) / 0.01 : 1e-4;
        if(rho_sd_2 < 1e-4) rho_sd_2 = 1e-4;
        if(Z_prime  < 0.01) Z_prime   = 0.01;
        t_hydro_star = 3.1 / pow(Sigma_gas_msun_pc2, 0.25)
                       + 100.0 / ((fc/5.0) * Z_prime * sqrt(rho_sd_2) * Sigma_gas_msun_pc2);
        t_hydro_gas  = 3.1 / pow(Sigma_gas_msun_pc2, 0.25)
                       + 360.0 / ((fc/5.0) * Z_prime * pow(Sigma_gas_msun_pc2, 2.0));
    }

    double t_dep = t_dep_2p;
    if(t_hydro_star < t_dep) t_dep = t_hydro_star;
    if(t_hydro_gas  < t_dep) t_dep = t_hydro_gas;
    return t_dep;
}

/*
 * H2 molecular fraction from the Krumholz & Dekel (2012) CNM-shielding model.
 *
 * Solves the non-linear s-function from KD12 eqs. 14-17 given the gas surface
 * density, metallicity (in Z/Z_sun), and a clumping factor. Returns f_H2 in [0, 1].
 */
float calculate_H2_fraction_KD12(const float surface_density_msun_pc2, const float metallicity, const float clumping_factor)
{
    if (surface_density_msun_pc2 <= 0.0) {
        return 0.0;
    }
    
    // Metallicity normalized to solar (Grevesse & Sauval 1998: Z_sun = 0.02).
    // Z0 = (M_Z/M_g)/Z_sun as defined in KD12 equation after (17).
    // No floor: at Z=0, tau_c=0 -> s=100 -> f_H2=0 is handled by the guard below.
    // A floor here would give f_H2->1 for primordial gas at high Sigma, which is wrong.
    float Z0 = (metallicity > 0.0f) ? metallicity / (float)Z_SOLAR_GD14 : 0.0f;
    
    // Convert surface density from M_sun/pc^2 to g/cm^2
    // Conversion: 1 M_sun/pc^2 = 2.088 * 10^-4 g/cm^2
    float Sigma_gcm2 = surface_density_msun_pc2 * 2.088e-4;
    
    // Surface density normalized to 1 g/cm^2 (as defined after KD12 Eq. 16)
    // Sigma_0 = Sigma / (1 g cm^-2)
    float Sigma_0 = Sigma_gcm2;  // dimensionless, in units of 1 g/cm^2
    
    // Calculate dust optical depth parameter (KD12 Eq. 21)
    // tau_c = 320 * c * Z0 * Sigma_0
    // where c is the clumping factor:
    //   c ~ 1 for Sigma measured on 100 pc scales
    //   c ~ 5 for Sigma measured on ~1 kpc scales (from text after Eq. 21)
    float tau_c = 320.0 * clumping_factor * Z0 * Sigma_0;
    
    // Self-shielding parameter chi (KD12 Eq. 20)
    // chi = 3.1 * (1 + Z0^0.365) / 4.1
    float chi = 3.1 * (1.0 + pow(Z0, 0.365)) / 4.1;
    
    // Compute s parameter (KD12 Eq. 19)
    // s = ln(1 + 0.6*chi + 0.01*chi^2) / (0.6 * tau_c)
    float chi_sq = chi * chi;
    float s;
    if(tau_c > 1e-10) {
        s = log(1.0 + 0.6 * chi + 0.01 * chi_sq) / (0.6 * tau_c);
    } else {
        s = 100.0;  // Large s implies f_H2 -> 0 (atomic dominated)
    }
    
    // Molecular fraction (KD12 Eq. 18)
    // f_H2 = 1 - (3/4) * s/(1 + 0.25*s)  for s < 2
    // f_H2 = 0                            for s >= 2
    float f_H2;
    if (s < 2.0) {
        f_H2 = 1.0 - 0.75 * s / (1.0 + 0.25 * s);
    } else {
        f_H2 = 0.0;
    }
    
    // Ensure fraction stays within bounds
    if (f_H2 < 0.0) f_H2 = 0.0;
    if (f_H2 > 1.0) f_H2 = 1.0;

    return f_H2;
}

/*
 * H2 molecular fraction from the Krumholz (2013) two-phase ISM model.
 *
 * Implements the K13 "two-phase" (2p) approximation for the slab geometry.
 * Sigma_gas_msun_pc2 is the gas surface density in Msun/pc^2.
 * metallicity is the absolute metal fraction Z_gas (not Z') -- normalised
 * internally to Z_SOLAR_ASPLUND09 = 0.014.  clumping_factor is the CNM
 * clumping factor (typically 5.0 for kpc-scale surface densities).
 * Returns f_H2 in [0, 1].
 */
double calculate_H2_fraction_K13(double Sigma_gas_msun_pc2, double metallicity, double clumping_factor)
{
    if(Sigma_gas_msun_pc2 <= 0.0) return 0.0;

    double Z_prime = (metallicity > 0.0) ? metallicity / Z_SOLAR_ASPLUND09 : 0.0;
    if(Z_prime < 0.01) Z_prime = 0.01;

    const double chi_2p = 3.1 * (1.0 + 3.1 * pow(Z_prime, 0.365)) / 4.1;
    const double tau_c  = 0.066 * clumping_factor * Z_prime * Sigma_gas_msun_pc2;
    const double s      = (tau_c > 0.0) ?
        log(1.0 + 0.6*chi_2p + 0.01*chi_2p*chi_2p) / (0.6*tau_c) : 100.0;

    double f_H2 = (s < 2.0) ? 1.0 - (0.75*s) / (1.0 + 0.25*s) : 0.0;
    if(f_H2 < 0.0) f_H2 = 0.0;
    if(f_H2 > 1.0) f_H2 = 1.0;
    return f_H2;
}

/*
 * H2 molecular fraction from the Gnedin & Draine (2014) self-shielding model.
 *
 * Implements the GD14 slab-geometry model (their eq. 11) assuming U_MW = 1.0
 * (Milky Way UV field).  Sigma_gas_msun_pc2 is the gas surface density in
 * Msun/pc^2.  metallicity is the absolute metal fraction Z_gas -- normalised
 * internally to Z_SOLAR_GD14 = 0.02 (Grevesse & Sauval 1998).  rs_pc is the
 * disk scale radius in pc, used to evaluate the clumping/turbulence parameter
 * S = 3 * rs_pc / 100 from their eq. 11.
 * Returns f_H2 in [0, 1].
 */
double calculate_H2_fraction_GD14(double Sigma_gas_msun_pc2, double metallicity, double rs_pc)
{
    if(Sigma_gas_msun_pc2 <= 0.0) return 0.0;

    double D_MW = (metallicity > 0.0) ? metallicity / Z_SOLAR_GD14 : 0.0;
    if(D_MW < 1e-4) D_MW = 1e-4;

    const double S        = 3.0 * rs_pc / 100.0;
    const double s_param  = pow(GD14_S_PARAM_UMW1, 0.7);  /* U_MW = 1.0 */
    const double D_star   = 0.17 * (2.0 + pow(S, 5.0)) / (1.0 + pow(S, 5.0));
    const double g        = sqrt(D_MW*D_MW + D_star*D_star);
    const double Sigma_R1 = (g > 0.0) ? (40.0/g) * (s_param/(1.0+s_param)) : 1e10;
    const double alpha    = 1.0 + 0.7*sqrt(s_param) / (1.0 + s_param);
    const double q        = (Sigma_R1 > 0.0) ?
        pow(Sigma_gas_msun_pc2 / Sigma_R1, alpha) : 0.0;

    double f_H2 = q / (1.0 + q);
    if(f_H2 < 0.0) f_H2 = 0.0;
    if(f_H2 > 1.0) f_H2 = 1.0;
    return f_H2;
}

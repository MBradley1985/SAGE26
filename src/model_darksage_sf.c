/*
 * model_darksage_sf.c -- annulus-resolved H2 and star formation (DarkSAGEMode == 1).
 *
 * Applies the same SFprescription / H2DepletionMode laws as
 * model_starformation_and_feedback.c, but per disc annulus (DiscGas[i],
 * DiscStars[i], DiscRadii[i]) following Stevens, Croton & Mutch (2016).
 * The annuli are themselves a radial integration over the actual disc, so
 * H2RadialIntegrationOn and H2DiskAreaOption are not used in this mode.
 *
 * The dynamical-time laws use the galaxy's t_dyn = 3 r_s / Vvir in every
 * annulus, as the bulk laws do (and DarkSage's SFprescription 2): an
 * annulus's own r_i / v_circ,i (~1 Myr at the centre) empties the inner disc.
 * The local circular velocity enters only the depletion-time laws.
 *
 * The annuli are the state: these functions supply the per-annulus SFR; the
 * stars, feedback and yields are booked annulus by annulus in
 * model_starformation_and_feedback.c.
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "core_allvars.h"
#include "macros.h"

#include "model_darksage_sf.h"
#include "model_h2_chemistry.h"
#include "model_starformation_and_feedback.h"
#include "model_misc.h"
#include "model_darksage_stellar_pop.h"
#include "model_darksage_instability.h"

/* Same constants as the bulk recipes in model_starformation_and_feedback.c */
static const double DS_SF_DISK_RADIUS_FRAC = 3.0;   /* star-forming radius in disc scale radii (SF_DISK_RADIUS_FRAC) */
static const double DS_SOMERVILLE25_SIGMA_CRIT = 30.0 / (M_PI * 4.302e-3);  /* Msun/pc^2 */
static const double DS_SOMERVILLE25_F_DENSE    = 0.5;
static const double DS_Z_SOLAR_ASPLUND09       = 0.014;

/* SFprescription == 8, DarkSage update_HI_H2() with H2prescription 0 (Stevens et al. 2024 Sec. 6):
 * H2/HI = DS_H2_FACTOR (P / P_0)^DS_H2_EXPONENT (Blitz & Rosolowsky 2006 as fitted by Fu et al. 2010),
 * P_0 = 5.93e-12 dyn cm^-2; neutral-fraction constants of DarkSage core_init.c [cgs] */
static const double DS_P0_CGS = 5.93e-12;
static const double DS_H2_FACTOR = 1.0;
static const double DS_H2_EXPONENT = 0.92;
static const double DS_UNI_ION_HIST = 0.24949;
static const double DS_UNI_FNEUT_BG = 2.351e-5;

/*
 * H2 mass of annulus i under DarkSage's H2 prescription (update_HI_H2(), H2prescription 0).
 *
 *   1. Midplane pressure P = (pi/2) G Sigma_gas (Sigma_gas + (sigma_gas / sigma_*) Sigma_*) h^2,
 *      the stars counting only when the stellar and gas discs are within ThetaThresh;
 *      H2/HI = (P / P_0)^0.92.
 *   2. Neutral fraction from the annulus's own young stars: its recent SFR (from the youngest
 *      populated age bin) keeps f_ion^2 ~ SFR sigma^2 area^2 / M_gas^3 of the hydrogen ionised,
 *      f_neutral = 1 - sqrt(uni_ion_hist (3X+1) (sigma area / X)^2 SFR / M_gas^3).
 *   3. The UV background caps it: gamma = uni_fneut_bg (3X+1) Gamma_HI(z) / X (sigma area / M_gas)^2,
 *      f_neutral <= 1 + gamma - sqrt(gamma^2 + 2 gamma).
 *   4. H2 = X f_neutral M_gas / (1 / (H2/HI) + 1).
 * Code units; X = DARKSAGE_X_H.  *f_neutral_out receives the neutral fraction.
 */
static double h2_darksage(const int p, const int i, const double gas, double *f_neutral_out,
                          const struct GALAXY *galaxies, const struct params *run_params)
{
    const struct GALAXY *g = &galaxies[p];
    const double X = DARKSAGE_X_H;
    const double h = run_params->Hubble_h;
    const double r_in = g->DiscRadii[i], r_out = g->DiscRadii[i+1];
    const double area = M_PI * (r_out * r_out - r_in * r_in);
    *f_neutral_out = 1.0;
    if(area <= 0.0 || gas <= 0.0) return 0.0;

    const double sigma_gas = darksage_cold_gas_dispersion(run_params->ZZ[g->SnapNum], run_params);
    const double cos_angle = g->SpinStars[0] * g->SpinGas[0] + g->SpinStars[1] * g->SpinGas[1] + g->SpinStars[2] * g->SpinGas[2];
    const int coplanar = (cos_angle >= cos(run_params->ThetaThresh * M_PI / 180.0));

    /* Step 1: pressure-based H2/HI */
    double P;
    if(coplanar && g->DiscStars[i] > 0.0f && g->VelDispStars[i] > 0.0f) {
        P = 0.5 * M_PI * run_params->G * gas * (gas + sigma_gas / g->VelDispStars[i] * g->DiscStars[i]) / (area * area) * h * h;
    } else {
        P = 0.5 * M_PI * run_params->G * (gas / area) * (gas / area) * h * h;
    }
    const double P0 = DS_P0_CGS / run_params->UnitMass_in_g * run_params->UnitLength_in_cm
                      * run_params->UnitTime_in_s * run_params->UnitTime_in_s;
    const double ratio = DS_H2_FACTOR * pow(P / P0, DS_H2_EXPONENT);

    /* Step 2: recent SFR of the annulus from its youngest populated age bin */
    const int k = g->AgeBinNow;
    double sfr_guess = 0.0;
    const double t_left = run_params->AgeBinEdge[k+1] - g->TimeNow;
    if(g->DiscStarsAge[i][k] > 0.0f && t_left > 0.0) {
        double inst[2];
        darksage_instant_return_and_sn(k, run_params, inst);
        sfr_guess = g->DiscStarsAge[i][k] / ((1.0 - inst[0]) * t_left);
    } else if(k < N_AGE_BINS - 1 && g->DiscStarsAge[i][k+1] > 0.0f) {
        double out[2];
        darksage_return_and_sn(g->TimeNow, 0.5 * (run_params->AgeBinEdge[k+2] + run_params->AgeBinEdge[k+1]), run_params, out);
        sfr_guess = g->DiscStarsAge[i][k+1] / ((1.0 - out[0]) * (run_params->AgeBinEdge[k+2] - run_params->AgeBinEdge[k+1]));
    }
    const double UL = run_params->UnitLength_in_cm, UM = run_params->UnitMass_in_g, UT = run_params->UnitTime_in_s;
    const double uni_ion_hist = DS_UNI_ION_HIST * pow(UL, 6.0) / (UM * UM * UT * UT * UT) / h;
    const double uni_fneut_bg = DS_UNI_FNEUT_BG * pow(UL, 6.0) / (UM * UM * UT * UT * UT) / h;
    const double sa = sigma_gas * area;
    double f_neutral = 1.0 - sqrt(uni_ion_hist * (3.0 * X + 1.0) * (sa / X) * (sa / X) * sfr_guess / (gas * gas * gas));

    /* Step 3: the UV background caps it */
    const double gamma = uni_fneut_bg * (3.0 * X + 1.0) * run_params->GammaHI[g->SnapNum] / X * (sa / gas) * (sa / gas);
    double f_max = 1.0 + gamma - sqrt(gamma * gamma + 2.0 * gamma);
    if(f_max < 0.0) f_max = 0.0;
    if(f_neutral > f_max) f_neutral = f_max;
    if(f_neutral < 0.0) f_neutral = 0.0;
    *f_neutral_out = f_neutral;

    /* Step 4 */
    return (ratio > 0.0) ? X * f_neutral * gas / (1.0 / ratio + 1.0) : 0.0;
}

/* KMT09 two-phase fit, as inlined in sfr_kmt09() */
static double fH2_kmt09(const double Sigma_gas, const double Z)
{
    const double Z_prime = (Z > 0.0) ? Z / 0.02 : 0.0;
    const double clumping_factor = 3.0;
    const double tau_c = 0.066 * clumping_factor * Z_prime * Sigma_gas;
    const double chi = 0.77 * (1.0 + 3.1 * pow(Z_prime, 0.365));
    const double s = (clumping_factor * Sigma_gas > 0.0 && tau_c > 1e-10)
        ? log(1.0 + 0.6 * chi + 0.01 * chi * chi) / (0.6 * tau_c) : 100.0;
    return (s < 2.0) ? 1.0 - (3.0 * s) / (4.0 + s) : 0.0;
}

double darksage_star_formation_rate(const int p, double sfr_bin[N_BINS],
                                    struct GALAXY *galaxies, const struct params *run_params)
{
    const int sfp = run_params->SFprescription;
    const float h = run_params->Hubble_h;
    const double rs_pc = CODE_LENGTH_TO_PC(galaxies[p].DiskScaleRadius, h);
    const double gyr_to_code = SEC_PER_GIGAYEAR * run_params->Hubble_h / run_params->UnitTime_in_s;

    double total_sfr = 0.0, total_h2 = 0.0;

    for(int i = 0; i < N_BINS; i++) {
        sfr_bin[i] = 0.0;
        galaxies[p].DiscH2[i] = 0.0;
    }
    galaxies[p].H2gas = 0.0;

    if(galaxies[p].Vvir <= 0.0 || galaxies[p].DiskScaleRadius <= 0.0) {
        return 0.0;
    }
    /* Galaxy dynamical time, as the bulk laws */
    const double tdyn = DS_SF_DISK_RADIUS_FRAC * galaxies[p].DiskScaleRadius / galaxies[p].Vvir;

    for(int i = 0; i < N_BINS; i++) {
        const double gas = galaxies[p].DiscGas[i];
        if(gas <= 0.0) continue;

        const double r_in = galaxies[p].DiscRadii[i];
        const double r_out = galaxies[p].DiscRadii[i+1];
        const double r_mid = 0.5 * (r_in + r_out);   /* Mpc/h */
        const double area_pc2 = M_PI * (CODE_LENGTH_TO_PC(r_out, h) * CODE_LENGTH_TO_PC(r_out, h)
                                        - CODE_LENGTH_TO_PC(r_in, h) * CODE_LENGTH_TO_PC(r_in, h));
        if(area_pc2 <= 0.0 || r_mid <= 0.0) continue;

        const double Sigma_gas = CODE_MASS_TO_MSUN(gas, h) / area_pc2;
        const double Sigma_star = CODE_MASS_TO_MSUN(galaxies[p].DiscStars[i], h) / area_pc2;
        const double Z = galaxies[p].DiscGasMetals[i] / gas;

        /* Local circular velocity from the j-bin, v = j/r, for the depletion-time laws */
        const double j_mid = 0.5 * (run_params->DiscBinEdge[i] + run_params->DiscBinEdge[i+1]);
        const double v_circ = (j_mid > 0.0) ? j_mid / r_mid : galaxies[p].Vvir;

        /* --- molecular fraction --- */
        double f_H2 = 0.0;
        switch(sfp) {
            case 1: case 3: f_H2 = calculate_molecular_fraction_BR06(Sigma_gas, Sigma_star, rs_pc); break;
            case 4: f_H2 = calculate_H2_fraction_KD12(Sigma_gas, Z, 5.0); break;
            case 5: f_H2 = fH2_kmt09(Sigma_gas, Z); break;
            case 6: f_H2 = calculate_H2_fraction_K13(Sigma_gas, Z, 5.0); break;
            case 7: f_H2 = calculate_H2_fraction_GD14(Sigma_gas, Z, rs_pc); break;
            default: f_H2 = 0.0; break;   /* 0, 2: no H2 tracking */
        }
        if(f_H2 < 0.0) f_H2 = 0.0;
        if(f_H2 > 1.0) f_H2 = 1.0;
        double f_neutral = 1.0;
        const double h2 = (sfp == 8) ? h2_darksage(p, i, gas, &f_neutral, galaxies, run_params)
                                     : f_H2 * gas * HYDROGEN_MASS_FRAC;
        galaxies[p].DiscFNeutral[i] = (float)f_neutral;
        galaxies[p].DiscH2[i] = h2;
        total_h2 += h2;

        /* --- star formation law --- */
        double sfr = 0.0;
        if((FG_TDEP(run_params) || sfp == 8) && sf_prescription_tracks_h2(sfp)) {
            /* SFR = M_mol / t_dep, as sfr_fixing_gas_tdep(); SFprescription 8 on its own uses
             * DarkSage's law, M_H2 / 2.30 Gyr (H2DepletionMode 5) */
            const int mode = FG_TDEP(run_params) ? run_params->H2DepletionMode : 5;
            const double t_dep = h2_depletion_time_gyr(Sigma_gas, CODE_LENGTH_TO_PC(r_mid, h), v_circ, mode);
            if(t_dep > 0.0) sfr = (h2 / HYDROGEN_MASS_FRAC) / (t_dep * gyr_to_code);
        } else if(sfp == 0) {
            /* Croton+06: the bulk threshold m_crit = 0.19 Vvir reff grows linearly with r,
             * so the annulus share is 0.19 Vvir (r_out - r_in) */
            const double cold_crit = 0.19 * galaxies[p].Vvir * (r_out - r_in);
            if(gas > cold_crit) sfr = run_params->SfrEfficiency * (gas - cold_crit) / tdyn;
        } else if(sfp == 2 || sfp == 3) {
            const double x = Sigma_gas / DS_SOMERVILLE25_SIGMA_CRIT;
            const double eps_cl = x / (1.0 + x);
            sfr = eps_cl * DS_SOMERVILLE25_F_DENSE * ((sfp == 2) ? gas : h2) / tdyn;
        } else if(sfp == 6) {
            /* K13: SFR = M_gas / t_dep */
            double Z_prime = Z / DS_Z_SOLAR_ASPLUND09;
            if(Z_prime < 0.01) Z_prime = 0.01;
            const double t_dep = calculate_tdep_K13_Gyr(Sigma_gas, Sigma_star, rs_pc, Z_prime, f_H2);
            if(t_dep > 0.0) sfr = gas / (t_dep * gyr_to_code);
        } else {
            sfr = run_params->SfrEfficiency * h2 / tdyn;
        }

        sfr_bin[i] = (sfr > 0.0) ? sfr : 0.0;
        total_sfr += sfr_bin[i];
    }

    if(sf_prescription_tracks_h2(sfp)) {
        galaxies[p].H2gas = total_h2;
        if(galaxies[p].H2gas > galaxies[p].ColdGas * HYDROGEN_MASS_FRAC) {
            galaxies[p].H2gas = galaxies[p].ColdGas * HYDROGEN_MASS_FRAC;
        }
        /* Effective whole-disc depletion time, for the H2DepletionTime_Gyr output */
        galaxies[p].H2DepletionTime_Gyr = (total_sfr > 0.0 && galaxies[p].H2gas > 0.0)
            ? (float)((galaxies[p].H2gas / HYDROGEN_MASS_FRAC) / total_sfr / gyr_to_code) : -1.0f;
    }

    return total_sfr;
}

void darksage_remove_disc_gas(const int p, const double mass, const float weight[N_BINS],
                              struct GALAXY *galaxies)
{
    if(mass <= 0.0) return;

    double w_sum = 0.0;
    for(int i = 0; i < N_BINS; i++) w_sum += weight[i];
    if(w_sum <= 0.0) return;

    double removed = 0.0;
    for(int i = 0; i < N_BINS; i++) {
        if(galaxies[p].DiscGas[i] <= 0.0f) continue;
        double m_i = mass * weight[i] / w_sum;
        if(m_i > galaxies[p].DiscGas[i]) m_i = galaxies[p].DiscGas[i];
        const double Z = galaxies[p].DiscGasMetals[i] / galaxies[p].DiscGas[i];
        galaxies[p].DiscGas[i] -= m_i;
        galaxies[p].DiscGasMetals[i] -= Z * m_i;
        if(galaxies[p].DiscGas[i] < 0.0f) galaxies[p].DiscGas[i] = 0.0f;
        if(galaxies[p].DiscGasMetals[i] < 0.0f) galaxies[p].DiscGasMetals[i] = 0.0f;
        removed += m_i;
    }

    /* Annuli that ran short: take the rest from all annuli in proportion to their remaining gas,
     * so the annuli lose exactly what the bulk ColdGas lost */
    const double shortfall = mass - removed;
    if(shortfall > 1.0e-6 * mass && weight != galaxies[p].DiscGas) {
        darksage_remove_disc_gas(p, shortfall, galaxies[p].DiscGas, galaxies);
    } else if(shortfall > 1.0e-6 * mass) {
        /* Already gas-weighted: only possible if the annuli hold less than mass in total; empty them */
        for(int i = 0; i < N_BINS; i++) galaxies[p].DiscGas[i] = galaxies[p].DiscGasMetals[i] = 0.0f;
    }
}

/* The bulk removes metals at the galaxy-average metallicity, the annuli at their local one;
 * rescale the annulus metals to the bulk total, keeping the radial metallicity profile */
void darksage_sync_disc_metals(const int p, struct GALAXY *galaxies)
{
    double z_sum = 0.0, gas_sum = 0.0;
    for(int i = 0; i < N_BINS; i++) {
        z_sum += galaxies[p].DiscGasMetals[i];
        gas_sum += galaxies[p].DiscGas[i];
    }
    const double target = (galaxies[p].MetalsColdGas > 0.0f) ? galaxies[p].MetalsColdGas : 0.0;

    if(z_sum > 0.0) {
        const double f = target / z_sum;
        for(int i = 0; i < N_BINS; i++) galaxies[p].DiscGasMetals[i] = (float)(galaxies[p].DiscGasMetals[i] * f);
    } else if(gas_sum > 0.0) {
        for(int i = 0; i < N_BINS; i++) galaxies[p].DiscGasMetals[i] = (float)(target * galaxies[p].DiscGas[i] / gas_sum);
    }
}

void darksage_add_disc_metals(const int p, const double metals, const double sfr_bin[N_BINS],
                              struct GALAXY *galaxies)
{
    if(metals <= 0.0) return;

    double sfr_sum = 0.0;
    for(int i = 0; i < N_BINS; i++) sfr_sum += sfr_bin[i];
    if(sfr_sum <= 0.0) return;

    for(int i = 0; i < N_BINS; i++) {
        galaxies[p].DiscGasMetals[i] += metals * sfr_bin[i] / sfr_sum;
    }
}

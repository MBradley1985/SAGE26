/*
 * model_regimes.c -- CGM/hot-halo and feedback-free-burst regime classification.
 *
 * determine_and_store_regime() implements the Dekel & Birnboim (2006)
 * shock-mass criterion; determine_and_store_ffb_regime() implements the
 * Li+24 and BK25 feedback-free burst thresholds with optional lognormal
 * concentration scatter, plus the Dekel+23 free-fall-time/density criterion
 * (eqs. 3-5) evaluated directly from the galaxy's own CGM free-fall time.
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

/* -------------------------------------------------------------------------
 * File-scope empirical constants (lifted per STYLE_C.md SS8).
 * -------------------------------------------------------------------------*/

/* Dekel & Birnboim (2006) critical virial-shock stability mass: halos below it
 * lack a stable virial shock and are classified as CGM-regime.  Default
 * 6e11 Msun from DB06 Fig. 1 / eq. 4, now settable as MShockMsun in the
 * parameter file so it can be varied without a rebuild. */

/* Parsec in cm (IAU 2012).  Used when converting radii between Mpc/h
 * (code units) and pc for surface-density calculations. */
static const double PC_IN_CM              =  3.08568e18;

/* Boylan-Kolchin (2025) Table 1: critical gravitational acceleration for FFB.
 * Units: M_sun / pc^2 (pre-multiplication by G to get acceleration). */
static const double BK25_G_CRIT_MSUN_PC2 = 3100.0;

/* Dekel et al. (2023) sec. 2, below eq. 4: mean molecular weight adopted for
 * neutral atomic H+He gas at T < 10^4 K (mu = 1.4 adjusted to 1.2 for gas). */
static const double MU_NEUTRAL_FFB = 1.2;

/* Dekel et al. (2023) eq. 4 normalisation: 10^3.5 cm^-3, the density unit n_3.5
 * is expressed in (n_3.5 = n / N_3P5_CM3). */
static const double N_3P5_CM3 = 3162.2776601683795;  /* 10^3.5 */

/* --- Dekel+23 DISC scenario (their sec. 7), used by FeedbackFreeModeOn=9 ----
 * Disc HALF-thickness as a fraction of the disc radius, H_d/R_e.  D23's
 * fiducial 0.33 (their (H_d/R_d)_0.33 normalisation), from V/sigma ~ 3.
 *
 * It is a half-thickness, not a full height: their thick-disc conversion
 * factor (2/3)(R_d/H_d) is exactly [(4/3)pi R^3] / [pi R^2 * 2 H_d], so the
 * cylinder they integrate over has height 2*H_d.  Using 0.33*R as the full
 * height instead makes the volume 2x too small.  Verified against their own
 * fiducials: M_gen = 0.95e9 Msun in pi R_e^2 * 2 H_d gives n_d = 531 cm^-3
 * against eq. 63's 580, while the 0.33*R-as-full-height reading gives 1063.
 *
 * SAGE26 computes no gas-disc scale height of its own, so this stays a
 * constant rather than pretending to a value the model does not have. */
static const double DISC_HEIGHT_FACTOR = 0.33;

/* Dekel+23's disc radius is the HALF-MASS (effective) radius throughout, not
 * an exponential scale length: "lambda = R_e / R_v" (eq. 42 section) and
 * "with the definition R_d = lambda R_v for the half-mass radius" (eq. 53/54),
 * anchored numerically to "R_e ~ 0.3 kpc" and matched against CEERS sizes.
 * SAGE26's DiskScaleRadius is an exponential scale length, so converting
 * takes R_e = 1.68 R_d,exp (the half-mass radius of an exponential profile).
 * Omitting this makes the radius 1.68x too small and the density 4.74x too
 * high -- and because the error goes as R^-3 it falls hardest on the most
 * compact, lowest-spin discs, exactly where a spurious FFB selection hurts. */
static const double EFFECTIVE_RADIUS_FACTOR = 1.68;

/* Dekel+23 eq. 10, the 2D<->3D bridge that converts a clump's volume density
 * and radius into its surface density:
 *     Sigma / 10^3.5 Msun pc^-2  =  (n / 10^3.5 cm^-3) * (2 R / 15 pc)
 * Implemented in exactly that normalised form rather than re-deriving
 * Sigma = n mu m_p 2R from scratch: the two differ by ~2.2x because eq. 10
 * carries D23's own molecular mean molecular weight for the clump, and the
 * point of mode 9 is to follow the paper, not to re-choose its micro-physics. */
static const double SIGMA_3P5_MSUN_PC2 = 3162.2776601683795;  /* 10^3.5 */
static const double EQ10_CLUMP_DIAMETER_PC = 15.0;

/* Dekel+23 eq. 49: the Toomre fragmentation radius R_T = (pi/4) Q delta R_d. */
static const double TOOMRE_RADIUS_PREFACTOR = 0.7853981633974483;  /* pi/4 */

/* Dekel+23 eq. 54: the total mass the disc's self-gravity is measured against,
 * M_tot ~ 0.5 f_b M_v, so delta = M_d / M_tot (eq. 48). */
static const double DISC_MTOT_BARYON_FRACTION = 0.5;

/* Dekel+23 eq. 12 cooling threshold, at their adopted T_4 = 1 and C = 1. */
static const double N_COOL_NORM_CM3 = 3.4e3;
/* Solar metallicity as an absolute mass fraction, SAGE26's usual convention. */
static const double SOLAR_METALLICITY_ABS = 0.02;
/* D23 normalise metallicity to 0.02 SOLAR (their Z_0.02), not to solar. */
static const double D23_METALLICITY_NORM = 0.02;



/*
 * Classify each central galaxy as CGM-regime or hot-halo regime (Voit 2015).
 *
 * Uses the Dekel & Birnboim (2006) criterion: halos below ~6e11 M_sun are in
 * the CGM regime (Regime == 0); more massive halos are in the hot regime
 * (Regime == 1). Regime is stored on the galaxy struct and used by
 * cooling_recipe_regime_aware() and model_infall.c.
 */
void determine_and_store_regime(const int ngal, struct GALAXY *galaxies,
                                const struct params *run_params)
{
    for(int p = 0; p < ngal; p++) {
        if(galaxies[p].mergeType > 0) continue;

        // Convert Mvir to physical units (Msun)
        // Mvir is stored in units of 10^10 Msun/h
        const double Mshock = MSUN_TO_CODE_MASS(run_params->MShockMsun, run_params->Hubble_h);  // Msun

        // Calculate mass ratio for sigmoid
        const double mass_ratio = galaxies[p].Mvir / Mshock;

        int32_t new_regime;
        if(mass_ratio <= 0.0) {
            new_regime = 0;  // Default to CGM regime for invalid mass
        } else {
            // Smooth sigmoid transition (consistent with FFB approach)
            // Width of transition in dex
            const double delta_log_M = 0.1;

            // Sigmoid argument: x = log10(M/Mshock) / width
            const double x = log10(mass_ratio) / delta_log_M;

            // Sigmoid function: probability of being in Hot regime
            // Smoothly varies from 0 (well below Mshock) to 1 (well above Mshock)
            const double hot_fraction = 1.0 / (1.0 + exp(-x));

            // RegimeRandomMode=1: reuse the persistent RegimeRandom draw from
            // galaxy creation, so the regime evolves deterministically with
            // Mvir relative to a fixed per-galaxy quantile and never thrashes
            // for borderline-mass centrals.
            // RegimeRandomMode=0: fresh draw each snapshot (original behaviour).
            const double random_uniform = (run_params->RegimeRandomMode == 1)
                ? (double)galaxies[p].RegimeRandom
                : (double)rand() / (double)RAND_MAX;
            new_regime = (random_uniform < hot_fraction) ? 1 : 0;
        }

        galaxies[p].Regime = new_regime;
    }
}

/*
 * Inverse normal CDF (probit function) via Peter Acklam's rational approximation.
 *
 * Converts a uniform variate p in (0, 1) to a standard normal variate.
 * Accurate to ~1e-9 across the full range.
 */
static double inverse_normal_cdf(double p)
{
    const double a[] = {-3.969683028665376e+01,  2.209460984245205e+02,
                        -2.759285104469687e+02,  1.383577518672690e+02,
                        -3.066479806614716e+01,  2.506628277459239e+00};
    const double b[] = {-5.447609879822406e+01,  1.615858368580409e+02,
                        -1.556989798598866e+02,  6.680131188771972e+01,
                        -1.328068155288572e+01};
    const double c[] = {-7.784894002430293e-03, -3.223964580411365e-01,
                        -2.400758277161838e+00, -2.549732539343734e+00,
                         4.374664141464968e+00,  2.938163982698783e+00};
    const double d[] = { 7.784695709041462e-03,  3.224671290700398e-01,
                         2.445134137142996e+00,  3.754408661907416e+00};

    const double p_low  = 0.02425;
    const double p_high = 1.0 - p_low;

    double q, r;

    if(p < p_low) {
        q = sqrt(-2.0 * log(p));
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    } else if(p <= p_high) {
        q = p - 0.5;
        r = q * q;
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0);
    } else {
        q = sqrt(-2.0 * log(1.0 - p));
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                 ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    }
}

/*
 * Dekel+23 post-shock SHELL number density n_sh [cm^-3], their eqs. 38-41.
 *
 * Accreting gas is funnelled through a stream of radius
 * FFBStreamRadiusFraction * Rvir at Vvir, so mass conservation (eq. 39) sets
 * the pre-shock density rho_str = Mdot_ac / (pi R_str^2 V_v), which the shock
 * compresses by Mach^2 (eq. 38).  Uses NO ColdGas: this is an inflow-flux
 * density, which is where eq. 62's halo-mass dependence comes from.
 *
 * Mdot_ac is infall_recipe()'s own infallingGas over the snapshot's dt --
 * SAGE26's baryonic accretion including the reionization modifier -- not
 * D23's analytic eq. 31, which would reproduce eq. 62 by construction.
 *
 * Returns a negative value when it cannot be evaluated: a satellite (whose
 * stream belongs to the FoF central), no accretion this step, a shrinking
 * halo, or a degenerate Rvir/Vvir.
 */
static double ffb_shell_density(const int p, const double infallingGas, const double dt,
                                 const struct GALAXY *galaxies, const struct params *run_params)
{
    if(galaxies[p].Type != 0 || infallingGas <= 0.0 || dt <= 0.0 ||
       galaxies[p].Rvir <= 0.0 || galaxies[p].Vvir <= 0.0) {
        return -1.0;
    }

    /* Code mass/time -> g/s.  Hubble_h cancels in a rate (mass carries 1/h,
     * time carries 1/h), matching save_gals_hdf5.c's SfrDisk conversion. */
    const double mdot_g_per_s = (infallingGas / dt) * run_params->UnitMass_in_g
                                / run_params->UnitTime_in_s;
    const double Rvir_cm = galaxies[p].Rvir * run_params->UnitLength_in_cm
                           / run_params->Hubble_h;
    const double R_str_cm = run_params->FFBStreamRadiusFraction * Rvir_cm;
    const double v_cm_per_s = galaxies[p].Vvir * run_params->UnitVelocity_in_cm_per_s;
    const double cs_cm_per_s = run_params->FFBShellSoundSpeedKms * 1.0e5;

    const double rho_str = mdot_g_per_s / (M_PI * R_str_cm * R_str_cm * v_cm_per_s);
    const double mach = v_cm_per_s / cs_cm_per_s;

    return run_params->FFBCloudClumping * rho_str * mach * mach
           / (MU_NEUTRAL_FFB * PROTONMASS);
}

/*
 * Dekel+23 DISC gas density n_d [cm^-3] (eq. 63) and, through *sigma_c_out,
 * the star-forming clumps' surface density Sigma_c [Msun/pc^2] (eqs. 49, 66, 10).
 *
 * The density is built on M_gen, the PER-GENERATION disc mass (eq. 60), not
 * on the galaxy's standing cold-gas reservoir.  That distinction is the whole
 * criterion.  D23 sec. 7.2.3: gas accumulates until the disc goes Toomre
 * unstable (Q <= 0.67), bursts, "the disc gas is largely depleted during each
 * generation", and the cycle repeats.  M_gen is the gas mass at the moment
 * instability TRIGGERS -- so n_d(M_gen) asks "if this disc went unstable,
 * would the burst be feedback-free?".  Substituting the model's ColdGas asks
 * a different question, "how much gas is lying around right now", which over-
 * selects gas-rich compact discs at an arbitrary phase of the cycle.
 *
 * Only eq. 63 reproduces D23's own stated disc threshold: n_d > n_fbk gives
 * z > 7.03 from eq. 63 against their eq. 64's z > 7.04, whereas the overall
 * disc density of eq. 42 gives z > 2.68.
 *
 * M_gen follows from closing eq. 55 against delta = M_d/M_tot (eq. 48), which
 * puts M_d on both sides and leaves a cube root (this is why M_gen is NOT
 * simply Mdot_ac * t_d -- that product, eq. 57, is an input to eq. 55):
 *
 *     V_d   = sqrt(f_b / (2 lambda)) V_v                          (eq. 53)
 *     t_d   = R_e / V_d
 *     M_gen = [ (1/3) lambda f_b (Mdot_ac t_d) Q^-2 M_v^2 ]^(1/3) (eq. 60)
 *
 * evaluated at fiducials as 9.478e8 Msun against eq. 60's 0.95e9.
 *
 * Because M_gen needs Mdot_ac, this is centrals-only for the same reason the
 * shell is: infallingGas is the FoF group's budget, credited to the central.
 * The disc and shell paths therefore share their inflow term and differ only
 * in geometry, which is what D23 describe.
 *
 * Returns a negative value when it cannot be evaluated.
 */
static double ffb_disc_density(const int p, double *sigma_c_out,
                                const double infallingGas, const double dt,
                                const struct GALAXY *galaxies, const struct params *run_params)
{
    *sigma_c_out = -1.0;

    const double R_d_exp = galaxies[p].DiskScaleRadius;
    if(galaxies[p].Type != 0 || infallingGas <= 0.0 || dt <= 0.0 ||
       R_d_exp <= 0.0 || galaxies[p].Mvir <= 0.0 ||
       galaxies[p].Rvir <= 0.0 || galaxies[p].Vvir <= 0.0) {
        return -1.0;
    }

    const double f_b = run_params->BaryonFrac;
    const double Q = run_params->FFBToomreQ;
    const double c_disc = run_params->FFBCloudClumpingDisk;

    /* D23 work in the half-mass radius; SAGE26 stores an exponential scale
     * length.  lambda = R_e / R_v is then SAGE26's own spin-set contraction. */
    const double R_e = EFFECTIVE_RADIUS_FACTOR * R_d_exp;          /* code length */
    const double lambda = R_e / galaxies[p].Rvir;
    if(lambda <= 0.0) {
        return -1.0;
    }

    /* eq. 53, then t_d = R_e / V_d.  Code length / code velocity is code time,
     * since UnitTime_in_s = UnitLength_in_cm / UnitVelocity_in_cm_per_s. */
    const double V_d = sqrt(f_b / (2.0 * lambda)) * galaxies[p].Vvir;
    if(V_d <= 0.0) {
        return -1.0;
    }
    const double t_d = R_e / V_d;

    /* eq. 60, all in code mass. */
    const double mdot_ac = infallingGas / dt;
    const double M_gen = cbrt((1.0 / 3.0) * lambda * f_b * (mdot_ac * t_d)
                              / (Q * Q) * galaxies[p].Mvir * galaxies[p].Mvir);
    if(M_gen <= 0.0) {
        return -1.0;
    }

    const double mass_g = M_gen * run_params->UnitMass_in_g / run_params->Hubble_h;
    const double radius_cm = R_e * run_params->UnitLength_in_cm / run_params->Hubble_h;
    const double half_thickness_cm = DISC_HEIGHT_FACTOR * radius_cm;
    const double volume_cm3 = M_PI * radius_cm * radius_cm * 2.0 * half_thickness_cm;
    const double n_d = c_disc * mass_g / volume_cm3 / (MU_NEUTRAL_FFB * PROTONMASS);
    if(n_d <= 0.0) {
        return -1.0;
    }

    /* delta at the instability trigger (eq. 48 with M_d = M_gen, eq. 54's
     * M_tot), then the Toomre radius (eq. 49), the clump c times smaller
     * (eq. 66), and eq. 10 to turn (n_d, R_c) into Sigma_c.  The SAME c
     * scales n_d and shrinks R_c -- in D23 these are one quantity. */
    const double M_tot = DISC_MTOT_BARYON_FRACTION * f_b * galaxies[p].Mvir;
    double delta = (M_tot > 0.0) ? M_gen / M_tot : 0.0;
    if(delta > 1.0) delta = 1.0;

    const double R_T_cm = TOOMRE_RADIUS_PREFACTOR * Q * delta * radius_cm;
    const double R_c_pc = (R_T_cm / c_disc) / PC_IN_CM;

    *sigma_c_out = SIGMA_3P5_MSUN_PC2 * (n_d / N_3P5_CM3)
                   * (2.0 * R_c_pc / EQ10_CLUMP_DIAMETER_PC);
    return n_d;
}

/*
 * Dekel+23 eq. 12 cooling density threshold n_cool [cm^-3]:
 *
 *     n_cool ~ 3.4e3 cm^-3 * Z_0.02^-2 * T_4^2 * C^-2
 *
 * Note the exponent on metallicity is -2, not -1: eq. 11's t_cool ~ Z^-1 set
 * equal to t_ff ~ n^-1/2 gives n ~ Z^-2.  Z_0.02 normalises to 0.02 SOLAR,
 * i.e. a very metal-poor starburst.  C = <n^2>/<n>^2 is the clumping factor
 * WITHIN the cloud and is explicitly NOT the density contrast c of eqs. 41/42
 * ("not to be confused with the clumping factor C in equation 11"); D23 adopt
 * C ~ 1 and T_4 ~ 1, which is what is hard-coded here.
 *
 * Always applied.  Note D23 do not propagate cooling into their eqs. 62/64/67
 * thresholds, and it is largely self-cancelling in practice: "for Z somewhat
 * higher than 0.02, this estimate of n_cool becomes lower than n_fbk, such
 * that the cooling cannot serve as a threshold for starburst above n_fbk."
 * SAGE26 tracks MetalsColdGas, so Z is per-galaxy and that regime is reached
 * routinely -- measured on millennium, requiring n > n_cool removes only ~3%
 * of mode-9 selections.
 *
 * Returns a negative value when metallicity is unavailable (no cold gas, or
 * no metals -- pristine gas cools freely, so the threshold does not bind).
 */
static double ffb_cooling_density(const int p, const struct GALAXY *galaxies)
{
    if(galaxies[p].ColdGas <= 0.0 || galaxies[p].MetalsColdGas <= 0.0) {
        return -1.0;
    }
    /* Absolute metal mass fraction -> D23's Z_0.02 (0.02 in solar units, and
     * solar is itself ~0.02 by mass in SAGE26's convention). */
    const double Z_abs = galaxies[p].MetalsColdGas / galaxies[p].ColdGas;
    const double Z_0p02 = Z_abs / (SOLAR_METALLICITY_ABS * D23_METALLICITY_NORM);
    if(Z_0p02 <= 0.0) {
        return -1.0;
    }
    return N_COOL_NORM_CM3 / (Z_0p02 * Z_0p02);
}

/*
 * Dekel+23's basic FFB condition on a density n: the free-fall time must beat
 * the feedback delay, t_ff = 0.84 Myr (n/10^3.5 cm^-3)^-1/2 < t_fbk (eqs. 3-4),
 * equivalently n > n_fbk ~ 2.23e3 cm^-3 (eq. 5).
 *
 * n_fbk is the same threshold whichever geometry supplied the density: it is
 * set by free-fall beating feedback, a property of the collapsing gas, not of
 * the shell or the disc.
 *
 * The gas must additionally be able to cool within that time, n > n_cool
 * (eq. 12) -- see ffb_cooling_density().
 */
static int ffb_free_fall_ok(const double n, const int p, const struct GALAXY *galaxies,
                             const struct params *run_params)
{
    if(n <= 0.0) {
        return 0;
    }
    const double tff_Myr = 0.84 * sqrt(N_3P5_CM3 / n);
    if(tff_Myr >= run_params->FFBFeedbackDelayMyr) {
        return 0;
    }
    const double n_cool = ffb_cooling_density(p, galaxies);
    if(n_cool > 0.0 && n < n_cool) {
        return 0;
    }
    return 1;
}

/*
 * Classify galaxies as feedback-free burst (FFB) or normal mode.
 *
 * When FeedbackFreeModeOn > 0, evaluates each central galaxy against the FFB
 * mass and redshift criteria (Li+2024), the BK25 acceleration criterion, or
 * the Dekel+23 free-fall-time criterion (modes 8, 9, 10), and sets
 * galaxies[p].FFBRegime.  Uses a lognormal scatter (via inverse_normal_cdf)
 * around the threshold when the scatter mode is enabled.  Skips all galaxies
 * when FFBmodeOn == 0.
 */
void determine_and_store_ffb_regime(const int ngal, const double Zcurr,
                                     const double infallingGas, const double dt,
                                     struct GALAXY *galaxies,
                                     const struct params *run_params)
{
    // Only apply FFB if the mode is enabled
    if(run_params->FeedbackFreeModeOn == 0) {
        // FFB mode disabled - mark all galaxies as normal
        for(int p = 0; p < ngal; p++) {
            galaxies[p].FFBRegime = 0;
        }
        return;
    }

    // Pre-compute g_crit in code units for BK25 modes (constant, doesn't depend on galaxy)
    // g_crit/G = 3100 M_sun/pc^2 (Boylan-Kolchin 2025, Table 1)
    double g_crit = 0.0;
    if(run_params->FeedbackFreeModeOn == 2 || run_params->FeedbackFreeModeOn == 3 ||
       run_params->FeedbackFreeModeOn == 4 || run_params->FeedbackFreeModeOn == 7) {
        const double Msun_code = SOLAR_MASS / run_params->UnitMass_in_g;
        const double pc_code = PC_IN_CM / run_params->UnitLength_in_cm;
        g_crit = run_params->G * BK25_G_CRIT_MSUN_PC2 * Msun_code / (pc_code * pc_code) / run_params->Hubble_h;
    }

    // Classify each galaxy as FFB or normal
    for(int p = 0; p < ngal; p++) {
        if(galaxies[p].mergeType > 0) continue;

        // By default, only CGM-regime halos are eligible for FFB.
        // FFBIgnoreRegime=1 removes this restriction, letting the Li+24/BK25
        // criteria apply regardless of halo regime.
        if(galaxies[p].Regime == 1 && !run_params->FFBIgnoreRegime) {
            galaxies[p].FFBRegime = 0;
            continue;
        }

        // FFBRandomMode=1: reuse the persistent draw assigned at galaxy creation.
        // FFBRandomMode=0: fresh draw each snapshot (no memory across timesteps).
        const double draw = (run_params->FFBRandomMode == 1)
            ? (double)galaxies[p].FFBRandom
            : (double)rand() / (double)RAND_MAX;

        if(run_params->FeedbackFreeModeOn == 1) {
            // Li et al. 2024 mass-based method (original)
            const double Mvir = galaxies[p].Mvir;

            // Calculate smooth FFB fraction using sigmoid transition (Li et al. 2024, eq. 3)
            const double f_ffb = calculate_ffb_fraction(Mvir, Zcurr, run_params);

            const double random_uniform = draw;

            if(random_uniform < f_ffb) {
                galaxies[p].FFBRegime = 1;  // FFB halo
            } else {
                galaxies[p].FFBRegime = 0;  // Normal halo
            }
        } else if(run_params->FeedbackFreeModeOn == 2) {
            // Boylan-Kolchin 2025 acceleration-based method (Ishiyama+21 lookup table concentration)
            // FFB regime when g_max > g_crit (sharp cutoff)
            const double g_max = calculate_gmax_BK25(p, Zcurr, galaxies, run_params);

            galaxies[p].g_max = g_max;

            if(g_max > g_crit) {
                galaxies[p].FFBRegime = 1;  // FFB halo - above critical acceleration
            } else {
                galaxies[p].FFBRegime = 0;  // Normal halo
            }
        } else if(run_params->FeedbackFreeModeOn == 3) {
            // BK25 acceleration-based method using galaxy's stored concentration
            // (Vmax/Vvir with infall freeze when ConcentrationOn=3)
            const double Mvir = galaxies[p].Mvir;
            const double Rvir = galaxies[p].Rvir;

            if(Mvir <= 0.0 || Rvir <= 0.0) {
                galaxies[p].FFBRegime = 0;
                galaxies[p].g_max = 0.0;
                continue;
            }

            double c = (double)galaxies[p].Concentration;
            if(c < 1.0) c = 1.0;

            const double g_vir = run_params->G * Mvir / (Rvir * Rvir);
            const double mu_c = log(1.0 + c) - c / (1.0 + c);
            const double g_max = (g_vir / mu_c) * (c * c / 2.0);

            galaxies[p].g_max = g_max;

            if(g_max > g_crit) {
                galaxies[p].FFBRegime = 1;  // FFB halo - above critical acceleration
            } else {
                galaxies[p].FFBRegime = 0;  // Normal halo
            }
        } else if(run_params->FeedbackFreeModeOn == 4) {
            // BK25 acceleration-based with log-normal concentration scatter.
            // The Ishiyama+21 table gives the mean concentration; individual halos
            // scatter around it following p(c)dc ~ exp(-(ln c - ln c0)^2 / 2sigma_c^2) d(ln c)
            // with sigma_c ~ 0.2 (Jing 2000; Bullock+01; Dolag+04).
            // The persistent FFBRandom draws a fixed quantile for each halo,
            // giving a deterministic scattered concentration and thus a smooth
            // FFb transition across the halo population.
            const double Mvir = galaxies[p].Mvir;
            const double Rvir = galaxies[p].Rvir;

            if(Mvir <= 0.0 || Rvir <= 0.0) {
                galaxies[p].FFBRegime = 0;
                galaxies[p].g_max = 0.0;
                continue;
            }

            // Mean concentration from Ishiyama+21 lookup table
            const double Mvir_Msun_h = Mvir * 1.0e10;
            const double logM = log10(Mvir_Msun_h);
            double c = interpolate_concentration_ishiyama21(logM, Zcurr, run_params);
            if(c < 1.0) c = 1.0;

            // Apply log-normal scatter: ln(c) ~ Normal(ln(c_mean), sigma_c)
            if(run_params->FFBConcSigma > 0.0) {
                double u = draw;
                if(u < 1.0e-6) u = 1.0e-6;
                if(u > 1.0 - 1.0e-6) u = 1.0 - 1.0e-6;
                const double z_normal = inverse_normal_cdf(u);
                c = c * exp(run_params->FFBConcSigma * z_normal);
                if(c < 1.0) c = 1.0;
            }

            // g_max with scattered concentration (BK25 Eq. 4)
            const double g_vir = run_params->G * Mvir / (Rvir * Rvir);
            const double mu_c = log(1.0 + c) - c / (1.0 + c);
            const double g_max = (g_vir / mu_c) * (c * c / 2.0);

            galaxies[p].g_max = g_max;

            if(g_max > g_crit) {
                galaxies[p].FFBRegime = 1;  // FFB halo
            } else {
                galaxies[p].FFBRegime = 0;  // Normal halo
            }
        } else if(run_params->FeedbackFreeModeOn == 5) {
            // Li et al. 2024 mass-based method with hard cutoff (no sigmoid)
            // FFB regime when Mvir > Mvir_ffb (sharp threshold)
            const double Mvir = galaxies[p].Mvir;
            const double Mvir_ffb = calculate_ffb_threshold_mass(Zcurr, run_params);

            if(Mvir > Mvir_ffb) {
                galaxies[p].FFBRegime = 1;  // FFB halo - above threshold mass
            } else {
                galaxies[p].FFBRegime = 0;  // Normal halo
            }
        } else if(run_params->FeedbackFreeModeOn == 6) {
            // Li+24 sigmoid + H2-based SF (same regime detection as mode 1)
            const double Mvir = galaxies[p].Mvir;
            const double f_ffb = calculate_ffb_fraction(Mvir, Zcurr, run_params);
            const double random_uniform = draw;

            if(random_uniform < f_ffb) {
                galaxies[p].FFBRegime = 1;  // FFB halo
            } else {
                galaxies[p].FFBRegime = 0;  // Normal halo
            }
        } else if(run_params->FeedbackFreeModeOn == 7) {
            // BK25 log-normal c scatter + H2-based SF (same regime detection as mode 4)
            const double Mvir = galaxies[p].Mvir;
            const double Rvir = galaxies[p].Rvir;

            if(Mvir <= 0.0 || Rvir <= 0.0) {
                galaxies[p].FFBRegime = 0;
                galaxies[p].g_max = 0.0;
                continue;
            }

            const double Mvir_Msun_h = Mvir * 1.0e10;
            const double logM = log10(Mvir_Msun_h);
            double c = interpolate_concentration_ishiyama21(logM, Zcurr, run_params);
            if(c < 1.0) c = 1.0;

            if(run_params->FFBConcSigma > 0.0) {
                double u = draw;
                if(u < 1.0e-6) u = 1.0e-6;
                if(u > 1.0 - 1.0e-6) u = 1.0 - 1.0e-6;
                const double z_normal = inverse_normal_cdf(u);
                c = c * exp(run_params->FFBConcSigma * z_normal);
                if(c < 1.0) c = 1.0;
            }

            const double g_vir = run_params->G * Mvir / (Rvir * Rvir);
            const double mu_c = log(1.0 + c) - c / (1.0 + c);
            const double g_max = (g_vir / mu_c) * (c * c / 2.0);

            galaxies[p].g_max = g_max;

            if(g_max > g_crit) {
                galaxies[p].FFBRegime = 1;  // FFB halo
            } else {
                galaxies[p].FFBRegime = 0;  // Normal halo
            }
        } else if(run_params->FeedbackFreeModeOn == 8) {
            // Dekel+23 SHELL scenario: FFB where the post-shock shell's
            // free-fall time beats the feedback delay (eqs. 3-5 applied to
            // n_sh, eqs. 38-41).  This is the quantity eq. 62 is built on --
            // D23 sec. 8.2 is explicit that a disc density is not equivalent,
            // its eq. 63 carrying only M_v^0.05.  See ffb_shell_density().
            const double n_sh = ffb_shell_density(p, infallingGas, dt, galaxies, run_params);
            if(n_sh <= 0.0) {
                galaxies[p].FFBRegime = 0;
                continue;
            }
            galaxies[p].FFBRegime = ffb_free_fall_ok(n_sh, p, galaxies, run_params) ? 1 : 0;

        } else if(run_params->FeedbackFreeModeOn == 9) {
            // Dekel+23 DISC scenario (sec. 7): BOTH of their disc conditions,
            // which D23 apply jointly rather than as alternatives --
            //   (i)  t_ff(n_d) < t_fbk, their eq. 64.  n_d carries no halo-mass
            //        dependence (eq. 42), so on its own this is a pure REDSHIFT
            //        threshold -- a vertical line in (z, Mvir).  It is also what
            //        produces their "FFB is excluded in the case of a disc" at
            //        z < 8, NOT the surface-density condition.
            //   (ii) Sigma_c > Sigma_crit, their eq. 67.  Separate physics:
            //        radiation pressure failing to unbind the clump (eq. 9,
            //        Fall+10; Grudic+18), not a free-fall timescale.  The
            //        halo-mass dependence enters only here, via R_c ~ M_v^0.38.
            //
            // CAVEAT worth knowing before trusting this mode: eq. 42 scales as
            // lambda^-3 and eq. 64's redshift floor as lambda^0.65, where
            // lambda is the disc-to-virial radius ratio.  D23 use a single
            // fiducial lambda = 0.025 and never discuss scatter in it; SAGE26
            // has a full spin distribution with a long low tail and no floor in
            // get_disk_radius(), so this criterion preferentially selects
            // low-spin haloes at redshifts where D23 would exclude them.
            double sigma_c;
            const double n_d = ffb_disc_density(p, &sigma_c, infallingGas, dt,
                                                galaxies, run_params);
            if(n_d <= 0.0) {
                galaxies[p].FFBRegime = 0;
                continue;
            }
            const int free_fall_ok = ffb_free_fall_ok(n_d, p, galaxies, run_params);
            const int surface_ok = (sigma_c > run_params->FFBSigmaCritMsunPc2);
            galaxies[p].FFBRegime = (free_fall_ok && surface_ok) ? 1 : 0;

        } else if(run_params->FeedbackFreeModeOn == 10) {
            // Dekel+23 COMBINED criterion: their own "robust" pairing of the
            // two geometries -- "equations (62) and (67) provide a robust
            // condition for FFB in the mass-redshift plane".
            //
            //   eq. 62  the SHELL 3D condition, n_sh > n_fbk.  D23 note the
            //           shell's own surface-density condition adds nothing:
            //           it "turns out to coincide with the condition implied
            //           by the basic requirement on the 3D density".
            //   eq. 67  the DISC surface-density condition, Sigma_c > Sigma_crit.
            //
            // Both are necessary, so this is strictly more restrictive than
            // either mode 8 or mode 9.  Note D23 present shell and disc as
            // alternative GEOMETRIES chosen by the stream's angular momentum,
            // not as two tests every galaxy passes; requiring both is the
            // conservative reading, and it is the pairing they themselves call
            // robust.
            const double n_sh = ffb_shell_density(p, infallingGas, dt, galaxies, run_params);
            double sigma_c;
            const double n_d = ffb_disc_density(p, &sigma_c, infallingGas, dt,
                                                galaxies, run_params);
            if(n_sh <= 0.0 || n_d <= 0.0) {
                galaxies[p].FFBRegime = 0;
                continue;
            }
            const int shell_ok = ffb_free_fall_ok(n_sh, p, galaxies, run_params);
            const int surface_ok = (sigma_c > run_params->FFBSigmaCritMsunPc2);
            galaxies[p].FFBRegime = (shell_ok && surface_ok) ? 1 : 0;

        } else if(run_params->FeedbackFreeModeOn == 11) {
            // Dekel+23 EITHER geometry: FFB if the galaxy qualifies as a shell
            // OR as a disc.  This is the union of modes 8 and 9, where mode 10
            // is their intersection.
            //
            // D23 present shell and disc as two limiting geometries selected by
            // the angular momentum of the feeding streams -- "the two scenarios
            // of shells and of discs" -- not as two tests a galaxy must pass
            // simultaneously.  A real galaxy is somewhere between them and the
            // paper gives no formula for which applies, so the honest per-galaxy
            // reading is that clearing EITHER is sufficient: the gas got dense
            // enough to burst before feedback, by one route or the other.
            //
            // The shell arm is the eq. 62 condition alone, because D23 note the
            // shell's own surface-density requirement "turns out to coincide
            // with the condition implied by the basic requirement on the 3D
            // density".  The disc arm needs both of its conditions (eqs. 64 and
            // 67), since for a disc the surface-density criterion is genuinely
            // independent and is what carries the halo-mass dependence.
            const double n_sh = ffb_shell_density(p, infallingGas, dt, galaxies, run_params);
            const int shell_ok = (n_sh > 0.0)
                                 && ffb_free_fall_ok(n_sh, p, galaxies, run_params);

            double sigma_c;
            const double n_d = ffb_disc_density(p, &sigma_c, infallingGas, dt,
                                                galaxies, run_params);
            const int disc_ok = (n_d > 0.0)
                                && ffb_free_fall_ok(n_d, p, galaxies, run_params)
                                && (sigma_c > run_params->FFBSigmaCritMsunPc2);

            galaxies[p].FFBRegime = (shell_ok || disc_ok) ? 1 : 0;
        }
    }
}

/*
 * FFB virial mass threshold at redshift z (Li+2024 eq. 2).
 *
 * M_v,ffb / 10^10.8 Msun ~ ((1+z)/10)^-6.2.  Returns the threshold in
 * code units (10^10 Msun/h).
 */
double calculate_ffb_threshold_mass(const double z, const struct params *run_params)
{
    // Equation (2) from Li et al. 2024
    // M_v,ffb / 10^10.8 M_sun ~ ((1+z)/10)^-6.2
    //
    // In code units (10^10 M_sun/h):
    // log(M_code) = log(M_sun) - 10 + log(h)
    //             = 10.8 - 6.2*log((1+z)/10) - 10 + log(h)
    //             = 0.8 + log(h) - 6.2*log((1+z)/10)

    const double h = run_params->Hubble_h;
    const double z_norm = (1.0 + z) / 10.0;
    /* FFBThresholdSlope defaults to -6.2 (Li+24). The 10^10.8 normalisation is
     * pinned at z = 9, where z_norm = 1, so changing the slope pivots the
     * threshold about that redshift rather than shifting it wholesale. */
    const double log_Mvir_ffb_code = 0.8 + log10(h) + run_params->FFBThresholdSlope * log10(z_norm);

    return pow(10.0, log_Mvir_ffb_code);
}

/*
 * Fraction of galaxies in the FFB regime at (Mvir, z) via Li+2024 eq. (3).
 *
 * Returns a sigmoid value in [0, 1] that rises sharply as Mvir approaches
 * the FFB threshold; returns 0 when FeedbackFreeModeOn == 0.
 */
double calculate_ffb_fraction(const double Mvir, const double z, const struct params *run_params)
{
    // Calculate the fraction of galaxies in FFB regime
    // Uses smooth sigmoid transition from Li et al. 2024, equation (3)
    
    if (run_params->FeedbackFreeModeOn == 0) {
        return 0.0;
    }

    const double Mvir_ffb = calculate_ffb_threshold_mass(z, run_params);

    if(Mvir <= 0.0 || Mvir_ffb <= 0.0) {
        return 0.0;
    }

    /* Li+2024 eq. (3): sigmoid over 0.15 dex around M_v,ffb. */
    const double delta_log_M = 0.15;
    const double x = log10(Mvir / Mvir_ffb) / delta_log_M;
    const double f_ffb = 1.0 / (1.0 + exp(-x));

    return f_ffb;
}

/*
 * Maximum NFW gravitational acceleration g_max (Boylan-Kolchin 2025).
 *
 * Computes g_vir = G*M_vir/R_vir^2 and then the NFW peak factor from the
 * halo concentration, returning g_max in CGS units (cm/s^2).  Used as the
 * FFB feedback threshold in the FeedbackFreeModeOn == 4 prescription.
 */
double calculate_gmax_BK25(const int p, const double z, const struct GALAXY *galaxies,
                            const struct params *run_params)
{
    // Boylan-Kolchin 2025: maximum NFW gravitational acceleration
    //
    // g_vir = G * M_vir / R_vir^2                                (Eq. 2)
    // g_max = (g_vir / mu(c)) * (c^2 / 2)                         (Eq. 4)
    // where mu(x) = ln(1+x) - x/(1+x)
    //
    // Always uses the Ishiyama+21 lookup table concentration for the FFB
    // threshold, even when ConcentrationOn=2 (Vmax/Vvir).  The BK25 threshold
    // is derived from average halo properties; using individual scatter would
    // produce spurious FFB activation at low redshift.
    //
    // Returns g_max in code units (UnitLength / UnitTime^2)

    const double Mvir = galaxies[p].Mvir;  // code mass units (10^10 M_sun / h)
    const double Rvir = galaxies[p].Rvir;  // code length units (Mpc / h)

    if(Mvir <= 0.0 || Rvir <= 0.0) {
        return 0.0;
    }

    // g_vir = G * M_vir / R_vir^2  (code units)
    const double g_vir = run_params->G * Mvir / (Rvir * Rvir);

    // Always use the lookup table concentration for the FFB determination
    const double Mvir_Msun_h = Mvir * 1.0e10;
    const double logM = log10(Mvir_Msun_h);
    double c = interpolate_concentration_ishiyama21(logM, z, run_params);
    if(c < 1.0) c = 1.0;

    // mu(c) = ln(1+c) - c/(1+c)
    const double mu_c = log(1.0 + c) - c / (1.0 + c);

    // g_max = (g_vir / mu(c)) * (c^2 / 2)   [BK25 Eq. 4]
    return (g_vir / mu_c) * (c * c / 2.0);
}

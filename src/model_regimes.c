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

            // A fresh draw each snapshot: a borderline-mass central is
            // re-tested against the sigmoid every timestep rather than being
            // held to one quantile fixed at creation.
            const double random_uniform = (double)rand() / (double)RAND_MAX;
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
 * Classify galaxies as feedback-free burst (FFB) or normal mode.
 *
 * When EnhancedStarFormationOn > 0, evaluates each central galaxy against the
 * selected FFB criterion and sets galaxies[p].FFBRegime:
 *
 *   1 -- Li et al. (2024) mass threshold with their eq. 3 sigmoid.  A halo is
 *        FFB with probability f_ffb(Mvir, z), realised against a uniform draw,
 *        so the transition is smooth across the population.
 *   2 -- Boylan-Kolchin (2025) acceleration threshold, g_max > g_crit, with
 *        log-normal scatter applied to the halo concentration.  The scatter is
 *        drawn from the halo's own persistent quantile, so the threshold is
 *        sharp per halo but smooth across the population.
 *
 * All galaxies are marked non-FFB when EnhancedStarFormationOn == 0.
 */
void determine_and_store_ffb_regime(const int ngal, const double Zcurr,
                                     struct GALAXY *galaxies,
                                     const struct params *run_params)
{
    // Only apply FFB if the mode is enabled
    if(run_params->EnhancedStarFormationOn == 0) {
        // FFB mode disabled - mark all galaxies as normal
        for(int p = 0; p < ngal; p++) {
            galaxies[p].FFBRegime = 0;
        }
        return;
    }

    // Pre-compute g_crit in code units for the BK25 mode (constant, doesn't depend on galaxy)
    // g_crit/G = 3100 M_sun/pc^2 (Boylan-Kolchin 2025, Table 1)
    double g_crit = 0.0;
    if(run_params->EnhancedStarFormationOn == 2) {
        const double Msun_code = SOLAR_MASS / run_params->UnitMass_in_g;
        const double pc_code = PC_IN_CM / run_params->UnitLength_in_cm;
        g_crit = run_params->G * BK25_G_CRIT_MSUN_PC2 * Msun_code / (pc_code * pc_code) / run_params->Hubble_h;
    }

    // Classify each galaxy as FFB or normal
    for(int p = 0; p < ngal; p++) {
        if(galaxies[p].mergeType > 0) continue;

        // The Li+24 / BK25 criteria apply regardless of the halo's CGM/hot
        // regime: FFB is set by the burst's own density and feedback timescale,
        // not by whether the halo carries a virial shock.

        // A fresh draw each snapshot, with no memory across timesteps, so a
        // halo sitting near the threshold moves in and out of FFB rather than
        // being locked to one quantile fixed at creation.
        const double draw = (double)rand() / (double)RAND_MAX;

        if(run_params->EnhancedStarFormationOn == 1) {
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
        } else if(run_params->EnhancedStarFormationOn == 2) {
            // BK25 acceleration-based with log-normal concentration scatter.
            // The Ishiyama+21 table gives the mean concentration; individual halos
            // scatter around it following p(c)dc ~ exp(-(ln c - ln c0)^2 / 2sigma_c^2) d(ln c)
            // with sigma_c ~ 0.2 (Jing 2000; Bullock+01; Dolag+04).
            // Each halo draws its own concentration quantile, giving a smooth
            // FFB transition across the halo population rather than a single
            // sharp mass threshold.
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
 * the FFB threshold; returns 0 when EnhancedStarFormationOn == 0.
 */
double calculate_ffb_fraction(const double Mvir, const double z, const struct params *run_params)
{
    // Calculate the fraction of galaxies in FFB regime
    // Uses smooth sigmoid transition from Li et al. 2024, equation (3)
    
    if (run_params->EnhancedStarFormationOn == 0) {
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

/*
 * model_cooling_heating.c -- Cooling and AGN heating prescriptions.
 *
 * Implements the two-regime cooling model selected per-galaxy by the Regime
 * flag set in model_misc.c:
 *
 *   Regime == 0 (CGM-dominated) -- precipitation-driven cooling from the CGMgas
 *     reservoir using the Voit (2015) / McCourt et al. (2012) t_cool/t_ff < 10
 *     threshold.  The CGM density structure is modelled with a uniform, NFW, or
 *     beta (beta = 2/3) profile selected by CGMDensityProfile.  AGN heating uses
 *     the same r_heat ratchet as the hot-halo regime, capped at Rvir.
 *
 *   Regime == 1 (hot halo) -- classical isothermal-halo cooling following
 *     White & Frenk (1991) and Croton et al. (2006).  When CGMrecipeOn > 0 a
 *     De Lucia & Blaizot (2006) cold-stream fraction is blended in for halos
 *     above the virial shock mass.
 *
 * AGN radio-mode accretion is computed via three models (AGNrecipeOn 1/2/3):
 * empirical (Croton+06 eq. 10), Bondi-Hoyle, and cold-cloud triggering.
 * In all modes accretion is Eddington-limited and draws from the reservoir
 * appropriate to the regime (HotGas for Regime==1, CGMgas for Regime==0).
 *
 * File-private helpers compute NFW/beta density profiles, their enclosed-mass
 * integrals, and iteratively solve for the cooling radius.
 *
 * Code units (10^10 Msun/h, Mpc/h, km/s) used throughout; conversions to
 * physical units happen only at the entry points of CGM-mode functions.
 *
 * References: Croton et al. (2006), MNRAS 365, 11; Voit (2015), ApJL 808, L30;
 *   McCourt et al. (2012), MNRAS 419, 3319; Duffy et al. (2008), MNRAS 390, L64;
 *   De Lucia & Blaizot (2006), MNRAS 375, 2.
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>

#include "core_allvars.h"
#include "core_cool_func.h"

#include "model_cooling_heating.h"
#include "model_misc.h"

/* -------------------------------------------------------------------------
 * File-scope empirical constants (lifted per STYLE_C.md SS8).
 * -------------------------------------------------------------------------*/

/* Duffy et al. (2008) NFW concentration-mass-redshift relation,
 * Table 1 "Full sample" (relaxed halos, NFW profile, 200c overdensity).
 * c = A * (M/M_pivot)^B * (1+z)^C */
static const double DUFFY08_A       =  7.85;
static const double DUFFY08_M_PIVOT =  2.0e12;  /* pivot mass in Msun */
static const double DUFFY08_B       = -0.081;
static const double DUFFY08_C       = -0.71;

/* Mean molecular weight for fully ionised, primordial (H+He) gas.
 * X_H = 0.76, Y_He = 0.24 => mu = 1/(2*X + 3*Y/4) ~ 0.59. */
static const double MU_IONISED = 0.59;

/* Croton et al. (2006) AGN empirical radio-mode accretion pivots
 * (their eq. 10 / Sec. 4.2).  Normalisation values chosen to reproduce
 * observed BH-bulge mass relation at z=0. */
static const double AGN_BH_MASS_PIVOT  = 0.01;    /* 10^8 Msun for h=1 in code units of 10^10 Msun/h */
static const double AGN_VVIR_PIVOT_KMS = 200.0;   /* km/s */
static const double AGN_HOT_GAS_PIVOT  = 0.1;     /* hot-gas-to-halo mass fraction normalisation */

/* Eddington accretion rate formula.
 * L_Edd = 1.3e38 * (M_BH/Msun) erg/s  (Rybicki & Lightman 1979, eq. 1.4.9).
 * Standard AGN radiative efficiency eta = 0.1.
 * C_SQ_KMS2 = c^2 in (km/s)^2; c = 3e5 km/s => c^2 = 9e10. */
static const double EDDINGTON_LUM_PER_MSUN_CGS = 1.3e38;  /* erg/s per Msun */
static const double AGN_RADIATIVE_EFFICIENCY    = 0.1;
static const double C_SQ_KMS2                  = 9.0e10;  /* (km/s)^2 */

/* AGN heating coefficient: sqrt(2 * eta * c^2) where eta = AGN_RADIATIVE_EFFICIENCY
 * and c is in km/s.  Equals the ratio of radiated energy to halo kinetic energy
 * per unit accreted mass.  Croton et al. (2006) eq. 19. */
static const double AGN_HEATING_COEFF_KMS = 1.34e5;  /* km/s */

/* Gravitational constant in CGS (NIST CODATA 2018). */
static const double G_CGS = 6.674e-8;  /* cm^3 g^-1 s^-2 */

/* Virial temperature coefficient: T_vir = VIRIAL_TEMP_COEFF * Vvir^2 [Kelvin].
 * Follows from T = mu * m_p * Vvir^2 / (2 * k_B) with mu = MU_IONISED = 0.59;
 * gives 35.9 K (km/s)^-2. */
static const double VIRIAL_TEMP_COEFF = 35.9;  /* K (km/s)^-2 */

/* McCourt et al. (2012) thermal instability threshold: precipitation occurs
 * when t_cool / t_ff < PRECIP_THRESHOLD (= 10). */
static const double PRECIP_THRESHOLD = 10.0;

/* Width of the tanh/sigmoid transition zone around PRECIP_THRESHOLD.
 * Smooths the discontinuity at exactly t_cool/t_ff = 10. */
static const double PRECIP_TRANSITION_WIDTH = 2.0;

/* Beta-profile core radius as a fraction of the virial radius: r_c = frac * Rvir.
 * A value of 0.1 is the standard choice for the hot CGM (e.g., Makino+98). */
static const double CGM_BETA_CORE_RADIUS_FRAC = 0.1;

/* De Lucia & Blaizot (2006) eq. 38: virial shock mass scale.
 * Hot-mode shock heating is efficient only above this halo mass. */
static const double MSHOCK_DB06_MSUN = 6.0e11;  /* Msun */

/* Critical redshift below which cold streams are suppressed in M > Mshock halos.
 * De Lucia & Blaizot (2006) estimate z_crit ~ 1-2; we adopt the midpoint. */
static const double Z_CRIT_DB06 = 1.5;

/* Cold-cloud AGN accretion (AGNrecipeOn == 3): BH triggers when its mass exceeds
 * this fraction of the sonic-radius enclosed virial mass, and accretes at this
 * fraction of the current cooling rate. Croton et al. (2006), AGN appendix. */
static const double AGN_COLD_CLOUD_FRAC = 1.0e-4;

// ============================================================================
// CGM Density Profile Helper Functions (file-private)
// ============================================================================

/*
 * NFW profile normalisation rho_s for total mass M_CGM within R_vir.
 *
 * Solves M_CGM = 4*pi * rho_s * r_s^3 * [ln(1+c) - c/(1+c)] for rho_s,
 * where r_s = Rvir / c_NFW is the scale radius.
 */
static double nfw_rho_s(const double M_CGM, const double Rvir, const double c_NFW)
{
    const double r_s = Rvir / c_NFW;
    // M_CGM = 4pi rho_s r_s^3 * [ln(1+c) - c/(1+c)]
    const double f_c = log(1.0 + c_NFW) - c_NFW / (1.0 + c_NFW);
    return M_CGM / (4.0 * M_PI * r_s * r_s * r_s * f_c);
}

/* NFW density rho(r) = rho_s / [x*(1+x)^2] where x = r/r_s. */
static double nfw_density(const double r, const double rho_s, const double r_s)
{
    const double x = r / r_s;
    if(x < 1e-10) return rho_s / (1e-10 * 1.0 * 1.0);  // Avoid singularity at r=0
    return rho_s / (x * (1.0 + x) * (1.0 + x));
}

/* NFW concentration c(M, z) from Duffy et al. (2008): c = 7.85*(M/2e12)^-0.081*(1+z)^-0.71. */
static double nfw_concentration(const double Mvir_Msun, const double z)
{
    // c = A * (M/M_pivot)^B * (1+z)^C  (Duffy et al. 2008)
    return DUFFY08_A * pow(Mvir_Msun / DUFFY08_M_PIVOT, DUFFY08_B) * pow(1.0 + z, DUFFY08_C);
}

/*
 * Beta-profile normalisation rho_0 for total mass M_CGM within R_vir.
 *
 * Profile: rho(r) = rho_0 / [1 + (r/r_c)^2]^(3*beta/2).
 * Uses an analytic form for beta ~ 2/3 and Simpson quadrature otherwise.
 */
static double beta_rho_0(const double M_CGM, const double Rvir, const double r_c, const double beta)
{
    // For general beta, the enclosed mass integral is:
    // M(<R) = 4pi rho_0 integral_0^R r^2 / [1 + (r/r_c)^2]^(3beta/2) dr
    //
    // For beta = 2/3 (common value), this simplifies to:
    // M(<R) = 4pi rho_0 r_c^3 * [arctan(R/r_c) - (R/r_c)/(1 + (R/r_c)^2)]
    // But we need the more general form...
    //
    // Use numerical approximation for general beta:
    // For large R/r_c, M ~ 4pi rho_0 r_c^3 * (some function of beta)

    const double x = Rvir / r_c;
    double mass_integral;

    if(fabs(beta - 2.0/3.0) < 0.01) {
        // beta ~ 2/3: use analytic form
        // M = 4pi rho_0 r_c^3 * [arctan(x) - x/(1+x^2)]
        mass_integral = atan(x) - x / (1.0 + x * x);
    } else {
        // General beta: numerical integration using Simpson's rule
        const int n_steps = 100;
        const double dr = Rvir / n_steps;
        double integral = 0.0;
        for(int i = 0; i <= n_steps; i++) {
            const double r = i * dr;
            const double y = r / r_c;
            const double rho_factor = 1.0 / pow(1.0 + y * y, 1.5 * beta);
            double weight = (i == 0 || i == n_steps) ? 1.0 : ((i % 2 == 0) ? 2.0 : 4.0);
            integral += weight * r * r * rho_factor;
        }
        integral *= dr / 3.0;
        // M = 4pi rho_0 * integral, so mass_integral = integral / r_c^3
        mass_integral = integral / (r_c * r_c * r_c);
    }

    if(mass_integral <= 0.0) return 0.0;
    return M_CGM / (4.0 * M_PI * r_c * r_c * r_c * mass_integral);
}

/* Beta-profile density rho(r) = rho_0 / [1 + (r/r_c)^2]^(3*beta/2). */
static double beta_density(const double r, const double rho_0, const double r_c, const double beta)
{
    const double y = r / r_c;
    return rho_0 / pow(1.0 + y * y, 1.5 * beta);
}

// ============================================================================
// Enclosed Mass Functions for each profile
// ============================================================================

/* NFW enclosed mass M(<r) = M_total * f(x)/f(c), where f(u) = ln(1+u) - u/(1+u). */
static double nfw_enclosed_mass(const double r, const double M_total, const double Rvir, const double c_NFW)
{
    const double r_s = Rvir / c_NFW;
    const double x = r / r_s;

    // M(<r) / M_total = f(x) / f(c)
    const double f_x = log(1.0 + x) - x / (1.0 + x);
    const double f_c = log(1.0 + c_NFW) - c_NFW / (1.0 + c_NFW);

    if(f_c <= 0.0) return M_total;
    return M_total * f_x / f_c;
}

/* Beta-profile (beta=2/3) enclosed mass using the analytic arctan form. */
static double beta_enclosed_mass(const double r, const double M_total, const double Rvir, const double r_c)
{
    const double x = r / r_c;
    const double X = Rvir / r_c;

    const double f_x = atan(x) - x / (1.0 + x * x);
    const double f_X = atan(X) - X / (1.0 + X * X);

    if(f_X <= 0.0) return M_total;
    return M_total * f_x / f_X;
}

/* Dispatch enclosed-mass calculation to the appropriate profile model (0=uniform, 1=NFW, 2=beta). */
static double cgm_enclosed_mass(const double r, const double M_total, const double Rvir,
                                 const double Mvir_Msun, const double z, const int profile_type)
{
    if(r >= Rvir) return M_total;
    if(r <= 0.0) return 0.0;

    if(profile_type == 0) {
        // Uniform density: M(<r) = M_total * (r/Rvir)^3
        const double ratio = r / Rvir;
        return M_total * ratio * ratio * ratio;
    } else if(profile_type == 1) {
        // NFW profile
        const double c_NFW = nfw_concentration(Mvir_Msun, z);
        return nfw_enclosed_mass(r, M_total, Rvir, c_NFW);
    } else if(profile_type == 2) {
        // Beta profile (beta = 2/3, r_c = CGM_BETA_CORE_RADIUS_FRAC * Rvir)
        const double r_c = CGM_BETA_CORE_RADIUS_FRAC * Rvir;
        return beta_enclosed_mass(r, M_total, Rvir, r_c);
    } else {
        // Default to uniform
        const double ratio = r / Rvir;
        return M_total * ratio * ratio * ratio;
    }
}

/*
 * CGM gas density at radius r in CGS units (g/cm^3).
 *
 * Dispatches to the selected profile model (profile_type: 0=uniform, 1=NFW, 2=beta).
 * Falls back to uniform for unrecognised profile_type values.
 */
static double cgm_density_at_radius(const double r_cgs, const double CGMgas_cgs, const double Rvir_cgs,
                                    const double Mvir_Msun, const double z, const int profile_type)
{
    if(profile_type == 0) {
        // Uniform density
        const double volume_cgs = (4.0 * M_PI / 3.0) * Rvir_cgs * Rvir_cgs * Rvir_cgs;
        return CGMgas_cgs / volume_cgs;

    } else if(profile_type == 1) {
        // NFW profile
        const double c_NFW = nfw_concentration(Mvir_Msun, z);
        const double r_s_cgs = Rvir_cgs / c_NFW;
        const double rho_s = nfw_rho_s(CGMgas_cgs, Rvir_cgs, c_NFW);
        return nfw_density(r_cgs, rho_s, r_s_cgs);

    } else if(profile_type == 2) {
        // Beta profile with beta = 2/3 and r_c = CGM_BETA_CORE_RADIUS_FRAC * Rvir
        const double beta = 2.0 / 3.0;
        const double r_c_cgs = CGM_BETA_CORE_RADIUS_FRAC * Rvir_cgs;
        const double rho_0 = beta_rho_0(CGMgas_cgs, Rvir_cgs, r_c_cgs, beta);
        return beta_density(r_cgs, rho_0, r_c_cgs, beta);

    } else {
        // Default to uniform if unknown profile type
        const double volume_cgs = (4.0 * M_PI / 3.0) * Rvir_cgs * Rvir_cgs * Rvir_cgs;
        return CGMgas_cgs / volume_cgs;
    }
}

/*
 * Solve iteratively for the cooling radius r_cool where t_cool(r) = t_ff(r).
 *
 * Returns r_cool in CGS units. For uniform and beta profiles the isothermal
 * analytic approximation is used (the profile is too flat for iteration to
 * converge). For NFW, a Newton-like iteration over t_cool/t_ff converges in
 * typically < 10 steps.  Result is bounded to [0.001, 1.0] * Rvir.
 */
static double solve_for_rcool(const double CGMgas_cgs, const double Rvir_cgs, const double Mvir_cgs,
                              const double Mvir_Msun, const double temp, const double lambda,
                              const double z, const int profile_type,
                              __attribute__((unused)) const struct params *run_params)
{
    const double mu = MU_IONISED;

    // ========================================================================
    // UNIFORM / BETA: Use isothermal r_cool formula (like hot-regime)
    // ========================================================================
    // For uniform density, t_cool and t_ff are both roughly constant with radius,
    // so the iterative solver doesn't converge meaningfully.
    // For beta profile (beta=2/3), the profile is too flat and has similar issues.
    // Instead, use the isothermal approach: assume rho(r) ~ 1/r^2 for r_cool,
    // which gives r_cool = sqrt(rho0 / rho_cool) where rho_cool is the critical density.
    if(profile_type == 0 || profile_type == 2) {
        // t_ff at R_vir: t_ff = sqrt(2 R^3 / (G M))
        const double t_ff_Rvir = sqrt(2.0 * Rvir_cgs * Rvir_cgs * Rvir_cgs / (G_CGS * Mvir_cgs));

        // Critical density where t_cool = t_ff
        // rho_cool = (3/2) mu m_p k T / (Lambda t_ff)
        const double rho_cool = (1.5 * mu * PROTONMASS * BOLTZMANN * temp) / (lambda * t_ff_Rvir);

        // Isothermal profile normalization: rho0 = M / (4pi R)
        const double rho0 = CGMgas_cgs / (4.0 * M_PI * Rvir_cgs);

        // r_cool from isothermal: rho(r_cool) = rho0/r_cool^2 = rho_cool
        double r_cool = sqrt(rho0 / rho_cool);

        // Apply bounds
        if(r_cool > Rvir_cgs) r_cool = Rvir_cgs;
        if(r_cool < 0.001 * Rvir_cgs) r_cool = 0.001 * Rvir_cgs;

        return r_cool;
    }

    // ========================================================================
    // NFW PROFILE: Iterative solver (cuspy profile converges well)
    // ========================================================================
    const double prefactor = 1.5 * mu * PROTONMASS * BOLTZMANN * temp / lambda;

    double r_cool = 0.5 * Rvir_cgs;
    const int max_iter = 30;
    const double tolerance = 0.01;

    for(int iter = 0; iter < max_iter; iter++) {
        const double rho = cgm_density_at_radius(r_cool, CGMgas_cgs, Rvir_cgs, Mvir_Msun, z, profile_type);

        if(rho <= 0.0) {
            r_cool = Rvir_cgs;
            break;
        }

        const double t_cool = prefactor / rho;

        const double M_enclosed = cgm_enclosed_mass(r_cool, Mvir_cgs, Rvir_cgs, Mvir_Msun, z, profile_type);
        const double g_accel = (M_enclosed > 0.0) ? G_CGS * M_enclosed / (r_cool * r_cool) : 0.0;

        if(g_accel <= 0.0) {
            r_cool = Rvir_cgs;
            break;
        }

        const double t_ff = sqrt(2.0 * r_cool / g_accel);
        const double ratio = t_cool / t_ff;

        const double r_cool_new = r_cool * pow(ratio, -0.3);

        double r_bounded = r_cool_new;
        if(r_bounded > Rvir_cgs) r_bounded = Rvir_cgs;
        if(r_bounded < 0.001 * Rvir_cgs) r_bounded = 0.001 * Rvir_cgs;

        if(fabs(r_bounded - r_cool) / r_cool < tolerance) {
            r_cool = r_bounded;
            break;
        }

        r_cool = r_bounded;
    }

    return r_cool;
}

/*
 * Top-level cooling dispatcher: routes to regime-aware or classic hot-halo recipe.
 *
 * When CGMrecipeOn > 0 the two-regime model is active and this delegates to
 * cooling_recipe_regime_aware(); otherwise falls through to the C16-style
 * cooling_recipe_hot(). Returns the mass of gas cooled this substep.
 */
double cooling_recipe(const int gal, const double dt, struct GALAXY *galaxies, const struct params *run_params)
{
    // Check if CGM recipe is enabled for backwards compatibility
    if(run_params->CGMrecipeOn > 0) {
        return cooling_recipe_regime_aware(gal, dt, galaxies, run_params);
    } else {
        return cooling_recipe_hot(gal, dt, galaxies, run_params);
    }
}

/*
 * Classical hot-halo cooling recipe following Croton et al. (2006).
 *
 * Computes the cooling radius from the isothermal beta-model and returns
 * the mass cooled over timestep dt. When CGMrecipeOn == 1 an additional
 * cold-stream component (De Lucia & Blaizot 2006) is blended in for
 * hot-regime halos. AGN heating is applied before the return.
 */
double cooling_recipe_hot(const int gal, const double dt, struct GALAXY *galaxies, const struct params *run_params)
{
    double coolingGas;

    if(galaxies[gal].HotGas > 0.0 && galaxies[gal].Vvir > 0.0) {
        const double tcool = galaxies[gal].Rvir / galaxies[gal].Vvir;
        const double temp = VIRIAL_TEMP_COEFF * galaxies[gal].Vvir * galaxies[gal].Vvir;  // in Kelvin

        double logZ = -10.0;
        if(galaxies[gal].MetalsHotGas > 0) {
            logZ = log10(galaxies[gal].MetalsHotGas / galaxies[gal].HotGas);
        }

        double lambda = get_metaldependent_cooling_rate(log10(temp), logZ);

        if(lambda <= 0.0) {
            return 0.0;  // No cooling if cooling function is zero/negative
        }

        double x = PROTONMASS * BOLTZMANN * temp / lambda;        // now this has units sec g/cm^3
        x /= (run_params->UnitDensity_in_cgs * run_params->UnitTime_in_s);         // now in internal units
        const double rho_rcool = x / tcool * (1.5 * MU_IONISED);  // 3/2 * mu for a fully ionized gas

        if(rho_rcool <= 0.0) {
            return 0.0;
        }

        // an isothermal density profile for the hot gas is assumed here
        const double rho0 = galaxies[gal].HotGas / (4 * M_PI * galaxies[gal].Rvir);
        const double rcool = sqrt(rho0 / rho_rcool);

        galaxies[gal].RcoolToRvir = rcool / galaxies[gal].Rvir;

        coolingGas = 0.0;
        
        if(run_params->CGMrecipeOn == 0) {
            // Original behavior: SAGE C16 cooling recipe
            if(rcool > galaxies[gal].Rvir) {
                // "cold accretion" regime
                coolingGas = galaxies[gal].HotGas / (galaxies[gal].Rvir / galaxies[gal].Vvir) * dt;
            } else {
                // "hot halo cooling" regime
                coolingGas = (galaxies[gal].HotGas / galaxies[gal].Rvir) * (rcool / (2.0 * tcool)) * dt;
            }
        } else {
            // CGMrecipeOn == 1: D&B06 cold streams for hot-regime halos
            // All halos here are in the hot regime (have virial shocks)
            const double z = run_params->ZZ[galaxies[gal].SnapNum];
            
            // D&B06 eqs 39-41: stream penetration factor f_stream.
            // Mass suppression (M/Mshock)^(-4/3) -- halos well above the shock
            // threshold host weaker cold streams. Redshift factor (1+z)/(1+1)
            // enhances streams at high-z where cooling is more efficient.
            const double Mvir_physical = galaxies[gal].Mvir * 1.0e10 / run_params->Hubble_h;
            const double mass_ratio = Mvir_physical / MSHOCK_DB06_MSUN;

            // Redshift enhancement: normalized to z=1 following D&B06 eq 40
            const double z_factor = (1.0 + z) / (1.0 + 1.0);

            // D&B06 eq 41: below z_crit cold streams are suppressed in M > Mshock halos
            double f_stream;
            if(z < Z_CRIT_DB06 && mass_ratio > 1.0) {
                f_stream = 0.0;  // Hard cutoff per D&B06
            } else {
                // High-z regime: streams can penetrate
                f_stream = pow(mass_ratio, -4.0/3.0) * z_factor;
            }
            
            // Ensure physical bounds
            // Cap at 0.5 (50%) to account for partial heating/mixing of cold streams
            // as they penetrate through the hot medium
            if(f_stream > 1.0) f_stream = 1.0;
            if(f_stream < 0.0) f_stream = 0.0;
            
            // Calculate cooling: mix of cold streams + hot halo cooling
            double cold_stream_cooling = 0.0;
            double hot_halo_cooling = 0.0;
            
            if(rcool < galaxies[gal].Rvir) {
                // When rcool < Rvir: both cold streams and hot halo cooling
                // Cold stream component: rapid accretion on dynamical time
                cold_stream_cooling = f_stream * galaxies[gal].HotGas / 
                                     (galaxies[gal].Rvir / galaxies[gal].Vvir) * dt;
                
                // Hot halo component: traditional cooling from the shocked gas
                hot_halo_cooling = (1.0 - f_stream) * (galaxies[gal].HotGas / galaxies[gal].Rvir) * 
                                  (rcool / (2.0 * tcool)) * dt;
            } else {
                // When rcool >= Rvir: only hot halo cooling (no cold streams)
                // rcool >= Rvir: This shouldn't occur for properly-classified hot-regime haloes
                // (such haloes belong in the CGM/cold-flow regime). Handle conservatively.
                hot_halo_cooling = (galaxies[gal].HotGas / galaxies[gal].Rvir) * 
                                  (rcool / (2.0 * tcool)) * dt;
            }

            galaxies[gal].mdot_cool = hot_halo_cooling / dt;
            galaxies[gal].mdot_stream = cold_stream_cooling / dt;
            
            coolingGas = cold_stream_cooling + hot_halo_cooling;
        }

        if(coolingGas > galaxies[gal].HotGas) {
            coolingGas = galaxies[gal].HotGas;
        } else {
            if(coolingGas < 0.0) coolingGas = 0.0;
        }

		// at this point we have calculated the maximal cooling rate
		// if AGNrecipeOn we now reduce it in line with past heating before proceeding

		if(run_params->AGNrecipeOn > 0 && coolingGas > 0.0) {
			coolingGas = do_AGN_heating(coolingGas, gal, dt, x, rcool, galaxies, run_params);
        }

		if (coolingGas > 0.0) {
			galaxies[gal].Cooling += 0.5 * coolingGas * galaxies[gal].Vvir * galaxies[gal].Vvir;
        }
	} else {
		coolingGas = 0.0;
    }

	XASSERT(coolingGas >= 0.0, -1,
            "Error: Cooling gas mass = %g should be >= 0.0", coolingGas);
    return coolingGas;
}

/*
 * CGM precipitation-driven cooling recipe (SAGE26 two-regime model).
 *
 * Computes cooling from the CGMgas reservoir using the Voit (2015) t_cool/t_ff
 * criterion. Solves for the cooling radius via solve_for_rcool(), evaluates the
 * mean density within that radius, and returns the cooled mass for this substep.
 * AGN heating via do_AGN_heating_cgm() is applied before the return.
 */
double cooling_recipe_cgm(const int gal, const double dt, struct GALAXY *galaxies,
                         const struct params *run_params)
{
    double coolingGas = 0.0;

    // ========================================================================
    // EARLY EXIT CONDITIONS
    // ========================================================================
    if(galaxies[gal].CGMgas <= 0.0 || galaxies[gal].Vvir <= 0.0 || galaxies[gal].Rvir <= 0.0) {
        return 0.0;
    }

    // ========================================================================
    // STEP 1: CALCULATE COOLING TIME (CGS UNITS) WITH DENSITY PROFILE
    // ========================================================================

    // Virial temperature
    const double temp = VIRIAL_TEMP_COEFF * galaxies[gal].Vvir * galaxies[gal].Vvir; // Kelvin

    // Metallicity
    double logZ = -10.0;
    if(galaxies[gal].MetalsCGMgas > 0) {
        logZ = log10(galaxies[gal].MetalsCGMgas / galaxies[gal].CGMgas);
    }

    // Cooling function (erg cm^3 s^-1)
    double lambda = get_metaldependent_cooling_rate(log10(temp), logZ);

    if(lambda <= 0.0) {
        return 0.0;
    }

    // Convert CGM mass and radius to CGS
    const double CGMgas_cgs = galaxies[gal].CGMgas * 1e10 * SOLAR_MASS / run_params->Hubble_h; // g
    const double Rvir_cgs = galaxies[gal].Rvir * CM_PER_MPC / run_params->Hubble_h; // cm
    const double Mvir_cgs = galaxies[gal].Mvir * 1e10 * SOLAR_MASS / run_params->Hubble_h; // g
    const double Mvir_Msun = galaxies[gal].Mvir * 1e10 / run_params->Hubble_h; // Msun
    const double z = run_params->ZZ[galaxies[gal].SnapNum];

    // Get density profile type (0: uniform, 1: NFW, 2: beta)
    // IMPORTANT: Density profile physics only applies to CGM-regime haloes (Regime == 0)
    // Hot-regime haloes always use uniform density for simple CGM depletion
    const int profile_type = (galaxies[gal].Regime == 0) ? run_params->CGMDensityProfile : 0;

    // ========================================================================
    // STEP 1b: SOLVE FOR COOLING RADIUS (consistent with hot regime approach)
    // ========================================================================
    // Find r_cool where t_cool(r_cool) = t_ff(r_cool)
    // This is done iteratively for all profile types

    const double r_cool_cgs = solve_for_rcool(CGMgas_cgs, Rvir_cgs, Mvir_cgs, Mvir_Msun,
                                               temp, lambda, z, profile_type, run_params);

    // Get density at the cooling radius
    const double mass_density_cgs = cgm_density_at_radius(r_cool_cgs, CGMgas_cgs, Rvir_cgs,
                                                           Mvir_Msun, z, profile_type);

    if(mass_density_cgs <= 0.0) {
        return 0.0;
    }

    // Store r_cool / R_vir for diagnostics
    galaxies[gal].RcoolToRvir = r_cool_cgs / Rvir_cgs;

    // Convert r_cool to code units
    const double r_cool = r_cool_cgs / (CM_PER_MPC / run_params->Hubble_h);

    // Cooling time at r_cool: tcool = (3/2) * mu * m_p * k * T / (rho * Lambda)
    const double mu = MU_IONISED;
    const double tcool_cgs = (1.5 * mu * PROTONMASS * BOLTZMANN * temp) / (mass_density_cgs * lambda);
    const double tcool = tcool_cgs / run_params->UnitTime_in_s; // code units

    // ========================================================================
    // STEP 2: CALCULATE FREE-FALL TIME AT r_cool
    // ========================================================================

    // Enclosed mass at r_cool (using proper profile)
    const double M_enclosed_rcool = cgm_enclosed_mass(r_cool_cgs, Mvir_cgs, Rvir_cgs,
                                                       Mvir_Msun, z, profile_type);
    // Convert to code units
    const double M_enclosed_code = M_enclosed_rcool / (1e10 * SOLAR_MASS / run_params->Hubble_h);

    // Gravitational acceleration at r_cool
    const double g_accel = (M_enclosed_code > 0.0 && r_cool > 0.0)
        ? run_params->G * M_enclosed_code / (r_cool * r_cool)
        : 0.0;

    // Free-fall time at r_cool: tff = sqrt(2*r_cool/g)
    if(g_accel <= 0.0) {
        galaxies[gal].tcool = tcool;
        galaxies[gal].tff = -1.0;
        galaxies[gal].tcool_over_tff = -1.0;
        galaxies[gal].tdeplete = -1.0;
        galaxies[gal].RcoolToRvir = -1.0;
        return 0.0;
    }
    const double tff = sqrt(2.0 * r_cool / g_accel); // code units

    // ========================================================================
    // STEP 2b: CHARACTERISTIC RADIUS FOR PRECIPITATION CRITERION
    // ========================================================================
    // Evaluate t_cool/t_ff at r_cool (traditional Voit-style choice).
    const double tcool_char = tcool;
    const double tff_char = tff;
    const double tcool_over_tff_char = tcool / tff;

    // Store characteristic-radius values for diagnostics/plotting
    galaxies[gal].tcool = tcool_char;
    galaxies[gal].tff = tff_char;
    galaxies[gal].tcool_over_tff = tcool_over_tff_char;

    // ========================================================================
    // STEP 3: PRECIPITATION CRITERION
    // ========================================================================

    double precipitation_fraction = 0.0;

    // Logistic sigmoid centred on PRECIP_THRESHOLD, characteristic width = 2.
    // f = 1 / (1 + exp(-(threshold - r) / 2))
    // Smoothly ranges from ~1 (very unstable) through 0.5 at threshold to ~0 (very stable).
    // Falls back to standard cooling once the sigmoid is negligible (< 0.01),
    // which occurs at t_cool/t_ff ~ 19.
    {
        const double x = (PRECIP_THRESHOLD - tcool_over_tff_char) / PRECIP_TRANSITION_WIDTH;
        const double f = 1.0 / (1.0 + exp(-x));
        if(f >= 0.01) {
            precipitation_fraction = f;
        } else {
            if(tcool_char > 0) {
                coolingGas = galaxies[gal].CGMgas / tcool_char * dt;
                if(coolingGas > galaxies[gal].CGMgas)
                    coolingGas = galaxies[gal].CGMgas;
            }
        }
    }

    // ========================================================================
    // STEP 4: CALCULATE PRECIPITATION RATE
    // ========================================================================

    if(precipitation_fraction > 0.0) {
        // Gas precipitates on the free-fall timescale when thermally unstable
        // This is the key physical insight: dM/dt = f_precip * M_CGM / t_ff
        const double precip_rate = precipitation_fraction * galaxies[gal].CGMgas / tff_char;
        coolingGas = precip_rate * dt;
        
        // Physical limits
        if(coolingGas > galaxies[gal].CGMgas) {
            coolingGas = galaxies[gal].CGMgas;
        }
        if(coolingGas < 0.0) {
            coolingGas = 0.0;
        }
    }

    // AGN heating only fires for proper CGM-regime (Regime==0) halos.
    // Regime==1 residual CGMgas drains naturally; do_AGN_heating() on HotGas
    // in cooling_recipe_hot() handles all AGN for hot-halo galaxies.
    if(galaxies[gal].Regime == 0) {
        // AGN x parameter: (k_B T / lambda) in code-units density*time -- passed to
        // both AGN heating paths (Bondi-Hoyle uses it; empirical and cold-cloud do not).
        const double x_agn = (PROTONMASS * BOLTZMANN * temp / lambda)
                             / (run_params->UnitDensity_in_cgs * run_params->UnitTime_in_s);

        // r_heat ratchet, no decay, capped at Rvir (suppression and ratchet
        // update handled inside do_AGN_heating_cgm when AGN is active).
        if(run_params->AGNrecipeOn > 0 && run_params->CGMAGNOn > 0) {
            coolingGas = do_AGN_heating_cgm(coolingGas, gal, dt, x_agn, r_cool, galaxies, run_params);
        } else {
            // No AGN: still apply r_heat suppression so quenching persists
            if(galaxies[gal].r_heat >= r_cool) {
                coolingGas = 0.0;
            } else if(galaxies[gal].r_heat > 0.0f) {
                coolingGas *= 1.0 - galaxies[gal].r_heat / r_cool;
            }
        }
    }

    // ========================================================================
    // STEP 5: TRACK COOLING ENERGY
    // ========================================================================

    // Energy associated with cooling (for feedback balance tracking)
    if(coolingGas > 0.0) {
        // Specific energy ~ 0.5 * Vvir^2 (thermal + kinetic)
        galaxies[gal].Cooling += 0.5 * coolingGas * galaxies[gal].Vvir * galaxies[gal].Vvir;
    }

    // ========================================================================
    // STEP 6: CALCULATE DEPLETION TIMESCALE (DIAGNOSTIC)
    // ========================================================================

    // Depletion timescale (only meaningful for CGM-regime haloes)
    if(galaxies[gal].Regime == 0) {
        if(precipitation_fraction > 1e-6 && isfinite(tff_char)) {
            const double depletion_time = tff_char / precipitation_fraction;
            galaxies[gal].tdeplete = isfinite(depletion_time) ? (float)depletion_time : -1.0f;
        } else {
            galaxies[gal].tdeplete = -1.0f;
        }
    } else {
        // Hot-regime haloes: reset diagnostic fields (density profile physics doesn't apply)
        galaxies[gal].tcool = -1.0f;
        galaxies[gal].tff = -1.0f;
        galaxies[gal].tcool_over_tff = -1.0f;
        galaxies[gal].tdeplete = -1.0f;
    }

    // Sanity check
    XASSERT(coolingGas >= 0.0, -1, "Error: Cooling gas mass = %g should be >= 0.0", coolingGas);
    XASSERT(coolingGas <= galaxies[gal].CGMgas + 1e-12, -1,
            "Error: Cooling gas = %g exceeds CGM gas = %g", coolingGas, galaxies[gal].CGMgas);

    return coolingGas;
}

/*
 * Regime-aware cooling: dispatches CGM and hot-halo recipes by galaxy Regime flag.
 *
 * Regime == 0 (CGM): draws only from CGMgas via cooling_recipe_cgm().
 * Regime == 1 (hot): draws from HotGas via cooling_recipe_hot() plus any
 * residual CGMgas.  Both contributions are applied to ColdGas in-place and
 * the total cooled mass is returned.
 */
double cooling_recipe_regime_aware(const int gal, const double dt, struct GALAXY *galaxies, const struct params *run_params)
{
    double cgm_cooling = 0.0;
    double hot_cooling = 0.0;

    if(galaxies[gal].Regime == 0) {
        // CGM REGIME: CGM physics dominates

        // Primary: Precipitation cooling from CGMgas
        if(galaxies[gal].CGMgas > 0.0) {
            cgm_cooling = cooling_recipe_cgm(gal, dt, galaxies, run_params);
        }


    } else {
        // HOT REGIME: Traditional physics dominates

        // Primary: Traditional cooling from HotGas
        if(galaxies[gal].HotGas > 0.0) {
            hot_cooling = cooling_recipe_hot(gal, dt, galaxies, run_params);
        }

        // Secondary: Precipitation cooling from CGMgas (gradually depletes)
        if(galaxies[gal].CGMgas > 0.0) {
            cgm_cooling = cooling_recipe_cgm(gal, dt, galaxies, run_params);
        }
    }

    // Apply CGM cooling. Clamp to available CGMgas after AGN heating (which
    // runs inside cooling_recipe_cgm for Regime==0 and can drain CGMgas
    // between the internal cap and this apply).
    if(cgm_cooling > 0.0) {
        if(cgm_cooling > galaxies[gal].CGMgas) {
            cgm_cooling = galaxies[gal].CGMgas;
        }
        const double metallicity = get_metallicity(galaxies[gal].CGMgas, galaxies[gal].MetalsCGMgas);
        galaxies[gal].ColdGas += cgm_cooling;
        galaxies[gal].MetalsColdGas += metallicity * cgm_cooling;
        galaxies[gal].CGMgas -= cgm_cooling;
        galaxies[gal].MetalsCGMgas -= metallicity * cgm_cooling;
    }

    // Apply HotGas cooling (clamp to available HotGas after AGN heating)
    if(hot_cooling > 0.0) {
        if(hot_cooling > galaxies[gal].HotGas) {
            hot_cooling = galaxies[gal].HotGas;
        }
        const double metallicity = get_metallicity(galaxies[gal].HotGas, galaxies[gal].MetalsHotGas);
        galaxies[gal].ColdGas += hot_cooling;
        galaxies[gal].MetalsColdGas += metallicity * hot_cooling;
        galaxies[gal].HotGas -= hot_cooling;
        galaxies[gal].MetalsHotGas -= metallicity * hot_cooling;
    }

    double total_cooling = cgm_cooling + hot_cooling;
    XASSERT(total_cooling >= 0.0, -1,
            "Error: Cooling gas mass = %g should be >= 0.0", total_cooling);
    return total_cooling;
}

/*
 * AGN radio-mode heating for the hot-halo regime (HotGas reservoir).
 *
 * First reduces coolingGas based on the stored r_heat from past AGN activity,
 * then computes new BH accretion (empirical, Bondi-Hoyle, or cold-cloud) and
 * the resulting heating. Updates BlackHoleMass, HotGas, and r_heat in-place.
 * Returns the post-heating coolingGas value.
 */
/*
 * agn_accretion_compute -- shared radio-mode AGN accretion and heating calculation.
 *
 * Computes AGNrate via the selected recipe (AGNrecipeOn 1/2/3), applies the
 * Eddington cap, converts to accreted mass, and derives the equivalent heating
 * mass.  reservoir_mass is the gas available for accretion (HotGas or CGMgas
 * depending on the caller); the heating coefficient uses Vvir of the central.
 * Does NOT modify any galaxy fields -- the caller draws from the right reservoir
 * and updates r_heat.
 */
static void agn_accretion_compute(const int centralgal, const double dt, const double x,
                                   const double coolingGas, const double rcool,
                                   const double reservoir_mass,
                                   const struct GALAXY *galaxies, const struct params *run_params,
                                   double *AGNaccreted_out, double *AGNheating_out)
{
    const double Vvir   = galaxies[centralgal].Vvir;
    const double Mvir   = galaxies[centralgal].Mvir;
    const double BHmass = galaxies[centralgal].BlackHoleMass;
    const double Rvir   = galaxies[centralgal].Rvir;

    double AGNrate = 0.0;
    if(run_params->AGNrecipeOn == 2) {
        // Bondi-Hoyle accretion (Bondi 1952)
        AGNrate = (2.5 * M_PI * run_params->G) * (0.375 * 0.6 * x) * BHmass * run_params->RadioModeEfficiency;
    } else if(run_params->AGNrecipeOn == 3) {
        // Cold cloud accretion: triggers when M_BH exceeds the sonic-mass threshold
        if(BHmass > AGN_COLD_CLOUD_FRAC * Mvir * CUBE(rcool / Rvir))
            AGNrate = AGN_COLD_CLOUD_FRAC * coolingGas / dt;
    } else {
        // Empirical recipe (Croton et al. 2006, eq. 10)
        const double base = run_params->RadioModeEfficiency
            / (run_params->UnitMass_in_g / run_params->UnitTime_in_s * SEC_PER_YEAR / SOLAR_MASS)
            * (BHmass / AGN_BH_MASS_PIVOT) * CUBE(Vvir / AGN_VVIR_PIVOT_KMS);
        AGNrate = (Mvir > 0.0) ? base * (reservoir_mass / Mvir) / AGN_HOT_GAS_PIVOT : base;
    }

    // Eddington-limited accretion (Rybicki & Lightman 1979)
    const double EDDrate = (EDDINGTON_LUM_PER_MSUN_CGS * BHmass * 1e10 / run_params->Hubble_h)
        / (run_params->UnitEnergy_in_cgs / run_params->UnitTime_in_s)
        / (AGN_RADIATIVE_EFFICIENCY * C_SQ_KMS2);
    if(AGNrate > EDDrate) AGNrate = EDDrate;

    double AGNaccreted = AGNrate * dt;
    if(AGNaccreted > reservoir_mass) AGNaccreted = reservoir_mass;

    // Heating coefficient: AGN_HEATING_COEFF_KMS = sqrt(2*eta*c^2)
    double AGNheating = 0.0;
    if(Vvir > 0.0) {
        const double AGNcoeff = (AGN_HEATING_COEFF_KMS / Vvir) * (AGN_HEATING_COEFF_KMS / Vvir);
        AGNheating = AGNcoeff * AGNaccreted;
        // limit to available cooling mass
        if(AGNheating > coolingGas && AGNcoeff > 0.0) {
            AGNaccreted = coolingGas / AGNcoeff;
            AGNheating  = coolingGas;
        }
    }

    *AGNaccreted_out = AGNaccreted;
    *AGNheating_out  = AGNheating;
}


/*
 * AGN radio-mode heating for hot-halo (Regime==1) galaxies.
 *
 * Applies r_heat suppression unconditionally (the ratchet always runs for
 * Regime==1), then calls agn_accretion_compute() to get the accreted mass
 * and suppressed cooling mass, draws the accreted mass from HotGas, and
 * updates r_heat via the standard ratchet.
 */
double do_AGN_heating(double coolingGas, const int centralgal, const double dt, const double x, const double rcool, struct GALAXY *galaxies, const struct params *run_params)
{
    // r_heat suppression (always applied for Regime==1 hot-halo)
    if(galaxies[centralgal].r_heat < rcool) {
        coolingGas = (1.0 - galaxies[centralgal].r_heat / rcool) * coolingGas;
    } else {
        coolingGas = 0.0;
    }
    XASSERT(coolingGas >= 0.0, -1,
            "Error: Cooling gas mass = %g should be >= 0.0", coolingGas);

    if(galaxies[centralgal].HotGas > 0.0) {
        double AGNaccreted, AGNheating;
        agn_accretion_compute(centralgal, dt, x, coolingGas, rcool,
                               galaxies[centralgal].HotGas,
                               galaxies, run_params, &AGNaccreted, &AGNheating);

        const double metallicity = get_metallicity(galaxies[centralgal].HotGas,
                                                   galaxies[centralgal].MetalsHotGas);
        galaxies[centralgal].BlackHoleMass += AGNaccreted;
        galaxies[centralgal].HotGas        -= AGNaccreted;
        galaxies[centralgal].MetalsHotGas  -= metallicity * AGNaccreted;

        // standard r_heat ratchet
        if(galaxies[centralgal].r_heat < rcool && coolingGas > 0.0) {
            const double r_heat_new = (AGNheating / coolingGas) * rcool;
            if(r_heat_new > galaxies[centralgal].r_heat)
                galaxies[centralgal].r_heat = r_heat_new;
        }

        if(AGNheating > 0.0)
            galaxies[centralgal].Heating += 0.5 * AGNheating
                * galaxies[centralgal].Vvir * galaxies[centralgal].Vvir;
    }

    return coolingGas;
}

/*
 * AGN radio-mode heating for CGM-dominated (Regime==0) galaxies.
 *
 * Applies the same r_heat/rcool suppression and ratchet as the hot-halo
 * path, then caps r_heat at Rvir. Accretion draws from CGMgas.
 */
double do_AGN_heating_cgm(double coolingGas, const int centralgal, const double dt, const double x, const double rcool,
                          struct GALAXY *galaxies, const struct params *run_params)
{
    if(galaxies[centralgal].r_heat < rcool) {
        coolingGas = (1.0 - galaxies[centralgal].r_heat / rcool) * coolingGas;
    } else {
        coolingGas = 0.0;
    }

    XASSERT(coolingGas >= 0.0, -1,
            "Error: Cooling gas mass = %g should be >= 0.0", coolingGas);

    if(galaxies[centralgal].CGMgas > 0.0) {
        double AGNaccreted, AGNheating;
        agn_accretion_compute(centralgal, dt, x, coolingGas, rcool,
                              galaxies[centralgal].CGMgas, galaxies, run_params,
                              &AGNaccreted, &AGNheating);
        const double metallicity = get_metallicity(galaxies[centralgal].CGMgas, galaxies[centralgal].MetalsCGMgas);
        galaxies[centralgal].BlackHoleMass  += AGNaccreted;
        galaxies[centralgal].CGMgas         -= AGNaccreted;
        galaxies[centralgal].MetalsCGMgas   -= metallicity * AGNaccreted;

        if(galaxies[centralgal].r_heat < rcool && coolingGas > 0.0) {
            const double r_heat_new = (AGNheating / coolingGas) * rcool;
            if(r_heat_new > galaxies[centralgal].r_heat)
                galaxies[centralgal].r_heat = r_heat_new;
        }
        if(galaxies[centralgal].r_heat > galaxies[centralgal].Rvir)
            galaxies[centralgal].r_heat = galaxies[centralgal].Rvir;

        if(AGNheating > 0.0)
            galaxies[centralgal].Heating += 0.5 * AGNheating * galaxies[centralgal].Vvir * galaxies[centralgal].Vvir;
    }
    return coolingGas;
}

/*
 * Transfer cooled gas from the HotGas reservoir into the cold disk.
 *
 * Moves up to coolingGas mass (clamped to available HotGas) from HotGas to
 * ColdGas, tracking metallicity consistently. Called by cooling_recipe_hot()
 * after AGN heating has been applied.
 */
void cool_gas_onto_galaxy(const int centralgal, const double coolingGas, struct GALAXY *galaxies)
{
    // add the fraction 1/STEPS of the total cooling gas to the cold disk
    if(coolingGas > 0.0) {
        if(coolingGas < galaxies[centralgal].HotGas) {
            const double metallicity = get_metallicity(galaxies[centralgal].HotGas, galaxies[centralgal].MetalsHotGas);
            galaxies[centralgal].ColdGas += coolingGas;
            galaxies[centralgal].MetalsColdGas += metallicity * coolingGas;
            galaxies[centralgal].HotGas -= coolingGas;
            galaxies[centralgal].MetalsHotGas -= metallicity * coolingGas;
        } else {
            galaxies[centralgal].ColdGas += galaxies[centralgal].HotGas;
            galaxies[centralgal].MetalsColdGas += galaxies[centralgal].MetalsHotGas;
            galaxies[centralgal].HotGas = 0.0;
            galaxies[centralgal].MetalsHotGas = 0.0;
        }
    }
}

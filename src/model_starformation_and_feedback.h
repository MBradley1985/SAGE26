/*
 * model_starformation_and_feedback.h -- public interface for SF and SN feedback.
 *
 * Declares starformation_and_feedback() (the main per-substep entry point),
 * update_from_star_formation() and update_from_feedback() (mass/metal bookkeeping
 * helpers), and starformation_ffb() (the feedback-free burst prescription).
 * The HYDROGEN_MASS_FRAC constant used across SF prescriptions is defined here.
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#pragma once

#define HYDROGEN_MASS_FRAC 0.74  // primordial hydrogen mass fraction

#ifdef __cplusplus
extern "C" {
#endif

    #include "core_allvars.h"

    /* After cold gas has been removed, fit H2gas and H1gas back into its hydrogen, 0.74 ColdGas:
     * H2gas first (unchanged if it fits), H1gas in the hydrogen left.  Capped separately, as before,
     * their sum could exceed the hydrogen.  float, as the SAGE26 clamps it replaces. */
    static inline void clamp_phases_to_hydrogen(struct GALAXY *g)
    {
        const float max_h = (g->ColdGas > 0.0f) ? g->ColdGas * HYDROGEN_MASS_FRAC : 0.0f;
        if(g->H2gas > max_h) g->H2gas = max_h;
        const float max_h1 = max_h - g->H2gas;
        if(g->H1gas > max_h1) g->H1gas = (max_h1 > 0.0f) ? max_h1 : 0.0f;
    }

    /* functions in model_starformation_and_feedback.c */
    extern void starformation_and_feedback(const int p, const int centralgal, const double time, const double dt, const int halonr, const int step,
                                           struct GALAXY *galaxies, const struct params *run_params);
    /* returns the dust added to MetalsStellarMass (DustySAGEMode 2; else 0) */
    extern double update_from_star_formation(const int p, const double stars, const double metallicity, struct GALAXY *galaxies, const struct params *run_params);
    extern void update_from_feedback(const int p, const int centralgal, const double reheated_mass, double ejected_mass, const double metallicity,
                                     struct GALAXY *galaxies, const struct params *run_params);

    extern void starformation_ffb(const int p, const int centralgal, const double dt, const int step,
                                  struct GALAXY *galaxies, const struct params *run_params);

    /* Print totals of the silent physics clamps (H2 cap, negative HI,
       reheated-mass cap); called from finalize_sage() in VERBOSE builds. */
    // extern void report_sf_clamp_counts(void);

#ifdef __cplusplus
}
#endif


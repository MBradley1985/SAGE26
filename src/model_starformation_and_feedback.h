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

    /* functions in model_starformation_and_feedback.c */
    extern void starformation_and_feedback(const int p, const int centralgal, const double time, const double dt, const int halonr, const int step,
                                           struct GALAXY *galaxies, const struct params *run_params);
    extern void update_from_star_formation(const int p, const double stars, const double metallicity, struct GALAXY *galaxies, const struct params *run_params);
    extern void update_from_feedback(const int p, const int centralgal, const double reheated_mass, double ejected_mass, const double metallicity,
                                     struct GALAXY *galaxies, const struct params *run_params);

    extern void starformation_ffb(const int p, const int centralgal, const double dt, const int step,
                                  struct GALAXY *galaxies, const struct params *run_params);

#ifdef __cplusplus
}
#endif


/*
 * model_darksage_feedback.h -- public interface for DarkSage outflow energetics (DarkSAGEMode == 1).
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

    #include "core_allvars.h"

    /* functions in model_darksage_feedback.c */

    /* Specific energy of galaxy p's hot gas: HotGasPotential + 0.5 (Vvir^2 + j_hot^2 / R2_hot_av),
     * j_hot = 2 Vvir CoolScaleRadius [(km/s)^2] */
    extern double darksage_hot_specific_energy(const int p, const struct GALAXY *galaxies);

    /* SupernovaRecipeOn == 1: split the gas removed from annulus i into reheated (fountain) and
     * ejected (outflow) by the DarkSage wind-speed test; eject_e is the outflow's specific energy */
    extern void darksage_split_feedback(const int p, const int centralgal, const int i, const double removed,
                                        double *reheated, double *ejected, double *eject_e,
                                        const struct GALAXY *galaxies, const struct params *run_params);

    /* SupernovaRecipeOn == 2: DarkSage energy-driven reheated and ejected masses for *stars formed in
     * annulus i; *stars, reheated and ejected are scaled down together to fit the annulus gas */
    extern void darksage_feedback_masses(const int p, const int centralgal, const int i, double *stars,
                                         double *reheated, double *ejected, double *eject_e,
                                         const struct GALAXY *galaxies, const struct params *run_params);

    /* Metallicity of gas blown out of an annulus of metallicity Z (DarkSage entrains 10% of the
     * metals under SupernovaRecipeOn == 2; SAGE26 feedback carries the full Z) */
    extern double darksage_outflow_metallicity(const double Z, const struct params *run_params);

    extern void darksage_add_to_fountain(const int centralgal, const double mass, const double metals,
                                         struct GALAXY *galaxies, const struct params *run_params);
    extern void darksage_add_to_outflow(const int centralgal, const double mass, const double metals,
                                        const double e_spec, struct GALAXY *galaxies);
    extern void darksage_eject_hot(const int centralgal, double ejected_mass, struct GALAXY *galaxies,
                                   const struct params *run_params);

#ifdef __cplusplus
}
#endif

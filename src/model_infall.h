/*
 * model_infall.h -- public interface for halo baryon infall.
 *
 * Declares infall_recipe() (baryon accretion from the cosmic web),
 * satellite stripping, reionization suppression (Gnedin 2000 filtering mass),
 * and add_infall_to_hot() which deposits infalling gas into HotGas or CGMgas
 * depending on the halo regime.
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

    #include "core_allvars.h"

    /* functions in model_infall.c */
    extern double infall_recipe(const int centralgal, const int ngal, const double Zcurr, struct GALAXY *galaxies, const struct params *run_params);
    extern void strip_from_satellite(const int centralgal, const int gal, const double Zcurr, struct GALAXY *galaxies, const struct params *run_params);
    extern double do_reionization(const int gal, const double Zcurr, struct GALAXY *galaxies, const struct params *run_params);
    extern void add_infall_to_hot(const int gal, double infallingGas, struct GALAXY *galaxies, const struct params *run_params);

#ifdef __cplusplus
}
#endif

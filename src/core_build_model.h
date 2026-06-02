/*
 * core_build_model.h -- public interface for the merger-tree walker.
 *
 * Declares construct_galaxies(), which recursively walks a merger-tree forest
 * depth-first and calls evolve_galaxies() once all progenitors of a halo have
 * been processed.
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "core_allvars.h"

    /* functions in core_build_model.c */
    extern int construct_galaxies(const int halonr, int *numgals, int *galaxycounter, int *maxgals, struct halo_data *halos,
                                  struct halo_aux_data *haloaux, struct GALAXY **ptr_to_galaxies, struct GALAXY **ptr_to_halogal,
                                  struct params *run_params);

#ifdef __cplusplus
}
#endif

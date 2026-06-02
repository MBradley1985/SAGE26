/*
 * model_disk_instability.h -- public interface for disk instability.
 *
 * Declares check_disk_instability(), which applies the Toomre (1964) stability
 * criterion to each galaxy disk and transfers unstable mass to the bulge and
 * black hole on the disk dynamical timescale.
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

    #include "core_allvars.h"

    /* functions in model_disk_instability.c */
    extern void check_disk_instability(const int p, const int centralgal, const int halonr, const double time, const double dt, const int step,
                                       struct GALAXY *galaxies, struct params *run_params);
    

#ifdef __cplusplus
}
#endif

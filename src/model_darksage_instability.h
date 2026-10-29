/*
 * model_darksage_instability.h -- public interface for annulus-resolved disc instability (DarkSAGEMode == 1).
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

    #include "core_allvars.h"

    /* functions in model_darksage_instability.c */
    extern void check_disk_instability_darksage(const int p, const int centralgal, const double dt, const int step,
                                                struct GALAXY *galaxies, const struct params *run_params);

#ifdef __cplusplus
}
#endif

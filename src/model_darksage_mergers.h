/*
 * model_darksage_mergers.h -- public interface for merger gas placement (DarkSAGEMode == 1).
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

    #include "core_allvars.h"

    /* functions in model_darksage_mergers.c */
    /* Lay satellite p's cold gas onto central t's annuli by angular momentum, before the masses are summed */
    extern void darksage_merge_gas(const int t, const int p, struct GALAXY *galaxies, const struct params *run_params);

#ifdef __cplusplus
}
#endif

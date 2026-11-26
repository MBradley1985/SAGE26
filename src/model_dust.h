/*
 * model_dust.h -- public interface for dust production and evolution (DustySAGEMode >= 1).
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

    #include "core_allvars.h"

    /* functions in model_dust.c */
    /* Read the stellar yield tables and set the snapshot times (once, after the snapshot list) */
    extern void dust_init(struct params *run_params);
    /* One substep of dust physics for galaxy p, after its star formation and mergers */
    extern void dust_evolve(const int p, const int centralgal, const double dt, struct GALAXY *galaxies,
                            const struct params *run_params);

#ifdef __cplusplus
}
#endif

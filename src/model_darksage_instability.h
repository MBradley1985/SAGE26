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
    /* DarkSage cold-gas velocity dispersion 11 + 11.3 z km/s, in code velocity units */
    extern double darksage_cold_gas_dispersion(const double z, const struct params *run_params);
    extern void check_disk_instability_darksage(const int p, const int centralgal, const double dt, const int step,
                                                struct GALAXY *galaxies, const struct params *run_params);

#ifdef __cplusplus
}
#endif

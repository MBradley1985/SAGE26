/*
 * model_darksage_stripping.h -- public interface for ram-pressure and tidal stripping (DarkSAGEMode == 1).
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

    #include "core_allvars.h"

    /* functions in model_darksage_stripping.c */
    /* Strip satellite gal's disc; returns 1 if it must instead be disrupted to the ICS */
    extern int darksage_ram_pressure_stripping(const int centralgal, const int gal, const double time,
                                               struct GALAXY *galaxies, const struct params *run_params);

#ifdef __cplusplus
}
#endif

/*
 * model_reincorporation.h -- public interface for ejected gas reincorporation.
 *
 * Declares reincorporate_gas(), which returns supernova wind-ejected material
 * from EjectedMass back to HotGas (or CGMgas) on a timescale proportional to
 * the halo dynamical time, following Croton et al. (2016).
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

    #include "core_allvars.h"

    /* functions in model_reincorporation.c */
    extern void reincorporate_gas(const int centralgal, const double dt, struct GALAXY *galaxies, const struct params *run_params);

#ifdef __cplusplus
}
#endif

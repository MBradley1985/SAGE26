/*
 * model_darksage_disc.h -- public interface for disc annulus radii (DarkSAGEMode == 1).
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

    #include "core_allvars.h"

    /* functions in model_darksage_disc.c */
    extern void update_disc_radii(const int p, struct GALAXY *galaxies, const struct halo_data *halos,
                                  const struct params *run_params);
    extern void darksage_project_disc(const float DiscMass[N_BINS], const double cos_angle, double NewDisc[N_BINS],
                                      const struct params *run_params, const double Rvir, const double Vvir);
    extern void darksage_precess_gas(const int p, const double dt, struct GALAXY *galaxies, const struct params *run_params);

#ifdef __cplusplus
}
#endif

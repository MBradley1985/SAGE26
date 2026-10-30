/*
 * model_darksage_sf.h -- public interface for annulus-resolved H2 and SF (DarkSAGEMode == 1).
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

    #include "core_allvars.h"

    /* functions in model_darksage_sf.c */
    extern double darksage_star_formation_rate(const int p, double sfr_bin[N_BINS],
                                               struct GALAXY *galaxies, const struct params *run_params);
    extern void darksage_remove_disc_gas(const int p, const double mass, const float weight[N_BINS],
                                         struct GALAXY *galaxies);
    extern void darksage_sync_disc_metals(const int p, struct GALAXY *galaxies);
    extern void darksage_add_disc_metals(const int p, const double metals, const double sfr_bin[N_BINS],
                                         struct GALAXY *galaxies);

#ifdef __cplusplus
}
#endif

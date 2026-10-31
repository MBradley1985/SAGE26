/*
 * model_darksage_instability.h -- public interface for where disc instability takes its mass from (DarkSAGEMode >= 1).
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
    /* Mass each annulus would have to lose to be stable (DarkSage Toomre Q), gas and stars */
    extern void darksage_unstable_profile(const int p, double gas_w[N_BINS], double star_w[N_BINS],
                                          const struct GALAXY *galaxies, const struct params *run_params);
    /* Annulus side of SAGE26 moving unstable_stars of disc stars to the instability bulge */
    extern void darksage_instability_stars_to_bulge(const int p, const double unstable_stars, const double star_w[N_BINS],
                                                    struct GALAXY *galaxies);

#ifdef __cplusplus
}
#endif

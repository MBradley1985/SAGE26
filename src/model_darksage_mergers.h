/*
 * model_darksage_mergers.h -- public interface for DarkSAGEMode merger placement of gas and stars.
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

    #include "core_allvars.h"

    /* functions in model_darksage_mergers.c */

    /* Place satellite p's gas in central t's annuli by angular momentum (minor) or rebuild the gas
     * disc in the orbital plane (major), shrink it after retrograde gas, update the merger bulge's
     * dispersion and j, and move p's fountain / outflow gas.  Call before the masses are combined. */
    extern void darksage_merge_galaxies(const int t, const int p, struct GALAXY *galaxies,
                                        const struct params *run_params);

    /* Major merger: set the merger bulge's dispersion from all t's stars and clear the secular bulge.
     * Call before the stellar disc is emptied. */
    extern void darksage_stars_to_bulge(const int t, struct GALAXY *galaxies);

#ifdef __cplusplus
}
#endif

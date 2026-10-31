/*
 * model_darksage_sf.h -- public interface for annulus-resolved H2 and SF (DarkSAGEMode >= 1).
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

    #include "core_allvars.h"


    /* functions in model_darksage_sf.c */
    /* SFprescription / H2DepletionMode SFR and H2 of each annulus, galaxy untouched */
    extern void darksage_annulus_rates(const int p, double sfr_bin[N_BINS], double h2_bin[N_BINS],
                                       const struct GALAXY *galaxies, const struct params *run_params);
    /* Same, setting DiscH2, H2gas and H2DepletionTime_Gyr from the annuli (DarkSAGEMode 2);
     * returns the total SFR */
    extern double darksage_star_formation_rate(const int p, double sfr_bin[N_BINS],
                                               struct GALAXY *galaxies, const struct params *run_params);
    /* Take mass from the gas (stars) annuli in proportion to weight (NULL: to what they hold),
     * at each annulus's metallicity; taken (may be NULL) receives what left each annulus */
    extern void darksage_remove_disc_gas(const int p, const double mass, const double weight[N_BINS], double taken[N_BINS],
                                         struct GALAXY *galaxies);
    extern void darksage_remove_disc_stars(const int p, const double mass, const double weight[N_BINS], double taken[N_BINS],
                                           struct GALAXY *galaxies);
    /* While set (non-NULL), weight replaces the caller's in darksage_remove_disc_gas() */
    extern void darksage_set_gas_removal_weights(const double *weight);
    /* Rescale the gas and disc-star annulus metals to MetalsColdGas and MetalsStellarMass - MetalsBulgeMass */
    extern void darksage_sync_disc_metals(const int p, struct GALAXY *galaxies);
    extern void darksage_add_disc_metals(const int p, const double metals, const double weight[N_BINS],
                                         struct GALAXY *galaxies);

#ifdef __cplusplus
}
#endif

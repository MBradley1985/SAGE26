/*
 * model_misc.h -- galaxy utilities, plus umbrella include for the physics
 * helper modules that were split out of model_misc.c.
 *
 * Declares galaxy initialisation, disk/bulge size calculations, and small
 * shared helpers, and pulls in model_regimes.h (CGM/FFB classification),
 * model_halo_properties.h (virial properties, concentration), and
 * model_h2_chemistry.h (H2 fraction prescriptions) so existing consumers
 * keep a single include.
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

    #include "core_allvars.h"

    #include "model_h2_chemistry.h"
    #include "model_halo_properties.h"
    #include "model_regimes.h"

    /*
     * Deposit metals (or gas + metals) into a galaxy's hot-phase reservoir,
     * routed by regime: CGMgas when the CGM recipe is active and the galaxy
     * is CGM-regime (Regime == 0), HotGas otherwise. These helpers replace
     * the identical if/else ladder previously repeated at every deposit site;
     * the += operations are unchanged, so results are bit-identical.
     */
    static inline void add_metals_to_hot_reservoir(struct GALAXY *central, const struct params *run_params, const double metals)
    {
        if(run_params->CGMrecipeOn == 1 && central->Regime == 0) {
            central->MetalsCGMgas += metals;
        } else {
            central->MetalsHotGas += metals;
        }
    }

    static inline void add_gas_to_hot_reservoir(struct GALAXY *central, const struct params *run_params, const double gas, const double metals)
    {
        if(run_params->CGMrecipeOn == 1 && central->Regime == 0) {
            central->CGMgas += gas;
            central->MetalsCGMgas += metals;
        } else {
            central->HotGas += gas;
            central->MetalsHotGas += metals;
        }
    }

    /*
     * SF prescriptions that track molecular gas (H2gas/H1gas): every
     * prescription except 0 (Croton+06 cold-gas threshold) and 2
     * (Somerville+25 cold-gas efficiency). SFprescription is validated to
     * [0, 7] at parameter-read time, so the two-term form is exact.
     */
    static inline int sf_prescription_tracks_h2(const int sfprescription)
    {
        return sfprescription != 0 && sfprescription != 2;
    }

    /* BR06 pressure-based H2 family: 1 (BR06) and 3 (Somerville+25 + BR06). */
    static inline int sf_prescription_is_br06(const int sfprescription)
    {
        return sfprescription == 1 || sfprescription == 3;
    }

    /* functions in model_misc.c */
    extern void init_galaxy(const int p, const int halonr, int *galaxycounter, const struct halo_data *halos, struct GALAXY *galaxies, const struct params *run_params);
    extern double get_metallicity(const double gas, const double metals);
    extern double get_disk_radius(const int halonr, const int p, const struct halo_data *halos, const struct GALAXY *galaxies);
    extern double get_bulge_radius(const int p, struct GALAXY *galaxies, const struct params *run_params);
    extern double dmax(const double x, const double y);
    extern void update_instability_bulge_radius(const int p, const double delta_mass,
                                     const double old_disk_radius,
                                     struct GALAXY *galaxies, const struct params *run_params);


#ifdef __cplusplus
}
#endif

/*
 * model_misc.h -- public interface for galaxy utilities and regime logic.
 *
 * Declares galaxy initialisation, virial property accessors, the CGM-regime
 * determination function (determine_and_store_regime, based on Voit 2015),
 * FFB regime classification, H2 fraction prescriptions (BR06, KD12, K13,
 * radial integration), halo concentration calculators, and size-mass relations.
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

    #include "core_allvars.h"

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
    extern double get_virial_velocity(const int halonr, const struct halo_data *halos, const struct params *run_params);
    extern double get_virial_radius(const int halonr, const struct halo_data *halos, const struct params *run_params);
    extern double get_virial_mass(const int halonr, const struct halo_data *halos, const struct params *run_params);
    extern double get_disk_radius(const int halonr, const int p, const struct halo_data *halos, const struct GALAXY *galaxies);
    extern double get_bulge_radius(const int p, struct GALAXY *galaxies, const struct params *run_params);
    extern double dmax(const double x, const double y);
    extern void determine_and_store_regime(const int ngal, struct GALAXY *galaxies, 
                                const struct params *run_params);
    extern float calculate_molecular_fraction_BR06(float gas_surface_density, float stellar_surface_density, float disk_scale_length_pc);

    extern float calculate_molecular_fraction_radial_integration(const int gal, struct GALAXY *galaxies,
                                                      const struct params *run_params,
                                                      double *strdot_code_out);
    extern double calculate_tdep_K13_Gyr(float Sigma_gas, float Sigma_star, float rs_pc, float Z_prime, float f_H2);

    extern double calculate_ffb_threshold_mass(const double z, const struct params *run_params);
    extern double calculate_ffb_fraction(const double Mvir, const double z, const struct params *run_params);

    extern void determine_and_store_ffb_regime(const int ngal, const double Zcurr, struct GALAXY *galaxies,
                                            const struct params *run_params);
    extern double interpolate_concentration_ishiyama21(const double logM, const double z, const struct params *run_params);
    extern double concentration_from_vmax_vvir(const double Vmax, const double Vvir);
    extern double get_halo_concentration(const int p, const double z, const struct GALAXY *galaxies,
                                          const struct params *run_params);
    extern double calculate_gmax_BK25(const int p, const double z, const struct GALAXY *galaxies,
                                       const struct params *run_params);
    extern void update_instability_bulge_radius(const int p, const double delta_mass, 
                                     const double old_disk_radius,
                                     struct GALAXY *galaxies, const struct params *run_params);

    extern float calculate_H2_fraction_KD12(const float surface_density, const float metallicity, const float clumping_factor);
    extern double calculate_H2_fraction_K13(double Sigma_gas_msun_pc2, double metallicity, double clumping_factor);
    extern double calculate_H2_fraction_GD14(double Sigma_gas_msun_pc2, double metallicity, double rs_pc);


#ifdef __cplusplus
}
#endif

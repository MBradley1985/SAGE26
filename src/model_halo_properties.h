/*
 * model_halo_properties.h -- public interface for virial properties and
 * halo concentration models.
 *
 * Virial mass/velocity/radius accessors and NFW concentration
 * (Ishiyama+21 lookup, Vmax/Vvir conversion, ConcentrationOn dispatch).
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

    #include "core_allvars.h"

    extern double get_virial_mass(const int halonr, const struct halo_data *halos, const struct params *run_params);
    extern double get_virial_velocity(const int halonr, const struct halo_data *halos, const struct params *run_params);
    extern double get_virial_radius(const int halonr, const struct halo_data *halos, const struct params *run_params);
    extern double virial_radius_from_mass(const double mvir, const int snapnum, const struct params *run_params);
    extern double interpolate_clustering_mass(const double z, const struct params *run_params);
    extern double interpolate_concentration_ishiyama21(const double logM, const double z, const struct params *run_params);
    extern double concentration_from_vmax_vvir(const double Vmax, const double Vvir);
    extern double get_halo_concentration(const int p, const double z, const struct GALAXY *galaxies,
                                          const struct params *run_params);

#ifdef __cplusplus
}
#endif

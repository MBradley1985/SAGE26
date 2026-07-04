/*
 * model_h2_chemistry.h -- public interface for the H2 fraction prescriptions.
 *
 * BR06 (Blitz & Rosolowsky 2006), KD12 (Krumholz & Dekel 2012), K13
 * (Krumholz 2013) with its depletion time, GD14 (Gnedin & Draine 2014),
 * and the shared radial integration over the exponential disk.
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

    #include "core_allvars.h"

    extern float calculate_molecular_fraction_BR06(float gas_surface_density, float stellar_surface_density, float disk_scale_length_pc);
    extern float calculate_molecular_fraction_radial_integration(const int gal, struct GALAXY *galaxies,
                                                      const struct params *run_params,
                                                      double *strdot_code_out);
    extern double calculate_tdep_K13_Gyr(float Sigma_gas, float Sigma_star, float rs_pc, float Z_prime, float f_H2);
    extern float calculate_H2_fraction_KD12(const float surface_density, const float metallicity, const float clumping_factor);
    extern double calculate_H2_fraction_K13(double Sigma_gas_msun_pc2, double metallicity, double clumping_factor);
    extern double calculate_H2_fraction_GD14(double Sigma_gas_msun_pc2, double metallicity, double rs_pc);

#ifdef __cplusplus
}
#endif

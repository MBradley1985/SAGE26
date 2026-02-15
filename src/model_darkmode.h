/**
 * @file model_darkmode.h
 * @brief DarkSage-style radially-resolved disk physics
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "core_allvars.h"

/**
 * @brief Compute local H2 fraction and SFR in each disk annulus
 */
extern double compute_local_star_formation(const int p, const double dt,
                                          struct GALAXY *galaxies, const struct params *run_params,
                                          double sfr_local[N_BINS], double h2_local[N_BINS]);

/**
 * @brief Apply local star formation in each disk annulus
 */
extern double apply_local_star_formation(const int p, const double dt, const int step,
                                        const double sfr_local[N_BINS],
                                        struct GALAXY *galaxies, const struct params *run_params);

/**
 * @brief Check for disk instabilities and transfer mass to bulge
 */
extern void check_local_disk_instability(const int p, const int centralgal, const double dt, const int step,
                                        struct GALAXY *galaxies, const struct params *run_params);

/**
 * @brief Apply radial gas flows due to viscous evolution
 */
extern void apply_radial_gas_flows(const int p, const double dt, struct GALAXY *galaxies,
                                  const struct params *run_params);

/**
 * @brief Compute Toomre Q parameter for disk stability
 */
extern double compute_toomre_Q(double Sigma_gas, double Sigma_stars, double r_mid, double Vvir,
                              const struct params *run_params);

#ifdef __cplusplus
}
#endif

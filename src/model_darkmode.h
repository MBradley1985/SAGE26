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

/* ========================================================================== */
/* FULL DARKMODE PHYSICS FUNCTIONS                                            */
/* ========================================================================== */

/**
 * @brief Rotate a 3D vector using Rodrigues' rotation formula
 */
extern void rotate_vector(float v[3], const float axis[3], const double angle_deg);

/**
 * @brief Compute total angular momentum of gas or stellar disc
 */
extern double get_disc_ang_mom(const int p, const int component, double J[3],
                               struct GALAXY *galaxies, const struct params *run_params);

/**
 * @brief Precess gas disc toward stellar disc alignment
 */
extern void precess_gas(const int p, const double dt, struct GALAXY *galaxies,
                        const struct params *run_params);

/**
 * @brief Compute combined two-fluid Toomre Q (Romeo & Wiegert 2011)
 */
extern double compute_combined_toomre_Q(double Sigma_gas, double Sigma_stars,
                                        double sigma_gas, double sigma_stars,
                                        double r_mid, double Vvir);

/**
 * @brief Handle unstable gas with j-conservation (sink inward)
 */
extern void deal_with_unstable_gas(const int p, const int bin, const double unstable_gas,
                                   struct GALAXY *galaxies, const struct params *run_params);

/**
 * @brief Handle unstable stars with j-conservation (migrate inward one bin)
 */
extern void deal_with_unstable_stars(const int p, const int bin, const double unstable_stars,
                                     struct GALAXY *galaxies, const struct params *run_params);

/**
 * @brief Check for disk instabilities with combined Q and j-conservation
 */
extern void check_full_disk_instability(const int p, const int centralgal, const double dt, const int step,
                                        struct GALAXY *galaxies, const struct params *run_params);

#ifdef __cplusplus
}
#endif

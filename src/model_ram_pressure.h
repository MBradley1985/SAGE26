/*
 * model_ram_pressure.h -- public interface for ram-pressure stripping of
 * satellite ISM.
 *
 * Declares the Gunn & Gott (1972) stripped-fraction criterion and the
 * per-satellite driver called from evolve_galaxies() when
 * RamPressureStrippingOn == 1.  Complementary to strip_from_satellite()
 * (model_infall.c), which strips the hot/CGM phase.
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

    #include "core_allvars.h"

    /* functions in model_ram_pressure.c */
    extern double ram_pressure_stripped_fraction(const double P_ram_cgs,
                                                 const double Sigma_gas0_cgs,
                                                 const double Sigma_disk0_cgs);
    extern void ram_pressure_strip_satellite(const int centralgal, const int gal,
                                             const double Zcurr, const double dt,
                                             const double t_strip,
                                             struct GALAXY *galaxies,
                                             const struct params *run_params);

#ifdef __cplusplus
}
#endif

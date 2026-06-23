#pragma once

#ifdef __cplusplus
extern "C" {
#endif

    #include "core_allvars.h"

    /* Eddington accretion and limiting functions for black hole growth. */
    double dynamical_time(const double r_bulge, const double M_bulge_encl, const struct params *run_params);

    /* Calculate the Eddington accretion rate for a black hole */
    double eddington_accretion_rate(const double black_hole_mass, const struct params *run_params);

    /* Limit accretion rate by Eddington limit if flag is set.
     * Returns the final accretion rate (either limited or unlimited).
     * Stores the pre-limited accretion rate in BHMaxaccretionRate[snapnum], the Eddington rate in BHEddingtonRateLimit[snapnum],
     * and the accretion type (0 or 1) in BHAccretionType[snapnum]. */
    double eddington_limited_accretion_rate(double accretion_rate, int eddington_flag, double black_hole_mass,
                                            int snapnum, int bh_accretion_type, const struct params *run_params,
                                            float BHAccretionType[ABSOLUTEMAXSNAPS], float BHMaxaccretionRate[ABSOLUTEMAXSNAPS], float BHEddingtonRateLimit[ABSOLUTEMAXSNAPS]);

#ifdef __cplusplus
}
#endif
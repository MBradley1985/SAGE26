/*
 * model_regimes.h -- public interface for regime classification.
 *
 * CGM vs hot-halo regime (Dekel & Birnboim 2006 shock mass) and the
 * feedback-free-burst regime (Li+24 mass thresholds, BK25 acceleration
 * criterion) with their threshold/fraction helpers.
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

    #include "core_allvars.h"

    extern void determine_and_store_regime(const int ngal, struct GALAXY *galaxies,
                                const struct params *run_params);
    extern void determine_and_store_ffb_regime(const int ngal, const double Zcurr, struct GALAXY *galaxies,
                                            const struct params *run_params);
    extern double calculate_ffb_threshold_mass(const double z, const struct params *run_params);
    extern double calculate_ffb_fraction(const double Mvir, const double z, const struct params *run_params);
    extern double calculate_gmax_BK25(const int p, const double z, const struct GALAXY *galaxies,
                                       const struct params *run_params);

#ifdef __cplusplus
}
#endif

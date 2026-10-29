/*
 * model_darksage_stellar_pop.h -- public interface for stellar age bins and delayed feedback (DarkSAGEMode == 1).
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

    #include "core_allvars.h"

    /* functions in model_darksage_stellar_pop.c */

    /* Mass fraction returned (out[0]) and supernovae per unit mass formed (out[1], code units) by a
     * stellar population between ages t0 and t1 [code time] */
    extern void darksage_return_and_sn(const double t0, const double t1, const struct params *run_params,
                                       double out[2]);
    /* Same for stars just formed, over half the width of age bin k_now */
    extern void darksage_instant_return_and_sn(const int k_now, const struct params *run_params, double out[2]);
    /* Age-bin edges and FinalRecycleFraction; call once after Age[] is set */
    extern void darksage_init_age_bins(struct params *run_params);
    /* Age bin containing lookback time `time` */
    extern int darksage_age_bin(const double time, const struct params *run_params);
    /* Move a fraction frac of every age bin from one component to another (NULL target: remove) */
    extern void darksage_move_ages(float mass_from[N_AGE_BINS], float metals_from[N_AGE_BINS], const double frac,
                                   float mass_to[N_AGE_BINS], float metals_to[N_AGE_BINS]);
    /* Mass return, metals and SN energy of the older stellar populations of galaxy p over dt */
    extern void darksage_delayed_feedback(const int p, const int centralgal, const double time, const double dt,
                                          struct GALAXY *galaxies, const struct params *run_params);
    /* Stellar mass whose age bins disagree with its component mass, as a fraction of StellarMass */
    extern float darksage_age_mismatch(const struct GALAXY *g);

#ifdef __cplusplus
}
#endif

/*
 * core_cool_func.h -- public interface for the Sutherland-Dopita cooling tables.
 *
 * Declares the cooling-table loader and the metallicity-dependent cooling rate
 * interpolator used by model_cooling_heating.c.  Tables cover log T in [4, 8.5]
 * and metallicities from primordial to solar.
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

    /* functions in core_cool_func.c */
    extern void read_cooling_functions(void);
    extern double get_metaldependent_cooling_rate(const double logTemp, double logZ);

#ifdef __cplusplus
}
#endif

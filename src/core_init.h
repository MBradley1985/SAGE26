/*
 * core_init.h -- public interface for model initialisation.
 *
 * Declares init(), which sets up unit conversions, initialises run-wide
 * constants from the parameter struct, and loads the Sutherland-Dopita
 * cooling tables before the first forest is processed.
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

    /* functions in core_init.c */
    extern void init(struct params *run_params);

#ifdef __cplusplus
}
#endif



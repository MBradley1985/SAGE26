/*
 * core_read_parameter_file.h -- public interface for the parameter file parser.
 *
 * Declares read_parameter_file(), which reads a SAGE .par file and populates
 * the run_params struct with all physics switches, file paths, unit choices,
 * and model parameters for the run.
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

    /* functions in core_read_parameter_file.c */
    extern int read_parameter_file(const char *fname, struct params *run_params);

#ifdef __cplusplus
}
#endif

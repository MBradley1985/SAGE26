/*
 * sage.h -- top-level public API for SAGE26.
 *
 * Declares run_sage() and finalize_sage(), the two entry points called by
 * main.c (serial) or by external callers via libsage.so (e.g. SAGE-PSO).
 * Also defines SAGE_VERSION and SAGE_DATA_VERSION, which identify output
 * binary compatibility -- do not change without bumping both constants.
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#pragma once

#include "core_allvars.h"

// DO NOT TOUCH THESE TWO DEFINITIONS.
// They are checked when we processed the output. Bad things will happen if you do touch them!
#define SAGE_DATA_VERSION "1.00"
#define SAGE_VERSION "1.00"


#ifdef __cplusplus
extern "C" {
#endif

    /* API for sage */
    extern int run_sage(const int ThisTask, const int NTasks, const char *param_file, void **params);
    extern int finalize_sage(void *params);

#ifdef __cplusplus
}
#endif

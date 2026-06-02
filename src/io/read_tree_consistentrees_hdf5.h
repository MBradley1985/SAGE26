/*
 * read_tree_consistentrees_hdf5.h -- public interface for the Consistent Trees HDF5 reader.
 *
 * Declares the three entry points used by core_io_tree.c to load merger trees
 * in the Consistent Trees HDF5 format (Behroozi+2013 stored in HDF5):
 * setup (per process), load (one forest), and cleanup.
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif /* working with c++ compiler */

/* for definition of struct halo_data */
#include "../core_allvars.h"

/* Proto-Types */
    extern int setup_forests_io_ctrees_hdf5(struct forest_info *forests_info, const int ThisTask, const int NTasks, struct params *run_params);
    extern int64_t load_forest_ctrees_hdf5(int64_t forestnr, struct halo_data **halos, struct forest_info *forests_info, struct params *run_params);
    extern void cleanup_forests_io_ctrees_hdf5(struct forest_info *forests_info);

#ifdef __cplusplus
}
#endif /* working with c++ compiler */

/*
 * read_tree_gadget4_hdf5.h -- public interface for the Gadget-4 HDF5 tree reader.
 *
 * Declares the three entry points used by core_io_tree.c to load merger trees
 * written by Gadget-4's FOF/SubFind group finder in HDF5 format:
 * setup (per process), load (one forest), and cleanup.
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif /* working with c++ compiler */

/* for definition of struct halo_data and struct forest_info */
#include "../core_allvars.h"

    /* Proto-Types */
    extern int setup_forests_io_gadget4_hdf5(struct forest_info *forests_info,
                                             const int ThisTask, const int NTasks, struct params *run_params);
    extern int64_t load_forest_gadget4_hdf5(const int64_t forestnr, struct halo_data **halos, struct forest_info *forests_info);
    extern void cleanup_forests_io_gadget4_hdf5(struct forest_info *forests_info);

#ifdef __cplusplus
}
#endif /* working with c++ compiler */

/*
 * read_tree_lhalo_hdf5.h -- public interface for the LHaloTree HDF5 tree reader.
 *
 * Declares the three entry points used by core_io_tree.c to load merger trees in
 * the LHaloTree HDF5 on-disk format (as produced for Illustris/TNG): setup (per
 * process at startup), load (per forest during the main loop), and cleanup (at
 * shutdown). Implementation lives in read_tree_lhalo_hdf5.c.
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif /* working with c++ compiler */

#include <hdf5.h>

/* for definition of struct halo_data and struct forest_info */
#include "../core_allvars.h"

    /* Proto-Types */
    extern int setup_forests_io_lht_hdf5(struct forest_info *forests_info,
                                         const int ThisTask, const int NTasks, struct params *run_params);
    extern int64_t load_forest_lht_hdf5(const int64_t forestnr, struct halo_data **halos, struct forest_info *forests_info);
    extern void cleanup_forests_io_lht_hdf5(struct forest_info *forests_info);

#ifdef __cplusplus
}
#endif /* working with c++ compiler */

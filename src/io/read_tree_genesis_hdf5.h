/*
 * read_tree_genesis_hdf5.h -- public interface for the Genesis HDF5 tree reader.
 *
 * Declares the three entry points used by core_io_tree.c to load merger trees in
 * the Genesis HDF5 on-disk format (ASTRO 3D VELOCIraptor+TreeFrog output): setup
 * (per process at startup), load (per forest during the main loop), and cleanup
 * (at shutdown). Implementation lives in read_tree_genesis_hdf5.c.
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
    extern int setup_forests_io_genesis_hdf5(struct forest_info *forests_info, const int ThisTask, const int NTasks, struct params *run_params);
    extern int64_t load_forest_genesis_hdf5(int64_t forestnr, struct halo_data **halos, struct forest_info *forests_info, struct params *run_params);
    extern void cleanup_forests_io_genesis_hdf5(struct forest_info *forests_info);

#ifdef __cplusplus
}
#endif /* working with c++ compiler */

/*
 * save_gals_hdf5.h -- public interface for the HDF5 galaxy catalogue writer.
 *
 * Defines HDF5_GALAXY_OUTPUT (the column-array struct holding all per-galaxy
 * properties for one forest before they are written as HDF5 datasets) and
 * declares the three entry points -- initialize, save, finalize -- plus
 * create_hdf5_master_file(), called by core_save.c when OutputFormat is
 * sage_hdf5.
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif // working with c++ compiler //

#include "../core_allvars.h"

#include "save_gals_hdf5_fields.h"

/* One heap array per output field per output snapshot (the write buffers).
 * The per-dataset fields are generated from GALAXY_OUTPUT_FIELDS (see
 * save_gals_hdf5_fields.h); TaskForestNr and the 2-D SFH arrays are the
 * only members handled outside that list. */
struct HDF5_GALAXY_OUTPUT
{
#define SAGE_FIELD_STRUCT_MEMBER(dset, field, ctype, h5t, desc, unit) ctype *field;
    GALAXY_OUTPUT_FIELDS(SAGE_FIELD_STRUCT_MEMBER)
#undef SAGE_FIELD_STRUCT_MEMBER

    int64_t *TaskForestNr;   /* cpu-local forest number; internal bookkeeping, never written */

    /* cumulative star formation history - tracks stellar mass formed at each snapshot (controlled by SaveFullSFH) */
    float *SFHMassDisk;      /* Shape: [ngalaxies, SimMaxSnaps] - mass formed in disk at each snapshot */
    float *SFHMassBulge;     /* Shape: [ngalaxies, SimMaxSnaps] - mass formed in bulge at each snapshot */
};
    
    // Proto-Types //
    extern int32_t initialize_hdf5_galaxy_files(const int filenr, struct save_info *save_info, const struct params *run_params);
    
    extern int32_t save_hdf5_galaxies(const int64_t task_forestnr, const int32_t num_gals, struct forest_info *forest_info,
                                      struct halo_data *halos, struct halo_aux_data *haloaux, struct GALAXY *halogal,
                                      struct save_info *save_info, const struct params *run_params);

    extern int32_t finalize_hdf5_galaxy_files(const struct forest_info *forest_info, struct save_info *save_info,
                                              const struct params *run_params);

    extern int32_t create_hdf5_master_file(const struct params *run_params);
    
    
#ifdef __cplusplus
}
#endif

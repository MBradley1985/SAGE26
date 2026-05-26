#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "core_allvars.h"
#include "core_save.h"
#include "core_mymalloc.h"

#include "io/save_gals_binary.h"

#ifdef HDF5
#include "io/save_gals_hdf5.h"
#endif

static int32_t generate_galaxy_indices(const struct halo_data *halos, const struct halo_aux_data *haloaux,
                                struct GALAXY *halogal, const int64_t numgals,
                                const int64_t treenr, const int32_t filenr,
                                const int64_t filenr_mulfac, const int64_t forestnr_mulfac,const struct params *run_params);

int32_t initialize_galaxy_files(const int rank, const struct forest_info *forest_info, struct save_info *save_info, const struct params *run_params)
{
    int32_t status;

    if(run_params->NumSnapOutputs > ABSOLUTEMAXSNAPS) {
        fprintf(stderr,"Error: Attempting to write snapshot = '%d' will exceed allocated memory space for '%d' snapshots\n",
                run_params->NumSnapOutputs, ABSOLUTEMAXSNAPS);
        fprintf(stderr,"To fix this error, simply increase the value of `ABSOLUTEMAXSNAPS` and recompile\n");
        return INVALID_OPTION_IN_PARAMS;
    }

    switch(run_params->OutputFormat) {

    case(sage_binary):
      status = initialize_binary_galaxy_files(rank, forest_info, save_info, run_params);
      break;

#ifdef HDF5
    case(sage_hdf5):
      status = initialize_hdf5_galaxy_files(rank, save_info, run_params);
      break;
#endif

    default:
      fprintf(stderr, "Error: Unknown OutputFormat in `initialize_galaxy_files()`.\n");
      status = INVALID_OPTION_IN_PARAMS;
      break;

    }

    return status;
}


int32_t save_galaxies(const int64_t task_forestnr, const int numgals, struct halo_data *halos,
                      struct forest_info *forest_info,
                      struct halo_aux_data *haloaux, struct GALAXY *halogal,
                      struct save_info *save_info, const struct params *run_params)
{
    int32_t status = EXIT_FAILURE;

    int32_t OutputGalCount[run_params->SimMaxSnaps];
    for(int32_t snap_idx = 0; snap_idx < run_params->SimMaxSnaps; snap_idx++) {
        OutputGalCount[snap_idx] = 0;
    }

    int32_t *OutputGalOrder = mymalloc(numgals * sizeof(*(OutputGalOrder)));
    if(OutputGalOrder == NULL) {
        fprintf(stderr,"Error: Could not allocate memory for %d int elements in array `OutputGalOrder`\n", numgals);
        return MALLOC_FAILURE;
    }

    for(int32_t gal_idx = 0; gal_idx < numgals; gal_idx++) {
        OutputGalOrder[gal_idx] = -1;
        haloaux[gal_idx].output_snap_n = -1;
    }

    /* remap mergeIntoID from galaxy-array indices to output-order indices */
    for(int32_t snap_idx = 0; snap_idx < run_params->NumSnapOutputs; snap_idx++) {
        for(int32_t gal_idx  = 0; gal_idx < numgals; gal_idx++) {
            if(halogal[gal_idx].SnapNum == run_params->ListOutputSnaps[snap_idx]) {
                OutputGalOrder[gal_idx] = OutputGalCount[snap_idx];
                OutputGalCount[snap_idx]++;
                haloaux[gal_idx].output_snap_n = snap_idx;
            }
        }
    }

    for(int32_t gal_idx = 0; gal_idx < numgals; gal_idx++) {
        const int mergeID = halogal[gal_idx].mergeIntoID;
        if(mergeID > -1) {
            if( ! (mergeID >= 0 && mergeID < numgals) ) {
                fprintf(stderr,"Error: For galaxy number %d, expected mergeintoID to be within [0, %d) "
                        "but found mergeintoID = %d instead\n", gal_idx, numgals, mergeID);
                fprintf(stderr,"Additional debugging info\n");
                fprintf(stderr,"task_forestnr = %"PRId64" snapshot = %d halonr = %d MostBoundID = %lld\n",
                        task_forestnr, halogal[gal_idx].SnapNum, halogal[gal_idx].HaloNr, halos[halogal[gal_idx].HaloNr].MostBoundID);
                return -1;
            }
            halogal[gal_idx].mergeIntoID = OutputGalOrder[halogal[gal_idx].mergeIntoID];
        }
    }

    /* GalaxyIndex needs the original file-tree number, not task-local forestnr */
    int64_t original_treenr = forest_info->original_treenr[task_forestnr];
    int32_t original_filenr = forest_info->FileNr[task_forestnr];

    status = generate_galaxy_indices(halos, haloaux, halogal, numgals,
                                     original_treenr, original_filenr,
                                     run_params->FileNr_Mulfac, run_params->ForestNr_Mulfac,
                                     run_params);
    if(status != EXIT_SUCCESS) {
        return EXIT_FAILURE;
    }

    switch(run_params->OutputFormat) {

    case(sage_binary):
        status = save_binary_galaxies(task_forestnr, numgals, OutputGalCount, forest_info,
                                      halos, haloaux, halogal, save_info, run_params);
        break;

#ifdef HDF5
    case(sage_hdf5):
        status = save_hdf5_galaxies(task_forestnr, numgals, forest_info, halos, haloaux, halogal, save_info, run_params);
        break;
#endif

    default:
        fprintf(stderr, "Uknown OutputFormat in `save_galaxies()`.\n");
        status = INVALID_OPTION_IN_PARAMS;

    }

    myfree(OutputGalOrder);

    return status;
}


int32_t finalize_galaxy_files(const struct forest_info *forest_info, struct save_info *save_info, const struct params *run_params)
{

    int32_t status = EXIT_FAILURE;

    switch(run_params->OutputFormat) {

    case(sage_binary):
        status = finalize_binary_galaxy_files(forest_info, save_info, run_params);
        break;

#ifdef HDF5
    case(sage_hdf5):
        status = finalize_hdf5_galaxy_files(forest_info, save_info, run_params);
        break;
#endif

    default:
        fprintf(stderr, "Error: Unknown OutputFormat in `finalize_galaxy_files()`.\n");
        status = INVALID_OPTION_IN_PARAMS;
        break;
    }

    return status;
}

static int32_t generate_galaxy_indices(const struct halo_data *halos, const struct halo_aux_data *haloaux,
                                struct GALAXY *halogal, const int64_t numgals,
                                const int64_t forestnr, const int32_t filenr,
                                const int64_t filenr_mulfac, const int64_t forestnr_mulfac,
                                const struct params *run_params)
{

    const enum Valid_TreeTypes TreeType = run_params->TreeType;
    for(int64_t gal_idx = 0; gal_idx < numgals; ++gal_idx) {
        struct GALAXY *this_gal = &halogal[gal_idx];

        uint32_t GalaxyNr = this_gal->GalaxyNr;
        uint32_t CentralGalaxyNr = halogal[haloaux[halos[this_gal->HaloNr].FirstHaloInFOFgroup].FirstGalaxy].GalaxyNr;

        if(GalaxyNr > forestnr_mulfac || (filenr_mulfac > 0 && forestnr*forestnr_mulfac > filenr_mulfac)) {
            fprintf(stderr, "When determining a unique Galaxy Number, we assume two things\n"
                    "1. Current galaxy numnber = %u is less than multiplication factor for trees (=%"PRId64")\n"
                    "2. That (the total number of trees * tree multiplication factor = %"PRId64") is less than the file "\
                    "multiplication factor = %"PRId64" (only relevant if file multiplication factor is non-zero).\n"
                    "At least one of these two assumptions have been broken.\n"
                    "Simulation trees file number %d\tOriginal tree number %"PRId64"\tGalaxy Number %d "
                    "forestnr_mulfac = %"PRId64" forestnr*forestnr_mulfac = %"PRId64"\n", GalaxyNr, forestnr_mulfac,
                    forestnr*forestnr_mulfac, filenr_mulfac,
                    filenr, forestnr, GalaxyNr, forestnr_mulfac, forestnr*forestnr_mulfac);
            switch (TreeType)
        {
#ifdef HDF5
        case lhalo_hdf5:
            fprintf(stderr,"It is likely that your tree file contains too many trees or a tree contains too many galaxies, you can increase the maximum number "\
                    "of trees per file with the parameter run_params->FileNr_Mulfac at l. 264 in src/io/read_tree_lhalo_hdf5.c. "\
                    "If a tree contains too many galaxies, you can increase run_params->ForestNr_Mulfac in the same location. "\
                    "If all trees are stored in a single file, FileNr_Mulfac can in principle be set to zero to remove the limit\n.");
            break;

        case gadget4_hdf5:
            fprintf(stderr,"It is likely that your tree file contains too many trees or a tree contains too many galaxies, you can increase the maximum number "\
                    "of trees per file with the parameter run_params->FileNr_Mulfac at l. 536 in src/io/read_tree_gadget4_hdf5.c. "\
                    "If a tree contains too many galaxies, you can increase run_params->ForestNr_Mulfac in the same location. "\
                    "If all trees are stored in a single file, FileNr_Mulfac can in principle be set to zero to remove the limit.\n");
            break;

        case genesis_hdf5:
            fprintf(stderr,"It is likely that your tree file contains too many trees or a tree contains too many galaxies, you can increase the maximum number "\
                    "of trees per file with the parameter run_params->FileNr_Mulfac at l. 492 in src/io/read_tree_genesis_hdf5.c. "\
                    "If a tree contains too many galaxies, you can increase run_params->ForestNr_Mulfac in the same location. "\
                    "If all trees are stored in a single file, FileNr_Mulfac can in principle be set to zero to remove the limit.\n");
            break;

        case consistent_trees_hdf5:
            fprintf(stderr,"It is likely that your tree file contains too many trees or a tree contains too many galaxies, you can increase the maximum number "\
                    "of trees per file with the parameter run_params->FileNr_Mulfac at l. 389 in src/io/read_tree_consistentrees_hdf5.c. "\
                    "If a tree contains too many galaxies, you can increase run_params->ForestNr_Mulfac in the same location. "\
                    "If all trees are stored in a single file, FileNr_Mulfac can in principle be set to zero to remove the limit.\n");
            break;
#endif

        case lhalo_binary:
            fprintf(stderr,"It is likely that your tree file contains too many trees or a tree contains too many galaxies, you can increase the maximum number "\
                    "of trees per file with the parameter run_params->FileNr_Mulfac at l. 226 in src/io/read_tree_lhalo_binary.c. "\
                    "If a tree contains too many galaxies, you can increase run_params->ForestNr_Mulfac in the same location. "\
                    "If all trees are stored in a single file, FileNr_Mulfac can in principle be set to zero to remove the limit.\n");

            break;

        case consistent_trees_ascii:
            fprintf(stderr,"It is likely that you have a tree with too many galaxies. For consistent trees the number of galaxies per trees "\
                    "is limited for the ID to to fit in 64 bits, see run_params->ForestNr_Mulfac at l. 319 in src/io/read_tree_consistentrees_ascii.c. "\
                    "If you have not set a finite run_params->FileNr_Mulfac, this format may not be ideal for your purpose.\n");
            break;

        default:
            fprintf(stderr, "Your tree type has not been included in the switch statement for function ``%s`` in file ``%s``.\n", __FUNCTION__, __FILE__);
            fprintf(stderr, "Please add it there.\n");
            return INVALID_OPTION_IN_PARAMS;
        }
            
          return EXIT_FAILURE;
        }

        if((forestnr > 0 && ((uint64_t) forestnr > (0xFFFFFFFFFFFFFFFFULL/(uint64_t) forestnr_mulfac))) ||
           (filenr > 0 && ((uint64_t) filenr > (0xFFFFFFFFFFFFFFFFULL/(uint64_t) filenr_mulfac)))) {
            fprintf(stderr,"Error: While generating an unique Galaxy Index. The multiplication required to "
                    "generate the ID will overflow 64-bit\n"
                    "forestnr = %"PRId64" forestnr_mulfac = %"PRId64" filenr = %d filenr_mulfac = %"PRId64"\n",
                    forestnr, forestnr_mulfac, filenr, filenr_mulfac);

            return EXIT_FAILURE;
        }

        const uint64_t id_from_forestnr = forestnr_mulfac * forestnr;
        const uint64_t id_from_filenr = filenr_mulfac * filenr;

        if(id_from_forestnr > (0xFFFFFFFFFFFFFFFFULL - id_from_filenr)) {
            fprintf(stderr,"Error: While generating an unique Galaxy Index. The addition required to generate the ID will overflow 64-bits \n");
            fprintf(stderr,"id_from_forestnr = %"PRIu64 "id_from_filenr = %"PRIu64"\n", id_from_forestnr, id_from_filenr);
            return EXIT_FAILURE;
        }

        const uint64_t id_from_forest_and_file = id_from_forestnr + id_from_filenr;
        if((GalaxyNr > (0xFFFFFFFFFFFFFFFFULL - id_from_forest_and_file)) ||
           (CentralGalaxyNr > (0xFFFFFFFFFFFFFFFFULL - id_from_forest_and_file))) {
            fprintf(stderr,"Error: While generating an unique Galaxy Index. The addition required to generate the ID will overflow 64-bits \n");
            fprintf(stderr,"id_from_forest_and_file = %"PRIu64" GalaxyNr = %u CentralGalaxyNr = %u\n", id_from_forest_and_file, GalaxyNr, CentralGalaxyNr);
            return EXIT_FAILURE;
        }

        this_gal->GalaxyIndex = GalaxyNr + id_from_forest_and_file;
        this_gal->CentralGalaxyIndex= CentralGalaxyNr + id_from_forest_and_file;
    }

    return EXIT_SUCCESS;
}

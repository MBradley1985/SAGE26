#include <stdlib.h>
#include <limits.h>

#include "core_tree_utils.h"
#include "sglib.h"

int reorder_lhalo_to_lhvt(const int32_t nhalos, struct halo_data *forest, int32_t test, int32_t **orig_index)
{
    int32_t *prog_len=NULL, *desc_len=NULL;
    int32_t *len=NULL, *foflen=NULL;
    if(test > 0) {
        prog_len = calloc(nhalos, sizeof(*prog_len));
        desc_len = calloc(nhalos, sizeof(*prog_len));
        len = calloc(nhalos, sizeof(*len));
        foflen = calloc(nhalos, sizeof(*foflen));

        if(prog_len == NULL || desc_len == NULL || len == NULL || foflen == NULL ) {
            fprintf(stderr,"Warning: malloc failure for LHalotree fields - disabling tests even though tests were requested\n");
            test = 0;
        }
    }


    int32_t *index = malloc(nhalos * sizeof(*index));
    if(index == NULL) {
        perror(NULL);
        fprintf(stderr,"Error: Could not allocate memory for the index array for a forest with nhalos = %d. "
                "Requested size = %zu\n",nhalos, sizeof(*index) * nhalos);
        return MALLOC_FAILURE;
    }

    for(int32_t i=0;i<nhalos;i++) {
        index[i] = i;
        if(test > 0) {
            len[i] = forest[i].Len;
            if(forest[i].FirstHaloInFOFgroup < 0 || forest[i].FirstHaloInFOFgroup >= nhalos){
                fprintf(stderr,"For halonum = %d fofhalo index = %d should be within limits [0, %d)",
                        i, forest[i].FirstHaloInFOFgroup, nhalos);
                return EXIT_FAILURE;
            }
            foflen[i] = forest[forest[i].FirstHaloInFOFgroup].Len;
            if(forest[i].FirstProgenitor == -1 || (forest[i].FirstProgenitor >= 0 && forest[i].FirstProgenitor < nhalos)) {
                prog_len[i] = forest[i].FirstProgenitor == -1 ? -1:forest[forest[i].FirstProgenitor].Len;
            } else {
                fprintf(stderr,"Error. In %s: halonum = %d with FirstProg = %d has invalid value. Should be within [0, %d)\n",
                        __FUNCTION__,i,forest[i].FirstProgenitor, nhalos);
                return EXIT_FAILURE;
            }
            desc_len[i] = forest[i].Descendant == -1 ? -1:forest[forest[i].Descendant].Len;
        }
    }

    /* sort key: snapshot, then FOF group, then FOF halo first, then descending subhalo mass */
#define SNAPNUM_FOFHALO_MVIR_COMPARATOR(x, i, j)    ((x[i].SnapNum != x[j].SnapNum) ? (x[i].SnapNum - x[j].SnapNum):FOFHALO_COMPARATOR(x, i, j))
#define FOFHALO_COMPARATOR(x, i, j) ((x[i].FirstHaloInFOFgroup != x[j].FirstHaloInFOFgroup) ? (x[i].FirstHaloInFOFgroup - x[j].FirstHaloInFOFgroup):FOFHALO_SUBLEN_COMPARATOR(x,i, j))

#define FOFHALO_SUBLEN_COMPARATOR(x, i, j)     ((x[i].FirstHaloInFOFgroup == index[i]) ? -1:( (x[j].FirstHaloInFOFgroup == index[j]) ? 1: (x[j].Len - x[i].Len)) )

#define MULTIPLE_ARRAY_EXCHANGER(type,a,i,j) {                      \
        SGLIB_ARRAY_ELEMENTS_EXCHANGER(struct halo_data, forest,i,j); \
        SGLIB_ARRAY_ELEMENTS_EXCHANGER(int32_t, index, i, j);       \
    }

    SGLIB_ARRAY_HEAP_SORT_MULTICOMP(struct halo_data, forest, nhalos, SNAPNUM_FOFHALO_MVIR_COMPARATOR, MULTIPLE_ARRAY_EXCHANGER);

#undef SNAPNUM_FOFHALO_MVIR_COMPARATOR
#undef FOFHALO_COMPARATOR
#undef FOFHALO_SUBLEN_COMPARATOR
#undef MULTIPLE_ARRAY_EXCHANGER


    int status = fix_mergertree_index(forest, nhalos, index);
    if(status != EXIT_SUCCESS) {
        return status;
    }

    if(test > 0) {
        status = EXIT_FAILURE;
        int32_t *index_for_old_order = calloc(nhalos, sizeof(*index_for_old_order));
        if(index_for_old_order == NULL) {
            return EXIT_FAILURE;
        }

        for(int32_t i=0;i<nhalos;i++) {
            index_for_old_order[index[i]] = i;
        }

        for(int32_t i=0;i<nhalos;i++) {
            const int32_t old_index = index[i];
            if(len[old_index] != forest[i].Len) {
                fprintf(stderr,"Error: forest[%d].Len = %d now. Old index claims len = %d\n",
                        i, forest[i].Len, len[old_index]);
                return EXIT_FAILURE;
            }

            if(foflen[old_index] != forest[forest[i].FirstHaloInFOFgroup].Len) {
                fprintf(stderr,"Error: forest[%d].FirstHaloInFOFgroup = %d fofLen = %d now. Old index = %d claims len = %d (nhalos=%d)\n",
                        i, forest[i].FirstHaloInFOFgroup, forest[forest[i].FirstHaloInFOFgroup].Len,
                        old_index,foflen[old_index], nhalos);
                fprintf(stderr,"%d %d %d %d\n",i,forest[i].FirstHaloInFOFgroup,index[i],old_index);
                return EXIT_FAILURE;
            }


            int32_t desc = forest[i].Descendant;
            if(desc == -1) {
                if(desc_len[old_index] != -1){
                    fprintf(stderr,"Error: forest[%d].descendant = %d (should be -1) now but old descendant contained %d particles\n",
                            i, forest[i].Descendant, desc_len[old_index]);
                    return EXIT_FAILURE;
                }
            } else {
                XRETURN(desc >= 0 && desc < nhalos, EXIT_FAILURE,
                        "Error: desc = %d should be in range [0, %d)",
                        desc, nhalos);
                if(desc_len[old_index] != forest[desc].Len) {
                    fprintf(stderr,"Error: forest[%d].Descendant (Len) = %d (desc=%d) now but old descendant contained %d particles\n",
                            i, forest[desc].Len, desc, desc_len[old_index]);
                    return EXIT_FAILURE;
                }
            }


            int32_t prog = forest[i].FirstProgenitor;
            if(prog == -1) {
                if(prog_len[old_index] != -1){
                    fprintf(stderr,"Error: forest[%d].FirstProgenitor = %d (should be -1) now but old FirstProgenitor contained %d particles\n",
                            i, forest[i].FirstProgenitor, desc_len[old_index]);
                    return EXIT_FAILURE;
                }
            } else {
                if( prog < 0 || prog >= nhalos) {
                    fprintf(stderr,"WEIRD: prog = %d for i=%d is not within [0, %d)\n",prog, i, nhalos);
                }
                XRETURN(prog >=0 && prog < nhalos, EXIT_FAILURE,
                        "Error: progenitor index = %d should be in range [0, %d)\n",
                        prog, nhalos);
                if(prog_len[old_index] != forest[prog].Len) {
                    fprintf(stderr,"Error: forest[%d].FirstProgenitor (Len) = %d (prog=%d) now but old FirstProgenitor contained %d particles\n",
                            i, forest[prog].Len, prog, prog_len[old_index]);
                    return EXIT_FAILURE;
                }
            }
        }

        if(forest[0].FirstHaloInFOFgroup != 0) {
            fprintf(stderr,"Error: The first halo should be an FOF halo and point to itself but it points to %d\n", forest[0].FirstHaloInFOFgroup);
            return EXIT_FAILURE;
        }


        int32_t start_fofindex = 0;
        while(start_fofindex < nhalos) {
            int32_t end_fofindex;
            for(end_fofindex=start_fofindex + 1;end_fofindex < nhalos; end_fofindex++) {
                if(forest[end_fofindex].FirstHaloInFOFgroup == end_fofindex) break;
            }

            for(int32_t i=0;i<nhalos;i++) {
                if(forest[i].FirstHaloInFOFgroup == start_fofindex) {
                    if(i >= start_fofindex && i < end_fofindex) {
                        continue;
                    }

                    fprintf(stderr,"Error: Expected FOF to come first and then *all* subhalos associated with that FOF halo\n");
                    fprintf(stderr,"Result truth condition would be for all (FOF+sub) halos to be contained within indices [%d, %d) \n",
                            start_fofindex, end_fofindex);
                    fprintf(stderr,"However, forest[%d].FirstHaloInFOFgroup = %d violates this truth condition\n", i, forest[i].FirstHaloInFOFgroup);
                    return EXIT_FAILURE;
                }
            }
            start_fofindex = end_fofindex;
        }

        free(index_for_old_order);
        free(prog_len); free(desc_len);
        free(len);free(foflen);
    }

    /* halos were re-ordered; return orig_index so callers can map new positions back to input indices */
    *orig_index = index;

    return EXIT_SUCCESS;
}


int fix_mergertree_index(struct halo_data *forest, const int64_t nhalos, const int32_t *index)
{
    if(nhalos > INT_MAX) {
        fprintf(stderr,"Error: nhalos=%"PRId64" can not be larger than INT_MAX=%d\n", nhalos, INT_MAX);
        return INTEGER_32BIT_TOO_SMALL;
    }

    int32_t *current_index_for_old_order = malloc(nhalos * sizeof(*current_index_for_old_order));
    if(current_index_for_old_order == NULL) {
        return MALLOC_FAILURE;
    }


    /* Build the inverse of index[]: index[i] = old position of halo now at i.
       current_index_for_old_order[old] = new position of halo that was at old.
       All index[] values are unique, so this scatter is safe to vectorise. */
    for(int32_t i=0;i<nhalos;i++) {
        current_index_for_old_order[index[i]] = i;
    }

#define UPDATE_LHALOTREE_INDEX(FIELD) {                                 \
        const int32_t ii = this_halo->FIELD;                            \
        if(ii >=0 && ii < nhalos) {                                     \
            const int32_t dst = current_index_for_old_order[ii];        \
            this_halo->FIELD = dst;                                     \
        }                                                               \
    }

    for(int64_t i=0;i<nhalos;i++) {
        struct halo_data *this_halo = &(forest[i]);
        UPDATE_LHALOTREE_INDEX(FirstProgenitor);
        UPDATE_LHALOTREE_INDEX(NextProgenitor);
        UPDATE_LHALOTREE_INDEX(Descendant);
        UPDATE_LHALOTREE_INDEX(FirstHaloInFOFgroup);
        UPDATE_LHALOTREE_INDEX(NextHaloInFOFgroup);
    }
#undef UPDATE_LHALOTREE_INDEX

    free(current_index_for_old_order);
    return EXIT_SUCCESS;
}


int get_nfofs_all_snaps(const struct halo_data *forest, const int nhalos, int *nfofs_all_snaps, const int nsnaps)
{
    for(int i=0;i<nsnaps;i++) {
        nfofs_all_snaps[i] = 0;
    }

    for(int i=0;i<nhalos;i++) {
        if(forest[i].FirstHaloInFOFgroup == i) {
            const int snap = forest[i].SnapNum;
            if(snap < 0 || snap >= nsnaps) {
                fprintf(stderr, "Validation error: snapshot = %d must be within [0, %d)\n", snap, nsnaps);
                return INVALID_MEMORY_ACCESS_REQUESTED;
            }
            nfofs_all_snaps[snap]++;
        }
    }

    return EXIT_SUCCESS;
}

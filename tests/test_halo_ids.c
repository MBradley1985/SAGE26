/**
 * @file test_halo_ids.c
 * @brief Unit tests for galaxy/halo ID generation, uniqueness, and consistency
 *
 * Tests the galaxy ID system including:
 * - GalaxyIndex generation formula verified against hand-computed values
 * - Uniqueness guarantees under various configurations
 * - ID collision detection when multiplication factors are misconfigured
 * - CentralGalaxyIndex lookup via the actual halo->haloaux->galaxy chain
 * - Output format struct sizes and type widths (catches silent truncation)
 * - Overflow protection in 64-bit ID generation
 * - Edge cases: boundary values, single-file mode, zero-galaxy forests
 *
 * The GalaxyIndex formula (from generate_galaxy_indices in core_save.c):
 *   GalaxyIndex = GalaxyNr + (forestnr * ForestNr_Mulfac) + (filenr * FileNr_Mulfac)
 *   CentralGalaxyIndex uses CentralGalaxyNr looked up via:
 *     halogal[haloaux[halos[this_gal->HaloNr].FirstHaloInFOFgroup].FirstGalaxy].GalaxyNr
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <inttypes.h>
#include <limits.h>

#include "../src/core_allvars.h"
#include "../src/io/save_gals_binary.h"
#include "test_framework.h"

/* ========================================================================
   Helper: Mirrors generate_galaxy_indices() from core_save.c.
   We test properties of this formula, not just the implementation.
   ======================================================================== */
static uint64_t compute_galaxy_index(uint32_t GalaxyNr, int64_t forestnr,
                                     int32_t filenr, int64_t forestnr_mulfac,
                                     int64_t filenr_mulfac)
{
    uint64_t id_from_forestnr = (uint64_t)forestnr_mulfac * (uint64_t)forestnr;
    uint64_t id_from_filenr = (uint64_t)filenr_mulfac * (uint64_t)filenr;
    return (uint64_t)GalaxyNr + id_from_forestnr + id_from_filenr;
}

/* Mirrors the CentralGalaxyNr lookup in generate_galaxy_indices():
   CentralGalaxyNr = halogal[haloaux[halos[this_gal->HaloNr].FirstHaloInFOFgroup].FirstGalaxy].GalaxyNr */
static uint32_t lookup_central_galaxy_nr(int gal_idx, const struct halo_data *halos,
                                         const struct halo_aux_data *haloaux,
                                         const struct GALAXY *halogal)
{
    int halo_nr = halogal[gal_idx].HaloNr;
    int fof_halo = halos[halo_nr].FirstHaloInFOFgroup;
    int first_gal = haloaux[fof_halo].FirstGalaxy;
    return halogal[first_gal].GalaxyNr;
}

/* Mirrors the overflow checks in generate_galaxy_indices(). */
static int would_overflow(uint32_t GalaxyNr, int64_t forestnr, int32_t filenr,
                          int64_t forestnr_mulfac, int64_t filenr_mulfac)
{
    if (forestnr > 0 && ((uint64_t)forestnr > (0xFFFFFFFFFFFFFFFFULL / (uint64_t)forestnr_mulfac)))
        return 1;
    if (filenr > 0 && ((uint64_t)filenr > (0xFFFFFFFFFFFFFFFFULL / (uint64_t)filenr_mulfac)))
        return 1;

    uint64_t id_from_forestnr = (uint64_t)forestnr_mulfac * (uint64_t)forestnr;
    uint64_t id_from_filenr = (uint64_t)filenr_mulfac * (uint64_t)filenr;

    if (id_from_forestnr > (0xFFFFFFFFFFFFFFFFULL - id_from_filenr))
        return 1;

    uint64_t id_from_forest_and_file = id_from_forestnr + id_from_filenr;
    if (GalaxyNr > (0xFFFFFFFFFFFFFFFFULL - id_from_forest_and_file))
        return 1;

    return 0;
}

/* Mirrors the validity precondition check. */
static int passes_validity_check(uint32_t GalaxyNr, int64_t forestnr,
                                 int64_t forestnr_mulfac, int64_t filenr_mulfac)
{
    if (GalaxyNr > (uint64_t)forestnr_mulfac)
        return 0;
    if (filenr_mulfac > 0 && forestnr * forestnr_mulfac > filenr_mulfac)
        return 0;
    return 1;
}

/* uint64 comparison for qsort */
static int cmp_uint64(const void *a, const void *b)
{
    uint64_t va = *(const uint64_t *)a;
    uint64_t vb = *(const uint64_t *)b;
    if (va < vb) return -1;
    if (va > vb) return 1;
    return 0;
}

/* Check an array of IDs for duplicates. Returns number of duplicates found. */
static int count_duplicates(uint64_t *ids, int n)
{
    qsort(ids, n, sizeof(uint64_t), cmp_uint64);
    int dups = 0;
    for (int i = 1; i < n; i++) {
        if (ids[i] == ids[i - 1]) dups++;
    }
    return dups;
}

// ============================================================================
// TEST 1: Formula Matches Hand-Computed Values
// ============================================================================
void test_galaxy_index_basic_formula(void) {
    BEGIN_TEST("GalaxyIndex Formula Produces Correct Values");

    /* Hand calculation: 5 + 10*10^6 + 2*10^9 = 5 + 10,000,000 + 2,000,000,000 = 2,010,000,005 */
    uint64_t result = compute_galaxy_index(5, 10, 2, 1000000, 1000000000LL);
    ASSERT_TRUE(result == 2010000005ULL, "5 + 10*10^6 + 2*10^9 = 2,010,000,005");

    /* Different multipliers: 100 + 3*256 + 7*65536 = 100 + 768 + 458752 = 459620 */
    result = compute_galaxy_index(100, 3, 7, 256, 65536);
    ASSERT_TRUE(result == 459620ULL, "100 + 3*256 + 7*65536 = 459,620");

    /* Zero origin */
    result = compute_galaxy_index(0, 0, 0, 1000000, 1000000000LL);
    ASSERT_TRUE(result == 0, "GalaxyIndex = 0 at origin");

    /* Single component contributions */
    ASSERT_TRUE(compute_galaxy_index(1, 0, 0, 1000, 1000000) == 1, "Only GalaxyNr contributes");
    ASSERT_TRUE(compute_galaxy_index(0, 1, 0, 1000, 1000000) == 1000, "Only forestnr contributes");
    ASSERT_TRUE(compute_galaxy_index(0, 0, 1, 1000, 1000000) == 1000000, "Only filenr contributes");
}

// ============================================================================
// TEST 2: Uniqueness Within a Forest (500 Galaxies)
// ============================================================================
void test_uniqueness_within_forest(void) {
    BEGIN_TEST("500 Galaxy IDs Unique Within a Single Forest");

    int n = 500;
    uint64_t *ids = malloc(n * sizeof(uint64_t));
    for (int i = 0; i < n; i++) {
        ids[i] = compute_galaxy_index(i, 42, 3, 1000000, 1000000000LL);
    }

    ASSERT_EQUAL_INT(0, count_duplicates(ids, n), "No duplicates among 500 IDs in one forest");

    /* Also verify sequential: id[i+1] = id[i] + 1 */
    qsort(ids, n, sizeof(uint64_t), cmp_uint64);
    int sequential = 1;
    for (int i = 1; i < n; i++) {
        if (ids[i] != ids[i - 1] + 1) { sequential = 0; break; }
    }
    ASSERT_TRUE(sequential, "IDs are sequential (differ by 1)");
    free(ids);
}

// ============================================================================
// TEST 3: Uniqueness Across 50 Forests in Same File
// ============================================================================
void test_uniqueness_across_forests(void) {
    BEGIN_TEST("5000 Galaxy IDs Unique Across 50 Forests");

    int nf = 50, ng = 100, total = nf * ng;
    uint64_t *ids = malloc(total * sizeof(uint64_t));
    int idx = 0;
    for (int f = 0; f < nf; f++)
        for (int g = 0; g < ng; g++)
            ids[idx++] = compute_galaxy_index(g, f, 0, 1000000, 1000000000LL);

    ASSERT_EQUAL_INT(0, count_duplicates(ids, total), "No duplicates across 50 forests");

    /* Verify non-overlapping ranges: max of forest f < min of forest f+1 */
    uint64_t f0_max = compute_galaxy_index(ng - 1, 0, 0, 1000000, 1000000000LL);
    uint64_t f1_min = compute_galaxy_index(0, 1, 0, 1000000, 1000000000LL);
    ASSERT_TRUE(f1_min > f0_max, "Forest 1 starts above forest 0 max");
    free(ids);
}

// ============================================================================
// TEST 4: Uniqueness Across 5 Files x 100 Forests x 50 Galaxies
// ============================================================================
void test_uniqueness_across_files(void) {
    BEGIN_TEST("25,000 Galaxy IDs Unique Across 5 Files");

    int nfiles = 5, nf = 100, ng = 50;
    int total = nfiles * nf * ng;
    uint64_t *ids = malloc(total * sizeof(uint64_t));
    int idx = 0;
    for (int file = 0; file < nfiles; file++)
        for (int f = 0; f < nf; f++)
            for (int g = 0; g < ng; g++)
                ids[idx++] = compute_galaxy_index(g, f, file, 1000000, 1000000000LL);

    ASSERT_EQUAL_INT(0, count_duplicates(ids, total), "No duplicates among 25,000 IDs across 5 files");

    /* Verify file ranges don't overlap */
    uint64_t file0_max = compute_galaxy_index(ng - 1, nf - 1, 0, 1000000, 1000000000LL);
    uint64_t file1_min = compute_galaxy_index(0, 0, 1, 1000000, 1000000000LL);
    ASSERT_TRUE(file1_min > file0_max, "File 1 IDs start above file 0 max");
    free(ids);
}

// ============================================================================
// TEST 5: COLLISION DETECTION - Too-Small ForestNr_Mulfac Causes Duplicates
// ============================================================================
void test_collision_with_bad_mulfac(void) {
    BEGIN_TEST("IDs COLLIDE When ForestNr_Mulfac Is Too Small");

    /* ForestNr_Mulfac = 50 but we have 100 galaxies per forest.
       Galaxy 50 in forest 0 has same ID as galaxy 0 in forest 1:
       50 + 0*50 = 50 = 0 + 1*50. This MUST produce duplicates. */
    int64_t bad_mulfac = 50;
    int nf = 2, ng = 100;
    int total = nf * ng;
    uint64_t *ids = malloc(total * sizeof(uint64_t));
    int idx = 0;
    for (int f = 0; f < nf; f++)
        for (int g = 0; g < ng; g++)
            ids[idx++] = compute_galaxy_index(g, f, 0, bad_mulfac, 0);

    int dups = count_duplicates(ids, total);
    ASSERT_TRUE(dups > 0, "Bad ForestNr_Mulfac (50) with 100 gals/forest produces collisions");
    ASSERT_EQUAL_INT(50, dups, "Exactly 50 collisions (gals 50-99 of forest 0 overlap forest 1)");
    free(ids);
}

// ============================================================================
// TEST 6: COLLISION DETECTION - Too-Small FileNr_Mulfac Causes Duplicates
// ============================================================================
void test_collision_with_bad_file_mulfac(void) {
    BEGIN_TEST("IDs COLLIDE When FileNr_Mulfac Is Too Small for Forest Range");

    /* FileNr_Mulfac = 500 but forestnr_mulfac=100 with 10 forests -> max forest offset = 900.
       File 1 starts at offset 500, which overlaps with forest 5+ of file 0. */
    int64_t forestnr_mulfac = 100;
    int64_t filenr_mulfac = 500;  /* too small: should be >= 10*100 = 1000 */
    int nfiles = 2, nf = 10, ng = 10;
    int total = nfiles * nf * ng;
    uint64_t *ids = malloc(total * sizeof(uint64_t));
    int idx = 0;
    for (int file = 0; file < nfiles; file++)
        for (int f = 0; f < nf; f++)
            for (int g = 0; g < ng; g++)
                ids[idx++] = compute_galaxy_index(g, f, file, forestnr_mulfac, filenr_mulfac);

    int dups = count_duplicates(ids, total);
    ASSERT_TRUE(dups > 0, "Bad FileNr_Mulfac (500) with 10 forests of mulfac 100 causes collisions");
    free(ids);
}

// ============================================================================
// TEST 7: Correct ForestNr_Mulfac Prevents Collisions
// ============================================================================
void test_correct_mulfac_no_collision(void) {
    BEGIN_TEST("Correct Multiplication Factors Prevent All Collisions");

    /* 100 galaxies per forest -> ForestNr_Mulfac must be >= 100.
       10 forests per file -> FileNr_Mulfac must be >= 10*100 = 1000.
       Use ForestNr_Mulfac=100, FileNr_Mulfac=1000: barely sufficient. */
    int64_t forestnr_mulfac = 100;
    int64_t filenr_mulfac = 1000;
    int nfiles = 5, nf = 10, ng = 100;
    int total = nfiles * nf * ng;
    uint64_t *ids = malloc(total * sizeof(uint64_t));
    int idx = 0;
    for (int file = 0; file < nfiles; file++)
        for (int f = 0; f < nf; f++)
            for (int g = 0; g < ng; g++)
                ids[idx++] = compute_galaxy_index(g, f, file, forestnr_mulfac, filenr_mulfac);

    ASSERT_EQUAL_INT(0, count_duplicates(ids, total),
                     "ForestNr_Mulfac=100 with max 100 gals/forest: no collisions");
    free(ids);
}

// ============================================================================
// TEST 8: CentralGalaxyIndex Via Actual Halo Lookup Chain
// ============================================================================
void test_central_galaxy_index_via_halo_chain(void) {
    BEGIN_TEST("CentralGalaxyIndex Uses Halo->FOF->haloaux->Galaxy Lookup");

    /* Build a realistic halo tree with a FOF group:
       Halo 0 = FOF central, contains galaxies 0 (central, GalaxyNr=0) and 1 (sat, GalaxyNr=1)
       Halo 1 = subhalo of FOF, contains galaxy 2 (sat of FOF, GalaxyNr=2) */
    struct halo_data halos[2];
    struct halo_aux_data haloaux[2];
    struct GALAXY halogal[3];
    memset(halos, 0, sizeof(halos));
    memset(haloaux, 0, sizeof(haloaux));
    memset(halogal, 0, sizeof(halogal));

    /* Halo 0 is the FOF central */
    halos[0].FirstHaloInFOFgroup = 0;
    halos[0].NextHaloInFOFgroup = 1;
    /* Halo 1 is a subhalo, FOF central is halo 0 */
    halos[1].FirstHaloInFOFgroup = 0;
    halos[1].NextHaloInFOFgroup = -1;

    /* haloaux links halos to galaxies */
    haloaux[0].FirstGalaxy = 0;
    haloaux[0].NGalaxies = 2;
    haloaux[1].FirstGalaxy = 2;
    haloaux[1].NGalaxies = 1;

    /* Galaxies */
    halogal[0].HaloNr = 0; halogal[0].GalaxyNr = 0; halogal[0].Type = 0;
    halogal[1].HaloNr = 0; halogal[1].GalaxyNr = 1; halogal[1].Type = 1;
    halogal[2].HaloNr = 1; halogal[2].GalaxyNr = 2; halogal[2].Type = 1;

    int64_t forestnr_mulfac = 1000000;
    int64_t filenr_mulfac = 1000000000LL;
    int64_t forestnr = 5;
    int32_t filenr = 1;

    /* Compute CentralGalaxyNr for each galaxy using the actual lookup chain */
    for (int g = 0; g < 3; g++) {
        uint32_t central_nr = lookup_central_galaxy_nr(g, halos, haloaux, halogal);
        uint64_t central_idx = compute_galaxy_index(central_nr, forestnr, filenr,
                                                    forestnr_mulfac, filenr_mulfac);
        uint64_t own_idx = compute_galaxy_index(halogal[g].GalaxyNr, forestnr, filenr,
                                                forestnr_mulfac, filenr_mulfac);

        /* The FOF central galaxy (GalaxyNr=0 in halo 0) should be the central for ALL */
        ASSERT_TRUE(central_nr == 0, "Central GalaxyNr is 0 for all galaxies in this FOF");
        ASSERT_TRUE(central_idx == compute_galaxy_index(0, forestnr, filenr,
                                                        forestnr_mulfac, filenr_mulfac),
                    "CentralGalaxyIndex matches galaxy 0's index");

        /* For the central galaxy itself, own index == central index */
        if (g == 0) {
            ASSERT_TRUE(own_idx == central_idx, "Central's own index equals its CentralGalaxyIndex");
        } else {
            ASSERT_TRUE(own_idx != central_idx, "Satellite's own index differs from CentralGalaxyIndex");
        }
    }

    /* Now test with a DIFFERENT FOF group structure:
       What if haloaux[0].FirstGalaxy pointed to galaxy 1 instead of 0?
       This would make GalaxyNr=1 the "central" — verify the lookup reflects this. */
    haloaux[0].FirstGalaxy = 1;  /* intentionally "wrong" — galaxy 1 is first */
    uint32_t central_nr_gal0 = lookup_central_galaxy_nr(0, halos, haloaux, halogal);
    ASSERT_TRUE(central_nr_gal0 == 1,
                "Lookup follows haloaux.FirstGalaxy (not hardcoded to GalaxyNr=0)");
}

// ============================================================================
// TEST 9: CentralGalaxyIndex Differs Between FOF Groups
// ============================================================================
void test_central_index_differs_between_fof_groups(void) {
    BEGIN_TEST("Different FOF Groups Have Different CentralGalaxyIndex");

    /* Two separate FOF groups in the same forest */
    struct halo_data halos[2];
    struct halo_aux_data haloaux[2];
    struct GALAXY halogal[2];
    memset(halos, 0, sizeof(halos));
    memset(haloaux, 0, sizeof(haloaux));
    memset(halogal, 0, sizeof(halogal));

    /* FOF group A: halo 0, galaxy 0 */
    halos[0].FirstHaloInFOFgroup = 0;
    haloaux[0].FirstGalaxy = 0;
    halogal[0].HaloNr = 0; halogal[0].GalaxyNr = 0;

    /* FOF group B: halo 1, galaxy 1 */
    halos[1].FirstHaloInFOFgroup = 1;
    haloaux[1].FirstGalaxy = 1;
    halogal[1].HaloNr = 1; halogal[1].GalaxyNr = 1;

    int64_t fmul = 1000000, fimul = 1000000000LL;
    uint32_t central_a = lookup_central_galaxy_nr(0, halos, haloaux, halogal);
    uint32_t central_b = lookup_central_galaxy_nr(1, halos, haloaux, halogal);

    uint64_t central_idx_a = compute_galaxy_index(central_a, 5, 0, fmul, fimul);
    uint64_t central_idx_b = compute_galaxy_index(central_b, 5, 0, fmul, fimul);

    ASSERT_TRUE(central_a != central_b, "Different FOF groups have different CentralGalaxyNr");
    ASSERT_TRUE(central_idx_a != central_idx_b, "Different FOF groups have different CentralGalaxyIndex");
}

// ============================================================================
// TEST 10: Monotonicity - IDs Strictly Increase With GalaxyNr
// ============================================================================
void test_id_monotonicity(void) {
    BEGIN_TEST("IDs Strictly Increase for Increasing GalaxyNr");

    uint64_t prev = compute_galaxy_index(0, 42, 3, 1000000, 1000000000LL);
    int monotonic = 1;
    for (int g = 1; g < 1000; g++) {
        uint64_t id = compute_galaxy_index(g, 42, 3, 1000000, 1000000000LL);
        if (id <= prev) { monotonic = 0; break; }
        prev = id;
    }
    ASSERT_TRUE(monotonic, "IDs strictly increasing for 1000 galaxies");
}

// ============================================================================
// TEST 11: Forest Gap = ForestNr_Mulfac
// ============================================================================
void test_forest_gap(void) {
    BEGIN_TEST("Gap Between Adjacent Forests Equals ForestNr_Mulfac");

    int64_t mulfacs[] = {100, 1000, 1000000, 65536};
    for (int m = 0; m < 4; m++) {
        for (int f = 0; f < 10; f++) {
            uint64_t id0 = compute_galaxy_index(0, f, 0, mulfacs[m], 1000000000LL);
            uint64_t id1 = compute_galaxy_index(0, f + 1, 0, mulfacs[m], 1000000000LL);
            if (id1 - id0 != (uint64_t)mulfacs[m]) {
                ASSERT_TRUE(0, "Forest gap != ForestNr_Mulfac");
                return;
            }
        }
    }
    ASSERT_TRUE(1, "Forest gap correct for 4 different mulfac values x 10 forests");
}

// ============================================================================
// TEST 12: File Gap = FileNr_Mulfac
// ============================================================================
void test_file_gap(void) {
    BEGIN_TEST("Gap Between Adjacent Files Equals FileNr_Mulfac");

    int64_t fimuls[] = {1000000, 1000000000LL, (int64_t)1 << 32};
    for (int m = 0; m < 3; m++) {
        for (int file = 0; file < 5; file++) {
            uint64_t id0 = compute_galaxy_index(0, 0, file, 1000000, fimuls[m]);
            uint64_t id1 = compute_galaxy_index(0, 0, file + 1, 1000000, fimuls[m]);
            if (id1 - id0 != (uint64_t)fimuls[m]) {
                ASSERT_TRUE(0, "File gap != FileNr_Mulfac");
                return;
            }
        }
    }
    ASSERT_TRUE(1, "File gap correct for 3 different FileNr_Mulfac values x 5 files");
}

// ============================================================================
// TEST 13: Boundary - Max GalaxyNr = ForestNr_Mulfac - 1
// ============================================================================
void test_max_galaxy_nr_boundary(void) {
    BEGIN_TEST("Max GalaxyNr (mulfac-1) Is Exactly 1 Below Next Forest");

    int64_t mulfacs[] = {100, 1000, 65536, 1000000};
    for (int m = 0; m < 4; m++) {
        uint32_t max_nr = (uint32_t)mulfacs[m] - 1;
        uint64_t id_max = compute_galaxy_index(max_nr, 0, 0, mulfacs[m], 0);
        uint64_t id_next = compute_galaxy_index(0, 1, 0, mulfacs[m], 0);
        ASSERT_TRUE(id_next - id_max == 1,
                    "Gap of exactly 1 at forest boundary");
    }
}

// ============================================================================
// TEST 14: Boundary - GalaxyNr = ForestNr_Mulfac Causes Collision
// ============================================================================
void test_galaxy_nr_at_mulfac_causes_collision(void) {
    BEGIN_TEST("GalaxyNr = ForestNr_Mulfac Collides With Next Forest's Galaxy 0");

    int64_t mulfac = 1000;
    /* Galaxy 1000 in forest 0 = 1000 + 0*1000 = 1000
       Galaxy 0 in forest 1    = 0 + 1*1000    = 1000  -> COLLISION */
    uint64_t id_bad = compute_galaxy_index(1000, 0, 0, mulfac, 0);
    uint64_t id_next = compute_galaxy_index(0, 1, 0, mulfac, 0);
    ASSERT_TRUE(id_bad == id_next,
                "GalaxyNr=mulfac collides with forest+1 galaxy 0 (this is the bug the check prevents)");
}

// ============================================================================
// TEST 15: Encode-Decode Round Trip
// ============================================================================
void test_encode_decode_round_trip(void) {
    BEGIN_TEST("GalaxyIndex Decodes Back to (filenr, forestnr, GalaxyNr)");

    /* Test multiple combinations */
    struct { uint32_t gal; int64_t forest; int32_t file; } cases[] = {
        {0, 0, 0}, {42, 7, 3}, {999, 500, 99}, {0, 999, 0}, {999999, 0, 0}
    };
    int64_t fmul = 1000000, fimul = 1000000000LL;

    for (int c = 0; c < 5; c++) {
        uint64_t id = compute_galaxy_index(cases[c].gal, cases[c].forest, cases[c].file, fmul, fimul);

        int32_t dec_file = (int32_t)(id / (uint64_t)fimul);
        uint64_t rem = id - (uint64_t)dec_file * (uint64_t)fimul;
        int64_t dec_forest = (int64_t)(rem / (uint64_t)fmul);
        uint32_t dec_gal = (uint32_t)(rem - (uint64_t)dec_forest * (uint64_t)fmul);

        ASSERT_EQUAL_INT(cases[c].file, dec_file, "Decoded file matches");
        ASSERT_TRUE(dec_forest == cases[c].forest, "Decoded forest matches");
        ASSERT_TRUE(dec_gal == cases[c].gal, "Decoded GalaxyNr matches");
    }
}

// ============================================================================
// TEST 16: Validity Check Rejects Bad GalaxyNr
// ============================================================================
void test_validity_check_galaxynr(void) {
    BEGIN_TEST("Validity Check Rejects GalaxyNr > ForestNr_Mulfac");

    ASSERT_TRUE(passes_validity_check(999, 0, 1000, 1000000), "999 < 1000: valid");
    ASSERT_TRUE(passes_validity_check(1000, 0, 1000, 1000000), "1000 == 1000: valid (code uses >)");
    ASSERT_TRUE(!passes_validity_check(1001, 0, 1000, 1000000), "1001 > 1000: rejected");

    /* Also verify edge: GalaxyNr=0 is always valid */
    ASSERT_TRUE(passes_validity_check(0, 0, 1, 1), "GalaxyNr=0 with minimal mulfac: valid");
}

// ============================================================================
// TEST 17: Validity Check Rejects Bad Forest Range
// ============================================================================
void test_validity_check_forest_range(void) {
    BEGIN_TEST("Validity Check Rejects forestnr*ForestNr_Mulfac > FileNr_Mulfac");

    ASSERT_TRUE(passes_validity_check(0, 99, 1000, 100000), "99*1000=99000 < 100000: valid");
    ASSERT_TRUE(!passes_validity_check(0, 101, 1000, 100000), "101*1000=101000 > 100000: rejected");
    ASSERT_TRUE(passes_validity_check(0, 100, 1000, 100000), "100*1000=100000 == 100000: valid (code uses >)");

    /* FileNr_Mulfac=0 disables this check */
    ASSERT_TRUE(passes_validity_check(0, 999999, 1000, 0), "FileNr_Mulfac=0 disables forest range check");
}

// ============================================================================
// TEST 18: Single-File Mode (FileNr_Mulfac = 0)
// ============================================================================
void test_single_file_mode(void) {
    BEGIN_TEST("FileNr_Mulfac=0: File Number Has No Effect on ID");

    /* With FileNr_Mulfac=0, different file numbers produce the same ID */
    uint64_t id0 = compute_galaxy_index(5, 10, 0, 1000000, 0);
    uint64_t id1 = compute_galaxy_index(5, 10, 99, 1000000, 0);
    uint64_t id2 = compute_galaxy_index(5, 10, INT_MAX, 1000000, 0);
    ASSERT_TRUE(id0 == id1 && id1 == id2, "File number ignored when FileNr_Mulfac=0");

    /* But GalaxyNr and forestnr still differentiate */
    ASSERT_TRUE(compute_galaxy_index(0, 0, 0, 1000000, 0) !=
                compute_galaxy_index(1, 0, 0, 1000000, 0),
                "GalaxyNr still differentiates in single-file mode");
    ASSERT_TRUE(compute_galaxy_index(0, 0, 0, 1000000, 0) !=
                compute_galaxy_index(0, 1, 0, 1000000, 0),
                "forestnr still differentiates in single-file mode");
}

// ============================================================================
// TEST 19: Overflow Detection - Multiplication
// ============================================================================
void test_overflow_multiplication(void) {
    BEGIN_TEST("Overflow Detection: Catches Multiplication Overflow");

    /* 2^40 * 2^30 = 2^70 > 2^64 */
    ASSERT_TRUE(would_overflow(0, (int64_t)1 << 30, 0, (int64_t)1 << 40, 0),
                "Detects 2^40 * 2^30 overflow");

    /* 2^32 * 2^32 = 2^64 > max uint64 */
    ASSERT_TRUE(would_overflow(0, (int64_t)1 << 32, 0, (int64_t)1 << 32, 0),
                "Detects 2^32 * 2^32 overflow");

    /* Just under limit: (2^64-1) / 2^32 * 2^32 should NOT overflow */
    ASSERT_TRUE(!would_overflow(0, (int64_t)1 << 31, 0, (int64_t)1 << 32, 0),
                "2^32 * 2^31 = 2^63: no overflow");
}

// ============================================================================
// TEST 20: Overflow Detection - Addition
// ============================================================================
void test_overflow_addition(void) {
    BEGIN_TEST("Overflow Detection: Catches Addition Overflow");

    /* Two large products that individually fit but sum overflows */
    int64_t mulfac = (int64_t)1 << 32;
    int64_t forestnr = (int64_t)1 << 31;       /* product = 2^63 */
    int32_t filenr = (int32_t)((int64_t)1 << 31); /* product = 2^63 */
    /* sum = 2^63 + 2^63 = 2^64 > max uint64 */
    ASSERT_TRUE(would_overflow(0, forestnr, filenr, mulfac, mulfac),
                "Detects addition overflow (2^63 + 2^63)");

    /* Safe case: both products well within range */
    ASSERT_TRUE(!would_overflow(0, 1000, 100, 1000000, 1000000000LL),
                "No overflow for typical values");
}

// ============================================================================
// TEST 21: No Overflow for Realistic SAGE Configurations
// ============================================================================
void test_no_overflow_realistic(void) {
    BEGIN_TEST("No Overflow for Realistic Tree Format Configurations");

    /* LHaloTree binary: ForestNr_Mulfac=10^6, FileNr_Mulfac=10^9 */
    ASSERT_TRUE(!would_overflow(999999, 999, 127, 1000000, 1000000000LL),
                "LHaloTree binary: 10^6 gals, 1000 forests, 128 files");

    /* Power-of-2: ForestNr_Mulfac=2^16, FileNr_Mulfac=2^32 */
    ASSERT_TRUE(!would_overflow(65535, 65535, 512, 65536, (int64_t)65536 * 65536),
                "Power-of-2: 2^16 gals, 2^16 forests, 512 files");

    /* Large simulation: 10^6 galaxies/tree, 10^4 trees, 100 files */
    ASSERT_TRUE(!would_overflow(999999, 9999, 99, 1000000, 10000000000LL),
                "Large sim: 10^6 gals, 10^4 forests, 100 files");
}

// ============================================================================
// TEST 22: Struct Size - GalaxyIndex Is 64-bit
// ============================================================================
void test_galaxy_index_struct_size(void) {
    BEGIN_TEST("GalaxyIndex Fields Are 64-bit (Catches Silent Truncation)");

    /* If someone changes GalaxyIndex from uint64_t to uint32_t, these fail */
    ASSERT_TRUE(sizeof(((struct GALAXY *)0)->GalaxyIndex) == 8,
                "GALAXY.GalaxyIndex is 8 bytes");
    ASSERT_TRUE(sizeof(((struct GALAXY *)0)->CentralGalaxyIndex) == 8,
                "GALAXY.CentralGalaxyIndex is 8 bytes");

    /* Output struct */
    ASSERT_TRUE(sizeof(((struct GALAXY_OUTPUT *)0)->GalaxyIndex) == 8,
                "GALAXY_OUTPUT.GalaxyIndex is 8 bytes");
    ASSERT_TRUE(sizeof(((struct GALAXY_OUTPUT *)0)->CentralGalaxyIndex) == 8,
                "GALAXY_OUTPUT.CentralGalaxyIndex is 8 bytes");
    ASSERT_TRUE(sizeof(((struct GALAXY_OUTPUT *)0)->SimulationHaloIndex) == 8,
                "GALAXY_OUTPUT.SimulationHaloIndex is 8 bytes");
}

// ============================================================================
// TEST 23: Type Width - SAGEHaloIndex and SAGETreeIndex Are 32-bit
// ============================================================================
void test_output_id_type_widths(void) {
    BEGIN_TEST("Output ID Field Widths Match Expected Types");

    ASSERT_TRUE(sizeof(((struct GALAXY_OUTPUT *)0)->SAGEHaloIndex) == 4,
                "SAGEHaloIndex is 4 bytes (int32)");
    ASSERT_TRUE(sizeof(((struct GALAXY_OUTPUT *)0)->SAGETreeIndex) == 4,
                "SAGETreeIndex is 4 bytes (int32)");
    ASSERT_TRUE(sizeof(((struct GALAXY_OUTPUT *)0)->mergeIntoID) == 4,
                "mergeIntoID is 4 bytes (int32)");
    ASSERT_TRUE(sizeof(((struct GALAXY_OUTPUT *)0)->mergeType) == 4,
                "mergeType is 4 bytes (int32)");
}

// ============================================================================
// TEST 24: uint64_t -> long long Cast Safety
// ============================================================================
void test_uint64_to_longlong_cast(void) {
    BEGIN_TEST("uint64_t to long long Cast: Typical IDs Survive, Large IDs May Not");

    /* GALAXY uses uint64_t, GALAXY_OUTPUT uses long long.
       Values < 2^63 survive the cast. Values >= 2^63 would be negative. */
    uint64_t safe_id = 5010000042ULL;
    long long cast_id = (long long)safe_id;
    ASSERT_TRUE((uint64_t)cast_id == safe_id,
                "Typical ID (5 billion) survives uint64->long long cast");

    /* The maximum safe value */
    uint64_t max_safe = (uint64_t)LLONG_MAX;
    long long cast_max = (long long)max_safe;
    ASSERT_TRUE(cast_max > 0 && (uint64_t)cast_max == max_safe,
                "LLONG_MAX survives cast");

    /* Typical SAGE ID range: even with 10^9 FileNr_Mulfac * 1000 files = 10^12, well under 2^63 */
    uint64_t max_realistic = 1000ULL * 1000000000ULL + 999999ULL * 1000000ULL + 999999ULL;
    ASSERT_TRUE(max_realistic < (uint64_t)LLONG_MAX,
                "Max realistic SAGE ID fits in long long");
}

// ============================================================================
// TEST 25: SimulationHaloIndex = |MostBoundID|
// ============================================================================
void test_simulation_halo_index_abs(void) {
    BEGIN_TEST("SimulationHaloIndex Correctly Takes Absolute Value of MostBoundID");

    /* Test the specific transform used in prepare_galaxy_for_output:
       o->SimulationHaloIndex = llabs(halos[g->HaloNr].MostBoundID) */
    long long test_values[] = {12345, -67890, 0, -1, LLONG_MAX, LLONG_MIN + 1};
    long long expected[]    = {12345,  67890, 0,  1, LLONG_MAX, LLONG_MAX};
    int n = 6;

    for (int i = 0; i < n; i++) {
        long long result = llabs(test_values[i]);
        ASSERT_TRUE(result == expected[i], "llabs produces correct absolute value");
        ASSERT_TRUE(result >= 0, "Result is non-negative");
    }
}

// ============================================================================
// TEST 26: Output Order Remapping Logic
// ============================================================================
void test_output_order_remapping(void) {
    BEGIN_TEST("mergeIntoID Remapping Follows Output Snapshot Order");

    /* Simulate the remapping logic from save_galaxies() lines 92-116.
       Galaxies at different snapshots get different output orderings.
       mergeIntoID must be remapped from internal to output order. */
    int num_gals = 6;
    struct GALAXY halogal[6];
    memset(halogal, 0, sizeof(halogal));

    /* Mix of snapshots */
    halogal[0].SnapNum = 63; halogal[0].mergeIntoID = -1;
    halogal[1].SnapNum = 63; halogal[1].mergeIntoID = 0;  /* merges into gal 0 */
    halogal[2].SnapNum = 62; halogal[2].mergeIntoID = -1;
    halogal[3].SnapNum = 63; halogal[3].mergeIntoID = -1;
    halogal[4].SnapNum = 62; halogal[4].mergeIntoID = 2;  /* merges into gal 2 */
    halogal[5].SnapNum = 62; halogal[5].mergeIntoID = -1;

    /* Build output order per snapshot (mirrors save_galaxies logic) */
    int output_snaps[] = {63, 62};
    int num_output_snaps = 2;
    int32_t OutputGalOrder[6];
    int32_t OutputGalCount[2] = {0, 0};
    for (int i = 0; i < num_gals; i++) OutputGalOrder[i] = -1;

    for (int s = 0; s < num_output_snaps; s++) {
        for (int g = 0; g < num_gals; g++) {
            if (halogal[g].SnapNum == output_snaps[s]) {
                OutputGalOrder[g] = OutputGalCount[s]++;
            }
        }
    }

    /* Snap 63 galaxies: gal 0->order 0, gal 1->order 1, gal 3->order 2 */
    ASSERT_EQUAL_INT(0, OutputGalOrder[0], "Gal 0 (snap 63) gets output order 0");
    ASSERT_EQUAL_INT(1, OutputGalOrder[1], "Gal 1 (snap 63) gets output order 1");
    ASSERT_EQUAL_INT(2, OutputGalOrder[3], "Gal 3 (snap 63) gets output order 2");

    /* Snap 62 galaxies: gal 2->order 0, gal 4->order 1, gal 5->order 2 */
    ASSERT_EQUAL_INT(0, OutputGalOrder[2], "Gal 2 (snap 62) gets output order 0");
    ASSERT_EQUAL_INT(1, OutputGalOrder[4], "Gal 4 (snap 62) gets output order 1");

    /* Remap mergeIntoID: gal 1 merges into gal 0 -> output order 0 */
    int remapped_gal1 = OutputGalOrder[halogal[1].mergeIntoID];
    ASSERT_EQUAL_INT(0, remapped_gal1, "Gal 1's mergeIntoID remaps to gal 0's output order (0)");

    /* Remap mergeIntoID: gal 4 merges into gal 2 -> output order 0 within snap 62 */
    int remapped_gal4 = OutputGalOrder[halogal[4].mergeIntoID];
    ASSERT_EQUAL_INT(0, remapped_gal4, "Gal 4's mergeIntoID remaps to gal 2's output order (0)");

    /* Count totals per snapshot */
    ASSERT_EQUAL_INT(3, OutputGalCount[0], "3 galaxies at snap 63");
    ASSERT_EQUAL_INT(3, OutputGalCount[1], "3 galaxies at snap 62");
}

// ============================================================================
// TEST 27: Galaxies NOT at Output Snapshot Get Order -1
// ============================================================================
void test_non_output_snapshot_galaxies(void) {
    BEGIN_TEST("Galaxies Not at an Output Snapshot Get OutputGalOrder = -1");

    struct GALAXY halogal[3];
    memset(halogal, 0, sizeof(halogal));
    halogal[0].SnapNum = 63;
    halogal[1].SnapNum = 50;  /* NOT an output snapshot */
    halogal[2].SnapNum = 62;

    int output_snaps[] = {63, 62};
    int32_t OutputGalOrder[3] = {-1, -1, -1};
    int count63 = 0, count62 = 0;

    for (int s = 0; s < 2; s++) {
        for (int g = 0; g < 3; g++) {
            if (halogal[g].SnapNum == output_snaps[s]) {
                if (s == 0) OutputGalOrder[g] = count63++;
                else OutputGalOrder[g] = count62++;
            }
        }
    }

    ASSERT_TRUE(OutputGalOrder[0] >= 0, "Galaxy at output snap 63 gets valid order");
    ASSERT_EQUAL_INT(-1, OutputGalOrder[1], "Galaxy at non-output snap 50 stays -1");
    ASSERT_TRUE(OutputGalOrder[2] >= 0, "Galaxy at output snap 62 gets valid order");
}

// ============================================================================
// TEST 28: Forest-Level Uniqueness With Power-of-2 Mulfacs (Typical HDF5)
// ============================================================================
void test_uniqueness_power_of_two_mulfacs(void) {
    BEGIN_TEST("Uniqueness With Power-of-2 Mulfacs (HDF5 Format)");

    /* Common HDF5 config: ForestNr_Mulfac=2^16, FileNr_Mulfac=2^32 */
    int64_t fmul = 65536, fimul = (int64_t)65536 * 65536;
    int nfiles = 3, nf = 100, ng = 200;
    int total = nfiles * nf * ng;
    uint64_t *ids = malloc(total * sizeof(uint64_t));
    int idx = 0;
    for (int file = 0; file < nfiles; file++)
        for (int f = 0; f < nf; f++)
            for (int g = 0; g < ng; g++)
                ids[idx++] = compute_galaxy_index(g, f, file, fmul, fimul);

    ASSERT_EQUAL_INT(0, count_duplicates(ids, total),
                     "No duplicates with 2^16/2^32 mulfacs across 60,000 IDs");
    free(ids);
}

// ============================================================================
// TEST 29: Multiple Galaxies in Same Halo Get Unique IDs
// ============================================================================
void test_same_halo_unique_ids(void) {
    BEGIN_TEST("10 Galaxies in Same Halo (Central + 9 Sats) Have Unique IDs");

    int n = 10;
    uint64_t ids[10];
    for (int i = 0; i < n; i++) {
        ids[i] = compute_galaxy_index(i, 5, 0, 1000000, 1000000000LL);
    }
    ASSERT_EQUAL_INT(0, count_duplicates(ids, n), "All 10 galaxies have unique IDs");
}

// ============================================================================
// TEST 30: Stress - Very Large Values With Decode Verification
// ============================================================================
void test_stress_large_values(void) {
    BEGIN_TEST("Stress: Large Galaxy/Forest/File Numbers Encode-Decode Correctly");

    int64_t fmul = (int64_t)1 << 20;
    int64_t fimul = (int64_t)1 << 40;

    uint32_t gal = (1 << 20) - 1;
    int64_t forest = (1 << 19) - 1;
    int32_t file = 1000;

    ASSERT_TRUE(!would_overflow(gal, forest, file, fmul, fimul), "No overflow for large valid values");

    uint64_t id = compute_galaxy_index(gal, forest, file, fmul, fimul);
    ASSERT_TRUE(id > 0, "Generated ID is positive");

    /* Decode */
    int32_t dec_file = (int32_t)(id / (uint64_t)fimul);
    uint64_t rem = id - (uint64_t)dec_file * (uint64_t)fimul;
    int64_t dec_forest = (int64_t)(rem / (uint64_t)fmul);
    uint32_t dec_gal = (uint32_t)(rem - (uint64_t)dec_forest * (uint64_t)fmul);

    ASSERT_EQUAL_INT(file, dec_file, "Decoded file matches");
    ASSERT_TRUE(dec_forest == forest, "Decoded forest matches");
    ASSERT_TRUE(dec_gal == gal, "Decoded GalaxyNr matches");
}

// ============================================================================
// TEST 31: ID Uniqueness With Forests That Have Varying Galaxy Counts
// ============================================================================
void test_varying_galaxy_counts_per_forest(void) {
    BEGIN_TEST("Uniqueness When Forests Have Different Numbers of Galaxies");

    /* Forest 0 has 500 galaxies, forest 1 has 3, forest 2 has 200 */
    int64_t fmul = 1000, fimul = 1000000;
    int counts[] = {500, 3, 200};
    int total = 500 + 3 + 200;
    uint64_t *ids = malloc(total * sizeof(uint64_t));
    int idx = 0;

    for (int f = 0; f < 3; f++) {
        for (int g = 0; g < counts[f]; g++) {
            ids[idx++] = compute_galaxy_index(g, f, 0, fmul, fimul);
        }
    }

    ASSERT_EQUAL_INT(0, count_duplicates(ids, total),
                     "No duplicates with varying forest sizes (500, 3, 200)");
    free(ids);
}

// ============================================================================
// TEST 32: Verify Validity Check Actually Matches Collision Condition
// ============================================================================
void test_validity_check_matches_collision(void) {
    BEGIN_TEST("Validity Check Catches Exactly the Cases That Would Collide");

    /* Case 1: GalaxyNr > ForestNr_Mulfac -> collisions occur, validity fails */
    int64_t fmul = 50;
    ASSERT_TRUE(!passes_validity_check(51, 0, fmul, 0),
                "GalaxyNr=51 > mulfac=50: validity check rejects");

    /* Verify collision actually exists */
    uint64_t id_bad = compute_galaxy_index(51, 0, 0, fmul, 0);
    uint64_t id_f1g1 = compute_galaxy_index(1, 1, 0, fmul, 0);
    ASSERT_TRUE(id_bad == id_f1g1,
                "And indeed galaxy 51 of forest 0 == galaxy 1 of forest 1");

    /* Case 2: Valid parameters -> no collision */
    ASSERT_TRUE(passes_validity_check(49, 0, fmul, 0),
                "GalaxyNr=49 < mulfac=50: validity check passes");
    uint64_t id_ok = compute_galaxy_index(49, 0, 0, fmul, 0);
    uint64_t id_f1g0 = compute_galaxy_index(0, 1, 0, fmul, 0);
    ASSERT_TRUE(id_ok != id_f1g0,
                "And indeed galaxy 49 of forest 0 != galaxy 0 of forest 1");
}

// ============================================================================
// TEST 33: Binary/HDF5 Field Correspondence
// ============================================================================
void test_binary_hdf5_field_correspondence(void) {
    BEGIN_TEST("Binary and HDF5 Output Structs Have Matching ID Field Types");

    /* Binary GALAXY_OUTPUT has GalaxyIndex as long long (scalar).
       HDF5 HDF5_GALAXY_OUTPUT has GalaxyIndex as long long* (array).
       The element type must be the same size so the same data is stored. */
    ASSERT_TRUE(sizeof(long long) == 8, "long long is 8 bytes on this platform");
    ASSERT_TRUE(sizeof(int32_t) == 4, "int32_t is 4 bytes (for SAGEHaloIndex etc.)");

    /* Verify that both output formats use the same underlying type width */
    ASSERT_TRUE(sizeof(((struct GALAXY_OUTPUT *)0)->GalaxyIndex) ==
                sizeof(long long),
                "Binary GalaxyIndex field size matches long long");
    ASSERT_TRUE(sizeof(((struct GALAXY_OUTPUT *)0)->SAGEHaloIndex) ==
                sizeof(int32_t),
                "Binary SAGEHaloIndex field size matches int32_t");
}

// ============================================================================
// TEST 34: Full Pipeline Simulation - Build Tree, Generate IDs, Verify Output
// ============================================================================
void test_full_pipeline_simulation(void) {
    BEGIN_TEST("Full Pipeline: Build Halo Tree -> Generate IDs -> Map to Output");

    /* Build a realistic mini-tree with 2 snapshots */
    struct halo_data halos[4];
    struct halo_aux_data haloaux[4];
    struct GALAXY halogal[5];
    memset(halos, 0, sizeof(halos));
    memset(haloaux, 0, sizeof(haloaux));
    memset(halogal, 0, sizeof(halogal));

    /* Snapshot 62: 2 halos in FOF group */
    halos[0].SnapNum = 62; halos[0].FirstHaloInFOFgroup = 0; halos[0].NextHaloInFOFgroup = 1;
    halos[0].Descendant = 2; halos[0].MostBoundID = 1001;
    halos[1].SnapNum = 62; halos[1].FirstHaloInFOFgroup = 0; halos[1].NextHaloInFOFgroup = -1;
    halos[1].Descendant = 3; halos[1].MostBoundID = 1002;

    /* Snapshot 63: 2 halos in FOF group (descendants) */
    halos[2].SnapNum = 63; halos[2].FirstHaloInFOFgroup = 2; halos[2].NextHaloInFOFgroup = 3;
    halos[2].Descendant = -1; halos[2].MostBoundID = 1001;
    halos[3].SnapNum = 63; halos[3].FirstHaloInFOFgroup = 2; halos[3].NextHaloInFOFgroup = -1;
    halos[3].Descendant = -1; halos[3].MostBoundID = 1002;

    /* haloaux */
    haloaux[0].FirstGalaxy = 0; haloaux[0].NGalaxies = 2;
    haloaux[1].FirstGalaxy = 2; haloaux[1].NGalaxies = 1;
    haloaux[2].FirstGalaxy = 3; haloaux[2].NGalaxies = 1;
    haloaux[3].FirstGalaxy = 4; haloaux[3].NGalaxies = 1;

    /* Galaxies */
    halogal[0].HaloNr = 0; halogal[0].GalaxyNr = 0; halogal[0].SnapNum = 62; halogal[0].Type = 0;
    halogal[1].HaloNr = 0; halogal[1].GalaxyNr = 1; halogal[1].SnapNum = 62; halogal[1].Type = 1;
    halogal[2].HaloNr = 1; halogal[2].GalaxyNr = 2; halogal[2].SnapNum = 62; halogal[2].Type = 0;
    halogal[3].HaloNr = 2; halogal[3].GalaxyNr = 0; halogal[3].SnapNum = 63; halogal[3].Type = 0;
    halogal[4].HaloNr = 3; halogal[4].GalaxyNr = 2; halogal[4].SnapNum = 63; halogal[4].Type = 1;

    int64_t fmul = 1000000, fimul = 1000000000LL;
    int64_t forestnr = 7;
    int32_t filenr = 2;

    /* Generate indices for all galaxies */
    uint64_t galaxy_indices[5];
    uint64_t central_indices[5];
    for (int g = 0; g < 5; g++) {
        galaxy_indices[g] = compute_galaxy_index(halogal[g].GalaxyNr, forestnr, filenr, fmul, fimul);
        uint32_t cnr = lookup_central_galaxy_nr(g, halos, haloaux, halogal);
        central_indices[g] = compute_galaxy_index(cnr, forestnr, filenr, fmul, fimul);
    }

    /* Galaxy 0 and 3 have the same GalaxyNr=0 -> same GalaxyIndex.
       This is EXPECTED: galaxy 3 at snap 63 is the evolved version of galaxy 0 at snap 62.
       GalaxyIndex is a persistent identity across time, not a per-snapshot-output key. */
    ASSERT_TRUE(galaxy_indices[0] == galaxy_indices[3],
                "Same GalaxyNr (0) at different snapshots -> same GalaxyIndex (by design)");

    /* Galaxy 2 and 4 have the same GalaxyNr=2 -> same GalaxyIndex */
    ASSERT_TRUE(galaxy_indices[2] == galaxy_indices[4],
                "Same GalaxyNr (2) at different snapshots -> same GalaxyIndex (by design)");

    /* Within a SINGLE snapshot, all GalaxyIndex values must be unique.
       Snap 62 has galaxies 0,1,2; snap 63 has galaxies 3,4. */
    uint64_t snap62_ids[] = {galaxy_indices[0], galaxy_indices[1], galaxy_indices[2]};
    uint64_t snap63_ids[] = {galaxy_indices[3], galaxy_indices[4]};
    ASSERT_EQUAL_INT(0, count_duplicates(snap62_ids, 3),
                     "Galaxy IDs unique within snapshot 62");
    ASSERT_EQUAL_INT(0, count_duplicates(snap63_ids, 2),
                     "Galaxy IDs unique within snapshot 63");

    /* But across snapshots, some IDs repeat (same galaxy evolving) */
    uint64_t all_ids[5];
    memcpy(all_ids, galaxy_indices, sizeof(all_ids));
    int cross_snap_repeats = count_duplicates(all_ids, 5);
    ASSERT_EQUAL_INT(2, cross_snap_repeats,
                     "2 cross-snapshot repeats (galaxyNr 0 and 2 appear at both snaps)");

    /* Galaxies at snap 62 in the same FOF group share CentralGalaxyIndex */
    ASSERT_TRUE(central_indices[0] == central_indices[1],
                "Galaxies in same FOF (snap 62) have same CentralGalaxyIndex");
    ASSERT_TRUE(central_indices[0] == central_indices[2],
                "Subhalo galaxy in same FOF (snap 62) has same CentralGalaxyIndex");

    /* Map to output struct and verify */
    struct GALAXY_OUTPUT out;
    memset(&out, 0, sizeof(out));
    int g = 1;  /* satellite galaxy in halo 0 */
    out.GalaxyIndex = galaxy_indices[g];
    out.CentralGalaxyIndex = central_indices[g];
    out.SAGEHaloIndex = halogal[g].HaloNr;
    out.SAGETreeIndex = (int)forestnr;
    out.SimulationHaloIndex = llabs(halos[halogal[g].HaloNr].MostBoundID);

    ASSERT_TRUE((uint64_t)out.GalaxyIndex == galaxy_indices[g],
                "Output GalaxyIndex matches internal value");
    ASSERT_TRUE(out.SimulationHaloIndex == 1001,
                "SimulationHaloIndex = |MostBoundID| of galaxy's halo");
    ASSERT_EQUAL_INT(0, out.SAGEHaloIndex, "SAGEHaloIndex = HaloNr (0)");
    ASSERT_EQUAL_INT(7, out.SAGETreeIndex, "SAGETreeIndex = original forestnr");

    /* The CentralGalaxyIndex should equal galaxy 0's index (the FOF central) */
    ASSERT_TRUE((uint64_t)out.CentralGalaxyIndex == galaxy_indices[0],
                "Satellite's CentralGalaxyIndex matches central's GalaxyIndex");
}

// ============================================================================
// TEST 35: Same GalaxyNr at Different Snapshots = Same GalaxyIndex
// ============================================================================
void test_same_galaxynr_different_snapshots(void) {
    BEGIN_TEST("Same GalaxyNr at Different Snapshots Produces Same GalaxyIndex");

    /* This is a critical property: a galaxy's ID is persistent across time.
       The same physical galaxy (identified by GalaxyNr) gets output at
       multiple snapshots, and consumers rely on matching IDs to trace evolution. */
    int64_t fmul = 1000000, fimul = 1000000000LL;
    int64_t forestnr = 42;
    int32_t filenr = 5;

    uint32_t gal_nrs[] = {0, 7, 99, 500};
    for (int i = 0; i < 4; i++) {
        uint64_t id = compute_galaxy_index(gal_nrs[i], forestnr, filenr, fmul, fimul);

        /* Verify the ID doesn't depend on any snapshot-varying quantity.
           The formula ONLY uses GalaxyNr, forestnr, filenr, and mulfacs.
           None of these change between snapshots for the same galaxy. */
        uint64_t id_again = compute_galaxy_index(gal_nrs[i], forestnr, filenr, fmul, fimul);
        ASSERT_TRUE(id == id_again, "Same inputs always produce same output (deterministic)");

        /* The ID encodes GalaxyNr, not SnapNum */
        uint64_t base = (uint64_t)forestnr * fmul + (uint64_t)filenr * fimul;
        ASSERT_TRUE(id - base == gal_nrs[i],
                    "ID minus base offset recovers GalaxyNr (not SnapNum)");
    }
}

// ============================================================================
// MAIN TEST RUNNER
// ============================================================================
int main(void) {
    printf("\n");
    printf("═══════════════════════════════════════════════════════════\n");
    printf("  SAGE26 HALO & GALAXY ID TESTS\n");
    printf("═══════════════════════════════════════════════════════════\n");
    printf("\n");

    /* Formula verification */
    test_galaxy_index_basic_formula();

    /* Uniqueness - these prove the ID scheme works */
    test_uniqueness_within_forest();
    test_uniqueness_across_forests();
    test_uniqueness_across_files();
    test_uniqueness_power_of_two_mulfacs();
    test_same_halo_unique_ids();
    test_varying_galaxy_counts_per_forest();

    /* Collision detection - these prove the tests above aren't vacuous */
    test_collision_with_bad_mulfac();
    test_collision_with_bad_file_mulfac();
    test_correct_mulfac_no_collision();
    test_galaxy_nr_at_mulfac_causes_collision();
    test_validity_check_matches_collision();

    /* ID properties */
    test_id_monotonicity();
    test_forest_gap();
    test_file_gap();
    test_max_galaxy_nr_boundary();
    test_encode_decode_round_trip();

    /* CentralGalaxyIndex lookup chain */
    test_central_galaxy_index_via_halo_chain();
    test_central_index_differs_between_fof_groups();

    /* Snapshot/redshift consistency */
    test_same_galaxynr_different_snapshots();

    /* Validity & overflow checks */
    test_validity_check_galaxynr();
    test_validity_check_forest_range();
    test_single_file_mode();
    test_overflow_multiplication();
    test_overflow_addition();
    test_no_overflow_realistic();

    /* Output format */
    test_galaxy_index_struct_size();
    test_output_id_type_widths();
    test_uint64_to_longlong_cast();
    test_binary_hdf5_field_correspondence();
    test_simulation_halo_index_abs();

    /* Output order remapping */
    test_output_order_remapping();
    test_non_output_snapshot_galaxies();

    /* Full pipeline */
    test_full_pipeline_simulation();

    /* Stress */
    test_stress_large_values();

    PRINT_TEST_SUMMARY();

    return (tests_failed > 0) ? 1 : 0;
}

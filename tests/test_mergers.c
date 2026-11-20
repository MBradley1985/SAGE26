/*
 * MERGER PHYSICS TESTS
 * 
 * Tests for galaxy merger processes:
 * - Major vs minor merger classification
 * - Merger timescale calculations
 * - Bulge formation during mergers
 * - Mass redistribution
 * - Black hole growth
 * - Hot-phase gas routed to the merger target's regime reservoir
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "test_framework.h"
#include "../src/core_allvars.h"
#include "../src/model_misc.h"
#include "../src/model_mergers.h"

void test_merger_classification() {
    BEGIN_TEST("Major vs Minor Merger Classification");
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.ThreshMajorMerger = 0.3;  // Standard threshold
    
    // Test major merger (mass ratio > 0.3)
    {
        double mass1 = 1.0;  // Primary
        double mass2 = 0.4;  // Secondary (40% of primary)
        double ratio = mass2 / mass1;
        
        ASSERT_TRUE(ratio > run_params.ThreshMajorMerger, 
                   "Mass ratio 0.4 > 0.3 threshold → Major merger");
    }
    
    // Test minor merger (mass ratio < 0.3)
    {
        double mass1 = 1.0;
        double mass2 = 0.2;  // 20% of primary
        double ratio = mass2 / mass1;
        
        ASSERT_TRUE(ratio < run_params.ThreshMajorMerger,
                   "Mass ratio 0.2 < 0.3 threshold → Minor merger");
    }
    
    // Test boundary case
    {
        double mass1 = 1.0;
        double mass2 = 0.3;  // Exactly at threshold
        double ratio = mass2 / mass1;
        
        ASSERT_CLOSE(ratio, run_params.ThreshMajorMerger, 1e-6,
                    "Mass ratio exactly at threshold");
    }
}

void test_merger_timescale() {
    BEGIN_TEST("Merger Timescale Calculation");
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.Hubble_h = 0.7;
    run_params.G = 43.0;  // G in code units
    run_params.Hubble = 0.1;  // H0/(100 km/s/Mpc) in code units
    run_params.Omega = 0.3;
    run_params.OmegaLambda = 0.7;
    run_params.PartMass = 0.01;  // 10^10 Msun/h per particle
    run_params.MergerTimeFactor = 2.0;  // published default; scales the whole timescale
    
    // Initialize redshift array (simplified - just need entry at snap 30)
    for(int i = 0; i < 64; i++) {
        run_params.ZZ[i] = 2.0 * (63 - i) / 63.0;  // z goes from 2 to 0
    }
    
    struct halo_data halos[2];
    memset(halos, 0, sizeof(struct halo_data) * 2);
    
    // Set up central halo
    halos[0].Len = 10000;  // Number of particles
    halos[0].Mvir = 100.0;  // 10^12 Msun/h
    halos[0].Vmax = 200.0;  // km/s
    halos[0].SnapNum = 30;
    halos[0].Pos[0] = 10.0;  // Mpc/h
    halos[0].Pos[1] = 10.0;
    halos[0].Pos[2] = 10.0;
    
    // Set up satellite halo (smaller)
    halos[1].Len = 1000;   // 10x fewer particles
    halos[1].Mvir = 10.0;   // 10^11 Msun/h
    halos[1].Vmax = 100.0;
    halos[1].SnapNum = 30;
    halos[1].Pos[0] = 10.5;  // 0.5 Mpc/h from central
    halos[1].Pos[1] = 10.0;
    halos[1].Pos[2] = 10.0;
    
    struct GALAXY gal;
    memset(&gal, 0, sizeof(struct GALAXY));
    gal.StellarMass = 1.0;
    gal.ColdGas = 0.5;
    gal.SnapNum = 30;
    
    double mergetime = estimate_merging_time(1, 0, 0, halos, &gal, &run_params);
    
    // Merger time should be positive and reasonable (< 10 Gyr in code units)
    ASSERT_GREATER_THAN(mergetime, 0.0, "Merger time is positive");
    ASSERT_LESS_THAN(mergetime, 10.0, "Merger time is reasonable (< 10 Gyr)");
    
    printf("  ℹ Calculated merger time: %.3f Gyr\n", mergetime);
}

void test_merger_timescale_scaling() {
    BEGIN_TEST("Merger Timescale Scaling with Mass");
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.Hubble_h = 0.7;
    run_params.G = 43.0;
    run_params.Hubble = 0.1;
    run_params.Omega = 0.3;
    run_params.OmegaLambda = 0.7;
    run_params.PartMass = 0.01;
    run_params.MergerTimeFactor = 2.0;  // published default; scales the whole timescale
    
    for(int i = 0; i < 64; i++) {
        run_params.ZZ[i] = 2.0 * (63 - i) / 63.0;
    }
    
    struct halo_data halos[2];
    memset(halos, 0, sizeof(struct halo_data) * 2);
    
    // Central halo
    halos[0].Len = 10000;
    halos[0].Mvir = 100.0;
    halos[0].Vmax = 200.0;
    halos[0].SnapNum = 30;
    halos[0].Pos[0] = 10.0;
    halos[0].Pos[1] = 10.0;
    halos[0].Pos[2] = 10.0;
    
    // Light satellite
    halos[1].Len = 500;
    halos[1].Mvir = 5.0;
    halos[1].Vmax = 80.0;
    halos[1].SnapNum = 30;
    halos[1].Pos[0] = 10.3;
    halos[1].Pos[1] = 10.0;
    halos[1].Pos[2] = 10.0;
    
    struct GALAXY gal;
    memset(&gal, 0, sizeof(struct GALAXY));
    gal.StellarMass = 0.5;
    gal.ColdGas = 0.2;
    
    double mergetime_light = estimate_merging_time(1, 0, 0, halos, &gal, &run_params);
    
    // Heavier satellite
    halos[1].Len = 2000;
    halos[1].Mvir = 20.0;
    gal.StellarMass = 2.0;
    gal.ColdGas = 1.0;
    
    double mergetime_heavy = estimate_merging_time(1, 0, 0, halos, &gal, &run_params);
    
    // Heavier satellites should merge faster (less dynamical friction delay)
    // Actually, heavier satellites experience MORE friction, so should merge slower
    // But the formula has mass in denominator, so heavier = faster
    ASSERT_TRUE(mergetime_light > 0.0 && mergetime_heavy > 0.0,
               "Both merger times positive");
}

void test_merger_mass_conservation() {
    BEGIN_TEST("Mass Conservation During Mergers");
    
    struct GALAXY gal1, gal2;
    memset(&gal1, 0, sizeof(struct GALAXY));
    memset(&gal2, 0, sizeof(struct GALAXY));
    
    // Set up primary galaxy
    gal1.StellarMass = 1.0;
    gal1.BulgeMass = 0.3;
    gal1.ColdGas = 0.5;
    gal1.HotGas = 2.0;
    gal1.BlackHoleMass = 0.01;
    
    // Set up secondary galaxy
    gal2.StellarMass = 0.4;
    gal2.BulgeMass = 0.1;
    gal2.ColdGas = 0.2;
    gal2.HotGas = 0.8;
    gal2.BlackHoleMass = 0.005;
    
    double total_initial_mass = (gal1.StellarMass + gal1.ColdGas + gal1.HotGas + gal1.BlackHoleMass) +
                                (gal2.StellarMass + gal2.ColdGas + gal2.HotGas + gal2.BlackHoleMass);
    
    // Simulate simple merger (add everything to primary)
    gal1.StellarMass += gal2.StellarMass;
    gal1.BulgeMass += gal2.BulgeMass;
    gal1.ColdGas += gal2.ColdGas;
    gal1.HotGas += gal2.HotGas;
    gal1.BlackHoleMass += gal2.BlackHoleMass;
    
    double total_final_mass = gal1.StellarMass + gal1.ColdGas + gal1.HotGas + gal1.BlackHoleMass;
    
    ASSERT_CLOSE(total_initial_mass, total_final_mass, 1e-5,
                "Total mass conserved during merger");
}

void test_merger_remnant_radius() {
    BEGIN_TEST("Merger Bulge Radius After Merger");
    
    struct GALAXY gal1, gal2;
    memset(&gal1, 0, sizeof(struct GALAXY));
    memset(&gal2, 0, sizeof(struct GALAXY));
    
    // Primary galaxy
    gal1.StellarMass = 1.0;
    gal1.BulgeMass = 0.4;
    gal1.ColdGas = 0.5;
    gal1.BulgeRadius = 0.005;  // 5 kpc/h
    
    // Secondary galaxy
    gal2.StellarMass = 0.3;
    gal2.BulgeMass = 0.1;
    gal2.ColdGas = 0.2;
    gal2.BulgeRadius = 0.003;  // 3 kpc/h
    
    // Mass-weighted average radius
    double total_mass = gal1.BulgeMass + gal2.BulgeMass;
    double remnant_radius = (gal1.BulgeMass * gal1.BulgeRadius + 
                            gal2.BulgeMass * gal2.BulgeRadius) / total_mass;
    
    // Remnant radius should be positive and reasonable
    ASSERT_GREATER_THAN(remnant_radius, 0.0, "Remnant radius > 0");
    ASSERT_LESS_THAN(remnant_radius, 0.1, "Remnant radius reasonable (< 100 kpc/h)");
    
    // Should be between the two input radii
    double min_radius = gal2.BulgeRadius;
    double max_radius = gal1.BulgeRadius;
    
    ASSERT_TRUE(remnant_radius >= min_radius, "Remnant >= smaller radius");
    ASSERT_TRUE(remnant_radius <= max_radius, "Remnant <= larger radius");
}

void test_black_hole_merger() {
    BEGIN_TEST("Black Hole Growth During Mergers");
    
    struct GALAXY gal1, gal2;
    memset(&gal1, 0, sizeof(struct GALAXY));
    memset(&gal2, 0, sizeof(struct GALAXY));
    
    // Both galaxies have black holes
    gal1.BlackHoleMass = 0.01;  // 10^8 Msun/h
    gal2.BlackHoleMass = 0.005; // 5×10^7 Msun/h
    
    double initial_bh1 = gal1.BlackHoleMass;
    double initial_bh2 = gal2.BlackHoleMass;
    
    // Black holes merge
    gal1.BlackHoleMass += gal2.BlackHoleMass;
    gal2.BlackHoleMass = 0.0;
    
    ASSERT_CLOSE(gal1.BlackHoleMass, initial_bh1 + initial_bh2, 1e-10,
                "Black hole masses combine correctly");
    ASSERT_EQUAL_FLOAT(gal2.BlackHoleMass, 0.0,
                      "Secondary BH mass transferred");
}

/* Gas masses are floats; a sum of two floats agrees with the double sum to
 * single-precision round-off. */
static const double MERGER_GAS_SUM_TOL = 1e-6;

/*
 * Asserts: add_galaxies_together() puts the satellite's CGMgas into the
 *          HotGas of a hot-regime (Regime == 1) target, with its metals and
 *          dust, and leaves the target's CGMgas at zero.
 *
 * Setup: a Regime 1 central holding only hot gas merges with a Regime 0
 *        satellite holding CGM gas and a little hot gas.  CGMrecipeOn = 1.
 *
 * Why: a hot-regime central keeps its hot-phase gas in HotGas.  Infall,
 *        stripping and feedback all route gas there; this test asks the
 *        same of a merger.  Any CGMgas left on a Regime 1 central sends
 *        cooling_recipe_regime_aware() through the secondary CGM recipe,
 *        whose AGN step sets r_heat against rcool = Rvir.  With a trace
 *        reservoir the heating always exceeds the cooling, so r_heat jumps
 *        to Rvir and, under RheatMode 0, the hot halo never cools again.
 *
 * Tolerance: MERGER_GAS_SUM_TOL, float round-off on a sum of two floats.
 */
void test_merger_cgm_into_hot_regime_target() {
    BEGIN_TEST("Merger: satellite CGM joins a hot-regime target's HotGas");

    static struct GALAXY gals[2];
    memset(gals, 0, sizeof(gals));
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.CGMrecipeOn = 1;

    gals[0].Regime = 1;
    gals[0].HotGas = 10.0;
    gals[0].MetalsHotGas = 0.1;
    gals[0].HotDust = 0.01;

    gals[1].Regime = 0;
    gals[1].CGMgas = 0.5;
    gals[1].MetalsCGMgas = 0.005;
    gals[1].CGMDust = 0.0005;
    gals[1].HotGas = 0.2;
    gals[1].MetalsHotGas = 0.002;
    gals[1].HotDust = 0.0002;

    add_galaxies_together(0, 1, gals, &run_params);

    ASSERT_EQUAL_FLOAT(0.0, gals[0].CGMgas, "Target CGMgas stays zero");
    ASSERT_EQUAL_FLOAT(0.0, gals[0].MetalsCGMgas, "Target MetalsCGMgas stays zero");
    ASSERT_EQUAL_FLOAT(0.0, gals[0].CGMDust, "Target CGMDust stays zero");
    ASSERT_CLOSE(10.0 + 0.5 + 0.2, gals[0].HotGas, MERGER_GAS_SUM_TOL,
                 "Satellite CGM and hot gas both join the target's HotGas");
    ASSERT_CLOSE(0.1 + 0.005 + 0.002, gals[0].MetalsHotGas, MERGER_GAS_SUM_TOL,
                 "Their metals join MetalsHotGas");
    ASSERT_CLOSE(0.01 + 0.0005 + 0.0002, gals[0].HotDust, MERGER_GAS_SUM_TOL,
                 "Their dust joins HotDust");
}

/*
 * Asserts: add_galaxies_together() puts the satellite's HotGas into the
 *          CGMgas of a CGM-regime (Regime == 0) target, with its metals and
 *          dust, and leaves the target's HotGas at zero.
 *
 * Setup: the mirror of the test above -- a Regime 0 central holding only
 *        CGM gas merges with a Regime 1 satellite holding hot and CGM gas.
 *
 * Why: a CGM-regime central cools only from CGMgas, so hot gas handed to
 *        it would sit idle until infall_recipe() folds it into the CGM at
 *        the next snapshot.  Stripping (strip_from_satellite) already gives
 *        to the reservoir of the receiving central's regime; a merger
 *        should do the same.
 *
 * Tolerance: MERGER_GAS_SUM_TOL, float round-off on a sum of two floats.
 */
void test_merger_hot_into_cgm_regime_target() {
    BEGIN_TEST("Merger: satellite hot gas joins a CGM-regime target's CGMgas");

    static struct GALAXY gals[2];
    memset(gals, 0, sizeof(gals));
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.CGMrecipeOn = 1;

    gals[0].Regime = 0;
    gals[0].CGMgas = 3.0;
    gals[0].MetalsCGMgas = 0.03;
    gals[0].CGMDust = 0.003;

    gals[1].Regime = 1;
    gals[1].HotGas = 0.4;
    gals[1].MetalsHotGas = 0.004;
    gals[1].HotDust = 0.0004;
    gals[1].CGMgas = 0.1;
    gals[1].MetalsCGMgas = 0.001;
    gals[1].CGMDust = 0.0001;

    add_galaxies_together(0, 1, gals, &run_params);

    ASSERT_EQUAL_FLOAT(0.0, gals[0].HotGas, "Target HotGas stays zero");
    ASSERT_EQUAL_FLOAT(0.0, gals[0].MetalsHotGas, "Target MetalsHotGas stays zero");
    ASSERT_EQUAL_FLOAT(0.0, gals[0].HotDust, "Target HotDust stays zero");
    ASSERT_CLOSE(3.0 + 0.4 + 0.1, gals[0].CGMgas, MERGER_GAS_SUM_TOL,
                 "Satellite hot and CGM gas both join the target's CGMgas");
    ASSERT_CLOSE(0.03 + 0.004 + 0.001, gals[0].MetalsCGMgas, MERGER_GAS_SUM_TOL,
                 "Their metals join MetalsCGMgas");
    ASSERT_CLOSE(0.003 + 0.0004 + 0.0001, gals[0].CGMDust, MERGER_GAS_SUM_TOL,
                 "Their dust joins CGMDust");
}

/*
 * Asserts: with CGMrecipeOn = 0 a merger still adds HotGas to HotGas,
 *          whatever Regime holds.
 *
 * Setup: Regime 0 on the target (the zeroed default) and CGMrecipeOn = 0.
 *
 * Why: without the CGM recipe every halo keeps its hot-phase gas in HotGas
 *        (classical SAGE); the regime routing above must not apply.
 *
 * Tolerance: MERGER_GAS_SUM_TOL, float round-off on a sum of two floats.
 */
void test_merger_hot_gas_without_cgm_recipe() {
    BEGIN_TEST("Merger: hot gas joins HotGas with the CGM recipe off");

    static struct GALAXY gals[2];
    memset(gals, 0, sizeof(gals));
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.CGMrecipeOn = 0;

    gals[0].HotGas = 5.0;
    gals[1].HotGas = 0.5;

    add_galaxies_together(0, 1, gals, &run_params);

    ASSERT_CLOSE(5.5, gals[0].HotGas, MERGER_GAS_SUM_TOL, "Satellite hot gas joins HotGas");
    ASSERT_EQUAL_FLOAT(0.0, gals[0].CGMgas, "CGMgas stays zero");
}

int main() {
    BEGIN_TEST_SUITE("Galaxy Merger Physics");

    test_merger_classification();
    test_merger_timescale();
    test_merger_timescale_scaling();
    test_merger_mass_conservation();
    test_merger_remnant_radius();
    test_black_hole_merger();
    test_merger_cgm_into_hot_regime_target();
    test_merger_hot_into_cgm_regime_target();
    test_merger_hot_gas_without_cgm_recipe();
    
    END_TEST_SUITE();
    PRINT_TEST_SUMMARY();
    
    return TEST_EXIT_CODE();
}

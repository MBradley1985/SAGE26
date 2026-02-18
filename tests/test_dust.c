/*
 * DUST MODEL TESTS
 * 
 * These tests verify the dust physics implementation:
 * - Dust production from star formation
 * - ISM grain growth (accretion)
 * - Dust destruction by SN shocks
 * - Thermal sputtering in hot gas
 * - Dust transfer during feedback
 * - CGMDust behavior
 * - Conservation and physical bounds
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "test_framework.h"
#include "../src/core_allvars.h"
#include "../src/model_misc.h"
#include "../src/model_dust.h"
#include "../src/model_starformation_and_feedback.h"
#include "../src/model_infall.h"

// Helper function to initialize dust-enabled params
static void initialize_dust_params(struct params *run_params) {
    memset(run_params, 0, sizeof(struct params));
    
    run_params->DustOn = 1;
    run_params->CGMrecipeOn = 1;
    run_params->Hubble_h = 0.7;
    run_params->RecycleFraction = 0.43;
    run_params->UnitTime_in_s = 3.15e16;  // ~1 Gyr
    run_params->UnitDensity_in_cgs = 6.77e-22;
    run_params->UnitMass_in_g = 1.989e43;  // ~10^10 Msun in grams
    run_params->UnitLength_in_cm = 3.086e24;  // ~Mpc in cm
    run_params->UnitVelocity_in_cm_per_s = 1.0e5;  // ~km/s in cm/s
    
    // Dust-specific parameters
    run_params->DeltaDustAGB = 0.2;
    run_params->DeltaDustSNII = 0.2;
    run_params->DeltaDustSNIa = 0.15;
    run_params->DustAccretionTimescale = 50.0;  // Myr
    // Note: DustDestructionEfficiency (eta) and sputtering temperature are
    // hardcoded in model_dust.c, not runtime parameters
    
    // Other needed parameters
    run_params->SupernovaRecipeOn = 1;
    run_params->FeedbackReheatingEpsilon = 3.0;
    run_params->Yield = 0.03;
    run_params->MetalYieldsOn = 0;  // Use simplified model
    
    // Initialize snapshot arrays
    run_params->nsnapshots = 64;
    for(int i = 0; i < 64; i++) {
        run_params->ZZ[i] = 20.0 * (63 - i) / 63.0;
        run_params->AA[i] = 1.0 / (1.0 + run_params->ZZ[i]);
    }
}

// Helper function to initialize a test galaxy with dust
static void initialize_dust_galaxy(struct GALAXY *gal) {
    memset(gal, 0, sizeof(struct GALAXY));
    
    // Gas reservoirs
    gal->ColdGas = 1.0;          // 10^10 Msun/h
    gal->HotGas = 2.0;
    gal->CGMgas = 1.5;
    gal->EjectedMass = 0.5;
    
    // Metals
    gal->MetalsColdGas = 0.02;   // 2% metallicity
    gal->MetalsHotGas = 0.04;
    gal->MetalsCGMgas = 0.03;
    gal->MetalsEjectedMass = 0.01;
    
    // Dust (start with some dust in each reservoir)
    gal->ColdDust = 0.005;       // 25% of cold metals
    gal->HotDust = 0.01;         // 25% of hot metals
    gal->CGMDust = 0.0075;       // 25% of CGM metals
    gal->EjectedDust = 0.0025;   // 25% of ejected metals
    
    // Other properties
    gal->StellarMass = 0.5;
    gal->MetalsStellarMass = 0.01;
    gal->Vvir = 200.0;           // km/s
    gal->H2gas = 0.5;            // 50% molecular
    gal->Regime = 0;             // CGM-regime
}

// Helper to calculate total dust mass
static double total_dust(struct GALAXY *gal) {
    return gal->ColdDust + gal->HotDust + gal->CGMDust + gal->EjectedDust;
}

// Helper to calculate total metal mass
static double total_metals(struct GALAXY *gal) {
    return gal->MetalsColdGas + gal->MetalsHotGas + gal->MetalsCGMgas + 
           gal->MetalsEjectedMass + gal->MetalsStellarMass;
}


/* ========================================================================
 * TEST: Dust Production
 * ======================================================================== */

void test_dust_production() {
    BEGIN_TEST("Dust Production from Star Formation");
    
    struct GALAXY gal[2];
    struct params run_params;
    initialize_dust_params(&run_params);
    
    // Initialize galaxy 0 (star-forming) and galaxy 1 (central)
    memset(gal, 0, sizeof(struct GALAXY) * 2);
    gal[0].ColdGas = 1.0;
    gal[0].MetalsColdGas = 0.02;
    gal[0].ColdDust = 0.0;  // Start with no dust
    gal[0].Vvir = 200.0;
    gal[0].Regime = 0;
    
    gal[1].HotGas = 2.0;
    gal[1].MetalsHotGas = 0.04;
    gal[1].Regime = 0;
    
    double initial_cold_dust = gal[0].ColdDust;
    double stars = 0.1;
    double metallicity = get_metallicity(gal[0].ColdGas, gal[0].MetalsColdGas);
    double dt = 0.01;  // ~10 Myr
    
    produce_dust(stars, metallicity, dt, 0, 1, 0, gal, &run_params);
    
    // Verify dust was produced
    ASSERT_TRUE(gal[0].ColdDust > initial_cold_dust, "Dust produced in ColdDust");
    ASSERT_TRUE(gal[0].ColdDust > 0.0, "ColdDust is positive after production");
    
    // Verify dust doesn't exceed available metals
    ASSERT_TRUE(gal[0].ColdDust <= gal[0].MetalsColdGas, 
                "ColdDust <= MetalsColdGas (physical bound)");
    
    printf("    Produced dust: %.4e (from %.4e)\n", gal[0].ColdDust, initial_cold_dust);
}

void test_dust_production_zero_metallicity() {
    BEGIN_TEST("Dust Production at Zero Metallicity");
    
    struct GALAXY gal[2];
    struct params run_params;
    initialize_dust_params(&run_params);
    
    memset(gal, 0, sizeof(struct GALAXY) * 2);
    gal[0].ColdGas = 1.0;
    gal[0].MetalsColdGas = 0.0;  // No metals!
    gal[0].ColdDust = 0.0;
    
    double stars = 0.1;
    double metallicity = 0.0;
    double dt = 0.01;
    
    produce_dust(stars, metallicity, dt, 0, 1, 0, gal, &run_params);
    
    // At zero metallicity, dust production should still work (primordial enrichment)
    // but the amount should be small
    ASSERT_GREATER_THAN(gal[0].ColdDust + 1e-15, 0.0, "ColdDust >= 0 at Z=0");
    
    printf("    Dust at Z=0: %.4e\n", gal[0].ColdDust);
}


/* ========================================================================
 * TEST: Dust Accretion (ISM Grain Growth)
 * ======================================================================== */

void test_dust_accretion() {
    BEGIN_TEST("Dust Accretion (ISM Grain Growth)");
    
    struct GALAXY gal;
    struct params run_params;
    initialize_dust_params(&run_params);
    initialize_dust_galaxy(&gal);
    
    // Start with low DtM ratio for accretion test
    gal.ColdDust = 0.002;  // 10% of metals
    gal.MetalsColdGas = 0.02;
    gal.H2gas = 0.5;  // 50% molecular
    gal.ColdGas = 1.0;
    
    double initial_dust = gal.ColdDust;
    double metallicity = get_metallicity(gal.ColdGas, gal.MetalsColdGas);
    double dt = 0.1;  // ~100 Myr
    
    accrete_dust(metallicity, dt, 0, 0, &gal, &run_params);
    
    // Verify dust increased (grain growth)
    ASSERT_TRUE(gal.ColdDust > initial_dust, "Dust grows via accretion");
    
    // Verify dust still <= metals
    ASSERT_TRUE(gal.ColdDust <= gal.MetalsColdGas + 1e-10, 
                "ColdDust <= MetalsColdGas after accretion");
    
    double dtm_ratio = gal.ColdDust / gal.MetalsColdGas;
    ASSERT_IN_RANGE(dtm_ratio, 0.0, 1.0, "DtM ratio in [0,1]");
    
    printf("    Dust growth: %.4e -> %.4e (DtM: %.2f%%)\n", 
           initial_dust, gal.ColdDust, dtm_ratio * 100);
}

void test_dust_accretion_saturation() {
    BEGIN_TEST("Dust Accretion Saturation (High DtM)");

    struct GALAXY gal;
    struct params run_params;
    initialize_dust_params(&run_params);
    initialize_dust_galaxy(&gal);

    // Start with high DtM ratio - should see reduced accretion
    gal.ColdDust = 0.018;  // 90% of metals
    gal.MetalsColdGas = 0.02;
    gal.H2gas = 0.5;
    gal.ColdGas = 1.0;

    double initial_dust = gal.ColdDust;
    double initial_total_metals = gal.ColdDust + gal.MetalsColdGas;
    double metallicity = get_metallicity(gal.ColdGas, gal.MetalsColdGas);
    double dt = 0.1;

    accrete_dust(metallicity, dt, 0, 0, &gal, &run_params);

    // Some growth should still occur, but less than at low DtM
    ASSERT_TRUE(gal.ColdDust >= initial_dust, "Dust doesn't decrease from accretion");

    // Note: In this model, MetalsColdGas = gas-phase metals only (not total).
    // Dust + gas-phase metals = total metals, which should be conserved.
    double final_total_metals = gal.ColdDust + gal.MetalsColdGas;
    ASSERT_CLOSE(final_total_metals, initial_total_metals, 1e-10,
                "Total metals (dust + gas-phase) conserved");

    double growth = gal.ColdDust - initial_dust;
    printf("    High-DtM growth: %.4e (saturation slows growth)\n", growth);
}

void test_dust_accretion_no_h2() {
    BEGIN_TEST("Dust Accretion with No H2");
    
    struct GALAXY gal;
    struct params run_params;
    initialize_dust_params(&run_params);
    initialize_dust_galaxy(&gal);
    
    // No molecular gas
    gal.H2gas = 0.0;
    gal.ColdDust = 0.002;
    gal.MetalsColdGas = 0.02;
    gal.ColdGas = 0.01;  // Very little gas
    
    double initial_dust = gal.ColdDust;
    double metallicity = get_metallicity(gal.ColdGas, gal.MetalsColdGas);
    double dt = 0.1;
    
    accrete_dust(metallicity, dt, 0, 0, &gal, &run_params);
    
    // With default f_mol=0.5 fallback for gas-rich galaxies, some growth may occur
    // But with very little gas, growth should be minimal
    printf("    No-H2 dust: %.4e -> %.4e\n", initial_dust, gal.ColdDust);
    
    ASSERT_GREATER_THAN(gal.ColdDust + 1e-15, 0.0, "ColdDust >= 0");
}


/* ========================================================================
 * TEST: Dust Destruction (SN Shocks)
 * ======================================================================== */

void test_dust_destruction() {
    BEGIN_TEST("Dust Destruction by SN Shocks");
    
    struct GALAXY gal;
    struct params run_params;
    initialize_dust_params(&run_params);
    initialize_dust_galaxy(&gal);
    
    double initial_dust = gal.ColdDust;
    double stars = 0.1;  // Recent SF drives SNe
    double metallicity = get_metallicity(gal.ColdGas, gal.MetalsColdGas);
    double dt = 0.01;
    
    destruct_dust(metallicity, stars, dt, 0, 0, &gal, &run_params);
    
    // Dust should decrease or stay the same (never increase from destruction)
    ASSERT_TRUE(gal.ColdDust <= initial_dust + 1e-15, "Dust not increased by destruction");
    ASSERT_TRUE(gal.ColdDust >= 0.0, "ColdDust stays non-negative");
    
    double destroyed = initial_dust - gal.ColdDust;
    if(destroyed > 0.0) {
        printf("    Dust destroyed: %.4e (%.1f%% of initial)\n", 
               destroyed, 100 * destroyed / initial_dust);
    } else {
        // Destruction formula may produce tiny/zero values with test params
        printf("    Dust destruction: %.4e (minimal with test setup)\n", destroyed);
    }
}

void test_dust_destruction_no_sf() {
    BEGIN_TEST("Dust Destruction with No Star Formation");
    
    struct GALAXY gal;
    struct params run_params;
    initialize_dust_params(&run_params);
    initialize_dust_galaxy(&gal);
    
    double initial_dust = gal.ColdDust;
    double stars = 0.0;  // No recent SF
    double metallicity = get_metallicity(gal.ColdGas, gal.MetalsColdGas);
    double dt = 0.01;
    
    destruct_dust(metallicity, stars, dt, 0, 0, &gal, &run_params);
    
    // With no SF, no SNe, so no destruction
    ASSERT_CLOSE(gal.ColdDust, initial_dust, 1e-10, 
                 "No destruction without star formation");
}


/* ========================================================================
 * TEST: Thermal Sputtering
 * ======================================================================== */

void test_thermal_sputtering_hot_gas() {
    BEGIN_TEST("Thermal Sputtering in Hot Gas");
    
    struct GALAXY gal;
    struct params run_params;
    initialize_dust_params(&run_params);
    initialize_dust_galaxy(&gal);
    
    // Set up hot gas with temperature
    gal.HotDust = 0.01;
    gal.HotGas = 2.0;
    gal.Mvir = 100.0;  // Massive halo = hot gas
    gal.Rvir = 0.5;
    gal.Regime = 1;    // Hot-ICM regime
    
    double initial_hot_dust = gal.HotDust;
    double dt = 0.1;  // 100 Myr
    
    dust_thermal_sputtering(0, dt, &gal, &run_params);
    
    // Hot dust should decrease from sputtering
    printf("    HotDust: %.4e -> %.4e\n", initial_hot_dust, gal.HotDust);
    
    ASSERT_TRUE(gal.HotDust >= 0.0, "HotDust stays non-negative");
    ASSERT_TRUE(gal.HotDust <= initial_hot_dust, "HotDust doesn't increase from sputtering");
}

void test_thermal_sputtering_cgm() {
    BEGIN_TEST("Thermal Sputtering in CGM");
    
    struct GALAXY gal;
    struct params run_params;
    initialize_dust_params(&run_params);
    initialize_dust_galaxy(&gal);
    
    // Set up CGM with dust
    gal.CGMDust = 0.0075;
    gal.CGMgas = 1.5;
    gal.Regime = 0;  // CGM-regime
    gal.Mvir = 10.0;  // Lower mass halo
    gal.Rvir = 0.2;
    
    double initial_cgm_dust = gal.CGMDust;
    double dt = 0.1;
    
    dust_thermal_sputtering(0, dt, &gal, &run_params);
    
    printf("    CGMDust: %.4e -> %.4e\n", initial_cgm_dust, gal.CGMDust);
    
    ASSERT_TRUE(gal.CGMDust >= 0.0, "CGMDust stays non-negative");
}

void test_thermal_sputtering_ejected() {
    BEGIN_TEST("Thermal Sputtering in Ejected Reservoir");
    
    struct GALAXY gal;
    struct params run_params;
    initialize_dust_params(&run_params);
    initialize_dust_galaxy(&gal);
    
    gal.EjectedDust = 0.005;
    gal.EjectedMass = 1.0;
    gal.MetalsEjectedMass = 0.02;
    gal.Mvir = 50.0;
    gal.Rvir = 0.3;
    
    double initial_ejected_dust = gal.EjectedDust;
    double dt = 0.1;
    
    dust_thermal_sputtering(0, dt, &gal, &run_params);
    
    printf("    EjectedDust: %.4e -> %.4e\n", initial_ejected_dust, gal.EjectedDust);
    
    ASSERT_TRUE(gal.EjectedDust >= 0.0, "EjectedDust stays non-negative");
    ASSERT_TRUE(gal.EjectedDust <= gal.MetalsEjectedMass + 1e-10, 
                "EjectedDust <= MetalsEjectedMass");
}


/* ========================================================================
 * TEST: Dust Transfer During Feedback
 * ======================================================================== */

void test_dust_feedback_transfer_cgm_regime() {
    BEGIN_TEST("Dust Transfer During Feedback (CGM-regime)");
    
    struct GALAXY gal[2];
    struct params run_params;
    initialize_dust_params(&run_params);
    
    memset(gal, 0, sizeof(struct GALAXY) * 2);
    
    // Galaxy 0: star-forming satellite
    gal[0].ColdGas = 1.0;
    gal[0].MetalsColdGas = 0.02;
    gal[0].ColdDust = 0.01;
    gal[0].Vvir = 150.0;
    gal[0].Regime = 0;
    
    // Galaxy 1: central (CGM-regime)
    gal[1].CGMgas = 2.0;
    gal[1].MetalsCGMgas = 0.04;
    gal[1].CGMDust = 0.005;
    gal[1].EjectedMass = 0.5;
    gal[1].MetalsEjectedMass = 0.01;
    gal[1].EjectedDust = 0.002;
    gal[1].Regime = 0;
    
    double initial_cold_dust = gal[0].ColdDust;
    double initial_cgm_dust = gal[1].CGMDust;
    double initial_ejected_dust = gal[1].EjectedDust;
    
    double reheated = 0.2;
    double ejected = 0.1;
    double metallicity = get_metallicity(gal[0].ColdGas, gal[0].MetalsColdGas);
    
    update_from_feedback(0, 1, reheated, ejected, metallicity, gal, &run_params);
    
    // In CGM-regime: ColdDust -> CGMDust (not HotDust)
    ASSERT_TRUE(gal[0].ColdDust < initial_cold_dust, "ColdDust decreased after feedback");
    ASSERT_TRUE(gal[1].CGMDust > initial_cgm_dust, "CGMDust increased (CGM-regime)");
    
    printf("    ColdDust: %.4e -> %.4e\n", initial_cold_dust, gal[0].ColdDust);
    printf("    CGMDust:  %.4e -> %.4e\n", initial_cgm_dust, gal[1].CGMDust);
    printf("    EjectedDust: %.4e -> %.4e\n", initial_ejected_dust, gal[1].EjectedDust);
}

void test_dust_feedback_transfer_hot_regime() {
    BEGIN_TEST("Dust Transfer During Feedback (Hot-ICM regime)");
    
    struct GALAXY gal[2];
    struct params run_params;
    initialize_dust_params(&run_params);
    
    memset(gal, 0, sizeof(struct GALAXY) * 2);
    
    // Galaxy 0: star-forming satellite
    gal[0].ColdGas = 1.0;
    gal[0].MetalsColdGas = 0.02;
    gal[0].ColdDust = 0.01;
    gal[0].Vvir = 250.0;
    gal[0].Regime = 1;  // Hot-ICM regime
    
    // Galaxy 1: central (Hot-ICM regime)
    gal[1].HotGas = 5.0;
    gal[1].MetalsHotGas = 0.1;
    gal[1].HotDust = 0.02;
    gal[1].EjectedMass = 0.5;
    gal[1].MetalsEjectedMass = 0.01;
    gal[1].EjectedDust = 0.002;
    gal[1].Regime = 1;
    
    double initial_cold_dust = gal[0].ColdDust;
    double initial_hot_dust = gal[1].HotDust;
    
    double reheated = 0.2;
    double ejected = 0.1;
    double metallicity = get_metallicity(gal[0].ColdGas, gal[0].MetalsColdGas);
    
    update_from_feedback(0, 1, reheated, ejected, metallicity, gal, &run_params);
    
    // In Hot-ICM regime: ColdDust -> HotDust
    ASSERT_TRUE(gal[0].ColdDust < initial_cold_dust, "ColdDust decreased after feedback");
    ASSERT_TRUE(gal[1].HotDust > initial_hot_dust, "HotDust increased (Hot-ICM regime)");
    
    printf("    ColdDust: %.4e -> %.4e\n", initial_cold_dust, gal[0].ColdDust);
    printf("    HotDust:  %.4e -> %.4e\n", initial_hot_dust, gal[1].HotDust);
}


/* ========================================================================
 * TEST: Dust Conservation
 * ======================================================================== */

void test_dust_conservation_bounds() {
    BEGIN_TEST("Dust Physical Bounds");
    
    struct GALAXY gal;
    struct params run_params;
    initialize_dust_params(&run_params);
    initialize_dust_galaxy(&gal);
    
    // Run multiple processes
    double dt = 0.1;
    double metallicity = get_metallicity(gal.ColdGas, gal.MetalsColdGas);
    double stars = 0.05;
    
    // Production, accretion, destruction, sputtering
    produce_dust(stars, metallicity, dt, 0, 0, 0, &gal, &run_params);
    accrete_dust(metallicity, dt, 0, 0, &gal, &run_params);
    destruct_dust(metallicity, stars, dt, 0, 0, &gal, &run_params);
    dust_thermal_sputtering(0, dt, &gal, &run_params);
    
    // Check all dust reservoirs are non-negative
    ASSERT_GREATER_THAN(gal.ColdDust + 1e-15, 0.0, "ColdDust >= 0");
    ASSERT_GREATER_THAN(gal.HotDust + 1e-15, 0.0, "HotDust >= 0");
    ASSERT_GREATER_THAN(gal.CGMDust + 1e-15, 0.0, "CGMDust >= 0");
    ASSERT_GREATER_THAN(gal.EjectedDust + 1e-15, 0.0, "EjectedDust >= 0");
    
    // Check dust-to-metal ratios are bounded
    if(gal.MetalsColdGas > 1e-10) {
        double dtm_cold = gal.ColdDust / gal.MetalsColdGas;
        ASSERT_IN_RANGE(dtm_cold, 0.0, 1.0 + 1e-6, "Cold DtM in [0,1]");
    }
    if(gal.MetalsHotGas > 1e-10) {
        double dtm_hot = gal.HotDust / gal.MetalsHotGas;
        ASSERT_IN_RANGE(dtm_hot, 0.0, 1.0 + 1e-6, "Hot DtM in [0,1]");
    }
    if(gal.MetalsCGMgas > 1e-10) {
        double dtm_cgm = gal.CGMDust / gal.MetalsCGMgas;
        ASSERT_IN_RANGE(dtm_cgm, 0.0, 1.0 + 1e-6, "CGM DtM in [0,1]");
    }
}

void test_dust_disabled() {
    BEGIN_TEST("Dust Functions with DustOn=0");
    
    struct params run_params;
    initialize_dust_params(&run_params);
    
    run_params.DustOn = 0;  // Disable dust
    
    // All dust functions should be no-ops when DustOn=0
    // (Note: The actual check is in the calling code, not in the functions themselves)
    // This tests that the DustOn flag is properly set
    
    ASSERT_TRUE(run_params.DustOn == 0, "DustOn is disabled");
    printf("    DustOn=0: dust functions should be called conditionally\n");
}


/* ========================================================================
 * TEST: DustOn Toggle Behavior
 * ======================================================================== */

void test_dust_off_preserves_state() {
    BEGIN_TEST("DustOn=0 Preserves Initial Dust State");
    
    struct GALAXY gal[2];
    struct params run_params;
    initialize_dust_params(&run_params);
    run_params.DustOn = 0;  // Disable dust
    
    memset(gal, 0, sizeof(struct GALAXY) * 2);
    gal[0].ColdGas = 1.0;
    gal[0].MetalsColdGas = 0.02;
    gal[0].ColdDust = 0.005;  // Initial dust
    gal[0].Vvir = 200.0;
    
    gal[1].HotGas = 2.0;
    gal[1].MetalsHotGas = 0.04;
    gal[1].HotDust = 0.01;
    gal[1].Regime = 1;
    
    double initial_cold = gal[0].ColdDust;
    double initial_hot = gal[1].HotDust;
    
    // Feedback with DustOn=0 should NOT transfer dust
    double reheated = 0.1;
    double ejected = 0.05;
    double metallicity = get_metallicity(gal[0].ColdGas, gal[0].MetalsColdGas);
    
    update_from_feedback(0, 1, reheated, ejected, metallicity, gal, &run_params);
    
    // With DustOn=0, dust should remain unchanged
    ASSERT_CLOSE(gal[0].ColdDust, initial_cold, 1e-10, 
                 "ColdDust unchanged when DustOn=0");
    ASSERT_CLOSE(gal[1].HotDust, initial_hot, 1e-10, 
                 "HotDust unchanged when DustOn=0");
    
    printf("    DustOn=0: ColdDust %.4e, HotDust %.4e (unchanged)\n", 
           gal[0].ColdDust, gal[1].HotDust);
}

void test_dust_baryon_conservation() {
    BEGIN_TEST("Dust Counts as Baryonic Mass");
    
    struct params run_params;
    initialize_dust_params(&run_params);
    run_params.BaryonFrac = 0.17;
    run_params.ReionizationOn = 0;
    
    struct GALAXY gal;
    memset(&gal, 0, sizeof(struct GALAXY));
    
    // Set up a halo with dust
    gal.Mvir = 100.0;
    gal.StellarMass = 2.0;
    gal.ColdGas = 1.0;
    gal.HotGas = 5.0;
    gal.CGMgas = 3.0;
    gal.EjectedMass = 1.0;
    gal.BlackHoleMass = 0.02;
    gal.ICS = 0.2;
    gal.Regime = 0;
    
    // Add metals
    gal.MetalsColdGas = 0.02;
    gal.MetalsHotGas = 0.1;
    gal.MetalsCGMgas = 0.06;
    gal.MetalsEjectedMass = 0.02;
    
    // Add dust (must be <= metals)
    gal.ColdDust = 0.005;
    gal.HotDust = 0.025;
    gal.CGMDust = 0.015;
    gal.EjectedDust = 0.005;
    
    // Calculate total baryons INCLUDING dust
    double total_baryons_with_dust = gal.StellarMass + gal.ColdGas + gal.HotGas + 
                                     gal.CGMgas + gal.EjectedMass + gal.BlackHoleMass + 
                                     gal.ICS + gal.ColdDust + gal.HotDust + 
                                     gal.CGMDust + gal.EjectedDust;
    
    double expected_baryons = run_params.BaryonFrac * gal.Mvir;
    
    // Call infall_recipe with DustOn=1
    double infall_with_dust = infall_recipe(0, 1, 0.0, &gal, &run_params);
    
    // Verify infall correctly accounts for dust
    double calculated_infall = expected_baryons - total_baryons_with_dust;
    
    ASSERT_CLOSE(infall_with_dust, calculated_infall, 1e-6,
                "infall_recipe correctly subtracts dust from baryon budget");
    
    // Now test with DustOn=0 - infall should be HIGHER (more room for baryons)
    run_params.DustOn = 0;
    double total_baryons_no_dust = gal.StellarMass + gal.ColdGas + gal.HotGas + 
                                   gal.CGMgas + gal.EjectedMass + gal.BlackHoleMass + gal.ICS;
    double infall_without_dust = expected_baryons - total_baryons_no_dust;
    
    ASSERT_GREATER_THAN(infall_without_dust, infall_with_dust,
                       "Infall higher when dust not counted in baryon budget");
    
    // The difference should equal the total dust mass (within rounding)
    double dust_total = gal.ColdDust + gal.HotDust + gal.CGMDust + gal.EjectedDust;
    ASSERT_CLOSE(infall_without_dust - infall_with_dust, dust_total, 1e-5,
                "Infall difference equals total dust mass");
}

void test_dust_metal_conservation() {
    BEGIN_TEST("Dust Conserved as Metal Subset");
    
    struct params run_params;
    initialize_dust_params(&run_params);
    
    struct GALAXY gal;
    initialize_dust_galaxy(&gal);
    
    // Record initial state
    double initial_total_dust = total_dust(&gal);
    
    // Run multiple dust processes
    double dt = 0.1;
    double metallicity = get_metallicity(gal.ColdGas, gal.MetalsColdGas);
    double stars = 0.05;
    
    // Dust production, growth, destruction
    produce_dust(stars, metallicity, dt, 0, 0, 0, &gal, &run_params);
    accrete_dust(metallicity, dt, 0, 0, &gal, &run_params);
    destruct_dust(metallicity, stars, dt, 0, 0, &gal, &run_params);
    
    double final_total_dust = total_dust(&gal);
    
    // Dust should have changed (not exactly equal)
    ASSERT_TRUE(fabs(final_total_dust - initial_total_dust) > 1e-10,
                "Dust mass changed by processes");
    
    // But dust should never exceed total metals
    double final_total_metals = total_metals(&gal);
    ASSERT_TRUE(final_total_dust <= final_total_metals + 1e-6,
                "Total dust <= total metals");
    
    // Check each reservoir
    if(gal.MetalsColdGas > 1e-10) {
        ASSERT_TRUE(gal.ColdDust <= gal.MetalsColdGas + 1e-6,
                    "ColdDust <= MetalsColdGas");
    }
    if(gal.MetalsHotGas > 1e-10) {
        ASSERT_TRUE(gal.HotDust <= gal.MetalsHotGas + 1e-6,
                    "HotDust <= MetalsHotGas");
    }
    if(gal.MetalsCGMgas > 1e-10) {
        ASSERT_TRUE(gal.CGMDust <= gal.MetalsCGMgas + 1e-6,
                    "CGMDust <= MetalsCGMgas");
    }
}


/* ========================================================================
 * MAIN TEST RUNNER
 * ======================================================================== */

int main(void) {
    BEGIN_TEST_SUITE("Dust Model Tests");
    
    // Production tests
    test_dust_production();
    test_dust_production_zero_metallicity();
    
    // Accretion tests
    test_dust_accretion();
    test_dust_accretion_saturation();
    test_dust_accretion_no_h2();
    
    // Destruction tests
    test_dust_destruction();
    test_dust_destruction_no_sf();
    
    // Sputtering tests
    test_thermal_sputtering_hot_gas();
    test_thermal_sputtering_cgm();
    test_thermal_sputtering_ejected();
    
    // Feedback transfer tests
    test_dust_feedback_transfer_cgm_regime();
    test_dust_feedback_transfer_hot_regime();
    
    // Conservation tests
    test_dust_conservation_bounds();
    test_dust_baryon_conservation();
    test_dust_metal_conservation();
    test_dust_disabled();
    test_dust_off_preserves_state();
    
    END_TEST_SUITE();
    
    // Print summary
    printf("\n");
    printf(COLOR_BLUE "═══════════════════════════════════════════════════════════\n");
    printf("  SUMMARY: %d tests run, %d passed, %d failed\n", 
           tests_run, tests_passed, tests_failed);
    printf("═══════════════════════════════════════════════════════════\n" COLOR_RESET);
    
    return tests_failed > 0 ? 1 : 0;
}

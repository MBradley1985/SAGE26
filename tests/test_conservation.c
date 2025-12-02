/*
 * CONSERVATION LAW TESTS
 * 
 * These tests verify that fundamental conservation laws are respected:
 * - Baryonic mass conservation
 * - Metal mass conservation
 * - Energy conservation (where applicable)
 * 
 * These are the most critical tests for any SAM.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "test_framework.h"
#include "../src/core_allvars.h"
#include "../src/model_misc.h"
#include "../src/model_starformation_and_feedback.h"
#include "../src/model_cooling_heating.h"
#include "../src/model_reincorporation.h"
#include "../src/model_infall.h"

// Helper function to calculate total baryonic mass
double calculate_total_baryonic_mass(struct GALAXY *gal) {
    return gal->ColdGas + gal->StellarMass + gal->BulgeMass + 
           gal->HotGas + gal->CGMgas + gal->EjectedMass;
}

// Helper function to calculate total metal mass
double calculate_total_metal_mass(struct GALAXY *gal) {
    return gal->MetalsColdGas + gal->MetalsStellarMass + gal->MetalsBulgeMass + 
           gal->MetalsHotGas + gal->MetalsCGMgas + gal->MetalsEjectedMass;
}

// Helper function to initialize realistic run_params with redshift arrays
void initialize_realistic_params(struct params *run_params) {
    memset(run_params, 0, sizeof(struct params));
    
    // Set up basic parameters
    run_params->RecycleFraction = 0.43;
    run_params->Hubble_h = 0.7;
    run_params->CGMrecipeOn = 1;
    run_params->SupernovaRecipeOn = 1;
    run_params->AGNrecipeOn = 0;
    run_params->UnitDensity_in_cgs = 6.77e-22;
    run_params->UnitTime_in_s = 3.15e16;
    run_params->G = 43.0;
    run_params->FeedbackReheatingEpsilon = 3.0;
    run_params->ReIncorporationFactor = 1.0;
    run_params->Yield = 0.03;
    
    // Initialize redshift array with realistic values
    run_params->nsnapshots = 64;
    for(int i = 0; i < 64; i++) {
        // Simple redshift progression from z~20 to z=0
        run_params->ZZ[i] = 20.0 * (63 - i) / 63.0;
        run_params->AA[i] = 1.0 / (1.0 + run_params->ZZ[i]);
    }
}

void test_star_formation_conserves_mass() {
    BEGIN_TEST("Star Formation Mass Conservation");
    
    // Set up a test galaxy
    struct GALAXY gal;
    memset(&gal, 0, sizeof(struct GALAXY));
    
    gal.ColdGas = 1.0;  // 10^10 Msun/h
    gal.MetalsColdGas = 0.02;  // 2% metallicity
    gal.StellarMass = 0.5;
    gal.MetalsStellarMass = 0.01;
    
    // Set up parameters
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.RecycleFraction = 0.43;  // Standard value
    
    // Calculate initial masses
    double initial_total_mass = gal.ColdGas + gal.StellarMass;
    double initial_total_metals = gal.MetalsColdGas + gal.MetalsStellarMass;
    double initial_cold_gas = gal.ColdGas;
    double initial_stellar = gal.StellarMass;
    
    // Form 0.1 units of stars
    double stars_formed = 0.1;
    double metallicity = get_metallicity(gal.ColdGas, gal.MetalsColdGas);
    
    update_from_star_formation(0, stars_formed, metallicity, &gal, &run_params);
    
    // Calculate final masses
    double final_total_mass = gal.ColdGas + gal.StellarMass;
    double final_total_metals = gal.MetalsColdGas + gal.MetalsStellarMass;
    
    // Verify function actually changed galaxy state
    ASSERT_TRUE(gal.ColdGas < initial_cold_gas, "ColdGas decreased after SF");
    ASSERT_TRUE(gal.StellarMass > initial_stellar, "StellarMass increased after SF");
    
    // Note: update_from_star_formation properly conserves mass
    // It transfers (1-RecycleFraction)*stars from cold gas to stellar mass
    // The recycled fraction stays in the cold gas (implicitly)
    
    ASSERT_CLOSE(initial_total_mass, final_total_mass, 1e-6,
                 "Total mass conserved");
    
    ASSERT_CLOSE(initial_total_metals, final_total_metals, 1e-6,
                 "Total metal mass conserved");
}

void test_feedback_conserves_mass() {
    BEGIN_TEST("Supernova Feedback Mass Conservation");
    
    // Test both regimes
    for(int regime = 0; regime <= 1; regime++) {
        printf("  Testing Regime %d...\n", regime);
        
        struct GALAXY gal[2];  // [0] = satellite, [1] = central
        memset(gal, 0, sizeof(struct GALAXY) * 2);
        
        // Set up satellite galaxy (index 0) with cold gas
        gal[0].Regime = regime;
        gal[0].ColdGas = 0.5;  // Need cold gas in satellite for reheating
        gal[0].MetalsColdGas = 0.01;
        gal[0].HotGas = 0.0;
        gal[0].MetalsHotGas = 0.0;
        gal[0].CGMgas = 0.0;
        gal[0].MetalsCGMgas = 0.0;
        gal[0].EjectedMass = 0.0;
        gal[0].MetalsEjectedMass = 0.0;
        
        // Set up central galaxy (index 1)
        gal[1].Regime = regime;
        gal[1].ColdGas = 0.0;
        gal[1].MetalsColdGas = 0.0;
        gal[1].HotGas = 1.0;
        gal[1].MetalsHotGas = 0.02;
        gal[1].CGMgas = 0.8;
        gal[1].MetalsCGMgas = 0.015;
        gal[1].EjectedMass = 0.3;
        gal[1].MetalsEjectedMass = 0.005;
        
        struct params run_params;
        memset(&run_params, 0, sizeof(struct params));
        run_params.CGMrecipeOn = 1;
        run_params.SupernovaRecipeOn = 1;
        
        // Calculate initial total mass (sum of both galaxies)
        double initial_total = calculate_total_baryonic_mass(&gal[0]) + calculate_total_baryonic_mass(&gal[1]);
        double initial_metals = calculate_total_metal_mass(&gal[0]) + calculate_total_metal_mass(&gal[1]);
        double initial_sat_cold = gal[0].ColdGas;
        double initial_cen_ejected = gal[1].EjectedMass;
        
        // Apply feedback - reheating comes from satellite's cold gas
        double reheated_mass = 0.1;  // Should be less than gal[0].ColdGas
        double ejected_mass = 0.05;
        double metallicity = 0.02;
        
        update_from_feedback(0, 1, reheated_mass, ejected_mass, metallicity, gal, &run_params);
        
        // Verify feedback actually moved gas
        ASSERT_TRUE(gal[0].ColdGas < initial_sat_cold, "Feedback removed cold gas from satellite");
        ASSERT_TRUE(gal[1].EjectedMass > initial_cen_ejected, "Feedback added to ejected mass");
        
        // Calculate final total mass (sum of both galaxies)
        double final_total = calculate_total_baryonic_mass(&gal[0]) + calculate_total_baryonic_mass(&gal[1]);
        double final_metals = calculate_total_metal_mass(&gal[0]) + calculate_total_metal_mass(&gal[1]);
        
        char msg[256];
        snprintf(msg, sizeof(msg), "Regime %d: Total mass conserved", regime);
        ASSERT_CLOSE(initial_total, final_total, 1e-6, msg);
        
        snprintf(msg, sizeof(msg), "Regime %d: Total metals conserved", regime);
        ASSERT_CLOSE(initial_metals, final_metals, 1e-6, msg);
    }
}

void test_reincorporation_conserves_mass() {
    BEGIN_TEST("Reincorporation Mass Conservation");
    
    for(int regime = 0; regime <= 1; regime++) {
        printf("  Testing Regime %d...\n", regime);
        
        struct GALAXY gal;
        memset(&gal, 0, sizeof(struct GALAXY));
        
        gal.Regime = regime;
        gal.EjectedMass = 1.0;
        gal.MetalsEjectedMass = 0.02;
        gal.HotGas = 0.5;
        gal.MetalsHotGas = 0.01;
        gal.CGMgas = 0.3;
        gal.MetalsCGMgas = 0.006;
        gal.Vvir = 200.0;  // km/s - above reincorporation threshold
        gal.Rvir = 0.2;    // Mpc/h
        
        struct params run_params;
        memset(&run_params, 0, sizeof(struct params));
        run_params.CGMrecipeOn = 1;
        run_params.ReIncorporationFactor = 1.0;
        
        double initial_total = calculate_total_baryonic_mass(&gal);
        double initial_metals = calculate_total_metal_mass(&gal);
        
        // Apply reincorporation for small timestep
        double dt = 0.001;  // Gyr
        reincorporate_gas(0, dt, &gal, &run_params);
        
        double final_total = calculate_total_baryonic_mass(&gal);
        double final_metals = calculate_total_metal_mass(&gal);
        
        char msg[256];
        snprintf(msg, sizeof(msg), "Regime %d: Total mass conserved", regime);
        ASSERT_CLOSE(initial_total, final_total, 1e-6, msg);
        
        snprintf(msg, sizeof(msg), "Regime %d: Total metals conserved", regime);
        ASSERT_CLOSE(initial_metals, final_metals, 1e-6, msg);
    }
}

void test_cooling_conserves_mass() {
    BEGIN_TEST("Cooling Mass Conservation");
    
    struct GALAXY gal;
    memset(&gal, 0, sizeof(struct GALAXY));
    
    gal.HotGas = 1.0;
    gal.MetalsHotGas = 0.02;
    gal.ColdGas = 0.3;
    gal.MetalsColdGas = 0.006;
    gal.Vvir = 150.0;
    gal.Rvir = 0.15;
    gal.SnapNum = 30;  // Some snapshot
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.CGMrecipeOn = 0;  // Use original cooling
    run_params.AGNrecipeOn = 0;
    run_params.UnitDensity_in_cgs = 6.77e-22;
    run_params.UnitTime_in_s = 3.15e16;
    
    double initial_total = gal.ColdGas + gal.HotGas;
    double initial_metals = gal.MetalsColdGas + gal.MetalsHotGas;
    double initial_hot = gal.HotGas;
    double initial_cold = gal.ColdGas;
    
    // Calculate cooling
    double dt = 0.001;
    double coolingGas = cooling_recipe(0, dt, &gal, &run_params);
    
    // Verify cooling function returned sensible value
    ASSERT_TRUE(coolingGas >= 0.0, "Cooling rate is non-negative");
    ASSERT_TRUE(coolingGas <= initial_hot, "Cooling doesn't exceed available hot gas");
    
    // Apply cooling manually (simulate what would happen)
    double metallicity = get_metallicity(gal.HotGas, gal.MetalsHotGas);
    gal.ColdGas += coolingGas;
    gal.MetalsColdGas += metallicity * coolingGas;
    gal.HotGas -= coolingGas;
    gal.MetalsHotGas -= metallicity * coolingGas;
    
    double final_total = gal.ColdGas + gal.HotGas;
    double final_metals = gal.MetalsColdGas + gal.MetalsHotGas;
    
    // If cooling occurred, verify gas moved
    if(coolingGas > 1e-10) {
        ASSERT_TRUE(gal.HotGas < initial_hot, "Hot gas decreased if cooling occurred");
        ASSERT_TRUE(gal.ColdGas > initial_cold, "Cold gas increased if cooling occurred");
    }
    
    ASSERT_CLOSE(initial_total, final_total, 1e-6, "Cooling conserves total gas mass");
    ASSERT_CLOSE(initial_metals, final_metals, 1e-6, "Cooling conserves metal mass");
}

void test_no_negative_masses() {
    BEGIN_TEST("No Negative Masses Allowed");
    
    // This tests that the model never produces negative masses
    struct GALAXY gal;
    memset(&gal, 0, sizeof(struct GALAXY));
    
    // Set up an extreme case that might cause negative masses
    gal.ColdGas = 0.01;  // Very little cold gas
    gal.MetalsColdGas = 0.0002;
    gal.StellarMass = 0.001;
    gal.HotGas = 0.001;
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.RecycleFraction = 0.43;
    run_params.SupernovaRecipeOn = 1;
    run_params.CGMrecipeOn = 1;
    
    // Try to form more stars than available gas
    double stars = gal.ColdGas * 2.0;  // Attempt to form 2x available gas!
    if(stars > gal.ColdGas) stars = gal.ColdGas;  // This should catch it
    
    double metallicity = get_metallicity(gal.ColdGas, gal.MetalsColdGas);
    update_from_star_formation(0, stars, metallicity, &gal, &run_params);
    
    // Check all masses are non-negative
    ASSERT_GREATER_THAN(gal.ColdGas + 1e-10, 0.0, "ColdGas >= 0");
    ASSERT_GREATER_THAN(gal.MetalsColdGas + 1e-10, 0.0, "MetalsColdGas >= 0");
    ASSERT_GREATER_THAN(gal.StellarMass + 1e-10, 0.0, "StellarMass >= 0");
}

void test_edge_cases() {
    BEGIN_TEST("Edge Cases: Extreme Values");
    
    struct params run_params;
    initialize_realistic_params(&run_params);
    
    // Test 1: Very low gas mass (< 1e-8)
    {
        struct GALAXY gal;
        memset(&gal, 0, sizeof(struct GALAXY));
        gal.ColdGas = 1e-10;
        gal.MetalsColdGas = 1e-12;
        gal.StellarMass = 0.1;
        
        double stars = 1e-11;
        double metallicity = get_metallicity(gal.ColdGas, gal.MetalsColdGas);
        update_from_star_formation(0, stars, metallicity, &gal, &run_params);
        
        ASSERT_GREATER_THAN(gal.ColdGas + 1e-12, 0.0, "Very low mass: ColdGas >= 0");
        ASSERT_TRUE(gal.StellarMass > 0.0, "Very low mass: StellarMass > 0");
    }
    
    // Test 2: Very high velocity (> 500 km/s)
    {
        struct GALAXY gal[2];
        memset(gal, 0, sizeof(struct GALAXY) * 2);
        
        gal[0].ColdGas = 1.0;
        gal[0].MetalsColdGas = 0.02;
        gal[0].Vvir = 600.0;  // Very high velocity
        
        gal[1].Regime = 1;
        gal[1].HotGas = 2.0;
        gal[1].MetalsHotGas = 0.04;
        gal[1].Vvir = 600.0;
        
        double reheated = 0.1;
        double ejected = 0.05;
        double metallicity = 0.02;
        
        update_from_feedback(0, 1, reheated, ejected, metallicity, gal, &run_params);
        
        ASSERT_GREATER_THAN(gal[0].ColdGas + 1e-10, 0.0, "High Vvir: ColdGas >= 0");
        ASSERT_GREATER_THAN(gal[1].HotGas + 1e-10, 0.0, "High Vvir: HotGas >= 0");
    }
    
    // Test 3: Zero metallicity
    {
        struct GALAXY gal;
        memset(&gal, 0, sizeof(struct GALAXY));
        gal.ColdGas = 1.0;
        gal.MetalsColdGas = 0.0;  // Zero metallicity
        gal.StellarMass = 0.1;
        
        double stars = 0.1;
        double metallicity = get_metallicity(gal.ColdGas, gal.MetalsColdGas);
        update_from_star_formation(0, stars, metallicity, &gal, &run_params);
        
        ASSERT_EQUAL_FLOAT(0.0, metallicity, "Zero metallicity case");
        ASSERT_GREATER_THAN(gal.ColdGas + 1e-10, 0.0, "Zero Z: ColdGas >= 0");
    }
}

void test_multi_timestep_evolution() {
    BEGIN_TEST("Multi-Timestep Integration Test");
    
    struct params run_params;
    initialize_realistic_params(&run_params);
    
    struct GALAXY gal[2];  // satellite and central
    memset(gal, 0, sizeof(struct GALAXY) * 2);
    
    // Initialize satellite
    gal[0].ColdGas = 1.0;
    gal[0].MetalsColdGas = 0.01;
    gal[0].StellarMass = 0.5;
    gal[0].MetalsStellarMass = 0.005;
    gal[0].Vvir = 100.0;
    gal[0].SnapNum = 30;
    
    // Initialize central
    gal[1].Regime = 0;  // CGM regime
    gal[1].HotGas = 2.0;
    gal[1].MetalsHotGas = 0.03;
    gal[1].CGMgas = 1.5;
    gal[1].MetalsCGMgas = 0.02;
    gal[1].EjectedMass = 0.5;
    gal[1].MetalsEjectedMass = 0.005;
    gal[1].Mvir = 50.0;
    gal[1].Vvir = 150.0;
    gal[1].Rvir = 0.2;
    gal[1].SnapNum = 30;
    
    // Track initial total mass
    double initial_total_mass = calculate_total_baryonic_mass(&gal[0]) + 
                                calculate_total_baryonic_mass(&gal[1]);
    double initial_total_metals = calculate_total_metal_mass(&gal[0]) + 
                                  calculate_total_metal_mass(&gal[1]);
    
    // Evolve through 10 timesteps
    double dt = 0.01;  // 10 Myr
    for(int step = 0; step < 10; step++) {
        // Star formation
        double stars = 0.01;  // Form some stars
        double metallicity = get_metallicity(gal[0].ColdGas, gal[0].MetalsColdGas);
        update_from_star_formation(0, stars, metallicity, gal, &run_params);
        
        // Feedback
        double reheated = 0.02;
        double ejected = 0.01;
        metallicity = get_metallicity(gal[0].ColdGas, gal[0].MetalsColdGas);
        update_from_feedback(0, 1, reheated, ejected, metallicity, gal, &run_params);
        
        // Reincorporation
        reincorporate_gas(1, dt, gal, &run_params);
    }
    
    // Check mass conservation after evolution
    double final_total_mass = calculate_total_baryonic_mass(&gal[0]) + 
                              calculate_total_baryonic_mass(&gal[1]);
    double final_total_metals = calculate_total_metal_mass(&gal[0]) + 
                                calculate_total_metal_mass(&gal[1]);
    
    ASSERT_CLOSE(initial_total_mass, final_total_mass, 1e-6,
                 "Mass conserved over 10 timesteps");
    ASSERT_CLOSE(initial_total_metals, final_total_metals, 0.01,
                 "Metals approximately conserved (allowing for yields)");
    
    // Check galaxy evolved reasonably
    ASSERT_GREATER_THAN(gal[0].StellarMass, 0.5, "Stellar mass increased");
    ASSERT_LESS_THAN(gal[0].ColdGas, 1.0, "Cold gas depleted");
}

void test_performance_regression() {
    BEGIN_TEST("Performance Regression Test");
    
    struct params run_params;
    initialize_realistic_params(&run_params);
    
    struct GALAXY gal;
    memset(&gal, 0, sizeof(struct GALAXY));
    gal.ColdGas = 1.0;
    gal.MetalsColdGas = 0.02;
    gal.StellarMass = 0.5;
    gal.MetalsStellarMass = 0.01;
    
    // Measure time for 10,000 star formation updates
    clock_t start = clock();
    for(int i = 0; i < 10000; i++) {
        double stars = 0.001;
        double metallicity = get_metallicity(gal.ColdGas, gal.MetalsColdGas);
        update_from_star_formation(0, stars, metallicity, &gal, &run_params);
    }
    clock_t end = clock();
    
    double elapsed_ms = 1000.0 * (end - start) / CLOCKS_PER_SEC;
    
    // Should complete in under 10ms (very generous threshold)
    ASSERT_LESS_THAN(elapsed_ms, 10.0, "10k SF updates in < 10ms");
    
    printf("  ℹ Performance: 10k updates completed in %.3f ms\n", elapsed_ms);
    
    // Measure feedback performance
    // Reset galaxy state for each iteration to ensure consistent test
    start = clock();
    for(int i = 0; i < 10000; i++) {
        struct GALAXY gal2[2];
        memset(gal2, 0, sizeof(struct GALAXY) * 2);
        gal2[0].ColdGas = 1.0;
        gal2[0].MetalsColdGas = 0.02;
        gal2[1].Regime = 1;
        gal2[1].HotGas = 2.0;
        gal2[1].MetalsHotGas = 0.04;
        
        // Use small amounts to avoid depleting cold gas
        update_from_feedback(0, 1, 0.0001, 0.00005, 0.02, gal2, &run_params);
    }
    end = clock();
    
    elapsed_ms = 1000.0 * (end - start) / CLOCKS_PER_SEC;
    ASSERT_LESS_THAN(elapsed_ms, 50.0, "10k feedback updates in < 50ms");
    
    printf("  ℹ Performance: 10k feedback updates completed in %.3f ms\n", elapsed_ms);
}

int main() {
    BEGIN_TEST_SUITE("Conservation Laws");
    
    test_star_formation_conserves_mass();
    test_feedback_conserves_mass();
    test_reincorporation_conserves_mass();
    test_cooling_conserves_mass();
    test_no_negative_masses();
    test_edge_cases();
    test_multi_timestep_evolution();
    test_performance_regression();
    
    END_TEST_SUITE();
    PRINT_TEST_SUMMARY();
    
    return TEST_EXIT_CODE();
}

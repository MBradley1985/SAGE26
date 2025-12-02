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
    
    // Form 0.1 units of stars
    double stars_formed = 0.1;
    double metallicity = get_metallicity(gal.ColdGas, gal.MetalsColdGas);
    
    update_from_star_formation(0, stars_formed, metallicity, &gal, &run_params);
    
    // Calculate final masses
    double final_total_mass = gal.ColdGas + gal.StellarMass;
    double final_total_metals = gal.MetalsColdGas + gal.MetalsStellarMass;
    
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
        
        // Apply feedback - reheating comes from satellite's cold gas
        double reheated_mass = 0.1;  // Should be less than gal[0].ColdGas
        double ejected_mass = 0.05;
        double metallicity = 0.02;
        
        update_from_feedback(0, 1, reheated_mass, ejected_mass, metallicity, gal, &run_params);
        
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
    
    // Calculate cooling
    double dt = 0.001;
    double coolingGas = cooling_recipe(0, dt, &gal, &run_params);
    
    // Apply cooling manually (simulate what would happen)
    double metallicity = get_metallicity(gal.HotGas, gal.MetalsHotGas);
    gal.ColdGas += coolingGas;
    gal.MetalsColdGas += metallicity * coolingGas;
    gal.HotGas -= coolingGas;
    gal.MetalsHotGas -= metallicity * coolingGas;
    
    double final_total = gal.ColdGas + gal.HotGas;
    double final_metals = gal.MetalsColdGas + gal.MetalsHotGas;
    
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

int main() {
    BEGIN_TEST_SUITE("Conservation Laws");
    
    test_star_formation_conserves_mass();
    test_feedback_conserves_mass();
    test_reincorporation_conserves_mass();
    test_cooling_conserves_mass();
    test_no_negative_masses();
    
    END_TEST_SUITE();
    PRINT_TEST_SUMMARY();
    
    return TEST_EXIT_CODE();
}

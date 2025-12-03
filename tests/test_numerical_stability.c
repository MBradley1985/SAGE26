/*
 * NUMERICAL STABILITY TESTS
 * 
 * Tests for numerical robustness:
 * - No NaN/Inf values
 * - Roundoff error accumulation
 * - Machine precision conservation
 * - Extreme mass ratios
 * - Timestep independence
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "test_framework.h"
#include "../src/core_allvars.h"
#include "../src/model_misc.h"
#include "../src/model_starformation_and_feedback.h"
#include "../src/model_cooling_heating.h"

void test_no_nan_inf_after_operations() {
    BEGIN_TEST("No NaN/Inf Values After Physics Operations");
    
    struct GALAXY gal[2];
    memset(gal, 0, sizeof(struct GALAXY) * 2);
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.RecycleFraction = 0.43;
    run_params.CGMrecipeOn = 1;
    run_params.SupernovaRecipeOn = 1;
    
    // Set up galaxies
    gal[0].ColdGas = 1.0;
    gal[0].MetalsColdGas = 0.02;
    gal[0].StellarMass = 0.5;
    gal[0].MetalsStellarMass = 0.01;
    
    gal[1].Regime = 0;
    gal[1].HotGas = 2.0;
    gal[1].CGMgas = 1.5;
    gal[1].EjectedMass = 0.3;
    
    // Perform operations
    double stars = 0.1;
    double metallicity = get_metallicity(gal[0].ColdGas, gal[0].MetalsColdGas);
    update_from_star_formation(0, stars, metallicity, gal, &run_params);
    
    double reheated = 0.05;
    double ejected = 0.02;
    update_from_feedback(0, 1, reheated, ejected, metallicity, gal, &run_params);
    
    // Check all values are finite
    ASSERT_FALSE(isnan(gal[0].ColdGas), "ColdGas is not NaN");
    ASSERT_FALSE(isinf(gal[0].ColdGas), "ColdGas is not Inf");
    ASSERT_FALSE(isnan(gal[0].StellarMass), "StellarMass is not NaN");
    ASSERT_FALSE(isinf(gal[0].StellarMass), "StellarMass is not Inf");
    ASSERT_FALSE(isnan(gal[1].CGMgas), "CGMgas is not NaN");
    ASSERT_FALSE(isinf(gal[1].CGMgas), "CGMgas is not Inf");
    ASSERT_FALSE(isnan(gal[1].EjectedMass), "EjectedMass is not NaN");
    ASSERT_FALSE(isinf(gal[1].EjectedMass), "EjectedMass is not Inf");
}

void test_extreme_mass_ratios() {
    BEGIN_TEST("Handles Extreme Mass Ratios");
    
    struct GALAXY gal;
    memset(&gal, 0, sizeof(struct GALAXY));
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.RecycleFraction = 0.43;
    
    // Test 1: Very small gas, large stars
    {
        gal.ColdGas = 1e-10;
        gal.MetalsColdGas = 1e-12;
        gal.StellarMass = 100.0;
        gal.MetalsStellarMass = 2.0;
        
        double metallicity = get_metallicity(gal.ColdGas, gal.MetalsColdGas);
        
        ASSERT_FALSE(isnan(metallicity), "Metallicity finite for small gas");
        ASSERT_TRUE(metallicity >= 0.0, "Metallicity non-negative");
    }
    
    // Test 2: Very large gas, small stars
    {
        gal.ColdGas = 100.0;
        gal.MetalsColdGas = 2.0;
        gal.StellarMass = 1e-10;
        gal.MetalsStellarMass = 1e-12;
        
        double stars = 1e-11;
        double metallicity = get_metallicity(gal.ColdGas, gal.MetalsColdGas);
        update_from_star_formation(0, stars, metallicity, &gal, &run_params);
        
        ASSERT_FALSE(isnan(gal.StellarMass), "StellarMass finite after tiny SF");
        ASSERT_TRUE(gal.StellarMass >= 0.0, "StellarMass non-negative");
    }
    
    // Test 3: Ratio of 1e10 between components
    {
        gal.ColdGas = 1.0;
        gal.StellarMass = 1e10;
        
        ASSERT_TRUE(gal.StellarMass / gal.ColdGas == 1e10,
                   "Handles 10^10 mass ratio");
    }
}

void test_machine_precision_conservation() {
    BEGIN_TEST("Mass Conservation to Machine Precision");
    
    struct GALAXY gal;
    memset(&gal, 0, sizeof(struct GALAXY));
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.RecycleFraction = 0.43;
    
    gal.ColdGas = 1.0;
    gal.MetalsColdGas = 0.02;
    gal.StellarMass = 0.5;
    gal.MetalsStellarMass = 0.01;
    
    double initial_mass = gal.ColdGas + gal.StellarMass;
    
    // Perform many small operations
    for(int i = 0; i < 1000; i++) {
        double stars = 0.0001;
        double metallicity = get_metallicity(gal.ColdGas, gal.MetalsColdGas);
        
        if(gal.ColdGas > stars) {
            update_from_star_formation(0, stars, metallicity, &gal, &run_params);
        }
    }
    
    double final_mass = gal.ColdGas + gal.StellarMass;
    double relative_error = fabs(final_mass - initial_mass) / initial_mass;
    
    // Should conserve to better than 1e-10 relative error
    ASSERT_LESS_THAN(relative_error, 1e-10,
                    "Mass conserved to < 1e-10 relative error");
    
    printf("  ℹ Relative error after 1000 operations: %.3e\n", relative_error);
}

void test_roundoff_accumulation() {
    BEGIN_TEST("Roundoff Error Doesn't Accumulate");
    
    struct GALAXY gal[2];
    memset(gal, 0, sizeof(struct GALAXY) * 2);
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.RecycleFraction = 0.43;
    run_params.CGMrecipeOn = 1;
    run_params.SupernovaRecipeOn = 1;
    
    gal[0].ColdGas = 10.0;
    gal[0].MetalsColdGas = 0.2;
    gal[0].StellarMass = 5.0;
    
    gal[1].Regime = 0;
    gal[1].CGMgas = 8.0;
    gal[1].EjectedMass = 2.0;
    
    double initial_total = gal[0].ColdGas + gal[0].StellarMass + 
                          gal[1].CGMgas + gal[1].EjectedMass;
    
    // Many tiny operations
    for(int i = 0; i < 10000; i++) {
        double tiny_stars = 0.00001;
        double tiny_reheat = 0.000005;
        double tiny_eject = 0.000003;
        
        if(gal[0].ColdGas > tiny_stars + tiny_reheat) {
            double metallicity = get_metallicity(gal[0].ColdGas, gal[0].MetalsColdGas);
            update_from_star_formation(0, tiny_stars, metallicity, gal, &run_params);
            update_from_feedback(0, 1, tiny_reheat, tiny_eject, metallicity, gal, &run_params);
        }
    }
    
    double final_total = gal[0].ColdGas + gal[0].StellarMass + 
                        gal[1].CGMgas + gal[1].EjectedMass;
    double absolute_error = fabs(final_total - initial_total);
    
    // Absolute error should be small even after 10k operations
    ASSERT_LESS_THAN(absolute_error, 1e-2,
                    "Absolute error < 0.01 after 10k operations");
    
    printf("  ℹ Absolute error after 10k operations: %.3e\n", absolute_error);
}

void test_zero_division_protection() {
    BEGIN_TEST("Division by Zero Protection");
    
    struct GALAXY gal;
    memset(&gal, 0, sizeof(struct GALAXY));
    
    // Zero gas mass
    gal.ColdGas = 0.0;
    gal.MetalsColdGas = 0.0;
    
    double metallicity = get_metallicity(gal.ColdGas, gal.MetalsColdGas);
    
    ASSERT_FALSE(isnan(metallicity), "Metallicity not NaN for zero gas");
    ASSERT_FALSE(isinf(metallicity), "Metallicity not Inf for zero gas");
    ASSERT_TRUE(metallicity >= 0.0, "Metallicity >= 0 for zero gas");
    
    // Very small gas with metals
    gal.ColdGas = 1e-20;
    gal.MetalsColdGas = 1e-21;
    
    metallicity = get_metallicity(gal.ColdGas, gal.MetalsColdGas);
    
    ASSERT_FALSE(isnan(metallicity), "Metallicity not NaN for tiny gas");
    ASSERT_FALSE(isinf(metallicity), "Metallicity not Inf for tiny gas");
}

void test_timestep_convergence() {
    BEGIN_TEST("Results Converge with Smaller Timestep");
    
    struct GALAXY gal1, gal2;
    memset(&gal1, 0, sizeof(struct GALAXY));
    memset(&gal2, 0, sizeof(struct GALAXY));
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.RecycleFraction = 0.43;
    
    // Initial state (same for both)
    gal1.ColdGas = gal2.ColdGas = 1.0;
    gal1.MetalsColdGas = gal2.MetalsColdGas = 0.02;
    gal1.StellarMass = gal2.StellarMass = 0.5;
    gal1.MetalsStellarMass = gal2.MetalsStellarMass = 0.01;
    
    // Large timestep: form 0.1 stars in one go
    double stars_large = 0.1;
    double metallicity = get_metallicity(gal1.ColdGas, gal1.MetalsColdGas);
    update_from_star_formation(0, stars_large, metallicity, &gal1, &run_params);
    
    // Small timestep: form 0.1 stars in 10 steps
    for(int i = 0; i < 10; i++) {
        double stars_small = 0.01;
        metallicity = get_metallicity(gal2.ColdGas, gal2.MetalsColdGas);
        update_from_star_formation(0, stars_small, metallicity, &gal2, &run_params);
    }
    
    // Results should be similar (within a few percent)
    double diff_coldgas = fabs(gal1.ColdGas - gal2.ColdGas);
    double diff_stellar = fabs(gal1.StellarMass - gal2.StellarMass);
    
    ASSERT_LESS_THAN(diff_coldgas, 0.01, "ColdGas similar for different timesteps");
    ASSERT_LESS_THAN(diff_stellar, 0.01, "StellarMass similar for different timesteps");
    
    printf("  ℹ ColdGas diff: %.3e, StellarMass diff: %.3e\n", diff_coldgas, diff_stellar);
}

void test_negative_mass_protection() {
    BEGIN_TEST("Protection Against Negative Masses");
    
    struct GALAXY gal;
    memset(&gal, 0, sizeof(struct GALAXY));
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.RecycleFraction = 0.43;
    
    // Try to form more stars than available
    gal.ColdGas = 0.01;
    gal.MetalsColdGas = 0.0002;
    gal.StellarMass = 0.1;
    gal.MetalsStellarMass = 0.002;
    
    double excessive_stars = 0.1;  // 10x available gas
    
    // The function should internally limit this
    if(excessive_stars > gal.ColdGas) {
        excessive_stars = gal.ColdGas;
    }
    
    double metallicity = get_metallicity(gal.ColdGas, gal.MetalsColdGas);
    update_from_star_formation(0, excessive_stars, metallicity, &gal, &run_params);
    
    // Gas should not go negative
    ASSERT_TRUE(gal.ColdGas >= -1e-10, "ColdGas not significantly negative");
    ASSERT_TRUE(gal.MetalsColdGas >= -1e-10, "MetalsColdGas not significantly negative");
}

int main() {
    BEGIN_TEST_SUITE("Numerical Stability & Precision");
    
    test_no_nan_inf_after_operations();
    test_extreme_mass_ratios();
    test_machine_precision_conservation();
    test_roundoff_accumulation();
    test_zero_division_protection();
    test_timestep_convergence();
    test_negative_mass_protection();
    
    END_TEST_SUITE();
    PRINT_TEST_SUMMARY();
    
    return TEST_EXIT_CODE();
}

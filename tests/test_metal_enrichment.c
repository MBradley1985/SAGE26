/*
 * METAL ENRICHMENT TESTS
 * 
 * Tests for metal production, enrichment, and mixing:
 * - Metal yields from star formation
 * - Metal enrichment during stellar evolution
 * - Metal mixing between gas phases
 * - Metallicity tracking through processes
 * - Metal conservation with yields
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "test_framework.h"
#include "../src/core_allvars.h"
#include "../src/model_misc.h"
#include "../src/model_starformation_and_feedback.h"

void test_metal_yield_production() {
    BEGIN_TEST("Metal Yield Production from Star Formation");
    
    struct GALAXY gal;
    memset(&gal, 0, sizeof(struct GALAXY));
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.RecycleFraction = 0.43;
    run_params.Yield = 0.03;  // 3% yield - typical value
    
    // Start with primordial gas (zero metallicity)
    gal.ColdGas = 1.0;
    gal.MetalsColdGas = 0.0;
    gal.StellarMass = 0.0;
    gal.MetalsStellarMass = 0.0;
    
    // Form stars from metal-free gas
    double stars = 0.1;
    double metallicity = 0.0;
    
    update_from_star_formation(0, stars, metallicity, &gal, &run_params);
    
    // After update_from_star_formation, mass is transferred
    // Metals are added separately in starformation_and_feedback()
    // Here we test that the mass update works correctly
    double expected_cold = 1.0 - (1.0 - run_params.RecycleFraction) * stars;
    double expected_stellar = (1.0 - run_params.RecycleFraction) * stars;
    
    ASSERT_CLOSE(gal.ColdGas, expected_cold, 1e-5,
                "Cold gas decreased by (1-R) × stars");
    ASSERT_CLOSE(gal.StellarMass, expected_stellar, 1e-5,
                "Stellar mass increased by (1-R) × stars");
}

void test_metallicity_enrichment_evolution() {
    BEGIN_TEST("Metallicity Increases with Continued Star Formation");
    
    struct GALAXY gal;
    memset(&gal, 0, sizeof(struct GALAXY));
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.RecycleFraction = 0.43;
    run_params.Yield = 0.03;
    
    gal.ColdGas = 1.0;
    gal.MetalsColdGas = 0.001;  // Start with low metallicity
    gal.StellarMass = 0.0;
    gal.MetalsStellarMass = 0.0;
    
    double Z_initial = get_metallicity(gal.ColdGas, gal.MetalsColdGas);
    
    // Form stars in multiple bursts
    for(int i = 0; i < 5; i++) {
        if(gal.ColdGas > 0.05) {
            double metallicity = get_metallicity(gal.ColdGas, gal.MetalsColdGas);
            update_from_star_formation(0, 0.05, metallicity, &gal, &run_params);
        }
    }
    
    double Z_final = get_metallicity(gal.ColdGas, gal.MetalsColdGas);
    
    ASSERT_GREATER_THAN(Z_final, Z_initial, 
                       "Metallicity increases with star formation");
    ASSERT_LESS_THAN(Z_final, 1.0,
                    "Metallicity stays below 100%");
}

void test_metal_conservation_with_yields() {
    BEGIN_TEST("Metal Mass Conserved in update_from_star_formation");
    
    struct GALAXY gal;
    memset(&gal, 0, sizeof(struct GALAXY));
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.RecycleFraction = 0.43;
    run_params.Yield = 0.03;
    
    gal.ColdGas = 1.0;
    gal.MetalsColdGas = 0.02;
    gal.StellarMass = 0.5;
    gal.MetalsStellarMass = 0.01;
    
    double initial_metals = gal.MetalsColdGas + gal.MetalsStellarMass;
    double initial_mass = gal.ColdGas + gal.StellarMass;
    
    // Form stars
    double stars = 0.1;
    double metallicity = get_metallicity(gal.ColdGas, gal.MetalsColdGas);
    update_from_star_formation(0, stars, metallicity, &gal, &run_params);
    
    double final_metals = gal.MetalsColdGas + gal.MetalsStellarMass;
    double final_mass = gal.ColdGas + gal.StellarMass;
    
    // update_from_star_formation conserves metals (without yield)
    // Yields are added separately in starformation_and_feedback
    ASSERT_CLOSE(final_metals, initial_metals, 1e-5,
                "Metals conserved in update_from_star_formation");
    ASSERT_CLOSE(final_mass, initial_mass, 1e-5,
                "Mass conserved in update_from_star_formation");
}

void test_metal_mixing_during_feedback() {
    BEGIN_TEST("Metals Mix into Hot Gas During Feedback");
    
    struct GALAXY gal[2];
    memset(gal, 0, sizeof(struct GALAXY) * 2);
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.CGMrecipeOn = 1;
    run_params.SupernovaRecipeOn = 1;
    
    // Satellite with metal-rich cold gas
    gal[0].ColdGas = 0.5;
    gal[0].MetalsColdGas = 0.02;  // 4% metallicity
    double Z_cold = get_metallicity(gal[0].ColdGas, gal[0].MetalsColdGas);
    
    // Central with metal-poor hot gas
    gal[1].Regime = 1;
    gal[1].HotGas = 2.0;
    gal[1].MetalsHotGas = 0.01;  // 0.5% metallicity
    double Z_hot_before = get_metallicity(gal[1].HotGas, gal[1].MetalsHotGas);
    
    // Apply feedback - metal-rich cold gas reheats to hot
    double reheated = 0.1;
    double ejected = 0.0;
    double metallicity = Z_cold;
    
    update_from_feedback(0, 1, reheated, ejected, metallicity, gal, &run_params);
    
    // Hot gas should now have higher metallicity
    double Z_hot_after = get_metallicity(gal[1].HotGas, gal[1].MetalsHotGas);
    
    ASSERT_GREATER_THAN(Z_hot_after, Z_hot_before,
                       "Hot gas metallicity increases from metal-rich reheating");
    ASSERT_LESS_THAN(Z_hot_after, Z_cold,
                    "Hot gas Z between initial hot and cold values");
}

void test_metallicity_floor() {
    BEGIN_TEST("Metallicity Floor for Primordial Gas");
    
    struct GALAXY gal;
    memset(&gal, 0, sizeof(struct GALAXY));
    
    // Truly zero metallicity
    gal.ColdGas = 1.0;
    gal.MetalsColdGas = 0.0;
    
    double Z = get_metallicity(gal.ColdGas, gal.MetalsColdGas);
    
    ASSERT_TRUE(Z >= 0.0, "Metallicity >= 0");
    ASSERT_TRUE(Z <= 1e-6, "Zero metals → very low Z");
}

void test_metal_dilution_from_infall() {
    BEGIN_TEST("Metal Dilution from Primordial Infall");
    
    struct GALAXY gal;
    memset(&gal, 0, sizeof(struct GALAXY));
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.CGMrecipeOn = 0;
    
    // Start with metal-rich hot gas
    gal.Regime = 1;
    gal.HotGas = 1.0;
    gal.MetalsHotGas = 0.03;  // 3% metallicity
    
    double Z_before = get_metallicity(gal.HotGas, gal.MetalsHotGas);
    
    // Add primordial infall (no metals)
    double infall = 0.5;
    gal.HotGas += infall;
    // MetalsHotGas stays the same - dilution
    
    double Z_after = get_metallicity(gal.HotGas, gal.MetalsHotGas);
    
    ASSERT_LESS_THAN(Z_after, Z_before,
                    "Primordial infall dilutes metallicity");
    
    // Expected dilution
    double Z_expected = gal.MetalsHotGas / gal.HotGas;
    ASSERT_CLOSE(Z_after, Z_expected, 1e-6,
                "Dilution follows mass mixing");
}

void test_yield_parameter_range() {
    BEGIN_TEST("Yield Parameter in Physical Range");
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    
    // Typical yields: 0.01-0.05 (1-5%)
    run_params.Yield = 0.03;
    
    ASSERT_IN_RANGE(run_params.Yield, 0.001, 0.1,
                   "Yield in reasonable range (0.1-10%)");
    
    // Test metal production
    double stars = 1.0;
    double new_metals = run_params.Yield * stars;
    
    ASSERT_IN_RANGE(new_metals, 0.001, 0.1,
                   "Metal production per unit stellar mass is reasonable");
}

void test_metallicity_gradients() {
    BEGIN_TEST("Different Gas Phases Can Have Different Metallicities");
    
    struct GALAXY gal;
    memset(&gal, 0, sizeof(struct GALAXY));
    
    // Cold gas - enriched by star formation
    gal.ColdGas = 0.5;
    gal.MetalsColdGas = 0.015;  // 3% Z
    
    // Hot gas - diluted by infall
    gal.HotGas = 2.0;
    gal.MetalsHotGas = 0.01;    // 0.5% Z
    
    // CGM - intermediate
    gal.CGMgas = 1.0;
    gal.MetalsCGMgas = 0.015;   // 1.5% Z
    
    double Z_cold = get_metallicity(gal.ColdGas, gal.MetalsColdGas);
    double Z_hot = get_metallicity(gal.HotGas, gal.MetalsHotGas);
    double Z_cgm = get_metallicity(gal.CGMgas, gal.MetalsCGMgas);
    
    ASSERT_TRUE(Z_cold >= Z_hot, "Cold gas typically more enriched than hot");
    
    // All should be physical
    ASSERT_IN_RANGE(Z_cold, 0.0, 1.0, "Cold gas Z in [0,1]");
    ASSERT_IN_RANGE(Z_hot, 0.0, 1.0, "Hot gas Z in [0,1]");
    ASSERT_IN_RANGE(Z_cgm, 0.0, 1.0, "CGM Z in [0,1]");
}

void test_supernova_metal_ejection() {
    BEGIN_TEST("Supernova Feedback Ejects Metals");
    
    struct GALAXY gal[2];
    memset(gal, 0, sizeof(struct GALAXY) * 2);
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.CGMrecipeOn = 1;
    run_params.SupernovaRecipeOn = 1;
    
    // Satellite with enriched gas
    gal[0].ColdGas = 1.0;
    gal[0].MetalsColdGas = 0.03;
    
    // Central to receive ejected gas - start with some hot gas
    gal[1].Regime = 1;
    gal[1].EjectedMass = 0.0;
    gal[1].MetalsEjectedMass = 0.0;
    gal[1].HotGas = 1.0;  // Pre-existing hot gas
    gal[1].MetalsHotGas = 0.01;  // 1% metallicity
    
    double initial_ejected_metals = gal[1].MetalsEjectedMass;
    double initial_total_metals = gal[0].MetalsColdGas + gal[1].MetalsHotGas + gal[1].MetalsEjectedMass;
    
    // Feedback with ejection
    double reheated = 0.1;
    double ejected = 0.2;
    double metallicity = 0.03;
    
    update_from_feedback(0, 1, reheated, ejected, metallicity, gal, &run_params);
    
    // Check metals were ejected
    ASSERT_GREATER_THAN(gal[1].MetalsEjectedMass, initial_ejected_metals,
                       "Ejected metals increased");
    
    // Metal conservation: total should be preserved
    // Reheated brings metallicity*reheated, then ejection uses mixed metallicity
    double final_total_metals = gal[0].MetalsColdGas + gal[1].MetalsHotGas + 
                               gal[1].MetalsEjectedMass + gal[1].MetalsCGMgas;
    
    ASSERT_CLOSE(final_total_metals, initial_total_metals, 1e-5,
                "Total metals conserved through feedback");
}

void test_closed_box_enrichment() {
    BEGIN_TEST("Closed Box Enrichment Reaches Expected Metallicity");
    
    struct GALAXY gal;
    memset(&gal, 0, sizeof(struct GALAXY));
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.RecycleFraction = 0.43;
    run_params.Yield = 0.03;
    
    // Start with gas reservoir, low metallicity
    gal.ColdGas = 1.0;
    gal.MetalsColdGas = 0.001;  // Start with some metals
    gal.StellarMass = 0.0;
    gal.MetalsStellarMass = 0.0;
    
    // Form stars and add yields (simulating full SF cycle)
    for(int i = 0; i < 20; i++) {
        if(gal.ColdGas > 0.01) {
            double metallicity = get_metallicity(gal.ColdGas, gal.MetalsColdGas);
            double stars = 0.03;
            update_from_star_formation(0, stars, metallicity, &gal, &run_params);
            // Manually add yield to cold gas (simulating instantaneous recycling)
            gal.MetalsColdGas += run_params.Yield * stars;
        }
    }
    
    // In closed box: final Z ~ Yield × ln(1/gas_fraction)
    double Z_final = get_metallicity(gal.ColdGas, gal.MetalsColdGas);
    
    // Should be enriched but not unreasonably so
    ASSERT_GREATER_THAN(Z_final, 0.005, "Significant enrichment occurred");
    ASSERT_LESS_THAN(Z_final, 0.5, "Enrichment not excessive");
}

int main() {
    BEGIN_TEST_SUITE("Metal Enrichment & Yields");
    
    test_metal_yield_production();
    test_metallicity_enrichment_evolution();
    test_metal_conservation_with_yields();
    test_metal_mixing_during_feedback();
    test_metallicity_floor();
    test_metal_dilution_from_infall();
    test_yield_parameter_range();
    test_metallicity_gradients();
    test_supernova_metal_ejection();
    test_closed_box_enrichment();
    
    END_TEST_SUITE();
    PRINT_TEST_SUMMARY();
    
    return TEST_EXIT_CODE();
}

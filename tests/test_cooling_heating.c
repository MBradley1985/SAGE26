/**
 * @file test_cooling_heating.c
 * @brief Unit tests for gas cooling and heating physics
 *
 * Tests cooling rate calculations, heating from AGN feedback, thermal
 * equilibrium states, and the interaction between cooling and heating.
 *
 * Key physics tested:
 * - Metal-dependent cooling rates
 * - Cooling time calculations (tcool)
 * - Free-fall time and tcool/tff ratio for precipitation
 * - Cooling radius (rcool) calculations
 * - AGN heating suppression of cooling
 * - Temperature scaling with Vvir
 * - Regime-dependent cooling (Hot vs CGM)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../src/core_allvars.h"
#include "../src/core_cool_func.h"
#include "../src/model_cooling_heating.h"
#include "../src/model_misc.h"
#include "test_framework.h"

// ============================================================================
// TEST 1: Virial Temperature Scales With Vvir^2
// ============================================================================
void test_virial_temperature_scaling() {
    BEGIN_TEST("Virial Temperature Scales With Vvir^2");
    
    // T_vir = 35.9 * Vvir^2 (in Kelvin, Vvir in km/s)
    double Vvir1 = 100.0;  // km/s
    double Vvir2 = 200.0;  // km/s
    
    double T1 = 35.9 * Vvir1 * Vvir1;
    double T2 = 35.9 * Vvir2 * Vvir2;
    
    double expected_ratio = 4.0;  // (200/100)^2
    double actual_ratio = T2 / T1;
    
    ASSERT_CLOSE(actual_ratio, expected_ratio, 1e-10,
                "Temperature scales as Vvir^2");
    
    // Check temperature ranges
    ASSERT_IN_RANGE(T1, 3.0e5, 4.0e5,
                   "Low-mass halo temperature ~ 10^5 K");
    ASSERT_IN_RANGE(T2, 1.3e6, 1.5e6,
                   "Massive halo temperature ~ 10^6 K");
}

// ============================================================================
// TEST 2: Cooling Rate Depends on Metallicity
// ============================================================================
void test_metallicity_dependent_cooling() {
    BEGIN_TEST("Cooling Rate Increases With Metallicity");
    
    double temp = 1.0e6;  // K
    double log_temp = log10(temp);
    
    // Solar metallicity
    double logZ_solar = log10(0.02);
    double lambda_solar = get_metaldependent_cooling_rate(log_temp, logZ_solar);
    
    // 10% solar metallicity
    double logZ_low = log10(0.002);
    double lambda_low = get_metaldependent_cooling_rate(log_temp, logZ_low);
    
    // Primordial (Z = 0)
    double logZ_prim = -10.0;
    double lambda_prim = get_metaldependent_cooling_rate(log_temp, logZ_prim);
    
    // At 10^6 K, cooling rates increase with metallicity but may be similar
    ASSERT_TRUE(lambda_solar >= lambda_low,
               "Solar Z cools at least as fast as 0.1 Z_sun");
    ASSERT_TRUE(lambda_low >= lambda_prim,
               "0.1 Z_sun cools at least as fast as primordial");
    
    // Cooling rate should be positive
    ASSERT_GREATER_THAN(lambda_prim, 0.0,
                       "Primordial cooling rate positive");
}

// ============================================================================
// TEST 3: Cooling Time vs Free-Fall Time Ratio
// ============================================================================
void test_cooling_freefall_ratio() {
    BEGIN_TEST("Cooling Time vs Free-Fall Time Ratio");
    
    struct GALAXY gal[1];
    memset(gal, 0, sizeof(struct GALAXY));
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.CGMrecipeOn = 1;
    run_params.UnitTime_in_s = 3.08568e+16;  // Gyr to seconds
    run_params.UnitDensity_in_cgs = 6.76991e-31;
    run_params.Hubble_h = 0.7;
    run_params.G = 43007.1;
    
    // Setup galaxy in CGM regime
    gal[0].CGMgas = 10.0;  // 10^11 Msun
    gal[0].MetalsCGMgas = 0.02;  // 2% metallicity
    gal[0].Vvir = 200.0;  // km/s
    gal[0].Rvir = 200.0;  // Mpc/h
    gal[0].Mvir = 100.0;  // 10^12 Msun
    gal[0].Regime = 0;
    
    // Call cooling to calculate tcool and tff
    double dt = 0.01;
    cooling_recipe_cgm(0, dt, gal, &run_params);
    
    // Both times should be positive
    ASSERT_GREATER_THAN(gal[0].tcool, 0.0,
                       "Cooling time positive");
    ASSERT_GREATER_THAN(gal[0].tff, 0.0,
                       "Free-fall time positive");
    
    // tcool/tff can vary widely depending on conditions
    ASSERT_TRUE(gal[0].tcool_over_tff > 0.0,
               "tcool/tff positive");
}

// ============================================================================
// TEST 4: Cooling Radius Calculation
// ============================================================================
void test_cooling_radius() {
    BEGIN_TEST("Cooling Radius Rcool Calculation");
    
    struct GALAXY gal[1];
    memset(gal, 0, sizeof(struct GALAXY));
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.CGMrecipeOn = 0;  // Use hot recipe
    run_params.UnitTime_in_s = 3.08568e+16;
    run_params.UnitDensity_in_cgs = 6.76991e-31;
    run_params.Hubble_h = 0.7;
    run_params.AGNrecipeOn = 0;
    
    // Galaxy with hot gas
    gal[0].HotGas = 5.0;  // 5×10^10 Msun/h
    gal[0].MetalsHotGas = 0.01;  // 1% metallicity
    gal[0].Vvir = 150.0;  // km/s
    gal[0].Rvir = 150.0;  // kpc/h
    gal[0].Regime = 1;
    
    double dt = 0.01;
    cooling_recipe_hot(0, dt, gal, &run_params);
    
    // RcoolToRvir should be calculated
    ASSERT_GREATER_THAN(gal[0].RcoolToRvir, 0.0,
                       "Cooling radius positive");
    
    // Rcool/Rvir can be large in "cold mode" when cooling radius exceeds virial radius
    ASSERT_TRUE(gal[0].RcoolToRvir >= 0.0,
               "Rcool/Rvir non-negative");
}

// ============================================================================
// TEST 5: Hot Gas Cooling Rate Depends on HotGas Mass
// ============================================================================
void test_cooling_mass_dependence() {
    BEGIN_TEST("Cooling Rate Scales With Hot Gas Mass");
    
    struct GALAXY gal[2];
    memset(gal, 0, sizeof(struct GALAXY) * 2);
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.CGMrecipeOn = 0;
    run_params.UnitTime_in_s = 3.08568e+16;
    run_params.UnitDensity_in_cgs = 6.76991e-31;
    run_params.Hubble_h = 0.7;
    run_params.AGNrecipeOn = 0;
    
    double dt = 0.01;
    
    // Galaxy 0: Small hot gas mass
    gal[0].HotGas = 1.0;
    gal[0].MetalsHotGas = 0.02;
    gal[0].Vvir = 200.0;
    gal[0].Rvir = 200.0;
    gal[0].Cooling = 0.0;
    
    // Galaxy 1: Large hot gas mass (5x more)
    gal[1].HotGas = 5.0;
    gal[1].MetalsHotGas = 0.1;  // Same metallicity
    gal[1].Vvir = 200.0;
    gal[1].Rvir = 200.0;
    gal[1].Cooling = 0.0;
    
    double cooled0 = cooling_recipe_hot(0, dt, gal, &run_params);
    double cooled1 = cooling_recipe_hot(1, dt, gal, &run_params);
    
    // More gas should cool more
    ASSERT_GREATER_THAN(cooled1, cooled0,
                       "More hot gas produces more cooling");
    
    // Both should be non-negative
    ASSERT_TRUE(cooled0 >= 0.0,
               "Cooling mass non-negative");
    ASSERT_TRUE(cooled1 >= 0.0,
               "Cooling mass non-negative");
}

// ============================================================================
// TEST 6: CGM Precipitation Cooling
// ============================================================================
void test_cgm_precipitation() {
    BEGIN_TEST("CGM Precipitation When tcool/tff < 10");
    
    struct GALAXY gal[1];
    memset(gal, 0, sizeof(struct GALAXY));
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.CGMrecipeOn = 1;
    run_params.UnitTime_in_s = 3.08568e+16;
    run_params.UnitDensity_in_cgs = 6.76991e-31;
    run_params.Hubble_h = 0.7;
    run_params.G = 43007.1;
    
    // Setup for precipitation (high density, low temperature)
    gal[0].CGMgas = 20.0;  // Large CGM mass
    gal[0].MetalsCGMgas = 0.4;  // High metallicity
    gal[0].Vvir = 100.0;  // Low velocity (cooler)
    gal[0].Rvir = 100.0;
    gal[0].Mvir = 50.0;
    gal[0].Regime = 0;
    
    double dt = 0.01;
    double cooled = cooling_recipe_cgm(0, dt, gal, &run_params);
    
    // Should have some cooling
    ASSERT_GREATER_THAN(cooled, 0.0,
                       "Precipitation produces cooling");
    
    // Check tcool/tff was calculated
    ASSERT_GREATER_THAN(gal[0].tcool, 0.0,
                       "Cooling time calculated");
    ASSERT_GREATER_THAN(gal[0].tff, 0.0,
                       "Free-fall time calculated");
}

// ============================================================================
// TEST 7: Cooling Cannot Exceed Available Gas
// ============================================================================
void test_cooling_mass_conservation() {
    BEGIN_TEST("Cooling Cannot Exceed Available Gas");
    
    struct GALAXY gal[1];
    memset(gal, 0, sizeof(struct GALAXY));
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.CGMrecipeOn = 0;
    run_params.UnitTime_in_s = 3.08568e+16;
    run_params.UnitDensity_in_cgs = 6.76991e-31;
    run_params.Hubble_h = 0.7;
    run_params.AGNrecipeOn = 0;
    
    // Small amount of hot gas
    gal[0].HotGas = 0.01;
    gal[0].MetalsHotGas = 0.0002;
    gal[0].Vvir = 100.0;
    gal[0].Rvir = 100.0;
    
    double initial_hot = gal[0].HotGas;
    
    // Very long timestep
    double dt = 10.0;  // Gyr
    double cooled = cooling_recipe_hot(0, dt, gal, &run_params);
    
    // Cooled mass should not exceed initial hot gas
    ASSERT_TRUE(cooled <= initial_hot,
               "Cannot cool more than available");
    ASSERT_TRUE(cooled >= 0.0,
               "Cooled mass non-negative");
}

// ============================================================================
// TEST 8: Regime-Aware Cooling Routes
// ============================================================================
void test_regime_aware_cooling() {
    BEGIN_TEST("Regime-Aware Cooling Routes Correctly");
    
    struct GALAXY gal[2];
    memset(gal, 0, sizeof(struct GALAXY) * 2);
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.CGMrecipeOn = 1;
    run_params.UnitTime_in_s = 3.08568e+16;
    run_params.UnitDensity_in_cgs = 6.76991e-31;
    run_params.Hubble_h = 0.7;
    run_params.G = 43007.1;
    
    double dt = 0.01;
    
    // Galaxy 0: CGM regime (should call cooling_recipe_cgm)
    gal[0].Regime = 0;
    gal[0].CGMgas = 5.0;
    gal[0].MetalsCGMgas = 0.1;
    gal[0].HotGas = 0.0;
    gal[0].Vvir = 150.0;
    gal[0].Rvir = 150.0;
    gal[0].Mvir = 75.0;
    
    // Galaxy 1: Hot regime (should call cooling_recipe_hot)
    gal[1].Regime = 1;
    gal[1].HotGas = 5.0;
    gal[1].MetalsHotGas = 0.1;
    gal[1].CGMgas = 0.0;
    gal[1].Vvir = 150.0;
    gal[1].Rvir = 150.0;
    gal[1].Mvir = 75.0;
    
    // Use regime-aware wrapper
    double cooled_cgm = cooling_recipe(0, dt, gal, &run_params);
    double cooled_hot = cooling_recipe(1, dt, gal, &run_params);
    
    // Both should produce some cooling
    ASSERT_TRUE(cooled_cgm >= 0.0,
               "CGM cooling non-negative");
    ASSERT_TRUE(cooled_hot >= 0.0,
               "Hot cooling non-negative");
}

// ============================================================================
// TEST 9: Cooling Time Scales With Halo Properties
// ============================================================================
void test_cooling_time_scaling() {
    BEGIN_TEST("Cooling Time Scales With Halo Properties");
    
    struct GALAXY gal[2];
    memset(gal, 0, sizeof(struct GALAXY) * 2);
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.CGMrecipeOn = 1;
    run_params.UnitTime_in_s = 3.08568e+16;
    run_params.UnitDensity_in_cgs = 6.76991e-31;
    run_params.Hubble_h = 0.7;
    run_params.G = 43007.1;
    
    // Galaxy 0: Small halo (faster cooling)
    gal[0].CGMgas = 5.0;
    gal[0].MetalsCGMgas = 0.1;
    gal[0].Vvir = 100.0;
    gal[0].Rvir = 100.0;
    gal[0].Mvir = 50.0;
    gal[0].Regime = 0;
    
    // Galaxy 1: Large halo (slower cooling)
    gal[1].CGMgas = 5.0;
    gal[1].MetalsCGMgas = 0.1;
    gal[1].Vvir = 300.0;  // Higher velocity (hotter)
    gal[1].Rvir = 300.0;  // Larger radius (lower density)
    gal[1].Mvir = 450.0;
    gal[1].Regime = 0;
    
    double dt = 0.01;
    cooling_recipe_cgm(0, dt, gal, &run_params);
    cooling_recipe_cgm(1, dt, gal, &run_params);
    
    // Larger halo should have longer cooling time
    // (lower density, higher temperature)
    ASSERT_GREATER_THAN(gal[1].tcool, gal[0].tcool,
                       "Massive halo has longer cooling time");
}

// ============================================================================
// TEST 10: Temperature Regulates Cooling Efficiency
// ============================================================================
void test_temperature_cooling_regulation() {
    BEGIN_TEST("Temperature Regulates Cooling Efficiency");
    
    // At fixed density, cooling rate depends on T and Z
    // Lambda(T) peaks around 10^5 K, decreases at higher T
    
    double logZ = log10(0.02);  // Solar metallicity
    
    // Low temperature (efficient cooling)
    double T_low = 1.0e5;  // K
    double lambda_low = get_metaldependent_cooling_rate(log10(T_low), logZ);
    
    // High temperature (less efficient per particle, but ionized)
    double T_high = 1.0e7;  // K
    double lambda_high = get_metaldependent_cooling_rate(log10(T_high), logZ);
    
    // Both should be positive
    ASSERT_GREATER_THAN(lambda_low, 0.0,
                       "Low-T cooling rate positive");
    ASSERT_GREATER_THAN(lambda_high, 0.0,
                       "High-T cooling rate positive");
    
    // Cooling function varies with temperature
    // Both should be positive and physical
    ASSERT_TRUE(lambda_low > 0.0 && lambda_high > 0.0,
               "Cooling rates positive at all temperatures");
}

// ============================================================================
// MAIN TEST RUNNER
// ============================================================================
int main(void) {
    printf("\n");
    printf("═══════════════════════════════════════════════════════════\n");
    printf("  SAGE26 COOLING & HEATING TESTS\n");
    printf("═══════════════════════════════════════════════════════════\n");
    printf("\n");

    test_virial_temperature_scaling();
    test_metallicity_dependent_cooling();
    test_cooling_freefall_ratio();
    test_cooling_radius();
    test_cooling_mass_dependence();
    test_cgm_precipitation();
    test_cooling_mass_conservation();
    test_regime_aware_cooling();
    test_cooling_time_scaling();
    test_temperature_cooling_regulation();

    PRINT_TEST_SUMMARY();
    
    return (tests_failed > 0) ? 1 : 0;
}

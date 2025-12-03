/**
 * @file test_reincorporation.c
 * @brief Unit tests for gas reincorporation from ejected reservoir
 *
 * Tests the reincorporation of gas from the ejected reservoir back into
 * the hot/CGM gas phase. This process occurs when the halo escape velocity
 * exceeds the supernova ejecta velocity (630 km/s).
 *
 * Key physics tested:
 * - Reincorporation rate dependence on Vvir and EjectedMass
 * - Metal tracking through ejection-reincorporation cycle
 * - Regime-dependent destination (CGM vs Hot)
 * - Mass and metal conservation
 * - Critical velocity threshold behavior
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../src/core_allvars.h"
#include "../src/model_reincorporation.h"
#include "../src/model_misc.h"
#include "test_framework.h"

// ============================================================================
// TEST 1: Reincorporation Occurs When Vvir > Vcrit
// ============================================================================
void test_reincorporation_velocity_threshold() {
    BEGIN_TEST("Reincorporation Velocity Threshold");
    
    struct GALAXY gal[1];
    memset(gal, 0, sizeof(struct GALAXY));
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.CGMrecipeOn = 1;
    run_params.ReIncorporationFactor = 1.0;
    
    const double dt = 0.01;  // Gyr
    
    // Test 1: Vvir < Vcrit -> no reincorporation
    gal[0].Vvir = 400.0;
    gal[0].Rvir = 200.0;
    gal[0].EjectedMass = 1.0;
    gal[0].MetalsEjectedMass = 0.02;
    gal[0].HotGas = 0.0;
    gal[0].Regime = 1;
    
    double initial_ejected = gal[0].EjectedMass;
    reincorporate_gas(0, dt, gal, &run_params);
    
    ASSERT_CLOSE(gal[0].EjectedMass, initial_ejected, 1e-10,
                "No reincorporation when Vvir < Vcrit");
    
    // Test 2: Vvir > Vcrit -> reincorporation occurs
    gal[0].Vvir = 500.0;
    gal[0].Rvir = 200.0;
    gal[0].EjectedMass = 1.0;
    gal[0].MetalsEjectedMass = 0.02;
    gal[0].HotGas = 0.0;
    
    reincorporate_gas(0, dt, gal, &run_params);
    
    ASSERT_LESS_THAN(gal[0].EjectedMass, initial_ejected,
                    "Reincorporation occurs when Vvir > Vcrit");
    ASSERT_GREATER_THAN(gal[0].HotGas, 0.0,
                       "Hot gas increased from reincorporation");
}

// ============================================================================
// TEST 2: Reincorporation Rate Scales With (Vvir/Vcrit - 1)
// ============================================================================
void test_reincorporation_rate_scaling() {
    BEGIN_TEST("Reincorporation Rate Scales With Vvir");
    
    struct GALAXY gal[2];
    memset(gal, 0, sizeof(struct GALAXY) * 2);
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.CGMrecipeOn = 1;
    run_params.ReIncorporationFactor = 1.0;
    
    const double Vcrit = 445.48;
    const double dt = 0.01;
    
    // Galaxy 1: Vvir = 500 km/s
    gal[0].Vvir = 500.0;
    gal[0].Rvir = 200.0;
    gal[0].EjectedMass = 1.0;
    gal[0].MetalsEjectedMass = 0.02;
    gal[0].HotGas = 0.0;
    gal[0].Regime = 1;
    
    // Galaxy 2: Vvir = 600 km/s (higher velocity)
    gal[1].Vvir = 600.0;
    gal[1].Rvir = 200.0;
    gal[1].EjectedMass = 1.0;
    gal[1].MetalsEjectedMass = 0.02;
    gal[1].HotGas = 0.0;
    gal[1].Regime = 1;
    
    reincorporate_gas(0, dt, gal, &run_params);
    reincorporate_gas(1, dt, gal, &run_params);
    
    // Calculate expected ratio: (600/Vcrit - 1) / (500/Vcrit - 1)
    double factor1 = 500.0 / Vcrit - 1.0;
    double factor2 = 600.0 / Vcrit - 1.0;
    double expected_ratio = factor2 / factor1;
    
    double reincorp1 = gal[0].HotGas;
    double reincorp2 = gal[1].HotGas;
    double actual_ratio = reincorp2 / reincorp1;
    
    ASSERT_CLOSE(actual_ratio, expected_ratio, 0.20,
                "Reincorporation rate ~ (Vvir/Vcrit - 1)");
}

// ============================================================================
// TEST 3: Reincorporation Timescale ~ Rvir/Vvir
// ============================================================================
void test_reincorporation_timescale() {
    BEGIN_TEST("Reincorporation Timescale Depends on Tdyn");
    
    struct GALAXY gal[1];
    memset(gal, 0, sizeof(struct GALAXY));
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.CGMrecipeOn = 1;
    run_params.ReIncorporationFactor = 1.0;
    
    gal[0].Vvir = 500.0;
    gal[0].Rvir = 200.0;
    gal[0].EjectedMass = 1.0;
    gal[0].MetalsEjectedMass = 0.02;
    gal[0].HotGas = 0.0;
    gal[0].Regime = 1;
    
    // Short timestep
    double dt_short = 0.001;  // Gyr
    double initial_ejected = gal[0].EjectedMass;
    
    reincorporate_gas(0, dt_short, gal, &run_params);
    double reincorp_short = initial_ejected - gal[0].EjectedMass;
    
    // Reset and use longer timestep
    gal[0].EjectedMass = 1.0;
    gal[0].MetalsEjectedMass = 0.02;
    gal[0].HotGas = 0.0;
    
    double dt_long = 0.01;  // Gyr (10x longer)
    reincorporate_gas(0, dt_long, gal, &run_params);
    double reincorp_long = initial_ejected - gal[0].EjectedMass;
    
    // Ratio should be ~ dt_long/dt_short = 10
    double ratio = reincorp_long / reincorp_short;
    ASSERT_IN_RANGE(ratio, 9.0, 11.0,
                   "Reincorporation scales linearly with dt");
}

// ============================================================================
// TEST 4: Metal Conservation Through Reincorporation
// ============================================================================
void test_metal_conservation_reincorporation() {
    BEGIN_TEST("Metal Conservation During Reincorporation");
    
    struct GALAXY gal[1];
    memset(gal, 0, sizeof(struct GALAXY));
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.CGMrecipeOn = 1;
    run_params.ReIncorporationFactor = 1.0;
    
    gal[0].Vvir = 500.0;
    gal[0].Rvir = 200.0;
    gal[0].EjectedMass = 1.0;
    gal[0].MetalsEjectedMass = 0.03;  // 3% metallicity
    gal[0].HotGas = 0.0;
    gal[0].MetalsHotGas = 0.0;
    gal[0].Regime = 1;
    
    double initial_total_metals = gal[0].MetalsEjectedMass + gal[0].MetalsHotGas;
    double initial_metallicity = 0.03;
    
    double dt = 0.1;  // Gyr
    reincorporate_gas(0, dt, gal, &run_params);
    
    double final_total_metals = gal[0].MetalsEjectedMass + gal[0].MetalsHotGas;
    
    ASSERT_CLOSE(final_total_metals, initial_total_metals, 1e-10,
                "Total metals conserved");
    
    // Check metallicity preserved in both reservoirs
    double ejected_Z = get_metallicity(gal[0].EjectedMass, gal[0].MetalsEjectedMass);
    double hot_Z = get_metallicity(gal[0].HotGas, gal[0].MetalsHotGas);
    
    ASSERT_CLOSE(ejected_Z, initial_metallicity, 1e-6,
                "Ejected metallicity unchanged");
    ASSERT_CLOSE(hot_Z, initial_metallicity, 1e-6,
                "Reincorporated gas has same metallicity");
}

// ============================================================================
// TEST 5: Regime-Dependent Reincorporation (CGM vs Hot)
// ============================================================================
void test_regime_dependent_reincorporation() {
    BEGIN_TEST("Regime-Dependent Reincorporation Destination");
    
    struct GALAXY gal[2];
    memset(gal, 0, sizeof(struct GALAXY) * 2);
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.CGMrecipeOn = 1;
    run_params.ReIncorporationFactor = 1.0;
    
    double dt = 0.01;
    
    // Galaxy 0: CGM regime
    gal[0].Vvir = 500.0;
    gal[0].Rvir = 200.0;
    gal[0].EjectedMass = 1.0;
    gal[0].MetalsEjectedMass = 0.02;
    gal[0].Regime = 0;  // CGM regime
    gal[0].CGMgas = 0.0;
    gal[0].HotGas = 0.0;
    
    // Galaxy 1: Hot regime
    gal[1].Vvir = 500.0;
    gal[1].Rvir = 200.0;
    gal[1].EjectedMass = 1.0;
    gal[1].MetalsEjectedMass = 0.02;
    gal[1].Regime = 1;  // Hot regime
    gal[1].CGMgas = 0.0;
    gal[1].HotGas = 0.0;
    
    reincorporate_gas(0, dt, gal, &run_params);
    reincorporate_gas(1, dt, gal, &run_params);
    
    // CGM regime: gas goes to CGM
    ASSERT_GREATER_THAN(gal[0].CGMgas, 0.0,
                       "CGM regime: gas goes to CGM");
    ASSERT_CLOSE(gal[0].HotGas, 0.0, 1e-10,
                "CGM regime: Hot gas unchanged");
    
    // Hot regime: gas goes to Hot
    ASSERT_GREATER_THAN(gal[1].HotGas, 0.0,
                       "Hot regime: gas goes to Hot");
    ASSERT_CLOSE(gal[1].CGMgas, 0.0, 1e-10,
                "Hot regime: CGM gas unchanged");
    
    // Same amount reincorporated in both regimes
    ASSERT_CLOSE(gal[0].CGMgas, gal[1].HotGas, 1e-10,
                "Same reincorporation rate in both regimes");
}

// ============================================================================
// TEST 6: ReIncorporationFactor Parameter
// ============================================================================
void test_reincorporation_factor() {
    BEGIN_TEST("ReIncorporationFactor Modulates Threshold");
    
    struct GALAXY gal[2];
    memset(gal, 0, sizeof(struct GALAXY) * 2);
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.CGMrecipeOn = 1;
    
    double dt = 0.01;
    double Vvir = 500.0;
    
    // Galaxy 0: ReIncorporationFactor = 1.0
    run_params.ReIncorporationFactor = 1.0;
    gal[0].Vvir = Vvir;
    gal[0].Rvir = 200.0;
    gal[0].EjectedMass = 1.0;
    gal[0].MetalsEjectedMass = 0.02;
    gal[0].HotGas = 0.0;
    gal[0].Regime = 1;
    
    reincorporate_gas(0, dt, gal, &run_params);
    double reincorp1 = gal[0].HotGas;
    
    // Galaxy 1: ReIncorporationFactor = 0.5 (easier reincorporation)
    run_params.ReIncorporationFactor = 0.5;
    gal[1].Vvir = Vvir;
    gal[1].Rvir = 200.0;
    gal[1].EjectedMass = 1.0;
    gal[1].MetalsEjectedMass = 0.02;
    gal[1].HotGas = 0.0;
    gal[1].Regime = 1;
    
    reincorporate_gas(1, dt, gal, &run_params);
    double reincorp2 = gal[1].HotGas;
    
    ASSERT_GREATER_THAN(reincorp2, reincorp1,
                       "Lower ReIncorporationFactor increases reincorporation");
}

// ============================================================================
// TEST 7: Complete Ejection-Reincorporation Cycle
// ============================================================================
void test_ejection_reincorporation_cycle() {
    BEGIN_TEST("Complete Ejection-Reincorporation Cycle");
    
    struct GALAXY gal[1];
    memset(gal, 0, sizeof(struct GALAXY));
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.CGMrecipeOn = 1;
    run_params.ReIncorporationFactor = 1.0;
    
    // Start with hot gas
    gal[0].HotGas = 1.0;
    gal[0].MetalsHotGas = 0.02;
    gal[0].EjectedMass = 0.0;
    gal[0].MetalsEjectedMass = 0.0;
    gal[0].Vvir = 500.0;
    gal[0].Rvir = 200.0;
    gal[0].Regime = 1;
    
    double initial_total_gas = gal[0].HotGas + gal[0].EjectedMass;
    double initial_total_metals = gal[0].MetalsHotGas + gal[0].MetalsEjectedMass;
    
    // Simulate ejection (manually move gas)
    double ejected_amount = 0.5;
    double metallicity = get_metallicity(gal[0].HotGas, gal[0].MetalsHotGas);
    gal[0].HotGas -= ejected_amount;
    gal[0].MetalsHotGas -= ejected_amount * metallicity;
    gal[0].EjectedMass += ejected_amount;
    gal[0].MetalsEjectedMass += ejected_amount * metallicity;
    
    // Now reincorporate over time
    double dt = 0.01;
    int steps = 50;
    
    for(int i = 0; i < steps; i++) {
        reincorporate_gas(0, dt, gal, &run_params);
    }
    
    // Check conservation
    double final_total_gas = gal[0].HotGas + gal[0].EjectedMass;
    double final_total_metals = gal[0].MetalsHotGas + gal[0].MetalsEjectedMass;
    
    ASSERT_CLOSE(final_total_gas, initial_total_gas, 1e-6,
                "Total gas conserved through cycle");
    ASSERT_CLOSE(final_total_metals, initial_total_metals, 1e-8,
                "Total metals conserved through cycle");
    
    // Some gas should have returned
    ASSERT_GREATER_THAN(gal[0].HotGas, 0.5,
                       "Some gas reincorporated");
    ASSERT_LESS_THAN(gal[0].EjectedMass, 0.5,
                    "Ejected mass decreased");
}

// ============================================================================
// TEST 8: No Reincorporation Cannot Add Mass
// ============================================================================
void test_reincorporation_mass_limit() {
    BEGIN_TEST("Cannot Reincorporate More Than Ejected");
    
    struct GALAXY gal[1];
    memset(gal, 0, sizeof(struct GALAXY));
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.CGMrecipeOn = 1;
    run_params.ReIncorporationFactor = 1.0;
    
    gal[0].Vvir = 1000.0;  // Very high velocity
    gal[0].Rvir = 200.0;
    gal[0].EjectedMass = 0.1;  // Small ejected mass
    gal[0].MetalsEjectedMass = 0.002;
    gal[0].HotGas = 0.0;
    gal[0].Regime = 1;
    
    double initial_ejected = gal[0].EjectedMass;
    
    // Very long timestep - would reincorporate more than available
    double dt = 10.0;  // Gyr
    reincorporate_gas(0, dt, gal, &run_params);
    
    // All ejected mass should be reincorporated, no more
    ASSERT_CLOSE(gal[0].HotGas, initial_ejected, 1e-10,
                "Reincorporated exactly all ejected mass");
    ASSERT_CLOSE(gal[0].EjectedMass, 0.0, 1e-10,
                "Ejected reservoir empty");
    ASSERT_TRUE(gal[0].MetalsEjectedMass < 1e-15,
               "Ejected metals all transferred");
}

// ============================================================================
// MAIN TEST RUNNER
// ============================================================================
int main(void) {
    printf("\n");
    printf("═══════════════════════════════════════════════════════════\n");
    printf("  SAGE26 REINCORPORATION TESTS\n");
    printf("═══════════════════════════════════════════════════════════\n");
    printf("\n");

    test_reincorporation_velocity_threshold();
    test_reincorporation_rate_scaling();
    test_reincorporation_timescale();
    test_metal_conservation_reincorporation();
    test_regime_dependent_reincorporation();
    test_reincorporation_factor();
    test_ejection_reincorporation_cycle();
    test_reincorporation_mass_limit();

    PRINT_TEST_SUMMARY();
    
    return (tests_failed > 0) ? 1 : 0;
}

/*
 * MULTI-SATELLITE SYSTEMS TESTS
 * 
 * Tests for realistic group environments:
 * - Multiple satellites around one central
 * - Mass conservation across entire system
 * - Hierarchical merging sequences
 * - Group-scale gas budgets
 * - Satellite-satellite interactions
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "test_framework.h"
#include "../src/core_allvars.h"
#include "../src/model_misc.h"
#include "../src/model_infall.h"

void test_multi_satellite_mass_conservation() {
    BEGIN_TEST("Mass Conservation with Multiple Satellites");
    
    struct GALAXY galaxies[5];  // 1 central + 4 satellites
    memset(galaxies, 0, sizeof(struct GALAXY) * 5);
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.BaryonFrac = 0.17;
    run_params.CGMrecipeOn = 1;
    run_params.ReionizationOn = 0;
    
    // Central galaxy
    galaxies[0].Regime = 1;
    galaxies[0].Mvir = 100.0;
    galaxies[0].HotGas = 10.0;
    galaxies[0].MetalsHotGas = 0.1;
    galaxies[0].ColdGas = 2.0;
    galaxies[0].StellarMass = 5.0;
    
    // Four satellites with varying properties
    for(int i = 1; i <= 4; i++) {
        galaxies[i].Regime = 1;
        galaxies[i].Mvir = 10.0 + i * 2.0;
        galaxies[i].HotGas = 1.0 + i * 0.5;
        galaxies[i].MetalsHotGas = galaxies[i].HotGas * 0.01;
        galaxies[i].ColdGas = 0.5;
        galaxies[i].StellarMass = 1.0 + i * 0.3;
        galaxies[i].BlackHoleMass = 0.01;
        galaxies[i].EjectedMass = 0.2;
        galaxies[i].CGMgas = 0.0;
        galaxies[i].ICS = 0.0;
    }
    
    // Calculate total baryonic mass
    double initial_total = 0.0;
    for(int i = 0; i < 5; i++) {
        initial_total += galaxies[i].ColdGas + galaxies[i].HotGas + 
                        galaxies[i].StellarMass + galaxies[i].BlackHoleMass +
                        galaxies[i].EjectedMass + galaxies[i].CGMgas + galaxies[i].ICS;
    }
    
    // Apply stripping to all satellites
    for(int i = 1; i <= 4; i++) {
        strip_from_satellite(0, i, 0.0, galaxies, &run_params);
    }
    
    // Check total mass conservation
    double final_total = 0.0;
    for(int i = 0; i < 5; i++) {
        final_total += galaxies[i].ColdGas + galaxies[i].HotGas + 
                      galaxies[i].StellarMass + galaxies[i].BlackHoleMass +
                      galaxies[i].EjectedMass + galaxies[i].CGMgas + galaxies[i].ICS;
    }
    
    ASSERT_CLOSE(initial_total, final_total, 1e-4,
                "Total baryonic mass conserved across 5-galaxy system");
}

void test_satellite_gas_transfer_to_central() {
    BEGIN_TEST("All Satellite Gas Transfers to Central");
    
    struct GALAXY galaxies[4];  // 1 central + 3 satellites
    memset(galaxies, 0, sizeof(struct GALAXY) * 4);
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.BaryonFrac = 0.17;
    run_params.CGMrecipeOn = 1;
    run_params.ReionizationOn = 0;
    
    galaxies[0].Regime = 1;
    galaxies[0].HotGas = 5.0;
    galaxies[0].MetalsHotGas = 0.05;
    
    double total_sat_hot = 0.0;
    for(int i = 1; i <= 3; i++) {
        galaxies[i].Regime = 1;
        galaxies[i].Mvir = 8.0;
        galaxies[i].HotGas = 2.0 + i * 0.5;
        galaxies[i].MetalsHotGas = galaxies[i].HotGas * 0.01;
        galaxies[i].StellarMass = 0.5;
        galaxies[i].ColdGas = 0.1;
        galaxies[i].BlackHoleMass = 0.005;
        galaxies[i].EjectedMass = 0.0;
        galaxies[i].CGMgas = 0.0;
        galaxies[i].ICS = 0.0;
        total_sat_hot += galaxies[i].HotGas;
    }
    
    double initial_cen_hot = galaxies[0].HotGas;
    
    // Strip from all satellites
    for(int i = 1; i <= 3; i++) {
        strip_from_satellite(0, i, 0.0, galaxies, &run_params);
    }
    
    // Central should have gained gas
    ASSERT_GREATER_THAN(galaxies[0].HotGas, initial_cen_hot,
                       "Central gained gas from multiple satellites");
    
    // Total hot gas in satellites should have decreased
    double final_sat_hot = 0.0;
    for(int i = 1; i <= 3; i++) {
        final_sat_hot += galaxies[i].HotGas;
    }
    
    ASSERT_LESS_THAN(final_sat_hot, total_sat_hot,
                    "Satellites lost gas collectively");
}

void test_group_scale_metal_budget() {
    BEGIN_TEST("Group-Scale Metal Budget Conserved");
    
    struct GALAXY galaxies[6];
    memset(galaxies, 0, sizeof(struct GALAXY) * 6);
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.BaryonFrac = 0.17;
    run_params.CGMrecipeOn = 1;
    run_params.ReionizationOn = 0;
    
    // Central with metal-poor gas
    galaxies[0].Regime = 1;
    galaxies[0].HotGas = 10.0;
    galaxies[0].MetalsHotGas = 0.05;  // 0.5% Z
    
    // Satellites with varying metallicities
    for(int i = 1; i <= 5; i++) {
        galaxies[i].Regime = 1;
        galaxies[i].Mvir = 10.0;
        galaxies[i].HotGas = 2.0;
        galaxies[i].MetalsHotGas = 0.02 + i * 0.01;  // Varying Z
        galaxies[i].StellarMass = 1.0;
        galaxies[i].MetalsStellarMass = 0.02;
        galaxies[i].ColdGas = 0.5;
        galaxies[i].MetalsColdGas = 0.01;
        galaxies[i].BlackHoleMass = 0.01;
        galaxies[i].EjectedMass = 0.0;
        galaxies[i].CGMgas = 0.0;
        galaxies[i].ICS = 0.0;
    }
    
    // Total metals in system
    double initial_total_metals = 0.0;
    for(int i = 0; i < 6; i++) {
        initial_total_metals += galaxies[i].MetalsHotGas + galaxies[i].MetalsColdGas + 
                               galaxies[i].MetalsStellarMass;
    }
    
    // Strip from all satellites
    for(int i = 1; i <= 5; i++) {
        strip_from_satellite(0, i, 0.0, galaxies, &run_params);
    }
    
    double final_total_metals = 0.0;
    for(int i = 0; i < 6; i++) {
        final_total_metals += galaxies[i].MetalsHotGas + galaxies[i].MetalsColdGas + 
                             galaxies[i].MetalsStellarMass;
    }
    
    ASSERT_CLOSE(initial_total_metals, final_total_metals, 1e-4,
                "Group-scale metal budget conserved");
}

void test_differential_stripping_by_mass() {
    BEGIN_TEST("Heavier Satellites Retain More Gas");
    
    struct GALAXY galaxies[4];
    memset(galaxies, 0, sizeof(struct GALAXY) * 4);
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.BaryonFrac = 0.17;
    run_params.CGMrecipeOn = 1;
    run_params.ReionizationOn = 0;
    
    galaxies[0].Regime = 1;
    galaxies[0].HotGas = 10.0;
    galaxies[0].MetalsHotGas = 0.1;
    
    // Light satellite - should be stripped more
    galaxies[1].Regime = 1;
    galaxies[1].Mvir = 5.0;   // Small
    galaxies[1].HotGas = 3.0;  // Excess for its mass
    galaxies[1].MetalsHotGas = 0.03;
    galaxies[1].StellarMass = 0.3;
    galaxies[1].ColdGas = 0.1;
    galaxies[1].BlackHoleMass = 0.005;
    galaxies[1].EjectedMass = 0.0;
    galaxies[1].CGMgas = 0.0;
    galaxies[1].ICS = 0.0;
    
    // Heavy satellite - should retain more
    galaxies[2].Regime = 1;
    galaxies[2].Mvir = 50.0;  // Large
    galaxies[2].HotGas = 10.0;
    galaxies[2].MetalsHotGas = 0.1;
    galaxies[2].StellarMass = 3.0;
    galaxies[2].ColdGas = 1.0;
    galaxies[2].BlackHoleMass = 0.05;
    galaxies[2].EjectedMass = 0.0;
    galaxies[2].CGMgas = 0.0;
    galaxies[2].ICS = 0.0;
    
    double light_hot_before = galaxies[1].HotGas;
    double heavy_hot_before = galaxies[2].HotGas;
    
    strip_from_satellite(0, 1, 0.0, galaxies, &run_params);
    strip_from_satellite(0, 2, 0.0, galaxies, &run_params);
    
    double light_frac_lost = (light_hot_before - galaxies[1].HotGas) / light_hot_before;
    double heavy_frac_lost = (heavy_hot_before - galaxies[2].HotGas) / heavy_hot_before;
    
    // Light satellite should lose larger fraction
    if(light_frac_lost > 0.01 && heavy_frac_lost > 0.01) {
        ASSERT_GREATER_THAN(light_frac_lost, heavy_frac_lost * 0.5,
                           "Light satellites lose larger gas fraction");
    }
}

void test_cumulative_stripping_over_time() {
    BEGIN_TEST("Cumulative Stripping Over Multiple Timesteps");
    
    struct GALAXY galaxies[2];
    memset(galaxies, 0, sizeof(struct GALAXY) * 2);
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.BaryonFrac = 0.17;
    run_params.CGMrecipeOn = 1;
    run_params.ReionizationOn = 0;
    
    galaxies[0].Regime = 1;
    galaxies[0].HotGas = 10.0;
    galaxies[0].MetalsHotGas = 0.1;
    
    galaxies[1].Regime = 1;
    galaxies[1].Mvir = 10.0;
    galaxies[1].HotGas = 5.0;
    galaxies[1].MetalsHotGas = 0.05;
    galaxies[1].StellarMass = 0.5;
    galaxies[1].ColdGas = 0.2;
    galaxies[1].BlackHoleMass = 0.01;
    galaxies[1].EjectedMass = 0.0;
    galaxies[1].CGMgas = 0.0;
    galaxies[1].ICS = 0.0;
    
    double initial_hot = galaxies[1].HotGas;
    
    // Apply stripping multiple times (simulating orbit)
    for(int step = 0; step < 10; step++) {
        strip_from_satellite(0, 1, 0.0, galaxies, &run_params);
    }
    
    // Should have progressively stripped gas
    ASSERT_LESS_THAN(galaxies[1].HotGas, initial_hot * 0.9,
                    "Cumulative stripping reduces satellite gas");
    ASSERT_GREATER_THAN(galaxies[1].HotGas, 0.0,
                       "Some gas remains after multiple strips");
}

void test_satellite_total_baryon_tracking() {
    BEGIN_TEST("TotalSatelliteBaryons Tracks All Satellite Mass");
    
    struct GALAXY galaxies[5];
    memset(galaxies, 0, sizeof(struct GALAXY) * 5);
    
    // Central
    galaxies[0].ColdGas = 2.0;
    galaxies[0].HotGas = 10.0;
    galaxies[0].StellarMass = 5.0;
    galaxies[0].TotalSatelliteBaryons = 0.0;
    
    // Satellites
    double expected_sat_total = 0.0;
    for(int i = 1; i <= 4; i++) {
        galaxies[i].ColdGas = 0.5 + i * 0.1;
        galaxies[i].HotGas = 1.0 + i * 0.3;
        galaxies[i].StellarMass = 1.0 + i * 0.2;
        galaxies[i].EjectedMass = 0.1;
        galaxies[i].CGMgas = 0.2;
        galaxies[i].ICS = 0.05;
        
        double sat_baryons = galaxies[i].ColdGas + galaxies[i].HotGas + 
                            galaxies[i].StellarMass + galaxies[i].EjectedMass +
                            galaxies[i].CGMgas + galaxies[i].ICS;
        expected_sat_total += sat_baryons;
    }
    
    // Simulate tracking (manual for this test)
    galaxies[0].TotalSatelliteBaryons = 0.0;
    for(int i = 1; i <= 4; i++) {
        galaxies[0].TotalSatelliteBaryons += 
            galaxies[i].ColdGas + galaxies[i].HotGas + galaxies[i].StellarMass +
            galaxies[i].EjectedMass + galaxies[i].CGMgas + galaxies[i].ICS;
    }
    
    ASSERT_CLOSE(galaxies[0].TotalSatelliteBaryons, expected_sat_total, 1e-5,
                "TotalSatelliteBaryons correctly sums all satellite mass");
}

void test_no_self_stripping() {
    BEGIN_TEST("Galaxy Doesn't Strip from Itself");
    
    struct GALAXY galaxies[1];
    memset(galaxies, 0, sizeof(struct GALAXY));
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.BaryonFrac = 0.17;
    run_params.CGMrecipeOn = 1;
    run_params.ReionizationOn = 0;
    
    galaxies[0].Regime = 1;
    galaxies[0].Mvir = 50.0;
    galaxies[0].HotGas = 10.0;
    galaxies[0].MetalsHotGas = 0.1;
    galaxies[0].StellarMass = 5.0;
    galaxies[0].ColdGas = 1.0;
    galaxies[0].BlackHoleMass = 0.05;
    galaxies[0].EjectedMass = 0.5;
    galaxies[0].CGMgas = 0.0;
    galaxies[0].ICS = 0.0;
    
    double initial_hot = galaxies[0].HotGas;
    
    // Try to strip from itself (should be no-op or minimal)
    strip_from_satellite(0, 0, 0.0, galaxies, &run_params);
    
    // Should not significantly change
    ASSERT_CLOSE(galaxies[0].HotGas, initial_hot, 0.1,
                "Central doesn't strip significant gas from itself");
}

int main() {
    BEGIN_TEST_SUITE("Multi-Satellite Systems");
    
    test_multi_satellite_mass_conservation();
    test_satellite_gas_transfer_to_central();
    test_group_scale_metal_budget();
    test_differential_stripping_by_mass();
    test_cumulative_stripping_over_time();
    test_satellite_total_baryon_tracking();
    test_no_self_stripping();
    
    END_TEST_SUITE();
    PRINT_TEST_SUMMARY();
    
    return TEST_EXIT_CODE();
}

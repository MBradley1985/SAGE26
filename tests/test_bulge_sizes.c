/*
 * BULGE SIZE PHYSICS TESTS
 * 
 * Tests for:
 * - Bulge size scaling relations (Shen et al. 2003)
 * - Separate tracking of merger vs instability bulges (Tonini et al. 2016)
 * - Radius evolution during disk instability
 * - Mass-weighted radius combination
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "test_framework.h"
#include "../src/core_allvars.h"
#include "../src/model_misc.h"
#include "../src/model_disk_instability.h"

void test_shen_scaling_relation() {
    BEGIN_TEST("Shen et al. 2003 Scaling Relation");
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.Hubble_h = 0.7;
    run_params.BulgeSizeOn = 1;  // Simple Shen relation
    
    // Test the scaling: log(R/kpc) = 0.56 log(M/Msun) - 5.54
    // For M = 1e11 Msun: log(R) = 0.56 × 11 - 5.54 = 0.62 → R = 4.17 kpc
    
    struct GALAXY gal;
    memset(&gal, 0, sizeof(struct GALAXY));
    
    // M = 1e11 Msun = 10 × 10^10 Msun/h (with h=0.7) = 14.3 code units
    double M_target_Msun = 1e11;
    gal.BulgeMass = M_target_Msun / (1e10 / run_params.Hubble_h);
    
    double R_bulge = get_bulge_radius(0, &gal, &run_params);
    
    // Expected: R = 4.17 kpc = 0.00417 Mpc/h = 0.00292 Mpc (for h=0.7)
    double expected_R_Mpc_h = 4.17e-3 * run_params.Hubble_h;  // kpc → Mpc/h
    
    ASSERT_CLOSE(expected_R_Mpc_h, R_bulge, 0.05, 
                 "Bulge radius matches Shen+2003 for M=1e11 Msun");
    
    // Test that radius scales as M^0.56
    gal.BulgeMass *= 10.0;  // Increase mass by 10x
    double R_bulge_10x = get_bulge_radius(0, &gal, &run_params);
    
    double expected_ratio = pow(10.0, 0.56);  // = 3.63
    double actual_ratio = R_bulge_10x / R_bulge;
    
    ASSERT_CLOSE(expected_ratio, actual_ratio, 0.05,
                 "Radius scales as M^0.56");
}

void test_two_regime_shen() {
    BEGIN_TEST("Two-Regime Shen Relation (Dwarfs vs Giants)");
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.Hubble_h = 0.7;
    run_params.BulgeSizeOn = 2;  // Two-regime Shen
    
    struct GALAXY gal;
    memset(&gal, 0, sizeof(struct GALAXY));
    
    // Test low-mass regime (M < 2e10 Msun)
    // log(R/kpc) = 0.14 log(M/Msun) - 1.21
    double M_dwarf_Msun = 1e10;
    gal.BulgeMass = M_dwarf_Msun / (1e10 / run_params.Hubble_h);
    
    double R_dwarf = get_bulge_radius(0, &gal, &run_params);
    
    // Expected: log(R) = 0.14 × 10 - 1.21 = 0.19 → R = 1.55 kpc
    double expected_R_dwarf = 1.55e-3 * run_params.Hubble_h;
    
    ASSERT_CLOSE(expected_R_dwarf, R_dwarf, 0.05,
                 "Dwarf bulge radius in low-mass regime");
    
    // Test high-mass regime (M > 2e10 Msun)
    // log(R/kpc) = 0.56 log(M/Msun) - 5.54
    double M_giant_Msun = 1e11;
    gal.BulgeMass = M_giant_Msun / (1e10 / run_params.Hubble_h);
    
    double R_giant = get_bulge_radius(0, &gal, &run_params);
    
    // Expected: log(R) = 0.56 × 11 - 5.54 = 0.62 → R = 4.17 kpc
    double expected_R_giant = 4.17e-3 * run_params.Hubble_h;
    
    ASSERT_CLOSE(expected_R_giant, R_giant, 0.05,
                 "Giant bulge radius in high-mass regime");
    
    // Different slopes mean ratio changes with mass
    ASSERT_TRUE(R_giant / R_dwarf > 2.0, "Giants have larger radii than dwarfs");
}

void test_tonini_separate_bulges() {
    BEGIN_TEST("Tonini et al. 2016 - Separate Merger/Instability Bulges");
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.Hubble_h = 0.7;
    run_params.BulgeSizeOn = 3;  // Tonini mode
    
    struct GALAXY gal;
    memset(&gal, 0, sizeof(struct GALAXY));
    
    // Set up a galaxy with both merger and instability bulges
    gal.MergerBulgeMass = 0.3;       // 3e10 Msun/h
    gal.InstabilityBulgeMass = 0.2;  // 2e10 Msun/h
    gal.BulgeMass = 0.5;             // Total
    
    // Set component radii
    gal.MergerBulgeRadius = 0.005;      // 5 kpc/h
    gal.InstabilityBulgeRadius = 0.003; // 3 kpc/h
    gal.DiskScaleRadius = 0.015;        // 15 kpc/h
    
    double R_combined = get_bulge_radius(0, &gal, &run_params);
    
    // Expected: R = (M_merger × R_merger + M_inst × R_inst) / M_total
    //             = (0.3 × 0.005 + 0.2 × 0.003) / 0.5
    //             = (0.0015 + 0.0006) / 0.5 = 0.0042
    double expected_R = (0.3 * 0.005 + 0.2 * 0.003) / 0.5;
    
    ASSERT_CLOSE(expected_R, R_combined, 0.01,
                 "Combined bulge radius is mass-weighted average");
    
    // Merger component should dominate since it's more massive
    ASSERT_TRUE(R_combined > gal.InstabilityBulgeRadius,
               "Combined radius > instability radius (merger-dominated)");
}

void test_instability_bulge_growth() {
    BEGIN_TEST("Instability Bulge Radius Evolution (Tonini eq. 15)");
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.Hubble_h = 0.7;
    run_params.BulgeSizeOn = 3;
    
    struct GALAXY gal;
    memset(&gal, 0, sizeof(struct GALAXY));
    
    // Initial state
    gal.InstabilityBulgeMass = 0.1;
    gal.InstabilityBulgeRadius = 0.002;  // 2 kpc/h
    gal.DiskScaleRadius = 0.015;         // 15 kpc/h
    
    double M_old = gal.InstabilityBulgeMass;
    double R_old = gal.InstabilityBulgeRadius;
    
    // Add mass from disk instability
    double delta_mass = 0.05;
    gal.InstabilityBulgeMass += delta_mass;
    
    update_instability_bulge_radius(0, delta_mass, &gal, &run_params);
    
    // Expected: R_new = (R_old × M_old + ΔM × 0.2 × R_disk) / M_new
    double expected_R = (R_old * M_old + delta_mass * 0.2 * gal.DiskScaleRadius) / 
                        (M_old + delta_mass);
    
    ASSERT_CLOSE(expected_R, gal.InstabilityBulgeRadius, 0.01,
                 "Instability bulge radius updates correctly");
    
    // Radius should increase when mass is added
    ASSERT_GREATER_THAN(gal.InstabilityBulgeRadius, R_old,
                       "Bulge radius increases with added mass");
}

void test_zero_mass_bulges() {
    BEGIN_TEST("Zero-Mass Bulge Handling");
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.Hubble_h = 0.7;
    
    struct GALAXY gal;
    memset(&gal, 0, sizeof(struct GALAXY));
    
    // Test all modes with zero mass
    for(int mode = 0; mode <= 3; mode++) {
        run_params.BulgeSizeOn = mode;
        gal.BulgeMass = 0.0;
        gal.MergerBulgeMass = 0.0;
        gal.InstabilityBulgeMass = 0.0;
        
        double R = get_bulge_radius(0, &gal, &run_params);
        
        char msg[128];
        snprintf(msg, sizeof(msg), "Mode %d: Zero mass → zero radius", mode);
        ASSERT_EQUAL_FLOAT(0.0, R, msg);
    }
}

void test_disk_shrinking_after_instability() {
    BEGIN_TEST("Disk Radius Shrinks After Instability");
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.Hubble_h = 0.7;
    run_params.G = 43.0;
    run_params.DiskInstabilityOn = 1;
    run_params.BulgeSizeOn = 3;
    
    struct GALAXY gal;
    memset(&gal, 0, sizeof(struct GALAXY));
    
    // Set up an unstable disk
    gal.ColdGas = 0.3;
    gal.StellarMass = 1.0;
    gal.BulgeMass = 0.2;
    gal.MetalsColdGas = 0.006;
    gal.MetalsStellarMass = 0.02;
    gal.MetalsBulgeMass = 0.004;
    gal.DiskScaleRadius = 0.020;  // 20 kpc/h
    gal.Vmax = 100.0;             // Low Vmax → unstable
    
    double initial_disk_radius = gal.DiskScaleRadius;
    double initial_disk_mass = gal.ColdGas + (gal.StellarMass - gal.BulgeMass);
    
    // Trigger instability
    check_disk_instability(0, 0, 0, 0.0, 0.001, 0, &gal, &run_params);
    
    double final_disk_mass = gal.ColdGas + (gal.StellarMass - gal.BulgeMass);
    
    // If mass was transferred, disk radius should shrink
    if(final_disk_mass < initial_disk_mass) {
        ASSERT_LESS_THAN(gal.DiskScaleRadius, initial_disk_radius,
                        "Disk radius shrinks when mass moves to bulge");
        
        // Check angular momentum conservation approximately
        // R_new / R_old ≈ M_new / M_old
        double expected_ratio = final_disk_mass / initial_disk_mass;
        double actual_ratio = gal.DiskScaleRadius / initial_disk_radius;
        
        ASSERT_CLOSE(expected_ratio, actual_ratio, 0.1,
                    "Disk radius scales with mass (angular momentum)");
    }
}

int main() {
    BEGIN_TEST_SUITE("Bulge Size Physics");
    
    test_shen_scaling_relation();
    test_two_regime_shen();
    test_tonini_separate_bulges();
    test_instability_bulge_growth();
    test_zero_mass_bulges();
    test_disk_shrinking_after_instability();
    
    END_TEST_SUITE();
    PRINT_TEST_SUMMARY();
    
    return TEST_EXIT_CODE();
}

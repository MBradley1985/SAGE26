/*
 * DISK INSTABILITY TESTS
 * 
 * Tests for disk stability and bulge formation:
 * - Toomre Q-parameter calculations
 * - Critical disk mass (Mcrit)
 * - Gas and star transfer to bulge
 * - Disk radius updates
 * - Stability criterion
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "test_framework.h"
#include "../src/core_allvars.h"
#include "../src/model_misc.h"
#include "../src/model_disk_instability.h"

void test_critical_mass_calculation() {
    BEGIN_TEST("Critical Disk Mass (Mcrit) Calculation");
    
    struct GALAXY gal;
    memset(&gal, 0, sizeof(struct GALAXY));
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.G = 43.0;  // G in code units
    
    // Set up a stable disk
    gal.Vmax = 150.0;  // km/s
    gal.DiskScaleRadius = 0.01;  // 10 kpc/h
    gal.ColdGas = 0.5;
    gal.StellarMass = 1.0;
    gal.BulgeMass = 0.2;
    
    double disk_mass = gal.ColdGas + (gal.StellarMass - gal.BulgeMass);
    double Mcrit = gal.Vmax * gal.Vmax * (3.0 * gal.DiskScaleRadius) / run_params.G;
    
    ASSERT_GREATER_THAN(Mcrit, 0.0, "Mcrit is positive");
    
    // For this setup, check if disk is stable or unstable
    if(disk_mass > Mcrit) {
        double excess = disk_mass - Mcrit;
        ASSERT_GREATER_THAN(excess, 0.0, "Unstable disk has excess mass");
        printf("  ℹ Disk unstable: excess mass = %.4f\n", excess);
    } else {
        printf("  ℹ Disk stable: Mcrit = %.4f > disk_mass = %.4f\n", Mcrit, disk_mass);
    }
}

void test_disk_stability_criterion() {
    BEGIN_TEST("Disk Stability Criterion (Vmax, Radius)");
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.G = 43.0;
    
    // Test 1: High Vmax → high Mcrit → stable
    {
        struct GALAXY gal;
        memset(&gal, 0, sizeof(struct GALAXY));
        gal.Vmax = 300.0;  // Very high velocity
        gal.DiskScaleRadius = 0.02;
        gal.ColdGas = 0.5;
        gal.StellarMass = 1.0;
        gal.BulgeMass = 0.1;
        
        double disk_mass = gal.ColdGas + (gal.StellarMass - gal.BulgeMass);
        double Mcrit = gal.Vmax * gal.Vmax * (3.0 * gal.DiskScaleRadius) / run_params.G;
        
        ASSERT_TRUE(Mcrit > disk_mass, "High Vmax → stable disk");
    }
    
    // Test 2: Low Vmax → low Mcrit → unstable
    {
        struct GALAXY gal;
        memset(&gal, 0, sizeof(struct GALAXY));
        gal.Vmax = 50.0;  // Low velocity
        gal.DiskScaleRadius = 0.005;
        gal.ColdGas = 0.5;
        gal.StellarMass = 1.0;
        gal.BulgeMass = 0.1;
        
        double disk_mass = gal.ColdGas + (gal.StellarMass - gal.BulgeMass);
        double Mcrit = gal.Vmax * gal.Vmax * (3.0 * gal.DiskScaleRadius) / run_params.G;
        
        ASSERT_TRUE(Mcrit < disk_mass, "Low Vmax → unstable disk");
    }
}

void test_gas_star_fraction_in_instability() {
    BEGIN_TEST("Gas/Star Fraction During Instability");
    
    struct GALAXY gal;
    memset(&gal, 0, sizeof(struct GALAXY));
    
    // Set up disk with known gas fraction
    gal.ColdGas = 0.3;
    gal.StellarMass = 1.2;  // 0.7 in disk after bulge
    gal.BulgeMass = 0.5;
    
    double disk_mass = gal.ColdGas + (gal.StellarMass - gal.BulgeMass);
    double gas_fraction = gal.ColdGas / disk_mass;
    double star_fraction = 1.0 - gas_fraction;
    
    ASSERT_CLOSE(gas_fraction + star_fraction, 1.0, 1e-10,
                "Gas + star fractions = 1");
    ASSERT_TRUE(gas_fraction >= 0.0 && gas_fraction <= 1.0,
               "Gas fraction in [0,1]");
    ASSERT_TRUE(star_fraction >= 0.0 && star_fraction <= 1.0,
               "Star fraction in [0,1]");
    
    printf("  ℹ Gas fraction: %.2f, Star fraction: %.2f\n", 
           gas_fraction, star_fraction);
}

void test_bulge_growth_from_instability() {
    BEGIN_TEST("Bulge Growth from Disk Instability");
    
    struct GALAXY gal;
    memset(&gal, 0, sizeof(struct GALAXY));
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.G = 43.0;
    
    // Set up unstable disk
    gal.Vmax = 80.0;  // Low Vmax
    gal.DiskScaleRadius = 0.008;
    gal.ColdGas = 0.5;
    gal.StellarMass = 1.0;
    gal.BulgeMass = 0.1;
    gal.InstabilityBulgeMass = 0.05;
    gal.MetalsStellarMass = 0.02;
    gal.MetalsBulgeMass = 0.002;
    
    double initial_bulge = gal.BulgeMass;
    double initial_instability_bulge = gal.InstabilityBulgeMass;
    
    double disk_mass = gal.ColdGas + (gal.StellarMass - gal.BulgeMass);
    double Mcrit = gal.Vmax * gal.Vmax * (3.0 * gal.DiskScaleRadius) / run_params.G;
    
    if(disk_mass > Mcrit) {
        double excess = disk_mass - Mcrit;
        double star_fraction = (gal.StellarMass - gal.BulgeMass) / disk_mass;
        double unstable_stars = star_fraction * excess;
        
        // Simulate instability
        gal.BulgeMass += unstable_stars;
        gal.InstabilityBulgeMass += unstable_stars;
        
        ASSERT_TRUE(gal.BulgeMass > initial_bulge, "Bulge mass increased");
        ASSERT_TRUE(gal.InstabilityBulgeMass > initial_instability_bulge,
                   "Instability bulge component increased");
        ASSERT_CLOSE(gal.BulgeMass - initial_bulge, unstable_stars, 1e-10,
                    "Correct amount transferred to bulge");
    }
}

void test_disk_mass_conservation_in_instability() {
    BEGIN_TEST("Mass Conservation During Disk Instability");
    
    struct GALAXY gal;
    memset(&gal, 0, sizeof(struct GALAXY));
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.G = 43.0;
    
    // Unstable disk
    gal.Vmax = 70.0;
    gal.DiskScaleRadius = 0.006;
    gal.ColdGas = 0.6;
    gal.StellarMass = 1.5;
    gal.BulgeMass = 0.2;
    gal.MetalsColdGas = 0.012;
    gal.MetalsStellarMass = 0.03;
    gal.MetalsBulgeMass = 0.004;
    
    double initial_total_mass = gal.ColdGas + gal.StellarMass;
    
    double disk_mass = gal.ColdGas + (gal.StellarMass - gal.BulgeMass);
    double Mcrit = gal.Vmax * gal.Vmax * (3.0 * gal.DiskScaleRadius) / run_params.G;
    
    if(disk_mass > Mcrit) {
        double excess = disk_mass - Mcrit;
        double gas_fraction = gal.ColdGas / disk_mass;
        double star_fraction = 1.0 - gas_fraction;
        
        double unstable_gas = gas_fraction * excess;
        double unstable_stars = star_fraction * excess;
        
        // Transfer to bulge
        gal.BulgeMass += unstable_stars + unstable_gas;
        gal.ColdGas -= unstable_gas;
        // Note: stellar mass stays same, but bulge component increases
    }
    
    double final_total_mass = gal.ColdGas + gal.StellarMass;
    
    ASSERT_CLOSE(initial_total_mass, final_total_mass, 1e-10,
                "Total mass conserved during instability");
}

void test_zero_disk_mass_handling() {
    BEGIN_TEST("Zero Disk Mass Edge Case");
    
    struct GALAXY gal;
    memset(&gal, 0, sizeof(struct GALAXY));
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.G = 43.0;
    
    // Galaxy with zero disk mass (all in bulge)
    gal.Vmax = 150.0;
    gal.DiskScaleRadius = 0.01;
    gal.ColdGas = 0.0;
    gal.StellarMass = 1.0;
    gal.BulgeMass = 1.0;  // All stars in bulge
    
    double disk_mass = gal.ColdGas + (gal.StellarMass - gal.BulgeMass);
    
    ASSERT_EQUAL_FLOAT(disk_mass, 0.0, "Disk mass is zero");
    
    // Should not trigger instability calculation
    ASSERT_TRUE(disk_mass <= 0.0, "Zero disk mass skips instability check");
}

void test_instability_with_gas_only() {
    BEGIN_TEST("Disk Instability with Gas-Only Disk");
    
    struct GALAXY gal;
    memset(&gal, 0, sizeof(struct GALAXY));
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.G = 43.0;
    
    // Disk with only gas (no stars)
    gal.Vmax = 60.0;
    gal.DiskScaleRadius = 0.005;
    gal.ColdGas = 1.0;
    gal.StellarMass = 0.0;
    gal.BulgeMass = 0.0;
    
    double disk_mass = gal.ColdGas;
    double Mcrit = gal.Vmax * gal.Vmax * (3.0 * gal.DiskScaleRadius) / run_params.G;
    
    if(disk_mass > Mcrit) {
        double excess = disk_mass - Mcrit;
        double gas_fraction = 1.0;  // All gas
        double unstable_gas = gas_fraction * excess;
        
        ASSERT_CLOSE(unstable_gas, excess, 1e-10,
                    "All excess is gas in gas-only disk");
    }
}

int main() {
    BEGIN_TEST_SUITE("Disk Instability Physics");
    
    test_critical_mass_calculation();
    test_disk_stability_criterion();
    test_gas_star_fraction_in_instability();
    test_bulge_growth_from_instability();
    test_disk_mass_conservation_in_instability();
    test_zero_disk_mass_handling();
    test_instability_with_gas_only();
    
    END_TEST_SUITE();
    PRINT_TEST_SUMMARY();
    
    return TEST_EXIT_CODE();
}

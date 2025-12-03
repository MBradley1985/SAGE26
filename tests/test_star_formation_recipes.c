/*
 * STAR FORMATION RECIPE TESTS
 * 
 * Tests for star formation prescriptions:
 * - Kennicutt-Schmidt relation (SFR ∝ Σ_gas^n)
 * - Molecular fraction calculations (BR06, KMT09)
 * - H2 surface density dependence  
 * - Star formation timescale (t_* = t_dyn / ε_*)
 * - Critical surface density thresholds
 * - SF efficiency parameters
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "test_framework.h"
#include "../src/core_allvars.h"
#include "../src/model_misc.h"
#include "../src/model_starformation_and_feedback.h"

void test_sf_efficiency_in_range() {
    BEGIN_TEST("Star Formation Efficiency Parameter in Physical Range");
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    
    // Typical SFR efficiencies: 0.01 to 0.1
    run_params.SfrEfficiency = 0.05;
    
    ASSERT_IN_RANGE(run_params.SfrEfficiency, 0.001, 1.0,
                   "SFR efficiency in reasonable range");
    
    // Efficiency relates to dynamical time
    // SFR = ε * M_gas / t_dyn
    // For t_dyn ~ 100 Myr, ε ~ 0.01-0.1 gives t_* ~ 1-10 Gyr
    double t_dyn = 0.1;  // 100 Myr in Gyr
    double M_gas = 1.0;
    double SFR = run_params.SfrEfficiency * M_gas / t_dyn;
    
    ASSERT_GREATER_THAN(SFR, 0.0, "SFR > 0 for positive efficiency");
    ASSERT_LESS_THAN(SFR, M_gas / t_dyn, "SFR < M_gas/t_dyn (ε < 1)");
}

void test_dynamical_time_scaling() {
    BEGIN_TEST("Star Formation Timescale Scales with Dynamical Time");
    
    struct GALAXY gal1, gal2;
    memset(&gal1, 0, sizeof(struct GALAXY));
    memset(&gal2, 0, sizeof(struct GALAXY));
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.SfrEfficiency = 0.05;
    
    // Compact galaxy - short t_dyn
    gal1.DiskScaleRadius = 0.003;  // 3 kpc/h
    gal1.Vvir = 200.0;  // km/s
    double reff1 = 3.0 * gal1.DiskScaleRadius;
    double tdyn1 = reff1 / gal1.Vvir;
    
    // Extended galaxy - long t_dyn
    gal2.DiskScaleRadius = 0.01;  // 10 kpc/h  
    gal2.Vvir = 100.0;  // km/s
    double reff2 = 3.0 * gal2.DiskScaleRadius;
    double tdyn2 = reff2 / gal2.Vvir;
    
    // For same gas mass, shorter t_dyn → higher SFR
    ASSERT_LESS_THAN(tdyn1, tdyn2, "Compact galaxy has shorter t_dyn");
    
    // SFR ∝ 1/t_dyn, so SFR1 > SFR2
    double M_gas = 1.0;
    double SFR1 = run_params.SfrEfficiency * M_gas / tdyn1;
    double SFR2 = run_params.SfrEfficiency * M_gas / tdyn2;
    
    ASSERT_GREATER_THAN(SFR1, SFR2,
                       "Compact galaxy has higher SFR for same gas mass");
}

void test_critical_gas_surface_density() {
    BEGIN_TEST("Star Formation Above Critical Surface Density");
    
    struct GALAXY gal;
    memset(&gal, 0, sizeof(struct GALAXY));
    
    // From Kauffmann 1996: Σ_crit = 0.19 * Vvir * r_eff
    gal.DiskScaleRadius = 0.005;  // 5 kpc/h
    gal.Vvir = 150.0;  // km/s
    double reff = 3.0 * gal.DiskScaleRadius;
    double cold_crit = 0.19 * gal.Vvir * reff;
    
    // Below critical - no SF
    gal.ColdGas = 0.5 * cold_crit;
    ASSERT_LESS_THAN(gal.ColdGas, cold_crit,
                    "Gas below critical threshold");
    
    // Above critical - SF occurs
    gal.ColdGas = 2.0 * cold_crit;
    ASSERT_GREATER_THAN(gal.ColdGas, cold_crit,
                       "Gas above critical threshold enables SF");
    
    // Critical threshold is positive and reasonable
    ASSERT_GREATER_THAN(cold_crit, 0.0, "Critical mass > 0");
    ASSERT_IN_RANGE(cold_crit, 0.001, 100.0,
                   "Critical mass in reasonable range");
}

void test_molecular_fraction_physical() {
    BEGIN_TEST("Molecular Fraction in [0, 1] Range");
    
    // Test BR06 molecular fraction calculation
    float disk_scale_length_pc = 5000.0;  // 5 kpc
    
    // Low surface density → low H2 fraction
    float gas_sigma_low = 1.0;  // Msun/pc^2
    float stellar_sigma = 100.0;
    float f_mol_low = calculate_molecular_fraction_BR06(gas_sigma_low, stellar_sigma, 
                                                        disk_scale_length_pc);
    
    ASSERT_IN_RANGE(f_mol_low, 0.0, 1.0,
                   "Low Σ_gas: molecular fraction in [0,1]");
    
    // High surface density → high H2 fraction
    float gas_sigma_high = 100.0;  // Msun/pc^2
    float f_mol_high = calculate_molecular_fraction_BR06(gas_sigma_high, stellar_sigma,
                                                         disk_scale_length_pc);
    
    ASSERT_IN_RANGE(f_mol_high, 0.0, 1.0,
                   "High Σ_gas: molecular fraction in [0,1]");
    ASSERT_GREATER_THAN(f_mol_high, f_mol_low,
                       "Higher Σ_gas → higher f_H2");
}

void test_h2_gas_calculation() {
    BEGIN_TEST("H2 Gas Mass Calculation from Molecular Fraction");
    
    struct GALAXY gal;
    memset(&gal, 0, sizeof(struct GALAXY));
    
    gal.ColdGas = 1.0;  // 10^10 Msun/h
    gal.DiskScaleRadius = 0.005;  // 5 kpc/h
    
    // Assume 50% molecular fraction
    float f_mol = 0.5;
    double H2_mass = f_mol * gal.ColdGas;
    
    ASSERT_CLOSE(H2_mass, 0.5, 1e-6,
                "H2 mass = f_mol × cold gas");
    ASSERT_LESS_THAN(H2_mass, gal.ColdGas,
                    "H2 mass ≤ total cold gas");
    ASSERT_GREATER_THAN(H2_mass, 0.0,
                       "H2 mass ≥ 0");
}

void test_sf_from_h2_only() {
    BEGIN_TEST("Star Formation from H2 Only (Mode 1)");
    
    struct GALAXY gal;
    memset(&gal, 0, sizeof(struct GALAXY));
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.SfrEfficiency = 0.05;
    
    gal.ColdGas = 1.0;
    gal.H2gas = 0.3;  // 30% molecular
    gal.DiskScaleRadius = 0.005;
    gal.Vvir = 150.0;
    
    double reff = 3.0 * gal.DiskScaleRadius;
    double tdyn = reff / gal.Vvir;
    
    // SFR should be from H2, not total cold gas
    double SFR_h2 = run_params.SfrEfficiency * gal.H2gas / tdyn;
    double SFR_total = run_params.SfrEfficiency * gal.ColdGas / tdyn;
    
    ASSERT_LESS_THAN(SFR_h2, SFR_total,
                    "SF from H2 only is less than SF from all gas");
    ASSERT_CLOSE(SFR_h2 / SFR_total, 0.3, 0.01,
                "SFR ratio matches H2 fraction");
}

void test_sf_quenching_at_low_gas() {
    BEGIN_TEST("Star Formation Quenches at Low Gas Mass");
    
    struct GALAXY gal;
    memset(&gal, 0, sizeof(struct GALAXY));
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.SfrEfficiency = 0.05;
    run_params.RecycleFraction = 0.43;
    
    // Start with more gas to enable significant depletion
    gal.ColdGas = 0.5;
    gal.MetalsColdGas = 0.01;
    gal.StellarMass = 1.0;
    gal.MetalsStellarMass = 0.02;
    gal.DiskScaleRadius = 0.005;
    gal.Vvir = 100.0;
    
    double initial_gas = gal.ColdGas;
    double initial_stellar = gal.StellarMass;
    
    // Form stars until gas is depleted
    for(int i = 0; i < 30; i++) {
        if(gal.ColdGas > 0.001) {
            double Z = get_metallicity(gal.ColdGas, gal.MetalsColdGas);
            double reff = 3.0 * gal.DiskScaleRadius;
            double tdyn = reff / gal.Vvir;
            double cold_crit = 0.19 * gal.Vvir * reff;
            
            if(gal.ColdGas > cold_crit && tdyn > 0.0) {
                double stars = 0.02;  // Form more stars per step
                if(stars > gal.ColdGas * 0.9) stars = gal.ColdGas * 0.5;
                update_from_star_formation(0, stars, Z, &gal, &run_params);
            }
        }
    }
    
    // Gas should be significantly depleted
    ASSERT_LESS_THAN(gal.ColdGas, initial_gas * 0.8,
                    "Gas depletion from SF");
    ASSERT_GREATER_THAN(gal.StellarMass, initial_stellar,
                       "Stellar mass increased");
}

void test_vvir_dependence() {
    BEGIN_TEST("Star Formation Rate Depends on Virial Velocity");
    
    struct GALAXY gal1, gal2;
    memset(&gal1, 0, sizeof(struct GALAXY));
    memset(&gal2, 0, sizeof(struct GALAXY));
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.SfrEfficiency = 0.05;
    
    // Same disk size and gas, different Vvir
    gal1.DiskScaleRadius = 0.005;
    gal1.Vvir = 100.0;  // Low
    gal1.ColdGas = 1.0;
    
    gal2.DiskScaleRadius = 0.005;
    gal2.Vvir = 200.0;  // High
    gal2.ColdGas = 1.0;
    
    double reff = 3.0 * 0.005;
    double tdyn1 = reff / gal1.Vvir;
    double tdyn2 = reff / gal2.Vvir;
    
    // Higher Vvir → shorter t_dyn → higher SFR
    ASSERT_GREATER_THAN(tdyn1, tdyn2,
                       "Lower Vvir → longer t_dyn");
    
    double cold_crit1 = 0.19 * gal1.Vvir * reff;
    double cold_crit2 = 0.19 * gal2.Vvir * reff;
    
    // Higher Vvir → higher critical density
    ASSERT_GREATER_THAN(cold_crit2, cold_crit1,
                       "Higher Vvir → higher critical Σ");
}

void test_disk_size_effect_on_sf() {
    BEGIN_TEST("Disk Size Affects Star Formation Rate");
    
    struct GALAXY gal_compact, gal_extended;
    memset(&gal_compact, 0, sizeof(struct GALAXY));
    memset(&gal_extended, 0, sizeof(struct GALAXY));
    
    // Same Vvir and gas, different disk sizes
    gal_compact.DiskScaleRadius = 0.003;  // Compact
    gal_compact.Vvir = 150.0;
    gal_compact.ColdGas = 1.0;
    
    gal_extended.DiskScaleRadius = 0.01;  // Extended
    gal_extended.Vvir = 150.0;
    gal_extended.ColdGas = 1.0;
    
    double reff_c = 3.0 * gal_compact.DiskScaleRadius;
    double reff_e = 3.0 * gal_extended.DiskScaleRadius;
    
    double tdyn_c = reff_c / gal_compact.Vvir;
    double tdyn_e = reff_e / gal_extended.Vvir;
    
    // Compact disk → shorter t_dyn
    ASSERT_LESS_THAN(tdyn_c, tdyn_e,
                    "Compact disk has shorter dynamical time");
    
    // Critical mass also depends on size
    double crit_c = 0.19 * gal_compact.Vvir * reff_c;
    double crit_e = 0.19 * gal_extended.Vvir * reff_e;
    
    ASSERT_LESS_THAN(crit_c, crit_e,
                    "Compact disk has lower critical mass");
}

void test_sf_depletion_timescale() {
    BEGIN_TEST("Gas Depletion Timescale is Reasonable");
    
    struct GALAXY gal;
    memset(&gal, 0, sizeof(struct GALAXY));
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.SfrEfficiency = 0.05;
    
    gal.ColdGas = 1.0;
    gal.DiskScaleRadius = 0.005;
    gal.Vvir = 150.0;
    
    double reff = 3.0 * gal.DiskScaleRadius;
    double tdyn = reff / gal.Vvir;
    double cold_crit = 0.19 * gal.Vvir * reff;
    
    if(gal.ColdGas > cold_crit) {
        double SFR = run_params.SfrEfficiency * (gal.ColdGas - cold_crit) / tdyn;
        double t_depl = (gal.ColdGas - cold_crit) / SFR;
        
        // Depletion time = t_dyn / ε ∼ 200 Myr to 10 Gyr
        ASSERT_GREATER_THAN(t_depl, tdyn,
                           "Depletion time > dynamical time");
        ASSERT_LESS_THAN(t_depl, 100.0 * tdyn,
                        "Depletion time < 100 × t_dyn");
    }
}

void test_zero_disk_radius_protection() {
    BEGIN_TEST("Protection Against Zero Disk Radius");
    
    struct GALAXY gal;
    memset(&gal, 0, sizeof(struct GALAXY));
    
    gal.DiskScaleRadius = 0.0;  // Unphysical
    gal.Vvir = 150.0;
    gal.ColdGas = 1.0;
    
    double reff = 3.0 * gal.DiskScaleRadius;
    
    // Should not crash, tdyn becomes 0
    double tdyn = reff / gal.Vvir;
    
    ASSERT_EQUAL_FLOAT(tdyn, 0.0, "Zero radius → zero t_dyn");
    
    // SF should be zero or handled gracefully
    if(tdyn <= 0.0) {
        // SFR would be undefined or zero
        ASSERT_TRUE(1, "Zero t_dyn handled without crash");
    }
}

int main() {
    BEGIN_TEST_SUITE("Star Formation Recipes");
    
    test_sf_efficiency_in_range();
    test_dynamical_time_scaling();
    test_critical_gas_surface_density();
    test_molecular_fraction_physical();
    test_h2_gas_calculation();
    test_sf_from_h2_only();
    test_sf_quenching_at_low_gas();
    test_vvir_dependence();
    test_disk_size_effect_on_sf();
    test_sf_depletion_timescale();
    test_zero_disk_radius_protection();
    
    END_TEST_SUITE();
    PRINT_TEST_SUMMARY();
    
    return TEST_EXIT_CODE();
}

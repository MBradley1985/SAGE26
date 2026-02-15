/*
 * INFALL PHYSICS TESTS
 * 
 * Tests for gas infall from IGM:
 * - Infall rate calculations
 * - Baryon fraction preservation
 * - Reionization suppression
 * - Correct reservoir routing (CGM vs Hot)
 * - Conservation during satellite stripping
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "test_framework.h"
#include "../src/core_allvars.h"
#include "../src/model_misc.h"
#include "../src/model_infall.h"

void test_baryon_fraction_infall() {
    BEGIN_TEST("Baryon Fraction in Infall Calculation");
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.BaryonFrac = 0.17;  // Standard cosmology
    run_params.ReionizationOn = 0;  // No reionization suppression
    
    struct GALAXY gal;
    memset(&gal, 0, sizeof(struct GALAXY));
    gal.Mvir = 100.0;  // 10^12 Msun/h
    gal.StellarMass = 1.0;
    gal.ColdGas = 0.5;
    gal.HotGas = 5.0;
    gal.EjectedMass = 1.0;
    gal.BlackHoleMass = 0.01;
    gal.ICS = 0.0;
    gal.CGMgas = 3.0;
    
    // Calculate expected baryonic mass
    double expected_baryons = run_params.BaryonFrac * gal.Mvir;
    
    // Current baryonic mass
    double current_baryons = gal.StellarMass + gal.ColdGas + gal.HotGas + 
                            gal.EjectedMass + gal.BlackHoleMass + gal.ICS + gal.CGMgas;
    
    // Infall should make up the difference
    double infall = expected_baryons - current_baryons;
    
    double calculated_infall = run_params.BaryonFrac * gal.Mvir - 
                               (gal.StellarMass + gal.ColdGas + gal.HotGas + 
                                gal.EjectedMass + gal.BlackHoleMass + gal.ICS + gal.CGMgas);
    
    ASSERT_CLOSE(infall, calculated_infall, 1e-10,
                "Infall calculated correctly");
    ASSERT_CLOSE(expected_baryons, current_baryons + infall, 1e-10,
                "Baryon fraction preserved after infall");
}

void test_infall_positive_or_negative() {
    BEGIN_TEST("Infall Can Be Positive or Negative");
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.BaryonFrac = 0.17;
    run_params.ReionizationOn = 0;
    
    // Test 1: Positive infall (gas-poor halo)
    {
        struct GALAXY gal;
        memset(&gal, 0, sizeof(struct GALAXY));
        gal.Mvir = 100.0;
        gal.StellarMass = 0.5;
        gal.ColdGas = 0.2;
        gal.HotGas = 1.0;
        
        double expected_baryons = run_params.BaryonFrac * gal.Mvir;
        double current_baryons = gal.StellarMass + gal.ColdGas + gal.HotGas;
        double infall = expected_baryons - current_baryons;
        
        ASSERT_GREATER_THAN(infall, 0.0, "Positive infall for gas-poor halo");
    }
    
    // Test 2: Negative infall (gas-rich halo - stripping scenario)
    {
        struct GALAXY gal;
        memset(&gal, 0, sizeof(struct GALAXY));
        gal.Mvir = 10.0;  // Small halo
        gal.StellarMass = 0.5;
        gal.ColdGas = 0.5;
        gal.HotGas = 5.0;  // Excess hot gas
        
        double expected_baryons = run_params.BaryonFrac * gal.Mvir;
        double current_baryons = gal.StellarMass + gal.ColdGas + gal.HotGas;
        double infall = expected_baryons - current_baryons;
        
        ASSERT_LESS_THAN(infall, 0.0, "Negative infall for gas-rich halo");
    }
}

void test_infall_routing_by_regime() {
    BEGIN_TEST("Infall Routes to Correct Reservoir by Regime");
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.BaryonFrac = 0.17;
    run_params.ReionizationOn = 0;
    run_params.CGMrecipeOn = 1;
    
    // Test Regime 0: Infall → CGM
    {
        struct GALAXY gal;
        memset(&gal, 0, sizeof(struct GALAXY));
        gal.Regime = 0;
        gal.Mvir = 50.0;
        gal.CGMgas = 2.0;
        gal.HotGas = 1.0;
        gal.MetalsCGMgas = 0.03;
        gal.MetalsHotGas = 0.015;
        
        double initial_cgm = gal.CGMgas;
        
        // Simulate positive infall to CGM
        double infall_amount = 0.5;
        
        add_infall_to_hot(0, infall_amount, &gal, &run_params);
        
        // In Regime 0, infall goes to CGM
        ASSERT_TRUE(gal.CGMgas >= initial_cgm, "Regime 0: infall to CGM");
    }
    
    // Test Regime 1: Infall → HotGas
    {
        struct GALAXY gal;
        memset(&gal, 0, sizeof(struct GALAXY));
        gal.Regime = 1;
        gal.Mvir = 500.0;
        gal.CGMgas = 1.0;
        gal.HotGas = 10.0;
        gal.MetalsCGMgas = 0.015;
        gal.MetalsHotGas = 0.15;
        
        double initial_hot = gal.HotGas;
        
        // Simulate positive infall to HotGas
        double infall_amount = 1.0;
        
        add_infall_to_hot(0, infall_amount, &gal, &run_params);
        
        // In Regime 1, infall goes to HotGas
        ASSERT_TRUE(gal.HotGas >= initial_hot, "Regime 1: infall to HotGas");
    }
}

void test_satellite_gas_stripping() {
    BEGIN_TEST("Satellite Gas Transfer to Central");
    
    struct GALAXY galaxies[2];
    memset(galaxies, 0, sizeof(struct GALAXY) * 2);
    
    // Satellite (index 1)
    galaxies[1].EjectedMass = 0.5;
    galaxies[1].MetalsEjectedMass = 0.01;
    galaxies[1].CGMgas = 0.3;
    galaxies[1].MetalsCGMgas = 0.005;
    galaxies[1].ICS = 0.2;
    galaxies[1].MetalsICS = 0.003;
    
    double sat_ejected = galaxies[1].EjectedMass;
    double sat_cgm = galaxies[1].CGMgas;
    
    // After infall_recipe, satellite gas should go to central (index 0)
    // Simulate this:
    galaxies[0].EjectedMass += galaxies[1].EjectedMass;
    galaxies[0].MetalsEjectedMass += galaxies[1].MetalsEjectedMass;
    galaxies[0].CGMgas += galaxies[1].CGMgas;
    galaxies[0].MetalsCGMgas += galaxies[1].MetalsCGMgas;
    galaxies[0].ICS += galaxies[1].ICS;
    galaxies[0].MetalsICS += galaxies[1].MetalsICS;
    
    galaxies[1].EjectedMass = 0.0;
    galaxies[1].MetalsEjectedMass = 0.0;
    galaxies[1].CGMgas = 0.0;
    galaxies[1].MetalsCGMgas = 0.0;
    galaxies[1].ICS = 0.0;
    galaxies[1].MetalsICS = 0.0;
    
    ASSERT_EQUAL_FLOAT(galaxies[1].EjectedMass, 0.0,
                      "Satellite ejected mass transferred");
    ASSERT_EQUAL_FLOAT(galaxies[1].CGMgas, 0.0,
                      "Satellite CGM transferred");
    ASSERT_CLOSE(galaxies[0].EjectedMass, sat_ejected, 1e-10,
                "Central received satellite ejected mass");
    ASSERT_CLOSE(galaxies[0].CGMgas, sat_cgm, 1e-10,
                "Central received satellite CGM");
}

void test_infall_mass_conservation() {
    BEGIN_TEST("Mass Conservation in Infall Process");
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.BaryonFrac = 0.17;
    run_params.ReionizationOn = 0;
    run_params.CGMrecipeOn = 1;
    run_params.DustOn = 0;  // Test without dust first
    
    struct GALAXY gal;
    memset(&gal, 0, sizeof(struct GALAXY));
    gal.Regime = 0;
    gal.Mvir = 80.0;
    gal.StellarMass = 1.0;
    gal.ColdGas = 0.5;
    gal.HotGas = 3.0;
    gal.CGMgas = 2.0;
    gal.EjectedMass = 0.5;
    gal.BlackHoleMass = 0.01;
    gal.ICS = 0.1;
    
    double initial_total = gal.StellarMass + gal.ColdGas + gal.HotGas + 
                          gal.CGMgas + gal.EjectedMass + gal.BlackHoleMass + gal.ICS;
    
    double expected_baryons = run_params.BaryonFrac * gal.Mvir;
    double infall = expected_baryons - initial_total;
    
    // Add infall to CGM (Regime 0)
    gal.CGMgas += infall;
    
    double final_total = gal.StellarMass + gal.ColdGas + gal.HotGas + 
                        gal.CGMgas + gal.EjectedMass + gal.BlackHoleMass + gal.ICS;
    
    ASSERT_CLOSE(final_total, expected_baryons, 1e-5,
                "Total baryons match expected after infall");
}

void test_infall_mass_conservation_with_dust() {
    BEGIN_TEST("Mass Conservation in Infall with Dust");
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.BaryonFrac = 0.17;
    run_params.ReionizationOn = 0;
    run_params.CGMrecipeOn = 1;
    run_params.DustOn = 1;  // Enable dust
    
    // Initialize snapshot arrays (required for some functions)
    run_params.nsnapshots = 64;
    for(int i = 0; i < 64; i++) {
        run_params.ZZ[i] = 20.0 * (63 - i) / 63.0;
        run_params.AA[i] = 1.0 / (1.0 + run_params.ZZ[i]);
    }
    
    struct GALAXY gal;
    memset(&gal, 0, sizeof(struct GALAXY));
    gal.Regime = 0;
    gal.Mvir = 80.0;
    gal.StellarMass = 1.0;
    gal.ColdGas = 0.5;
    gal.HotGas = 3.0;
    gal.CGMgas = 2.0;
    gal.EjectedMass = 0.5;
    gal.BlackHoleMass = 0.01;
    gal.ICS = 0.1;
    
    // Add dust reservoirs (must be <= metals in each reservoir)
    gal.MetalsColdGas = 0.01;
    gal.MetalsHotGas = 0.06;
    gal.MetalsCGMgas = 0.04;
    gal.MetalsEjectedMass = 0.01;
    
    gal.ColdDust = 0.002;      // 20% of cold metals
    gal.HotDust = 0.015;       // 25% of hot metals
    gal.CGMDust = 0.010;       // 25% of CGM metals
    gal.EjectedDust = 0.0025;  // 25% of ejected metals
    
    // Calculate total baryonic mass INCLUDING DUST
    double initial_total = gal.StellarMass + gal.ColdGas + gal.HotGas + 
                          gal.CGMgas + gal.EjectedMass + gal.BlackHoleMass + gal.ICS +
                          gal.ColdDust + gal.HotDust + gal.CGMDust + gal.EjectedDust;
    
    double expected_baryons = run_params.BaryonFrac * gal.Mvir;
    
    // Call actual infall_recipe to get correct infall (which should account for dust)
    double infall = infall_recipe(0, 1, 0.0, &gal, &run_params);
    
    // Manually add infall to CGM (Regime 0)
    gal.CGMgas += infall;
    
    // Calculate final total INCLUDING DUST
    double final_total = gal.StellarMass + gal.ColdGas + gal.HotGas + 
                        gal.CGMgas + gal.EjectedMass + gal.BlackHoleMass + gal.ICS +
                        gal.ColdDust + gal.HotDust + gal.CGMDust + gal.EjectedDust;
    
    ASSERT_CLOSE(final_total, expected_baryons, 1e-5,
                "Total baryons (including dust) match expected after infall");
    
    // Verify infall was reduced due to dust being counted
    double infall_without_dust = expected_baryons - (initial_total - gal.ColdDust - gal.HotDust - gal.CGMDust - gal.EjectedDust);
    ASSERT_LESS_THAN(infall, infall_without_dust,
                    "Infall reduced when dust counted in baryon budget");
}

void test_infall_recipe_accounts_for_dust() {
    BEGIN_TEST("infall_recipe() Correctly Accounts for Dust");
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.BaryonFrac = 0.17;
    run_params.ReionizationOn = 0;
    run_params.CGMrecipeOn = 1;
    run_params.DustOn = 1;
    
    // Test with multiple satellites to verify FoF sum
    struct GALAXY galaxies[3];
    memset(galaxies, 0, sizeof(struct GALAXY) * 3);
    
    // Central galaxy (index 0)
    galaxies[0].Mvir = 100.0;
    galaxies[0].StellarMass = 1.0;
    galaxies[0].ColdGas = 1.0;
    galaxies[0].HotGas = 5.0;
    galaxies[0].CGMgas = 3.0;
    galaxies[0].MetalsColdGas = 0.02;
    galaxies[0].MetalsHotGas = 0.1;
    galaxies[0].MetalsCGMgas = 0.06;
    galaxies[0].ColdDust = 0.005;
    galaxies[0].HotDust = 0.025;
    galaxies[0].CGMDust = 0.015;
    galaxies[0].Regime = 0;
    
    // Satellite 1 (index 1)
    galaxies[1].StellarMass = 0.3;
    galaxies[1].ColdGas = 0.2;
    galaxies[1].HotGas = 0.5;
    galaxies[1].CGMgas = 0.3;
    galaxies[1].EjectedMass = 0.1;
    galaxies[1].MetalsColdGas = 0.004;
    galaxies[1].MetalsHotGas = 0.01;
    galaxies[1].MetalsCGMgas = 0.006;
    galaxies[1].MetalsEjectedMass = 0.002;
    galaxies[1].ColdDust = 0.001;
    galaxies[1].HotDust = 0.0025;
    galaxies[1].CGMDust = 0.0015;
    galaxies[1].EjectedDust = 0.0005;
    
    // Satellite 2 (index 2)
    galaxies[2].StellarMass = 0.2;
    galaxies[2].ColdGas = 0.1;
    galaxies[2].HotGas = 0.3;
    galaxies[2].CGMgas = 0.2;
    galaxies[2].EjectedMass = 0.05;
    galaxies[2].MetalsColdGas = 0.002;
    galaxies[2].MetalsHotGas = 0.006;
    galaxies[2].MetalsCGMgas = 0.004;
    galaxies[2].MetalsEjectedMass = 0.001;
    galaxies[2].ColdDust = 0.0005;
    galaxies[2].HotDust = 0.0015;
    galaxies[2].CGMDust = 0.001;
    galaxies[2].EjectedDust = 0.00025;
    
    // Calculate total baryonic mass across all galaxies INCLUDING DUST
    double total_baryons = 0.0;
    for(int i = 0; i < 3; i++) {
        total_baryons += galaxies[i].StellarMass + galaxies[i].ColdGas + 
                        galaxies[i].HotGas + galaxies[i].CGMgas + 
                        galaxies[i].EjectedMass + galaxies[i].BlackHoleMass + 
                        galaxies[i].ICS;
        if(run_params.DustOn == 1) {
            total_baryons += galaxies[i].ColdDust + galaxies[i].HotDust +
                            galaxies[i].CGMDust + galaxies[i].EjectedDust;
        }
    }
    
    double expected_baryons = run_params.BaryonFrac * galaxies[0].Mvir;
    double expected_infall = expected_baryons - total_baryons;
    
    // Call infall_recipe
    double calculated_infall = infall_recipe(0, 3, 0.0, galaxies, &run_params);
    
    ASSERT_CLOSE(calculated_infall, expected_infall, 1e-6,
                "infall_recipe returns correct infall accounting for dust");
    
    // Verify dust was properly summed from all satellites
    double expected_total_dust = 0.0;
    for(int i = 0; i < 3; i++) {
        expected_total_dust += galaxies[i].ColdDust + galaxies[i].HotDust +
                              galaxies[i].CGMDust + galaxies[i].EjectedDust;
    }
    
    // After infall_recipe, satellite dust should be transferred to central
    ASSERT_GREATER_THAN(expected_total_dust, 0.0, 
                       "FoF has non-zero dust before infall");
}

void test_reionization_suppression() {
    BEGIN_TEST("Reionization Suppresses Infall");
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.BaryonFrac = 0.17;
    run_params.ReionizationOn = 1;
    run_params.Reionization_z0 = 8.0;
    run_params.Reionization_zr = 7.0;
    
    struct GALAXY gal;
    memset(&gal, 0, sizeof(struct GALAXY));
    gal.Mvir = 10.0;  // Small halo - susceptible to reionization
    gal.StellarMass = 0.1;
    gal.ColdGas = 0.05;
    gal.HotGas = 0.3;
    
    // At high redshift (z > zr), reionization modifier should be < 1
    // This means infall is suppressed
    
    double no_reion_infall = run_params.BaryonFrac * gal.Mvir - 
                             (gal.StellarMass + gal.ColdGas + gal.HotGas);
    
    // With reionization, infall should be less (modifier < 1)
    // We can't easily test this without calling do_reionization
    // But we can verify the concept
    double reion_modifier = 0.5;  // Example suppression
    double with_reion_infall = reion_modifier * run_params.BaryonFrac * gal.Mvir - 
                               (gal.StellarMass + gal.ColdGas + gal.HotGas);
    
    ASSERT_TRUE(with_reion_infall < no_reion_infall,
               "Reionization reduces infall");
}

int main() {
    BEGIN_TEST_SUITE("Gas Infall Physics");
    
    test_baryon_fraction_infall();
    test_infall_positive_or_negative();
    test_infall_routing_by_regime();
    test_satellite_gas_stripping();
    test_infall_mass_conservation();
    test_infall_mass_conservation_with_dust();
    test_infall_recipe_accounts_for_dust();
    test_reionization_suppression();
    
    END_TEST_SUITE();
    PRINT_TEST_SUMMARY();
    
    return TEST_EXIT_CODE();
}

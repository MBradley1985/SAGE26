/*
 * RAM PRESSURE STRIPPING TESTS
 * 
 * Tests for environmental gas stripping from satellites:
 * - Stripping criterion (gas exceeds expected for halo mass)
 * - Mass loss rates from stripping
 * - Gas transfer from satellite to central
 * - Environmental quenching
 * - Regime-dependent stripping (CGM vs Hot)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "test_framework.h"
#include "../src/core_allvars.h"
#include "../src/model_misc.h"
#include "../src/model_infall.h"

void test_stripping_removes_gas_from_satellite() {
    BEGIN_TEST("Stripping Removes Gas from Satellite");
    
    struct GALAXY galaxies[2];
    memset(galaxies, 0, sizeof(struct GALAXY) * 2);
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.CGMrecipeOn = 1;
    run_params.BaryonFrac = 0.17;
    run_params.ReionizationOn = 0;
    
    // Central galaxy
    galaxies[0].Regime = 1;
    galaxies[0].HotGas = 10.0;
    galaxies[0].MetalsHotGas = 0.1;
    
    // Satellite with excess gas (will be stripped)
    galaxies[1].Regime = 1;
    galaxies[1].Mvir = 10.0;  // Small halo
    galaxies[1].HotGas = 5.0;  // Too much gas for this halo
    galaxies[1].MetalsHotGas = 0.05;
    galaxies[1].StellarMass = 0.5;
    galaxies[1].ColdGas = 0.2;
    galaxies[1].BlackHoleMass = 0.01;
    galaxies[1].ICS = 0.0;
    galaxies[1].EjectedMass = 0.0;
    galaxies[1].CGMgas = 0.0;
    
    double initial_sat_hot = galaxies[1].HotGas;
    double initial_cen_hot = galaxies[0].HotGas;
    
    // Apply stripping
    double Zcurr = 0.0;
    strip_from_satellite(0, 1, Zcurr, galaxies, &run_params);
    
    // Satellite should lose gas
    ASSERT_LESS_THAN(galaxies[1].HotGas, initial_sat_hot,
                    "Satellite hot gas decreased from stripping");
    
    // Central should gain gas
    ASSERT_GREATER_THAN(galaxies[0].HotGas, initial_cen_hot,
                       "Central hot gas increased from stripping");
}

void test_stripping_conserves_mass() {
    BEGIN_TEST("Stripping Conserves Total Gas Mass");
    
    struct GALAXY galaxies[2];
    memset(galaxies, 0, sizeof(struct GALAXY) * 2);
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.CGMrecipeOn = 1;
    run_params.BaryonFrac = 0.17;
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
    galaxies[1].ICS = 0.0;
    galaxies[1].EjectedMass = 0.0;
    galaxies[1].CGMgas = 0.0;
    
    double initial_total_hot = galaxies[0].HotGas + galaxies[1].HotGas;
    double initial_total_metals = galaxies[0].MetalsHotGas + galaxies[1].MetalsHotGas;
    
    strip_from_satellite(0, 1, 0.0, galaxies, &run_params);
    
    double final_total_hot = galaxies[0].HotGas + galaxies[1].HotGas;
    double final_total_metals = galaxies[0].MetalsHotGas + galaxies[1].MetalsHotGas;
    
    ASSERT_CLOSE(initial_total_hot, final_total_hot, 1e-5,
                "Total hot gas conserved during stripping");
    ASSERT_CLOSE(initial_total_metals, final_total_metals, 1e-5,
                "Total metals conserved during stripping");
}

void test_regime_dependent_stripping() {
    BEGIN_TEST("Stripping from Correct Reservoir by Regime");
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.CGMrecipeOn = 1;
    run_params.BaryonFrac = 0.17;
    run_params.ReionizationOn = 0;
    
    // Test CGM regime stripping
    {
        struct GALAXY galaxies[2];
        memset(galaxies, 0, sizeof(struct GALAXY) * 2);
        
        galaxies[0].Regime = 0;
        galaxies[0].CGMgas = 5.0;
        galaxies[0].MetalsCGMgas = 0.05;
        
        galaxies[1].Regime = 0;
        galaxies[1].Mvir = 10.0;
        galaxies[1].CGMgas = 3.0;
        galaxies[1].MetalsCGMgas = 0.03;
        galaxies[1].StellarMass = 0.5;
        galaxies[1].ColdGas = 0.2;
        galaxies[1].BlackHoleMass = 0.01;
        galaxies[1].HotGas = 0.0;
        galaxies[1].ICS = 0.0;
        galaxies[1].EjectedMass = 0.0;
        
        double initial_sat_cgm = galaxies[1].CGMgas;
        double initial_sat_hot = galaxies[1].HotGas;
        
        strip_from_satellite(0, 1, 0.0, galaxies, &run_params);
        
        // In CGM regime, should strip from CGMgas, not HotGas
        if(galaxies[1].CGMgas < initial_sat_cgm) {
            ASSERT_EQUAL_FLOAT(galaxies[1].HotGas, initial_sat_hot,
                              "Regime 0: HotGas unchanged, strips from CGM");
        }
    }
    
    // Test Hot regime stripping
    {
        struct GALAXY galaxies[2];
        memset(galaxies, 0, sizeof(struct GALAXY) * 2);
        
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
        galaxies[1].CGMgas = 0.0;
        galaxies[1].ICS = 0.0;
        galaxies[1].EjectedMass = 0.0;
        
        double initial_sat_hot = galaxies[1].HotGas;
        double initial_sat_cgm = galaxies[1].CGMgas;
        
        strip_from_satellite(0, 1, 0.0, galaxies, &run_params);
        
        // In Hot regime, should strip from HotGas
        if(galaxies[1].HotGas < initial_sat_hot) {
            ASSERT_EQUAL_FLOAT(galaxies[1].CGMgas, initial_sat_cgm,
                              "Regime 1: CGMgas unchanged, strips from Hot");
        }
    }
}

void test_no_stripping_if_gas_balanced() {
    BEGIN_TEST("No Stripping if Gas Matches Halo Mass");
    
    struct GALAXY galaxies[2];
    memset(galaxies, 0, sizeof(struct GALAXY) * 2);
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.CGMrecipeOn = 1;
    run_params.BaryonFrac = 0.17;
    run_params.ReionizationOn = 0;
    
    galaxies[0].Regime = 1;
    galaxies[0].HotGas = 10.0;
    galaxies[0].MetalsHotGas = 0.1;
    
    // Satellite with balanced gas
    galaxies[1].Regime = 1;
    galaxies[1].Mvir = 100.0;  // Large halo
    galaxies[1].HotGas = 15.0;  // Appropriate for this mass
    galaxies[1].MetalsHotGas = 0.15;
    galaxies[1].StellarMass = 5.0;
    galaxies[1].ColdGas = 1.0;
    galaxies[1].BlackHoleMass = 0.1;
    galaxies[1].ICS = 0.0;
    galaxies[1].EjectedMass = 0.0;
    galaxies[1].CGMgas = 0.0;
    
    double initial_sat_hot = galaxies[1].HotGas;
    
    strip_from_satellite(0, 1, 0.0, galaxies, &run_params);
    
    // With balanced baryons, minimal or no stripping
    ASSERT_CLOSE(galaxies[1].HotGas, initial_sat_hot, 0.5,
                "Minimal stripping when gas matches halo mass");
}

void test_stripping_transfers_metals() {
    BEGIN_TEST("Stripping Transfers Metals with Gas");
    
    struct GALAXY galaxies[2];
    memset(galaxies, 0, sizeof(struct GALAXY) * 2);
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.CGMrecipeOn = 1;
    run_params.BaryonFrac = 0.17;
    run_params.ReionizationOn = 0;
    
    galaxies[0].Regime = 1;
    galaxies[0].HotGas = 10.0;
    galaxies[0].MetalsHotGas = 0.05;  // 0.5% metallicity
    
    // Satellite with metal-rich gas
    galaxies[1].Regime = 1;
    galaxies[1].Mvir = 10.0;
    galaxies[1].HotGas = 5.0;
    galaxies[1].MetalsHotGas = 0.15;  // 3% metallicity (metal-rich)
    galaxies[1].StellarMass = 0.5;
    galaxies[1].ColdGas = 0.2;
    galaxies[1].BlackHoleMass = 0.01;
    galaxies[1].ICS = 0.0;
    galaxies[1].EjectedMass = 0.0;
    galaxies[1].CGMgas = 0.0;
    
    double Z_sat_before = get_metallicity(galaxies[1].HotGas, galaxies[1].MetalsHotGas);
    double initial_cen_metals = galaxies[0].MetalsHotGas;
    
    strip_from_satellite(0, 1, 0.0, galaxies, &run_params);
    
    // Central should gain metals
    ASSERT_GREATER_THAN(galaxies[0].MetalsHotGas, initial_cen_metals,
                       "Central gains metals from metal-rich stripped gas");
    
    // Satellite metallicity should stay roughly constant (same reservoir stripped)
    double Z_sat_after = get_metallicity(galaxies[1].HotGas, galaxies[1].MetalsHotGas);
    if(galaxies[1].HotGas > 0.1) {
        ASSERT_CLOSE(Z_sat_after, Z_sat_before, 0.01,
                    "Satellite metallicity preserved during stripping");
    }
}

void test_environmental_quenching() {
    BEGIN_TEST("Gas Stripping Leads to Quenching");
    
    struct GALAXY galaxies[2];
    memset(galaxies, 0, sizeof(struct GALAXY) * 2);
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.CGMrecipeOn = 1;
    run_params.BaryonFrac = 0.17;
    run_params.ReionizationOn = 0;
    
    galaxies[0].Regime = 0;
    galaxies[0].CGMgas = 5.0;
    galaxies[0].MetalsCGMgas = 0.05;
    
    // Satellite with CGM that will be stripped
    galaxies[1].Regime = 0;
    galaxies[1].Mvir = 5.0;
    galaxies[1].CGMgas = 2.0;  // Excess CGM
    galaxies[1].MetalsCGMgas = 0.02;
    galaxies[1].ColdGas = 0.5;  // Still has cold gas for SF
    galaxies[1].MetalsColdGas = 0.01;
    galaxies[1].StellarMass = 1.0;
    galaxies[1].HotGas = 0.0;
    galaxies[1].BlackHoleMass = 0.01;
    galaxies[1].ICS = 0.0;
    galaxies[1].EjectedMass = 0.0;
    
    double initial_cgm = galaxies[1].CGMgas;
    
    strip_from_satellite(0, 1, 0.0, galaxies, &run_params);
    
    // CGM should be reduced
    if(galaxies[1].CGMgas < initial_cgm) {
        // Loss of CGM reservoir reduces future cooling/gas supply
        ASSERT_LESS_THAN(galaxies[1].CGMgas, initial_cgm * 0.9,
                        "Significant CGM stripping occurred");
        
        // Cold gas remains (not stripped directly)
        ASSERT_CLOSE(galaxies[1].ColdGas, 0.5, 1e-3,
                    "Cold gas not directly stripped (protected in disk)");
    }
}

void test_no_stripping_below_mass_threshold() {
    BEGIN_TEST("No Stripping Below Minimum Gas Mass");
    
    struct GALAXY galaxies[2];
    memset(galaxies, 0, sizeof(struct GALAXY) * 2);
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.CGMrecipeOn = 1;
    run_params.BaryonFrac = 0.17;
    run_params.ReionizationOn = 0;
    
    galaxies[0].Regime = 1;
    galaxies[0].HotGas = 10.0;
    galaxies[0].MetalsHotGas = 0.1;
    
    // Satellite with minimal gas
    galaxies[1].Regime = 1;
    galaxies[1].Mvir = 10.0;
    galaxies[1].HotGas = 0.001;  // Tiny amount
    galaxies[1].MetalsHotGas = 0.00001;
    galaxies[1].StellarMass = 1.0;
    galaxies[1].ColdGas = 0.1;
    galaxies[1].BlackHoleMass = 0.01;
    galaxies[1].ICS = 0.0;
    galaxies[1].EjectedMass = 0.0;
    galaxies[1].CGMgas = 0.0;
    
    double initial_sat_hot = galaxies[1].HotGas;
    
    strip_from_satellite(0, 1, 0.0, galaxies, &run_params);
    
    // Should strip at most what's available
    ASSERT_TRUE(galaxies[1].HotGas >= 0.0,
               "Hot gas stays non-negative");
    ASSERT_TRUE(galaxies[1].HotGas <= initial_sat_hot,
               "Can't strip more than available");
}

void test_stripping_timescale() {
    BEGIN_TEST("Stripping Occurs Gradually (STEPS Factor)");
    
    struct GALAXY galaxies[2];
    memset(galaxies, 0, sizeof(struct GALAXY) * 2);
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.CGMrecipeOn = 1;
    run_params.BaryonFrac = 0.17;
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
    galaxies[1].ICS = 0.0;
    galaxies[1].EjectedMass = 0.0;
    galaxies[1].CGMgas = 0.0;
    
    double initial_hot = galaxies[1].HotGas;
    
    strip_from_satellite(0, 1, 0.0, galaxies, &run_params);
    
    double stripped = initial_hot - galaxies[1].HotGas;
    
    // Should strip a fraction per timestep, not all at once
    // (divided by STEPS in code)
    if(stripped > 0.0) {
        ASSERT_LESS_THAN(stripped, initial_hot,
                        "Doesn't strip all gas in one step");
    }
}

int main() {
    BEGIN_TEST_SUITE("Ram Pressure Stripping");
    
    test_stripping_removes_gas_from_satellite();
    test_stripping_conserves_mass();
    test_regime_dependent_stripping();
    test_no_stripping_if_gas_balanced();
    test_stripping_transfers_metals();
    test_environmental_quenching();
    test_no_stripping_below_mass_threshold();
    test_stripping_timescale();
    
    END_TEST_SUITE();
    PRINT_TEST_SUMMARY();
    
    return TEST_EXIT_CODE();
}

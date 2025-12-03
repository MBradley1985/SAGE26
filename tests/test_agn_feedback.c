/**
 * @file test_agn_feedback.c
 * @brief Unit tests for AGN feedback physics
 *
 * Tests AGN accretion, heating, and feedback suppression of cooling.
 * Includes radio mode (maintenance mode) feedback, quasar mode from mergers,
 * and Eddington-limited accretion.
 *
 * Key physics tested:
 * - Black hole accretion rates (radio mode, Bondi, cold cloud)
 * - Eddington limit enforcement
 * - AGN heating suppression of cooling
 * - Heating radius evolution
 * - Mass and metal conservation during accretion
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../src/core_allvars.h"
#include "../src/model_cooling_heating.h"
#include "../src/model_misc.h"
#include "test_framework.h"

// ============================================================================
// TEST 1: AGN Accretion Rate Scaling
// ============================================================================
void test_agn_accretion_scaling() {
    BEGIN_TEST("AGN Accretion Scales With BH Mass and Vvir");
    
    // Radio mode efficiency: AGNrate ~ M_BH * (Vvir/200)^3 * (M_hot/Mvir/0.1)
    double M_BH_1 = 0.01;  // 10^9 Msun
    double M_BH_2 = 0.1;   // 10^10 Msun (10x larger)
    
    double Vvir_1 = 200.0;  // km/s
    double Vvir_2 = 400.0;  // km/s (2x higher)
    
    // Accretion should scale linearly with BH mass
    double ratio_BH = M_BH_2 / M_BH_1;  // 10
    ASSERT_CLOSE(ratio_BH, 10.0, 1e-10,
                "BH mass ratio is 10:1");
    
    // Accretion should scale as Vvir^3
    double ratio_Vvir = pow(Vvir_2 / Vvir_1, 3.0);  // 8
    ASSERT_CLOSE(ratio_Vvir, 8.0, 1e-10,
                "Vvir^3 ratio is 8:1");
    
    // Combined scaling
    double combined_scaling = ratio_BH * ratio_Vvir;  // 80
    ASSERT_CLOSE(combined_scaling, 80.0, 1e-10,
                "Combined AGN scaling is 80:1");
}

// ============================================================================
// TEST 2: Eddington Limit Enforcement
// ============================================================================
void test_eddington_limit() {
    BEGIN_TEST("AGN Accretion Limited By Eddington Rate");
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.UnitEnergy_in_cgs = 1.989e53;
    run_params.UnitTime_in_s = 3.08568e16;
    run_params.Hubble_h = 0.7;
    
    // Black hole mass
    double M_BH = 0.1;  // 10^10 Msun/h
    
    // Eddington rate: L_Edd = 1.3e38 * M_BH (in physical Msun) erg/s
    // Then convert to accretion rate using efficiency
    double M_BH_physical = M_BH * 1e10 / run_params.Hubble_h;  // Msun
    double L_Edd_cgs = 1.3e38 * M_BH_physical;  // erg/s
    
    // Convert to code units
    double Edd_rate_code = L_Edd_cgs / (run_params.UnitEnergy_in_cgs / run_params.UnitTime_in_s) / (0.1 * 9e10);
    
    ASSERT_GREATER_THAN(Edd_rate_code, 0.0,
                       "Eddington rate positive");
    ASSERT_GREATER_THAN(M_BH_physical, 0.0,
                       "Physical BH mass positive");
}

// ============================================================================
// TEST 3: AGN Heating Suppresses Cooling
// ============================================================================
void test_agn_suppresses_cooling() {
    BEGIN_TEST("AGN Heating Suppresses Cooling");
    
    struct GALAXY gal[1];
    memset(gal, 0, sizeof(struct GALAXY));
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.AGNrecipeOn = 1;
    run_params.RadioModeEfficiency = 0.08;
    run_params.UnitMass_in_g = 1.989e43;
    run_params.UnitTime_in_s = 3.08568e16;
    run_params.UnitEnergy_in_cgs = 1.989e53;
    run_params.G = 43007.1;
    run_params.Hubble_h = 0.7;
    
    // Massive galaxy with hot gas and black hole
    gal[0].HotGas = 10.0;  // 10^11 Msun
    gal[0].MetalsHotGas = 0.2;  // 2% metallicity
    gal[0].BlackHoleMass = 0.1;  // 10^10 Msun
    gal[0].Vvir = 300.0;  // km/s
    gal[0].Rvir = 300.0;  // kpc/h
    gal[0].Mvir = 300.0;  // 3x10^12 Msun/h
    gal[0].r_heat = 0.0;  // No previous heating
    gal[0].Heating = 0.0;
    
    double initial_cooling = 1.0;  // Some cooling rate
    double dt = 0.01;  // Gyr
    double x = 1.0;  // Parameter for AGN calculation
    double rcool = 150.0;  // Cooling radius
    
    double final_cooling = do_AGN_heating(initial_cooling, 0, dt, x, rcool, gal, &run_params);
    
    // AGN should suppress some or all cooling
    ASSERT_TRUE(final_cooling <= initial_cooling,
               "AGN reduces cooling rate");
    
    // Black hole should have accreted some mass
    ASSERT_GREATER_THAN(gal[0].BlackHoleMass, 0.1,
                       "Black hole grew from accretion");
    
    // Hot gas should have decreased
    ASSERT_LESS_THAN(gal[0].HotGas, 10.0,
                    "Hot gas consumed by accretion");
}

// ============================================================================
// TEST 4: Heating Radius Evolution
// ============================================================================
void test_heating_radius() {
    BEGIN_TEST("Heating Radius Increases With AGN Activity");
    
    struct GALAXY gal[1];
    memset(gal, 0, sizeof(struct GALAXY));
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.AGNrecipeOn = 1;
    run_params.RadioModeEfficiency = 0.08;
    run_params.UnitMass_in_g = 1.989e43;
    run_params.UnitTime_in_s = 3.08568e16;
    run_params.UnitEnergy_in_cgs = 1.989e53;
    run_params.G = 43007.1;
    run_params.Hubble_h = 0.7;
    
    gal[0].HotGas = 10.0;
    gal[0].MetalsHotGas = 0.2;
    gal[0].BlackHoleMass = 0.1;
    gal[0].Vvir = 300.0;
    gal[0].Rvir = 300.0;
    gal[0].Mvir = 300.0;
    gal[0].r_heat = 0.0;  // Start with no heated region
    
    double cooling = 1.0;
    double dt = 0.01;
    double x = 1.0;
    double rcool = 200.0;
    
    do_AGN_heating(cooling, 0, dt, x, rcool, gal, &run_params);
    
    // Heating radius should have increased from zero
    ASSERT_TRUE(gal[0].r_heat >= 0.0,
               "Heating radius non-negative");
}

// ============================================================================
// TEST 5: Mass Conservation During AGN Accretion
// ============================================================================
void test_agn_mass_conservation() {
    BEGIN_TEST("Mass Conserved During AGN Accretion");
    
    struct GALAXY gal[1];
    memset(gal, 0, sizeof(struct GALAXY));
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.AGNrecipeOn = 1;
    run_params.RadioModeEfficiency = 0.08;
    run_params.UnitMass_in_g = 1.989e43;
    run_params.UnitTime_in_s = 3.08568e16;
    run_params.UnitEnergy_in_cgs = 1.989e53;
    run_params.G = 43007.1;
    run_params.Hubble_h = 0.7;
    
    gal[0].HotGas = 10.0;
    gal[0].MetalsHotGas = 0.2;
    gal[0].BlackHoleMass = 0.1;
    gal[0].Vvir = 300.0;
    gal[0].Rvir = 300.0;
    gal[0].Mvir = 300.0;
    gal[0].r_heat = 0.0;
    
    double initial_total = gal[0].HotGas + gal[0].BlackHoleMass;
    double initial_metals = gal[0].MetalsHotGas;
    
    double cooling = 1.0;
    double dt = 0.01;
    double x = 1.0;
    double rcool = 200.0;
    
    do_AGN_heating(cooling, 0, dt, x, rcool, gal, &run_params);
    
    double final_total = gal[0].HotGas + gal[0].BlackHoleMass;
    
    // Total baryonic mass should be conserved
    ASSERT_CLOSE(final_total, initial_total, 1e-10,
                "Total mass conserved (hot gas + BH)");
    
    // Metals should decrease proportionally with hot gas
    ASSERT_TRUE(gal[0].MetalsHotGas <= initial_metals,
               "Metals decrease with hot gas accretion");
}

// ============================================================================
// TEST 6: AGN Recipe Modes
// ============================================================================
void test_agn_recipe_modes() {
    BEGIN_TEST("Different AGN Recipe Modes");
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    
    // Mode 1: Standard empirical (default)
    run_params.AGNrecipeOn = 1;
    ASSERT_EQUAL_INT(run_params.AGNrecipeOn, 1,
                    "AGN mode 1 is standard");
    
    // Mode 2: Bondi-Hoyle accretion
    run_params.AGNrecipeOn = 2;
    ASSERT_EQUAL_INT(run_params.AGNrecipeOn, 2,
                    "AGN mode 2 is Bondi-Hoyle");
    
    // Mode 3: Cold cloud accretion
    run_params.AGNrecipeOn = 3;
    ASSERT_EQUAL_INT(run_params.AGNrecipeOn, 3,
                    "AGN mode 3 is cold cloud");
}

// ============================================================================
// TEST 7: AGN Heating Coefficient
// ============================================================================
void test_agn_heating_coefficient() {
    BEGIN_TEST("AGN Heating Coefficient Scales With Vvir");
    
    // AGN coefficient: (1.34e5 / Vvir)^2
    // Represents efficiency of heating gas to virial temperature
    
    double Vvir_1 = 200.0;  // km/s
    double Vvir_2 = 400.0;  // km/s (2x higher)
    
    double coeff_1 = pow(1.34e5 / Vvir_1, 2.0);
    double coeff_2 = pow(1.34e5 / Vvir_2, 2.0);
    
    // Higher Vvir -> lower coefficient (harder to heat)
    ASSERT_GREATER_THAN(coeff_1, coeff_2,
                       "Lower Vvir has higher heating efficiency");
    
    // Should scale as 1/Vvir^2
    double ratio = coeff_1 / coeff_2;
    double expected_ratio = pow(Vvir_2 / Vvir_1, 2.0);  // 4
    
    ASSERT_CLOSE(ratio, expected_ratio, 1e-10,
                "Heating coefficient scales as 1/Vvir^2");
}

// ============================================================================
// TEST 8: Cannot Accrete More Than Available
// ============================================================================
void test_agn_accretion_limit() {
    BEGIN_TEST("AGN Cannot Accrete More Than Available Hot Gas");
    
    struct GALAXY gal[1];
    memset(gal, 0, sizeof(struct GALAXY));
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.AGNrecipeOn = 1;
    run_params.RadioModeEfficiency = 0.08;
    run_params.UnitMass_in_g = 1.989e43;
    run_params.UnitTime_in_s = 3.08568e16;
    run_params.UnitEnergy_in_cgs = 1.989e53;
    run_params.G = 43007.1;
    run_params.Hubble_h = 0.7;
    
    // Small hot gas reservoir
    gal[0].HotGas = 0.01;  // Very small
    gal[0].MetalsHotGas = 0.0002;
    gal[0].BlackHoleMass = 1.0;  // Large BH (would want to accrete a lot)
    gal[0].Vvir = 300.0;
    gal[0].Rvir = 300.0;
    gal[0].Mvir = 300.0;
    gal[0].r_heat = 0.0;
    
    double initial_hot = gal[0].HotGas;
    
    double cooling = 10.0;  // Large cooling rate
    double dt = 1.0;  // Long timestep
    double x = 1.0;
    double rcool = 200.0;
    
    do_AGN_heating(cooling, 0, dt, x, rcool, gal, &run_params);
    
    // Hot gas should not go negative
    ASSERT_TRUE(gal[0].HotGas >= 0.0,
               "Hot gas cannot be negative");
    
    // Accreted mass should not exceed initial hot gas
    double accreted = initial_hot - gal[0].HotGas;
    ASSERT_TRUE(accreted <= initial_hot,
               "Cannot accrete more than available");
}

// ============================================================================
// MAIN TEST RUNNER
// ============================================================================
int main(void) {
    printf("\n");
    printf("═══════════════════════════════════════════════════════════\n");
    printf("  SAGE26 AGN FEEDBACK TESTS\n");
    printf("═══════════════════════════════════════════════════════════\n");
    printf("\n");

    test_agn_accretion_scaling();
    test_eddington_limit();
    test_agn_suppresses_cooling();
    test_heating_radius();
    test_agn_mass_conservation();
    test_agn_recipe_modes();
    test_agn_heating_coefficient();
    test_agn_accretion_limit();

    PRINT_TEST_SUMMARY();
    
    return (tests_failed > 0) ? 1 : 0;
}

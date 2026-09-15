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
    strip_from_satellite(0, 1, 0.0, 1, galaxies, &run_params);
    
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
    
    strip_from_satellite(0, 1, 0.0, 1, galaxies, &run_params);
    
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
        
        strip_from_satellite(0, 1, 0.0, 1, galaxies, &run_params);
        
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
        
        strip_from_satellite(0, 1, 0.0, 1, galaxies, &run_params);
        
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
    
    strip_from_satellite(0, 1, 0.0, 1, galaxies, &run_params);
    
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
    
    strip_from_satellite(0, 1, 0.0, 1, galaxies, &run_params);
    
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
    
    strip_from_satellite(0, 1, 0.0, 1, galaxies, &run_params);
    
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
    
    strip_from_satellite(0, 1, 0.0, 1, galaxies, &run_params);
    
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
    
    strip_from_satellite(0, 1, 0.0, 1, galaxies, &run_params);
    
    double stripped = initial_hot - galaxies[1].HotGas;
    
    // Should strip a fraction per timestep, not all at once
    // (divided by STEPS in code)
    if(stripped > 0.0) {
        ASSERT_LESS_THAN(stripped, initial_hot,
                        "Doesn't strip all gas in one step");
    }
}

/* Helper: build a satellite whose only strippable reservoir is HotGas, strip it
 * once with the given substep count, and return the HotGas remaining. The
 * satellite holds 5.0 against a BaryonFrac*Mvir floor of 1.7, so the excess
 * that strip_from_satellite() acts on is exactly 3.3. */
static double strip_hotgas_once(int nsteps) {
    struct GALAXY galaxies[2];
    memset(galaxies, 0, sizeof(struct GALAXY) * 2);

    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.CGMrecipeOn = 0;
    run_params.BaryonFrac = 0.17;
    run_params.ReionizationOn = 0;

    galaxies[0].Regime = 1;
    galaxies[1].Mvir = 10.0;           // BaryonFrac*Mvir = 1.7
    galaxies[1].HotGas = 5.0;          // excess = 3.3
    galaxies[1].MetalsHotGas = 0.05;

    strip_from_satellite(0, 1, 0.0, nsteps, galaxies, &run_params);
    return galaxies[1].HotGas;
}

/* Single-precision HotGas accumulates round-off over repeated strips; 1e-5 is
 * several ulp at these magnitudes and well inside any physical difference. */
static const double STRIP_FRACTION_TOL = 1.0e-5;

void test_stripping_removes_excess_over_nsteps() {
    BEGIN_TEST("Stripping removes excess/nsteps in a single call");

    /*
     * Asserts: one call to strip_from_satellite() removes exactly 1/nsteps of
     *          the satellite's current baryon excess.
     *
     * Setup: a satellite with HotGas = 5.0 against a BaryonFrac*Mvir floor of
     *        1.7, so the excess is 3.3, stripped with nsteps = 4.
     *
     * Physical reason: the prescription divides the excess by the number of
     *        substeps because it is invoked once per substep, so that a whole
     *        snapshot removes the excess progressively rather than at once.
     *        With nsteps = 4 a single call takes 3.3/4 = 0.825, leaving 4.175.
     */
    const int nsteps = 4;
    const double excess0 = 5.0 - 0.17 * 10.0;
    const double expected = 5.0 - excess0 / nsteps;

    ASSERT_CLOSE(expected, strip_hotgas_once(nsteps), STRIP_FRACTION_TOL,
                 "One call strips exactly excess/nsteps");
}

void test_stripping_full_snapshot_leaves_one_over_e() {
    BEGIN_TEST("Stripping over a full snapshot leaves ~1/e of the excess");

    /*
     * Asserts: calling strip_from_satellite() nsteps times with divisor nsteps
     *          -- one whole snapshot's worth of substeps -- leaves
     *          (1 - 1/nsteps)^nsteps of the original excess.
     *
     * Setup: the same satellite as above, stripped 16 times with nsteps = 16.
     *
     * Physical reason: each call removes 1/nsteps of whatever excess remains,
     *        so the excess decays geometrically. For nsteps = 16 the surviving
     *        fraction is (15/16)^16 = 0.356, tending to 1/e = 0.368 as the
     *        substep count grows. This is the behaviour of the original SAGE
     *        prescription and is the reason a full snapshot removes roughly
     *        63 per cent of the excess regardless of how finely it is split.
     */
    const int nsteps = 16;
    struct GALAXY galaxies[2];
    memset(galaxies, 0, sizeof(struct GALAXY) * 2);

    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.CGMrecipeOn = 0;
    run_params.BaryonFrac = 0.17;
    run_params.ReionizationOn = 0;

    galaxies[0].Regime = 1;
    galaxies[1].Mvir = 10.0;
    galaxies[1].HotGas = 5.0;
    galaxies[1].MetalsHotGas = 0.05;

    for(int i = 0; i < nsteps; i++) {
        strip_from_satellite(0, 1, 0.0, nsteps, galaxies, &run_params);
    }

    const double floor_mass = 0.17 * 10.0;
    const double excess0 = 5.0 - floor_mass;
    const double expected = floor_mass + excess0 * pow(1.0 - 1.0 / nsteps, nsteps);

    ASSERT_CLOSE(expected, galaxies[1].HotGas, STRIP_FRACTION_TOL,
                 "A full snapshot leaves (1-1/nsteps)^nsteps of the excess");
    ASSERT_GREATER_THAN(galaxies[1].HotGas, floor_mass - 1e-9,
                        "Never strips below the BaryonFrac*Mvir floor");
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
    test_stripping_removes_excess_over_nsteps();
    test_stripping_full_snapshot_leaves_one_over_e();

    END_TEST_SUITE();
    PRINT_TEST_SUMMARY();

    return TEST_EXIT_CODE();
}

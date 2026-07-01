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
    strip_from_satellite(0, 1, 0.0, STEPS, 0.0, 0.0, galaxies, &run_params);
    
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
    
    strip_from_satellite(0, 1, 0.0, STEPS, 0.0, 0.0, galaxies, &run_params);
    
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
        
        strip_from_satellite(0, 1, 0.0, STEPS, 0.0, 0.0, galaxies, &run_params);
        
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
        
        strip_from_satellite(0, 1, 0.0, STEPS, 0.0, 0.0, galaxies, &run_params);
        
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
    
    strip_from_satellite(0, 1, 0.0, STEPS, 0.0, 0.0, galaxies, &run_params);
    
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
    
    strip_from_satellite(0, 1, 0.0, STEPS, 0.0, 0.0, galaxies, &run_params);
    
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
    
    strip_from_satellite(0, 1, 0.0, STEPS, 0.0, 0.0, galaxies, &run_params);
    
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
    
    strip_from_satellite(0, 1, 0.0, STEPS, 0.0, 0.0, galaxies, &run_params);
    
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
    
    strip_from_satellite(0, 1, 0.0, STEPS, 0.0, 0.0, galaxies, &run_params);
    
    double stripped = initial_hot - galaxies[1].HotGas;
    
    // Should strip a fraction per timestep, not all at once
    // (divided by STEPS in code)
    if(stripped > 0.0) {
        ASSERT_LESS_THAN(stripped, initial_hot,
                        "Doesn't strip all gas in one step");
    }
}

/*
 * Physical (timescale-based) stripping: PhysicalStrippingOn = 1.
 * These tests exercise the Option-2 path where the fraction of the satellite's
 * baryon excess stripped over one snapshot interval is 1-exp(-dT/t_strip),
 * independent of the substep count in the large-N limit.
 */

/* Helper: strip a satellite over N substeps of one snapshot of duration dT,
 * on the physical path, and return the fraction of the initial excess removed.
 * The satellite's only strippable/variable reservoir is HotGas, so the excess
 * evolves purely by stripping and telescopes cleanly. */
static double physical_strip_excess_fraction(int N, double dT, double t_strip) {
    struct GALAXY galaxies[2];
    memset(galaxies, 0, sizeof(struct GALAXY) * 2);

    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.PhysicalStrippingOn = 1;
    run_params.CGMrecipeOn = 0;
    run_params.BaryonFrac = 0.17;
    run_params.ReionizationOn = 0;

    galaxies[0].Regime = 1;            // central sink
    galaxies[0].HotGas = 0.0;

    galaxies[1].Mvir = 10.0;           // BF*Mvir = 1.7
    galaxies[1].HotGas = 5.0;          // excess_0 = 5.0 - 1.7 = 3.3, all in HotGas
    galaxies[1].MetalsHotGas = 0.05;

    const double excess0 = galaxies[1].HotGas - run_params.BaryonFrac * galaxies[1].Mvir;
    const double dt = dT / (double)N;
    for(int step = 0; step < N; step++) {
        strip_from_satellite(0, 1, 0.0, N, dt, t_strip, galaxies, &run_params);
    }
    const double excessN = galaxies[1].HotGas - run_params.BaryonFrac * galaxies[1].Mvir;
    return (excess0 - excessN) / excess0;
}

void test_physical_stripping_matches_exponential() {
    BEGIN_TEST("Physical Stripping Follows 1-exp(-dT/t_strip)");

    const double dT = 1.0, t_strip = 2.0;
    const int N = 30;
    const double got = physical_strip_excess_fraction(N, dT, t_strip);

    // Discrete forward-Euler value the code should produce exactly...
    const double discrete = 1.0 - pow(1.0 - dT / (N * t_strip), N);
    ASSERT_CLOSE(discrete, got, 1e-6,
                 "Matches discrete 1-(1-dT/(N*t_strip))^N");

    // ...which is close to the continuum limit 1-exp(-dT/t_strip).
    const double continuum = 1.0 - exp(-dT / t_strip);
    ASSERT_CLOSE(continuum, got, 0.01,
                 "Close to continuum 1-exp(-dT/t_strip)");
}

void test_physical_stripping_is_N_invariant() {
    BEGIN_TEST("Physical Stripping Is Substep-Count Invariant");

    const double dT = 1.0, t_strip = 2.0;
    const double f10 = physical_strip_excess_fraction(10, dT, t_strip);
    const double f30 = physical_strip_excess_fraction(30, dT, t_strip);

    // Both track the same physical fraction regardless of N (unlike the legacy
    // geometric path, whose fraction swings ~65%->64% and up to 100% at N=1).
    // ~1.3% residual is the expected first-order forward-Euler difference.
    ASSERT_CLOSE(f10, f30, 0.02,
                 "Stripped fraction nearly independent of substep count");
    const double continuum = 1.0 - exp(-dT / t_strip);
    ASSERT_CLOSE(continuum, f10, 0.02, "N=10 near continuum");
    ASSERT_CLOSE(continuum, f30, 0.02, "N=30 near continuum");
}

void test_physical_stripping_caps_at_full_excess() {
    BEGIN_TEST("Physical Stripping Caps When dt >= t_strip");

    struct GALAXY galaxies[2];
    memset(galaxies, 0, sizeof(struct GALAXY) * 2);

    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.PhysicalStrippingOn = 1;
    run_params.CGMrecipeOn = 0;
    run_params.BaryonFrac = 0.17;
    run_params.ReionizationOn = 0;

    galaxies[0].Regime = 1;
    galaxies[1].Mvir = 10.0;           // BF*Mvir = 1.7
    galaxies[1].HotGas = 5.0;          // excess = 3.3
    galaxies[1].MetalsHotGas = 0.05;

    // dt (3.0) > t_strip (1.0): frac caps at 1.0, whole excess stripped at once.
    strip_from_satellite(0, 1, 0.0, 1, 3.0, 1.0, galaxies, &run_params);

    ASSERT_CLOSE(1.7, galaxies[1].HotGas, 1e-5,
                 "HotGas driven down to BF*Mvir (full excess removed)");
}

/* Helper for scheme 2: strip once with dt = full dT, return HotGas remaining.
 * effective_steps is passed but must not affect the result. */
static double analytic_strip_hotgas(int effective_steps, double dT, double t_strip) {
    struct GALAXY galaxies[2];
    memset(galaxies, 0, sizeof(struct GALAXY) * 2);

    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    run_params.PhysicalStrippingOn = 2;
    run_params.CGMrecipeOn = 0;
    run_params.BaryonFrac = 0.17;
    run_params.ReionizationOn = 0;

    galaxies[0].Regime = 1;
    galaxies[1].Mvir = 10.0;           // BF*Mvir = 1.7
    galaxies[1].HotGas = 5.0;          // excess = 3.3
    galaxies[1].MetalsHotGas = 0.05;

    strip_from_satellite(0, 1, 0.0, effective_steps, dT, t_strip, galaxies, &run_params);
    return galaxies[1].HotGas;
}

void test_analytic_stripping_matches_exact_exponential() {
    BEGIN_TEST("Analytic Stripping (mode 2) = exact 1-exp(-dT/t_strip)");

    const double dT = 1.0, t_strip = 2.0;
    const double excess0 = 5.0 - 0.17 * 10.0;      // 3.3
    const double hot = analytic_strip_hotgas(10, dT, t_strip);

    // Exact: HotGas -> BF*Mvir + excess0*exp(-dT/t_strip), in ONE call.
    const double expected = 0.17 * 10.0 + excess0 * exp(-dT / t_strip);
    ASSERT_CLOSE(expected, hot, 1e-6, "Strips exactly 1-exp(-dT/t_strip) of the excess");
}

void test_analytic_stripping_ignores_substep_count() {
    BEGIN_TEST("Analytic Stripping (mode 2) Independent of effective_steps");

    const double dT = 1.0, t_strip = 2.0;
    const double h_lo = analytic_strip_hotgas(2, dT, t_strip);
    const double h_hi = analytic_strip_hotgas(999, dT, t_strip);

    // effective_steps must not enter mode 2 at all -> identical to machine precision.
    ASSERT_CLOSE(h_lo, h_hi, 1e-9, "Result identical for effective_steps 2 vs 999");
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
    test_physical_stripping_matches_exponential();
    test_physical_stripping_is_N_invariant();
    test_physical_stripping_caps_at_full_excess();
    test_analytic_stripping_matches_exact_exponential();
    test_analytic_stripping_ignores_substep_count();

    END_TEST_SUITE();
    PRINT_TEST_SUMMARY();

    return TEST_EXIT_CODE();
}

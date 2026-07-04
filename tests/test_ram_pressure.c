/*
 * RAM-PRESSURE ISM STRIPPING TESTS
 *
 * Tests for the Gunn & Gott (1972) stripping of satellite ColdGas
 * (model_ram_pressure.c, enabled by RamPressureStrippingOn):
 * - f_strip bounds, limits, and monotonicity in P_ram
 * - f_strip = 1 when ram pressure beats the restoring force everywhere
 * - f_strip -> 0 for deep potentials / weak ram pressure
 * - mass and metal conservation across ram_pressure_strip_satellite()
 * - regime-aware routing of the stripped gas (HotGas vs CGMgas host reservoir)
 * - gradual removal: 1 - exp(-dt/t_strip) cadence factor
 * - guards: no ambient medium, gas-free satellite, degenerate velocity
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "test_framework.h"
#include "../src/core_allvars.h"
#include "../src/model_misc.h"
#include "../src/model_ram_pressure.h"

/* Central restoring-force scale used across the criterion tests:
 * 2*pi*G*Sigma_disk0*Sigma_gas0 with Sigma_gas0 = 5e-3, Sigma_disk0 = 8e-3
 * g/cm^2 (a gas-rich dwarf disk) is ~1.7e-11 erg/cm^3. */
static const double SIGMA_GAS0 = 5.0e-3;   /* g/cm^2 */
static const double SIGMA_DISK0 = 8.0e-3;  /* g/cm^2 */
static const double RESTORING0 = 2.0 * M_PI * 6.674e-8 * 8.0e-3 * 5.0e-3;

void test_fstrip_bounds() {
    BEGIN_TEST("f_strip: bounded in [0,1] over 12 decades of ram pressure");

    for(double P_ram = 1e-18; P_ram < 1e-6; P_ram *= 10.0) {
        double f = ram_pressure_stripped_fraction(P_ram, SIGMA_GAS0, SIGMA_DISK0);
        ASSERT_IN_RANGE(f, 0.0, 1.0, "f_strip in [0,1]");
    }
}

void test_fstrip_guards() {
    BEGIN_TEST("f_strip: zero for non-positive pressure or surface densities");

    ASSERT_EQUAL_FLOAT(ram_pressure_stripped_fraction(0.0, SIGMA_GAS0, SIGMA_DISK0), 0.0,
                       "Zero ram pressure strips nothing");
    ASSERT_EQUAL_FLOAT(ram_pressure_stripped_fraction(-1.0, SIGMA_GAS0, SIGMA_DISK0), 0.0,
                       "Negative ram pressure strips nothing");
    ASSERT_EQUAL_FLOAT(ram_pressure_stripped_fraction(1e-12, 0.0, SIGMA_DISK0), 0.0,
                       "Zero gas surface density strips nothing");
    ASSERT_EQUAL_FLOAT(ram_pressure_stripped_fraction(1e-12, SIGMA_GAS0, 0.0), 0.0,
                       "Zero disk surface density strips nothing");
}

void test_fstrip_monotonic_in_pram() {
    BEGIN_TEST("f_strip: monotonically non-decreasing in P_ram");

    double f_prev = 0.0;
    for(double P_ram = 1e-16; P_ram < 1e-8; P_ram *= 2.0) {
        double f = ram_pressure_stripped_fraction(P_ram, SIGMA_GAS0, SIGMA_DISK0);
        ASSERT_TRUE(f >= f_prev, "f_strip does not decrease as P_ram grows");
        f_prev = f;
    }
}

void test_fstrip_limits() {
    BEGIN_TEST("f_strip: full stripping above the central restoring force, ~0 far below");

    ASSERT_EQUAL_FLOAT(ram_pressure_stripped_fraction(RESTORING0, SIGMA_GAS0, SIGMA_DISK0), 1.0,
                       "P_ram = central restoring force gives f_strip = 1 (r_strip = 0)");
    ASSERT_EQUAL_FLOAT(ram_pressure_stripped_fraction(10.0 * RESTORING0, SIGMA_GAS0, SIGMA_DISK0), 1.0,
                       "P_ram above the central restoring force gives f_strip = 1");
    ASSERT_LESS_THAN(ram_pressure_stripped_fraction(1e-10 * RESTORING0, SIGMA_GAS0, SIGMA_DISK0),
                     1e-3, "Deep potential (P_ram << restoring force) gives f_strip ~ 0");
}

void test_fstrip_analytic_value() {
    BEGIN_TEST("f_strip: matches the analytic (1 + x) exp(-x) with x = ln(ratio)/2");

    /* P_ram a factor e^4 below the central restoring force: x = 2 exactly. */
    const double P_ram = RESTORING0 * exp(-4.0);
    const double expected = 3.0 * exp(-2.0);
    ASSERT_CLOSE(expected, ram_pressure_stripped_fraction(P_ram, SIGMA_GAS0, SIGMA_DISK0),
                 1e-12, "f_strip(x=2) = 3 e^-2");
}

/* Shared two-galaxy setup: a 10^12 Msun/h hot-regime central hosting a
 * gas-rich satellite at 0.1 Mpc/h moving at 500 km/s. Values chosen so a
 * healthy fraction (but not all) of the disk strips (f_strip ~ 0.4). */
static void setup_host_and_satellite(struct GALAXY *galaxies, struct params *run_params)
{
    memset(galaxies, 0, sizeof(struct GALAXY) * 2);
    memset(run_params, 0, sizeof(struct params));

    run_params->Hubble_h = 0.7;
    run_params->BoxSize = 62.5;
    run_params->CGMrecipeOn = 0;
    run_params->RamPressureStrippingOn = 1;
    run_params->RamPressureEpsilon = 1.0;

    /* central: hot-regime host */
    galaxies[0].Type = 0;
    galaxies[0].Regime = 1;
    galaxies[0].Mvir = 100.0;
    galaxies[0].Rvir = 0.2;
    galaxies[0].Vvir = 150.0;
    galaxies[0].HotGas = 10.0;
    galaxies[0].MetalsHotGas = 0.1;
    galaxies[0].Pos[0] = 50.0; galaxies[0].Pos[1] = 50.0; galaxies[0].Pos[2] = 50.0;

    /* satellite: cold disk 0.1 (10^9 Msun/h) with a 2 kpc/h scale radius */
    galaxies[1].Type = 1;
    galaxies[1].ColdGas = 0.1;
    galaxies[1].MetalsColdGas = 0.002;
    galaxies[1].StellarMass = 0.05;
    galaxies[1].BulgeMass = 0.01;
    galaxies[1].DiskScaleRadius = 0.002;
    galaxies[1].Pos[0] = 50.1; galaxies[1].Pos[1] = 50.0; galaxies[1].Pos[2] = 50.0;
    galaxies[1].Vel[0] = 500.0;
}

void test_strip_conserves_mass_and_metals() {
    BEGIN_TEST("Driver: satellite + central mass and metals conserved across the strip");

    struct GALAXY galaxies[2];
    struct params run_params;
    setup_host_and_satellite(galaxies, &run_params);

    const double total_gas0 = galaxies[1].ColdGas + galaxies[0].HotGas;
    const double total_met0 = galaxies[1].MetalsColdGas + galaxies[0].MetalsHotGas;
    const double cold0 = galaxies[1].ColdGas;

    /* t_strip = 0 -> full f_strip removed in one call */
    ram_pressure_strip_satellite(0, 1, 0.0, 1.0, 0.0, galaxies, &run_params);

    ASSERT_LESS_THAN(galaxies[1].ColdGas, cold0, "Satellite ColdGas decreased");
    ASSERT_GREATER_THAN(galaxies[1].ColdGas, 0.0, "Partial strip leaves some ColdGas");
    ASSERT_CLOSE(total_gas0, galaxies[1].ColdGas + galaxies[0].HotGas, 1e-6,
                 "ColdGas + host HotGas conserved");
    ASSERT_CLOSE(total_met0, galaxies[1].MetalsColdGas + galaxies[0].MetalsHotGas, 1e-6,
                 "Metals conserved");

    /* stripped gas carries the satellite's cold-gas metallicity */
    const double stripped = cold0 - galaxies[1].ColdGas;
    const double stripped_met = galaxies[0].MetalsHotGas - 0.1;
    ASSERT_CLOSE(0.002 / 0.1, stripped_met / stripped, 1e-5,
                 "Stripped gas has the satellite's metallicity");
}

void test_strip_routes_to_cgm_reservoir() {
    BEGIN_TEST("Driver: CGM-regime host receives stripped gas in CGMgas");

    struct GALAXY galaxies[2];
    struct params run_params;
    setup_host_and_satellite(galaxies, &run_params);

    run_params.CGMrecipeOn = 1;
    galaxies[0].Regime = 0;
    galaxies[0].CGMgas = galaxies[0].HotGas;
    galaxies[0].MetalsCGMgas = galaxies[0].MetalsHotGas;
    galaxies[0].HotGas = 0.0;
    galaxies[0].MetalsHotGas = 0.0;

    const double cgm0 = galaxies[0].CGMgas;
    const double cold0 = galaxies[1].ColdGas;

    ram_pressure_strip_satellite(0, 1, 0.0, 1.0, 0.0, galaxies, &run_params);

    ASSERT_LESS_THAN(galaxies[1].ColdGas, cold0, "Satellite ColdGas decreased");
    ASSERT_GREATER_THAN(galaxies[0].CGMgas, cgm0, "Host CGMgas increased");
    ASSERT_EQUAL_FLOAT(galaxies[0].HotGas, 0.0, "Host HotGas untouched in CGM regime");
    ASSERT_CLOSE(cold0 + cgm0, galaxies[1].ColdGas + galaxies[0].CGMgas, 1e-6,
                 "ColdGas + host CGMgas conserved");
}

void test_strip_gradual_timescale() {
    BEGIN_TEST("Driver: removal follows the 1 - exp(-dt/t_strip) cadence factor");

    struct GALAXY sudden[2], gradual[2];
    struct params run_params;
    setup_host_and_satellite(sudden, &run_params);
    setup_host_and_satellite(gradual, &run_params);

    /* identical states: the sudden call (t_strip = 0) removes the full
     * f_strip fraction; the gradual call removes exactly
     * (1 - exp(-dt/t_strip)) of it. */
    const double cold0 = sudden[1].ColdGas;
    const double dt = 1.0, t_strip = 2.0;

    ram_pressure_strip_satellite(0, 1, 0.0, dt, 0.0, sudden, &run_params);
    ram_pressure_strip_satellite(0, 1, 0.0, dt, t_strip, gradual, &run_params);

    const double full_strip = cold0 - sudden[1].ColdGas;
    const double part_strip = cold0 - gradual[1].ColdGas;

    ASSERT_GREATER_THAN(full_strip, 0.0, "Sudden strip removes gas");
    ASSERT_CLOSE(full_strip * (1.0 - exp(-dt / t_strip)), part_strip, 1e-4,
                 "Gradual strip = full strip * (1 - exp(-dt/t_strip))");
}

void test_strip_guards() {
    BEGIN_TEST("Driver: guards -- no ambient medium, no cold gas, no disk radius");

    struct GALAXY galaxies[2];
    struct params run_params;

    /* host with no hot-phase gas: nothing to strip against */
    setup_host_and_satellite(galaxies, &run_params);
    galaxies[0].HotGas = 0.0;
    galaxies[0].MetalsHotGas = 0.0;
    double cold0 = galaxies[1].ColdGas;
    ram_pressure_strip_satellite(0, 1, 0.0, 1.0, 0.0, galaxies, &run_params);
    ASSERT_EQUAL_FLOAT(galaxies[1].ColdGas, cold0, "No ambient medium: no stripping");

    /* satellite with no cold gas: no-op */
    setup_host_and_satellite(galaxies, &run_params);
    galaxies[1].ColdGas = 0.0;
    galaxies[1].MetalsColdGas = 0.0;
    ram_pressure_strip_satellite(0, 1, 0.0, 1.0, 0.0, galaxies, &run_params);
    ASSERT_EQUAL_FLOAT(galaxies[1].ColdGas, 0.0, "Gas-free satellite stays gas-free");
    ASSERT_EQUAL_FLOAT(galaxies[0].HotGas, 10.0, "Host unchanged for gas-free satellite");

    /* satellite with no disk scale radius: surface densities undefined, no-op */
    setup_host_and_satellite(galaxies, &run_params);
    galaxies[1].DiskScaleRadius = 0.0;
    cold0 = galaxies[1].ColdGas;
    ram_pressure_strip_satellite(0, 1, 0.0, 1.0, 0.0, galaxies, &run_params);
    ASSERT_EQUAL_FLOAT(galaxies[1].ColdGas, cold0, "Zero disk radius: no stripping");
}

void test_strip_degenerate_velocity_fallback() {
    BEGIN_TEST("Driver: degenerate velocity difference falls back to host Vvir");

    struct GALAXY galaxies[2];
    struct params run_params;
    setup_host_and_satellite(galaxies, &run_params);

    /* identical velocities: v_sat comes from the host Vvir instead */
    galaxies[1].Vel[0] = 0.0;
    const double cold0 = galaxies[1].ColdGas;

    ram_pressure_strip_satellite(0, 1, 0.0, 1.0, 0.0, galaxies, &run_params);

    ASSERT_LESS_THAN(galaxies[1].ColdGas, cold0,
                     "Vvir fallback still produces finite stripping");
    ASSERT_TRUE(isfinite(galaxies[1].ColdGas) && galaxies[1].ColdGas >= 0.0,
                "ColdGas stays finite and non-negative");
}

void test_orphan_uses_host_vvir() {
    BEGIN_TEST("Driver: Type 2 orphans ignore their frozen velocity and use host Vvir");

    /* an orphan with an absurd frozen velocity must strip exactly like a
     * Type 1 satellite moving at the host Vvir from the same position */
    struct GALAXY orphan[2], reference[2];
    struct params run_params;
    setup_host_and_satellite(orphan, &run_params);
    setup_host_and_satellite(reference, &run_params);

    orphan[1].Type = 2;
    orphan[1].Vel[0] = 9999.0;          /* stale junk: must not be used */

    reference[1].Type = 1;
    reference[1].Vel[0] = reference[0].Vvir;   /* moving at exactly Vvir */

    ram_pressure_strip_satellite(0, 1, 0.0, 1.0, 0.0, orphan, &run_params);
    ram_pressure_strip_satellite(0, 1, 0.0, 1.0, 0.0, reference, &run_params);

    ASSERT_LESS_THAN(orphan[1].ColdGas, 0.1, "Orphan is stripped");
    ASSERT_CLOSE(reference[1].ColdGas, orphan[1].ColdGas, 1e-6,
                 "Orphan strip equals Type 1 at v = Vvir (frozen velocity ignored)");
}

void test_strip_weaker_at_larger_radius() {
    BEGIN_TEST("Driver: satellite further out in the isothermal host strips less");

    struct GALAXY inner[2], outer[2];
    struct params run_params;
    setup_host_and_satellite(inner, &run_params);
    setup_host_and_satellite(outer, &run_params);

    outer[1].Pos[0] = 50.2;   /* 0.2 Mpc/h = Rvir instead of 0.1 */

    const double cold0 = inner[1].ColdGas;
    ram_pressure_strip_satellite(0, 1, 0.0, 1.0, 0.0, inner, &run_params);
    ram_pressure_strip_satellite(0, 1, 0.0, 1.0, 0.0, outer, &run_params);

    const double stripped_inner = cold0 - inner[1].ColdGas;
    const double stripped_outer = cold0 - outer[1].ColdGas;
    ASSERT_GREATER_THAN(stripped_inner, stripped_outer,
                        "rho_host ~ 1/r^2: less stripping at larger orbital radius");
}

int main() {
    BEGIN_TEST_SUITE("Ram-Pressure ISM Stripping");

    test_fstrip_bounds();
    test_fstrip_guards();
    test_fstrip_monotonic_in_pram();
    test_fstrip_limits();
    test_fstrip_analytic_value();
    test_strip_conserves_mass_and_metals();
    test_strip_routes_to_cgm_reservoir();
    test_strip_gradual_timescale();
    test_strip_guards();
    test_strip_degenerate_velocity_fallback();
    test_orphan_uses_host_vvir();
    test_strip_weaker_at_larger_radius();

    END_TEST_SUITE();
    PRINT_TEST_SUMMARY();

    return TEST_EXIT_CODE();
}

/*
 * test_rheat.c -- unit tests for the AGN heating radius r_heat across the regimes.
 *
 * Validates: the CGM recipe (Regime 0) has no radio mode and neither reads
 * nor writes r_heat, so a halo whose r_heat reached Rvir still cools; the
 * hot-halo radio mode does not cap r_heat at Rvir.
 *
 * Run: from tests/, `make test_rheat && ./test_build/test_rheat`.
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "test_framework.h"
#include "../src/core_allvars.h"
#include "../src/core_cool_func.h"
#include "../src/model_cooling_heating.h"

/* r_heat is a float field: an untouched value comes back exactly. */
static const double RHEAT_UNCHANGED_TOL = 1e-6;

/* Relative tolerance on cooling masses computed by the same double-precision
 * expression in two calls. */
static const double COOLING_REL_TOL = 1e-12;

/* One substep, in code time units (~1 Myr). */
static const double DT = 0.001;

/* SAGE standard units (Mpc/h, km/s, 10^10 Msun/h), CGM regimes on and the
 * Bondi radio mode, as in test_regime_cgm.c. */
static void setup_params(struct params *run_params)
{
    memset(run_params, 0, sizeof(struct params));
    run_params->MShockMsun = 6.0e11;
    run_params->Hubble_h = 0.7;
    run_params->UnitLength_in_cm = 3.08568e24;
    run_params->UnitVelocity_in_cm_per_s = 1e5;
    run_params->UnitMass_in_g = 1.989e43;
    run_params->UnitTime_in_s = run_params->UnitLength_in_cm / run_params->UnitVelocity_in_cm_per_s;
    run_params->UnitDensity_in_cgs = run_params->UnitMass_in_g / pow(run_params->UnitLength_in_cm, 3);
    run_params->UnitEnergy_in_cgs = run_params->UnitMass_in_g * pow(run_params->UnitVelocity_in_cm_per_s, 2);
    run_params->G = 6.672e-8 / pow(run_params->UnitLength_in_cm, 3)
                    * run_params->UnitMass_in_g * pow(run_params->UnitTime_in_s, 2);
    run_params->CGMrecipeOn = 1;
    run_params->AGNrecipeOn = 2;
    run_params->RadioModeEfficiency = 0.08;
}

/* A CGM-regime central like those locked out at z = 0 in v0.95: Mvir = 3 x
 * 10^11 Msun/h with 10% of it as CGM, and a 10^7 Msun/h black hole.  Rvir
 * follows from Mvir and Vvir (G ~ 43 in code units). */
static double cgm_cooling(const double r_heat_over_rvir, struct GALAXY *gal)
{
    struct params run_params;
    setup_params(&run_params);
    memset(gal, 0, sizeof(struct GALAXY));
    gal->Regime = 0;
    gal->Mvir = 30.0;
    gal->Vvir = 120.0;
    gal->Rvir = 43.0 * gal->Mvir / (gal->Vvir * gal->Vvir);
    gal->CGMgas = 3.0;
    gal->MetalsCGMgas = 0.003;
    gal->BlackHoleMass = 0.001;
    gal->r_heat = r_heat_over_rvir * gal->Rvir;
    return cooling_recipe_cgm(0, /*halo_snapnum*/ 0, DT, gal, &run_params);
}

/*
 * Asserts: the CGM recipe leaves r_heat, the black hole and the radio-mode
 *          tallies alone, and its cooling does not depend on the stored r_heat.
 *
 * Setup: the same CGM-regime central with r_heat = 0, 0.5 Rvir and Rvir,
 *        through cooling_recipe_cgm().
 *
 * Why: r_heat is the entropy floor of a hot atmosphere, which a Regime 0 halo
 *        lacks.  In v0.95 r_heat = Rvir gave zero CGM cooling for good.
 */
void test_cgm_recipe_ignores_rheat(void)
{
    BEGIN_TEST("CGM recipe ignores r_heat, no radio mode");

    struct GALAXY gal;

    const double free_cool = cgm_cooling(0.0, &gal);
    ASSERT_GREATER_THAN(free_cool, 0.0, "the CGM cools");
    ASSERT_CLOSE(0.0, gal.r_heat, RHEAT_UNCHANGED_TOL, "r_heat stays 0");
    ASSERT_CLOSE((float)0.001, gal.BlackHoleMass, 1e-15, "no radio-mode black-hole growth (float field)");
    ASSERT_CLOSE(0.0, gal.RadioModeBHaccretionMass[0], 1e-30, "no radio-mode accretion tallied");
    ASSERT_CLOSE(0.0, gal.Heating, 1e-30, "no heating tallied");

    const double half_cool = cgm_cooling(0.5, &gal);
    ASSERT_CLOSE(free_cool, half_cool, COOLING_REL_TOL * free_cool, "r_heat = 0.5 Rvir does not suppress");
    ASSERT_CLOSE(0.5 * gal.Rvir, gal.r_heat, RHEAT_UNCHANGED_TOL, "r_heat = 0.5 Rvir unchanged");

    const double full_cool = cgm_cooling(1.0, &gal);
    ASSERT_CLOSE(free_cool, full_cool, COOLING_REL_TOL * free_cool, "r_heat = Rvir does not suppress");
    ASSERT_CLOSE(gal.Rvir, gal.r_heat, RHEAT_UNCHANGED_TOL, "r_heat = Rvir unchanged");
}

/*
 * Asserts: the hot-halo radio mode keeps an r_heat beyond Rvir.
 *
 * Setup: a hot-regime central with r_heat = 1.5 Rvir and rcool = 2 Rvir
 *        (a halo cooling partially with its floor beyond Rvir), through
 *        do_AGN_heating() with no black hole, so the ratchet does not move.
 *
 * Why: up to v0.95 r_heat was capped at Rvir with CGMrecipeOn > 0, a
 *        leftover of the removed rcool <= Rvir cap; SAGE16 has no such cap.
 */
void test_hot_rheat_no_rvir_cap(void)
{
    BEGIN_TEST("hot-halo radio mode: no Rvir cap on r_heat");

    struct params run_params;
    setup_params(&run_params);
    struct GALAXY gal;
    memset(&gal, 0, sizeof(gal));
    gal.Regime = 1;
    gal.Mvir = 1000.0;
    gal.Vvir = 450.0;
    gal.Rvir = 0.45;
    gal.HotGas = 100.0;
    gal.MetalsHotGas = 2.0;
    gal.r_heat = 1.5 * gal.Rvir;

    const double cooling = do_AGN_heating(1.0, 0, /*halo_snapnum*/ 0, DT, 1.0, 2.0 * gal.Rvir, &gal, &run_params);
    ASSERT_CLOSE(1.5 * gal.Rvir, gal.r_heat, RHEAT_UNCHANGED_TOL, "r_heat kept at 1.5 Rvir");
    ASSERT_CLOSE(0.25, cooling, RHEAT_UNCHANGED_TOL, "cooling suppressed by 1 - r_heat / rcool = 0.25 (float r_heat)");
}

int main(void)
{
    BEGIN_TEST_SUITE("r_heat across the regimes");

    read_cooling_functions();

    test_cgm_recipe_ignores_rheat();
    test_hot_rheat_no_rvir_cap();

    END_TEST_SUITE();
    PRINT_TEST_SUMMARY();

    return TEST_EXIT_CODE();
}

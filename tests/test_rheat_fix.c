/*
 * test_rheat_fix.c -- unit tests for the RheatFix options (r_heat in the CGM regime).
 *
 * Validates: RheatFix 1 (the CGM recipe neither reads nor writes r_heat and
 * has no radio mode), 2 (no hot-regime Rvir cap), 3 (memoryless Regime 0
 * radio mode) and 4 (moving CGM cooling radius), each against RheatFix 0
 * (v0.95).
 *
 * Run: from tests/, `make test_rheat_fix && ./test_build/test_rheat_fix`.
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
static void setup_params(struct params *run_params, const int rheat_fix)
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
    run_params->RheatFix = rheat_fix;
}

/* A CGM-regime central like the locked-out ones at z = 0: Mvir = 3 x 10^11
 * Msun/h with 10% of it as CGM at 0.1 Zsun-ish, and a 10^7 Msun/h black hole.
 * Rvir follows from Mvir and Vvir (G ~ 43 in code units). */
static void setup_cgm_central(struct GALAXY *gal, const double r_heat_over_rvir, const double bh_mass)
{
    memset(gal, 0, sizeof(struct GALAXY));
    gal->Regime = 0;
    gal->Mvir = 30.0;
    gal->Vvir = 120.0;
    gal->Rvir = 43.0 * gal->Mvir / (gal->Vvir * gal->Vvir);
    gal->CGMgas = 3.0;
    gal->MetalsCGMgas = 0.003;
    gal->BlackHoleMass = bh_mass;
    gal->r_heat = r_heat_over_rvir * gal->Rvir;
}

/* CGM cooling of one substep for a fresh setup_cgm_central() galaxy. */
static double cgm_cooling(const int rheat_fix, const double r_heat_over_rvir, const double bh_mass,
                          struct GALAXY *gal_out)
{
    struct params run_params;
    setup_params(&run_params, rheat_fix);
    setup_cgm_central(gal_out, r_heat_over_rvir, bh_mass);
    return cooling_recipe_cgm(0, DT, gal_out, &run_params);
}

/*
 * Asserts: under RheatFix 1 the CGM recipe leaves r_heat, the black hole and
 *          the radio-mode tallies alone, and its cooling does not depend on
 *          the stored r_heat.
 *
 * Setup: the same CGM-regime central with r_heat = 0, 0.5 Rvir and Rvir
 *        (locked out under v0.95), through cooling_recipe_cgm().
 *
 * Why: hot-only treats r_heat as the floor of a hot atmosphere, which a
 *        Regime 0 halo lacks.  The control (RheatFix 0) shows the lock-out:
 *        r_heat = Rvir gives zero cooling.
 */
void test_rheat_fix_hot_only_ignores_rheat(void)
{
    BEGIN_TEST("RheatFix 1: CGM recipe ignores r_heat, no radio mode");

    struct GALAXY gal;
    const double bh = 0.001;

    const double locked_v095 = cgm_cooling(0, 1.0, bh, &gal);
    ASSERT_CLOSE(0.0, locked_v095, 1e-30, "RheatFix 0: r_heat = Rvir gives zero CGM cooling (control)");

    const double free_cool = cgm_cooling(1, 0.0, bh, &gal);
    ASSERT_GREATER_THAN(free_cool, 0.0, "RheatFix 1: the CGM cools");
    ASSERT_CLOSE(0.0, gal.r_heat, RHEAT_UNCHANGED_TOL, "RheatFix 1: r_heat stays 0");
    ASSERT_CLOSE((float)bh, gal.BlackHoleMass, 1e-15, "RheatFix 1: no radio-mode black-hole growth (float field)");
    ASSERT_CLOSE(0.0, gal.RadioModeBHaccretionMass, 1e-30, "RheatFix 1: no radio-mode accretion tallied");
    ASSERT_CLOSE(0.0, gal.Heating, 1e-30, "RheatFix 1: no heating tallied");

    const double half_cool = cgm_cooling(1, 0.5, bh, &gal);
    ASSERT_CLOSE(free_cool, half_cool, COOLING_REL_TOL * free_cool, "RheatFix 1: r_heat = 0.5 Rvir does not suppress");
    ASSERT_CLOSE(0.5 * gal.Rvir, gal.r_heat, RHEAT_UNCHANGED_TOL, "RheatFix 1: r_heat = 0.5 Rvir unchanged");

    const double full_cool = cgm_cooling(1, 1.0, bh, &gal);
    ASSERT_CLOSE(free_cool, full_cool, COOLING_REL_TOL * free_cool, "RheatFix 1: r_heat = Rvir does not suppress");
}

/*
 * Asserts: the hot-regime Rvir cap on r_heat applies under RheatFix 0 and 1
 *          and is gone under RheatFix 2 and 4.
 *
 * Setup: a hot-regime central with r_heat = 1.5 Rvir and rcool = 2 Rvir
 *        (a halo cooling partially with its floor beyond Rvir), through
 *        do_AGN_heating() with no black hole, so the ratchet does not move.
 *
 * Why: the cap exists only because v0.95 shared r_heat with the Rvir-scale
 *        CGM recipe; classic SAGE has no such cap.
 */
void test_rheat_fix_hot_rvir_cap(void)
{
    BEGIN_TEST("RheatFix 2/4: no hot-regime Rvir cap on r_heat");

    const int modes[] = {0, 1, 2, 4};
    const int capped[] = {1, 1, 0, 0};
    for(size_t i = 0; i < sizeof(modes) / sizeof(modes[0]); i++) {
        struct params run_params;
        setup_params(&run_params, modes[i]);
        struct GALAXY gal;
        memset(&gal, 0, sizeof(gal));
        gal.Regime = 1;
        gal.Mvir = 1000.0;
        gal.Vvir = 450.0;
        gal.Rvir = 0.45;
        gal.HotGas = 100.0;
        gal.MetalsHotGas = 2.0;
        gal.r_heat = 1.5 * gal.Rvir;
        do_AGN_heating(1.0, 0, DT, 1.0, 2.0 * gal.Rvir, &gal, &run_params);
        const double expected = capped[i] ? gal.Rvir : 1.5 * gal.Rvir;
        char msg[128];
        snprintf(msg, sizeof(msg), "RheatFix %d: r_heat %s", modes[i], capped[i] ? "capped at Rvir" : "kept at 1.5 Rvir");
        ASSERT_CLOSE(expected, gal.r_heat, RHEAT_UNCHANGED_TOL, msg);
    }
}

/*
 * Asserts: under RheatFix 3 the radio mode cuts this substep's CGM cooling
 *          by the heating, grows the black hole, and leaves r_heat alone,
 *          also when r_heat = Rvir.
 *
 * Setup: the CGM-regime central with a 10^7 Msun/h black hole, compared
 *        with RheatFix 1 (no radio mode) on the same galaxy.
 *
 * Why: memoryless heating offsets only the current cooling; the reduction
 *        equals the heating mass, which is the accreted mass times
 *        (1.34e5 km/s / Vvir)^2 and is capped at the cooling.
 */
void test_rheat_fix_memoryless(void)
{
    BEGIN_TEST("RheatFix 3: memoryless Regime 0 radio mode");

    struct GALAXY gal;
    const double bh = 0.001;
    const double no_radio = cgm_cooling(1, 1.0, bh, &gal);
    const double with_radio = cgm_cooling(3, 1.0, bh, &gal);

    ASSERT_LESS_THAN(with_radio, no_radio, "RheatFix 3: heating reduces the cooling");
    ASSERT_GREATER_THAN(with_radio, 0.0, "RheatFix 3: r_heat = Rvir does not stop the cooling");
    ASSERT_CLOSE(gal.Rvir, gal.r_heat, RHEAT_UNCHANGED_TOL, "RheatFix 3: r_heat unchanged");
    ASSERT_GREATER_THAN(gal.RadioModeBHaccretionMass, 0.0, "RheatFix 3: radio-mode accretion tallied");

    const double coeff = (1.34e5 / gal.Vvir) * (1.34e5 / gal.Vvir);
    ASSERT_CLOSE(coeff * gal.RadioModeBHaccretionMass, no_radio - with_radio,
                 1e-6 * no_radio, "RheatFix 3: cooling reduced by exactly the heating mass");
}

/*
 * Asserts: under RheatFix 4 a CGM-regime central with r_heat = Rvir still
 *          cools when its CGM cooling radius lies beyond Rvir, and the
 *          ratchet may then take r_heat beyond Rvir.
 *
 * Setup: the gas-rich CGM-regime central (CGM = 10% of Mvir), first with
 *        no black hole and r_heat = Rvir, then with a black hole and r_heat
 *        = 0.
 *
 * Why: the reference radius is the profile's own cooling radius (median
 *        ~2-3 Rvir for such haloes), so a floor at Rvir only suppresses part
 *        of the cooling, as in SAGE16's rapid-cooling regime.
 */
void test_rheat_fix_cgm_rcool(void)
{
    BEGIN_TEST("RheatFix 4: moving CGM cooling radius");

    struct GALAXY gal;
    const double unsuppressed = cgm_cooling(4, 0.0, 0.0, &gal);
    const double floor_at_rvir = cgm_cooling(4, 1.0, 0.0, &gal);
    ASSERT_GREATER_THAN(floor_at_rvir, 0.0, "RheatFix 4: r_heat = Rvir leaves some cooling");
    ASSERT_LESS_THAN(floor_at_rvir, unsuppressed, "RheatFix 4: r_heat = Rvir still suppresses");

    /* rcool / Rvir follows from the linear suppression 1 - Rvir / rcool. */
    const double rcool_over_rvir = 1.0 / (1.0 - floor_at_rvir / unsuppressed);
    printf("  info: CGM rcool = %.2f Rvir for this halo\n", rcool_over_rvir);

    /* A large black hole out-heats the cooling: the ratchet saturates at
     * rcool, beyond Rvir, with no cap. */
    cgm_cooling(4, 0.0, 1.0, &gal);
    ASSERT_GREATER_THAN(gal.r_heat, gal.Rvir, "RheatFix 4: r_heat may exceed Rvir");
}

int main(void)
{
    BEGIN_TEST_SUITE("RheatFix: r_heat in the CGM regime");

    read_cooling_functions();

    test_rheat_fix_hot_only_ignores_rheat();
    test_rheat_fix_hot_rvir_cap();
    test_rheat_fix_memoryless();
    test_rheat_fix_cgm_rcool();

    END_TEST_SUITE();
    PRINT_TEST_SUMMARY();

    return TEST_EXIT_CODE();
}

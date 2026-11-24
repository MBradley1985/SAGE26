/*
 * test_rheat_fix.c -- unit tests for the RheatFix options (r_heat in the CGM regime).
 *
 * Validates: RheatFix 1 (the CGM recipe neither reads nor writes r_heat and
 * has no radio mode), 2 (no hot-regime Rvir cap), 3 (one-way CGM -> hot
 * transition), 4 (fixed per-galaxy regime quantile), 5 (memoryless Regime 0
 * radio mode), 6 (moving CGM cooling radius) and 7 (profile-matched
 * suppression), each against RheatFix 0 (v0.95).
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
#include "../src/model_regimes.h"

/* r_heat is a float field: an untouched value comes back exactly. */
static const double RHEAT_UNCHANGED_TOL = 1e-6;

/* Relative tolerance on cooling masses computed by the same double-precision
 * expression in two calls. */
static const double COOLING_REL_TOL = 1e-12;

/* One substep, in code time units (~1 Myr). */
static const double DT = 0.001;

/* Number of independent halo draws in the regime tests. */
static const int N_DRAWS = 1000;

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
 *          and is gone under RheatFix 2, 6 and 7.
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
    BEGIN_TEST("RheatFix 2/6/7: no hot-regime Rvir cap on r_heat");

    const int modes[] = {0, 1, 2, 6, 7};
    const int capped[] = {1, 1, 0, 0, 0};
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

/* Fraction of N_DRAWS distinct haloes drawn hot at log10(Mvir/Mshock) =
 * dex_from_shock, each starting in start_regime, at the given snapshot. */
static double hot_fraction(const int rheat_fix, const double dex_from_shock, const int start_regime, const int snap)
{
    struct params run_params;
    setup_params(&run_params, rheat_fix);
    int hot = 0;
    for(int j = 0; j < N_DRAWS; j++) {
        struct GALAXY gal;
        memset(&gal, 0, sizeof(gal));
        gal.Mvir = run_params.MShockMsun * pow(10.0, dex_from_shock) / (1e10 / run_params.Hubble_h);
        gal.MostBoundID = j;
        gal.SnapNum = snap;
        gal.Regime = start_regime;
        determine_and_store_regime(1, &gal, &run_params);
        hot += (gal.Regime == 1);
    }
    return (double)hot / N_DRAWS;
}

/*
 * Asserts: under RheatFix 3 a hot central 0.2 dex below Mshock stays hot,
 *          one 0.4 dex below returns to the draw, and a CGM-regime central
 *          is still drawn from the sigmoid.
 *
 * Setup: 1000 distinct haloes per case through determine_and_store_regime().
 *
 * Why: the exit threshold is 0.3 dex below Mshock.  At -0.2 dex the sigmoid
 *        (width 0.1 dex) gives the hot regime 12%, so RheatFix 1 sends most
 *        hot haloes back to the CGM regime and RheatFix 3 none.
 */
void test_rheat_fix_one_way_regime(void)
{
    BEGIN_TEST("RheatFix 3: one-way CGM -> hot transition");

    ASSERT_IN_RANGE(hot_fraction(1, -0.2, 1, 50), 0.07, 0.18, "RheatFix 1: hot halo at -0.2 dex redrawn (~12% hot)");
    ASSERT_CLOSE(1.0, hot_fraction(3, -0.2, 1, 50), 1e-12, "RheatFix 3: hot halo at -0.2 dex stays hot");
    ASSERT_IN_RANGE(hot_fraction(3, -0.4, 1, 50), 0.0, 0.05, "RheatFix 3: hot halo at -0.4 dex redrawn (~2% hot)");
    ASSERT_IN_RANGE(hot_fraction(3, -0.2, 0, 50), 0.07, 0.18, "RheatFix 3: CGM halo at -0.2 dex still drawn (~12% hot)");
}

/*
 * Asserts: under RheatFix 4 a galaxy at fixed mass keeps one regime over
 *          many snapshots, while under RheatFix 1 it is redrawn.
 *
 * Setup: 1000 haloes at Mshock (50% hot), each classified at snapshots
 *        10-29 with its galaxy struct carried over, as the tree walk does.
 *
 * Why: the quantile is drawn once and stored on the galaxy, so the outcome
 *        at fixed mass cannot change; a fresh per-snapshot draw at 50% flips
 *        almost every halo at least once in 20 snapshots.
 */
void test_rheat_fix_fixed_quantile(void)
{
    BEGIN_TEST("RheatFix 4: fixed per-galaxy regime quantile");

    for(int mode = 1; mode <= 4; mode += 3) {
        struct params run_params;
        setup_params(&run_params, mode);
        int flipped = 0, hot = 0;
        for(int j = 0; j < N_DRAWS; j++) {
            struct GALAXY gal;
            memset(&gal, 0, sizeof(gal));
            gal.Mvir = run_params.MShockMsun / (1e10 / run_params.Hubble_h);
            gal.MostBoundID = j;
            gal.Regime = -1;
            int first = -1, changed = 0;
            for(int snap = 10; snap < 30; snap++) {
                gal.SnapNum = snap;
                determine_and_store_regime(1, &gal, &run_params);
                if(first < 0) first = gal.Regime;
                if(gal.Regime != first) changed = 1;
            }
            flipped += changed;
            hot += (first == 1);
        }
        if(mode == 4) {
            ASSERT_EQUAL_INT(0, flipped, "RheatFix 4: no galaxy changes regime at fixed mass");
            ASSERT_IN_RANGE((double)hot / N_DRAWS, 0.45, 0.55, "RheatFix 4: half the galaxies hot at Mshock");
        } else {
            ASSERT_GREATER_THAN((double)flipped / N_DRAWS, 0.99, "RheatFix 1: almost every galaxy flips at Mshock");
        }
    }
}

/*
 * Asserts: under RheatFix 5 the radio mode cuts this substep's CGM cooling
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
    BEGIN_TEST("RheatFix 5: memoryless Regime 0 radio mode");

    struct GALAXY gal;
    const double bh = 0.001;
    const double no_radio = cgm_cooling(1, 1.0, bh, &gal);
    const double with_radio = cgm_cooling(5, 1.0, bh, &gal);

    ASSERT_LESS_THAN(with_radio, no_radio, "RheatFix 5: heating reduces the cooling");
    ASSERT_GREATER_THAN(with_radio, 0.0, "RheatFix 5: r_heat = Rvir does not stop the cooling");
    ASSERT_CLOSE(gal.Rvir, gal.r_heat, RHEAT_UNCHANGED_TOL, "RheatFix 5: r_heat unchanged");
    ASSERT_GREATER_THAN(gal.RadioModeBHaccretionMass, 0.0, "RheatFix 5: radio-mode accretion tallied");

    const double coeff = (1.34e5 / gal.Vvir) * (1.34e5 / gal.Vvir);
    ASSERT_CLOSE(coeff * gal.RadioModeBHaccretionMass, no_radio - with_radio,
                 1e-6 * no_radio, "RheatFix 5: cooling reduced by exactly the heating mass");
}

/*
 * Asserts: under RheatFix 6 a CGM-regime central with r_heat = Rvir still
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
    BEGIN_TEST("RheatFix 6: moving CGM cooling radius");

    struct GALAXY gal;
    const double unsuppressed = cgm_cooling(6, 0.0, 0.0, &gal);
    const double floor_at_rvir = cgm_cooling(6, 1.0, 0.0, &gal);
    ASSERT_GREATER_THAN(floor_at_rvir, 0.0, "RheatFix 6: r_heat = Rvir leaves some cooling");
    ASSERT_LESS_THAN(floor_at_rvir, unsuppressed, "RheatFix 6: r_heat = Rvir still suppresses");

    /* rcool / Rvir follows from the linear suppression 1 - Rvir / rcool. */
    const double rcool_over_rvir = 1.0 / (1.0 - floor_at_rvir / unsuppressed);
    printf("  info: CGM rcool = %.2f Rvir for this halo\n", rcool_over_rvir);

    /* A large black hole out-heats the cooling: the ratchet saturates at
     * rcool, beyond Rvir, with no cap. */
    cgm_cooling(6, 0.0, 1.0, &gal);
    ASSERT_GREATER_THAN(gal.r_heat, gal.Rvir, "RheatFix 6: r_heat may exceed Rvir");
}

/*
 * Asserts: under RheatFix 7 the suppression is 1 - (r_heat/rcool)^1.6 for
 *          the same rcool that RheatFix 6 suppresses as 1 - r_heat/rcool.
 *
 * Setup: no black hole (no heating, so r_heat does not move), r_heat = 0.5
 *        Rvir, under RheatFix 6 and 7.  u = r_heat/rcool is recovered from
 *        RheatFix 6.
 *
 * Why: M(<r) ~ r^1.6 in the alpha = 1.4 profile, so the heated mass fraction
 *        inside r_heat is u^1.6.
 */
void test_rheat_fix_profile_matched(void)
{
    BEGIN_TEST("RheatFix 7: profile-matched suppression");

    struct GALAXY gal;
    const double unsuppressed = cgm_cooling(7, 0.0, 0.0, &gal);
    const double linear = cgm_cooling(6, 0.5, 0.0, &gal);
    const double matched = cgm_cooling(7, 0.5, 0.0, &gal);
    const double u = 1.0 - linear / unsuppressed;
    ASSERT_CLOSE(1.0 - pow(u, 1.6), matched / unsuppressed, 1e-10,
                 "RheatFix 7: suppression 1 - (r_heat/rcool)^1.6");
}

int main(void)
{
    BEGIN_TEST_SUITE("RheatFix: r_heat in the CGM regime");

    read_cooling_functions();

    test_rheat_fix_hot_only_ignores_rheat();
    test_rheat_fix_hot_rvir_cap();
    test_rheat_fix_one_way_regime();
    test_rheat_fix_fixed_quantile();
    test_rheat_fix_memoryless();
    test_rheat_fix_cgm_rcool();
    test_rheat_fix_profile_matched();

    END_TEST_SUITE();
    PRINT_TEST_SUMMARY();

    return TEST_EXIT_CODE();
}

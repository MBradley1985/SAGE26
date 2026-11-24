/*
 * test_rheat.c -- unit tests for the AGN heating radius r_heat across the regimes.
 *
 * Validates: the CGM recipe (Regime 0) has no radio mode and neither reads
 * nor writes r_heat, so a halo whose r_heat reached Rvir still cools; the
 * hot-halo radio mode does not cap r_heat at Rvir; with StreamRheatFix 1 the
 * hot-regime cold streams bypass the AGN step, and with 2 they are limited to
 * f_stream times the halo's infall rate.
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

/* A shorter substep for the stream halo, whose dynamical time is ~0.4 Myr,
 * so the streams stay well below the hot reservoir. */
static const double STREAM_DT = 1.0e-5;

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
    return cooling_recipe_cgm(0, DT, gal, &run_params);
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
    ASSERT_CLOSE(0.0, gal.RadioModeBHaccretionMass, 1e-30, "no radio-mode accretion tallied");
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

    const double cooling = do_AGN_heating(1.0, 0, DT, 1.0, 2.0 * gal.Rvir, &gal, &run_params);
    ASSERT_CLOSE(1.5 * gal.Rvir, gal.r_heat, RHEAT_UNCHANGED_TOL, "r_heat kept at 1.5 Rvir");
    ASSERT_CLOSE(0.25, cooling, RHEAT_UNCHANGED_TOL, "cooling suppressed by 1 - r_heat / rcool = 0.25 (float r_heat)");
}

/* A hot-regime halo below Mshock at z = 3 (Mvir = 3 x 10^11 Msun/h), where
 * the Dekel & Birnboim (2006) criterion lets most of the inflow come in as
 * cold streams, through cooling_recipe_hot(). */
static double stream_halo_cooling_infall(const int stream_fix, const double r_heat_over_rvir, const double bh_mass,
                                         const double infall_rate, struct GALAXY *gal)
{
    struct params run_params;
    setup_params(&run_params);
    run_params.StreamRheatFix = stream_fix;
    run_params.ZZ[0] = 3.0;
    memset(gal, 0, sizeof(struct GALAXY));
    gal->Regime = 1;
    gal->Mvir = 30.0;
    gal->Vvir = 150.0;
    gal->Rvir = 43.0 * gal->Mvir / (gal->Vvir * gal->Vvir);
    gal->HotGas = 3.0;
    gal->MetalsHotGas = 0.003;
    gal->BlackHoleMass = bh_mass;
    gal->r_heat = r_heat_over_rvir * gal->Rvir;
    gal->InfallRate = infall_rate;
    return cooling_recipe_hot(0, STREAM_DT, gal, &run_params);
}

static double stream_halo_cooling(const int stream_fix, const double r_heat_over_rvir, const double bh_mass,
                                  struct GALAXY *gal)
{
    return stream_halo_cooling_infall(stream_fix, r_heat_over_rvir, bh_mass, 0.0, gal);
}

/*
 * Asserts: with StreamRheatFix 1 the cold streams of a hot-regime halo are
 *          delivered when r_heat suppresses the cooling flow completely, and
 *          neither move r_heat nor grow the black hole; with no AGN both
 *          modes give the same cooling.
 *
 * Setup: the stream_halo_cooling() halo with r_heat = 10 Rvir (beyond rcool)
 *        and with r_heat = 0 and no black hole, under StreamRheatFix 0 and 1.
 *
 * Why: filaments that penetrate the hot atmosphere do not feel its entropy
 *        floor, so only the cooling flow goes through the AGN step.
 */
void test_streams_bypass_agn(void)
{
    BEGIN_TEST("StreamRheatFix 1: cold streams bypass the AGN step");

    struct GALAXY gal;

    const double free0 = stream_halo_cooling(0, 0.0, 0.0, &gal);
    const double free_stream = gal.mdot_stream * STREAM_DT;
    ASSERT_GREATER_THAN(free_stream, 0.5 * free0, "the halo cools mostly through streams");
    const double free1 = stream_halo_cooling(1, 0.0, 0.0, &gal);
    ASSERT_CLOSE(free0, free1, COOLING_REL_TOL * free0, "no AGN: both modes cool the same");

    const double off = stream_halo_cooling(0, 10.0, 0.001, &gal);
    ASSERT_CLOSE(0.0, off, 1e-30, "StreamRheatFix 0: r_heat beyond rcool stops all cooling");

    const double on = stream_halo_cooling(1, 10.0, 0.001, &gal);
    ASSERT_CLOSE(free_stream, on, RHEAT_UNCHANGED_TOL * free_stream, "StreamRheatFix 1: the streams still cool (float mdot_stream)");
    ASSERT_CLOSE(on, gal.mdot_stream * STREAM_DT, RHEAT_UNCHANGED_TOL * on, "StreamRheatFix 1: mdot_stream is the delivered rate");
    ASSERT_CLOSE(10.0 * gal.Rvir, gal.r_heat, RHEAT_UNCHANGED_TOL * 10.0 * gal.Rvir, "StreamRheatFix 1: r_heat unchanged");
    ASSERT_CLOSE((float)0.001, gal.BlackHoleMass, 1e-15, "StreamRheatFix 1: no radio-mode growth from streams (float field)");
}

/*
 * Asserts: with StreamRheatFix 2 the streams carry at most f_stream times the
 *          halo's infall rate, none without infall, and still bypass r_heat.
 *
 * Setup: the stream_halo_cooling() halo with r_heat = 10 Rvir, so the
 *        cooling flow is fully suppressed and only streams cool, with an
 *        infall rate of 1% of HotGas / t_dyn and with none.
 *
 * Why: Dekel et al. (2009): the streams are the cold part of the
 *        cosmological inflow, which caps them at the infall rate.
 */
void test_streams_limited_to_infall(void)
{
    BEGIN_TEST("StreamRheatFix 2: cold streams limited to the infall rate");

    struct GALAXY gal;
    stream_halo_cooling(1, 10.0, 0.001, &gal);
    const double uncapped = gal.mdot_stream;
    const double infall = 0.01 * uncapped;

    const double capped = stream_halo_cooling_infall(2, 10.0, 0.001, infall, &gal);
    ASSERT_LESS_THAN(capped, infall * STREAM_DT * (1.0 + RHEAT_UNCHANGED_TOL), "streams <= infall rate x dt");
    ASSERT_GREATER_THAN(capped, 0.5 * infall * STREAM_DT, "streams ~ f_stream x infall rate x dt (f_stream ~ 1 here)");
    ASSERT_CLOSE(10.0 * gal.Rvir, gal.r_heat, RHEAT_UNCHANGED_TOL * 10.0 * gal.Rvir, "r_heat unchanged");

    const double none = stream_halo_cooling_infall(2, 10.0, 0.001, 0.0, &gal);
    ASSERT_CLOSE(0.0, none, 1e-30, "no infall: no streams, and the suppressed flow gives no cooling");
}

int main(void)
{
    BEGIN_TEST_SUITE("r_heat across the regimes");

    read_cooling_functions();

    test_cgm_recipe_ignores_rheat();
    test_hot_rheat_no_rvir_cap();
    test_streams_bypass_agn();
    test_streams_limited_to_infall();

    END_TEST_SUITE();
    PRINT_TEST_SUMMARY();

    return TEST_EXIT_CODE();
}

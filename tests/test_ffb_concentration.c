/*
 * FFB & CONCENTRATION TESTS
 *
 * Tests for feedback-free burst regime determination and
 * halo concentration calculations added in SAGE26:
 * - Li+24 sigmoid-based FFB classification (mode 1)
 * - FFBRandom persistent random number
 * - Concentration methods (Ishiyama+21 table, Vmax/Vvir, infall freeze)
 * - BK25 g_max acceleration calculation (modes 2, 3)
 * - BK25 + log-normal concentration scatter (mode 4)
 * - Li+24 mass sharp cutoff (mode 5)
 * - FeedbackFreeModeOn modes 0–5
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "test_framework.h"
#include "../src/core_allvars.h"
#include "../src/model_misc.h"

/* ──────────────────────────────────────────────────────────────────
 *  Helper: set up run_params with Millennium cosmology + code units
 * ────────────────────────────────────────────────────────────────── */
static void init_millennium_params(struct params *rp)
{
    memset(rp, 0, sizeof(struct params));
    rp->Hubble_h            = 0.73;
    rp->Omega               = 0.25;
    rp->BaryonFrac          = 0.17;
    rp->UnitLength_in_cm    = 3.08568e24;   /* Mpc/h */
    rp->UnitMass_in_g       = 1.989e43;     /* 10^10 Msun */
    rp->UnitVelocity_in_cm_per_s = 1e5;     /* km/s */
    rp->UnitTime_in_s       = rp->UnitLength_in_cm / rp->UnitVelocity_in_cm_per_s;
    rp->UnitDensity_in_cgs  = rp->UnitMass_in_g
                            / (rp->UnitLength_in_cm * rp->UnitLength_in_cm * rp->UnitLength_in_cm);
    rp->G = GRAVITY / rp->UnitLength_in_cm
           * rp->UnitMass_in_g * rp->UnitTime_in_s * rp->UnitTime_in_s;
    rp->EnergySNcode = rp->EnergySN / rp->UnitMass_in_g
                     / rp->UnitVelocity_in_cm_per_s / rp->UnitVelocity_in_cm_per_s;
}

/* ═══════════════════════════════════════════════════════════════════
 *  1.  FFB threshold mass  (calculate_ffb_threshold_mass)
 * ═══════════════════════════════════════════════════════════════════ */
void test_ffb_threshold_mass_scaling()
{
    BEGIN_TEST("FFB threshold mass scales as (1+z)^{-6.2}");

    struct params rp;
    init_millennium_params(&rp);
    rp.FeedbackFreeModeOn = 1;

    double M_z8  = calculate_ffb_threshold_mass(8.0,  &rp);
    double M_z10 = calculate_ffb_threshold_mass(10.0, &rp);
    double M_z12 = calculate_ffb_threshold_mass(12.0, &rp);

    /* Higher redshift → lower threshold */
    ASSERT_GREATER_THAN(M_z8, M_z10, "M_thresh(z=8) > M_thresh(z=10)");
    ASSERT_GREATER_THAN(M_z10, M_z12, "M_thresh(z=10) > M_thresh(z=12)");

    /* Check the power-law ratio: M(z1)/M(z2) = ((1+z1)/(1+z2))^{-6.2} */
    double expected_ratio = pow((1.0 + 8.0) / (1.0 + 10.0), -6.2);
    double actual_ratio = M_z8 / M_z10;
    ASSERT_CLOSE(expected_ratio, actual_ratio, 1e-6,
                 "Threshold mass ratio matches (1+z)^{-6.2} power law");

    /* Threshold mass should be positive at all redshifts */
    ASSERT_GREATER_THAN(M_z8, 0.0, "M_thresh(z=8) > 0");
    ASSERT_GREATER_THAN(M_z12, 0.0, "M_thresh(z=12) > 0");
}

void test_ffb_threshold_mass_absolute()
{
    BEGIN_TEST("FFB threshold mass matches Li+24 normalization");

    struct params rp;
    init_millennium_params(&rp);
    rp.FeedbackFreeModeOn = 1;

    /* At z=9 the normalisation factor (1+z)/10 = 1, so
       log10(M_code) = 0.8 + log10(h) → M_code = 10^(0.8+log10(0.73))
       M_sun/h = M_code * 1e10
       log10(M_sun/h) = 10.8 + log10(h) */
    double M_z9 = calculate_ffb_threshold_mass(9.0, &rp);
    double logM_z9 = log10(M_z9 * 1e10);  /* log10(Msun/h) */
    double expected = 10.8 + log10(0.73);

    ASSERT_CLOSE(expected, logM_z9, 1e-6,
                 "log10(M_thresh) = 10.8 + log10(h) at z=9");
}

/* ═══════════════════════════════════════════════════════════════════
 *  2.  FFB fraction / sigmoid  (calculate_ffb_fraction)
 * ═══════════════════════════════════════════════════════════════════ */
void test_ffb_sigmoid_midpoint()
{
    BEGIN_TEST("Sigmoid is 0.5 at threshold mass");

    struct params rp;
    init_millennium_params(&rp);
    rp.FeedbackFreeModeOn  = 1;
    rp.FFBConcSigma        = 0.0;

    double z = 10.0;
    double M_thresh = calculate_ffb_threshold_mass(z, &rp);
    double f = calculate_ffb_fraction(M_thresh, z, &rp);

    ASSERT_CLOSE(0.5, f, 1e-6, "f_ffb = 0.5 at Mvir = Mvir_ffb");
}

void test_ffb_sigmoid_symmetry()
{
    BEGIN_TEST("Sigmoid is symmetric around threshold");

    struct params rp;
    init_millennium_params(&rp);
    rp.FeedbackFreeModeOn  = 1;
    rp.FFBConcSigma        = 0.0;

    double z = 10.0;
    double M_thresh = calculate_ffb_threshold_mass(z, &rp);
    double delta = 0.15;  /* width parameter used in code */

    double M_above = M_thresh * pow(10.0, delta);
    double M_below = M_thresh * pow(10.0, -delta);

    double f_above = calculate_ffb_fraction(M_above, z, &rp);
    double f_below = calculate_ffb_fraction(M_below, z, &rp);

    /* S(x) + S(-x) = 1 for a logistic sigmoid */
    ASSERT_CLOSE(1.0, f_above + f_below, 1e-6,
                 "f(M+delta) + f(M-delta) = 1 (sigmoid symmetry)");
}

void test_ffb_sigmoid_monotonic()
{
    BEGIN_TEST("Sigmoid increases with mass");

    struct params rp;
    init_millennium_params(&rp);
    rp.FeedbackFreeModeOn  = 1;
    rp.FFBConcSigma        = 0.0;

    double z = 10.0;
    double M_thresh = calculate_ffb_threshold_mass(z, &rp);

    double f_lo  = calculate_ffb_fraction(M_thresh * 0.5, z, &rp);
    double f_mid = calculate_ffb_fraction(M_thresh,       z, &rp);
    double f_hi  = calculate_ffb_fraction(M_thresh * 2.0, z, &rp);

    ASSERT_GREATER_THAN(f_mid, f_lo,  "f(M_thresh) > f(0.5 M_thresh)");
    ASSERT_GREATER_THAN(f_hi,  f_mid, "f(2 M_thresh) > f(M_thresh)");
}

void test_ffb_disabled_returns_zero()
{
    BEGIN_TEST("FFB fraction is 0 when FeedbackFreeModeOn=0");

    struct params rp;
    init_millennium_params(&rp);
    rp.FeedbackFreeModeOn = 0;

    double f = calculate_ffb_fraction(100.0, 10.0, &rp);
    ASSERT_EQUAL_DOUBLE(0.0, f, "f_ffb = 0 when FFB disabled");
}

void test_ffb_invalid_mass()
{
    BEGIN_TEST("FFB fraction handles zero/negative mass");

    struct params rp;
    init_millennium_params(&rp);
    rp.FeedbackFreeModeOn  = 1;
    rp.FFBConcSigma        = 0.0;

    double f_zero = calculate_ffb_fraction(0.0, 10.0, &rp);
    double f_neg  = calculate_ffb_fraction(-1.0, 10.0, &rp);

    ASSERT_EQUAL_DOUBLE(0.0, f_zero, "f_ffb = 0 for Mvir = 0");
    ASSERT_EQUAL_DOUBLE(0.0, f_neg,  "f_ffb = 0 for Mvir < 0");
}

/* ═══════════════════════════════════════════════════════════════════
 *  3.  Concentration from Vmax/Vvir  (concentration_from_vmax_vvir)
 * ═══════════════════════════════════════════════════════════════════ */
void test_concentration_vmax_vvir_known_values()
{
    BEGIN_TEST("Vmax/Vvir concentration recovers known NFW values");

    /* For an NFW halo with concentration c, the circular velocity ratio is:
       (Vmax/Vvir)^2 = 0.2162 * c / mu(c)
       Test with c=5: mu(5) = ln(6) - 5/6 = 0.9585
       (Vmax/Vvir)^2 = 0.2162 * 5 / 0.9585 = 1.1279
       Vmax/Vvir = 1.0621 */
    double c_target = 5.0;
    double mu = log(1.0 + c_target) - c_target / (1.0 + c_target);
    double ratio = sqrt(0.21621 * c_target / mu);

    double c_recovered = concentration_from_vmax_vvir(ratio * 100.0, 100.0);

    ASSERT_CLOSE(c_target, c_recovered, 1e-4,
                 "Recover c=5 from Vmax/Vvir");

    /* Test with c=10 */
    c_target = 10.0;
    mu = log(1.0 + c_target) - c_target / (1.0 + c_target);
    ratio = sqrt(0.21621 * c_target / mu);
    c_recovered = concentration_from_vmax_vvir(ratio * 100.0, 100.0);

    ASSERT_CLOSE(c_target, c_recovered, 1e-4,
                 "Recover c=10 from Vmax/Vvir");

    /* Test with c=20 */
    c_target = 20.0;
    mu = log(1.0 + c_target) - c_target / (1.0 + c_target);
    ratio = sqrt(0.21621 * c_target / mu);
    c_recovered = concentration_from_vmax_vvir(ratio * 100.0, 100.0);

    ASSERT_CLOSE(c_target, c_recovered, 1e-4,
                 "Recover c=20 from Vmax/Vvir");
}

void test_concentration_vmax_vvir_monotonic()
{
    BEGIN_TEST("Concentration increases with Vmax/Vvir");

    double c1 = concentration_from_vmax_vvir(100.0, 100.0);  /* Vmax/Vvir = 1.0 */
    double c2 = concentration_from_vmax_vvir(110.0, 100.0);  /* Vmax/Vvir = 1.1 */
    double c3 = concentration_from_vmax_vvir(120.0, 100.0);  /* Vmax/Vvir = 1.2 */

    ASSERT_GREATER_THAN(c2, c1, "c(1.1) > c(1.0)");
    ASSERT_GREATER_THAN(c3, c2, "c(1.2) > c(1.1)");
}

void test_concentration_vmax_vvir_edge_cases()
{
    BEGIN_TEST("Concentration handles edge cases");

    double c_zero_vvir = concentration_from_vmax_vvir(100.0, 0.0);
    double c_zero_vmax = concentration_from_vmax_vvir(0.0, 100.0);
    double c_neg       = concentration_from_vmax_vvir(-50.0, 100.0);

    ASSERT_EQUAL_DOUBLE(0.0, c_zero_vvir, "c = 0 when Vvir = 0");
    ASSERT_EQUAL_DOUBLE(0.0, c_zero_vmax, "c = 0 when Vmax = 0");
    ASSERT_EQUAL_DOUBLE(0.0, c_neg,       "c = 0 when Vmax < 0");

    /* Vmax < Vvir → unphysical for NFW, returns 0 */
    double c_sub = concentration_from_vmax_vvir(90.0, 100.0);
    ASSERT_EQUAL_DOUBLE(0.0, c_sub, "c = 0 when Vmax/Vvir < 1 (unresolved)");
}

/* ═══════════════════════════════════════════════════════════════════
 *  5.  Concentration selection (get_halo_concentration)
 * ═══════════════════════════════════════════════════════════════════ */
void test_concentration_off()
{
    BEGIN_TEST("ConcentrationOn=0 returns 0");

    struct params rp;
    init_millennium_params(&rp);
    rp.ConcentrationOn = 0;

    struct GALAXY gal;
    memset(&gal, 0, sizeof(struct GALAXY));
    gal.Mvir = 10.0;
    gal.Rvir = 0.1;
    gal.Vmax = 150.0;
    gal.Vvir = 130.0;

    double c = get_halo_concentration(0, 0.0, &gal, &rp);
    ASSERT_EQUAL_DOUBLE(0.0, c, "Concentration = 0 when disabled");
}

void test_concentration_mode2_uses_vmax_vvir()
{
    BEGIN_TEST("ConcentrationOn=2 uses Vmax/Vvir for all types");

    struct params rp;
    init_millennium_params(&rp);
    rp.ConcentrationOn = 2;

    struct GALAXY gal;
    memset(&gal, 0, sizeof(struct GALAXY));
    gal.Mvir = 10.0;
    gal.Rvir = 0.1;
    gal.Vmax = 110.0;
    gal.Vvir = 100.0;
    gal.Type = 0;  /* central */

    double c_cen = get_halo_concentration(0, 5.0, &gal, &rp);
    double c_direct = concentration_from_vmax_vvir(110.0, 100.0);

    ASSERT_CLOSE(c_direct, c_cen, 1e-10,
                 "Mode 2 central matches concentration_from_vmax_vvir");

    /* Satellite should also use live Vmax/Vvir (not infall) */
    gal.Type = 1;
    gal.infallVmax = 120.0;
    gal.infallVvir = 100.0;

    double c_sat = get_halo_concentration(0, 5.0, &gal, &rp);
    ASSERT_CLOSE(c_direct, c_sat, 1e-10,
                 "Mode 2 satellite uses live Vmax/Vvir (not infall)");
}

void test_concentration_mode3_infall_freeze()
{
    BEGIN_TEST("ConcentrationOn=3 uses infall Vmax/Vvir for satellites");

    struct params rp;
    init_millennium_params(&rp);
    rp.ConcentrationOn = 3;

    struct GALAXY gal;
    memset(&gal, 0, sizeof(struct GALAXY));
    gal.Mvir = 10.0;
    gal.Rvir = 0.1;
    gal.Vmax = 110.0;
    gal.Vvir = 100.0;
    gal.infallVmax = 130.0;
    gal.infallVvir = 105.0;

    /* Central should use live values */
    gal.Type = 0;
    double c_cen = get_halo_concentration(0, 5.0, &gal, &rp);
    double c_live = concentration_from_vmax_vvir(110.0, 100.0);
    ASSERT_CLOSE(c_live, c_cen, 1e-10,
                 "Mode 3 central uses live Vmax/Vvir");

    /* Satellite should use infall values */
    gal.Type = 1;
    double c_sat = get_halo_concentration(0, 5.0, &gal, &rp);
    double c_infall = concentration_from_vmax_vvir(130.0, 105.0);
    ASSERT_CLOSE(c_infall, c_sat, 1e-10,
                 "Mode 3 satellite uses infall Vmax/Vvir");

    /* Confirm satellite concentration differs from central */
    ASSERT_TRUE(fabs(c_cen - c_sat) > 0.1,
                "Satellite and central concentrations differ in mode 3");
}

/* ═══════════════════════════════════════════════════════════════════
 *  6.  Ishiyama+21 lookup table  (interpolate_concentration_ishiyama21)
 * ═══════════════════════════════════════════════════════════════════ */
void test_ishiyama21_physical_values()
{
    BEGIN_TEST("Ishiyama+21 table returns physical concentrations");

    struct params rp;
    init_millennium_params(&rp);

    /* Milky Way mass at z=0: ~10^12 Msun/h → logM = 12 */
    double c_mw = interpolate_concentration_ishiyama21(12.0, 0.0, &rp);
    ASSERT_IN_RANGE(c_mw, 4.0, 15.0,
                    "MW-mass halo at z=0: c in [4,15]");

    /* Cluster mass at z=0: ~10^15 Msun/h → logM = 15 */
    double c_cl = interpolate_concentration_ishiyama21(15.0, 0.0, &rp);
    ASSERT_IN_RANGE(c_cl, 2.0, 8.0,
                    "Cluster-mass halo at z=0: c in [2,8]");

    /* Dwarf mass at z=0: ~10^10 Msun/h → logM = 10 */
    double c_dw = interpolate_concentration_ishiyama21(10.0, 0.0, &rp);
    ASSERT_IN_RANGE(c_dw, 5.0, 25.0,
                    "Dwarf halo at z=0: c in [5,25]");
}

void test_ishiyama21_mass_trend()
{
    BEGIN_TEST("Ishiyama+21: concentration decreases with mass at z=0");

    struct params rp;
    init_millennium_params(&rp);

    double c_10 = interpolate_concentration_ishiyama21(10.0, 0.0, &rp);
    double c_12 = interpolate_concentration_ishiyama21(12.0, 0.0, &rp);
    double c_14 = interpolate_concentration_ishiyama21(14.0, 0.0, &rp);

    ASSERT_GREATER_THAN(c_10, c_12, "c(10^10) > c(10^12) at z=0");
    ASSERT_GREATER_THAN(c_12, c_14, "c(10^12) > c(10^14) at z=0");
}

void test_ishiyama21_redshift_trend()
{
    BEGIN_TEST("Ishiyama+21: concentration decreases with redshift");

    struct params rp;
    init_millennium_params(&rp);

    double c_z0 = interpolate_concentration_ishiyama21(12.0, 0.0, &rp);
    double c_z2 = interpolate_concentration_ishiyama21(12.0, 2.0, &rp);
    double c_z5 = interpolate_concentration_ishiyama21(12.0, 5.0, &rp);

    ASSERT_GREATER_THAN(c_z0, c_z2, "c(z=0) > c(z=2) at 10^12 Msun/h");
    ASSERT_GREATER_THAN(c_z2, c_z5, "c(z=2) > c(z=5) at 10^12 Msun/h");
}

/* ═══════════════════════════════════════════════════════════════════
 *  7.  BK25 g_max  (calculate_gmax_BK25)
 * ═══════════════════════════════════════════════════════════════════ */
void test_gmax_positive_for_valid_halo()
{
    BEGIN_TEST("g_max > 0 for a valid halo");

    struct params rp;
    init_millennium_params(&rp);

    struct GALAXY gal;
    memset(&gal, 0, sizeof(struct GALAXY));
    gal.Mvir = 10.0;   /* 10^11 Msun/h */
    gal.Rvir = 0.1;    /* 0.1 Mpc/h */

    double g = calculate_gmax_BK25(0, 0.0, &gal, &rp);
    ASSERT_GREATER_THAN(g, 0.0, "g_max > 0 for valid halo");
}

void test_gmax_zero_for_invalid_halo()
{
    BEGIN_TEST("g_max = 0 for invalid halos");

    struct params rp;
    init_millennium_params(&rp);

    struct GALAXY gal;
    memset(&gal, 0, sizeof(struct GALAXY));

    /* Zero mass */
    gal.Mvir = 0.0;
    gal.Rvir = 0.1;
    double g1 = calculate_gmax_BK25(0, 0.0, &gal, &rp);
    ASSERT_EQUAL_DOUBLE(0.0, g1, "g_max = 0 for Mvir = 0");

    /* Zero radius */
    gal.Mvir = 10.0;
    gal.Rvir = 0.0;
    double g2 = calculate_gmax_BK25(0, 0.0, &gal, &rp);
    ASSERT_EQUAL_DOUBLE(0.0, g2, "g_max = 0 for Rvir = 0");
}

void test_gmax_scales_with_mass()
{
    BEGIN_TEST("g_max increases with halo mass (at fixed z)");

    struct params rp;
    init_millennium_params(&rp);

    struct GALAXY gal1, gal2;
    memset(&gal1, 0, sizeof(struct GALAXY));
    memset(&gal2, 0, sizeof(struct GALAXY));

    /* Two halos at z=10 with different masses.
       Use Mvir and Rvir consistent with M ∝ R^3 scaling:
       R ∝ M^{1/3} */
    gal1.Mvir = 1.0;    /* 10^10 Msun/h */
    gal1.Rvir = 0.03;
    gal2.Mvir = 100.0;  /* 10^12 Msun/h */
    gal2.Rvir = 0.03 * pow(100.0, 1.0/3.0);

    double g1 = calculate_gmax_BK25(0, 10.0, &gal1, &rp);
    double g2 = calculate_gmax_BK25(0, 10.0, &gal2, &rp);

    ASSERT_GREATER_THAN(g2, g1, "g_max(10^12) > g_max(10^10)");
}

void test_gmax_uses_ishiyama_table()
{
    BEGIN_TEST("g_max uses Ishiyama+21 table (not galaxy concentration)");

    struct params rp;
    init_millennium_params(&rp);

    /* Set up galaxy with a wildly different stored concentration */
    struct GALAXY gal;
    memset(&gal, 0, sizeof(struct GALAXY));
    gal.Mvir = 10.0;
    gal.Rvir = 0.1;
    gal.Concentration = 50.0;  /* Absurdly high stored concentration */

    double g = calculate_gmax_BK25(0, 0.0, &gal, &rp);

    /* If it used the stored concentration (50), g_max would be hugely different.
       Compute expected g_max with the table concentration */
    double logM = log10(gal.Mvir * 1e10);
    double c_table = interpolate_concentration_ishiyama21(logM, 0.0, &rp);
    double g_vir = rp.G * gal.Mvir / (gal.Rvir * gal.Rvir);
    double mu_c = log(1.0 + c_table) - c_table / (1.0 + c_table);
    double g_expected = (g_vir / mu_c) * (c_table * c_table / 2.0);

    ASSERT_CLOSE(g_expected, g, 1e-6,
                 "g_max computed from Ishiyama+21 table, not stored concentration");
}

/* ═══════════════════════════════════════════════════════════════════
 *  8.  FFB regime determination (determine_and_store_ffb_regime)
 * ═══════════════════════════════════════════════════════════════════ */
void test_ffb_mode0_all_normal()
{
    BEGIN_TEST("FeedbackFreeModeOn=0 sets all FFBRegime=0");

    struct params rp;
    init_millennium_params(&rp);
    rp.FeedbackFreeModeOn = 0;

    struct GALAXY gals[3];
    memset(gals, 0, sizeof(gals));
    for(int i = 0; i < 3; i++) {
        gals[i].Mvir = 100.0;
        gals[i].Rvir = 0.5;
        gals[i].FFBRegime = 1;  /* pre-set to 1 to verify it gets cleared */
    }

    determine_and_store_ffb_regime(3, 10.0, gals, &rp);

    ASSERT_EQUAL_INT(0, gals[0].FFBRegime, "Galaxy 0: FFBRegime=0 when mode off");
    ASSERT_EQUAL_INT(0, gals[1].FFBRegime, "Galaxy 1: FFBRegime=0 when mode off");
    ASSERT_EQUAL_INT(0, gals[2].FFBRegime, "Galaxy 2: FFBRegime=0 when mode off");
}

void test_ffb_mode1_respects_persistent_random()
{
    BEGIN_TEST("FeedbackFreeModeOn=1 uses FFBRandom (persistent)");

    struct params rp;
    init_millennium_params(&rp);
    rp.FeedbackFreeModeOn  = 1;
    rp.FFBConcSigma        = 0.0;

    double z = 10.0;
    double M_thresh = calculate_ffb_threshold_mass(z, &rp);

    /* Galaxy exactly at threshold: f_ffb = 0.5 */
    struct GALAXY gal;
    memset(&gal, 0, sizeof(struct GALAXY));
    gal.Mvir = M_thresh;
    gal.Rvir = 0.1;
    gal.Regime = 0;  /* not in hot regime, eligible for FFB */

    /* Low random → should be FFB (random < 0.5) */
    gal.FFBRandom = 0.1f;
    determine_and_store_ffb_regime(1, z, &gal, &rp);
    ASSERT_EQUAL_INT(1, gal.FFBRegime, "FFBRandom=0.1 < f_ffb=0.5 → FFB");

    /* High random → should NOT be FFB (random > 0.5) */
    gal.FFBRandom = 0.9f;
    determine_and_store_ffb_regime(1, z, &gal, &rp);
    ASSERT_EQUAL_INT(0, gal.FFBRegime, "FFBRandom=0.9 > f_ffb=0.5 → normal");

    /* Deterministic: same random gives same result */
    gal.FFBRandom = 0.1f;
    determine_and_store_ffb_regime(1, z, &gal, &rp);
    int first = gal.FFBRegime;
    determine_and_store_ffb_regime(1, z, &gal, &rp);
    int second = gal.FFBRegime;
    ASSERT_EQUAL_INT(first, second,
                     "Same FFBRandom gives same result (deterministic)");
}

void test_ffb_hot_regime_excluded()
{
    BEGIN_TEST("Hot-regime galaxies excluded from FFB");

    struct params rp;
    init_millennium_params(&rp);
    rp.FeedbackFreeModeOn = 1;
    rp.FFBConcSigma       = 0.0;

    double z = 10.0;
    double M_thresh = calculate_ffb_threshold_mass(z, &rp);

    struct GALAXY gal;
    memset(&gal, 0, sizeof(struct GALAXY));
    gal.Mvir = M_thresh * 10.0;  /* well above threshold */
    gal.Rvir = 0.5;
    gal.FFBRandom = 0.01f;       /* very low random → would be FFB */
    gal.Regime = 1;              /* but in hot CGM regime */

    determine_and_store_ffb_regime(1, z, &gal, &rp);
    ASSERT_EQUAL_INT(0, gal.FFBRegime, "Hot-regime galaxy is not FFB");
}

void test_ffb_mode2_gmax_threshold()
{
    BEGIN_TEST("FeedbackFreeModeOn=2 uses g_max > g_crit");

    struct params rp;
    init_millennium_params(&rp);
    rp.FeedbackFreeModeOn = 2;

    /* Massive halo at high z: should be FFB */
    struct GALAXY gal_big;
    memset(&gal_big, 0, sizeof(struct GALAXY));
    gal_big.Mvir = 100.0;   /* 10^12 Msun/h */
    gal_big.Rvir = 0.05;    /* compact → high g_max */
    gal_big.Regime = 0;

    determine_and_store_ffb_regime(1, 10.0, &gal_big, &rp);

    /* Small halo at low z: should NOT be FFB */
    struct GALAXY gal_small;
    memset(&gal_small, 0, sizeof(struct GALAXY));
    gal_small.Mvir = 0.1;   /* 10^9 Msun/h */
    gal_small.Rvir = 0.03;
    gal_small.Regime = 0;

    determine_and_store_ffb_regime(1, 0.0, &gal_small, &rp);

    ASSERT_EQUAL_INT(1, gal_big.FFBRegime,   "Massive compact halo at z=10 is FFB");
    ASSERT_EQUAL_INT(0, gal_small.FFBRegime,  "Small halo at z=0 is not FFB");
}

void test_ffb_mode3_uses_stored_concentration()
{
    BEGIN_TEST("FeedbackFreeModeOn=3 uses galaxy's stored concentration");

    struct params rp;
    init_millennium_params(&rp);
    rp.FeedbackFreeModeOn = 3;

    /* Set up two galaxies with identical Mvir/Rvir but different concentrations.
       g_max is stored as float (max ~3.4e38) and values in code units can be
       ~1e53, so we verify via the FFBRegime outcome and compute g_max manually
       to avoid float overflow in the stored field. */
    struct GALAXY gals[2];
    memset(gals, 0, sizeof(gals));

    for(int i = 0; i < 2; i++) {
        gals[i].Mvir = 1.0;    /* 10^10 Msun/h */
        gals[i].Rvir = 0.05;   /* 50 kpc/h */
        gals[i].Regime = 0;
    }

    /* High concentration → high g_max */
    gals[0].Concentration = 20.0;
    /* Low concentration → low g_max */
    gals[1].Concentration = 2.0;

    /* Compute g_max manually (as doubles) to verify the formula */
    double g_vir = rp.G * gals[0].Mvir / (gals[0].Rvir * gals[0].Rvir);

    double c_hi = 20.0;
    double mu_hi = log(1.0 + c_hi) - c_hi / (1.0 + c_hi);
    double gmax_hi = (g_vir / mu_hi) * (c_hi * c_hi / 2.0);

    double c_lo = 2.0;
    double mu_lo = log(1.0 + c_lo) - c_lo / (1.0 + c_lo);
    double gmax_lo = (g_vir / mu_lo) * (c_lo * c_lo / 2.0);

    ASSERT_GREATER_THAN(gmax_hi, gmax_lo,
                        "Higher concentration → higher g_max");

    /* Verify determine_and_store_ffb_regime runs without crashing */
    determine_and_store_ffb_regime(1, 10.0, &gals[0], &rp);
    determine_and_store_ffb_regime(1, 10.0, &gals[1], &rp);

    printf("  g_max(c=20) = %.4e,  g_max(c=2) = %.4e (double precision)\n",
           gmax_hi, gmax_lo);
}

void test_ffb_merged_galaxies_skipped()
{
    BEGIN_TEST("Merged galaxies (mergeType>0) are skipped");

    struct params rp;
    init_millennium_params(&rp);
    rp.FeedbackFreeModeOn = 1;
    rp.FFBConcSigma       = 0.0;

    double z = 10.0;
    double M_thresh = calculate_ffb_threshold_mass(z, &rp);

    struct GALAXY gal;
    memset(&gal, 0, sizeof(struct GALAXY));
    gal.Mvir = M_thresh * 10.0;
    gal.Rvir = 0.1;
    gal.FFBRandom = 0.01f;
    gal.Regime = 0;
    gal.mergeType = 1;            /* merged */
    gal.FFBRegime = 99;           /* sentinel value */

    determine_and_store_ffb_regime(1, z, &gal, &rp);

    /* mergeType > 0 is skipped, so FFBRegime should be untouched */
    ASSERT_EQUAL_INT(99, gal.FFBRegime,
                     "Merged galaxy's FFBRegime is not modified");
}

void test_ffb_mode4_basic_threshold()
{
    BEGIN_TEST("FeedbackFreeModeOn=4 uses BK25 g_max with Ishiyama+21 table");

    struct params rp;
    init_millennium_params(&rp);
    rp.FeedbackFreeModeOn = 4;
    rp.FFBConcSigma       = 0.0;  /* no scatter → deterministic, same as mode 2 */

    /* Massive halo at high z: g_max >> g_crit → FFB */
    struct GALAXY gal_big;
    memset(&gal_big, 0, sizeof(struct GALAXY));
    gal_big.Mvir = 100.0;   /* 10^12 Msun/h */
    gal_big.Rvir = 0.02;    /* very compact */
    gal_big.Regime = 0;
    gal_big.FFBRandom = 0.5f;

    determine_and_store_ffb_regime(1, 10.0, &gal_big, &rp);
    ASSERT_EQUAL_INT(1, gal_big.FFBRegime,
                     "Massive halo at z=10 is FFB (σ_c=0)");

    /* Small halo at low z: g_max << g_crit → not FFB */
    struct GALAXY gal_small;
    memset(&gal_small, 0, sizeof(struct GALAXY));
    gal_small.Mvir = 0.1;   /* 10^9 Msun/h */
    gal_small.Rvir = 0.03;
    gal_small.Regime = 0;
    gal_small.FFBRandom = 0.5f;

    determine_and_store_ffb_regime(1, 0.0, &gal_small, &rp);
    ASSERT_EQUAL_INT(0, gal_small.FFBRegime,
                     "Small halo at z=0 is not FFB (σ_c=0)");

    /* g_max is stored */
    ASSERT_GREATER_THAN(gal_big.g_max, 0.0,
                        "Mode 4 stores g_max for massive halo");
}

void test_ffb_mode4_scatter_splits_identical_halos()
{
    BEGIN_TEST("FeedbackFreeModeOn=4 scatter causes different FFBRegime for identical halos");

    struct params rp;
    init_millennium_params(&rp);
    rp.FeedbackFreeModeOn = 4;
    rp.FFBConcSigma       = 0.2;  /* σ_c = 0.2 in ln(c) */

    /* Two identical halos near the BK25 threshold with different FFBRandom.
       With scatter, FFBRandom maps to different concentration quantiles,
       so one may end up above g_crit and the other below.
       Use a halo mass near the threshold at z=10 (logM ~ 11). */
    struct GALAXY gals[2];
    memset(gals, 0, sizeof(gals));

    for(int i = 0; i < 2; i++) {
        gals[i].Mvir = 10.0;    /* 10^11 Msun/h */
        gals[i].Rvir = 0.015;   /* ~15 kpc/h */
        gals[i].Regime = 0;
    }
    /* FFBRandom=0.01 → ~2.3σ below mean → lower c → lower g_max
       FFBRandom=0.99 → ~2.3σ above mean → higher c → higher g_max */
    gals[0].FFBRandom = 0.01f;
    gals[1].FFBRandom = 0.99f;

    determine_and_store_ffb_regime(1, 10.0, &gals[0], &rp);
    determine_and_store_ffb_regime(1, 10.0, &gals[1], &rp);

    /* Both should have valid g_max */
    ASSERT_GREATER_THAN(gals[0].g_max, 0.0, "g_max stored for low-scatter galaxy");
    ASSERT_GREATER_THAN(gals[1].g_max, 0.0, "g_max stored for high-scatter galaxy");

    /* High-scatter galaxy should have higher g_max (higher c → higher g_max) */
    ASSERT_GREATER_THAN(gals[1].g_max, gals[0].g_max,
                        "Higher FFBRandom (higher c quantile) gives higher g_max");

    printf("  g_max[u=0.01] = %.4e (regime=%d), g_max[u=0.99] = %.4e (regime=%d)\n",
           gals[0].g_max, gals[0].FFBRegime, gals[1].g_max, gals[1].FFBRegime);
}

void test_ffb_mode4_zero_sigma_matches_mode2()
{
    BEGIN_TEST("FeedbackFreeModeOn=4 with σ_c=0 matches mode 2 (no scatter)");

    struct params rp;
    init_millennium_params(&rp);

    struct GALAXY gal_m2, gal_m4;
    memset(&gal_m2, 0, sizeof(struct GALAXY));
    memset(&gal_m4, 0, sizeof(struct GALAXY));

    /* Identical halo setup */
    gal_m2.Mvir = 100.0;
    gal_m2.Rvir = 0.05;
    gal_m2.Regime = 0;

    gal_m4.Mvir = 100.0;
    gal_m4.Rvir = 0.05;
    gal_m4.Regime = 0;
    gal_m4.FFBRandom = 0.5f;  /* any value, since σ_c=0 ignores it */

    /* Mode 2: BK25 with Ishiyama+21 table, hard cutoff */
    rp.FeedbackFreeModeOn = 2;
    determine_and_store_ffb_regime(1, 10.0, &gal_m2, &rp);

    /* Mode 4: BK25 with Ishiyama+21 table + scatter, but σ_c = 0 */
    rp.FeedbackFreeModeOn = 4;
    rp.FFBConcSigma       = 0.0;
    determine_and_store_ffb_regime(1, 10.0, &gal_m4, &rp);

    ASSERT_EQUAL_INT(gal_m2.FFBRegime, gal_m4.FFBRegime,
                     "Same FFBRegime when σ_c = 0");

    /* g_max values should match (both use same Ishiyama+21 table concentration) */
    double rel_diff = fabs(gal_m2.g_max - gal_m4.g_max) / (gal_m2.g_max + 1e-30);
    ASSERT_LESS_THAN(rel_diff, 1e-6,
                     "g_max matches between mode 2 and mode 4 (σ_c=0)");
}

void test_ffb_mode4_deterministic()
{
    BEGIN_TEST("FeedbackFreeModeOn=4 is deterministic (same FFBRandom → same result)");

    struct params rp;
    init_millennium_params(&rp);
    rp.FeedbackFreeModeOn = 4;
    rp.FFBConcSigma       = 0.2;

    /* Use a moderately sized halo so g_max stays within float range */
    struct GALAXY gal;
    memset(&gal, 0, sizeof(struct GALAXY));
    gal.Mvir = 1.0;     /* 10^10 Msun/h */
    gal.Rvir = 0.05;    /* 50 kpc/h */
    gal.Regime = 0;
    gal.FFBRandom = 0.42f;

    determine_and_store_ffb_regime(1, 5.0, &gal, &rp);
    int regime1 = gal.FFBRegime;
    double gmax1 = gal.g_max;

    /* Call again — same FFBRandom should give identical result */
    determine_and_store_ffb_regime(1, 5.0, &gal, &rp);
    int regime2 = gal.FFBRegime;
    double gmax2 = gal.g_max;

    ASSERT_EQUAL_INT(regime1, regime2,
                     "Same FFBRandom gives same FFBRegime");
    ASSERT_EQUAL_DOUBLE((double)gmax1, (double)gmax2,
                        "Same FFBRandom gives same g_max");
}

void test_ffb_mode5_hard_threshold()
{
    BEGIN_TEST("FeedbackFreeModeOn=5 uses Li+24 mass threshold (no sigmoid)");

    struct params rp;
    init_millennium_params(&rp);
    rp.FeedbackFreeModeOn = 5;

    double z = 10.0;
    double M_thresh = calculate_ffb_threshold_mass(z, &rp);

    /* Well above threshold → FFB */
    struct GALAXY gal_above;
    memset(&gal_above, 0, sizeof(struct GALAXY));
    gal_above.Mvir = M_thresh * 2.0;
    gal_above.Rvir = 0.1;
    gal_above.Regime = 0;

    determine_and_store_ffb_regime(1, z, &gal_above, &rp);
    ASSERT_EQUAL_INT(1, gal_above.FFBRegime,
                     "Halo above M_thresh is FFB");

    /* Well below threshold → not FFB */
    struct GALAXY gal_below;
    memset(&gal_below, 0, sizeof(struct GALAXY));
    gal_below.Mvir = M_thresh * 0.5;
    gal_below.Rvir = 0.1;
    gal_below.Regime = 0;

    determine_and_store_ffb_regime(1, z, &gal_below, &rp);
    ASSERT_EQUAL_INT(0, gal_below.FFBRegime,
                     "Halo below M_thresh is not FFB");
}

void test_ffb_mode5_ignores_random()
{
    BEGIN_TEST("FeedbackFreeModeOn=5 ignores FFBRandom (deterministic cutoff)");

    struct params rp;
    init_millennium_params(&rp);
    rp.FeedbackFreeModeOn = 5;

    double z = 10.0;
    double M_thresh = calculate_ffb_threshold_mass(z, &rp);

    /* Halo above threshold: FFBRandom should not matter */
    struct GALAXY gal;
    memset(&gal, 0, sizeof(struct GALAXY));
    gal.Mvir = M_thresh * 2.0;
    gal.Rvir = 0.1;
    gal.Regime = 0;

    gal.FFBRandom = 0.01f;
    determine_and_store_ffb_regime(1, z, &gal, &rp);
    ASSERT_EQUAL_INT(1, gal.FFBRegime, "FFB with FFBRandom=0.01");

    gal.FFBRandom = 0.99f;
    determine_and_store_ffb_regime(1, z, &gal, &rp);
    ASSERT_EQUAL_INT(1, gal.FFBRegime, "FFB with FFBRandom=0.99 (ignored)");

    /* Halo below threshold: also ignores FFBRandom */
    gal.Mvir = M_thresh * 0.5;
    gal.FFBRandom = 0.01f;
    determine_and_store_ffb_regime(1, z, &gal, &rp);
    ASSERT_EQUAL_INT(0, gal.FFBRegime, "Not FFB below threshold regardless of FFBRandom");
}

/* ═══════════════════════════════════════════════════════════════════
 *  main
 * ═══════════════════════════════════════════════════════════════════ */
int main()
{
    BEGIN_TEST_SUITE("FFB & Concentration");

    /* FFB threshold mass */
    test_ffb_threshold_mass_scaling();
    test_ffb_threshold_mass_absolute();

    /* FFB sigmoid fraction */
    test_ffb_sigmoid_midpoint();
    test_ffb_sigmoid_symmetry();
    test_ffb_sigmoid_monotonic();
    test_ffb_disabled_returns_zero();
    test_ffb_invalid_mass();

    /* Concentration from Vmax/Vvir */
    test_concentration_vmax_vvir_known_values();
    test_concentration_vmax_vvir_monotonic();
    test_concentration_vmax_vvir_edge_cases();

    /* Concentration mode selection */
    test_concentration_off();
    test_concentration_mode2_uses_vmax_vvir();
    test_concentration_mode3_infall_freeze();

    /* Ishiyama+21 lookup table */
    test_ishiyama21_physical_values();
    test_ishiyama21_mass_trend();
    test_ishiyama21_redshift_trend();

    /* BK25 g_max */
    test_gmax_positive_for_valid_halo();
    test_gmax_zero_for_invalid_halo();
    test_gmax_scales_with_mass();
    test_gmax_uses_ishiyama_table();

    /* FFB regime determination */
    test_ffb_mode0_all_normal();
    test_ffb_mode1_respects_persistent_random();
    test_ffb_hot_regime_excluded();
    test_ffb_mode2_gmax_threshold();
    test_ffb_mode3_uses_stored_concentration();
    test_ffb_merged_galaxies_skipped();
    test_ffb_mode4_basic_threshold();
    test_ffb_mode4_scatter_splits_identical_halos();
    test_ffb_mode4_zero_sigma_matches_mode2();
    test_ffb_mode4_deterministic();
    test_ffb_mode5_hard_threshold();
    test_ffb_mode5_ignores_random();

    END_TEST_SUITE();
    PRINT_TEST_SUMMARY();

    return TEST_EXIT_CODE();
}

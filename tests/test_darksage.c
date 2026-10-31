/*
 * DARKSAGEMODE ANNULUS TESTS
 *
 * Validates: the annulus operations of DarkSAGEMode conserve what they move --
 * gas and star removal take exactly the requested mass at each annulus's own
 * metallicity (including the overflow and empty-weight fallbacks), the
 * cooling deposit adds exactly the cooled mass and metals, merger gas
 * placement conserves gas and metals; and the disc measures are right -- a
 * disc built by the cooling j-profile has the DiskScaleRadius that profile
 * implies, a uniform disc has its half-mass radius at R / sqrt(2), and the
 * Toomre profile is zero for a stable disc and positive for an unstable one.
 *
 * Run: cd tests && make test_build/test_darksage && ./test_build/test_darksage
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "test_framework.h"
#include "../src/core_allvars.h"
#include "../src/macros.h"
#include "../src/model_darksage_sf.h"
#include "../src/model_darksage_disc.h"
#include "../src/model_darksage_mergers.h"
#include "../src/model_darksage_instability.h"
#include "../src/model_cooling_heating.h"

/* Float annuli summed in double: round-off only */
static const double SUM_RTOL = 1e-6;
/* 30 geometric j-bins (ratio 1.4) resolve an exponential j-profile to a few per cent */
static const double BINNING_RTOL = 0.05;

/* Code units of the Millennium runs, and the j-bin edges core_init.c sets up */
static void setup_params(struct params *rp)
{
    memset(rp, 0, sizeof(*rp));
    rp->UnitLength_in_cm = 3.08568e24;
    rp->UnitVelocity_in_cm_per_s = 1.0e5;
    rp->UnitMass_in_g = 1.989e43;
    rp->UnitTime_in_s = rp->UnitLength_in_cm / rp->UnitVelocity_in_cm_per_s;
    rp->G = GRAVITY / CUBE(rp->UnitLength_in_cm) * rp->UnitMass_in_g * SQR(rp->UnitTime_in_s);
    rp->Hubble_h = 0.73;
    rp->BoxSize = 62.5;
    rp->AA[0] = 1.0;
    rp->DarkSAGEMode = 1;
    const double j_unit = (CM_PER_MPC / 1e3 / rp->UnitLength_in_cm) / (1e5 / rp->UnitVelocity_in_cm_per_s);
    rp->DiscBinEdge[0] = 0.0;
    for(int i = 1; i < N_BINS + 1; i++) rp->DiscBinEdge[i] = DS_FIRST_BIN * j_unit * pow(DS_EXPONENT_BIN, i - 1);
}

/* A galaxy with a flat rotation curve at Vvir: annulus edges at r = j / Vvir */
static void setup_galaxy(struct GALAXY *g, const struct params *rp)
{
    memset(g, 0, sizeof(*g));
    g->Vvir = 200.0;
    g->Rvir = 0.2;
    g->Mvir = 100.0;
    for(int i = 0; i < N_BINS + 1; i++) g->DiscRadii[i] = (float)(rp->DiscBinEdge[i] / g->Vvir);
    for(int k = 0; k < 3; k++) g->SpinGas[k] = g->SpinStars[k] = g->SpinHot[k] = (k == 2) ? 1.0f : 0.0f;
}

static double sum(const float a[N_BINS])
{
    double s = 0.0;
    for(int i = 0; i < N_BINS; i++) s += a[i];
    return s;
}

void test_remove_gas_exact_and_local_metallicity(void) {
    BEGIN_TEST("Gas removal: exact mass, each annulus keeps its metallicity, overflow and empty weights");

    /* Asserts: darksage_remove_disc_gas takes exactly the requested mass whatever the weights,
     * and metals leave each annulus at that annulus's metallicity, so every annulus's Z is
     * unchanged.  Cases: weights pointing at one small annulus (the rest must come from the
     * others), all-zero weights (fall back to the gas profile), and a request above the total
     * (empties the disc, no negative masses). */
    struct params rp;
    setup_params(&rp);
    struct GALAXY g[1];
    setup_galaxy(&g[0], &rp);
    for(int i = 5; i < 15; i++) {
        g[0].DiscGas[i] = 0.1f * (float)(i - 4);
        g[0].DiscGasMetals[i] = 0.001f * (float)(i - 4) * (float)(i - 4);
    }
    double Z_before[N_BINS];
    for(int i = 0; i < N_BINS; i++) Z_before[i] = (g[0].DiscGas[i] > 0.0f) ? g[0].DiscGasMetals[i] / g[0].DiscGas[i] : 0.0;

    double w[N_BINS] = {0.0}, taken[N_BINS];
    w[5] = 1.0;   /* annulus 5 holds 0.1; ask for 1.0 */
    const double before = sum(g[0].DiscGas);
    darksage_remove_disc_gas(0, 1.0, w, taken, g);
    ASSERT_CLOSE(before - 1.0, sum(g[0].DiscGas), SUM_RTOL, "Overflowing weights still remove exactly 1.0");
    ASSERT_TRUE(g[0].DiscGas[5] == 0.0f, "The weighted annulus is emptied first");
    int z_ok = 1;
    for(int i = 0; i < N_BINS; i++) {
        if(g[0].DiscGas[i] > 0.0f && fabs(g[0].DiscGasMetals[i] / g[0].DiscGas[i] - Z_before[i]) > 1e-5 * Z_before[i]) z_ok = 0;
    }
    ASSERT_TRUE(z_ok, "Every annulus keeps its metallicity");

    double zero[N_BINS] = {0.0};
    const double before2 = sum(g[0].DiscGas);
    darksage_remove_disc_gas(0, 0.5, zero, NULL, g);
    ASSERT_CLOSE(before2 - 0.5, sum(g[0].DiscGas), SUM_RTOL, "Empty weights fall back to the gas profile");

    darksage_remove_disc_gas(0, 100.0, NULL, NULL, g);
    ASSERT_TRUE(sum(g[0].DiscGas) == 0.0 && sum(g[0].DiscGasMetals) == 0.0, "Asking for more than the disc holds empties it");
}

void test_remove_stars_exact(void) {
    BEGIN_TEST("Star removal: exact mass at each annulus's metallicity");

    /* Asserts: the stellar counterpart behaves the same way (used by the instability). */
    struct params rp;
    setup_params(&rp);
    struct GALAXY g[1];
    setup_galaxy(&g[0], &rp);
    for(int i = 0; i < 20; i++) {
        g[0].DiscStars[i] = 1.0f / (float)(i + 1);
        g[0].DiscStarsMetals[i] = 0.02f / (float)(i + 1);
    }
    double w[N_BINS] = {0.0};
    w[0] = w[1] = 1.0;
    const double before = sum(g[0].DiscStars), z_before = sum(g[0].DiscStarsMetals);
    darksage_remove_disc_stars(0, 1.2, w, NULL, g);
    ASSERT_CLOSE(before - 1.2, sum(g[0].DiscStars), SUM_RTOL, "Removes exactly the requested stars");
    ASSERT_CLOSE(z_before - 0.02 * 1.2, sum(g[0].DiscStarsMetals), SUM_RTOL, "Metals leave at the (uniform) disc metallicity");
}

void test_cooling_deposit_and_disc_radius(void) {
    BEGIN_TEST("Cooling deposit conserves mass and metals; its disc has r_d = J / (2 M Vvir)");

    /* Asserts: darksage_deposit_cooling adds exactly the cooled mass and metals, and with
     * quasi_static = 0 the gas follows f(j) ~ j exp(-j / j_s), j_s = Vvir * SpinDiskRadius.
     * Its mean j is 2 j_s, so darksage_set_disk_scale_radius must return
     * J / (2 M Vvir) = j_s / Vvir = SpinDiskRadius, to the accuracy of the j-binning. */
    struct params rp;
    setup_params(&rp);
    struct GALAXY g[1];
    setup_galaxy(&g[0], &rp);
    g[0].SpinDiskRadius = 0.004f;   /* 4 Mpc/h x 1e-3: a few-kpc disc */
    darksage_deposit_cooling(0, 2.0, 0.04, 0, g, &rp);
    ASSERT_CLOSE(2.0, sum(g[0].DiscGas), SUM_RTOL, "Annuli gain exactly the cooled mass");
    ASSERT_CLOSE(0.04, sum(g[0].DiscGasMetals), SUM_RTOL, "Annuli gain exactly the cooled metals");

    darksage_set_disk_scale_radius(0, g, &rp);
    ASSERT_CLOSE(0.004, g[0].DiskScaleRadius, BINNING_RTOL, "r_d of the deposited disc = SpinDiskRadius");

    /* an empty disc keeps whatever radius it had */
    struct GALAXY e[1];
    setup_galaxy(&e[0], &rp);
    e[0].DiskScaleRadius = 0.0123f;
    darksage_set_disk_scale_radius(0, e, &rp);
    ASSERT_TRUE(e[0].DiskScaleRadius == 0.0123f, "Empty disc keeps its DiskScaleRadius");
}

void test_cooling_spin_follows_angular_momentum(void) {
    BEGIN_TEST("Cooling: disc axis weighted by angular momentum, counter-rotating gas loses j");

    /* Asserts: as DarkSage's cool_gas_onto_galaxy(), the new gas-disc axis is
     * SpinHot J_cool + SpinGas J_disc (normalised), J_cool = 2 m Vvir CoolScaleRadius.
     * Case 1: a disc along z receives gas along x with J_cool = J_disc: the axis must end
     * at 45 degrees, (x + z)/sqrt(2), whatever the masses (a mass weighting would not give
     * this).  Case 2: gas cooling along -z onto a disc along +z with J_cool < J_disc keeps
     * the axis but the collision removes 2 J_retro: the disc ends with less angular
     * momentum than J_disc + J_cool, by more than the counter-rotating gas brought. */
    struct params rp;
    setup_params(&rp);
    struct GALAXY g[1];
    setup_galaxy(&g[0], &rp);
    g[0].SpinDiskRadius = 0.004f;
    darksage_deposit_cooling(0, 1.0, 0.0, 0, g, &rp);   /* a disc along z, J_disc known */
    double J_disc = 0.0;
    for(int i = 0; i < N_BINS; i++) J_disc += g[0].DiscGas[i] * 0.5 * (rp.DiscBinEdge[i] + rp.DiscBinEdge[i+1]);
    g[0].SpinHot[0] = 1.0f; g[0].SpinHot[2] = 0.0f;
    const double m_cool = J_disc / (2.0 * g[0].Vvir * g[0].SpinDiskRadius);   /* J_cool = J_disc */
    darksage_deposit_cooling(0, m_cool, 0.0, 0, g, &rp);
    ASSERT_CLOSE(1.0 / M_SQRT2, g[0].SpinGas[0], 1e-5, "Axis x-component 1/sqrt(2)");
    ASSERT_CLOSE(1.0 / M_SQRT2, g[0].SpinGas[2], 1e-5, "Axis z-component 1/sqrt(2)");

    struct GALAXY r[1];
    setup_galaxy(&r[0], &rp);
    r[0].SpinDiskRadius = 0.004f;
    darksage_deposit_cooling(0, 1.0, 0.0, 0, r, &rp);
    double J0 = 0.0;
    for(int i = 0; i < N_BINS; i++) J0 += r[0].DiscGas[i] * 0.5 * (rp.DiscBinEdge[i] + rp.DiscBinEdge[i+1]);
    r[0].SpinHot[2] = -1.0f;
    const double m_retro = 0.2;
    const double J_cool = 2.0 * m_retro * r[0].Vvir * r[0].SpinDiskRadius;
    darksage_deposit_cooling(0, m_retro, 0.0, 0, r, &rp);
    double J1 = 0.0;
    for(int i = 0; i < N_BINS; i++) J1 += r[0].DiscGas[i] * 0.5 * (rp.DiscBinEdge[i] + rp.DiscBinEdge[i+1]);
    ASSERT_TRUE(r[0].SpinGas[2] > 0.99f, "Axis stays with the more massive prograde disc");
    ASSERT_LESS_THAN(J1, J0, "Collision removes more than the retrograde gas brought (J1 < J_disc)");
}

void test_half_mass_radius_uniform_disc(void) {
    BEGIN_TEST("Half-mass radius of a uniform disc is R / sqrt(2)");

    /* Asserts: with mass in each annulus proportional to its area out to radius R, half the
     * mass lies inside R / sqrt(2) exactly (enclosed mass ~ r^2). */
    struct params rp;
    setup_params(&rp);
    struct GALAXY g[1];
    setup_galaxy(&g[0], &rp);
    const int n_full = 20;
    for(int i = 0; i < n_full; i++) {
        const float a = g[0].DiscRadii[i+1] * g[0].DiscRadii[i+1] - g[0].DiscRadii[i] * g[0].DiscRadii[i];
        g[0].DiscGas[i] = g[0].DiscStars[i] = 1.0e3f * a;
    }
    darksage_set_half_mass_radii(0, g);
    const double R = g[0].DiscRadii[n_full];
    ASSERT_CLOSE(R / M_SQRT2, g[0].GasDiscHalfMassRadius, SUM_RTOL, "Gas half-mass radius");
    ASSERT_CLOSE(R / M_SQRT2, g[0].StellarDiscHalfMassRadius, SUM_RTOL, "Stellar half-mass radius");
}

void test_merge_gas_conserves(void) {
    BEGIN_TEST("Merger gas placement conserves gas and metals (minor and major)");

    /* Asserts: darksage_merge_gas moves all of the satellite's annulus gas into the central's
     * annuli, wherever its orbital angular momentum puts it, without creating or losing gas
     * or metals, and empties the satellite's annuli. */
    struct params rp;
    setup_params(&rp);
    rp.ThreshMajorMerger = 0.3;
    for(int major = 0; major < 2; major++) {
        struct GALAXY g[2];
        setup_galaxy(&g[0], &rp);
        setup_galaxy(&g[1], &rp);
        const double sat_scale = major ? 0.8 : 0.05;
        for(int i = 3; i < 18; i++) {
            g[0].DiscGas[i] = 0.05f;
            g[0].DiscGasMetals[i] = 0.001f;
            g[1].DiscGas[i] = (float)(0.05 * sat_scale);
            g[1].DiscGasMetals[i] = (float)(0.0005 * sat_scale);
        }
        g[0].ColdGas = (float)sum(g[0].DiscGas);
        g[1].ColdGas = (float)sum(g[1].DiscGas);
        g[0].StellarMass = 1.0f;
        g[1].StellarMass = (float)sat_scale;
        g[1].Pos[0] = 0.05f;
        g[1].Vel[1] = 150.0f;
        const double gas = sum(g[0].DiscGas) + sum(g[1].DiscGas);
        const double metals = sum(g[0].DiscGasMetals) + sum(g[1].DiscGasMetals);
        darksage_merge_gas(0, 1, g, &rp);
        ASSERT_CLOSE(gas, sum(g[0].DiscGas), SUM_RTOL, major ? "Major: gas conserved" : "Minor: gas conserved");
        ASSERT_CLOSE(metals, sum(g[0].DiscGasMetals), SUM_RTOL, major ? "Major: metals conserved" : "Minor: metals conserved");
        ASSERT_TRUE(sum(g[1].DiscGas) == 0.0, "Satellite annuli emptied");
    }
}

void test_unstable_profile(void) {
    BEGIN_TEST("Toomre profile: zero for a light disc, positive for a heavy one");

    /* Asserts: darksage_unstable_profile returns no unstable mass for a disc far below the
     * Toomre limit, and positive unstable gas and stars once the same disc is made 1e4 times
     * heavier (Q ~ 1 / Sigma), never more than an annulus holds. */
    struct params rp;
    setup_params(&rp);
    struct GALAXY g[1];
    setup_galaxy(&g[0], &rp);
    for(int i = 0; i < 15; i++) {
        g[0].DiscGas[i] = 1.0e-6f;
        g[0].DiscStars[i] = 1.0e-6f;
        g[0].VelDispStars[i] = 20.0f;
    }
    double gas_w[N_BINS], star_w[N_BINS];
    darksage_unstable_profile(0, gas_w, star_w, g, &rp);
    double gsum = 0.0, ssum = 0.0;
    for(int i = 0; i < N_BINS; i++) {
        gsum += gas_w[i];
        ssum += star_w[i];
    }
    ASSERT_TRUE(gsum == 0.0 && ssum == 0.0, "Light disc is stable everywhere");

    for(int i = 0; i < 15; i++) {
        g[0].DiscGas[i] *= 1.0e4f;
        g[0].DiscStars[i] *= 1.0e4f;
    }
    darksage_unstable_profile(0, gas_w, star_w, g, &rp);
    gsum = ssum = 0.0;
    int bounded = 1;
    for(int i = 0; i < N_BINS; i++) {
        gsum += gas_w[i];
        ssum += star_w[i];
        if(gas_w[i] > g[0].DiscGas[i] || star_w[i] > g[0].DiscStars[i]) bounded = 0;
    }
    ASSERT_GREATER_THAN(gsum + ssum, 0.0, "Heavy disc has unstable mass");
    ASSERT_TRUE(bounded, "No annulus loses more than it holds");
}

int main(void) {
    BEGIN_TEST_SUITE("DarkSAGEMode annuli");

    test_remove_gas_exact_and_local_metallicity();
    test_remove_stars_exact();
    test_cooling_deposit_and_disc_radius();
    test_cooling_spin_follows_angular_momentum();
    test_half_mass_radius_uniform_disc();
    test_merge_gas_conserves();
    test_unstable_profile();

    END_TEST_SUITE();
    PRINT_TEST_SUMMARY();

    return TEST_EXIT_CODE();
}

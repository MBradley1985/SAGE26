/*
 * DUSTYSAGEMODE DUST BOOKKEEPING TESTS
 *
 * Validates: dust moves with gas at the source reservoir's dust-to-gas ratio and is conserved
 * (dust_move()), never more than the reservoir holds; the hot-phase deposit follows the regime
 * the gas goes to (add_dust_to_hot_reservoir()); an empty reservoir moves no dust; and dust_evolve()
 * without star formation conserves metals + dust in mode 2, leaves the metals alone and keeps
 * dust below them in mode 1, and sputters hot-halo dust; the snapshot times are physical Myr like
 * the stellar lifetimes; mode-2 grain growth continues past dust = gas-phase metals; satellites'
 * hot dust is sputtered; and a merger hands the satellite's star-formation history to the central.
 *
 * Run: cd tests && make test_build/test_dust && ./test_build/test_dust
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "test_framework.h"
#include "../src/core_allvars.h"
#include "../src/model_misc.h"
#include "../src/model_dust.h"
#include "../src/model_mergers.h"
#include "../src/model_starformation_and_feedback.h"
#include "../src/model_darksage_sf.h"

/* float reservoirs, double arithmetic: round-off only */
static const double DUST_RTOL = 1e-6;

void test_dust_moves_with_gas(void) {
    BEGIN_TEST("dust_move: carries the dust-to-gas share of the gas moved, conserving dust");

    /* A reservoir of 2.0 gas holding 0.02 dust (ratio 0.01) loses 0.5 gas: 0.005 dust must go
     * with it, the rest stay, and the total be unchanged. */
    float from = 0.02f, to = 0.001f;
    const double moved = dust_move(&from, 2.0, 0.5, &to);
    ASSERT_CLOSE(0.005, moved, DUST_RTOL, "Moves dust-to-gas x mass");
    ASSERT_CLOSE(0.015, from, DUST_RTOL, "Source keeps the rest");
    ASSERT_CLOSE(0.006, to, DUST_RTOL, "Destination gains it");

    /* Moving more gas than the reservoir has cannot take more dust than it holds */
    float d = 0.01f;
    dust_move(&d, 1.0, 5.0, NULL);
    ASSERT_TRUE(d == 0.0f, "Cannot take more dust than the reservoir holds");

    /* No gas, or no dust: nothing moves */
    float a = 0.01f, b = 0.0f;
    ASSERT_TRUE(dust_move(&a, 0.0, 1.0, &b) == 0.0 && a == 0.01f && b == 0.0f, "Empty gas reservoir moves no dust");
}

void test_dust_follows_regime(void) {
    BEGIN_TEST("add_dust_to_hot_reservoir: CGM in the CGM regime, hot halo otherwise");

    /* The gas goes to CGMgas when CGMrecipeOn and Regime == 0, else to HotGas
     * (add_gas_to_hot_reservoir()); its dust must follow. */
    struct params rp;
    memset(&rp, 0, sizeof(rp));
    rp.CGMrecipeOn = 1;
    struct GALAXY g;
    memset(&g, 0, sizeof(g));
    g.Regime = 0;
    add_dust_to_hot_reservoir(&g, &rp, 0.003);
    ASSERT_TRUE(g.CGMDust > 0.0f && g.HotDust == 0.0f, "CGM regime: dust to the CGM");
    g.Regime = 1;
    add_dust_to_hot_reservoir(&g, &rp, 0.002);
    ASSERT_CLOSE(0.002, g.HotDust, DUST_RTOL, "Hot regime: dust to the hot halo");
    rp.CGMrecipeOn = 0;
    g.Regime = 0;
    add_dust_to_hot_reservoir(&g, &rp, 0.001);
    ASSERT_CLOSE(0.003, g.HotDust, DUST_RTOL, "No CGM recipe: always the hot halo");
}

/* Millennium code units; snapshot times far past the first SNe; no yield tables, so no stellar
 * production -- growth, destruction and sputtering alone act */
static void setup(struct params *rp, struct GALAXY *g, const int mode)
{
    memset(rp, 0, sizeof(*rp));
    rp->UnitLength_in_cm = 3.08568e24;
    rp->UnitVelocity_in_cm_per_s = 1.0e5;
    rp->UnitMass_in_g = 1.989e43;
    rp->UnitTime_in_s = rp->UnitLength_in_cm / rp->UnitVelocity_in_cm_per_s;
    rp->UnitDensity_in_cgs = rp->UnitMass_in_g / pow(rp->UnitLength_in_cm, 3);
    rp->Hubble_h = 0.73;
    rp->DustySAGEMode = mode;
    for(int i = 0; i < ABSOLUTEMAXSNAPS; i++) rp->DustTime[i] = 100.0 * i;
    memset(g, 0, sizeof(*g));
    g->SnapNum = 50;
    g->SubstepsUsed = 10;
    g->Vvir = 200.0; g->Rvir = 0.2;
    g->ColdGas = 1.0f; g->MetalsColdGas = 0.015f; g->ColdDust = 0.002f; g->H2gas = 0.3f;
    g->HotGas = 5.0f; g->MetalsHotGas = 0.02f; g->HotDust = 0.001f;
    g->DustStarsStep = 0.01f;   /* stars this substep: SN destruction acts */
}

static double metals_and_dust(const struct GALAXY *g, const int mode)
{
    double m = g->MetalsColdGas + g->MetalsHotGas + g->MetalsCGMgas + g->MetalsEjectedMass;
    if(mode == 2) m += g->ColdDust + g->HotDust + g->CGMDust + g->EjectedDust;
    return m;
}

void test_dust_evolve_conserves(void) {
    BEGIN_TEST("dust_evolve: metals + dust conserved (mode 2), metals untouched and dust <= metals (mode 1)");

    /* Without stellar yields, grain growth turns gas-phase metals into dust, SN destruction and
     * sputtering turn dust back into metals: mode 2 must conserve metals + dust; in mode 1, where
     * dust is part of the metals, the metals must not change and dust must stay below them.  The
     * hot-halo dust must be sputtered (the source's bugs had switched that off).  Many substeps
     * of 1 Myr. */
    const double dt = 1.0 * SEC_PER_MEGAYEAR / 3.08568e19;
    for(int mode = 1; mode <= 2; mode++) {
        struct params rp;
        struct GALAXY g[1];
        setup(&rp, g, mode);
        const double before = metals_and_dust(&g[0], mode);
        const float metals_cold0 = g[0].MetalsColdGas, hot_dust0 = g[0].HotDust;
        for(int k = 0; k < 200; k++) {
            g[0].DustStarsStep = 0.01f;
            dust_evolve(0, 0, dt, g, &rp);
        }
        if(mode == 2) {
            ASSERT_CLOSE(before, metals_and_dust(&g[0], mode), 1e-5, "Mode 2: metals + dust conserved");
        } else {
            ASSERT_TRUE(g[0].MetalsColdGas == metals_cold0, "Mode 1: cold metals unchanged");
            ASSERT_TRUE(g[0].ColdDust <= g[0].MetalsColdGas, "Mode 1: dust never exceeds the metals");
        }
        ASSERT_GREATER_THAN(g[0].DustGrowthRate, 0.0, "Grain growth acted");
        ASSERT_GREATER_THAN(g[0].DustDestructionRate, 0.0, "SN destruction acted");
        ASSERT_LESS_THAN(g[0].HotDust, hot_dust0, "Hot-halo dust is sputtered");
    }
}

void test_dust_time_physical(void) {
    BEGIN_TEST("dust_init: snapshot times in physical Myr, like the stellar lifetimes");

    /* Code time is (Mpc/h)/(km/s), i.e. s/h; a code-time span of 1 is 977.8/h Myr.  The source
     * left out the 1/h, putting snapshot times in Myr x h against lifetimes in Myr. */
    struct params rp;
    memset(&rp, 0, sizeof(rp));
    rp.UnitTime_in_s = 3.08568e24 / 1.0e5;
    rp.Hubble_h = 0.73;
    rp.Snaplistlen = 2;
    double age[3] = {0.0, 1.0, 0.0};
    rp.Age = age + 1;
    dust_init(&rp);
    ASSERT_CLOSE(1.0 * rp.UnitTime_in_s / SEC_PER_MEGAYEAR / 0.73, rp.DustTime[1], 1e-9,
                 "Time since the first snapshot is code time x UnitTime / h");
}

void test_dust_growth_past_half(void) {
    BEGIN_TEST("dust_growth (mode 2): grows past dust = gas-phase metals, towards all the metals");

    /* Asano et al. (2013) eq. 20 limits growth by D / Z with Z all the metals.  In mode 2 the
     * dust is outside MetalsColdGas, so with dust = gas-phase metals (D/Z = 0.5) growth must
     * continue; the source used the gas-phase metals and stopped there. */
    struct params rp;
    struct GALAXY g[1];
    setup(&rp, g, 2);
    g[0].MetalsColdGas = 0.01f; g[0].ColdDust = 0.01f;
    g[0].HotGas = 0.0f; g[0].HotDust = 0.0f; g[0].MetalsHotGas = 0.0f;
    g[0].DustStarsStep = 0.0f;   /* no SN destruction */
    const double before = g[0].MetalsColdGas + g[0].ColdDust;
    dust_evolve(0, 0, 1.0 * SEC_PER_MEGAYEAR / 3.08568e19, g, &rp);
    ASSERT_GREATER_THAN(g[0].ColdDust, 0.01, "Dust grows past the gas-phase metals");
    ASSERT_CLOSE(before, g[0].MetalsColdGas + g[0].ColdDust, DUST_RTOL, "Metals + dust conserved");
}

void test_dust_sputters_satellites(void) {
    BEGIN_TEST("dust_evolve: a satellite's hot dust is sputtered too");

    /* SAGE26 satellites keep hot gas; the source sputtered only the central's dust */
    struct params rp;
    struct GALAXY g[2];
    setup(&rp, &g[0], 2);
    setup(&rp, &g[1], 2);
    const float hot_dust0 = g[1].HotDust;
    dust_evolve(1, 0, 1.0 * SEC_PER_MEGAYEAR / 3.08568e19, g, &rp);
    ASSERT_LESS_THAN(g[1].HotDust, hot_dust0, "Satellite hot dust is sputtered");
}

void test_dust_merger_keeps_sfh(void) {
    BEGIN_TEST("add_galaxies_together: the satellite's star-formation history joins the central's");

    /* The merged stars keep returning delayed yields: their history and this substep's stars
     * (not yet in the history; the satellite gets no dust_evolve()) must pass to the central. */
    struct params rp;
    memset(&rp, 0, sizeof(rp));
    rp.DustySAGEMode = 2;
    struct GALAXY g[2];
    memset(g, 0, sizeof(g));
    g[0].DustSfh[10] = 1.0f; g[1].DustSfh[10] = 2.0f; g[1].DustSfh[20] = 3.0f;
    g[0].DustStarsStep = 0.1f; g[1].DustStarsStep = 0.2f; g[1].DustMetalsStep = 0.05f;
    add_galaxies_together(0, 1, g, &rp);
    ASSERT_CLOSE(3.0, g[0].DustSfh[10], DUST_RTOL, "Histories add");
    ASSERT_CLOSE(3.0, g[0].DustSfh[20], DUST_RTOL, "Satellite-only snapshots carried");
    ASSERT_CLOSE(0.3, g[0].DustStarsStep, DUST_RTOL, "This substep's stars carried");
    ASSERT_CLOSE(0.05, g[0].DustMetalsStep, DUST_RTOL, "This substep's SAGE26 metals carried");
}

void test_dust_locked_reported(void) {
    BEGIN_TEST("update_from_star_formation: reports the dust it locks into stellar metals");

    /* In mode 2 the stars' share of the cold dust joins MetalsStellarMass; the caller must know
     * how much, to add it where the stars go (the bulge in a burst) and to the metal histories.
     * In mode 1 the dust is already in the metals, so none is added. */
    for(int mode = 1; mode <= 2; mode++) {
        struct params rp;
        memset(&rp, 0, sizeof(rp));
        rp.DustySAGEMode = mode;
        rp.RecycleFraction = 0.5;
        struct GALAXY g[1];
        memset(g, 0, sizeof(g));
        g[0].ColdGas = 1.0f; g[0].MetalsColdGas = 0.02f; g[0].ColdDust = 0.01f;
        const double locked = update_from_star_formation(0, 0.2, 0.02, g, &rp);
        if(mode == 2) {
            ASSERT_CLOSE(0.001, locked, DUST_RTOL, "Mode 2: the stars' dust share (dust-to-gas x stars kept)");
            ASSERT_CLOSE(0.002 + 0.001, g[0].MetalsStellarMass, DUST_RTOL, "Mode 2: in the stellar metals");
        } else {
            ASSERT_TRUE(locked == 0.0, "Mode 1: no dust added to the stellar metals");
            ASSERT_CLOSE(0.002, g[0].MetalsStellarMass, DUST_RTOL, "Mode 1: stellar metals from the gas metallicity");
        }
    }
}

/* Two-annulus disc for the DarkSAGEMode dust tests: equal gas, metals and dust, but all the H2
 * and all the star formation in annulus 0 */
static void setup_annuli(struct params *rp, struct GALAXY *g, const int dark, const int dusty)
{
    setup(rp, g, dusty);
    rp->DarkSAGEMode = dark;
    g->HotGas = 0.0f; g->HotDust = 0.0f; g->MetalsHotGas = 0.0f;
    for(int i = 0; i < 2; i++) {
        g->DiscGas[i] = 0.5f; g->DiscGasMetals[i] = 0.0075f; g->DiscDust[i] = 0.001f;
        g->DiscStars[i] = 1.0f;
    }
    g->DiscH2[0] = 0.3f;
    g->DiscSFR[0] = 1.0f;
}

void test_dust_annuli_follow_gas(void) {
    BEGIN_TEST("darksage_remove_disc_gas / sync: annulus dust leaves with the gas and sums to ColdDust");

    /* Removing half an annulus's gas takes half its dust; the sync rescales the annuli to
     * ColdDust, keeping the profile. */
    struct params rp;
    struct GALAXY g[1];
    setup_annuli(&rp, g, 1, 2);
    double w[N_BINS] = {0.0};
    w[0] = 1.0;
    darksage_remove_disc_gas(0, 0.25, w, NULL, g);
    ASSERT_CLOSE(0.0005, g[0].DiscDust[0], DUST_RTOL, "Half the gas leaves annulus 0 with half its dust");
    ASSERT_CLOSE(0.001, g[0].DiscDust[1], DUST_RTOL, "Annulus 1 untouched");
    g[0].ColdDust = 0.003f;
    darksage_sync_disc_dust(0, g);
    ASSERT_CLOSE(0.001, g[0].DiscDust[0], DUST_RTOL, "Sync rescales to ColdDust ...");
    ASSERT_CLOSE(0.002, g[0].DiscDust[1], DUST_RTOL, "... keeping the profile");
}

void test_dust_annuli_physics(void) {
    BEGIN_TEST("dust_evolve (DarkSAGEMode): local growth and destruction, annuli sum to ColdDust");

    /* Growth needs H2 and SN destruction needs star formation: both act in annulus 0 only.
     * DarkSAGEMode 1-2 keep SAGE26's galaxy-wide totals (as for the bulk SF recipes); mode 3 sums
     * the annuli.  Either way the annuli must sum to ColdDust, and in DustySAGEMode 2 metals +
     * dust are conserved. */
    const double dt = 1.0 * SEC_PER_MEGAYEAR / 3.08568e19;
    for(int dark = 1; dark <= 3; dark++) {
        struct params rp;
        struct GALAXY g[1], bulk[1];
        setup_annuli(&rp, g, dark, 2);
        g[0].DustStarsStep = 0.0f;   /* growth only */
        const double before = g[0].MetalsColdGas + g[0].ColdDust;
        dust_evolve(0, 0, dt, g, &rp);
        const double sum = g[0].DiscDust[0] + g[0].DiscDust[1];
        ASSERT_CLOSE(g[0].ColdDust, sum, DUST_RTOL, "Annuli sum to ColdDust");
        ASSERT_GREATER_THAN(g[0].DiscDust[0], 0.001, "Dust grows where the H2 is");
        ASSERT_CLOSE(0.001, g[0].DiscDust[1], DUST_RTOL, "No growth without H2");
        ASSERT_CLOSE(before, g[0].MetalsColdGas + g[0].ColdDust, DUST_RTOL, "Metals + dust conserved");
        if(dark <= 2) {
            setup_annuli(&rp, bulk, 0, 2);
            bulk[0].DustStarsStep = 0.0f;
            dust_evolve(0, 0, dt, bulk, &rp);
            ASSERT_CLOSE(bulk[0].ColdDust, g[0].ColdDust, DUST_RTOL, "DarkSAGEMode 1-2: the galaxy-wide growth");
        }

        /* Unequal dust: mode 3's growth is annulus 0's own, that of a galaxy holding just its gas */
        setup_annuli(&rp, g, dark, 2);
        g[0].DustStarsStep = 0.0f;
        g[0].DiscDust[0] = 0.0015f; g[0].DiscDust[1] = 0.0005f;
        dust_evolve(0, 0, dt, g, &rp);
        if(dark == 3) {
            setup(&rp, bulk, 2);
            bulk[0].HotGas = 0.0f; bulk[0].HotDust = 0.0f; bulk[0].MetalsHotGas = 0.0f;
            bulk[0].ColdGas = 0.5f; bulk[0].MetalsColdGas = 0.0075f; bulk[0].ColdDust = 0.0015f; bulk[0].H2gas = 0.3f;
            bulk[0].DustStarsStep = 0.0f;
            dust_evolve(0, 0, dt, bulk, &rp);
            ASSERT_CLOSE(bulk[0].ColdDust - 0.0015, g[0].ColdDust - 0.002, 1e-4, "DarkSAGEMode 3: the annuli's own growth");
        }

        /* Annuli not yet carrying ColdDust (all zero) take its profile first, then grow */
        setup_annuli(&rp, g, dark, 2);
        g[0].DustStarsStep = 0.0f;
        g[0].DiscDust[0] = g[0].DiscDust[1] = 0.0f;
        dust_evolve(0, 0, dt, g, &rp);
        ASSERT_GREATER_THAN(g[0].DiscDust[0], g[0].DiscDust[1], "Annuli synced to ColdDust before the physics: growth where the H2 is");

        setup_annuli(&rp, g, dark, 2);
        g[0].H2gas = 0.0f; g[0].DiscH2[0] = 0.0f;   /* destruction only */
        dust_evolve(0, 0, dt, g, &rp);
        ASSERT_LESS_THAN(g[0].DiscDust[0], g[0].DiscDust[1], "SNe destroy dust where the stars form");
        ASSERT_CLOSE(g[0].ColdDust, g[0].DiscDust[0] + g[0].DiscDust[1], DUST_RTOL, "Annuli sum to ColdDust");
    }
}

void test_dust_zero_dt_resets(void) {
    BEGIN_TEST("dust_evolve: a zero-length substep still clears the substep's star and metal counters");

    /* Otherwise they carry into the next substep, inflating its SN rate and mode-1 cap */
    struct params rp;
    struct GALAXY g[1];
    setup(&rp, g, 2);
    g[0].DustMetalsStep = 0.001f;
    dust_evolve(0, 0, 0.0, g, &rp);
    ASSERT_TRUE(g[0].DustStarsStep == 0.0f && g[0].DustMetalsStep == 0.0f, "Counters cleared");
}

/* The galaxy of setup() in physical units [1e10 Msun, Mpc, Myr] expressed in code units for h */
static void setup_physical(struct params *rp, struct GALAXY *g, const double h)
{
    setup(rp, g, 2);
    rp->Hubble_h = h;
    g->ColdGas = (float)(1.0 * h); g->MetalsColdGas = (float)(0.015 * h); g->ColdDust = (float)(0.002 * h);
    g->H2gas = (float)(0.3 * h);
    g->HotGas = (float)(5.0 * h); g->MetalsHotGas = (float)(0.02 * h); g->HotDust = (float)(0.001 * h);
    g->Rvir = (float)(0.2 * h);
    g->DustStarsStep = (float)(0.01 * h);
}

void test_dust_h_invariant(void) {
    BEGIN_TEST("dust_evolve: grain growth, SN destruction and sputtering independent of h");

    /* The same galaxy in physical units, written in code units (1e10 Msun/h, Mpc/h, code time
     * (Mpc/h)/(km/s)) for two values of h, must lose and grow the same physical dust.  The source
     * left 1/h out of every dust timescale and h^2 out of the sputtering density. */
    double cold[2], hot[2];
    const double hs[2] = {0.73, 1.0};
    for(int k = 0; k < 2; k++) {
        struct params rp;
        struct GALAXY g[1];
        setup_physical(&rp, g, hs[k]);
        const double dt = 1.0 * SEC_PER_MEGAYEAR * hs[k] / rp.UnitTime_in_s;   /* 1 Myr */
        for(int n = 0; n < 50; n++) {
            g[0].DustStarsStep = (float)(0.01 * hs[k]);
            dust_evolve(0, 0, dt, g, &rp);
        }
        cold[k] = g[0].ColdDust / hs[k];
        hot[k] = g[0].HotDust / hs[k];
    }
    ASSERT_CLOSE(cold[1], cold[0], 1e-4, "Cold dust (growth, SN destruction) the same for h = 0.73 and 1");
    ASSERT_CLOSE(hot[1], hot[0], 1e-4, "Hot dust (sputtering) the same for h = 0.73 and 1");
}

void test_dust_current_sfr(void) {
    BEGIN_TEST("dust_production: yields from the current snapshot's mean SFR so far, not a partial sum");

    /* Constant SFR all along: the first and the last substep of a snapshot must make the same
     * yields.  The snapshot's SFH entry is a sum over its substeps, so the first substep holds
     * only 1/N of the SFR that prompt SNII progenitors (< 40 Myr) were born with. */
    double made[2];
    for(int k = 0; k < 2; k++) {
        struct params rp;
        struct GALAXY g[1];
        setup(&rp, g, 2);
        rp.Snaplistlen = 2;
        double age[3] = {0.0, 1.0, 0.0};
        rp.Age = age + 1;
        dust_init(&rp);
        for(int i = 0; i < ABSOLUTEMAXSNAPS; i++) rp.DustTime[i] = 100.0 * i;   /* Myr */
        const double dt = 10.0 * SEC_PER_MEGAYEAR * rp.Hubble_h / rp.UnitTime_in_s;   /* 10 substeps */
        for(int i = 0; i < 50; i++) g[0].DustSfh[i] = 1.0f;
        g[0].H2gas = 0.0f; g[0].HotGas = 0.0f; g[0].HotDust = 0.0f;
        g[0].DustStarsStep = (float)(1.0 * dt);   /* SFR 1 */
        g[0].DustSfh[50] = (k == 0) ? 0.0f : 0.9f;   /* first, or last, of the 10 substeps */
        g[0].DustSubsteps = (k == 0) ? 0 : 9;
        dust_evolve(0, 0, dt, g, &rp);
        made[k] = g[0].DustFormationRate;
    }
    ASSERT_GREATER_THAN(made[1], 0.0, "Yields make dust");
    ASSERT_CLOSE(made[1], made[0], 1e-3, "First substep makes what the last does");
}

int main(void) {
    BEGIN_TEST_SUITE("DustySAGEMode dust bookkeeping");

    test_dust_moves_with_gas();
    test_dust_follows_regime();
    test_dust_evolve_conserves();
    test_dust_time_physical();
    test_dust_growth_past_half();
    test_dust_sputters_satellites();
    test_dust_merger_keeps_sfh();
    test_dust_locked_reported();
    test_dust_annuli_follow_gas();
    test_dust_annuli_physics();
    test_dust_zero_dt_resets();
    test_dust_h_invariant();
    test_dust_current_sfr();

    END_TEST_SUITE();
    PRINT_TEST_SUMMARY();

    return TEST_EXIT_CODE();
}

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

    END_TEST_SUITE();
    PRINT_TEST_SUMMARY();

    return TEST_EXIT_CODE();
}

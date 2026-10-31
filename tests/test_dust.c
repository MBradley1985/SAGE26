/*
 * DUSTYSAGEMODE DUST BOOKKEEPING TESTS
 *
 * Validates: dust moves with gas at the source reservoir's dust-to-gas ratio and is conserved
 * (dust_move()), never more than the reservoir holds; the hot-phase deposit follows the regime
 * the gas goes to (add_dust_to_hot_reservoir()); an empty reservoir moves no dust; and dust_evolve()
 * without star formation conserves metals + dust in mode 2, leaves the metals alone and keeps
 * dust below them in mode 1, and sputters hot-halo dust.
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

int main(void) {
    BEGIN_TEST_SUITE("DustySAGEMode dust bookkeeping");

    test_dust_moves_with_gas();
    test_dust_follows_regime();
    test_dust_evolve_conserves();

    END_TEST_SUITE();
    PRINT_TEST_SUMMARY();

    return TEST_EXIT_CODE();
}

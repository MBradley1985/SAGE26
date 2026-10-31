/*
 * DUSTYSAGEMODE DUST BOOKKEEPING TESTS
 *
 * Validates: dust moves with gas at the source reservoir's dust-to-gas ratio and is conserved
 * (dust_move()), never more than the reservoir holds; the hot-phase deposit follows the regime
 * the gas goes to (add_dust_to_hot_reservoir()); and an empty reservoir moves no dust.
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

int main(void) {
    BEGIN_TEST_SUITE("DustySAGEMode dust bookkeeping");

    test_dust_moves_with_gas();
    test_dust_follows_regime();

    END_TEST_SUITE();
    PRINT_TEST_SUMMARY();

    return TEST_EXIT_CODE();
}

/*
 * test_let_orphans_live.c -- unit tests for the Contini et al. (2014) orphan
 * treatment selected by LetOrphansLive == 1 (their model Disr., Sec. 3.1).
 *
 * Validates: survival of a compact satellite whose mean baryon density beats
 * the halo density at pericentre, complete disruption of a diffuse satellite on
 * a plunging orbit, and the fallback to unconditional disruption for satellites
 * that carry no baryons.
 *
 * Their model Tid. (Sec. 3.2) was implemented and removed -- see
 * model_mergers.h for why -- so it is no longer covered here.
 *
 * Run: from tests/, `make test_let_orphans_live && ./test_let_orphans_live`.
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "test_framework.h"

#include "../src/core_allvars.h"
#include "../src/model_mergers.h"

/* Gravitational constant in SAGE code units (10^10 Msun/h, Mpc/h, km/s). */
static const double G_CODE = 43007.1;

/* The GALAXY reservoirs and radii are stored as float, so comparisons are held
 * to single precision rather than to the double precision of the arithmetic. */
static const double MASS_TOL = 1.0e-6;

/*
 * Build a two-galaxy system: index 0 is the central that supplies the
 * potential and receives stripped material, index 1 the orphan under test.
 * The caller adjusts the satellite fields that each test is about.
 */
static void setup_pair(struct GALAXY *galaxies, struct params *run_params,
                       const double Mvir, const double Vvir)
{
    memset(galaxies, 0, 2 * sizeof(struct GALAXY));
    memset(run_params, 0, sizeof(struct params));

    run_params->G = G_CODE;
    run_params->BoxSize = 62.5;          /* mini-Millennium, Mpc/h */
    run_params->TrackICSAssembly = 1;
    run_params->CGMrecipeOn = 0;         /* stripped cold gas lands in HotGas */

    galaxies[0].Type = 0;
    galaxies[0].Mvir = Mvir;
    galaxies[0].Vvir = Vvir;
    galaxies[0].Regime = 1;

    galaxies[1].Type = 2;
    galaxies[1].CentralGal = 0;
    galaxies[1].mergeType = 0;
}

/*
 * A satellite that is dense compared with the halo at its pericentre must be
 * left alone -- this is the whole point of the gate, and the behaviour that
 * removes the excess intracluster light.
 */
void test_dense_satellite_survives(void)
{
    BEGIN_TEST("Dense satellite survives the density gate");

    struct GALAXY galaxies[2];
    struct params run_params;
    setup_pair(galaxies, &run_params, 100.0, 200.0);

    /* Compact disk galaxy 0.5 Mpc/h out on a near-circular orbit, so the
     * pericentre stays far from the halo centre. */
    galaxies[1].Pos[0] = 0.5;
    galaxies[1].OrbitRadius = 0.5;
    galaxies[1].Vel[1] = 200.0;
    galaxies[1].StellarMass = 0.1;
    galaxies[1].ColdGas = 0.02;
    galaxies[1].DiskScaleRadius = 0.005;

    const double mstar_before = galaxies[1].StellarMass;

    const int destroyed = contini14_disruption_model(0, 0, 1, 1.0, galaxies, &run_params);

    ASSERT_EQUAL_INT(0, destroyed, "Gate stays shut: satellite is not destroyed");
    ASSERT_EQUAL_INT(0, galaxies[1].mergeType, "mergeType is untouched for a survivor");
    ASSERT_CLOSE(mstar_before, galaxies[1].StellarMass, MASS_TOL,
                 "A surviving satellite keeps all of its stars");
    ASSERT_EQUAL_FLOAT(0.0, galaxies[0].ICS,
                       "Nothing is added to the central's ICS");

}

/*
 * A diffuse satellite driven onto a radial orbit reaches a pericentre where the
 * halo is denser than the satellite, so the gate opens and the tidal radius cuts
 * inside the bulge -- complete disruption, exactly as in the ungated model.
 */
void test_diffuse_satellite_disrupted(void)
{
    BEGIN_TEST("Diffuse satellite on a plunging orbit is disrupted");

    struct GALAXY galaxies[2];
    struct params run_params;
    setup_pair(galaxies, &run_params, 1000.0, 1000.0);

    /* Radial infall: no tangential velocity means no turning point. */
    galaxies[1].Pos[0] = 0.05;
    galaxies[1].OrbitRadius = 0.05;
    galaxies[1].Vel[0] = -500.0;
    galaxies[1].StellarMass = 0.01;
    galaxies[1].MetalsStellarMass = 0.0002;
    galaxies[1].ColdGas = 0.005;
    galaxies[1].DiskScaleRadius = 0.05;   /* very extended, hence low density */
    galaxies[1].BulgeRadius = 0.04;

    const double stars_before = galaxies[1].StellarMass;

    const int destroyed = contini14_disruption_model(0, 0, 1, 1.0, galaxies, &run_params);

    ASSERT_EQUAL_INT(1, destroyed, "Gate opens: satellite is destroyed");
    ASSERT_EQUAL_INT(4, galaxies[1].mergeType, "mergeType records a disruption to the ICS");
    ASSERT_EQUAL_FLOAT(0.0, galaxies[1].StellarMass, "Satellite stars are all removed");
    ASSERT_CLOSE(stars_before, galaxies[0].ICS, MASS_TOL,
                 "Every disrupted star lands in the central's ICS");

}


/*
 * An orphan with no baryons has no density to compare, and must not be allowed
 * to orbit for ever. It falls through to the ungated disruption path.
 */
void test_empty_satellite_falls_back(void)
{
    BEGIN_TEST("Baryon-free satellite falls back to unconditional disruption");

    struct GALAXY galaxies[2];
    struct params run_params;
    setup_pair(galaxies, &run_params, 100.0, 200.0);

    galaxies[1].Pos[0] = 0.5;
    galaxies[1].OrbitRadius = 0.5;
    galaxies[1].Vel[1] = 200.0;
    galaxies[1].StellarMass = 0.0;
    galaxies[1].ColdGas = 0.0;
    galaxies[1].DiskScaleRadius = 0.005;

    const int destroyed = contini14_disruption_model(0, 0, 1, 1.0, galaxies, &run_params);

    ASSERT_EQUAL_INT(1, destroyed, "An empty satellite is removed rather than kept");
    ASSERT_EQUAL_INT(4, galaxies[1].mergeType, "mergeType records the disruption");

}


int main(void)
{
    BEGIN_TEST_SUITE("Contini+14 model Disr. (LetOrphansLive)");

    test_dense_satellite_survives();
    test_diffuse_satellite_disrupted();
    test_empty_satellite_falls_back();

    END_TEST_SUITE();
    PRINT_TEST_SUMMARY();

    return TEST_EXIT_CODE();
}

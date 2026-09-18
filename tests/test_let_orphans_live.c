/*
 * test_let_orphans_live.c -- unit tests for the Contini et al. (2014) orphan
 * disruption gate (LetOrphansLive == 1).
 *
 * Validates: survival of a compact satellite whose mean baryon density exceeds
 * the halo density at pericentre, complete disruption of a diffuse satellite on
 * a plunging orbit, mass and metal conservation during partial tidal stripping,
 * the disk scalelength reset that follows a stripping episode, and the
 * fallback to unconditional disruption for satellites that carry no baryons,
 * and the Henriques & Thomas (2010) continuous tidal stripping of orphans.
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
    galaxies[1].Vel[1] = 200.0;
    galaxies[1].StellarMass = 0.1;
    galaxies[1].ColdGas = 0.02;
    galaxies[1].DiskScaleRadius = 0.005;

    const double mstar_before = galaxies[1].StellarMass;

    const int destroyed = disrupt_satellite_gated(0, 0, 1, 1.0, galaxies, &run_params);

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
    galaxies[1].Vel[0] = -500.0;
    galaxies[1].StellarMass = 0.01;
    galaxies[1].MetalsStellarMass = 0.0002;
    galaxies[1].ColdGas = 0.005;
    galaxies[1].DiskScaleRadius = 0.05;   /* very extended, hence low density */
    galaxies[1].BulgeRadius = 0.04;

    const double stars_before = galaxies[1].StellarMass;

    const int destroyed = disrupt_satellite_gated(0, 0, 1, 1.0, galaxies, &run_params);

    ASSERT_EQUAL_INT(1, destroyed, "Gate opens: satellite is destroyed");
    ASSERT_EQUAL_INT(4, galaxies[1].mergeType, "mergeType records a disruption to the ICS");
    ASSERT_EQUAL_FLOAT(0.0, galaxies[1].StellarMass, "Satellite stars are all removed");
    ASSERT_CLOSE(stars_before, galaxies[0].ICS, MASS_TOL,
                 "Every disrupted star lands in the central's ICS");

}

/*
 * When the gate opens but the tidal radius still lies outside the bulge, only
 * the stellar disk beyond R_t is unbound. Stars must be conserved between the
 * satellite and the central's ICS, and cold gas must follow in proportion.
 */
void test_partial_stripping_conserves_mass(void)
{
    BEGIN_TEST("Partial stripping conserves stars, metals and cold gas");

    struct GALAXY galaxies[2];
    struct params run_params;
    setup_pair(galaxies, &run_params, 1000.0, 1000.0);

    /* A real pericentre needs angular momentum; a purely radial orbit takes
     * the plunge branch and is destroyed outright. */
    galaxies[1].Pos[0] = 0.05;
    galaxies[1].Vel[0] = -500.0;
    galaxies[1].Vel[1] = 200.0;
    galaxies[1].StellarMass = 0.01;
    galaxies[1].MetalsStellarMass = 0.0002;
    galaxies[1].ColdGas = 0.005;
    galaxies[1].MetalsColdGas = 0.0001;
    galaxies[1].DiskScaleRadius = 0.05;
    galaxies[1].BulgeRadius = 0.0;        /* pure disk: no complete disruption */

    const double stars_before = galaxies[1].StellarMass;
    const double metals_before = galaxies[1].MetalsStellarMass;
    const double cold_before = galaxies[1].ColdGas;

    const int destroyed = disrupt_satellite_gated(0, 0, 1, 1.0, galaxies, &run_params);

    ASSERT_EQUAL_INT(0, destroyed, "A partially stripped satellite survives");
    ASSERT_GREATER_THAN(galaxies[0].ICS, 0.0, "Stripped stars reach the ICS");
    ASSERT_LESS_THAN(galaxies[1].StellarMass, stars_before,
                     "The satellite loses stellar mass");
    ASSERT_CLOSE(stars_before, galaxies[1].StellarMass + galaxies[0].ICS, MASS_TOL,
                 "Stellar mass is conserved across the transfer");
    ASSERT_CLOSE(metals_before, galaxies[1].MetalsStellarMass + galaxies[0].MetalsICS, MASS_TOL,
                 "Stellar metals are conserved across the transfer");
    ASSERT_CLOSE(cold_before, galaxies[1].ColdGas + galaxies[0].HotGas, MASS_TOL,
                 "Cold gas stripped from the disk reaches the central's hot phase");
    ASSERT_CLOSE(galaxies[0].ICS, galaxies[0].ICS_disrupt, MASS_TOL,
                 "ICS assembly tracking records the stripped mass");

}

/*
 * SAGE26 uses DiskScaleRadius directly as the exponential scale length behind
 * Sigma_0 = M / (2 pi r_s^2) in the H2 and star formation prescriptions, so a
 * galaxy that has just lost its outskirts must not come back with a *smaller*
 * scale length and hence a higher central surface density. The scalelength is
 * therefore left alone, deviating from the R_t/10 reset of Contini et al.
 */
void test_scalelength_preserved(void)
{
    BEGIN_TEST("Partial stripping lowers the disk mass without compressing it");

    struct GALAXY galaxies[2];
    struct params run_params;
    setup_pair(galaxies, &run_params, 1000.0, 1000.0);

    galaxies[1].Pos[0] = 0.05;
    galaxies[1].Vel[0] = -500.0;
    galaxies[1].Vel[1] = 200.0;
    galaxies[1].StellarMass = 0.01;
    galaxies[1].ColdGas = 0.005;
    galaxies[1].DiskScaleRadius = 0.05;
    galaxies[1].BulgeRadius = 0.0;

    const double r_scale_before = galaxies[1].DiskScaleRadius;
    const double stars_before = galaxies[1].StellarMass;

    disrupt_satellite_gated(0, 0, 1, 1.0, galaxies, &run_params);

    ASSERT_CLOSE(r_scale_before, galaxies[1].DiskScaleRadius, MASS_TOL,
                 "Scale length is unchanged by stripping");
    ASSERT_LESS_THAN(galaxies[1].StellarMass, stars_before,
                     "Disk mass is reduced, so Sigma_0 falls rather than rises");
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
    galaxies[1].Vel[1] = 200.0;
    galaxies[1].StellarMass = 0.0;
    galaxies[1].ColdGas = 0.0;
    galaxies[1].DiskScaleRadius = 0.005;

    const int destroyed = disrupt_satellite_gated(0, 0, 1, 1.0, galaxies, &run_params);

    ASSERT_EQUAL_INT(1, destroyed, "An empty satellite is removed rather than kept");
    ASSERT_EQUAL_INT(4, galaxies[1].mergeType, "mergeType records the disruption");

}

/*
 * Henriques & Thomas (2010) strip the stellar material lying outside the tidal
 * radius on every timestep. A satellite whose orbit has decayed close in loses
 * most of its disk; the stars must arrive intact in the central's ICS.
 */
void test_continuous_stripping_conserves_stars(void)
{
    BEGIN_TEST("Continuous stripping moves disk stars into the ICS");

    struct GALAXY galaxies[2];
    struct params run_params;
    setup_pair(galaxies, &run_params, 1000.0, 1000.0);

    galaxies[1].Vvir = 100.0;
    galaxies[1].OrbitRadius = 0.1;
    galaxies[1].StellarMass = 0.01;
    galaxies[1].MetalsStellarMass = 0.0002;
    galaxies[1].DiskScaleRadius = 0.005;

    const double stars_before = galaxies[1].StellarMass;
    const double metals_before = galaxies[1].MetalsStellarMass;

    /* R_t = (1/sqrt(2)) (Vvir_sat/Vvir_halo) r_orbit; the exponential disk
     * beyond it is M (1 + x) exp(-x) with x = R_t / R_sl. */
    const double R_t = M_SQRT1_2 * (100.0 / 1000.0) * 0.1;
    const double x = R_t / 0.005;
    const double expected = stars_before * (1.0 + x) * exp(-x);

    strip_orphan_stars(0, 0, 1, 1.0, galaxies, &run_params);

    ASSERT_CLOSE(expected, galaxies[0].ICS, MASS_TOL,
                 "Stripped mass matches the exponential-disk integral");
    ASSERT_CLOSE(stars_before, galaxies[1].StellarMass + galaxies[0].ICS, MASS_TOL,
                 "Stellar mass is conserved");
    ASSERT_CLOSE(metals_before, galaxies[1].MetalsStellarMass + galaxies[0].MetalsICS, MASS_TOL,
                 "Metals leave in the same proportion as stars");
    ASSERT_GREATER_THAN(galaxies[1].StellarMass, 0.0, "The satellite is not destroyed");
}

/*
 * A satellite whose tidal radius sits far outside its own disk keeps its stars:
 * the stripping term must vanish rather than nibble at every timestep.
 */
void test_wide_orbit_strips_nothing(void)
{
    BEGIN_TEST("A satellite well inside its tidal radius is left alone");

    struct GALAXY galaxies[2];
    struct params run_params;
    setup_pair(galaxies, &run_params, 1000.0, 1000.0);

    galaxies[1].Vvir = 500.0;
    galaxies[1].OrbitRadius = 1.0;     /* R_t ends up ~70 disk scalelengths out */
    galaxies[1].StellarMass = 0.01;
    galaxies[1].DiskScaleRadius = 0.005;

    const double stars_before = galaxies[1].StellarMass;

    strip_orphan_stars(0, 0, 1, 1.0, galaxies, &run_params);

    ASSERT_CLOSE(stars_before, galaxies[1].StellarMass, MASS_TOL,
                 "Stellar mass is unchanged");
    /* The exponential tail is not identically zero, just negligible, so compare
     * against an absolute floor rather than a relative tolerance about zero. */
    ASSERT_LESS_THAN(galaxies[0].ICS, MASS_TOL, "Nothing measurable reaches the ICS");
}

/*
 * The bulge is stripped on its own profile, and the two Tonini bulge components
 * have to shrink with the total so they stay a partition of it.
 */
void test_bulge_stripping_keeps_components_consistent(void)
{
    BEGIN_TEST("Bulge stripping scales the Tonini components with the total");

    struct GALAXY galaxies[2];
    struct params run_params;
    setup_pair(galaxies, &run_params, 1000.0, 1000.0);

    galaxies[1].Vvir = 100.0;
    galaxies[1].OrbitRadius = 0.1;
    galaxies[1].StellarMass = 0.01;
    galaxies[1].MetalsStellarMass = 0.0002;
    galaxies[1].DiskScaleRadius = 0.005;
    galaxies[1].BulgeMass = 0.002;
    galaxies[1].MetalsBulgeMass = 0.00004;
    galaxies[1].BulgeRadius = 0.004;
    galaxies[1].MergerBulgeMass = 0.0015;
    galaxies[1].InstabilityBulgeMass = 0.0005;

    strip_orphan_stars(0, 0, 1, 1.0, galaxies, &run_params);

    ASSERT_LESS_THAN(galaxies[1].BulgeMass, 0.002, "The bulge loses mass");
    ASSERT_CLOSE(galaxies[1].BulgeMass,
                 galaxies[1].MergerBulgeMass + galaxies[1].InstabilityBulgeMass, MASS_TOL,
                 "Merger and instability components still sum to the bulge");
    ASSERT_LESS_THAN(galaxies[1].BulgeMass, galaxies[1].StellarMass,
                     "The bulge stays a part of the stellar mass");
}

/*
 * With no orbit recorded there is no tidal radius to compute, so an orphan that
 * predates the toggle being switched on must simply be left alone.
 */
void test_no_orbit_strips_nothing(void)
{
    BEGIN_TEST("No recorded orbit means no stripping");

    struct GALAXY galaxies[2];
    struct params run_params;
    setup_pair(galaxies, &run_params, 1000.0, 1000.0);

    galaxies[1].Vvir = 100.0;
    galaxies[1].OrbitRadius = 0.0;
    galaxies[1].StellarMass = 0.01;
    galaxies[1].DiskScaleRadius = 0.005;

    strip_orphan_stars(0, 0, 1, 1.0, galaxies, &run_params);

    ASSERT_CLOSE(0.01, galaxies[1].StellarMass, MASS_TOL, "Stellar mass is unchanged");
    ASSERT_CLOSE(0.0, galaxies[0].ICS, MASS_TOL, "Nothing is added to the ICS");
}

int main(void)
{
    BEGIN_TEST_SUITE("Contini+14 Orphan Disruption Gate");

    test_dense_satellite_survives();
    test_diffuse_satellite_disrupted();
    test_partial_stripping_conserves_mass();
    test_scalelength_preserved();
    test_empty_satellite_falls_back();
    test_continuous_stripping_conserves_stars();
    test_wide_orbit_strips_nothing();
    test_bulge_stripping_keeps_components_consistent();
    test_no_orbit_strips_nothing();

    END_TEST_SUITE();
    PRINT_TEST_SUMMARY();

    return TEST_EXIT_CODE();
}

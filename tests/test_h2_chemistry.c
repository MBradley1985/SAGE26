/*
 * H2 CHEMISTRY AND GAS-PHASE PARTITION TESTS
 *
 * Tests for the molecular gas fraction prescriptions and the HI/H2/ionised
 * partition of the cold gas:
 * - BR06 pressure-based f_H2 (bounds, limits, power-law index, stellar floor)
 * - KD12 / K13 / GD14 shielding-based f_H2 (bounds, metallicity guards)
 * - K13 depletion time (positivity, two-phase term)
 * - Radial integration over the exponential disk (hydrogen budget invariant)
 * - HI bookkeeping through starformation_and_feedback():
 *     H1 = (1 - f_ion) * (X_H * ColdGas - H2)  >= 0 by construction
 *
 * Reference values pin the frozen single-precision behaviour of the fits
 * (see docs/physics/units.md); they were generated with the unit-test build
 * flags (-O2, no -march=native) and use loose-enough tolerances to absorb
 * libm differences across platforms.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "test_framework.h"
#include "../src/core_allvars.h"
#include "../src/model_misc.h"
#include "../src/model_starformation_and_feedback.h"

/* Tolerance for pinned float-precision reference values: tight enough to
 * catch any real change to the fits, loose enough for cross-platform libm. */
#define GOLDEN_RTOL 1e-4

void test_br06_bounds_and_limits() {
    BEGIN_TEST("BR06: bounds, zero-density and zero-radius guards");

    ASSERT_EQUAL_FLOAT(calculate_molecular_fraction_BR06(0.0f, 100.0f, 3000.0f), 0.0,
                       "Zero gas surface density gives f_H2 = 0");
    ASSERT_EQUAL_FLOAT(calculate_molecular_fraction_BR06(10.0f, 100.0f, 0.0f), 0.0,
                       "Zero disk scale length gives f_H2 = 0");
    ASSERT_EQUAL_FLOAT(calculate_molecular_fraction_BR06(10.0f, 100.0f, -1.0f), 0.0,
                       "Negative disk scale length gives f_H2 = 0");

    /* extreme surface densities stay in [0, 1] */
    float f_lo = calculate_molecular_fraction_BR06(1e-5f, 1e-5f, 3000.0f);
    float f_hi = calculate_molecular_fraction_BR06(1e5f, 1e5f, 3000.0f);
    ASSERT_IN_RANGE(f_lo, 0.0, 1.0, "f_H2 in [0,1] at very low surface density");
    ASSERT_IN_RANGE(f_hi, 0.0, 1.0, "f_H2 in [0,1] at very high surface density");
    ASSERT_GREATER_THAN(f_hi, 0.99, "f_H2 -> 1 in the high-pressure limit");
}

void test_br06_power_law_index() {
    BEGIN_TEST("BR06: R_mol follows the eq. 13 power law (alpha = 0.92)");

    /* R_mol = f/(1-f) = (P/P0)^0.92 and P is linear in Sigma_gas, so
     * raising Sigma_gas x10 at fixed Sigma_star and r_s must raise
     * R_mol by 10^0.92. This pins the published fit index without
     * reimplementing the pressure formula. */
    float f1 = calculate_molecular_fraction_BR06(5.0f, 100.0f, 3000.0f);
    float f2 = calculate_molecular_fraction_BR06(50.0f, 100.0f, 3000.0f);
    double R1 = f1 / (1.0 - f1);
    double R2 = f2 / (1.0 - f2);
    ASSERT_CLOSE(pow(10.0, 0.92), R2 / R1, 1e-3,
                 "R_mol(10 Sigma) / R_mol(Sigma) = 10^0.92");

    ASSERT_GREATER_THAN(f2, f1, "f_H2 increases with Sigma_gas");
}

void test_br06_stellar_floor() {
    BEGIN_TEST("BR06: stellar surface density floor at 0.1 Msun/pc^2");

    /* below the floor the effective Sigma_star is fixed at 0.1, so any two
     * values under it must give bit-identical results */
    float f_under = calculate_molecular_fraction_BR06(10.0f, 0.05f, 3000.0f);
    float f_floor = calculate_molecular_fraction_BR06(10.0f, 0.1f, 3000.0f);
    ASSERT_TRUE(f_under == f_floor, "Sigma_star below 0.1 clamps to the floor");

    float f_above = calculate_molecular_fraction_BR06(10.0f, 10.0f, 3000.0f);
    ASSERT_GREATER_THAN(f_above, f_floor, "f_H2 increases with Sigma_star above the floor");
}

void test_br06_reference_values() {
    BEGIN_TEST("BR06: pinned reference values (frozen float behaviour)");

    ASSERT_CLOSE(2.208095342e-01, calculate_molecular_fraction_BR06(10.0f, 100.0f, 3000.0f),
                 GOLDEN_RTOL, "f_H2(10, 100, 3000 pc)");
    ASSERT_CLOSE(1.352882572e-02, calculate_molecular_fraction_BR06(1.0f, 10.0f, 2000.0f),
                 GOLDEN_RTOL, "f_H2(1, 10, 2000 pc)");
    ASSERT_CLOSE(8.594519496e-01, calculate_molecular_fraction_BR06(100.0f, 1000.0f, 4000.0f),
                 GOLDEN_RTOL, "f_H2(100, 1000, 4000 pc)");
    ASSERT_CLOSE(3.886122722e-04, calculate_molecular_fraction_BR06(0.3f, 0.05f, 5000.0f),
                 GOLDEN_RTOL, "f_H2(0.3, 0.05, 5000 pc)");
}

void test_kd12_guards_and_reference() {
    BEGIN_TEST("KD12: metallicity guard and pinned reference values");

    ASSERT_EQUAL_FLOAT(calculate_H2_fraction_KD12(10.0f, 0.0f, 5.0f), 0.0,
                       "Primordial gas (Z = 0) gives f_H2 = 0, not 1");
    ASSERT_EQUAL_FLOAT(calculate_H2_fraction_KD12(0.0f, 0.02f, 5.0f), 0.0,
                       "Zero surface density gives f_H2 = 0");

    float f1 = calculate_H2_fraction_KD12(3.0f, 0.02f, 5.0f);
    float f2 = calculate_H2_fraction_KD12(10.0f, 0.02f, 5.0f);
    ASSERT_GREATER_THAN(f2, f1, "f_H2 increases with Sigma_gas");
    ASSERT_IN_RANGE(f1, 0.0, 1.0, "f_H2 in [0,1]");

    ASSERT_CLOSE(7.725965381e-01, calculate_H2_fraction_KD12(10.0f, 0.02f, 5.0f),
                 GOLDEN_RTOL, "f_H2(10, Zsun, c=5)");
    ASSERT_CLOSE(9.018998146e-01, calculate_H2_fraction_KD12(100.0f, 0.004f, 5.0f),
                 GOLDEN_RTOL, "f_H2(100, 0.2 Zsun, c=5)");
    ASSERT_CLOSE(3.559085429e-01, calculate_H2_fraction_KD12(3.0f, 0.02f, 5.0f),
                 GOLDEN_RTOL, "f_H2(3, Zsun, c=5)");
}

void test_k13_floor_and_reference() {
    BEGIN_TEST("K13: Z' floor at 0.01 and pinned reference values");

    ASSERT_EQUAL_FLOAT(calculate_H2_fraction_K13(0.0, 0.014, 5.0), 0.0,
                       "Zero surface density gives f_H2 = 0");

    /* both metallicities are below the Z' = 0.01 floor, so they clamp to
     * the same value and must agree exactly */
    double f_a = calculate_H2_fraction_K13(50.0, 0.014 * 0.005, 5.0);
    double f_b = calculate_H2_fraction_K13(50.0, 0.014 * 0.009, 5.0);
    ASSERT_TRUE(f_a == f_b, "Z' below 0.01 clamps to the floor");

    ASSERT_CLOSE(6.388650441e-01, calculate_H2_fraction_K13(10.0, 0.014, 5.0),
                 GOLDEN_RTOL, "f_H2(10, Zsun, c=5)");
    ASSERT_CLOSE(7.442803629e-01, calculate_H2_fraction_K13(100.0, 0.0014, 5.0),
                 GOLDEN_RTOL, "f_H2(100, 0.1 Zsun, c=5)");
}

void test_gd14_bounds_and_reference() {
    BEGIN_TEST("GD14: bounds, monotonicity and pinned reference values");

    ASSERT_EQUAL_FLOAT(calculate_H2_fraction_GD14(0.0, 0.02, 3000.0), 0.0,
                       "Zero surface density gives f_H2 = 0");

    double f1 = calculate_H2_fraction_GD14(1.0, 0.02, 3000.0);
    double f2 = calculate_H2_fraction_GD14(10.0, 0.02, 3000.0);
    ASSERT_GREATER_THAN(f2, f1, "f_H2 increases with Sigma_gas");
    ASSERT_IN_RANGE(f2, 0.0, 1.0, "f_H2 in [0,1]");

    ASSERT_CLOSE(6.282095313e-01, calculate_H2_fraction_GD14(10.0, 0.02, 3000.0),
                 GOLDEN_RTOL, "f_H2(10, Zsun, 3000 pc)");
    ASSERT_CLOSE(1.159931730e-02, calculate_H2_fraction_GD14(1.0, 0.002, 2000.0),
                 GOLDEN_RTOL, "f_H2(1, 0.1 Zsun, 2000 pc)");
}

void test_tdep_k13() {
    BEGIN_TEST("K13 depletion time: positivity and pinned reference values");

    double t1 = calculate_tdep_K13_Gyr(10.0f, 100.0f, 3000.0f, 1.0f, 0.5f);
    ASSERT_GREATER_THAN(t1, 0.0, "t_dep > 0");

    /* t_dep is the min of the two-phase and hydrostatic branches, so it can
     * never exceed the two-phase term 3.1 / (f_H2 * Sigma^0.25) */
    double t_2p = 3.1 / (0.5 * pow(10.0, 0.25));
    ASSERT_TRUE(t1 <= t_2p + 1e-9, "t_dep <= two-phase depletion time");

    ASSERT_CLOSE(3.486516216e+00, t1, GOLDEN_RTOL, "t_dep(10, 100, 3000, Z'=1, f=0.5)");
    ASSERT_CLOSE(1.089229001e+00, calculate_tdep_K13_Gyr(100.0f, 300.0f, 2000.0f, 0.1f, 0.9f),
                 GOLDEN_RTOL, "t_dep(100, 300, 2000, Z'=0.1, f=0.9)");
}

/* Common setup for the radial-integration and full-driver tests. */
static void setup_br06_radial_params(struct params *rp) {
    memset(rp, 0, sizeof(*rp));
    rp->Hubble_h = 0.73;
    rp->SFprescription = 1;      /* BR06 */
    rp->H2RadialIntegrationOn = 1;
    rp->H2RadialNBins = 25;
    rp->H2RadialRMaxFactor = 5.0;
    rp->SfrEfficiency = 0.05;
    rp->RecycleFraction = 0.43;
}

void test_radial_integration_budget() {
    BEGIN_TEST("Radial integration: H2 never exceeds the hydrogen budget");

    struct params rp;
    setup_br06_radial_params(&rp);

    /* sweep gas mass, disk size, and stellar mass; the integral over
     * f_H2(r) <= 1 truncated at 5 r_s can never claim more hydrogen than
     * X_H * ColdGas, with no cap applied */
    const double coldgas_vals[] = {1e-4, 1e-2, 0.1, 1.0, 10.0};
    const double rs_vals[] = {0.0005, 0.002, 0.008, 0.02};
    const double mstar_vals[] = {0.0, 0.5, 5.0};

    int violations = 0, positive = 0;
    for(size_t i = 0; i < sizeof(coldgas_vals)/sizeof(coldgas_vals[0]); i++) {
        for(size_t j = 0; j < sizeof(rs_vals)/sizeof(rs_vals[0]); j++) {
            for(size_t k = 0; k < sizeof(mstar_vals)/sizeof(mstar_vals[0]); k++) {
                struct GALAXY gal[1];
                memset(gal, 0, sizeof(gal));
                gal[0].ColdGas = coldgas_vals[i];
                gal[0].DiskScaleRadius = rs_vals[j];
                gal[0].StellarMass = mstar_vals[k];
                gal[0].BulgeMass = 0.2 * mstar_vals[k];
                float h2 = calculate_molecular_fraction_radial_integration(0, gal, &rp, NULL);
                if(h2 < 0.0f || h2 > gal[0].ColdGas * HYDROGEN_MASS_FRAC) violations++;
                if(h2 > 0.0f) positive++;
            }
        }
    }
    ASSERT_EQUAL_INT(0, violations, "0 <= H2 <= X_H * ColdGas across the grid (uncapped)");
    ASSERT_GREATER_THAN((double)positive, 0.0, "Grid produces non-trivial H2 masses");
}

void test_radial_integration_reference() {
    BEGIN_TEST("Radial integration: pinned reference value");

    struct params rp;
    setup_br06_radial_params(&rp);

    struct GALAXY gal[1];
    memset(gal, 0, sizeof(gal));
    gal[0].ColdGas = 0.1;            /* 10^9 Msun/h */
    gal[0].StellarMass = 0.5;
    gal[0].BulgeMass = 0.1;
    gal[0].DiskScaleRadius = 0.003;  /* ~3 kpc/h */

    float h2 = calculate_molecular_fraction_radial_integration(0, gal, &rp, NULL);
    ASSERT_CLOSE(2.841610461e-03, h2, GOLDEN_RTOL,
                 "H2(ColdGas=0.1, M*=0.5, MB=0.1, rs=0.003, h=0.73)");
    ASSERT_CLOSE(h2, gal[0].H2gas, 1e-12, "Result stored in H2gas");
}

/* Run the full SF driver once on a fresh galaxy; returns the galaxy state.
 * dt is kept small against tdyn = 3 r_s / Vvir so the step consumes only a
 * sliver of the disk: the post-SF budget clamp in update_from_star_formation
 * (which rescales H1/H2 to the depleted ColdGas, and fires routinely for
 * gas-devouring steps) must not mask the partition being tested. */
static void run_sf_driver(struct GALAXY *gal, const struct params *rp,
                          double coldgas, double rs, double mstar) {
    memset(gal, 0, 2 * sizeof(*gal));
    gal[0].ColdGas = coldgas;
    gal[0].MetalsColdGas = 0.02 * coldgas;
    gal[0].StellarMass = mstar;
    gal[0].BulgeMass = 0.2 * mstar;
    gal[0].DiskScaleRadius = rs;
    gal[0].Vvir = 200.0;
    gal[0].Mvir = 100.0;
    gal[0].SnapNum = 0;
    starformation_and_feedback(0, 0, /*time*/ 1.0, /*dt*/ 1e-6, /*halonr*/ 0, /*step*/ 0,
                               gal, rp);
}

void test_hi_partition_atomic_remainder() {
    BEGIN_TEST("HI partition: H1 is the atomic remainder reduced by the (always-on) ionisation cut");

    struct params rp;
    setup_br06_radial_params(&rp);

    struct GALAXY gal[2];
    const double coldgas_pre = 1.0;
    run_sf_driver(gal, &rp, coldgas_pre, 0.003, 2.0);

    /* H1 is set from the pre-SF ColdGas, before update_from_star_formation.
     * The ionisation cut removes the diffuse low-column part, so H1 lies
     * strictly between 0 and the full atomic remainder X_H*ColdGas - H2. */
    const double atomic = coldgas_pre * HYDROGEN_MASS_FRAC - gal[0].H2gas;
    ASSERT_GREATER_THAN(gal[0].H1gas, 0.0, "Disk retains some neutral HI");
    ASSERT_LESS_THAN(gal[0].H1gas, atomic, "Ionisation reduces HI below the atomic remainder");
    ASSERT_TRUE(gal[0].H1gas + gal[0].H2gas <= coldgas_pre * HYDROGEN_MASS_FRAC + 1e-12,
                "H1 + H2 <= X_H * ColdGas (no overdraw)");
    ASSERT_GREATER_THAN(gal[0].H2gas, 0.0, "Gas-rich disk forms H2");
    ASSERT_GREATER_THAN(gal[0].SfrDisk[0], 0.0, "Gas-rich disk forms stars");
    ASSERT_LESS_THAN(gal[0].ColdGas, coldgas_pre, "SF consumed cold gas");
}

void test_hi_partition_density_dependence() {
    BEGIN_TEST("HI partition: denser disk keeps a larger neutral fraction");

    struct params rp;
    setup_br06_radial_params(&rp);

    /* Same disk size, more gas => higher surface density => less of it below
     * Sigma_crit => smaller ionised fraction. The neutral fraction
     * (1 - f_ion) = H1 / (X_H*ColdGas - H2) is compared directly. */
    struct GALAXY sparse[2], dense[2];
    run_sf_driver(sparse, &rp, 1.0, 0.003, 2.0);
    run_sf_driver(dense,  &rp, 4.0, 0.003, 2.0);

    const double neutral_sparse = sparse[0].H1gas / (1.0 * HYDROGEN_MASS_FRAC - sparse[0].H2gas);
    const double neutral_dense  = dense[0].H1gas  / (4.0 * HYDROGEN_MASS_FRAC - dense[0].H2gas);

    ASSERT_IN_RANGE(neutral_sparse, 0.0, 1.0, "Sparse neutral fraction in [0,1]");
    ASSERT_IN_RANGE(neutral_dense, 0.0, 1.0, "Dense neutral fraction in [0,1]");
    ASSERT_GREATER_THAN(neutral_dense, neutral_sparse,
                        "Denser disk is less ionised (larger neutral fraction)");
    ASSERT_TRUE(dense[0].H1gas + dense[0].H2gas <= 4.0 * HYDROGEN_MASS_FRAC + 1e-12,
                "H1 + H2 <= X_H * ColdGas (no overdraw)");
}

void test_hi_partition_fully_ionized_dwarf() {
    BEGIN_TEST("HI partition: fully ionised dwarf has H1 = 0, H2 >= 0");

    struct params rp;
    setup_br06_radial_params(&rp);

    /* central surface density well below SigmaHIcrit = 0.5 Msun/pc^2:
     * f_ion = 1, so all atomic hydrogen is ionised. Under the pre-fix
     * bookkeeping H1 would have gone negative by -H2 and been silently
     * clamped; now H1 = 0 exactly with no clamp. */
    struct GALAXY gal[2];
    run_sf_driver(gal, &rp, 1e-4, 0.005, 0.01);

    ASSERT_TRUE(gal[0].H1gas == 0.0, "Fully ionised disk has exactly zero HI");
    ASSERT_TRUE(gal[0].H2gas >= 0.0, "H2 remains non-negative");
    ASSERT_TRUE(gal[0].H1gas + gal[0].H2gas <= 1e-4 * HYDROGEN_MASS_FRAC + 1e-15,
                "Phases within the hydrogen budget");
}

int main() {
    BEGIN_TEST_SUITE("H2 Chemistry and Gas-Phase Partition");

    test_br06_bounds_and_limits();
    test_br06_power_law_index();
    test_br06_stellar_floor();
    test_br06_reference_values();
    test_kd12_guards_and_reference();
    test_k13_floor_and_reference();
    test_gd14_bounds_and_reference();
    test_tdep_k13();
    test_radial_integration_budget();
    test_radial_integration_reference();
    test_hi_partition_atomic_remainder();
    test_hi_partition_density_dependence();
    test_hi_partition_fully_ionized_dwarf();

    END_TEST_SUITE();
    PRINT_TEST_SUMMARY();

    return TEST_EXIT_CODE();
}

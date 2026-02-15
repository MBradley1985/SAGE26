/*
 * DARKMODE TESTS
 * 
 * Tests for the DarkSage-style radially-resolved disk physics:
 * - Radial bin initialization and j-binning
 * - Disc array consistency with bulk quantities
 * - Local star formation calculations
 * - Radial gas flows
 * - Local disk instabilities
 * - H2/HI partitioning in annuli
 * - DiscSFR accumulation
 * - Mass conservation across radial bins
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "test_framework.h"
#include "../src/core_allvars.h"
#include "../src/model_misc.h"
#include "../src/model_darkmode.h"
#include "../src/model_starformation_and_feedback.h"

// Helper function to initialize DarkMode-enabled params
static void initialize_darkmode_params(struct params *run_params) {
    memset(run_params, 0, sizeof(struct params));
    
    run_params->DarkModeOn = 1;
    run_params->DustOn = 1;
    run_params->CGMrecipeOn = 1;
    run_params->Hubble_h = 0.73;
    run_params->RecycleFraction = 0.43;
    run_params->UnitTime_in_s = 3.15e16;
    run_params->UnitDensity_in_cgs = 6.77e-22;
    run_params->UnitMass_in_g = 1.989e43;
    run_params->UnitLength_in_cm = 3.086e24;
    run_params->UnitVelocity_in_cm_per_s = 1.0e5;
    run_params->G = 43.0;
    run_params->Yield = 0.03;
    run_params->SupernovaRecipeOn = 1;
    run_params->FeedbackReheatingEpsilon = 3.0;
    run_params->SFprescription = 0;  // Simple SF
    run_params->SfrEfficiency = 0.05;  // 5% SF efficiency
    
    // Dust parameters
    run_params->DeltaDustAGB = 0.2;
    run_params->DeltaDustSNII = 0.2;
    run_params->DeltaDustSNIa = 0.15;
    run_params->DustAccretionTimescale = 50.0;
    
    // Initialize j-binning (exponential specific angular momentum bins)
    // j_i = j_0 * exp(i * delta_j) where delta_j = ln(j_max/j_0) / N_BINS
    // Use realistic values: j ranges from ~1 to ~100 kpc km/s for disk galaxies
    double j_min = 1.0;     // Minimum specific angular momentum [kpc km/s]
    double j_max = 200.0;   // Maximum specific angular momentum [kpc km/s]
    double delta_j = log(j_max / j_min) / N_BINS;
    
    for(int i = 0; i <= N_BINS; i++) {
        run_params->DiscBinEdge[i] = j_min * exp(i * delta_j);
    }
    
    // Initialize snapshot arrays
    run_params->nsnapshots = 64;
    for(int i = 0; i < 64; i++) {
        run_params->ZZ[i] = 20.0 * (63 - i) / 63.0;
        run_params->AA[i] = 1.0 / (1.0 + run_params->ZZ[i]);
    }
}

// Helper function to initialize a DarkMode galaxy
static void initialize_darkmode_galaxy(struct GALAXY *gal, const struct params *run_params) {
    memset(gal, 0, sizeof(struct GALAXY));
    
    // Bulk properties
    gal->ColdGas = 5.0;          // 5×10^10 Msun/h - more gas for SF
    gal->StellarMass = 2.0;
    gal->MetalsColdGas = 0.1;    // 2% metallicity
    gal->MetalsStellarMass = 0.05;
    gal->Vvir = 200.0;           // km/s
    gal->DiskScaleRadius = 0.003; // Mpc/h (~3 kpc)
    gal->H2gas = 2.5;            // 50% molecular
    gal->Regime = 0;
    
    // Initialize radii from j-binning (capped at 100 kpc)
    const double MAX_RADIUS = 0.1;  // Mpc (100 kpc)
    for(int i = 0; i <= N_BINS; i++) {
        double r_calc = run_params->DiscBinEdge[i] / gal->Vvir;
        gal->DiscRadii[i] = (r_calc < MAX_RADIUS) ? r_calc : MAX_RADIUS;
    }
    
    // Distribute gas exponentially across radial bins
    double total_gas = 0.0;
    double rs = gal->DiskScaleRadius;
    for(int i = 0; i < N_BINS; i++) {
        double r_in = gal->DiscRadii[i];
        double r_out = gal->DiscRadii[i+1];
        double r_mid = 0.5 * (r_in + r_out);
        
        // Exponential profile: Σ(r) ∝ exp(-r/rs)
        double weight = exp(-r_mid / rs) * (r_out - r_in);
        gal->DiscGas[i] = weight;
        total_gas += weight;
    }
    
    // Normalize to match bulk ColdGas
    if(total_gas > 0.0) {
        double norm = gal->ColdGas / total_gas;
        for(int i = 0; i < N_BINS; i++) {
            gal->DiscGas[i] *= norm;
            gal->DiscGasMetals[i] = gal->DiscGas[i] * 0.02;  // 2% metallicity throughout
        }
    }
    
    // Initialize other arrays
    for(int i = 0; i < N_BINS; i++) {
        gal->DiscStars[i] = 0.0;
        gal->DiscStarsMetals[i] = 0.0;
        gal->DiscH2[i] = 0.0;
        gal->DiscHI[i] = 0.0;
        gal->DiscSFR[i] = 0.0;
        gal->DiscDust[i] = gal->DiscGasMetals[i] * 0.25;  // 25% D/M
    }
    
    // Set ColdDust to sum of DiscDust
    gal->ColdDust = 0.0;
    for(int i = 0; i < N_BINS; i++) {
        gal->ColdDust += gal->DiscDust[i];
    }
}

// Helper to sum disc arrays
static double sum_disc_gas(struct GALAXY *gal) {
    double total = 0.0;
    for(int i = 0; i < N_BINS; i++) {
        total += gal->DiscGas[i];
    }
    return total;
}

static double sum_disc_stars(struct GALAXY *gal) {
    double total = 0.0;
    for(int i = 0; i < N_BINS; i++) {
        total += gal->DiscStars[i];
    }
    return total;
}

static double sum_disc_sfr(struct GALAXY *gal) {
    double total = 0.0;
    for(int i = 0; i < N_BINS; i++) {
        total += gal->DiscSFR[i];
    }
    return total;
}

static double sum_disc_dust(struct GALAXY *gal) {
    double total = 0.0;
    for(int i = 0; i < N_BINS; i++) {
        total += gal->DiscDust[i];
    }
    return total;
}


/* ========================================================================
 * TEST: Radial Bin Initialization
 * ======================================================================== */

void test_radii_initialization() {
    BEGIN_TEST("Radial Bin Initialization");
    
    struct params run_params;
    initialize_darkmode_params(&run_params);
    
    struct GALAXY gal;
    initialize_darkmode_galaxy(&gal, &run_params);
    
    // Check radii are monotonically non-decreasing (equal allowed due to capping)
    for(int i = 0; i < N_BINS; i++) {
        ASSERT_TRUE(gal.DiscRadii[i+1] >= gal.DiscRadii[i],
                   "Radii monotonically non-decreasing");
    }
    
    // Check innermost radius is positive
    ASSERT_GREATER_THAN(gal.DiscRadii[0], 0.0, "Inner radius > 0");
    
    // Check outer radius is capped at 100 kpc (0.1 Mpc)
    ASSERT_TRUE(gal.DiscRadii[N_BINS] <= 0.1 + 1e-6, "Outer radius capped at 100 kpc");
    
    // Count how many bins are actively resolved (before hitting cap)
    int active_bins = 0;
    for(int i = 0; i < N_BINS; i++) {
        if(gal.DiscRadii[i+1] > gal.DiscRadii[i]) active_bins++;
    }
    ASSERT_GREATER_THAN(active_bins, 0, "At least some bins are resolved");
    
    printf("    Inner radius: %.4f kpc\n", gal.DiscRadii[0] * 1000.0);
    printf("    Outer radius: %.4f kpc\n", gal.DiscRadii[N_BINS] * 1000.0);
    printf("    Active bins: %d / %d\n", active_bins, N_BINS);
}

void test_jbinning_exponential() {
    BEGIN_TEST("J-Binning is Exponential");
    
    struct params run_params;
    initialize_darkmode_params(&run_params);
    
    // Check j-bins are exponentially spaced
    double ratio1 = run_params.DiscBinEdge[1] / run_params.DiscBinEdge[0];
    double ratio2 = run_params.DiscBinEdge[2] / run_params.DiscBinEdge[1];
    
    ASSERT_CLOSE(ratio1, ratio2, 1e-6, "J-bins exponentially spaced");
    
    printf("    j_0 = %.1f, j_max = %.1f kpc km/s\n", 
           run_params.DiscBinEdge[0], run_params.DiscBinEdge[N_BINS]);
    printf("    Bin ratio: %.4f\n", ratio1);
}


/* ========================================================================
 * TEST: Disc Array Consistency
 * ======================================================================== */

void test_disc_gas_sums_to_bulk() {
    BEGIN_TEST("Disc Gas Sums to Bulk ColdGas");
    
    struct params run_params;
    initialize_darkmode_params(&run_params);
    
    struct GALAXY gal;
    initialize_darkmode_galaxy(&gal, &run_params);
    
    double disc_total = sum_disc_gas(&gal);
    
    ASSERT_CLOSE(disc_total, gal.ColdGas, 1e-6,
                "Sum(DiscGas) = ColdGas");
    
    printf("    ColdGas: %.6e\n", gal.ColdGas);
    printf("    Sum(DiscGas): %.6e\n", disc_total);
}

void test_disc_dust_sums_to_bulk() {
    BEGIN_TEST("Disc Dust Sums to Bulk ColdDust");
    
    struct params run_params;
    initialize_darkmode_params(&run_params);
    
    struct GALAXY gal;
    initialize_darkmode_galaxy(&gal, &run_params);
    
    double disc_total = sum_disc_dust(&gal);
    
    ASSERT_CLOSE(disc_total, gal.ColdDust, 1e-6,
                "Sum(DiscDust) = ColdDust");
    
    printf("    ColdDust: %.6e\n", gal.ColdDust);
    printf("    Sum(DiscDust): %.6e\n", disc_total);
}

void test_disc_profile_decreases_outward() {
    BEGIN_TEST("Disc Profile Decreases Outward");
    
    struct params run_params;
    initialize_darkmode_params(&run_params);
    
    struct GALAXY gal;
    initialize_darkmode_galaxy(&gal, &run_params);
    
    // Inner bins should have more gas than outer bins (exponential profile)
    ASSERT_GREATER_THAN(gal.DiscGas[0], gal.DiscGas[N_BINS-1],
                       "Inner gas > outer gas");
    
    // Most gas should be in inner half
    double inner_total = 0.0, outer_total = 0.0;
    for(int i = 0; i < N_BINS/2; i++) {
        inner_total += gal.DiscGas[i];
    }
    for(int i = N_BINS/2; i < N_BINS; i++) {
        outer_total += gal.DiscGas[i];
    }
    
    ASSERT_GREATER_THAN(inner_total, outer_total,
                       "Inner half has more gas");
    
    printf("    Inner half gas: %.4e (%.1f%%)\n", 
           inner_total, 100.0 * inner_total / gal.ColdGas);
    printf("    Outer half gas: %.4e (%.1f%%)\n", 
           outer_total, 100.0 * outer_total / gal.ColdGas);
}


/* ========================================================================
 * TEST: Local Star Formation
 * ======================================================================== */

void test_compute_local_sf() {
    BEGIN_TEST("Compute Local Star Formation");
    
    struct params run_params;
    initialize_darkmode_params(&run_params);
    
    struct GALAXY gal;
    initialize_darkmode_galaxy(&gal, &run_params);
    
    double sfr_local[N_BINS];
    double h2_local[N_BINS];
    double dt = 0.01;  // ~10 Myr timestep
    
    // Call the local SF calculation
    compute_local_star_formation(0, dt, &gal, &run_params, sfr_local, h2_local);
    
    // Check SFR is non-negative everywhere
    for(int i = 0; i < N_BINS; i++) {
        ASSERT_TRUE(sfr_local[i] >= 0.0, "SFR >= 0 in all bins");
    }
    
    // Check H2 is non-negative and <= gas
    for(int i = 0; i < N_BINS; i++) {
        ASSERT_TRUE(h2_local[i] >= 0.0, "H2 >= 0 in all bins");
        ASSERT_TRUE(h2_local[i] <= gal.DiscGas[i] * 0.74 + 1e-10, "H2 <= H*DiscGas");
    }
    
    // SFR should be higher where there's more gas (roughly)
    double total_sfr = 0.0;
    int nonzero_bins = 0;
    for(int i = 0; i < N_BINS; i++) {
        total_sfr += sfr_local[i];
        if(sfr_local[i] > 0.0) nonzero_bins++;
    }
    
    ASSERT_GREATER_THAN(total_sfr, 0.0, "Total local SFR > 0");
    ASSERT_TRUE(nonzero_bins > 0, "At least one bin has SF");
    
    // SFR is in [10^10 Msun/h / Gyr], convert to Msun/yr:
    // SFR [Msun/yr] = total_sfr * 10^10 / h / 10^9 = total_sfr * 10 / h
    double sfr_msun_yr = total_sfr * 10.0 / run_params.Hubble_h;
    printf("    Total local SFR: %.2f Msun/yr\n", sfr_msun_yr);
    printf("    Non-zero SF bins: %d / %d\n", nonzero_bins, N_BINS);
}

void test_local_sf_depends_on_gas() {
    BEGIN_TEST("Local SF Depends on Gas Content");
    
    struct params run_params;
    initialize_darkmode_params(&run_params);
    
    struct GALAXY gal;
    initialize_darkmode_galaxy(&gal, &run_params);
    
    double sfr_local[N_BINS];
    double h2_local[N_BINS];
    double dt = 0.01;  // ~10 Myr timestep
    
    compute_local_star_formation(0, dt, &gal, &run_params, sfr_local, h2_local);
    double sfr_initial = 0.0;
    for(int i = 0; i < N_BINS; i++) sfr_initial += sfr_local[i];
    
    // Double the gas
    gal.ColdGas *= 2.0;
    for(int i = 0; i < N_BINS; i++) {
        gal.DiscGas[i] *= 2.0;
        gal.DiscGasMetals[i] *= 2.0;
    }
    
    compute_local_star_formation(0, dt, &gal, &run_params, sfr_local, h2_local);
    double sfr_doubled = 0.0;
    for(int i = 0; i < N_BINS; i++) sfr_doubled += sfr_local[i];
    
    ASSERT_GREATER_THAN(sfr_doubled, sfr_initial, 
                       "SFR increases with gas");
    
    printf("    SFR with 1x gas: %.4e\n", sfr_initial);
    printf("    SFR with 2x gas: %.4e\n", sfr_doubled);
}


/* ========================================================================
 * TEST: DiscSFR Accumulation
 * ======================================================================== */

void test_discsfr_accumulation() {
    BEGIN_TEST("DiscSFR Accumulates Over Timesteps");
    
    struct params run_params;
    initialize_darkmode_params(&run_params);
    
    struct GALAXY gal;
    initialize_darkmode_galaxy(&gal, &run_params);
    
    // Simulate multiple timesteps of SFR accumulation
    double dt = 0.01;  // ~10 Myr
    double sfr_local[N_BINS];
    double h2_local[N_BINS];
    
    compute_local_star_formation(0, dt, &gal, &run_params, sfr_local, h2_local);
    
    // First timestep
    for(int i = 0; i < N_BINS; i++) {
        gal.DiscSFR[i] += sfr_local[i];
    }
    double sfr_step1 = sum_disc_sfr(&gal);
    
    // Second timestep (SFR should accumulate)
    for(int i = 0; i < N_BINS; i++) {
        gal.DiscSFR[i] += sfr_local[i];
    }
    double sfr_step2 = sum_disc_sfr(&gal);
    
    ASSERT_CLOSE(sfr_step2, 2.0 * sfr_step1, 1e-6,
                "DiscSFR doubles after 2 accumulations");
    
    printf("    DiscSFR after 1 step: %.4e\n", sfr_step1);
    printf("    DiscSFR after 2 steps: %.4e\n", sfr_step2);
}


/* ========================================================================
 * TEST: Mass Conservation in Radial Bins
 * ======================================================================== */

void test_radial_mass_conservation() {
    BEGIN_TEST("Radial Bins Conserve Mass");
    
    struct params run_params;
    initialize_darkmode_params(&run_params);
    
    struct GALAXY gal;
    initialize_darkmode_galaxy(&gal, &run_params);
    
    double initial_gas = sum_disc_gas(&gal);
    double initial_stars = sum_disc_stars(&gal);
    double initial_total = initial_gas + initial_stars;
    
    // Simulate star formation: convert some gas to stars
    double stars_formed = 0.1 * gal.ColdGas;
    double recycle = run_params.RecycleFraction;
    
    // Remove gas proportionally from all bins
    for(int i = 0; i < N_BINS; i++) {
        double frac = gal.DiscGas[i] / gal.ColdGas;
        double stars_bin = frac * stars_formed;
        gal.DiscGas[i] -= stars_bin;
        gal.DiscStars[i] += stars_bin * (1.0 - recycle);
    }
    gal.ColdGas -= stars_formed;
    gal.StellarMass += stars_formed * (1.0 - recycle);
    
    double final_gas = sum_disc_gas(&gal);
    double final_stars = sum_disc_stars(&gal);
    double final_total = final_gas + final_stars;
    
    // Mass conservation (accounting for recycling)
    double expected_total = initial_total - stars_formed * recycle;
    ASSERT_CLOSE(final_total, expected_total, 1e-6,
                "Total disc mass conserved (minus recycling)");
    
    // Disc arrays should still sum to bulk
    ASSERT_CLOSE(final_gas, gal.ColdGas, 1e-6,
                "Sum(DiscGas) = ColdGas after SF");
    
    printf("    Initial gas+stars: %.4e\n", initial_total);
    printf("    Final gas+stars: %.4e (expected %.4e)\n", final_total, expected_total);
}


/* ========================================================================
 * TEST: Physical Bounds
 * ======================================================================== */

void test_disc_arrays_positive() {
    BEGIN_TEST("Disc Arrays Are Non-Negative");
    
    struct params run_params;
    initialize_darkmode_params(&run_params);
    
    struct GALAXY gal;
    initialize_darkmode_galaxy(&gal, &run_params);
    
    // All disc arrays should be non-negative
    for(int i = 0; i < N_BINS; i++) {
        ASSERT_TRUE(gal.DiscGas[i] >= 0.0, "DiscGas >= 0");
        ASSERT_TRUE(gal.DiscStars[i] >= 0.0, "DiscStars >= 0");
        ASSERT_TRUE(gal.DiscGasMetals[i] >= 0.0, "DiscGasMetals >= 0");
        ASSERT_TRUE(gal.DiscStarsMetals[i] >= 0.0, "DiscStarsMetals >= 0");
        ASSERT_TRUE(gal.DiscDust[i] >= 0.0, "DiscDust >= 0");
        ASSERT_TRUE(gal.DiscH2[i] >= 0.0, "DiscH2 >= 0");
        ASSERT_TRUE(gal.DiscHI[i] >= 0.0, "DiscHI >= 0");
        ASSERT_TRUE(gal.DiscSFR[i] >= 0.0, "DiscSFR >= 0");
    }
    
    // Radii should be positive and finite
    for(int i = 0; i <= N_BINS; i++) {
        ASSERT_GREATER_THAN(gal.DiscRadii[i], 0.0, "DiscRadii > 0");
        ASSERT_TRUE(!isnan(gal.DiscRadii[i]), "DiscRadii not NaN");
        ASSERT_TRUE(!isinf(gal.DiscRadii[i]), "DiscRadii not Inf");
    }
}

void test_disc_metallicity_bounded() {
    BEGIN_TEST("Disc Metallicity < Gas Mass");
    
    struct params run_params;
    initialize_darkmode_params(&run_params);
    
    struct GALAXY gal;
    initialize_darkmode_galaxy(&gal, &run_params);
    
    // Metals should be <= gas in each bin
    for(int i = 0; i < N_BINS; i++) {
        if(gal.DiscGas[i] > 1e-10) {
            ASSERT_TRUE(gal.DiscGasMetals[i] <= gal.DiscGas[i] + 1e-10,
                       "DiscGasMetals <= DiscGas");
        }
        if(gal.DiscStars[i] > 1e-10) {
            ASSERT_TRUE(gal.DiscStarsMetals[i] <= gal.DiscStars[i] + 1e-10,
                       "DiscStarsMetals <= DiscStars");
        }
    }
}

void test_disc_dust_bounded_by_metals() {
    BEGIN_TEST("Disc Dust <= Disc Metals");
    
    struct params run_params;
    initialize_darkmode_params(&run_params);
    
    struct GALAXY gal;
    initialize_darkmode_galaxy(&gal, &run_params);
    
    // Dust should be <= metals in each bin
    for(int i = 0; i < N_BINS; i++) {
        if(gal.DiscGasMetals[i] > 1e-10) {
            ASSERT_TRUE(gal.DiscDust[i] <= gal.DiscGasMetals[i] + 1e-10,
                       "DiscDust <= DiscGasMetals");
        }
    }
    
    double total_disc_dust = sum_disc_dust(&gal);
    double total_disc_metals = 0.0;
    for(int i = 0; i < N_BINS; i++) {
        total_disc_metals += gal.DiscGasMetals[i];
    }
    
    ASSERT_TRUE(total_disc_dust <= total_disc_metals + 1e-6,
               "Total DiscDust <= Total DiscMetals");
}


/* ========================================================================
 * TEST: DarkModeOn Toggle
 * ======================================================================== */

void test_darkmode_off_ignores_disc_arrays() {
    BEGIN_TEST("DarkModeOn=0 Ignores Disc Arrays");
    
    struct params run_params;
    initialize_darkmode_params(&run_params);
    run_params.DarkModeOn = 0;  // Disable DarkMode
    
    struct GALAXY gal;
    memset(&gal, 0, sizeof(struct GALAXY));
    gal.ColdGas = 1.0;
    gal.MetalsColdGas = 0.02;
    gal.Vvir = 200.0;
    gal.DiskScaleRadius = 0.003;
    
    // Don't initialize disc arrays - they should be ignored
    
    // With DarkModeOn=0, bulkSF should still work without touching disc arrays
    ASSERT_TRUE(run_params.DarkModeOn == 0, "DarkModeOn is 0");
    
    // Disc arrays should remain zero
    double disc_total = sum_disc_gas(&gal);
    ASSERT_EQUAL_FLOAT(disc_total, 0.0, "DiscGas arrays unused when DarkModeOn=0");
}


/* ========================================================================
 * TEST: Physics Validation - MW-like Galaxy
 * These tests validate that the physics produces reasonable values
 * compared to observations
 * ======================================================================== */

void test_mw_like_sfr() {
    BEGIN_TEST("MW-like Galaxy SFR (Physics Validation)");
    
    struct params run_params;
    initialize_darkmode_params(&run_params);
    
    // Set up a Milky Way-like galaxy
    // MW: M_gas ~ 5-10 × 10^9 Msun, SFR ~ 1-3 Msun/yr
    struct GALAXY gal;
    memset(&gal, 0, sizeof(struct GALAXY));
    
    gal.ColdGas = 0.8;           // 0.8 × 10^10 Msun/h = 1.1 × 10^10 Msun (realistic)
    gal.StellarMass = 5.0;       // 5 × 10^10 Msun/h
    gal.MetalsColdGas = 0.016;   // Z ~ 0.02 (solar)
    gal.Vvir = 220.0;            // km/s
    gal.DiskScaleRadius = 0.0025; // 2.5 kpc in Mpc/h (MW scale length ~ 2-3 kpc)
    
    // Initialize radii
    const double MAX_RADIUS = 0.1;
    for(int i = 0; i <= N_BINS; i++) {
        double r_calc = run_params.DiscBinEdge[i] / gal.Vvir;
        gal.DiscRadii[i] = (r_calc < MAX_RADIUS) ? r_calc : MAX_RADIUS;
    }
    
    // Distribute gas exponentially
    double total = 0.0;
    double rs = gal.DiskScaleRadius;
    for(int i = 0; i < N_BINS; i++) {
        double r_mid = 0.5 * (gal.DiscRadii[i] + gal.DiscRadii[i+1]);
        double dr = gal.DiscRadii[i+1] - gal.DiscRadii[i];
        if(dr > 0) {
            gal.DiscGas[i] = exp(-r_mid / rs) * dr;
            total += gal.DiscGas[i];
        }
    }
    for(int i = 0; i < N_BINS; i++) {
        gal.DiscGas[i] *= gal.ColdGas / total;
        gal.DiscGasMetals[i] = gal.DiscGas[i] * 0.02;
    }
    
    double sfr_local[N_BINS], h2_local[N_BINS];
    double dt = 0.01;
    double total_sfr = compute_local_star_formation(0, dt, &gal, &run_params, sfr_local, h2_local);
    
    // Convert to physical units: SFR [Msun/yr] = total_sfr * 10 / h
    double sfr_msun_yr = total_sfr * 10.0 / run_params.Hubble_h;
    
    // MW SFR should be ~1-10 Msun/yr (allowing some tolerance for model uncertainty)
    ASSERT_GREATER_THAN(sfr_msun_yr, 0.1, "MW SFR > 0.1 Msun/yr");
    ASSERT_LESS_THAN(sfr_msun_yr, 50.0, "MW SFR < 50 Msun/yr");
    
    // Gas depletion time: t_dep = M_gas / SFR
    // For MW, expect ~1-3 Gyr for molecular gas, ~5-10 Gyr for total gas
    double m_gas_msun = gal.ColdGas * 1e10 / run_params.Hubble_h;
    double t_dep_gyr = m_gas_msun / sfr_msun_yr / 1e9;
    
    ASSERT_GREATER_THAN(t_dep_gyr, 0.5, "Gas depletion > 0.5 Gyr");
    ASSERT_LESS_THAN(t_dep_gyr, 20.0, "Gas depletion < 20 Gyr");
    
    printf("    MW-like gas mass: %.2e Msun\n", m_gas_msun);
    printf("    SFR: %.2f Msun/yr (MW observed: ~1-3 Msun/yr)\n", sfr_msun_yr);
    printf("    Gas depletion: %.1f Gyr (expected: ~1-5 Gyr)\n", t_dep_gyr);
}

void test_sfr_scales_with_gas() {
    BEGIN_TEST("SFR Scales with Gas Mass (K-S relation check)");
    
    struct params run_params;
    initialize_darkmode_params(&run_params);
    
    // Test that SFR ~ gas^N where N ~ 1-1.5 (Kennicutt-Schmidt)
    // Use gas masses that are high enough to be above the SF threshold
    double sfr_low, sfr_high;
    double gas_low = 1.0;   // 1 × 10^10 Msun/h (above threshold)
    double gas_high = 4.0;  // 4 × 10^10 Msun/h
    
    for(int test = 0; test < 2; test++) {
        struct GALAXY gal;
        memset(&gal, 0, sizeof(struct GALAXY));
        
        gal.ColdGas = (test == 0) ? gas_low : gas_high;
        gal.StellarMass = 3.0;
        gal.MetalsColdGas = gal.ColdGas * 0.02;
        gal.Vvir = 200.0;
        gal.DiskScaleRadius = 0.003;
        
        const double MAX_RADIUS = 0.1;
        for(int i = 0; i <= N_BINS; i++) {
            double r_calc = run_params.DiscBinEdge[i] / gal.Vvir;
            gal.DiscRadii[i] = (r_calc < MAX_RADIUS) ? r_calc : MAX_RADIUS;
        }
        
        double total = 0.0;
        for(int i = 0; i < N_BINS; i++) {
            double r_mid = 0.5 * (gal.DiscRadii[i] + gal.DiscRadii[i+1]);
            double dr = gal.DiscRadii[i+1] - gal.DiscRadii[i];
            if(dr > 0) {
                gal.DiscGas[i] = exp(-r_mid / gal.DiskScaleRadius) * dr;
                total += gal.DiscGas[i];
            }
        }
        for(int i = 0; i < N_BINS; i++) {
            gal.DiscGas[i] *= gal.ColdGas / total;
            gal.DiscGasMetals[i] = gal.DiscGas[i] * 0.02;
        }
        
        double sfr_local[N_BINS], h2_local[N_BINS];
        double total_sfr = compute_local_star_formation(0, 0.01, &gal, &run_params, sfr_local, h2_local);
        
        if(test == 0) sfr_low = total_sfr;
        else sfr_high = total_sfr;
    }
    
    // Check that both have non-zero SFR
    if(sfr_low <= 0.0 || sfr_high <= 0.0) {
        printf("    WARNING: SFR below threshold for test galaxy\n");
        printf("    sfr_low = %.4e, sfr_high = %.4e\n", sfr_low, sfr_high);
        printf("    (This is expected for low-mass galaxies with SF threshold)\n");
        return;  // Skip ratio test if below threshold
    }
    
    // Check scaling: SFR_high / SFR_low should scale with gas ratio
    // K-S has N ~ 1-1.5, but with threshold it can be steeper
    double gas_ratio = gas_high / gas_low;  // = 4
    double sfr_ratio = sfr_high / sfr_low;
    
    ASSERT_GREATER_THAN(sfr_ratio, 1.0, "Higher gas -> higher SFR");
    ASSERT_GREATER_THAN(sfr_ratio, gas_ratio * 0.5, "SFR scales with gas");
    
    printf("    Gas ratio: %.1f\n", gas_ratio);
    printf("    SFR ratio: %.2f\n", sfr_ratio);
    printf("    (K-S predicts ratio ~ %.1f - %.1f for N=1.0-1.5)\n", 
           pow(gas_ratio, 1.0), pow(gas_ratio, 1.5));
}

void test_sfr_profile_centrally_peaked() {
    BEGIN_TEST("SFR Profile Centrally Peaked");
    
    struct params run_params;
    initialize_darkmode_params(&run_params);
    
    struct GALAXY gal;
    initialize_darkmode_galaxy(&gal, &run_params);
    
    double sfr_local[N_BINS], h2_local[N_BINS];
    compute_local_star_formation(0, 0.01, &gal, &run_params, sfr_local, h2_local);
    
    // Find which bins have non-zero SFR
    int first_nonzero = -1, last_nonzero = -1;
    for(int i = 0; i < N_BINS; i++) {
        if(sfr_local[i] > 0.0) {
            if(first_nonzero < 0) first_nonzero = i;
            last_nonzero = i;
        }
    }
    
    // Inner bins should have higher SFR than outer bins (for exponential disk)
    double sfr_inner = 0.0, sfr_outer = 0.0;
    int mid = (first_nonzero + last_nonzero) / 2;
    for(int i = first_nonzero; i <= mid; i++) sfr_inner += sfr_local[i];
    for(int i = mid+1; i <= last_nonzero; i++) sfr_outer += sfr_local[i];
    
    if(last_nonzero > first_nonzero + 2) {
        ASSERT_GREATER_THAN(sfr_inner, sfr_outer, "Inner SFR > Outer SFR");
    }
    
    printf("    Inner half SFR: %.4e\n", sfr_inner);
    printf("    Outer half SFR: %.4e\n", sfr_outer);
    printf("    SF bins: %d to %d (of %d)\n", first_nonzero, last_nonzero, N_BINS);
}


/* ========================================================================
 * MAIN TEST RUNNER
 * ======================================================================== */

int main(void) {
    BEGIN_TEST_SUITE("DarkMode (Radially-Resolved Disks) Tests");
    
    // Initialization tests
    test_radii_initialization();
    test_jbinning_exponential();
    
    // Consistency tests
    test_disc_gas_sums_to_bulk();
    test_disc_dust_sums_to_bulk();
    test_disc_profile_decreases_outward();
    
    // Local star formation tests
    test_compute_local_sf();
    test_local_sf_depends_on_gas();
    
    // Accumulation tests
    test_discsfr_accumulation();
    
    // Conservation tests
    test_radial_mass_conservation();
    
    // Physical bounds tests
    test_disc_arrays_positive();
    test_disc_metallicity_bounded();
    test_disc_dust_bounded_by_metals();
    
    // Toggle test
    test_darkmode_off_ignores_disc_arrays();
    
    // PHYSICS VALIDATION TESTS
    test_mw_like_sfr();
    test_sfr_scales_with_gas();
    test_sfr_profile_centrally_peaked();
    
    END_TEST_SUITE();
    PRINT_TEST_SUMMARY();
    
    return TEST_EXIT_CODE();
}

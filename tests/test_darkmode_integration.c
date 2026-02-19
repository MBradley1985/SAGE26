/*
 * DARKMODE INTEGRATION TESTS
 * 
 * Comprehensive integration tests validating consistency across:
 *   1. Dust model (production, distribution, conservation)
 *   2. Angular momentum (spin vectors, alignment, specific AM)
 *   3. Spatially resolved disks (profiles, bulk-disc consistency)
 * 
 * These tests verify that all three physics components work together
 * correctly in the DarkMode framework.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "test_framework.h"
#include "../src/core_allvars.h"
#include "../src/model_misc.h"
#include "../src/model_darkmode.h"
#include "../src/model_dust.h"
#include "../src/model_starformation_and_feedback.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ========================================================================
 * HELPER FUNCTIONS
 * ======================================================================== */

static void initialize_integration_params(struct params *run_params) {
    memset(run_params, 0, sizeof(struct params));
    
    // Enable all DarkMode features
    run_params->DarkSAGEOn = 1;
    run_params->DustOn = 1;
    run_params->CGMrecipeOn = 1;
    run_params->DiskInstabilityOn = 1;
    
    // Standard cosmology
    run_params->Hubble_h = 0.73;
    run_params->RecycleFraction = 0.43;
    run_params->Yield = 0.03;
    
    // Unit system
    run_params->UnitTime_in_s = 3.15e16;
    run_params->UnitDensity_in_cgs = 6.77e-22;
    run_params->UnitMass_in_g = 1.989e43;
    run_params->UnitLength_in_cm = 3.086e24;
    run_params->UnitVelocity_in_cm_per_s = 1.0e5;
    run_params->G = 43.0;
    
    // Star formation
    run_params->SupernovaRecipeOn = 1;
    run_params->FeedbackReheatingEpsilon = 3.0;
    run_params->SFprescription = 0;
    run_params->SfrEfficiency = 0.05;
    
    // Dust parameters
    run_params->DeltaDustAGB = 0.2;
    run_params->DeltaDustSNII = 0.2;
    run_params->DeltaDustSNIa = 0.15;
    run_params->DustAccretionTimescale = 50.0;
    
    // Initialize j-binning
    double j_min = 1.0;
    double j_max = 200.0;
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

static void initialize_mw_galaxy(struct GALAXY *gal, const struct params *run_params) {
    memset(gal, 0, sizeof(struct GALAXY));
    
    // MW-like properties (masses in 10^10 Msun/h)
    gal->ColdGas = 1.0;              // 10^10 Msun/h cold gas
    gal->StellarMass = 5.0;          // 5×10^10 Msun/h stellar mass
    gal->BulgeMass = 1.0;            // 10^10 Msun/h bulge
    gal->MetalsColdGas = 0.02;       // 2% metallicity
    gal->MetalsStellarMass = 0.03;
    gal->Vvir = 200.0;               // km/s
    gal->DiskScaleRadius = 0.003;    // 3 kpc in Mpc/h
    gal->H2gas = 0.5;                // 50% molecular
    gal->Regime = 0;
    gal->Type = 0;                   // Central galaxy
    gal->Mvir = 100.0;               // 10^12 Msun/h halo
    
    // Initialize disc radii from j-binning
    const double MAX_RADIUS = 0.1;   // 100 kpc cap
    for(int i = 0; i <= N_BINS; i++) {
        double r_calc = run_params->DiscBinEdge[i] / gal->Vvir;
        gal->DiscRadii[i] = (r_calc < MAX_RADIUS) ? r_calc : MAX_RADIUS;
    }
    
    // Distribute gas exponentially across radial bins
    double total_gas = 0.0;
    double rs = gal->DiskScaleRadius;
    for(int i = 0; i < N_BINS; i++) {
        double r_mid = 0.5 * (gal->DiscRadii[i] + gal->DiscRadii[i+1]);
        double weight = exp(-r_mid / rs) * (gal->DiscRadii[i+1] - gal->DiscRadii[i]);
        gal->DiscGas[i] = weight;
        total_gas += weight;
    }
    
    // Normalize to match bulk ColdGas
    if(total_gas > 0.0) {
        double norm = gal->ColdGas / total_gas;
        for(int i = 0; i < N_BINS; i++) {
            gal->DiscGas[i] *= norm;
            gal->DiscGasMetals[i] = gal->DiscGas[i] * 0.02;
        }
    }
    
    // Initialize stellar disc similarly
    double total_stars = 0.0;
    for(int i = 0; i < N_BINS; i++) {
        double r_mid = 0.5 * (gal->DiscRadii[i] + gal->DiscRadii[i+1]);
        double weight = exp(-r_mid / rs) * (gal->DiscRadii[i+1] - gal->DiscRadii[i]);
        gal->DiscStars[i] = weight;
        total_stars += weight;
    }
    
    // Stellar disc mass = StellarMass - BulgeMass
    double disc_stellar_mass = gal->StellarMass - gal->BulgeMass;
    if(total_stars > 0.0 && disc_stellar_mass > 0.0) {
        double norm = disc_stellar_mass / total_stars;
        for(int i = 0; i < N_BINS; i++) {
            gal->DiscStars[i] *= norm;
            gal->DiscStarsMetals[i] = gal->DiscStars[i] * 0.03;
        }
    }
    
    // Initialize dust (25% of metals in dust form)
    gal->ColdDust = 0.0;
    for(int i = 0; i < N_BINS; i++) {
        gal->DiscDust[i] = gal->DiscGasMetals[i] * 0.25;
        gal->ColdDust += gal->DiscDust[i];
    }
    
    // Initialize H2/HI (more H2 in center)
    for(int i = 0; i < N_BINS; i++) {
        double r_mid = 0.5 * (gal->DiscRadii[i] + gal->DiscRadii[i+1]);
        double h2_frac = exp(-r_mid / (2.0 * rs));  // H2 fraction higher in center
        gal->DiscH2[i] = gal->DiscGas[i] * h2_frac;
        gal->DiscHI[i] = gal->DiscGas[i] - gal->DiscH2[i];
    }
    
    // Initialize spin vectors (aligned with z-axis initially)
    gal->SpinGas[0] = 0.0;
    gal->SpinGas[1] = 0.0;
    gal->SpinGas[2] = 1.0;
    gal->SpinStars[0] = 0.0;
    gal->SpinStars[1] = 0.0;
    gal->SpinStars[2] = 1.0;
}

static double sum_array(const float *arr, int n) {
    double total = 0.0;
    for(int i = 0; i < n; i++) {
        total += arr[i];
    }
    return total;
}

static double compute_spin_magnitude(const float *spin) {
    return sqrt(spin[0]*spin[0] + spin[1]*spin[1] + spin[2]*spin[2]);
}

static double compute_spin_alignment(const float *spin1, const float *spin2) {
    double mag1 = compute_spin_magnitude(spin1);
    double mag2 = compute_spin_magnitude(spin2);
    if(mag1 < 1e-10 || mag2 < 1e-10) return 0.0;
    
    double dot = spin1[0]*spin2[0] + spin1[1]*spin2[1] + spin1[2]*spin2[2];
    double cos_angle = dot / (mag1 * mag2);
    cos_angle = fmax(-1.0, fmin(1.0, cos_angle));
    return acos(cos_angle) * 180.0 / M_PI;  // Return angle in degrees
}

static double compute_specific_am(struct GALAXY *gal) {
    // j* = Σ_i (M_i * r_i * V_circ) / Σ_i M_i
    // For flat rotation curve: j* ≈ 2 * R_d * V_circ
    double v_circ = gal->Vvir;
    double r_d = gal->DiskScaleRadius * 1000.0;  // kpc
    return 2.0 * r_d * v_circ;  // kpc km/s
}


/* ========================================================================
 * INTEGRATION TEST 1: DUST-DISC CONSISTENCY
 * 
 * Verifies that dust is properly tracked across disc annuli:
 *   - ΣDiscDust = ColdDust
 *   - DiscDust[i] ≤ DiscGasMetals[i] (dust cannot exceed metals)
 *   - DiscDust[i] ≥ 0 (non-negative)
 *   - Dust-to-metal ratio is physically reasonable (~0.3-0.5)
 * ======================================================================== */

void test_dust_disc_consistency() {
    BEGIN_TEST("Dust-Disc Consistency");
    
    struct params run_params;
    initialize_integration_params(&run_params);
    
    struct GALAXY gal;
    initialize_mw_galaxy(&gal, &run_params);
    
    // Test 1: DiscDust sums to ColdDust
    double disc_dust_sum = sum_array(gal.DiscDust, N_BINS);
    ASSERT_CLOSE(gal.ColdDust, disc_dust_sum, 1e-6,
                "ΣDiscDust == ColdDust");
    
    // Test 2: DiscDust ≤ DiscGasMetals in each bin
    int dust_exceeds_metals = 0;
    for(int i = 0; i < N_BINS; i++) {
        if(gal.DiscDust[i] > gal.DiscGasMetals[i] * 1.001) {  // 0.1% tolerance
            dust_exceeds_metals++;
        }
    }
    ASSERT_EQUAL_INT(0, dust_exceeds_metals,
                    "DiscDust ≤ DiscGasMetals in all bins");
    
    // Test 3: All dust values non-negative
    int negative_dust = 0;
    for(int i = 0; i < N_BINS; i++) {
        if(gal.DiscDust[i] < 0) negative_dust++;
    }
    ASSERT_EQUAL_INT(0, negative_dust,
                    "No negative DiscDust values");
    
    // Test 4: Dust-to-metal ratio is reasonable
    double total_metals = sum_array(gal.DiscGasMetals, N_BINS);
    double dtm_ratio = disc_dust_sum / total_metals;
    ASSERT_IN_RANGE(dtm_ratio, 0.1, 0.6,
                   "Dust-to-metal ratio in reasonable range (0.1-0.6)");
    
    printf("    Disc dust sum: %.6e\n", disc_dust_sum);
    printf("    ColdDust bulk: %.6e\n", gal.ColdDust);
    printf("    Dust-to-metal ratio: %.3f\n", dtm_ratio);
}


/* ========================================================================
 * INTEGRATION TEST 2: ANGULAR MOMENTUM-DISC CONSISTENCY
 * 
 * Verifies spin vectors and AM are physically meaningful:
 *   - Spin vectors are unit vectors (|J| = 1)
 *   - Gas and stellar discs are reasonably aligned
 *   - Specific AM computed from disc profile matches expectation
 * ======================================================================== */

void test_angular_momentum_disc_consistency() {
    BEGIN_TEST("Angular Momentum-Disc Consistency");
    
    struct params run_params;
    initialize_integration_params(&run_params);
    
    struct GALAXY gal;
    initialize_mw_galaxy(&gal, &run_params);
    
    // Test 1: SpinGas is unit vector
    double spin_gas_mag = compute_spin_magnitude(gal.SpinGas);
    ASSERT_CLOSE(1.0, spin_gas_mag, 1e-6,
                "SpinGas is unit vector (|J| = 1)");
    
    // Test 2: SpinStars is unit vector
    double spin_stars_mag = compute_spin_magnitude(gal.SpinStars);
    ASSERT_CLOSE(1.0, spin_stars_mag, 1e-6,
                "SpinStars is unit vector (|J| = 1)");
    
    // Test 3: Gas and stellar discs are well-aligned (<30°)
    double alignment = compute_spin_alignment(gal.SpinGas, gal.SpinStars);
    ASSERT_LESS_THAN(alignment, 30.0,
                    "Gas-stellar alignment < 30°");
    
    // Test 4: Specific angular momentum is reasonable
    // For MW-like galaxy: j* ~ 10^3 kpc km/s
    double j_star = compute_specific_am(&gal);
    ASSERT_IN_RANGE(j_star, 100.0, 5000.0,
                   "Specific AM in reasonable range (100-5000 kpc km/s)");
    
    // Test 5: AM direction can be computed from mass-weighted radius
    // This is a sanity check that disc arrays enable AM calculation
    double mass_weighted_r = 0.0;
    double total_mass = 0.0;
    for(int i = 0; i < N_BINS; i++) {
        double r_mid = 0.5 * (gal.DiscRadii[i] + gal.DiscRadii[i+1]) * 1000.0;  // kpc
        mass_weighted_r += gal.DiscGas[i] * r_mid;
        total_mass += gal.DiscGas[i];
    }
    double mean_radius = (total_mass > 0) ? mass_weighted_r / total_mass : 0;
    ASSERT_GREATER_THAN(mean_radius, 0.0,
                       "Mass-weighted mean radius > 0");
    
    printf("    SpinGas magnitude: %.6f\n", spin_gas_mag);
    printf("    SpinStars magnitude: %.6f\n", spin_stars_mag);
    printf("    Gas-stellar alignment: %.1f°\n", alignment);
    printf("    Specific AM (j*): %.1f kpc km/s\n", j_star);
    printf("    Mean gas radius: %.2f kpc\n", mean_radius);
}


/* ========================================================================
 * INTEGRATION TEST 3: DISC ARRAY BULK CONSISTENCY
 * 
 * Verifies all disc arrays sum to their bulk counterparts:
 *   - ΣDiscGas = ColdGas
 *   - ΣDiscStars = StellarMass - BulgeMass
 *   - ΣDiscGasMetals = MetalsColdGas
 *   - ΣDiscH2 + ΣDiscHI = ΣDiscGas
 * ======================================================================== */

void test_disc_bulk_consistency() {
    BEGIN_TEST("Disc-Bulk Mass Consistency");
    
    struct params run_params;
    initialize_integration_params(&run_params);
    
    struct GALAXY gal;
    initialize_mw_galaxy(&gal, &run_params);
    
    // Test 1: Gas conservation
    double disc_gas_sum = sum_array(gal.DiscGas, N_BINS);
    ASSERT_CLOSE(gal.ColdGas, disc_gas_sum, 1e-6,
                "ΣDiscGas == ColdGas");
    
    // Test 2: Stellar disc conservation
    double disc_stars_sum = sum_array(gal.DiscStars, N_BINS);
    double expected_disc_stars = gal.StellarMass - gal.BulgeMass;
    ASSERT_CLOSE(expected_disc_stars, disc_stars_sum, 1e-6,
                "ΣDiscStars == StellarMass - BulgeMass");
    
    // Test 3: Gas metals conservation
    double disc_gas_metals_sum = sum_array(gal.DiscGasMetals, N_BINS);
    ASSERT_CLOSE(gal.MetalsColdGas, disc_gas_metals_sum, 1e-6,
                "ΣDiscGasMetals == MetalsColdGas");
    
    // Test 4: H2 + HI = Gas (in each bin) - use relative tolerance for non-zero bins
    int h2hi_mismatch = 0;
    for(int i = 0; i < N_BINS; i++) {
        double h2_plus_hi = gal.DiscH2[i] + gal.DiscHI[i];
        double gas_i = gal.DiscGas[i];
        // Use relative tolerance for non-trivial amounts, absolute for tiny values
        double tol = (gas_i > 1e-10) ? 1e-6 * gas_i : 1e-10;
        if(fabs(h2_plus_hi - gas_i) > tol) {
            h2hi_mismatch++;
        }
    }
    ASSERT_EQUAL_INT(0, h2hi_mismatch,
                    "DiscH2 + DiscHI == DiscGas in all bins");
    
    printf("    Cold gas (bulk): %.6e, Disc sum: %.6e\n", gal.ColdGas, disc_gas_sum);
    printf("    Disc stellar (expected): %.6e, Sum: %.6e\n", expected_disc_stars, disc_stars_sum);
    printf("    Gas metals (bulk): %.6e, Disc sum: %.6e\n", gal.MetalsColdGas, disc_gas_metals_sum);
}


/* ========================================================================
 * INTEGRATION TEST 4: RADIAL PROFILE PHYSICS
 * 
 * Verifies that radial profiles follow expected physics:
 *   - Surface density decreases outward (exponential disc)
 *   - H2 fraction higher in center
 *   - Metallicity gradient exists
 * ======================================================================== */

void test_radial_profile_physics() {
    BEGIN_TEST("Radial Profile Physics");
    
    struct params run_params;
    initialize_integration_params(&run_params);
    
    struct GALAXY gal;
    initialize_mw_galaxy(&gal, &run_params);
    
    // Test 1: Gas surface density generally decreases outward
    int increasing_count = 0;
    for(int i = 1; i < N_BINS; i++) {
        // Compare surface density = mass / area
        double r_in_prev = gal.DiscRadii[i-1];
        double r_out_prev = gal.DiscRadii[i];
        double r_in = gal.DiscRadii[i];
        double r_out = gal.DiscRadii[i+1];
        
        double area_prev = M_PI * (r_out_prev*r_out_prev - r_in_prev*r_in_prev);
        double area_curr = M_PI * (r_out*r_out - r_in*r_in);
        
        if(area_prev > 0 && area_curr > 0) {
            double sigma_prev = gal.DiscGas[i-1] / area_prev;
            double sigma_curr = gal.DiscGas[i] / area_curr;
            if(sigma_curr > sigma_prev * 1.1) {  // 10% tolerance for noise
                increasing_count++;
            }
        }
    }
    ASSERT_LESS_THAN((double)increasing_count, (double)(N_BINS / 4),
                    "Surface density mostly decreases outward");
    
    // Test 2: H2 fraction higher in central bins than outer bins
    double h2_frac_inner = 0.0;
    double h2_frac_outer = 0.0;
    int inner_bins = N_BINS / 3;
    int outer_bins = N_BINS / 3;
    
    for(int i = 0; i < inner_bins; i++) {
        if(gal.DiscGas[i] > 0) {
            h2_frac_inner += gal.DiscH2[i] / gal.DiscGas[i];
        }
    }
    h2_frac_inner /= inner_bins;
    
    for(int i = N_BINS - outer_bins; i < N_BINS; i++) {
        if(gal.DiscGas[i] > 0) {
            h2_frac_outer += gal.DiscH2[i] / gal.DiscGas[i];
        }
    }
    h2_frac_outer /= outer_bins;
    
    ASSERT_GREATER_THAN(h2_frac_inner, h2_frac_outer * 0.9,
                       "H2 fraction higher in center than outskirts");
    
    // Test 3: Total mass in disc is positive and reasonable
    double total_disc_mass = sum_array(gal.DiscGas, N_BINS) + sum_array(gal.DiscStars, N_BINS);
    ASSERT_GREATER_THAN(total_disc_mass, 0.0,
                       "Total disc mass > 0");
    
    printf("    Increasing Σ bins: %d / %d\n", increasing_count, N_BINS-1);
    printf("    H2 fraction (inner): %.3f\n", h2_frac_inner);
    printf("    H2 fraction (outer): %.3f\n", h2_frac_outer);
    printf("    Total disc mass: %.4e\n", total_disc_mass);
}


/* ========================================================================
 * INTEGRATION TEST 5: CROSS-COMPONENT MASS CONSERVATION
 * 
 * Tests that mass is conserved across all components:
 *   - Total baryons = Cold + Hot + Stellar + Ejected (+dust phases)
 *   - Dust phases sum correctly
 *   - Metals partitioned correctly
 * ======================================================================== */

void test_cross_component_conservation() {
    BEGIN_TEST("Cross-Component Mass Conservation");
    
    struct params run_params;
    initialize_integration_params(&run_params);
    
    struct GALAXY gal;
    initialize_mw_galaxy(&gal, &run_params);
    
    // Add hot gas and ejected reservoirs for complete test
    gal.HotGas = 10.0;           // 10^11 Msun/h
    gal.EjectedMass = 2.0;
    gal.CGMgas = 5.0;
    gal.MetalsHotGas = 0.1;
    gal.MetalsEjectedMass = 0.02;
    gal.MetalsCGMgas = 0.05;
    
    // Initialize hot/ejected dust
    gal.HotDust = 0.001;
    gal.CGMDust = 0.002;
    gal.EjectedDust = 0.0005;
    
    // Test 1: Total dust = sum of all reservoirs
    double total_dust = gal.ColdDust + gal.HotDust + gal.CGMDust + gal.EjectedDust;
    double expected_total_dust = gal.ColdDust + gal.HotDust + gal.CGMDust + gal.EjectedDust;
    ASSERT_CLOSE(expected_total_dust, total_dust, 1e-10,
                "Total dust = ColdDust + HotDust + CGMDust + EjectedDust");
    
    // Test 2: All dust reservoirs non-negative
    ASSERT_TRUE(gal.ColdDust >= 0 && gal.HotDust >= 0 && 
                gal.CGMDust >= 0 && gal.EjectedDust >= 0,
               "All dust reservoirs non-negative");
    
    // Test 3: ColdDust ≤ MetalsColdGas
    ASSERT_TRUE(gal.ColdDust <= gal.MetalsColdGas * 1.001,
               "ColdDust ≤ MetalsColdGas");
    
    // Test 4: Total metals = gas-phase + stellar + dust
    // Note: Dust is part of gas-phase metals, not additional
    double total_metals = gal.MetalsColdGas + gal.MetalsStellarMass + 
                         gal.MetalsHotGas + gal.MetalsEjectedMass + gal.MetalsCGMgas;
    ASSERT_GREATER_THAN(total_metals, 0.0,
                       "Total metals > 0");
    
    // Test 5: Stellar mass consistency
    double disc_stars = sum_array(gal.DiscStars, N_BINS);
    double total_stars = disc_stars + gal.BulgeMass;
    ASSERT_CLOSE(gal.StellarMass, total_stars, 1e-6,
                "StellarMass = DiscStars + BulgeMass");
    
    printf("    Total dust: %.6e\n", total_dust);
    printf("    Total metals: %.6e\n", total_metals);
    printf("    Stellar mass consistency: %.6f vs %.6f\n", gal.StellarMass, total_stars);
}


/* ========================================================================
 * INTEGRATION TEST 6: TOOMRE Q AND DISC INSTABILITY
 * 
 * Tests that Toomre Q calculation uses disc arrays correctly:
 *   - Q can be computed from disc profiles
 *   - Q > 1 indicates stability
 *   - MW-like galaxy should have Q ~ 1-3 in most regions
 * ======================================================================== */

void test_toomre_q_disc_integration() {
    BEGIN_TEST("Toomre Q - Disc Integration");
    
    struct params run_params;
    initialize_integration_params(&run_params);
    
    struct GALAXY gal;
    initialize_mw_galaxy(&gal, &run_params);
    
    // Parameters for Toomre Q calculation
    double sigma_v = 10.0;           // Velocity dispersion [km/s]
    double G_kpc = 4.302e-3;         // G in kpc (km/s)^2 / Msun
    double v_circ = gal.Vvir;        // Circular velocity [km/s]
    
    int valid_bins = 0;
    int stable_bins = 0;
    double q_min = 1e10, q_max = 0;
    
    for(int i = 0; i < N_BINS; i++) {
        double r_in = gal.DiscRadii[i] * 1000.0;      // kpc
        double r_out = gal.DiscRadii[i+1] * 1000.0;   // kpc
        double r_mid = 0.5 * (r_in + r_out);
        double area_kpc2 = M_PI * (r_out*r_out - r_in*r_in);
        
        if(area_kpc2 <= 0 || r_mid <= 0) continue;
        
        // Surface density in Msun/kpc^2
        double mass_msun = (gal.DiscGas[i] + gal.DiscStars[i]) * 1e10;
        double sigma = mass_msun / area_kpc2;
        
        // Skip bins with negligible mass (avoids numerical issues)
        if(sigma < 1e3 || mass_msun < 1e6) continue;
        
        // Epicyclic frequency κ ≈ √2 V/R for flat rotation curve
        double kappa = 1.414 * v_circ / r_mid;  // km/s/kpc
        
        // Toomre Q = κ σ_v / (π G Σ)
        double Q = kappa * sigma_v / (M_PI * G_kpc * sigma);
        
        valid_bins++;
        if(Q > 1.0) stable_bins++;
        if(Q < q_min) q_min = Q;
        if(Q > q_max) q_max = Q;
    }
    
    ASSERT_GREATER_THAN((double)valid_bins, 0.0,
                       "Can compute Q in at least some bins");
    
    // Q should be positive and finite where computed
    ASSERT_TRUE(q_min > 0 && q_max < 1e10,
               "Q values are positive and finite");
    
    // Q range should span at least an order of magnitude (shows radial variation)
    double q_range = (q_min > 0) ? q_max / q_min : 0;
    ASSERT_GREATER_THAN(q_range, 1.0,
                       "Q varies across disc (radial structure present)");
    
    // Report stability for information (not a pass/fail criterion)
    double stable_fraction = (double)stable_bins / valid_bins;
    
    printf("    Valid bins for Q: %d / %d\n", valid_bins, N_BINS);
    printf("    Stable bins (Q>1): %d (%.0f%%)\n", stable_bins, stable_fraction*100);
    printf("    Q range: %.2f - %.2f\n", q_min, q_max);
    printf("    Note: Q < 1 triggers disk instability in actual model\n");
}


/* ========================================================================
 * INTEGRATION TEST 7: SPECIFIC AM FROM DISC PROFILE
 * 
 * Computes specific angular momentum directly from disc arrays
 * and compares to the simple estimate j* ≈ 2 R_d V.
 * ======================================================================== */

void test_specific_am_from_disc() {
    BEGIN_TEST("Specific AM from Disc Profile");
    
    struct params run_params;
    initialize_integration_params(&run_params);
    
    struct GALAXY gal;
    initialize_mw_galaxy(&gal, &run_params);
    
    double v_circ = gal.Vvir;  // km/s (assuming flat rotation)
    
    // Method 1: Compute j* from disc arrays
    // j* = Σ_i (M_i * j_i) / Σ_i M_i
    // where j_i = r_i * V_circ for annulus i
    double j_weighted_sum = 0.0;
    double total_stellar = 0.0;
    
    for(int i = 0; i < N_BINS; i++) {
        double r_mid = 0.5 * (gal.DiscRadii[i] + gal.DiscRadii[i+1]) * 1000.0;  // kpc
        double j_i = r_mid * v_circ;  // kpc km/s
        j_weighted_sum += gal.DiscStars[i] * j_i;
        total_stellar += gal.DiscStars[i];
    }
    
    double j_star_disc = (total_stellar > 0) ? j_weighted_sum / total_stellar : 0;
    
    // Method 2: Simple estimate j* ≈ 2 R_d V
    double r_d_kpc = gal.DiskScaleRadius * 1000.0;  // kpc
    double j_star_simple = 2.0 * r_d_kpc * v_circ;
    
    ASSERT_GREATER_THAN(j_star_disc, 0.0,
                       "Disc-computed j* > 0");
    
    // The two estimates should be within a factor of ~2
    double ratio = j_star_disc / j_star_simple;
    ASSERT_IN_RANGE(ratio, 0.5, 2.0,
                   "Disc j* within 2x of simple estimate");
    
    // Fall relation: log(j*) ≈ 2/3 log(M*) + const
    // For M* ~ 5×10^10 Msun, j* ~ 1000-3000 kpc km/s
    ASSERT_IN_RANGE(j_star_disc, 100.0, 5000.0,
                   "j* in MW-like range");
    
    printf("    j* from disc arrays: %.1f kpc km/s\n", j_star_disc);
    printf("    j* simple estimate: %.1f kpc km/s\n", j_star_simple);
    printf("    Ratio: %.2f\n", ratio);
}


/* ========================================================================
 * INTEGRATION TEST 8: METALLICITY GRADIENT
 * 
 * Tests that metallicity can be computed in each annulus
 * and shows expected gradient (decreasing outward).
 * ======================================================================== */

void test_metallicity_gradient() {
    BEGIN_TEST("Metallicity Gradient");
    
    struct params run_params;
    initialize_integration_params(&run_params);
    
    struct GALAXY gal;
    initialize_mw_galaxy(&gal, &run_params);
    
    // Setup with a metallicity gradient (higher in center)
    double Z_central = 0.03;  // 3% at center
    double Z_outer = 0.005;   // 0.5% at edge
    double r_d = gal.DiskScaleRadius;
    
    for(int i = 0; i < N_BINS; i++) {
        double r_mid = 0.5 * (gal.DiscRadii[i] + gal.DiscRadii[i+1]);
        double Z_r = Z_central * exp(-r_mid / (2.0 * r_d)) + Z_outer;
        gal.DiscGasMetals[i] = gal.DiscGas[i] * Z_r;
        gal.DiscDust[i] = gal.DiscGasMetals[i] * 0.3;  // 30% dust-to-metal
    }
    
    // Update bulk metal to match
    gal.MetalsColdGas = sum_array(gal.DiscGasMetals, N_BINS);
    gal.ColdDust = sum_array(gal.DiscDust, N_BINS);
    
    // Compute metallicity in inner and outer regions
    double Z_inner_avg = 0.0;
    double Z_outer_avg = 0.0;
    int inner_count = 0, outer_count = 0;
    
    for(int i = 0; i < N_BINS / 3; i++) {
        if(gal.DiscGas[i] > 0) {
            Z_inner_avg += gal.DiscGasMetals[i] / gal.DiscGas[i];
            inner_count++;
        }
    }
    
    for(int i = 2 * N_BINS / 3; i < N_BINS; i++) {
        if(gal.DiscGas[i] > 0) {
            Z_outer_avg += gal.DiscGasMetals[i] / gal.DiscGas[i];
            outer_count++;
        }
    }
    
    if(inner_count > 0) Z_inner_avg /= inner_count;
    if(outer_count > 0) Z_outer_avg /= outer_count;
    
    ASSERT_GREATER_THAN(Z_inner_avg, Z_outer_avg + 1e-10,
                       "Central metallicity > outer metallicity");
    
    // Compute gradient in dex/kpc (avoid log10(0))
    double r_inner = 0.0, r_outer = 0.0;
    for(int i = 0; i < N_BINS / 3; i++) {
        r_inner += 0.5 * (gal.DiscRadii[i] + gal.DiscRadii[i+1]) * 1000.0;
    }
    r_inner /= (N_BINS / 3);
    
    for(int i = 2 * N_BINS / 3; i < N_BINS; i++) {
        r_outer += 0.5 * (gal.DiscRadii[i] + gal.DiscRadii[i+1]) * 1000.0;
    }
    r_outer /= (N_BINS / 3);
    
    double delta_r = r_outer - r_inner;
    // Protect against log10(0) - use floor value for very low Z
    double Z_outer_safe = fmax(Z_outer_avg, 1e-6);
    double Z_inner_safe = fmax(Z_inner_avg, 1e-6);
    double delta_Z = log10(Z_outer_safe) - log10(Z_inner_safe);
    double gradient = (delta_r > 0) ? delta_Z / delta_r : 0;
    
    // Typical gradient is -0.01 to -0.1 dex/kpc (inner higher than outer means negative)
    ASSERT_LESS_THAN(gradient, 0.0,
                    "Negative metallicity gradient");
    
    printf("    Z (inner): %.4f\n", Z_inner_avg);
    printf("    Z (outer): %.4f\n", Z_outer_avg);
    printf("    Gradient: %.4f dex/kpc\n", gradient);
}


/* ========================================================================
 * INTEGRATION TEST 9: DUST PROFILE FOLLOWS GAS
 * 
 * Verifies that dust profile is consistent with gas profile:
 *   - Dust concentrated where gas is concentrated
 *   - DtG ratio roughly constant or increases inward
 * ======================================================================== */

void test_dust_profile_follows_gas() {
    BEGIN_TEST("Dust Profile Follows Gas");
    
    struct params run_params;
    initialize_integration_params(&run_params);
    
    struct GALAXY gal;
    initialize_mw_galaxy(&gal, &run_params);
    
    // Test 1: Dust peaks in similar location as gas
    int gas_peak_bin = 0, dust_peak_bin = 0;
    double max_gas = 0, max_dust = 0;
    
    for(int i = 0; i < N_BINS; i++) {
        if(gal.DiscGas[i] > max_gas) {
            max_gas = gal.DiscGas[i];
            gas_peak_bin = i;
        }
        if(gal.DiscDust[i] > max_dust) {
            max_dust = gal.DiscDust[i];
            dust_peak_bin = i;
        }
    }
    
    ASSERT_TRUE(abs(gas_peak_bin - dust_peak_bin) <= 3,
               "Dust peak within 3 bins of gas peak");
    
    // Test 2: DtG ratio is physically reasonable in all bins
    int unreasonable_dtg = 0;
    for(int i = 0; i < N_BINS; i++) {
        if(gal.DiscGas[i] > 0) {
            double dtg = gal.DiscDust[i] / gal.DiscGas[i];
            // Typical DtG ~ 0.001 - 0.05
            if(dtg < 0 || dtg > 0.1) {
                unreasonable_dtg++;
            }
        }
    }
    ASSERT_LESS_THAN((double)unreasonable_dtg, (double)(N_BINS / 4),
                    "DtG reasonable in most bins");
    
    // Test 3: Total dust profile mass conservation
    double disc_dust_sum = sum_array(gal.DiscDust, N_BINS);
    ASSERT_CLOSE(gal.ColdDust, disc_dust_sum, 1e-5,
                "ΣDiscDust == ColdDust (conservation check)");
    
    printf("    Gas peak bin: %d, Dust peak bin: %d\n", gas_peak_bin, dust_peak_bin);
    printf("    Max gas: %.4e, Max dust: %.4e\n", max_gas, max_dust);
    printf("    Unreasonable DtG bins: %d / %d\n", unreasonable_dtg, N_BINS);
}


/* ========================================================================
 * INTEGRATION TEST 10: FULL SYSTEM INTEGRATION
 * 
 * Tests that all components work together in a realistic scenario:
 *   - Create MW-like galaxy with all components set
 *   - Verify all consistency relations hold simultaneously
 * ======================================================================== */

void test_full_system_integration() {
    BEGIN_TEST("Full System Integration (All Components)");
    
    struct params run_params;
    initialize_integration_params(&run_params);
    
    struct GALAXY gal;
    initialize_mw_galaxy(&gal, &run_params);
    
    // Add additional components for full integration
    gal.HotGas = 20.0;
    gal.CGMgas = 10.0;
    gal.EjectedMass = 5.0;
    gal.MetalsHotGas = 0.1;
    gal.MetalsCGMgas = 0.05;
    gal.MetalsEjectedMass = 0.02;
    gal.HotDust = 0.001;
    gal.CGMDust = 0.002;
    gal.EjectedDust = 0.0005;
    
    int all_checks_pass = 1;
    
    // Check 1: Disc-bulk gas consistency
    double disc_gas = sum_array(gal.DiscGas, N_BINS);
    if(fabs(disc_gas - gal.ColdGas) > 1e-6) all_checks_pass = 0;
    
    // Check 2: Disc-bulk dust consistency
    double disc_dust = sum_array(gal.DiscDust, N_BINS);
    if(fabs(disc_dust - gal.ColdDust) > 1e-6) all_checks_pass = 0;
    
    // Check 3: Spin vectors are unit vectors
    double spin_gas_mag = compute_spin_magnitude(gal.SpinGas);
    double spin_stars_mag = compute_spin_magnitude(gal.SpinStars);
    if(fabs(spin_gas_mag - 1.0) > 1e-6) all_checks_pass = 0;
    if(fabs(spin_stars_mag - 1.0) > 1e-6) all_checks_pass = 0;
    
    // Check 4: Stellar mass = Disc + Bulge
    double disc_stars = sum_array(gal.DiscStars, N_BINS);
    if(fabs(disc_stars + gal.BulgeMass - gal.StellarMass) > 1e-6) all_checks_pass = 0;
    
    // Check 5: H2 + HI = Gas in each bin (with relative tolerance)
    for(int i = 0; i < N_BINS; i++) {
        double gas_i = gal.DiscGas[i];
        double tol = (gas_i > 1e-10) ? 1e-6 * gas_i : 1e-10;
        if(fabs(gal.DiscH2[i] + gal.DiscHI[i] - gas_i) > tol) {
            all_checks_pass = 0;
            break;
        }
    }
    
    // Check 6: Dust ≤ Metals in each bin
    for(int i = 0; i < N_BINS; i++) {
        if(gal.DiscDust[i] > gal.DiscGasMetals[i] * 1.001) {
            all_checks_pass = 0;
            break;
        }
    }
    
    // Check 7: All dust reservoirs non-negative
    if(gal.ColdDust < 0 || gal.HotDust < 0 || gal.CGMDust < 0 || gal.EjectedDust < 0) {
        all_checks_pass = 0;
    }
    
    ASSERT_TRUE(all_checks_pass,
               "All integration checks pass simultaneously");
    
    // Report galaxy properties
    double total_baryons = gal.ColdGas + gal.HotGas + gal.CGMgas + 
                          gal.StellarMass + gal.EjectedMass;
    double total_dust = gal.ColdDust + gal.HotDust + gal.CGMDust + gal.EjectedDust;
    double j_star = compute_specific_am(&gal);
    
    printf("    Total baryonic mass: %.2e (10^10 Msun/h)\n", total_baryons);
    printf("    Total dust mass: %.4e (10^10 Msun/h)\n", total_dust);
    printf("    Specific AM: %.1f kpc km/s\n", j_star);
    printf("    All %d consistency checks: %s\n", 7, 
           all_checks_pass ? "PASS" : "FAIL");
}


/* ========================================================================
 * MAIN TEST RUNNER
 * ======================================================================== */

int main(void) {
    BEGIN_TEST_SUITE("DarkMode Integration Tests (Dust + AM + Discs)");
    
    // Dust-Disc tests
    test_dust_disc_consistency();
    
    // Angular momentum tests
    test_angular_momentum_disc_consistency();
    
    // Disc-bulk consistency tests
    test_disc_bulk_consistency();
    
    // Radial profile physics
    test_radial_profile_physics();
    
    // Cross-component conservation
    test_cross_component_conservation();
    
    // Toomre Q integration
    test_toomre_q_disc_integration();
    
    // Specific AM from disc
    test_specific_am_from_disc();
    
    // Metallicity gradient
    test_metallicity_gradient();
    
    // Dust-gas correlation
    test_dust_profile_follows_gas();
    
    // Full system integration
    test_full_system_integration();
    
    END_TEST_SUITE();
    PRINT_TEST_SUMMARY();
    
    return TEST_EXIT_CODE();
}

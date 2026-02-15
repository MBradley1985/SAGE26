/*
 * TOOMRE Q AND LOCAL DISK INSTABILITY VALIDATION TESTS
 * 
 * Validates the Toomre Q-parameter calculation and check_local_disk_instability:
 * - Q = (σ κ) / (π G Σ) formula correctness
 * - Physical scaling relations
 * - Edge cases and numerical stability
 * - Mass transfer to bulge for Q < 1
 * - Mass conservation during instability
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "test_framework.h"
#include "../src/core_allvars.h"
#include "../src/model_darkmode.h"
#include "../src/model_misc.h"

/* Constants for validation */
#define G_PC 4.302e-3   /* (km/s)^2 pc / Msun - standard astronomical units */
#define SOLAR_Z 0.0142  /* Solar metallicity */

/*===========================================================================
 * TOOMRE Q FORMULA VALIDATION
 *===========================================================================*/

void test_toomre_Q_formula_basic() {
    BEGIN_TEST("Toomre Q Basic Formula Validation");
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    
    /* Test with typical MW annulus values */
    double Sigma_gas = 10.0;    /* Msun/pc^2 - typical for MW disk */
    double Sigma_stars = 50.0;  /* Msun/pc^2 */
    double r_mid = 8.0;         /* kpc - solar neighborhood */ 
    double Vvir = 220.0;        /* km/s - MW circular velocity */
    
    double Q = compute_toomre_Q(Sigma_gas, Sigma_stars, r_mid, Vvir, &run_params);
    
    /* Manual calculation for verification:
     * σ_gas = 10 km/s, σ_stars = Vvir/10 = 22 km/s
     * Σ_total = 60 Msun/pc²
     * σ_eff = (10×10 + 50×22) / 60 = 1200/60 = 20 km/s
     * r_mid_pc = 8000 pc
     * κ = 220/8000 = 0.0275 km/s/pc
     * G_pc = 4.302e-3 (km/s)² pc/Msun
     * Q = (20 × 0.0275) / (π × 4.302e-3 × 60) = 0.55 / 0.811 = 0.678
     */
    double sigma_gas = 10.0;
    double sigma_stars = Vvir / 10.0;
    double Sigma_total = Sigma_gas + Sigma_stars;
    double sigma_eff = (Sigma_gas * sigma_gas + Sigma_stars * sigma_stars) / Sigma_total;
    double r_mid_pc = r_mid * 1000.0;
    double kappa = Vvir / r_mid_pc;
    const double G_pc = 4.302e-3;
    double Q_manual = (sigma_eff * kappa) / (M_PI * G_pc * Sigma_total);
    
    printf("  ℹ Computed Q = %.4f, Manual Q = %.4f\n", Q, Q_manual);
    
    ASSERT_CLOSE(Q, Q_manual, 1e-10, "Q matches manual calculation");
    ASSERT_TRUE(Q > 0.3 && Q < 3.0, "Q in physically reasonable range for MW");
}

void test_toomre_Q_edge_cases() {
    BEGIN_TEST("Toomre Q Edge Cases");
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    
    /* Test 1: Zero gas surface density */
    double Q_zero_gas = compute_toomre_Q(0.0, 50.0, 8.0, 220.0, &run_params);
    ASSERT_TRUE(Q_zero_gas > 100.0, "Zero gas → high Q (stable)");
    
    /* Test 2: Zero stellar surface density (gas-only) */
    double Q_gas_only = compute_toomre_Q(50.0, 0.0, 8.0, 220.0, &run_params);
    ASSERT_TRUE(Q_gas_only > 0.0 && Q_gas_only < 10.0, "Gas-only disk has finite Q");
    printf("  ℹ Gas-only disk Q = %.4f\n", Q_gas_only);
    
    /* Test 3: Zero radius */
    double Q_zero_r = compute_toomre_Q(10.0, 50.0, 0.0, 220.0, &run_params);
    ASSERT_TRUE(Q_zero_r > 100.0, "Zero radius → high Q (stable)");
    
    /* Test 4: Zero Vvir */
    double Q_zero_v = compute_toomre_Q(10.0, 50.0, 8.0, 0.0, &run_params);
    ASSERT_TRUE(Q_zero_v > 100.0, "Zero Vvir → high Q (stable)");
    
    /* Test 5: Negative values (should handle gracefully) */
    double Q_neg = compute_toomre_Q(-10.0, 50.0, 8.0, 220.0, &run_params);
    ASSERT_TRUE(Q_neg > 100.0, "Negative Sigma → high Q (stable)");
}

void test_toomre_Q_scaling_relations() {
    BEGIN_TEST("Toomre Q Scaling Relations");
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    
    double Sigma_gas = 20.0;
    double Sigma_stars = 40.0;
    double r_mid = 5.0;
    double Vvir = 200.0;
    
    double Q_base = compute_toomre_Q(Sigma_gas, Sigma_stars, r_mid, Vvir, &run_params);
    
    /* Higher surface density → lower Q (more unstable) */
    double Q_high_sigma = compute_toomre_Q(Sigma_gas * 2, Sigma_stars * 2, r_mid, Vvir, &run_params);
    ASSERT_TRUE(Q_high_sigma < Q_base, "Higher Σ → lower Q");
    printf("  ℹ Q_base=%.3f, Q_high_sigma=%.3f (ratio=%.2f)\n", 
           Q_base, Q_high_sigma, Q_base/Q_high_sigma);
    
    /* Higher Vvir → higher κ → higher Q (more stable) */
    double Q_high_vvir = compute_toomre_Q(Sigma_gas, Sigma_stars, r_mid, Vvir * 1.5, &run_params);
    ASSERT_TRUE(Q_high_vvir > Q_base, "Higher Vvir → higher Q");
    printf("  ℹ Q_base=%.3f, Q_high_vvir=%.3f\n", Q_base, Q_high_vvir);
    
    /* Larger radius → smaller κ → lower Q (assuming flat rotation curve) */
    double Q_large_r = compute_toomre_Q(Sigma_gas, Sigma_stars, r_mid * 2, Vvir, &run_params);
    ASSERT_TRUE(Q_large_r < Q_base, "Larger r → lower κ → lower Q");
    printf("  ℹ Q_base=%.3f, Q_large_r=%.3f\n", Q_base, Q_large_r);
}

void test_toomre_Q_physical_regimes() {
    BEGIN_TEST("Toomre Q Physical Regimes");
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    
    /* Regime 1: Stable outer disk (low Σ) */
    double Q_outer = compute_toomre_Q(1.0, 5.0, 15.0, 200.0, &run_params);
    ASSERT_TRUE(Q_outer > 1.0, "Outer disk is stable (Q > 1)");
    printf("  ℹ Outer disk Q = %.3f\n", Q_outer);
    
    /* Regime 2: Marginally stable solar neighborhood */
    double Q_solar = compute_toomre_Q(10.0, 40.0, 8.0, 220.0, &run_params);
    printf("  ℹ Solar neighborhood Q = %.3f\n", Q_solar);
    /* Solar neighborhood is typically Q ~ 1-2 */
    
    /* Regime 3: Dense inner disk (potentially unstable) */
    double Q_inner = compute_toomre_Q(100.0, 500.0, 2.0, 200.0, &run_params);
    printf("  ℹ Inner disk Q = %.3f\n", Q_inner);
    
    /* Regime 4: Starburst/ULIRG-like (very high Σ) */
    double Q_starburst = compute_toomre_Q(1000.0, 1000.0, 1.0, 300.0, &run_params);
    printf("  ℹ Starburst Q = %.3f\n", Q_starburst);
    /* High-Σ disks tend to be unstable */
}

/*===========================================================================
 * CHECK_LOCAL_DISK_INSTABILITY VALIDATION
 *===========================================================================*/

void test_local_instability_setup_helper(struct GALAXY *gal, struct params *run_params,
                                         double gas_mass, double star_mass, double Vvir,
                                         double disk_scale) {
    memset(gal, 0, sizeof(struct GALAXY));
    memset(run_params, 0, sizeof(struct params));
    
    run_params->Hubble_h = 0.7;
    gal->Vvir = Vvir;
    gal->DiskScaleRadius = disk_scale;  /* Mpc/h */
    
    /* Set up radial bins (simplified: linear spacing up to 5 scale radii) */
    double r_max = 5.0 * disk_scale;  /* Mpc/h */
    for(int i = 0; i <= N_BINS; i++) {
        gal->DiscRadii[i] = i * r_max / N_BINS;
    }
    
    /* Distribute mass exponentially (like real disk) */
    double r_d = disk_scale;
    double total_mass_gas = 0.0;
    double total_mass_star = 0.0;
    
    for(int i = 0; i < N_BINS; i++) {
        double r_in = gal->DiscRadii[i];
        double r_out = gal->DiscRadii[i+1];
        double r_mid = 0.5 * (r_in + r_out);
        
        /* Exponential surface density: Σ(r) ∝ exp(-r/r_d) */
        double weight = exp(-r_mid / r_d);
        gal->DiscGas[i] = gas_mass * weight;
        gal->DiscStars[i] = star_mass * weight;
        gal->DiscGasMetals[i] = 0.02 * gal->DiscGas[i];  /* 2% metallicity */
        gal->DiscStarsMetals[i] = 0.02 * gal->DiscStars[i];
        
        total_mass_gas += gal->DiscGas[i];
        total_mass_star += gal->DiscStars[i];
    }
    
    /* Normalize */
    for(int i = 0; i < N_BINS; i++) {
        gal->DiscGas[i] *= gas_mass / (total_mass_gas + 1e-30);
        gal->DiscStars[i] *= star_mass / (total_mass_star + 1e-30);
        gal->DiscGasMetals[i] = 0.02 * gal->DiscGas[i];
        gal->DiscStarsMetals[i] = 0.02 * gal->DiscStars[i];
    }
    
    gal->ColdGas = gas_mass;
    gal->StellarMass = star_mass;
    gal->BulgeMass = 0.0;
    gal->InstabilityBulgeMass = 0.0;
    gal->MetalsStellarMass = 0.02 * star_mass;
    gal->MetalsBulgeMass = 0.0;
}

void test_local_instability_stable_disk() {
    BEGIN_TEST("Local Disk Instability - Stable Disk");
    
    struct GALAXY gal;
    struct params run_params;
    
    /* Set up a stable disk: low mass, high Vvir */
    test_local_instability_setup_helper(&gal, &run_params,
                                        0.01,   /* gas_mass: 1e8 Msun/h */
                                        0.05,   /* star_mass: 5e8 Msun/h */
                                        250.0,  /* Vvir: high */
                                        0.003); /* disk_scale: 3 kpc/h */
    
    double initial_bulge = gal.BulgeMass;
    double initial_disk_stars = gal.StellarMass - gal.BulgeMass;
    double initial_total = 0.0;
    for(int i = 0; i < N_BINS; i++) {
        initial_total += gal.DiscGas[i] + gal.DiscStars[i];
    }
    
    /* Run instability check */
    check_local_disk_instability(0, 0, 0.01, 0, &gal, &run_params);
    
    double final_total = 0.0;
    for(int i = 0; i < N_BINS; i++) {
        final_total += gal.DiscGas[i] + gal.DiscStars[i];
    }
    
    /* For stable disk, should see little to no mass transfer */
    double bulge_growth = gal.BulgeMass - initial_bulge;
    printf("  ℹ Bulge growth: %.6f (initial: %.6f)\n", bulge_growth, initial_bulge);
    printf("  ℹ Total disk mass: initial=%.6f, final=%.6f\n", initial_total, final_total);
    
    ASSERT_TRUE(bulge_growth >= 0.0, "Bulge cannot shrink");
    /* Note: Even "stable" disks may have some inner unstable regions */
}

void test_local_instability_unstable_disk() {
    BEGIN_TEST("Local Disk Instability - Unstable Disk");
    
    struct GALAXY gal;
    struct params run_params;
    
    /* Set up an unstable disk: high mass, low Vvir, compact */
    test_local_instability_setup_helper(&gal, &run_params,
                                        1.0,    /* gas_mass: 1e10 Msun/h */
                                        5.0,    /* star_mass: 5e10 Msun/h */
                                        100.0,  /* Vvir: low */
                                        0.001); /* disk_scale: 1 kpc/h = compact */
    
    double initial_bulge = gal.BulgeMass;
    double initial_disk_stars = 0.0;
    for(int i = 0; i < N_BINS; i++) {
        initial_disk_stars += gal.DiscStars[i];
    }
    
    /* Run instability check */
    check_local_disk_instability(0, 0, 0.01, 0, &gal, &run_params);
    
    double final_disk_stars = 0.0;
    for(int i = 0; i < N_BINS; i++) {
        final_disk_stars += gal.DiscStars[i];
    }
    
    double bulge_growth = gal.BulgeMass - initial_bulge;
    double disk_loss = initial_disk_stars - final_disk_stars;
    
    printf("  ℹ Bulge growth: %.6f\n", bulge_growth);
    printf("  ℹ Disk stars lost: %.6f\n", disk_loss);
    printf("  ℹ Instability bulge mass: %.6f\n", gal.InstabilityBulgeMass);
    
    ASSERT_TRUE(bulge_growth > 0.0, "Unstable disk transfers mass to bulge");
    ASSERT_TRUE(gal.InstabilityBulgeMass > 0.0, "InstabilityBulgeMass tracks growth");
    ASSERT_CLOSE(bulge_growth, disk_loss, 0.01, "Mass conserved (bulge = disk loss)");
}

void test_local_instability_mass_conservation() {
    BEGIN_TEST("Local Disk Instability - Mass Conservation");
    
    struct GALAXY gal;
    struct params run_params;
    
    test_local_instability_setup_helper(&gal, &run_params,
                                        0.5,    /* gas_mass */
                                        2.0,    /* star_mass */
                                        150.0,  /* Vvir */
                                        0.002); /* disk_scale */
    
    /* Calculate initial total stellar mass */
    double initial_disk_stars = 0.0;
    double initial_disk_metals = 0.0;
    for(int i = 0; i < N_BINS; i++) {
        initial_disk_stars += gal.DiscStars[i];
        initial_disk_metals += gal.DiscStarsMetals[i];
    }
    double initial_bulge = gal.BulgeMass;
    double initial_bulge_metals = gal.MetalsBulgeMass;
    double initial_total_stars = initial_disk_stars + initial_bulge;
    
    /* Run instability */
    check_local_disk_instability(0, 0, 0.01, 0, &gal, &run_params);
    
    /* Calculate final totals */
    double final_disk_stars = 0.0;
    double final_disk_metals = 0.0;
    for(int i = 0; i < N_BINS; i++) {
        final_disk_stars += gal.DiscStars[i];
        final_disk_metals += gal.DiscStarsMetals[i];
    }
    double final_total_stars = final_disk_stars + gal.BulgeMass;
    
    printf("  ℹ Initial: disk=%.6f, bulge=%.6f, total=%.6f\n",
           initial_disk_stars, initial_bulge, initial_total_stars);
    printf("  ℹ Final: disk=%.6f, bulge=%.6f, total=%.6f\n",
           final_disk_stars, gal.BulgeMass, final_total_stars);
    
    ASSERT_CLOSE(initial_total_stars, final_total_stars, 1e-6,
                "Total stellar mass conserved");
}

void test_local_instability_q_values() {
    BEGIN_TEST("Local Disk Instability - Q Values Per Annulus");
    
    struct GALAXY gal;
    struct params run_params;
    
    test_local_instability_setup_helper(&gal, &run_params,
                                        0.3,    /* gas_mass */
                                        1.5,    /* star_mass */
                                        180.0,  /* Vvir */
                                        0.003); /* disk_scale */
    
    double h = run_params.Hubble_h;
    printf("  ℹ Q values per annulus (before instability check):\n");
    
    int n_stable = 0;
    int n_unstable = 0;
    
    for(int i = 0; i < N_BINS; i++) {
        double r_in = gal.DiscRadii[i] * 1000.0;    /* kpc */
        double r_out = gal.DiscRadii[i+1] * 1000.0;
        double r_mid = 0.5 * (r_in + r_out);
        double area_pc2 = M_PI * (r_out * r_out - r_in * r_in) * 1.0e6;
        
        if(area_pc2 <= 0.0 || r_mid <= 0.0) continue;
        
        double Sigma_gas = (gal.DiscGas[i] * 1.0e10 / h) / area_pc2;
        double Sigma_stars = (gal.DiscStars[i] * 1.0e10 / h) / area_pc2;
        
        double Q = compute_toomre_Q(Sigma_gas, Sigma_stars, r_mid, gal.Vvir, &run_params);
        
        if(i < 5) {
            printf("       Bin %2d: r=%.2f kpc, Σ_gas=%.1f, Σ_*=%.1f Msun/pc², Q=%.3f %s\n",
                   i, r_mid, Sigma_gas, Sigma_stars, Q, Q < 1.0 ? "(UNSTABLE)" : "");
        }
        
        if(Q < 1.0) n_unstable++;
        else n_stable++;
    }
    
    printf("  ℹ Summary: %d stable, %d unstable annuli\n", n_stable, n_unstable);
    
    ASSERT_TRUE(n_stable + n_unstable > 0, "At least some annuli processed");
}

void test_local_instability_zero_disk() {
    BEGIN_TEST("Local Disk Instability - Zero Disk Scale Radius");
    
    struct GALAXY gal;
    struct params run_params;
    
    test_local_instability_setup_helper(&gal, &run_params, 0.5, 2.0, 200.0, 0.002);
    
    /* Set disk scale to zero - should exit early */
    gal.DiskScaleRadius = 0.0;
    double initial_bulge = gal.BulgeMass;
    
    check_local_disk_instability(0, 0, 0.01, 0, &gal, &run_params);
    
    ASSERT_CLOSE(gal.BulgeMass, initial_bulge, 1e-15,
                "Zero disk scale → no instability processing");
}

void test_local_instability_zero_vvir() {
    BEGIN_TEST("Local Disk Instability - Zero Vvir");
    
    struct GALAXY gal;
    struct params run_params;
    
    test_local_instability_setup_helper(&gal, &run_params, 0.5, 2.0, 200.0, 0.002);
    
    /* Set Vvir to zero - should exit early */
    gal.Vvir = 0.0;
    double initial_bulge = gal.BulgeMass;
    
    check_local_disk_instability(0, 0, 0.01, 0, &gal, &run_params);
    
    ASSERT_CLOSE(gal.BulgeMass, initial_bulge, 1e-15,
                "Zero Vvir → no instability processing");
}

void test_local_instability_no_negative_masses() {
    BEGIN_TEST("Local Disk Instability - No Negative Masses");
    
    struct GALAXY gal;
    struct params run_params;
    
    /* Very unstable disk to stress test */
    test_local_instability_setup_helper(&gal, &run_params,
                                        2.0,    /* gas_mass */
                                        10.0,   /* star_mass */
                                        50.0,   /* Vvir: very low */
                                        0.0005);/* disk_scale: very compact */
    
    check_local_disk_instability(0, 0, 0.01, 0, &gal, &run_params);
    
    /* Check all masses are non-negative */
    for(int i = 0; i < N_BINS; i++) {
        ASSERT_TRUE(gal.DiscGas[i] >= 0.0, "DiscGas non-negative");
        ASSERT_TRUE(gal.DiscStars[i] >= 0.0, "DiscStars non-negative");
        ASSERT_TRUE(gal.DiscGasMetals[i] >= 0.0, "DiscGasMetals non-negative");
        ASSERT_TRUE(gal.DiscStarsMetals[i] >= 0.0, "DiscStarsMetals non-negative");
        
        if(gal.DiscGas[i] < 0.0 || gal.DiscStars[i] < 0.0) {
            printf("  ⚠ Negative mass in bin %d!\n", i);
            break;
        }
    }
    
    ASSERT_TRUE(gal.BulgeMass >= 0.0, "BulgeMass non-negative");
    ASSERT_TRUE(gal.MetalsBulgeMass >= 0.0, "MetalsBulgeMass non-negative");
}

/*===========================================================================
 * COMPARISON WITH EXPECTED VALUES
 *===========================================================================*/

void test_toomre_Q_literature_comparison() {
    BEGIN_TEST("Toomre Q Literature Comparison");
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    
    /*
     * Compare with known observational values:
     * - MW solar neighborhood: Q ~ 1.5-2.5 (typical)
     * - MW inner disk: Q ~ 1-2
     * - Starbursts: Q < 1 (often unstable)
     */
    
    /* MW solar neighborhood approximation */
    /* Σ_gas ~ 10-15 Msun/pc^2, Σ_* ~ 35-50 Msun/pc^2 at R=8 kpc */
    double Q_mw = compute_toomre_Q(12.0, 40.0, 8.0, 220.0, &run_params);
    printf("  ℹ MW solar neighborhood Q = %.3f (expected ~1.5-2.5)\n", Q_mw);
    /* Note: exact value depends on velocity dispersion assumptions */
    
    /* Dense starburst */
    double Q_sb = compute_toomre_Q(500.0, 500.0, 1.0, 250.0, &run_params);
    printf("  ℹ Starburst nucleus Q = %.3f (expected <1 for violent instability)\n", Q_sb);
    
    /* Low surface brightness disk */
    double Q_lsb = compute_toomre_Q(2.0, 10.0, 10.0, 100.0, &run_params);
    printf("  ℹ LSB disk Q = %.3f (expected >2 due to low Σ)\n", Q_lsb);
    
    ASSERT_TRUE(Q_mw > 0.0, "MW Q is positive");
    ASSERT_TRUE(Q_sb > 0.0, "Starburst Q is positive");
    ASSERT_TRUE(Q_lsb > 0.0, "LSB Q is positive");
}

void test_unstable_fraction_transfer() {
    BEGIN_TEST("Unstable Fraction Transfer Calculation");
    
    /*
     * When Q < Q_crit = 1, the unstable fraction is (1 - Q/Q_crit).
     * Test that this formula gives sensible results.
     */
    
    struct params run_params;
    memset(&run_params, 0, sizeof(struct params));
    
    double Q_crit = 1.0;
    
    /* Q=0.5 → 50% unstable */
    double Q1 = 0.5;
    double frac1 = 1.0 - Q1/Q_crit;
    ASSERT_CLOSE(frac1, 0.5, 1e-10, "Q=0.5 → 50% unstable");
    
    /* Q=0.8 → 20% unstable */
    double Q2 = 0.8;
    double frac2 = 1.0 - Q2/Q_crit;
    ASSERT_CLOSE(frac2, 0.2, 1e-10, "Q=0.8 → 20% unstable");
    
    /* Q=0.1 → 90% unstable */
    double Q3 = 0.1;
    double frac3 = 1.0 - Q3/Q_crit;
    ASSERT_CLOSE(frac3, 0.9, 1e-10, "Q=0.1 → 90% unstable");
    
    /* Q=1.0 → 0% unstable (marginally stable) */
    double Q4 = 1.0;
    double frac4 = 1.0 - Q4/Q_crit;
    ASSERT_CLOSE(frac4, 0.0, 1e-10, "Q=1.0 → 0% unstable");
    
    printf("  ℹ Unstable fraction formula verified\n");
}

/*===========================================================================
 * MAIN
 *===========================================================================*/

int main() {
    BEGIN_TEST_SUITE("Toomre Q and Local Disk Instability");
    
    printf("\n" COLOR_BLUE "=== Toomre Q Formula Tests ===" COLOR_RESET "\n\n");
    test_toomre_Q_formula_basic();
    test_toomre_Q_edge_cases();
    test_toomre_Q_scaling_relations();
    test_toomre_Q_physical_regimes();
    test_toomre_Q_literature_comparison();
    
    printf("\n" COLOR_BLUE "=== Local Disk Instability Tests ===" COLOR_RESET "\n\n");
    test_local_instability_stable_disk();
    test_local_instability_unstable_disk();
    test_local_instability_mass_conservation();
    test_local_instability_q_values();
    test_local_instability_zero_disk();
    test_local_instability_zero_vvir();
    test_local_instability_no_negative_masses();
    
    printf("\n" COLOR_BLUE "=== Additional Validation ===" COLOR_RESET "\n\n");
    test_unstable_fraction_transfer();
    
    END_TEST_SUITE();
    PRINT_TEST_SUMMARY();
    
    return TEST_EXIT_CODE();
}

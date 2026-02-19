#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "core_allvars.h"
#include "model_darkmode.h"
#include "model_misc.h"

#define HYDROGEN_MASS_FRAC 0.74

double compute_local_star_formation(const int p, const double dt,
                                   struct GALAXY *galaxies, const struct params *run_params,
                                   double sfr_local[N_BINS], double h2_local[N_BINS])
{
    // Initialize outputs
    for(int i = 0; i < N_BINS; i++) {
        sfr_local[i] = 0.0;
        h2_local[i] = 0.0;
    }
    
    // Safety checks
    if(galaxies[p].Vvir <= 0.0 || galaxies[p].DiskScaleRadius <= 0.0 || dt <= 0.0) {
        return 0.0;
    }
    
    const float h = run_params->Hubble_h;
    const float rs_pc = galaxies[p].DiskScaleRadius * 1.0e6 / h;  // Scale radius in pc
    
    if(rs_pc <= 0.0) {
        return 0.0;
    }
    
    // Local dynamical time for each annulus: t_dyn = r / Vvir
    // We'll use mid-radius of each bin
    double total_sfr = 0.0;
    double total_h2 = 0.0;
    
    for(int i = 0; i < N_BINS; i++) {
        // Skip empty bins
        if(galaxies[p].DiscGas[i] <= 0.0) {
            continue;
        }
        
        // Annulus geometry
        double r_in = galaxies[p].DiscRadii[i] * 1000.0 / h;    // physical kpc
        double r_out = galaxies[p].DiscRadii[i+1] * 1000.0 / h; // physical kpc
        double r_mid = 0.5 * (r_in + r_out);                // kpc
        double area_kpc2 = M_PI * (r_out * r_out - r_in * r_in);
        
        if(area_kpc2 <= 0.0 || r_mid <= 0.0) {
            continue;
        }
        
        // Convert to pc^2 for surface density calculations
        double area_pc2 = area_kpc2 * 1.0e6;
        
        // Local surface densities [Msun/pc^2]
        double Sigma_gas = (galaxies[p].DiscGas[i] * 1.0e10 / h) / area_pc2;
        double Sigma_stars = (galaxies[p].DiscStars[i] * 1.0e10 / h) / area_pc2;
        
        // Local metallicity
        double Z_local = (galaxies[p].DiscGas[i] > 0.0) ?
            galaxies[p].DiscGasMetals[i] / galaxies[p].DiscGas[i] : 0.0;
        
        // Local dynamical time [code units]
        double tdyn_local = r_mid / galaxies[p].Vvir;
        
        if(tdyn_local <= 0.0) {
            continue;
        }
        
        // --- Compute H2 fraction based on SFprescription ---
        double f_H2 = 0.0;
        
        if(run_params->SFprescription == 0) {
            // Simple gas-based SF (no H2 tracking)
            f_H2 = 1.0;
            
        } else if(run_params->SFprescription == 1) {
            // Blitz & Rosolowsky 2006
            f_H2 = calculate_molecular_fraction_BR06(Sigma_gas, Sigma_stars, rs_pc);
            
        } else if(run_params->SFprescription == 2) {
            // Somerville 2025 (no H2, density-modulated efficiency)
            f_H2 = 1.0;  // Use all cold gas
            
        } else if(run_params->SFprescription == 3) {
            // Somerville 2025 + BR06 H2
            f_H2 = calculate_molecular_fraction_BR06(Sigma_gas, Sigma_stars, rs_pc);
            
        } else if(run_params->SFprescription == 4) {
            // Krumholz & Dekel 2012
            float clumping = 5.0;
            f_H2 = calculate_H2_fraction_KD12(Sigma_gas, Z_local, clumping);
            
        } else if(run_params->SFprescription == 5) {
            // Krumholz, McKee, & Tumlinson 2009
            double Z_prime = Z_local / 0.02;
            if(Z_prime < 0.05) Z_prime = 0.05;
            
            const double clumping = 3.0;
            double tau_c = 0.066 * clumping * Z_prime * Sigma_gas;
            double chi = 0.77 * (1.0 + 3.1 * pow(Z_prime, 0.365));
            
            double s = (tau_c > 0.0) ? 
                log(1.0 + 0.6 * chi + 0.01 * chi * chi) / (0.6 * tau_c) : 100.0;
            
            if(s < 2.0) {
                f_H2 = 1.0 - (3.0 * s) / (4.0 + s);
            } else {
                f_H2 = 0.0;
            }
            
        } else if(run_params->SFprescription == 6) {
            // Krumholz 2013
            double Z_prime = Z_local / 0.014;
            if(Z_prime < 0.01) Z_prime = 0.01;
            
            double fc = 5.0;
            double chi_2p = 3.1 * (1.0 + 3.1 * pow(Z_prime, 0.365)) / 4.1;
            double tau_c = 0.066 * fc * Z_prime * Sigma_gas;
            double s = (tau_c > 0.0) ? 
                log(1.0 + 0.6 * chi_2p + 0.01 * chi_2p * chi_2p) / (0.6 * tau_c) : 100.0;
            
            if(s < 2.0) {
                f_H2 = 1.0 - (0.75 * s) / (1.0 + 0.25 * s);
            } else {
                f_H2 = 0.0;
            }
            
        } else if(run_params->SFprescription == 7) {
            // Gnedin & Draine 2014
            double Z_prime = Z_local / 0.02;
            if(Z_prime < 0.01) Z_prime = 0.01;
            
            double D_MW = 1.5 * Z_prime;
            double S = log(1.0 + 0.6 * Z_prime) / log(1.0 + 0.4);
            double D_star = 0.17 * (2.0 + pow(S, 5.0)) / (1.0 + pow(S, 5.0));
            double g = sqrt(D_MW * D_MW + D_star * D_star);
            
            double s_param = log(1.0 + 0.6 * Z_prime + 0.01 * Z_prime * Z_prime) / 
                           (0.04 * Sigma_gas * Z_prime);
            double Sigma_R1 = (40.0 / g) * (s_param / (1.0 + s_param));
            double alpha = 1.0 + 0.7 * sqrt(s_param) / (1.0 + s_param);
            
            double q = (Sigma_R1 > 0.0 && Sigma_gas > 0.0) ? 
                pow(Sigma_gas / Sigma_R1, alpha) : 0.0;
            double eta = 0.0;
            double R = q * (1.0 + eta * q) / (1.0 + eta);
            f_H2 = R / (1.0 + R);
        }
        
        // Clamp H2 fraction
        if(f_H2 < 0.0) f_H2 = 0.0;
        if(f_H2 > 1.0) f_H2 = 1.0;
        
        // H2 mass in this annulus [10^10 Msun/h]
        h2_local[i] = f_H2 * galaxies[p].DiscGas[i] * HYDROGEN_MASS_FRAC;
        total_h2 += h2_local[i];
        
        // --- Compute SFR based on prescription ---
        double sfr_bin = 0.0;
        
        if(run_params->SFprescription == 0) {
            // Kauffmann 1996: simple threshold + efficiency
            const double reff_local = r_mid / 1000.0;  // Mpc/h
            const double cold_crit = 0.19 * galaxies[p].Vvir * reff_local;
            
            if(galaxies[p].DiscGas[i] > cold_crit && tdyn_local > 0.0) {
                sfr_bin = run_params->SfrEfficiency * (galaxies[p].DiscGas[i] - cold_crit) / tdyn_local;
            }
            
        } else if(run_params->SFprescription == 2) {
            // Somerville 2025: density-modulated efficiency
            const double Sigma_crit = 30.0 / (M_PI * 4.302e-3);
            double epsilon_cl = (Sigma_gas / Sigma_crit) / (1.0 + Sigma_gas / Sigma_crit);
            const double f_dense = 0.5;
            
            if(tdyn_local > 0.0 && Sigma_gas > 0.0) {
                sfr_bin = epsilon_cl * f_dense * galaxies[p].DiscGas[i] / tdyn_local;
            }
            
        } else if(run_params->SFprescription == 3) {
            // Somerville 2025 + BR06: density-modulated + H2
            const double Sigma_crit = 30.0 / (M_PI * 4.302e-3);
            double epsilon_cl = (Sigma_gas / Sigma_crit) / (1.0 + Sigma_gas / Sigma_crit);
            const double f_dense = 0.5;
            
            if(tdyn_local > 0.0 && Sigma_gas > 0.0 && h2_local[i] > 0.0) {
                sfr_bin = epsilon_cl * f_dense * h2_local[i] / tdyn_local;
            }
            
        } else {
            // All other prescriptions: SFR = efficiency × H2 / t_dyn
            if(h2_local[i] > 0.0 && tdyn_local > 0.0) {
                sfr_bin = run_params->SfrEfficiency * h2_local[i] / tdyn_local;
            }
        }
        
        sfr_local[i] = sfr_bin;
        total_sfr += sfr_bin;
    }
    
    // Store total H2 in bulk galaxy property
    galaxies[p].H2gas = total_h2;
    if(galaxies[p].H2gas > galaxies[p].ColdGas) {
        galaxies[p].H2gas = galaxies[p].ColdGas;
    }
    
    return total_sfr;
}

double apply_local_star_formation(const int p, const double dt, const int step,
                                 const double sfr_local[N_BINS],
                                 struct GALAXY *galaxies, const struct params *run_params)
{
    (void)step;  // Reserved for future use
    const double RecycleFraction = run_params->RecycleFraction;
    double total_stars_formed = 0.0;
    
    for(int i = 0; i < N_BINS; i++) {
        if(sfr_local[i] <= 0.0) {
            // Store SFR for diagnostics
            galaxies[p].DiscSFR[i] = 0.0;
            continue;
        }
        
        // Stars formed this step [10^10 Msun/h]
        double stars_bin = sfr_local[i] * dt;
        
        // Enforce local gas availability
        if(stars_bin > galaxies[p].DiscGas[i]) {
            stars_bin = galaxies[p].DiscGas[i];
        }
        
        // Calculate local metallicity and DTG BEFORE modifying gas
        double Z_local = (galaxies[p].DiscGas[i] > 0.0) ?
            galaxies[p].DiscGasMetals[i] / galaxies[p].DiscGas[i] : 0.0;
        double DTG_local = 0.0;
        if(run_params->DustOn == 1 && galaxies[p].DiscGas[i] > 0.0) {
            DTG_local = galaxies[p].DiscDust[i] / galaxies[p].DiscGas[i];
            if(DTG_local > 1.0) DTG_local = 1.0;  // Cap at unity
        }

        // Update gas
        galaxies[p].DiscGas[i] -= (1.0 - RecycleFraction) * stars_bin;
        galaxies[p].DiscGasMetals[i] -= Z_local * (1.0 - RecycleFraction) * stars_bin;

        // Update stars
        galaxies[p].DiscStars[i] += (1.0 - RecycleFraction) * stars_bin;
        galaxies[p].DiscStarsMetals[i] += Z_local * (1.0 - RecycleFraction) * stars_bin;

        // Dust (using pre-computed DTG)
        if(run_params->DustOn == 1) {
            galaxies[p].DiscDust[i] -= DTG_local * (1.0 - RecycleFraction) * stars_bin;
            if(galaxies[p].DiscDust[i] < 0.0) galaxies[p].DiscDust[i] = 0.0;
        }
        
        // Store SFR for this bin
        galaxies[p].DiscSFR[i] = sfr_local[i];
        
        // Safety clamps
        if(galaxies[p].DiscGas[i] < 0.0) galaxies[p].DiscGas[i] = 0.0;
        if(galaxies[p].DiscGasMetals[i] < 0.0) galaxies[p].DiscGasMetals[i] = 0.0;
        
        total_stars_formed += stars_bin;
    }
    
    return total_stars_formed;
}

double compute_toomre_Q(double Sigma_gas, double Sigma_stars, double r_mid, double Vvir,
                       const struct params *run_params)
{
    (void)run_params;  // Reserved for future use
    if(Sigma_gas <= 0.0 || r_mid <= 0.0 || Vvir <= 0.0) {
        return 1000.0;  // Large Q = stable
    }
    
    // Velocity dispersion [km/s]
    // For gas: σ ~ 10-20 km/s (ISM turbulence + thermal)
    // For stars: σ ~ Vvir / 10 (roughly)
    double sigma_gas = 10.0;  // km/s, typical for cold ISM
    double sigma_stars = Vvir / 10.0;
    
    // Combined effective velocity dispersion (mass-weighted)
    double Sigma_total = Sigma_gas + Sigma_stars;
    double sigma_eff = 0.0;
    if(Sigma_total > 0.0) {
        sigma_eff = (Sigma_gas * sigma_gas + Sigma_stars * sigma_stars) / Sigma_total;
    } else {
        return 1000.0;
    }
    
    // Epicyclic frequency κ = √2 * Vcirc / r for flat rotation curve
    // Vcirc ≈ Vvir
    double kappa = sqrt(2.0) * Vvir / r_mid;  // (km/s) / kpc
    
    // Convert Σ from Msun/pc^2 to Msun/kpc^2 for Q calculation
    double Sigma_total_kpc2 = Sigma_total * 1.0e6;
    
    // Toomre Q = (σ κ) / (π G Σ)
    // G = 4.302e-3 (km/s)^2 pc / Msun
    // G in kpc units: 4.302e-3 / 1e3 = 4.302e-6 (km/s)^2 kpc / Msun
    const double G_kpc = 4.302e-6;  // (km/s)^2 kpc / Msun
    
    double Q = (sigma_eff * kappa) / (M_PI * G_kpc * Sigma_total_kpc2);
    
    return Q;
}

void check_local_disk_instability(const int p, const int centralgal, const double dt, const int step,
                                 struct GALAXY *galaxies, const struct params *run_params)
{
    (void)centralgal;  // Reserved for future use
    (void)dt;  // Reserved for future use
    (void)step;  // Reserved for future use
    
    if(galaxies[p].Vvir <= 0.0 || galaxies[p].DiskScaleRadius <= 0.0) {
        return;
    }
    
    const float h = run_params->Hubble_h;
    const double Q_crit = 1.0;  // Critical Q for marginal stability
    
    double total_unstable_stars = 0.0;
    
    for(int i = 0; i < N_BINS; i++) {
        double r_in = galaxies[p].DiscRadii[i] * 1000.0 / h;    // physical kpc
        double r_out = galaxies[p].DiscRadii[i+1] * 1000.0 / h;
        double r_mid = 0.5 * (r_in + r_out);
        double area_pc2 = M_PI * (r_out * r_out - r_in * r_in) * 1.0e6;

        if(area_pc2 <= 0.0) {
            continue;
        }

        double Sigma_gas = (galaxies[p].DiscGas[i] * 1.0e10 / h) / area_pc2;
        double Sigma_stars = (galaxies[p].DiscStars[i] * 1.0e10 / h) / area_pc2;

        // Compute local Q
        double Q = compute_toomre_Q(Sigma_gas, Sigma_stars, r_mid, galaxies[p].Vvir, run_params);
        
        if(Q < Q_crit) {
            // Unstable! Transfer (1 - Q/Q_crit) fraction to bulge
            double unstable_frac = 1.0 - Q / Q_crit;
            
            double unstable_stars_bin = unstable_frac * galaxies[p].DiscStars[i];
            double unstable_gas_bin = unstable_frac * galaxies[p].DiscGas[i];
            
            double Z_stars = (galaxies[p].DiscStars[i] > 0.0) ?
                galaxies[p].DiscStarsMetals[i] / galaxies[p].DiscStars[i] : 0.0;
            double Z_gas = (galaxies[p].DiscGas[i] > 0.0) ?
                galaxies[p].DiscGasMetals[i] / galaxies[p].DiscGas[i] : 0.0;
            
            // Remove from disk
            galaxies[p].DiscStars[i] -= unstable_stars_bin;
            galaxies[p].DiscGas[i] -= unstable_gas_bin;
            galaxies[p].DiscStarsMetals[i] -= Z_stars * unstable_stars_bin;
            galaxies[p].DiscGasMetals[i] -= Z_gas * unstable_gas_bin;
            
            total_unstable_stars += unstable_stars_bin;
            
            // Safety
            if(galaxies[p].DiscStars[i] < 0.0) galaxies[p].DiscStars[i] = 0.0;
            if(galaxies[p].DiscGas[i] < 0.0) galaxies[p].DiscGas[i] = 0.0;
            if(galaxies[p].DiscStarsMetals[i] < 0.0) galaxies[p].DiscStarsMetals[i] = 0.0;
            if(galaxies[p].DiscGasMetals[i] < 0.0) galaxies[p].DiscGasMetals[i] = 0.0;
        }
    }
    
    // Add unstable mass to bulge
    if(total_unstable_stars > 0.0) {
        double Z_disk = get_metallicity(galaxies[p].StellarMass - galaxies[p].BulgeMass,
                                       galaxies[p].MetalsStellarMass - galaxies[p].MetalsBulgeMass);
        galaxies[p].BulgeMass += total_unstable_stars;
        galaxies[p].InstabilityBulgeMass += total_unstable_stars;
        galaxies[p].MetalsBulgeMass += Z_disk * total_unstable_stars;
    }
    
    // TODO: Handle unstable gas - for now it stays in disk (could trigger burst)
    // In DarkSage this might trigger BH growth or starburst
}

void apply_radial_gas_flows(const int p, const double dt, struct GALAXY *galaxies,
                           const struct params *run_params)
{
    const double Vvir = galaxies[p].Vvir;
    if(Vvir <= 0.0 || dt <= 0.0) {
        return;
    }

    /* OPTIMIZATION: Early exit if no gas in disk */
    if(galaxies[p].ColdGas <= 0.0) {
        return;
    }

    /* Cache pointers with restrict for faster access */
    const float * restrict DiscRadii = galaxies[p].DiscRadii;
    const float * restrict DiscGas_src = galaxies[p].DiscGas;
    const float * restrict DiscGasMetals_src = galaxies[p].DiscGasMetals;
    const int DustOn = run_params->DustOn;

    /* Pre-compute constants outside loop */
    const double h = run_params->Hubble_h;
    const double kpc_factor = 1000.0 / h;

    /* Viscosity parameter (α-disk prescription) and sound speed */
    const double alpha = 0.01;
    const double cs = 10.0;  /* km/s, typical for cold ISM */
    const double alpha_cs_sq = alpha * cs * cs;  /* Pre-compute for inner loop */
    const double inv_Vvir_sq = 1.0 / (Vvir * Vvir);  /* Pre-compute reciprocal */

    /* Arrays to hold updated gas distribution */
    double DiscGas_new[N_BINS];
    double DiscGasMetals_new[N_BINS];
    double DiscDust_new[N_BINS];

    /* Initialize new arrays - single loop */
    if(DustOn == 1) {
        const float * restrict DiscDust_src = galaxies[p].DiscDust;
        for(int i = 0; i < N_BINS; i++) {
            DiscGas_new[i] = DiscGas_src[i];
            DiscGasMetals_new[i] = DiscGasMetals_src[i];
            DiscDust_new[i] = DiscDust_src[i];
        }
    } else {
        for(int i = 0; i < N_BINS; i++) {
            DiscGas_new[i] = DiscGas_src[i];
            DiscGasMetals_new[i] = DiscGasMetals_src[i];
        }
    }

    /* Diffusive flow from each bin to inner bin */
    for(int i = N_BINS - 1; i > 0; i--) {
        const double gas_i = DiscGas_src[i];
        if(gas_i <= 0.0) {
            continue;
        }

        const double r_mid = 0.5 * (DiscRadii[i] + DiscRadii[i+1]) * kpc_factor;  /* physical kpc */

        /* Optimized viscous timescale calculation:
         * h_scale = cs * r_mid / Vvir
         * nu = alpha * cs * h_scale = alpha * cs * cs * r_mid / Vvir
         * t_visc_code = r_mid^2 / nu / Vvir = r_mid * Vvir / (alpha * cs^2)
         * flow_frac = dt / t_visc_code = dt * alpha * cs^2 / (r_mid * Vvir)
         * Simplify: flow_frac = dt * alpha_cs_sq / (r_mid * Vvir)
         */
        const double t_visc_code = r_mid / (alpha_cs_sq * inv_Vvir_sq * Vvir);  /* = r_mid * Vvir / alpha_cs_sq */
        if(t_visc_code <= 0.0) {
            continue;
        }

        double flow_frac = dt / t_visc_code;
        if(flow_frac > 0.5) flow_frac = 0.5;  /* Limit for stability */

        const double flow_mass = flow_frac * gas_i;
        const double flow_metals = flow_frac * DiscGasMetals_src[i];

        /* Remove from current bin, add to inner bin */
        DiscGas_new[i] -= flow_mass;
        DiscGasMetals_new[i] -= flow_metals;
        DiscGas_new[i-1] += flow_mass;
        DiscGasMetals_new[i-1] += flow_metals;

        if(DustOn == 1) {
            const double flow_dust = flow_frac * galaxies[p].DiscDust[i];
            DiscDust_new[i] -= flow_dust;
            DiscDust_new[i-1] += flow_dust;
        }
    }

    /* Update galaxy arrays with safety clamps - single pass */
    float * restrict DiscGas_dst = galaxies[p].DiscGas;
    float * restrict DiscGasMetals_dst = galaxies[p].DiscGasMetals;

    if(DustOn == 1) {
        float * restrict DiscDust_dst = galaxies[p].DiscDust;
        for(int i = 0; i < N_BINS; i++) {
            DiscGas_dst[i] = (DiscGas_new[i] > 0.0) ? (float)DiscGas_new[i] : 0.0f;
            DiscGasMetals_dst[i] = (DiscGasMetals_new[i] > 0.0) ? (float)DiscGasMetals_new[i] : 0.0f;
            DiscDust_dst[i] = (DiscDust_new[i] > 0.0) ? (float)DiscDust_new[i] : 0.0f;
        }
    } else {
        for(int i = 0; i < N_BINS; i++) {
            DiscGas_dst[i] = (DiscGas_new[i] > 0.0) ? (float)DiscGas_new[i] : 0.0f;
            DiscGasMetals_dst[i] = (DiscGasMetals_new[i] > 0.0) ? (float)DiscGasMetals_new[i] : 0.0f;
        }
    }
}


/* ========================================================================== */
/* FULL DARKMODE PHYSICS FUNCTIONS                                            */
/* Enhanced disk physics: combined Q, precession, j-conservation              */
/* ========================================================================== */

void rotate_vector(float v[3], const float axis[3], const double angle_deg)
{
    if(angle_deg == 0.0) return;

    const double angle_rad = angle_deg * M_PI / 180.0;
    const double cos_a = cos(angle_rad);
    const double sin_a = sin(angle_rad);

    /* Cross product: k × v */
    double cross[3];
    cross[0] = axis[1] * v[2] - axis[2] * v[1];
    cross[1] = axis[2] * v[0] - axis[0] * v[2];
    cross[2] = axis[0] * v[1] - axis[1] * v[0];

    /* Dot product: k · v */
    double dot = axis[0] * v[0] + axis[1] * v[1] + axis[2] * v[2];

    /* Rodrigues formula */
    for(int j = 0; j < 3; j++) {
        v[j] = (float)(v[j] * cos_a + cross[j] * sin_a + axis[j] * dot * (1.0 - cos_a));
    }
}

double get_disc_ang_mom(const int p, const int component, double J[3],
                        struct GALAXY *galaxies, const struct params *run_params)
{
    double total_J = 0.0;
    J[0] = J[1] = J[2] = 0.0;

    /* Get spin direction */
    const float *spin = (component == 0) ? galaxies[p].SpinGas : galaxies[p].SpinStars;

    for(int i = 0; i < N_BINS; i++) {
        /* j-bin midpoint specific angular momentum */
        double j_mid = 0.5 * (run_params->DiscBinEdge[i] + run_params->DiscBinEdge[i+1]);

        /* Mass in this annulus */
        double mass = (component == 0) ? galaxies[p].DiscGas[i] : galaxies[p].DiscStars[i];

        if(mass > 0.0 && j_mid > 0.0) {
            double J_bin = mass * j_mid;
            total_J += J_bin;

            for(int k = 0; k < 3; k++) {
                J[k] += J_bin * spin[k];
            }
        }
    }

    return total_J;
}

void project_disc(const float DiscMass[N_BINS], const double cos_angle,
                  double NewDisc[N_BINS], const struct params *run_params,
                  const double Rvir, const double Vvir)
{
    double cos_ang = fabs(cos_angle);  /* Handle retrograde by taking absolute value */

    /* If nearly aligned, just copy - avoid floating point issues */
    if(cos_ang > 0.99) {
        for(int i = 0; i < N_BINS; i++) {
            NewDisc[i] = (double)DiscMass[i];
        }
        return;
    }

    /* Make a working copy (convert float to double) */
    double WorkMass[N_BINS];
    for(int i = 0; i < N_BINS; i++) {
        WorkMass[i] = (double)DiscMass[i];
    }

    int j_old = 0;

    for(int i = 0; i < N_BINS; i++) {
        /* Upper j-boundary of new bin i, projected */
        double high_bound = run_params->DiscBinEdge[i+1] / cos_ang;

        /* Find which old bins contribute to new bin i */
        int j = j_old;
        while(j < N_BINS && run_params->DiscBinEdge[j] < high_bound) {
            j++;
        }
        j--;
        if(j < 0) j = 0;

        /* Sum mass from old bins that fully fit into new bin i */
        NewDisc[i] = 0.0;
        for(int l = j_old; l < j && l < N_BINS; l++) {
            NewDisc[i] += WorkMass[l];
            WorkMass[l] = 0.0;
        }

        /* Handle partial contribution from the boundary bin */
        if(i < N_BINS - 1 && j < N_BINS) {
            double ratio_last_bin;
            if(j < N_BINS - 1) {
                double dj = run_params->DiscBinEdge[j+1] - run_params->DiscBinEdge[j];
                if(dj > 0.0) {
                    ratio_last_bin = (high_bound - run_params->DiscBinEdge[j]) / dj;
                    ratio_last_bin = ratio_last_bin * ratio_last_bin;  /* Area scaling */
                    if(ratio_last_bin > 1.0) ratio_last_bin = 1.0;
                } else {
                    ratio_last_bin = 1.0;
                }
            } else {
                /* Outermost bin: use virial radius as upper limit */
                double j_vir = Rvir * Vvir;  /* Approximate max j */
                if(j_vir > run_params->DiscBinEdge[j]) {
                    ratio_last_bin = (high_bound - run_params->DiscBinEdge[j]) /
                                     (j_vir - run_params->DiscBinEdge[j]);
                    ratio_last_bin = ratio_last_bin * ratio_last_bin;
                    if(ratio_last_bin > 1.0) ratio_last_bin = 1.0;
                } else {
                    ratio_last_bin = 1.0;
                }
            }
            NewDisc[i] += ratio_last_bin * WorkMass[j];
            WorkMass[j] -= ratio_last_bin * WorkMass[j];
        } else if(i == N_BINS - 1) {
            /* Last bin gets all remaining mass */
            for(int l = j_old; l < N_BINS; l++) {
                NewDisc[i] += WorkMass[l];
            }
        }

        /* Safety check */
        if(NewDisc[i] < 0.0) NewDisc[i] = 0.0;

        j_old = j;
    }
}

void precess_gas(const int p, const double dt, struct GALAXY *galaxies,
                 const struct params *run_params)
{
    if(run_params->DarkSAGEOn != 1) return;

    /* OPTIMIZATION: Cache pointers with restrict for faster access */
    const float * restrict DiscGas = galaxies[p].DiscGas;
    const float * restrict DiscRadii = galaxies[p].DiscRadii;
    const double * restrict DiscBinEdge = run_params->DiscBinEdge;

    /* Get total disc gas - single pass */
    double DiscGasSum = 0.0;
    for(int i = 0; i < N_BINS; i++) {
        DiscGasSum += DiscGas[i];
    }
    if(DiscGasSum <= 0.0) return;

    /* Calculate angle between gas and stellar spins */
    const float * restrict SpinGas = galaxies[p].SpinGas;
    const float * restrict SpinStars = galaxies[p].SpinStars;
    double cos_theta = SpinGas[0] * SpinStars[0] +
                       SpinGas[1] * SpinStars[1] +
                       SpinGas[2] * SpinStars[2];

    /* Clamp to [-1, 1] for numerical safety */
    if(cos_theta > 1.0) cos_theta = 1.0;
    if(cos_theta < -1.0) cos_theta = -1.0;

    /* If already coplanar (within threshold), no precession needed */
    const double cos_thresh = cos(run_params->ThetaThresh * M_PI / 180.0);
    if(fabs(cos_theta) > cos_thresh) {
        return;
    }

    /* Calculate mass-weighted precession angle like DarkSage */
    /* Each annulus precesses at its own rate based on local t_dyn */
    const double DegPerTdyn = run_params->DegPerTdyn;
    const double inv_DiscGasSum = 1.0 / DiscGasSum;  /* Avoid division in loop */
    double deg = 0.0;

    for(int i = N_BINS - 1; i >= 0; i--) {
        const double gas_i = DiscGas[i];
        if(gas_i <= 0.0) continue;

        const double j_edge = DiscBinEdge[i+1];
        if(j_edge <= 0.0) continue;

        /* t_dyn = r²/j (DarkSage formula) */
        const double r_outer = DiscRadii[i+1];
        const double tdyn = (r_outer * r_outer) / j_edge;
        if(tdyn <= 0.0) continue;

        deg += (DegPerTdyn * dt / tdyn) * gas_i * inv_DiscGasSum;
    }

    if(deg <= 0.0) return;

    /* Calculate target angle after precession */
    double cos_angle_precess = cos(deg * M_PI / 180.0);

    /* Don't precess past alignment */
    if(cos_angle_precess < fabs(cos_theta)) {
        cos_angle_precess = fabs(cos_theta);
    }

    /* Project disc to new orientation (DarkSage-style) */
    double NewDiscGas[N_BINS];
    double NewDiscGasMetals[N_BINS];
    double NewDiscDust[N_BINS];

    project_disc(galaxies[p].DiscGas, cos_angle_precess, NewDiscGas,
                 run_params, galaxies[p].Rvir, galaxies[p].Vvir);
    project_disc(galaxies[p].DiscGasMetals, cos_angle_precess, NewDiscGasMetals,
                 run_params, galaxies[p].Rvir, galaxies[p].Vvir);

    /* Also project dust if DustOn */
    if(run_params->DustOn == 1) {
        project_disc(galaxies[p].DiscDust, cos_angle_precess, NewDiscDust,
                     run_params, galaxies[p].Rvir, galaxies[p].Vvir);
    }

    /* Update disc arrays */
    for(int i = 0; i < N_BINS; i++) {
        galaxies[p].DiscGas[i] = NewDiscGas[i];
        galaxies[p].DiscGasMetals[i] = NewDiscGasMetals[i];
        if(run_params->DustOn == 1) {
            galaxies[p].DiscDust[i] = NewDiscDust[i];
        }
    }

    /* Update spin vector */
    if(cos_angle_precess == fabs(cos_theta)) {
        /* Aligned or counter-aligned */
        if(cos_theta >= 0.0) {
            for(int j = 0; j < 3; j++) {
                galaxies[p].SpinGas[j] = galaxies[p].SpinStars[j];
            }
        } else {
            for(int j = 0; j < 3; j++) {
                galaxies[p].SpinGas[j] = -galaxies[p].SpinStars[j];
            }
        }
    } else {
        /* Rotate spin vector toward stellar spin */
        float axis[3];
        axis[0] = galaxies[p].SpinGas[1] * galaxies[p].SpinStars[2] -
                  galaxies[p].SpinGas[2] * galaxies[p].SpinStars[1];
        axis[1] = galaxies[p].SpinGas[2] * galaxies[p].SpinStars[0] -
                  galaxies[p].SpinGas[0] * galaxies[p].SpinStars[2];
        axis[2] = galaxies[p].SpinGas[0] * galaxies[p].SpinStars[1] -
                  galaxies[p].SpinGas[1] * galaxies[p].SpinStars[0];

        if(cos_theta < 0.0) {
            for(int j = 0; j < 3; j++) axis[j] *= -1.0f;
        }

        double axis_mag = sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]);
        if(axis_mag > 0.0) {
            for(int j = 0; j < 3; j++) {
                axis[j] /= (float)axis_mag;
            }
            /* Angle to rotate is acos(cos_angle_precess) */
            rotate_vector(galaxies[p].SpinGas, axis, acos(cos_angle_precess) * 180.0 / M_PI);

            /* Re-normalize */
            double spin_mag = sqrt(galaxies[p].SpinGas[0]*galaxies[p].SpinGas[0] +
                                   galaxies[p].SpinGas[1]*galaxies[p].SpinGas[1] +
                                   galaxies[p].SpinGas[2]*galaxies[p].SpinGas[2]);
            if(spin_mag > 0.0) {
                for(int j = 0; j < 3; j++) {
                    galaxies[p].SpinGas[j] /= (float)spin_mag;
                }
            }
        }
    }
}

double compute_epicyclic_frequency(int bin, double r_inner, double r_outer,
                                   const struct params *run_params)
{
    double dj = run_params->DiscBinEdge[bin+1] - run_params->DiscBinEdge[bin];
    double dr = r_outer - r_inner;

    if(dr <= 0.0 || dj <= 0.0) {
        return 0.0;
    }

    double Kappa;
    if(bin > 0) {
        double j_inner = run_params->DiscBinEdge[bin];
        double r_inner_cubed = r_inner * r_inner * r_inner;
        if(r_inner_cubed <= 0.0) return 0.0;
        Kappa = sqrt(2.0 * j_inner / r_inner_cubed * dj / dr);
    } else {
        /* Innermost bin: use outer edge */
        double j_outer = run_params->DiscBinEdge[bin+1];
        double r_outer_cubed = r_outer * r_outer * r_outer;
        if(r_outer_cubed <= 0.0) return 0.0;
        Kappa = sqrt(2.0 * j_outer / r_outer_cubed * dj / dr);
    }

    return Kappa;
}

double compute_combined_toomre_Q_darkmode(int bin, double DiscGas, double DiscStars,
                                          double sigma_gas, double sigma_stars,
                                          double r_inner, double r_outer,
                                          const struct params *run_params)
{
    if(r_outer <= 0.0) {
        return 1000.0;  /* Stable */
    }

    /* Epicyclic frequency using DarkSage j-bin formula */
    double Kappa = compute_epicyclic_frequency(bin, r_inner, r_outer, run_params);
    if(Kappa <= 0.0) {
        return 1000.0;
    }

    /* Annulus area factor: (r_outer² - r_inner²) */
    double r_out_sq = r_outer * r_outer;
    double r_in_sq = r_inner * r_inner;
    double area_factor = r_out_sq - r_in_sq;
    if(area_factor <= 0.0) {
        return 1000.0;
    }

    /* G in code units: (km/s)² * (Mpc/h) / (10^10 Msun/h) */
    double G = run_params->G;

    /* Individual Q values using DarkSage formula:
     * Q_gas = c_s * Kappa * (r²_out - r²_in) / (G * M_gas)
     * Q_star = Kappa * sigma_R * 0.935 * (r²_out - r²_in) / (G * M_stars)
     * Note: 0.935 ≈ 1/(3.36/π) accounts for stellar disk geometry
     */
    double Q_gas = 1000.0;
    double Q_stars = 1000.0;

    if(DiscGas > 0.0 && sigma_gas > 0.0) {
        Q_gas = sigma_gas * Kappa * area_factor / (G * DiscGas);
    }

    if(DiscStars > 0.0 && sigma_stars > 0.0) {
        Q_stars = Kappa * sigma_stars * 0.935 * area_factor / (G * DiscStars);
    }

    /* If either component is absent, return single-component Q */
    if(DiscGas <= 0.0) return Q_stars;
    if(DiscStars <= 0.0) return Q_gas;

    /* Romeo & Wiegert two-fluid weighting factor */
    double sigma_R_sq = sigma_stars * sigma_stars;
    double sigma_g_sq = sigma_gas * sigma_gas;
    double W = 2.0 * sigma_stars * sigma_gas / (sigma_R_sq + sigma_g_sq);

    /* Combined Q depends on which dispersion is larger */
    double Q_tot;
    if(sigma_stars >= sigma_gas) {
        /* Stars are dynamically hotter */
        double denom = W / Q_gas + 1.0 / Q_stars;
        Q_tot = (denom > 0.0) ? 1.0 / denom : 1000.0;
    } else {
        /* Gas is dynamically hotter */
        double denom = 1.0 / Q_gas + W / Q_stars;
        Q_tot = (denom > 0.0) ? 1.0 / denom : 1000.0;
    }

    return Q_tot;
}

/* Legacy Q function for non-DarkMode use (kept for compatibility) */
double compute_combined_toomre_Q(double Sigma_gas, double Sigma_stars,
                                 double sigma_gas, double sigma_stars,
                                 double r_mid, double Vvir)
{
    if(r_mid <= 0.0 || Vvir <= 0.0) {
        return 1000.0;  /* Stable */
    }

    /* Epicyclic frequency κ = √2 * Vcirc / r for flat rotation curve */
    double kappa = sqrt(2.0) * Vvir / r_mid;  /* km/s/kpc */

    /* G in units suitable for (km/s)^2 kpc / Msun */
    const double G_kpc = 4.302e-6;

    /* Individual Q values */
    double Q_gas = 1000.0;
    double Q_stars = 1000.0;

    if(Sigma_gas > 0.0 && sigma_gas > 0.0) {
        double Sigma_gas_kpc2 = Sigma_gas * 1.0e6;  /* Msun/pc^2 -> Msun/kpc^2 */
        Q_gas = (sigma_gas * kappa) / (M_PI * G_kpc * Sigma_gas_kpc2);
    }

    if(Sigma_stars > 0.0 && sigma_stars > 0.0) {
        double Sigma_stars_kpc2 = Sigma_stars * 1.0e6;
        /* For stars, use radial velocity dispersion with factor 3.36 (not π) */
        Q_stars = (sigma_stars * kappa) / (3.36 * G_kpc * Sigma_stars_kpc2);
    }

    /* If either component is absent, return single-component Q */
    if(Sigma_gas <= 0.0) return Q_stars;
    if(Sigma_stars <= 0.0) return Q_gas;

    /* Romeo & Wiegert weighting factor */
    double sigma_R_sq = sigma_stars * sigma_stars;
    double sigma_g_sq = sigma_gas * sigma_gas;
    double W = 2.0 * sigma_stars * sigma_gas / (sigma_R_sq + sigma_g_sq);

    /* Combined Q depends on which dispersion is larger */
    double Q_tot;
    if(sigma_stars >= sigma_gas) {
        /* Stars are dynamically hotter */
        double denom = W / Q_gas + 1.0 / Q_stars;
        Q_tot = (denom > 0.0) ? 1.0 / denom : 1000.0;
    } else {
        /* Gas is dynamically hotter */
        double denom = 1.0 / Q_gas + W / Q_stars;
        Q_tot = (denom > 0.0) ? 1.0 / denom : 1000.0;
    }

    return Q_tot;
}

void deal_with_unstable_gas(const int p, const int bin, const double unstable_gas,
                            struct GALAXY *galaxies, const struct params *run_params)
{
    if(unstable_gas <= 0.0 || bin <= 0) return;

    /* Get j-bin edges */
    double j_this = 0.5 * (run_params->DiscBinEdge[bin] + run_params->DiscBinEdge[bin+1]);
    double j_inner = 0.5 * (run_params->DiscBinEdge[bin-1] + run_params->DiscBinEdge[bin]);

    /* Angular momentum lost by sinking gas */
    double j_lose = j_this - j_inner;

    /* Angular momentum gained by gas moving outward (to conserve total j) */
    double j_gain = 0.0;
    if(bin < N_BINS - 1) {
        double j_outer = 0.5 * (run_params->DiscBinEdge[bin+1] + run_params->DiscBinEdge[bin+2]);
        j_gain = j_outer - j_this;
    } else {
        /* Outermost bin: j goes to hot gas / CGM */
        j_gain = j_lose;  /* Approximate */
    }

    if(j_gain <= 0.0 || j_lose <= 0.0) return;

    /* Mass fractions for j-conservation: m_up/m_down = j_lose/j_gain */
    double total_ratio = j_lose + j_gain;
    double m_down = unstable_gas * j_gain / total_ratio;  /* Sinks inward */
    double m_up = unstable_gas * j_lose / total_ratio;    /* Moves outward */

    /* Get metallicity and dust ratios */
    double Z_gas = (galaxies[p].DiscGas[bin] > 0.0) ?
        galaxies[p].DiscGasMetals[bin] / galaxies[p].DiscGas[bin] : 0.0;
    double DTG = 0.0;
    if(run_params->DustOn == 1 && galaxies[p].DiscGas[bin] > 0.0) {
        DTG = galaxies[p].DiscDust[bin] / galaxies[p].DiscGas[bin];
    }

    /* Remove from current bin */
    galaxies[p].DiscGas[bin] -= unstable_gas;
    galaxies[p].DiscGasMetals[bin] -= Z_gas * unstable_gas;
    if(run_params->DustOn == 1) {
        galaxies[p].DiscDust[bin] -= DTG * unstable_gas;
    }

    /* Add to inner bin */
    galaxies[p].DiscGas[bin-1] += m_down;
    galaxies[p].DiscGasMetals[bin-1] += Z_gas * m_down;
    if(run_params->DustOn == 1) {
        galaxies[p].DiscDust[bin-1] += DTG * m_down;
    }

    /* Add to outer bin (or outermost) */
    if(bin < N_BINS - 1) {
        galaxies[p].DiscGas[bin+1] += m_up;
        galaxies[p].DiscGasMetals[bin+1] += Z_gas * m_up;
        if(run_params->DustOn == 1) {
            galaxies[p].DiscDust[bin+1] += DTG * m_up;
        }
    } else {
        /* Outermost bin: mass goes to last bin */
        galaxies[p].DiscGas[N_BINS-1] += m_up;
        galaxies[p].DiscGasMetals[N_BINS-1] += Z_gas * m_up;
        if(run_params->DustOn == 1) {
            galaxies[p].DiscDust[N_BINS-1] += DTG * m_up;
        }
    }

    /* Safety clamps */
    if(galaxies[p].DiscGas[bin] < 0.0) galaxies[p].DiscGas[bin] = 0.0;
    if(galaxies[p].DiscGasMetals[bin] < 0.0) galaxies[p].DiscGasMetals[bin] = 0.0;
    if(run_params->DustOn == 1 && galaxies[p].DiscDust[bin] < 0.0) {
        galaxies[p].DiscDust[bin] = 0.0;
    }
}

void deal_with_unstable_stars(const int p, const int bin, const double unstable_stars,
                              struct GALAXY *galaxies, const struct params *run_params)
{
    if(unstable_stars <= 0.0 || bin <= 0) return;

    /* j-bin midpoints for angular momentum conservation */
    double j_this = 0.5 * (run_params->DiscBinEdge[bin] + run_params->DiscBinEdge[bin+1]);
    double j_inner = 0.5 * (run_params->DiscBinEdge[bin-1] + run_params->DiscBinEdge[bin]);

    double j_lose = j_this - j_inner;

    double j_gain = 0.0;
    if(bin < N_BINS - 1) {
        double j_outer = 0.5 * (run_params->DiscBinEdge[bin+1] + run_params->DiscBinEdge[bin+2]);
        j_gain = j_outer - j_this;
    } else {
        j_gain = j_lose;
    }

    if(j_gain <= 0.0 || j_lose <= 0.0) return;

    /* j-conservation: m_down sinks inward, m_up compensates outward */
    double total_ratio = j_lose + j_gain;
    double m_down = unstable_stars * j_gain / total_ratio;
    double m_up = unstable_stars * j_lose / total_ratio;

    double Z_stars = (galaxies[p].DiscStars[bin] > 0.0) ?
        galaxies[p].DiscStarsMetals[bin] / galaxies[p].DiscStars[bin] : 0.0;

    /* Remove from current bin */
    galaxies[p].DiscStars[bin] -= unstable_stars;
    galaxies[p].DiscStarsMetals[bin] -= Z_stars * unstable_stars;

    /* Add to inner bin */
    galaxies[p].DiscStars[bin-1] += m_down;
    galaxies[p].DiscStarsMetals[bin-1] += Z_stars * m_down;

    /* Velocity dispersion mixing for inner bin */
    double sigma_src = (double)galaxies[p].VelDispStars[bin];
    double sigma_dst = (double)galaxies[p].VelDispStars[bin-1];
    double m_dst = galaxies[p].DiscStars[bin-1] - m_down;  /* mass before addition */
    if(m_dst < 0.0) m_dst = 0.0;
    double m_total_inner = m_dst + m_down;
    if(m_total_inner > 0.0) {
        double new_sq = (m_dst * sigma_dst * sigma_dst + m_down * sigma_src * sigma_src) / m_total_inner;
        galaxies[p].VelDispStars[bin-1] = (float)sqrt(new_sq);
    }

    /* Add compensating mass to outer bin */
    if(bin < N_BINS - 1) {
        galaxies[p].DiscStars[bin+1] += m_up;
        galaxies[p].DiscStarsMetals[bin+1] += Z_stars * m_up;
    } else {
        galaxies[p].DiscStars[N_BINS-1] += m_up;
        galaxies[p].DiscStarsMetals[N_BINS-1] += Z_stars * m_up;
    }

    /* Safety clamps */
    if(galaxies[p].DiscStars[bin] < 0.0) galaxies[p].DiscStars[bin] = 0.0;
    if(galaxies[p].DiscStarsMetals[bin] < 0.0) galaxies[p].DiscStarsMetals[bin] = 0.0;
}

void check_full_disk_instability(const int p, const int centralgal, const double dt, const int step,
                                 struct GALAXY *galaxies, const struct params *run_params)
{
    (void)centralgal;
    (void)step;
    (void)dt;

    if(run_params->DarkSAGEOn != 1) return;
    if(galaxies[p].Vvir <= 0.0 || galaxies[p].DiskScaleRadius <= 0.0) return;

    /* OPTIMIZATION: Quick check if disc has any mass at all */
    const double diskmass = galaxies[p].ColdGas +
                           (galaxies[p].StellarMass - galaxies[p].BulgeMass);
    if(diskmass <= 0.0) return;

    /* OPTIMIZATION: Skip if disk is globally stable (diskmass < Mcrit) */
    /* This avoids expensive per-bin calculations for stable disks */
    const double Mcrit = galaxies[p].Vmax * galaxies[p].Vmax *
                        (3.0 * galaxies[p].DiskScaleRadius) / run_params->G;

    /* Global stability factor with soft floor */
    const double f_floor = 0.15;
    double f_global = 1.0;
    if(Mcrit > 0.0 && diskmass < 2.0 * Mcrit) {
        double ratio = diskmass / (2.0 * Mcrit);
        f_global = f_floor + (1.0 - f_floor) * ratio;
        if(f_global > 1.0) f_global = 1.0;
    }

    const double Q_min = run_params->QTotMin;
    const double gas_sink_rate = run_params->GasSinkRate;
    const double c_s = 10.0;  /* km/s - sound speed for cold ISM */
    const double G = run_params->G;

    /* Cache DiscBinEdge pointer for faster access */
    const double * restrict DiscBinEdge = run_params->DiscBinEdge;

    /* Cache galaxy disc pointers */
    float * restrict DiscGas = galaxies[p].DiscGas;
    float * restrict DiscStars = galaxies[p].DiscStars;
    float * restrict DiscRadii = galaxies[p].DiscRadii;
    float * restrict VelDispStars = galaxies[p].VelDispStars;

    /* --- Pass 1: bins 1..N_BINS-1 migrate excess mass inward one bin --- */
    for(int i = N_BINS - 1; i >= 1; i--) {
        const double disc_gas_i = DiscGas[i];
        const double disc_stars_i = DiscStars[i];

        if(disc_gas_i <= 0.0 && disc_stars_i <= 0.0) continue;

        const double r_inner = DiscRadii[i];
        const double r_outer = DiscRadii[i+1];
        if(r_outer <= 0.0) continue;

        double sigma_stars = (double)VelDispStars[i];
        if(sigma_stars < 10.0) sigma_stars = 10.0;

        /* INLINED Q calculation for speed */
        double Q = 1000.0;
        {
            /* Compute Kappa inline */
            const double dj = DiscBinEdge[i+1] - DiscBinEdge[i];
            const double dr = r_outer - r_inner;

            if(dr > 0.0 && dj > 0.0) {
                const double j_inner = DiscBinEdge[i];
                const double r_inner_cubed = r_inner * r_inner * r_inner;
                const double Kappa = (r_inner_cubed > 0.0) ?
                    sqrt(2.0 * j_inner / r_inner_cubed * dj / dr) : 0.0;

                if(Kappa > 0.0) {
                    const double area_factor = r_outer * r_outer - r_inner * r_inner;

                    double Q_gas = 1000.0, Q_stars = 1000.0;
                    if(disc_gas_i > 0.0) {
                        Q_gas = c_s * Kappa * area_factor / (G * disc_gas_i);
                    }
                    if(disc_stars_i > 0.0) {
                        Q_stars = Kappa * sigma_stars * 0.935 * area_factor / (G * disc_stars_i);
                    }

                    /* Combined Q */
                    if(disc_gas_i > 0.0 && disc_stars_i > 0.0) {
                        const double W = 2.0 * sigma_stars * c_s / (sigma_stars * sigma_stars + c_s * c_s);
                        if(sigma_stars >= c_s) {
                            const double denom = W / Q_gas + 1.0 / Q_stars;
                            Q = (denom > 0.0) ? 1.0 / denom : 1000.0;
                        } else {
                            const double denom = 1.0 / Q_gas + W / Q_stars;
                            Q = (denom > 0.0) ? 1.0 / denom : 1000.0;
                        }
                    } else if(disc_gas_i <= 0.0) {
                        Q = Q_stars;
                    } else {
                        Q = Q_gas;
                    }
                }
            }
        }

        if(Q >= Q_min) continue;  /* Stable */

        /* StarSinkRate from DarkSage: varies with local velocity dispersion */
        /* When sigma_stars is small (cold disk), StarSinkRate ~ GasSinkRate */
        /* When sigma_stars is large (hot disk), StarSinkRate can be larger */
        double star_sink_rate = 1.0 - (1.0 - gas_sink_rate) * c_s / sigma_stars;
        if(star_sink_rate < 0.0) star_sink_rate = 0.0;
        if(star_sink_rate > 1.0) star_sink_rate = 1.0;

        /* Q_deficit is the fractional amount Q is below Q_min */
        double Q_deficit = (Q_min - Q) / Q_min;
        if(Q_deficit > 1.0) Q_deficit = 1.0;  /* Safety cap */

        /* DarkSage formula: unstable_mass = SinkRate * mass * (1 - Q/Q_min) */
        /* Apply f_global to smoothly suppress instability in globally stable disks */
        double gas_excess = f_global * gas_sink_rate * galaxies[p].DiscGas[i] * Q_deficit;
        double stars_excess = f_global * star_sink_rate * galaxies[p].DiscStars[i] * Q_deficit;

        /* Migrate stars inward one bin */
        if(stars_excess > 0.0) {
            deal_with_unstable_stars(p, i, stars_excess, galaxies, run_params);
        }

        /* Migrate gas inward one bin */
        if(gas_excess > 0.0) {
            deal_with_unstable_gas(p, i, gas_excess, galaxies, run_params);
        }

        /* Disk heating: increase velocity dispersion to restore stability */
        /* DarkSage formula: σ_new = σ_old * [(1-StarSinkRate) * Q_min/Q + StarSinkRate] */
        /* Part of instability resolved by mass transfer, part by heating */
        if(galaxies[p].DiscStars[i] > 0.0 && Q < Q_min) {
            double heating_factor = (1.0 - star_sink_rate) * (Q_min / Q) + star_sink_rate;
            if(heating_factor > 2.0) heating_factor = 2.0;  /* Cap extreme heating */
            if(heating_factor > 1.0) {
                galaxies[p].VelDispStars[i] *= (float)heating_factor;
            }
        }

        /* Safety clamps */
        if(galaxies[p].DiscStars[i] < 0.0) galaxies[p].DiscStars[i] = 0.0;
        if(galaxies[p].DiscStarsMetals[i] < 0.0) galaxies[p].DiscStarsMetals[i] = 0.0;
        if(galaxies[p].DiscGas[i] < 0.0) galaxies[p].DiscGas[i] = 0.0;
        if(galaxies[p].DiscGasMetals[i] < 0.0) galaxies[p].DiscGasMetals[i] = 0.0;
    }

    /* --- Pass 2: Innermost bin (bin 0) feeds bulge and BH --- */
    if(galaxies[p].DiscStars[0] <= 0.0 && galaxies[p].DiscGas[0] <= 0.0) return;

    /* Use radii in code units (Mpc/h) for DarkMode Q calculation */
    double r_outer_0 = galaxies[p].DiscRadii[1];

    if(r_outer_0 <= 0.0) return;

    double sigma_stars_0 = (double)galaxies[p].VelDispStars[0];
    if(sigma_stars_0 < 10.0) sigma_stars_0 = 10.0;

    /* Use DarkMode Q calculation with mass (not surface density) and j-bin Kappa */
    double Q0 = compute_combined_toomre_Q_darkmode(0, galaxies[p].DiscGas[0], galaxies[p].DiscStars[0],
                                                   c_s, sigma_stars_0, 0.0, r_outer_0, run_params);

    if(Q0 >= Q_min) return;  /* Stable */

    /* StarSinkRate for bin 0 */
    double star_sink_rate_0 = 1.0 - (1.0 - gas_sink_rate) * c_s / sigma_stars_0;
    if(star_sink_rate_0 < 0.0) star_sink_rate_0 = 0.0;
    if(star_sink_rate_0 > 1.0) star_sink_rate_0 = 1.0;

    /* Compute excess mass using combined Q deficit (same as outer bins) */
    double Q_deficit_0 = (Q_min - Q0) / Q_min;
    if(Q_deficit_0 > 1.0) Q_deficit_0 = 1.0;

    /* DarkSage formula with f_global suppression */
    double gas_excess_0 = f_global * gas_sink_rate * galaxies[p].DiscGas[0] * Q_deficit_0;
    double stars_excess_0 = f_global * star_sink_rate_0 * galaxies[p].DiscStars[0] * Q_deficit_0;

    /* --- J-conservation for bin 0 (DarkSage formula) --- */
    /* For bin 0, part of unstable mass goes to bulge/BH (m_down), part to bin 1 (m_up) */
    /* j_lose = midpoint j of bin 0 */
    /* j_gain = (DiscBinEdge[2] - DiscBinEdge[0]) / 2 = midpoint distance to bin 1 */
    double j_lose = 0.5 * (run_params->DiscBinEdge[1] - run_params->DiscBinEdge[0]);
    double j_gain = 0.5 * (run_params->DiscBinEdge[2] - run_params->DiscBinEdge[0]);

    /* Avoid division by zero if j_gain + j_lose is too small */
    double j_total = j_gain + j_lose;
    if(j_total <= 0.0) j_total = 1.0;  /* Safety fallback */

    /* --- Transfer excess stars from bin 0 with j-conservation --- */
    if(stars_excess_0 > 0.0) {
        double Z_s = (galaxies[p].DiscStars[0] > 0.0) ?
            galaxies[p].DiscStarsMetals[0] / galaxies[p].DiscStars[0] : 0.0;

        /* m_up goes to bin 1 to conserve j, m_down goes to bulge */
        double m_up_stars = j_lose / j_total * stars_excess_0;
        double m_down_stars = j_gain / j_total * stars_excess_0;

        /* Remove from bin 0 */
        galaxies[p].DiscStars[0] -= stars_excess_0;
        galaxies[p].DiscStarsMetals[0] -= Z_s * stars_excess_0;

        /* Add m_up to bin 1 */
        galaxies[p].DiscStars[1] += m_up_stars;
        galaxies[p].DiscStarsMetals[1] += Z_s * m_up_stars;

        /* Add m_down to bulge */
        galaxies[p].SecularBulgeMass += m_down_stars;
        galaxies[p].SecularMetalsBulgeMass += Z_s * m_down_stars;
        galaxies[p].BulgeMass += m_down_stars;
        galaxies[p].InstabilityBulgeMass += m_down_stars;
        galaxies[p].MetalsBulgeMass += Z_s * m_down_stars;

        /* Update bulge velocity dispersion */
        double sigma_old = galaxies[p].VelDispBulge;
        double m_old = galaxies[p].SecularBulgeMass - m_down_stars;
        if(m_old < 0.0) m_old = 0.0;
        double sigma_new = galaxies[p].Vvir / 3.0;
        if(sigma_new < 50.0) sigma_new = 50.0;
        double m_tot = m_old + m_down_stars;
        if(m_tot > 0.0 && sigma_old > 0.0) {
            double sq = (m_old * sigma_old * sigma_old +
                         m_down_stars * sigma_new * sigma_new) / m_tot;
            galaxies[p].VelDispBulge = (float)sqrt(sq);
        } else {
            galaxies[p].VelDispBulge = (float)sigma_new;
        }
    }

    /* --- Transfer excess gas from bin 0 with j-conservation --- */
    if(gas_excess_0 > 0.0) {
        double Z_g = (galaxies[p].DiscGas[0] > 0.0) ?
            galaxies[p].DiscGasMetals[0] / galaxies[p].DiscGas[0] : 0.0;
        double DTG_0 = 0.0;
        if(run_params->DustOn == 1 && galaxies[p].DiscGas[0] > 0.0) {
            DTG_0 = galaxies[p].DiscDust[0] / galaxies[p].DiscGas[0];
        }

        /* m_up goes to bin 1 to conserve j, m_down goes to BH */
        double m_up_gas = j_lose / j_total * gas_excess_0;
        double m_down_gas = j_gain / j_total * gas_excess_0;

        /* Apply BH growth rate and Vvir suppression to the m_down portion */
        double Vvir = galaxies[p].Vvir;
        double suppression_factor = 1.0 / (1.0 + (280.0 * 280.0) / (Vvir * Vvir));
        double BH_accrete = run_params->BlackHoleGrowthRate * suppression_factor * m_down_gas;
        if(BH_accrete > m_down_gas) {
            BH_accrete = m_down_gas;
        }

        /* Remove gas from bin 0 */
        galaxies[p].DiscGas[0] -= gas_excess_0;
        galaxies[p].DiscGasMetals[0] -= Z_g * gas_excess_0;
        galaxies[p].ColdGas -= (BH_accrete + (m_down_gas - BH_accrete));  /* m_down portion leaves cold gas */
        galaxies[p].MetalsColdGas -= Z_g * (BH_accrete + (m_down_gas - BH_accrete));
        if(run_params->DustOn == 1) {
            galaxies[p].DiscDust[0] -= DTG_0 * gas_excess_0;
            galaxies[p].ColdDust -= DTG_0 * (BH_accrete + (m_down_gas - BH_accrete));
        }

        /* Add m_up to bin 1 */
        galaxies[p].DiscGas[1] += m_up_gas;
        galaxies[p].DiscGasMetals[1] += Z_g * m_up_gas;
        if(run_params->DustOn == 1) {
            galaxies[p].DiscDust[1] += DTG_0 * m_up_gas;
        }

        /* Accrete onto BH */
        if(run_params->AGNrecipeOn > 0) {
            galaxies[p].BlackHoleMass += BH_accrete;
        }

        /* Form stars from gas that didn't go to BH (m_down - BH_accrete) */
        /* This is instability-driven star formation */
        double gas_for_sf = m_down_gas - BH_accrete;
        if(gas_for_sf > 0.0) {
            const double RecycleFraction = run_params->RecycleFraction;
            const double SfrEfficiency = run_params->SfrEfficiency;

            /* Gas that participates in SF (controlled by efficiency) */
            double gas_consumed = SfrEfficiency * gas_for_sf;
            /* Permanent stellar mass formed */
            double stars_formed = (1.0 - RecycleFraction) * gas_consumed;

            /* Stars form in innermost bin (bin 0) */
            galaxies[p].DiscStars[0] += stars_formed;
            galaxies[p].DiscStarsMetals[0] += Z_g * stars_formed;
            galaxies[p].StellarMass += stars_formed;
            galaxies[p].MetalsStellarMass += Z_g * stars_formed;

            /* Produce new metals from star formation (using Yield) */
            double new_metals = run_params->Yield * stars_formed;
            galaxies[p].DiscGasMetals[0] += new_metals;
            galaxies[p].MetalsColdGas += new_metals;
        }
    }

    /* Safety clamps */
    if(galaxies[p].DiscStars[0] < 0.0) galaxies[p].DiscStars[0] = 0.0;
    if(galaxies[p].DiscStarsMetals[0] < 0.0) galaxies[p].DiscStarsMetals[0] = 0.0;
    if(galaxies[p].DiscGas[0] < 0.0) galaxies[p].DiscGas[0] = 0.0;
    if(galaxies[p].DiscGasMetals[0] < 0.0) galaxies[p].DiscGasMetals[0] = 0.0;
    if(galaxies[p].ColdGas < 0.0) galaxies[p].ColdGas = 0.0;
    if(galaxies[p].MetalsColdGas < 0.0) galaxies[p].MetalsColdGas = 0.0;
    if(run_params->DustOn == 1) {
        if(galaxies[p].DiscDust[0] < 0.0) galaxies[p].DiscDust[0] = 0.0;
        if(galaxies[p].DiscDust[1] < 0.0) galaxies[p].DiscDust[1] = 0.0;
        if(galaxies[p].ColdDust < 0.0) galaxies[p].ColdDust = 0.0;
    }
}

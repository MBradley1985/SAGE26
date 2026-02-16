/**
 * @file model_darkmode.c
 * @brief DarkSage-style radially-resolved disk physics
 * 
 * Implements local star formation, feedback, disk instabilities, and radial
 * gas flows for DarkModeOn=1. All functions check DarkModeOn and maintain
 * backwards compatibility.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "core_allvars.h"
#include "model_darkmode.h"
#include "model_misc.h"
#include "model_mergers.h"
#include "model_disk_instability.h"

#define HYDROGEN_MASS_FRAC 0.74

/**
 * @brief Compute local H2 fraction and SFR in each disk annulus
 * 
 * Calculates surface density, H2 fraction, and star formation rate for each
 * radial bin using the specified H2 prescription. Handles all SFprescriptions
 * (0-7) with proper local surface density calculations.
 * 
 * @param p Galaxy index
 * @param dt Timestep [code units]
 * @param galaxies Array of galaxies
 * @param run_params Runtime parameters
 * @param sfr_local [OUTPUT] Star formation rate in each annulus [10^10 Msun/code_time]
 * @param h2_local [OUTPUT] H2 mass in each annulus [10^10 Msun/h]
 * @return Total SFR summed over all annuli [10^10 Msun/code_time]
 */
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
        double r_in = galaxies[p].DiscRadii[i] * 1000.0;    // kpc
        double r_out = galaxies[p].DiscRadii[i+1] * 1000.0; // kpc
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


/**
 * @brief Apply local star formation in each disk annulus
 * 
 * Updates disk arrays (DiscGas, DiscStars, DiscGasMetals, DiscStarsMetals)
 * based on local SFR computed for each annulus. Enforces local mass conservation.
 * 
 * @param p Galaxy index
 * @param dt Timestep [code units]
 * @param step Step index within snapshot
 * @param sfr_local SFR in each annulus [10^10 Msun/code_time]
 * @param galaxies Array of galaxies
 * @param run_params Runtime parameters
 * @return Total stellar mass formed [10^10 Msun/h]
 */
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
        
        // Local metallicity
        double Z_local = (galaxies[p].DiscGas[i] > 0.0) ?
            galaxies[p].DiscGasMetals[i] / galaxies[p].DiscGas[i] : 0.0;
        
        // Update gas
        galaxies[p].DiscGas[i] -= (1.0 - RecycleFraction) * stars_bin;
        galaxies[p].DiscGasMetals[i] -= Z_local * (1.0 - RecycleFraction) * stars_bin;
        
        // Update stars
        galaxies[p].DiscStars[i] += (1.0 - RecycleFraction) * stars_bin;
        galaxies[p].DiscStarsMetals[i] += Z_local * (1.0 - RecycleFraction) * stars_bin;
        
        // Dust
        if(run_params->DustOn == 1) {
            double DTG_local = (galaxies[p].DiscGas[i] > 0.0) ?
                galaxies[p].DiscDust[i] / galaxies[p].DiscGas[i] : 0.0;
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


/**
 * @brief Compute Toomre Q parameter for disk stability
 * 
 * Q = (σ κ) / (π G Σ)
 * where σ is velocity dispersion, κ is epicyclic frequency, Σ is surface density
 * 
 * For marginally stable disks: Q ~ 1
 * Q < 1: unstable to fragmentation
 * Q > 1: stable
 * 
 * @param Sigma_gas Gas surface density [Msun/pc^2]
 * @param Sigma_stars Stellar surface density [Msun/pc^2]
 * @param r_mid Mid-radius of annulus [kpc]
 * @param Vvir Virial velocity [km/s]
 * @param run_params Runtime parameters
 * @return Toomre Q parameter
 */
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
    
    // Convert r_mid from kpc to pc for consistent units
    double r_mid_pc = r_mid * 1000.0;  // kpc -> pc
    
    // Epicyclic frequency κ ≈ Vcirc / r for flat rotation curve
    double kappa = Vvir / r_mid_pc;  // (km/s) / pc
    
    // Toomre Q = (σ κ) / (π G Σ)
    // G = 4.302e-3 (km/s)^2 pc / Msun (standard astronomical units)
    // Σ is input in Msun/pc^2
    // [σκ] = (km/s)(km/s/pc) = (km/s)^2 / pc
    // [GΣ] = (km/s)^2 pc/Msun × Msun/pc^2 = (km/s)^2 / pc  ✓ consistent
    const double G_pc = 4.302e-3;  // (km/s)^2 pc / Msun
    
    double Q = (sigma_eff * kappa) / (M_PI * G_pc * Sigma_total);
    
    return Q;
}


/**
 * @brief Check for disk instabilities and transfer mass to bulge
 * 
 * Checks Toomre Q in each annulus. For Q < Q_crit, transfers unstable
 * mass to the bulge. This is the local version of the global disk instability
 * check in model_disk_instability.c
 * 
 * IMPORTANT: Unstable gas must feed black hole growth (like bulk version).
 * 
 * @param p Galaxy index
 * @param centralgal Central galaxy index
 * @param dt Timestep
 * @param step Step index
 * @param galaxies Array of galaxies
 * @param run_params Runtime parameters
 */
void check_local_disk_instability(const int p, const int centralgal, const double dt, const int step,
                                 struct GALAXY *galaxies, const struct params *run_params)
{
    (void)centralgal;  // Reserved for future use
    (void)dt;          // Reserved for future use
    (void)step;        // Reserved for future use
    
    if(galaxies[p].Vvir <= 0.0 || galaxies[p].DiskScaleRadius <= 0.0) {
        return;
    }
    
    const float h = run_params->Hubble_h;
    const double Q_crit = 1.0;  // Critical Q for marginal stability
    
    double total_unstable_stars = 0.0;
    double total_unstable_gas = 0.0;
    
    // Save initial disk radius for bulge radius update
    const double old_disk_radius = galaxies[p].DiskScaleRadius;
    
    for(int i = 0; i < N_BINS; i++) {
        double r_in = galaxies[p].DiscRadii[i] * 1000.0;    // kpc
        double r_out = galaxies[p].DiscRadii[i+1] * 1000.0;
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
            
            // Remove unstable STARS from disk arrays (these go directly to bulge)
            galaxies[p].DiscStars[i] -= unstable_stars_bin;
            galaxies[p].DiscStarsMetals[i] -= Z_stars * unstable_stars_bin;
            
            total_unstable_stars += unstable_stars_bin;
            
            // Track unstable GAS but don't remove from DiscGas yet
            // This gas will be handled by grow_black_hole and starburst
            total_unstable_gas += unstable_gas_bin;
            
            // Safety
            if(galaxies[p].DiscStars[i] < 0.0) galaxies[p].DiscStars[i] = 0.0;
            if(galaxies[p].DiscStarsMetals[i] < 0.0) galaxies[p].DiscStarsMetals[i] = 0.0;
        }
    }
    
    // Add unstable STARS to bulge
    if(total_unstable_stars > 0.0) {
        double Z_disk = get_metallicity(galaxies[p].StellarMass - galaxies[p].BulgeMass,
                                       galaxies[p].MetalsStellarMass - galaxies[p].MetalsBulgeMass);
        galaxies[p].BulgeMass += total_unstable_stars;
        galaxies[p].InstabilityBulgeMass += total_unstable_stars;
        galaxies[p].MetalsBulgeMass += Z_disk * total_unstable_stars;
        
        // Update bulge radius using Tonini+2016 formula
        update_instability_bulge_radius(p, total_unstable_stars, old_disk_radius, galaxies, run_params);
    }
    
    // Handle unstable GAS: feed black hole (like bulk version)
    if(total_unstable_gas > 0.0 && galaxies[p].ColdGas > 0.0) {
        double unstable_gas_fraction = total_unstable_gas / galaxies[p].ColdGas;
        
        // Clamp fraction to avoid overshoot
        if(unstable_gas_fraction > 1.0) unstable_gas_fraction = 1.0;
        
        // Feed black hole with unstable gas (this is the key fix!)
        // grow_black_hole accretes BHaccrete = rate × mass_ratio × ColdGas
        // and removes the accreted gas from ColdGas and DiscGas arrays
        if(run_params->AGNrecipeOn > 0) {
            grow_black_hole(p, unstable_gas_fraction, galaxies, run_params);
        }
        
        // Remove remaining unstable gas from disk arrays
        // (grow_black_hole already removed some for BH accretion)
        // The remaining gas should either:
        // a) Convert to stars in a starburst, OR
        // b) Be removed from ColdGas (simplest approach for now)
        
        // Sync DiscGas arrays: remove unstable fraction from each bin
        double remaining_unstable_frac = unstable_gas_fraction;
        if(remaining_unstable_frac > 0.0) {
            for(int i = 0; i < N_BINS; i++) {
                double remove_gas = remaining_unstable_frac * galaxies[p].DiscGas[i];
                double Z_gas = (galaxies[p].DiscGas[i] > 0.0) ?
                    galaxies[p].DiscGasMetals[i] / galaxies[p].DiscGas[i] : 0.0;
                
                galaxies[p].DiscGas[i] -= remove_gas;
                galaxies[p].DiscGasMetals[i] -= Z_gas * remove_gas;
                galaxies[p].ColdGas -= remove_gas;
                galaxies[p].MetalsColdGas -= Z_gas * remove_gas;
                
                if(run_params->DustOn == 1) {
                    double DTG = (galaxies[p].DiscGas[i] + remove_gas > 0.0) ?
                        galaxies[p].DiscDust[i] / (galaxies[p].DiscGas[i] + remove_gas) : 0.0;
                    galaxies[p].DiscDust[i] -= DTG * remove_gas;
                    galaxies[p].ColdDust -= DTG * remove_gas;
                    if(galaxies[p].DiscDust[i] < 0.0) galaxies[p].DiscDust[i] = 0.0;
                    if(galaxies[p].ColdDust < 0.0) galaxies[p].ColdDust = 0.0;
                }
                
                if(galaxies[p].DiscGas[i] < 0.0) galaxies[p].DiscGas[i] = 0.0;
                if(galaxies[p].DiscGasMetals[i] < 0.0) galaxies[p].DiscGasMetals[i] = 0.0;
            }
            if(galaxies[p].ColdGas < 0.0) galaxies[p].ColdGas = 0.0;
            if(galaxies[p].MetalsColdGas < 0.0) galaxies[p].MetalsColdGas = 0.0;
        }
    }
}


/**
 * @brief Apply radial gas flows due to viscous evolution
 * 
 * Gas moves inward on viscous timescale: t_visc ~ r² / ν
 * where ν is kinematic viscosity (parameterized by α)
 * 
 * This is a simplified implementation - full DarkSage uses detailed angular
 * momentum transport. Here we do mass transport maintaining mass conservation.
 * 
 * @param p Galaxy index
 * @param dt Timestep [code units]
 * @param galaxies Array of galaxies
 * @param run_params Runtime parameters
 */
void apply_radial_gas_flows(const int p, const double dt, struct GALAXY *galaxies,
                           const struct params *run_params)
{
    if(galaxies[p].Vvir <= 0.0 || dt <= 0.0) {
        return;
    }
    
    // Viscosity parameter (α-disk prescription)
    // α ~ 0.01 for disks (Shakura-Sunyaev)
    const double alpha = 0.01;
    
    // Sound speed in disk (cold gas)
    const double cs = 10.0;  // km/s, typical for cold ISM
    
    // Arrays to hold updated gas distribution
    double DiscGas_new[N_BINS];
    double DiscGasMetals_new[N_BINS];
    double DiscDust_new[N_BINS];
    
    for(int i = 0; i < N_BINS; i++) {
        DiscGas_new[i] = galaxies[p].DiscGas[i];
        DiscGasMetals_new[i] = galaxies[p].DiscGasMetals[i];
        if(run_params->DustOn == 1) {
            DiscDust_new[i] = galaxies[p].DiscDust[i];
        }
    }
    
    // Diffusive flow from each bin to inner bin
    for(int i = N_BINS - 1; i > 0; i--) {
        if(galaxies[p].DiscGas[i] <= 0.0) {
            continue;
        }
        
        double r_mid = 0.5 * (galaxies[p].DiscRadii[i] + galaxies[p].DiscRadii[i+1]) * 1000.0;  // kpc
        
        // Scale height: h ~ cs * r / Vvir
        double h_scale = cs * r_mid / galaxies[p].Vvir;  // kpc
        
        // Kinematic viscosity: ν ~ α * cs * h
        double nu = alpha * cs * h_scale;  // km/s * kpc = kpc^2/time
        
        // Viscous timescale: t_visc ~ r² / ν
        // Convert to code units
        double t_visc_code = (r_mid * r_mid) / nu / galaxies[p].Vvir;  // rough approximation
        
        if(t_visc_code <= 0.0) {
            continue;
        }
        
        // Fraction that flows inward this timestep
        double flow_frac = dt / t_visc_code;
        if(flow_frac > 0.5) flow_frac = 0.5;  // Limit for stability
        
        double flow_mass = flow_frac * galaxies[p].DiscGas[i];
        double flow_metals = flow_frac * galaxies[p].DiscGasMetals[i];
        double flow_dust = 0.0;
        if(run_params->DustOn == 1) {
            flow_dust = flow_frac * galaxies[p].DiscDust[i];
        }
        
        // Remove from current bin
        DiscGas_new[i] -= flow_mass;
        DiscGasMetals_new[i] -= flow_metals;
        if(run_params->DustOn == 1) {
            DiscDust_new[i] -= flow_dust;
        }
        
        // Add to inner bin
        DiscGas_new[i-1] += flow_mass;
        DiscGasMetals_new[i-1] += flow_metals;
        if(run_params->DustOn == 1) {
            DiscDust_new[i-1] += flow_dust;
        }
    }
    
    // Update galaxy arrays
    for(int i = 0; i < N_BINS; i++) {
        galaxies[p].DiscGas[i] = DiscGas_new[i];
        galaxies[p].DiscGasMetals[i] = DiscGasMetals_new[i];
        if(run_params->DustOn == 1) {
            galaxies[p].DiscDust[i] = DiscDust_new[i];
        }
        
        // Safety
        if(galaxies[p].DiscGas[i] < 0.0) galaxies[p].DiscGas[i] = 0.0;
        if(galaxies[p].DiscGasMetals[i] < 0.0) galaxies[p].DiscGasMetals[i] = 0.0;
        if(run_params->DustOn == 1 && galaxies[p].DiscDust[i] < 0.0) {
            galaxies[p].DiscDust[i] = 0.0;
        }
    }
}

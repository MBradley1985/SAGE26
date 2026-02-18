#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "core_allvars.h"
#include "model_darkmode.h"
#include "model_misc.h"

#define HYDROGEN_MASS_FRAC 0.74

/**
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


/**
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


/**
 *  
 * Checks Toomre Q in each annulus. For Q < Q_crit, transfers unstable
 * mass to the bulge. This is the local version of the global disk instability
 * check in model_disk_instability.c
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


/**
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

    const double h = run_params->Hubble_h;

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

        double r_mid = 0.5 * (galaxies[p].DiscRadii[i] + galaxies[p].DiscRadii[i+1]) * 1000.0 / h;  // physical kpc

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


/* ========================================================================== */
/* FULL DARKMODE PHYSICS FUNCTIONS                                            */
/* Enhanced disk physics: combined Q, precession, j-conservation              */
/* ========================================================================== */

/**
 * Rotate a 3D vector around an axis using Rodrigues' rotation formula.
 *
 * v_rot = v*cos(θ) + (k × v)*sin(θ) + k*(k·v)*(1 - cos(θ))
 *
 * @param v         Input vector to rotate (modified in place)
 * @param axis      Rotation axis (must be unit vector)
 * @param angle_deg Rotation angle in degrees
 */
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


/**
 * Compute total angular momentum vector of gas or stellar disc.
 * J = Σ_i m_i * j_i * spin_direction
 * where j_i is the specific angular momentum of annulus i.
 *
 * @param p           Galaxy index
 * @param component   0 = gas, 1 = stars
 * @param J           Output angular momentum vector [3]
 * @param galaxies    Galaxy array
 * @param run_params  Runtime parameters
 * @return            Total angular momentum magnitude
 */
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


/**
 * Precess gas disc toward stellar disc.
 * The gas disc aligns with the stellar disc over time due to dynamical friction
 * and torques. The precession rate is parameterized by DegPerTdyn.
 *
 * @param p          Galaxy index
 * @param dt         Timestep [code units]
 * @param galaxies   Galaxy array
 * @param run_params Runtime parameters
 */
void precess_gas(const int p, const double dt, struct GALAXY *galaxies,
                 const struct params *run_params)
{
    if(run_params->FullDarkModeOn != 1) return;

    /* Calculate angle between gas and stellar spins */
    double cos_theta = 0.0;
    for(int j = 0; j < 3; j++) {
        cos_theta += galaxies[p].SpinGas[j] * galaxies[p].SpinStars[j];
    }

    /* Clamp to [-1, 1] for numerical safety */
    if(cos_theta > 1.0) cos_theta = 1.0;
    if(cos_theta < -1.0) cos_theta = -1.0;

    double theta_deg = acos(cos_theta) * 180.0 / M_PI;

    /* If already coplanar (within threshold), no precession needed */
    if(theta_deg < run_params->ThetaThresh) {
        return;
    }

    /* Calculate dynamical time at half-mass radius */
    double r_half = galaxies[p].DiskScaleRadius * 1.68;  /* ~1.68 r_s for exponential */
    double tdyn = (galaxies[p].Vvir > 0.0) ? r_half / galaxies[p].Vvir : 1.0;

    if(tdyn <= 0.0) return;

    /* Precession angle this timestep */
    double precess_deg = run_params->DegPerTdyn * dt / tdyn;

    /* Don't overshoot */
    if(precess_deg > theta_deg - run_params->ThetaThresh) {
        precess_deg = theta_deg - run_params->ThetaThresh;
    }

    if(precess_deg <= 0.0) return;

    /* Find rotation axis: perpendicular to both spins */
    float axis[3];
    axis[0] = galaxies[p].SpinGas[1] * galaxies[p].SpinStars[2] -
              galaxies[p].SpinGas[2] * galaxies[p].SpinStars[1];
    axis[1] = galaxies[p].SpinGas[2] * galaxies[p].SpinStars[0] -
              galaxies[p].SpinGas[0] * galaxies[p].SpinStars[2];
    axis[2] = galaxies[p].SpinGas[0] * galaxies[p].SpinStars[1] -
              galaxies[p].SpinGas[1] * galaxies[p].SpinStars[0];

    /* Normalize axis */
    double axis_mag = sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]);
    if(axis_mag > 0.0) {
        for(int j = 0; j < 3; j++) {
            axis[j] /= (float)axis_mag;
        }

        /* Rotate gas spin toward stellar spin */
        rotate_vector(galaxies[p].SpinGas, axis, precess_deg);

        /* Re-normalize after rotation */
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


/**
 * Compute combined two-fluid Toomre Q parameter (Romeo & Wiegert 2011).
 * Accounts for both gas and stellar components with proper weighting.
 *
 * Q_tot = 1 / (W/Q_gas + 1/Q_star)  if σ_R >= σ_gas
 * Q_tot = 1 / (1/Q_gas + W/Q_star)  if σ_R < σ_gas
 *
 * where W = 2*σ_R*σ_gas / (σ_R² + σ_gas²)
 *
 * @param Sigma_gas    Gas surface density [Msun/pc^2]
 * @param Sigma_stars  Stellar surface density [Msun/pc^2]
 * @param sigma_gas    Gas velocity dispersion [km/s]
 * @param sigma_stars  Stellar radial velocity dispersion [km/s]
 * @param r_mid        Mid-radius of annulus [kpc]
 * @param Vvir         Virial velocity [km/s]
 * @return             Combined Toomre Q parameter
 */
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


/**
 * Handle unstable gas: sink it inward with angular momentum conservation.
 * Unstable gas moves inward, losing specific angular momentum.
 * Mass conservation: m_up/m_down = j_lose/j_gain
 *
 * @param p           Galaxy index
 * @param bin         Annulus index of unstable gas
 * @param unstable_gas Mass of unstable gas in this bin
 * @param galaxies    Galaxy array
 * @param run_params  Runtime parameters
 */
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


/**
 * Handle unstable stars: migrate inward one bin with angular momentum conservation.
 * Mirrors deal_with_unstable_gas but for the stellar component.
 * Stars move inward one bin per call, conserving total angular momentum.
 *
 * @param p              Galaxy index
 * @param bin            Annulus index of unstable stars
 * @param unstable_stars Mass of unstable stars to migrate
 * @param galaxies       Galaxy array
 * @param run_params     Runtime parameters
 */
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


/**
 * Check for disk instabilities using combined Toomre Q and j-conservation.
 * Enhanced instability check for FullDarkMode.
 *
 * Approach: compute the exact excess surface density above marginal stability
 * (Q = Q_min) and transfer only that excess. Capped at GasSinkRate per timestep
 * to prevent over-stripping in massive galaxies.
 *
 * Stars migrate inward one bin at a time with j-conservation.
 * Only the innermost bin (bin 0) feeds the secular bulge and BH.
 *
 * @param p           Galaxy index
 * @param centralgal  Central galaxy index
 * @param dt          Timestep [code units]
 * @param step        Step index within snapshot
 * @param galaxies    Galaxy array
 * @param run_params  Runtime parameters
 */
void check_full_disk_instability(const int p, const int centralgal, const double dt, const int step,
                                 struct GALAXY *galaxies, const struct params *run_params)
{
    (void)centralgal;
    (void)step;
    (void)dt;

    if(run_params->FullDarkModeOn != 1) return;
    if(galaxies[p].Vvir <= 0.0 || galaxies[p].DiskScaleRadius <= 0.0) return;

    /* --- Global stability check (Mo, Mao & White 1998) --- */
    /* Only proceed with local ToomreQ instability if disk fails global criterion */
    /* This prevents intermediate-mass galaxies with globally stable disks from */
    /* losing mass to bulge through local instabilities */
    const double diskmass = galaxies[p].ColdGas +
                           (galaxies[p].StellarMass - galaxies[p].BulgeMass);
    if(diskmass <= 0.0) return;

    const double Mcrit = galaxies[p].Vmax * galaxies[p].Vmax *
                        (3.0 * galaxies[p].DiskScaleRadius) / run_params->G;

    /* If disk mass < critical mass, disk is globally stable - skip local instability */
    if(diskmass <= Mcrit) return;

    const double h = run_params->Hubble_h;
    const double Q_min = run_params->QTotMin;
    const double sink_rate = run_params->GasSinkRate;  /* max fraction transferred per call */

    /* Sound speed for gas (cold ISM) */
    const double sigma_gas = 10.0;  /* km/s */

    /* --- Pass 1: bins 1..N_BINS-1 migrate excess mass inward one bin --- */
    for(int i = N_BINS - 1; i >= 1; i--) {
        if(galaxies[p].DiscGas[i] <= 0.0 && galaxies[p].DiscStars[i] <= 0.0) continue;

        double r_in = galaxies[p].DiscRadii[i] * 1000.0 / h;
        double r_out = galaxies[p].DiscRadii[i+1] * 1000.0 / h;
        double r_mid = 0.5 * (r_in + r_out);
        double area_pc2 = M_PI * (r_out * r_out - r_in * r_in) * 1.0e6;

        if(area_pc2 <= 0.0 || r_mid <= 0.0) continue;

        double Sigma_gas = (galaxies[p].DiscGas[i] * 1.0e10 / h) / area_pc2;
        double Sigma_stars = (galaxies[p].DiscStars[i] * 1.0e10 / h) / area_pc2;

        double sigma_stars = (double)galaxies[p].VelDispStars[i];
        if(sigma_stars < 10.0) sigma_stars = 10.0;

        double Q = compute_combined_toomre_Q(Sigma_gas, Sigma_stars, sigma_gas, sigma_stars,
                                            r_mid, galaxies[p].Vvir);

        if(Q >= Q_min) continue;  /* Stable */

        /* Compute excess mass using combined Q deficit (matches DarkSage approach) */
        /* Q_deficit is the fractional amount Q is below Q_min */
        /* This ensures we only transfer enough mass to restore stability */
        double Q_deficit = (Q_min - Q) / Q_min;
        if(Q_deficit > 1.0) Q_deficit = 1.0;  /* Safety cap */

        /* Transfer proportionally from both components */
        /* This respects the two-fluid nature of the combined Q calculation */
        double gas_excess = Q_deficit * galaxies[p].DiscGas[i];
        double stars_excess = Q_deficit * galaxies[p].DiscStars[i];

        /* Cap at sink_rate fraction of current bin mass (per-timestep limit) */
        if(gas_excess > sink_rate * galaxies[p].DiscGas[i]) {
            gas_excess = sink_rate * galaxies[p].DiscGas[i];
        }
        if(stars_excess > sink_rate * galaxies[p].DiscStars[i]) {
            stars_excess = sink_rate * galaxies[p].DiscStars[i];
        }

        /* Migrate stars inward one bin */
        if(stars_excess > 0.0) {
            deal_with_unstable_stars(p, i, stars_excess, galaxies, run_params);
        }

        /* Migrate gas inward one bin */
        if(gas_excess > 0.0) {
            deal_with_unstable_gas(p, i, gas_excess, galaxies, run_params);
        }

        /* Disk heating: increase velocity dispersion to restore stability */
        /* This is crucial to prevent repeated instability on the same annulus */
        /* Formula from DarkSage: σ_new = σ_old * [(1-sink_rate) * Q_min/Q + sink_rate] */
        /* Part of instability resolved by mass transfer, part by heating */
        if(galaxies[p].DiscStars[i] > 0.0 && Q < Q_min) {
            double heating_factor = (1.0 - sink_rate) * (Q_min / Q) + sink_rate;
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

    double r_out_0 = galaxies[p].DiscRadii[1] * 1000.0 / h;
    double area_pc2_0 = M_PI * r_out_0 * r_out_0 * 1.0e6;

    if(area_pc2_0 <= 0.0) return;

    double Sigma_gas_0 = (galaxies[p].DiscGas[0] * 1.0e10 / h) / area_pc2_0;
    double Sigma_stars_0 = (galaxies[p].DiscStars[0] * 1.0e10 / h) / area_pc2_0;
    double sigma_stars_0 = (double)galaxies[p].VelDispStars[0];
    if(sigma_stars_0 < 10.0) sigma_stars_0 = 10.0;

    double r_mid_0 = 0.5 * r_out_0;
    double Q0 = compute_combined_toomre_Q(Sigma_gas_0, Sigma_stars_0, sigma_gas, sigma_stars_0,
                                          r_mid_0, galaxies[p].Vvir);

    if(Q0 >= Q_min) return;  /* Stable */

    /* Compute excess mass using combined Q deficit (same as outer bins) */
    double Q_deficit_0 = (Q_min - Q0) / Q_min;
    if(Q_deficit_0 > 1.0) Q_deficit_0 = 1.0;

    /* Transfer proportionally from both components */
    double gas_excess_0 = Q_deficit_0 * galaxies[p].DiscGas[0];
    double stars_excess_0 = Q_deficit_0 * galaxies[p].DiscStars[0];

    /* Cap at sink_rate fraction */
    if(gas_excess_0 > sink_rate * galaxies[p].DiscGas[0]) {
        gas_excess_0 = sink_rate * galaxies[p].DiscGas[0];
    }
    if(stars_excess_0 > sink_rate * galaxies[p].DiscStars[0]) {
        stars_excess_0 = sink_rate * galaxies[p].DiscStars[0];
    }

    /* Transfer excess stars from bin 0 → secular bulge */
    if(stars_excess_0 > 0.0) {
        double Z_s = (galaxies[p].DiscStars[0] > 0.0) ?
            galaxies[p].DiscStarsMetals[0] / galaxies[p].DiscStars[0] : 0.0;

        galaxies[p].DiscStars[0] -= stars_excess_0;
        galaxies[p].DiscStarsMetals[0] -= Z_s * stars_excess_0;

        galaxies[p].SecularBulgeMass += stars_excess_0;
        galaxies[p].SecularMetalsBulgeMass += Z_s * stars_excess_0;
        galaxies[p].BulgeMass += stars_excess_0;
        galaxies[p].InstabilityBulgeMass += stars_excess_0;
        galaxies[p].MetalsBulgeMass += Z_s * stars_excess_0;

        /* Update bulge velocity dispersion */
        double sigma_old = galaxies[p].VelDispBulge;
        double m_old = galaxies[p].SecularBulgeMass - stars_excess_0;
        if(m_old < 0.0) m_old = 0.0;
        double sigma_new = galaxies[p].Vvir / 3.0;
        if(sigma_new < 50.0) sigma_new = 50.0;
        double m_tot = m_old + stars_excess_0;
        if(m_tot > 0.0 && sigma_old > 0.0) {
            double sq = (m_old * sigma_old * sigma_old +
                         stars_excess_0 * sigma_new * sigma_new) / m_tot;
            galaxies[p].VelDispBulge = (float)sqrt(sq);
        } else {
            galaxies[p].VelDispBulge = (float)sigma_new;
        }
    }

    /* Transfer excess gas from bin 0 → BH accretion */
    /* Apply Vvir-dependent suppression (same as grow_black_hole in model_mergers.c) */
    /* This prevents low-mass galaxies from growing overly massive BHs */
    if(gas_excess_0 > 0.0 && run_params->AGNrecipeOn > 0) {
        double Z_g = (galaxies[p].DiscGas[0] > 0.0) ?
            galaxies[p].DiscGasMetals[0] / galaxies[p].DiscGas[0] : 0.0;
        double DTG_0 = 0.0;
        if(run_params->DustOn == 1 && galaxies[p].DiscGas[0] > 0.0) {
            DTG_0 = galaxies[p].DiscDust[0] / galaxies[p].DiscGas[0];
        }

        /* Calculate BH accretion with Vvir suppression factor */
        /* Suppression: 1 / (1 + (280/Vvir)^2) - strongly suppresses BH growth in low-Vvir galaxies */
        double Vvir = galaxies[p].Vvir;
        double suppression_factor = 1.0 / (1.0 + (280.0 * 280.0) / (Vvir * Vvir));

        /* BH accretion is limited by efficiency and suppression */
        double BH_accrete = run_params->BlackHoleGrowthRate * suppression_factor * gas_excess_0;

        /* Cannot accrete more than the excess gas */
        if(BH_accrete > gas_excess_0) {
            BH_accrete = gas_excess_0;
        }

        /* Remove only the accreted gas from innermost bin */
        galaxies[p].DiscGas[0] -= BH_accrete;
        galaxies[p].DiscGasMetals[0] -= Z_g * BH_accrete;
        galaxies[p].ColdGas -= BH_accrete;
        galaxies[p].MetalsColdGas -= Z_g * BH_accrete;
        if(run_params->DustOn == 1) {
            galaxies[p].DiscDust[0] -= DTG_0 * BH_accrete;
            galaxies[p].ColdDust -= DTG_0 * BH_accrete;
        }

        /* Accrete onto BH */
        galaxies[p].BlackHoleMass += BH_accrete;

        /* Form stars from remaining excess gas (instability-driven starburst) */
        /* This is crucial for metal production - matches DarkSage behavior */
        /* Apply SfrEfficiency to control how much of the excess gas forms stars */
        double gas_for_sf = gas_excess_0 - BH_accrete;
        if(gas_for_sf > 0.0 && galaxies[p].DiscGas[0] >= gas_for_sf) {
            const double RecycleFraction = run_params->RecycleFraction;
            const double SfrEfficiency = run_params->SfrEfficiency;

            /* Gas that participates in SF (controlled by efficiency) */
            double gas_consumed = SfrEfficiency * gas_for_sf;
            /* Permanent stellar mass formed */
            double stars_formed = (1.0 - RecycleFraction) * gas_consumed;

            /* Update gas: remove gas that forms stars (net of recycling) */
            galaxies[p].DiscGas[0] -= stars_formed;
            galaxies[p].DiscGasMetals[0] -= Z_g * stars_formed;
            galaxies[p].ColdGas -= stars_formed;
            galaxies[p].MetalsColdGas -= Z_g * stars_formed;
            if(run_params->DustOn == 1) {
                galaxies[p].DiscDust[0] -= DTG_0 * stars_formed;
                galaxies[p].ColdDust -= DTG_0 * stars_formed;
            }

            /* Update stars: add newly formed stars to innermost bin */
            galaxies[p].DiscStars[0] += stars_formed;
            galaxies[p].DiscStarsMetals[0] += Z_g * stars_formed;
            galaxies[p].StellarMass += stars_formed;
            galaxies[p].MetalsStellarMass += Z_g * stars_formed;

            /* Produce new metals from star formation (using Yield) */
            /* New metals go back to the gas phase */
            double new_metals = run_params->Yield * stars_formed;
            galaxies[p].DiscGasMetals[0] += new_metals;
            galaxies[p].MetalsColdGas += new_metals;

        }

        /* Safety clamps */
        if(galaxies[p].DiscGas[0] < 0.0) galaxies[p].DiscGas[0] = 0.0;
        if(galaxies[p].DiscGasMetals[0] < 0.0) galaxies[p].DiscGasMetals[0] = 0.0;
        if(galaxies[p].ColdGas < 0.0) galaxies[p].ColdGas = 0.0;
        if(galaxies[p].MetalsColdGas < 0.0) galaxies[p].MetalsColdGas = 0.0;
        if(run_params->DustOn == 1) {
            if(galaxies[p].DiscDust[0] < 0.0) galaxies[p].DiscDust[0] = 0.0;
            if(galaxies[p].ColdDust < 0.0) galaxies[p].ColdDust = 0.0;
        }
        if(galaxies[p].DiscStars[0] < 0.0) galaxies[p].DiscStars[0] = 0.0;
        if(galaxies[p].DiscStarsMetals[0] < 0.0) galaxies[p].DiscStarsMetals[0] = 0.0;
    }

    if(galaxies[p].DiscStars[0] < 0.0) galaxies[p].DiscStars[0] = 0.0;
    if(galaxies[p].DiscStarsMetals[0] < 0.0) galaxies[p].DiscStarsMetals[0] = 0.0;
}

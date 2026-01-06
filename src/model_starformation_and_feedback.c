#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "core_allvars.h"

#include "model_starformation_and_feedback.h"
#include "model_misc.h"
#include "model_disk_instability.h"


void starformation_and_feedback(const int p, const int centralgal, const double time, const double dt, const int halonr, const int step,
                                struct GALAXY *galaxies, const struct params *run_params)
{
    // BUG FIX: Validate step is within array bounds
    XASSERT(step >= 0 && step < STEPS, -1,
            "Error: step = %d is out of bounds [0, %d)\n", step, STEPS);

    // ========================================================================
    // CHECK FOR FFB REGIME - EARLY EXIT IF FFB
    // ========================================================================
    if(run_params->FeedbackFreeModeOn == 1 && galaxies[p].FFBRegime == 1) {
        // This is a Feedback-Free Burst halo
        // Use specialized FFB star formation (no feedback)
        starformation_ffb(p, centralgal, dt, step, galaxies, run_params);
        return;  // Exit early - FFB path complete
    }

    double reff, tdyn, strdot, stars, ejected_mass, metallicity, total_molecular_gas;

    // Initialise variables
    strdot = 0.0;
    metallicity = 0.0;

    // star formation recipes
    if(run_params->SFprescription == 0) {
        // we take the typical star forming region as 3.0*r_s using the Milky Way as a guide
        reff = 3.0 * galaxies[p].DiskScaleRadius;

        // BUG FIX: Check Vvir > 0 before division to avoid NaN/Inf
        if(galaxies[p].Vvir <= 0.0) {
            strdot = 0.0;
        } else {
            tdyn = reff / galaxies[p].Vvir;

            // from Kauffmann (1996) eq7 x piR^2, (Vvir in km/s, reff in Mpc/h) in units of 10^10Msun/h
            const double cold_crit = 0.19 * galaxies[p].Vvir * reff;
            if(galaxies[p].ColdGas > cold_crit && tdyn > 0.0) {
                strdot = run_params->SfrEfficiency * (galaxies[p].ColdGas - cold_crit) / tdyn;
            } else {
                strdot = 0.0;
            }
        }
    } else if(run_params->SFprescription == 1) {

        // ========================================================================
        // Blitz and Rosolowsky (2006) - BR06 Model
        // ========================================================================

        // we take the typical star forming region as 3.0*r_s using the Milky Way as a guide
        reff = 3.0 * galaxies[p].DiskScaleRadius;

        // BUG FIX: Check Vvir > 0 before division
        if(galaxies[p].Vvir <= 0.0) {
            galaxies[p].H2gas = 0.0;
            strdot = 0.0;
        } else {
            tdyn = reff / galaxies[p].Vvir;
            // BR06 model
            const float h = run_params->Hubble_h;
            const float rs_pc = galaxies[p].DiskScaleRadius * 1.0e6 / h;
            if (rs_pc <= 0.0) {
                galaxies[p].H2gas = 0.0;
                strdot = 0.0;
            } else {
                // float disk_area_pc2 = M_PI * rs_pc * rs_pc;
                float disk_area_pc2 = M_PI * pow(3.0 * rs_pc, 2); // 3× scale radius captures ~95% of mass
                float gas_surface_density = (galaxies[p].ColdGas * 1.0e10 / h) / disk_area_pc2; // M☉/pc²
                float stellar_surface_density = (galaxies[p].StellarMass * 1.0e10 / h) / disk_area_pc2; // M☉/pc²

                total_molecular_gas = calculate_molecular_fraction_BR06(gas_surface_density, stellar_surface_density,
                                                                       rs_pc) * galaxies[p].ColdGas;

                galaxies[p].H2gas = total_molecular_gas;

                if (galaxies[p].H2gas > 0.0 && tdyn > 0.0) {
                    strdot = run_params->SfrEfficiency * galaxies[p].H2gas / tdyn;
                } else {
                    strdot = 0.0;
                }
            }
        }
    } else if(run_params->SFprescription == 2) {

        // Somerville et al. 2025: Density Modulated Star Formation Efficiency
        // Using Equation 3 for efficiency: epsilon = (Sigma/Sigma_crit)/(1 + Sigma/Sigma_crit)

        // we take the typical star forming region as 3.0*r_s using the Milky Way as a guide
        reff = 3.0 * galaxies[p].DiskScaleRadius;

        // BUG FIX: Check Vvir > 0 before division
        if(galaxies[p].Vvir <= 0.0) {
            strdot = 0.0;
        } else {
            tdyn = reff / galaxies[p].Vvir;
            const float h = run_params->Hubble_h;
            const float rs_pc = galaxies[p].DiskScaleRadius * 1.0e6 / h;
            float disk_area_pc2 = M_PI * pow(3.0 * rs_pc, 2); // pc^2
            float gas_surface_density = (disk_area_pc2 > 0.0) ?
                (galaxies[p].ColdGas * 1.0e10 / h) / disk_area_pc2 : 0.0; // Msun/pc^2

            // Critical surface density from Equation 2
            const double Sigma_crit = 30.0 / (M_PI * 4.302e-3); // ~2176 Msun/pc^2

            // Cloud-scale star formation efficiency from Equation 3
            double epsilon_cl = (gas_surface_density / Sigma_crit) / (1.0 + gas_surface_density / Sigma_crit);

            // Fraction of gas in dense clouds (f_dense from Equation 8)
            const double f_dense = 0.5;

            // Star formation rate: SFR ~ epsilon_cl * f_dense * m_gas / tdyn
            if(tdyn > 0.0 && gas_surface_density > 0.0) {
                strdot = epsilon_cl * f_dense * galaxies[p].ColdGas / tdyn;
            } else {
                strdot = 0.0;
            }
        }
    } else if(run_params->SFprescription == 3) {

        // Somerville et al. 2025: Density Modulated Star Formation Efficiency with H2
        // Using Equation 3 for efficiency: epsilon = (Sigma/Sigma_crit)/(1 + Sigma/Sigma_crit)
        // But replacing cold gas with H2 gas using Blitz & Rosolowsky 2006

        // we take the typical star forming region as 3.0*r_s using the Milky Way as a guide
        reff = 3.0 * galaxies[p].DiskScaleRadius;

        // BUG FIX: Check Vvir > 0 before division
        if(galaxies[p].Vvir <= 0.0) {
            galaxies[p].H2gas = 0.0;
            strdot = 0.0;
        } else {
            tdyn = reff / galaxies[p].Vvir;
            const float h = run_params->Hubble_h;
            const float rs_pc = galaxies[p].DiskScaleRadius * 1.0e6 / h;

            if (rs_pc <= 0.0) {
                galaxies[p].H2gas = 0.0;
                strdot = 0.0;
            } else {
                float disk_area_pc2 = M_PI * pow(3.0 * rs_pc, 2); // pc^2
                float gas_surface_density = (galaxies[p].ColdGas * 1.0e10 / h) / disk_area_pc2; // Msun/pc^2
                float stellar_surface_density = (galaxies[p].StellarMass * 1.0e10 / h) / disk_area_pc2; // Msun/pc^2

                // Calculate molecular fraction using Blitz & Rosolowsky 2006
                total_molecular_gas = calculate_molecular_fraction_BR06(gas_surface_density, stellar_surface_density,
                                                                       rs_pc) * galaxies[p].ColdGas;

                galaxies[p].H2gas = total_molecular_gas;

                // Critical surface density from Equation 2
                const double Sigma_crit = 30.0 / (M_PI * 4.302e-3); // ~2176 Msun/pc^2

                // Cloud-scale star formation efficiency from Equation 3
                double epsilon_cl = (gas_surface_density / Sigma_crit) / (1.0 + gas_surface_density / Sigma_crit);

                // Fraction of gas in dense clouds (f_dense from Equation 8)
                const double f_dense = 0.5;

                // Star formation rate using H2 gas instead of total cold gas
                if(tdyn > 0.0 && gas_surface_density > 0.0 && galaxies[p].H2gas > 0.0) {
                    strdot = epsilon_cl * f_dense * galaxies[p].H2gas / tdyn;
                } else {
                    strdot = 0.0;
                }
            }
        }

    } else if(run_params->SFprescription == 4) {

        // ========================================================================
        // Krumholz and Dekel (2012) - KD12 Model
        // ========================================================================

        // we take the typical star forming region as 3.0*r_s using the Milky Way as a guide
        reff = 3.0 * galaxies[p].DiskScaleRadius;

        tdyn = 3.0 * galaxies[p].DiskScaleRadius / galaxies[p].Vvir;
        const float h = run_params->Hubble_h;
        const float rs_pc = galaxies[p].DiskScaleRadius * 1.0e6 / h;
        if (rs_pc <= 0.0) {
            galaxies[p].H2gas = 0.0;
            strdot = 0.0;
        } else {
            float disk_area = M_PI * galaxies[p].DiskScaleRadius * galaxies[p].DiskScaleRadius;; // pc^2
            // float disk_area =  M_PI * pow(3.0 * rs_pc, 2);
            if(disk_area <= 0.0) {
                galaxies[p].H2gas = 0.0;
                return;
            }
            float surface_density = galaxies[p].ColdGas / disk_area;
            // double metallicity = 0.0;
            if(galaxies[p].ColdGas > 0.0) {
                metallicity = galaxies[p].MetalsColdGas / galaxies[p].ColdGas; // absolute fraction
            }
            float clumping_factor = 5.0;
            // if (metallicity < 0.01) {
            //     clumping_factor = 0.5 * pow(0.01, -0.05);
            // } else if (metallicity < 1.0) {
            //     clumping_factor = 0.5 * pow(metallicity, -0.05);
            // }
            
            total_molecular_gas = calculate_H2_fraction_KD12(surface_density, metallicity, clumping_factor) * galaxies[p].ColdGas;

            galaxies[p].H2gas = total_molecular_gas;

            if (galaxies[p].H2gas > 0.0 && tdyn > 0.0) {
                strdot = run_params->SfrEfficiency * galaxies[p].H2gas / tdyn;
            } else {
                strdot = 0.0;
            }
        }
    } else if(run_params->SFprescription == 5) {

        // ========================================================================
        // Krumholz, McKee, & Tumlinson (2009) - KMT09 Model
        // ========================================================================

        // we take the typical star forming region as 3.0*r_s using the Milky Way as a guide
        reff = 3.0 * galaxies[p].DiskScaleRadius;
        
        // 1. Geometry and Units [cite: 60-64]
        reff = 3.0 * galaxies[p].DiskScaleRadius;
        
        // Check for physical validity
        if(galaxies[p].Vvir <= 0.0 || galaxies[p].DiskScaleRadius <= 0.0) {
            galaxies[p].H2gas = 0.0;
            strdot = 0.0;
        } else {
            const float h = run_params->Hubble_h;
            // Scale radius in pc
            const float rs_pc = galaxies[p].DiskScaleRadius * 1.0e6 / h;
            
            // Disk Area (pc^2) - 3*rs captures ~95% of mass
            float disk_area_pc2 = M_PI * pow(3.0 * rs_pc, 2);
            
            // Gas Surface Density (Msun/pc^2) - Sigma_g
            // ColdGas is in 10^10 Msun/h
            float gas_surface_density = (disk_area_pc2 > 0.0) ? 
                (galaxies[p].ColdGas * 1.0e10 / h) / disk_area_pc2 : 0.0;
                
            // 2. Metallicity (Normalized to Solar) [cite: 81]
            // Calculate absolute metallicity Z, then normalize to Solar (approx 0.02)
            float metallicity_abs = 0.0;
            if(galaxies[p].ColdGas > 0.0) {
                metallicity_abs = galaxies[p].MetalsColdGas / galaxies[p].ColdGas;
            }
            // Z' = Z / Z_solar. Clamp to a minimum to avoid division by zero in log terms.
            float Z_prime = metallicity_abs / 0.02; 
            if (Z_prime < 0.01) Z_prime = 0.01;

            // 3. Molecular Fraction f_H2 (Equation 2 from KMT09) [cite: 80-81]
            // Clumping factor c: Paper suggests c ~ 5 for kpc-scale observations/models 
            const float clumping_factor = 5.0; 
            float Sigma_comp = clumping_factor * gas_surface_density; // Surface density of complexes

            // Chi factor: chi = 0.77 * (1 + 3.1 * Z'^0.365)
            float chi = 0.77 * (1.0 + 3.1 * pow(Z_prime, 0.365));

            // s = ln(1 + 0.6*chi) / (0.04 * Sigma_comp * Z')
            // Note: 0.04 constant includes units for Sigma in Msun/pc^2
            float s = 0.0;
            if (Sigma_comp > 0.0) {
                s = log(1.0 + 0.6 * chi) / (0.04 * Sigma_comp * Z_prime);
            } else {
                s = 1.0e5; // Large s implies f_H2 -> 0
            }

            // delta = 0.0712 * (0.1/s + 0.675)^-2.8
            float delta = 0.0712 * pow(0.1/s + 0.675, -2.8);

            // f_H2 formula: 1 - [1 + (0.75 * s / (1+delta))^-5]^-1/5
            float term_inner = 0.75 * s / (1.0 + delta);
            float f_H2 = 1.0 - pow(1.0 + pow(term_inner, -5.0), -0.2);

            // Safety clamps
            if (f_H2 < 0.0) f_H2 = 0.0;
            if (f_H2 > 1.0) f_H2 = 1.0;

            // Store H2 mass
            galaxies[p].H2gas = f_H2 * galaxies[p].ColdGas;

            // 4. Star Formation Timescale (Equation 10 from KMT09) 
            // The paper specifies a depletion time for the molecular gas:
            // t_dep = 2.6 Gyr * (Sigma_g / 85 Msun/pc^2)^-0.33  [for Low Density]
            // t_dep = 2.6 Gyr * (Sigma_g / 85 Msun/pc^2)^+0.33  [for High Density]
            
            double t_sf_gyr = 2.6; // Normalization from Eq 10
            double sigma_crit = 85.0; // Critical density Msun/pc^2
            
            double density_ratio = 1.0;
            if (gas_surface_density > 0.0) {
                density_ratio = gas_surface_density / sigma_crit;
            }

            double timescale_factor = 1.0;
            if (gas_surface_density < sigma_crit) {
                // Low density: internal regulation, t_ff depends on Jeans mass
                timescale_factor = pow(density_ratio, -0.333); 
            } else {
                // High density: ambient pressure regulation
                timescale_factor = pow(density_ratio, 0.333);
            }
            
            double t_depletion_gyr = t_sf_gyr * timescale_factor;

            // Convert Gyr to code time units
            // UnitTime_in_Megayears is typically ~978 Myr/h for SAGE, but we use the variable
            double t_depletion_code = t_depletion_gyr * 1000.0 / run_params->UnitTime_in_Megayears;

            // 5. Calculate SFR
            if (galaxies[p].H2gas > 0.0 && t_depletion_code > 0.0) {
                // SFR = M_H2 / t_depletion
                // Note: We apply SfrEfficiency here to allow standard tuning, 
                // though KMT09 provides an absolute prediction (Efficiency ~ 1.0).
                strdot = run_params->SfrEfficiency * galaxies[p].H2gas / t_depletion_code;
            } else {
                strdot = 0.0;
            }
        }
    } else if(run_params->SFprescription == 6) {

        // we take the typical star forming region as 3.0*r_s using the Milky Way as a guide
        reff = 3.0 * galaxies[p].DiskScaleRadius;

        // ========================================================================
        // Krumholz 2013 (KMT+) Model
        // "The star formation law in molecule-poor galaxies"
        // Uses the analytic approximation for depletion time (Equation 28)
        // ========================================================================

        reff = 3.0 * galaxies[p].DiskScaleRadius;

        // Basic safety checks
        if(galaxies[p].Vvir <= 0.0 || galaxies[p].ColdGas <= 0.0 || galaxies[p].DiskScaleRadius <= 0.0) {
            strdot = 0.0;
            galaxies[p].H2gas = 0.0;
        } else {
            tdyn = reff / galaxies[p].Vvir; // Code units

            const float h = run_params->Hubble_h;
            const float rs_pc = galaxies[p].DiskScaleRadius * 1.0e6 / h;
            
            // Calculate surface densities within 3*scale_radius (captures ~95% of mass)
            const float area_pc2 = M_PI * pow(3.0 * rs_pc, 2);
            
            if(area_pc2 > 0.0) {
                // Surface densities in Msun/pc^2
                double Sigma_gas = (galaxies[p].ColdGas * 1.0e10 / h) / area_pc2;
                double Sigma_star = (galaxies[p].StellarMass * 1.0e10 / h) / area_pc2;

                // Metallicity Z' (normalized to solar). 
                // Using Z_sun ~ 0.014. Floor at 0.01 to avoid numerical singularities.
                double Z_gas = (galaxies[p].ColdGas > 0.0) ? (galaxies[p].MetalsColdGas / galaxies[p].ColdGas) : 0.0;
                double Z_prime = Z_gas / 0.014; 
                if(Z_prime < 0.01) Z_prime = 0.01;

                // Clumping factor fc = 5 is recommended for ~kpc scales (Section 3.1) [cite: 377]
                double fc = 5.0;

                // ----------------------------------------------------------------
                // 1. Calculate Standard KMT Depletion Time (Molecule-Rich Regime)
                // ----------------------------------------------------------------
                
                // Normalized Radiation Field chi_2p (Eq 13) [cite: 116]
                double chi_2p = 3.1 * (1.0 + 3.1 * pow(Z_prime, 0.365)) / 4.1;
                
                // Optical Depth tau_c (Eq 12) [cite: 123]
                double tau_c = 0.066 * fc * Z_prime * Sigma_gas;
                
                // Parameter s (Eq 11) [cite: 122]
                // Protected against tau_c=0
                double s = (tau_c > 0.0) ? log(1.0 + 0.6 * chi_2p + 0.01 * chi_2p * chi_2p) / (0.6 * tau_c) : 100.0;
                
                // H2 Fraction f_H2 (Eq 10) [cite: 120]
                double f_H2_2p = 0.0;
                if(s < 2.0) {
                    f_H2_2p = 1.0 - (0.75 * s) / (1.0 + 0.25 * s);
                }
                if(f_H2_2p < 0.0) f_H2_2p = 0.0;
                
                // t_dep_2p (Eq 27) 
                // t_dep = 3.1 Gyr / (f_H2 * Sigma^0.25)
                double t_dep_2p_Gyr;
                if(f_H2_2p > 1e-6) {
                    t_dep_2p_Gyr = 3.1 / (f_H2_2p * pow(Sigma_gas, 0.25));
                } else {
                    t_dep_2p_Gyr = 1.0e5; // Cap at very large timescale if no H2
                }

                // ----------------------------------------------------------------
                // 2. Calculate Hydrostatic Limits (Molecule-Poor Regimes)
                // ----------------------------------------------------------------

                // Estimate stellar density rho_sd for Eq 21. 
                // We approximate h_z ~ 0.1 * R_d. 
                // rho_sd_2 is rho_sd in units of 0.01 Msun/pc^3 (e.g., rho_sd,-2 in paper)
                double h_z = 0.1 * rs_pc;
                double rho_sd_2 = 0.0;
                if(h_z > 0.0) {
                     double rho_star = Sigma_star / (2.0 * h_z); // Msun/pc^3
                     rho_sd_2 = rho_star / 0.01;
                }
                if(rho_sd_2 < 1e-4) rho_sd_2 = 1e-4; // Avoid div by zero

                // t_dep_hydro_star (Eq 21) 
                // 3.1/Sigma^0.25 + 100 / ( (fc/5) * Z' * sqrt(rho_sd_2) * Sigma )
                double t_hydro_star_Gyr = 3.1 / pow(Sigma_gas, 0.25) + 
                                          100.0 / ((fc/5.0) * Z_prime * sqrt(rho_sd_2) * Sigma_gas);

                // t_dep_hydro_gas (Eq 22) 
                // 3.1/Sigma^0.25 + 360 / ( (fc/5) * Z' * Sigma^2 )
                double t_hydro_gas_Gyr = 3.1 / pow(Sigma_gas, 0.25) + 
                                         360.0 / ((fc/5.0) * Z_prime * pow(Sigma_gas, 2.0));

                // ----------------------------------------------------------------
                // 3. Analytic Approximation for Depletion Time
                // ----------------------------------------------------------------
                
                // Eq 28: t_dep ~ min(t_2p, t_hydro_star, t_hydro_gas) 
                double t_dep_Gyr = t_dep_2p_Gyr;
                if(t_hydro_star_Gyr < t_dep_Gyr) t_dep_Gyr = t_hydro_star_Gyr;
                if(t_hydro_gas_Gyr < t_dep_Gyr) t_dep_Gyr = t_hydro_gas_Gyr;

                // ----------------------------------------------------------------
                // 4. Calculate SFR and Back-calculate H2 Fraction
                // ----------------------------------------------------------------

                // Convert t_dep (Gyr) to Code Units
                double UnitTime_Gyr = run_params->UnitTime_in_Megayears / 1000.0;
                double t_dep_Code = t_dep_Gyr / UnitTime_Gyr;

                if(t_dep_Code > 0.0) {
                    strdot = galaxies[p].ColdGas / t_dep_Code;
                } else {
                    strdot = 0.0;
                }

                // Calculate the effective H2 fraction consistent with this SFR
                // Inverting Eq 27: f_H2_eff = 3.1 / (t_dep_Gyr * Sigma^0.25)
                double f_H2_eff = 3.1 / (t_dep_Gyr * pow(Sigma_gas, 0.25));
                
                // Clamp fraction between 0 and 1
                if(f_H2_eff > 1.0) f_H2_eff = 1.0;
                if(f_H2_eff < 0.0) f_H2_eff = 0.0;
                
                galaxies[p].H2gas = f_H2_eff * galaxies[p].ColdGas;

            } else {
                strdot = 0.0;
                galaxies[p].H2gas = 0.0;
            }
        }
    } else if(run_params->SFprescription == 7) {
        
        // ========================================================================
        // Gnedin & Draine (2014) - GD14 Model
        // Implemented using the "more accurate and simpler fit" from the 
        // 2016 Erratum (ApJ, 830, 54)
        // ========================================================================

        // we take the typical star forming region as 3.0*r_s using the Milky Way as a guide
        reff = 3.0 * galaxies[p].DiskScaleRadius;

        reff = 3.0 * galaxies[p].DiskScaleRadius;

        // Basic safety checks
        if(galaxies[p].Vvir <= 0.0 || galaxies[p].ColdGas <= 0.0 || galaxies[p].DiskScaleRadius <= 0.0) {
            strdot = 0.0;
            galaxies[p].H2gas = 0.0;
        } else {
            tdyn = reff / galaxies[p].Vvir; // Code units
            
            const float h = run_params->Hubble_h;
            // Scale radius in pc
            const float rs_pc = galaxies[p].DiskScaleRadius * 1.0e6 / h;
            
            // 1. Calculate Geometry and Gas Surface Density
            // Averaging over 3*scale_radius (captures ~95% of mass)
            const float disk_area_pc2 = M_PI * pow(3.0 * rs_pc, 2);
            
            double Sigma_gas = 0.0;
            if(disk_area_pc2 > 0.0) {
                // Surface density in Msun/pc^2 (ColdGas is 10^10 Msun/h)
                Sigma_gas = (galaxies[p].ColdGas * 1.0e10 / h) / disk_area_pc2;
            }

            // 2. Dust-to-Gas Ratio (D_MW)
            // Normalized to Milky Way. We assume dust tracks metallicity.
            // Z_solar approx 0.02 (consistent with KMT09 usage in this file)
            double metallicity_abs = 0.0;
            if(galaxies[p].ColdGas > 0.0) {
                metallicity_abs = galaxies[p].MetalsColdGas / galaxies[p].ColdGas;
            }
            double D_MW = metallicity_abs / 0.02; 
            // Floor small value to avoid sqrt(0) issues
            if(D_MW < 1e-4) D_MW = 1e-4;

            // 3. UV Radiation Field (U_MW)
            // Normalized to Milky Way. 
            // Ideally this should scale with SFR surface density (e.g., prev step).
            // Lacking an explicit input, we default to MW-like (U=1) or allow 
            // for user-defined scaling.
            double U_MW = 1.0; 
            
            // 4. Characteristic Scale Parameter (S) 
            // S = L / 100 pc. 
            // For a galactic disk, the characteristic size L is roughly the diameter (3*Rs, following SAGE C16)
            double L_pc = 3.0 * rs_pc;
            double S = L_pc / 100.0;

            // 5. Calculate Fitting Parameters (Erratum 2016)
            
            // s parameter [cite: 404]
            // s = (0.001 + 0.1 * U_MW)^0.7
            double s_param = pow(0.001 + 0.1 * U_MW, 0.7);

            // D_star parameter [cite: 218]
            // Accounts for line overlap saturation on large scales
            double D_star = 0.17 * (2.0 + pow(S, 5.0)) / (1.0 + pow(S, 5.0));

            // g factor [cite: 215]
            double g = sqrt(D_MW * D_MW + D_star * D_star);

            // Sigma_R=1 (Surface density where f_H2 ~ 0.5) [cite: 402]
            double Sigma_R1 = (40.0 / g) * (s_param / (1.0 + s_param));

            // Power law slope alpha [cite: 399]
            double alpha = 1.0 + 0.7 * sqrt(s_param) / (1.0 + s_param);

            // 6. Calculate Molecular Ratio R
            // q parameter [cite: 396]
            double q = 0.0;
            if(Sigma_R1 > 0.0 && Sigma_gas > 0.0) {
                q = pow(Sigma_gas / Sigma_R1, alpha);
            }

            // R = Sigma_H2 / Sigma_HI [cite: 395]
            // eta approx 0 on kpc scales [cite: 398]
            double eta = 0.0; 
            double R = q * (1.0 + eta * q) / (1.0 + eta);

            // 7. H2 Fraction and Mass
            // f_H2 = R / (1 + R)
            double f_H2 = R / (1.0 + R);
            
            // Clamp to physical range
            if(f_H2 > 1.0) f_H2 = 1.0;
            if(f_H2 < 0.0) f_H2 = 0.0;

            galaxies[p].H2gas = f_H2 * galaxies[p].ColdGas;

            // 8. Calculate Star Formation Rate
            // Standard relation: SFR = Efficiency * H2 / t_dyn
            if(galaxies[p].H2gas > 0.0 && tdyn > 0.0) {
                strdot = run_params->SfrEfficiency * galaxies[p].H2gas / tdyn;
            } else {
                strdot = 0.0;
            }
        }
    } else {
        fprintf(stderr, "No star formation prescription selected!\n");
        ABORT(0);
    }

    stars = strdot * dt;
    if(stars < 0.0) {
        stars = 0.0;
    }

    // Calculate reheated mass - use FIRE model if enabled, otherwise use original feedback
    double reheated_mass = 0.0;
    
    if(run_params->SupernovaRecipeOn == 1) {
        if(run_params->FIREmodeOn == 1) {
            // FIRE: Calculate velocity/redshift scaling from Muratov et al. 2015
            const double z = run_params->ZZ[galaxies[p].SnapNum];
            const double vc = galaxies[p].Vvir;
            const double V_CRIT = 60.0;
            
            // Check for valid inputs to avoid NaN
            if(vc <= 0.0 || z < 0.0) {
                reheated_mass = 0.0;
            } else {
                double z_term = pow(1.0 + z, run_params->RedshiftPowerLawExponent);
                double v_term;
                if (vc < V_CRIT) {
                    v_term = pow(vc / V_CRIT, -3.2);
                } else {
                    v_term = pow(vc / V_CRIT, -1.0);
                }
                double scaling_factor = z_term * v_term;
                
                // Reheating with Muratov scaling: η = 2.9 × (1+z)^α × (V/60)^β
                double eta_reheat = run_params->FeedbackReheatingEpsilon * scaling_factor;
                // Store mass loading for analysis (cast to float)
                galaxies[p].MassLoading = (float)eta_reheat;
                reheated_mass = eta_reheat * stars;
            }
            
        } else {
            reheated_mass = run_params->FeedbackReheatingEpsilon * stars;
        }
    }

	XASSERT(reheated_mass >= 0.0, -1,
            "Error: Expected reheated gas-mass = %g to be >=0.0\n", reheated_mass);

    // cant use more cold gas than is available! so balance SF and feedback
    if((stars + reheated_mass) > galaxies[p].ColdGas && (stars + reheated_mass) > 0.0) {
        const double fac = galaxies[p].ColdGas / (stars + reheated_mass);
        stars *= fac;
        reheated_mass *= fac;
    }

    // determine ejection
    if(run_params->SupernovaRecipeOn == 1) {
        // BUG FIX: Check galaxies[p].Vvir consistently (was checking centralgal but using p)
        if(galaxies[p].Vvir > 0.0) {
            if(run_params->FIREmodeOn == 1) {
                // FIRE model: Energy-based ejection following Hirschmann+2016
                // Energy from supernovae (with Muratov scaling)
                const double z = run_params->ZZ[galaxies[p].SnapNum];
                const double vc = galaxies[p].Vvir;
                const double V_CRIT = 60.0;
                
                // Check for valid inputs to avoid NaN
                if(vc <= 0.0 || z < 0.0) {
                    ejected_mass = 0.0;
                } else {
                    double z_term = pow(1.0 + z, run_params->RedshiftPowerLawExponent);
                    double v_term;
                    if (vc < V_CRIT) {
                        v_term = pow(vc / V_CRIT, -3.2);
                    } else {
                        v_term = pow(vc / V_CRIT, -1.0);
                    }
                    double scaling_factor = z_term * v_term;
                    
                    // Total feedback energy: E_FB = ε_eject × scaling × 0.5 × M_* × (η_SN × E_SN)
                    double E_FB = run_params->FeedbackEjectionEfficiency * scaling_factor * 
                                  0.5 * stars * (run_params->EtaSNcode * run_params->EnergySNcode);
                    
                    // Energy needed to lift reheated gas to virial radius: E_lift = 0.5 × M_reheat × V_vir²
                    double E_lift = 0.5 * reheated_mass * vc * vc;
                    
                    // Leftover energy ejects additional gas: E_eject = E_FB - E_lift
                    // Ejected mass: M_eject = E_eject / (0.5 × V_vir²)
                    if(E_FB > E_lift) {
                        ejected_mass = (E_FB - E_lift) / (0.5 * vc * vc);
                    } else {
                        ejected_mass = 0.0;
                    }
                }
            } else {
                // Original non-FIRE calculation
                ejected_mass = (run_params->FeedbackEjectionEfficiency * 
                               (run_params->EtaSNcode * run_params->EnergySNcode) / 
                               (galaxies[p].Vvir * galaxies[p].Vvir) -
                               run_params->FeedbackReheatingEpsilon) * stars;
            }
        } else {
            ejected_mass = 0.0;
        }
        
        if(ejected_mass < 0.0) {
            ejected_mass = 0.0;
        }
    } else {
        ejected_mass = 0.0;
    }


    // update the star formation rate
    galaxies[p].SfrDisk[step] += stars / dt;
    galaxies[p].SfrDiskColdGas[step] = galaxies[p].ColdGas;
    galaxies[p].SfrDiskColdGasMetals[step] = galaxies[p].MetalsColdGas;

    // update for star formation
    metallicity = get_metallicity(galaxies[p].ColdGas, galaxies[p].MetalsColdGas);
    update_from_star_formation(p, stars, metallicity, galaxies, run_params);

    // recompute the metallicity of the cold phase
    metallicity = get_metallicity(galaxies[p].ColdGas, galaxies[p].MetalsColdGas);

    // Safety check: ensure reheated_mass doesn't exceed remaining ColdGas (floating-point precision)
    if(reheated_mass > galaxies[p].ColdGas) {
        reheated_mass = galaxies[p].ColdGas;
    }

    // update from SN feedback
    update_from_feedback(p, centralgal, reheated_mass, ejected_mass, metallicity, galaxies, run_params);

    // check for disk instability
    if(run_params->DiskInstabilityOn) {
        check_disk_instability(p, centralgal, halonr, time, dt, step, galaxies, (struct params *) run_params);
    }

    // formation of new metals - instantaneous recycling approximation - only SNII
    if(galaxies[p].ColdGas > 1.0e-8) {
        const double FracZleaveDiskVal = run_params->FracZleaveDisk * exp(-1.0 * galaxies[centralgal].Mvir / 30.0);  // Krumholz & Dekel 2011 Eq. 22
        
        // Metals that stay in disk (same for all regimes)
        galaxies[p].MetalsColdGas += run_params->Yield * (1.0 - FracZleaveDiskVal) * stars;
        
        // Metals that leave disk - regime dependent
        const double metals_leaving_disk = run_params->Yield * FracZleaveDiskVal * stars;
        
        if(run_params->CGMrecipeOn == 1) {
            if(galaxies[centralgal].Regime == 0) {
                // CGM-regime: metals go to CGM
                galaxies[centralgal].MetalsCGMgas += metals_leaving_disk;
            } else {
                // Hot-ICM-regime: metals go to HotGas
                galaxies[centralgal].MetalsHotGas += metals_leaving_disk;
            }
        } else {
            // Original SAGE behavior: metals go to HotGas
            galaxies[centralgal].MetalsHotGas += metals_leaving_disk;
        }
        
    } else {
        // All metals leave disk when ColdGas is very low - regime dependent
        const double all_metals = run_params->Yield * stars;
        
        if(run_params->CGMrecipeOn == 1) {
            if(galaxies[centralgal].Regime == 0) {
                // CGM-regime: metals go to CGM
                galaxies[centralgal].MetalsCGMgas += all_metals;
            } else {
                // Hot-ICM-regime: metals go to HotGas
                galaxies[centralgal].MetalsHotGas += all_metals;
            }
        } else {
            // Original SAGE behavior: metals go to HotGas
            galaxies[centralgal].MetalsHotGas += all_metals;
        }
    }
}



void update_from_star_formation(const int p, const double stars, const double metallicity, struct GALAXY *galaxies, const struct params *run_params)
{
    const double RecycleFraction = run_params->RecycleFraction;
    // update gas and metals from star formation
    galaxies[p].ColdGas -= (1 - RecycleFraction) * stars;
    galaxies[p].MetalsColdGas -= metallicity * (1 - RecycleFraction) * stars;
    galaxies[p].StellarMass += (1 - RecycleFraction) * stars;
    galaxies[p].MetalsStellarMass += metallicity * (1 - RecycleFraction) * stars;
}



void update_from_feedback(const int p, const int centralgal, const double reheated_mass, double ejected_mass, const double metallicity,
                          struct GALAXY *galaxies, const struct params *run_params)
{

    XASSERT(reheated_mass >= 0.0, -1,
            "Error: For galaxy = %d (halonr = %d, centralgal = %d) with MostBoundID = %lld, the reheated mass = %g should be >=0.0",
            p, galaxies[p].HaloNr, centralgal, galaxies[p].MostBoundID, reheated_mass);
    XASSERT(reheated_mass <= galaxies[p].ColdGas, -1,
            "Error: Reheated mass = %g should be <= the coldgas mass of the galaxy = %g",
            reheated_mass, galaxies[p].ColdGas);

    if(run_params->SupernovaRecipeOn == 1) {
        // Remove reheated mass from cold gas (same for all regimes)
        galaxies[p].ColdGas -= reheated_mass;
        galaxies[p].MetalsColdGas -= metallicity * reheated_mass;

        if(run_params->CGMrecipeOn == 1) {
            if(galaxies[centralgal].Regime == 0) {
                // CGM-regime: Cold --> CGM --> Ejected
                
                // Add reheated gas to CGM
                galaxies[centralgal].CGMgas += reheated_mass;
                galaxies[centralgal].MetalsCGMgas += metallicity * reheated_mass;

                // Check if ejection is possible from CGM
                if(ejected_mass > galaxies[centralgal].CGMgas) {
                    ejected_mass = galaxies[centralgal].CGMgas;
                }
                const double metallicityCGM = get_metallicity(galaxies[centralgal].CGMgas, galaxies[centralgal].MetalsCGMgas);

                // Eject from CGM to EjectedMass
                galaxies[centralgal].CGMgas -= ejected_mass;
                galaxies[centralgal].MetalsCGMgas -= metallicityCGM * ejected_mass;
                galaxies[centralgal].EjectedMass += ejected_mass;
                galaxies[centralgal].MetalsEjectedMass += metallicityCGM * ejected_mass;

            } else {
                // Hot-ICM-regime: Cold --> HotGas --> Ejected
                
                // Add reheated gas to HotGas
                galaxies[centralgal].HotGas += reheated_mass;
                galaxies[centralgal].MetalsHotGas += metallicity * reheated_mass;

                // Check if ejection is possible from HotGas
                if(ejected_mass > galaxies[centralgal].HotGas) {
                    ejected_mass = galaxies[centralgal].HotGas;
                }
                const double metallicityHot = get_metallicity(galaxies[centralgal].HotGas, galaxies[centralgal].MetalsHotGas);

                // Eject from HotGas to EjectedMass
                galaxies[centralgal].HotGas -= ejected_mass;
                galaxies[centralgal].MetalsHotGas -= metallicityHot * ejected_mass;
                galaxies[centralgal].EjectedMass += ejected_mass;
                galaxies[centralgal].MetalsEjectedMass += metallicityHot * ejected_mass;
            }
        } else {
            // Original SAGE behavior: Cold --> HotGas --> Ejected
            
            // Add reheated gas to HotGas
            galaxies[centralgal].HotGas += reheated_mass;
            galaxies[centralgal].MetalsHotGas += metallicity * reheated_mass;

            // Check if ejection is possible from HotGas
            if(ejected_mass > galaxies[centralgal].HotGas) {
                ejected_mass = galaxies[centralgal].HotGas;
            }
            const double metallicityHot = get_metallicity(galaxies[centralgal].HotGas, galaxies[centralgal].MetalsHotGas);

            // Eject from HotGas to EjectedMass
            galaxies[centralgal].HotGas -= ejected_mass;
            galaxies[centralgal].MetalsHotGas -= metallicityHot * ejected_mass;
            galaxies[centralgal].EjectedMass += ejected_mass;
            galaxies[centralgal].MetalsEjectedMass += metallicityHot * ejected_mass;
        }

        galaxies[p].OutflowRate += reheated_mass;
    }
}

void starformation_ffb(const int p, const int centralgal, const double dt, const int step,
                       struct GALAXY *galaxies, const struct params *run_params)
{
    // ========================================================================
    // FEEDBACK-FREE BURST (FFB) STAR FORMATION
    // Implementation of Li et al. 2024 - Equation (4) (modified to be Kauffmann-like)
    // ========================================================================
    
    double reff, tdyn, strdot, stars, metallicity;
    
    // Calculate dynamical time
    reff = 3.0 * galaxies[p].DiskScaleRadius;
    tdyn = (reff > 0.0 && galaxies[p].Vvir > 0.0) ? reff / galaxies[p].Vvir : 0.0;
    
    // Safety checks for NaN in inputs
    if(isnan(galaxies[p].ColdGas) || isinf(galaxies[p].ColdGas) ||
       isnan(galaxies[p].Vvir) || isinf(galaxies[p].Vvir) ||
       isnan(reff) || isinf(reff) || isnan(tdyn) || isinf(tdyn)) {
        stars = 0.0;
    } else if(tdyn > 0.0 && galaxies[p].ColdGas > 0.0) {
        // Equation (4): SFR = ε_FFB × M_gas / t_dyn
        // Use maximum FFB efficiency (typically 0.2, can be up to 1.0)
        const double epsilon_ffb = run_params->FFBMaxEfficiency;
        const double cold_crit = 0.19 * galaxies[p].Vvir * reff;
        
        // Safety check on cold_crit
        if(isnan(cold_crit) || isinf(cold_crit) || cold_crit < 0.0) {
            stars = 0.0;
        } else if(galaxies[p].ColdGas > 0.0) {
            // Only form stars if above critical density
            strdot = epsilon_ffb * (galaxies[p].ColdGas) / tdyn;
            
            // Safety check on strdot
            if(isnan(strdot) || isinf(strdot) || strdot < 0.0) {
                stars = 0.0;
            } else {
                stars = strdot * dt;
                
                // Can't form more stars than gas available
                if(stars > galaxies[p].ColdGas) {
                    stars = galaxies[p].ColdGas;
                }
                
                // Final safety check
                if(isnan(stars) || isinf(stars) || stars < 0.0) {
                    stars = 0.0;
                }
            }
        } else {
            // Below critical density - no star formation
            stars = 0.0;
        }
        
        // Debug output (only on first step to avoid spam)
        // if(step == 0) {
        //     const double z = run_params->ZZ[galaxies[p].SnapNum];
        //     printf("FFB SF: z=%.2f, Mvir=%.2e, eps=%.1f%%, M_gas=%.2e, t_dyn=%.3f Gyr, SFR=%.2e Msun/yr\n",
        //            z, galaxies[p].Mvir, epsilon_ffb*100, galaxies[p].ColdGas, 
        //            tdyn * run_params->UnitTime_in_Megayears / 1000.0, strdot);
        // }
    } else {
        stars = 0.0;
    }
    
    // Update star formation rate tracking
    galaxies[p].SfrDisk[step] += stars / dt;
    galaxies[p].SfrDiskColdGas[step] = galaxies[p].ColdGas;
    galaxies[p].SfrDiskColdGasMetals[step] = galaxies[p].MetalsColdGas;
    
    // Update for star formation (convert gas to stars)
    metallicity = get_metallicity(galaxies[p].ColdGas, galaxies[p].MetalsColdGas);
    update_from_star_formation(p, stars, metallicity, galaxies, run_params);
    
    // ========================================================================
    // Stars first form, then feedback acts on them
    // Key physics: star formation completes on free-fall time (~1 Myr)
    // before feedback from these stars can act (~2 Myr)
    // ========================================================================
    
    // Calculate reheated mass - use FIRE model if enabled, otherwise use original feedback
    double reheated_mass = 0.0;
    double ejected_mass = 0.0;
    
    if(run_params->SupernovaRecipeOn == 1) {
        if(run_params->FIREmodeOn == 1) {
            // FIRE: Calculate velocity/redshift scaling from Muratov et al. 2015
            const double z = run_params->ZZ[galaxies[p].SnapNum];
            const double vc = galaxies[p].Vvir;
            const double V_CRIT = 60.0;
            
            // Check for valid inputs to avoid NaN
            if(vc <= 0.0 || z < 0.0) {
                reheated_mass = 0.0;
            } else {
                double z_term = pow(1.0 + z, run_params->RedshiftPowerLawExponent);
                double v_term;
                if (vc < V_CRIT) {
                    v_term = pow(vc / V_CRIT, -3.2);
                } else {
                    v_term = pow(vc / V_CRIT, -1.0);
                }
                double scaling_factor = z_term * v_term;
                
                // Reheating with Muratov scaling: η = 2.9 × (1+z)^α × (V/60)^β
                double eta_reheat = run_params->FeedbackReheatingEpsilon * scaling_factor;
                // Store mass loading for analysis (cast to float)
                galaxies[p].MassLoading = (float)eta_reheat;
                reheated_mass = eta_reheat * stars;
            }
        } else {
            reheated_mass = run_params->FeedbackReheatingEpsilon * stars;
        }
    }

	XASSERT(reheated_mass >= 0.0, -1,
            "Error: Expected reheated gas-mass = %g to be >=0.0\n", reheated_mass);

    // cant use more cold gas than is available! so balance SF and feedback
    if((stars + reheated_mass) > galaxies[p].ColdGas && (stars + reheated_mass) > 0.0) {
        const double fac = galaxies[p].ColdGas / (stars + reheated_mass);
        stars *= fac;
        reheated_mass *= fac;
    }

    // determine ejection
    if(run_params->SupernovaRecipeOn == 1) {
        // BUG FIX: Check galaxies[p].Vvir consistently (was checking centralgal but using p)
        if(galaxies[p].Vvir > 0.0) {
            if(run_params->FIREmodeOn == 1) {
                // FIRE model: Energy-based ejection following Hirschmann+2016
                // Energy from supernovae (with Muratov scaling)
                const double z = run_params->ZZ[galaxies[p].SnapNum];
                const double vc = galaxies[p].Vvir;
                const double V_CRIT = 60.0;
                
                // Check for valid inputs to avoid NaN
                if(vc <= 0.0 || z < 0.0) {
                    ejected_mass = 0.0;
                } else {
                    double z_term = pow(1.0 + z, run_params->RedshiftPowerLawExponent);
                    double v_term;
                    if (vc < V_CRIT) {
                        v_term = pow(vc / V_CRIT, -3.2);
                    } else {
                        v_term = pow(vc / V_CRIT, -1.0);
                    }
                    double scaling_factor = z_term * v_term;
                    
                    // Total feedback energy: E_FB = ε_eject × scaling × 0.5 × M_* × (η_SN × E_SN)
                    double E_FB = run_params->FeedbackEjectionEfficiency * scaling_factor * 
                                  0.5 * stars * (run_params->EtaSNcode * run_params->EnergySNcode);
                    
                    // Energy needed to lift reheated gas to virial radius: E_lift = 0.5 × M_reheat × V_vir²
                    double E_lift = 0.5 * reheated_mass * vc * vc;
                    
                    // Leftover energy ejects additional gas: E_eject = E_FB - E_lift
                    // Ejected mass: M_eject = E_eject / (0.5 × V_vir²)
                    if(E_FB > E_lift) {
                        ejected_mass = (E_FB - E_lift) / (0.5 * vc * vc);
                    } else {
                        ejected_mass = 0.0;
                    }
                }
            } else {
                // Original non-FIRE calculation
                ejected_mass = (run_params->FeedbackEjectionEfficiency * 
                               (run_params->EtaSNcode * run_params->EnergySNcode) / 
                               (galaxies[p].Vvir * galaxies[p].Vvir) -
                               run_params->FeedbackReheatingEpsilon) * stars;
            }
        } else {
            ejected_mass = 0.0;
        }
        
        if(ejected_mass < 0.0) {
            ejected_mass = 0.0;
        }
    } else {
        ejected_mass = 0.0;
    }

    // Safety check: ensure reheated_mass doesn't exceed remaining ColdGas (floating-point precision)
    if(reheated_mass > galaxies[p].ColdGas) {
        reheated_mass = galaxies[p].ColdGas;
    }

     // update from SN feedback
    update_from_feedback(p, centralgal, reheated_mass, ejected_mass, metallicity, galaxies, run_params);


    // H2 for merger-compatibility, but isn't used for stars
    if(run_params->SFprescription == 1 && galaxies[p].ColdGas > 0.0) {
        const float h = run_params->Hubble_h;
        const float rs_pc = galaxies[p].DiskScaleRadius * 1.0e6 / h;
        
        if(rs_pc > 0.0) {
            float disk_area_pc2 = M_PI * pow(3.0 * rs_pc, 2);
            float gas_surface_density = (galaxies[p].ColdGas * 1.0e10 / h) / disk_area_pc2;
            float stellar_surface_density = (galaxies[p].StellarMass * 1.0e10 / h) / disk_area_pc2;
            
            float f_mol = calculate_molecular_fraction_BR06(gas_surface_density, 
                                                            stellar_surface_density, rs_pc);
            galaxies[p].H2gas = f_mol * galaxies[p].ColdGas;
        } else {
            galaxies[p].H2gas = 0.0;
        }
    }

    if(run_params->SFprescription == 4 && galaxies[p].ColdGas > 0.0) {
        const float h = run_params->Hubble_h;
        const float rs_pc = galaxies[p].DiskScaleRadius * 1.0e6 / h;
        
        if(rs_pc > 0.0) {
            const float disk_area = M_PI * galaxies[p].DiskScaleRadius * galaxies[p].DiskScaleRadius;; // pc^2
            if(disk_area > 0.0) {
                const float surface_density = galaxies[p].ColdGas / disk_area;
                // double metallicity = 0.0;
                if(galaxies[p].ColdGas > 0.0) {
                    metallicity = galaxies[p].MetalsColdGas / galaxies[p].ColdGas; // absolute fraction
                }
                float clumping_factor = 5.0;
                
                float f_mol = calculate_H2_fraction_KD12(surface_density, metallicity, clumping_factor);
                galaxies[p].H2gas = f_mol * galaxies[p].ColdGas;
            } else {
                galaxies[p].H2gas = 0.0;
            }
        } else {
            galaxies[p].H2gas = 0.0;
        }
    }
    
    // ========================================================================
    // METAL PRODUCTION (instantaneous recycling approximation - SNII only)
    // ========================================================================
    
    if(galaxies[p].ColdGas > 1.0e-8) {
        // Metals that stay in disk
        const double FracZleaveDiskVal = run_params->FracZleaveDisk * exp(-1.0 * galaxies[centralgal].Mvir / 30.0);
        galaxies[p].MetalsColdGas += run_params->Yield * (1.0 - FracZleaveDiskVal) * stars;
        
        // Metals that leave disk - goes to appropriate reservoir based on CGM regime
        const double metals_leaving_disk = run_params->Yield * FracZleaveDiskVal * stars;
        
        if(run_params->CGMrecipeOn == 1) {
            if(galaxies[centralgal].Regime == 0) {
                // CGM-regime: metals go to CGM
                galaxies[centralgal].MetalsCGMgas += metals_leaving_disk;
            } else {
                // Hot-ICM-regime: metals go to HotGas
                galaxies[centralgal].MetalsHotGas += metals_leaving_disk;
            }
        } else {
            // Original SAGE behavior: metals go to HotGas
            galaxies[centralgal].MetalsHotGas += metals_leaving_disk;
        }
        
    } else {
        // All metals leave disk when ColdGas is very low
        const double all_metals = run_params->Yield * stars;
        
        if(run_params->CGMrecipeOn == 1) {
            if(galaxies[centralgal].Regime == 0) {
                galaxies[centralgal].MetalsCGMgas += all_metals;
            } else {
                galaxies[centralgal].MetalsHotGas += all_metals;
            }
        } else {
            galaxies[centralgal].MetalsHotGas += all_metals;
        }
    }
    
    // ========================================================================
    // NO DISK INSTABILITY CHECK
    // Rapid star formation stabilizes the disk
    // ========================================================================
}
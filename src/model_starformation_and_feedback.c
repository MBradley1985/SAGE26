/*
 * model_starformation_and_feedback.c -- Star formation and supernova feedback.
 *
 * Implements multiple star formation prescriptions (SFprescription 0-7),
 * FIRE stellar feedback, and the feedback-free burst (FFB) mode. Updates cold
 * gas, stellar mass, metals, and reheated/ejected gas reservoirs each substep.
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "core_allvars.h"

#include "model_starformation_and_feedback.h"
#include "model_misc.h"
#include "model_disk_instability.h"


/*
 * Main star formation and feedback driver for one galaxy per substep.
 *
 * Selects the active SF prescription (run_params->SFprescription) and
 * computes star formation rate, reheated mass, and ejected mass. Applies
 * disk instability if triggered and records SFR history. Calls
 * update_from_star_formation() and update_from_feedback() to commit changes.
 */
void starformation_and_feedback(const int p, const int centralgal, const double time, const double dt, const int halonr, const int step,
                                struct GALAXY *galaxies, const struct params *run_params)
{
    // BUG FIX: Validate step is within array bounds
    XASSERT(step >= 0 && step < STEPS, -1,
            "Error: step = %d is out of bounds [0, %d)\n", step, STEPS);

    // ========================================================================
    // CHECK FOR FFB REGIME - EARLY EXIT IF FFB
    // ========================================================================
    if(run_params->FeedbackFreeModeOn >= 1 && galaxies[p].FFBRegime == 1) {
        // This is a Feedback-Free Burst halo
        // Use specialized FFB star formation (no feedback)
        starformation_ffb(p, centralgal, dt, step, galaxies, run_params);
        return;  // Exit early - FFB path complete
    }

    double reff, tdyn, strdot, stars, ejected_mass, metallicity, total_molecular_gas;

    // Initialise variables
    strdot = 0.0;
    tdyn = 0.0;
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
                // H2 mass via BR06: radial integration or single-slab
                if(run_params->H2RadialIntegrationOn) {
                    calculate_molecular_fraction_radial_integration(p, galaxies, run_params, NULL);
                    // result already stored in galaxies[p].H2gas by the function
                } else {
                    float disk_area_pc2;
                    if (run_params->H2DiskAreaOption == 0) {
                        disk_area_pc2 = M_PI * pow(rs_pc, 2);
                    } else if (run_params->H2DiskAreaOption == 1) {
                        disk_area_pc2 = M_PI * pow(3.0 * rs_pc, 2);
                    } else {
                        disk_area_pc2 = 2.0 * M_PI * pow(rs_pc, 2);
                    }
                    const float gas_surface_density  = (galaxies[p].ColdGas * 1.0e10 / h) / disk_area_pc2;
                    const float star_surface_density = (galaxies[p].StellarMass - galaxies[p].BulgeMass)
                                                       * 1.0e10 / h / disk_area_pc2;
                    galaxies[p].H2gas = calculate_molecular_fraction_BR06(gas_surface_density, star_surface_density,
                                                                           rs_pc) * (galaxies[p].ColdGas * HYDROGEN_MASS_FRAC);
                }

                if(galaxies[p].H2gas > galaxies[p].ColdGas * HYDROGEN_MASS_FRAC)
                    galaxies[p].H2gas = galaxies[p].ColdGas * HYDROGEN_MASS_FRAC;

                if(galaxies[p].H2gas > 0.0 && tdyn > 0.0) {
                    strdot = run_params->SfrEfficiency * galaxies[p].H2gas / tdyn;
                } else {
                    strdot = 0.0;
                }
            }
        }
    } else if(run_params->SFprescription == 2) {

        // =======================================================================
        // Somerville et al. 2025: Density Modulated Star Formation Efficiency
        // Using Equation 3 for efficiency: epsilon = (Sigma/Sigma_crit)/(1 + Sigma/Sigma_crit)
        // =======================================================================

        // No H2 tracking in this prescription
        galaxies[p].H2gas = 0.0;

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

        // =======================================================================
        // Somerville et al. 2025: Density Modulated Star Formation Efficiency with H2
        // Using Equation 3 for efficiency: epsilon = (Sigma/Sigma_crit)/(1 + Sigma/Sigma_crit)
        // But replacing cold gas with H2 gas using Blitz & Rosolowsky 2006
        // =======================================================================

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
                // H2 mass via BR06: radial integration or single-slab
                float gas_surface_density = 0.0f;
                if(run_params->H2RadialIntegrationOn) {
                    calculate_molecular_fraction_radial_integration(p, galaxies, run_params, NULL);
                    // result already stored in galaxies[p].H2gas by the function
                    // compute gas_surface_density for epsilon_cl below using pi*(3*r_s)^2 as reference
                    const float ref_area = (float)(M_PI * pow(3.0 * rs_pc, 2));
                    gas_surface_density = (ref_area > 0.0f) ? (galaxies[p].ColdGas * 1.0e10f / h) / ref_area : 0.0f;
                } else {
                    float disk_area_pc2;
                    if (run_params->H2DiskAreaOption == 0) {
                        disk_area_pc2 = M_PI * pow(rs_pc, 2);
                    } else if (run_params->H2DiskAreaOption == 1) {
                        disk_area_pc2 = M_PI * pow(3.0 * rs_pc, 2);
                    } else {
                        disk_area_pc2 = 2.0 * M_PI * pow(rs_pc, 2);
                    }
                    gas_surface_density = (galaxies[p].ColdGas * 1.0e10 / h) / disk_area_pc2;
                    const float stellar_surface_density = ((galaxies[p].StellarMass - galaxies[p].BulgeMass) * 1.0e10 / h) / disk_area_pc2;
                    total_molecular_gas = calculate_molecular_fraction_BR06(gas_surface_density, stellar_surface_density,
                                                                            rs_pc) * (galaxies[p].ColdGas * HYDROGEN_MASS_FRAC);
                    galaxies[p].H2gas = total_molecular_gas;
                }

                if(galaxies[p].H2gas > galaxies[p].ColdGas * HYDROGEN_MASS_FRAC)
                    galaxies[p].H2gas = galaxies[p].ColdGas * HYDROGEN_MASS_FRAC;

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

        // BUG FIX: Check Vvir > 0 before division to avoid NaN/Inf
        if(galaxies[p].Vvir <= 0.0) {
            galaxies[p].H2gas = 0.0;
            strdot = 0.0;
        } else {
            reff = 3.0 * galaxies[p].DiskScaleRadius;
            tdyn = reff / galaxies[p].Vvir;
            const float h = run_params->Hubble_h;
            const float rs_pc = galaxies[p].DiskScaleRadius * 1.0e6 / h;
            if (rs_pc <= 0.0) {
                galaxies[p].H2gas = 0.0;
                strdot = 0.0;
            } else {
                if(run_params->H2RadialIntegrationOn) {
                    calculate_molecular_fraction_radial_integration(p, galaxies, run_params, NULL);
                } else {
                    // Choose disk area based on H2DiskAreaOption
                    float disk_area;
                    if (run_params->H2DiskAreaOption == 0) {
                        disk_area = M_PI * pow(rs_pc, 2);
                    } else if (run_params->H2DiskAreaOption == 1) {
                        disk_area = M_PI * pow(3.0 * rs_pc, 2);
                    } else {
                        disk_area = 2.0 * M_PI * pow(rs_pc, 2);
                    }
                    if(disk_area <= 0.0) {
                        galaxies[p].H2gas = 0.0;
                    } else {
                        float surface_density = (galaxies[p].ColdGas * 1.0e10 / h) / disk_area;
                        if(galaxies[p].ColdGas > 0.0) {
                            metallicity = galaxies[p].MetalsColdGas / galaxies[p].ColdGas;
                        }
                        float clumping_factor = 5.0;
                        total_molecular_gas = calculate_H2_fraction_KD12(surface_density, metallicity, clumping_factor) * (galaxies[p].ColdGas * HYDROGEN_MASS_FRAC);
                        galaxies[p].H2gas = total_molecular_gas;
                    }
                }
                // Safety check: H2 fraction cannot exceed 1.0
                if(galaxies[p].H2gas > galaxies[p].ColdGas * HYDROGEN_MASS_FRAC) {
                    galaxies[p].H2gas = galaxies[p].ColdGas * HYDROGEN_MASS_FRAC;
                }

                if (galaxies[p].H2gas > 0.0 && tdyn > 0.0) {
                    strdot = run_params->SfrEfficiency * galaxies[p].H2gas / tdyn;
                } else {
                    strdot = 0.0;
                }
            }
        }
    } else if(run_params->SFprescription == 5) {

        // ========================================================================
        // Krumholz, McKee, & Tumlinson (2009) - KMT09 Model
        // ========================================================================

        
        // 1. Geometry and Units [cite: 60-64]
        reff = 3.0 * galaxies[p].DiskScaleRadius;
        tdyn = reff / galaxies[p].Vvir;
        
        // Check for physical validity
        if(galaxies[p].Vvir <= 0.0 || galaxies[p].DiskScaleRadius <= 0.0) {
            galaxies[p].H2gas = 0.0;
            strdot = 0.0;
        } else {
            const float h = run_params->Hubble_h;
            // Scale radius in pc
            const float rs_pc = galaxies[p].DiskScaleRadius * 1.0e6 / h;

            if(run_params->H2RadialIntegrationOn) {
                calculate_molecular_fraction_radial_integration(p, galaxies, run_params, NULL);
            } else {
                // Choose disk area based on H2DiskAreaOption
                float disk_area_pc2;
                if (run_params->H2DiskAreaOption == 0) {
                    disk_area_pc2 = M_PI * pow(rs_pc, 2);
                } else if (run_params->H2DiskAreaOption == 1) {
                    disk_area_pc2 = M_PI * pow(3.0 * rs_pc, 2);
                } else {
                    disk_area_pc2 = 2.0 * M_PI * pow(rs_pc, 2);
                }

                // Gas Surface Density (Msun/pc^2) - Sigma_g
                float gas_surface_density = (disk_area_pc2 > 0.0) ?
                    (galaxies[p].ColdGas * 1.0e10 / h) / disk_area_pc2 : 0.0;

                float metallicity_abs = 0.0;
                if(galaxies[p].ColdGas > 0.0) {
                    metallicity_abs = galaxies[p].MetalsColdGas / galaxies[p].ColdGas;
                }
                float Z_prime = (metallicity_abs > 0.0) ? metallicity_abs / 0.02 : 0.0;

                const float clumping_factor = 3.0;
                float Sigma_comp = clumping_factor * gas_surface_density;
                double tau_c = 0.066 * clumping_factor * Z_prime * gas_surface_density;
                float chi = 0.77 * (1.0 + 3.1 * pow(Z_prime, 0.365));
                float s = 0.0;
                if (Sigma_comp > 0.0 && tau_c > 1e-10) {
                    s = log(1.0 + 0.6 * chi + 0.01 * chi * chi) / (0.6 * tau_c);
                } else {
                    s = 100.0;
                }

                float f_H2 = 0.0;
                if (s < 2.0) {
                    f_H2 = 1.0 - (3.0 * s) / (4.0 + s);
                }
                if (f_H2 < 0.0) f_H2 = 0.0;
                if (f_H2 > 1.0) f_H2 = 1.0;

                galaxies[p].H2gas = f_H2 * (galaxies[p].ColdGas * HYDROGEN_MASS_FRAC);
            }

            // Can't create more H2 than total cold gas
            if(galaxies[p].H2gas > galaxies[p].ColdGas * HYDROGEN_MASS_FRAC) {
                galaxies[p].H2gas = galaxies[p].ColdGas * HYDROGEN_MASS_FRAC;
            }

            if (galaxies[p].H2gas > 0.0 && tdyn > 0.0) {
                strdot = run_params->SfrEfficiency * galaxies[p].H2gas / tdyn;
            } else {
                strdot = 0.0;
            }
        }
    } else if(run_params->SFprescription == 6) {

        // ========================================================================
        // Krumholz 2013 (KMT+) Model
        // "The star formation law in molecule-poor galaxies"
        // Uses the analytic approximation for depletion time (Equation 28)
        // ========================================================================

        reff = 3.0 * galaxies[p].DiskScaleRadius;
        tdyn = reff / galaxies[p].Vvir;

        // Basic safety checks
        if(galaxies[p].Vvir <= 0.0 || galaxies[p].ColdGas <= 0.0 || galaxies[p].DiskScaleRadius <= 0.0) {
            strdot = 0.0;
            galaxies[p].H2gas = 0.0;
        } else {
            tdyn = reff / galaxies[p].Vvir; // Code units

            const float h = run_params->Hubble_h;
            const float rs_pc = galaxies[p].DiskScaleRadius * 1.0e6 / h;

            if(run_params->H2RadialIntegrationOn) {
                // Radially integrate both H2 mass and K13 SFR consistently.
                // Sigma(r)/t_dep(r) is summed over the disk using the local f_H2(r) at each annulus,
                // avoiding the single-slab Sigma = M/(pi r_s^2) = 2Sigma0 overestimate.
                double strdot_k13 = 0.0;
                calculate_molecular_fraction_radial_integration(p, galaxies, run_params, &strdot_k13);
                if(galaxies[p].H2gas > galaxies[p].ColdGas * HYDROGEN_MASS_FRAC)
                    galaxies[p].H2gas = galaxies[p].ColdGas * HYDROGEN_MASS_FRAC;
                // H2DepletionTime_Gyr = M_gas / SFR_K13_integrated, set inside function
                strdot = strdot_k13;
            } else {
                // Slab path: single representative surface density from H2DiskAreaOption
                double Sigma_gas_k13 = 0.0, Sigma_star_k13 = 0.0, Z_prime_k13 = 0.01, f_H2_2p_k13 = 0.0;
                float area_pc2;
                if (run_params->H2DiskAreaOption == 0) {
                    area_pc2 = M_PI * pow(rs_pc, 2);
                } else if (run_params->H2DiskAreaOption == 1) {
                    area_pc2 = M_PI * pow(3.0 * rs_pc, 2);
                } else {
                    area_pc2 = 2.0 * M_PI * pow(rs_pc, 2);
                }

                if(area_pc2 > 0.0) {
                    Sigma_gas_k13  = (galaxies[p].ColdGas * 1.0e10 / h) / area_pc2;
                    Sigma_star_k13 = ((galaxies[p].StellarMass - galaxies[p].BulgeMass) * 1.0e10 / h) / area_pc2;
                    double Z_gas = (galaxies[p].ColdGas > 0.0) ? (galaxies[p].MetalsColdGas / galaxies[p].ColdGas) : 0.0;
                    Z_prime_k13 = Z_gas / 0.014; if(Z_prime_k13 < 0.01) Z_prime_k13 = 0.01;
                    const double fc = 5.0;
                    const double chi_2p = 3.1 * (1.0 + 3.1 * pow(Z_prime_k13, 0.365)) / 4.1;
                    const double tau_c = 0.066 * fc * Z_prime_k13 * Sigma_gas_k13;
                    const double s = (tau_c > 0.0) ? log(1.0 + 0.6 * chi_2p + 0.01 * chi_2p * chi_2p) / (0.6 * tau_c) : 100.0;
                    f_H2_2p_k13 = (s < 2.0) ? 1.0 - (0.75 * s) / (1.0 + 0.25 * s) : 0.0;
                    if(f_H2_2p_k13 < 0.0) f_H2_2p_k13 = 0.0;
                    if(f_H2_2p_k13 > 1.0) f_H2_2p_k13 = 1.0;
                    galaxies[p].H2gas = f_H2_2p_k13 * (galaxies[p].ColdGas * HYDROGEN_MASS_FRAC);
                } else {
                    galaxies[p].H2gas = 0.0;
                }

                if(galaxies[p].H2gas > galaxies[p].ColdGas * HYDROGEN_MASS_FRAC)
                    galaxies[p].H2gas = galaxies[p].ColdGas * HYDROGEN_MASS_FRAC;

                const double t_dep_Gyr = calculate_tdep_K13_Gyr((float)Sigma_gas_k13, (float)Sigma_star_k13,
                                                                   rs_pc, (float)Z_prime_k13, (float)f_H2_2p_k13);
                const double t_dep_code = t_dep_Gyr * 1000.0 / run_params->UnitTime_in_Megayears;
                galaxies[p].H2DepletionTime_Gyr = (t_dep_Gyr > 0.0) ? (float)t_dep_Gyr : -1.0f;

                strdot = (galaxies[p].H2gas > 0.0 && tdyn > 0.0)
                         ? run_params->SfrEfficiency * galaxies[p].H2gas / tdyn : 0.0;

                // Alternative: use K13 depletion time directly for SFR, bypassing tdyn. This is more faithful to K13 but less consistent with other prescriptions that use tdyn.
                // strdot = (galaxies[p].ColdGas > 0.0 && t_dep_code > 0.0)
                //          ? galaxies[p].ColdGas / t_dep_code : 0.0;
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

        // Basic safety checks
        if(galaxies[p].Vvir <= 0.0 || galaxies[p].ColdGas <= 0.0 || galaxies[p].DiskScaleRadius <= 0.0) {
            strdot = 0.0;
            galaxies[p].H2gas = 0.0;
        } else {
            tdyn = reff / galaxies[p].Vvir; // Code units
            
            const float h = run_params->Hubble_h;
            // Scale radius in pc
            const float rs_pc = galaxies[p].DiskScaleRadius * 1.0e6 / h;

            if(run_params->H2RadialIntegrationOn) {
                calculate_molecular_fraction_radial_integration(p, galaxies, run_params, NULL);
            } else {
                // Choose disk area based on H2DiskAreaOption
                float disk_area_pc2;
                if (run_params->H2DiskAreaOption == 0) {
                    disk_area_pc2 = M_PI * pow(rs_pc, 2);
                } else if (run_params->H2DiskAreaOption == 1) {
                    disk_area_pc2 = M_PI * pow(3.0 * rs_pc, 2);
                } else {
                    disk_area_pc2 = 2.0 * M_PI * pow(rs_pc, 2);
                }

                double Sigma_gas = 0.0;
                if(disk_area_pc2 > 0.0) {
                    Sigma_gas = (galaxies[p].ColdGas * 1.0e10 / h) / disk_area_pc2;
                }

                double metallicity_abs = 0.0;
                if(galaxies[p].ColdGas > 0.0) {
                    metallicity_abs = galaxies[p].MetalsColdGas / galaxies[p].ColdGas;
                }
                double D_MW = metallicity_abs / 0.02;
                if(D_MW < 1e-4) D_MW = 1e-4;

                const double U_MW    = 1.0;
                const double S       = 3.0 * rs_pc / 100.0;
                const double s_param = pow(0.001 + 0.1 * U_MW, 0.7);
                const double D_star  = 0.17 * (2.0 + pow(S, 5.0)) / (1.0 + pow(S, 5.0));
                const double g       = sqrt(D_MW * D_MW + D_star * D_star);
                const double Sigma_R1 = (40.0 / g) * (s_param / (1.0 + s_param));
                const double alpha   = 1.0 + 0.7 * sqrt(s_param) / (1.0 + s_param);

                double q = 0.0;
                if(Sigma_R1 > 0.0 && Sigma_gas > 0.0) {
                    q = pow(Sigma_gas / Sigma_R1, alpha);
                }
                double f_H2 = q / (1.0 + q);
                if(f_H2 > 1.0) f_H2 = 1.0;
                if(f_H2 < 0.0) f_H2 = 0.0;

                galaxies[p].H2gas = f_H2 * (galaxies[p].ColdGas * HYDROGEN_MASS_FRAC);
            }

            // Can't create more H2 than total cold gas
            if(galaxies[p].H2gas > galaxies[p].ColdGas * HYDROGEN_MASS_FRAC) {
                galaxies[p].H2gas = galaxies[p].ColdGas * HYDROGEN_MASS_FRAC;
            }

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

    // H2SFRMode override: applies to all H2-computing prescriptions except 0, 2, and 6.
    // SFprescription==6 (K13) is excluded because it already uses K13 t_dep natively.
    if(run_params->H2SFRMode > 0 && galaxies[p].H2gas > 0.0 && tdyn > 0.0
       && run_params->SFprescription != 0 && run_params->SFprescription != 2
       && run_params->SFprescription != 6) {
        if(run_params->H2SFRMode == 1) {
            double tdep_code = run_params->H2DepletionTime_Gyr * 1000.0 / run_params->UnitTime_in_Megayears;
            galaxies[p].H2DepletionTime_Gyr = (float)run_params->H2DepletionTime_Gyr;
            if(tdep_code > 0.0) strdot = galaxies[p].H2gas / tdep_code;
        } else {
            // H2SFRMode == 2: K13 depletion time using local f_H2 from base prescription
            if(run_params->H2RadialIntegrationOn) {
                // Re-run radial integration to get K13 SFR with local f_H2(r) from base prescription.
                // H2gas result is identical to the first call; strdot_ri is the new output.
                double strdot_ri = 0.0;
                calculate_molecular_fraction_radial_integration(p, galaxies, run_params, &strdot_ri);
                // H2DepletionTime_Gyr = M_gas / SFR_K13_integrated, set inside function
                if(strdot_ri > 0.0) strdot = strdot_ri;
            } else {
                // Slab path: use H2DiskAreaOption for Sigma; pass f_H2=1 so SFR = H2/tau_dep,H2
                const float h_pp  = run_params->Hubble_h;
                const float rs_pp = (float)(galaxies[p].DiskScaleRadius * 1.0e6 / h_pp);
                float area_pp;
                if(run_params->H2DiskAreaOption == 0)      area_pp = (float)M_PI * rs_pp * rs_pp;
                else if(run_params->H2DiskAreaOption == 1) area_pp = (float)M_PI * 9.0f * rs_pp * rs_pp;
                else                                        area_pp = 2.0f * (float)M_PI * rs_pp * rs_pp;
                const float Sg_pp = (area_pp > 0.0f) ? (float)(galaxies[p].ColdGas * 1.0e10 / h_pp) / area_pp : 0.0f;
                const float Ss_pp = (area_pp > 0.0f) ? (float)((galaxies[p].StellarMass - galaxies[p].BulgeMass) * 1.0e10 / h_pp) / area_pp : 0.0f;
                const float Zp_pp = (galaxies[p].ColdGas > 0.0) ? (float)(galaxies[p].MetalsColdGas / galaxies[p].ColdGas / 0.014) : 0.02f;
                const double tdep_Gyr = calculate_tdep_K13_Gyr(Sg_pp, Ss_pp, rs_pp, Zp_pp, 1.0f);
                double tdep_code = tdep_Gyr * 1000.0 / run_params->UnitTime_in_Megayears;
                galaxies[p].H2DepletionTime_Gyr = (tdep_Gyr > 0.0) ? (float)tdep_Gyr : -1.0f;
                if(tdep_code > 0.0) strdot = galaxies[p].H2gas / tdep_code;
            }
        }
    }

    // Calculate HI (atomic hydrogen) as the remainder of hydrogen after H2
    // Total hydrogen = ColdGas * HYDROGEN_MASS_FRAC (0.74)
    // HI = Total hydrogen - H2
    galaxies[p].H1gas = (galaxies[p].ColdGas * HYDROGEN_MASS_FRAC) - galaxies[p].H2gas;
    if(galaxies[p].H1gas < 0.0) {
        galaxies[p].H1gas = 0.0;  // Safety check
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
                /* BUG FIX: Apply floor to vc to prevent overflow in pow() for very small halos */
                double vc_floored = (vc < 1.0) ? 1.0 : vc;  /* Floor at 1 km/s */
                if (vc_floored < V_CRIT) {
                    v_term = pow(vc_floored / V_CRIT, -3.2);
                } else {
                    v_term = pow(vc_floored / V_CRIT, -1.0);
                }
                double scaling_factor = z_term * v_term;

                // Reheating with Muratov scaling: eta = 2.9 * (1+z)^alpha * (V/60)^beta
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
                    /* BUG FIX: Apply floor to vc to prevent overflow in pow() for very small halos */
                    double vc_floored = (vc < 1.0) ? 1.0 : vc;  /* Floor at 1 km/s */
                    if (vc_floored < V_CRIT) {
                        v_term = pow(vc_floored / V_CRIT, -3.2);
                    } else {
                        v_term = pow(vc_floored / V_CRIT, -1.0);
                    }
                    double scaling_factor = z_term * v_term;
                    
                    // Total feedback energy: E_FB = epsilon_eject * scaling * 0.5 * M_* * (eta_SN * E_SN)
                    double E_FB = run_params->FeedbackEjectionEfficiency * scaling_factor * 
                                  0.5 * stars * (run_params->EtaSNcode * run_params->EnergySNcode);
                    
                    // Energy needed to lift reheated gas to virial radius: E_lift = 0.5 * M_reheat * V_vir^2
                    double E_lift = 0.5 * reheated_mass * vc * vc;
                    
                    // Leftover energy ejects additional gas: E_eject = E_FB - E_lift
                    // Ejected mass: M_eject = E_eject / (0.5 * V_vir^2)
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

    // Track star formation history - accumulate stellar mass formed at this snapshot
    // Note: RecycleFraction * stars is instantly recycled, so actual stellar mass added is (1 - RecycleFraction) * stars
    if(run_params->SaveFullSFH) {
        const int snapnum = galaxies[p].SnapNum;
        if(snapnum >= 0 && snapnum < ABSOLUTEMAXSNAPS) {
            galaxies[p].SFHMassDisk[snapnum] += (1.0 - run_params->RecycleFraction) * stars;
        }
    }

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

/*
 * Apply a star formation event: update cold gas, metals, and stellar mass.
 *
 * Removes (1 - RecycleFraction) * stars from ColdGas (the rest is recycled
 * immediately), increments StellarMass, and tracks metals consistently.
 * Called for each SF event within starformation_and_feedback().
 */
void update_from_star_formation(const int p, const double stars, const double metallicity, struct GALAXY *galaxies, const struct params *run_params)
{
    const double RecycleFraction = run_params->RecycleFraction;
    // update gas and metals from star formation
    galaxies[p].ColdGas -= (1 - RecycleFraction) * stars;
    galaxies[p].MetalsColdGas -= metallicity * (1 - RecycleFraction) * stars;
    galaxies[p].StellarMass += (1 - RecycleFraction) * stars;
    galaxies[p].MetalsStellarMass += metallicity * (1 - RecycleFraction) * stars;

    // H2gas and H1gas were computed before SF depleted ColdGas; clamp so they
    // remain consistent with the remaining cold gas. Only applies to H2-tracking
    // prescriptions (0=Croton and 2=Somerville-noH2 never set H2gas/H1gas).
    const int sf = run_params->SFprescription;
    if(sf != 0 && sf != 2) {
        const float max_h = (galaxies[p].ColdGas > 0.0f) ? galaxies[p].ColdGas * HYDROGEN_MASS_FRAC : 0.0f;
        if(galaxies[p].H2gas > max_h) galaxies[p].H2gas = max_h;
        if(galaxies[p].H1gas > max_h) galaxies[p].H1gas = max_h;
    }
}

// ============================================================================
/*
 * Apply supernova feedback: reheat cold gas and eject hot gas.
 *
 * Transfers reheated_mass from ColdGas to HotGas (tracking metals), and
 * ejects ejected_mass from HotGas to EjectedMass. Handles routing to CGMgas
 * when CGMrecipeOn is active. Both quantities may be zero for quiescent steps.
 */
void update_from_feedback(const int p, const int centralgal, double reheated_mass, double ejected_mass, const double metallicity,
                          struct GALAXY *galaxies, const struct params *run_params)
{
    // Safety: Clamp reheated_mass to available ColdGas to handle floating-point precision errors
    // This can occur when dealing with very small masses (e.g., 1e-44) where rounding errors
    // cause reheated_mass to slightly exceed ColdGas after star formation has consumed gas

    if(reheated_mass > galaxies[p].ColdGas) {
        reheated_mass = galaxies[p].ColdGas;
    }

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
        const int sf_fb = run_params->SFprescription;
        if(sf_fb != 0 && sf_fb != 2) {
            const float max_h_fb = (galaxies[p].ColdGas > 0.0f) ? galaxies[p].ColdGas * HYDROGEN_MASS_FRAC : 0.0f;
            if(galaxies[p].H2gas > max_h_fb) galaxies[p].H2gas = max_h_fb;
            if(galaxies[p].H1gas > max_h_fb) galaxies[p].H1gas = max_h_fb;
        }

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

                // FIX 2.1: Bounds check on metals to prevent floating-point precision issues
                double metalsCGM_to_eject = metallicityCGM * ejected_mass;
                if(metalsCGM_to_eject > galaxies[centralgal].MetalsCGMgas) {
                    metalsCGM_to_eject = galaxies[centralgal].MetalsCGMgas;
                }

                // Eject from CGM to EjectedMass
                galaxies[centralgal].CGMgas -= ejected_mass;
                galaxies[centralgal].MetalsCGMgas -= metalsCGM_to_eject;
                galaxies[centralgal].EjectedMass += ejected_mass;
                galaxies[centralgal].MetalsEjectedMass += metalsCGM_to_eject;

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

                // FIX 2.1: Bounds check on metals to prevent floating-point precision issues
                double metalsHot_to_eject = metallicityHot * ejected_mass;
                if(metalsHot_to_eject > galaxies[centralgal].MetalsHotGas) {
                    metalsHot_to_eject = galaxies[centralgal].MetalsHotGas;
                }

                // Eject from HotGas to EjectedMass
                galaxies[centralgal].HotGas -= ejected_mass;
                galaxies[centralgal].MetalsHotGas -= metalsHot_to_eject;
                galaxies[centralgal].EjectedMass += ejected_mass;
                galaxies[centralgal].MetalsEjectedMass += metalsHot_to_eject;
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

            // FIX 2.1: Bounds check on metals to prevent floating-point precision issues
            double metalsHot_to_eject = metallicityHot * ejected_mass;
            if(metalsHot_to_eject > galaxies[centralgal].MetalsHotGas) {
                metalsHot_to_eject = galaxies[centralgal].MetalsHotGas;
            }

            // Eject from HotGas to EjectedMass
            galaxies[centralgal].HotGas -= ejected_mass;
            galaxies[centralgal].MetalsHotGas -= metalsHot_to_eject;
            galaxies[centralgal].EjectedMass += ejected_mass;
            galaxies[centralgal].MetalsEjectedMass += metalsHot_to_eject;
        }

        galaxies[p].OutflowRate += reheated_mass;
    }
}

// ============================================================================
/*
 * Feedback-free burst (FFB) star formation (Li et al. 2024).
 *
 * Triggered when FeedbackFreeModeOn > 0. Computes a burst SFR from the cold
 * gas with no supernova feedback, updating StellarMass, BulgeMass, SFR history,
 * and cold gas in a single substep. Called instead of the main SF loop when
 * the FFB criterion is met.
 */
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

    // ========================================================================
    // H2 CALCULATION -- only for FeedbackFreeModeOn=6/7 (H2-based FFB SF modes).
    // All other FFB modes use ColdGas for SF and leave H2gas = 0.
    // H1 is derived immediately after.
    // ========================================================================
    const int uses_h2 = (run_params->FeedbackFreeModeOn == 6 || run_params->FeedbackFreeModeOn == 7);
    galaxies[p].H2gas = 0.0;

    if(uses_h2 && galaxies[p].ColdGas > 0.0 && galaxies[p].DiskScaleRadius > 0.0) {
        const float h     = run_params->Hubble_h;
        const float rs_pc = galaxies[p].DiskScaleRadius * 1.0e6 / h;
        const int sfpres  = run_params->SFprescription;
        const int has_h2  = (sfpres == 1 || sfpres == 3 || sfpres == 4 ||
                              sfpres == 5 || sfpres == 6 || sfpres == 7);

        if(rs_pc > 0.0 && has_h2) {
            if(run_params->H2RadialIntegrationOn) {
                // Unified radial integration path -- handles all H2 prescriptions internally
                calculate_molecular_fraction_radial_integration(p, galaxies, run_params, NULL);
            } else {
                // Single-slab path
                float disk_area_pc2;
                if(run_params->H2DiskAreaOption == 0)
                    disk_area_pc2 = M_PI * pow(rs_pc, 2);
                else if(run_params->H2DiskAreaOption == 1)
                    disk_area_pc2 = M_PI * pow(3.0 * rs_pc, 2);
                else
                    disk_area_pc2 = 2.0 * M_PI * pow(rs_pc, 2);

                if(disk_area_pc2 > 0.0) {
                    const float Sigma_gas = (galaxies[p].ColdGas * 1.0e10 / h) / disk_area_pc2;

                    if(sfpres == 1 || sfpres == 3) {
                        // BR06
                        const float Sigma_star = (galaxies[p].StellarMass - galaxies[p].BulgeMass)
                                                 * 1.0e10 / h / disk_area_pc2;
                        galaxies[p].H2gas = calculate_molecular_fraction_BR06(Sigma_gas, Sigma_star, rs_pc)
                                            * (galaxies[p].ColdGas * HYDROGEN_MASS_FRAC);

                    } else if(sfpres == 4) {
                        // KD12
                        const double met = (galaxies[p].ColdGas > 0.0) ?
                            galaxies[p].MetalsColdGas / galaxies[p].ColdGas : 0.0;
                        galaxies[p].H2gas = calculate_H2_fraction_KD12(Sigma_gas, met, 5.0f)
                                            * (galaxies[p].ColdGas * HYDROGEN_MASS_FRAC);

                    } else if(sfpres == 5) {
                        // KMT09
                        float met_abs = (galaxies[p].ColdGas > 0.0) ?
                            galaxies[p].MetalsColdGas / galaxies[p].ColdGas : 0.0;
                        float Z_prime = (met_abs > 0.0f) ? met_abs / 0.02f : 0.0f;
                        const float tau_c = 0.066f * 3.0f * Z_prime * Sigma_gas;
                        const float chi = 0.77f * (1.0f + 3.1f * powf(Z_prime, 0.365f));
                        const float s = (tau_c > 1e-10f) ?
                            logf(1.0f + 0.6f*chi + 0.01f*chi*chi) / (0.6f*tau_c) : 100.0f;
                        float f_H2 = (s < 2.0f) ? 1.0f - (3.0f*s)/(4.0f+s) : 0.0f;
                        if(f_H2 < 0.0f) f_H2 = 0.0f;
                        if(f_H2 > 1.0f) f_H2 = 1.0f;
                        galaxies[p].H2gas = f_H2 * (galaxies[p].ColdGas * HYDROGEN_MASS_FRAC);

                    } else if(sfpres == 6) {
                        // K13: store two-phase molecular fraction (f_H2_2p), matching non-FFB path
                        double Z_gas = (galaxies[p].ColdGas > 0.0) ?
                            galaxies[p].MetalsColdGas / galaxies[p].ColdGas : 0.0;
                        double Z_prime = Z_gas / 0.014;
                        if(Z_prime < 0.01) Z_prime = 0.01;
                        const double chi_2p = 3.1 * (1.0 + 3.1 * pow(Z_prime, 0.365)) / 4.1;
                        const double tau_c = 0.066 * 5.0 * Z_prime * Sigma_gas;
                        const double s = (tau_c > 0.0) ?
                            log(1.0 + 0.6*chi_2p + 0.01*chi_2p*chi_2p) / (0.6*tau_c) : 100.0;
                        double f_H2_2p = (s < 2.0) ? 1.0 - (0.75*s)/(1.0+0.25*s) : 0.0;
                        if(f_H2_2p < 0.0) f_H2_2p = 0.0;
                        if(f_H2_2p > 1.0) f_H2_2p = 1.0;
                        galaxies[p].H2gas = f_H2_2p * (galaxies[p].ColdGas * HYDROGEN_MASS_FRAC);

                    } else if(sfpres == 7) {
                        // GD14
                        double met_abs = (galaxies[p].ColdGas > 0.0) ?
                            galaxies[p].MetalsColdGas / galaxies[p].ColdGas : 0.0;
                        double D_MW = met_abs / 0.02;
                        if(D_MW < 1e-4) D_MW = 1e-4;
                        const double S       = 3.0 * rs_pc / 100.0;
                        const double s_param = pow(0.101, 0.7);  // U_MW = 1.0
                        const double D_star  = 0.17 * (2.0 + pow(S, 5.0)) / (1.0 + pow(S, 5.0));
                        const double g       = sqrt(D_MW*D_MW + D_star*D_star);
                        const double Sigma_R1 = (g > 0.0) ? (40.0/g) * (s_param/(1.0+s_param)) : 1e10;
                        const double alpha   = 1.0 + 0.7*sqrt(s_param)/(1.0+s_param);
                        const double q       = (Sigma_R1 > 0.0 && Sigma_gas > 0.0) ?
                            pow(Sigma_gas / Sigma_R1, alpha) : 0.0;
                        double f_H2 = q / (1.0 + q);
                        if(f_H2 > 1.0) f_H2 = 1.0;
                        if(f_H2 < 0.0) f_H2 = 0.0;
                        galaxies[p].H2gas = f_H2 * (galaxies[p].ColdGas * HYDROGEN_MASS_FRAC);
                    }
                }
            }
        }
    }

    if(galaxies[p].H2gas > galaxies[p].ColdGas * HYDROGEN_MASS_FRAC) galaxies[p].H2gas = galaxies[p].ColdGas * HYDROGEN_MASS_FRAC;

    // HI = total hydrogen - H2, matching non-FFB path
    galaxies[p].H1gas = (galaxies[p].ColdGas * HYDROGEN_MASS_FRAC) - galaxies[p].H2gas;
    if(galaxies[p].H1gas < 0.0) galaxies[p].H1gas = 0.0;

    // ========================================================================
    // SELECT GAS RESERVOIR FOR FFB STAR FORMATION
    // ========================================================================
    const double gas_for_sf = uses_h2 ? galaxies[p].H2gas : galaxies[p].ColdGas;

    // ========================================================================
    // COMPUTE STAR FORMATION RATE
    // SFR = epsilon_FFB * M_gas / t_dyn  (no critical density threshold in FFB)
    // ========================================================================
    if(isnan(galaxies[p].ColdGas) || isinf(galaxies[p].ColdGas) ||
       isnan(galaxies[p].Vvir)    || isinf(galaxies[p].Vvir)    ||
       isnan(reff) || isinf(reff) || isnan(tdyn) || isinf(tdyn)) {
        stars = 0.0;
    } else if(tdyn > 0.0 && gas_for_sf > 0.0) {
        const double epsilon_ffb = run_params->FFBMaxEfficiency;
        strdot = epsilon_ffb * gas_for_sf / tdyn;

        if(isnan(strdot) || isinf(strdot) || strdot < 0.0) {
            stars = 0.0;
        } else {
            stars = strdot * dt;
            if(stars > galaxies[p].ColdGas) stars = galaxies[p].ColdGas;
            if(isnan(stars) || isinf(stars) || stars < 0.0) stars = 0.0;
        }
    } else {
        stars = 0.0;
    }

    // ========================================================================
    // SFR TRACKING AND STAR FORMATION UPDATE
    // ========================================================================
    galaxies[p].SfrDisk[step] += stars / dt;
    galaxies[p].SfrDiskColdGas[step]       = galaxies[p].ColdGas;
    galaxies[p].SfrDiskColdGasMetals[step] = galaxies[p].MetalsColdGas;

    metallicity = get_metallicity(galaxies[p].ColdGas, galaxies[p].MetalsColdGas);
    update_from_star_formation(p, stars, metallicity, galaxies, run_params);

    if(run_params->SaveFullSFH) {
        const int snapnum = galaxies[p].SnapNum;
        if(snapnum >= 0 && snapnum < ABSOLUTEMAXSNAPS) {
            galaxies[p].SFHMassDisk[snapnum] += (1.0 - run_params->RecycleFraction) * stars;
        }
    }

    // ========================================================================
    // SUPERNOVA FEEDBACK
    // ========================================================================
    double reheated_mass = 0.0;
    double ejected_mass  = 0.0;

    if(run_params->SupernovaRecipeOn == 1) {
        if(run_params->FIREmodeOn == 1) {
            const double z       = run_params->ZZ[galaxies[p].SnapNum];
            const double vc      = galaxies[p].Vvir;
            const double V_CRIT  = 60.0;

            if(vc > 0.0 && z >= 0.0) {
                const double vc_floored  = (vc < 1.0) ? 1.0 : vc;
                const double z_term      = pow(1.0 + z, run_params->RedshiftPowerLawExponent);
                const double v_term      = (vc_floored < V_CRIT) ?
                    pow(vc_floored / V_CRIT, -3.2) : pow(vc_floored / V_CRIT, -1.0);
                const double scaling     = z_term * v_term;
                const double eta_reheat  = run_params->FeedbackReheatingEpsilon * scaling;
                galaxies[p].MassLoading  = (float)eta_reheat;
                reheated_mass            = eta_reheat * stars;
            }
        } else {
            reheated_mass = run_params->FeedbackReheatingEpsilon * stars;
        }
    }

    XASSERT(reheated_mass >= 0.0, -1,
            "Error: Expected reheated gas-mass = %g to be >=0.0\n", reheated_mass);

    if((stars + reheated_mass) > galaxies[p].ColdGas && (stars + reheated_mass) > 0.0) {
        const double fac = galaxies[p].ColdGas / (stars + reheated_mass);
        stars         *= fac;
        reheated_mass *= fac;
    }

    if(run_params->SupernovaRecipeOn == 1) {
        if(galaxies[p].Vvir > 0.0) {
            if(run_params->FIREmodeOn == 1) {
                const double z      = run_params->ZZ[galaxies[p].SnapNum];
                const double vc     = galaxies[p].Vvir;
                const double V_CRIT = 60.0;

                if(vc > 0.0 && z >= 0.0) {
                    const double vc_floored = (vc < 1.0) ? 1.0 : vc;
                    const double z_term     = pow(1.0 + z, run_params->RedshiftPowerLawExponent);
                    const double v_term     = (vc_floored < V_CRIT) ?
                        pow(vc_floored / V_CRIT, -3.2) : pow(vc_floored / V_CRIT, -1.0);
                    const double scaling    = z_term * v_term;
                    const double E_FB       = run_params->FeedbackEjectionEfficiency * scaling
                                              * 0.5 * stars
                                              * (run_params->EtaSNcode * run_params->EnergySNcode);
                    const double E_lift     = 0.5 * reheated_mass * vc * vc;
                    ejected_mass = (E_FB > E_lift) ? (E_FB - E_lift) / (0.5 * vc * vc) : 0.0;
                }
            } else {
                ejected_mass = (run_params->FeedbackEjectionEfficiency
                                * (run_params->EtaSNcode * run_params->EnergySNcode)
                                / (galaxies[p].Vvir * galaxies[p].Vvir)
                                - run_params->FeedbackReheatingEpsilon) * stars;
            }
        }
        if(ejected_mass < 0.0) ejected_mass = 0.0;
    }

    if(reheated_mass > galaxies[p].ColdGas) reheated_mass = galaxies[p].ColdGas;

    update_from_feedback(p, centralgal, reheated_mass, ejected_mass, metallicity, galaxies, run_params);

    // ========================================================================
    // METAL PRODUCTION (instantaneous recycling approximation - SNII only)
    // ========================================================================
    if(galaxies[p].ColdGas > 1.0e-8) {
        const double FracZleaveDiskVal = run_params->FracZleaveDisk
                                         * exp(-1.0 * galaxies[centralgal].Mvir / 30.0);
        galaxies[p].MetalsColdGas += run_params->Yield * (1.0 - FracZleaveDiskVal) * stars;

        const double metals_leaving_disk = run_params->Yield * FracZleaveDiskVal * stars;
        if(run_params->CGMrecipeOn == 1) {
            if(galaxies[centralgal].Regime == 0)
                galaxies[centralgal].MetalsCGMgas  += metals_leaving_disk;
            else
                galaxies[centralgal].MetalsHotGas  += metals_leaving_disk;
        } else {
            galaxies[centralgal].MetalsHotGas += metals_leaving_disk;
        }
    } else {
        const double all_metals = run_params->Yield * stars;
        if(run_params->CGMrecipeOn == 1) {
            if(galaxies[centralgal].Regime == 0)
                galaxies[centralgal].MetalsCGMgas += all_metals;
            else
                galaxies[centralgal].MetalsHotGas += all_metals;
        } else {
            galaxies[centralgal].MetalsHotGas += all_metals;
        }
    }

    // ========================================================================
    // NO DISK INSTABILITY CHECK -- rapid SF stabilizes the disk
    // ========================================================================
}
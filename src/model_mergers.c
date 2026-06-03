/*
 * model_mergers.c -- galaxy merger physics.
 *
 * Implements the full merger pipeline: dynamical friction timescale
 * (estimate_merging_time), remnant bulge-radius calculation
 * (calculate_merger_remnant_radius, file-private), merger classification
 * and mass redistribution (deal_with_galaxy_merger), AGN accretion modes
 * (grow_black_hole, quasar_mode_wind), galaxy addition (add_galaxies_together,
 * make_bulge_from_burst), the collisional starburst recipe for both mergers
 * and disk instabilities (collisional_starburst_recipe), and satellite
 * disruption into the ICS (disrupt_satellite_to_ICS).
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "core_allvars.h"

#include "model_mergers.h"
#include "model_misc.h"
#include "model_starformation_and_feedback.h"
#include "model_disk_instability.h"

/* -------------------------------------------------------------------------
 * File-scope empirical constants (lifted per STYLE_C.md SS8).
 * -------------------------------------------------------------------------*/

/* Exponential disk half-mass radius factor: r_half = DISK_HALF_MASS_FRAC * r_s.
 * Exact result from integration of Sigma(r) = Sigma_0 exp(-r/r_s). */
static const double DISK_HALF_MASS_FRAC = 1.68;

/* Covington et al. (2011) radiative loss coefficient for gas-rich mergers.
 * Scales the energy radiated by the gas component during the merger. */
static const double COVINGTON11_C_RAD = 2.75;

/* Kauffmann & Haehnelt (2000) BH-growth velocity scale: cold-gas accretion rate
 * falls off as 1 / (1 + (BH_GROWTH_V_KMS / Vvir)^2). */
static const double BH_GROWTH_V_KMS = 280.0;  /* km/s */

/* Somerville et al. (2001) starburst burst fraction for disk instabilities:
 * eburst = STARBURST_FRAC_COEFF * mass_ratio^STARBURST_MASS_POWER.
 * Merger mode uses eburst = mass_ratio directly. */
static const double STARBURST_FRAC_COEFF = 0.56;
static const double STARBURST_MASS_POWER = 0.7;

/* FIRE (Muratov et al. 2015) critical velocity separating the two power-law
 * slopes of the wind loading factor.  Same value as in
 * model_starformation_and_feedback.c. */
static const double FIRE_V_CRIT_KMS = 60.0;  /* km/s */

/* Krumholz & Dekel (2011) characteristic halo mass for metal enrichment scaling:
 * FracZleaveDisk ~ exp(-Mvir / KD11_METAL_HALO_MASS) in code units (10^10 Msun/h).
 * Same constant used in model_starformation_and_feedback.c. */
static const double KD11_METAL_HALO_MASS = 30.0;  /* 10^10 Msun/h */

/* Solar metallicity (Asplund et al. 2009): used to normalise Z' in K13.
 * Z' = Z_gas / Z_SOLAR_ASPLUND09.
 * Same constant used in model_starformation_and_feedback.c. */
static const double Z_SOLAR_ASPLUND09 = 0.014;

/* Solar metallicity used by GD14 (Grevesse & Sauval 1998 value).
 * Dust-to-gas ratio normalisation: D_MW = Z_gas / Z_SOLAR_GD14.
 * Also used as the Z-normalisation in KMT09.
 * Same constant used in model_starformation_and_feedback.c. */
static const double Z_SOLAR_GD14 = 0.02;

/* Gnedin & Draine (2014) UV-field s-parameter at U_MW = 1.0 (Milky Way).
 * s_param = pow(GD14_S_PARAM_UMW1, 0.7) from their eq. 11 / Table 1.
 * Same constant used in model_starformation_and_feedback.c. */
static const double GD14_S_PARAM_UMW1 = 0.101;

/* File-private forward declaration */
static double calculate_merger_remnant_radius(const struct GALAXY *g1, const struct GALAXY *g2);

/*
 * estimate_merging_time -- compute the dynamical friction merger timescale
 * for a satellite entering mother_halo.
 *
 * Uses the Binney & Tremaine (1987) dynamical friction formula scaled by
 * MergerTimeFactor.  Returns the merger time in code units (Myr/h).
 */
double estimate_merging_time(const int sat_halo, const int mother_halo, const int ngal, struct halo_data *halos, struct GALAXY *galaxies, const struct params *run_params)
{
    double mergtime;
    const int MinNumPartSatHalo = 10;

    if(sat_halo == mother_halo) {
        fprintf(stderr, "Error: \t\tSnapNum, Type, IDs, sat radius:\t%i\t%i\t%i\t%i\t--- sat/cent have the same ID\n",
               galaxies[ngal].SnapNum, galaxies[ngal].Type, sat_halo, mother_halo);
        return -1.0;
    }

    const double coulomb = log1p(halos[mother_halo].Len / ((double) halos[sat_halo].Len) );//MS: 12/9/2019. As pointed out by codacy -> log1p(x) is better than log(1 + x)

    const double SatelliteMass = get_virial_mass(sat_halo, halos, run_params) + galaxies[ngal].StellarMass + galaxies[ngal].ColdGas;
    const double SatelliteRadius = get_virial_radius(mother_halo, halos, run_params);

    if(SatelliteMass > 0.0 && coulomb > 0.0 && halos[sat_halo].Len >= MinNumPartSatHalo) {
        mergtime = 2.0 *
            1.17 * SatelliteRadius * SatelliteRadius * get_virial_velocity(mother_halo, halos, run_params) / (coulomb * run_params->G * SatelliteMass);
    } else {
        mergtime = -1.0;
    }

    if (mergtime >= 999.0)
    {
        mergtime = 998.0;
        // implementing time ceiling since some objects have merge times longer than universe age when using
        // TNG50 merger trees because of lower simulation particle mass 
    }

    return mergtime;

}

// ============================================================================
// Determine the radius of merger remnant
// ============================================================================

/*
 * calculate_merger_remnant_radius -- compute the virial radius of the merger
 * remnant bulge using energy conservation.
 *
 * Applies the binding energy formula: R_rem = (M1+M2)^2 / (M1/R1 + M2/R2 +
 * 0.5*(M1+M2)^2/(M1*R1+M2*R2)) where M, R are baryonic mass and half-mass
 * radius for each progenitor.
 */
static double calculate_merger_remnant_radius(const struct GALAXY *g1, const struct GALAXY *g2)
{
    // 1. Calculate Total Baryonic Mass (Stars + Gas) for both progenitors
    double M1 = g1->StellarMass + g1->ColdGas;
    double M2 = g2->StellarMass + g2->ColdGas;
    double M_tot = M1 + M2;

    if (M_tot <= 0.0) return 0.0;

    // 2. Calculate Half-Mass Radius for both progenitors
    // For Discs: R_half ~ 1.68 * R_scale (Exponential profile)
    // For Bulges: We assume the stored radius is the half-mass radius
    
    // Progenitor 1 (Central)
    double R1_disk_half = DISK_HALF_MASS_FRAC * g1->DiskScaleRadius;
    double R1_bulge_half = g1->BulgeRadius;
    double R1;

    if (g1->StellarMass + g1->ColdGas > 0) {
        // Mass-weighted average radius of the whole galaxy
        // Note: For pure discs, BulgeMass is 0, so this works naturally
        double M1_disk = g1->ColdGas + (g1->StellarMass - g1->BulgeMass);
        double M1_bulge = g1->BulgeMass;
        R1 = (M1_disk * R1_disk_half + M1_bulge * R1_bulge_half) / M1;
    } else {
        R1 = 0.0;
    }

    // Progenitor 2 (Satellite)
    double R2_disk_half = DISK_HALF_MASS_FRAC * g2->DiskScaleRadius;
    double R2_bulge_half = g2->BulgeRadius;
    double R2;

    if (g2->StellarMass + g2->ColdGas > 0) {
        double M2_disk = g2->ColdGas + (g2->StellarMass - g2->BulgeMass);
        double M2_bulge = g2->BulgeMass;
        R2 = (M2_disk * R2_disk_half + M2_bulge * R2_bulge_half) / M2;
    } else {
        R2 = 0.0;
    }

    // Safeguard against zero radius (e.g., pure gas cloud with no set radius yet)
    if (R1 <= 0.0) R1 = R2; 
    if (R2 <= 0.0) R2 = R1;
    if (R1 <= 0.0) return 0.0; // Both zero

    // 3. Calculate Energy Terms (ignoring G, as it cancels out)
    // We use "Potential" units: P = M^2 / R
    
    // E_initial (Eq 21): Self-binding energy of progenitors
    double E_init = (M1 * M1) / R1 + (M2 * M2) / R2;

    // E_orbital (Eq 22): Interaction energy at merger
    // Approximated as circular orbit energy at separation R1 + R2
    double E_orb = (M1 * M2) / (R1 + R2);

    // E_rad (Eq 23): Radiative losses due to gas
    // C_rad from Covington et al. (2011), calibrated to hydrodynamic simulations
    double C_rad = COVINGTON11_C_RAD;
    double f_gas = (g1->ColdGas + g2->ColdGas) / M_tot;
    double E_rad = C_rad * E_init * f_gas;

    // 4. Total Final Energy (Eq 20)
    // E_final = E_init + E_orb + E_rad
    double E_final = E_init + E_orb + E_rad;

    // High gas fractions can make E_rad dominant; fall back to mass-weighted average
    if(E_final <= 0.0) {
        // Fallback: use mass-weighted average of progenitor radii
        return (M1 * R1 + M2 * R2) / M_tot;
    }

    // 5. Final Radius (Eq 17 rearranged)
    // R_final = M_tot^2 / E_final
    double R_final = (M_tot * M_tot) / E_final;

    return R_final;
}

// ============================================================================
// Actually merge the galaxies and apply the starburst recipe
// This is called from both mergers and disk instabilities, but the merger case is more complex
// ============================================================================

/*
 * deal_with_galaxy_merger -- process one galaxy merger event.
 *
 * Classifies the event as major (mass_ratio > MajorMergerFraction) or minor,
 * calls collisional_starburst_recipe(), updates bulge mass and merger-origin
 * bulge radius, merges stellar/gas reservoirs via add_galaxies_together(), and
 * disrupts the satellite.  AGN growth is triggered on both major and minor
 * mergers via grow_black_hole().
 */
void deal_with_galaxy_merger(const int p, const int merger_centralgal, const int centralgal,
                             const double time, const double dt, const int halonr, const int step,
                             struct GALAXY *galaxies, const struct params *run_params)
{
    double mi, ma, mass_ratio;

    // calculate mass ratio of merging galaxies
    if(galaxies[p].StellarMass + galaxies[p].ColdGas < galaxies[merger_centralgal].StellarMass + galaxies[merger_centralgal].ColdGas) {
        mi = galaxies[p].StellarMass + galaxies[p].ColdGas;
        ma = galaxies[merger_centralgal].StellarMass + galaxies[merger_centralgal].ColdGas;
    } else {
        mi = galaxies[merger_centralgal].StellarMass + galaxies[merger_centralgal].ColdGas;
        ma = galaxies[p].StellarMass + galaxies[p].ColdGas;
    }

    if(ma > 0) {
        mass_ratio = mi / ma;
    } else if(mi > 0) {
        mass_ratio = 1.0;
    } else {
        mass_ratio = 0.0;
    }

    // Determine Central Morphology BEFORE adding satellite
    // This determines where burst stars will go
    double central_disk_mass = galaxies[merger_centralgal].StellarMass - galaxies[merger_centralgal].BulgeMass;
    int is_disk_dominated = (central_disk_mass > 0.5 * galaxies[merger_centralgal].StellarMass);
    
    // Save disc radius BEFORE merger for instability bulge radius update
    const double old_disk_radius = galaxies[merger_centralgal].DiskScaleRadius;

    add_galaxies_together(merger_centralgal, p, galaxies, run_params);

    // grow black hole through accretion from cold disk during mergers
    if(run_params->AGNrecipeOn) {
        grow_black_hole(merger_centralgal, mass_ratio, galaxies, run_params);
    }

    // Determine which bulge component will receive burst stars
    // This must be decided BEFORE the starburst
    int burst_to_merger_bulge = 0;  // 0 = instability, 1 = merger
    
    if(mass_ratio > run_params->ThreshMajorMerger) {
        // Major merger: all stars go to merger-driven bulge
        burst_to_merger_bulge = 1;
    } else {
        // Minor merger: depends on morphology
        if(is_disk_dominated) {
            // Disc-dominated: burst goes to instability bulge
            burst_to_merger_bulge = 0;
        } else {
            // Spheroid-dominated: burst goes to merger bulge
            burst_to_merger_bulge = 1;
        }
    }

    // starburst recipe - now tracks which bulge component receives the stars
    collisional_starburst_recipe(mass_ratio, merger_centralgal, centralgal, time, dt, halonr,
                                 0, step, burst_to_merger_bulge, old_disk_radius,
                                 galaxies, run_params);

    // Sync the central's BulgeRadius after add_galaxies_together + starburst have
    // modified bulge mass.  calculate_merger_remnant_radius reads BulgeRadius for
    // the energy conservation calculation; a stale or zero value there would bias
    // the resulting remnant radius.
    get_bulge_radius(merger_centralgal, galaxies, run_params);

    // 1. Calculate the merger remnant radius via Energy Conservation
    // We do this AFTER the starburst so the energy budget includes burst stars
    double new_merger_radius = calculate_merger_remnant_radius(&galaxies[merger_centralgal], &galaxies[p]);

    if(mass_ratio > run_params->ThreshMajorMerger) {
        // CASE 1: MAJOR MERGER (Section 5.2.3)
        // Destroys disc, creates pure merger-driven bulge
        make_bulge_from_burst(merger_centralgal, galaxies);
        
        // Apply the Energy Conservation Radius; then call get_bulge_radius so
        // the Shen fallsafe fires immediately if new_merger_radius == 0 (edge
        // case: both progenitors were orphan satellites with DiskScaleRadius==0).
        galaxies[merger_centralgal].MergerBulgeRadius = new_merger_radius;
        get_bulge_radius(merger_centralgal, galaxies, run_params);

        galaxies[merger_centralgal].TimeOfLastMajorMerger = time;
        galaxies[p].mergeType = 2; 

    } else {
        // CASE 2: MINOR MERGER
        galaxies[p].mergeType = 1;
        galaxies[merger_centralgal].TimeOfLastMinorMerger = time;

        if (is_disk_dominated) {
            // Minor merger on DISC (Section 5.2.1)
            // InstabilityBulgeRadius is updated inside update_instability_bulge_radius.
            // We still call get_bulge_radius here to recompute BulgeRadius and to run
            // the Shen failsafe for MergerBulgeRadius, which can be stale when the
            // satellite carried MergerBulgeMass but had no disk mass (satellite_disk_mass==0)
            // and no starburst fired (stars==0), leaving no radius-update path.
            get_bulge_radius(merger_centralgal, galaxies, run_params);
        } else {
            // Minor merger on SPHEROID (Section 5.2.3)
            // Update merger bulge radius with energy conservation
            galaxies[merger_centralgal].MergerBulgeRadius = new_merger_radius;
            get_bulge_radius(merger_centralgal, galaxies, run_params);
        }
    }
}

// ============================================================================
// Grow black hole through accretion from cold disk during mergers
// ============================================================================

/*
 * grow_black_hole -- accrete cold gas onto the central black hole.
 *
 * Scales the accreted mass by (mass_ratio / (mass_ratio + BlackHoleCouplingFactor))
 * and removes it from the cold gas reservoir.  Computes quasar-mode energy
 * output for AGNrecipeOn == 1 via quasar_mode_wind().
 */
void grow_black_hole(const int merger_centralgal, const double mass_ratio, struct GALAXY *galaxies, const struct params *run_params)
{
    double BHaccrete, metallicity;

    if(galaxies[merger_centralgal].ColdGas > 0.0) {
        BHaccrete = run_params->BlackHoleGrowthRate * mass_ratio /
            (1.0 + SQR(BH_GROWTH_V_KMS / galaxies[merger_centralgal].Vvir)) * galaxies[merger_centralgal].ColdGas;

        // cannot accrete more gas than is available!
        if(BHaccrete > galaxies[merger_centralgal].ColdGas) {
            BHaccrete = galaxies[merger_centralgal].ColdGas;
        }

        metallicity = get_metallicity(galaxies[merger_centralgal].ColdGas, galaxies[merger_centralgal].MetalsColdGas);
        galaxies[merger_centralgal].BlackHoleMass += BHaccrete;
        galaxies[merger_centralgal].ColdGas -= BHaccrete;
        galaxies[merger_centralgal].MetalsColdGas -= metallicity * BHaccrete;
        if(galaxies[merger_centralgal].MetalsColdGas < 0.0) {
            galaxies[merger_centralgal].MetalsColdGas = 0.0;
        }
        const int sf_bh = run_params->SFprescription;
        if(sf_bh != 0 && sf_bh != 2) {
            const float max_h_bh = (galaxies[merger_centralgal].ColdGas > 0.0f)
                                    ? galaxies[merger_centralgal].ColdGas * HYDROGEN_MASS_FRAC : 0.0f;
            if(galaxies[merger_centralgal].H2gas > max_h_bh) galaxies[merger_centralgal].H2gas = max_h_bh;
            if(galaxies[merger_centralgal].H1gas > max_h_bh) galaxies[merger_centralgal].H1gas = max_h_bh;
        }

        quasar_mode_wind(merger_centralgal, BHaccrete, galaxies, run_params);

        galaxies[merger_centralgal].QuasarModeBHaccretionMass += BHaccrete;
    }
}

// ============================================================================
// QUASARS: Eject gas from galaxy based on energy of quasar-mode wind
// ============================================================================

/*
 * quasar_mode_wind -- eject cold gas via quasar-mode feedback.
 *
 * Computes the quasar wind energy from BH accretion and ejects cold gas
 * proportionally.  Ejected mass goes to the EjectedMass reservoir.
 */
void quasar_mode_wind(const int gal, const double BHaccrete, struct GALAXY *galaxies, const struct params *run_params)
{
    // work out total energy in quasar wind (eta*m*c^2)
    const double quasar_energy = run_params->QuasarModeEfficiency * 0.1 * BHaccrete * (C / run_params->UnitVelocity_in_cm_per_s) * (C / run_params->UnitVelocity_in_cm_per_s);
    const double cold_gas_energy = 0.5 * galaxies[gal].ColdGas * galaxies[gal].Vvir * galaxies[gal].Vvir;

    // compare quasar wind and cold gas energies and eject cold
    if(quasar_energy > cold_gas_energy) {
        galaxies[gal].EjectedMass += galaxies[gal].ColdGas;
        galaxies[gal].MetalsEjectedMass += galaxies[gal].MetalsColdGas;

        galaxies[gal].ColdGas = 0.0;
        galaxies[gal].MetalsColdGas = 0.0;
    }

    // compare quasar wind and cold+hot/CGM gas energies and eject from appropriate reservoir
    if(run_params->CGMrecipeOn == 1) {
        if(galaxies[gal].Regime == 0) {
            // CGM-regime: check and eject from CGM
            const double cgm_gas_energy = 0.5 * galaxies[gal].CGMgas * galaxies[gal].Vvir * galaxies[gal].Vvir;
            
            if(quasar_energy > cold_gas_energy + cgm_gas_energy) {
                galaxies[gal].EjectedMass += galaxies[gal].CGMgas;
                galaxies[gal].MetalsEjectedMass += galaxies[gal].MetalsCGMgas;

                galaxies[gal].CGMgas = 0.0;
                galaxies[gal].MetalsCGMgas = 0.0;
            }
        } else {
            // Hot-ICM-regime: check and eject from HotGas
            const double hot_gas_energy = 0.5 * galaxies[gal].HotGas * galaxies[gal].Vvir * galaxies[gal].Vvir;
            
            if(quasar_energy > cold_gas_energy + hot_gas_energy) {
                galaxies[gal].EjectedMass += galaxies[gal].HotGas;
                galaxies[gal].MetalsEjectedMass += galaxies[gal].MetalsHotGas;

                galaxies[gal].HotGas = 0.0;
                galaxies[gal].MetalsHotGas = 0.0;
            }
        }
    } else {
        // Original SAGE behavior: check and eject from HotGas
        const double hot_gas_energy = 0.5 * galaxies[gal].HotGas * galaxies[gal].Vvir * galaxies[gal].Vvir;
        
        if(quasar_energy > cold_gas_energy + hot_gas_energy) {
            galaxies[gal].EjectedMass += galaxies[gal].HotGas;
            galaxies[gal].MetalsEjectedMass += galaxies[gal].MetalsHotGas;

            galaxies[gal].HotGas = 0.0;
            galaxies[gal].MetalsHotGas = 0.0;
        }
    }
}

// ============================================================================
// Actually merge the galaxies together by adding their properties, and apply the starburst recipe
// ============================================================================

/*
 * add_galaxies_together -- merge all baryonic reservoirs of satellite p into
 * central t.
 *
 * Adds stellar mass, cold/hot/ejected gas, metals, ICS, H2, and CGM gas from p
 * to t, transferring satellite-owned data (SFH arrays, infall properties, ICS
 * assembly history) to the central.
 */
void add_galaxies_together(const int t, const int p, struct GALAXY *galaxies, const struct params *run_params)
{
    galaxies[t].ColdGas += galaxies[p].ColdGas;
    galaxies[t].MetalsColdGas += galaxies[p].MetalsColdGas;

    galaxies[t].StellarMass += galaxies[p].StellarMass;
    galaxies[t].MetalsStellarMass += galaxies[p].MetalsStellarMass;

    galaxies[t].HotGas += galaxies[p].HotGas;
    galaxies[t].MetalsHotGas += galaxies[p].MetalsHotGas;

    galaxies[t].EjectedMass += galaxies[p].EjectedMass;
    galaxies[t].MetalsEjectedMass += galaxies[p].MetalsEjectedMass;

    galaxies[t].ICS += galaxies[p].ICS;
    galaxies[t].MetalsICS += galaxies[p].MetalsICS;

    galaxies[t].BlackHoleMass += galaxies[p].BlackHoleMass;

    galaxies[t].CGMgas += galaxies[p].CGMgas;
    galaxies[t].MetalsCGMgas += galaxies[p].MetalsCGMgas;

    if (run_params->SFprescription == 1 || run_params->SFprescription == 3 ||
        run_params->SFprescription == 4 || run_params->SFprescription == 5 ||
        run_params->SFprescription == 6 || run_params->SFprescription == 7) {
        galaxies[t].H2gas += galaxies[p].H2gas;
        galaxies[t].H1gas += galaxies[p].H1gas;
    }

    // add merger to bulge
    galaxies[t].BulgeMass += galaxies[p].StellarMass;
    galaxies[t].MetalsBulgeMass += galaxies[p].MetalsStellarMass;

    // Transfer the satellite's bulge component breakdown to the central
    galaxies[t].InstabilityBulgeMass += galaxies[p].InstabilityBulgeMass;
    galaxies[t].MergerBulgeMass += galaxies[p].MergerBulgeMass;

    // The satellite's DISK mass (StellarMass - BulgeMass) becomes new bulge mass
    // Track this based on the central's current morphology (Tonini+2016 logic)
    const double satellite_disk_mass = galaxies[p].StellarMass - galaxies[p].BulgeMass;

    if(satellite_disk_mass > 0.0) {
        const double disk_mass = galaxies[t].StellarMass - galaxies[t].BulgeMass;
        const double disk_fraction = (galaxies[t].StellarMass > 0.0) ?
                                     disk_mass / galaxies[t].StellarMass : 0.0;

        if(disk_fraction > 0.5) {
            // Disc-dominated: minor merger triggers instability
            galaxies[t].InstabilityBulgeMass += satellite_disk_mass;
            const double old_disk_radius = galaxies[t].DiskScaleRadius;

            // UPDATE: Tonini incremental radius evolution (equation 16)
            update_instability_bulge_radius(t, satellite_disk_mass, old_disk_radius, galaxies, run_params);
        } else {
            // Spheroid-dominated: grows merger bulge
            galaxies[t].MergerBulgeMass += satellite_disk_mass;
        }
    }

    for(int step = 0; step < STEPS; step++) {
        galaxies[t].SfrBulge[step] += galaxies[p].SfrDisk[step] + galaxies[p].SfrBulge[step];
        galaxies[t].SfrBulgeColdGas[step] += galaxies[p].SfrDiskColdGas[step] + galaxies[p].SfrBulgeColdGas[step];
        galaxies[t].SfrBulgeColdGasMetals[step] += galaxies[p].SfrDiskColdGasMetals[step] + galaxies[p].SfrBulgeColdGasMetals[step];
    }

    // Transfer star formation history from satellite to central
    // During a merger, the central inherits all star formation history from the satellite
    if(run_params->SaveFullSFH) {
        for(int snap = 0; snap < ABSOLUTEMAXSNAPS; snap++) {
            galaxies[t].SFHMassDisk[snap] += galaxies[p].SFHMassDisk[snap];
            galaxies[t].SFHMassBulge[snap] += galaxies[p].SFHMassBulge[snap];
        }
    }
}

// ============================================================================
// Bulges
// ============================================================================

/*
 * make_bulge_from_burst -- transfer all stellar disk mass to the bulge after a
 * major merger starburst.
 */
void make_bulge_from_burst(const int p, struct GALAXY *galaxies)
{
    // generate bulge
    galaxies[p].BulgeMass = galaxies[p].StellarMass;
    galaxies[p].MergerBulgeMass = galaxies[p].StellarMass;      // All merger-driven
    galaxies[p].InstabilityBulgeMass = 0.0;                      // Destroyed
    galaxies[p].MetalsBulgeMass = galaxies[p].MetalsStellarMass;

    // update the star formation rate
    for(int step = 0; step < STEPS; step++) {
        galaxies[p].SfrBulge[step] += galaxies[p].SfrDisk[step];
        galaxies[p].SfrBulgeColdGas[step] += galaxies[p].SfrDiskColdGas[step];
        galaxies[p].SfrBulgeColdGasMetals[step] += galaxies[p].SfrDiskColdGasMetals[step];
        galaxies[p].SfrDisk[step] = 0.0;
        galaxies[p].SfrDiskColdGas[step] = 0.0;
        galaxies[p].SfrDiskColdGasMetals[step] = 0.0;
    }
}

// ============================================================================
// Starbursts
// ============================================================================

/*
 * collisional_starburst_recipe -- trigger an interaction-driven starburst.
 *
 * Called both during mergers (mode=1) and disk instabilities (mode=0).
 * Computes the burst SFR from the Somerville (2001) mass_ratio scaling, forms
 * stars into BulgeMass/MergerBulgeMass (mergers) or disk stars (instabilities),
 * applies SN feedback routing (FIRE or standard), and updates SFH arrays.
 */
void collisional_starburst_recipe(const double mass_ratio, const int merger_centralgal, const int centralgal,
                                  const double time, const double dt, const int halonr, const int mode, const int step,
                                  const int burst_to_merger_bulge, const double old_disk_radius,
                                  struct GALAXY *galaxies, const struct params *run_params)
{
    XASSERT(step >= 0 && step < STEPS, -1,
            "Error: step = %d is out of bounds [0, %d)\n", step, STEPS);
    XASSERT(dt > 0.0, -1,
            "Error: dt = %g must be > 0 for SFR calculation\n", dt);

    double stars, reheated_mass, ejected_mass, fac, metallicity, eburst, gas_for_starburst;

    // This is the major and minor merger starburst recipe of Somerville et al. 2001.
    // The coefficients in eburst are taken from TJ Cox's PhD thesis and should be more accurate then previous.

    // the bursting fraction
    if(mode == 1) {
        eburst = mass_ratio;
    } else {
        eburst = STARBURST_FRAC_COEFF * pow(mass_ratio, STARBURST_MASS_POWER);
    }

    if (run_params->StarburstColdGasOn == 0 &&
        (run_params->SFprescription == 1 || run_params->SFprescription == 3 || run_params->SFprescription == 4 ||
         run_params->SFprescription == 5 || run_params->SFprescription == 6 ||
         run_params->SFprescription == 7)) {
        // Recompute H2gas from the current ColdGas rather than using the stored value.
        // The stored H2gas was set during disk SF earlier in this timestep, but ColdGas has
        // since been depleted by SF, feedback, and satellite stripping, making the stored
        // value stale (often H2gas >> 0.74*ColdGas, and even > ColdGas at high-z).
        // Using the stale value + clamp would silently fall back to ColdGas for most events.
        const int cgal = merger_centralgal;
        double h2gas_fresh = 0.0;
        if(galaxies[cgal].ColdGas > 0.0 && galaxies[cgal].DiskScaleRadius > 0.0) {
            const float h     = run_params->Hubble_h;
            const float rs_pc = (float)(galaxies[cgal].DiskScaleRadius * 1.0e6 / h);
            if(rs_pc > 0.0f) {
                if(run_params->H2RadialIntegrationOn) {
                    // Radial integration stores result in galaxies[cgal].H2gas
                    calculate_molecular_fraction_radial_integration(cgal, galaxies, run_params, NULL);
                    h2gas_fresh = galaxies[cgal].H2gas;
                } else {
                    float disk_area_pc2;
                    if(run_params->H2DiskAreaOption == 0)
                        disk_area_pc2 = (float)M_PI * rs_pc * rs_pc;
                    else if(run_params->H2DiskAreaOption == 1)
                        disk_area_pc2 = (float)M_PI * 9.0f * rs_pc * rs_pc;
                    else
                        disk_area_pc2 = 2.0f * (float)M_PI * rs_pc * rs_pc;

                    const float Sigma_gas = (float)(galaxies[cgal].ColdGas * 1.0e10 / h) / disk_area_pc2;

                    if(run_params->SFprescription == 1 || run_params->SFprescription == 3) {
                        // BR06 / Somerville+H2
                        const float Sigma_star = (float)((galaxies[cgal].StellarMass - galaxies[cgal].BulgeMass)
                                                 * 1.0e10 / h) / disk_area_pc2;
                        h2gas_fresh = calculate_molecular_fraction_BR06(Sigma_gas, Sigma_star, rs_pc)
                                      * (galaxies[cgal].ColdGas * HYDROGEN_MASS_FRAC);
                    } else if(run_params->SFprescription == 4) {
                        // KD12
                        const float met = (float)((galaxies[cgal].ColdGas > 0.0) ?
                            galaxies[cgal].MetalsColdGas / galaxies[cgal].ColdGas : 0.0);
                        h2gas_fresh = calculate_H2_fraction_KD12(Sigma_gas, met, 5.0f)
                                      * (galaxies[cgal].ColdGas * HYDROGEN_MASS_FRAC);
                    } else if(run_params->SFprescription == 5) {
                        // KMT09
                        float met_abs = (float)((galaxies[cgal].ColdGas > 0.0) ?
                            galaxies[cgal].MetalsColdGas / galaxies[cgal].ColdGas : 0.0);
                        float Z_prime = (met_abs > 0.0f) ? met_abs / (float)Z_SOLAR_GD14 : 0.0f;
                        const float tau_c = 0.066f * 3.0f * Z_prime * Sigma_gas;
                        const float chi = 0.77f * (1.0f + 3.1f * powf(Z_prime, 0.365f));
                        const float s_kmt = (tau_c > 1e-10f) ?
                            logf(1.0f + 0.6f*chi + 0.01f*chi*chi) / (0.6f*tau_c) : 100.0f;
                        float f_H2 = (s_kmt < 2.0f) ? 1.0f - (3.0f*s_kmt)/(4.0f+s_kmt) : 0.0f;
                        if(f_H2 < 0.0f) f_H2 = 0.0f;
                        if(f_H2 > 1.0f) f_H2 = 1.0f;
                        h2gas_fresh = f_H2 * (galaxies[cgal].ColdGas * HYDROGEN_MASS_FRAC);
                    } else if(run_params->SFprescription == 6) {
                        // K13: two-phase molecular fraction
                        double Z_gas = (galaxies[cgal].ColdGas > 0.0) ?
                            galaxies[cgal].MetalsColdGas / galaxies[cgal].ColdGas : 0.0;
                        double Z_prime = Z_gas / Z_SOLAR_ASPLUND09;
                        if(Z_prime < 0.01) Z_prime = 0.01;
                        const double chi_2p = 3.1 * (1.0 + 3.1 * pow(Z_prime, 0.365)) / 4.1;
                        const double tau_c = 0.066 * 5.0 * Z_prime * (double)Sigma_gas;
                        const double s_k13 = (tau_c > 0.0) ?
                            log(1.0 + 0.6*chi_2p + 0.01*chi_2p*chi_2p) / (0.6*tau_c) : 100.0;
                        double f_H2_2p = (s_k13 < 2.0) ? 1.0 - (0.75*s_k13)/(1.0+0.25*s_k13) : 0.0;
                        if(f_H2_2p < 0.0) f_H2_2p = 0.0;
                        if(f_H2_2p > 1.0) f_H2_2p = 1.0;
                        h2gas_fresh = f_H2_2p * (galaxies[cgal].ColdGas * HYDROGEN_MASS_FRAC);
                    } else if(run_params->SFprescription == 7) {
                        // GD14
                        double met_abs = (galaxies[cgal].ColdGas > 0.0) ?
                            galaxies[cgal].MetalsColdGas / galaxies[cgal].ColdGas : 0.0;
                        double D_MW = met_abs / Z_SOLAR_GD14;
                        if(D_MW < 1e-4) D_MW = 1e-4;
                        const double S       = 3.0 * rs_pc / 100.0;
                        const double s_param = pow(GD14_S_PARAM_UMW1, 0.7);  // U_MW = 1.0
                        const double D_star  = 0.17 * (2.0 + pow(S, 5.0)) / (1.0 + pow(S, 5.0));
                        const double g       = sqrt(D_MW*D_MW + D_star*D_star);
                        const double Sigma_R1 = (g > 0.0) ? (40.0/g) * (s_param/(1.0+s_param)) : 1e10;
                        const double alpha_gd = 1.0 + 0.7 * sqrt(s_param) / (1.0 + s_param);
                        const double q = (Sigma_R1 > 0.0 && Sigma_gas > 0.0) ?
                            pow((double)Sigma_gas / Sigma_R1, alpha_gd) : 0.0;
                        double f_H2 = q / (1.0 + q);
                        if(f_H2 > 1.0) f_H2 = 1.0;
                        if(f_H2 < 0.0) f_H2 = 0.0;
                        h2gas_fresh = f_H2 * (galaxies[cgal].ColdGas * HYDROGEN_MASS_FRAC);
                    }
                }
            }
        }
        if(h2gas_fresh > galaxies[merger_centralgal].ColdGas * HYDROGEN_MASS_FRAC)
            h2gas_fresh = galaxies[merger_centralgal].ColdGas * HYDROGEN_MASS_FRAC;
        if(h2gas_fresh < 0.0)
            h2gas_fresh = 0.0;
        galaxies[merger_centralgal].H2gas = h2gas_fresh;
        gas_for_starburst = h2gas_fresh;
    } else {
        gas_for_starburst = galaxies[merger_centralgal].ColdGas;
    }
    if(gas_for_starburst < 0.0) gas_for_starburst = 0.0;

    stars = eburst * gas_for_starburst;
    if(stars < 0.0) {
        stars = 0.0;
    }
    
    // FIRE velocity/redshift scaling (Muratov et al. 2015) -- pre-computed once
    // and reused for both reheating and ejection.
    double fire_scaling = 0.0;
    if(run_params->FIREmodeOn == 1 && run_params->SupernovaRecipeOn == 1) {
        const double z_fire = run_params->ZZ[galaxies[merger_centralgal].SnapNum];
        const double vc_fire = galaxies[merger_centralgal].Vvir;
        if(vc_fire > 0.0 && z_fire >= 0.0) {
            const double vc_floored = (vc_fire < 1.0) ? 1.0 : vc_fire;
            const double v_term = (vc_floored < FIRE_V_CRIT_KMS)
                ? pow(vc_floored / FIRE_V_CRIT_KMS, -3.2)
                : pow(vc_floored / FIRE_V_CRIT_KMS, -1.0);
            fire_scaling = pow(1.0 + z_fire, run_params->RedshiftPowerLawExponent) * v_term;
        }
    }

    // this bursting results in SN feedback on the cold/hot gas
    if(run_params->SupernovaRecipeOn == 1) {
        if(run_params->FIREmodeOn == 1) {
            reheated_mass = run_params->FeedbackReheatingEpsilon * fire_scaling * stars;
        } else {
            reheated_mass = run_params->FeedbackReheatingEpsilon * stars;
        }
    } else {
        reheated_mass = 0.0;
    }

    XASSERT(reheated_mass >= 0.0, -1, "Error: Reheated mass = %g should be >= 0.0", reheated_mass);

    // can't use more gas than is available for the burst
    if((stars + reheated_mass) > gas_for_starburst) {
        fac = gas_for_starburst / (stars + reheated_mass);
        stars *= fac;
        reheated_mass *= fac;
    }

    // determine ejection
    if(run_params->SupernovaRecipeOn == 1) {
        if(galaxies[merger_centralgal].Vvir > 0.0) {
            if(run_params->FIREmodeOn == 1) {
                // FIRE energy-based ejection; fire_scaling pre-computed above
                const double vc = galaxies[merger_centralgal].Vvir;
                const double E_FB = run_params->FeedbackEjectionEfficiency * fire_scaling *
                                    0.5 * stars * (run_params->EtaSNcode * run_params->EnergySNcode);
                const double E_lift = 0.5 * reheated_mass * vc * vc;
                ejected_mass = (E_FB > E_lift) ? (E_FB - E_lift) / (0.5 * vc * vc) : 0.0;
            } else {
                ejected_mass =
                    (run_params->FeedbackEjectionEfficiency * (run_params->EtaSNcode * run_params->EnergySNcode) / 
                     (galaxies[merger_centralgal].Vvir * galaxies[merger_centralgal].Vvir) -
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

    // starbursts add to the bulge
    galaxies[merger_centralgal].SfrBulge[step] += stars / dt;
    galaxies[merger_centralgal].SfrBulgeColdGas[step] += galaxies[merger_centralgal].ColdGas;
    galaxies[merger_centralgal].SfrBulgeColdGasMetals[step] += galaxies[merger_centralgal].MetalsColdGas;

    metallicity = get_metallicity(galaxies[merger_centralgal].ColdGas, galaxies[merger_centralgal].MetalsColdGas);
    update_from_star_formation(merger_centralgal, stars, metallicity, galaxies, run_params);

    // Track star formation history for bulge starbursts
    if(run_params->SaveFullSFH) {
        const int snapnum = galaxies[merger_centralgal].SnapNum;
        if(snapnum >= 0 && snapnum < ABSOLUTEMAXSNAPS) {
            galaxies[merger_centralgal].SFHMassBulge[snapnum] += (1.0 - run_params->RecycleFraction) * stars;
        }
    }

    const double recycled_stars = (1 - run_params->RecycleFraction) * stars;
    
    galaxies[merger_centralgal].BulgeMass += recycled_stars;
    galaxies[merger_centralgal].MetalsBulgeMass += metallicity * recycled_stars;
    
    if(burst_to_merger_bulge) {
        // Add to merger-driven bulge
        galaxies[merger_centralgal].MergerBulgeMass += recycled_stars;
        // Radius will be recalculated in deal_with_galaxy_merger using energy conservation
    } else {
        // Add to instability-driven bulge
        galaxies[merger_centralgal].InstabilityBulgeMass += recycled_stars;
        // Update radius using Tonini equation (15)
        update_instability_bulge_radius(merger_centralgal, recycled_stars, old_disk_radius, 
                                       galaxies, run_params);
    }

    // recompute the metallicity of the cold phase
    metallicity = get_metallicity(galaxies[merger_centralgal].ColdGas, galaxies[merger_centralgal].MetalsColdGas);

    // update from feedback
    update_from_feedback(merger_centralgal, centralgal, reheated_mass, ejected_mass, metallicity, galaxies, run_params);

    // Clamp H2/H1 after gas has been consumed and ejected, so any chained merger
    // or disk-instability check that reads H2gas gets a physically consistent value.
    if (run_params->SFprescription == 1 || run_params->SFprescription == 3 ||
        run_params->SFprescription == 4 || run_params->SFprescription == 5 ||
        run_params->SFprescription == 6 || run_params->SFprescription == 7) {
        if(galaxies[merger_centralgal].H2gas > galaxies[merger_centralgal].ColdGas * HYDROGEN_MASS_FRAC)
            galaxies[merger_centralgal].H2gas = galaxies[merger_centralgal].ColdGas * HYDROGEN_MASS_FRAC;
        galaxies[merger_centralgal].H1gas = (galaxies[merger_centralgal].ColdGas * HYDROGEN_MASS_FRAC)
                                            - galaxies[merger_centralgal].H2gas;
        if(galaxies[merger_centralgal].H1gas < 0.0) galaxies[merger_centralgal].H1gas = 0.0;
    }

    // check for disk instability
    if(run_params->DiskInstabilityOn && mode == 0) {
        if(mass_ratio < run_params->ThreshMajorMerger) {
            check_disk_instability(merger_centralgal, centralgal, halonr, time, dt, step, galaxies, (struct params *) run_params);
        }
    }

    // formation of new metals - instantaneous recycling approximation - only SNII
    if(galaxies[merger_centralgal].ColdGas > 1e-8 && mass_ratio < run_params->ThreshMajorMerger) {
        // MINOR MERGER with sufficient cold gas: some metals stay in disk
        const double FracZleaveDiskVal = run_params->FracZleaveDisk * exp(-1.0 * galaxies[centralgal].Mvir / KD11_METAL_HALO_MASS);
        
        // Metals that stay in disk
        galaxies[merger_centralgal].MetalsColdGas += run_params->Yield * (1.0 - FracZleaveDiskVal) * stars;
        
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
        // MAJOR MERGER or very low cold gas: ALL metals leave disk
        // No functional disk left, so all metals go directly to CGM/HotGas
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

// ============================================================================
// Intracluster Stars (ICS) and Disruption
// ============================================================================

/*
 * disrupt_satellite_to_ICS -- disrupt satellite gal into the central's ICS
 * (intra-cluster stars) reservoir.
 *
 * Transfers all stellar mass, metals, and gas from the satellite to the
 * central's ICS and hot/CGM reservoirs.  Optionally tracks disruption time
 * and mass if TrackICSAssembly is set.
 */
void disrupt_satellite_to_ICS(const int centralgal, const int gal, const double time, struct GALAXY *galaxies, const struct params *run_params)
{
    // Transfer satellite's gas to central's hot/CGM reservoir (regime-dependent)
    const double total_gas = galaxies[gal].ColdGas + galaxies[gal].HotGas + galaxies[gal].CGMgas;
    const double total_metals_gas = galaxies[gal].MetalsColdGas + galaxies[gal].MetalsHotGas + galaxies[gal].MetalsCGMgas;
    
    if(run_params->CGMrecipeOn == 1) {
        if(galaxies[centralgal].Regime == 0) {
            // CGM-regime: disrupted gas goes to CGM
            galaxies[centralgal].CGMgas += total_gas;
            galaxies[centralgal].MetalsCGMgas += total_metals_gas;
        } else {
            // Hot-ICM-regime: disrupted gas goes to HotGas
            galaxies[centralgal].HotGas += total_gas;
            galaxies[centralgal].MetalsHotGas += total_metals_gas;
        }
    } else {
        // Original SAGE behavior: disrupted gas goes to HotGas
        galaxies[centralgal].HotGas += total_gas;
        galaxies[centralgal].MetalsHotGas += total_metals_gas;
    }

    // Transfer ejected mass (same for all regimes)
    galaxies[centralgal].EjectedMass += galaxies[gal].EjectedMass;
    galaxies[centralgal].MetalsEjectedMass += galaxies[gal].MetalsEjectedMass;

    // Transfer ICS (same for all regimes)
    galaxies[centralgal].ICS += galaxies[gal].ICS;
    galaxies[centralgal].MetalsICS += galaxies[gal].MetalsICS;

    // Track ICS assembly: pre-existing satellite ICS goes to ICS_accrete
    // This ICS was formed elsewhere (in the satellite's halo) and is being brought in
    if(run_params->TrackICSAssembly && galaxies[gal].ICS > 0.0) {
        galaxies[centralgal].ICS_accrete += galaxies[gal].ICS;
        // Inherit satellite's mass-weighted deposit-time accumulator so the
        // mean ICS-assembly time reflects when the stars were *originally* stripped,
        // not when this packet transferred into the central's reservoir.
        galaxies[centralgal].ICS_sum_mt += galaxies[gal].ICS_sum_mt;
    }

    // Disrupt stellar mass: split between ICS and BCG
    double frac_to_ICS;
    if(run_params->DynamicDisruptionSplit >= 1) {
        // Dynamic split based on halo mass ratio: f_ICL = 1 - (Msub/Mhost)^alpha_eff
        // Low mass-ratio satellites -> mostly ICL (disrupted on wide orbits)
        // High mass-ratio satellites -> more to BCG (deposited near centre)
        const double Msub = (double)galaxies[gal].infallMvir;
        const double Mhost = (double)galaxies[centralgal].Mvir;
        if(Msub > 0.0 && Mhost > 0.0) {
            double mass_ratio = Msub / Mhost;
            if(mass_ratio > 1.0) mass_ratio = 1.0;

            double alpha_eff = run_params->DisruptionSplitAlpha;
            if(run_params->DynamicDisruptionSplit == 2) {
                // Concentration-weighted: concentrated satellites resist stripping
                // alpha_eff = alpha_0 * (c_ref / c_sat)
                // High c_sat -> small alpha -> f_ICL closer to 0 -> more to BCG
                // Low c_sat  -> large alpha -> f_ICL closer to 1 -> more to ICL
                const double c_sat = (double)galaxies[gal].Concentration;
                if(c_sat > 0.0) {
                    alpha_eff *= run_params->DisruptionSplitCref / c_sat;
                }
            }

            frac_to_ICS = 1.0 - pow(mass_ratio, alpha_eff);
        } else {
            frac_to_ICS = run_params->FractionDisruptedToICS;  // fallback
        }
    } else {
        // Fixed fraction mode (original behavior)
        frac_to_ICS = run_params->FractionDisruptedToICS;
    }
    const double frac_to_BCG = 1.0 - frac_to_ICS;
    const double new_ICS_from_stripping = frac_to_ICS * galaxies[gal].StellarMass;

    galaxies[centralgal].ICS += new_ICS_from_stripping;
    galaxies[centralgal].MetalsICS += frac_to_ICS * galaxies[gal].MetalsStellarMass;
    
    // Track ICS assembly: newly disrupted stellar mass goes to ICS_disrupt
    if(run_params->TrackICSAssembly) {
        galaxies[centralgal].ICS_disrupt += new_ICS_from_stripping;
        // Record deposition time for the mass-weighted assembly-time accumulator
        galaxies[centralgal].ICS_sum_mt += new_ICS_from_stripping * time;
    }

    // Add remainder to BCG bulge (accreted onto outer envelope)
    galaxies[centralgal].StellarMass += frac_to_BCG * galaxies[gal].StellarMass;
    galaxies[centralgal].MetalsStellarMass += frac_to_BCG * galaxies[gal].MetalsStellarMass;
    galaxies[centralgal].BulgeMass += frac_to_BCG * galaxies[gal].StellarMass;
    galaxies[centralgal].MetalsBulgeMass += frac_to_BCG * galaxies[gal].MetalsStellarMass;
    galaxies[centralgal].MergerBulgeMass += frac_to_BCG * galaxies[gal].StellarMass;  // Track as merger-driven
    get_bulge_radius(centralgal, galaxies, run_params);

    // Transfer star formation history from disrupted satellite to central
    // - Fraction going to BCG bulge: track in SFHMassBulge (stellar ages)
    // Note: For ICS stellar ages, we would need SFHMassICS, but that's been replaced
    // by ICS_disrupt/ICS_accrete which track assembly times, not stellar ages
    if(run_params->SaveFullSFH) {
        for(int snap = 0; snap < ABSOLUTEMAXSNAPS; snap++) {
            const double sat_sfh = galaxies[gal].SFHMassDisk[snap] + galaxies[gal].SFHMassBulge[snap];
            galaxies[centralgal].SFHMassBulge[snap] += frac_to_BCG * sat_sfh;
        }
    }

    // Transfer black hole mass to central (avoid baryons disappearing)
    galaxies[centralgal].BlackHoleMass += galaxies[gal].BlackHoleMass;

    // Zero all satellite baryonic fields after transfer -- defensive cleanup so
    // no downstream code can accidentally recount baryons from a merged galaxy.
    galaxies[gal].ColdGas         = galaxies[gal].MetalsColdGas     = 0.0f;
    galaxies[gal].HotGas          = galaxies[gal].MetalsHotGas      = 0.0f;
    galaxies[gal].CGMgas          = galaxies[gal].MetalsCGMgas      = 0.0f;
    galaxies[gal].EjectedMass     = galaxies[gal].MetalsEjectedMass = 0.0f;
    galaxies[gal].ICS             = galaxies[gal].MetalsICS         = 0.0f;
    galaxies[gal].StellarMass     = galaxies[gal].MetalsStellarMass = 0.0f;
    galaxies[gal].BulgeMass       = galaxies[gal].MetalsBulgeMass   = 0.0f;
    galaxies[gal].BlackHoleMass   = 0.0f;

    galaxies[gal].mergeType = 4;  // mark as disruption to the ICS
}

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
 * disruption into the ICS (disrupt_satellite_to_ICS, plus the gated
 * Contini et al. 2014 variant disrupt_satellite_gated and the Henriques &
 * Thomas 2010 continuous stripping strip_orphan_stars, both selected by
 * DisruptionGate).
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
#include "model_halo_properties.h"

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

/* Solar metallicity (Grevesse & Sauval 1998): used in KMT09 Z' normalisation.
 * Z' = metallicity / Z_SOLAR_GD14.
 * Also the canonical definition in model_misc.c (calculate_H2_fraction_GD14). */
static const double Z_SOLAR_GD14 = 0.02;

/* Contini et al. (2014) tidal radius, their eq. 5 (after Binney & Tremaine 2008):
 * R_t = (M_sat / (CONTINI14_TIDAL_DENOM * M_DM,halo))^(1/3) * D. */
static const double CONTINI14_TIDAL_DENOM = 3.0;

/* Contini et al. (2014) Sec. 3.2: the stellar disk is truncated at
 * R_sat = CONTINI14_DISK_TRUNC_FRAC * R_sl, which encloses 99.9 per cent of an
 * exponential disk.  After a stripping episode the scalelength is reset to
 * R_t / CONTINI14_DISK_TRUNC_FRAC. */
static const double CONTINI14_DISK_TRUNC_FRAC = 10.0;

/* Henriques & Thomas (2010) eq. 4: R_t = (1/sqrt(2)) (sigma_sat/sigma_halo) r_sat.
 * For an isothermal sphere sigma = Vvir/sqrt(2) on both sides of the ratio, so
 * only this leading factor remains once the ratio is written in Vvir. */
static const double HT10_TIDAL_COEFF = M_SQRT1_2;

/* Henriques & Thomas (2010) eq. 8: the bulge mass profile M(<r) = M r^2/(r^2+a^2)
 * uses a scale radius a = HT10_BULGE_SCALE_FRAC * R_b, with R_b the half-mass
 * (effective) bulge radius. */
static const double HT10_BULGE_SCALE_FRAC = 0.56;

/* Bisection controls for the pericentre solution of Contini et al. (2014) eq. 3.
 * The equation is monotonic above its single minimum, so a fixed iteration count
 * converges to well below the precision the surrounding physics needs. */
static const int CONTINI14_PERI_ITERATIONS = 60;
static const double CONTINI14_PERI_MAX_RATIO = 1.0e6;

/* File-private forward declaration */
static double calculate_merger_remnant_radius(const struct GALAXY *g1, const struct GALAXY *g2);
static double contini14_pericentre(const double D, const double v_total, const double v_tang, const double Vvir);
static double contini14_half_mass_radius(const int gal, const struct GALAXY *galaxies);

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

    // Track ICS assembly: a merging satellite's ICS was formed in its own halo,
    // so it enters the central's reservoir through the accreted (ex-situ) channel,
    // exactly as in infall_recipe() and disrupt_satellite_to_ICS().  Without this
    // the central's ICS grows while ICS_disrupt + ICS_accrete does not, breaking
    // the accounting identity, and the satellite's deposit-time history is lost.
    // In practice infall_recipe() sweeps satellite ICS to the central at the top
    // of every snapshot, so galaxies[p].ICS is almost always 0 here -- but it is
    // not guaranteed to be, since a satellite can acquire ICS mid-snapshot by
    // hosting a disruption of its own before merging.
    if(run_params->TrackICSAssembly && galaxies[p].ICS > 0.0) {
        galaxies[t].ICS_accrete += galaxies[p].ICS;
        // Inherit the mass-weighted deposit-time accumulator so the mean assembly
        // time reflects when these stars were originally stripped, not when the
        // packet transferred into the central.
        galaxies[t].ICS_sum_mt += galaxies[p].ICS_sum_mt;
    }

    galaxies[t].ICS += galaxies[p].ICS;
    galaxies[t].MetalsICS += galaxies[p].MetalsICS;

    // The assembly history now belongs to the central; clear it on the satellite
    // so no later pass can count it twice.
    galaxies[p].ICS_disrupt = galaxies[p].ICS_accrete = galaxies[p].ICS_sum_mt = 0.0;

    galaxies[t].BlackHoleMass += galaxies[p].BlackHoleMass;

    galaxies[t].CGMgas += galaxies[p].CGMgas;
    galaxies[t].MetalsCGMgas += galaxies[p].MetalsCGMgas;

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
        sf_prescription_tracks_h2(run_params->SFprescription)) {
        // Recompute H2gas from the current ColdGas rather than using the stored value.
        // The stored H2gas was set during disk SF earlier in this timestep, but ColdGas has
        // since been depleted by SF, feedback, and satellite stripping, making the stored
        // value stale (often H2gas >> 0.74*ColdGas, and even > ColdGas at high-z).
        // Using the stale value + clamp would silently fall back to ColdGas for most events.
        const int cgal = merger_centralgal;
        double h2gas_fresh = 0.0;
        if(galaxies[cgal].ColdGas > 0.0 && galaxies[cgal].DiskScaleRadius > 0.0) {
            const float h     = run_params->Hubble_h;  /* float on purpose: frozen single-precision behaviour, do not promote (see docs/physics/units.md) */
            const float rs_pc = (float)(CODE_LENGTH_TO_PC(galaxies[cgal].DiskScaleRadius, h));
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

                    const float Sigma_gas = (float)(CODE_MASS_TO_MSUN(galaxies[cgal].ColdGas, h)) / disk_area_pc2;

                    if(sf_prescription_is_br06(run_params->SFprescription)) {
                        // BR06 / Somerville+H2
                        const float Sigma_star = (float)(CODE_MASS_TO_MSUN(galaxies[cgal].StellarMass - galaxies[cgal].BulgeMass, h)) / disk_area_pc2;
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
                        const double Z_gas = (galaxies[cgal].ColdGas > 0.0) ?
                            galaxies[cgal].MetalsColdGas / galaxies[cgal].ColdGas : 0.0;
                        const double f_H2_2p = calculate_H2_fraction_K13((double)Sigma_gas, Z_gas, 5.0);
                        h2gas_fresh = f_H2_2p * (galaxies[cgal].ColdGas * HYDROGEN_MASS_FRAC);
                    } else if(run_params->SFprescription == 7) {
                        // GD14
                        const double met_abs = (galaxies[cgal].ColdGas > 0.0) ?
                            galaxies[cgal].MetalsColdGas / galaxies[cgal].ColdGas : 0.0;
                        const double f_H2 = calculate_H2_fraction_GD14((double)Sigma_gas, met_abs, (double)rs_pc);
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
            reheated_mass = capped_eta_reheat(
                run_params->FeedbackReheatingEpsilon * fire_scaling,
                galaxies[merger_centralgal].Vvir, run_params) * stars;
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
                const double E_FB = sn_energy_coupling(fire_scaling, run_params) *
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
    // Note: this quick refresh skips the HIIonizationOn cut (H1 here is the full
    // atomic remainder); the next SF substep recomputes H1 with the ionisation
    // correction applied.
    if (sf_prescription_tracks_h2(run_params->SFprescription)) {
        if(galaxies[merger_centralgal].H2gas > galaxies[merger_centralgal].ColdGas * HYDROGEN_MASS_FRAC)
            galaxies[merger_centralgal].H2gas = galaxies[merger_centralgal].ColdGas * HYDROGEN_MASS_FRAC;
        galaxies[merger_centralgal].H1gas = (galaxies[merger_centralgal].ColdGas * HYDROGEN_MASS_FRAC)
                                            - galaxies[merger_centralgal].H2gas;
        if(galaxies[merger_centralgal].H1gas < 0.0) galaxies[merger_centralgal].H1gas = 0.0;
    }

    // check for disk instability
    if(run_params->DiskInstabilityOn && mode == 0) {
        if(mass_ratio < run_params->ThreshMajorMerger) {
            check_disk_instability(merger_centralgal, centralgal, halonr, time, dt, step, galaxies, run_params);
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
        
        add_metals_to_hot_reservoir(&galaxies[centralgal], run_params, metals_leaving_disk);
    } else {
        // MAJOR MERGER or very low cold gas: ALL metals leave disk
        // No functional disk left, so all metals go directly to CGM/HotGas
        const double all_metals = run_params->Yield * stars;
        
        add_metals_to_hot_reservoir(&galaxies[centralgal], run_params, all_metals);
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
    
    add_gas_to_hot_reservoir(&galaxies[centralgal], run_params, total_gas, total_metals_gas);

    // Transfer ejected mass (same for all regimes)
    galaxies[centralgal].EjectedMass += galaxies[gal].EjectedMass;
    galaxies[centralgal].MetalsEjectedMass += galaxies[gal].MetalsEjectedMass;

    // Transfer ICS (same for all regimes)
    galaxies[centralgal].ICS += galaxies[gal].ICS;
    galaxies[centralgal].MetalsICS += galaxies[gal].MetalsICS;

    // Transfer satellite's stellar mass to central's ICS (intra-cluster stars)
    galaxies[centralgal].ICS += galaxies[gal].StellarMass;
    galaxies[centralgal].MetalsICS += galaxies[gal].MetalsStellarMass;

    // Track ICS assembly: newly disrupted stellar mass goes to ICS_disrupt.
    // These stars become unbound here and now, so `time` (the lookback time of
    // this event) is the correct deposit time for the m*t accumulator -- unlike
    // the accreted channel above, which inherits the satellite's own history.
    if(run_params->TrackICSAssembly && galaxies[gal].StellarMass > 0.0) {
        galaxies[centralgal].ICS_disrupt += galaxies[gal].StellarMass;
        galaxies[centralgal].ICS_sum_mt += galaxies[gal].StellarMass * time;
    }

    // Transfer black hole mass to central (avoid baryons disappearing)
    galaxies[centralgal].BlackHoleMass += galaxies[gal].BlackHoleMass;

    // Track ICS assembly: pre-existing satellite ICS goes to the ex-situ channel.
    // This ICS was formed elsewhere (by disruption in the satellite's own halo) and
    // is only being carried in here -- so ICS_accrete records where a packet came
    // from, not how it was made.
    if(run_params->TrackICSAssembly && galaxies[gal].ICS > 0.0) {
        galaxies[centralgal].ICS_accrete += galaxies[gal].ICS;
        // Inherit satellite's mass-weighted deposit-time accumulator so the
        // mean ICS-assembly time reflects when the stars were *originally* stripped,
        // not when this packet transferred into the central's reservoir.
        galaxies[centralgal].ICS_sum_mt += galaxies[gal].ICS_sum_mt;
    }

    // Zero all satellite baryonic fields after transfer -- defensive cleanup so
    // no downstream code can accidentally recount baryons from a merged galaxy.
    galaxies[gal].ColdGas         = galaxies[gal].MetalsColdGas     = 0.0f;
    galaxies[gal].HotGas          = galaxies[gal].MetalsHotGas      = 0.0f;
    galaxies[gal].CGMgas          = galaxies[gal].MetalsCGMgas      = 0.0f;
    galaxies[gal].EjectedMass     = galaxies[gal].MetalsEjectedMass = 0.0f;
    galaxies[gal].ICS             = galaxies[gal].MetalsICS         = 0.0f;
    galaxies[gal].ICS_disrupt     = galaxies[gal].ICS_accrete       = 0.0f;
    galaxies[gal].ICS_sum_mt      = 0.0f;
    galaxies[gal].StellarMass     = galaxies[gal].MetalsStellarMass = 0.0f;
    galaxies[gal].BulgeMass       = galaxies[gal].MetalsBulgeMass   = 0.0f;
    galaxies[gal].BlackHoleMass   = 0.0f;

    galaxies[gal].mergeType = 4;  // mark as disruption to the ICS
}

// ============================================================================
// Contini et al. (2014) disruption gate -- DisruptionGate == 1
// ============================================================================

/*
 * contini14_pericentre -- pericentric distance of a satellite orbit in a
 * singular isothermal halo potential.
 *
 * Physical setup:
 *   The parent halo is represented by the singular isothermal potential of
 *   Contini et al. (2014) eq. 2, phi(R) = Vvir^2 ln R.  Conserving energy and
 *   angular momentum between the satellite's current position and its
 *   pericentre gives their eq. 3,
 *
 *     (R / R_peri)^2 = [ ln(R / R_peri) + 0.5 (V / Vvir)^2 ]
 *                      / [ 0.5 (V_t / Vvir)^2 ],
 *
 *   where V is the satellite speed relative to the halo centre and V_t its
 *   tangential part.
 *
 * Algorithm:
 *   1. Substitute u = R / R_peri (u >= 1) and rearrange eq. 3 into the root
 *      form f(u) = b u^2 - ln u - a, with a = 0.5 (V/Vvir)^2 and
 *      b = 0.5 (V_t/Vvir)^2.
 *   2. f(1) = b - a <= 0 because V_t <= V, and f grows without bound, so a
 *      single root u >= 1 exists whenever b > 0.  Bracket it by doubling.
 *   3. Bisect for a fixed number of iterations and return R / u.
 *
 * Inputs:
 *   D        -- satellite distance from the halo centre, code units (Mpc/h).
 *   v_total  -- speed relative to the halo centre, code units (km/s).
 *   v_tang   -- tangential part of that velocity, code units (km/s).
 *   Vvir     -- virial velocity of the parent halo, code units (km/s).
 *
 * Returns:
 *   The pericentric distance in code units.  Returns 0.0 for a purely radial
 *   orbit (v_tang == 0), which has no turning point and plunges to the centre.
 *
 * References:
 *   - Contini et al. (2014), MNRAS 437, 3787, Sec. 3.1, eqs. 2-3.
 *
 * Invariants:
 *   - The returned radius never exceeds D (a pericentre is a minimum).
 */
static double contini14_pericentre(const double D, const double v_total, const double v_tang, const double Vvir)
{
    if(D <= 0.0 || Vvir <= 0.0 || v_tang <= 0.0) {
        return 0.0;
    }

    const double a = 0.5 * (v_total / Vvir) * (v_total / Vvir);
    const double b = 0.5 * (v_tang / Vvir) * (v_tang / Vvir);

    /* Step 1 and 2: bracket the root of f(u) = b u^2 - ln u - a above u = 1. */
    double u_lo = 1.0;
    double u_hi = 2.0;
    while(b * u_hi * u_hi - log(u_hi) - a < 0.0) {
        u_hi *= 2.0;
        if(u_hi > CONTINI14_PERI_MAX_RATIO) {
            /* The orbit is so radial that the pericentre is unresolved here;
             * treat it as a plunge to the halo centre. */
            return 0.0;
        }
    }

    /* Step 3: bisect. */
    for(int i = 0; i < CONTINI14_PERI_ITERATIONS; i++) {
        const double u_mid = 0.5 * (u_lo + u_hi);
        if(b * u_mid * u_mid - log(u_mid) - a < 0.0) {
            u_lo = u_mid;
        } else {
            u_hi = u_mid;
        }
    }

    return D / (0.5 * (u_lo + u_hi));
}

/*
 * contini14_half_mass_radius -- baryonic half-mass radius of a satellite.
 *
 * Contini et al. (2014) Sec. 3.1 approximate the satellite half-mass radius by
 * the mass-weighted average of the half-mass radius of the disk and that of
 * the bulge.  The disk half-mass radius of an exponential profile is
 * DISK_HALF_MASS_FRAC * R_sl; the bulge radius stored by get_bulge_radius() is
 * already a half-mass radius.
 *
 * The weighting mirrors calculate_merger_remnant_radius() above, which forms
 * the same mass-weighted average elsewhere in SAGE26: the disk carries the cold
 * gas as well as the non-bulge stars, and the normalisation is the total
 * baryonic mass.  That also matches the paper, whose M_sat in eq. 4 is the
 * baryonic mass rather than the stellar mass alone.
 *
 * Inputs:
 *   gal      -- index of the satellite.  Reads StellarMass, BulgeMass,
 *               DiskScaleRadius and BulgeRadius.
 *
 * Returns:
 *   The half-mass radius in code units (Mpc/h), or 0.0 when the galaxy has no
 *   stars or no size information, which the caller treats as "cannot evaluate
 *   the gate".
 *
 * References:
 *   - Contini et al. (2014), MNRAS 437, 3787, Sec. 3.1.
 */
static double contini14_half_mass_radius(const int gal, const struct GALAXY *galaxies)
{
    double disk_mass = galaxies[gal].ColdGas + (galaxies[gal].StellarMass - galaxies[gal].BulgeMass);
    if(disk_mass < 0.0) {
        disk_mass = 0.0;
    }
    const double bulge_mass = galaxies[gal].BulgeMass;

    const double disk_half = DISK_HALF_MASS_FRAC * galaxies[gal].DiskScaleRadius;
    const double bulge_half = galaxies[gal].BulgeRadius;

    double weight = 0.0;
    double sum = 0.0;
    if(disk_mass > 0.0 && disk_half > 0.0) {
        sum += disk_mass * disk_half;
        weight += disk_mass;
    }
    if(bulge_mass > 0.0 && bulge_half > 0.0) {
        sum += bulge_mass * bulge_half;
        weight += bulge_mass;
    }

    if(weight <= 0.0) {
        return 0.0;
    }

    return sum / weight;
}

/*
 * disrupt_satellite_gated -- Contini et al. (2014) survival gate and partial
 * tidal stripping for an orphan satellite.
 *
 * Physical setup:
 *   SAGE's default behaviour destroys an orphan outright the moment its
 *   subhalo leaves the merger tree.  Contini et al. (2014) instead require the
 *   parent halo to be dense enough at the satellite's pericentre to unbind it.
 *   Their model Disr. (Sec. 3.1, following Guo et al. 2011) compares the halo
 *   density at pericentre with the mean baryon density of the satellite inside
 *   its half-mass radius, and only destroys satellites that lose that
 *   comparison.  Satellites that survive keep their reservoirs and carry on to
 *   the next snapshot, where the test is repeated; the dynamical-friction
 *   clock continues to run, so a survivor eventually merges instead.
 *
 *   When the gate does open, the material removed is set by the tidal radius
 *   of their Sec. 3.2 (eqs. 5-6) rather than by wholesale destruction: only the
 *   stellar disk outside R_t is unbound, and total destruction is reserved for
 *   satellites whose tidal radius has cut inside the bulge.
 *
 * Algorithm:
 *   1. Evaluate the satellite's separation D and relative velocity from the
 *      parent halo centre, using the minimum-image convention for the periodic
 *      box.
 *   2. Solve eq. 3 for the pericentre and form the halo density there,
 *      rho_halo = M_DM,halo(R_peri) / R_peri^3, with M(<R) = Vvir^2 R / G for
 *      the isothermal potential of eq. 2.
 *   3. Form the satellite density rho_sat = M_sat / R_half^3 with
 *      M_sat = StellarMass + ColdGas (eq. 4).
 *   4. If rho_halo <= rho_sat the satellite survives untouched: return 0.
 *   5. Otherwise compute R_t from eq. 5.  If R_t is inside the bulge radius
 *      the satellite is completely disrupted (Sec. 3.2); hand over to
 *      disrupt_satellite_to_ICS() and return 1.
 *   6. If R_t lies inside the truncation radius R_sat = 10 R_sl, strip the
 *      exponential-disk mass in the shell R_t to R_sat into the central's ICS,
 *      move the same fraction of the cold gas to the central's hot phase, and
 *      reset the scalelength to R_t / 10.  The satellite survives: return 0.
 *
 * Inputs:
 *   centralgal       -- central galaxy of the parent FOF halo.  Supplies the
 *                       potential (Mvir, Vvir) and the reference position.
 *   merger_centralgal-- galaxy that receives the stripped material.
 *   gal              -- the orphan being tested.
 *   time             -- lookback time of this substep, code units, recorded in
 *                       the ICS assembly accumulator.
 *
 * Returns:
 *   1 if the satellite was completely disrupted (mergeType is now set and the
 *   caller must not touch it again), 0 if it survived this substep.
 *
 * References:
 *   - Contini et al. (2014), MNRAS 437, 3787, Sec. 3.1 (eqs. 2-4) and
 *     Sec. 3.2 (eqs. 5-6).
 *   - Guo et al. (2011), MNRAS 413, 101 -- the disruption criterion adopted by
 *     Contini et al. as their model Disr.
 *
 * Invariants:
 *   - Stripped mass never exceeds the stellar disk mass.
 *   - A satellite with no baryons, or with no usable size information, is
 *     passed to disrupt_satellite_to_ICS() so that empty orphans cannot
 *     survive indefinitely.
 */
int disrupt_satellite_gated(const int centralgal, const int merger_centralgal, const int gal,
                            const double time, struct GALAXY *galaxies, const struct params *run_params)
{
    const double sat_mass = galaxies[gal].StellarMass + galaxies[gal].ColdGas;
    const double half_mass_radius = contini14_half_mass_radius(gal, galaxies);

    /* Degenerate satellites cannot be tested; fall back to the ungated
     * behaviour so they are removed rather than left orbiting forever. */
    if(sat_mass <= 0.0 || half_mass_radius <= 0.0 ||
       galaxies[centralgal].Vvir <= 0.0 || galaxies[centralgal].Mvir <= 0.0) {
        disrupt_satellite_to_ICS(merger_centralgal, gal, time, galaxies, run_params);
        return 1;
    }

    /* Step 1: separation and relative velocity from the parent halo centre.
     * Halo positions are periodic, so a FOF group straddling a box face would
     * otherwise report a separation of order the box size. */
    double dx[3];
    double dv[3];
    double D_sq = 0.0;
    for(int j = 0; j < 3; j++) {
        double d = galaxies[gal].Pos[j] - galaxies[centralgal].Pos[j];
        if(d > 0.5 * run_params->BoxSize) {
            d -= run_params->BoxSize;
        } else if(d < -0.5 * run_params->BoxSize) {
            d += run_params->BoxSize;
        }
        dx[j] = d;
        dv[j] = galaxies[gal].Vel[j] - galaxies[centralgal].Vel[j];
        D_sq += d * d;
    }
    const double D = sqrt(D_sq);

    if(D <= 0.0) {
        /* Sitting on the halo centre: nothing survives there. */
        disrupt_satellite_to_ICS(merger_centralgal, gal, time, galaxies, run_params);
        return 1;
    }

    double v_sq = 0.0;
    double v_radial = 0.0;
    for(int j = 0; j < 3; j++) {
        v_sq += dv[j] * dv[j];
        v_radial += dv[j] * dx[j] / D;
    }
    const double v_total = sqrt(v_sq);
    double v_tang_sq = v_sq - v_radial * v_radial;
    if(v_tang_sq < 0.0) {
        v_tang_sq = 0.0;
    }
    const double v_tang = sqrt(v_tang_sq);

    /* Step 2: halo density at pericentre.  For phi(R) = Vvir^2 ln R the
     * enclosed mass is M(<R) = Vvir^2 R / G, so M(<R_peri) / R_peri^3 reduces
     * to Vvir^2 / (G R_peri^2). */
    const double r_peri = contini14_pericentre(D, v_total, v_tang, galaxies[centralgal].Vvir);
    if(r_peri <= 0.0) {
        /* Radial plunge: the density at pericentre diverges and the gate is
         * always open. */
        disrupt_satellite_to_ICS(merger_centralgal, gal, time, galaxies, run_params);
        return 1;
    }
    const double rho_halo = galaxies[centralgal].Vvir * galaxies[centralgal].Vvir /
                            (run_params->G * r_peri * r_peri);

    /* Step 3: mean baryon density of the satellite inside its half-mass radius. */
    const double rho_sat = sat_mass / (half_mass_radius * half_mass_radius * half_mass_radius);

    /* Step 4: the gate.  A satellite denser than its surroundings at pericentre
     * is not disrupted, and lives to be tested again next substep. */
    if(rho_halo <= rho_sat) {
        return 0;
    }

    /* Step 5: tidal radius, and complete disruption once it cuts into the bulge. */
    const double R_t = cbrt(sat_mass / (CONTINI14_TIDAL_DENOM * galaxies[centralgal].Mvir)) * D;

    if(R_t < galaxies[gal].BulgeRadius) {
        disrupt_satellite_to_ICS(merger_centralgal, gal, time, galaxies, run_params);
        return 1;
    }

    /* Step 6: strip the exponential stellar disk outside R_t. */
    const double r_scale = galaxies[gal].DiskScaleRadius;
    double disk_mass = galaxies[gal].StellarMass - galaxies[gal].BulgeMass;
    if(disk_mass < 0.0) {
        disk_mass = 0.0;
    }

    if(r_scale <= 0.0 || disk_mass <= 0.0) {
        return 0;
    }

    const double R_sat = CONTINI14_DISK_TRUNC_FRAC * r_scale;
    if(R_t >= R_sat) {
        return 0;
    }

    /* Enclosed mass of an exponential disk outside radius R is
     * M (1 + R/R_sl) exp(-R/R_sl); the shell between R_t and the truncation
     * radius is the difference of the two. */
    const double x_t = R_t / r_scale;
    const double x_sat = R_sat / r_scale;
    double stripped = disk_mass * ((1.0 + x_t) * exp(-x_t) - (1.0 + x_sat) * exp(-x_sat));

    if(stripped <= 0.0) {
        return 0;
    }
    if(stripped > disk_mass) {
        stripped = disk_mass;
    }

    /* Stellar metals follow the disk, whose metal mass is the stellar total
     * less what is locked in the bulge. */
    double disk_metals = galaxies[gal].MetalsStellarMass - galaxies[gal].MetalsBulgeMass;
    if(disk_metals < 0.0) {
        disk_metals = 0.0;
    }
    const double metallicity = get_metallicity(disk_mass, disk_metals);
    double stripped_metals = stripped * metallicity;
    if(stripped_metals > disk_metals) {
        stripped_metals = disk_metals;
    }

    galaxies[gal].StellarMass -= stripped;
    galaxies[gal].MetalsStellarMass -= stripped_metals;

    galaxies[merger_centralgal].ICS += stripped;
    galaxies[merger_centralgal].MetalsICS += stripped_metals;

    if(run_params->TrackICSAssembly) {
        galaxies[merger_centralgal].ICS_disrupt += stripped;
        galaxies[merger_centralgal].ICS_sum_mt += stripped * time;
    }

    /* Contini et al. (2014) Sec. 3.2: a proportional fraction of the cold gas
     * follows the stripped stars into the central's hot phase. */
    const double strip_fraction = stripped / disk_mass;
    const double cold_stripped = strip_fraction * galaxies[gal].ColdGas;
    const double cold_metals_stripped = strip_fraction * galaxies[gal].MetalsColdGas;
    if(cold_stripped > 0.0) {
        galaxies[gal].ColdGas -= cold_stripped;
        galaxies[gal].MetalsColdGas -= cold_metals_stripped;
        add_gas_to_hot_reservoir(&galaxies[merger_centralgal], run_params, cold_stripped, cold_metals_stripped);
    }

    /* Contini et al. re-describe the truncated disk with a scalelength of
     * R_t / 10 so that it is again truncated at ten scalelengths. That step is
     * deliberately not taken here: SAGE26 uses DiskScaleRadius directly as the
     * exponential scale length of the surface density that drives the H2 and
     * star formation prescriptions (Sigma_0 = M / (2 pi r_s^2) in
     * model_h2_chemistry.c), so compressing r_s tenfold *raises* the central
     * surface density of a galaxy that has just lost its outskirts, and sends
     * its star formation rate up rather than down. Leaving r_s alone keeps the
     * retained material on the profile it already had inside R_t, and lets the
     * reduced mass lower Sigma_0 as it should. */

    return 0;
}

/*
 * strip_orphan_stars -- continuous tidal stripping of an orphan satellite.
 *
 * Physical setup:
 *   Henriques & Thomas (2010) strip stellar material from orphans on every
 *   timestep rather than only at a disruption event.  Material lying outside
 *   the satellite's tidal radius is unbound and joins the intracluster
 *   component.  Assuming an isothermal profile for both the satellite and the
 *   parent halo, and a circular orbit, their eq. 4 gives
 *
 *     R_t = (1 / sqrt(2)) * (sigma_sat / sigma_halo) * r_sat,
 *
 *   where r_sat is the current halocentric radius of the decaying orbit.  For
 *   an isothermal sphere sigma = Vvir / sqrt(2), so the dispersion ratio is
 *   just the ratio of virial velocities and the sqrt(2) factors cancel.
 *
 *   The mass outside R_t is evaluated per component.  The disk is exponential,
 *   so from their eq. 6 the mass beyond R_t is M_disk (1 + x) exp(-x) with
 *   x = R_t / R_sl.  The bulge follows their eq. 9, M(<r) = M r^2 / (r^2 + a^2)
 *   with a = HT10_BULGE_SCALE_FRAC * R_b, leaving M a^2 / (R_t^2 + a^2)
 *   outside.  R_b is the half-mass bulge radius, which SAGE26 already models
 *   via get_bulge_radius(); Henriques & Thomas instead recover it from a
 *   Djorgovski & Davis (1987) relation because their base model had no bulge
 *   sizes at all.
 *
 *   The paper assumes a uniform metallicity distribution, so stars and metals
 *   are stripped in equal fractions.
 *
 * Algorithm:
 *   1. Bail out unless the orbit and both virial velocities are usable.
 *   2. Form R_t from eq. 4.
 *   3. Accumulate the disk mass beyond R_t (eq. 6) and the bulge mass beyond
 *      R_t (eq. 9).
 *   4. Move that mass, and the same fraction of the stellar metals, into the
 *      central's ICS; shrink the bulge components in proportion.
 *
 * Inputs:
 *   centralgal -- central galaxy of the parent FOF halo, supplying sigma_halo.
 *   icsgal     -- galaxy whose ICS reservoir receives the stripped stars.
 *   gal        -- the orphan being stripped.
 *   time       -- lookback time of this substep, code units, recorded in the
 *                 ICS assembly accumulator.
 *
 * References:
 *   - Henriques & Thomas (2010), MNRAS 403, 768, Sec. 2.2, eqs. 4, 6 and 9.
 *
 * Invariants:
 *   - Stripped mass never exceeds the satellite's stellar mass, and neither
 *     StellarMass nor BulgeMass is driven negative.
 *   - Nothing is stripped when R_t is large enough to enclose the galaxy.
 */
void strip_orphan_stars(const int centralgal, const int icsgal, const int gal,
                        const double time, struct GALAXY *galaxies, const struct params *run_params)
{
    /* Step 1: the orbit has to be resolved and both velocity dispersions known. */
    if(galaxies[gal].OrbitRadius <= 0.0 || galaxies[gal].Vvir <= 0.0 ||
       galaxies[centralgal].Vvir <= 0.0 || galaxies[gal].StellarMass <= 0.0) {
        return;
    }

    /* Step 2: tidal radius.  sigma = Vvir / sqrt(2) for an isothermal sphere in
     * both numerator and denominator, so only the velocity ratio survives. */
    const double R_t = HT10_TIDAL_COEFF * (galaxies[gal].Vvir / galaxies[centralgal].Vvir)
                       * galaxies[gal].OrbitRadius;

    /* Step 3: mass beyond R_t, component by component. */
    double disk_mass = galaxies[gal].StellarMass - galaxies[gal].BulgeMass;
    if(disk_mass < 0.0) {
        disk_mass = 0.0;
    }
    const double bulge_mass = galaxies[gal].BulgeMass;

    double stripped_disk = 0.0;
    if(disk_mass > 0.0 && galaxies[gal].DiskScaleRadius > 0.0) {
        const double x = R_t / galaxies[gal].DiskScaleRadius;
        stripped_disk = disk_mass * (1.0 + x) * exp(-x);
    }

    double stripped_bulge = 0.0;
    if(bulge_mass > 0.0 && galaxies[gal].BulgeRadius > 0.0) {
        const double a = HT10_BULGE_SCALE_FRAC * galaxies[gal].BulgeRadius;
        stripped_bulge = bulge_mass * a * a / (R_t * R_t + a * a);
    }

    if(stripped_disk > disk_mass) {
        stripped_disk = disk_mass;
    }
    if(stripped_bulge > bulge_mass) {
        stripped_bulge = bulge_mass;
    }

    const double stripped = stripped_disk + stripped_bulge;
    if(stripped <= 0.0) {
        return;
    }

    /* Step 4: hand the unbound stars to the central's ICS.  Uniform metallicity
     * means metals leave in the same proportion as stars. */
    const double metallicity = get_metallicity(galaxies[gal].StellarMass, galaxies[gal].MetalsStellarMass);
    double stripped_metals = stripped * metallicity;
    if(stripped_metals > galaxies[gal].MetalsStellarMass) {
        stripped_metals = galaxies[gal].MetalsStellarMass;
    }

    galaxies[gal].StellarMass -= stripped;
    galaxies[gal].MetalsStellarMass -= stripped_metals;

    if(stripped_bulge > 0.0) {
        /* Keep the two Tonini bulge components consistent with the total. */
        const double remaining_frac = (bulge_mass - stripped_bulge) / bulge_mass;
        const double bulge_metals = galaxies[gal].MetalsBulgeMass * (1.0 - remaining_frac);

        galaxies[gal].BulgeMass -= stripped_bulge;
        galaxies[gal].MetalsBulgeMass -= bulge_metals;
        galaxies[gal].MergerBulgeMass *= remaining_frac;
        galaxies[gal].InstabilityBulgeMass *= remaining_frac;
    }

    galaxies[icsgal].ICS += stripped;
    galaxies[icsgal].MetalsICS += stripped_metals;

    if(run_params->TrackICSAssembly) {
        galaxies[icsgal].ICS_disrupt += stripped;
        galaxies[icsgal].ICS_sum_mt += stripped * time;
    }
}

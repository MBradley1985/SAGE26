#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "core_allvars.h"

#include "model_misc.h"

void init_galaxy(const int p, const int halonr, int *galaxycounter, const struct halo_data *halos,
                 struct GALAXY *galaxies, const struct params *run_params)
{

	XASSERT(halonr == halos[halonr].FirstHaloInFOFgroup, -1,
            "Error: halonr = %d should be equal to the FirsthaloInFOFgroup = %d\n",
            halonr, halos[halonr].FirstHaloInFOFgroup);

    galaxies[p].Type = 0;
    galaxies[p].Regime = -1;
    galaxies[p].FFBRegime = 0;
    galaxies[p].Concentration = 0.0;

    galaxies[p].GalaxyNr = *galaxycounter;
    (*galaxycounter)++;

    galaxies[p].HaloNr = halonr;
    galaxies[p].MostBoundID = halos[halonr].MostBoundID;
    galaxies[p].SnapNum = halos[halonr].SnapNum - 1;

    galaxies[p].mergeType = 0;
    galaxies[p].mergeIntoID = -1;
    galaxies[p].mergeIntoSnapNum = -1;
    galaxies[p].dT = -1.0;

    for(int j = 0; j < 3; j++) {
        galaxies[p].Pos[j] = halos[halonr].Pos[j];
        galaxies[p].Vel[j] = halos[halonr].Vel[j];
    }

    galaxies[p].Len = halos[halonr].Len;
    galaxies[p].Vmax = halos[halonr].Vmax;
    galaxies[p].Vvir = get_virial_velocity(halonr, halos, run_params);
    galaxies[p].Mvir = get_virial_mass(halonr, halos, run_params);
    galaxies[p].Rvir = get_virial_radius(halonr, halos, run_params);

    galaxies[p].deltaMvir = 0.0;

    galaxies[p].ColdGas = 0.0;
    galaxies[p].StellarMass = 0.0;
    galaxies[p].BulgeMass = 0.0;
    galaxies[p].MergerBulgeMass = 0.0;   
    galaxies[p].InstabilityBulgeMass = 0.0; 
    galaxies[p].HotGas = 0.0;
    galaxies[p].EjectedMass = 0.0;
    galaxies[p].BlackHoleMass = 0.0;
    
    galaxies[p].ICS = 0.0;
    galaxies[p].CGMgas = 0.0;
    galaxies[p].H2gas = 0.0;
    galaxies[p].H1gas = 0.0;

    galaxies[p].MetalsColdGas = 0.0;
    galaxies[p].MetalsStellarMass = 0.0;
    galaxies[p].MetalsBulgeMass = 0.0;
    galaxies[p].MetalsHotGas = 0.0;
    galaxies[p].MetalsEjectedMass = 0.0;
    galaxies[p].MetalsICS = 0.0;
    galaxies[p].MetalsCGMgas = 0.0;

    for(int step = 0; step < STEPS; step++) {
        galaxies[p].SfrDisk[step] = 0.0;
        galaxies[p].SfrBulge[step] = 0.0;
        galaxies[p].SfrDiskColdGas[step] = 0.0;
        galaxies[p].SfrDiskColdGasMetals[step] = 0.0;
        galaxies[p].SfrBulgeColdGas[step] = 0.0;
        galaxies[p].SfrBulgeColdGasMetals[step] = 0.0;
    }

    // Initialize star formation history arrays (tracks mass formed at each snapshot)
    // Only need to initialize if SaveFullSFH is enabled, otherwise these arrays are unused
    if(run_params->SaveFullSFH) {
        for(int snap = 0; snap < ABSOLUTEMAXSNAPS; snap++) {
            galaxies[p].SFHMassDisk[snap] = 0.0;
            galaxies[p].SFHMassBulge[snap] = 0.0;
        }
    }
    // Initialize ICS assembly tracking (cumulative mass through each channel)
    galaxies[p].ICS_disrupt = 0.0;
    galaxies[p].ICS_accrete = 0.0;

    galaxies[p].DiskScaleRadius = get_disk_radius(halonr, p, halos, galaxies);
    galaxies[p].BulgeRadius = get_bulge_radius(p, galaxies, run_params);
    galaxies[p].MergerBulgeRadius = get_bulge_radius(p, galaxies, run_params);
    galaxies[p].InstabilityBulgeRadius = get_bulge_radius(p, galaxies, run_params);
    galaxies[p].MergTime = 999.9f;
    galaxies[p].Cooling = 0.0;
    galaxies[p].Heating = 0.0;
    galaxies[p].r_heat = 0.0;
    galaxies[p].QuasarModeBHaccretionMass = 0.0;
    galaxies[p].TimeOfLastMajorMerger = -1.0;
    galaxies[p].TimeOfLastMinorMerger = -1.0;
    galaxies[p].OutflowRate = 0.0;
	galaxies[p].TotalSatelliteBaryons = 0.0;
    galaxies[p].RcoolToRvir = -1.0;
    galaxies[p].MassLoading = 0.0;
    galaxies[p].tcool = -1.0;
    galaxies[p].tff = -1.0;
    galaxies[p].tcool_over_tff = -1.0;
    galaxies[p].tdeplete = -1.0;

	// infall properties
    galaxies[p].infallMvir = -1.0;
    galaxies[p].infallVvir = -1.0;
    galaxies[p].infallVmax = -1.0;
    galaxies[p].infallStellarMass = -1.0;
    galaxies[p].TimeOfInfall = -1.0;

    galaxies[p].mdot_cool = 0.0;
    galaxies[p].mdot_stream = 0.0;

    galaxies[p].g_max = 0.0;


}



double get_disk_radius(const int halonr, const int p, const struct halo_data *halos, const struct GALAXY *galaxies)
{
	if(galaxies[p].Vvir > 0.0 && galaxies[p].Rvir > 0.0) {
		// See Mo, Shude & White (1998) eq12, and using a Bullock style lambda.
		double SpinMagnitude = sqrt(halos[halonr].Spin[0] * halos[halonr].Spin[0] +
                                    halos[halonr].Spin[1] * halos[halonr].Spin[1] + halos[halonr].Spin[2] * halos[halonr].Spin[2]);

		double SpinParameter = SpinMagnitude / ( 1.414 * galaxies[p].Vvir * galaxies[p].Rvir);
		return (SpinParameter / 1.414 ) * galaxies[p].Rvir;
        /* return SpinMagnitude * 0.5 / galaxies[p].Vvir; /\* should be equivalent to previous call *\/ */
	} else {
		return 0.1 * galaxies[p].Rvir;
    }
}


double get_bulge_radius(const int p, struct GALAXY *galaxies, const struct params *run_params)
{
    // BulgeSizeOn == 0: No bulge size calculation
    if(run_params->BulgeSizeOn == 0) {
        galaxies[p].BulgeRadius = 0.0;
        galaxies[p].MergerBulgeRadius = 0.0;
        galaxies[p].InstabilityBulgeRadius = 0.0;
        return 0.0;
    }
    
    const double h = run_params->Hubble_h;
    
    // BulgeSizeOn == 1: Shen equation 33 (simple power-law)
    if(run_params->BulgeSizeOn == 1) {
        if(galaxies[p].BulgeMass <= 0.0) {
            galaxies[p].BulgeRadius = 0.0;
            galaxies[p].MergerBulgeRadius = 0.0;
            galaxies[p].InstabilityBulgeRadius = 0.0;
            return 0.0;
        }
        
        // Convert bulge mass from 10^10 M_sun/h to M_sun
        const double M_bulge_sun = galaxies[p].BulgeMass * 1.0e10 / h;
        
        // Shen+2003 equation (33): log(R/kpc) = 0.56 log(M/Msun) - 5.54
        const double log_R_kpc = 0.56 * log10(M_bulge_sun) - 5.54;
        double R_bulge_kpc = pow(10.0, log_R_kpc);
        
        // Convert to code units (Mpc/h)
        const double R_bulge = R_bulge_kpc * 1.0e-3 * h;
        
        galaxies[p].BulgeRadius = R_bulge;
        galaxies[p].MergerBulgeRadius = 0.0;
        galaxies[p].InstabilityBulgeRadius = 0.0;
        
        return R_bulge;
    }
    
    // BulgeSizeOn == 2: Shen equation 32 (two-regime power-law)
    if(run_params->BulgeSizeOn == 2) {
        if(galaxies[p].BulgeMass <= 0.0) {
            galaxies[p].BulgeRadius = 0.0;
            galaxies[p].MergerBulgeRadius = 0.0;
            galaxies[p].InstabilityBulgeRadius = 0.0;
            return 0.0;
        }
        
        // Convert bulge mass from 10^10 M_sun/h to M_sun
        const double M_bulge_sun = galaxies[p].BulgeMass * 1.0e10 / h;
        
        // Transition mass from Shen et al. (2003) equation (32)
        const double M_transition = 2.0e10;  // M_sun
        
        double R_bulge_kpc;
        
        if(M_bulge_sun > M_transition) {
            // High-mass regime: like giant ellipticals
            // log(R/kpc) = 0.56 log(M) - 5.54
            const double log_R = 0.56 * log10(M_bulge_sun) - 5.54;
            R_bulge_kpc = pow(10.0, log_R);
        } else {
            // Low-mass regime: like dwarf ellipticals  
            // log(R/kpc) = 0.14 log(M) - 1.21
            const double log_R = 0.14 * log10(M_bulge_sun) - 1.21;
            R_bulge_kpc = pow(10.0, log_R);
        }
        
        // Convert to code units (Mpc/h)
        const double R_bulge = R_bulge_kpc * 1.0e-3 * h;
        
        galaxies[p].BulgeRadius = R_bulge;
        galaxies[p].MergerBulgeRadius = 0.0;
        galaxies[p].InstabilityBulgeRadius = 0.0;
        
        return R_bulge;
    }
    
    // BulgeSizeOn == 3: Tonini setup (separate merger and instability bulges)
    if(run_params->BulgeSizeOn == 3) {
        const double M_merger = galaxies[p].MergerBulgeMass;
        const double M_instability = galaxies[p].InstabilityBulgeMass;
        const double M_total = M_merger + M_instability;
        
        if(M_total <= 0.0) {
            return 0.0;
        }
        
        // 1. Retrieve the Merger Radius
        // This is now calculated in model_mergers.c via Energy Conservation
        // and stored persistently.
        double R_merger = galaxies[p].MergerBulgeRadius;
        
        // Failsafe: If mass exists but radius is 0 (e.g. initialization), use Shen as fallback
        if(M_merger > 0.0 && R_merger <= 0.0) {
             const double M_merger_sun = M_merger * 1.0e10 / h;
             const double log_R_kpc = 0.56 * log10(M_merger_sun) - 5.54;
             R_merger = pow(10.0, log_R_kpc) * 1.0e-3 * h;
             // Store it so we don't recalculate
             galaxies[p].MergerBulgeRadius = R_merger;
        }

        // 2. Retrieve Instability Radius (Already correct in your code)
        double R_instability = galaxies[p].InstabilityBulgeRadius;
        
        // Failsafe for instability radius
        if(M_instability > 0.0 && R_instability <= 0.0) {
            const double R_disc = galaxies[p].DiskScaleRadius;
            R_instability = 0.2 * R_disc;
            galaxies[p].InstabilityBulgeRadius = R_instability;
        }
        
        // 3. Weighted Average (Equation 25)
        double R_bulge = (M_merger * R_merger + M_instability * R_instability) / M_total;
        
        galaxies[p].BulgeRadius = R_bulge;
        return R_bulge;
    }
    
    // Default fallback (should not reach here)
    galaxies[p].BulgeRadius = 0.0;
    galaxies[p].MergerBulgeRadius = 0.0;
    galaxies[p].InstabilityBulgeRadius = 0.0;
    return 0.0;
}


void update_instability_bulge_radius(const int p, const double delta_mass, 
                                     const double old_disk_radius,
                                     struct GALAXY *galaxies, const struct params *run_params)
{
    // Tonini+2016 equation (15): incremental radius evolution
    // R_i = (R_i,OLD * M_i,OLD + δM * 0.2 * R_D) / (M_i,OLD + δM)
    //
    // IMPORTANT: old_disk_radius should be the disc radius BEFORE the instability event
    // This ensures we use the correct R_D value as prescribed in the paper
    
    if(run_params->BulgeSizeOn != 3) return;  // Only for Tonini mode
    if(delta_mass <= 0.0) return;
    
    const double h = run_params->Hubble_h;
    const double M_old = galaxies[p].InstabilityBulgeMass - delta_mass;  // Mass before addition
    const double R_old = galaxies[p].InstabilityBulgeRadius;
    
    // Use the OLD disc radius (pre-instability) passed as parameter
    // Convert to kpc for calculation
    const double R_disc_kpc = old_disk_radius * 1.0e3 / h;
    
    // New mass contribution scales with 0.2 * R_disc (Tonini+2016 eq. 15)
    const double R_new_contribution_kpc = 0.2 * R_disc_kpc;
    const double R_new_contribution = R_new_contribution_kpc * 1.0e-3 * h;  // to Mpc/h
    
    double R_new;
    if(M_old > 0.0 && R_old > 0.0) {
        // Incremental update (equation 15)
        const double R_old_kpc = R_old * 1.0e3 / h;
        const double M_new = galaxies[p].InstabilityBulgeMass;
        const double R_new_kpc = (R_old_kpc * M_old + R_new_contribution_kpc * delta_mass) / M_new;
        R_new = R_new_kpc * 1.0e-3 * h;
    } else {
        // First mass addition: initialize with 0.2 * R_disc
        R_new = R_new_contribution;
    }
    
    galaxies[p].InstabilityBulgeRadius = R_new;
}


double get_metallicity(const double gas, const double metals)
{
  double metallicity = 0.0;

  if(gas > 0.0 && metals > 0.0) {
      metallicity = metals / gas;
      metallicity = metallicity >= 1.0 ? 1.0:metallicity;
  }

  return metallicity;
}



double dmax(const double x, const double y)
{
    return (x > y) ? x:y;
}



double get_virial_mass(const int halonr, const struct halo_data *halos, const struct params *run_params)
{
  if(halonr == halos[halonr].FirstHaloInFOFgroup && halos[halonr].Mvir >= 0.0)
    return halos[halonr].Mvir;   /* take spherical overdensity mass estimate */
  else
    return halos[halonr].Len * run_params->PartMass;
}



double get_virial_velocity(const int halonr, const struct halo_data *halos, const struct params *run_params)
{
	double Rvir;

	Rvir = get_virial_radius(halonr, halos, run_params);

    if(Rvir > 0.0)
		return sqrt(run_params->G * get_virial_mass(halonr, halos, run_params) / Rvir);
	else
		return 0.0;
}


double get_virial_radius(const int halonr, const struct halo_data *halos, const struct params *run_params)
{
  // return halos[halonr].Rvir;  // Used for Bolshoi
  const int snapnum = halos[halonr].SnapNum;
  const double zplus1 = 1.0 + run_params->ZZ[snapnum];
  const double hubble_of_z_sq =
      run_params->Hubble * run_params->Hubble *(run_params->Omega * zplus1 * zplus1 * zplus1 + (1.0 - run_params->Omega - run_params->OmegaLambda) * zplus1 * zplus1 +
                                              run_params->OmegaLambda);

  const double rhocrit = 3.0 * hubble_of_z_sq / (8.0 * M_PI * run_params->G);
  const double fac = 1.0 / (200.0 * 4.0 * M_PI / 3.0 * rhocrit);

  return cbrt(get_virial_mass(halonr, halos, run_params) * fac);
}

void determine_and_store_regime(const int ngal, struct GALAXY *galaxies,
                                const struct params *run_params)
{
    for(int p = 0; p < ngal; p++) {
        if(galaxies[p].mergeType > 0) continue;

        // Convert Mvir to physical units (Msun)
        // Mvir is stored in units of 10^10 Msun/h
        const double Mvir_physical = galaxies[p].Mvir * 1.0e10 / run_params->Hubble_h;

        // Shock mass threshold (Dekel & Birnboim 2006)
        const double Mshock = 6.0e11;  // Msun

        // Calculate mass ratio for sigmoid
        const double mass_ratio = Mvir_physical / Mshock;

        // BUG FIX: Protect against log10(0) or log10(negative)
        if(mass_ratio <= 0.0) {
            galaxies[p].Regime = 0;  // Default to CGM regime for invalid mass
            continue;
        }

        // Smooth sigmoid transition (consistent with FFB approach)
        // Width of transition in dex
        const double delta_log_M = 0.1;

        // Sigmoid argument: x = log10(M/Mshock) / width
        const double x = log10(mass_ratio) / delta_log_M;

        // Sigmoid function: probability of being in Hot regime
        // Smoothly varies from 0 (well below Mshock) to 1 (well above Mshock)
        const double hot_fraction = 1.0 / (1.0 + exp(-x));

        // Probabilistic assignment based on sigmoid
        const double random_uniform = (double)rand() / (double)RAND_MAX;

        galaxies[p].Regime = (random_uniform < hot_fraction) ? 1 : 0;

    }
}

void determine_and_store_ffb_regime(const int ngal, const double Zcurr, struct GALAXY *galaxies,
                                     const struct params *run_params)
{
    // Only apply FFB if the mode is enabled
    if(run_params->FeedbackFreeModeOn == 0) {
        // FFB mode disabled - mark all galaxies as normal
        for(int p = 0; p < ngal; p++) {
            galaxies[p].FFBRegime = 0;
        }
        return;
    }

    // Classify each galaxy as FFB or normal
    for(int p = 0; p < ngal; p++) {
        if(galaxies[p].mergeType > 0) continue;

        // Only CGM regime halos can be FFB, so we check that first
        if(galaxies[p].Regime == 1) {
            galaxies[p].FFBRegime = 0;  // Normal halo - in hot CGM regime, not eligible for FFB
            continue;
        }

        if(run_params->FeedbackFreeModeOn == 1) {
            // Li et al. 2024 mass-based method (original)
            const double Mvir = galaxies[p].Mvir;

            // Calculate smooth FFB fraction using sigmoid transition (Li et al. 2024, eq. 3)
            const double f_ffb = calculate_ffb_fraction(Mvir, Zcurr, run_params);

            // Probabilistic assignment based on smooth sigmoid function
            const double random_uniform = (double)rand() / (double)RAND_MAX;

            if(random_uniform < f_ffb) {
                galaxies[p].FFBRegime = 1;  // FFB halo
            } else {
                galaxies[p].FFBRegime = 0;  // Normal halo
            }
        } else if(run_params->FeedbackFreeModeOn == 2) {
            // Boylan-Kolchin 2025 acceleration-based method
            // FFB regime when g_max > g_crit (sharp cutoff)
            // Uses Ishiyama+21 c-M relation for concentration (lookup table)
            const double g_max = calculate_gmax_BK25(p, galaxies, run_params);

            // Store g_max for analysis (optional)
            galaxies[p].g_max = g_max;

            // g_crit/G = 3100 M_sun/pc^2 (Boylan-Kolchin 2025, Table 1)
            // g_crit = G * 3100 * M_sun / pc^2  (~4.3e-10 m/s^2)
            // In code units: g_crit = run_params->G * 3100 * (M_sun/UnitMass) / (pc/UnitLength)^2
            const double Msun_code = SOLAR_MASS / run_params->UnitMass_in_g;
            const double pc_code = 3.08568e18 / run_params->UnitLength_in_cm;  // 1 pc in cm
            const double g_crit = run_params->G * 3100.0 * Msun_code / (pc_code * pc_code);

            if(g_max > g_crit) {
                galaxies[p].FFBRegime = 1;  // FFB halo - above critical acceleration
            } else {
                galaxies[p].FFBRegime = 0;  // Normal halo
            }
        }
    }
}

double interpolate_concentration_ishiyama21(const double logM, const double z)
{
    // Ishiyama+21 concentration-mass lookup table (mdef=vir, halo_sample=all, c_type=fit)
    // Generated by Colossus (colossus_cm_comparison.py) with planck18 cosmology
    // Rows: log10(Mvir / [Msun/h]), Columns: redshift
    #define CM_TABLE_N_MASS 41
    #define CM_TABLE_N_Z 31
    const double cm_table_logmass[CM_TABLE_N_MASS] = {8.0, 8.2, 8.4, 8.6, 8.8, 9.0, 9.2, 9.4, 9.6, 9.8, 10.0, 10.2, 10.4, 10.6, 10.8, 11.0, 11.2, 11.4, 11.6, 11.8, 12.0, 12.2, 12.4, 12.6, 12.8, 13.0, 13.2, 13.4, 13.6, 13.8, 14.0, 14.2, 14.4, 14.6, 14.8, 15.0, 15.2, 15.4, 15.6, 15.8, 16.0};
    const double cm_table_z[CM_TABLE_N_Z] = {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0};
    const double cm_table[CM_TABLE_N_MASS][CM_TABLE_N_Z] = {
        {25.593, 18.494, 13.809, 10.784, 8.756, 7.338, 6.307, 5.536, 4.945, 4.484, 4.119, 3.828, 3.593, 3.404, 3.251, 3.127, 3.027, 2.947, 2.885, 2.837, 2.802, 2.777, 2.762, 2.755, 2.756, 2.763, 2.775, 2.793, 2.816, 2.843, 2.874},
        {24.790, 17.905, 13.365, 10.436, 8.476, 7.108, 6.114, 5.373, 4.806, 4.365, 4.018, 3.741, 3.520, 3.342, 3.199, 3.085, 2.994, 2.923, 2.868, 2.828, 2.799, 2.781, 2.773, 2.772, 2.778, 2.791, 2.810, 2.834, 2.862, 2.894, 2.931},
        {23.993, 17.321, 12.925, 10.093, 8.200, 6.880, 5.925, 5.213, 4.670, 4.250, 3.920, 3.658, 3.450, 3.284, 3.152, 3.047, 2.966, 2.903, 2.856, 2.823, 2.802, 2.791, 2.789, 2.795, 2.808, 2.827, 2.851, 2.881, 2.915, 2.953, 2.995},
        {23.201, 16.741, 12.489, 9.753, 7.927, 6.656, 5.738, 5.056, 4.538, 4.138, 3.825, 3.579, 3.384, 3.230, 3.109, 3.015, 2.942, 2.888, 2.850, 2.824, 2.811, 2.807, 2.812, 2.824, 2.844, 2.869, 2.900, 2.936, 2.976, 3.020, 3.068},
        {22.414, 16.165, 12.056, 9.416, 7.657, 6.436, 5.555, 4.903, 4.410, 4.030, 3.735, 3.504, 3.323, 3.181, 3.071, 2.987, 2.924, 2.879, 2.849, 2.832, 2.826, 2.829, 2.842, 2.861, 2.888, 2.920, 2.957, 2.999, 3.046, 3.097, 3.151},
        {21.636, 15.596, 11.630, 9.085, 7.392, 6.220, 5.377, 4.755, 4.286, 3.927, 3.649, 3.434, 3.266, 3.137, 3.038, 2.964, 2.911, 2.875, 2.854, 2.846, 2.848, 2.859, 2.879, 2.906, 2.940, 2.979, 3.024, 3.073, 3.126, 3.184, 3.245},
        {20.870, 15.037, 11.211, 8.760, 7.134, 6.010, 5.204, 4.612, 4.167, 3.829, 3.569, 3.369, 3.215, 3.099, 3.012, 2.949, 2.905, 2.879, 2.867, 2.867, 2.878, 2.898, 2.925, 2.960, 3.001, 3.048, 3.100, 3.157, 3.218, 3.282, 3.350},
        {20.117, 14.488, 10.801, 8.443, 6.881, 5.806, 5.037, 4.474, 4.054, 3.736, 3.494, 3.310, 3.171, 3.067, 2.992, 2.940, 2.907, 2.891, 2.888, 2.897, 2.917, 2.945, 2.981, 3.024, 3.074, 3.129, 3.189, 3.253, 3.321, 3.394, 3.470},
        {19.374, 13.947, 10.397, 8.131, 6.635, 5.607, 4.876, 4.342, 3.947, 3.650, 3.426, 3.258, 3.133, 3.042, 2.979, 2.938, 2.917, 2.910, 2.918, 2.937, 2.965, 3.003, 3.048, 3.100, 3.158, 3.221, 3.290, 3.362, 3.439, 3.520, 3.604},
        {18.642, 13.415, 10.001, 7.826, 6.395, 5.414, 4.720, 4.216, 3.845, 3.569, 3.364, 3.212, 3.101, 3.024, 2.974, 2.945, 2.935, 2.939, 2.957, 2.986, 3.025, 3.072, 3.126, 3.187, 3.255, 3.327, 3.405, 3.486, 3.572, 3.662, 3.755},
        {17.919, 12.889, 9.610, 7.527, 6.160, 5.227, 4.569, 4.096, 3.750, 3.495, 3.308, 3.173, 3.078, 3.015, 2.977, 2.961, 2.962, 2.978, 3.007, 3.047, 3.096, 3.153, 3.218, 3.289, 3.366, 3.448, 3.536, 3.627, 3.722, 3.821, 3.924},
        {17.203, 12.371, 9.226, 7.234, 5.931, 5.046, 4.425, 3.982, 3.661, 3.428, 3.260, 3.142, 3.062, 3.014, 2.990, 2.987, 3.000, 3.029, 3.069, 3.120, 3.180, 3.248, 3.324, 3.406, 3.494, 3.587, 3.684, 3.786, 3.892, 4.001, 4.114},
        {16.495, 11.859, 8.848, 6.947, 5.707, 4.870, 4.287, 3.874, 3.579, 3.368, 3.220, 3.119, 3.056, 3.023, 3.013, 3.024, 3.051, 3.091, 3.144, 3.207, 3.280, 3.360, 3.447, 3.540, 3.640, 3.744, 3.853, 3.966, 4.083, 4.204, 4.329},
        {15.794, 11.353, 8.476, 6.665, 5.490, 4.701, 4.156, 3.774, 3.504, 3.316, 3.188, 3.106, 3.060, 3.042, 3.048, 3.073, 3.114, 3.168, 3.234, 3.311, 3.396, 3.489, 3.589, 3.695, 3.806, 3.923, 4.045, 4.170, 4.300, 4.433, 4.570},
        {15.100, 10.854, 8.111, 6.390, 5.279, 4.539, 4.032, 3.681, 3.438, 3.273, 3.166, 3.103, 3.075, 3.074, 3.096, 3.136, 3.192, 3.261, 3.342, 3.432, 3.531, 3.638, 3.752, 3.871, 3.997, 4.127, 4.262, 4.401, 4.544, 4.691, 4.841},
        {14.418, 10.364, 7.753, 6.123, 5.077, 4.385, 3.916, 3.598, 3.382, 3.241, 3.156, 3.113, 3.103, 3.120, 3.159, 3.216, 3.288, 3.373, 3.469, 3.574, 3.689, 3.810, 3.939, 4.074, 4.214, 4.359, 4.509, 4.663, 4.821, 4.983, 5.148},
        {13.750, 9.888, 7.407, 5.866, 4.884, 4.241, 3.812, 3.525, 3.337, 3.221, 3.159, 3.137, 3.147, 3.183, 3.240, 3.314, 3.403, 3.505, 3.618, 3.741, 3.871, 4.010, 4.155, 4.306, 4.462, 4.624, 4.790, 4.961, 5.135, 5.313, 5.494},
        {13.098, 9.423, 7.073, 5.620, 4.702, 4.108, 3.718, 3.464, 3.305, 3.215, 3.176, 3.176, 3.207, 3.263, 3.340, 3.434, 3.542, 3.662, 3.793, 3.934, 4.083, 4.239, 4.402, 4.572, 4.746, 4.926, 5.110, 5.299, 5.491, 5.687, 5.887},
        {12.461, 8.971, 6.749, 5.385, 4.531, 3.986, 3.636, 3.416, 3.287, 3.224, 3.210, 3.234, 3.287, 3.365, 3.463, 3.577, 3.706, 3.847, 3.998, 4.159, 4.328, 4.504, 4.687, 4.876, 5.071, 5.271, 5.475, 5.684, 5.896, 6.113, 6.333},
        {11.836, 8.531, 6.436, 5.160, 4.371, 3.876, 3.567, 3.383, 3.285, 3.250, 3.263, 3.312, 3.389, 3.491, 3.612, 3.749, 3.900, 4.063, 4.237, 4.420, 4.611, 4.810, 5.015, 5.226, 5.443, 5.665, 5.892, 6.123, 6.358, 6.597, 6.839},
        {11.225, 8.103, 6.135, 4.948, 4.224, 3.780, 3.513, 3.365, 3.300, 3.296, 3.337, 3.413, 3.517, 3.644, 3.791, 3.953, 4.129, 4.317, 4.515, 4.723, 4.939, 5.162, 5.392, 5.628, 5.870, 6.117, 6.369, 6.625, 6.885, 7.149, 7.417},
        {10.627, 7.686, 5.846, 4.748, 4.090, 3.698, 3.475, 3.366, 3.336, 3.364, 3.435, 3.541, 3.674, 3.830, 4.005, 4.195, 4.398, 4.614, 4.840, 5.075, 5.319, 5.570, 5.827, 6.091, 6.361, 6.636, 6.916, 7.200, 7.489, 7.781, 8.078},
        {10.042, 7.283, 5.570, 4.561, 3.971, 3.633, 3.456, 3.388, 3.395, 3.458, 3.563, 3.701, 3.867, 4.054, 4.259, 4.481, 4.715, 4.962, 5.219, 5.485, 5.760, 6.042, 6.331, 6.626, 6.928, 7.234, 7.546, 7.862, 8.182, 8.507, 8.836},
        {9.472, 6.894, 5.308, 4.391, 3.869, 3.588, 3.459, 3.434, 3.481, 3.582, 3.724, 3.899, 4.100, 4.322, 4.562, 4.819, 5.088, 5.369, 5.661, 5.963, 6.273, 6.590, 6.914, 7.246, 7.583, 7.925, 8.273, 8.625, 8.982, 9.343, 9.709},
        {8.925, 6.525, 5.066, 4.240, 3.789, 3.566, 3.489, 3.510, 3.601, 3.744, 3.927, 4.141, 4.382, 4.643, 4.923, 5.219, 5.528, 5.849, 6.181, 6.522, 6.872, 7.229, 7.594, 7.966, 8.344, 8.727, 9.116, 9.509, 9.908, 10.311, 10.718},
        {8.403, 6.178, 4.846, 4.112, 3.734, 3.572, 3.549, 3.620, 3.759, 3.949, 4.177, 4.437, 4.722, 5.029, 5.353, 5.693, 6.047, 6.413, 6.791, 7.177, 7.573, 7.976, 8.388, 8.806, 9.231, 9.661, 10.097, 10.538, 10.984, 11.435, 11.891},
        {7.905, 5.854, 4.649, 4.009, 3.707, 3.610, 3.646, 3.772, 3.964, 4.206, 4.486, 4.796, 5.132, 5.490, 5.865, 6.257, 6.662, 7.080, 7.509, 7.948, 8.397, 8.853, 9.318, 9.790, 10.268, 10.753, 11.244, 11.739, 12.240, 12.747, 13.258},
        {7.433, 5.555, 4.477, 3.935, 3.712, 3.685, 3.785, 3.973, 4.225, 4.525, 4.863, 5.232, 5.626, 6.042, 6.476, 6.927, 7.391, 7.869, 8.359, 8.859, 9.368, 9.886, 10.412, 10.946, 11.487, 12.035, 12.589, 13.148, 13.713, 14.283, 14.859},
        {6.987, 5.282, 4.334, 3.893, 3.755, 3.804, 3.976, 4.233, 4.552, 4.919, 5.324, 5.759, 6.221, 6.704, 7.207, 7.726, 8.260, 8.807, 9.367, 9.937, 10.518, 11.107, 11.705, 12.312, 12.926, 13.547, 14.175, 14.808, 15.447, 16.093, 16.743},
        {6.569, 5.039, 4.224, 3.889, 3.843, 3.977, 4.228, 4.564, 4.960, 5.404, 5.887, 6.400, 6.940, 7.502, 8.085, 8.684, 9.299, 9.928, 10.570, 11.223, 11.886, 12.560, 13.243, 13.934, 14.634, 15.341, 16.055, 16.775, 17.502, 18.236, 18.975},
        {6.184, 4.829, 4.153, 3.932, 3.987, 4.215, 4.558, 4.983, 5.469, 6.003, 6.576, 7.180, 7.812, 8.467, 9.143, 9.837, 10.547, 11.272, 12.011, 12.762, 13.524, 14.297, 15.079, 15.871, 16.672, 17.481, 18.297, 19.120, 19.951, 20.788, 21.632},
        {5.836, 4.660, 4.129, 4.031, 4.198, 4.533, 4.981, 5.511, 6.102, 6.742, 7.421, 8.134, 8.874, 9.639, 10.427, 11.233, 12.056, 12.895, 13.749, 14.617, 15.496, 16.386, 17.287, 18.199, 19.120, 20.050, 20.988, 21.934, 22.887, 23.848, 24.816},
        {5.532, 4.540, 4.163, 4.201, 4.494, 4.953, 5.523, 6.176, 6.891, 7.656, 8.463, 9.303, 10.174, 11.071, 11.991, 12.931, 13.890, 14.866, 15.858, 16.865, 17.884, 18.916, 19.959, 21.015, 22.080, 23.155, 24.239, 25.331, 26.432, 27.542, 28.659},
        {5.281, 4.478, 4.267, 4.457, 4.896, 5.499, 6.214, 7.013, 7.876, 8.791, 9.750, 10.746, 11.773, 12.828, 13.908, 15.010, 16.133, 17.274, 18.433, 19.607, 20.796, 21.998, 23.214, 24.442, 25.681, 26.932, 28.192, 29.461, 30.740, 32.029, 33.326},
        {5.092, 4.489, 4.460, 4.823, 5.432, 6.205, 7.093, 8.068, 9.110, 10.207, 11.350, 12.533, 13.749, 14.997, 16.272, 17.571, 18.892, 20.234, 21.595, 22.974, 24.369, 25.778, 27.202, 28.641, 30.092, 31.556, 33.030, 34.515, 36.011, 37.517, 39.034},
        {4.978, 4.589, 4.765, 5.327, 6.138, 7.116, 8.213, 9.401, 10.661, 11.980, 13.349, 14.761, 16.210, 17.693, 19.206, 20.746, 22.310, 23.898, 25.508, 27.137, 28.783, 30.447, 32.127, 33.823, 35.534, 37.259, 38.996, 40.744, 42.506, 44.279, 46.064},
        {4.956, 4.802, 5.212, 6.011, 7.063, 8.291, 9.643, 11.094, 12.622, 14.214, 15.861, 17.556, 19.292, 21.066, 22.873, 24.711, 26.576, 28.467, 30.383, 32.321, 34.279, 36.256, 38.253, 40.268, 42.299, 44.346, 46.407, 48.482, 50.571, 52.674, 54.790},
        {5.050, 5.158, 5.842, 6.925, 8.273, 9.807, 11.475, 13.251, 15.111, 17.043, 19.036, 21.082, 23.174, 25.309, 27.482, 29.689, 31.928, 34.197, 36.493, 38.815, 41.160, 43.527, 45.916, 48.326, 50.755, 53.202, 55.666, 58.145, 60.641, 63.153, 65.681},
        {5.286, 5.697, 6.705, 8.134, 9.846, 11.759, 13.820, 16.000, 18.274, 20.628, 23.051, 25.534, 28.070, 30.654, 33.282, 35.949, 38.651, 41.389, 44.159, 46.957, 49.783, 52.633, 55.510, 58.411, 61.334, 64.279, 67.242, 70.224, 73.224, 76.244, 79.282},
        {5.714, 6.473, 7.875, 9.730, 11.894, 14.282, 16.835, 19.521, 22.313, 25.196, 28.158, 31.188, 34.278, 37.425, 40.622, 43.864, 47.147, 50.471, 53.833, 57.228, 60.654, 64.109, 67.595, 71.110, 74.651, 78.216, 81.803, 85.412, 89.043, 92.697, 96.372},
        {6.389, 7.566, 9.450, 11.837, 14.570, 17.556, 20.729, 24.052, 27.499, 31.049, 34.691, 38.411, 42.201, 46.057, 49.971, 53.937, 57.951, 62.014, 66.120, 70.266, 74.448, 78.665, 82.918, 87.204, 91.522, 95.868, 100.241, 104.638, 109.062, 113.514, 117.990}
    };

    // Bilinear interpolation of the Ishiyama+21 c-M table
    // Clamp to table boundaries
    double lm = logM;
    double zz = z;
    if(lm < cm_table_logmass[0]) lm = cm_table_logmass[0];
    if(lm > cm_table_logmass[CM_TABLE_N_MASS - 1]) lm = cm_table_logmass[CM_TABLE_N_MASS - 1];
    if(zz < cm_table_z[0]) zz = cm_table_z[0];
    if(zz > cm_table_z[CM_TABLE_N_Z - 1]) zz = cm_table_z[CM_TABLE_N_Z - 1];

    // Find mass index (step = 0.2)
    int im = (int)((lm - 8.0) / 0.2);
    if(im < 0) im = 0;
    if(im >= CM_TABLE_N_MASS - 1) im = CM_TABLE_N_MASS - 2;

    // Find redshift index (step = 0.5)
    int iz = (int)(zz / 0.5);
    if(iz < 0) iz = 0;
    if(iz >= CM_TABLE_N_Z - 1) iz = CM_TABLE_N_Z - 2;

    // Fractional positions
    const double fm = (lm - cm_table_logmass[im]) / (cm_table_logmass[im + 1] - cm_table_logmass[im]);
    const double fz = (zz - cm_table_z[iz]) / (cm_table_z[iz + 1] - cm_table_z[iz]);

    // Bilinear interpolation
    const double c00 = cm_table[im][iz];
    const double c10 = cm_table[im + 1][iz];
    const double c01 = cm_table[im][iz + 1];
    const double c11 = cm_table[im + 1][iz + 1];

    return c00 * (1.0 - fm) * (1.0 - fz)
         + c10 * fm * (1.0 - fz)
         + c01 * (1.0 - fm) * fz
         + c11 * fm * fz;
}

double get_halo_concentration(const int p, const double z, const struct GALAXY *galaxies,
                               const struct params *run_params)
{
    (void)run_params;  // reserved for future c-M options
    const double Mvir = galaxies[p].Mvir;  // 10^10 M_sun / h
    if(Mvir <= 0.0) return 1.0;

    const double Mvir_Msun_h = Mvir * 1.0e10;  // Msun/h (table units)
    const double logM = log10(Mvir_Msun_h);

    // Ishiyama+21 c-M relation (lookup table from Colossus)
    double c = interpolate_concentration_ishiyama21(logM, z);
    if(c < 1.0) c = 1.0;

    // printf("Galaxy %d: Mvir = %.3e (Msun/h), logM = %.3f, z = %.2f, c = %.3f\n", p, Mvir_Msun_h, logM, z, c);

    return c;
}

double calculate_gmax_BK25(const int p, const struct GALAXY *galaxies,
                            const struct params *run_params)
{
    // Boylan-Kolchin 2025: maximum NFW gravitational acceleration
    //
    // g_vir = G * M_vir / R_vir^2                                (Eq. 2)
    // g_max = (g_vir / mu(c)) * (c^2 / 2)                         (Eq. 4)
    // where mu(x) = ln(1+x) - x/(1+x)
    //
    // Returns g_max in code units (UnitLength / UnitTime^2)

    const double Mvir = galaxies[p].Mvir;  // code mass units (10^10 M_sun / h)
    const double Rvir = galaxies[p].Rvir;  // code length units (Mpc / h)

    if(Mvir <= 0.0 || Rvir <= 0.0) {
        return 0.0;
    }

    // g_vir = G * M_vir / R_vir^2  (code units)
    const double g_vir = run_params->G * Mvir / (Rvir * Rvir);  // Convert Mvir to physical units (Msun) for g_vir

    // Use pre-computed concentration from the galaxy struct
    double c = galaxies[p].Concentration;
    if(c < 1.0) c = 1.0;

    // mu(c) = ln(1+c) - c/(1+c)
    const double mu_c = log(1.0 + c) - c / (1.0 + c);

    // g_max = (g_vir / mu(c)) * (c^2 / 2)   [BK25 Eq. 4]
    return (g_vir / mu_c) * (c * c / 2.0);
}


float calculate_stellar_scale_height_BR06(float disk_scale_length_pc)
{
    // BR06 equation (9): log h* = -0.23 - 0.8 log R*
    // where h* and R* are measured in parsecs
    if (disk_scale_length_pc <= 0.0) {
        return 0.0; // Default fallback value in pc
    }
    
    float log_h_star = -0.23 + 0.8 * log10(disk_scale_length_pc);
    float h_star_pc = pow(10.0, log_h_star);
    
    // Apply reasonable physical bounds (from 10 pc to 10 kpc)
    // if (h_star_pc < 10.0) h_star_pc = 10.0;
    // if (h_star_pc > 10000.0) h_star_pc = 10000.0;
    
    return h_star_pc;
}


float calculate_midplane_pressure_BR06(float sigma_gas, float sigma_stars, float disk_scale_length_pc)
{
    // Early termination for edge cases
    if (sigma_gas <= 0.5 || disk_scale_length_pc <= 0.0) {
        return 0.0;
    }
    
    // For very low stellar surface density, use a minimal value to avoid numerical issues
    // but don't artificially boost it like before
    float effective_sigma_stars = sigma_stars;
    if (sigma_stars < 0.1) {
        effective_sigma_stars = 0.1;  // Minimal floor just to avoid sqrt(0)
    }
    
    // Calculate stellar scale height using exact BR06 equation (9)
    float h_star_pc = calculate_stellar_scale_height_BR06(disk_scale_length_pc);
    
    // BR06 hardcoded parameters EXACTLY as in paper
    const float v_g = 8.0;          // km/s, gas velocity dispersion (BR06 Table text)
    
    // BR06 Equation (5) EXACTLY as written in paper:
    // P_ext/k = 272 cm⁻³ K × (Σ_gas/M_⊙ pc⁻²) × (Σ_*/M_⊙ pc⁻²)^0.5 × (v_g/km s⁻¹) × (h_*/pc)^-0.5
    float pressure = 272.0 * sigma_gas * sqrt(effective_sigma_stars) * v_g / sqrt(h_star_pc);


    return pressure; // K cm⁻³
}


float calculate_molecular_fraction_BR06(float gas_surface_density, float stellar_surface_density, 
                                         float disk_scale_length_pc)
{

    // Calculate midplane pressure using exact BR06 formula
    float pressure = calculate_midplane_pressure_BR06(gas_surface_density, stellar_surface_density, 
                                                     disk_scale_length_pc);
    
    if (pressure <= 0.0) {
        return 0.0;
    }
    
    // BR06 parameters from equation (13) for non-interacting galaxies
    // These are the exact values from the paper
    const float P0 = 4.54e4;    // Reference pressure, K cm⁻³ (equation 13)
    const float alpha = 0.92;  // Power law index (equation 13)
    
    // BR06 Equation (11): R_mol = (P_ext/P₀)^α
    float pressure_ratio = pressure / P0;
    float R_mol = pow(pressure_ratio, alpha);
    
    // Convert to molecular fraction: f_mol = R_mol / (1 + R_mol)
    // This is the standard conversion from molecular-to-atomic ratio to molecular fraction
    double f_mol = R_mol / (1.0 + R_mol);
    
    return f_mol;
}

float calculate_molecular_fraction_radial_integration(const int gal, struct GALAXY *galaxies, 
                                                      const struct params *run_params)
{
    const float h = run_params->Hubble_h;
    const float rs_pc = galaxies[gal].DiskScaleRadius * 1.0e6 / h;  // Scale radius in pc
    
    if (rs_pc <= 0.0 || galaxies[gal].ColdGas <= 0.0) {
        return 0.0;
    }
    
    // Total masses in physical units (M☉)
    const float M_gas_total = galaxies[gal].ColdGas * 1.0e10 / h;
    const float M_star_total = galaxies[gal].StellarMass * 1.0e10 / h;
    
    // Central surface densities for exponential profiles: Σ₀ = M_total / (2π r_s²)
    const float sigma_gas_0 = M_gas_total / (2.0 * M_PI * rs_pc * rs_pc);
    const float sigma_star_0 = M_star_total / (2.0 * M_PI * rs_pc * rs_pc);
    
    // Radial integration parameters
    const int N_BINS = 10;  // Number of radial bins
    const float R_MAX = 3.0 * rs_pc;  // Integrate out to 3 scale radii (~95% of mass)
    const float dr = R_MAX / N_BINS;
    
    // Integrate molecular gas mass
    float M_H2_total = 0.0;
    
    for (int i = 0; i < N_BINS; i++) {
        // Bin center radius
        const float r = (i + 0.5) * dr;
        
        // Exponential surface density profiles: Σ(r) = Σ₀ exp(-r/r_s)
        const float exp_factor = exp(-r / rs_pc);
        const float sigma_gas_r = sigma_gas_0 * exp_factor;
        const float sigma_star_r = sigma_star_0 * exp_factor;
        
        // Skip bins with negligible gas
        if (sigma_gas_r < 1e-3) continue;
        
        // Calculate molecular fraction at this radius using BR06
        const float f_mol_r = calculate_molecular_fraction_BR06(sigma_gas_r, sigma_star_r, 
                                                                rs_pc);
        
        // Mass of molecular gas in this annulus: dM = 2π r Σ_gas f_mol dr
        const float dM_H2 = 2.0 * M_PI * r * sigma_gas_r * f_mol_r * dr;
        
        M_H2_total += dM_H2;
    }
    
    // Convert back to code units (10^10 M☉/h)
    const float H2_code_units = M_H2_total * h / 1.0e10;
    
    // Store and return
    galaxies[gal].H2gas = H2_code_units;
    return H2_code_units;
}

double calculate_ffb_threshold_mass(const double z, const struct params *run_params)
{
    // Equation (2) from Li et al. 2024
    // M_v,ffb / 10^10.8 M_sun ~ ((1+z)/10)^-6.2
    //
    // In code units (10^10 M_sun/h):
    // log(M_code) = log(M_sun) - 10 + log(h)
    //             = 10.8 - 6.2*log((1+z)/10) - 10 + log(h)
    //             = 0.8 + log(h) - 6.2*log((1+z)/10)

    const double h = run_params->Hubble_h;
    const double z_norm = (1.0 + z) / 10.0;
    const double log_Mvir_ffb_code = 0.8 + log10(h) - 6.2 * log10(z_norm);

    return pow(10.0, log_Mvir_ffb_code);
}


double calculate_ffb_fraction(const double Mvir, const double z, const struct params *run_params)
{
    // Calculate the fraction of galaxies in FFB regime
    // Uses smooth sigmoid transition from Li et al. 2024, equation (3)
    
    if (run_params->FeedbackFreeModeOn == 0) {
        return 0.0;  // FFB mode disabled
    }

    // if (z < 5.0) {
    //     return 0.0;  // FFB only active at z >= 6.2
    // }
    
    // Calculate FFB threshold mass
    const double Mvir_ffb = calculate_ffb_threshold_mass(z, run_params);

    // BUG FIX: Protect against log10(0) or log10(negative)
    if(Mvir <= 0.0 || Mvir_ffb <= 0.0) {
        return 0.0;  // Return no FFB for invalid masses
    }

    // Width of transition in dex (Li et al. use 0.15 dex)
    const double delta_log_M = 0.15;

    // Calculate argument for sigmoid function
    const double x = log10(Mvir / Mvir_ffb) / delta_log_M;
    
    // Sigmoid function: S(x) = 1 / (1 + exp(-x))
    // const double k = 5.0;  // Steepness parameter (can be adjusted)
    // Sigmoid function with adjustable steepness
    // Smoothly varies from 0 (well below threshold) to 1 (well above threshold)
    const double f_ffb = 1.0 / (1.0 + exp(-x));
    // Steeper transition
    // const double f_ffb = 1.0 / (1.0 + exp(-k * x));
    
    return f_ffb;
}

// Calculate molecular fraction using Krumholz & Dekel (2012) model
// Based on equations 18-21 from Krumholz & Dekel 2012, ApJ 753:16

float calculate_H2_fraction_KD12(const float surface_density, const float metallicity, const float clumping_factor) 
{
    if (surface_density <= 0.0) {
        return 0.0;
    }
    
    // Metallicity normalized to solar (Z_sun = 0.02)
    // Z0 = (M_Z/M_g)/Z_sun as defined in KD12 equation after (17)
    // Apply floor to prevent numerical issues and unphysical zero H2 at very low Z
    float metallicity_floored = metallicity;
    if (metallicity_floored < 0.0002) {  // Z = 0.01 Z_sun minimum
        metallicity_floored = 0.0002;
    }
    float Z0 = metallicity_floored / 0.02;
    
    // Convert surface density from M_sun/pc^2 to g/cm^2
    // Conversion: 1 M_sun/pc^2 = 2.088 × 10^-4 g/cm^2
    float Sigma_gcm2 = surface_density * 2.088e-4;
    
    // Surface density normalized to 1 g/cm^2 (as defined after KD12 Eq. 16)
    // Sigma_0 = Sigma / (1 g cm^-2)
    float Sigma_0 = Sigma_gcm2;  // dimensionless, in units of 1 g/cm^2
    
    // Calculate dust optical depth parameter (KD12 Eq. 21)
    // tau_c = 320 * c * Z0 * Sigma_0
    // where c is the clumping factor:
    //   c ≈ 1 for Sigma measured on 100 pc scales
    //   c ≈ 5 for Sigma measured on ~1 kpc scales (from text after Eq. 21)
    float tau_c = 320.0 * clumping_factor * Z0 * Sigma_0;
    
    // Self-shielding parameter chi (KD12 Eq. 20)
    // chi = 3.1 * (1 + Z0^0.365) / 4.1
    float chi = 3.1 * (1.0 + pow(Z0, 0.365)) / 4.1;
    
    // Compute s parameter (KD12 Eq. 19)
    // s = ln(1 + 0.6*chi + 0.01*chi^2) / (0.6 * tau_c)
    float chi_sq = chi * chi;
    float s;
    // BUG FIX: Protect against division by zero when tau_c is very small
    if(tau_c > 1e-10) {
        s = log(1.0 + 0.6 * chi + 0.01 * chi_sq) / (0.6 * tau_c);
    } else {
        s = 100.0;  // Large s implies f_H2 -> 0 (atomic dominated)
    }
    
    // Molecular fraction (KD12 Eq. 18)
    // f_H2 = 1 - (3/4) * s/(1 + 0.25*s)  for s < 2
    // f_H2 = 0                            for s >= 2
    float f_H2;
    if (s < 2.0) {
        f_H2 = 1.0 - 0.75 * s / (1.0 + 0.25 * s);
    } else {
        f_H2 = 0.0;
    }
    
    // Ensure fraction stays within bounds
    if (f_H2 < 0.0) f_H2 = 0.0;
    if (f_H2 > 1.0) f_H2 = 1.0;
    
    return f_H2;
}
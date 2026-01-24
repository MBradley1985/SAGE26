#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "core_allvars.h"
#include "core_cool_func.h"

#include "model_cooling_heating.h"
#include "model_misc.h"


double cooling_recipe(const int gal, const double dt, struct GALAXY *galaxies, const struct params *run_params)
{
    // Check if CGM recipe is enabled for backwards compatibility
    if(run_params->CGMrecipeOn > 0) {
        return cooling_recipe_regime_aware(gal, dt, galaxies, run_params);
    } else {
        return cooling_recipe_hot(gal, dt, galaxies, run_params);
    }
}

double cooling_recipe_hot(const int gal, const double dt, struct GALAXY *galaxies, const struct params *run_params)
{
    double coolingGas;

    if(galaxies[gal].HotGas > 0.0 && galaxies[gal].Vvir > 0.0) {
        
        // ====================================================================
        // STEP 1: Standard cooling radius calculation
        // ====================================================================
        const double tcool = galaxies[gal].Rvir / galaxies[gal].Vvir;
        const double temp = 35.9 * galaxies[gal].Vvir * galaxies[gal].Vvir;  // Kelvin

        double logZ = -10.0;
        if(galaxies[gal].MetalsHotGas > 0) {
            logZ = log10(galaxies[gal].MetalsHotGas / galaxies[gal].HotGas);
        }

        double lambda = get_metaldependent_cooling_rate(log10(temp), logZ);

        // Check lambda > 0 to avoid division by zero
        if(lambda <= 0.0) {
            return 0.0;
        }

        double x = PROTONMASS * BOLTZMANN * temp / lambda;
        x /= (run_params->UnitDensity_in_cgs * run_params->UnitTime_in_s);
        const double rho_rcool = x / tcool * 0.885;  // 0.885 = 3/2 * mu, mu=0.59

        if(rho_rcool <= 0.0) {
            return 0.0;
        }

        // Isothermal density profile for hot gas
        const double rho0 = galaxies[gal].HotGas / (4 * M_PI * galaxies[gal].Rvir);
        const double rcool = sqrt(rho0 / rho_rcool);

        galaxies[gal].RcoolToRvir = rcool / galaxies[gal].Rvir;

        // ====================================================================
        // STEP 2: Calculate base cooling rate from hot halo
        // ====================================================================
        double hot_halo_cooling = 0.0;
        
        if(rcool > galaxies[gal].Rvir) {
            // Rapid cooling regime: cooling time < dynamical time
            // All hot gas can cool on dynamical timescale
            hot_halo_cooling = galaxies[gal].HotGas / tcool * dt;
        } else {
            // Standard hot halo cooling from within rcool
            hot_halo_cooling = (galaxies[gal].HotGas / galaxies[gal].Rvir) * 
                               (rcool / (2.0 * tcool)) * dt;
        }

        // ====================================================================
        // STEP 3: Cold streams in hot halos (D&B06)
        // Only applies when CGMrecipeOn is enabled
        // ====================================================================
        double cold_stream_cooling = 0.0;
        
        if(run_params->CGMrecipeOn == 1) {
            cold_stream_cooling = calculate_cold_stream_cooling(gal, dt, galaxies, run_params);
        }

        // ====================================================================
        // STEP 4: Total cooling
        // ====================================================================
        coolingGas = hot_halo_cooling + cold_stream_cooling;

        // Physical limits
        if(coolingGas > galaxies[gal].HotGas) {
            coolingGas = galaxies[gal].HotGas;
        }
        if(coolingGas < 0.0) {
            coolingGas = 0.0;
        }

        // ====================================================================
        // STEP 5: AGN heating (if enabled)
        // ====================================================================
        if(run_params->AGNrecipeOn > 0 && coolingGas > 0.0) {
            coolingGas = do_AGN_heating(coolingGas, gal, dt, x, rcool, galaxies, run_params);
        }

        // Track cooling energy
        if(coolingGas > 0.0) {
            galaxies[gal].Cooling += 0.5 * coolingGas * galaxies[gal].Vvir * galaxies[gal].Vvir;
        }
        
    } else {
        coolingGas = 0.0;
    }

    XASSERT(coolingGas >= 0.0, -1,
            "Error: Cooling gas mass = %g should be >= 0.0", coolingGas);
    return coolingGas;
}


double calculate_cold_stream_cooling(const int gal, const double dt, 
                                     struct GALAXY *galaxies, const struct params *run_params)
{
    // ========================================================================
    // Cold streams in hot halos following Dekel & Birnboim (2006)
    //
    // Key physics:
    // - Cold filamentary streams can penetrate shock-heated halos
    // - Stream survival depends on stream density vs halo density
    // - More prominent at high-z when halos are rare peaks
    // - Streams deliver gas on ~dynamical timescale
    //
    // D&B06 equations 37-41:
    //   (tcool/tcomp)_stream = (rho_halo/rho_stream) * (M/M_shock)^(4/3)
    //   rho_stream/rho_halo ~ (fM_*/M)^(-2/3)
    //   => f_stream ~ (fM_*/M)^(2/3) * (M_shock/M)^(4/3)
    // ========================================================================
    
    if(galaxies[gal].HotGas <= 0.0 || galaxies[gal].Vvir <= 0.0) {
        return 0.0;
    }
    
    // Physical parameters
    const double Mshock = 6.0e11;      // Msun - shock heating threshold
    const double f_cluster = 3.0;      // Clustering parameter
    
    // Get current redshift
    const double z = run_params->ZZ[galaxies[gal].SnapNum];
    
    // Calculate M_* at current redshift (Press-Schechter)
    double log_Mstar;
    if(z < 4.0) {
        log_Mstar = 13.1 - 1.3 * z;
    } else {
        log_Mstar = 13.1 - 1.3 * 4.0 - 0.5 * (z - 4.0);
    }
    const double Mstar = pow(10.0, log_Mstar);  // Msun
    
    // Critical redshift
    const double z_crit = (13.1 - log10(Mshock / f_cluster)) / 1.3;  // ~1.4 for f=3
    
    // Halo mass in physical units
    const double Mvir_physical = galaxies[gal].Mvir * 1.0e10 / run_params->Hubble_h;
    
    // ========================================================================
    // Cold stream fraction calculation
    // ========================================================================
    double f_stream = 0.0;
    
    if(z > z_crit && Mvir_physical > Mshock) {
        // ====================================================================
        // High-z regime: cold streams can penetrate hot halos
        // 
        // D&B06 eq 39: The ratio tcool/tcomp in streams depends on:
        //   - Density enhancement in streams: (fM_*/M)^(-2/3)
        //   - Mass ratio: (M/M_shock)^(4/3)
        //
        // Stream fraction ~ probability that a parcel arrives cold
        // ====================================================================
        
        // Mass-dependent suppression: more massive halos have fewer streams
        // (M_shock/M)^(4/3) from D&B06 eq 38
        const double mass_suppression = pow(Mshock / Mvir_physical, 4.0/3.0);
        
        // Density enhancement in streams relative to spherical infall
        // Streams are denser by factor ~ (M / fM_*)^(2/3)
        // This HELPS streams survive (shorter tcool in stream)
        // So stream fraction INCREASES when M > fM_*
        double density_factor;
        if(Mvir_physical > f_cluster * Mstar) {
            // Halo is massive relative to clustering scale
            // Streams are dense and can penetrate
            density_factor = pow(Mvir_physical / (f_cluster * Mstar), 2.0/3.0);
        } else {
            // Halo is typical or below clustering scale
            // Less prominent filamentary feeding
            density_factor = 1.0;
        }
        
        // Combined stream fraction
        // Normalize so that at M = M_shock, z ~ 2-3, we get f_stream ~ 0.3-0.5
        f_stream = mass_suppression * density_factor;
        
        // Redshift modulation: streams more effective at higher z
        // Linear ramp from z_crit to z ~ 4
        const double z_factor = (z - z_crit) / (4.0 - z_crit);
        f_stream *= fmin(z_factor, 1.0);
        
        // ====================================================================
        // Physical bounds
        // ====================================================================
        // Maximum ~50%: even with cold streams, some gas is shock-heated
        // and streams may be disrupted/heated as they penetrate
        if(f_stream > 0.5) f_stream = 0.5;
        
        // Minimum 0
        if(f_stream < 0.0) f_stream = 0.0;
        
        // Suppress streams in very massive halos (clusters)
        // ICM is too hot and dense for streams to survive
        if(Mvir_physical > 1.0e13) {
            f_stream *= exp(-(Mvir_physical - 1.0e13) / 1.0e13);
        }
        
    } else if(Mvir_physical < Mshock) {
        // ====================================================================
        // Below M_shock: no virial shock, ALL gas arrives cold
        // But this should be handled by Regime = 0 (CGM regime)
        // If we're in this function, galaxy is Regime = 1 (HOT)
        // So this case shouldn't occur often - just return 0
        // ====================================================================
        f_stream = 0.0;
    }
    // else: z < z_crit and M > M_shock: classical hot halo, no cold streams
    
    // ========================================================================
    // Calculate cold stream cooling rate
    // Streams deliver gas on dynamical timescale (free-fall along filament)
    // ========================================================================
    double cold_stream_cooling = 0.0;
    
    if(f_stream > 0.0) {
        const double t_dyn = galaxies[gal].Rvir / galaxies[gal].Vvir;
        
        // Cold streams tap into the hot gas reservoir
        // (In reality they're separate, but in this simplified model
        // we account for it as a fraction of the total gas supply)
        cold_stream_cooling = f_stream * galaxies[gal].HotGas / t_dyn * dt;
        
        // Don't exceed available gas
        if(cold_stream_cooling > f_stream * galaxies[gal].HotGas) {
            cold_stream_cooling = f_stream * galaxies[gal].HotGas;
        }
    }
    
    return cold_stream_cooling;
}


double cooling_recipe_cgm(const int gal, const double dt, struct GALAXY *galaxies, 
                         const struct params *run_params)
{
    static long precipitation_debug_counter = 0;
    precipitation_debug_counter++;
    
    double coolingGas = 0.0;

    // ========================================================================
    // EARLY EXIT CONDITIONS
    // ========================================================================
    if(galaxies[gal].CGMgas <= 0.0 || galaxies[gal].Vvir <= 0.0 || galaxies[gal].Rvir <= 0.0) {
        if(precipitation_debug_counter % 50000 == 0) {
            printf("DEBUG PRECIP [%ld]: Early exit - CGMgas=%.2e, Vvir=%.2f, Rvir=%.2e\n",
                   precipitation_debug_counter, galaxies[gal].CGMgas, 
                   galaxies[gal].Vvir, galaxies[gal].Rvir);
        }
        return 0.0;
    }

    // ========================================================================
    // STEP 1: CALCULATE COOLING TIME (CGS UNITS)
    // ========================================================================
    
    // Virial temperature
    const double temp = 35.9 * galaxies[gal].Vvir * galaxies[gal].Vvir; // Kelvin
    
    // Metallicity
    double logZ = -10.0;
    if(galaxies[gal].MetalsCGMgas > 0) {
        logZ = log10(galaxies[gal].MetalsCGMgas / galaxies[gal].CGMgas);
    }
    
    // Cooling function (erg cm^3 s^-1)
    double lambda = get_metaldependent_cooling_rate(log10(temp), logZ);

    
    // Convert CGM mass and radius to CGS
    const double CGMgas_cgs = galaxies[gal].CGMgas * 1e10 * SOLAR_MASS / run_params->Hubble_h; // g
    const double Rvir_cgs = galaxies[gal].Rvir * CM_PER_MPC / run_params->Hubble_h; // cm
    
    // Volume and mass density
    const double volume_cgs = (4.0 * M_PI / 3.0) * Rvir_cgs * Rvir_cgs * Rvir_cgs; // cm^3
    const double mass_density_cgs = CGMgas_cgs / volume_cgs; // g cm^-3
    
    // Cooling time: tcool = (3/2) * μ * m_p * k * T / (ρ * Λ)
    // where μ = 0.59 for fully ionized gas
    const double mu = 0.59;
    const double tcool_cgs = (1.5 * mu * PROTONMASS * BOLTZMANN * temp) / (mass_density_cgs * lambda); // s
    const double tcool = tcool_cgs / run_params->UnitTime_in_s; // code units

    // ========================================================================
    // STEP 2: CALCULATE FREE-FALL TIME
    // ========================================================================
    
    // Gravitational acceleration at Rvir
    const float g_accel = run_params->G * galaxies[gal].Mvir / (galaxies[gal].Rvir * galaxies[gal].Rvir);
    
    // Free-fall time: tff = sqrt(2*R/g)
    const float tff = sqrt(2.0 * galaxies[gal].Rvir / g_accel); // code units

    // Critical ratio for precipitation
    const float tcool_over_tff = tcool / tff;


    // Save to galaxy struct for potential diagnostics
    galaxies[gal].tcool = tcool;
    galaxies[gal].tff = tff;
    galaxies[gal].tcool_over_tff = tcool_over_tff;

    // ========================================================================
    // STEP 3: PRECIPITATION CRITERION
    // ========================================================================

    const double precipitation_threshold = 10;  // default=10, McCourt et al. 2012
    const double transition_width = 2.0;  // Smooth transition over factor ~2
    
    double precipitation_fraction = 0.0;
    
    if(tcool_over_tff < precipitation_threshold) {
        // UNSTABLE: Precipitation cooling
        double instability_factor = precipitation_threshold / tcool_over_tff;
        instability_factor = fmin(instability_factor, 3.0);
        precipitation_fraction = tanh(instability_factor / 2.0);
        
    } else if(tcool_over_tff < precipitation_threshold + transition_width) {
        // TRANSITION: Smoothly reduce precipitation_fraction to zero
        const double x = (tcool_over_tff - precipitation_threshold) / transition_width;
        precipitation_fraction = 0.5 * (1.0 - tanh(x));
        
    } else {
        if(tcool > 0) {
        // Cooling rate: dM/dt = M_CGM / t_cool
        coolingGas = galaxies[gal].CGMgas / tcool * dt;
        
        // Safety check
        if(coolingGas > galaxies[gal].CGMgas) {
            coolingGas = galaxies[gal].CGMgas;
        }
    } else {
        coolingGas = 0.0;
        }

    }

    // Adding this diagnostic for output
    const double x = (tcool_over_tff - precipitation_threshold) / transition_width;

    const double rho_rcool = x / tcool * 0.885;  // 0.885 = 3/2 * mu, mu=0.59 for a fully ionized gas

    // an isothermal density profile for the hot gas is assumed here
    const double rho0 = galaxies[gal].CGMgas / (4 * M_PI * galaxies[gal].Rvir);
    const double rcool = sqrt(rho0 / rho_rcool);

    galaxies[gal].RcoolToRvir = rcool / galaxies[gal].Rvir;
    // else: tcool_over_tff >= 15, precipitation_fraction = 0.0 (thermally stable)

    // ========================================================================
    // STEP 4: CALCULATE PRECIPITATION RATE
    // ========================================================================
    
    if(precipitation_fraction > 0.0) {
        // Gas precipitates on the free-fall timescale when thermally unstable
        // This is the key physical insight: dM/dt = f_precip * M_CGM / t_ff
        const double precip_rate = precipitation_fraction * galaxies[gal].CGMgas / tff;
        coolingGas = precip_rate * dt;
        
        // Physical limits
        if(coolingGas > galaxies[gal].CGMgas) {
            coolingGas = galaxies[gal].CGMgas;
        }
        if(coolingGas < 0.0) {
            coolingGas = 0.0;
        }
    }

    // ========================================================================
    // STEP 5: TRACK COOLING ENERGY
    // ========================================================================
    
    // Energy associated with cooling (for feedback balance tracking)
    if(coolingGas > 0.0) {
        // Specific energy ~ 0.5 * Vvir^2 (thermal + kinetic)
        galaxies[gal].Cooling += 0.5 * coolingGas * galaxies[gal].Vvir * galaxies[gal].Vvir;
    }

    // ========================================================================
    // STEP 6: CALCULATE DEPLETION TIMESCALE (DIAGNOSTIC)
    // ========================================================================

    // Depletion timescale
    if(coolingGas > 0.0) {
        const float depletion_time = galaxies[gal].CGMgas * tff / (precipitation_fraction * galaxies[gal].CGMgas);
        // const float depletion_time_myr = depletion_time * run_params->UnitTime_in_s / (1e6 * SEC_PER_YEAR);

        // Store depletion time for diagnostics
        galaxies[gal].tdeplete = depletion_time;
    }
        
    //     printf("============================================\n\n");
    // }

    // Sanity check
    XASSERT(coolingGas >= 0.0, -1, "Error: Cooling gas mass = %g should be >= 0.0", coolingGas);
    XASSERT(coolingGas <= galaxies[gal].CGMgas + 1e-12, -1,
            "Error: Cooling gas = %g exceeds CGM gas = %g", coolingGas, galaxies[gal].CGMgas);
    
    return coolingGas;
}

double cooling_recipe_regime_aware(const int gal, const double dt, struct GALAXY *galaxies, const struct params *run_params)
{
    double cgm_cooling = 0.0;
    double hot_cooling = 0.0;
    
    if(galaxies[gal].Regime == 0) {
        // CGM REGIME: CGM physics dominates
        
        // Primary: Precipitation cooling from CGMgas
        if(galaxies[gal].CGMgas > 0.0) {
            cgm_cooling = cooling_recipe_cgm(gal, dt, galaxies, run_params);
        }
        
    
    } else {
        // HOT REGIME: Traditional physics dominates
        
        // Primary: Traditional cooling from HotGas
        if(galaxies[gal].HotGas > 0.0) {
            hot_cooling = cooling_recipe_hot(gal, dt, galaxies, run_params);
        }
        
        // Secondary: Precipitation cooling from CGMgas (gradually depletes)
        if(galaxies[gal].CGMgas > 0.0) {
            cgm_cooling = cooling_recipe_cgm(gal, dt, galaxies, run_params);
        }
    }

    // Now apply the cooling directly to preserve the physics-based split
    // Apply CGM cooling
    if(cgm_cooling > 0.0) {
        const double metallicity = get_metallicity(galaxies[gal].CGMgas, galaxies[gal].MetalsCGMgas);
        galaxies[gal].ColdGas += cgm_cooling;
        galaxies[gal].MetalsColdGas += metallicity * cgm_cooling;
        galaxies[gal].CGMgas -= cgm_cooling;
        galaxies[gal].MetalsCGMgas -= metallicity * cgm_cooling;
    }
    
    // Apply HotGas cooling
    if(hot_cooling > 0.0) {
        const double metallicity = get_metallicity(galaxies[gal].HotGas, galaxies[gal].MetalsHotGas);
        galaxies[gal].ColdGas += hot_cooling;
        galaxies[gal].MetalsColdGas += metallicity * hot_cooling;
        galaxies[gal].HotGas -= hot_cooling;
        galaxies[gal].MetalsHotGas -= metallicity * hot_cooling;
    }

    double total_cooling = cgm_cooling + hot_cooling;
    XASSERT(total_cooling >= 0.0, -1,
            "Error: Cooling gas mass = %g should be >= 0.0", total_cooling);
    return total_cooling;
}



double do_AGN_heating(double coolingGas, const int centralgal, const double dt, const double x, const double rcool, struct GALAXY *galaxies, const struct params *run_params)
{
    double AGNrate, EDDrate, AGNaccreted, AGNcoeff, AGNheating, metallicity;

	// first update the cooling rate based on the past AGN heating
	if(galaxies[centralgal].r_heat < rcool) {
		coolingGas = (1.0 - galaxies[centralgal].r_heat / rcool) * coolingGas;
    } else {
		coolingGas = 0.0;
    }

	XASSERT(coolingGas >= 0.0, -1,
            "Error: Cooling gas mass = %g should be >= 0.0", coolingGas);

	// now calculate the new heating rate
    if(galaxies[centralgal].HotGas > 0.0) {
        if(run_params->AGNrecipeOn == 2 || run_params->AGNrecipeOn == 4) {
            // Bondi-Hoyle accretion recipe (AGNrecipeOn==4 uses seeded BHs)
            AGNrate = (2.5 * M_PI * run_params->G) * (0.375 * 0.6 * x) * galaxies[centralgal].BlackHoleMass * run_params->RadioModeEfficiency;
        } else if(run_params->AGNrecipeOn == 3) {
            // Cold cloud accretion: trigger: rBH > 1.0e-4 Rsonic, and accretion rate = 0.01% cooling rate
            if(galaxies[centralgal].BlackHoleMass > 0.0001 * galaxies[centralgal].Mvir * CUBE(rcool/galaxies[centralgal].Rvir)) {
                AGNrate = 0.0001 * coolingGas / dt;
            } else {
                AGNrate = 0.0;
            }
        } else {
            // empirical (standard) accretion recipe
            if(galaxies[centralgal].Mvir > 0.0) {
                AGNrate = run_params->RadioModeEfficiency / (run_params->UnitMass_in_g / run_params->UnitTime_in_s * SEC_PER_YEAR / SOLAR_MASS)
                    * (galaxies[centralgal].BlackHoleMass / 0.01) * CUBE(galaxies[centralgal].Vvir / 200.0)
                    * ((galaxies[centralgal].HotGas / galaxies[centralgal].Mvir) / 0.1);
            } else {
                AGNrate = run_params->RadioModeEfficiency / (run_params->UnitMass_in_g / run_params->UnitTime_in_s * SEC_PER_YEAR / SOLAR_MASS)
                    * (galaxies[centralgal].BlackHoleMass / 0.01) * CUBE(galaxies[centralgal].Vvir / 200.0);
            }
        }

        // Eddington rate
        EDDrate = (1.3e38 * galaxies[centralgal].BlackHoleMass * 1e10 / run_params->Hubble_h) / (run_params->UnitEnergy_in_cgs / run_params->UnitTime_in_s) / (0.1 * 9e10);

        // accretion onto BH is always limited by the Eddington rate
        if(AGNrate > EDDrate) {
            AGNrate = EDDrate;
        }

        // accreted mass onto black hole
        AGNaccreted = AGNrate * dt;

        // cannot accrete more mass than is available!
        if(AGNaccreted > galaxies[centralgal].HotGas) {
            AGNaccreted = galaxies[centralgal].HotGas;
        }

        // coefficient to heat the cooling gas back to the virial temperature of the halo
        // 1.34e5 = sqrt(2*eta*c^2), eta=0.1 (standard efficiency) and c in km/s
        // BUG FIX: Check Vvir > 0 to avoid division by zero
        if(galaxies[centralgal].Vvir <= 0.0) {
            AGNcoeff = 0.0;
            AGNheating = 0.0;
        } else {
            AGNcoeff = (1.34e5 / galaxies[centralgal].Vvir) * (1.34e5 / galaxies[centralgal].Vvir);

            // cooling mass that can be suppresed from AGN heating
            AGNheating = AGNcoeff * AGNaccreted;

            /// the above is the maximal heating rate. we now limit it to the current cooling rate
            if(AGNheating > coolingGas && AGNcoeff > 0.0) {
                AGNaccreted = coolingGas / AGNcoeff;
                AGNheating = coolingGas;
            }
        }

        // accreted mass onto black hole
        metallicity = get_metallicity(galaxies[centralgal].HotGas, galaxies[centralgal].MetalsHotGas);
        galaxies[centralgal].BlackHoleMass += AGNaccreted;
        galaxies[centralgal].HotGas -= AGNaccreted;
        galaxies[centralgal].MetalsHotGas -= metallicity * AGNaccreted;

        // update the heating radius as needed
        if(galaxies[centralgal].r_heat < rcool && coolingGas > 0.0) {
            double r_heat_new = (AGNheating / coolingGas) * rcool;
            if(r_heat_new > galaxies[centralgal].r_heat) {
                galaxies[centralgal].r_heat = r_heat_new;
            }
        }

        if (AGNheating > 0.0) {
            galaxies[centralgal].Heating += 0.5 * AGNheating * galaxies[centralgal].Vvir * galaxies[centralgal].Vvir;
        }
    }

    return coolingGas;
}

double do_AGN_heating_cgm(double coolingGas, const int centralgal, const double dt, const double x, const double rcool, 
                         struct GALAXY *galaxies, const struct params *run_params)
{
    double AGNrate, EDDrate, AGNaccreted, AGNcoeff, AGNheating, metallicity;

	// first update the cooling rate based on the past AGN heating
	if(galaxies[centralgal].r_heat < rcool) {
		coolingGas = (1.0 - galaxies[centralgal].r_heat / rcool) * coolingGas;
    } else {
		coolingGas = 0.0;
    }

	XASSERT(coolingGas >= 0.0, -1,
            "Error: Cooling gas mass = %g should be >= 0.0", coolingGas);

	// now calculate the new heating rate
    if(galaxies[centralgal].CGMgas > 0.0) {
        if(run_params->AGNrecipeOn == 2 || run_params->AGNrecipeOn == 4) {
            // Bondi-Hoyle accretion recipe (AGNrecipeOn==4 uses seeded BHs)
            AGNrate = (2.5 * M_PI * run_params->G) * (0.375 * 0.6 * x) * galaxies[centralgal].BlackHoleMass * run_params->RadioModeEfficiency;
        } else if(run_params->AGNrecipeOn == 3) {
            // Cold cloud accretion: trigger: rBH > 1.0e-4 Rsonic, and accretion rate = 0.01% cooling rate
            if(galaxies[centralgal].BlackHoleMass > 0.0001 * galaxies[centralgal].Mvir * CUBE(rcool/galaxies[centralgal].Rvir)) {
                AGNrate = 0.0001 * coolingGas / dt;
            } else {
                AGNrate = 0.0;
            }
        } else {
            // empirical (standard) accretion recipe
            if(galaxies[centralgal].Mvir > 0.0) {
                AGNrate = run_params->RadioModeEfficiency / (run_params->UnitMass_in_g / run_params->UnitTime_in_s * SEC_PER_YEAR / SOLAR_MASS)
                    * (galaxies[centralgal].BlackHoleMass / 0.01) * CUBE(galaxies[centralgal].Vvir / 200.0)
                    * ((galaxies[centralgal].CGMgas / galaxies[centralgal].Mvir) / 0.1);
            } else {
                AGNrate = run_params->RadioModeEfficiency / (run_params->UnitMass_in_g / run_params->UnitTime_in_s * SEC_PER_YEAR / SOLAR_MASS)
                    * (galaxies[centralgal].BlackHoleMass / 0.01) * CUBE(galaxies[centralgal].Vvir / 200.0);
            }
        }

        // Eddington rate
        EDDrate = (1.3e38 * galaxies[centralgal].BlackHoleMass * 1e10 / run_params->Hubble_h) / (run_params->UnitEnergy_in_cgs / run_params->UnitTime_in_s) / (0.1 * 9e10);

        // accretion onto BH is always limited by the Eddington rate
        if(AGNrate > EDDrate) {
            AGNrate = EDDrate;
        }

        // accreted mass onto black hole
        AGNaccreted = AGNrate * dt;

        // cannot accrete more mass than is available!
        if(AGNaccreted > galaxies[centralgal].CGMgas) {
            AGNaccreted = galaxies[centralgal].CGMgas;
        }

        // coefficient to heat the cooling gas back to the virial temperature of the halo
        // 1.34e5 = sqrt(2*eta*c^2), eta=0.1 (standard efficiency) and c in km/s
        // BUG FIX: Check Vvir > 0 to avoid division by zero
        if(galaxies[centralgal].Vvir <= 0.0) {
            AGNcoeff = 0.0;
            AGNheating = 0.0;
        } else {
            AGNcoeff = (1.34e5 / galaxies[centralgal].Vvir) * (1.34e5 / galaxies[centralgal].Vvir);

            // cooling mass that can be suppresed from AGN heating
            AGNheating = AGNcoeff * AGNaccreted;

            /// the above is the maximal heating rate. we now limit it to the current cooling rate
            if(AGNheating > coolingGas && AGNcoeff > 0.0) {
                AGNaccreted = coolingGas / AGNcoeff;
                AGNheating = coolingGas;
            }
        }

        // accreted mass onto black hole
        metallicity = get_metallicity(galaxies[centralgal].CGMgas, galaxies[centralgal].MetalsCGMgas);
        galaxies[centralgal].BlackHoleMass += AGNaccreted;
        galaxies[centralgal].CGMgas -= AGNaccreted;
        galaxies[centralgal].MetalsCGMgas -= metallicity * AGNaccreted;

        // update the heating radius as needed
        if(galaxies[centralgal].r_heat < rcool && coolingGas > 0.0) {
            double r_heat_new = (AGNheating / coolingGas) * rcool;
            if(r_heat_new > galaxies[centralgal].r_heat) {
                galaxies[centralgal].r_heat = r_heat_new;
            }
        }

        if (AGNheating > 0.0) {
            galaxies[centralgal].Heating += 0.5 * AGNheating * galaxies[centralgal].Vvir * galaxies[centralgal].Vvir;
        }
    }
    return coolingGas;
}

void cool_gas_onto_galaxy(const int centralgal, const double coolingGas, struct GALAXY *galaxies)
{
    // add the fraction 1/STEPS of the total cooling gas to the cold disk
    if(coolingGas > 0.0) {
        if(coolingGas < galaxies[centralgal].HotGas) {
            const double metallicity = get_metallicity(galaxies[centralgal].HotGas, galaxies[centralgal].MetalsHotGas);
            galaxies[centralgal].ColdGas += coolingGas;
            galaxies[centralgal].MetalsColdGas += metallicity * coolingGas;
            galaxies[centralgal].HotGas -= coolingGas;
            galaxies[centralgal].MetalsHotGas -= metallicity * coolingGas;
        } else {
            galaxies[centralgal].ColdGas += galaxies[centralgal].HotGas;
            galaxies[centralgal].MetalsColdGas += galaxies[centralgal].MetalsHotGas;
            galaxies[centralgal].HotGas = 0.0;
            galaxies[centralgal].MetalsHotGas = 0.0;
        }
    }
}

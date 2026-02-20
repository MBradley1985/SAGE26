#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "core_allvars.h"

#include "model_reincorporation.h"
#include "model_misc.h"

void reincorporate_gas(const int centralgal, const double dt, struct GALAXY *galaxies, const struct params *run_params)
{
    // SN velocity is 630km/s, and the condition for reincorporation is that the
    // halo has an escape velocity greater than this, i.e. V_SN/sqrt(2) = 445.48km/s
    const double Vcrit = 445.48 * run_params->ReIncorporationFactor;

    // BUG FIX: Also check Rvir > 0 to avoid division issues
    if(galaxies[centralgal].Vvir > Vcrit && galaxies[centralgal].Rvir > 0.0) {
        // Note: Vvir > Vcrit already ensures Vvir > 0, so Rvir/Vvir is safe
        double reincorporated =
            ( galaxies[centralgal].Vvir / Vcrit - 1.0 ) *
            galaxies[centralgal].EjectedMass / (galaxies[centralgal].Rvir / galaxies[centralgal].Vvir) * dt;

        if(reincorporated > galaxies[centralgal].EjectedMass)
            reincorporated = galaxies[centralgal].EjectedMass;

        const double metallicity = get_metallicity(galaxies[centralgal].EjectedMass, galaxies[centralgal].MetalsEjectedMass);
        
        // Remove from ejected reservoir (same for all regimes)
        galaxies[centralgal].EjectedMass -= reincorporated;
        galaxies[centralgal].MetalsEjectedMass -= metallicity * reincorporated;
        if(galaxies[centralgal].MetalsEjectedMass < 0.0) galaxies[centralgal].MetalsEjectedMass = 0.0;
        
        // Add to appropriate hot reservoir (regime-dependent)
        if(run_params->FountainGasOn == 1) {
            // FountainGas mode: EjectedMass → FountainGas (not directly to HotGas)
            // Update FountainTime with weighted average
            const double new_fountain_time = 1.0;  // ~1 Gyr
            if(galaxies[centralgal].FountainGas > 0.0) {
                double total = galaxies[centralgal].FountainGas + reincorporated;
                galaxies[centralgal].FountainTime =
                    (galaxies[centralgal].FountainGas * galaxies[centralgal].FountainTime +
                     reincorporated * new_fountain_time) / total;
            } else {
                galaxies[centralgal].FountainTime = new_fountain_time;
            }
            galaxies[centralgal].FountainGas += reincorporated;
            galaxies[centralgal].MetalsFountainGas += metallicity * reincorporated;
        } else if(run_params->CGMrecipeOn == 1) {
            if(galaxies[centralgal].Regime == 0) {
                // CGM-regime: reincorporate to CGM
                galaxies[centralgal].CGMgas += reincorporated;
                galaxies[centralgal].MetalsCGMgas += metallicity * reincorporated;
            } else {
                // Hot-ICM-regime: reincorporate to HotGas
                galaxies[centralgal].HotGas += reincorporated;
                galaxies[centralgal].MetalsHotGas += metallicity * reincorporated;
            }
        } else {
            // Original SAGE behavior: reincorporate to HotGas
            galaxies[centralgal].HotGas += reincorporated;
            galaxies[centralgal].MetalsHotGas += metallicity * reincorporated;
        }

        if(run_params->DustOn == 1) {
            const double DTG = get_DTG(galaxies[centralgal].EjectedMass, galaxies[centralgal].EjectedDust);
            double reinc_dust = DTG * reincorporated;
            if(reinc_dust > galaxies[centralgal].EjectedDust) reinc_dust = galaxies[centralgal].EjectedDust;
            galaxies[centralgal].EjectedDust -= reinc_dust;

            if(run_params->FountainGasOn == 1) {
                // FountainGas mode: reincorporate to FountainDust
                galaxies[centralgal].FountainDust += reinc_dust;
                if(galaxies[centralgal].FountainDust > galaxies[centralgal].MetalsFountainGas)
                    galaxies[centralgal].FountainDust = galaxies[centralgal].MetalsFountainGas;
            } else if(run_params->CGMrecipeOn == 1 && galaxies[centralgal].Regime == 0) {
                // CGM-regime: reincorporate dust to CGMDust
                galaxies[centralgal].CGMDust += reinc_dust;
                if(galaxies[centralgal].CGMDust > galaxies[centralgal].MetalsCGMgas)
                    galaxies[centralgal].CGMDust = galaxies[centralgal].MetalsCGMgas;
            } else {
                // Hot-ICM-regime or original: reincorporate dust to HotDust
                galaxies[centralgal].HotDust += reinc_dust;
                if(galaxies[centralgal].HotDust > galaxies[centralgal].MetalsHotGas)
                    galaxies[centralgal].HotDust = galaxies[centralgal].MetalsHotGas;
            }
        }
    }
}


/* ========================================================================== */
/* FOUNTAIN GAS REINCORPORATION                                               */
/* DarkSage-style gas reservoir cycling                                       */
/* ========================================================================== */

void reincorporate_fountain_gas(const int centralgal, const double dt, struct GALAXY *galaxies,
                                const struct params *run_params)
{
    /* Only run if FountainGasOn is enabled */
    if(run_params->FountainGasOn != 1) {
        return;
    }

    /* ================================================================== */
    /* 1. FountainGas → HotGas (on FountainTime timescale)                */
    /* ================================================================== */
    if(galaxies[centralgal].FountainGas > 0.0 && galaxies[centralgal].FountainTime > 0.0) {
        /* Exponential decay: amount = FountainGas * (1 - exp(-dt/FountainTime)) */
        /* For small dt/FountainTime, this ≈ FountainGas * dt / FountainTime */
        double reinc_frac = dt / galaxies[centralgal].FountainTime;
        if(reinc_frac > 1.0) reinc_frac = 1.0;

        double reincorporated_fountain = reinc_frac * galaxies[centralgal].FountainGas;
        if(reincorporated_fountain > galaxies[centralgal].FountainGas) {
            reincorporated_fountain = galaxies[centralgal].FountainGas;
        }

        const double metallicity_fountain = get_metallicity(galaxies[centralgal].FountainGas,
                                                            galaxies[centralgal].MetalsFountainGas);

        /* Transfer from FountainGas to HotGas */
        galaxies[centralgal].FountainGas -= reincorporated_fountain;
        galaxies[centralgal].MetalsFountainGas -= metallicity_fountain * reincorporated_fountain;
        if(galaxies[centralgal].MetalsFountainGas < 0.0) galaxies[centralgal].MetalsFountainGas = 0.0;

        galaxies[centralgal].HotGas += reincorporated_fountain;
        galaxies[centralgal].MetalsHotGas += metallicity_fountain * reincorporated_fountain;

        /* Handle dust */
        if(run_params->DustOn == 1) {
            const double DTG_fountain = get_DTG(galaxies[centralgal].FountainGas,
                                                galaxies[centralgal].FountainDust);
            double reinc_fountain_dust = DTG_fountain * reincorporated_fountain;
            if(reinc_fountain_dust > galaxies[centralgal].FountainDust) {
                reinc_fountain_dust = galaxies[centralgal].FountainDust;
            }
            galaxies[centralgal].FountainDust -= reinc_fountain_dust;
            galaxies[centralgal].HotDust += reinc_fountain_dust;
            if(galaxies[centralgal].HotDust > galaxies[centralgal].MetalsHotGas) {
                galaxies[centralgal].HotDust = galaxies[centralgal].MetalsHotGas;
            }
        }
    }

    /* ================================================================== */
    /* 2. OutflowGas → EjectedMass (on OutflowTime timescale)             */
    /* ================================================================== */
    if(galaxies[centralgal].OutflowGas > 0.0 && galaxies[centralgal].OutflowTime > 0.0) {
        double reinc_frac = dt / galaxies[centralgal].OutflowTime;
        if(reinc_frac > 1.0) reinc_frac = 1.0;

        double transferred_outflow = reinc_frac * galaxies[centralgal].OutflowGas;
        if(transferred_outflow > galaxies[centralgal].OutflowGas) {
            transferred_outflow = galaxies[centralgal].OutflowGas;
        }

        const double metallicity_outflow = get_metallicity(galaxies[centralgal].OutflowGas,
                                                           galaxies[centralgal].MetalsOutflowGas);

        /* Transfer from OutflowGas to EjectedMass */
        galaxies[centralgal].OutflowGas -= transferred_outflow;
        galaxies[centralgal].MetalsOutflowGas -= metallicity_outflow * transferred_outflow;
        if(galaxies[centralgal].MetalsOutflowGas < 0.0) galaxies[centralgal].MetalsOutflowGas = 0.0;

        galaxies[centralgal].EjectedMass += transferred_outflow;
        galaxies[centralgal].MetalsEjectedMass += metallicity_outflow * transferred_outflow;

        /* Handle dust */
        if(run_params->DustOn == 1) {
            const double DTG_outflow = get_DTG(galaxies[centralgal].OutflowGas,
                                               galaxies[centralgal].OutflowDust);
            double transferred_outflow_dust = DTG_outflow * transferred_outflow;
            if(transferred_outflow_dust > galaxies[centralgal].OutflowDust) {
                transferred_outflow_dust = galaxies[centralgal].OutflowDust;
            }
            galaxies[centralgal].OutflowDust -= transferred_outflow_dust;
            galaxies[centralgal].EjectedDust += transferred_outflow_dust;
            if(galaxies[centralgal].EjectedDust > galaxies[centralgal].MetalsEjectedMass) {
                galaxies[centralgal].EjectedDust = galaxies[centralgal].MetalsEjectedMass;
            }
        }
    }

    /* Safety: clamp negative values */
    if(galaxies[centralgal].FountainGas < 0.0) galaxies[centralgal].FountainGas = 0.0;
    if(galaxies[centralgal].MetalsFountainGas < 0.0) galaxies[centralgal].MetalsFountainGas = 0.0;
    if(galaxies[centralgal].FountainDust < 0.0) galaxies[centralgal].FountainDust = 0.0;
    if(galaxies[centralgal].OutflowGas < 0.0) galaxies[centralgal].OutflowGas = 0.0;
    if(galaxies[centralgal].MetalsOutflowGas < 0.0) galaxies[centralgal].MetalsOutflowGas = 0.0;
    if(galaxies[centralgal].OutflowDust < 0.0) galaxies[centralgal].OutflowDust = 0.0;
}
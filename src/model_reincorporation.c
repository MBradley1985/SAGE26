/*
 * model_reincorporation.c -- return of SN-ejected gas to the hot reservoir.
 *
 * Implements reincorporate_gas(), which moves gas from the EjectedMass
 * reservoir back into the hot gas reservoir once the halo's circular velocity
 * exceeds the supernova re-entry threshold (Vcrit = 445 km/s scaled by
 * ReIncorporationFactor).  With CGMrecipeOn, the destination reservoir is
 * regime-dependent: CGMgas for CGM-regime haloes, HotGas otherwise.
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "core_allvars.h"

#include "model_reincorporation.h"
#include "model_misc.h"

/* -------------------------------------------------------------------------
 * File-scope empirical constants (lifted per STYLE_C.md SS8).
 * -------------------------------------------------------------------------*/

/* Critical velocity for gas reincorporation.
 * SN ejecta leave the disk at V_SN ~ 501 km/s; a halo can retain the gas when
 * its escape velocity (V_esc = sqrt(2)*V_vir) exceeds V_SN, i.e. when
 * V_vir > V_SN / sqrt(2) ~ 354.53 km/s.  Scaled further by
 * ReIncorporationFactor in the parameter file. */
// static const double REINC_VCRIT_KMS = 445.48;  /* km/s */
static const double REINC_VCRIT_KMS = 354.53;  /* km/s */

/*
 * reincorporate_gas -- move ejected gas back into the hot reservoir for one
 * central galaxy over timestep dt.
 *
 * Reincorporation rate: (Vvir/Vcrit - 1) * EjectedMass / t_dyn, where
 * t_dyn = Rvir/Vvir.  Only runs when Vvir > Vcrit.  With CGMrecipeOn, routes
 * to CGMgas (Regime 0) or HotGas (Regime 1); without it, always to HotGas.
 */
void reincorporate_gas(const int centralgal, const double dt, struct GALAXY *galaxies, const struct params *run_params)
{
    const double Vcrit = REINC_VCRIT_KMS * run_params->ReIncorporationFactor;

    if(Vcrit > 0.0 && galaxies[centralgal].Vvir > Vcrit && galaxies[centralgal].Rvir > 0.0) {
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
        
        if(run_params->DarkSAGEMode == 1 && !(run_params->CGMrecipeOn == 1 && galaxies[centralgal].Regime == 0)) {
            // DarkSAGEMode, hot regime: EjectedMass -> FountainGas -> HotGas (reincorporate_fountain_gas).
            // FountainTime is the mass-weighted dynamical time at which the gas joined the fountain.
            const double t_dyn = get_dynamical_time(galaxies[centralgal].SnapNum, run_params);
            if(galaxies[centralgal].FountainGas > 0.0) {
                galaxies[centralgal].FountainTime =
                    (galaxies[centralgal].FountainGas * galaxies[centralgal].FountainTime + reincorporated * t_dyn)
                    / (galaxies[centralgal].FountainGas + reincorporated);
            } else {
                galaxies[centralgal].FountainTime = t_dyn;
            }
            galaxies[centralgal].FountainGas += reincorporated;
            galaxies[centralgal].MetalsFountainGas += metallicity * reincorporated;
        } else {
            // Add to appropriate hot reservoir (regime-dependent); CGM haloes have no fountain
            add_gas_to_hot_reservoir(&galaxies[centralgal], run_params, reincorporated, metallicity * reincorporated);
        }
    }
}

/*
 * reincorporate_fountain_gas -- DarkSAGEMode reservoir cycling for one central over dt.
 *
 * FountainGas returns to the hot reservoir (CGMgas for Regime 0 under
 * CGMrecipeOn) at rate FountainGas / FountainTime; OutflowGas reaches
 * EjectedMass at rate OutflowGas / OutflowTime.
 */
void reincorporate_fountain_gas(const int centralgal, const double dt, struct GALAXY *galaxies, const struct params *run_params)
{
    if(run_params->DarkSAGEMode != 1) return;

    // FountainGas -> HotGas / CGMgas
    if(galaxies[centralgal].FountainGas > 0.0 && galaxies[centralgal].FountainTime > 0.0) {
        double frac = dt / galaxies[centralgal].FountainTime;
        if(frac > 1.0) frac = 1.0;
        const double returned = frac * galaxies[centralgal].FountainGas;
        const double metallicity = get_metallicity(galaxies[centralgal].FountainGas, galaxies[centralgal].MetalsFountainGas);

        galaxies[centralgal].FountainGas -= returned;
        galaxies[centralgal].MetalsFountainGas -= metallicity * returned;
        add_gas_to_hot_reservoir(&galaxies[centralgal], run_params, returned, metallicity * returned);
    }

    // OutflowGas -> EjectedMass
    if(galaxies[centralgal].OutflowGas > 0.0 && galaxies[centralgal].OutflowTime > 0.0) {
        double frac = dt / galaxies[centralgal].OutflowTime;
        if(frac > 1.0) frac = 1.0;
        const double transferred = frac * galaxies[centralgal].OutflowGas;
        const double metallicity = get_metallicity(galaxies[centralgal].OutflowGas, galaxies[centralgal].MetalsOutflowGas);

        galaxies[centralgal].OutflowGas -= transferred;
        galaxies[centralgal].MetalsOutflowGas -= metallicity * transferred;
        galaxies[centralgal].EjectedMass += transferred;
        galaxies[centralgal].MetalsEjectedMass += metallicity * transferred;
    }

    if(galaxies[centralgal].FountainGas < 0.0f) galaxies[centralgal].FountainGas = 0.0f;
    if(galaxies[centralgal].MetalsFountainGas < 0.0f) galaxies[centralgal].MetalsFountainGas = 0.0f;
    if(galaxies[centralgal].OutflowGas < 0.0f) galaxies[centralgal].OutflowGas = 0.0f;
    if(galaxies[centralgal].MetalsOutflowGas < 0.0f) galaxies[centralgal].MetalsOutflowGas = 0.0f;
}
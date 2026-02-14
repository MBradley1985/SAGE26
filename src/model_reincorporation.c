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
        if(run_params->CGMrecipeOn == 1) {
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

            if(run_params->CGMrecipeOn == 1 && galaxies[centralgal].Regime == 0) {
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
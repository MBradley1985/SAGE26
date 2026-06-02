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
 * SN ejecta leave the disk at V_SN ~ 630 km/s; a halo can retain the gas when
 * its escape velocity (V_esc = sqrt(2)*V_vir) exceeds V_SN, i.e. when
 * V_vir > V_SN / sqrt(2) ~ 445.48 km/s.  Scaled further by
 * ReIncorporationFactor in the parameter file. */
static const double REINC_VCRIT_KMS = 445.48;  /* km/s */

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
    }
}
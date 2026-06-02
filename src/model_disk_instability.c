/*
 * model_disk_instability.c -- Toomre disk-instability channel for bulge growth.
 *
 * Implements check_disk_instability(), which evaluates the Mo, Mao & White
 * (1998) instability criterion each timestep.  When the disk (cold gas + disk
 * stars) exceeds the Toomre critical mass M_crit = Vmax^2 * 3*R_d / G, the
 * excess stellar mass is transferred to the bulge (tracking instability-origin
 * bulge mass and radius via update_instability_bulge_radius()) and the excess
 * gas is consumed via a collisional starburst, with a fraction feeding the
 * central black hole if AGNrecipeOn > 0.
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "core_allvars.h"

#include "model_disk_instability.h"
#include "model_misc.h"
#include "model_mergers.h"

/* -------------------------------------------------------------------------
 * File-scope empirical constants (lifted per STYLE_C.md SS8).
 * -------------------------------------------------------------------------*/

/* Numerical pre-factor in the Mo, Mao & White (1998) Toomre critical mass:
 * M_crit = V_max^2 * (TOOMRE_DISK_FACTOR * R_d) / G.
 * The factor of 3 comes from the disk half-mass radius being ~1.68 r_s and
 * the MW98 stability analysis for an exponential disk. */
static const double TOOMRE_DISK_FACTOR = 3.0;

/*
 * check_disk_instability -- apply the Toomre instability criterion and
 * redistribute unstable mass for galaxy p at the current timestep.
 *
 * Computes M_crit from Vmax and DiskScaleRadius; any excess disk stars are
 * moved to InstabilityBulgeMass (with an incremental bulge radius update per
 * Tonini+ 2016 eq. 15), and any excess gas is burst via
 * collisional_starburst_recipe().  Saves DiskScaleRadius before any mass
 * transfer so the Tonini formula uses the pre-instability disc scale.
 */
void check_disk_instability(const int p, const int centralgal, const int halonr, const double time, const double dt, const int step,
                            struct GALAXY *galaxies, struct params *run_params)
{
    // Here we calculate the stability of the stellar and gaseous disk as discussed in Mo, Mao & White (1998).
    // For unstable stars and gas, we transfer the required ammount to the bulge to make the disk stable again

    // Disk mass has to be > 0.0
    const double diskmass = galaxies[p].ColdGas + (galaxies[p].StellarMass - galaxies[p].BulgeMass);
    if(diskmass > 0.0) {
        // calculate critical disk mass
        double Mcrit = galaxies[p].Vmax * galaxies[p].Vmax * (TOOMRE_DISK_FACTOR * galaxies[p].DiskScaleRadius) / run_params->G;
        if(Mcrit > diskmass) {
            Mcrit = diskmass;
        }

        // use disk mass here
        const double gas_fraction   = galaxies[p].ColdGas / diskmass;
        const double unstable_gas   = gas_fraction * (diskmass - Mcrit);
        const double star_fraction  = 1.0 - gas_fraction;
        const double unstable_stars = star_fraction * (diskmass - Mcrit);

        // CRITICAL: Save the disc radius BEFORE any mass transfers or updates
        // This is the R_D used in Tonini+2016 equation (15)
        const double old_disk_radius = galaxies[p].DiskScaleRadius;

        // add excess stars to the bulge
        if(unstable_stars > 0.0) {
            // Use disk metallicity here
            const double metallicity = get_metallicity(galaxies[p].StellarMass - galaxies[p].BulgeMass, galaxies[p].MetalsStellarMass - galaxies[p].MetalsBulgeMass);

            galaxies[p].BulgeMass += unstable_stars;
            galaxies[p].InstabilityBulgeMass += unstable_stars;  // Track origin of bulge mass
            galaxies[p].MetalsBulgeMass += metallicity * unstable_stars;
            
            // UPDATE: Tonini incremental radius evolution (equation 15)
            // Pass the OLD disc radius explicitly to ensure we use pre-instability value
            update_instability_bulge_radius(p, unstable_stars, old_disk_radius, galaxies, run_params);
        }

        // Disc scale radius is unchanged by instability: the remaining disk retains the
        // same specific angular momentum per unit mass as before, so r_d stays constant.

        // burst excess gas and feed black hole (really need a dedicated model for bursts and BH growth here)
        if(unstable_gas > 0.0 && galaxies[p].ColdGas > 0.0) {
#ifdef VERBOSE
            if(unstable_gas > 1.0001 * galaxies[p].ColdGas ) {
                fprintf(stdout, "unstable_gas > galaxies[p].ColdGas\t%e\t%e\n", unstable_gas, galaxies[p].ColdGas);
                run_params->interrupted = 1;
            }
#endif

            const double unstable_gas_fraction = unstable_gas / galaxies[p].ColdGas;
            if(run_params->AGNrecipeOn > 0) {
                grow_black_hole(p, unstable_gas_fraction, galaxies, run_params);
            }

            collisional_starburst_recipe(unstable_gas_fraction, p, centralgal, time, dt, halonr, 1, step, 
                             0, galaxies[p].DiskScaleRadius,  // burst_to_merger_bulge=0, use current disc radius
                             galaxies, run_params);
        }
    }
}
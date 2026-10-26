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
/*
 * FixingGas 5 -- disk instabilities (Lagos et al. 2018, Section 4.4.8): eq. (35),
 * eps = V_circ / sqrt(1.68 G M_disk / r_disk), with V_circ the maximum circular
 * velocity, r_disk the disk half-mass radius and M_disk the disk mass (gas plus
 * stars).  eps < eps_disc = 0.8 (Table 4) is unstable: the disk's stars and gas go
 * to the spheroid and the gas drives a starburst (shark_starburst in
 * model_starformation_and_feedback.c).  Checked once per snapshot, before the
 * galaxy evolves, as the shark code does (shark_runner.cpp).  Called with any bulge
 * gas set aside, so ColdGas here is the disk gas.
 */
static const double SHARK_DI_STABLE = 0.8;

static void shark_disk_instability(const int p, struct GALAXY *galaxies, const struct params *run_params)
{
    /* shark evaluates disk instabilities once per snapshot, before the galaxy evolves
     * (shark_runner.cpp: mergers, then evaluate_disk_instability, then evolve_galaxy) */
    if(galaxies[p].SharkDISnap == galaxies[p].SnapNum) {
        return;
    }
    galaxies[p].SharkDISnap = galaxies[p].SnapNum;
    const double disk_stars = galaxies[p].StellarMass - galaxies[p].BulgeMass;
    const double md = galaxies[p].ColdGas + disk_stars;
    const double rd = 1.678346990 * galaxies[p].DiskScaleRadius;    /* half-mass radius */
    if(md <= 0.0 || rd <= 0.0 || galaxies[p].Vmax <= 0.0) {
        return;
    }
    const double eps = galaxies[p].Vmax / sqrt(1.68 * run_params->G * md / rd);
    if(eps >= SHARK_DI_STABLE) {
        return;
    }

    if(disk_stars > 0.0) {
        const double old_disk_radius = galaxies[p].DiskScaleRadius;
        move_disk_sfh_to_bulge(p, 1.0, galaxies, run_params);
        galaxies[p].InstabilityBulgeMass += disk_stars;
        galaxies[p].BulgeMass = galaxies[p].StellarMass;
        galaxies[p].MetalsBulgeMass = galaxies[p].MetalsStellarMass;
        update_instability_bulge_radius(p, disk_stars, old_disk_radius, galaxies, run_params);
    }
    if(galaxies[p].ColdGas > 0.0) {
        if(run_params->AGNrecipeOn > 0) {
            grow_black_hole(p, 1.0, galaxies, run_params);
        }
        galaxies[p].BulgeColdGas = galaxies[p].ColdGas;
        galaxies[p].MetalsBulgeColdGas = galaxies[p].MetalsColdGas;
        galaxies[p].BulgeGasOrigin = 2;
    }
}

void check_disk_instability(const int p, const int centralgal, const int halonr, const double time, const double dt, const int step,
                            struct GALAXY *galaxies, const struct params *run_params)
{
    if(FG_SHARK_BURSTS(run_params)) {
        shark_disk_instability(p, galaxies, run_params);
        return;
    }

    // Here we calculate the stability of the stellar and gaseous disk as discussed in Mo, Mao & White (1998).
    // For unstable stars and gas, we transfer the required amount to the bulge to make the disk stable again

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

            // The moved stars take their share of the disk's star-formation history with them
            const double disk_stars = galaxies[p].StellarMass - galaxies[p].BulgeMass;
            if(disk_stars > 0.0) {
                move_disk_sfh_to_bulge(p, unstable_stars / disk_stars, galaxies, run_params);
            }

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
                /* progressbar redraw flag -- the only mutation of run_params in the
                   physics modules, so the const is cast away here rather than
                   forcing a non-const signature onto every caller. */
                ((struct params *) run_params)->interrupted = 1;
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
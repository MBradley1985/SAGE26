#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "core_allvars.h"

#include "model_disk_instability.h"
#include "model_misc.h"
#include "model_mergers.h"

// Forward declaration for SHARK-style instability
static void shark_style_disk_instability(const int p, const int centralgal, const int halonr,
                                         const double time, const double dt, const int step,
                                         struct GALAXY *galaxies, struct params *run_params);

void check_disk_instability(const int p, const int centralgal, const int halonr, const double time, const double dt, const int step,
                            struct GALAXY *galaxies, struct params *run_params)
{
    // Here we calculate the stability of the stellar and gaseous disk as discussed in Mo, Mao & White (1998).
    // For unstable stars and gas, we transfer the required ammount to the bulge to make the disk stable again

    // Disk mass has to be > 0.0
    const double diskmass = galaxies[p].ColdGas + (galaxies[p].StellarMass - galaxies[p].BulgeMass);
    if(diskmass <= 0.0) {
        return;
    }

    // Calculate critical disk mass (Toomre stability criterion)
    double Mcrit = galaxies[p].Vmax * galaxies[p].Vmax * (3.0 * galaxies[p].DiskScaleRadius) / run_params->G;
    if(Mcrit > diskmass) {
        Mcrit = diskmass;
    }

    // Check if disk is stable
    if(diskmass <= Mcrit) {
        return;  // Disk is stable, nothing to do
    }

    // Disk is unstable - handle according to mode
    if(run_params->DiskInstabilityMode == 1) {
        // SHARK-style: transfer ALL disk mass to bulge, trigger full starburst
        shark_style_disk_instability(p, centralgal, halonr, time, dt, step, galaxies, run_params);
    } else {
        // Original mode (0): transfer only the unstable fraction

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

#ifdef VERBOSE
            if((galaxies[p].BulgeMass >  1.0001 * galaxies[p].StellarMass)  || (galaxies[p].MetalsBulgeMass >  1.0001 * galaxies[p].MetalsStellarMass)) {
                /* fprintf(stderr, "\nInstability: Mbulge > Mtot (stars or metals)\n"); */
                /* run_params->interrupted = 1; */
            }
#endif
        }

        // CRITICAL: Recalculate disc radius after mass transfer
        // The disc is now less massive, so its scale radius should shrink
        // Use conservation of angular momentum: smaller mass → smaller radius
        if(unstable_stars > 0.0 || unstable_gas > 0.0) {
            // Disc mass after instability
            const double new_diskmass = galaxies[p].ColdGas + (galaxies[p].StellarMass - galaxies[p].BulgeMass);

            if(new_diskmass > 0.0 && diskmass > 0.0) {
                // Simple scaling: R_new = R_old × (M_new / M_old)
                // This conserves specific angular momentum per unit mass
                const double mass_ratio = new_diskmass / diskmass;
                galaxies[p].DiskScaleRadius *= mass_ratio;

                // Safety check: don't let disc radius go to zero or become huge
                if(galaxies[p].DiskScaleRadius < 0.01 * galaxies[p].Rvir) {
                    galaxies[p].DiskScaleRadius = 0.01 * galaxies[p].Rvir;
                }
                if(galaxies[p].DiskScaleRadius > galaxies[p].Rvir) {
                    galaxies[p].DiskScaleRadius = galaxies[p].Rvir;
                }
            } else {
                // Disc has been completely consumed by bulge
                galaxies[p].DiskScaleRadius = 0.0;
            }
        }

        // burst excess gas and feed black hole
        // BUG FIX: Also check ColdGas > 0 to avoid division by zero
        if(unstable_gas > 0.0 && galaxies[p].ColdGas > 0.0) {
#ifdef VERBOSE
            if(unstable_gas > 1.0001 * galaxies[p].ColdGas ) {
                fprintf(stdout, "unstable_gas > galaxies[p].ColdGas\t%e\t%e\n", unstable_gas, galaxies[p].ColdGas);
                run_params->interrupted = 1;
            }
#endif

            const double unstable_gas_fraction = unstable_gas / galaxies[p].ColdGas;
            if(run_params->AGNrecipeOn > 0) {
                grow_black_hole(p, unstable_gas_fraction, 1, galaxies, run_params);
            }

            collisional_starburst_recipe(unstable_gas_fraction, p, centralgal, time, dt, halonr, 1, step,
                             0, galaxies[p].DiskScaleRadius,  // burst_to_merger_bulge=0, use current disc radius
                             galaxies, run_params);
        }
    }
}


/**
 * SHARK-style disk instability handling.
 *
 * When the disk becomes unstable (Toomre parameter < stable threshold):
 * 1. Transfer ALL disk stars to the bulge
 * 2. Feed BH from ALL cold gas (using SHARK's formula)
 * 3. Trigger starburst on ALL remaining cold gas
 * 4. Zero out the disk
 *
 * This avoids the numerical issue where unstable_gas > ColdGas can occur
 * in the original partial transfer approach.
 */
static void shark_style_disk_instability(const int p, const int centralgal, const int halonr,
                                         const double time, const double dt, const int step,
                                         struct GALAXY *galaxies, struct params *run_params)
{
    // Save disk properties before transfer
    const double disk_stars = galaxies[p].StellarMass - galaxies[p].BulgeMass;
    const double disk_metals = galaxies[p].MetalsStellarMass - galaxies[p].MetalsBulgeMass;
    const double cold_gas = galaxies[p].ColdGas;
    const double old_disk_radius = galaxies[p].DiskScaleRadius;

    // 1. Transfer ALL disk stars to bulge
    if(disk_stars > 0.0) {
        galaxies[p].BulgeMass += disk_stars;
        galaxies[p].InstabilityBulgeMass += disk_stars;  // Track origin
        galaxies[p].MetalsBulgeMass += disk_metals;

        // Update bulge radius using Tonini prescription
        update_instability_bulge_radius(p, disk_stars, old_disk_radius, galaxies, run_params);
    }

    // 2. & 3. Feed BH and trigger starburst on ALL cold gas
    if(cold_gas > 0.0) {
        // BH growth: use mass_ratio = 1.0 since we're using all gas
        if(run_params->AGNrecipeOn > 0) {
            grow_black_hole(p, 1.0, 1, galaxies, run_params);
        }

        // Starburst: use mass_ratio = 1.0 for full gas consumption
        // mode=1 means eburst = mass_ratio (direct, not 0.56*mass_ratio^0.7)
        collisional_starburst_recipe(1.0, p, centralgal, time, dt, halonr, 1, step,
                                     0, old_disk_radius,  // burst_to_merger_bulge=0
                                     galaxies, run_params);
    }

    // 4. Zero out disk radius (disk has been completely consumed)
    galaxies[p].DiskScaleRadius = 0.0;
}

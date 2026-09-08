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

#include <math.h>
#include <stdlib.h>

/* Eddington accretion rate formula.
 * L_Edd = 1.3e38 * (M_BH/Msun) erg/s  (Rybicki & Lightman 1979, eq. 1.4.9).
 * Standard AGN radiative efficiency eta = 0.1.
 * C_SQ_KMS2 = c^2 in (km/s)^2; c = 3e5 km/s => c^2 = 9e10. */
static const double EDDINGTON_LUM_PER_MSUN_CGS = 1.3e38;  /* erg/s per Msun */
static const double AGN_RADIATIVE_EFFICIENCY    = 0.1;
static const double C_SQ_KMS2                  = 9.0e10;  /* (km/s)^2 */


double seed_black_hole(const int p, const struct GALAXY *galaxies, const struct params *run_params)
{
    if(run_params->BlackHoleSeedingOn == 0) {
        return 0.0; // No seeding
    }

    // Light seeding from power-law distribution (Ricarte & Natarajan 2018)
    // if(run_params->BlackHoleSeedingOn == 1) {
    //     if(galaxies[p].Mvir > run_params->BHSeedMinHaloMass && galaxies[p].BlackHoleMass <= 0.0) {
            
    //         double seed_mass = 0.0;

    //         // Draw from a power law with bounds 30 M_sun < M_seed < 100 M_sun and slope -0.3
    //         // Following Ricarte & Natarajan 2018
    //         // Power law: p(M) ∝ M^α, where α = -0.3
    //         // We use inverse transform sampling to draw from this distribution
            
    //         double M_min = 30.0;   // Lower bound in solar masses
    //         double M_max = 100.0;  // Upper bound in solar masses
    //         double alpha = -0.3;   // Power law slope
            
    //         // Generate uniform random number in [0, 1)
    //         double u = drand48(); // or use your preferred RNG
            
    //         // Inverse transform for power law sampling
    //         // For α ≠ -1: M = M_min * (1 + u * (M_max^(α+1) / M_min^(α+1) - 1))^(1/(α+1))
    //         // Simplified form:
    //         // M = (M_min^(α+1) + u * (M_max^(α+1) - M_min^(α+1)))^(1/(α+1))
            
    //         double exp = 1.0 / (alpha + 1.0);  // exponent = 1 / (α + 1) = 1 / 0.7 ≈ 1.4286
    //         double M_min_pow = pow(M_min, alpha + 1.0);
    //         double M_max_pow = pow(M_max, alpha + 1.0);
            
    //         seed_mass = pow(M_min_pow + u * (M_max_pow - M_min_pow), exp);

    //         return seed_mass / (1.0e10 / run_params->Hubble_h); // Convert to code units
    //     } 
    // }

    if(run_params->BlackHoleSeedingOn == 1) {
        if(galaxies[p].Mvir > run_params->BHSeedMinHaloMass && galaxies[p].BlackHoleMass <= 0.0) {
            return (1.0e2) / (1.0e10 / run_params->Hubble_h); // Light BH Seeds: constant 10^2 solar masses in code units
        }
    }

    if(run_params->BlackHoleSeedingOn == 2) {
        if(galaxies[p].Mvir > run_params->BHSeedMinHaloMass && galaxies[p].BlackHoleMass <= 0.0) {
            return (1.0e5) / (1.0e10 / run_params->Hubble_h); // Heavy BH Seeds: constant 10^5 solar masses in code units
        }
    }

    return 0.0; // Default fallback
}


// -------------------------------------------------------------------
// Eddington Accretion Rate and Limiter Functions
// -------------------------------------------------------------------

double dynamical_time(const double r_bulge, const double M_bulge_encl, const struct params *run_params)
{
    // Dynamical time calculation: t_dyn ~ r / v_circular
    // where v_circular = sqrt(GM/r)
    
    double rscale = r_bulge / 1.67;
    double vbulge = sqrt(run_params->G * M_bulge_encl / rscale);  // velocity in code units
                                   // length in code units
    double t_dyn = rscale / vbulge;                               // time in code units
    
    // Convert to Megayears for output
    double t_dyn_myr = t_dyn * run_params->UnitTime_in_Megayears;

    if(isnan(t_dyn) || t_dyn <= 0.0) {
        // Fallback: compute from disk scale radius instead
        double r_disk = r_bulge; // or use DiskScaleRadius if available
        if(r_disk <= 0.0) r_disk = 1.0; // Minimum 1 kpc
        t_dyn = r_disk / sqrt(run_params->G * M_bulge_encl / r_disk);
        if(t_dyn <= 0.0 || isnan(t_dyn)) t_dyn = 1.0; // Final fallback
        
        // if(r_bulge<=0 && M_bulge_encl>0){
        // FILE *fp = fopen("tdynbad.txt", "a");
        // if(fp != NULL) {
        //     fprintf(fp, "%g\n", t_dyn);
        //     fclose(fp);
        // }}
    }

    return t_dyn;  // Return in code units for internal use
}

double eddington_accretion_rate(const double black_hole_mass, const struct params *run_params)
{
    // Eddington luminosity: L_Edd = 1.3e38 * M_BH (in Msun) erg/s
    // Convert to code units: divide by UnitEnergy_in_cgs and UnitTime_in_s

    if(black_hole_mass <= 0.0) {
        return 0.0; // No accretion for non-positive mass
    }
    // Eddington-limited accretion (Rybicki & Lightman 1979)
    const double EDDrate = (EDDINGTON_LUM_PER_MSUN_CGS * black_hole_mass * 1e10 / run_params->Hubble_h)
        / (run_params->UnitEnergy_in_cgs / run_params->UnitTime_in_s)
        / (AGN_RADIATIVE_EFFICIENCY * C_SQ_KMS2);
    
    //double EDDrate = (1.3e38 * black_hole_mass * 1e10 / run_params->Hubble_h) / (run_params->UnitEnergy_in_cgs / run_params->UnitTime_in_s) / (0.1 * 9e10);

    return EDDrate;
}

// Accretion rate limiter by Eddington limit
double eddington_limited_accretion_rate(double accretion_rate, int eddington_flag, double black_hole_mass,
                                        int snapnum, int bh_accretion_type, const struct params *run_params,
                                        float BHAccretionType[ABSOLUTEMAXSNAPS], float BHMaxaccretionRate[ABSOLUTEMAXSNAPS],
                                        float BHEddingtonRateLimit[ABSOLUTEMAXSNAPS], float BHMassatAccretion[ABSOLUTEMAXSNAPS])
{
    double edd_rate = 0.0;
    double return_rate = accretion_rate;
    const int valid_snap = (snapnum >= 0 && snapnum < ABSOLUTEMAXSNAPS);
    const int is_seed_bh = (black_hole_mass <= 0.0);

    if(valid_snap) {
        BHAccretionType[snapnum] = (float)bh_accretion_type;
        BHMaxaccretionRate[snapnum]    = (float)accretion_rate;
        BHEddingtonRateLimit[snapnum]  = is_seed_bh ? 0.0f
                                        : (float)eddington_accretion_rate(black_hole_mass, run_params);
        BHMassatAccretion[snapnum]     = (float)black_hole_mass;
    }

    if(is_seed_bh) return accretion_rate;

    edd_rate = (double)BHEddingtonRateLimit[snapnum];

    if(accretion_rate > edd_rate && eddington_flag == 1) {
        return_rate = edd_rate;
        BHMaxaccretionRate[snapnum] = (float)edd_rate;
    }

    return return_rate;
}


static int scenario_disk_only_unlimited(const struct GALAXY *gal, int eddtype,
                                         const struct params *run_params)
{
    // Instability channel runs free; merger channel stays capped.
    return (eddtype == 2) ? 0 : 1;
}

static int scenario_merger_only_unlimited(const struct GALAXY *gal, int eddtype,
                                           const struct params *run_params)
{
    // Merger channel runs free; instability channel stays capped.
    return (eddtype == 1) ? 0 : 1;
}

static int scenario_first_event_unlimited(const struct GALAXY *gal, int eddtype,
                                           const struct params *run_params)
{
    // The first quasar-mode event (merger- or instability-driven, whichever
    // fires first) runs free; every quasar-mode event after that is capped.
    // Deliberately independent of BlackHoleMass/BHSeedMass: the radio-mode
    // channel grows BlackHoleMass every snapshot regardless of scenario, so
    // keying this off BlackHoleMass<=BHSeedMass closed the "first event"
    // window within a snapshot of seeding, before any real quasar-mode event
    // had a chance to run unlimited.
    return gal->QuasarModeEventOccurred ? 1 : 0;
}

// Vvir cuts computed from SAGE's own Bryan & Norman (1998) Rvir relation
// (DELTA_VIRT=200, millennium.par cosmology) at z=0. A fixed Mvir cut
// corresponds to a different physical halo (different Vvir/Tvir) at every
// redshift, since Rvir shrinks with H(z); Vvir is computed per-halo at its
// own redshift, so a fixed Vvir cut tracks the same physical regime at all z.
// The two scenarios run independently (AGNAccretionScheme selects only one
// per simulation), so they need not share a single pivot value.
static const double VVIR_SMALL_HALO_KMS = 75.5;   // Vvir at Mvir=10 code units (~1.4e11 Msun)
static const double VVIR_LARGE_HALO_KMS = 123.5;  // Vvir at Mvir=6e11 Msun (MSHOCK_DB06_MSUN/DEKEL06_M_SHOCK_MSUN)

static int scenario_small_halo_unlimited(const struct GALAXY *gal, int eddtype,
                                          const struct params *run_params)
{
    return (gal->Vvir < VVIR_SMALL_HALO_KMS) ? 0 : 1;
}

static int scenario_massive_halo_unlimited(const struct GALAXY *gal, int eddtype,
                                            const struct params *run_params)
{
    return (gal->Vvir > VVIR_LARGE_HALO_KMS) ? 0 : 1;
}

static int scenario_conc_bulge_unlimited(const struct GALAXY *gal, int eddtype,
                                          const struct params *run_params)
{
    // Bulge-to-total stellar mass ratio above a cut -> unlimited.
    if (gal->StellarMass <= 0.0) return 1;
    const double bulge_frac = gal->BulgeMass / gal->StellarMass;
    return (bulge_frac > 0.7) ? 0 : 1;
}

static int scenario_minor_merger_unlimited(const struct GALAXY *gal, double mass_ratio,
                                            const struct params *run_params)
{
    return (mass_ratio <= run_params->ThreshMajorMerger) ? 0 : 1;
}

static int scenario_major_merger_unlimited(const struct GALAXY *gal, double mass_ratio,
                                            const struct params *run_params)
{
    return (mass_ratio > run_params->ThreshMajorMerger) ? 0 : 1;
}

static int scenario_disk_dominated_unlimited(const struct GALAXY *gal, int eddtype,
                                          const struct params *run_params)
{
    // Bulge-to-total stellar mass ratio above a cut -> unlimited.
    if (gal->StellarMass <= 0.0) return 1;
    const double bulge_frac = gal->BulgeMass / gal->StellarMass;
    return (bulge_frac < 0.3) ? 0 : 1;
}

int accretion_scenario(int scenario_id, const struct GALAXY *gal,
                        int eddtype, double mass_ratio, const struct params *run_params)
{
    switch (scenario_id) {
        case 0: return run_params->EddingtonLimitOn;          // current global behaviour: capped everywhere
        case 1: return scenario_disk_only_unlimited(gal, eddtype, run_params);
        case 2: return scenario_merger_only_unlimited(gal, eddtype, run_params);
        case 3: return scenario_first_event_unlimited(gal, eddtype, run_params);
        case 4: return scenario_small_halo_unlimited(gal, eddtype, run_params);
        case 5: return scenario_massive_halo_unlimited(gal, eddtype, run_params);
        case 6: return scenario_conc_bulge_unlimited(gal, eddtype, run_params);
        case 7: return scenario_minor_merger_unlimited(gal, mass_ratio, run_params);
        case 8: return scenario_major_merger_unlimited(gal, mass_ratio, run_params);
        case 9: return scenario_disk_dominated_unlimited(gal, eddtype, run_params);
        default: return run_params->EddingtonLimitOn;         // safe fallback
    }
}


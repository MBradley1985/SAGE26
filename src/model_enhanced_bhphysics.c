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
    // const double EDDrate = (EDDINGTON_LUM_PER_MSUN_CGS * black_hole_mass * 1e10 / run_params->Hubble_h)
    //     / (run_params->UnitEnergy_in_cgs / run_params->UnitTime_in_s)
    //     / (AGN_RADIATIVE_EFFICIENCY * C_SQ_KMS2);
    
    double EDDrate = (1.3e38 * black_hole_mass * 1e10 / run_params->Hubble_h) / (run_params->UnitEnergy_in_cgs / run_params->UnitTime_in_s) / (0.1 * 9e10);

    return EDDrate;
}

// Accretion rate limiter by Eddington limit
double eddington_limited_accretion_rate(double accretion_rate, int eddington_flag, double black_hole_mass,
                                        int snapnum, int bh_accretion_type, const struct params *run_params,
                                        float BHAccretionType[ABSOLUTEMAXSNAPS], float BHMaxaccretionRate[ABSOLUTEMAXSNAPS], float BHEddingtonRateLimit[ABSOLUTEMAXSNAPS])
{
    double edd_rate = 0.0;
    double return_rate = accretion_rate;
    const int valid_snap = (snapnum >= 0 && snapnum < ABSOLUTEMAXSNAPS);
    const int is_seed_bh = (black_hole_mass <= 0.0);

    // Store the accretion type for diagnostics
    if(valid_snap) {
        BHAccretionType[snapnum] = (float)bh_accretion_type;
        //printf("DEBUG: Snapnum = %d, BH Accretion Type = %d\n", snapnum, bh_accretion_type);
    }

    if (accretion_rate > 0.0) {
        // Store the unlimited rate for diagnostics before any limit is applied.
        if(valid_snap) {
            BHMaxaccretionRate[snapnum] = (float)accretion_rate;
        }

        if(is_seed_bh) {
            // Seed black holes accrete without Eddington limiting.
            if(valid_snap) {
                BHEddingtonRateLimit[snapnum] = 0.0f;
            }
            return accretion_rate;
        }

        // Calculate Eddington accretion rate 
        edd_rate = eddington_accretion_rate(black_hole_mass, run_params);
        if(valid_snap) {
            BHEddingtonRateLimit[snapnum] = (float)edd_rate;
        }

        // If accretion exceeds Eddington limit and flag is set, apply the limit
        if (accretion_rate > edd_rate && eddington_flag == 1) {
            return_rate = edd_rate;
            BHMaxaccretionRate[snapnum] = (float)return_rate; // Update to the limited rate for diagnostics
        }
    }

    return return_rate;
}
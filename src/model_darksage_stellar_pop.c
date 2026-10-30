/*
 * model_darksage_stellar_pop.c -- stellar age bins (DarkSAGEMode == 1).
 *
 * Stars in each disc annulus, the two bulges and the intrahalo stars are kept
 * in N_AGE_BINS bins of formation (lookback) time, as in DarkSage (Stevens et
 * al. 2024 Sec. 5).  The bins are moved with the stars wherever they change
 * component; SFprescription 8 reads the recent star formation from them and
 * SupernovaRecipeOn 2 returns mass and SN energy from them (delayed feedback).
 *
 * The mass returned and the supernovae per unit mass formed between stellar
 * ages t0 and t1 come from DarkSage's analytic Chabrier-like IMF integrals:
 * stars of initial mass m live t(m) = 10 Gyr m^-2.5, stars below 1 Msun and
 * above 50 Msun neither return mass nor explode, stars above 8 Msun explode
 * as type II supernovae and those of 1-8 Msun as type Ia with a number ratio
 * Ia/II = 0.15.
 *
 * Code units throughout (10^10 Msun/h, Mpc/h, km/s, (Mpc/h)/(km/s)).
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "core_allvars.h"
#include "macros.h"

#include "model_darksage_stellar_pop.h"
#include "model_misc.h"
#include "model_darksage_feedback.h"

/* DarkSage MIN_STARFORMATION: below this returned mass an annulus has no delayed outflow [10^10 Msun/h] */
static const double DS_MIN_RETURNED = 1.0e-10;
/* DarkSage number ratio of type Ia to type II supernovae (main.c Ratio_Ia_II) */
static const double DS_RATIO_IA_II = 0.15;
/* Stellar mass limits of the IMF integrals [Msun]: no mass return below 1, collapse to a black
 * hole above 50, IMF truncated at 0.1 and 100 */
static const double IMF_M_LOW = 0.1, IMF_M_RETURN = 1.0, IMF_M_BH = 50.0, IMF_M_HIGH = 100.0;
/* Integral of m phi(m) dm over [m0, m1] [Msun] */
static double integrate_m_IMF(const double m0, double m1)
{
    if(m1 > IMF_M_HIGH) m1 = IMF_M_HIGH;
    if(m0 < 1.0) return 0.405 - 0.7945925666666667 * (pow(m1, -0.3) - 1.0);
    return -0.7945925666666667 * (pow(m1, -0.3) - pow(m0, -0.3));
}

/* Indefinite integrals of m_remnant phi(m), in four mass pieces (DarkSage indef_integral_mremnant_IMF()) */
static double indef_integral_mremnant_IMF(const double m, const int piece)
{
    if(piece == 4) return -0.7945925666666667 * pow(m, -0.3);
    if(piece == 1) return -0.08141517683076922 * pow(m, -1.3) - 0.0667457756 * pow(m, -0.3);
    if(piece == 2) return -0.07683098894615384 * pow(m, -1.3) - 0.08661058976666666 * pow(m, -0.3);
    return -0.2567145215384615 * pow(m, -1.3);
}

/* Integral of m_remnant phi(m) dm over [m0, m1] */
static double integrate_mremnant_IMF(const double m0, const double m1)
{
    double sum = 0.0;
    if(m0 < 1.0) sum += integrate_m_IMF(fmax(IMF_M_LOW, m0), fmin(1.0, m1));
    if(m1 <= 1.0) return sum;
    if(m0 < 7.0) sum += indef_integral_mremnant_IMF(fmin(7.0, m1), 1) - indef_integral_mremnant_IMF(fmax(1.0, m0), 1);
    if(m1 <= 7.0) return sum;
    if(m0 < 8.0) sum += indef_integral_mremnant_IMF(fmin(8.0, m1), 2) - indef_integral_mremnant_IMF(fmax(7.0, m0), 2);
    if(m1 <= 8.0) return sum;
    if(m0 < IMF_M_BH) sum += indef_integral_mremnant_IMF(fmin(IMF_M_BH, m1), 3) - indef_integral_mremnant_IMF(fmax(8.0, m0), 3);
    if(m1 <= IMF_M_BH) return sum;
    return sum + integrate_m_IMF(fmax(IMF_M_BH, m0), fmin(IMF_M_HIGH, m1));
}

/* Supernovae from stars of initial mass [m0, m1] per unit mass formed (before normalisation) */
static double num_sn_per_mass(double m0, double m1)
{
    if(m0 < 1.0) m0 = 1.0;
    if(m1 > IMF_M_BH) m1 = IMF_M_BH;
    double sum = 0.0;
    if(m0 < 8.0) sum += 0.01194933634822933 * DS_RATIO_IA_II * (pow(fmax(1.0, m0), -1.3) - pow(fmin(8.0, m1), -1.3));
    if(m1 <= 8.0) return sum;
    return sum + 0.18336751538461538 * (pow(fmax(8.0, m0), -1.3) - pow(fmin(IMF_M_BH, m1), -1.3));
}

void darksage_return_and_sn(const double t0, const double t1, const struct params *run_params, double out[2])
{
    out[0] = out[1] = 0.0;
    if(!(t1 > t0)) return;

    /* stellar lifetime t = 10 Gyr m^-2.5 inverted for the masses dying between ages t0 and t1 */
    const double to_10gyr = run_params->UnitTime_in_s / SEC_PER_MEGAYEAR * 1.0e-4 / run_params->Hubble_h;
    double m0 = 1.0 / pow(t1 * to_10gyr, 0.4);
    if(m0 > IMF_M_BH) return;
    if(m0 < IMF_M_RETURN) m0 = IMF_M_RETURN;
    double m1 = IMF_M_BH;
    if(t0 > 0.0) {
        m1 = 1.0 / pow(t0 * to_10gyr, 0.4);
        if(m1 > IMF_M_BH) m1 = IMF_M_BH;
        if(m1 <= IMF_M_RETURN) return;
    }
    if(!(m1 > m0)) return;

    const double denom = integrate_m_IMF(IMF_M_LOW, m1) + integrate_mremnant_IMF(m1, IMF_M_HIGH);
    out[0] = (integrate_m_IMF(m0, m1) - integrate_mremnant_IMF(m0, m1)) / denom;
    out[1] = num_sn_per_mass(m0, m1) / (denom * SOLAR_MASS * run_params->Hubble_h) * run_params->UnitMass_in_g;
}

/*
 * HI photoionisation rate of the UV background, Faucher-Giguere (2020), at z = 0, 0.2, ..., 8.8
 * [1e-12 s^-1] (DarkSage main.c GammaHI_arr), looked up as DarkSage does: from the first entry
 * at or above z, along the segment above it.  Source bug: DarkSage fills its redshift grid only
 * up to z = 6 (31 entries) while reading 45, so above z = 6 it reads unset memory; the table's
 * own 0.2 spacing is used for all 45 here.
 */
static const double FG20_GAMMA_HI[45] = {
    3.62e-2, 8.72e-2, 0.177, 0.312, 0.488, 0.682, 0.862, 1.003, 1.091, 1.136, 1.136, 1.109, 1.069, 1.021, 0.968,
    0.915, 0.857, 0.805, 0.757, 0.711, 0.670, 0.631, 0.592, 0.558, 0.525, 0.493, 0.463, 0.436, 0.409, 0.384,
    0.356, 0.329, 0.304, 0.279, 0.257, 0.237, 0.167, 1.25e-2, 1.44e-4, 2.80e-5, 4.26e-6, 2.81e-7, 1.24e-9, 1e-28, 1e-28};
static const double FG20_DZ = 0.2;

void darksage_init_age_bins(struct params *run_params)
{
    for(int s = 0; s < run_params->Snaplistlen && s < ABSOLUTEMAXSNAPS; s++) {
        const double z = run_params->ZZ[s];
        int i = 0;
        while(i < 45 && FG20_DZ * i < z) i++;
        double g;
        if(i >= 44) g = FG20_GAMMA_HI[44];
        else g = FG20_GAMMA_HI[i] + (FG20_GAMMA_HI[i+1] - FG20_GAMMA_HI[i]) * (z - FG20_DZ * i) / FG20_DZ;
        run_params->GammaHI[s] = g * 1.0e-12 * run_params->UnitTime_in_s / run_params->Hubble_h;
    }

    /* Bin edges in lookback time: the latest output snapshot, then snapshots spaced evenly back
     * to the first (DarkSage main.c) */
    const int last = run_params->ListOutputSnaps[0];
    if(N_AGE_BINS == last) {
        for(int i = 0; i < N_AGE_BINS + 1; i++) run_params->AgeBinEdge[i] = run_params->Age[last - i];
    } else {
        const double ratio = (double)last / (double)N_AGE_BINS;
        run_params->AgeBinEdge[0] = run_params->Age[last];
        for(int i = 1; i < N_AGE_BINS; i++) run_params->AgeBinEdge[i] = run_params->Age[(int)(last - i * ratio)];
        run_params->AgeBinEdge[N_AGE_BINS] = run_params->Age[0];
    }
    double out[2];
    darksage_return_and_sn(0.0, 1.0e20, run_params, out);
    run_params->FinalRecycleFraction = out[0];
}

int darksage_age_bin(const double time, const struct params *run_params)
{
    int k = 0;
    while(k < N_AGE_BINS - 1 && time > run_params->AgeBinEdge[k+1]) k++;
    return k;
}

void darksage_instant_return_and_sn(const int k_now, const struct params *run_params, double out[2])
{
    darksage_return_and_sn(0.0, 0.5 * (run_params->AgeBinEdge[k_now+1] - run_params->AgeBinEdge[k_now]), run_params, out);
}

void darksage_move_ages(float mass_from[N_AGE_BINS], float metals_from[N_AGE_BINS], const double frac,
                        float mass_to[N_AGE_BINS], float metals_to[N_AGE_BINS])
{
    if(frac <= 0.0) return;
    for(int k = 0; k < N_AGE_BINS; k++) {
        const float dm = (float)(frac * mass_from[k]), dz = (float)(frac * metals_from[k]);
        mass_from[k] -= dm;
        metals_from[k] -= dz;
        if(mass_to != NULL) {
            mass_to[k] += dm;
            metals_to[k] += dz;
        }
    }
}

/* Mass lost by an age-binned component over this substep: the mass and metals returned, the new
 * metals made by the returning stars (DarkSage: Yield (1 - Z) / FinalRecycleFraction per unit mass
 * returned), and the SN energy (returned) */
static double age_losses(float mass[N_AGE_BINS], float metals[N_AGE_BINS], const int k_now,
                         const double *ret, const double *sn, const double inv_R_final, const double yield,
                         double *lost, double *lost_metals, double *new_metals, const struct params *run_params)
{
    double energy = 0.0;
    *lost = *lost_metals = *new_metals = 0.0;
    for(int k = k_now + 1; k < N_AGE_BINS; k++) {
        if(mass[k] <= 0.0f || ret[k] <= 0.0) continue;
        const double Z = get_metallicity(mass[k], metals[k]);
        energy += mass[k] * run_params->EnergySNcode * sn[k];
        const double dm = ret[k] * mass[k], dz = ret[k] * metals[k];
        mass[k] -= (float)dm;
        metals[k] -= (float)dz;
        *lost += dm;
        *lost_metals += dz;
        *new_metals += inv_R_final * dm * yield * (1.0 - Z);
    }
    return energy;
}

void darksage_delayed_feedback(const int p, const int centralgal, const double time, const double dt,
                               struct GALAXY *galaxies, const struct params *run_params)
{
    struct GALAXY *g = &galaxies[p];
    const int k_now = g->AgeBinNow;
    if(k_now >= N_AGE_BINS - 1) return;

    /* Step 1: fraction returned and supernovae per unit mass for each older bin over this substep */
    double ret[N_AGE_BINS] = {0.0}, sn[N_AGE_BINS] = {0.0};
    for(int k = k_now + 1; k < N_AGE_BINS; k++) {
        const double t0 = 0.5 * (run_params->AgeBinEdge[k] + run_params->AgeBinEdge[k+1]) - (time + 0.5 * dt);
        double out[2];
        darksage_return_and_sn(t0, t0 + dt, run_params, out);
        ret[k] = out[0];
        sn[k] = out[1];
    }
    const double inv_R_final = 1.0 / run_params->FinalRecycleFraction;

    /* Step 2: bulge populations -- their mass joins the fountain, their energy acts on the hot gas */
    double lost, lost_z, new_z, energy_onto_hot = 0.0;
    energy_onto_hot += age_losses(g->MergerBulgeMassAge, g->MergerBulgeMetalsAge, k_now, ret, sn, inv_R_final,
                                  run_params->Yield, &lost, &lost_z, &new_z, run_params);
    g->MergerBulgeMass -= lost;
    g->BulgeMass -= lost;
    g->StellarMass -= lost;
    g->MetalsBulgeMass -= lost_z;
    g->MetalsStellarMass -= lost_z;
    darksage_add_to_fountain(centralgal, lost, lost_z + new_z, galaxies, run_params);
    energy_onto_hot += age_losses(g->InstabilityBulgeMassAge, g->InstabilityBulgeMetalsAge, k_now, ret, sn, inv_R_final,
                                  run_params->Yield, &lost, &lost_z, &new_z, run_params);
    g->InstabilityBulgeMass -= lost;
    g->SecularBulgeMass -= lost;
    g->BulgeMass -= lost;
    g->StellarMass -= lost;
    g->MetalsBulgeMass -= lost_z;
    g->MetalsStellarMass -= lost_z;
    darksage_add_to_fountain(centralgal, lost, lost_z + new_z, galaxies, run_params);

    /* intrahalo stars: as the bulges.  ICS_disrupt / ICS_accrete / ICS_sum_mt stay records of the
     * mass deposited, so in DarkSAGEMode they no longer sum to ICS */
    energy_onto_hot += age_losses(g->ICSAge, g->MetalsICSAge, k_now, ret, sn, inv_R_final,
                                  run_params->Yield, &lost, &lost_z, &new_z, run_params);
    g->ICS -= lost;
    g->MetalsICS -= lost_z;
    if(g->ICS < 0.0f) g->ICS = 0.0f;
    if(g->MetalsICS < 0.0f) g->MetalsICS = 0.0f;
    darksage_add_to_fountain(centralgal, lost, lost_z + new_z, galaxies, run_params);

    /* Step 3: disc populations -- their mass returns to the annulus, their energy drives its outflow */
    for(int i = 0; i < N_BINS; i++) {
        const double energy = age_losses(g->DiscStarsAge[i], g->DiscStarsMetalsAge[i], k_now, ret, sn, inv_R_final,
                                         run_params->Yield, &lost, &lost_z, &new_z, run_params);
        if(lost <= 0.0) continue;
        g->DiscStars[i] -= (float)lost;
        g->DiscStarsMetals[i] -= (float)lost_z;
        g->StellarMass -= lost;
        g->MetalsStellarMass -= lost_z;
        g->DiscGas[i] += (float)lost;
        g->DiscGasMetals[i] += (float)(lost_z + new_z);
        /* DarkSage skips the outflow when the returned mass is negligible (MIN_STARFORMATION) */
        if(energy > 0.0 && lost > DS_MIN_RETURNED) {
            energy_onto_hot += darksage_energy_feedback(p, centralgal, i, energy, galaxies, run_params);
        }
    }

    /* Step 4: remaining energy ejects hot-phase gas */
    darksage_hot_energy_ejection(p, centralgal, energy_onto_hot, galaxies, run_params);

    double gas = 0.0, metals = 0.0;
    for(int i = 0; i < N_BINS; i++) {
        if(g->DiscStars[i] < 0.0f) g->DiscStars[i] = 0.0f;
        gas += g->DiscGas[i];
        metals += g->DiscGasMetals[i];
    }
    g->ColdGas = gas;
    g->MetalsColdGas = metals;
}

float darksage_age_mismatch(const struct GALAXY *g)
{
    /* Mass booked in the wrong place, as a fraction of the galaxy's stellar mass (so float rounding
     * on a nearly empty component does not register as a large error) */
    if(g->StellarMass + g->ICS <= 0.0f) return 0.0f;
    double err = 0.0, mb = 0.0, ib = 0.0;
    for(int i = 0; i < N_BINS; i++) {
        double sum = 0.0;
        for(int k = 0; k < N_AGE_BINS; k++) sum += g->DiscStarsAge[i][k];
        err += fabs(sum - g->DiscStars[i]);
    }
    for(int k = 0; k < N_AGE_BINS; k++) {
        mb += g->MergerBulgeMassAge[k];
        ib += g->InstabilityBulgeMassAge[k];
    }
    err += fabs(mb - g->MergerBulgeMass) + fabs(ib - g->InstabilityBulgeMass);
    double ics = 0.0;
    for(int k = 0; k < N_AGE_BINS; k++) ics += g->ICSAge[k];
    err += fabs(ics - g->ICS);
    return (float)(err / (g->StellarMass + g->ICS));
}

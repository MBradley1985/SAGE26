/*
 * model_darksage_sf.c -- annulus-resolved H2 and star formation (DarkSAGEMode >= 1).
 *
 * Applies the same SFprescription / H2DepletionMode laws as
 * model_starformation_and_feedback.c, but per disc annulus (DiscGas[i],
 * DiscStars[i], DiscRadii[i]) following Stevens, Croton & Mutch (2016).
 * DarkSAGEMode 2 forms its stars with these rates (the annuli are themselves a
 * radial integration over the actual disc, so H2RadialIntegrationOn and
 * H2DiskAreaOption are not used there); DarkSAGEMode 1 uses them only as the
 * shape over which SAGE26's bulk rate is shared.  The helpers below move gas,
 * stars and metals in and out of the annuli for every mode.
 *
 * The dynamical-time laws use the galaxy's t_dyn = 3 r_s / Vvir in every
 * annulus, as the bulk laws do (and DarkSage's SFprescription 2): an
 * annulus's own r_i / v_circ,i (~1 Myr at the centre) empties the inner disc.
 * The local circular velocity enters only the depletion-time laws.
 *
 * The stars, feedback and yields are booked in model_starformation_and_feedback.c.
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "core_allvars.h"
#include "macros.h"

#include "model_darksage_sf.h"
#include "model_h2_chemistry.h"
#include "model_starformation_and_feedback.h"
#include "model_misc.h"

/* Same constants as the bulk recipes in model_starformation_and_feedback.c */
static const double DS_SF_DISK_RADIUS_FRAC = 3.0;   /* star-forming radius in disc scale radii (SF_DISK_RADIUS_FRAC) */
static const double DS_SOMERVILLE25_SIGMA_CRIT = 30.0 / (M_PI * 4.302e-3);  /* Msun/pc^2 */
static const double DS_SOMERVILLE25_F_DENSE    = 0.5;
static const double DS_Z_SOLAR_ASPLUND09       = 0.014;

/* KMT09 two-phase fit, as inlined in sfr_kmt09() */
static double fH2_kmt09(const double Sigma_gas, const double Z)
{
    const double Z_prime = (Z > 0.0) ? Z / 0.02 : 0.0;
    const double clumping_factor = 3.0;
    const double tau_c = 0.066 * clumping_factor * Z_prime * Sigma_gas;
    const double chi = 0.77 * (1.0 + 3.1 * pow(Z_prime, 0.365));
    const double s = (clumping_factor * Sigma_gas > 0.0 && tau_c > 1e-10)
        ? log(1.0 + 0.6 * chi + 0.01 * chi * chi) / (0.6 * tau_c) : 100.0;
    return (s < 2.0) ? 1.0 - (3.0 * s) / (4.0 + s) : 0.0;
}

void darksage_annulus_rates(const int p, double sfr_bin[N_BINS], double h2_bin[N_BINS],
                            const struct GALAXY *galaxies, const struct params *run_params)
{
    const int sfp = run_params->SFprescription;
    const float h = run_params->Hubble_h;
    const double rs_pc = CODE_LENGTH_TO_PC(galaxies[p].DiskScaleRadius, h);
    const double gyr_to_code = SEC_PER_GIGAYEAR * run_params->Hubble_h / run_params->UnitTime_in_s;

    for(int i = 0; i < N_BINS; i++) sfr_bin[i] = h2_bin[i] = 0.0;
    if(galaxies[p].Vvir <= 0.0 || galaxies[p].DiskScaleRadius <= 0.0) return;
    /* Galaxy dynamical time, as the bulk laws */
    const double tdyn = DS_SF_DISK_RADIUS_FRAC * galaxies[p].DiskScaleRadius / galaxies[p].Vvir;

    for(int i = 0; i < N_BINS; i++) {
        const double gas = galaxies[p].DiscGas[i];
        if(gas <= 0.0) continue;

        const double r_in = galaxies[p].DiscRadii[i];
        const double r_out = galaxies[p].DiscRadii[i+1];
        const double r_mid = 0.5 * (r_in + r_out);   /* Mpc/h */
        const double area_pc2 = M_PI * (CODE_LENGTH_TO_PC(r_out, h) * CODE_LENGTH_TO_PC(r_out, h)
                                        - CODE_LENGTH_TO_PC(r_in, h) * CODE_LENGTH_TO_PC(r_in, h));
        if(area_pc2 <= 0.0 || r_mid <= 0.0) continue;

        const double Sigma_gas = CODE_MASS_TO_MSUN(gas, h) / area_pc2;
        const double Sigma_star = CODE_MASS_TO_MSUN(galaxies[p].DiscStars[i], h) / area_pc2;
        const double Z = galaxies[p].DiscGasMetals[i] / gas;

        /* Local circular velocity from the j-bin, v = j/r, for the depletion-time laws */
        const double j_mid = 0.5 * (run_params->DiscBinEdge[i] + run_params->DiscBinEdge[i+1]);
        const double v_circ = (j_mid > 0.0) ? j_mid / r_mid : galaxies[p].Vvir;

        /* --- molecular fraction --- */
        double f_H2 = 0.0;
        switch(sfp) {
            case 1: case 3: f_H2 = calculate_molecular_fraction_BR06(Sigma_gas, Sigma_star, rs_pc); break;
            case 4: f_H2 = calculate_H2_fraction_KD12(Sigma_gas, Z, 5.0); break;
            case 5: f_H2 = fH2_kmt09(Sigma_gas, Z); break;
            case 6: f_H2 = calculate_H2_fraction_K13(Sigma_gas, Z, 5.0); break;
            case 7: f_H2 = calculate_H2_fraction_GD14(Sigma_gas, Z, rs_pc); break;
            default: f_H2 = 0.0; break;   /* 0, 2: no H2 tracking */
        }
        if(f_H2 < 0.0) f_H2 = 0.0;
        if(f_H2 > 1.0) f_H2 = 1.0;
        const double h2 = f_H2 * gas * HYDROGEN_MASS_FRAC;
        h2_bin[i] = h2;

        /* --- star formation law --- */
        double sfr = 0.0;
        if(FG_TDEP(run_params) && sf_prescription_tracks_h2(sfp)) {
            /* SFR = M_mol / t_dep, as sfr_fixing_gas_tdep() */
            const double t_dep = h2_depletion_time_gyr(Sigma_gas, CODE_LENGTH_TO_PC(r_mid, h), v_circ,
                                                       run_params->H2DepletionMode);
            if(t_dep > 0.0) sfr = (h2 / HYDROGEN_MASS_FRAC) / (t_dep * gyr_to_code);
        } else if(sfp == 0) {
            /* Croton+06: the bulk threshold m_crit = 0.19 Vvir reff grows linearly with r,
             * so the annulus share is 0.19 Vvir (r_out - r_in) */
            const double cold_crit = 0.19 * galaxies[p].Vvir * (r_out - r_in);
            if(gas > cold_crit) sfr = run_params->SfrEfficiency * (gas - cold_crit) / tdyn;
        } else if(sfp == 2 || sfp == 3) {
            const double x = Sigma_gas / DS_SOMERVILLE25_SIGMA_CRIT;
            const double eps_cl = x / (1.0 + x);
            sfr = eps_cl * DS_SOMERVILLE25_F_DENSE * ((sfp == 2) ? gas : h2) / tdyn;
        } else if(sfp == 6) {
            /* K13: SFR = M_gas / t_dep */
            double Z_prime = Z / DS_Z_SOLAR_ASPLUND09;
            if(Z_prime < 0.01) Z_prime = 0.01;
            const double t_dep = calculate_tdep_K13_Gyr(Sigma_gas, Sigma_star, rs_pc, Z_prime, f_H2);
            if(t_dep > 0.0) sfr = gas / (t_dep * gyr_to_code);
        } else {
            sfr = run_params->SfrEfficiency * h2 / tdyn;
        }

        sfr_bin[i] = (sfr > 0.0) ? sfr : 0.0;
    }
}

double darksage_star_formation_rate(const int p, double sfr_bin[N_BINS],
                                    struct GALAXY *galaxies, const struct params *run_params)
{
    double h2_bin[N_BINS], total_sfr = 0.0, total_h2 = 0.0;
    darksage_annulus_rates(p, sfr_bin, h2_bin, galaxies, run_params);
    for(int i = 0; i < N_BINS; i++) {
        galaxies[p].DiscH2[i] = (float)h2_bin[i];
        total_sfr += sfr_bin[i];
        total_h2 += h2_bin[i];
    }
    galaxies[p].H2gas = 0.0;
    if(galaxies[p].Vvir <= 0.0 || galaxies[p].DiskScaleRadius <= 0.0) return 0.0;

    if(sf_prescription_tracks_h2(run_params->SFprescription)) {
        const double gyr_to_code = SEC_PER_GIGAYEAR * run_params->Hubble_h / run_params->UnitTime_in_s;
        galaxies[p].H2gas = total_h2;
        if(galaxies[p].H2gas > galaxies[p].ColdGas * HYDROGEN_MASS_FRAC) {
            galaxies[p].H2gas = galaxies[p].ColdGas * HYDROGEN_MASS_FRAC;
        }
        /* Effective whole-disc depletion time, for the H2DepletionTime_Gyr output */
        galaxies[p].H2DepletionTime_Gyr = (total_sfr > 0.0 && galaxies[p].H2gas > 0.0)
            ? (float)((galaxies[p].H2gas / HYDROGEN_MASS_FRAC) / total_sfr / gyr_to_code) : -1.0f;
    }

    return total_sfr;
}

/* Weights that replace the caller's in darksage_remove_disc_gas() while set (instability gas) */
static const double *gas_removal_override = NULL;

void darksage_set_gas_removal_weights(const double *weight)
{
    gas_removal_override = weight;
}

/*
 * remove_weighted -- take mass from arr[] in proportion to weight[], each element giving at
 * most what it holds; what an element cannot give comes from the others in proportion to what
 * they still hold, so exactly min(mass, sum(arr)) leaves.  taken[] receives what left each element.
 */
static void remove_weighted(float arr[N_BINS], const double mass, const double weight[N_BINS], double taken[N_BINS])
{
    double w_sum = 0.0, have = 0.0;
    for(int i = 0; i < N_BINS; i++) {
        taken[i] = 0.0;
        if(arr[i] <= 0.0f) continue;
        have += arr[i];
        if(weight != NULL && weight[i] > 0.0) w_sum += weight[i];
    }
    const double want = (mass < have) ? mass : have;
    if(want <= 0.0) return;

    double left = want;
    if(w_sum > 0.0) {
        for(int i = 0; i < N_BINS; i++) {
            if(arr[i] <= 0.0f || weight[i] <= 0.0) continue;
            double m = want * weight[i] / w_sum;
            if(m > arr[i]) m = arr[i];
            taken[i] = m;
            left -= m;
        }
    }
    if(left > 0.0) {
        double rest = 0.0;
        for(int i = 0; i < N_BINS; i++) if(arr[i] > 0.0f) rest += arr[i] - taken[i];
        if(rest > 0.0) {
            for(int i = 0; i < N_BINS; i++) if(arr[i] > 0.0f) taken[i] += left * (arr[i] - taken[i]) / rest;
        }
    }
    for(int i = 0; i < N_BINS; i++) {
        arr[i] -= (float)taken[i];
        if(arr[i] < 0.0f) arr[i] = 0.0f;
    }
}

void darksage_remove_disc_gas(const int p, const double mass, const double weight[N_BINS], double taken[N_BINS],
                              struct GALAXY *galaxies)
{
    double t[N_BINS];
    if(taken == NULL) taken = t;
    if(mass <= 0.0) {
        for(int i = 0; i < N_BINS; i++) taken[i] = 0.0;
        return;
    }
    if(gas_removal_override != NULL) weight = gas_removal_override;

    struct GALAXY *g = &galaxies[p];
    float before[N_BINS];
    for(int i = 0; i < N_BINS; i++) before[i] = g->DiscGas[i];
    remove_weighted(g->DiscGas, mass, weight, taken);
    for(int i = 0; i < N_BINS; i++) {
        if(before[i] <= 0.0f) continue;
        /* metals at the annulus's metallicity; H2 and HI keep the annulus's fractions */
        const double keep = 1.0 - taken[i] / before[i];
        g->DiscGasMetals[i] = (float)(g->DiscGasMetals[i] * keep);
        g->DiscH2[i] = (float)(g->DiscH2[i] * keep);
        g->DiscHI[i] = (float)(g->DiscHI[i] * keep);
    }
}

void darksage_remove_disc_stars(const int p, const double mass, const double weight[N_BINS], double taken[N_BINS],
                                struct GALAXY *galaxies)
{
    double t[N_BINS];
    if(taken == NULL) taken = t;
    struct GALAXY *g = &galaxies[p];
    float before[N_BINS];
    for(int i = 0; i < N_BINS; i++) before[i] = g->DiscStars[i];
    remove_weighted(g->DiscStars, mass, weight, taken);
    for(int i = 0; i < N_BINS; i++) {
        if(before[i] > 0.0f) g->DiscStarsMetals[i] = (float)(g->DiscStarsMetals[i] * (1.0 - taken[i] / before[i]));
    }
}

/* Rescale arr[] so it sums to target, keeping its shape; an empty arr[] takes the shape of shape[] */
static void rescale_to(float arr[N_BINS], const double target, const float shape[N_BINS])
{
    double sum = 0.0, shape_sum = 0.0;
    for(int i = 0; i < N_BINS; i++) {
        sum += arr[i];
        shape_sum += shape[i];
    }
    const double t = (target > 0.0) ? target : 0.0;
    if(sum > 0.0) {
        for(int i = 0; i < N_BINS; i++) arr[i] = (float)(arr[i] * t / sum);
    } else if(shape_sum > 0.0) {
        for(int i = 0; i < N_BINS; i++) arr[i] = (float)(t * shape[i] / shape_sum);
    }
}

/* The bulk moves metals at galaxy-average metallicities, the annuli at their local ones; rescale
 * the annulus metals to the bulk totals, keeping the radial metallicity profiles */
void darksage_sync_disc_metals(const int p, struct GALAXY *galaxies)
{
    struct GALAXY *g = &galaxies[p];
    rescale_to(g->DiscGasMetals, g->MetalsColdGas, g->DiscGas);
    rescale_to(g->DiscStarsMetals, g->MetalsStellarMass - g->MetalsBulgeMass, g->DiscStars);
}

/* Rescale the annulus H2 and HI to H2gas and H1gas, keeping their radial shapes (the bulk
 * phases are adjusted after star formation by bursts and clamps that the annuli do not see) */
void darksage_sync_disc_phases(const int p, struct GALAXY *galaxies)
{
    struct GALAXY *g = &galaxies[p];
    rescale_to(g->DiscH2, g->H2gas, g->DiscGas);
    rescale_to(g->DiscHI, g->H1gas, g->DiscGas);
}

void darksage_add_disc_metals(const int p, const double metals, const double weight[N_BINS],
                              struct GALAXY *galaxies)
{
    if(metals <= 0.0) return;
    struct GALAXY *g = &galaxies[p];

    /* where the weights say, among annuli that still hold gas; else in proportion to the gas */
    double w_sum = 0.0, gas_sum = 0.0;
    for(int i = 0; i < N_BINS; i++) {
        gas_sum += g->DiscGas[i];
        if(g->DiscGas[i] > 0.0f && weight != NULL && weight[i] > 0.0) w_sum += weight[i];
    }
    for(int i = 0; i < N_BINS; i++) {
        if(w_sum > 0.0) {
            if(g->DiscGas[i] > 0.0f && weight[i] > 0.0) g->DiscGasMetals[i] += (float)(metals * weight[i] / w_sum);
        } else if(gas_sum > 0.0) {
            g->DiscGasMetals[i] += (float)(metals * g->DiscGas[i] / gas_sum);
        }
    }
}

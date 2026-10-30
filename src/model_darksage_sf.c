/*
 * model_darksage_sf.c -- annulus-resolved H2 and star formation (DarkSAGEMode == 1).
 *
 * Applies the same SFprescription / H2DepletionMode laws as
 * model_starformation_and_feedback.c, but per disc annulus (DiscGas[i],
 * DiscStars[i], DiscRadii[i]) following Stevens, Croton & Mutch (2016).
 * The annuli are themselves a radial integration over the actual disc, so
 * H2RadialIntegrationOn and H2DiskAreaOption are not used in this mode.
 *
 * The bulk path stays authoritative for the totals (stars formed, SN
 * reheating, ejection, yields); these functions supply the bulk SFR and
 * mirror the bulk mass changes onto the annuli, weighted by the per-annulus SFR.
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

double darksage_star_formation_rate(const int p, double sfr_bin[N_BINS],
                                    struct GALAXY *galaxies, const struct params *run_params)
{
    const int sfp = run_params->SFprescription;
    const float h = run_params->Hubble_h;
    const double rs_pc = CODE_LENGTH_TO_PC(galaxies[p].DiskScaleRadius, h);
    const double gyr_to_code = SEC_PER_GIGAYEAR * run_params->Hubble_h / run_params->UnitTime_in_s;

    double total_sfr = 0.0, total_h2 = 0.0;

    for(int i = 0; i < N_BINS; i++) {
        sfr_bin[i] = 0.0;
        galaxies[p].DiscH2[i] = 0.0;
    }
    galaxies[p].H2gas = 0.0;

    if(galaxies[p].Vvir <= 0.0 || galaxies[p].DiskScaleRadius <= 0.0) {
        return 0.0;
    }

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

        /* Local circular velocity and dynamical time from the j-bins: v = j/r, t_dyn = r/v */
        const double j_mid = 0.5 * (run_params->DiscBinEdge[i] + run_params->DiscBinEdge[i+1]);
        const double v_circ = (j_mid > 0.0) ? j_mid / r_mid : galaxies[p].Vvir;
        const double tdyn = r_mid / v_circ;

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
        galaxies[p].DiscH2[i] = h2;
        total_h2 += h2;

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
        total_sfr += sfr_bin[i];
    }

    if(sf_prescription_tracks_h2(sfp)) {
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

void darksage_update_disc_from_sf(const int p, const double stars, const double reheated_mass,
                                  const double dt, const double sfr_bin[N_BINS],
                                  struct GALAXY *galaxies, const struct params *run_params)
{
    double sfr_sum = 0.0;
    for(int i = 0; i < N_BINS; i++) sfr_sum += sfr_bin[i];

    if(sfr_sum <= 0.0 || (stars <= 0.0 && reheated_mass <= 0.0)) return;

    const double R = run_params->RecycleFraction;
    double shortfall = 0.0;

    // New stars carry the gas-disc spin into the stellar disc (mass-weighted with the existing disc stars)
    double disc_stars = 0.0;
    for(int i = 0; i < N_BINS; i++) disc_stars += galaxies[p].DiscStars[i];
    evolve_spin(galaxies[p].SpinStars, disc_stars, galaxies[p].SpinGas, (1.0 - R) * stars);

    for(int i = 0; i < N_BINS; i++) {
        const double w = sfr_bin[i] / sfr_sum;
        if(dt > 0.0) galaxies[p].DiscSFR[i] += (float)(w * stars / dt);
        if(w <= 0.0 || galaxies[p].DiscGas[i] <= 0.0) continue;

        const double Z = galaxies[p].DiscGasMetals[i] / galaxies[p].DiscGas[i];
        double stars_i = (1.0 - R) * w * stars;
        double reheat_i = w * reheated_mass;
        const double removed = stars_i + reheat_i;
        if(removed > galaxies[p].DiscGas[i]) {
            /* Annulus runs short: stars still form here, the missing gas comes from the other annuli below */
            const double f = galaxies[p].DiscGas[i] / removed;
            shortfall += removed - galaxies[p].DiscGas[i];
            galaxies[p].DiscStars[i] += stars_i * (1.0 - f);
            galaxies[p].DiscStarsMetals[i] += Z * stars_i * (1.0 - f);
            stars_i *= f;
            reheat_i *= f;
        }

        galaxies[p].DiscGas[i] -= stars_i + reheat_i;
        galaxies[p].DiscGasMetals[i] -= Z * (stars_i + reheat_i);
        galaxies[p].DiscStars[i] += stars_i;
        galaxies[p].DiscStarsMetals[i] += Z * stars_i;

        if(galaxies[p].DiscGas[i] < 0.0f) galaxies[p].DiscGas[i] = 0.0f;
        if(galaxies[p].DiscGasMetals[i] < 0.0f) galaxies[p].DiscGasMetals[i] = 0.0f;
    }

    if(shortfall > 0.0) darksage_remove_disc_gas(p, shortfall, galaxies[p].DiscGas, galaxies);
}

void darksage_remove_disc_gas(const int p, const double mass, const float weight[N_BINS],
                              struct GALAXY *galaxies)
{
    if(mass <= 0.0) return;

    double w_sum = 0.0;
    for(int i = 0; i < N_BINS; i++) w_sum += weight[i];
    if(w_sum <= 0.0) return;

    double removed = 0.0;
    for(int i = 0; i < N_BINS; i++) {
        if(galaxies[p].DiscGas[i] <= 0.0f) continue;
        double m_i = mass * weight[i] / w_sum;
        if(m_i > galaxies[p].DiscGas[i]) m_i = galaxies[p].DiscGas[i];
        const double Z = galaxies[p].DiscGasMetals[i] / galaxies[p].DiscGas[i];
        galaxies[p].DiscGas[i] -= m_i;
        galaxies[p].DiscGasMetals[i] -= Z * m_i;
        if(galaxies[p].DiscGas[i] < 0.0f) galaxies[p].DiscGas[i] = 0.0f;
        if(galaxies[p].DiscGasMetals[i] < 0.0f) galaxies[p].DiscGasMetals[i] = 0.0f;
        removed += m_i;
    }

    /* Annuli that ran short: take the rest from all annuli in proportion to their remaining gas,
     * so the annuli lose exactly what the bulk ColdGas lost */
    const double shortfall = mass - removed;
    if(shortfall > 1.0e-6 * mass && weight != galaxies[p].DiscGas) {
        darksage_remove_disc_gas(p, shortfall, galaxies[p].DiscGas, galaxies);
    } else if(shortfall > 1.0e-6 * mass) {
        /* Already gas-weighted: only possible if the annuli hold less than mass in total; empty them */
        for(int i = 0; i < N_BINS; i++) galaxies[p].DiscGas[i] = galaxies[p].DiscGasMetals[i] = 0.0f;
    }
}

/* The bulk removes metals at the galaxy-average metallicity, the annuli at their local one;
 * rescale the annulus metals to the bulk total, keeping the radial metallicity profile */
void darksage_sync_disc_metals(const int p, struct GALAXY *galaxies)
{
    double z_sum = 0.0, gas_sum = 0.0;
    for(int i = 0; i < N_BINS; i++) {
        z_sum += galaxies[p].DiscGasMetals[i];
        gas_sum += galaxies[p].DiscGas[i];
    }
    const double target = (galaxies[p].MetalsColdGas > 0.0f) ? galaxies[p].MetalsColdGas : 0.0;

    if(z_sum > 0.0) {
        const double f = target / z_sum;
        for(int i = 0; i < N_BINS; i++) galaxies[p].DiscGasMetals[i] = (float)(galaxies[p].DiscGasMetals[i] * f);
    } else if(gas_sum > 0.0) {
        for(int i = 0; i < N_BINS; i++) galaxies[p].DiscGasMetals[i] = (float)(target * galaxies[p].DiscGas[i] / gas_sum);
    }
}

void darksage_add_disc_metals(const int p, const double metals, const double sfr_bin[N_BINS],
                              struct GALAXY *galaxies)
{
    if(metals <= 0.0) return;

    double sfr_sum = 0.0;
    for(int i = 0; i < N_BINS; i++) sfr_sum += sfr_bin[i];
    if(sfr_sum <= 0.0) return;

    for(int i = 0; i < N_BINS; i++) {
        galaxies[p].DiscGasMetals[i] += metals * sfr_bin[i] / sfr_sum;
    }
}

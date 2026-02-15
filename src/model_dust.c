/* Dust modeling for SAGE26
 * Based on dusty-sage (Triani et al. 2020, arXiv:2002.05343)
 * Physics from: Popping et al. 2017, Asano et al. 2013, Arrigoni et al. 2010
 *
 * Tracks three dust reservoirs: ColdDust, HotDust, EjectedDust
 * Implements: production, accretion, destruction, thermal sputtering
 *
 * Two production modes controlled by MetalYieldsOn:
 *   0 = Simplified model (weighted-average delta_eff * Yield * stars)
 *   1 = Full yield-table model (per-element, IMF-weighted, delayed enrichment)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef GSL_FOUND
#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>
#endif

#include "core_allvars.h"
#include "model_dust.h"
#include "model_misc.h"


/* ========================================================================
 * produce_dust: Simplified dust production (MetalYieldsOn=0)
 * ======================================================================== */
void produce_dust(const double stars, const double metallicity, const double dt,
                  const int p, const int centralgal, const int step,
                  struct GALAXY *galaxies, const struct params *run_params)
{
    (void)centralgal; (void)dt;

    if(stars <= 0.0 || galaxies[p].ColdGas <= 1.0e-10) return;

    const double metals_produced = run_params->Yield * stars;

    const double f_agb = 0.3, f_snii = 0.6, f_snia = 0.1;
    const double delta_eff = f_agb * run_params->DeltaDustAGB +
                             f_snii * run_params->DeltaDustSNII +
                             f_snia * run_params->DeltaDustSNIa;

    double dust_produced = delta_eff * metals_produced;

    if(metallicity < 0.01) dust_produced *= metallicity / 0.01;
    if(dust_produced > metals_produced) dust_produced = metals_produced;

    /* Store formation rate in code units (will be converted in output) */
    if(dt > 0.0 && step >= 0 && step < STEPS) {
        galaxies[p].DustDotForm[step] += (float)(dust_produced / dt);
    }

    galaxies[p].ColdDust += dust_produced;
    if(galaxies[p].ColdDust > galaxies[p].MetalsColdGas)
        galaxies[p].ColdDust = galaxies[p].MetalsColdGas;
    if(galaxies[p].ColdDust < 0.0) galaxies[p].ColdDust = 0.0;
}


#ifdef GSL_FOUND
/* ========================================================================
 * produce_metals_dust: Full yield-table dust production (MetalYieldsOn=1)
 *
 * Per-element yields from AGB (Karakas 2010), SNII (WW95/Nomoto06),
 * SNIa (Iwamoto99/Seitenzahl13), convolved with the IMF and the
 * galaxy's SFR history for delayed enrichment.
 *
 * Dust condensation from Popping et al. 2017 eqs. 4-6.
 * SNIa rate from Arrigoni et al. 2010 eq. 9-10.
 * ======================================================================== */
void produce_metals_dust(const double metallicity, const double dt,
                         const int p, const int centralgal, const int step,
                         struct GALAXY *galaxies, const struct params *run_params)
{
    (void)centralgal;   /* reserved for future use */
    if(galaxies[p].ColdGas <= 1.0e-10 || dt <= 0.0) return;

    const double A = run_params->BinaryFraction;
    const int snapnum = galaxies[p].SnapNum;

    /* Local copies of SFR history and snapshot ages */
    double age[ABSOLUTEMAXSNAPS], sfh[ABSOLUTEMAXSNAPS];
    for(int i = 0; i < run_params->Snaplistlen; i++) {
        age[i] = run_params->lbtime[i];
        sfh[i] = (double)galaxies[p].Sfr[i];
    }

    /* ---------- SNIa channel (Arrigoni+2010 eq 9-10) ---------- */
    /* Uses pre-computed mbin_snia, mu_snia, fmu_snia, taum_snia from init */
    const int count = 20;
    const double up_binary = 16.0;
    const double max_mu = 0.5;

    double yCrmu[20], yFemu[20], yNimu[20];
    double yCrphi[20], yFephi[20], yNiphi[20];

    for(int i = 0; i < count; i++) {
        double time = age[snapnum] - run_params->taum_snia[i];
        double sfr = 0.0;
        if(time >= run_params->lbtime[0]) {
            sfr = interpolate_arr(age, sfh, run_params->Snaplistlen, time);
        }
        yCrmu[i] = run_params->fmu_snia[i] * sfr * run_params->qCrsnia;
        yFemu[i] = run_params->fmu_snia[i] * sfr * run_params->qFesnia;
        yNimu[i] = run_params->fmu_snia[i] * sfr * run_params->qNisnia;
    }

    double yCr = integrate_arr(run_params->mu_snia, yCrmu, count, max_mu / count, max_mu);
    double yFe = integrate_arr(run_params->mu_snia, yFemu, count, max_mu / count, max_mu);
    double yNi = integrate_arr(run_params->mu_snia, yNimu, count, max_mu / count, max_mu);
    for(int i = 0; i < count; i++) {
        yCrphi[i] = yCr * run_params->phi_snia[i];
        yFephi[i] = yFe * run_params->phi_snia[i];
        yNiphi[i] = yNi * run_params->phi_snia[i];
    }

    /* ---------- AGB channel (Karakas 2010) ---------- */
    /* Uses pre-computed phi_agb, taum_agb from init; stack arrays for speed */
    const double low_agb = 1.0, up_agb = 6.0, low_binary = 3.0;

    /* Find nearest metallicity grid index */
    int j_agb = 0;
    if(run_params->AGBYields == 0) {
        double Z_std[7] = {0.0, 1e-4, 4e-4, 4e-3, 8e-3, 0.02, 0.05};
        double sfrz = metallicity;
        if(sfrz < Z_std[0]) sfrz = Z_std[0];
        else if(sfrz > Z_std[6]) sfrz = Z_std[6];
        for(int i = 0; i < 7; i++) {
            if(Z_std[i] <= sfrz) j_agb = i;
        }
    }

    /* Stack arrays instead of calloc (countagb is typically ~16) */
    double yCagb[MAXYIELDS], yNagb[MAXYIELDS], yOagb[MAXYIELDS];

    for(int i = 0; i < run_params->countagb; i++) {
        if(run_params->magb[i] != 0.0) {
            double time = age[snapnum] - run_params->taum_agb[i];
            double sfr = 0.0;
            if(time >= run_params->lbtime[0]) {
                sfr = interpolate_arr(age, sfh, run_params->Snaplistlen, time);
            }
            yCagb[i] = run_params->qCagb[i][j_agb] * run_params->phi_agb[i] * sfr;
            yNagb[i] = run_params->qNagb[i][j_agb] * run_params->phi_agb[i] * sfr;
            yOagb[i] = run_params->qOagb[i][j_agb] * run_params->phi_agb[i] * sfr;
        } else {
            yCagb[i] = yNagb[i] = yOagb[i] = 0.0;
        }
    }

    /* ---------- SNII channel ---------- */
    /* Uses pre-computed phi_sn, taum_sn from init; stack arrays for speed */
    const double low_sn = run_params->msn[0];
    const double up_sn = 40.0;

    int j_sn = 0;
    if(run_params->SNIIYields == 0) {
        double Z_std[7] = {0.0, 1e-4, 4e-4, 4e-3, 8e-3, 0.02, 0.05};
        double sfrz = metallicity;
        if(sfrz < Z_std[0]) sfrz = Z_std[0];
        else if(sfrz > Z_std[6]) sfrz = Z_std[6];
        for(int i = 0; i < 7; i++) {
            if(Z_std[i] <= sfrz) j_sn = i;
        }
    } else if(run_params->SNIIYields == 1) {
        double Z_std[4] = {0.0, 0.001, 0.004, 0.02};
        double sfrz = metallicity;
        if(sfrz < Z_std[0]) sfrz = Z_std[0];
        else if(sfrz > Z_std[3]) sfrz = Z_std[3];
        for(int i = 0; i < 4; i++) {
            if(Z_std[i] <= sfrz) j_sn = i;
        }
    }

    /* Stack arrays instead of calloc (countsn is typically ~13) */
    double yCsn[MAXYIELDS], yOsn[MAXYIELDS], yMgsn[MAXYIELDS], ySisn[MAXYIELDS];
    double ySsn[MAXYIELDS], yCasn[MAXYIELDS], yFesn[MAXYIELDS];

    for(int i = 0; i < run_params->countsn; i++) {
        if(run_params->msn[i] != 0.0) {
            double time = age[snapnum] - run_params->taum_sn[i];
            double sfr = 0.0;
            if(time >= run_params->lbtime[0]) {
                sfr = interpolate_arr(age, sfh, run_params->Snaplistlen, time);
            }
            yCsn[i]  = run_params->qCsn[i][j_sn]  * run_params->phi_sn[i] * sfr;
            yOsn[i]  = run_params->qOsn[i][j_sn]  * run_params->phi_sn[i] * sfr;
            yMgsn[i] = run_params->qMgsn[i][j_sn] * run_params->phi_sn[i] * sfr;
            ySisn[i] = run_params->qSisn[i][j_sn] * run_params->phi_sn[i] * sfr;
            ySsn[i]  = run_params->qSsn[i][j_sn]  * run_params->phi_sn[i] * sfr;
            yCasn[i] = run_params->qCasn[i][j_sn] * run_params->phi_sn[i] * sfr;
            yFesn[i] = run_params->qFesn[i][j_sn] * run_params->phi_sn[i] * sfr;
        } else {
            yCsn[i] = yOsn[i] = yMgsn[i] = ySisn[i] = ySsn[i] = yCasn[i] = yFesn[i] = 0.0;
        }
    }

    /* ---------- Integrate over IMF ---------- */
    /* Use magb/msn/mbin_snia directly since we no longer copy to m_agb/m_sn */
    double yCr_snia = A * integrate_arr(run_params->mbin_snia, yCrphi, count, run_params->mbin_snia[0], up_binary);
    double yNi_snia = A * integrate_arr(run_params->mbin_snia, yNiphi, count, run_params->mbin_snia[0], up_binary);
    double yFe_snia = A * integrate_arr(run_params->mbin_snia, yFephi, count, run_params->mbin_snia[0], up_binary);

    double yC_agb = (1.0 - A) * integrate_arr(run_params->magb, yCagb, run_params->countagb, low_binary, up_agb)
                               + integrate_arr(run_params->magb, yCagb, run_params->countagb, low_agb, low_binary);
    double yN_agb = (1.0 - A) * integrate_arr(run_params->magb, yNagb, run_params->countagb, low_binary, up_agb)
                               + integrate_arr(run_params->magb, yNagb, run_params->countagb, low_agb, low_binary);
    double yO_agb = (1.0 - A) * integrate_arr(run_params->magb, yOagb, run_params->countagb, low_binary, up_agb)
                               + integrate_arr(run_params->magb, yOagb, run_params->countagb, low_agb, low_binary);

    double yC_sn  = (1.0 - A) * integrate_arr(run_params->msn, yCsn,  run_params->countsn, low_sn, up_binary)
                               + integrate_arr(run_params->msn, yCsn,  run_params->countsn, up_binary, up_sn);
    double yO_sn  = (1.0 - A) * integrate_arr(run_params->msn, yOsn,  run_params->countsn, low_sn, up_binary)
                               + integrate_arr(run_params->msn, yOsn,  run_params->countsn, up_binary, up_sn);
    double yMg_sn = (1.0 - A) * integrate_arr(run_params->msn, yMgsn, run_params->countsn, low_sn, up_binary)
                               + integrate_arr(run_params->msn, yMgsn, run_params->countsn, up_binary, up_sn);
    double ySi_sn = (1.0 - A) * integrate_arr(run_params->msn, ySisn, run_params->countsn, low_sn, up_binary)
                               + integrate_arr(run_params->msn, ySisn, run_params->countsn, up_binary, up_sn);
    double yS_sn  = (1.0 - A) * integrate_arr(run_params->msn, ySsn,  run_params->countsn, low_sn, up_binary)
                               + integrate_arr(run_params->msn, ySsn,  run_params->countsn, up_binary, up_sn);
    double yCa_sn = (1.0 - A) * integrate_arr(run_params->msn, yCasn, run_params->countsn, low_sn, up_binary)
                               + integrate_arr(run_params->msn, yCasn, run_params->countsn, up_binary, up_sn);
    double yFe_sn = (1.0 - A) * integrate_arr(run_params->msn, yFesn, run_params->countsn, low_sn, up_binary)
                               + integrate_arr(run_params->msn, yFesn, run_params->countsn, up_binary, up_sn);

    /* No more free() calls needed - using stack arrays now */

    /* ---------- Element masses produced this timestep ---------- */
    double Cr_snia = yCr_snia * dt;
    double Fe_snia = yFe_snia * dt;
    double Ni_snia = yNi_snia * dt;
    (void)Cr_snia; (void)Fe_snia; (void)Ni_snia; /* SNIa dust currently disabled */

    double C_agb = yC_agb * dt;
    double N_agb = yN_agb * dt;
    double O_agb = yO_agb * dt;

    double C_sn  = yC_sn  * dt;
    double O_sn  = yO_sn  * dt;
    double Mg_sn = yMg_sn * dt;
    double Si_sn = ySi_sn * dt;
    double S_sn  = yS_sn  * dt;
    double Ca_sn = yCa_sn * dt;
    double Fe_sn = yFe_sn * dt;

    /* ---------- Dust condensation (Popping+2017 eqs 4-6) ---------- */
    double dustdot = 0.0;
    const double delta_agb = run_params->DeltaDustAGB;
    const double delta_sn  = run_params->DeltaDustSNII;

    /* AGB dust: eq 4-5 */
    if(O_agb > 0.0 && C_agb / O_agb > 1.0) {
        dustdot += delta_agb * (C_agb - 0.75 * O_agb) / dt;
    } else {
        dustdot += delta_agb * (C_agb + N_agb + O_agb) / dt;
    }

    /* SNII dust: eq 6 — carbon + oxygen + silicates (stoichiometric) */
    dustdot += delta_sn * C_sn / dt;
    dustdot += delta_sn * O_sn / dt;
    dustdot += 16.0 * delta_sn * (Mg_sn/24.0 + Si_sn/28.0 + S_sn/32.0
                                 + Ca_sn/40.0 + Fe_sn/56.0) / dt;

    /* SNIa dust: commented out in dusty-sage, kept disabled here
     * dustdot += 16.0 * delta_snia * (Fe_snia/56.0) / dt;
     * dustdot += delta_snia * (Cr_snia + Ni_snia) / dt;
     */

    /* Store formation rate in code units (will be converted in output) */
    if(dustdot > 0.0 && step >= 0 && step < STEPS) {
        galaxies[p].DustDotForm[step] += (float)(dustdot);
    }

    /* Apply dust to cold phase */
    if(dustdot > 0.0) {
        galaxies[p].ColdDust += dustdot * dt;
    }

    /* Safety: can't exceed metals in cold gas */
    if(galaxies[p].ColdDust > galaxies[p].MetalsColdGas) {
        galaxies[p].ColdDust = galaxies[p].MetalsColdGas;
    }
    if(galaxies[p].ColdDust < 0.0) galaxies[p].ColdDust = 0.0;
}
#endif /* GSL_FOUND */


void accrete_dust(const double metallicity, const double dt, const int p,
                  const int step, struct GALAXY *galaxies, const struct params *run_params)
{
    /* ISM grain growth / dust accretion
     * Based on Asano et al. 2013 equation 20
     *
     * Dust grains grow by accreting gas-phase metals in dense ISM
     * Growth timescale depends on metallicity and H2 fraction
     */

    if(galaxies[p].ColdGas <= 1.0e-10 || galaxies[p].MetalsColdGas <= 1.0e-10) {
        return;
    }

    /* Current dust-to-metal ratio */
    const double dust_to_metal = galaxies[p].ColdDust / galaxies[p].MetalsColdGas;

    /* If already saturated (all metals are in dust), no more accretion */
    if(dust_to_metal >= 1.0) {
        return;
    }

    /* Reference accretion timescale at solar metallicity
     * Default: 20 Myr (from parameter file) */
    const double tacc_zero = run_params->DustAccretionTimescale * SEC_PER_MEGAYEAR /
                             run_params->UnitTime_in_s;

    /* Accretion timescale scales inversely with metallicity
     * Higher metallicity = more gas-phase metals = faster accretion */
    double tacc = tacc_zero;
    if(metallicity > 0.0) {
        tacc = tacc_zero * 0.02 / metallicity;  /* 0.02 = solar metallicity */
    }

    /* Only accretion happens in molecular gas (where densities are high enough)
     * Use H2 fraction from galaxy if available, otherwise estimate */
    double f_molecular = 0.0;
    if(galaxies[p].ColdGas > 0.0) {
        f_molecular = galaxies[p].H2gas / galaxies[p].ColdGas;
    }
    if(f_molecular > 1.0) f_molecular = 1.0;
    if(f_molecular < 0.0) f_molecular = 0.0;

    /* For galaxies without H2 tracking, assume some fraction based on gas mass */
    if(f_molecular < 1.0e-10 && galaxies[p].ColdGas > 0.1) {
        f_molecular = 0.5;  /* Default assumption */
    }

    /* Grain growth rate (Asano+13 eq. 20, with quadratic self-regulation)
     * d(dust)/dt = (1 - dust/metals)^2 * f_H2 * dust / tacc
     * The squared term prevents DtM saturation in metal-rich galaxies
     * while leaving low-DtM systems nearly unchanged */
    if(tacc > 0.0 && f_molecular > 0.0) {
        const double one_minus_f = 1.0 - dust_to_metal;
        const double dustdot = one_minus_f * one_minus_f * f_molecular *
                               galaxies[p].ColdDust / tacc;

        double delta_dust = dustdot * dt;

        /* Can't accrete more than available gas-phase metals */
        const double gas_phase_metals = galaxies[p].MetalsColdGas - galaxies[p].ColdDust;
        if(delta_dust > gas_phase_metals) {
            delta_dust = gas_phase_metals;
        }

        /* Store growth rate in code units (will be converted in output) */
        if(delta_dust > 0.0 && step >= 0 && step < STEPS) {
            galaxies[p].DustDotGrowth[step] += (float)(delta_dust / dt);
        }

        if(delta_dust > 0.0) {
            galaxies[p].ColdDust += delta_dust;
        }
    }

    /* Safety checks */
    if(galaxies[p].ColdDust < 0.0) {
        galaxies[p].ColdDust = 0.0;
    }
    if(galaxies[p].ColdDust > galaxies[p].MetalsColdGas) {
        galaxies[p].ColdDust = galaxies[p].MetalsColdGas;
    }
}


void destruct_dust(const double metallicity, const double stars, const double dt,
                   const int p, const int step, struct GALAXY *galaxies, const struct params *run_params)
{
    /* Dust destruction by SN shocks
     * Based on Asano et al. 2013 equations 12, 14
     * 
     * IMPORTANT: SN rate now uses DELAYED enrichment to be consistent with
     * dust production. SNe explode from stars that formed taum (stellar lifetime) ago,
     * not from stars forming now. This prevents the timing mismatch where
     * destruction responds instantly to SFR but production is delayed.
     */
    (void)stars;  /* Now using delayed SN rate from SFR history instead */

    if(galaxies[p].ColdGas <= 1.0e-10 || galaxies[p].ColdDust <= 0.0) {
        return;
    }

    const double eta_sn = run_params->EtaSNDust;  /* destruction efficiency (tunable) */
    const double m_swept = 1535.0 * pow(metallicity / 0.02 + 0.039, -0.289)
                           * run_params->Hubble_h / 1.0e10;  /* code units */

    if(m_swept <= 0.0) return;

    /* Compute DELAYED SN rate from past SFR history */
    double Rsn = 0.0;

#ifdef GSL_FOUND
    {
        const int snapnum = galaxies[p].SnapNum;
        
        /* Local copies of SFR history and snapshot ages */
        double age[ABSOLUTEMAXSNAPS], sfh[ABSOLUTEMAXSNAPS];
        for(int i = 0; i < run_params->Snaplistlen; i++) {
            age[i] = run_params->lbtime[i];
            sfh[i] = (double)galaxies[p].Sfr[i];
        }
        
        /* Integrate SN rate over massive stars (8-40 Msun) using cached mass grid */
        const int nbins = 20;
        const double m_low = 8.0, m_up = 40.0;
        double sn_rate[20], mphi[20];
        
        for(int i = 0; i < nbins; i++) {
            /* Use pre-computed mass, phi, and taum from init */
            double time = age[snapnum] - run_params->taum_destruct[i];
            
            /* Look up SFR at that past time */
            double sfr_past = 0.0;
            if(time >= run_params->lbtime[0] && time <= run_params->lbtime[run_params->Snaplistlen - 1]) {
                sfr_past = interpolate_arr(age, sfh, run_params->Snaplistlen, time);
            }
            
            /* SN rate contribution: each massive star that formed at time and is dying now */
            sn_rate[i] = run_params->phi_destruct[i] * sfr_past;
            mphi[i] = run_params->mass_destruct[i] * run_params->phi_destruct[i];
        }
        
        /* Integrate over IMF to get total SN rate */
        double total_sn = integrate_arr(run_params->mass_destruct, sn_rate, nbins, m_low, m_up);
        double total_phi = integrate_arr(run_params->mass_destruct, run_params->phi_destruct, nbins, m_low, m_up);
        
        if(total_phi > 0.0) {
            /* mean_mass in physical Msun */
            double mean_mass = integrate_arr(run_params->mass_destruct, mphi, nbins, m_low, m_up) / total_phi;
            
            /* Convert mean_mass from Msun to code units */
            double mean_mass_code = mean_mass * run_params->Hubble_h / 1.0e10;
            
            if(mean_mass_code > 0.0 && total_sn > 0.0) {
                /* Rsn = integrated SN rate / mean mass of SN progenitors */
                Rsn = total_sn / mean_mass_code;
            }
        }
    }
#else
    /* Fallback without GSL: use instantaneous SFR (less accurate) */
    if(stars > 0.0) {
        Rsn = stars / dt / (100.0 * run_params->Hubble_h / 1.0e10);
    }
#endif

    if(Rsn > 0.0 && galaxies[p].ColdGas > 0.0) {
        /* tsn = ColdGas / (eta * m_swept * Rsn) is in code time */
        double tsn = galaxies[p].ColdGas / (eta_sn * m_swept * Rsn);

        if(tsn > 0.0) {
            /* dustdot = ColdDust / tsn is in code mass / code time */
            double dustdot = galaxies[p].ColdDust / tsn;

            /* Store destruction rate in code mass / code time (same as formation/growth) */
            if(dustdot > 0.0 && step >= 0 && step < STEPS) {
                galaxies[p].DustDotDestruct[step] += (float)(dustdot);
            }

            /* Subtract destroyed dust */
            if(galaxies[p].ColdDust - dustdot * dt > 0.0) {
                galaxies[p].ColdDust -= dustdot * dt;
            } else {
                galaxies[p].ColdDust = 0.0;
            }
        }
    }

    if(galaxies[p].ColdDust < 0.0) galaxies[p].ColdDust = 0.0;
}


void dust_thermal_sputtering(const int gal, const double dt,
                             struct GALAXY *galaxies, const struct params *run_params)
{
    /* Thermal sputtering in hot gas (Popping+17 section 3.5)
     * With grain size evolution following dusty-sage:
     *   adot = rho / x, where x = -3.2e-18 / mp / (T0/T)^gamma + 1)
     *   a -= adot * dt, then a is used in the sputtering timescale.
     */

    const double a0_cm = 1.0e-5;  /* Initial grain radius: 0.1 micrometer in cm */
    const double a0 = a0_cm / run_params->UnitLength_in_cm;  /* in code units */

    /* ---- HotDust ---- */
    if(galaxies[gal].HotDust > 0.0 && galaxies[gal].Vvir > 0.0 && galaxies[gal].Rvir > 0.0) {
        const double temp0 = 2.0e6;
        const double temp = 35.9 * galaxies[gal].Vvir * galaxies[gal].Vvir;
        const double gamma_sput = 2.5;

        const double volume = (4.0 / 3.0) * M_PI * galaxies[gal].Rvir
                            * galaxies[gal].Rvir * galaxies[gal].Rvir;
        double rho = (volume > 0.0) ? galaxies[gal].HotGas / volume : 0.0;

        if(rho > 0.0) {
            /* Compute grain erosion rate in code units */
            double x = -3.2e-18 / PROTONMASS / (pow(temp0 / temp, gamma_sput) + 1.0);
            x /= run_params->UnitLength_in_cm / run_params->UnitDensity_in_cgs
               / run_params->UnitTime_in_s;
            double adot = rho / x;
            double a = a0 - adot * dt;

            if(a > 0.0) {
                /* Normalize to 0.1 micrometer units for the timescale formula */
                double a_norm = a / a0;
                double rho_cgs = rho * run_params->UnitDensity_in_cgs;
                const double tau0 = 170.0 * SEC_PER_MEGAYEAR / run_params->UnitTime_in_s;
                const double tau = tau0 * (a_norm / rho_cgs) * (pow(temp0 / temp, gamma_sput) + 1.0);

                if(tau > 0.0) {
                    double mdot = galaxies[gal].HotDust / tau / 3.0;
                    double delta_dust = mdot * dt;
                    if(delta_dust > galaxies[gal].HotDust) delta_dust = galaxies[gal].HotDust;
                    galaxies[gal].HotDust -= delta_dust;
                    galaxies[gal].MetalsHotGas += delta_dust;
                }
            } else {
                /* Grain fully eroded — destroy all remaining dust */
                galaxies[gal].MetalsHotGas += galaxies[gal].HotDust;
                galaxies[gal].HotDust = 0.0;
            }
        }
    }

    /* ---- CGMDust ---- */
    if(galaxies[gal].CGMDust > 0.0 && galaxies[gal].Vvir > 0.0 && galaxies[gal].Rvir > 0.0) {
        const double temp0_cgm = 2.0e6;
        const double temp_cgm = 35.9 * galaxies[gal].Vvir * galaxies[gal].Vvir;
        const double gamma_sput_cgm = 2.5;

        const double volume_cgm = (4.0 / 3.0) * M_PI * galaxies[gal].Rvir
                            * galaxies[gal].Rvir * galaxies[gal].Rvir;
        double rho_cgm = (volume_cgm > 0.0) ? galaxies[gal].CGMgas / volume_cgm : 0.0;

        if(rho_cgm > 0.0) {
            double x_cgm = -3.2e-18 / PROTONMASS / (pow(temp0_cgm / temp_cgm, gamma_sput_cgm) + 1.0);
            x_cgm /= run_params->UnitLength_in_cm / run_params->UnitDensity_in_cgs
               / run_params->UnitTime_in_s;
            double adot_cgm = rho_cgm / x_cgm;
            double a_cgm = a0 - adot_cgm * dt;

            if(a_cgm > 0.0) {
                double a_norm_cgm = a_cgm / a0;
                double rho_cgs_cgm = rho_cgm * run_params->UnitDensity_in_cgs;
                const double tau0_cgm = 170.0 * SEC_PER_MEGAYEAR / run_params->UnitTime_in_s;
                const double tau_cgm = tau0_cgm * (a_norm_cgm / rho_cgs_cgm) * (pow(temp0_cgm / temp_cgm, gamma_sput_cgm) + 1.0);

                if(tau_cgm > 0.0) {
                    double mdot_cgm = galaxies[gal].CGMDust / tau_cgm / 3.0;
                    double delta_dust_cgm = mdot_cgm * dt;
                    if(delta_dust_cgm > galaxies[gal].CGMDust) delta_dust_cgm = galaxies[gal].CGMDust;
                    galaxies[gal].CGMDust -= delta_dust_cgm;
                    galaxies[gal].MetalsCGMgas += delta_dust_cgm;
                }
            } else {
                galaxies[gal].MetalsCGMgas += galaxies[gal].CGMDust;
                galaxies[gal].CGMDust = 0.0;
            }
        }
    }

    /* ---- EjectedDust ---- */
    if(galaxies[gal].EjectedDust > 0.0 && galaxies[gal].Vvir > 0.0 && galaxies[gal].Rvir > 0.0) {
        const double temp0 = 2.0e6;
        const double temp = 35.9 * galaxies[gal].Vvir * galaxies[gal].Vvir;
        const double gamma_sput = 2.5;

        const double volume = (4.0 / 3.0) * M_PI * galaxies[gal].Rvir
                            * galaxies[gal].Rvir * galaxies[gal].Rvir;
        double rho = (volume > 0.0) ? galaxies[gal].EjectedMass / volume : 0.0;

        if(rho > 0.0) {
            double x = -3.2e-18 / PROTONMASS / (pow(temp0 / temp, gamma_sput) + 1.0);
            x /= run_params->UnitLength_in_cm / run_params->UnitDensity_in_cgs
               / run_params->UnitTime_in_s;
            double adot = rho / x;
            double a = a0 - adot * dt;

            if(a > 0.0) {
                double a_norm = a / a0;
                double rho_cgs = rho * run_params->UnitDensity_in_cgs;
                const double tau0 = 170.0 * SEC_PER_MEGAYEAR / run_params->UnitTime_in_s;
                const double tau = tau0 * (a_norm / rho_cgs) * (pow(temp0 / temp, gamma_sput) + 1.0);

                if(tau > 0.0) {
                    double mdot = galaxies[gal].EjectedDust / tau / 3.0;
                    double delta_dust = mdot * dt;
                    if(delta_dust > galaxies[gal].EjectedDust) delta_dust = galaxies[gal].EjectedDust;
                    galaxies[gal].EjectedDust -= delta_dust;
                    galaxies[gal].MetalsEjectedMass += delta_dust;
                }
            } else {
                galaxies[gal].MetalsEjectedMass += galaxies[gal].EjectedDust;
                galaxies[gal].EjectedDust = 0.0;
            }
        }
    }

    if(galaxies[gal].HotDust < 0.0) galaxies[gal].HotDust = 0.0;
    if(galaxies[gal].CGMDust < 0.0) galaxies[gal].CGMDust = 0.0;
    if(galaxies[gal].EjectedDust < 0.0) galaxies[gal].EjectedDust = 0.0;
}

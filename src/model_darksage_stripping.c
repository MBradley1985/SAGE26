/*
 * model_darksage_stripping.c -- ram-pressure and tidal stripping of satellite discs (DarkSAGEMode == 1).
 *
 * Follows DarkSage's ram_pressure_stripping() (model_infall.c) and
 * get_Mhost_internal() (model_misc.c); Stevens et al. (2024) Sec. 11.
 * Called every substep for Type 1 satellites with cold gas.
 *
 * Tidal radius (Tormen, Diaferio & Syer 1998; paper eq. 116):
 *   r_tidal = R_sat (M_sat / (2 M_host(<R_sat) - R_sat dM_host/dR))^(1/3).
 * With M_sat < ThreshMajorMerger M_host(<R_sat), a satellite whose first
 * annulus lies outside r_tidal is disrupted; otherwise the gas of the annuli
 * outside r_tidal joins the satellite's hot gas and their stars the
 * central's intrahalo stars (RamPressureOn == 3).
 *
 * Ram pressure (Gunn & Gott 1972; paper eqs 117-118), per annulus:
 *   P_ram = rho_hot(R_sat) V_sat^2,  P_grav = 2 pi G Sigma_disc Sigma_gas,
 * Sigma_disc includes the stars when the gas and stellar discs lie within
 * ThetaThresh degrees.  RamPressureOn 1 and 3: once the innermost annulus is
 * stripped, all the cold gas goes; other annuli are stripped where
 * P_ram >= P_grav, unless shielded by the hot halo (M_cold + M_* < M_hot and
 * r_ann^2 < <R_hot^2>).  RamPressureOn 2: where P_ram >= P_grav, a fraction
 * P_ram / P_grav of the annulus goes per snapshot.  Stripped gas joins the satellite's hot (or CGM)
 * reservoir.
 *
 * Source bug: DarkSage's tidal loop zeroes DiscGas[i] instead of DiscGas[j], which
 * leaves the gas in the annulus after taking it out of ColdGas (gas is created).
 * As in the source, tidally stripped stars leave the ICS mean radius unchanged.
 *
 * Code units throughout (10^10 Msun/h, Mpc/h, km/s).
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "core_allvars.h"

#include "model_darksage_stripping.h"
#include "model_darksage_disc.h"
#include "model_darksage_stellar_pop.h"
#include "model_misc.h"

/* Move a fraction f of annulus i's gas to the satellite's hot reservoir */
static void strip_annulus_gas(struct GALAXY *s, const int i, const double f, const struct params *run_params)
{
    const double m = f * s->DiscGas[i], mz = f * s->DiscGasMetals[i];
    add_gas_to_hot_reservoir(s, run_params, m, mz);
    s->ColdGas -= m;
    s->MetalsColdGas -= mz;
    s->H2gas -= f * s->DiscH2[i];
    s->H1gas -= f * s->DiscHI[i];
    s->DiscGas[i] -= m;
    s->DiscGasMetals[i] -= mz;
    s->DiscH2[i] -= f * s->DiscH2[i];
    s->DiscHI[i] -= f * s->DiscHI[i];
}

/* Satellite mass: Mvir, but never below its baryons (DarkSage get_satellite_mass()) */
static double satellite_mass(const struct GALAXY *s)
{
    const double baryons = s->StellarMass + s->ColdGas + s->HotGas + s->CGMgas + s->BlackHoleMass + s->EjectedMass
                           + s->FountainGas + s->OutflowGas + s->ICS;
    return (s->Mvir > baryons) ? s->Mvir : baryons;
}

int darksage_ram_pressure_stripping(const int centralgal, const int gal, const double time, const int n_steps,
                                    struct GALAXY *galaxies, const struct params *run_params)
{
    const int mode = run_params->RamPressureOn;
    struct GALAXY *s = &galaxies[gal], *c = &galaxies[centralgal];
    if(mode == 0 || s->ColdGas <= 0.0) return 0;

    /* Physical distance and relative speed */
    const double r_sat = darksage_satellite_radius(c, s, run_params);
    double v2 = 0.0;
    for(int k = 0; k < 3; k++) {
        const double dv = s->Vel[k] - c->Vel[k];
        v2 += dv * dv;
    }
    if(r_sat <= 0.0) return 0;

    /* Tidal radius */
    const double M_sat = satellite_mass(s);
    const double M_host = darksage_host_mass_within(c, r_sat, M_sat, run_params);
    const double dr = 0.01 * r_sat;
    const double dMdr = (darksage_host_mass_within(c, r_sat + dr, M_sat, run_params) - M_host) / dr;
    const double denom = 2.0 * M_host - r_sat * dMdr;
    const double r_tidal = (denom > 0.0) ? r_sat * cbrt(M_sat / denom) : HUGE_VAL;
    const int tides = (mode == 3 && M_sat < run_params->ThreshMajorMerger * M_host);

    if(tides && s->DiscRadii[1] > r_tidal) return 1;   /* fully disrupted */

    const double P_ram = darksage_hot_density(c, r_sat, run_params) * v2;
    const double cos_angle = s->SpinStars[0] * s->SpinGas[0] + s->SpinStars[1] * s->SpinGas[1] + s->SpinStars[2] * s->SpinGas[2];
    const int coplanar = (cos_angle >= cos(run_params->ThetaThresh * M_PI / 180.0));
    const int unshielded = (s->ColdGas + s->StellarMass > s->HotGas + s->CGMgas);

    for(int i = 0; i < N_BINS; i++) {
        const double r_in = s->DiscRadii[i], r_out = s->DiscRadii[i+1];
        const double area = M_PI * (r_out * r_out - r_in * r_in);
        if(area > 0.0 && s->DiscGas[i] > 0.0) {
            const double Sigma_gas = s->DiscGas[i] / area;
            const double Sigma_disc = coplanar ? Sigma_gas + s->DiscStars[i] / area : Sigma_gas;
            const double P_grav = 2.0 * M_PI * run_params->G * Sigma_disc * Sigma_gas;

            if(mode == 2) {
                const double f = P_ram / P_grav / n_steps;
                if(P_ram >= P_grav) strip_annulus_gas(s, i, (f < 1.0) ? f : 1.0, run_params);
            } else if(P_ram >= P_grav && i == 0) {
                for(int j = 0; j < N_BINS; j++) strip_annulus_gas(s, j, 1.0, run_params);
                break;
            } else if(P_ram >= P_grav && (unshielded || 0.5 * (r_out * r_out + r_in * r_in) > s->R2_hot_av)) {
                strip_annulus_gas(s, i, 1.0, run_params);
            }
        }

        /* Everything beyond the tidal radius goes */
        if(tides && r_out >= r_tidal && i < N_BINS - 1) {
            for(int j = i + 1; j < N_BINS; j++) {
                strip_annulus_gas(s, j, 1.0, run_params);

                const double m = s->DiscStars[j];
                c->ICS += m;
                c->MetalsICS += s->DiscStarsMetals[j];
                c->ICS_disrupt += m;
                c->ICS_sum_mt += m * time;
                s->StellarMass -= m;
                s->MetalsStellarMass -= s->DiscStarsMetals[j];
                s->DiscStars[j] = s->DiscStarsMetals[j] = 0.0f;
                darksage_move_ages(s->DiscStarsAge[j], s->DiscStarsMetalsAge[j], 1.0, c->ICSAge, c->MetalsICSAge);
            }
            break;
        }
    }

    /* Float sums: keep the bulk fields non-negative */
    if(s->ColdGas < 0.0f) s->ColdGas = 0.0f;
    if(s->MetalsColdGas < 0.0f) s->MetalsColdGas = 0.0f;
    if(s->H2gas < 0.0f) s->H2gas = 0.0f;
    if(s->H1gas < 0.0f) s->H1gas = 0.0f;
    if(s->StellarMass < 0.0f) s->StellarMass = 0.0f;
    if(s->MetalsStellarMass < 0.0f) s->MetalsStellarMass = 0.0f;
    return 0;
}

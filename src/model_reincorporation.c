/*
 * model_reincorporation.c -- return of SN-ejected gas to the hot reservoir.
 *
 * Implements reincorporate_gas(), which moves gas from the EjectedMass
 * reservoir back into the hot gas reservoir once the halo's circular velocity
 * exceeds the supernova re-entry threshold (Vcrit = 445 km/s scaled by
 * ReIncorporationFactor).  With CGMrecipeOn, the destination reservoir is
 * regime-dependent: CGMgas for CGM-regime haloes, HotGas otherwise.
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "core_allvars.h"

#include "model_reincorporation.h"
#include "model_misc.h"
#include "model_darksage_feedback.h"
#include "model_darksage_disc.h"

/* -------------------------------------------------------------------------
 * File-scope empirical constants (lifted per STYLE_C.md SS8).
 * -------------------------------------------------------------------------*/

/* Critical velocity for gas reincorporation.
 * SN ejecta leave the disk at V_SN ~ 501 km/s; a halo can retain the gas when
 * its escape velocity (V_esc = sqrt(2)*V_vir) exceeds V_SN, i.e. when
 * V_vir > V_SN / sqrt(2) ~ 354.53 km/s.  Scaled further by
 * ReIncorporationFactor in the parameter file. */
// static const double REINC_VCRIT_KMS = 445.48;  /* km/s */
static const double REINC_VCRIT_KMS = 354.53;  /* km/s */

/*
 * reincorporate_gas -- move ejected gas back into the hot reservoir for one
 * central galaxy over timestep dt.
 *
 * Reincorporation rate: (Vvir/Vcrit - 1) * EjectedMass / t_dyn, where
 * t_dyn = Rvir/Vvir.  Only runs when Vvir > Vcrit.  With CGMrecipeOn, routes
 * to CGMgas (Regime 0) or HotGas (Regime 1); without it, always to HotGas.
 */
void reincorporate_gas(const int centralgal, const double dt, struct GALAXY *galaxies, const struct params *run_params)
{
    const double Vcrit = REINC_VCRIT_KMS * run_params->ReIncorporationFactor;

    if(Vcrit > 0.0 && galaxies[centralgal].Vvir > Vcrit && galaxies[centralgal].Rvir > 0.0) {
        // Note: Vvir > Vcrit already ensures Vvir > 0, so Rvir/Vvir is safe
        double reincorporated =
            ( galaxies[centralgal].Vvir / Vcrit - 1.0 ) *
            galaxies[centralgal].EjectedMass / (galaxies[centralgal].Rvir / galaxies[centralgal].Vvir) * dt;

        if(reincorporated > galaxies[centralgal].EjectedMass)
            reincorporated = galaxies[centralgal].EjectedMass;

        const double metallicity = get_metallicity(galaxies[centralgal].EjectedMass, galaxies[centralgal].MetalsEjectedMass);
        
        // Remove from ejected reservoir (same for all regimes)
        galaxies[centralgal].EjectedMass -= reincorporated;
        galaxies[centralgal].MetalsEjectedMass -= metallicity * reincorporated;
        
        // Add to appropriate hot reservoir (regime-dependent)
        add_gas_to_hot_reservoir(&galaxies[centralgal], run_params, reincorporated, metallicity * reincorporated);
    }
}

/* Bisection limits for the turnaround radius of outflowing gas (DarkSage reincorporate_gas()) */
static const int TURNAROUND_MAX_ITER = 200;
static const double TURNAROUND_TOL = 1.0e-3;
static const double TURNAROUND_R_MAX_FRAC = 10.0;

/*
 * Radius beyond Rvir where gas of specific energy e_spec turns around:
 * Phi_NFW(R) + 0.5 Vvir^2 = e_spec, bisected on [Rvir, 10 Rvir] to a relative
 * tolerance of 1e-3 in energy (DarkSage reincorporate_gas()).
 */
static double turnaround_radius(const int c, const double e_spec, const struct GALAXY *galaxies,
                                const struct params *run_params)
{
    const double thermal = 0.5 * galaxies[c].Vvir * galaxies[c].Vvir;
    double R_min = galaxies[c].Rvir, R_max = TURNAROUND_R_MAX_FRAC * galaxies[c].Rvir, R = R_min;
    for(int iter = 0; iter < TURNAROUND_MAX_ITER; iter++) {
        R = 0.5 * (R_min + R_max);
        const double e = darksage_nfw_potential(c, R, galaxies, run_params) + thermal;
        if(fabs((e - e_spec) / e_spec) < TURNAROUND_TOL) break;
        if(e > e_spec) R_max = R;
        else R_min = R;
    }
    return R;
}

/* Move mass from one reservoir to the fountain, mixing FountainTime with t_dyn */
static void to_fountain(struct GALAXY *g, const double mass, const double metals, const double t_dyn)
{
    if(mass <= 0.0) return;
    g->FountainTime = (g->FountainGas * g->FountainTime + mass * t_dyn) / (g->FountainGas + mass);
    g->FountainGas += mass;
    g->MetalsFountainGas += metals;
}

/*
 * darksage_reincorporate_gas -- DarkSAGEMode reservoir cycling for one central over dt.
 *
 * Algorithm (DarkSage reincorporate_gas(), central-galaxy branch):
 *   1. Fountain -> hot phase: a fraction dt / FountainTime returns (all of it
 *      if dt >= FountainTime), and FountainTime counts down by dt.  The gas
 *      joins CGMgas in CGM-regime haloes, HotGas otherwise.
 *   2. Ejected -> fountain on ReincTime in the same way.
 *   3. Outflow -> ejected: a fraction dt / OutflowTime leaves Rvir.  Its
 *      radial speed at Rvir, sqrt(2 (e_out - Phi(Rvir) - 0.5 Vvir^2)), and
 *      the radius R_ta where it turns around give the time to get there and
 *      back, 4 (R_ta - Rvir) / v, mixed into ReincTime by mass; the energy at
 *      turnaround is mixed into EjectedSpecificEnergy.  Gas that cannot pass
 *      Rvir falls straight into the fountain.
 *
 * FountainTime, ReincTime and OutflowTime [code time] are mass-weighted means
 * that count down; t_dyn = 0.1 / H(z) is the fountain time of new gas.
 */
void darksage_reincorporate_gas(const int centralgal, const double dt, struct GALAXY *galaxies,
                                const struct params *run_params)
{
    struct GALAXY *c = &galaxies[centralgal];
    const double t_dyn = get_dynamical_time(c->SnapNum, run_params);

    /* Step 1: fountain -> hot phase */
    if(c->FountainGas > 0.0f) {
        if(dt >= c->FountainTime) {
            add_gas_to_hot_reservoir(c, run_params, c->FountainGas, c->MetalsFountainGas);
            c->FountainGas = c->MetalsFountainGas = 0.0f;
            c->FountainTime = 0.0f;
        } else {
            const double f = dt / c->FountainTime;
            add_gas_to_hot_reservoir(c, run_params, f * c->FountainGas, f * c->MetalsFountainGas);
            c->FountainGas -= f * c->FountainGas;
            c->MetalsFountainGas -= f * c->MetalsFountainGas;
            c->FountainTime -= dt;
        }
    }

    /* Step 2: ejected -> fountain */
    if(c->EjectedMass > 0.0f) {
        if(dt >= c->ReincTime) {
            to_fountain(c, c->EjectedMass, c->MetalsEjectedMass, t_dyn);
            c->EjectedMass = c->MetalsEjectedMass = 0.0f;
            c->ReincTime = 0.0f;
        } else {
            const double f = dt / c->ReincTime;
            to_fountain(c, f * c->EjectedMass, f * c->MetalsEjectedMass, t_dyn);
            c->EjectedMass -= f * c->EjectedMass;
            c->MetalsEjectedMass -= f * c->MetalsEjectedMass;
            c->ReincTime -= dt;
        }
    }

    /* Step 3: outflow -> ejected */
    if(c->OutflowGas > 0.0f) {
        const double move = (c->OutflowTime > 0.0f) ? dt / c->OutflowTime : 1.0;
        const double eject = c->OutflowGas * ((move < 1.0) ? move : 1.0);

        const double thermal = 0.5 * c->Vvir * c->Vvir;
        const double ke = c->OutflowSpecificEnergy - c->EjectedPotential - thermal;
        const double v_r = (ke > 0.0) ? sqrt(2.0 * ke) : 0.0;
        const double R_ta = turnaround_radius(centralgal, c->OutflowSpecificEnergy, galaxies, run_params);
        const double reinc_new = (v_r > 0.0) ? 4.0 * (R_ta - c->Rvir) / v_r : 0.0;

        if(eject > 0.0 && reinc_new > 0.0) {
            const double e_ta = darksage_nfw_potential(centralgal, R_ta, galaxies, run_params) + thermal;
            c->ReincTime = (c->EjectedMass * c->ReincTime + eject * reinc_new) / (c->EjectedMass + eject);
            c->EjectedSpecificEnergy = (c->EjectedMass * c->EjectedSpecificEnergy + eject * e_ta) / (c->EjectedMass + eject);
        }

        if(reinc_new <= 0.0) {
            /* the outflow turns around inside Rvir: it is fountain gas */
            to_fountain(c, c->OutflowGas, c->MetalsOutflowGas, t_dyn);
            c->OutflowGas = c->MetalsOutflowGas = 0.0f;
            c->OutflowTime = 0.0f;
        } else if(move >= 1.0) {
            c->EjectedMass += c->OutflowGas;
            c->MetalsEjectedMass += c->MetalsOutflowGas;
            c->OutflowGas = c->MetalsOutflowGas = 0.0f;
            c->OutflowTime = 0.0f;
        } else {
            c->EjectedMass += eject;
            c->MetalsEjectedMass += move * c->MetalsOutflowGas;
            c->OutflowGas -= eject;
            c->MetalsOutflowGas -= move * c->MetalsOutflowGas;
            c->OutflowTime -= dt;
        }
    }
}

/*
 * darksage_fountain_from_growth -- adjust the outflowing and ejected
 * reservoirs of a central whose Rvir grew since the last snapshot
 * (DarkSage update_galactic_fountain_from_growth()).
 *
 *   Outflow: keep its travel time to the new Rvir.  The specific energy
 *     needed for that is target_e; the low-energy fraction
 *     (target_e - e_out) / target_e no longer gets out and joins the
 *     fountain, raising e_out to target_e.  If even that is impossible, all
 *     of it joins the fountain.
 *   Ejected: if its energy no longer reaches the new Rvir it all joins the
 *     fountain; otherwise ReincTime shrinks in proportion to the shorter
 *     distance from the turnaround radius to Rvir.
 * Called once per snapshot, before the substeps, when the halo accretes.
 */
void darksage_fountain_from_growth(const int centralgal, struct GALAXY *galaxies, const struct params *run_params)
{
    struct GALAXY *c = &galaxies[centralgal];
    if(c->Rvir <= c->prevRvir || c->Vvir <= 0.0) return;

    const double t_dyn = c->Rvir / c->Vvir;
    const double thermal = 0.5 * c->Vvir * c->Vvir;
    const double hot_e = darksage_hot_specific_energy(centralgal, galaxies);

    if(c->OutflowGas > 0.0f && c->OutflowTime > 0.0f) {
        const double radial_speed = 2.0 * c->Rvir / c->OutflowTime
                                    - sqrt(2.0 * (c->OutflowSpecificEnergy - c->Potential[0] - thermal));
        const double target_e = thermal + c->EjectedPotential + 0.5 * radial_speed * radial_speed;

        if(!(radial_speed > 0.0 && target_e > hot_e)) {
            to_fountain(c, c->OutflowGas, c->MetalsOutflowGas, t_dyn);
            c->OutflowGas = c->MetalsOutflowGas = 0.0f;
            c->OutflowTime = 0.0f;
        } else if(c->OutflowSpecificEnergy < target_e) {
            double reinc = c->OutflowGas * (target_e - c->OutflowSpecificEnergy) / target_e;
            if(reinc < 0.0) reinc = 0.0;
            if(reinc <= c->OutflowGas) {
                const double Z = get_metallicity(c->OutflowGas, c->MetalsOutflowGas);
                to_fountain(c, reinc, Z * reinc, t_dyn);
                c->OutflowGas -= reinc;
                c->MetalsOutflowGas -= Z * reinc;
                c->OutflowSpecificEnergy = (float)target_e;
            } else {
                to_fountain(c, c->OutflowGas, c->MetalsOutflowGas, t_dyn);
                c->OutflowGas = c->MetalsOutflowGas = 0.0f;
                c->OutflowTime = 0.0f;
            }
        }
    }

    if(c->EjectedMass > 0.0f) {
        if(c->EjectedSpecificEnergy < c->EjectedPotential + thermal) {
            to_fountain(c, c->EjectedMass, c->MetalsEjectedMass, t_dyn);
            c->EjectedMass = c->MetalsEjectedMass = 0.0f;
            c->ReincTime = 0.0f;
        } else {
            const double R_ta = turnaround_radius(centralgal, c->EjectedSpecificEnergy, galaxies, run_params);
            const double ratio = (R_ta - c->Rvir) / (R_ta - c->prevRvir);
            if(ratio < 1.0 && ratio >= 0.0) c->ReincTime *= ratio;
        }
    }
}

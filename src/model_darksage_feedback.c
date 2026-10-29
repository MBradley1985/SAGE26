/*
 * model_darksage_feedback.c -- DarkSage outflow energetics (DarkSAGEMode == 1).
 *
 * Gas pushed out of a disc annulus by supernovae either stays bound to the
 * halo (the fountain reservoir, which returns to the hot gas) or leaves Rvir
 * (the outflowing reservoir, which becomes the ejected reservoir and later
 * falls back).  Which of the two happens is decided by the wind speed the
 * gas would need to reach the hot halo's specific energy, compared with the
 * local escape speed (Stevens et al. 2024, Sec. 9; DarkSage source
 * calculate_feedback_masses(), model_starformation_and_feedback.c).
 *
 * Two feedback options use this:
 *   SupernovaRecipeOn == 1 -- SAGE26 sets how much gas leaves the annulus
 *       (eta x SFR, FIRE or fixed epsilon); the DarkSage wind-speed test only
 *       splits it into fountain and outflow (darksage_split_feedback()).
 *   SupernovaRecipeOn == 2 -- DarkSage's own energy-driven feedback: the SN
 *       energy sets both masses (darksage_feedback_masses()).
 *
 * The reservoirs belong to the central galaxy: SAGE26 sends a satellite's
 * reheated gas to its central.  Outflowing gas carries a specific energy that
 * sets its travel time to Rvir (OutflowTime) and, in model_reincorporation.c,
 * its turnaround radius and return time (ReincTime).
 *
 * Code units throughout (10^10 Msun/h, Mpc/h, km/s; energies per unit mass in
 * (km/s)^2).
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "core_allvars.h"

#include "model_darksage_feedback.h"
#include "model_misc.h"

/* DarkSage vertical velocity dispersion of cold gas, sigma_z = 11 + 11.3 z km/s
 * (model_starformation_and_feedback.c; Stevens et al. 2024 Sec. 9) [cm/s] */
static const double DS_SIGMA_Z0_CGS = 1.1e6;
static const double DS_SIGMA_Z1_CGS = 1.13e6;

/* DarkSage fraction of an annulus's metals carried out by its outflow
 * (update_from_feedback(), SupernovaRecipeOn == 2 only) */
static const double DS_METAL_ENTRAINMENT = 0.1;

/*
 * Wind kinematics of annulus i (DarkSage calculate_feedback_masses()).
 *
 *   cold_e   -- specific energy of the annulus gas: 0.5 (v_rot^2 + sigma_z^2)
 *               plus the mean potential of the annulus edges
 *   reheat_e -- energy per unit mass needed to raise it to the hot halo's
 *   v_wind   -- launch speed that delivers reheat_e: v_wind^2 = 2 reheat_e - Vvir^2
 *   esc2     -- escape speed squared from the annulus to Rvir
 *
 * The wind is launched in a random direction relative to the rotation, so
 * the gas speed lies between |v_wind - v_rot| and v_wind + v_rot:
 *   all of it escapes if even the slowest does; none if even the fastest
 *   cannot; in between, ratio = reheated / ejected follows from the fraction
 *   of directions that escape.
 */
/* Rotation speed of annulus i: mid-bin j over its mass-weighted mid radius */
static double annulus_velocity(const struct GALAXY *g, const int i, const struct params *run_params)
{
    const double r_ann = sqrt(0.5 * (g->DiscRadii[i] * g->DiscRadii[i] + g->DiscRadii[i+1] * g->DiscRadii[i+1]));
    return (r_ann > 0.0) ? 0.5 * (run_params->DiscBinEdge[i] + run_params->DiscBinEdge[i+1]) / r_ann : 0.0;
}

/* Specific energy of annulus i's cold gas: 0.5 (v_rot^2 + sigma_z^2) plus the mean potential of its edges */
static double annulus_cold_energy(const struct GALAXY *g, const int i, const struct params *run_params)
{
    const double v_ann = annulus_velocity(g, i, run_params);
    const double v_z = (DS_SIGMA_Z0_CGS + DS_SIGMA_Z1_CGS * run_params->ZZ[g->SnapNum]) / run_params->UnitVelocity_in_cm_per_s;
    return 0.5 * (v_ann * v_ann + v_z * v_z + g->Potential[i] + g->Potential[i+1]);
}

enum wind_case { WIND_ALL_REHEATED, WIND_SPLIT, WIND_ALL_EJECTED };

struct wind {
    enum wind_case kind;
    double cold_e, reheat_e;
    double ratio;      /* reheated / ejected mass for WIND_SPLIT */
    double eject_e;    /* specific energy of the ejected gas */
};

static struct wind wind_kinematics(const int p, const int centralgal, const int i,
                                   const struct GALAXY *galaxies, const struct params *run_params)
{
    const struct GALAXY *g = &galaxies[p];
    struct wind w = {WIND_ALL_REHEATED, 0.0, 0.0, 0.0, 0.0};

    const double v_ann = annulus_velocity(g, i, run_params);
    const double v_therm2 = g->Vvir * g->Vvir;

    w.cold_e = annulus_cold_energy(g, i, run_params);
    w.reheat_e = darksage_hot_specific_energy(p, galaxies) - w.cold_e;

    const double v_wind2 = 2.0 * w.reheat_e - v_therm2;
    const double esc2 = 2.0 * (g->EjectedPotential - g->Potential[i]);

    if(w.reheat_e < 0.0) {
        /* Annulus gas already more energetic than the hot halo: it can only be ejected */
        w.kind = WIND_ALL_EJECTED;
        double floor_e = (g->EjectedPotential > w.cold_e) ? g->EjectedPotential : w.cold_e;
        floor_e += 0.5 * v_therm2;
        const double out_e = galaxies[centralgal].OutflowSpecificEnergy;
        w.eject_e = (out_e > floor_e) ? out_e : floor_e;
        return w;
    }
    if(v_wind2 <= 0.0) return w;   /* no launch speed beyond the thermal one: all reheated */

    const double v_wind = sqrt(v_wind2);
    if((v_wind - v_ann) * (v_wind - v_ann) > esc2) {
        w.kind = WIND_ALL_EJECTED;
        const double e1 = w.cold_e + 0.5 * (v_therm2 + v_wind2);
        const double e2 = g->EjectedPotential + 0.5 * v_therm2;
        w.eject_e = (e1 > e2) ? e1 : e2;
    } else if((v_wind + v_ann) * (v_wind + v_ann) > esc2) {
        w.kind = WIND_SPLIT;
        w.ratio = 1.0 / (0.5 - (esc2 - v_ann * v_ann - v_wind2) / (4.0 * v_ann * v_wind)) - 1.0;
        w.eject_e = 0.25 * (g->Potential[i] + g->Potential[i+1] + (v_wind + v_ann) * (v_wind + v_ann))
                    + 0.5 * (g->EjectedPotential + v_therm2);
    }
    return w;
}

/*
 * Travel time of newly outflowing gas from the centre to Rvir, mixed into
 * OutflowTime by mass (DarkSage update_outflow_time()).  The radial speed is
 * the mean of its values at r = 0 and at Rvir from energy conservation.
 * Must be called before new_mass is added to OutflowGas.
 */
static void update_outflow_time(struct GALAXY *c, const double new_mass, const double new_e)
{
    if(new_mass <= 0.0) return;

    const double thermal = 0.5 * c->Vvir * c->Vvir;
    const double ke_initial = new_e - c->Potential[0] - thermal;
    double ke_final = new_e - c->EjectedPotential - thermal;
    if(ke_final < 0.0) ke_final = 0.0;
    if(ke_initial <= 0.0) return;

    const double new_time = M_SQRT2 * c->Rvir / (sqrt(ke_initial) + sqrt(ke_final));
    c->OutflowTime = (c->OutflowGas * c->OutflowTime + new_mass * new_time) / (c->OutflowGas + new_mass);
}

double darksage_hot_specific_energy(const int p, const struct GALAXY *galaxies)
{
    const struct GALAXY *g = &galaxies[p];
    const double j_hot = 2.0 * g->Vvir * g->CoolScaleRadius;
    const double rot = (g->R2_hot_av > 0.0) ? j_hot * j_hot / g->R2_hot_av : 0.0;
    return g->HotGasPotential + 0.5 * (g->Vvir * g->Vvir + rot);
}

void darksage_split_feedback(const int p, const int centralgal, const int i, const double removed,
                             double *reheated, double *ejected, double *eject_e,
                             const struct GALAXY *galaxies, const struct params *run_params)
{
    *reheated = removed;
    *ejected = 0.0;
    *eject_e = 0.0;
    if(removed <= 0.0) return;

    const struct wind w = wind_kinematics(p, centralgal, i, galaxies, run_params);
    if(w.kind == WIND_ALL_EJECTED) {
        *reheated = 0.0;
        *ejected = removed;
    } else if(w.kind == WIND_SPLIT) {
        *ejected = removed / (1.0 + w.ratio);
        *reheated = removed - *ejected;
    }
    *eject_e = w.eject_e;
}

void darksage_feedback_masses(const int p, const int centralgal, const int i, const int stars_fixed,
                              const double max_consume, double *stars, double *reheated, double *ejected,
                              double *eject_e, const struct GALAXY *galaxies, const struct params *run_params)
{
    *reheated = *ejected = *eject_e = 0.0;
    const double gas = (max_consume < galaxies[p].DiscGas[i]) ? max_consume : galaxies[p].DiscGas[i];
    if(*stars > gas) *stars = gas;
    if(*stars <= 0.0 || gas <= 0.0) return;

    /* SN energy of the stars formed: SAGE26's eta_SN SNe per unit mass of E_SN each.  DarkSage
     * instead integrates its IMF (get_RecycleFraction_and_NumSNperMass(), Type Ia/II ratio 0.15),
     * giving 1.282e-2 SNe per Msun over a population's lifetime, with EnergySN = 1.1e51 erg in its
     * MTNG parameter file; set EtaSN and EnergySN to those values to compare with DarkSage. */
    double energy = *stars * run_params->EtaSNcode * run_params->EnergySNcode;

    const struct wind w = wind_kinematics(p, centralgal, i, galaxies, run_params);
    if(w.kind == WIND_ALL_EJECTED) {
        *ejected = energy / (w.eject_e - w.cold_e);
    } else if(w.kind == WIND_SPLIT) {
        *ejected = energy / (w.ratio * w.reheat_e + w.eject_e - w.cold_e);
        *reheated = w.ratio * *ejected;
    } else {
        *reheated = energy / w.reheat_e;
    }
    *eject_e = w.eject_e;

    /* Can't use more cold gas than the annulus has: balance SF and feedback, or for a
     * feedback-free burst (stars_fixed) cap the feedback at the gas the burst leaves */
    const double locked = (1.0 - run_params->RecycleFraction) * *stars;
    const double used = locked + *reheated + *ejected;
    if(used > gas) {
        if(stars_fixed) {
            const double left = gas - locked;
            const double fac = (left > 0.0) ? left / (*reheated + *ejected) : 0.0;
            *reheated *= fac;
            *ejected *= fac;
        } else {
            const double fac = gas / used;
            *stars *= fac;
            *reheated *= fac;
            *ejected *= fac;
        }
    }
}

double darksage_outflow_metallicity(const double Z, const struct params *run_params)
{
    return (run_params->SupernovaRecipeOn == 2) ? DS_METAL_ENTRAINMENT * Z : Z;
}

void darksage_add_to_fountain(const int centralgal, const double mass, const double metals,
                              struct GALAXY *galaxies, const struct params *run_params)
{
    if(mass <= 0.0) return;
    struct GALAXY *c = &galaxies[centralgal];
    const double t_fountain = get_dynamical_time(c->SnapNum, run_params);
    c->FountainTime = (c->FountainGas * c->FountainTime + mass * t_fountain) / (c->FountainGas + mass);
    c->FountainGas += mass;
    c->MetalsFountainGas += metals;
}

void darksage_add_to_outflow(const int centralgal, const double mass, const double metals, const double e_spec,
                             struct GALAXY *galaxies)
{
    if(mass <= 0.0) return;
    struct GALAXY *c = &galaxies[centralgal];
    c->OutflowSpecificEnergy = (c->OutflowGas * c->OutflowSpecificEnergy + mass * e_spec) / (c->OutflowGas + mass);
    update_outflow_time(c, mass, e_spec);
    c->OutflowGas += mass;
    c->MetalsOutflowGas += metals;
}

/*
 * darksage_eject_hot -- push hot-phase gas out of the halo (DarkSage
 * update_from_ejection()).
 *
 * Takes ejected_mass from the hot-phase reservoir (CGMgas in CGM-regime
 * haloes, HotGas otherwise) and the fountain, in proportion to their masses,
 * into the outflowing reservoir.  Its specific energy is at least the
 * energy needed to reach Rvir, EjectedPotential + 0.5 Vvir^2.
 */
void darksage_eject_hot(const int centralgal, double ejected_mass, struct GALAXY *galaxies,
                        const struct params *run_params)
{
    struct GALAXY *c = &galaxies[centralgal];
    if(ejected_mass <= 0.0) return;

    const int cgm = (run_params->CGMrecipeOn == 1 && c->Regime == 0);
    float *hot = cgm ? &c->CGMgas : &c->HotGas;
    float *hot_metals = cgm ? &c->MetalsCGMgas : &c->MetalsHotGas;
    const double pool = *hot + c->FountainGas;
    if(pool <= 0.0) return;
    if(ejected_mass > pool) ejected_mass = pool;

    const double e_min = c->EjectedPotential + 0.5 * c->Vvir * c->Vvir;
    if(c->OutflowSpecificEnergy < e_min) c->OutflowSpecificEnergy = (float)e_min;

    const double f_hot = *hot / pool;
    const double Z_hot = get_metallicity(*hot, *hot_metals);
    const double Z_fountain = get_metallicity(c->FountainGas, c->MetalsFountainGas);
    const double metals = ejected_mass * (f_hot * Z_hot + (1.0 - f_hot) * Z_fountain);

    update_outflow_time(c, ejected_mass, c->OutflowSpecificEnergy);
    c->OutflowGas += ejected_mass;
    c->MetalsOutflowGas += metals;
    *hot -= ejected_mass * f_hot;
    *hot_metals -= ejected_mass * f_hot * Z_hot;
    c->FountainGas -= ejected_mass * (1.0 - f_hot);
    c->MetalsFountainGas -= ejected_mass * (1.0 - f_hot) * Z_fountain;

    if(*hot < 0.0f) *hot = 0.0f;
    if(*hot_metals < 0.0f) *hot_metals = 0.0f;
    if(c->FountainGas < 0.0f) c->FountainGas = 0.0f;
    if(c->MetalsFountainGas < 0.0f) c->MetalsFountainGas = 0.0f;
}

/*
 * darksage_quasar_wind -- quasar-mode feedback from black-hole growth BHaccrete,
 * annulus by annulus (DarkSage quasar_mode_wind(); Stevens et al. 2024 Sec. 9.2).
 *
 * The wind energy is SAGE26's, QuasarModeEfficiency * 0.1 * BHaccrete c^2 (as
 * quasar_mode_wind()); DarkSage uses RadiativeEfficiency * BHaccrete c^2 with
 * RadiativeEfficiency = 0.40, i.e. QuasarModeEfficiency = 4.
 *
 * Algorithm:
 *   1. The ejected gas leaves with specific energy e_ej = max(e_out,
 *      EjectedPotential + 0.5 (Vvir^2 + j_hot^2 / R2_hot_av)), e_out that of
 *      the gas already outflowing.
 *   2. From the innermost annulus out, lift annulus k's gas from its own
 *      specific energy to e_ej into the outflowing reservoir until the energy
 *      runs out (all of it where e_ej is no higher than the annulus's).
 *   3. Energy left over ejects hot-phase and fountain gas, at
 *      e_ej - e_hot per unit mass (darksage_eject_hot()).
 * ColdGas and MetalsColdGas are reset to the annulus sums.
 */
void darksage_quasar_wind(const int p, const double BHaccrete, struct GALAXY *galaxies, const struct params *run_params)
{
    struct GALAXY *g = &galaxies[p];
    const double c_code = C / run_params->UnitVelocity_in_cm_per_s;
    double energy = run_params->QuasarModeEfficiency * 0.1 * BHaccrete * c_code * c_code;
    if(energy <= 0.0) return;

    /* Step 1: specific energy of the wind's gas */
    const double j_hot = 2.0 * g->Vvir * g->CoolScaleRadius;
    const double hot_tk = 0.5 * (g->Vvir * g->Vvir + ((g->R2_hot_av > 0.0) ? j_hot * j_hot / g->R2_hot_av : 0.0));
    const double hot_e = g->HotGasPotential + hot_tk;
    double eject_e = g->EjectedPotential + hot_tk;
    if(g->OutflowGas > 0.0f && g->OutflowSpecificEnergy > eject_e) eject_e = g->OutflowSpecificEnergy;

    /* Step 2: blow out the disc from the centre */
    for(int k = 0; k < N_BINS && energy > 0.0; k++) {
        const double gas = g->DiscGas[k];
        if(gas <= 0.0) continue;
        double delta = eject_e - annulus_cold_energy(g, k, run_params);
        if(delta < 0.0) delta = 0.0;
        const double m_max = (delta > 0.0) ? energy / delta : gas;

        if(m_max >= gas) {
            darksage_add_to_outflow(p, gas, g->DiscGasMetals[k], eject_e, galaxies);
            energy -= gas * delta;
            g->DiscGas[k] = g->DiscGasMetals[k] = 0.0f;
            g->DiscH2[k] = 0.0f;
        } else {
            const double Z = get_metallicity(gas, g->DiscGasMetals[k]);
            darksage_add_to_outflow(p, m_max, Z * m_max, eject_e, galaxies);
            g->DiscGas[k] -= m_max;
            g->DiscGasMetals[k] -= Z * m_max;
            energy = 0.0;
        }
    }
    double gas_sum = 0.0, metals_sum = 0.0;
    for(int k = 0; k < N_BINS; k++) {
        gas_sum += g->DiscGas[k];
        metals_sum += g->DiscGasMetals[k];
    }
    g->ColdGas = gas_sum;
    g->MetalsColdGas = metals_sum;

    /* Step 3: remaining energy ejects hot-phase gas */
    const double delta_hot = eject_e - hot_e;
    if(energy > 0.0 && delta_hot > 0.0) {
        const int cgm = (run_params->CGMrecipeOn == 1 && g->Regime == 0);
        const double pool = (cgm ? g->CGMgas : g->HotGas) + g->FountainGas;
        double m_hot = energy / delta_hot;
        if(m_hot > pool) m_hot = pool;
        if(m_hot > 0.0) {
            g->OutflowSpecificEnergy = (float)((g->OutflowGas * g->OutflowSpecificEnergy + m_hot * eject_e) / (g->OutflowGas + m_hot));
            darksage_eject_hot(p, m_hot, galaxies, run_params);
        }
    }
}

/*
 * model_darksage_instability.c -- annulus-resolved disc instability (DarkSAGEMode == 1).
 *
 * Replaces check_disk_instability() when DarkSAGEMode == 1, following
 * DarkSage's check_disk_instability() and deal_with_unstable_gas()
 * (model_disk_instability.c; Stevens, Croton & Mutch 2016 Sec. 3.7;
 * Stevens et al. 2024 Sec. 8).  Each annulus is tested against the
 * Toomre Q of its gas and stars, combined with the Romeo & Wiegert (2011)
 * two-fluid weighting W = 2 sigma c_s / (sigma^2 + c_s^2).
 *
 * Gas pass, outer annulus to inner:
 *   The unstable gas is the part that must go for Q_gas to reach its
 *   minimum.  GasSinkRate of it sinks: it is split between the neighbouring
 *   annuli so that angular momentum is conserved (from annulus 0 the inward
 *   share feeds the black hole).  The rest forms stars in the annulus, with
 *   the usual SN feedback (darksage_instability_burst()).  Black-hole growth
 *   drives a quasar wind.
 * Stellar pass, outer annulus to inner:
 *   StarSinkRate of the unstable stars migrate the same way (from annulus 0
 *   inward into the secular bulge); the rest of the instability is removed
 *   by heating the annulus's stars until Q_star = Q_star_min.
 *
 * Gas and stellar discs more than ThetaThresh degrees apart are tested as
 * separate one-component discs.
 *
 * Code units throughout (10^10 Msun/h, Mpc/h, km/s).
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "core_allvars.h"

#include "model_darksage_instability.h"
#include "model_starformation_and_feedback.h"
#include "model_mergers.h"
#include "model_misc.h"
#include "model_darksage_stellar_pop.h"

/* DarkSage cold-gas velocity dispersion (= sound speed here), c_s = 11 + 11.3 z km/s [cm/s] */
static const double DS_CS0_CGS = 1.1e6;
static const double DS_CS1_CGS = 1.13e6;
/* Stellar-disc Toomre factor 3.36 / pi inverted (Toomre 1964) */
static const double TOOMRE_STARS = 0.935;
/* DarkSage minimum unstable mass that is acted on, and minimum mass that forms stars
 * (MIN_STARFORMATION in core_allvars.h) [10^10 Msun/h] */
static const double DS_MIN_UNSTABLE = 1.0e-10;
static const double DS_MIN_STARFORMATION = 1.0e-10;

double darksage_cold_gas_dispersion(const double z, const struct params *run_params)
{
    return (DS_CS0_CGS + DS_CS1_CGS * z) / run_params->UnitVelocity_in_cm_per_s;
}

/* Epicyclic frequency of annulus i from its j-bin edges: kappa^2 = 2 j / r^3 dj/dr */
static double epicyclic_frequency(const int i, const struct GALAXY *g, const struct params *run_params)
{
    const double *edge = run_params->DiscBinEdge;
    const double r_inner = g->DiscRadii[i], r_outer = g->DiscRadii[i+1];
    if(r_outer <= r_inner) return 0.0;
    const double j = (i > 0) ? edge[i] : edge[i+1];
    const double r = (i > 0) ? r_inner : r_outer;
    return sqrt(2.0 * j / (r * r * r) * (edge[i+1] - edge[i]) / (r_outer - r_inner));
}

/*
 * Split mass leaving annulus i between i-1 (m_down) and i+1 (m_up) so its
 * angular momentum is unchanged: m_up j_gain = m_down j_lose, with j_gain and
 * j_lose the j steps to the neighbouring bin centres.
 */
static void j_split(const int i, const double mass, const struct params *run_params, double *m_down, double *m_up)
{
    const double *edge = run_params->DiscBinEdge;
    const double j_gain = 0.5 * (edge[i+2] - edge[i]);
    const double j_lose = (i > 0) ? 0.5 * (edge[i+1] - edge[i-1]) : 0.5 * (edge[i+1] - edge[i]);
    *m_up = j_lose / (j_gain + j_lose) * mass;
    *m_down = *m_up * j_gain / j_lose;
}

/* Mass-weighted quadrature mix of a velocity dispersion */
static float mix_dispersion(const double m_old, const double s_old, const double m_add, const double s_add)
{
    return (m_old + m_add > 0.0) ? (float)sqrt((m_old * s_old * s_old + m_add * s_add * s_add) / (m_old + m_add)) : (float)s_add;
}

/*
 * Sink unstable gas from annulus i (DarkSage deal_with_unstable_gas()) and
 * return the gas left to form stars there.
 */
static double sink_unstable_gas(const int p, const int i, double unstable_gas, double *bh_growth,
                                struct GALAXY *galaxies, const struct params *run_params)
{
    struct GALAXY *g = &galaxies[p];
    if(unstable_gas > g->DiscGas[i]) unstable_gas = g->DiscGas[i];
    const double Z = get_metallicity(g->DiscGas[i], g->DiscGasMetals[i]);

    double gas_sink = run_params->GasSinkRate * unstable_gas;
    if(unstable_gas - gas_sink < DS_MIN_STARFORMATION) gas_sink = unstable_gas;

    g->DiscGas[i] -= gas_sink;
    g->DiscGasMetals[i] -= Z * gas_sink;

    if(i == N_BINS - 1) {
        g->DiscGas[i-1] += gas_sink;
        g->DiscGasMetals[i-1] += Z * gas_sink;
    } else {
        double m_down, m_up;
        j_split(i, gas_sink, run_params, &m_down, &m_up);
        if(i == 0) {
            g->BlackHoleMass += m_down;
            g->QuasarModeBHaccretionMass += m_down;   /* instability-fed (quasar-mode) accretion */
            *bh_growth += m_down;
        } else {
            g->DiscGas[i-1] += m_down;
            g->DiscGasMetals[i-1] += Z * m_down;
        }
        g->DiscGas[i+1] += m_up;
        g->DiscGasMetals[i+1] += Z * m_up;
    }
    if(g->DiscGas[i] < 0.0f) g->DiscGas[i] = 0.0f;
    if(g->DiscGasMetals[i] < 0.0f) g->DiscGasMetals[i] = 0.0f;

    return unstable_gas - gas_sink;
}

/* Move unstable stars out of annulus i, conserving j; from annulus 0 the inward share joins the secular bulge */
static void sink_unstable_stars(const int p, const int i, const double unstable_stars,
                                struct GALAXY *galaxies, const struct params *run_params)
{
    struct GALAXY *g = &galaxies[p];
    const double Z = get_metallicity(g->DiscStars[i], g->DiscStarsMetals[i]);
    const double sigma = g->VelDispStars[i];
    const double m_before = g->DiscStars[i];

    g->DiscStars[i] -= unstable_stars;
    g->DiscStarsMetals[i] = Z * g->DiscStars[i];

    if(i == N_BINS - 1) {
        g->VelDispStars[i-1] = mix_dispersion(g->DiscStars[i-1], g->VelDispStars[i-1], unstable_stars, sigma);
        g->DiscStars[i-1] += unstable_stars;
        g->DiscStarsMetals[i-1] += Z * unstable_stars;
        darksage_move_ages(g->DiscStarsAge[i], g->DiscStarsMetalsAge[i], unstable_stars / m_before,
                           g->DiscStarsAge[i-1], g->DiscStarsMetalsAge[i-1]);
        return;
    }

    double m_down, m_up;
    j_split(i, unstable_stars, run_params, &m_down, &m_up);
    /* the migrating stars carry the annulus's age mix: m_down of it inwards, then m_up of the rest outwards */
    darksage_move_ages(g->DiscStarsAge[i], g->DiscStarsMetalsAge[i], m_down / m_before,
                       (i > 0) ? g->DiscStarsAge[i-1] : g->InstabilityBulgeMassAge,
                       (i > 0) ? g->DiscStarsMetalsAge[i-1] : g->InstabilityBulgeMetalsAge);
    darksage_move_ages(g->DiscStarsAge[i], g->DiscStarsMetalsAge[i], m_up / (m_before - m_down),
                       g->DiscStarsAge[i+1], g->DiscStarsMetalsAge[i+1]);
    if(i > 0) {
        g->VelDispStars[i-1] = mix_dispersion(g->DiscStars[i-1], g->VelDispStars[i-1], m_down, sigma);
        g->DiscStars[i-1] += m_down;
        g->DiscStarsMetals[i-1] += Z * m_down;
    } else {
        /* Disc -> secular (instability-driven) bulge; StellarMass is unchanged */
        for(int s = 0; s < 3; s++) {
            g->SpinSecularBulge[s] = (float)(g->SpinSecularBulge[s] * g->SecularBulgeMass / (g->SecularBulgeMass + m_down));
        }
        g->VelDispBulge = mix_dispersion(g->SecularBulgeMass, g->VelDispBulge, m_down, sigma);
        g->SecularBulgeMass += m_down;
        g->SecularMetalsBulgeMass += Z * m_down;
        g->BulgeMass += m_down;
        g->InstabilityBulgeMass += m_down;
        g->MetalsBulgeMass += Z * m_down;
        update_instability_bulge_radius(p, m_down, g->DiskScaleRadius, galaxies, run_params);
    }
    g->VelDispStars[i+1] = mix_dispersion(g->DiscStars[i+1], g->VelDispStars[i+1], m_up, sigma);
    g->DiscStars[i+1] += m_up;
    g->DiscStarsMetals[i+1] += Z * m_up;
}

/*
 * check_disk_instability_darksage -- one substep of DarkSage disc instability.
 *
 * Algorithm (DarkSage check_disk_instability()):
 *   1. Gas pass, i = N_BINS-1 .. 0, over annuli with gas:
 *        Q_gas  = c_s kappa (r_out^2 - r_in^2) / (G m_gas)
 *        Q_star = 0.935 kappa sigma (r_out^2 - r_in^2) / (G m_star)
 *      With coplanar stars (angle <= ThetaThresh) the two-fluid
 *      Q_tot = 1 / (W/Q_gas + 1/Q_star) if Q_gas >= Q_star, else
 *      1 / (1/Q_gas + W/Q_star); stable if Q_tot >= QTotMin.  If both Q are
 *      below QTotMin + W the gas must reach that; if only the gas is, it must
 *      reach the Q_gas that brings Q_tot to QTotMin; otherwise the stars
 *      resolve it.  Without coplanar stars Q_gas_min = QTotMin.
 *      Unstable gas m_gas (1 - Q_gas / Q_gas_min) sinks (GasSinkRate) and the
 *      rest forms stars in the annulus with SN feedback.
 *   2. The black hole growth from annulus 0 drives a quasar wind.
 *   3. Stellar pass, i = N_BINS-1 .. 0, over annuli with stars: Q_star_min
 *      from the two-fluid Q (or QTotMin alone); unstable stars
 *      StarSinkRate m_star (1 - Q_star / Q_star_min) migrate with
 *      StarSinkRate = 1 - (1 - GasSinkRate) c_s / sigma, and the rest heat the
 *      annulus: sigma *= (1 - StarSinkRate) Q_star_min / Q_star + StarSinkRate.
 *   4. ColdGas and MetalsColdGas are reset to the annulus sums.
 */
void check_disk_instability_darksage(const int p, const int centralgal, const double dt, const int step,
                                     struct GALAXY *galaxies, const struct params *run_params)
{
    struct GALAXY *g = &galaxies[p];
    const double G = run_params->G;
    const double Q_min = run_params->QTotMin;
    const double c_s = darksage_cold_gas_dispersion(run_params->ZZ[g->SnapNum], run_params);

    double disc_gas = 0.0, disc_stars = 0.0;
    for(int i = 0; i < N_BINS; i++) {
        disc_gas += g->DiscGas[i];
        disc_stars += g->DiscStars[i];
    }
    if(disc_gas <= 0.0 && disc_stars <= 0.0) return;

    double cos_angle = g->SpinStars[0] * g->SpinGas[0] + g->SpinStars[1] * g->SpinGas[1] + g->SpinStars[2] * g->SpinGas[2];
    if(cos_angle > 1.0) cos_angle = 1.0;
    if(cos_angle < -1.0) cos_angle = -1.0;
    const int coplanar = (acos(cos_angle) * 180.0 / M_PI <= run_params->ThetaThresh);

    /* Step 1: gas instabilities */
    double bh_growth = 0.0;
    for(int i = N_BINS - 1; i >= 0; i--) {
        if(g->DiscGas[i] <= 0.0f) continue;
        const double kappa = epicyclic_frequency(i, g, run_params);
        if(kappa <= 0.0) continue;
        const double area = g->DiscRadii[i+1] * g->DiscRadii[i+1] - g->DiscRadii[i] * g->DiscRadii[i];
        const double sigma = g->VelDispStars[i];
        const double Q_gas = c_s * kappa * area / (G * g->DiscGas[i]);

        double Q_gas_min = Q_min;
        if(g->DiscStars[i] > 0.0f && coplanar) {
            const double Q_star = kappa * sigma * TOOMRE_STARS * area / (G * g->DiscStars[i]);
            const double W = 2.0 * sigma * c_s / (sigma * sigma + c_s * c_s);
            const double Q_tot = (Q_gas >= Q_star) ? 1.0 / (W / Q_gas + 1.0 / Q_star) : 1.0 / (1.0 / Q_gas + W / Q_star);
            if(Q_tot >= Q_min) continue;

            const double Q_stable = Q_min + W;
            if(Q_gas < Q_stable && Q_star < Q_stable) Q_gas_min = Q_stable;
            else if(Q_gas < Q_stable) Q_gas_min = 1.0 / (1.0 / Q_min - W / Q_star);
            else continue;   /* the stars' job to restore stability */
        }
        if(Q_gas >= Q_gas_min) continue;

        const double unstable_gas = g->DiscGas[i] * (1.0 - Q_gas / Q_gas_min);
        if(unstable_gas <= DS_MIN_UNSTABLE) continue;

        const double stars = sink_unstable_gas(p, i, unstable_gas, &bh_growth, galaxies, run_params);
        if(g->DiscGas[i] > 0.0f && stars > 0.0) {
            darksage_instability_burst(p, centralgal, i, stars, stars / (1.0 - run_params->RecycleFraction),
                                       dt, step, galaxies, run_params);
        }
    }

    /* Step 2: quasar wind from the instability-fed black hole growth */
    double gas_sum = 0.0, metals_sum = 0.0;
    for(int i = 0; i < N_BINS; i++) {
        gas_sum += g->DiscGas[i];
        metals_sum += g->DiscGasMetals[i];
    }
    g->ColdGas = gas_sum;
    g->MetalsColdGas = metals_sum;
    if(bh_growth > 0.0 && run_params->AGNrecipeOn > 0) {
        quasar_mode_wind(p, bh_growth, galaxies, run_params);
    }

    /* Step 3: stellar instabilities */
    for(int i = N_BINS - 1; i >= 0; i--) {
        if(g->DiscStars[i] <= 0.0f) continue;
        const double kappa = epicyclic_frequency(i, g, run_params);
        if(kappa <= 0.0) continue;
        const double area = g->DiscRadii[i+1] * g->DiscRadii[i+1] - g->DiscRadii[i] * g->DiscRadii[i];
        const double sigma = g->VelDispStars[i];
        if(sigma <= 0.0) continue;
        const double Q_star = kappa * sigma * TOOMRE_STARS * area / (G * g->DiscStars[i]);

        double Q_star_min = Q_min;
        if(g->DiscGas[i] > 0.0f && coplanar) {
            const double Q_gas = c_s * kappa * area / (G * g->DiscGas[i]);
            const double W = 2.0 * sigma * c_s / (sigma * sigma + c_s * c_s);
            double Q_tot;
            if(Q_gas >= Q_star) {
                Q_tot = 1.0 / (W / Q_gas + 1.0 / Q_star);
                Q_star_min = (1.0 / Q_min - W / Q_gas > 0.0) ? 1.0 / (1.0 / Q_min - W / Q_gas) : 1.0;
            } else {
                Q_tot = 1.0 / (1.0 / Q_gas + W / Q_star);
                Q_star_min = (1.0 / Q_min - 1.0 / Q_gas > 0.0) ? W / (1.0 / Q_min - 1.0 / Q_gas) : 1.0;
            }
            /* 0.99 absorbs rounding (DarkSage) */
            if(Q_tot >= 0.99 * Q_min || Q_star >= 0.99 * Q_star_min) continue;
        }
        if(Q_star >= Q_star_min) continue;

        const double star_sink_rate = 1.0 - (1.0 - run_params->GasSinkRate) * c_s / sigma;
        double unstable_stars = star_sink_rate * g->DiscStars[i] * (1.0 - Q_star / Q_star_min);
        if(unstable_stars > g->DiscStars[i]) unstable_stars = g->DiscStars[i];
        if(unstable_stars <= DS_MIN_UNSTABLE) continue;

        sink_unstable_stars(p, i, unstable_stars, galaxies, run_params);

        /* What the migration did not fix is fixed by heating the annulus's stars */
        if(g->DiscStars[i] > 0.0f) {
            g->VelDispStars[i] *= (float)((1.0 - star_sink_rate) * Q_star_min / Q_star + star_sink_rate);
        }
    }
}

/*
 * model_darksage_instability.c -- where disc instability takes its mass from (DarkSAGEMode >= 1).
 *
 * SAGE26's check_disk_instability() decides how much of the disc is unstable,
 * in every mode.  In DarkSAGEMode the annuli then choose where that mass
 * leaves from: each annulus is tested against the Toomre Q of its gas and
 * stars as in DarkSage's check_disk_instability() (Stevens, Croton & Mutch
 * 2016 Sec. 3.7; Stevens et al. 2024 Sec. 8), combined with the Romeo &
 * Wiegert (2011) two-fluid weighting W = 2 sigma c_s / (sigma^2 + c_s^2), and
 * the mass each annulus would have to lose to become stable weights the
 * removal.  Gas and stellar discs more than DS_THETA_THRESH_DEG degrees apart
 * are tested as separate one-component discs.  With DarkSAGERadialFlowsOn,
 * gas still unstable then sinks to the neighbouring annuli as in DarkSage.
 *
 * Code units throughout (10^10 Msun/h, Mpc/h, km/s).
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "core_allvars.h"
#include "macros.h"

#include "model_darksage_instability.h"
#include "model_darksage_sf.h"

/* DarkSage cold-gas velocity dispersion (= sound speed here), c_s = 11 + 11.3 z km/s [cm/s] */
static const double DS_CS0_CGS = 1.1e6;
static const double DS_CS1_CGS = 1.13e6;
/* Stellar-disc Toomre factor 3.36 / pi inverted (Toomre 1964) */
static const double TOOMRE_STARS = 0.935;

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
 * darksage_unstable_profile -- the gas and stellar mass each annulus would have to lose to be
 * stable (DarkSage check_disk_instability(), without acting on it).
 *
 *   Q_gas  = c_s kappa (r_out^2 - r_in^2) / (G m_gas)
 *   Q_star = 0.935 kappa sigma (r_out^2 - r_in^2) / (G m_star)
 * Gas: with coplanar stars the two-fluid Q_tot = 1 / (W/Q_gas + 1/Q_star) if Q_gas >= Q_star,
 *   else 1 / (1/Q_gas + W/Q_star); a stable annulus has Q_tot >= DS_QTOT_MIN.  If both Q are
 *   below DS_QTOT_MIN + W the gas must reach that; if only the gas is, the Q_gas that brings
 *   Q_tot to DS_QTOT_MIN; otherwise the stars resolve it and the gas loses nothing.  Without
 *   coplanar stars Q_gas_min = DS_QTOT_MIN.  gas_w[i] = m_gas (1 - Q_gas / Q_gas_min).
 * Stars: Q_star_min likewise from the two-fluid Q (or DS_QTOT_MIN alone);
 *   star_w[i] = m_star (1 - Q_star / Q_star_min).
 * Annuli that are stable get zero.
 */
/* Gas and stellar mass annulus i must lose to be stable (one step of the loop below) */
static void annulus_unstable(const int i, const struct GALAXY *g, const double c_s, const int coplanar,
                             double *gas_w, double *star_w, const struct params *run_params)
{
    const double G = run_params->G;
    const double Q_min = DS_QTOT_MIN;
    *gas_w = *star_w = 0.0;
    const double kappa = epicyclic_frequency(i, g, run_params);
    if(kappa <= 0.0) return;
    const double area = g->DiscRadii[i+1] * g->DiscRadii[i+1] - g->DiscRadii[i] * g->DiscRadii[i];
    const double sigma = g->VelDispStars[i];
    const int has_gas = (g->DiscGas[i] > 0.0f), has_stars = (g->DiscStars[i] > 0.0f && sigma > 0.0);
    const double Q_gas = has_gas ? c_s * kappa * area / (G * g->DiscGas[i]) : 0.0;
    const double Q_star = has_stars ? kappa * sigma * TOOMRE_STARS * area / (G * g->DiscStars[i]) : 0.0;
    const int two_fluid = (has_gas && has_stars && coplanar);
    const double W = has_stars ? 2.0 * sigma * c_s / (sigma * sigma + c_s * c_s) : 0.0;

    if(has_gas) {
        double Q_gas_min = Q_min;
        int stable = 0;
        if(two_fluid) {
            const double Q_tot = (Q_gas >= Q_star) ? 1.0 / (W / Q_gas + 1.0 / Q_star) : 1.0 / (1.0 / Q_gas + W / Q_star);
            const double Q_stable = Q_min + W;
            if(Q_tot >= Q_min) stable = 1;
            else if(Q_gas < Q_stable && Q_star < Q_stable) Q_gas_min = Q_stable;
            else if(Q_gas < Q_stable) Q_gas_min = 1.0 / (1.0 / Q_min - W / Q_star);
            else stable = 1;   /* the stars' job to restore stability */
        }
        if(!stable && Q_gas < Q_gas_min) *gas_w = g->DiscGas[i] * (1.0 - Q_gas / Q_gas_min);
    }

    if(has_stars) {
        double Q_star_min = Q_min;
        int stable = 0;
        if(two_fluid) {
            double Q_tot;
            if(Q_gas >= Q_star) {
                Q_tot = 1.0 / (W / Q_gas + 1.0 / Q_star);
                Q_star_min = (1.0 / Q_min - W / Q_gas > 0.0) ? 1.0 / (1.0 / Q_min - W / Q_gas) : 1.0;
            } else {
                Q_tot = 1.0 / (1.0 / Q_gas + W / Q_star);
                Q_star_min = (1.0 / Q_min - 1.0 / Q_gas > 0.0) ? W / (1.0 / Q_min - 1.0 / Q_gas) : 1.0;
            }
            /* 0.99 absorbs rounding (DarkSage) */
            if(Q_tot >= 0.99 * Q_min || Q_star >= 0.99 * Q_star_min) stable = 1;
        }
        if(!stable && Q_star < Q_star_min) *star_w = g->DiscStars[i] * (1.0 - Q_star / Q_star_min);
    }
}

/* Whether the gas and stellar discs are close enough to be one two-fluid disc */
static int discs_coplanar(const struct GALAXY *g)
{
    double cos_angle = g->SpinStars[0] * g->SpinGas[0] + g->SpinStars[1] * g->SpinGas[1] + g->SpinStars[2] * g->SpinGas[2];
    if(cos_angle > 1.0) cos_angle = 1.0;
    if(cos_angle < -1.0) cos_angle = -1.0;
    return (acos(cos_angle) * 180.0 / M_PI <= DS_THETA_THRESH_DEG);
}

void darksage_unstable_profile(const int p, double gas_w[N_BINS], double star_w[N_BINS],
                               const struct GALAXY *galaxies, const struct params *run_params)
{
    const struct GALAXY *g = &galaxies[p];
    const double c_s = darksage_cold_gas_dispersion(run_params->ZZ[g->SnapNum], run_params);
    const int coplanar = discs_coplanar(g);
    for(int i = 0; i < N_BINS; i++) annulus_unstable(i, g, c_s, coplanar, &gas_w[i], &star_w[i], run_params);
}

/*
 * darksage_instability_gas_sink -- DarkSAGERadialFlowsOn: unstable gas sinks through the disc
 * (the gas_sink step of DarkSage's deal_with_unstable_gas(); Stevens, Croton & Mutch 2016
 * Sec. 3.7).  Call after SAGE26's instability has taken its mass.
 *
 * Algorithm, from the outermost annulus inwards as DarkSage (gas an annulus receives is tested
 * when its turn comes):
 *   1. unstable gas u_i = m_i (1 - Q_gas / Q_gas_min) (darksage_unstable_profile());
 *   2. DS_GAS_SINK_RATE u_i leaves annulus i: the outermost annulus gives it all to annulus i-1;
 *      the others send m_down to i-1 and m_up to i+1 with m_down j_lose = m_up j_gain, j_lose and
 *      j_gain the steps between the j-bin midpoints, so the disc keeps its angular momentum;
 *   3. metals, dust, H2 and HI go with the gas in the annulus's proportions;
 *   4. with DarkSAGERadialFlowsOn 2, unstable stars do the same: a fraction
 *      1 - (1 - DS_GAS_SINK_RATE) c_s / sigma of m_star (1 - Q_star / Q_star_min) (DarkSage's
 *      StarSinkRate, DiskInstabilityOn 1), with their metals and velocity dispersion.
 * Differences from DarkSage: SAGE26 has already decided the instability's star formation and
 * black-hole growth, so the rest of u_i stays where it is, and annulus 0 (whose m_down DarkSage
 * feeds to the black hole, and the stars' to the bulge) keeps its mass.  Only the annuli change:
 * ColdGas, MetalsColdGas, ColdDust and the disc's stellar mass and metals are untouched.
 */
void darksage_instability_gas_sink(const int p, struct GALAXY *galaxies, const struct params *run_params)
{
    struct GALAXY *g = &galaxies[p];
    const double *edge = run_params->DiscBinEdge;
    const double c_s = darksage_cold_gas_dispersion(run_params->ZZ[g->SnapNum], run_params);
    const int coplanar = discs_coplanar(g);

    for(int i = N_BINS - 1; i > 0; i--) {
        double unstable_gas, unstable_stars;
        annulus_unstable(i, g, c_s, coplanar, &unstable_gas, &unstable_stars, run_params);
        /* the j-conserving split: m_down j_lose = m_up j_gain (all inwards from the last annulus) */
        double f_up = 0.0;
        if(i < N_BINS - 1) {
            const double j_gain = 0.5 * (edge[i+2] - edge[i]);
            const double j_lose = 0.5 * (edge[i+1] - edge[i-1]);
            f_up = j_lose / (j_gain + j_lose);
        }

        const double sink = DS_GAS_SINK_RATE * unstable_gas;
        if(sink > 0.0 && g->DiscGas[i] > 0.0f) {
            /* gas, metals, dust, H2 and HI all move in the same proportion */
            float *arr[5] = {g->DiscGas, g->DiscGasMetals, g->DiscDust, g->DiscH2, g->DiscHI};
            const double f_down = (1.0 - f_up) * sink / g->DiscGas[i], f_upm = f_up * sink / g->DiscGas[i];
            for(int k = 0; k < 5; k++) {
                const double m = arr[k][i];
                arr[k][i] = (float)(m * (1.0 - f_down - f_upm));
                arr[k][i-1] += (float)(m * f_down);
                if(i < N_BINS - 1) arr[k][i+1] += (float)(m * f_upm);
            }
        }

        /* DarkSAGERadialFlowsOn 2: unstable stars migrate the same way, a fraction
         * 1 - (1 - GasSinkRate) c_s / sigma of them (DarkSage StarSinkRate), mixing dispersions by mass */
        const double sigma = g->VelDispStars[i];
        if(run_params->DarkSAGERadialFlowsOn < 2 || unstable_stars <= 0.0 || g->DiscStars[i] <= 0.0f || sigma <= 0.0) continue;
        double moved = (1.0 - (1.0 - DS_GAS_SINK_RATE) * c_s / sigma) * unstable_stars;
        if(moved > g->DiscStars[i]) moved = g->DiscStars[i];
        if(moved <= 0.0) continue;
        const double z = g->DiscStarsMetals[i] / g->DiscStars[i];
        const double to[2] = {(1.0 - f_up) * moved, f_up * moved};
        g->DiscStars[i] -= (float)moved;
        g->DiscStarsMetals[i] = (float)(z * g->DiscStars[i]);
        for(int k = 0; k < 2; k++) {
            const int j = (k == 0) ? i - 1 : i + 1;
            if(to[k] <= 0.0) continue;
            const double m_old = g->DiscStars[j], s_old = g->VelDispStars[j];
            g->VelDispStars[j] = (float)sqrt((m_old * s_old * s_old + to[k] * sigma * sigma) / (m_old + to[k]));
            g->DiscStars[j] += (float)to[k];
            g->DiscStarsMetals[j] += (float)(z * to[k]);
        }
    }
}

/*
 * darksage_instability_stars_to_bulge -- the annulus side of SAGE26 moving unstable_stars of
 * disc stars into the instability bulge.  Call after the bulk move.  The stars leave the
 * annuli in proportion to star_w (darksage_unstable_profile(); in proportion to the disc where
 * no annulus is unstable) and bring their velocity dispersion to the bulge's; the annulus
 * metals are then rescaled to the bulk disc metals.
 */
void darksage_instability_stars_to_bulge(const int p, const double unstable_stars, const double star_w[N_BINS],
                                         struct GALAXY *galaxies)
{
    struct GALAXY *g = &galaxies[p];
    double taken[N_BINS];
    float sigma[N_BINS];
    for(int i = 0; i < N_BINS; i++) sigma[i] = g->VelDispStars[i];
    darksage_remove_disc_stars(p, unstable_stars, star_w, taken, galaxies);

    double m_in = 0.0, msig2_in = 0.0;
    for(int i = 0; i < N_BINS; i++) {
        m_in += taken[i];
        msig2_in += taken[i] * sigma[i] * sigma[i];
    }
    const double m_old = g->InstabilityBulgeMass - unstable_stars;
    if(m_in > 0.0) {
        const double m = (m_old > 0.0) ? m_old : 0.0;
        g->VelDispBulge = (float)sqrt((m * g->VelDispBulge * g->VelDispBulge + msig2_in) / (m + m_in));
    }
    darksage_sync_disc_metals(p, galaxies);
}

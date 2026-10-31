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
 * are tested as separate one-component discs.
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
void darksage_unstable_profile(const int p, double gas_w[N_BINS], double star_w[N_BINS],
                               const struct GALAXY *galaxies, const struct params *run_params)
{
    const struct GALAXY *g = &galaxies[p];
    const double G = run_params->G;
    const double Q_min = DS_QTOT_MIN;
    const double c_s = darksage_cold_gas_dispersion(run_params->ZZ[g->SnapNum], run_params);

    double cos_angle = g->SpinStars[0] * g->SpinGas[0] + g->SpinStars[1] * g->SpinGas[1] + g->SpinStars[2] * g->SpinGas[2];
    if(cos_angle > 1.0) cos_angle = 1.0;
    if(cos_angle < -1.0) cos_angle = -1.0;
    const int coplanar = (acos(cos_angle) * 180.0 / M_PI <= DS_THETA_THRESH_DEG);

    for(int i = 0; i < N_BINS; i++) {
        gas_w[i] = star_w[i] = 0.0;
        const double kappa = epicyclic_frequency(i, g, run_params);
        if(kappa <= 0.0) continue;
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
            if(!stable && Q_gas < Q_gas_min) gas_w[i] = g->DiscGas[i] * (1.0 - Q_gas / Q_gas_min);
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
            if(!stable && Q_star < Q_star_min) star_w[i] = g->DiscStars[i] * (1.0 - Q_star / Q_star_min);
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

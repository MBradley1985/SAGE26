/*
 * model_darksage_instability.c -- annulus-resolved disc instability (DarkSAGEMode == 1).
 *
 * Replaces check_disk_instability() when DarkSAGEMode == 1.  Each annulus is
 * tested against the two-fluid (gas + stars) Toomre Q (Romeo & Wiegert 2011)
 * using the j-bin epicyclic frequency of Stevens, Croton & Mutch (2016).
 * Unstable mass in annulus i > 0 is split between annuli i-1 and i+1 so that
 * angular momentum is conserved; unstable mass in annulus 0 feeds the bulge
 * (stars) and the black hole plus a central burst (gas), which goes through
 * the normal SF + SN feedback bookkeeping (darksage_central_burst()).
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "core_allvars.h"

#include "model_darksage_instability.h"
#include "model_starformation_and_feedback.h"

static const double DS_SOUND_SPEED   = 10.0;   /* km/s, cold ISM */
static const double DS_SIGMA_FLOOR   = 10.0;   /* km/s, stellar dispersion floor */
static const double DS_Q_STABLE      = 1000.0;

static int last_physical_bin(const int p, const struct GALAXY *galaxies)
{
    for(int i = N_BINS - 1; i >= 0; i--) {
        if(galaxies[p].DiscRadii[i+1] > galaxies[p].DiscRadii[i]) return i;
    }
    return 0;
}

/* Epicyclic frequency from the j-bin edges: kappa^2 = 2 j / r^3 dj/dr */
static double epicyclic_frequency(const int bin, const double r_inner, const double r_outer,
                                  const struct params *run_params)
{
    const double dj = run_params->DiscBinEdge[bin+1] - run_params->DiscBinEdge[bin];
    const double dr = r_outer - r_inner;
    if(dr <= 0.0 || dj <= 0.0) return 0.0;

    const double j = (bin > 0) ? run_params->DiscBinEdge[bin] : run_params->DiscBinEdge[bin+1];
    const double r = (bin > 0) ? r_inner : r_outer;
    if(r <= 0.0) return 0.0;
    return sqrt(2.0 * j / (r * r * r) * dj / dr);
}

/* Two-fluid Toomre Q of one annulus, in code units (masses, not surface densities) */
static double combined_toomre_Q(const int bin, const double gas, const double stars,
                                const double sigma_stars, const double r_inner, const double r_outer,
                                const struct params *run_params)
{
    if(r_outer <= 0.0) return DS_Q_STABLE;

    const double kappa = epicyclic_frequency(bin, r_inner, r_outer, run_params);
    const double area_factor = r_outer * r_outer - r_inner * r_inner;
    if(kappa <= 0.0 || area_factor <= 0.0) return DS_Q_STABLE;

    const double G = run_params->G;
    const double c_s = DS_SOUND_SPEED;
    const double Q_gas = (gas > 0.0) ? c_s * kappa * area_factor / (G * gas) : DS_Q_STABLE;
    /* 0.935 = pi / 3.36 for a stellar disc */
    const double Q_stars = (stars > 0.0) ? kappa * sigma_stars * 0.935 * area_factor / (G * stars) : DS_Q_STABLE;

    if(gas <= 0.0) return Q_stars;
    if(stars <= 0.0) return Q_gas;

    const double W = 2.0 * sigma_stars * c_s / (sigma_stars * sigma_stars + c_s * c_s);
    const double denom = (sigma_stars >= c_s) ? W / Q_gas + 1.0 / Q_stars : 1.0 / Q_gas + W / Q_stars;
    return (denom > 0.0) ? 1.0 / denom : DS_Q_STABLE;
}

/* j-conservation split of mass leaving annulus `bin`: m_up * j_gain = m_down * j_lose */
static void j_split(const int bin, const double mass, const struct params *run_params,
                    double *m_down, double *m_up)
{
    const double *edge = run_params->DiscBinEdge;
    const double j_lose = 0.5 * (edge[bin+1] - edge[(bin > 0) ? bin-1 : 0]);
    const double j_gain = (bin < N_BINS - 1) ? 0.5 * (edge[bin+2] - edge[bin]) : j_lose;
    const double j_total = j_lose + j_gain;
    if(j_lose <= 0.0 || j_gain <= 0.0 || j_total <= 0.0) {
        *m_down = *m_up = 0.0;
        return;
    }
    *m_down = mass * j_gain / j_total;
    *m_up   = mass * j_lose / j_total;
}

static void move_unstable_gas(const int p, const int bin, const double unstable,
                              struct GALAXY *galaxies, const struct params *run_params)
{
    if(unstable <= 0.0 || bin <= 0 || galaxies[p].DiscGas[bin] <= 0.0) return;

    double m_down, m_up;
    j_split(bin, unstable, run_params, &m_down, &m_up);
    if(m_down + m_up <= 0.0) return;

    const double Z = galaxies[p].DiscGasMetals[bin] / galaxies[p].DiscGas[bin];
    int up = (bin < N_BINS - 1) ? bin + 1 : N_BINS - 1;
    const int last = last_physical_bin(p, galaxies);
    if(up > last) up = last;

    galaxies[p].DiscGas[bin] -= m_down + m_up;
    galaxies[p].DiscGasMetals[bin] -= Z * (m_down + m_up);
    galaxies[p].DiscGas[bin-1] += m_down;
    galaxies[p].DiscGasMetals[bin-1] += Z * m_down;
    galaxies[p].DiscGas[up] += m_up;
    galaxies[p].DiscGasMetals[up] += Z * m_up;

    if(galaxies[p].DiscGas[bin] < 0.0f) galaxies[p].DiscGas[bin] = 0.0f;
    if(galaxies[p].DiscGasMetals[bin] < 0.0f) galaxies[p].DiscGasMetals[bin] = 0.0f;
}

static void move_unstable_stars(const int p, const int bin, const double unstable,
                                struct GALAXY *galaxies, const struct params *run_params)
{
    if(unstable <= 0.0 || bin <= 0 || galaxies[p].DiscStars[bin] <= 0.0) return;

    double m_down, m_up;
    j_split(bin, unstable, run_params, &m_down, &m_up);
    if(m_down + m_up <= 0.0) return;

    const double Z = galaxies[p].DiscStarsMetals[bin] / galaxies[p].DiscStars[bin];
    int up = (bin < N_BINS - 1) ? bin + 1 : N_BINS - 1;
    const int last = last_physical_bin(p, galaxies);
    if(up > last) up = last;

    /* Mass-weighted dispersion mixing into the inner annulus */
    const double m_dst = galaxies[p].DiscStars[bin-1];
    const double s_src = galaxies[p].VelDispStars[bin];
    const double s_dst = galaxies[p].VelDispStars[bin-1];
    if(m_dst + m_down > 0.0) {
        galaxies[p].VelDispStars[bin-1] = (float)sqrt((m_dst * s_dst * s_dst + m_down * s_src * s_src) / (m_dst + m_down));
    }

    galaxies[p].DiscStars[bin] -= m_down + m_up;
    galaxies[p].DiscStarsMetals[bin] -= Z * (m_down + m_up);
    galaxies[p].DiscStars[bin-1] += m_down;
    galaxies[p].DiscStarsMetals[bin-1] += Z * m_down;
    galaxies[p].DiscStars[up] += m_up;
    galaxies[p].DiscStarsMetals[up] += Z * m_up;

    if(galaxies[p].DiscStars[bin] < 0.0f) galaxies[p].DiscStars[bin] = 0.0f;
    if(galaxies[p].DiscStarsMetals[bin] < 0.0f) galaxies[p].DiscStarsMetals[bin] = 0.0f;
}

void check_disk_instability_darksage(const int p, const int centralgal, const double dt, const int step,
                                     struct GALAXY *galaxies, const struct params *run_params)
{
    if(galaxies[p].Vvir <= 0.0 || galaxies[p].DiskScaleRadius <= 0.0) return;

    const double diskmass = galaxies[p].ColdGas + (galaxies[p].StellarMass - galaxies[p].BulgeMass);
    if(diskmass <= 0.0) return;

    /* Soft global-stability suppression (floor 0.15) using the Mo, Mao & White critical mass */
    const double Mcrit = galaxies[p].Vmax * galaxies[p].Vmax * (3.0 * galaxies[p].DiskScaleRadius) / run_params->G;
    const double f_floor = 0.15;
    double f_global = 1.0;
    if(Mcrit > 0.0 && diskmass < 2.0 * Mcrit) {
        f_global = f_floor + (1.0 - f_floor) * diskmass / (2.0 * Mcrit);
        if(f_global > 1.0) f_global = 1.0;
    }

    const double Q_min = run_params->QTotMin;
    const double gas_sink_rate = run_params->GasSinkRate;
    const double c_s = DS_SOUND_SPEED;

    /* --- annuli N_BINS-1 .. 1: migrate unstable mass with j-conservation, heat the stars --- */
    for(int i = N_BINS - 1; i >= 0; i--) {
        if(galaxies[p].DiscGas[i] <= 0.0 && galaxies[p].DiscStars[i] <= 0.0) continue;

        const double r_inner = (i > 0) ? galaxies[p].DiscRadii[i] : 0.0;
        const double r_outer = galaxies[p].DiscRadii[i+1];

        double sigma_stars = galaxies[p].VelDispStars[i];
        if(sigma_stars < DS_SIGMA_FLOOR) sigma_stars = DS_SIGMA_FLOOR;

        const double Q = combined_toomre_Q(i, galaxies[p].DiscGas[i], galaxies[p].DiscStars[i],
                                           sigma_stars, r_inner, r_outer, run_params);
        if(Q >= Q_min) continue;

        /* Stellar sink rate grows with the stellar dispersion (Stevens+16) */
        double star_sink_rate = 1.0 - (1.0 - gas_sink_rate) * c_s / sigma_stars;
        if(star_sink_rate < 0.0) star_sink_rate = 0.0;
        if(star_sink_rate > 1.0) star_sink_rate = 1.0;

        double Q_deficit = (Q_min - Q) / Q_min;
        if(Q_deficit > 1.0) Q_deficit = 1.0;

        const double gas_excess   = f_global * gas_sink_rate  * galaxies[p].DiscGas[i]   * Q_deficit;
        const double stars_excess = f_global * star_sink_rate * galaxies[p].DiscStars[i] * Q_deficit;

        if(i > 0) {
            move_unstable_stars(p, i, stars_excess, galaxies, run_params);
            move_unstable_gas(p, i, gas_excess, galaxies, run_params);

            /* The rest of the instability is resolved by heating:
             * sigma_new = sigma_old [(1 - StarSinkRate) Q_min/Q + StarSinkRate] */
            if(galaxies[p].DiscStars[i] > 0.0) {
                double heating = (1.0 - star_sink_rate) * (Q_min / Q) + star_sink_rate;
                if(heating > 2.0) heating = 2.0;
                if(heating > 1.0) galaxies[p].VelDispStars[i] *= (float)heating;
            }
            continue;
        }

        /* --- annulus 0: m_down feeds the bulge (stars) or BH + central burst (gas), m_up goes to annulus 1 --- */
        double m_down, m_up;

        if(stars_excess > 0.0 && galaxies[p].DiscStars[0] > 0.0) {
            j_split(0, stars_excess, run_params, &m_down, &m_up);
            const double Z = galaxies[p].DiscStarsMetals[0] / galaxies[p].DiscStars[0];

            galaxies[p].DiscStars[0] -= m_down + m_up;
            galaxies[p].DiscStarsMetals[0] -= Z * (m_down + m_up);
            galaxies[p].DiscStars[1] += m_up;
            galaxies[p].DiscStarsMetals[1] += Z * m_up;

            /* Disc -> bulge: StellarMass unchanged */
            const double m_old = galaxies[p].SecularBulgeMass;
            galaxies[p].SecularBulgeMass += m_down;
            galaxies[p].SecularMetalsBulgeMass += Z * m_down;
            galaxies[p].BulgeMass += m_down;
            galaxies[p].InstabilityBulgeMass += m_down;
            galaxies[p].MetalsBulgeMass += Z * m_down;

            double sigma_new = galaxies[p].Vvir / 3.0;
            if(sigma_new < 50.0) sigma_new = 50.0;
            const double sigma_old = galaxies[p].VelDispBulge;
            galaxies[p].VelDispBulge = (sigma_old > 0.0)
                ? (float)sqrt((m_old * sigma_old * sigma_old + m_down * sigma_new * sigma_new) / (m_old + m_down))
                : (float)sigma_new;
        }

        if(gas_excess > 0.0 && galaxies[p].DiscGas[0] > 0.0) {
            j_split(0, gas_excess, run_params, &m_down, &m_up);
            const double Z = galaxies[p].DiscGasMetals[0] / galaxies[p].DiscGas[0];

            galaxies[p].DiscGas[0] -= m_up;
            galaxies[p].DiscGasMetals[0] -= Z * m_up;
            galaxies[p].DiscGas[1] += m_up;
            galaxies[p].DiscGasMetals[1] += Z * m_up;

            /* BH accretion with the Kauffmann & Haehnelt (2000) Vvir suppression */
            double BH_accrete = 0.0;
            if(run_params->AGNrecipeOn > 0) {
                const double Vvir = galaxies[p].Vvir;
                BH_accrete = run_params->BlackHoleGrowthRate * m_down / (1.0 + (280.0 * 280.0) / (Vvir * Vvir));
                if(BH_accrete > m_down) BH_accrete = m_down;
                galaxies[p].BlackHoleMass += BH_accrete;
                galaxies[p].DiscGas[0] -= BH_accrete;
                galaxies[p].DiscGasMetals[0] -= Z * BH_accrete;
                galaxies[p].ColdGas -= BH_accrete;
                galaxies[p].MetalsColdGas -= Z * BH_accrete;
                if(galaxies[p].ColdGas < 0.0f) galaxies[p].ColdGas = 0.0f;
                if(galaxies[p].MetalsColdGas < 0.0f) galaxies[p].MetalsColdGas = 0.0f;
            }

            /* The remainder (still in annulus 0) bursts: counted in SfrDisk / SFH, with SN feedback and yields */
            darksage_central_burst(p, centralgal, m_down - BH_accrete, dt, step, galaxies, run_params);
        }

        if(galaxies[p].DiscStars[0] < 0.0f) galaxies[p].DiscStars[0] = 0.0f;
        if(galaxies[p].DiscStarsMetals[0] < 0.0f) galaxies[p].DiscStarsMetals[0] = 0.0f;
        if(galaxies[p].DiscGas[0] < 0.0f) galaxies[p].DiscGas[0] = 0.0f;
        if(galaxies[p].DiscGasMetals[0] < 0.0f) galaxies[p].DiscGasMetals[0] = 0.0f;
        if(galaxies[p].ColdGas < 0.0f) galaxies[p].ColdGas = 0.0f;
        if(galaxies[p].MetalsColdGas < 0.0f) galaxies[p].MetalsColdGas = 0.0f;
    }
}

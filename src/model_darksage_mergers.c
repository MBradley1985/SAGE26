/*
 * model_darksage_mergers.c -- where merging gas and stars go (DarkSAGEMode == 1).
 *
 * Follows Stevens et al. (2024) Secs. 12.2-12.3 and DarkSage's
 * add_galaxies_together() and stars_to_bulge() (model_mergers.c):
 *
 * Minor mergers: the satellite's cold gas is spread evenly over the central's
 *   annuli whose j lies within j_orb . n_cold +- j_spread, with j_orb the
 *   satellite's orbital specific angular momentum and n_cold the central's
 *   gas-disc axis (eq. 123).  If the orbit is retrograde the central's gas
 *   disc is then shrunk so its angular momentum is what conservation requires
 *   (eq. 124).
 * Major mergers: the new gas disc lies in the last orbital plane.  Each
 *   annulus of both progenitors is cut into 10 segments whose j about the
 *   centre of momentum, projected on the new axis, picks the annulus they join;
 *   retrograde segments shrink the disc in the same way.
 * Both: the merger-driven bulge's velocity dispersion and specific angular
 *   momentum are updated from the progenitors' components and orbital motion,
 *   for its Jeans size (BulgeSizeOn == 4, eq. 128).
 *
 * Two places where this follows the paper rather than the DarkSage source:
 *   - the retrograde shrink.  deal_with_galaxy_merger() sums the disc angular
 *     momentum with '=' instead of '+=' (only the innermost annulus counts) and
 *     shrinks the satellite's disc, which is discarded, so it has no effect;
 *     here the central's disc is shrunk using the full sums (eq. 124).
 *   - the projected j of a major-merger segment: the source takes
 *     sqrt(sum_k (n_k j_k)^2); here it is the projection |j . n|.
 * The stellar masses themselves are moved by add_galaxies_together() and
 * make_bulge_from_burst() with SAGE26's bulge bookkeeping.
 *
 * Code units throughout (10^10 Msun/h, Mpc/h, km/s).
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "core_allvars.h"

#include "model_darksage_mergers.h"
#include "model_darksage_disc.h"
#include "model_darksage_feedback.h"
#include "model_misc.h"

/* Segments each annulus is cut into when rebuilding a major-merger gas disc (DarkSage N_ann_segs) */
#define N_ANNULUS_SEGMENTS 10
/* Skip the retrograde shrink when it would change J by less than this (DarkSage's 1% safety net) */
static const double RETRO_SHRINK_MIN = 0.99;

static double dot3(const double a[3], const double b[3])
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static void cross3(const double a[3], const double b[3], double c[3])
{
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];
}

/* Annulus i of the shared j grid: the bin j falls in */
static int j_bin(const double j, const struct params *run_params)
{
    int k = 1;
    while(k < N_BINS && j >= run_params->DiscBinEdge[k]) k++;
    return k - 1;
}

/* Angular momentum of a set of annuli, sum of j_mid m */
static double disc_ang_mom(const double *mass, const struct params *run_params)
{
    double J = 0.0;
    for(int i = 0; i < N_BINS; i++) J += 0.5 * (run_params->DiscBinEdge[i] + run_params->DiscBinEdge[i+1]) * mass[i];
    return J;
}

/*
 * Shrink galaxy t's gas disc by rescaling every annulus's j by J_should / J_current
 * (darksage_project_disc()), so the disc carries the angular momentum it should
 * after retrograde gas was added at |j| (eq. 124).
 */
static void shrink_gas_disc(const int t, const double J_current, const double J_should,
                            struct GALAXY *galaxies, const struct params *run_params)
{
    if(J_current <= 0.0 || J_should >= RETRO_SHRINK_MIN * J_current) return;
    double factor = J_should / J_current;
    if(factor < 1.0e-3) factor = 1.0e-3;   /* net counter-rotation: collapse to the centre */

    double gas[N_BINS], metals[N_BINS];
    darksage_project_disc(galaxies[t].DiscGas, factor, gas, run_params, galaxies[t].Rvir, galaxies[t].Vvir);
    darksage_project_disc(galaxies[t].DiscGasMetals, factor, metals, run_params, galaxies[t].Rvir, galaxies[t].Vvir);
    for(int i = 0; i < N_BINS; i++) {
        galaxies[t].DiscGas[i] = (float)gas[i];
        galaxies[t].DiscGasMetals[i] = (float)metals[i];
    }
}

/* Unit vector perpendicular to the unit vector s */
static void perpendicular(const double s[3], double e[3])
{
    const double ref[3] = {(fabs(s[0]) < 0.9) ? 1.0 : 0.0, (fabs(s[0]) < 0.9) ? 0.0 : 1.0, 0.0};
    cross3(s, ref, e);
    const double n = sqrt(dot3(e, e));
    for(int k = 0; k < 3; k++) e[k] /= n;
}

/*
 * Minor merger: deposit the satellite's gas by orbital j (eq. 123), then shrink
 * the central's gas disc if the orbit was retrograde (eq. 124).  Returns the
 * satellite's orbital radial velocity for the bulge dispersion.
 */
static double merge_gas_minor(const int t, const int p, const double dpos[3], const double dvel[3],
                              struct GALAXY *galaxies, const struct params *run_params)
{
    const double *edge = run_params->DiscBinEdge;
    double j_orb[3];
    cross3(dpos, dvel, j_orb);
    const double sep = sqrt(dot3(dpos, dpos));
    const double v_rad = (sep > 0.0) ? dot3(dvel, dpos) / sep : 0.0;

    double gas_sat = 0.0, metals_sat = 0.0;
    for(int i = 0; i < N_BINS; i++) {
        gas_sat += galaxies[p].DiscGas[i];
        metals_sat += galaxies[p].DiscGasMetals[i];
    }
    if(gas_sat <= 0.0) return v_rad;

    /* Annuli k..l of the central the satellite's gas spreads over */
    const double n_cold[3] = {galaxies[t].SpinGas[0], galaxies[t].SpinGas[1], galaxies[t].SpinGas[2]};
    const double j_orb_mag = sqrt(dot3(j_orb, j_orb));
    double cos_orb = 0.0;
    int i_min = 0, i_max = 1;
    if(j_orb_mag > 0.0) {
        cos_orb = dot3(n_cold, j_orb) / j_orb_mag;
        const double j_proj = j_orb_mag * fabs(cos_orb);
        const double j_max = j_proj + galaxies[p].Vvir * fabs(cos_orb) * sep;
        double j_min = 2.0 * j_proj - j_max;
        if(j_min < 0.0) j_min = 0.0;

        i_min = 0;
        while(i_min < N_BINS && edge[i_min] <= j_min) i_min++;
        i_min -= 1;
        if(i_min < 0) i_min = 0;
        i_max = i_min;
        while(i_max < N_BINS && edge[i_max] <= j_max) i_max++;
        if(i_max <= i_min) i_max = i_min + 1;
    }
    const int n_bins = i_max - i_min;

    /* Deposit, and the angular momentum the disc should end with: retrograde gas adds -j, not +j */
    double J_retro = 0.0;
    for(int i = i_min; i < i_max; i++) {
        galaxies[t].DiscGas[i] += (float)(gas_sat / n_bins);
        galaxies[t].DiscGasMetals[i] += (float)(metals_sat / n_bins);
        if(cos_orb < 0.0) J_retro += 0.5 * (edge[i] + edge[i+1]) * gas_sat / n_bins;
    }
    double gas[N_BINS];
    for(int i = 0; i < N_BINS; i++) gas[i] = galaxies[t].DiscGas[i];
    const double J_current = disc_ang_mom(gas, run_params);
    shrink_gas_disc(t, J_current, J_current - 2.0 * J_retro, galaxies, run_params);
    return v_rad;
}

/*
 * Major merger: rebuild the gas disc from both progenitors in the centre-of-
 * momentum frame, in the last orbital plane.  Sets SpinGas and returns the two
 * progenitors' orbital specific angular momenta and radial velocities.
 */
static void merge_gas_major(const int t, const int p, const double dpos[3], const double dvel[3],
                            double j_t[3], double j_p[3], double *vr_t, double *vr_p,
                            struct GALAXY *galaxies, const struct params *run_params)
{
    /* Centre of momentum, weighting by the haloes' masses (baryons if a halo is gone) */
    double m_t = galaxies[t].Mvir, m_p = galaxies[p].Mvir;
    if(m_t <= 0.0) m_t = galaxies[t].StellarMass + galaxies[t].ColdGas;
    if(m_p <= 0.0) m_p = galaxies[p].StellarMass + galaxies[p].ColdGas;
    const double f_t = (m_t + m_p > 0.0) ? m_t / (m_t + m_p) : 0.5;
    double r_p[3], r_t[3], v_p[3], v_t[3];
    for(int k = 0; k < 3; k++) {
        r_p[k] = dpos[k] * f_t;
        r_t[k] = -dpos[k] * (1.0 - f_t);
        v_p[k] = dvel[k] * f_t;
        v_t[k] = -dvel[k] * (1.0 - f_t);
    }
    cross3(r_p, v_p, j_p);
    cross3(r_t, v_t, j_t);
    const double rp = sqrt(dot3(r_p, r_p)), rt = sqrt(dot3(r_t, r_t));
    *vr_p = (rp > 0.0) ? dot3(r_p, v_p) / rp : 0.0;
    *vr_t = (rt > 0.0) ? dot3(r_t, v_t) / rt : 0.0;

    double gas_t = 0.0, gas_p = 0.0;
    for(int i = 0; i < N_BINS; i++) {
        gas_t += galaxies[t].DiscGas[i];
        gas_p += galaxies[p].DiscGas[i];
    }
    if(gas_t + gas_p <= 0.0) return;

    double n_new[3];
    for(int k = 0; k < 3; k++) n_new[k] = j_t[k] * gas_t + j_p[k] * gas_p;
    const double n_mag = sqrt(dot3(n_new, n_new));
    if(n_mag <= 0.0) {
        for(int k = 0; k < 3; k++) n_new[k] = galaxies[t].SpinGas[k];
    } else {
        for(int k = 0; k < 3; k++) n_new[k] /= n_mag;
    }

    double new_gas[N_BINS] = {0.0}, new_metals[N_BINS] = {0.0}, J_retro = 0.0;
    const int gals[2] = {p, t};
    for(int g = 0; g < 2; g++) {
        const int gal = gals[g];
        const double *r_off = (gal == p) ? r_p : r_t;
        const double *v_off = (gal == p) ? v_p : v_t;
        const double s[3] = {galaxies[gal].SpinGas[0], galaxies[gal].SpinGas[1], galaxies[gal].SpinGas[2]};
        if(dot3(s, s) <= 0.0) continue;
        double e1[3], e2[3];
        perpendicular(s, e1);
        cross3(s, e1, e2);   /* e1 x e2 = s: the annulus rotates about s */

        for(int i = 0; i < N_BINS; i++) {
            if(galaxies[gal].DiscGas[i] <= 0.0f) continue;
            const double r_ann = sqrt(0.5 * (galaxies[gal].DiscRadii[i] * galaxies[gal].DiscRadii[i]
                                             + galaxies[gal].DiscRadii[i+1] * galaxies[gal].DiscRadii[i+1]));
            if(r_ann <= 0.0) continue;
            const double v_ann = 0.5 * (run_params->DiscBinEdge[i] + run_params->DiscBinEdge[i+1]) / r_ann;
            const double dm = galaxies[gal].DiscGas[i] / N_ANNULUS_SEGMENTS;
            const double dz = galaxies[gal].DiscGasMetals[i] / N_ANNULUS_SEGMENTS;
            for(int seg = 0; seg < N_ANNULUS_SEGMENTS; seg++) {
                const double phi = 2.0 * M_PI * seg / N_ANNULUS_SEGMENTS;
                double r[3], v[3], j[3];
                for(int k = 0; k < 3; k++) {
                    r[k] = r_ann * (cos(phi) * e1[k] + sin(phi) * e2[k]) + r_off[k];
                    v[k] = v_ann * (-sin(phi) * e1[k] + cos(phi) * e2[k]) + v_off[k];
                }
                cross3(r, v, j);
                const double j_par = dot3(j, n_new);
                const int bin = j_bin(fabs(j_par), run_params);
                new_gas[bin] += dm;
                new_metals[bin] += dz;
                if(j_par < 0.0) J_retro += 0.5 * (run_params->DiscBinEdge[bin] + run_params->DiscBinEdge[bin+1]) * dm;
            }
        }
    }

    for(int i = 0; i < N_BINS; i++) {
        galaxies[t].DiscGas[i] = (float)new_gas[i];
        galaxies[t].DiscGasMetals[i] = (float)new_metals[i];
        galaxies[p].DiscGas[i] = galaxies[p].DiscGasMetals[i] = 0.0f;
    }
    for(int k = 0; k < 3; k++) galaxies[t].SpinGas[k] = (float)n_new[k];
    const double J_current = disc_ang_mom(new_gas, run_params);
    shrink_gas_disc(t, J_current, J_current - 2.0 * J_retro, galaxies, run_params);
}

void darksage_merge_galaxies(const int t, const int p, struct GALAXY *galaxies, const struct params *run_params)
{
    struct GALAXY *gt = &galaxies[t], *gp = &galaxies[p];

    /* Mass ratio as in deal_with_galaxy_merger() */
    const double m1 = gp->StellarMass + gp->ColdGas, m2 = gt->StellarMass + gt->ColdGas;
    const double mi = (m1 < m2) ? m1 : m2, ma = (m1 < m2) ? m2 : m1;
    const double mass_ratio = (ma > 0.0) ? mi / ma : 1.0;
    const int major = (mass_ratio > run_params->ThreshMajorMerger);

    /* Satellite position and velocity relative to the central: physical, periodic box */
    const double a = run_params->AA[gt->SnapNum];
    double dpos[3], dvel[3];
    for(int k = 0; k < 3; k++) {
        dpos[k] = gp->Pos[k] - gt->Pos[k];
        if(dpos[k] > 0.5 * run_params->BoxSize) dpos[k] -= run_params->BoxSize;
        if(dpos[k] < -0.5 * run_params->BoxSize) dpos[k] += run_params->BoxSize;
        dpos[k] *= a;
        dvel[k] = gp->Vel[k] - gt->Vel[k];
    }

    /* Step 1: gas placement, and the orbital terms of the merger-bulge kinematics */
    double msig2 = 0.0, mj[3] = {0.0, 0.0, 0.0}, m_bulge;
    if(!major) {
        const double v_rad = merge_gas_minor(t, p, dpos, dvel, galaxies, run_params);
        for(int i = 0; i < N_BINS; i++) gp->DiscGas[i] = gp->DiscGasMetals[i] = 0.0f;

        /* Merger bulge j: the central's, the satellite's merger bulge and its stellar disc's J */
        double stars_p[N_BINS];
        for(int i = 0; i < N_BINS; i++) stars_p[i] = gp->DiscStars[i];
        const double J_sat_disc = disc_ang_mom(stars_p, run_params);
        for(int k = 0; k < 3; k++) {
            mj[k] = gt->MergerBulgeMass * gt->SpinMergerBulge[k] + gp->MergerBulgeMass * gp->SpinMergerBulge[k]
                    + J_sat_disc * gp->SpinStars[k];
        }
        m_bulge = gt->MergerBulgeMass + gp->StellarMass;
        msig2 += gp->StellarMass * v_rad * v_rad;
    } else {
        double j_t[3] = {0.0}, j_p[3] = {0.0}, vr_t = 0.0, vr_p = 0.0;
        merge_gas_major(t, p, dpos, dvel, j_t, j_p, &vr_t, &vr_p, galaxies, run_params);
        const double m_stars = gp->StellarMass + gt->StellarMass;
        for(int k = 0; k < 3; k++) mj[k] = (m_stars > 0.0) ? (j_p[k] * gp->StellarMass + j_t[k] * gt->StellarMass) : 0.0;
        m_bulge = (m_stars > 0.0) ? m_stars : 1.0;
        msig2 += gp->StellarMass * vr_p * vr_p + gt->StellarMass * vr_t * vr_t;
    }

    /* Step 2: merger-bulge dispersion from the components it absorbs (DarkSage add_galaxies_together()) */
    msig2 += gt->MergerBulgeMass * gt->VelDispMergerBulge * gt->VelDispMergerBulge;
    msig2 += gp->MergerBulgeMass * gp->VelDispMergerBulge * gp->VelDispMergerBulge;
    msig2 += gp->SecularBulgeMass * gp->VelDispBulge * gp->VelDispBulge;
    for(int i = 0; i < N_BINS; i++) msig2 += gp->DiscStars[i] * gp->VelDispStars[i] * gp->VelDispStars[i];
    const double m_new = gt->MergerBulgeMass + gp->StellarMass;
    if(m_new > 0.0) gt->VelDispMergerBulge = (float)sqrt(msig2 / m_new);
    for(int k = 0; k < 3; k++) gt->SpinMergerBulge[k] = (float)(mj[k] / m_bulge);

    /* Step 3: the satellite's fountain and outflowing gas -- the hot phase after a minor merger,
     * the fountain after a major one, whose turbulence delays its return (Sec. 12.3) */
    if(major) {
        darksage_add_to_fountain(t, gp->FountainGas + gp->OutflowGas, gp->MetalsFountainGas + gp->MetalsOutflowGas,
                                 galaxies, run_params);
    } else {
        add_gas_to_hot_reservoir(gt, run_params, gp->FountainGas + gp->OutflowGas,
                                 gp->MetalsFountainGas + gp->MetalsOutflowGas);
    }
    gp->FountainGas = gp->MetalsFountainGas = 0.0f;
    gp->OutflowGas = gp->MetalsOutflowGas = 0.0f;
}

void darksage_stars_to_bulge(const int t, struct GALAXY *galaxies)
{
    struct GALAXY *g = &galaxies[t];
    /* Dispersion of the new all-stellar merger bulge: quadrature sum of its parts (DarkSage stars_to_bulge()) */
    double msig2 = g->MergerBulgeMass * g->VelDispMergerBulge * g->VelDispMergerBulge
                   + g->SecularBulgeMass * g->VelDispBulge * g->VelDispBulge;
    for(int i = 0; i < N_BINS; i++) msig2 += g->DiscStars[i] * g->VelDispStars[i] * g->VelDispStars[i];
    if(g->StellarMass > 0.0) g->VelDispMergerBulge = (float)sqrt(msig2 / g->StellarMass);

    g->SecularBulgeMass = g->SecularMetalsBulgeMass = 0.0f;
    for(int k = 0; k < 3; k++) g->SpinSecularBulge[k] = 0.0f;
}

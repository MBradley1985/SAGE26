/*
 * model_darksage_mergers.c -- where merging gas goes (DarkSAGEMode >= 1).
 *
 * Follows Stevens et al. (2024) Secs. 12.2-12.3 and DarkSage's
 * add_galaxies_together() (model_mergers.c):
 *
 * Minor mergers: the satellite's cold gas is spread evenly over the central's
 *   annuli whose j lies within j_orb . n_cold +- j_spread, with j_orb the
 *   satellite's orbital specific angular momentum and n_cold the central's
 *   gas-disc axis (eq. 123).
 * Major mergers: the new gas disc lies in the last orbital plane.  Each
 *   annulus of both progenitors is cut into 10 segments whose j about the
 *   centre of momentum, projected on the new axis, picks the annulus they join.
 *
 * As in the DarkSage source, retrograde gas is not shrunk afterwards (its
 * deal_with_galaxy_merger() shrinks the satellite's disc, which has already
 * been merged), and a major-merger segment's j is sqrt(sum_k (n_k j_k)^2).
 * Major or minor follows SAGE26's mass ratio and ThreshMajorMerger.  The
 * satellite's stars join the bulge SAGE26's add_galaxies_together() puts them in
 * (model_mergers.c); the bulges' spin and velocity dispersions are set here,
 * following DarkSage where all of them join the merger bulge.  The bursts and
 * black holes are SAGE26's.
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
#include "model_misc.h"

/* Segments each annulus is cut into when rebuilding a major-merger gas disc (DarkSage N_ann_segs) */
#define N_ANNULUS_SEGMENTS 10

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

/* Unit vector in galaxy s's disc plane where its segments start (DarkSage add_galaxies_together()) */
static void perpendicular(const double s[3], double e[3])
{
    const double n = sqrt(s[1] * s[1] + s[2] * s[2]);
    if(n > 0.0) {
        e[0] = 0.0;
        e[1] = s[2] / n;
        e[2] = -s[1] / n;
    } else {
        e[0] = 0.0;
        e[1] = 1.0;
        e[2] = 0.0;
    }
}

/*
 * Minor merger: deposit the satellite's gas by orbital j (eq. 123).  Returns the
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

    for(int i = i_min; i < i_max; i++) {
        galaxies[t].DiscGas[i] += (float)(gas_sat / n_bins);
        galaxies[t].DiscGasMetals[i] += (float)(metals_sat / n_bins);
    }
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
    /* Centre of momentum, weighting by the host's mass inside the satellite's radius and the
     * satellite's mass (DarkSage get_Mhost_internal(), get_satellite_mass()) */
    const double m_p = darksage_satellite_mass(&galaxies[p]);
    const double m_t = darksage_host_mass_within(&galaxies[t], sqrt(dot3(dpos, dpos)), m_p, run_params);
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

    double new_gas[N_BINS] = {0.0}, new_metals[N_BINS] = {0.0};
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
                const double j_seg = sqrt((n_new[0] * j[0]) * (n_new[0] * j[0]) + (n_new[1] * j[1]) * (n_new[1] * j[1])
                                          + (n_new[2] * j[2]) * (n_new[2] * j[2]));
                const int bin = j_bin(j_seg, run_params);
                new_gas[bin] += dm;
                new_metals[bin] += dz;
            }
        }
    }

    for(int i = 0; i < N_BINS; i++) {
        galaxies[t].DiscGas[i] = (float)new_gas[i];
        galaxies[t].DiscGasMetals[i] = (float)new_metals[i];
        galaxies[p].DiscGas[i] = galaxies[p].DiscGasMetals[i] = 0.0f;
    }
    for(int k = 0; k < 3; k++) galaxies[t].SpinGas[k] = (float)n_new[k];
}

/*
 * Bulge kinematics after the satellite's stars have joined the central's bulges, from the
 * pre-merger masses (DarkSage add_galaxies_together()).  The stars go where SAGE26's
 * add_galaxies_together() puts them: each satellite bulge into the same central bulge, the
 * satellite's disc into the instability bulge if the central stays disc-dominated, else into
 * the merger bulge.  For each central bulge
 *   sigma^2 = [M sigma^2 of the bulge + sum over the stars it receives of m (sigma^2 + v_r^2)]
 *             / (bulge + stars received),
 * v_r being the satellite's radial velocity; in a major merger the central's own m v_r^2 also
 * goes to the merger bulge (to the instability bulge if the merger bulge receives nothing,
 * so it is not lost), which make_bulge_from_burst() then builds from everything.
 * Merger-bulge spin: minor -- the mass-weighted spins of both merger bulges and, if it joins,
 * the satellite's disc J; major -- the two progenitors' orbital j about the centre of momentum.
 * The instability bulge carries no spin (DarkSage).  The satellite's intrahalo stars join the
 * central's ICS, so they are not counted.
 */
static void merge_bulge_kinematics(const int t, const int p, const int major, const double j_t[3], const double j_p[3],
                                   const double vr_t, const double vr_p, struct GALAXY *galaxies,
                                   const struct params *run_params)
{
    struct GALAXY *gt = &galaxies[t], *gp = &galaxies[p];
    const double cb_t = gt->MergerBulgeMass, ib_t = gt->InstabilityBulgeMass;
    const double ms_t = gt->StellarMass, ms_p = gp->StellarMass;
    const double vr2 = vr_p * vr_p;

    double m_disc = 0.0, msig2_disc = 0.0, J_disc = 0.0;
    for(int i = 0; i < N_BINS; i++) {
        m_disc += gp->DiscStars[i];
        msig2_disc += gp->DiscStars[i] * gp->VelDispStars[i] * gp->VelDispStars[i];
        J_disc += gp->DiscStars[i] * 0.5 * (run_params->DiscBinEdge[i] + run_params->DiscBinEdge[i+1]);
    }

    /* Same test as add_galaxies_together(): the central's disc fraction once the satellite's
     * stars are in its bulge */
    const double sat_disc = ms_p - gp->BulgeMass;
    const int disc_to_ib = (sat_disc > 0.0 && ms_t + ms_p > 0.0 && (ms_t - gt->BulgeMass) / (ms_t + ms_p) > 0.5);

    double m_cb = gp->MergerBulgeMass;
    double msig2_cb = gp->MergerBulgeMass * (gp->VelDispMergerBulge * gp->VelDispMergerBulge + vr2);
    double m_ib = gp->InstabilityBulgeMass;
    double msig2_ib = gp->InstabilityBulgeMass * (gp->VelDispBulge * gp->VelDispBulge + vr2);
    if(disc_to_ib) {
        m_ib += m_disc;
        msig2_ib += msig2_disc + m_disc * vr2;
    } else {
        m_cb += m_disc;
        msig2_cb += msig2_disc + m_disc * vr2;
    }

    if(major) {
        if(cb_t + m_cb > 0.0) msig2_cb += ms_t * vr_t * vr_t;
        else msig2_ib += ms_t * vr_t * vr_t;
        for(int k = 0; k < 3; k++) {
            const double s = (ms_p + ms_t > 0.0) ? (j_p[k] * ms_p + j_t[k] * ms_t) / (ms_p + ms_t) : 0.0;
            gt->SpinClassicalBulge[k] = isfinite(s) ? (float)s : 0.0f;
        }
    } else if(cb_t + m_cb > 0.0) {
        const double J_in = disc_to_ib ? 0.0 : J_disc;
        for(int k = 0; k < 3; k++) {
            gt->SpinClassicalBulge[k] = (float)((cb_t * gt->SpinClassicalBulge[k] + gp->MergerBulgeMass * gp->SpinClassicalBulge[k]
                                                 + J_in * gp->SpinStars[k]) / (cb_t + m_cb));
        }
    }
    if(cb_t + m_cb > 0.0) {
        gt->VelDispMergerBulge = (float)sqrt((cb_t * gt->VelDispMergerBulge * gt->VelDispMergerBulge + msig2_cb) / (cb_t + m_cb));
    }
    if(ib_t + m_ib > 0.0) {
        gt->VelDispBulge = (float)sqrt((ib_t * gt->VelDispBulge * gt->VelDispBulge + msig2_ib) / (ib_t + m_ib));
    }
}

void darksage_merge_gas(const int t, const int p, struct GALAXY *galaxies, const struct params *run_params)
{
    struct GALAXY *gt = &galaxies[t], *gp = &galaxies[p];

    /* Mass ratio as in deal_with_galaxy_merger() */
    const double m1 = gp->StellarMass + gp->ColdGas, m2 = gt->StellarMass + gt->ColdGas;
    const double mi = (m1 < m2) ? m1 : m2, ma = (m1 < m2) ? m2 : m1;
    const double mass_ratio = (ma > 0.0) ? mi / ma : 1.0;

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

    double j_t[3] = {0.0, 0.0, 0.0}, j_p[3] = {0.0, 0.0, 0.0}, vr_t = 0.0, vr_p;
    const int major = (mass_ratio > run_params->ThreshMajorMerger);
    if(major) {
        merge_gas_major(t, p, dpos, dvel, j_t, j_p, &vr_t, &vr_p, galaxies, run_params);
    } else {
        vr_p = merge_gas_minor(t, p, dpos, dvel, galaxies, run_params);
    }
    merge_bulge_kinematics(t, p, major, j_t, j_p, vr_t, vr_p, galaxies, run_params);
    for(int i = 0; i < N_BINS; i++) gp->DiscGas[i] = gp->DiscGasMetals[i] = 0.0f;
}

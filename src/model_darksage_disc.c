/*
 * model_darksage_disc.c -- disc annulus radii from the rotation curve (DarkSAGEMode == 1).
 *
 * Follows DarkSage's update_disc_radii() (model_misc.c; Stevens, Croton &
 * Mutch 2016 Sec. 3.2; Stevens et al. 2024 Sec. 4).  The rotation curve is
 * built on a 51-point radial grid, spaced by ExponentBin inwards from
 * max(10 j_max / Vvir, Rvir), from
 *   - an NFW halo (the dark matter left after the disc, bulge, hot gas and
 *     black hole), concentration from SAGE26's ConcentrationOn when set,
 *     otherwise Dutton & Maccio (2014) with the Di Cintio et al. (2014)
 *     baryonic modification;
 *   - Hernquist instability- and merger-driven bulges (SAGE26's bulge radii);
 *   - the black hole as a point mass;
 *   - exponential stellar and gas discs;
 *   - a beta-profile hot halo (hot / CGM, fountain and outflowing gas);
 * with only the fraction f_support = 1 - exp(-r / RotSupportScaleRadius) of
 * the support in rotation: V_rot^2 = f_support V_circ^2, j = V_rot r.  Annulus
 * edge i sits at the radius where j = DiscBinEdge[i]; the potential, integrated
 * inwards, gives Potential[] at the edges.  Galaxies with no disc (B/T >= 1)
 * keep their last radii, as in DarkSage.
 *
 * Deviations from the source: the grid is interpolated linearly (DarkSage uses
 * a cubic spline); intrahalo stars follow the halo's NFW profile (DarkSage
 * gives them their own Hernquist profile).
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "core_allvars.h"

#include "model_darksage_disc.h"
#include "model_halo_properties.h"

/* Radial grid of the rotation curve (DarkSage NUM_R_BINS) */
#define N_RC_GRID 51
/* Innermost grid radius relative to the halo scale radius (DarkSage's rrat cut) */
static const double RC_MIN_RRAT = 1.0e-8;
/* Times the outermost grid radius may grow x10 to reach the largest j-bin edge */
static const int RC_MAX_GROW = 10;
/* Floor on the rotation-support fraction (DarkSage) */
static const double F_SUPPORT_MIN = 1.0e-5;
/* DarkSage minimum core radius of the hot-gas beta profile (MIN_C_BETA) */
static const double DS_MIN_C_BETA = 0.05;
/* DarkSage cold-gas velocity dispersion 11 + 11.3 z km/s [cm/s] */
static const double DS_CS0_CGS = 1.1e6;
static const double DS_CS1_CGS = 1.13e6;

/* An exponential disc encloses 10%, 20%, ..., 90% of its mass within PERCENT_RADIUS
 * scale lengths, so r_d = r_x / PERCENT_RADIUS (DarkSage main.c DiscScalePercentConversion) */
static const double PERCENT_VALUES[9] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
static const double PERCENT_RADIUS[9] = {0.531812, 0.824388, 1.09735, 1.37642, 1.67835, 2.02231, 2.43922, 2.99431, 3.88927};

/* Exponential-disc contribution to V_circ^2 (dimensionless); x = r/r_d, c = Rvir/r_d */
static inline double v2_disc_cterm(const double x, const double c)
{
    const double cx = c * x;
    const double inv_cx = 1.0 / cx;
    return (c + 4.8 * c * exp(-0.35 * cx - 3.5 * inv_cx)) / (cx + inv_cx * inv_cx + 2.0 * sqrt(inv_cx));
}

/*
 * Exponential scale length of an annulus mass distribution (DarkSage
 * update_stellardisc_scaleradius()): the mean over x = 10%, ..., 90% of
 * r_x / PERCENT_RADIUS[x], r_x interpolated from the cumulative mass.  An empty
 * disc keeps its current value (the disc scale radius if it has none).
 */
static double disc_scaleradius(const float mass[N_BINS], const double current, const struct GALAXY *g)
{
    double total = 0.0;
    for(int i = 0; i < N_BINS; i++) total += mass[i];
    if(total <= 0.0) return (current > 0.0) ? current : g->DiskScaleRadius;

    double cum[N_BINS + 1];
    cum[0] = 0.0;
    for(int i = 0; i < N_BINS; i++) cum[i+1] = cum[i] + mass[i] / total;

    double Rscale_sum = 0.0;
    int i_min = 1;
    for(int j = 0; j < 9; j++) {
        for(int i = i_min; i < N_BINS + 1; i++) {
            if(cum[i] >= PERCENT_VALUES[j]) {
                const double dcum = cum[i] - cum[i-1];
                if(dcum > 0.0) {
                    const double r = (g->DiscRadii[i] * (PERCENT_VALUES[j] - cum[i-1])
                                      + g->DiscRadii[i-1] * (cum[i] - PERCENT_VALUES[j])) / dcum;
                    Rscale_sum += r / PERCENT_RADIUS[j];
                }
                i_min = i;
                break;
            }
        }
    }
    const double rs = Rscale_sum / 9.0;
    return (rs > 0.0) ? rs : g->DiskScaleRadius;
}

/* Hernquist scale length a whose mass within Rvir is half inside r_half */
static double hernquist_a_from_half_mass(const double r_half, const double Rvir)
{
    if(r_half <= 0.0) return 0.0;
    if(Rvir > M_SQRT2 * r_half) return r_half * Rvir * (M_SQRT2 - 1.0) / (Rvir - M_SQRT2 * r_half);
    return r_half / (1.0 + M_SQRT2);
}

/* Linear interpolation of y(x) on the valid part of a grid with increasing x */
static double grid_interp(const double *x, const double *y, const int first, const double xv)
{
    if(xv <= x[first]) return y[first];
    for(int k = first + 1; k < N_RC_GRID; k++) {
        if(xv <= x[k]) return y[k-1] + (y[k] - y[k-1]) * (xv - x[k-1]) / (x[k] - x[k-1]);
    }
    return y[N_RC_GRID - 1];
}

/* Radius where rotational support f_rot = v / sqrt(v^2 + sigma^2) reaches 0.5, as an exponential
 * scale (DarkSage update_rotation_support_scale_radius()); gas-only annuli use the cold-gas dispersion */
static void update_rotation_support_scale_radius(const int p, struct GALAXY *galaxies, const struct params *run_params)
{
    double rad_old = 0.0, f_rot_old = 0.0;
    const double sigma_gas = (DS_CS0_CGS + DS_CS1_CGS * run_params->ZZ[galaxies[p].SnapNum]) / run_params->UnitVelocity_in_cm_per_s;

    for(int i = 0; i < N_BINS; i++) {
        const double r_in = galaxies[p].DiscRadii[i];
        const double r_out = galaxies[p].DiscRadii[i+1];
        const double rad = sqrt(0.5 * (r_in * r_in + r_out * r_out));
        if(rad <= 0.0) continue;

        const double v_circ = 0.5 * (run_params->DiscBinEdge[i] + run_params->DiscBinEdge[i+1]) / rad;
        const double sigma = (galaxies[p].DiscStars[i] > 0.0) ? galaxies[p].VelDispStars[i] : sigma_gas;
        const double f_rot = v_circ / sqrt(v_circ * v_circ + sigma * sigma);

        if(f_rot >= 0.5 && f_rot > f_rot_old) {
            const double rad_50 = rad - (f_rot - 0.5) / (f_rot - f_rot_old) * (rad - rad_old);
            galaxies[p].RotSupportScaleRadius = rad_50 * 1.442695;   /* -1/ln(0.5) */
            return;
        }
        rad_old = rad;
        f_rot_old = f_rot;
    }

    galaxies[p].RotSupportScaleRadius = (f_rot_old > 0.0 && f_rot_old < 1.0)
        ? -rad_old / log(1.0 - f_rot_old) : galaxies[p].DiskScaleRadius;
}

/*
 * update_disc_radii -- annulus radii and potentials of galaxy p from its rotation curve.
 *
 * Algorithm:
 *   1. Disc scale radii and the rotation-support scale radius from the annuli.
 *   2. Halo: the dark matter left after the baryons, NFW; c_beta(z) of the hot halo.
 *   3. If the galaxy has a disc (B/T < 1), build the grid inwards from
 *      max(10 j_max / Vvir, Rvir): enclosed spherical mass (halo, bulges, hot
 *      gas, black hole), the two discs, V_rot^2 = f_support V_circ^2 and
 *      j = V_rot r.  If the outermost j is below j_max, the radius grows x10.
 *      The potential is integrated inwards with dPhi/dr = V_circ^2 / r.
 *   4. Annulus edges at j = DiscBinEdge[i] and their potentials.
 */
void update_disc_radii(const int p, struct GALAXY *galaxies, const struct halo_data *halos,
                       const struct params *run_params)
{
    struct GALAXY *g = &galaxies[p];
    (void)halos;
    if(g->Vvir <= 0.0 || g->Rvir <= 0.0 || g->Mvir <= 0.0) return;

    const double Rvir = g->Rvir;
    const double G = run_params->G;
    const double z = run_params->ZZ[g->SnapNum];

    /* Step 1: disc and rotation-support scale radii */
    g->StellarDiscScaleRadius = (float)disc_scaleradius(g->DiscStars, g->StellarDiscScaleRadius, g);
    g->GasDiscScaleRadius = (float)disc_scaleradius(g->DiscGas, g->GasDiscScaleRadius, g);
    update_rotation_support_scale_radius(p, galaxies, run_params);

    double DiscStars = 0.0, DiscGas = 0.0;
    for(int i = 0; i < N_BINS; i++) {
        DiscStars += g->DiscStars[i];
        DiscGas += g->DiscGas[i];
    }

    /* Step 2: halo and hot gas */
    const double M_hot_total = g->HotGas + g->CGMgas + g->FountainGas + g->OutflowGas;
    double M_DM = g->Mvir - (M_hot_total + g->ColdGas + g->StellarMass + g->BlackHoleMass);
    if(M_DM < 0.0) M_DM = 0.0;

    double conc;
    if(run_params->ConcentrationOn > 0 && g->Concentration > 0.0f) {
        conc = g->Concentration;
    } else {
        const double zc = (z > 5.0) ? 5.0 : z;
        const double a_conc = 0.520 + (0.905 - 0.520) * exp(-0.617 * pow(zc, 1.21));
        const double b_conc = -0.101 + 0.026 * zc;
        conc = pow(10.0, a_conc + b_conc * (log10(g->Mvir * run_params->UnitMass_in_g / SOLAR_MASS) - 12.0));
        if(g->Type == 0 && g->StellarMass > 0.0) {
            const double X = log10(g->StellarMass / g->Mvir);
            conc *= 1.0 + 3.0e-5 * exp(3.4 * (X + 4.5));   /* Di Cintio et al. (2014b) */
        }
    }
    const double r_s = Rvir / conc;
    g->HaloScaleRadius = (float)r_s;
    const double rho_norm = M_DM / (log(1.0 + Rvir / r_s) - Rvir / (Rvir + r_s));

    double c_beta = 0.20 * exp(-1.5 * z) - 0.039 * z + 0.28;
    if(c_beta < DS_MIN_C_BETA) c_beta = DS_MIN_C_BETA;
    g->c_beta = (float)c_beta;
    const double hot_norm = M_hot_total / (1.0 - c_beta * atan(1.0 / c_beta));

    const double total_stellar_gas = g->StellarMass + g->ColdGas;
    const double BTT = (total_stellar_gas > 0.0) ? g->BulgeMass / total_stellar_gas : 0.0;
    if(BTT >= 1.0) return;   /* no disc: keep the last radii */

    /* Hernquist bulges, normalised to their mass within Rvir */
    const double a_SB = hernquist_a_from_half_mass((g->InstabilityBulgeRadius > 0.0f) ? g->InstabilityBulgeRadius : g->BulgeRadius, Rvir);
    const double a_CB = hernquist_a_from_half_mass((g->MergerBulgeRadius > 0.0f) ? g->MergerBulgeRadius : g->BulgeRadius, Rvir);
    const double M_SB_inf = g->InstabilityBulgeMass * ((Rvir + a_SB) / Rvir) * ((Rvir + a_SB) / Rvir);
    const double M_CB_inf = g->MergerBulgeMass * ((Rvir + a_CB) / Rvir) * ((Rvir + a_CB) / Rvir);

    double a_sdisc = g->StellarDiscScaleRadius, a_gdisc = g->GasDiscScaleRadius;
    if(a_sdisc <= 0.0) a_sdisc = g->DiskScaleRadius;
    if(a_gdisc <= 0.0) a_gdisc = g->DiskScaleRadius;
    const double r_support = (g->RotSupportScaleRadius > 0.0f) ? g->RotSupportScaleRadius : g->DiskScaleRadius;

    /* Step 3: the grid, from the outside in */
    double gr_r[N_RC_GRID], gr_j[N_RC_GRID], gr_phi[N_RC_GRID], f_sup[N_RC_GRID];
    int first = 0, n_grow = 0;
    double r = 10.0 * run_params->DiscBinEdge[N_BINS] / g->Vvir;
    if(r < Rvir) r = Rvir;
    gr_r[0] = gr_j[0] = gr_phi[0] = 0.0;
    int i;
    for(i = N_RC_GRID - 1; i > 0; i--) {
        const double rrat = r / r_s;
        if(rrat < RC_MIN_RRAT) { first = i + 1; break; }
        gr_r[i] = r;
        const double x = r / Rvir;
        double M_int = rho_norm * (log(1.0 + rrat) - rrat / (1.0 + rrat));
        if(M_int < 0.0) M_int = 0.0;
        M_int += M_SB_inf * (r / (r + a_SB)) * (r / (r + a_SB)) + M_CB_inf * (r / (r + a_CB)) * (r / (r + a_CB));
        M_int += hot_norm * (x - c_beta * atan(x / c_beta)) + g->BlackHoleMass;
        double v2 = G * M_int / r;
        if(DiscStars > 0.0 && a_sdisc > 0.0) v2 += G * DiscStars / Rvir * v2_disc_cterm(x, Rvir / a_sdisc);
        if(DiscGas > 0.0 && a_gdisc > 0.0) v2 += G * DiscGas / Rvir * v2_disc_cterm(x, Rvir / a_gdisc);

        double f = 1.0 - exp(-r / r_support);
        if(f < F_SUPPORT_MIN) f = F_SUPPORT_MIN;
        f_sup[i] = f;
        gr_j[i] = sqrt(f * v2) * r;

        if(i == N_RC_GRID - 1) {
            gr_phi[i] = -G * M_int / r;
            if(gr_j[i] < run_params->DiscBinEdge[N_BINS] && n_grow < RC_MAX_GROW) {
                r *= 10.0;   /* grid must reach j_max */
                n_grow++;
                i++;
                continue;
            }
        } else {
            if(!(gr_j[i] < gr_j[i+1])) gr_j[i] = 0.999 * gr_j[i+1];   /* keep j(r) invertible */
            gr_phi[i] = gr_phi[i+1] - (gr_r[i+1] - gr_r[i]) * 0.5
                        * (gr_j[i+1] * gr_j[i+1] / (f_sup[i+1] * gr_r[i+1] * gr_r[i+1] * gr_r[i+1]) + v2 / r);
        }
        r /= run_params->ExponentBin;
    }
    if(i == 0) gr_phi[0] = gr_phi[1];   /* the potential flattens at the centre */

    /* Step 4: annulus edges and their potentials */
    g->DiscRadii[0] = 0.0f;
    for(int k = 1; k < N_BINS + 1; k++) {
        g->DiscRadii[k] = (float)grid_interp(gr_j, gr_r, first, run_params->DiscBinEdge[k]);
        if(g->DiscRadii[k] <= g->DiscRadii[k-1]) g->DiscRadii[k] = 1.0001f * g->DiscRadii[k-1] + 1.0e-9f;
        g->Potential[k] = (float)grid_interp(gr_r, gr_phi, first, g->DiscRadii[k]);
    }
    g->Potential[0] = (float)gr_phi[first];
}

/* ---------------------------------------------------------------------------
 * Gas-disc precession towards the stellar disc
 * -------------------------------------------------------------------------*/

/* Rodrigues rotation of v about unit axis by angle_deg */
static void rotate_vector(float v[3], const float axis[3], const double angle_deg)
{
    if(angle_deg == 0.0) return;

    const double a = angle_deg * M_PI / 180.0;
    const double cos_a = cos(a), sin_a = sin(a);
    const double cross[3] = {axis[1] * v[2] - axis[2] * v[1],
                             axis[2] * v[0] - axis[0] * v[2],
                             axis[0] * v[1] - axis[1] * v[0]};
    const double dot = axis[0] * v[0] + axis[1] * v[1] + axis[2] * v[2];
    for(int j = 0; j < 3; j++) {
        v[j] = (float)(v[j] * cos_a + cross[j] * sin_a + axis[j] * dot * (1.0 - cos_a));
    }
}

/* Re-bin a disc tilted by angle acos(cos_angle): its j along the new axis is j cos(angle) */
static void project_disc(const float DiscMass[N_BINS], const double cos_angle, double NewDisc[N_BINS],
                         const struct params *run_params, const double Rvir, const double Vvir)
{
    const double cos_ang = fabs(cos_angle);
    const double *edge = run_params->DiscBinEdge;

    if(cos_ang > 0.99) {
        for(int i = 0; i < N_BINS; i++) NewDisc[i] = DiscMass[i];
        return;
    }

    double work[N_BINS];
    for(int i = 0; i < N_BINS; i++) work[i] = DiscMass[i];

    int j_old = 0;
    for(int i = 0; i < N_BINS; i++) {
        const double high_bound = edge[i+1] / cos_ang;

        int j = j_old;
        while(j < N_BINS && edge[j] < high_bound) j++;
        j--;
        if(j < 0) j = 0;

        /* Old bins wholly inside new bin i */
        NewDisc[i] = 0.0;
        for(int l = j_old; l < j && l < N_BINS; l++) {
            NewDisc[i] += work[l];
            work[l] = 0.0;
        }

        if(i < N_BINS - 1 && j < N_BINS) {
            /* Partial share of the boundary bin, by area */
            const double upper = (j < N_BINS - 1) ? edge[j+1] : Rvir * Vvir;
            double ratio = 1.0;
            if(upper > edge[j]) {
                ratio = (high_bound - edge[j]) / (upper - edge[j]);
                ratio *= ratio;
                if(ratio > 1.0) ratio = 1.0;
                if(ratio < 0.0) ratio = 0.0;
            }
            NewDisc[i] += ratio * work[j];
            work[j] -= ratio * work[j];
        } else if(i == N_BINS - 1) {
            /* Last bin takes everything left */
            for(int l = j_old; l < N_BINS; l++) NewDisc[i] += work[l];
        }

        if(NewDisc[i] < 0.0) NewDisc[i] = 0.0;
        j_old = j;
    }
}

void darksage_precess_gas(const int p, const double dt, struct GALAXY *galaxies, const struct params *run_params)
{
    double gas_sum = 0.0;
    for(int i = 0; i < N_BINS; i++) gas_sum += galaxies[p].DiscGas[i];
    if(gas_sum <= 0.0 || dt <= 0.0) return;

    float *SpinGas = galaxies[p].SpinGas;
    const float *SpinStars = galaxies[p].SpinStars;

    double cos_theta = SpinGas[0] * SpinStars[0] + SpinGas[1] * SpinStars[1] + SpinGas[2] * SpinStars[2];
    if(cos_theta > 1.0) cos_theta = 1.0;
    if(cos_theta < -1.0) cos_theta = -1.0;

    /* Already (counter-)aligned within ThetaThresh */
    if(fabs(cos_theta) > cos(run_params->ThetaThresh * M_PI / 180.0)) return;

    /* Mass-weighted precession angle: DegPerTdyn per local t_dyn = r^2 / j */
    double deg = 0.0;
    for(int i = 0; i < N_BINS; i++) {
        const double j_edge = run_params->DiscBinEdge[i+1];
        const double r_out = galaxies[p].DiscRadii[i+1];
        if(galaxies[p].DiscGas[i] <= 0.0 || j_edge <= 0.0 || r_out <= 0.0) continue;
        deg += run_params->DegPerTdyn * dt / (r_out * r_out / j_edge) * galaxies[p].DiscGas[i] / gas_sum;
    }
    if(deg <= 0.0) return;

    /* Rotate by deg, but not past alignment */
    double cos_precess = cos(deg * M_PI / 180.0);
    if(cos_precess < fabs(cos_theta)) cos_precess = fabs(cos_theta);

    double new_gas[N_BINS], new_metals[N_BINS];
    project_disc(galaxies[p].DiscGas, cos_precess, new_gas, run_params, galaxies[p].Rvir, galaxies[p].Vvir);
    project_disc(galaxies[p].DiscGasMetals, cos_precess, new_metals, run_params, galaxies[p].Rvir, galaxies[p].Vvir);
    for(int i = 0; i < N_BINS; i++) {
        galaxies[p].DiscGas[i] = (float)new_gas[i];
        galaxies[p].DiscGasMetals[i] = (float)new_metals[i];
    }

    if(cos_precess == fabs(cos_theta)) {
        /* Reached (counter-)alignment */
        const float sign = (cos_theta >= 0.0) ? 1.0f : -1.0f;
        for(int j = 0; j < 3; j++) SpinGas[j] = sign * SpinStars[j];
        return;
    }

    /* Rotate SpinGas towards +/-SpinStars about their common normal */
    float axis[3] = {SpinGas[1] * SpinStars[2] - SpinGas[2] * SpinStars[1],
                     SpinGas[2] * SpinStars[0] - SpinGas[0] * SpinStars[2],
                     SpinGas[0] * SpinStars[1] - SpinGas[1] * SpinStars[0]};
    if(cos_theta < 0.0) {
        for(int j = 0; j < 3; j++) axis[j] = -axis[j];
    }
    const double axis_mag = sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
    if(axis_mag <= 0.0) return;
    for(int j = 0; j < 3; j++) axis[j] = (float)(axis[j] / axis_mag);

    rotate_vector(SpinGas, axis, acos(cos_precess) * 180.0 / M_PI);

    const double spin_mag = sqrt(SpinGas[0] * SpinGas[0] + SpinGas[1] * SpinGas[1] + SpinGas[2] * SpinGas[2]);
    if(spin_mag > 0.0) {
        for(int j = 0; j < 3; j++) SpinGas[j] = (float)(SpinGas[j] / spin_mag);
    }
}

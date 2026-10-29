/*
 * model_darksage_disc.c -- disc annulus radii from the rotation curve (DarkSAGEMode == 1).
 *
 * Each j-bin edge j_i is mapped to the radius r_i solving j_i = r_i V_circ(r_i),
 * with V_circ from an NFW halo (Dutton & Maccio 2014 concentration), a
 * Hernquist bulge, exponential stellar and gas discs and a beta-profile hot
 * halo (Stevens, Croton & Mutch 2016).  Radii are capped at the current Rvir.
 *
 * The same NFW potential sets the energetics DarkSage's feedback and
 * reincorporation use (Stevens et al. 2024, Secs. 9-10): the potential at each
 * annulus edge, the mass-weighted potential of the hot halo (DarkSage's beta
 * profile, or SAGE26's rho ~ r^-1.4 CGM profile for CGM-regime haloes) and the
 * potential at Rvir that ejected gas must climb to.
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "core_allvars.h"

#include "model_darksage_disc.h"
#include "model_halo_properties.h"
#include "model_misc.h"

/* Components of the rotation curve used to place the annuli */
struct rotation_curve {
    double G, rho_norm, r_s;          /* NFW dark matter */
    double M_bulge, a_bulge;          /* Hernquist bulge */
    double M_sdisc, a_sdisc;          /* exponential stellar disc */
    double M_gdisc, a_gdisc;          /* exponential gas disc */
    double M_hot, c_beta;             /* beta-profile hot gas */
    double Rvir;
};

static inline double v2_disc_cterm(const double x, const double c);

/* V_circ^2 at radius r [(km/s)^2] from all rotation-curve components */
static double circular_velocity2(const struct rotation_curve *rc, const double r)
{
    double v2 = rc->G * rc->rho_norm * (log(1.0 + r / rc->r_s) - r / (r + rc->r_s)) / r;   /* NFW */
    if(v2 < 0.0) v2 = 0.0;
    if(rc->M_bulge > 0.0 && rc->a_bulge > 0.0) {
        v2 += rc->G * rc->M_bulge * r / ((r + rc->a_bulge) * (r + rc->a_bulge));   /* Hernquist */
    }
    if(rc->M_sdisc > 0.0 && rc->a_sdisc > 0.0) {
        v2 += rc->G * rc->M_sdisc / rc->Rvir * v2_disc_cterm(r / rc->a_sdisc, rc->Rvir / rc->a_sdisc);
    }
    if(rc->M_gdisc > 0.0 && rc->a_gdisc > 0.0) {
        v2 += rc->G * rc->M_gdisc / rc->Rvir * v2_disc_cterm(r / rc->a_gdisc, rc->Rvir / rc->a_gdisc);
    }
    if(rc->M_hot > 0.0 && rc->c_beta > 0.0) {
        const double cb_norm = 1.0 / (1.0 - rc->c_beta * atan(1.0 / rc->c_beta));
        const double M_hot_enc = rc->M_hot * cb_norm * (atan(r / rc->Rvir / rc->c_beta) - rc->c_beta * atan(1.0 / rc->c_beta));
        if(M_hot_enc > 0.0) v2 += rc->G * M_hot_enc / r;
    }
    return v2;
}

/* DarkSage radial grid for the bulge-size solution (update_disc_radii(), NUM_R_BINS) */
#define N_BULGE_GRID 51

/*
 * BulgeSizeOn == 4: DarkSage's dynamical size of the instability-driven bulge
 * (update_disc_radii() and core_save.c).
 *
 * The Hernquist scale length a of a bulge of 1-D dispersion sigma in
 * equilibrium in the potential Phi satisfies, at each radius r,
 *   a = 3 / (dPhi/dr / (3 sigma^2) - 1/r) - r,
 * with dPhi/dr = V_circ^2 / r.  DarkSage evaluates this on 51 radii from the
 * centre outwards (spaced by ExponentBin, outermost max(10 j_max / Vvir,
 * Rvir)) and keeps the running mean once it changes by <= 1%.  The output is
 * the radius enclosing half the bulge mass within Rvir, the positive root of
 *   (2a^2 + 4 a Rvir + Rvir^2) r^2 - 2 a Rvir^2 r - a^2 Rvir^2 = 0.
 * Sets InstabilityBulgeRadius; leaves it unchanged if the mean never settles.
 */
static void darksage_instability_bulge_size(const int p, const struct rotation_curve *rc, struct GALAXY *galaxies,
                                            const struct params *run_params)
{
    const double sigma2 = 3.0 * galaxies[p].VelDispBulge * galaxies[p].VelDispBulge;
    const double Rvir = galaxies[p].Rvir;
    if(galaxies[p].InstabilityBulgeMass <= 0.0f || sigma2 <= 0.0 || Rvir <= 0.0) return;

    double r_out = 10.0 * run_params->DiscBinEdge[N_BINS] / galaxies[p].Vvir;
    if(r_out < Rvir) r_out = Rvir;

    double g_prev = 0.0, a_av = 0.0, a_av_prev = 0.0, a = -1.0;
    for(int k = 1; k < N_BULGE_GRID; k++) {
        const double r = r_out * pow(run_params->ExponentBin, -(double)(N_BULGE_GRID - 1 - k));
        const double g = circular_velocity2(rc, r) / r;   /* dPhi/dr */
        const double grad = (k == 1) ? g : 0.5 * (g_prev + g);
        const double a_new = 3.0 / (grad / sigma2 - 1.0 / r) - r;
        a_av = ((k - 1) * a_av + a_new) / k;
        if(k > 1 && fabs(a_av - a_av_prev) <= 0.01 * a_av) {
            a = a_av;
            break;
        }
        a_av_prev = a_av;
        g_prev = g;
    }
    if(!(a > 0.0) || !isfinite(a)) return;

    const double qa = 2.0 * a * a + 4.0 * a * Rvir + Rvir * Rvir;
    const double qb = -2.0 * a * Rvir * Rvir;
    const double qc = -a * a * Rvir * Rvir;
    galaxies[p].InstabilityBulgeRadius = (float)((-qb + sqrt(qb * qb - 4.0 * qa * qc)) / (2.0 * qa));
}

/* Mass-percentile radii of an exponential disc, as multiples of the scale length's fit */
static const double PERCENT_VALUES[9]     = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
static const double PERCENT_CONVERSION[9] = {0.3567, 0.4214, 0.4765, 0.5268, 0.5752, 0.6244, 0.6785, 0.7491, 0.8684};

/* Exponential-disc contribution to V_circ^2 (dimensionless); x = r/r_d, c = Rvir/r_d */
static inline double v2_disc_cterm(const double x, const double c)
{
    const double cx = c * x;
    const double inv_cx = 1.0 / cx;
    return (c + 4.8 * c * exp(-0.35 * cx - 3.5 * inv_cx)) / (cx + inv_cx * inv_cx + 2.0 * sqrt(inv_cx));
}

/* DarkSage minimum core radius of the hot-gas beta profile (MIN_C_BETA, core_allvars.h) */
static const double DS_MIN_C_BETA = 0.05;
/* Coefficient of DarkSage's R2_hot_av for the beta profile (model_cooling.c) */
static const double DS_R2_HOT_COEFF = 0.15343;
/* Carr et al. (2023) CGM profile used by the SAGE26 CGM regime (model_cooling_heating.c):
 * rho ~ r^-CGM_ALPHA between CGM_R0_FRAC * Rvir and Rvir */
static const double CGM_ALPHA = 1.4;
static const double CGM_R0_FRAC = 0.1;
/* Radial sampling of the hot-gas profile for its mass-weighted potential: log-spaced
 * from PROFILE_R_MIN_FRAC * Rvir to Rvir */
#define N_PROFILE 200
static const double PROFILE_R_MIN_FRAC = 1.0e-4;

/*
 * NFW potential at radius r [code units], normalised to Mvir within Rvir.
 * At r = 0 it returns the finite central value -G Mvir / (r_s norm).
 */
double darksage_nfw_potential(const int p, const double r, const struct GALAXY *galaxies, const struct params *run_params)
{
    const double rs = galaxies[p].HaloScaleRadius;
    if(r < 0.0 || rs <= 0.0) return 0.0;

    const double c = galaxies[p].Rvir / rs;
    const double norm = log(1.0 + c) - c / (1.0 + c);
    if(norm <= 0.0) return 0.0;

    if(r == 0.0) return -run_params->G * galaxies[p].Mvir / (rs * norm);
    return -run_params->G * galaxies[p].Mvir / r * log(1.0 + r / rs) / norm;
}

/*
 * update_hot_energetics -- potentials of the hot and ejected reservoirs.
 *
 * Follows DarkSage's update_disc_radii() (model_misc.c) and model_cooling.c:
 *   c_beta      = max(0.05, 0.20 exp(-1.5 z) - 0.039 z + 0.28)
 *   HotGasPotential = mass-weighted potential of the hot gas within Rvir:
 *                 beta profile rho ~ 1 / (c_beta^2 + x^2), x = r / Rvir, whose
 *                 enclosed mass grows as x - c_beta atan(x / c_beta);
 *                 for CGM-regime haloes (CGMrecipeOn, Regime == 0) SAGE26's
 *                 rho ~ r^-1.4 profile on [0.1, 1] Rvir instead.
 *   R2_hot_av   = (Rvir c_beta^2 cb 0.15343)^2 for the beta profile (DarkSage);
 *                 for the CGM profile 1 / <1/r^2>, the radius at which
 *                 0.5 j^2 / r^2 equals its mass-weighted mean.
 *   EjectedPotential = max(0.9999 HotGasPotential, Phi(Rvir)).
 * Potentials come from the NFW halo alone, as Potential[] does here.
 */
static void update_hot_energetics(const int p, const double z, struct GALAXY *galaxies, const struct params *run_params)
{
    const double Rvir = galaxies[p].Rvir;
    if(Rvir <= 0.0 || galaxies[p].HaloScaleRadius <= 0.0) return;

    double c_beta = 0.20 * exp(-1.5 * z) - 0.039 * z + 0.28;
    if(c_beta < DS_MIN_C_BETA) c_beta = DS_MIN_C_BETA;
    galaxies[p].c_beta = (float)c_beta;

    const int cgm_regime = (run_params->CGMrecipeOn == 1 && galaxies[p].Regime == 0);
    const double x_min = cgm_regime ? CGM_R0_FRAC : PROFILE_R_MIN_FRAC;
    const double dlnx = -log(x_min) / (N_PROFILE - 1);

    /* Trapezoid sums over the profile's mass shells; for the beta profile the shell
     * mass is the exact difference of x - c_beta atan(x / c_beta) */
    double pot_sum = 0.0, inv_r2_sum = 0.0, mass_sum = 0.0;
    double x_prev = x_min, phi_prev = darksage_nfw_potential(p, x_min * Rvir, galaxies, run_params);
    for(int k = 1; k < N_PROFILE; k++) {
        const double x = x_min * exp(k * dlnx);
        const double phi = darksage_nfw_potential(p, x * Rvir, galaxies, run_params);
        double dm;
        if(cgm_regime) {
            dm = (pow(x, 3.0 - CGM_ALPHA) - pow(x_prev, 3.0 - CGM_ALPHA)) / (3.0 - CGM_ALPHA);
        } else {
            dm = (x - c_beta * atan(x / c_beta)) - (x_prev - c_beta * atan(x_prev / c_beta));
        }
        pot_sum += 0.5 * (phi + phi_prev) * dm;
        inv_r2_sum += dm / (x * x_prev);
        mass_sum += dm;
        x_prev = x;
        phi_prev = phi;
    }
    if(mass_sum <= 0.0) return;

    galaxies[p].HotGasPotential = (float)(pot_sum / mass_sum);
    if(cgm_regime) {
        galaxies[p].R2_hot_av = (float)(Rvir * Rvir * mass_sum / inv_r2_sum);
    } else {
        const double cb_term = 1.0 / (1.0 - c_beta * atan(1.0 / c_beta));
        const double r_hot = Rvir * c_beta * c_beta * cb_term * DS_R2_HOT_COEFF;
        galaxies[p].R2_hot_av = (float)(r_hot * r_hot);
    }

    const double phi_rvir = darksage_nfw_potential(p, Rvir, galaxies, run_params);
    galaxies[p].EjectedPotential = (float)((phi_rvir > 0.9999 * galaxies[p].HotGasPotential)
                                           ? phi_rvir : 0.9999 * galaxies[p].HotGasPotential);
}

/* Scale length from the radii enclosing 10..90% of an annulus mass distribution */
static double disc_scaleradius(const int p, const float mass[N_BINS], const struct GALAXY *galaxies)
{
    double total = 0.0;
    for(int i = 0; i < N_BINS; i++) total += mass[i];
    if(total <= 0.0) return galaxies[p].DiskScaleRadius;

    double cum[N_BINS + 1];
    cum[0] = 0.0;
    for(int i = 0; i < N_BINS; i++) cum[i+1] = cum[i] + mass[i] / total;

    double Rscale_sum = 0.0;
    int n_found = 0, i_min = 1;
    for(int j = 0; j < 9; j++) {
        for(int i = i_min; i < N_BINS + 1; i++) {
            if(cum[i] >= PERCENT_VALUES[j]) {
                const double dcum = cum[i] - cum[i-1];
                if(dcum > 0.0) {
                    const double r = (galaxies[p].DiscRadii[i] * (PERCENT_VALUES[j] - cum[i-1])
                                      + galaxies[p].DiscRadii[i-1] * (cum[i] - PERCENT_VALUES[j])) / dcum;
                    Rscale_sum += r * PERCENT_CONVERSION[j];
                    n_found++;
                }
                i_min = i;
                break;
            }
        }
    }

    double rs = (n_found > 0) ? Rscale_sum / n_found : galaxies[p].DiskScaleRadius;
    /* Floor at 1% of Rvir */
    if(galaxies[p].Rvir > 0.0 && rs < 0.01 * galaxies[p].Rvir) rs = 0.01 * galaxies[p].Rvir;
    return rs;
}

/* Radius where rotational support f_rot = v/sqrt(v^2 + sigma^2) reaches 0.5, as an exponential scale */
static void update_rotation_support_scale_radius(const int p, struct GALAXY *galaxies, const struct params *run_params)
{
    double rad_old = 0.0, f_rot_old = 0.0;
    const double default_sigma = 8.0e5 / run_params->UnitVelocity_in_cm_per_s;   /* 8 km/s */

    for(int i = 0; i < N_BINS; i++) {
        const double r_in = galaxies[p].DiscRadii[i];
        const double r_out = galaxies[p].DiscRadii[i+1];
        const double rad = sqrt(0.5 * (r_in * r_in + r_out * r_out));
        if(rad <= 0.0) continue;

        const double v_circ = 0.5 * (run_params->DiscBinEdge[i] + run_params->DiscBinEdge[i+1]) / rad;
        const double sigma = (galaxies[p].DiscStars[i] > 0.0) ? galaxies[p].VelDispStars[i] : default_sigma;
        const double v2_tot = v_circ * v_circ + sigma * sigma;
        const double f_rot = (v2_tot > 0.0) ? v_circ / sqrt(v2_tot) : 0.0;

        if(f_rot >= 0.5 && f_rot > f_rot_old && rad > rad_old) {
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

/* Strictly increasing radii, capped at Rvir_cap */
static void enforce_monotonic_radii(const int p, const double Rvir_cap, struct GALAXY *galaxies)
{
    const double dr_min = 0.001 * Rvir_cap;
    for(int i = 1; i < N_BINS + 1; i++) {
        if(galaxies[p].DiscRadii[i] <= galaxies[p].DiscRadii[i-1]) {
            galaxies[p].DiscRadii[i] = galaxies[p].DiscRadii[i-1] + dr_min;
        }
        if(galaxies[p].DiscRadii[i] > Rvir_cap) galaxies[p].DiscRadii[i] = Rvir_cap;
    }
}

void update_disc_radii(const int p, struct GALAXY *galaxies, const struct halo_data *halos,
                       const struct params *run_params)
{
    if(galaxies[p].Vvir <= 0.0 || galaxies[p].Rvir <= 0.0) return;

    /* Current (possibly stripped) virial radius; orphans keep their last Rvir */
    double current_Rvir = get_virial_radius(galaxies[p].HaloNr, halos, run_params);
    if(galaxies[p].Type == 2 || current_Rvir <= 0.0) current_Rvir = galaxies[p].Rvir;

    const double Rvir = galaxies[p].Rvir;
    const double Vvir = galaxies[p].Vvir;
    const double G = run_params->G;

    galaxies[p].StellarDiscScaleRadius = disc_scaleradius(p, galaxies[p].DiscStars, galaxies);
    galaxies[p].GasDiscScaleRadius = disc_scaleradius(p, galaxies[p].DiscGas, galaxies);
    update_rotation_support_scale_radius(p, galaxies, run_params);

    double DiscStars = 0.0, DiscGas = 0.0;
    for(int i = 0; i < N_BINS; i++) {
        DiscStars += galaxies[p].DiscStars[i];
        DiscGas += galaxies[p].DiscGas[i];
    }

    /* Hot-phase mass for the beta profile: whichever halo reservoir is in use plus the fountain/outflow gas */
    const double M_hot_total = galaxies[p].HotGas + galaxies[p].CGMgas + galaxies[p].FountainGas + galaxies[p].OutflowGas;

    double M_DM = galaxies[p].Mvir - (M_hot_total + galaxies[p].ColdGas + galaxies[p].StellarMass
                                      + galaxies[p].BlackHoleMass + galaxies[p].ICS);
    if(M_DM < 0.0) M_DM = 0.0;

    /* NFW concentration, Dutton & Maccio (2014) */
    const double z_snap = run_params->ZZ[halos[galaxies[p].HaloNr].SnapNum];
    double z = z_snap;
    if(z > 5.0) z = 5.0;
    const double a_conc = 0.520 + (0.905 - 0.520) * exp(-0.617 * pow(z, 1.21));
    const double b_conc = -0.101 + 0.026 * z;
    const double log_Mvir = log10(galaxies[p].Mvir * run_params->UnitMass_in_g / SOLAR_MASS) - 12.0;
    const double r_s = Rvir / pow(10.0, a_conc + b_conc * log_Mvir);
    galaxies[p].HaloScaleRadius = r_s;

    double rho_norm = M_DM / (log(1.0 + Rvir / r_s) - Rvir / (Rvir + r_s));
    if(rho_norm < 0.0) rho_norm = 0.0;

    const double total_stellar_gas = galaxies[p].StellarMass + galaxies[p].ColdGas;
    const double BTT = (total_stellar_gas > 0.0) ? galaxies[p].BulgeMass / total_stellar_gas : 0.0;

    if(galaxies[p].Mvir <= 0.0 || BTT >= 1.0) {
        /* No disc: flat rotation curve */
        for(int i = 0; i < N_BINS + 1; i++) {
            const double r = run_params->DiscBinEdge[i] / Vvir;
            galaxies[p].DiscRadii[i] = (r > current_Rvir) ? current_Rvir : r;
        }
        enforce_monotonic_radii(p, current_Rvir, galaxies);
        for(int i = 0; i < N_BINS + 1; i++) {
            galaxies[p].Potential[i] = darksage_nfw_potential(p, galaxies[p].DiscRadii[i], galaxies, run_params);
        }
        update_hot_energetics(p, z_snap, galaxies, run_params);
        return;
    }

    const double v_max = (galaxies[p].Vmax > Vvir) ? 2.0 * galaxies[p].Vmax : 2.0 * Vvir;
    const double c_beta = galaxies[p].c_beta;
    double a_stars = galaxies[p].StellarDiscScaleRadius;
    double a_gas = galaxies[p].GasDiscScaleRadius;
    if(a_stars <= 0.0) a_stars = galaxies[p].DiskScaleRadius;
    if(a_gas <= 0.0) a_gas = galaxies[p].DiskScaleRadius;

    const double r_min = (0.01 / 1000.0 > 0.001 * current_Rvir) ? 0.01 / 1000.0 : 0.001 * current_Rvir;

    const struct rotation_curve rc = {
        G, rho_norm, r_s,
        galaxies[p].BulgeMass, galaxies[p].BulgeRadius,
        DiscStars, a_stars, DiscGas, a_gas,
        M_hot_total, c_beta, Rvir
    };

    for(int i = 0; i < N_BINS + 1; i++) {
        const double j = run_params->DiscBinEdge[i];
        double r = j / Vvir;   /* flat-curve initial guess */

        /* Iterate j = r V_circ(r) */
        for(int iter = 0; iter < 20; iter++) {
            if(r <= 0.0) { r = j / Vvir; break; }

            double V_circ = sqrt(circular_velocity2(&rc, r));
            if(V_circ > v_max) V_circ = v_max;
            if(V_circ < 0.1 * Vvir) V_circ = 0.1 * Vvir;

            const double r_new = j / V_circ;
            const int converged = fabs(r_new - r) < 1.0e-6 * r;
            r = r_new;
            if(converged) break;
        }

        if(r > current_Rvir) r = current_Rvir;
        if(r < r_min && j > 0.0) r = r_min;
        galaxies[p].DiscRadii[i] = r;
    }

    enforce_monotonic_radii(p, current_Rvir, galaxies);

    for(int i = 0; i < N_BINS + 1; i++) {
        galaxies[p].Potential[i] = darksage_nfw_potential(p, galaxies[p].DiscRadii[i], galaxies, run_params);
    }
    update_hot_energetics(p, z_snap, galaxies, run_params);

    if(run_params->BulgeSizeOn == 4) {
        darksage_instability_bulge_size(p, &rc, galaxies, run_params);
        get_bulge_radius(p, galaxies, run_params);
    }
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

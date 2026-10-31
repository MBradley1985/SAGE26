/*
 * model_dust.c -- dust production, growth, destruction and sputtering (DustySAGEMode >= 1).
 *
 * Follows Dusty-SAGE (Triani et al. 2020; source model_dust.c, core_init.c):
 *   - Production: element yields of AGB stars (Karakas 2010), SNII (Woosley & Weaver 1995, or
 *     Nomoto et al. 2006) and SNIa (Iwamoto et al. 1999, or Seitenzahl et al. 2013) from the
 *     galaxy's star-formation history, delayed by stellar lifetimes (Raiteri et al. 1996) and
 *     weighted by the IMF (Arrigoni et al. 2010 eqs. 9-11); a fraction of the C, N, O, Mg, Si,
 *     S, Ca and Fe condenses into dust (Popping et al. 2017 eqs. 4-6).
 *   - Mode 1: SAGE26's metals are unchanged and dust is a passive part of them; production is
 *     capped by the metals SAGE26 put into the cold gas in the substep.
 *   - Mode 2: the delayed yields replace SAGE26's metal production and dust is taken out of
 *     MetalsColdGas, as in Dusty-SAGE (MetalYieldsOn = 1).
 * Known source bugs fixed here (SAGE26_investigation.md Sec. 5): the yield metallicity bin is
 * the one containing Z (the source's test never matched, so zero-metallicity yields were always
 * used); the lookup time is a double (the source truncated it to an int); the lognormal IMF
 * term is squared; the Nomoto metallicity grid reads 0.004 (an octal 0004 in the source).
 * The yields are made once per galaxy per substep: the source also called its production in
 * every merger burst, making that substep's metals twice.  Also changed from the source: the
 * snapshot times are physical Myr (the source left out 1/h); mode-2 grain growth is limited by
 * all the cold metals, gas phase + dust (Asano et al. 2013), not the gas phase alone; satellites'
 * hot dust is sputtered too; and a merged satellite's star-formation history joins the central's.
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_spline.h>

#include "core_allvars.h"
#include "macros.h"
#include "model_dust.h"
#include "model_misc.h"

/* Mass ranges of the channels [Msun] (Dusty-SAGE produce_metals_dust()) */
static const double AGB_LOW = 1.0, AGB_UP = 6.0, BINARY_LOW = 3.0, BINARY_UP = 16.0, SN_UP = 40.0;
static const double SNIA_GAMMA = 2.0;     /* binary mass-ratio distribution exponent (Arrigoni+10 eq. 10) */
static const double SNIA_MU_MAX = 0.5;    /* maximum secondary-to-binary mass ratio */
enum { N_INTEG = 20 };                    /* points of the source's integration grids */
/* Growth, destruction and sputtering constants (Asano et al. 2013; Popping et al. 2017; source values) */
static const double DUST_TACC0_MYR = 20.0;        /* grain-growth time at Z = DUST_ZREF */
static const double DUST_ZREF = 0.02;
static const double DUST_SN_MLOW = 8.0;           /* lowest SN progenitor mass [Msun] */
static const double DUST_MSWEPT_MSUN = 1535.0;    /* ISM swept clean per SN at Z = 0 */
static const double DUST_SN_ETA = 0.1;            /* destruction efficiency */
static const double DUST_TVIR_COEFF = 35.9;       /* T = 35.9 Vvir^2 [K (km/s)^-2] */
static const double DUST_SPUTTER_T0 = 2.0e6;      /* [K] */
static const double DUST_SPUTTER_OMEGA = 2.5;
static const double DUST_SPUTTER_RATE_CGS = 3.2e-18;   /* [cm^4 s^-1] */
static const double DUST_SPUTTER_TAU0_MYR = 170.0;
static const double DUST_GRAIN_CM = 1.0e-5;       /* initial grain radius, 0.1 um */
static const double DUST_GRAIN_MIN_FRAC = 1.0e-3; /* grains shrink no further than this (the source asserts a > 0) */
/* Condensation efficiencies (Popping et al. 2017; fixed in the source) */
static const double DELTA_AGB = 0.2, DELTA_SN = 0.2;
/* Cold gas needed for yields to stay in the disc, and to make dust [10^10 Msun/h] (source) */
static const double YIELD_COLD_MIN = 5.0e-2, DUST_COLD_MIN = 1.0e-2;
/* Yield metallicity grids (Karakas 2010 and Woosley & Weaver 1995; Nomoto et al. 2006) */
static const double Z_GRID_7[DUST_METALGRID] = {0.0, 1e-4, 4e-4, 4e-3, 8e-3, 0.02, 0.05};
static const double Z_GRID_NOMOTO[4] = {0.0, 0.001, 0.004, 0.02};

/* Read a whitespace table with one header line; returns rows read (Dusty-SAGE read_table()) */
static int read_table(const char *fname, const int ncols, double tbl[][DUST_TABLE_COLS])
{
    FILE *f = fopen(fname, "r");
    if(f == NULL) {
        fprintf(stderr, "Error: could not open the dust yield table %s\n", fname);
        ABORT(EXIT_FAILURE);
    }
    if(fscanf(f, "%*[^\n]") < 0) { /* header line */ }
    int i = 0;
    while(i < DUST_TABLE_ROWS) {
        int ok = 1;
        for(int j = 0; j < ncols; j++) {
            if(fscanf(f, " %lf", &tbl[i][j]) != 1) { ok = 0; break; }
        }
        if(!ok) break;
        if(fscanf(f, "%*[^\n]") < 0) { /* rest of line */ }
        i++;
    }
    fclose(f);
    return i;
}

/* Store one metallicity column of a table: rows whose Z is z_grid[j], in order (core_init.c) */
static int grid_rows(double tbl[][DUST_TABLE_COLS], const int rows, const double z, int *idx)
{
    int n = 0;
    for(int i = 0; i < rows; i++) if(fabs(tbl[i][0] - z) < 1e-9) idx[n++] = i;
    return n;
}

void dust_init(struct params *run_params)
{
    static double tbl[DUST_TABLE_ROWS][DUST_TABLE_COLS];
    int idx[DUST_TABLE_ROWS];
    char fname[MAX_STRING_LEN];

    /* AGB: Karakas (2010), C = C12 + C13P + C13, N = N14P + N14, O = O16 */
    snprintf(fname, MAX_STRING_LEN - 1, ROOT_DIR "/src/auxdata/yields/table2d.dat");
    int rows = read_table(fname, 13, tbl);
    for(int j = 0; j < DUST_METALGRID; j++) {
        const int n = grid_rows(tbl, rows, Z_GRID_7[j], idx);
        if(j == 0) run_params->DustCountAGB = n;
        for(int k = 0; k < n; k++) {
            const double *r = tbl[idx[k]];
            run_params->DustMassAGB[k] = r[1];
            run_params->DustQAGB[0][k][j] = r[6] + r[7] + r[11];
            run_params->DustQAGB[1][k][j] = r[8] + r[12];
            run_params->DustQAGB[2][k][j] = r[9];
        }
    }

    /* SNII: element order C, O, Mg, Si, S, Ca, Fe */
    if(run_params->DustSNIIYields == 0) {   /* Woosley & Weaver (1995), C = C12 + C13s */
        snprintf(fname, MAX_STRING_LEN - 1, ROOT_DIR "/src/auxdata/yields/table4a.dat");
        rows = read_table(fname, 17, tbl);
        for(int j = 0; j < DUST_METALGRID; j++) {
            const int n = grid_rows(tbl, rows, Z_GRID_7[j], idx);
            if(j == 0) run_params->DustCountSN = n;
            for(int k = 0; k < n; k++) {
                const double *r = tbl[idx[k]];
                run_params->DustMassSN[k] = r[1];
                const double q[7] = {r[6] + r[15], r[7], r[9], r[10], r[11], r[12], r[13]};
                for(int e = 0; e < 7; e++) run_params->DustQSN[e][k][j] = q[e];
            }
        }
    } else {                                 /* Nomoto et al. (2006) */
        snprintf(fname, MAX_STRING_LEN - 1, ROOT_DIR "/src/auxdata/yields/Nomoto.dat");
        rows = read_table(fname, 13, tbl);
        for(int j = 0; j < 4; j++) {
            const int n = grid_rows(tbl, rows, Z_GRID_NOMOTO[j], idx);
            if(j == 0) run_params->DustCountSN = n;
            for(int k = 0; k < n; k++) {
                const double *r = tbl[idx[k]];
                run_params->DustMassSN[k] = r[1];
                const double q[7] = {r[2] + r[11], r[3], r[5], r[6], r[7], r[8], r[9]};
                for(int e = 0; e < 7; e++) run_params->DustQSN[e][k][j] = q[e];
            }
        }
    }

    /* SNIa: Cr, Fe, Ni per event */
    if(run_params->DustSNIaYields == 0) {   /* Iwamoto et al. (1999) */
        run_params->DustQSNIa[0] = 0.0168; run_params->DustQSNIa[1] = 0.587; run_params->DustQSNIa[2] = 0.0314;
    } else {                                 /* Seitenzahl et al. (2013) */
        run_params->DustQSNIa[0] = 0.00857; run_params->DustQSNIa[1] = 0.622; run_params->DustQSNIa[2] = 0.069;
    }

    /* Time since the first snapshot of each snapshot [Myr], physical like the stellar lifetimes
     * below (code time is s/h; the source left out the 1/h) */
    for(int i = 0; i < run_params->Snaplistlen && i < ABSOLUTEMAXSNAPS; i++) {
        run_params->DustTime[i] = (run_params->Age[0] - run_params->Age[i]) * run_params->UnitTime_in_s
                                / SEC_PER_MEGAYEAR / run_params->Hubble_h;
    }
}

/* Linear-interpolation integral of y(x) between lo and hi (Dusty-SAGE integrate_arr()) */
static double integrate(const double *x, const double *y, const int n, const double lo, const double hi)
{
    if(n < 2 || hi <= lo) return 0.0;
    gsl_interp_accel *acc = gsl_interp_accel_alloc();
    gsl_spline *spl = gsl_spline_alloc(gsl_interp_linear, n);
    gsl_spline_init(spl, x, y, n);
    const double a = (lo < x[0]) ? x[0] : lo, b = (hi > x[n-1]) ? x[n-1] : hi;
    const double q = (b > a) ? gsl_spline_eval_integ(spl, a, b, acc) : 0.0;
    gsl_spline_free(spl);
    gsl_interp_accel_free(acc);
    return q;
}

/* IMF of Arrigoni et al. (2010) eq. 11: lognormal below 1 Msun (squared, a source bug fixed),
 * power law above */
static double imf(const double m)
{
    const double A = 0.9098, B = 0.2539, x = 1.3, sigma = 0.69, mc = 0.079;
    if(m < 1.0) {
        const double d = log10(m) - log10(mc);
        return A * exp(-d * d / (2.0 * sigma * sigma));
    }
    return B * pow(m, -x);
}

/* Stellar lifetime [Myr] of a star of mass m at Z = 0.02 (Raiteri et al. 1996 eq. 3) */
static double lifetime(const double m)
{
    const double lz = log10(0.02);
    const double a0 = 10.13 + 0.07547 * lz - 0.008084 * lz * lz;
    const double a1 = -4.424 - 0.7939 * lz - 0.1187 * lz * lz;
    const double a2 = 1.262 + 0.3385 * lz + 0.05417 * lz * lz;
    const double lm = log10(m);
    return pow(10.0, a0 + a1 * lm + a2 * lm * lm) / 1.0e6;
}

/* The galaxy's SFR when the stars of lifetime tau were born (0 before the first snapshot) */
static double sfr_at(const struct GALAXY *g, const double tau, const struct params *run_params)
{
    const int now = g->SnapNum;
    const double t = run_params->DustTime[now] - tau;
    if(t < run_params->DustTime[0]) return 0.0;
    int i = 0;
    while(i < now && run_params->DustTime[i+1] < t) i++;
    if(i >= now) return g->DustSfh[now];
    const double f = (t - run_params->DustTime[i]) / (run_params->DustTime[i+1] - run_params->DustTime[i]);
    return g->DustSfh[i] + f * (g->DustSfh[i+1] - g->DustSfh[i]);
}

/* Yield metallicity bin: Z_grid[j] <= Z < Z_grid[j+1] (the source's test never matched) */
static int z_bin(const double Z, const double *grid, const int n)
{
    int j = 0;
    for(int i = 0; i < n; i++) if(grid[i] <= Z) j = i;
    return j;
}

/*
 * element_rates -- production rates [10^10 Msun/h per code time, from the SFH in the same units]
 * of C, N, O (AGB), C, O, Mg, Si, S, Ca, Fe (SNII) and Cr, Fe, Ni (SNIa) for galaxy g, as
 * Dusty-SAGE produce_metals_dust() (Arrigoni et al. 2010 eq. 9): each channel integrates the
 * yield of stars of mass m, the IMF and the SFR one lifetime ago over m; a fraction
 * BinaryFraction of stars between 3 and 16 Msun are SNIa binaries.
 */
static void element_rates(const struct GALAXY *g, const double Z, double agb[3], double sn[7], double snia[3],
                          const struct params *run_params)
{
    const double A = run_params->DustBinaryFraction;

    /* SNIa: binaries of total mass mbin and mass ratio mu (source grids) */
    double mu[N_INTEG], mbin[N_INTEG], ymu[3][N_INTEG], yphi[3][N_INTEG];
    for(int i = 0; i < N_INTEG; i++) {
        mbin[i] = BINARY_LOW + (BINARY_UP - BINARY_LOW) / (N_INTEG - i);
        mu[i] = SNIA_MU_MAX / (N_INTEG - i);
        const double fmu = pow(2.0, 1.0 + SNIA_GAMMA) * (1.0 + SNIA_GAMMA) * pow(mu[i], SNIA_GAMMA);
        const double sfr = sfr_at(g, lifetime(mu[i] * mbin[i]), run_params);
        for(int e = 0; e < 3; e++) ymu[e][i] = fmu * sfr * run_params->DustQSNIa[e];
    }
    for(int e = 0; e < 3; e++) {
        const double y = integrate(mu, ymu[e], N_INTEG, SNIA_MU_MAX / N_INTEG, SNIA_MU_MAX);
        for(int i = 0; i < N_INTEG; i++) yphi[e][i] = y * imf(mbin[i]);
        snia[e] = A * integrate(mbin, yphi[e], N_INTEG, mbin[0], BINARY_UP);
    }

    /* AGB */
    const int nagb = run_params->DustCountAGB;
    const int jagb = z_bin(Z, Z_GRID_7, DUST_METALGRID);
    double magb[DUST_MAXYIELDS], yagb[3][DUST_MAXYIELDS];
    for(int k = 0; k < nagb; k++) {
        magb[k] = run_params->DustMassAGB[k];
        const double w = imf(magb[k]) * sfr_at(g, lifetime(magb[k]), run_params);
        for(int e = 0; e < 3; e++) yagb[e][k] = run_params->DustQAGB[e][k][jagb] * w;
    }
    for(int e = 0; e < 3; e++) {
        agb[e] = (1.0 - A) * integrate(magb, yagb[e], nagb, BINARY_LOW, AGB_UP)
               + integrate(magb, yagb[e], nagb, AGB_LOW, BINARY_LOW);
    }

    /* SNII */
    const int nsn = run_params->DustCountSN;
    const int jsn = (run_params->DustSNIIYields == 0) ? z_bin(Z, Z_GRID_7, DUST_METALGRID) : z_bin(Z, Z_GRID_NOMOTO, 4);
    double msn[DUST_MAXYIELDS], ysn[7][DUST_MAXYIELDS];
    for(int k = 0; k < nsn; k++) {
        msn[k] = run_params->DustMassSN[k];
        const double w = imf(msn[k]) * sfr_at(g, lifetime(msn[k]), run_params);
        for(int e = 0; e < 7; e++) ysn[e][k] = run_params->DustQSN[e][k][jsn] * w;
    }
    const double low_sn = (nsn > 0) ? msn[0] : 0.0;
    for(int e = 0; e < 7; e++) {
        sn[e] = (1.0 - A) * integrate(msn, ysn[e], nsn, low_sn, BINARY_UP) + integrate(msn, ysn[e], nsn, BINARY_UP, SN_UP);
    }
}

/*
 * dust_production -- one substep of metal and dust production for galaxy p (Dusty-SAGE
 * produce_metals_dust()).  Mode 2 adds the yields to MetalsColdGas (to the central's hot phase
 * when there is too little cold gas) and turns the condensed part into ColdDust; mode 1 only
 * makes dust, capped by the metals SAGE26 put into the cold gas this substep (DustMetalsStep).
 * Dust: AGB stars condense delta (C - 0.75 O) if C/O > 1, else delta (C + N + O); SNII
 * condense delta (C + O) + 16 delta (Mg/24 + Si/28 + S/32 + Ca/40 + Fe/56) (Popping et al.
 * 2017 eqs. 4-6); SNIa dust is off, as in the source.
 */
static void dust_production(const int p, const int centralgal, const double dt, struct GALAXY *galaxies,
                            const struct params *run_params)
{
    struct GALAXY *g = &galaxies[p];
    const double Z = get_metallicity(g->ColdGas, g->MetalsColdGas);
    double agb[3], sn[7], snia[3];
    element_rates(g, Z, agb, sn, snia, run_params);

    if(run_params->DustySAGEMode == 2) {
        double yield = 0.0;
        for(int e = 0; e < 3; e++) yield += agb[e] + snia[e];
        for(int e = 0; e < 7; e++) yield += sn[e];
        if(g->ColdGas > YIELD_COLD_MIN) g->MetalsColdGas += yield * dt;
        else add_metals_to_hot_reservoir(&galaxies[centralgal], run_params, yield * dt);
    }
    if(g->ColdGas <= DUST_COLD_MIN) return;

    double dust = 0.0;
    if(agb[2] > 0.0 && agb[0] / agb[2] > 1.0) dust += DELTA_AGB * (agb[0] - 0.75 * agb[2]);
    else dust += DELTA_AGB * (agb[0] + agb[1] + agb[2]);
    dust += DELTA_SN * (sn[0] + sn[1]);
    dust += 16.0 * DELTA_SN * (sn[2] / 24.0 + sn[3] / 28.0 + sn[4] / 32.0 + sn[5] / 40.0 + sn[6] / 56.0);
    dust *= dt;
    if(dust <= 0.0) return;

    if(run_params->DustySAGEMode == 1) {
        /* passive: no more than SAGE26 made, nor more than the cold gas's metals not yet dust */
        if(dust > g->DustMetalsStep) dust = g->DustMetalsStep;
        if(dust > g->MetalsColdGas - g->ColdDust) dust = g->MetalsColdGas - g->ColdDust;
        if(dust > 0.0) g->ColdDust += (float)dust;
    } else {
        if(dust > g->MetalsColdGas) dust = g->MetalsColdGas;
        g->ColdDust += (float)dust;
        g->MetalsColdGas -= (float)dust;
    }
    if(dust > 0.0) g->DustFormationRate += (float)dust;   /* mass, a rate after evolve_galaxies() */
}

/* Grain growth in the ISM (Asano et al. 2013 eq. 20; Dusty-SAGE accrete_dust()):
 *   dD/dt = (1 - D / Z_metals) f_H2 D / tau_acc,  tau_acc = 20 Myr x 0.02 / Z,
 * f_H2 = H2gas / ColdGas, Z = Z_metals / ColdGas, Z_metals all the cold metals as in Asano:
 * MetalsColdGas in mode 1, MetalsColdGas + ColdDust in mode 2 (the source used MetalsColdGas in
 * mode 2 too, which stops growth at dust = gas-phase metals, D/Z = 0.5).  Growth draws on the
 * gas-phase metals, Z_metals - D. */
static void dust_growth(const int p, const double dt, struct GALAXY *galaxies, const struct params *run_params)
{
    struct GALAXY *g = &galaxies[p];
    if(g->ColdGas <= 0.0f || g->ColdDust <= 0.0f) return;
    const double metals = (run_params->DustySAGEMode == 2) ? (double)g->MetalsColdGas + g->ColdDust : g->MetalsColdGas;
    const double gas_phase = metals - g->ColdDust;
    if(gas_phase <= 0.0) return;
    const double Z = metals / g->ColdGas;
    const double tacc = DUST_TACC0_MYR * SEC_PER_MEGAYEAR / run_params->UnitTime_in_s * DUST_ZREF / Z;
    const double f_h2 = g->H2gas / g->ColdGas;
    double grow = (1.0 - g->ColdDust / metals) * f_h2 * g->ColdDust / tacc * dt;
    if(grow <= 0.0) return;
    if(grow > gas_phase) grow = gas_phase;
    g->ColdDust += (float)grow;
    if(run_params->DustySAGEMode == 2) g->MetalsColdGas -= (float)grow;
    g->DustGrowthRate += (float)grow;
}

/* Destruction by supernova shocks (Asano et al. 2013 eqs. 12, 14; Dusty-SAGE destruct_dust()):
 * the SN rate R = stars / dt / m_*, m_* = int m phi dm / int_8^40 phi dm on the source's mass
 * grid; each SN sweeps m_swept = 1535 (Z/0.02 + 0.039)^-0.289 Msun clean with efficiency 0.1,
 * so tau_SN = ColdGas / (0.1 m_swept R) and dD/dt = D / tau_SN.  The destroyed dust returns
 * to the gas-phase metals (mode 2). */
static void dust_destruction(const int p, const double dt, struct GALAXY *galaxies, const struct params *run_params)
{
    struct GALAXY *g = &galaxies[p];
    const double stars = g->DustStarsStep;
    if(stars <= 0.0 || g->ColdGas <= 0.0f || g->ColdDust <= 0.0f) return;
    if(run_params->DustTime[g->SnapNum] <= lifetime(SN_UP)) return;   /* no SN yet (source) */

    double mass[N_INTEG], phi[N_INTEG], mphi[N_INTEG];
    for(int i = 0; i < N_INTEG; i++) {
        mass[i] = 1.0 + (SN_UP - 1.0) / (N_INTEG - i);
        phi[i] = imf(mass[i]);
        mphi[i] = mass[i] * phi[i];
    }
    const double mstar = integrate(mass, mphi, N_INTEG, mass[0], SN_UP) / integrate(mass, phi, N_INTEG, DUST_SN_MLOW, SN_UP);
    if(!(mstar > 0.0)) return;
    const double Z = get_metallicity(g->ColdGas, g->MetalsColdGas);
    const double m_swept = DUST_MSWEPT_MSUN * pow(Z / DUST_ZREF + 0.039, -0.289) * run_params->Hubble_h / 1.0e10;
    const double rsn = stars / dt / mstar * run_params->UnitMass_in_g / run_params->UnitTime_in_s * SEC_PER_YEAR / SOLAR_MASS;
    if(!(rsn > 0.0)) return;
    const double tsn = g->ColdGas / (DUST_SN_ETA * m_swept * rsn) / (run_params->UnitTime_in_s / SEC_PER_YEAR);
    double lost = g->ColdDust / tsn * dt;
    if(lost > g->ColdDust) lost = g->ColdDust;
    g->ColdDust -= (float)lost;
    if(run_params->DustySAGEMode == 2) g->MetalsColdGas += (float)lost;
    g->DustDestructionRate += (float)lost;
}

/* Thermal sputtering of dust in hot gas of density rho at the virial temperature (Popping et
 * al. 2017 Sec. 3.5; Dusty-SAGE dust_thermal_sputtering()).  Grains of a = 0.1 um shrink at
 * da/dt = rho x, x = -3.2e-18 / m_p / ((T0/T)^2.5 + 1) cm^4 g^-1 s^-1, over the substep, and
 * the dust is lost on tau = 0.17 Gyr (a / 0.1 um)(1e-27 g cm^-3 / rho)((T0/T)^2.5 + 1), at
 * dM/dt = M / (tau / 3).  Fixes three source bugs that switched it off: da/dt is rho x (the
 * source divided), 4/3 is not integer division, and rho is in units of 1e-27 g cm^-3.
 * Returns the dust sputtered, which returns to the metals (mode 2). */
static double sputter(const double gas, float *dust, const double Rvir, const double Vvir, const double dt,
                      const struct params *run_params)
{
    if(*dust <= 0.0f || gas <= 0.0 || Vvir <= 0.0 || Rvir <= 0.0) return 0.0;
    const double temp = DUST_TVIR_COEFF * Vvir * Vvir;
    const double tfac = pow(DUST_SPUTTER_T0 / temp, DUST_SPUTTER_OMEGA) + 1.0;
    const double a0 = DUST_GRAIN_CM / run_params->UnitLength_in_cm;
    const double x = -DUST_SPUTTER_RATE_CGS / PROTONMASS / tfac
                     / (run_params->UnitLength_in_cm / run_params->UnitDensity_in_cgs / run_params->UnitTime_in_s);
    const double rho = gas / (4.0 / 3.0 * M_PI * Rvir * Rvir * Rvir);
    double a = a0 + rho * x * dt;
    if(a < DUST_GRAIN_MIN_FRAC * a0) a = DUST_GRAIN_MIN_FRAC * a0;
    const double rho_27 = rho * run_params->UnitDensity_in_cgs / 1.0e-27;
    const double tau = DUST_SPUTTER_TAU0_MYR * SEC_PER_MEGAYEAR / run_params->UnitTime_in_s * (a / a0) / rho_27 * tfac;
    double lost = *dust / (tau / 3.0) * dt;
    if(lost > *dust) lost = *dust;
    *dust -= (float)lost;
    return lost;
}

/* Sputtering of the hot, CGM and ejected dust of galaxy p, central or satellite (the source
 * sputtered only the central's; SAGE26 satellites keep hot gas) */
static void dust_sputtering(const int p, const double dt, struct GALAXY *galaxies, const struct params *run_params)
{
    struct GALAXY *g = &galaxies[p];
    const int metals = (run_params->DustySAGEMode == 2);
    double d = sputter(g->HotGas, &g->HotDust, g->Rvir, g->Vvir, dt, run_params);
    if(metals) g->MetalsHotGas += (float)d;
    d = sputter(g->CGMgas, &g->CGMDust, g->Rvir, g->Vvir, dt, run_params);
    if(metals) g->MetalsCGMgas += (float)d;
    d = sputter(g->EjectedMass, &g->EjectedDust, g->Rvir, g->Vvir, dt, run_params);
    if(metals) g->MetalsEjectedMass += (float)d;
}

void dust_evolve(const int p, const int centralgal, const double dt, struct GALAXY *galaxies, const struct params *run_params)
{
    struct GALAXY *g = &galaxies[p];
    if(dt <= 0.0) return;

    /* this substep's star formation joins the history the yields read (snapshot-mean SFR) */
    const int snap = g->SnapNum;
    if(snap >= 0 && snap < ABSOLUTEMAXSNAPS) g->DustSfh[snap] += (float)(g->DustStarsStep / (dt * sfr_rate_divisor(g->SubstepsUsed)));

    dust_production(p, centralgal, dt, galaxies, run_params);
    dust_growth(p, dt, galaxies, run_params);
    dust_destruction(p, dt, galaxies, run_params);
    dust_sputtering(p, dt, galaxies, run_params);

    g->DustStarsStep = 0.0f;
    g->DustMetalsStep = 0.0f;
}

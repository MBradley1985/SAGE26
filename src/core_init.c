#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>

#ifdef GSL_FOUND
#include <gsl/gsl_integration.h>
#endif

#include "core_allvars.h"
#include "core_init.h"
#include "core_mymalloc.h"
#include "core_cool_func.h"
#include "model_misc.h"


/* These functions do not need to be exposed externally */
double integrand_time_to_present(const double a, void *param);
void set_units(struct params *run_params);
void read_snap_list(struct params *run_params);
double time_to_present(const double z, struct params *run_params);

/* Yield table reader (from dusty-sage, Triani et al. 2020) */
struct yield_table {
    double tbl[1000][17];   /* up to 1000 rows x 17 columns */
    int nr;                 /* number of rows read */
};

static struct yield_table read_table(const char *fname, const int ncols)
{
    struct yield_table dt;
    memset(&dt, 0, sizeof(dt));

    FILE *file = fopen(fname, "r");
    if(file == NULL) {
        fprintf(stderr, "Error: cannot open yield table '%s'\n", fname);
        ABORT(1);
    }

    /* skip header line */
    char buf[4096];
    if(fgets(buf, sizeof(buf), file) == NULL) {
        fclose(file);
        dt.nr = 0;
        return dt;
    }

    int i = 0;
    while(i < 1000) {
        int nread = 0;
        for(int j = 0; j < ncols; j++) {
            if(fscanf(file, " %lf", &dt.tbl[i][j]) == 1)
                nread++;
        }
        /* If we couldn't read all columns, we're done */
        if(nread != ncols) break;

        /* skip rest of line */
        int c;
        while((c = fgetc(file)) != '\n' && c != EOF) {}

        i++;
    }
    fclose(file);
    dt.nr = i;
    return dt;
}

static void read_metal_yield(struct params *run_params)
{
    char fname[MAX_STRING_LEN];
    int i, j, rows;

    /* ---- AGB yields (Karakas 2010) ---- */
    if(run_params->AGBYields == 0) {
        double Z_std[7] = {0.0, 1e-4, 4e-4, 4e-3, 8e-3, 0.02, 0.05};
        snprintf(fname, MAX_STRING_LEN, "%s/src/auxdata/yields/table2d.dat", ROOT_DIR);

        struct yield_table data = read_table(fname, 13);
        rows = data.nr;

        /* Count mass bins for the first metallicity */
        int count = 0;
        for(i = 0; i < rows; i++) {
            if(data.tbl[i][0] == Z_std[0]) count++;
        }
        run_params->countagb = count;

        for(j = 0; j < 7; j++) {
            int index = 0;
            for(i = 0; i < rows; i++) {
                if(data.tbl[i][0] == Z_std[j]) {
                    run_params->magb[index] = data.tbl[i][1];                     /* mass */
                    run_params->qCagb[index][j] = data.tbl[i][6] + data.tbl[i][7] + data.tbl[i][11]; /* C */
                    run_params->qNagb[index][j] = data.tbl[i][8] + data.tbl[i][12];                  /* N */
                    run_params->qOagb[index][j] = data.tbl[i][9];                                    /* O */
                    run_params->Qagb[index][j] = run_params->qCagb[index][j] +
                                                 run_params->qNagb[index][j] +
                                                 run_params->qOagb[index][j];
                    index++;
                }
            }
        }
    }

    /* ---- SNII yields ---- */
    if(run_params->SNIIYields == 0) {
        /* Woosley & Weaver 1995 */
        double Z_std[7] = {0.0, 1e-4, 4e-4, 4e-3, 8e-3, 0.02, 0.05};
        snprintf(fname, MAX_STRING_LEN, "%s/src/auxdata/yields/table4a.dat", ROOT_DIR);

        struct yield_table data = read_table(fname, 17);
        rows = data.nr;

        int count = 0;
        for(i = 0; i < rows; i++) {
            if(data.tbl[i][0] == Z_std[0]) count++;
        }
        run_params->countsn = count;

        for(j = 0; j < 7; j++) {
            int index = 0;
            for(i = 0; i < rows; i++) {
                if(data.tbl[i][0] == Z_std[j]) {
                    run_params->msn[index] = data.tbl[i][1];                       /* mass */
                    run_params->qCsn[index][j] = data.tbl[i][6] + data.tbl[i][15]; /* C */
                    run_params->qOsn[index][j] = data.tbl[i][7];                   /* O */
                    run_params->qMgsn[index][j] = data.tbl[i][9];                  /* Mg */
                    run_params->qSisn[index][j] = data.tbl[i][10];                 /* Si */
                    run_params->qSsn[index][j] = data.tbl[i][11];                  /* S */
                    run_params->qCasn[index][j] = data.tbl[i][12];                 /* Ca */
                    run_params->qFesn[index][j] = data.tbl[i][13];                 /* Fe */
                    run_params->Qsn[index][j] = run_params->qCsn[index][j] +
                                                run_params->qOsn[index][j] +
                                                run_params->qMgsn[index][j] +
                                                run_params->qSisn[index][j] +
                                                run_params->qSsn[index][j] +
                                                run_params->qCasn[index][j] +
                                                run_params->qFesn[index][j];
                    index++;
                }
            }
        }
    } else if(run_params->SNIIYields == 1) {
        /* Nomoto et al. 2006 */
        double Z_std[4] = {0.0, 0.001, 0.004, 0.02};
        snprintf(fname, MAX_STRING_LEN, "%s/src/auxdata/yields/Nomoto.dat", ROOT_DIR);

        struct yield_table data = read_table(fname, 13);
        rows = data.nr;

        int count = 0;
        for(i = 0; i < rows; i++) {
            if(data.tbl[i][0] == Z_std[0]) count++;
        }
        run_params->countsn = count;

        for(j = 0; j < 4; j++) {
            int index = 0;
            for(i = 0; i < rows; i++) {
                if(data.tbl[i][0] == Z_std[j]) {
                    run_params->msn[index] = data.tbl[i][1];
                    run_params->qCsn[index][j] = data.tbl[i][2] + data.tbl[i][11]; /* C */
                    run_params->qOsn[index][j] = data.tbl[i][3];                   /* O */
                    run_params->qMgsn[index][j] = data.tbl[i][5];                  /* Mg */
                    run_params->qSisn[index][j] = data.tbl[i][6];                  /* Si */
                    run_params->qSsn[index][j] = data.tbl[i][7];                   /* S */
                    run_params->qCasn[index][j] = data.tbl[i][8];                  /* Ca */
                    run_params->qFesn[index][j] = data.tbl[i][9];                  /* Fe */
                    run_params->Qsn[index][j] = run_params->qCsn[index][j] +
                                                run_params->qOsn[index][j] +
                                                run_params->qMgsn[index][j] +
                                                run_params->qSisn[index][j] +
                                                run_params->qSsn[index][j] +
                                                run_params->qCasn[index][j] +
                                                run_params->qFesn[index][j];
                    index++;
                }
            }
        }
    }

    /* ---- SNIa yields (scalar constants) ---- */
    if(run_params->SNIaYields == 0) {
        /* Iwamoto 1999 */
        run_params->qCrsnia = 0.0168;
        run_params->qFesnia = 0.587;
        run_params->qNisnia = 0.0314;
    } else if(run_params->SNIaYields == 1) {
        /* Seitenzahl et al. 2013 */
        run_params->qCrsnia = 0.00857;
        run_params->qFesnia = 0.622;
        run_params->qNisnia = 0.069;
    }
}

void init(struct params *run_params)
{
#ifdef VERBOSE
    const int ThisTask = run_params->ThisTask;
#endif

    run_params->Age = mymalloc(ABSOLUTEMAXSNAPS*sizeof(run_params->Age[0]));

    set_units(run_params);

    read_snap_list(run_params);

    /* Store lookback time from z=1000 for delayed enrichment calculations */
    run_params->Age_at_z1000 = time_to_present(1000.0, run_params);

    for(int i = 0; i < run_params->Snaplistlen; i++) {
        run_params->ZZ[i] = 1 / run_params->AA[i] - 1;
        run_params->Age[i] = time_to_present(run_params->ZZ[i], run_params);
    }

    /* Compute age of universe at each snapshot (in Myr) for delayed enrichment */
    for(int i = 0; i < run_params->Snaplistlen; i++) {
        run_params->lbtime[i] = (run_params->Age_at_z1000 - run_params->Age[i])
                                * run_params->UnitTime_in_s / SEC_PER_MEGAYEAR;
    }

    run_params->a0 = 1.0 / (1.0 + run_params->Reionization_z0);
    run_params->ar = 1.0 / (1.0 + run_params->Reionization_zr);

    read_cooling_functions();

    /* Read stellar yield tables if element-by-element dust production is on */
    if(run_params->MetalYieldsOn == 1) {
        read_metal_yield(run_params);

        /* Pre-compute IMF and stellar lifetimes for yield integration (big speedup) */
        for(int i = 0; i < run_params->countagb; i++) {
            run_params->phi_agb[i] = compute_imf(run_params->magb[i]);
            run_params->taum_agb[i] = compute_taum(run_params->magb[i])
                                      * SEC_PER_MEGAYEAR / run_params->UnitTime_in_s;
        }
        for(int i = 0; i < run_params->countsn; i++) {
            run_params->phi_sn[i] = compute_imf(run_params->msn[i]);
            run_params->taum_sn[i] = compute_taum(run_params->msn[i])
                                     * SEC_PER_MEGAYEAR / run_params->UnitTime_in_s;
        }

        /* Pre-compute SNIa binary mass grid (fixed values from Arrigoni+2010) */
        const int count = 20;
        const double gamma_bin = 2.0;
        const double low_binary = 3.0, up_binary = 16.0;
        const double max_mu = 0.5;
        for(int i = 0; i < count; i++) {
            run_params->mbin_snia[i] = low_binary + ((up_binary - low_binary) / (double)(count - i));
            run_params->mu_snia[i] = max_mu / (double)(count - i);
            run_params->fmu_snia[i] = pow(2.0, 1.0 + gamma_bin) * (1.0 + gamma_bin)
                                      * pow(run_params->mu_snia[i], gamma_bin);
            run_params->phi_snia[i] = compute_imf(run_params->mbin_snia[i]);
            run_params->taum_snia[i] = compute_taum(run_params->mu_snia[i] * run_params->mbin_snia[i])
                                       * SEC_PER_MEGAYEAR / run_params->UnitTime_in_s;
        }

        /* Pre-compute SN destruction mass grid (8-40 Msun, 20 bins) */
        const double m_low = 8.0, m_up = 40.0;
        for(int i = 0; i < 20; i++) {
            run_params->mass_destruct[i] = m_low + (m_up - m_low) * (double)i / 19.0;
            run_params->phi_destruct[i] = compute_imf(run_params->mass_destruct[i]);
            run_params->taum_destruct[i] = compute_taum(run_params->mass_destruct[i])
                                           * SEC_PER_MEGAYEAR / run_params->UnitTime_in_s;
        }

#ifdef VERBOSE
        if(ThisTask == 0) {
            fprintf(stdout, "metal yield tables read (AGB: %d bins, SNII: %d bins)\n",
                    run_params->countagb, run_params->countsn);
        }
#endif
    }

    /* Initialize DarkMode: specific angular momentum bin edges */
    if(run_params->DarkSAGEOn == 1) {
        /* Set defaults if not specified */
        if(run_params->FirstBin <= 0) run_params->FirstBin = 100.0;   /* kpc km/s */
        if(run_params->ExponentBin <= 0) run_params->ExponentBin = 1.4;

        /* Convert FirstBin from kpc km/s to internal units (Mpc/h * km/s / h = Mpc km/s / h^2) */
        /* Factor: (kpc -> Mpc/h) / (km/s -> code velocity) */
        double j_unit = (CM_PER_MPC / 1e3 / run_params->UnitLength_in_cm)
                      / (1e5 / run_params->UnitVelocity_in_cm_per_s);

        run_params->DiscBinEdge[0] = 0.0;
        for(int i = 1; i < N_BINS + 1; i++) {
            run_params->DiscBinEdge[i] = run_params->FirstBin * j_unit * pow(run_params->ExponentBin, i - 1);
        }

#ifdef VERBOSE
        if(ThisTask == 0) {
            fprintf(stdout, "DarkMode enabled: %d radial bins, FirstBin=%.1f kpc*km/s, ExponentBin=%.2f\n",
                    N_BINS, run_params->FirstBin, run_params->ExponentBin);
        }
#endif
    }

#ifdef VERBOSE
    if(ThisTask == 0) {
        fprintf(stdout, "cooling functions read\n\n");
    }
#endif
}



void set_units(struct params *run_params)
{

    run_params->UnitTime_in_s = run_params->UnitLength_in_cm / run_params->UnitVelocity_in_cm_per_s;
    run_params->UnitTime_in_Megayears = run_params->UnitTime_in_s / SEC_PER_MEGAYEAR;
    run_params->G = GRAVITY / CUBE(run_params->UnitLength_in_cm) * run_params->UnitMass_in_g * SQR(run_params->UnitTime_in_s);
    run_params->UnitDensity_in_cgs = run_params->UnitMass_in_g / CUBE(run_params->UnitLength_in_cm);
    run_params->UnitPressure_in_cgs = run_params->UnitMass_in_g / run_params->UnitLength_in_cm / SQR(run_params->UnitTime_in_s);
    run_params->UnitCoolingRate_in_cgs = run_params->UnitPressure_in_cgs / run_params->UnitTime_in_s;
    run_params->UnitEnergy_in_cgs = run_params->UnitMass_in_g * SQR(run_params->UnitLength_in_cm) / SQR(run_params->UnitTime_in_s);

    run_params->EnergySNcode = run_params->EnergySN / run_params->UnitEnergy_in_cgs * run_params->Hubble_h;
    run_params->EtaSNcode = run_params->EtaSN * (run_params->UnitMass_in_g / SOLAR_MASS) / run_params->Hubble_h;

    // convert some physical input parameters to internal units
    run_params->Hubble = HUBBLE * run_params->UnitTime_in_s;

    // compute a few quantitites
    run_params->RhoCrit = 3.0 * run_params->Hubble * run_params->Hubble / (8 * M_PI * run_params->G);
}



void read_snap_list(struct params *run_params)
{
#ifdef VERBOSE
    const int ThisTask = run_params->ThisTask;
#endif

    char fname[MAX_STRING_LEN+1];

    snprintf(fname, MAX_STRING_LEN, "%s", run_params->FileWithSnapList);
    FILE *fd = fopen(fname, "r");
    if(fd == NULL) {
        fprintf(stderr, "can't read output list in file '%s'\n", fname);
        ABORT(0);
    }

    run_params->Snaplistlen = 0;
    do {
        if(fscanf(fd, " %lg ", &(run_params->AA[run_params->Snaplistlen])) == 1) {
            run_params->Snaplistlen++;
        } else {
            break;
        }
    } while(run_params->Snaplistlen < run_params->SimMaxSnaps);
    fclose(fd);

#ifdef VERBOSE
    if(ThisTask == 0) {
        fprintf(stdout, "found %d defined times in snaplist\n", run_params->Snaplistlen);
    }
#endif
}

double time_to_present(const double z, struct params *run_params)
{
    const double end_limit = 1.0;
    const double start_limit = 1.0/(1 + z);
    double result;
#ifdef GSL_FOUND
#define WORKSIZE 1000
    gsl_function F;
    gsl_integration_workspace *workspace;
    double abserr;

    workspace = gsl_integration_workspace_alloc(WORKSIZE);
    F.function = &integrand_time_to_present;
    F.params = run_params;

    gsl_integration_qag(&F, start_limit, end_limit, 1.0 / run_params->Hubble,
                        1.0e-9, WORKSIZE, GSL_INTEG_GAUSS21, workspace, &result, &abserr);

    gsl_integration_workspace_free(workspace);

#undef WORKSIZE
#else
    /* Do not have GSL - let's integrate numerically ourselves */
    const double step  = 1e-7;
    const int64_t nsteps = (end_limit - start_limit)/step;
    result = 0.0;
    const double y0 = integrand_time_to_present(start_limit + 0*step, run_params);
    const double yn = integrand_time_to_present(start_limit + nsteps*step, run_params);
    for(int64_t i=1; i<nsteps; i++) {
        result  += integrand_time_to_present(start_limit + i*step, run_params);
    }

    result = (step*0.5)*(y0 + yn + 2.0*result);
#endif

    /* convert into Myrs/h (I think -> MS 23/6/2018) */
    const double time = 1.0 / run_params->Hubble * result;

    // return time to present as a function of redshift
    return time;
}

double integrand_time_to_present(const double a, void *param)
{
    const struct params *run_params = (struct params *) param;
    return 1.0 / sqrt(run_params->Omega / a + (1.0 - run_params->Omega - run_params->OmegaLambda) + run_params->OmegaLambda * a * a);
}

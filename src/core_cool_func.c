/*
 * core_cool_func.c -- metal-dependent radiative cooling rate tables.
 *
 * Loads 8 pre-tabulated Sutherland & Dopita (1993) cooling function files
 * (CoolFunctions/stripped_m*.cie) covering metallicities from primordial to
 * 0.5 dex above solar.  Each table has 91 entries spanning log10(T/K) = 4.0
 * to 8.5 in steps of 0.05.  Provides read_cooling_functions() to load the
 * tables at startup, and get_metaldependent_cooling_rate() to interpolate
 * bilinearly in log-temperature and log-metallicity at runtime.
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "core_allvars.h"
#include "core_cool_func.h"

#define TABSIZE 91
#define LAST_TAB_INDEX (TABSIZE - 1)


static char *name[] = {
	"stripped_mzero.cie",
	"stripped_m-30.cie",
	"stripped_m-20.cie",
	"stripped_m-15.cie",
	"stripped_m-10.cie",
	"stripped_m-05.cie",
	"stripped_m-00.cie",
	"stripped_m+05.cie"
};


// Metallicities with respect to solar. Will be converted to absolute metallicities by adding log10(Z_sun), Zsun=0.02
static double metallicities[8] = {
	-5.0,   // actually primordial -> -infinity
	-3.0,
	-2.0,
	-1.5,
	-1.0,
	-0.5,
	+0.0,
	+0.5
};

static double get_rate(int tab, double logTemp);

#define NUM_METALS_TABLE        sizeof(metallicities)/sizeof(metallicities[0])

static double CoolRate[NUM_METALS_TABLE][TABSIZE];

/*
 * read_cooling_functions -- load the 8 CIE cooling tables from disk.
 *
 * Converts the relative metallicity array from solar-relative to absolute
 * log10(Z) by adding log10(0.02), then reads each of the 8 *.cie files
 * (TABSIZE=91 rows each) into CoolRate[][].  Must be called once at startup
 * before any call to get_metaldependent_cooling_rate().
 */
void read_cooling_functions(void)
{
    char buf[MAX_STRING_LEN];

    const double log10_zerop02 = log10(0.02);
    for(size_t i = 0; i < NUM_METALS_TABLE; i++) {
        metallicities[i] += log10_zerop02;     // add solar metallicity
    }

    for(size_t i = 0; i < NUM_METALS_TABLE; i++) {
        /* Concatenates the actual path to the root directory
           The variable ROOT_DIR is defined in the Makefile. C token pasting
           automatically concats the ROOT_DIR string and the "extra/..." string
        */
        snprintf(buf, MAX_STRING_LEN - 1, ROOT_DIR "/src/auxdata/CoolFunctions/%s", name[i]);
        FILE *fd = fopen(buf, "r");
        if(fd == NULL) {
            fprintf(stderr, "file `%s' not found\n", buf);
            ABORT(0);
        }
        for(int n = 0; n < TABSIZE; n++) {
            float sd_logLnorm;
            const int nitems = fscanf(fd, " %*f %*f %*f %*f %*f %f%*[^\n]",
                                      &sd_logLnorm);
            if(nitems != 1) {
                fprintf(stderr,"Error: Could not read cooling rate on line %d\n", n);
                ABORT(0);
            }
            CoolRate[i][n] = sd_logLnorm;
        }

        fclose(fd);
    }

}


/*
 * get_rate -- linearly interpolate the cooling rate for metallicity table tab
 * at log-temperature logTemp.
 *
 * Clamps logTemp to [4.0, 8.5].  Returns log10(cooling rate) in CGS.
 */
static double get_rate(int tab, double logTemp)
{
    const double dlogT = 0.05;
    const double inv_dlogT = 1.0/dlogT;

    if(logTemp < 4.0) {
        logTemp = 4.0;
    }

    int index = (int) ((logTemp - 4.0) * inv_dlogT);
    if(index >= LAST_TAB_INDEX) {
        /*MS: because index+1 is also accessed, therefore index can be at most LAST_TAB_INDEX */
        index = LAST_TAB_INDEX - 1;
    }

    const double logTindex = 4.0 + dlogT * index;

    const double rate1 = CoolRate[tab][index];
    const double rate2 = CoolRate[tab][index + 1];

    const double rate = rate1 + (rate2 - rate1) * inv_dlogT * (logTemp - logTindex);

    return rate;
}

/*
 * get_metaldependent_cooling_rate -- bilinear interpolation in (logT, logZ)
 * returning the radiative cooling rate in CGS (erg cm^3 s^-1).
 *
 * Clamps logZ to the table metallicity range, selects the two bracketing
 * metallicity tables, calls get_rate() for each, and linearly interpolates.
 */
double get_metaldependent_cooling_rate(const double logTemp, double logZ)
{
    if(logZ < metallicities[0])
        logZ = metallicities[0];

    if(logZ > metallicities[7])
        logZ = metallicities[7];

    int i = 0;
    while(logZ > metallicities[i + 1]) {
        i++;
    }

    // look up at i and i+1
    const double rate1 = get_rate(i, logTemp);
    const double rate2 = get_rate(i + 1, logTemp);
    const double rate = rate1 + (rate2 - rate1) / (metallicities[i + 1] - metallicities[i]) * (logZ - metallicities[i]);

    return pow(10.0, rate);
}




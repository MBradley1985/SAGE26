/* File: progressbar.c */
/*
  This file is a part of the Corrfunc package
  Copyright (C) 2015-- Manodeep Sinha (manodeep@gmail.com)
  License: MIT LICENSE. See LICENSE file under the top-level
  directory at https://github.com/manodeep/Corrfunc/
*/

#include "progressbar.h"
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>

#include "core_utils.h"

#ifndef MAXLEN
#define MAXLEN 1000
#endif

static int64_t total_steps = 0;
static struct timeval tstart;
static int prev_percent = -1;

void init_my_progressbar(FILE *stream, const int64_t N, int *interrupted)
{
    if (N <= 0) {
        fprintf(stream, "WARNING: N=%" PRId64 " is not positive. Progress bar will not be printed\n", N);
        total_steps = 0;
    } else {
        total_steps = N;
    }
    *interrupted = 0;
    prev_percent = -1;
    gettimeofday(&tstart, NULL);
}

void my_progressbar(FILE *stream, const int64_t curr_index, int *interrupted)
{
    if (total_steps <= 0) return;

    if (*interrupted == 1) {
        fprintf(stream, "\n");
        *interrupted = 0;
        prev_percent = -1; // Force redraw
    }

    // Calculate percentage
    double percent = (double)(curr_index + 1) / total_steps * 100.0;
    int integer_percent = (int)percent;

    // Only update if percentage changed or it's the first time
    if (integer_percent != prev_percent) {
        prev_percent = integer_percent;

        struct timeval tnow;
        gettimeofday(&tnow, NULL);
        
        double elapsed = (tnow.tv_sec - tstart.tv_sec) + (tnow.tv_usec - tstart.tv_usec) / 1000000.0;
        double rate = (curr_index + 1) / elapsed;
        double remaining = 0.0;
        if (rate > 0) {
             remaining = (total_steps - (curr_index + 1)) / rate;
        }
        
        int eta_h = (int)(remaining / 3600);
        int eta_m = (int)((remaining - eta_h * 3600) / 60);
        int eta_s = (int)(remaining - eta_h * 3600 - eta_m * 60);

        // Create bar
        int bar_width = 30;
        int pos = (int)((bar_width * percent) / 100.0);
        
        fprintf(stream, "\rProgress: [");
        for (int i = 0; i < bar_width; ++i) {
            if (i < pos) fprintf(stream, "=");
            else if (i == pos) fprintf(stream, ">");
            else fprintf(stream, " ");
        }
        fprintf(stream, "] %3d%% | ETA: %02d:%02d:%02d", integer_percent, eta_h, eta_m, eta_s);
        fflush(stream);
    }
}

void finish_myprogressbar(FILE *stream, int *interrupted)
{
    if (total_steps > 0) {
        // Ensure 100% is shown
        fprintf(stream, "\rProgress: [");
        for (int i = 0; i < 30; ++i) fprintf(stream, "=");
        fprintf(stream, "] 100%% | ETA: 00:00:00");
        fprintf(stream, "\n");
    }

    struct timeval t1;
    gettimeofday(&t1, NULL);
    char *time_string = get_time_string(tstart, t1);
    fprintf(stream, "Done. Time taken = %s\n", time_string);
    free(time_string);
    
    if (*interrupted) *interrupted = 0;
}

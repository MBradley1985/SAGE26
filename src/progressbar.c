/*
  Adapted from the Corrfunc package
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

static const char *SAGE26_ART[] = {
    " ██████   █████   ██████  ███████ ██████   ██████      . * .     ",
    "██       ██   ██ ██       ██           ██ ██        . ( * ) * .  ",
    "███████  ███████ ██   ███ █████    █████  ███████  * . )*( . *  ",
    "     ██  ██   ██ ██    ██ ██      ██      ██   ██   * ( * ) .   ",
    "███████  ██   ██  ██████  ███████ ███████  ██████      * .      "
};
#define SAGE26_LINES 5

static const char *RAINBOW[] = {
    "\033[1;31m",      /* Red         */
    "\033[38;5;208m",  /* Orange      */
    "\033[1;33m",      /* Yellow      */
    "\033[38;5;154m",  /* Yellow-Green*/
    "\033[1;32m",      /* Green       */
    "\033[38;5;51m",   /* Cyan-Green  */
    "\033[1;36m",      /* Cyan        */
    "\033[38;5;39m",   /* Sky Blue    */
    "\033[1;34m",      /* Blue        */
    "\033[38;5;93m",   /* Purple      */
    "\033[1;35m",      /* Magenta     */
    "\033[38;5;201m"   /* Pink        */
};
#define NUM_RAINBOW_COLORS 12

static void print_sage26_progress(FILE *stream, double percent)
{
    for(int line = 0; line < SAGE26_LINES; line++) {
        const char *art_line = SAGE26_ART[line];
        int len = strlen(art_line);
        int reveal_up_to = (int)((len * percent) / 100.0);

        fprintf(stream, "%s", RAINBOW[line % NUM_RAINBOW_COLORS]);
        for(int i = 0; i < len; i++) {
            if(i < reveal_up_to) {
                fprintf(stream, "%c", art_line[i]);
            } else {
                fprintf(stream, "\033[2m%c\033[0m%s",
                        art_line[i] == ' ' ? ' ' : '.',
                        RAINBOW[line % NUM_RAINBOW_COLORS]);
            }
        }
        fprintf(stream, "\033[0m\033[K\n");
    }
}

void init_my_progressbar(FILE *stream, const int64_t N, int *interrupted)
{
    if(N <= 0) {
        fprintf(stream, "WARNING: N=%" PRId64 " is not positive. Progress bar will not be printed\n", N);
        total_steps = 0;
    } else {
        total_steps = N;
    }
    *interrupted = 0;
    prev_percent = -1;
    gettimeofday(&tstart, NULL);

    fprintf(stream, "\033[0;37m\n===========================================================\n\033[0m");
    fprintf(stream, "\033[0;37mYou're creating an entire population of galaxies, have fun!\033[0m\n");
    fprintf(stream, "\033[0;37m===========================================================\n\n\033[0m");
}

void my_progressbar(FILE *stream, const int64_t curr_index, int *interrupted)
{
    if(total_steps <= 0) return;

    if(*interrupted == 1) {
        fprintf(stream, "\n");
        *interrupted = 0;
        prev_percent = -1;
    }

    double percent = (double)(curr_index + 1) / total_steps * 100.0;
    int integer_percent = (int)percent;

    if(integer_percent != prev_percent) {
        struct timeval tnow;
        gettimeofday(&tnow, NULL);

        double elapsed = (tnow.tv_sec - tstart.tv_sec) + (tnow.tv_usec - tstart.tv_usec) / 1000000.0;
        double rate = (curr_index + 1) / elapsed;
        double remaining = (rate > 0) ? (total_steps - (curr_index + 1)) / rate : 0.0;

        int eta_h = (int)(remaining / 3600);
        int eta_m = (int)((remaining - eta_h * 3600) / 60);
        int eta_s = (int)(remaining - eta_h * 3600 - eta_m * 60);

        /* Move cursor up to redraw the entire display (skip on first draw) */
        if(prev_percent >= 0) {
            fprintf(stream, "\033[%dA", SAGE26_LINES + 2);
        }

        prev_percent = integer_percent;

        print_sage26_progress(stream, percent);
        fprintf(stream, "\033[K\n");

        int bar_width = 52;
        int pos = (int)((bar_width * percent) / 100.0);

        fprintf(stream, "Progress: [");
        for(int i = 0; i < bar_width; ++i) {
            fprintf(stream, "%s", RAINBOW[i % NUM_RAINBOW_COLORS]);
            if(i < pos)       fprintf(stream, "*");
            else if(i == pos) fprintf(stream, ">");
            else              fprintf(stream, " ");
        }
        fprintf(stream, "\033[0m] %3d%% | ETA: %02d:%02d:%02d", integer_percent, eta_h, eta_m, eta_s);
        fprintf(stream, "\033[K\n");
        fflush(stream);
    }
}

void finish_myprogressbar(FILE *stream, int *interrupted)
{
    if(total_steps > 0) {
        /* Move cursor up to redraw at 100% */
        fprintf(stream, "\033[%dA", SAGE26_LINES + 2);

        print_sage26_progress(stream, 100.0);
        fprintf(stream, "\033[K\n");

        fprintf(stream, "Progress: [");
        for(int i = 0; i < 52; ++i) {
            fprintf(stream, "%s*", RAINBOW[i % NUM_RAINBOW_COLORS]);
        }
        fprintf(stream, "\033[0m] 100%% | ETA: 00:00:00\n");
    }

    struct timeval t1;
    gettimeofday(&t1, NULL);
    char *time_string = get_time_string(tstart, t1);
    fprintf(stream, "\n");
    fprintf(stream, "=============================================================\n");
    fprintf(stream, "Done...\n");
    fprintf(stream, "=============================================================\n");
    fprintf(stream, "** Congratulations! You've successfully created galaxies! **\n");
    fprintf(stream, "=============================================================\n");
    fprintf(stream, "Time taken = %s\n", time_string);
    fprintf(stream, "=============================================================\n");
    free(time_string);

    if(*interrupted) *interrupted = 0;
}

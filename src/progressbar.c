/* File: progressbar.c */
/*
  Enhanced progress bar for SAGE26 with adaptive ETA
  Based on original Corrfunc progressbar by Manodeep Sinha
  
  Improvements for DarkMode/radial integration workloads:
  - Exponential moving average (EMA) for rate estimation
  - Time-based updates for smoother display
  - Current rate display (forests/min)
  - Adaptive ETA that responds to work intensity changes
*/

#include "progressbar.h"
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "core_utils.h"

#ifndef MAXLEN
#define MAXLEN 1000
#endif

/* Progress bar state */
static int64_t total_steps = 0;
static int64_t prev_index = -1;
static struct timeval tstart;
static struct timeval tlast_update;
static struct timeval tlast_step;
static double ema_rate = 0.0;          /* Exponential moving average of processing rate */
static double ema_alpha = 0.15;        /* EMA smoothing factor (lower = smoother, higher = more responsive) */
static int prev_percent = -1;
static double min_update_interval = 0.25;  /* Minimum seconds between display updates */

/* ASCII art for SAGE26 */
static const char *SAGE26_ART[] = {
    " ██████   █████   ██████  ███████ ██████   ██████      . * .     ",
    "██       ██   ██ ██       ██           ██ ██        . ( * ) * .  ",
    "███████  ███████ ██   ███ █████    █████  ███████  * . )*( . *  ",
    "     ██  ██   ██ ██    ██ ██      ██      ██   ██   * ( * ) .   ",
    "███████  ██   ██  ██████  ███████ ███████  ██████      * .      "
};
#define SAGE26_LINES 5

/* Helper function to get time difference in seconds */
static double get_elapsed_seconds(struct timeval start, struct timeval end)
{
    return (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;
}

/* Helper function to print the ASCII art with progressive reveal */
static void print_sage26_progress(FILE *stream, double percent)
{
    /* Extended rainbow colors (12 colors for smoother gradient) */
    const char *colors[] = {
        "\033[1;31m",      /* Red */
        "\033[38;5;208m",  /* Orange */
        "\033[1;33m",      /* Yellow */
        "\033[1;32m",      /* Green */
        "\033[38;5;51m",   /* Cyan */
        "\033[1;36m",      /* Bright Cyan */
        "\033[38;5;39m",   /* Sky Blue */
        "\033[1;34m",      /* Blue */
        "\033[38;5;27m",   /* Deep Blue */
        "\033[38;5;93m",   /* Purple */
        "\033[1;35m",      /* Magenta */
        "\033[38;5;201m"   /* Pink */
    };
    int num_colors = 12;
    
    /* Print SAGE26 ASCII art */
    for (int line = 0; line < SAGE26_LINES; line++) {
        const char *art_line = SAGE26_ART[line];
        int len = strlen(art_line);
        int reveal_up_to = (int)((len * percent) / 100.0);
        
        fprintf(stream, "%s", colors[line % num_colors]);
        for (int i = 0; i < len; i++) {
            if (i < reveal_up_to) {
                fprintf(stream, "%c", art_line[i]);
            } else {
                fprintf(stream, "\033[2m%c\033[0m%s", art_line[i] == ' ' ? ' ' : '.', colors[line % num_colors]);
            }
        }
        fprintf(stream, "\033[0m\033[K\n");
    }
}

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
    prev_index = -1;
    ema_rate = 0.0;
    gettimeofday(&tstart, NULL);
    tlast_update = tstart;
    tlast_step = tstart;
    
    /* Adjust EMA responsiveness based on total work */
    /* For smaller runs, be more responsive; for larger runs, smoother */
    if (total_steps < 100) {
        ema_alpha = 0.3;
    } else if (total_steps < 1000) {
        ema_alpha = 0.2;
    } else if (total_steps < 10000) {
        ema_alpha = 0.15;
    } else {
        ema_alpha = 0.1;
    }
    
    /* Print the white message once at initialization */
    const char *message = "\n===========================================================\n";
    fprintf(stream, "\033[0;37m%s\033[0m", message);
    const char *message2 = "You're creating an entire population of galaxies, have fun!";
    fprintf(stream, "\033[0;37m%s\033[0m\n", message2);
    const char *message3 = "===========================================================\n\n";
    fprintf(stream, "\033[0;37m%s\033[0m", message3);
}

void my_progressbar(FILE *stream, const int64_t curr_index, int *interrupted)
{
    if (total_steps <= 0) return;

    if (*interrupted == 1) {
        fprintf(stream, "\n");
        *interrupted = 0;
        prev_percent = -1;
        prev_index = -1;
    }

    struct timeval tnow;
    gettimeofday(&tnow, NULL);
    
    /* Track per-step timing for EMA rate calculation */
    if (prev_index >= 0 && curr_index > prev_index) {
        double step_elapsed = get_elapsed_seconds(tlast_step, tnow);
        int64_t steps_done = curr_index - prev_index;
        
        if (step_elapsed > 0.0) {
            double instant_rate = steps_done / step_elapsed;  /* forests per second */
            
            /* Update EMA rate */
            if (ema_rate <= 0.0) {
                /* First measurement - initialize EMA */
                ema_rate = instant_rate;
            } else {
                /* EMA update: rate = α × instant + (1-α) × prev_rate */
                ema_rate = ema_alpha * instant_rate + (1.0 - ema_alpha) * ema_rate;
            }
        }
        tlast_step = tnow;
    }
    prev_index = curr_index;

    /* Calculate percentage */
    double percent = (double)(curr_index + 1) / total_steps * 100.0;
    int integer_percent = (int)percent;
    
    /* Check if we should update display (time-based or percent change) */
    double time_since_update = get_elapsed_seconds(tlast_update, tnow);
    int should_update = (integer_percent != prev_percent) || 
                        (time_since_update >= min_update_interval);
    
    if (should_update) {
        tlast_update = tnow;
        
        double elapsed = get_elapsed_seconds(tstart, tnow);
        
        /* Calculate remaining time using EMA rate (preferred) or fallback to average */
        double remaining = 0.0;
        double display_rate = 0.0;  /* forests per minute for display */
        
        if (ema_rate > 0.0) {
            /* Use EMA rate for more accurate ETA */
            remaining = (total_steps - (curr_index + 1)) / ema_rate;
            display_rate = ema_rate * 60.0;  /* Convert to per minute */
        } else if (elapsed > 0.0 && curr_index > 0) {
            /* Fallback to simple average rate */
            double avg_rate = (curr_index + 1) / elapsed;
            remaining = (total_steps - (curr_index + 1)) / avg_rate;
            display_rate = avg_rate * 60.0;
        }
        
        int eta_h = (int)(remaining / 3600);
        int eta_m = (int)((remaining - eta_h * 3600) / 60);
        int eta_s = (int)(remaining - eta_h * 3600 - eta_m * 60);

        /* Move cursor up to redraw the entire display (except on first draw) */
        if (prev_percent >= 0) {
            fprintf(stream, "\033[%dA", SAGE26_LINES + 3);  /* Move up N+2 lines (extra line for rate) */
        }
        
        prev_percent = integer_percent;
        
        /* Print the ASCII art with current progress */
        print_sage26_progress(stream, percent);

        /* Add blank line */
        fprintf(stream, "\033[K\n");
        
        /* Print the rainbow progress bar */
        int bar_width = 52;
        int pos = (int)((bar_width * percent) / 100.0);
        
        /* Extended rainbow colors for progress bar */
        const char *colors[] = {
            "\033[1;31m", "\033[38;5;208m", "\033[1;33m", "\033[38;5;154m",
            "\033[1;32m", "\033[38;5;51m", "\033[1;36m", "\033[38;5;39m",
            "\033[1;34m", "\033[38;5;93m", "\033[1;35m", "\033[38;5;201m"
        };
        int num_colors = 12;
        
        fprintf(stream, "Progress: [");
        for (int i = 0; i < bar_width; ++i) {
            fprintf(stream, "%s", colors[i % num_colors]);
            if (i < pos) fprintf(stream, "█");
            else if (i == pos) fprintf(stream, "▓");
            else fprintf(stream, "░");
        }
        fprintf(stream, "\033[0m] %5.1f%%", percent);
        fprintf(stream, "\033[K\n");
        
        /* Print rate and ETA on separate line */
        int elapsed_h = (int)(elapsed / 3600);
        int elapsed_m = (int)((elapsed - elapsed_h * 3600) / 60);
        int elapsed_s = (int)(elapsed - elapsed_h * 3600 - elapsed_m * 60);
        
        fprintf(stream, "Rate: \033[1;36m%6.1f\033[0m forests/min | ", display_rate);
        fprintf(stream, "Elapsed: \033[1;33m%02d:%02d:%02d\033[0m | ", elapsed_h, elapsed_m, elapsed_s);
        fprintf(stream, "ETA: \033[1;32m%02d:%02d:%02d\033[0m", eta_h, eta_m, eta_s);
        fprintf(stream, "\033[K\n");
        
        fflush(stream);
    }
}

void finish_myprogressbar(FILE *stream, int *interrupted)
{
    if (total_steps > 0) {
        /* Move cursor up to redraw final state */
        fprintf(stream, "\033[%dA", SAGE26_LINES + 3);
        
        /* Print final SAGE26 art (100% revealed) */
        print_sage26_progress(stream, 100.0);

        /* Add blank line */
        fprintf(stream, "\033[K\n");
        
        /* Extended rainbow colors for final progress bar */
        const char *colors[] = {
            "\033[1;31m", "\033[38;5;208m", "\033[1;33m", "\033[38;5;154m",
            "\033[1;32m", "\033[38;5;51m", "\033[1;36m", "\033[38;5;39m",
            "\033[1;34m", "\033[38;5;93m", "\033[1;35m", "\033[38;5;201m"
        };
        int num_colors = 12;
        
        /* Ensure 100% is shown on progress bar */
        fprintf(stream, "Progress: [");
        for (int i = 0; i < 52; ++i) {
            fprintf(stream, "%s█", colors[i % num_colors]);
        }
        fprintf(stream, "\033[0m] 100.0%%");
        fprintf(stream, "\n");
        
        /* Final stats line */
        struct timeval tend;
        gettimeofday(&tend, NULL);
        double total_elapsed = get_elapsed_seconds(tstart, tend);
        double final_rate = (total_elapsed > 0) ? (total_steps / total_elapsed * 60.0) : 0.0;
        
        int elapsed_h = (int)(total_elapsed / 3600);
        int elapsed_m = (int)((total_elapsed - elapsed_h * 3600) / 60);
        int elapsed_s = (int)(total_elapsed - elapsed_h * 3600 - elapsed_m * 60);
        
        fprintf(stream, "Rate: \033[1;36m%6.1f\033[0m forests/min | ", final_rate);
        fprintf(stream, "Elapsed: \033[1;33m%02d:%02d:%02d\033[0m | ", elapsed_h, elapsed_m, elapsed_s);
        fprintf(stream, "ETA: \033[1;32m00:00:00\033[0m");
        fprintf(stream, "\033[K\n");
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
    
    if (*interrupted) *interrupted = 0;
}

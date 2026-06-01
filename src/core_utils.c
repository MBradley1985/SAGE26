/*
 * core_utils.c -- low-level I/O and utility helpers.
 *
 * Derived from the Corrfunc package by Manodeep Sinha
 * (https://github.com/manodeep/Corrfunc/), MIT License.
 *
 * Provides: my_snprintf (bounds-checked snprintf wrapper), get_time_string
 * (human-readable elapsed time), getnumlines (count non-comment lines),
 * myfread/myfwrite/myfseek (stdio wrappers), mywrite/mypread/mypwrite
 * (POSIX fd wrappers with retry loops), and AlmostEqualRelativeAndAbs_double
 * (floating-point comparison with absolute and relative tolerances).
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#include <inttypes.h>    //defines PRId64 for printing int64_t + includes stdint.h
#include <math.h>
#include <string.h>
#include <limits.h>
#include <stdarg.h>
#include <ctype.h>

/* for read/write/pread/pwrite */
#include <sys/types.h>
#include <unistd.h>


#include "core_allvars.h"
#include "core_utils.h"


/*
 * my_snprintf -- bounds-checked vsnprintf wrapper.
 *
 * Behaves exactly like snprintf but prints an error and returns -1 if the
 * output was truncated (nwritten > len) or an encoding error occurred.
 */
int my_snprintf(char *buffer, int len, const char *format, ...)
{
  va_list args;
  int nwritten = 0;

  va_start(args, format);
  nwritten = vsnprintf(buffer, (size_t)len, format, args);
  va_end(args);
  if (nwritten > len || nwritten < 0) {
    fprintf(stderr,
            "ERROR: printing to string failed (wrote %d characters while only %d characters were "
            "allocated)\n",
            nwritten, len);
    fprintf(stderr, "Increase `len'=%d in the header file\n", len);
    return -1;
  }
  return nwritten;
}

/*
 * get_time_string -- format elapsed time between t0 and t1 as a human-readable
 * string ("N secs", "N mins N secs", etc.).  Caller must free() the result.
 */
char *get_time_string(struct timeval t0, struct timeval t1)
{
  const size_t MAXLINESIZE = 1024;
  char *time_string = malloc(MAXLINESIZE * sizeof(char));
  if(time_string == NULL)  {
      fprintf(stderr,"Error: Could not allocate memory to hold string variable representing time in function '%s'..returning\n", __FUNCTION__);
      return NULL;
  }
  double timediff = t1.tv_sec - t0.tv_sec;
  double ratios[] = {24 * 3600.0, 3600.0, 60.0, 1};
  double timeleft = timediff;

  if (timediff < ratios[2]) {
      my_snprintf(time_string, MAXLINESIZE, "%6.3lf secs",
                  1e-6 * (t1.tv_usec - t0.tv_usec) + timediff);
  } else {
      int which = 0;
      size_t curr_index = 0;
      while (which < 4) {
          char units[4][10] = {"days", "hrs", "mins", "secs"};

          double time_to_print = floor(timeleft / ratios[which]);
          if (time_to_print > 1) {
              timeleft -= (time_to_print * ratios[which]);
              char tmp[MAXLINESIZE];
              my_snprintf(tmp, MAXLINESIZE, "%5d %s", (int)time_to_print, units[which]);
              const size_t len = strlen(tmp);
              const size_t required_len = curr_index + len + 1;
              XRETURN(MAXLINESIZE >= required_len, NULL,
                      "buffer overflow will occur: string has space for %zu bytes while concatenating "
                      "requires at least %zu bytes\n",
                      MAXLINESIZE, required_len);
              strcpy(time_string + curr_index, tmp);
              curr_index += len;
          }
          which++;
      }
  }

  return time_string;
}

/* getnumlines -- count non-empty, non-comment lines in a text file. */
int64_t getnumlines(const char *fname,const char comment)
{
    const int MAXLINESIZE = 10000;
    int64_t nlines=0;
    char str_line[MAXLINESIZE];

    FILE *fp = fopen(fname,"rt");
    if(fp == NULL) {
        return -1;
    }

    while(1){
        if(fgets(str_line, MAXLINESIZE,fp)!=NULL) {
            /*
              fgets always terminates the string with a '\0'
              on a successful read
             */
            char *c = &str_line[0];
            while(*c != '\0' && isspace(*c)) {
                c++;
            }
            if(*c != '\0' && *c !=comment) {
                 nlines++;
            }
        } else {
            break;
        }
    }
    fclose(fp);
    return nlines;
}


/* myfread/myfwrite/myfseek -- thin wrappers around fread/fwrite/fseeko. */
size_t myfread(void *ptr, const size_t size, const size_t nmemb, FILE * stream)
{
    return fread(ptr, size, nmemb, stream);
}

size_t myfwrite(const void *ptr, const size_t size, const size_t nmemb, FILE * stream)
{
    return fwrite(ptr, size, nmemb, stream);
}

int myfseek(FILE * stream, const long offset, const int whence)
{
    return fseeko(stream, offset, whence);
}

/*
 * mywrite -- write nbytes to fd, retrying on partial writes until all bytes
 * are written.  Aborts on a write error.
 */
ssize_t mywrite(int fd, const void *ptr, size_t nbytes)
{
    size_t nbytes_left = nbytes;
    ssize_t tot_nbytes_written = 0;
    char *buf = (char *) ptr;
    while(nbytes_left > 0) {
        ssize_t bytes_written = write(fd, buf, nbytes_left);
        if(bytes_written > 0 ) {
            nbytes_left -= bytes_written;
            buf += bytes_written;
            tot_nbytes_written += bytes_written;
        } else {
            fprintf(stderr,"Error whiling writing to file\n");
            perror(NULL);
            ABORT(FILE_WRITE_ERROR);
        }
    }

    return tot_nbytes_written;
}



/*
 * mypread -- positional read, retrying until all nbytes are read from offset.
 * Aborts on read error.
 */
ssize_t mypread(int fd, void *ptr, const size_t nbytes, off_t offset)
{
    size_t nbytes_left = nbytes;
    ssize_t tot_nbytes_read = 0;
    char *buf = (char *) ptr;
    while(nbytes_left > 0) {
        ssize_t bytes_read = pread(fd, buf, nbytes_left, offset);
        if(bytes_read > 0 ) {
            nbytes_left -= bytes_read;
            buf += bytes_read;
            offset += bytes_read;
            tot_nbytes_read += bytes_read;
        } else {
            perror(NULL);
            ABORT(FILE_READ_ERROR);
        }
    }

    return tot_nbytes_read;
}

/*
 * mypwrite -- positional write, retrying until all nbytes are written from offset.
 * Returns FILE_WRITE_ERROR on failure.
 */
ssize_t mypwrite(int fd, const void *ptr, const size_t nbytes, off_t offset)
{
    size_t nbytes_left = nbytes;
    ssize_t tot_nbytes_written = 0;
    char *buf = (char *) ptr;
    while(nbytes_left > 0) {
        ssize_t bytes_written = pwrite(fd, buf, nbytes_left, offset);
        if(bytes_written > 0 ) {
            nbytes_left -= bytes_written;
            buf += bytes_written;
            offset += bytes_written;
            tot_nbytes_written += bytes_written;
        } else {
            return FILE_WRITE_ERROR;
        }
    }

    return tot_nbytes_written;
}

/*
 * AlmostEqualRelativeAndAbs_double -- return EXIT_SUCCESS when A and B are
 * within maxDiff (absolute) or maxRelDiff * max(|A|,|B|) (relative).
 */
int AlmostEqualRelativeAndAbs_double(double A, double B,
                                     const double maxDiff,
                                     const double maxRelDiff)
{
    // Check if the numbers are really close -- needed
    // when comparing numbers near zero.
    double diff = fabs(A - B);
    if (diff <= maxDiff)
        return EXIT_SUCCESS;

    A = fabs(A);
    B = fabs(B);
    double largest = (B > A) ? B : A;

    if (diff <= largest * maxRelDiff)
        return EXIT_SUCCESS;

    /* fprintf(stderr,"diff = %e largest * maxRelDiff = %e\n", diff, largest * maxRelDiff); */
    return EXIT_FAILURE;
}

/*
 * core_mymalloc.h -- public interface for tracked memory allocation wrappers.
 *
 * Declares thin wrappers around malloc/calloc/realloc/free that abort with
 * file/line context on allocation failure rather than returning NULL silently.
 * All SAGE26 heap allocations go through these wrappers.
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

    /* functions in core_mymalloc.c */
    extern void *mymalloc(size_t n);
    extern void *mycalloc(const size_t count, const size_t size);
    extern void *myrealloc(void *p, size_t n);
    extern void myfree(void *p);
#ifdef VERBOSE
    extern void print_allocated(void);
#endif

#ifdef __cplusplus
}
#endif

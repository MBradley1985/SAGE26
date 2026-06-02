/*
 * buffered_io.h -- write-buffering layer for binary galaxy catalogue output.
 *
 * Declares the buffered_io struct and three functions (setup, write, cleanup)
 * that accumulate writes in an in-memory buffer and flush to a file descriptor
 * only when the buffer is full, reducing syscall overhead for binary output.
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

struct buffered_io{
    size_t bytes_allocated;
    size_t bytes_stored;
    int file_descriptor;
    off_t current_offset;
    void *buffer;
};

    extern int setup_buffered_io(struct buffered_io *buf_io, const size_t buffer_size, int output_fd, const off_t start_offset);
    extern int write_buffered_io(struct buffered_io *buf_io, const void *src, size_t num_bytes_to_write);
    extern int cleanup_buffered_io(struct buffered_io *buf_io);

#ifdef __cplusplus
}
#endif

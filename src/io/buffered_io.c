#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>

#include "buffered_io.h"
#include "../core_utils.h"


int setup_buffered_io(struct buffered_io *buf_io, const size_t buffer_size, int output_fd, const off_t start_offset) 
{
    /* start_offset may legitimately be negative (off_t is signed); only check the other two */
    if(buffer_size == 0 || output_fd <= 0) {
        fprintf(stderr,"Error: In %s> buffer size = %zd (bytes) and output file descriptor = %d must be greater than 0\n",
                      __FUNCTION__, buffer_size, output_fd);
        return -1;
    }

    buf_io->buffer = malloc(buffer_size);
    if (buf_io->buffer == NULL) {
        fprintf(stderr,"Error: In %s> Could not allocate memory of size %zu bytes for buffered io\n", __FUNCTION__, buffer_size);
        return -1;
    }

    buf_io->bytes_allocated = buffer_size;
    buf_io->bytes_stored = 0;
    buf_io->file_descriptor = output_fd;
    buf_io->current_offset = start_offset;

    return EXIT_SUCCESS;
}

int write_buffered_io(struct buffered_io *buf_io, const void *src, size_t num_bytes_to_write)
{
    if(buf_io == NULL || src == NULL)  {
        fprintf(stderr,"Error: In %s> Could not validate input parameters.  buffer pointer address = %p, "
                       "source pointer address = %p\n",
                       __FUNCTION__, buf_io, src);
        return -1;
    }

    if ((buf_io->bytes_stored + num_bytes_to_write) < buf_io->bytes_allocated) {
        char* dest = (char*)buf_io->buffer + buf_io->bytes_stored;
        memcpy(dest, src, num_bytes_to_write);
        buf_io->bytes_stored += num_bytes_to_write;
        return EXIT_SUCCESS;
    }

    /* buffer full — flush it first */
    ssize_t bytes_written = mypwrite(buf_io->file_descriptor, buf_io->buffer, buf_io->bytes_stored, buf_io->current_offset);
    if (bytes_written < 0) {
        return bytes_written;
    }
    buf_io->current_offset += bytes_written;
    buf_io->bytes_stored = 0;

    if (num_bytes_to_write >= buf_io->bytes_allocated) {
        /* new write exceeds buffer capacity — bypass buffer and write directly */
        const ssize_t new_bytes_written = mypwrite(buf_io->file_descriptor, src, num_bytes_to_write, buf_io->current_offset);
        if (new_bytes_written < 0) {
            return new_bytes_written;
        }

        if(new_bytes_written != (ssize_t) num_bytes_to_write) {
            fprintf(stderr,"Error: In function %s> Expected to write %zd bytes but wrote %zd bytes instead\n",
                            __FUNCTION__, num_bytes_to_write, new_bytes_written);
            return -1;
        }

        buf_io->current_offset += new_bytes_written;
        bytes_written += new_bytes_written;
    } else {
        memcpy(buf_io->buffer, src, num_bytes_to_write);
        buf_io->bytes_stored += num_bytes_to_write;
    }

    return EXIT_SUCCESS;
}

int cleanup_buffered_io(struct buffered_io *buf_io) 
{
    if(buf_io == NULL) {
        fprintf(stderr,"Error: In %s> Could not validate input parameters. buffer pointer address = %p", __FUNCTION__, buf_io);
        return -1;
    }

    ssize_t bytes_written = mypwrite(buf_io->file_descriptor, buf_io->buffer, buf_io->bytes_stored, buf_io->current_offset);
    if (bytes_written < 0) {
        fprintf(stderr,"Error: In %s> Could not finalise the file in buffered io\n", __FUNCTION__);
        perror(NULL);
        return bytes_written;
    }
    buf_io->current_offset += bytes_written;

    free(buf_io->buffer);
    buf_io->buffer = NULL;
    buf_io->bytes_allocated = 0;
    buf_io->bytes_stored = 0;

    return EXIT_SUCCESS;
}
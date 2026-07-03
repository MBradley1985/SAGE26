/*
 * buffered_io.c -- write-buffering layer for binary galaxy catalogue output.
 *
 * Accumulates writes in an in-memory buffer and flushes to a file descriptor
 * when the buffer is full, reducing syscall overhead for the many small per-
 * galaxy writes produced by save_binary_galaxies().
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h> //for memcpy

#include "buffered_io.h"
#include "../core_utils.h" //for mypwrite


int setup_buffered_io(struct buffered_io *buf_io, const size_t buffer_size, int output_fd, const off_t start_offset) 
{
    /*start_offset can (in theory at least) be negative. off_t is a signed integer (32 or 64-bit, depending on on the compile-time
    constatnt for FILE_OFFSET_BITS). Therefore, we only check for the two other parameters */
    if(buffer_size <= 0 || output_fd <= 0) {
        fprintf(stderr,"Error: In %s> All buffer size = %zd (bytes) and output file descriptor = %d must be greater than 0\n", 
                      __FUNCTION__, buffer_size, output_fd);
        return -1;
    }

    // Allocate memory for the buffer
    buf_io->buffer = malloc(buffer_size);
    if (buf_io->buffer == NULL) {
        fprintf(stderr,"Error: In %s> Could not allocate memory of size %zu bytes for buffered io\n", __FUNCTION__, buffer_size);
        return -1; // Return -1 if malloc fails
    }

    // Initialize the buffered_io struct
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

    // Check if the buffer has enough space to hold new data
    if ((buf_io->bytes_stored + num_bytes_to_write) < buf_io->bytes_allocated) {
        // Copy data to the buffer
        char* dest = (char*)buf_io->buffer + buf_io->bytes_stored;
        memcpy(dest, src, num_bytes_to_write);

        // Increment the number of bytes stored in the buffered_io struct
        buf_io->bytes_stored += num_bytes_to_write;
        return EXIT_SUCCESS;
    } 
    
    //If we are here, then we are exceeding the allocated buffer size. Therefore, we need
    //to first write out all the data that exists within the buffer
    ssize_t bytes_written = mypwrite(buf_io->file_descriptor, buf_io->buffer, buf_io->bytes_stored, buf_io->current_offset);
    if (bytes_written < 0) {
        return bytes_written; // Return error if pwrite fails
    }

    // Update the offset and number of bytes in the buffered_io struct
    buf_io->current_offset += bytes_written;
    buf_io->bytes_stored = 0;

    //Now we need to consider the new write request. If the number of bytes to write
    //exceeds the allocated buffer size, then we should just directly write out all the 
    //data from the src pointer.
    if (num_bytes_to_write >= buf_io->bytes_allocated) {
        const ssize_t new_bytes_written = mypwrite(buf_io->file_descriptor, src, num_bytes_to_write, buf_io->current_offset);
        if (new_bytes_written < 0) {
            return new_bytes_written; // Return error if pwrite fails
        }

        if(new_bytes_written != (ssize_t) num_bytes_to_write) {
            fprintf(stderr,"Error: In function %s> Expected to write %zd bytes but wrote %zd bytes instead\n", 
                            __FUNCTION__, num_bytes_to_write, num_bytes_to_write);
            return -1;
        }

        // Update the offset and remaining data to write
        buf_io->current_offset += new_bytes_written;
        bytes_written += new_bytes_written;
    } else {

        // Copy the new data to the buffer and increment the number of bytes stored
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

    // Write out any remaining data in the buffer
    ssize_t bytes_written = mypwrite(buf_io->file_descriptor, buf_io->buffer, buf_io->bytes_stored, buf_io->current_offset);
    if (bytes_written < 0) {
        fprintf(stderr,"Error: In %s> Could not finalise the file in buffered io\n", __FUNCTION__);
        perror(NULL);
        return bytes_written;
    }

    // Update the offset and number of bytes in the buffered_io struct
    buf_io->current_offset += bytes_written;

    // Free the memory allocated for the buffer
    free(buf_io->buffer);
    buf_io->buffer = NULL;
    buf_io->bytes_allocated = 0;
    buf_io->bytes_stored = 0;

    return EXIT_SUCCESS;
}
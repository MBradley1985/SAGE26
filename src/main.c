#include <stdio.h>
#include <stdlib.h>

#ifdef MPI
#include <mpi.h>
#endif

#include "sage.h"


int main(int argc, char **argv)
{
    int ThisTask = 0;
    int NTasks = 1;

#ifdef MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    MPI_Comm_size(MPI_COMM_WORLD, &NTasks);
#endif

    if(argc != 2) {
        fprintf(stderr, "\n  usage: %s <parameterfile>\n\n", argv[0]);
#ifdef MPI
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        MPI_Finalize();
#endif
        return EXIT_FAILURE;
    }

    void *run_params;
    int status = run_sage(ThisTask, NTasks, argv[1], &run_params);
    if(status != EXIT_SUCCESS) {
        goto err;
    }

#ifdef MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    status = finalize_sage(run_params);
    if(status != EXIT_SUCCESS) {
        goto err;
    }

#ifdef MPI
    MPI_Finalize();
#endif
    return EXIT_SUCCESS;

err:
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    MPI_Finalize();
#endif
    fprintf(stderr, "If the fix to this isn't obvious, please feel free to open an issue on our GitHub page.\n"
                    "https://github.com/sage-home/SAGE26/issues/new\n");
    return status;

}

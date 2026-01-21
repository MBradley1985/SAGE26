#!/bin/bash

#SBATCH --job-name=sage_flythrough_mpi
#SBATCH --output=slurm-flythrough-mpi-%A_%a.out
#SBATCH --array=0-7
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=2G

# MPI-parallelized flythrough rendering
# Each array task runs a different mode/color combination
# Each task uses 16 MPI ranks to parallelize frame rendering

# List of argument sets for each array job
ARGS_LIST=(
  "orbit density mov"
  "orbit type mov"
  "flythrough density mov"
  "flythrough type mov"
  "evolution density mov"
  "evolution type mov"
  "combined density mov"
  "combined type mov"
)

ml purge
ml restore basic

ARGS=${ARGS_LIST[$SLURM_ARRAY_TASK_ID]}
set -- $ARGS

MODE=$1
COLOR=$2
FORMAT=$3

echo "Running mode=$MODE color=$COLOR format=$FORMAT with $SLURM_NTASKS MPI ranks"

srun python plotting/flythrough.py --mode $MODE --color-by $COLOR --format $FORMAT

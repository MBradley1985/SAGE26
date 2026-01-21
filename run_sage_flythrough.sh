#!/bin/bash

#SBATCH --job-name=sage_flythrough
#SBATCH --output=slurm-flythrough-%A_%a.out
#SBATCH --array=0-7
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --mem=2G

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

if [[ $MODE == "evolution" || $MODE == "combined" ]]; then
    python plotting/flythrough.py --mode $MODE  --color-by $2 --format $3
else
    python plotting/flythrough.py --mode $MODE --color-by $2 --format $3
fi
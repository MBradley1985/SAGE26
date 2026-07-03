#!/bin/bash

#SBATCH --job-name=sage_batch
#SBATCH --output=slurm-sage-%A_%a.out
#SBATCH --array=0-2
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem=5G

# List of config files to run (edit as needed; keep --array in sync with the list length)
CONFIG_FILES=(
  "input/millennium.par"
  "input/millennium_all.par"
  "input/millennium_vanilla.par"
)

CONFIG_FILE=${CONFIG_FILES[$SLURM_ARRAY_TASK_ID]}
echo "Running SAGE with config: $CONFIG_FILE"
./sage "$CONFIG_FILE"

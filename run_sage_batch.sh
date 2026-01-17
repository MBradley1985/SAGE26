#!/bin/bash

#SBATCH --job-name=sage_batch
#SBATCH --output=slurm-sage-%A_%a.out
#SBATCH --array=0-16
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem=5G

# List of config files to run (edit as needed)
CONFIG_FILES=(
  "input/millennium10.par"
  "input/millennium20.par"
  "input/millennium30.par"
  "input/millennium40.par"
  "input/millennium50.par"
  "input/millennium60.par"
  "input/millennium70.par"
  "input/millennium80.par"
  "input/millennium90.par"
  "input/millennium100.par"
  "input/miniuchuu2.par"
  "input/millenniumkmt09.par"
  "input/millenniumkd12.par"
  "input/millenniumk13.par"
  "input/millenniumgd14.par"
  "input/millenniumnoffb.par"
  "input/millenniumc16feedback.par"
)

CONFIG_FILE=${CONFIG_FILES[$SLURM_ARRAY_TASK_ID]}
echo "Running SAGE with config: $CONFIG_FILE"
./sage "$CONFIG_FILE"

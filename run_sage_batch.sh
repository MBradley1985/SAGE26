#!/bin/bash

#SBATCH --job-name=sage_batch
#SBATCH --output=slurm-sage-%A_%a.out
#SBATCH --array=0-8
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem=5G

# List of config files to run (edit as needed; keep --array in sync with the list length)
# The millennium_no*.par entries are the ablation series: each differs from
# millennium_all.par by a single physics switch. Plot them with
# `python plotting/ablation_series.py`.
CONFIG_FILES=(
  "input/millennium.par"
  "input/millennium_all.par"
  "input/millennium_vanilla.par"
  "input/millennium_nofire.par"
  "input/millennium_noh2.par"
  "input/millennium_nocgm.par"
  "input/millennium_noffb.par"
  "input/millennium_norps.par"
  "input/millennium_noallfour.par"
)

CONFIG_FILE=${CONFIG_FILES[$SLURM_ARRAY_TASK_ID]}
echo "Running SAGE with config: $CONFIG_FILE"
./sage "$CONFIG_FILE"

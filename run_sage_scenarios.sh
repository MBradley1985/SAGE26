#!/bin/bash

#SBATCH --job-name=sage_scenarios
#SBATCH --output=slurm-sage-%A_%a.out
#SBATCH --array=0-6
#SBATCH --cpus-per-task=1
#SBATCH --time=06:30:00
#SBATCH --mem=10G

# List of config files to run (edit as needed; keep --array in sync with the list length)
# Full-Millennium (512 tree files) BH growth scenarios, 2026-10-05 batch. Each
# tests one physically-motivated mechanism for producing LRD-like overmassive
# black holes; see bh_investigation/GROWTH_STORIES.md (Stories B, C, H) for
# the mini-Millennium results these are following up on. Plot comparisons with
# `python plotting/compare_runs.py --run <dir>:<label> ...`.
CONFIG_FILES=(
  "input/scenarios/baseline_heavy_eddlim.par"
  "input/scenarios/unlimited_heavy.par"
  "input/scenarios/unlimited_light.par"
  "input/scenarios/early_merger_burst_scheme3.par"
  "input/scenarios/early_z_unlimited_scheme11.par"
  "input/scenarios/low_mass_host_scheme13.par"
  "input/scenarios/continuous_cold_cloud_radioeff.par"
)

CONFIG_FILE=${CONFIG_FILES[$SLURM_ARRAY_TASK_ID]}
echo "Running SAGE with config: $CONFIG_FILE"
./sage "$CONFIG_FILE"

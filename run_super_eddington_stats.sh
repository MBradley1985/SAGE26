#!/bin/bash

#SBATCH --job-name=sage_supereddington
#SBATCH --output=slurm-supedd-%A_%a.out
#SBATCH --array=0-20
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=16G

# Super-Eddington / LRD event statistics (plotting/bh_super_eddington_stats.py),
# one array task per run directory. Each task streams that run's model_*.hdf5
# (one file or many) and writes <OUT>/<run>/summary.md + super_eddington_stats.json.
#
# When all tasks are done, build the combined README (comparison table + every
# per-run summary) with:
#     python plotting/bh_super_eddington_stats.py --combine <OUT>
#
# Submit from the SAGE26 directory (plotting/... is a relative path).
# Keep --array in sync with the length of RUNS (0 .. N-1).

ml h5py/3.8.0
ml numpy/1.24.2-scipy-bundle-2023.02

ROOT=/fred/oz004/jwillingham
OUT=super_eddington_stats

RUNS=(
  baseline_heavy_eddlim
  baseline_sage26
  continuous_cold_cloud_radioeff
  early_merger_burst_scheme3
  early_z_unlimited_scheme11
  heavy_firstevent
  low_mass_host_scheme13
  millennium_powerlaw_cap_scheme14
  millennium_seed_mstar_threshold
  miniuchuu_baseline_heavy_eddlim
  miniuchuu_continuous_cold_cloud_radioeff
  miniuchuu_early_merger_burst_scheme3
  miniuchuu_early_z_unlimited_scheme11
  miniuchuu_low_mass_host_scheme13
  miniuchuu_powerlaw_cap_scheme14
  miniuchuu_seed_mstar_threshold
  miniuchuu_unlimited_heavy
  miniuchuu_unlimited_light
  miniuchuu_vanilla
  unlimited_heavy
  unlimited_light
)

RUN=${RUNS[$SLURM_ARRAY_TASK_ID]}
if [[ -z "$RUN" ]]; then
  echo "No run for array index $SLURM_ARRAY_TASK_ID" >&2
  exit 1
fi

echo "=============================================================="
echo "Super-Eddington stats: $ROOT/$RUN"
echo "=============================================================="

python -u plotting/bh_super_eddington_stats.py "$ROOT/$RUN" -o "$OUT" --no-readme

echo "Done."

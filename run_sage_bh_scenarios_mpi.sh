#!/bin/bash
#SBATCH --job-name=SAGE26_BH
#SBATCH --output=slurm-sage-bh-%A_%a.out
#SBATCH --array=0-11
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000MB
#SBATCH --time=06:00:00

# BH growth scenarios on full Millennium and mini-Uchuu (512 tree files each).
# One array task per .par file: runs SAGE under MPI (10 ranks), then, in the same
# job, makes the allresults + multi-z plots for that run (as run_sage_scenarios_plotting.sh).
# Keep --array in sync with the length of CONFIG_FILES (0 .. N-1).
# Plots go to <OutputDir>/plots/.

module purge
ml restore basic

cd "$SLURM_SUBMIT_DIR"

CONFIG_FILES=(
  # ---- Millennium (other scenarios + vanilla already run) ----
  "input/scenarios/millennium_powerlaw_cap_scheme14.par"
  "input/scenarios/millennium_seed_mstar_threshold.par"
  # ---- mini-Uchuu ----
  "input/scenarios/miniuchuu_baseline_heavy_eddlim.par"
  "input/scenarios/miniuchuu_unlimited_heavy.par"
  "input/scenarios/miniuchuu_unlimited_light.par"
  "input/scenarios/miniuchuu_early_merger_burst_scheme3.par"
  "input/scenarios/miniuchuu_early_z_unlimited_scheme11.par"
  "input/scenarios/miniuchuu_low_mass_host_scheme13.par"
  "input/scenarios/miniuchuu_continuous_cold_cloud_radioeff.par"
  "input/scenarios/miniuchuu_powerlaw_cap_scheme14.par"
  "input/scenarios/miniuchuu_seed_mstar_threshold.par"
  "input/scenarios/miniuchuu_vanilla.par"
)

CONFIG_FILE=${CONFIG_FILES[$SLURM_ARRAY_TASK_ID]}
if [[ -z "$CONFIG_FILE" || ! -f "$CONFIG_FILE" ]]; then
  echo "No config for array index $SLURM_ARRAY_TASK_ID" >&2
  exit 1
fi

# Output directory comes straight from the par file, so plotting can't drift from the run
DIR=$(awk '$1=="OutputDir"{print $2; exit}' "$CONFIG_FILE")
DIR=${DIR%/}
OUT="$DIR/plots"
GLOB="$DIR/model_*.hdf5"

export OMP_NUM_THREADS=1
echo "Running SAGE on $SLURM_NTASKS ranks with config: $CONFIG_FILE"
echo "Output dir: $DIR"

srun ./sage "$CONFIG_FILE" || { echo "SAGE failed; skipping plots" >&2; exit 1; }

# ---------------------------- plotting ----------------------------
ml h5py/3.8.0
ml numpy/1.24.2-scipy-bundle-2023.02
ml matplotlib/3.7.0

echo "=============================================================="
echo "Plotting: $DIR"
echo "=============================================================="

echo "--- allresults-history ---"
python plotting/allresults-history.py "$DIR" -o "$OUT"

echo "--- allresults-local ---"
python plotting/allresults-local.py "$DIR" -o "$OUT"

echo "--- allresults-blackholes ---"
python plotting/allresults-blackholes.py -i "$GLOB" -o "$OUT"

echo "--- bh_lrd_analysis_multiz (Mdot-M_BH + f_BH-M_BH only) ---"
python plotting/bh_lrd_analysis_multiz.py -i "$GLOB" --outdir "$OUT" \
  --no-panel-c --no-panel-d --no-panel-e --no-panel-f

echo "Done."

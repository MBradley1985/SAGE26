#!/bin/bash
#SBATCH --job-name=SAGE26_BH_rerun
#SBATCH --output=slurm-sage-bh-rerun-%A_%a.out
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000MB
#SBATCH --time=12:00:00

# RERUN of the two broken runs only (millennium_powerlaw_cap_scheme14, miniuchuu_vanilla).
# BH growth scenarios on full Millennium and mini-Uchuu (512 tree files each).
# One array task per .par file: runs SAGE under MPI (10 ranks), then, in the same
# job, makes the allresults + multi-z plots for that run (as run_sage_scenarios_plotting.sh).
# Keep --array in sync with the length of CONFIG_FILES (0 .. N-1).
# Plots go to <OutputDir>/plots/.

ml h5py/3.8.0
ml numpy/1.24.2-scipy-bundle-2023.02
ml matplotlib/3.7.0
ml astropy/5.2.2

cd "$SLURM_SUBMIT_DIR"

CONFIG_FILES=(
  # Reruns of runs whose model_*.hdf5 were left truncated/unreadable.
  "input/scenarios/millennium_powerlaw_cap_scheme14.par"
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

# History is I/O-bound and slow; run it as its own job.
# echo "--- allresults-history ---"
# python plotting/allresults-history.py "$DIR" -o "$OUT"

echo "--- allresults-local ---"
python plotting/allresults-local.py "$DIR" -o "$OUT"

echo "--- allresults-blackholes ---"
python plotting/allresults-blackholes.py -i "$GLOB" -o "$OUT"

echo "--- bh_lrd_analysis_multiz (Mdot-M_BH + f_BH-M_BH only) ---"
python plotting/bh_lrd_analysis_multiz.py -i "$GLOB" --outdir "$OUT" \
  --no-panel-c --no-panel-d --no-panel-e --no-panel-f

echo "Done."

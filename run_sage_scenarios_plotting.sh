#!/bin/bash

#SBATCH --job-name=sage_scenarios_plots
#SBATCH --output=slurm-plots-%A_%a.out
#SBATCH --array=0-6
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=8G

# Plotting pass for the 2026-10-05 full-Millennium BH growth scenarios (see
# run_sage_scenarios.sh for the sims that produced these outputs). One array
# task per run directory. For each run, generates:
#   - allresults-history.py   (galaxy history/evolution plots)
#   - allresults-local.py     (z=0 "local" diagnostic plots)
#   - allresults-blackholes.py (BH growth channels, accretion rate function, BHMF, BH-bulge)
#   - bh_lrd_analysis_multiz.py, panels a+b only (Mdot-M_BH and f_BH-M_BH, multi-redshift)
# Each script defaults its own output to <run_dir>/plots/, set explicitly
# below for clarity.

ml h5py/3.8.0
ml numpy/1.24.2-scipy-bundle-2023.02
ml matplotlib/3.7.0

RUN_DIRS=(
  "/fred/oz004/jwillingham/baseline_heavy_eddlim"
  "/fred/oz004/jwillingham/unlimited_heavy"
  "/fred/oz004/jwillingham/unlimited_light"
  "/fred/oz004/jwillingham/early_merger_burst_scheme3"
  "/fred/oz004/jwillingham/early_z_unlimited_scheme11"
  "/fred/oz004/jwillingham/low_mass_host_scheme13"
  "/fred/oz004/jwillingham/continuous_cold_cloud_radioeff"
)

DIR=${RUN_DIRS[$SLURM_ARRAY_TASK_ID]}
OUT="$DIR/plots"
GLOB="$DIR/model_*.hdf5"

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

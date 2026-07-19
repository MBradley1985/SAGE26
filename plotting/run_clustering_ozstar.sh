#!/bin/bash -l
#SBATCH --job-name=mqg_clustering
#SBATCH --mail-type=ALL
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8GB
# =============================================================================
# Run plotting/mqg_clustering_bias.py on Ngarrgu Tindebeek (tooarrana).
# Prereq: run plotting/setup_clustering_env.sh once (in an interactive job).
# Bump --mem-per-cpu for bigger boxes: the loader holds one snapshot's galaxy
# arrays in RAM (tens of GB for miniUchuu; more for full Uchuu).
# =============================================================================
ml purge
# SAME modules as setup (gcc/12.3.0 toolchain):
module load gcc/12.3.0 gsl/2.7 python/3.11.3 scipy-bundle/2023.07 matplotlib/3.7.2 astropy/5.3.3

source "$HOME/envs/sage-clustering/bin/activate"

export MPLBACKEND=Agg                          # headless figures (also enforced in-script)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK     # Corrfunc parallelises xi_gg with OpenMP

echo "Job $SLURM_JOB_ID on $SLURM_NODELIST  |  python: $(which python)"

python plotting/mqg_clustering_bias.py \
    --model output/miniuchuu \
    --redshifts 0 0.99 1.91 2.83 3.87 \
    --nthreads "$SLURM_CPUS_PER_TASK" \
    --sigma-8 0.8159 --n-s 0.9667 \
    --mvir-lim 11 15 --bias-lim 0 8.5

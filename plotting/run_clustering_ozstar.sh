#!/bin/bash -l
#SBATCH --job-name=mqg_clustering
#SBATCH --mail-type=ALL
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8GB
# =============================================================================
# Run plotting/mqg_clustering_bias.py on an Lmod cluster (no conda).
# Prereq: run plotting/setup_clustering_env.sh once to build the venv.
# Adjust --mem-per-cpu to the box: the loader holds one snapshot's galaxy
# arrays in RAM (tens of GB for miniUchuu; more for full Uchuu).
# =============================================================================
ml purge
ml restore basic
ml load gcc gsl python hdf5          # SAME modules used in setup (edit to your site)

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

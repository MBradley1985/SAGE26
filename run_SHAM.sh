#!/bin/bash
#SBATCH --job-name=sham
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0:30:00

ml h5py/3.8.0
ml numpy/1.24.2-scipy-bundle-2023.02
ml matplotlib/3.7.0
ml astropy/5.2.2

python SHAM/run_sham.py --z 0 1 2 4 6 8 --nproc 8
python SHAM/plot_sham_fbh.py --grid
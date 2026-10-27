#!/bin/bash
#SBATCH --job-name=sham_uchuu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0:30:00

ml h5py/3.8.0
ml numpy/1.24.2-scipy-bundle-2023.02
ml matplotlib/3.7.0
ml astropy/5.2.2

python SHAM/run_sham.py \
  --trees-dir /fred/oz004/msinha/simulations/uchuu_suite/U400/trees/Uchuu400Planck2016_C50/lhalo \
  --tree-name lhalotree.bin \
  --a-list /fred/oz004/simulations/uchuu_suite/U400/Uchuu400_scalefactor.txt \
  --n-files-total 512 --box-size 400 --hubble 0.6774 \
  --z 0 1 2 4 6 8 --nproc 8 \
  --out-dir SHAM/output_uchuu

python SHAM/plot_sham_fbh.py --sham-dir SHAM/output_uchuu --out-dir SHAM/plots_uchuu --grid --no-sage
python SHAM/plot_sham_mbh_mstar.py --sham-dir SHAM/output_uchuu --out-dir SHAM/plots_uchuu
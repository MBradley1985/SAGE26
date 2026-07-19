#!/bin/bash -l
# =============================================================================
# One-time Python venv setup for plotting/mqg_clustering_bias.py on
# Ngarrgu Tindebeek (tooarrana) -- Lmod hierarchical modules, NO conda.
#
#   RUN INSIDE AN INTERACTIVE JOB so Corrfunc compiles on a compute node
#   (its SIMD is baked in at build time; login-node builds can throw
#   "illegal instruction" on the compute nodes):
#       sinteractive            # or salloc / srun --pty bash
#       bash plotting/setup_clustering_env.sh
#       source $HOME/envs/sage-clustering/bin/activate
#
# Modules below are the gcc/12.3.0 (2023a) toolchain on this cluster. If they
# ever change, rerun `module load gcc/<ver>` then
#   `module avail 2>&1 | grep -iE 'gsl/|python/|scipy-bundle/|matplotlib/|astropy/'`
# and update the load line.
# =============================================================================
set -e
VENV="$HOME/envs/sage-clustering"

ml purge                         # (nvidia/slurm are sticky and stay loaded -- fine)
# numpy/scipy/pandas via scipy-bundle; matplotlib + astropy as modules;
# gsl -> Corrfunc build; python -> the venv interpreter.
module load gcc/12.3.0 gsl/2.7 python/3.11.3 scipy-bundle/2023.07 matplotlib/3.7.2 astropy/5.3.3

command -v gsl-config >/dev/null || { echo "ERROR: gsl-config not on PATH after 'module load gsl/2.7'"; exit 1; }
echo "python: $(which python)  |  gcc: $(gcc -dumpversion)  |  gsl: $(gsl-config --version)"

# --system-site-packages: reuse the module numpy/scipy/matplotlib/astropy so pip
# doesn't rebuild them; pip only adds what the modules don't provide.
python -m venv --system-site-packages "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel

# h5py: no module on this cluster -> pip wheel (bundles its own HDF5).
# Corrfunc: compiles against gcc/12.3.0 + gsl/2.7.  colossus: pure python.
CC=gcc python -m pip install Corrfunc colossus h5py

python - <<'PY'
import numpy, scipy, matplotlib, astropy, h5py, colossus       # noqa
from Corrfunc.theory import xi                                   # noqa
print(f"OK: numpy {numpy.__version__} | h5py {h5py.__version__} | Corrfunc + colossus + astropy import cleanly")
PY

echo
echo "Done. Activate with:  source $VENV/bin/activate"
echo "Then run e.g.:        sbatch plotting/run_clustering_ozstar.sh"

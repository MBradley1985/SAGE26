#!/bin/bash -l
# =============================================================================
# One-time setup of a Python venv for plotting/mqg_clustering_bias.py on an
# Lmod (module) cluster -- NO conda. Mirrors the run_sage_ozstar.sh pattern
# (`ml purge; ml restore basic`), then adds the compilers/libraries Corrfunc
# needs and builds a venv on top.
#
#   Usage:   bash plotting/setup_clustering_env.sh
#   Then:    source $HOME/envs/sage-clustering/bin/activate
#
# IMPORTANT: module names/versions are site-specific. Find yours with:
#   module spider gcc ; module spider gsl ; module spider python ; module spider hdf5
# and edit the `ml load` line below to match. Corrfunc needs gcc + GSL
# (with `gsl-config` on PATH) to compile.
# =============================================================================
set -e

VENV="$HOME/envs/sage-clustering"

# --- modules -----------------------------------------------------------------
ml purge
ml restore basic                     # your saved base collection (as in run_sage_ozstar.sh)
# Edit these to the exact module names on your cluster (see `module spider`):
ml load gcc gsl python hdf5          # gcc+gsl -> Corrfunc build; python -> venv; hdf5 -> h5py

echo "Using: $(which python)  |  gcc: $(gcc --version | head -1)  |  gsl: $(gsl-config --version 2>/dev/null || echo 'NOT FOUND')"
[ -z "$(command -v gsl-config)" ] && { echo "ERROR: gsl-config not on PATH -- load a gsl module before Corrfunc will build."; exit 1; }

# --- venv --------------------------------------------------------------------
# --system-site-packages lets you reuse module-provided numpy/scipy/h5py/matplotlib
# if the python module ships them; pip then only adds what's missing (Corrfunc, colossus).
python -m venv --system-site-packages "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel

# --- python deps -------------------------------------------------------------
# NOTE on Corrfunc + AVX: it compiles for the CPU it is BUILT on. If your login
# node is a different/older CPU than the compute nodes you may hit "illegal
# instruction" at runtime. Safest: run THIS SCRIPT inside an interactive job on a
# compute node (e.g. `salloc`/`srun --pty bash`), so Corrfunc matches the target.
CC=gcc python -m pip install -r "$(dirname "$0")/requirements-clustering.txt"

# --- verify ------------------------------------------------------------------
python - <<'PY'
from Corrfunc.theory import xi          # noqa
from colossus.cosmology import cosmology
import h5py, numpy, scipy, matplotlib   # noqa
print("OK: Corrfunc + colossus + stack import cleanly")
PY

echo
echo "Done. Activate with:  source $VENV/bin/activate"
echo "Then run e.g.:        sbatch plotting/run_clustering_ozstar.sh"

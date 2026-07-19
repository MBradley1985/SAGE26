#!/bin/bash -l
# =============================================================================
# One-time Python venv setup for plotting/mqg_clustering_bias.py on
# Ngarrgu Tindebeek (tooarrana) -- Lmod hierarchical modules, NO conda.
#
#   RUN ON THE LOGIN NODE. Compute nodes here are air-gapped (no DNS/internet),
#   so pip cannot reach PyPI from inside a job. On the login node:
#       cd SAGE26
#       bash plotting/setup_clustering_env.sh
#       source $HOME/envs/sage-clustering/bin/activate
#   Then TEST on a compute node:
#       python -c "from Corrfunc.theory import xi; print('ok')"
#   If that throws "Illegal instruction" (login built a newer SIMD than the
#   compute CPU), use the OFFLINE rebuild at the very bottom of this file.
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

# =============================================================================
# OFFLINE Corrfunc rebuild -- ONLY if the login-built Corrfunc throws
# "Illegal instruction" on a compute node (SIMD mismatch). Splits download
# (needs internet, login node) from build (needs the compute CPU, no internet).
#
#   # 1) LOGIN node (internet): fetch the Corrfunc source tarball
#   source $HOME/envs/sage-clustering/bin/activate
#   pip download Corrfunc --no-deps --no-binary :all: -d $HOME/pip_offline
#
#   # 2) COMPUTE node (interactive job; modules loaded + venv active): build here
#   module load gcc/12.3.0 gsl/2.7 python/3.11.3 scipy-bundle/2023.07 matplotlib/3.7.2 astropy/5.3.3
#   source $HOME/envs/sage-clustering/bin/activate
#   pip uninstall -y Corrfunc
#   CC=gcc pip install --no-index --no-build-isolation $HOME/pip_offline/[Cc]orrfunc-*.tar.gz
#   python -c "from Corrfunc.theory import xi; print('rebuilt ok')"
# =============================================================================

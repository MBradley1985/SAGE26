# Getting Started

## Dependencies

| Package | Required | Notes |
|---------|----------|-------|
| C99 compiler (gcc or clang) | Yes | |
| [GSL](https://www.gnu.org/software/gsl/) | Optional | auto-detected via `gsl-config`; the build warns and disables tests without it |
| [HDF5](https://www.hdfgroup.org/) | Optional | HDF5 tree reading and output. **On by default** (`USE-HDF5 := yes`) |
| MPI | Optional | parallel execution. **On by default** (`USE-MPI := yes`) |

## Build

```bash
git clone https://github.com/MBradley1985/SAGE26.git
cd SAGE26
make                   # default build -- MPI and HDF5 both ON; produces ./sage and libsage.so
make USE-MPI=           # serial build (empty value disables MPI)
make USE-HDF5=          # build without HDF5 support
make MEM-CHECK=yes     # address/UB sanitizers for debugging (gcc only)
make clean             # remove all build artefacts
```

Note that `USE-MPI` and `USE-HDF5` are both set to `yes` at the top of the
`Makefile`, so a bare `make` gives you an MPI build compiled with `mpicc`. To
get a serial binary -- which the regression baseline requires -- pass an
*empty* value: `make clean && make USE-MPI=`.

## First run

On a fresh clone, run `first_run.sh` to create the output directories and download the
Mini-Millennium test trees (~50 MB):

```bash
./first_run.sh
```

## Running the model

```bash
# Serial
./sage input/millennium.par

# MPI parallel
mpirun -np 4 ./sage input/millennium.par
```

Output is written to the `OutputDir` specified in the parameter file (default
`output/millennium/`). With `OutputFormat sage_hdf5` each snapshot produces a
`model_N.hdf5` file containing one dataset per galaxy property.

## Plotting

Python plotting scripts are in `plotting/`. Install dependencies with:

```bash
pip install -r requirements.txt
```

Then:

```bash
python plotting/allresults-local.py                       # z=0 diagnostics
python plotting/allresults-local.py path/to/output/       # specify output dir
python plotting/allresults-history.py                     # multi-redshift diagnostics
python plotting/paper_plots.py                            # all paper figures
python plotting/paper_plots.py 1 3 5                      # specific figure numbers
```

## Tests

```bash
cd tests && make test               # build and run all 19 suites
cd tests && make test_conservation  # conservation tests only (fastest)
cd tests && make quick              # single fastest check
bash tests/run_integration_tests.sh # full integration test (slower)
```

Other focused targets in `tests/Makefile`: `test_physics`, `test_regime`,
`test_ffb`, `test_bulge`.

The regression baseline verifies that output is bit-identical across ~5444
datasets. It needs a serial build (see the build note above):

```bash
make clean && make USE-MPI=
bash tests/regression_baseline.sh    # or: make regression
```

After any physics change, recapture the baseline with:

```bash
python3 tests/regression_baseline.py capture input/millennium.par
```

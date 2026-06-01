# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is SAGE26?

SAGE (Semi-Analytic Galaxy Evolution) is a C99 code for modelling galaxy formation in a cosmological context. It reads N-body merger tree files, evolves galaxies through cosmic time using semi-analytic prescriptions, and writes galaxy catalogues. SAGE26 is a major update to Croton et al. (2016), adding a two-regime CGM model, feedback-free burst (FFB) galaxies, FIRE feedback, and extended bulge/ICS tracking.

## Build

```bash
make          # builds ./sage executable and libsage.so
make clean    # removes all compiled objects and binaries
```

Key Makefile flags (edit at top of `Makefile`):
- `USE-MPI := yes` — enables MPI parallel execution (switches compiler to `mpicc`)
- `USE-HDF5 := yes` — enables HDF5 tree reading and output
- `MEM-CHECK = yes` — enables address/undefined-behaviour sanitizers (gcc only, ~2x slowdown)

## Running the model

```bash
# Serial
./sage input/millennium.par

# MPI parallel
mpirun -np 4 ./sage input/millennium.par

# Run all microuchuu parameter variants in parallel (logs to logs/)
./run_microuchuu_local.sh
```

On a first run, execute `./first_run.sh` to create output directories and download Mini-Millennium trees.

## Tests

Tests live in `tests/` and are standalone C programs linked against the sage sources. GSL is required.

```bash
make tests                 # from root — invokes ./tests/test_sage.sh
cd tests && make test      # alternatively, run from tests/ directly
make test_conservation     # run a single suite (37 tests)
make test_regime           # CGM/regime physics (21 tests)
make test_physics          # physics validation (31 tests)
make quick                 # fastest check — conservation only
make check_dependencies    # show which suites can be built with current sources
bash run_integration_tests.sh  # full integration test (slower, realistic physics)
```

The test build skips suites whose required source files are missing rather than failing — useful when working with a partial source tree. Logs go to `tests/test_output/*.log`. New tests need a corresponding entry in `tests/Makefile` and `run_integration_tests.sh`. The test framework is in `tests/test_framework.h` (macros: `ASSERT_CLOSE`, `ASSERT_IN_RANGE`, etc.).

## Plotting

Python plotting scripts are in `plotting/`. Dependencies: `pip install -r requirements.txt` (cffi, mpi4py, h5py, numpy, matplotlib, scipy, pandas, astropy, colossus).

```bash
python plotting/allresults-local.py                         # z=0 diagnostics on default output
python plotting/allresults-local.py path/to/output/         # specify output dir
python plotting/allresults-history.py                       # multi-redshift diagnostics
python plotting/paper_plots.py                              # all paper figures
python plotting/paper_plots.py 1 3 5                        # specific figure numbers
```

Linting: `flake8` (configured in `.flake8`; ignores E501 long lines, F401 unused imports). Pre-commit hook runs flake8 automatically.

## Code architecture

### Execution flow (`src/`)

1. `main.c` — parses CLI args, calls `run_sage()`
2. `sage.c` (`run_sage()`) — reads parameter file, loads tree forests, loops over forests calling `construct_galaxies()`
3. `core_build_model.c` (`construct_galaxies()`) — walks the merger tree recursively, then calls `evolve_galaxies()` per halo
4. `evolve_galaxies()` — the per-timestep physics loop: infall → cooling → star formation → feedback → disk instability → reincorporation → mergers → output

### Sub-stepping between snapshots

Each snapshot interval is integrated in `STEPS=10` substeps (defined in `macros.h`). `evolve_galaxies()` can adaptively raise this up to `MAX_STEPS=30` when timescale ratios demand finer resolution, but SFR history arrays (`SfrDisk[STEPS]`, etc.) are always sized to the fixed `STEPS`, and adaptive sub-steps are mapped back into those bins. `ABSOLUTEMAXSNAPS=200` caps total snapshot count.

### Unit convention

Internal code units are set in the parameter file (defaults: `UnitLength_in_cm=Mpc/h`, `UnitMass_in_g=10^10 M_sun`, `UnitVelocity_in_cm_per_s=km/s`). All masses in the `GALAXY` struct are in `10^10 M_sun/h`, distances in `Mpc/h`, velocities in `km/s`. Convert before comparing to observations.

### Error-handling macros (`macros.h`)

The codebase uses its own assertion macros, not `assert()`. Use these in new code:
- `XASSERT(expr, exit_code, fmt, ...)` — abort with location info if `expr` is false
- `XRETURN(expr, return_value, fmt, ...)` — return early instead of aborting
- `CHECK_STATUS_AND_RETURN_ON_FAIL(status, ret, fmt, ...)` — propagate negative status codes
- `CHECK_POINTER_AND_RETURN_ON_NULL(ptr, fmt, ...)` — guard malloc/calloc results
- `ABORT(sigterm)` — fail hard with file/line and a link to file an issue

All compile out under `-DNDEBUG`. Error enum values live in `core_allvars.h` (`enum sage_error_types`).

### Key source files

| File | Role |
|------|------|
| `core_allvars.h` | All structs: `GALAXY`, `params`, `halo_data`, tree-format structs |
| `core_simulation.h` | Simulation constants and snapshot arrays |
| `macros.h` | Compile-time constants (`STEPS`, `ABSOLUTEMAXSNAPS`, etc.) |
| `model_cooling_heating.c` | Cooling, AGN heating; regime-aware (`cooling_recipe_cgm`, `cooling_recipe_regime_aware`) |
| `model_starformation_and_feedback.c` | SF prescriptions (H2, FIRE, FFB), SN feedback routing |
| `model_mergers.c` | Major/minor merger classification, dynamical friction timescales, BH growth |
| `model_disk_instability.c` | Toomre instability criterion, bulge/BH growth channel |
| `model_infall.c` | Baryon infall, reionization suppression, regime-routing to CGM vs HotGas |
| `model_reincorporation.c` | Return of ejected gas to hot reservoir |
| `model_misc.c` | Halo concentration, bulge radii, utility functions |
| `core_save.c` | Galaxy output; `GalaxyIndex` generation |
| `io/` | Tree readers (lhalo binary/HDF5, ConsistentTrees ASCII/HDF5, Genesis, Gadget4) and galaxy writers (binary, HDF5) |

### The `GALAXY` struct (`core_allvars.h`)

The central data structure. Key baryonic reservoirs: `ColdGas`, `StellarMass`, `BulgeMass`, `HotGas`, `EjectedMass`, `BlackHoleMass`, `CGMgas`, `H2gas`, `ICS`. Metals shadow each reservoir (`MetalsColdGas`, etc.). CGM properties: `tcool`, `tff`, `tcool_over_tff`. Separate merger- vs instability-driven bulge tracking: `MergerBulgeMass`, `InstabilityBulgeMass`, `MergerBulgeRadius`, `InstabilityBulgeRadius`. Full SFH arrays: `SFHMassDisk[ABSOLUTEMAXSNAPS]`, `SFHMassBulge[ABSOLUTEMAXSNAPS]`.

### Parameter files (`input/*.par`)

Control all physics switches and model parameters. Key switches:

| Parameter | Values | Effect |
|-----------|--------|--------|
| `SFprescription` | 0–7 | 0=Croton06; 1=BR06 H2; 2,3=Somerville25; 4=KD12; 5=KMT09; 6=K13; 7=GD14 |
| `AGNrecipeOn` | 0–3 | 0=off; 1=empirical; 2=Bondi-Hoyle; 3=cold cloud accretion |
| `CGMrecipeOn` | 0/1 | Two-regime CGM model (key SAGE26 innovation) |
| `FIREmodeOn` | 0/1 | FIRE feedback physics |
| `FeedbackFreeModeOn` | 0–7 | FFB galaxy formation (0=off) |
| `ConcentrationOn` | 0–3 | Halo concentration method |
| `SaveFullSFH` | 0/1 | Track full star formation history |
| `TrackICSAssembly` | 0/1 | Track ICS disruption/accretion history |
| `OutputFormat` | sage_binary/sage_hdf5 | Output format |

### Two-regime CGM model

The key physics innovation. `Regime` field on `GALAXY` is set by `determine_and_store_regime()` based on the Voit (2015) criterion (M/Mshock)^(4/3). Regime 0 = hot halo (classical cooling), Regime 1 = CGM-dominated. In CGM regime, gas routes to `CGMgas` reservoir and cooling goes through `cooling_recipe_cgm()`. AGN heating in CGM regime uses a persistent `HeatingReservoir` that decays on the dynamical time.

### Supported tree formats

Set via `TreeType` in the parameter file. Valid values: `lhalo_binary`, `lhalo_hdf5`, `consistent_trees_ascii`, `consistent_trees_hdf5`, `genesis_lhalo_hdf5`, `gadget4_hdf5`. Each format has a corresponding reader in `src/io/read_tree_*.c`.

### Output

HDF5 output (`model_N.hdf5`, written via `io/save_gals_hdf5.c`) contains one dataset per galaxy property. Binary output (`io/save_gals_binary.c`) is also available. Python scripts read HDF5 directly with `h5py`. Each parameter file writes to its own `OutputDir` (e.g. `output/millennium/`, `output/microuchuu_Karl/`), and many physics-variant directories coexist for comparison plots. Observational comparison data is in `data/`.

### Shared library and bindings

The build produces both `./sage` (executable) and `libsage.so` (shared library) so the same compiled code can be driven from Python via cffi (`make pyext`). The git commit SHA is embedded into the binary at compile time as `GITREF_STR` for run provenance.

### SAGE-PSO companion package

Particle Swarm Optimization for automated parameter calibration lives in a separate repo: <https://github.com/MBradley1985/SAGE-PSO>. It calls `libsage.so` to evaluate parameter samples against observational constraints. Not part of this tree.

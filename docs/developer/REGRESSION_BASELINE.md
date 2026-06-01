# Regression Baseline (Phase 0)

The regression baseline is the safety net that lets the cleanup pass proceed. Every cleanup commit must leave it passing **bit-identically** against the frozen reference.

## Policy

- **Bit-identical at the dataset level.** Every dataset inside every output HDF5 file must hash to the same SHA-256 as the baseline. This is what "bit-identical scientific output" means in practice.
- **HDF5 container bytes are not required to match.** Re-running SAGE with identical scientific output still produces non-identical raw `.hdf5` files — HDF5 stores B-tree layout, chunk allocation order, and internal timestamps that vary between runs. The verifier records these file-level differences as informational warnings, not failures.
- **Serial only.** The baseline is captured and verified on the serial (non-MPI) build. MPI parity is out of scope until the serial cleanup is complete.
- **Frozen reference commit.** The reference is a single tagged commit (`baseline/pre-cleanup`). It is not re-captured during cleanup. If a labelled bug fix lands and changes outputs, the baseline is re-captured at that commit and the new hash set replaces the old one — with the bug-fix commit cited in `CHANGELOG.md`.

## What gets captured

For each golden parameter file, after a clean serial run:

1. **Raw output files.** Every `model_N.hdf5` produced into the parameter file's `OutputDir`. Hashed with SHA-256.
2. **HDF5 dataset hashes.** For each dataset inside each `model_N.hdf5`, the SHA-256 of the dataset's raw bytes. This isolates which field drifted when a top-level hash changes.
3. **Smoke statistics.** A handful of scalar summaries (z=0 stellar mass function bin counts, BHMF bin counts, integrated SFRD per snapshot, total counts per galaxy type). Fast to compute, useful as a first-look diagnostic when a baseline fails.

All three are stored under `tests/baseline/<param_file_stem>/` as JSON manifests.

## Golden parameter files

### Per-conversation sweep

- **`input/millennium.par`** — Mini-Millennium trees, classical CGM-off regime. ~11 s serial wall-time, ~1.5 GB output, 5380 datasets. Exercises the baseline cooling/SF/feedback path. This is the one config that `tests/regression_baseline.sh` runs by default at the end of every cleanup conversation.

### Release-time spot check (optional, not in default sweep)

- **`input/microuchuu.par`** — Uchuu100, consistent_trees_ascii. Exercises the SAGE26-specific physics (`CGMrecipeOn=1`, FFB, FIRE). ~255 s serial wall-time, ~17 GB output, 4204 datasets. Verified bit-identical at the per-dataset level during Phase 0 setup. Too slow for per-commit use; run manually before tagging a release or after any change that plausibly touches CGM/FFB/FIRE code paths:

```bash
python3 tests/regression_baseline.py capture input/microuchuu.par   # once, to seed
python3 tests/regression_baseline.py verify  input/microuchuu.par   # to check
```

The microuchuu manifest is not committed by default (it's a 1+ MB file recording a 17 GB capture). Capture locally when needed; the script will write it under `tests/baseline/microuchuu/`.

## Build invariants

The baseline only holds against a fixed build configuration. Capture and verification must use:

- The same compiler and compiler version.
- The same optimisation flags (the Makefile defaults at the frozen commit).
- `USE-MPI := no`, `USE-HDF5 := yes`, `MEM-CHECK = no`.
- The same GSL version (record it in the manifest).

The manifest records the build environment alongside the hashes so verification on a different machine can fail loudly rather than silently.

## Capture procedure

1. Check out `baseline/pre-cleanup`.
2. `make clean && make` with the invariants above.
3. For each golden parameter file:
   - Delete its `OutputDir`.
   - Run `./sage <param_file>` to completion.
   - Compute file hashes, per-dataset hashes, smoke statistics.
   - Write `tests/baseline/<param_file_stem>/manifest.json`.
4. Commit `tests/baseline/` into the repository.

## Verification procedure

`tests/regression_baseline.sh` performs:

1. Assert the build matches the recorded invariants (warn loudly on mismatch).
2. For each golden parameter file:
   - Clean its `OutputDir`.
   - Run `./sage <param_file>`.
   - Recompute hashes and smoke statistics.
   - Diff against `manifest.json`.
3. Exit 0 only if every hash matches. On failure, print which dataset(s) drifted in which file.

## Workflow integration

- The script runs at the end of every cleanup conversation, before commit.
- If it fails on a cleanup batch that was meant to be cosmetic: revert and investigate. Don't commit the drift.
- If it fails because the batch was actually a bug fix: split the commit. The bug fix lands separately with `fix:` in the message, the baseline is re-captured against that new commit, and `CHANGELOG.md` records the behavioural change.

## How to use

The baseline is implemented as `tests/regression_baseline.py` (the capture/verify engine) plus `tests/regression_baseline.sh` (a thin driver that iterates the golden configs).

### Verify (the common case)

Before any commit on a cleanup branch:

```bash
make clean && make USE-MPI=
bash tests/regression_baseline.sh
```

Exit 0 means every golden config still reproduces its baseline byte-for-byte. Exit 1 means at least one config drifted; the script prints which files and which datasets changed. Exit 2 means setup is broken (sage missing, linked against MPI, etc.).

### Capture (only when re-baselining)

After a commit that is **explicitly a bug fix** and is intended to change outputs:

```bash
make clean && make USE-MPI=
bash tests/regression_baseline.sh --capture
git add tests/baseline/
git commit -m "fix: re-baseline after <description>"
```

The new hashes replace the old ones. The commit message must reference the bug-fix commit that caused the change; `CHANGELOG.md` records the behavioural difference.

### Adding a new golden config

Two edits:

1. Add the .par path to the `GOLDEN=(...)` array in `tests/regression_baseline.sh`.
2. Run `bash tests/regression_baseline.sh --capture` to seed the manifest, then commit `tests/baseline/<new_stem>/manifest.json`.

### What lives where

| Path | Purpose |
|------|---------|
| `tests/regression_baseline.py` | Capture/verify engine. Single Python file, depends only on h5py + numpy. |
| `tests/regression_baseline.sh` | Driver that runs the engine over every golden config. |
| `tests/baseline/<stem>/manifest.json` | Recorded hashes + smoke stats per golden config. Committed. |
| `output/<dir>/` | Sage's actual output directory. Wiped on every capture/verify. Not committed. |

## Open questions for Phase 0 kickoff

- Whether smoke statistics should include any z>0 quantities beyond what the script currently records (per-snapshot type counts and log10 mean/min/max of StellarMass, BlackHoleMass, ColdGas, HotGas).
- Whether to additionally hash the binary output format (currently only HDF5 is captured, since plotting scripts read HDF5).
- Whether to wire `tests/regression_baseline.sh` into a `pre-commit` hook (currently it's manual).

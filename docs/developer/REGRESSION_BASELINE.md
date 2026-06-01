# Regression Baseline (Phase 0)

The regression baseline is the safety net that lets the cleanup pass proceed. Every cleanup commit must leave it passing **bit-identically** against the frozen reference.

## Policy

- **Bit-identical, not tolerance-bounded.** Outputs must match the captured baseline byte-for-byte. Any drift means the change altered behaviour and must be reverted or explicitly relabelled as a bug fix.
- **Serial only.** The baseline is captured and verified on the serial (non-MPI) build. MPI parity is out of scope until the serial cleanup is complete.
- **Frozen reference commit.** The reference is a single tagged commit (`baseline/pre-cleanup`). It is not re-captured during cleanup. If a labelled bug fix lands and changes outputs, the baseline is re-captured at that commit and the new hash set replaces the old one — with the bug-fix commit cited in `CHANGELOG.md`.

## What gets captured

For each golden parameter file, after a clean serial run:

1. **Raw output files.** Every `model_N.hdf5` produced into the parameter file's `OutputDir`. Hashed with SHA-256.
2. **HDF5 dataset hashes.** For each dataset inside each `model_N.hdf5`, the SHA-256 of the dataset's raw bytes. This isolates which field drifted when a top-level hash changes.
3. **Smoke statistics.** A handful of scalar summaries (z=0 stellar mass function bin counts, BHMF bin counts, integrated SFRD per snapshot, total counts per galaxy type). Fast to compute, useful as a first-look diagnostic when a baseline fails.

All three are stored under `tests/baseline/<param_file_stem>/` as JSON manifests.

## Golden parameter files

Two configs cover the active physics surface:

- `input/millennium.par` — Mini-Millennium trees, classical CGM-off regime. Smallest fast-running config; exercises the baseline cooling/SF/feedback path.
- One `input/microuchuu_*.par` variant — exercises the SAGE26-specific physics: `CGMrecipeOn=1`, FFB, FIRE. Final choice to be made when Phase 0 begins.

Both must run serially in a tractable wall-time on a developer laptop. If the microuchuu variant is too slow, reduce to a single forest subset for baseline purposes — but the same reduced configuration must be used for both capture and verification.

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

## Open questions for Phase 0 kickoff

- Exact microuchuu variant to use as the second golden config.
- Whether smoke statistics should include any z>0 quantities or just z=0.
- Whether to additionally hash the binary output format (currently only HDF5 is planned, since plotting scripts read HDF5).

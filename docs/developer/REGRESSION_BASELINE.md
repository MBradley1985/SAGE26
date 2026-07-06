# Regression baseline: bit-identical output verification

SAGE26 guards its science output with a byte-for-byte regression baseline.
Any change that is not supposed to alter the physics must reproduce the
committed baseline exactly — every HDF5 dataset bit-identical.

## The two systems

| System | Entry point | Build required | What it hashes |
|---|---|---|---|
| Dataset-level baseline (primary) | `make regression` or `tests/regression_baseline.sh` | serial (`make clean && make USE-MPI=`) | SHA-256 of every dataset in every `model*.hdf5` (~5,444 datasets for mini-Millennium), plus file-level hashes and smoke statistics |
| Binary checksum benchmark | `tests/run_benchmark_test.sh verify` | MPI (`make clean && make`) | SHA-256 of the 64 `sage_binary` output files from `tests/benchmark/benchmark.par` |

The dataset-level system is the release gate. The binary benchmark
additionally covers the `sage_binary` output writer, which the HDF5 baseline
does not exercise.

## Everyday usage

```bash
# one-time (or after intentional physics changes): capture the baseline
make clean && make USE-MPI=
python3 tests/regression_baseline.py capture input/millennium.par

# after every code change: verify
python3 tests/regression_baseline.py verify input/millennium.par
# or run the whole default sweep:
make regression
```

`verify` exits non-zero on any dataset drift and prints exactly which
datasets changed. Differences confined to HDF5 container metadata (file
timestamps etc.) are reported as notes, not failures — the science data is
compared dataset by dataset.

## Policy

- Cleanup/refactor commits must keep `verify` green. A refactor that moves
  a single hash is a physics change in disguise (often via floating-point
  reassociation or FMA contraction under `-march=native`) and must be
  reverted or reworked.
- Intentional physics changes re-capture the baseline in the same commit,
  and the commit message must say so explicitly.
- The bit-identical guarantee holds only within a fixed toolchain. The
  manifest records a build-environment fingerprint (compiler, GSL, HDF5,
  platform, git commit); `verify` warns loudly when it does not match.
  Cross-platform hash equality is *not* expected (different libm, FMA
  behavior with `-march=native`).
- Manifests for the default sweep configs (`tests/regression_baseline.sh`,
  currently `input/millennium.par`) are committed in `tests/baseline/`.
  Slow optional configs (e.g. microUchuu, ~17 GB output) are captured
  locally on demand and gitignored.

## Option-combination coverage

The physics option matrix (8 SF prescriptions x FIRE x FFB x CGM x AGN x
H2 options x ...) is far larger than what any baseline can cover.
Out-of-range flag values and physically meaningless combinations are
rejected at startup by `read_parameter_file()`; everything else runs, but
**only the combinations exercised by a baseline carry a bit-identical
guarantee**:

| Config | Exercises |
|---|---|
| `input/millennium.par` (default sweep) | SFprescription=1 (BR06), AGNrecipeOn=2 (Bondi), CGMrecipeOn=1, FIREmodeOn=1, FeedbackFreeModeOn=1 (Li+24 sigmoid), HDF5 output |
| `input/microuchuu.par` (release spot check) | same physics on Uchuu100 trees: CGM + FFB + FIRE at higher resolution |
| `input/millennium_all.par` (`tests/baseline/millennium_all/`) | extended output-field coverage |
| `tests/benchmark/benchmark.par` | `sage_binary` output writer, MPI build |

Changes to code paths outside this coverage (other SF prescriptions, AGN
recipes 0/1/3, FFB modes 2-7, ...) are guarded only by the unit tests in
`tests/`; treat them with proportionally more suspicion during review.

## Determinism

Two consecutive serial runs of the same binary on the same machine produce
bit-identical datasets. If `verify` fails immediately after a `capture`
with no code change, suspect the build (stale objects, changed flags)
before suspecting the harness.

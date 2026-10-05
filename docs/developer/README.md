# SAGE26 Developer Guide

For contributors working on SAGE26 itself: conventions, internals, and the
checks a change has to pass. End-user documentation (installing, running,
parameters, output format) lives one level up in [`docs/`](../) and at
[sage26.readthedocs.io](https://sage26.readthedocs.io/en/latest/).

## Start here

| Document | What it covers |
|---|---|
| [REGRESSION_BASELINE.md](REGRESSION_BASELINE.md) | The bit-identical output guarantee and how to verify or re-capture it. Read this before changing anything under `src/`. |
| [CLEANUP_PLAN.md](CLEANUP_PLAN.md) | The pre-release hygiene pass: phases, sequencing, and the physics-preservation invariant. |

## Style guides

| Document | Scope |
|---|---|
| [STYLE_C.md](STYLE_C.md) | Every `.c` and `.h` file under `src/` and `tests/`. |
| [STYLE_DOCS.md](STYLE_DOCS.md) | Every `.md` file, plus the top-level `README.md`. |
| [STYLE_TESTS.md](STYLE_TESTS.md) | The unit-test suite under `tests/`. |
| [STYLE_COMMITS.md](STYLE_COMMITS.md) | Commit messages. |

## Investigations and findings

| Document | Subject |
|---|---|
| [RUBRIC_SCORES_shark.md](RUBRIC_SCORES_shark.md) | Scoring of [shark](https://github.com/ICRAR/shark) as an exemplar codebase; the style guides are derived from it. |
| [DYNAMIC_TIMESTEP_CONVERGENCE.md](DYNAMIC_TIMESTEP_CONVERGENCE.md) | Convergence of the adaptive substepping scheme. |
| [SATELLITE_STRIPPING_FINDINGS.md](SATELLITE_STRIPPING_FINDINGS.md) | Satellite stripping behaviour. |

## Before you commit

```bash
make clean && make USE-MPI=              # serial build
./tests/regression_baseline.sh           # datasets must be bit-identical
make tests                               # all unit-test suites
```

Note that `make tests` depends on the `sage` target and relinks it with MPI,
so run the regression baseline first.

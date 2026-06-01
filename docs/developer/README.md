# SAGE26 Developer Documentation

This directory holds the developer-facing documents that govern the **pre-release code hygiene pass** for SAGE26. It is for contributors, not end users. End-user docs (install, run, cite) live in the top-level `README.md`.

## Status

Planning and scaffolding only. Implementation of the cleanup pass is **blocked** until the user signals that the science version of SAGE26 is frozen. Nothing in `src/`, `plotting/`, `tests/`, or `input/` should be touched on hygiene grounds until that signal is given.

## Documents

| File | Purpose | Phase |
|------|---------|-------|
| [CLEANUP_PLAN.md](CLEANUP_PLAN.md) | Master plan with the full phased todo list | All |
| [REGRESSION_BASELINE.md](REGRESSION_BASELINE.md) | How the bit-identical regression baseline is captured and checked | Phase 0 |
| [RUBRIC.md](RUBRIC.md) | Rubric used to evaluate exemplar codebases and derive our style guidelines | Phase 1 |
| [STYLE_C.md](STYLE_C.md) | C source code conventions | Phase 1 → 2 |
| [STYLE_PYTHON.md](STYLE_PYTHON.md) | Python (plotting + bindings) conventions | Phase 1 → 2 |
| [STYLE_DOCS.md](STYLE_DOCS.md) | Markdown docs and README conventions | Phase 1 → 2 |
| [STYLE_TESTS.md](STYLE_TESTS.md) | Test file conventions (C + integration shell) | Phase 1 → 2 |
| [STYLE_COMMITS.md](STYLE_COMMITS.md) | Git commit message conventions | Phase 1 → 2 |

## Workflow

1. Wait for science freeze.
2. Land Phase 0 (baseline) on the frozen commit. Nothing else proceeds without it.
3. Fill in [RUBRIC.md](RUBRIC.md), score exemplars (shark first), distill into the `STYLE_*.md` files.
4. Apply style guides file-by-file. Regression baseline + test suite must pass after every batch.
5. Land release-surface docs (top-level README, CITATION.cff, LICENSE check, CONTRIBUTING.md, CHANGELOG.md).
6. Optional Phase 4: CI.

## The physics-preservation invariant

**Cleanup commits must not change the physics.** Every dataset SAGE26 writes today must continue to be writable, bit-for-bit identical at the per-dataset level, by every commit landed during the cleanup pass. The regression baseline ([REGRESSION_BASELINE.md](REGRESSION_BASELINE.md)) enforces this; every style-guide cleanup checklist ends with "regression baseline passes" for exactly this reason. See [CLEANUP_PLAN.md](CLEANUP_PLAN.md#the-physics-preservation-invariant) for the full statement.

## Hard rules during cleanup

- No camelCase ↔ snake_case renames or other repo-wide cosmetic mass changes.
- No "while I'm here" refactors. One batch = one stated goal.
- Every batch ends with a green regression baseline before commit.
- Bug fixes are allowed but must be **explicitly labelled** `fix:` in the commit message — they are the only legitimate source of output drift during cleanup, and they require a follow-up re-baseline commit (STYLE_COMMITS.md §4).

# Test Style Guide

**Status: stub.** This file will be filled in during Phase 1 of the [CLEANUP_PLAN](CLEANUP_PLAN.md), after the [RUBRIC](RUBRIC.md) has been applied to exemplar codebases. Do not invent rules here speculatively.

## Scope

Every test file under `tests/`, plus the integration shell scripts (`run_integration_tests.sh`, the planned `regression_baseline.sh`). Both C tests and shell-driven integration tests are in scope.

## What this guide will cover

Anticipated sections (subject to revision based on rubric scoring):

- **Test file layout.** Standard structure for a C test source file: header comment (what is being validated, why), includes, fixture setup, test functions, `main()`. Use the existing `tests/test_framework.h` macros (`ASSERT_CLOSE`, `ASSERT_IN_RANGE`, etc.) — no rolling new assertion helpers.
- **Naming.** Test source files (`test_*.c`), test functions (`test_<thing>_<scenario>`), one assertion per logical check.
- **What a test must document.** Every test function names: (a) the function or behaviour under test, (b) the input scenario, (c) the expected outcome, (d) the physical or numerical reason that outcome is correct. Reviewers should be able to evaluate the test without re-deriving the physics.
- **Tolerances.** Numerical comparisons use justified tolerances, not magic epsilons. The tolerance is a constant with a comment explaining its origin (floating-point round-off, expected physical scatter, etc.).
- **Adding a new test.** Required steps: new entry in `tests/Makefile`, new entry in `run_integration_tests.sh`, runs cleanly under `make tests` from the repo root.
- **Regression baseline tests.** Lives in `tests/regression_baseline.sh` (Phase 0). Treated as the ground truth for "does this commit change behaviour". Distinct from unit-style physics tests, which validate specific prescriptions.
- **Test data.** Where fixture tree files or expected-output snapshots live, how large they are allowed to be, what gets gitignored vs committed.
- **Discouraged patterns.** Tests that pass without exercising the code path. Tests that depend on machine-specific paths. Tests that only print and never assert.

## What this guide will not do

- Force adoption of a new test framework. The existing `test_framework.h` macros stay.
- Require 100% coverage. Coverage targets, if any, will be set by the rubric pass.

## Examples policy

Same as the other style guides: rules without examples don't go in. Pull examples from existing well-written tests in `tests/` where possible.

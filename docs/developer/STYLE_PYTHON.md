# Python Style Guide

**Status: stub.** This file will be filled in during Phase 1 of the [CLEANUP_PLAN](CLEANUP_PLAN.md), after the [RUBRIC](RUBRIC.md) has been applied to exemplar codebases. Do not invent rules here speculatively.

## Scope

Every `.py` file under `plotting/` and any Python bindings (cffi glue, helper scripts). Tests live under `tests/` and may have additional conventions in [STYLE_TESTS.md](STYLE_TESTS.md).

## What this guide will cover

Anticipated sections (subject to revision based on rubric scoring):

- **PEP 8 baseline.** `flake8` is already configured in `.flake8` and runs as a pre-commit hook. Codify the existing ignores (E501, F401) and explain why each is allowed.
- **Module-level docstring.** Required at the top of every script and module: one-line summary, longer description if needed, usage example for executable scripts.
- **Import ordering.** Standard library → third-party → local. Within each group, sorted. (A tool like `isort` may be adopted to enforce mechanically.)
- **Path handling.** No hardcoded absolute paths inside scripts. Output directories are passed as arguments or read from the parameter file.
- **Plotting conventions.** Centralised figure size, font, colour palette. A single helper module (e.g. `plotting/style.py`) so paper figures are visually consistent.
- **Numerical conventions.** Unit handling matches the C side (10^10 M_sun/h, Mpc/h, km/s) unless explicitly converting for display — and the conversion is commented.
- **Script invocation.** Every executable script accepts `--help` and documents its arguments. No silent defaults that depend on cwd.
- **Discouraged patterns.** Surprises uncovered during rubric pass (e.g. mutable default arguments, bare `except:`).

## What this guide will not do

- Force a type-annotation campaign. Annotations may be encouraged but not required.
- Force a black/ruff reformat. flake8 stays the source of truth unless explicitly upgraded.
- Touch the cffi-generated `libsage.so` bindings.

## Examples policy

Same as [STYLE_C.md](STYLE_C.md): every rule comes with a before/after example or it doesn't go in.

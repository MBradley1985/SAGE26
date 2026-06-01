# C Style Guide

**Status: stub.** This file will be filled in during Phase 1 of the [CLEANUP_PLAN](CLEANUP_PLAN.md), after the [RUBRIC](RUBRIC.md) has been applied to exemplar codebases. Do not invent rules here speculatively.

## What this guide will cover

Once populated, this guide governs every `.c` and `.h` file under `src/` and `tests/`. Anticipated sections (subject to revision based on rubric scoring):

- **File header.** Standard top-of-file block: purpose, primary references (papers), authorship/license header if adopted.
- **Header guards.** Convention (`#pragma once` vs `#ifndef GUARD` blocks) and naming.
- **Include order.** System headers, third-party, project headers.
- **Naming.** Conventions for types, functions, macros, file-scope statics. **No mass renames** — codify what already exists where it's consistent, only flag exceptions.
- **Error handling.** Use the existing macros (`XASSERT`, `XRETURN`, `CHECK_STATUS_AND_RETURN_ON_FAIL`, `CHECK_POINTER_AND_RETURN_ON_NULL`, `ABORT`) consistently. Forbid raw `assert()` in new code.
- **Comments.** When to comment a function (non-trivial physics, non-obvious invariants), what a comment must contain (the *why*, not the *what*), what is forbidden (dead-code comments, commented-out code blocks, change-log comments inline).
- **Magic numbers.** When to lift them into `macros.h` vs leave inline. Document the source of physical constants.
- **Formatting.** Brace style, indentation, max line length, blank-line conventions. A `clang-format` config file may be added so this is enforced mechanically rather than by review.
- **Public vs static.** Every function not declared in a header must be `static`. Header order: types → constants → public function declarations.
- **Discouraged patterns.** Things found during the rubric pass that we want to phase out (e.g. silent failure paths, mixed unit conversions).

## What this guide will not do

- Force camelCase ↔ snake_case sweeps. Existing names stay.
- Force signature changes. Public API is frozen at the baseline commit.
- Prescribe architectural patterns. This is hygiene, not redesign.

## Examples policy

Every rule must come with a short worked example: a *before* (drawn from the current codebase if it illustrates the rule cleanly) and an *after*. Rules without an example are not enforceable.

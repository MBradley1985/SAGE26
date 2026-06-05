# SAGE26 Pre-Release Cleanup Plan

This is the master plan for the code-hygiene pass that prepares SAGE26 for paper submission and public release. It is sequenced strictly: each phase must complete before the next begins.

## Context

SAGE26 has been in a deliberate "move fast, explore ideas" phase. The codebase reflects that — and the user has explicitly stated this is **not a criticism**. Public release simply raises the bar for what outside readers, reviewers, and future users see. This cleanup brings the repo to a pseudo-professional standard without changing the science.

## Sequencing constraint

Cleanup work is **blocked** until the user signals that the science version of SAGE26 is frozen. Until that signal, this directory holds plans only — no files outside `docs/developer/` should be touched on hygiene grounds.

## The physics-preservation invariant

**The cleanup pass MUST NOT change the physics.** Every dataset SAGE26 writes today must continue to be writable, byte-for-byte identical at the per-dataset level, by every commit landed during the cleanup. This is the single most important rule of the cleanup pass; it is non-negotiable.

How it is enforced:

1. The **regression baseline** at [REGRESSION_BASELINE.md](REGRESSION_BASELINE.md) hashes every dataset in every output HDF5 file and verifies them against the frozen `baseline/pre-cleanup` reference. Any drift fails the verify.
2. Every style-guide cleanup checklist ends with "regression baseline passes". A cleanup commit that fails the baseline is not a cleanup commit — it is a behavioural change, and must be reverted (or relabelled as a bug fix per STYLE_COMMITS.md §4).
3. The only legitimate source of output drift during the cleanup is a commit **explicitly labelled `fix:`** with a body describing what changed numerically and why the previous behaviour was wrong. That commit, and the re-baseline that follows it, are visible in `git log` for any future reader.

What this means in practice: cleanup commits can rename files, lift magic numbers to named constants, rewrite comments, add file headers, swap `#ifndef` guards for `#pragma once`, reorganise include blocks, and apply formatting rules. None of those operations can alter the math. If a cleanup commit triggers a baseline failure, the safe move is always to revert and investigate — never to "fix the baseline" by re-capturing it.

## Scope decisions

- **Bit-identical regression policy.** Baseline outputs must match the frozen reference bit-for-bit at the dataset level. (HDF5 container bytes vary between runs; the verifier records that as informational, not a failure — see REGRESSION_BASELINE.md.)
- **Serial-first.** All baseline runs and verification use the serial (non-MPI) build. MPI parity is a bonus to be addressed after the serial cleanup is complete. Do not let MPI considerations leak into Phase 0–3 decisions.
- **No invasive renames.** No camelCase ↔ snake_case sweeps, no signature changes, no scope creep. Cleanup means consistent formatting, frontmatter, commenting, and documentation — not architectural change.

## Phase 0 — Regression baseline (blocks everything)

See [REGRESSION_BASELINE.md](REGRESSION_BASELINE.md) for the full spec.

- [ ] Freeze a "final" SAGE26 commit as the reference point. Tag it `baseline/pre-cleanup`.
- [ ] Pick golden parameter files. Default sweep: `input/millennium.par` (Mini-Millennium classical regime, ~11 s, 1.5 GB output). `input/microuchuu.par` is verified bit-identical but kept as a manual release-time spot check rather than per-commit, because it takes ~255 s and writes ~17 GB.
- [ ] Capture baseline outputs from the serial build: all HDF5 datasets per `model_N.hdf5`, exact byte hashes.
- [ ] Capture key derived statistics (z=0 SMF, BHMF, SFRD history) for fast smoke-testing.
- [ ] Write `tests/regression_baseline.sh` — runs the golden configs serially, diffs against captured outputs bit-for-bit, returns nonzero on any drift.
- [ ] Document the rule: the regression baseline must pass before any cleanup commit lands. Conversations end with a baseline run.

## Phase 1 — Rubric and style guidelines

- [ ] Fill in [RUBRIC.md](RUBRIC.md) — define what we evaluate exemplar codebases against.
- [ ] Score exemplars 1–5 per rubric line. Start with [shark](https://github.com/ICRAR/shark). Add 1–2 more if useful (candidates: meraxes, l-galaxies, galform).
- [ ] Use the rubric scores to identify what our own style guides should cover. Don't write rules we can't motivate.
- [ ] Draft each `STYLE_*.md`. Keep them tight — bounce between models, prune ruthlessly, don't over-engineer.
- [ ] Pilot the guidelines on a small branch: one C source file + one plotting script. Verify the rules make sense in practice before going wide.

## Phase 2 — Apply the style guides

- [ ] Group source files by area (e.g. cooling/heating, infall, star formation, mergers, I/O, plotting). One group per batch.
- [ ] For each batch: reference the relevant `STYLE_*.md`, apply minimal changes to bring files into compliance.
- [ ] After every batch: run the regression baseline (Phase 0). Commit only if it passes bit-identically.
- [ ] If the baseline fails, the batch contained a real behavioural change — investigate. Either it's an accidental change (revert), or it's a bug fix (separate commit, explicitly labelled).
- [ ] Hard rule: one batch = one stated goal. No bundling unrelated cleanup.

## Phase 2B — Header hygiene, dead code, magic numbers

Three targeted batches addressing style-guide items not covered in Phase 2.

### Batch 2B-1: Header file hygiene
- `core_allvars.h`: `#ifndef`/`#define`/`#endif` guard → `#pragma once`; add `extern "C"` wrap; add §1 file header; fix 31 non-ASCII bytes.
- `core_simulation.h`: add `#pragma once`; add `extern "C"` wrap; add §1 file header.
- `macros.h`: already has `#pragma once`; update old `/* File: */` header to §1 format. No `extern "C"` (no function declarations).
- `core_utils.h`, `progressbar.h`: third-party Corrfunc origin — leave copyright headers unchanged.
- Run regression baseline; commit if bit-identical.

### Batch 2B-2: Dead code removal
- `sage.c:376`: investigate and remove `#if 0` block if truly dead.
- `progressbar.h:18-20`: remove stray `#if 0 } #endif` (dead brace from extern "C" mistake).
- Run regression baseline; commit if bit-identical.

### Batch 2B-3: Magic numbers → named constants
- Scan each physics `.c` file for inline literals whose meaning isn't obvious from context.
- Lift to file-scope `static const` with a comment citing the paper and equation number.
- Mathematical constants (`0.5`, `2.0`, `4.0/3.0`, etc.) are **not** lifted — only empirical coefficients and fit parameters.
- Run regression baseline; commit if bit-identical.

## Phase 3 — Release surface

- [ ] Rewrite top-level `README.md` for end users: install, quickstart, citation, link to paper.
- [ ] Add `CITATION.cff` for academic citation.
- [ ] Confirm `LICENSE` exists and is correct.
- [ ] Add `CONTRIBUTING.md` describing how outside contributors should engage.
- [ ] Start `CHANGELOG.md` from the `baseline/pre-cleanup` tag forward.
- [ ] Create one canonical parameter reference doc covering every switch in `input/*.par` (currently scattered across project notes and inline comments).

## Phase 4 — CI (optional but recommended)

- [ ] GitHub Actions: build matrix (serial only initially; add MPI later), run test suite, run flake8, run a reduced regression baseline against a tiny merger tree.
- [ ] Tag the release commit.
- [ ] Archive on Zenodo for a DOI to cite in the paper.

## Out of scope

Anything that changes scientific outputs (unless it is an explicitly labelled bug fix found during cleanup). Anything MPI-specific until Phase 0–3 are done in serial. Adding new features. Refactors. Architectural changes.

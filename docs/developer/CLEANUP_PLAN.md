# SAGE26 Pre-Release Cleanup Plan

This is the master plan for the code-hygiene pass that prepares SAGE26 for paper submission and public release. It is sequenced strictly: each phase must complete before the next begins.

## Context

SAGE26 has been in a deliberate "move fast, explore ideas" phase. The codebase reflects that — and the user has explicitly stated this is **not a criticism**. Public release simply raises the bar for what outside readers, reviewers, and future users see. This cleanup brings the repo to a pseudo-professional standard without changing the science.

## Sequencing constraint

Cleanup work is **blocked** until the user signals that the science version of SAGE26 is frozen. Until that signal, this directory holds plans only — no files outside `docs/developer/` should be touched on hygiene grounds.

## Scope decisions

- **Bit-identical regression policy.** Baseline outputs must match the frozen reference bit-for-bit, not within tolerance. The only legitimate sources of drift during cleanup are commits explicitly labelled as bug fixes.
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

## Phase 3 — Release surface

- [ ] Rewrite top-level `README.md` for end users: install, quickstart, citation, link to paper.
- [ ] Add `CITATION.cff` for academic citation.
- [ ] Confirm `LICENSE` exists and is correct.
- [ ] Add `CONTRIBUTING.md` describing how outside contributors should engage.
- [ ] Start `CHANGELOG.md` from the `baseline/pre-cleanup` tag forward.
- [ ] Create one canonical parameter reference doc covering every switch in `input/*.par` (currently scattered across CLAUDE.md and comments).

## Phase 4 — CI (optional but recommended)

- [ ] GitHub Actions: build matrix (serial only initially; add MPI later), run test suite, run flake8, run a reduced regression baseline against a tiny merger tree.
- [ ] Tag the release commit.
- [ ] Archive on Zenodo for a DOI to cite in the paper.

## Out of scope

Anything that changes scientific outputs (unless it is an explicitly labelled bug fix found during cleanup). Anything MPI-specific until Phase 0–3 are done in serial. Adding new features. Refactors. Architectural changes.

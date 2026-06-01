# Commit Message Style Guide

**Status: stub.** This file will be filled in during Phase 1 of the [CLEANUP_PLAN](CLEANUP_PLAN.md), after the [RUBRIC](RUBRIC.md) has been applied to exemplar codebases. Do not invent rules here speculatively.

## Scope

Every commit message on the `main` and `dev` branches from the `baseline/pre-cleanup` tag forward. Pre-existing history is not rewritten.

## What this guide will cover

Anticipated sections (subject to revision based on rubric scoring):

- **Subject line.** Concise (target ≤ 72 characters), imperative mood ("add", "fix", "rename" — not "added" or "adds"), no trailing period.
- **Optional prefix.** Whether to adopt a Conventional Commits style (`feat:`, `fix:`, `docs:`, `style:`, `refactor:`, `test:`, `chore:`) or a simpler scope tag (`[cooling] ...`). Decision pending rubric pass.
- **Body.** When required (anything non-trivial), what it must contain (the *why*, not the *what* — the diff shows the what), wrapped at a reasonable width.
- **Bug fixes during cleanup.** Commits that change scientific outputs during the cleanup pass **must** be labelled as bug fixes in the subject line. They are the only legitimate source of regression-baseline drift. The body must describe what changed in the output and why the previous behaviour was wrong.
- **Cleanup commits.** Style-only commits state the style doc they enforce ("apply STYLE_C.md to cooling module") and the scope (which files / which area). The body confirms the regression baseline passed bit-identically.
- **Reference linking.** When a commit closes an issue, references a paper, or implements a parameter file change, the body links to it.
- **Co-authorship and tooling.** When AI assistance is used heavily for a commit, the convention for attribution.

## What this guide will not do

- Rewrite existing commit history.
- Enforce a hard line on commit size — judgement applies, but the rubric pass may suggest soft guidance.

## Examples policy

Include 3–5 worked examples drawn from the kinds of commits expected during cleanup: a pure style commit, a labelled bug-fix commit, a doc-only commit, and a phase-completion commit (e.g. "land Phase 0 regression baseline").

# Documentation Style Guide

**Status: stub.** This file will be filled in during Phase 1 of the [CLEANUP_PLAN](CLEANUP_PLAN.md), after the [RUBRIC](RUBRIC.md) has been applied to exemplar codebases. Do not invent rules here speculatively.

## Scope

Every `.md` file in the repository, plus the top-level `README.md`. Includes:

- The end-user README at the repository root.
- Developer docs in this directory (`docs/developer/`).
- Any future user-facing documentation under `docs/`.
- `CONTRIBUTING.md`, `CHANGELOG.md`, `CITATION.cff` (when those land in Phase 3).

## What this guide will cover

Anticipated sections (subject to revision based on rubric scoring):

- **Audience separation.** End-user docs (install, run, cite) vs developer docs (architecture, conventions, internals). Each file states its audience in the first sentence.
- **README structure.** Canonical section order for the top-level README: what is it → install → quickstart → cite → links to deeper docs.
- **Markdown formatting.** Heading hierarchy (one H1 per file = the title), code-block language tags required, tables for structured data, link style (reference vs inline).
- **Frontmatter.** Whether to use YAML frontmatter on docs (likely no for this repo — but the decision is documented).
- **References to code.** When a doc names a source file, function, or parameter, link to it (relative path) or quote the exact symbol. Stale references are a doc bug.
- **Physics and parameter docs.** Every parameter in `input/*.par` is documented in one canonical place, with: name, allowed values, units, default, brief description, paper reference where applicable. No more parameter docs scattered between CLAUDE.md, code comments, and example .par files.
- **Tone.** Direct, present tense, no marketing language. "SAGE26 reads N-body merger trees and writes galaxy catalogues" — not "SAGE26 is a powerful tool that enables researchers to...".
- **Image and figure policy.** Where figures live, naming convention, whether to commit binary images or generate them from scripts.

## What this guide will not do

- Mandate a documentation generator (e.g. Sphinx, MkDocs) unless the rubric pass surfaces a clear need.
- Force translation of CLAUDE.md content into separate user docs wholesale — CLAUDE.md remains the AI-facing onboarding doc.

## Examples policy

Same as the other style guides: rules without examples don't go in.

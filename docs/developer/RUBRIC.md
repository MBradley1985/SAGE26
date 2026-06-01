# Code Hygiene Rubric

This rubric is the tool we use to **derive** our own style guidelines. We do not invent guidelines in a vacuum — instead we score exemplar scientific codebases against this rubric, see what they do well and where they fall short, and let those scores tell us what our own `STYLE_*.md` files should cover.

## How to use

1. Score each exemplar repository on each rubric line, 1–5:
   - **1** — absent or actively confusing
   - **2** — partial, inconsistent
   - **3** — present and adequate
   - **4** — strong, with examples
   - **5** — exemplary; we should imitate this directly
2. Record one-line evidence for each score (file path, link, or quote).
3. After scoring, look at the lines where exemplars cluster at 4–5 — those are the patterns worth adopting. Lines where everyone scores low are areas where we either don't care or should define our own answer.
4. Translate the surviving lines into concrete rules in the `STYLE_*.md` files.

## Exemplars to score (in priority order)

1. **shark** — <https://github.com/ICRAR/shark>. C++ SAM in the same scientific niche. First and most important comparison.
2. (TBD) one of: meraxes, l-galaxies, galform — pick one whose ergonomics the user already likes.
3. (Optional) a non-astro reference for "what good looks like" in scientific C, e.g. GADGET-4 or AREPO if accessible.

## Rubric

### Repository surface

| # | Criterion | Why it matters |
|---|-----------|----------------|
| R1 | Top-level README explains in one screen what the code is, who it's for, and how to cite | First contact for any new user |
| R2 | Build instructions are complete and current | New users get stuck here first |
| R3 | Quickstart actually runs end-to-end as written | Trust signal |
| R4 | LICENSE file present and unambiguous | Required for academic reuse |
| R5 | CITATION.cff or equivalent | Required for academic credit |
| R6 | CONTRIBUTING.md describes how to engage | Lowers barrier for outside contributions |
| R7 | CHANGELOG.md tracks user-visible changes | Lets users understand version differences |

### C source conventions

| # | Criterion | Why it matters |
|---|-----------|----------------|
| C1 | Consistent file header (purpose, references) at top of every source file | Helps readers orient |
| C2 | Consistent function-level comments at non-trivial functions | Reduces re-reading cost |
| C3 | Single, consistent error-handling idiom | Mixing styles invites bugs |
| C4 | Header guards / `#pragma once` used consistently | Compile reliability |
| C5 | Public vs static functions clearly distinguished | API surface clarity |
| C6 | No commented-out dead code | Indicates discipline |
| C7 | Magic numbers extracted to named constants | Readability and provenance |
| C8 | Consistent formatting (brace style, indentation, line length) | Diff noise reduction |

### Python conventions

| # | Criterion | Why it matters |
|---|-----------|----------------|
| P1 | PEP 8 compliance via configured linter | Industry baseline |
| P2 | Module-level docstring on every file | Orientation |
| P3 | Consistent import ordering | Diff noise reduction |
| P4 | No vendored data paths hardcoded inside scripts | Reproducibility |
| P5 | Plotting style (figure size, fonts, colour) centralised | Visual consistency |

### Tests

| # | Criterion | Why it matters |
|---|-----------|----------------|
| T1 | Tests are runnable with one command from the repo root | New contributor onboarding |
| T2 | Test files follow a discoverable naming convention | Findability |
| T3 | Each test has a clear "what is being validated" comment | Maintenance |
| T4 | Regression / reproducibility tests exist | Trust under refactor |
| T5 | CI runs the tests | Tests that don't run rot |

### Documentation

| # | Criterion | Why it matters |
|---|-----------|----------------|
| D1 | Physics modules are documented (what's modelled, key references) | Scientific transparency |
| D2 | Every parameter file switch is documented in one canonical place | Reproducibility |
| D3 | Output format documented (every dataset, units, meaning) | Downstream analysis |
| D4 | Developer docs separate from user docs | Reader-appropriate detail |
| D5 | Markdown formatting consistent (headings, code blocks, tables) | Polish |

### Commits and version control

| # | Criterion | Why it matters |
|---|-----------|----------------|
| V1 | Commit messages follow a consistent convention | Skimmable history |
| V2 | Commits are scoped (one concern per commit) | Bisectability |
| V3 | Branches are pruned after merge | Repo cleanliness |
| V4 | Generated artefacts are gitignored | No noise commits |

## Scoring sheet template

When scoring an exemplar, copy this block into a new file (e.g. `RUBRIC_SCORES_shark.md`):

```
# Exemplar: <name>
Repository: <url>
Scored on: <date>
Commit reviewed: <sha>

## Repository surface
R1: <1-5> — <evidence>
R2: <1-5> — <evidence>
...

## C source conventions
C1: <1-5> — <evidence>
...
```

## Translating scores into guidelines

A rubric line is worth a rule in our style guides when **at least one exemplar scores 4+ on it and we can articulate why that pattern beats what we currently do**. Lines where no exemplar reaches 4 may indicate the line itself is wrong for our domain — revisit before writing a rule. Resist the temptation to write rules just because the rubric has a row for them.

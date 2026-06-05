# Commit Message Style Guide

Governs every commit on `main` and `dev` from the `baseline/pre-cleanup` tag forward. **Pre-existing history is not rewritten.**

shark's commit log scored 3/5 on convention (V1): no Conventional Commits prefix, but consistent imperative mood and one logical change per commit. SAGE26's recent history is more variable (subjects of `-`, `baseline?`, `PHASE 0`, multi-sentence run-on subjects). The cleanup commit landed at `475179d` ("Phase 0: regression baseline implementation (millennium golden config)") is a reasonable template.

## Out of scope

- Rewriting existing history. The pre-`baseline/pre-cleanup` log stays as-is.
- Hard line on commit size.
- Adopting an automated commit-message linter.

## Rules

### 1. Subject line

- **Imperative mood**: "add", "fix", "rename" — not "added" or "adds".
- **Target ≤ 72 characters**. Hard limit 80.
- **No trailing period**.
- **First word capitalised** (after any optional prefix).

**Before (recent SAGE26):**
```text
-
baseline?
PHASE 0
Hopefully fixed H2 properly now, f_H2 should behave correctly with redshift now.
```

**After:**
```text
Add regression baseline for millennium golden config
Fix H2 fraction redshift dependence
Update microuchuu_Karl parameter defaults
```

### 2. Optional prefix

A small `<kind>:` prefix is encouraged but not required. When used, it must come from this short set:

| Prefix | Meaning |
|--------|---------|
| `fix:` | Bug fix that changes scientific outputs. **Required** for any cleanup-phase commit that fails the regression baseline. |
| `feat:` | New user-visible feature. |
| `docs:` | Documentation-only change. |
| `style:` | Hygiene/formatting; must not change behaviour (regression baseline must stay green). |
| `test:` | Adding or modifying tests; no behaviour change. |
| `refactor:` | Internal restructuring; no behaviour change. |
| `chore:` | Build, dependencies, repo plumbing. |

**Reason:** shark doesn't use prefixes (V1=3) and survives. The reason to adopt them in SAGE26 is the cleanup pass specifically: `fix:` and `style:` need to be visually distinguishable in `git log` so that a future reader can tell at a glance which cleanup commits altered behaviour. Once the cleanup is done, prefixes can stay or fade — judgement.

### 3. Body

Required when the subject isn't self-explanatory (anything beyond a one-line trivial change). The body explains the **why**, not the **what** — the diff shows the what.

- **Wrap at ~72 columns.**
- **Blank line between subject and body.**
- **Reference the issue, paper, or earlier commit if relevant.**

**Template for a non-trivial commit:**
```text
<subject>

Why this change is needed (one paragraph: the underlying problem,
the motivation, the trade-off if any).

What changed at a high level (only if non-obvious from the diff —
typically one or two bullets).

References:
- Closes #N (if applicable)
- See PaperAuthor (Year) for the prescription used
```

### 4. Bug-fix commits during cleanup

A commit that changes scientific outputs during the cleanup pass **must**:

- Use the `fix:` prefix in the subject.
- Describe in the body what changed numerically and why the previous behaviour was wrong.
- Be followed by a re-capture of the regression baseline (separate commit):
  ```bash
  bash tests/regression_baseline.sh --capture
  git commit -m "chore: re-baseline after fix(<area>): <one-line>"
  ```
- The re-baseline commit's body references the bug-fix commit's SHA.

**Reason:** Bug fixes are the only legitimate source of regression-baseline drift during cleanup. Making them visible in the log is what lets a future reader trust the baseline.

### 5. Cleanup commits

Style-only commits state which style doc they enforce and which files were touched:

```text
style: apply STYLE_C.md to cooling module

Applies the file-header, header-guard, and Unicode-in-comments rules
from docs/developer/STYLE_C.md to:
- src/model_cooling_heating.c
- src/model_cooling_heating.h

Regression baseline (millennium) passes bit-identically.
```

The "Regression baseline ... passes bit-identically" line is required for `style:`, `refactor:`, and `test:` commits during the cleanup pass.

### 6. References and linking

- Issue numbers: `Closes #42` or `Refs #42` in the body.
- Paper references: `See Croton et al. (2016)` or full BibTeX-ish reference if first mention.
- Parameter file changes: subject line says which parameter, body explains why.

## Examples

### Good (cleanup-phase target)

```text
Phase 0: regression baseline implementation (millennium golden config)

Implements the bit-identical regression baseline that gates every cleanup
commit. See docs/developer/REGRESSION_BASELINE.md.

What's new:
- tests/regression_baseline.py: capture/verify engine ...
- tests/regression_baseline.sh: thin driver ...
- tests/baseline/millennium/manifest.json: seed manifest captured at this
  commit.
```

### Good (a hypothetical bug fix)

```text
fix: correct H2 fraction normalisation at high redshift

The H2 fraction in BR06 used a redshift-independent normalisation that
diverged from the original prescription at z > 4. Replaced with the
explicit (1+z)^alpha scaling from Blitz & Rosolowsky (2006), eqn 3.

This changes outputs: the millennium baseline manifest must be
re-captured in the following commit.

Refs: see CHANGELOG.md entry for Phase 2.
```

### Good (a docs-only commit)

```text
docs: add parameter reference for new FFB switches

Adds FeedbackFreeModeOn entries 6 and 7 to docs/parameters.md following
the table format established in Phase 3.
```

### Bad (do not do this)

```text
-
baseline?
some changes but nothing major
Hopefully fixed H2 properly now, f_H2 should behave correctly with redshift now.
```

Subjects with no content, ambiguous markers, or run-on sentences make `git log` unreadable.

## Cleanup checklist

When writing a cleanup-phase commit:

- [ ] Subject ≤ 72 chars, imperative, no trailing period.
- [ ] Prefix is one of the approved set if used.
- [ ] Body present unless the subject is genuinely self-explanatory.
- [ ] If the commit altered scientific outputs: `fix:` prefix, body explains the change, re-baseline commit follows.
- [ ] If the commit is `style:` / `refactor:` / `test:`: body confirms regression baseline passes bit-identically.

# Documentation Style Guide

Governs every `.md` file in the repository plus the top-level `README.md`. Derived from rubric scoring of shark ([RUBRIC_SCORES_shark.md](RUBRIC_SCORES_shark.md)) plus SAGE26-specific decisions.

shark's strongest pattern was its **canonical parameter reference** (D2=5) and **auto-generated output documentation** (D3=5). SAGE26 currently scatters parameter info between project notes, comments in `input/*.par`, and source code -- converging that is the highest-leverage Phase 3 task.

## Out of scope

- Adopting Sphinx, MkDocs, or any other generator. Markdown in-repo, rendered by GitHub, is enough for SAGE26's audience.
- Inventing a YAML frontmatter scheme. None of the existing docs need it.

## Rules

### 1. Audience separation

Every documentation file states its audience in the first sentence or paragraph.

| Location | Audience |
|----------|----------|
| `README.md` (root) | End users: install, run, cite. |
| `docs/developer/` | Contributors: architecture, conventions, internals. |
| `docs/` (other) | End users: deeper usage docs (parameter reference, output format, physics notes). |

**Reason:** A reader who lands in the wrong file should know within the first paragraph. shark scores 3/5 on this — they don't separate dev from user docs. We already have `docs/developer/`; codifying the split.

### 2. README structure (root)

The top-level `README.md` follows this section order. Add sections, but don't reorder:

1. **One-line description** — what SAGE26 is.
2. **Install** — dependencies, build.
3. **Quickstart** — minimal end-to-end run.
4. **Citation** — BibTeX entry, ADS link.
5. **Links** — pointers to deeper docs (`docs/`, paper, repository).
6. **License** — one line + link to LICENSE.

shark's README is 34 lines and hits all of this. Brevity is a feature.

### 3. Markdown formatting

- **One H1 per file**, used as the title. Never more.
- **Code blocks always have a language tag.** ` ```c `, ` ```python `, ` ```bash `, ` ```text ` for plain.
- **Tables** for structured data (parameter lists, file inventories). Don't use bullet lists with colons when a table fits.
- **Links** are inline (`[text](url)`), not reference-style.
- **Hard wrap at 100 columns** or use one-sentence-per-line. Pick one per file; don't mix.

**Reason:** GitHub renders most things, but missing language tags lose syntax highlighting; mixing one-sentence-per-line with paragraph wrapping makes diffs hostile.

### 4. References to code

When a doc names a source file, function, or parameter, link to it (relative path) or quote the exact symbol. Stale references are a doc bug — they get caught when a file is renamed and the link breaks.

**Before:**
> The cooling function is in the cooling file.

**After:**
> The cooling function is `cooling_recipe_cgm` in [`src/model_cooling_heating.c`](../../src/model_cooling_heating.c).

### 5. Physics and parameter docs

Every parameter accepted in `input/*.par` is documented in **one canonical place** (planned for Phase 3 as `docs/parameters.md`). Each parameter entry includes:

- **Name** (matching the .par file exactly).
- **Allowed values** (list, range, or "any positive float").
- **Units** (code units; conversion to physical units if non-obvious).
- **Default** (what value the code uses if the .par file omits the line).
- **Description** — one to three sentences. Says *what the parameter controls*, not what physics module it belongs to.
- **Paper reference** if the parameter comes from a specific paper.

**Reason:** shark's `doc/configuration/names.rst` is the model here (D2=5). SAGE26's parameter docs currently exist in two places: `% comment` lines inside .par files, and constants in `core_read_parameter_file.c`. A reader who wants to understand a parameter has to triangulate. One canonical file ends that.

The .par files themselves keep their inline `% short comment` for at-a-glance reference, but the canonical reference is the markdown doc.

### 6. Output format documentation

Walk every HDF5 file once, emit a `docs/output_format.md` listing every group, every dataset, dtype, shape semantics, and meaning. `tests/regression_baseline.py` already walks every dataset to compute hashes — extending it to emit the doc is ~30 lines.

**Reason:** shark generates this from the HDF5 files themselves (D3=5) so the docs cannot drift from the code. Same approach is feasible for SAGE26. Not blocking for Phase 2 cleanup; goes in Phase 3.

### 7. Tone

Direct, present tense, no marketing language.

- "SAGE26 reads N-body merger trees and writes galaxy catalogues." ✓
- "SAGE26 is a powerful and flexible tool that enables researchers to..." ✗

**Reason:** Scientific software docs are read by scientists in a hurry. Marketing tone wastes their time.

### 8. Unicode

Unicode (Greek letters, math symbols, superscripts) **is fine** in markdown docs — GitHub renders UTF-8 consistently. Use it for physics formulae.

**Note:** This is the opposite of the rule for C source files ([STYLE_C.md §5](STYLE_C.md)). The reason is different rendering pipelines: GitHub markdown is reliable; IDEs editing source files are not.

### 9. Image and figure policy

- Figures committed under `docs/figures/`.
- One figure per file, named by what it shows.
- Source scripts that generate figures live in `plotting/` or `scripts/`, not interleaved with docs.

Don't commit large binary images (>1 MB) without a reason. Vector formats (SVG, PDF) preferred for diagrams.

## Cleanup checklist (Phase 2 / Phase 3)

When applying this guide to a doc file:

- [ ] First paragraph names the audience.
- [ ] One H1 per file, used as the title.
- [ ] Every code block has a language tag.
- [ ] Tables used where structure is tabular.
- [ ] Links are inline `[text](url)`, relative paths for in-repo references.
- [ ] No marketing tone.
- [ ] If the doc references a file/function/parameter, the reference is correct and links work.
- [ ] If the doc duplicates info that lives in `docs/parameters.md` (when that exists), link instead of duplicating.

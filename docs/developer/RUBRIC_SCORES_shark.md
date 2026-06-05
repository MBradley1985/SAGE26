# Exemplar: shark

Repository: <https://github.com/ICRAR/shark>
Local clone: `/Users/mbradley/Documents/PhD/shark/`
Scored on: 2026-06-01
Version reviewed: 2.0.0 (commit `135b27b0`)

shark is the obvious first comparison: same scientific niche (semi-analytic galaxy formation), comparable scale, mature project (ASCL.net registered, Read the Docs hosted, GitHub Actions CI). Written in C++17, not C, so source conventions don't transfer one-to-one — but structural patterns do.

## Headline observations

Three patterns shark does **very well** that SAGE26 should imitate:

1. **Auto-generated output documentation** (`doc/hdf5_properties/`). The HDF5 dataset reference is generated from the files themselves, so it cannot drift from the code. Exemplary.
2. **Canonical parameter reference** (`doc/configuration/names.rst` + `sample.rst`). One file lists every option, units, defaults, and meaning. SAGE26's equivalent is scattered across project notes, .par comments, and code.
3. **Universal file headers** — every C++/Python/CMake/config file starts with the same ICRAR/GPL block, no exceptions. Cheap to adopt, immediately raises the "this is a serious project" signal.

Three patterns where shark is **adequate but not aspirational**:

- Python plotting scripts lack module docstrings and don't follow PEP 8 strictly.
- No CONTRIBUTING.md.
- No CITATION.cff (BibTeX is in README; works but not the modern standardised format).

One pattern shark uses that's **directly relevant to our Phase 0 work**: their CI runs tests with a **fixed seed** (`TEST_FIXED_SEED: 123456`) for reproducibility, validating that the model is bit-identical under controlled conditions. Same pattern we just landed.

## Repository surface

| # | Score | Evidence |
|---|-------|----------|
| R1 | 4 | `README.md` is 34 lines: one-sentence description, badges (CI × 2, RTD, ASCL), citation BibTeX, link to RTD. Minimalist but complete; the full story lives at <https://shark-sam.readthedocs.io>. |
| R2 | 3 | CMake build, instructions live in `doc/building.rst` not the README. Requires reading external docs to get going. |
| R3 | 3 | `sample.cfg` files exist and are documented, but a real run requires fetching SURFS data not bundled with the repo. Not a `clone && make && run` story. |
| R4 | 5 | `LICENSE` is the full GPLv3 text (35 KB), unambiguous. |
| R5 | 3 | No `CITATION.cff`. BibTeX in README + `doc/index.rst`, ASCL.net entry 1811.005 listed. Functionally equivalent but not the modern standardised format GitHub will surface. |
| R6 | 1 | No `CONTRIBUTING.md` anywhere in the tree. |
| R7 | 4 | `doc/changelog.rst` has version sections with bulleted user-visible changes. Not at repo root but discoverable via docs. |

## C source conventions (shark is C++; lessons transfer)

| # | Score | Evidence |
|---|-------|----------|
| C1 | 5 | **Every** `.cpp` and `.h` starts with the identical 17-line ICRAR/GPL header followed by `/** @file */`. Zero exceptions across the source tree. |
| C2 | 3 | Doxygen `@file` marker is universal; in-function comments are sparse and inconsistent (e.g. `// relevant for Lagos 23 model` in `agn_feedback.cpp` — useful but not systematic). |
| C3 | 4 | Throws typed exceptions (`invalid_option`, etc.) consistently, defined in `include/exceptions.h`. No mixing with C-style return codes. |
| C4 | 5 | Headers consistently use `#ifndef INCLUDE_<FILENAME>_H_` guards (e.g. `INCLUDE_AGN_FEEDBACK_H_`). Never `#pragma once`. |
| C5 | 4 | All code in `namespace shark { ... }`. Classes use explicit `public:` sections. Free functions in `utils.h` etc., headers are deliberate about what's exposed. |
| C6 | 4 | Sampled files contain no commented-out code blocks. |
| C7 | 3 | Some inline numeric literals in physics code (`std::pow(alpha_adaf, 2.0)`, `0.001`, `0.0005` in `agn_feedback.cpp::AGNFeedbackParameters`). Mixed — not lifted to named constants. |
| C8 | 4 | Consistent tab indentation, K&R braces, reasonable line lengths. No visible `.clang-format` but the result is uniform — likely strong review discipline. |

## Python conventions

| # | Score | Evidence |
|---|-------|----------|
| P1 | 2 | No `flake8`/`ruff`/`black` config visible. `standard_plots/all.py` uses `print("..." % n_procs)` and other pre-PEP-8 stylings. |
| P2 | 1 | Plotting scripts open with the GPL header but no module docstring saying what the script plots or how to invoke it. |
| P3 | 3 | Imports separated stdlib → local with a blank line; ordering within groups is ad hoc but consistent. |
| P4 | 4 | `common.parse_args()` centralises CLI/path parsing across plotting scripts. Env var `SHARK_PLOT_PROCS` for tuning. No hardcoded user-specific paths. |
| P5 | 4 | `common.py` is a real shared module that other plot scripts import for path handling, plot defaults, etc. |

## Tests

| # | Score | Evidence |
|---|-------|----------|
| T1 | 4 | CMake-driven: `make test` / `ctest` from the build dir. Standard for CMake; no exotic incantation. |
| T2 | 5 | Strict `test_<area>.h` convention with cxxtest. Discoverable at a glance: `test_options.h`, `test_components.h`, `test_naming_convention.h`, etc. |
| T3 | 3 | Test method names are clear (`test_parse_options_invalid`). No docstring/comment explaining the contract being asserted or why the input matters. |
| T4 | 4 | CI runs a real integration test against mini-SURFS data with a **fixed seed** (`TEST_FIXED_SEED: 123456`) — bit-identical reproducibility check, same shape as our Phase 0 baseline. |
| T5 | 5 | `.github/workflows/build-and-test.yaml` runs build + tests on push & PR. Matrix over OpenMP on/off. Tests run on every commit. |

## Documentation

| # | Score | Evidence |
|---|-------|----------|
| D1 | 4 | `doc/index.rst` and `doc/configuration/` cover the physics modules — what's implemented, what options exist. Could go deeper but well-structured. |
| D2 | 5 | `doc/configuration/names.rst` + `doc/configuration/sample.rst` are **the** parameter reference. One canonical place lists every option. This is what SAGE26 is missing. |
| D3 | 5 | `doc/hdf5_properties/galaxies.rst` is **auto-generated from the HDF5 files themselves**. Doc cannot drift from code. The pinnacle of "the documentation is the truth". |
| D4 | 3 | `doc/` is mostly user-facing. No clear `docs/developer/` separation. Dev-relevant info appears only in changelog. |
| D5 | 4 | RST (not markdown) used consistently throughout `doc/`. Heading hierarchy and code-block conventions uniform. |

## Commits

| # | Score | Evidence |
|---|-------|----------|
| V1 | 3 | No Conventional Commits prefix. Imperative mood is consistent: "Add missing HDF5 properties' documentation", "Switch to using std::filesystem", "Update boost dependency". Pragmatic, not religious. |
| V2 | 4 | Sampled log shows one logical change per commit. |
| V3 | n/a | Branch-pruning hygiene not directly inspectable from a clone. |
| V4 | 4 | `.gitignore` covers `/build`, `/builds`, IDE configs (`.cproject`, `.project`, `.pydevproject`, `.settings`), `_build`, `*.pyc`. Clean. |

## Patterns SAGE26 should consider adopting

Ordered by leverage (highest impact first):

1. **One canonical parameter reference doc.** D2. SAGE26's parameter docs are scattered between project notes, comments inside `input/*.par`, and code. A single `docs/parameters.md` (or RST equivalent) listing every switch with name, units, default, description, paper reference, valid range -- would be a step change for outside users.
2. **Auto-generated output documentation.** D3. Walk the HDF5 file once, emit a `docs/output_format.md` listing every group + dataset + dtype + shape semantics. The `tests/regression_baseline.py` script already walks every dataset — could be extended in 30 lines to write this doc.
3. **Universal file header.** C1. Cheap to adopt, raises the perceived bar instantly. ICRAR uses a copyright+license block + a `@file` marker; SAGE26 could use a shorter version (author/year + license + one-line file purpose).
4. **Consistent header guards.** C4. Pick `#pragma once` *or* `#ifndef SAGE_<FILENAME>_H_` and apply repo-wide. Mixed styles look sloppy.
5. **CITATION.cff.** R5. Modern academic citation format that GitHub surfaces in the sidebar; one short YAML file, no downside.
6. **CONTRIBUTING.md.** R6. Even one short page ("how to build, how to test, how to submit a PR") is better than nothing.
7. **Python module docstrings.** P2. Every executable plotting script should open with one line saying what it produces. shark doesn't do this either — we can do better than the exemplar here.

## Patterns SAGE26 should NOT bother imitating

- **Sphinx + Read the Docs.** Real cost (maintaining `conf.py`, RST conversions, RTD hosting). For SAGE26's audience, markdown in-repo is enough.
- **CMake.** SAGE26's Makefile works. Switching to CMake is a port, not a hygiene improvement.
- **Wrapping every file in a 17-line license block.** A shorter header (3–5 lines) carries the same signal at less visual cost.

## Patterns where SAGE26 should set its own standard

These rubric lines either scored low across the board or have no obvious answer from shark:

- **In-function comments (C2).** shark is sparse; SAGE26 has more physics complexity per function and probably benefits from a brief "what's modelled, citation" comment at the top of each non-trivial function. Define our own rule.
- **Test "what is validated" comments (T3).** shark relies on test names. SAGE26's tests would benefit from a 1–2 line preamble per test stating the physical invariant being checked.
- **Dev-vs-user doc separation (D4).** Already in flight: `docs/developer/` exists. shark doesn't have this; we're ahead.

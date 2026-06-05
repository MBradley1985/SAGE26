# Contributing to SAGE26

This guide is for contributors to the SAGE26 codebase.

## Getting started

```bash
git clone https://github.com/MBradley1985/SAGE26.git
cd SAGE26
make                # build ./sage and libsage.so
cd tests && make test   # build and run the test suites
```

See the developer style guides under
<https://github.com/MBradley1985/SAGE26/tree/main/docs/developer>
for architecture notes and code conventions.

## Branch model

- `main` — stable, paper-freeze state.
- `dev` — integration branch; all work goes here first.
- Feature branches off `dev`; merge back via PR.

## Style guides

| Language | Guide |
|----------|-------|
| C | [STYLE_C.md](https://github.com/MBradley1985/SAGE26/blob/main/docs/developer/STYLE_C.md) |
| Commits | [STYLE_COMMITS.md](https://github.com/MBradley1985/SAGE26/blob/main/docs/developer/STYLE_COMMITS.md) |
| Docs | [STYLE_DOCS.md](https://github.com/MBradley1985/SAGE26/blob/main/docs/developer/STYLE_DOCS.md) |
| Tests | [STYLE_TESTS.md](https://github.com/MBradley1985/SAGE26/blob/main/docs/developer/STYLE_TESTS.md) |

Key rules for C:
- Every file has a `/* filename.c -- one-line summary. */` header.
- File-private functions are `static`; public functions are declared in the matching `.h`.
- Function docstrings use `/* ... */` block comments immediately before each definition.
- ASCII-only in source files (no Greek letters, superscripts, or em-dashes).

## Tests

Always run tests from inside the `tests/` directory; the root `make tests` target
references a launcher script that does not exist in this tree.

```bash
cd tests && make test               # all suites
cd tests && make test_conservation  # conservation only (fastest)
cd tests && make quick              # single fastest check
bash tests/run_integration_tests.sh # full integration test (slower)
```

New physics changes must keep the regression baseline bit-identical:

```bash
bash tests/regression_baseline.sh
```

## Submitting changes

1. Run `make tests` and verify all suites pass.
2. Run the regression baseline and confirm 5380 datasets are bit-identical.
3. Open a pull request from your branch into `dev`.
4. Tag `@MBradley1985` for review.

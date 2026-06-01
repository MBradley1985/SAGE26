# Test Style Guide

Governs every test file under `tests/`, plus the integration shell scripts (`tests/run_integration_tests.sh`, `tests/regression_baseline.sh`).

SAGE26's tests are already better-documented at the file header level than shark's (T3=3 for shark; SAGE26 file headers like `test_conservation.c` clearly state what's tested). The gaps are: header style is split between Doxygen `/** @file */` and plain `/* HEADER */`; per-test "what is validated" comments are sparse; tolerance constants are sometimes inline literals.

## Out of scope

- Adopting a new test framework. `tests/test_framework.h` macros (`ASSERT_CLOSE`, `ASSERT_IN_RANGE`, etc.) stay.
- Coverage targets. Not defined; not enforced.
- Re-numbering or renaming existing tests.

## Rules

### 1. File header

Tests follow the same plain-comment file-header style as [STYLE_C.md §1](STYLE_C.md), with two extra fields:

```c
/*
 * test_cooling_heating.c — unit tests for gas cooling and heating physics.
 *
 * Validates: metal-dependent cooling rates, cooling and free-fall times,
 * cooling radius, AGN heating suppression, temperature scaling with Vvir,
 * and regime-dependent cooling (hot halo vs CGM).
 *
 * Run: from tests/, `make test_cooling_heating && ./test_cooling_heating`.
 *
 * SAGE26 — released under MIT (see LICENSE).
 */
```

**Reason:** Existing test headers vary between `/** @file ... @brief ... */` and `/* HEADER */`. Doxygen isn't actively wired up to build documentation, so the plain style aligns with the rest of the codebase. The `Validates:` and `Run:` lines tell a reviewer immediately what the test covers and how to invoke it.

**Before (Doxygen style):**
```c
/**
 * @file test_agn_feedback.c
 * @brief Unit tests for AGN feedback physics
 *
 * Tests AGN accretion, heating, and feedback suppression of cooling.
 * ...
 */
```

**After (plain):**
```c
/*
 * test_agn_feedback.c — unit tests for AGN feedback physics.
 *
 * Validates: BH accretion (radio mode, Bondi, cold cloud), Eddington-limit
 * enforcement, AGN heating suppression of cooling, heating radius evolution,
 * mass and metal conservation during accretion.
 *
 * Run: from tests/, `make test_agn_feedback && ./test_agn_feedback`.
 *
 * SAGE26 — released under MIT (see LICENSE).
 */
```

### 2. Test function naming

`test_<thing>_<scenario>`. Existing convention; codifying.

```c
void test_cooling_metal_dependent(void)    /* good */
void test_cooling_hot_regime(void)         /* good */
void test1(void)                            /* bad — what is it testing? */
```

### 3. Per-test preamble

Every test function opens with a preamble that walks the reader through what's being checked and why the expected outcome is physically correct. **Comment generously** — same rationale as [STYLE_C.md §7](STYLE_C.md): a reader should be able to judge whether the test is correct without re-deriving the physics. shark relies on test method names alone (T3=3); we do better.

The preamble names:

1. What's being asserted (the function or behaviour under test).
2. The input scenario being constructed.
3. The expected outcome — the *value* being asserted.
4. Why that value is correct (the physical or numerical reason).
5. If the tolerance is non-trivial, why it's chosen.

For simple invariants this is 2–3 lines. For complex physics tests it may be a paragraph or more — that's fine and encouraged.

**Before:**
```c
void test_cooling_metal_dependent(void) {
    double Lambda_low = cooling_function(1e6, 0.0001);
    double Lambda_high = cooling_function(1e6, 0.02);
    ASSERT_CLOSE(Lambda_high / Lambda_low, 10.0, 0.5);
}
```

**Adequate (short invariant):**
```c
void test_cooling_metal_dependent(void) {
    /* At T = 1e6 K the Sutherland-Dopita cooling curve enhances by roughly
     * 10x going from primordial (Z = 1e-4) to solar (Z = 0.02) metallicity.
     * Tolerance of 0.5 accommodates table interpolation between adjacent
     * metallicity bins. */
    double Lambda_low = cooling_function(1e6, 0.0001);
    double Lambda_high = cooling_function(1e6, 0.02);
    ASSERT_CLOSE(Lambda_high / Lambda_low, 10.0, 0.5);
}
```

**Better (complex physics invariant):**
```c
void test_cgm_precipitation_radius_monotonic_in_metallicity(void) {
    /*
     * Asserts: r_cool (precipitation radius in the CGM regime) increases
     *          monotonically with metallicity at fixed halo mass.
     *
     * Setup: construct a Mvir = 1e12 Msun/h halo with M_CGM at f_b * Mvir,
     *        beta profile (beta = 2/3), and vary Z from 1e-4 to 0.02.
     *
     * Physical reason: r_cool is defined as the radius at which t_cool(r) ==
     *        t_dyn(r). Higher metallicity increases cooling efficiency at
     *        fixed density and temperature (Sutherland & Dopita 1993), so
     *        t_cool decreases at every radius. The radius where t_cool first
     *        equals t_dyn therefore moves outward. See Voit (2015) for the
     *        precipitation framework.
     *
     * Tolerance: none on monotonicity (strict inequality); we only check
     *        the ordering of three sampled metallicities.
     */
    const double Z[3] = {1e-4, 1e-3, 2e-2};
    double r_cool[3];
    for (int i = 0; i < 3; i++) {
        struct GALAXY g = build_test_halo(1e12, Z[i]);
        r_cool[i] = find_precipitation_radius(...);
    }
    ASSERT_TRUE(r_cool[0] < r_cool[1]);
    ASSERT_TRUE(r_cool[1] < r_cool[2]);
}
```

The long preamble is the test's documentation. A reviewer reading the file later (perhaps a referee, perhaps a student) understands why the assertion is the right one without consulting external notes.

### 4. Tolerances

Numerical comparisons use **named** tolerance constants, not magic epsilons. Each tolerance constant has a comment stating its origin: floating-point round-off, expected physical scatter, lookup-table interpolation precision, etc.

**Before:**
```c
ASSERT_CLOSE(result, 7.85, 0.01);
```

**After:**
```c
/* Duffy08 fit coefficients are stated to two decimal places. */
static const double DUFFY08_FIT_TOL = 0.01;
ASSERT_CLOSE(result, 7.85, DUFFY08_FIT_TOL);
```

**Reason:** A reader hitting a failing test needs to know whether the failure is real (drift outside acceptable physical scatter) or numerical (round-off). A named tolerance with a comment answers that.

### 5. Adding a new test

Every new test requires three additions:

1. New `.c` file under `tests/` following the header conventions above.
2. New entry in `tests/Makefile` so `make test_<name>` builds it.
3. New entry in `tests/run_integration_tests.sh` so the integration sweep picks it up.

A test that exists but is wired into neither the Makefile nor the integration script is a test that doesn't run. Don't ship one.

### 6. Regression baseline tests

`tests/regression_baseline.sh` is the ground truth for "does this commit change behaviour" (see [REGRESSION_BASELINE.md](REGRESSION_BASELINE.md)). It is **not** a physics unit test — it is the bit-identical reproducibility check for the whole pipeline. Different role; keep distinct.

Cleanup commits must pass it. Physics-test failures may indicate a real problem and should be investigated.

### 7. Test data

- Small fixture data (a handful of halos, a synthetic cooling table) lives under `tests/test_data/`, committed.
- Large baseline outputs (>10 MB) are captured locally on demand and gitignored. Currently this means `tests/baseline/microuchuu/` per [REGRESSION_BASELINE.md](REGRESSION_BASELINE.md).
- The `tests/baseline/millennium/` manifest is committed (~1.3 MB) because it is the per-conversation regression target.

### 8. Discouraged patterns

- **Tests that pass without exercising the code path.** A test that only asserts on a constant doesn't validate anything.
- **Tests that depend on machine-specific paths** (`/home/karl/...`). Use relative paths from the test executable's cwd, which is `tests/` by convention.
- **Tests that print but never assert.** If the only "verification" is human eyeballs on stdout, it isn't a test — make it an analysis script and move it out of `tests/`.

## Cleanup checklist (Phase 2)

When applying this guide to a test file:

- [ ] File header is plain `/* ... */` style with `Validates:` and `Run:` lines.
- [ ] Test function names follow `test_<thing>_<scenario>`.
- [ ] Each test function opens with 1–3 lines stating what + scenario + why.
- [ ] Tolerances are named constants with a comment for the source.
- [ ] No `assert()` — use the macros in `test_framework.h`.
- [ ] If a new test was added, it appears in both `tests/Makefile` and `tests/run_integration_tests.sh`.
- [ ] `make tests` from the repo root passes (or the specific test target if scoping a single area).
- [ ] Regression baseline (`bash tests/regression_baseline.sh`) passes.

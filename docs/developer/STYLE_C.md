# C Style Guide

Governs every `.c` and `.h` file under `src/` and `tests/`. Derived from rubric scoring of shark ([RUBRIC_SCORES_shark.md](RUBRIC_SCORES_shark.md)) plus SAGE26-specific decisions. Every rule below has a stated reason; if the reason no longer applies, the rule should be revised.

## Out of scope

These are deliberately **not** enforced — flag if you see violations during cleanup but **do not change them**:

- camelCase ↔ snake_case renames. Existing names stay.
- Public API signature changes. Frozen at `baseline/pre-cleanup`.
- Brace style, indentation width, max line length. Whatever the file currently uses is fine.
- Architectural patterns. This is hygiene, not redesign.

If a cleanup batch produces a regression-baseline failure, the change is out of scope — revert and reconsider.

## Rules

### 1. File header

Every `.c` and `.h` file opens with a short comment block:

```c
/*
 * model_cooling_heating.c — gas cooling and AGN heating prescriptions.
 *
 * Implements regime-aware cooling (classical hot-halo vs CGM) per Voit (2015),
 * AGN radio-mode heating, and the persistent HeatingReservoir used in CGM mode.
 *
 * SAGE26 — released under MIT (see LICENSE).
 */
```

**Reason:** Universal file headers are the lowest-cost change that immediately raises "this is a serious project". shark uses a 17-line ICRAR/GPL block; we use a shorter version that says what the file is *for*, not just what it is. The license line is a pointer, not the full text.

**Before (current SAGE26):**
```c
#include <stdio.h>
#include <stdlib.h>
```

**After:**
```c
/*
 * core_build_model.c — top-level merger-tree walk and evolve_galaxies() driver.
 *
 * SAGE26 — released under MIT (see LICENSE).
 */

#include <stdio.h>
#include <stdlib.h>
```

The one-line purpose is the only required field. Multi-line descriptions are optional but encouraged for files implementing non-obvious physics.

### 2. Header guards

Use `#pragma once`. It is already the majority style in `src/*.h` and is supported by every compiler SAGE26 cares about.

**Reason:** `#pragma once` is shorter, can't be copy-pasted into a new file with a stale guard name, and is what most existing headers already use. The one outlier (`src/core_allvars.h` uses `#ifndef ALLVARS_H`) is the inconsistency we're fixing.

**Before:**
```c
#ifndef ALLVARS_H
#define ALLVARS_H
/* ... */
#endif
```

**After:**
```c
#pragma once
/* ... */
```

### 3. extern "C" wrapping in public headers

Public headers (anything other code includes) wrap declarations in:

```c
#ifdef __cplusplus
extern "C" {
#endif

/* declarations */

#ifdef __cplusplus
}
#endif
```

**Reason:** This is already the pattern in `src/core_*.h` and is what lets `libsage.so` be linked from C++ callers (including the SAGE-PSO companion). Codifying so it doesn't get dropped accidentally.

### 4. Include order

Three blocks separated by single blank lines: system, third-party, project. Within each block, ordering is whatever already exists in the file — don't sort speculatively.

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <gsl/gsl_integration.h>

#include "core_allvars.h"
#include "model_cooling_heating.h"
```

**Reason:** Already followed by every .c file; codifying so it stays consistent. The three-block split makes it obvious where a new include should go.

### 5. Unicode in source comments

**Don't use Unicode characters in `.c` / `.h` / `.par` comments.** Use ASCII transliterations. Specific characters to avoid (this is a non-exhaustive list — when in doubt, use ASCII):

- Greek letters (`α β γ π ρ σ Σ Ω`).
- Math symbols (`× ÷ ≤ ≥ ≠ ≈ ∞ ∫ ∂`).
- Superscripts and subscripts (`² ³ ⁻ ₀ ₁`).
- Em-dash and en-dash (`—` and `–`). Use `--` instead. AI-generated comments commonly contain these.
- Curly quotes (`" " ' '`). Use straight `"` and `'`.

**Reason:** IDEs and text editors don't all preserve UTF-8 consistently. We have a documented incident (2026-06-01) where opening `input/millennium.par` in an IDE silently stripped two parameter lines because the comments contained `π Σ ²`. The regression baseline caught it, but the underlying risk is real, and the AI agents we use for cleanup default to em-dashes — the rule needs to be loud enough that both editors and AI tools stay in ASCII land.

**Before:**
```c
// NFW profile: ρ(r) = ρ_s / [(r/r_s)(1 + r/r_s)²]
// M(<R) = 4π ρ_0 r_c³ × [arctan(R/r_c) - ...]
```

**After:**
```c
// NFW profile: rho(r) = rho_s / [(r/r_s) * (1 + r/r_s)^2]
// M(<R) = 4*pi * rho_0 * r_c^3 * [arctan(R/r_c) - ...]
```

**Exception:** Unicode is fine in markdown files under `docs/` where rendering is controlled by a fixed pipeline (GitHub renders UTF-8 consistently).

### 6. Error handling

Use the macros in `macros.h`. Never raw `assert()`.

| Macro | When |
|-------|------|
| `XASSERT(expr, exit_code, fmt, ...)` | Invariants that should be true; abort with location info if not. |
| `XRETURN(expr, return_value, fmt, ...)` | Same as XASSERT but return early instead of aborting. |
| `CHECK_STATUS_AND_RETURN_ON_FAIL(status, ret, fmt, ...)` | Propagate negative status codes from callees. |
| `CHECK_POINTER_AND_RETURN_ON_NULL(ptr, fmt, ...)` | Guard malloc/calloc results. |
| `ABORT(sigterm)` | Fail hard with file/line and an issue-tracker link. |

**Reason:** These macros carry file/line information, are consistent across the codebase, and compile out under `-DNDEBUG`. Raw `assert()` doesn't carry context and behaves differently under NDEBUG. Mixing the two leads to error paths that work in some build configurations and not others.

**Before:**
```c
assert(gal->ColdGas >= 0.0);
if (!ptr) { fprintf(stderr, "alloc failed\n"); exit(1); }
```

**After:**
```c
XASSERT(gal->ColdGas >= 0.0, EXIT_FAILURE, "ColdGas went negative: %g\n", gal->ColdGas);
CHECK_POINTER_AND_RETURN_ON_NULL(ptr, "failed to allocate galaxy buffer");
```

### 7. Comments

SAGE26 is scientific code, and **every non-trivial process in the pipeline deserves a generous walk-through** — not just the physics prescriptions. A reader stepping through the codebase for the first time should be able to learn:

- **The physics**: cooling, star formation, feedback, mergers, instability, reincorporation.
- **The pipeline machinery**: how trees are read, how forests are partitioned across files, how the merger tree is walked recursively in `construct_galaxies`, how galaxies are spawned, joined, and inherited from progenitors, how `evolve_galaxies` orchestrates the per-substep physics loop, how outputs are written.
- **The data lifecycle**: which arrays grow during a forest pass, where buffers are reallocated, what happens to in-memory galaxies between snapshots.

All of it needs comments. The minimalist "comment only the why" style that suits general-purpose software is wrong here — students, collaborators, and reviewers read this code to understand both *what* SAGE26 models and *how* it does it mechanically.

The one thing comments should **not** do is restate the code mechanically:

**Bad — restates the code:**
```c
// Compute the cooling
double cooling = compute_cooling(temp, metallicity, density);

// Loop over galaxies
for (int i = 0; i < ngal; i++) {
    // Get the galaxy
    struct GALAXY *g = &gals[i];
    ...
```

**Good — explains nothing, but at least doesn't lie:**
```c
double cooling = compute_cooling(temp, metallicity, density);
for (int i = 0; i < ngal; i++) {
    struct GALAXY *g = &gals[i];
    ...
```

**Better — walks through the physics:**
```c
/*
 * Cool gas at the local rate. compute_cooling() returns dM_cool/dt in code
 * units; we integrate over the current substep to get the mass cooled, then
 * subtract the AGN heating contribution (if Regime == 0) before depositing
 * the remainder into ColdGas. The factor 0.5 below is the standard semi-
 * implicit trapezoidal correction (Croton+06, eqn 9).
 */
double cooling_rate = compute_cooling(temp, metallicity, density);
double mass_cooled = cooling_rate * dt - 0.5 * heating_rate * dt;
```

#### File header: one paragraph minimum

The file header described in §1 should genuinely orient a reader. For a physics module that means a paragraph (not a single line) describing what's modelled, the major prescriptions implemented, and the references. Use multiple paragraphs if the file mixes several prescriptions.

```c
/*
 * model_cooling_heating.c — gas cooling and AGN heating prescriptions.
 *
 * Implements two cooling regimes selected per-galaxy by Regime:
 *   Regime == 0 (hot halo)   — classical isothermal halo cooling per
 *                              White & Frenk (1991) and Croton et al. (2006),
 *                              with cooling rate computed against the
 *                              Sutherland-Dopita (1993) cooling tables.
 *   Regime == 1 (CGM)        — beta-profile CGM with cooling integrated
 *                              over the density distribution; the
 *                              precipitation criterion follows Voit (2015)
 *                              and Sharma et al. (2012).
 *
 * AGN radio-mode heating is computed in both regimes. In the CGM regime the
 * heating is accumulated into a persistent HeatingReservoir that decays on
 * the halo dynamical time (Sec. 3.2 of the SAGE26 paper) — this prevents
 * "all-or-nothing" cooling shutoffs that single-snapshot heating produces.
 *
 * Code units (10^10 Msun/h, Mpc/h, km/s) are used throughout. Conversions
 * to physical units happen only at the public entry points.
 *
 * SAGE26 — released under MIT (see LICENSE).
 */
```

#### Per-function: lead with a docstring

Non-trivial physics functions open with a substantial comment block:

- **What is modelled**: the physical process and the regime in which it applies.
- **Algorithm**: a short walk-through (numbered steps if multi-step).
- **References**: paper citations for the prescription, including equation numbers when relevant.
- **Inputs and outputs**: what the function takes, what it returns, in what units.
- **Invariants and caveats**: mass conservation, monotonicity, what happens at extreme inputs.

```c
/*
 * cooling_recipe_cgm — cooling rate in the CGM regime (Regime == 1).
 *
 * Physical setup:
 *   The CGM is modelled as a beta profile (beta = 2/3) with the total CGM
 *   mass distributed between the inner core (r < r_c) and the outer halo.
 *   The cooling rate at each radius depends on local density and metallicity;
 *   we integrate the cooling rate over the radial profile to get the total
 *   dM_cool/dt for the halo.
 *
 * Algorithm:
 *   1. Compute the beta-profile normalisation rho_0 from M_CGM and Rvir.
 *   2. Integrate the local cooling rate from r_c outward to r_cool, where
 *      r_cool is the radius at which the cooling time equals the dynamical
 *      time (the precipitation radius per Voit 2015).
 *   3. Subtract heating contributed by the HeatingReservoir, which decays
 *      with timescale tau_dyn between calls.
 *   4. Return the net mass cooled in the current substep.
 *
 * Inputs:
 *   gal     — the central galaxy of the halo. Reads M_CGM, Rvir, metallicity,
 *             HeatingReservoir; writes nothing.
 *   dt      — substep duration in code units (Myr / unit_time).
 *
 * Returns:
 *   Mass cooled into the central galaxy's ColdGas reservoir in code units
 *   (10^10 Msun/h). Guaranteed non-negative; zero if heating dominates.
 *
 * References:
 *   - Voit (2015), ApJL 808, L30 — precipitation criterion (M/M_shock)^4/3.
 *   - Sharma et al. (2012), MNRAS 420, 3174 — beta-profile cooling.
 *   - SAGE26 paper Sec. 3.2 — HeatingReservoir formulation.
 *
 * Invariants:
 *   - Returned mass <= M_CGM (energy and mass conservation).
 *   - Returned mass continuous in M_CGM: no discontinuities at regime boundaries.
 */
double cooling_recipe_cgm(struct GALAXY *gal, double dt)
{
    ...
}
```

This is verbose by software-engineering standards. It is correct by scientific-code standards: a student reading the function later can follow what's happening without re-deriving the model.

#### Per-function: pipeline/I/O functions deserve the same treatment

The same depth applies to **pipeline orchestration and I/O code**, not just physics. A reader who needs to understand how the merger tree is walked, or how a tree file is parsed, or how output files are laid out, should find the answer in the source comments — not by reverse-engineering from a debugger.

**Tree-walking driver (pipeline machinery):**
```c
/*
 * construct_galaxies — recursive tree walk that builds galaxies for one
 * forest, then hands each halo to evolve_galaxies().
 *
 * Walks the merger tree in depth-first order starting from the given halo.
 * The recursion guarantees that by the time a halo is processed, all of its
 * progenitors have already been processed and their galaxies are available
 * to be inherited.
 *
 * Algorithm:
 *   1. If this halo's first progenitor has not yet been processed, recurse
 *      into it. Likewise for every next-progenitor sibling. This ordering
 *      ensures the progenitor galaxy lists are populated before the
 *      descendant tries to inherit from them.
 *   2. Walk down the FOF group: for each subhalo in the same FOF that
 *      hasn't been processed, recurse into it as well.
 *   3. Once all progenitors and FOF members are ready, call
 *      join_galaxies_of_progenitors() to spawn or inherit galaxies for
 *      this halo, then call evolve_galaxies() to run the physics on them
 *      over the snapshot interval.
 *
 * Side effects:
 *   - galaxies_done[halonr] is set when the halo is fully processed; this
 *     is the recursion-termination signal.
 *   - *numgals and *maxgals are mutated as new galaxies are spawned;
 *     the galaxy array is reallocated by join_galaxies_of_progenitors
 *     when *maxgals is exceeded.
 *
 * Inputs / outputs:
 *   See the parameter list. Most arrays are mutated in place.
 *
 * Notes:
 *   The recursion depth is bounded by the merger tree depth (typically <
 *   200 for cosmological trees), well within typical stack limits.
 */
int construct_galaxies(const int halonr, int *numgals, ...)
```

**Tree-file reader (I/O):**
```c
/*
 * setup_forests_io_lht_binary — prepare per-task forest read state for
 * lhalo-binary tree files.
 *
 * Called once per process at startup. Reads the per-file forest counts
 * (forests are independent trees stored sequentially within each tree
 * file), then assigns forests to this MPI task using the
 * ForestDistributionScheme rule from the parameter file.
 *
 * On-disk layout (lhalo-binary format):
 *   File header:    int32  Ntrees
 *                   int32  TotNHalos
 *                   int32  TreeNHalos[Ntrees]
 *   Halo records:   one fixed-size struct halo_data per halo, written
 *                   sequentially in tree-major order (tree 0 first, then
 *                   tree 1, ...). Within a tree, halos are stored in a
 *                   format-defined order so that halo indices into the
 *                   record array form a self-consistent merger tree.
 *
 * The function does not load halo records — it only reads the headers and
 * computes offsets so that later calls to load_forest_lht_binary() can
 * seek directly to a forest's halos.
 *
 * Inputs:
 *   forests_info  — gets filled in with file offsets, halo counts, and
 *                   the forest -> file mapping for this task.
 *   ThisTask, NTasks — MPI rank info used to partition forests across
 *                      tasks per ForestDistributionScheme.
 *
 * Returns:
 *   EXIT_SUCCESS on success, or a negative error code from io/forest_utils
 *   if file headers don't validate.
 */
int setup_forests_io_lht_binary(struct forest_info *forests_info, ...)
```

The same template works for `evolve_galaxies` (the per-substep physics-loop driver), `join_galaxies_of_progenitors` (galaxy inheritance and spawning), `save_gals_hdf5` (output writer), `infall_recipe` (halo baryon growth) — any function that does substantial work, regardless of whether its work is physics or plumbing.

#### Inline comments

Use them generously inside non-trivial functions to label the major steps:

```c
double cooling_recipe_cgm(struct GALAXY *gal, double dt)
{
    /* Step 1: density profile normalisation. */
    const double rho_0 = beta_rho_0(gal->CGMgas, gal->Rvir, gal->r_c, BETA_DEFAULT);

    /* Step 2: find the precipitation radius. r_cool is where t_cool == t_dyn;
     * gas inside r_cool can cool within a dynamical time. */
    const double r_cool = find_precipitation_radius(rho_0, gal->metallicity, gal->Rvir);

    /* Step 3: integrate the local cooling rate from the inner core out to r_cool. */
    const double gross_cooling = integrate_cooling(rho_0, gal->metallicity, gal->r_c, r_cool, dt);

    /* Step 4: subtract heating drawn from the persistent reservoir.
     * The reservoir decays exponentially on tau_dyn between calls. */
    const double net_cooling = apply_heating_reservoir(gal, gross_cooling, dt);

    return net_cooling > 0.0 ? net_cooling : 0.0;
}
```

The numbered "Step N" labels mirror the algorithm in the docstring — easy cross-reference when the reader is following along.

#### Hard rules

A few things comments should never do:

- **Restate the code mechanically.** `// increment i` next to `i++` is noise.
- **Live in commented-out blocks.** Delete dead code; git remembers.
- **Track changes inline.** `// 2026-05-30 added by Karl, fixed bug in cooling` belongs in the commit message, not the source. Inline change-logs rot.
- **Reference current task or PR.** `// for issue #42` rots after the issue is closed. The commit message owns task references.

#### Calibrating depth

Use this rough scale to decide how much to write. The function category matters more than line count:

| Function category | Comment depth |
|---|---|
| Trivial getter/setter, format helper, obvious utility | None or one line. |
| Single-step physics with a clear name | A paragraph docstring; inline comments rarely. |
| Multi-step physics, regime branching, paper citations | Full docstring (what / algorithm / refs / inputs / invariants) plus inline step labels. |
| Numerical method (root-finder, integrator, ODE solver) | Docstring + comment every non-obvious step + cite the algorithm. |
| Tree-walking / recursion driver (`construct_galaxies`) | Full docstring covering the walk order, why that order is correct, what side effects occur, and what invariants hold at each recursion entry. |
| Per-step pipeline orchestration (`evolve_galaxies`, `join_galaxies_of_progenitors`) | Full docstring listing the steps in order with their role (infall → cooling → SF → feedback → ...), plus inline labels separating each physics module call. |
| Tree-file reader / output writer | Full docstring describing the on-disk format (with a byte-layout sketch if non-trivial), what the function loads vs computes, and any assumptions about file ordering. |
| Memory and lifecycle code (`core_mymalloc`, galaxy-array reallocation, buffer flush) | Full docstring covering when this is called, what state is mutated, and what the caller must know to use it safely. |

When in doubt, **write the comment**. The cost of an over-commented file is small; the cost of an under-commented function — whether physics, I/O, or pipeline orchestration — is a future student lost for a day reverse-engineering what the code is doing.

### 8. Magic numbers

Physical constants, parameter defaults, and any number whose meaning isn't obvious from context belong in `macros.h` or as a file-scope `static const` with a comment citing the source.

**Inline literals are fine** for genuinely mathematical constants (the `2` in `r*r`, the `0.5` in a midpoint, the `4.0/3.0` in a sphere volume).

**Before:**
```c
return 7.85 * pow(Mvir_Msun / 2.0e12, -0.081) * pow(1.0 + z, -0.71);
```

**After:**
```c
/* Duffy et al. (2008) M-c relation, table 1 row "Full sample, NFW, M200". */
static const double DUFFY08_A = 7.85;
static const double DUFFY08_M_PIVOT = 2.0e12;
static const double DUFFY08_B = -0.081;
static const double DUFFY08_C = -0.71;
return DUFFY08_A * pow(Mvir_Msun / DUFFY08_M_PIVOT, DUFFY08_B) * pow(1.0 + z, DUFFY08_C);
```

The "after" form makes the source citable in a comment and makes the constants searchable when a paper revision arrives.

### 9. Static vs public

Every function not declared in a header must be `static`. No exceptions.

**Reason:** Non-static functions add to the linker's global symbol table and become part of the de facto API. SAGE26's API is what's in the headers — anything else is an implementation detail.

### 10. Discouraged patterns

- **Silent failure paths.** A function that returns success when a precondition is violated will eventually bite someone. Use `XRETURN` with a message.
- **Mixed unit conversions inside a single function.** Convert at boundaries, keep internals in code units (`10^10 M_sun/h`, `Mpc/h`, `km/s`).
- **`printf` for error output.** Use `fprintf(stderr, ...)` or the error macros, which route to stderr.

## Cleanup checklist (Phase 2)

When applying this guide to a file:

- [ ] File header present, one-line purpose accurate.
- [ ] Header guard is `#pragma once` (if a header).
- [ ] `extern "C"` wrap is in place (if a public header).
- [ ] Includes ordered: system, third-party, project, with blank-line separators.
- [ ] No Unicode in comments — transliterate any Greek letters, sub/superscripts, math symbols.
- [ ] No raw `assert()` — replace with `XASSERT` / `XRETURN`.
- [ ] No commented-out code blocks or change-log comments.
- [ ] File header paragraph genuinely orients a reader to what the file models (not just one line if the file is physics-heavy).
- [ ] Non-trivial physics functions have a substantial docstring covering what / algorithm / refs / inputs / invariants.
- [ ] Multi-step physics functions have inline "Step N" labels that mirror the docstring algorithm.
- [ ] Comments explain physics, not the mechanics of the code — but they are present and generous, not minimal.
- [ ] Magic numbers lifted to named constants where source is non-obvious.
- [ ] Non-header-declared functions are `static`.
- [ ] Regression baseline (`bash tests/regression_baseline.sh`) passes after the change.

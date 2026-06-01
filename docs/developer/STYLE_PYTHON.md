# Python Style Guide

Governs every `.py` file under `plotting/` and any auxiliary Python (cffi glue, helper scripts). Tests live under `tests/` and may have additional conventions in [STYLE_TESTS.md](STYLE_TESTS.md).

Derived from rubric scoring of shark ([RUBRIC_SCORES_shark.md](RUBRIC_SCORES_shark.md)) plus SAGE26-specific decisions. shark's plotting code (`standard_plots/`) scored low on docstrings (P2=1) — we can do better than the exemplar here.

## Out of scope

- Type annotations across the board. Welcome but not required.
- Adopting `black` or `ruff`. `flake8` stays the source of truth.
- Rewriting cffi-generated bindings.
- Mass renames or signature changes.

## Rules

### 1. flake8 baseline

`.flake8` is already configured at the repo root and runs as a pre-commit hook. Don't change the ignore set during cleanup unless you have a specific reason and a replacement plan. Current ignores: `E203, E266, E501, W503, F403, F401`. Current `max-line-length = 79`, `max-complexity = 18`.

**Reason:** Mechanical formatting rules are not worth fighting over. flake8 is already wired up — codify what it accepts and move on.

### 2. Module-level docstring

Every executable script and every importable module opens with a docstring:

```python
"""
<one-line summary, ends with period>

<optional: longer description>

<optional: Usage block for executable scripts>
"""
```

**Reason:** shark scored 1/5 on this — their plotting scripts have no docstrings, so a reader has to dig through `import` blocks and `main()` to learn what the file does. `plotting/paper_plots.py` already follows this pattern; codifying so it spreads.

**Before (current `plotting/allresults-local.py`):**
```python
#!/usr/bin/env python

import h5py as h5 # type: ignore
import numpy as np
```

**After:**
```python
#!/usr/bin/env python
"""
allresults-local.py — z=0 diagnostic plots for a local SAGE26 output.

Reads model_*.hdf5 from the given output directory and writes one figure per
panel (SMF, BHMF, sSFR, cold-gas fraction, ...) into the same directory.

Usage:
    python plotting/allresults-local.py                       # uses default output dir
    python plotting/allresults-local.py path/to/output/       # specify dir
"""

import h5py as h5
import numpy as np
```

The `# type: ignore` for h5py also goes away — flake8 doesn't require it.

### 3. Import ordering

Three blocks separated by blank lines:

1. Standard library
2. Third-party
3. Local / project

Within each block, ordering is whatever already exists — don't sort speculatively.

**Before:**
```python
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
from collections import defaultdict
from scipy.stats import gaussian_kde, stats
```

**After:**
```python
import argparse
import os
import sys
from collections import defaultdict

import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, stats
```

**Reason:** Stdlib mixed in with third-party hides what the script's actual external dependencies are.

### 4. Path handling

No hardcoded absolute paths inside scripts. Output and input directories come from CLI arguments, environment variables, or the parameter file being analysed.

**Reason:** shark's `common.parse_args()` centralises this (P4=4). SAGE26 plotting scripts already mostly do the right thing; codifying so it doesn't slip.

**Forbidden:**
```python
SAGE_DIR = '/Users/karl/Documents/SAGE26/output/microuchuu_Karl/'
```

**Acceptable:**
```python
parser.add_argument('output_dir', nargs='?', default='./output/millennium/')
```

### 5. Plotting style centralisation

Figure size, font, DPI, and colour palette settings live in **one** module, not duplicated across every script.

If a script needs to override a value, do so explicitly after importing the shared defaults.

**Reason:** Paper figures must look consistent. shark uses a shared `common.py` for this (P5=4). SAGE26 currently sets `plt.rcParams[...]` at the top of multiple scripts — converging these is a Phase 2 task.

**Pattern:**
```python
# plotting/style.py (new, single source of truth)
import matplotlib.pyplot as plt

def apply_paper_style():
    plt.rcParams["figure.figsize"] = (8.34, 6.25)
    plt.rcParams["figure.dpi"] = 96
    plt.rcParams["font.size"] = 14

# in every plotting script
from style import apply_paper_style
apply_paper_style()
```

Don't create this module pre-emptively — wait until at least two plotting scripts are being cleaned up in the same pass, then extract.

### 6. Unit handling

Match the C side: code-unit quantities stay in code units (10^10 M_sun/h, Mpc/h, km/s) unless converting for display. Any conversion is commented:

```python
stellar_mass_msun = stellar_mass_code_units * 1e10 / hubble_h   # 10^10 Msun/h -> Msun
```

**Reason:** SAGE26 has internal units and observational units. Mixing them silently inside a function is a known source of bugs.

### 7. CLI conventions

Every executable script accepts `--help` and documents its positional and optional arguments via `argparse`. No silent reliance on `cwd`.

**Reason:** A script that breaks when run from a different directory wastes everyone's time. `argparse --help` is the standard self-documentation.

### 8. Comments and docstrings (process and physics walk-throughs)

SAGE26's plotting scripts are not throwaway. They produce the figures that go into papers, and they manipulate physical quantities (binning by halo mass, converting units, computing densities). **Comment generously.** A reader should be able to walk through a plotting routine and understand both the calculation and the physical meaning of each step.

The "comment the why" minimalism that suits general-purpose software is the wrong default here. See [STYLE_C.md §7](STYLE_C.md) for the longer rationale — the same logic applies.

#### Function and method docstrings

Non-trivial analysis functions get a docstring covering: what's computed, the algorithm, inputs (with units), outputs (with units), references where applicable.

**Bad — no docstring, no orientation:**
```python
def compute_smf(masses, weights, bins):
    counts, edges = np.histogram(masses, bins=bins, weights=weights)
    dlog_m = np.diff(edges)
    return edges[:-1], counts / dlog_m / volume_mpc3
```

**Good — walks through what's happening:**
```python
def compute_smf(masses, weights, bins):
    """
    Compute the stellar mass function (SMF) from a sample of galaxies.

    Bins galaxies by log10(stellar mass), weights each by `weights` (use 1.0
    for an unweighted count; use 1/V_max for a volume-corrected estimator),
    and divides by bin width and survey volume to return dn/dlogM in
    Mpc^-3 dex^-1.

    Inputs:
        masses  — log10(M*/Msun) per galaxy, shape (N,).
        weights — per-galaxy weight, shape (N,). Pass np.ones(N) for raw counts.
        bins    — bin edges in log10(M*/Msun), shape (K+1,).

    Returns:
        bin_centres, phi  — bin lower edges and dn/dlogM in Mpc^-3 dex^-1.

    Reference:
        Standard SMF construction; see e.g. Baldry et al. (2012) for the
        Vmax weighting convention.
    """
    # Histogram in log-mass bins with the per-galaxy weight applied.
    counts, edges = np.histogram(masses, bins=bins, weights=weights)

    # Convert to number density per dex: divide by bin width (already in dex)
    # and by the simulation volume so the result is comparable to observations.
    dlog_m = np.diff(edges)
    return edges[:-1], counts / dlog_m / volume_mpc3
```

#### Inline comments inside calculations

Walk the reader through major steps, especially where units are converted, where a model assumption is baked in, or where the calculation departs from a textbook reference.

```python
# Convert SAGE code units (10^10 Msun/h) to log10(Msun) for plotting.
log_mstar = np.log10(stellar_mass_code_units * 1e10 / hubble_h)

# Mask out galaxies below the mass-completeness cut for this survey volume.
# Below this threshold the SMF normalisation is dominated by incompleteness.
mask = log_mstar > MASS_COMPLETENESS_LIMIT
log_mstar = log_mstar[mask]
```

#### Module docstring

Already covered in §2 — but for plotting modules, the docstring should also say what figures the script produces and where they go.

#### Hard rules (same as C)

- **Don't restate the code.** `# increment i` next to `i += 1` is noise.
- **No commented-out code.** Delete it; git remembers.
- **No inline change-logs.** `# 2026-05-30 added by Karl` rots; use the commit message.

#### Calibrating depth

Same scale as [STYLE_C.md §7](STYLE_C.md):

| Function | Comment depth |
|---|---|
| Trivial wrapper, obvious data loader | One-line docstring. |
| Calculation with clear name, single step | Paragraph docstring. |
| Multi-step physics conversion or estimator | Full docstring (what / algorithm / refs / units) + inline step labels. |
| Anything touching unit conversion | Comment the conversion explicitly with source and destination units. |

When in doubt, write the comment.

### 9. Discouraged patterns

- **Mutable default arguments** (`def f(x=[]):`). Use `None` and create inside.
- **Bare `except:`**. Catch the specific exception class. If you really mean "everything", `except Exception:` at minimum.
- **`warnings.filterwarnings("ignore")` at module scope** without a reason in a comment. Hides genuine issues. If a specific warning needs suppressing, scope it tightly and say why.

## Cleanup checklist (Phase 2)

When applying this guide to a file:

- [ ] Module-level docstring present, accurate, includes Usage for executable scripts.
- [ ] Imports in three blocks: stdlib, third-party, local — blank-line separated.
- [ ] No hardcoded user-specific paths.
- [ ] `plt.rcParams` setup not duplicated across scripts (extract if a `style.py` helper is in play).
- [ ] Unit conversions commented at the boundary.
- [ ] `flake8 plotting/` passes (already required by pre-commit; verify explicitly after edits).
- [ ] Non-trivial analysis functions have a docstring covering inputs/outputs/units/algorithm.
- [ ] Unit conversions are commented at the point of conversion.
- [ ] No mutable default args, no bare `except`, no module-scope `warnings.filterwarnings` without a reason.
- [ ] Regression baseline (`bash tests/regression_baseline.sh`) passes — most plotting changes are no-ops for the baseline, but run anyway in case a script is imported by something else.

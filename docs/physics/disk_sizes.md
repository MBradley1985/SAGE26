# Disk sizes

The disk scale radius is one line of arithmetic in `get_disk_radius()`
(`src/model_misc.c`), and it sets the surface density that every H2, star
formation, disk instability and ram-pressure prescription reads. This note
records what it computes, how hard the rest of the model leans on it, and what
the `DiskRadiusOn` levels change.

## What the published expression is

Mo, Mao & White (1998) eq. 12 with a Bullock-style spin parameter:

```
lambda = |j| / (sqrt(2) * Vvir * Rvir)
r_d    = f_j * (lambda / sqrt(2)) * Rvir
```

`Rvir` cancels, so the disk radius is really

```
r_d = f_j * |j| / (2 * Vvir)
```

Only the halo specific angular momentum and the circular velocity enter. `Vvir`
is the peak-retained value (`core_build_model.c` only updates it when `Mvir`
grows), so a temporary dip in tree mass lowers `|j|` without lowering `Vvir` and
shrinks the disk spuriously.

Two naming traps: `r_d` is the exponential **scale length**, not the half-mass
radius (callers wanting `r_half` multiply by `DISK_HALF_MASS_FRAC = 1.68`), and
the `Rvir` column in the HDF5 output is the *instantaneous* virial radius
(`save_gals_hdf5.c`), not the peak-retained one the physics uses.

## How sensitive the model is

Tracking Millennium centrals across one snapshot and selecting the 18 800 with
`|dlog ColdGas| < 0.02`, so that geometry is the only thing changing:

| response | d/dlog r_d | correlation |
|---|---|---|
| `dlog H2` | -2.59 | -0.99 |
| `dlog SFR` | -3.66 | -0.98 |

`SFR ~ r_d^-3.7`. Everything below is amplified by that exponent. Consistently,
`f_j = 0.55` versus `1.0` on Millennium changes the total HI mass by 2.9x, the
total stellar mass by 0.14 dex, and the fraction of galaxies above 1e10 Msun
from 0.069 to 0.094. `f_j` is degenerate with `SfrEfficiency`,
`H2DiskAreaOption` and `SF_DISK_RADIUS_FRAC`.

## Where it disagrees with the size data

Disk-dominated z=0 centrals (`B/T < 0.5`, `M* > 1e8 Msun`):

| quantity | SAGE26 (`DiskRadiusOn=0`) | observed |
|---|---|---|
| `r_d / Rvir` | 0.028-0.031, mildly declining | 0.016-0.018, flat (Somerville et al. 2018, GAMA) |
| size-mass slope | 0.13 | 0.2-0.3 (late types) |
| scatter at fixed `M*` | 0.27 dex | 0.16-0.20 dex |

The slope and the scatter are structural: MMW98 has no baryonic term, so the
model cannot produce a size-mass relation with a mass-dependent slope, and its
scatter is halo spin scatter by construction. Fixing those needs a
baryon-responsive radius (tracked disk `j`, or the MMW98 `f_R(lambda, c, m_d,
j_d)` correction), which no `DiskRadiusOn` level currently implements.

## Halo spin is resolution-biased

Median Bullock `lambda` for z=0 centrals by particle count:

| `Len` | 20-50 | 50-100 | 100-300 | 300-1000 | 1000-3000 | >3000 |
|---|---|---|---|---|---|---|
| median `lambda` | 0.079 | 0.059 | 0.052 | 0.047 | 0.046 | 0.042 |
| `sigma(log lambda)` | 0.301 | 0.298 | 0.285 | 0.261 | 0.243 | 0.263 |

Well-resolved halos converge on the canonical distribution, so the units are
right. But 55% of z=0 Millennium centrals have `Len < 50`, where discreteness
noise inflates `lambda` by ~1.9x. The noise adds in quadrature to a
positive-definite `|j|`, so it is a *bias*, not just scatter, and it makes disks
in poorly-resolved halos too large, their gas too diffuse, and their HI
over-retained.

This is why `DiskRadiusOn = 2` averages the spin **vector** rather than `|j|` or
`r_d`: the noise components average toward zero while the physical spin
persists. Averaging magnitudes would suppress the jitter and keep the bias.

## What each level does

| `DiskRadiusOn` | behaviour |
|---|---|
| 0 | Published. Reproduced bit-for-bit; this is the default. |
| 1 | Adds a working virial fallback and bounds `r_d/Rvir` to `[0.002, DiskRadiusMaxFrac]`. The published else-branch is only reachable when `Rvir == 0` (`get_virial_velocity` returns 0 exactly then), so it returns `r_d = 0` and the galaxy is permanently inert -- every downstream path guards on `DiskScaleRadius > 0`. Level 1 rebuilds the virial scale from `Len * PartMass`, as `get_virial_mass` already does for subhalos. |
| 2 | As 1, plus the spin-vector running mean, weighted `w = dt / (dt + Rvir/Vvir)`. No free parameter: the disk cannot restructure faster than a halo dynamical time, and the spin measurement decorrelates on the snapshot spacing. On Millennium `w ~ 0.2` at z=0, rising to ~0.5 by z=3. |

Measured on Millennium file 0 (4196 z=0 galaxies):

| | `DiskRadiusOn=0` | 1 | 2 |
|---|---|---|---|
| median `r_d` [kpc] | 3.49 | 3.49 | 3.18 |
| median `r_d/Rvir` | 0.0466 | 0.0466 | 0.0420 |
| fraction with `r_d = 0` | 0.33% | 0 | 0 |
| median `\|dlog r_d\|` per snapshot | 0.047 | 0.047 | 0.016 |
| 84th percentile of that | 0.152 | 0.151 | 0.043 |
| fraction jumping >0.3 dex | 4.7% | 4.6% | 0.2% |
| `sum M_HI / sum M*` | 0.576 | 0.575 | 0.463 |
| size-mass scatter [dex] | 0.268 | 0.266 | 0.266 |

Level 2 cuts the per-snapshot jitter by 3x and near-eliminates the >0.3 dex
swings (which, at `r_d^-3.7`, were >1 dex swings in SFR). It also pulls the
median `r_d/Rvir` from 0.047 onto 0.042, the value well-resolved halos give --
the de-biasing working as intended. It does **not** reduce the size-mass
scatter, because that is dominated by genuine halo-to-halo `lambda` scatter
rather than measurement noise.

## What is deliberately unchanged

- **Satellites.** `get_disk_radius()` is only called for FOF centrals, so a
  Type 1 disk size stays frozen at infall under every level (73% of galaxies
  that stay Type 1 from snapshot 50 to 63 have a bit-identical `r_d`). Updating
  them would mean trusting subhalo spin measurements, which are worse than
  central ones.
- **`Mvir = 0` FOF centrals.** Level 1 stops these from having `r_d = 0`, but
  they stay empty: `Vvir = 0` also disables infall and cooling. Reviving them
  properly means falling back to `Len * PartMass` in `get_virial_mass()` for FOF
  centrals, which is a separate change with a much wider blast radius.
- **The literal `1.414`.** Both occurrences together give `1/1.414^2` instead of
  `1/2`, a 0.06% shift. Frozen for reproducibility, not because it matters.

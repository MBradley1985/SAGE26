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

`DiskRadiusOn = 2` averages the spin **vector** rather than `|j|` or `r_d`, on the
expectation that the noise components would average toward zero while the physical
spin persists. **Measured, that does not happen.** The bias ratio
`lambda(Len 20-50) / lambda(Len > 1000)` is 1.569 at level 0 and 1.583 at level 2 --
unchanged.

The reason is that the discreteness error comes from finite particle sampling, and
halo membership turns over slowly: over one dynamical time most of the same particles
are still in the halo, so the sampling error is strongly correlated between adjacent
snapshots and does not average down. Time-averaging only suppresses the genuinely
fast-fluctuating component. What level 2 actually delivers is the 3x jitter reduction
below, plus a near-uniform ~9% shrink in `r_d` from averaging over spin-axis tumbling.

Correcting the resolution bias needs an explicit N-dependent de-biasing (subtracting
the expected noise contribution to `|j|^2` in quadrature, calibrated against
well-resolved haloes). No level here implements that.

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
| `sum M_HI` | 554 | 553 | 620 |
| `sum M*` | 962 | 962 | 1340 |
| size-mass scatter [dex] | 0.268 | 0.266 | 0.266 |

Level 2 cuts the per-snapshot jitter by 3x and near-eliminates the >0.3 dex
swings (which, at `r_d^-3.7`, were >1 dex swings in SFR). The shift in median
`r_d/Rvir` from 0.047 to 0.042 is a near-uniform ~9% shrink at every particle
count, not a de-biasing -- it lands near the well-resolved value by coincidence.
It does **not** reduce the size-mass scatter, because that is dominated by
genuine halo-to-halo `lambda` scatter rather than measurement noise, and it does
**not** reduce the HI mass function excess (see below).

## What level 2 does not fix: the HI mass function

The high-mass end of the HIMF is overproduced, and no disk-radius setting repairs it.
Residuals against Jones et al. (2018), in dex, mini-Millennium **file 0 only**
(1/8 of the box -- see the caveat below):

| `log M_HI` | N (default) | level 0 | level 2 | `f_j = 0.55` | `chi = 1.7` | `MaxFrac = 0.05` | `Radio = 0.5` | `f_j 0.55 + Radio 0.5` |
|---|---|---|---|---|---|---|---|---|
| 10.1 | 93 | +0.35 | +0.39 | +0.03 | +0.38 | +0.40 | +0.22 | **-0.34** |
| 10.4 | 55 | +0.79 | +0.85 | +0.05 | +0.84 | +0.85 | +0.48 | **-0.25** |
| 10.6 | 16 | +1.22 | +1.36 | +0.50 | +1.29 | +1.45 | +0.72 | **+0.02** |

**Caveat: these bins are Poisson-limited.** In one tree file the 10^10.6 bin holds 16
galaxies at default and 1 in the best configuration, and 10^10.9 holds 3 and 0. Anything
above `log M_HI = 10.5` needs the full 8-file run before it is quotable. The residual is
solid at 10.1-10.4 (93 and 55 galaxies) and the HIMF is already within 0.2 dex below 10^10.

Level 2 makes it marginally worse. Clamping the large-disc tail (`DiskRadiusMaxFrac = 0.05`)
and raising `chi` are both null, and so is lowering `MShockMsun` to 2.5e11 or 1e11.

## The excess is a bimodal tail, not a mass window

The galaxies with `M_HI > 10^10.3` in the calibrated model (N = 61) are ordinary,
well-resolved, disc-dominated centrals: median `Len = 566`, `B/T = 0.07`,
`log M* = 10.21`, `log Mvir = 11.83`, `r_d/Rvir = 0.047` -- the population median, not
the high-spin tail. That is why clamping the large-disc tail does nothing. They hold
**45% of all their baryons as cold gas**, with a `ColdGas/SFR` depletion time of **44 Gyr**.

They are not typical for their halo mass: the *median* central at `log Mvir = 11.75` has
a cold fraction of only 0.015. The excess is a minority tail at fixed halo mass. Splitting
centrals at `log Mvir = 11.55-12.05` (N = 184) by cold-gas fraction shows what separates
them, at essentially identical stellar mass:

| | gas-rich (top 20%) | gas-poor (bottom 20%) |
|---|---|---|
| `ColdGas / (f_b Mvir)` | 0.427 | 0.001 |
| `r_heat / Rvir` | 0.175 | 1.000 |
| fraction with `r_heat / Rvir > 0.9` | 0.081 | **1.000** |
| fraction with `r_heat == 0` | 0.189 | 0.000 |
| `log M_BH` | 6.07 | 7.19 |
| `B/T` | 0.043 | 0.395 |
| `log M*` | 10.26 | 10.17 |

The sink is AGN radio-mode heating, and it is **all-or-nothing**. Every gas-poor galaxy
has `r_heat` saturated at `Rvir`; only 8% of the gas-rich ones do, and 19% have `r_heat`
exactly zero, meaning the AGN never fired at all. The causal chain is
bulge growth -> black hole growth -> radio heating -> the `r_heat` ratchet reaching `Rvir`
-> cooling shut off. A galaxy with a quiet merger history stays disc-dominated, never grows
a black hole (1.1 dex lighter at the same halo mass), never trips the ratchet, and keeps
accreting indefinitely. Because `r_heat` only ratchets up and never decays, there is no
intermediate state.

Three things follow, none of which is a disc-size change:

1. **Preventive heating should not be gated entirely on the black hole.** Above
   `Mvir ~ 10^11.5` a hot corona forms regardless of AGN activity; SAGE26 ties all cooling
   suppression to the AGN-driven `r_heat`.
2. **Disc-dominated galaxies have no black-hole growth channel.** Growth is merger- and
   instability-driven, so a quiet disc at `log Mvir = 11.8` sits at `M_BH = 10^6.1`.
   Secular fuelling would close the 1.1 dex gap.
3. **The ratchet is binary.** A continuously varying suppression would remove the
   bimodality rather than sorting galaxies into fully quenched and fully unquenched.

## The disc radius *is* the root cause -- via instability, not surface density

`check_disk_instability()` uses `Mcrit = Vmax^2 * (3 r_d) / G`, so **`Mcrit` scales with
the disc radius**, and `grow_black_hole()` is called *only* when the disc is unstable.
For a galaxy with a quiet merger history that is the sole black-hole growth channel. So
an over-large disc is Toomre-stable, never builds a bulge, never grows a black hole,
never trips the `r_heat` ratchet, and never acquires a sink.

The historical record confirms it. At `log Mvir = 11.55-12.05`, gas-rich versus gas-poor:

| | gas-rich | gas-poor |
|---|---|---|
| `InstabilityBulge / M*` | 0.013 | **0.218** |
| `MergerBulge / M*` | 0.012 | 0.021 |
| `r_d` [kpc] | 7.00 | 3.43 |

The gas-poor galaxies built their bulges by *instability*, not merging, and they have half
the disc radius. And the disc radius is independently over-constrained: SAGE26 gives
`r_d/Rvir = 0.034` against the Somerville et al. (2018) GAMA value of 0.016-0.018.

Setting `f_j = 0.55` -- the value that size ratio independently requires, and the expected
`j_disk/j_halo` -- restores the chain from the inside, with **no AGN parameter touched**:

| at `log Mvir = 11.55-12.05` | default | `f_j = 0.55` | `Radio = 0.5` |
|---|---|---|---|
| `log M_BH` | 6.90 | **7.10** | 6.48 |
| `InstabilityBulge / M*` | 0.102 | **0.303** | 0.017 |
| fraction gas-rich | 0.348 | **0.250** | 0.261 |
| median `r_d/Rvir` | 0.0337 | **0.0204** | 0.0297 |

and the HIMF residual (vs Jones+18, dex):

| `log M_HI` | 9.6 | 9.9 | 10.1 | 10.4 |
|---|---|---|---|---|
| default | +0.17 | +0.13 | +0.33 | +0.81 |
| `f_j = 0.55` alone | **-0.09** | **-0.13** | **-0.12** | **-0.09** |
| `Radio = 0.5` alone | +0.14 | +0.07 | +0.21 | +0.50 |

Raising `RadioModeEfficiency` is a band-aid that works by starving the disc, which
*suppresses* the very channel that should be doing the work: the instability bulge
collapses to 0.017 and black holes end up 0.4 dex lighter than the default. It treats the
symptom by breaking the mechanism further.

**Costs of `f_j = 0.55`, honestly.** The SMF massive end rises 0.10-0.22 dex, and `B/T` at
`M* ~ 10^10.3` goes from 0.257 to 0.395 against an observed 0.2-0.3 -- the instability
channel now over-fires. That points at the next targets: `TOOMRE_DISK_FACTOR = 3.0`, the
use of `Vmax` rather than a disc circular velocity in `Mcrit`, and the fact that the
size-mass *slope* is still 0.13 against an observed 0.2-0.3, so a single global `f_j`
fixes the normalisation but not the shape.

The structural criticisms above still stand -- `r_heat` is binary and never decays,
preventive heating is entirely black-hole-gated, and there is no secular fuelling channel
-- but they are downstream. The disc size is the upstream error. All of this is one tree
file; confirm on the full volume before quoting.

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

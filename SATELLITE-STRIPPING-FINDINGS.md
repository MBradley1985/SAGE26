# Satellite Stripping in SAGE26: Scheme, Cadence-Dependence, and Satellite HI

**Status:** Working notes / summary of findings. **Date:** 2026-07-02.
**Context:** Follow-up to `DYNAMIC-TIMESTEP-CONVERGENCE-NOTES.md`, which flagged that
satellite stripping in stock SAGE is tied to the substep *count* rather than to
elapsed time — a latent issue that becomes visible now that SAGE26 uses
dynamical-time (adaptive) substepping.

---

## TL;DR

1. **Two code changes were made and are good additions:**
   - **CGM-stripping bug fix** (a genuine correctness fix): CGM-regime satellites
     were never being stripped at all.
   - **Dynamical-time stripping** (`PhysicalStrippingOn = 2`, now the default): a
     physically-motivated, cadence-invariant, substep-independent stripping rule.
2. **Neither change materially affects the observable galaxy population** — they
   act on the (largely unobservable) hot halo. So they are "free" improvements:
   more defensible physics and cleaner cross-simulation comparison at no
   observational cost.
3. **The physical scheme removes a cadence artifact** worth ~27% of the
   Millennium↔microUchuu discrepancy in satellite hot-halo content — a spurious
   difference that came purely from the two runs being saved at different
   snapshot spacings.
4. **Separately, SAGE26 over-produces satellite HI** by ~0.4–0.6 dex vs xGASS.
   This is *not* a stripping-timescale problem; it is (a) a ~0.2 dex global
   cold-gas/HI excess plus (b) a missing *cold-gas* (ram-pressure) stripping term.

---

## 1. The problem with legacy stripping

`strip_from_satellite` removed a satellite's baryon excess (baryons above
`BaryonFrac·Mvir`) as `excess / effective_steps` once per substep, with the
excess recomputed each substep. Over one snapshot the stripped fraction is

```
1 − (1 − 1/N)^N        (N = substep count)
```

which depends only on `N` and has **no connection to elapsed time `dT` or any
physical timescale**. It converges to a discretisation artefact (`1 − 1/e ≈ 63%`).
Under fixed substepping this was a constant bias absorbed into calibration; under
adaptive/dynamical substepping `N` varies, so it becomes a live artefact.

## 2. The stripping schemes (parameter `PhysicalStrippingOn`)

| Mode | Where it runs | Fraction stripped / snapshot | Substep-count dependence |
|---|---|---|---|
| 0 — legacy | in the substep loop | `1−(1−1/N)^N` | strong (unphysical) |
| 1 — physical, per-substep | in the substep loop | `1−exp(−dT/t_strip)` | ~1% residual (`O(1/N)`) |
| **2 — physical, analytic (DEFAULT)** | **once, outside the loop** | `1−exp(−dT/t_strip)` | **none** |

`t_strip = StrippingTimescaleFactor · t_dyn(host)`, with `t_dyn = Rvir/Vvir` of
the central. Mode 2 is the exact `N→∞` limit of mode 1 — it removes the substep
dependence entirely *and* is invariant to snapshot cadence (because
`exp(−dT₁/t)·exp(−dT₂/t) = exp(−(dT₁+dT₂)/t)`).

**New runtime parameters (all default to no-change / physical values):**
- `PhysicalStrippingOn` — `0`/`1`/`2`, default **2**.
- `SubstepResolution` — multiplier on the adaptive substep count (default `1.0`);
  a knob for convergence testing without recompiling.
- `StrippingTimescaleFactor` — `f` in `t_strip = f·t_dyn` (default `1.0`).

Legacy behaviour (`0`) is reproduced bit-for-bit, so it remains available and
fully reproducible.

## 3. The CGM-stripping fix (a real bug)

In the default CGM recipe most satellites keep their hot phase in `CGMgas`
(`HotGas ≈ 0`). The stripping trigger gated on `HotGas > 0`, so **CGM-regime
satellites were never stripped at all** — environmental stripping was silently
switched off for the majority of satellites. Fixed by triggering on
`HotGas > 0 || CGMgas > 0`; the routine already strips the correct reservoir
(CGM-regime → `CGMgas`, Hot-regime → `HotGas`) and donates into the central's
regime-appropriate reservoir. Legacy (`CGMrecipeOn=0`) runs are unaffected
(`CGMgas` is zeroed for satellites there). *This changes the calibrated baseline
and requires the regression baseline to be regenerated.*

## 4. Cadence dependence and cross-simulation consistency

Legacy strips a fixed ~65% **per snapshot** regardless of `dT`; the physical rule
strips `1−exp(−dT/t_dyn)`. The two simulations sample time differently:

| | low-z `dT` | host `t_dyn` | `dT/t_dyn` | physical strip/snapshot |
|---|---|---|---|---|
| Millennium (64 snaps) | 0.29 Gyr | 1.34 Gyr | **0.22** | 19% |
| microUchuu (50 snaps) | 0.59 Gyr | 1.44 Gyr | **0.41** | 33% |

So **legacy stripping gives a different answer for the same physical satellite
depending on how finely the trees were saved** — a spurious component of any
Millennium-vs-microUchuu comparison. The physical rule depends only on elapsed
time and is cadence-invariant.

Quantified with a **2D (host-mass × satellite-mass) matched** comparison of the
fully-stripped satellite fraction:

```
cross-sim |Millennium − microUchuu|:  legacy = 0.131,  physical = 0.096   → 27% reduction
```

i.e. the physical scheme removes ~1/4 of the cross-simulation discrepancy — the
cadence artefact — leaving a residual that is genuine resolution physics
(different halo/subhalo populations, infall histories). A matched-*stellar*-mass
cut gives the same ~26%; a naive matched-*host*-mass-only cut is confounded by
the satellite mass-resolution difference and must not be used alone.
(See `plotting/stripping_cadence_analysis.py` → `stripping_cadence_story.png`.)

## 5. Calibrating the timescale factor `f` — and why it can't be tuned from data

A sweep `f = StrippingTimescaleFactor ∈ {0.5, 1, 2, 4}` on Millennium:

| across the `f` range | response |
|---|---|
| CGM (hot halo) mean | **+36%** (works as intended) |
| satellite HI ratio | flat (< 0.06 dex) |
| satellite cold-gas fraction | flat (~1%) |
| satellite quenched fraction | flat (0.326 → 0.323) |

`f` strongly controls the **hot halo** — the reservoir stripping acts on — but has
**essentially no effect on any observable** (HI, cold gas, quenching). In SAGE26
the hot halo is decoupled from the cold ISM on satellite timescales: stripping
removes the future *cooling supply*, but the cold gas is consumed by SF /
reheated by feedback faster than the lost supply matters.

**Consequences:** (a) `f` cannot be calibrated from HI/quenching data — it should
be set on physical grounds (`f = 1`, one host dynamical time); (b) the observable
predictions are *robust* to the entire stripping choice, so mode 2 is a free
improvement. (See `plotting/calibrate_stripping.py` → `stripping_calibration.png`.)

## 6. Satellite HI over-production (a separate problem)

SAGE26 satellites are HI-rich vs **xGASS satellites** (Stevens+19) by **+0.4–0.6
dex** at intermediate mass — robust across resolution (microUchuu shows the same
+0.36 to +0.54 dex).

Decomposition:
- **Not partitioning:** the atomic fraction `HI/ColdGas ≈ 0.65` is normal and
  identical for satellites and centrals. The HI excess tracks a **cold-gas** excess.
- **Global component (~0.2 dex):** the total HI mass function sits ~0.2 dex above
  Zwaan+05 at `logMHI ~ 9–10` and has a **too-extended high-mass tail** (the model
  makes `MHI > 10^10.5` galaxies that the data lacks). This is central-dominated
  (satellites are only 11–25% of the HIMF) → a global **SF-efficiency / feedback /
  cooling** calibration issue.
- **Satellite-specific component (~0.2–0.3 dex):** the *extra* deficit that only
  satellites should carry needs **direct cold-gas (ram-pressure) stripping** — a
  process SAGE26 does not have. Hot-halo stripping (this work) cannot supply it.

```
satellite HI excess (~0.5 dex) ≈ global cold-gas excess (~0.2) + missing cold-ISM stripping (~0.2–0.3)
```

(See `himf_check.png`.) *Caveat:* ~0.1–0.15 dex of the global normalisation is
h-convention/IMF systematics; the robust global signal is the high-mass tail
shape. The low-`MHI` deficit is Millennium resolution incompleteness.

---

## Recommendations

1. **Keep the CGM fix** — it is a correctness fix; regenerate the regression baseline.
2. **Adopt mode 2 (dynamical-time, analytic) as the default**, `f = 1` — more
   defensible physics, removes the cadence artefact, no observational cost. Keep
   the toggle so legacy stays reproducible.
3. **Do not tune `f` against HI/quenching** — it doesn't move them; fix its value
   on physical grounds.
4. **Treat the satellite-HI mismatch as separate future work** requiring (a) a
   global cold-gas/feedback recalibration and (b) a new cold-ISM stripping term.

## Artefacts

- Code (branch `option2-physical-tstrip`): `src/model_infall.c`,
  `src/core_build_model.c`, `src/core_read_parameter_file.c`, `src/core_allvars.h`;
  the CGM fix is also on `dev` (`f2dfd73`).
- Diagnostics: `plotting/compare_stripping.py` (`--subset` cut),
  `plotting/stripping_cadence_analysis.py`, `plotting/calibrate_stripping.py`.
- Figures: `stripping_cadence_story.png`, `stripping_calibration.png`,
  `himf_check.png`.
- Sweep configs (gitignored): `input/millennium_f{05,10,20,40}.par`.

# Star Formation and Feedback

This page covers the gas-to-stars conversion step: which star-formation
prescription is used, how supernova feedback returns gas to the hot/CGM
reservoir and ejects some beyond the halo, and how feedback-free burst (FFB)
galaxies bypass the SN loop entirely.

Source: [`src/model_starformation_and_feedback.c`](https://github.com/MBradley1985/SAGE26/blob/main/src/model_starformation_and_feedback.c)

Called from: [Per-halo physics loop](../core_build_model.md) -- step 9 of the
substep ordering.

## Per-substep orchestration -- `starformation_and_feedback()`

The function executes the following block once per galaxy per substep:

1. **FFB early exit.** If `FeedbackFreeModeOn >= 1` and the galaxy is in FFB
   regime (`FFBRegime == 1`, set in `core_build_model.c`), control jumps to
   `starformation_ffb()` and the standard SF/feedback path is skipped.
2. **Compute the SFR** via the prescription selected by `SFprescription`
   (see below). Returns a star formation rate `strdot` in code units.
3. **Compute H1/H2 split** for H2-tracking prescriptions (1, 3, 4, 5, 6, 7),
   storing `H2gas` and `H1gas` on the galaxy. `H1gas` is the atomic remainder
   `X_H * ColdGas - H2gas`, from which the Shark-style ionised-outer-disk
   fraction (gas below `Sigma_HI_crit = 0.5` Msun/pc^2) is removed -- the cut
   is applied to that *remainder* (H2 is central and shielded), so the
   molecular and ionised components cannot overdraw the hydrogen budget and
   HI is non-negative by construction.
4. **Compute reheated and ejected masses.** Standard Croton+06 budget, or
   the FIRE-scaling alternative (Muratov+15) when `FIREmodeOn = 1`.
5. **Apply mass and metal updates** via `update_from_star_formation()` and
   `update_from_feedback()`.
6. **Record the SFR** in the fixed-size `SfrDisk[step]` history bin.
7. **Check disk instability** (`check_disk_instability()` if
   `DiskInstabilityOn`).
8. **Produce metals** via the instantaneous-recycling approximation and
   route the fraction that leaves the disk to either `CGMgas` (Regime 0)
   or `HotGas` (Regime 1) using the Krumholz & Dekel (2011) Eq. 22
   suppression factor `FracZleaveDisk * exp(-Mvir / 3e11)`.

## Star formation prescriptions

The `SFprescription` parameter selects one of eight recipes. All share the
same effective star-forming radius `r_eff = 3.0 * DiskScaleRadius` (Milky
Way calibration) and dynamical time `t_dyn = r_eff / V_vir`.

| Value | Recipe | What sets the SFR |
|-------|--------|-------------------|
| 0 | Croton+2006 | Kauffmann+96 cold-gas threshold; `SFR = epsilon * (ColdGas - cold_crit) / t_dyn` |
| 1 | Blitz & Rosolowsky 2006 (BR06) | H2 from pressure-based fitting; `SFR = epsilon * H2gas / t_dyn` |
| 2 | Somerville+25 (no H2) | Density-modulated efficiency; `SFR = epsilon_cl * f_dense * ColdGas / t_dyn` |
| 3 | Somerville+25 + H2 | Same as 2, but with `ColdGas` replaced by `H2gas` from BR06 |
| 4 | Krumholz & Dekel 2012 (KD12) | Metallicity-dependent H2 fraction; `SFR = epsilon * H2gas / t_dyn` |
| 5 | Krumholz, McKee & Tumlinson 2009 (KMT09) | Photodissociation-balance H2; same form as KD12 |
| 6 | Krumholz 2013 (K13) | Two-phase H2 model |
| 7 | Gnedin & Draine 2014 (GD14) | UV-modulated H2; same form as KMT09 |

The H2-tracking prescriptions compute `H2gas` and `H1gas` either via a
single-slab calculation (cheaper) or a radial ring integration
(`H2RadialIntegrationOn = 1`, more accurate). The single-slab disk area is
selected by `H2DiskAreaOption`:

| Value | Area |
|-------|------|
| 0 | pi * r_disk^2 |
| 1 | pi * (3 r_disk)^2 (default; matches Milky Way calibration) |
| 2 | 2 pi * r_disk^2 |

The radial mode uses `H2RadialNBins` rings out to `H2RadialRMaxFactor *
r_disk`.

## Supernova feedback

Two reheating/ejection budgets are available, selected by `FIREmodeOn`.

### Standard Croton+2006 budget (`FIREmodeOn = 0`)

- **Reheated mass:** `reheated_mass = FeedbackReheatingEpsilon * stars`.
  Transferred from `ColdGas` to `HotGas` (or `CGMgas` in Regime 0).
- **Ejected mass:** energy balance with the unused SN energy:
  `ejected_mass = (FeedbackEjectionEfficiency * eta_SN * E_SN / V_vir^2
                    - FeedbackReheatingEpsilon) * stars`.
  Transferred from `HotGas` (or `CGMgas`) to `EjectedMass`. Clipped to zero
  if negative (deeper potentials retain all the reheated gas).

### FIRE scaling (`FIREmodeOn = 1`)

Replaces the fixed coefficients with the Muratov et al. (2015) velocity-
and redshift-dependent scaling:

```
fire_scaling = (1 + z)^RedshiftPowerLawExponent * (V_vir / V_crit)^beta
```

with `V_crit = 60 km/s` and the broken-power-law exponent
`beta = -3.2` below `V_crit`, `-1.0` above. The reheating mass becomes
`eta_reheat = FeedbackReheatingEpsilon * fire_scaling`, and the ejection
mass is computed from the surplus SN energy after lifting the reheated
gas (Hirschmann+2016 energy budget):

```
E_FB = eps_eff * 0.5 * stars * eta_SN * E_SN,  eps_eff = FeedbackEjectionEfficiency * fire_scaling
E_lift = 0.5 * reheated_mass * V_vir^2
ejected_mass = max(E_FB - E_lift, 0) / (0.5 * V_vir^2)
```

The FIRE coefficient is computed once per call and reused for both reheating
and ejection.  `V_vir` is used throughout -- for the FIRE scaling and for the
lift term -- in the main SF path, the FFB path, and merger starbursts (which
use the merger central's `V_vir`).  No disk circular velocity enters the
feedback.  A floor `max(V_vir, 1 km/s)` is applied to the FIRE scaling only.

Two consequences of this form are worth stating explicitly.

**The FIRE scaling cancels in the ejection threshold.** Because `fire_scaling`
multiplies both `E_FB` (through `eps_eff`) and `E_lift` (through
`reheated_mass = eta_reheat * stars`), it drops out of the ratio:

```
E_FB / E_lift = FeedbackEjectionEfficiency * eta_SN E_SN
                / (FeedbackReheatingEpsilon * V_vir^2)
ejected_mass / stars = eta_reheat * max(E_FB/E_lift - 1, 0)
```

The threshold is therefore a function of `V_vir` alone, independent of
redshift: with the fiducial parameters ejection operates below
`V_vir = 161 km/s` at every epoch and never above it.  Read physically, FIRE
sets how much mass is launched out of the ISM and the halo potential alone
sets what fraction of it escapes the halo.

**The effective coupling is bounded by the SN budget.** `eps_eff =
FeedbackEjectionEfficiency * fire_scaling` inherits the
`(1+z)^alpha (V_vir/60)^beta` scaling and is unbounded in itself, so `E_FB`
would otherwise exceed the total SN energy `stars * eta_SN * E_SN` released by
the stars driving it (that happens when `eps_eff > 2`).
`SNEnergyConservationOn`, on by default, caps `eps_eff` at
`MaxSNEnergyCoupling` (default 2.0 -- the whole SN budget; 1.0 caps it at
half).  The bound applies to the energy only; the empirical FIRE mass loading
in `eta_reheat` is left untouched.  Setting `SNEnergyConservationOn = 0`
restores the unbounded coupling of the earlier published behaviour.

**The reheating term has its own optional bound.** `mdot_reheat = eta_reheat * mdot_*`
carries no energy check by default. Where the model ejects, the total energy
spent is exactly `E_FB` and is bounded by `SNEnergyConservationOn`; the residual
is the non-ejecting regime (`V_vir` above the 161 km/s ejection threshold), where
the reheating cost `0.5 * eta_reheat * V_vir^2` grows linearly with `V_vir`.
Setting `ReheatEnergyConservationOn = 1` caps `eta_reheat` so that cost cannot
exceed the same `MaxSNEnergyCoupling` budget. It uses the `E_lift` cost
convention, so the two equations agree about what reheating costs. Default on;
set to 0 to recover the unbounded mass loading.

Note that `Vvir` in the output catalogue is the *instantaneous* virial velocity,
recomputed at output time, whereas the feedback is evaluated with the
peak-retained `galaxies[p].Vvir` (updated only when `Mvir` grows). The two
diverge for haloes past their peak -- by a factor of two at `V_vir ~ 11 km/s`
in microUchuu, and by under 3 per cent above ~25 km/s. `VvirPeak` in the output
carries the value the physics used; pair feedback quantities with that, not with
`Vvir`.

**Neither `eta_reheat` nor `reheated_mass` is capped directly.** What limits
the reheating is the cold gas balance: if `stars + reheated_mass > ColdGas`
both are rescaled by `ColdGas / (stars + reheated_mass)`, so the stars formed
in a substep cannot exceed `ColdGas / (1 + eta_reheat)`.  In practice this
rescale fires in a few tenths of a per cent of star formation events and
removes well under 1 per cent of the stars formed.  `ejected_mass` is
separately capped at the available `HotGas` (or `CGMgas` in Regime 0) inside
`update_from_feedback()`.

## `update_from_star_formation()` and `update_from_feedback()`

These two helpers do the actual reservoir bookkeeping.

**`update_from_star_formation()`** removes `(1 - RecycleFraction) * stars`
from `ColdGas` (the rest is instantaneously recycled), increments
`StellarMass`, and tracks metallicity. H1 and H2 are clamped so they remain
consistent with the post-SF cold gas.

**`update_from_feedback()`** transfers `reheated_mass` from `ColdGas` to
the central's hot reservoir (regime-aware: `CGMgas` if Regime 0,
`HotGas` if Regime 1), then transfers `ejected_mass` from that same
reservoir to `EjectedMass`. Metals follow the gas in both transfers.

## Metal production and routing

After SF and feedback the function produces new metals via the
instantaneous-recycling approximation: `metals_new = Yield * stars`. A
fraction stays in the disk and the rest leaves to the hot reservoir,
controlled by:

```
FracZleaveDiskVal = FracZleaveDisk * exp(-Mvir / 30)
```

(Krumholz & Dekel 2011 Eq. 22; mass in `10^10 Msun/h`, so the
characteristic scale is `3 x 10^11 Msun/h`). The leaving fraction routes
to `MetalsCGMgas` in Regime 0 and `MetalsHotGas` in Regime 1
(or always `MetalsHotGas` when `CGMrecipeOn = 0`). If `ColdGas` is
exhausted, all metals leave the disk.

## Feedback-free burst mode -- `starformation_ffb()`

When the FFB regime classification flags a galaxy as bursty, the SFR is
set directly by the FFB efficiency rather than the usual Kauffmann
threshold:

```
SFR = FFBMaxEfficiency * gas_for_sf / t_dyn
```

The "feedback-free" label refers to the **physical regime**, not to the
code path. The mechanism (Li+2024, Boylan-Kolchin+2025): in dense compact
gas reservoirs the SN energy escapes the cloud before it can disrupt the
burst, so the *star formation efficiency* reaches
`epsilon_FFB ~ 0.2-1.0` instead of the usual few percent. The code still
bookkeeps the resulting SN feedback; it just does not let feedback
throttle the SFR.

Specifically, the only differences between `starformation_ffb()` and the
standard `starformation_and_feedback()` path are:

1. **SFR formula:** `epsilon_FFB * gas_for_sf / t_dyn` -- no cold-gas
   threshold, no mediation by molecular fraction in `ColdGas` modes.
2. **No disk-instability check** after SF -- rapid burst SF is assumed
   to stabilise the disk.

Everything else still runs:

- SN reheating: `reheated_mass = FeedbackReheatingEpsilon * stars` (or
  the FIRE-scaled `eta_reheat * stars` when `FIREmodeOn = 1`).
- SN ejection: the same energy-balance budget (Croton or FIRE form) as
  the main path.
- `update_from_feedback()` transfers reheated and ejected mass with
  full regime-aware routing.
- Metal production and routing via the same Krumholz & Dekel (2011)
  Eq. 22 factor, regime-aware to `MetalsCGMgas` or `MetalsHotGas`.

`gas_for_sf` is either the full `ColdGas` (modes 1-5) or the molecular
fraction `H2gas` (modes 6-7). For H2 modes the H2 calculation is run
inline using whichever underlying SFprescription (BR06, KD12, KMT09,
K13, GD14) is set.

The seven sub-modes of `FeedbackFreeModeOn` (1-7) control which threshold
classifies a galaxy as FFB-eligible (Li+2024 vs Boylan-Kolchin+2025, sigmoid
vs sharp, concentration source). The classification itself lives in
`determine_and_store_ffb_regime()`; this function only consumes the
`FFBRegime` flag.

## What is NOT in this module

- **Quasar-mode AGN.** BH accretion driven by mergers is handled in
  `model_mergers.c`. See [Mergers and disruption](mergers_and_disruptions.md).
- **Radio-mode AGN.** Handled in `model_cooling_heating.c`. See
  [Cooling and AGN heating](cooling_and_heating.md).
- **Disk instability.** Called from this module (step 7 above) but
  implemented in `model_disk_instability.c`. See the dedicated
  [Disk instability](disk_instability.md) page.

## Switches and parameters

| Parameter | Effect |
|-----------|--------|
| `SFprescription` | Select SF recipe (0-7). |
| `SfrEfficiency` | Efficiency factor in `epsilon * (Cold or H2) / t_dyn`. Used by SFprescription 0, 1, 4, 5, 7 always, and by 6 in slab mode only. Unused by 2 and 3 (Somerville+25) and by 6 in radial mode. |
| `SupernovaRecipeOn` | 0 disables SN feedback entirely. |
| `FeedbackReheatingEpsilon` | Mass-loading scaling for the reheating term. |
| `FeedbackEjectionEfficiency` | Fraction of SN energy that drives ejection. |
| `EnergySN`, `EtaSN` | Energy per SN and number of SN per solar mass. |
| `Yield`, `RecycleFraction` | Metal yield and instantaneous recycling fraction. |
| `FracZleaveDisk` | Fraction of new metals that leave the disk (modulated by halo mass). |
| `FIREmodeOn` | Switch to FIRE/Muratov+15 reheating and ejection. |
| `RedshiftPowerLawExponent` | Redshift exponent in the FIRE scaling. |
| `FeedbackFreeModeOn` | FFB regime classification mode (0-7); 0 disables the FFB path. |
| `FFBMaxEfficiency` | SF efficiency in FFB mode. |
| `H2DiskAreaOption`, `H2RadialIntegrationOn`, `H2RadialNBins`, `H2RadialRMaxFactor` | H2 surface-density geometry. |
| `DiskInstabilityOn` | Run the Toomre check after SF. |

See [`parameters.md`](../parameters.md) for full descriptions and defaults.

## References

- Croton et al. (2006), MNRAS 365, 11 -- original SAGE star formation and
  feedback recipes.
- Kauffmann (1996), MNRAS 281, 487 -- cold-gas threshold.
- Blitz & Rosolowsky (2006), ApJ 650, 933 -- pressure-based H2 fraction.
- Krumholz, McKee & Tumlinson (2009), ApJ 693, 216 -- photodissociation
  H2 model.
- Krumholz, Dekel & McKee (2011), ApJ 745, 69 -- metal-leaving-disk
  scaling (Eq. 22).
- Krumholz & Dekel (2012), ApJ 753, 16 -- KD12 H2 prescription.
- Krumholz (2013), MNRAS 436, 2747 -- two-phase H2.
- Gnedin & Draine (2014), ApJ 795, 37 -- UV-modulated H2.
- Somerville et al. (2025) -- density-modulated SF efficiency.
- Muratov et al. (2015), MNRAS 454, 2691 -- FIRE wind mass loading.
- Hirschmann et al. (2016), MNRAS 461, 1760 -- energy-balance ejection.
- Li et al. (2024) -- feedback-free burst threshold.
- Boylan-Kolchin (2025) -- FFB at high redshift.

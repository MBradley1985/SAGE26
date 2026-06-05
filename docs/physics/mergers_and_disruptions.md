# Mergers and Disruption

This page covers what happens when two galaxies merge or when a satellite
is tidally destroyed: how the events are classified, how mass and metals
flow between the merging pair, how the starburst and quasar-mode AGN
operate, and how the bulge grows in the two-channel SAGE26 model.

Source: [`src/model_mergers.c`](https://github.com/MBradley1985/SAGE26/blob/main/src/model_mergers.c)

Called from: [Per-halo physics loop](../core_build_model.md) -- step 10 of
the substep ordering (the satellite/merger block after the per-galaxy
cooling and SF loop).

## When this step fires

Each satellite carries a `MergTime` set when it first becomes a Type 1
(by `estimate_merging_time()`, computed from a dynamical-friction estimate
based on satellite/host mass ratio and the satellite's orbital radius at
infall). At each substep:

1. `MergTime` is decremented by `deltaT / effective_steps`.
2. If `M_vir / baryon_mass <= ThresholdSatDisruption` (or the satellite
   has zero baryons), the satellite is flagged for an event.
3. If `MergTime > 0`, the satellite is **disrupted** to ICS via
   `disrupt_satellite_to_ICS()`.
4. If `MergTime <= 0`, the satellite has reached the central -- the
   event is a **merger** handled by `deal_with_galaxy_merger()`.

Type 1 satellites merge into the central of the host halo. Type 2
(orphan) satellites merge into whatever was their central at the time
their subhalo was lost.

## `deal_with_galaxy_merger()` -- the merger dispatcher

The function classifies and processes one merger event. Mass ratio is
computed from baryonic mass (`StellarMass + ColdGas`):

```
mass_ratio = min(m_sat, m_central) / max(m_sat, m_central)
```

The threshold `mass_ratio > ThreshMajorMerger` (default 0.3) classifies
the event as a major merger; otherwise it is minor. Before any reservoir
transfers happen, the function decides where burst stars will go:

| Merger type | Central morphology | Burst destination |
|-------------|--------------------|--------------------|
| Major | irrelevant | Merger-driven bulge |
| Minor | disk-dominated (disk > 0.5 stellar) | Instability bulge |
| Minor | spheroid-dominated | Merger-driven bulge |

This morphology-aware routing is the key SAGE26 change to the original
Croton+06 merger model -- minor mergers onto disks no longer pollute the
merger-driven bulge channel.

### Step-by-step execution

1. `add_galaxies_together()` -- transfers the satellite's gas, stars,
   metals, BH mass, and ICS into the central. Regime-aware for the gas
   reservoirs (CGMgas vs HotGas).
2. `grow_black_hole()` -- quasar-mode BH accretion if `AGNrecipeOn > 0`
   (see below).
3. `collisional_starburst_recipe()` -- the merger-driven starburst (see
   below).
4. `get_bulge_radius()` -- recomputes the displayed `BulgeRadius` after
   bulge masses changed.
5. `calculate_merger_remnant_radius()` -- energy-conservation calculation
   for the post-merger bulge radius.
6. **Major merger branch:** `make_bulge_from_burst()` destroys the disk
   (all stellar mass becomes bulge), sets `MergerBulgeRadius` from the
   energy-conservation calculation, stamps `TimeOfLastMajorMerger`,
   and marks the satellite `mergeType = 2`.
7. **Minor merger branch:** mark `mergeType = 1`, stamp
   `TimeOfLastMinorMerger`, and either update `InstabilityBulgeRadius`
   (disk-dominated central) or `MergerBulgeRadius` (spheroid-dominated)
   from the energy-conservation calculation.

## Quasar-mode AGN -- `grow_black_hole()`

Triggered on every merger (major or minor). Accreted mass is:

```
BHaccrete = BlackHoleGrowthRate * mass_ratio
            * ColdGas / (1 + (280 km/s / V_vir)^2)
```

So accretion scales with merger violence and is suppressed in shallow
potential wells (the `V_vir = 280 km/s` floor is hard-coded from
Croton+06). The accreted gas is removed from `ColdGas` with metallicity
tracking, deposited into `BlackHoleMass`, and tallied in
`QuasarModeBHaccretionMass`.

### `quasar_mode_wind()` -- ejection from quasar-mode energy

Each merger then computes the quasar wind energy:

```
E_quasar = QuasarModeEfficiency * 0.1 * BHaccrete * c^2
```

(the `0.1` is the radiative efficiency.) The function compares
`E_quasar` against successive reservoir binding energies:

1. If `E_quasar > 0.5 * ColdGas * V_vir^2`, the entire cold reservoir is
   ejected to `EjectedMass`.
2. Then it checks against the hot reservoir (regime-aware: `HotGas` for
   Regime 1, `CGMgas` for Regime 0). If `E_quasar` exceeds the combined
   cold + hot energy, the hot reservoir is ejected too.

This is what gives bright quasars the ability to expel the entire
baryonic content of low-mass hosts.

## `collisional_starburst_recipe()` -- the merger starburst

Implements the Somerville+2001 / Cox PhD-thesis form:

```
eburst = STARBURST_FRAC_COEFF * mass_ratio^STARBURST_MASS_POWER
       = 0.56 * mass_ratio^0.7    (mergers, mode == 0)
eburst = mass_ratio               (disk instabilities, mode == 1)
```

`stars_burst = eburst * gas_for_starburst` becomes the burst stellar
mass. The `gas_for_starburst` is normally `ColdGas`, but when
`StarburstColdGasOn = 0` and an H2-tracking SFprescription is in use,
it is recomputed from the current `ColdGas` using the same H2 recipe as
the disk SF path -- this avoids the stale stored `H2gas` value if SF and
feedback have already depleted cold gas earlier in the substep.

The burst then applies SN feedback through the same `update_from_feedback()`
helper used by `starformation_and_feedback()`, including FIRE scaling when
`FIREmodeOn = 1`, and routes the burst stars into either the
`MergerBulgeMass` or `InstabilityBulgeMass` channel per the
`burst_to_merger_bulge` flag set in `deal_with_galaxy_merger()`.

The SFR is recorded into `SfrBulge[step]` (separate from disk SF, which
goes into `SfrDisk[step]`).

## The two-channel bulge model

Every bulge-growth path in SAGE26 routes its contribution into one of two
channels:

| Channel | Mass field | Radius field | Sources |
|---------|------------|--------------|---------|
| Merger | `MergerBulgeMass` | `MergerBulgeRadius` | Major mergers (whole disk); minor merger bursts onto spheroid-dominated centrals |
| Instability | `InstabilityBulgeMass` | `InstabilityBulgeRadius` | Toomre instability; minor merger bursts onto disk-dominated centrals |

The two channels are tracked independently so that the bulge formation
history can be decomposed into merger-driven and secular contributions.
The combined `BulgeMass = MergerBulgeMass + InstabilityBulgeMass` and
the displayed `BulgeRadius` is a mass-weighted average computed by
`get_bulge_radius()` in [`model_misc.c`](https://github.com/MBradley1985/SAGE26/blob/main/src/model_misc.c)
(Tonini+2016 prescription, `BulgeSizeOn = 3`).

Merger remnant radii are set via energy conservation
(`calculate_merger_remnant_radius()` -- Covington+11):

```
1 / R_new = 1 / R_pair + C_rad * (m1 * m2) / (m_total^2 * (R1 + R2))
```

with `C_rad = 2.75`, where `R_pair` is the mass-weighted progenitor
radius and the second term accounts for orbital energy dissipated during
the encounter.

## Satellite disruption -- `disrupt_satellite_to_ICS()`

When the satellite has reached the central before the merger clock runs
out, it is disrupted instead of merged. The function:

1. **Transfers gas** to the central (regime-aware: total
   `ColdGas + HotGas + CGMgas` goes to the central's `CGMgas` if Regime 0
   or `HotGas` if Regime 1).
2. **Transfers ejected mass and pre-existing ICS** unchanged.
3. **Disrupts the satellite's stellar mass** -- splits it between the
   central's ICS and the central's stellar mass (BCG accretion). The
   split is controlled by `DynamicDisruptionSplit`:

| Value | Split | Formula |
|-------|-------|---------|
| 0 | Fixed | `f_ICS = FractionDisruptedToICS` |
| 1 | Mass-ratio | `f_ICS = 1 - (Msub / Mhost)^DisruptionSplitAlpha` -- low mass-ratio satellites are stripped on wider orbits and contribute more to ICS |
| 2 | Mass-ratio with concentration weighting | as mode 1, but with `alpha_eff = DisruptionSplitAlpha * DisruptionSplitCref / c_sat` -- concentrated satellites resist stripping and deposit more onto the BCG |

4. **Records assembly history** if `TrackICSAssembly = 1`:
   `ICS_disrupt` accumulates the satellite stellar mass newly disrupted
   into ICS; `ICS_accrete` accumulates ICS that the satellite already
   carried in. `ICS_sum_mt` tracks the mass-weighted deposit time so that
   the mean ICS-assembly time reflects when the stars were originally
   stripped, not when this packet transferred into the central.

The satellite is then marked `mergeType = 3` (disrupted, not merged) and
will be skipped on subsequent substeps.

## What is NOT in this module

- **Radio-mode AGN.** Handled in `model_cooling_heating.c`. See
  [Cooling and AGN heating](cooling_and_heating.md).
- **Disk SF.** Handled in `model_starformation_and_feedback.c`. See
  [Star formation and feedback](starformation_and_feedback.md).
- **Toomre disk instability.** Implemented in `model_disk_instability.c`.
  The starburst it triggers reuses `collisional_starburst_recipe(mode=1)`.
- **`BulgeRadius` calculation.** Implemented in `model_misc.c`
  (`get_bulge_radius()`). The merger code only sets `MergerBulgeRadius`
  and lets `get_bulge_radius()` derive the displayed `BulgeRadius`.

## Switches and parameters

| Parameter | Effect |
|-----------|--------|
| `AGNrecipeOn` | 0 disables quasar-mode BH growth and wind during mergers. |
| `BlackHoleGrowthRate` | Scaling for `grow_black_hole()` accretion. |
| `QuasarModeEfficiency` | Coupling between BH accretion energy and wind energy. |
| `ThreshMajorMerger` | Mass-ratio threshold for major vs minor classification. |
| `ThresholdSatDisruption` | M_vir/baryon threshold below which a satellite is eligible for an event. |
| `StarburstColdGasOn` | 0 forces the burst to recompute H2 from current ColdGas; 1 uses the stored value. |
| `DynamicDisruptionSplit` | ICS-vs-BCG split mode for disrupted satellites (0 fixed, 1 mass-ratio, 2 mass-ratio with concentration weighting). |
| `FractionDisruptedToICS` | Fixed-split fraction (mode 0) or fallback. |
| `DisruptionSplitAlpha` | Exponent in the mass-ratio split (modes 1 and 2). |
| `DisruptionSplitCref` | Reference concentration for mode 2. |
| `BulgeSizeOn` | Bulge radius model (0 off, 1-2 Shen+2003, 3 Tonini+2016 multi-channel). |
| `TrackICSAssembly` | Record `ICS_disrupt` and `ICS_accrete` assembly history. |
| `SFprescription` | Used to pick the H2 recipe in the burst when `StarburstColdGasOn = 0`. |
| `FIREmodeOn` | Applies FIRE scaling to burst SN feedback if enabled. |

See [`parameters.md`](../parameters.md) for full descriptions and defaults.

## References

- Croton et al. (2006), MNRAS 365, 11 -- original SAGE merger and AGN
  prescriptions.
- Somerville et al. (2001), MNRAS 320, 504 -- collisional starburst form.
- Cox et al. (2008), MNRAS 384, 386 -- merger-driven SF efficiency calibration.
- Covington et al. (2011), ApJ 743, 76 -- energy-conservation remnant radius.
- Tonini et al. (2016), MNRAS 459, 4109 -- two-channel bulge formation.
- Shen et al. (2003), MNRAS 343, 978 -- bulge size-mass relation.
- Hopkins et al. (2010), MNRAS 401, 1099 -- merger-driven bulge growth.
- Dynamical-friction merger timescale: Binney & Tremaine (2008) eq. 8.12.

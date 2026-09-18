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

### `DisruptionGate` -- the Contini et al. (2014) orphan gate

With `DisruptionGate = 0` (the default) step 3 above destroys the
satellite outright. Because an orphan's `M_vir` is set to zero the moment
its subhalo leaves the tree, and the `currentMvir` ramp in
`evolve_galaxies()` reaches exactly zero on the final substep, the
condition in step 2 always succeeds for an orphan: every orphan is
therefore destroyed within the snapshot in which it was created, and
`ThresholdSatDisruption` cannot prevent it.

`DisruptionGate = 1` replaces that behaviour, for Type 2 galaxies only,
with the prescription of Contini et al. (2014). Their model Disr.
(Sec. 3.1, following Guo et al. 2011) requires the halo to be dense
enough at the satellite's pericentre to unbind it:

1. The parent halo is a singular isothermal sphere, `phi(R) = Vvir^2 ln R`
   (their eq. 2). The satellite's pericentre follows from conserving
   energy and angular momentum along the orbit (their eq. 3), solved by
   bisection.
2. The halo density there is `rho_halo = M(<R_peri) / R_peri^3`, which for
   the isothermal profile reduces to `Vvir^2 / (G R_peri^2)`.
3. The satellite density is `rho_sat = M_sat / R_half^3` with
   `M_sat = StellarMass + ColdGas` and `R_half` the mass-weighted mean of
   the disk and bulge half-mass radii (their eq. 4).
4. If `rho_halo <= rho_sat` **nothing happens**. The orphan keeps its
   reservoirs and is carried to the next snapshot, where the test is
   repeated. Its `MergTime` keeps counting down, so a survivor eventually
   merges instead of being shredded.

Between those tests the orphan is not left untouched. Henriques & Thomas
(2010) strip stellar material from orphans on *every* timestep, and
`DisruptionGate = 1` does the same:

1. When a galaxy first becomes an orphan, `OrbitRadius` is seeded with the
   virial radius of the halo it is falling into -- the same radius
   `estimate_merging_time()` assumed when it built the merging clock.
2. Their eq. 2 (the dynamical-friction formula SAGE already uses) scales as
   `r_sat^2` at fixed satellite mass, so each substep the orbit shrinks by
   the square root of the fraction of the clock still left. Tying the orbit
   to the existing clock leaves merger *timing* untouched -- only the
   stripping is new.
3. The tidal radius follows their eq. 4,
   `R_t = (1/sqrt(2)) (sigma_sat/sigma_halo) r_sat`. For an isothermal
   sphere `sigma = Vvir/sqrt(2)` on both sides, so only the ratio of virial
   velocities survives.
4. Stellar mass outside `R_t` is unbound and joins the central's ICS: the
   exponential disk contributes `M (1 + x) exp(-x)` with `x = R_t/R_sl`
   (their eq. 6), and the bulge contributes `M a^2/(R_t^2 + a^2)` with
   `a = 0.56 R_b` (their eqs. 8-9). Metals leave in the same proportion,
   as the paper assumes a uniform metallicity. `R_b` is SAGE26's own
   half-mass `BulgeRadius`; Henriques & Thomas recover it from a
   Djorgovski & Davis (1987) relation only because their base model had no
   bulge sizes at all.

This is what moves the ICS from a large undershoot into the observed range;
see the changelog for the measured effect.

When the gate does open, the amount removed is set by the tidal radius of
their Sec. 3.2 rather than by wholesale destruction:

```
R_t = (M_sat / (3 M_DM,halo))^(1/3) * D            (their eq. 5)
```

- `R_t < BulgeRadius`: the satellite is completely disrupted, exactly as
  in the ungated model.
- `R_t < 10 * DiskScaleRadius`: the exponential stellar disk outside `R_t`
  is moved to the central's ICS and the same fraction of the cold gas goes
  to the central's hot phase.
- Otherwise nothing is stripped.

Contini et al. additionally reset the scalelength to `R_t / 10` so the
truncated disk is again truncated at ten scalelengths. **SAGE26 does not
do this**, deliberately: `DiskScaleRadius` is used directly as the
exponential scale length behind `Sigma_0 = M / (2 pi r_s^2)` in the H2 and
star formation prescriptions ([`model_h2_chemistry.c`](https://github.com/MBradley1985/SAGE26/blob/main/src/model_h2_chemistry.c)),
so compressing `r_s` tenfold would *raise* the central surface density of a
galaxy that just lost its outskirts and push its star formation rate up
rather than down. Leaving `r_s` alone keeps the retained material on the
profile it already had inside `R_t` and lets the reduced mass lower
`Sigma_0`, as it should.

Two satellites bypass the gate and take the ungated path, so that they
cannot orbit indefinitely: one carrying no baryons at all, and one whose
orbit is purely radial (no pericentre, so it plunges to the centre).

Because an orphan can now outlive its snapshot, `Type == 2` galaxies
appear in the output catalogue when this gate is on; with it off they
never do.

### What a surviving orphan experiences

An orphan is evolved by the same per-substep loop as everything else: it
cools, forms stars, and feels supernova feedback. It does **not** receive
infall or reincorporation, which are central-only.

It *is* passed to `strip_from_satellite()` once this gate is on. That
function is otherwise restricted to Type 1, a restriction inherited from
SAGE16 where it cost nothing: an orphan was destroyed inside the snapshot
that created it and handed everything to the central anyway. Once orphans
survive, the restriction strands their hot and CGM gas -- it can neither be
stripped nor cooled, since the CGM free-fall time is built from `Mvir` and
an orphan's `Mvir` is zero. Contini et al. (2014) associate no hot
component with satellites at all (their footnote 5), so returning it to the
central is what their model assumes. Including orphans recovers about 91
per cent of that gas: 1.8e13 Msun was stranded in orphans at z = 0 before,
1.6e12 Msun after.

Three things follow from an orphan outliving its snapshot, all handled
under the toggle:

- **Regime.** `determine_and_store_regime()` keys off `Mvir`, which is zero
  for an orphan, so every orphan would be forced into the CGM regime
  regardless of the halo it actually had. The regime is instead frozen
  along with `Rvir`, `Vvir` and `Vmax` at the last snapshot the subhalo was
  resolved.
- **Free-fall time.** In the CGM regime `t_ff` is built from `g = G Mvir /
  Rvir^2`, which vanishes for an orphan, so `t_ff` diverges. The cooling
  rate correctly goes to zero through the `1/(t_cool + t_ff)` term -- an
  orphan does not cool -- but the stored diagnostic would carry an infinity
  into the output, so it is replaced with the `-1` "not evaluated" sentinel.
- **Per-snapshot accumulators.** `SfrDisk`, `SfrBulge`, `Cooling`,
  `Heating`, `OutflowRate` and `QuasarModeBHaccretionMass` are cleared in
  the Type 0/1 branch of `join_galaxies_of_progenitors()`, which an orphan
  skips. They are cleared on the same schedule for surviving orphans, so
  the reported rates do not accumulate across snapshots.

Orphans reaching the output brings one reporting wrinkle with it. `Rvir`
and `Vvir` are normally read from `halos[HaloNr]`, which for Type 0 and
Type 1 is the galaxy's own halo; for an orphan it is the *host*, so every
orphan in a cluster would report that cluster's virial quantities against
its own `Mvir` of zero. Both output writers therefore fall back on the
orphan's own frozen `Rvir` and `Vvir` -- see
[Output fields](../outputs.md).

**Two notions of distance.** The Contini gate above measures `D` from the
stored positions, which for an orphan are frozen at the moment its subhalo
was lost, while the Henriques & Thomas stripping uses the decaying
`OrbitRadius`. Feeding `OrbitRadius` into the gate as well was tested and
rejected: it opens the gate so readily that almost no orphan survives
(1557 rather than 7906 at z = 0) and the ICS returns to the overshoot the
gate exists to remove (60 per cent in group-scale haloes rather than 28).
The frozen separation is the more conservative of the two and is what
keeps the ICS in the observed range.

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
   reservoirs (CGMgas vs HotGas). The satellite's existing
   `BulgeMass`, `MergerBulgeMass`, and `InstabilityBulgeMass` are added
   to the central's like-for-like.

   The satellite's **disk** mass (`StellarMass - BulgeMass`) is then
   routed by the central's *post-add* morphology:
   - Disk-dominated central (disk fraction > 0.5): the satellite's disk
     mass joins `InstabilityBulgeMass` and `InstabilityBulgeRadius` is
     updated via Tonini+2016 incremental evolution.
   - Spheroid-dominated central: the satellite's disk mass joins
     `MergerBulgeMass`.

   Note this is independent of the burst-stars routing decided in
   `deal_with_galaxy_merger()` (which uses the *pre-add* morphology
   captured before this step runs).
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
(`calculate_merger_remnant_radius()` -- Covington+11). The function uses
baryonic mass (stellar + cold gas) and a mass-weighted half-mass radius
for each progenitor (disk half-mass = 1.68 * `DiskScaleRadius`, bulge
half-mass = `BulgeRadius`):

```
E_init = M1^2 / R1 + M2^2 / R2                  (self-binding)
E_orb  = M1 * M2 / (R1 + R2)                    (orbital interaction)
E_rad  = C_rad * E_init * f_gas                 (radiative dissipation)
R_final = (M1 + M2)^2 / (E_init + E_orb + E_rad)
```

with `C_rad = 2.75` from Covington+11 and
`f_gas = (ColdGas_1 + ColdGas_2) / (M1 + M2)`. If the total energy comes
out non-positive (very gas-rich pairs at the cap of `E_rad`), the
function falls back to a mass-weighted average of the progenitor radii.

## Satellite disruption -- `disrupt_satellite_to_ICS()`

When the satellite has reached the central before the merger clock runs
out, it is disrupted instead of merged. This is the primary formation
channel for intracluster stars -- see the dedicated
[Intracluster stars (ICS)](ics.md) page for the full lifecycle of the
ICS reservoir. The function:

1. **Transfers gas** to the central (regime-aware: total
   `ColdGas + HotGas + CGMgas` goes to the central's `CGMgas` if Regime 0
   or `HotGas` if Regime 1).
2. **Transfers ejected mass and pre-existing ICS** unchanged.
3. **Disrupts the satellite's stellar mass** -- all of it is added to the
   central's `ICS` and `MetalsICS`. There is no ICS-versus-BCG split, so
   disruption contributes nothing to the central's `StellarMass`,
   `BulgeMass` or `MergerBulgeMass`. The satellite's black hole is
   transferred to the central so that baryons are conserved.

   (Earlier versions split this mass with `DynamicDisruptionSplit`,
   `FractionDisruptedToICS`, `DisruptionSplitAlpha` and
   `DisruptionSplitCref`, including concentration-weighted variants. That
   machinery and those parameters have been removed; `get_halo_concentration()`
   survives and is still used elsewhere, e.g. by the FFB and disk-size models.)

4. **Records assembly history** if `TrackICSAssembly = 1`:
   `ICS_disrupt` accumulates the satellite stellar mass newly disrupted
   into ICS; `ICS_accrete` accumulates ICS that the satellite already
   carried in. `ICS_sum_mt` tracks the mass-weighted deposit time so that
   the mean ICS-assembly time reflects when the stars were originally
   stripped, not when this packet transferred into the central.

The satellite is then marked `mergeType = 4` (disrupted to ICS) and
will be skipped on subsequent substeps. `mergeType` values defined in
`core_allvars.h` are: 0 (still active / no event), 1 (minor merger),
2 (major merger), 3 (reserved for disk instability; not currently set
by any code path), 4 (disrupted to ICS).

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

# Changelog

## `PrecipCriterionOn` now selects the individual factors of the precipitation rate (September 2026) — byte-identical at the default

The CGM inflow rate carries two separable suppression factors,

    mdot = S((threshold - r) / width) * (M_CGM - M_eq) / t_ff,
    r = t_cool/t_ff,   M_eq = M_CGM * r / threshold,

and `PrecipCriterionOn` was a 0/1 switch on both of them at once. It now selects them
individually: 1 = both (default, unchanged), 2 = `M_eq` only, dropping the sigmoid,
3 = sigmoid only, dropping the condensation term, 4 = neither while keeping the rest of
the precipitation path, 0 = neither *and* skipping the hand-over to standard cooling, so
`mdot = M_CGM/t_ff` for every CGM halo however stable.

Modes 1-4 are a 2x2 factorial differing by nothing but the two factors, so **mode 4 is
the reference the single-factor rows should be measured against.** Mode 0 changes the
hand-over as well, so it moves two things at once and is the weaker control. The sigmoid
is evaluated in all four active modes even where it is not multiplied into the rate,
because it also supplies that hand-over test (`f < 0.01`, i.e. `r ~ 19.2`); keeping it
common is what makes the factorial clean. The same gate applies to the
`PreventiveHeatingOn = 5` hot-halo ceiling, built from the same two factors, so the
switch means one thing everywhere.

Mode 1 is bit-for-bit unchanged: verified on mini-Millennium against a rebuild of the
pre-change source, 5440 datasets over 64 snapshots, zero differing.

**Neither factor does much, and the sigmoid does essentially nothing.** Total stellar
mass in the box relative to mode 4:

| mode | z = 0 | z = 1 | z = 2 | z = 4 |
|------|-------|-------|-------|-------|
| 1, both        | 0.9913 | 0.9874 | 0.9850 | 0.9844 |
| 2, no sigmoid  | 0.9913 | 0.9890 | 0.9881 | 0.9887 |
| 3, no `M_eq`   | 0.9996 | 0.9988 | 0.9983 | 0.9972 |

and the stellar mass function offset from mode 4, in dex rms over bins holding at least
10 galaxies:

| mode | z = 0 | z = 1 | z = 2 | z = 4 |
|------|-------|-------|-------|-------|
| 1, both        | 0.008 | 0.014 | 0.008 | 0.020 |
| 2, no sigmoid  | 0.007 | 0.014 | 0.007 | 0.019 |
| 3, no `M_eq`   | 0.005 | 0.006 | 0.004 | 0.014 |

The condensation term supplies essentially all of the criterion's effect: mode 2 (`M_eq`
alone) reproduces mode 1 to within 0.001-0.004 dex, so the sigmoid adds nothing on top
of it. On its own the sigmoid costs 0.03-0.3% of the stellar mass and 0.004-0.014 dex
rms, under the Poisson error in every bin at every redshift. The whole criterion is
worth ~1% of the stellar mass and under 0.02 dex, because the regime split already
routes the marginally stable haloes into the hot branch before the criterion is
consulted: CGM-regime haloes sit at `r ~ 0.1-0.4`, two orders of magnitude below the
threshold of 10, so the sigmoid never leaves its ceiling `S(5) = 0.9933` (inflow-weighted
mean 0.9919 at z = 0 rising to 0.9930 at z = 6, in every mass bin from `log Mvir` 10 to
12.5).

Mode 2 therefore removes `PRECIP_TRANSITION_WIDTH` from the model at no measurable cost.

Modes 0 and 4 agree in the mean to better than 1 part in 10^4 in total stellar mass,
despite the hand-over touching 40% of z = 0 galaxies (max 0.30 dex on any one galaxy,
falling to 0.23% of galaxies and 0.0004 dex by z = 4). The population the hand-over acts
on carries no mass: above the threshold, 96% of haloes have `CGMgas == 0` exactly, where
`t_cool` diverges because the density is zero rather than because the halo is held at
marginal stability, and the whole `r >= 10` population holds 0.014% of the CGM mass.

One defect is preserved rather than fixed, so that it cannot contaminate the factorial:
for `10 <= r < 19.2` the condensation term is zero while the sigmoid is still above the
hand-over threshold, so inflow is exactly zero, whereas at `r >= 19.2` the hand-over
fires and inflow resumes at `M_CGM/t_cool`. Inflow is therefore non-monotonic in
stability across that interval. It affects 0.29% of z = 0 CGM-regime centrals, so it
biases nothing, but it is a defect and it lives in the branch that exists only to
service the sigmoid.

Mode 5 was added afterwards: SAGE16 cold accretion, `mdot = M_CGM / (Rvir/Vvir)`, the rate
the published model used on its `rcool > Rvir` rapid-cooling branch. It bypasses the criterion
like mode 0, but drains on the dynamical rather than the free-fall time. Since
`t_ff = sqrt(2) Rvir/Vvir` exactly for the uniform profile, mode 5 is a uniform `sqrt(2)` = 1.41x
faster than mode 0 -- the same `M_CGM/t` shape with a different constant. It is by far the
largest of the six modes:

| mode | z = 0 | z = 1 | z = 2 | z = 4 |
|------|-------|-------|-------|-------|
| 0, free-fall | 1.0088 | 1.0128 | 1.0152 | 1.0158 |
| 5, SAGE16 `t_dyn` | 1.0345 | 1.0843 | 1.1494 | 1.2409 |

as total stellar mass relative to mode 1 (SMF offsets 0.031-0.096 dex rms). A 41% change in the
inflow-rate normalisation therefore moves the model 3-15x more than deleting the criterion's
two suppression factors altogether -- the normalisation is what matters, not the shape.

Ablation variants `precip`, `sigmoid`, `meq` and `precipfactors` were added to
`plotting/ablation_series.py`, covering modes 0, 2, 3 and 4. The existing `nocgm`
variant removes the CGM machinery wholesale and so cannot separate the Voit criterion
from the regime split plus free-fall accretion; these can.

## Bug fix: SFR outputs were scaled by the substep count (September 2026) — NOT byte-identical

`SfrDisk` and `SfrBulge` in both output formats were divided by the compile-time
`STEPS` (10) rather than by the number of substeps `evolve_galaxies()` actually
integrated. The `Sfr*` arrays hold `STEPS` bins but receive one `+=` per substep, so
when the adaptive scheme (or `SubstepResolution`) makes `effective_steps != STEPS`,
several substeps land in one bin and the bin sum is a sum over substeps. The reported
SFR was therefore the true mean multiplied by `effective_steps / STEPS`.

Verified on mini-Millennium against the SFH mass accumulators, which were always
correct: output/true was exactly 0.500, 2.000 and 2.999 at `SubstepResolution` 0.5,
2.0 and 3.0, and is now 1.000 at every setting. The knob was unusable for its stated
purpose -- convergence testing -- because the diagnostic scaled with `N` by
construction.

At the default `SubstepResolution = 1.0` the adaptive path still fires for a minority
of fast-evolving high-z haloes: 889 of 173 219 galaxy records (0.51%) change, with
`effective_steps` of 11-20, peaking at 1-2% of galaxies per snapshot around z = 2-8.
Their SFR was overstated by 10-100%, but they carry 0.05% of the total SFR, so global
statistics (SMF, cosmic SFRD) are unaffected. **The regression baseline needs
re-capturing** -- `SfrDisk` and `SfrBulge` are the only datasets that change.

`SfrDiskZ` / `SfrBulgeZ` are assignments rather than sums, so exactly
`min(effective_steps, STEPS)` bins carry a value; they now divide by that. This only
matters for `effective_steps < STEPS`, where they were low by the same factor
(`SubstepResolution = 0.5` halved them). Default output is unchanged.

Galaxies now carry `SubstepsUsed`, set per halo per snapshot in `evolve_galaxies()`, and
`model_misc.h` provides `sfr_rate_divisor()` / `sfr_metallicity_divisor()`. The
`SFHMassDisk` / `SFHMassBulge` histories accumulate mass rather than rate and were correct
at every substep count; they are untouched.

The fix is applied in both output paths. It was briefly reverted while chasing a
Millennium mismatch that turned out to be an unrelated `DiskRadiusFactor = 0.55` left in
`input/millennium.par`; with `f_j` restored to 1.0 this fix is the only remaining
difference from the pre-fix model, and it moves 0.05% of the total SFR.

## Plotting: Madau & Dickinson IMF conversion corrected (September 2026)

`paper_plots.py` shifted the Madau & Dickinson (2014) SFRD fit by `x 1/0.63` (+0.20 dex,
Chabrier -> Salpeter) while `load_madau_dickinson_2014_data()` and the SMD equivalent
applied no conversion at all, and no caller compensated. Both are wrong in the same
direction: MD14 is quoted for a Salpeter IMF and SAGE is Chabrier (`whichimf = 1` in
upstream `allresults-history.py`), so the observations must come DOWN by 0.24 dex.

Verified against the repository's own data rather than asserted: MD14's stellar mass
density sits +0.26 dex (median, 23 points over z = 0.45-8) above `data/sfrd/SMD.ecsv`,
which declares `IMF: Chabrier03`. As a second check, the MD14 fit peaked at 0.21
Msun/yr/Mpc^3 as previously plotted, above any published compilation for any IMF; shifted
to Chabrier it peaks at 0.077, inside the accepted 0.08-0.10 range.

`SALPETER_TO_CHABRIER_DEX = -0.24` is now defined once and used by both MD14 loaders, the
fit, and the SMF loader (which already used the literal -0.24). Consequence: the two
observational curves on the CSFRD axes were 0.6 dex apart and are now 0.15 dex apart, and
SAGE26 sits on MD14-Chabrier at cosmic noon rather than 0.2 dex below it. **This changes
figures, not the model.**

## Disk radius behind a toggle (September 2026) — byte-identical at defaults

`DiskRadiusOn` (default 0) selects the disk scale radius model: 0 reproduces the
published Mo+98 expression bit-for-bit, 1 adds a working virial fallback and bounds
`r_d/Rvir`, 2 additionally smooths the halo spin **vector** over a dynamical time, which cuts the
snapshot-to-snapshot jitter in `r_d` by 3x. It does *not* remove the low-particle-count
bias in `|j|` -- that error is correlated between snapshots, so time-averaging cannot
reach it -- and it does not improve the HI mass function. New parameters `DiskRadiusMaxFrac` and
`GasDiskRadiusFactor` (the atomic-to-stellar scale length ratio, applied only in the
HI ionisation cut). See [docs/physics/disk_sizes.md](docs/physics/disk_sizes.md).

## Locked-in physics: six validated toggles removed (July 2026) — byte-identical

After the toggle sweep confirmed the new default-on physics, six parameters
were removed and their calibrated behaviour hardcoded. Each was fixed at its
former default, so output is byte-for-byte identical (mini-Millennium,
`millennium_all`, microUchuu, and the binary benchmark all verify unchanged;
the full unit suite passes).

- **`PhysicalStrippingOn`** — satellite hot-gas stripping is always the
  analytic once-per-snapshot scheme (`1-exp(-dT/t_strip)`, substep-invariant;
  former default 2). The legacy geometric and per-substep schemes and the
  in-loop strip call are gone; `strip_from_satellite()` no longer takes an
  `effective_steps` argument.
- **`StrippingTimescaleFactor`** — the stripping timescale is `t_strip =
  t_dyn(host) = Rvir/Vvir` (former factor 1.0).
- **`PrecipRegulationOn`** — CGM precipitation is always self-regulating,
  condensing only the gas above the `t_cool/t_ff = 10` Voit equilibrium
  (former default 1); the free-fall-drain branch is gone.
- **`HIIonizationOn`** — the Shark-style HI ionisation cut is always applied
  to the atomic remainder (former default 1).
- **`SigmaHIcrit`** — fixed at `SIGMA_HI_CRIT = 0.5` Msun/pc^2.
- **`CGMAGNOn`** — CGM-regime AGN heating fires whenever `AGNrecipeOn > 0`
  (former default 1).

Struct fields, parameter registrations, defaults, range-checks, and the
`millennium_all.par` tags were removed; the docs and unit tests were updated
to the single behaviour. `SubstepResolution` and `RamPressureStrippingOn` /
`RamPressureEpsilon` remain as live parameters.

## Ram-pressure stripping and self-regulating precipitation, now default-on (July 2026) — intentional output change

Two physics channels were added and then enabled by default after validation
against the microUchuu simulation. This **changes the default output**; the
mini-Millennium, `millennium_all`, microUchuu, and binary-benchmark baselines
were all regenerated in the same commit to bless the new physics as canonical.
Classic-SAGE output remains reproducible via the vanilla configs, which pin the
new toggles off.

- **Ram-pressure ISM stripping** (`RamPressureStrippingOn`, default 1). New
  module `src/model_ram_pressure.c` implements the Gunn & Gott (1972)
  criterion: satellite cold disk gas at radius *r* is stripped where
  `eps * rho_host(R_orb) * v_sat^2 > 2*pi*G * Sigma_disk(r) * Sigma_gas(r)`.
  With exponential gas and stellar disks sharing the scale radius the stripped
  mass fraction is analytic, `(1 + r_strip/r_s) * exp(-r_strip/r_s)`, applied
  once per snapshot with the same `1 - exp(-dT/t_strip)` cadence as the
  analytic hot-gas scheme. The ambient density uses the same profiles the
  cooling recipes assume; stripped gas and metals route to the central's
  hot/CGM reservoir. This channel removes the ISM (`ColdGas`) and is
  complementary to and independent of `PhysicalStrippingOn`, which strips the
  hot/CGM phase (starvation). Type 1 satellites and Type 2 orphans are covered
  (orphans via a frozen-orbit approximation, dormant in the default config).
  `RamPressureEpsilon` (default 1.0) is the order-unity geometry prefactor.
  Validation reproduces the satellite HI-deficiency-vs-host-mass trend and
  lifts the cluster-scale quenched-satellite fraction toward Wetzel et al.
  (2012).

- **Self-regulating precipitation** (`PrecipRegulationOn`, default 1). The
  CGM precipitation flow (`cooling_recipe_cgm`) now condenses only the gas
  above the `t_cool/t_ff = 10` Voit (2015) equilibrium — `M_eq = M_CGM *
  (t_cool/t_ff) / 10`, so `dM/dt = f_precip * (M_CGM - M_eq) / t_ff` — instead
  of draining the whole reservoir at the free-fall rate. The flow relaxes to
  the equilibrium and shuts off rather than emptying the CGM. Setting the
  toggle to 0 restores the legacy free-fall drain.

- **Bugfix: CGM overdraw in AGN heating.** `do_AGN_heating_cgm()` clamped the
  cooling flow against the pre-accretion `CGMgas`, then let the Bondi draw
  reduce the reservoir, so the returned `coolingGas` could exceed the
  remaining CGM by the accreted amount. Latent with the default uniform
  profile (the overlap stayed below the assertion tolerance); it aborted
  immediately under `CGMDensityProfile = 1`. The cooling flow is now re-capped
  after accretion.

Both toggles are exposed in the parameter file and documented in
`docs/parameters.md`, `docs/physics/infall.md` (stripping), and
`docs/physics/cooling_and_heating.md` (precipitation regulation). Unit
coverage: `tests/test_ram_pressure.c` and new precipitation-regulation cases
in `tests/test_cooling_heating.c`.

## HI bookkeeping fix (July 2026) — intentional output change

The `HIIonizationOn` correction and the H2 prescriptions previously claimed
their hydrogen shares independently: the ionised fraction was removed from
the *total* hydrogen budget while H2 was capped against that same total, so
`H1 = (1 - f_ion) X_H ColdGas - H2` went negative and was silently zeroed
(~615k clamp events per mini-Millennium run, dominated by fully-ionised
low-surface-density dwarfs; at z ~ 2-3 molecule-rich galaxies overdrew up to
~80% of their hydrogen). The ionisation cut now applies to the *atomic
remainder* only — `H1 = (1 - f_ion)(X_H ColdGas - H2)` — treating H2 as
central and shielded, which makes HI non-negative by construction.

`H1gas` is a pure diagnostic (no physics rate reads it), and the regression
audit confirms the blast radius: of 5,444 datasets, **only the 55 `H1gas`
datasets changed**; every other dataset is bit-identical. Total z=0 HI rises
0.4%. All baselines (mini-Millennium, millennium_all, microUchuu, binary
benchmark) re-captured in the same commit per the regression policy. The
negative-HI clamp counter now reads zero and remains as a guard.

## Physics-code readability pass (July 2026)

A restructuring pass over the physics modules. No physics changes: every
commit was gated on the mini-Millennium regression baseline (5444 HDF5
datasets bit-identical), with the microUchuu baseline (4254 datasets)
additionally verified for every structural change.

- Units: every struct field, parameter, and physical constant now carries a
  units comment; `docs/physics/units.md` defines the code-unit system, the
  h-factor conventions, and the unit-suffix naming rule. Inline conversions
  replaced by order-preserving macros (`CODE_MASS_TO_MSUN`,
  `CODE_LENGTH_TO_PC`, `MSUN_TO_CODE_MASS`); the H2 chemistry API parameters
  carry explicit unit suffixes.
- Structure: each of the eight SF prescriptions is its own function
  (`sfr_croton06` ... `sfr_gd14`) behind a switch dispatch; SN feedback mass
  computation extracted to `compute_sn_feedback()` (+ FFB variant);
  `model_misc.c` split into `model_h2_chemistry.c`,
  `model_halo_properties.c`, and `model_regimes.c` (near-duplicate branches
  were extracted, never merged — their differences are intentional).
- Frozen behaviour made explicit: deliberate single-precision sites are
  marked `float on purpose` and documented; promoting them to double changes
  the calibrated output.
- Guard rails: all 22 physics option flags are range-validated at startup
  and two silently-meaningless combinations (H2-based FFB modes with a
  non-H2 SF prescription; log-normal FFB modes with zero scatter) are
  rejected with clear messages. The silent physics clamps are now counted
  and reported at finalisation in VERBOSE builds. Baseline option-matrix
  coverage is documented in `docs/developer/REGRESSION_BASELINE.md`.

## Pre-release code review and cleanup (July 2026)

A full review of `src/` ahead of public release. No physics changes: every
commit in this pass was gated on the regression baseline (all 5444 HDF5
datasets bit-identical) plus the binary-output benchmark and the 559-test
unit suite.

- Removed dead code: the unfinished LHVT processing scaffolding
  (`core_tree_utils.c`, `PROCESS_LHVT_STYLE`), the never-enabled MCMC-mode
  paths, `#if 0` blocks, commented-out debug prints, and unused fields
  (`GALAXY.CentralMvir`, `GALAXY.TotalSatelliteBaryons`, `CGMrecipeSAGEOn`).
- Encapsulation and consistency: file-local functions made `static`, magic
  numbers replaced by the existing named constants, tabs/spaces normalized,
  stale comments corrected.
- Simplification: shared regime-routed deposit helpers
  (`add_metals_to_hot_reservoir` / `add_gas_to_hot_reservoir`), named SF
  prescription predicates (`sf_prescription_tracks_h2` /
  `sf_prescription_is_br06`), single FIRE-scaling computation in the FFB
  path, de-duplicated galaxy-index error guidance. `SFprescription` is now
  validated to [0, 7] at startup.
- Performance: radius-independent quantities hoisted out of the H2 radial
  integration hot loop (~4% faster mini-Millennium run, output identical).
- Infrastructure: fixed the broken root `make tests` target, added
  `make regression`, added a GitHub Actions CI workflow (build + unit tests
  + MPI compile check + mini-Millennium smoke run), untracked generated
  documentation, and documented the regression-baseline policy
  (`docs/developer/REGRESSION_BASELINE.md`).

## Physical satellite stripping and adaptive substeps (July 2026)

Two coupled changes to how satellites are stripped and how the per-snapshot
physics loop is integrated in time.

- **Physical satellite stripping** (`PhysicalStrippingOn`, `StrippingTimescaleFactor`).
  The stock SAGE scheme strips a fixed `1/N` of a satellite's baryon excess
  each of the `N` substeps, so the fraction removed over a snapshot is
  `1 - (1 - 1/N)^N` -- it depends on the substep count, not on the elapsed
  time or any physical timescale, converging to a discretization artifact
  (~`1 - 1/e`) as `N` grows. Three schemes are now selectable:
  - `0` -- legacy geometric (`excess / N` per substep), an exact reproduction
    of the stock behaviour, kept for reference.
  - `1` -- physical timescale, per-substep forward-Euler of
    `d(excess)/dt = -excess / t_strip`; telescopes to `1 - exp(-dT/t_strip)`
    over a snapshot (N-invariant in the limit, with an O(1/N) residual).
  - `2` -- **default**: the analytic `1 - exp(-dT/t_strip)` applied once per
    snapshot outside the substep loop, exactly the N→∞ limit of scheme 1 --
    no substep-count dependence and invariant to how the interval is split
    into snapshots. `t_strip = StrippingTimescaleFactor * t_dyn(host)`
    (default factor 1.0). This is the stripping timescale and cadence the
    ram-pressure ISM stripping channel reuses.

- **Adaptive substeps** (`SubstepResolution`). The snapshot interval is now
  integrated with a substep count that scales with `deltaT / t_dyn`, so
  high-redshift snapshots spanning several dynamical times are resolved with
  more substeps (bounded by a `STEPS` floor and a `MAX_STEPS` cap) instead of
  a fixed count. The compile-time `STEPS`-length SFR history arrays are
  unchanged; adaptive substeps map back into those bins. `SubstepResolution`
  (default 1.0) is a runtime multiplier on both the floor and the cap, so the
  substep count can be swept from the parameter file for convergence /
  N-invariance testing without recompiling.

Startup validation for the physics option flags (including
`PhysicalStrippingOn`, valid range [0, 2]) and their combinations was added
in the same period, rejecting out-of-range values before a run begins.

## SAGE26 (2026) — Major release

Built on [Croton et al. (2016)](https://arxiv.org/abs/1601.04709).

### New physics

- **Two-regime CGM model** (`CGMrecipeOn`): galaxies are classified as CGM-regime
  (below the Dekel & Birnboim 2006 shock mass) or hot-halo regime. Each regime
  uses a dedicated cooling recipe. CGM-regime cooling uses the Voit (2015)
  precipitation criterion. AGN feedback in the CGM regime uses the same
  `r_heat` ratchet as the hot-halo path, additionally capped at R_vir.
- **FIRE stellar feedback** (`FIREmodeOn`): FIRE-calibrated wind mass-loading and
  ejection efficiencies replace the fixed Croton+2016 values.
- **Feedback-free burst galaxies** (`FeedbackFreeModeOn`): implements the Li+2024
  and Boylan-Kolchin+2025 FFB criteria. Multiple sub-modes available (0–7).
- **NFW/beta-profile CGM density** (`CGMDensityProfile`): cooling in CGM-regime
  halos can use a uniform, NFW, or beta-profile gas distribution.
- **Extended bulge tracking**: merger-driven and instability-driven bulge
  components tracked separately (`MergerBulgeMass`, `InstabilityBulgeMass`,
  `MergerBulgeRadius`, `InstabilityBulgeRadius`). Radii follow Tonini+2016 eq. 15.
- **ICS assembly tracking** (`TrackICSAssembly`): records satellite disruption
  contributions to intracluster stars.
- **Satellite-disruption mass split** (`DynamicDisruptionSplit`,
  `DisruptionSplitAlpha`, `DisruptionSplitCref`): controls how disrupted
  satellite stellar mass is partitioned between the intracluster stars and
  the central. Modes: `0` fixed fraction (`FractionDisruptedToICS`); `1`
  mass-ratio split `f_ICS = 1 - (M_sub/M_host)^alpha`; `2` mass-ratio split
  with concentration weighting.

### New SF prescriptions (`SFprescription`)

| Value | Prescription |
|-------|-------------|
| 0 | Croton et al. (2006) original |
| 1 | Blitz & Rosolowsky (2006) H₂ |
| 2 | Somerville et al. (2025) SFR |
| 3 | Somerville et al. (2025) SFR + H₂ |
| 4 | Krumholz & Dekel (2012) |
| 5 | Krumholz, McKee & Tumlinson (2009) |
| 6 | Krumholz (2013) |
| 7 | Gnedin & Draine (2014) |

### New tree formats

ConsistentTrees ASCII and HDF5 (`consistent_trees_ascii`, `consistent_trees_hdf5`),
Genesis HDF5 (`genesis_lhalo_hdf5`), Gadget-4 HDF5 (`gadget4_hdf5`).

### Infrastructure

- HDF5 output format (`OutputFormat sage_hdf5`) with buffered writes.
- `libsage.so` shared library for Python bindings and PSO parameter calibration.
- Full star formation history arrays (`SaveFullSFH`).
- Regression baseline test (5444 datasets, bit-identical per dataset).

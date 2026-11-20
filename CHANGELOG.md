# Changelog

Notable changes to SAGE26. This file starts at the pre-release cleanup;
earlier history is in `git log`.

The cleanup pass is bound by a bit-identical output guarantee: every entry
below reproduces the committed regression baseline dataset-for-dataset unless
it says otherwise. See
[docs/developer/REGRESSION_BASELINE.md](docs/developer/REGRESSION_BASELINE.md).

## Unreleased

### Merger gas joins the target's regime reservoir (changes output)

- With `CGMrecipeOn 1`, `add_galaxies_together()` now adds the satellite's
  `HotGas` and `CGMgas` (with metals and dust) to the reservoir of the
  target's regime: `HotGas` for Regime 1, `CGMgas` for Regime 0, as stripping
  and infall already do. It used to add `CGMgas` to `CGMgas` whatever the
  regime. A hot-regime central holding that CGM ran the secondary CGM
  cooling recipe, whose AGN step sets `r_heat` against `rcool = Rvir`; a
  small reservoir is always out-heated, so `r_heat` jumped to `Rvir` and,
  as `r_heat` only moves outwards, the halo never cooled again.
- The effect grows with resolution, since smaller satellites bring smaller
  CGM packets. Reported on Shin-Uchuu, where nearly all centrals above
  10^12.5 Msun had stopped cooling by z = 3. In Millennium and microUchuu
  about 1% of hot-regime centrals above 10^12.5 Msun had `r_heat` pinned at
  `Rvir` (none after the fix); the z = 0 stellar mass function and
  stellar-halo mass relation barely change.
- As a safeguard, the CGM recipe no longer sets `r_heat` in a Regime 1 halo
  that holds hot gas; `do_AGN_heating()` already sets it against the hot
  halo's `rcool`. Output is bit-identical (Millennium; microUchuu, 25200
  datasets), since after the merger fix such haloes carry no CGM.

### Millennium particle mass (changes output)

- `PartMass` in `input/millennium.par` and `input/millennium_all.par` is now
  Millennium's 0.0860657 (10^10 Msun/h), not 0.086, which made particle-count
  masses 0.08% low. Output is not bit-identical to earlier runs; the
  regression baseline needs recapturing.
- Output attributes: `TimeOfInfall` is described as a snapshot number
  (unitless, previously labelled Myr), and `Concentration`'s description names
  the method for each `ConcentrationOn`. Values are unchanged.
- `tcool` in Regime 1 was written about 14,000 times too large: the hot-halo
  path converted code time to Gyr with an inverted factor and a stray 1e10/h.
  It now uses the CGM path's conversion, so Regime 1 holds `Rvir/Vvir` in
  Gyr/h. `tcool` and `tff` are labelled Gyr/h (both were in Gyr/h, labelled
  Gyr). Only the `tcool` dataset changes (Millennium file 0, 5632 datasets).

### RheatMode parameter removed

- The experimental `RheatMode` toggle (r_heat relaxation and the Croton et
  al. 2016 eqs. 18-20 order) is gone; its default, mode 0, is the model:
  `r_heat` only moves outwards and the heating is measured against the
  suppressed cooling. `RheatMode` is no longer read or written to the HDF5
  header; a parameter file that sets it is rejected as an unknown key.
  Output is bit-identical to the default.

### input/millennium_all.par lists every parameter

- Added `MergerTimeFactor`, `ConcentrationScatter` and
  `DustGrowthMaxDtoZ`, at their code defaults. `docs/parameters.md` now gives
  `MergerTimeFactor`'s default as 2.0, as in the code.

### RadioModeBHaccretionMassUncapped output (new dataset)

- Radio-mode accretion is capped so that the heating never exceeds the
  cooling, so once `r_heat >= r_cool` a black hole accretes nothing in radio
  mode; at z=0 on Millennium 87% of centrals above 10^12.5 Msun have
  `RadioModeBHaccretionMass` = 0 and none reach a radio-loud jet power. The
  new dataset records the accretion before that cap (the recipe's rate,
  Eddington- and reservoir-limited) as a jet-power diagnostic. It feeds no
  recipe: every existing dataset is bit-identical.

### ConcentrationScatter parameter

- The log-normal scatter in halo concentration for the BK25 threshold
  (`EnhancedStarFormationOn = 2`) was fixed at σ_ln c = 0.2 in the code; it is
  now the parameter `ConcentrationScatter` (default 0.2), so the width of the
  transition can be varied without a rebuild. At the default the output is
  bit-identical.

### H1gas within the cold gas's hydrogen (H1gas not bit-identical)

- After star formation, feedback and black-hole accretion take cold gas,
  `H2gas` and `H1gas` were each capped at 0.74 ColdGas, so their sum could
  exceed the hydrogen (3% of galaxies at z=0, 72% at z~5). `H2gas` is still
  capped first and is unchanged; `H1gas` now takes the hydrogen left
  (`clamp_phases_to_hydrogen()`). Only `H1gas` (and `DiscHI`) change; it feeds
  no other recipe, so every other output is bit-identical. Median `H1gas`
  falls by up to 15% at z~2 and more at z~5 in small galaxies; z=0 HI mass
  functions and fractions are unchanged.

### Radially resolved discs (DarkSAGEMode)

- New `DarkSAGEMode` (default 0): galaxies carry their gas and stellar discs in
  30 annuli of specific angular momentum, following DarkSage (Stevens et al.
  2016, 2024): rotation-curve radii, gas and stellar spins, precession, cooling
  and merger gas placed by j, per-annulus H2, star formation and instability.
  Mode 1 shares SAGE26's bulk recipes over the annuli and keeps SAGE26's
  `DiskScaleRadius`, so its output is identical to SAGE26; mode 2 measures
  `DiskScaleRadius` from the annuli; mode 3 also evaluates the star-formation
  law per annulus.
  New outputs: `DiscRadii`, `DiscVcirc`, `DiscGas`, `DiscGasMetals`,
  `DiscStars`, `DiscStarsMetals`, `DiscH2`, `DiscHI`, `DiscSFR`,
  `VelDispStars`, spins, bulge dispersions, disc scale and half-mass radii.
- New `ConcentrationOn = 4` (Dutton & Macciò 2014, DarkSage) and
  `H2DepletionMode = 5` (DarkSage, M_H2 / 2.30 Gyr).
- With `DarkSAGEMode = 0` the output is bit-identical to before.

### Dust (DustySAGEMode)

- New `DustySAGEMode` (default 0): cold, hot, CGM and ejected dust following
  Dusty-SAGE (Triani et al. 2020), with delayed AGB and SNII yields, grain
  growth, SN destruction and thermal sputtering. Mode 1 keeps SAGE26's metals
  unchanged (every non-dust output bit-identical to mode 0); mode 2 replaces
  SAGE26's metal production with the delayed yields and holds dust outside
  the metals. Several source bugs are fixed; see
  `docs/physics/darksage_and_dust.md`.
- New parameters `DustSNIIYields`, `DustSNIaYields`, `DustBinaryFraction`;
  new outputs `ColdDust`, `HotDust`, `CGMDust`, `EjectedDust`,
  `DustFormationRate`, `DustGrowthRate`, `DustDestructionRate`, and
  `DiscDust` when `DarkSAGEMode` is also on.
- With `DustySAGEMode = 0` the output is bit-identical to before.

### Stellar metallicity histories

- New `SaveFullMH` toggle (default 0, needs `SaveFullSFH = 1`) writes
  `MHMetalsDisk` and `MHMetalsBulge`, the metal mass in the stars of each
  `SFHMassDisk`/`SFHMassBulge` bin. Output only: the existing datasets are
  bit-identical.

### Quasar burst timing

- New output field `TimeOfLastQuasarBurst`, the lookback time of the last
  quasar-mode accretion from either a merger or a disk instability. It is
  meaningful to the snapshot interval, not to the substep. Output only: the
  existing datasets are bit-identical.

## v0.9 -- 2026-10-27

### Reproducible regime draws (not bit-identical)

- The CGM/hot classification now draws from a hash of the halo's MostBoundID
  and the snapshot, as the FFB classification already did, instead of the
  global `rand()` stream. Each classification has its own hash stream, so a
  halo's CGM/hot and FFB draws are independent. The same trees give the same
  output on every run, and changing the substep count or other physics no
  longer reshuffles which haloes land either side of the thresholds.
- FFB draws are unchanged. CGM/hot draws are a new Monte Carlo realisation,
  so outputs differ from the previous baseline.
- `rand()` is no longer called anywhere in the model; the discarded draws in
  `init_galaxy()` and `determine_and_store_ffb_regime()` are gone.


### Removed and renamed parameters (breaking)

The release build carries only the prescriptions the papers use. Switches that
selected alternatives are gone, along with the parameters that only those
alternatives read.

**An unrecognised tag is a startup error.** A parameter file written against
an earlier version aborts with `Tag 'X' not allowed or multiply defined` until
the removed lines are deleted. The same names are also gone from the
`Header/Runtime` attributes of the HDF5 output.

[docs/parameters.md](docs/parameters.md#removed-parameters) has the full table
of what each one became.

Renamed:

- `FeedbackFreeModeOn` -> `EnhancedStarFormationOn`, reduced from twelve modes
  to three: `0` off, `1` Li+24 mass threshold with their eq. 3 sigmoid, `2`
  Boylan-Kolchin 2025 acceleration threshold with log-normal concentration
  scatter. Old mode 4 is the new mode 2; old modes 2, 3, 5, 6, 7 and 8-11 are
  gone.
- `FFBMaxEfficiency` -> `EnhancedSFEfficiency` (still tunable).

Hardcoded to their published values, and the switch removed:

- `ColdStreamCeilingOn` -- Dekel & Birnboim (2006) eq. 39 is the only
  cold-stream prescription.
- `TrackICSAssembly` -- ICS assembly history is always accumulated.
- `FFBIgnoreRegime` -- the FFB criteria always apply, whatever the halo regime.
- `RegimeRandomMode`, `FFBRandomMode` -- both draws are always fresh each
  snapshot.
- `SNEnergyConservationOn` -- the supernova energy bound is always applied to
  both the reheating and the ejection term.

Moved to named constants in the source, cited at their definition:

- `StreamMassFactor`, `StreamThresholdWidthDex`, `GasDiskRadiusFactor`,
  `MaxSNEnergyCoupling`, `FFBConcSigma`, `FFBThresholdSlope`,
  `RedshiftPowerLawExponent`.

Removed outright with the Dekel+23 shell and disc FFB criteria:

- `FFBFeedbackDelayMyr`, `FFBCloudClumping`, `FFBCloudClumpingDisk`,
  `FFBStreamRadiusFraction`, `FFBShellSoundSpeedKms`, `FFBToomreQ`,
  `FFBSigmaCritMsunPc2`, `StreamZCritWidth`.

### Removed API

- `calculate_gmax_BK25()` (the old FFB mode 2/3 implementation) is gone from
  `model_regimes.h`.
- `determine_and_store_ffb_regime()` no longer takes `infallingGas` and `dt`;
  they were only read by the removed Dekel+23 criteria.

### Removed galaxy fields

- `FFBRandom` and `RegimeRandom` are gone from `struct GALAXY`. They were never
  written to output.

### Fixed

- `docs/parameters.md` and `README.md` documented `H2DiskAreaOption` as taking
  1/2/3; the code accepts 0/1/2, so the documented meaning of the default was
  wrong by one.
- Two merger-timescale unit tests asserted a positive merger time against a
  value that was identically zero, because their fixtures never set
  `MergerTimeFactor`.
- `MergerTimeFactor` was missing from the parameter reference.
- This file did not exist, so the `docs/changelog.md` Sphinx include was broken.

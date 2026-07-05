# Changelog

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

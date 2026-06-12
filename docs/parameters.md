# SAGE26 Parameter Reference

This document is the canonical reference for every parameter accepted by SAGE26
parameter files (`input/*.par`). Parameter files are parsed by
[`src/core_read_parameter_file.c`](../src/core_read_parameter_file.c).

**Syntax:** `ParameterName  value  % optional comment`

Lines beginning with `%` are comments. Required parameters must be present;
optional parameters take the listed default if omitted.

---

## Output

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `FileNameGalaxies` | string | yes | — | Base name for output files (e.g. `model` → `model_0.hdf5`). |
| `OutputDir` | string | yes | — | Directory for galaxy output. Created if absent. |
| `OutputFormat` | string | no | `sage_hdf5` | `sage_hdf5` or `sage_binary`. |
| `NumOutputs` | int | no | `-1` | Number of snapshot outputs; `-1` = all snapshots. |
| `SaveFullSFH` | 0/1 | no | `1` | Store per-snapshot SFR history arrays (`SFHMassDisk`, `SFHMassBulge`). |
| `TrackICSAssembly` | 0/1 | no | `1` | Record satellite disruption contributions to ICS (`ICS_disrupt`, `ICS_accrete`). |

---

## Simulation

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `TreeType` | string | yes | — | Merger tree format: `lhalo_binary`, `lhalo_hdf5`, `consistent_trees_ascii`, `consistent_trees_hdf5`, `genesis_lhalo_hdf5`, `gadget4_hdf5`. |
| `TreeName` | string | yes | — | Tree file basename (files are named `TreeName.N`). |
| `SimulationDir` | string | yes | — | Directory containing tree files. |
| `FileWithSnapList` | string | yes | — | File listing snapshot scale factors, one per line. |
| `FirstFile` | int | yes | — | First tree file index to process. |
| `LastFile` | int | yes | — | Last tree file index to process (inclusive). |
| `NumSimulationTreeFiles` | int | yes | — | Total number of tree files (may differ from FirstFile–LastFile range). |
| `LastSnapshotNr` | int | yes | — | Index of the final snapshot in the tree files. |
| `Omega` | double | yes | — | Matter density parameter Ω_m. |
| `OmegaLambda` | double | yes | — | Dark energy density parameter Ω_Λ. |
| `BaryonFrac` | double | yes | — | Universal baryon fraction f_b = Ω_b / Ω_m. |
| `Hubble_h` | double | yes | — | Dimensionless Hubble parameter h (H₀ = 100 h km/s/Mpc). |
| `PartMass` | double | yes | — | N-body particle mass in 10¹⁰ M_sun/h. |
| `BoxSize` | double | yes | — | Simulation box side length in Mpc/h. |

---

## Units

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `UnitLength_in_cm` | double | yes | — | 1 internal length unit in cm. Typically `3.08568e+24` (= Mpc/h). |
| `UnitMass_in_g` | double | yes | — | 1 internal mass unit in g. Typically `1.989e+43` (= 10¹⁰ M_sun). |
| `UnitVelocity_in_cm_per_s` | double | yes | — | 1 internal velocity unit in cm/s. Typically `100000` (= km/s). |

---

## Physics switches

| Parameter | Type | No | Default | Values and meaning |
|-----------|------|----|---------|-------------------|
| `SFprescription` | int | no | `1` | Star formation prescription: 0=Croton+06; 1=Blitz & Rosolowsky 06 H₂; 2=Somerville+25 SFR; 3=Somerville+25 SFR+H₂; 4=Krumholz & Dekel 12; 5=KMT 09; 6=Krumholz 13; 7=Gnedin & Draine 14. |
| `AGNrecipeOn` | int | no | `2` | AGN feedback: 0=off; 1=empirical; 2=Bondi-Hoyle; 3=cold cloud accretion. |
| `SupernovaRecipeOn` | 0/1 | no | `1` | SN feedback: 0=off; 1=Croton+16 reheating/ejection. |
| `ReionizationOn` | 0/1 | no | `1` | Reionization suppression of infall: 0=off; 1=Kravtsov+04 analytic fit. |
| `DiskInstabilityOn` | 0/1 | no | `1` | Disk instability: 0=off; 1=Toomre criterion drives bulge and BH growth. |
| `CGMrecipeOn` | 0/1 | no | `1` | Two-regime CGM model: 0=off (classical C16 cooling only); 1=on. |
| `FIREmodeOn` | 0/1 | no | `1` | FIRE stellar feedback: 0=off; 1=on. |
| `FeedbackFreeModeOn` | int | no | `1` | Feedback-free burst galaxies: 0=off; 1=Li+24 sigmoid; 2=BK25 (Ishiyama+21 c); 3=BK25 (ConcentrationOn method); 4=BK25 + log-normal c scatter; 5=Li+24 sharp; 6=Li+24 sigmoid + H₂ SF; 7=BK25 log-normal c scatter + H₂ SF. |
| `ConcentrationOn` | int | no | `3` | Halo concentration method: 0=off; 1=Ishiyama+21 table; 2=V_max/V_vir; 3=V_max/V_vir with infall freeze for satellites. |
| `BulgeSizeOn` | int | no | `3` | Bulge radius model: 0=off; 1=Shen+2003 eq.33; 2=Shen+2003 eq.32; 3=Tonini+2016 (separate merger and instability channels, mass-weighted average). |
| `StarburstColdGasOn` | 0/1 | no | `1` | Include cold gas contribution during merger starbursts. |
| `DynamicDisruptionSplit` | int | no | `2` | ICS-vs-BCG split for disrupted satellite stellar mass: 0=fixed fraction `FractionDisruptedToICS`; 1=mass-ratio split `f_ICS = 1 - (infallMvir / Mhost)^DisruptionSplitAlpha`; 2=mass-ratio split with concentration weighting (`alpha_eff = DisruptionSplitAlpha * DisruptionSplitCref / c_sat`). |

---

## CGM model parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `CGMDensityProfile` | int | no | `0` | CGM gas density profile for precipitation: 0=uniform; 1=NFW; 2=beta (β=2/3). |
| `CGMAGNOn` | 0/1 | no | `1` | Enable AGN heating coupling to the CGM-regime cooling path. |
| `CGMHeatingRheatOn` | int | no | `2` | CGM-regime AGN cooling suppression mechanism: 0=off (AGN fires but no radius/fraction suppression); 1=`f_heat_cgm` decaying suppression fraction on `t_dyn`; 2=standard `r_heat` ratchet capped at R_vir (mirrors hot-halo path). |

---

## FFB parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `FFBMaxEfficiency` | double | no | `0.2` | Maximum star formation efficiency during FFB bursts. `0.2` matches observations; `1.0` is the theoretical maximum. |
| `FFBConcSigma` | double | no | `0.2` | Log-normal scatter in halo concentration used by `FeedbackFreeModeOn=4,7` (dex). |
| `FFBIgnoreRegime` | 0/1 | no | `1` | Apply FFB criterion regardless of CGM regime classification. |
| `FFBRandomMode` | 0/1 | no | `0` | Use random number scatter in FFB threshold instead of deterministic sigmoid. |
| `RedshiftPowerLawExponent` | double | no | `1.25` | Redshift exponent in the FFB mass threshold scaling (Li+24 eq. 2). |

---

## H₂ star formation parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `H2DiskAreaOption` | int | no | `1` | Disk area for H₂ surface density: 1=π r_disk²; 2=π (3 r_disk)²; 3=2π r_disk². |
| `H2RadialIntegrationOn` | 0/1 | no | `1` | Use radial ring integration for H₂ fraction (more accurate, slower). |
| `H2RadialNBins` | int | no | `25` | Number of radial bins for the ring integration. |
| `H2RadialRMaxFactor` | double | no | `5.0` | Outer integration radius as a multiple of the disk scale radius. |

---

## Model parameters

### Star formation

| Parameter | Units | Default | Description |
|-----------|-------|---------|-------------|
| `SfrEfficiency` | dimensionless | `0.05` | Cold/H2 gas consumption efficiency per dynamical time. Used by SFprescription 0, 1, 4, 5, 7 unconditionally, and by 6 (K13) only in the single-slab path (`H2RadialIntegrationOn=0`). Unused by 2 and 3 (Somerville+25 use their own density-modulated `epsilon_cl`) and by 6 in the radial path (uses the K13 local depletion time natively). |
| `RecycleFraction` | dimensionless | `0.43` | Fraction of stellar mass instantaneously recycled to cold gas. |
| `Yield` | dimensionless | `0.025` | Fraction of stellar mass returned as metals. |
| `FracZleaveDisk` | dimensionless | `0.0` | Fraction of newly produced metals transferred directly to hot gas. |

### Supernova feedback

| Parameter | Units | Default | Description |
|-----------|-------|---------|-------------|
| `FeedbackReheatingEpsilon` | dimensionless | `2.9` | Mass of cold gas reheated per unit of stellar mass formed (Martin 1999). |
| `FeedbackEjectionEfficiency` | dimensionless | `0.3` | Fraction of SN energy deposited into hot gas for ejection. |
| `EnergySN` | erg | `1.0e51` | Energy per supernova event. |
| `EtaSN` | M_sun⁻¹ | `5.0e-3` | Number of supernovae per solar mass of stars formed. |

### AGN feedback

| Parameter | Units | Default | Description |
|-----------|-------|---------|-------------|
| `RadioModeEfficiency` | dimensionless | `0.08` | AGN radio-mode heating efficiency (AGNrecipeOn=2). |
| `QuasarModeEfficiency` | dimensionless | `0.005` | AGN quasar-mode wind heating efficiency (AGNrecipeOn > 0). |
| `BlackHoleGrowthRate` | dimensionless | `0.015` | Fraction of cold gas accreted onto the BH during mergers (AGNrecipeOn > 0). |

### Mergers

| Parameter | Units | Default | Description |
|-----------|-------|---------|-------------|
| `ThreshMajorMerger` | dimensionless | `0.3` | Mass ratio above which a merger is classified as major. |
| `ThresholdSatDisruption` | dimensionless | `1.0` | M_vir-to-baryonic mass ratio below which a satellite is disrupted rather than merged. |
| `FractionDisruptedToICS` | dimensionless | `0.8` | Fixed fraction of disrupted satellite stellar mass that goes to ICS (vs. central BCG). Used when `DynamicDisruptionSplit=0`, and as the fallback when modes 1/2 cannot compute a mass ratio. |
| `DisruptionSplitAlpha` | dimensionless | `0.25` | Power-law exponent for the mass-dependent disruption split. |
| `DisruptionSplitCref` | dimensionless | `10.0` | Reference concentration for the disruption split. |

### Gas cycling

| Parameter | Units | Default | Description |
|-----------|-------|---------|-------------|
| `ReIncorporationFactor` | dimensionless | `0.15` | Fraction of ejected mass reincorporated per dynamical time. |

### Reionization

| Parameter | — | Default | Description |
|-----------|---|---------|-------------|
| `Reionization_z0` | — | `8.0` | Characteristic redshift for reionization suppression (Kravtsov+04). |
| `Reionization_zr` | — | `7.0` | Width parameter for reionization suppression. |

See the **FFB parameters** section above for `FFBMaxEfficiency`,
`FFBConcSigma`, and `RedshiftPowerLawExponent`.

---

## MPI forest distribution

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `ForestDistributionScheme` | string | no | `generic_power_in_nhalos` | How forests are distributed over MPI tasks: `uniform_in_forests`, `linear_in_nhalos`, `quadratic_in_nhalos`, `exponent_in_nhalos`, `generic_power_in_nhalos`. |
| `ExponentForestDistributionScheme` | double | no | `0.7` | Exponent for `exponent_in_nhalos` or `generic_power_in_nhalos` schemes. |

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
| `CGMrecipeOn` | 0/1 | no | `1` | Two-regime CGM model: 0=off (classical C16 cooling only, including its rapid cold-accretion branch at `r_cool > R_vir`); 1=on. |
| `RegimeRandomMode` | 0/1 | no | `0` | Source of the random draw in the CGM/hot regime sigmoid: 0=a fresh uniform draw each snapshot (default), so a borderline-mass central can flip regime between snapshots; 1=the persistent `RegimeRandom` quantile assigned at galaxy creation, so the regime evolves deterministically with `Mvir` and never thrashes. |
| `ColdStreamCeilingOn` | 0/1 | no | `0` | How the Dekel & Birnboim cold-stream fraction `f_stream` is set in the hot regime. 0=the SAGE26 smooth fraction `(Mvir/Mshock)^(-4/3) (1+z)/2` with a hard cut below `z_crit` for `Mvir > Mshock` (default); 1=DB06 eq. 40 as published, the ceiling `Mstream = Mshock^2/(f Mstar)`, making `f_stream` exactly 1 or 0. See [Cooling and AGN heating](physics/cooling_and_heating.md). |
| `KarpovModeOn` | 0/1 | no | `0` | Metallicity of SN-reheated and ejected gas: 0=the full Karpov+23 recipe; 1=a low-metallicity floor at `Z/Z_sun = 0.01`. Only acts when the reheating/ejection path computes a metallicity for the outflow. |
| `FIREmodeOn` | 0/1 | no | `1` | FIRE stellar feedback: 0=off; 1=on. |
| `SNEnergyConservationOn` | 0/1 | no | `1` | Bound both supernova feedback terms by the energy actually available: 0=off (recovers the unbounded behaviour); 1=on (default). Caps the ejection coupling at `MaxSNEnergyCoupling` and the mass loading at `MaxSNEnergyCoupling * eta_SN E_SN / V_vir^2`, using the same `0.5*eta*V_vir^2` cost convention as `E_lift`, so the model cannot spend more energy than the supernovae release. Only acts when `FIREmodeOn=1`. |
| `MaxSNEnergyCoupling` | double | no | `2.0` | Cap applied to `eps_eff` when `SNEnergyConservationOn=1`. `2.0` means `E_FB <= m_* eta_SN E_SN` (all of the SN energy); `1.0` caps it at half. Bounds the *energy*, not the empirical FIRE mass loading, which is applied unmodified in `eta_reheat`. |
| `FeedbackFreeModeOn` | int | no | `1` | Feedback-free burst galaxies: 0=off; 1=Li+24 sigmoid; 2=BK25 (Ishiyama+21 c); 3=BK25 (ConcentrationOn method); 4=BK25 + log-normal c scatter; 5=Li+24 sharp; 6=Li+24 sigmoid + H₂ SF; 7=BK25 log-normal c scatter + H₂ SF; 8=Dekel+23 free-fall-time criterion on the **post-shock shell** density: FFB when `t_ff < FFBFeedbackDelayMyr` (eq. 3), where `t_ff = 0.84 Myr (n_sh/10^3.5 cm^-3)^-1/2` (eq. 4) and `n_sh` is the shocked accretion-stream density (eqs. 38-41), `n_sh = c * M^2 * Mdot_ac / (π R_str^2 V_vir μ m_p)` with `M = V_vir/FFBShellSoundSpeedKms`, `R_str = FFBStreamRadiusFraction * Rvir` and `Mdot_ac` the halo's own baryonic accretion rate (`infall_recipe()`'s `infallingGas` over the snapshot `dt`). Uses **no `ColdGas`** — this is an inflow-flux density, not a reservoir density, which is why it carries eq. 62's halo-mass dependence where a disc density does not (Dekel+23 sec. 8.2: the disc criterion "translates to a threshold in redshift with no explicit mass dependence"). Centrals only. |
| `ConcentrationOn` | int | no | `3` | Halo concentration method: 0=off; 1=Ishiyama+21 table; 2=V_max/V_vir; 3=V_max/V_vir with infall freeze for satellites. |
| `BulgeSizeOn` | int | no | `3` | Bulge radius model: 0=off; 1=Shen+2003 eq.33; 2=Shen+2003 eq.32; 3=Tonini+2016 (separate merger and instability channels, mass-weighted average). |
| `StarburstColdGasOn` | 0/1 | no | `1` | Include cold gas contribution during merger starbursts. |

---


## FFB parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `FFBMaxEfficiency` | double | no | `0.2` | Maximum star formation efficiency during FFB bursts. `0.2` matches observations; `1.0` is the theoretical maximum. |
| `FFBConcSigma` | double | no | `0.2` | Log-normal scatter in halo concentration used by `FeedbackFreeModeOn=4,7` (dex). |
| `FFBIgnoreRegime` | 0/1 | no | `1` | Apply FFB criterion regardless of CGM regime classification. |
| `FFBThresholdSlope` | double | no | `-6.2` | Slope of the Li+24 FFB halo-mass threshold, `log10 Mvir_ffb = 10.8 + slope * log10((1+z)/10)`. The `10^10.8 Msun` normalisation is pinned at `z = 9`, so changing the slope pivots the threshold about that redshift rather than shifting it wholesale. |
| `FFBFeedbackDelayMyr` | double | no | `1.0` | `t_fbk` in Dekel+23 eq. 3: the delay before stellar winds/SNe become effective in a low-metallicity starburst. Used only by `FeedbackFreeModeOn=8`. |
| `FFBCloudClumping` | double | no | `1.0` | Density contrast `c` (>= 1) between the star-forming clouds and the mean post-shock shell density (Dekel+23 eq. 41). Used only by `FeedbackFreeModeOn=8`, which multiplies `n_sh` by `c` before evaluating eq. (4). **Keep this at 1.0** unless you mean to depart from eq. 62: in the shell scenario the `M^2 ~ 275x` shock compression already supplies the contrast that the *disc* scenario needs `c` for, and eq. 41's own coefficient is already `~n_fbk`, so `c = 1` reproduces the eq. 62 threshold mass. Do not carry over the `c ~ 8` Dekel+23's Fig. 6 uses for their disc threshold — on the shell that double-counts the compression and drags the threshold mass down by ~1.1 dex. |
| `FFBStreamRadiusFraction` | double | no | `0.05` | `R_str/Rvir` in Dekel+23 eq. 61: the cold accretion stream's cross-sectional radius as a fraction of the virial radius, through which `Mdot_ac` is funnelled. Sets the pre-shock stream density via eq. 39. Used only by `FeedbackFreeModeOn=8`; `n_sh ~ R_str^-2`, so this is the single most leveraged shell parameter. |
| `FFBShellSoundSpeedKms` | double | no | `13.0` | Sound speed `c_s` [km/s] of the post-shock gas, setting the Mach compression `M = V_vir/c_s` in eq. 38 (and hence the `T_4^-1` term of eq. 41). Dekel+23 take the shocked gas to cool rapidly to `T ~ 10^4 K`; `13 km/s` reproduces their quoted `M ~ 15` at `V_vir ~ 200 km/s`. Used only by `FeedbackFreeModeOn=8`. Note this asserts Dekel+23's rapid-cooling assumption — SAGE26's own infall goes to the *hot* reservoir at `T_vir` (`add_infall_to_hot()`). |
| `FFBRandomMode` | 0/1 | no | `0` | Where the FFB draw comes from when it is compared against the Li+24 fraction `f_ffb(M_vir, z)`: 0=a fresh uniform draw each snapshot (default); 1=the persistent `FFBRandom` assigned at galaxy creation. Both compare against the same sigmoid — the difference is temporal. With 0 a galaxy re-enters the lottery every snapshot, so it moves in and out of FFB and a transient low-redshift FFB population persists; with 1 each galaxy holds a fixed quantile, so once `f_ffb` falls below it the galaxy leaves FFB permanently and both the oscillation and the low-z population disappear. |
| `RedshiftPowerLawExponent` | double | no | `1.25` | Exponent alpha of the `(1+z)^alpha` term in the FIRE mass-loading scaling `eta_reheat = FeedbackReheatingEpsilon * (1+z)^alpha * (V_vir/60 km/s)^beta` (Muratov+15). Used only when `FIREmodeOn=1`. |

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

### Disk sizes

| Parameter | Units | Default | Description |
|-----------|-------|---------|-------------|
| `GasDiskRadiusFactor` | dimensionless | `1.0` | `chi`: ratio of the atomic-gas scale length to the stellar/H2 scale length, applied **only** in the HI ionisation truncation. 1.0 = cospatial (published behaviour); observed disks have `chi ~ 1.5-2`. Independent of `DiskRadiusOn`. Raising it spreads the same HI over a larger area so more of it falls below `SIGMA_HI_CRIT`, without touching the H2 midplane pressure (H2 is central and shielded, which is why one radius should not set both). A modest lever: `chi = 1.7` moves the ionised fraction from 0.13 to 0.28 at the median surface density. |

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

### Gas cycling

| Parameter | Units | Default | Description |
|-----------|-------|---------|-------------|
| `ReIncorporationFactor` | dimensionless | `0.15` | Fraction of ejected mass reincorporated per dynamical time. |

### Cooling and cold streams

| Parameter | Units | Default | Description |
|-----------|-------|---------|-------------|
| `MShockMsun` | Msun | `6.0e11` | Dekel & Birnboim (2006) virial-shock stability mass. Used twice: it sets the CGM/hot regime classification in `determine_and_store_regime()`, and it sets the cold-stream criterion in `cooling_recipe_hot()`. Both must see the same value, so change it here rather than in either site. |
| `StreamMassFactor` | dimensionless | `3.0` | The factor `f` in DB06 eqs 40-41, relating the stream width to the clustering mass `M_*(z)`; they adopt 3. Used by both `ColdStreamCeilingOn` branches. Note that the hardcoded `Z_CRIT_DB06 = 1.2` used by the `ColdStreamCeilingOn = 0` branch was derived from `f = 3`, `MShockMsun = 6e11` and the Millennium cosmology, so it must be recomputed if either of these changes. |

### Reionization

| Parameter | — | Default | Description |
|-----------|---|---------|-------------|
| `Reionization_z0` | — | `8.0` | Characteristic redshift for reionization suppression (Kravtsov+04). |
| `Reionization_zr` | — | `7.0` | Width parameter for reionization suppression. |

See the **FFB parameters** section above for `FFBMaxEfficiency`,
`FFBConcSigma`, and `RedshiftPowerLawExponent`.

---

## Numerical time resolution

The per-snapshot physics loop (cooling, star formation, feedback) is integrated
with an *adaptive* number of sub-timesteps: the count scales with `deltaT / t_dyn`,
so a snapshot interval spanning several halo dynamical times is resolved with more
substeps (bounded by a `STEPS` floor and a `MAX_STEPS` cap). Tying the effective
resolution to `t_dyn` rather than the raw snapshot cadence is what lets one
calibration transfer across simulations with different output spacing.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `SubstepResolution` | double | `1.0` | Runtime multiplier on the adaptive-substep floor **and** cap. **Calibration-locked numerical knob, not a physics choice** — the model is calibrated at `1.0`; do not change it for science runs without recalibrating. Coarse steps over-cool (cooling outruns the AGN `r_heat` response before it can react), so raising the resolution lowers the massive-end SMF and total stellar mass. The shift from `1.0` to fully converged is only **~0.1 dex** at the massive end (within typical observational SMF scatter), but runtime grows **~linearly** with the substep count. Use higher values only for deliberate convergence / resolution studies. Note that the reported `SfrDisk`/`SfrBulge` are normalised by the substep count actually integrated (`SubstepsUsed`), not by `STEPS`; before September 2026 they were divided by `STEPS`, which scaled the reported SFR by `effective_steps/STEPS` and made this knob useless for convergence testing. **Run convergence sweeps with `FFBRandomMode=1` and `RegimeRandomMode=1`.** With the default per-snapshot draws, changing the substep count shifts the global `rand()` sequence, and the resulting scatter is non-monotonic and larger than the convergence signal: total stellar mass on mini-Millennium goes 1528 / 962 / 1255 / 1191 for N = 5 / 10 / 20 / 30. With persistent per-galaxy draws the same sweep is monotonic and converging -- 1529 / 1349 / 1262 / 1211, i.e. -12%, -6.4%, -4%. Note the two modes agree to within 2% at N = 5, 20 and 30 but differ by 30% at the default N = 10, which is worth understanding before quoting either number. |

**Convergence note.** The substep dependence is a long-standing property of SAGE's
cooling/feedback operator-splitting (it is present, and slightly *stronger*, with
the classic `CGMrecipeOn=0` cooling), not something introduced by the two-regime
CGM physics. Because the effect is only ~0.1 dex on the SMF, the calibrated `1.0` model
sits within observational constraints; a converged model would need at most a light
retune and would show more simulation-consistent behaviour, at higher compute cost.

---

## MPI forest distribution

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `ForestDistributionScheme` | string | no | `generic_power_in_nhalos` | How forests are distributed over MPI tasks: `uniform_in_forests`, `linear_in_nhalos`, `quadratic_in_nhalos`, `exponent_in_nhalos`, `generic_power_in_nhalos`. |
| `ExponentForestDistributionScheme` | double | no | `0.7` | Exponent for `exponent_in_nhalos` or `generic_power_in_nhalos` schemes. |

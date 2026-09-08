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
| `SNEnergyConservationOn` | 0/1 | no | `1` | Bound both supernova feedback terms by the energy actually available: 0=off (recovers the unbounded behaviour); 1=on (default). Caps the ejection coupling at `MaxSNEnergyCoupling` and the mass loading at `MaxSNEnergyCoupling * eta_SN E_SN / V_vir^2`, using the same `0.5*eta*V_vir^2` cost convention as `E_lift`, so the model cannot spend more energy than the supernovae release. Only acts when `FIREmodeOn=1`. |
| `MaxSNEnergyCoupling` | double | no | `2.0` | Cap applied to `eps_eff` when `SNEnergyConservationOn=1`. `2.0` means `E_FB <= m_* eta_SN E_SN` (all of the SN energy); `1.0` caps it at half. Bounds the *energy*, not the empirical FIRE mass loading, which is applied unmodified in `eta_reheat`. |
| `FeedbackFreeModeOn` | int | no | `1` | Feedback-free burst galaxies: 0=off; 1=Li+24 sigmoid; 2=BK25 (Ishiyama+21 c); 3=BK25 (ConcentrationOn method); 4=BK25 + log-normal c scatter; 5=Li+24 sharp; 6=Li+24 sigmoid + H₂ SF; 7=BK25 log-normal c scatter + H₂ SF. |
| `ConcentrationOn` | int | no | `3` | Halo concentration method: 0=off; 1=Ishiyama+21 table; 2=V_max/V_vir; 3=V_max/V_vir with infall freeze for satellites. |
| `BulgeSizeOn` | int | no | `3` | Bulge radius model: 0=off; 1=Shen+2003 eq.33; 2=Shen+2003 eq.32; 3=Tonini+2016 (separate merger and instability channels, mass-weighted average). |
| `StarburstColdGasOn` | 0/1 | no | `1` | Include cold gas contribution during merger starbursts. |
| `DynamicDisruptionSplit` | int | no | `2` | ICS-vs-BCG split for disrupted satellite stellar mass: 0=fixed fraction `FractionDisruptedToICS`; 1=mass-ratio split `f_ICS = 1 - (infallMvir / Mhost)^DisruptionSplitAlpha`; 2=mass-ratio split with concentration weighting (`alpha_eff = DisruptionSplitAlpha * DisruptionSplitCref / c_sat`). |
| `RamPressureStrippingOn` | 0/1 | no | `1` | Gunn & Gott (1972) ram-pressure stripping of satellite cold gas (ISM): 1=on (default); 0=off. Independent of the always-on hot-gas (starvation) stripping; see `docs/physics/infall.md`. |
| `RamPressureEpsilon` | double | no | `1.0` | Order-unity prefactor on the ram pressure `P_ram = eps * rho_host * v_sat^2`, absorbing the disk-orientation geometry uncertainty. Used only when `RamPressureStrippingOn=1`. |
| `DiskRadiusOn` | int | no | `0` | Disk scale radius model. 0=published Mo, Mao & White (1998) eq. 12 from the instantaneous halo spin, unbounded. 1=adds a working virial fallback (the published else-branch is only reachable when `Rvir == 0`, so it returns `r_d = 0` and leaves the galaxy permanently inert; here the virial scale is rebuilt from `Len * PartMass`) and bounds `r_d / Rvir` to `[0.002, DiskRadiusMaxFrac]`. 2=as 1, but `\|j\|` comes from a running mean of the spin **vector** over a halo dynamical time, which cuts the snapshot-to-snapshot jitter in `r_d` by 3x and shrinks `r_d` by a near-uniform ~9%. It does **not** remove the low-particle-count bias in `\|j\|` -- that error is correlated between adjacent snapshots, so time-averaging cannot reach it. See [physics/disk_sizes.md](physics/disk_sizes.md). |

---

## CGM model parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `CGMDensityProfile` | int | no | `0` | CGM gas density profile for precipitation: 0=uniform; 1=NFW; 2=beta (β=2/3). |
| `PrecipCriterionOn` | 0-5 | no | `1` | Which of the two suppression factors in the Voit precipitation rate `ṁ = S((10 - r)/2) × (M_CGM - M_eq)/t_ff` are applied, where `r = t_cool/t_ff` and `M_eq = M_CGM r/10`. 1=both (the submitted rate); 2=`M_eq` only, dropping the `f_inflow` sigmoid; 3=sigmoid only, dropping the condensation term (the bare-sigmoid form printed in the first submission); 4=neither, keeping the rest of the precipitation path; 0=neither *and* skipping the hand-over to standard cooling, so `ṁ = M_CGM/t_ff` for every CGM halo however stable. Modes 1-4 form a 2×2 factorial in the two factors, differing by nothing else, so **mode 4 is the reference the single-factor rows should be measured against**; mode 0 changes the hand-over as well and is the weaker control. Measured on mini-Millennium, the total stellar mass relative to mode 4 is 0.985-0.991 for mode 1, 0.988-0.991 for mode 2, and 0.997-1.000 for mode 3: `M_eq` supplies essentially all of the suppression and the sigmoid 0.03-0.3%, because CGM-regime haloes sit at `r ~ 0.1-0.4` and the sigmoid never leaves its ceiling `S(5) = 0.9933`. SMF offsets from mode 4 are 0.004-0.014 dex rms for the sigmoid alone, below Poisson error in every bin. 5=SAGE16 cold accretion, `ṁ = M_CGM/(R_vir/V_vir)`, bypassing the criterion like mode 0 but draining on the dynamical rather than the free-fall time; since `t_ff = √2 R_vir/V_vir` exactly for the uniform profile, mode 5 is a uniform `√2 = 1.41×` faster than mode 0. Measured on mini-Millennium it raises the total stellar mass by 3.5% at z=0 rising to 24% at z=4 (SMF 0.031-0.096 dex rms), so the inflow-rate normalisation matters far more than the criterion's shape. Also gates the same two factors in the `PreventiveHeatingOn=5` hot-halo ceiling. |
| `RegimeRandomMode` | 0/1 | no | `0` | Where the CGM/hot regime draw comes from: 0=a fresh uniform draw each snapshot (default); 1=the persistent `RegimeRandom` assigned at galaxy creation. As for `FFBRandomMode`, the difference is temporal rather than statistical: with 0 borderline-mass galaxies switch regime repeatedly, with 1 the regime evolves monotonically with `M_vir`. |
| `ColdStreamCeilingOn` | 0/1 | no | `0` | How cold streams shut off below the critical redshift in `M_vir > Mshock` haloes: 0=hard cut at `z = 1.5` (published behaviour); 1=Dekel & Birnboim (2006) eqs 39-41, where the criterion is the stream cooling-to-compression ratio `R = (f Mstar/M_vir)^(2/3) (M_vir/Mshock)^(4/3)` and `f_stream` is a sigmoid in `log10 R` of width 0.3 dex. With 1 the redshift dependence enters through the clustering mass `Mstar(z)` rather than the explicit `(1+z)/2` factor, and `z_crit` emerges from `f Mstar(z_crit) = Mshock` (1.20 for Millennium, 1.01 for miniUchuu at `f = 3`) instead of being imposed, so `f_stream` is continuous in redshift. |
| `StreamMassFactor` | double | no | `3.0` | The order-unity factor `f` in Dekel & Birnboim (2006) eqs 40-41, setting the stream width relative to the clustering scale. Used only when `ColdStreamCeilingOn=1`; they adopt `f = 3`. |

---

## FFB parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `FFBMaxEfficiency` | double | no | `0.2` | Maximum star formation efficiency during FFB bursts. `0.2` matches observations; `1.0` is the theoretical maximum. |
| `FFBConcSigma` | double | no | `0.2` | Log-normal scatter in halo concentration used by `FeedbackFreeModeOn=4,7` (dex). |
| `FFBIgnoreRegime` | 0/1 | no | `1` | Apply FFB criterion regardless of CGM regime classification. |
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
| `DiskRadiusFactor` | dimensionless | `1.0` | Angular-momentum retention factor `f_j` multiplying the Mo+98 disk scale radius. Since `R_vir` cancels out of MMW98 eq. 12, the radius is really `r_d = f_j * |j| / (2 Vvir)`. 1.0 = full retention (published). Matching the Somerville et al. (2018) GAMA stellar-size/halo-size ratio (`R*/Rvir ~ 0.017`, flat in mass) needs `f_j ~ 0.55`, which is also the expected `j_disk/j_halo`. Strongly degenerate with `SfrEfficiency`: SFR responds as `r_d^-3.7`, so on Millennium `f_j = 0.55` lowers the total HI mass by 2.9x while raising the total stellar mass by 0.14 dex. |
| `DiskRadiusMaxFrac` | dimensionless | `0.15` | Ceiling on `r_d / Rvir`, applied after `f_j`, when `DiskRadiusOn > 0`. Bounded against the peak-retained `Rvir` the physics uses, not the instantaneous `Rvir` written to the output column, so a halo below its peak mass can show a larger ratio on output. The default corresponds to `lambda ~ 0.21`; unbounded, 11.5% of Millennium galaxies at z=0 exceed `0.1 Rvir` and the 99.9th-percentile `r_d` is 32 kpc. Set very large to disable. |
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
the classic `CGMrecipeOn=0` cooling), not something introduced by the CGM/precipitation
physics. Because the effect is only ~0.1 dex on the SMF, the calibrated `1.0` model
sits within observational constraints; a converged model would need at most a light
retune and would show more simulation-consistent behaviour, at higher compute cost.

---

## MPI forest distribution

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `ForestDistributionScheme` | string | no | `generic_power_in_nhalos` | How forests are distributed over MPI tasks: `uniform_in_forests`, `linear_in_nhalos`, `quadratic_in_nhalos`, `exponent_in_nhalos`, `generic_power_in_nhalos`. |
| `ExponentForestDistributionScheme` | double | no | `0.7` | Exponent for `exponent_in_nhalos` or `generic_power_in_nhalos` schemes. |

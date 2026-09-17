# Cooling and AGN Heating

This page covers how gas moves from the hot/CGM reservoir into the cold disk,
how AGN radio-mode feedback suppresses that flow, and how SAGE26's two-regime
CGM model selects between classical hot-halo cooling and Carr et al. (2023)
bulk CGM cooling.

Source: [`src/model_cooling_heating.c`](https://github.com/MBradley1985/SAGE26/blob/main/src/model_cooling_heating.c)

Regime classification: [`src/model_regimes.c`](https://github.com/MBradley1985/SAGE26/blob/main/src/model_regimes.c)

Called from: [Per-halo physics loop](../core_build_model.md) -- step 8 of the
substep ordering.

## The two cooling regimes

A central galaxy is classified once per snapshot by
`determine_and_store_regime()` (`model_regimes.c`) against the Dekel &
Birnboim (2006) shock mass `M_shock`, set by the `MShockMsun` parameter
(default `6 x 10^11 Msun`).

The classification is **stochastic, and depends on mass only** -- there is no
redshift term. A sigmoid in `log10(Mvir / M_shock)` of width 0.1 dex gives the
probability of being in the hot regime, and a uniform draw decides the
outcome:

```
hot_fraction = 1 / (1 + exp(-log10(Mvir / M_shock) / 0.1))
Regime       = (random_uniform < hot_fraction) ? 1 : 0
```

`RegimeRandomMode` controls the draw: `0` redraws every snapshot (original
behaviour, so borderline-mass centrals can flip regime between snapshots);
`1` reuses the persistent per-galaxy `RegimeRandom` quantile assigned at
galaxy creation, so the regime evolves deterministically with `Mvir` and
never thrashes.

- **Regime 0 (CGM):** below `M_shock`. Cooling proceeds via the Carr et al.
  (2023) bulk formula on the `CGMgas` reservoir.
- **Regime 1 (hot halo):** above `M_shock`. Cooling proceeds via the
  classical Croton+06 isothermal recipe on the `HotGas` reservoir, with a
  Dekel & Birnboim cold-stream fraction splitting the flow.

Note that the cold-stream criterion (`Z_CRIT_DB06`, `StreamMassFactor`) plays
no part in this classification -- it acts *within* the hot regime, in
`cooling_recipe_hot()`. Regime 0 galaxies have no cold-stream term at all.

`determine_and_store_regime()` is called unconditionally from
`core_build_model.c`, but `Regime` is only consumed when `CGMrecipeOn = 1`.
With `CGMrecipeOn = 0` every galaxy uses `cooling_recipe_hot()` and the
`CGMgas` reservoir is unused.

## Dispatcher -- `cooling_recipe_regime_aware()`

The top-level entry point when `CGMrecipeOn = 1`:

| Galaxy regime | Primary path | Secondary path |
|---------------|--------------|----------------|
| Regime 0 (CGM) | `cooling_recipe_cgm()` on `CGMgas` | -- |
| Regime 1 (hot) | `cooling_recipe_hot()` on `HotGas` | `cooling_recipe_cgm()` on any residual `CGMgas` |

The secondary CGM path for Regime 1 ensures that any leftover CGMgas
(typical of a halo that crossed `M_shock` mid-life) drains naturally rather
than being frozen.

The dispatcher then transfers both contributions into `ColdGas` itself,
re-clamping each against its own reservoir first (AGN accretion inside the
recipes can drain a reservoir between the internal cap and the transfer) and
tracking metallicity from each donor separately.

With `CGMrecipeOn = 0`, `core_build_model.c` instead calls `cooling_recipe()`
-- which forwards to `cooling_recipe_hot()` -- and performs the transfer via
`cool_gas_onto_galaxy()`.

## `cooling_recipe_hot()` -- Regime 1 (classical hot halo)

1. Compute the halo dynamical time `t_cool = R_vir / V_vir` and
   `t_ff = sqrt(2) * t_cool`, storing both as diagnostics.
2. Compute the virial temperature
   `T_vir = VIRIAL_TEMP_COEFF * Vvir^2 = 35.9 * Vvir^2` (K, with Vvir in km/s).
3. Look up the metal-dependent cooling rate `Lambda(T, Z)` from the
   Sutherland & Dopita (1993) tables via
   `get_metaldependent_cooling_rate()`.
4. Compute the cooling radius `r_cool` for a singular isothermal hot-gas
   profile (`rho_0 = HotGas / (4 pi R_vir)`) as the radius at which
   `t_cool = R_vir / V_vir`. `RcoolToRvir` is stored for diagnostics.

   `r_cool` is **not capped** at `R_vir` on either path: `r_cool > R_vir` is a
   meaningful state (a corona cooling faster than it can be shock-heated) and
   is allowed to set the cooling rate and the AGN heating radius.
5. Compute the cooling rate. This is where the two paths diverge:

   **`CGMrecipeOn = 0`** -- Croton et al. (2016) verbatim, both branches
   including the factor-2 discontinuity at `R_vir`:

   | Condition | Rate |
   |-----------|------|
   | `r_cool > R_vir` | `HotGas / t_cool` -- rapid cold accretion |
   | `r_cool <= R_vir` | `(HotGas / R_vir) * (r_cool / (2 t_cool))` -- quasi-static flow |

   **`CGMrecipeOn = 1`** -- the flow is split by the cold-stream fraction
   `f_stream` (below), with **no `r_cool` vs `R_vir` branch**:

   ```
   mdot_stream = f_stream * HotGas / t_cool
   mdot_cool   = (1 - f_stream) * (HotGas / R_vir) * (r_cool / (2 t_cool))
   ```

   Streams accrete on the dynamical time; the non-penetrating remainder cools
   as a quasi-static flow from within `r_cool`. Both are stored as
   diagnostics and summed into the returned cooling mass.

   Because the two channels run at different rates (`1/t_cool` versus
   `r_cool / (2 R_vir t_cool)`), `f_stream` sets the *total* cooling rate and
   not merely its labelling -- the dynamic range between `f_stream = 0` and
   `1` is a factor `2 R_vir / r_cool`.
6. Clamp the total to the available `HotGas`.
7. Apply `do_AGN_heating()` if `AGNrecipeOn > 0`.
8. Accumulate `Cooling` energy and `CoolingRate`, and return the cooled mass.

### The cold-stream fraction `f_stream`

Set by `ColdStreamCeilingOn`. Both forms are built from the Dekel & Birnboim
(2006) ingredients: the shock mass `M_shock` (`MShockMsun`), the clustering
mass `M_*(z)` from `interpolate_clustering_mass()`, and the factor
`f = StreamMassFactor` (default 3, as they adopt).

**`ColdStreamCeilingOn = 0` (default)** -- a smooth fraction:

```
f_stream = (Mvir / M_shock)^(-4/3) * (1 + z) / 2,   clamped to [0, 1]
f_stream = 0                                        for z < Z_CRIT_DB06 and Mvir > M_shock
```

This is a **SAGE26 prescription motivated by** DB06 rather than one of their
results: the `(M/M_shock)^(-4/3)` suppression is the reciprocal of the halo
ratio in their eq. 38, and the explicit `(1 + z)` factor has no counterpart
in their work (their redshift dependence enters through `M_*(z)`). DB06's
equations predict *whether* streams penetrate, not what fraction of the
corona they carry.

`Z_CRIT_DB06 = 1.2` is not a free parameter: DB06 eq. 41 defines it by
`f M_*(z_crit) = M_shock`, which for `f = 3`, `MShockMsun = 6e11` and the
Millennium/WMAP1 `M_*(z)` table solves to 1.20. It is tied to that cosmology
and shock mass -- the Uchuu/Planck15 table gives 1.01 -- and is hardcoded, so
it must be recomputed if either changes.

**`ColdStreamCeilingOn = 1`** -- DB06 eq. 40 as published, a threshold rather
than a fraction, so `f_stream` is exactly 1 or 0:

```
M_stream = M_shock^2 / (f M_*)        for f M_* < M_shock   (eq. 40)
M_max    = M_stream, or M_shock if f M_* > M_shock          (their low-z limit)
f_stream = (Mvir < M_max) ? 1 : 0
```

This is the inversion of their eq. 39 test `R < 1` and reproduces it exactly.
Both limits of eq. 40 follow from the one test, and the crossover is derived
from `f M_*(z) = M_shock`, so this path needs no hardcoded `z_crit` and
self-adjusts to cosmology and `MShockMsun`.

Because `f_stream` is binary here, the hot-halo and stream channels are
mutually exclusive -- one of `mdot_cool` / `mdot_stream` is always exactly
zero -- and a halo crossing the ceiling jumps by the full `2 R_vir / r_cool`
factor in a single snapshot. DB06's threshold is genuinely discontinuous, so
this is faithful to the paper; the smooth default trades that fidelity for
continuity.

## `cooling_recipe_cgm()` -- Regime 0 (Carr et al. 2023 bulk cooling)

The CGM recipe does not construct a cooling radius or evaluate a local
instability criterion. It evaluates one bulk cooling time for the whole
reservoir and drains it on `t_cool + t_ff`.

### Step 1 -- density profile

A single power law, Carr et al. (2023) with `alpha = 1.4` and
`r_0 = 0.1 R_vir`, so `R_vir / r_0 = 10` by construction. Two volume
integrals follow, for mass and for the density-squared cooling weight:

```
I_M    = (10^(3 - alpha)     - 1) / (3 - alpha)
I_cool = (10^(3 - 2 alpha)   - 1) / (3 - 2 alpha)
rho_0    = CGMgas / (4 pi r_0^3 I_M)
rho_eff  = rho_0 * I_cool / I_M
```

There is no profile switch: `alpha` is fixed in the source. (The former
`CGMDensityProfile` parameter, and the uniform / NFW / beta profile helpers
it selected, have been removed.)

### Step 2 -- timescales

```
t_cool = (3/2) * mu * x / rho_eff        with x = m_p k_B T / Lambda(T, Z)
t_ff   = sqrt(2 R_vir / g),  g = G Mvir / R_vir^2
```

`t_ff` is evaluated at the virial radius, as in the paper. Both are stored
as diagnostics in Gyr.

### Step 3 -- bulk cooling rate

```
coolingGas = CGMgas / (t_cool + t_ff) * dt
```

clamped to the available `CGMgas`. There is no `t_cool / t_ff` threshold, no
precipitation fraction, and no equilibrium-reservoir term -- the sum of the
two timescales sets the rate directly, so the flow is slow when either
cooling or free-fall is slow.

### Step 4 -- AGN heating

`r_cool` is pinned to `R_vir` -- the flow is a bulk quantity with no cooling
radius of its own -- and used only to pass a heating radius to
`do_AGN_heating_cgm()`, which fires whenever `AGNrecipeOn > 0`.

### Step 5 -- diagnostics

`cooling_recipe_cgm()` populates `tcool`, `tff` and `CoolingRate`. It does
*not* set `RcoolToRvir` (that is a hot-path quantity). Because the function
is only entered when `CGMgas > 0`, `reset_cgm_diagnostics()` clears `tcool`,
`tff` and `RcoolToRvir` for a halo that has drained its reservoir, so the
outputs do not carry stale values from the last snapshot that had gas.

Sentinel conventions in the output: `tff = -1` means "not a CGM-path value"
(the dispatcher sets it for Regime 1, where `tcool` is instead the halo
dynamical time `R_vir / V_vir`), and `RcoolToRvir = -1` means the cooling
radius was never evaluated.

## AGN radio-mode heating

Two functions implement the AGN suppression for the two regimes. Both
share the accretion calculation via the file-private
`agn_accretion_compute()` helper.

### `agn_accretion_compute()` -- the shared accretion calculation

Computes the BH accretion rate from one of three recipes
(`AGNrecipeOn = 1, 2, 3`), applies the Eddington limit, and derives the
corresponding heating mass. It modifies no galaxy fields; the caller draws
from the appropriate reservoir and updates `r_heat`.

| `AGNrecipeOn` | Recipe | Source |
|---------------|--------|--------|
| 1 | Empirical Croton+06 eq. 10: scales with `BH_mass / 10^8 Msun`, `(V_vir/200)^3`, `HotGas / M_vir` | Croton+06 |
| 2 | Bondi-Hoyle accretion: `dM/dt = 2.5 pi G * (3/8) * 0.6 * x * BH_mass * eta` | Bondi (1952) |
| 3 | Cold-cloud accretion: triggers when `BH_mass > 1e-4 * M_vir * (r_cool / R_vir)^3` | --- |

Accretion is then capped at the Eddington rate. The resulting heating mass
is converted from accreted mass via `AGNheating = (1.34e5 / V_vir)^2 *
AGNaccreted` (with the coefficient derived from `sqrt(2 * eta * c^2)`).

Note that recipe 3's threshold scales as `(r_cool / R_vir)^3`, so with
`r_cool` uncapped it rises steeply for `r_cool > R_vir` haloes and the
trigger fires *less* often for them, not more.

### `do_AGN_heating()` -- Regime 1 (hot halo)

The classical Croton+06 ratchet:

1. Suppress cooling by `(1 - r_heat / r_cool)`. If `r_heat >= r_cool`,
   cooling is fully zeroed. When `CGMrecipeOn = 1` the suppression is also
   short-circuited to zero once `r_heat >= 0.99 * r_cool`, avoiding a
   vanishing residual flow that never quite quenches.
2. Compute accretion via `agn_accretion_compute()`, draw it from `HotGas`,
   credit it to `BlackHoleMass`.
3. Update `r_heat` monotonically: `r_heat_new = (AGNheating / coolingGas)
   * r_cool`; if larger than the stored `r_heat`, replace it.

The ratchet is never reduced, so `r_heat` can only grow over time. In the
hot-halo path there is no `R_vir` cap -- once `r_heat >= r_cool`, cooling
stays fully suppressed for that substep.

### `do_AGN_heating_cgm()` -- Regime 0 (CGM)

Differences from the hot-halo path:

- Accretion draws from `CGMgas`, not `HotGas`.
- After the ratchet updates `r_heat`, the value is capped at `R_vir` so
  the heating radius cannot grow past the halo boundary.
- The returned `coolingGas` is re-capped against the post-accretion
  `CGMgas`: the Bondi draw reduces the reservoir, and without the re-cap a
  cooling flow that was already reservoir-limited could be handed back to
  the caller exceeding the remaining CGM.

## `cool_gas_onto_galaxy()`

Transfers cooled gas from `HotGas` to `ColdGas` with metallicity tracking.
Called from `core_build_model.c` on the `CGMrecipeOn = 0` path only; on the
regime-aware path `cooling_recipe_regime_aware()` does the transfer itself,
because it has to move mass out of two different reservoirs.

## Switches and parameters

| Parameter | Effect |
|-----------|--------|
| `CGMrecipeOn` | 0 disables the two-regime split entirely (pure Croton+06 hot-halo cooling); 1 enables it. |
| `MShockMsun` | Dekel & Birnboim shock mass in Msun (default 6e11). Sets both the regime classification and the cold-stream criterion. |
| `RegimeRandomMode` | 0 redraws the regime every snapshot; 1 uses the persistent per-galaxy quantile (deterministic in `Mvir`). |
| `ColdStreamCeilingOn` | 0 the SAGE26 smooth `f_stream` (default); 1 the DB06 eq. 40 threshold. |
| `StreamMassFactor` | `f` in DB06 eqs 40-41 (default 3). Used by both `f_stream` forms. |
| `AGNrecipeOn` | Radio-mode BH accretion recipe: 0 off, 1 empirical, 2 Bondi-Hoyle, 3 cold-cloud. |
| `RadioModeEfficiency` | Overall scaling on radio-mode accretion. |
| `QuasarModeEfficiency` | Used by the merger-driven AGN path -- see [Mergers and disruption](mergers_and_disruptions.md). |

See [`parameters.md`](../parameters.md) for full descriptions and defaults.

## References

- White & Frenk (1991), ApJ 379, 52 -- classical halo cooling framework.
- Sutherland & Dopita (1993), ApJS 88, 253 -- metal-dependent cooling tables.
- Croton et al. (2006), MNRAS 365, 11 -- original SAGE cooling and AGN
  radio-mode prescriptions.
- Croton et al. (2016), ApJS 222, 22 -- SAGE public release; the
  `CGMrecipeOn = 0` cooling branches.
- Dekel & Birnboim (2006), MNRAS 368, 2 -- shock-mass criterion (eq. 38),
  stream penetration ratio (eq. 39), stream ceiling (eq. 40) and critical
  redshift (eq. 41).
- Dekel et al. (2009), Nature 457, 451 -- cold streams in massive high-z
  haloes.
- Carr et al. (2023), ApJ 949, 21 -- power-law CGM profile and the bulk
  `CGMgas / (t_cool + t_ff)` cooling rate.
- Duffy et al. (2008), MNRAS 390, L64 -- NFW concentration scaling.
- Bondi (1952), MNRAS 112, 195 -- accretion onto compact objects.
- Rybicki & Lightman (1979) -- Eddington luminosity.

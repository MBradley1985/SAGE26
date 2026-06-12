# Cooling and AGN Heating

This page covers how gas moves from the hot/CGM reservoir into the cold disk,
how AGN radio-mode feedback suppresses that flow, and how SAGE26's two-regime
CGM model selects between classical hot-halo cooling and precipitation-driven
CGM cooling.

Source: [`src/model_cooling_heating.c`](https://github.com/MBradley1985/SAGE26/blob/main/src/model_cooling_heating.c)

Called from: [Per-halo physics loop](../core_build_model.md) -- step 8 of the
substep ordering.

## The two cooling regimes

A central galaxy is classified once per snapshot by
`determine_and_store_regime()` based on the Dekel & Birnboim (2006) shock
mass `M_shock ~ 6 x 10^11 Msun` (modulated by redshift via `Z_CRIT_DB06 = 1.5`):

- **Regime 0 (CGM):** halo is below `M_shock` (or above it but at z < z_crit
  where cold streams penetrate). Cooling proceeds via the Voit (2015)
  precipitation criterion on the `CGMgas` reservoir.
- **Regime 1 (hot halo):** halo is above `M_shock` and has a stable virial
  shock. Cooling proceeds via the classical Croton+06 isothermal recipe on
  the `HotGas` reservoir, optionally with a Dekel & Birnboim cold-stream
  fraction blended in.

When `CGMrecipeOn = 0`, every galaxy uses Regime 1 unconditionally and the
`CGMgas` reservoir is unused.

## Dispatcher -- `cooling_recipe_regime_aware()`

The top-level entry point. Both regimes can produce some cooling from both
reservoirs:

| Galaxy regime | Primary path | Secondary path |
|---------------|--------------|----------------|
| Regime 0 (CGM) | `cooling_recipe_cgm()` on `CGMgas` | -- |
| Regime 1 (hot) | `cooling_recipe_hot()` on `HotGas` | `cooling_recipe_cgm()` on any residual `CGMgas` |

The secondary CGM path for Regime 1 ensures that any leftover CGMgas
(typical of a halo that crossed `M_shock` mid-life) drains naturally rather
than being frozen.

After computing the cooled masses, the dispatcher transfers them into
`ColdGas` in-place, tracking metallicity from each donor reservoir
separately.

## `cooling_recipe_hot()` -- Regime 1 (classical hot halo)

1. Compute the virial temperature
   `T_vir = VIRIAL_TEMP_COEFF * Vvir^2 = 35.9 * Vvir^2` (K, with Vvir in km/s).
2. Look up the metal-dependent cooling rate `Lambda(T, Z)` from the
   Sutherland & Dopita (1993) tables via
   `get_metaldependent_cooling_rate()`.
3. Compute the cooling radius `r_cool` from the isothermal beta-model:
   the density at which `t_cool = t_dyn = R_vir / V_vir`.
4. Pick the cooling regime:
   - If `r_cool > R_vir`: cold accretion -- the whole `HotGas` reservoir
     cools on the dynamical time.
   - If `r_cool <= R_vir`: hot-halo cooling -- mass flux is
     `(HotGas / R_vir) * (r_cool / (2 * t_cool))` per unit time.
5. If `CGMrecipeOn = 1` (Regime 1 only), blend in a Dekel & Birnboim (2006)
   cold-stream fraction `f_stream = (M_vir / M_shock)^(-4/3) * (1 + z) / 2`,
   capped at 0.5 and hard-cut to 0 at `z < 1.5` for halos above `M_shock`.
6. Apply `do_AGN_heating()` if `AGNrecipeOn > 0`.
7. Return the net cooled mass to the dispatcher.

## `cooling_recipe_cgm()` -- Regime 0 (precipitation)

The CGM recipe replaces the isothermal cooling-radius construction with a
Voit (2015) precipitation criterion based on the ratio of cooling time to
free-fall time.

### Step 1 -- density profile

The CGM gas distribution is selected by `CGMDensityProfile` (Regime 0 only;
Regime 1 always uses uniform for the residual `CGMgas` drain):

| Value | Profile |
|-------|---------|
| 0 | Uniform |
| 1 | NFW (concentration from Duffy+08) |
| 2 | Beta profile with `beta = 2/3`, core radius `r_c = 0.1 R_vir` |

Helper functions `nfw_density()`, `beta_density()`, and
`cgm_density_at_radius()` evaluate the profile; `cgm_enclosed_mass()`
returns the enclosed mass at any radius for free-fall calculations.

### Step 2 -- cooling radius

`solve_for_rcool()` iteratively finds the radius at which
`t_cool(r) = t_ff(r)`. The cooling time is
`t_cool = (3/2) * mu * m_p * k_B * T / (rho(r) * Lambda(T, Z))`, and the
free-fall time at that radius is `t_ff = sqrt(2 r / g)` with
`g = G M_enc(r) / r^2`.

### Step 3 -- characteristic radius

The precipitation criterion is evaluated at `r_cool` itself -- the
traditional Voit-style choice.

### Step 4 -- precipitation fraction

A smooth logistic sigmoid centred on `t_cool / t_ff = 10` with
characteristic width 2 sets the precipitation fraction. It falls back to
standard cooling on the cooling timescale when the sigmoid is negligible
(`f < 0.01`).

When the gas is "stable" (`t_cool / t_ff` well above threshold), it cools
slowly on the cooling timescale: `dM/dt = CGMgas / t_cool`. When it is
"unstable" (below threshold), it precipitates on the free-fall timescale:
`dM/dt = f_precip * CGMgas / t_ff`.

### Step 5 -- AGN heating (Regime 0 only)

The CGM cooling is passed to the standard `r_heat` ratchet, identical to
the hot-halo path but capped at `R_vir`. Suppression formula:
`coolingGas *= (1 - r_heat / r_cool)`.

`CGMAGNOn = 0` disables CGM-regime AGN coupling entirely (so no accretion
from `CGMgas`); the `r_heat` suppression still applies so quenching
persists after the AGN turns off.

### Step 6 -- diagnostics

`cooling_recipe_cgm()` populates several diagnostic fields on the galaxy:
`tcool`, `tff`, `tcool_over_tff`, `tdeplete`, `RcoolToRvir`. These are
the values reported in the HDF5 output for plotting.

## AGN radio-mode heating

Two functions implement the AGN suppression for the two regimes. Both
share the accretion calculation via the file-private
`agn_accretion_compute()` helper.

### `agn_accretion_compute()` -- the shared accretion calculation

Computes the BH accretion rate from one of three recipes
(`AGNrecipeOn = 1, 2, 3`), applies the Eddington limit, and derives the
corresponding heating mass. Three recipes:

| `AGNrecipeOn` | Recipe | Source |
|---------------|--------|--------|
| 1 | Empirical Croton+06 eq. 10: scales with `BH_mass / 10^8 Msun`, `(V_vir/200)^3`, `HotGas / M_vir` | Croton+06 |
| 2 | Bondi-Hoyle accretion: `dM/dt = 2.5 pi G * (3/8) * 0.6 * x * BH_mass * eta` | Bondi (1952) |
| 3 | Cold-cloud accretion: triggers when `BH_mass > 1e-4 * M_vir * (r_cool / R_vir)^3` | --- |

Accretion is then capped at the Eddington rate. The resulting heating mass
is converted from accreted mass via `AGNheating = (1.34e5 / V_vir)^2 *
AGNaccreted` (with the coefficient derived from `sqrt(2 * eta * c^2)`).

### `do_AGN_heating()` -- Regime 1 (hot halo)

The classical Croton+06 ratchet:

1. Suppress cooling by `(1 - r_heat / r_cool)`. If `r_heat >= r_cool`,
   cooling is fully zeroed.
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

## `cool_gas_onto_galaxy()`

For the original SAGE path (`CGMrecipeOn = 0`), `cooling_recipe()` calls
`cool_gas_onto_galaxy()` to transfer the cooled gas from `HotGas` to
`ColdGas` with metallicity tracking. In the regime-aware path the
dispatcher does the transfer itself.

## Switches and parameters

| Parameter | Effect |
|-----------|--------|
| `CGMrecipeOn` | 0 disables the two-regime split entirely; 1 enables it. |
| `CGMDensityProfile` | CGM density profile: 0 uniform, 1 NFW, 2 beta. |
| `CGMAGNOn` | Enables AGN heating coupling in the CGM regime. |
| `AGNrecipeOn` | Radio-mode BH accretion recipe: 0 off, 1 empirical, 2 Bondi-Hoyle, 3 cold-cloud. |
| `RadioModeEfficiency` | Overall scaling on radio-mode accretion. |
| `QuasarModeEfficiency` | Used by the merger-driven AGN path -- see [Mergers and disruption](mergers_and_disruptions.md). |

See [`parameters.md`](../parameters.md) for full descriptions and defaults.

## References

- White & Frenk (1991), ApJ 379, 52 -- classical halo cooling framework.
- Sutherland & Dopita (1993), ApJS 88, 253 -- metal-dependent cooling tables.
- Croton et al. (2006), MNRAS 365, 11 -- original SAGE cooling and AGN
  radio-mode prescriptions.
- Dekel & Birnboim (2006), MNRAS 368, 2 -- shock-mass criterion and cold
  streams.
- McCourt et al. (2012), MNRAS 419, 3319 -- thermal instability criterion.
- Sharma et al. (2012), MNRAS 420, 3174 -- precipitation-limited cooling.
- Voit (2015), ApJL 808, L30 -- precipitation criterion at `t_cool/t_ff = 10`.
- Duffy et al. (2008), MNRAS 390, L64 -- NFW concentration scaling.
- Bondi (1952), MNRAS 112, 195 -- accretion onto compact objects.
- Rybicki & Lightman (1979) -- Eddington luminosity.

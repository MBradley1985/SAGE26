# Infall, Reincorporation, and Stripping

This page covers the gas-supply side of the model: how baryons enter the
halo, how previously ejected gas comes back, and how satellite gas is
transferred to the central. All three operations are functions of the
halo's current potential and regime classification, not the local star
formation state.

Sources:
[`src/model_infall.c`](https://github.com/MBradley1985/SAGE26/blob/main/src/model_infall.c),
[`src/model_reincorporation.c`](https://github.com/MBradley1985/SAGE26/blob/main/src/model_reincorporation.c)

Called from: [Per-halo physics loop](../core_build_model.md) -- steps 4-7
of the substep ordering.

## What infall represents

At each snapshot, a halo of virial mass `Mvir` "should" contain a baryonic
mass `BaryonFrac * Mvir`. If it contains less, the missing baryons fall
in over the snapshot interval. If it contains more (the halo lost mass,
or its satellites brought in baryon-rich progenitors), the excess is
removed -- first from the ejected reservoir, then from the hot/CGM
reservoir.

SAGE26 routes the infalling gas to one of two destinations depending on the
central's regime classification:

- **Regime 0 (CGM):** infall accumulates in `CGMgas` and cools through the
  precipitation recipe.
- **Regime 1 (hot halo):** infall accumulates in `HotGas` and cools through
  the classical isothermal recipe.

## `infall_recipe()` -- the per-snapshot budget

Computed once per snapshot, before the substep loop opens:

```
infalling_mass = reionization_modifier * BaryonFrac * Mvir
               - (StellarMass + ColdGas + HotGas + EjectedMass
                  + BlackHoleMass + ICS + CGMgas)
```

The function also performs the per-snapshot satellite-to-central transfers:

- **Satellite EjectedMass** is reassigned to the central (centrals own the
  full ejected reservoir of the FoF group).
- **Satellite ICS** is added to the central's ICS; if `TrackICSAssembly`
  is on, the mass-weighted assembly history is also inherited and the
  contribution is recorded as accretion.
- **Satellite CGMgas** is summed into the central's appropriate reservoir:
  `CGMgas` if the central is in Regime 0, `HotGas` if it is in Regime 1
  (or unconditionally `HotGas` if `CGMrecipeOn` is off).

After these transfers, the function returns the per-snapshot total
`infalling_mass` (may be negative). The substep loop in `evolve_galaxies()`
then injects `infalling_mass / effective_steps` per substep via
`add_infall_to_hot()`.

## Reionization suppression -- `do_reionization()`

For halos with virial mass close to the filtering mass at the current
redshift, photoheating from the UV background prevents some of the
universal baryon fraction from falling in. SAGE26 uses the Gnedin (2000)
prescription with the Kravtsov et al. (2004) Appendix B fitting formulae:

```
modifier = (1 + 0.26 * M_F / Mvir)^-3
```

where `M_F` is the larger of the Gnedin filtering mass and the
characteristic mass corresponding to a virial temperature of 10^4 K.
The transition redshifts are controlled by:

- `Reionization_z0` -- the redshift at which reionization begins.
- `Reionization_zr` -- the redshift by which reionization is complete.

When `ReionizationOn = 0`, the modifier is fixed at 1 (no suppression).

## `add_infall_to_hot()` -- per-substep injection

Receives `infalling_gas = infalling_mass / effective_steps` per substep
and deposits it into the central's hot reservoir. The destination depends
on regime and the sign of the infall:

| Case | Destination |
|------|-------------|
| `infalling_gas > 0`, Regime 0 | `CGMgas` (metallicity unchanged) |
| `infalling_gas > 0`, Regime 1 | `HotGas` (metallicity unchanged) |
| `infalling_gas > 0`, `CGMrecipeOn = 0` | `HotGas` |
| `infalling_gas < 0` | first drained from `EjectedMass`, remainder from `CGMgas`/`HotGas` via metal-weighted draw |

The metal-weighted draw on negative infall preserves the metallicity of
the donor reservoir so that mass and metal accounting stay consistent.

## `reincorporate_gas()` -- ejected mass returning to hot

When supernova feedback ejects gas from the halo (into `EjectedMass`),
some of it falls back over the dynamical time. SAGE26 uses a
velocity-thresholded recipe scaled by `ReIncorporationFactor`:

```
reincorporated = (Vvir / Vcrit - 1) * EjectedMass * dt / t_dyn
```

where `t_dyn = Rvir / Vvir` and `Vcrit = 445.48 km/s * ReIncorporationFactor`.
The function only fires when `Vvir > Vcrit`, so low-mass halos do not
recover ejected material -- this is the mechanism that lets SN feedback
permanently quench faint galaxies.

The reincorporated gas inherits the metallicity of the ejected reservoir
and is routed by regime: `CGMgas` if Regime 0, `HotGas` if Regime 1
(or always `HotGas` when `CGMrecipeOn = 0`).

## `strip_from_satellite()` -- per-substep satellite stripping

For Type 1 satellites (those with their own dark matter subhalo) the
function transfers excess gas from the satellite to the central, one
substep at a time. The "excess" is defined as the satellite's current
baryons above `BaryonFrac * Mvir_sat * reionization_modifier`. A fraction
`1 / effective_steps` of the excess is stripped per call, so the total
per-snapshot stripping fraction is independent of the adaptive substep
count.

For CGM-regime satellites the bulk transfer has already happened in
`infall_recipe()` (CGMgas was zeroed and merged into the central's
reservoir). If the satellite's `CGMgas` is empty but `HotGas` remains
-- typical of a satellite that crossed the M_shock threshold during its
lifetime -- the function falls back to stripping `HotGas` so that residual
hot gas is not stranded.

Type 2 satellites (orphan satellites with no remaining subhalo) have no
hot reservoir and are skipped.

## Switches and parameters

| Parameter | Effect |
|-----------|--------|
| `ReionizationOn` | 0 disables the Gnedin/Kravtsov filtering-mass modifier. |
| `Reionization_z0` | Reionization onset redshift. |
| `Reionization_zr` | Reionization completion redshift. |
| `BaryonFrac` | Universal baryon fraction f_b = Omega_b / Omega_m. |
| `ReIncorporationFactor` | Sets `Vcrit` for the reincorporation cutoff. Larger values delay reincorporation in low-mass halos. |
| `CGMrecipeOn` | Routes infall, reincorporation, and satellite CGM by regime when set. |
| `TrackICSAssembly` | Records satellite-derived ICS mass into `ICS_accrete` for the central. |

See [`parameters.md`](../parameters.md) for full descriptions and defaults.

## References

- Gnedin (2000), ApJ 542, 535 -- reionization filtering mass.
- Kravtsov, Gnedin & Klypin (2004), ApJ 609, 482 -- Appendix B fitting
  formulae for the filtering mass.
- Bryan & Norman (1998), ApJ 495, 80 -- virial overdensity Delta_c(z).
- Croton et al. (2006), MNRAS 365, 11 -- original SAGE infall recipe,
  baryon budget, and velocity-thresholded reincorporation.
- Dekel & Birnboim (2006), MNRAS 368, 2 -- M_shock criterion underlying
  regime classification.
- Voit (2015), ApJL 808, L30 -- CGM precipitation framework that informs
  the regime-aware routing.

# Per-Halo Physics Loop

This page is the hub for the SAGE26 physics walkthrough. It describes
what happens inside `evolve_galaxies()` -- the function that advances every
galaxy in a halo from one snapshot to the next -- and links out to the four
detailed module pages.

Source: [`src/core_build_model.c`](https://github.com/MBradley1985/SAGE26/blob/main/src/core_build_model.c)

## In one paragraph

`evolve_galaxies()` takes a halo and the galaxy list inherited from its
progenitors, advances every galaxy through the snapshot interval, and writes
the result back. Internally the interval is split into a number of substeps
and the same physics block runs on every substep: gas infall, reincorporation,
satellite stripping, cooling, star formation and feedback, then mergers and
disruption. Two regimes (CGM vs hot halo) are classified once at the top of
the function and reused throughout.

## Sub-stepping

Each snapshot interval is integrated in `STEPS = 10` substeps by default
(defined in [`src/macros.h`](https://github.com/MBradley1985/SAGE26/blob/main/src/macros.h)).
At high redshift the snapshot spacing can exceed the halo dynamical time
`t_dyn = R_vir / V_vir`, so `evolve_galaxies()` raises the substep count
adaptively:

```
effective_steps = clamp(ceil(STEPS * deltaT_total / t_dyn), STEPS, MAX_STEPS)
```

with `MAX_STEPS = 30`. The SFR history arrays (`SFHMassDisk[STEPS]`,
`SFHMassBulge[STEPS]`, etc.) remain sized to the fixed `STEPS`. Adaptive
substeps are mapped back into those bins via
`step_bin = (step * STEPS) / effective_steps`, so the output schema is
stable regardless of the integration cadence.

## Order of operations within a substep

The substep loop is the spine of the model. Steps 1-3 run once per snapshot,
before the loop opens.

| # | Step | Where |
|---|------|-------|
| 1 | Halo concentration (if `ConcentrationOn > 0`) | `get_halo_concentration()` |
| 2 | CGM regime classification (if `CGMrecipeOn`) | `determine_and_store_regime()` |
| 3 | FFB regime classification (if `FeedbackFreeModeOn`) | `determine_and_store_ffb_regime()` |
| 4 | Compute total infalling gas for the snapshot | `infall_recipe()` |
| 5 | Inject the per-substep share of infall into HotGas or CGMgas (central only) | `add_infall_to_hot()` |
| 6 | Return ejected gas to hot reservoir on `t_dyn` (central only) | `reincorporate_gas()` |
| 7 | Strip satellite hot gas to the central (Type 1 satellites only) | `strip_from_satellite()` |
| 8 | Cool gas from the hot/CGM reservoir into cold gas | `cooling_recipe_regime_aware()` |
| 9 | Form stars and apply SN + AGN feedback | `starformation_and_feedback()` |
| 10 | Decrement `MergTime`; merge or disrupt flagged satellites | `deal_with_galaxy_merger()` / `disrupt_satellite_to_ICS()` |

Steps 4-10 run on every substep. The infall total is divided by
`effective_steps` and injected as `infallingGas / effective_steps` per pass.

Detailed treatment of each physics step lives in its own page:

- Step 4-7: [Infall, reincorporation, stripping](physics/infall.md)
- Step 8: [Cooling and AGN heating](physics/cooling_and_heating.md)
- Step 9: [Star formation and feedback](physics/starformation_and_feedback.md)
- Step 10: [Mergers and disruption](physics/mergers_and_disruptions.md)

## Post-step bookkeeping

After the substep loop completes, `evolve_galaxies()` does several
end-of-snapshot tasks:

- Normalise `Cooling`, `Heating`, and `OutflowRate` to per-unit-time
  by dividing the accumulated values by `deltaT`.
- Sum `TotalSatelliteBaryons` on the central from each remaining satellite.
- Re-attach the galaxy list to `haloaux[]` for downstream output.
- Shift `mergeIntoID` to account for merged galaxies that will not be
  written out (output indices are a contiguous range, so every preceding
  merged galaxy bumps the index down by one).

## The two-regime split as a code path map

The CGM model (`CGMrecipeOn = 1`) classifies every galaxy as Regime 0
(CGM / precipitation, below the Dekel & Birnboim 2006 M_shock) or
Regime 1 (hot halo, classical Croton+06). The classification controls
several decisions in the substep loop:

| Step | Regime 0 (CGM) | Regime 1 (hot halo) |
|------|----------------|---------------------|
| Infall destination | `CGMgas` | `HotGas` |
| Cooling recipe | `cooling_recipe_cgm()` | `cooling_recipe()` |
| Density profile | `CGMDensityProfile` (uniform / NFW / beta) | isothermal |
| AGN suppression mechanism | `CGMHeatingRheatOn` selector (off / `f_heat_cgm` decay / `r_heat` ratchet) | `r_heat` ratchet |
| Precipitation threshold | `t_cool / t_ff` (Voit 2015) | Sutherland-Dopita cooling time |

When `CGMrecipeOn = 0`, every galaxy uses the classical Regime 1 path
regardless of mass, recovering the original SAGE behaviour.

## Galaxy types

The substep loop branches on `galaxies[p].Type`:

| Type | Meaning | Special handling |
|------|---------|------------------|
| 0 | Central of its FoF halo | Receives infall, reincorporation, cooling, SF |
| 1 | Satellite with its own dark matter subhalo | Hot gas is stripped to the central; otherwise evolves like a central |
| 2 | Orphan satellite (subhalo lost) | No hot reservoir; only SF from existing cold gas; merger timer counts down |

Galaxies with `mergeType > 0` are skipped in the loop -- they have already
been absorbed by a merger event earlier in the snapshot.

## Reading the source

The function is intentionally a flat, top-to-bottom dispatcher with no
hidden state. To trace any single galaxy through one snapshot, follow these
landmarks in `core_build_model.c`:

1. Pre-step classifications -- lines around `evolve_galaxies` entry.
2. The `for(int step = 0; step < effective_steps; step++)` loop body.
3. Inside it, the `for(int p = 0; p < ngal; p++)` per-galaxy loop for cooling
   and SF, followed by the satellite/merger loop.
4. After the substep loop, the per-galaxy normalisation block and the
   `haloaux[]` linkage block.

## Where to go next

- [Infall, reincorporation, stripping](physics/infall.md)
- [Cooling and AGN heating](physics/cooling_and_heating.md)
- [Star formation and feedback](physics/starformation_and_feedback.md)
- [Mergers and disruption](physics/mergers_and_disruptions.md)
- [Parameter reference](parameters.md)

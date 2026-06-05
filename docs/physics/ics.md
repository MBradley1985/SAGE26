# Intracluster Stars (ICS)

This page describes the intracluster (or intragroup) stellar component:
the diffuse stellar light that fills the host halo outside any single
galaxy. SAGE26 tracks ICS as a separate reservoir on every central, with
optional assembly-history bookkeeping.

Sources:
[`src/model_infall.c`](https://github.com/MBradley1985/SAGE26/blob/main/src/model_infall.c),
[`src/model_mergers.c`](https://github.com/MBradley1985/SAGE26/blob/main/src/model_mergers.c),
[`src/core_allvars.h`](https://github.com/MBradley1985/SAGE26/blob/main/src/core_allvars.h)
(GALAXY struct).

## Fields

ICS lives on every galaxy but is only ever non-zero on centrals -- at the
start of each snapshot any satellite ICS is transferred to the central
(see below).

| Field | Units | Meaning |
|-------|-------|---------|
| `ICS` | 10^10 M_sun/h | Current intracluster stellar mass on this central. |
| `MetalsICS` | 10^10 M_sun/h | Metal mass in the ICS reservoir. |
| `ICS_disrupt` | 10^10 M_sun/h | Cumulative stellar mass *disrupted* into ICS via `disrupt_satellite_to_ICS()`. Only populated when `TrackICSAssembly = 1`. |
| `ICS_accrete` | 10^10 M_sun/h | Cumulative ICS mass *inherited* from satellites that already carried their own ICS (i.e. former group centrals that fell in). Only populated when `TrackICSAssembly = 1`. |
| `ICS_sum_mt` | 10^10 M_sun/h * code time | Mass-weighted accumulator: sum over all ICS deposition events of `delta_M * t_deposit`. Used to derive the mean assembly lookback as `ICS_sum_mt / (ICS_disrupt + ICS_accrete)`. |

The split between `ICS_disrupt` and `ICS_accrete` lets you decompose the
assembly history into "stars stripped here vs stars that arrived already
stripped from a previous host" without needing a full SFH array on the
ICS reservoir.

## How ICS forms

There are three pathways:

### 1. Per-snapshot inheritance (always)

`infall_recipe()` runs once at the start of each snapshot. As it walks
the FoF galaxy list, it:

- Sums every galaxy's `ICS` and `MetalsICS` into a per-FoF total.
- Zeros each satellite's `ICS` and `MetalsICS`.
- Assigns the total to the central:
  ```
  galaxies[centralgal].ICS        = tot_ICS;
  galaxies[centralgal].MetalsICS  = tot_ICSMetals;
  ```

This makes the FoF central own all of the group's ICS by construction --
which is what you want for plotting (no need to integrate over satellites
when measuring a central's ICS).

If `TrackICSAssembly = 1` and an incoming satellite carried ICS, the
amount is recorded in the central's `ICS_accrete` accumulator and the
satellite's `ICS_sum_mt` is added to the central's. This preserves the
deposition history of stars that were stripped in an earlier host before
this halo absorbed them.

### 2. Satellite disruption

When `disrupt_satellite_to_ICS()` fires (see
[Mergers and disruption](mergers_and_disruptions.md)), some fraction of
the satellite's stellar mass is added to the central's ICS:

```
new_ICS_from_stripping = f_ICS * StellarMass_sat
```

where `f_ICS` is set by `DynamicDisruptionSplit`:

- Mode 0: fixed `FractionDisruptedToICS`.
- Mode 1: mass-ratio split `f_ICS = 1 - (infallMvir_sat / Mhost)^DisruptionSplitAlpha`.
- Mode 2: as mode 1, with `alpha_eff = DisruptionSplitAlpha * DisruptionSplitCref / c_sat` so concentrated satellites resist stripping and deposit more onto the BCG.

The remaining `1 - f_ICS` is added to the central's `StellarMass`,
`BulgeMass`, and `MergerBulgeMass` (BCG growth).

When `TrackICSAssembly = 1`, this event contributes:

- `ICS_disrupt += new_ICS_from_stripping`
- `ICS_sum_mt += new_ICS_from_stripping * time` (with `time` in code units, the substep midpoint)

### 3. Satellite merger transfer

When a satellite is merged (rather than disrupted) by
`deal_with_galaxy_merger()` -> `add_galaxies_together()`, any ICS the
satellite already carried is added to the central:

```
galaxies[t].ICS        += galaxies[p].ICS;
galaxies[t].MetalsICS  += galaxies[p].MetalsICS;
```

This is bookkeeping rather than new ICS formation -- the stars existed
already; they just changed hosts.

## How to compute the mean ICS assembly time

`ICS_sum_mt` is a mass-weighted sum: every time `delta_M` of stars is
deposited at code time `t`, the accumulator gains `delta_M * t`. The
mean deposition time is therefore:

```
<t_assembly> = ICS_sum_mt / (ICS_disrupt + ICS_accrete)
```

To convert to lookback time, subtract from the host's current snapshot
age. SAGE26 does not store this derived quantity directly -- compute it
in post-processing from the three tracked fields.

The denominator is the *cumulative deposited mass*, not the current
`ICS`. The two can differ if stars merge into the BCG over time
(consumed from the ICS reservoir but still counted in the assembly
totals).

## ICS in the baryon budget

`ICS` participates in the baryon accounting in `infall_recipe()`:

```
infalling_mass = reionization_modifier * BaryonFrac * Mvir
               - (StellarMass + ColdGas + HotGas + EjectedMass
                  + BlackHoleMass + ICS + CGMgas)
```

So ICS stars count toward the halo's baryon budget for infall purposes --
infalling gas is reduced by however much mass is already in the ICS
reservoir, as it should be.

ICS does **not** participate in cooling, star formation, or feedback --
the stars are passive once deposited. No SFR is recorded for ICS, no
SFH array is maintained, no metal production fires. If you need stellar
ages for ICS, derive them in post-processing from the assembly history.

## Type-0 vs satellite ICS

Strictly, only Type 0 (centrals) hold ICS. Satellites entering a halo
that themselves were former centrals can carry ICS, but at every
snapshot the FoF central absorbs all of it (see Pathway 1 above). So
in any output snapshot you should typically see:

- `Type == 0`: `ICS >= 0` (may be nonzero)
- `Type == 1` or `2`: `ICS == 0` (transferred to the central in the
  same snapshot it entered)

If you find a satellite with nonzero `ICS` in the output, it's the
result of the satellite having just become a satellite this snapshot
and the data being written before the next infall pass -- not a bug,
but worth knowing when computing population statistics.

## Switches and parameters

| Parameter | Effect |
|-----------|--------|
| `TrackICSAssembly` | 0 disables `ICS_disrupt` / `ICS_accrete` / `ICS_sum_mt` tracking; the `ICS` field itself is always tracked. |
| `DynamicDisruptionSplit` | Controls the disruption split between ICS and BCG (see [Mergers and disruption](mergers_and_disruptions.md)). |
| `FractionDisruptedToICS` | Fixed split fraction (mode 0) or fallback. |
| `DisruptionSplitAlpha` | Exponent in the mass-ratio split (modes 1 and 2). |
| `DisruptionSplitCref` | Reference concentration for mode 2. |

See [`parameters.md`](../parameters.md) for full descriptions and defaults.

## References

- Murante et al. (2007), MNRAS 377, 2 -- ICS as a distinct component
  growing primarily through satellite disruption.
- Conroy, Wechsler & Kravtsov (2007), ApJ 668, 826 -- BCG-vs-ICS
  partition during disruption.
- Pillepich et al. (2018), MNRAS 475, 648 -- mass-ratio dependence of
  the disruption split.
- Contini et al. (2014), MNRAS 437, 3787 -- semi-analytic ICS treatment
  underlying the SAGE26 implementation.

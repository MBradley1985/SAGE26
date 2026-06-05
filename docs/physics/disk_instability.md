# Disk Instability

This page describes the Toomre disk-instability channel that grows the
instability-driven bulge component and feeds the central black hole.

Source: [`src/model_disk_instability.c`](https://github.com/MBradley1985/SAGE26/blob/main/src/model_disk_instability.c)
(with the Tonini+2016 radius update in
[`src/model_misc.c`](https://github.com/MBradley1985/SAGE26/blob/main/src/model_misc.c))

Called from: `starformation_and_feedback()` after star formation, when
`DiskInstabilityOn = 1`. Not run in the FFB path
([Star formation and feedback](starformation_and_feedback.md)).

## What disk instability does

After each star-formation event, the disk's combined mass of cold gas and
disk stars is compared against the Mo, Mao & White (1998) Toomre critical
mass:

```
M_crit = V_max^2 * 3 * DiskScaleRadius / G
```

(the factor of 3 comes from the disk half-mass radius being ~1.68 r_s and
the Mo-Mao-White stability analysis for an exponential disk; see
`TOOMRE_DISK_FACTOR` in the source). If the disk mass exceeds `M_crit`,
the excess is "unstable" and must be redistributed.

## Splitting the unstable mass

The unstable excess is split between gas and stars in proportion to the
current cold-gas fraction:

```
gas_fraction   = ColdGas / disk_mass
unstable_gas   = gas_fraction       * (disk_mass - M_crit)
unstable_stars = (1 - gas_fraction) * (disk_mass - M_crit)
```

where `disk_mass = ColdGas + (StellarMass - BulgeMass)`.

## Where the unstable stars go

Unstable stars are transferred straight into the **instability-driven
bulge channel**:

- `BulgeMass += unstable_stars`
- `InstabilityBulgeMass += unstable_stars` (separate channel from
  merger-driven bulge growth -- see
  [Mergers and disruption](mergers_and_disruptions.md))
- `MetalsBulgeMass` is updated with the disk-star metallicity

`InstabilityBulgeRadius` is then updated via the Tonini+2016 incremental
radius evolution (Eq. 15), using a snapshot of `DiskScaleRadius` taken
**before** any mass transfer in this event:

```
R_new = (R_old * M_old + delta_M * 0.2 * R_disc) / (M_old + delta_M)
```

The 0.2 factor (`TONINI16_DISK_FRAC` in the source) is the fraction of the
disk scale radius at which instability-driven bulge stars are deposited.
On the first instability event when `M_old == 0`, the radius is
initialised directly as `0.2 * R_disc`. This update only fires when
`BulgeSizeOn = 3` (Tonini mode); other bulge models leave
`InstabilityBulgeRadius` unchanged.

The disk scale radius itself is **not changed** by the instability: the
remaining disk retains the same specific angular momentum per unit mass,
so `DiskScaleRadius` stays constant.

## Where the unstable gas goes

If there is unstable gas in the disk:

1. **Black hole accretion** (if `AGNrecipeOn > 0`):
   ```
   grow_black_hole(p, unstable_gas_fraction, ...)
   ```
   uses the same quasar-mode channel as mergers, with the unstable gas
   fraction playing the role of the merger mass ratio. This both grows
   the BH and triggers `quasar_mode_wind()` (see
   [Mergers and disruption](mergers_and_disruptions.md)).

2. **Collisional starburst** (mode 1):
   ```
   collisional_starburst_recipe(unstable_gas_fraction, p, centralgal,
                                time, dt, halonr,
                                /*mode=*/1, step,
                                /*burst_to_merger_bulge=*/0,
                                DiskScaleRadius, galaxies, run_params)
   ```
   Mode 1 makes the burst efficiency `eburst = mass_ratio` (i.e. the
   entire unstable gas fraction is consumed) rather than the
   Cox-thesis power law used by mergers. The `burst_to_merger_bulge = 0`
   flag routes burst stars into the instability bulge, consistent with
   the secular origin of the event.

The starburst itself applies SN feedback through the standard
`update_from_feedback()` path, with FIRE scaling when `FIREmodeOn = 1`.

## Order of operations and bookkeeping

Inside the function the order matters because the bulge-radius update
needs the pre-event disk radius:

1. Compute `M_crit` and clip if it exceeds the actual disk mass.
2. Compute `unstable_gas` and `unstable_stars` from the gas fraction.
3. **Save `old_disk_radius = DiskScaleRadius` before any mass transfers.**
4. Transfer unstable stars to the bulge; call
   `update_instability_bulge_radius()` with `old_disk_radius`.
5. Burst the unstable gas (BH growth + starburst).

The "save disk radius first" step is the reason this function passes
`old_disk_radius` explicitly into both helper paths rather than letting
them read the live value.

## What is NOT in this module

- **Toomre Q itself.** The criterion is implemented as a mass comparison
  against `M_crit`, not a stability parameter Q.
- **Pseudobulge classification.** The instability bulge is treated as a
  classical bulge component for radius and ICS purposes.
- **`BulgeRadius` derivation.** Done in `model_misc.c` via
  `get_bulge_radius()` as a mass-weighted combination of
  `InstabilityBulgeRadius` and `MergerBulgeRadius` (when `BulgeSizeOn = 3`).

## Switches and parameters

| Parameter | Effect |
|-----------|--------|
| `DiskInstabilityOn` | 0 disables the entire channel. |
| `AGNrecipeOn` | 0 skips the BH accretion step but still runs the gas burst. |
| `BulgeSizeOn` | Must be 3 (Tonini+2016 multi-channel) for the incremental `InstabilityBulgeRadius` update to fire. |
| `FIREmodeOn` | Applies FIRE scaling to the burst SN feedback. |
| `StarburstColdGasOn` | Controls whether the burst uses stored `H2gas` or recomputes from `ColdGas`. |
| `BlackHoleGrowthRate` | Scaling for `grow_black_hole()` accretion in the instability path. |
| `QuasarModeEfficiency` | Coupling between BH accretion and wind energy in `quasar_mode_wind()`. |

See [`parameters.md`](../parameters.md) for full descriptions and defaults.

## References

- Mo, Mao & White (1998), MNRAS 295, 319 -- Toomre critical mass for an
  exponential disk; the form `M_crit = V_max^2 * 3 R_d / G` used here.
- Toomre (1964), ApJ 139, 1217 -- the underlying axisymmetric stability
  criterion for thin disks.
- Tonini et al. (2016), MNRAS 459, 4109 -- two-channel bulge formation
  with separate radius evolution for instability- and merger-driven
  components (Eq. 15 used here).
- Croton et al. (2006), MNRAS 365, 11 -- original SAGE disk-instability
  treatment.

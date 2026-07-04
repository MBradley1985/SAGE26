# Units and Conventions

This page defines the internal unit system used throughout the SAGE26 source,
the h-factor conventions inherited from the input merger trees, and the naming
convention that marks any variable *not* in code units.

Source: unit setup in [`src/core_init.c`](https://github.com/MBradley1985/SAGE26/blob/main/src/core_init.c)
(`set_units()`); cgs constants in [`src/macros.h`](https://github.com/MBradley1985/SAGE26/blob/main/src/macros.h).

## Code units

The three base units are set in the parameter file and default to the
standard Millennium/LHaloTree convention:

| Quantity | Parameter            | Default value            | Meaning        |
|----------|----------------------|--------------------------|----------------|
| Length   | `UnitLength_in_cm`   | 3.08568e24 (1 Mpc)       | 1 code length = 1 Mpc/h |
| Mass     | `UnitMass_in_g`      | 1.989e43 (10^10 Msun)    | 1 code mass = 10^10 Msun/h |
| Velocity | `UnitVelocity_in_cm_per_s` | 1e5 (1 km/s)       | 1 code velocity = 1 km/s |

All derived units follow from these three (computed once in `set_units()`):

- **Time**: `UnitTime_in_s = UnitLength_in_cm / UnitVelocity_in_cm_per_s`
  (~0.98 h^-1 Gyr for the defaults). `UnitTime_in_Megayears` is the same
  interval expressed in Myr.
- **Energy**: `UnitEnergy_in_cgs = UnitMass_in_g * UnitLength_in_cm^2 / UnitTime_in_s^2`
- **Density**: `UnitDensity_in_cgs = UnitMass_in_g / UnitLength_in_cm^3`
- **G** and **Hubble** are converted into code units so that dynamical
  formulae can be written without unit factors; `RhoCrit` is the critical
  density in code units.

## h-factor conventions

The little-h conventions come from the LHaloTree input format and propagate
unchanged through the code:

| Quantity | Units | Notes |
|----------|-------|-------|
| Masses (all reservoirs, Mvir, ...) | 10^10 Msun/h | `ColdGas`, `StellarMass`, `HotGas`, `CGMgas`, ... |
| Lengths (Rvir, disk/bulge radii)   | Mpc/h        | positions are *comoving*; radii are physical at the galaxy's epoch |
| Velocities (Vvir, Vmax, Vel)       | km/s         | physical (peculiar), no h |
| Times (dT, MergTime, tcool, ...)   | code time    | multiply by `UnitTime_in_Megayears` for Myr |
| SFR trackers (`SfrDisk[]`, ...)    | 10^10 Msun/h / code time | converted to Msun/yr at output |
| Cooling/Heating luminosities       | code energy / code time | converted to log10(erg/s) at output |

## Departures from code units

Two areas of the physics deliberately work in non-code units:

- **H2 / surface-density calculations** (BR06, KD12, K13, GD14 and the
  Somerville+25 efficiency) work in **Msun/pc^2** and **pc**, because the
  source papers' fits are calibrated in those units. Inputs are converted on
  entry (e.g. `rs_pc = DiskScaleRadius * 1.0e6 / Hubble_h`).
- **Cooling tables and Eddington/AGN normalisations** use **cgs** constants
  from `macros.h`.

The **output layer** (`io/save_gals_binary.c`, `io/save_gals_hdf5.c`) is the
only other place unit conversions happen: SFRs to Msun/yr, times to Myr,
cooling/heating to erg/s. Everything between input and output stays in code
units.

## Frozen single-precision sites

Several calculations deliberately drop to single precision mid-stream:
`const float h = run_params->Hubble_h` in the SF prescriptions, one
`1.0e10f` float literal in `sfr_somerville25_h2()`, and the float
internals of the H2 fraction fits in `model_h2_chemistry.c`. In ordinary
code review these would be defects; here they are **calibrated, frozen
numerical behaviour** -- promoting them to double changes the model
output. Each site carries a `float on purpose` comment. Do not "fix"
them; if a future recalibration intentionally changes precision, all
regression baselines must be re-captured at the same time.

## Naming convention

A physical-quantity variable with **no unit suffix is in code units**. Any
variable in other units carries an explicit suffix:

- `_pc`, `_pc2` -- parsecs, parsecs squared (`rs_pc`, `disk_area_pc2`)
- `_kms` -- km/s (`FIRE_V_CRIT_KMS`)
- `_cgs` -- cgs (`G_CGS`, `EDDINGTON_LUM_PER_MSUN_CGS`)
- `_Gyr`, `_Myr` -- gigayears / megayears (`H2DepletionTime_Gyr`)
- `_Msun` -- solar masses without the 10^10/h scaling (`MSHOCK_DB06_MSUN`)

When adding new code, keep calculations in code units for as long as
possible and convert once, at the point where a non-code-unit quantity is
actually required.

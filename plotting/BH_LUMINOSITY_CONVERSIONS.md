# BH luminosity/magnitude conversions used in `bh_lrd_analysis.py`

This note documents every physical constant and conversion used by panels
(d) and (e) of `bh_lrd_analysis.py` (bolometric luminosity and rest-frame UV
absolute magnitude, both derived from the simulated BH accretion rate
`Mdot_BH`), plus the Eddington-luminosity constant shared with panels (a)
and (c). Each step lists the formula, the constants involved, and the
literature source.

## 1. Bolometric luminosity from accretion rate

Standard thin-disk accretion (Shakura & Sunyaev 1973, A&A, 24, 337) relates
the bolometric luminosity radiated by the accretion flow to the mass
accretion rate through a radiative efficiency `eta`:

```
L_bol = eta * Mdot_BH * c^2
```

- `eta = 0.1` — the canonical thin-disk efficiency for a non-spinning
  (Schwarzschild) black hole (Shakura & Sunyaev 1973). This is the **same**
  value already assumed by the C simulation code in
  `AGN_RADIATIVE_EFFICIENCY` (`src/model_enhanced_bhphysics.c`), so `L_bol`
  computed here is dynamically consistent with the Eddington-limiting
  already applied inside SAGE26.
- `c = 2.99792458e10 cm/s` (exact, SI definition).
- `Mdot_BH` is read from `BHMaxaccretionRate` (M_sun/yr, converted to g/s
  internally: `Mdot_BH[g/s] = Mdot_BH[Msun/yr] * M_sun[g] / (s/yr)`, with
  `M_sun = 1.989e33 g`).

**Caveat:** real radiative efficiencies depend on BH spin and can range
from ~0.04 (retrograde, maximally spinning) to ~0.32 (prograde, maximally
spinning) (Novikov & Thorne 1973); `eta=0.1` is a fiducial value, not a
per-object measurement. Because SAGE26 does not track spin, `L_bol` here
should be read as "the L_bol implied if these BHs accrete at the standard
efficiency", not a spin-resolved prediction.

Implemented in `lbol_from_mdot()`.

## 2. Eddington luminosity (used for the lambda_Edd lines in panels a, c, d)

```
L_Edd = 1.3e38 * (M_BH / M_sun)  erg/s
```

This is the standard electron-scattering Eddington limit,
`L_Edd = 4*pi*G*M_BH*m_p*c / sigma_T` (Rybicki & Lightman 1979, *Radiative
Processes in Astrophysics*, eq. 1.4.9); the coefficient evaluates to
`1.26e38 erg/s/Msun`, commonly rounded to `1.3e38`. This is the exact
constant already hardcoded in `model_enhanced_bhphysics.c`
(`EDDINGTON_LUM_PER_MSUN_CGS`), so the diagonal lines in panels a/c/d are
guaranteed to use the same Eddington relation as the simulation's own
Eddington-limiter.

Implemented in `eddington_luminosity()`.

### Bug fixed alongside this work: `T_SALPETER_YR`

Panels (a) and (c)'s "Mdot_BH = Mdot_Edd" reference line is drawn from
`Mdot_Edd = M_BH / T_Salpeter`, where the Salpeter time is the standard
e-folding growth timescale,

```
T_Salpeter = eta * (M_BH * c^2) / L_Edd(M_BH)
```

which (substituting `eta=0.1` and `L_Edd/M_BH = 1.3e38 erg/s/Msun` above)
evaluates to **~4.36e7 yr (~44 Myr)** — the standard textbook Salpeter time.
The constant previously hardcoded in the script was `4.5e8 yr`, 10x too
large, which pushed the analytic reference line about **1.15 dex below**
the true simulated Eddington rate (confirmed by comparing against the
sim's own `BHEddingtonRateLimit` column). This only affected the plotted
reference *line* and the LRD shaded-region boundary in panel (a) — the
actual LRD red/blue selection masks always used the simulation's own
`BHEddingtonRateLimit` values and were unaffected. `T_SALPETER_YR` is now
*derived* from `AGN_RADIATIVE_EFFICIENCY` and `EDDINGTON_LUM_PER_MSUN_CGS`
rather than hardcoded, so it cannot drift out of sync with those two
constants again.

## 3. Bolometric correction: L_bol -> L(1450A)

To get from the bolometric luminosity to a UV magnitude, `L_bol` is
converted to the monochromatic quantity `lambda*L_lambda` at rest-frame
1450A using the constant bolometric correction of Runnoe, Brotherton &
Shang (2012, MNRAS, 422, 478; coefficients corrected in the December 2012
erratum, MNRAS, 427, 1800):

```
L_iso = BC_1450 * lambda*L_lambda(1450A),   BC_1450 = 4.20 +/- 0.15
```

`L_iso` is their notation for the isotropic bolometric luminosity
(integrated ~1 micron to 8 keV from a mean quasar SED, fit to 63 bright
quasars at z=0.03-1.4), which is used here interchangeably with `L_bol`.
This is the standard shortcut used throughout the AGN/quasar literature to
go between an observed monochromatic UV luminosity and a bolometric
luminosity (or vice versa, as done here).

Runnoe et al. (2012) also give a luminosity-dependent (power-law) form,
`log(L_iso) = 4.745 + 0.910*log(lambda*L_lambda(1450A))`, which differs
from the constant-ratio form by less than ~0.1 dex over the luminosity
range relevant here; the constant-ratio form is used for simplicity.

**Caveat:** this correction is calibrated on unobscured, low-to-moderate
redshift, moderate-to-high luminosity quasars. Its applicability to
z>5 "Little Red Dots", whose continuum SEDs are still debated (some
authors favor a reddened AGN, others a mixed AGN+stellar origin), is an
extrapolation, not a validated measurement — treat panel (e)'s M_1450
values as illustrative, not as photometric predictions.

Implemented in the first step of `m1450_from_lbol()`.

## 4. L(1450A) -> M_1450 (rest-frame absolute AB magnitude)

This step is a purely definitional unit conversion (Oke & Gunn 1983, ApJ,
266, 713 — definition of the AB magnitude system), independent of any AGN
physics:

1. Convert `lambda*L_lambda` [erg/s] to a specific luminosity `L_nu`
   [erg/s/Hz] at `lambda = 1450A`:
   ```
   L_nu = (lambda*L_lambda) * lambda / c
   ```
2. Place the source at the AB/absolute-magnitude reference distance of
   10 pc (`d_10pc = 3.0856775814913673e19 cm`) and compute the flux
   density there:
   ```
   f_nu = L_nu / (4*pi*d_10pc^2)
   ```
3. Apply the AB zero point:
   ```
   M_1450 = -2.5*log10(f_nu) - 48.60
   ```

Because this is an *absolute* magnitude evaluated at the fixed 10 pc
reference distance (not the object's actual luminosity distance), no
redshift or K-correction enters here — `M_1450` is already a rest-frame
quantity by construction. This matches the convention used throughout the
high-z UV luminosity function literature (e.g., Bouwens et al. 2015, ApJ,
803, 34) for quoting `M_1450` / `M_UV`.

Implemented in the second step of `m1450_from_lbol()`.

## Summary of the pipeline

```
Mdot_BH [Msun/yr]                              (SAGE26: BHMaxaccretionRate)
   |  x eta=0.1, x c^2                          (Shakura & Sunyaev 1973)
   v
L_bol [erg/s]
   |  / BC_1450=4.20                            (Runnoe, Brotherton & Shang 2012)
   v
lambda*L_lambda(1450A) [erg/s]
   |  x lambda/c                                (unit conversion)
   v
L_nu(1450A) [erg/s/Hz]
   |  / (4 pi d_10pc^2), AB zero point           (Oke & Gunn 1983)
   v
M_1450
```

All constants and functions above live in `plotting/bh_lrd_analysis.py`
(`lbol_from_mdot`, `eddington_luminosity`, `m1450_from_lbol`), alongside
the constants block near the top of the file
(`AGN_RADIATIVE_EFFICIENCY`, `EDDINGTON_LUM_PER_MSUN_CGS`, `BC_1450`,
`AB_ZEROPOINT`, etc.).

# SAGE26 issues 1 and 2: cold gas and metallicities

Investigation of items 1 (massive galaxies too metal-poor) and 2 (too much cold gas) in `SAGE26_investigation.md`. Nothing here changes or reruns the model.

- **Data:** microUchuu Snap_48 (z = 0.022; Snap_49 is avoided because of the last-snapshot artefact), plus snapshots 15, 20, 25, 30 and 40 for evolution. Code SHA 11b1ae8.
- **Fixes:** each one is post-processed onto the model's own galaxies, using only relations taken from the literature.
- **Sources:** every number comes from a cited paper. Source notes, with quotes and verification status, are in `literature_sources.md`. The papers were read from the local PDFs or arXiv.

Scripts:
- `load.py` caches the HDF5 fields.
- `common.py` holds a numpy copy of SAGE26's SFprescription 1. It reproduces the model's H2 and SFR: median ratio 0.99, 5–95% range 0.95–1.0 for H2.
- `fig_diagnosis.py`, `fig_fixes.py` and `fig_metal_fixes.py` make the figures.

## Figures

| | |
|---|---|
| `fig1_gas_diagnosis.png` | Gas content at z = 0 against xCOLD GASS and xGASS, depletion times, disk sizes, star-forming sequence |
| `fig2_metals_diagnosis.png` | Stellar and gas metallicity–mass relations, where the produced metals sit, gas against stellar metallicity with redshift |
| `fig3_gas_fixes.png` | Four literature-based gas fixes and their effect on H2, HI, t_dep and gas metallicity |
| `fig4_metal_fixes.png` | Yield and gas fixes for stellar and gas metallicity, the z ≈ 2.3 check, and why the two issues are only weakly coupled |

## Issue 2: too much cold gas

**What the comparison should be.** The xCOLD GASS and xGASS medians are for *all* galaxies, with non-detections set to their upper limits. ViSAGE compared them against model *star-forming* galaxies. Helium bookkeeping also differs:
- xCOLD GASS catalogue M_H2 includes helium (Saintonge+17; catalogue column 20). So does Tacconi+18 (36% He).
- SAGE26 `H2gas` is hydrogen only (0.74 × f_mol × ColdGas).

So the model's H2 should be multiplied by 1.36 before comparing.

**Findings (Fig. 1, Fig. 3 "SAGE26 now"):**
- **Molecular gas, main-sequence galaxies, He included:**
  - log⟨M_H2/M*⟩ = −0.65 at 10^10.25 M☉.
  - Saintonge+17 Table 5 main-sequence stacks give −1.05 there.
  - The model is **0.4 dex high**. All galaxies together are 0.1–0.3 dex high below 10^10.5 and agree above.
- **Atomic gas, all galaxies:**
  - Median log M_HI/M* = −0.60 at 10^10.25.
  - xGASS gives −0.97 at 10^10.15 (median) and −1.18 at 10^10.34 (Catinella+18 Table 1 mean).
  - The model is **0.3–0.6 dex high** between 10^9.5 and 10^10.5.
- **Molecular depletion time, He included:**
  - SAGE26 gives log t_dep = 9.44 (2.8 Gyr), flat with mass.
  - Observed:
    - Saintonge+17 main sequence: 8.98 ± 0.27 (0.95 Gyr).
    - Tacconi+18 at z = 0 on the main sequence: 1.1–1.2 Gyr.
    - Leroy+13: 2.2 Gyr.
  - The model converts H2 to stars **2–3× too slowly** compared with xCOLD GASS and Tacconi+18. Against the resolved-disk value of Leroy+13 it is only slightly slow.
- **Disk sizes:**
  - Model star-forming disks have r_e = 1.678 r_s.
  - They sit near the top of the Lange+16 disc distribution below 10^10 M☉. The Lange+16 fit is R_e = 5.141 (M/10^10)^0.274 kpc; the model is 4.3 kpc against 2.9 kpc at 10^9.1.
  - They agree above 10^10.3.
- **Star-forming sequence:** the model is about 0.2 dex above both the Speagle+14 and Catinella+18 main sequences at 10^10–10^10.6, and flatter than both.

**Fixes tested (Fig. 3).** Following Lilly+13 (Eq. 7, μ = ε⁻¹ sSFR), the SFR is held fixed and each galaxy's gas is solved for under each change:

| Change (source) | t_dep(H2+He) | MS log⟨H2/M*⟩ at 10^10.25 | All-galaxy HI/M* at 10^10.25 |
|---|---|---|---|
| SAGE26 now | 2.8 Gyr | −0.65 | −0.60 |
| SFR = M_H2 / t_dep(Tacconi+18 Table 3a) | 0.95 Gyr | **−1.16** (obs −1.05) | −0.80 |
| Disk sizes rescaled to Lange+16 | 2.0–2.5 Gyr | −0.70 | −0.69 |
| Leroy+08 pressure law (their Eq. 9 pressure, Table 6: log P0/k = 4.23, α = 0.80) | unchanged | unchanged | −0.87 |
| Tacconi t_dep + Lange sizes | 0.95 Gyr | −1.16 | **−0.87** (obs −0.97 to −1.18) |

- **The depletion time is the lever for H2.** Matching Tacconi+18 brings main-sequence H2 within about 0.1 dex of xCOLD GASS.
- **HI needs a second change.** Either smaller low-mass disks (Lange+16) or the Leroy+08 pressure law, which puts more of the gas into H2 at a given pressure. Each removes a further 0.1–0.3 dex.
- These match what other models report:
  - Lagos+11: the H2/HI evolution is "primarily due to the evolution in the sizes of galaxies".
  - Somerville+15 and Xie+17: global stellar properties are insensitive to the star-formation law.

**Suggested code changes (not applied; Paper II model frozen):**
1. Replace `SfrEfficiency × H2 / (3 r_s/V_vir)` with SFR = M_H2(1.36) / t_dep, taking t_dep from Tacconi+18 or a constant ~1 Gyr (Saintonge+17). Alternatively, raise `SfrEfficiency` so the z = 0 main-sequence t_dep(H2+He) ≈ 1 Gyr.
2. Move the BR06 constants to Leroy+08 (with Leroy's pressure formula and σ_g = 11 km/s), or to BR06 Eq. 12 (P0/k = 3.5×10⁴, the Shark default). Note that SAGE26 currently uses Eq. 13, the noninteracting subset.
3. Check the low-mass disk-size normalisation against Lange+16. A larger gas disk than stellar disk (χ_gas = 1.7, Popping+14; 0.59 the other way, Somerville+15) would *lower* the pressure and make the HI excess worse unless the star-formation law changes as well.
4. Compare model H2 with He included (×1.36) in all plots.

## Issue 1: massive galaxies too metal-poor

**Findings (Fig. 2, Fig. 4):**
- **Gas-phase metallicity is right where it matters most.**
  - At z = 2.46, model 12+log(O/H) is 8.38, 8.50 and 8.61 at 10^9.5, 10^10 and 10^10.5 M☉.
  - Sanders+21 (MOSDEF, z ≈ 2.3, Table 3) gives 8.36, 8.51 and 8.66.
  - The model uses the Asplund+09 solar values, Z = 0.0134 ↔ 8.69, and assumes a solar abundance pattern.
- **At z = 0 the model gas is 0.1 dex low and too steep.** It reads 8.54 and 8.65 against Sanders+21's z ≈ 0 values of 8.64 and 8.77 at 10^9.5 and 10^10, and overshoots above 10^10.6.
- **Part of the stellar offset comes from weighting.** Gallazzi+05 metallicities are optical light-weighted. Peeples+14 Eq. 7 (from Peeples & Somerville 2013, BC03 B-band) converts them: log Z_mass = 1.08 log Z_light − 0.16. Against mass-weighted Gallazzi:

  | log M* | Model log Z*/0.02 | Gallazzi light | Gallazzi mass-weighted |
  |---|---|---|---|
  | 10.5 | −0.29 | −0.01 | −0.18 |
  | 11.0 | −0.26 | +0.08 | −0.07 |
  | 11.5 | −0.27 | +0.13 | −0.02 |

  - The gap above 10^10.5 shrinks from 0.3–0.4 dex to **0.1–0.25 dex**.
  - The model now sits *above* mass-weighted Gallazzi below 10^10.
  - What remains is mainly the **model relation being too flat** above 10^10.5.
  - Caveat: Peeples' conversion comes from star-forming galaxy histories, so applying it to quiescent galaxies extrapolates it.
- **The yield is at the low end of the literature.**
  - SAGE26 uses Y = 0.025 per unit mass of stars *formed* (Croton+16 definition).
  - Peeples+14 adopt a core-collapse yield of 0.030 per unit mass formed, ranging 0.0214–0.0408 across stellar models.
  - Vincenzo+16 (Chabrier) and Madau & Dickinson 14 give 0.019–0.033 once converted to the same normalisation.
  - Metals enter SAGE26 linearly in Y, so Δlog Z = log(Y/0.025): +0.08 dex for 0.030, and −0.07 to +0.21 across the range.
- **Where the metals end up in massive galaxies (Fig. 2c).** Above 10^10.5, 50–60% of all metals produced sit in the hot halo and about 20–25% in stars. Hirschmann+16 diagnose the same pattern in GAEA: "more metals are ejected … to the hot gas phase at late times … the metallicity of the cooled gas is typically lower … keeps the metallicity of the stars formed later low".
- **Fig. 2d.** Today's massive galaxies have Z* ≈ −0.27, similar to the gas of 10^9.5–10^10 galaxies at z ≈ 1–2.5. Their progenitors at 10^10.5–10^11 had gas at about −0.2 to −0.1. A sizeable part of their stars must therefore come from lower-mass, lower-Z progenitors or from accreted satellites. The model doesn't track accreted (ex-situ) stars separately, so it can't yet tell which.

**How the two issues connect (Fig. 3d, Fig. 4d).** In the Lilly+13 equilibrium, Z_eq = Z0 + y/[1 + λ/(1−R) + μ] (Eq. 27a), where μ is the gas-to-star mass ratio and λ the mass loading:
- SAGE26's FIRE mass loading gives λ/(1−R) ≈ 2–4 at z = 0 and 3–13 at z = 2.5. That dominates the denominator over μ ≈ 0.2–2.
- So cutting the gas excess raises gas metallicity by only **0.05–0.1 dex at low mass and below 0.02 dex above 10^10.5**.
- Propagated through each galaxy's star-formation history, the Tacconi fix raises Z* by **+0.02–0.05 dex**.
- The issues share a mechanism, but fixing the gas will not fix the massive-galaxy metallicities. The gas fix does close most of the 0.1 dex low-mass offset in the z = 0 gas metallicity relation (Fig. 4b).

**Suggested changes (not applied):**
1. Raise `Yield` to 0.030 (Peeples+14 fiducial): +0.08 dex everywhere. Together with the gas fix and mass-weighting, this leaves 0.03–0.17 dex at the massive end.
2. Massive end: the literature fix for the flat, sub-solar high-mass relation is to reduce late-time dilution of enriched gas in the hot or ejected reservoir:
   - Hirschmann+16 report that their FIRE-parametrised and zDEP models reach supersolar Z*, while their Hen13 model, with strong late ejection, does not.
   - Henriques+20 removed the deficit with two changes:
     - 30% of SN-II and SN-Ia ejecta go directly to the CGM, which they tuned to the gas metallicity normalisation;
     - the SN-Ia fraction rises from 0.028 to 0.04.
   - These need a model rerun to test and can't be post-processed reliably.
3. Diagnostic first: track the metal mass of accreted stars (merged satellites) separately. With the SFH fix (#3) in place, that would show whether the massive-end flatness is ex-situ dilution or in-situ star formation from diluted gas.

## Caveats
- The fixed-SFR assumption is the gas-regulator argument (Lilly+13, Bouché+10). In a rerun, a shorter t_dep would also move SFR and feedback somewhat.
- The metallicity response is bracketed between pristine inflow (Z0 = 0) and the model's current enriched inflow held fixed (band edges in Fig. 3d).
- The Tacconi+18 size term (E = 0.11 ± 0.12, consistent with zero) is ignored.
- Lange+16 assumes H0 = 70 and SAGE26 0.6774: a ~3% effect, ignored.
- Whether the Lange+16 relation is in disc or total stellar mass, and the band of R_e, are unverified.
- Gallazzi+05's Z⊙ = 0.02 comes via Yates+13 (private communication with Gallazzi). Peeples' Eq. 7 is applied in Gallazzi's own Z⊙ units.
- About 6% (median) of produced metals are missing from every reservoir in the budget (Fig. 2c). This bookkeeping gap is not yet explained.

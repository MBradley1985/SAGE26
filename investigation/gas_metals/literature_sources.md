# Literature facts for the SAGE26 metallicity / cold-gas investigation

How each item was checked: VERIFIED means the text was read in the paper's PDF. "Local" means the PDF was already on disk (path given). Otherwise the arXiv PDF was downloaded to scratchpad/papers/ and read with pdftotext.
Journal refs come from the arXiv API `journal_ref` field or from the PDF header. Where neither gave one, the ref is marked "jref not checked".
Lines marked DERIVED are my own arithmetic from the printed numbers, not statements made in the paper.

---
## A. H2–pressure law

**Blitz & Rosolowsky 2006, ApJ 650, 933** (astro-ph/0605035). LOCAL: `/Users/mbradley/Documents/PhD/SAGE26/Blitz and Rosolowsky - 2006 - The Role of Pressure in GMC Formation II The Hsu.pdf`. VERIFIED.
- Eq. 1 (from BR04): P_ext = (2G)^0.5 Σ_g v_g [ρ*^0.5 + (π/4 ρ_g)^0.5]. The paper assumes "the gas scale height is much less than the stellar scale height".
- They use Σ* = 2ρ* h* and h* = (v*²/2πGρ*)^0.5, and neglect ρ_g. Eq. 2: **P_ext = 0.84 (G Σ*)^0.5 Σ_g v_g / (h*)^0.5**. The paper says this is "accurate to 10% for Σ* > 20 M⊙ pc⁻²".
- Eq. 5 (numerical form): P_ext/k = 272 cm⁻³ K (Σ_g/M⊙pc⁻²)(Σ*/M⊙pc⁻²)^0.5 (v_g/km s⁻¹)(h*/pc)^-0.5.
- Assumptions: "a constant gas velocity dispersion of v_g = 8 km s⁻¹". Eq. 9 gives h* from the disk scale length (Kregel+02 fit): log h* = (−0.23±0.05) + (0.8±0.1) log R* (both in pc).
- The fit form is Eq. 11, R_mol = (P_ext/P0)^α.
  - Eq. 12, all 14 galaxies: R_mol = [ (P_ext/k) / ((3.5±0.6)×10⁴) ]^(0.92±0.07).
  - Eq. 13, noninteracting galaxies only: P0/k = (4.3±0.6)×10⁴ cm⁻³ K, α = 0.92±0.10.
  - Table 2, "Combined data" row: α = 0.94±0.02, P0/k = 4.5×10⁴.
- Units and conversions: Σ_HI uses a mean particle mass of 1.36 m_H, so helium is included. Σ_H2 uses X_CO = 2×10²⁰ cm⁻²(K km s⁻¹)⁻¹ and 1.36 m_H. They adopt M_K/L_K = 0.5.

**Leroy et al. 2008, AJ 136, 2782** (0810.2556). VERIFIED.
- Their Eq. 9 is the Elmegreen (1989) form: P_h ≈ (π/2) G Σ_gas (Σ_gas + (σ_g/σ*,z) Σ*).
- Assumptions: σ_gas = 11 km s⁻¹. They take l*/h* = 7.3±2.2 (Kregel+02), with σ*,z = sqrt(2πG l*/7.3) Σ*^0.5 (Appendix Eq. B3).
- Table 6, fits of R_mol = (P_h/P0)^α over R_mol = 0.1–10, giving log10 P0/k_B (cm⁻³ K) and α:

  | Sample | log10 P0/k_B | α |
  |---|---|---|
  | spiral, CO/HI | 4.19 | 0.73 |
  | spiral, SFR/HI | 4.30 | 0.79 |
  | **spiral, combined** | **4.23** | **0.80** |
  | dwarf, SFE/HI | 4.51 | 1.05 |
  | BR06 (as listed by Leroy) | 4.54±0.07 | 0.92±0.07 |

- At the HI–H2 transition in spirals they find log10 P_h/k_B ≈ 4.36. Table 5 gives P_h/k_B ≈ 2.3×10⁴ there.
- Σ_HI and Σ_H2 "include a factor of 1.36 to account for helium".

**Uses in other SAMs (VERIFIED):**
- Lagos+11 GALFORM (1105.2294) adopts α = 0.92±0.07, log P0/k = 4.54±0.07, ν_SF = 5.25±2.5×10⁻¹⁰ yr⁻¹, σ_g = 10 km/s and σ* = sqrt(πG h* Σ*). For h* it uses "R_eff/h* ≈ 7.3 ± 1.2 (Kregel et al. 2002)".
- Shark (Lagos+18, MNRAS 481, 3573; LOCAL `/Users/mbradley/Documents/PhD/Papers/sty2440.pdf`) defaults: P0 = 34,673 K cm⁻³, α_P = 0.92, σ_gas = 10 km/s, ν_SF = 1 Gyr⁻¹ (allowed range 0.25–1.25).

## B. Molecular depletion times

- **Bigiel+08, AJ 136, 2846** (0810.2541). VERIFIED.
  - Fit: N = 1.0±0.2, "average molecular gas depletion time is ∼2·10⁹ years". Their Fig. 5 text: "mean H2 gas depletion time (including helium) is 2.0·10⁹ yrs with an RMS scatter of 0.8·10⁹ yrs".
  - Caution: the quoted surface densities "do not include any contribution from helium… multiply… by ∼1.36". X_CO = 2.0×10²⁰.
- **Leroy+08**. VERIFIED. SFE(H2) = 5.25±2.5×10⁻¹⁰ yr⁻¹, "equivalent to an H2 depletion time of 1.9×10⁹ yr". Helium is included.
- **Leroy+13** (1301.2328, AJ; jref not checked). VERIFIED. Uses a fixed Milky Way α_CO = 4.35 with "a factor of 1.36 to account for helium". Result: τ_dep^mol ≈ 2.2 Gyr with 0.3 dex 1σ scatter, and N = 1 ± 0.15.
- **Saintonge+11b, MNRAS 415, 61** (1104.0019). VERIFIED.
  - Mean t_dep(H2) is 1 Gyr. It "increases by a factor of 6 from ∼0.5 Gyr for M*∼10¹⁰ to ∼3 Gyr for a few ×10¹¹".
  - Fit as printed: log t_dep(H2) = (0.36±0.07)(log M* − 10.70) + (9.03±0.99).
  - t_dep(HI) ≈ 3 Gyr, independent of mass.
  - **α_CO = 3.2, "does not account for the presence of Helium."** On this common scale the HERACLES (Leroy+08) mean t_dep is 0.9 Gyr in the COLD GASS mass range, against 1.05±0.74 Gyr for COLD GASS.
- **Saintonge+17, ApJS 233, 22** (1710.02157). VERIFIED.
  - α_CO is the metallicity-dependent function of Accurso+17. Catalogue column 20, log M_H2, is the "Total molecular gas mass, **including the Helium contribution**".
  - Main-sequence galaxies: ⟨log t_dep(H2)⟩ = **8.98 ± 0.27** (yr), so about 1 Gyr. ⟨log t_dep(HI)⟩ = 9.65 ± 0.44.
  - f_H2 = M_H2/M* is "on average 10% in galaxies with M* < 5×10¹⁰ M⊙, and then plummets rapidly to 1% as M* increases to 10¹¹".
  - Table 5, main-sequence-only stacking, log⟨f_H2⟩ by log M*:

    | log M* | 10.07 | 10.25 | 10.46 | 10.68 | 10.82 | 11.10 |
    |---|---|---|---|---|---|---|
    | log⟨f_H2⟩ | −0.89 | −1.05 | −0.99 | −1.10 | −1.26 | −1.41 |

  - Same table, all galaxies, stacking: at log M* = 10.24 it is −1.08; at 10.46 it is −1.05; at 10.67 it is −1.34.
  - Molecular-to-atomic ratio: "mean molecular-to-atomic ratio for massive galaxies is ⟨R_mol⟩∼0.35, it drops to only 0.1 for the lowest mass".
  - Main sequence: plotted, with |Δ(MS)| < 0.4 used as the cut. I did not find an explicit MS equation in the S17 text (UNVERIFIED which MS they used).
- **Catinella+18, MNRAS 476, 875** (1802.02373; jref not checked). VERIFIED.
  - The αCO average of 4.44 is "taking into account the contribution of Helium". However, **"we use molecular gas masses without Helium contribution, M_H2, to compute molecular-to-atomic … ratios"**. Total gas is M_gas = 1.3(M_HI + M_H2).
  - Their SFMS (Eq. 2): log sSFR_MS = −0.344(log M* − 9) − 9.822, with σ_MS = 0.088(log M* − 9) + 0.188.
  - Table 1, **all xGASS galaxies** (not SF-only), weighted ⟨log M_HI/M*⟩:

    | log M* | 10.07 | 10.34 | 10.65 | 10.95 | 11.20 |
    |---|---|---|---|---|---|
    | ⟨log M_HI/M*⟩ | −0.869 | −1.175 | −1.231 | −1.475 | −1.589 |

  - I did NOT find an SF-only HI table, so the "0.2–0.3 for SF galaxies" figure is UNVERIFIED from tables. They do state that for SFMS galaxies the gas-fraction relations steepen.
  - Table 3, ⟨log M_H2/M_HI⟩: −0.704 at log M* = 10.22, −0.662 at 10.59, −0.530 at 10.91. The text says the median R_mol "slowly increases with stellar mass (from 9% to 28%)".
  - Table 2, total gas incl. He, ⟨log M_gas/M*⟩: −0.817 at log M* = 10.34.
- **Tacconi+18, ApJ 853, 179** (1702.01140; arXiv jref absent). VERIFIED.
  - Mmolgas includes "a 36% mass fraction of helium" plus a CO-dark correction.
  - MS is Speagle+14 (their Eq. 1): log sSFR(MS,z,M*)[Gyr⁻¹] = (−0.16 − 0.026 t_c)(log M* + 0.025) − (6.51 − 0.11 t_c) + 9, with log t_c = 1.143 − 1.026 log(1+z) − 0.599 log²(1+z) + 0.528 log³(1+z). IMF is Chabrier.
  - Table 3a "best": log t_dep(Gyr) = A + B log(1+z) + C log δMS + D(log M* − 10.7) + E log(Re/Re0), with **A = +0.09±0.05, B = −0.62±0.13, C = −0.44±0.04, D = +0.09±0.05, E = +0.11±0.12**.
  - Table 3b "best", μ_gas = Mmol/M*:
    - β = 2 form: A = +0.12±0.15, B = −3.62±0.4, F = +0.66±0.1, C = +0.53±0.03, D = −0.35±0.03.
    - β = 0 form: A = −1.19±0.04, B = +2.49±0.2, C = +0.52, D = −0.36.
  - Abstract: t_dep ∝ (1+z)^−0.6 (δMS)^−0.44; μ ∝ (1+z)^2.5 (δMS)^0.52 (M*)^−0.36.
  - α_CO is metallicity-dependent: 4.36 × R_J1 × χ(Z), where χ is the Genzel+12/Bolatto-type metallicity factor.
  - DERIVED (z = 0, δMS = 1): t_dep = 1.23 Gyr at log M* = 10.7 and ≈1.12 Gyr at 10.25 (He included). μ_gas (β=0) = 0.065 at 10.7 and ≈0.094 at 10.25; μ_gas (β=2) = 0.035 at 10.7 and ≈0.050 at 10.25.
- **Helium bookkeeping caution for the model comparison:**
  - xCOLD GASS catalogue M_H2 includes He. Tacconi includes He.
  - xGASS R_mol excludes He. S11 α_CO = 3.2 excludes He. Bigiel's surface densities exclude He, but its 2 Gyr t_dep includes He.
  - The model's M_H2 convention must be matched to whichever survey it is compared against.

## C. Gas regulator

**Lilly et al. 2013, ApJ 772, 119** (1303.5059; jref not checked). VERIFIED, read from PDF pp. 10–13 and 21–22.
- Definitions:
  - Eq. 4: SFR = ε m_gas, and Eq. 5: τ_gas = ε⁻¹. Here m_gas is the *total* reservoir gas, "not just the molecular gas".
  - Eq. 6: μ = m_gas/m_star.
  - Eq. 1: rsSFR = (1−R)·sSFR (R is the instantaneous return fraction, taken as 0.4).
  - Eq. 8: Ψ = λ·SFR (λ is the mass-loading factor).
  - Z0 is the metallicity of the inflow.
  - y is defined as "the mass of metals returned to the ISM per unit mass that is locked up into long-lived stars, i.e. (1−R) times the mass of stars formed".
- Gas equations:
  - Eq. 7: **μ = ε⁻¹·sSFR**. For SFR = ε m_gas^κ (Eq. 7b): μ = (ε⁻¹ sSFR)^(1/κ) m_star^−((κ−1)/κ).
  - Eq. 9: Φ = (1 − R + λ)·SFR + dm_gas/dt.
  - Eq. 11: Φ = ((1−R)(1+μ) + λ + ε⁻¹ dlnμ/dt)·SFR.
  - Eq. 12: f_star = 1/[1 + (1−R)⁻¹λ + ε⁻¹(sSFR + (1−R)⁻¹ dlnμ/dt)].
- Metallicity:
  - Metal equation: dm_Z/dt = y(1−R)SFR − Z(1−R+λ)SFR + ΦZ0.
  - Eq. 25: Z_eq = Z0 + y(1−R) / [(1−R) + λ + ε⁻¹(rsSFR + dlnμ/dt)].
  - Eq. 26: **Z_eq = Z0 + y / [1 + λ(1−R)⁻¹ + ε⁻¹(sSFR + (1−R)⁻¹ dlnμ/dt)]**.
  - Eq. 27: Z_eq = Z0 + y / [1 + λ(1−R)⁻¹ + μ + (1−R)⁻¹ε⁻¹ dlnμ/dt].
  - Eq. 29: Z_eq = Z0 + f_star·y.
  - "Ideal" forms with dlnμ/dt = 0: Eq. 26a, Z_eq = Z0 + y/[1 + λ(1−R)⁻¹ + ε⁻¹ sSFR]; Eq. 27a, Z_eq = Z0 + y/[1 + λ(1−R)⁻¹ + μ].
- Fit results: y ≈ "+9.0 in units of 12+log(O/H), i.e. about y = 0.016 as a mass ratio". The text gives λ10 = 0.2–0.3, with λ ∝ m^a, a ≈ −0.9.

**Davé, Finlator & Oppenheimer 2012, MNRAS 421, 98** (1108.0426; jref not checked). VERIFIED.
- Eq. 7: f_gas ≡ M_gas/(M_gas + M*) = 1/[1 + (t_dep sSFR)⁻¹].
- Eq. 9: **Z_ISM = y SFR/Ṁ_in = (y/(1+η)) · 1/(1−α_Z)**, where α_Z ≡ Z_in/Z_ISM.
- Eq. 12: SFR = ζ Ṁ_grav / [(1+η)(1−α_Z)].
- Their sample model uses t_dep = 0.4 t_H (M*/10¹⁰)^−0.3 and 18% instantaneous recycling for Chabrier.

**Bouché+10, ApJ 718, 1001** (0912.1858). VERIFIED.
- Eq. 3: Ṁ_gas = Ṁ_gas,in − (1−R)Ṁ* − Ṁ_gas,out.
- Eq. 7: SFR = ε_sfr M_gas/t_dyn, with t_dyn = R_1/2/V_c and ε_sfr = 0.02.
- They use "R = 0.52 … recycled gas fraction for a Chabrier IMF (BC03)".
- **No metallicity equilibrium relation in this paper.**

## D. Stellar metallicity normalisation and weighting

- **Gallazzi+05, MNRAS 362, 41** (astro-ph/0506539). VERIFIED.
  - Metallicity is a single fixed value per model, "which we interpret as the '(optical) light-weighted' metallicity". Ages are r-band light-weighted. Model library covers 0.02–2.5 Z⊙ (BC03).
  - The text gives no numerical Z⊙. **Yates+13 (1305.7231) states G05 used "Z⊙ = 0.02 assumed in the stellar population synthesis models… (A. Gallazzi, priv. comm.)"** (VERIFIED in Yates+13).
  - Table 2, median log(Z/Z⊙) [P16, P84]:

    | log M* | 10.31 | 10.51 | 10.72 | 10.91 | 11.11 | 11.31 |
    |---|---|---|---|---|---|---|
    | P50 | −0.11 | −0.01 | 0.04 | 0.07 | 0.10 | 0.12 |
    | [P16, P84] | [−0.65, 0.17] | [−0.41, 0.20] | [−0.24, 0.22] | [−0.14, 0.24] | [−0.09, 0.25] | [−0.06, 0.26] |

  - The sample includes passive galaxies. Henriques+20 notes passive galaxies "tend to have higher Z* than star-forming galaxies of the same mass".
- **Asplund+09, ARA&A 47, 481** (0909.0948). VERIFIED.
  - Present-day photosphere: X = 0.7381, Y = 0.2485, **Z = 0.0134**. Bulk (protosolar): 0.7154, 0.2703, **0.0142**.
  - Quote: "no longer the canonical 2% recommended by Anders & Grevesse (1989)".
  - Renormalising log Z/Z⊙ from 0.02 to 0.0142 shifts values up by log(0.02/0.0142) = +0.149 dex; to 0.0134 the shift is +0.174 dex (DERIVED).
- Henriques+20 (LOCAL `/Users/mbradley/Documents/PhD/Papers/stz3233.pdf`) "normalized … to the protosolar … Z = 0.0142". VERIFIED.
- Light- vs mass-weighted offsets. Henriques+20 compares with Zahid+17, which gives luminosity- and mass-weighted points. Henriques+20 says its model normalisation is "a factor of ∼2 above the observed mass-weighted relation" at low mass. VERIFIED.
  - The size of any light- vs mass-weighted offset in Trager & Somerville 09, De Lucia+14, Gallazzi+14/21 or Fontanot+21 is **UNVERIFIED** (not extracted).
  - Hirschmann+16 states that its *ages* are "luminosity weighted (r-band, SDSS)". Whether its metallicities are light-weighted is UNVERIFIED.

## E. SAM fixes for metal-poor massive galaxies

- **Hirschmann+16, MNRAS 461, 1760** (GAEA). LOCAL `/Users/mbradley/Documents/PhD/SAGE26/Hirschmann et al. - 2016 - Galaxy assembly, stellar feedback and metal enrich.pdf`. VERIFIED, Sec. 7.1.
  - FIRE-parametrised (Muratov) ejection and zDEP models: stellar Z "strongly increasing with decreasing redshift, approaching supersolar metallicity for the most massive galaxies, in good agreement".
  - The Hen13 model (stronger late reheating/ejection) "predicts a too strongly delayed metal enrichment so that even the most massive galaxies are always below solar levels".
  - Their explanation: "Since the metal flows are assumed to follow the gas flows, more metals are ejected from the cold … phase … to the hot gas phase at late times… Due to mixing with less metal enriched gas in the hot gas phase (or in the ejected phase), the metallicity of the cooled gas is typically lower… keeps the metallicity of the stars formed later low."
  - Relevance: late-time ejection and reincorporation dilution in massive galaxies directly lowers their Z*.
- **Henriques+20 (L-GALAXIES 2020), MNRAS 491, 5795.** LOCAL stz3233.pdf. VERIFIED.
  - They increased the SN-Ia progenitor fraction from A = 0.028 to 0.04.
  - "30 per cent of the ejecta from both SNe-II and SNe-Ia [are] directly dumped into the CGM … metal-rich wind", tuned to the gas-phase MZR normalisation; "10 to 50 per cent" gives similar results.
  - Result: "good correspondence" with SDSS stellar MZR. "The large metallicity deficit previously noted by Hirschmann et al. (2016) is not present."
- **Yates+13, MNRAS 435, 3500** (1305.7231; jref not checked). VERIFIED.
  - Introduced delayed enrichment (AGB, SN-II, SN-Ia), metallicity-dependent yields and an optional metal-rich wind dumping SN-II material into hot gas.
  - Below 10^10.5 M⊙ the new stellar MZR is about 0.1 dex higher. They conclude "further improvement … is still required in order to better match the observed total stellar metallicities".
  - It does not specifically diagnose high-mass Z* deficits.
- **De Lucia+14, MNRAS 445, 970** (1407.7867; jref not checked). VERIFIED.
  - Its older IRA scheme used Y = 0.03 and R = 0.43 for Chabrier, with ejecta going to hot gas in haloes below 3×10¹⁰ M⊙.
  - The paper is Milky-Way focused. It shows that non-IRA lowers the effective R (they mimic it with R = 0.15, "H+He restored by SNII only") and Y ≈ 0.02.
  - It does not address the massive-galaxy MZR.
- **Shark (Lagos+18).** VERIFIED. Uses IRA with p = 0.029, R = 0.46 (Chabrier, Conroy+09 models). Shark galaxies have "slightly lower stellar metallicities than … observations… within the uncertainties".
- Collacchioni+18 (1808.09089) uses AG89 solar Z = 0.02 (VERIFIED). Its conclusions on the MZR are UNVERIFIED (not read).
- Fontanot+21 (variable IMF): **UNVERIFIED** (paper not located/read).

## F. Yields and return fraction (Chabrier)

- **Vincenzo+16, MNRAS 455, 4183** (1503.08300). VERIFIED.
  - y_Z is the "total mass of metals that a stellar population releases … normalized to the mass locked up into low-mass (long-lived) stars and stellar remnants". m_up = 100.
  - Table 2 (Romano+10 yields), Chabrier:

    | Z | R | y_O | y_Z |
    |---|---|---|---|
    | 10⁻³ | 0.438 | 0.039 | 0.060 |
    | 0.01 | 0.451 | 0.038 | 0.060 |
    | 0.02 | 0.455 | 0.037 | 0.065 |

  - Table 3 (Nomoto+13 yields), Chabrier:

    | Z | R | y_O | y_Z |
    |---|---|---|---|
    | 0.01 | 0.463 | 0.032 | 0.051 |
    | 0.02 | 0.466 | 0.030 | 0.049 |

  - Salpeter (Romano yields, Z = 0.02): R = 0.298, y_Z = 0.031.
- **Madau & Dickinson 2014, ARA&A 52, 415** (1403.0007). VERIFIED.
  - y is "per unit mass locked into stars". Salpeter (0.1–100, m0 = 1) gives **R = 0.27**; Chabrier gives **R = 0.41**.
  - Maeder92 yields to m_BH = 40: y = 0.016 (Salpeter), 0.032 (Chabrier). To m_BH = 60: y = 0.023 (R = 0.29) and 0.048 (R = 0.44).
  - Hirschi+05 yields: y = 0.019 / 0.038. Chieffi & Limongi04: 0.015 / 0.030.
- SAGE (Croton+16) defines Y per unit mass *formed*: "a yield Y of metals are returned for each solar mass of stars formed". This is a different normalisation from Vincenzo and MD14 (per mass locked).
  - DERIVED: per-formed equivalent = y_Z(1−R). Vincenzo Chabrier/Romano Z = 0.01 gives 0.060 × 0.549 ≈ 0.033. Nomoto Z = 0.02 gives 0.049 × 0.534 ≈ 0.026. MD14 Chabrier (40 M⊙ cut) gives 0.032 × 0.59 ≈ 0.019.
- Somerville+15 uses y = 1.6 Z⊙ and χ_gas = 0.59 (Table 1). VERIFIED.

## G. SAMs with too much gas or H2

- **Somerville+15** (1503.00755) and **Popping+14** (1308.6764). VERIFIED.
  - Gas disc larger than stellar disc: Popping r_star = r_gas/χ_gas, "χ_gas = 1.7 fixed to match stellar scale lengths at z = 0". Somerville r* = χ_gas r_gas, with χ_gas = 0.59.
  - Popping abstract: both the pressure and metallicity recipes "reproduce the gas fractions and gas-to-stellar mass ratios of HI and H2 in local galaxies".
  - Somerville+15: stellar properties are "remarkably insensitive" to HI/H2 partitioning. Massive galaxies at high z depend on t_dep(H2).
- **Henriques+20.** VERIFIED. HI fractions match xGASS, but "H2 fractions in the model are significantly higher than those observed, particularly at high-mass… above 10^10.5". They suggest AGN radiation as a possible H2-destruction mechanism (not implemented).
- **Xie+17, MNRAS 469, 968** (1611.09372). **NOTE: this is GAEA (De Lucia group), not Dark Sage.**
  - Updated disc sizes (Guo+11-type angular momentum tracking of gas and stellar discs separately). Sizes are larger above 10^10.5, with "no significant differences for other galaxy properties".
  - "Modifying the star formation law does not affect significantly the global properties."
- **Lagos+11** (1105.2294; MNRAS 418, 1649 per brief, jref not checked). VERIFIED. The BR06 version reproduces the HI mass function, CO luminosity function and H2/HI–M* correlations. Evolution of H2/HI is "primarily due to the evolution in the sizes of galaxies".
- **Fu+10** (1004.2325, MNRAS 409, 515). VERIFIED abstract only. Uses ring-resolved discs, KMT08 or pressure-based H2, and Σ_SFR ∝ Σ_H2 (H2-dominated) / ∝ Σ²_gas (HI-dominated).
  - Fu+12/13 (1203.5280) abstract: the MZR evolution "favours models where [gas-to-star conversion time] remains constant at values around 1-2 Gyr".
  - Specific changes and effects: UNVERIFIED beyond the abstracts.
- Dark Sage (Stevens+16, 1605.00647) abstract read; gas-specific fixes UNVERIFIED. Stevens+18 not read (UNVERIFIED). Dark Sage 2024 PASA is local (`/Users/mbradley/Documents/PhD/Papers/dark-sage-next-generation-...pdf`) but was not read in detail; it uses Z⊙ = 0.0142 for O/H.

## H. Sizes

- **Shen+03, MNRAS 343, 978.** VERIFIED.
  - Sizes are Sérsic z-band R50. Stellar-mass fit (Table 1, "Figure 11" row) for late types (n < 2.5): α = 0.14, β = 0.39, γ = 0.10, M0 = 3.98×10¹⁰ M⊙, σ1 = 0.47, σ2 = 0.34.
  - Text: low-mass "R̄ ∝ M^0.14 … σ_lnR = 0.47"; high-mass "R̄ ∝ M^0.39 and σ_lnR = 0.34". Early types: a = 0.56, b = 3.47×10⁻⁵.
  - The functional form is R̄ = γ M^α (1 + M/M0)^(β−α), by analogy with their Eq. 16. **Exact Eq. 18 typography UNVERIFIED.**
  - h = 0.7 is "assumed" (abstract, for luminosity). The h used for kpc in the mass fit is UNVERIFIED.
- **Lange+16** (1607.01096, MNRAS 462, 1470 per brief; jref not checked). VERIFIED.
  - GAMA bulge+disc decomposition with H0 = 70. Fit form R_e = a (M*/10¹⁰)^b.
  - Combined z = 0 disc relation: **R_e = 5.141 (M*/10¹⁰ M⊙)^0.274 kpc**. Spheroids: 2.063 (M*/10¹⁰)^0.263.
  - R_e is the half-light radius of the component, measured with GALFIT. The band of the fitted R_e is UNVERIFIED (r-band magnitudes from GALFIT are mentioned).

## I–L. Additional items

- I (Tacconi+18): see B.
- L: see B. xCOLD GASS M_H2 includes He. xGASS R_mol uses M_H2 without He. Catinella+18 uses an α_CO average of 4.44 (Galactic 4.35), both including He.
- K (**Peeples+14**, 1310.2253; arXiv jref prints "ApJ, 768, 54", which looks like a typo for ApJ 786, 54). VERIFIED.
  - A "surprisingly constant 20–25% mass fraction of produced metals remain in galaxies' stars, interstellar gas and interstellar dust". The fraction retained is "∼20%" for 9.5 ≲ log M* ≲ 11.5.
  - "Only a few percent … in interstellar dust, ∼5–10% … in interstellar gas". Massive galaxies have more metals trapped in stars.
  - "∼40% of metals produced by M*∼10¹⁰ M⊙ galaxies can be easily accounted for out to 150 kpc" (CGM included).
  - Uses Chabrier, Z⊙ = 0.0153 (Caffau+11) and M_g = 1.366 M_HI. Fit: log F_g = log(M_g/M*) = −0.48 log M* + 4.39.
- J (Erb+06, ApJ 644, 813; Maiolino+08, A&A 488, 463; Mannucci+10; Sanders+21; Zahid+14): PDFs downloaded to scratchpad/papers but **not read: UNVERIFIED**.

## Added during figure work (checked in the paper text)
- **Sanders+21 (MOSDEF; arXiv 2009.07292)**, Table 3 and Eq. 7. At z ~ 2.3: 12+log(O/H) = 0.30 (±0.02) × log(M*/10^10) + 8.51 (±0.02), fit over 9.0–10.5. At z ~ 3.3: 0.29 m10 + 8.41. z ~ 0 (DIG-corrected; Eq. 8, Curti form): Z0 = 8.82, log M_TO = 10.16, γ = 0.28, Δ = 3.43. Chabrier IMF.
- **Peeples+14 Eq. 7** (from Peeples & Somerville 2013): log(Z*,mass/Z⊙) = 1.08 log(Z*,light/Z⊙) − 0.16, B-band weighted, BC03. Core-collapse yield y_z,ii = 0.030 per unit mass of stars formed (range 0.0214–0.0408).
- **Shen+03 Eq. 18:** R̄(kpc) = γ (M/M⊙)^α (1 + M/M0)^(β−α). Confirmed from the paper text.

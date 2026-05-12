# Accelerated by Dark Matter: A High-Redshift Pathway to Efficient Galaxy-Scale Star Formation

**Author:** Michael Boylan-Kolchin  
**Affiliation:** Department of Astronomy, The University of Texas at Austin, 2515 Speedway, Stop C1400, Austin, TX 78712-1205, USA  
**Journal:** Monthly Notices of the Royal Astronomical Society (MNRAS)  
**Volume/Pages:** 538, 3210–3218 (2025)  
**DOI:** https://doi.org/10.1093/mnras/staf471  
**Received:** 2025 March 18 | **Accepted:** 2025 March 20 | **Published:** 2025 March 22 (Advance Access)  
**Original Submission:** 2024 July 15

---

## Abstract (Summary)

In the local Universe, star formation is typically **inefficient** — both globally and when measured as the fraction of gas converted into stars per local free-fall time. An important exception occurs in regions of **high gravitational accelerations** `g`, or equivalently high surface densities `Σ = g/(πG)`, where stellar feedback cannot overcome the self-gravity of dense gas clouds.

This paper explores whether **dark matter** can play an analogous role in providing the requisite accelerations at the scale of **entire galaxies** in the early cosmos (high redshift).

**Key insight:** Characteristic accelerations in dark matter haloes scale as `(1 + z)²` at fixed halo mass. This is sufficient to make dark matter the source of intense accelerations that could induce efficient star formation on galactic scales at **cosmic dawn** in sufficiently massive haloes.

**Key scaling:** The halo mass characterising this regime scales as `(1 + z)^−6` and corresponds to a relatively constant comoving number density of `n(>M_vir) ≈ 10^−4 Mpc^−3` at `z ≥ 8`.

**Predictions for rarer haloes:**
- Stellar masses of `M_★ ~ 10^9 M_☉` can form in regions ending up with sizes of `O(100 pc)`
- Over timescales of ~40 Myr at `z ≈ 12–14`
- These numbers compare well to the brightest JWST-observed galaxies at that epoch

**Conclusion:** Dark matter and standard cosmological evolution may be crucial for explaining the surprisingly high levels of star formation in the early Universe revealed by JWST.

---

## Keywords

`galaxies: formation` · `galaxies: high-redshift` · `dark matter` · `cosmology: theory`

---

## 1. Introduction

### Background: Star Formation Efficiency

- Star formation is regulated by **stellar feedback**: young massive stars emit prodigious UV radiation, inhibiting further star formation
- The star formation efficiency per free-fall time, `ε_ff` (fraction of gas converted to stars per free-fall time), is typically **< 2%**, even in molecular clouds
  - References: Kennicutt 1998; Murray, Quataert & Thompson 2010; Krumholz, McKee & Bland-Hawthorn 2019; Evans, Kim & Ostriker 2022; Hu et al. 2022

### The High-Acceleration Exception

- In regions of **high gravitational acceleration**, momentum injection from massive stars (`ṗ/m`) is insufficient to overcome gravity
- In this regime, star formation becomes efficient: `M_★ = ε M_gas`, with `ε ~ O(1)`
- **Critical acceleration:** `g_crit ≈ ṗ/m ≈ 5 × 10^−10 m s^−2`
  - Equivalently, critical surface density: `Σ_crit ≈ 1000 M_☉ pc^−2`
  - Below `g_crit`: stellar feedback removes most star-forming gas (inefficient)
  - Above `g_crit`: gravity overcomes feedback (efficient)
- References for this threshold: Fall, Krumholz & Matzner 2010; Colín et al. 2013; Geen, Soler & Hennebelle 2017; Kim, Kim & Ostriker 2018; Kruijssen et al. 2019; Grudić et al. 2020; Polak et al. 2024

### Dark Matter in the Local Universe (z = 0)

- At `z = 0`, dark matter **cannot** provide sufficiently high accelerations:
  - Milky Way halo edge: `g/G ≈ 14 M_☉ pc^−2`
  - Milky Way halo centre: `~400 M_☉ pc^−2` (still below `g_crit`)
  - Even galaxy-cluster-mass haloes (`M_vir ~ 10^15 M_☉`): `g_max/G ≈ 1500 M_☉ pc^−2`

### The Key Physical Insight: Redshift Scaling

At **fixed halo mass**, the virial radius scales inversely with `(1 + z)`:

> `R_vir ∝ (1 + z)^−1`

Therefore, the virial acceleration scales as:

> **`g_vir ∝ (1 + z)²`**

This means haloes of the same mass experience **~100× higher accelerations** at `z ≈ 10` compared to `z = 0`. Dark matter can therefore provide the high accelerations needed for efficient star formation in the early Universe.

### Observational Motivation: JWST

- JWST has revealed **unexpectedly active** galaxy and black hole growth at cosmic dawn
- References: Eisenstein et al. 2023; Greene et al. 2023; Akins et al. 2024; Donnan et al. 2024; Dressler et al. 2024; Adamo et al. 2024b (review)
- Various models have attempted to either challenge or explain these observations:
  - Challenging: Boylan-Kolchin 2023; Keller et al. 2023; Lovell et al. 2023; Mason, Trenti & Treu 2023; Shen et al. 2023
  - Explaining: Dekel et al. 2023; Ferrara, Pallottini & Dayal 2023; Mirocha & Furlanetto 2023; McGaugh et al. 2024; Nusser 2024; Rennehan 2024

### Paper Goal

To demonstrate that efficient star formation on large scales — well beyond the mass scales of giant molecular clouds — is a **simple but unavoidable consequence** of the evolution of dark matter densities in an expanding Universe.

---

## Cosmological Parameters Assumed

Standard ΛCDM cosmology (Planck Collaboration VI 2020):

| Parameter | Value |
|-----------|-------|
| H₀ | 67.32 km s⁻¹ Mpc⁻¹ |
| Ω_m | 0.3158 |
| Ω_Λ | 1 − Ω_m |
| n_s | 0.96605 |
| σ₈ | 0.8120 |
| f_b ≡ Ω_b/Ω_m | 0.156 |
| Σ_crit | 1000 M_☉ pc⁻² |
| g_crit/G | 3100 M_☉ pc⁻² |

> Note: The precise value of `g_crit` does not affect the qualitative picture, as the `ε–g` relation increases quickly for `g < g_crit` and saturates for `g > g_crit` (Fall et al. 2010; Grudić et al. 2020). Quantitative predictions are sensitive to this value.

---

## 2. Background: Halo Acceleration Profiles

### Virial Definitions

The virial radius is defined via:

```
M_vir = (4π/3) R_vir(z)³ Δ(z) ρ_m(z)
```

where `Δ_vir(z) ≡ Δ(z) Ω_m(z)` is the overdensity from the spherical top-hat collapse model (Bryan & Norman 1998).

The **virial acceleration** is:

```
g_vir ≡ G M_vir / R_vir²
```

### NFW Profile Acceleration

For a Navarro, Frenk & White (NFW; 1996, 1997) profile, the acceleration at radius `r̃ ≡ r/R_vir` is:

```
g(r) = g_vir · [μ(c) / μ(c r̃)] / r̃²
```

where `μ(x) ≡ ln(1 + x) − x/(1 + x)` and `c` is the halo concentration parameter.

**Maximum (central) acceleration** as `r → 0` (since `ρ ∝ r^−1` gives `M(<r) ∝ r²`):

```
g_max = g_vir · μ(c) / (c²/2)
```

### Properties of the NFW Acceleration Profile (Figure 1)

- For very **concentrated** haloes (`c` large):
  - 50% of mass experiences acceleration `> 7 g_vir`
  - Extends over ~30% of halo
- For **low-concentration** haloes (`c = 2`):
  - Maximum acceleration of ~`(5–7) g_vir` reached at ~2% of virial radius
  - Region contains < 1% of halo mass

**Right panel of Figure 1 key result:** At `z = 0`, `g < g_crit` for all halo masses plotted (`10^8–10^12 M_☉`). At `z = 10`, a substantial fraction of total mass in **massive haloes** can lie **above** `g_crit`.

---

## 3. Accelerations at High Redshift

### 3.1 Halo Mass, Virial Accelerations, and Abundances Across Redshifts

**Virial radius scaling with redshift** (at fixed mass):

```
R_vir = 7 kpc · (M_vir / 10^10 M_☉)^(1/3) · ((1+z)/10)^−1 · (Δ(z)/18π²)^(−1/3)
```

**Virial acceleration scaling with redshift:**

```
g_vir = 2.84 × 10^−11 m s^−2 · (M_vir / 10^10 M_☉)^(1/3) · ((1+z)/10)² · (Δ(z)/18π²)^(2/3)
```

Equivalently in surface density units:

```
g_vir/G = 204 M_☉ pc^−2 · (M_vir / 10^10 M_☉)^(1/3) · ((1+z)/10)² · (Δ(z)/18π²)^(2/3)
```

**Critical implication:**
- Galaxy-scale haloes (`M_vir ~ 10^10–10^12 M_☉`) can reach or exceed `g_crit` at high redshift
- Even `M_vir ≈ 10^9 M_☉` haloes can have `g > g_crit` at `z > 17`

**Inverted: Virial mass at fixed virial acceleration:**

```
M_vir = 10^10 M_☉ · (g_vir/G / 204 M_☉ pc^−2)³ · ((1+z)/10)^−6 · (Δ(z)/18π²)^−2
```

> **Key result:** The virial mass giving a fixed virial acceleration scales as `(1+z)^−6`

### The Threshold Virial Acceleration: `g_thresh`

- Defined by: `g_max(g_vir = g_thresh, z) = g_crit`
- This is the **minimum virial acceleration** for any mass in the halo to exceed `g_crit`
- `g_thresh` corresponds to a nearly constant cumulative comoving number density of:

```
n(> M_vir) ≈ 10^−4 Mpc^−3  (from z = 8 to z = 20)
```

### Haloes at Fixed Number Density Track Fixed Virial Acceleration

**Figure 2 key results:**
- Solid coloured lines show evolution of `g_vir/G = [100, 250, 500, 1000] M_☉ pc^−2` haloes in `M_vir–z` space
- These track the evolution of haloes at **fixed cumulative abundance** very closely for `z > 8`
- Fixed comoving number density ↔ fixed virial acceleration at high redshift

**Cannot use very massive haloes to explain JWST results:**
- `M_vir = 10^12 M_☉` haloes have abundance `≈ 10^−6.5 Mpc^−3` at `z = 7.5`, dropping rapidly at higher z
- Observed bright galaxy number densities from JWST are an order of magnitude larger

### Feedback-Free Burst (FFB) Comparison

- Dekel et al. 2023 (D23) identified conditions for "feedback-free bursts" at `g_vir/G = 381 M_☉ pc^−2`, with `M_vir(z=9) = 10^10.8 M_☉`
- The dotted blue curve (`g_FFB`) in Figure 2 **nearly coincides** with `g_thresh` across the full plotted redshift range
- Both trace `n(> M_vir) ≈ 10^−4 Mpc^−3`
- This reinforces the possibility that dark matter can catalyze efficient galaxy-wide star formation

### Cumulative Number Density as a Function of `g_vir(z)` (Figure 3)

**Left panel:**
- Number density at fixed `g_vir` increases quickly from `z = 0` to `z ≈ 4–5`
- Above `z ≈ 5`, evolution at fixed `g_vir` is almost negligible
- Haloes at fixed number density → nearly fixed `g_vir` at high redshift

**Right panel (fixed number densities `n(>M_vir) = 10^−(4,5,6,7) Mpc^−3`):**
- Both `g_vir` and `g_max` at fixed number densities rise rapidly from `z = 0` to `z ≈ 6–8`, then remain **virtually constant** to `z = 20`
- All four number densities plotted have `g_max > g_crit` at high redshift
- **Critical finding:** These haloes have been above `g_crit` since at least `z ~ 20`, meaning efficient star formation is likely the **first** mode of star formation — no prior feedback can reduce central densities

---

### 3.2 From Halo Properties to Baryons

**Assumption:** Each halo has its cosmic fraction of baryons:
```
M_b(< R_vir) = f_b · M_vir
```
with baryons spatially distributed to match the dark matter.

This provides an **upper limit** to the stellar mass from efficient baryon conversion.

**Figure 4 (Left) — Baryonic mass above `g_crit` vs. redshift:**

| Scenario | Result |
|----------|--------|
| Haloes at `g_thresh` (minimum threshold) | Only `~10^6 M_☉` baryons above `g_crit`, independent of redshift |
| `z = 10`, `M_b(>g_crit) = 10^10 M_☉` | Number density `≈ 10^−7 Mpc^−3` |
| `M_b(>g_crit) > 10^11 M_☉` | Rarer than `1 Gpc^−3` |

> Even with 100% efficiency (`ε = 1`) in the high-acceleration regime: haloes with `10^10 M_☉` of stars will be very rare; haloes with `M_★ > 10^11 M_☉` should not exist at `z > 10`.

**Figure 4 (Right) — `M_b(>g_crit)` as a function of `M_vir` at six redshifts:**

- For survey volumes `V ≈ 10^5 Mpc^3` (rarest objects: `n ≈ 10^−5 Mpc^−3`):
  - At `z ≈ 20`: most massive object should have `M ≈ 10^7 M_☉`
  - At `z ~ 10`: objects with `M ≈ 10^9 M_☉` may be present
- Surveys with `100×` larger volumes can see `10×` more massive objects at fixed redshift
- **Even for extremely rare haloes** (`n = 1 Gpc^−3`), baryon content at high accelerations does not approach `f_b M_vir`

---

### 3.3 Star Formation Rates and Stellar Masses

The region of efficient star formation has a size `r_crit` defined by `g(< r_crit) = g_crit`:

**Size of the efficient star-forming region:**
```
r_crit = 1.8 kpc · (M_tot,crit / 10^10 M_☉)^(1/2)
```

**Mean total density within `r_crit`:**
```
ρ_tot(< r_crit) = 0.4 M_☉ pc^−3 · (M_tot,crit / 10^10 M_☉)^(−1/2)
```

**Free-fall time at `r_crit`:**
```
t_ff(r_crit) = 12.7 Myr · (M_tot,crit / 10^10 M_☉)^(1/4)
```

**Baryonic mass available:**
```
M_b(< r_crit) = f_b · c_b · M_tot(< r_crit)
```
where `c_b` parametrizes baryon concentration relative to dark matter (`c_b = 1` → baryons trace total matter).

**Star formation rate:**
```
Ṁ_★ = ε_ff · M_b(<r_crit) / t_ff(r_crit)
```

**Key parameters:**
- `η_ff ≈ 3`: star formation persists for approximately 3 free-fall times (Elmegreen 2000; Grudić et al. 2018; Kim et al. 2018; Guszejnov et al. 2023)
- Integrated efficiency `ε = η_ff · ε_ff` should be **high (~0.5)** since feedback is ineffective
- Total duration: at most `η_ff · t_ff ≈ 40 Myr` for `M_b(<r_crit) = 2 × 10^9 M_☉`

> Warning: Star formation indicators sensitive to longer timescales will **underestimate** the true instantaneous SFR in these regions.

**Star formation rate expressed in solar masses per year:**
```
Ṁ_★ = 24 M_☉ yr^−1 · (c_b/0.5)^(1/4) · (η_ff/3)^−1 · (M_★/10^9 M_☉)^(3/4)
```

**Specific star formation rate:** `24 Gyr^−1` at `M_★ = 10^9 M_☉`

**Stellar mass dependence:** `Ṁ_★/M_★ ∝ M_★^(−1/4)` (set by `(t_ff · η_ff)^−1`)

**UV magnitude (no attenuation, Salpeter IMF):**
```
M_UV,un = −21.7 − 1.875 log₁₀(M_★/10^9 M_☉) − 0.625 log₁₀(c_b/0.5) + 2.5 log₁₀(K_UV / K_UV,0 · η_ff/3)
```

where `K_UV,0 = 1.15 × 10^−28 M_☉ yr^−1 erg^−1 s Hz` (Salpeter IMF).

> For Chabrier IMF: `K_UV/K_UV,0 ≈ 0.6`, making a given `M_★` brighter by **0.55 mag** in the UV.

---

### 3.4 Comparison with JWST Observations (Figure 5)

**Setup:**
- Shows evolution of `M_★` in systems at fixed cumulative comoving number densities `n(>M_vir) = 10^−(5,6,7,8) Mpc^−3` vs. redshift
- Assumes integrated star formation efficiency `ε = 0.5`
- Survey volumes: CEERS (Finkelstein et al. 2023) and JADES (Eisenstein et al. 2023): `V ≈ (a few) × 10^5 Mpc^3`

**Galaxies shown for comparison (spectroscopically confirmed by JWST):**

| Galaxy | Redshift Reference | Notes |
|--------|-------------------|-------|
| GS-z14 | Carniani et al. 2024 | Highest-z confirmed galaxy |
| GHZ12 | Castellano et al. 2022, 2024; Naidu et al. 2022; Zavala et al. 2025 | |
| Maisie's Galaxy | Finkelstein et al. 2022; Arrabal Haro et al. 2023 | |
| GN-z11 | Oesch et al. 2016; Bunker et al. 2023; Tacchella et al. 2023 | Best-studied case |
| GS-z9 | Curti et al. 2024 | |
| RUBIES-EGS-55604/966323 | Labbé et al. 2023; Wang et al. 2024 | 'Medium' stellar mass values used |

All galaxies shown have inferred stellar masses **consistent** with predictions from this model.

**Detailed GN-z11 comparison (Tacchella et al. 2023):**

| Observable | Observed | Model Prediction |
|------------|----------|-----------------|
| Stellar mass | `≈ 10^9 M_☉` | `~10^9 M_☉` ✓ |
| Half-light radius | 64 pc | `O(100 pc)` ✓ |
| SFR | `≈ 20 M_☉ yr^−1` | `24 M_☉ yr^−1` ✓ |
| M_UV (with 0.2 mag obscuration) | −21.6 | −21.7 (unattenuated) ✓ |
| Expected halo mass | `≈ 10^10.7–10^11 M_☉` | Consistent with Tacchella et al. 2023 ✓ |

> GN-z11 may have been caught directly during its maximally efficient star formation phase.

**GHZ12 comparison (Castellano et al. 2024):**

| Observable | Observed |
|------------|----------|
| Effective radius | `R_e ≈ 100 pc` |
| Stellar mass | `M_★ = 1.1 × 10^9 M_☉` |
| M_UV | −20.53 |
| SFR | `5 M_☉ yr^−1` |

- Model predicts 1–1.5 mag brighter and SFR 3–5× higher if **all** stellar mass formed in a single efficient burst
- A match is achieved if ~20% of mass formed recently in the high-acceleration regime, or if star formation persisted for `~10 t_ff`

**Physical size estimate (for `M_tot,crit ≈ 10^10 M_☉`):**
- Initial: `~0.1 M_☉ pc^−3 ≈ 16 cm^−3` baryons in a region of `~2 kpc`
- After collapse by factor of ~10: `≈ 10^4 cm^−3` in a region of **100 pc**
- This matches observed sizes of bright high-z galaxies

**Future predictions:**
- Baryonic mass within efficient star-forming region drops by factor `≈ 2` from `z = 14` to `z = 16`, and another factor of 2 at `z = 18` at fixed halo number density
- GS-z14's detection (Carniani et al. 2024) augurs well for finding similar galaxies out to `z ~ 18`

---

## 4. Discussion

### Core Idea

At fixed virial mass, dark matter haloes at high redshift are much **denser** than at low redshift. This higher density causes large quantities of baryons to experience accelerations high enough that stellar feedback becomes ineffective. In this regime, dark matter provides the high acceleration needed for efficient star formation on **galaxy-wide scales** — much larger than the dense molecular cloud cores where this occurs locally.

> "This general picture appears unavoidable."

### Dependences and Uncertainties

**Halo concentrations:**
- Control the amount of mass above `g_crit`
- Using median `c(M_vir|z)` relations from Ishiyama et al. 2021 (z=0) and Yung et al. 2024 (high-z)

**Value of `g_crit`:**
- Treated as a characteristic scale, not a sharp threshold
- More top-heavy IMF (invoked in some JWST explanations: Inayoshi et al. 2022; Steinhardt et al. 2023; van Dokkum & Conroy 2024; Lu et al. 2025; Menon et al. 2024) would increase `ṗ/m` and therefore `g_crit`

### Robustness

The picture is robust **unless** there is a mechanism for reducing central halo densities at very high redshifts. Candidates evaluated:

1. **Star formation feedback:** Unlikely — haloes enter the high-acceleration regime before forming significant stars (Fig. 3)
2. **Dark matter self-interactions (SIDM):** Generally reduces central densities (Spergel & Steinhardt 2000; Bullock & Boylan-Kolchin 2017; Buckley & Peter 2018; Tulin & Yu 2018), but likely inefficient here due to rapid mass assembly acting as heat supply, preventing core creation (Davé et al. 2001)
3. **AGN feedback:** Requires massive black holes; likely operates only **after** an epoch of efficient galaxy formation

### Efficiency Limits

- The process does **not** approach the theoretical maximum of `f_b M_vir`:
  - Most baryons reside at low densities far from the halo centre
  - Populations requiring `M_★/(f_b M_halo) ≈ 1` would remain difficult to explain within ΛCDM
- Provides a natural explanation for why efficient large-scale star formation **cannot** continue to lower redshifts

### Star Formation Duty Cycles

Haloes remaining in the `g > g_crit` regime for extended periods may undergo **repeated cycles** of efficient bursts:

- Burst timescale: local free-fall time `t_ff` (equation 11)
- Re-supply timescale: virial crossing time `t_vir = 75 Myr` at `z = 10` (independent of halo mass), scaling as `(1+z)^(−3/2)`

| Halo Rarity | Duty Cycle |
|-------------|------------|
| `n(>M_vir) ≈ 10^−8 Mpc^−3` (very rare) | `~35%` (more continuous) |
| `n(>M_vir) ≈ 10^−5 Mpc^−3` (more common) | `≤ 15%` |

Most massive/rarest galaxies → more evidence of continuous efficient star formation.

### Comparison with Feedback-Free Bursts (Dekel et al. 2023)

Despite nearly identical redshift-dependent threshold masses (Fig. 2), the models differ substantially:

| Property | Dekel et al. 2023 (FFB) | This Work |
|----------|------------------------|-----------|
| `M_★` at `z ≈ 10`, `M_vir ≈ 10^10.8 M_☉` | `≈ 10^10 M_☉` | `~10^9 M_☉` (order of magnitude lower) |
| SFR at same halo mass | `65 M_☉ yr^−1` | `~20–30 M_☉ yr^−1` (2–3× lower) |
| Threshold mass scaling | Similar `(1+z)^−6` | Same |
| Physical mechanism | Different (feedback-free) | Dark matter gravity |

The two models may have concordant or conflicting predictions — worth detailed comparison.

### Additional Implications

**Star cluster formation:**
- At high efficiency (`ε > 0.25`), fraction of stars forming in **bound clusters approaches unity** (Grudić et al. 2021; Hills 1980; Krumholz et al. 2019; Li et al. 2019)
- Result: galaxy regions dominated by **young star clusters**
- Supported by JWST observations of lensed systems (Adamo et al. 2024a; Bradley et al. 2024; Fujimoto et al. 2024)
- `~10^6 M_☉` baryons may form **individual globular clusters** as early as `z ≈ 20` in haloes with `n > 10^−4 Mpc^−3`

**Globular cluster connection:**
- Minimum baryonic mass above `g_crit` in lowest-mass qualifying haloes: `~10^6 M_☉` — intriguingly similar to globular cluster masses, at **all redshifts**

**Top-heavy IMF and black hole seeds:**
- Dense clusters may host top-heavy IMFs (Haghi et al. 2020) and supermassive stars
- Supermassive stars: leading candidate for anomalous chemical abundances in massive globular clusters (Denissenkov & Hartwick 2014; Bastian & Lardo 2018)
- This star formation mode may be conducive to:
  - IMF variations
  - Massive black hole seed formation
  - **Direct collapse black holes** in metal-free gas at higher redshift

**Self-enrichment:**
- Escape velocity from these star-forming regions is much higher than for typical molecular clouds (greater mass + large dark matter reservoirs on larger scales)
- May result in more efficient **self-enrichment** of early galaxies and star clusters
- Could explain abundance anomalies observed in subsets of stars in massive globular clusters

### Fate of High-Density Regions Under Hierarchical Assembly

Two options for why similar densities don't occur in more massive haloes at lower redshift:

**Option (1): Densities reduce through hierarchical assembly**
- Controlled dark matter merger simulations indicate central densities actually **increase** in physical units during mergers (Boylan-Kolchin & Ma 2004; Kazantzidis, Zentner & Kravtsov 2006; Drakos et al. 2019)
- Diemand, Kuhlen & Madau 2007: relative constancy of `M(<r)` at small physical radii during cosmological evolution of MW-mass haloes
- However, the required effect only needs to operate in rare haloes (`n < 10^−5 Mpc^−3`) — not surprising it hasn't been seen in typical MW-mass zoom-in simulations
- Loeb & Peebles 2003; Gao et al. 2004: models do point to reduction of physical densities in halo centres at fixed number density over cosmic time
- **Galaxy cluster-scale zoom-in simulations** (e.g. Gao et al. 2012) are of great interest here

**Option (2): Descendants survive with high densities today**
- High-acceleration haloes may end up with **high concentrations** for their mass at `z = 0`
- Or as **dense substructure** in more massive systems (Ishiyama 2014; Errani, Peñarrubia & Walker 2018; van den Bosch & Ogiya 2018; Delos & White 2023)
- Physical stellar densities of the most massive galaxies at high redshift are comparable to those of the most massive **ellipticals** in the local Universe (Hopkins et al. 2010; Baggen et al. 2023)
- Galaxies in earliest-forming massive haloes → excellent probes of efficient galaxy formation at high redshift

---

## 5. Conclusions

### Summary of Key Results

1. **Star formation is generally inefficient** (`ε_ff < 2%`), but becomes efficient when `g > g_crit ≈ 5 × 10^−10 m s^−2` (equivalently `Σ_crit = g_crit/(πG) ≈ 1000 M_☉ pc^−2`)

2. **At low redshift:** only baryon-dominated regions (dense molecular cloud cores) achieve these accelerations. Dark matter cannot.

3. **At high redshift:** the significantly higher mean density of the Universe results in galaxy-mass dark matter haloes where dark matter can provide accelerations sufficient for efficient star formation of **galaxy-scale quantities of stars** (`M_★ ~ 10^8–10^10 M_☉`)

4. **Virial mass scaling:** `M_vir(g_vir = const) ∝ (1+z)^−6`, tracking roughly constant cumulative comoving number densities at early times

5. **Threshold virial acceleration:** `g_vir/G ≈ 380 M_☉ pc^−2` (where central halo acceleration just exceeds `g_crit`), corresponding to `n(>M_vir) ≈ 10^−4 Mpc^−3`

6. **Predicted properties of star-forming regions** (for `M_b(>g_crit) ≈ 2 × 10^9 M_☉`):
   - Initial size: `≈ 2 kpc`
   - Baryonic density: `≈ 0.1 M_☉ pc^−3`
   - Free-fall time: `≈ 13 Myr`
   - Stellar mass formed (with `ε = 0.5`): `10^9 M_☉` in `≈ 40 Myr`
   - SFR: `24 M_☉ yr^−1`
   - UV magnitude: `−21.7` (Salpeter IMF, unattenuated)
   - Specific SFR: scales as `M_★^(−1/4)`
   - Final size after collapse (factor ~10): `O(100 pc)`

7. **Comparison with JWST observations:** Good agreement with luminous high-z systems such as GN-z11; predicted number densities `n(>M_vir) ≈ 10^−5.5–10^−6.5 Mpc^−3` are consistent with CEERS and JADES survey volumes

8. **Near-identical threshold mass evolution** to feedback-free bursts (Dekel et al. 2023), despite different underlying physics; quantitative predictions differ non-trivially (lower stellar masses and SFRs in this model)

9. **Future detections:** Model predicts similar or slightly less massive/bright galaxies should be detectable out to `z ~ 18`

### Open Questions for Future Work

- Full `ε–g` relation (Fall et al. 2010; Grudić et al. 2018; Hopkins et al. 2022)
- Cosmological distribution of halo concentrations at fixed mass and redshift
- Fate of high-density regions under hierarchical assembly
- Detailed comparison with feedback-free burst model predictions
- IMF variations at high surface density
- Direct collapse black hole formation in this context

---

## Acknowledgements

The author thanks Volker Bromm, James Bullock, Neal Evans, Phil Hopkins, Pawan Kumar, Stella Offner, Eliot Quataert, Jenna Samuel, and Stuart Wyithe for helpful conversations.

**Funding:** NSF CAREER award AST-1752913, NSF grants AST-1910346 and AST-2108962, NASA grant 80NSSC22K0827, HST-GO-16686, HST-AR-17028, HST-AR-17043 (STScI/AURA, NASA contract NAS5-26555).

**Software used:** NumPy (Harris et al. 2020), SciPy (Virtanen et al. 2020), Matplotlib (Hunter 2007), IPython (Pérez & Granger 2007), HMF (Murray et al. 2013; Murray 2014), COLOSSUS (Diemer 2018).

**Data availability:** No new data were generated or analysed.

---

## Key Equations Reference Sheet

| Equation | Formula | Description |
|----------|---------|-------------|
| Virial mass definition | `M_vir = (4π/3) R_vir³ Δ_vir ρ_m` | Eq. 1 |
| Virial acceleration | `g_vir = G M_vir / R_vir²` | Eq. 2 |
| NFW acceleration profile | `g(r) = g_vir · μ(c)/μ(cr̃) / r̃²` | Eq. 3 |
| Maximum NFW acceleration | `g_max = g_vir · μ(c) / (c²/2)` | Eq. 4 |
| Virial radius vs. redshift | `R_vir = 7 kpc · (M/10^10)^(1/3) · ((1+z)/10)^−1 · ...` | Eq. 5 |
| Virial acceleration vs. redshift | `g_vir/G = 204 M_☉ pc^−2 · (M/10^10)^(1/3) · ((1+z)/10)²` | Eq. 7 |
| Halo mass at fixed acceleration | `M_vir ∝ (1+z)^−6` | Eq. 8 |
| Size of efficient region | `r_crit = 1.8 kpc · (M_tot,crit/10^10)^(1/2)` | Eq. 9 |
| Density in efficient region | `ρ_tot = 0.4 M_☉ pc^−3 · (M_tot,crit/10^10)^(−1/2)` | Eq. 10 |
| Free-fall time | `t_ff = 12.7 Myr · (M_tot,crit/10^10)^(1/4)` | Eq. 11 |
| SFR | `Ṁ_★ = 24 M_☉/yr · (c_b/0.5)^(1/4) · (η_ff/3)^−1 · (M_★/10^9)^(3/4)` | Eq. 14 |
| UV magnitude | `M_UV = −21.7 − 1.875 log₁₀(M_★/10^9) − ...` | Eq. 15 |

---

## Selected Key References

- **Boylan-Kolchin 2023** — Prior work on JWST tension with ΛCDM
- **Dekel et al. 2023** — Feedback-free bursts model (FFB); very similar threshold mass evolution
- **Fall, Krumholz & Matzner 2010** — Critical acceleration/surface density threshold theory
- **Grudić et al. 2020** — Numerical results on star formation efficiency vs. surface density
- **Ishiyama et al. 2021** — Halo concentration–mass relation used (z=0)
- **Yung et al. 2024** — Halo concentration–mass relation used (high-z)
- **Tacchella et al. 2023** — GN-z11 observations (main comparison galaxy)
- **Carniani et al. 2024** — GS-z14 (highest-redshift confirmed galaxy)
- **Castellano et al. 2024** — GHZ12 observations
- **Planck Collaboration VI 2020** — Cosmological parameters adopted
- **NFW 1996, 1997** — Halo density profile used throughout
- **Bryan & Norman 1998** — Virial overdensity definition

---

*This document is a structured markdown summary of Boylan-Kolchin (2025), MNRAS 538, 3210–3218. Open Access under Creative Commons Attribution License (CC BY 4.0).*

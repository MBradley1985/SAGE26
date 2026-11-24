# SAGE26: the AGN heating radius in the CGM regime — problem and proposed fix (2026-11-23)

For the SAGE26 team. This note concerns the design of `r_heat` in SAGE26's two-regime cooling model, not a coding slip. It follows the merger-CGM bug fixed in v0.95 (`2ea9df5`, `07f63f7`), which was one visible symptom of what is described here. Code references are to SAGE26 **v0.95**, `src/model_cooling_heating.c`, unless stated otherwise.

## Summary

- **What `r_heat` is for.** It is the classic SAGE AGN heating radius: a memory of past radio-mode heating in a hot, quasi-static atmosphere, acting as an entropy floor that only moves outwards. It is defined relative to the hot halo's cooling radius `rcool`, both being radii in the same isothermal gas profile, and the scale it sets grows and shrinks with the hot gas.
- **What SAGE26 does with it.** The CGM-regime (Regime 0) recipe reuses `r_heat` with `rcool` pinned to `Rvir`. There it is not a radius in the gas: it ratchets towards a fixed ceiling (`Rvir`) and, once there, shuts off cooling permanently, however much CGM gas the halo later holds.
- **What happens.** The ratchet carries `r_heat` all the way to `Rvir` whenever the radio-mode heating stays above about a quarter of the cooling, and that needs only a black hole. In Millennium at z = 0, about half to three-quarters (62% overall) of CGM-regime centrals hosting a black hole are locked out. That is 18–23% of all centrals with stellar mass 10^8.5–10^10.5 Msun, and they keep 5–12% of their virial mass as CGM gas that never cools.
- **Proposal.** Restrict `r_heat` to the hot regime:
  - the CGM recipe neither reads nor writes `r_heat`;
  - no radio-mode accretion in the CGM regime, which costs black holes essentially nothing (§5);
  - `r_heat` is left as it is while a halo is in Regime 0.

  Because the regime is re-drawn every snapshot, this needs a companion decision on how the regime is drawn near `Mshock` (§5.6). The largest response is expected near the knee of the stellar mass function, not only in dwarfs.
- **Alternative.** If the team wants AGN regulation to continue below `Mshock`, the second choice is a SAGE16-style moving reference radius in the CGM regime, in place of the pinned `Rvir` (§5.7).

## 1. Intended behaviour: `r_heat` in the hot regime

This is the classic SAGE radio-mode model (Croton et al. 2006, 2016), kept by SAGE26 in `cooling_recipe_hot()` and `do_AGN_heating()`.

- **Cooling radius.** The hot gas is a singular isothermal sphere, so its profile is fixed by `HotGas` and `Rvir` (`rho0 = HotGas / (4π Rvir)`, line 187). `rcool` is the radius where the cooling time equals the halo dynamical time (line 188), so `rcool ∝ √HotGas`.
- **Heating radius.** `r_heat` marks how far out past AGN heating has raised the gas entropy. Gas inside it no longer cools.
- **Suppression.** Cooling comes from the shell between `r_heat` and `rcool`. With `M(<r) ∝ r` in this profile, the un-heated fraction of the cooling is `(1 − r_heat/rcool)` (`suppress_cooling()`, line 119, applied at line 784). Cooling stops when `r_heat ≥ rcool`; with `CGMrecipeOn 1` it stops already at `0.99 rcool`.
- **Ratchet.** After the suppression, the radio-mode heating is compared with the suppressed cooling, and `r_heat_new = (AGNheating / coolingGas) · rcool` (lines 803–808). `r_heat` only moves outwards. The heating does not itself remove any of this substep's cooling (the function returns the cooling suppressed by the *old* `r_heat`); it acts only through `r_heat`, from the next substep on.
- **Threshold to full shut-off.** Because the heating is measured against the already-suppressed cooling, the ratchet maps `x = r_heat/rcool` to `x → max(x, min(f/(1−x), 1))`, where `f` is the heating over the unsuppressed cooling. For a sustained `f ≤ 1/4`, `x` settles at a partial suppression (at most one half). For `f > 1/4`, it runs to full shut-off within a few substeps (6 substeps at `f = 0.3`). This is how SAGE shuts down massive haloes.
- **Why it only moves out.** Radio jets leave a lasting mark on a hot atmosphere. Before `r_heat`, each new cooling episode proceeded as if there had never been an AGN. Group and cluster X-ray atmospheres show this lasting heating as an entropy excess above the gravity-only expectation (Ponman, Cannon & Navarro 1999; Voit 2005).
- **The floor is set in physical units.** `rcool` scales with the hot gas, so a floor set while the atmosphere was thin is a small radius. Cooling resumes outside it once the halo builds more hot gas.
- **Black-hole growth.** Radio-mode accretion is drawn from `HotGas`. Since SAGE commit `bcead05` (April 2015), its heating is capped at the suppressed cooling and the accretion cut back to match (line 763). So once `r_heat ≥ rcool`, a black hole accretes nothing in radio mode. The same cap bounds a black hole's lifetime radio-mode growth by the total gas cooled divided by `AGNcoeff = (1.34e5 km/s / Vvir)²`.

## 2. What SAGE26 does in the CGM regime

Below the shock mass `Mshock`, a central is classified Regime 0 and its gas is held as `CGMgas`.

- **Cooling.** `cooling_recipe_cgm()` cools the CGM as a whole, with a Carr et al. (2023) α = 1.4 profile and rate `CGMgas / (tcool + tff)` (lines 337–399). It has no cooling radius.
- **The pinned radius.** For the AGN step it sets `rcool = Rvir` (line 381: "Pin rcool to Rvir since we are evaluating bulk accretion").
- **The AGN step.** It calls `do_AGN_heating_cgm()` (line 394), which repeats the hot-halo logic (lines 827–876):
  - suppression by `(1 − r_heat/Rvir)`, and zero cooling once `r_heat ≥ 0.99 Rvir`;
  - Bondi accretion drawn from `CGMgas`, its heating capped at the suppressed cooling;
  - the ratchet `r_heat_new = (AGNheating / coolingGas) · Rvir`, capped at `Rvir`.
- **One field, both regimes.** `r_heat` is a single field per galaxy.
- **The regime draw.** Each snapshot, the regime is re-drawn from a 0.1-dex sigmoid around `Mshock`, keyed on the halo and snapshot (`src/model_regimes.c:119-125`). A halo near `Mshock` therefore switches regime from snapshot to snapshot. With `MShockMsun = 6e11`, the chance of a Regime 0 draw is 23% at 10^11.9 Msun, 10% at 10^12.0 and 1.5% at 10^12.2.
- **The fold on a switch.** `infall_recipe()` moves the whole hot-phase reservoir across at each switch: `HotGas` to `CGMgas` for Regime 0, `CGMgas` to `HotGas` for Regime 1.

## 3. The problem

### 3.1 No physical radius behind it

With `rcool` fixed at `Rvir`, `r_heat` is no longer a radius in the gas. The suppression `(1 − r_heat/rcool)` is the un-heated mass fraction only for an isothermal profile that cools from the inside out. The CGM recipe uses an α = 1.4 profile (`M(<r) ∝ r^1.6`) and cools the reservoir as a whole, with no shell structure, so neither the suppression nor the ratchet has a geometric meaning there.

### 3.2 A fixed ceiling makes the lock-out permanent

In the hot regime, the ratchet's scale (`rcool`) moves with the gas. In the CGM regime it is fixed at `Rvir`, and the ratchet threshold (§1) applies against that fixed scale. Bondi accretion scales with `M_BH`, so once a central has a black hole whose heating stays above about a quarter of its CGM cooling, `r_heat` runs to `Rvir` and the CGM stops cooling for good.

Nothing undoes this. Refilling the reservoir does not change the reference scale, and `Rvir` grows only slowly. In practice, the CGM regime's AGN model is a one-way switch, thrown once by a black hole of modest size.

### 3.3 The wrong physical setting

Below `Mshock`, by the model's own definition (Dekel & Birnboim 2006), the halo cannot sustain a stable, virial-shock-heated atmosphere. SAGE26 models its gas with a precipitation-style recipe, and precipitation regulation is self-regulating, not one-way (Voit et al. 2015). Entropy floors are measured in group and cluster atmospheres, not in galaxy-scale haloes below the shock mass. A permanent shut-off of cooling to represent past heating fits poorly here.

### 3.4 One variable, two meanings

The same `r_heat` is measured against the hot `rcool` in Regime 1 and against `Rvir` in Regime 0, and the per-snapshot regime draw lets one leak into the other.

- **Regime 0 → 1.** A Regime 0 lock-out (`r_heat = Rvir`) enters the hot regime as a floor at `Rvir`. That shuts off hot cooling wherever `rcool ≤ Rvir/0.99`, and suppresses it in part above that.
- **Regime 1 → 0.** With `CGMrecipeOn > 0`, `do_AGN_heating()` caps `r_heat` at `Rvir` (lines 815–816). So a hot halo with `rcool > Rvir` can reach `r_heat = Rvir` while still cooling partially. If the next draw is Regime 0, the same value zeroes all its CGM cooling.

**The v0.95 merger bug was this mismatch reaching into hot haloes.** A merger left a small CGM deposit in a hot-regime central, and the CGM recipe's `Rvir`-scale ratchet overwrote the hot halo's floor. For a deposit that small the CGM cooling time far exceeds `tff`, so its cooling falls as `CGMgas²` and the AGN out-heats it at once. `2ea9df5` removed that source and `07f63f7` guards against others. Regime 0 itself is unchanged in v0.95; its test `test_cgm_recipe_leaves_hot_halo_rheat` asserts the Regime 0 lock-out as its control case.

## 4. Consequences (measured)

**Provenance.**

- **Source.** Mimic's `sage26` package, on Millennium tree files 0–15 of 512 (1/32 of the volume). Fractions are representative; the absolute counts are for that sub-volume only.
- **Version.** The package reproduces SAGE26 at `11b1ae8`, which predates v0.95 by 162 commits. Among the differences, v0.95 draws the regime from a keyed hash rather than `rand()`. The CGM-regime AGN logic is the same in both, so similar numbers are expected on v0.95, but they should be re-measured there.
- **Definition.** *Locked out* means a Regime 0 central with `r_heat ≥ 0.99 Rvir`; its CGM cooling is zero.

**Regime 0 centrals at z = 0, by halo mass.** 99.7–100% of locked-out centrals have `CoolingRate` exactly 0, and 92–100% still hold CGM gas.

| log10 Mvir [Msun] | Locked out | Locked out among black-hole hosts | CGM / Mvir, locked / not | Median sSFR, locked / not [yr⁻¹] |
|---|---|---|---|---|
| 10.50 | 8% | 72% | 0.05 / 0.005 | 1e-14 / 7e-11 |
| 11.00 | 28% | 55% | 0.12 / 0.01 | 2e-13 / 9e-11 |
| 11.50 | 58% | 62% | 0.11 / 0.04 | 1e-11 / 1e-10 |
| 11.75 | 73% | 75% | 0.10 / 0.05 | 3e-11 / 1e-10 |

**Interpretation.**

- **A black hole is the trigger.** None of the 51,765 locked-out Regime 0 centrals at 10^10.5–10^12 Msun lacks a black hole, and 48–75% of those with one are locked out, by bin (62% overall).
- **The black holes come from quasar mode, not radio mode.** By the bound in §1, lifetime radio-mode growth is at most about `f_b·Mvir / AGNcoeff`, of order 10^3–10^4 Msun for a 10^11 Msun halo, against a median `M_BH` of about 10^6 Msun.
- **The locked-out haloes are gas-rich.** Their CGM is a large reservoir, not a remnant: in most of them it would out-cool the AGN if the lock-out were lifted.
- **Median CGM cooling times are roughly 0.03–0.2 of `tff` by bin, always well below 1**, so CGM cooling there is close to linear in the reservoir.

**All centrals by stellar mass.** Quenched means sSFR < 10⁻¹¹ yr⁻¹. Each z = 0 cell reads: locked out / locked and quenched / quenched but not locked.

| log10 M* [Msun] | z = 0 | Quenched at z ≈ 1.4 | Quenched at z ≈ 3 |
|---|---|---|---|
| 8.5–9.0 | 18% / 12% / 15% | 11% | 4% |
| 9.0–9.5 | 18% / 10% / 11% | 6% | 1% |
| 9.5–10.0 | 23% / 13% / 9% | 3% | 0.3% |
| 10.0–10.5 | 23% / 9% / 19% | 4% | 0.3% |
| 10.5–11.0 | 7% / 1% / 47% | 14% | 1% |

- **Lock-out accounts for a third to over half of the quenched centrals at 10^8.5–10^10.5 Msun at z = 0** (33–58% by bin). It accounts for a similar share among the smallest centrals at higher redshift, where far fewer are quenched.
- **The comparison with isolated dwarfs.** Isolated dwarf galaxies are observed to be almost never quenched (Geha et al. 2012). The model quenches 27% of centrals at 10^8.5–10^9 Msun. Removing the lock-out would remove at most the 12% that are locked and quenched; about 15% stay quenched for other reasons. The fix roughly halves this tension rather than removing it. A direct comparison also needs matched definitions.
- **Leakage into the hot regime.** In the same output, 609, 199 and 21 locked-out Regime 1 centrals at log10 Mvir = 11.75, 12.0 and 12.25 have a valid hot `rcool` below `0.99 Rvir`. These are candidates for a lock-out carried over from Regime 0, or, before the fix, set by merger CGM. Most other locked-out Regime 1 centrals near `Mshock` come from the ordinary hot ratchet clipped at `Rvir`.

## 5. Proposed solution: `r_heat` in the hot regime only

**Principle.** `r_heat` represents an entropy floor in a hot, quasi-static atmosphere. The model has such an atmosphere only in Regime 1, so `r_heat` should act only there.

1. **The CGM recipe stops reading `r_heat`.** CGM cooling is no longer suppressed by it. This applies to every galaxy that runs `cooling_recipe_cgm()`: Regime 0 centrals and satellites, and the secondary CGM call for a Regime 1 halo (lines 637–639). After `2ea9df5` that secondary call rarely has gas to work on.
2. **The CGM recipe stops writing `r_heat`.** No ratchet and no `Rvir` cap. The `07f63f7` guard becomes redundant. `2ea9df5` stays: routing merger gas to the target's regime reservoir is correct on its own terms.
3. **No radio-mode accretion in the CGM regime (recommended).** Remove the `do_AGN_heating_cgm()` call; black holes in Regime 0 then grow in quasar mode only.
   - Bondi accretion from a hot atmosphere, and radio-mode maintenance of it, presuppose the hot halo that Regime 0 lacks.
   - Low-mass haloes in a SAM are regulated by supernova feedback.
   - It costs essentially no black-hole growth (§4). If anything, the restored cooling gives more cold gas to quasar-mode growth.

   **Alternative:** keep radio-mode heating in the CGM regime with no memory, offsetting only the current substep's cooling. This is new code, since heating currently acts only through `r_heat` and is never subtracted from the cooling. It would also need its own justification: a central with a black hole would then have its CGM cooling cut back substep by substep. It is not recommended.
4. **Leave `r_heat` untouched in Regime 0; do not reset it.** `r_heat` then only ever holds hot-regime history.
   - A halo that switches between regimes near `Mshock` keeps its genuine hot-regime floor, as the entropy-floor picture intends.
   - A reset on Regime 1 → 0 would erase that history on every random switch.
   - A halo whose first hot spell is its first Regime 1 draw starts from whatever it has, which is zero unless a stray Regime 1 draw at lower mass already set it.
5. **The hot regime is unchanged, except one item for the team to decide: the `Rvir` cap.** With `CGMrecipeOn > 0`, the cap on `r_heat` in `do_AGN_heating()` (lines 815–816) exists only because `r_heat` was shared with the `Rvir`-scale CGM recipe. Under this proposal it has no remaining purpose. Meanwhile, it stops haloes with `rcool > Rvir` from ever being fully shut off, unlike classic SAGE. Removing it restores the classic behaviour. Suppression, the ratchet, the accretion cap and the `0.99 rcool` saturation stay as they are.
6. **The regime draw near `Mshock` (the main risk).** At 10^11.5–10^12.0 Msun, 58–73% of Regime 0 centrals are currently locked out, holding about 10% of `Mvir` as CGM.
   - The proposal releases this gas at the CGM cooling rate. Because of the fold on each switch (§2), a halo that is AGN-quenched in the hot regime will also cool freely during any snapshot in which it is drawn Regime 0: about one snapshot in four at 10^11.9 Msun and one in ten at 10^12.0.
   - Today the shared `r_heat` hides this. Under the proposal, galaxies near the knee could keep forming stars that the hot-regime AGN would otherwise have shut off.
   - **Options, at SAM depth:**
     - (a) Accept it, and recalibrate.
     - (b) Draw the regime from a fixed per-galaxy quantile, so that a halo crosses `Mshock` once as it grows. This keeps the sigmoid's scatter in the transition mass. The code comment at `src/model_regimes.c:119-121` shows SAGE26 has moved away from this before, so the reason for that change should be weighed.
     - (c) Make the transition one-way: once hot, stay hot unless `Mvir` falls clearly below `Mshock`.

   Option (b) needs the quantile stored on the galaxy (the current hash includes the snapshot), and still flips when `Mvir` fluctuates about that halo's own threshold. Only (c), or (b) combined with (c)'s hysteresis, removes the flip-flop fully. Either is a few lines of code. The team should choose one with the rest of this change.

### 5.7 The main alternative: a moving reference radius in the CGM regime (SAGE16-style)

If the team prefers to keep AGN regulation of haloes below `Mshock`, the closest fix in SAGE16's spirit is to remove the *fixed reference*, not the cap.

**The cap alone does not help.** The heating is capped at the suppressed cooling, so `AGNheating / coolingGas ≤ 1`, and the ratchet can never take `r_heat` past the radius it is measured against. In the CGM recipe that radius is pinned at `Rvir`, so `r_heat ≤ Rvir` holds anyway. The CGM recipe's own `Rvir` cap matters only for a value carried in from the hot regime. The ceiling that causes the lock-out is the pinned `rcool = Rvir` (line 381), not a cap line.

**What a moving reference would look like.** Replace the pinned `rcool` with a real cooling radius in the CGM profile, defined as in SAGE16 (where the cooling time equals `Rvir/Vvir`), and drop the cap. The table below estimates that radius for z = 0 Millennium Regime 0 centrals. It is a rough estimate from the stored CGM cooling times: the stored value is the profile's effective cooling time, and the cooling time at `Rvir` is 3.0 times that for the α = 1.4 profile, scaling as `r^1.4`.

| log10 Mvir [Msun] | `rcool > Rvir` | Median `rcool/Rvir` |
|---|---|---|
| 10.50 | 78% | 1.7 |
| 11.00 | 92% | 2.9 |
| 11.50 | 86% | 2.1 |
| 11.75 | 77% | 1.7 |

Most CGM-regime haloes would sit in SAGE16's rapid-cooling regime (`rcool > Rvir`), which fits haloes without a stable hot atmosphere. An `rcool` beyond `Rvir` extrapolates the profile past the halo boundary, as SAGE16's own isothermal `rcool` does in that regime.

**What it would fix:**

- **Some lock-outs become temporary.** `rcool` grows as the CGM refills, so a fixed `r_heat` falls behind and cooling resumes, as in SAGE16. But the ratchet threshold depends on the ratio of heating to cooling, not on the reference radius. A halo whose heating stays above about a quarter of its cooling stays shut off, or shuts off again.
  - An order-of-magnitude reconstruction from the output fields suggests this applies to roughly a third of today's locked-out centrals at 10^10.5–10^11.5 Msun, rising to about half to two-thirds near `Mshock`.
  - The rest would cool again. That is closer to the self-regulation the precipitation picture describes (§3.3), but it is only a partial release.
- **Mixing from Regime 1 into 0 is milder.** Both references would be real radii in the gas. For the same reservoir, the CGM-profile `rcool` is about 1.5–2 times the isothermal one, so a hot-regime floor carried into a Regime 0 snapshot would only partly suppress the cooling. This shrinks the flip-flop risk of §5.6, but does not remove it.
- **Mixing from Regime 0 into 1 can be harsher.** A partial CGM floor above roughly 0.5–0.65 of the CGM `rcool` exceeds the smaller hot `rcool`, so it shuts off hot cooling completely on entry. Carrying the floor across as a fraction rather than a radius would avoid this, but it is more machinery.

**What it would not fix:**

- **The profile mismatch remains (§3.1).** `(1 − r_heat/rcool)` is the shell fraction for an isothermal profile (`M ∝ r`) that cools from the inside out. The CGM recipe cools the whole reservoir in an α = 1.4 profile (`M ∝ r^1.6`).
  - SAGE16's linear form is equally notional when `rcool > Rvir`, so keeping it as an accepted approximation is reasonable.
  - A profile-matched form is optional. It would have to change the suppression and the ratchet together: `1 − (r_heat/rcool)^1.6` and `r_heat_new = (AGNheating/coolingGas)^(1/1.6) · rcool`. Changing only one recreates the §3.1 inconsistency.
  - With both matched, the dynamics in heated-mass fraction are unchanged. Only how fast the floor falls behind a growing `rcool` differs.
- **The physical premise remains (§3.3).** It keeps a persistent AGN entropy floor in haloes where none is observed, and Bondi "maintenance" accretion without a hot atmosphere. The radio-mode black-hole growth it preserves is negligible anyway (§4).

**What it costs.** At minimum, a cooling-radius line for the CGM profile in place of line 381, removal of both `Rvir` caps, and tests. That is small, but it adds code where the main proposal removes it. Matching the suppression and ratchet to the profile, or carrying the floor across regimes as a fraction, would add more.

**Other repairs, briefly.** One could instead store the floor as a heated mass, or let it relax while a halo is in Regime 0. Each adds machinery, and possibly a parameter, for the same physical premise.

**Recommendation.** Hot-only (§5.1–5.6) remains the recommendation: it is simpler, it is better grounded in where entropy floors are observed, and it deletes code. The moving reference is the reasonable second choice if the team wants AGN regulation to continue below `Mshock`. If they take it, they should also take §5.5 (remove the hot-regime cap) so that both regimes use real radii with no ceiling.

## 6. Expected effects and validation

**Expected effects** (direction only; magnitudes need a run):

- **Low-mass centrals.** CGM-regime centrals with black holes keep cooling. The quenched fraction of low-mass centrals falls, by up to the "locked and quenched" share in §4 (about 9–13% at z = 0).
- **The knee.** The largest change is likely there (M* ≈ 10^10–10^10.7), through the released CGM near `Mshock` and the regime draw (§5.6). The stellar mass function and quenched fraction at 10^10–10^11 Msun may rise and fall respectively.
- **Black holes.** In Regime 0 their growth is essentially unchanged; quasar mode may grow them slightly more. In Regime 0, `Heating` and `RadioModeBHaccretionMass` become zero.
- **`r_heat` output.** It is zero for haloes that have never been hot. Otherwise it holds its last hot-regime value while the halo is in Regime 0.
- **Recalibration.** It may be needed for supernova feedback (`FeedbackReheatingEpsilon`, `FeedbackEjectionEfficiency`), reincorporation, `MShockMsun`, `RadioModeEfficiency` and, depending on §5.6, the regime draw.

**Unit tests:**

- A Regime 0 call leaves `r_heat` unchanged.
- Regime 0 CGM cooling is the same whatever the stored `r_heat`.
- A Regime 1 → 0 → 1 sequence preserves the hot-regime `r_heat`.
- Regime 0 radio-mode accretion is zero.
- The Regime 0 control assertion in `test_cgm_recipe_leaves_hot_halo_rheat` must be inverted.

**Science checks** on Millennium and microUchuu:

- the stellar mass function at z = 0–3;
- the stellar–halo mass relation;
- the quenched fraction of centrals against stellar mass (and of isolated dwarfs, if possible);
- cooling and star formation rates at `Mshock` ± 0.3 dex, split by regime;
- the black-hole–bulge relation;
- the cosmic star-formation-rate density;
- the fraction of Regime 1 centrals just above `Mshock` with `r_heat ≥` the hot `rcool`.

**Regression baseline:** must be re-captured.

**Text to update with the change:**

- the field description "capped at Rvir in the CGM regime" in `src/core_allvars.h:207`, `src/io/save_gals_binary.h:117` and `src/io/save_gals_hdf5_fields.h:309`;
- the comment at `src/model_mergers.c:520-523`;
- the `do_AGN_heating_cgm()` section of `docs/physics/cooling_and_heating.md`.

## 7. Related points (outside this proposal)

- **Documentation.** `docs/physics/cooling_and_heating.md` says the hot-halo path has "no `R_vir` cap". The code caps `r_heat` at `Rvir` in `do_AGN_heating()` when `CGMrecipeOn > 0` (lines 815–816); see §5.5.
- **Cold streams.** In the hot regime, the AGN suppression also applies to the Dekel & Birnboim cold-stream cooling, as the 2016 code did for its rapid-cooling branch. Cold filaments that penetrate the hot atmosphere arguably should not feel its entropy floor. This is a separate physics question.

## 8. For Mimic

Mimic's `sage26` package follows SAGE26 at `11b1ae8` and has the same Regime 0 behaviour (`models/sage26/shared/sage_cooling_physics.h`, `mimic_sage_do_agn_heating_cgm`). Mimic draws the regime from a keyed per-snapshot deviate, so the flip-flop in §5.6 applies there too. Whatever SAGE26 adopts will be carried over at the planned SAGE26 v1.0 migration, together with the v0.95 merger fix.

## References

These are cited from memory and should be checked before use in a paper.

- Carr, Bryan, Fielding, Pandya & Somerville 2023, ApJ 949, 21 — the CGM cooling model used in Regime 0
- Croton et al. 2006, MNRAS 365, 11 — radio-mode AGN feedback and the heating radius
- Croton et al. 2016, ApJS 222, 22 — SAGE
- Dekel & Birnboim 2006, MNRAS 368, 2 — the shock mass and cold and hot accretion modes
- Geha, Blanton, Yan & Tinker 2012, ApJ 757, 85 — quenched fraction of isolated dwarfs
- Ponman, Cannon & Navarro 1999, Nature 397, 135 — entropy excess in group and cluster atmospheres
- Voit 2005, Rev. Mod. Phys. 77, 207 — review of the thermodynamics of cluster gas
- Voit, Bryan, O'Shea & Donahue 2015, ApJL 808, L30 — precipitation-regulated star formation, as SAGE26 cites it

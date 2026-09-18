# Orphan (Type 2) galaxy analysis

How SAGE26 treats orphan satellites, and what actually happens to them in a run.

```
figures/   fig1_orphan_fates.png     fates, mass split, mass function, disrupted fraction vs mass
           fig2_orphan_timing.png    redshift evolution, DF clock, baryon transfer, run-to-run robustness
           fig3_ics_budget.png       f_ICS vs halo mass, ICS provenance, growth history, threshold test
           fig4_literature_comparison.png  SAGE26 vs Henriques & Thomas 2010 / Contini et al. 2014
           fig5_massive_quiescent.png      central/satellite split of the massive quiescent population (orphans counted as satellites)
data/      orphan_events_*.csv       one row per orphan merger/disruption event
           orphan_analysis.json      reduced summary statistics (input to the figures)
           ics_budget.json           z=0 ICS budget for three ThresholdSatDisruption runs
scripts/   analyse_orphan_events.py  CSV  -> orphan_analysis.json
           plot_orphan_figures.py    JSON -> fig1, fig2
           plot_ics_budget.py        JSON -> fig3
           plot_literature_comparison.py   JSON + CSV -> fig4
           plot_massive_quiescent.py       model outputs -> fig5 (set GATE_RUN to the gate-on model file)
           core_build_model.instrumented.c   the instrumented source that produced the CSVs
```

## Provenance

Runs are mini-Millennium (`input/millennium.par`, 8 tree files, 64 snapshots):

- `orphan_events_fiducial.csv` — `input/millennium.par` as committed (`CGMrecipeOn=1`, `FeedbackFreeModeOn=1`)
- `orphan_events_vanilla.csv` — `input/millennium_vanilla.par` (`CGMrecipeOn=0`, FFB off)

The CSVs come from a **scratch build only** — `scripts/core_build_model.instrumented.c`
is `src/core_build_model.c` plus counters and an event dump, and was compiled in a
temporary directory. `src/` in this repo was never modified. To regenerate, copy
`src/` and `Makefile` elsewhere, drop the instrumented file over
`src/core_build_model.c`, adjust the hard-coded CSV path in `instr_ev()`, and run.

To rebuild the figures from the CSVs already here:

```sh
python3 scripts/analyse_orphan_events.py   # -> data/orphan_analysis.json
python3 scripts/plot_orphan_figures.py     # -> figures/fig1, fig2
python3 scripts/plot_ics_budget.py         # -> figures/fig3
python3 scripts/plot_literature_comparison.py   # -> figures/fig4
GATE_RUN=<gate-on model_0.hdf5> python3 scripts/plot_massive_quiescent.py   # -> figures/fig5
```

## Headline numbers (fiducial)

| | |
|---|---|
| Orphan events | 45,563 |
| Merged / disrupted to ICS | 24,467 (53.7%) / 21,096 (46.3%) |
| Stellar mass merged / to ICS | 1.0×10¹³ / 3.5×10¹³ M⊙ (23% / 77%) |
| Median M⋆, merged / disrupted | 1.3×10⁷ / 1.9×10⁸ M⊙ |
| Orphans surviving a snapshot | 0 |
| Type 1 satellites dying while still Type 1 | 88 (0.2% of all events) |
| Median DF time left when subhalo lost (disrupted) | 3.6 Gyr |

Type 0 → 2 demotions merge 100% of the time (`MergTime` is forced to 0 at
demotion, `src/core_build_model.c:295-301`); Type 1 → 2 demotions disrupt to ICS
93% of the time. Orphans are a single-snapshot transient — they are always
removed before the write step, so `Type == 2` never appears in the output
catalogue.

Note: `*.png` and `*.txt` are in `.gitignore`, so the figures here are local
artefacts and are not tracked.

## The ICS connection

`disrupt_satellite_to_ICS()` (`src/model_mergers.c:830`) is the **only** source of
intra-cluster stars in the model, and 99.7% of its calls act on orphans. At z = 0
that gives f_ICS = ICS/(ICS + M⋆) = 24% globally and **55–65% in 10^13–10^14 M⊙
groups**, against an observed ICL fraction of roughly 10–40%.

`ThresholdSatDisruption` cannot moderate it. `currentMvir` for an orphan ramps to
exactly 0 on the final substep (`src/core_build_model.c:521`), so the
merger/disruption test always passes regardless of the threshold:

| ThresholdSatDisruption | global f_ICS | f_ICS (10^13–10^14 M⊙) |
|---|---|---|
| 0.0 ("never disrupt") | 22.8% | 59.6% |
| 1.0 (fiducial) | 24.0% | 61.4% |
| 10.0 | 36.9% | 85.2% |

## Published prescriptions that fix this

**Henriques & Thomas (2010)**, MNRAS 403, 768 (`mnras0403-0768.pdf`) — closest to
SAGE26's architecture. Keeps orphans and the DF clock but replaces all-or-nothing
destruction with *continuous partial* tidal stripping:

- track the orbital radius r_sat as the orphan decays inward
- tidal radius R_t ≈ (1/√2)(σ_sat/σ_halo) r_sat  (isothermal, circular orbit)
- strip only the mass outside R_t, using M_disc(<R) = M[1 − (1+R/R_d)e^(−R/R_d)]
  and M_bulge(<r) = M r²/(r²+a²) with a = 0.56 R_b
- merge when r_sat < R_central + R_sat — a physical criterion, not clock expiry
- result: f_ICL ≈ 18% above 10¹³ M⊙, ≈ 7% at 10¹² M⊙
- side effect: stripped satellites feel less drag, so merger times lengthen 10–20%

**Contini et al. (2014)**, MNRAS 437, 3787 (`stt2174.pdf`) — compares three
prescriptions on the same base model:

- *Disr.* (Guo+11): orphans only, but disrupt **only if** ρ_DM,halo(R_peri) > ρ_sat
  = M_sat/R_half³, with R_peri from energy + angular momentum conservation in an
  isothermal potential. Compact satellites survive. Best match to Gonzalez+07.
- *Tid.*: R_t = (M_sat/3M_DM,halo)^(1/3) D, applied to **both** type 1 and type 2;
  strip the shell outside R_t, full disruption only if R_t < bulge radius; disc
  scalelength reset to R_t/10 after each episode.
- *Cont. Strip.* (Villalobos+12 fitting formula): overpredicts ICL+BCG.
- Plus a **merger channel**: 20% of a merging satellite's stars become unbound
  (+25% ICL). SAGE26 currently puts 100% of them into the bulge.
- All give f_ICL ≈ 20–40% with **no halo-mass trend**, bulk from M⋆ ~ 10¹⁰⁻¹¹
  donors, formed below z ~ 1.
- Caveat that applies directly here: ICL fractions are resolution-dependent —
  coarser sims make more type 2 galaxies and give 30–40% more ICL, so predictions
  "should be regarded as an upper limit".

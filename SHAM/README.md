# SHAM: theoretical stellar-mass and black-hole-mass functions on the Millennium trees

Subhalo abundance matching (SHAM) on the full Millennium merger trees
(`input/millennium/trees/trees_063.*`, 512 files), used to ask: **if black holes
and galaxies simply follow the observed/inferred SMF and BHMF, where do they land
on the f_BH = M_BH/M_* vs M_BH plane, and can they ever reach the LRD
observations?** This is independent of SAGE26 itself; SAGE26 galaxies are only
drawn underneath for comparison.

## Files

| file | role |
|---|---|
| `sham_trees.py` | reads the LHalo binary trees, builds Vpeak for every (sub)halo (parallel over files) |
| `sham_benchmarks.py` | the target SMF and BHMF curves, plus the scatter de-convolution |
| `run_sham.py` | does the matching + scatter, writes `output/sham_snapNN.npz` |
| `plot_sham_fbh.py` | f_BH vs M_BH figure (panel b of `plotting/bh_lrd_analysis.py`), closure figure, text summary |

## Running

```bash
# 1. match (reads all 512 tree files once, then caches the per-snapshot halo lists)
python SHAM/run_sham.py --z 3 5 6.7 --nproc 8
# 2. plot
python SHAM/plot_sham_fbh.py            # SAGE26 grey points drawn if output/millennium/model_*.hdf5 exists
python SHAM/plot_sham_fbh.py --no-sage  # SHAM only
```

Outputs: `SHAM/plots/sham_fbh_mbh_snapNN.png`, `sham_closure_snapNN.png`,
`sham_summary_snapNN.txt`. `--z` picks the nearest Millennium snapshot (printed);
`--snaps` takes snapshot numbers directly (27 = z 3.06, same default as
`bh_lrd_analysis.py`). The slow step is the tree sweep; its results are cached in
`SHAM/output/cache/`, so changing scatter, benchmark variant or property is
instant. Memory: one tree file (~0.2 GB) plus a few derived arrays per process.
`--max-files N` runs on the first N files only (volume is scaled by N/512).

## The method, step by step

### 1. Vpeak from the trees (`sham_trees.py`)
Each halo's matching property is **Vpeak = the maximum Vmax over its main-progenitor
branch up to that snapshot**. The main branch is followed through `FirstProgenitor`,
which in LHalo trees is the progenitor with the most massive history
(De Lucia & Blaizot 2007). Progenitors always sit at lower `SnapNum`, so one sweep
over snapshots in increasing order gives `Vpeak[h] = max(Vmax[h], Vpeak[FirstProgenitor[h]])`.
All (sub)haloes in the trees (centrals *and* subhaloes) are ranked together, which is
what makes this *sub*halo abundance matching. A peak-particle-count `Lpeak` (a Mpeak
proxy) is computed the same way for the resolution cut and for `--property lpeak`.

**Is Vpeak the best choice?** It is the most widely used and well-tested single
property: it ranks galaxies better than present-day Vmax or Mvir because stripping
lowers a subhalo's *current* mass/Vmax but not its history (Conroy et al. 2006), and
Vpeak reproduces SDSS clustering and conditional mass functions at least as well as
the alternatives (Reddick et al. 2013; Behroozi et al. 2013). UniverseMachine uses a
Vpeak-type property (Behroozi et al. 2019). "Best" is not unanimous: Mpeak does
comparably in some tests and all pure-SHAM models miss galaxy assembly bias
(Chaves-Montero et al. 2016). `--property vmax|lpeak` lets you test the sensitivity.
Millennium-specific caveats: only 64 snapshots, so the true peak can fall between
outputs (Vpeak is biased slightly low), and Vmax of a ~20-50 particle subhalo is noisy.

The tree files carry Vmax (and `Len`), so no extra halo catalogue is needed.

### 2. Resolution limit
Haloes enter the ranking only above the Vpeak where >= 90% of the catalogued haloes
have a peak particle count >= `--lmin` (default 50); below that the list is
incomplete and the cumulative number density would be wrong. The limit is printed per
snapshot. At z ~ 6 this is roughly log M* ~ 8.4, i.e. **Millennium cannot resolve the
low-mass hosts of many LRDs** (M* ~ 10^7.5-10^9); the SHAM black holes therefore only
cover the more massive end. This is a real limitation, not a bug.

### 3. Abundance matching (`run_sham.py`)
Sort haloes by Vpeak, highest first; the i-th halo has cumulative number density
`n_i = (i - 0.5) / V`, with `V = (500/0.73)^3 * (files found / 512)` Mpc^3 (physical
Mpc^3, the same convention as `allresults-history.py`/`allresults-blackholes.py`;
tree files are non-overlapping sub-volumes). The stellar mass solves
`N_SMF(> M*) = n_i` and the black-hole mass independently solves
`N_BHMF(> M_BH) = n_i`, where `N(>M) = integral of phi dlogM` above M
(Kravtsov et al. 2004; Vale & Ostriker 2004; Conroy et al. 2006). Both quantities are
therefore monotonic in Vpeak before scatter: the zero-scatter curve in the plot is
the abundance-matched M_BH-M_* relation implied by the two benchmark functions alone.
Haloes whose n exceeds the BHMF at log M_BH = 4 (lowest TRINITY bin) host BHs outside
the table and are dropped from the f_BH plane.

### 4. Benchmark curves (`sham_benchmarks.py`)
Same curves the plotting scripts use:
* **BHMF**: TRINITY (Zhang et al., TRINITY IV, arXiv:2309.07210;
  framework in Zhang et al. 2023, MNRAS 518, 2123) tables `data/bh/fig4_bhmf_z*.txt`
  (z = 0.1, 1, 2, 4, 6, 8), interpolated linearly in log phi between tables. `--bhmf-variant
  lo|hi` swaps in the 16th/84th percentile curves to bracket the model uncertainty.
  Note TRINITY is a *model-inferred* BHMF (it includes all BHs, not only active ones,
  and at z > 4 is largely extrapolated from quasar/AGN constraints), not a measurement.
* **SMF**: the observational compilation of `plotting/paper_plots.py`
  (`_load_smf_grid_observations`: Baldry+08, Muzzin+13, Santini+12, Thorne+21, Wright+18,
  COSMOS-Web, Song+16, Stefanon+21, Navarro-Carrera+23, Weibel+24, Kikuchihara+20,
  Harvey+25, with the h and IMF conversions defined there). Those are points, so the
  ones within +/- max(0.3, 0.15(1+z)) of the target redshift are pooled and fit with a
  Schechter (1976) function (single for z >= 3, double with shared M* for z < 3) by
  weighted least squares in log phi (errors floored at 0.1 dex). This gives a smooth,
  integrable curve. Pooling neighbouring redshift bins smears evolution slightly, and
  the fit is *extrapolated* below the lowest data point (reported per snapshot; BHs
  hosted there are drawn as open symbols). The compiled datasets disagree at the
  0.2-0.5 dex level at high z (chi2_red ~ 3-4 in the fits), so treat the SMF as
  uncertain at that level.

### 5. Scatter (simple, log-normal)
Default: sigma_* = 0.15 dex (typical SHAM value; Behroozi et al. 2010 / Reddick et
al. 2013 find ~0.15-0.2) and sigma_BH = 0.30 dex (about the intrinsic scatter of local
BH-host relations; Kormendy & Ho 2013), drawn independently (`--rho` correlates
them). Adding scatter to a matched relation broadens the output functions (the
steep high-mass tail is inflated by Eddington bias), so the benchmark is first
**de-convolved** with the Gaussian (Richardson 1972; Lucy 1974 iteration, 200 steps,
edges padded with the local slope), the matching is done on that intrinsic curve,
and the scatter is applied afterwards. The model SMF/BHMF then reproduce the
benchmarks by construction; `sham_closure_snapNN.png` verifies this (the deficit at the
low-mass end is the halo resolution limit, marked in the plot).

### 6. Plot (`plot_sham_fbh.py`)
Panel (b) of `bh_lrd_analysis.py`: f_BH vs M_BH with the f_BH = 3% LRD threshold
(Chen & Mo 2026, as used there), M_BH/M_* = 0.1, and the LRD/broad-line-AGN
observations at that redshift (Pang+26, Kocevski+23/+25, Harikane+23, Maiolino+23,
Taylor+25 from `lrd_literature_data.py`). Blue = SHAM black holes (points, 68/95/99.7%
KDE contours, median and 16-84% band); black line = zero scatter; grey = SAGE26. The
text summary lists, per M_BH bin, the number density of SHAM BHs and of those above
f_BH = 3%, and the maximum f_BH reached.

## Things to keep in mind when reading the result
* SHAM black holes are the *typical* BH of a given halo, assuming BH mass is a
  monotonic function of Vpeak. LRDs are rare and apparently over-massive for their
  hosts; if SHAM cannot reach them even with `--bhmf-variant hi` and a larger
  `--sigma-bh`, it says the observed objects must sit far in the tail of (or outside)
  a single-property relation, not that the data are wrong.
* Millennium resolution (above) removes the lowest-M_* hosts, which are exactly where
  f_BH would be highest at fixed M_BH. Where the plot shows no SHAM BH at low M_*,
  that is incompleteness, not absence. A higher-resolution simulation (e.g. miniUchuu)
  would test this.
* The TRINITY BHMF and SMF are both in Mpc^-3 with the h conventions of the repo's
  plotting scripts; small h mismatches between datasets are not corrected here.
* `plotting/bh_lrd_analysis.py` falls back to a hard-coded snapshot->z table
  (`MILLENNIUM_SNAP_TO_Z`) when the HDF5 header lacks redshifts; that table does not
  match `millennium.a_list` (e.g. snap 10 is z = 11.9, not 5.48). With SAGE HDF5
  output the header values are used, so it only matters for the fallback.

## References
* Behroozi, Conroy & Wechsler 2010, ApJ 717, 379 (SHAM with scatter, Eddington bias)
* Behroozi, Wechsler & Conroy 2013, ApJ 770, 57 (SMHM relation, Vpeak-type matching)
* Behroozi et al. 2019, MNRAS 488, 3143 (UniverseMachine)
* Chaves-Montero et al. 2016, MNRAS 458, 2621 (SHAM vs hydro, assembly bias)
* Chen & Mo 2026, arXiv:2605.31077 (LRD selection used in `bh_lrd_analysis.py`)
* Conroy, Wechsler & Kravtsov 2006, ApJ 647, 201 (SHAM, Vmax at accretion)
* De Lucia & Blaizot 2007, MNRAS 375, 2 (Millennium trees, main progenitor)
* Kormendy & Ho 2013, ARA&A 51, 511 (BH-host relations, scatter)
* Kravtsov et al. 2004, ApJ 609, 35 (subhalo-galaxy matching)
* Lucy 1974, AJ 79, 745; Richardson 1972, JOSA 62, 55 (iterative de-convolution)
* Reddick, Wechsler, Tinker & Behroozi 2013, ApJ 771, 30 (Vpeak best single property)
* Schechter 1976, ApJ 203, 297
* Springel et al. 2005, Nature 435, 629 (Millennium Simulation); Springel et al. 2001,
  MNRAS 328, 726 (SUBFIND)
* Vale & Ostriker 2004, MNRAS 353, 189
* Zhang, Behroozi et al., TRINITY IV, arXiv:2309.07210; Zhang et al. 2023, MNRAS 518, 2123
* SMF datasets: see `plotting/paper_plots.py` (`_load_smf_grid_observations`) for the
  per-dataset references; LRD data: `plotting/lrd_literature_data.py`
* SAGE: Croton et al. 2016, ApJS 222, 22

Please check the bibliographic details (volume/page numbers) against ADS before
publication; the author-year-topic attributions above are the part I am confident in.

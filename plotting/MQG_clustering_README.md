# MQG clustering bias — analysis + handoff notes

`mqg_clustering_bias.py` measures the large-scale clustering bias of massive
quiescent galaxies (MQGs) through cosmic time and compares it to the
Tinker et al. (2010) halo bias.

## What it does
1. Selects MQGs per snapshot: `log10(M*/Msun) > --min-logmstar` and
   `sSFR < ssfr0 * E(z)` (evolving quiescence floor; default `ssfr0 = 1e-11 /yr`).
2. Measures `b(z) = sqrt(<xi_gg / xi_mm>)` over `r = 5–25 Mpc/h` from the galaxy
   positions (`Posx/y/z`): `xi_gg` from **Corrfunc** (periodic `theory.xi`),
   `xi_mm` the linear matter correlation function from **colossus**.
   Errors via a delete-one sub-cube jackknife.
3. Compares to Tinker+10 `b(Mvir, z)` (colossus) evaluated at the sample's host
   masses. Plots bias vs both `CentralMvir` (host) and `Mvir` (subhalo).

## Dependencies
`Corrfunc`, `colossus`, `numpy`, `scipy`, `h5py`, `matplotlib`, `astropy`.
Corrfunc is the only one that compiles (needs a C compiler + GSL).

On an Lmod (module) cluster -- no conda (module lines pinned for Ngarrgu
Tindebeek / tooarrana, gcc/12.3.0 toolchain):
    # setup on the LOGIN node -- compute nodes are air-gapped (no internet for pip):
    bash plotting/setup_clustering_env.sh           # venv on modules; pip adds Corrfunc+colossus+h5py
    # run the analysis as a batch job (handles resources/modules/venv itself):
    sbatch plotting/run_clustering_ozstar.sh
    # OPTIONAL pre-flight only: check the login-built Corrfunc runs on a compute node
    #   sinteractive ... ; source $HOME/envs/sage-clustering/bin/activate
    #   python -c "from Corrfunc.theory import xi; print('ok')"
numpy/scipy/matplotlib/astropy come from modules (`scipy-bundle`, `matplotlib`,
`astropy`); `gsl` is loaded so Corrfunc compiles; only Corrfunc, colossus, h5py
are pip-installed. If the toolchain versions change, `module load gcc/<ver>` then
`module avail | grep -iE 'gsl/|python/|scipy-bundle/|matplotlib/|astropy/'` and
update the load line in both scripts. Build Corrfunc on a COMPUTE node so its
SIMD matches the run nodes. `requirements-clustering.txt` is the full portable
list for non-module machines. The script forces the Agg backend and drops LaTeX
text when no `latex` is on PATH, so it runs headless.

## Run
    python plotting/mqg_clustering_bias.py \
        --model output/<sim> --redshifts 0 0.99 1.91 2.83 3.87 \
        --mvir-lim 11 15 --bias-lim 0 8.5
Cosmology (Om, OL, h, box) is read from the SAGE header; pass `--sigma-8`/`--n-s`
to match the simulation (defaults are Uchuu/Planck-15). Observational overlay:
`data/clustering/quiescent_bias_obs.dat` (edit or `--obs-file none`).

Sample selection -- quiescence is always the Donnari sSFR floor; "massive" is
defined by ONE of these (an explicit flag always overrides the default):
- `--number-density n` [(Mpc/h)^-3] (**RECOMMENDED DEFAULT**, n = 1e-4): the
  N = n*box^3 most massive quiescent galaxies -- a constant comoving abundance.
  This picks the massive, central-dominated host population (plain `Mvir` ~=
  `CentralMvir`) that traces the clean declining bias-vs-mass trend, with plenty
  of objects for a robust `xi_gg` on a big box. On miniUchuu (400 Mpc/h) this is
  N ~ 6400; on Uchuu it scales with the volume. Go lower (~1.5e-5, N ~ 1000 on
  miniUchuu) for the most extreme massive-central trend (the number-scan sweet
  spot); below ~1e-5 the sample becomes shot-noise limited and the trend breaks
  down. A value >=1 is instead read as a plain COUNT (e.g. `--number-density
  500`). Applied automatically when no other selection flag is passed.
- `--top-percent P`: the most massive P%% of ALL galaxies per snapshot
  (per-snapshot M* percentile), then quiescent. Broad, tends satellite-heavy ->
  plain `Mvir` a "J". Note P is of the whole galaxy population, so even a small P
  is not necessarily BCG-scale.
- `--min-logmstar X`: fixed stellar-mass cut log10(M*) > X.
- `--match-highz-count`: count the quiescent galaxies at the HIGHEST requested
  redshift and select that same number (most massive quiescent) at every z --
  abundance-matched to the high-z quiescent count. The sample's mass scale is set
  by that count, i.e. by how high the top redshift is (higher top z -> fewer
  quiescent -> smaller N -> more massive matched sample).

All selection modes now measure the bias whenever a snapshot has >=2 MQGs (a
WARNING is printed below 50 instead of skipping); only <2 is skipped.

## Key result (established on microUchuu; structural, so volume-independent)
- **bias vs plain `Mvir` is a "J"**: the z=0 median `Mvir` hooks back to LOW mass,
  because at z=0 the quenched sample is flooded by stripped satellites (low
  subhalo `Mvir`) and accumulated low-mass quenched centrals.
- **bias vs `CentralMvir` is an "L"**: clean decline, z=0 at high mass, because
  satellites carry their massive host mass.
- This shape is set by the MASS FIELD, not the physics. Verified across 4 code
  versions (May–Jul 2026), vanilla, all single physics toggles
  (CGM/FIRE/FFB/RPS/H2-SF), all-on, and all-on minus the over-quenchers. None
  turn `Mvir` into an L. (For centrals, `Mvir == CentralMvir` exactly.)
- The physics that most affects *which* galaxies quench: `SFprescription=1`
  (H2-based SF) over-quenches low-mass centrals (quenched frac 0.22 -> 0.53),
  `FIREmode` secondary; `FFB`/`RPS` negligible. But this changes the low-mass
  flood, not the J/L shape.

## Decisive test to run on the big boxes (miniUchuu / Uchuu)
microUchuu (100 Mpc/h) tops out at ~10^13.5 and can't reach the cluster masses
(z=0 ~ 10^14) or high-z massive-quiescent counts of the reference plots. On a
big box, print at z=0 for the MQG sample:
- median plain `Mvir` vs median `CentralMvir`.
If plain `Mvir` reaches ~14 -> sample is massive-central-dominated and `Mvir` can
be an L. If plain `Mvir` stays ~12–13 (J) while `CentralMvir` ~14 (L), then any
"declining trend in Mvir" at ~14 was actually host mass (`CentralMvir`).

## Robust bias: cross-correlation with the halo trees (`mqg_tree_cross.py`)
The MQG *autocorrelation* is shot-noise limited when the sample is sparse — the
high-z / rare-selection regime where a fixed count over-reaches the massive-
quiescent supply. `mqg_tree_cross.py` cross-correlates the sparse MQG sample
against the *dense* halo field read straight from the LHaloTree **binary** files:

    b_g = xi_gh / sqrt(xi_hh * xi_mm)     (averaged over the linear fit range)

The cross term pairs the few MQGs against ~10^6 halos, so its noise is set by the
dense sample, not the MQG count. It also produces:
- a **measured halo-bias baseline** `b_h = sqrt(xi_hh/xi_mm)` — an apples-to-apples
  check on the analytic Tinker+10 curve (same box, cosmology, estimator, mdef);
- a **mass-definition comparison** — MQG bias vs halo mass in `M_vir` (tophat),
  `M_200c`, `M_200m`. The trees carry all three (`Group_M_TopHat200/Crit200/Mean200`);
  the MQG sample's mass is converted `M_vir -> 200c/200m` with colossus using the
  SAGE per-galaxy `Concentration`. Shows the Tinker mdef systematic (~0.1 dex).

Run (needs the trees SAGE was run on; same `TreeName.n` binary files):
    python plotting/mqg_tree_cross.py --model output/miniuchuu \
        --tree-dir <SimulationDir> --tree-name miniUchuu_STC --tree-nfiles 8 \
        --redshifts 0 0.99 1.91 2.83 3.87 --number-density 1e-5 --nthreads 8
Selection flags mirror `mqg_clustering_bias.py`. `--min-len` (default 20) sets the
halo resolution cut for the field; `--halo-subsample` (default 100000) caps the
field size so the jackknife stays tractable (a dense subsample still vastly
outnumbers the MQGs). The binary struct layout (104 B) is taken from
`src/core_simulation.h` (`struct halo_data`); masses are 10^10 Msun/h, Pos Mpc/h.
This does NOT remove the volume/supply limit — it makes the measurement of the
objects that DO exist far less noisy. Still needs a big box for high z.

### Halo-mass-binned mode (`--mass-bins`) -- NO count/density knob
The number/density selection sets the sample's mass scale through a chosen count,
which over-reaches the massive-quiescent supply at high z. `--mass-bins` removes
that knob entirely: it selects ALL quiescent galaxies (Donnari floor only) and
bins them by their own `Mvir`, measuring the cross-correlation bias per bin. The
count per bin then falls out of the halo mass function -- it is not chosen -- and
a bin that runs out of quiescent galaxies at high z is simply skipped
(`--min-per-bin`, default 5), not padded with satellites.

    python plotting/mqg_tree_cross.py --model output/miniuchuu \
        --tree-dir <SimulationDir> --tree-name miniUchuu_STC --tree-nfiles 8 \
        --redshifts 0 0.99 1.91 2.83 --mass-bins --nthreads 8
Pass edges (log10 Msun/h) after `--mass-bins`, or the flag alone for the defaults
(12.0 12.5 13.0 13.5 14.0 14.5). Output: `mqg_tree_cross_massbins.<fmt>` -- one
b(Mvir) track per redshift vs Tinker+10, with the observational compilation
(`--obs-file`, default `data/clustering/quiescent_bias_obs.dat`) overlaid as
z-coloured stars. This is the MODEL-VALIDATION figure: it tests b(Mvir,z) against
theory (Tinker) AND data, the correct external benchmark -- not against any prior
figure. (Obs is physical Msun; the plot shifts it by +log10(h) onto the Msun/h
axis.) This is the recommended way to present the result: a physical bias-vs-Mvir
relation with no arbitrary abundance. Errors jackknife the galaxy sample only
(dense halo field held fixed).

## Physics toggle sweep recipe
Base `input/<sim>_vanilla.par` (physics off); enable one toggle at a time to a
separate output folder; measure. No source changes needed — all par-driven.

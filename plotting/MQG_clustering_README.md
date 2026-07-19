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
    sinteractive                                    # build Corrfunc on a compute node
    bash plotting/setup_clustering_env.sh           # venv on modules; pip adds Corrfunc+colossus+h5py
    source $HOME/envs/sage-clustering/bin/activate
    sbatch plotting/run_clustering_ozstar.sh        # SLURM run template
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

## Physics toggle sweep recipe
Base `input/<sim>_vanilla.par` (physics off); enable one toggle at a time to a
separate output folder; measure. No source changes needed — all par-driven.

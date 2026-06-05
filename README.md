<p align="center">
  <img src="SAGElogo.png" width="300" alt="SAGE26 logo"/>
</p>

# SAGE26 — Semi-Analytic Galaxy Evolution

[![Documentation Status](https://readthedocs.org/projects/sage26/badge/?version=latest)](https://sage26.readthedocs.io/en/latest/)

SAGE26 is a C99 semi-analytic code for modelling galaxy formation in a cosmological
context. It is a major update to [Croton et al. (2016)](https://arxiv.org/abs/1601.04709),
adding a two-regime CGM model, FIRE stellar feedback, feedback-free burst (FFB) galaxies,
multiple H2-based star formation prescriptions, and extended bulge/ICS tracking.

SAGE reads N-body merger trees, evolves galaxies through cosmic time using
semi-analytic prescriptions, and writes galaxy catalogues in HDF5 or binary format.
It runs on any simulation whose trees are in a supported format and contain a minimum set
of halo properties. Test trees for the
[Mini-Millennium Simulation](http://arxiv.org/abs/astro-ph/0504097) are provided.

---

## What is new in SAGE26

| Feature | Parameter | Reference |
|---------|-----------|-----------|
| Two-regime CGM model | `CGMrecipeOn` | Dekel & Birnboim (2006), Voit (2015) |
| FIRE stellar feedback | `FIREmodeOn` | Muratov et al. (2015) |
| Feedback-free burst galaxies | `FeedbackFreeModeOn` | Li et al. (2024), Boylan-Kolchin (2025) |
| NFW/beta CGM density profiles | `CGMDensityProfile` | — |
| 7 H2 star formation prescriptions | `SFprescription` | BR06, KMT09, KD12, K13, GD14, S25 |
| Separate merger/instability bulge tracking | `BulgeSizeOn` | Tonini et al. (2016) |
| ICS assembly tracking | `TrackICSAssembly` | — |
| Full star formation history arrays | `SaveFullSFH` | — |
| Concentration | `ConcentrationOn` | Ishiyama+21 lookup table, Vmax/Vvir, Vmax/Vvir with subhalo infall freeze |
| ConsistentTrees, Genesis, Gadget-4 tree readers | `TreeType` | — |
| HDF5 output + libsage.so for Python/PSO | `OutputFormat` | — |

---

## Install

### Dependencies

| Package | Required | Notes |
|---------|----------|-------|
| C99 compiler (gcc or clang) | Yes | |
| [GSL](https://www.gnu.org/software/gsl/) | Yes | required for tests |
| [HDF5](https://www.hdfgroup.org/) | Optional | HDF5 tree reading and output |
| MPI | Optional | parallel execution |

### Build

```bash
git clone https://github.com/MBradley1985/SAGE26.git
cd SAGE26
make                   # serial build -- produces ./sage and libsage.so
make USE-MPI=yes       # MPI-parallel build (switches compiler to mpicc)
make USE-HDF5=yes      # enable HDF5 support
make MEM-CHECK=yes     # address/UB sanitizers for debugging (gcc only)
make clean             # remove all build artefacts
```

---

## Quickstart

```bash
./first_run.sh                          # create output dirs, download Mini-Millennium trees
./sage input/millennium.par             # run the model (serial)
mpirun -np 4 ./sage input/millennium.par  # run in parallel
python plotting/allresults-local.py     # z=0 diagnostic plots
python plotting/allresults-history.py   # multi-redshift diagnostics
```

Full details on the parameter file format, output format, and all physics switches are in
[`docs/parameters.md`](docs/parameters.md).

---

## Physics options

Every switch below is parsed by [`src/core_read_parameter_file.c`](src/core_read_parameter_file.c).
Full descriptions, defaults, and the model-parameter (non-switch) list are in
[`docs/parameters.md`](docs/parameters.md).

### Star formation (`SFprescription`)

| Value | Prescription |
|-------|-------------|
| 0 | Croton et al. (2006) original |
| 1 | Blitz & Rosolowsky (2006) H2 |
| 2 | Somerville et al. (2025) SFR |
| 3 | Somerville et al. (2025) SFR + H2 |
| 4 | Krumholz & Dekel (2012) |
| 5 | Krumholz, McKee & Tumlinson (2009) |
| 6 | Krumholz (2013) |
| 7 | Gnedin & Draine (2014) |

### AGN feedback (`AGNrecipeOn`)

| Value | Mode |
|-------|------|
| 0 | Off |
| 1 | Empirical (Croton+2016 radio mode) |
| 2 | Bondi-Hoyle accretion |
| 3 | Cold-cloud accretion |

### Supernova feedback (`SupernovaRecipeOn`)

| Value | Mode |
|-------|------|
| 0 | Off |
| 1 | Croton+2016 reheating/ejection |

### Reionization (`ReionizationOn`)

| Value | Mode |
|-------|------|
| 0 | Off |
| 1 | Kravtsov+2004 analytic suppression of infall |

### Disk instability (`DiskInstabilityOn`)

| Value | Mode |
|-------|------|
| 0 | Off |
| 1 | Toomre criterion drives bulge growth, instability starbursts, BH growth |

### Two-regime CGM model (`CGMrecipeOn`)

Galaxies below the Dekel & Birnboim (2006) shock mass are placed in Regime 0 (CGM /
precipitation regime); those above are in Regime 1 (hot-halo / classical cooling regime).
Each regime uses a dedicated cooling recipe.

| Parameter | Values | Effect |
|-----------|--------|--------|
| `CGMrecipeOn` | 0/1 | 0=off (classical C16 cooling only); 1=on |
| `CGMDensityProfile` | 0–2 | 0=uniform; 1=NFW; 2=beta-profile (β=2/3) |

### FIRE stellar feedback (`FIREmodeOn`)

| Value | Mode |
|-------|------|
| 0 | Off |
| 1 | Muratov+2015 FIRE mass loading |

### Feedback-free burst galaxies (`FeedbackFreeModeOn`)

| Value | Mode |
|-------|------|
| 0 | Off |
| 1 | Li+2024 sigmoid |
| 2 | Boylan-Kolchin+2025 (Ishiyama+21 concentration) |

### Halo concentration (`ConcentrationOn`)

| Value | Method |
|-------|--------|
| 0 | Off |
| 1 | Ishiyama+2021 lookup table |
| 2 | V_max / V_vir from the simulation |
| 3 | V_max / V_vir with subhalo infall freeze for satellites |

### H2 star formation (auxiliary switches)

| Parameter | Values | Effect |
|-----------|--------|--------|
| `H2DiskAreaOption` | 1–3 | Disk area for H2 surface density: 1=π r_disk²; 2=π (3 r_disk)²; 3=2π r_disk² |
| `H2RadialIntegrationOn` | 0/1 | Radial ring integration for H2 fraction (slower, more accurate) |
| `H2RadialNBins` | int | Number of radial bins for ring integration |

### Mergers and ICS tracking

| Parameter | Values | Effect |
|-----------|--------|--------|
| `DynamicDisruptionSplit` | 0–2 | Route disrupted satellite mass: 0=all to ICS; 1=all to hot gas; 2=split via `FractionDisruptedToICS` |
| `TrackICSAssembly` | 0/1 | Record satellite disruption / accretion contributions to ICS |

### Output

| Parameter | Values | Effect |
|-----------|--------|--------|
| `OutputFormat` | string | `sage_hdf5` or `sage_binary` |
| `SaveFullSFH` | 0/1 | Track per-snapshot SFR history arrays |
| `NumOutputs` | int | Number of snapshot outputs; `-1` = all snapshots |

### Supported tree formats (`TreeType`)

`lhalo_binary`, `lhalo_hdf5`, `consistent_trees_ascii`, `consistent_trees_hdf5`,
`genesis_lhalo_hdf5`, `gadget4_hdf5`

---

## Tests

```bash
cd tests && make test               # build and run all suites
cd tests && make test_conservation  # conservation tests only (fastest)
cd tests && make quick              # single fastest check
bash tests/run_integration_tests.sh # full integration test (slower)
```

The regression baseline checks that output is bit-identical across 5380 datasets:

```bash
bash tests/regression_baseline.sh
```

After any physics change, recapture the baseline with:

```bash
python3 tests/regression_baseline.py capture input/millennium.par
```

---

## Parameter calibration (SAGE-PSO)

Automated parameter calibration via Particle Swarm Optimization is available as a separate
package: [SAGE-PSO](https://github.com/MBradley1985/SAGE-PSO). It drives `libsage.so` to
evaluate parameter samples against observational constraints (stellar mass functions,
star formation rates, etc.) and supports machine-learning emulators to accelerate
optimization.

---

## Citation

If you use SAGE26 in a publication, please cite:

```bibtex
@article{bradley2026sage26,
  author  = {Bradley, Michael and Croton, Darren J.},
  title   = {SAGE26 Paper I: Modelling the baryon cycle from cosmic dawn to the present day},
  journal = {in preparation},
  year    = {2026},
}
```

and the original SAGE paper:

```bibtex
@article{croton2016sage,
  author  = {Croton, D.~J. and Stevens, A.~R.~H. and Tonini, C. and
             Garel, T. and Bernyk, M. and Bibiano, A. and Hodkinson, L. and
             Mutch, S.~J. and Poole, G.~B. and Shattow, G.~M.},
  title   = {Semi-Analytic Galaxy Evolution (SAGE): Model Calibration and Basic Results},
  journal = {ApJS},
  year    = {2016},
  volume  = {222},
  pages   = {22},
  doi     = {10.3847/0067-0049/222/2/22},
  eprint  = {1601.04709},
}
```

Original SAGE is also available on [ascl.net](http://ascl.net/1601.006).

---

## Links

- [Documentation](https://sage26.readthedocs.io/en/latest/)
- [Parameter reference](https://sage26.readthedocs.io/en/latest/parameters.html)
- [Developer guide](docs/developer/README.md)
- [Changelog](CHANGELOG.md)
- [Contributing](CONTRIBUTING.md)
- [SAGE-PSO](https://github.com/MBradley1985/SAGE-PSO) — automated parameter calibration

---

## Authors and maintainers

- Michael Bradley ([@MBradley1985](https://github.com/MBradley1985)) — mbradley@swin.edu.au
- Darren Croton ([@darrencroton](https://github.com/darrencroton))

Questions and comments welcome via GitHub Issues or email.

---

## License

MIT — see [LICENSE](LICENSE).

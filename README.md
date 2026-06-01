# SAGE26 — Semi-Analytic Galaxy Evolution

SAGE26 is a C99 semi-analytic code for modelling galaxy formation in a
cosmological context, extending [Croton et al. (2016)](https://arxiv.org/abs/1601.04709)
with a two-regime CGM model, FIRE stellar feedback, feedback-free burst (FFB)
galaxies, and extended bulge/ICS tracking.

## Install

**Dependencies:**

| Package | Required | Notes |
|---------|----------|-------|
| C99 compiler (gcc/clang) | Yes | |
| [GSL](https://www.gnu.org/software/gsl/) | Yes | for tests |
| [HDF5](https://www.hdfgroup.org/) | Optional | enables HDF5 tree reading and output |
| MPI | Optional | parallel execution |

**Build:**

```bash
git clone https://github.com/MBradley1985/SAGE26.git
cd SAGE26
make                   # serial build — produces ./sage and libsage.so
make USE-MPI=yes       # MPI build
```

On a first clone, run `./first_run.sh` to create output directories and download
the Mini-Millennium test trees.

## Quickstart

```bash
./first_run.sh                         # create dirs, download Mini-Millennium trees
./sage input/millennium.par            # run the model
python plotting/allresults-local.py    # z=0 diagnostic plots
```

MPI run:

```bash
mpirun -np 4 ./sage input/millennium.par
```

Run the test suite:

```bash
make tests
```

Full details on physics options, parameter file format, output format, and tree
readers are in [`docs/`](docs/).

## Citation

If you use SAGE26 in a publication, please cite:

```bibtex
@article{bradley2026sage26,
  author  = {Bradley, Michael and Croton, Darren J.},
  title   = {{SAGE26}: A Two-Regime CGM Model for Semi-Analytic Galaxy Evolution},
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
  title   = {Semi-Analytic Galaxy Evolution ({SAGE}): Model Calibration and Basic Results},
  journal = {ApJS},
  year    = {2016},
  volume  = {222},
  pages   = {22},
  doi     = {10.3847/0067-0049/222/2/22},
  eprint  = {1601.04709},
}
```

## Links

- [Parameter reference](docs/parameters.md)
- [Developer guide](docs/developer/README.md)
- [SAGE-PSO](https://github.com/MBradley1985/SAGE-PSO) — automated parameter calibration
- [Original SAGE on ascl.net](http://ascl.net/1601.006)

## License

MIT — see [LICENSE](LICENSE).

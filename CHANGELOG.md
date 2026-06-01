# Changelog

## SAGE26 (2026) — Major release

Built on [Croton et al. (2016)](https://arxiv.org/abs/1601.04709).

### New physics

- **Two-regime CGM model** (`CGMrecipeOn`): galaxies are classified as CGM-regime
  (below the Dekel & Birnboim 2006 shock mass) or hot-halo regime. Each regime
  uses a dedicated cooling recipe. CGM-regime cooling uses the Voit (2015)
  precipitation criterion and a HeatingReservoir for AGN feedback with
  dynamical-time memory.
- **FIRE stellar feedback** (`FIREmodeOn`): FIRE-calibrated wind mass-loading and
  ejection efficiencies replace the fixed Croton+2016 values.
- **Feedback-free burst galaxies** (`FeedbackFreeModeOn`): implements the Li+2024
  and Boylan-Kolchin+2025 FFB criteria. Multiple sub-modes available (0–7).
- **NFW/beta-profile CGM density** (`CGMDensityProfile`): cooling in CGM-regime
  halos can use a uniform, NFW, or beta-profile gas distribution.
- **Extended bulge tracking**: merger-driven and instability-driven bulge
  components tracked separately (`MergerBulgeMass`, `InstabilityBulgeMass`,
  `MergerBulgeRadius`, `InstabilityBulgeRadius`). Radii follow Tonini+2016 eq. 15.
- **ICS assembly tracking** (`TrackICSAssembly`): records satellite disruption
  contributions to intracluster stars.

### New SF prescriptions (`SFprescription`)

| Value | Prescription |
|-------|-------------|
| 0 | Croton et al. (2006) original |
| 1 | Blitz & Rosolowsky (2006) H₂ |
| 2 | Somerville et al. (2025) SFR |
| 3 | Somerville et al. (2025) SFR + H₂ |
| 4 | Krumholz & Dekel (2012) |
| 5 | Krumholz, McKee & Tumlinson (2009) |
| 6 | Krumholz (2013) |
| 7 | Gnedin & Draine (2014) |

### New tree formats

ConsistentTrees ASCII and HDF5 (`consistent_trees_ascii`, `consistent_trees_hdf5`),
Genesis HDF5 (`genesis_lhalo_hdf5`), Gadget-4 HDF5 (`gadget4_hdf5`).

### Infrastructure

- HDF5 output format (`OutputFormat sage_hdf5`) with buffered writes.
- `libsage.so` shared library for Python bindings and PSO parameter calibration.
- Full star formation history arrays (`SaveFullSFH`).
- Regression baseline test (5380 datasets, bit-identical per dataset).

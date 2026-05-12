#!/usr/bin/env python

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
from scipy import interpolate
from scipy import stats
from scipy.integrate import quad
from scipy.ndimage import gaussian_filter
from random import sample, seed
import matplotlib.cm as cm
import pandas as pd

import warnings
warnings.filterwarnings("ignore")
try:
    from astropy.table import Table
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False
    print("Warning: astropy not available, observational data will not be loaded")

try:
    from colossus.cosmology import cosmology as _colossus_cosmology
    from colossus.halo import concentration as _colossus_conc
    _colossus_cosmology.setCosmology('custom_millennium', flat=True,
                                     H0=73.0, Om0=0.25, Ob0=0.045,
                                     sigma8=0.90, ns=1.0, relspecies=False)
    _HAS_COLOSSUS = True
except Exception:
    _HAS_COLOSSUS = False



# ========================== CONFIGURATION ==========================

# File paths
PRIMARY_DIR = './output/millennium/'
MINIUCHUU_DIR = './output/microuchuu/'
MODEL_FILE = 'model_0.hdf5'
OBS_DIR = './data/'

# Plotting (analysis choices — not simulation parameters)
OUTPUT_FORMAT = '.pdf'
DILUTE = 7500
SEED = 2222

# Analysis thresholds (not simulation parameters)
MIN_PARTICLES = 20     # minimum DM particles for a resolved halo (applied at load time)
MIN_MVIR_PARTICLES = 20  # independent halo-mass cut: require Mvir >= MIN_MVIR_PARTICLES * m_p
SSFR_CUT = -11.0       # log10(sSFR/yr^-1) dividing quiescent from star-forming

# Solar metallicity (Asplund et al. 2009)
Z_SUN = 0.0134

# Solar mass in grams (for MASS_CONVERT derivation)
_MSUN_CGS = 1.989e33


# --------------- HDF5 header reader ---------------

import glob as _glob_early


def _find_model_files_early(directory):
    """Minimal file discovery used during module init (before full I/O helpers)."""
    pattern = os.path.join(directory, 'model_*.hdf5')
    files = sorted(_glob_early.glob(pattern))
    if not files:
        single = os.path.join(directory, MODEL_FILE)
        if os.path.exists(single):
            files = [single]
    return files


def _read_sim_header(directory):
    """
    Read simulation parameters from the HDF5 header of the first model
    file found in *directory*.

    Returns a dict of parameters, or ``None`` if no model files exist.
    The ``volume_fraction`` key is the *total* fraction across all MPI
    files (summed ``frac_volume_processed``).
    """
    files = _find_model_files_early(directory)
    if not files:
        return None

    try:
        with h5.File(files[0], 'r') as f:
            sim = f['Header/Simulation']
            runtime = f['Header/Runtime']

            header = {
                'hubble_h':       float(sim.attrs['hubble_h']),
                'box_size':       float(sim.attrs['box_size']),
                'omega_matter':   float(sim.attrs['omega_matter']),
                'omega_lambda':   float(sim.attrs['omega_lambda']),
                'last_snap_nr':   int(sim.attrs['LastSnapshotNr']),
                'unit_mass_in_g': float(runtime.attrs['UnitMass_in_g']),
                'baryon_frac':    float(runtime.attrs.get('BaryonFrac', 0.17)),
                'redshifts':      list(f['Header/snapshot_redshifts'][:]),
                'output_snaps':   list(f['Header/output_snapshots'][:]),
                'particle_mass':  float(sim.attrs['particle_mass']),
            }

        # Sum frac_volume_processed across all MPI files to get the total
        total_fvp = 0.0
        for fp in files:
            with h5.File(fp, 'r') as f:
                total_fvp += float(f['Header/Runtime'].attrs['frac_volume_processed'])
        header['volume_fraction'] = total_fvp
    except Exception as e:
        print(f"Warning: could not read header from {directory}: {e}")
        return None

    return header


def _snap_for_z(redshifts, target_z):
    """
    Return the snapshot index of the last output snapshot whose redshift
    is >= *target_z*.  This reproduces the standard convention of choosing
    the snapshot just above the target redshift (e.g. z=4.179 for target 4).
    """
    neg_z = -np.array(redshifts)          # make increasing for searchsorted
    idx = int(np.searchsorted(neg_z, -target_z, side='right')) - 1
    return max(idx, 0)


# --------------- Primary simulation parameters (from HDF5) ---------------

_primary_hdr = _read_sim_header(PRIMARY_DIR)
if _primary_hdr is not None:
    HUBBLE_H         = _primary_hdr['hubble_h']
    BOX_SIZE         = _primary_hdr['box_size']
    VOLUME_FRACTION  = _primary_hdr['volume_fraction']
    VOLUME           = (BOX_SIZE / HUBBLE_H)**3 * VOLUME_FRACTION  # Mpc^3
    MASS_CONVERT     = _primary_hdr['unit_mass_in_g'] / _MSUN_CGS / HUBBLE_H
    OMEGA_M          = _primary_hdr['omega_matter']
    OMEGA_L          = _primary_hdr['omega_lambda']
    BARYON_FRAC      = _primary_hdr['baryon_frac']
    OMEGA_B          = BARYON_FRAC * OMEGA_M
    SNAPSHOT         = f"Snap_{_primary_hdr['last_snap_nr']}"
    REDSHIFTS        = _primary_hdr['redshifts']
    PART_MASS         = _primary_hdr['particle_mass']
    OUTPUT_DIR       = os.path.join(PRIMARY_DIR, 'plots/')

    # Snapshot aliases for key redshifts (derived from the redshift table)
    SNAP_Z0  = _snap_for_z(REDSHIFTS, 0.0)
    SNAP_Z1  = _snap_for_z(REDSHIFTS, 1.0)
    SNAP_Z2  = _snap_for_z(REDSHIFTS, 2.0)
    SNAP_Z3  = _snap_for_z(REDSHIFTS, 3.0)
    SNAP_Z4  = _snap_for_z(REDSHIFTS, 4.0)
    SNAP_Z5  = _snap_for_z(REDSHIFTS, 5.0)
    SNAP_Z7  = _snap_for_z(REDSHIFTS, 7.0)
    SNAP_Z10 = _snap_for_z(REDSHIFTS, 10.0)
else:
    # Fallback if primary HDF5 files are not available
    print("Warning: could not read primary model header — using hardcoded defaults")
    HUBBLE_H         = 0.73
    BOX_SIZE         = 62.5
    VOLUME_FRACTION  = 1.0
    VOLUME           = (BOX_SIZE / HUBBLE_H)**3 * VOLUME_FRACTION
    MASS_CONVERT     = 1.0e10 / HUBBLE_H
    OMEGA_M          = 0.25
    OMEGA_L          = 0.75
    BARYON_FRAC      = 0.17
    OMEGA_B          = 0.045
    SNAPSHOT         = 'Snap_63'
    REDSHIFTS        = [
        127.000, 79.998, 50.000, 30.000, 19.916, 18.244, 16.725, 15.343,
         14.086, 12.941, 11.897, 10.944, 10.073,  9.278,  8.550,  7.883,
          7.272,  6.712,  6.197,  5.724,  5.289,  4.888,  4.520,  4.179,
          3.866,  3.576,  3.308,  3.060,  2.831,  2.619,  2.422,  2.239,
          2.070,  1.913,  1.766,  1.630,  1.504,  1.386,  1.276,  1.173,
          1.078,  0.989,  0.905,  0.828,  0.755,  0.687,  0.624,  0.564,
          0.509,  0.457,  0.408,  0.362,  0.320,  0.280,  0.242,  0.208,
          0.175,  0.144,  0.116,  0.089,  0.064,  0.041,  0.020,  0.000,
    ]
    PART_MASS         = 0.0860657 * 1.0e10 / HUBBLE_H
    OUTPUT_DIR = './output/millennium/plots/'
    SNAP_Z0  = 63
    SNAP_Z1  = 39
    SNAP_Z2  = 32
    SNAP_Z3  = 27
    SNAP_Z4  = 23
    SNAP_Z5  = 20
    SNAP_Z7  = 16
    SNAP_Z10 = 12


# --------------- miniUchuu simulation parameters (from HDF5) ---------------

_miniuchuu_hdr = _read_sim_header(MINIUCHUU_DIR)
if _miniuchuu_hdr is not None:
    MINIUCHUU_HUBBLE_H        = _miniuchuu_hdr['hubble_h']
    MINIUCHUU_BOX_SIZE        = _miniuchuu_hdr['box_size']
    MINIUCHUU_VOLUME_FRACTION = _miniuchuu_hdr['volume_fraction']
    MINIUCHUU_VOLUME          = (MINIUCHUU_BOX_SIZE / MINIUCHUU_HUBBLE_H)**3 * MINIUCHUU_VOLUME_FRACTION
    MINIUCHUU_MASS_CONVERT    = _miniuchuu_hdr['unit_mass_in_g'] / _MSUN_CGS / MINIUCHUU_HUBBLE_H
    MINIUCHUU_FIRST_SNAP      = min(_miniuchuu_hdr['output_snaps'])
    MINIUCHUU_LAST_SNAP       = max(_miniuchuu_hdr['output_snaps'])
    MINIUCHUU_REDSHIFTS       = _miniuchuu_hdr['redshifts']
    MINIUCHUU_PART_MASS       = _miniuchuu_hdr['particle_mass']
else:
    # Fallback if miniUchuu HDF5 files are not available
    MINIUCHUU_HUBBLE_H        = 0.677
    MINIUCHUU_BOX_SIZE        = 400.0
    MINIUCHUU_VOLUME_FRACTION = 0.3
    MINIUCHUU_VOLUME          = (MINIUCHUU_BOX_SIZE / MINIUCHUU_HUBBLE_H)**3 * MINIUCHUU_VOLUME_FRACTION
    MINIUCHUU_MASS_CONVERT    = 1.0e10 / MINIUCHUU_HUBBLE_H
    MINIUCHUU_FIRST_SNAP      = 0
    MINIUCHUU_LAST_SNAP       = 49
    MINIUCHUU_REDSHIFTS       = [
        13.9334, 12.67409, 11.50797, 10.44649, 9.480752, 8.58543, 7.77447,
        7.032387, 6.344409, 5.721695, 5.153127, 4.629078, 4.26715, 3.929071,
        3.610462, 3.314082, 3.128427, 2.951226, 2.77809, 2.616166, 2.458114,
        2.309724, 2.16592, 2.027963, 1.8962, 1.770958, 1.65124, 1.535928,
        1.426272, 1.321656, 1.220303, 1.124166, 1.031983, 0.9441787, 0.8597281,
        0.779046, 0.7020205, 0.6282588, 0.5575475, 0.4899777, 0.4253644,
        0.3640053, 0.3047063, 0.2483865, 0.1939743, 0.1425568, 0.09296665,
        0.0455745, 0.02265383, 0.0001130128,
    ]
    MINIUCHUU_PART_MASS       = 0.0325 * 1.0e10 / MINIUCHUU_HUBBLE_H

# Properties stored in HDF5 mass units (need MASS_CONVERT)
_MASS_PROPS = frozenset({
    'CentralMvir', 'Mvir', 'StellarMass', 'BulgeMass', 'BlackHoleMass',
    'MetalsStellarMass', 'MetalsColdGas', 'MetalsEjectedMass',
    'MetalsHotGas', 'MetalsCGMgas', 'ColdGas', 'HotGas', 'CGMgas',
    'EjectedMass', 'H2gas', 'H1gas', 'IntraClusterStars',
    'MergerBulgeMass', 'InstabilityBulgeMass',
})

# Default properties to load for the primary model
_DEFAULT_PROPERTIES = [
    'StellarMass', 'BulgeMass', 'ColdGas', 'HotGas', 'CGMgas',
    'EjectedMass', 'H2gas', 'H1gas', 'BlackHoleMass',
    'IntraClusterStars', 'CentralMvir', 'Mvir',
    'MergerBulgeMass', 'InstabilityBulgeMass',
    'MetalsStellarMass', 'MetalsColdGas', 'MetalsHotGas',
    'MetalsEjectedMass', 'MetalsCGMgas',
    'SfrDisk', 'SfrBulge', 'Vvir', 'Vmax', 'Rvir',
    'DiskRadius', 'BulgeRadius',
    'Type', 'CentralGalaxyIndex',
    'Posx', 'Posy', 'Posz',
    'OutflowRate', 'MassLoading', 'Cooling', 'Regime',
]

# Properties to load for evolution (multi-snapshot) plots
_EVOLUTION_PROPERTIES = [
    'StellarMass', 'SfrDisk', 'SfrBulge', 'Mvir', 'Rvir',
    'CGMgas', 'HotGas', 'MetalsStellarMass', 'DiskRadius',
    'FFBRegime', 'Regime', 'tcool_over_tff', 'tdeplete', 'tff',
    'GalaxyIndex', 'Type',
]


# ========================== PLOTTING STYLE ==========================

def setup_style():
    """Configure matplotlib for publication-quality white-background plots."""
    plt.style.use("./plotting/kieren_cohare_palatino_sty.mplstyle")

# ========================== DATA I/O ==========================


def find_model_files(directory):
    """
    Find all model_*.hdf5 files in *directory*.

    Returns a sorted list of absolute paths.  Falls back to the single
    ``model_0.hdf5`` if no files match (backward-compatible).
    """
    return _find_model_files_early(directory)


def model_files_exist(directory):
    """Return True if at least one model HDF5 file exists in *directory*."""
    return len(find_model_files(directory)) > 0


def read_snap_from_files(filepaths, snap_key, properties, mass_convert=MASS_CONVERT,
                         particle_mass=None, mvir_particles_min=None):
    """
    Read *properties* from *snap_key* across multiple HDF5 files and
    concatenate the results.

    Parameters
    ----------
    filepaths : list of str
        HDF5 file paths (e.g. from ``find_model_files``).
    snap_key : str
        Snapshot group name, e.g. ``'Snap_63'``.
    properties : list of str
        Dataset names to read.
    mass_convert : float
        Multiplicative factor applied to properties in ``_MASS_PROPS``.
    particle_mass : float, optional
        DM particle mass for the simulation.
    mvir_particles_min : int, optional
        If provided alongside *particle_mass*, also applies a halo-mass
        resolution cut ``Mvir >= mvir_particles_min * particle_mass`` (after
        unit conversion).

    Returns
    -------
    dict : property name -> numpy array (concatenated across files).
           Empty dict if no file contains *snap_key*.
    """
    caller_wants_len = 'Len' in properties
    caller_wants_mvir = 'Mvir' in properties

    load_props = list(properties)
    if not caller_wants_len:
        load_props.append('Len')

    # If we're asked to apply a particle-mass-derived resolution cut, we need Mvir.
    if (particle_mass is not None and mvir_particles_min is not None) and 'Mvir' not in load_props:
        load_props.append('Mvir')

    chunks = {prop: [] for prop in load_props}
    found_snap = False

    for fp in filepaths:
        try:
            with h5.File(fp, 'r') as f:
                if snap_key not in f:
                    continue
                found_snap = True
                grp = f[snap_key]
                for prop in load_props:
                    if prop in grp:
                        chunks[prop].append(np.array(grp[prop]))
        except Exception as e:
            print(f"  Warning: could not read {fp}: {e}")
            continue

    if not found_snap:
        return {}

    data = {}
    for prop in load_props:
        if chunks[prop]:
            arr = np.concatenate(chunks[prop])
            if prop in _MASS_PROPS:
                arr = arr * mass_convert
            data[prop] = arr

    if 'Len' in data:
        mask = data['Len'] >= MIN_PARTICLES
        if (particle_mass is not None and mvir_particles_min is not None) and 'Mvir' in data:
            min_mvir = int(mvir_particles_min) * float(particle_mass) * mass_convert
            mask &= data['Mvir'] >= min_mvir
        data = {p: arr[mask] for p, arr in data.items()}

    if not caller_wants_len:
        data.pop('Len', None)

    if (particle_mass is not None and mvir_particles_min is not None) and not caller_wants_mvir:
        data.pop('Mvir', None)

    return data


def load_model(directory, filename=None, snapshot=SNAPSHOT,
               properties=None, mass_convert=MASS_CONVERT, particle_mass=None,
               mvir_particles_min=None):
    """
    Load galaxy properties from one or more model HDF5 files.

    When SAGE is run with MPI each rank writes its own file
    (``model_0.hdf5``, ``model_1.hdf5``, …).  This function automatically
    discovers all such files and concatenates their datasets.

    Parameters
    ----------
    directory : str
        Path to the model output directory.
    filename : str, optional
        Kept for backward compatibility.  If given, only that single file
        is read; otherwise every ``model_*.hdf5`` in *directory* is used.
    snapshot : str
        Snapshot key (e.g. ``'Snap_63'``).
    properties : list of str, optional
        Properties to load.  If *None*, loads ``_DEFAULT_PROPERTIES``.

    Returns
    -------
    dict : property name -> numpy array (converted where applicable).
    """
    if properties is None:
        properties = _DEFAULT_PROPERTIES

    if filename is not None:
        filepaths = [os.path.join(directory, filename)]
    else:
        filepaths = find_model_files(directory)

    if not filepaths:
        print(f"  Warning: no model files found in {directory}")
        return {}

    data = read_snap_from_files(filepaths, snapshot, properties,
                                mass_convert=mass_convert,
                                particle_mass=particle_mass,
                                mvir_particles_min=mvir_particles_min)
    if not data:
        print(f"  Warning: {snapshot} not found in any file in {directory}")
    return data


def load_snapshots(directory, snaps, properties=None, filename=None,
                   mass_convert=MASS_CONVERT, particle_mass=None,
                   mvir_particles_min=None):
    """
    Load multiple snapshots from one or more HDF5 files.

    Parameters
    ----------
    directory : str
        Path to model output directory.
    snaps : list of int
        Snapshot numbers to load.
    properties : list of str, optional
        Properties to load.  Defaults to ``_EVOLUTION_PROPERTIES``.
    filename : str, optional
        If given, only that single file is read; otherwise every
        ``model_*.hdf5`` in *directory* is used.

    Returns
    -------
    dict : {snap_num: {prop_name: numpy array}}
    """
    if properties is None:
        properties = _EVOLUTION_PROPERTIES

    if filename is not None:
        filepaths = [os.path.join(directory, filename)]
    else:
        filepaths = find_model_files(directory)

    if not filepaths:
        print(f"  Warning: no model files found in {directory}")
        return {}

    snapdata = {}
    for snap in snaps:
        snap_key = f'Snap_{snap}'
        data = read_snap_from_files(filepaths, snap_key, properties,
                                    mass_convert=mass_convert,
                                    particle_mass=particle_mass,
                                    mvir_particles_min=mvir_particles_min)
        if data:
            snapdata[snap] = data
        else:
            print(f"  Warning: {snap_key} not found, skipping.")

    return snapdata


# ========================== COMPUTATION UTILITIES ==========================

def mass_function(log_masses, volume, binwidth=0.1, mass_range=None):
    """
    Compute a mass function (log10 number density per dex per Mpc^3).

    Parameters
    ----------
    log_masses : array
        log10 masses.
    volume : float
        Comoving volume in Mpc^3.
    binwidth : float
        Bin width in dex.
    mass_range : tuple of (float, float), optional
        (min, max) for histogram. Auto-determined if None.

    Returns
    -------
    centers : array
        Bin centres.
    phi : array
        log10(number density). NaN where counts == 0.
    mrange : tuple
        (min, max) used, so subsets can reuse the same bins.
    """
    if mass_range is None:
        mi = np.floor(np.min(log_masses)) - 2
        ma = np.floor(np.max(log_masses)) + 2
    else:
        mi, ma = mass_range

    nbins = int(round((ma - mi) / binwidth))
    counts, edges = np.histogram(log_masses, range=(mi, ma), bins=nbins)
    centers = edges[:-1] + 0.5 * binwidth

    with np.errstate(divide='ignore'):
        phi = np.log10(counts / volume / binwidth)
    phi[~np.isfinite(phi)] = np.nan

    return centers, phi, (mi, ma)



def metallicity_12logOH(metals_cold_gas, cold_gas):
    """
    Gas-phase metallicity in 12 + log10(O/H).

    Uses Z_cold = MetalsColdGas / ColdGas, solar reference Z_sun = 0.02,
    and 12 + log10(O/H)_sun = 9.0.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.log10((metals_cold_gas / cold_gas) / 0.02) + 9.0


def stellar_metallicity(metals_stellar_mass, stellar_mass):
    """
    Stellar metallicity log10(Z/Z_sun).
    Uses Z_star = MetalsStellarMass / StellarMass, solar reference Z_sun = 0.02.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.log10((metals_stellar_mass / stellar_mass) / 0.02)


def log_ssfr(sfr_disk, sfr_bulge, stellar_mass):
    """Compute log10(sSFR / yr^-1)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.log10((sfr_disk + sfr_bulge) / stellar_mass)


def binned_median(x, y, bins, min_count=5):
    """Binned median with 25th/75th percentiles."""
    centers = 0.5 * (bins[:-1] + bins[1:])
    n = len(bins) - 1
    med = np.full(n, np.nan)
    p25 = np.full(n, np.nan)
    p75 = np.full(n, np.nan)

    for i in range(n):
        mask = (x >= bins[i]) & (x < bins[i + 1])
        count = np.sum(mask)
        if count >= min_count:
            vals = y[mask]
            med[i] = np.median(vals)
            p25[i] = np.percentile(vals, 25)
            p75[i] = np.percentile(vals, 75)

    return centers, med, p25, p75


def binned_percentiles(x, y, bins, percentiles=(16, 50, 84), min_count=20):
    """Compute binned percentiles of *y* as a function of *x*.

    Parameters
    ----------
    x, y : array-like
        Data arrays.
    bins : array-like
        Bin edges in x.
    percentiles : tuple
        Percentiles to compute (e.g. (16, 50, 84)).
    min_count : int
        Minimum number of points required in a bin.

    Returns
    -------
    centers : array
        Bin centers.
    pct : array, shape (len(percentiles), nbins)
        Percentiles per bin; NaN for bins with insufficient counts.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]

    centers = 0.5 * (bins[:-1] + bins[1:])
    nbins = len(bins) - 1
    pct = np.full((len(percentiles), nbins), np.nan)

    for i in range(nbins):
        m = (x >= bins[i]) & (x < bins[i + 1])
        if np.sum(m) >= min_count:
            pct[:, i] = np.percentile(y[m], percentiles)

    return centers, pct


def plot_binned_median_1sigma(
    ax,
    x,
    y,
    bins,
    *,
    color,
    label,
    alpha=0.25,
    lw=3.0,
    ls='-',
    min_count=20,
    zorder_fill=3,
    zorder_line=4,
):
    """Plot a median line with a 16--84% (1\u03c3) shaded band."""
    centers, pct = binned_percentiles(x, y, bins, percentiles=(16, 50, 84), min_count=min_count)
    p16, p50, p84 = pct
    valid = np.isfinite(p50) & np.isfinite(p16) & np.isfinite(p84)
    if not np.any(valid):
        return None

    ax.fill_between(centers[valid], p16[valid], p84[valid],
                    color=color, alpha=alpha, lw=0.0, zorder=zorder_fill)
    (line,) = ax.plot(centers[valid], p50[valid],
                      color=color, lw=lw, ls=ls, label=label, zorder=zorder_line)
    return line


def baryon_fractions_by_halo_mass(primary, halo_bins=None):
    """
    Compute mean baryon component fractions binned by halo mass.

    Uses np.bincount to sum components per halo in O(N), avoiding
    per-halo Python loops.

    Returns
    -------
    mass_centers : array
        Mean log10(Mvir) in each occupied bin.
    results : dict
        {component_name: {'mean': array, 'upper': array, 'lower': array}}
    """
    if halo_bins is None:
        halo_bins = np.arange(11.0, 16.1, 0.1)

    cgi = primary['CentralGalaxyIndex'].astype(np.int64)

    # Remap CentralGalaxyIndex IDs to compact 0-based group indices
    unique_ids, compact_idx = np.unique(cgi, return_inverse=True)
    ngroups = len(unique_ids)

    # Components to track
    comp_keys = ['StellarMass', 'ColdGas', 'HotGas', 'CGMgas',
                 'IntraClusterStars', 'BlackHoleMass', 'EjectedMass']

    # Sum each component by halo using bincount — O(N), fully vectorized
    halo_sums = {}
    for key in comp_keys:
        halo_sums[key] = np.bincount(compact_idx, weights=primary[key],
                                     minlength=ngroups)
    halo_sums['Total'] = sum(halo_sums[k] for k in comp_keys)

    # Central galaxies define halos
    central_mask = primary['Type'] == 0
    central_compact = compact_idx[central_mask]
    mvir = primary['Mvir'][central_mask]
    log_mvir = np.log10(mvir)

    # Fractions: component_sum / Mvir for each halo
    fractions = {}
    all_keys = ['Total'] + comp_keys
    for key in all_keys:
        fractions[key] = halo_sums[key][central_compact] / mvir

    # Bin by halo mass and compute mean +/- stderr
    bin_idx = np.digitize(log_mvir, halo_bins) - 1
    results = {k: {'mean': [], 'upper': [], 'lower': []} for k in all_keys}
    mass_centers = []

    for i in range(len(halo_bins) - 1):
        w = bin_idx == i
        n_halos = np.sum(w)
        if n_halos < 3:
            continue

        mass_centers.append(np.mean(log_mvir[w]))
        sqrt_n = np.sqrt(n_halos)

        for key in all_keys:
            vals = fractions[key][w]
            mean = np.mean(vals)
            err = np.std(vals) / sqrt_n
            results[key]['mean'].append(mean)
            results[key]['upper'].append(mean + err)
            results[key]['lower'].append(max(mean - err, 1e-6))

    # Convert to arrays
    mass_centers = np.array(mass_centers)
    for key in results:
        for stat in results[key]:
            results[key][stat] = np.array(results[key][stat])

    return mass_centers, results


def snap_to_redshift(snap):
    """Return the redshift for a given snapshot number."""
    return REDSHIFTS[snap]


def cosmic_time_gyr(z):
    """Age of the universe at redshift z, in Gyr."""
    t_H = 977.8 / (HUBBLE_H * 100)  # Hubble time in Gyr

    def integrand(zp):
        return 1.0 / ((1 + zp) * np.sqrt(OMEGA_M * (1 + zp)**3 + OMEGA_L))

    result, _ = quad(integrand, z, 1000.0)
    return t_H * result


# ========================== FIGURE UTILITIES ==========================

def save_figure(fig, filepath):
    """Save figure to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath)
    print(f'  Saved: {filepath}')
    plt.close(fig)


def _standard_legend(ax, loc='lower left', handles=None, labels=None, **kwargs):
    """Apply consistent legend formatting with fully opaque handles."""
    kwargs.setdefault('frameon', False)
    if handles is not None and labels is not None:
        leg = ax.legend(handles, labels, loc=loc, numpoints=1,
                        labelspacing=0.1, **kwargs)
    else:
        leg = ax.legend(loc=loc, numpoints=1, labelspacing=0.1, **kwargs)
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    return leg

# ========================== PLOTS ==========================

def plot_1_number_counts(primary, miniuchuu):
    """2×5 panel number-count figure comparing Millennium vs miniUchuu.

    Panels (requested):
    - Mvir centrals
    - Mvir satellites
    - Stellar mass centrals
    - Stellar mass satellites
    - Cold gas
    - H1
    - H2
    - Gas metallicity (12 + log(O/H))
    - Black hole mass
    - ICS mass

    Notes
    -----
    Resolution thresholds are applied at load time in `read_snap_from_files()`:
    - `Len >= MIN_PARTICLES`
    - `Mvir >= MIN_MVIR_PARTICLES * particle_mass` (per simulation)
    """
    print('Plot 1: Number counts (2x5)')

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    axes = axes.flatten()

    models = [
        ('miniMillennium', primary, 'steelblue'),
        ('microUchuu', miniuchuu, 'coral'),
    ]

    # Resolution mass limits in physical Msun (for optional reference lines).
    mmin_primary = MIN_MVIR_PARTICLES * PART_MASS * MASS_CONVERT
    mmin_mini = MIN_MVIR_PARTICLES * MINIUCHUU_PART_MASS * MINIUCHUU_MASS_CONVERT

    def _log10_positive(x):
        x = np.asarray(x, dtype=float)
        ok = np.isfinite(x) & (x > 0.0)
        return np.log10(x[ok])

    def _finite(x):
        x = np.asarray(x, dtype=float)
        return x[np.isfinite(x)]

    def _bins_for(series_list, binwidth, pad_bins=1):
        series_list = [s for s in series_list if s is not None and len(s) > 0]
        if not series_list:
            return None
        vals = np.concatenate(series_list)
        lo = np.nanmin(vals)
        hi = np.nanmax(vals)
        lo = np.floor(lo / binwidth) * binwidth - pad_bins * binwidth
        hi = np.ceil(hi / binwidth) * binwidth + pad_bins * binwidth
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return None
        return np.arange(lo, hi + binwidth, binwidth)

    def _step_hist(ax, values, bins, *, color, label, ls='-', lw=2.2):
        counts, edges = np.histogram(values, bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        counts = counts.astype(float)
        counts[counts <= 0] = np.nan
        ax.plot(centers, counts, drawstyle='steps-mid', color=color, ls=ls, lw=lw,
            label=label, alpha=0.85)

    def _panel(ax, title, xlabel, series, *, binwidth=0.2, legend=False, vlines=None):
        bins = _bins_for([v for (_, _, _, v) in series], binwidth)
        if bins is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return

        for label, color, ls, vals in series:
            vals = _finite(vals)
            if len(vals) == 0:
                continue
            _step_hist(ax, vals, bins, color=color, label=label, ls=ls)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('N')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.25)

        if vlines:
            for x, color, ls in vlines:
                ax.axvline(x, color=color, ls=ls, lw=1.6, alpha=0.85)

        if legend:
            ax.legend(loc='upper right', frameon=False, fontsize=9)

    vlines_mvir = [
        (np.log10(mmin_primary), 'steelblue', ':'),
        (np.log10(mmin_mini), 'coral', ':'),
    ]

    # 1) Mvir centrals
    _panel(
        axes[0],
        'Mvir (centrals)',
        r'$\log_{10}\, M_{\rm vir}\,[M_\odot]$',
        [(name, color, '-', _log10_positive(d['Mvir'][d['Type'] == 0])) for name, d, color in models],
        legend=True,
        vlines=vlines_mvir,
    )

    # 2) Mvir satellites
    _panel(
        axes[1],
        'Mvir (satellites)',
        r'$\log_{10}\, M_{\rm vir}\,[M_\odot]$',
        [(name, color, '-', _log10_positive(d['Mvir'][d['Type'] != 0])) for name, d, color in models],
        vlines=vlines_mvir,
    )

    # 3) Stellar mass centrals
    _panel(
        axes[2],
        'Stellar mass (centrals)',
        r'$\log_{10}\, m_{\ast}\,[M_\odot]$',
        [(name, color, '-', _log10_positive(d['StellarMass'][d['Type'] == 0])) for name, d, color in models],
    )

    # 4) Stellar mass satellites
    _panel(
        axes[3],
        'Stellar mass (satellites)',
        r'$\log_{10}\, m_{\ast}\,[M_\odot]$',
        [(name, color, '-', _log10_positive(d['StellarMass'][d['Type'] != 0])) for name, d, color in models],
    )

    # 5) Cold gas
    _panel(
        axes[4],
        'Cold gas',
        r'$\log_{10}\, m_{\rm cold}\,[M_\odot]$',
        [(name, color, '-', _log10_positive(d['ColdGas'])) for name, d, color in models],
    )

    # 6) H1
    _panel(
        axes[5],
        'H1',
        r'$\log_{10}\, m_{\rm H1}\,[M_\odot]$',
        [(name, color, '-', _log10_positive(d['H1gas'])) for name, d, color in models],
    )

    # 7) H2
    _panel(
        axes[6],
        'H2',
        r'$\log_{10}\, m_{\rm H2}\,[M_\odot]$',
        [(name, color, '-', _log10_positive(d['H2gas'])) for name, d, color in models],
    )

    # 8) Metallicity
    _panel(
        axes[7],
        r'Gas metallicity',
        r'$12 + \log_{10}(\mathrm{O/H})$',
        [(name, color, '-', _finite(metallicity_12logOH(d['MetalsColdGas'], d['ColdGas']))) for name, d, color in models],
        binwidth=0.05,
    )

    # 9) Black hole mass
    _panel(
        axes[8],
        'Black hole mass',
        r'$\log_{10}\, m_{\rm BH}\,[M_\odot]$',
        [(name, color, '-', _log10_positive(d['BlackHoleMass'])) for name, d, color in models],
    )

    # 10) ICS
    _panel(
        axes[9],
        'ICS mass',
        r'$\log_{10}\, m_{\rm ICS}\,[M_\odot]$',
        [(name, color, '-', _log10_positive(d['IntraClusterStars'])) for name, d, color in models],
    )

    fig.suptitle(
        f'Number counts ({len(primary["StellarMass"]):,} galaxies in Millennium, {len(miniuchuu["StellarMass"]):,} in miniUchuu)',
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    save_figure(fig, os.path.join(OUTPUT_DIR, 'NumberCounts_2x5' + OUTPUT_FORMAT))


def plot_2_diagnostics(primary, miniuchuu):
    """2×5 panel diagnostics comparing Millennium vs miniUchuu at z=0.

    Panels (requested):
    - Stellar Mass Function
    - Stellar mass vs Star Formation Rate
    - Baryonic Tully-Fisher Relation
    - Stellar to halo mass ratio
    - Stellar to halo mass relation
    - Stellar mass vs metallicity
    - Black hole mass vs bulge mass
    - Cold gas Mass Function
    - H2 Mass function
    - Stellar mass vs virial velocity
    """
    print('Plot 2: Diagnostics (2x5)')

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    axes = axes.flatten()

    models = [
        ('miniMillennium', primary, 'steelblue', VOLUME),
        ('microUchuu', miniuchuu, 'coral', MINIUCHUU_VOLUME),
    ]

    def _finite(x):
        x = np.asarray(x, dtype=float)
        return x[np.isfinite(x)]

    def _auto_bins(x_list, step, pad=1):
        xs = [np.asarray(x) for x in x_list if x is not None and len(x) > 0]
        if not xs:
            return None
        x = np.concatenate(xs)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return None
        lo = np.floor(np.min(x) / step) * step - pad * step
        hi = np.ceil(np.max(x) / step) * step + pad * step
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return None
        return np.arange(lo, hi + step, step)

    def _log10_pos(x):
        x = np.asarray(x, dtype=float)
        ok = np.isfinite(x) & (x > 0.0)
        return np.log10(x[ok])

    def _mass_function_panel(ax, prop, title, xlabel, *, binwidth=0.2, legend=False):
        series = []
        for name, d, color, vol in models:
            vals = _log10_pos(d[prop]) if prop in d else np.array([])
            series.append((name, vals, color, vol))

        all_vals = [v for (_, v, _, _) in series if v.size > 0]
        if not all_vals:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return

        combined = np.concatenate(all_vals)
        mi = np.floor(np.min(combined)) - 1
        ma = np.ceil(np.max(combined)) + 1
        mass_range = (mi, ma)

        for name, vals, color, vol in series:
            if vals.size == 0:
                continue
            centers, phi, _ = mass_function(vals, vol, binwidth=binwidth, mass_range=mass_range)
            ax.plot(centers, phi, color=color, lw=2.6, alpha=0.9, label=name)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r'$\log_{10}\,\Phi\ [{\rm Mpc}^{-3}\,{\rm dex}^{-1}]$')
        ax.grid(True, alpha=0.25)
        if legend:
            ax.legend(loc='lower left', frameon=False, fontsize=9)

    def _relation_panel(ax, title, xlabel, ylabel, xys, *, xstep=0.2, legend=False, vlines=None):
        bins = _auto_bins([x for (_, _, x, _) in xys], xstep, pad=1)
        if bins is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return

        for name, color, x, y in xys:
            if x.size == 0 or y.size == 0:
                continue
            plot_binned_median_1sigma(
                ax,
                x,
                y,
                bins,
                color=color,
                label=name,
                alpha=0.20,
                lw=2.8,
                min_count=30,
            )

        if vlines:
            for xv, color, ls in vlines:
                ax.axvline(xv, color=color, ls=ls, lw=1.6, alpha=0.85)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        if legend:
            ax.legend(loc='best', frameon=False, fontsize=9)

    # Precompute halo-mass resolution limits (physical Msun)
    mmin_primary = MIN_MVIR_PARTICLES * PART_MASS * MASS_CONVERT
    mmin_mini = MIN_MVIR_PARTICLES * MINIUCHUU_PART_MASS * MINIUCHUU_MASS_CONVERT
    vlines_mvir = [
        (np.log10(mmin_primary), 'steelblue', ':'),
        (np.log10(mmin_mini), 'coral', ':'),
    ]

    # 1) Stellar Mass Function
    _mass_function_panel(
        axes[0],
        'StellarMass',
        'Stellar Mass Function',
        r'$\log_{10}\, M_{\ast}\,[M_\odot]$',
        binwidth=0.2,
        legend=True,
    )

    # 2) Stellar mass vs SFR
    xys = []
    for name, d, color, _vol in models:
        mstar = np.asarray(d['StellarMass'], dtype=float)
        sfr = np.asarray(d['SfrDisk'], dtype=float) + np.asarray(d['SfrBulge'], dtype=float)
        ok = np.isfinite(mstar) & (mstar > 0.0) & np.isfinite(sfr) & (sfr > 0.0)
        x = np.log10(mstar[ok])
        y = np.log10(sfr[ok])
        xys.append((name, color, x, y))
    _relation_panel(
        axes[1],
        r'$M_{\ast}$ vs SFR',
        r'$\log_{10}\, M_{\ast}\,[M_\odot]$',
        r'$\log_{10}\, {\rm SFR}\,[M_\odot\,{\rm yr}^{-1}]$',
        xys,
        xstep=0.2,
        legend=False,
    )

    # 3) Baryonic Tully-Fisher Relation (Sb/c selection; median lines)
    xys = []
    for name, d, color, _vol in models:
        if 'Vmax' not in d:
            xys.append((name, color, np.array([]), np.array([])))
            continue

        # Sb/c selection (as in your example):
        # centrals, Mbar>0, 0.1 < B/T < 0.5 (with B/T ~ BulgeMass/StellarMass)
        t = np.asarray(d['Type'])
        mstar = np.asarray(d['StellarMass'], dtype=float)
        mcold = np.asarray(d['ColdGas'], dtype=float)
        mbulge = np.asarray(d['BulgeMass'], dtype=float)
        v = np.asarray(d['Vmax'], dtype=float)

        with np.errstate(divide='ignore', invalid='ignore'):
            bt = mbulge / mstar
        mbar = mstar + mcold

        ok = (
            (t == 0)
            & np.isfinite(v) & (v > 0.0)
            & np.isfinite(mbar) & (mbar > 0.0)
            & np.isfinite(bt) & (bt > 0.1) & (bt < 0.5)
        )
        x = np.log10(v[ok])
        y = np.log10(mbar[ok])
        xys.append((name, color, x, y))

    # Overplot Stark, McGaugh & Swatters 2009 band in (logV, logMbar) space.
    ax = axes[2]
    xgrid = np.arange(1.4, 2.9 + 1e-6, 0.05)
    tf = 3.94 * xgrid + 1.79
    ax.fill_between(xgrid, tf - 0.26, tf + 0.26, color='blue', alpha=0.2, zorder=1)

    _relation_panel(
        ax,
        'Baryonic TF (Sb/c)',
        r'$\log_{10}\, V_{\max}\,[{\rm km\,s^{-1}}]$',
        r'$\log_{10}\, (M_{\ast}+M_{\rm cold})\,[M_\odot]$',
        xys,
        xstep=0.05,
        legend=True,
    )

    from matplotlib.ticker import MultipleLocator
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(0.25))
    ax.set_xlim(1.4, 2.9)
    ax.set_ylim(7.5, 12.0)

    # 4) Stellar-to-halo mass ratio (centrals)
    xys = []
    for name, d, color, _vol in models:
        c = (d['Type'] == 0)
        mvir = np.asarray(d['Mvir'], dtype=float)[c]
        mstar = np.asarray(d['StellarMass'], dtype=float)[c]
        ok = np.isfinite(mvir) & (mvir > 0.0) & np.isfinite(mstar) & (mstar > 0.0)
        x = np.log10(mvir[ok])
        with np.errstate(divide='ignore', invalid='ignore'):
            y = np.log10(mstar[ok] / mvir[ok])
        xys.append((name, color, x, y))
    _relation_panel(
        axes[3],
        r'Stellar-to-halo ratio',
        r'$\log_{10}\, M_{\rm vir}\,[M_\odot]$',
        r'$\log_{10}(M_{\ast}/M_{\rm vir})$',
        xys,
        xstep=0.2,
        legend=False,
        vlines=vlines_mvir,
    )

    # 5) Stellar-to-halo mass relation (centrals)
    xys = []
    for name, d, color, _vol in models:
        c = (d['Type'] == 0)
        mvir = np.asarray(d['Mvir'], dtype=float)[c]
        mstar = np.asarray(d['StellarMass'], dtype=float)[c]
        ok = np.isfinite(mvir) & (mvir > 0.0) & np.isfinite(mstar) & (mstar > 0.0)
        x = np.log10(mvir[ok])
        y = np.log10(mstar[ok])
        xys.append((name, color, x, y))
    _relation_panel(
        axes[4],
        r'Stellar-to-halo relation',
        r'$\log_{10}\, M_{\rm vir}\,[M_\odot]$',
        r'$\log_{10}\, M_{\ast}\,[M_\odot]$',
        xys,
        xstep=0.2,
        legend=False,
        vlines=vlines_mvir,
    )

    # 6) Stellar mass vs metallicity
    xys = []
    for name, d, color, _vol in models:
        mstar = np.asarray(d['StellarMass'], dtype=float)
        z = metallicity_12logOH(d['MetalsColdGas'], d['ColdGas'])
        ok = np.isfinite(mstar) & (mstar > 0.0) & np.isfinite(z)
        x = np.log10(mstar[ok])
        y = np.asarray(z, dtype=float)[ok]
        xys.append((name, color, x, y))
    _relation_panel(
        axes[5],
        r'$M_{\ast}$ vs metallicity',
        r'$\log_{10}\, M_{\ast}\,[M_\odot]$',
        r'$12 + \log_{10}(\mathrm{O/H})$',
        xys,
        xstep=0.2,
        legend=False,
    )

    # 7) Black hole mass vs bulge mass
    xys = []
    for name, d, color, _vol in models:
        mbulge = np.asarray(d['BulgeMass'], dtype=float)
        mbh = np.asarray(d['BlackHoleMass'], dtype=float)
        ok = np.isfinite(mbulge) & (mbulge > 0.0) & np.isfinite(mbh) & (mbh > 0.0)
        x = np.log10(mbulge[ok])
        y = np.log10(mbh[ok])
        xys.append((name, color, x, y))

    # Haring & Rix (2004) reference band (log10 Mbh vs log10 Mbulge)
    # Common parameterization:
    #   log10(M_BH/Msun) = 8.20 + 1.12 * (log10(M_bulge/Msun) - 11)
    # with ~0.3 dex scatter.
    ax = axes[6]
    hr_slope = 1.12
    hr_norm = 8.20
    hr_pivot = 11.0
    hr_scatter = 0.30
    xgrid = np.arange(8.0, 12.6 + 1e-6, 0.05)
    ygrid = hr_norm + hr_slope * (xgrid - hr_pivot)
    ax.fill_between(
        xgrid,
        ygrid - hr_scatter,
        ygrid + hr_scatter,
        color='0.3',
        alpha=0.15,
        zorder=1,
        label=r'Haring \& Rix (2004)',
    )

    _relation_panel(
        ax,
        r'$M_{\rm BH}$ vs $M_{\rm bulge}$',
        r'$\log_{10}\, M_{\rm bulge}\,[M_\odot]$',
        r'$\log_{10}\, M_{\rm BH}\,[M_\odot]$',
        xys,
        xstep=0.2,
        legend=True,
    )

    # 8) Cold gas Mass Function
    _mass_function_panel(
        axes[7],
        'ColdGas',
        'Cold gas MF',
        r'$\log_{10}\, M_{\rm cold}\,[M_\odot]$',
        binwidth=0.2,
        legend=False,
    )

    # 9) H2 Mass Function
    _mass_function_panel(
        axes[8],
        'H2gas',
        'H2 MF',
        r'$\log_{10}\, M_{\rm H2}\,[M_\odot]$',
        binwidth=0.2,
        legend=False,
    )

    # 10) Stellar mass vs virial velocity (Vvir)
    xys = []
    for name, d, color, _vol in models:
        t = np.asarray(d['Type'])
        vvir = np.asarray(d['Vvir'], dtype=float)
        mstar = np.asarray(d['StellarMass'], dtype=float)
        ok = (t == 0) & np.isfinite(vvir) & (vvir > 0.0) & np.isfinite(mstar) & (mstar > 0.0)
        x = np.log10(vvir[ok])
        y = np.log10(mstar[ok])
        xys.append((name, color, x, y))
    _relation_panel(
        axes[9],
        r'$M_{\ast}$ vs $V_{\rm vir}$',
        r'$\log_{10}\, V_{\rm vir}\,[{\rm km\,s^{-1}}]$',
        r'$\log_{10}\, M_{\ast}\,[M_\odot]$',
        xys,
        xstep=0.05,
        legend=False,
    )

    fig.suptitle(
        f'Diagnostics (Len $>=$ {MIN_PARTICLES}; Mvir $>=$ {MIN_MVIR_PARTICLES} × particle mass)',
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    save_figure(fig, os.path.join(OUTPUT_DIR, 'Diagnostics_2x5' + OUTPUT_FORMAT))

# ========================== MAIN ==========================

# Registry of plot functions
# z=0 plots take (primary, miniuchuu); evolution plots take (snapdata)
Z0_PLOTS = {
    1: plot_1_number_counts,
    2: plot_2_diagnostics,
}

EVOLUTION_PLOTS = {
    
}

# Standalone plots (load their own data)
STANDALONE_PLOTS = {
   
}

ALL_PLOTS = {**Z0_PLOTS, **EVOLUTION_PLOTS, **STANDALONE_PLOTS}


def main():
    seed(SEED)
    np.random.seed(SEED)
    setup_style()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Determine which plots to generate
    if len(sys.argv) > 1:
        plot_nums = [int(x) for x in sys.argv[1:]]
    else:
        plot_nums = sorted(ALL_PLOTS.keys())

    need_z0 = any(n in Z0_PLOTS for n in plot_nums)
    need_evo = any(n in EVOLUTION_PLOTS for n in plot_nums)

    primary = miniuchuu = snapdata = None

    # Load z=0 data only if needed
    if need_z0:
        print('Loading primary model from', PRIMARY_DIR)
        primary = load_model(PRIMARY_DIR,
                             particle_mass=PART_MASS,
                             mvir_particles_min=MIN_MVIR_PARTICLES)
        print(f'  {len(primary["StellarMass"]):,} galaxies loaded')

        print('Loading miniUchuu model from', MINIUCHUU_DIR)
        miniuchuu = load_model(MINIUCHUU_DIR,
                               snapshot=f'Snap_{MINIUCHUU_LAST_SNAP}',
                               properties=['StellarMass', 'SfrDisk', 'SfrBulge',
                                           'ColdGas', 'MetalsColdGas', 'H1gas', 'H2gas',
                                           'BlackHoleMass', 'BulgeMass',
                                           'Mvir', 'Vmax', 'Vvir',
                                           'Type', 'IntraClusterStars'],
                               mass_convert=MINIUCHUU_MASS_CONVERT,
                               particle_mass=MINIUCHUU_PART_MASS,
                               mvir_particles_min=MIN_MVIR_PARTICLES)
        print(f'  {len(miniuchuu["StellarMass"]):,} galaxies loaded')
        print()

    # Load multi-snapshot data only if needed
    if need_evo:
        key_snaps = [SNAP_Z0, SNAP_Z1, SNAP_Z2, SNAP_Z3, SNAP_Z4, SNAP_Z5, SNAP_Z10]
        sfh_snaps = list(range(8, 64))
        all_snaps = sorted(set(key_snaps + sfh_snaps))

        print(f'Loading {len(all_snaps)} snapshots from', PRIMARY_DIR)
        snapdata = load_snapshots(PRIMARY_DIR,
                      all_snaps,
                      particle_mass=PART_MASS,
                      mvir_particles_min=MIN_MVIR_PARTICLES)
        print(f'  {len(snapdata)} snapshots loaded')
        print()

    # Generate requested plots
    for num in plot_nums:
        if num in Z0_PLOTS:
            Z0_PLOTS[num](primary, miniuchuu)
        elif num in EVOLUTION_PLOTS:
            EVOLUTION_PLOTS[num](snapdata)
        elif num in STANDALONE_PLOTS:
            STANDALONE_PLOTS[num]()
        else:
            print(f'Warning: Plot {num} not defined, skipping.')
        print()

    print('Done.')


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""
SAGE26 Paper Plots
==================
Publication-quality figures for the SAGE26 paper.

Usage:
    python paper_plots.py              # Generate all plots
    python paper_plots.py 1            # Generate plot 1 only
    python paper_plots.py 1 3 5        # Generate plots 1, 3, 5
"""

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, ScalarFormatter
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



# ========================== CONFIGURATION ==========================

# File paths
PRIMARY_DIR = './output/millennium/'
VANILLA_DIR = './output/millennium_vanilla/'
MINIUCHUU_DIR = './output/microuchuu/'
MODEL_FILE = 'model_0.hdf5'
OBS_DIR = './data/'

# Plotting (analysis choices — not simulation parameters)
OUTPUT_FORMAT = '.pdf'
# Mass range over which Shen+2003 fitted their early-type size-mass relation
# (SDSS, log10 m/Msun). The relation is only drawn here; outside it the line would
# be extrapolation, not data.
SHEN03_MASS_RANGE = (10.0, 11.5)
DILUTE = 7500
SEED = 2222

# Draw order: model 1-sigma bands sit beneath the observations (so they tint
# rather than hide the markers), while the model lines sit on top of them.
Z_MODEL_BAND = 2       # primary model band
Z_MODEL_BAND_ALT = 3   # comparison model band
Z_OBS = 5              # observational markers, error bars and fitted relations
Z_MODEL_LINE = 10      # primary model line
Z_MODEL_LINE_ALT = 11  # comparison model line

# Analysis thresholds (not simulation parameters)
SSFR_CUT = -11.0       # log10(sSFR/yr^-1) dividing quiescent from star-forming

# Objects required in a bin before a median is drawn from it.  Every binned
# statistic in this module takes its threshold from here, so raising it raises
# all of them together.  10 suits the current test boxes at the cluster end;
# the production volumes (a third of the 500/h Mpc Millennium, a third of the
# 400/h Mpc miniUchuu) should hold of order 340 haloes above 10^14 at z = 0 and
# comfortably support 20.
#
# No particle-count cut is applied at load time: every halo SAGE wrote is kept,
# including ones sitting on a handful of particles.  Anything needing resolved
# haloes only must say so itself, by cutting on 'Len'.
MIN_COUNT = 2

# Solar metallicity (Asplund et al. 2009)
Z_SUN = 0.0134

# IMF convention.
#
# SAGE26's RecycleFraction of 0.43 is a Chabrier instantaneous return fraction
# (Salpeter gives ~0.3), so every model SFR and stellar mass in this module is on
# a Chabrier scale.  Observational compilations quoted for a Salpeter IMF must
# therefore be shifted DOWN by this amount before being compared with the model,
# never up: a Salpeter fit converts the same light into ~1.7x more stellar mass.
#
# Getting the sign wrong on one dataset and not another puts two observational
# curves on the same axes 0.2 dex apart, which is how a real ~0.2 dex model
# excess at cosmic noon came to look like agreement with Madau & Dickinson and a
# disagreement with COSMOS-Web (which is natively Chabrier and needs no shift).
SALPETER_TO_CHABRIER_DEX = -0.24

# Shift in dex applied to a log10 stellar mass, SFR, or volume density to bring it
# from the named IMF onto the model's Chabrier scale.  Kroupa (2001) and Chabrier
# (2003) differ by only ~0.04 dex; Bell & de Jong's "diet Salpeter" sits 0.15 dex
# below true Salpeter, hence -0.24 + 0.15.  The same factor is used for masses and
# SFRs: both scale with the mass-to-light ratio of the assumed IMF.
IMF_TO_CHABRIER_DEX = {
    'chabrier':      0.00,
    'kroupa':       -0.04,
    'salpeter':     SALPETER_TO_CHABRIER_DEX,
    'diet-salpeter': SALPETER_TO_CHABRIER_DEX + 0.15,
    'unknown':       0.00,
}

# Native IMF of every observational dataset this module plots on a stellar-mass or
# SFR axis, with where that assignment comes from:
#   'file' -- stated in the data file's own header. Authoritative.
#   'lit'  -- from the paper, not recorded in the file. Worth spot-checking.
#   None   -- not established. NO shift is applied and the dataset is listed by
#             report_imf_audit() so it stays visible instead of silently wrong.
# Datasets plotted only against gas mass, halo mass, velocity or magnitude are not
# listed: no IMF enters those axes.
OBS_IMF = {
    # --- stated in the data file header ---
    'Brinchmann+04':      ('kroupa',   'file'),
    'Harvey+25':          ('kroupa',   'file'),
    'Muzzin+13':          ('kroupa',   'file'),
    'Santini+12':         ('salpeter', 'file'),
    'Tremonti+04':        ('kroupa',   'file'),
    'Andrews+13':         ('kroupa',   'file'),
    'Kewley+08':          ('kroupa',   'file'),
    'Gallazzi+05':        ('chabrier', 'file'),
    'Lange+16':           ('chabrier', 'file'),
    'COSMOS-Web':         ('chabrier', 'file'),
    'CSFRD-from-SMD':     ('chabrier', 'file'),
    'SMD (COSMOS-Web)':   ('chabrier', 'file'),
    'xGASS gas ratios':   ('chabrier', 'file'),
    # --- from the paper ---
    'Madau+Dickinson 14': ('salpeter', 'lit'),
    'Baldry+08':          ('chabrier', 'lit'),
    'Baldry+12':          ('chabrier', 'lit'),
    'Moffett+16':         ('chabrier', 'lit'),
    'Wright+18':          ('chabrier', 'lit'),
    'Thorne+21':          ('chabrier', 'lit'),
    'Weaver+23':          ('chabrier', 'lit'),
    'Song+16':            ('chabrier', 'lit'),
    'Bellstedt+20':       ('chabrier', 'lit'),
    'Bell+03':            ('diet-salpeter', 'lit'),
    'Curti+20':           ('kroupa',   'lit'),
    'Moster+13':          ('chabrier', 'lit'),
    # --- not established: no shift applied, reported at run time ---
    'Stefanon+21':        (None, None),
    'Navarro-Carrera+23': (None, None),
    'Weibel+24':          (None, None),
    'Kikuchihara+20':     (None, None),
    'Kikuchihara+20 SMD': (None, None),
    'Papovich+23':        (None, None),
    'Oesch+18':           (None, None),
    'McLeod+24':          (None, None),
    'Harikane+23':        (None, None),
    'Terrazas+17':        (None, None),
    'Kravtsov+18':        (None, None),
    'Taylor+20':          (None, None),
    'Romeo+20':           (None, None),
    'Scott+13 BH-bulge':  (None, None),
    'Outflow compilation': (None, None),
}


def imf_shift(dataset):
    """
    Dex to add to a log10 stellar mass / SFR from *dataset* to put it on the model's
    Chabrier scale.  Returns 0.0 for datasets already Chabrier and for those whose IMF
    is not established -- the latter are surfaced by report_imf_audit() rather than
    silently guessed at.
    """
    imf, _ = OBS_IMF.get(dataset, (None, None))
    if imf is None:
        return 0.0
    return IMF_TO_CHABRIER_DEX[imf]


def report_imf_audit():
    """Print which observational datasets are shifted onto Chabrier, and which cannot be."""
    print('IMF audit (model is Chabrier; observations shifted onto that scale):')
    shifted, native, unknown = [], [], []
    for name, (imf, source) in sorted(OBS_IMF.items()):
        if imf is None:
            unknown.append(name)
        elif IMF_TO_CHABRIER_DEX[imf] == 0.0:
            native.append(f'{name} [{source}]')
        else:
            shifted.append(f'{name} {imf} {IMF_TO_CHABRIER_DEX[imf]:+.2f} dex [{source}]')
    for line in shifted:
        print(f'    shifted : {line}')
    print(f'    already Chabrier ({len(native)}): ' + ', '.join(native))
    if unknown:
        print(f'    IMF NOT ESTABLISHED, no shift applied ({len(unknown)}):')
        print('      ' + ', '.join(unknown))
        print('      Add the IMF to OBS_IMF once confirmed against each paper.')

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


def _snap_nearest_z(redshifts, target_z):
    """
    Return the snapshot index whose redshift is *closest* to *target_z*,
    from either side.  Use this where panels are labelled by the round
    target redshift, so the snapshot sits as near to it as the output
    table allows.
    """
    return int(np.argmin(np.abs(np.array(redshifts) - target_z)))


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
    OUTPUT_DIR       = os.path.join(PRIMARY_DIR, 'ICS_plots/')

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
    OUTPUT_DIR = './output/millennium/ICS_plots/'
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

# Properties stored in HDF5 mass units (need MASS_CONVERT)
_MASS_PROPS = frozenset({
    'CentralMvir', 'Mvir', 'StellarMass', 'BulgeMass', 'BlackHoleMass',
    'MetalsStellarMass', 'MetalsColdGas', 'MetalsEjectedMass',
    'MetalsHotGas', 'MetalsCGMgas', 'ColdGas', 'HotGas', 'CGMgas',
    'EjectedMass', 'H2gas', 'H1gas', 'IntraClusterStars',
    'MergerBulgeMass', 'InstabilityBulgeMass',
    'ICS_disrupt', 'ICS_accrete',
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
    'OutflowRate', 'MassLoading', 'Cooling', 'Regime', 'CoolingRate'
]

# Properties to load for evolution (multi-snapshot) plots
_EVOLUTION_PROPERTIES = [
    'StellarMass', 'SfrDisk', 'SfrBulge', 'Mvir', 'Rvir',
    'CGMgas', 'HotGas', 'MetalsStellarMass', 'DiskRadius', 'BulgeRadius',
    'CoolingRate',
    'FFBRegime', 'Regime', 'tcool_over_tff', 'tdeplete', 'tff',
    'GalaxyIndex', 'Type',
]


# ========================== PLOTTING STYLE ==========================

def setup_style():
    """Configure matplotlib for publication-quality white-background plots."""
    plt.style.use("./plotting/kieren_cohare_palatino_sty.mplstyle")


def _tex_safe(s):
    """Make label strings safe for both usetex and non-usetex modes."""
    if not plt.rcParams.get('text.usetex', False):
        s = s.replace(r"\'{e}", "\u00e9")   # é
        s = s.replace(r'\&', '&')
        s = s.replace(r'\%', '%')
    return s


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


def read_snap_from_files(filepaths, snap_key, properties, mass_convert=MASS_CONVERT):
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

    Returns
    -------
    dict : property name -> numpy array (concatenated across files).
           Empty dict if no file contains *snap_key*.
    """
    load_props = list(properties)

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

    return data


def load_model(directory, filename=None, snapshot=SNAPSHOT,
               properties=None):
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

    data = read_snap_from_files(filepaths, snapshot, properties)
    if not data:
        print(f"  Warning: {snapshot} not found in any file in {directory}")
    return data


def load_snapshots(directory, snaps, properties=None, filename=None):
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
        data = read_snap_from_files(filepaths, snap_key, properties)
        if data:
            snapdata[snap] = data
        else:
            print(f"  Warning: {snap_key} not found, skipping.")

    return snapdata

def binned_percentiles(x, y, bins, percentiles=(16, 50, 84), min_count=MIN_COUNT):
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
    ax, x, y, bins, *, color, label, alpha=0.25, lw=3.0, ls='-', 
    min_count=MIN_COUNT, zorder_fill=3, zorder_line=4):
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

def stellarmass_within_halo(data, verbose=True):
    """
    All the stellar mass within each FOF halo: central + satellites + ICS.

    Galaxies are grouped by ``CentralGalaxyIndex``.  SAGE builds that index from
    the forest and file number (``core_save.c``), so it is unique across MPI
    output files and groups never collide when several ``model_*.hdf5`` are
    concatenated by ``load_model``.

    At output time all of a halo's intracluster light already sits on the FOF
    central -- ``infall_recipe`` pools each satellite's ICS into the central and
    zeroes the satellite's (``model_infall.c``) -- but the sum is taken over
    every member anyway, so a satellite that has not yet been pooled still
    counts once and only once.

    Every halo SAGE wrote is included -- no particle-count cut is applied at
    load time -- so the sums run over all group members, down to ones resting on
    very few particles.  The stars of anything already merged or disrupted are
    in ``ICS`` rather than ``Stars``.

    Parameters
    ----------
    data : dict
        A loaded model.  Needs 'StellarMass', 'IntraClusterStars', 'Mvir',
        'Type' and 'CentralGalaxyIndex'.
    verbose : bool
        Report groups dropped for want of a resolved central.  Set False when
        sweeping many snapshots, where the message is noise rather than news.

    Returns
    -------
    dict of arrays, one element per FOF halo, all aligned and in ascending
    CentralGalaxyIndex order:
        'Mvir'        virial mass of the FOF halo (the central's Mvir) [Msun]
        'Total'       Stars + ICS, i.e. every star in the halo [Msun]
        'Stars'       central + satellite stellar mass, no ICS [Msun]
        'ICS'         intracluster stars [Msun]
        'Central'     the central galaxy's own stellar mass [Msun]
        'Satellites'  summed satellite stellar mass [Msun]
        'Ngal'        number of resolved galaxies in the halo
    """
    required = ('StellarMass', 'IntraClusterStars', 'Mvir', 'Type',
                'CentralGalaxyIndex')
    missing = [p for p in required if p not in data]
    if missing:
        raise KeyError('stellarmass_within_halo needs ' + ', '.join(missing)
                       + ' -- load them with load_model(properties=[...]).')

    cgi = data['CentralGalaxyIndex'].astype(np.int64)

    # Remap CentralGalaxyIndex IDs to compact 0-based group indices
    unique_ids, compact_idx = np.unique(cgi, return_inverse=True)
    ngroups = len(unique_ids)

    # Sum by halo using bincount — O(N), fully vectorized
    stars = np.bincount(compact_idx, weights=data['StellarMass'], minlength=ngroups)
    ics = np.bincount(compact_idx, weights=data['IntraClusterStars'], minlength=ngroups)
    ngal = np.bincount(compact_idx, minlength=ngroups)

    # The central defines the halo: its Mvir is the FOF mass (a satellite's Mvir
    # is only its own subhalo), and its stellar mass is what the satellite total
    # is measured against.
    is_central = data['Type'] == 0
    central_group = compact_idx[is_central]

    mvir = np.full(ngroups, np.nan)
    central_stars = np.zeros(ngroups)
    mvir[central_group] = data['Mvir'][is_central]
    central_stars[central_group] = data['StellarMass'][is_central]

    # A group with no Type 0 member has no halo mass to bin against, so drop it
    # rather than carry a NaN downstream.
    keep = np.isfinite(mvir)
    n_dropped = ngroups - int(np.count_nonzero(keep))
    if n_dropped and verbose:
        print(f'  stellarmass_within_halo: {n_dropped:,} of {ngroups:,} groups '
              'have no resolved central, dropped')

    return {
        'Mvir':       mvir[keep],
        'Total':      (stars + ics)[keep],
        'Stars':      stars[keep],
        'ICS':        ics[keep],
        'Central':    central_stars[keep],
        'Satellites': (stars - central_stars)[keep],
        'Ngal':       ngal[keep],
    }


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


# ========================== Observations ==========================

# --------------- Intracluster light fraction against redshift ---------------
#
# A digitised literature compilation, carried over unchanged from
# plot_ics_observations() in random_plotting_scripts/BCG_ICS_fraction.py, where
# it was assembled for this same comparison.  Values are fractions, not per cent.
#
# Two things to hold in mind before reading agreement or tension off these
# points:
#
#  * f_ICL is not one measurement.  Authors cut the ICL at different surface
#    brightnesses and different radii, and disagree over whether the BCG belongs
#    in the numerator, the denominator, or neither.  Furnell+21 and Burke+15
#    overlap in redshift and still differ by a factor of a few; that spread is
#    method, not physics.  No attempt is made here to homogenise them.
#  * the model counts every intracluster star SAGE tracks, with no surface
#    brightness limit at all, so f_ICS should sit at or above a
#    surface-brightness-limited measurement rather than on top of it.
#
# 'scale' separates cluster-scale hosts from group-scale samples so each lands
# on the panel whose halo mass bin it actually belongs to.  'highlight' marks a
# point worth drawing in its own right rather than folding into the compilation.
_ICL_FRACTION_OBS = (
    dict(label='Spavone+20', scale='cluster', highlight=False,
         note='Fornax, Fornax Deep Survey',
         z=(0,),
         f=(0.3408,)),
    dict(label='Kluge+21', scale='cluster', highlight=False,
         note='ICL and host cluster',
         z=(0.03,),
         f=(0.1792,)),
    dict(label='Zibetti+05', scale='cluster', highlight=False,
         note='stacked SDSS, z = 0.25',
         z=(0.243,),
         f=(0.1085,)),
    dict(label='Feldmeier+04', scale='cluster', highlight=False,
         note='deep CCD imaging',
         z=(0.162, 0.162, 0.162, 0.185,),
         f=(0.1521, 0.1215, 0.1026, 0.0731,)),
    dict(label='Burke+15', scale='cluster', highlight=False,
         note='CLASH',
         z=(0.403, 0.387, 0.397, 0.339, 0.344, 0.342, 0.291, 0.225, 0.218,
            0.213, 0.195, 0.177,),
         f=(0.0259, 0.0271, 0.033, 0.0554, 0.0601, 0.0719, 0.1297, 0.125,
            0.1627, 0.1804, 0.1686, 0.2311,)),
    dict(label='Furnell+21', scale='cluster', highlight=False,
         note='XCS-HSC, 0.1 < z < 0.5',
         z=(0.144, 0.127, 0.122, 0.081, 0.225, 0.215, 0.256, 0.306, 0.261,
            0.294, 0.322, 0.342, 0.372, 0.337, 0.377, 0.329, 0.496, 0.425,
            0.109,),
         f=(0.3856, 0.3066, 0.3101, 0.2889, 0.2653, 0.2358, 0.2854, 0.2972,
            0.3255, 0.2748, 0.2759, 0.2665, 0.1981, 0.1887, 0.1545, 0.1533,
            0.1132, 0.0967, 0.316,)),
    dict(label='Montes & Trujillo 18', scale='cluster', highlight=False,
         note='Frontier Fields',
         z=(0.301, 0.39, 0.342, 0.537, 0.537, 0.37, 0.043,),
         f=(0.0767, 0.0861, 0.1309, 0.066, 0.0578, 0.0483, 0.1085,)),
    dict(label='Montes & Trujillo 18', scale='cluster', highlight=False,
         note='Frontier Fields, second estimate',
         z=(0.534, 0.544, 0.367, 0.397, 0.342, 0.304, 0.048,),
         f=(0.0153, 0, 0.0106, 0.0153, 0.0271, 0.033, 0.0861,)),
    dict(label='Presotto+14', scale='cluster', highlight=False,
         note='MACS J1206.2-0947',
         z=(0.435,),
         f=(0.1226,)),
    dict(label='Presotto+14', scale='cluster', highlight=False,
         note='MACS J1206.2-0947, second estimate',
         z=(0.433,),
         f=(0.0554,)),
    dict(label='Ragusa+23', scale='cluster', highlight=False,
         note='VEGAS, Antlia',
         z=(0.05,),
         f=(0.35,)),
    dict(label='Burke+12', scale='cluster', highlight=False,
         note='z ~ 1 clusters',
         z=(0.947, 0.83, 0.795, 0.808, 1.223,),
         f=(0.0142, 0.0259, 0.0377, 0.0153, 0.0236,)),
    dict(label='Ko & Jee 18', scale='cluster', highlight=False,
         note='ICL detected at z = 1.24',
         z=(1.238,),
         f=(0.0991,)),
    dict(label='XLSSC 122 (JWST)', scale='cluster', highlight=True,
         note='the only constraint near z = 2',
         z=(1.98,),
         f=(0.17,)),
    dict(label='Ragusa+23', scale='group', highlight=False,
         note='VEGAS groups',
         z=(0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
            0.05, 0.05, 0.05, 0.05, 0.05,),
         f=(0.16, 0.05, 0.05, 0.17, 0.05, 0.27, 0.34, 0.17, 0.08, 0.35, 0.18,
            0.07, 0.2, 0.22, 0.28, 0.3,)),
    dict(label='Ahad+25', scale='group', highlight=False,
         note='KiDS+GAMA groups',
         z=(0.12, 0.12, 0.12, 0.18, 0.18, 0.18, 0.24, 0.24, 0.24,),
         f=(0.16, 0.1, 0.04, 0.15, 0.12, 0.08, 0.13, 0.15, 0.05,)),
)


def load_icl_fraction_observations(scale='cluster'):
    """
    The f_ICL-versus-redshift compilation for hosts of a given *scale*.

    Parameters
    ----------
    scale : {'cluster', 'group'}
        Which host scale to return.  'cluster' is everything from Fornax
        upwards; 'group' is the group-scale samples (Ragusa+23 VEGAS,
        Ahad+25 KiDS+GAMA), which belong against the 12.5-13.5 panel rather
        than the cluster one.

    Returns
    -------
    list of dict, each with 'label', 'note', 'highlight', and 'z'/'f' arrays.
    """
    return [dict(o, z=np.asarray(o['z'], dtype=float),
                    f=np.asarray(o['f'], dtype=float))
            for o in _ICL_FRACTION_OBS if o['scale'] == scale]


# ========================== Figures ==========================





# ========================== MAIN ==========================

# Registry of plot functions
# z=0 plots take (primary, vanilla); evolution plots take (snapdata)
Z0_PLOTS = {}

EVOLUTION_PLOTS = {}

# Plot 2 reads three simulations across many snapshots, each with its own
# snapshot numbering and mass units, so it does its own loading.
STANDALONE_PLOTS = {}

ALL_PLOTS = {**Z0_PLOTS, **EVOLUTION_PLOTS, **STANDALONE_PLOTS}


def main():
    seed(SEED)
    np.random.seed(SEED)
    setup_style()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_imf_audit()
    print()

    # Determine which plots to generate
    if len(sys.argv) > 1:
        plot_nums = [int(x) for x in sys.argv[1:]]
    else:
        plot_nums = sorted(ALL_PLOTS.keys())

    need_z0 = any(n in Z0_PLOTS for n in plot_nums)
    need_evo = any(n in EVOLUTION_PLOTS for n in plot_nums)

    primary = vanilla = snapdata = None

    # Load z=0 data only if needed
    if need_z0:
        print('Loading primary model from', PRIMARY_DIR)
        primary = load_model(PRIMARY_DIR)
        print(f'  {len(primary["StellarMass"]):,} galaxies loaded')

        print('Loading vanilla model from', VANILLA_DIR)
        vanilla = load_model(VANILLA_DIR,
                             properties=['StellarMass', 'SfrDisk', 'SfrBulge',
                                         'ColdGas', 'MetalsColdGas',
                                         'BlackHoleMass', 'BulgeMass',
                                         'HotGas', 'CGMgas', 'EjectedMass',
                                         'IntraClusterStars', 'CentralGalaxyIndex',
                                         'Mvir', 'Vvir', 'CoolingRate', 'Regime', 'Type'])
        print(f'  {len(vanilla["StellarMass"]):,} galaxies loaded')
        print()

    # Load multi-snapshot data only if needed
    if need_evo:
        key_snaps = [SNAP_Z0, SNAP_Z1, SNAP_Z2, SNAP_Z3, SNAP_Z4, SNAP_Z5, SNAP_Z10]
        sfh_snaps = list(range(8, 64))
        all_snaps = sorted(set(key_snaps + sfh_snaps))

        print(f'Loading {len(all_snaps)} snapshots from', PRIMARY_DIR)
        snapdata = load_snapshots(PRIMARY_DIR, all_snaps)
        print(f'  {len(snapdata)} snapshots loaded')
        print()

    # Generate requested plots
    for num in plot_nums:
        if num in Z0_PLOTS:
            Z0_PLOTS[num](primary, vanilla)
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
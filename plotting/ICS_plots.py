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


def _ics_fraction_by_halo(data, verbose=True):
    """
    log10(Mvir) and f_ICS for every FOF halo that contains stars.

    f_ICS is the intracluster mass over *all* the stellar mass bound to the
    halo -- central + satellites + ICS -- which is the quantity observations
    measure (ICL over total cluster light), rather than ICS over the central
    alone.  Haloes with no stars at all are dropped: their fraction is 0/0.
    """
    halo = stellarmass_within_halo(data, verbose=verbose)
    w = (halo['Mvir'] > 0) & (halo['Total'] > 0)
    return np.log10(halo['Mvir'][w]), halo['ICS'][w] / halo['Total'][w]


def plot_1_fics_vs_halomass(primary, vanilla):
    """
    Intracluster star fraction as a function of halo mass at z=0.

    f_ICS = m_ICS / (m_ICS + central + satellites), summed over each FOF group,
    as median lines with 16--84% shading.

    Three curves: SAGE26 and SAGE16 on Millennium, plus SAGE26 on miniUchuu.
    The 62.5/h Mpc Millennium box runs out of haloes above log10(Mvir) ~ 13.6,
    which is exactly where the intracluster light matters, so the 100/h Mpc
    MicroUchuu run (MINIUCHUU_DIR, named for miniUchuu but pointing at
    ./output/microuchuu/) carries SAGE26 into the cluster regime.  Its particle mass is
    about half Millennium's, so the low-mass ends of the two SAGE26 curves are a
    resolution comparison rather than a physics one; where they overlap in the
    group regime they agree, which is what licenses reading the miniUchuu curve
    as the same model extended to richer haloes.

    Above where the Millennium medians stop, its individual haloes are drawn as
    points for both models.  miniUchuu covers that range for SAGE26 but was not
    run for SAGE16, so without the points the cluster end would show one model
    only.

    miniUchuu is read here with its own mass conversion, its h differing from
    Millennium's.
    """
    print('Plot 1: intracluster star fraction as a function of halo mass')

    log_mvir_p, fics_p = _ics_fraction_by_halo(primary)
    log_mvir_v, fics_v = _ics_fraction_by_halo(vanilla)

    # --- SAGE26, miniUchuu ---
    log_mvir_mu = fics_mu = None
    if os.path.exists(MINIUCHUU_DIR):
        mu_files = find_model_files(MINIUCHUU_DIR)
        mu = read_snap_from_files(
            mu_files, f'Snap_{MINIUCHUU_LAST_SNAP}',
            ['StellarMass', 'IntraClusterStars', 'Mvir', 'Type',
             'CentralGalaxyIndex'],
            mass_convert=MINIUCHUU_MASS_CONVERT) if mu_files else {}
        if mu:
            log_mvir_mu, fics_mu = _ics_fraction_by_halo(mu)
        else:
            print('  miniUchuu: no usable z = 0 snapshot -- curve omitted')

    # --- Plot ---
    fig = plt.figure()
    ax = fig.add_subplot(111)

    halo_bins = np.arange(11.0, 15.0 + 0.25, 0.25)

    plot_binned_median_1sigma(
        ax, log_mvir_p, fics_p, halo_bins,
        color='steelblue', label='SAGE26 (Millennium)',
        alpha=0.25, lw=3.5, min_count=MIN_COUNT,
        zorder_fill=Z_MODEL_BAND, zorder_line=Z_MODEL_LINE,
    )
    if log_mvir_mu is not None:
        plot_binned_median_1sigma(
            ax, log_mvir_mu, fics_mu, halo_bins,
            color='darkorange', label='SAGE26 (miniUchuu)', ls='-.',
            alpha=0.18, lw=3.0, min_count=MIN_COUNT,
            zorder_fill=Z_MODEL_BAND, zorder_line=Z_MODEL_LINE,
        )
    plot_binned_median_1sigma(
        ax, log_mvir_v, fics_v, halo_bins,
        color='purple', label='SAGE16', ls='--',
        alpha=0.20, lw=3.0, min_count=MIN_COUNT,
        zorder_fill=Z_MODEL_BAND_ALT, zorder_line=Z_MODEL_LINE_ALT,
    )

    # Individual Millennium haloes beyond the last bin either Millennium curve
    # could fill.  The box holds too few of them to take a median, but they are
    # the only SAGE16 constraint in the cluster regime -- miniUchuu was run for
    # SAGE26 alone -- so dropping them would leave that comparison blank where
    # it matters most.  The cut is set by the Millennium curves only; miniUchuu
    # reaching further right does not hide them.
    _, pct_p = binned_percentiles(log_mvir_p, fics_p, halo_bins, min_count=MIN_COUNT)
    _, pct_v = binned_percentiles(log_mvir_v, fics_v, halo_bins, min_count=MIN_COUNT)
    filled = np.isfinite(pct_p[1]) | np.isfinite(pct_v[1])
    if np.any(filled):
        edge = halo_bins[np.max(np.flatnonzero(filled)) + 1]
        for lm, f, color, marker in ((log_mvir_p, fics_p, 'steelblue', 'o'),
                                     (log_mvir_v, fics_v, 'purple', 's')):
            rare = lm >= edge
            if np.any(rare):
                ax.plot(lm[rare], f[rare], marker, color=color, ms=6,
                        markeredgecolor='k', markeredgewidth=0.8, ls='none',
                        alpha=0.8, zorder=Z_OBS)
        ax.plot([], [], 'ko', ms=6, markerfacecolor='none', ls='none',
                label='individual Millennium haloes')

    # --- QUANTITATIVE COMPARISON: f_ICS at z=0 ---
    print('  QUANTITATIVE COMPARISON: median f_ICS at z=0')
    for lo in (12.0, 13.0, 13.5, 14.0, 14.5):
        hi = lo + 0.5
        cols = []
        for name, lm, f in (('SAGE26', log_mvir_p, fics_p),
                            ('SAGE16', log_mvir_v, fics_v),
                            ('miniUchuu', log_mvir_mu, fics_mu)):
            if lm is None:
                continue
            w = (lm >= lo) & (lm < hi)
            if np.any(w):
                cols.append(f'{name}={np.median(f[w]):.3f} (N={np.sum(w):,})')
        if cols:
            print(f'    log10(Mvir)={lo:.1f}-{hi:.1f}: ' + ', '.join(cols))

    ax.set_xlim(11.0, 14.5)
    ax.set_ylim(0.0, 0.8)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax.set_xlabel(r'$\log_{10}\ M_{\mathrm{vir}}\ [M_{\odot}]$')
    ax.set_ylabel(r'$f_{\mathrm{ICS}} = m_{\mathrm{ICS}}\ /\ m_{\mathrm{*,halo}}$')

    _standard_legend(ax, loc='upper left')
    fig.tight_layout()

    save_figure(fig, os.path.join(OUTPUT_DIR,
                    'ICS_fraction_Mvir' + OUTPUT_FORMAT))


# Halo mass bins used by the f_ICS evolution figure, in log10(Mvir/Msun).
# The cluster bin starts at 10^14, the mass of the hosts the ICL measurements
# plotted beside it were made in.  The group bin is correspondingly wide
# (12.5-14.0) to run up to it.  How much of the cluster panel is populated is a
# question about the box, not the bin: see MIN_COUNT.
_FICS_HALO_BINS = (
    (11.5, 12.5, r'$11.5 \leq \log_{10} M_{\mathrm{vir}} < 12.5$'),
    (12.5, 14.0, r'$12.5 \leq \log_{10} M_{\mathrm{vir}} < 14.0$  (groups)'),
    (14.0, 16.0, r'$\log_{10} M_{\mathrm{vir}} \geq 14.0$  (clusters)'),
)

_FICS_PROPERTIES = ['StellarMass', 'IntraClusterStars', 'Mvir', 'Type',
                    'CentralGalaxyIndex']


def fics_redshift_track(directory, snaps, redshifts, mass_convert=MASS_CONVERT,
                        min_count=MIN_COUNT):
    """
    Median f_ICS and its 16--84% spread in each halo mass bin, snapshot by snapshot.

    Parameters
    ----------
    directory : str
        Model output directory.
    snaps : iterable of int
        Snapshot numbers to read, in any order.
    redshifts : sequence
        That simulation's redshift table, indexed by snapshot number.
    mass_convert : float
        Mass unit conversion for this simulation (its h is baked in).
    min_count : int
        Haloes required in a bin before a median is taken; below it the entry
        is NaN, so the curve breaks rather than jumping around on two objects.

    Returns
    -------
    (z, pct) where *z* is an array of redshifts sorted ascending and *pct* has
    shape (len(_FICS_HALO_BINS), len(z), 3) holding the 16th, 50th and 84th
    percentiles of f_ICS.  Returns (None, None) if *directory* has no files.
    """
    files = find_model_files(directory)
    if not files:
        return None, None

    snaps = sorted(snaps, key=lambda s: -redshifts[s])   # high z first
    z, pct = [], []

    for snap in snaps:
        data = read_snap_from_files(files, f'Snap_{snap}', _FICS_PROPERTIES,
                                    mass_convert=mass_convert)
        if not data:
            continue
        log_mvir, fics = _ics_fraction_by_halo(data, verbose=False)

        snap_pct = np.full((len(_FICS_HALO_BINS), 3), np.nan)
        for i, (lo, hi, _) in enumerate(_FICS_HALO_BINS):
            w = (log_mvir >= lo) & (log_mvir < hi)
            if np.count_nonzero(w) >= min_count:
                snap_pct[i] = np.percentile(fics[w], (16, 50, 84))

        z.append(redshifts[snap])
        pct.append(snap_pct)

    if not z:
        return None, None

    # (nsnap, nbin, 3) -> (nbin, nsnap, 3)
    return np.array(z), np.transpose(np.array(pct), (1, 0, 2))


def plot_2_fics_vs_redshift():
    """
    Growth of the intracluster star fraction from z = 2 to the present.

    One panel per halo mass bin, each carrying the same three curves as plot 1:
    SAGE26 and SAGE16 on Millennium, and SAGE26 on miniUchuu.  A curve is drawn
    only where its box holds MIN_COUNT haloes in that bin, so on the small
    test boxes the cluster panel is sparse and carries miniUchuu alone; on the
    production volumes all three should run its full width.

    This is the population at a fixed *present* halo mass, not a merger-tree
    track: the haloes in a bin at z = 2 are not the progenitors of the ones in
    that bin at z = 0, since a halo grows out of its bin as it assembles.  Read
    it as "how much of a 10^14 Msun halo's stellar mass was intracluster, at
    each epoch", not as the history of any one object.

    The SAGE26 band is shaded; SAGE16 and miniUchuu are lines only, or three
    overlapping bands per panel would hide the curves they belong to.

    Observations are placed on the panel matching the halo mass of the hosts
    they were measured in -- group samples in the middle, cluster samples on the
    right -- and are a heterogeneous compilation; see _ICL_FRACTION_OBS for what
    that does and does not license concluding.
    """
    print('Plot 2: intracluster star fraction versus redshift')

    Z_MAX = 2.0

    # Millennium: every output snapshot from z = 2 to z = 0.
    mill_snaps = [s for s in range(len(REDSHIFTS)) if REDSHIFTS[s] <= Z_MAX + 0.1]
    mill_snaps = [s for s in mill_snaps if s >= SNAP_Z2]

    print(f'  Millennium: {len(mill_snaps)} snapshots')
    z_p, pct_p = fics_redshift_track(PRIMARY_DIR, mill_snaps, REDSHIFTS,
                                     min_count=MIN_COUNT)
    z_v, pct_v = fics_redshift_track(VANILLA_DIR, mill_snaps, REDSHIFTS,
                                     min_count=MIN_COUNT)

    # miniUchuu: its own redshift table, snapshot numbering and mass conversion.
    z_mu = pct_mu = None
    if os.path.exists(MINIUCHUU_DIR):
        mu_snaps = [s for s in range(MINIUCHUU_FIRST_SNAP, MINIUCHUU_LAST_SNAP + 1)
                    if MINIUCHUU_REDSHIFTS[s] <= Z_MAX + 0.1]
        print(f'  miniUchuu: {len(mu_snaps)} snapshots')
        z_mu, pct_mu = fics_redshift_track(MINIUCHUU_DIR, mu_snaps,
                                           MINIUCHUU_REDSHIFTS,
                                           mass_convert=MINIUCHUU_MASS_CONVERT,
                                           min_count=MIN_COUNT)
        if z_mu is None:
            print('  miniUchuu: no usable snapshots -- curves omitted')

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

    for i, (ax, (lo, hi, title)) in enumerate(zip(axes, _FICS_HALO_BINS)):
        if z_p is not None:
            ax.fill_between(z_p, pct_p[i, :, 0], pct_p[i, :, 2],
                            color='steelblue', alpha=0.25, lw=0.0,
                            zorder=Z_MODEL_BAND)
            ax.plot(z_p, pct_p[i, :, 1], color='steelblue', lw=3.5,
                    label='SAGE26 (Millennium)', zorder=Z_MODEL_LINE)
        if z_mu is not None:
            ax.plot(z_mu, pct_mu[i, :, 1], color='darkorange', lw=3.0, ls='-.',
                    label='SAGE26 (miniUchuu)', zorder=Z_MODEL_LINE)
        if z_v is not None:
            ax.plot(z_v, pct_v[i, :, 1], color='purple', lw=3.0, ls='--',
                    label='SAGE16', zorder=Z_MODEL_LINE_ALT)

        # Observations go on the panel whose halo mass bin matches the hosts
        # they were actually measured in: the group-scale samples on the middle
        # panel, the cluster-scale ones on the right.  Nothing in the
        # compilation reaches down to the Milky Way-mass panel.
        obs_scale = {1: 'group', 2: 'cluster'}.get(i)
        if obs_scale is not None:
            obs = load_icl_fraction_observations(obs_scale)
            plain = [o for o in obs if not o['highlight']]
            if plain:
                ax.plot(np.concatenate([o['z'] for o in plain]),
                        np.concatenate([o['f'] for o in plain]),
                        'o', ms=6, markerfacecolor='gray', markeredgecolor='k',
                        markeredgewidth=0.8, ls='none', alpha=0.6,
                        label='ICL observations', zorder=Z_OBS)
            for o in obs:
                if o['highlight']:
                    ax.plot(o['z'], o['f'], '*', ms=14, markerfacecolor='gold',
                            markeredgecolor='k', markeredgewidth=0.8, ls='none',
                            label=o['label'], zorder=Z_OBS + 1)
            # Only the observation handles: the model legend is in panel 0 and
            # repeating it three times says nothing new.
            obs_labels = ({'ICL observations'}
                          | {o['label'] for o in obs if o['highlight']})
            handles, labels = ax.get_legend_handles_labels()
            _standard_legend(
                ax, loc='upper right',
                handles=[h for h, l in zip(handles, labels) if l in obs_labels],
                labels=[l for l in labels if l in obs_labels])

        ax.set_title(title)
        ax.set_xlabel(r'$z$')
        ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))

    axes[0].set_xlim(0.0, Z_MAX)          # z = 0 on the left, shared by all panels
    axes[0].set_ylim(0.0, 0.8)
    axes[0].set_ylabel(r'$f_{\mathrm{ICS}} = m_{\mathrm{ICS}}\ /\ m_{\mathrm{*,halo}}$')
    _standard_legend(axes[0], loc='upper left')

    for scale in ('group', 'cluster'):
        sets = load_icl_fraction_observations(scale)
        npt = sum(len(o['z']) for o in sets)
        print(f'  {scale} ICL observations, {npt} points from '
              + ', '.join(sorted({o['label'] for o in sets})))

    # --- QUANTITATIVE COMPARISON: growth of f_ICS since z = 2 ---
    print('  QUANTITATIVE COMPARISON: median f_ICS at z = 2, 1, 0')
    for i, (lo, hi, _) in enumerate(_FICS_HALO_BINS):
        for name, z, pct in (('SAGE26   ', z_p, pct_p),
                             ('SAGE16   ', z_v, pct_v),
                             ('miniUchuu', z_mu, pct_mu)):
            if z is None:
                continue
            vals = []
            for z_target in (2.0, 1.0, 0.0):
                j = int(np.argmin(np.abs(z - z_target)))
                v = pct[i, j, 1]
                vals.append('  n/a' if not np.isfinite(v) else f'{v:.3f}')
            print(f'    log10(Mvir) {lo:.1f}-{hi:.1f}  {name}: '
                  f'z=2 {vals[0]}, z=1 {vals[1]}, z=0 {vals[2]}')

    fig.tight_layout()

    save_figure(fig, os.path.join(OUTPUT_DIR,
                    'ICS_fraction_redshift' + OUTPUT_FORMAT))



def plot_3_ics_overproduction(diagnostic_dir=None):
    """
    Why the model puts too much stellar mass into the ICS.

    Three panels, all from the same run (MicroUchuu by default: the ICS problem
    is a cluster problem and it holds ~20x more clusters than mini-Millennium).

    (a) how a halo's stellar mass is divided between the central, its surviving
        satellites and the ICS;
    (b) what that division costs the central, against Kravtsov+18 BCG masses,
        with central+ICS shown as the ceiling the central would reach if every
        disrupted star had instead been merged onto it;
    (c) which channel filled the ICS -- stars disrupted in this halo
        (ICS_disrupt) against ICS inherited from infalling satellites that were
        themselves once centrals (ICS_accrete).

    The mechanism the panels are evidence for: SAGE dissolves a satellite
    entirely once its subhalo mass drops below its own baryonic mass
    (ThresholdSatDisruption = 1, core_build_model.c), sending 100% of its stars
    to the ICS in one step (disrupt_satellite_to_ICS, model_mergers.c).  Because
    that test is met while MergTime is still positive, disruption preempts the
    merger branch: there are no Type 2 orphans left at any redshift in either
    model, so no satellite survives to sink onto the BCG.  Stars that should
    have built the central end up as intracluster light instead.
    """
    print('Plot 3: where the ICS comes from')

    directory = diagnostic_dir or MINIUCHUU_DIR
    is_mu = os.path.abspath(directory) == os.path.abspath(MINIUCHUU_DIR)
    mass_convert = MINIUCHUU_MASS_CONVERT if is_mu else MASS_CONVERT
    snap_key = f'Snap_{MINIUCHUU_LAST_SNAP}' if is_mu else SNAPSHOT
    run_label = 'SAGE26 (MicroUchuu)' if is_mu else 'SAGE26 (Millennium)'

    props = ['StellarMass', 'IntraClusterStars', 'Mvir', 'Type',
             'CentralGalaxyIndex', 'ICS_disrupt', 'ICS_accrete']
    data = read_snap_from_files(find_model_files(directory), snap_key, props,
                                mass_convert=mass_convert)
    if not data:
        print(f'  no usable snapshot in {directory} -- figure skipped')
        return

    halo = stellarmass_within_halo(data, verbose=False)

    # ICS_disrupt/ICS_accrete are cumulative per-galaxy counters; sum them over
    # each FOF group the same way the masses were summed.
    cgi = data['CentralGalaxyIndex'].astype(np.int64)
    unique_ids, compact = np.unique(cgi, return_inverse=True)
    ngroups = len(unique_ids)
    disrupt = np.bincount(compact, weights=data['ICS_disrupt'], minlength=ngroups)
    accrete = np.bincount(compact, weights=data['ICS_accrete'], minlength=ngroups)
    # stellarmass_within_halo drops groups with no Type 0 member; apply the same
    # selection here so every array below indexes the same haloes.
    has_central = np.zeros(ngroups, dtype=bool)
    has_central[compact[data['Type'] == 0]] = True
    disrupt, accrete = disrupt[has_central], accrete[has_central]

    w = (halo['Mvir'] > 0) & (halo['Total'] > 0)
    log_mvir = np.log10(halo['Mvir'][w])
    total = halo['Total'][w]
    f_cen = halo['Central'][w] / total
    f_sat = halo['Satellites'][w] / total
    f_ics = halo['ICS'][w] / total
    disrupt, accrete = disrupt[w], accrete[w]

    bins = np.arange(11.0, 15.25 + 0.25, 0.25)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # --- (a) how the halo's stars are divided ---
    ax = axes[0]
    for y, color, label in ((f_cen, 'steelblue', 'central'),
                            (f_sat, 'seagreen', 'satellites'),
                            (f_ics, 'firebrick', 'ICS')):
        plot_binned_median_1sigma(ax, log_mvir, y, bins, color=color,
                                  label=label, alpha=0.18, lw=3.0,
                                  zorder_fill=Z_MODEL_BAND,
                                  zorder_line=Z_MODEL_LINE)
    ax.axhline(0.5, color='0.5', ls=':', lw=1.2, zorder=1)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel(r'fraction of $m_{\mathrm{*,halo}}$')
    ax.set_title('(a) where the halo stars sit')
    _standard_legend(ax, loc='upper left')

    # --- (b) what it costs the central ---
    ax = axes[1]
    cen_mass = halo['Central'][w]
    ok = cen_mass > 0
    plot_binned_median_1sigma(
        ax, log_mvir[ok], np.log10(cen_mass[ok]), bins,
        color='steelblue', label='central', alpha=0.25, lw=3.5,
        zorder_fill=Z_MODEL_BAND, zorder_line=Z_MODEL_LINE)
    ceiling = cen_mass + halo['ICS'][w]
    ok2 = ceiling > 0
    plot_binned_median_1sigma(
        ax, log_mvir[ok2], np.log10(ceiling[ok2]), bins,
        color='darkorange', label='central + ICS', ls='-.', alpha=0.15, lw=3.0,
        zorder_fill=Z_MODEL_BAND, zorder_line=Z_MODEL_LINE)

    kpath = os.path.join(OBS_DIR, 'morphology/SatKinsAndClusters_Kravtsov18.dat')
    if os.path.exists(kpath):
        k = np.loadtxt(kpath)
        ax.plot(k[:, 0], k[:, 1] + imf_shift('Kravtsov+18'), 's', ms=8,
                markerfacecolor='gray', markeredgecolor='k', markeredgewidth=0.8,
                ls='none', alpha=0.7, label='Kravtsov+18 BCG', zorder=Z_OBS)

    ax.set_ylim(9.0, 13.0)
    ax.set_ylabel(r'$\log_{10}\ m_{\mathrm{*}}\ [M_{\odot}]$')
    ax.set_title('(b) the central pays for it')
    _standard_legend(ax, loc='upper left')

    # --- (c) which channel filled the ICS ---
    ax = axes[2]
    built = disrupt + accrete
    ok = built > 0
    plot_binned_median_1sigma(
        ax, log_mvir[ok], disrupt[ok] / built[ok], bins,
        color='firebrick', label='disrupted in situ', alpha=0.18, lw=3.0,
        zorder_fill=Z_MODEL_BAND, zorder_line=Z_MODEL_LINE)
    plot_binned_median_1sigma(
        ax, log_mvir[ok], accrete[ok] / built[ok], bins,
        color='purple', label='accreted as ICS', ls='--', alpha=0.18, lw=3.0,
        zorder_fill=Z_MODEL_BAND_ALT, zorder_line=Z_MODEL_LINE_ALT)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel(r'fraction of ICS built')
    ax.set_title('(c) which channel filled it')
    _standard_legend(ax, loc='center left')

    for ax in axes:
        ax.set_xlim(11.0, 15.0)
        ax.set_xlabel(r'$\log_{10}\ M_{\mathrm{vir}}\ [M_{\odot}]$')
        ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.25))

    fig.suptitle(run_label, y=1.0)

    # --- QUANTITATIVE: the BCG deficit this predicts ---
    print('  no Type 2 (orphan) galaxies survive: '
          f'{np.count_nonzero(data["Type"] == 2):,} at this snapshot')
    print('  median central stellar mass against Kravtsov+18:')
    if os.path.exists(kpath):
        k = np.loadtxt(kpath)
        kx, ky = k[:, 0], k[:, 1] + imf_shift('Kravtsov+18')
        for lo in (13.0, 13.5, 14.0, 14.5):
            hi = lo + 0.5
            sm = (log_mvir >= lo) & (log_mvir < hi) & (cen_mass > 0)
            so = (kx >= lo) & (kx < hi)
            if np.any(sm) and np.any(so):
                mm = np.median(np.log10(cen_mass[sm]))
                mo = np.median(ky[so])
                mc = np.median(np.log10(ceiling[sm]))
                print(f'    log10(Mvir)={lo:.1f}-{hi:.1f}: central={mm:.2f}, '
                      f'central+ICS={mc:.2f}, Kravtsov+18={mo:.2f} '
                      f'(deficit {mo - mm:+.2f} dex)')

    fig.tight_layout()
    save_figure(fig, os.path.join(OUTPUT_DIR,
                    'ICS_overproduction_diagnosis' + OUTPUT_FORMAT))




def plot_4_satellite_stripping(models=None):
    """
    What a satellite loses after it falls in, and what it does not.

    SAGE already strips satellite gas properly: strip_from_satellite() removes
    the baryon excess over BaryonFrac*Mvir exponentially on the host's dynamical
    time, CGM first and then hot, and it runs to completion.  Panel (a) shows
    that working -- the gas is gone within a couple of Gyr of infall.

    Stars are on the other side of that routine's ledger.  Its closing comment
    is explicit: an excess sitting in "non-strippable reservoirs (stars, cold
    gas, BH, ICS)" is "left in place".  Panel (b) shows the consequence: a
    satellite's stellar mass relative to its value at infall never falls below
    one.  It keeps every star it arrived with, keeps forming more from the cold
    gas stripping does not touch, and then loses the entire accumulated total in
    a single step when disrupt_satellite_to_ICS() fires.

    That is the gap: there is no channel by which a surviving satellite gives up
    stars gradually, so all of the intracluster light has to come from an
    all-or-nothing event.
    """
    print('Plot 4: what satellites lose after infall')

    if models is None:
        models = ((PRIMARY_DIR, 'SAGE26', 'steelblue', '-'),
                  (VANILLA_DIR, 'SAGE16', 'purple', '--'))

    props = ['Type', 'TimeOfInfall', 'infallMvir', 'infallStellarMass',
             'StellarMass', 'HotGas', 'CGMgas', 'ColdGas', 'Mvir',
             'EjectedMass', 'BlackHoleMass', 'IntraClusterStars']

    # Cosmic time at each snapshot, so infall can be quoted in Gyr.
    age = np.array([cosmic_time_gyr(z) for z in REDSHIFTS])

    bins = np.arange(0.0, 10.0 + 0.5, 0.5)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    found = 0

    for directory, label, color, ls in models:
        files = find_model_files(directory)
        if not files:
            continue
        data = read_snap_from_files(files, SNAPSHOT, props)
        missing = [p for p in props if p not in data]
        if missing:
            print(f'  {label}: missing {", ".join(missing)} -- skipped')
            continue
        found += 1

        # Satellites that still exist and carry a record of their infall.
        w = ((data['Type'] == 1) & (data['TimeOfInfall'] > 0)
             & (data['infallStellarMass'] > 0) & (data['infallMvir'] > 0)
             & (data['Mvir'] > 0))
        snap_infall = data['TimeOfInfall'][w].astype(int)
        t_since = age[-1] - age[np.clip(snap_infall, 0, len(age) - 1)]

        gas = (data['HotGas'] + data['CGMgas'])[w]
        gas_budget = BARYON_FRAC * data['infallMvir'][w] * MASS_CONVERT
        f_gas = gas / np.maximum(gas_budget, 1e-30)

        f_star = data['StellarMass'][w] / (data['infallStellarMass'][w] * MASS_CONVERT)

        plot_binned_median_1sigma(axes[0], t_since, f_gas, bins, color=color,
                                  label=label, ls=ls, alpha=0.20, lw=3.0,
                                  zorder_fill=Z_MODEL_BAND, zorder_line=Z_MODEL_LINE)
        plot_binned_median_1sigma(axes[1], t_since, f_star, bins, color=color,
                                  label=label, ls=ls, alpha=0.20, lw=3.0,
                                  zorder_fill=Z_MODEL_BAND, zorder_line=Z_MODEL_LINE)

        # The single ratio that drives both existing rules: gas stripping starts
        # when baryons exceed BaryonFrac*Mvir, disruption fires at unity.
        baryons = (data['StellarMass'] + data['ColdGas'] + data['HotGas']
                   + data['CGMgas'] + data['EjectedMass']
                   + data['BlackHoleMass'] + data['IntraClusterStars'])[w]
        ratio = data['Mvir'][w] / np.maximum(baryons, 1e-30)
        plot_binned_median_1sigma(axes[2], t_since, ratio, bins, color=color,
                                  label=label, ls=ls, alpha=0.20, lw=3.0,
                                  zorder_fill=Z_MODEL_BAND, zorder_line=Z_MODEL_LINE)

        for lo in (0.0, 2.0, 5.0):
            s = (t_since >= lo) & (t_since < lo + 2.0)
            if np.count_nonzero(s) >= MIN_COUNT:
                print(f'  {label}: {lo:.0f}-{lo + 2:.0f} Gyr since infall '
                      f'(N={np.count_nonzero(s):,}): gas/budget = {np.median(f_gas[s]):.3f}, '
                      f'm*/m*_infall = {np.median(f_star[s]):.3f}')

    if found == 0:
        print('  no usable models -- figure skipped')
        plt.close(fig)
        return

    # A satellite that never loses a star sits on or above this line for ever.
    axes[1].axhline(1.0, color='0.4', ls=':', lw=1.5, zorder=1)
    axes[1].text(0.97, 1.02, 'no stars lost', transform=axes[1].get_yaxis_transform(),
                 ha='right', va='bottom', fontsize=10, color='0.35')

    axes[0].set_ylim(0.0, 0.15)
    axes[0].set_ylabel(r'$(m_{\mathrm{hot}} + m_{\mathrm{CGM}})\ /\ f_{\mathrm{b}} M_{\mathrm{vir,infall}}$')
    axes[0].set_title('(a) the gas is stripped')

    axes[1].set_ylim(0.5, 2.0)
    axes[1].set_ylabel(r'$m_{\mathrm{*}}\ /\ m_{\mathrm{*,infall}}$')
    axes[1].set_title('(b) the stars are not')

    # Between these two lines the satellite is losing gas and nothing else.
    # Below the lower one it is deleted whole.  That band is where a gradual
    # stellar stripping term would have to act.
    axes[2].axhspan(1.0, 1.0 / BARYON_FRAC, color='0.7', alpha=0.30, zorder=1)
    axes[2].axhline(1.0 / BARYON_FRAC, color='0.4', ls=':', lw=1.4, zorder=2)
    axes[2].axhline(1.0, color='firebrick', ls=':', lw=1.8, zorder=2)
    axes[2].text(9.7, 1.0 / BARYON_FRAC * 1.08, 'gas stripping starts',
                 ha='right', va='bottom', fontsize=9, color='0.35')
    axes[2].text(9.7, 1.12, 'disrupted whole', ha='right', va='bottom',
                 fontsize=9, color='firebrick')
    axes[2].set_yscale('log')
    axes[2].set_ylim(0.7, 60.0)
    axes[2].set_ylabel(r'$M_{\mathrm{vir}}\ /\ m_{\mathrm{baryon}}$')
    axes[2].set_title('(c) the window in between')

    for ax in axes:
        ax.set_xlim(0.0, 10.0)
        ax.set_xlabel('time since infall [Gyr]')
        ax.xaxis.set_major_locator(plt.MultipleLocator(2.0))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
        _standard_legend(ax, loc='upper right')

    fig.tight_layout()
    save_figure(fig, os.path.join(OUTPUT_DIR,
                    'ICS_satellite_stripping' + OUTPUT_FORMAT))


# ========================== MAIN ==========================

# Registry of plot functions
# z=0 plots take (primary, vanilla); evolution plots take (snapdata)
Z0_PLOTS = {1:plot_1_fics_vs_halomass}

EVOLUTION_PLOTS = {}

# Plot 2 reads three simulations across many snapshots, each with its own
# snapshot numbering and mass units, so it does its own loading.
STANDALONE_PLOTS = {2:plot_2_fics_vs_redshift,
                    3:plot_3_ics_overproduction,
                    4:plot_4_satellite_stripping}

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
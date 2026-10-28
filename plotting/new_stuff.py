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

NOFFB_DIR = './output/millennium_noffb/'
NOCGM_DIR = './output/millennium_nocgm/'
C16_FEEDBACK_DIR = './output/millennium_c16feedback/'
GD14_DIR = './output/millennium_gd14/'

KD12_DIR = './output/millennium_kd12/'
KMT09_DIR = './output/millennium_kmt09/'
K13_DIR = './output/millennium_k13/'

FFB_BK25_DIR        = './output/millennium_mbk/'
FFB_BK25_SMOOTH_DIR = './output/millennium_mbk_smooth/'
FFB100_DIR          = './output/millennium_ffb100/'
FFB_BK25_FFB100_DIR = './output/millennium_mbk100/'

H2_DEPLETE_MILLENNIUM_DIR = './output/millennium_h2deplete/'
H2_DEPLETE_MINIUCHUU_DIR = './output/microuchuu_h2deplete/'

BETA_CLOCK_MILLENNIUM_075_DIR = './output/millennium_beta_clock_075/'
BETA_CLOCK_MINIUCHUU_075_DIR = './output/microuchuu_beta_clock_075/'

BETA_CLOCK_075_H2_DEPLETE_MILLENNIUM_DIR = './output/millennium_beta_clock_075_h2_deplete/'
BETA_CLOCK_075_H2_DEPLETE_MINIUCHUU_DIR = './output/microuchuu_beta_clock_075_h2_deplete/'

MODEL_FILE = 'model_0.hdf5'
OBS_DIR = './data/'

# Plotting (analysis choices — not simulation parameters)
OUTPUT_FORMAT = '.pdf'

# Model line colours for the SFRD/SMD history plots, sampled from plasma. The
# yellow end is avoided (unreadable on white). Millennium and miniUchuu runs of
# the same model share a colour and are told apart by line style.
_PLASMA = plt.get_cmap('plasma')
# Base size (before the model legend is added beneath) of the SFRD/SMD
# history figures.
HISTORY_FIGSIZE = (10.5, 7.9)

MODEL_COLORS = {
    'sage16':  _PLASMA(0.00),
    'sage26':  _PLASMA(0.22),
    'beta':    _PLASMA(0.45),
    'h2':      _PLASMA(0.65),
    'h2_beta': _PLASMA(0.85),
}
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
MIN_PARTICLES = 20     # minimum DM particles for a resolved halo (applied at load time)
SSFR_CUT = -11.0       # log10(sSFR/yr^-1) dividing quiescent from star-forming

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

# FFB model variants (different max star-formation efficiencies)
FFB_MODELS = [
    {'name': r'FFB 10\%',  'dir': './output/millennium_ffb10/',  'sfe': 0.10},
    {'name': r'FFB 20\%',  'dir': './output/millennium_ffb20/',  'sfe': 0.20},
    {'name': r'FFB 30\%',  'dir': './output/millennium_ffb30/',  'sfe': 0.30},
    {'name': r'FFB 40\%',  'dir': './output/millennium_ffb40/',  'sfe': 0.40},
    {'name': r'FFB 50\%',  'dir': './output/millennium_ffb50/',  'sfe': 0.50},
    {'name': r'FFB 60\%',  'dir': './output/millennium_ffb60/',  'sfe': 0.60},
    {'name': r'FFB 70\%',  'dir': './output/millennium_ffb70/',  'sfe': 0.70},
    {'name': r'FFB 80\%',  'dir': './output/millennium_ffb80/',  'sfe': 0.80},
    {'name': r'FFB 90\%',  'dir': './output/millennium_ffb90/',  'sfe': 0.90},
    {'name': r'FFB 100\%', 'dir': './output/millennium_ffb100/', 'sfe': 1.00},
]

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
    caller_wants_len = 'Len' in properties
    load_props = list(properties) if caller_wants_len else list(properties) + ['Len']

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
        data = {p: arr[mask] for p, arr in data.items()}

    if not caller_wants_len:
        data.pop('Len', None)

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


# ========================== COMPUTATION UTILITIES ==========================

def calculate_muratov_mass_loading(vvir, z=0.0):
    """
    Calculate mass loading factor using Muratov et al. (2015) formulation
    Vectorized for better performance
    """
    # Constants from Muratov et al. (2015) and SAGE implementation
    V_CRIT = 60.0      # Critical velocity where the power law breaks
    NORM = 2.9         # Normalization factor
    Z_EXP = 1.3        # Redshift power-law exponent
    LOW_V_EXP = -3.2   # Low velocity power-law exponent
    HIGH_V_EXP = -1.0  # High velocity power-law exponent
    
    # Vectorized calculation for better performance
    z_term = np.power(1.0 + z, Z_EXP)
    v_ratio = vvir / V_CRIT
    
    # Vectorized broken power law
    v_term = np.where(vvir < V_CRIT, 
                      np.power(v_ratio, LOW_V_EXP),
                      np.power(v_ratio, HIGH_V_EXP))
    
    # Calculate final mass loading factor
    eta = NORM * z_term * v_term
    
    # Vectorized capping and finite value handling
    eta = np.clip(eta, 0.0, 100.0)
    eta = np.where(np.isfinite(eta), eta, 0.0)
    
    return eta

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


def mass_function_bootstrap(log_masses, volume, binwidth=0.1, mass_range=None,
                            n_boot=100):
    """
    Compute a mass function with bootstrap confidence intervals.

    Returns
    -------
    centers : array
        Bin centres.
    phi : array
        log10(number density).
    phi_lo : array
        16th percentile (lower bound).
    phi_hi : array
        84th percentile (upper bound).
    mrange : tuple
        (min, max) used.
    """
    # First compute the main mass function to get bin edges
    centers, phi, mrange = mass_function(log_masses, volume, binwidth, mass_range)

    n_gal = len(log_masses)
    if n_gal == 0:
        return centers, phi, phi, phi, mrange

    mi, ma = mrange
    nbins = int(round((ma - mi) / binwidth))

    # Bootstrap resampling
    boot_phi = np.full((n_boot, len(centers)), np.nan)
    for b in range(n_boot):
        idx = np.random.randint(0, n_gal, n_gal)
        boot_masses = log_masses[idx]
        counts, _ = np.histogram(boot_masses, range=(mi, ma), bins=nbins)
        with np.errstate(divide='ignore'):
            vals = np.log10(counts / volume / binwidth)
        # Replace -inf (empty bins) with nan so percentiles ignore them
        boot_phi[b, :] = np.where(np.isfinite(vals), vals, np.nan)

    # Compute percentiles (nanpercentile skips nan, so empty-bin samples are excluded)
    with np.errstate(invalid='ignore'):
        phi_lo = np.nanpercentile(boot_phi, 16, axis=0)
        phi_hi = np.nanpercentile(boot_phi, 84, axis=0)

    return centers, phi, phi_lo, phi_hi, mrange


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
    ax, x, y, bins, *, color, label, alpha=0.25, lw=3.0, ls='-', 
    min_count=20, zorder_fill=3, zorder_line=4):
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


def density_contour(x, y, bins=100, weights=None, smooth=1.5):
    """
    Generate a 2D density map for contour plotting.

    Parameters
    ----------
    x, y : array-like
        The x and y coordinates of the data points.
    bins : int or [int, int]
        The number of bins in each dimension.
    weights : array-like, optional
        An array of weights for each point.
    smooth : float or None
        Gaussian smoothing sigma in bin units. None to disable.

    Returns
    -------
    X, Y : array-like
        The coordinates of the bin centers.
    Z : array-like
        The 2D density map (or weighted counts).
    """
    H, xedges, yedges = np.histogram2d(x, y, bins=bins, weights=weights)

    if smooth:
        H = gaussian_filter(H, sigma=smooth)

    # Convert to bin centers
    X = 0.5 * (xedges[:-1] + xedges[1:])
    Y = 0.5 * (yedges[:-1] + yedges[1:])

    # The histogram needs to be transposed for contour plotting
    return X, Y, H.T


def sigma_contour_levels(Z):
    """
    Compute density thresholds enclosing 1-3 sigma of a 2D distribution.

    For a 2D distribution the fraction enclosed within N sigma is
    f(N) = 1 - exp(-N^2 / 2).

    Returns levels ordered [3sigma, 2sigma, 1sigma, Z_max]
    suitable for direct use in contourf (ascending density).
    """
    flat = Z.flatten()
    flat = flat[flat > 0]
    if len(flat) == 0:
        return None
    sorted_Z = np.sort(flat)[::-1]
    cumsum = np.cumsum(sorted_Z) / np.sum(sorted_Z)

    fractions = [1 - np.exp(-0.5 * n**2) for n in [3, 2, 1]]
    levels = []
    for f in fractions:
        idx = np.searchsorted(cumsum, f)
        idx = min(idx, len(sorted_Z) - 1)
        levels.append(sorted_Z[idx])
    levels.append(Z.max())
    return levels


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


def baryon_fractions_by_halo_mass_vanilla(vanilla, halo_bins=None):
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

    cgi = vanilla['CentralGalaxyIndex'].astype(np.int64)

    # Remap CentralGalaxyIndex IDs to compact 0-based group indices
    unique_ids, compact_idx = np.unique(cgi, return_inverse=True)
    ngroups = len(unique_ids)

    # Components to track
    comp_keys = ['StellarMass', 'ColdGas', 'HotGas',
                 'IntraClusterStars', 'BlackHoleMass', 'EjectedMass']

    # Sum each component by halo using bincount — O(N), fully vectorized
    halo_sums = {}
    for key in comp_keys:
        halo_sums[key] = np.bincount(compact_idx, weights=vanilla[key],
                                     minlength=ngroups)
    halo_sums['Total'] = sum(halo_sums[k] for k in comp_keys)

    # Central galaxies define halos
    central_mask = vanilla['Type'] == 0
    central_compact = compact_idx[central_mask]
    mvir = vanilla['Mvir'][central_mask]
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


PRECIP_THRESHOLD = 10.0     # matches PRECIP_THRESHOLD in model_cooling_heating.c
PRECIP_WIDTH     = 2.0      # matches PRECIP_TRANSITION_WIDTH


def precipitation_fraction(tcool_over_tff, include_condensation=True):
    """Effective inflow fraction f_inflow as implemented in SAGE26.

    The model condenses only the CGM mass in excess of the marginally stable
    reservoir m_eq = m_CGM (t_cool/t_ff) / threshold, so the rate is

        mdot = S((threshold - r)/width) * (m_CGM - m_eq) / t_ff
             = [ S((threshold - r)/width) * max(0, 1 - r/threshold) ] * m_CGM / t_ff

    and the bracketed quantity is the effective inflow fraction.  It reaches
    0.9 at r = 0.90 and is identically zero for r >= threshold.

    include_condensation=False returns the bare sigmoid only.  That is the form
    printed as Eq. 5 in the first submission, which omitted the condensation
    term; it is retained so the two can be shown side by side.
    """
    ratio = np.atleast_1d(np.array(tcool_over_tff, dtype=float))
    x_sig = (PRECIP_THRESHOLD - ratio) / PRECIP_WIDTH
    f = 1.0 / (1.0 + np.exp(-np.clip(x_sig, -700.0, 700.0)))
    if include_condensation:
        f = f * np.clip(1.0 - ratio / PRECIP_THRESHOLD, 0.0, None)
    return f.squeeze()


def ffb_threshold_mass_msun(z):
    """FFB threshold mass from Li et al. (2024) Eq. 2."""
    z_norm = (1.0 + z) / 10.0
    log_M_code = 0.8 + np.log10(HUBBLE_H) - 6.2 * np.log10(z_norm)
    return 10.0**log_M_code * 1.0e10 / HUBBLE_H


def ffb_fraction(Mvir_msun, z, delta_log_M=0.15):
    """
    Theoretical FFB fraction as a logistic sigmoid at the threshold mass.

    Matches the C implementation: f = 1 / (1 + exp(-x))
    where x = log10(Mvir / Mvir_ffb) / delta_log_M.
    """
    M_thresh = ffb_threshold_mass_msun(z)
    x = (np.log10(Mvir_msun) - np.log10(M_thresh)) / delta_log_M
    return 1.0 / (1.0 + np.exp(-x))


# -------- MBK25 (Boylan-Kolchin 2025) theoretical FFB fraction --------

try:
    from colossus.cosmology import cosmology as _colossus_cosmology
    from colossus.halo import concentration as _colossus_conc
    _colossus_cosmology.setCosmology('custom_millennium', flat=True,
                                     H0=73.0, Om0=OMEGA_M, Ob0=0.045,
                                     sigma8=0.90, ns=1.0, relspecies=False)
    _HAS_COLOSSUS = True
except Exception:
    _HAS_COLOSSUS = False


def _delta_vir_bn98(z):
    """Halo overdensity relative to rho_crit(z), matching the model.

    SAGE defines R_vir with DELTA_VIRT = 200 rho_crit (model_halo_properties.c),
    and the Ishiyama+21 concentration table it reads is mdef=200c.  The MBK25
    criterion combines R_vir with c through c^2/(2 mu(c)), so both must use the
    same definition; a Bryan & Norman virial overdensity here (the previous
    behaviour) mixed a BN98 R_vir with a 200c concentration and biased the
    threshold mass.  Kept under the original name so existing callers are
    unaffected.
    """
    return 200.0 + 0.0 * np.asarray(z, dtype=float)


def _rvir_m(Mvir_msun, z):
    """Virial radius [m] from M_vir [M_sun] using Bryan & Norman overdensity."""
    H0_si = HUBBLE_H * 1.0e5 / 3.085678e22   # H_0 in s^-1
    Ez = np.sqrt(OMEGA_M * (1.0 + z)**3 + OMEGA_L)
    rho_crit = 3.0 * (H0_si * Ez)**2 / (8.0 * np.pi * 6.674e-11)
    delta = _delta_vir_bn98(z)
    return (3.0 * Mvir_msun * 1.989e30 / (4.0 * np.pi * delta * rho_crit))**(1.0 / 3.0)


# g_crit = G * 3100 M_sun / pc^2 (BK25 Table 1)
_G_CRIT_SI = 6.674e-11 * 3100.0 * 1.989e30 / (3.085678e16)**2


def _c_ishiyama21(Mvir_msun, z):
    """
    Mean Ishiyama+21 concentration (200c) for an array of M_vir [M_sun] at redshift z.
    Falls back to a Bullock+01 power-law if colossus is not available.
    """
    M_h = np.asarray(Mvir_msun) * HUBBLE_H   # M_sun/h, as colossus expects
    if _HAS_COLOSSUS:
        try:
            c = _colossus_conc.concentration(M_h, '200c', z, model='ishiyama21')
            return np.maximum(np.atleast_1d(np.asarray(c, dtype=float)), 1.0)
        except Exception:
            pass
    # Fallback power-law approximation (Bullock+01 style)
    c = 9.0 / (1.0 + z) * (M_h / 1.0e12)**(-0.13)
    return np.maximum(c, 1.0)


def ffb_fraction_mbk25(Mvir_msun, z, sigma_c=0.2):
    """
    MBK25 FFB fraction via the Boylan-Kolchin 2025 maximum-acceleration criterion.

    g_max = G M_vir / R_vir^2 * c^2 / (2 mu(c)),  mu(c) = ln(1+c) - c/(1+c)
    FFB when g_max > g_crit = G * 3100 M_sun / pc^2  (BK25 Table 1).

    With sigma_c > 0, concentration scatters log-normally around the Ishiyama+21
    mean (matching EnhancedStarFormationOn=2 in the C code):
        f_ffb(M, z) = P(c > c_thresh) = norm.sf((ln c_thresh - ln c_mean) / sigma_c)
    With sigma_c = 0, returns a sharp step function (the unscattered threshold).

    Parameters
    ----------
    Mvir_msun : array_like  Halo virial mass [M_sun].
    z         : float       Redshift.
    sigma_c   : float       Log-normal scatter in ln(c); 0.2 matches FFB_CONC_SIGMA
                            in src/model_regimes.c.
    """
    from scipy.optimize import brentq
    from scipy.stats import norm as _snorm

    Mvir_msun = np.atleast_1d(np.asarray(Mvir_msun, dtype=float))
    c_mean = _c_ishiyama21(Mvir_msun, z)
    Rvir = _rvir_m(Mvir_msun, z)
    g_vir = 6.674e-11 * Mvir_msun * 1.989e30 / Rvir**2

    if sigma_c == 0.0:
        mu = np.log(1.0 + c_mean) - c_mean / (1.0 + c_mean)
        g_max = g_vir * c_mean**2 / (2.0 * mu)
        return (g_max > _G_CRIT_SI).astype(float)

    f = np.zeros(len(Mvir_msun))
    for i in range(len(Mvir_msun)):
        gv = float(g_vir[i])

        def _obj(c_val):
            mu = np.log(1.0 + c_val) - c_val / (1.0 + c_val)
            return gv * c_val**2 / (2.0 * mu) - _G_CRIT_SI

        if _obj(1.0) > 0.0:       # even c=1 exceeds g_crit
            f[i] = 1.0
            continue
        if _obj(200.0) < 0.0:     # even c=200 is below g_crit
            f[i] = 0.0
            continue
        try:
            c_thresh = brentq(_obj, 1.0, 200.0, xtol=1e-3, rtol=1e-4)
            f[i] = _snorm.sf((np.log(c_thresh) - np.log(float(c_mean[i]))) / sigma_c)
        except ValueError:
            f[i] = 0.0

    return f


def mbk25_threshold_mass_msun(z, c):
    """
    MBK25 FFB threshold virial mass [M_sun] at redshift *z* for a fixed
    halo concentration *c*.

    Inverts the maximum-acceleration criterion g_max(M, z, c) = g_crit.  With
        g_max = G M / R_vir^2 * c^2 / (2 mu(c)),   mu(c) = ln(1+c) - c/(1+c)
    and R_vir^3 = 3 M / (4 pi Delta_vir(z) rho_crit(z)), the virial acceleration
    scales as g_vir = G M^{1/3} (4 pi Delta rho_crit / 3)^{2/3}, so the criterion
    is linear in M^{1/3} and inverts in closed form.

    A more concentrated halo (larger c) reaches g_crit at lower mass, so the
    threshold line drops with increasing c.
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    G = 6.674e-11
    H0_si = HUBBLE_H * 1.0e5 / 3.085678e22
    Ez2 = OMEGA_M * (1.0 + z)**3 + OMEGA_L
    rho_crit = 3.0 * (H0_si**2) * Ez2 / (8.0 * np.pi * G)
    delta = _delta_vir_bn98(z)
    A = 4.0 * np.pi * delta * rho_crit / 3.0          # R_vir^3 = M_kg / A

    mu = np.log(1.0 + c) - c / (1.0 + c)
    shape = c**2 / (2.0 * mu)                          # g_max / g_vir
    # g_max = G * A^{2/3} * shape * M_kg^{1/3} = g_crit
    M_kg_cbrt = _G_CRIT_SI / (G * A**(2.0 / 3.0) * shape)
    M_kg = M_kg_cbrt**3
    return (M_kg / 1.989e30).squeeze()


def mbk25_threshold_median_c(z):
    """
    MBK25 threshold mass [M_sun] at the median concentration, and that concentration.

    c depends on M and M on c, so the pair is solved self-consistently at *z*.
    Returns (M_thr, c_med).
    """
    c = 3.3
    M = float(np.atleast_1d(mbk25_threshold_mass_msun(z, c)))
    for _ in range(20):
        c_new = float(np.atleast_1d(_c_ishiyama21(np.array([M]), z)))
        if not np.isfinite(c_new) or c_new <= 1.0:
            break
        M_new = float(np.atleast_1d(mbk25_threshold_mass_msun(z, c_new)))
        if abs(np.log10(M_new) - np.log10(M)) < 1e-4:
            M, c = M_new, c_new
            break
        M, c = M_new, c_new
    return M, c


def mbk25_threshold_concentration(Mvir_msun, z):
    """
    Threshold concentration c_thresh such that a halo of mass *Mvir_msun* at
    redshift *z* exactly satisfies the MBK25 criterion g_max = g_crit.

    A halo is FFB (mode-4) when its drawn concentration exceeds this value, so a
    selected galaxy lies on the fixed-c threshold line for c = c_thresh.  Returns
    NaN where even c = 200 fails to reach g_crit (never FFB at any concentration).
    """
    from scipy.optimize import brentq

    Mvir_msun = np.atleast_1d(np.asarray(Mvir_msun, dtype=float))
    G = 6.674e-11
    Rvir = _rvir_m(Mvir_msun, z)
    g_vir = G * Mvir_msun * 1.989e30 / Rvir**2

    out = np.full(len(Mvir_msun), np.nan)
    for i, gv in enumerate(g_vir):
        def _obj(c):
            mu = np.log(1.0 + c) - c / (1.0 + c)
            return gv * c**2 / (2.0 * mu) - _G_CRIT_SI
        if _obj(1.0) > 0.0:
            out[i] = 1.0
        elif _obj(200.0) < 0.0:
            out[i] = np.nan
        else:
            out[i] = brentq(_obj, 1.0, 200.0, xtol=1e-3, rtol=1e-4)
    return out.squeeze()


# ========================== FIGURE UTILITIES ==========================

def save_figure(fig, filepath):
    """Save figure to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath)
    print(f'  Saved: {filepath}')
    plt.close(fig)


def _scaled_font_rc(scale):
    """rcParams overrides with every font size scaled by *scale*.

    Used by figures whose canvas is smaller than the stylesheet default, so
    the text renders at the same size *relative to the axes* as elsewhere.
    """
    from matplotlib.font_manager import font_scalings

    base = plt.rcParams['font.size']
    keys = ('font.size', 'axes.labelsize', 'axes.titlesize',
            'xtick.labelsize', 'ytick.labelsize',
            'legend.fontsize', 'legend.title_fontsize')
    out = {}
    for k in keys:
        v = plt.rcParams[k]
        if isinstance(v, str):
            v = base * font_scalings.get(v, 1.0)
        out[k] = v * scale
    return out


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


# Row-by-row reading order for the model legend under the SFRD/SMD history
# plots: fiducials first, then the miniUchuu variants, then the Millennium ones.
_HISTORY_LEGEND_ORDER = [
    'SAGE26 (miniUchuu)', 'SAGE26 (Millennium)', 'SAGE16',
    'SAGE26 (miniUchuu beta=0.75)', 'SAGE26 (miniUchuu H2)',
    'SAGE26 (miniUchuu H2 + beta=0.75)',
    'SAGE26 (Millennium beta=0.75)', 'SAGE26 (Millennium H2)',
    'SAGE26 (Millennium H2 + beta=0.75)',
]


def _model_legend_below(fig, handles, labels, ncol=3, order=None):
    """
    Place a model legend in ``ncol`` columns beneath the x-axis label.

    The figure is made taller by the legend's height so the axes keep their
    size. Entries in *order* are laid out to read row by row.
    """
    if order is not None:
        rank = {l: i for i, l in enumerate(order)}
        pairs = sorted(zip(labels, handles), key=lambda p: rank.get(p[0], len(rank)))
        labels = [p[0] for p in pairs]
        handles = [p[1] for p in pairs]
    # matplotlib fills legend columns top to bottom; transpose so it reads by row
    nrow = -(-len(labels) // ncol)
    idx = [r * ncol + c for c in range(ncol) for r in range(nrow)
           if r * ncol + c < len(labels)]
    handles = [handles[i] for i in idx]
    labels = [labels[i] for i in idx]

    fig.tight_layout()
    leg = fig.legend(handles, labels, loc='lower center', ncol=ncol,
                     bbox_to_anchor=(0.5, 0.0), frameon=False, numpoints=1,
                     labelspacing=0.2, columnspacing=1.2, handlelength=2.0)
    for lh in leg.legend_handles:
        lh.set_alpha(1)

    # Grow the figure by the legend height (plus a small gap), and widen it if
    # the legend is wider than the figure. The original layout is kept, centred,
    # in the upper part so the axes do not change size.
    fig.canvas.draw()
    w, h = fig.get_size_inches()
    bbox = leg.get_window_extent()
    leg_w = bbox.width / fig.dpi + 0.2
    leg_h = bbox.height / fig.dpi + 0.15
    new_w, new_h = max(w, leg_w), h + leg_h
    fig.set_size_inches(new_w, new_h)
    x0 = 0.5 * (1.0 - w / new_w)
    fig.tight_layout(rect=(x0, leg_h / new_h, 1.0 - x0, 1.0))
    return leg


# ========================== OBSERVATIONAL DATA ==========================

def load_gama_smf_morph():
    """Load GAMA morphological SMF (Moffett et al. 2016)."""
    path = os.path.join(OBS_DIR, 'smf/gama_smf_morph.ecsv')
    data = np.genfromtxt(path, comments='#', skip_header=1)
    return {
        'mass': data[:, 0] + imf_shift('Moffett+16'),
        'E_HE': data[:, 1],
        'E_HE_err': data[:, 2],
        'D': data[:, 7],
        'D_err': data[:, 8],
    }


def load_baldry_blue_red():
    """Load Baldry et al. blue/red SMF data."""
    path = os.path.join(OBS_DIR, 'morphology/baldry_blue_red.csv')
    data = np.genfromtxt(path, delimiter=',', skip_header=2)
    return {
        'sf_mass': data[:, 0] + imf_shift('Baldry+12'),
        'sf_phi': data[:, 1],
        'q_mass': data[:, 2] + imf_shift('Baldry+12'),
        'q_phi': data[:, 3],
    }


def load_mzr_observations():
    """
    Load mass-metallicity relation observational data.

    Stellar masses are shifted onto the model's Chabrier scale via OBS_IMF. Every
    file here states its own IMF in its header except Curti+20, whose masses come
    from the MPA-JHU catalogue and so share the Kroupa scale of Tremonti+04 and
    Andrews & Martini+13 -- worth confirming against the paper.

    Returns a list of dicts, each with keys:
        'mass', 'Z', 'yerr' (optional), 'fmt', 'label'
    """
    obs = []

    # Tremonti et al. 2004
    path = os.path.join(OBS_DIR, 'metallicity/Tremonti04.dat')
    if os.path.exists(path):
        d = np.loadtxt(path)
        obs.append({
            'mass': d[:, 0] + imf_shift('Tremonti+04'), 'Z': d[:, 1],
            'yerr': [d[:, 1] - d[:, 2], d[:, 3] - d[:, 1]],
            'fmt': 'o', 'color': 'k', 'label': 'Tremonti+04',
        })
    else:
        # Polynomial fallback
        m = np.arange(7.0, 13.0, 0.1)
        z = -1.492 + 1.847 * m - 0.08026 * m * m
        obs.append({
            'mass': m + imf_shift('Tremonti+04'), 'Z': z, 'yerr': None,
            'fmt': 'o', 'color': 'k', 'label': 'Tremonti+04 (fit)',
        })

    # Curti et al. 2020
    path = os.path.join(OBS_DIR, 'metallicity/Curti2020.dat')
    if os.path.exists(path):
        d = np.loadtxt(path)
        obs.append({
            'mass': d[:, 0] + imf_shift('Curti+20'), 'Z': d[:, 1],
            'yerr': [d[:, 1] - d[:, 2], d[:, 3] - d[:, 1]],
            'fmt': 's', 'color': 'k', 'label': 'Curti+20',
        })

    # Andrews & Martini 2013
    path = os.path.join(OBS_DIR, 'metallicity/MMAdrews13.dat')
    if os.path.exists(path):
        d = np.loadtxt(path)
        obs.append({
            'mass': d[:, 0] + imf_shift('Andrews+13'), 'Z': d[:, 1],
            'yerr': [d[:, 1] - d[:, 2], d[:, 3] - d[:, 1]],
            'fmt': '^', 'color': 'k',
            'label': _tex_safe(r'Andrews \& Martini 2013'),
        })

    # Kewley & Ellison 2008 - T04 calibration
    path = os.path.join(OBS_DIR, 'metallicity/MMR-Kewley08.dat')
    if os.path.exists(path):
        d = np.loadtxt(path)
        obs.append({
            'mass': d[59:74, 0] + imf_shift('Kewley+08'), 'Z': d[59:74, 1], 'yerr': None,
            'fmt': 'd', 'color': 'k',
            'label': _tex_safe(r'Kewley \& Ellison 2008'),
        })

    # Gallazzi et al. 2005 (stellar -> gas-phase conversion)
    path = os.path.join(OBS_DIR, 'metallicity/MSZR-Gallazzi05.dat')
    if os.path.exists(path):
        d = np.loadtxt(path)
        m = d[7:, 0] + imf_shift('Gallazzi+05')
        z_gas = d[7:, 1] + 8.69
        z_lo = d[7:, 2] + 8.69
        z_hi = d[7:, 3] + 8.69
        obs.append({
            'mass': m, 'Z': z_gas,
            'yerr': [z_gas - z_lo, z_hi - z_gas],
            'fmt': 'v', 'color': 'k', 'label': 'Gallazzi+05 (conv.)',
        })

    return obs


def load_bh_bulge_observations():
    """
    Load black hole - bulge mass observational data.

    Returns
    -------
    dict with keys:
        'M_sph', 'M_BH', 'xerr', 'yerr', 'core' (boolean mask),
        'haring_rix_x', 'haring_rix_y' (relation line).
    """
    h_ratio = (0.7 / HUBBLE_H)**2

    M_BH_obs = h_ratio * 1e8 * np.array([
        39, 11, 0.45, 25, 24, 0.044, 1.4, 0.73, 9.0, 58, 0.10, 8.3, 0.39,
        0.42, 0.084, 0.66, 0.73, 15, 4.7, 0.083, 0.14, 0.15, 0.4, 0.12,
        1.7, 0.024, 8.8, 0.14, 2.0, 0.073, 0.77, 4.0, 0.17, 0.34, 2.4,
        0.058, 3.1, 1.3, 2.0, 97, 8.1, 1.8, 0.65, 0.39, 5.0, 3.3, 4.5,
        0.075, 0.68, 1.2, 0.13, 4.7, 0.59, 6.4, 0.79, 3.9, 47, 1.8, 0.06,
        0.016, 210, 0.014, 7.4, 1.6, 6.8, 2.6, 11, 37, 5.9, 0.31, 0.10,
        3.7, 0.55, 13, 0.11])
    M_BH_hi = h_ratio * 1e8 * np.array([
        4, 2, 0.17, 7, 10, 0.044, 0.9, 0.0, 0.9, 3.5, 0.10, 2.7, 0.26,
        0.04, 0.003, 0.03, 0.69, 2, 0.6, 0.004, 0.02, 0.09, 0.04, 0.005,
        0.2, 0.024, 10, 0.1, 0.5, 0.015, 0.04, 1.0, 0.01, 0.02, 0.3,
        0.008, 1.4, 0.5, 1.1, 30, 2.0, 0.6, 0.07, 0.01, 1.0, 0.9, 2.3,
        0.002, 0.13, 0.4, 0.08, 0.5, 0.03, 0.4, 0.38, 0.4, 10, 0.2,
        0.014, 0.004, 160, 0.014, 4.7, 0.3, 0.7, 0.4, 1, 18, 2.0, 0.004,
        0.001, 2.6, 0.26, 5, 0.005])
    M_BH_lo = h_ratio * 1e8 * np.array([
        5, 2, 0.10, 7, 10, 0.022, 0.3, 0.0, 0.8, 3.5, 0.05, 1.3, 0.09,
        0.04, 0.003, 0.03, 0.35, 2, 0.6, 0.004, 0.13, 0.1, 0.05, 0.005,
        0.2, 0.012, 2.7, 0.06, 0.5, 0.015, 0.06, 1.0, 0.02, 0.02, 0.3,
        0.008, 0.6, 0.5, 0.6, 26, 1.9, 0.3, 0.07, 0.01, 1.0, 2.5, 1.5,
        0.002, 0.13, 0.9, 0.08, 0.5, 0.09, 0.4, 0.33, 0.4, 10, 0.1,
        0.014, 0.004, 160, 0.007, 3.0, 0.4, 0.7, 1.5, 1, 11, 2.0, 0.004,
        0.001, 1.5, 0.19, 4, 0.005])
    M_sph_obs = h_ratio * 1e10 * np.array([
        69, 37, 1.4, 55, 27, 2.4, 0.46, 1.0, 19, 23, 0.61, 4.6, 11, 1.9,
        4.5, 1.4, 0.66, 4.7, 26, 2.0, 0.39, 0.35, 0.30, 3.5, 6.7, 0.88,
        1.9, 0.93, 1.24, 0.86, 2.0, 5.4, 1.2, 4.9, 2.0, 0.66, 5.1, 2.6,
        3.2, 100, 1.4, 0.88, 1.3, 0.56, 29, 6.1, 0.65, 3.3, 2.0, 6.9,
        1.4, 7.7, 0.9, 3.9, 1.8, 8.4, 27, 6.0, 0.43, 1.0, 122, 0.30, 29,
        11, 20, 2.8, 24, 78, 96, 3.6, 2.6, 55, 1.4, 64, 1.2])
    M_sph_hi = h_ratio * 1e10 * np.array([
        59, 32, 2.0, 80, 23, 3.5, 0.68, 1.5, 16, 19, 0.89, 6.6, 9, 2.7,
        6.6, 2.1, 0.91, 6.9, 22, 2.9, 0.57, 0.52, 0.45, 5.1, 5.7, 1.28,
        2.7, 1.37, 1.8, 1.26, 1.7, 4.7, 1.7, 7.1, 2.9, 0.97, 7.4, 3.8,
        2.7, 86, 2.1, 1.30, 1.9, 0.82, 25, 5.2, 0.96, 4.9, 3.0, 5.9, 1.2,
        6.6, 1.3, 5.7, 2.7, 7.2, 23, 5.2, 0.64, 1.5, 105, 0.45, 25, 10,
        17, 2.4, 20, 67, 83, 5.2, 3.8, 48, 2.0, 55, 1.8])
    M_sph_lo = h_ratio * 1e10 * np.array([
        32, 17, 0.8, 33, 12, 1.4, 0.28, 0.6, 9, 10, 0.39, 2.7, 5, 1.1,
        2.7, 0.8, 0.40, 2.8, 12, 1.2, 0.23, 0.21, 0.18, 2.1, 3.1, 0.52,
        1.1, 0.56, 0.7, 0.51, 0.9, 2.5, 0.7, 2.9, 1.2, 0.40, 3.0, 1.5,
        1.5, 46, 0.9, 0.53, 0.8, 0.34, 13, 2.8, 0.39, 2.0, 1.2, 3.2, 0.6,
        3.6, 0.5, 2.3, 1.1, 3.9, 12, 2.8, 0.26, 0.6, 57, 0.18, 13, 5, 9,
        1.3, 11, 36, 44, 2.1, 1.5, 26, 0.8, 30, 0.7])
    core = np.array([
        1,1,0,1,1,0,0,0,1,1,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,
        0,0,0,0,0,0,1,1,1,0,0,0,1,1,0,0,0,0,0,1,0,1,0,0,1,0,0,0,1,0,1,0,
        1,0,1,1,1,0,0,1,0,1,0], dtype=bool)

    # Log-space error bars
    yerr_hi = np.log10((M_BH_obs + M_BH_hi) / M_BH_obs)
    yerr_lo = -np.log10((M_BH_obs - M_BH_lo) / M_BH_obs)
    xerr_hi = np.log10((M_sph_obs + M_sph_hi) / M_sph_obs)
    xerr_lo = -np.log10((M_sph_obs - M_sph_lo) / M_sph_obs)

    # Haring & Rix 2004 relation
    hr_x = np.linspace(8, 13, 100)
    hr_y = 8.2 + 1.12 * (hr_x - 11.0)

    return {
        # Bulge masses: the IMF behind this compilation is not established, so no
        # shift is applied. See report_imf_audit().
        'log_M_sph': np.log10(M_sph_obs) + imf_shift('Scott+13 BH-bulge'),
        'log_M_BH': np.log10(M_BH_obs),
        'xerr': [xerr_lo, xerr_hi],
        'yerr': [yerr_lo, yerr_hi],
        'core': core,
        'haring_rix_x': hr_x,
        'haring_rix_y': hr_y,
    }

def load_shmr_observations():
    """
    Load stellar-to-halo mass relation observational data.

    Returns a dict with keys:
        'moster'   : dict with 'mvir', 'mstar' (best-fit line)
        'romeo'    : dict with 'mvir', 'mstar' (combined all + ETGs)
        'kravtsov' : dict with 'mvir', 'mstar', 'xerr_lo', 'xerr_hi',
                     'has_xerr' (combined ETGs + LTGs + Sat.)
        'taylor'   : dict with 'mvir', 'mstar', 'xerr', 'yerr'

    Stellar masses pass through imf_shift(); only Moster+13 has an established IMF
    (Chabrier, so no shift). Romeo+20, Kravtsov+18 and Taylor+20 are unresolved and
    currently unshifted -- see report_imf_audit().
    """
    obs = {}

    # Moster et al. 2013 — best-fit relation (first pair of columns)
    path = os.path.join(OBS_DIR, 'smhm/Moster_2013.csv')
    if os.path.exists(path):
        d = np.genfromtxt(path)
        valid = ~np.isnan(d[:, 0])
        obs['moster'] = {
            'mvir': d[valid, 0],
            'mstar': d[valid, 1] + imf_shift('Moster+13'),
        }

    # Romeo et al. 2020 — combined (all galaxies + ETGs)
    # Format: (log_Mvir, log_M*/Mvir)
    mvir_parts, mstar_parts = [], []
    for fname in ['smhm/Romeo20_SMHM.dat', 'smhm/Romeo20_SMHM_ETGs.dat']:
        path = os.path.join(OBS_DIR, fname)
        if os.path.exists(path):
            d = np.loadtxt(path)
            mvir_parts.append(d[:, 0])
            mstar_parts.append(d[:, 0] + d[:, 1] + imf_shift('Romeo+20'))
    if mvir_parts:
        obs['romeo'] = {
            'mvir': np.concatenate(mvir_parts),
            'mstar': np.concatenate(mstar_parts),
        }

    # Kravtsov et al. 2018 — combined (ETGs + LTGs + Sat/Clusters)
    k_mvir, k_mstar, k_xerr_lo, k_xerr_hi, k_has_xerr = [], [], [], [], []
    for fname in ['morphology/ETGs_Kravtsov18.dat', 'morphology/LTGs_Kravtsov18.dat']:
        path = os.path.join(OBS_DIR, fname)
        if os.path.exists(path):
            d = np.loadtxt(path)
            k_mvir.append(d[:, 0])
            k_mstar.append(d[:, 1] + imf_shift('Kravtsov+18'))
            k_xerr_lo.append(d[:, 0] - d[:, 2])
            k_xerr_hi.append(d[:, 3] - d[:, 0])
            k_has_xerr.append(np.ones(len(d), dtype=bool))
    path = os.path.join(OBS_DIR, 'morphology/SatKinsAndClusters_Kravtsov18.dat')
    if os.path.exists(path):
        d = np.loadtxt(path)
        k_mvir.append(d[:, 0])
        k_mstar.append(d[:, 1] + imf_shift('Kravtsov+18'))
        k_xerr_lo.append(np.zeros(len(d)))
        k_xerr_hi.append(np.zeros(len(d)))
        k_has_xerr.append(np.zeros(len(d), dtype=bool))
    if k_mvir:
        obs['kravtsov'] = {
            'mvir': np.concatenate(k_mvir),
            'mstar': np.concatenate(k_mstar),
            'xerr_lo': np.concatenate(k_xerr_lo),
            'xerr_hi': np.concatenate(k_xerr_hi),
            'has_xerr': np.concatenate(k_has_xerr),
        }

    # Taylor et al. 2020
    # File columns, per its own header:
    #   log10(m*/Msun)  log10(m*)_lo  log10(m*)_hi  Mhalo/1e12Msun  Mhalo_lo  Mhalo_hi
    # Note the stellar mass is logarithmic and the halo mass is LINEAR in units of
    # 1e12 Msun. This was previously read as (log_Mhalo, lo, hi, ratio, lo, hi), which
    # swapped the two axes and log10'd a halo mass as though it were a ratio -- putting
    # the points at log_Mvir ~ 10.3-10.6 with m*/Mvir up to 0.7, above the cosmic baryon
    # fraction and so unphysical.
    path = os.path.join(OBS_DIR, 'morphology/Taylor20.dat')
    if os.path.exists(path):
        d = np.atleast_2d(np.loadtxt(path))
        _t20 = imf_shift('Taylor+20')
        log_mstar = d[:, 0] + _t20
        log_mstar_lo = d[:, 1] + _t20
        log_mstar_hi = d[:, 2] + _t20
        log_mvir = np.log10(d[:, 3]) + 12.0
        log_mvir_lo = np.log10(d[:, 4]) + 12.0
        log_mvir_hi = np.log10(d[:, 5]) + 12.0
        log_ratio = log_mstar - log_mvir
        obs['taylor'] = {
            'mvir': log_mvir,
            'mstar': log_mstar,
            'xerr': [log_mvir - log_mvir_lo, log_mvir_hi - log_mvir],
            'yerr': [log_mstar - log_mstar_lo, log_mstar_hi - log_mstar],
            # Ratio uncertainty takes the outer corners of both intervals, so the bar
            # spans the full range the two independent measurements allow.
            'ratio': log_ratio,
            'ratio_err': [log_ratio - (log_mstar_lo - log_mvir_hi),
                          (log_mstar_hi - log_mvir_lo) - log_ratio],
        }

    return obs

def load_lange16_bulge_contours():
    """
    Bulge effective radius vs bulge mass from Lange et al. (2016, MNRAS 462, 1470).

    data/SizesAndAM/rbulge_L16.dat holds three blocks, each headed "#Nth percentile".
    They are not percentiles of size at fixed mass -- each block is a *closed contour*
    in the mass-size plane enclosing that fraction of the GAMA bulges. The giveaway is
    in the numbers: mass rises through the block and then falls back, and the last row
    repeats the first row's radius. Reading the file as three curves of r_e(M) instead,
    or concatenating the blocks into one array, produces a line that doubles back on
    itself and is not any relation.

    Returns a list of (enclosed_fraction, log10 M/Msun, log10 r_e/kpc) ordered outermost
    first, with each contour explicitly closed. Empty if the file is absent.
    """
    path = os.path.join(OBS_DIR, 'SizesAndAM/rbulge_L16.dat')
    if not os.path.exists(path):
        print(f'  Warning: {path} not found; Lange+16 contours omitted')
        return []

    blocks, frac = [], None
    with open(path) as f:
        for line in f:
            if line.startswith('#'):
                if 'percentile' in line:
                    frac = int(line.strip('#').split('th')[0])
                    blocks.append([frac, [], []])
                continue
            parts = line.split()
            if len(parts) >= 2 and blocks:
                blocks[-1][1].append(float(parts[0]))
                blocks[-1][2].append(float(parts[1]))

    contours = []
    for frac, mass, radius in blocks:
        if len(mass) < 3:
            continue
        # Close the loop: the file ends near, but not exactly on, its first point.
        log_m = np.log10(np.array(mass + mass[:1])) + imf_shift('Lange+16')
        log_r = np.log10(np.array(radius + radius[:1]))
        contours.append((frac, log_m, log_r))

    return sorted(contours, key=lambda c: -c[0])


def load_madau_dickinson_2014_data():
    """Load Madau and Dickinson 2014 SFRD data."""
    if not HAS_ASTROPY:
        return None, None, None, None
    filename = './data/sfrd/MandD_sfrd_2014.ecsv'
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None, None, None, None

    try:
        table = Table.read(filename, format='ascii.ecsv')
        z = table['z_min']
        # Madau & Dickinson (2014) quote their compilation for a Salpeter IMF;
        # shift it onto the model's Chabrier scale. The errors are in dex and so
        # are unchanged by the shift.
        re = table['log_psi'] + imf_shift('Madau+Dickinson 14')
        re_err_plus = table['e_log_psi_up']
        re_err_minus = table['e_log_psi_lo']
        return z, re, re_err_plus, re_err_minus
    except Exception as e:
        print(f"Error loading Madau and Dickinson 2014 SFRD data: {e}")
    return None, None, None, None

def load_madau_dickinson_smd_2014_data():
    """Load Madau and Dickinson 2014 SMD data."""
    if not HAS_ASTROPY:
        return None, None, None, None
    filename = './data/sfrd/MandD_smd_2014.ecsv'
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None, None, None, None

    try:
        table = Table.read(filename, format='ascii.ecsv')
        z = table['z_min']
        # Salpeter -> Chabrier, as for the SFRD compilation above. Keeps this
        # curve on the same footing as the COSMOS-Web SMD, which is Chabrier.
        re = table['log_rho'] + imf_shift('Madau+Dickinson 14')
        re_err_plus = table['e_log_rho_up']
        re_err_minus = table['e_log_rho_lo']
        return z, re, re_err_plus, re_err_minus
    except Exception as e:
        print(f"Error loading Madau and Dickinson 2014 SMD data: {e}")
    return None, None, None, None

def load_kikuchihara_smd_2020_data():
    """Load Kikuchihara et al. 2020 SMD data."""
    if not HAS_ASTROPY:
        return None, None, None, None
    filename = './data/sfrd/kikuchihara_smd_2020.ecsv'
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None, None, None, None

    try:
        table = Table.read(filename, format='ascii.ecsv')
        z = table['z']
        re = table['log_rho_star'] + imf_shift('Kikuchihara+20 SMD')
        re_err_plus = table['e_log_rho_star_upper']
        re_err_minus = table['e_log_rho_star_lower']
        return z, re, re_err_plus, re_err_minus
    except Exception as e:
        print(f"Error loading Kikuchihara 2020 SMD data: {e}")
    return None, None, None, None

def load_papovich_smd_2023_data():
    """Load Papovich et al. 2023 SMD data."""
    if not HAS_ASTROPY:
        return None, None, None, None
    filename = './data/sfrd/papovich_smd_2023.ecsv'
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None, None, None, None

    try:
        table = Table.read(filename, format='ascii.ecsv')
        z = table['z']
        re = table['log_rho_star'] + imf_shift('Papovich+23')
        re_err_plus = table['e_log_rho_star_upper']
        re_err_minus = table['e_log_rho_star_lower']
        return z, re, re_err_plus, re_err_minus
    except Exception as e:
        print(f"Error loading Papovich 2023 SMD data: {e}")
    return None, None, None, None

def load_oesch_sfrd_2018_data():
    """Load Oesch et al. 2018 SFRD data."""
    if not HAS_ASTROPY:
        return None, None, None, None
    filename = './data/sfrd/oesch_sfrd_2018.ecsv'
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None, None, None, None

    try:
        table = Table.read(filename, format='ascii.ecsv')
        z = table['z']
        re = table['log_rho_sfr'] + imf_shift('Oesch+18')
        re_err_plus = table['e_log_rho_sfr_upper']
        re_err_minus = table['e_log_rho_sfr_lower']
        return z, re, re_err_plus, re_err_minus
    except Exception as e:
        print(f"Error loading Oesch 2018 SFRD data: {e}")
    return None, None, None, None

def load_mcleod_rho_sfr_2024_data():
    """Load McLeod et al. 2024 SFR density data."""
    if not HAS_ASTROPY:
        return None, None, None, None
    filename = './data/sfrd/mcleod_rhouv_2024.ecsv'
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None, None, None, None

    try:
        table = Table.read(filename, format='ascii.ecsv')
        z = table['z']
        re = table['log_rho_sfr'] + imf_shift('McLeod+24')
        re_err_plus = np.zeros_like(re)
        re_err_minus = np.zeros_like(re)
        return z, re, re_err_plus, re_err_minus
    except Exception as e:
        print(f"Error loading McLeod 2024 SFRD data: {e}")
    return None, None, None, None

def load_harikane_sfr_density_2023_data():
    """Load Harikane et al. 2023 SFR density data."""
    if not HAS_ASTROPY:
        return None, None, None, None
    filename = './data/sfrd/harikane_density_2023.ecsv'
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None, None, None, None

    try:
        table = Table.read(filename, format='ascii.ecsv')
        z = table['z']
        re = table['log_rho_SFR_UV'] + imf_shift('Harikane+23')
        re_err_plus = table['e_log_rho_SFR_UV_upper']
        re_err_minus = table['e_log_rho_SFR_UV_lower']
        return z, re, re_err_plus, re_err_minus
    except Exception as e:
        print(f"Error loading Harikane 2023 SFRD data: {e}")
    return None, None, None, None

def load_brinchmann_sfr_mass_2004_data():
    """Load Brinchmann et al. 2004 SFR vs Stellar Mass data."""
    if not HAS_ASTROPY:
        return None, None
    filename = './data/sfr/Brinchmann04.dat'
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None, None

    try:
        # Read lines up to the stop marker
        data_lines = []
        with open(filename, 'r') as f:
            for line in f:
                if line.strip().startswith('#low boundary of 0.02 probability'):
                    break
                if line.strip().startswith('#') or not line.strip():
                    continue
                data_lines.append(line)
        # Use astropy Table to parse the collected lines
        from io import StringIO
        # Pass the list of lines directly as an iterable
        table = Table.read(
            data_lines,
            format='ascii.no_header',
            names=['log_mass', 'log_sfr'],
            delimiter=' ',  # whitespace
            guess=False,
            fast_reader=False
        )
        # Kroupa per the file header; mass and SFR both scale with the IMF's M/L.
        mass = table['log_mass'] + imf_shift('Brinchmann+04')
        sfr = table['log_sfr'] + imf_shift('Brinchmann+04')
        return mass, sfr
    except Exception as e:
        print(f"Error loading Brinchmann 2004 SFR-Mass data: {e}")
    return None, None

#
# Load Terrazas+17 MBH host galaxy SFR data
def load_terrazas17_mbh_host_sfr_data():
    """Load Terrazas et al. 2017 MBH host galaxy SFR data."""
    import numpy as np
    filename = './data/bh/MBH_host_gals_Terrazas17.dat'
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None, None
    try:
        data = np.loadtxt(filename, comments='#', usecols=(0,1))
        log_mstar = data[:,0] + imf_shift('Terrazas+17')
        sfr = data[:,1] + imf_shift('Terrazas+17')
        return log_mstar, sfr
    except Exception as e:
        print(f"Error loading Terrazas+17 MBH host SFR data: {e}")
        return None, None
    
# Load and process GAMA ProSpect Claudia data for SFR vs stellar mass
def load_gama_prospect_claudia(obsdir=None):
    """Load GAMA ProSpect Claudia data, apply SFR floor, and return log10(mass), log10(SFR)."""
    import numpy as np
    # If obsdir is given, use it; else assume data/ subdir
    filename = os.path.join(obsdir, 'GAMA/ProSpect_Claudia.txt') if obsdir else './data/morphology/ProSpect_Claudia.txt'
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None, None
    try:
        data = np.genfromtxt(filename, comments='#', usecols=(1,5))
        ms_gama = data[:,0]
        sfr_gama = data[:,1]
        sfr_gama[sfr_gama < 1e-3] = 1e-3
        log_ms = np.log10(ms_gama) + imf_shift('Bellstedt+20')
        log_sfr = np.log10(sfr_gama) + imf_shift('Bellstedt+20')
        return log_ms, log_sfr
    except Exception as e:
        print(f"Error loading GAMA ProSpect Claudia data: {e}")
        return None, None
    
# Load Bell+03 SMF starforming data
def load_bell_smf_sf_data():
    """Load Bell+03 SMF starforming data."""
    import numpy as np
    filename = './data/morphology/Bell_z0pt0_blue.dat'
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None, None
    try:
        data = np.loadtxt(filename, comments='#', usecols=(0,1,2,3))
        log_mstar = data[:,0] + imf_shift('Bell+03')
        sfr = data[:,1]
        error_high = data[:,2]
        error_low = data[:,3]
        return log_mstar, sfr, error_high, error_low
    except Exception as e:
        print(f"Error loading Bell+03 SMF starforming data: {e}")
        return None, None, None, None
    
# Load Bell+03 SMF quiescent data
def load_bell_smf_q_data():
    """Load Bell+03 SMF quiescent data."""
    import numpy as np
    filename = './data/morphology/Bell_z0pt0_red.dat'
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None, None
    try:
        data = np.loadtxt(filename, comments='#', usecols=(0,1,2,3))
        log_mstar = data[:,0] + imf_shift('Bell+03')
        sfr = data[:,1]
        error_high = data[:,2]
        error_low = data[:,3]
        return log_mstar, sfr, error_high, error_low
    except Exception as e:
        print(f"Error loading Bell+03 SMF starforming data: {e}")
        return None, None, None, None


def load_himf_observations():
    """
    Load HI mass function observations from Jones+18 and Zwaan+05.

    Returns a list of dicts, each with:
        'label': str
        'mass': array of log10(M_HI/Msun)
        'phi': array of log10(phi / Mpc^-3 dex^-1)
        'phi_lo': lower error (absolute phi values or error bars)
        'phi_hi': upper error
        'marker': plot marker style
        'color': plot color
    """
    observations = []

    # Jones et al. (2018) - ALFALFA 100
    jones_path = os.path.join(OBS_DIR, 'Gas/HIMF_Jones18.dat')
    if os.path.exists(jones_path):
        try:
            data = np.loadtxt(jones_path, comments='#')
            # Columns: log(MHI), log(phi), lower_bound, upper_bound
            # h=0.7 assumed in Jones+18, same as our simulation
            observations.append({
                'label': 'Jones+18 (ALFALFA)',
                'mass': data[:, 0],
                'phi': data[:, 1],
                'phi_lo': data[:, 2],  # These are absolute values, not errors
                'phi_hi': data[:, 3],
                'marker': 'o',
                'color': 'k',
            })
        except Exception as e:
            print(f"Warning: Could not load Jones+18 HIMF: {e}")

    # Zwaan et al. (2005) - HIPASS
    zwaan_path = os.path.join(OBS_DIR, 'Gas/HIMF_Zwaan2005.dat')
    if os.path.exists(zwaan_path):
        try:
            data = np.loadtxt(zwaan_path, comments='#')
            # Columns: log(MHI), log(Theta), lower_1sigma, upper_1sigma
            # h=0.75 assumed, need to convert to h=0.7
            # M_HI scales as h^-2, so log(M) shifts by -2*log(h_new/h_old)
            # phi scales as h^3, so log(phi) shifts by 3*log(h_new/h_old)
            h_zwaan = 0.75
            h_ours = 0.7  # HUBBLE_H from simulation
            h_ratio = h_ours / h_zwaan
            mass_shift = -2.0 * np.log10(h_ratio)
            phi_shift = 3.0 * np.log10(h_ratio)

            observations.append({
                'label': 'Zwaan+05 (HIPASS)',
                'mass': data[:, 0] + mass_shift,
                'phi': data[:, 1] + phi_shift,
                # Errors are relative (sigma values to add/subtract)
                'phi_err_lo': data[:, 2],  # These are error magnitudes
                'phi_err_hi': data[:, 3],
                'marker': 's',
                'color': 'gray',
            })
        except Exception as e:
            print(f"Warning: Could not load Zwaan+05 HIMF: {e}")

    return observations


def _gasmf_obs_yerr(obs, mask):
    """Normalise the two error conventions used by the gas MF observation loaders.

    Jones+18 and the H2 sets store absolute phi bounds in 'phi_lo'/'phi_hi';
    Zwaan+05 stores error magnitudes in 'phi_err_lo'/'phi_err_hi'.
    """
    phi = obs['phi'][mask]
    if 'phi_err_lo' in obs:
        return [obs['phi_err_lo'][mask], obs['phi_err_hi'][mask]]
    return [phi - obs['phi_lo'][mask], obs['phi_hi'][mask] - phi]


def load_h2mf_observations():
    """
    Load H2 mass function observations from Fletcher+21 and Boselli+14.

    Returns a list of dicts with 'label', 'mass', 'phi', 'phi_lo', 'phi_hi',
    'marker', 'color'.
    """
    observations = []

    # Boselli et al. (2014) - H2 mass function
    # The file contains two sections: constant X_CO and luminosity-dependent X_CO.
    path_b14 = os.path.join(OBS_DIR, 'Gas', 'B14_MH2MF.dat')
    if os.path.exists(path_b14):
        try:
            with open(path_b14, 'r') as fh:
                lines = fh.readlines()

            sections = []
            current = []
            for line in lines:
                stripped = line.strip()
                if not stripped or stripped.startswith('#'):
                    if current:
                        sections.append(np.array(current, dtype=float))
                        current = []
                    continue
                current.append([float(x) for x in stripped.split()[:4]])
            if current:
                sections.append(np.array(current, dtype=float))

            if len(sections) >= 1:
                data = sections[0]
                observations.append({
                    'label': 'Boselli+14 (const. X_CO)',
                    'mass': data[:, 0],
                    'phi': data[:, 1],
                    'phi_lo': data[:, 2],
                    'phi_hi': data[:, 3],
                    'marker': 'D',
                    'color': 'gray',
                    'edgecolor': 'k',
                })
            if len(sections) >= 2:
                data = sections[1]
                observations.append({
                    'label': 'Boselli+14 (lum.-dep. X_CO)',
                    'mass': data[:, 0],
                    'phi': data[:, 1],
                    'phi_lo': data[:, 2],
                    'phi_hi': data[:, 3],
                    'marker': '^',
                    'color': 'gray',
                    'edgecolor': 'k',
                })
        except Exception as e:
            print(f"  Warning: could not load {path_b14}: {e}")

    # Fletcher et al. (2021) - Detected + Non-detected
    path_det = os.path.join(OBS_DIR, 'Gas/H2MF_Fletcher21_DetNonDet.dat')
    if os.path.exists(path_det):
        try:
            data = np.loadtxt(path_det, comments='#')
            observations.append({
                'label': 'Fletcher+20 (Det+NonDet)',
                'mass': data[:, 0],
                'phi': data[:, 1],
                'phi_lo': data[:, 2],
                'phi_hi': data[:, 3],
                'marker': 's',
                'color': 'k',
                'edgecolor': 'k',
            })
        except Exception as e:
            print(f"  Warning: could not load {path_det}: {e}")

    # Fletcher et al. (2021) - Estimated
    path_est = os.path.join(OBS_DIR, 'Gas/H2MF_Fletcher21_Estimated.dat')
    if os.path.exists(path_est):
        try:
            data = np.loadtxt(path_est, comments='#')
            observations.append({
                'label': 'Fletcher+20 (Estimated)',
                'mass': data[:, 0],
                'phi': data[:, 1],
                'phi_lo': data[:, 2],
                'phi_hi': data[:, 3],
                'marker': 'o',
                'color': 'gray',
                'edgecolor': 'k',
            })
        except Exception as e:
            print(f"  Warning: could not load {path_est}: {e}")

    return observations


# ========================== PLOT 1: COSMIC SFR DENSITY HISTORY (CSRDH) ==========================

def plot_1_sfrd_history():
    """
    Plot G: Cosmic SFR Density History.
    Replicates the structure of the uploaded paper_plots.py.
    Iterates snapshots 0-63 explicitly to capture full history.
    """
    print('Plot 1: SFR Density History (CSRDH)')
    
    # --- 1. SETUP & DEFINITIONS ---
    # Exact redshift list from the uploaded script
    redshifts = np.array([
        127.000, 79.998, 50.000, 30.000, 19.916, 18.244, 16.725, 15.343, 14.086, 12.941, 
        11.897, 10.944, 10.073, 9.278, 8.550, 7.883, 7.272, 6.712, 6.197, 5.724, 
        5.289, 4.888, 4.520, 4.179, 3.866, 3.576, 3.308, 3.060, 2.831, 2.619, 
        2.422, 2.239, 2.070, 1.913, 1.766, 1.630, 1.504, 1.386, 1.276, 1.173, 
        1.078, 0.989, 0.905, 0.828, 0.755, 0.687, 0.624, 0.564, 0.509, 0.457, 
        0.408, 0.362, 0.320, 0.280, 0.242, 0.208, 0.175, 0.144, 0.116, 0.089, 
        0.064, 0.041, 0.020, 0.000
    ])

    FirstSnap = 0
    LastSnap = 63
    
    # Define models to process
    # We construct a list similar to SFR_SimDirs
    redshifts_mu = np.array(MINIUCHUU_REDSHIFTS)
    sim_dirs = []

    # 1. Primary Model (SAGE26 Millennium)
    if os.path.exists(PRIMARY_DIR):
        sim_dirs.append({
            'path': PRIMARY_DIR, 'label': 'SAGE26 (Millennium)',
            'color': MODEL_COLORS['sage26'], 'ls': '--', 'lw': 1.5,
            'redshifts': redshifts, 'first_snap': FirstSnap, 'last_snap': LastSnap,
            'volume': VOLUME,
        })

    # 2. Vanilla Model (C16)
    if os.path.exists(VANILLA_DIR):
        sim_dirs.append({
            'path': VANILLA_DIR, 'label': 'SAGE16',
            'color': MODEL_COLORS['sage16'], 'ls': '-.', 'lw': 1.5,
            'redshifts': redshifts, 'first_snap': FirstSnap, 'last_snap': LastSnap,
            'volume': VOLUME,
        })

    # 3. miniUchuu Model
    if os.path.exists(MINIUCHUU_DIR):
        sim_dirs.append({
            'path': MINIUCHUU_DIR, 'label': 'SAGE26 (miniUchuu)',
            'color': MODEL_COLORS['sage26'], 'ls': '-', 'lw': 1.5,
            'redshifts': redshifts_mu, 'first_snap': MINIUCHUU_FIRST_SNAP, 'last_snap': MINIUCHUU_LAST_SNAP,
            'volume': MINIUCHUU_VOLUME,
        })

    # 4. Millennium Beta_clock=0.75 model
    if os.path.exists(BETA_CLOCK_MILLENNIUM_075_DIR):
        sim_dirs.append({
            'path': BETA_CLOCK_MILLENNIUM_075_DIR, 'label': 'SAGE26 (Millennium beta=0.75)',
            'color': MODEL_COLORS['beta'], 'ls': '--', 'lw': 0.8,
            'redshifts': redshifts, 'first_snap': FirstSnap, 'last_snap': LastSnap,
            'volume': VOLUME,
        })

    # 5. Millennium H2 depletion model
    if os.path.exists(H2_DEPLETE_MILLENNIUM_DIR):
        sim_dirs.append({
            'path': H2_DEPLETE_MILLENNIUM_DIR, 'label': 'SAGE26 (Millennium H2)',
            'color': MODEL_COLORS['h2'], 'ls': '--', 'lw': 0.8,
            'redshifts': redshifts, 'first_snap': FirstSnap, 'last_snap': LastSnap,
            'volume': VOLUME,
        })

    # 6. Millennium H2 depletion + β=0.75 model
    if os.path.exists(BETA_CLOCK_075_H2_DEPLETE_MILLENNIUM_DIR):
        sim_dirs.append({
            'path': BETA_CLOCK_075_H2_DEPLETE_MILLENNIUM_DIR, 'label': 'SAGE26 (Millennium H2 + beta=0.75)',
            'color': MODEL_COLORS['h2_beta'], 'ls': '--', 'lw': 0.8,
            'redshifts': redshifts, 'first_snap': FirstSnap, 'last_snap': LastSnap,
            'volume': VOLUME,
        })

    # 7. miniUchuu Beta_clock=0.75 model
    if os.path.exists(BETA_CLOCK_MINIUCHUU_075_DIR):
        sim_dirs.append({
            'path': BETA_CLOCK_MINIUCHUU_075_DIR, 'label': 'SAGE26 (miniUchuu beta=0.75)',
            'color': MODEL_COLORS['beta'], 'ls': '-', 'lw': 1.5,
            'redshifts': redshifts_mu, 'first_snap': MINIUCHUU_FIRST_SNAP, 'last_snap': MINIUCHUU_LAST_SNAP,
            'volume': MINIUCHUU_VOLUME,
        })
    
    # 8. miniUchuu H2 depletion model
    if os.path.exists(H2_DEPLETE_MINIUCHUU_DIR):
        sim_dirs.append({
            'path': H2_DEPLETE_MINIUCHUU_DIR, 'label': 'SAGE26 (miniUchuu H2)',
            'color': MODEL_COLORS['h2'], 'ls': '-', 'lw': 1.5,
            'redshifts': redshifts_mu, 'first_snap': MINIUCHUU_FIRST_SNAP, 'last_snap': MINIUCHUU_LAST_SNAP,
            'volume': MINIUCHUU_VOLUME,
        })

    # 9. miniUchuu H2 depletion + β=0.75 model
    if os.path.exists(BETA_CLOCK_075_H2_DEPLETE_MINIUCHUU_DIR):
        sim_dirs.append({
            'path': BETA_CLOCK_075_H2_DEPLETE_MINIUCHUU_DIR, 'label': 'SAGE26 (miniUchuu H2 + beta=0.75)',
            'color': MODEL_COLORS['h2_beta'], 'ls': '-', 'lw': 1.5,
            'redshifts': redshifts_mu, 'first_snap': MINIUCHUU_FIRST_SNAP, 'last_snap': MINIUCHUU_LAST_SNAP,
            'volume': MINIUCHUU_VOLUME,
        })


    fig = plt.figure(figsize=HISTORY_FIGSIZE)
    ax = fig.add_subplot(111)

    # --- 2. PLOT OBSERVATIONAL DATA (Croton et al. 2006 Compilation) ---
    # Exact array from uploaded file
    ObsSFRdensity = np.array([
        [0, 0.0158489, 0, 0, 0.0251189, 0.01000000],
        [0.150000, 0.0173780, 0, 0.300000, 0.0181970, 0.0165959],
        [0.0425000, 0.0239883, 0.0425000, 0.0425000, 0.0269153, 0.0213796],
        [0.200000, 0.0295121, 0.100000, 0.300000, 0.0323594, 0.0269154],
        [0.350000, 0.0147911, 0.200000, 0.500000, 0.0173780, 0.0125893],
        [0.625000, 0.0275423, 0.500000, 0.750000, 0.0331131, 0.0229087],
        [0.825000, 0.0549541, 0.750000, 1.00000, 0.0776247, 0.0389045],
        [0.625000, 0.0794328, 0.500000, 0.750000, 0.0954993, 0.0660693],
        [0.700000, 0.0323594, 0.575000, 0.825000, 0.0371535, 0.0281838],
        [1.25000, 0.0467735, 1.50000, 1.00000, 0.0660693, 0.0331131],
        [0.750000, 0.0549541, 0.500000, 1.00000, 0.0389045, 0.0776247],
        [1.25000, 0.0741310, 1.00000, 1.50000, 0.0524807, 0.104713],
        [1.75000, 0.0562341, 1.50000, 2.00000, 0.0398107, 0.0794328],
        [2.75000, 0.0794328, 2.00000, 3.50000, 0.0562341, 0.112202],
        [4.00000, 0.0309030, 3.50000, 4.50000, 0.0489779, 0.0194984],
        [0.250000, 0.0398107, 0.00000, 0.500000, 0.0239883, 0.0812831],
        [0.750000, 0.0446684, 0.500000, 1.00000, 0.0323594, 0.0776247],
        [1.25000, 0.0630957, 1.00000, 1.50000, 0.0478630, 0.109648],
        [1.75000, 0.0645654, 1.50000, 2.00000, 0.0489779, 0.112202],
        [2.50000, 0.0831764, 2.00000, 3.00000, 0.0512861, 0.158489],
        [3.50000, 0.0776247, 3.00000, 4.00000, 0.0416869, 0.169824],
        [4.50000, 0.0977237, 4.00000, 5.00000, 0.0416869, 0.269153],
        [5.50000, 0.0426580, 5.00000, 6.00000, 0.0177828, 0.165959],
        [3.00000, 0.120226, 2.00000, 4.00000, 0.173780, 0.0831764],
        [3.04000, 0.128825, 2.69000, 3.39000, 0.151356, 0.109648],
        [4.13000, 0.114815, 3.78000, 4.48000, 0.144544, 0.0912011],
        [0.350000, 0.0346737, 0.200000, 0.500000, 0.0537032, 0.0165959],
        [0.750000, 0.0512861, 0.500000, 1.00000, 0.0575440, 0.0436516],
        [1.50000, 0.0691831, 1.00000, 2.00000, 0.0758578, 0.0630957],
        [2.50000, 0.147911, 2.00000, 3.00000, 0.169824, 0.128825],
        [3.50000, 0.0645654, 3.00000, 4.00000, 0.0776247, 0.0512861],
    ], dtype=np.float32)

    ObsRedshift = ObsSFRdensity[:, 0]
    xErrLo = np.abs(ObsSFRdensity[:, 0]-ObsSFRdensity[:, 2])
    xErrHi = np.abs(ObsSFRdensity[:, 3]-ObsSFRdensity[:, 0])
    ObsSFR = np.log10(ObsSFRdensity[:, 1])
    yErrLo = np.abs(np.log10(ObsSFRdensity[:, 1])-np.log10(ObsSFRdensity[:, 4]))
    yErrHi = np.abs(np.log10(ObsSFRdensity[:, 5])-np.log10(ObsSFRdensity[:, 1]))

    ax.errorbar(ObsRedshift, ObsSFR, yerr=[yErrLo, yErrHi], xerr=[xErrLo, xErrHi],
                fmt='o', markerfacecolor='gray', markeredgecolor='k', markeredgewidth=1.0,
                ecolor='k', color='k', ms=8, lw=1.0, alpha=0.6, ls='none',
                zorder=Z_OBS, label='Somerville+01 compilation')

    # Madau & Dickinson 2014 Fit
    def MD14_sfrd(z):
        psi = 0.015 * (1+z)**2.7 / (1 + ((1+z)/2.9)**5.6)
        return psi

    # Their eq. 15 is a Salpeter-IMF fit, so it comes DOWN onto the model's
    # Chabrier scale. This previously multiplied by 1/0.63, raising it by 0.2 dex
    # instead of lowering it by 0.24 -- a 0.44 dex error in the wrong direction,
    # which made the model look ~0.2 dex low against this curve while looking
    # ~0.2 dex high against the natively-Chabrier COSMOS-Web curve beside it.
    z_values = np.linspace(0, 8, 200)
    md14 = np.log10(MD14_sfrd(z_values)) + imf_shift('Madau+Dickinson 14')
    ax.plot(z_values, md14, color='gray', lw=1.5, alpha=0.6, zorder=Z_OBS,
            label=_tex_safe(r'Madau \& Dickinson 2014'))

    # --- 3. PROCESS & PLOT MODELS ---
    N_BOOT = 100
    model_results = {}  # Store results for comparison
    for sim in sim_dirs:
        sim_path = sim['path']
        sim_label = sim['label']
        sim_redshifts = sim['redshifts']
        sim_first = sim['first_snap']
        sim_last = sim['last_snap']
        sim_volume = sim['volume']
        do_bootstrap = True  # bootstrap error shading for all three models
        n_snaps = sim_last - sim_first + 1
        sfr_density = np.zeros(n_snaps)
        sfr_density_lo = np.zeros(n_snaps)
        sfr_density_hi = np.zeros(n_snaps)

        # Loop strictly from first_snap to last_snap
        for snap in range(sim_first, sim_last + 1):
            snap_name = f'Snap_{snap}'
            idx = snap - sim_first

            try:
                model_files = find_model_files(sim_path)
                d = read_snap_from_files(model_files, snap_name,
                                         ['SfrDisk', 'SfrBulge'])
                if d:
                    sfr_disk = d['SfrDisk']
                    sfr_bulge = d['SfrBulge']
                    sfr_total = sfr_disk + sfr_bulge
                    sfr_density[idx] = np.sum(sfr_total) / sim_volume

                    if do_bootstrap and len(sfr_total) > 0:
                        n_gal = len(sfr_total)
                        boot = np.array([
                            np.sum(sfr_total[np.random.randint(0, n_gal, n_gal)])
                            for _ in range(N_BOOT)
                        ]) / sim_volume
                        sfr_density_lo[idx] = np.percentile(boot, 16)
                        sfr_density_hi[idx] = np.percentile(boot, 84)
            except Exception:
                continue

        # Plot
        nonzero = np.where(sfr_density > 0.0)[0]
        if len(nonzero) > 0:
            z_vals = sim_redshifts[sim_first:sim_last+1]
            ax.plot(z_vals[nonzero], np.log10(sfr_density[nonzero]),
                    lw=sim['lw'], color=sim['color'], linestyle=sim['ls'],
                    zorder=Z_MODEL_LINE, label=sim_label)

            if do_bootstrap:
                valid = nonzero[sfr_density_lo[nonzero] > 0]
                if len(valid) > 0:
                    ax.fill_between(z_vals[valid],
                                    np.log10(sfr_density_lo[valid]),
                                    np.log10(sfr_density_hi[valid]),
                                    color=sim['color'], alpha=0.2,
                                    zorder=Z_MODEL_BAND)

            # Store results for comparison
            model_results[sim_label] = {
                'z': z_vals[nonzero],
                'sfrd': np.log10(sfr_density[nonzero])
            }

    # --- COSMOS-Web ---
    if HAS_ASTROPY:
        csfrd_file = './data/sfrd/CSFRD_inferred_from_SMD.ecsv'
        if os.path.exists(csfrd_file):
            try:
                csfrd_table = Table.read(csfrd_file, format='ascii.ecsv')
                z_csfrd = np.array(csfrd_table['Redshift'])
                sfrd_50 = np.log10(np.array(csfrd_table['sfrd_50']))
                sfrd_16 = np.log10(np.array(csfrd_table['sfrd_16']))
                sfrd_84 = np.log10(np.array(csfrd_table['sfrd_84']))
                # Densely sampled (~4000 pts) inferred curve -> subsample to
                # discrete markers spaced ~0.25 in redshift so they read as points.
                dz = np.median(np.diff(z_csfrd)) if len(z_csfrd) > 1 else 0.25
                stride = max(1, int(round(0.25 / dz))) if dz > 0 else 1
                sel = slice(None, None, stride)
                ax.errorbar(z_csfrd[sel], sfrd_50[sel],
                            yerr=[(sfrd_50 - sfrd_16)[sel], (sfrd_84 - sfrd_50)[sel]],
                            fmt='s', color='k', markerfacecolor='gray',
                            markeredgecolor='k', markeredgewidth=1.0, ecolor='k',
                            ms=8, lw=1.0, alpha=0.6, ls='none',
                            zorder=Z_OBS, label='COSMOS-Web')
            except Exception as e:
                print(f"Error loading CSFRD inferred from SMD: {e}")

    # --- 4. FORMATTING ---
    ax.set_ylabel(r'$\log_{10}\ {\rho_{\rm SFR}}\ (M_{\odot}\ \mathrm{yr}^{-1}\ \mathrm{Mpc}^{-3})$')
    ax.set_xlabel(r'$\mathrm{Redshift}$')
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.set_xlim(0.0, 7.5)
    ax.set_ylim(-3.0, -0.5)

    sim_names = {sim['label'] for sim in sim_dirs}
    handles, labels = ax.get_legend_handles_labels()
    sim_h = [h for h, l in zip(handles, labels) if l in sim_names]
    sim_l = [l for l in labels if l in sim_names]
    obs_h = [h for h, l in zip(handles, labels) if l not in sim_names]
    obs_l = [l for l in labels if l not in sim_names]
    _standard_legend(ax, loc='lower left', handles=obs_h, labels=obs_l)
    _model_legend_below(fig, sim_h, sim_l, order=_HISTORY_LEGEND_ORDER)

    outputFile = os.path.join(OUTPUT_DIR, 'SFR_Density_History_Comparison' + OUTPUT_FORMAT)
    save_figure(fig, outputFile)


# ========================== PLOT 2: STELLAR MASS DENSITY HISTORY (SMDH) ==========================

def plot_2_smd_history():
    """
    Plot H: Stellar Mass Density History.
    Replicates structure of uploaded paper_plots.py using explicit snap loops 0-63.
    """
    print('Plot 2: Stellar Mass Density History (SMDH)')
    
    # --- 1. SETUP ---
    redshifts = np.array([
        127.000, 79.998, 50.000, 30.000, 19.916, 18.244, 16.725, 15.343, 14.086, 12.941, 
        11.897, 10.944, 10.073, 9.278, 8.550, 7.883, 7.272, 6.712, 6.197, 5.724, 
        5.289, 4.888, 4.520, 4.179, 3.866, 3.576, 3.308, 3.060, 2.831, 2.619, 
        2.422, 2.239, 2.070, 1.913, 1.766, 1.630, 1.504, 1.386, 1.276, 1.173, 
        1.078, 0.989, 0.905, 0.828, 0.755, 0.687, 0.624, 0.564, 0.509, 0.457, 
        0.408, 0.362, 0.320, 0.280, 0.242, 0.208, 0.175, 0.144, 0.116, 0.089, 
        0.064, 0.041, 0.020, 0.000
    ])

    FirstSnap = 0
    LastSnap = 63
    
    redshifts_mu = np.array(MINIUCHUU_REDSHIFTS)
    sim_dirs = []
    if os.path.exists(PRIMARY_DIR):
        sim_dirs.append({
            'path': PRIMARY_DIR, 'label': 'SAGE26 (Millennium)', 'color': MODEL_COLORS['sage26'], 'ls': '--', 'lw': 1.5,
            'redshifts': redshifts, 'first_snap': FirstSnap, 'last_snap': LastSnap,
            'volume': VOLUME, 'mass_convert': MASS_CONVERT,
        })
    if os.path.exists(VANILLA_DIR):
        sim_dirs.append({
            'path': VANILLA_DIR, 'label': 'SAGE16', 'color': MODEL_COLORS['sage16'], 'ls': '--', 'lw': 2.0,
            'redshifts': redshifts, 'first_snap': FirstSnap, 'last_snap': LastSnap,
            'volume': VOLUME, 'mass_convert': MASS_CONVERT,
        })
    if os.path.exists(MINIUCHUU_DIR):
        sim_dirs.append({
            'path': MINIUCHUU_DIR, 'label': 'SAGE26 (miniUchuu)', 'color': MODEL_COLORS['sage26'], 'ls': '-', 'lw': 1.5,
            'redshifts': redshifts_mu, 'first_snap': MINIUCHUU_FIRST_SNAP, 'last_snap': MINIUCHUU_LAST_SNAP,
            'volume': MINIUCHUU_VOLUME, 'mass_convert': MINIUCHUU_MASS_CONVERT,
        })
    if os.path.exists(BETA_CLOCK_MILLENNIUM_075_DIR):
        sim_dirs.append({
            'path': BETA_CLOCK_MILLENNIUM_075_DIR, 'label': 'SAGE26 (Millennium beta=0.75)', 'color': MODEL_COLORS['beta'], 'ls': '--', 'lw': 0.8,
            'redshifts': redshifts, 'first_snap': FirstSnap, 'last_snap': LastSnap,
            'volume': VOLUME, 'mass_convert': MASS_CONVERT,
        })
    if os.path.exists(H2_DEPLETE_MILLENNIUM_DIR):
        sim_dirs.append({
            'path': H2_DEPLETE_MILLENNIUM_DIR, 'label': 'SAGE26 (Millennium H2)', 'color': MODEL_COLORS['h2'], 'ls': '--', 'lw': 0.8,
            'redshifts': redshifts, 'first_snap': FirstSnap, 'last_snap': LastSnap,
            'volume': VOLUME, 'mass_convert': MASS_CONVERT,
        })
    if os.path.exists(BETA_CLOCK_075_H2_DEPLETE_MILLENNIUM_DIR):
        sim_dirs.append({
            'path': BETA_CLOCK_075_H2_DEPLETE_MILLENNIUM_DIR, 'label': 'SAGE26 (Millennium H2 + beta=0.75)', 'color': MODEL_COLORS['h2_beta'], 'ls': '--', 'lw': 0.8,
            'redshifts': redshifts, 'first_snap': FirstSnap, 'last_snap': LastSnap,
            'volume': VOLUME, 'mass_convert': MASS_CONVERT,
        })
    if os.path.exists(BETA_CLOCK_MINIUCHUU_075_DIR):
        sim_dirs.append({
            'path': BETA_CLOCK_MINIUCHUU_075_DIR, 'label': 'SAGE26 (miniUchuu beta=0.75)', 'color': MODEL_COLORS['beta'], 'ls': '-', 'lw': 1.5,
            'redshifts': redshifts_mu, 'first_snap': MINIUCHUU_FIRST_SNAP, 'last_snap': MINIUCHUU_LAST_SNAP,
            'volume': MINIUCHUU_VOLUME, 'mass_convert': MINIUCHUU_MASS_CONVERT,
        })
    if os.path.exists(H2_DEPLETE_MINIUCHUU_DIR):
        sim_dirs.append({
            'path': H2_DEPLETE_MINIUCHUU_DIR, 'label': 'SAGE26 (miniUchuu H2)', 'color': MODEL_COLORS['h2'], 'ls': '-', 'lw': 1.5,
            'redshifts': redshifts_mu, 'first_snap': MINIUCHUU_FIRST_SNAP, 'last_snap': MINIUCHUU_LAST_SNAP,
            'volume': MINIUCHUU_VOLUME, 'mass_convert': MINIUCHUU_MASS_CONVERT,
        })
    if os.path.exists(BETA_CLOCK_075_H2_DEPLETE_MINIUCHUU_DIR):
        sim_dirs.append({
            'path': BETA_CLOCK_075_H2_DEPLETE_MINIUCHUU_DIR, 'label': 'SAGE26 (miniUchuu H2 + beta=0.75)', 'color': MODEL_COLORS['h2_beta'], 'ls': '-', 'lw': 1.5,
            'redshifts': redshifts_mu, 'first_snap': MINIUCHUU_FIRST_SNAP, 'last_snap': MINIUCHUU_LAST_SNAP,
            'volume': MINIUCHUU_VOLUME, 'mass_convert': MINIUCHUU_MASS_CONVERT,
        })
    # for m in FFB_MODELS:
    #     if os.path.exists(m['dir']):
    #         sim_dirs.append({'path': m['dir'], 'label': m['name'], 'color': 'blue', 'ls': ':'})

    fig = plt.figure(figsize=HISTORY_FIGSIZE)
    ax = fig.add_subplot(111)

    # --- 2. PLOT OBSERVATIONAL DATA (Marchesini et al. 2009 Compilation) ---
    # Values are (minz, maxz, rho,-err,+err)
    dickenson2003 = np.array(((0.6,1.4,8.26,0.08,0.08),(1.4,2.0,7.86,0.22,0.33),
                     (2.0,2.5,7.58,0.29,0.54),(2.5,3.0,7.52,0.51,0.48)),float)
    drory2005 = np.array(((0.25,0.75,8.3,0.15,0.15),(0.75,1.25,8.16,0.15,0.15),
                (1.25,1.75,8.0,0.16,0.16),(1.75,2.25,7.85,0.2,0.2),
                (2.25,3.0,7.75,0.2,0.2),(3.0,4.0,7.58,0.2,0.2)),float)
    PerezGonzalez2008 = np.array(((0.2,0.4,8.41,0.06,0.06),(0.4,0.6,8.37,0.04,0.04),
             (0.6,0.8,8.32,0.05,0.05),(0.8,1.0,8.24,0.05,0.05),
             (1.0,1.3,8.15,0.05,0.05),(1.3,1.6,7.95,0.07,0.07),
             (1.6,2.0,7.82,0.07,0.07),(2.0,2.5,7.67,0.08,0.08),
             (2.5,3.0,7.56,0.18,0.18),(3.0,3.5,7.43,0.14,0.14),
             (3.5,4.0,7.29,0.13,0.13)),float)
    glazebrook2004 = np.array(((0.8,1.1,7.98,0.14,0.1),(1.1,1.3,7.62,0.14,0.11),
                     (1.3,1.6,7.9,0.14,0.14),(1.6,2.0,7.49,0.14,0.12)),float)
    fontana2006 = np.array(((0.4,0.6,8.26,0.03,0.03),(0.6,0.8,8.17,0.02,0.02),
                  (0.8,1.0,8.09,0.03,0.03),(1.0,1.3,7.98,0.02,0.02),
                  (1.3,1.6,7.87,0.05,0.05),(1.6,2.0,7.74,0.04,0.04),
                  (2.0,3.0,7.48,0.04,0.04),(3.0,4.0,7.07,0.15,0.11)),float)
    rudnick2006 = np.array(((0.0,1.0,8.17,0.27,0.05),(1.0,1.6,7.99,0.32,0.05),
                  (1.6,2.4,7.88,0.34,0.09),(2.4,3.2,7.71,0.43,0.08)),float)
    elsner2008 = np.array(((0.25,0.75,8.37,0.03,0.03),(0.75,1.25,8.17,0.02,0.02),
                 (1.25,1.75,8.02,0.03,0.03),(1.75,2.25,7.9,0.04,0.04),
                 (2.25,3.0,7.73,0.04,0.04),(3.0,4.0,7.39,0.05,0.05)),float)
    
    obs = (dickenson2003,drory2005,PerezGonzalez2008,glazebrook2004,fontana2006,rudnick2006,elsner2008)
    whichimf = 1  # 1 = Chabrier
    
    # Define your colors list
    obs_colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'orange', 'purple']
    
    label_added = False
    
    # Use enumerate(obs) to get both the index (i) and the data (o)
    for i, o in enumerate(obs):
        xval = ((o[:,1]-o[:,0])/2.)+o[:,0]
        if(whichimf == 0):
            yval = np.log10(10**o[:,2] *1.6)
        elif(whichimf == 1):
            yval = np.log10(10**o[:,2] *1.6/1.8)

        # Select color safely
        current_color = obs_colors[i % len(obs_colors)]
            
        lbl = 'Observations' if not label_added else None
        
        ax.errorbar(xval, yval, xerr=(xval-o[:,0], o[:,1]-xval), yerr=(o[:,3], o[:,4]), 
                    alpha=0.4, lw=1.0, marker='o', ls='none', label=lbl, 
                    markeredgecolor='k', markeredgewidth=0.8,
                            markerfacecolor = 'gray',
                    color=current_color)
        
        if not label_added: label_added = True

    # --- 3. PROCESS & PLOT MODELS ---
    N_BOOT = 100
    model_results_17 = {}  # Store for quantitative comparison
    for sim in sim_dirs:
        sim_path = sim['path']
        sim_label = sim['label']
        sim_redshifts = sim['redshifts']
        sim_first = sim['first_snap']
        sim_last = sim['last_snap']
        sim_volume = sim['volume']
        sim_mass_convert = sim['mass_convert']
        do_bootstrap = (sim_label == 'SAGE26 (Millennium)')
        n_snaps = sim_last - sim_first + 1
        smd = np.zeros(n_snaps)
        smd_lo = np.zeros(n_snaps)
        smd_hi = np.zeros(n_snaps)

        for snap in range(sim_first, sim_last + 1):
            snap_name = f'Snap_{snap}'
            idx = snap - sim_first

            try:
                model_files = find_model_files(sim_path)
                d = read_snap_from_files(model_files, snap_name,
                                         ['StellarMass'],
                                         mass_convert=sim_mass_convert)
                if d:
                    m_stars = d['StellarMass']

                    # Apply limits 1e8 < M < 1e13 (from uploaded script)
                    w = np.where((m_stars > 1.0e8) & (m_stars < 1.0e13))[0]
                    if len(w) > 0:
                        m_sel = m_stars[w]
                        smd[idx] = np.sum(m_sel) / sim_volume

                        if do_bootstrap:
                            n_gal = len(m_sel)
                            boot = np.array([
                                np.sum(m_sel[np.random.randint(0, n_gal, n_gal)])
                                for _ in range(N_BOOT)
                            ]) / sim_volume
                            smd_lo[idx] = np.percentile(boot, 16)
                            smd_hi[idx] = np.percentile(boot, 84)
            except Exception:
                continue

        # Plot
        nonzero = np.where(smd > 0.0)[0]
        if len(nonzero) > 0:
            z_vals = sim_redshifts[sim_first:sim_last+1]
            ax.plot(z_vals[nonzero], np.log10(smd[nonzero]),
                    lw=sim['lw'], color=sim['color'], linestyle=sim['ls'], label=sim_label)

            if do_bootstrap:
                valid = nonzero[smd_lo[nonzero] > 0]
                if len(valid) > 0:
                    ax.fill_between(z_vals[valid],
                                    np.log10(smd_lo[valid]),
                                    np.log10(smd_hi[valid]),
                                    color=sim['color'], alpha=0.2)

            model_results_17[sim_label] = {
                'z': z_vals[nonzero],
                'smd': np.log10(smd[nonzero]),
            }

    # ===== QUANTITATIVE COMPARISON: SAGE26 vs SAGE16 (SMD) =====
    if 'SAGE26 (Millennium)' in model_results_17 and 'SAGE16' in model_results_17:
        from scipy.interpolate import interp1d as _interp1d
        print("\n" + "="*60)
        print("QUANTITATIVE COMPARISON: SAGE26 vs SAGE16 (SMD)")
        print("="*60)
        _z26 = model_results_17['SAGE26 (Millennium)']['z']
        _s26 = model_results_17['SAGE26 (Millennium)']['smd']
        _zc16 = model_results_17['SAGE16']['z']
        _sc16 = model_results_17['SAGE16']['smd']
        _z_mask = _z26 <= 7.5
        _z26 = _z26[_z_mask];  _s26 = _s26[_z_mask]
        _c16_interp = _interp1d(_zc16, _sc16, bounds_error=False, fill_value=np.nan)
        _sc16_m = _c16_interp(_z26)
        _ok = ~np.isnan(_sc16_m)
        if np.sum(_ok) > 0:
            _zv = _z26[_ok];  _diff = _sc16_m[_ok] - _s26[_ok]
            print(f"\n  Comparison over z = {_zv.min():.1f} to {_zv.max():.1f}")
            print(f"  Mean difference (SAGE16 - SAGE26):  {np.mean(_diff):+.3f} dex")
            print(f"  Median difference:               {np.median(_diff):+.3f} dex")
            print(f"  Std of difference:               {np.std(_diff):.3f} dex")
            print(f"  Max at z={_zv[np.argmax(_diff)]:.1f}: {np.max(_diff):+.3f} dex ({10**np.max(_diff):.1f}x)")
            print(f"  Min at z={_zv[np.argmin(_diff)]:.1f}: {np.min(_diff):+.3f} dex ({10**np.min(_diff):.1f}x)")
            print(f"\n  SMD at specific redshifts:")
            for _tz in [0, 1, 2, 3, 4, 5, 6]:
                _ti = np.argmin(np.abs(_zv - _tz))
                if np.abs(_zv[_ti] - _tz) < 0.5:
                    print(f"    z~{_zv[_ti]:.1f}: SAGE16={_sc16_m[_ok][_ti]:.2f}, SAGE26={_s26[_ok][_ti]:.2f}, "
                          f"Δ={_diff[_ti]:+.2f} dex ({10**_diff[_ti]:.1f}x)")
        print("="*60 + "\n")

    # --- COSMOS-Web ---
    if HAS_ASTROPY:
        smd_file = './data/sfrd/SMD.ecsv'
        if os.path.exists(smd_file):
            try:
                smd_table = Table.read(smd_file, format='ascii.ecsv')
                z_smd = np.array(smd_table['z'])
                rho_50 = np.log10(np.array(smd_table['rho_50']))
                rho_16 = np.log10(np.array(smd_table['rho_16']))
                rho_84 = np.log10(np.array(smd_table['rho_84']))
                ax.plot(z_smd, rho_50, color='darkorange', lw=2,
                        label='COSMOS-Web')
                ax.fill_between(z_smd, rho_16, rho_84,
                                color='orange', alpha=0.3)
            except Exception as e:
                print(f"Error loading SMD data: {e}")

    # --- 4. FORMATTING ---
    ax.set_ylabel(r'$\log_{10}\ \rho_{*}\ (M_{\odot}\ \mathrm{Mpc}^{-3})$')
    ax.set_xlabel(r'$\mathrm{Redshift}$')
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.set_xlim(0.0, 5.0)
    ax.set_ylim(6.0, 9.0)

    sim_names = {sim['label'] for sim in sim_dirs}
    handles, labels = ax.get_legend_handles_labels()
    sim_h = [h for h, l in zip(handles, labels) if l in sim_names]
    sim_l = [l for l in labels if l in sim_names]
    obs_h = [h for h, l in zip(handles, labels) if l not in sim_names]
    obs_l = [l for l in labels if l not in sim_names]
    _standard_legend(ax, loc='lower left', handles=obs_h, labels=obs_l)
    _model_legend_below(fig, sim_h, sim_l, order=_HISTORY_LEGEND_ORDER)

    outputFile = os.path.join(OUTPUT_DIR, 'Stellar_Mass_Density_History_Comparison' + OUTPUT_FORMAT)
    save_figure(fig, outputFile)


def _z0_models():
    """
    The z=0 model set shared by the stacked gas mass function and metallicity
    plots, in the style of plots 1 and 2. Each entry carries its own snapshot,
    volume and mass conversion. ``band`` marks the models drawn with a 1-sigma
    band; ``no_h2`` marks SAGE16, which does not split the cold gas. Runs whose
    output is missing are dropped.
    """
    mill = {'snapshot': SNAPSHOT, 'volume': VOLUME, 'mass_convert': MASS_CONVERT}
    mu = {'snapshot': f'Snap_{MINIUCHUU_LAST_SNAP}', 'volume': MINIUCHUU_VOLUME,
          'mass_convert': MINIUCHUU_MASS_CONVERT}
    models = [
        {'path': MINIUCHUU_DIR, 'label': 'SAGE26 (miniUchuu)',
         'color': MODEL_COLORS['sage26'], 'ls': '-', 'lw': 1.5, 'band': True, **mu},
        {'path': PRIMARY_DIR, 'label': 'SAGE26 (Millennium)',
         'color': MODEL_COLORS['sage26'], 'ls': '--', 'lw': 1.5, 'band': True, **mill},
        {'path': VANILLA_DIR, 'label': 'SAGE16',
         'color': MODEL_COLORS['sage16'], 'ls': '-.', 'lw': 1.5, 'band': True,
         'no_h2': True, **mill},
        {'path': BETA_CLOCK_MINIUCHUU_075_DIR, 'label': 'SAGE26 (miniUchuu beta=0.75)',
         'color': MODEL_COLORS['beta'], 'ls': '-', 'lw': 1.5, **mu},
        {'path': H2_DEPLETE_MINIUCHUU_DIR, 'label': 'SAGE26 (miniUchuu H2)',
         'color': MODEL_COLORS['h2'], 'ls': '-', 'lw': 1.5, **mu},
        {'path': BETA_CLOCK_075_H2_DEPLETE_MINIUCHUU_DIR, 'label': 'SAGE26 (miniUchuu H2 + beta=0.75)',
         'color': MODEL_COLORS['h2_beta'], 'ls': '-', 'lw': 1.5, **mu},
        {'path': BETA_CLOCK_MILLENNIUM_075_DIR, 'label': 'SAGE26 (Millennium beta=0.75)',
         'color': MODEL_COLORS['beta'], 'ls': '--', 'lw': 0.8, **mill},
        {'path': H2_DEPLETE_MILLENNIUM_DIR, 'label': 'SAGE26 (Millennium H2)',
         'color': MODEL_COLORS['h2'], 'ls': '--', 'lw': 0.8, **mill},
        {'path': BETA_CLOCK_075_H2_DEPLETE_MILLENNIUM_DIR, 'label': 'SAGE26 (Millennium H2 + beta=0.75)',
         'color': MODEL_COLORS['h2_beta'], 'ls': '--', 'lw': 0.8, **mill},
    ]
    return [m for m in models if model_files_exist(m['path'])]


# ========================== PLOT 3: COLD / HI / H2 MASS FUNCTIONS (STACKED) ==========================

def plot_3_gas_mass_functions_stacked():
    """
    Cold gas, HI and H2 mass functions at z=0 stacked in one column.

    Adapted from plot_39 in paper_plots.py, with the gas-partition recipes
    replaced by the same model set as plots 1 and 2. The fiducial models and
    SAGE16 have bootstrap 1-sigma shading. SAGE16 does not split the cold gas
    and so has no H2 line.
    """
    print('Plot 3: Cold / HI / H2 mass functions (stacked)')

    binwidth = 0.2
    N_BOOT = 100
    MASS_CUT = 1e8

    models = _z0_models()

    panels = [
        {'field': 'ColdGas', 'tag': r'$M_{\mathrm{cold}}$',
         'obs': load_himf_observations() + load_h2mf_observations()},
        {'field': 'H1gas', 'tag': r'$M_{\mathrm{HI}}$',
         'obs': load_himf_observations()},
        {'field': 'H2gas', 'tag': r'$M_{\mathrm{H_2}}$',
         'obs': load_h2mf_observations()},
    ]

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 17), sharex=True)

    def _mf(model, field):
        d = read_snap_from_files(find_model_files(model['path']), model['snapshot'],
                                 [field], mass_convert=model['mass_convert'])
        if not d or field not in d:
            return None
        g = d[field]
        valid = g > MASS_CUT
        if not np.any(valid):
            return None
        return mass_function_bootstrap(np.log10(g[valid]), model['volume'],
                                       binwidth=binwidth, n_boot=N_BOOT)

    model_handles = {}
    for ax, cfg in zip(axes, panels):
        for model in models:
            if cfg['field'] == 'H2gas' and model.get('no_h2'):
                continue
            res = _mf(model, cfg['field'])
            if res is None:
                continue
            centers, phi, plo, phi_hi, _ = res
            good = np.isfinite(phi) & np.isfinite(plo) & np.isfinite(phi_hi)
            line, = ax.plot(centers[good], phi[good], color=model['color'],
                            ls=model['ls'], lw=model['lw'], zorder=Z_MODEL_LINE)
            model_handles.setdefault(model['label'], line)
            if model.get('band'):
                ax.fill_between(centers[good], plo[good], phi_hi[good],
                                color=model['color'], alpha=0.2, lw=0.0,
                                zorder=Z_MODEL_BAND)

        for obs in cfg['obs']:
            mass = obs['mass']; phi_obs = obs['phi']
            m = mass >= 8.0
            if not np.any(m):
                continue
            ax.errorbar(mass[m], phi_obs[m],
                        yerr=_gasmf_obs_yerr(obs, m),
                        fmt=obs['marker'], color=obs['color'],
                        markerfacecolor='gray',
                        markeredgecolor=obs.get('edgecolor', 'k'),
                        markeredgewidth=1.0, ms=6, lw=1.0, capsize=2,
                        alpha=0.8, label=obs['label'], zorder=Z_OBS)

        ax.text(0.04, 0.06, cfg['tag'], transform=ax.transAxes,
                ha='left', va='bottom', fontsize=20)
        ax.set_ylim(-5.5, -0.5)
        ax.set_ylabel(r'$\log_{10}\ \phi\ [\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1}]$')
        ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.2))
        if cfg['obs']:
            _standard_legend(ax, loc='upper right', fontsize=12)

    axes[-1].set_xlim(8.0, 11.0)
    axes[-1].set_xlabel(r'$\log_{10}\ M_{\mathrm{gas}}\ [M_{\odot}]$')
    axes[-1].xaxis.set_major_locator(plt.MultipleLocator(1.0))
    axes[-1].xaxis.set_minor_locator(plt.MultipleLocator(0.2))

    _model_legend_below(fig, list(model_handles.values()), list(model_handles.keys()),
                        order=_HISTORY_LEGEND_ORDER)
    fig.subplots_adjust(hspace=0.0)
    save_figure(fig, os.path.join(OUTPUT_DIR,
                'Gas_Mass_Functions_Stacked' + OUTPUT_FORMAT))


# ========================== PLOT 4: GAS METALLICITY VS STELLAR MASS ==========================

def plot_4_gas_metallicity_vs_stellar_mass():
    """
    Gas-phase metallicity vs. stellar mass at z=0.

    Adapted from plot_3 in paper_plots.py, with the same model set as plots
    1-3. Lines are binned medians, with no scatter band. Every galaxy with m* > 1e8 Msun and cold gas
    is kept, as in paper_plots.py, which is a broader sample than the
    emission-line selected observations.
    """
    print('Plot 4: Gas metallicity vs stellar mass')

    fig = plt.figure(figsize=HISTORY_FIGSIZE)
    ax = fig.add_subplot(111)

    mass_bins = np.arange(8.0, 13.0 + 0.1, 0.1)
    model_h, model_l = [], []
    for model in _z0_models():
        d = read_snap_from_files(find_model_files(model['path']), model['snapshot'],
                                 ['StellarMass', 'ColdGas', 'MetalsColdGas'],
                                 mass_convert=model['mass_convert'])
        if not d:
            continue
        w = ((d['StellarMass'] > 1e8) & (d['ColdGas'] > 0)
             & (d['MetalsColdGas'] > 0))
        if not np.any(w):
            continue
        line = plot_binned_median_1sigma(
            ax, np.log10(d['StellarMass'][w]),
            metallicity_12logOH(d['MetalsColdGas'][w], d['ColdGas'][w]),
            mass_bins, color=model['color'], label=None,
            alpha=0.0,
            lw=model['lw'], ls=model['ls'], min_count=20,
            zorder_fill=Z_MODEL_BAND, zorder_line=Z_MODEL_LINE,
        )
        if line is not None:
            model_h.append(line)
            model_l.append(model['label'])

    for obs in load_mzr_observations():
        if obs['yerr'] is not None:
            ax.errorbar(obs['mass'], obs['Z'], yerr=obs['yerr'],
                        fmt=obs['fmt'], color=obs['color'],
                        markeredgecolor='k', markeredgewidth=1.0, linewidth=1.0,
                        markerfacecolor='gray', ms=8,
                        label=obs['label'], alpha=0.6, zorder=Z_OBS)
        else:
            ax.plot(obs['mass'], obs['Z'], obs['fmt'],
                    markeredgecolor='k', markeredgewidth=1.0, linewidth=1.0,
                    markerfacecolor='gray', ms=8,
                    color=obs['color'], label=obs['label'], alpha=0.6,
                    zorder=Z_OBS)

    ax.set_xlim(8.0, 12.0)
    ax.set_ylim(7.5, 9.5)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.set_xlabel(r'$\log_{10}\ m_{\mathrm{*}}\ [M_{\odot}]$')
    ax.set_ylabel(r'$12\ +\ \log_{10}\ (\mathrm{O/H})$')

    _standard_legend(ax, loc='upper left')
    _model_legend_below(fig, model_h, model_l, order=_HISTORY_LEGEND_ORDER)
    save_figure(fig, os.path.join(OUTPUT_DIR,
                'MetallicityStellarMass' + OUTPUT_FORMAT))


# ========================== PLOT 5: BULGE-TO-TOTAL RATIO VS STELLAR MASS ==========================

def plot_5_bulge_to_total():
    """
    Bulge-to-total mass ratio vs. stellar mass at z=0.

    Same model set and style as plots 1-4, no shading. The model line is the
    mass-weighted bulge fraction in each stellar mass bin,
    sum(m_bulge) / sum(m*), over every galaxy with m* > 1e8 Msun (discs with
    no bulge included). That is the quantity Moffett+16 tabulate ("fraction of
    mass in bulge" at fixed stellar mass), so model and data are like for like.
    """
    print('Plot 5: Bulge-to-total ratio vs stellar mass')

    fig = plt.figure(figsize=HISTORY_FIGSIZE)
    ax = fig.add_subplot(111)

    binwidth = 0.1
    min_count = 20
    edges = np.arange(8.0, 12.5 + binwidth, binwidth)
    centers = 0.5 * (edges[1:] + edges[:-1])

    model_h, model_l = [], []
    for model in _z0_models():
        d = read_snap_from_files(find_model_files(model['path']), model['snapshot'],
                                 ['StellarMass', 'BulgeMass'],
                                 mass_convert=model['mass_convert'])
        if not d:
            continue
        w = d['StellarMass'] > 1e8
        lm = np.log10(d['StellarMass'][w])
        n, _ = np.histogram(lm, bins=edges)
        m_tot, _ = np.histogram(lm, bins=edges, weights=d['StellarMass'][w])
        m_bul, _ = np.histogram(lm, bins=edges, weights=d['BulgeMass'][w])
        good = n >= min_count
        if not np.any(good):
            continue
        line, = ax.plot(centers[good], m_bul[good] / m_tot[good],
                        color=model['color'], ls=model['ls'], lw=model['lw'],
                        zorder=Z_MODEL_LINE)
        model_h.append(line)
        model_l.append(model['label'])

    moffett_file = os.path.join(OBS_DIR, 'morphology', 'Moffet16.dat')
    if os.path.exists(moffett_file):
        mof = np.loadtxt(moffett_file, comments='#')
        ax.errorbar(mof[:, 0] + imf_shift('Moffett+16'), mof[:, 1],
                    yerr=[mof[:, 1] - mof[:, 2], mof[:, 3] - mof[:, 1]],
                    fmt='o', color='k', markerfacecolor='gray',
                    markeredgecolor='k', markeredgewidth=1.0, ms=8, lw=1.0,
                    alpha=0.6, label='Moffett+16', zorder=Z_OBS)

    ax.set_xlim(8.0, 12.0)
    ax.set_ylim(0.0, 1.05)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax.set_xlabel(r'$\log_{10}\ m_{\mathrm{*}}\ [M_{\odot}]$')
    ax.set_ylabel(r'$m_{\mathrm{bulge}}\ /\ m_{\mathrm{*}}$')

    _standard_legend(ax, loc='upper left')
    _model_legend_below(fig, model_h, model_l, order=_HISTORY_LEGEND_ORDER)
    save_figure(fig, os.path.join(OUTPUT_DIR,
                'BulgeToTotal' + OUTPUT_FORMAT))


# ========================== PLOT 6: QUIESCENT FRACTION VS STELLAR MASS ==========================

def _red_fraction_obs():
    """
    Red fractions f = phi_red / (phi_red + phi_blue) from the colour-split
    z=0 stellar mass functions already used elsewhere: Bell+03 (on a shared
    mass grid, errors propagated from both functions) and Baldry+12 (digitised,
    no errors; the blue function is interpolated onto the red mass grid).
    Returns a list of dicts with 'label', 'mass', 'f', 'ferr', 'fmt'.
    """
    out = []

    red = load_bell_smf_q_data()
    blue = load_bell_smf_sf_data()
    if red[0] is not None and blue[0] is not None:
        # col 2/3 are the +/- errors on log phi; use their mean
        bm = {round(m, 3): (p, 0.5 * (eh + el)) for m, p, eh, el in zip(*blue)}
        mass, f, ferr = [], [], []
        for m, p, eh, el in zip(*red):
            key = round(m, 3)
            if key not in bm:
                continue
            pb, sb = bm[key]
            sr = 0.5 * (eh + el)
            fr = 10**p / (10**p + 10**pb)
            mass.append(m)
            f.append(fr)
            ferr.append(np.log(10) * fr * (1 - fr) * np.hypot(sr, sb))
        if mass:
            out.append({'label': 'Bell+03', 'mass': np.array(mass), 'f': np.array(f),
                        'ferr': np.array(ferr), 'fmt': 'o'})

    try:
        bal = load_baldry_blue_red()
        qm, qp = bal['q_mass'], bal['q_phi']
        ok = np.isfinite(qm) & np.isfinite(qp)
        qm, qp = qm[ok], qp[ok]
        sm, sp = bal['sf_mass'], bal['sf_phi']
        ok = np.isfinite(sm) & np.isfinite(sp)
        sm, sp = sm[ok], sp[ok]
        inside = (qm >= sm.min()) & (qm <= sm.max())
        pb = np.interp(qm[inside], sm, sp)
        fr = 10**qp[inside] / (10**qp[inside] + 10**pb)
        out.append({'label': 'Baldry+12', 'mass': qm[inside], 'f': fr,
                    'ferr': None, 'fmt': 's'})
    except Exception as e:
        print(f"  Warning: could not derive Baldry+12 red fraction: {e}")

    return out


def plot_6_quiescent_fraction():
    """
    Quiescent fraction vs. stellar mass at z=0, all galaxies.

    Same model set and style as plots 1-5, no shading. A galaxy is quiescent
    when log10(sSFR/yr^-1) < SSFR_CUT. The observations are red fractions
    from colour-split mass functions, so the selection is by colour, not sSFR.
    """
    print('Plot 6: Quiescent fraction vs stellar mass')

    fig = plt.figure(figsize=HISTORY_FIGSIZE)
    ax = fig.add_subplot(111)

    binwidth = 0.1
    min_count = 20
    edges = np.arange(8.0, 12.5 + binwidth, binwidth)
    centers = 0.5 * (edges[1:] + edges[:-1])

    model_h, model_l = [], []
    for model in _z0_models():
        d = read_snap_from_files(find_model_files(model['path']), model['snapshot'],
                                 ['StellarMass', 'SfrDisk', 'SfrBulge'],
                                 mass_convert=model['mass_convert'])
        if not d:
            continue
        w = d['StellarMass'] > 1e8
        ms = d['StellarMass'][w]
        sfr = (d['SfrDisk'] + d['SfrBulge'])[w]
        lm = np.log10(ms)
        quiescent = sfr < 10**SSFR_CUT * ms
        n, _ = np.histogram(lm, bins=edges)
        nq, _ = np.histogram(lm[quiescent], bins=edges)
        good = n >= min_count
        if not np.any(good):
            continue
        line, = ax.plot(centers[good], nq[good] / n[good],
                        color=model['color'], ls=model['ls'], lw=model['lw'],
                        zorder=Z_MODEL_LINE)
        model_h.append(line)
        model_l.append(model['label'])

    for obs in _red_fraction_obs():
        ax.errorbar(obs['mass'], obs['f'], yerr=obs['ferr'], fmt=obs['fmt'],
                    color='k', markerfacecolor='gray', markeredgecolor='k',
                    markeredgewidth=1.0, ms=8, lw=1.0, alpha=0.6, ls='none',
                    label=obs['label'], zorder=Z_OBS)

    ax.set_xlim(8.0, 12.0)
    ax.set_ylim(0.0, 1.05)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax.set_xlabel(r'$\log_{10}\ m_{\mathrm{*}}\ [M_{\odot}]$')
    ax.set_ylabel(r'$f_{\mathrm{Q}}$')

    _standard_legend(ax, loc='upper left')
    _model_legend_below(fig, model_h, model_l, order=_HISTORY_LEGEND_ORDER)
    save_figure(fig, os.path.join(OUTPUT_DIR,
                'QuiescentFraction' + OUTPUT_FORMAT))


# ========================== MAIN ==========================

# Registry of plot functions
# z=0 plots take (primary, vanilla); evolution plots take (snapdata)
Z0_PLOTS = {
    
}

EVOLUTION_PLOTS = {
   
}

STANDALONE_PLOTS = {
  1: plot_1_sfrd_history,
  2: plot_2_smd_history,
  3: plot_3_gas_mass_functions_stacked,
  4: plot_4_gas_metallicity_vs_stellar_mass,
  5: plot_5_bulge_to_total,
  6: plot_6_quiescent_fraction,
}

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
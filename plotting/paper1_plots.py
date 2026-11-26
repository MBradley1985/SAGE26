#!/usr/bin/env python
"""
paper1_plots.py -- SAGE26 Paper I: every figure made from the model runs, plus the
referee-response figures, and the numbers the text quotes.

Figures (file names as written by the functions named):
  2   SFRD history                     paper_plots.plot_16_sfrd_history
  3   Mass-metallicity relation        paper_plots.plot_3_gas_metallicity_vs_stellar_mass
  4   Cold / HI / H2 MFs + SMF         paper1_referee.task_gas            (gas_fig4)
  5   SFR vs stellar mass              paper_plots.plot_15_sfr_vs_stellar_mass
  6   Mass loading and ejection        paper_plots.plot_24_mass_loading_vs_velocity
  7   Stellar-halo mass relation       paper_plots.plot_5_stellar_halo_mass
  8   SMF, star-forming / quiescent    paper_plots.plot_1_stellar_mass_function_ssfr_combined
  9   Black hole-bulge mass            paper_plots.plot_4_bh_bulge_mass
  10  Bulge size-mass by channel       paper_plots.plot_6b_bulge_mass_size_median
  11  FFB fraction vs halo mass        paper_plots.plot_13_ffb_vs_redshift
  12  SMF grid, z = 0-12               paper_plots.plot_18_smf_redshift_grid
  (13, the ablation series, needs the mini-Millennium variant runs: plotting/ablation_series.py)
  R   Referee letter: cosmology (Fig 1), reionisation, SMF and SFRD offsets
                                       paper1_referee.task_cosmo / reion / smf / sfrd
(Figure 1 is a schematic.)

The plotting code is not copied: the functions above draw the figures, so the
figures and their names are exactly those of paper_plots.py and paper1_referee.py.
What changes is how they get their data, which is what makes the full boxes slow:
  * One column cache.  Every snapshot column is read once, from all MPI files in
    parallel, Len >= 20 applied, and shared by every figure.  The z = 0 catalogues
    and the multi-snapshot data are loaded lazily, column by column, so only
    what a figure touches is read.  After each figure every snapshot but z = 0 is
    freed, so memory stays at about one figure's worth of columns.
  * Mass-function bootstraps are drawn from the binned counts (multinomial),
    which has the same distribution as resampling the galaxies, instead of
    re-histogramming 100 resampled copies of up to ~10^7 galaxies.
  * Bootstrapped means of more than 10^5 values (Fig 11) use their normal limit,
    as paper_plots already does for the summed densities (BOOT_EXACT_MAX).
Everything printed is also written to <out>/paper1_numbers.txt.

Usage (from anywhere; paths are relative to the SAGE26 root):
    python plotting/paper1_plots.py                    # mini-Millennium runs in output/
    python plotting/paper1_plots.py --box big          # full boxes (Millennium_*, miniUchuu_FFB) in output/
    python plotting/paper1_plots.py --figs 2 12 R      # some figures only
"""

import os
import sys
import time
import argparse
from collections.abc import Mapping

os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')
# One thread per process, so 4 workers use 4 CPUs (numpy/BLAS would otherwise start one per core).
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np
import h5py as h5

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALL_FIGS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', 'R']

# The bundled code below picks its run directories when it is loaded, so --box is read now.
if '--box' in sys.argv[1:-1] and sys.argv[sys.argv.index('--box') + 1] == 'big':
    os.environ['SAGE26_RUNS'] = 'big'
_CWD = os.getcwd()
os.chdir(ROOT)        # the bundled code reads ./data and ./output relative to the SAGE26 root
import matplotlib
matplotlib.use('Agg')

# The paper_plots.py and paper1_referee.py code these figures use, copied in so the
# script runs on its own.
# ==================== BEGIN SHARED CODE (from the former paper_plots.py) ====================
# ---- matplotlib style (plotting/kieren_cohare_palatino_sty.mplstyle) ----
_BUNDLED_STYLE = "# Set default figure size\nfigure.figsize : 8.34, 6.25\nfigure.dpi : 140\n\n# Set x axis\nxtick.major.size : 7.5\nxtick.major.width : 1.5\nxtick.minor.size : 5.5\nxtick.minor.width : 0.5\nxtick.direction : in\nxtick.top : True\n\n# Set y axis\nytick.major.size : 7.5\nytick.major.width : 1.5\nytick.minor.size : 5.5\nytick.minor.width : 0.5\nytick.direction : in\nytick.right : True\n\n# Fontsizes\nxtick.labelsize : 16\nytick.labelsize : 16\nxtick.major.pad : 9\nlegend.fontsize : 14\nlegend.title_fontsize : 16\naxes.titlesize : 12\naxes.labelsize : 20\n\n# Line widths\naxes.linewidth : 1.5\ngrid.linewidth : 1\nlines.linewidth : 2\nlines.solid_capstyle : round\n\n# Font style\nfont.family : serif\ntext.usetex : True\nfont.size:    20.0\nfont.serif: Palatino\n\n# margins\n#axes.xmargin : 0\n#axes.ymargin : 0\n\n# Legend\nlegend.frameon : False\n\n# Set color cycle:\n# blue, orange, green, red, purple, black, grey\naxes.prop_cycle : cycler('color', ['0C5DA5', '00B945', 'FF9500', 'FF2C00', '845B97', '474747', '9e9e9e'])\n\n# tight layout\nfigure.autolayout : True  \n"

# ---- from plotting/paper_plots.py ----
import h5py as h5

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm

from matplotlib.collections import LineCollection

from matplotlib.lines import Line2D

from matplotlib.ticker import FuncFormatter, ScalarFormatter

from matplotlib import font_manager

import logging

import subprocess

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

# File paths
PRIMARY_DIR = './output/millennium/'

VANILLA_DIR = './output/millennium_vanilla/'

NOFFB_DIR = './output/millennium_noffb/'

GD14_DIR = './output/millennium_gd14/'

KD12_DIR = './output/millennium_kd12/'

KMT09_DIR = './output/millennium_kmt09/'

K13_DIR = './output/millennium_k13/'

# CGM_DYN_DIR = './output/millennium_cgmdyn/'
# DISK_SMOOTH_DIR = './output/millennium_disk2/'
MINIUCHUU_DIR = './output/microuchuu/'

# SAGE26_RUNS=big selects the full-box runs (the directory names on the supercomputer).
if os.environ.get('SAGE26_RUNS', '').lower() == 'big':
    PRIMARY_DIR = './output/Millennium_FFB/'
    VANILLA_DIR = './output/Millennium_vanilla/'
    NOFFB_DIR = './output/Millennium_noFFB/'
    NOCGM_DIR = './output/Millennium_noCGM/'
    C16_FEEDBACK_DIR = './output/Millennium_c16feedback/'
    GD14_DIR = './output/Millennium_gd14/'
    KD12_DIR = './output/Millennium_kd12/'
    KMT09_DIR = './output/Millennium_kmt09/'
    K13_DIR = './output/Millennium_k13/'
    FFB_BK25_DIR = './output/Millennium_FFB_MBK25/'
    FFB_BK25_SMOOTH_DIR = './output/Millennium_FFB_MBK25/'
    FFB_BK25_FFB100_DIR = './output/Millennium_MBK_FFB100/'
    FFB_NOSIGMOID_DIR = './output/Millennium_FFB_nosigmoid/'
    MINIUCHUU_DIR = './output/miniUchuu_FFB/'
    FFB100_DIR = './output/Millennium_FFB100/'

MODEL_FILE = 'model_0.hdf5'

OBS_DIR = './data/'

# Plotting (analysis choices — not simulation parameters)
OUTPUT_FORMAT = '.pdf'

# Mass range over which Shen+2003 fitted their early-type size-mass relation
# (SDSS, log10 m/Msun). The relation is only drawn here; outside it the line would
# be extrapolation, not data.
SHEN03_MASS_RANGE = (10.0, 11.5)

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
    'Driver+18':          ('chabrier', 'file'),
    "D'Silva+23":         ('chabrier', 'lit'),
    'Novak+17':           ('chabrier', 'file'),
    'Bouwens+15':         ('salpeter', 'file'),
    'Finkelstein+15':     ('salpeter', 'file'),
    'Weaver+23':          ('chabrier', 'lit'),
    'Song+16':            ('chabrier', 'lit'),
    'Bellstedt+20':       ('chabrier', 'lit'),
    'Bell+03':            ('diet-salpeter', 'lit'),
    'Curti+20':           ('kroupa',   'lit'),
    'Moster+13':          ('chabrier', 'lit'),
    # --- not established: no shift applied, reported at run time ---
    'Stefanon+21':        (None, None),
    'Navarro-Carrera+24': (None, None),
    'Weibel+24':          (None, None),
    'Kikuchihara+20':     (None, None),
    'Kikuchihara+20 SMD': (None, None),
    'Papovich+23':        (None, None),
    'Oesch+18':           (None, None),
    'McLeod+23':          (None, None),
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
                'particle_mass':  float(sim.attrs.get('particle_mass', np.nan)),  # code units
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

_primary_hdr = _read_sim_header(PRIMARY_DIR)

if _primary_hdr is not None:
    HUBBLE_H         = _primary_hdr['hubble_h']
    BOX_SIZE         = _primary_hdr['box_size']
    VOLUME_FRACTION  = _primary_hdr['volume_fraction']
    VOLUME           = (BOX_SIZE / HUBBLE_H)**3 * VOLUME_FRACTION  # Mpc^3
    MASS_CONVERT     = _primary_hdr['unit_mass_in_g'] / _MSUN_CGS / HUBBLE_H
    PARTICLE_MASS_MSUN = _primary_hdr['particle_mass'] * MASS_CONVERT
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
    PARTICLE_MASS_MSUN = 0.086 * MASS_CONVERT   # Millennium, 8.6e8 Msun/h
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

def setup_style():
    """Configure matplotlib for publication-quality white-background plots."""
    _use_bundled_style()
    # Fall back to mathtext where the TeX install can't render the style
    # (e.g. no Palatino fonts on the supercomputer).
    fig = plt.figure()
    fig.text(0.5, 0.5, r'$\log_{10} M_\odot$ test')
    try:
        fig.canvas.draw()
    except Exception as exc:
        print(f'TeX rendering failed ({type(exc).__name__}); using mathtext instead.')
        plt.rcParams['text.usetex'] = False
        # Without Palatino matplotlib drops to the much wider DejaVu Sans, so
        # every label grows. Use a Palatino clone if installed, else STIX
        # (shipped with matplotlib), with maths in the same font. TeX installs
        # often carry the TeX Gyre Pagella clone where matplotlib can't see it.
        for name in ('regular', 'italic', 'bold', 'bolditalic'):
            try:
                path = subprocess.run(['kpsewhich', f'texgyrepagella-{name}.otf'],
                                      capture_output=True, text=True, timeout=10).stdout.strip()
                if path:
                    font_manager.fontManager.addfont(path)
            except (OSError, subprocess.SubprocessError):
                break
        plt.rcParams['font.serif'] = ['Palatino', 'TeX Gyre Pagella', 'P052', 'URW Palladio L',
                                      'Palatino Linotype', 'STIXGeneral', 'DejaVu Serif']
        plt.rcParams['mathtext.fontset'] = 'custom'
        plt.rcParams['mathtext.rm'] = 'serif'
        plt.rcParams['mathtext.it'] = 'serif:italic'
        plt.rcParams['mathtext.bf'] = 'serif:bold'
        plt.rcParams['mathtext.fallback'] = 'stix'
        logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
        print(f"  font: {font_manager.findfont('serif')}")
    finally:
        plt.close(fig)

def _tex_safe(s):
    """Make label strings safe for both usetex and non-usetex modes."""
    if not plt.rcParams.get('text.usetex', False):
        s = s.replace(r"\'{e}", "\u00e9")   # é
        s = s.replace(r'\&', '&')
        s = s.replace(r'\%', '%')
    return s

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

def load_driver_sfrd_2018_data():
    """Load Driver et al. 2018 SFRD (GAMA/G10-COSMOS/3D-HST, Chabrier, h = 0.7).

    Returns bin-centre z, log SFRD, the quadrature sum of the four quoted error
    terms, and the lower/upper half-widths of each redshift bin.
    """
    filename = './data/sfrd/Driver18_sfr.dat'
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None, None, None, None, None
    d = np.loadtxt(filename, comments='#')
    z = 0.5 * (d[:, 0] + d[:, 1])
    err = np.sqrt(d[:, 3]**2 + d[:, 4]**2 + d[:, 5]**2 + d[:, 6]**2)
    return z, d[:, 2] + imf_shift('Driver+18'), err, z - d[:, 0], d[:, 1] - z

def load_dsilva_sfrd_2023_data():
    """Load D'Silva et al. 2023 SFRD (GAMA/DEVILS, Chabrier, h = 0.7).

    Returns z, log SFRD, lower/upper SFRD errors and lower/upper z errors.
    """
    filename = './data/sfrd/DSilva23_sfr.dat'
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None, None, None, None, None, None
    d = np.loadtxt(filename, comments='#')
    return (d[:, 3], d[:, 0] + imf_shift("D'Silva+23"), d[:, 2], d[:, 1],
            d[:, 5], d[:, 4])

# Tabulated SFRD files with columns z, log_rho_sfr, e_log_rho_sfr_upper/lower.
# Each header records the paper, table and IMF; the UV ones are dust corrected.
SFRD_TABLES = {
    'Novak+17':       './data/sfrd/novak_sfrd_2017.ecsv',
    'Bouwens+15':     './data/sfrd/bouwens_sfrd_2015.ecsv',
    'Finkelstein+15': './data/sfrd/finkelstein_sfrd_2015.ecsv',
}

def load_sfrd_table(dataset):
    """Load one SFRD_TABLES entry: z, log SFRD (Chabrier), upper and lower errors."""
    filename = SFRD_TABLES[dataset]
    if not HAS_ASTROPY or not os.path.exists(filename):
        print(f"Warning: {filename} not available.")
        return None, None, None, None
    t = Table.read(filename, format='ascii.ecsv')
    return (np.array(t['z']), np.array(t['log_rho_sfr']) + imf_shift(dataset),
            np.array(t['e_log_rho_sfr_upper']), np.array(t['e_log_rho_sfr_lower']))

def load_mcleod_rho_sfr_2024_data():
    """Load McLeod et al. 2023 SFR density data."""
    if not HAS_ASTROPY:
        return None, None, None, None
    filename = './data/sfrd/mcleod_rhouv_2024.ecsv'
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None, None, None, None

    try:
        table = Table.read(filename, format='ascii.ecsv')
        z = table['z']
        re = table['log_rho_sfr'] + imf_shift('McLeod+23')
        re_err_plus = np.zeros_like(re)
        re_err_minus = np.zeros_like(re)
        return z, re, re_err_plus, re_err_minus
    except Exception as e:
        print(f"Error loading McLeod 2023 SFRD data: {e}")
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

def plot_1_stellar_mass_function_ssfr_combined(primary, vanilla):
    """
    Star-forming and quiescent stellar mass functions side by side.

    Combines StellarMassFunction_SF and StellarMassFunction_Q into a single
    1x2 figure. Each panel keeps the same style and formatting as the
    standalone plots (colours, markers, bootstrap shading, observations).
    """
    print('Plot 1 (combined): Stellar mass function (SF | Q) with Bootstrap Errors')

    binwidth = 0.1
    N_BOOT = 100  # Number of bootstrap samples

    # --- Primary model ---
    w = primary['StellarMass'] > 0
    mass = np.log10(primary['StellarMass'][w])
    ssfr = log_ssfr(primary['SfrDisk'][w], primary['SfrBulge'][w],
                     primary['StellarMass'][w])

    # Establish common bins from the total MF, then split populations
    x, _, mrange = mass_function(mass, VOLUME, binwidth)
    mass_q = mass[ssfr < SSFR_CUT]
    mass_sf = mass[ssfr > SSFR_CUT]
    _, phi_q, _ = mass_function(mass_q, VOLUME, binwidth, mass_range=mrange)
    _, phi_sf, _ = mass_function(mass_sf, VOLUME, binwidth, mass_range=mrange)

    # Bootstrap error calculation (identical to the standalone plots)
    def calc_bootstrap_errors(data_mass, m_range, vol, bw, n_boot=100):
        if len(data_mass) == 0:
            return np.nan, np.nan
        mi, ma = m_range
        nbins = int(round((ma - mi) / bw))
        edges = np.linspace(mi, ma, nbins + 1)
        boot_phis = []
        n_obj = len(data_mass)
        for _ in range(n_boot):
            sample = data_mass[np.random.randint(0, n_obj, n_obj)]
            counts, _ = np.histogram(sample, bins=edges)
            with np.errstate(divide='ignore'):
                phi = np.log10(counts / vol / bw)
            phi[~np.isfinite(phi)] = np.nan
            boot_phis.append(phi)
        boot_phis = np.array(boot_phis)
        lo = np.nanpercentile(boot_phis, 16, axis=0)
        hi = np.nanpercentile(boot_phis, 84, axis=0)
        return lo, hi

    print(f'  Bootstrapping SAGE26 data ({N_BOOT} iterations)...')
    phi_q_lo, phi_q_hi = calc_bootstrap_errors(mass_q, mrange, VOLUME, binwidth, N_BOOT)
    phi_sf_lo, phi_sf_hi = calc_bootstrap_errors(mass_sf, mrange, VOLUME, binwidth, N_BOOT)

    # --- Vanilla model ---
    w2 = vanilla['StellarMass'] > 0
    mass_v = np.log10(vanilla['StellarMass'][w2])
    ssfr_v = log_ssfr(vanilla['SfrDisk'][w2], vanilla['SfrBulge'][w2],
                       vanilla['StellarMass'][w2])
    x_v, _, mrange_v = mass_function(mass_v, VOLUME, binwidth)
    _, phi_q_v, _ = mass_function(mass_v[ssfr_v < SSFR_CUT], VOLUME, binwidth,
                                  mass_range=mrange_v)
    _, phi_sf_v, _ = mass_function(mass_v[ssfr_v > SSFR_CUT], VOLUME, binwidth,
                                   mass_range=mrange_v)

    # --- Observations ---
    gama = load_gama_smf_morph()
    baldry = load_baldry_blue_red()

    # --- Plot ---
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # SAGE26 star-forming (steelblue) and quiescent (firebrick), with bootstrap shading
    ax.plot(x, phi_sf, color='steelblue', lw=4, label='SAGE26 Star-forming', zorder=10)
    ax.fill_between(x, phi_sf_lo, phi_sf_hi, color='steelblue', alpha=0.3, edgecolor='none', zorder=10)
    ax.plot(x, phi_q, color='firebrick', lw=3, label='SAGE26 Quiescent', zorder=10)
    ax.fill_between(x, phi_q_lo, phi_q_hi, color='firebrick', alpha=0.3, edgecolor='none', zorder=10)

    # SAGE16 (vanilla)
    ax.plot(x_v, phi_sf_v, color='steelblue', lw=2, ls='--', label='SAGE16 Star-forming')
    ax.plot(x_v, phi_q_v, color='firebrick', lw=2, ls='--', label='SAGE16 Quiescent')

    # Observations -- grey markers keep their per-source shapes (Moffett=diamond,
    # Baldry=circle, Bell=square); outlined by population (blue=SF, red=Q) so the
    # two are distinguishable on a single panel.
    valid_D = ~np.isnan(gama['D'])
    valid_E = ~np.isnan(gama['E_HE'])
    # Star-forming observations (blue outline)
    ax.errorbar(gama['mass'][valid_D], gama['D'][valid_D], yerr=gama['D_err'][valid_D],
                fmt='d', color='gray', markeredgecolor='steelblue', markeredgewidth=1.0,
                linewidth=1.0, markerfacecolor='gray', ms=8, alpha=0.6, zorder=9)
    ax.scatter(baldry['sf_mass'], baldry['sf_phi'], edgecolor='steelblue', facecolor='gray',
               marker='o', s=50, alpha=0.6, zorder=9)
    bell_mass, bell_phi, bell_err_hi, bell_err_lo = load_bell_smf_sf_data()
    if bell_mass is not None:
        ax.errorbar(bell_mass, bell_phi, yerr=[bell_err_lo, bell_err_hi],
                    markeredgecolor='steelblue', markerfacecolor='gray',
                    fmt='s', color='gray', ms=8, lw=1.0, alpha=0.6, zorder=9)
    # Quiescent observations (red outline)
    ax.errorbar(gama['mass'][valid_E], gama['E_HE'][valid_E], yerr=gama['E_HE_err'][valid_E],
                fmt='d', color='gray', markeredgecolor='firebrick', markerfacecolor='gray',
                ms=8, lw=1, alpha=0.6, zorder=9)
    ax.scatter(baldry['q_mass'], baldry['q_phi'], edgecolor='firebrick', facecolor='gray',
               marker='o', s=50, alpha=0.6, zorder=9)
    bell_mass, bell_phi, bell_err_hi, bell_err_lo = load_bell_smf_q_data()
    if bell_mass is not None:
        ax.errorbar(bell_mass, bell_phi, yerr=[bell_err_lo, bell_err_hi],
                    markeredgecolor='firebrick', markerfacecolor='gray',
                    fmt='s', color='gray', ms=8, lw=1, alpha=0.6, zorder=9)

    # Combined observations legend: one grey marker per source (SF and Q share an entry)
    obs_h = [
        plt.Line2D([], [], marker='d', color='gray', markerfacecolor='gray',
                   markeredgecolor='k', ms=8, lw=0, alpha=0.6),
        plt.Line2D([], [], marker='o', color='gray', markerfacecolor='gray',
                   markeredgecolor='k', ms=8, lw=0, alpha=0.6),
        plt.Line2D([], [], marker='s', color='gray', markerfacecolor='gray',
                   markeredgecolor='k', ms=8, lw=0, alpha=0.6),
    ]
    obs_l = ['Moffett+16', 'Baldry+12', 'Bell+03']

    ax.set_xlim(8, 12)
    ax.set_ylim(-6, -1)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.set_ylabel(r'$\log_{10}\ \phi\ [\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1}]$')
    ax.set_xlabel(r'$\log_{10}\ m_{\mathrm{*}}\ [M_{\odot}]$')

    handles, labels = ax.get_legend_handles_labels()
    sim_h = [h for h, l in zip(handles, labels) if l.startswith(('SAGE26', 'SAGE16'))]
    sim_l = [l for l in labels if l.startswith(('SAGE26', 'SAGE16'))]
    leg1 = _standard_legend(ax, loc='lower left', handles=sim_h, labels=sim_l)
    ax.add_artist(leg1)
    _standard_legend(ax, loc='upper right', handles=obs_h, labels=obs_l)
    fig.tight_layout()

    save_figure(fig, os.path.join(OUTPUT_DIR,
                'StellarMassFunction_SF_Q' + OUTPUT_FORMAT))

def plot_3_gas_metallicity_vs_stellar_mass(primary, vanilla):
    """
    Gas-phase metallicity vs. stellar mass distribution.

    Shows the distribution of galaxies in the metallicity-mass plane
    as a KDE contour plot, with observational data overplotted.
    """
    print('Plot 3: Gas metallicity vs stellar mass')

    # Selection, applied identically to BOTH models: every galaxy that has a
    # gas-phase metallicity to report.
    #
    # SAGE26 previously carried an extra f_gas > 0.1 floor that SAGE16 did not.
    # SAGE26 is the more gas-poor model above m_* ~ 1e10 (median f_gas 0.03 against
    # 0.11 at log m_* = 10.5), so that floor bit far harder on it -- 33% of SAGE26
    # galaxies survived that bin against 53% of SAGE16's -- and with the per-bin
    # minimum then set to 50 it ended the SAGE26 curve at log m_* = 10.6 and SAGE16's
    # at 11.2. Most of that 0.6 dex gap was the mismatched selection, not the models.
    #
    # Both now keep every galaxy with cold gas, which carries the curves to 11.1 and
    # 11.2. Note this is a broader sample than the observations beside it: Tremonti+04
    # and the rest are emission-line selected and so effectively star-forming, whereas
    # this includes gas-poor galaxies whose metallicity would be hard to measure.
    def _gas_rich(data):
        return ((data['StellarMass'] > 1e8)
                & (data['ColdGas'] > 0)
                & (data['MetalsColdGas'] > 0))

    # --- Primary model ---
    w = _gas_rich(primary)
    log_mass = np.log10(primary['StellarMass'][w])
    gas_Z = metallicity_12logOH(primary['MetalsColdGas'][w],
                                primary['ColdGas'][w])

    # --- Plot ---
    fig = plt.figure()
    ax = fig.add_subplot(111)

    mass_bins = np.arange(8.0, 13.0 + 0.1, 0.1)
    plot_binned_median_1sigma(
        ax, log_mass, gas_Z, mass_bins,
        color='steelblue', label='SAGE26',
        alpha=0.25, lw=3.5, min_count=20,
        zorder_fill=Z_MODEL_BAND, zorder_line=Z_MODEL_LINE,
    )

    # --- C16 (Vanilla) model ---
    w_v = _gas_rich(vanilla)
    if np.any(w_v):
        log_mass_v = np.log10(vanilla['StellarMass'][w_v])
        gas_Z_v = metallicity_12logOH(vanilla['MetalsColdGas'][w_v],
                                      vanilla['ColdGas'][w_v])
        plot_binned_median_1sigma(
            ax, log_mass_v, gas_Z_v, mass_bins,
            color='purple', label='SAGE16', ls='--',
            alpha=0.20, lw=3.0, min_count=20,
            zorder_fill=Z_MODEL_BAND_ALT, zorder_line=Z_MODEL_LINE_ALT,
        )

    # --- Observational data ---
    for obs in load_mzr_observations():
        if obs['yerr'] is not None:
            ax.errorbar(obs['mass'], obs['Z'], yerr=obs['yerr'],
                        fmt=obs['fmt'], color=obs['color'],
                        markeredgecolor='k', markeredgewidth=1.0, linewidth=1.0,
                        markerfacecolor = 'gray', ms=8,
                        label=obs['label'], alpha=0.6, zorder=Z_OBS)
        else:
            ax.plot(obs['mass'], obs['Z'], obs['fmt'],
                    markeredgecolor='k', markeredgewidth=1.0, linewidth=1.0,
                    markerfacecolor = 'gray', ms=8,
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

    handles, labels = ax.get_legend_handles_labels()
    sim_set = {'SAGE26', 'SAGE16'}
    sim_h = [h for h, l in zip(handles, labels) if l in sim_set]
    sim_l = [l for l in labels if l in sim_set]
    obs_h = [h for h, l in zip(handles, labels) if l not in sim_set]
    obs_l = [l for l in labels if l not in sim_set]
    leg1 = _standard_legend(ax, loc='lower right', handles=sim_h, labels=sim_l)
    ax.add_artist(leg1)
    _standard_legend(ax, loc='upper left', handles=obs_h, labels=obs_l)
    fig.tight_layout()

    save_figure(fig, os.path.join(OUTPUT_DIR,
                'MetallicityStellarMass' + OUTPUT_FORMAT))

def plot_4_bh_bulge_mass(primary, vanilla):
    """
    Black hole mass vs. bulge mass relation.

    Shows the distribution of galaxies in the BH-bulge mass plane
    as a KDE contour plot, with observational data overplotted.
    """
    print('Plot 4: Black hole - bulge mass relation')

    # --- Primary model ---
    w = (primary['BlackHoleMass'] > 0) & (primary['BulgeMass'] > 0)
    log_bulge = np.log10(primary['BulgeMass'][w])
    log_bh = np.log10(primary['BlackHoleMass'][w])

    # --- Plot ---
    fig = plt.figure()
    ax = fig.add_subplot(111)

    bulge_bins = np.arange(8.0, 12.0 + 0.1, 0.1)
    plot_binned_median_1sigma(
        ax, log_bulge, log_bh, bulge_bins,
        color='steelblue', label='SAGE26',
        alpha=0.25, lw=3.5, min_count=20,
        zorder_fill=Z_MODEL_BAND, zorder_line=Z_MODEL_LINE,
    )

    # --- C16 (Vanilla) model ---
    w_v = (vanilla['BlackHoleMass'] > 0) & (vanilla['BulgeMass'] > 0)
    if np.any(w_v):
        log_bulge_v = np.log10(vanilla['BulgeMass'][w_v])
        log_bh_v = np.log10(vanilla['BlackHoleMass'][w_v])
        plot_binned_median_1sigma(
            ax, log_bulge_v, log_bh_v, bulge_bins,
            color='purple', label='SAGE16', ls='--',
            alpha=0.20, lw=3.0, min_count=20,
            zorder_fill=Z_MODEL_BAND_ALT, zorder_line=Z_MODEL_LINE_ALT,
        )

    # --- Observational data ---
    obs = load_bh_bulge_observations()
    sersic = ~obs['core']

    ax.errorbar(obs['log_M_sph'][sersic], obs['log_M_BH'][sersic],
                yerr=[obs['yerr'][0][sersic], obs['yerr'][1][sersic]],
                xerr=[obs['xerr'][0][sersic], obs['xerr'][1][sersic]],
                color='k', ls='none', lw=1, marker='d', ms=8, alpha=0.6,
                zorder=Z_OBS,
                markeredgecolor='k', markeredgewidth=0.8,
                        markerfacecolor = 'gray',
                label='S13 core')
    ax.errorbar(obs['log_M_sph'][obs['core']], obs['log_M_BH'][obs['core']],
                yerr=[obs['yerr'][0][obs['core']], obs['yerr'][1][obs['core']]],
                xerr=[obs['xerr'][0][obs['core']], obs['xerr'][1][obs['core']]],
                color='k', ls='none', lw=1, marker='o', ms=8,
                markeredgecolor='k', markeredgewidth=0.8, alpha=0.6,
                zorder=Z_OBS,
                        markerfacecolor = 'gray',
                label=_tex_safe(r'S13 S\'{e}rsic'))

    ax.plot(obs['haring_rix_x'], obs['haring_rix_y'], 'k--', zorder=Z_OBS,
            label=_tex_safe(r'Häring \& Rix 2004'))

    ax.set_xlim(8.0, 12.0)
    ax.set_ylim(6.0, 10.0)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.set_xlabel(r'$\log_{10}\ m_{\mathrm{bulge}}\ [M_{\odot}]$')
    ax.set_ylabel(r'$\log_{10}\ m_{\mathrm{BH}}\ [M_{\odot}]$')

    handles, labels = ax.get_legend_handles_labels()
    sim_set = {'SAGE26', 'SAGE16'}
    sim_h = [h for h, l in zip(handles, labels) if l in sim_set]
    sim_l = [l for l in labels if l in sim_set]
    obs_h = [h for h, l in zip(handles, labels) if l not in sim_set]
    obs_l = [l for l in labels if l not in sim_set]
    leg1 = _standard_legend(ax, loc='upper left', handles=obs_h, labels=obs_l)
    ax.add_artist(leg1)
    _standard_legend(ax, loc='lower right', handles=sim_h, labels=sim_l)
    fig.tight_layout()

    save_figure(fig, os.path.join(OUTPUT_DIR,
                'BlackHoleBulgeMass' + OUTPUT_FORMAT))

def plot_5_stellar_halo_mass(primary, vanilla):
    """
    Stellar mass vs. halo virial mass relation.

    Shows the SAGE26 distribution as a KDE contour plot
    with C16 as a scatter overlay and observational data.
    """
    print('Plot 5: Stellar-to-halo mass relation')

    # --- Primary model ---
    w = (primary['StellarMass'] > 0) & (primary['Mvir'] > 0)
    log_mvir = np.log10(primary['Mvir'][w])
    log_mstar = np.log10(primary['StellarMass'][w])

    # --- Plot ---
    fig = plt.figure()
    ax = fig.add_subplot(111)

    mvir_bins = np.arange(10.0, 15.0 + 0.1, 0.1)
    plot_binned_median_1sigma(
        ax, log_mvir, log_mstar, mvir_bins,
        color='steelblue', label='SAGE26',
        alpha=0.25, lw=3.5, min_count=20,
        zorder_fill=Z_MODEL_BAND, zorder_line=Z_MODEL_LINE,
    )

    # --- C16 (Vanilla) model ---
    w_v = (vanilla['StellarMass'] > 0) & (vanilla['Mvir'] > 0)
    if np.any(w_v):
        log_mvir_v = np.log10(vanilla['Mvir'][w_v])
        log_mstar_v = np.log10(vanilla['StellarMass'][w_v])
        plot_binned_median_1sigma(
            ax, log_mvir_v, log_mstar_v, mvir_bins,
            color='purple', label='SAGE16', ls='--',
            alpha=0.20, lw=3.0, min_count=20,
            zorder_fill=Z_MODEL_BAND_ALT, zorder_line=Z_MODEL_LINE_ALT,
        )

    # --- Observational data ---
    obs = load_shmr_observations()

    if 'moster' in obs:
        ax.plot(obs['moster']['mvir'], obs['moster']['mstar'],
                'k-', lw=2, label='Moster+13', zorder=Z_OBS)

    if 'romeo' in obs:
        ax.scatter(obs['romeo']['mvir'], obs['romeo']['mstar'],
                   marker='o', s=50, c='gray', label='Romeo+20',
                   edgecolor='k', linewidth=0.8, alpha=0.6, zorder=Z_OBS)

    if 'kravtsov' in obs:
        k = obs['kravtsov']
        xerr = [k['xerr_lo'], k['xerr_hi']]
        ax.errorbar(k['mvir'], k['mstar'], xerr=xerr,
                    fmt='s', color='k', ms=8, lw=1,
                    markeredgecolor='k', markeredgewidth=0.8,
                    markerfacecolor = 'gray', alpha=0.6, zorder=Z_OBS,
                    label='Kravtsov+18')

    if 'taylor' in obs:
        t = obs['taylor']
        ax.errorbar(t['mvir'], t['mstar'],
                    xerr=t['xerr'], yerr=t['yerr'],
                    fmt='d', color='k', ms=8, lw=1,
                    markeredgecolor='k', markeredgewidth=0.8,
                    markerfacecolor = 'gray', alpha=0.6, zorder=Z_OBS,
                    label='Taylor+20')

    ax.set_xlim(10.0, 15.0)
    ax.set_ylim(8.0, 12.0)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.set_xlabel(r'$\log_{10}\ M_{\mathrm{vir}}\ [M_{\odot}]$')
    ax.set_ylabel(r'$\log_{10}\ m_{\mathrm{*}}\ [M_{\odot}]$')

    handles, labels = ax.get_legend_handles_labels()
    sim_set = {'SAGE26', 'SAGE16'}
    sim_h = [h for h, l in zip(handles, labels) if l in sim_set]
    sim_l = [l for l in labels if l in sim_set]
    obs_h = [h for h, l in zip(handles, labels) if l not in sim_set]
    obs_l = [l for l in labels if l not in sim_set]
    leg1 = _standard_legend(ax, loc='upper left', handles=sim_h, labels=sim_l)
    ax.add_artist(leg1)
    _standard_legend(ax, loc='lower right', handles=obs_h, labels=obs_l)
    fig.tight_layout()

    save_figure(fig, os.path.join(OUTPUT_DIR,
                'StellarHaloMass' + OUTPUT_FORMAT))

def plot_6b_bulge_mass_size_median(primary, vanilla):
    """
    As plot 6, but with binned medians and 16--84% (1sigma) bands
    instead of scatter points, one for each formation channel.
    """
    print('Plot 6b: Bulge mass-size medians by formation type')

    w = (primary['BulgeMass'] > 0) & (primary['BulgeRadius'] > 0)
    bulge_mass = primary['BulgeMass'][w]
    bulge_radius = primary['BulgeRadius'][w] / HUBBLE_H / 0.001  # kpc
    inst_ratio = primary['InstabilityBulgeMass'][w] / bulge_mass

    log_mass = np.log10(bulge_mass)
    log_rad = np.log10(bulge_radius)

    merger_mask = inst_ratio < 0.1
    inst_mask = inst_ratio > 0.9
    mixed_mask = (inst_ratio >= 0.1) & (inst_ratio <= 0.9)

    n_tot = np.sum(w)
    print(f'  Total galaxies with bulges: {n_tot}')
    print(f'  Merger-dominated (ratio<0.1): {np.sum(merger_mask)}'
          f' ({100*np.sum(merger_mask)/n_tot:.1f}%)')
    print(f'  Instability-dominated (ratio>0.9): {np.sum(inst_mask)}'
          f' ({100*np.sum(inst_mask)/n_tot:.1f}%)')
    print(f'  Mixed (0.1-0.9): {np.sum(mixed_mask)}'
          f' ({100*np.sum(mixed_mask)/n_tot:.1f}%)')

    # --- Plot ---
    fig = plt.figure()
    ax = fig.add_subplot(111)

    bins = np.arange(8.0, 12.01, 0.2)
    cmap = plt.cm.plasma
    colors = [cmap(x) for x in (0.05, 0.45, 0.8)]
    channels = [
        (merger_mask, colors[0], 'Merger-driven'),
        (inst_mask, colors[1], 'Instability-driven'),
        (mixed_mask, colors[2], 'Mixed'),
    ]
    # Observed distribution first, so the model medians draw over it.
    lange = load_lange16_bulge_contours()
    # The percent sign must be escaped: the style file sets text.usetex, where a bare
    # % opens a comment and silently eats the rest of the label.
    lange_label = _tex_safe(
        'Lange+2016 (' + '/'.join(str(frac) for frac, _, _ in lange)
        + r'\% of GAMA bulges)') if lange else None
    for k, (frac, l_mass, l_rad) in enumerate(lange):
        if k == 0:
            ax.fill(l_mass, l_rad, color='0.55', alpha=0.13, lw=0, zorder=1)
        ax.plot(l_mass, l_rad, color='0.35', lw=1.3, alpha=0.9, zorder=2,
                label=lange_label if k == 0 else None)

    for mask, color, label in channels:
        plot_binned_median_1sigma(ax, log_mass[mask], log_rad[mask], bins,
                                  color=color, label=label, alpha=0.25,
                                  min_count=10)

    # Theoretical relations. Shen+2003 is drawn only across the mass range their
    # early-type sample actually covers -- extending it to the edges of the axes
    # advertises a measurement over two decades where they have no galaxies.
    log_M_shen = np.linspace(*SHEN03_MASS_RANGE, 100)
    ax.plot(log_M_shen, 0.56 * log_M_shen - 5.54, 'k--', lw=2,
            label='Shen+2003 (classical)', zorder=10)
    log_M = np.linspace(8, 12, 100)
    ax.plot(log_M, 0.25 * log_M - 2.5, ls='--', color='dimgray', lw=2,
            alpha=0.8, label='Pseudo-bulge (shallow)', zorder=10)

    ax.set_xlim(8.0, 12.0)
    ax.set_ylim(-2.0, 3.0)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.set_xlabel(r'$\log_{10}\ m_{\mathrm{bulge}}\ [M_{\odot}]$')
    ax.set_ylabel(r'$\log_{10}\ R_{\mathrm{bulge}}\ [\mathrm{kpc}]$')

    handles, labels = ax.get_legend_handles_labels()
    median_names = {'Merger-driven', 'Instability-driven', 'Mixed'}
    med_h = [h for h, l in zip(handles, labels) if l in median_names]
    med_l = [l for l in labels if l in median_names]
    line_h = [h for h, l in zip(handles, labels) if l not in median_names]
    line_l = [l for l in labels if l not in median_names]
    leg1 = _standard_legend(ax, loc='upper left', handles=med_h, labels=med_l)
    ax.add_artist(leg1)
    _standard_legend(ax, loc='lower right', handles=line_h, labels=line_l)
    fig.tight_layout()

    save_figure(fig, os.path.join(OUTPUT_DIR,
                'BulgeMassSize_Median' + OUTPUT_FORMAT))

# Tracked figures need nothing below z ~ 1.5 (plot 12's axis ends at 4 Gyr,
# z ~ 1.6; the others stop at z = 3), so tracking stops there.
_TRACK_Z_MIN = 1.5

_TRACK_SNAPS = [s for s in range(8, 64) if REDSHIFTS[s] >= _TRACK_Z_MIN]

def plot_13_ffb_vs_redshift(snapdata):
    """
    FFB fraction as a function of halo mass at different redshifts.

    Shows theoretical sigmoid curves (from the SAGE26 model) overlaid
    with binned simulation data with bootstrap error bars.
    """
    print('Plot 13: FFB fraction vs redshift')

    # Target redshifts (defined early so we can pre-load only the needed snaps)
    redshift_targets = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    needed_snaps = sorted({np.argmin(np.abs(np.array(REDSHIFTS) - z_t))
                           for z_t in redshift_targets})

    # MBK25 smooth simulation data (square markers).  The MBK25 points below are
    # commented out, so the load is too -- re-enable both together.
    snapdata_bk25 = {}
    # if os.path.exists(FFB_BK25_SMOOTH_DIR):
    #     print('  Loading MBK25 smooth snapshots...')
    #     snapdata_bk25 = load_snapshots(FFB_BK25_SMOOTH_DIR, needed_snaps)

    def _bin_ffb(d):
        """Bin FFB fraction vs halo mass. Returns (centres, fracs, ffb_errs, mass_errs)."""
        central = d['Type'] == 0
        Mvir_d = d['Mvir'][central]
        ffb_d = d['FFBRegime'][central].astype(float)
        pos = Mvir_d > 0
        lM = np.log10(Mvir_d[pos])
        ffb_d = ffb_d[pos]
        bin_edges = np.linspace(8, 14, 17)
        fracs, ferrs, merrs, centres = [], [], [], []
        for j in range(len(bin_edges) - 1):
            mask = (lM >= bin_edges[j]) & (lM < bin_edges[j + 1])
            n = np.sum(mask)
            if n < 10:
                continue
            vals = ffb_d[mask]
            frac = np.mean(vals)
            if frac == 0.0 or frac == 1.0:
                zs = 1.0
                denom = 1 + zs**2 / n
                centre_w = (frac + zs**2 / (2 * n)) / denom
                margin = zs * np.sqrt((frac * (1 - frac) + zs**2 / (4 * n)) / n) / denom
                el = max(0, frac - (centre_w - margin))
                eh = max(0, (centre_w + margin) - frac)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = stats.bootstrap((vals,), np.mean, n_resamples=1000,
                                          confidence_level=0.6827, method='percentile')
                el = max(0, frac - res.confidence_interval.low)
                eh = max(0, res.confidence_interval.high - frac)
            fracs.append(frac)
            ferrs.append([el, eh])
            masses = lM[mask]
            mm = np.mean(masses)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mres = stats.bootstrap((masses,), np.mean, n_resamples=1000,
                                        confidence_level=0.6827, method='percentile')
            merrs.append([max(0, mm - mres.confidence_interval.low),
                          max(0, mres.confidence_interval.high - mm)])
            centres.append(mm)
        return centres, fracs, ferrs, merrs

    fig, ax = plt.subplots()

    # Halo mass range (log10 M_sun)
    log_Mvir = np.linspace(8, 14, 500)
    Mvir = 10.0**log_Mvir

    cmap = plt.cm.plasma
    # Truncate colormap to avoid lightest yellow
    colors = [cmap(i / (len(redshift_targets) - 1) * 0.85)
              for i in range(len(redshift_targets))]

    delta_log_M = 0.15  # model smoothing width

    for z_target, color in zip(redshift_targets, colors):
        # Find closest snapshot
        snap_idx = np.argmin(np.abs(np.array(REDSHIFTS) - z_target))
        actual_z = REDSHIFTS[snap_idx]

        # Theoretical curve at actual snapshot redshift
        f_theory = ffb_fraction(Mvir, actual_z, delta_log_M)
        M_thresh = ffb_threshold_mass_msun(actual_z)
        ax.plot(log_Mvir, f_theory, color=color, lw=2,
                label=f'$z = {actual_z:.2f}$')
        ax.axvline(np.log10(M_thresh), color=color, ls=':', alpha=1.0, lw=1)

        # MBK25 theoretical curve (log-normal concentration scatter, sigma_c=0.2)
        # f_mbk25 = ffb_fraction_mbk25(Mvir, actual_z, sigma_c=0.2)
        # ax.plot(log_Mvir, f_mbk25, color=color, lw=2, ls='--', alpha=0.8)

        # Li+24 simulation data — circles
        if snap_idx in snapdata:
            centres, fracs, ferrs, merrs = _bin_ffb(snapdata[snap_idx])
            if len(centres) > 0:
                ax.errorbar(centres, fracs,
                            xerr=np.array(merrs).T, yerr=np.array(ferrs).T,
                            fmt='o', color=color, markersize=8, capsize=3,
                            alpha=0.6, markeredgecolor='k', markeredgewidth=0.3)

        # MBK25 smooth simulation data — squares
        # if snap_idx in snapdata_bk25:
        #     centres, fracs, ferrs, merrs = _bin_ffb(snapdata_bk25[snap_idx])
        #     if len(centres) > 0:
        #         ax.errorbar(centres, fracs,
        #                     xerr=np.array(merrs).T, yerr=np.array(ferrs).T,
        #                     fmt='s', color=color, markersize=8, capsize=3,
        #                     alpha=0.6, markeredgecolor='k', markeredgewidth=0.3)

    ax.axhline(0.5, color='gray', ls='--', alpha=1.0, lw=1)

    ax.set_xlabel(r'$\log_{10}\ M_{\mathrm{vir}}\ [M_{\odot}]$')
    ax.set_ylabel(r'$f_{\mathrm{FFB}}$')
    ax.set_xlim(8.25, 13)
    ax.set_ylim(0, 1)

    # Proxy artists: line styles (theory) and marker shapes (simulation)
    from matplotlib.lines import Line2D
    kw_mk = dict(lw=0, markersize=8, markeredgecolor='k', markeredgewidth=0.3, alpha=0.6)
    # proxy_li24_line  = Line2D([0], [0], color='gray', lw=2, ls='-',
    #                           label='Li+24')
    # proxy_mbk25_line = Line2D([0], [0], color='gray', lw=2, ls='--',
    #                           label=r'MBK25 (theory, log-normal $c$)')
    proxy_li24_pts   = Line2D([0], [0], color='gray', marker='o',
                              label='FFB', **kw_mk)
    # proxy_mbk25_pts  = Line2D([0], [0], color='gray', marker='s',
    #                           label='MBK25', **kw_mk)
    style_handles = [
                     proxy_li24_pts]
    z_handles, z_labels = ax.get_legend_handles_labels()
    _standard_legend(ax, loc='upper left',
                     handles=style_handles + z_handles,
                     labels=[h.get_label() for h in style_handles] + z_labels)
    fig.tight_layout()

    save_figure(fig, os.path.join(OUTPUT_DIR,
                'FFBvsRedshift' + OUTPUT_FORMAT))

def plot_15_sfr_vs_stellar_mass(primary, vanilla):
    """
    Star formation rate vs. stellar mass distribution.

    Shows the distribution of galaxies in the SFR-mass plane
    as a KDE contour plot, with C16 as a scatter overlay.
    Includes median lines exclusively for the star-forming populations.
    """
    print('Plot 15: SFR vs stellar mass (All + SF Medians)')

    # --- Primary model ---
    sfr = primary['SfrDisk'] + primary['SfrBulge']
    w = (primary['StellarMass'] > 1e8) & (sfr > 0)
    log_mass = np.log10(primary['StellarMass'][w])
    log_sfr = np.log10(sfr[w])

    # Safely calculate the SF mask on the already-filtered arrays
    # log(SFR/Mass) = log(SFR) - log(Mass)
    starforming_sage26 = (log_sfr - log_mass) > SSFR_CUT

    # --- Plot ---
    fig = plt.figure()
    ax = fig.add_subplot(111)

    mass_bins = np.arange(8.0, 12.0 + 0.1, 0.1)
    
    # Plot ALL SAGE26 galaxies
    plot_binned_median_1sigma(
        ax, log_mass, log_sfr, mass_bins,
        color='steelblue', label='SAGE26 (All)',
        alpha=0.25, lw=3.5, min_count=20,
        zorder_fill=Z_MODEL_BAND, zorder_line=Z_MODEL_LINE,
    )

    # Plot SAGE26 median for SF only
    centers, med_sfr, _, _ = binned_median(log_mass[starforming_sage26], log_sfr[starforming_sage26], mass_bins)
    ax.plot(centers, med_sfr, color='steelblue', lw=1.5, ls=':', label='SAGE26 (SF)', zorder=Z_MODEL_LINE+1)


    # --- C16 (Vanilla) model ---
    sfr_v = vanilla['SfrDisk'] + vanilla['SfrBulge']
    w_v = (vanilla['StellarMass'] > 1e8) & (sfr_v > 0)
    if np.any(w_v):
        log_mass_v = np.log10(vanilla['StellarMass'][w_v])
        log_sfr_v = np.log10(sfr_v[w_v])
        
        starforming_vanilla = (log_sfr_v - log_mass_v) > SSFR_CUT

        # Plot ALL SAGE16 galaxies
        plot_binned_median_1sigma(
            ax, log_mass_v, log_sfr_v, mass_bins,
            color='purple', label='SAGE16 (All)', ls='--',
            alpha=0.20, lw=3.5, min_count=20,
            zorder_fill=Z_MODEL_BAND_ALT, zorder_line=Z_MODEL_LINE_ALT,
        )

        # Plot SAGE16 median for SF only
        centers_v, med_sfr_v, _, _ = binned_median(log_mass_v[starforming_vanilla], log_sfr_v[starforming_vanilla], mass_bins)
        ax.plot(centers_v, med_sfr_v, color='purple', lw=1.5, ls=':', label='SAGE16 (SF)', zorder=Z_MODEL_LINE_ALT+1)

    # --- Load Brinchmann et al. (2004) data ---
    bz04_mass, bz04_sfr = load_brinchmann_sfr_mass_2004_data()
    if bz04_mass is not None and bz04_sfr is not None:
        ax.scatter(bz04_mass, bz04_sfr, marker='d', alpha=0.6, zorder=Z_OBS,
                facecolors='gray', edgecolors='black', s=50,
                label='Brinchmann+04')

    # --- Load Terrazas et al. (2017) data ---
    ter_mass, ter_sfr = load_terrazas17_mbh_host_sfr_data()
    if ter_mass is not None and ter_sfr is not None:
        # Plot with error bars
        ax.errorbar(ter_mass, ter_sfr, xerr=0.2, yerr=0.3, fmt='o', ecolor='black', alpha=0.6, zorder=Z_OBS,
                   mfc='gray', mec='black', ms=8, mew=1.0, elinewidth=1.0, label='Terrazas+17')
        
    # --- Load and plot GAMA ProSpect Claudia data ---
    log_ms, log_sfr = load_gama_prospect_claudia()
    if log_ms is not None and log_sfr is not None:
        # Plot binned medians/errors
        bins = np.linspace(8, 12, 13)
        centers_gama, med_gama, p25_gama, p75_gama = binned_median(log_ms, log_sfr, bins)
        valid = ~np.isnan(med_gama)
        ax.errorbar(centers_gama[valid], med_gama[valid], 
                    yerr=[med_gama[valid] - p25_gama[valid], p75_gama[valid] - med_gama[valid]],
                    fmt='s', color='black', label='Bellstedt+20', markersize=8, alpha=0.6, zorder=Z_OBS,
                    markeredgewidth=0.8, markerfacecolor='gray', markeredgecolor='black')
        

    ax.set_xlim(8.0, 12.0)
    ax.set_ylim(-4.0, 2.0)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.set_xlabel(r'$\log_{10}\ m_{\mathrm{*}}\ [M_{\odot}]$')
    ax.set_ylabel(r'$\log_{10}\ \mathrm{SFR}\ [M_{\odot}\,\mathrm{yr}^{-1}]$')

    # Use startswith to cleanly capture the newly named 'All' and 'SF' simulation labels
    handles, labels = ax.get_legend_handles_labels()
    sim_h = [h for h, l in zip(handles, labels) if l.startswith(('SAGE26', 'SAGE16'))]
    sim_l = [l for l in labels if l.startswith(('SAGE26', 'SAGE16'))]
    obs_h = [h for h, l in zip(handles, labels) if not l.startswith(('SAGE26', 'SAGE16'))]
    obs_l = [l for l in labels if not l.startswith(('SAGE26', 'SAGE16'))]
    
    leg1 = _standard_legend(ax, loc='lower left', handles=sim_h, labels=sim_l)
    ax.add_artist(leg1)
    _standard_legend(ax, loc='upper left', handles=obs_h, labels=obs_l)
    fig.tight_layout()

    save_figure(fig, os.path.join(OUTPUT_DIR,
                'StarFormationRate' + OUTPUT_FORMAT))

# Above this many galaxies the bootstrap of a summed density is replaced by its
# normal limit, sum +/- sqrt(N) * std; resampling ~10^7 galaxies 100 times per
# snapshot is what made the density histories slow on the full boxes.
BOOT_EXACT_MAX = 100_000

def _bootstrap_sum_range(x, n_boot):
    """16th and 84th percentiles of the bootstrapped sum of *x*."""
    n = len(x)
    if n <= BOOT_EXACT_MAX:
        boot = np.array([np.sum(x[np.random.randint(0, n, n)]) for _ in range(n_boot)])
        return np.percentile(boot, 16), np.percentile(boot, 84)
    total, sigma = np.sum(x), np.sqrt(n) * np.std(x)
    return total - sigma, total + sigma

def _density_history(path, first, last, volume, props, quantity, tag,
                     mass_convert=MASS_CONVERT, n_boot=100):
    """
    Per-snapshot density sum(quantity(d)) / volume for snapshots first..last, with
    bootstrap 16th-84th percentiles (zero when n_boot = 0). Cached in OUTPUT_DIR
    and reused until the model files change; *tag* names the quantity in the cache.
    """
    files = find_model_files(path)
    n_snaps = last - first + 1
    dens, lo, hi = np.zeros(n_snaps), np.zeros(n_snaps), np.zeros(n_snaps)
    if not files:
        return dens, lo, hi
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cache = os.path.join(OUTPUT_DIR, f'history_{tag}_{os.path.basename(os.path.normpath(path))}.npz')
    stamp = max(os.path.getmtime(f) for f in files)
    key = f'{os.path.abspath(path)}|{first}|{last}|{volume}|{mass_convert}|{n_boot}|{BOOT_EXACT_MAX}|{stamp}'
    if os.path.exists(cache):
        c = np.load(cache)
        if str(c['key']) == key:
            print(f'  {tag} history for {path} from cache')
            return c['dens'], c['lo'], c['hi']

    for snap in range(first, last + 1):
        d = read_snap_from_files(files, f'Snap_{snap}', props, mass_convert=mass_convert)
        if not d:
            continue
        x = quantity(d)
        if len(x) == 0:
            continue
        i = snap - first
        dens[i] = np.sum(x) / volume
        if n_boot:
            b_lo, b_hi = _bootstrap_sum_range(x, n_boot)
            lo[i], hi[i] = b_lo / volume, b_hi / volume
    np.savez(cache, dens=dens, lo=lo, hi=hi, key=key)
    return dens, lo, hi

def plot_16_sfrd_history():
    """
    Plot G: Cosmic SFR Density History.
    Replicates the structure of the uploaded paper_plots.py.
    Iterates snapshots 0-63 explicitly to capture full history.
    """
    print('Plot 16: SFR Density History (CSRDH)')
    
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
            'color': 'steelblue', 'ls': '-', 'lw': 3.5,
            'redshifts': redshifts, 'first_snap': FirstSnap, 'last_snap': LastSnap,
            'volume': VOLUME,
        })

    # 2. Vanilla Model (C16)
    if os.path.exists(VANILLA_DIR):
        sim_dirs.append({
            'path': VANILLA_DIR, 'label': 'SAGE16',
            'color': 'purple', 'ls': '--', 'lw': 3.5,
            'redshifts': redshifts, 'first_snap': FirstSnap, 'last_snap': LastSnap,
            'volume': VOLUME,
        })

    # 3. miniUchuu Model
    if os.path.exists(MINIUCHUU_DIR):
        sim_dirs.append({
            'path': MINIUCHUU_DIR, 'label': 'SAGE26 (miniUchuu)',
            'color': 'steelblue', 'ls': '--', 'lw': 3.5,
            'redshifts': redshifts_mu, 'first_snap': MINIUCHUU_FIRST_SNAP, 'last_snap': MINIUCHUU_LAST_SNAP,
            'volume': MINIUCHUU_VOLUME,
        })

    # # 4. CGM Dynamical Time Model
    # if os.path.exists(CGM_DYN_DIR):
    #     sim_dirs.append({
    #         'path': CGM_DYN_DIR, 'label': 'SAGE26 (CGM Dyn Time)',
    #         'color': 'darkorange', 'ls': '-.', 'lw': 3.5,
    #         'redshifts': redshifts, 'first_snap': FirstSnap, 'last_snap': LastSnap,
    #         'volume': VOLUME,
    #     })

    # # 5. Simple CGM with disk smoothing
    # if os.path.exists(DISK_SMOOTH_DIR):
    #     sim_dirs.append({
    #         'path': DISK_SMOOTH_DIR, 'label': 'SAGE26 (Disk Smooth)',
    #         'color': 'darkgreen', 'ls': ':', 'lw': 3.5,
    #         'redshifts': redshifts, 'first_snap': FirstSnap, 'last_snap': LastSnap,
    #         'volume': VOLUME,
    #     })

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # --- 2. PLOT OBSERVATIONAL DATA ---
    obs_style = dict(markerfacecolor='gray', markeredgecolor='k', markeredgewidth=1.0,
                     ecolor='k', color='k', ms=8, lw=1.0, alpha=0.6, ls='none',
                     zorder=Z_OBS)
    z_d18, sfrd_d18, err_d18, zlo_d18, zhi_d18 = load_driver_sfrd_2018_data()
    if z_d18 is not None:
        ax.errorbar(z_d18, sfrd_d18, yerr=err_d18, xerr=[zlo_d18, zhi_d18],
                    fmt='o', label='Driver+18', **obs_style)
    z_ds, sfrd_ds, elo_ds, ehi_ds, zlo_ds, zhi_ds = load_dsilva_sfrd_2023_data()
    if z_ds is not None:
        ax.errorbar(z_ds, sfrd_ds, yerr=[elo_ds, ehi_ds], xerr=[zlo_ds, zhi_ds],
                    fmt='D', label="D'Silva+23", **obs_style)
    for dataset, fmt in (('Novak+17', '^'), ('Bouwens+15', '<'), ('Finkelstein+15', '>')):
        z_t, sfrd_t, eu_t, el_t = load_sfrd_table(dataset)
        if z_t is not None:
            ax.errorbar(z_t, sfrd_t, yerr=[el_t, eu_t], fmt=fmt, label=dataset, **obs_style)

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
        sfr_density, sfr_density_lo, sfr_density_hi = _density_history(
            sim_path, sim_first, sim_last, sim_volume, ['SfrDisk', 'SfrBulge'],
            lambda d: d['SfrDisk'] + d['SfrBulge'], 'sfrd',
            n_boot=N_BOOT if do_bootstrap else 0)

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

    # ===== QUANTITATIVE COMPARISON: SAGE26 vs SAGE16 =====
    if 'SAGE26 (Millennium)' in model_results and 'SAGE16' in model_results:
        print("\n" + "="*60)
        print("QUANTITATIVE COMPARISON: SAGE26 vs SAGE16 (SFRD)")
        print("="*60)

        z_sage = model_results['SAGE26 (Millennium)']['z']
        sfrd_sage = model_results['SAGE26 (Millennium)']['sfrd']
        z_c16 = model_results['SAGE16']['z']
        sfrd_c16 = model_results['SAGE16']['sfrd']

        # Filter to plot range (z <= 7.5)
        z_mask = z_sage <= 7.5
        z_sage = z_sage[z_mask]
        sfrd_sage = sfrd_sage[z_mask]

        # Interpolate C16 to SAGE26 redshifts for direct comparison
        from scipy.interpolate import interp1d
        c16_interp = interp1d(z_c16, sfrd_c16, bounds_error=False, fill_value=np.nan)
        sfrd_c16_matched = c16_interp(z_sage)

        # Find valid comparison points
        valid = ~np.isnan(sfrd_c16_matched)
        z_valid = z_sage[valid]
        sfrd_diff = sfrd_c16_matched[valid] - sfrd_sage[valid]

        print(f"\n  Comparison over z = {z_valid.min():.1f} to {z_valid.max():.1f}")
        print(f"  Mean difference (SAGE16 - SAGE26):  {np.mean(sfrd_diff):+.3f} dex")
        print(f"  Median difference:               {np.median(sfrd_diff):+.3f} dex")
        print(f"  Std of difference:               {np.std(sfrd_diff):.3f} dex")
        print(f"  Max difference at z={z_valid[np.argmax(sfrd_diff)]:.1f}: {np.max(sfrd_diff):+.3f} dex ({10**np.max(sfrd_diff):.1f}x)")
        print(f"  Min difference at z={z_valid[np.argmin(sfrd_diff)]:.1f}: {np.min(sfrd_diff):+.3f} dex ({10**np.min(sfrd_diff):.1f}x)")

        print("\n  SFRD at specific redshifts:")
        for target_z in [0, 1, 2, 3, 4, 5, 6]:
            idx = np.argmin(np.abs(z_valid - target_z))
            if np.abs(z_valid[idx] - target_z) < 0.5:
                sage_val = sfrd_sage[valid][idx]
                c16_val = sfrd_c16_matched[valid][idx]
                diff = c16_val - sage_val
                print(f"    z~{z_valid[idx]:.1f}: SAGE16={c16_val:.2f}, SAGE26={sage_val:.2f}, Δ={diff:+.2f} dex ({10**diff:.1f}x)")

        print("="*60 + "\n")

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

    sim_names = {'SAGE26 (Millennium)', 'SAGE26 (miniUchuu)', 'SAGE16', 'SAGE26 (CGM Dyn Time)'}
    handles, labels = ax.get_legend_handles_labels()
    sim_h = [h for h, l in zip(handles, labels) if l in sim_names]
    sim_l = [l for l in labels if l in sim_names]
    obs_h = [h for h, l in zip(handles, labels) if l not in sim_names]
    obs_l = [l for l in labels if l not in sim_names]
    # Legend order: SAGE16 and SAGE26 (Millennium) swapped
    sim_order = {'SAGE16': 0, 'SAGE26 (Millennium)': 1, 'SAGE26 (miniUchuu)': 2, 'SAGE26 (CGM Dyn Time)': 3}
    _sim_pairs = sorted(zip(sim_l, sim_h), key=lambda p: sim_order.get(p[0], 99))
    sim_l = [p[0] for p in _sim_pairs]
    sim_h = [p[1] for p in _sim_pairs]
    leg1 = _standard_legend(ax, loc='lower right', handles=sim_h, labels=sim_l)
    ax.add_artist(leg1)
    _standard_legend(ax, loc='lower left', handles=obs_h, labels=obs_l)

    fig.tight_layout()

    outputFile = os.path.join(OUTPUT_DIR, 'SFR_Density_History_Comparison' + OUTPUT_FORMAT)
    save_figure(fig, outputFile)

def _select_grid_snapshot(redshifts, first_snap, last_snap, z_lo, z_hi,
                          z_ref=None, tol=0.05):
    """
    Snapshot representing the redshift bin [z_lo, z_hi] for one model.

    Without *z_ref*, returns the in-bin snapshot nearest the bin centre.

    With *z_ref* -- the redshift the panel is anchored to, from
    _grid_reference_redshift() -- returns the in-bin snapshot nearest *z_ref* instead.
    Selecting each model independently against the bin centre looks equivalent but is
    not: the simulations have different snapshot tables (Millennium 64 snapshots,
    miniUchuu 50), so at high redshift the two curves end up at genuinely different
    epochs while the panel is labelled with a single bin. Measured on the shipped runs,
    that put them dz = 0.70 apart in 8.5 < z < 9.5 and dz = 0.47 apart in 5.5 < z < 6.5,
    on opposite sides of the bin centre -- comparable to 0.2-0.3 dex of SMF evolution at
    the massive end, i.e. an offset between the model curves that is an artefact of
    snapshot sampling rather than physics.

    Alignment is not pursued at any cost: candidates within *tol* of the best |z - z_ref|
    count as ties and are broken toward the bin centre, so a snapshot is not dragged away
    from the epoch the panel advertises in exchange for a negligible gain in alignment.
    In 4.5 < z < 5.5 that guard matters -- matching would have improved alignment by 0.01
    while nearly tripling the distance from the bin centre.

    Returns (snap_num, snap_redshift), or (None, None) if the model has no snapshot in
    the bin. Residual offsets that no choice can remove are reported by
    _report_grid_alignment().
    """
    sub = np.asarray(redshifts)[first_snap:last_snap + 1]
    idx = np.where((sub >= z_lo) & (sub <= z_hi))[0]
    if idx.size == 0:
        return None, None
    z_mid = 0.5 * (z_lo + z_hi)
    if z_ref is None:
        pick = idx[np.argmin(np.abs(sub[idx] - z_mid))]
    else:
        d = np.abs(sub[idx] - z_ref)
        near = idx[d <= d.min() + tol]
        pick = near[np.argmin(np.abs(sub[near] - z_mid))]
    return int(pick + first_snap), float(sub[pick])

def _grid_reference_redshift(models, z_lo, z_hi, tol=0.05):
    """
    Epoch a redshift-grid panel is anchored to, chosen so the overlaid models land as
    close to a common redshift as their snapshot tables allow.

    Anchoring on one nominated model -- the first listed, say -- does not do this. The
    followers can only choose from snapshots they actually have, so the anchor decides
    which *pairing* of snapshots the panel ends up comparing, and the pairing reached
    from the first model is often not the tightest one available. In 5.5 < z < 6.5 the
    shipped runs each have two snapshots: anchoring on Millennium pairs z = 6.197 with
    miniUchuu's z = 6.342 (dz = 0.15), while the z = 5.724 / z = 5.722 pairing sitting
    right there in the same bin is dz = 0.002 and no further from the bin centre.

    So every in-bin snapshot redshift of every model is tried as the anchor and the one
    giving the smallest spread across models wins. Alignment is not pursued at any cost:
    anchors within *tol* of the best spread count as ties and are broken toward the bin
    centre, so the panel is not dragged to the edge of the range it advertises in
    exchange for a negligible gain. That guard binds in 4.5 < z < 5.5, where the tightest
    pairing (dz = 0.113) sits at z ~ 4.5 and is passed over for one at dz = 0.134 that is
    0.2 closer to the centre.

    Returns None if no model has a snapshot in the bin, which leaves
    _select_grid_snapshot() to fall back on the bin centre. Residual offsets that no
    anchor can remove are reported by _report_grid_alignment().
    """
    z_mid = 0.5 * (z_lo + z_hi)

    anchors = set()
    for model in models:
        sub = np.asarray(model['redshifts'])[model['first_snap']:model['last_snap'] + 1]
        anchors.update(sub[(sub >= z_lo) & (sub <= z_hi)].tolist())
    if not anchors:
        return None

    scored = []
    for z_try in sorted(anchors):
        picked = [z for _, z in (
            _select_grid_snapshot(m['redshifts'], m['first_snap'], m['last_snap'],
                                  z_lo, z_hi, z_ref=z_try, tol=tol)
            for m in models) if z is not None]
        spread = max(picked) - min(picked)
        offset = max(abs(z - z_mid) for z in picked)
        scored.append((spread, offset, z_try))

    best_spread = min(sc[0] for sc in scored)
    ties = [sc for sc in scored if sc[0] <= best_spread + tol]
    return min(ties, key=lambda sc: sc[1])[2]

def _report_grid_alignment(chosen, threshold=0.10):
    """
    Print the redshift each model was drawn at per bin, and flag bins where the models
    are further apart than *threshold* in z.

    Those bins are limited by the coarser snapshot sampling of one simulation, not by the
    selection: the offset cannot be removed without a snapshot that does not exist. They
    are printed so the residual is visible in the log rather than hidden inside a panel
    labelled with a single redshift range.

    *chosen* maps bin label -> list of (model_label, snap_num, snap_redshift).
    """
    print('  snapshot alignment across models:')
    worst = []
    for bin_label, entries in chosen.items():
        if len(entries) < 2:
            continue
        zs = [e[2] for e in entries]
        spread = max(zs) - min(zs)
        detail = ',  '.join(f'{lab}: Snap_{sn} z={zz:.3f}' for lab, sn, zz in entries)
        flag = '   <-- limited by snapshot sampling' if spread > threshold else ''
        print(f'    {bin_label:<14s} spread dz={spread:.3f}   {detail}{flag}')
        if spread > threshold:
            worst.append((bin_label, spread))
    if worst:
        print(f'    {len(worst)} of {len(chosen)} bins exceed dz = {threshold:g}: '
              + ', '.join(f'{b} ({d:.2f})' for b, d in worst))
    else:
        print(f'    all bins aligned to within dz = {threshold:g}')

def _load_smf_grid_observations():
    """
    Load all observational SMF datasets for the redshift grid plot.

    Returns list of dicts with keys:
        z, log_mass, log_phi, err_lo, err_hi, label, marker, ms
    err_lo / err_hi are positive offsets in dex (for errorbar yerr).
    They may be None when errors are unavailable.
    """
    obs = []
    h = HUBBLE_H  # 0.73

    # ------------------------------------------------------------------
    # 1. Baldry+08  (z ~ 0, inline data)
    # ------------------------------------------------------------------
    try:
        _baldry = np.array([
            [7.05,1.3531e-01,6.0741e-02],[7.15,1.3474e-01,6.0109e-02],
            [7.25,2.0971e-01,7.7965e-02],[7.35,1.7161e-01,3.1841e-02],
            [7.45,2.1648e-01,5.7832e-02],[7.55,2.1645e-01,3.9988e-02],
            [7.65,2.0837e-01,4.8713e-02],[7.75,2.0402e-01,7.0061e-02],
            [7.85,1.5536e-01,3.9182e-02],[7.95,1.5232e-01,2.6824e-02],
            [8.05,1.5067e-01,4.8824e-02],[8.15,1.3032e-01,2.1892e-02],
            [8.25,1.2545e-01,3.5526e-02],[8.35,9.8472e-02,2.7181e-02],
            [8.45,8.7194e-02,2.8345e-02],[8.55,7.0758e-02,2.0808e-02],
            [8.65,5.8190e-02,1.3359e-02],[8.75,5.6057e-02,1.3512e-02],
            [8.85,5.1380e-02,1.2815e-02],[8.95,4.4206e-02,9.6866e-03],
            [9.05,4.1149e-02,1.0169e-02],[9.15,3.4959e-02,6.7898e-03],
            [9.25,3.3111e-02,8.3704e-03],[9.35,3.0138e-02,4.7741e-03],
            [9.45,2.6692e-02,5.5029e-03],[9.55,2.4656e-02,4.4359e-03],
            [9.65,2.2885e-02,3.7915e-03],[9.75,2.1849e-02,3.9812e-03],
            [9.85,2.0383e-02,3.2930e-03],[9.95,1.9929e-02,2.9370e-03],
            [10.05,1.8865e-02,2.4624e-03],[10.15,1.8136e-02,2.5208e-03],
            [10.25,1.7657e-02,2.4217e-03],[10.35,1.6616e-02,2.2784e-03],
            [10.45,1.6114e-02,2.1783e-03],[10.55,1.4366e-02,1.8819e-03],
            [10.65,1.2588e-02,1.8249e-03],[10.75,1.1372e-02,1.4436e-03],
            [10.85,9.1213e-03,1.5816e-03],[10.95,6.1125e-03,9.6735e-04],
            [11.05,4.3923e-03,9.6254e-04],[11.15,2.5463e-03,5.0038e-04],
            [11.25,1.4298e-03,4.2816e-04],[11.35,6.4867e-04,1.6439e-04],
            [11.45,2.8294e-04,9.9799e-05],[11.55,1.0617e-04,4.9085e-05],
            [11.65,3.2702e-05,2.4546e-05],[11.75,1.2571e-05,1.2571e-05],
            [11.85,8.4589e-06,8.4589e-06],[11.95,7.4764e-06,7.4764e-06],
        ], dtype=np.float32)
        # h^-2 to the model's h, then onto Chabrier. The IMF part was folded into
        # the old -0.26 constant; it is split out so the registry can audit it.
        log_m = np.log10(10**_baldry[:, 0] / h / h) - 0.22 + imf_shift('Baldry+08')
        phi_c = _baldry[:, 1] * h**3
        phi_u = (_baldry[:, 1] + _baldry[:, 2]) * h**3
        phi_l = (_baldry[:, 1] - _baldry[:, 2]) * h**3
        ok = phi_l > 0
        lp = np.log10(phi_c[ok])
        obs.append({'z': 0.05, 'log_mass': log_m[ok], 'log_phi': lp,
                     'err_lo': lp - np.log10(phi_l[ok]),
                     'err_hi': np.log10(phi_u[ok]) - lp,
                     'label': 'Baldry+08', 'marker': 'o', 'ms': 8})
    except Exception as e:
        print(f"  Baldry+08 load error: {e}")

    # ------------------------------------------------------------------
    # 2. Thorne+21  (GAMA/DEVILS; CSV: logM, phi, phi_16, phi_84 — linear)
    #    Loaded for z=1.6–4.0 only; lower-z bins are already well-covered
    #    by Baldry+08, Muzzin+13, and SMF_data_points.
    # ------------------------------------------------------------------
    _thorne = [
        ('./data/smf/Thorne21/SMFvals_z1.6.csv', 1.6),
        ('./data/smf/Thorne21/SMFvals_z2.csv',   2.0),
        ('./data/smf/Thorne21/SMFvals_z2.4.csv', 2.4),
        ('./data/smf/Thorne21/SMFvals_z3.csv',   3.0),
        ('./data/smf/Thorne21/SMFvals_z3.5.csv', 3.5),
        ('./data/smf/Thorne21/SMFvals_z4.csv',   4.0),
    ]
    h_t = 0.7
    for fpath, z_val in _thorne:
        try:
            if not os.path.exists(fpath):
                continue
            d = np.genfromtxt(fpath, delimiter=',', skip_header=1)
            if d.ndim != 2 or d.shape[1] < 4:
                continue
            m, phi, p16, p84 = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
            ok = np.isfinite(m) & (phi > 0) & (p16 > 0) & (p84 > 0)
            if not np.any(ok):
                continue
            phi_c  = phi[ok]  * (h_t / h)**3
            p16_c  = p16[ok]  * (h_t / h)**3
            p84_c  = p84[ok]  * (h_t / h)**3
            lp = np.log10(phi_c)
            obs.append({'z': z_val, 'log_mass': m[ok] + imf_shift('Thorne+21'),
                         'log_phi': lp,
                         'err_lo': lp - np.log10(p16_c),
                         'err_hi': np.log10(p84_c) - lp,
                         'label': 'Thorne+21', 'marker': 's', 'ms': 6})
        except Exception as e:
            print(f"  Thorne+21 z={z_val} load error: {e}")

    # ------------------------------------------------------------------
    # 3. Weaver+23  (farmer TXT: logM, bw, phi, phi_lo, phi_hi — linear)
    # ------------------------------------------------------------------
    _weaver = [
        ('./data/smf/COSMOS2020/SMF_Farmer_v2.1_1.5z2.0_total.txt', 1.75),
        ('./data/smf/COSMOS2020/SMF_Farmer_v2.1_2.0z2.5_total.txt', 2.25),
        ('./data/smf/COSMOS2020/SMF_Farmer_v2.1_2.5z3.0_total.txt', 2.75),
        ('./data/smf/COSMOS2020/SMF_Farmer_v2.1_3.0z3.5_total.txt', 3.25),
        ('./data/smf/COSMOS2020/SMF_Farmer_v2.1_3.5z4.5_total.txt', 4.0),
        ('./data/smf/COSMOS2020/SMF_Farmer_v2.1_4.5z5.5_total.txt', 5.0),
        ('./data/smf/COSMOS2020/SMF_Farmer_v2.1_5.5z6.5_total.txt', 6.0),
        ('./data/smf/COSMOS2020/SMF_Farmer_v2.1_6.5z7.5_total.txt', 7.0),
    ]
    for fpath, z_val in _weaver:
        try:
            if not os.path.exists(fpath):
                continue
            d = None
            for delim in [None, ',', '\t', ' ']:
                for skip in [0, 1, 2]:
                    try:
                        d = np.genfromtxt(fpath, delimiter=delim, skip_header=skip)
                        if d.ndim == 2 and d.shape[1] >= 5:
                            break
                    except Exception:
                        d = None
                if d is not None and d.ndim == 2:
                    break
            if d is None or d.ndim != 2 or d.shape[1] < 5:
                continue
            m, phi, plo, phi_hi = d[:, 0], d[:, 2], d[:, 3], d[:, 4]
            ok = np.isfinite(m) & (phi > 0) & (plo > 0) & (phi_hi > 0)
            if np.any(ok):
                lp = np.log10(phi[ok])
                obs.append({'z': z_val, 'log_mass': m[ok] + imf_shift('Weaver+23'),
                             'log_phi': lp,
                             'err_lo': lp - np.log10(plo[ok]),
                             'err_hi': np.log10(phi_hi[ok]) - lp,
                             'label': 'Weaver+23', 'marker': 'D', 'ms': 8})
        except Exception as e:
            print(f"  Weaver+23 z={z_val} load error: {e}")

    # ------------------------------------------------------------------
    # 4. Muzzin+13  (dat: z_lo z_hi M_star E_M logPhi)
    # ------------------------------------------------------------------
    try:
        _muz_file = './data/smf/SMF_Muzzin2013.dat'
        if os.path.exists(_muz_file):
            h_m = 0.7
            bins = {}
            with open(_muz_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    p = line.split()
                    if len(p) < 5:
                        continue
                    zl, zh, ms, _, lp = float(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])
                    if lp < -10:
                        continue
                    key = (zl, zh)
                    if key not in bins:
                        bins[key] = {'m': [], 'lp': []}
                    phi_c = 10**lp * (h_m / h)**3
                    bins[key]['m'].append(ms + imf_shift('Muzzin+13'))
                    bins[key]['lp'].append(np.log10(phi_c))
            for (zl, zh), v in bins.items():
                m_arr = np.array(v['m'])
                lp_arr = np.array(v['lp'])
                obs.append({'z': 0.5*(zl+zh), 'log_mass': m_arr, 'log_phi': lp_arr,
                             'err_lo': None, 'err_hi': None,
                             'label': 'Muzzin+13', 'marker': '^', 'ms': 8})
    except Exception as e:
        print(f"  Muzzin+13 load error: {e}")

    # ------------------------------------------------------------------
    # 5. Santini+12  (dat: z_lo z_hi lg_mass lg_phi err_hi err_lo ...)
    # ------------------------------------------------------------------
    try:
        _san_file = './data/smf/SMF_Santini2012.dat'
        if os.path.exists(_san_file):
            h_s = 0.7
            bins = {}
            with open(_san_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    p = line.split()
                    if len(p) < 6:
                        continue
                    zl, zh = float(p[0]), float(p[1])
                    lg_m, lg_p = float(p[2]), float(p[3])
                    ehi, elo = float(p[4]), float(p[5])
                    if lg_p < -10 or not np.isfinite(lg_p):
                        continue
                    key = (zl, zh)
                    if key not in bins:
                        bins[key] = {'m': [], 'lp': [], 'ehi': [], 'elo': []}
                    phi_c = 10**lg_p * (h_s / h)**3
                    bins[key]['m'].append(lg_m + imf_shift('Santini+12'))
                    bins[key]['lp'].append(np.log10(phi_c))
                    bins[key]['ehi'].append(ehi)
                    bins[key]['elo'].append(elo)
            for (zl, zh), v in bins.items():
                obs.append({'z': 0.5*(zl+zh), 'log_mass': np.array(v['m']),
                             'log_phi': np.array(v['lp']),
                             'err_lo': np.array(v['elo']),
                             'err_hi': np.array(v['ehi']),
                             'label': 'Santini+12', 'marker': 'v', 'ms': 8})
    except Exception as e:
        print(f"  Santini+12 load error: {e}")

    # ------------------------------------------------------------------
    # 6. Wright+18  (dat: med_z mass log_y dlog_yu dlog_yd ycv)
    # ------------------------------------------------------------------
    try:
        _wr_file = './data/smf/Wright18_CombinedSMF.dat'
        if os.path.exists(_wr_file):
            h_w = 0.7
            bins = {}
            with open(_wr_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    p = line.split()
                    if len(p) < 6:
                        continue
                    mz, sm, ly = float(p[0]), float(p[1]), float(p[2])
                    dyu, dyd = float(p[3]), float(p[4])
                    if ly < -10 or not np.isfinite(ly):
                        continue
                    ly_corr = ly + np.log10(1.0 / 0.25)  # bin-width correction
                    phi_c = 10**ly_corr * (h_w / h)**3
                    if mz not in bins:
                        bins[mz] = {'m': [], 'lp': [], 'ehi': [], 'elo': []}
                    bins[mz]['m'].append(sm + imf_shift('Wright+18'))
                    bins[mz]['lp'].append(np.log10(phi_c))
                    bins[mz]['ehi'].append(dyu)
                    bins[mz]['elo'].append(dyd)
            for z_val, v in bins.items():
                obs.append({'z': z_val, 'log_mass': np.array(v['m']),
                             'log_phi': np.array(v['lp']),
                             'err_lo': np.array(v['elo']),
                             'err_hi': np.array(v['ehi']),
                             'label': 'Wright+18', 'marker': 'p', 'ms': 8})
    except Exception as e:
        print(f"  Wright+18 load error: {e}")

    # ------------------------------------------------------------------
    # 7. Observational compilation  (SMF_data_points.ecsv)
    # ------------------------------------------------------------------
    if HAS_ASTROPY:
        try:
            _obs_file = './data/smf/SMF_data_points.ecsv'
            if os.path.exists(_obs_file):
                with open(_obs_file, 'r') as f:
                    lines = f.readlines()
                data_lines = [l.strip() for l in lines
                              if not l.startswith('#') and l.strip()]
                bins = {}
                for line in data_lines[1:]:
                    if '"' not in line:
                        continue
                    parts = line.split('"')
                    zbin_str = parts[1]
                    dp = parts[2].strip().split()
                    if len(dp) < 3:
                        continue
                    ms, phi, dphi = float(dp[0]), float(dp[1]), float(dp[2])
                    # Parse z_center from string like "0.2 < z < 0.5"
                    nums = [float(s) for s in zbin_str.replace('<', ' ').replace('z', ' ').split()
                            if s.replace('.', '', 1).replace('-', '', 1).isdigit()]
                    if len(nums) >= 2:
                        zc = 0.5 * (nums[0] + nums[-1])
                    else:
                        continue
                    if zc not in bins:
                        bins[zc] = {'m': [], 'phi': [], 'dphi': []}
                    bins[zc]['m'].append(ms)
                    bins[zc]['phi'].append(phi)
                    bins[zc]['dphi'].append(dphi)
                for zc, v in bins.items():
                    m_arr = np.array(v['m'])
                    phi_arr = np.array(v['phi'])
                    dphi_arr = np.array(v['dphi'])
                    ok = phi_arr > 0
                    if np.any(ok):
                        lp = np.log10(phi_arr[ok])
                        upper = phi_arr[ok] + dphi_arr[ok]
                        lower = phi_arr[ok] - dphi_arr[ok]
                        ehi = np.where(upper > 0, np.log10(upper) - lp, 0.0)
                        elo = np.where(lower > 0, lp - np.log10(lower), 0.0)
                        obs.append({'z': zc,
                                     'log_mass': m_arr[ok] + imf_shift('COSMOS-Web'),
                                     'log_phi': lp,
                                     'err_lo': elo, 'err_hi': ehi,
                                     'label': 'COSMOS-Web', 'marker': 'h', 'ms': 8})
        except Exception as e:
            print(f"  COSMOS-Web load error: {e}")

    # ------------------------------------------------------------------
    # 8. Harvey+25  (ECSV: z, log10Mstar, phi, phi_error_low, phi_error_upp)
    # ------------------------------------------------------------------
    if HAS_ASTROPY:
        try:
            _har_file = './data/smf/FiducialBagpipesGSMF.ecsv'
            if os.path.exists(_har_file):
                t = Table.read(_har_file, format='ascii.ecsv')
                # Handle possible column name variations
                _pcol_lo = ([c for c in t.colnames if 'low' in c and 'phi' in c] + ['phi_error_low'])[0]
                _pcol_hi = ([c for c in t.colnames if 'upp' in c and 'phi' in c] + ['phi_error_upp'])[0]
                for z_val in np.unique(t['z']):
                    mask = t['z'] == z_val
                    s = t[mask]
                    log_m = np.array(s['log10Mstar'])
                    phi = np.array(s['phi'])
                    phi_elo = np.array(s[_pcol_lo])
                    phi_ehi = np.array(s[_pcol_hi])
                    ok = phi > 0
                    if np.any(ok):
                        lp = np.log10(phi[ok])
                        upper = phi[ok] + phi_ehi[ok]
                        lower = phi[ok] - phi_elo[ok]
                        ehi = np.where(upper > 0, np.log10(upper) - lp, 0.0)
                        elo = np.where(lower > 0, lp - np.log10(lower), 0.0)
                        obs.append({'z': float(z_val),
                                     'log_mass': log_m[ok] + imf_shift('Harvey+25'),
                                     'log_phi': lp,
                                     'err_lo': elo, 'err_hi': ehi,
                                     'label': 'Harvey+25', 'marker': 'H', 'ms': 8})
        except Exception as e:
            print(f"  Harvey+25 load error: {e}")

    # ------------------------------------------------------------------
    # 9–12.  High-z ECSV datasets (Stefanon+21, Navarro-Carrera+23,
    #         Weibel+24, Kikuchihara+20)
    # ------------------------------------------------------------------
    _highz_ecsv = [
        {'file': './data/smf/stefanon_smf_2021.ecsv',
         'label': 'Stefanon+21', 'marker': '*', 'ms': 8,
         'zcol': 'redshift_bin', 'mcol': 'log_M',
         'phi_col': 'phi', 'phi_eu': 'phi_err_up', 'phi_el': 'phi_err_low',
         'phi_scale': 1e-4, 'phi_log': False, 'zbins': [6, 7, 8, 9, 10]},
        {'file': './data/smf/navarro_carrera_smf_2023.ecsv',
         'label': 'Navarro-Carrera+24', 'marker': 'X', 'ms': 8,
         'zcol': 'redshift_bin', 'mcol': 'log_M',
         'phi_col': 'phi', 'phi_eu': 'phi_err_up', 'phi_el': 'phi_err_low',
         'phi_scale': 1e-4, 'phi_log': False, 'zbins': [6, 7, 8]},
        {'file': './data/smf/weibel_smf_2024.ecsv',
         'label': 'Weibel+24', 'marker': 'P', 'ms': 8,
         'zcol': 'redshift_bin', 'mcol': 'log_M',
         'phi_col': 'log_phi', 'phi_eu': 'log_phi_err_up', 'phi_el': 'log_phi_err_low',
         'phi_scale': 1.0, 'phi_log': True, 'zbins': [6, 7, 8, 9]},
        {'file': './data/smf/kikuchihara_smf_2020.ecsv',
         'label': 'Kikuchihara+20', 'marker': 'd', 'ms': 8,
         'zcol': 'redshift_approx', 'mcol': 'log_M_star',
         'phi_col': 'phi_star', 'phi_eu': 'phi_star_err_up', 'phi_el': 'phi_star_err_low',
         'phi_scale': 1e-5, 'phi_log': False, 'zbins': [6, 7, 8, 9]},
    ]
    if HAS_ASTROPY:
        for cfg in _highz_ecsv:
            try:
                if not os.path.exists(cfg['file']):
                    continue
                t = Table.read(cfg['file'], format='ascii.ecsv')
                for zb in cfg['zbins']:
                    zm = t[cfg['zcol']] == zb
                    if not np.any(zm):
                        continue
                    s = t[zm]
                    log_m = np.array(s[cfg['mcol']]) + imf_shift(cfg['label'])
                    if cfg['phi_log']:
                        lp = np.array(s[cfg['phi_col']])
                        eu = np.array(s[cfg['phi_eu']])
                        el = np.array(s[cfg['phi_el']])
                        lp_hi = lp + eu
                        lp_lo = lp - el
                        lp_lo[el == 0] = np.nan
                        ok = np.isfinite(lp)
                        obs.append({'z': float(zb), 'log_mass': log_m[ok],
                                     'log_phi': lp[ok],
                                     'err_lo': np.where(np.isfinite(lp_lo[ok]),
                                                        lp[ok] - lp_lo[ok], 0.0),
                                     'err_hi': np.where(np.isfinite(lp_hi[ok]),
                                                        lp_hi[ok] - lp[ok], 0.0),
                                     'label': cfg['label'], 'marker': cfg['marker'],
                                     'ms': cfg['ms']})
                    else:
                        phi_lin = np.array(s[cfg['phi_col']], dtype=float) * cfg['phi_scale']
                        eu_lin = np.array(s[cfg['phi_eu']], dtype=float) * cfg['phi_scale']
                        el_lin = np.array(s[cfg['phi_el']], dtype=float) * cfg['phi_scale']
                        ok = phi_lin > 0
                        if not np.any(ok):
                            continue
                        lp = np.log10(phi_lin[ok])
                        upper = phi_lin[ok] + eu_lin[ok]
                        lower = phi_lin[ok] - el_lin[ok]
                        ehi = np.where(upper > 0, np.log10(upper) - lp, 0.0)
                        elo = np.where(lower > 0, lp - np.log10(lower), 0.0)
                        obs.append({'z': float(zb), 'log_mass': log_m[ok],
                                     'log_phi': lp, 'err_lo': elo, 'err_hi': ehi,
                                     'label': cfg['label'], 'marker': cfg['marker'],
                                     'ms': cfg['ms']})
            except Exception as e:
                print(f"  {cfg['label']} load error: {e}")

    # ------------------------------------------------------------------
    # 13. Song+16  (CANDELS; wide ECSV: log_M + log_phi columns per z)
    #     Covers z=4–8; h=0.7, Chabrier IMF.
    # ------------------------------------------------------------------
    if HAS_ASTROPY:
        try:
            _song_file = './data/smf/song_smf_2016.ecsv'
            if os.path.exists(_song_file):
                t = Table.read(_song_file, format='ascii.ecsv')
                h_s = 0.7
                log_phi_corr = 3.0 * np.log10(h_s / h)
                log_m_song = np.array(t['log_M'], dtype=float)
                _song_bins = [
                    (4, 'phi_z4', 'phi_z4_err_up', 'phi_z4_err_lo'),
                    (5, 'phi_z5', 'phi_z5_err_up', 'phi_z5_err_lo'),
                    (6, 'phi_z6', 'phi_z6_err_up', 'phi_z6_err_lo'),
                    (7, 'phi_z7', 'phi_z7_err_up', 'phi_z7_err_lo'),
                    (8, 'phi_z8', 'phi_z8_err_up', 'phi_z8_err_lo'),
                ]
                for z_val, pcol, eu_col, el_col in _song_bins:
                    lp  = np.array(t[pcol],  dtype=float)
                    eu  = np.array(t[eu_col], dtype=float)
                    el  = np.array(t[el_col], dtype=float)
                    ok  = np.isfinite(lp) & np.isfinite(eu) & np.isfinite(el)
                    if not np.any(ok):
                        continue
                    obs.append({'z': float(z_val),
                                'log_mass': log_m_song[ok] + imf_shift('Song+16'),
                                'log_phi':  lp[ok] + log_phi_corr,
                                'err_lo': el[ok], 'err_hi': eu[ok],
                                'label': 'Song+16', 'marker': '<', 'ms': 6})
        except Exception as e:
            print(f"  Song+16 load error: {e}")

    print(f"  Loaded {len(obs)} observational SMF datasets")
    return obs

def plot_18_smf_redshift_grid():
    """
    Plot: 3x5 grid of Stellar Mass Functions at 15 redshift bins.
    Each panel shows the SMF for SAGE26 (Millennium) and SAGE26 (miniUchuu).
    """
    print('Plot 18: SMF Redshift Grid')

    # Redshift bins: (z_lo, z_hi)
    z_bins = [
        (0.0, 0.5),   (0.5, 0.8),   (0.8, 1.1),
        (1.1, 1.5),   (1.5, 2.0),   (2.0, 2.5),
        (2.5, 3.0),   (3.0, 3.5),   (3.5, 4.5),
        (4.5, 5.5),   (5.5, 6.5),   (6.5, 7.5),
        (7.5, 8.5),   (8.5, 9.5),   (9.5, 12.0),
    ]

    # Models to plot
    mill_redshifts = np.array(REDSHIFTS)
    mu_redshifts = np.array(MINIUCHUU_REDSHIFTS)

    models = []
    if os.path.exists(PRIMARY_DIR):
        models.append({
            'path': PRIMARY_DIR, 'label': 'SAGE26 (Millennium)',
            'color': 'black', 'ls': '-', 'lw': 3.5,
            'redshifts': mill_redshifts, 'first_snap': 0, 'last_snap': 63,
            'volume': VOLUME, 'mass_convert': MASS_CONVERT,
        })
    if os.path.exists(MINIUCHUU_DIR):
        models.append({
            'path': MINIUCHUU_DIR, 'label': 'SAGE26 (miniUchuu)',
            'color': 'steelblue', 'ls': '--', 'lw': 2.5,
            'redshifts': mu_redshifts, 'first_snap': MINIUCHUU_FIRST_SNAP, 'last_snap': MINIUCHUU_LAST_SNAP,
            'volume': MINIUCHUU_VOLUME, 'mass_convert': MINIUCHUU_MASS_CONVERT,
        })
    if os.path.exists(VANILLA_DIR):
        models.append({
            'path': VANILLA_DIR, 'label': 'SAGE16',
            'color': 'firebrick', 'ls': '--', 'lw': 2.5,
            'redshifts': mill_redshifts, 'first_snap': 0, 'last_snap': 63,
            'volume': VOLUME, 'mass_convert': MASS_CONVERT,
        })
    if os.path.exists(NOFFB_DIR):
        models.append({
            'path': NOFFB_DIR, 'label': 'SAGE26 (no FFBs)', 'band': False,
            'color': 'black', 'ls': '--', 'lw': 2.5,
            'redshifts': mill_redshifts, 'first_snap': 0, 'last_snap': 63,
            'volume': VOLUME, 'mass_convert': MASS_CONVERT,
        })

    # Load observational data
    all_obs = _load_smf_grid_observations()
    labels_used = set()  # track legend entries to avoid duplicates

    fig, axes = plt.subplots(5, 3, figsize=(15, 25), sharex=True, sharey=True)
    fig.set_tight_layout(False)
    axes_flat = axes.flatten()
    binwidth = 0.1
    _bin18_smf = {}  # store per-bin SMF for comparison printout
    _alignment = {}  # bin -> [(model, snap, z)] for the alignment report

    for i, (z_lo, z_hi) in enumerate(z_bins):
        ax = axes_flat[i]
        z_mid = 0.5 * (z_lo + z_hi)

        # The model that samples this bin most coarsely sets the epoch for the panel;
        # the rest are matched to the redshift it actually landed on, not to the bin
        # centre, so overlaid curves are compared at the same epoch.
        # See _grid_reference_redshift() and _select_grid_snapshot().
        z_ref_panel = _grid_reference_redshift(models, z_lo, z_hi)
        for model in models:
            snap_num, snap_z = _select_grid_snapshot(
                model['redshifts'], model['first_snap'], model['last_snap'],
                z_lo, z_hi, z_ref=z_ref_panel)
            if snap_num is None:
                continue
            _alignment.setdefault(f'{z_lo:.1f}-{z_hi:.1f}', []).append(
                (model['label'], snap_num, snap_z))
            snap_name = f'Snap_{snap_num}'

            try:
                model_files = find_model_files(model['path'])
                d = read_snap_from_files(model_files, snap_name,
                                         ['StellarMass'],
                                         mass_convert=model['mass_convert'])
                if not d:
                    continue
                m_stars = d['StellarMass']
                w = m_stars > 0
                if np.sum(w) == 0:
                    continue
                log_m = np.log10(m_stars[w])

                    # Use bootstrap for SAGE26 models
                if model['label'].startswith('SAGE26') and model.get('band', True):
                    x, phi, phi_lo, phi_hi, _ = mass_function_bootstrap(
                        log_m, model['volume'], binwidth, n_boot=100)
                    valid = np.isfinite(phi)
                    ax.plot(x[valid], phi[valid],
                            lw=model['lw'], color=model['color'],
                            ls=model['ls'],
                            label=model['label'] if i == 0 else None)
                    # Bootstrap shading
                    boot_valid = np.isfinite(phi_lo) & np.isfinite(phi_hi)
                    if np.any(boot_valid):
                        ax.fill_between(x[boot_valid], phi_lo[boot_valid], phi_hi[boot_valid],
                                        color=model['color'], alpha=0.2, linewidth=0)
                else:
                    x, phi, _ = mass_function(log_m, model['volume'], binwidth)
                    valid = np.isfinite(phi)
                    ax.plot(x[valid], phi[valid],
                            lw=model['lw'], color=model['color'],
                            ls=model['ls'],
                            label=model['label'] if i == 0 else None)
                _bin18_smf.setdefault(i, {})[model['label']] = (x[valid], phi[valid])
            except Exception as e:
                print(f"  Error loading {snap_name} from {model['path']}: {e}")
                continue

        # Print model comparisons for this z-bin
        _b18 = _bin18_smf.get(i, {})
        _ref18 = 'SAGE26 (Millennium)'
        if _ref18 in _b18 and len(_b18) > 1:
            _x26, _p26 = _b18[_ref18]
            print(f"\n  SMF z=[{z_lo:.1f},{z_hi:.1f}] (SAGE26 Millennium vs others):")
            for _olab, (_xo, _po) in _b18.items():
                if _olab == _ref18 or len(_xo) == 0 or len(_x26) == 0:
                    continue
                _xc = np.round(np.intersect1d(np.round(_x26, 4), np.round(_xo, 4)), 4)
                if len(_xc) == 0:
                    continue
                _ip18 = np.array([np.argmin(np.abs(_x26 - xc)) for xc in _xc])
                _io18 = np.array([np.argmin(np.abs(_xo - xc)) for xc in _xc])
                _dv = _p26[_ip18] - _po[_io18]
                _ok18 = np.isfinite(_dv)
                if np.sum(_ok18) == 0:
                    continue
                _dv_ok = _dv[_ok18];  _xv18 = _xc[_ok18]
                print(f"    vs {_olab}: Mean Δ={np.mean(_dv_ok):+.3f} dex, "
                      f"Median={np.median(_dv_ok):+.3f} dex, "
                      f"Max at log10(M*)={_xv18[np.argmax(_dv_ok)]:.2f}: {np.max(_dv_ok):+.3f} dex")
                for _tm in [9.0, 10.0, 10.5, 11.0]:
                    _ti = np.argmin(np.abs(_xv18 - _tm))
                    if np.abs(_xv18[_ti] - _tm) < binwidth:
                        print(f"      log10(M*)={_xv18[_ti]:.2f}: SAGE26={_p26[_ip18][_ok18][_ti]:.2f}, "
                              f"{_olab}={_po[_io18][_ok18][_ti]:.2f}, Δ={_dv_ok[_ti]:+.2f} dex")

        # Plot observational data for this redshift bin
        for od in all_obs:
            z_obs = od['z']
            # Match obs to bin: inclusive lower, exclusive upper (last bin inclusive)
            if i == len(z_bins) - 1:
                in_bin = z_lo <= z_obs <= z_hi
            else:
                in_bin = z_lo <= z_obs < z_hi
            if not in_bin:
                continue
            lbl = od['label'] if od['label'] not in labels_used else None
            if lbl is not None:
                labels_used.add(od['label'])
            yerr = None
            if od['err_lo'] is not None and od['err_hi'] is not None:
                yerr = [od['err_lo'], od['err_hi']]
            ax.errorbar(od['log_mass'], od['log_phi'], yerr=yerr,
                        fmt=od['marker'], color='grey', ms=od['ms'],
                        markeredgecolor='k', markeredgewidth=0.8,
                            markerfacecolor = 'gray',
                        alpha=0.6, lw=1.5, capsize=1.5, label=lbl, zorder=1)

        # Redshift label in each panel
        ax.text(0.95, 0.95, rf'${z_lo:.1f} < z < {z_hi:.1f}$',
                transform=ax.transAxes, ha='right', va='top')

    # Axis limits and labels
    axes_flat[0].set_xlim(8.001, 12.2)
    axes_flat[0].set_ylim(-6, -0.8)

    for i, ax in enumerate(axes_flat):
        row, col = divmod(i, 3)
        if col == 0:
            ax.set_ylabel(r'$\log_{10}\ \phi\ [\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1}]$')
        if row == 4:
            ax.set_xlabel(r'$\log_{10}\ m_{\mathrm{*}}\ [M_{\odot}]$')
        # Re-apply tick style from stylesheet (sharex/sharey overrides these)
        ax.tick_params(axis='both', which='both', direction='in',
                       top=True, bottom=True, left=True, right=True)

    # Per-panel legends (only panels with new labelled entries get a legend)
    for ax in axes_flat:
        _, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(loc='lower left', frameon=False)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.001, wspace=0.001)

    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.yaxis.set_major_locator(plt.MultipleLocator(2.0))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.2))

    _report_grid_alignment(_alignment)

    outputFile = os.path.join(OUTPUT_DIR, 'SMF_Redshift_Grid' + OUTPUT_FORMAT)
    save_figure(fig, outputFile)

def _feedback_params(directory=PRIMARY_DIR):
    """Feedback parameters straight from the run header, so the analytic curves
    cannot drift from the model that produced the points."""
    fallback = dict(eps_disk=2.9, eps_halo=0.3, alpha_z=FIRE_REDSHIFT_EXPONENT,
                    eta_sn=5.0e-3, energy_sn=1.0e51,
                    eps_max=MAX_SN_ENERGY_COUPLING, sn_bound=1, reheat_bound=1)
    files = _find_model_files_early(directory)
    if not files:
        return fallback
    with h5.File(files[0], 'r') as f:
        r = f['Header/Runtime'].attrs
        get = lambda k, d: (float(r[k]) if k in r else d)
        return dict(eps_disk=get('FeedbackReheatingEpsilon', 2.9),
                    eps_halo=get('FeedbackEjectionEfficiency', 0.3),
                    alpha_z=FIRE_REDSHIFT_EXPONENT,
                    eta_sn=get('EtaSN', 5.0e-3),
                    energy_sn=get('EnergySN', 1.0e51),
                    # The SN energy bound is hardcoded on in the model, and the
                    # cap is MAX_SN_ENERGY_COUPLING in src/model_misc.h. Both the
                    # ejection coupling and the reheating mass loading use it --
                    # there is no separate reheating switch, and drawing the
                    # analytic curves without the cap leaves a spurious offset at
                    # high V_vir and high z.
                    eps_max=MAX_SN_ENERGY_COUPLING,
                    sn_bound=1,
                    reheat_bound=1)

# Cap on the effective SN coupling, applied to both the ejection term and the
# reheating mass loading. Matches MAX_SN_ENERGY_COUPLING in src/model_misc.h.
MAX_SN_ENERGY_COUPLING = 2.0

# FIRE (Muratov et al. 2015) critical circular velocity separating the two
# power-law slopes of the wind loading factor. Matches FIRE_V_CRIT_KMS in
# src/model_starformation_and_feedback.c.
FIRE_V_CRIT = 60.0

# FIRE redshift scaling of the wind loading factor, eta ~ (1+z)^alpha. Matches
# FIRE_REDSHIFT_EXPONENT in src/model_starformation_and_feedback.c.
FIRE_REDSHIFT_EXPONENT = 1.25

FIRE_BETA_LOW = -3.2

FIRE_BETA_HIGH = -1.0

def _fire_scaling(vvir, z, p):
    """f(V_vir, z) = (1+z)^alpha (V_vir/60)^beta, the shared factor in both
    the mass loading and the ejection energy."""
    v = np.maximum(np.asarray(vvir, dtype=float), 1.0)
    beta = np.where(v < FIRE_V_CRIT, FIRE_BETA_LOW, FIRE_BETA_HIGH)
    return (1.0 + z) ** p['alpha_z'] * (v / FIRE_V_CRIT) ** beta

def _sn_energy_per_mass_kms2(p):
    """eta_SN * E_SN in (km/s)^2 per unit mass -- the combination the code
    carries as EtaSNcode * EnergySNcode (the Hubble_h factors cancel)."""
    return p['eta_sn'] * p['energy_sn'] / _MSUN_CGS / 1.0e10

def _log10_tick_formatter(decimals=1):
    """Label a log-scaled axis with the log10 of the tick position.

    The axes are log-scaled and the labels say log10(...), so the numbers
    printed must be the exponents (-1, 0, 1, 2), not the values (0.1, 1, 10).
    *decimals* is fixed rather than per-tick so a set like 1.2, 1.4, 1.6, 1.8,
    2.0 does not render its one integral member as a bare "2".
    """
    def fmt(x, _pos):
        return '' if x <= 0 else f'{np.log10(x):.{decimals}f}'
    return FuncFormatter(fmt)

def plot_24_mass_loading_vs_velocity(primary, vanilla):
    """
    Supernova mass loading and ejection against halo virial velocity.

    Both panels show quantities MEASURED from SAGE26 at several redshifts, with
    the analytic expressions as thin reference lines.  Measured rather than
    analytic is the point: Major Comment 2 was "I cannot reproduce the model's
    stated behaviour from the equations", so a curve that satisfies the algebra
    by construction answers nothing.  What the panels demonstrate is that the
    code reproduces the equations, including where the energy bound departs
    from them.

    Panel (a): eta_reheat, the stored MassLoading, which is the value actually
    applied after the reheating bound.  It rises with redshift as (1+z)^1.25.

    Panel (b): mdot_eject / mdot_*, reconstructed per galaxy from the stored
    MassLoading, V_vir and the snapshot redshift, exactly as
    compute_sn_feedback() evaluates it.  It falls to zero at the SAME
    V_vir = V_SN sqrt(eps_halo/eps_disk) at every redshift, because f cancels
    between the feedback energy and the lifting energy.

    The contrast between the two panels is the figure's argument: mass loading
    is strongly redshift dependent, the ejection threshold is not.

    Not captured in panel (b): ejection is additionally limited to the gas
    present in the CGM or hot reservoir, which depends on reservoir state at the
    timestep and cannot be reconstructed from a snapshot.  The measured curve is
    therefore an upper bound at low V_vir.
    """
    print('Plot 24: Mass loading and ejection vs virial velocity')

    p = _feedback_params()
    esn = _sn_energy_per_mass_kms2(p)
    v_eject = np.sqrt(p['eps_halo'] * esn / p['eps_disk'])
    z_targets = [0.0, 1.0, 2.0, 4.0, 6.0]
    # Lower edge of the measured curves.  load_model() applies the MIN_PARTICLES
    # cut, but between 20 and ~30 particles the recorded V_vir and Mvir cease to
    # be mutually consistent and the measured eta falls away from the scaling by
    # up to an order of magnitude.  Truncate where the measurement is
    # trustworthy rather than where galaxies merely exist.
    V_MEASURED_MIN = 24.0
    vbins = np.logspace(np.log10(V_MEASURED_MIN), np.log10(600.0), 24)

    cmap = plt.get_cmap('plasma')
    colours = [cmap(x) for x in np.linspace(0.05, 0.85, len(z_targets))]

    # figsize and fonts match the other 1x2 figures (plot_9, plot_7b), which
    # take the stylesheet raw at this canvas size.  A smaller canvas with the
    # same absolute font sizes renders the text larger relative to the axes.
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6))

    # Each simulation has its own snapshot grid, so a target redshift maps to a
    # different actual z in each.  Two targets can also collide on one snapshot
    # if the grid is coarse at high z, which would draw two identical curves
    # under different labels.  Report the mapping and drop duplicates, so a
    # coincidence in the snapshot grid is never mistaken for a physical result.
    _drawn_snaps = {}
    _sat_marks = []
    print('  Fig. 8 redshift mapping (target -> snapshot -> actual z):')

    # The secondary simulation picks the redshift and the primary is matched to
    # it, rather than both snapping independently to the nominal target.  Taking
    # the nearest snapshot to the target in each simulation separately put the
    # two curves labelled z=6 at z=6.197 (Millennium) and z=5.722 (miniUchuu):
    # eta goes as (1+z)^alpha, so that Dz=0.475 alone offsets them by 8%, which
    # reads as a resolution difference and is nothing of the kind.  Millennium
    # Snap_19 sits at z=5.724, so matching on the secondary's redshift brings
    # the pair to Dz=0.002.  The analytic curve is drawn at the matched primary
    # redshift for the same reason.
    _have_secondary = model_files_exist(MINIUCHUU_DIR)

    for z_t, c in zip(z_targets, colours):
        if _have_secondary:
            snap2 = _snap_nearest_z(MINIUCHUU_REDSHIFTS, z_t)
            z_ref = MINIUCHUU_REDSHIFTS[snap2]
        else:
            snap2, z_ref = None, z_t
        snap = _snap_nearest_z(REDSHIFTS, z_ref)
        z = REDSHIFTS[snap]
        if snap in _drawn_snaps:
            print(f'    z={z_t:.0f} -> Snap_{snap} -> z={z:.3f}  SKIPPED, same'
                  f' snapshot as target z={_drawn_snaps[snap]:.0f}')
            continue
        _drawn_snaps[snap] = z_t
        # Label the redshift actually plotted, not the target it was chosen
        # from, so a matched pair at z=5.72 is never printed as "z = 6".
        _zlab = (rf'$z = {z:.0f}$' if abs(z - round(z)) < 0.1
                 else rf'$z = {z:.1f}$')
        print(f'    z={z_t:.0f} -> Snap_{snap} -> z={z:.3f} (primary)')

        # Where the ejection coupling saturates.  sn_energy_coupling() caps
        # eps_halo*f at eps_max, so E_FB never exceeds the whole supernova
        # budget -- but Eq. 14 as printed is unbounded, and left of these ticks
        # the plotted curve is the capped value, not what the equation gives.
        # Collected here and drawn after set_xlim(): tested against the axis
        # limits in-loop, the first redshift is compared with matplotlib's
        # default (0, 1) and silently dropped.
        if p['sn_bound']:
            _fc = p['eps_max'] / p['eps_halo']
            _fz = (1.0 + z) ** p['alpha_z']
            _sat_marks.append(
                ((60.0 * (_fz / _fc)) if _fz >= _fc
                 else 60.0 * (_fc / _fz) ** (-1.0 / 3.2), c))
        # No analytic reference curves in either panel.  Both quantities follow
        # their scalings by construction -- capped_eta_reheat() stores exactly
        # eps_disk*f, and the ejected mass is reconstructed from it -- so an
        # overlaid analytic curve restates the input rather than testing it.
        try:
            d = load_model(PRIMARY_DIR, snapshot=f'Snap_{snap}',
                           properties=['MassLoading', 'Vvir'])
        except Exception:                                       # noqa: BLE001
            continue
        vv = np.asarray(d['Vvir'], float)
        eta = np.asarray(d['MassLoading'], float)
        w = (eta > 0) & (vv > 0)
        if w.sum() < 100:
            continue
        plot_binned_median_1sigma(
            axL, vv[w], eta[w], vbins, color=c,
            label=_zlab, alpha=0.18, lw=2.2, min_count=20,
            zorder_fill=Z_MODEL_BAND, zorder_line=Z_MODEL_LINE)

        # Second simulation, dashed and without a shaded band so that ten
        # curves per panel stay readable.  MINIUCHUU_DIR must point at the
        # miniUchuu production output for the label below to be accurate.
        if snap2 is not None:
            z2 = z_ref
            # Residual mismatch after matching.  eta goes as (1+z)^alpha, so
            # report what the leftover Dz is worth: anything above a couple of
            # per cent is a redshift effect, not resolution.
            _off = 100.0 * (((1.0 + z2) / (1.0 + z)) ** p['alpha_z'] - 1.0)
            print(f'    z={z_t:.0f} -> Snap_{snap2} -> z={z2:.3f} (miniUchuu)'
                  f'   dz={z2 - z:+.3f} vs primary'
                  + (f'  ==> {_off:+.1f}% offset in eta from redshift alone'
                     if abs(_off) > 2.0 else '  (matched)'))
            try:
                d2 = load_model(MINIUCHUU_DIR, snapshot=f'Snap_{snap2}',
                                properties=['MassLoading', 'Vvir'])
                v2 = np.asarray(d2['Vvir'], float)
                e2 = np.asarray(d2['MassLoading'], float)
                w2 = (e2 > 0) & (v2 > 0)
                if w2.sum() >= 100:
                    plot_binned_median_1sigma(
                        axL, v2[w2], e2[w2], vbins, color=c, label=None,
                        ls='--', alpha=0.0, lw=1.8, min_count=20,
                        zorder_fill=Z_MODEL_BAND, zorder_line=Z_MODEL_LINE)
                    c2 = p['eps_halo'] * _fire_scaling(v2[w2], MINIUCHUU_REDSHIFTS[snap2], p)
                    if p['sn_bound']:
                        c2 = np.minimum(c2, p['eps_max'])
                    ef2 = c2 * 0.5 * esn
                    el2 = 0.5 * e2[w2] * v2[w2] ** 2
                    
                    # Retain all galaxies, setting zero/negative ejection to 1e-5
                    j2 = (ef2 - el2) / (0.5 * v2[w2] ** 2)
                    j2 = np.maximum(j2, 1e-5)
                    k2 = np.isfinite(j2)
                    
                    if k2.sum() > 100:
                        plot_binned_median_1sigma(
                            axR, v2[w2][k2], j2[k2], vbins, color=c, label=None,
                            ls='--', alpha=0.0, lw=1.8, min_count=20,
                            zorder_fill=Z_MODEL_BAND, zorder_line=Z_MODEL_LINE)
            except Exception:                                   # noqa: BLE001
                pass

        # mdot_eject / mdot_*, reconstructed exactly as the code evaluates it,
        # but using the STORED eta rather than the analytic one, so the
        # reheating bound is included as applied.
        coupling = p['eps_halo'] * _fire_scaling(vv[w], z, p)
        if p['sn_bound']:
            coupling = np.minimum(coupling, p['eps_max'])
        e_fb = coupling * 0.5 * esn
        e_lift = 0.5 * eta[w] * vv[w] ** 2
        
        # Retain all galaxies, setting zero/negative ejection to 1e-5
        ej = (e_fb - e_lift) / (0.5 * vv[w] ** 2)
        ej = np.maximum(ej, 1e-5)
        ok = np.isfinite(ej)
        
        if ok.sum() > 100:
            plot_binned_median_1sigma(
                axR, vv[w][ok], ej[ok], vbins, color=c, label=None,
                alpha=0.18, lw=2.2, min_count=20,
                zorder_fill=Z_MODEL_BAND, zorder_line=Z_MODEL_LINE)

    # observations, left panel only
    obs_handles = []
    for fname, marker, lbl in [('Chisholm_17_ml.csv', 'o', 'Chisholm+17'),
                               ('Heckman_15_ml.csv', 'X', 'Heckman+15'),
                               ('Rupke_05_ml.csv', 's', 'Rupke+05'),
                               ('Sugahara_17_ml.csv', 'd', 'Sugahara+17')]:
        fp = os.path.join('./data/outflows', fname)
        if not os.path.exists(fp):
            continue
        try:
            # these .csv files are whitespace/tab separated despite the suffix
            dd = np.loadtxt(fp, unpack=True)
        except Exception:                                       # noqa: BLE001
            continue
        h = axL.scatter(dd[0], dd[1], marker=marker, s=45, color='k',
                        alpha=0.6, label=lbl, zorder=Z_OBS)
        obs_handles.append(h)

    axL.axvline(FIRE_V_CRIT, color='0.55', ls=':', lw=1.4, zorder=1)
    # Keep this inside axL's y-range: it was pinned at 1.6e3 and disappeared
    # when the limit came down to 500.
    # In-panel annotations take the legend size, not font.size: at the
    # stylesheet default (20) they render as large as the axis labels and
    # dominate the panel.  Still stylesheet-driven, just the smaller of the
    # two values it defines.
    _ann_fs = plt.rcParams['legend.fontsize']
    axL.annotate(rf'$V_{{\rm vir}}={FIRE_V_CRIT:.0f}$ km s$^{{-1}}$',
                 xy=(FIRE_V_CRIT, 260), xytext=(3, 0), fontsize=_ann_fs,
                 textcoords='offset points', color='0.4')
    axR.axvline(v_eject, color='crimson', ls='--', lw=1.4, zorder=1)
    axR.annotate(rf'$\dot{{E}}_{{\rm FB}}=\dot{{E}}_{{\rm lift}}$'
                 '\n' rf'$V_{{\rm vir}}={v_eject:.0f}$ km s$^{{-1}}$',
                 xy=(v_eject, 9.0e2), xytext=(-6, 0), ha='right',
                 fontsize=_ann_fs,
                 textcoords='offset points', color='crimson')

    for a in (axL, axR):
        a.set_xscale('log'); a.set_yscale('log')
        a.set_xlabel(r'$\rm \log_{10}\,V_{vir}\,[km\,s^{-1}]$')
        # Plain values on both axes rather than 10^n, matching the rest of the
        # paper.  The explicit tick lists below control which ones appear.
        a.xaxis.set_major_formatter(_log10_tick_formatter(1))
        a.yaxis.set_major_formatter(_log10_tick_formatter(0))
        a.set_xticks([], minor=True)
        a.set_yticks([], minor=True)
    axL.set_xlim(15, 600); axL.set_ylim(0.05, 500)
    axL.set_xticks([10**e for e in (1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6)])
    axL.set_yticks([10**e for e in (-1, 0, 1, 2)])
    axR.set_xlim(20, 200); axR.set_ylim(1e-2, 1e4)
    # for _v, _c in _sat_marks:
    #     if 20 <= _v <= 200:
    #         axR.axvline(_v, ymin=0.0, ymax=0.05, color=_c, lw=2.4,
    #                     solid_capstyle='butt', zorder=Z_MODEL_LINE)
    axR.set_xticks([10**e for e in (1.4, 1.6, 1.8, 2.0, 2.2)])
    axR.set_yticks([10**e for e in (-2, -1, 0, 1, 2, 3, 4)])
    # Both are mass ratios -- Msun of gas per Msun of stars formed -- so they
    # are dimensionless and carry no unit in brackets.
    # \eta must stay OUTSIDE \rm: mathtext has no Greek glyph in the roman
    # font, so "\rm \eta" renders as a fallback box.  Roman subscript only.
    # \rm goes after the \eta so the Greek stays italic and everything from the
    # "=" onwards is upright, matching axR.
    axL.set_ylabel(r'$\rm \log_{10}\,\dot{m}_{reheat}/\dot{m}_{*}$')
    axR.set_ylabel(r'$\rm \log_{10}\,\dot{m}_{eject}/\dot{m}_{*}$')
    # No hardcoded fontsize anywhere in this figure: sizes come from
    # kieren_cohare_palatino_sty.mplstyle (legend.fontsize 14, font.size 20)
    # so this figure matches every other panel in the paper.
    # Simulation key lives in the ejection panel, lower left, where nothing is
    # drawn.  The left panel already carries the redshift key (upper right) and
    # the observations key (lower left), and a third block there is one too
    # many.  Proxy artists, since every measured curve is drawn label=None.
    if model_files_exist(MINIUCHUU_DIR):
        axR.plot([], [], '-',  color='0.3', lw=2.2, label='Millennium')
        axR.plot([], [], '--', color='0.3', lw=1.8, label='miniUchuu')
    model_handles = [h for h in axL.get_legend_handles_labels()[0]
                     if h not in obs_handles]
    model_labels = [l for h, l in zip(*axL.get_legend_handles_labels())
                    if h not in obs_handles]
    leg1 = _standard_legend(axL, loc='upper right',
                           handles=model_handles, labels=model_labels)
    axL.add_artist(leg1)
    if obs_handles:
        _standard_legend(axL, loc='lower left', handles=obs_handles,
                         labels=[h.get_label() for h in obs_handles])
    _standard_legend(axR, loc='lower left')

    fig.tight_layout()
    outputFile = os.path.join(OUTPUT_DIR, 'MassLoading_vs_Velocity' + OUTPUT_FORMAT)
    save_figure(fig, outputFile)

# Panels at z = 0, 1, 2, 3, 4.  The output table has no snapshot exactly at
# these redshifts, so take the nearest one and label it by the round target.
_MDOT_Z_TARGETS = [0.0, 1.0, 2.0, 3.0, 4.0]

_MDOT_SNAP_PANELS = [
    (_snap_nearest_z(REDSHIFTS, z), f'$z = {z:.0f}$')
    for z in _MDOT_Z_TARGETS
]

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

import glob

import matplotlib.lines as mlines

def _find_model_files(directory):
    files = sorted(glob.glob(os.path.join(directory, 'model_*.hdf5')))
    if not files:
        single = os.path.join(directory, 'model_0.hdf5')
        if os.path.exists(single):
            files = [single]
    return files

def read_snap(directory, snap, properties, min_particles=None):
    """
    Read properties for a single snapshot, concatenated across MPI files.
    Halos with Len < min_particles (default MIN_PARTICLES) are removed.
    """
    if min_particles is None:
        min_particles = MIN_PARTICLES
    # Always load Len for the resolution cut
    load_props = list(properties)
    caller_wants_len = 'Len' in load_props
    if not caller_wants_len:
        load_props.append('Len')

    files = _find_model_files(directory)
    if not files:
        return {}
    snap_key = f'Snap_{snap}'
    chunks = {p: [] for p in load_props}
    found = False
    for fp in files:
        try:
            with h5.File(fp, 'r') as f:
                if snap_key not in f:
                    continue
                found = True
                grp = f[snap_key]
                for p in load_props:
                    if p in grp:
                        chunks[p].append(np.array(grp[p]))
        except Exception as e:
            print(f"  Warning: {fp}: {e}")
    if not found:
        return {}

    # Concatenate all chunks
    data = {}
    for p in load_props:
        if chunks[p]:
            arr = np.concatenate(chunks[p])
            data[p] = arr * MASS_CONVERT if p in _MASS_PROPS else arr

    # Apply resolution cut
    if 'Len' in data:
        mask = data['Len'] >= min_particles
        data = {p: arr[mask] for p, arr in data.items()}

    # Drop Len if the caller didn't ask for it
    if not caller_wants_len:
        data.pop('Len', None)

    return data


def _use_bundled_style():
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.mplstyle', delete=False) as f:
        f.write(_BUNDLED_STYLE)
    plt.style.use(f.name)
    os.remove(f.name)


pp = sys.modules[__name__]       # the bundled paper_plots code, reached as pp.X

# ---- from plotting/random_plotting_scripts/paper1_referee.py ----
import argparse

import os

import sys

import h5py as h5

import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

# Cosmological parameters the SAGE header does not record (sigma_8, n_s, Omega_b).
# Millennium: Springel et al. (2005).  Uchuu: Ishiyama et al. (2021), Planck 2015.
COSMOLOGIES = {
    'Millennium': dict(flat=True, H0=73.0, Om0=0.25, Ob0=0.045, sigma8=0.9, ns=1.0),
    'Planck15':   dict(flat=True, H0=67.74, Om0=0.3089, Ob0=0.0486, sigma8=0.8159, ns=0.9667),
}

# Halo mass definition SAGE's Mvir carries for each tree type: LHaloTree-style
# readers store M_200c in Mvir; consistent-trees stores the BN98 virial mass.
TREE_MDEF = {0: '200c', 1: '200c', 2: '200c', 3: 'vir', 4: 'vir', 5: '200c'}

HMF_MODEL = 'despali16'   # valid for any mass definition

COSMO_Z = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]

N_LEVELS = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]     # Mpc^-3, cumulative

MASS_BIN = 0.1

COL_MILL = 'black'

COL_UCHUU = 'steelblue'

COL_SAGE16 = 'purple'

RECIPE_COLOURS = {'BR06': 'black', 'GD14': 'goldenrod', 'KD12': 'steelblue',
                  'KMT09': 'limegreen', 'K13': 'firebrick'}

class Run:
    """One SAGE output directory with the unit and cosmology bookkeeping it needs."""

    def __init__(self, path, label, color, ls='-', cosmology=None):
        hdr = pp._read_sim_header(path)
        if hdr is None:
            raise FileNotFoundError(path)
        self.path, self.label, self.color, self.ls = path, label, color, ls
        self.files = pp.find_model_files(path)
        self.h = hdr['hubble_h']
        self.box = hdr['box_size']
        self.volume = (self.box / self.h)**3 * hdr['volume_fraction']
        self.mass_convert = hdr['unit_mass_in_g'] / pp._MSUN_CGS / self.h
        self.part_mass = hdr['particle_mass'] * self.mass_convert
        self.omega_m, self.omega_l = hdr['omega_matter'], hdr['omega_lambda']
        self.baryon_frac = hdr['baryon_frac']
        self.redshifts = np.array(hdr['redshifts'])
        self.snaps = np.array(sorted(hdr['output_snaps']))
        with h5.File(self.files[0], 'r') as f:
            rt = f['Header/Runtime'].attrs
            self.tree_type = int(rt['TreeType'])
            self.z0 = float(rt.get('Reionization_z0', 8.0))
            self.zr = float(rt.get('Reionization_zr', 7.0))
        self.mdef = TREE_MDEF.get(self.tree_type, 'vir')
        if cosmology is None:
            cosmology = 'Millennium' if abs(self.omega_m - 0.25) < 0.01 else 'Planck15'
        self.cosmology = cosmology

    def snap_near(self, z):
        """Output snapshot nearest redshift z, and its redshift."""
        zs = self.redshifts[self.snaps]
        i = int(np.argmin(np.abs(zs - z)))
        return int(self.snaps[i]), float(zs[i])

    def snaps_bracketing(self, z):
        """The two output snapshots either side of z (or the nearest one twice)."""
        zs = self.redshifts[self.snaps]
        order = np.argsort(zs)
        zs, sn = zs[order], self.snaps[order]
        if z <= zs[0]:
            return (int(sn[0]), float(zs[0])), (int(sn[0]), float(zs[0]))
        if z >= zs[-1]:
            return (int(sn[-1]), float(zs[-1])), (int(sn[-1]), float(zs[-1]))
        j = int(np.searchsorted(zs, z))
        return (int(sn[j - 1]), float(zs[j - 1])), (int(sn[j]), float(zs[j]))

    def load(self, snap, props):
        return pp.read_snap_from_files(self.files, f'Snap_{snap}', props,
                                       mass_convert=self.mass_convert)

    def describe(self):
        return (f'{self.label}: {self.path}\n'
                f'    box {self.box:.1f} Mpc/h, h = {self.h:.4f}, V = {self.volume:.3e} Mpc^3, '
                f'm_p = {self.part_mass:.3e} Msun, Om = {self.omega_m:.4f}, f_b = {self.baryon_frac:.3f}, '
                f'TreeType {self.tree_type} -> Mvir = M_{self.mdef}, cosmology {self.cosmology}')

def try_run(path, label, color, ls='-'):
    try:
        return Run(path, label, color, ls)
    except (FileNotFoundError, KeyError, IndexError) as e:
        print(f'  [skip] {label}: no usable output in {path} ({e})')
        return None

def save(fig, out, name):
    for ext in ('.png', '.pdf'):
        fig.savefig(os.path.join(out, name + ext), dpi=110 if ext == '.png' else None,
                    bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {os.path.join(out, name)}.png/.pdf')

_COLOSSUS_ADDED = set()

def analytic_hmf(cosmology, z, log_m, mdef):
    """dn/dlog10 M [Mpc^-3 dex^-1] at masses 10**log_m [Msun] (no h)."""
    from colossus.cosmology import cosmology as cc
    from colossus.lss import mass_function
    if cosmology not in _COLOSSUS_ADDED:
        cc.addCosmology(cosmology, **COSMOLOGIES[cosmology])
        _COLOSSUS_ADDED.add(cosmology)
    cc.setCosmology(cosmology)
    h = COSMOLOGIES[cosmology]['H0'] / 100.0
    m_h = 10**np.asarray(log_m) * h
    dndlnm = mass_function.massFunction(m_h, z, mdef=mdef, model=HMF_MODEL, q_out='dndlnM')
    return dndlnm * h**3 * np.log(10.0)

def analytic_cumulative(cosmology, z, log_m, mdef):
    """n(>M) [Mpc^-3] on the grid log_m, integrating to log M = 16."""
    grid = np.arange(min(log_m), 16.0, 0.01)
    dn = analytic_hmf(cosmology, z, grid, mdef)
    cum = np.cumsum((dn * 0.01)[::-1])[::-1]
    return np.interp(log_m, grid, cum)

def mass_function_poisson(log_m, volume, lo=8.0, hi=16.0, dx=MASS_BIN):
    """log10 phi and its Poisson 1-sigma bounds; NaN where empty."""
    edges = np.arange(lo, hi + dx / 2, dx)
    n, _ = np.histogram(log_m, bins=edges)
    x = 0.5 * (edges[1:] + edges[:-1])
    with np.errstate(divide='ignore', invalid='ignore'):
        phi = np.where(n > 0, np.log10(n / volume / dx), np.nan)
        lo_ = np.where(n > 1, np.log10((n - np.sqrt(n)) / volume / dx), np.nan)
        hi_ = np.where(n > 0, np.log10((n + np.sqrt(n)) / volume / dx), np.nan)
    return x, phi, lo_, hi_, n

def mass_at_density(log_m, volume, levels=N_LEVELS):
    """log10 mass at which the cumulative number density n(>M) reaches each level."""
    s = np.sort(log_m)[::-1]
    out = []
    for n in levels:
        k = int(round(n * volume))
        out.append(s[k - 1] if 1 <= k <= len(s) else np.nan)
    return np.array(out)

def ref_binned_median(x, y, edges, min_count=10):
    mid = 0.5 * (edges[1:] + edges[:-1])
    med, p16, p84 = (np.full(len(mid), np.nan) for _ in range(3))
    idx = np.digitize(x, edges) - 1
    for i in range(len(mid)):
        w = idx == i
        if w.sum() >= min_count:
            med[i], p16[i], p84[i] = np.percentile(y[w], [50, 16, 84])
    return mid, med, p16, p84

def interp_in_z(run, z, func):
    """func(snap) evaluated at redshift z by linear interpolation between snapshots."""
    (s1, z1), (s2, z2) = run.snaps_bracketing(z)
    a = func(s1)
    if s1 == s2:
        return a
    b = func(s2)
    t = (z - z1) / (z2 - z1)
    return (1 - t) * a + t * b

def _central_props(run, snap):
    d = run.load(snap, ['Mvir', 'StellarMass', 'Type', 'FFBRegime'])
    if not d:
        return None
    cen = d['Type'] == 0
    return {k: v[cen] for k, v in d.items()}

def task_cosmo(runs, out):
    mill, uchuu = runs['mill'], runs['uchuu']
    if mill is None or uchuu is None:
        print('  cosmo needs both simulations')
        return

    print('\n=== Analytic expectation (Despali+16) ===')
    print('dn/dlogM ratio Millennium / Planck15, both in M_vir; then with each run\'s own Mvir definition')
    lm = np.array([10.0, 10.5, 11.0, 11.5, 12.0, 12.5])
    for z in COSMO_Z:
        r_vir = analytic_hmf('Millennium', z, lm, 'vir') / analytic_hmf('Planck15', z, lm, 'vir')
        r_sim = analytic_hmf('Millennium', z, lm, mill.mdef) / analytic_hmf('Planck15', z, lm, uchuu.mdef)
        print(f'  z = {z:4.1f}  ' + '  '.join(f'{m:.1f}: {a:5.2f} ({b:5.2f})' for m, a, b in zip(lm, r_vir, r_sim)))

    print('\nlog M_vir at fixed n(>M), analytic, Millennium / Planck15 (M_vir):')
    grid = np.arange(8.0, 15.5, 0.01)
    for z in COSMO_Z:
        cm = analytic_cumulative('Millennium', z, grid, 'vir')
        cu = analytic_cumulative('Planck15', z, grid, 'vir')
        row = []
        for n in N_LEVELS:
            a = np.interp(-np.log10(n), -np.log10(cm), grid) if cm.min() < n < cm.max() else np.nan
            b = np.interp(-np.log10(n), -np.log10(cu), grid) if cu.min() < n < cu.max() else np.nan
            row.append(f'{n:.0e}: {a:5.2f}/{b:5.2f}')
        print(f'  z = {z:4.1f}  ' + '  '.join(row))

    # ---------- measured, per redshift ----------
    zs = COSMO_Z
    fig, axes = plt.subplots(4, len(zs), figsize=(4.2 * len(zs), 15), sharex='row',
                             gridspec_kw={'height_ratios': [3, 1.4, 2.2, 2.2]})
    print('\n=== Measured (centrals, Len >= 20) ===')
    fixed_n_rows = []
    for j, z in enumerate(zs):
        ax_hmf, ax_rat, ax_shmr, ax_n = axes[:, j]
        snap_m, z_m = mill.snap_near(z)
        snap_u, z_u = uchuu.snap_near(z)
        dm, du = _central_props(mill, snap_m), _central_props(uchuu, snap_u)
        print(f'\n z target {z}: Millennium snap {snap_m} (z = {z_m:.2f}), '
              f'Uchuu snap {snap_u} (z = {z_u:.2f})')
        hmf = {}
        for run, d, zz in ((mill, dm, z_m), (uchuu, du, z_u)):
            if d is None or len(d['Mvir']) == 0:
                continue
            x, phi, plo, phi_hi, n = mass_function_poisson(np.log10(d['Mvir']), run.volume)
            hmf[run.label] = (x, phi, n)
            ok = np.isfinite(phi)
            ax_hmf.plot(x[ok], phi[ok], color=run.color, ls=run.ls, lw=2.5, label=run.label)
            okb = np.isfinite(plo) & np.isfinite(phi_hi)
            ax_hmf.fill_between(x[okb], plo[okb], phi_hi[okb], color=run.color, alpha=0.2, lw=0)
            xa = np.arange(9.0, 15.0, 0.05)
            ax_hmf.plot(xa, np.log10(analytic_hmf(run.cosmology, zz, xa, run.mdef)),
                        color=run.color, ls=':', lw=1.5,
                        label=f'{run.cosmology} ({HMF_MODEL}, $M_{{\\rm {run.mdef}}}$)')
            for npart, lsv in ((20, '--'), (100, '-.')):
                ax_hmf.axvline(np.log10(npart * run.part_mass), color=run.color, ls=lsv, lw=0.8, alpha=0.6)
            # measured / analytic
            ana = np.log10(analytic_hmf(run.cosmology, zz, x, run.mdef))
            sel = ok & (n >= 10) & (x > np.log10(100 * run.part_mass))
            if sel.any():
                print(f'   {run.label:22s} measured - analytic (n >= 10, > 100 particles): '
                      f'median {np.median(phi[sel] - ana[sel]):+.2f} dex over '
                      f'log M = {x[sel].min():.1f}-{x[sel].max():.1f}')
        if z >= 4 and z <= 12:
            lffb = np.log10(pp.ffb_threshold_mass_msun(z))
            for ax in (ax_hmf, ax_rat, ax_shmr):
                ax.axvline(lffb, color='grey', ls=':', lw=1.2)
        ax_hmf.set_title(rf'$z \approx {z:g}$  (Mill {z_m:.2f}, Uchuu {z_u:.2f})', fontsize=14)
        ax_hmf.set_ylim(-7, 0)
        ax_hmf.set_xlim(9.5, 15.0)

        # ratio panel: Uchuu interpolated to the Millennium redshift, and only
        # above 100 particles of the coarser run
        def _hmf_phi(run):
            def f(snap):
                d = _central_props(run, snap)
                _, phi, _, _, n = mass_function_poisson(np.log10(d['Mvir']), run.volume)
                return np.where(n >= 5, phi, np.nan)
            return f
        x = mass_function_poisson(np.array([]), 1.0)[0]
        pm = _hmf_phi(mill)(snap_m)
        pu = interp_in_z(uchuu, z_m, _hmf_phi(uchuu))
        m_res = np.log10(100 * max(mill.part_mass, uchuu.part_mass))
        ok = np.isfinite(pm) & np.isfinite(pu) & (x > m_res)
        ax_rat.plot(x[ok], pm[ok] - pu[ok], color='k', lw=2, label='measured')
        for lm_ in (11.0, 11.5, 12.0, 12.5):
            i = np.argmin(np.abs(x - lm_))
            if ok[i]:
                print(f'   measured log(phi_Mill / phi_Uchuu) at log M = {lm_} (z = {z_m:.2f}): '
                      f'{pm[i] - pu[i]:+.2f} dex')
        xa = np.arange(9.5, 15.0, 0.05)
        ax_rat.plot(xa, np.log10(analytic_hmf('Millennium', z_m, xa, mill.mdef) /
                                 analytic_hmf('Planck15', z_m, xa, uchuu.mdef)),
                    color='firebrick', ls='--', lw=1.5, label='analytic, as simulated')
        ax_rat.plot(xa, np.log10(analytic_hmf('Millennium', z, xa, 'vir') /
                                 analytic_hmf('Planck15', z, xa, 'vir')),
                    color='grey', ls=':', lw=1.5, label=r'analytic, same $z$ and $M_{\rm vir}$')
        ax_rat.axhline(0, color='k', lw=0.5)
        ax_rat.set_ylim(-0.5, 1.5)

        # SHMR panel, Uchuu interpolated to the Millennium redshift
        edges = np.arange(9.5, 15.01, 0.25)

        def _shmr(run):
            def f(snap):
                d = _central_props(run, snap)
                w = d['StellarMass'] > 0
                return np.array(ref_binned_median(np.log10(d['Mvir'][w]),
                                              np.log10(d['StellarMass'][w] / d['Mvir'][w]), edges)[1:])
            return f
        shmr = {}
        for run, d in ((mill, dm), (uchuu, du)):
            if d is None:
                continue
            med, p16, p84 = _shmr(run)(snap_m) if run is mill else interp_in_z(uchuu, z_m, _shmr(uchuu))
            mid = 0.5 * (edges[1:] + edges[:-1])
            shmr[run.label] = (mid, med)
            ok = np.isfinite(med)
            ax_shmr.plot(mid[ok], med[ok], color=run.color, ls=run.ls, lw=2.5)
            ax_shmr.fill_between(mid[ok], p16[ok], p84[ok], color=run.color, alpha=0.2, lw=0)
            ffb = d['FFBRegime'] == 1
            nffb = int(ffb.sum())
            print(f'   {run.label:22s} FFB centrals: {nffb:7d}  n = {nffb / run.volume:.2e} Mpc^-3')
        if len(shmr) == 2:
            mid, a = shmr[mill.label]
            _, b = shmr[uchuu.label]
            for lm_ in (10.5, 11.0, 11.5, 12.0):
                i = np.argmin(np.abs(mid - lm_))
                if np.isfinite(a[i]) and np.isfinite(b[i]):
                    print(f'   SHMR median log(m*/Mvir) at log Mvir = {mid[i]:.2f}: '
                          f'Mill {a[i]:+.2f}, Uchuu {b[i]:+.2f}, diff {a[i] - b[i]:+.2f} dex')
        ax_shmr.set_ylim(-4.5, -0.5)

        # fixed-number-density panel, Uchuu interpolated to the Millennium redshift
        def _mass_levels(run, field):
            def f(snap):
                d = _central_props(run, snap)
                if d is None:
                    return np.full(len(N_LEVELS), np.nan)
                v = d[field][d[field] > 0]
                return mass_at_density(np.log10(v), run.volume)
            return f
        mv_m = _mass_levels(mill, 'Mvir')(snap_m)
        ms_m = _mass_levels(mill, 'StellarMass')(snap_m)
        mv_u = interp_in_z(uchuu, z_m, _mass_levels(uchuu, 'Mvir'))
        ms_u = interp_in_z(uchuu, z_m, _mass_levels(uchuu, 'StellarMass'))
        ln = np.log10(N_LEVELS)
        ax_n.plot(ln, mv_m - mv_u, 'o-', color='k', label=r'$\Delta\log M_{\rm vir}$')
        ax_n.plot(ln, ms_m - ms_u, 's--', color='darkorange', label=r'$\Delta\log m_*$')
        cm = analytic_cumulative('Millennium', z_m, grid, mill.mdef)
        cu = analytic_cumulative('Planck15', z_m, grid, uchuu.mdef)
        ana = []
        for n in N_LEVELS:
            a = np.interp(-np.log10(n), -np.log10(cm), grid) if cm.min() < n < cm.max() else np.nan
            b = np.interp(-np.log10(n), -np.log10(cu), grid) if cu.min() < n < cu.max() else np.nan
            ana.append(a - b)
        ax_n.plot(ln, ana, ':', color='firebrick', label=r'analytic $\Delta\log M$')
        ax_n.axhline(0, color='k', lw=0.5)
        ax_n.set_ylim(-0.6, 0.8)
        print(f'   at fixed n(>M), Mill - Uchuu (Uchuu interpolated to z = {z_m:.2f}):')
        for k, n in enumerate(N_LEVELS):
            print(f'     n = {n:.0e}: dlogMvir {mv_m[k] - mv_u[k]:+.2f} (analytic {ana[k]:+.2f}), '
                  f'dlog m* {ms_m[k] - ms_u[k]:+.2f}')
        fixed_n_rows.append((z_m, mv_m - mv_u, ms_m - ms_u, np.array(ana)))

        if j == 0:
            ax_hmf.set_ylabel(r'$\log_{10}\phi_{\rm halo}\ [\mathrm{Mpc^{-3}\,dex^{-1}}]$')
            ax_rat.set_ylabel(r'$\log_{10}\,\phi_{\rm Mill}/\phi_{\rm Uchuu}$')
            ax_shmr.set_ylabel(r'$\log_{10}\, m_*/M_{\rm vir}$')
            ax_n.set_ylabel(r'Mill $-$ Uchuu [dex]')
            ax_hmf.legend(loc='lower left', fontsize=9, frameon=False)
            ax_rat.legend(loc='upper left', fontsize=9, frameon=False)
            ax_n.legend(loc='upper left', fontsize=9, frameon=False)
        ax_shmr.set_xlabel(r'$\log_{10} M_{\rm vir}\ [M_\odot]$')
        ax_rat.set_xlabel(r'$\log_{10} M_{\rm vir}\ [M_\odot]$')
        ax_n.set_xlabel(r'$\log_{10}\, n(>M)\ [\mathrm{Mpc^{-3}}]$')
    fig.tight_layout()
    save(fig, out, 'cosmo_hmf_shmr')

    _cosmo_ffb_and_smf(mill, uchuu, out)

def _cosmo_ffb_and_smf(mill, uchuu, out):
    """FFB abundance against redshift, and the SMF ratio at z >= 4."""
    fig, (ax_n, ax_f, ax_s) = plt.subplots(1, 3, figsize=(18, 5.5))
    print('\n=== FFB centrals against redshift ===')
    for run in (mill, uchuu):
        zs_all = run.redshifts[run.snaps]
        sel = run.snaps[(zs_all >= 4.0) & (zs_all <= 14.5)]
        zz, n_meas, n_ana, f_mass = [], [], [], []
        m_res = np.log10(20 * run.part_mass)
        for snap in sorted(sel):
            z = float(run.redshifts[snap])
            d = run.load(snap, ['Mvir', 'StellarMass', 'Type', 'FFBRegime'])
            if not d:
                continue
            cen = d['Type'] == 0
            ffb = cen & (d['FFBRegime'] == 1)
            tot = d['StellarMass'].sum()
            xa = np.arange(m_res, 15.0, 0.01)
            dn = analytic_hmf(run.cosmology, z, xa, run.mdef)
            n_expect = np.sum(dn * pp.ffb_fraction(10**xa, z) * 0.01)
            zz.append(z)
            n_meas.append(ffb.sum() / run.volume)
            n_ana.append(n_expect)
            f_mass.append(d['StellarMass'][ffb].sum() / tot if tot > 0 else np.nan)
        zz, n_meas, n_ana, f_mass = map(np.array, (zz, n_meas, n_ana, f_mass))
        with np.errstate(divide='ignore'):
            ax_n.plot(zz, np.log10(n_meas), 'o-', color=run.color, label=f'{run.label} (measured)')
        ax_n.plot(zz, np.log10(n_ana), ':', color=run.color, label=f'{run.cosmology} HMF $\\times f_{{\\rm FFB}}$')
        ax_f.plot(zz, f_mass, 'o-', color=run.color, label=run.label)
        print(f'  {run.label}:')
        for a, b, c, e in zip(zz, n_meas, n_ana, f_mass):
            print(f'    z = {a:5.2f}: n_FFB measured {b:.2e}, analytic {c:.2e}, '
                  f'ratio {b / c if c > 0 else np.nan:.2f}; FFB share of rho_* {e:.2f}')
    ax_n.set_xlabel('Redshift'); ax_n.set_ylabel(r'$\log_{10}\, n_{\rm FFB}\ [\mathrm{Mpc^{-3}}]$')
    ax_f.set_xlabel('Redshift'); ax_f.set_ylabel(r'FFB centrals share of $\rho_*$')
    ax_n.legend(fontsize=9, frameon=False); ax_f.legend(fontsize=9, frameon=False)

    print('\n=== SMF ratio Millennium / Uchuu at fixed m* (all galaxies) ===')
    cols = plt.get_cmap('viridis')(np.linspace(0, 0.9, 5))
    for c, z in zip(cols, (4.0, 6.0, 8.0, 10.0, 12.0)):
        snap_m, z_m = mill.snap_near(z)
        def _smf(run):
            def f(snap):
                d = run.load(snap, ['StellarMass'])
                m = d['StellarMass'][d['StellarMass'] > 0]
                x, phi, _, _, n = mass_function_poisson(np.log10(m), run.volume, lo=7.0, hi=12.5)
                return np.where(n >= 5, phi, np.nan)
            return f
        pm = _smf(mill)(snap_m)
        pu = interp_in_z(uchuu, z_m, _smf(uchuu))
        x = mass_function_poisson(np.array([]), 1.0, lo=7.0, hi=12.5)[0]
        ok = np.isfinite(pm) & np.isfinite(pu)
        ax_s.plot(x[ok], pm[ok] - pu[ok], color=c, lw=2, label=rf'$z = {z_m:.2f}$')
        if ok.any():
            print(f'  z = {z_m:.2f}: ' + ', '.join(f'{xx:.1f}: {d:+.2f}' for xx, d in
                                                 zip(x[ok][::3], (pm[ok] - pu[ok])[::3])))
    ax_s.axhline(0, color='k', lw=0.5)
    ax_s.set_xlabel(r'$\log_{10} m_*\ [M_\odot]$')
    ax_s.set_ylabel(r'$\log_{10}\,\phi_{\rm Mill}/\phi_{\rm Uchuu}$ (SMF)')
    ax_s.set_ylim(-1, 1.5)
    ax_s.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    save(fig, out, 'cosmo_ffb_smf')

def _gas_offsets(name, x, phi, obs_list, masses=(9.5, 10.0, 10.5)):
    """Model - observation offset at the observed points nearest each mass."""
    ok = np.isfinite(phi)
    for obs in obs_list:
        mo, po = np.asarray(obs['mass']), np.asarray(obs['phi'])
        mod = np.interp(mo, x[ok], phi[ok], left=np.nan, right=np.nan)
        parts = []
        for m in masses:
            i = np.argmin(np.abs(mo - m))
            if abs(mo[i] - m) < 0.15 and np.isfinite(mod[i]):
                parts.append(f'{mo[i]:.2f}: {mod[i] - po[i]:+.2f}')
        above = (mo >= 9.5) & np.isfinite(mod) & np.isfinite(po)
        mean_above = np.mean(mod[above] - po[above]) if above.any() else np.nan
        print(f'    {name:16s} vs {obs["label"]:28s} ' + '  '.join(parts) +
              f'   | mean above 9.5: {mean_above:+.2f}')

def task_gas(runs, out):
    mill = runs['mill']
    if mill is None:
        print('  gas needs the Millennium run')
        return
    recipes = [('BR06', mill)] + [(k, r) for k, r in runs['recipes'].items() if r is not None]
    vanilla = runs['vanilla']

    obs_hi, obs_h2 = pp.load_himf_observations(), pp.load_h2mf_observations()
    panels = [('ColdGas', r'm_{\rm cold}', obs_hi + obs_h2, True),
              ('H1gas', r'm_{\rm HI}', obs_hi, True),
              ('H2gas', r'm_{\rm H_2}', obs_h2, False)]

    # Layout, labels and legend sizes follow plot_40 in paper_plots.py: the gas
    # panels share one x axis with no gaps, and the m* panel sits below.
    fig = plt.figure(figsize=(7, 17.5))
    gs = fig.add_gridspec(5, 1, height_ratios=[1, 1, 1, 0.32, 1.05], hspace=0.0)
    axes = [fig.add_subplot(gs[0])]
    axes += [fig.add_subplot(gs[i], sharex=axes[0]) for i in (1, 2)]
    axes.append(fig.add_subplot(gs[4]))
    ylabel = r'$\log_{10}\ \phi\ [\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1}]$'
    obs_kw = dict(markerfacecolor='gray', markeredgewidth=1.0, ms=6, lw=1.0, capsize=2,
                  alpha=0.8, ls='none', zorder=5)
    snap_m, _ = mill.snap_near(0.0)
    print(f'\n=== Gas mass functions, z = 0 (snap {snap_m}) ===')
    for centrals_only in (True, False):
        print(f'\n  {"centrals only" if centrals_only else "all galaxies"}: '
              'model - obs [dex] at log m_gas')
        for ax, (field, tag, obs_list, show16) in zip(axes, panels):
            print(f'   {field}:')
            entries = list(recipes) + ([('SAGE16', vanilla)] if (show16 and vanilla) else [])
            for name, run in entries:
                d = run.load(run.snap_near(0.0)[0], [field, 'Type'])
                g = d[field]
                w = (g > 1e8) & ((d['Type'] == 0) if centrals_only else True)
                x, phi, plo, phi_hi, _ = pp.mass_function_bootstrap(np.log10(g[w]), run.volume,
                                                                    binwidth=0.2, n_boot=50)
                _gas_offsets(name, x, phi, obs_list)
                if not centrals_only:
                    continue
                ok = np.isfinite(phi) & np.isfinite(plo) & np.isfinite(phi_hi)
                if name == 'SAGE16':
                    col, lw, ls, z, lab = COL_SAGE16, 3.0, '--', 11, 'SAGE16'
                else:
                    col, ls = RECIPE_COLOURS[name], '-'
                    lw, z = (3.5, 12) if name == 'BR06' else (2.0, 9)
                    lab = 'BR06 (default)' if name == 'BR06' else name
                ax.fill_between(x[ok], plo[ok], phi_hi[ok], color=col, alpha=0.2, lw=0, zorder=z - 1)
                ax.plot(x[ok], phi[ok], color=col, lw=lw, ls=ls, label=lab, zorder=z)
            if centrals_only:
                for obs in obs_list:
                    m = np.isfinite(obs['phi']) & (obs['mass'] >= 8.0)
                    ax.errorbar(obs['mass'][m], obs['phi'][m], yerr=pp._gasmf_obs_yerr(obs, m),
                                fmt=obs['marker'], color=obs['color'],
                                markeredgecolor=obs.get('edgecolor', 'k'), label=obs['label'], **obs_kw)
                ax.text(0.04, 0.05, f'${tag}$', transform=ax.transAxes, fontsize=17)
                ax.set_ylim(-5.5, -0.5)
                ax.set_ylabel(ylabel)
                ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
                ax.yaxis.set_minor_locator(plt.MultipleLocator(0.2))

    # helium and phases: what ColdGas holds besides HI + H2
    d = mill.load(snap_m, ['ColdGas', 'H1gas', 'H2gas', 'Type'])
    w = (d['ColdGas'] > 1e9) & (d['Type'] == 0)
    if w.any():
        f = (d['H1gas'][w] + d['H2gas'][w]) / d['ColdGas'][w]
        print(f'\n  (HI + H2) / ColdGas for centrals with m_cold > 1e9: median {np.median(f):.3f}, '
              f'16-84: {np.percentile(f, 16):.3f}-{np.percentile(f, 84):.3f}')

    # SMF panel: total (solid) and quiescent (dashed) for each recipe
    ax = axes[3]
    print('\n=== z = 0 SMF by H2 recipe (all galaxies) ===')
    base = {}
    for name, run in list(recipes) + ([('SAGE16', vanilla)] if vanilla else []):
        d = run.load(run.snap_near(0.0)[0], ['StellarMass', 'SfrDisk', 'SfrBulge'])
        m = d['StellarMass']
        w = m > 1e8
        lm = np.log10(m[w])
        ssfr = pp.log_ssfr(d['SfrDisk'][w], d['SfrBulge'][w], m[w])
        x, phi, _ = pp.mass_function(lm, run.volume, 0.1, (8.0, 12.5))
        _, phq, _ = pp.mass_function(lm[ssfr < pp.SSFR_CUT], run.volume, 0.1, (8.0, 12.5))
        col = COL_SAGE16 if name == 'SAGE16' else RECIPE_COLOURS[name]
        lw = 3.5 if name == 'BR06' else 2
        ax.plot(x, phi, color=col, lw=lw, ls='-' if name != 'SAGE16' else '-.')
        ax.plot(x, phq, color=col, lw=lw * 0.7, ls='--' if name != 'SAGE16' else ':')
        base[name] = (x, phi, phq)
        qf = []
        for lm0 in (8.5, 9.0, 10.0, 11.0):
            sel = (lm > lm0 - 0.25) & (lm < lm0 + 0.25)
            qf.append(f'{lm0}: {np.mean(ssfr[sel] < pp.SSFR_CUT):.2f}' if sel.sum() > 20 else f'{lm0}: -')
        print(f'  {name:7s} quiescent fraction ' + ', '.join(qf))
    if 'BR06' in base:
        xb, pb, qb = base['BR06']
        for name, (x, phi, phq) in base.items():
            if name == 'BR06':
                continue
            parts = []
            for lm0 in (8.5, 9.5, 10.5, 11.0, 11.5):
                i = np.argmin(np.abs(x - lm0))
                parts.append(f'{lm0}: tot {phi[i] - pb[i]:+.2f} q {phq[i] - qb[i]:+.2f}')
            print(f'  {name:7s} - BR06 [dex]  ' + '  '.join(parts))
    try:
        baldry = pp.load_baldry_blue_red()
        ax.errorbar(baldry['q_mass'], baldry['q_phi'], fmt='o', color='k', markeredgecolor='k',
                    label='Baldry+12 (red)', **obs_kw)
        bm, bp, beh, bel = pp.load_bell_smf_q_data()
        if bm is not None:
            ax.errorbar(bm, bp, yerr=[bel, beh], fmt='^', color='k', markeredgecolor='k',
                        label='Bell+03 (red)', **obs_kw)
        tot = [o for o in pp._load_smf_grid_observations() if o['label'].startswith('Baldry+08')]
        for o in tot[:1]:
            ax.errorbar(o['log_mass'], o['log_phi'], fmt='s', color='k', markeredgecolor='k',
                        label='Baldry+08 (all)', **dict(obs_kw, markerfacecolor='white'))
    except Exception as e:
        print(f'  SMF observations unavailable: {e}')
    ax.text(0.04, 0.05, r'$m_*$', transform=ax.transAxes, fontsize=17)
    ax.text(0.13, 0.055, '(solid: all, dashed: quiescent)', transform=ax.transAxes, fontsize=10)
    ax.set_xlim(8.0, 12.0); ax.set_ylim(-5.5, -0.5)
    ax.set_xlabel(r'$\log_{10}\ m_{*}\ [M_{\odot}]$')
    ax.set_ylabel(ylabel)
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_locator(plt.MultipleLocator(1.0))
        a.set_minor_locator(plt.MultipleLocator(0.2))
    ax.legend(loc='upper right', fontsize=10, frameon=False)

    axes[2].set_xlim(8.0, 11.0)
    axes[2].set_xlabel(r'$\log_{10}\ m_{\mathrm{gas}}\ [M_{\odot}]$')
    axes[2].xaxis.set_major_locator(plt.MultipleLocator(1.0))
    axes[2].xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    for a in axes[:2]:
        a.tick_params(labelbottom=False)

    # Top panel: the one model legend, titled, and SAGE16.  Its observations are
    # the HI and H2 sets, identified in the two panels below.  HI and H2 panels:
    # observations only.
    for k, a in enumerate(axes[:3]):
        h, l = a.get_legend_handles_labels()
        obs_l = {o['label'] for o in panels[k][2]}
        if k == 0:
            mh = [hh for hh, ll in zip(h, l) if ll not in obs_l and ll != 'SAGE16']
            ml = [ll for ll in l if ll not in obs_l and ll != 'SAGE16']
            leg = a.legend(mh, ml, loc='lower left', bbox_to_anchor=(0.0, 0.12), fontsize=10,
                           frameon=False, title='SAGE26 using:')
            leg.get_title().set_fontsize(12)
            leg._legend_box.align = 'left'
            a.add_artist(leg)
            sh = [hh for hh, ll in zip(h, l) if ll == 'SAGE16']
            a.legend(sh, ['SAGE16'], loc='upper right', fontsize=10, frameon=False)
        else:
            oh = [hh for hh, ll in zip(h, l) if ll in obs_l]
            ol = [ll for ll in l if ll in obs_l]
            a.legend(oh, ol, loc='upper right', fontsize=10, frameon=False)
    fig.tight_layout()
    save(fig, out, 'gas_fig4')

def _sfrd_curve(run, out, zmax=13.0):
    # Reading every snapshot of a full box is slow, so keep the curve beside the
    # figures and reuse it until the model files change.
    cache = os.path.join(out, 'sfrd_cache_' + os.path.basename(os.path.normpath(run.path)) + '.npz')
    stamp = max(os.path.getmtime(f) for f in run.files)
    if os.path.exists(cache):
        c = np.load(cache)
        if str(c['path']) == run.path and float(c['stamp']) == stamp:
            print(f'  {run.label}: SFRD from cache {cache}')
            return c['z'], c['rho']
    zs, rho = [], []
    for snap in run.snaps:
        z = float(run.redshifts[snap])
        if z > zmax:
            continue
        d = run.load(snap, ['SfrDisk', 'SfrBulge'])
        if not d:
            continue
        s = (d['SfrDisk'] + d['SfrBulge']).sum() / run.volume
        if s > 0:
            zs.append(z); rho.append(np.log10(s))
    o = np.argsort(zs)
    z, rho = np.array(zs)[o], np.array(rho)[o]
    np.savez(cache, z=z, rho=rho, path=run.path, stamp=stamp)
    return z, rho

def task_sfrd(runs, out):
    models = [(k, runs[k]) for k in ('mill', 'uchuu', 'vanilla') if runs[k] is not None]
    # Size, colours and labels follow Figure 2 in paper_plots.py.
    paper_style = {'mill': ('steelblue', '-'), 'uchuu': ('steelblue', '--'), 'vanilla': ('purple', '--')}
    fig, ax = plt.subplots()
    curves, model_lines = {}, []
    for key, run in models:
        z, rho = _sfrd_curve(run, out)
        curves[run.label] = (z, rho)
        color, ls = paper_style[key]
        model_lines += ax.plot(z, rho, color=color, ls=ls, lw=3.5, label=run.label, zorder=10)

    obs_style = dict(markerfacecolor='gray', markeredgecolor='k', markeredgewidth=1.0,
                     ecolor='k', color='k', ms=8, lw=1.0, alpha=0.6, ls='none', zorder=5)
    zf = np.linspace(0, 8, 200)
    md = 0.015 * (1 + zf)**2.7 / (1 + ((1 + zf) / 2.9)**5.6)
    ax.plot(zf, np.log10(md) + pp.imf_shift('Madau+Dickinson 14'), color='gray', lw=1.5,
            alpha=0.7, label=pp._tex_safe(r'Madau \& Dickinson 2014'))
    obs = {}
    dr = pp.load_driver_sfrd_2018_data()
    if dr[0] is not None:
        ax.errorbar(dr[0], dr[1], yerr=dr[2], xerr=[dr[3], dr[4]], fmt='o', label='Driver+18', **obs_style)
        obs['Driver+18'] = (dr[0], dr[1])
    ds = pp.load_dsilva_sfrd_2023_data()
    if ds[0] is not None:
        ax.errorbar(ds[0], ds[1], yerr=[ds[2], ds[3]], xerr=[ds[4], ds[5]], fmt='D',
                    label="D'Silva+23", **obs_style)
        obs["D'Silva+23"] = (ds[0], ds[1])
    for dataset, fmt in (('Novak+17', '^'), ('Bouwens+15', '<'), ('Finkelstein+15', '>')):
        z, r, eu, el = pp.load_sfrd_table(dataset)
        if z is not None:
            ax.errorbar(z, r, yerr=[el, eu], fmt=fmt, label=dataset, **obs_style)
            obs[dataset] = (z, r)
    for loader, label, fmt in ((pp.load_oesch_sfrd_2018_data, 'Oesch+18', 'h'),
                               (pp.load_harikane_sfr_density_2023_data, 'Harikane+23', 'v'),
                               (pp.load_mcleod_rho_sfr_2024_data, 'McLeod+24', 'p')):
        z, r, ep, em = loader()
        if z is None:
            continue
        z, r = np.asarray(z, float), np.asarray(r, float)
        ax.errorbar(z, r, yerr=[np.asarray(em, float), np.asarray(ep, float)], fmt=fmt, label=label, **obs_style)
        obs[label] = (z, r)
    try:
        from astropy.table import Table
        t = Table.read('./data/sfrd/CSFRD_inferred_from_SMD.ecsv', format='ascii.ecsv')
        zc = np.array(t['Redshift'])
        s50, s16, s84 = (np.log10(np.array(t[k])) for k in ('sfrd_50', 'sfrd_16', 'sfrd_84'))
        st = max(1, int(round(0.25 / np.median(np.diff(zc)))))
        sl = slice(None, None, st)
        ax.errorbar(zc[sl], s50[sl], yerr=[(s50 - s16)[sl], (s84 - s50)[sl]], fmt='s',
                    label='COSMOS-Web', **obs_style)
        obs['COSMOS-Web'] = (zc, s50)
    except Exception as e:
        print(f'  COSMOS-Web unavailable: {e}')

    # Same limits and ticks as Figure 2 in paper_plots.py.
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.set_xlim(0.0, 7.5); ax.set_ylim(-3.0, -0.5)
    ax.set_xlabel(r'$\mathrm{Redshift}$')
    ax.set_ylabel(r'$\log_{10}\ {\rho_{\rm SFR}}\ (M_{\odot}\ \mathrm{yr}^{-1}\ \mathrm{Mpc}^{-3})$')
    # Models top right, observations bottom left.
    obs_handles = [h for h, l in zip(*ax.get_legend_handles_labels()) if h not in model_lines]
    ax.add_artist(pp._standard_legend(ax, loc='upper right', handles=model_lines,
                                      labels=[h.get_label() for h in model_lines]))
    pp._standard_legend(ax, loc='lower left', handles=obs_handles,
                        labels=[h.get_label() for h in obs_handles], ncol=2)
    fig.tight_layout()
    save(fig, out, 'sfrd_fig2')

    print('\n=== SFRD: model values and model - observation ===')
    for label, (z, rho) in curves.items():
        print(f'  {label}: ' + ', '.join(f'z={zz:.1f}: {np.interp(zz, z, rho):+.2f}'
                                         for zz in (0, 1, 2, 4, 6, 8, 10) if z.min() <= zz <= z.max()))
        for olab, (zo, ro) in obs.items():
            inside = (zo >= z.min()) & (zo <= z.max())
            if inside.any():
                dlt = np.interp(zo[inside], z, rho) - ro[inside]
                print(f'      vs {olab:12s} median {np.median(dlt):+.2f} dex over z = '
                      f'{zo[inside].min():.1f}-{zo[inside].max():.1f}')
    if runs['mill'] is not None and runs['vanilla'] is not None:
        zm, rm = curves[runs['mill'].label]
        zv, rv = curves[runs['vanilla'].label]
        for zz in (4, 6, 8):
            print(f'  SAGE16 - SAGE26 at z = {zz}: {np.interp(zz, zv, rv) - np.interp(zz, zm, rm):+.2f} dex')
    pp.report_imf_audit()

Z_BINS = [(0.0, 0.5), (0.5, 0.8), (0.8, 1.1), (1.1, 1.5), (1.5, 2.0), (2.0, 2.5),
          (2.5, 3.0), (3.0, 3.5), (3.5, 4.5), (4.5, 5.5), (5.5, 6.5), (6.5, 7.5),
          (7.5, 8.5), (8.5, 9.5), (9.5, 12.0)]

MASS_RANGES = [(8.0, 9.5), (9.5, 10.5), (10.5, 12.5)]

def task_smf(runs, out):
    models = [r for r in (runs['mill'], runs['uchuu'], runs['vanilla']) if r is not None]
    all_obs = pp._load_smf_grid_observations()
    fig, axes = plt.subplots(5, 3, figsize=(15, 20), sharex=True, sharey=True)
    axes = axes.flatten()
    print('\n=== SMF model - observation [dex], median per stellar-mass range ===')
    header = '  bin          model                ' + '  '.join(f'{a:.1f}-{b:.1f}' for a, b in MASS_RANGES) + '   |med abs|'
    print(header)
    for i, (zlo, zhi) in enumerate(Z_BINS):
        ax = axes[i]
        in_bin = [o for o in all_obs if (zlo <= o['z'] <= zhi if i == len(Z_BINS) - 1 else zlo <= o['z'] < zhi)]
        mo = np.concatenate([np.asarray(o['log_mass'], float) for o in in_bin]) if in_bin else np.array([])
        po = np.concatenate([np.asarray(o['log_phi'], float) for o in in_bin]) if in_bin else np.array([])
        for run in models:
            snap, zs = pp._select_grid_snapshot(run.redshifts, int(run.snaps.min()), int(run.snaps.max()), zlo, zhi)
            if snap is None or len(mo) == 0:
                continue
            d = run.load(snap, ['StellarMass'])
            m = d['StellarMass'][d['StellarMass'] > 0]
            x, phi, _ = pp.mass_function(np.log10(m), run.volume, 0.1, (7.0, 12.5))
            ok = np.isfinite(phi)
            mod = np.interp(mo, x[ok], phi[ok], left=np.nan, right=np.nan)
            dl = mod - po
            good = np.isfinite(dl)
            ax.plot(mo[good], dl[good], 'o', color=run.color, ms=4, alpha=0.6, label=run.label if i == 0 else None)
            parts = []
            for a, b in MASS_RANGES:
                s = good & (mo >= a) & (mo < b)
                parts.append(f'{np.median(dl[s]):+6.2f}' if s.sum() >= 2 else '     -')
            mad = np.median(np.abs(dl[good])) if good.any() else np.nan
            print(f'  {zlo:4.1f}-{zhi:<4.1f} {run.label:22s}' + '    '.join(parts) + f'    {mad:5.2f}')
        ax.axhline(0, color='k', lw=0.8)
        ax.axhspan(-0.3, 0.3, color='grey', alpha=0.12, lw=0)
        ax.text(0.95, 0.93, rf'${zlo:.1f}<z<{zhi:.1f}$', transform=ax.transAxes, ha='right', va='top')
        ax.set_ylim(-1.5, 1.5); ax.set_xlim(8.0, 12.2)
    axes[0].legend(loc='lower left', fontsize=10, frameon=False)
    for k in (12, 13, 14):
        axes[k].set_xlabel(r'$\log_{10} m_*\ [M_\odot]$')
    for k in (0, 3, 6, 9, 12):
        axes[k].set_ylabel('model $-$ obs [dex]')
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.02, wspace=0.02)
    save(fig, out, 'smf_offsets')

def filtering_mass_code(z, run):
    """Port of do_reionization(): the mass used, in 1e10 Msun/h (code units)."""
    alpha, tvir = 6.0, 1e4
    a0, ar = 1.0 / (1 + run.z0), 1.0 / (1 + run.zr)
    a = 1.0 / (1 + z)
    x0, xr = a / a0, a / ar
    if a <= a0:
        f = 3 * a / ((2 + alpha) * (5 + 2 * alpha)) * x0**alpha
    elif a < ar:
        f = (3 / a) * a0**2 * (1 / (2 + alpha) - 2 * x0**-0.5 / (5 + 2 * alpha)) + a * a / 10 - (a0**2 / 10) * (5 - 4 * x0**-0.5)
    else:
        f = (3 / a) * (a0**2 * (1 / (2 + alpha) - 2 * x0**-0.5 / (5 + 2 * alpha)) + (ar**2 / 10) * (5 - 4 * xr**-0.5)
                       - (a0**2 / 10) * (5 - 4 / np.sqrt(x0)) + a * ar / 3 - (ar**2 / 3) * (3 - 2 / np.sqrt(xr)))
    mfilt = 25.0 / np.sqrt(run.omega_m) * 2.21 * f**1.5
    g_code = 6.674e-8 * 1.989e43 / 3.08568e24 / 1e10        # (km/s)^2 Mpc / (1e10 Msun)
    vchar = np.sqrt(tvir / 36.0)
    om_z = run.omega_m * (1 + z)**3 / (run.omega_m * (1 + z)**3 + run.omega_l)
    xz = om_z - 1
    dcrit = 18 * np.pi**2 + 82 * xz - 39 * xz**2
    hz = 100.0 * np.sqrt(run.omega_m * (1 + z)**3 + run.omega_l)
    mchar = vchar**3 / (g_code * hz * np.sqrt(0.5 * dcrit))
    return max(mfilt, mchar), mfilt, mchar

def task_reion(runs, out):
    sims = [r for r in (runs['mill'], runs['uchuu']) if r is not None]
    print('\n=== Filtering mass used by do_reionization() (half the universal f_b at Mvir = M_F) ===')
    for run in sims:
        print(f'  {run.label} (z0 = {run.z0}, zr = {run.zr}, Om = {run.omega_m}):')
        for z in (0, 1, 2, 4, 6, 7, 8, 10):
            used, mf, mc = filtering_mass_code(z, run)
            print(f'    z = {z:2d}: M_F = {used * 1e10:.2e} Msun/h = 10^{np.log10(used * 1e10 / run.h):.2f} Msun'
                  f'   ({"Kravtsov filtering" if mf >= mc else "T_vir = 1e4 K"})')
        print(f'    20 particles = {20 * run.part_mass * run.h:.2e} Msun/h, 100 particles = {100 * run.part_mass * run.h:.2e} Msun/h')

    zs = (0.0, 2.0, 4.0, 6.0)
    fig, axes = plt.subplots(1, len(zs), figsize=(5 * len(zs), 5), sharey=True)
    props = ['Mvir', 'StellarMass', 'ColdGas', 'HotGas', 'CGMgas', 'EjectedMass',
             'BlackHoleMass', 'IntraClusterStars', 'Type', 'CentralGalaxyIndex', 'GalaxyIndex']
    print('\n=== Halo baryon fraction / f_b, median per Mvir bin (all galaxies in the halo summed) ===')
    for ax, z in zip(axes, zs):
        lm = np.linspace(8.5, 13, 200)
        for run in sims:
            snap, zz = run.snap_near(z)
            mused = filtering_mass_code(zz, run)[0] * 1e10 / run.h
            ax.plot(lm, 1.0 / (1 + 0.26 * mused / 10**lm)**3, color=run.color, ls=':', lw=1.5)
            d = run.load(snap, props)
            if not d:
                continue
            bary = sum(d[k] for k in props[1:8])
            cen = d['Type'] == 0
            # sum each halo's baryons onto its central
            ucen, inv = np.unique(d['CentralGalaxyIndex'], return_inverse=True)
            tot = np.bincount(inv, weights=bary)
            gi = d['GalaxyIndex'][cen]
            pos = np.searchsorted(ucen, gi)
            okc = (pos < len(ucen)) & (ucen[np.minimum(pos, len(ucen) - 1)] == gi)
            mvir = d['Mvir'][cen][okc]
            fb = tot[pos[okc]] / (run.baryon_frac * mvir)
            edges = np.arange(9.0, 13.01, 0.2)
            mid, med, p16, p84 = ref_binned_median(np.log10(mvir), fb, edges)
            ok = np.isfinite(med)
            ax.plot(mid[ok], med[ok], color=run.color, ls=run.ls, lw=2.5, label=run.label)
            ax.fill_between(mid[ok], p16[ok], p84[ok], color=run.color, alpha=0.2, lw=0)
            for npart, lsv in ((20, '--'), (100, '-.')):
                ax.axvline(np.log10(npart * run.part_mass), color=run.color, ls=lsv, lw=0.8)
            print(f'  {run.label} z = {zz:.2f}: ' + ', '.join(f'{a:.1f}: {b:.2f}' for a, b in zip(mid[ok][::2], med[ok][::2])))
        ax.axhline(1, color='k', lw=0.5)
        ax.set_title(rf'$z \approx {z:g}$', fontsize=14)
        ax.set_xlabel(r'$\log_{10} M_{\rm vir}\ [M_\odot]$')
        ax.set_xlim(8.5, 13); ax.set_ylim(0, 1.3)
    axes[0].set_ylabel(r'$m_{\rm baryon}/(f_b M_{\rm vir})$')
    axes[0].legend(loc='lower right', fontsize=10, frameon=False)
    fig.tight_layout()
    save(fig, out, 'reion_baryon_fraction')

TASKS = {'cosmo': task_cosmo, 'gas': task_gas, 'sfrd': task_sfrd, 'smf': task_smf, 'reion': task_reion}

ref = pp                         # the bundled paper1_referee code, reached as ref.X
# ==================== END SHARED CODE ====================


# ========================== PARALLEL COLUMN CACHE ==========================

def _read_file(task):
    """Worker: raw columns *props* of group *key* in one file, or None if the group is absent."""
    path, key, props = task
    with h5.File(path, 'r') as f:
        if key not in f:
            return None
        g = f[key]
        return {p: g[p][()] for p in props if p in g}


class ColumnCache:
    """(files, snapshot, column) -> concatenated raw array, read once, in parallel over files."""

    def __init__(self, workers):
        self.workers, self.pool = workers, None
        self.raw, self.found, self.view = {}, {}, {}
        self.n_read = 0

    def _map(self, tasks):
        if self.workers <= 1 or len(tasks) <= 1:
            return [_read_file(t) for t in tasks]
        if self.pool is None:
            import multiprocessing as mp
            ctx = mp.get_context('fork' if sys.platform.startswith('linux') else 'spawn')
            self.pool = ctx.Pool(self.workers)
        return self.pool.map(_read_file, tasks, chunksize=1)

    def columns(self, files, key, props, mass_convert, min_len, mass_props):
        files = tuple(files)
        want = list(dict.fromkeys(list(props) + ['Len']))
        missing = [p for p in want if (files, key, p) not in self.raw]
        if missing:
            res = self._map([(fp, key, missing) for fp in files])
            self.found[(files, key)] = self.found.get((files, key), False) or any(r is not None for r in res)
            for p in missing:
                ch = [r[p] for r in res if r is not None and p in r]
                self.raw[(files, key, p)] = np.concatenate(ch) if ch else None
                self.n_read += 1
        if not self.found.get((files, key)):
            return {}
        out = {}
        for p in want:
            vk = (files, key, p, mass_convert, min_len)
            if vk not in self.view:
                a = self.raw[(files, key, p)]
                if a is None:
                    continue
                ln = self.raw[(files, key, 'Len')]
                if ln is not None and min_len > 0:
                    a = a[ln >= min_len]
                if p in mass_props:
                    a = a * mass_convert
                a.flags.writeable = False          # shared between figures: never modified in place
                self.view[vk] = a
            out[p] = self.view[vk]
        if 'Len' not in props:
            out.pop('Len', None)
        return out

    def trim(self, keep_key):
        """Free every snapshot except *keep_key* (the z = 0 catalogues the next figures share)."""
        for d in (self.raw, self.view, self.found):
            for k in [k for k in d if k[1] != keep_key]:
                del d[k]

    def fields(self, files, key):
        with h5.File(files[0], 'r') as f:
            return list(f[key].keys()) if key in f else []

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()


class LazySnap(Mapping):
    """A snapshot catalogue whose columns are read from the cache when first used."""

    def __init__(self, pp, cache, directory, snap):
        self.pp, self.cache = pp, cache
        self.files = pp.find_model_files(directory)
        self.key = f'Snap_{snap}'
        self._fields = None

    def _f(self):
        if self._fields is None:
            self._fields = [p for p in self.cache.fields(self.files, self.key) if p != 'Len'] if self.files else []
        return self._fields

    def __getitem__(self, p):
        d = self.pp.read_snap_from_files(self.files, self.key, [p])
        if p not in d:
            raise KeyError(p)
        return d[p]

    def __contains__(self, p):
        return p in self._f()

    def __iter__(self):
        return iter(self._f())

    def __len__(self):
        return len(self._f())


class LazySnapdata(Mapping):
    """{snap: LazySnap} over the snapshots present in a run."""

    def __init__(self, pp, cache, directory):
        self.pp, self.cache, self.directory = pp, cache, directory
        files = pp.find_model_files(directory)
        with h5.File(files[0], 'r') as f:
            self.snaps = sorted(int(k.split('_')[1]) for k in f if k.startswith('Snap_'))

    def __getitem__(self, s):
        if s not in self.snaps:
            raise KeyError(s)
        return LazySnap(self.pp, self.cache, self.directory, s)

    def __contains__(self, s):
        return s in self.snaps

    def __iter__(self):
        return iter(self.snaps)

    def __len__(self):
        return len(self.snaps)


# ========================== FAST STATISTICS ==========================

def install_fast_paths(pp, cache):
    """Route paper_plots' readers through the cache and swap in the count-based bootstraps."""
    orig_read_snap = pp.read_snap

    def read_snap_from_files(filepaths, snap_key, properties, mass_convert=None):
        mc = pp.MASS_CONVERT if mass_convert is None else mass_convert
        return dict(cache.columns(filepaths, snap_key, properties, mc, pp.MIN_PARTICLES, pp._MASS_PROPS))

    def read_snap(directory, snap, properties, min_particles=None):
        mp_ = pp.MIN_PARTICLES if min_particles is None else min_particles
        files = pp._find_model_files(directory)
        if not files:
            return {}
        return dict(cache.columns(files, f'Snap_{snap}', properties, pp.MASS_CONVERT, mp_, pp._MASS_PROPS))

    def mass_function_bootstrap(log_masses, volume, binwidth=0.1, mass_range=None, n_boot=100):
        centers, phi, mrange = pp.mass_function(log_masses, volume, binwidth, mass_range)
        n = len(log_masses)
        if n == 0:
            return centers, phi, phi, phi, mrange
        mi, ma = mrange
        counts, _ = np.histogram(log_masses, range=(mi, ma), bins=int(round((ma - mi) / binwidth)))
        p = np.append(counts, n - counts.sum()) / n            # last bucket: outside the range
        boots = np.random.multinomial(n, p, size=n_boot)[:, :-1].astype(float)
        with np.errstate(divide='ignore'):
            bp = np.log10(boots / volume / binwidth)
        bp[~np.isfinite(bp)] = np.nan
        with np.errstate(invalid='ignore'):
            return centers, phi, np.nanpercentile(bp, 16, axis=0), np.nanpercentile(bp, 84, axis=0), mrange

    class _CI:
        def __init__(self, lo, hi):
            self.low, self.high = lo, hi

    class _Res:
        def __init__(self, lo, hi):
            self.confidence_interval = _CI(lo, hi)

    class StatsProxy:
        """scipy.stats, except that bootstrapped means of large samples use the normal limit."""

        def __getattr__(self, k):
            return getattr(_scipy_stats, k)

        def bootstrap(self, data, statistic, confidence_level=0.95, **kw):
            x = np.asarray(data[0], dtype=float)
            if statistic is np.mean and x.size > pp.BOOT_EXACT_MAX:
                from scipy.stats import norm
                half = norm.ppf(0.5 + confidence_level / 2) * x.std() / np.sqrt(x.size)
                return _Res(x.mean() - half, x.mean() + half)
            return _scipy_stats.bootstrap(data, statistic, confidence_level=confidence_level, **kw)

    from scipy import stats as _scipy_stats
    pp.read_snap_from_files = read_snap_from_files
    pp.read_snap = read_snap
    pp.mass_function_bootstrap = mass_function_bootstrap
    pp.stats = StatsProxy()
    return orig_read_snap


# ========================== OUTPUT ==========================

class _Tee:
    def __init__(self, path):
        self.file, self.stdout = open(path, 'w'), sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        sys.stdout = self.stdout
        self.file.close()


# ========================== MAIN ==========================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--box', choices=('mini', 'big'), default='mini',
                    help="run directories: 'mini' (output/millennium*, as paper_plots.py) or 'big' "
                         "(output/Millennium_*, miniUchuu_FFB)")
    ap.add_argument('--figs', nargs='*', default=ALL_FIGS, choices=ALL_FIGS)
    ap.add_argument('-j', '--workers', type=int,
                    default=min(4, os.cpu_count() or 1),
                    help='worker processes (default 4: the head-node allowance)')
    ap.add_argument('--out', default=None, help='output directory (default <primary run>/paper1/)')
    args = ap.parse_args()
    out = os.path.join(_CWD, args.out) if args.out else os.path.abspath(os.path.join(pp.PRIMARY_DIR, 'paper1'))
    os.makedirs(out, exist_ok=True)
    pp.OUTPUT_DIR = out + os.sep
    tee = _Tee(os.path.join(out, 'paper1_numbers.txt'))
    sys.stdout = tee

    t0 = time.time()
    cache = ColumnCache(args.workers)
    install_fast_paths(pp, cache)
    pp.setup_style()
    np.random.seed(pp.SEED)
    pp.seed(pp.SEED)
    print(f'Box: {args.box}   primary: {pp.PRIMARY_DIR}   output: {out}   workers: {args.workers}')
    pp.report_imf_audit()

    primary = LazySnap(pp, cache, pp.PRIMARY_DIR, pp.SNAP_Z0)
    vanilla = LazySnap(pp, cache, pp.VANILLA_DIR, pp.SNAP_Z0) if pp.model_files_exist(pp.VANILLA_DIR) else {}

    if '4' in args.figs or 'R' in args.figs:
        runs = {
            'mill':    ref.try_run(pp.PRIMARY_DIR, 'SAGE26 (Millennium)', ref.COL_MILL, '-'),
            'uchuu':   ref.try_run(pp.MINIUCHUU_DIR, 'SAGE26 (miniUchuu)', ref.COL_UCHUU, '--'),
            'vanilla': ref.try_run(pp.VANILLA_DIR, 'SAGE16', ref.COL_SAGE16, '--'),
            'recipes': {k: ref.try_run(d, k, ref.RECIPE_COLOURS[k]) for k, d in
                        (('GD14', pp.GD14_DIR), ('KD12', pp.KD12_DIR), ('KMT09', pp.KMT09_DIR),
                         ('K13', pp.K13_DIR))},
        }

    jobs = {
        '2':  ('Figure 2', lambda: pp.plot_16_sfrd_history()),
        '3':  ('Figure 3', lambda: pp.plot_3_gas_metallicity_vs_stellar_mass(primary, vanilla)),
        '4':  ('Figure 4', lambda: ref.task_gas(runs, out)),
        '5':  ('Figure 5', lambda: pp.plot_15_sfr_vs_stellar_mass(primary, vanilla)),
        '6':  ('Figure 6', lambda: pp.plot_24_mass_loading_vs_velocity(primary, vanilla)),
        '7':  ('Figure 7', lambda: pp.plot_5_stellar_halo_mass(primary, vanilla)),
        '8':  ('Figure 8', lambda: pp.plot_1_stellar_mass_function_ssfr_combined(primary, vanilla)),
        '9':  ('Figure 9', lambda: pp.plot_4_bh_bulge_mass(primary, vanilla)),
        '10': ('Figure 10', lambda: pp.plot_6b_bulge_mass_size_median(primary, vanilla)),
        '11': ('Figure 11', lambda: pp.plot_13_ffb_vs_redshift(LazySnapdata(pp, cache, pp.PRIMARY_DIR))),
        '12': ('Figure 12', lambda: pp.plot_18_smf_redshift_grid()),
        'R':  ('Referee letter', lambda: [ref.TASKS[t](runs, out) for t in ('cosmo', 'reion', 'smf', 'sfrd')]),
    }
    try:
        for f in args.figs:
            title, job = jobs[f]
            t = time.time()
            print(f'\n######## {title} ########')
            job()
            cache.trim(f'Snap_{pp.SNAP_Z0}')
            print(f'  [{title}: {time.time() - t:.1f} s]')
        print(f'\nTotal {time.time() - t0:.1f} s; {cache.n_read} columns read')
    finally:
        cache.close()
        tee.close()
    print(f'Numbers written to {os.path.join(out, "paper1_numbers.txt")}')


if __name__ == '__main__':
    main()

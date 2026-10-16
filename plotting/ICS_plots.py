#!/usr/bin/env python
"""
SAGE26 -- the merger clock and the intracluster stars
=====================================================

One parameter decides where a destroyed satellite's stars end up.  When the
disruption gate in ``core_build_model.c`` fires, the only test applied is the
sign of the satellite's dynamical-friction clock:

    MergTime >  0   ->  disrupt_satellite_to_ICS()      stars become intracluster
    MergTime <= 0   ->  deal_with_galaxy_merger()       stars land on the central

and that clock is set once, at infall, by ``estimate_merging_time``:

    T_df = alpha * 1.17 * R_vir^2 * V_vir / (ln(1 + M_host/M_sat) * G * M_sat)

with ``alpha = MergerTimeFactor``.  The published value is alpha = 2.0, which
makes T_df far longer than a satellite actually survives, so the clock is almost
never expired when the gate fires and nearly all accreted stellar mass is routed
to the ICS.  Shortening alpha moves mass back onto the central.

This module is the paper: four runs identical in every respect except alpha,

    alpha = 0.5, 1.0, 2.0

and five figures across them.

    1  f_ICS = m_ICS / (all stars in the halo), against halo mass and redshift.
       Each run is drawn twice: the SB-free total SAGE tracks, and the same
       clusters re-measured through a surface-brightness cut, so the model is
       compared with the observations on the observations' own terms
       (see MODEL_SB_RECOVERY)
    2  three columns, each with a residual strip: (a) bulge-to-total against
       stellar mass vs Moffett+16, (b) cosmic SFR density vs the compilation,
       (c) the clock -- t_clock = beta_clock t_friction handed out at infall --
       vs beta_clock = 2
    4  the ICS mass function in a grid of redshifts, split by host halo mass,
       beta_clock as line style, residual against beta_clock = 2
    (figures 3 and 5 were folded into figure 2)
    6  figure 1 as raw data: every halo, through the cut, no medians hiding the
       scatter and no subsampling hiding the sample size

and six more that put SAGE26's clusters through the analysis of Kimmig et al.
(2025, A&A 700, A95), "Intra-cluster light as a dynamical clock":

    7  assembly: z_form, halo growth M(z)/M(0), and f_ICL+BCG(z), split by z_form
       and by f_ICL+BCG (their figs 4 and 5)
    8  f_ICL+BCG against halo mass and z_form, f_sub against z_form, and the
       stellar-to-halo mass relation coloured by z_form (their figs 3, 6, 7, 10)
    9  f_ICL+BCG against M12, M14, the stellar centre shift and phi_BCG (fig 8)
   10  the correlation matrix of every tracer (fig 9)
   11  the stellar budget of the main progenitor over time -- BCG, ICS, second
       most massive galaxy, other satellites (fig 11)
   12  the shredding rate: change in f_ICL+BCG over ~1 Gyr against fractional
       halo growth (fig 12)

and two more for the paper itself:

   13  ICS assembly by channel -- in situ (disrupted here) and ex situ (arrived
       pre-processed) -- against lookback time, per halo-mass bin (after
       random_plotting_scripts/ics_formation_assembly_grid.py)
   14  the stellar mass function

and three that put SAGE26's clusters through the hindcast of Pearce et al.
(2026, arXiv:2610.06660), "Disentangling the assembly of the central and diffuse
stellar components in galaxy clusters" (see THE PEARCE ET AL. COMPARISON):

   18  one cluster's infall events: mass ratio against lookback time, coloured
       by each event's share of the final BCG, with the Boylan-Kolchin+08 and
       SAGE26 merger times (their fig 2)
   19  the same for every cluster without a central swap (their fig 4)
   20  how few events build the BCG: the mean cumulative share of the k largest
       events, and the distribution of N50 (their figs 3 and 5, section 3)

   22  their fig 1 for the figure 18 cluster, one figure per run: the BCG, the
       ICS made in the cluster, the satellites and the ICS that arrived
       pre-processed, fixed at z = 0 and traced back to z ~ 0.3 and z ~ 1

and one that tests a rule of their kind in place of the clock:

   21  where, not when: destroyed satellites re-sorted by r < f R_vir of their
       target at the last output, against the share of destroyed mass sent to
       the ICS -- f_ICS in groups and clusters, and the BCG's in situ share and
       four largest events, beside the beta_clock runs
       (see THE POSITIONAL RULE TEST)

Every run also prints the diagnostics the paper is written from (report_*):
the global stellar budget, groups and clusters, BCG shifts with alpha, ICS
assembly channels and deposit times, redshift evolution, the massive end of
the SMF, B/T, SFRD, the clock broken down by satellite mass and redshift, the
ICS mass function and the Kimmig+25 comparison.

Figures 1-4 read the z = 0 galaxy catalogues; figure 5 reads the per-event
disruption log written by the ``SAGE_DISRUPT_LOG`` diagnostic in
``core_build_model.c``, which records MergTime at the moment the gate fires
together with the infall snapshot, so the clock's starting value is recoverable.
Figures 7-12 follow every z = 0 cluster's main branch back through the
snapshots (see THE KIMMIG ET AL. COMPARISON below).

Usage:
    python plotting/ICS_plots.py                 # all figures
    python plotting/ICS_plots.py 1 5             # figures 1 and 5 only
    python plotting/ICS_plots.py 7 8 9 10 11 12  # the Kimmig+25 comparison
    python plotting/ICS_plots.py 18 19 20 22     # the Pearce+26 comparison
    python plotting/ICS_plots.py 21              # the positional rule test
    python plotting/ICS_plots.py 19 20 21 --pearce=top    # stack their 107 (or range, all)
    python plotting/ICS_plots.py --refresh       # rebuild the cached scans

Parallelism: ICS_SCAN_JOBS (default 8) sets the snapshot scan's workers, each
holding one snapshot of every file; ICS_FILE_JOBS (default ICS_SCAN_JOBS) sets
the file-by-file scans' workers.  On miniUchuu, ICS_SCAN_JOBS=8 ICS_FILE_JOBS=24.
"""

import os
import sys
import glob
import time
from concurrent.futures import ProcessPoolExecutor

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.integrate import quad

import warnings
warnings.filterwarnings("ignore")

try:
    from astropy.table import Table
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False


# ========================== CONFIGURATION ==========================

# The alpha sweep.  Every run is microUchuu with input/microuchuu.par physics;
# only MergerTimeFactor differs.  The colours are sampled from plasma (0.78,
# 0.45, 0.05), darkest for the published value, stopping short of plasma's
# pale-yellow end, which vanishes on white.  Ordering in beta_clock stays
# legible without the legend and survives greyscale.
# alpha = 0.25 was dropped from the sweep: it sits below the observed f_ICS in
# both groups and clusters however the surface-brightness cut is set, so it
# rules itself out rather than bounding anything.  The run is still on disk at
# ./output/microuchuu_mtf_0.25/ if it is ever wanted back.
RUNS = (
    (0.5,  './output/microuchuu_mtf_0.5/',  '#fb9f3a'),
    (1.0,  './output/microuchuu_mtf_1.0/',  '#bf3984'),
    (2.0,  './output/microuchuu/',  '#2a0593'),
)
FIDUCIAL = 2.0                      # the published value, drawn heaviest

# Figure 6 separates group- from cluster-scale hosts by both colour and marker.
# Clusters keep the blue ramp the rest of the module uses for alpha; groups get
# a matched orange ramp, so shade still reads as alpha within either class.
GROUP_COLOURS = ('#fdd0a2', '#fdae6b', '#e6550d', '#a63603')


def group_colour(rank, n):
    """The orange ramp sliced from its dark end, so the largest alpha is darkest."""
    return GROUP_COLOURS[max(0, len(GROUP_COLOURS) - n) + rank]
# A star reads smaller than a filled shape of the same nominal size, so the
# cluster marker is set larger to keep the two at similar visual weight.
SCALE_STYLE = {
    'group':   dict(marker='h', ms=5.5, label='groups'),
    'cluster': dict(marker='*', ms=11.0, label='clusters'),
}

OUT_DIR = './output/plots/mergetime/'
CACHE_DIR = os.path.join(OUT_DIR, 'cache')
OBS_DIR = './data/'
OUTPUT_FORMAT = '.pdf'

# Cache version.  Bump when the contents of a scan change so a stale file is
# rebuilt instead of silently loaded with missing columns.
CACHE_VERSION = 'v9'

# Worker processes for the snapshot scan.  Snapshots are read independently, so
# this mostly buys concurrent I/O; each worker holds one snapshot of every file
# in memory at once, so lower it (ICS_SCAN_JOBS=1 is the old serial loop) on a
# node short of RAM.
SCAN_JOBS = int(os.environ.get('ICS_SCAN_JOBS', min(8, os.cpu_count() or 1)))
# Worker processes for the scans that go file by file (ICS assembly, Kimmig,
# Pearce, positional rules).  Each of these workers holds only one file -- at
# most two of its snapshots at a time plus what it accumulates -- so far more of
# them fit in memory than snapshot-scan workers, which each hold one snapshot of
# EVERY file.  Raise this one (e.g. ICS_FILE_JOBS=24) and leave ICS_SCAN_JOBS
# modest: 24 snapshot workers on miniUchuu run a node out of memory and hang.
FILE_JOBS = int(os.environ.get('ICS_FILE_JOBS', SCAN_JOBS))

_MSUN_CGS = 1.989e33
_GYR_S = 3.15576e16

MIN_COUNT = 10                      # objects needed in a bin before it is drawn

# Host-mass slices.  Groups and clusters are the two regimes the observations
# separate, and the two the ICS behaves differently in.
GROUP_LO, GROUP_HI = 1e13, 1e14
CLUSTER_LO = 1e14

# Shared binning.
MVIR_BINS  = np.arange(11.0, 15.01, 0.25)
MSTAR_BINS = np.arange(8.0, 12.51, 0.25)
MF_BINWIDTH = 0.2

# The ICS mass function grid (figure 4), as plotting/random_plotting_scripts/
# ICS_paper_plots.py draws it: centrals with ICS > 0, split by host halo mass.
MF_ICS_LO, MF_ICS_HI, MF_ICS_BW = 4.5, 12.75, 0.25
MF_ICS_EDGES = np.arange(MF_ICS_LO, MF_ICS_HI + 0.5 * MF_ICS_BW, MF_ICS_BW)
MF_TARGET_Z = (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
# The low-redshift column (figure 16): the same grid where most of the ICS forms.
MF_TARGET_Z_LOW = (0.0, 0.5, 1.0, 2.0)
MF_HALO_BINS = (10.0, 12.0, 14.0, 17.0)                  # log10 Mvir [Msun]
MF_HALO_COLOURS = ('firebrick', 'green', 'slateblue')
MF_HALO_LABELS = (r'$10^{10} < M_{\rm vir} < 10^{12}\,{\rm M}_\odot$',
                  r'$10^{12} < M_{\rm vir} < 10^{14}\,{\rm M}_\odot$',
                  r'$M_{\rm vir} > 10^{14}\,{\rm M}_\odot$')

# Draw order.
Z_BAND, Z_OBS, Z_LINE = 2, 5, 10

# Figure 6 draws one marker per halo, every halo, no subsampling -- half a
# million points per run.  The cloud is rasterised at this dpi so the PDF stays
# openable while the axes, lines and text around it stay vector.
SCATTER_RASTER_DPI = 450

# Which edge of the observational-offset band figure 6's scatter is drawn
# through: 'low' is the larger (more negative) offset, the one most favourable
# to the model.
SCATTER_G = 'low'

# Salpeter -> Chabrier for the Madau & Dickinson (2014) SFRD: the factor 0.63
# they quote for SFRs (-0.20 dex).  plotting/paper_plots.py uses -0.24, a
# stellar-mass factor; it does not apply to SFR densities.  The model is Chabrier.
SALPETER_TO_CHABRIER_DEX = np.log10(0.63)


# ========================== RUN HEADERS ==========================

def find_model_files(directory):
    """All model_*.hdf5 files in *directory*, sorted.  Empty list if none."""
    files = sorted(glob.glob(os.path.join(directory, 'model_*.hdf5')))
    # model.hdf5 (no rank suffix) is a concatenation of the per-rank files and
    # would double-count every galaxy if it were picked up alongside them.
    return [f for f in files if os.path.basename(f) != 'model.hdf5']


def read_sim_header(directory):
    """Simulation parameters from the HDF5 header.  None if no model files."""
    files = find_model_files(directory)
    if not files:
        return None

    with h5.File(files[0], 'r') as f:
        sim, runtime = f['Header/Simulation'], f['Header/Runtime']
        hdr = {
            'hubble_h':      float(sim.attrs['hubble_h']),
            'box_size':      float(sim.attrs['box_size']),
            'omega_matter':  float(sim.attrs['omega_matter']),
            'omega_lambda':  float(sim.attrs['omega_lambda']),
            'last_snap_nr':  int(sim.attrs['LastSnapshotNr']),
            'unit_mass_in_g': float(runtime.attrs['UnitMass_in_g']),
            'unit_length_in_cm': float(runtime.attrs['UnitLength_in_cm']),
            'unit_velocity_in_cm_per_s': float(runtime.attrs['UnitVelocity_in_cm_per_s']),
            'redshifts':     np.array(f['Header/snapshot_redshifts'][:]),
            'output_snaps':  sorted(int(s) for s in f['Header/output_snapshots'][:]),
            'merger_time_factor':    float(runtime.attrs.get('MergerTimeFactor', np.nan)),
            'thresh_sat_disruption': float(runtime.attrs.get('ThresholdSatDisruption', np.nan)),
        }

    total_fvp = 0.0
    for fp in files:
        with h5.File(fp, 'r') as f:
            total_fvp += float(f['Header/Runtime'].attrs['frac_volume_processed'])
    hdr['volume_fraction'] = total_fvp
    hdr['files'] = files
    hdr['directory'] = directory
    hdr['mass_convert'] = hdr['unit_mass_in_g'] / _MSUN_CGS / hdr['hubble_h']
    # SAGE's code time unit in Gyr: length / velocity, with the 1/h of the length unit
    hdr['time_convert_gyr'] = (hdr['unit_length_in_cm'] / hdr['unit_velocity_in_cm_per_s']
                               / _GYR_S / hdr['hubble_h'])
    hdr['volume'] = (hdr['box_size'] / hdr['hubble_h']) ** 3 * total_fvp   # Mpc^3
    return hdr


def available_runs():
    """The (alpha, header, colour) triples whose output actually exists."""
    out = []
    for alpha, directory, colour in RUNS:
        hdr = read_sim_header(directory)
        if hdr is None:
            print(f'  missing: {directory} (alpha = {alpha:g}) -- skipped')
            continue
        got = hdr['merger_time_factor']
        if np.isfinite(got) and not np.isclose(got, alpha):
            print(f'  warning: {directory} has MergerTimeFactor = {got:g}, '
                  f'expected {alpha:g}')
        out.append((alpha, hdr, colour))
    return out


ALL_RUNS = available_runs()
if not ALL_RUNS:
    sys.exit('No runs found.  Expected the alpha sweep under ./output/.')

# Cosmology and snapshot grid are shared by every run in the sweep; take them
# from the first one and assert nothing else disagrees.
REF = ALL_RUNS[0][1]
HUBBLE_H  = REF['hubble_h']
OMEGA_M   = REF['omega_matter']
OMEGA_L   = REF['omega_lambda']
REDSHIFTS = REF['redshifts']
SNAPS     = REF['output_snaps']
LAST_SNAP = REF['last_snap_nr']

# consistent-trees collapses the FOF grouping at the very last snapshot of a
# run: centrals drop by a third and the missing ones reappear as satellites of
# something else.  Every halo-grouped z = 0 quantity here is therefore measured
# one snapshot earlier.
Z0_SNAP = LAST_SNAP - 1


# ========================== COSMIC TIME ==========================

def cosmic_time_gyr(z):
    """Age of the universe at redshift *z*, in Gyr."""
    t_H = 977.8 / (HUBBLE_H * 100.0)

    def integrand(zp):
        return 1.0 / ((1 + zp) * np.sqrt(OMEGA_M * (1 + zp) ** 3 + OMEGA_L))

    result, _ = quad(integrand, z, 1000.0)
    return t_H * result


AGE_AT_SNAP = np.array([cosmic_time_gyr(z) for z in REDSHIFTS])
AGE_NOW = AGE_AT_SNAP[-1]
LOOKBACK_AT_SNAP = AGE_NOW - AGE_AT_SNAP


def snap_to_age(snap):
    """Age of the universe in Gyr at (possibly fractional) snapshot number."""
    return np.interp(np.asarray(snap, dtype=float),
                     np.arange(len(AGE_AT_SNAP)), AGE_AT_SNAP)


# ========================== STYLE ==========================

# The house style is sized for a single full-width panel.  Every figure here is
# a multi-panel grid, so the type is scaled down and each panel is given roughly
# the style's native aspect.  Typeface and tick style are left alone; TeX is off.
PANEL_FONT_SCALE = 0.60
PANEL_W, PANEL_H = 5.6, 4.5


def setup_style():
    plt.style.use("./plotting/kieren_cohare_palatino_sty.mplstyle")
    # No TeX: the style sheet asks for usetex, but TeX is unreliable on the
    # cluster (its luatex aborts and dumps core), so text is set by matplotlib
    # itself.  Maths uses STIX, a serif face that sits with the style's
    # Palatino; the serif list falls back to Palatino clones, then DejaVu Serif,
    # where Palatino is not installed.
    plt.rcParams.update({
        'text.usetex': False,
        'mathtext.fontset': 'stix',
        'font.serif': ['Palatino', 'Palatino Linotype', 'TeX Gyre Pagella',
                       'URW Palladio L', 'DejaVu Serif'],
    })
    f = PANEL_FONT_SCALE
    plt.rcParams.update({
        'font.size':       plt.rcParams['font.size'] * f,
        'axes.labelsize':  plt.rcParams['axes.labelsize'] * f,
        'axes.titlesize':  plt.rcParams['axes.titlesize'] * f,
        'xtick.labelsize': plt.rcParams['xtick.labelsize'] * f,
        'ytick.labelsize': plt.rcParams['ytick.labelsize'] * f,
        'legend.fontsize': plt.rcParams['legend.fontsize'] * f,
        'xtick.major.size': 5.0, 'ytick.major.size': 5.0,
        'xtick.minor.size': 3.0, 'ytick.minor.size': 3.0,
        'axes.linewidth': 1.0,
        'figure.autolayout': False,
    })


def panel_grid(nrows, ncols, **kwargs):
    return plt.subplots(nrows, ncols,
                        figsize=(ncols * PANEL_W, nrows * PANEL_H), **kwargs)


def save_figure(fig, name, dpi=None):
    path = os.path.join(OUT_DIR, name + OUTPUT_FORMAT)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, **({'dpi': dpi} if dpi else {}))
    print(f'  Saved: {path}')
    plt.close(fig)


def legend(ax, loc='best', **kwargs):
    kwargs.setdefault('frameon', False)
    kwargs.setdefault('fontsize', 7.5)
    leg = ax.legend(loc=loc, numpoints=1, labelspacing=0.15, **kwargs)
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    return leg


def alpha_label(alpha):
    """Legend entry for a run, marking the published value."""
    s = rf'$\beta_{{\rm clock}} = {alpha:g}$'
    return s + ' (published)' if alpha == FIDUCIAL else s


def line_style(alpha):
    """Heavier line for the published value, so it reads as the reference.
    Dash pattern from alpha_ls, so every figure gives a run the same style."""
    return dict(lw=3.0 if alpha == FIDUCIAL else 2.0,
                ls=alpha_ls(alpha),
                zorder=Z_LINE + (1 if alpha == FIDUCIAL else 0))


def alpha_ls(alpha):
    """Line style that carries alpha when colour is already spent on something else.

    The published value is solid; the others take dashed, dotted, dash-dot in
    the order RUNS lists them.
    """
    if alpha == FIDUCIAL:
        return '-'
    others = [a for a, _, _ in RUNS if a != FIDUCIAL]
    cycle = ('--', ':', '-.', (0, (5, 1, 1, 1)))
    return cycle[others.index(alpha) % len(cycle)] if alpha in others else '--'


# ========================== BINNING HELPERS ==========================

def binned_percentiles(x, y, bins, pct=(16, 50, 84), min_count=MIN_COUNT):
    """Bin centres and percentiles of *y* in bins of *x*, NaN where too sparse."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    centres = 0.5 * (bins[:-1] + bins[1:])
    out = np.full((len(pct), len(bins) - 1), np.nan)
    for i in range(len(bins) - 1):
        m = (x >= bins[i]) & (x < bins[i + 1])
        if m.sum() >= min_count:
            out[:, i] = np.percentile(y[m], pct)
    return centres, out


def binned_fraction(x, sel, bins, min_count=MIN_COUNT):
    """Per-bin fraction of objects satisfying *sel*."""
    x = np.asarray(x, float)
    out = np.full(len(bins) - 1, np.nan)
    for i in range(len(bins) - 1):
        m = np.isfinite(x) & (x >= bins[i]) & (x < bins[i + 1])
        if m.sum() >= min_count:
            out[i] = sel[m].sum() / m.sum()
    return 0.5 * (bins[:-1] + bins[1:]), out


def mass_function(log_masses, volume, binwidth=MF_BINWIDTH, mass_range=None):
    """Number density per dex.  Returns (bin centres, phi) with empty bins NaN."""
    log_masses = np.asarray(log_masses, float)
    log_masses = log_masses[np.isfinite(log_masses)]
    if log_masses.size == 0:
        return np.array([]), np.array([])
    if mass_range is None:
        lo = np.floor(log_masses.min() / binwidth) * binwidth
        hi = np.ceil(log_masses.max() / binwidth) * binwidth
    else:
        lo, hi = mass_range
    bins = np.arange(lo, hi + binwidth, binwidth)
    counts, edges = np.histogram(log_masses, bins=bins)
    phi = counts / volume / binwidth
    phi = np.where(counts > 0, phi, np.nan)
    return 0.5 * (edges[:-1] + edges[1:]), phi


# ========================== OBSERVATIONS ==========================
#
# f_ICL compilation, digitised.  Values are fractions, not per cent.  Two
# things to hold in mind before reading agreement or tension off these points:
#
#  * f_ICL is not one measurement.  Authors cut the ICL at different surface
#    brightnesses and radii and disagree over whether the BCG belongs in the
#    numerator, the denominator, or neither.  Furnell+21 and Burke+15 overlap
#    in redshift and still differ by a factor of a few; that spread is method,
#    not physics.  No attempt is made here to homogenise them.
#  * the model counts every intracluster star SAGE tracks, with no surface
#    brightness limit, so f_ICS should sit at or above a surface-brightness-
#    limited measurement rather than on top of it.
#
# 'scale' separates cluster-scale hosts from group-scale samples so each is
# drawn against the model line for the halo mass it belongs to.
_ICL_FRACTION_OBS = (
    dict(label='Spavone+20', scale='cluster', z=(0,), f=(0.3408,)),
    dict(label='Kluge+21', scale='cluster', z=(0.03,), f=(0.1792,)),
    dict(label='Zibetti+05', scale='cluster', z=(0.243,), f=(0.1085,)),
    dict(label='Feldmeier+04', scale='cluster',
         z=(0.162, 0.162, 0.162, 0.185),
         f=(0.1521, 0.1215, 0.1026, 0.0731)),
    dict(label='Burke+15', scale='cluster',
         z=(0.403, 0.387, 0.397, 0.339, 0.344, 0.342, 0.291, 0.225, 0.218,
            0.213, 0.195, 0.177),
         f=(0.0259, 0.0271, 0.033, 0.0554, 0.0601, 0.0719, 0.1297, 0.125,
            0.1627, 0.1804, 0.1686, 0.2311)),
    dict(label='Furnell+21', scale='cluster',
         z=(0.144, 0.127, 0.122, 0.081, 0.225, 0.215, 0.256, 0.306, 0.261,
            0.294, 0.322, 0.342, 0.372, 0.337, 0.377, 0.329, 0.496, 0.425,
            0.109),
         f=(0.3856, 0.3066, 0.3101, 0.2889, 0.2653, 0.2358, 0.2854, 0.2972,
            0.3255, 0.2748, 0.2759, 0.2665, 0.1981, 0.1887, 0.1545, 0.1533,
            0.1132, 0.0967, 0.316)),
    dict(label=r'Montes & Trujillo 18', scale='cluster',
         z=(0.301, 0.39, 0.342, 0.537, 0.537, 0.37, 0.043),
         f=(0.0767, 0.0861, 0.1309, 0.066, 0.0578, 0.0483, 0.1085)),
    dict(label='Presotto+14', scale='cluster', z=(0.435, 0.433), f=(0.1226, 0.0554)),
    dict(label='Ragusa+23', scale='cluster', z=(0.05,), f=(0.35,)),
    dict(label='Burke+12', scale='cluster',
         z=(0.947, 0.83, 0.795, 0.808, 1.223),
         f=(0.0142, 0.0259, 0.0377, 0.0153, 0.0236)),
    dict(label=r'Ko & Jee 18', scale='cluster', z=(1.238,), f=(0.0991,)),
    dict(label='XLSSC 122 (JWST)', scale='cluster', z=(1.98,), f=(0.17,)),
    dict(label='Ragusa+23', scale='group',
         z=(0.05,) * 16,
         f=(0.16, 0.05, 0.05, 0.17, 0.05, 0.27, 0.34, 0.17, 0.08, 0.35, 0.18,
            0.07, 0.2, 0.22, 0.28, 0.3)),
    dict(label='Ahad+25', scale='group',
         z=(0.12, 0.12, 0.12, 0.18, 0.18, 0.18, 0.24, 0.24, 0.24),
         f=(0.16, 0.1, 0.04, 0.15, 0.12, 0.08, 0.13, 0.15, 0.05)),
)

_OBS_MARKERS = ('o', 's', '^', 'v', 'D', 'P', 'X', '<', '>', 'h', '*', 'p', 'd')


# ---------------- Observing the model's clusters ----------------------------
#
# The model counts every intracluster star SAGE tracks, with no surface
# brightness limit and no aperture.  An observation counts only the ICL its own
# method admits.  The two are not the same quantity, so a model line and an
# observed point should not be plotted as though they were.
#
# The correction is taken, unmodified, from the literature: Brough et al. (2024,
# MNRAS 528, 771) ran eight observers' ICL methods on LSST-depth (mu_r = 30.3)
# mock images of 61 simulated clusters (10^14-10^14.5 Msun, z ~ 0) and quote the
# mean difference between the observed ICL fraction and the simulated one.
# Against their fiducial simulated ICL -- all stars 100 kpc-1 Mpc from the BCG,
# which they adopt as "the closest to the observed ICL fractions" -- the
# overall mean (observed - simulated) is -0.09 +/- 0.08, from -0.05 (CICLE,
# DAWIS-SS) to -0.13 (DAWIS-W) method by method (their section 4.3 and Table 3).
# The model is shifted by that offset, halo by halo:
#
#     f_obs = f_ICS + delta,   delta = -0.09 (band: -0.13 to -0.05), floored at 0
#
# The one assumption is ours and must be stated: SAGE's radius-less ICS is
# treated as their 100 kpc-1 Mpc ICL.  The calibration is for 10^14-10^14.5 Msun
# at z ~ 0 only; applying it to groups and higher redshift is an extrapolation.
# The method matters as much as the depth: Montes & Trujillo (2018, Table 2)
# measure 0.4-3.5 per cent of the light in the ICL with a surface-brightness
# threshold (mu_V > 26) but 9-30 per cent within R_limit from profile fits for
# the same clusters, and Kluge et al. (2021, section 5) find the ICL share of
# BCG+ICL changing from 34 per cent (SB cut at 27 g') to 52 per cent (double
# Sersic).
#
# PROVENANCE.  Every entry records where its value came from:
#   'literature' -- a number quoted in a published paper, used as quoted
#   'assumed'    -- a documented placeholder, NOT a measurement
# report_sb_correction() flags an assumed value at run time so it cannot
# quietly reach a draft.
APPLY_SB_CORRECTION = True

# (low, high) edges of the offset band and its centre, as quoted.
MODEL_SB_RECOVERY = dict(
    edges=(-0.13, -0.05),
    centre=-0.09,
    source='literature',
    note='Brough et al. (2024, sec. 4.3, Table 3): mean (observed - simulated) ICL '
         'fraction -0.09 +/- 0.08 against the simulated 100 kpc-1 Mpc ICL, '
         '-0.05 to -0.13 by method; for 10^14-10^14.5 Msun at z ~ 0.',
)


def observed_fics(ics, stars, delta):
    """f_ICS as an observer would measure it: the model fraction plus the quoted
    observed-minus-simulated offset *delta*, floored at zero."""
    ics, stars = np.asarray(ics, float), np.asarray(stars, float)
    tot = ics + stars
    with np.errstate(invalid='ignore', divide='ignore'):
        f = np.where(tot > 0, ics / tot, np.nan)
    return np.clip(f + delta, 0.0, 1.0)


def report_sb_correction():
    """State the cut applied to the model, and flag it if it is a placeholder."""
    if not APPLY_SB_CORRECTION:
        print('Surface-brightness cut on the model: OFF -- f_ICS is the SB-free total')
        return
    lo, hi = MODEL_SB_RECOVERY['edges']
    print("Observational correction applied to SAGE26's clusters:")
    print(f'  f_obs = f_ICS + delta,  delta = {MODEL_SB_RECOVERY["centre"]:+.2f} '
          f'(band {lo:+.2f} to {hi:+.2f}), floored at 0  [{MODEL_SB_RECOVERY["source"]}]')
    print(f'  {MODEL_SB_RECOVERY["note"]}')
    if MODEL_SB_RECOVERY['source'] != 'literature':
        print('  WARNING: the correction is ASSUMED, not taken from the literature.')
    print('  Assumption to state: the model ICS is treated as the 100 kpc-1 Mpc ICL;')
    print('  groups and z > 0 extrapolate the 10^14-10^14.5 Msun, z ~ 0 calibration.')


# ---------------- Does the halo-mass trend survive? (referee 3a, 3c, 3e) ------
#
# Three forms of the Brough et al. (2024) correction, applied halo by halo:
#   'additive'       f + delta, delta = -0.09, floored at 0 (the paper's form);
#   'multiplicative' f x 0.13/0.22 -- the ratio of their overall mean observed ICL
#                    fraction (0.13 +/- 0.05, Table 3) to their mean simulated
#                    100 kpc-1 Mpc fraction (0.22 +/- 0.09, Table 1).  The ratio is
#                    OURS, formed from two quoted means; they do not quote it;
#   'kinematic'      their eq. (1), as quoted: f_image,obs / f_kin,sim =
#                    -0.23 f_kin,sim + 0.36, i.e. treating the model ICS as the
#                    kinematic (unbound-star) ICL.  Fitted for 61 clusters of
#                    log M200c = 14-14.5 with f_kin of roughly 0.4-0.9.
# Their calibration spans log M200c = 14-14.5 only, and they "do not observe a
# dependence of any of the ICL fraction measures on cluster halo mass in this
# narrow cluster mass range" (sec. 4.3): it says nothing either way about a
# mass-dependent correction across groups and clusters.
TREND_CORRECTIONS = {
    'none':           lambda f: f,
    'additive':       lambda f: np.clip(f - 0.09, 0.0, 1.0),
    'multiplicative': lambda f: f * (0.13 / 0.22),
    'kinematic':      lambda f: f * (0.36 - 0.23 * f),
}

# The observed group samples' own halo masses, for a mass-matched model median.
# Ragusa+23 (A&A 670, L20) Table 1: Mvir of the 16 VEGAS groups and sub-groups
# [1e13 Msun] (their H0 = 73), in the order of _ICL_FRACTION_OBS.  Their 17th
# system, the Antlia cluster, is Mvir = 26.3e13 with f_ICL = 0.32 +/- 0.03.
RAGUSA23_GROUP_MVIR = np.array([0.85, 1.55, 4.00, 5.89, 1.78, 1.17, 8.99, 1.26,
                                0.79, 2.69, 0.40, 3.02, 26.9, 6.31, 16.6, 2.69]) * 1e13
# Ahad+25 (A&A 702, A271): stacks of GAMA groups with 12.5 <= log M200c <= 14.0
# (halo masses from group luminosity); no per-stack masses are tabulated.
AHAD25_M200C = (10 ** 12.5, 10 ** 14.0)


def report_trend(scans):
    """The f_ICS - halo mass trend under each correction, with the group median
    taken over (i) all haloes of 1e13-1e14 (the paper), (ii) haloes matched to
    Ragusa+23's group masses and (iii) Ahad+25's mass range.  Prints the slope
    d f / d log Mvir of a straight-line fit to every halo of 1e12.5-1e15.5."""
    print()
    print('T. Does the halo-mass trend survive the correction and the mass matching?')
    conv = _m200c_to_mvir_factor()
    lo_a, hi_a = AHAD25_M200C[0] * conv, AHAD25_M200C[1] * conv
    lr = np.log10(RAGUSA23_GROUP_MVIR)
    for alpha, hdr, colour, s in scans:
        mvir, stars, ics = s['h_mvir'], s['h_mstar'], s['h_ics']
        tot = stars + ics
        ok = np.isfinite(mvir) & (mvir > 0) & (tot > 0)
        lm, f = np.log10(mvir[ok]), ics[ok] / tot[ok]
        print(f'  beta_clock = {alpha:g}')
        print(f'    {"correction":15s} {"groups 13-14":>13s} {"Ragusa-mass":>12s} '
              f'{"Ahad range":>11s} {"clusters":>9s} {"cl - gr":>8s} {"slope/dex":>10s}')
        for name, fn in TREND_CORRECTIONS.items():
            g = fn(f)
            gr = np.median(g[(lm >= 13) & (lm < 14)])
            # each Ragusa group contributes the median of haloes within 0.1 dex of it
            rag = np.median([np.median(g[np.abs(lm - x) < 0.1]) for x in lr
                             if (np.abs(lm - x) < 0.1).sum() >= MIN_COUNT])
            ahad = np.median(g[(lm >= np.log10(lo_a)) & (lm < np.log10(hi_a))])
            cl = np.median(g[lm >= 14]) if (lm >= 14).sum() >= MIN_COUNT else np.nan
            fit = (lm >= 12.5) & (lm < 15.5)
            slope = np.polyfit(lm[fit], g[fit], 1)[0]
            print(f'    {name:15s} {gr:13.3f} {rag:12.3f} {ahad:11.3f} {cl:9.3f} '
                  f'{cl - gr:+8.3f} {slope:+10.3f}')
    print('  observed: groups 0.15 (16-84th 0.05-0.29 over both classes), clusters 0.13;')
    print('  Ragusa+23 find no significant trend over Mvir = 1e12.5-1e15.5.')


def icl_fraction_observations(scale):
    return [dict(o, z=np.asarray(o['z'], float), f=np.asarray(o['f'], float))
            for o in _ICL_FRACTION_OBS if o['scale'] == scale]


def load_moffett16_bulge_fraction():
    """GAMA bulge mass fraction against stellar mass (Moffett et al. 2016).

    Columns are log m*, B/T, and the statistical 1-sigma bounds.  Chabrier
    already, so no IMF shift is applied.
    """
    path = os.path.join(OBS_DIR, 'morphology/Moffet16.dat')
    if not os.path.exists(path):
        return None
    d = np.loadtxt(path)
    return dict(logm=d[:, 0], bt=d[:, 1], lo=d[:, 2], hi=d[:, 3])


def _read_ecsv(name, zcol, ycol, errup=None, errlo=None, shift=0.0):
    """One SFRD compilation from data/sfrd, on the model's Chabrier scale."""
    if not HAS_ASTROPY:
        return None
    path = os.path.join(OBS_DIR, 'sfrd', name)
    if not os.path.exists(path):
        return None
    try:
        t = Table.read(path, format='ascii.ecsv')
        z = np.asarray(t[zcol], float)
        y = np.asarray(t[ycol], float) + shift
        eu = np.asarray(t[errup], float) if errup else np.zeros_like(y)
        el = np.asarray(t[errlo], float) if errlo else np.zeros_like(y)
        return dict(z=z, y=y, eu=eu, el=el)
    except Exception as exc:
        print(f'  could not read {name}: {exc}')
        return None


def load_sfrd_observations():
    """The SFRD compilations, each with its marker and colour."""
    out = []
    md = _read_ecsv('MandD_sfrd_2014.ecsv', 'z_min', 'log_psi',
                    'e_log_psi_up', 'e_log_psi_lo',
                    shift=SALPETER_TO_CHABRIER_DEX)
    if md:
        out.append(dict(md, label=r'Madau & Dickinson 14', marker='o',
                        colour='0.45'))
    oe = _read_ecsv('oesch_sfrd_2018.ecsv', 'z', 'log_rho_sfr',
                    'e_log_rho_sfr_upper', 'e_log_rho_sfr_lower')
    if oe:
        out.append(dict(oe, label='Oesch+18', marker='s', colour='0.55'))
    ha = _read_ecsv('harikane_density_2023.ecsv', 'z', 'log_rho_SFR_UV',
                    'e_log_rho_SFR_UV_upper', 'e_log_rho_SFR_UV_lower')
    if ha:
        out.append(dict(ha, label='Harikane+23', marker='^', colour='0.6'))
    mc = _read_ecsv('mcleod_rhouv_2024.ecsv', 'z', 'log_rho_sfr')
    if mc:
        out.append(dict(mc, label='McLeod+24', marker='D', colour='0.65'))
    # GAMA/G10-COSMOS/3D-HST and GAMA/DEVILS, both Chabrier (copied from shark/data/Global).
    # Driver+18: bin centre; error = its four published terms (Eddington bias, Poisson,
    # cosmic variance, AGN) in quadrature, as shark plots it.
    path = os.path.join(OBS_DIR, 'sfrd', 'Driver18_sfr.dat')
    if os.path.exists(path):
        d = np.loadtxt(path)
        err = np.sqrt((d[:, 3:7] ** 2).sum(axis=1))
        out.append(dict(z=0.5 * (d[:, 0] + d[:, 1]), y=d[:, 2], eu=err, el=err,
                        label='Driver+18', marker='v', colour='0.35'))
    path = os.path.join(OBS_DIR, 'sfrd', 'DSilva23_sfr.dat')
    if os.path.exists(path):
        d = np.loadtxt(path)
        out.append(dict(z=d[:, 3], y=d[:, 0], eu=d[:, 1], el=d[:, 2],
                        label="D'Silva+23", marker='p', colour='0.4'))
    return out


# ========================== THE SNAPSHOT SCAN ==========================
#
# One pass over every snapshot of a run produces everything figures 1-4 need:
# the cosmic SFR density, the halo-grouped f_ICS in groups and clusters as a
# function of redshift, and the full z = 0 galaxy and halo catalogues.
#
# Galaxies are grouped into haloes by CentralGalaxyIndex, which is unique per
# FOF group within a file, so the grouping is done file by file.
#
# f_ICS follows the definition the ICS paper uses throughout:
#
#     f_ICS = m_ICS(halo) / (m_ICS(halo) + sum of m* over every galaxy in it)
#
# -- the denominator is all the stars in the halo including satellites, not the
# BCG alone.  m_ICS is summed over the halo too: satellites carry their own ICS
# until they are absorbed.

_SCAN_PROPS = ('StellarMass', 'BulgeMass', 'IntraClusterStars', 'Mvir', 'Type',
               'CentralGalaxyIndex', 'SfrDisk', 'SfrBulge')
# The ICS assembly bookkeeping (TrackICSAssembly).  Read when present, zeros
# otherwise, so a run made before it existed still scans.
_SCAN_OPTIONAL = ('ICS_disrupt', 'ICS_accrete', 'ICS_sum_mt')
_HALO_EXTRA = ('bcg', 'bcg_bulge', 'ics_dis', 'ics_acc', 'ics_mt')
# z = 0 galaxy columns for the bulge-channel, black-hole and gas diagnostics
# (report_budget section H): output field read for each, all masses in Msun.
_Z0_EXTRA_SRC = {'g_ibulge': 'InstabilityBulgeMass', 'g_mbulge': 'MergerBulgeMass',
                 'g_bh': 'BlackHoleMass', 'g_cold': 'ColdGas', 'g_hot': 'HotGas',
                 'g_cgm': 'CGMgas', 'g_cmvir': 'CentralMvir',
                 'g_qbh': 'QuasarModeBHaccretionMass'}
# Read as stored, without the mass conversion, and only ever reported as ratios
# between runs or as dimensionless ratios.  Heating is the cooling mass the AGN
# offset this snapshot, capped at the cooling left after the heated radius r_heat
# has suppressed it (model_cooling_heating.c) -- so it is ZERO once r_heat >= r_cool,
# when feedback is strongest.  Feedback strength is read from r_heat / r_cool and
# the net CoolingRate instead.
_Z0_RAW_SRC = {'g_heat': 'Heating', 'g_rheat': 'r_heat', 'g_rcool_rvir': 'RcoolToRvir',
               'g_rvir': 'Rvir', 'g_coolrate': 'CoolingRate', 'g_mdotcool': 'mdot_cool'}
_Z0_EXTRA = tuple(_Z0_EXTRA_SRC) + tuple(_Z0_RAW_SRC) + ('g_sfr',)


def _halo_totals(d, conv):
    """Per-FOF-halo (Mvir, total stellar mass, total ICS, extras) for one file+snapshot.

    The extras are the central's own stellar and bulge mass (the BCG) and the
    halo's ICS assembly bookkeeping: in-situ (disrupted here) and ex-situ
    (carried in) ICS, and the mass-weighted deposit time sum in code mass x
    code time.
    """
    _, idx = np.unique(d['CentralGalaxyIndex'].astype(np.int64),
                       return_inverse=True)
    n = idx.max() + 1 if idx.size else 0
    if n == 0:
        return (np.array([]),) * 3 + ({k: np.array([]) for k in _HALO_EXTRA},)
    stars = np.bincount(idx, weights=d['StellarMass'], minlength=n) * conv
    ics = np.bincount(idx, weights=d['IntraClusterStars'], minlength=n) * conv
    cen = d['Type'] == 0
    mvir = np.full(n, np.nan)
    mvir[idx[cen]] = d['Mvir'][cen] * conv
    extra = {k: np.full(n, np.nan) for k in ('bcg', 'bcg_bulge')}
    extra['bcg'][idx[cen]] = d['StellarMass'][cen] * conv
    extra['bcg_bulge'][idx[cen]] = d['BulgeMass'][cen] * conv
    for k, src, c in (('ics_dis', 'ICS_disrupt', conv), ('ics_acc', 'ICS_accrete', conv),
                      ('ics_mt', 'ICS_sum_mt', conv)):
        extra[k] = np.bincount(idx, weights=d[src], minlength=n) * c
    return mvir, stars, ics, extra


def _read_snapshot(args):
    """Per-halo totals for one snapshot across every file of a run.

    A free function of its arguments so scan_run can farm snapshots out to
    worker processes: the read is I/O-bound and every snapshot is independent.
    The z = 0 galaxy columns come back only for Z0_SNAP.
    """
    files, snap, conv = args
    key = f'Snap_{snap}'
    out = dict(sfr_sum=0.0, ngal=0, mvir=[], stars=[], ics=[], missing=set(),
               extra={k: [] for k in _HALO_EXTRA},
               g_mstar=[], g_bulge=[], g_type=[],
               **{k: [] for k in _Z0_EXTRA})
    for fp in files:
        with h5.File(fp, 'r') as f:
            if key not in f:
                continue
            g = f[key]
            if g['StellarMass'].shape[0] == 0:
                continue
            d = {p: g[p][:] for p in _SCAN_PROPS}
            for p in _SCAN_OPTIONAL:
                if p in g:
                    d[p] = g[p][:]
                else:
                    d[p] = np.zeros(g['StellarMass'].shape[0])
                    out['missing'].add(p)
        out['ngal'] += d['StellarMass'].shape[0]
        out['sfr_sum'] += float((d['SfrDisk'] + d['SfrBulge']).sum())
        mv, st, ic, ex = _halo_totals(d, conv)
        out['mvir'].append(mv)
        out['stars'].append(st)
        out['ics'].append(ic)
        for k in _HALO_EXTRA:
            out['extra'][k].append(ex[k])
        if snap == Z0_SNAP:
            out['g_mstar'].append(d['StellarMass'] * conv)
            out['g_bulge'].append(d['BulgeMass'] * conv)
            out['g_type'].append(d['Type'].astype(np.int8))
            out['g_sfr'].append((d['SfrDisk'] + d['SfrBulge']).astype(np.float32))
            with h5.File(fp, 'r') as f:
                g = f[key]
                for k, src in _Z0_EXTRA_SRC.items():
                    out[k].append(g[src][:] * conv if src in g else
                                  np.zeros(g['StellarMass'].shape[0]))
                for k, src in _Z0_RAW_SRC.items():
                    out[k].append(g[src][:].astype(float) if src in g else
                                  np.zeros(g['StellarMass'].shape[0]))
    return out


def scan_run(hdr, verbose=True):
    """Walk every snapshot of one run and collect the per-snapshot summaries."""
    conv, files = hdr['mass_convert'], hdr['files']
    nsnap = len(REDSHIFTS)

    sfrd = np.zeros(nsnap)
    fics_cl = np.full((3, nsnap), np.nan)       # 16/50/84 per cent, clusters
    fics_gr = np.full((3, nsnap), np.nan)       # groups
    # the same, re-measured through each edge of the surface-brightness cut
    n_g = len(MODEL_SB_RECOVERY['edges'])
    fics_cl_sb = np.full((n_g, 3, nsnap), np.nan)
    fics_gr_sb = np.full((n_g, 3, nsnap), np.nan)
    n_cl = np.zeros(nsnap, dtype=int)
    n_gr = np.zeros(nsnap, dtype=int)

    z0 = {k: [] for k in ('h_mvir', 'h_mstar', 'h_ics',
                          'g_mstar', 'g_bulge', 'g_type')
          + _Z0_EXTRA + tuple('h_' + e for e in _HALO_EXTRA)}

    # global budgets and the BCG of groups and clusters, at every snapshot
    rho_star = np.zeros(nsnap)                  # Msun / Mpc^3, galaxies
    rho_ics = np.zeros(nsnap)                   # Msun / Mpc^3, intracluster
    bcg_cl = np.full((3, nsnap), np.nan)        # 16/50/84 of m_BCG, clusters
    bcg_gr = np.full((3, nsnap), np.nan)
    missing_optional = set()
    # ICS mass function counts: (snapshot, halo-mass bin + 'all', ICS bin)
    mf_counts = np.zeros((nsnap, len(MF_HALO_BINS), len(MF_ICS_EDGES) - 1))

    # every group- and cluster-scale halo at every snapshot, kept per object so
    # figure 6 can draw the raw distribution at any g rather than a percentile
    gc = {k: [] for k in ('gc_z', 'gc_mvir', 'gc_mstar', 'gc_ics')}

    jobs = min(SCAN_JOBS, len(SNAPS))
    if verbose:
        print(f'  {len(files)} files x {len(SNAPS)} snapshots, {jobs} worker(s)')
    t_start = time.time()
    if jobs > 1:
        pool = ProcessPoolExecutor(max_workers=jobs)
        reads = pool.map(_read_snapshot, [(files, s, conv) for s in SNAPS])
    else:
        pool = None
        reads = (_read_snapshot((files, s, conv)) for s in SNAPS)

    for i, (snap, r) in enumerate(zip(SNAPS, reads)):
        if verbose:
            print(f'    Snap_{snap:<3d} {r["ngal"]:>12,d} galaxies  '
                  f'[{i + 1}/{len(SNAPS)}, {time.time() - t_start:7.0f} s]', flush=True)
        missing_optional |= r['missing']
        sfr_sum = r['sfr_sum']
        mvir_a, stars_a, ics_a = r['mvir'], r['stars'], r['ics']
        extra_a = r['extra']
        if snap == Z0_SNAP:
            for k in ('g_mstar', 'g_bulge', 'g_type') + _Z0_EXTRA:
                z0[k].extend(r[k])

        sfrd[snap] = sfr_sum / hdr['volume']
        if not mvir_a:
            continue
        mvir = np.concatenate(mvir_a)
        stars = np.concatenate(stars_a)
        ics = np.concatenate(ics_a)
        extra = {k: np.concatenate(v) for k, v in extra_a.items()}
        rho_star[snap] = stars.sum() / hdr['volume']
        rho_ics[snap] = ics.sum() / hdr['volume']
        has = np.isfinite(mvir) & (ics > 0)
        if has.any():
            lic, lmv = np.log10(ics[has]), np.log10(mvir[has])
            for b in range(len(MF_HALO_BINS) - 1):
                sel = (lmv >= MF_HALO_BINS[b]) & (lmv < MF_HALO_BINS[b + 1])
                mf_counts[snap, b] = np.histogram(lic[sel], bins=MF_ICS_EDGES)[0]
            mf_counts[snap, -1] = np.histogram(lic, bins=MF_ICS_EDGES)[0]

        tot = stars + ics
        with np.errstate(invalid='ignore', divide='ignore'):
            fics = np.where(tot > 0, ics / tot, np.nan)

        cl = np.isfinite(mvir) & (mvir >= CLUSTER_LO) & np.isfinite(fics)
        gr = np.isfinite(mvir) & (mvir >= GROUP_LO) & (mvir < GROUP_HI) & np.isfinite(fics)
        n_cl[snap], n_gr[snap] = cl.sum(), gr.sum()
        if cl.sum() >= 3:
            fics_cl[:, snap] = np.percentile(fics[cl], (16, 50, 84))
        if gr.sum() >= MIN_COUNT:
            fics_gr[:, snap] = np.percentile(fics[gr], (16, 50, 84))
        b_ok = np.isfinite(extra['bcg'])
        if (cl & b_ok).sum() >= 3:
            bcg_cl[:, snap] = np.percentile(extra['bcg'][cl & b_ok], (16, 50, 84))
        if (gr & b_ok).sum() >= MIN_COUNT:
            bcg_gr[:, snap] = np.percentile(extra['bcg'][gr & b_ok], (16, 50, 84))

        for j, gval in enumerate(MODEL_SB_RECOVERY['edges']):
            f_obs = observed_fics(ics, stars, gval)
            if cl.sum() >= 3:
                fics_cl_sb[j, :, snap] = np.percentile(f_obs[cl], (16, 50, 84))
            if gr.sum() >= MIN_COUNT:
                fics_gr_sb[j, :, snap] = np.percentile(f_obs[gr], (16, 50, 84))

        keep = np.isfinite(mvir) & (mvir >= GROUP_LO) & (stars + ics > 0)
        if keep.any():
            gc['gc_z'].append(np.full(int(keep.sum()), REDSHIFTS[snap]))
            gc['gc_mvir'].append(mvir[keep])
            gc['gc_mstar'].append(stars[keep])
            gc['gc_ics'].append(ics[keep])

        if snap == Z0_SNAP:
            z0['h_mvir'].append(mvir)
            z0['h_mstar'].append(stars)
            z0['h_ics'].append(ics)
            for k in _HALO_EXTRA:
                z0['h_' + k].append(extra[k])

    if pool is not None:
        pool.shutdown()
    if missing_optional and verbose:
        print(f'  note: {sorted(missing_optional)} not in this output -- ICS assembly '
              f'diagnostics will read zero (run with TrackICSAssembly on)')
    out = dict(sfrd=sfrd, fics_cl=fics_cl, fics_gr=fics_gr,
               rho_star=rho_star, rho_ics=rho_ics, bcg_cl=bcg_cl, bcg_gr=bcg_gr,
               mf_counts=mf_counts,
               time_convert_gyr=np.array([hdr['time_convert_gyr']]),
               fics_cl_sb=fics_cl_sb, fics_gr_sb=fics_gr_sb,
               n_cl=n_cl, n_gr=n_gr, volume=np.array([hdr['volume']]),
               sb_g=np.asarray(MODEL_SB_RECOVERY['edges'], float))
    for k, v in list(z0.items()) + list(gc.items()):
        out[k] = np.concatenate(v) if v else np.array([])
    if verbose:
        print(f'  {len(out["g_mstar"]):,} galaxies and {len(out["h_mvir"]):,} '
              f'haloes at Snap_{Z0_SNAP} (z = {REDSHIFTS[Z0_SNAP]:.3f})')
    return out


def load_scan(hdr, refresh=False, verbose=True):
    """The snapshot scan for one run, from the npz cache when it is current."""
    tag = os.path.basename(os.path.normpath(hdr['directory']))
    cache = os.path.join(CACHE_DIR, f'scan_{CACHE_VERSION}_{tag}.npz')
    newest = max(os.path.getmtime(f) for f in hdr['files'])
    if not refresh and os.path.exists(cache) and os.path.getmtime(cache) > newest:
        if verbose:
            print(f'  scan from cache: {cache}')
        z = np.load(cache)
        return {k: z[k] for k in z.files}

    if verbose:
        print(f'  scanning {hdr["directory"]}')
    scan = scan_run(hdr, verbose=verbose)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(cache, **scan)
    return scan


def all_scans(refresh=False):
    """(alpha, header, colour, scan) for every run in the sweep."""
    out = []
    for alpha, hdr, colour in ALL_RUNS:
        print(f'alpha = {alpha:g}')
        out.append((alpha, hdr, colour, load_scan(hdr, refresh=refresh)))
    return out


# ========================== THE DISRUPTION LOG ==========================
#
# core_build_model.c writes one row per destruction event when SAGE_DISRUPT_LOG
# is set, recording what the gate saw before the destruction routines zero the
# galaxy.  The columns used here:
#
#   mergtime     the clock's remaining value at destruction [code time]
#   dest         4 = ICS, 1 = onto the central
#   stellarmass  the satellite's stellar mass at destruction [code mass]
#   type         1 = still has a subhalo, 2 = orphan
#   snap_infall  snapshot the satellite fell in at, -1 if never a satellite
#   event_time   the event's own time on SAGE's Age scale [code time]
#   age_infall   Age[snap_infall] on the same scale
#
# SAGE's Age array is a lookback time in code units, so the clock's starting
# value -- the dynamical friction time the satellite was handed at infall -- is
#
#   T_df = mergtime + (age_infall - event_time)
#
# because MergTime is set once, in the same block that sets TimeOfInfall, and
# decremented by the elapsed time at every substep thereafter.
#
# Two channels bypass estimate_merging_time entirely and must be excluded from
# any statement about T_df:
#   * a satellite whose subhalo has fewer than MinNumPartSatHalo = 10 particles
#     is handed MergTime = -1 (about 7 per cent of events; it shows up as a
#     recovered T_df of about -1 code unit);
#   * a galaxy dropping straight from Type 0 to orphan is handed MergTime = 0
#     without its TimeOfInfall being refreshed.
# Both always merge onto the central, and both are kept in the routing counts --
# they are real destinations -- but excluded from the T_df distributions.

_DISRUPT_COLS = ('snap', 'type', 'mergtime', 'stellarmass', 'coldgas', 'dest',
                 'snap_infall', 'event_time', 'age_infall')


def _read_disrupt_csv(paths):
    """The disruption log as a dict of columns, using pandas when available."""
    try:
        import pandas as pd
        frames = [pd.read_csv(p, usecols=list(_DISRUPT_COLS)) for p in paths]
        df = pd.concat(frames, ignore_index=True)
        return {c: df[c].to_numpy() for c in _DISRUPT_COLS}
    except ImportError:
        cols = {c: [] for c in _DISRUPT_COLS}
        for p in paths:
            with open(p) as fh:
                names = fh.readline().strip().split(',')
            idx = [names.index(c) for c in _DISRUPT_COLS]
            d = np.loadtxt(p, delimiter=',', skiprows=1, usecols=idx)
            for j, c in enumerate(_DISRUPT_COLS):
                cols[c].append(d[:, j])
        return {c: np.concatenate(v) for c, v in cols.items()}


def _code_time_to_gyr(snap_infall, age_infall):
    """Calibrate SAGE's code time unit against the cosmology, in Gyr per unit.

    Age[] is a lookback time, so every (snap_infall, age_infall) pair in the log
    is a direct measurement of the conversion.  Taking the median over all of
    them is immune to a stray row.
    """
    ok = (snap_infall >= 0) & (age_infall > 0)
    if not ok.any():
        return np.nan
    snaps = snap_infall[ok].astype(int)
    ratio = LOOKBACK_AT_SNAP[snaps] / age_infall[ok]
    ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
    return float(np.median(ratio)) if ratio.size else np.nan


def load_disrupt(hdr, verbose=True):
    """The per-event disruption log for one run, with derived times in Gyr.

    Returns None when the run was made without SAGE_DISRUPT_LOG set.
    """
    paths = sorted(glob.glob(os.path.join(hdr['directory'], 'disrupt', '*.csv')))
    if not paths:
        return None
    d = _read_disrupt_csv(paths)

    conv = hdr['mass_convert']
    unit = _code_time_to_gyr(d['snap_infall'], d['age_infall'])

    ev = dict(
        snap=d['snap'].astype(int),
        sat_type=d['type'].astype(int),
        mstar=d['stellarmass'] * conv,
        mcold=d['coldgas'] * conv,
        to_ics=d['dest'] == 4,
        snap_infall=d['snap_infall'],
    )
    ev['to_bcg'] = ~ev['to_ics']
    ev['z_dest'] = REDSHIFTS[np.clip(ev['snap'], 0, len(REDSHIFTS) - 1)]
    ev['t_dest'] = AGE_AT_SNAP[np.clip(ev['snap'], 0, len(AGE_AT_SNAP) - 1)]

    # the clock, and the life it was meant to time
    elapsed = (d['age_infall'] - d['event_time']) * unit          # Gyr
    ev['t_remaining'] = d['mergtime'] * unit                      # Gyr, signed
    ev['t_df'] = ev['t_remaining'] + elapsed                      # Gyr
    ev['t_life'] = elapsed                                        # Gyr

    # Rows whose clock never came from estimate_merging_time.  A recovered
    # T_df <= 0 is the signature of the failure branch: a subhalo below
    # MinNumPartSatHalo = 10 particles is handed MergTime = -1 outright, which
    # comes back as T_df ~ -1 code unit and always merges onto the central.
    ev['clock_known'] = ((d['snap_infall'] >= 0) & np.isfinite(elapsed)
                         & (elapsed >= 0) & (ev['t_df'] > 0))
    for k in ('t_df', 't_life'):
        ev[k] = np.where(ev['clock_known'], ev[k], np.nan)

    ev['unit_gyr'] = unit
    if verbose:
        n = len(ev['snap'])
        print(f'  {n:,} destruction events; code time unit = {unit:.1f} Gyr; '
              f'{ev["clock_known"].sum() / n * 100:.1f} per cent with a clock from '
              f'estimate_merging_time')
    return ev


def all_disrupt():
    """(alpha, header, colour, events) for every run that has a disruption log."""
    out = []
    for alpha, hdr, colour in ALL_RUNS:
        print(f'alpha = {alpha:g}')
        ev = load_disrupt(hdr)
        if ev is None:
            print(f'  no disruption log in {hdr["directory"]} -- '
                  f'rerun with SAGE_DISRUPT_LOG set')
            continue
        out.append((alpha, hdr, colour, ev))
    return out


# ============ FIGURE 1: THE INTRACLUSTER FRACTION ============

# ---------------- main panels with residual strips --------------------------
#
# Figures 1-5 carry a short residual panel under each main panel.  The residual
# is taken against the observations wherever a panel has them (figs 1, 2a, 3a)
# and against the published beta_clock = 2 run where it does not (figs 2b, 4, 5);
# each strip's y label says which.  Fractions are shown as a ratio on a log axis,
# log-scale quantities as a log ratio in dex.

RESID_HEIGHT = (3, 1)          # main : residual panel heights
RATIO_TICKS = (0.25, 0.5, 1, 2, 4)


def resid_grid(nrows, ncols, full_height=(), scale=(1.0, 1.0)):
    """A figure of main panels, each with a residual strip sharing its x axis.

    Cells listed in *full_height* get one tall panel and no strip (their
    residual entry is None).  *scale* enlarges the figure (width, height) while
    the fonts stay fixed, giving crowded legends room.  Returns (fig, main
    axes, residual axes) as 2-D object arrays.
    """
    fig = plt.figure(figsize=(ncols * PANEL_W * scale[0],
                              nrows * PANEL_H * 1.3 * scale[1]))
    outer = fig.add_gridspec(nrows, ncols)
    main = np.empty((nrows, ncols), dtype=object)
    res = np.empty((nrows, ncols), dtype=object)
    for r in range(nrows):
        for c in range(ncols):
            if (r, c) in full_height:
                main[r, c], res[r, c] = fig.add_subplot(outer[r, c]), None
                continue
            inner = outer[r, c].subgridspec(2, 1, height_ratios=RESID_HEIGHT, hspace=0.0)
            ax = fig.add_subplot(inner[0])
            rax = fig.add_subplot(inner[1], sharex=ax)
            ax.tick_params(labelbottom=False)
            main[r, c], res[r, c] = ax, rax
    return fig, main, res


class _BlankLowest(mticker.Formatter):
    """Wrap an axis's formatter so the tick at the bottom (or top) of the axis is blank.

    Tick positions are only known when the figure is drawn, so the check is made
    then, per tick, against the axis limits.
    """

    def __init__(self, ax, orig, edge='bottom'):
        self.ax, self.orig, self.edge = ax, orig, edge

    def set_axis(self, axis):
        super().set_axis(axis)
        self.orig.set_axis(axis)

    def set_locs(self, locs):
        self.orig.set_locs(locs)

    def __call__(self, x, pos=None):
        lo, hi = self.ax.get_ylim()
        if self.edge == 'top':
            lo, hi = hi, lo
        if self.ax.get_yscale() == 'log':
            close = x > 0 and abs(np.log10(x) - np.log10(lo)) < 1e-6 * abs(np.log10(hi / lo))
        else:
            close = abs(x - lo) < 1e-6 * abs(hi - lo)
        return '' if close else self.orig(x, pos)


def hide_bottom_ticklabel(main, res=None):
    """Blank each main panel's lowest y tick label where a residual strip sits
    directly beneath it, and the strip's top label: either one, at the join,
    reads as belonging to the other panel."""
    for ax in np.ravel(main):
        if ax is None or not ax.get_visible():
            continue
        ax.yaxis.set_major_formatter(_BlankLowest(ax, ax.yaxis.get_major_formatter()))
    for rax in np.ravel(res) if res is not None else ():
        if rax is None or not rax.get_visible():
            continue
        rax.yaxis.set_major_formatter(_BlankLowest(rax, rax.yaxis.get_major_formatter(), 'top'))


def resid_axis(rax, label, kind='ratio', lim=None):
    """Reference line, scale and label for a residual strip.

    kind = 'ratio' draws run/reference on a log axis about 1; 'dex' draws a log
    ratio in dex about 0.
    """
    if kind == 'ratio':
        rax.set_yscale('log')
        rax.axhline(1.0, color='0.6', lw=0.7, ls='-', zorder=1)
        lim = lim or (0.2, 5.0)
        rax.set_ylim(*lim)
        ticks = [t for t in RATIO_TICKS if lim[0] <= t <= lim[1]]
        rax.set_yticks(ticks)
        rax.set_yticklabels([f'{t:g}' for t in ticks])
        rax.minorticks_off()
    else:
        rax.axhline(0.0, color='0.6', lw=0.7, ls='-', zorder=1)
        if lim:
            rax.set_ylim(*lim)
    rax.set_ylabel(label, fontsize=7.5)


def _binned_obs(z, y, edges, min_count=2):
    """Median and 16th/84th of pooled observations in redshift bins.

    The residual strips are drawn as lines, so scattered observed points are
    pooled across samples and reduced to one median per bin first.  Bins with
    fewer than *min_count* points are dropped.
    """
    z, y = np.asarray(z, float), np.asarray(y, float)
    zc, med, lo, hi = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (z >= a) & (z < b) & np.isfinite(y)
        if sel.sum() < min_count:
            continue
        zc.append(np.median(z[sel]))
        p = np.percentile(y[sel], (16, 50, 84))
        lo.append(p[0]); med.append(p[1]); hi.append(p[2])
    return tuple(np.asarray(v) for v in (zc, med, lo, hi))


OBS_Z_EDGES = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.2])
SFRD_Z_EDGES = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.5, 10.0])


def _ratio(num, den):
    with np.errstate(invalid='ignore', divide='ignore'):
        r = np.asarray(num, float) / np.asarray(den, float)
    return np.where(np.isfinite(r) & (r > 0), r, np.nan)


RESID_OBS_STYLE = dict(color='0.3', ms=3.5, mec='white', mew=0.4, ls='none',
                       elinewidth=0.8, capsize=0, alpha=0.9, zorder=Z_OBS)
# Open markers for the individual observed clusters (figure 1), which overlap
OPEN_OBS_STYLE = dict(RESID_OBS_STYLE, ms=4.5, mfc='none', mec='0.25', mew=0.8)


def plot_1_fics(scans):
    """f_ICS against halo mass at z = 0, and against redshift for clusters.

    Every residual is against the published beta_clock = 2 run: lines are each
    run's SB-free median over beta_clock = 2's, bands each run's
    through-the-cut median over beta_clock = 2's.  The observations sit in the
    strips as grey markers, divided by beta_clock = 2 through the cut (the
    like-for-like model value): (a) the observed group and cluster medians,
    with their 16-84th range, at the centre of each mass class; (b) every
    observed cluster at its redshift.
    """
    print('Figure 1: intracluster fraction')
    # one column: f_ICS against halo mass at z = 0 on top, clusters against z below
    # enlarged: the lower panel carries a model legend and a two-column
    # observation legend side by side
    fig, main, res = resid_grid(2, 1, scale=(1.4, 1.2))
    ax_m, ax_z = main[:, 0]
    rx_m, rx_z = res[:, 0]

    # --- (a) f_ICS against halo mass at z = 0 ---
    #
    # Each run is drawn twice: the line is the SB-free total SAGE tracks, the
    # band beneath it is the same haloes re-measured through the Brough+24
    # correction, between its two edges.
    curves = {}
    for alpha, hdr, colour, s in scans:
        mvir, stars, ics = s['h_mvir'], s['h_mstar'], s['h_ics']
        tot = stars + ics
        ok = np.isfinite(mvir) & (mvir > 0) & (tot > 0)
        x = np.log10(mvir[ok])
        centres, pct = binned_percentiles(x, ics[ok] / tot[ok], MVIR_BINS)
        edges = []
        if APPLY_SB_CORRECTION:
            for gval in MODEL_SB_RECOVERY['edges']:
                _, pe = binned_percentiles(
                    x, observed_fics(ics[ok], stars[ok], gval), MVIR_BINS)
                edges.append(pe[1])
        curves[alpha] = (centres, pct[1], edges)

        st = line_style(alpha)
        if edges:
            good = np.isfinite(edges[0]) & np.isfinite(edges[1])
            ax_m.fill_between(centres[good], edges[0][good], edges[1][good],
                              color=colour, alpha=0.30, lw=0, zorder=Z_BAND,
                              label=alpha_label(alpha) + ', as observed')
        good = np.isfinite(pct[1])
        ax_m.plot(centres[good], pct[1][good], color=colour,
                  label=alpha_label(alpha), **st)

    ref_c, ref_med, ref_edges = curves.get(FIDUCIAL, (None, None, []))
    if ref_med is not None:
        for alpha, hdr, colour, s in scans:
            centres, med, edges = curves[alpha]
            rx_m.plot(centres, _ratio(med, ref_med), color=colour, **line_style(alpha))
            if edges and ref_edges:
                lo_r, hi_r = _ratio(edges[0], ref_edges[0]), _ratio(edges[1], ref_edges[1])
                good = np.isfinite(lo_r) & np.isfinite(hi_r)
                rx_m.fill_between(centres[good], lo_r[good], hi_r[good], color=colour,
                                  alpha=0.30, lw=0, zorder=Z_BAND)
    ref_cut = (0.5 * (ref_edges[0] + ref_edges[1])) if ref_edges else None

    # The observed ranges, as published.  The compilation has no halo masses, so
    # each class is drawn across the mass range it corresponds to; in the strip,
    # its median and 16-84th range over beta_clock = 2 through the cut.
    # Neutral greys: the runs' plasma colours take every hue.
    for lo, hi, scale, colour in ((13.0, 14.0, 'group', '#c7c7c7'),
                                  (14.0, 15.0, 'cluster', '#8c8c8c')):
        f = np.concatenate([o['f'] for o in icl_fraction_observations(scale)])
        f = f[f > 0]
        med, p16, p84 = np.median(f), *np.percentile(f, (16, 84))
        ax_m.fill_between([lo, hi], p16, p84, facecolor=colour, alpha=0.30,
                          edgecolor=colour, lw=1.0, zorder=2,
                          label=f'observed {scale}s, 16-84th')
        if ref_cut is not None:
            good = np.isfinite(ref_cut) & (ref_cut > 0)
            xm = 0.5 * (lo + hi)
            if good.sum() >= 2 and ref_c[good].min() <= xm:
                ref_at = np.interp(min(xm, ref_c[good].max()), ref_c[good], ref_cut[good])
                rx_m.errorbar([xm], [med / ref_at], xerr=[[xm - lo], [hi - xm]],
                              yerr=[[(med - p16) / ref_at], [(p84 - med) / ref_at]],
                              fmt='s' if scale == 'cluster' else 'o', **RESID_OBS_STYLE)

    ax_m.set_ylabel(r'$f_{\rm ICS} = m_{\rm ICS} / (m_{\rm ICS} + m_{\star,\rm halo})$')
    ax_m.set_xlim(11.5, 15.0)
    ax_m.set_ylim(0.0, 0.8)
    legend(ax_m, loc='upper left', ncol=2)
    ax_m.text(0.97, 0.96, rf'$z = {REDSHIFTS[Z0_SNAP]:.2f}$', transform=ax_m.transAxes, ha='right', va='top')
    resid_axis(rx_m, rf'/ $\beta_{{\rm clock}} = {FIDUCIAL:g}$', lim=(0.05, 2.0))
    rx_m.set_yticks([0.05, 0.1, 0.25, 0.5, 1, 2])
    rx_m.set_yticklabels(['0.05', '0.1', '0.25', '0.5', '1', '2'])
    rx_m.set_xlabel(r'$\log_{10}(M_{\rm vir}\ [{\rm M}_\odot])$')

    # --- (b) cluster f_ICS against redshift ---
    obs = icl_fraction_observations('cluster')
    zcurves = {}
    for alpha, hdr, colour, s in scans:
        med = s['fics_cl'][1]
        ok = np.isfinite(med)
        st = line_style(alpha)
        sb = s['fics_cl_sb'] if (APPLY_SB_CORRECTION and 'fics_cl_sb' in s) else None
        if sb is not None:
            g2 = np.isfinite(sb[0, 1]) & np.isfinite(sb[1, 1])
            ax_z.fill_between(REDSHIFTS[g2], sb[0, 1][g2], sb[1, 1][g2],
                              color=colour, alpha=0.30, lw=0, zorder=Z_BAND,
                              label=alpha_label(alpha) + ', as observed')
        ax_z.plot(REDSHIFTS[ok], med[ok], color=colour,
                  label=alpha_label(alpha), **st)
        zcurves[alpha] = (med, sb)

    if FIDUCIAL in zcurves:
        ref_med_z, ref_sb = zcurves[FIDUCIAL]
        for alpha, hdr, colour, s in scans:
            med, sb = zcurves[alpha]
            r = _ratio(med, ref_med_z)
            g = np.isfinite(r)
            rx_z.plot(REDSHIFTS[g], r[g], color=colour, **line_style(alpha))
            if sb is not None and ref_sb is not None:
                lo_r, hi_r = _ratio(sb[0, 1], ref_sb[0, 1]), _ratio(sb[1, 1], ref_sb[1, 1])
                g = np.isfinite(lo_r) & np.isfinite(hi_r)
                rx_z.fill_between(REDSHIFTS[g], lo_r[g], hi_r[g], color=colour,
                                  alpha=0.30, lw=0, zorder=Z_BAND)
        # each observed cluster over beta_clock = 2 through the cut at its redshift
        if ref_sb is not None:
            cut_mid = 0.5 * (ref_sb[0, 1] + ref_sb[1, 1])
            g = np.isfinite(cut_mid) & (cut_mid > 0)
            if g.sum() >= 2:
                zz, mm = REDSHIFTS[g][::-1], cut_mid[g][::-1]
                for i, o in enumerate(obs):
                    inside = (o['z'] >= zz.min()) & (o['z'] <= zz.max()) & (o['f'] > 0)
                    if inside.any():
                        rx_z.errorbar(o['z'][inside],
                                      o['f'][inside] / np.interp(o['z'][inside], zz, mm),
                                      fmt=_OBS_MARKERS[i % len(_OBS_MARKERS)],
                                      **OPEN_OBS_STYLE)

    for i, o in enumerate(obs):
        ax_z.plot(o['z'], o['f'], _OBS_MARKERS[i % len(_OBS_MARKERS)],
                  ms=5.5, mfc='none', mec='0.25', mew=0.9, ls='none',
                  zorder=Z_OBS, label=o['label'])

    ax_z.set_ylabel(r'$f_{\rm ICS}$, hosts with $M_{\rm vir} > 10^{14}\,{\rm M}_\odot$')
    ax_z.set_xlim(0.0, 2.2)
    ax_z.set_ylim(0.0, 1.0)
    # two legends: the model runs top left, the observational samples top right
    handles, labels = ax_z.get_legend_handles_labels()
    obs_labels = {o['label'] for o in obs}
    model = [(h, l) for h, l in zip(handles, labels) if l not in obs_labels]
    data = [(h, l) for h, l in zip(handles, labels) if l in obs_labels]
    leg_model = legend(ax_z, loc='upper left', handles=[h for h, _ in model],
                       labels=[l for _, l in model])
    ax_z.add_artist(leg_model)
    legend(ax_z, loc='upper right', ncol=2, handles=[h for h, _ in data],
           labels=[l for _, l in data])
    resid_axis(rx_z, rf'/ $\beta_{{\rm clock}} = {FIDUCIAL:g}$', lim=(0.05, 2.0))
    rx_z.set_yticks([0.05, 0.1, 0.25, 0.5, 1, 2])
    rx_z.set_yticklabels(['0.05', '0.1', '0.25', '0.5', '1', '2'])
    rx_z.set_xlabel(r'$z$')

    hide_bottom_ticklabel(main, res)
    fig.tight_layout()
    save_figure(fig, 'fig1_fICS')


# ============ FIGURE 2: BULGE-TO-TOTAL ============

def _mass_weighted_bt(m, b, bins, min_count=MIN_COUNT):
    """Summed bulge mass over summed stellar mass per bin of log m* -- the bulge
    mass fraction as Moffett+16 measure it, not a median of B/T."""
    x = np.log10(m)
    out = np.full(len(bins) - 1, np.nan)
    for i in range(len(bins) - 1):
        sel = (x >= bins[i]) & (x < bins[i + 1])
        if sel.sum() >= min_count:
            out[i] = np.clip(b[sel], 0, None).sum() / m[sel].sum()
    return 0.5 * (bins[:-1] + bins[1:]), out


def _draw_bulge_to_total(ax_bt, rx_bt, scans):
    """Bulge mass fraction (summed bulge / summed stellar mass) against stellar
    mass, like for like with Moffett+16.  Residual: each run over beta_clock = 2;
    Moffett+16 in the strip as markers, over beta_clock = 2 at the observed
    masses, with their 1-sigma range."""
    obs = load_moffett16_bulge_fraction()
    meds = {}
    for alpha, hdr, colour, s in scans:
        m, b = s['g_mstar'], s['g_bulge']
        ok = (m > 1e8) & np.isfinite(b)
        centres, frac = _mass_weighted_bt(m[ok], b[ok], MSTAR_BINS)
        good = np.isfinite(frac)
        ax_bt.plot(centres[good], frac[good], color=colour,
                   label=alpha_label(alpha), **line_style(alpha))
        meds[alpha] = (centres, frac)

    if FIDUCIAL in meds:
        cref, ref = meds[FIDUCIAL]
        for alpha, hdr, colour, s in scans:
            r = _ratio(meds[alpha][1], ref)
            g = np.isfinite(r)
            rx_bt.plot(cref[g], r[g], color=colour, **line_style(alpha))
        good = np.isfinite(ref) & (ref > 0)
        if obs is not None and good.sum() >= 2:
            inside = (obs['logm'] >= cref[good].min()) & (obs['logm'] <= cref[good].max())
            r_at = np.interp(obs['logm'][inside], cref[good], ref[good])
            y = obs['bt'][inside] / r_at
            rx_bt.errorbar(obs['logm'][inside], y,
                           yerr=[np.clip(y - obs['lo'][inside] / r_at, 0, None),
                                 np.clip(obs['hi'][inside] / r_at - y, 0, None)],
                           fmt='o', **RESID_OBS_STYLE)

    if obs is not None:
        ax_bt.fill_between(obs['logm'], obs['lo'], obs['hi'], color='0.75',
                           alpha=0.5, lw=0, zorder=1)
        ax_bt.plot(obs['logm'], obs['bt'], 'o', color='0.3', ms=4.5,
                   mec='white', mew=0.5, ls='none', zorder=Z_OBS,
                   label='Moffett+16 (GAMA)')

    ax_bt.set_ylabel(r'bulge mass fraction, $\sum m_{\rm bulge} / \sum m_\star$')
    ax_bt.set_xlim(8.0, 12.2)
    ax_bt.set_ylim(0.0, 1.05)
    ax_bt.text(0.03, 0.97, rf'$z = {REDSHIFTS[Z0_SNAP]:.2f}$', transform=ax_bt.transAxes, ha='left', va='top')
    legend(ax_bt, loc='upper left', bbox_to_anchor=(0.0, 0.90))
    resid_axis(rx_bt, rf'/ $\beta_{{\rm clock}} = {FIDUCIAL:g}$', lim=(0.3, 30.0))
    rx_bt.set_yticks([0.3, 1, 3, 10, 30])
    rx_bt.set_yticklabels(['0.3', '1', '3', '10', '30'])
    rx_bt.set_xlabel(r'$\log_{10}(m_\star\ [{\rm M}_\odot])$')


def _draw_sfrd(ax, rx, scans):
    """Cosmic SFR density against redshift.  Residual: each run over
    beta_clock = 2 in dex; the observed compilation in the strip as markers,
    minus beta_clock = 2 at each observed redshift, with their error bars."""
    observations = load_sfrd_observations()
    logs = {}
    for alpha, hdr, colour, s in scans:
        sfrd = s['sfrd']
        ok = sfrd > 0
        ax.plot(REDSHIFTS[ok], np.log10(sfrd[ok]), color=colour,
                label=alpha_label(alpha), **line_style(alpha))
        logs[alpha] = np.where(ok, np.log10(np.where(ok, sfrd, 1.0)), np.nan)

    for o in observations:
        ax.errorbar(o['z'], o['y'], yerr=[o['el'], o['eu']], fmt=o['marker'],
                    color=o['colour'], ms=3.5, mec='white', mew=0.4, lw=0.8,
                    ls='none', alpha=0.85, zorder=Z_OBS, label=o['label'])

    if FIDUCIAL in logs:
        ref = logs[FIDUCIAL]
        for alpha, hdr, colour, s in scans:
            d = logs[alpha] - ref
            g = np.isfinite(d)
            rx.plot(REDSHIFTS[g], d[g], color=colour, **line_style(alpha))
        g = np.isfinite(ref)
        zz, rr = REDSHIFTS[g][::-1], ref[g][::-1]
        for o in observations:
            inside = (o['z'] >= zz.min()) & (o['z'] <= zz.max())
            if inside.any():
                rx.errorbar(o['z'][inside], o['y'][inside] - np.interp(o['z'][inside], zz, rr),
                            yerr=[o['el'][inside], o['eu'][inside]], fmt=o['marker'],
                            **RESID_OBS_STYLE)

    ax.set_ylabel(r'$\log_{10}(\rho_{\rm SFR}\ [{\rm M}_\odot\,{\rm yr}^{-1}\,{\rm Mpc}^{-3}])$')
    ax.set_xlim(0.0, 8.0)
    ax.set_ylim(-3.0, -0.4)
    handles, labels = ax.get_legend_handles_labels()
    obs_labels = {o['label'] for o in observations}
    model = [(h, l) for h, l in zip(handles, labels) if l not in obs_labels]
    data = [(h, l) for h, l in zip(handles, labels) if l in obs_labels]
    ax.add_artist(legend(ax, loc='lower left', handles=[h for h, _ in model],
                         labels=[l for _, l in model]))
    legend(ax, loc='upper right', handles=[h for h, _ in data],
           labels=[l for _, l in data])
    resid_axis(rx, rf'$-$ $\beta_{{\rm clock}} = {FIDUCIAL:g}$ [dex]', kind='dex', lim=(-0.8, 0.8))
    rx.set_xlabel(r'$z$')


# ============ FIGURE 4: THE ICS MASS FUNCTION ============

def plot_4_ics_mass_function(scans):
    """The ICS mass function in a grid of redshifts, split by host halo mass.

    The figure of plotting/random_plotting_scripts/ICS_paper_plots.py (plot 2):
    every central holding ICS, one panel per target redshift, coloured by host
    halo mass.  The all-halo total is not drawn (it is still printed by
    report_ics_mass_function).  Colour is spent on halo mass, so beta_clock is
    carried by line style -- solid for the published value.  No shading: with
    three runs overlaid it hides the lines it sits under.  There are no data for
    the ICS mass function, so each panel's residual strip is the run over the
    published beta_clock = 2 run.
    """
    print('Figure 4: ICS mass function grid')
    _ics_mass_function_grid(scans, MF_TARGET_Z, 2, 'fig4_ics_mass_function')


def plot_16_ics_mass_function_low_z(scans):
    """Figure 4 as one column at z = 0.02, 0.49, 1.03 and 2.03."""
    print('Figure 16: ICS mass function, low-redshift column')
    _ics_mass_function_grid(scans, MF_TARGET_Z_LOW, 1, 'fig16_ics_mass_function_low_z')


def _ics_mass_function_grid(scans, target_z, max_cols, name):
    """The ICS mass function at the snapshots nearest *target_z*, at most
    *max_cols* panels across; see plot_4_ics_mass_function."""
    snaps = []
    for zt in target_z:
        k = _snap_near(zt) if zt > 0 else Z0_SNAP
        if k not in snaps:
            snaps.append(k)
    ncols = min(max_cols, len(snaps))
    nrows = -(-len(snaps) // ncols)
    fig, axes, raxes = resid_grid(nrows, ncols)
    centres = 0.5 * (MF_ICS_EDGES[:-1] + MF_ICS_EDGES[1:])
    colours = MF_HALO_COLOURS          # the halo-mass bins only; no all-halo total
    ref_scan = next((s for a, _, _, s in scans if a == FIDUCIAL), None)

    for pi, snap in enumerate(snaps):
        ax, rax = axes.flat[pi], raxes.flat[pi]
        for alpha, hdr, colour, s in scans:
            if 'mf_counts' not in s:
                continue
            vol = float(s['volume'][0])
            for b, c in enumerate(colours):
                phi = s['mf_counts'][snap, b] / vol / MF_ICS_BW
                good = phi > 0
                if not good.any():
                    continue
                kw = dict(color=c, ls=alpha_ls(alpha),
                          lw=2.0 if alpha == FIDUCIAL else 1.5, alpha=0.9,
                          zorder=Z_LINE + (1 if alpha == FIDUCIAL else 0))
                ax.plot(centres[good], phi[good], **kw)
                # no data for the ICS mass function: residual against the published run
                if ref_scan is not None and alpha != FIDUCIAL:
                    vref = float(ref_scan['volume'][0])
                    phi_ref = ref_scan['mf_counts'][snap, b] / vref / MF_ICS_BW
                    r = _ratio(phi, phi_ref)
                    g = np.isfinite(r)
                    if g.any():
                        rax.plot(centres[g], np.log10(r[g]), **kw)
        ax.text(0.95, 0.95, rf'$z = {REDSHIFTS[snap]:.2f}$', transform=ax.transAxes,
                ha='right', va='top')
        ax.set_yscale('log')
        ax.set_xlim(5.5, 12.5)
        ax.set_ylim(1e-6, 1e-2)
        ax.set_xticks([6, 7, 8, 9, 10, 11, 12])
        resid_axis(rax, rf'$\log_{{10}}(\phi / \phi_{{{FIDUCIAL:g}}})$',
                   kind='dex', lim=(-2.0, 2.0))
        if pi % ncols == 0:
            ax.set_ylabel(r'$\phi\ [{\rm Mpc}^{-3}\,{\rm dex}^{-1}]$')
        else:
            ax.tick_params(labelleft=False)
            rax.tick_params(labelleft=False)
            rax.set_ylabel('')
        if pi // ncols == nrows - 1:
            rax.set_xlabel(r'$\log_{10}(m_{\rm ICS}\ [{\rm M}_\odot])$')
    for pi in range(len(snaps), axes.size):
        axes.flat[pi].set_visible(False)
        raxes.flat[pi].set_visible(False)
    hide_bottom_ticklabel(axes, raxes)

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, lw=2.0, label=l)
               for c, l in zip(colours, MF_HALO_LABELS)]
    handles += [Line2D([], [], color='0.3', lw=2.0, ls=alpha_ls(a), label=alpha_label(a))
                for a, *_ in scans]
    # below the grid, as ICS_paper_plots.py has it: inside any panel it sits on data
    leg_cols = len(handles) if nrows == 1 else (2 if ncols == 1 else 4)
    leg_rows = -(-len(handles) // leg_cols)
    bottom = (0.25 + 0.18 * leg_rows) / fig.get_figheight()
    fig.tight_layout(rect=(0, bottom, 1, 1))
    fig.legend(handles, [h.get_label() for h in handles], loc='lower center',
               ncol=leg_cols, frameon=False, fontsize=8)
    save_figure(fig, name)


# z ~ 0 total stellar mass functions for figure 14, converted exactly as
# plotting/paper_plots.py converts them for the SAGE26 model paper, so the two
# papers show the same data the same way.  Both are Chabrier already.
#   Baldry+08 (SDSS, z ~ 0): mass in h^-2 units and phi in h^3 Mpc^-3 dex^-1;
#       masses go to the model's h and take paper_plots' -0.22 dex offset.
#   Thorne+21 (GAMA/DEVILS, z = 0.064): linear phi with 16th/84th percentiles,
#       scaled by (0.7/h)^3 as paper_plots does.
_BALDRY08 = np.array([
    (7.05, 1.3531e-01, 6.0741e-02), (7.15, 1.3474e-01, 6.0109e-02), (7.25, 2.0971e-01, 7.7965e-02),
    (7.35, 1.7161e-01, 3.1841e-02), (7.45, 2.1648e-01, 5.7832e-02), (7.55, 2.1645e-01, 3.9988e-02),
    (7.65, 2.0837e-01, 4.8713e-02), (7.75, 2.0402e-01, 7.0061e-02), (7.85, 1.5536e-01, 3.9182e-02),
    (7.95, 1.5232e-01, 2.6824e-02), (8.05, 1.5067e-01, 4.8824e-02), (8.15, 1.3032e-01, 2.1892e-02),
    (8.25, 1.2545e-01, 3.5526e-02), (8.35, 9.8472e-02, 2.7181e-02), (8.45, 8.7194e-02, 2.8345e-02),
    (8.55, 7.0758e-02, 2.0808e-02), (8.65, 5.8190e-02, 1.3359e-02), (8.75, 5.6057e-02, 1.3512e-02),
    (8.85, 5.1380e-02, 1.2815e-02), (8.95, 4.4206e-02, 9.6866e-03), (9.05, 4.1149e-02, 1.0169e-02),
    (9.15, 3.4959e-02, 6.7898e-03), (9.25, 3.3111e-02, 8.3704e-03), (9.35, 3.0138e-02, 4.7741e-03),
    (9.45, 2.6692e-02, 5.5029e-03), (9.55, 2.4656e-02, 4.4359e-03), (9.65, 2.2885e-02, 3.7915e-03),
    (9.75, 2.1849e-02, 3.9812e-03), (9.85, 2.0383e-02, 3.2930e-03), (9.95, 1.9929e-02, 2.9370e-03),
    (10.05, 1.8865e-02, 2.4624e-03), (10.15, 1.8136e-02, 2.5208e-03), (10.25, 1.7657e-02, 2.4217e-03),
    (10.35, 1.6616e-02, 2.2784e-03), (10.45, 1.6114e-02, 2.1783e-03), (10.55, 1.4366e-02, 1.8819e-03),
    (10.65, 1.2588e-02, 1.8249e-03), (10.75, 1.1372e-02, 1.4436e-03), (10.85, 9.1213e-03, 1.5816e-03),
    (10.95, 6.1125e-03, 9.6735e-04), (11.05, 4.3923e-03, 9.6254e-04), (11.15, 2.5463e-03, 5.0038e-04),
    (11.25, 1.4298e-03, 4.2816e-04), (11.35, 6.4867e-04, 1.6439e-04), (11.45, 2.8294e-04, 9.9799e-05),
    (11.55, 1.0617e-04, 4.9085e-05), (11.65, 3.2702e-05, 2.4546e-05), (11.75, 1.2571e-05, 1.2571e-05),
    (11.85, 8.4589e-06, 8.4589e-06), (11.95, 7.4764e-06, 7.4764e-06),
])


def load_smf_z0_observations():
    """Observed z ~ 0 stellar mass functions as (label, marker, log m, log phi, err lo, err hi)."""
    out = []
    h = HUBBLE_H
    b = _BALDRY08
    log_m = np.log10(10 ** b[:, 0] / h / h) - 0.22
    phi_c, phi_u, phi_l = b[:, 1] * h ** 3, (b[:, 1] + b[:, 2]) * h ** 3, (b[:, 1] - b[:, 2]) * h ** 3
    ok = phi_l > 0
    lp = np.log10(phi_c[ok])
    out.append(('Baldry+08', 'o', log_m[ok], lp, lp - np.log10(phi_l[ok]),
                np.log10(phi_u[ok]) - lp))
    path = os.path.join(OBS_DIR, 'smf', 'Thorne21', 'SMFvals_z0.064.csv')
    if os.path.exists(path):
        d = np.genfromtxt(path, delimiter=',', skip_header=1)
        m, phi, p16, p84 = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
        ok = np.isfinite(m) & (phi > 0) & (p16 > 0) & (p84 > 0)
        fac = (0.7 / h) ** 3
        lp = np.log10(phi[ok] * fac)
        out.append(('Thorne+21', 's', m[ok], lp, lp - np.log10(p16[ok] * fac),
                    np.log10(p84[ok] * fac) - lp))
    return out


def plot_14_stellar_mass_function(scans):
    """The stellar mass function at the analysis snapshot, one line per beta_clock,
    against the z ~ 0 observations."""
    print('Figure 14: stellar mass function')
    fig, ax = panel_grid(1, 1)
    for alpha, hdr, colour, s in scans:
        vol = float(s['volume'][0])
        m = s['g_mstar']
        x, phi = mass_function(np.log10(m[m > 0]), vol, mass_range=(8.0, 12.5))
        ax.plot(x, np.log10(phi), color=colour, label=alpha_label(alpha),
                **line_style(alpha))
    for label, marker, lm, lp, elo, ehi in load_smf_z0_observations():
        ax.errorbar(lm, lp, yerr=[elo, ehi], fmt=marker, color='0.35', ms=4.0,
                    mec='white', mew=0.4, lw=0.8, ls='none', alpha=0.9,
                    zorder=Z_OBS, label=label)
    ax.set_xlabel(r'$\log_{10}(m_\star\ [{\rm M}_\odot])$')
    ax.set_ylabel(r'$\log_{10}(\phi\ [{\rm Mpc}^{-3}\,{\rm dex}^{-1}])$')
    ax.set_xlim(8.0, 12.4)
    ax.set_ylim(-6.0, -1.0)
    legend(ax, loc='lower left')
    fig.tight_layout()
    save_figure(fig, 'fig14_stellar_mass_function')


def _load_bh_bulge_obs():
    """Black hole against spheroid stellar mass: Sahu+19 (early types) and
    Davis+19 (spirals), as tabulated in data/bh.  Used as published."""
    out = []
    for name, label, cols, marker in (('MBH_Mstar_Sahu2019.dat', 'Sahu+19 (E/S0)', (0, 2), 'o'),
                                      ('MBH_Mstar_Davis2019.dat', 'Davis+19 (spirals)', (0, 3), 's')):
        path = os.path.join(OBS_DIR, 'bh', name)
        if os.path.exists(path):
            d = np.loadtxt(path)
            out.append((label, marker, d[:, cols[1]], d[:, cols[0]]))
    return out


# BCG stellar-mass growth along the main progenitor, as published.  Each study
# follows the progenitors of clusters of a given z = 0 mass and quotes one
# growth factor between two redshifts; both use a Chabrier IMF.
#   Lidman+12 (MNRAS 427, 550), abstract and sec. 5.3.4: "the stellar mass of
#     BCGs increases by a factor of 1.8 +/- 0.3 between z = 0.9 and z = 0.2";
#     the matched comparison (their Table 8) has a median z = 0 cluster mass
#     M200c = 0.84e15 Msun.  BCG masses in 16 kpc-diameter apertures.
#   Lin+13 (ApJ 771, 61), abstract: progenitors of z = 0 clusters of
#     M200c = (2.5-4.5)e14 Msun; "between z=1.5 and 0.5, the BCGs have grown in
#     stellar mass by a factor of 2.3" (no uncertainty quoted).  32 kpc apertures.
# The model BCG mass is total (no aperture), so it is an upper bound on the
# aperture growth if the outskirts grow fastest.
# 'm200c' is the z = 0 window the model clusters are drawn from: Lin's range as
# quoted; for Lidman, +/- 0.15 dex about their median (our choice, as they
# quote a matched distribution, not a range).
BCG_GROWTH_OBS = (
    dict(label='Lidman+12', z_hi=0.9, z_lo=0.2, factor=1.8, err=0.3,
         m200c=(0.84e15 / 10 ** 0.15, 0.84e15 * 10 ** 0.15)),
    dict(label='Lin+13', z_hi=1.5, z_lo=0.5, factor=2.3, err=None,
         m200c=(2.5e14, 4.5e14)),
)
# The observed z = 0 masses are M200c; SAGE26 hosts are Mvir.  Converted with an
# NFW profile of concentration c200c = 5, the profile Lidman+12 assume.
BCG_GROWTH_C200C = 5.0
# Clusters needed in a window before the model growth is drawn.  Lowered from
# KIMMIG_MIN (5) for now so microUchuu draws something; raise it for the paper.
BCG_GROWTH_MIN = 2


def _m200c_to_mvir_factor(c200c=BCG_GROWTH_C200C):
    """Mvir / M200c at z = 0 for an NFW halo of concentration *c200c*."""
    from colossus.cosmology import cosmology
    from colossus.halo import mass_defs
    cosmology.setCosmology('sage', params=dict(flat=True, H0=100 * HUBBLE_H,
                           Om0=OMEGA_M, Ob0=0.0486, sigma8=0.8159, ns=0.9667))
    m200 = 1e14
    mvir, _, _ = mass_defs.changeMassDefinition(m200, c200c, 0.0, '200c', 'vir')
    return mvir / m200


def bcg_growth(K, z_hi, z_lo, mvir_lo, mvir_hi, nboot=500, seed=1):
    """Median main-progenitor BCG mass at z_lo over that at z_hi, for z = 0
    clusters with mvir_lo <= Mvir < mvir_hi, with its bootstrap 16-84th range.
    A ratio of medians, as the observations compare sample medians."""
    i_hi, i_lo = _snap_near(z_hi), _snap_near(z_lo)
    m0 = K['mhost'][:, Z0_SNAP]
    a, b = K['m_bcg'][:, i_lo], K['m_bcg'][:, i_hi]
    sel = (m0 >= mvir_lo) & (m0 < mvir_hi) & (a > 0) & (b > 0)
    n = int(sel.sum())
    if n < BCG_GROWTH_MIN:
        return np.nan, (np.nan, np.nan), n, (REDSHIFTS[i_hi], REDSHIFTS[i_lo])
    a, b = a[sel], b[sel]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(nboot, n))
    boot = np.median(a[idx], axis=1) / np.median(b[idx], axis=1)
    return (np.median(a) / np.median(b), tuple(np.percentile(boot, (16, 84))), n,
            (REDSHIFTS[i_hi], REDSHIFTS[i_lo]))


def _draw_bcg_growth(ax, kimmig, report=True):
    """BCG main-progenitor growth factor against Lidman+12 and Lin+13: one
    column per study, the observed factor in grey and each run beside it."""
    conv = _m200c_to_mvir_factor()
    say = print if report else (lambda *a, **k: None)
    say(f'  BCG growth (Mvir / M200c = {conv:.3f} for c200c = {BCG_GROWTH_C200C:g}):')
    offsets = np.linspace(-0.22, 0.22, len(kimmig) + 1)
    for j, o in enumerate(BCG_GROWTH_OBS):
        lo, hi = o['m200c'][0] * conv, o['m200c'][1] * conv
        ax.errorbar([j + offsets[0]], [o['factor']],
                    yerr=None if o['err'] is None else [[o['err']], [o['err']]],
                    fmt='o', color='0.3', ms=6.0, mec='white', mew=0.5, lw=1.0,
                    capsize=0, zorder=Z_OBS, label='observed' if j == 0 else None)
        n_min = None
        for k, (alpha, hdr, colour, K) in enumerate(kimmig):
            g, (g16, g84), n, (zh, zl) = bcg_growth(K, o['z_hi'], o['z_lo'], lo, hi)
            if not np.isfinite(g):
                n_min = n if n_min is None else min(n_min, n)
            say(f'    {o["label"]:10s} beta_clock = {alpha:<4g} m_BCG(z={zl:.2f}) / '
                  f'm_BCG(z={zh:.2f}) = {g:.2f} [{g16:.2f}-{g84:.2f}], N = {n} '
                  f'(observed {o["factor"]:g})')
            if np.isfinite(g):
                ax.errorbar([j + offsets[k + 1]], [g], yerr=[[g - g16], [g84 - g]],
                            fmt='s', color=colour, ms=5.5, mec='white', mew=0.4,
                            lw=1.2, capsize=0, zorder=Z_LINE,
                            label=alpha_label(alpha) if j == 0 else None)
        # say so on the panel rather than leave the column silently empty
        if n_min is not None:
            ax.text(j + 0.1, 1.1, f'too few clusters\nin this mass range\n(N = {n_min})',
                    ha='center', va='bottom', fontsize=6.5, color='0.4')
    ax.axhline(1.0, color='0.6', lw=0.7, zorder=1)
    ax.set_xticks(range(len(BCG_GROWTH_OBS)))
    ax.set_xticklabels([f'{o["label"]}\n$z = {o["z_hi"]:g} \\rightarrow {o["z_lo"]:g}$'
                        for o in BCG_GROWTH_OBS])
    ax.set_xlim(-0.6, len(BCG_GROWTH_OBS) - 0.4)
    ax.set_ylim(0.8, 4.0)
    ax.set_ylabel(r'BCG growth, $m_{\star,\rm BCG}(z_{\rm low}) / m_{\star,\rm BCG}(z_{\rm high})$')
    legend(ax, loc='upper left')


def plot_15_appendix(data):
    """Appendix: how far each run moves from the calibrated model.  Left, the
    stellar mass function against the z ~ 0 data; centre, black hole mass against
    bulge mass against Sahu+19 and Davis+19; right, the growth of the BCG's main
    progenitor against Lidman+12 and Lin+13.  *data* is (snapshot scans, cluster
    main-branch histories); the right panel is left out without the histories."""
    print('Figure 15: appendix -- stellar mass function, black hole-bulge relation, BCG growth')
    _appendix(data, column=False, name='fig15_appendix_smf_bh')


def plot_17_appendix_column(data):
    """Figure 15 as one column: the panels stacked top to bottom."""
    print('Figure 17: appendix, one column')
    _appendix(data, column=True, name='fig17_appendix_smf_bh_column', report=False)


def _appendix(data, column, name, report=True):
    """The appendix panels in a row, or stacked in one column; see plot_15_appendix."""
    scans, kimmig = data
    n = 3 if kimmig else 2
    fig, axes = panel_grid(n, 1) if column else panel_grid(1, n)
    axes = np.ravel(axes)
    ax, bx = axes[0], axes[1]
    for alpha, hdr, colour, s in scans:
        vol = float(s['volume'][0])
        m = s['g_mstar']
        x, phi = mass_function(np.log10(m[m > 0]), vol, mass_range=(8.0, 12.5))
        ax.plot(x, np.log10(phi), color=colour, label=alpha_label(alpha), **line_style(alpha))
    for label, marker, lm, lp, elo, ehi in load_smf_z0_observations():
        ax.errorbar(lm, lp, yerr=[elo, ehi], fmt=marker, color='0.35', ms=4.0,
                    mec='white', mew=0.4, lw=0.8, ls='none', alpha=0.9,
                    zorder=Z_OBS, label=label)
    ax.set_xlabel(r'$\log_{10}(m_\star\ [{\rm M}_\odot])$')
    ax.set_ylabel(r'$\log_{10}(\phi\ [{\rm Mpc}^{-3}\,{\rm dex}^{-1}])$')
    ax.set_xlim(8.0, 12.4)
    ax.set_ylim(-6.0, -1.0)
    legend(ax, loc='lower left')

    if 'g_bh' in scans[0][3]:
        bins = np.arange(8.5, 12.51, 0.25)
        for alpha, hdr, colour, s in scans:
            b, bh = s['g_bulge'], s['g_bh']
            ok = (b > 0) & (bh > 0)
            c, pct = binned_percentiles(np.log10(b[ok]), np.log10(bh[ok]), bins)
            g = np.isfinite(pct[1])
            if alpha == FIDUCIAL:
                bx.fill_between(c[g], pct[0][g], pct[2][g], color=colour, alpha=0.18,
                                lw=0, zorder=Z_BAND)
            bx.plot(c[g], pct[1][g], color=colour, label=alpha_label(alpha), **line_style(alpha))
    for label, marker, lsph, lbh in _load_bh_bulge_obs():
        bx.plot(lsph, lbh, marker, color='0.35', ms=4.0, mec='white', mew=0.4,
                ls='none', alpha=0.9, zorder=Z_OBS, label=label)
    bx.set_xlabel(r'$\log_{10}(m_{\rm bulge}\ [{\rm M}_\odot])$')
    bx.set_ylabel(r'$\log_{10}(m_{\rm BH}\ [{\rm M}_\odot])$')
    bx.set_xlim(8.5, 12.5)
    bx.set_ylim(5.0, 10.5)
    bx.text(0.97, 0.05, rf'$z = {REDSHIFTS[Z0_SNAP]:.2f}$', transform=bx.transAxes,
            ha='right', va='bottom')
    legend(bx, loc='upper left')
    if kimmig:
        _draw_bcg_growth(axes[2], kimmig, report=report)
    fig.tight_layout()
    save_figure(fig, name)


def report_ics_mass_function(scans):
    """Numbers behind figure 4: per redshift, halo bin and alpha."""
    print()
    print('-' * 78)
    print('ICS mass function (figure 4): haloes with ICS, total ICS density, and the')
    print('ICS mass where phi falls to 1e-4 / Mpc^3 / dex, per host-mass bin')
    names = ('1e10-1e12', '1e12-1e14', '>1e14', 'all')
    centres = 0.5 * (MF_ICS_EDGES[:-1] + MF_ICS_EDGES[1:])
    for zt in MF_TARGET_Z:
        snap = _snap_near(zt) if zt > 0 else Z0_SNAP
        print(f'  z = {REDSHIFTS[snap]:.2f} (Snap_{snap})')
        for alpha, hdr, colour, s in scans:
            if 'mf_counts' not in s:
                continue
            vol = float(s['volume'][0])
            parts = []
            for b, nm in enumerate(names):
                cnt = s['mf_counts'][snap, b]
                phi = cnt / vol / MF_ICS_BW
                above = np.flatnonzero(phi >= 1e-4)
                knee = centres[above[-1]] if above.size else np.nan
                parts.append(f'{nm} N={int(cnt.sum()):>7,d} M(1e-4)={_f(knee, ".2f", 5)}')
            print(f'    a = {alpha:<5g} rho_ICS = {s["rho_ics"][snap]:.3e} Msun/Mpc^3 | '
                  + ' | '.join(parts))
    print('-' * 78)


# ============ FIGURE 5: THE CLOCK ============

def _draw_merger_clock(ax_df, rx_df, runs):
    """The clock each satellite is given at infall, t_clock = beta_clock t_friction.

    beta_clock shifts the whole distribution rigidly in the log; the dotted line
    is the age of the universe, the ceiling any clock has to beat to expire.
    There are no data for the clock, so the residual is each run's density over
    the published beta_clock = 2 run.  The lifetimes and the routing that follow
    are printed by report_clock() and report_clock_detail().
    """
    log_bins = np.linspace(-3.0, 2.0, 60)
    bc = 0.5 * (log_bins[:-1] + log_bins[1:])

    dens = {}
    for alpha, hdr, colour, ev in runs or ():
        ok = ev['clock_known']
        t = ev['t_df'][ok & (ev['t_df'] > 0)]
        if not t.size:
            continue
        h, _ = np.histogram(np.log10(t), bins=log_bins, density=True)
        dens[alpha] = h
        ax_df.step(bc, h, where='mid', color=colour,
                   label=alpha_label(alpha) + rf'  (median {np.median(t):.2f} Gyr)',
                   **line_style(alpha))

    if FIDUCIAL in dens:
        for alpha, hdr, colour, ev in runs:
            if alpha == FIDUCIAL or alpha not in dens:
                continue
            r = _ratio(dens[alpha], dens[FIDUCIAL])
            g = np.isfinite(r)
            rx_df.step(bc[g], np.log10(r[g]), where='mid', color=colour, **line_style(alpha))
    if not dens:
        ax_df.text(0.5, 0.5, 'no disruption log\n(run SAGE with SAGE_DISRUPT_LOG set)',
                   transform=ax_df.transAxes, ha='center', va='center', fontsize=8)

    ax_df.set_ylabel(r'probability density per dex')
    ax_df.set_xlim(-2.5, 1.8)
    ax_df.axvline(np.log10(AGE_NOW), color='0.5', lw=0.9, ls=':', zorder=1)
    ax_df.text(np.log10(AGE_NOW) - 0.04, ax_df.get_ylim()[1] * 0.97,
               'age of the universe', rotation=90, ha='right', va='top',
               fontsize=6.5, color='0.4')
    if dens:
        legend(ax_df, loc='upper left')
    rx_df.axvline(np.log10(AGE_NOW), color='0.5', lw=0.9, ls=':', zorder=1)
    resid_axis(rx_df, rf'$\log_{{10}}$(density / $\beta_{{\rm clock}} = {FIDUCIAL:g}$)',
               kind='dex', lim=(-2.0, 2.0))
    rx_df.set_xlabel(r'$\log_{10}(t_{\rm clock}\ [{\rm Gyr}])$')


def plot_2_galaxies_and_clock(data):
    """Three columns, each with its residual strip: (a) B/T against Moffett+16,
    (b) the cosmic SFR density against the compilation, (c) the merger clock
    against beta_clock = 2.  *data* is (snapshot scans, disruption logs); the
    clock panel says so if the logs are missing."""
    scans, disrupt = data
    print('Figure 2: bulge-to-total, SFR density and the merger clock')
    fig, main, res = resid_grid(1, 3)
    _draw_bulge_to_total(main[0, 0], res[0, 0], scans)
    _draw_sfrd(main[0, 1], res[0, 1], scans)
    _draw_merger_clock(main[0, 2], res[0, 2], disrupt)
    hide_bottom_ticklabel(main, res)
    fig.tight_layout()
    save_figure(fig, 'fig2_bt_sfrd_clock')


# ============ FIGURE 6: FIGURE 1 AS RAW DATA ============

def plot_6_fics_scatter(scans):
    """Figure 1 as raw data: one marker per halo, no medians, nothing binned.

    Only the model, only through the surface-brightness cut, and only the
    haloes the observations actually sample.  Groups and clusters are separated
    by both marker and colour -- triangles on an orange ramp and circles on a
    blue one -- with shade still reading as alpha inside either class.  Every
    halo at every snapshot is drawn, so the cluster end shows what it really
    is: a handful of objects per run in a 100 Mpc/h box.

    The correction uses one edge of the offset band, named in each panel title.  SCATTER_G picks
    which; the default is the stronger cut, the one most favourable to the
    model, so an overproduction seen here is not an artefact of a lenient
    choice.
    """
    print('Figure 6: f_ICS through the cut, halo by halo')
    if not APPLY_SB_CORRECTION:
        print('  APPLY_SB_CORRECTION is off -- nothing to draw')
        return
    if 'gc_z' not in scans[0][3]:
        print('  scan cache predates this figure -- rerun with --refresh')
        return

    gval = min(MODEL_SB_RECOVERY['edges']) if SCATTER_G == 'low' \
        else max(MODEL_SB_RECOVERY['edges'])
    fig, axes = panel_grid(1, 2)
    ax_m, ax_z = axes

    order = np.argsort([a for a, _, _, _ in scans])
    for rank, i in enumerate(order):
        alpha, hdr, colour, s = scans[i]
        z, mvir = s['gc_z'], s['gc_mvir']
        f = observed_fics(s['gc_ics'], s['gc_mstar'], gval)
        near0 = np.abs(z - REDSHIFTS[Z0_SNAP]) < 1e-6

        for scale, sel in (('group', (mvir >= GROUP_LO) & (mvir < GROUP_HI)),
                           ('cluster', mvir >= CLUSTER_LO)):
            st = SCALE_STYLE[scale]
            c = group_colour(rank, len(scans)) if scale == 'group' else colour
            # clusters sit on top: there are 16 of them against 456 groups at
            # z = 0, and they are the objects the observations constrain
            kw = dict(marker=st['marker'], ms=st['ms'], color=c,
                      alpha=0.40 if scale == 'group' else 0.75,
                      mec='white', mew=0.4, ls='none',
                      zorder=(Z_BAND if scale == 'group' else Z_LINE) + rank,
                      rasterized=True)
            ax_m.plot(np.log10(mvir[sel & near0]), f[sel & near0], **kw)
            # the redshift panel is clusters only: the group cloud is two
            # orders of magnitude more numerous and buries the 16 objects the
            # cluster observations actually constrain
            if scale == 'cluster':
                ax_z.plot(z[sel], f[sel], **kw)
            print(f'  alpha = {alpha:<5g} {scale:<8s} '
                  f'{int((sel & near0).sum()):>5,d} at z = 0, '
                  f'{int(sel.sum()):>6,d} over all snapshots')

    # the observations, collapsed to one entry per class: the per-paper legend
    # lives in figure 1, and here it would crowd out the model
    for lo, hi, scale, c in ((13.0, 14.0, 'group', '#e6550d'),
                             (14.0, 15.0, 'cluster', '#253494')):
        fo = np.concatenate([o['f'] for o in icl_fraction_observations(scale)])
        fo = fo[fo > 0]
        p16, p84 = np.percentile(fo, (16, 84))
        ax_m.fill_between([lo, hi], p16, p84, facecolor=c, alpha=0.12,
                          edgecolor=c, lw=0.9, ls=':', zorder=1)
        if scale != 'cluster':
            continue
        zo = np.concatenate([o['z'] for o in icl_fraction_observations(scale)])
        keep = np.concatenate([o['f'] for o in icl_fraction_observations(scale)]) > 0
        ax_z.plot(zo[keep], fo, SCALE_STYLE[scale]['marker'], color='0.15',
                  ms=SCALE_STYLE[scale]['ms'] * 0.8, mec='white', mew=0.6,
                  ls='none', alpha=0.95, zorder=Z_LINE + 20,
                  label=f'observed {SCALE_STYLE[scale]["label"]}')

    # legend: alpha as shade, class as marker and ramp
    from matplotlib.lines import Line2D
    handles = []
    for scale in ('cluster', 'group'):
        st = SCALE_STYLE[scale]
        for rank, i in enumerate(order):
            alpha = scans[i][0]
            c = group_colour(rank, len(scans)) if scale == 'group' else scans[i][2]
            handles.append(Line2D([], [], marker=st['marker'], color=c,
                                  ms=st['ms'] * 0.8, mec='white', mew=0.4,
                                  ls='none',
                                  label=rf'{st["label"]}, $\beta_{{\rm clock}} = {alpha:g}$'))
    legend(ax_m, loc='upper left', ncol=2, handles=handles,
           labels=[h.get_label() for h in handles], fontsize=6.5)

    cut = rf'offset ${gval:+.2f}$ (Brough+24)'
    ax_m.set_title(f'groups and clusters at $z = 0$  ({cut})', fontsize=8)
    ax_m.set_xlabel(r'$\log_{10}(M_{\rm vir}\ [{\rm M}_\odot])$')
    ax_m.set_xlim(np.log10(GROUP_LO), 15.0)

    ax_z.set_title(rf'clusters only, $M_{{\rm vir}} > 10^{{14}}\,{{\rm M}}_\odot$, '
                   rf'every snapshot  ({cut})', fontsize=8)
    ax_z.set_xlabel(r'$z$')
    ax_z.set_xlim(0.0, 2.2)
    legend(ax_z, loc='upper right')

    for ax in (ax_m, ax_z):
        ax.set_ylabel(r'$f_{\rm ICS}$ as observed through the cut')
        ax.set_ylim(0.0, 1.0)

    fig.tight_layout()
    save_figure(fig, 'fig6_fICS_scatter', dpi=SCATTER_RASTER_DPI)


# ============ FIGURE 13: ICS ASSEMBLY BY CHANNEL ============
#
# After plotting/random_plotting_scripts/ics_formation_assembly_grid.py.  Every
# z = 0 central holding ICS (up to ASSEMBLY_N_MAX per halo-mass bin) is followed
# back by GalaxyIndex, and its two ICS bookkeeping channels are normalised to
# their own z = 0 values:
#
#   in situ   ICS_disrupt(t) / ICS_disrupt(z=0)  satellites disrupted into this halo
#   ex situ   ICS_accrete(t) / ICS_accrete(z=0)  ICS that arrived already made,
#                                                pre-processed in infalling groups
#
# The curves are when each channel was assembled onto the main branch.  The
# accretion channel books a packet when it arrives, not when it was stripped;
# the stripping time is carried separately in ICS_sum_mt.

ASSEMBLY_BINS = (
    (1e10, 1e11, r'$10^{10} < M_{\rm vir} < 10^{11}\,{\rm M}_\odot$'),
    (1e11, 1e12, r'$10^{11} < M_{\rm vir} < 10^{12}\,{\rm M}_\odot$'),
    (1e12, 1e13, r'$10^{12} < M_{\rm vir} < 10^{13}\,{\rm M}_\odot$'),
    (1e13, 1e14, r'$10^{13} < M_{\rm vir} < 10^{14}\,{\rm M}_\odot$'),
    (1e14, 1e15, r'$10^{14} < M_{\rm vir} < 10^{15}\,{\rm M}_\odot$'),
    (1e15, 1e18, r'$M_{\rm vir} > 10^{15}\,{\rm M}_\odot$'),
)
ASSEMBLY_N_MAX = 7500
ASSEMBLY_SEED = 42
DISRUPT_COLOUR, ACCRETE_COLOUR, TOTAL_COLOUR = '#1B7837', '#762A83', '0.25'
_A_FIELDS = ('ics', 'dis', 'acc')


def _assembly_file(args):
    """(snapshot, rows of *gids*, {field: values}) for every sampled galaxy in one file."""
    fp, gids, conv, snaps = args
    out = []
    with h5.File(fp, 'r') as f:
        for snap in snaps:
            key = f'Snap_{snap}'
            if key not in f or f[key]['Type'].shape[0] == 0 or 'ICS_disrupt' not in f[key]:
                continue
            g = f[key]
            gi = g['GalaxyIndex'][:].astype(np.int64)
            pos = np.clip(np.searchsorted(gids, gi), 0, gids.size - 1)
            hit = gids[pos] == gi
            if not hit.any():
                continue
            out.append((snap, pos[hit], {
                'ics': g['IntraClusterStars'][:][hit] * conv,
                'dis': g['ICS_disrupt'][:][hit] * conv,
                'acc': g['ICS_accrete'][:][hit] * conv}))
    return out


def scan_assembly(hdr, verbose=True):
    """ICS, ICS_disrupt and ICS_accrete along the main branch of sampled centrals."""
    conv, files = hdr['mass_convert'], hdr['files']
    rng = np.random.default_rng(ASSEMBLY_SEED)
    gids, bins_, mvir0 = [], [], []
    for fp in files:
        with h5.File(fp, 'r') as f:
            g = f[f'Snap_{Z0_SNAP}']
            if g['Type'].shape[0] == 0 or 'ICS_disrupt' not in g:
                continue
            t, mv = g['Type'][:], g['Mvir'][:] * conv
            ics, gi = g['IntraClusterStars'][:], g['GalaxyIndex'][:].astype(np.int64)
            for b, (lo, hi, _) in enumerate(ASSEMBLY_BINS):
                idx = np.flatnonzero((t == 0) & (mv >= lo) & (mv < hi) & (ics > 0))
                gids.append(gi[idx])
                bins_.append(np.full(idx.size, b))
                mvir0.append(mv[idx])
    if not gids:
        return None
    gids, bins_, mvir0 = map(np.concatenate, (gids, bins_, mvir0))
    keep = np.zeros(gids.size, bool)
    for b in range(len(ASSEMBLY_BINS)):
        idx = np.flatnonzero(bins_ == b)
        if idx.size > ASSEMBLY_N_MAX:
            idx = rng.choice(idx, ASSEMBLY_N_MAX, replace=False)
        keep[idx] = True
    gids, bins_, mvir0 = gids[keep], bins_[keep], mvir0[keep]
    o = np.argsort(gids)
    gids, bins_, mvir0 = gids[o], bins_[o], mvir0[o]

    A = {k: np.full((gids.size, len(REDSHIFTS)), np.nan) for k in _A_FIELDS}
    snaps = [s_ for s_ in SNAPS if s_ <= Z0_SNAP]
    work = [(fp, gids, conv, snaps) for fp in files]
    jobs = min(FILE_JOBS, len(work))
    if jobs > 1:
        pool = ProcessPoolExecutor(max_workers=jobs)
        parts = pool.map(_assembly_file, work)
    else:
        pool = None
        parts = (_assembly_file(w) for w in work)
    for hits in parts:
        for snap, rows, vals in hits:
            for k in _A_FIELDS:
                A[k][rows, snap] = vals[k]
    if pool is not None:
        pool.shutdown()
    A.update(gid=gids, bin=bins_, mvir0=mvir0)
    if verbose:
        counts = ', '.join(f'{(bins_ == b).sum():,}' for b in range(len(ASSEMBLY_BINS)))
        print(f'  assembly sample per halo-mass bin: {counts}')
    return A


def load_assembly(hdr, refresh=False, verbose=True):
    tag = os.path.basename(os.path.normpath(hdr['directory']))
    cache = os.path.join(CACHE_DIR, f'assembly_{CACHE_VERSION}_{tag}.npz')
    newest = max(os.path.getmtime(f) for f in hdr['files'])
    if not refresh and os.path.exists(cache) and os.path.getmtime(cache) > newest:
        if verbose:
            print(f'  assembly from cache: {cache}')
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    if verbose:
        print(f'  following ICS-holding centrals in {hdr["directory"]}')
    A = scan_assembly(hdr, verbose=verbose)
    if A is None:
        print('  no ICS_disrupt / ICS_accrete in this output -- run with TrackICSAssembly')
        return None
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(cache, **A)
    return A


def all_assembly(refresh=False):
    out = []
    for alpha, hdr, colour in ALL_RUNS:
        print(f'alpha = {alpha:g}')
        A = load_assembly(hdr, refresh=refresh)
        if A is not None:
            out.append((alpha, hdr, colour, A))
    return out


def _normalised(A, key):
    """Each history over its own z = 0 value; histories with nothing at z = 0 dropped."""
    z0 = A[key][:, Z0_SNAP]
    with np.errstate(invalid='ignore', divide='ignore'):
        r = A[key] / z0[:, None]
    r[~(z0 > 0)] = np.nan
    return r


def _t_reach(R, frac):
    """Lookback time [Gyr] at which each normalised history first reaches *frac*."""
    out = np.full(R.shape[0], np.nan)
    for i in range(R.shape[0]):
        ok = np.flatnonzero(np.isfinite(R[i]) & (R[i] >= frac))
        if ok.size:
            out[i] = LOOKBACK_AT_SNAP[ok[0]]
    return out


def plot_13_ics_assembly(runs):
    """ICS assembly by channel against lookback time, one panel per halo-mass bin.

    Colour is the channel (green in situ, purple ex situ, grey the total); line
    style is alpha; the 15th-85th band is drawn for the published run only.
    """
    print('Figure 13: ICS assembly by channel')
    ncols = 3
    nrows = (len(ASSEMBLY_BINS) + ncols - 1) // ncols
    fig, axes = panel_grid(nrows, ncols, sharex=True, sharey=True)
    lb = LOOKBACK_AT_SNAP
    for bi, (lo, hi, label) in enumerate(ASSEMBLY_BINS):
        ax = axes.flat[bi]
        drew = False
        for alpha, hdr, colour, A in runs:
            sel = A['bin'] == bi
            if sel.sum() < 2:
                continue
            for key, c in (('dis', DISRUPT_COLOUR), ('acc', ACCRETE_COLOUR),
                           ('ics', TOTAL_COLOUR)):
                R = _normalised(A, key)[sel]
                n = np.isfinite(R).sum(axis=0)
                ok = n >= min(MIN_COUNT, sel.sum())
                ok[Z0_SNAP + 1:] = False
                if not ok.any():
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    p = np.nanpercentile(R[:, ok], (15, 50, 85), axis=0)
                if alpha == FIDUCIAL and key != 'ics':
                    ax.fill_between(lb[ok], p[0], p[2], color=c, alpha=0.18, lw=0,
                                    zorder=Z_BAND)
                ax.plot(lb[ok], p[1], color=c, ls=alpha_ls(alpha),
                        lw=2.0 if alpha == FIDUCIAL else 1.5,
                        zorder=Z_LINE + (1 if alpha == FIDUCIAL else 0))
                drew = True
            n_lab = sel.sum()
        if not drew:
            ax.text(0.5, 0.5, 'no data', transform=ax.transAxes, ha='center', va='center')
        ax.set_title(label, fontsize=8)
        ax.set_xlim(0, 13)
        ax.set_ylim(0, 1.05)
        if bi % ncols == 0:
            ax.set_ylabel(r'fraction of $z = 0$ value')
        if bi // ncols == nrows - 1:
            ax.set_xlabel('lookback time [Gyr]')
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=DISRUPT_COLOUR, lw=2, label='in situ (ICS_disrupt)'),
               Line2D([], [], color=ACCRETE_COLOUR, lw=2, label='ex situ (ICS_accrete)'),
               Line2D([], [], color=TOTAL_COLOUR, lw=2, label='total ICS')]
    handles += [Line2D([], [], color='0.3', lw=2, ls=alpha_ls(a), label=alpha_label(a))
                for a, *_ in runs]
    legend(axes.flat[0], loc='upper right', handles=handles,
           labels=[h.get_label() for h in handles], fontsize=6.5)
    fig.tight_layout()
    save_figure(fig, 'fig13_ics_assembly')


def report_assembly(runs):
    """Numbers behind figure 13, per halo-mass bin and alpha."""
    print()
    print('-' * 78)
    print('ICS assembly by channel (figure 13): z = 0 in-situ share, and the lookback')
    print('time [Gyr] by which each channel (and the total) reached 50 / 90 per cent')
    print(f'  {"bin":<14}{"alpha":>6}{"N":>7}{"in situ":>9}'
          f'{"t50 dis":>9}{"t50 acc":>9}{"t50 tot":>9}{"z50 tot":>9}{"t90 tot":>9}')
    for bi, (lo, hi, _) in enumerate(ASSEMBLY_BINS):
        for alpha, hdr, colour, A in runs:
            sel = A['bin'] == bi
            if not sel.any():
                continue
            d0, a0 = A['dis'][sel, Z0_SNAP], A['acc'][sel, Z0_SNAP]
            with np.errstate(invalid='ignore', divide='ignore'):
                insitu = np.nansum(d0) / np.nansum(d0 + a0)
            t50 = [np.nanmedian(_t_reach(_normalised(A, k)[sel], 0.5)) for k in ('dis', 'acc', 'ics')]
            t90 = np.nanmedian(_t_reach(_normalised(A, 'ics')[sel], 0.9))
            print(f'  {f"{lo:.0e}-{hi:.0e}":<14}{alpha:>6g}{sel.sum():>7,d}{insitu:>9.3f}'
                  + ''.join(_f(v, '.2f', 9) for v in t50)
                  + _f(_lookback_to_z(t50[2]) if np.isfinite(t50[2]) else np.nan, '.2f', 9)
                  + _f(t90, '.2f', 9))
    print('  in situ is mass-weighted over the bin; t50/t90 are per-halo medians')
    print('-' * 78)


# ========================== THE KIMMIG ET AL. COMPARISON ==========================
#
# Kimmig et al. (2025, A&A 700, A95) measure, in four hydrodynamical simulations,
# the fraction of a cluster's stars that are not in satellites,
#
#     f_ICL+BCG = m*(BCG + ICL) / m*(BCG + ICL + satellites within R200c),
#
# and show it is a dynamical clock: it tracks z_form, the redshift at which the
# cluster first reached half its z = 0 mass, better than the usual tracers.
# They deliberately do not split the BCG from the ICL.
#
# The SAGE26 equivalent, per FOF group:
#
#     f_ICL+BCG = (m*_central + m_ICS) / (m*_central + m_ICS + sum m*_sat)
#
# with satellites counted inside KIMMIG_APERTURE x Rvir of the central, the
# stand-in for their R200c.  Because it sums the central and the ICS, alpha
# only reaches it through the small difference between a merger (the
# satellite's cold gas bursts into stars on the central) and a disruption (the
# gas goes to the hot halo).  It is a test of when SAGE26 destroys satellites,
# which the merger tree decides, not of where alpha sends their stars.  Figure
# 11 is where alpha shows: it splits the BCG from the ICS.
#
# Differences from the paper to keep in mind when reading the figures:
#   * Mvir (the tree's Bryan & Norman mass) stands in for M200c everywhere;
#   * SAGE26 has no gradual stellar stripping -- a satellite keeps every star
#     until it is destroyed whole -- so f_ICL+BCG grows in steps, and the
#     second and fourth most massive galaxies never lose mass while they live;
#   * the ICS has no position, so the stellar centre shift s_stars is built
#     from the satellites alone, with the BCG and ICS at the central's position;
#   * phi_BCG = m*_BCG / R_half uses SAGE's model radii (1.68 R_disc for the
#     disc, BulgeRadius for the bulge, mass-weighted), not a particle half-mass
#     radius inside 0.1 R200c;
#   * there is no gas centre shift s_gas -- SAGE has no gas positions.
#
# The main branch of a cluster is followed through the central galaxy:
# GalaxyIndex is constant along a galaxy's life, so the z = 0 central is found
# at every earlier snapshot, and the FOF group it sat in then
# (CentralGalaxyIndex) is the progenitor measured.

KIMMIG_HOST_LO = 1e14        # z = 0 sample: Mvir above this [Msun], as their M200c > 1e14
KIMMIG_APERTURE = 1.0        # satellites counted inside this many Rvir; None counts the whole FOF
KIMMIG_WINDOW_GYR = 1.0      # figure 12's time window, their T = 1 Gyr (a crossing time)
KIMMIG_TAIL_PCT = 16.0       # the early/late and high/low f splits take this tail on each side
KIMMIG_N_MASS_BINS = 10      # figure 11 picks its extremes within this many bins of halo mass
KIMMIG_NSAT_MSTAR = 1e10     # N_sat and the passive fraction count satellites above this [Msun]
KIMMIG_SSFR_PASSIVE = 1e-11  # passive below this specific SFR [1/yr]
KIMMIG_MIN = 5               # objects needed before a median or a correlation is drawn
KIMMIG_CACHE_VERSION = 'k1'

# z_form is drawn on a warm ramp so it cannot be confused with the blue alpha
# ramp the fit lines use.  The light end is clipped: pale yellow vanishes on white.
ZFORM_CMAP = plt.get_cmap('YlOrRd')
ZFORM_CMAP = ZFORM_CMAP.from_list('zform', ZFORM_CMAP(np.linspace(0.25, 1.0, 256)))
EARLY_COLOUR, LATE_COLOUR = '#b2182b', '#2166ac'

# Their published numbers, in the order Magneticum, Hydrangea, Horizon-AGN,
# TNG100.  Correlations are Pearson (figs 7, 8); fits are their Table 1,
# orthogonal distance regression, in the form y = A x + B written beside each.
KIMMIG_SIMS = ('Magneticum', 'Hydrangea', 'Horizon-AGN', 'TNG100')
KIMMIG_SIM_LS = ('-', '--', '-.', ':')
KIMMIG_REF = dict(
    median_f=(0.65, 0.47, 0.46, 0.55),
    median_zform=(0.67, 0.71, 0.44, 0.78),
    r_zform_f=(0.69, 0.72, 0.82, 0.83),
    r_zform_fsub=(-0.56, -0.74, -0.55, -0.75),
    r_f_M12=(0.87, 0.78, 0.85, 0.82),
    r_f_M14=(0.88, 0.82, 0.77, 0.87),
    r_f_sstars=(-0.73, -0.58, -0.32, -0.69),
    r_f_phi=(0.52, 0.20, 0.11, 0.20),
    fit_zform_f=((2.90, -1.14), (3.00, -0.78), (3.80, -1.23), (2.72, -0.66)),      # z_form = A f + B
    fit_zform_fsub=((-15.7, 1.7), (-10.9, 1.7), (-8.5, 1.4), (-13.1, 1.8)),        # z_form = A f_sub + B
    fit_f_M12=((0.35, 0.37), (0.40, 0.26), (0.37, 0.19), (0.38, 0.27)),            # f = A log M12 + B
    fit_f_M14=((0.44, 0.12), (0.47, 0.04), (0.45, 0.04), (0.50, 0.01)),            # f = A log M14 + B
    fit_f_sstars=((-0.40, 0.17), (-0.38, 0.02), (-0.24, 0.19), (-0.38, 0.10)),     # f = A log s + B
    fit_f_phi=((0.9, -8.5), (1.7, -17.3), (0.5, -4.8), (1.5, -15.4)),              # f = A log phi + B
)
# Magneticum, z = 0: f_ICL+BCG of the clusters in the early / late z_form tails
# (their fig 5), and the shredding rate -- the rise in f_ICL+BCG per Gyr at
# fixed halo mass, ~4 per cent in groups and 3-4 per cent in clusters (fig 12).
KIMMIG_TAIL_F = dict(early=0.74, late=0.45)
KIMMIG_SHRED = (0.03, 0.04)

_K_PROPS = ('Type', 'GalaxyIndex', 'CentralGalaxyIndex', 'StellarMass', 'BulgeMass',
            'IntraClusterStars', 'Mvir', 'Posx', 'Posy', 'Posz', 'Rvir',
            'DiskRadius', 'BulgeRadius', 'SfrDisk', 'SfrBulge')
_K_FIELDS = ('mhost', 'm_bcg', 'm_ics', 'm_sat', 'm_2nd', 'm_4th', 'f',
             'fsub', 'f8', 's_stars', 'phi', 'fq', 'nsat')


def _rank_in_group(group, value):
    """Order rows by group then descending *value*; return (order, rank in group)."""
    o = np.lexsort((-value, group))
    g = group[o]
    pos = np.arange(g.size)
    new = np.r_[True, g[1:] != g[:-1]] if g.size else np.array([], bool)
    first = np.maximum.accumulate(np.where(new, pos, 0)) if g.size else pos
    return o, pos - first


def _group_measures(d, conv, z, box, want):
    """Every Kimmig+25 tracer for the FOF groups whose CentralGalaxyIndex is in *want*.

    Returns a dict of arrays aligned with *want*, NaN where a group is absent.
    Masses in Msun, phi in Msun/kpc, s_stars in units of Rvir.
    """
    n = len(want)
    out = {k: np.full(n, np.nan) for k in _K_FIELDS}
    cgi = d['CentralGalaxyIndex'].astype(np.int64)
    rows = np.flatnonzero(np.isin(cgi, want))
    if rows.size == 0:
        return out
    L = {k: d[k][rows] for k in _K_PROPS}
    o = np.argsort(want)
    g = o[np.searchsorted(want[o], cgi[rows])]          # group of every row

    is_cen = L['GalaxyIndex'].astype(np.int64) == cgi[rows]
    cen = np.full(n, -1)
    cen[g[is_cen]] = np.flatnonzero(is_cen)
    has = cen >= 0
    ok_row = has[g]
    c = np.where(has, cen, 0)                           # safe index; masked by *has*

    ms = L['StellarMass'] * conv
    # comoving offsets from the central, periodic box; Rvir is physical
    dx = np.stack([L[k] - L[k][c][g] for k in ('Posx', 'Posy', 'Posz')], axis=1)
    dx = (dx + 0.5 * box) % box - 0.5 * box
    dist = np.sqrt((dx ** 2).sum(axis=1))
    rvir_com = L['Rvir'][c] * (1.0 + z)                 # comoving, per group
    member = ok_row & ~is_cen
    if KIMMIG_APERTURE is not None:
        member &= dist <= KIMMIG_APERTURE * rvir_com[g]

    mhost = L['Mvir'][c] * conv
    bcg = ms[c]
    ics = np.bincount(g[ok_row], L['IntraClusterStars'][ok_row] * conv, minlength=n)
    msat = np.bincount(g[member], ms[member], minlength=n)

    # second and fourth most massive galaxies, counting the BCG: satellite ranks 0 and 2
    idx = np.flatnonzero(member)
    m2, m4 = np.full(n, np.nan), np.full(n, np.nan)
    if idx.size:
        so, rk = _rank_in_group(g[idx], ms[idx])
        gi, mi = g[idx][so], ms[idx][so]
        m2[gi[rk == 0]] = mi[rk == 0]
        m4[gi[rk == 2]] = mi[rk == 2]

    # subhalo mass: only satellites that still have one (Type 1)
    sub = member & (L['Type'] == 1)
    mv = L['Mvir'] * conv
    fsub = np.bincount(g[sub], mv[sub], minlength=n)
    f8 = np.full(n, np.nan)
    idx = np.flatnonzero(sub)
    if idx.size:
        so, rk = _rank_in_group(g[idx], mv[idx])
        gi, mi = g[idx][so], mv[idx][so]
        f8[gi[rk == 7]] = mi[rk == 7]

    # stellar barycentre; the BCG and the ICS sit at the central's position
    mtot = bcg + ics + msat
    shift = np.zeros((n, 3))
    for j in range(3):
        shift[:, j] = np.bincount(g[member], ms[member] * dx[member, j], minlength=n)
    with np.errstate(invalid='ignore', divide='ignore'):
        s_stars = np.sqrt((shift ** 2).sum(axis=1)) / mtot / rvir_com

    # central potential proxy, m*_BCG / R_half in Msun/kpc (radii are physical Mpc/h)
    bulge = L['BulgeMass'][c] * conv
    disc = np.clip(bcg - bulge, 0.0, None)
    with np.errstate(invalid='ignore', divide='ignore'):
        r_half = (disc * 1.68 * L['DiskRadius'][c] + bulge * L['BulgeRadius'][c]) / bcg
        phi = bcg / (r_half * 1e3 / HUBBLE_H)

    # passive fraction and richness of the massive satellites
    big = member & (ms > KIMMIG_NSAT_MSTAR)
    with np.errstate(invalid='ignore', divide='ignore'):
        ssfr = (L['SfrDisk'] + L['SfrBulge']) / ms
    nsat = np.bincount(g[big], minlength=n).astype(float)
    npas = np.bincount(g[big & (ssfr < KIMMIG_SSFR_PASSIVE)], minlength=n)

    with np.errstate(invalid='ignore', divide='ignore'):
        vals = dict(mhost=mhost, m_bcg=bcg, m_ics=ics, m_sat=msat, m_2nd=m2, m_4th=m4,
                    f=(bcg + ics) / mtot, fsub=fsub / mhost, f8=f8 / mhost,
                    s_stars=s_stars, phi=phi,
                    fq=np.where(nsat > 0, npas / nsat, np.nan), nsat=nsat)
    for k, v in vals.items():
        out[k] = np.where(has, v, np.nan)
    return out


# Rows of the tracked groups closer together than this are read as one slice.
# One HDF5 chunk is 8192 rows, so a smaller gap saves no I/O.
_K_READ_GAP = 8192


def _row_runs(rows, gap=_K_READ_GAP):
    """Sorted *rows* merged into (start, stop) slices wherever they lie within *gap*."""
    if rows.size == 0:
        return []
    cut = np.flatnonzero(np.diff(rows) > gap)
    lo = np.r_[rows[0], rows[cut + 1]]
    hi = np.r_[rows[cut], rows[-1]] + 1
    return list(zip(lo, hi))


def _kimmig_file(args):
    """The main-branch histories of the clusters in one file, (cluster, snapshot) per field.

    Only GalaxyIndex and CentralGalaxyIndex are read in full.  The other
    columns are read over the row slices that cover the tracked groups: a FOF
    group's galaxies are written contiguously within their tree, so this is a
    few per cent of each snapshot rather than all of it.
    """
    fp, gi_track, conv, box, snaps = args
    n = gi_track.size
    H = {k: np.full((n, len(REDSHIFTS)), np.nan) for k in _K_FIELDS}
    rest = tuple(p for p in _K_PROPS if p not in ('GalaxyIndex', 'CentralGalaxyIndex'))
    with h5.File(fp, 'r') as f:
        for s in snaps:
            key = f'Snap_{s}'
            if key not in f or f[key]['Type'].shape[0] == 0:
                continue
            g = f[key]
            gal = g['GalaxyIndex'][:].astype(np.int64)
            cgi = g['CentralGalaxyIndex'][:].astype(np.int64)
            o = np.argsort(gal)
            pos = np.clip(np.searchsorted(gal[o], gi_track), 0, gal.size - 1)
            found = gal[o][pos] == gi_track
            if not found.any():
                continue
            host = cgi[o][pos[found]]
            uh, inv = np.unique(host, return_inverse=True)

            runs = _row_runs(np.flatnonzero(np.isin(cgi, uh)))
            take = np.concatenate([np.arange(lo, hi) for lo, hi in runs])
            d = {'GalaxyIndex': gal[take], 'CentralGalaxyIndex': cgi[take]}
            for p in rest:
                ds = g[p]
                d[p] = np.concatenate([ds[lo:hi] for lo, hi in runs])
            m = _group_measures(d, conv, REDSHIFTS[s], box, uh)
            for k in _K_FIELDS:
                H[k][found, s] = m[k][inv]
    return H


def scan_kimmig(hdr, verbose=True):
    """Every z = 0 cluster's main progenitor, measured at every snapshot.

    Returns 2-D arrays (cluster, snapshot) for every field in _K_FIELDS.
    """
    conv, box, files = hdr['mass_convert'], hdr['box_size'], hdr['files']
    track = []
    for fp in files:
        with h5.File(fp, 'r') as f:
            g = f[f'Snap_{Z0_SNAP}']
            if g['Type'].shape[0] == 0:
                track.append(np.array([], np.int64))
                continue
            sel = (g['Type'][:] == 0) & (g['Mvir'][:] * conv >= KIMMIG_HOST_LO)
            track.append(g['GalaxyIndex'][:][sel].astype(np.int64))
    ntot = sum(t.size for t in track)
    snaps = [s for s in SNAPS if s <= Z0_SNAP]

    work = [(fp, t, conv, box, snaps) for fp, t in zip(files, track) if t.size]
    jobs = min(FILE_JOBS, len(work))
    if verbose:
        print(f'  {ntot:,} hosts in {len(work)} files, {jobs} worker(s)')
    t_start = time.time()
    if jobs > 1:
        pool = ProcessPoolExecutor(max_workers=jobs)
        parts = pool.map(_kimmig_file, work)
    else:
        pool = None
        parts = (_kimmig_file(w) for w in work)
    got = []
    for i, part in enumerate(parts):
        got.append(part)
        if verbose:
            print(f'    file {i + 1}/{len(work)} [{time.time() - t_start:7.0f} s]', flush=True)
    if pool is not None:
        pool.shutdown()

    H = {k: (np.concatenate([p[k] for p in got]) if got
             else np.full((0, len(REDSHIFTS)), np.nan)) for k in _K_FIELDS}
    if verbose:
        print(f'  {ntot:,} hosts above Mvir = {KIMMIG_HOST_LO:.1e} Msun at '
              f'Snap_{Z0_SNAP}, followed through {len(snaps)} snapshots')
    return H


def load_kimmig(hdr, refresh=False, verbose=True):
    """The main-branch histories for one run, from the npz cache when current."""
    tag = os.path.basename(os.path.normpath(hdr['directory']))
    ap = 'fof' if KIMMIG_APERTURE is None else f'{KIMMIG_APERTURE:g}rv'
    cache = os.path.join(CACHE_DIR, f'kimmig_{KIMMIG_CACHE_VERSION}_{tag}_'
                         f'{np.log10(KIMMIG_HOST_LO):.2f}_{ap}.npz')
    newest = max(os.path.getmtime(f) for f in hdr['files'])
    if not refresh and os.path.exists(cache) and os.path.getmtime(cache) > newest:
        if verbose:
            print(f'  histories from cache: {cache}')
        z = np.load(cache)
        H = {k: z[k] for k in z.files}
    else:
        if verbose:
            print(f'  following cluster main branches in {hdr["directory"]}')
        H = scan_kimmig(hdr, verbose=verbose)
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez_compressed(cache, **H)
    return derive_kimmig(H)


def _zform(mhost):
    """Redshift at which each history first reaches half its z = 0 mass, interpolated."""
    m0 = mhost[:, Z0_SNAP]
    out = np.full(len(m0), np.nan)
    for i in range(len(m0)):
        r = mhost[i] / m0[i]
        snaps = np.flatnonzero(np.isfinite(r))
        hit = snaps[r[snaps] >= 0.5]
        if hit.size == 0:
            continue
        s = hit[0]
        prev = snaps[snaps < s]
        if prev.size == 0:
            out[i] = REDSHIFTS[s]
            continue
        p = prev[-1]
        w = (0.5 - r[p]) / (r[s] - r[p])
        out[i] = REDSHIFTS[p] + w * (REDSHIFTS[s] - REDSHIFTS[p])
    return out


def derive_kimmig(H):
    """Add the z = 0 tracers, z_form and the tail selections to the histories."""
    K = dict(H)
    z0 = {k: H[k][:, Z0_SNAP] for k in _K_FIELDS}
    K['z0'] = z0
    K['zform'] = _zform(H['mhost'])
    with np.errstate(invalid='ignore', divide='ignore'):
        K['M12'] = z0['m_bcg'] / z0['m_2nd']
        K['M14'] = z0['m_bcg'] / z0['m_4th']
        K['mgrowth'] = H['mhost'] / H['mhost'][:, Z0_SNAP][:, None]

    def tails(x):
        ok = np.isfinite(x)
        if ok.sum() < 2:
            return np.zeros_like(ok), np.zeros_like(ok)
        lo, hi = np.percentile(x[ok], (KIMMIG_TAIL_PCT, 100 - KIMMIG_TAIL_PCT))
        return ok & (x >= hi), ok & (x <= lo)

    K['early'], K['late'] = tails(K['zform'])
    K['f_high'], K['f_low'] = tails(z0['f'])
    return K


def all_kimmig(refresh=False):
    """(alpha, header, colour, histories) for every run in the sweep."""
    out = []
    for alpha, hdr, colour in ALL_RUNS:
        print(f'alpha = {alpha:g}')
        out.append((alpha, hdr, colour, load_kimmig(hdr, refresh=refresh)))
    return out


# ---------------- small helpers shared by figures 7-12 ----------------------

def _pearson(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < KIMMIG_MIN:
        return np.nan
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def _odr_line(x, y):
    """Orthogonal distance regression y = A x + B, as their Table 1.  None if too few."""
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < KIMMIG_MIN:
        return None
    a0, b0 = np.polyfit(x[ok], y[ok], 1)
    try:
        from scipy import odr
        res = odr.ODR(odr.RealData(x[ok], y[ok]), odr.unilinear,
                      beta0=[a0, b0]).run()
        return tuple(res.beta)
    except Exception:
        return a0, b0


def _fiducial(runs):
    """The published-alpha run when present, otherwise the first."""
    for r in runs:
        if r[0] == FIDUCIAL:
            return r
    return runs[0]


def _median_band(ax, x, Y, colour, lw=2.4, ls='-', label=None, bands=(1,),
                 band_alpha=0.18, zorder=Z_LINE):
    """Median of the columns of Y against x, with 1- and/or 2-sigma bands.

    A subsample smaller than KIMMIG_MIN (a tail of a small box) is drawn from
    as few as two objects rather than not at all.
    """
    n = np.isfinite(Y).sum(axis=0)
    ok = n >= min(KIMMIG_MIN, max(2, Y.shape[0]))
    if not ok.any():
        return
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pct = np.nanpercentile(Y[:, ok], (2.3, 16, 50, 84, 97.7), axis=0)
    xs = x[ok]
    if 2 in bands:
        ax.fill_between(xs, pct[0], pct[4], color=colour, alpha=band_alpha * 0.5,
                        lw=0, zorder=Z_BAND)
    if 1 in bands:
        ax.fill_between(xs, pct[1], pct[3], color=colour, alpha=band_alpha,
                        lw=0, zorder=Z_BAND)
    ax.plot(xs, pct[2], color=colour, lw=lw, ls=ls, label=label, zorder=zorder)


def _zform_scatter(ax, x, y, zf, **kw):
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(zf)
    return ax.scatter(x[ok], y[ok], c=zf[ok], cmap=ZFORM_CMAP, vmin=0.0, vmax=1.6,
                      s=kw.pop('s', 14), edgecolors='white', linewidths=0.3,
                      zorder=kw.pop('zorder', Z_OBS), rasterized=True, **kw)


def _paper_lines(ax, fits, xlim, invert=False):
    """Their Table 1 fits, one grey line per simulation.

    *invert* is for a fit written x = A y + B (their z_form = A f + B); *xlim*
    is then the range of y it is drawn over.
    """
    xs = np.linspace(*xlim, 50)
    for (a, b), name, ls in zip(fits, KIMMIG_SIMS, KIMMIG_SIM_LS):
        if invert:
            ax.plot(a * xs + b, xs, color='0.45', lw=1.1, ls=ls, zorder=Z_OBS - 1,
                    label=name)
        else:
            ax.plot(xs, a * xs + b, color='0.45', lw=1.1, ls=ls, zorder=Z_OBS - 1,
                    label=name)


def _fit_lines(ax, runs, xkey, ykey, xlim, log_x=False):
    """One ODR line per alpha run, with its Pearson r in the legend."""
    for alpha, hdr, colour, K in runs:
        x, y = _get(K, xkey), _get(K, ykey)
        if log_x:
            with np.errstate(invalid='ignore', divide='ignore'):
                x = np.log10(x)
        fit = _odr_line(x, y)
        r = _pearson(x, y)
        if fit is None:
            continue
        xs = np.linspace(*xlim, 50)
        ax.plot(xs, fit[0] * xs + fit[1], color=colour,
                label=alpha_label(alpha) + rf', $r = {r:.2f}$', **line_style(alpha))


def _get(K, key):
    """A z = 0 tracer by name: derived ones live on K, measured ones on K['z0']."""
    return K[key] if key in K and np.ndim(K[key]) == 1 else K['z0'][key]


def _zform_colourbar(fig, mappable, ax):
    cb = fig.colorbar(mappable, ax=ax, pad=0.01, fraction=0.05)
    cb.set_label(r'$z_{\rm form}$')
    return cb


def _redshift_top_axis(ax):
    """Redshift ticks along the top of a cosmic-time axis."""
    t, z = AGE_AT_SNAP[::-1], REDSHIFTS[::-1]

    def t2z(x):
        return np.interp(x, AGE_AT_SNAP, REDSHIFTS)

    def z2t(x):
        return np.interp(x, z, t)
    top = ax.secondary_xaxis('top', functions=(t2z, z2t))
    top.set_xticks([3, 2, 1, 0.5, 0.25, 0.1])
    top.set_xticklabels(['3', '2', '1', '0.5', '0.25', '0.1'])
    top.set_xlabel(r'$z$')


# ============ FIGURE 7: ASSEMBLY (their figs 4 and 5) ============

def plot_7_assembly(runs):
    """z_form, halo growth, and f_ICL+BCG through time, split two ways.

    Columns: (left) the z_form distribution and f_ICL+BCG(z) for every alpha;
    (centre) the published run split into its earliest- and latest-forming
    tails; (right) the same run split by f_ICL+BCG at z = 0.  Top row is halo
    growth M(z)/M(0), bottom row f_ICL+BCG(z).  The halo histories come from the
    tree and are the same in every run; only f_ICL+BCG can move with alpha.
    """
    print('Figure 7: assembly and f_ICL+BCG through time')
    fig, axes = panel_grid(2, 3)
    alpha_f, hdr_f, colour_f, Kf = _fiducial(runs)
    z = REDSHIFTS

    # (a) z_form distribution
    ax = axes[0, 0]
    zf = Kf['zform'][np.isfinite(Kf['zform'])]
    if zf.size:
        ax.hist(zf, bins=np.linspace(0, 2.0, 21), color=colour_f, alpha=0.6,
                weights=np.full(zf.size, 100.0 / zf.size), label='SAGE26 (tree)')
        p = np.percentile(zf, (16, 50, 84))
        ax.axvline(p[1], color=colour_f, lw=2.0)
        for v in (p[0], p[2]):
            ax.axvline(v, color=colour_f, lw=1.0, ls='--')
    for m, name, ls in zip(KIMMIG_REF['median_zform'], KIMMIG_SIMS, KIMMIG_SIM_LS):
        ax.axvline(m, color='0.45', lw=1.1, ls=ls, label=f'{name} median')
    ax.set_xlabel(r'$z_{\rm form}$')
    ax.set_ylabel('clusters [%]')
    ax.set_title(rf'$N = {zf.size}$ hosts, $M_{{\rm vir}} > 10^{{{np.log10(KIMMIG_HOST_LO):.1f}}}$'
                 r'$\,{\rm M}_\odot$', fontsize=8)
    legend(ax, loc='upper right')

    # (d) f_ICL+BCG(z), every alpha
    ax = axes[1, 0]
    for alpha, hdr, colour, K in runs:
        _median_band(ax, z, K['f'], colour, label=alpha_label(alpha),
                     bands=(1,) if alpha == FIDUCIAL else (), **{
                         k: v for k, v in line_style(alpha).items() if k in ('lw', 'ls')})
    ax.set_ylabel(r'$f_{\rm ICL+BCG}$')
    legend(ax, loc='lower left')

    # (b, e) split by z_form; (c, f) split by f_ICL+BCG
    for col, (hi, lo, lab_hi, lab_lo) in (
            (1, ('early', 'late', 'earliest', 'latest')),
            (2, ('f_high', 'f_low', r'highest $f_{\rm ICL+BCG}$', r'lowest $f_{\rm ICL+BCG}$'))):
        for row, key in ((0, 'mgrowth'), (1, 'f')):
            ax = axes[row, col]
            _median_band(ax, z, Kf[key], '0.35', lw=2.0, bands=(1, 2),
                         label='all')
            for sel, c, lab in ((Kf[hi], EARLY_COLOUR, lab_hi),
                                (Kf[lo], LATE_COLOUR, lab_lo)):
                _median_band(ax, z, Kf[key][sel], c, lw=2.0,
                             label=rf'{lab} {KIMMIG_TAIL_PCT:g}% ($N = {sel.sum()}$)')
            if row == 0:
                ax.set_yscale('log')
                ax.set_ylim(0.01, 2.0)
                ax.axhline(0.5, color='0.6', lw=0.8, ls=':')
                ax.set_ylabel(r'$M_{\rm vir}(z) / M_{\rm vir}(z=0)$')
            else:
                ax.set_ylabel(r'$f_{\rm ICL+BCG}$')
                ax.axhline(KIMMIG_TAIL_F['early'], color=EARLY_COLOUR, lw=0.9, ls=':')
                ax.axhline(KIMMIG_TAIL_F['late'], color=LATE_COLOUR, lw=0.9, ls=':')
            legend(ax, loc='lower left' if row == 0 else 'lower right')
        axes[0, col].set_title(alpha_label(alpha_f) + ', split by ' +
                               (r'$z_{\rm form}$' if col == 1 else
                                r'$f_{\rm ICL+BCG}(z=0)$'), fontsize=8)

    # the median halo's z20 / z50 / z90, as their dash-dotted lines
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        med = np.nanmedian(Kf['mgrowth'], axis=0)
    for frac in (0.2, 0.5, 0.9):
        ok = np.isfinite(med) & (med >= frac)
        if ok.any():
            zf_ = z[np.flatnonzero(ok)[0]]
            for ax in axes[:, 1:].ravel():
                ax.axvline(zf_, color='0.55', lw=0.8, ls='-.', zorder=1)

    for ax in axes.ravel()[[1, 2, 3, 4, 5]]:
        ax.set_xlim(3.0, 0.0)
        ax.set_xlabel(r'$z$')
    for ax in axes[1]:
        ax.set_ylim(0.0, 1.0)
    fig.text(0.5, 0.005, r'dotted horizontal lines: Magneticum clusters at $z = 0$ in '
             r'the early (red) and late (blue) $z_{\rm form}$ tails (Kimmig+25 fig. 5)',
             ha='center', fontsize=7, color='0.35')
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    save_figure(fig, 'fig7_kimmig_assembly')


# ============ FIGURE 8: f_ICL+BCG AS A CLOCK (their figs 3, 6, 7, 10) ============

def _row_colourbar(fig, mappable):
    """One z_form colour bar for a whole grid, in a strip on the right."""
    fig.tight_layout(rect=(0, 0, 0.94, 1))
    cax = fig.add_axes([0.952, 0.15, 0.010, 0.70])
    cb = fig.colorbar(mappable, cax=cax)
    cb.set_label(r'$z_{\rm form}$')


def _row_label(ax, alpha):
    """Name the run a row belongs to, in its first panel."""
    ax.text(0.03, 0.97, alpha_label(alpha), transform=ax.transAxes, ha='left',
            va='top', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.7', lw=0.6))


def plot_8_clock(runs):
    """f_ICL+BCG against halo mass and z_form, f_sub against z_form, and the SHMR.

    One row per beta_clock run, four columns:
      (a) f_ICL+BCG against halo mass (their fig 6), the run's median as a line
          and the four simulations' medians as a grey band;
      (b) f_ICL+BCG against z_form (their fig 7, top) with the run's ODR fit and
          their Table 1 fits in grey;
      (c) f_sub against z_form (their fig 7, bottom) -- a halo quantity, so the
          rows differ only through which hosts pass the cut;
      (d) the stellar-to-halo mass relation: everything (grey), BCG + ICS
          (coloured by z_form) and the BCG alone (line).
    Points are coloured by z_form throughout.
    """
    print('Figure 8: f_ICL+BCG as a dynamical clock')
    order = np.argsort([a for a, _, _, _ in runs])
    fig, axes = panel_grid(len(runs), 4, squeeze=False)
    sc = None
    for row, i in enumerate(order):
        alpha, hdr, colour, K = runs[i]
        ax_m, ax_z, ax_s, ax_shmr = axes[row]
        zf = K['zform']
        lm = np.log10(K['z0']['mhost'])
        bins = np.linspace(np.log10(KIMMIG_HOST_LO), max(np.nanmax(lm), 14.0) + 0.01, 7)

        # (a) f against halo mass
        sc = _zform_scatter(ax_m, lm, K['z0']['f'], zf)
        c, p = binned_percentiles(lm, K['z0']['f'], bins, min_count=KIMMIG_MIN)
        ok = np.isfinite(p[1])
        ax_m.plot(c[ok], p[1][ok], color=colour, lw=2.4, zorder=Z_LINE + 5,
                  label='median')
        lo, hi = min(KIMMIG_REF['median_f']), max(KIMMIG_REF['median_f'])
        ax_m.axhspan(lo, hi, color='0.6', alpha=0.25, lw=0, zorder=1,
                     label='Kimmig+25 medians (4 sims)')
        ax_m.set_xlabel(r'$\log_{10}(M_{\rm vir}\ [{\rm M}_\odot])$')
        ax_m.set_ylabel(r'$f_{\rm ICL+BCG}$')
        ax_m.set_ylim(0.0, 1.0)
        legend(ax_m, loc='lower left')
        _row_label(ax_m, alpha)

        # (b) f against z_form; their fit is z_form(f), ours is drawn the same way round
        _zform_scatter(ax_z, zf, K['z0']['f'], zf)
        _paper_lines(ax_z, KIMMIG_REF['fit_zform_f'], (0.0, 1.0), invert=True)
        fit = _odr_line(K['z0']['f'], zf)
        r = _pearson(K['z0']['f'], zf)
        if fit is not None:
            fs = np.linspace(0.0, 1.0, 50)
            ax_z.plot(fit[0] * fs + fit[1], fs, color=colour, lw=2.4, zorder=Z_LINE + 5,
                      label=rf'SAGE26, $r = {r:.2f}$')
        ax_z.set_xlabel(r'$z_{\rm form}$')
        ax_z.set_ylabel(r'$f_{\rm ICL+BCG}$')
        ax_z.set_xlim(0.0, 2.0)
        ax_z.set_ylim(0.0, 1.0)
        if row == 0:
            ax_z.set_title(r'grey: Kimmig+25 $z_{\rm form} = A f + B$, $r = 0.69$-$0.83$',
                           fontsize=8)
        legend(ax_z, loc='lower left', ncol=2, fontsize=6.0)

        # (c) f_sub against z_form
        _zform_scatter(ax_s, zf, K['z0']['fsub'], zf)
        _paper_lines(ax_s, KIMMIG_REF['fit_zform_fsub'], (0.0, 0.3), invert=True)
        fit = _odr_line(K['z0']['fsub'], zf)
        r = _pearson(K['z0']['fsub'], zf)
        if fit is not None:
            fs = np.linspace(0.0, 0.3, 50)
            ax_s.plot(fit[0] * fs + fit[1], fs, color=colour, lw=2.4, zorder=Z_LINE + 5,
                      label=rf'SAGE26, $r = {r:.2f}$')
        ax_s.set_xlabel(r'$z_{\rm form}$')
        ax_s.set_ylabel(r'$f_{\rm sub}$')
        ax_s.set_xlim(0.0, 2.0)
        ax_s.set_ylim(0.0, 0.3)
        if row == 0:
            ax_s.set_title(r'grey: Kimmig+25, $r = -0.55$ to $-0.75$', fontsize=8)
        legend(ax_s, loc='lower left', ncol=2, fontsize=6.0)

        # (d) stellar-to-halo mass
        tot = K['z0']['m_bcg'] + K['z0']['m_ics'] + K['z0']['m_sat']
        ax_shmr.plot(lm, np.log10(tot), 'o', color='0.65', ms=3.0, mec='none',
                     ls='none', zorder=Z_BAND, label=r'BCG + ICS + satellites',
                     rasterized=True)
        _zform_scatter(ax_shmr, lm, np.log10(K['z0']['m_bcg'] + K['z0']['m_ics']), zf,
                       label=r'BCG + ICS')
        c, p = binned_percentiles(lm, np.log10(K['z0']['m_bcg']), bins,
                                  min_count=KIMMIG_MIN)
        ok = np.isfinite(p[1])
        ax_shmr.plot(c[ok], p[1][ok], color=colour, lw=2.4, zorder=Z_LINE + 5,
                     label='BCG alone (median)')
        ax_shmr.set_xlabel(r'$\log_{10}(M_{\rm vir}\ [{\rm M}_\odot])$')
        ax_shmr.set_ylabel(r'$\log_{10}(m_\star\ [{\rm M}_\odot])$')
        legend(ax_shmr, loc='lower left' if row else 'upper left')

    if sc is not None:
        _row_colourbar(fig, sc)
    save_figure(fig, 'fig8_kimmig_clock')


# ============ FIGURE 9: OBSERVABLE PROXIES (their fig 8) ============

def plot_9_proxies(runs):
    """f_ICL+BCG against M12, M14, the stellar centre shift and phi_BCG.

    One row per beta_clock run; each panel carries that run's ODR fit and
    Pearson r, with their Table 1 fits in grey.
    """
    print('Figure 9: observable proxies for f_ICL+BCG')
    order = np.argsort([a for a, _, _, _ in runs])
    fig, axes = panel_grid(len(runs), 4, squeeze=False)
    panels = (
        ('M12', r'$\log_{10}(M_{12})$', 'fit_f_M12', 'r_f_M12', (0.0, 2.0)),
        ('M14', r'$\log_{10}(M_{14})$', 'fit_f_M14', 'r_f_M14', (0.0, 2.2)),
        ('s_stars', r'$\log_{10}(s_{\rm stars})$', 'fit_f_sstars', 'r_f_sstars', (-3.0, -0.3)),
        ('phi', r'$\log_{10}(\phi_{\rm BCG}\ [{\rm M}_\odot\,{\rm kpc}^{-1}])$',
         'fit_f_phi', 'r_f_phi', (9.5, 12.0)),
    )
    sc = None
    for row, i in enumerate(order):
        alpha, hdr, colour, K = runs[i]
        zf = K['zform']
        for col, (key, xlabel, fitkey, rkey, xlim) in enumerate(panels):
            ax = axes[row, col]
            with np.errstate(invalid='ignore', divide='ignore'):
                x = np.log10(_get(K, key))
            y = K['z0']['f']
            sc = _zform_scatter(ax, x, y, zf)
            _paper_lines(ax, KIMMIG_REF[fitkey], xlim)
            fit = _odr_line(x, y)
            r = _pearson(x, y)
            if fit is not None:
                xs = np.linspace(*xlim, 50)
                ax.plot(xs, fit[0] * xs + fit[1], color=colour, lw=2.4, zorder=Z_LINE + 5,
                        label=rf'SAGE26, $r = {r:.2f}$')
            if row == 0:
                rs = ', '.join(f'{v:.2f}' for v in KIMMIG_REF[rkey])
                ax.set_title(rf'Kimmig+25 $r$ = {rs}', fontsize=7.5)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(r'$f_{\rm ICL+BCG}$')
            ax.set_xlim(*xlim)
            ax.set_ylim(0.0, 1.0)
            legend(ax, loc='lower left', fontsize=6.0)
            if col == 0:
                _row_label(ax, alpha)
    if sc is not None:
        _row_colourbar(fig, sc)
    save_figure(fig, 'fig9_kimmig_proxies')


# ============ FIGURE 10: THE CORRELATION MATRIX (their fig 9) ============

_MATRIX = (
    ('zform', r'$z_{\rm form}$', False),
    ('f', r'$f_{\rm ICL+BCG}$', False),
    ('s_stars', r'$s_{\rm stars}$', True),
    ('fsub', r'$f_{\rm sub}$', False),
    ('f8', r'$f_8$', False),
    ('M12', r'$M_{12}$', True),
    ('M14', r'$M_{14}$', True),
    ('phi', r'$\phi_{\rm BCG}$', True),
    ('fq', r'$f_q$', False),
    ('nsat', r'$N_{\rm sat}$', False),
    ('mhost', r'$M_{\rm vir}$', True),
)


def kimmig_matrix(K):
    """|Pearson r| between every pair of tracers, logs where they fit in logs."""
    cols = []
    for key, _, logged in _MATRIX:
        v = np.asarray(_get(K, key), float)
        with np.errstate(invalid='ignore', divide='ignore'):
            cols.append(np.log10(v) if logged else v)
    n = len(cols)
    R = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            R[i, j] = _pearson(cols[i], cols[j])
    return R


def plot_10_matrix(runs):
    """One correlation matrix per alpha run, signed r written in each cell."""
    print('Figure 10: correlation matrix')
    fig, axes = panel_grid(1, len(runs))
    axes = np.atleast_1d(axes)
    labels = [lab for _, lab, _ in _MATRIX]
    for ax, (alpha, hdr, colour, K) in zip(axes, runs):
        R = kimmig_matrix(K)
        im = ax.imshow(np.abs(R), cmap='Greys', vmin=0.0, vmax=1.0)
        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                if np.isfinite(R[i, j]) and i != j:
                    ax.text(j, i, f'{R[i, j]:.2f}', ha='center', va='center',
                            fontsize=4.8, color='white' if abs(R[i, j]) > 0.55 else 'black')
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=60, fontsize=6.5)
        ax.set_yticklabels(labels, fontsize=6.5)
        ax.minorticks_off()
        ax.set_title(alpha_label(alpha) + rf', $N = {np.isfinite(K["zform"]).sum()}$',
                     fontsize=8)
    cb = fig.colorbar(im, ax=list(axes), pad=0.01, fraction=0.02)
    cb.set_label(r'$|r|$ (Pearson)')
    save_figure(fig, 'fig10_kimmig_matrix')


# ============ FIGURE 11: THE STELLAR BUDGET THROUGH TIME (their fig 11) ============

def _extremes_by_mass(K):
    """Highest and lowest f_ICL+BCG within bins of z = 0 halo mass, as their fig 11."""
    m, f = K['z0']['mhost'], K['z0']['f']
    ok = np.isfinite(m) & np.isfinite(f)
    hi_sel, lo_sel = np.zeros(m.size, bool), np.zeros(m.size, bool)
    if ok.sum() < 2:
        return hi_sel, lo_sel
    edges = np.logspace(np.log10(KIMMIG_HOST_LO), np.log10(m[ok].max()) + 1e-6,
                        KIMMIG_N_MASS_BINS + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        idx = np.flatnonzero(ok & (m >= a) & (m < b))
        if idx.size < 2:
            continue
        k = max(1, int(round(idx.size * KIMMIG_TAIL_PCT / 100.0)))
        o = idx[np.argsort(f[idx])]
        lo_sel[o[:k]] = True
        hi_sel[o[-k:]] = True
    return hi_sel, lo_sel


def plot_11_budget(runs):
    """Mean fraction of the stars in the BCG, ICS, second galaxy and the rest.

    One row per alpha, columns all / highest f_ICL+BCG / lowest f_ICL+BCG at
    fixed halo mass.  This is the one comparison alpha controls directly: it
    moves mass between the red and gold bands without touching their sum.  Their
    BCG/ICL boundary is 0.1 R200c; ours is the merger clock.
    """
    print('Figure 11: the stellar budget through time')
    fig, axes = panel_grid(len(runs), 3, squeeze=False)
    t = AGE_AT_SNAP
    comp_colours = ('#d73027', '#fdae61', '#8073ac', '#4575b4')
    comp_labels = ('BCG', 'ICS', 'second most massive', 'other satellites')
    for row, (alpha, hdr, colour, K) in enumerate(runs):
        hi_sel, lo_sel = _extremes_by_mass(K)
        tot = K['m_bcg'] + K['m_ics'] + K['m_sat']
        with np.errstate(invalid='ignore', divide='ignore'):
            parts = (K['m_bcg'] / tot, K['m_ics'] / tot, K['m_2nd'] / tot,
                     (K['m_sat'] - np.nan_to_num(K['m_2nd'])) / tot)
        for col, (sel, title) in enumerate((
                (np.ones(tot.shape[0], bool), 'all'),
                (hi_sel, r'highest $f_{\rm ICL+BCG}$'),
                (lo_sel, r'lowest $f_{\rm ICL+BCG}$'))):
            ax = axes[row, col]
            ok = np.isfinite(tot[sel]).sum(axis=0) >= max(1, min(KIMMIG_MIN, sel.sum()))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                means = [np.nan_to_num(np.nanmean(p[sel], axis=0))[ok] for p in parts]
            if ok.any():
                ax.stackplot(t[ok], *means, colors=comp_colours, labels=comp_labels,
                             alpha=0.9, lw=0)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                mmed = np.nanmedian(K['mhost'][sel], axis=0)
            for mcut, ls in ((1e13, ':'), (5e13, '--'), (1e14, '-.')):
                above = np.isfinite(mmed) & (mmed >= mcut)
                if above.any():
                    ax.axvline(t[np.flatnonzero(above)[0]], color='k', lw=0.9, ls=ls)
            ax.set_xlim(t[ok][0] if ok.any() else 0, t[Z0_SNAP])
            ax.set_ylim(0.0, 1.0)
            ax.set_xlabel(r'cosmic time [Gyr]')
            ax.set_ylabel(r'fraction of stellar mass')
            ax.set_title(f'{alpha_label(alpha)}, {title} ($N = {sel.sum()}$)', fontsize=8)
            _redshift_top_axis(ax)
            if row == 0 and col == 0:
                legend(ax, loc='lower left', fontsize=6.5)
    fig.text(0.5, 0.003, r'vertical lines: median host passes $10^{13}$ (dotted), '
             r'$5\times10^{13}$ (dashed), $10^{14}\,{\rm M}_\odot$ (dash-dotted).  '
             r'Kimmig+25, Magneticum, $z = 0$: ICL $\approx 25\%$, BCG $\approx 40\%$ '
             r'(BCG/ICL split at $0.1\,R_{200c}$)',
             ha='center', fontsize=7, color='0.35')
    fig.tight_layout(rect=(0, 0.015, 1, 1))
    save_figure(fig, 'fig11_kimmig_budget')


# ============ FIGURE 12: THE SHREDDING RATE (their fig 12) ============

_SHRED_ROWS = (('all', 0.0, np.inf), ('small groups', 1e13, 5e13),
               ('massive groups', 5e13, 1e14), ('clusters', 1e14, np.inf))
_SHRED_COLS = (('$z > 0.8$', 0.8, np.inf), ('$0.8 > z > 0.2$', 0.2, 0.8),
               ('$z < 0.2$', -1.0, 0.2))
_DM_BINS = np.arange(-0.3, 0.6001, 0.05)


def shredding_pairs(K):
    """(z2, M2, dM/M2, df per window) for every main-branch snapshot pair ~T apart.

    For each later snapshot the earlier one is the snapshot closest to
    KIMMIG_WINDOW_GYR before it; df is rescaled to exactly one window, since the
    snapshot spacing rarely lands on it.
    """
    out = {k: [] for k in ('z2', 'm2', 'dm', 'df')}
    snaps = np.array([s for s in SNAPS if s <= Z0_SNAP])
    for s2 in snaps:
        dt = AGE_AT_SNAP[s2] - AGE_AT_SNAP[snaps]
        cand = np.flatnonzero(dt > 0)
        if cand.size == 0:
            continue
        s1 = snaps[cand[np.argmin(np.abs(dt[cand] - KIMMIG_WINDOW_GYR))]]
        dt12 = AGE_AT_SNAP[s2] - AGE_AT_SNAP[s1]
        if abs(dt12 - KIMMIG_WINDOW_GYR) > 0.5 * KIMMIG_WINDOW_GYR:
            continue
        m1, m2 = K['mhost'][:, s1], K['mhost'][:, s2]
        f1, f2 = K['f'][:, s1], K['f'][:, s2]
        ok = np.isfinite(m1) & np.isfinite(m2) & np.isfinite(f1) & np.isfinite(f2) & (m2 > 0)
        out['z2'].append(np.full(ok.sum(), REDSHIFTS[s2]))
        out['m2'].append(m2[ok])
        out['dm'].append((m2[ok] - m1[ok]) / m2[ok])
        out['df'].append((f2[ok] - f1[ok]) * KIMMIG_WINDOW_GYR / dt12)
    return {k: np.concatenate(v) if v else np.array([]) for k, v in out.items()}


def shredding_rate(dm, df):
    """y0: the median change in f where the halo did not grow, read off the median line."""
    c, p = binned_percentiles(dm, df, _DM_BINS, min_count=KIMMIG_MIN)
    ok = np.isfinite(p[1])
    if ok.sum() >= 2 and c[ok].min() < 0 < c[ok].max():
        return float(np.interp(0.0, c[ok], p[1][ok]))
    near = np.abs(dm) < 0.025
    return float(np.median(df[near])) if near.sum() >= KIMMIG_MIN else np.nan


def plot_12_shredding(runs):
    """Change in f_ICL+BCG over ~1 Gyr against fractional halo growth.

    The plasma histogram (fully opaque) is the published run's distribution;
    lines are each run's median with y0 -- the shredding rate -- in the legend,
    in colours plasma does not contain (cyan, green, white) with a black edge so
    they read over any part of the colour map.  The two black dashed lines bound
    Kimmig+25's 3-4 per cent per Gyr.  The main branches are those of the z = 0
    sample, binned by their mass at the later snapshot, as they do.
    """
    print('Figure 12: the shredding rate')
    import matplotlib.patheffects as pe
    edge = [pe.Stroke(linewidth=4.2, foreground='black'), pe.Normal()]
    line_colours = ('#00e5ff', '#39ff14', '#ffffff', '#ff4fd8')
    order = sorted(a for a, _, _, _ in runs)
    shred_colour = {a: line_colours[k % len(line_colours)] for k, a in enumerate(order)}
    fig, axes = panel_grid(len(_SHRED_ROWS), len(_SHRED_COLS), squeeze=False,
                           sharex=True, sharey=True)
    pairs = [(alpha, colour, shredding_pairs(K)) for alpha, hdr, colour, K in runs]
    alpha_f = _fiducial(runs)[0]
    for i, (rname, mlo, mhi) in enumerate(_SHRED_ROWS):
        for j, (cname, zlo, zhi) in enumerate(_SHRED_COLS):
            ax = axes[i, j]
            for k, yk in enumerate(KIMMIG_SHRED):
                ax.axhline(yk, color='black', lw=1.1, ls='--', zorder=Z_LINE + 2,
                           label=r'Kimmig+25 $y_0$' if k == 0 else None)
            ax.axhline(0, color='0.6', lw=0.7, zorder=Z_LINE + 1)
            ax.axvline(0, color='0.6', lw=0.7, zorder=Z_LINE + 1)
            for alpha, colour, P in pairs:
                sel = ((P['m2'] >= mlo) & (P['m2'] < mhi) &
                       (P['z2'] > zlo) & (P['z2'] <= zhi))
                if alpha == alpha_f and sel.sum():
                    ax.hist2d(P['dm'][sel], P['df'][sel], bins=(40, 40),
                              range=((-0.3, 0.6), (-0.4, 0.4)), cmap='plasma',
                              cmin=1, zorder=Z_BAND, rasterized=True)
                c, p = binned_percentiles(P['dm'][sel], P['df'][sel], _DM_BINS,
                                          min_count=KIMMIG_MIN)
                ok = np.isfinite(p[1])
                y0 = shredding_rate(P['dm'][sel], P['df'][sel]) if sel.sum() else np.nan
                lab = alpha_label(alpha) + (rf', $y_0 = {y0 * 100:.1f}\%$'
                                            if np.isfinite(y0) else '')
                st = line_style(alpha)
                st['zorder'] = Z_LINE + 10
                st['lw'] = 2.2
                ax.plot(c[ok], p[1][ok], color=shred_colour[alpha], label=lab,
                        path_effects=edge, **st)
            ax.set_title(f'{rname}, {cname}', fontsize=8)
            leg = legend(ax, loc='lower left', fontsize=6.0, frameon=True)
            leg.get_frame().set_facecolor('white')
            leg.get_frame().set_alpha(0.85)
            leg.get_frame().set_edgecolor('0.6')
            leg.set_zorder(Z_LINE + 20)
            if i == len(_SHRED_ROWS) - 1:
                ax.set_xlabel(r'$\delta M = \Delta M_{\rm vir} / M_{\rm vir}(t_2)$')
            if j == 0:
                ax.set_ylabel(rf'$\Delta f_{{\rm ICL+BCG}}$ per {KIMMIG_WINDOW_GYR:g} Gyr')
    axes[0, 0].set_xlim(-0.3, 0.6)
    axes[0, 0].set_ylim(-0.4, 0.4)
    fig.tight_layout()
    save_figure(fig, 'fig12_kimmig_shredding')


def report_kimmig(runs):
    """The comparison in numbers, SAGE26 beside the paper's four simulations."""
    print()
    print(f'Kimmig+25 comparison: hosts with Mvir > {KIMMIG_HOST_LO:.1e} Msun at '
          f'Snap_{Z0_SNAP}, satellites inside '
          f'{"the FOF group" if KIMMIG_APERTURE is None else f"{KIMMIG_APERTURE:g} Rvir"}')
    ref = KIMMIG_REF
    rows = (('N hosts', None),
            ('median z_form', 'median_zform'), ('median f_ICL+BCG', 'median_f'),
            ('r(z_form, f)', 'r_zform_f'), ('r(z_form, f_sub)', 'r_zform_fsub'),
            ('r(f, log M12)', 'r_f_M12'), ('r(f, log M14)', 'r_f_M14'),
            ('r(f, log s_stars)', 'r_f_sstars'), ('r(f, log phi_BCG)', 'r_f_phi'),
            ('f early tail', None), ('f late tail', None),
            ('BCG / total', None), ('ICS / total', None),
            ('r(z_form, f_ICS)', None), ('r(z_form, BCG/total)', None),
            ('ICS in place z=2', None), ('ICS in place z=1', None),
            ('ICS in place z=0.5', None), ('BCG in place z=1', None),
            ('BCG in place z=0.5', None), ('z_half ICS', None), ('z_half BCG', None),
            ('y0 groups', None), ('y0 clusters', None))
    head = f'{"":>20}' + ''.join(f'{"beta_clock=" + format(a, "g"):>22}'
                                 for a, _, _, _ in runs)
    head += ''.join(f'{s:>13}' for s in KIMMIG_SIMS)
    print(head)
    for name, rk in rows:
        vals = []
        for alpha, hdr, colour, K in runs:
            z0, zf = K['z0'], K['zform']
            lzf = lambda k: np.log10(_get(K, k))
            with np.errstate(invalid='ignore', divide='ignore'):
                v = {
                    'N hosts': float(np.isfinite(zf).sum()),
                    'median z_form': np.nanmedian(zf),
                    'median f_ICL+BCG': np.nanmedian(z0['f']),
                    'r(z_form, f)': _pearson(zf, z0['f']),
                    'r(z_form, f_sub)': _pearson(zf, z0['fsub']),
                    'r(f, log M12)': _pearson(z0['f'], lzf('M12')),
                    'r(f, log M14)': _pearson(z0['f'], lzf('M14')),
                    'r(f, log s_stars)': _pearson(z0['f'], lzf('s_stars')),
                    'r(f, log phi_BCG)': _pearson(z0['f'], lzf('phi')),
                    'f early tail': np.nanmedian(z0['f'][K['early']]),
                    'f late tail': np.nanmedian(z0['f'][K['late']]),
                    'BCG / total': np.nanmean(z0['m_bcg'] / (z0['m_bcg'] + z0['m_ics'] + z0['m_sat'])),
                    'ICS / total': np.nanmean(z0['m_ics'] / (z0['m_bcg'] + z0['m_ics'] + z0['m_sat'])),
                    'r(z_form, f_ICS)': _pearson(zf, z0['m_ics'] / (z0['m_bcg'] + z0['m_ics'] + z0['m_sat'])),
                    'r(z_form, BCG/total)': _pearson(zf, z0['m_bcg'] / (z0['m_bcg'] + z0['m_ics'] + z0['m_sat'])),
                    'ICS in place z=2': _in_place(K['m_ics'], 2.0),
                    'ICS in place z=1': _in_place(K['m_ics'], 1.0),
                    'ICS in place z=0.5': _in_place(K['m_ics'], 0.5),
                    'BCG in place z=1': _in_place(K['m_bcg'], 1.0),
                    'BCG in place z=0.5': _in_place(K['m_bcg'], 0.5),
                    'z_half ICS': np.nanmedian(_zform(K['m_ics'])),
                    'z_half BCG': np.nanmedian(_zform(K['m_bcg'])),
                }.get(name)
            if name.startswith('y0'):
                P = shredding_pairs(K)
                lo, hi = (1e13, 1e14) if name == 'y0 groups' else (1e14, np.inf)
                sel = (P['m2'] >= lo) & (P['m2'] < hi)
                v = shredding_rate(P['dm'][sel], P['df'][sel]) if sel.sum() else np.nan
            vals.append(v)
        line = f'{name:>20}' + ''.join(f'{v:>22.3g}' if np.isfinite(v) else f'{"--":>22}'
                                       for v in vals)
        if rk:
            line += ''.join(f'{x:>13.2f}' for x in ref[rk])
        elif name == 'f early tail':
            line += f'{KIMMIG_TAIL_F["early"]:>13.2f}'
        elif name == 'f late tail':
            line += f'{KIMMIG_TAIL_F["late"]:>13.2f}'
        elif name.startswith('y0'):
            line += f'{"0.03-0.04":>13}'
        print(line)
    print('  "in place" = median fraction of the z = 0 main-branch mass already on the main')
    print('  progenitor at that redshift (arrival, not formation: group ICS counts on infall);')
    print('  z_half = median redshift at which the main branch first held half its z = 0 mass;')
    print('  f_ICS here = ICS / (BCG + ICS + satellites in the aperture); y0 per Gyr.')
    print('  Pearson r with bootstrap 16-84th range and the number of hosts:')
    for alpha, hdr, colour, K in runs:
        z0, zf = K['z0'], K['zform']
        tot = z0['m_bcg'] + z0['m_ics'] + z0['m_sat']
        with np.errstate(invalid='ignore', divide='ignore'):
            for label, y in (('r(z_form, f_ICL+BCG)', z0['f']),
                             ('r(z_form, BCG/total)', z0['m_bcg'] / tot),
                             ('r(z_form, ICS/total)', z0['m_ics'] / tot)):
                r, lo, hi, n = _pearson_boot(zf, y)
                print(f'    beta_clock = {alpha:<4g} {label:<22} {_f(r, ".3f", 6)} '
                      f'[{_f(lo, ".3f", 6)}, {_f(hi, ".3f", 6)}]  N = {n:,d}')

    # per-cluster listing: small samples deserve to be seen whole
    alpha_f, _, _, Kf = _fiducial(runs)
    n = len(Kf['zform'])
    if n <= 40:
        print()
        print(f'  every host, beta_clock = {alpha_f:g} '
              f'(log Mvir, z_form, f_ICL+BCG, f_ICS, log m_BCG, log M12, f_sub)')
        z0 = Kf['z0']
        tot = z0['m_bcg'] + z0['m_ics'] + z0['m_sat']
        for i in np.argsort(-z0['mhost']):
            with np.errstate(invalid='ignore', divide='ignore'):
                print(f'    {np.log10(z0["mhost"][i]):6.2f}  {Kf["zform"][i]:5.2f}  '
                      f'{z0["f"][i]:5.2f}  {z0["m_ics"][i] / tot[i]:5.2f}  '
                      f'{np.log10(z0["m_bcg"][i]):6.2f}  {np.log10(Kf["M12"][i]):5.2f}  '
                      f'{z0["fsub"][i]:5.3f}')


def _in_place(H, z):
    """Median fraction of each history's z = 0 value already present at redshift *z*."""
    s = int(np.argmin(np.abs(REDSHIFTS[:Z0_SNAP + 1] - z)))
    with np.errstate(invalid='ignore', divide='ignore'):
        r = np.nan_to_num(H[:, s]) / H[:, Z0_SNAP]
    return float(np.nanmedian(r[np.isfinite(r)])) if np.isfinite(r).any() else np.nan


# ========================== THE PEARCE ET AL. COMPARISON ==========================
#
# Pearce et al. (2026, arXiv:2610.06660) trace the central stars of The Three
# Hundred clusters back through their merger trees and ask which infall events
# built the BCG.  Every halo that falls onto the cluster's main branch is one
# event, placed by its lookback time and mass ratio
#
#     mu = M_host,max / M_sat,max
#
# (each the largest mass reached before infall), and credited with the share of
# the final BCG stellar mass it brought.  Half the BCG comes from N50 = 3 events
# (16-84th: 3-5), the four largest giving 24.1, 15.1, 10.4 and 7.6 per cent,
# about 10 per cent is in situ, and most events give nothing (their figs 2-5).
#
# The SAGE26 equivalent, per z = 0 cluster:
#   * the main branch is the z = 0 central's GalaxyIndex, as in the Kimmig scan;
#   * an infall event is the FOF group a galaxy sat in the snapshot before it
#     first joined the main branch's FOF (its CentralGalaxyIndex then); a galaxy
#     that first appears already inside the cluster is an event of its own;
#   * galaxies in the cluster when tracking starts (PEARCE_Z_START, their
#     z = 3.5) are in situ, as their stars belonging to the cluster then are;
#   * an event's BCG contribution is the stellar mass, at its last snapshot, of
#     each of its galaxies that merges onto the main branch's galaxy (mergeType
#     1 or 2, target from mergeIntoID).  Orphans merge onto their own subhalo's
#     galaxy, which may be a satellite; that satellite then carries the mass to
#     the BCG if it merges, so nothing is counted twice;
#   * in situ = the BCG's z = 0 mass minus every event's contribution: stars
#     formed in the central, the bursts mergers trigger there, and galaxies
#     already in the cluster at PEARCE_Z_START.
#
# Differences to keep in mind:
#   * their BCG is the cluster-bound stellar mass inside 147.6 kpc; ours is the
#     central galaxy, with no aperture and no ICS;
#   * M_sat,max is the largest Mvir the event's central reached while a central,
#     M_host,max the largest the main central reached, both before the infall
#     snapshot; Mvir stands in for their AHF M200c, and infall is joining the
#     FOF, not crossing R200;
#   * mergeIntoID counts from the first row of the forest's block in the target
#     snapshot.  The forest is GalaxyIndex // ForestNr_Mulfac, not SAGETreeIndex,
#     which restarts in every input tree file (see _forest_mulfac).  Checked on
#     microUchuu: every target resolved this way gains at least the stellar mass
#     merged into it;
#   * credit[c, s] (mass credited at each snapshot) and mbcg_hist[c, s] (the
#     main branch's stellar mass) are kept so report_pearce can check that no
#     step credits more than the BCG grew;
#   * a cluster whose central is a satellite at any point within
#     PEARCE_ALIGN_GYR is a central swap, the case their stacks exclude
#     (their section 2.1); it is dropped from figures 19 and 20.

PEARCE_HOST_LO = KIMMIG_HOST_LO   # z = 0 sample: Mvir above this [Msun]; theirs is M200c > 7.5e14
PEARCE_M200C_LO = 7.5e14          # their sample's lower M200c, for the matched subsample in the report
PEARCE_Z_START = 3.5              # tracking starts here; the cluster's stars then are in situ
PEARCE_ALIGN_GYR = 10.0           # no central swap allowed within this lookback time (their fig 4)
PEARCE_MU_MAX = 3.0               # figure 19 truncates log10 mu here, as their fig 4
PEARCE_N_RANK = 10                # ranked events shown in figure 20, as their pies

# Which clusters figures 19-21 stack (chosen after the cache is read, so a
# switch needs no rescan; figures 18 and 22 always use the most massive):
#   'all'   every cluster above PEARCE_HOST_LO without a central swap
#   'range' only those in their mass range, M200c >= PEARCE_M200C_LO
#   'top'   the PEARCE_N_MATCH most massive without a central swap, the size of
#           their hindcast sample (the 107 clusters of their figs 4 and 5)
# Set here or with --pearce=all|range|top on the command line.
PEARCE_SAMPLE = 'all'
PEARCE_N_MATCH = 107
PEARCE_M200C_RANGE = (7.51e14, 3.97e15)   # their hindcast sample's M200c span, for the report
PEARCE_CACHE_VERSION = 'p5'

# Their numbers.  Mean ranked fractions of the BCG-aperture mass (fig 5, the
# five labelled slices), in situ and the remainder beyond the top ten; and the
# N50 distribution quoted in section 3 as cumulative fractions of clusters.
PEARCE_RANKED = (0.241, 0.151, 0.104, 0.076, 0.058)
PEARCE_INSITU, PEARCE_REMAINDER = 0.090, 0.139
PEARCE_N50_CDF = ((3, 0.50), (4, 0.70), (5, 0.84))

# Boylan-Kolchin et al. (2008) eq. 5 with their coefficients, as Pearce et al.
# take it: circularity eta = 0.5 for the line, and a band over eta standing in
# for their 15-85th percentile orbits, at r_c(E) / r_vir = 1.
BK08 = dict(A=0.216, b=1.3, c=1.9, d=1.0)
PEARCE_ETA, PEARCE_ETA_BAND, PEARCE_RC = 0.5, (0.25, 0.75), 1.0

# The event colour is a share of the final BCG mass, a magnitude: one hue,
# light to dark, capped at 0.5 as theirs.  Area is proportional to the stellar
# mass the event brought in, PEARCE_AREA points^2 per PEARCE_AREA_MASS.
PEARCE_CMAP = plt.get_cmap('Reds')
PEARCE_FMAX = 0.5
PEARCE_AREA, PEARCE_AREA_MASS = 60.0, 1e11
PEARCE_STACK_SHRINK = 5.0         # their fig 4 draws areas a factor 5 smaller than fig 2

_P_PROPS = ('Type', 'GalaxyIndex', 'CentralGalaxyIndex', 'StellarMass',
            'IntraClusterStars', 'Mvir', 'mergeType', 'mergeIntoID')
_P_EVENT = ('cl', 'snap', 'mu', 'm_in', 'm_bcg')
_P_CLUSTER = ('mhost', 'mbcg', 'swap', 'started', 'm_insitu_merged', 'credit', 'mbcg_hist',
              'gi0', 'file_idx')


def _forest_mulfac(f):
    """ForestNr_Mulfac of an open model file.  GalaxyIndex // it is the forest,
    unique within the run.  SAGETreeIndex is not: it is the tree's number in its
    original tree file and restarts in each, so when a run reads several tree
    files (miniUchuu) it repeats within an output file and is not sorted."""
    return int(f['TreeInfo'].attrs['ForestNr_Mulfac'])


def _forest_blocks(fkey):
    """Each forest's contiguous block in one snapshot: (keys sorted, first row,
    end row, number of forests found in more than one block).  SAGE writes a
    forest's galaxies together, so the last number should be zero."""
    if fkey.size == 0:
        e = np.array([], np.int64)
        return e, e, e, 0
    start = np.flatnonzero(np.r_[True, fkey[1:] != fkey[:-1]])
    end = np.r_[start[1:], fkey.size]
    k = fkey[start]
    o = np.argsort(k, kind='stable')
    k, start, end = k[o], start[o], end[o]
    return k, start, end, int((k[1:] == k[:-1]).sum())


def _forest_first(d, fk):
    """First row (in the file) of forest *fk* at the snapshot of *d*, -1 if absent."""
    i = np.searchsorted(d['_fk'], fk)
    return int(d['_ff'][i]) if i < d['_fk'].size and d['_fk'][i] == fk else -1


def _p_read(g, forests, conv, fmul, props=_P_PROPS):
    """The rows of *forests* in one snapshot, whole forests only, masses in Msun.

    'tree' is each row's forest (GalaxyIndex // ForestNr_Mulfac) and 'first' the
    first row of its block, the origin mergeIntoID counts from.
    """
    fkey = g['GalaxyIndex'][:].astype(np.int64) // fmul
    k, st, en, dup = _forest_blocks(fkey)
    pos = np.clip(np.searchsorted(k, forests), 0, max(k.size - 1, 0))
    hit = (k[pos] == forests) if k.size else np.zeros(len(forests), bool)
    lo, hi = st[pos[hit]], en[pos[hit]]
    order = np.argsort(lo)
    lo, hi = lo[order], hi[order]
    if lo.size == 0:
        return None
    rows = np.concatenate([np.arange(a, b) for a, b in zip(lo, hi)])
    runs = _row_runs(rows)
    covered = np.concatenate([np.arange(a, b) for a, b in runs])
    idx = np.searchsorted(covered, rows)
    d = {p: np.concatenate([g[p][a:b] for a, b in runs])[idx] for p in props}
    for p in ('GalaxyIndex', 'CentralGalaxyIndex'):
        d[p] = d[p].astype(np.int64)
    for p in ('StellarMass', 'IntraClusterStars', 'Mvir', 'ICS_disrupt', 'ICS_accrete'):
        if p in d:
            d[p] = d[p].astype(np.float64) * conv
    d['tree'] = fkey[rows]
    d['first'] = np.repeat(lo, hi - lo)
    d['row'] = rows
    d['_fk'] = np.sort(k[pos[hit]])
    d['_ff'] = st[pos[hit]][np.argsort(k[pos[hit]])]
    d['_dup'] = dup
    # lookups by galaxy, and the stellar mass (galaxies + ICS) of every FOF group
    d['_o'] = np.argsort(d['GalaxyIndex'], kind='stable')
    d['_k'] = d['GalaxyIndex'][d['_o']]
    ug, inv = np.unique(d['CentralGalaxyIndex'], return_inverse=True)
    d['_grp'] = ug
    d['_grp_m'] = np.bincount(inv, weights=d['StellarMass'] + d['IntraClusterStars'])
    return d


def _p_find(d, gal):
    """Row of each GalaxyIndex in *gal* within snapshot data *d*, -1 where absent."""
    gal = np.atleast_1d(np.asarray(gal, np.int64))
    if d is None or d['_k'].size == 0:
        return np.full(gal.size, -1)
    pos = np.clip(np.searchsorted(d['_k'], gal), 0, d['_k'].size - 1)
    return np.where(d['_k'][pos] == gal, d['_o'][pos], -1)


def _p_lookup(keys, vals, q, missing=np.nan):
    """vals[keys == q] for a sorted *keys*, *missing* where q is absent."""
    if keys.size == 0:
        return missing
    p = min(int(np.searchsorted(keys, q)), keys.size - 1)
    return vals[p] if keys[p] == q else missing


def _pearce_file(args):
    """Every infall event onto the main branch of each cluster in one file.

    Snapshots are walked forward.  At each one the main branch's FOF members
    are found, newcomers are assigned to an event, and the destructions
    recorded on the previous snapshot are credited to the event of the galaxy
    destroyed onto the main branch, when it merges.

    An optional third argument replays other merger rules in the same pass
    (see THE POSITIONAL RULE TEST): dict(gal, merged, mass, bcg_gal, bcg_mass)
    with merged and mass of shape (rules, destroyed galaxies) -- whether each
    destroyed galaxy merged under each rule and the stellar mass it carried --
    and bcg_mass (rules, clusters) the z = 0 BCG masses each rule produces.
    Membership and events do not depend on the rule, so they are found once;
    only the merged mass is kept per rule.  An event is made by any destruction
    onto the main branch, merged or not, so every rule sees the same events.

    Returns a list with one (clusters, events) pair per rule.
    """
    fp, conv = args[:2]
    ov = args[2] if len(args) > 2 else None
    nr = 1 if ov is None else ov['merged'].shape[0]
    lb_at = AGE_AT_SNAP[Z0_SNAP] - AGE_AT_SNAP
    s_start = int(np.flatnonzero(REDSHIFTS <= PEARCE_Z_START)[0])

    with h5.File(fp, 'r') as f:
        g = f[f'Snap_{Z0_SNAP}']
        if g['Type'].shape[0] == 0:
            return None
        typ, mvir = g['Type'][:], g['Mvir'][:] * conv
        sel = (typ == 0) & (mvir >= PEARCE_HOST_LO)
        if not sel.any():
            return None
        gi0 = g['GalaxyIndex'][:][sel].astype(np.int64)
        fmul = _forest_mulfac(f)
        trees = np.unique(gi0 // fmul)
        nc = gi0.size
        if ov is None:
            mbcg = (g['StellarMass'][:][sel] * conv)[None]
        else:
            mbcg = np.array([[_p_lookup(ov['bcg_gal'], ov['bcg_mass'][r], x) for x in gi0]
                             for r in range(nr)])
        C = dict(mhost=mvir[sel], mbcg=mbcg, gi0=gi0, file_idx=np.full(gi0.size, -1),
                 swap=np.zeros(nc, bool), started=np.zeros(nc, bool),
                 m_insitu_merged=np.zeros((nr, nc)),
                 credit=np.zeros((nr, nc, len(REDSHIFTS))),    # merged mass landing at each snapshot
                 mbcg_hist=np.full((nc, len(REDSHIFTS)), np.nan))  # the main branch's stellar mass
        start_snap = np.full(nc, -1)
        gi0_to_c = {int(x): i for i, x in enumerate(gi0)}
        E = {k: [] for k in _P_EVENT}                 # m_bcg holds one array of nr per event
        events = {}                                   # (cluster, key) -> event
        member = {}                                   # GalaxyIndex -> (cluster, event); -1 is in situ
        mem_gal = np.array([], np.int64)
        pk_k, pk_v = np.array([], np.int64), np.array([])  # peak Mvir while a central

        def event(c, key, s, prev, cur):
            """The event (c, key) falling in at snapshot s, created on first sight."""
            ev = events.get((c, key))
            if ev is not None:
                return ev
            host = _p_lookup(pk_k, pk_v, gi0[c])
            sat = _p_lookup(pk_k, pk_v, key)
            if not np.isfinite(sat):
                r = _p_find(cur, key)[0]
                sat = cur['Mvir'][r] if r >= 0 else np.nan
            m_in = _p_lookup(prev['_grp'], prev['_grp_m'], key) if prev is not None else np.nan
            if not np.isfinite(m_in):
                r = _p_find(cur, key)[0]
                m_in = (cur['StellarMass'][r] + cur['IntraClusterStars'][r]) if r >= 0 else 0.0
            ev = len(E['cl'])
            events[(c, key)] = ev
            for k, v in zip(_P_EVENT, (c, s, host / sat if sat > 0 else np.nan, m_in, np.zeros(nr))):
                E[k].append(v)
            return ev

        prev, prev_snap = None, -1
        for s in range(Z0_SNAP + 1):
            key = f'Snap_{s}'
            d = (_p_read(f[key], trees, conv, fmul)
                 if key in f and f[key]['Type'].shape[0] else None)
            if d is None:
                continue
            if d['_dup']:
                print(f'    WARNING {os.path.basename(fp)} Snap_{s}: {d["_dup"]} forests '
                      f'split over more than one block; mergeIntoID may resolve wrongly', flush=True)

            main_cen = np.full(nc, -1, np.int64)       # FOF central of each main branch at s
            central_ok = np.zeros(nc, bool)            # the main branch is that central
            if s >= s_start:
                r0 = _p_find(d, gi0)
                found = r0 >= 0
                main_cen[found] = d['CentralGalaxyIndex'][r0[found]]
                C['mbcg_hist'][found, s] = d['StellarMass'][r0[found]]
                central_ok[found] = d['Type'][r0[found]] == 0
                C['swap'] |= found & ~central_ok & (lb_at[s] <= PEARCE_ALIGN_GYR)

                # membership: newcomers to each main branch's FOF
                cen_to_c = {}
                for c in np.flatnonzero(found):
                    cen_to_c.setdefault(int(main_cen[c]), c)
                rows = np.flatnonzero(np.isin(d['CentralGalaxyIndex'],
                                              np.fromiter(cen_to_c, np.int64)))
                new = rows[~np.isin(d['GalaxyIndex'][rows], mem_gal)]
                first_time = found & (start_snap < 0)
                start_snap[first_time] = s
                for r in new:
                    gal = int(d['GalaxyIndex'][r])
                    c = cen_to_c[int(d['CentralGalaxyIndex'][r])]
                    if start_snap[c] == s:
                        member[gal] = (c, -1)
                        continue
                    rp = _p_find(prev, gal)[0] if prev is not None else -1
                    k = int(prev['CentralGalaxyIndex'][rp]) if rp >= 0 else gal
                    member[gal] = (c, event(c, k, s, prev, d))
                if new.size:
                    mem_gal = np.concatenate([mem_gal, d['GalaxyIndex'][new]])

                # destructions recorded on the previous snapshot land at this one,
                # on the galaxy mergeIntoID names within the forest's block at s
                if prev is not None and prev_snap == s - 1:
                    for r in np.flatnonzero(np.isin(prev['mergeType'], (1, 2, 4))):
                        if ov is None:
                            merged = np.array([prev['mergeType'][r] in (1, 2)])
                            m = np.array([prev['StellarMass'][r]])
                        else:
                            i = np.searchsorted(ov['gal'], prev['GalaxyIndex'][r])
                            if i >= ov['gal'].size or ov['gal'][i] != prev['GalaxyIndex'][r]:
                                continue
                            merged, m = ov['merged'][:, i], ov['mass'][:, i]
                        f0 = _forest_first(d, prev['tree'][r])
                        if f0 < 0:
                            continue
                        t_row = f0 + int(prev['mergeIntoID'][r])
                        j = np.searchsorted(d['row'], t_row)
                        if j >= d['row'].size or d['row'][j] != t_row:
                            continue
                        c = gi0_to_c.get(int(d['GalaxyIndex'][j]))
                        if c is None or start_snap[c] < 0 or start_snap[c] > s - 1:
                            continue
                        gal = int(prev['GalaxyIndex'][r])
                        owner = member.get(gal)
                        if owner is not None and owner[0] == c:
                            ev = owner[1]
                        else:
                            # not yet a member: its group fell in during this step
                            ev = event(c, int(prev['CentralGalaxyIndex'][r]), s, prev, d)
                        add = np.where(merged, m, 0.0)
                        if ev < 0:
                            C['m_insitu_merged'][:, c] += add
                        else:
                            E['m_bcg'][ev] += add
                        C['credit'][:, c, s] += add

            # the peak Mvir of every central, after this snapshot's infall used the old one
            cen = d['Type'] == 0
            k = np.concatenate([pk_k, d['GalaxyIndex'][cen]])
            v = np.concatenate([pk_v, d['Mvir'][cen]])
            o = np.lexsort((-v, k))
            k, v = k[o], v[o]
            keep = np.r_[True, k[1:] != k[:-1]]
            pk_k, pk_v = k[keep], v[keep]
            prev, prev_snap = d, s

        C['started'] = (start_snap >= 0) & (start_snap <= s_start)

    nev = len(E['cl'])
    m_bcg = np.array(E['m_bcg']).reshape(nev, nr)
    base = {k: np.asarray(E[k], dtype=float if k not in ('cl', 'snap') else int)
            for k in _P_EVENT if k != 'm_bcg'}
    out = []
    for r in range(nr):
        Cr = dict(C, mbcg=C['mbcg'][r], m_insitu_merged=C['m_insitu_merged'][r],
                  credit=C['credit'][r])
        if ov is not None:
            Cr['mbcg_hist'] = np.full_like(C['mbcg_hist'], np.nan)  # the observed history no longer applies
        out.append((Cr, dict(base, m_bcg=m_bcg[:, r])))
    return out


def scan_pearce(hdr, verbose=True, overrides=None, prefix='    '):
    """Every infall event onto every z = 0 cluster of one run.

    Without *overrides* returns one dict of clusters and events.  With
    *overrides* -- one dict per file, each carrying every rule (see
    _pearce_file) -- returns a list of such dicts, one per rule, from a single
    pass over the files.
    """
    files, conv = hdr['files'], hdr['mass_convert']
    jobs = min(FILE_JOBS, len(files))
    if verbose and overrides is None:
        print(f'  {len(files)} files, {jobs} worker(s)')
    t_start = time.time()
    work = [(fp, conv) + ((overrides[i],) if overrides and overrides[i] is not None else ())
            for i, fp in enumerate(files)]
    pool = ProcessPoolExecutor(max_workers=jobs) if jobs > 1 else None
    it = pool.map(_pearce_file, work) if pool else (_pearce_file(w) for w in work)
    parts = []
    for i, part in enumerate(it):
        if part is not None:
            for C, E in part:
                C['file_idx'][:] = i
        parts.append(part)
        if verbose:
            n = 0 if part is None else part[0][0]['mhost'].size
            print(f'{prefix}file {i + 1}/{len(work)}: {n} clusters '
                  f'[{time.time() - t_start:7.0f} s]', flush=True)
    if pool is not None:
        pool.shutdown()
    parts = [p for p in parts if p is not None]
    nr = len(parts[0]) if parts else 1

    def assemble(r):
        P, offset = {k: [] for k in _P_CLUSTER + tuple('ev_' + k for k in _P_EVENT)}, 0
        for part in parts:
            C, E = part[r]
            for k in _P_CLUSTER:
                P[k].append(C[k])
            for k in _P_EVENT:
                P['ev_' + k].append(E[k] + (offset if k == 'cl' else 0))
            offset += C['mhost'].size
        return {k: (np.concatenate(v) if v else np.array([])) for k, v in P.items()}

    out = [assemble(r) for r in range(nr)]
    return out if overrides is not None else out[0]


def load_pearce(hdr, refresh=False, verbose=True):
    """The infall events for one run, from the npz cache when current."""
    tag = os.path.basename(os.path.normpath(hdr['directory']))
    cache = os.path.join(CACHE_DIR, f'pearce_{PEARCE_CACHE_VERSION}_{tag}_'
                         f'{np.log10(PEARCE_HOST_LO):.2f}_z{PEARCE_Z_START:g}.npz')
    newest = max(os.path.getmtime(f) for f in hdr['files'])
    if not refresh and os.path.exists(cache) and os.path.getmtime(cache) > newest:
        if verbose:
            print(f'  infall events from cache: {cache}')
        z = np.load(cache)
        P = {k: z[k] for k in z.files}
    else:
        if verbose:
            print(f'  following infall onto cluster main branches in {hdr["directory"]}')
        P = scan_pearce(hdr, verbose=verbose)
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez_compressed(cache, **P)
    return derive_pearce(P)


_MVIR_PER_M200C = None


def _mvir_per_m200c():
    """Mvir / M200c for the NFW profile the module uses (cached; 1.22 without colossus)."""
    global _MVIR_PER_M200C
    if _MVIR_PER_M200C is None:
        try:
            _MVIR_PER_M200C = _m200c_to_mvir_factor()
        except Exception:
            _MVIR_PER_M200C = 1.22
    return _MVIR_PER_M200C


def _sample_tag():
    """Filename suffix for the stacked figures, empty for the default sample."""
    return '' if PEARCE_SAMPLE == 'all' else f'_{PEARCE_SAMPLE}'


def _pearce_sample(mhost, eligible):
    """The clusters PEARCE_SAMPLE stacks, out of the *eligible* ones."""
    if PEARCE_SAMPLE == 'all':
        return eligible.copy()
    if PEARCE_SAMPLE == 'range':
        return eligible & (mhost >= PEARCE_M200C_LO * _mvir_per_m200c())
    if PEARCE_SAMPLE == 'top':
        idx = np.flatnonzero(eligible)
        keep = idx[np.argsort(-mhost[idx])][:PEARCE_N_MATCH]
        out = np.zeros_like(eligible)
        out[keep] = True
        return out
    sys.exit(f'PEARCE_SAMPLE = {PEARCE_SAMPLE!r}: choose all, range or top')


def derive_pearce(P):
    """Each event's share of its BCG, each cluster's ranked shares and N50."""
    P = dict(P)
    ncl = P['mhost'].size
    P['ev_cl'] = P['ev_cl'].astype(int)
    P['ev_snap'] = P['ev_snap'].astype(int)
    P['ev_lookback'] = AGE_AT_SNAP[Z0_SNAP] - AGE_AT_SNAP[P['ev_snap']]
    with np.errstate(invalid='ignore', divide='ignore'):
        P['ev_f'] = P['ev_m_bcg'] / P['mbcg'][P['ev_cl']]
    P['eligible'] = P['started'].astype(bool) & ~P['swap'].astype(bool)
    P['stack'] = _pearce_sample(P['mhost'], P['eligible'])

    ranked = np.zeros((ncl, PEARCE_N_RANK))
    P['n50'] = np.full(ncl, np.nan)
    P['f_events'] = np.zeros(ncl)
    for c in range(ncl):
        f = np.sort(P['ev_f'][P['ev_cl'] == c])[::-1]
        f = f[np.isfinite(f)]
        P['f_events'][c] = f.sum()
        ranked[c, :min(f.size, PEARCE_N_RANK)] = f[:PEARCE_N_RANK]
        hit = np.flatnonzero(np.cumsum(f) >= 0.5)
        if hit.size:
            P['n50'][c] = hit[0] + 1
    P['ranked'] = ranked
    P['f_insitu'] = 1.0 - P['f_events']
    P['f_remainder'] = P['f_events'] - ranked.sum(axis=1)
    return P


def all_pearce(refresh=False):
    """(alpha, header, colour, events) for every run in the sweep."""
    out = []
    for alpha, hdr, colour in ALL_RUNS:
        print(f'alpha = {alpha:g}')
        out.append((alpha, hdr, colour, load_pearce(hdr, refresh=refresh)))
    return out


# ---------------- the merger-time curves -------------------------------------

def _t_dyn_gyr(z):
    """R_vir / V_vir at redshift z [Gyr], for the Bryan & Norman (1998) overdensity."""
    ez2 = OMEGA_M * (1 + z) ** 3 + OMEGA_L
    x = OMEGA_M * (1 + z) ** 3 / ez2 - 1.0
    delta = 18 * np.pi ** 2 + 82 * x - 39 * x ** 2
    return 977.8 / (100 * HUBBLE_H) / np.sqrt(ez2) / np.sqrt(delta / 2.0)


def _bk08_shape(mu, eta=PEARCE_ETA):
    """t_merge / t_dyn from Boylan-Kolchin et al. (2008) eq. 5."""
    return (BK08['A'] * mu ** BK08['b'] / np.log(1 + mu)
            * np.exp(BK08['c'] * eta) * PEARCE_RC ** BK08['d'])


def _clock_shape(mu, beta):
    """t_clock / t_dyn for SAGE26: beta 1.17 (M_host / m_sat) / ln(1 + M_host / m_sat)."""
    return beta * 1.17 * mu / np.log(1 + mu)


def _just_merged_mu(shape, lookback):
    """For each lookback time, the mu whose merger time equals it: an event that
    fell in then with a smaller mu has had time to merge.  NaN where none."""
    from scipy.optimize import brentq
    z_of = lambda t: np.interp(AGE_AT_SNAP[Z0_SNAP] - t, AGE_AT_SNAP, REDSHIFTS)
    out = np.full(len(lookback), np.nan)
    for i, t in enumerate(lookback):
        g = lambda lm: shape(10 ** lm) * _t_dyn_gyr(z_of(t)) - t
        lo, hi = -1.0, 6.0
        if g(lo) < 0 < g(hi):
            out[i] = brentq(g, lo, hi)
    return out


def _draw_merger_curves(ax, alpha, colour):
    """BK08 with its orbital band, and this run's own clock."""
    t = np.linspace(0.2, AGE_AT_SNAP[Z0_SNAP] - AGE_AT_SNAP[0] - 0.05, 120)
    t = t[t < AGE_AT_SNAP[Z0_SNAP] - 0.1]
    bands = [_just_merged_mu(lambda m, e=e: _bk08_shape(m, e), t) for e in PEARCE_ETA_BAND]
    ax.fill_between(t, bands[0], bands[1], color='#9ecae1', alpha=0.45, lw=0,
                    zorder=Z_BAND, label=rf'BK08, $\eta$ = {PEARCE_ETA_BAND[0]:g}-{PEARCE_ETA_BAND[1]:g}')
    ax.plot(t, _just_merged_mu(_bk08_shape, t), color='#3182bd', lw=1.8,
            zorder=Z_LINE, label=rf'BK08, $\eta$ = {PEARCE_ETA:g}')
    ax.plot(t, _just_merged_mu(lambda m: _clock_shape(m, alpha), t), color=colour,
            lw=1.8, ls='--', zorder=Z_LINE,
            label=rf'SAGE26 clock, $\beta_{{\rm clock}} = {alpha:g}$')


def _draw_events(ax, x, y, f, m_in, shrink=1.0, rasterized=False):
    """Events as circles: area by stellar mass brought in, fill by share of the BCG.
    Events that gave the BCG nothing are open, as theirs."""
    area = np.clip(PEARCE_AREA * m_in / PEARCE_AREA_MASS / shrink, 1.0, None)
    o = np.argsort(-area)                              # small ones drawn on top
    zero = (f[o] <= 0) | ~np.isfinite(f[o])
    ax.scatter(x[o][zero], y[o][zero], s=area[o][zero], facecolors='none',
               edgecolors='0.6', linewidths=0.4, zorder=Z_OBS, rasterized=rasterized)
    sc = ax.scatter(x[o][~zero], y[o][~zero], s=area[o][~zero], c=f[o][~zero],
                    cmap=PEARCE_CMAP, vmin=0, vmax=PEARCE_FMAX, edgecolors='0.3',
                    linewidths=0.4, zorder=Z_OBS + 1, rasterized=rasterized)
    return sc


def _event_axes(ax, ylim):
    ax.set_xlim(0, AGE_AT_SNAP[Z0_SNAP] - AGE_AT_SNAP[0])
    ax.set_ylim(*ylim)                                 # mu = 1 at the top, as theirs
    ax.set_xlabel('Lookback time [Gyr]')
    ax.set_ylabel(r'$\log_{10}(M_{\rm main} / M_{\rm sat})$')


def _size_key(ax, shrink=1.0):
    """Three reference circles for the stellar mass an event brought in."""
    hs = [ax.scatter([], [], s=PEARCE_AREA * m / PEARCE_AREA_MASS / shrink,
                     facecolors='none', edgecolors='0.3', linewidths=0.6,
                     label=rf'$10^{{{np.log10(m):.0f}}}\,{{\rm M}}_\odot$ in')
          for m in (1e10, 1e11, 1e12)]
    return hs


def _event_colourbar(fig, sc, axes):
    cb = fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.01)
    cb.set_label('fraction of final BCG mass brought in by this event')


def _event_layout(fig, axes, sc, shrink=1.0):
    """Room for a legend under the panels, out of the event cloud, then the colourbar."""
    fig.subplots_adjust(left=0.05, right=0.93, bottom=0.2, top=0.92, wspace=0.2)
    handles = [h for ax in axes for h in ax.get_legend_handles_labels()[0]]
    labels = [l for ax in axes for l in ax.get_legend_handles_labels()[1]]
    seen = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    keys = _size_key(axes[0], shrink)
    fig.legend(list(seen.values()) + keys, list(seen) + [k.get_label() for k in keys],
               loc='lower center', ncol=len(seen) + len(keys), frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.48, 0.0), scatterpoints=1)
    if sc is not None:
        _event_colourbar(fig, sc, axes)


# ============ FIGURE 18: ONE CLUSTER'S INFALL EVENTS (their fig 2) ============

def plot_18_infall_events(runs):
    fig, axes = panel_grid(1, len(runs), squeeze=False)
    axes = axes[0]
    sc = None
    for ax, (alpha, hdr, colour, P) in zip(axes, runs):
        pool = np.flatnonzero(P['stack']) if P['stack'].any() else np.arange(P['mhost'].size)
        if pool.size == 0:
            continue
        c = pool[np.argmax(P['mhost'][pool])]
        e = P['ev_cl'] == c
        sc = _draw_events(ax, P['ev_lookback'][e], np.log10(P['ev_mu'][e]),
                          P['ev_f'][e], P['ev_m_in'][e])
        _draw_merger_curves(ax, alpha, colour)
        _event_axes(ax, (5.5, -0.5))
        ax.set_title(alpha_label(alpha), fontsize=plt.rcParams['axes.labelsize'])
        ax.text(0.97, 0.03,
                rf'$M_{{\rm vir}} = 10^{{{np.log10(P["mhost"][c]):.2f}}}\,{{\rm M}}_\odot$'
                f'\n{e.sum()} events, '
                + (f'$N_{{50}}$ = {P["n50"][c]:.0f}' if np.isfinite(P['n50'][c])
                   else 'events give < half the BCG')
                + f'\nin situ {P["f_insitu"][c]:.2f}',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=7.5)
    _event_layout(fig, axes, sc)
    save_figure(fig, '18_infall_events_one_cluster')


# ============ FIGURE 19: EVERY CLUSTER'S INFALL EVENTS (their fig 4) ============

def plot_19_infall_events_stacked(runs):
    fig, axes = panel_grid(1, len(runs), squeeze=False)
    axes = axes[0]
    sc = None
    for ax, (alpha, hdr, colour, P) in zip(axes, runs):
        e = P['stack'][P['ev_cl']] & (np.log10(P['ev_mu']) <= PEARCE_MU_MAX)
        sc = _draw_events(ax, P['ev_lookback'][e], np.log10(P['ev_mu'][e]), P['ev_f'][e],
                          P['ev_m_in'][e], shrink=PEARCE_STACK_SHRINK, rasterized=True)
        _draw_merger_curves(ax, alpha, colour)
        _event_axes(ax, (PEARCE_MU_MAX, -0.5))
        ax.set_title(alpha_label(alpha), fontsize=plt.rcParams['axes.labelsize'])
        ax.text(0.97, 0.03, f'{int(P["stack"].sum())} clusters, {int(e.sum()):,} events',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=7.5)
    _event_layout(fig, axes, sc, PEARCE_STACK_SHRINK)
    save_figure(fig, '19_infall_events_stacked' + _sample_tag(), dpi=SCATTER_RASTER_DPI)


# ============ FIGURE 20: HOW FEW EVENTS BUILD THE BCG (their figs 3 and 5) ============

def plot_20_bcg_oligarchy(runs):
    """Left: mean cumulative share of the BCG from its k largest events.
    Right: the fraction of clusters with N50 <= N.  Pearce et al. in grey."""
    fig, (ax_r, ax_n) = panel_grid(1, 2)
    k = np.arange(1, PEARCE_N_RANK + 1)
    for alpha, hdr, colour, P in runs:
        st = P['stack']
        if not st.any():
            continue
        ax_r.plot(k, np.cumsum(P['ranked'][st].mean(axis=0)), color=colour,
                  marker='o', ms=4, label=alpha_label(alpha), **line_style(alpha))
        n50 = P['n50'][st]
        nn = np.arange(1, 16)
        ax_n.plot(nn, [(n50 <= n).sum() / st.sum() for n in nn], color=colour,
                  drawstyle='steps-post', label=alpha_label(alpha), **line_style(alpha))
    obs = dict(color='0.3', marker='D', ms=6, ls='none', mfc='0.75', mec='0.25',
               zorder=Z_LINE + 2, label='Pearce+26')
    ax_r.plot(np.arange(1, len(PEARCE_RANKED) + 1), np.cumsum(PEARCE_RANKED), **obs)
    ax_r.axhline(0.5, color='0.6', lw=0.8, ls=':')
    ax_r.set_xlim(0.5, PEARCE_N_RANK + 0.5)
    ax_r.set_ylim(0, 1)
    ax_r.set_xlabel(r'number of largest infall events, $k$')
    ax_r.set_ylabel('mean cumulative fraction of the BCG')
    legend(ax_r, loc='upper left')
    ax_n.plot(*zip(*PEARCE_N50_CDF), **obs)
    ax_n.set_xlim(0.5, 15)
    ax_n.set_ylim(0, 1.02)
    ax_n.set_xlabel(r'$N_{50}$')
    ax_n.set_ylabel(r'fraction of clusters with $N_{50} \leq$ $N$')
    # the runs plateau at very different heights; the gap between the lowest
    # two stays clear whatever the sample
    legend(ax_n, loc='center right', bbox_to_anchor=(1.0, 0.3))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.12, top=0.95, wspace=0.25)
    save_figure(fig, '20_bcg_oligarchy' + _sample_tag())


# ============ FIGURE 22: WHERE THE CLUSTER'S STARS WERE (their fig 1) ============
#
# Pearce et al. split every star particle near a z = 0 cluster into four
# components and then show where those same stars were at z = 0.3 and z ~ 1.
# SAGE26 has no particles, so the split is made per galaxy: every galaxy's
# stars share one z = 0 fate, found by following it forward --
#   * it survives: its stars are in the BCG (it is the main branch) or a
#     satellite of the cluster, or elsewhere;
#   * it merges onto a galaxy: its stars share that galaxy's fate;
#   * it is disrupted: its stars become ICS on its target, and follow the ICS.
# ICS moves with the galaxy that holds it: to the central when its holder
# becomes a satellite, and onto the target when its holder is destroyed.  At
# z = 0 the cluster's ICS is split as SAGE splits it: ICS_disrupt (satellites
# disrupted onto the main branch itself, "made in the cluster") and
# ICS_accrete (ICS that arrived already formed, carried in by groups --
# "pre-processed", the counterpart of their GroupICL).  Traced back, a star
# that is later disrupted onto the main branch is "made in the cluster"; one
# disrupted anywhere else and carried in is "pre-processed".
#
# The rows are therefore fixed at z = 0, as theirs, and each panel draws, at
# that epoch, the galaxies whose stars will end up in that row (filled
# circles) and the ICS that will (open diamonds, at the galaxy that holds it:
# SAGE's ICS has no position of its own).  Marker area is the mass that will
# end in the row.  Differences from theirs: rows are by fate, not by radius
# (their BCG is the bound stars inside 147.6 kpc); satellites are the
# cluster's own FOF members, not every galaxy in the cube; and only the
# cluster's own forest is read, so unrelated haloes in the cube are absent.

SPATIAL_Z = (0.0, 0.3, 1.0)            # their columns; z = 0 means Z0_SNAP
SPATIAL_CUBE = 2.4                     # panel side in units of the z = 0 Rvir (their 2.4 R200)
SPATIAL_HALO_MIN = 10 ** 12.5          # other haloes in the forest drawn with their Rvir circle [Msun]
SPATIAL_AREA, SPATIAL_AREA_MASS = 40.0, 1e11   # marker area (points^2) per this mass
SPATIAL_ROWS = (('bcg', 'BCG', '#e08214'),
                ('ics_in', 'ICS made in the cluster', '#2c5fbf'),
                ('sat', 'satellite galaxies', '#c51b3a'),
                ('ics_pre', 'ICS arrived pre-processed', '#2f8f46'))
_S_PROPS = _P_PROPS + ('Posx', 'Posy', 'Posz', 'Rvir', 'ICS_disrupt', 'ICS_accrete')


def _spatial_cluster(P):
    """The cluster figure 18 draws: the most massive one without a central swap."""
    pool = np.flatnonzero(P['stack']) if P['stack'].any() else np.arange(P['mhost'].size)
    return None if pool.size == 0 else pool[np.argmax(P['mhost'][pool])]


def trace_spatial(hdr, gi0, file_idx):
    """Every galaxy and every ICS holder of one cluster's forest at the epochs of
    SPATIAL_Z, each with the z = 0 row its stars end in.

    Returns {snap: dict(x, y, m, row, kind, ...)} in physical kpc about the main
    progenitor, kind 'gal' for stars in galaxies and 'ics' for ICS, plus the
    z = 0 truth for the validation.
    """
    import sys as _sys
    _sys.setrecursionlimit(max(_sys.getrecursionlimit(), 20000))
    t_start = time.time()
    conv, box, h = hdr['mass_convert'], hdr['box_size'], HUBBLE_H
    fp = hdr['files'][int(file_idx)]
    epochs = sorted({Z0_SNAP if z == 0 else _snap_near(z) for z in SPATIAL_Z})
    with h5.File(fp, 'r') as f:
        g = f[f'Snap_{Z0_SNAP}']
        fmul = _forest_mulfac(f)
        tree = np.array([gi0 // fmul], np.int64)
        D = {}
        for s in range(epochs[0], Z0_SNAP + 1):
            key = f'Snap_{s}'
            if key in f and f[key]['Type'].shape[0]:
                d = _p_read(f[key], tree, conv, fmul, _S_PROPS)
                if d is not None:
                    D[s] = d

    print(f'    forest {int(tree[0])} of {os.path.basename(fp)}: read Snap_{min(D)}-Snap_{max(D)}, '
          + ', '.join(f'{D[s]["GalaxyIndex"].size:,} galaxies at z = {REDSHIFTS[s]:.2f}'
                      for s in epochs if s in D)
          + f' [{time.time() - t_start:.0f} s]', flush=True)
    # each galaxy's last snapshot and, if destroyed, its fate and target
    last = {}
    for s in sorted(D):
        for x in D[s]['GalaxyIndex']:
            last[int(x)] = s
    fate = {}
    for gal, e in last.items():
        if e >= Z0_SNAP:
            continue
        d, nxt = D[e], D.get(e + 1)
        r = _p_find(d, gal)[0]
        mt = int(d['mergeType'][r])
        tgt = None
        if mt > 0 and nxt is not None:
            t_row = nxt['first'][0] + int(d['mergeIntoID'][r])
            j = np.searchsorted(nxt['row'], t_row)
            if j < nxt['row'].size and nxt['row'][j] == t_row:
                tgt = int(nxt['GalaxyIndex'][j])
        fate[gal] = (mt, tgt)
    n_dest = sum(1 for mt, t in fate.values() if mt > 0)
    n_unres = sum(1 for mt, t in fate.values() if mt > 0 and t is None)
    n_vanish = sum(1 for mt, t in fate.values() if mt == 0)
    print(f'    {len(last):,} galaxies in the forest since z = {REDSHIFTS[min(D)]:.2f}: '
          f'{n_dest:,} destroyed ({n_unres:,} with no resolvable target), '
          f'{n_vanish:,} vanish without a merger record', flush=True)

    memo_s, memo_i = {}, {}

    def ics_dest(c, s, ch):
        """Row of the ICS held by galaxy c from snapshot s; ch 'in' or 'pre'."""
        k = (c, s, ch)
        if k in memo_i:
            return memo_i[k]
        out = 'lost'
        for s2 in range(s, Z0_SNAP + 1):
            r = _p_find(D[s2], c)[0] if s2 in D else -1
            if r < 0:
                mt, t = fate.get(c, (0, None))
                out = ics_dest(t, s2, 'pre') if t is not None else 'lost'
                break
            if D[s2]['Type'][r] != 0:
                cen = int(D[s2]['CentralGalaxyIndex'][r])
                out = ics_dest(cen, s2, 'pre') if cen != c else 'lost'
                break
        else:
            out = f'ics_{ch}' if c == gi0 else 'out'
        memo_i[k] = out
        return out

    def stars_dest(gal):
        """Row the stars of galaxy gal end in."""
        if gal in memo_s:
            return memo_s[gal]
        e = last[gal]
        if e >= Z0_SNAP:
            r = _p_find(D[Z0_SNAP], gal)[0]
            out = ('bcg' if gal == gi0 else
                   'sat' if D[Z0_SNAP]['CentralGalaxyIndex'][r] == gi0 else 'out')
        else:
            mt, t = fate[gal]
            if t is None or t not in last:
                out = 'lost'
            elif mt in (1, 2):
                out = stars_dest(t)
            elif mt == 4:
                out = ics_dest(t, e + 1, 'in' if t == gi0 else 'pre')
            else:
                out = 'lost'
        memo_s[gal] = out
        return out

    T = {}
    for s in epochs:
        d = D[s]
        a = 1.0 / (1.0 + REDSHIFTS[s])
        rc = _p_find(d, gi0)[0]
        if rc < 0:
            continue
        pos = np.stack([d['Posx'], d['Posy'], d['Posz']], axis=1).astype(np.float64)
        dx = pos - pos[rc]
        dx -= box * np.round(dx / box)
        dx *= 1000.0 * a / h                                   # physical kpc
        xs, ys, ms, rows, kinds = [], [], [], [], []
        for r, gal in enumerate(d['GalaxyIndex']):
            gal = int(gal)
            if d['StellarMass'][r] > 0:
                xs.append(dx[r, 0]); ys.append(dx[r, 1]); ms.append(d['StellarMass'][r])
                rows.append(stars_dest(gal)); kinds.append('gal')
            ics = d['IntraClusterStars'][r]
            if ics > 0:
                if gal == gi0:
                    # the main branch's own ICS, split as SAGE records it
                    for part, ch in ((d['ICS_disrupt'][r], 'in'), (d['ICS_accrete'][r], 'pre')):
                        if part > 0:
                            xs.append(dx[r, 0]); ys.append(dx[r, 1]); ms.append(part)
                            rows.append(ics_dest(gal, s, ch)); kinds.append('ics')
                else:
                    xs.append(dx[r, 0]); ys.append(dx[r, 1]); ms.append(ics)
                    rows.append(ics_dest(gal, s, 'pre')); kinds.append('ics')
        halo = (d['Type'] == 0) & (d['Mvir'] >= SPATIAL_HALO_MIN)
        T[s] = dict(x=np.array(xs), y=np.array(ys), m=np.array(ms), row=np.array(rows),
                    kind=np.array(kinds), z=REDSHIFTS[s],
                    halo_x=dx[halo, 0], halo_y=dx[halo, 1],
                    halo_r=d['Rvir'][halo] * 1000.0 * a / h,
                    main_r=d['Rvir'][rc] * 1000.0 * a / h)
    # z = 0 truth, as SAGE wrote it
    d = D[Z0_SNAP]
    rc = _p_find(d, gi0)[0]
    sat = (d['CentralGalaxyIndex'] == gi0) & (d['GalaxyIndex'] != gi0)
    print(f'    traced {len(memo_s):,} galaxy fates and {len(memo_i):,} ICS holders '
          f'[{time.time() - t_start:.0f} s]', flush=True)
    truth = dict(bcg=d['StellarMass'][rc], ics_in=d['ICS_disrupt'][rc],
                 ics_pre=d['ICS_accrete'][rc] + d['IntraClusterStars'][sat].sum(),
                 sat=d['StellarMass'][sat].sum(), mvir=d['Mvir'][rc],
                 rvir=d['Rvir'][rc] * 1000.0 / (1.0 + REDSHIFTS[Z0_SNAP]) / h)
    return T, truth


def plot_22_spatial(runs):
    """One figure per run: the four z = 0 components of the figure 18 cluster,
    traced back to z ~ 0.3 and z ~ 1 (their fig 1)."""
    print()
    print('Figure 22: where the cluster\'s z = 0 stars were (Pearce+26 fig 1)')
    for alpha, hdr, colour, P in runs:
        c = _spatial_cluster(P)
        if c is None:
            print(f'  beta_clock = {alpha:g}: no cluster -- skipped')
            continue
        rank = int((P['mhost'] > P['mhost'][c]).sum()) + 1
        print(f'  beta_clock = {alpha:g}: cluster {c} (rank {rank} of {P["mhost"].size} by Mvir; '
              f'{"no central swap" if P["stack"][c] else "central swap -- none without one"}), '
              f'GalaxyIndex {int(P["gi0"][c])}, N50 = '
              + (f'{P["n50"][c]:.0f}' if np.isfinite(P['n50'][c]) else 'undefined')
              + f', in situ {P["f_insitu"][c]:.2f}')
        T, truth = trace_spatial(hdr, int(P['gi0'][c]), int(P['file_idx'][c]))
        snaps = sorted(T, reverse=True)
        half = 0.5 * SPATIAL_CUBE * truth['rvir']
        print(f'    Mvir = {truth["mvir"]:.3e} Msun, Rvir = {truth["rvir"]:.0f} kpc; panels +/- {half:.0f} kpc; '
              f'z = 0 truth: BCG {truth["bcg"]:.3e}, ICS_disrupt {truth["ics_in"]:.3e}, '
              f'ICS_accrete + satellite ICS {truth["ics_pre"]:.3e}, satellites {truth["sat"]:.3e} Msun')
        print(f'    {"z":>6s} {"row":>26s} {"mass [Msun]":>12s} {"of z=0":>7s} '
              f'{"in gals":>8s} {"as ICS":>7s} {"N gals":>7s} {"outside":>8s}')

        fig, axes = plt.subplots(len(SPATIAL_ROWS), len(snaps), squeeze=False,
                                 figsize=(3.3 * len(snaps) + 0.6, 3.3 * len(SPATIAL_ROWS)))
        for j, s in enumerate(snaps):
            t = T[s]
            for i, (key, label, col) in enumerate(SPATIAL_ROWS):
                ax = axes[i, j]
                sel = t['row'] == key
                gal = sel & (t['kind'] == 'gal')
                ics = sel & (t['kind'] == 'ics')
                area = lambda m: np.clip(SPATIAL_AREA * m / SPATIAL_AREA_MASS, 0.3, None)
                o = np.argsort(-t['m'][gal])
                ax.scatter(t['x'][gal][o], t['y'][gal][o], s=area(t['m'][gal][o]), color=col,
                           alpha=0.75, lw=0, zorder=Z_OBS, rasterized=True)
                ax.scatter(t['x'][ics], t['y'][ics], s=area(t['m'][ics]), facecolors='none',
                           edgecolors=col, marker='D', lw=1.2, zorder=Z_OBS + 1)
                for hx, hy, hr in zip(t['halo_x'], t['halo_y'], t['halo_r']):
                    ax.add_patch(plt.Circle((hx, hy), hr, fill=False, lw=0.5, color='0.35',
                                            zorder=Z_BAND))
                ax.add_patch(plt.Circle((0, 0), t['main_r'], fill=False, lw=1.0, color='k',
                                        zorder=Z_BAND + 1))
                ax.plot(0, 0, marker='+', color='k', ms=8, mew=1.2, zorder=Z_LINE)
                ax.set_xlim(-half, half)
                ax.set_ylim(-half, half)
                ax.set_aspect('equal')
                mass = t['m'][sel].sum()
                frac = mass / truth[key] if truth[key] > 0 else np.nan
                out = sel & ((np.abs(t['x']) > half) | (np.abs(t['y']) > half))
                ax.text(0.03, 0.97, f'{mass:.2e} M$_\\odot$\n{frac:.2f} of z = 0',
                        transform=ax.transAxes, ha='left', va='top', fontsize=6.5)
                if j == 0:
                    ax.set_ylabel(f'{label}\ny [kpc]')
                else:
                    ax.set_yticklabels([])
                if i == len(SPATIAL_ROWS) - 1:
                    ax.set_xlabel('x [kpc]')
                else:
                    ax.set_xticklabels([])
                if i == 0:
                    ax.set_title(f'z = {t["z"]:.2f}', fontsize=plt.rcParams['axes.labelsize'])
                print(f'    {t["z"]:6.2f} {label:>26s} {mass:12.3e} {frac:7.2f} '
                      f'{t["m"][gal].sum() / max(mass, 1e-30):8.2f} '
                      f'{t["m"][ics].sum() / max(mass, 1e-30):7.2f} {int(gal.sum()):7d} '
                      f'{t["m"][out].sum() / max(mass, 1e-30):8.2f}')
            for nm in ('out', 'lost'):
                sel = t['row'] == nm
                print(f'    {t["z"]:6.2f} {"(" + nm + ")":>26s} {t["m"][sel].sum():12.3e}')
        # z = 0 validation: the traced rows against SAGE's own totals
        t = T[Z0_SNAP]
        bad = []
        for key, label, col in SPATIAL_ROWS:
            got = t['m'][t['row'] == key].sum()
            if truth[key] > 0 and abs(got / truth[key] - 1) > 1e-3:
                bad.append(f'{label} {got:.3e} vs {truth[key]:.3e}')
        print('    z = 0 check against SAGE (BCG, ICS_disrupt, ICS_accrete + satellite ICS, '
              'satellites): ' + ('exact' if not bad else 'MISMATCH ' + '; '.join(bad)))

        handles = [plt.Line2D([], [], marker='o', ls='none', color='0.4', ms=6,
                              label='stars in galaxies'),
                   plt.Line2D([], [], marker='D', ls='none', mfc='none', mec='0.4', ms=6,
                              mew=1.2, label='ICS (drawn at the galaxy holding it)'),
                   plt.Line2D([], [], color='k', lw=1.0, label=r'main progenitor $R_{\rm vir}$'),
                   plt.Line2D([], [], color='0.35', lw=0.5,
                              label=rf'other haloes $> 10^{{{np.log10(SPATIAL_HALO_MIN):.1f}}}\,{{\rm M}}_\odot$')]
        fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=7.5,
                   bbox_to_anchor=(0.5, 0.0))
        fig.suptitle(rf'{alpha_label(alpha)}: $M_{{\rm vir}} = 10^{{{np.log10(truth["mvir"]):.2f}}}'
                     rf'\,{{\rm M}}_\odot$ at z = {REDSHIFTS[Z0_SNAP]:.2f}',
                     fontsize=plt.rcParams['axes.labelsize'], y=0.995)
        fig.subplots_adjust(left=0.12, right=0.98, bottom=0.07, top=0.95, wspace=0.05, hspace=0.05)
        save_figure(fig, f'22_spatial_beta{alpha:g}', dpi=SCATTER_RASTER_DPI)


def report_pearce(runs):
    """The numbers figures 18-20 show, beside Pearce et al.'s."""
    from scipy.stats import spearmanr
    try:
        m_lo = PEARCE_M200C_LO * _m200c_to_mvir_factor()
    except Exception:
        m_lo = np.nan
    print()
    print('=' * 78)
    print('THE PEARCE ET AL. (2026) COMPARISON: which infall events build the BCG')
    print('=' * 78)
    print(f'  clusters Mvir >= {PEARCE_HOST_LO:.1e} Msun at Snap_{Z0_SNAP}; tracking from '
          f'z = {PEARCE_Z_START:g}; no central swap within {PEARCE_ALIGN_GYR:g} Gyr')
    conv = _mvir_per_m200c()
    desc = dict(all=f'every cluster above {PEARCE_HOST_LO:.1e} without a central swap',
                range=f'their mass range, Mvir >= {PEARCE_M200C_LO * conv:.2e} '
                      f'(M200c >= {PEARCE_M200C_LO:.2e})',
                top=f'the {PEARCE_N_MATCH} most massive without a central swap')[PEARCE_SAMPLE]
    print(f'  stacked sample (PEARCE_SAMPLE = {PEARCE_SAMPLE!r}): {desc}')
    print(f'  theirs: {PEARCE_N_MATCH} clusters, M200c {PEARCE_M200C_RANGE[0]:.2e}-{PEARCE_M200C_RANGE[1]:.2e} '
          f'(Mvir ~ {PEARCE_M200C_RANGE[0] * conv:.2e}-{PEARCE_M200C_RANGE[1] * conv:.2e})')
    print(f'  Pearce+26: N50 median 3 (16-84th 3-5, range 2-13); top four '
          f'{", ".join(f"{x * 100:.1f}" for x in PEARCE_RANKED[:4])} per cent '
          f'(sum {sum(PEARCE_RANKED[:4]) * 100:.1f}); in situ {PEARCE_INSITU * 100:.0f}, '
          f'remainder {PEARCE_REMAINDER * 100:.0f}')
    for alpha, hdr, colour, P in runs:
        st = P['stack']
        n = int(st.sum())
        print(f'\n  beta_clock = {alpha:g}: {P["mhost"].size} clusters, {int(P["swap"].sum())} with a '
              f'central swap, {int((~P["started"].astype(bool)).sum())} whose central forms after '
              f'z = {PEARCE_Z_START:g}; {int(P["eligible"].sum())} eligible, {n} stacked')
        if n:
            ms = P['mhost'][st]
            print(f'    stacked Mvir {ms.min():.2e}-{ms.max():.2e} Msun (M200c ~ {ms.min() / conv:.2e}-'
                  f'{ms.max() / conv:.2e}), median {np.median(ms):.2e}')
        if PEARCE_SAMPLE == 'top' and n < PEARCE_N_MATCH:
            print(f'    note: only {n} eligible clusters, fewer than their {PEARCE_N_MATCH}')
        if np.isfinite(m_lo):
            print(f'    in their mass range (Mvir >= {m_lo:.2e}, M200c >= {PEARCE_M200C_LO:.1e}): '
                  f'{int((st & (P["mhost"] >= m_lo)).sum())}')
        if n == 0:
            continue
        n50 = P['n50'][st]
        ok = np.isfinite(n50)
        if ok.any():
            q = np.percentile(n50[ok], (16, 50, 84))
            print(f'    N50 of the {int(ok.sum())} clusters whose events reach half the BCG: '
                  f'median {q[1]:.0f} (16-84th {q[0]:.0f}-{q[2]:.0f}, range '
                  f'{n50[ok].min():.0f}-{n50[ok].max():.0f}); <= 3, 4, 5 in '
                  + ', '.join(f'{(n50 <= x).sum() / n * 100:.0f}' for x in (3, 4, 5))
                  + ' per cent')
        print(f'    {int((~ok).sum())} clusters never reach 50 per cent from events (in situ dominates)')
        mr = P['ranked'][st].mean(axis=0)
        print('    mean ranked shares (per cent): '
              + ' '.join(f'{x * 100:.1f}' for x in mr)
              + f';  top four {mr[:4].sum() * 100:.1f}')
        print(f'    in situ {P["f_insitu"][st].mean() * 100:.1f} per cent (median '
              f'{np.median(P["f_insitu"][st]) * 100:.1f}); remainder beyond the top '
              f'{PEARCE_N_RANK} {P["f_remainder"][st].mean() * 100:.1f}')
        merged = P['m_insitu_merged'][st].sum() / P['mbcg'][st].sum()
        print(f'    of which galaxies already in the cluster at z = {PEARCE_Z_START:g}: '
              f'{merged * 100:.1f} per cent of the summed BCG mass')

        e = st[P['ev_cl']]
        give = e & (P['ev_m_bcg'] > 0)
        print(f'    {int(e.sum()):,} infall events, {int(give.sum()):,} '
              f'({give.sum() / max(e.sum(), 1) * 100:.1f} per cent) give the BCG anything')
        mb = P['ev_m_bcg'][e]
        mu, lb = P['ev_mu'][e], P['ev_lookback'][e]
        if mb.sum() > 0:
            print(f'    accreted BCG mass from mu < 10: {mb[mu < 10].sum() / mb.sum() * 100:.1f} '
                  f'per cent; from infall > 4 Gyr ago: {mb[lb > 4].sum() / mb.sum() * 100:.1f}; '
                  f'both: {mb[(mu < 10) & (lb > 4)].sum() / mb.sum() * 100:.1f}')
        with np.errstate(invalid='ignore'):
            over = P['credit'][:, 1:] - np.diff(P['mbcg_hist'], axis=1)
        print(f'    check: steps crediting more than the BCG grew (by > 1e6 Msun): '
              f'{int((over > 1e6).sum())}; min in situ share {P["f_insitu"][st].min():.3f}')
        if ok.sum() >= KIMMIG_MIN:
            r1 = spearmanr(n50[ok], P['mhost'][st][ok])
            r2 = spearmanr(P['ranked'][st][:, :4].sum(axis=1), P['mhost'][st])
            print(f'    Spearman with Mvir: N50 r_s = {r1.statistic:.3f} (p = {r1.pvalue:.3f}); '
                  f'F4 r_s = {r2.statistic:.3f} (p = {r2.pvalue:.3f})   '
                  f'[theirs: 0.125 (0.199), -0.133 (0.171)]')

        # by halo mass: fixed bins (the last is their sample's range) and equal thirds
        mh = P['mhost']
        bins = [('1e14-3e14', (mh >= 1e14) & (mh < 3e14)),
                ('3e14-their lower', (mh >= 3e14) & (mh < (m_lo if np.isfinite(m_lo) else np.inf))),
                ('their range', mh >= m_lo if np.isfinite(m_lo) else np.zeros_like(st))]
        if n >= 3:
            edges = np.percentile(mh[st], (0, 100 / 3, 200 / 3, 100))
            for i, nm in enumerate(('lowest third', 'middle third', 'highest third')):
                hi_ok = mh <= edges[i + 1] if i == 2 else mh < edges[i + 1]
                bins.append((nm, (mh >= edges[i]) & hi_ok))
        print(f'    {"by Mvir":>17s} {"N":>4s} {"top 4":>6s} {"in situ":>8s} {"N50 ok":>7s} '
              f'{"N50 med":>8s} {"16-84th":>8s}')
        for nm, b in bins:
            S = _pearce_stats(P, st & b)
            if S['n'] == 0:
                print(f'    {nm:>17s} {0:4d}')
                continue
            print(f'    {nm:>17s} {S["n"]:4d} {S["top4"]:6.3f} {S["insitu"]:8.3f} {S["reach"]:7.2f} '
                  f'{S["n50_med"]:8.1f} {S["n50_lo"]:4.0f}-{S["n50_hi"]:<3.0f}')

        # the events against the merger-time curves of figures 18 and 19: below
        # a curve (smaller mu) an event has had that long to merge
        snaps = np.unique(P['ev_snap'][e])
        lbs = AGE_AT_SNAP[Z0_SNAP] - AGE_AT_SNAP[snaps]
        good = lbs > 0.05
        curves = {}
        for name, shape in (('BK08', _bk08_shape),
                            ('clock', lambda m, a=alpha: _clock_shape(m, a))):
            c = np.full(snaps.size, np.nan)
            c[good] = _just_merged_mu(shape, lbs[good])
            curves[name] = dict(zip(snaps, c))
        lmu = np.log10(P['ev_mu'][e])
        sn = P['ev_snap'][e]
        mm = P['ev_m_in'][e]
        for name in ('BK08', 'clock'):
            lim = np.array([curves[name].get(x, np.nan) for x in sn])
            had = np.isfinite(lim) & (lmu < lim)
            gave = mb > 0
            print(f'    vs {name:5s}: {had.sum():,} events had time to merge; '
                  f'{(had & gave).sum():,} of them gave the BCG mass, {(had & ~gave).sum():,} '
                  f'gave nothing ({mm[had & ~gave].sum():.2e} Msun brought in); '
                  f'{mb[had].sum() / max(mb.sum(), 1e-30) * 100:.1f} per cent of the merged BCG mass '
                  f'is from events that had time')

    # the curves themselves
    print('\n  merger-time curves of figures 18 and 19: log10 mu that just merges by now, '
          'for infall this long ago (nan: not even mu = 0.1 merges that fast)')
    lb_grid = np.array([1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
    lb_grid = lb_grid[lb_grid < AGE_AT_SNAP[Z0_SNAP] - AGE_AT_SNAP[0]]
    print(f'    {"lookback [Gyr]":>28s}' + ''.join(f'{x:7.0f}' for x in lb_grid))
    rows = [(f'BK08 eta = {PEARCE_ETA:g}', _bk08_shape)]
    rows += [(f'BK08 eta = {eta:g}', lambda m, e_=eta: _bk08_shape(m, e_)) for eta in PEARCE_ETA_BAND]
    rows += [(f'SAGE26 clock beta = {a:g}', lambda m, a_=a: _clock_shape(m, a_)) for a, *_ in runs]
    for name, shape in rows:
        print(f'    {name:>28s}' + ''.join(f'{v:7.2f}' for v in _just_merged_mu(shape, lb_grid)))


def _pearce_stats(P, sel):
    """Mean top-four and in situ shares, the fraction of clusters whose events
    reach half the BCG, and N50 among those, for the clusters in *sel*."""
    n = int(sel.sum())
    out = dict(n=n, top4=np.nan, insitu=np.nan, reach=np.nan,
               n50_med=np.nan, n50_lo=np.nan, n50_hi=np.nan)
    if n == 0:
        return out
    n50 = P['n50'][sel]
    ok = np.isfinite(n50)
    out.update(top4=float(P['ranked'][sel][:, :4].sum(axis=1).mean()),
               insitu=float(P['f_insitu'][sel].mean()), reach=float(ok.mean()))
    if ok.any():
        out['n50_lo'], out['n50_med'], out['n50_hi'] = np.percentile(n50[ok], (16, 50, 84))
    return out


# ========================== THE POSITIONAL RULE TEST ==========================
#
# beta_clock decides a destroyed satellite's fate by WHEN it is destroyed,
# against a clock.  Pearce et al. decide it by WHERE: whether the core reached
# the centre.  This test re-sorts the stars SAGE26 has already destroyed by a
# rule of the second kind and asks what the ICS and the BCG would then be:
#
#     merge onto its target  if  r / R_vir < f,   else disrupt to the ICS
#
# where r is the satellite's distance from the galaxy it is destroyed onto, and
# R_vir that galaxy's halo radius, both at the satellite's last output.  An
# orphan never outlives the step it is made in, so that last output is always
# the last snapshot its subhalo was found: where the subhalo was lost, to
# within one snapshot.
#
# The stellar mass is re-propagated through the merger tree under each rule,
# snapshot by snapshot, with three things held fixed:
#   * when and onto which galaxy every satellite is destroyed (the merger tree
#     decides this; beta_clock does not);
#   * every galaxy's own stellar growth, i.e. its change in mass between
#     snapshots minus what merged into it (its star formation, including the
#     bursts mergers trigger);
#   * the ICS a destroyed galaxy carries follows it onto its target.
# So the rule changes where destroyed stars end up, and therefore how massive
# the galaxies that later merge or disrupt are, but not the knock-on physics: a
# rule that merges more would also feed more cold gas to bursts and black
# holes, and that needs a SAGE run.  Rule 0 replays SAGE's own decisions; it
# reproduces every galaxy's stellar mass exactly and the ICS to within the
# stars a satellite forms between its last output and its destruction, which
# report_positional prints as the validation.
#
# The results are drawn against the share of destroyed stellar mass each rule
# sends to the ICS, the axis on which the beta_clock runs also sit, so the two
# kinds of rule are compared at equal routing: the difference is which
# satellites merge, not how many.

POS_F = (0.05, 0.1, 0.2, 0.3, 0.5, 1.0)     # merge inside f R_vir of the target
POS_CACHE_VERSION = 'r3'
POS_COLS = ('GalaxyIndex', 'CentralGalaxyIndex', 'Type', 'StellarMass',
            'IntraClusterStars', 'Mvir', 'mergeType', 'mergeIntoID',
            'Posx', 'Posy', 'Posz', 'Rvir')
# observed medians the paper quotes (report_trend): clusters, groups, 16-84th
POS_OBS = dict(cluster=0.13, group=0.15, band=(0.05, 0.29))
POS_RULE_CMAP = plt.get_cmap('Greens')


def _reclass_file(args):
    """Re-propagate one file's stellar mass under SAGE's own rule and each POS_F.

    Returns the z = 0 FOF totals per rule, the routing totals, every
    destruction's r / R_vir, and, for the trees of z = 0 clusters, each
    destroyed galaxy's fate and mass per rule plus the BCG masses (the
    overrides the Pearce scan replays).
    """
    fp, conv, box = args
    fr = np.asarray(POS_F)
    nr = 1 + fr.size
    with h5.File(fp, 'r') as f:
        g = f[f'Snap_{Z0_SNAP}']
        if g['Type'].shape[0] == 0:
            return None
        cl = (g['Type'][:] == 0) & (g['Mvir'][:] * conv >= PEARCE_HOST_LO)
        fmul = _forest_mulfac(f)
        cl_gal = np.sort(g['GalaxyIndex'][:][cl].astype(np.int64))
        cl_trees = np.unique(cl_gal // fmul)
        n_dup = 0

        routed = np.zeros(nr)                         # destroyed stellar mass, all of it
        to_ics = np.zeros(nr)                         # ... and the part sent to the ICS
        xs, own_m, mass0 = [], [], []                 # every destruction: r/Rvir, fate, mass
        rec_gal, rec_merged, rec_mass = [], [], []    # destructions in cluster trees
        prev = Mn = In = None
        prev_snap = -1
        for s in range(Z0_SNAP + 1):
            key = f'Snap_{s}'
            if key not in f or f[key]['Type'].shape[0] == 0:
                continue
            d = {c: f[key][c][:] for c in POS_COLS}
            gal = d['GalaxyIndex'].astype(np.int64)
            fkey = gal // fmul
            fk, fst, fen, dup = _forest_blocks(fkey)
            n_dup += dup
            m_obs = d['StellarMass'].astype(np.float64) * conv
            i_obs = d['IntraClusterStars'].astype(np.float64) * conv
            pos = np.stack([d['Posx'], d['Posy'], d['Posz']], axis=1).astype(np.float64)
            n = gal.size
            if prev is None or prev_snap != s - 1:
                Mn = np.repeat(m_obs[None], nr, axis=0)
                In = np.repeat(i_obs[None], nr, axis=0)
            else:
                rp = _p_find(prev, gal)
                has = rp >= 0
                # the destroyed galaxies of the previous snapshot and their targets here
                D = np.flatnonzero(np.isin(prev['mergeType'], (1, 2, 4)))
                b = np.clip(np.searchsorted(fk, prev['tree'][D]), 0, max(fk.size - 1, 0))
                ok = fk[b] == prev['tree'][D] if fk.size else np.zeros(D.size, bool)
                tgt = fst[b] + prev['mergeIntoID'][D].astype(np.int64)
                ok &= (tgt >= fst[b]) & (tgt < fen[b])     # inside the forest's own block
                D, tgt = D[ok], tgt[ok]
                own = np.isin(prev['mergeType'][D], (1, 2))
                # where it was, against its target, at its last output
                tp = rp[tgt]
                known = tp >= 0
                x = np.full(D.size, np.nan)
                dx = prev['pos'][D[known]] - prev['pos'][tp[known]]
                dx -= box * np.round(dx / box)
                with np.errstate(invalid='ignore', divide='ignore'):
                    x[known] = np.linalg.norm(dx, axis=1) / prev['Rvir'][tp[known]]
                known &= np.isfinite(x)
                merged = np.empty((nr, D.size), bool)
                merged[0] = own
                merged[1:] = np.where(known[None], x[None] < fr[:, None], own[None])
                m_dest = prev['Mn'][:, D]
                i_dest = prev['In'][:, D]

                # each galaxy's own growth, from what SAGE did
                m_in_obs = np.zeros(n)
                np.add.at(m_in_obs, tgt[own], prev['m_obs'][D[own]])
                Mn = np.repeat(m_obs[None], nr, axis=0)
                In = np.repeat(i_obs[None], nr, axis=0)
                grow = m_obs[has] - prev['m_obs'][rp[has]] - m_in_obs[has]
                Mn[:, has] = prev['Mn'][:, rp[has]] + grow[None]
                In[:, has] = prev['In'][:, rp[has]]
                # and the destroyed stars, wherever each rule sends them
                for r in range(nr):
                    mg = merged[r]
                    np.add.at(Mn[r], tgt[mg], m_dest[r, mg])
                    np.add.at(In[r], tgt, i_dest[r] + np.where(mg, 0.0, m_dest[r]))
                    routed[r] += m_dest[r].sum()
                    to_ics[r] += m_dest[r, ~mg].sum()
                xs.append(x)
                own_m.append(own)
                mass0.append(m_dest[0])
                inc = np.isin(prev['tree'][D], cl_trees)
                rec_gal.append(prev['gal'][D[inc]])
                rec_merged.append(merged[:, inc])
                rec_mass.append(m_dest[:, inc])
            prev = dict(gal=gal, m_obs=m_obs, Mn=Mn, In=In, pos=pos,
                        Rvir=d['Rvir'].astype(np.float64), tree=fkey,
                        mergeType=d['mergeType'], mergeIntoID=d['mergeIntoID'])
            prev['_o'] = np.argsort(gal, kind='stable')
            prev['_k'] = gal[prev['_o']]
            prev_snap = s

        # z = 0 FOF totals under every rule, and as SAGE wrote them
        cgi = d['CentralGalaxyIndex'].astype(np.int64)
        _, idx = np.unique(cgi, return_inverse=True)
        nh = idx.max() + 1
        cen = d['Type'] == 0
        mvir = np.full(nh, np.nan)
        mvir[idx[cen]] = d['Mvir'][cen] * conv
        stars = np.array([np.bincount(idx, weights=Mn[r], minlength=nh) for r in range(nr)])
        ics = np.array([np.bincount(idx, weights=In[r], minlength=nh) for r in range(nr)])
        ics_obs = np.bincount(idx, weights=i_obs, minlength=nh)
        r0 = _p_find(prev, cl_gal)
        bcg = Mn[:, r0]

    cat = lambda L, ax=0: np.concatenate(L, axis=ax) if L else np.array([])
    return dict(mvir=mvir, stars=stars, ics=ics, ics_obs=ics_obs,
                routed=routed, to_ics=to_ics, n_dup=n_dup,
                x=cat(xs), own=cat(own_m), mass0=cat(mass0),
                rec_gal=cat(rec_gal), rec_merged=cat(rec_merged, 1), rec_mass=cat(rec_mass, 1),
                cl_gal=cl_gal, bcg=bcg)


def scan_positional(hdr, verbose=True):
    """Every rule's z = 0 budget for one run, and the Pearce test under each."""
    files, conv, box = hdr['files'], hdr['mass_convert'], hdr['box_size']
    jobs = min(FILE_JOBS, len(files))
    t_start = time.time()
    work = [(fp, conv, box) for fp in files]
    if verbose:
        print(f'  {len(files)} files, {jobs} worker(s); each holds two snapshots of one file '
              f'with {1 + len(POS_F)} rules -- lower ICS_FILE_JOBS if memory runs short')
    pool = ProcessPoolExecutor(max_workers=jobs) if jobs > 1 else None
    it = pool.map(_reclass_file, work) if pool else (_reclass_file(w) for w in work)
    parts = []
    for i, part in enumerate(it):
        parts.append(part)
        if verbose:
            n = 0 if part is None else part['x'].size
            dup = 0 if part is None else part['n_dup']
            print(f'    re-propagated file {i + 1}/{len(work)}: {n:,} destructions'
                  + (f', WARNING {dup} forest blocks split' if dup else '')
                  + f' [{time.time() - t_start:7.0f} s]', flush=True)
    if pool is not None:
        pool.shutdown()
    good = [p for p in parts if p is not None]
    R = dict(
        mvir=np.concatenate([p['mvir'] for p in good]),
        stars=np.concatenate([p['stars'] for p in good], axis=1),
        ics=np.concatenate([p['ics'] for p in good], axis=1),
        ics_obs=np.concatenate([p['ics_obs'] for p in good]),
        routed=sum(p['routed'] for p in good), to_ics=sum(p['to_ics'] for p in good),
        x=np.concatenate([p['x'] for p in good]),
        own=np.concatenate([p['own'] for p in good]).astype(bool),
        mass0=np.concatenate([p['mass0'] for p in good]),
    )
    # the Pearce scan, every rule in one pass over the files
    nr = 1 + len(POS_F)
    ovs = []
    for p in parts:
        if p is None or p['cl_gal'].size == 0:
            ovs.append(None)
            continue
        o = np.argsort(p['rec_gal'])
        ovs.append(dict(gal=p['rec_gal'][o], merged=p['rec_merged'][:, o],
                        mass=p['rec_mass'][:, o], bcg_gal=p['cl_gal'], bcg_mass=p['bcg']))
    if verbose:
        print(f'    replaying the Pearce test under {nr} rules in one pass', flush=True)
    for r, P in enumerate(scan_pearce(hdr, verbose=verbose, overrides=ovs, prefix='      ')):
        for k, v in P.items():
            R[f'p{r}_{k}'] = v
    if verbose:
        print(f'    Pearce test under {nr} rules [{time.time() - t_start:7.0f} s]')
    return R


def load_positional(hdr, refresh=False, verbose=True):
    """The positional-rule test for one run, from the npz cache when current."""
    tag = os.path.basename(os.path.normpath(hdr['directory']))
    cache = os.path.join(CACHE_DIR, f'positional_{POS_CACHE_VERSION}_{tag}_'
                         f'{"-".join(f"{x:g}" for x in POS_F)}.npz')
    newest = max(os.path.getmtime(f) for f in hdr['files'])
    if not refresh and os.path.exists(cache) and os.path.getmtime(cache) > newest:
        if verbose:
            print(f'  positional rules from cache: {cache}')
        z = np.load(cache)
        R = {k: z[k] for k in z.files}
    else:
        if verbose:
            print(f'  re-sorting destroyed satellites in {hdr["directory"]}')
        R = scan_positional(hdr, verbose=verbose)
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez_compressed(cache, **R)
    nr = 1 + len(POS_F)
    R['pearce'] = [derive_pearce({k[len(f'p{r}_'):]: v for k, v in R.items()
                                  if k.startswith(f'p{r}_')}) for r in range(nr)]
    return R


def all_positional(refresh=False):
    """(alpha, header, colour, rules) for every run in the sweep."""
    out = []
    for alpha, hdr, colour in ALL_RUNS:
        print(f'alpha = {alpha:g}')
        out.append((alpha, hdr, colour, load_positional(hdr, refresh=refresh)))
    return out


def _pos_fics(R, r, lo, hi=np.inf):
    """Median observed-frame f_ICS (the paper's additive Brough correction) of
    haloes lo <= Mvir < hi under rule r."""
    sel = (R['mvir'] >= lo) & (R['mvir'] < hi) & (R['stars'][r] + R['ics'][r] > 0)
    if sel.sum() < MIN_COUNT:
        return np.nan
    f = observed_fics(R['ics'][r][sel], R['stars'][r][sel], MODEL_SB_RECOVERY['centre'])
    return float(np.median(f))


def _pos_summary(R, r):
    """Every number figure 21 draws for rule r."""
    P = R['pearce'][r]
    st = P['stack']
    return dict(
        routed=R['to_ics'][r] / R['routed'][r],
        cluster=_pos_fics(R, r, CLUSTER_LO),
        group=_pos_fics(R, r, GROUP_LO, GROUP_HI),
        insitu=float(P['f_insitu'][st].mean()) if st.any() else np.nan,
        top4=float(P['ranked'][st][:, :4].sum(axis=1).mean()) if st.any() else np.nan,
        reach=float(np.isfinite(P['n50'][st]).mean()) if st.any() else np.nan,
    )


# ============ FIGURE 21: WHERE, NOT WHEN ============

def plot_21_positional(runs):
    """Left: observed-frame f_ICS of groups and clusters.  Right: the BCG's in
    situ share and its four largest events.  Both against the share of
    destroyed stellar mass sent to the ICS, for the positional rules (applied to
    the published run, green, labelled by f) and the beta_clock runs."""
    fig, (ax_f, ax_b) = panel_grid(1, 2)
    base = next(R for a, h, c, R in runs if a == FIDUCIAL) if any(
        a == FIDUCIAL for a, *_ in runs) else runs[0][3]
    S = [_pos_summary(base, r) for r in range(1, 1 + len(POS_F))]
    x = np.array([s['routed'] for s in S])
    cols = POS_RULE_CMAP(np.linspace(0.45, 0.95, len(POS_F)))
    o = np.argsort(x)

    def rule_line(ax, key, ls, label):
        y = np.array([s[key] for s in S])
        ax.plot(x[o], y[o], color='#238b45', ls=ls, lw=1.6, zorder=Z_LINE, label=label)
        ax.scatter(x, y, c=cols, s=28, edgecolors='white', linewidths=0.6, zorder=Z_LINE + 1)

    def beta_points(ax, key, marker):
        ms = 13 if marker == '*' else 8
        for alpha, hdr, colour, R in runs:
            s0 = _pos_summary(R, 0)
            ax.plot(s0['routed'], s0[key], marker=marker, ms=ms, color=colour, ls='none',
                    mec='white', mew=0.8, zorder=Z_LINE + 3)

    def shape_key(pairs):
        return [plt.Line2D([], [], marker=m, ms=13 if m == '*' else 7, color='0.4', ls='none',
                           mec='white', label=l) for m, l in pairs]

    rule_line(ax_f, 'cluster', '-', r'$r < f R_{\rm vir}$: clusters')
    rule_line(ax_f, 'group', '--', r'$r < f R_{\rm vir}$: groups')
    beta_points(ax_f, 'cluster', '*')
    beta_points(ax_f, 'group', 'h')
    lo, hi = POS_OBS['band']
    ax_f.axhspan(lo, hi, color='0.88', zorder=Z_BAND, lw=0, label='observed 16-84th')
    ax_f.axhline(POS_OBS['cluster'], color='0.45', lw=1.0, zorder=Z_BAND + 1,
                 label=f'observed clusters ({POS_OBS["cluster"]:g})')
    ax_f.axhline(POS_OBS['group'], color='0.45', lw=1.0, ls='--', zorder=Z_BAND + 1,
                 label=f'observed groups ({POS_OBS["group"]:g})')
    ax_f.set_ylabel(r'median $f_{\rm ICS}$, observed frame')

    rule_line(ax_b, 'insitu', '-', r'$r < f R_{\rm vir}$: BCG in situ')
    rule_line(ax_b, 'top4', '--', r'$r < f R_{\rm vir}$: four largest events')
    beta_points(ax_b, 'insitu', 'o')
    beta_points(ax_b, 'top4', 's')
    obs = dict(color='0.45', lw=1.0, zorder=Z_BAND + 1)
    ax_b.axhline(PEARCE_INSITU, ls='-', label=f'Pearce+26 in situ ({PEARCE_INSITU:g})', **obs)
    ax_b.axhline(sum(PEARCE_RANKED[:4]), ls='--',
                 label=f'Pearce+26 four largest ({sum(PEARCE_RANKED[:4]):.3f})', **obs)
    ax_b.set_ylabel('mean fraction of the BCG')

    for ax in (ax_f, ax_b):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1 if ax is ax_b else 0.7)
        ax.set_xlabel('fraction of destroyed stellar mass sent to the ICS')
        for s, c, f in zip(S, cols, POS_F):
            ax.annotate(f'{f:g}', (s['routed'], s['cluster' if ax is ax_f else 'insitu']),
                        textcoords='offset points', xytext=(4, 4), fontsize=6.5, color='#00441b')
    # the beta runs once in the legend, by colour
    handles = [plt.Line2D([], [], marker='o', ms=7, color=c, ls='none', mec='white',
                          label=alpha_label(a)) for a, h, c, R in runs]
    legend(ax_f, loc='upper left', handles=ax_f.get_legend_handles_labels()[0] + handles
           + shape_key((('*', r'$\beta_{\rm clock}$ run: clusters'),
                        ('h', r'$\beta_{\rm clock}$ run: groups'))))
    legend(ax_b, loc='upper right', handles=ax_b.get_legend_handles_labels()[0]
           + shape_key((('o', r'$\beta_{\rm clock}$ run: in situ'),
                        ('s', r'$\beta_{\rm clock}$ run: four largest'))))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.12, top=0.95, wspace=0.25)
    save_figure(fig, '21_positional_rule' + _sample_tag())


def report_positional(runs):
    """The numbers figure 21 draws, the validation, and where satellites die."""
    print()
    print('=' * 78)
    print('THE POSITIONAL RULE TEST: merge if r < f R_vir of the target at the last output')
    print('=' * 78)
    print('  observed: f_ICS clusters 0.13, groups 0.15 (16-84th 0.05-0.29); '
          f'Pearce+26 BCG in situ {PEARCE_INSITU:g}, four largest {sum(PEARCE_RANKED[:4]):.3f}')
    for alpha, hdr, colour, R in runs:
        print(f'\n  base run beta_clock = {alpha:g}')
        # validation: rule 0 must give back SAGE's own ICS
        big = R['mvir'] >= GROUP_LO
        d = R['ics'][0][big].sum() / R['ics_obs'][big].sum() - 1
        print(f'    validation, rule 0 vs SAGE: ICS in haloes > 1e13 differs by {d * 100:+.2f} per cent')
        x, own, m = R['x'], R['own'], R['mass0']
        ok = np.isfinite(x)
        print(f'    where satellites die: {ok.mean() * 100:.1f} per cent of destructions have a target position;'
              f' r/Rvir median {np.median(x[ok]):.2f}, mass-weighted '
              f'{np.average(x[ok], weights=np.maximum(m[ok], 1e-30)):.2f}')
        for lab, sel in (('merged by SAGE', ok & own), ('disrupted by SAGE', ok & ~own)):
            if sel.any():
                q = np.percentile(x[sel], (16, 50, 84))
                print(f'      {lab:18s}: r/Rvir 16-50-84th {q[0]:.2f} {q[1]:.2f} {q[2]:.2f} '
                      f'({sel.sum():,} events, {m[sel].sum():.3e} Msun)')
        # the massive satellites, the ones the clock decides (section 3.3)
        for lo in (1e10, 1e11):
            big_m = ok & (m > lo)
            for lab, sel in (('merged by SAGE', big_m & own), ('disrupted by SAGE', big_m & ~own)):
                if sel.any():
                    print(f'      M* > {lo:.0e}, {lab:18s}: {sel.sum():6,} events, r/Rvir median '
                          f'{np.median(x[sel]):.2f}, inside 0.5 Rvir {(x[sel] < 0.5).mean() * 100:.0f} '
                          f'per cent, inside 0.2 Rvir {(x[sel] < 0.2).mean() * 100:.0f} per cent')

        try:
            m_lo = PEARCE_M200C_LO * _m200c_to_mvir_factor()
        except Exception:
            m_lo = np.nan
        print(f'    {"rule":>10s} {"to ICS":>7s} {"f_ICS cl":>9s} {"f_ICS gr":>9s} {"cl-gr":>7s} '
              f'{"in situ":>8s} {"top 4":>6s} {"N50 ok":>7s}   their mass range: '
              f'{"N":>3s} {"in situ":>8s} {"top 4":>6s}')
        for r in range(1 + len(POS_F)):
            s = _pos_summary(R, r)
            P = R['pearce'][r]
            S = _pearce_stats(P, P['stack'] & (P['mhost'] >= m_lo)) if np.isfinite(m_lo) else \
                dict(n=0, insitu=np.nan, top4=np.nan)
            name = 'SAGE' if r == 0 else f'f = {POS_F[r - 1]:g}'
            print(f'    {name:>10s} {s["routed"]:7.2f} {s["cluster"]:9.3f} {s["group"]:9.3f} '
                  f'{s["cluster"] - s["group"]:+7.3f} {s["insitu"]:8.2f} {s["top4"]:6.2f} '
                  f'{s["reach"]:7.2f}   {"":>17s}{S["n"]:3d} {S["insitu"]:8.2f} {S["top4"]:6.2f}')

    # how much routing each kind of rule needs to meet the observations, and
    # whether the two kinds differ at equal routing
    base = next((R for a, h, c, R in runs if a == FIDUCIAL), runs[0][3])
    S = [_pos_summary(base, r) for r in range(1, 1 + len(POS_F))]
    xr = np.array([s['routed'] for s in S])
    o = np.argsort(xr)
    B = [(a, _pos_summary(R, 0)) for a, h, c, R in runs]
    xb = np.array([b['routed'] for _, b in B])
    ob = np.argsort(xb)

    def crossing(x, y, target):
        """Routing at which y first reaches *target*, by linear interpolation."""
        for i in range(len(x) - 1):
            if (y[i] - target) * (y[i + 1] - target) <= 0 and y[i] != y[i + 1]:
                return x[i] + (target - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i])
        return np.nan

    print(f'\n  routing that meets each target (fraction of destroyed stellar mass sent to the ICS)')
    for key, target, lab in (('cluster', POS_OBS['cluster'], 'f_ICS clusters = 0.13'),
                             ('group', POS_OBS['group'], 'f_ICS groups = 0.15'),
                             ('top4', sum(PEARCE_RANKED[:4]), 'BCG top four = 0.572'),
                             ('insitu', PEARCE_INSITU, 'BCG in situ = 0.09')):
        yr = np.array([s[key] for s in S])
        yb = np.array([b[key] for _, b in B])
        print(f'    {lab:24s}: positional rules {crossing(xr[o], yr[o], target):5.2f};  '
              f'beta_clock runs {crossing(xb[ob], yb[ob], target):5.2f}   (nan: never reached)')
    print(f'\n  at equal routing: each beta_clock run beside the positional rules '
          f'(on the beta = {FIDUCIAL:g} run) interpolated to its routing')
    print(f'    {"run":>12s} {"to ICS":>7s}   {"f_ICS cl":>22s} {"f_ICS gr":>22s} '
          f'{"in situ":>22s} {"top 4":>22s}')
    for a, b in B:
        cells = []
        for key in ('cluster', 'group', 'insitu', 'top4'):
            yr = np.array([s[key] for s in S])
            yi = np.interp(b['routed'], xr[o], yr[o])
            cells.append(f'{b[key]:5.3f} vs {yi:5.3f} {b[key] - yi:+6.3f}'.rjust(22))
        print(f'    {"beta=" + format(a, "g"):>12s} {b["routed"]:7.2f}   ' + ' '.join(cells))
    print('    (clock vs positional, then the difference)')


# ========================== REPORTS ==========================

def report_runs(scans):
    """The z = 0 budget each figure is a view of."""
    print()
    box = REF['box_size']
    print(f'Sweep over MergerTimeFactor, {box:g} Mpc/h box, z = {REDSHIFTS[Z0_SNAP]:.3f} '
          f'(Snap_{Z0_SNAP}, not Snap_{LAST_SNAP} -- see Z0_SNAP)')
    print(f'{"alpha":>6}  {"N_halo":>8}  {"f_ICS(all)":>10}  {"f_ICS(group)":>12}  '
          f'{"f_ICS(cluster)":>14}  {"median B/T":>10}  {"rho_SFR(z=0)":>12}')
    for alpha, hdr, colour, s in scans:
        mvir, stars, ics = s['h_mvir'], s['h_mstar'], s['h_ics']
        tot = stars + ics
        ok = np.isfinite(mvir) & (tot > 0)
        f_all = ics[ok].sum() / tot[ok].sum()
        gr = ok & (mvir >= GROUP_LO) & (mvir < GROUP_HI)
        cl = ok & (mvir >= CLUSTER_LO)
        f_gr = np.median(ics[gr] / tot[gr]) if gr.sum() else np.nan
        f_cl = np.median(ics[cl] / tot[cl]) if cl.sum() else np.nan
        m, b = s['g_mstar'], s['g_bulge']
        sel = m > 1e9
        bt = np.median(np.clip(b[sel] / m[sel], 0, 1)) if sel.sum() else np.nan
        print(f'{alpha:>6g}  {ok.sum():>8,d}  {f_all:>10.3f}  {f_gr:>12.3f}  '
              f'{f_cl:>14.3f}  {bt:>10.3f}  {s["sfrd"][Z0_SNAP]:>12.4e}')

    if not APPLY_SB_CORRECTION:
        return
    print()
    print('The same clusters and groups put through the surface-brightness cut')
    print(f'{"alpha":>6}  {"f_ICS cluster (SB-free -> through the cut)":>44}  '
          f'{"f_ICS group":>28}')
    for alpha, hdr, colour, s in scans:
        mvir, stars, ics = s['h_mvir'], s['h_mstar'], s['h_ics']
        ok = np.isfinite(mvir) & (stars + ics > 0)
        row = []
        for lo_m, hi_m in ((CLUSTER_LO, np.inf), (GROUP_LO, GROUP_HI)):
            sel = ok & (mvir >= lo_m) & (mvir < hi_m)
            raw = np.median(ics[sel] / (ics[sel] + stars[sel])) if sel.sum() else np.nan
            seen = [np.median(observed_fics(ics[sel], stars[sel], g))
                    if sel.sum() else np.nan for g in MODEL_SB_RECOVERY['edges']]
            row.append(f'{raw:.3f} -> {min(seen):.3f}--{max(seen):.3f}')
        print(f'{alpha:>6g}  {row[0]:>44}  {row[1]:>28}')


def report_clock(runs):
    """The clock, the lifetime, and the routing, in numbers.

    The t_life columns are the ones to read carefully.  The median over all
    destroyed satellites barely moves with alpha -- the disruption gate is a
    merger-tree event, not a clock event -- but the medians of the two
    destination channels move a great deal, because alpha decides which
    satellites fall into which channel.
    """
    print()
    print('The merger clock and where the satellites went')
    def _wmedian(x, w):
        if x.size == 0:
            return np.nan
        o = np.argsort(x)
        c = np.cumsum(w[o]) / w[o].sum()
        return float(x[o][np.searchsorted(c, 0.5)])

    print(f'{"alpha":>6}  {"N_events":>9}  {"med T_df":>9}  {"t_life ICS":>10}  '
          f'{"t_life BCG":>10}  {"(mw) ICS":>9}  {"(mw) BCG":>9}  '
          f'{"M->ICS":>8}  {"N->ICS":>8}')
    for alpha, hdr, colour, ev in runs:
        ok = ev['clock_known'] & (ev['t_life'] > 0)
        life, m = ev['t_life'], ev['mstar']
        ics, bcg = ok & ev['to_ics'], ok & ev['to_bcg']
        print(f'{alpha:>6g}  {len(ev["snap"]):>9,d}  '
              f'{np.nanmedian(ev["t_df"][ok]):>9.2f}  '
              f'{np.median(life[ics]):>10.3f}  {np.median(life[bcg]):>10.3f}  '
              f'{_wmedian(life[ics], m[ics]):>9.3f}  '
              f'{_wmedian(life[bcg], m[bcg]):>9.3f}  '
              f'{m[ev["to_ics"]].sum() / m.sum():>8.3f}  '
              f'{ev["to_ics"].mean():>8.3f}')
    print('  T_df and t_life in Gyr; t_life is infall to destruction, split by')
    print('  destination, plain median then weighted by the stellar mass carried.')


# ---------------- the diagnostics the paper is written from -----------------
#
# Everything below prints; nothing draws.  The sections follow the paper:
# where the stars are (results), how the ICS was assembled and when (discussion),
# and the clock itself (method).  Every table has one column per alpha so the
# lever can be read straight across.

def _snap_near(z):
    """The output snapshot nearest redshift *z*, never past the analysis snapshot."""
    return int(np.argmin(np.abs(REDSHIFTS[:Z0_SNAP + 1] - z)))


def _f(v, fmt='.3f', width=10):
    """A number for a table, '--' when it is not finite."""
    return f'{format(v, fmt):>{width}}' if v is not None and np.isfinite(v) else f'{"--":>{width}}'


def _pct_str(x, fmt='.3f'):
    """'median [p16, p84]' of the finite values of *x*."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return '--'
    p = np.percentile(x, (16, 50, 84))
    return f'{p[1]:{fmt}} [{p[0]:{fmt}}, {p[2]:{fmt}}]'


def _alpha_head(runs, width=12, first=''):
    return f'{first:<26}' + ''.join(f'{"a=" + format(a, "g"):>{width}}' for a, *_ in runs)


def _lookback_to_z(t_gyr):
    """Redshift at lookback time *t_gyr*, interpolated on the snapshot grid."""
    lb = LOOKBACK_AT_SNAP[::-1]
    return np.interp(t_gyr, lb, REDSHIFTS[::-1])


def report_header():
    """The runs this invocation sees, so a log always says what it was made from."""
    print('=' * 78)
    print('SAGE26 merger clock / ICS analysis')
    print('=' * 78)
    print(f'  cosmology: h = {HUBBLE_H:.4f}, Omega_m = {OMEGA_M:.4f}, '
          f'Omega_L = {OMEGA_L:.4f}')
    print(f'  analysis snapshot Snap_{Z0_SNAP} (z = {REDSHIFTS[Z0_SNAP]:.4f}); last '
          f'snapshot Snap_{LAST_SNAP} skipped (consistent-trees FOF collapse)')
    print(f'  {len(SNAPS)} output snapshots, age of the universe {AGE_NOW:.2f} Gyr')
    print(f'  group {GROUP_LO:.0e}-{GROUP_HI:.0e} Msun, cluster > {CLUSTER_LO:.0e} Msun')
    for alpha, hdr, colour in ALL_RUNS:
        print(f'  alpha = {alpha:<5g} {hdr["directory"]:<40s} {len(hdr["files"])} file(s), '
              f'V = {hdr["volume"]:.3e} Mpc^3 (box {hdr["box_size"]:g} Mpc/h, '
              f'{hdr["volume_fraction"] * 100:.1f}% processed), '
              f'ThresholdSatDisruption = {hdr["thresh_sat_disruption"]:g}')
    print()


def report_budget(scans):
    """Where the stars are at z = 0 and how that changes with alpha.

    Sections:
      A  the global budget: galaxies, ICS, and their sum (alpha should not move it)
      B  groups and clusters: f_ICS, the BCG, ICS/BCG
      C  the same in finer bins of halo mass, with the BCG shift against alpha = FIDUCIAL
      D  how the ICS was assembled: in situ vs carried in, and when it was deposited
      E  redshift evolution of the global and halo ICS fractions and BCG masses
      F  galaxies: the massive end of the SMF and bulge-to-total
      G  star formation rate density
    """
    ref = next((s for a, _, _, s in scans if a == FIDUCIAL), scans[0][3])
    print()
    print('-' * 78)
    print('A. Global stellar budget at the analysis snapshot  [Msun / Mpc^3]')
    print(_alpha_head(scans))
    rows = (('rho_* (galaxies)', lambda s: s['rho_star'][Z0_SNAP], '.4g'),
            ('rho_ICS', lambda s: s['rho_ics'][Z0_SNAP], '.4g'),
            ('rho_* + rho_ICS', lambda s: s['rho_star'][Z0_SNAP] + s['rho_ics'][Z0_SNAP], '.4g'),
            ('ICS share of all stars', lambda s: s['rho_ics'][Z0_SNAP]
             / (s['rho_star'][Z0_SNAP] + s['rho_ics'][Z0_SNAP]), '.4f'),
            ('total vs fiducial', lambda s: (s['rho_star'][Z0_SNAP] + s['rho_ics'][Z0_SNAP])
             / (ref['rho_star'][Z0_SNAP] + ref['rho_ics'][Z0_SNAP]), '.4f'))
    for name, fn, fmt in rows:
        print(f'{name:<26}' + ''.join(_f(fn(s), fmt, 12) for *_, s in scans))
    print('  alpha only routes existing stars, so any change in the total is indirect: a')
    print('  merger brings the satellite\'s cold gas into a burst and grows the black hole')
    print('  (quasar winds, then radio-mode heating), a disruption sends that gas to the hot halo')

    print()
    print('B. Groups and clusters at the analysis snapshot  (median [16th, 84th])')
    for cname, lo, hi in (('groups', GROUP_LO, GROUP_HI), ('clusters', CLUSTER_LO, np.inf)):
        print(f'  {cname} ({lo:.0e} <= Mvir < {hi:.0e} Msun)')
        for alpha, hdr, colour, s in scans:
            mv, st, ic, bcg = s['h_mvir'], s['h_mstar'], s['h_ics'], s['h_bcg']
            sel = np.isfinite(mv) & (mv >= lo) & (mv < hi) & (st + ic > 0) & np.isfinite(bcg)
            with np.errstate(invalid='ignore', divide='ignore'):
                print(f'    a = {alpha:<5g} N = {sel.sum():>7,d}')
                print(f'      f_ICS = ICS/(ICS+all stars)  {_pct_str(ic[sel] / (ic[sel] + st[sel]))}')
                print(f'      log m_BCG                   {_pct_str(np.log10(bcg[sel]), ".2f")}')
                print(f'      log (m_BCG + m_ICS)         {_pct_str(np.log10(bcg[sel] + ic[sel]), ".2f")}')
                print(f'      m_ICS / m_BCG               {_pct_str(ic[sel] / bcg[sel], ".2f")}')
                print(f'      BCG share of all stars      {_pct_str(bcg[sel] / (st[sel] + ic[sel]))}')
                print(f'      satellite share             {_pct_str((st[sel] - bcg[sel]) / (st[sel] + ic[sel]))}')
                print(f'      BCG bulge-to-total          {_pct_str(s["h_bcg_bulge"][sel] / bcg[sel])}')
                print(f'      summed: ICS {ic[sel].sum():.3e}, BCG {bcg[sel].sum():.3e}, '
                      f'satellites {(st[sel] - bcg[sel]).sum():.3e} Msun')

    print()
    print('C. Median f_ICS and log m_BCG in bins of log Mvir; dBCG = shift against '
          f'alpha = {FIDUCIAL:g} [dex]')
    edges = (11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 16.0)
    head = f'{"log Mvir":<12}{"N":>8}' + ''.join(
        f'{"fICS a=" + format(a, "g"):>11}{"BCG a=" + format(a, "g"):>11}{"dBCG":>7}'
        for a, *_ in scans)
    print(head)
    ref_med = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        meds = []
        for alpha, hdr, colour, s in scans:
            mv, st, ic, bcg = s['h_mvir'], s['h_mstar'], s['h_ics'], s['h_bcg']
            with np.errstate(invalid='ignore', divide='ignore'):
                lm = np.log10(mv)
                sel = np.isfinite(lm) & (lm >= lo) & (lm < hi) & (st + ic > 0) & (bcg > 0)
                fi = np.median(ic[sel] / (ic[sel] + st[sel])) if sel.sum() else np.nan
                lb = np.median(np.log10(bcg[sel])) if sel.sum() else np.nan
            meds.append((sel.sum(), fi, lb))
            if alpha == FIDUCIAL:
                ref_med[lo] = lb
        line = f'{lo:>5.1f}-{hi:<5.1f} {meds[0][0]:>8,d}'
        for n_, fi, lb in meds:
            line += _f(fi, '.3f', 11) + _f(lb, '.2f', 11) + _f(lb - ref_med.get(lo, np.nan), '+.2f', 7)
        print(line)

    print()
    print('D. ICS assembly at the analysis snapshot, mass-weighted over each class')
    print('   in situ = disrupted into this halo; ex situ = arrived already made '
          '(pre-processed in groups)')
    print('   deposit = mass-weighted mean lookback time of the ICS deposits, and the redshift')
    for cname, lo, hi in (('groups', GROUP_LO, GROUP_HI), ('clusters', CLUSTER_LO, np.inf),
                          ('all haloes', 0.0, np.inf)):
        print(f'  {cname}')
        for alpha, hdr, colour, s in scans:
            mv, ic = s['h_mvir'], s['h_ics']
            dis, acc, mt = s['h_ics_dis'], s['h_ics_acc'], s['h_ics_mt']
            sel = np.isfinite(mv) & (mv >= lo) & (mv < hi) & (ic > 0)
            tot = (dis[sel] + acc[sel]).sum()
            if tot <= 0:
                print(f'    a = {alpha:<5g} no ICS assembly bookkeeping in this output')
                continue
            unit = float(s['time_convert_gyr'][0]) if 'time_convert_gyr' in s else np.nan
            t_mw = mt[sel].sum() / tot * unit
            with np.errstate(invalid='ignore', divide='ignore'):
                t_each = mt[sel] / (dis[sel] + acc[sel]) * unit
                insitu_each = dis[sel] / (dis[sel] + acc[sel])
            print(f'    a = {alpha:<5g} in situ {dis[sel].sum() / tot:.3f} (per halo '
                  f'{_pct_str(insitu_each)}),  deposit {t_mw:.2f} Gyr ago, '
                  f'z = {_lookback_to_z(t_mw):.2f} (per halo {_pct_str(t_each, ".2f")} Gyr);  '
                  f'bookkeeping closes to {tot / ic[sel].sum():.4f}')

    print()
    print('E. Evolution: global ICS share, median halo f_ICS, median log m_BCG')
    zs = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
    for name, key, fn in (
            ('ICS share of all stars', None,
             lambda s, k: s['rho_ics'][k] / (s['rho_ics'][k] + s['rho_star'][k])),
            ('f_ICS, clusters (median)', None, lambda s, k: s['fics_cl'][1, k]),
            ('f_ICS, groups (median)', None, lambda s, k: s['fics_gr'][1, k]),
            ('log m_BCG, clusters', None, lambda s, k: np.log10(s['bcg_cl'][1, k])),
            ('log m_BCG, groups', None, lambda s, k: np.log10(s['bcg_gr'][1, k])),
            ('N clusters', None, lambda s, k: float(s['n_cl'][k])),
            ('N groups', None, lambda s, k: float(s['n_gr'][k]))):
        print(f'  {name}')
        print('    ' + f'{"z":<8}' + ''.join(f'{z:>9.2f}' for z in zs))
        for alpha, hdr, colour, s in scans:
            with np.errstate(invalid='ignore', divide='ignore'):
                vals = [fn(s, _snap_near(z)) for z in zs]
            fmt = '.0f' if name.startswith('N ') else ('.2f' if 'log' in name else '.3f')
            print('    ' + f'{"a=" + format(alpha, "g"):<8}' + ''.join(_f(v, fmt, 9) for v in vals))

    print()
    print('F. Galaxies at the analysis snapshot')
    print('  number density above a stellar mass [Mpc^-3], and the ratio to alpha = '
          f'{FIDUCIAL:g}')
    cuts = (10.0, 10.5, 11.0, 11.5, 12.0)
    print('    ' + f'{"log m* >":<10}' + ''.join(f'{c:>16.1f}' for c in cuts))
    ref_n = {c: (ref['g_mstar'] > 10 ** c).sum() / float(ref['volume'][0]) for c in cuts}
    for alpha, hdr, colour, s in scans:
        vol = float(s['volume'][0])
        line = '    ' + f'{"a=" + format(alpha, "g"):<10}'
        for c in cuts:
            nd = (s['g_mstar'] > 10 ** c).sum() / vol
            r = nd / ref_n[c] if ref_n[c] > 0 else np.nan
            line += f'{nd:>10.3e} ({r:4.2f})' if np.isfinite(r) else f'{nd:>10.3e} (  --)'
        print(line)
    print('  median B/T [16th, 84th] in bins of log m*, all galaxies / centrals only')
    bins = ((9.0, 10.0), (10.0, 10.5), (10.5, 11.0), (11.0, 11.5), (11.5, 13.0))
    for alpha, hdr, colour, s in scans:
        m, b, t = s['g_mstar'], s['g_bulge'], s['g_type']
        print(f'    a = {alpha:<5g}')
        for lo, hi in bins:
            sel = (m > 10 ** lo) & (m <= 10 ** hi)
            cen = sel & (t == 0)
            bt = np.clip(b / np.where(m > 0, m, np.nan), 0, 1)
            print(f'      {lo:4.1f}-{hi:<4.1f} N = {sel.sum():>8,d}  B/T {_pct_str(bt[sel], ".2f"):<22}'
                  f' centrals {_pct_str(bt[cen], ".2f"):<22} B/T > 0.5: '
                  f'{(bt[sel] > 0.5).mean() if sel.sum() else np.nan:.3f}')

    print()
    print('G. Cosmic star formation rate density [Msun/yr/Mpc^3] and ratio to alpha = '
          f'{FIDUCIAL:g}')
    zs = (0.0, 0.5, 1.0, 2.0, 4.0, 6.0)
    print('    ' + f'{"z":<8}' + ''.join(f'{z:>19.1f}' for z in zs))
    for alpha, hdr, colour, s in scans:
        line = '    ' + f'{"a=" + format(alpha, "g"):<8}'
        for z in zs:
            k = int(np.argmin(np.abs(REDSHIFTS - z)))
            v, r0 = s['sfrd'][k], ref['sfrd'][k]
            line += f'{v:>11.3e} ({v / r0:5.3f})' if r0 > 0 else f'{v:>11.3e} (   --)'
        print(line)

    print()
    print('H. What the extra mergers do: bulge channels, black holes, gas, star formation')
    print('   (InstabilityBulgeMass also receives minor-merger stars when the central is')
    print('   disc-dominated, so "instability" is the morphology-based label, not the process)')
    if 'g_bh' not in ref:
        print('   scan cache predates these columns -- rerun with --refresh')
    else:
        print(f'  {"density [Msun/Mpc^3]":<26}' + ''.join(f'{"a=" + format(a, "g"):>14}' for a, *_ in scans))
        for name, key in (('black holes', 'g_bh'), ('bulges', 'g_bulge'),
                          ('  instability-labelled', 'g_ibulge'), ('  merger-labelled', 'g_mbulge'),
                          ('cold gas', 'g_cold'), ('hot gas + CGM', None)):
            vals = []
            for alpha, hdr, colour, s in scans:
                v = (s['g_hot'] + s['g_cgm']) if key is None else s[key]
                vals.append(v.sum() / float(s['volume'][0]))
            rv = vals[[a for a, *_ in scans].index(FIDUCIAL)] if FIDUCIAL in [a for a, *_ in scans] else vals[-1]
            print(f'  {name:<26}' + ''.join(f'{v:>9.3e}({v / rv:4.2f})' for v in vals))
        print('  per stellar-mass bin: median B/T | merger-labelled share of bulge mass | median log m_BH')
        for lo, hi in ((9.0, 10.0), (10.0, 10.5), (10.5, 11.0), (11.0, 11.5), (11.5, 13.0)):
            line = f'    {lo:4.1f}-{hi:<4.1f}'
            for alpha, hdr, colour, s in scans:
                m = s['g_mstar']
                sel = (m > 10 ** lo) & (m <= 10 ** hi)
                if not sel.any():
                    line += '   --'
                    continue
                bt = np.median(np.clip(s['g_bulge'][sel] / m[sel], 0, 1))
                share = s['g_mbulge'][sel].sum() / max(s['g_bulge'][sel].sum(), 1e-30)
                with np.errstate(divide='ignore'):
                    lbh = np.median(np.log10(s['g_bh'][sel] + 1.0))
                line += f'  a={alpha:g}: {bt:.2f} | {share:.2f} | {lbh:.2f}'
            print(line)
        print('  star formation rate density relative to the published run, and the')
        print('  published run\'s share of the total, by host halo mass and by stellar mass')
        refs = next((s for a, _, _, s in scans if a == FIDUCIAL), scans[-1][3])
        tot_ref = refs['g_sfr'].sum() / float(refs['volume'][0])
        for label, key, edges in (('host Mvir', 'g_cmvir', ((0, 1e12), (1e12, 1e13), (1e13, 1e14), (1e14, 1e20))),
                                  ('stellar mass', 'g_mstar', ((0, 1e9), (1e9, 1e10), (1e10, 10 ** 10.5),
                                                               (10 ** 10.5, 1e11), (1e11, 1e14)))):
            for lo, hi in edges:
                sel_r = (refs[key] >= lo) & (refs[key] < hi)
                d_ref = refs['g_sfr'][sel_r].sum() / float(refs['volume'][0])
                line = f'    {label} {lo:.0e}-{hi:.0e}: share {d_ref / tot_ref:.2f} '
                for alpha, hdr, colour, s in scans:
                    sel = (s[key] >= lo) & (s[key] < hi)
                    d = s['g_sfr'][sel].sum() / float(s['volume'][0])
                    line += f' a={alpha:g}: {d / d_ref if d_ref > 0 else np.nan:.2f}'
                print(line)
    print('-' * 78)


# ---------------- numbers asked for in review ----------------------------

def _boot(stat, x, n=1000, seed=0):
    """(statistic, 16th, 84th percentile of its bootstrap distribution)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    b = [stat(x[rng.integers(0, x.size, x.size)]) for _ in range(n)]
    return float(stat(x)), *np.percentile(b, (16, 84))


def _pearson_boot(x, y, n=1000, seed=0):
    """Pearson r of the finite pairs, its bootstrap 16th-84th range, and N."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3:
        return np.nan, np.nan, np.nan, int(x.size)
    rng = np.random.default_rng(seed)
    b = []
    for _ in range(n):
        i = rng.integers(0, x.size, x.size)
        b.append(np.corrcoef(x[i], y[i])[0, 1])
    return float(np.corrcoef(x, y)[0, 1]), *np.nanpercentile(b, (16, 84)), int(x.size)


def _rheat_ratio(s, k):
    """Heated radius over cooling radius for the galaxies in mask *k*."""
    rcool = s['g_rcool_rvir'][k] * s['g_rvir'][k]
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(rcool > 0, s['g_rheat'][k] / rcool, np.nan)


def report_referee(scans):
    """Halo counts with bootstrap errors on f_ICS, B/T as Moffett+16 measure it,
    and what changes in massive galaxies.  All at the analysis snapshot."""
    ref = next((s for a, _, _, s in scans if a == FIDUCIAL), scans[-1][3])
    delta = MODEL_SB_RECOVERY['centre']
    print()
    print('-' * 78)
    print(f'R1. Haloes per mass bin and median f_ICS with bootstrap 16-84th errors '
          f'(z = {REDSHIFTS[Z0_SNAP]:.3f})')
    print(f'    raw = SAGE26 f_ICS;  obs = f_ICS {delta:+.2f} (Brough+24), floored at 0')
    bins = ((11.5, 12.0), (12.0, 12.5), (12.5, 13.0), (13.0, 13.5), (13.5, 14.0),
            (14.0, 14.5), (14.5, 15.0), (15.0, 16.0), (13.0, 14.0), (14.0, 16.0))
    for lo, hi in bins:
        name = {(13.0, 14.0): 'groups', (14.0, 16.0): 'clusters'}.get((lo, hi), '')
        print(f'  log Mvir {lo:4.1f}-{hi:4.1f} {name}')
        for alpha, hdr, colour, s in scans:
            mv, st, ic = s['h_mvir'], s['h_mstar'], s['h_ics']
            sel = np.isfinite(mv) & (mv >= 10 ** lo) & (mv < 10 ** hi) & (st + ic > 0)
            f = ic[sel] / (ic[sel] + st[sel])
            fo = observed_fics(ic[sel], st[sel], delta)
            m, l, h = _boot(np.median, f)
            mo, lo_, ho = _boot(np.median, fo)
            print(f'    a = {alpha:<4g} N = {sel.sum():>7,d}   raw {_f(m, ".3f", 6)} '
                  f'[{_f(l, ".3f", 5)}, {_f(h, ".3f", 5)}]   obs {_f(mo, ".3f", 6)} '
                  f'[{_f(lo_, ".3f", 5)}, {_f(ho, ".3f", 5)}]')

    print()
    print('R2. B/T as Moffett+16 measure it.  med = median of B/T;  mw = mass-weighted,')
    print('    sum(bulge) / sum(stars) in the bin, the statistic of a bulge mass fraction')
    obs = load_moffett16_bulge_fraction()
    edges = np.arange(9.0, 12.01, 0.25)
    print(f'  {"log m*":>11}' + ''.join(f'{"a=" + format(a, "g") + " med/mw":>18}' for a, *_ in scans)
          + f'{"Moffett":>9}')
    for lo, hi in zip(edges[:-1], edges[1:]):
        line = f'  {lo:5.2f}-{hi:5.2f}'
        for alpha, hdr, colour, s in scans:
            m, b = s['g_mstar'], s['g_bulge']
            sel = (m >= 10 ** lo) & (m < 10 ** hi) & np.isfinite(b)
            if sel.sum() < MIN_COUNT:
                line += f'{"--":>18}'
                continue
            med = np.median(np.clip(b[sel] / m[sel], 0, 1))
            mw = b[sel].sum() / m[sel].sum()
            line += f'{med:>11.2f} /{mw:>5.2f}'
        if obs is not None:
            c = 0.5 * (lo + hi)
            ok = (obs['logm'] >= lo) & (obs['logm'] < hi)
            line += f'{np.mean(obs["bt"][ok]):>9.2f}' if ok.any() else f'{"--":>9}'
        print(line)
    print('  share of galaxies that are pure spheroids (B/T >= 0.99) and disc-dominated (B/T < 0.5)')
    for cut in (11.0, 11.5):
        line = f'    m* > 10^{cut:g}:'
        for alpha, hdr, colour, s in scans:
            m, b = s['g_mstar'], s['g_bulge']
            sel = m > 10 ** cut
            bt = np.clip(b[sel] / m[sel], 0, 1)
            line += (f'  a={alpha:g}: N={sel.sum():,d} pure {np.mean(bt >= 0.99):.2f} '
                     f'disc {np.mean(bt < 0.5):.2f}')
        print(line)

    print()
    print('R3. Central galaxies above 10^11 Msun: what the extra mergers do')
    print('    each over the published run in brackets.  Radio-mode feedback strength is')
    print('    r_heat / r_cool and the share of centrals where it has shut cooling off')
    print('    (r_heat >= r_cool); "Heating" is only the cooling still being offset, so it')
    print('    FALLS as feedback wins.  Net cooling = CoolingRate (after suppression);')
    print('    potential cooling = mdot_cool (before).  Quasar-mode BH accretion is over the')
    print('    last snapshot interval; merger-labelled bulge mass is cumulative (merger-')
    print('    built stars incl. burst stars: starburst mass alone is not an output)')
    if 'g_heat' not in ref:
        print('    scan cache predates these columns -- rerun with --refresh')
    else:
        def cen(s):
            return (s['g_type'] == 0) & (s['g_mstar'] > 1e11)
        rows = (('N per (100 Mpc)^3', lambda s, k: k.sum() / float(s['volume'][0]) * 1e6, 'sum'),
                ('median log m_BH', lambda s, k: np.median(np.log10(s['g_bh'][k] + 1.0)), 'dex'),
                ('cooling shut off', lambda s, k: np.mean(_rheat_ratio(s, k) >= 0.99), 'frac'),
                ('median r_heat/r_cool', lambda s, k: np.nanmedian(_rheat_ratio(s, k)), 'frac'),
                ('share still heating', lambda s, k: np.mean(s['g_heat'][k] > 0), 'frac'),
                ('mean net cooling', lambda s, k: np.mean(s['g_coolrate'][k]), 'ratio'),
                ('median potential cool.', lambda s, k: np.median(s['g_mdotcool'][k]), 'ratio'),
                ('sum quasar BH acc.', lambda s, k: s['g_qbh'][k].sum() / float(s['volume'][0]), 'ratio'),
                ('sum merger bulge', lambda s, k: s['g_mbulge'][k].sum() / float(s['volume'][0]), 'ratio'),
                ('median cold gas', lambda s, k: np.median(s['g_cold'][k]), 'ratio'),
                ('median hot + CGM', lambda s, k: np.median(s['g_hot'][k] + s['g_cgm'][k]), 'ratio'),
                ('sum SFR', lambda s, k: s['g_sfr'][k].sum() / float(s['volume'][0]), 'ratio'))
        for name, fn, kind in rows:
            r0 = fn(ref, cen(ref))
            line = f'  {name:<22}'
            for alpha, hdr, colour, s in scans:
                v = fn(s, cen(s))
                if kind == 'dex':
                    line += f'  {v:8.3f} ({v - r0:+.2f} dex)'
                elif kind == 'frac':
                    line += f'  {v:8.3f}           '
                else:
                    line += f'  {v:10.3e} ({v / r0 if r0 else np.nan:5.2f})'
            print(line)
    print('-' * 78)


def report_bh_bulge(scans):
    """Median log m_BH in bins of bulge mass for each run, beside the median of
    the Sahu+19 and Davis+19 galaxies in the same bins."""
    print()
    print('-' * 78)
    print('R5. Black hole mass against bulge mass: median log m_BH per bin of log m_bulge')
    obs = _load_bh_bulge_obs()
    lsph = np.concatenate([o[2] for o in obs]) if obs else np.array([])
    lbh = np.concatenate([o[3] for o in obs]) if obs else np.array([])
    print(f'  {"log m_bulge":>12}' + ''.join(f'{"a=" + format(a, "g"):>16}' for a, *_ in scans)
          + f'{"observed (N)":>16}')
    for lo in (9.5, 10.0, 10.5, 11.0, 11.5, 12.0):
        hi = lo + 0.5
        line = f'  {lo:5.1f}-{hi:<5.1f} '
        ref = None
        vals = []
        for alpha, hdr, colour, s in scans:
            b, bh = s['g_bulge'], s['g_bh']
            sel = (b >= 10 ** lo) & (b < 10 ** hi) & (bh > 0)
            v = np.median(np.log10(bh[sel])) if sel.sum() >= MIN_COUNT else np.nan
            vals.append(v)
            if alpha == FIDUCIAL:
                ref = v
        for v in vals:
            line += f'{_f(v, ".2f", 8)} ({v - ref:+.2f})' if np.isfinite(v) and ref is not None \
                else f'{"--":>16}'
        o = (lsph >= lo) & (lsph < hi)
        line += f'{_f(np.median(lbh[o]) if o.any() else np.nan, ".2f", 10)} ({o.sum():>2d})'
        print(line)
    print('  bracket = offset from the published run in dex')
    print('-' * 78)


def report_referee_clock(runs):
    """Survival time is the same in every run, and where destroyed satellites'
    cold gas goes: to the central (burst and black hole) or to the hot halo."""
    print()
    print('-' * 78)
    print('R4. Survival time (infall to destruction) and the cold gas of destroyed satellites')
    print('    t_life = time from the infall snapshot to the snapshot the satellite is')
    print('    destroyed, which is almost always the snapshot its sub-halo is lost')
    for alpha, hdr, colour, ev in runs:
        ok = ev['clock_known'] & (ev['t_life'] > 0)
        life, m = ev['t_life'], ev['mstar']
        big = ok & (m > 1e10) & (m <= 1e11)
        mc = ev['mcold']
        tot = mc.sum()
        print(f'  a = {alpha:<4g} median t_life all {np.nanmedian(life[ok]):.2f} Gyr, '
              f'm* 10^10-10^11 {np.nanmedian(life[big]):.2f} Gyr | cold gas of destroyed '
              f'satellites: {mc.sum():.3e} Msun, to central (burst/BH) '
              f'{mc[ev["to_bcg"]].sum() / tot if tot > 0 else np.nan:.3f}, to hot halo '
              f'{mc[ev["to_ics"]].sum() / tot if tot > 0 else np.nan:.3f}')
        # satellites with no clock (sub-halo < 10 particles at infall, or a central
        # that became an orphan directly) always merge; the outliving fractions are
        # over satellites WITH a clock, the mass-to-ICS fractions over all
        clk = ev['clock_known']
        with np.errstate(invalid='ignore'):
            outl = (ev['t_life'][clk] > ev['t_df'][clk]).mean()
        print(f'           clockless: {np.mean(~clk):.3f} of satellites, {m[~clk].sum() / m.sum():.4f} of '
              f'destroyed stellar mass, to ICS {m[~clk & ev["to_ics"]].sum() / max(m[~clk].sum(), 1e-30):.3f} | '
              f'with a clock: outlive it {outl:.3f}, mass to ICS '
              f'{m[clk & ev["to_ics"]].sum() / max(m[clk].sum(), 1e-30):.3f} (all satellites '
              f'{m[ev["to_ics"]].sum() / m.sum():.3f})')
    print('-' * 78)


def report_clock_detail(runs):
    """The clock in the detail the method section needs.

    Event counts and how the clock was set, the clock and the lifetime as
    distributions, and the routing broken down by satellite mass and by the
    redshift of destruction -- which satellites alpha actually moves.
    """
    print()
    print('-' * 78)
    print('The clock in detail (from the disruption log)')
    for alpha, hdr, colour, ev in runs:
        n = len(ev['snap'])
        ok = ev['clock_known']
        vol = hdr['volume']
        m = ev['mstar']
        print(f'  a = {alpha:g}')
        print(f'    events {n:,}: clock from estimate_merging_time {ok.sum():,} '
              f'({ok.mean() * 100:.1f}%), failure branch / no clock {(~ok).sum():,}')
        for t_ in (1, 2):
            sel = ev['sat_type'] == t_
            print(f'    Type {t_} at destruction: {sel.sum():>9,d} ({sel.mean() * 100:5.1f}% of '
                  f'events, {m[sel].sum() / m.sum() * 100:5.1f}% of mass), to ICS by mass '
                  f'{m[sel & ev["to_ics"]].sum() / max(m[sel].sum(), 1e-30):.3f}')
        print(f'    T_df  [Gyr]  {_pct_str(ev["t_df"][ok], ".2f")}')
        print(f'    t_life [Gyr] {_pct_str(ev["t_life"][ok], ".2f")}  (infall to destruction)')
        with np.errstate(invalid='ignore'):
            outl = (ev['t_life'][ok] > ev['t_df'][ok]).mean()
        print(f'    satellites outliving their clock: {outl:.3f} by number')
        # survival over the unscaled friction time: the beta at which the clock
        # equals a satellite's own lifetime (discussion: what should beta be?)
        with np.errstate(invalid='ignore', divide='ignore'):
            ratio = ev['t_life'] / (ev['t_df'] / alpha)
        rk = ok & np.isfinite(ratio)
        o = np.argsort(ratio[rk])
        cw = np.cumsum(m[rk][o]) / m[rk].sum()
        print(f'    t_survive / t_friction: median {np.median(ratio[rk]):.2f}, '
              f'mass-weighted median {ratio[rk][o][np.searchsorted(cw, 0.5)]:.2f}')
        print(f'    destroyed stellar mass {m.sum() / vol:.3e} Msun/Mpc^3: to ICS '
              f'{m[ev["to_ics"]].sum() / vol:.3e}, to BCG {m[ev["to_bcg"]].sum() / vol:.3e}')

        print(f'    by satellite stellar mass at destruction')
        print(f'      {"log m*":<11}{"N":>9}{"mass share":>12}{"N->ICS":>9}{"M->ICS":>9}'
              f'{"T_df med":>10}{"t_life med":>11}{"t_s/t_f med":>12}')
        for lo, hi in ((0, 8), (8, 9), (9, 10), (10, 11), (11, 13)):
            with np.errstate(divide='ignore'):
                lm = np.log10(m)
            sel = (lm >= lo) & (lm < hi)
            if sel.sum() == 0:
                continue
            k = sel & ok
            print(f'      {lo:>4.0f}-{hi:<5.0f}{sel.sum():>9,d}{m[sel].sum() / m.sum():>12.3f}'
                  f'{ev["to_ics"][sel].mean():>9.3f}'
                  f'{m[sel & ev["to_ics"]].sum() / max(m[sel].sum(), 1e-30):>9.3f}'
                  f'{np.nanmedian(ev["t_df"][k]) if k.any() else np.nan:>10.2f}'
                  f'{np.nanmedian(ev["t_life"][k]) if k.any() else np.nan:>11.2f}'
                  f'{np.nanmedian(ratio[k & rk]) if (k & rk).any() else np.nan:>12.2f}')

        print(f'    by redshift of destruction')
        print(f'      {"z":<11}{"N":>9}{"mass share":>12}{"N->ICS":>9}{"M->ICS":>9}'
              f'{"T_df med":>10}{"t_life med":>11}')
        for lo, hi in ((0, 0.5), (0.5, 1), (1, 2), (2, 4), (4, 20)):
            sel = (ev['z_dest'] >= lo) & (ev['z_dest'] < hi)
            if sel.sum() == 0:
                continue
            k = sel & ok
            print(f'      {lo:>4.1f}-{hi:<5.1f}{sel.sum():>9,d}{m[sel].sum() / m.sum():>12.3f}'
                  f'{ev["to_ics"][sel].mean():>9.3f}'
                  f'{m[sel & ev["to_ics"]].sum() / max(m[sel].sum(), 1e-30):>9.3f}'
                  f'{np.nanmedian(ev["t_df"][k]) if k.any() else np.nan:>10.2f}'
                  f'{np.nanmedian(ev["t_life"][k]) if k.any() else np.nan:>11.2f}')
    print('-' * 78)


# ========================== DRIVER ==========================

FIGURES = {
    1: ('scan', plot_1_fics),
    2: ('scan+disrupt', plot_2_galaxies_and_clock),
    4: ('scan', plot_4_ics_mass_function),
    6: ('scan', plot_6_fics_scatter),
    7: ('kimmig', plot_7_assembly),
    8: ('kimmig', plot_8_clock),
    9: ('kimmig', plot_9_proxies),
    10: ('kimmig', plot_10_matrix),
    11: ('kimmig', plot_11_budget),
    12: ('kimmig', plot_12_shredding),
    13: ('assembly', plot_13_ics_assembly),
    14: ('scan', plot_14_stellar_mass_function),
    15: ('scan+kimmig', plot_15_appendix),
    17: ('scan+kimmig', plot_17_appendix_column),
    16: ('scan', plot_16_ics_mass_function_low_z),
    18: ('pearce', plot_18_infall_events),
    19: ('pearce', plot_19_infall_events_stacked),
    20: ('pearce', plot_20_bcg_oligarchy),
    21: ('positional', plot_21_positional),
    22: ('pearce', plot_22_spatial),
}


def main(argv):
    refresh = '--refresh' in argv
    global PEARCE_SAMPLE
    for a in argv:
        if a.startswith('--pearce='):
            PEARCE_SAMPLE = a.split('=', 1)[1]
    wanted = sorted(int(a) for a in argv if a.isdigit()) or sorted(FIGURES)
    bad = [n for n in wanted if n not in FIGURES]
    if bad:
        sys.exit(f'No such figure: {bad}.  Choose from {sorted(FIGURES)}.')

    setup_style()
    os.makedirs(OUT_DIR, exist_ok=True)
    report_header()

    needs_scan = any('scan' in FIGURES[n][0].split('+') for n in wanted)
    needs_disrupt = any('disrupt' in FIGURES[n][0].split('+') for n in wanted)
    needs_kimmig = any('kimmig' in FIGURES[n][0].split('+') for n in wanted)
    needs_assembly = any(FIGURES[n][0] == 'assembly' for n in wanted)
    needs_pearce = any(FIGURES[n][0] == 'pearce' for n in wanted)
    needs_positional = any(FIGURES[n][0] == 'positional' for n in wanted)

    scans = disrupt = kimmig = assembly = pearce = positional = None
    if needs_scan:
        print('Loading snapshot scans')
        scans = all_scans(refresh=refresh)
        report_runs(scans)
        report_budget(scans)
        report_referee(scans)
        report_bh_bulge(scans)
        report_ics_mass_function(scans)
        print()
        report_sb_correction()
        report_trend(scans)
    if needs_disrupt:
        print()
        print('Loading disruption logs')
        disrupt = all_disrupt()
        if disrupt:
            report_clock(disrupt)
            report_clock_detail(disrupt)
            report_referee_clock(disrupt)
    if needs_kimmig:
        print()
        print('Loading cluster main-branch histories')
        kimmig = all_kimmig(refresh=refresh)
        report_kimmig(kimmig)
    if needs_assembly:
        print()
        print('Loading ICS assembly histories')
        assembly = all_assembly(refresh=refresh)
        if assembly:
            report_assembly(assembly)
    if needs_pearce:
        print()
        print('Loading infall events onto cluster main branches')
        pearce = all_pearce(refresh=refresh)
        report_pearce(pearce)
    if needs_positional:
        print()
        print('Re-sorting destroyed satellites by where they die')
        positional = all_positional(refresh=refresh)
        report_positional(positional)

    print()
    for n in wanted:
        kind, fn = FIGURES[n]
        if kind == 'scan+disrupt':
            # the combined figure draws what it has; only the scans are required
            data = (scans, disrupt) if scans else None
        elif kind == 'scan+kimmig':
            data = (scans, kimmig) if scans else None
        else:
            data = {'scan': scans, 'disrupt': disrupt, 'kimmig': kimmig,
                    'assembly': assembly, 'pearce': pearce,
                    'positional': positional}[kind]
        if not data:
            print(f'Figure {n}: no {kind} data -- skipped')
            continue
        fn(data)

    print()
    print(f'Done.  Figures in {OUT_DIR}')


if __name__ == '__main__':
    main(sys.argv[1:])

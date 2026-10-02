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
    2  bulge-to-total ratio against stellar mass
    3  cosmic star formation rate density
    4  the ICS mass function in a grid of redshifts, split by host halo mass
       (after random_plotting_scripts/ICS_paper_plots.py), alpha as line style
    5  the clock itself -- T_df handed out, how long satellites actually live,
       and the routing split that follows from comparing the two
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
    python plotting/ICS_plots.py --refresh       # rebuild the cached scans
"""

import os
import sys
import glob

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
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
# only MergerTimeFactor differs.  The colour ramp is ColorBrewer YlGnBu, so the
# ordering in alpha is legible without the legend and survives greyscale.
# alpha = 0.25 was dropped from the sweep: it sits below the observed f_ICS in
# both groups and clusters however the surface-brightness cut is set, so it
# rules itself out rather than bounding anything.  The run is still on disk at
# ./output/microuchuu_mtf_0.25/ if it is ever wanted back.
RUNS = (
    (0.5,  './output/microuchuu_mtf_0.5/',  '#41b6c4'),
    (1.0,  './output/microuchuu_mtf_1.0/',  '#2c7fb8'),
    (2.0,  './output/microuchuu_mtf_2.0/',  '#253494'),
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
CACHE_VERSION = 'v5'

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

# Which edge of the g bracket figure 6's scatter is drawn through: 'low' is the
# stronger cut and the one most favourable to the model.
SCATTER_G = 'low'

# Salpeter -> Chabrier, matching plotting/paper_plots.py.  The model is Chabrier.
SALPETER_TO_CHABRIER_DEX = -0.24


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
# the style's native aspect.  Typeface, usetex and tick style are left alone.
PANEL_FONT_SCALE = 0.60
PANEL_W, PANEL_H = 5.6, 4.5


def setup_style():
    plt.style.use("./plotting/kieren_cohare_palatino_sty.mplstyle")
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
    s = rf'$\alpha = {alpha:g}$'
    return s + ' (published)' if alpha == FIDUCIAL else s


def line_style(alpha):
    """Heavier line for the published value, so it reads as the reference."""
    return dict(lw=3.0 if alpha == FIDUCIAL else 2.0,
                ls='-' if alpha == FIDUCIAL else '--',
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
    dict(label=r'Montes \& Trujillo 18', scale='cluster',
         z=(0.301, 0.39, 0.342, 0.537, 0.537, 0.37, 0.043),
         f=(0.0767, 0.0861, 0.1309, 0.066, 0.0578, 0.0483, 0.1085)),
    dict(label='Presotto+14', scale='cluster', z=(0.435, 0.433), f=(0.1226, 0.0554)),
    dict(label='Ragusa+23', scale='cluster', z=(0.05,), f=(0.35,)),
    dict(label='Burke+12', scale='cluster',
         z=(0.947, 0.83, 0.795, 0.808, 1.223),
         f=(0.0142, 0.0259, 0.0377, 0.0153, 0.0236)),
    dict(label=r'Ko \& Jee 18', scale='cluster', z=(1.238,), f=(0.0991,)),
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
# cut admits.  The two are not the same quantity, so a model line and an
# observed point should not be plotted as though they were.
#
# The fix applied here is to put SAGE26's clusters through the same cut.  For a
# cut that admits a fraction g of the halo's intracluster light, the fraction an
# observer would report is
#
#     f_obs = g * m_ICS / (g * m_ICS + m_star,halo)
#
# -- not f_ICS / g, because the ICL sits in the denominator as well, so the
# model has to be re-measured halo by halo rather than rescaled at the end.
#
# g is NOT computed from the model.  SAGE stores the ICS as one scalar per halo
# with no profile and no radius, so any surface brightness computed from it
# would come entirely from an assumed profile -- the systematic being corrected
# for.  g is taken instead from published ICL profiles, which measure it
# directly: the fraction of the total ICL lying above a stated limit.  That
# keeps the assumed-profile problem out of the figure.
#
# What this does NOT do: the stellar mass in the denominator gets no cut, so
# faint satellites an observer would miss are still counted.  That pushes f_obs
# low and is the main reason to treat the band as a bracket, not a prediction.
#
# PROVENANCE.  Every entry records where its value came from:
#   'measured' -- read off a published ICL profile or a stated recovery fraction
#   'assumed'  -- a documented placeholder, NOT a measurement
# report_sb_correction() lists the assumed ones at run time so they cannot
# quietly reach a draft.  Kluge+21 and Montes & Trujillo reach deep enough to
# state g directly; fill those in and flip 'source'.
APPLY_SB_CORRECTION = True

# (low, high) bracket on g.  A bracket, not a value, so the result is a band
# whose width is honest about how little is pinned down.
MODEL_SB_RECOVERY = dict(
    g=(0.55, 0.85),
    source='assumed',
    note='placeholder: a profile keeping 85 per cent of its light above the cut '
         'needs almost no correction, one keeping 55 per cent loses a third of '
         'its ICL.  Both are plausible across the limits these samples use.',
)


def observed_fics(ics, stars, g):
    """f_ICS as an observer admitting a fraction *g* of the ICL would measure it."""
    ics, stars = np.asarray(ics, float), np.asarray(stars, float)
    seen = g * ics
    tot = seen + stars
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(tot > 0, seen / tot, np.nan)


def report_sb_correction():
    """State the cut applied to the model, and flag it if it is a placeholder."""
    if not APPLY_SB_CORRECTION:
        print('Surface-brightness cut on the model: OFF -- f_ICS is the SB-free total')
        return
    lo, hi = MODEL_SB_RECOVERY['g']
    print("Surface-brightness cut applied to SAGE26's clusters:")
    print(f'  f_obs = g*m_ICS / (g*m_ICS + m_star),  g = {lo:.2f}--{hi:.2f}  '
          f'[{MODEL_SB_RECOVERY["source"]}]')
    print(f'  {MODEL_SB_RECOVERY["note"]}')
    if MODEL_SB_RECOVERY['source'] != 'measured':
        print('  WARNING: g is ASSUMED, not measured.  Fill it from the published')
        print('  ICL profiles (Kluge+21, Montes & Trujillo) before the figure is used.')
    print('  Note: no cut is applied to the stellar mass in the denominator, so')
    print('  f_obs is a lower bound -- an observer misses faint satellites too.')


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
        out.append(dict(md, label=r'Madau \& Dickinson 14', marker='o',
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


def scan_run(hdr, verbose=True):
    """Walk every snapshot of one run and collect the per-snapshot summaries."""
    conv, files = hdr['mass_convert'], hdr['files']
    nsnap = len(REDSHIFTS)

    sfrd = np.zeros(nsnap)
    fics_cl = np.full((3, nsnap), np.nan)       # 16/50/84 per cent, clusters
    fics_gr = np.full((3, nsnap), np.nan)       # groups
    # the same, re-measured through each edge of the surface-brightness cut
    n_g = len(MODEL_SB_RECOVERY['g'])
    fics_cl_sb = np.full((n_g, 3, nsnap), np.nan)
    fics_gr_sb = np.full((n_g, 3, nsnap), np.nan)
    n_cl = np.zeros(nsnap, dtype=int)
    n_gr = np.zeros(nsnap, dtype=int)

    z0 = {k: [] for k in ('h_mvir', 'h_mstar', 'h_ics',
                          'g_mstar', 'g_bulge', 'g_type')
          + tuple('h_' + e for e in _HALO_EXTRA)}

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

    for snap in SNAPS:
        key = f'Snap_{snap}'
        sfr_sum = 0.0
        mvir_a, stars_a, ics_a = [], [], []
        extra_a = {k: [] for k in _HALO_EXTRA}
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
                        missing_optional.add(p)
            sfr_sum += float((d['SfrDisk'] + d['SfrBulge']).sum())
            mv, st, ic, ex = _halo_totals(d, conv)
            mvir_a.append(mv)
            stars_a.append(st)
            ics_a.append(ic)
            for k in _HALO_EXTRA:
                extra_a[k].append(ex[k])
            if snap == Z0_SNAP:
                z0['g_mstar'].append(d['StellarMass'] * conv)
                z0['g_bulge'].append(d['BulgeMass'] * conv)
                z0['g_type'].append(d['Type'].astype(np.int8))

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

        for j, gval in enumerate(MODEL_SB_RECOVERY['g']):
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

    if missing_optional and verbose:
        print(f'  note: {sorted(missing_optional)} not in this output -- ICS assembly '
              f'diagnostics will read zero (run with TrackICSAssembly on)')
    out = dict(sfrd=sfrd, fics_cl=fics_cl, fics_gr=fics_gr,
               rho_star=rho_star, rho_ics=rho_ics, bcg_cl=bcg_cl, bcg_gr=bcg_gr,
               mf_counts=mf_counts,
               time_convert_gyr=np.array([hdr['time_convert_gyr']]),
               fics_cl_sb=fics_cl_sb, fics_gr_sb=fics_gr_sb,
               n_cl=n_cl, n_gr=n_gr, volume=np.array([hdr['volume']]),
               sb_g=np.asarray(MODEL_SB_RECOVERY['g'], float))
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

_DISRUPT_COLS = ('snap', 'type', 'mergtime', 'stellarmass', 'dest',
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

def plot_1_fics(scans):
    """f_ICS against halo mass at z = 0, and against redshift for clusters."""
    print('Figure 1: intracluster fraction')
    fig, axes = panel_grid(1, 2)
    ax_m, ax_z = axes

    # --- (a) f_ICS against halo mass at z = 0 ---
    #
    # Each run is drawn twice: the line is the SB-free total SAGE tracks, the
    # band beneath it is the same haloes re-measured through the cut, between
    # the two edges of g.  The gap between line and band is the size of the
    # systematic, and it is not small.
    for alpha, hdr, colour, s in scans:
        mvir, stars, ics = s['h_mvir'], s['h_mstar'], s['h_ics']
        tot = stars + ics
        ok = np.isfinite(mvir) & (mvir > 0) & (tot > 0)
        x = np.log10(mvir[ok])
        st = line_style(alpha)

        if APPLY_SB_CORRECTION:
            edges = []
            for gval in MODEL_SB_RECOVERY['g']:
                c, p = binned_percentiles(
                    x, observed_fics(ics[ok], stars[ok], gval), MVIR_BINS)
                edges.append(p[1])
            good = np.isfinite(edges[0]) & np.isfinite(edges[1])
            ax_m.fill_between(c[good], edges[0][good], edges[1][good],
                              color=colour, alpha=0.30, lw=0, zorder=Z_BAND,
                              label=alpha_label(alpha) + ', through the cut')

        centres, pct = binned_percentiles(x, ics[ok] / tot[ok], MVIR_BINS)
        good = np.isfinite(pct[1])
        ax_m.plot(centres[good], pct[1][good], color=colour,
                  label=alpha_label(alpha), **st)

    # The observed ranges, as published.  The compilation has no halo masses, so
    # it cannot be drawn point by point on a halo-mass axis; groups and clusters
    # are shown at the mass range each sample corresponds to.
    for lo, hi, scale, colour in ((13.0, 14.0, 'group', '#fec44f'),
                                  (14.0, 15.0, 'cluster', '#d95f0e')):
        f = np.concatenate([o['f'] for o in icl_fraction_observations(scale)])
        f = f[f > 0]
        p16, p84 = np.percentile(f, (16, 84))
        ax_m.fill_between([lo, hi], p16, p84, facecolor=colour, alpha=0.30,
                          edgecolor=colour, lw=1.0, zorder=2,
                          label=f'observed {scale}s, 16--84th')

    ax_m.set_xlabel(r'$\log_{10}(M_{\rm vir}\ [{\rm M}_\odot])$')
    ax_m.set_ylabel(r'$f_{\rm ICS} = m_{\rm ICS} / (m_{\rm ICS} + m_{\star,\rm halo})$')
    ax_m.set_xlim(11.5, 15.0)
    ax_m.set_ylim(0.0, 1.0)
    legend(ax_m, loc='upper left', ncol=2)

    # --- (b) cluster f_ICS against redshift ---
    for alpha, hdr, colour, s in scans:
        med = s['fics_cl'][1]
        ok = np.isfinite(med)
        st = line_style(alpha)
        if APPLY_SB_CORRECTION and 'fics_cl_sb' in s:
            sb = s['fics_cl_sb']
            g2 = np.isfinite(sb[0, 1]) & np.isfinite(sb[1, 1])
            ax_z.fill_between(REDSHIFTS[g2], sb[0, 1][g2], sb[1, 1][g2],
                              color=colour, alpha=0.30, lw=0, zorder=Z_BAND,
                              label=alpha_label(alpha) + ', through the cut')
        ax_z.plot(REDSHIFTS[ok], med[ok], color=colour,
                  label=alpha_label(alpha), **st)

    obs = icl_fraction_observations('cluster')

    for i, o in enumerate(obs):
        ax_z.plot(o['z'], o['f'], _OBS_MARKERS[i % len(_OBS_MARKERS)],
                  color='0.35', ms=4.5, mec='white', mew=0.5, ls='none',
                  alpha=0.9, zorder=Z_OBS, label=o['label'])

    ax_z.set_xlabel(r'$z$')
    ax_z.set_ylabel(r'$f_{\rm ICS}$, hosts with $M_{\rm vir} > 10^{14}\,{\rm M}_\odot$')
    ax_z.set_xlim(0.0, 2.2)
    ax_z.set_ylim(0.0, 1.0)
    legend(ax_z, loc='upper right', ncol=2)

    fig.tight_layout()
    save_figure(fig, 'fig1_fICS')


# ============ FIGURE 2: BULGE-TO-TOTAL ============

def plot_2_bulge_to_total(scans):
    """Median B/T against stellar mass, and the bulge-dominated fraction."""
    print('Figure 2: bulge-to-total')
    fig, axes = panel_grid(1, 2)
    ax_bt, ax_fr = axes

    for alpha, hdr, colour, s in scans:
        m, b = s['g_mstar'], s['g_bulge']
        ok = (m > 1e8) & np.isfinite(b)
        x = np.log10(m[ok])
        bt = np.clip(b[ok] / m[ok], 0.0, 1.0)
        st = line_style(alpha)

        centres, pct = binned_percentiles(x, bt, MSTAR_BINS)
        good = np.isfinite(pct[1])
        if alpha == FIDUCIAL:
            ax_bt.fill_between(centres[good], pct[0][good], pct[2][good],
                               color=colour, alpha=0.18, lw=0, zorder=Z_BAND)
        ax_bt.plot(centres[good], pct[1][good], color=colour,
                   label=alpha_label(alpha), **st)

        c2, frac = binned_fraction(x, bt > 0.5, MSTAR_BINS)
        g2 = np.isfinite(frac)
        ax_fr.plot(c2[g2], frac[g2], color=colour, label=alpha_label(alpha), **st)

    obs = load_moffett16_bulge_fraction()
    if obs is not None:
        ax_bt.fill_between(obs['logm'], obs['lo'], obs['hi'], color='0.75',
                           alpha=0.5, lw=0, zorder=1)
        ax_bt.plot(obs['logm'], obs['bt'], 'o', color='0.3', ms=4.5,
                   mec='white', mew=0.5, ls='none', zorder=Z_OBS,
                   label='Moffett+16 (GAMA)')

    ax_bt.set_xlabel(r'$\log_{10}(m_\star\ [{\rm M}_\odot])$')
    ax_bt.set_ylabel(r'$B/T$')
    ax_bt.set_xlim(8.0, 12.2)
    ax_bt.set_ylim(0.0, 1.05)
    legend(ax_bt, loc='upper left')

    ax_fr.set_xlabel(r'$\log_{10}(m_\star\ [{\rm M}_\odot])$')
    ax_fr.set_ylabel(r'fraction with $B/T > 0.5$')
    ax_fr.set_xlim(8.0, 12.2)
    ax_fr.set_ylim(0.0, 1.05)
    legend(ax_fr, loc='upper left')

    fig.tight_layout()
    save_figure(fig, 'fig2_bulge_to_total')


# ============ FIGURE 3: STAR FORMATION RATE DENSITY ============

def plot_3_sfrd(scans):
    """Cosmic SFR density against redshift, and the ratio to the published run."""
    print('Figure 3: star formation rate density')
    fig, axes = panel_grid(1, 2)
    ax, ax_r = axes

    ref = None
    for alpha, hdr, colour, s in scans:
        if alpha == FIDUCIAL:
            ref = s['sfrd']

    for alpha, hdr, colour, s in scans:
        sfrd = s['sfrd']
        ok = sfrd > 0
        st = line_style(alpha)
        ax.plot(REDSHIFTS[ok], np.log10(sfrd[ok]), color=colour,
                label=alpha_label(alpha), **st)
        if ref is not None:
            with np.errstate(invalid='ignore', divide='ignore'):
                r = np.where((ref > 0) & (sfrd > 0), sfrd / ref, np.nan)
            g = np.isfinite(r)
            ax_r.plot(REDSHIFTS[g], r[g], color=colour,
                      label=alpha_label(alpha), **st)

    for o in load_sfrd_observations():
        ax.errorbar(o['z'], o['y'], yerr=[o['el'], o['eu']], fmt=o['marker'],
                    color=o['colour'], ms=3.5, mec='white', mew=0.4, lw=0.8,
                    ls='none', alpha=0.85, zorder=Z_OBS, label=o['label'])

    ax.set_xlabel(r'$z$')
    ax.set_ylabel(r'$\log_{10}(\rho_{\rm SFR}\ [{\rm M}_\odot\,{\rm yr}^{-1}\,{\rm Mpc}^{-3}])$')
    ax.set_xlim(0.0, 8.0)
    ax.set_ylim(-3.0, -0.4)
    legend(ax, loc='lower left', ncol=2)

    ax_r.axhline(1.0, color='0.6', lw=0.8, zorder=1)
    ax_r.set_xlabel(r'$z$')
    ax_r.set_ylabel(rf'$\rho_{{\rm SFR}} / \rho_{{\rm SFR}}(\alpha = {FIDUCIAL:g})$')
    ax_r.set_xlim(0.0, 8.0)
    legend(ax_r, loc='lower right')

    fig.tight_layout()
    save_figure(fig, 'fig3_sfrd')


# ============ FIGURE 4: THE ICS MASS FUNCTION ============

def plot_4_ics_mass_function(scans):
    """The ICS mass function in a grid of redshifts, split by host halo mass.

    The figure of plotting/random_plotting_scripts/ICS_paper_plots.py (plot 2):
    every central holding ICS, one panel per target redshift, coloured by host
    halo mass.  The all-halo total is not drawn (it is still printed by
    report_ics_mass_function).  Colour is spent on halo mass, so alpha
    is carried by line style -- solid for the published value.  No shading:
    with three runs overlaid it hides the lines it sits under.
    """
    print('Figure 4: ICS mass function grid')
    snaps = []
    for zt in MF_TARGET_Z:
        k = _snap_near(zt) if zt > 0 else Z0_SNAP
        if k not in snaps:
            snaps.append(k)
    ncols = 2
    nrows = (len(snaps) + 1) // ncols
    fig, axes = panel_grid(nrows, ncols, sharey=True)
    axes = np.atleast_2d(axes)
    centres = 0.5 * (MF_ICS_EDGES[:-1] + MF_ICS_EDGES[1:])
    colours = MF_HALO_COLOURS          # the halo-mass bins only; no all-halo total

    for pi, snap in enumerate(snaps):
        ax = axes.flat[pi]
        for alpha, hdr, colour, s in scans:
            if 'mf_counts' not in s:
                continue
            vol = float(s['volume'][0])
            for b, c in enumerate(colours):
                phi = s['mf_counts'][snap, b] / vol / MF_ICS_BW
                good = phi > 0
                if not good.any():
                    continue
                ax.plot(centres[good], phi[good], color=c, ls=alpha_ls(alpha),
                        lw=2.0 if alpha == FIDUCIAL else 1.5, alpha=0.9,
                        zorder=Z_LINE + (1 if alpha == FIDUCIAL else 0))
        ax.text(0.95, 0.95, rf'$z = {REDSHIFTS[snap]:.2f}$', transform=ax.transAxes,
                ha='right', va='top')
        ax.set_yscale('log')
        ax.set_xlim(5.5, 12.5)
        ax.set_ylim(1e-6, 1e-2)
        ax.set_xticks([6, 7, 8, 9, 10, 11, 12])
        if pi % ncols == 0:
            ax.set_ylabel(r'$\phi\ [{\rm Mpc}^{-3}\,{\rm dex}^{-1}]$')
        if pi // ncols == nrows - 1:
            ax.set_xlabel(r'$\log_{10}(m_{\rm ICS}\ [{\rm M}_\odot])$')
    for pi in range(len(snaps), axes.size):
        axes.flat[pi].set_visible(False)

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, lw=2.0, label=l)
               for c, l in zip(colours, MF_HALO_LABELS)]
    handles += [Line2D([], [], color='0.3', lw=2.0, ls=alpha_ls(a), label=alpha_label(a))
                for a, *_ in scans]
    # below the grid, as ICS_paper_plots.py has it: inside any panel it sits on data
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.legend(handles, [h.get_label() for h in handles], loc='lower center',
               ncol=4, frameon=False, fontsize=8)
    save_figure(fig, 'fig4_ics_mass_function')


def plot_14_stellar_mass_function(scans):
    """The stellar mass function at the analysis snapshot, one line per alpha."""
    print('Figure 14: stellar mass function')
    fig, ax = panel_grid(1, 1)
    for alpha, hdr, colour, s in scans:
        vol = float(s['volume'][0])
        m = s['g_mstar']
        x, phi = mass_function(np.log10(m[m > 0]), vol, mass_range=(8.0, 12.5))
        ax.plot(x, np.log10(phi), color=colour, label=alpha_label(alpha),
                **line_style(alpha))
    ax.set_xlabel(r'$\log_{10}(m_\star\ [{\rm M}_\odot])$')
    ax.set_ylabel(r'$\log_{10}(\phi\ [{\rm Mpc}^{-3}\,{\rm dex}^{-1}])$')
    ax.set_xlim(8.0, 12.4)
    ax.set_ylim(-6.0, -1.0)
    legend(ax, loc='lower left')
    fig.tight_layout()
    save_figure(fig, 'fig14_stellar_mass_function')


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

def plot_5_merger_clock(runs):
    """The clock alpha sets, the time satellites get, and where their stars go.

    Three panels, in the order the argument runs:

    (a) the clock.  T_df is the dynamical-friction time written into MergTime at
        infall, T_df = alpha * t_dyn.fric., so alpha shifts the whole
        distribution rigidly in the log.  This is the answer to "is it 1 Gyr or
        2 Gyr": at the published alpha = 2 the median is 5.5 Gyr, longer than
        most satellites are ever going to get.
    (b) the time a satellite actually gets between infall and the end, split by
        which end it was.  Stars go to the ICS while the clock is still running
        and to the BCG once it has expired, so shortening alpha hands the BCG
        channel the satellites that have been in the halo longest and leaves the
        ICS the ones destroyed soon after they fell in.
    (c) the split that follows, by mass and by number.

    The per-channel medians behind (b), including the mass-weighted ones, are
    printed by report_clock() rather than drawn.
    """
    print('Figure 5: the merger clock')
    fig, axes = panel_grid(1, 3)
    ax_df, ax_life, ax_route = axes

    log_bins = np.linspace(-3.0, 2.0, 60)

    def _step(ax, values, bins, colour, label, st):
        if values.size == 0:
            return
        h, e = np.histogram(values, bins=bins, density=True)
        ax.step(0.5 * (e[:-1] + e[1:]), h, where='mid', color=colour,
                label=label, **st)

    alphas = []
    for alpha, hdr, colour, ev in runs:
        st = line_style(alpha)
        ok = ev['clock_known']
        lw = 2.8 if alpha == FIDUCIAL else 1.8

        t = ev['t_df'][ok & (ev['t_df'] > 0)]
        _step(ax_df, np.log10(t), log_bins, colour,
              alpha_label(alpha) + rf'  (median {np.median(t):.2f} Gyr)', st)

        # cumulative, not a density: eight overlapping histograms in a range
        # this narrow are unreadable, and what matters is the offset between
        # the two channels, which a cumulative curve shows directly
        live = ok & (ev['t_life'] > 0)
        ics, bcg = live & ev['to_ics'], live & ev['to_bcg']
        for sel, ls in ((ics, '-'), (bcg, '--')):
            v = np.sort(ev['t_life'][sel])
            if v.size:
                ax_life.plot(np.log10(v), np.arange(1, v.size + 1) / v.size,
                             color=colour, lw=lw, ls=ls, zorder=Z_LINE)

        alphas.append(alpha)

    ax_df.set_title(r'(a) the clock alpha sets at infall, '
                    r'$T_{\rm df} = \alpha\,t_{\rm dyn.fric.}$', fontsize=8)
    ax_df.set_xlabel(r'$\log_{10}(T_{\rm df}\ [{\rm Gyr}])$')
    ax_life.set_title(r'(b) time from infall to the end, by where the stars went',
                      fontsize=8)
    ax_life.set_xlabel(r'$\log_{10}(t_{\rm infall \to end}\ [{\rm Gyr}])$')
    ax_df.set_ylabel(r'probability density per dex')
    ax_life.set_ylabel(r'cumulative fraction of the channel')
    ax_life.set_ylim(0.0, 1.0)
    ax_life.axhline(0.5, color='0.7', lw=0.8, ls=':', zorder=1)
    for ax in (ax_df, ax_life):
        ax.set_xlim(-2.5, 1.8)

    # the age of the universe is the ceiling any clock has to beat to expire
    ax_df.axvline(np.log10(AGE_NOW), color='0.5', lw=0.9, ls=':', zorder=1)
    ax_df.text(np.log10(AGE_NOW) - 0.04, ax_df.get_ylim()[1] * 0.97,
               'age of the universe', rotation=90, ha='right', va='top',
               fontsize=6.5, color='0.4')
    legend(ax_df, loc='upper left')

    # colour carries alpha, line style carries the destination
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, lw=2.8 if a == FIDUCIAL else 1.8,
                      label=alpha_label(a)) for a, _, c, _ in runs]
    handles += [Line2D([], [], color='0.25', lw=1.8, ls='-', label='to the ICS'),
                Line2D([], [], color='0.25', lw=1.8, ls='--', label='to the BCG')]
    legend(ax_life, loc='upper left', handles=handles,
           labels=[h.get_label() for h in handles])

    order = np.argsort(alphas)
    a = np.asarray(alphas)[order]
    pos = np.arange(len(a))        # evenly spaced: the sweep is a factor of 2

    # (c) where the accreted stellar mass ends up, as a function of alpha
    f_mass, f_num = [], []
    for _, _, _, ev in runs:
        m = ev['mstar']
        tot = m.sum()
        f_mass.append(m[ev['to_ics']].sum() / tot if tot > 0 else np.nan)
        f_num.append(ev['to_ics'].mean())
    fm = np.asarray(f_mass)[order]
    fn = np.asarray(f_num)[order]

    ax_route.plot(pos, fm, 'o-', color='#08519c', lw=2.4, ms=6,
                  label='by stellar mass', zorder=Z_LINE)
    ax_route.plot(pos, fn, 's--', color='#f16913', lw=2.0, ms=5.5,
                  label='by number of satellites', zorder=Z_LINE)
    for xi, yi in zip(pos, fm):
        ax_route.annotate(rf'{yi * 100:.0f}\%', (xi, yi),
                          textcoords='offset points', xytext=(0, 8),
                          ha='center', fontsize=7, color='#08519c')
    for xi, yi in zip(pos, fn):
        ax_route.annotate(rf'{yi * 100:.0f}\%', (xi, yi),
                          textcoords='offset points', xytext=(0, -12),
                          ha='center', fontsize=7, color='#f16913')

    ax_route.set_xticks(pos)
    ax_route.set_xticklabels([f'{v:g}' for v in a])
    ax_route.set_xlim(-0.35, len(a) - 0.65)
    ax_route.set_xlabel(r'$\alpha$ (MergerTimeFactor)')
    ax_route.set_ylabel(r'fraction of destroyed satellites routed to the ICS')
    ax_route.set_ylim(0.0, 1.05)
    ax_route.set_title(r'(c) the routing that follows', fontsize=8)
    legend(ax_route, loc='lower right')

    fig.tight_layout()
    save_figure(fig, 'fig5_merger_clock')


# ============ FIGURE 6: FIGURE 1 AS RAW DATA ============

def plot_6_fics_scatter(scans):
    """Figure 1 as raw data: one marker per halo, no medians, nothing binned.

    Only the model, only through the surface-brightness cut, and only the
    haloes the observations actually sample.  Groups and clusters are separated
    by both marker and colour -- triangles on an orange ramp and circles on a
    blue one -- with shade still reading as alpha inside either class.  Every
    halo at every snapshot is drawn, so the cluster end shows what it really
    is: a handful of objects per run in a 100 Mpc/h box.

    The cut uses one edge of g, named in each panel title.  SCATTER_G picks
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

    gval = min(MODEL_SB_RECOVERY['g']) if SCATTER_G == 'low' \
        else max(MODEL_SB_RECOVERY['g'])
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
                                  label=rf'{st["label"]}, $\alpha = {alpha:g}$'))
    legend(ax_m, loc='upper left', ncol=2, handles=handles,
           labels=[h.get_label() for h in handles], fontsize=6.5)

    cut = rf'$g = {gval:.2f}$, {(1 - gval) * 100:.0f}\% of the ICL below the cut'
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
    for snap in [s_ for s_ in SNAPS if s_ <= Z0_SNAP]:
        for fp in files:
            with h5.File(fp, 'r') as f:
                key = f'Snap_{snap}'
                if key not in f or f[key]['Type'].shape[0] == 0 or 'ICS_disrupt' not in f[key]:
                    continue
                g = f[key]
                gi = g['GalaxyIndex'][:].astype(np.int64)
                pos = np.clip(np.searchsorted(gids, gi), 0, gids.size - 1)
                hit = gids[pos] == gi
                if not hit.any():
                    continue
                rows = pos[hit]
                A['ics'][rows, snap] = g['IntraClusterStars'][:][hit] * conv
                A['dis'][rows, snap] = g['ICS_disrupt'][:][hit] * conv
                A['acc'][rows, snap] = g['ICS_accrete'][:][hit] * conv
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
    handles = [Line2D([], [], color=DISRUPT_COLOUR, lw=2, label=r'in situ (\texttt{ICS\_disrupt})'),
               Line2D([], [], color=ACCRETE_COLOUR, lw=2, label=r'ex situ (\texttt{ICS\_accrete})'),
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
    H = {k: np.full((ntot, len(REDSHIFTS)), np.nan) for k in _K_FIELDS}
    snaps = [s for s in SNAPS if s <= Z0_SNAP]

    off = 0
    for fp, gi_track in zip(files, track):
        n = gi_track.size
        if n == 0:
            continue
        with h5.File(fp, 'r') as f:
            for s in snaps:
                key = f'Snap_{s}'
                if key not in f or f[key]['Type'].shape[0] == 0:
                    continue
                d = {p: f[key][p][:] for p in _K_PROPS}
                gal = d['GalaxyIndex'].astype(np.int64)
                o = np.argsort(gal)
                pos = np.clip(np.searchsorted(gal[o], gi_track), 0, gal.size - 1)
                found = gal[o][pos] == gi_track
                if not found.any():
                    continue
                host = d['CentralGalaxyIndex'][o][pos[found]].astype(np.int64)
                uh, inv = np.unique(host, return_inverse=True)
                m = _group_measures(d, conv, REDSHIFTS[s], box, uh)
                for k in _K_FIELDS:
                    H[k][off:off + n, s][found] = m[k][inv]
        off += n
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
    ax.set_ylabel(r'clusters [\%]')
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
                             label=rf'{lab} {KIMMIG_TAIL_PCT:g}\% ($N = {sel.sum()}$)')
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
             r'the early (red) and late (blue) $z_{\rm form}$ tails (Kimmig+25 fig.~5)',
             ha='center', fontsize=7, color='0.35')
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    save_figure(fig, 'fig7_kimmig_assembly')


# ============ FIGURE 8: f_ICL+BCG AS A CLOCK (their figs 3, 6, 7, 10) ============

def plot_8_clock(runs):
    """f_ICL+BCG against halo mass and z_form, f_sub against z_form, and the SHMR.

    Points are the published run, coloured by z_form.  Lines in the alpha
    colours are each run's median (a) or ODR fit (b, c); grey lines are the
    paper's four simulations.
    """
    print('Figure 8: f_ICL+BCG as a dynamical clock')
    fig, axes = panel_grid(2, 2)
    (ax_m, ax_z), (ax_s, ax_shmr) = axes
    alpha_f, hdr_f, colour_f, Kf = _fiducial(runs)
    zf = Kf['zform']
    lm = np.log10(Kf['z0']['mhost'])

    # (a) f against halo mass, coloured by z_form (their fig 6)
    sc = _zform_scatter(ax_m, lm, Kf['z0']['f'], zf)
    bins = np.linspace(np.log10(KIMMIG_HOST_LO), max(np.nanmax(lm), 14.0) + 0.01, 7)
    for alpha, hdr, colour, K in runs:
        c, p = binned_percentiles(np.log10(K['z0']['mhost']), K['z0']['f'], bins,
                                  min_count=KIMMIG_MIN)
        ok = np.isfinite(p[1])
        ax_m.plot(c[ok], p[1][ok], color=colour, label=alpha_label(alpha),
                  **line_style(alpha))
    lo, hi = min(KIMMIG_REF['median_f']), max(KIMMIG_REF['median_f'])
    ax_m.axhspan(lo, hi, color='0.6', alpha=0.2, lw=0, zorder=1,
                 label='Kimmig+25 medians (4 sims)')
    ax_m.set_xlabel(r'$\log_{10}(M_{\rm vir}\ [{\rm M}_\odot])$')
    ax_m.set_ylabel(r'$f_{\rm ICL+BCG}$')
    ax_m.set_ylim(0.0, 1.0)
    legend(ax_m, loc='lower left')
    _zform_colourbar(fig, sc, ax_m)

    # (b) f against z_form (their fig 7, top)
    sc = _zform_scatter(ax_z, zf, Kf['z0']['f'], zf)
    _paper_lines(ax_z, KIMMIG_REF['fit_zform_f'], (0.0, 1.0), invert=True)
    # their fit is z_form(f); ours is drawn the same way round for a like comparison
    for alpha, hdr, colour, K in runs:
        fit = _odr_line(K['z0']['f'], K['zform'])
        r = _pearson(K['z0']['f'], K['zform'])
        if fit is None:
            continue
        fs = np.linspace(0.0, 1.0, 50)
        ax_z.plot(fit[0] * fs + fit[1], fs, color=colour,
                  label=alpha_label(alpha) + rf', $r = {r:.2f}$', **line_style(alpha))
    ax_z.set_xlabel(r'$z_{\rm form}$')
    ax_z.set_ylabel(r'$f_{\rm ICL+BCG}$')
    ax_z.set_xlim(0.0, 2.0)
    ax_z.set_ylim(0.0, 1.0)
    ax_z.set_title(r'grey: Kimmig+25 $z_{\rm form} = A f + B$, $r = 0.69$--$0.83$',
                   fontsize=8)
    legend(ax_z, loc='lower right', ncol=2)

    # (c) f_sub against z_form (their fig 7, bottom) -- a halo quantity, alpha-free
    _zform_scatter(ax_s, zf, Kf['z0']['fsub'], zf)
    _paper_lines(ax_s, KIMMIG_REF['fit_zform_fsub'], (0.0, 0.3), invert=True)
    fit = _odr_line(Kf['z0']['fsub'], zf)
    r = _pearson(Kf['z0']['fsub'], zf)
    if fit is not None:
        fs = np.linspace(0.0, 0.3, 50)
        ax_s.plot(fit[0] * fs + fit[1], fs, color=colour_f, lw=2.4,
                  label=rf'SAGE26 (tree), $r = {r:.2f}$', zorder=Z_LINE)
    ax_s.set_xlabel(r'$z_{\rm form}$')
    ax_s.set_ylabel(r'$f_{\rm sub}$ (satellite subhalo mass / $M_{\rm vir}$)')
    ax_s.set_xlim(0.0, 2.0)
    ax_s.set_ylim(0.0, 0.3)
    ax_s.set_title(r'grey: Kimmig+25, $r = -0.55$ to $-0.75$', fontsize=8)
    legend(ax_s, loc='upper right', ncol=2)

    # (d) stellar-to-halo mass: everything (grey), BCG+ICS (by z_form), BCG alone
    tot = Kf['z0']['m_bcg'] + Kf['z0']['m_ics'] + Kf['z0']['m_sat']
    ax_shmr.plot(lm, np.log10(tot), 'o', color='0.65', ms=3.5, mec='none',
                 ls='none', zorder=Z_BAND, label=r'BCG + ICS + satellites', rasterized=True)
    sc = _zform_scatter(ax_shmr, lm, np.log10(Kf['z0']['m_bcg'] + Kf['z0']['m_ics']), zf,
                        label=r'BCG + ICS')
    for alpha, hdr, colour, K in runs:
        c, p = binned_percentiles(np.log10(K['z0']['mhost']), np.log10(K['z0']['m_bcg']),
                                  bins, min_count=KIMMIG_MIN)
        ok = np.isfinite(p[1])
        ax_shmr.plot(c[ok], p[1][ok], color=colour,
                     label=alpha_label(alpha) + r', BCG alone', **line_style(alpha))
    ax_shmr.set_xlabel(r'$\log_{10}(M_{\rm vir}\ [{\rm M}_\odot])$')
    ax_shmr.set_ylabel(r'$\log_{10}(m_\star\ [{\rm M}_\odot])$')
    legend(ax_shmr, loc='upper left')
    _zform_colourbar(fig, sc, ax_shmr)

    fig.tight_layout()
    save_figure(fig, 'fig8_kimmig_clock')


# ============ FIGURE 9: OBSERVABLE PROXIES (their fig 8) ============

def plot_9_proxies(runs):
    """f_ICL+BCG against M12, M14, the stellar centre shift and phi_BCG."""
    print('Figure 9: observable proxies for f_ICL+BCG')
    fig, axes = panel_grid(1, 4)
    alpha_f, hdr_f, colour_f, Kf = _fiducial(runs)
    zf = Kf['zform']
    panels = (
        ('M12', r'$\log_{10}(M_{12})$', 'fit_f_M12', 'r_f_M12', (0.0, 2.0)),
        ('M14', r'$\log_{10}(M_{14})$', 'fit_f_M14', 'r_f_M14', (0.0, 2.2)),
        ('s_stars', r'$\log_{10}(s_{\rm stars})$', 'fit_f_sstars', 'r_f_sstars', (-3.0, -0.3)),
        ('phi', r'$\log_{10}(\phi_{\rm BCG}\ [{\rm M}_\odot\,{\rm kpc}^{-1}])$',
         'fit_f_phi', 'r_f_phi', (9.5, 12.0)),
    )
    for ax, (key, xlabel, fitkey, rkey, xlim) in zip(axes, panels):
        with np.errstate(invalid='ignore', divide='ignore'):
            x = np.log10(_get(Kf, key))
        sc = _zform_scatter(ax, x, Kf['z0']['f'], zf)
        _paper_lines(ax, KIMMIG_REF[fitkey], xlim)
        _fit_lines(ax, runs, key, 'f', xlim, log_x=True)
        rs = ', '.join(f'{v:.2f}' for v in KIMMIG_REF[rkey])
        ax.set_title(rf'Kimmig+25 $r$ = {rs}', fontsize=7.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r'$f_{\rm ICL+BCG}$')
        ax.set_xlim(*xlim)
        ax.set_ylim(0.0, 1.0)
        legend(ax, loc='lower right' if key != 's_stars' else 'lower left',
               fontsize=6.0)
    _zform_colourbar(fig, sc, axes[-1])
    fig.tight_layout()
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

    Grey shading is the published run's distribution; lines are each run's
    median with y0 -- the shredding rate -- in the legend.  The band is
    Kimmig+25's 3-4 per cent per Gyr.  The main branches are those of the z = 0
    sample, binned by their mass at the later snapshot, as they do.
    """
    print('Figure 12: the shredding rate')
    fig, axes = panel_grid(len(_SHRED_ROWS), len(_SHRED_COLS), squeeze=False,
                           sharex=True, sharey=True)
    pairs = [(alpha, colour, shredding_pairs(K)) for alpha, hdr, colour, K in runs]
    alpha_f = _fiducial(runs)[0]
    for i, (rname, mlo, mhi) in enumerate(_SHRED_ROWS):
        for j, (cname, zlo, zhi) in enumerate(_SHRED_COLS):
            ax = axes[i, j]
            ax.axhspan(*KIMMIG_SHRED, color='#fdae61', alpha=0.35, lw=0, zorder=1,
                       label=r'Kimmig+25 $y_0$')
            ax.axhline(0, color='0.6', lw=0.7)
            ax.axvline(0, color='0.6', lw=0.7)
            for alpha, colour, P in pairs:
                sel = ((P['m2'] >= mlo) & (P['m2'] < mhi) &
                       (P['z2'] > zlo) & (P['z2'] <= zhi))
                if alpha == alpha_f and sel.sum():
                    ax.hist2d(P['dm'][sel], P['df'][sel], bins=(40, 40),
                              range=((-0.3, 0.6), (-0.4, 0.4)), cmap='Greys',
                              cmin=1, alpha=0.6, zorder=Z_BAND, rasterized=True)
                c, p = binned_percentiles(P['dm'][sel], P['df'][sel], _DM_BINS,
                                          min_count=KIMMIG_MIN)
                ok = np.isfinite(p[1])
                y0 = shredding_rate(P['dm'][sel], P['df'][sel]) if sel.sum() else np.nan
                lab = alpha_label(alpha) + (rf', $y_0 = {y0 * 100:.1f}\%$'
                                            if np.isfinite(y0) else '')
                ax.plot(c[ok], p[1][ok], color=colour, label=lab, **line_style(alpha))
            ax.set_title(f'{rname}, {cname}', fontsize=8)
            legend(ax, loc='upper right', fontsize=6.0)
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
    head = f'{"":>20}' + ''.join(f'{alpha_label(a).replace("$", ""):>22}'
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

    # per-cluster listing: small samples deserve to be seen whole
    alpha_f, _, _, Kf = _fiducial(runs)
    n = len(Kf['zform'])
    if n <= 40:
        print()
        print(f'  every host, {alpha_label(alpha_f).replace("$", "")} '
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


# ========================== REPORTS ==========================

def report_runs(scans):
    """The z = 0 budget each figure is a view of."""
    print()
    print(f'Sweep over MergerTimeFactor, microUchuu, z = {REDSHIFTS[Z0_SNAP]:.3f} '
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
                    if sel.sum() else np.nan for g in MODEL_SB_RECOVERY['g']]
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
        print(f'    destroyed stellar mass {m.sum() / vol:.3e} Msun/Mpc^3: to ICS '
              f'{m[ev["to_ics"]].sum() / vol:.3e}, to BCG {m[ev["to_bcg"]].sum() / vol:.3e}')

        print(f'    by satellite stellar mass at destruction')
        print(f'      {"log m*":<11}{"N":>9}{"mass share":>12}{"N->ICS":>9}{"M->ICS":>9}'
              f'{"T_df med":>10}{"t_life med":>11}')
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
                  f'{np.nanmedian(ev["t_life"][k]) if k.any() else np.nan:>11.2f}')

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
    2: ('scan', plot_2_bulge_to_total),
    3: ('scan', plot_3_sfrd),
    4: ('scan', plot_4_ics_mass_function),
    5: ('disrupt', plot_5_merger_clock),
    6: ('scan', plot_6_fics_scatter),
    7: ('kimmig', plot_7_assembly),
    8: ('kimmig', plot_8_clock),
    9: ('kimmig', plot_9_proxies),
    10: ('kimmig', plot_10_matrix),
    11: ('kimmig', plot_11_budget),
    12: ('kimmig', plot_12_shredding),
    13: ('assembly', plot_13_ics_assembly),
    14: ('scan', plot_14_stellar_mass_function),
}


def main(argv):
    refresh = '--refresh' in argv
    wanted = sorted(int(a) for a in argv if a.isdigit()) or sorted(FIGURES)
    bad = [n for n in wanted if n not in FIGURES]
    if bad:
        sys.exit(f'No such figure: {bad}.  Choose from {sorted(FIGURES)}.')

    setup_style()
    os.makedirs(OUT_DIR, exist_ok=True)
    report_header()

    needs_scan = any(FIGURES[n][0] == 'scan' for n in wanted)
    needs_disrupt = any(FIGURES[n][0] == 'disrupt' for n in wanted)
    needs_kimmig = any(FIGURES[n][0] == 'kimmig' for n in wanted)
    needs_assembly = any(FIGURES[n][0] == 'assembly' for n in wanted)

    scans = disrupt = kimmig = assembly = None
    if needs_scan:
        print('Loading snapshot scans')
        scans = all_scans(refresh=refresh)
        report_runs(scans)
        report_budget(scans)
        report_ics_mass_function(scans)
        print()
        report_sb_correction()
    if needs_disrupt:
        print()
        print('Loading disruption logs')
        disrupt = all_disrupt()
        if disrupt:
            report_clock(disrupt)
            report_clock_detail(disrupt)
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

    print()
    for n in wanted:
        kind, fn = FIGURES[n]
        data = {'scan': scans, 'disrupt': disrupt, 'kimmig': kimmig,
                'assembly': assembly}[kind]
        if not data:
            print(f'Figure {n}: no {kind} data -- skipped')
            continue
        fn(data)

    print()
    print(f'Done.  Figures in {OUT_DIR}')


if __name__ == '__main__':
    main(sys.argv[1:])

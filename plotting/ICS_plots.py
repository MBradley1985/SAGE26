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

    1  f_ICS = M_ICS / (all stars in the halo), against halo mass and redshift.
       Each run is drawn twice: the SB-free total SAGE tracks, and the same
       clusters re-measured through a surface-brightness cut, so the model is
       compared with the observations on the observations' own terms
       (see MODEL_SB_RECOVERY)
    2  bulge-to-total ratio against stellar mass
    3  cosmic star formation rate density
    4  the ICS mass function (and the stellar mass function beside it)
    5  the clock itself -- T_df handed out, how long satellites actually live,
       and the routing split that follows from comparing the two
    6  figure 1 as raw data: every halo, through the cut, no medians hiding the
       scatter and no subsampling hiding the sample size

Figures 1-4 read the z = 0 galaxy catalogues; figure 5 reads the per-event
disruption log written by the ``SAGE_DISRUPT_LOG`` diagnostic in
``core_build_model.c``, which records MergTime at the moment the gate fires
together with the infall snapshot, so the clock's starting value is recoverable.

Usage:
    python plotting/ICS_plots.py                 # all figures
    python plotting/ICS_plots.py 1 5             # figures 1 and 5 only
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
CACHE_VERSION = 'v3'

_MSUN_CGS = 1.989e33

MIN_COUNT = 10                      # objects needed in a bin before it is drawn

# Host-mass slices.  Groups and clusters are the two regimes the observations
# separate, and the two the ICS behaves differently in.
GROUP_LO, GROUP_HI = 1e13, 1e14
CLUSTER_LO = 1e14

# Shared binning.
MVIR_BINS  = np.arange(11.0, 15.01, 0.25)
MSTAR_BINS = np.arange(8.0, 12.51, 0.25)
MF_BINWIDTH = 0.2

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
#     f_obs = g * M_ICS / (g * M_ICS + M_star,halo)
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
    print(f'  f_obs = g*M_ICS / (g*M_ICS + M_star),  g = {lo:.2f}--{hi:.2f}  '
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

    Columns are log M*, B/T, and the statistical 1-sigma bounds.  Chabrier
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
#     f_ICS = M_ICS(halo) / (M_ICS(halo) + sum of M* over every galaxy in it)
#
# -- the denominator is all the stars in the halo including satellites, not the
# BCG alone.  M_ICS is summed over the halo too: satellites carry their own ICS
# until they are absorbed.

_SCAN_PROPS = ('StellarMass', 'BulgeMass', 'IntraClusterStars', 'Mvir', 'Type',
               'CentralGalaxyIndex', 'SfrDisk', 'SfrBulge')


def _halo_totals(d, conv):
    """Per-FOF-halo (Mvir, total stellar mass, total ICS) for one file+snapshot."""
    _, idx = np.unique(d['CentralGalaxyIndex'].astype(np.int64),
                       return_inverse=True)
    n = idx.max() + 1 if idx.size else 0
    if n == 0:
        return (np.array([]),) * 3
    stars = np.bincount(idx, weights=d['StellarMass'], minlength=n) * conv
    ics = np.bincount(idx, weights=d['IntraClusterStars'], minlength=n) * conv
    cen = d['Type'] == 0
    mvir = np.full(n, np.nan)
    mvir[idx[cen]] = d['Mvir'][cen] * conv
    return mvir, stars, ics


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
                          'g_mstar', 'g_bulge', 'g_type')}

    # every group- and cluster-scale halo at every snapshot, kept per object so
    # figure 6 can draw the raw distribution at any g rather than a percentile
    gc = {k: [] for k in ('gc_z', 'gc_mvir', 'gc_mstar', 'gc_ics')}

    for snap in SNAPS:
        key = f'Snap_{snap}'
        sfr_sum = 0.0
        mvir_a, stars_a, ics_a = [], [], []
        for fp in files:
            with h5.File(fp, 'r') as f:
                if key not in f:
                    continue
                g = f[key]
                if g['StellarMass'].shape[0] == 0:
                    continue
                d = {p: g[p][:] for p in _SCAN_PROPS}
            sfr_sum += float((d['SfrDisk'] + d['SfrBulge']).sum())
            mv, st, ic = _halo_totals(d, conv)
            mvir_a.append(mv)
            stars_a.append(st)
            ics_a.append(ic)
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

    out = dict(sfrd=sfrd, fics_cl=fics_cl, fics_gr=fics_gr,
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
    ax_m.set_ylabel(r'$f_{\rm ICS} = M_{\rm ICS} / (M_{\rm ICS} + M_{\star,\rm halo})$')
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

    ax_bt.set_xlabel(r'$\log_{10}(M_\star\ [{\rm M}_\odot])$')
    ax_bt.set_ylabel(r'$B/T$')
    ax_bt.set_xlim(8.0, 12.2)
    ax_bt.set_ylim(0.0, 1.05)
    legend(ax_bt, loc='upper left')

    ax_fr.set_xlabel(r'$\log_{10}(M_\star\ [{\rm M}_\odot])$')
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

def plot_4_mass_functions(scans):
    """The halo ICS mass function at z = 0, and the stellar mass function."""
    print('Figure 4: mass functions')
    fig, axes = panel_grid(1, 2)
    ax_ics, ax_smf = axes

    for alpha, hdr, colour, s in scans:
        vol = float(s['volume'][0])
        st = line_style(alpha)

        ics = s['h_ics']
        x, phi = mass_function(np.log10(ics[ics > 0]), vol,
                               mass_range=(7.0, 13.5))
        ax_ics.plot(x, np.log10(phi), color=colour,
                    label=alpha_label(alpha), **st)

        m = s['g_mstar']
        x2, phi2 = mass_function(np.log10(m[m > 0]), vol,
                                 mass_range=(8.0, 12.5))
        ax_smf.plot(x2, np.log10(phi2), color=colour,
                    label=alpha_label(alpha), **st)

    ax_ics.set_xlabel(r'$\log_{10}(M_{\rm ICS}\ [{\rm M}_\odot])$')
    ax_ics.set_ylabel(r'$\log_{10}(\phi\ [{\rm Mpc}^{-3}\,{\rm dex}^{-1}])$')
    ax_ics.set_xlim(8.0, 13.2)
    ax_ics.set_ylim(-6.0, -1.0)
    legend(ax_ics, loc='lower left')

    ax_smf.set_xlabel(r'$\log_{10}(M_\star\ [{\rm M}_\odot])$')
    ax_smf.set_ylabel(r'$\log_{10}(\phi\ [{\rm Mpc}^{-3}\,{\rm dex}^{-1}])$')
    ax_smf.set_xlim(8.0, 12.4)
    ax_smf.set_ylim(-6.0, -1.0)
    legend(ax_smf, loc='lower left')

    fig.tight_layout()
    save_figure(fig, 'fig4_mass_functions')


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


# ========================== DRIVER ==========================

FIGURES = {
    1: ('scan', plot_1_fics),
    2: ('scan', plot_2_bulge_to_total),
    3: ('scan', plot_3_sfrd),
    4: ('scan', plot_4_mass_functions),
    5: ('disrupt', plot_5_merger_clock),
    6: ('scan', plot_6_fics_scatter),
}


def main(argv):
    refresh = '--refresh' in argv
    wanted = sorted(int(a) for a in argv if a.isdigit()) or sorted(FIGURES)
    bad = [n for n in wanted if n not in FIGURES]
    if bad:
        sys.exit(f'No such figure: {bad}.  Choose from {sorted(FIGURES)}.')

    setup_style()
    os.makedirs(OUT_DIR, exist_ok=True)

    needs_scan = any(FIGURES[n][0] == 'scan' for n in wanted)
    needs_disrupt = any(FIGURES[n][0] == 'disrupt' for n in wanted)

    scans = disrupt = None
    if needs_scan:
        print('Loading snapshot scans')
        scans = all_scans(refresh=refresh)
        report_runs(scans)
        print()
        report_sb_correction()
    if needs_disrupt:
        print()
        print('Loading disruption logs')
        disrupt = all_disrupt()
        if disrupt:
            report_clock(disrupt)

    print()
    for n in wanted:
        kind, fn = FIGURES[n]
        data = scans if kind == 'scan' else disrupt
        if not data:
            print(f'Figure {n}: no {kind} data -- skipped')
            continue
        fn(data)

    print()
    print(f'Done.  Figures in {OUT_DIR}')


if __name__ == '__main__':
    main(sys.argv[1:])

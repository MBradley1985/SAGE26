#!/usr/bin/env python
"""
SAGE26 -- where satellite stars end up
======================================

Every satellite SAGE destroys is routed to exactly one of two places, decided by
a single test in core_build_model.c once the disruption criterion has fired:

    MergTime >  0   ->  disrupt_satellite_to_ICS()      stars become intracluster
    MergTime <= 0   ->  deal_with_galaxy_merger()       stars land on the central

The galaxy's last written record carries the verdict in ``mergeType``:
    4        disrupted into the ICS
    1        minor merger onto the central   (mass ratio <= ThreshMajorMerger)
    2        major merger onto the central   (mass ratio >  ThreshMajorMerger)
    0        still alive at this snapshot
and that record still holds the satellite's pre-destruction ``StellarMass`` --
the zeroing in model_mergers.c happens on the working copy, after the output
copy is taken.  Walking every snapshot and collecting the non-zero mergeType
records therefore gives a complete, time-resolved catalogue of the accreted
stellar mass and its destination.  That catalogue is what this module is.

Four figures:
    1  which satellites go to the ICS and which to the BCG
    2  how long they survive between infall and destruction
    3  when the ICS is built, and out of how much satellite mass
    4  when the BCG's accreted component is built, and out of how much

Everything is the default (published) SAGE26 run -- no parameter sweeps.

Usage:
    python plotting/ICS_plots.py               # all figures
    python plotting/ICS_plots.py 1 3           # figures 1 and 3 only
    python plotting/ICS_plots.py --refresh     # rebuild the event catalogue
"""

import os
import sys

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

import warnings
warnings.filterwarnings("ignore")


# ========================== CONFIGURATION ==========================

# The default SAGE26 run.  This module deliberately reads one box: the
# parameter sweeps and the FFB variants are a different question.
MODEL_DIR = './output/microuchuu/'
OUTPUT_FORMAT = '.pdf'
OBS_DIR = './data/'

_MSUN_CGS = 1.989e33
_SEC_PER_GYR = 3.15576e16

# Objects required in a bin before a median or a ratio is drawn from it.
MIN_COUNT = 10

# Figure 5 draws one marker per halo; cap the plotted cloud so the PDF stays
# light, with a fixed seed so the figure is reproducible.
MAX_SCATTER_POINTS = 2500
SCATTER_SEED = 2222

# Marker sizes for figure 5.  The model cloud is drawn semi-transparent so a
# dense selection reads as shading while a sparse one stays legible as points.
MODEL_MS, MODEL_ALPHA = 4.0, 0.30
OBS_MS = 7.5

# Draw order.
Z_BAND = 2
Z_OBS = 5
Z_LINE = 10

# The two destinations, and the colours they keep in every figure.
DEST_ICS = 'ICS'
DEST_BCG = 'BCG'
COLOUR = {DEST_ICS: '#08519c', DEST_BCG: '#f16913'}
DEST_LABEL = {
    DEST_ICS: r'to ICS ($\mathtt{mergeType}=4$)',
    DEST_BCG: r'to BCG ($\mathtt{mergeType}=1,2$)',
}

# Host-mass slices used wherever a figure splits by environment.  Chosen to
# separate the regime where the ICS is a minor reservoir from the cluster
# regime where it dominates.
HOST_BINS = (
    (11.0, 12.0, r'$10^{11}$--$10^{12}$'),
    (12.0, 13.0, r'$12$--$13$'),
    (13.0, 14.0, r'$13$--$14$'),
    (14.0, 15.5, r'$>10^{14}$'),
)
HOST_BIN_COLOURS = ('#c6dbef', '#6baed6', '#2171b5', '#08306b')

# Shared binning.
MSTAR_BINS = np.arange(6.0, 12.01, 0.25)     # satellite stellar mass
MVIR_BINS  = np.arange(10.5, 15.01, 0.25)    # host halo mass


# ========================== SIMULATION HEADER ==========================

def find_model_files(directory):
    """All model_*.hdf5 files in *directory*, sorted.  Empty list if none."""
    import glob
    files = sorted(glob.glob(os.path.join(directory, 'model_*.hdf5')))
    # model.hdf5 (no rank suffix) is a concatenation of the per-rank files and
    # would double-count every event if it were picked up alongside them.
    return [f for f in files if os.path.basename(f) != 'model.hdf5']


def read_sim_header(directory):
    """Simulation parameters from the HDF5 header.  None if no model files."""
    files = find_model_files(directory)
    if not files:
        return None

    with h5.File(files[0], 'r') as f:
        sim, runtime = f['Header/Simulation'], f['Header/Runtime']
        hdr = {
            'hubble_h':       float(sim.attrs['hubble_h']),
            'box_size':       float(sim.attrs['box_size']),
            'omega_matter':   float(sim.attrs['omega_matter']),
            'omega_lambda':   float(sim.attrs['omega_lambda']),
            'particle_mass':  float(sim.attrs.get('particle_mass', np.nan)),
            'last_snap_nr':   int(sim.attrs['LastSnapshotNr']),
            'unit_mass_in_g': float(runtime.attrs['UnitMass_in_g']),
            'unit_length_in_cm':   float(runtime.attrs['UnitLength_in_cm']),
            'unit_velocity_in_cms': float(runtime.attrs['UnitVelocity_in_cm_per_s']),
            'redshifts':      np.array(f['Header/snapshot_redshifts'][:]),
            'output_snaps':   sorted(int(s) for s in f['Header/output_snapshots'][:]),
            'merger_time_factor':     float(runtime.attrs.get('MergerTimeFactor', np.nan)),
            'thresh_major_merger':    float(runtime.attrs.get('ThreshMajorMerger', np.nan)),
            'thresh_sat_disruption':  float(runtime.attrs.get('ThresholdSatDisruption', np.nan)),
            'track_ics_assembly':     int(runtime.attrs.get('TrackICSAssembly', 0)),
        }

    total_fvp = 0.0
    for fp in files:
        with h5.File(fp, 'r') as f:
            total_fvp += float(f['Header/Runtime'].attrs['frac_volume_processed'])
    hdr['volume_fraction'] = total_fvp
    hdr['files'] = files
    return hdr


HDR = read_sim_header(MODEL_DIR)
if HDR is None:
    sys.exit(f'No model_*.hdf5 found in {MODEL_DIR}')

HUBBLE_H     = HDR['hubble_h']
OMEGA_M      = HDR['omega_matter']
OMEGA_L      = HDR['omega_lambda']
REDSHIFTS    = HDR['redshifts']
SNAPS        = HDR['output_snaps']
LAST_SNAP    = HDR['last_snap_nr']
MODEL_FILES  = HDR['files']
VOLUME       = (HDR['box_size'] / HUBBLE_H) ** 3 * HDR['volume_fraction']   # Mpc^3
MASS_CONVERT = HDR['unit_mass_in_g'] / _MSUN_CGS / HUBBLE_H                 # code -> Msun

# consistent-trees collapses the FOF grouping at the very last snapshot of a
# run: centrals drop by a third and the missing ones reappear as satellites of
# something else.  Every halo-grouped z=0 quantity in this module is therefore
# measured one snapshot earlier.  Destruction events are unaffected -- they are
# read at the snapshot they fired on, and the last snapshot stamps none.
Z0_SNAP = LAST_SNAP - 1

OUTPUT_DIR = os.path.join(MODEL_DIR, 'ICS_plots/')
# v2 adds the host Rvir/Vvir columns; the name is bumped so a stale v1
# cache is rebuilt rather than silently loaded without them.
CACHE_FILE = os.path.join(OUTPUT_DIR, 'satellite_events_v2.npz')


# ========================== COSMIC TIME ==========================

def cosmic_time_gyr(z):
    """Age of the universe at redshift *z*, in Gyr."""
    t_H = 977.8 / (HUBBLE_H * 100.0)     # Hubble time in Gyr

    def integrand(zp):
        return 1.0 / ((1 + zp) * np.sqrt(OMEGA_M * (1 + zp) ** 3 + OMEGA_L))

    result, _ = quad(integrand, z, 1000.0)
    return t_H * result


# Age of the universe at every snapshot, so the event catalogue can convert a
# snapshot number to a time without re-integrating.
AGE_AT_SNAP = np.array([cosmic_time_gyr(z) for z in REDSHIFTS])
AGE_NOW = AGE_AT_SNAP[-1]

# Redshift width each snapshot stands for, used to spread scatter plots that
# would otherwise stack every halo on a handful of exact snapshot redshifts.
_SNAP_Z_SORTED = np.sort(REDSHIFTS)
_SNAP_DZ_SORTED = np.gradient(_SNAP_Z_SORTED)


def snap_to_age(snap):
    """Age of the universe in Gyr at (possibly fractional) snapshot number."""
    return np.interp(np.asarray(snap, dtype=float),
                     np.arange(len(AGE_AT_SNAP)), AGE_AT_SNAP)


# Every cosmic-time x-axis in this module runs with the present day on the
# LEFT and lookback increasing to the right, so the figures read the way an
# assembly history is usually discussed: "by z = 1 the ICS was already ...".
def time_axis(ax):
    """Apply the today-on-the-left convention to an axis whose x is cosmic time."""
    ax.set_xlim(AGE_NOW, 0.0)


def _z_axis(ax, zticks=(0, 0.2, 0.5, 1, 2, 3, 5, 8)):
    """Put a redshift axis on top of an axis whose x is cosmic time in Gyr."""
    top = ax.twiny()
    top.set_xlim(ax.get_xlim())
    ages = [cosmic_time_gyr(z) for z in zticks]
    top.set_xticks(ages)
    top.set_xticklabels([f'{z:g}' for z in zticks], fontsize=7.5)
    top.set_xlabel(r'$z$', fontsize=8, labelpad=4)
    return top


# ========================== STYLE AND FIGURE UTILITIES ==========================

# The house style is sized for a single full-width panel (20 pt axis labels on
# an 8.34 x 6.25 in canvas).  Every figure here is a 2x2 or 1x3 grid, so the
# type is scaled down by PANEL_FONT_SCALE and each panel is given roughly the
# style's native aspect at PANEL_W x PANEL_H.  Typeface, usetex, tick style and
# line weights are left exactly as the style sets them.
PANEL_FONT_SCALE = 0.60
PANEL_W, PANEL_H = 5.6, 4.5


def setup_style():
    plt.style.use("./plotting/kieren_cohare_palatino_sty.mplstyle")
    f = PANEL_FONT_SCALE
    plt.rcParams.update({
        'font.size':        plt.rcParams['font.size'] * f,
        'axes.labelsize':   plt.rcParams['axes.labelsize'] * f,
        'axes.titlesize':   plt.rcParams['axes.titlesize'] * f,
        'xtick.labelsize':  plt.rcParams['xtick.labelsize'] * f,
        'ytick.labelsize':  plt.rcParams['ytick.labelsize'] * f,
        'legend.fontsize':  plt.rcParams['legend.fontsize'] * f,
        'xtick.major.size': 5.0, 'ytick.major.size': 5.0,
        'xtick.minor.size': 3.0, 'ytick.minor.size': 3.0,
        'xtick.major.pad':  4,
        'axes.linewidth':   1.0,
        # tight_layout is called explicitly; autolayout fights the twin axes
        'figure.autolayout': False,
    })


def panel_grid(nrows, ncols, **kwargs):
    """A figure whose panels each get the style's native proportions."""
    return plt.subplots(nrows, ncols,
                        figsize=(ncols * PANEL_W, nrows * PANEL_H), **kwargs)


def _tex_safe(s):
    """Make a label safe when usetex is off (the style file turns it on)."""
    if not plt.rcParams.get('text.usetex', False):
        s = s.replace(r'\&', '&').replace(r'\%', '%')
    return s


def save_figure(fig, name):
    path = os.path.join(OUTPUT_DIR, name + OUTPUT_FORMAT)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    print(f'  Saved: {path}')
    plt.close(fig)


def legend(ax, loc='best', **kwargs):
    kwargs.setdefault('frameon', False)
    kwargs.setdefault('fontsize', 7.5)
    leg = ax.legend(loc=loc, numpoints=1, labelspacing=0.1, **kwargs)
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    return leg


def binned_median(x, y, bins, weights=None, min_count=MIN_COUNT):
    """Bin centres and the 16/50/84th percentiles of *y*, NaN where too sparse."""
    x, y = np.asarray(x), np.asarray(y)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    centres = 0.5 * (bins[:-1] + bins[1:])
    pct = np.full((3, len(bins) - 1), np.nan)
    for i in range(len(bins) - 1):
        m = (x >= bins[i]) & (x < bins[i + 1])
        if m.sum() >= min_count:
            pct[:, i] = np.percentile(y[m], (16, 50, 84))
    return centres, pct


def binned_weighted_median(x, y, w, bins, min_count=MIN_COUNT):
    """Per-bin median of *y* weighted by *w* -- the epoch the mass, not the count, sees."""
    x, y, w = np.asarray(x), np.asarray(y), np.asarray(w)
    ok = np.isfinite(x) & np.isfinite(y) & (w > 0)
    x, y, w = x[ok], y[ok], w[ok]
    out = np.full(len(bins) - 1, np.nan)
    for i in range(len(bins) - 1):
        msk = (x >= bins[i]) & (x < bins[i + 1])
        if msk.sum() >= min_count:
            o = np.argsort(y[msk])
            c = np.cumsum(w[msk][o]) / w[msk].sum()
            out[i] = y[msk][o][np.searchsorted(c, 0.5)]
    return 0.5 * (bins[:-1] + bins[1:]), out


def plot_median_band(ax, x, y, bins, *, color, label=None, lw=2.6, ls='-',
                     alpha=0.18, min_count=MIN_COUNT):
    centres, (p16, p50, p84) = binned_median(x, y, bins, min_count=min_count)
    ok = np.isfinite(p50)
    if not ok.any():
        return
    ax.fill_between(centres[ok], p16[ok], p84[ok], color=color, alpha=alpha,
                    lw=0, zorder=Z_BAND)
    ax.plot(centres[ok], p50[ok], color=color, lw=lw, ls=ls, label=label,
            zorder=Z_LINE)


def binned_mass_fraction(x, mass, sel, bins, min_count=MIN_COUNT):
    """Per-bin sum(mass[sel]) / sum(mass) -- a mass-weighted routing fraction."""
    out = np.full(len(bins) - 1, np.nan)
    for i in range(len(bins) - 1):
        m = (x >= bins[i]) & (x < bins[i + 1])
        tot = mass[m].sum()
        if m.sum() >= min_count and tot > 0:
            out[i] = mass[m & sel].sum() / tot
    return 0.5 * (bins[:-1] + bins[1:]), out


# ========================== OBSERVATIONS ==========================
#
# Digitised f_ICL-versus-redshift compilation.  Values are fractions, not per
# cent.  Two things to hold in mind before reading agreement or tension off
# these points:
#
#  * f_ICL is not one measurement.  Authors cut the ICL at different surface
#    brightnesses and different radii, and disagree over whether the BCG belongs
#    in the numerator, the denominator, or neither.  Furnell+21 and Burke+15
#    overlap in redshift and still differ by a factor of a few; that spread is
#    method, not physics.  No attempt is made here to homogenise them.
#  * the model counts every intracluster star SAGE tracks, with no surface
#    brightness limit at all, so f_ICS should sit at or above a
#    surface-brightness-limited measurement rather than on top of it.  Cosmological
#    dimming biases the high-redshift points low for the same reason.
#
# 'scale' separates cluster-scale hosts from group-scale samples so each is
# drawn against the model line for the halo mass it actually belongs to.
_ICL_FRACTION_OBS = (
    dict(label='Spavone+20', scale='cluster', note='Fornax, Fornax Deep Survey',
         z=(0,), f=(0.3408,)),
    dict(label='Kluge+21', scale='cluster', note='ICL and host cluster',
         z=(0.03,), f=(0.1792,)),
    dict(label='Zibetti+05', scale='cluster', note='stacked SDSS',
         z=(0.243,), f=(0.1085,)),
    dict(label='Feldmeier+04', scale='cluster', note='deep CCD imaging',
         z=(0.162, 0.162, 0.162, 0.185),
         f=(0.1521, 0.1215, 0.1026, 0.0731)),
    dict(label='Burke+15', scale='cluster', note='CLASH',
         z=(0.403, 0.387, 0.397, 0.339, 0.344, 0.342, 0.291, 0.225, 0.218,
            0.213, 0.195, 0.177),
         f=(0.0259, 0.0271, 0.033, 0.0554, 0.0601, 0.0719, 0.1297, 0.125,
            0.1627, 0.1804, 0.1686, 0.2311)),
    dict(label='Furnell+21', scale='cluster', note='XCS-HSC',
         z=(0.144, 0.127, 0.122, 0.081, 0.225, 0.215, 0.256, 0.306, 0.261,
            0.294, 0.322, 0.342, 0.372, 0.337, 0.377, 0.329, 0.496, 0.425,
            0.109),
         f=(0.3856, 0.3066, 0.3101, 0.2889, 0.2653, 0.2358, 0.2854, 0.2972,
            0.3255, 0.2748, 0.2759, 0.2665, 0.1981, 0.1887, 0.1545, 0.1533,
            0.1132, 0.0967, 0.316)),
    dict(label=r'Montes \& Trujillo 18', scale='cluster', note='Frontier Fields',
         z=(0.301, 0.39, 0.342, 0.537, 0.537, 0.37, 0.043),
         f=(0.0767, 0.0861, 0.1309, 0.066, 0.0578, 0.0483, 0.1085)),
    dict(label=r'Montes \& Trujillo 18b', scale='cluster',
         note='Frontier Fields, second estimate',
         z=(0.534, 0.544, 0.367, 0.397, 0.342, 0.304, 0.048),
         f=(0.0153, 0, 0.0106, 0.0153, 0.0271, 0.033, 0.0861)),
    # two independent estimates for the same cluster, kept as separate points
    dict(label='Presotto+14', scale='cluster', note='MACS J1206.2-0947',
         z=(0.435, 0.433), f=(0.1226, 0.0554)),
    dict(label='Ragusa+23', scale='cluster', note='VEGAS, Antlia',
         z=(0.05,), f=(0.35,)),
    dict(label='Burke+12', scale='cluster', note='z ~ 1 clusters',
         z=(0.947, 0.83, 0.795, 0.808, 1.223),
         f=(0.0142, 0.0259, 0.0377, 0.0153, 0.0236)),
    dict(label=r'Ko \& Jee 18', scale='cluster', note='ICL detected at z = 1.24',
         z=(1.238,), f=(0.0991,)),
    dict(label='XLSSC 122 (JWST)', scale='cluster',
         note='the only constraint near z = 2',
         z=(1.98,), f=(0.17,)),
    dict(label='Ragusa+23', scale='group', note='VEGAS groups',
         z=(0.05,) * 16,
         f=(0.16, 0.05, 0.05, 0.17, 0.05, 0.27, 0.34, 0.17, 0.08, 0.35, 0.18,
            0.07, 0.2, 0.22, 0.28, 0.3)),
    dict(label='Ahad+25', scale='group', note='KiDS+GAMA groups',
         z=(0.12, 0.12, 0.12, 0.18, 0.18, 0.18, 0.24, 0.24, 0.24),
         f=(0.16, 0.1, 0.04, 0.15, 0.12, 0.08, 0.13, 0.15, 0.05)),
)

# One marker per source, so a reader can tell which sample a point came from
# rather than seeing an undifferentiated grey cloud.
_OBS_MARKERS = ('o', 's', '^', 'v', 'D', 'P', 'X', '<', '>', 'h', '*', 'p', 'd')


def load_icl_fraction_observations(scale):
    """The f_ICL compilation for hosts of a given *scale* ('cluster' or 'group')."""
    return [dict(o, z=np.asarray(o['z'], float), f=np.asarray(o['f'], float))
            for o in _ICL_FRACTION_OBS if o['scale'] == scale]


# ========================== THE EVENT CATALOGUE ==========================
#
# One row per destroyed satellite, across every snapshot of the run.
#
#   snap_dest     snapshot the destruction fired on
#   merge_type    4 = to ICS, 1 = minor merger onto central, 2 = major merger
#   sat_type      the satellite's Type at destruction (1 = has a subhalo)
#   mstar         its stellar mass immediately before destruction [Msun]
#   infall_mstar  its stellar mass when it first became a satellite [Msun]
#   snap_infall   snapshot of that first infall (-1 if never recorded)
#   host_mvir     Mvir of the central it was destroyed into [Msun]
#   host_mstar    that central's stellar mass at the same snapshot [Msun]
#
# Reading is done file by file so the CentralGalaxyIndex -> central lookup stays
# inside the file that wrote both records.

_EVENT_FIELDS = ('snap_dest', 'merge_type', 'sat_type', 'mstar',
                 'infall_mstar', 'snap_infall', 'host_mvir', 'host_mstar',
                 'host_rvir', 'host_vvir')


def build_event_catalogue(verbose=True, ctx=None):
    """Walk every snapshot of every model file and collect the destruction events."""
    files = MODEL_FILES if ctx is None else ctx['files']
    snaps = SNAPS if ctx is None else ctx['snaps']
    conv = MASS_CONVERT if ctx is None else ctx['mass_convert']
    cols = {k: [] for k in _EVENT_FIELDS}
    n_no_host = 0

    for fp in files:
        with h5.File(fp, 'r') as f:
            for snap in snaps:
                key = f'Snap_{snap}'
                if key not in f:
                    continue
                g = f[key]
                mt = g['mergeType'][:]
                ev = mt != 0
                if not ev.any():
                    continue

                gi  = g['GalaxyIndex'][:]
                cgi = g['CentralGalaxyIndex'][:]
                sm  = g['StellarMass'][:]
                mv  = g['Mvir'][:]

                # central lookup: GalaxyIndex is unique within a file
                order = np.argsort(gi)
                pos = np.searchsorted(gi, cgi[ev], sorter=order)
                pos = np.clip(pos, 0, len(gi) - 1)
                host = order[pos]
                good = gi[host] == cgi[ev]
                n_no_host += int((~good).sum())

                rv, vv = g['Rvir'][:], g['Vvir'][:]
                host_mvir  = np.where(good, mv[host], np.nan)
                host_mstar = np.where(good, sm[host], np.nan)
                host_rvir  = np.where(good, rv[host], np.nan)
                host_vvir  = np.where(good, vv[host], np.nan)

                cols['snap_dest'].append(np.full(ev.sum(), snap, dtype=np.int16))
                cols['merge_type'].append(mt[ev].astype(np.int8))
                cols['sat_type'].append(g['Type'][:][ev].astype(np.int8))
                cols['mstar'].append(sm[ev] * conv)
                cols['infall_mstar'].append(g['infallStellarMass'][:][ev] * conv)
                cols['snap_infall'].append(g['TimeOfInfall'][:][ev])
                cols['host_mvir'].append(host_mvir * conv)
                cols['host_mstar'].append(host_mstar * conv)
                cols['host_rvir'].append(host_rvir)
                cols['host_vvir'].append(host_vvir)

    ev = {k: np.concatenate(v) for k, v in cols.items()}
    if verbose:
        print(f'  {len(ev["snap_dest"]):,} destruction events across '
              f'{len(snaps)} snapshots and {len(files)} files')
        if n_no_host:
            print(f'  warning: {n_no_host:,} events whose central was not found')
    return ev


def load_events(refresh=False, verbose=True, directory=None):
    """The event catalogue, from the npz cache when it is newer than the model files."""
    if directory is None or os.path.normpath(directory) == os.path.normpath(MODEL_DIR):
        directory, ctx, cache = MODEL_DIR, None, CACHE_FILE
        files = MODEL_FILES
    else:
        ctx = run_context(directory)
        if ctx is None:
            return None
        files = ctx['files']
        tag = os.path.basename(os.path.normpath(directory))
        cache = os.path.join(OUTPUT_DIR, f'satellite_events_v2_{tag}.npz')

    if not refresh and os.path.exists(cache):
        if os.path.getmtime(cache) > max(os.path.getmtime(f) for f in files):
            if verbose:
                print(f'Reading event catalogue from {cache}')
            z = np.load(cache)
            return _derive({k: z[k] for k in _EVENT_FIELDS})

    if verbose:
        print(f'Building event catalogue from {directory}')
    ev = build_event_catalogue(verbose=verbose, ctx=ctx)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.savez_compressed(cache, **ev)
    if verbose:
        print(f'  Cached: {cache}')
    return _derive(ev)


def _derive(ev):
    """Add the columns that follow from the raw ones: destination and times."""
    ev = dict(ev)
    ev['to_ics'] = ev['merge_type'] == 4
    ev['to_bcg'] = (ev['merge_type'] == 1) | (ev['merge_type'] == 2)

    ev['t_dest'] = snap_to_age(ev['snap_dest'])
    ev['z_dest'] = REDSHIFTS[ev['snap_dest']]

    # TimeOfInfall is only trustworthy for a record written with Type != 0.
    # save_gals_hdf5.c zero-fills every infall field (infallMvir, infallVmax,
    # infallStellarMass, TimeOfInfall) when a galaxy is written as Type 0, so a
    # central that is absorbed and destroyed inside a single timestep -- before
    # its Type is reassigned -- comes out with TimeOfInfall == 0.  Taken at face
    # value that reads as "fell in at the first snapshot" and hands the event a
    # residence time equal to the entire age of the universe at its destruction.
    # Those rows get a NaN residence time instead.  They are numerous but light:
    # see report_catalogue() for the event and mass fractions excluded.
    ev['infall_known'] = (ev['sat_type'] != 0) & (ev['snap_infall'] >= 0)
    known = ev['infall_known']

    safe = np.where(known, ev['snap_infall'], 0)
    ev['t_infall'] = np.where(known, snap_to_age(safe), np.nan)
    ev['z_infall'] = np.where(known, REDSHIFTS[safe.astype(int)], np.nan)
    ev['t_res'] = ev['t_dest'] - ev['t_infall']

    ev['log_mstar'] = np.log10(np.where(ev['mstar'] > 0, ev['mstar'], np.nan))
    ev['log_host_mvir'] = np.log10(np.where(ev['host_mvir'] > 0, ev['host_mvir'], np.nan))
    ev['mass_ratio'] = np.where(ev['host_mstar'] > 0, ev['mstar'] / ev['host_mstar'], np.nan)

    # Host dynamical time at the moment of destruction.  Rvir is physical
    # Mpc/h and Vvir is km/s (verified against sqrt(GM/R) for a 1.6e14 halo),
    # so t_dyn = (Rvir/h)/Vvir in Mpc/(km/s), and 1 Mpc/(km/s) = 977.8 Gyr.
    # This is the natural clock to measure a disruption timescale against: if
    # satellites were destroyed on a crossing time, t_res/t_dyn would be ~1
    # independent of mass, redshift and host.
    with np.errstate(divide='ignore', invalid='ignore'):
        ev['t_dyn'] = np.where(ev['host_vvir'] > 0,
                               (ev['host_rvir'] / HUBBLE_H) / ev['host_vvir'] * 977.8,
                               np.nan)
    ev['t_res_over_tdyn'] = ev['t_res'] / ev['t_dyn']
    ev['log_infall_mstar'] = np.log10(
        np.where(ev['infall_mstar'] > 0, ev['infall_mstar'], np.nan))
    return ev


def report_catalogue(ev):
    """Print the budget the four figures are all views of."""
    ics, bcg = ev['to_ics'], ev['to_bcg']
    m = ev['mstar']
    tot = m[ics].sum() + m[bcg].sum()

    print('Run:', MODEL_DIR)
    print(f'  box {HDR["box_size"]:g} Mpc/h, {HDR["volume_fraction"]*100:.0f} per cent processed '
          f'-> {VOLUME:.4g} Mpc^3;  m_p = {HDR["particle_mass"]:.4g} x 1e10 Msun/h')
    print(f'  MergerTimeFactor = {HDR["merger_time_factor"]:g}, '
          f'ThreshMajorMerger = {HDR["thresh_major_merger"]:g}, '
          f'ThresholdSatDisruption = {HDR["thresh_sat_disruption"]:g}')
    print(f'  z = 0 halo statistics taken at Snap_{Z0_SNAP} (z = {REDSHIFTS[Z0_SNAP]:.4f}), '
          f'not Snap_{LAST_SNAP} -- see Z0_SNAP')
    print()
    print('Accreted stellar mass, by destination:')
    for name, sel in ((DEST_ICS, ics), (DEST_BCG, bcg)):
        print(f'  {name:4s}  {sel.sum():8,d} events   '
              f'{m[sel].sum():.4g} Msun   ({m[sel].sum()/tot*100:5.1f} per cent of accreted)   '
              f'median log M* = {np.nanmedian(ev["log_mstar"][sel]):.2f}')
    minor, major = ev['merge_type'] == 1, ev['merge_type'] == 2
    print(f'        of the BCG channel: {m[minor].sum()/m[bcg].sum()*100:.1f} per cent minor, '
          f'{m[major].sum()/m[bcg].sum()*100:.1f} per cent major by mass')
    print()
    known = ev['infall_known']
    print('Infall-to-destruction time:')
    for name, sel in ((DEST_ICS, ics), (DEST_BCG, bcg)):
        drop = sel & ~known
        print(f'  {name:4s}  no usable TimeOfInfall for {drop.sum():,} events '
              f'({drop.sum()/sel.sum()*100:.1f} per cent of events, '
              f'{m[drop].sum()/m[sel].sum()*100:.1f} per cent of the mass) -- excluded')
    for name, sel in ((DEST_ICS, ics), (DEST_BCG, bcg)):
        s_ = sel & known
        t, w = ev['t_res'][s_], m[s_]
        o = np.argsort(t)
        mw_med = t[o][np.searchsorted(np.cumsum(w[o]) / w.sum(), 0.5)]
        print(f'  {name:4s}  per-event median {np.median(t):5.2f} Gyr '
              f'(16--84 {np.percentile(t,16):.2f}--{np.percentile(t,84):.2f});   '
              f'mass-weighted median {mw_med:5.2f} Gyr, mean {np.average(t, weights=w):.2f} Gyr')
    print()
    for name, sel in ((DEST_ICS, ics), (DEST_BCG, bcg)):
        t50 = _half_mass_time(ev['t_dest'][sel], m[sel])
        print(f'  {name} half of the accreted mass is in place by t = {t50:.2f} Gyr '
              f'(z = {_z_at_age(t50):.2f})')


def report_by_satellite_mass(ev):
    """Routing and timing as a function of what kind of satellite was destroyed."""
    m, ics, known = ev['mstar'], ev['to_ics'], ev['infall_known']
    edges = [6.0, 8.0, 9.0, 10.0, 10.5, 11.0, 12.0]
    print()
    print('By satellite stellar mass at destruction:')
    print('  log M*        events    M* accreted   to ICS   median t_res [Gyr]')
    print('                                        (mass)    ICS     BCG')
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = (ev['log_mstar'] >= lo) & (ev['log_mstar'] < hi)
        if not b.any():
            continue
        f = m[b & ics].sum() / m[b].sum()
        t_i = ev['t_res'][b & ics & known]
        t_b = ev['t_res'][b & ~ics & known]
        med = lambda a: f'{np.median(a):5.2f}' if a.size >= MIN_COUNT else '    -'
        print(f'  {lo:4.1f}--{hi:4.1f}  {b.sum():8,d}   {m[b].sum():.3e}   '
              f'{f*100:5.1f}%   {med(t_i)}   {med(t_b)}')
    print()


def report_by_host_mass(ev):
    """The same accounting per host halo, which is the axis the ICL literature uses."""
    m, ics, known = ev['mstar'], ev['to_ics'], ev['infall_known']
    print('By host halo mass at the time of destruction:')
    print('  log Mvir      events    M* accreted   to ICS    t_res [Gyr]      half-mass epoch [Gyr]')
    print('                                        (mass)    ICS     BCG      ICS     BCG')
    for lo, hi, _ in HOST_BINS:
        b = (ev['log_host_mvir'] >= lo) & (ev['log_host_mvir'] < hi)
        if not b.any():
            continue
        f = m[b & ics].sum() / m[b].sum()
        t_i, t_b = ev['t_res'][b & ics & known], ev['t_res'][b & ~ics & known]
        med = lambda a: f'{np.median(a):5.2f}' if a.size >= MIN_COUNT else '    -'
        h = lambda sel: (f'{_half_mass_time(ev["t_dest"][sel], m[sel]):5.2f}'
                         if sel.sum() >= MIN_COUNT else '    -')
        print(f'  {lo:4.1f}--{hi:4.1f}  {b.sum():8,d}   {m[b].sum():.3e}   '
              f'{f*100:5.1f}%   {med(t_i)}   {med(t_b)}    {h(b & ics)}   {h(b & ~ics)}')
    print()


def report_assembly(ev):
    """When each reservoir was laid down, and how fast it is still growing."""
    m = ev['mstar']
    print('Assembly of each reservoir (all hosts):')
    for name, sel in ((DEST_ICS, ev['to_ics']), (DEST_BCG, ev['to_bcg'])):
        t, w = ev['t_dest'][sel], m[sel]
        o = np.argsort(t)
        c = np.cumsum(w[o]) / w.sum()
        q = {p: t[o][np.searchsorted(c, p)] for p in (0.1, 0.25, 0.5, 0.75, 0.9)}
        print(f'  {name}: mass-weighted mean epoch {np.average(t, weights=w):.2f} Gyr '
              f'(lookback {AGE_NOW - np.average(t, weights=w):.2f} Gyr)')
        print('     ' + '  '.join(
            f'{int(p*100)} per cent by {v:5.2f} Gyr (z={_z_at_age(v):.2f})'
            for p, v in list(q.items())[::2]))
        last2 = t > AGE_NOW - 2.0
        print(f'     {w[last2].sum()/w.sum()*100:.1f} per cent of it deposited in the '
              f'last 2 Gyr;  {w[t < 3.0].sum()/w.sum()*100:.1f} per cent before t = 3 Gyr')
    print()


def _half_mass_time(t, m):
    o = np.argsort(t)
    c = np.cumsum(m[o]) / m.sum()
    return float(t[o][np.searchsorted(c, 0.5)])


def _z_at_age(t):
    """Invert the snapshot age table; adequate for annotating a figure."""
    return float(np.interp(t, AGE_AT_SNAP, REDSHIFTS))


# ========================== FIGURE 1: WHICH SATELLITES ==========================

def plot_1_satellite_population(ev):
    """
    Which satellites end up in the ICS and which end up in the BCG.

    Top row: the raw distributions -- satellite stellar mass at destruction and
    the halo mass of the host it was destroyed into, for each destination,
    weighted by the mass each event delivers (the unweighted curves are drawn
    behind in outline, since the two differ a great deal: the ICS is fed by many
    small events and a few large ones).
    Bottom row: the same information as a routing fraction, the share of
    accreted stellar mass that goes to the ICS rather than the BCG, against
    those two axes.  This is the panel that says which satellites the
    dynamical-friction clock is still running for.
    """
    print('Figure 1: the satellite population, split by destination')
    ics, bcg, m = ev['to_ics'], ev['to_bcg'], ev['mstar']

    fig, axes = panel_grid(2, 2)

    for ax, x, bins, xlabel in (
            (axes[0, 0], ev['log_mstar'], MSTAR_BINS,
             r'$\log_{10}(M_\star/{\rm M}_\odot)$ of the satellite'),
            (axes[0, 1], ev['log_host_mvir'], MVIR_BINS,
             r'$\log_{10}(M_{\rm vir}/{\rm M}_\odot)$ of the host')):
        for dest, sel in ((DEST_ICS, ics), (DEST_BCG, bcg)):
            ok = sel & np.isfinite(x)
            ax.hist(x[ok], bins=bins, weights=m[ok] / m[ok].sum(),
                    color=COLOUR[dest], alpha=0.55, zorder=Z_LINE,
                    label=DEST_LABEL[dest])
            ax.hist(x[ok], bins=bins, weights=np.full(ok.sum(), 1.0 / ok.sum()),
                    histtype='step', color=COLOUR[dest], lw=1.5, ls='--',
                    zorder=Z_LINE + 1,
                    label='   ' + dest + ', per event')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('fraction per bin')
        ax.set_xlim(bins[0], bins[-1])
        legend(ax, loc='upper left')

    for ax, x, bins, xlabel in (
            (axes[1, 0], ev['log_mstar'], MSTAR_BINS,
             r'$\log_{10}(M_\star/{\rm M}_\odot)$ of the satellite'),
            (axes[1, 1], ev['log_host_mvir'], MVIR_BINS,
             r'$\log_{10}(M_{\rm vir}/{\rm M}_\odot)$ of the host')):
        ok = np.isfinite(x)
        c, frac = binned_mass_fraction(x[ok], m[ok], ics[ok], bins)
        good = np.isfinite(frac)
        ax.plot(c[good], frac[good], color=COLOUR[DEST_ICS], lw=2.8, zorder=Z_LINE,
                label='mass-weighted')
        # Within a 0.25 dex satellite-mass bin the two weightings are almost
        # degenerate, so the dashed curve only separates in the host-mass panel.
        c, fracn = binned_mass_fraction(x[ok], np.ones(ok.sum()), ics[ok], bins)
        goodn = np.isfinite(fracn)
        ax.plot(c[goodn], fracn[goodn], color=COLOUR[DEST_ICS], lw=1.5, ls='--',
                zorder=Z_LINE, label='per event')
        ax.axhline(0.5, color='0.55', lw=1.0, ls=':', zorder=1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r'$M_\star$ fraction routed to ICS')
        ax.set_xlim(bins[0], bins[-1])
        ax.set_ylim(0, 1)
        legend(ax, loc='lower right')

    fig.tight_layout()
    save_figure(fig, 'fig1_satellite_population')


# ========================== FIGURE 2: TIMESCALES ==========================

def plot_2_satellite_timescales(ev):
    """
    How long a satellite survives between infall and destruction, by destination.

    The residence time is t(destruction) - t(TimeOfInfall), both read off the
    snapshot table: the *realised* survival time that the dynamical-friction
    clock and the disruption criterion together deliver, not the MergTime
    estimated at infall.  A satellite reaches the ICS only while that clock is
    still running, so the ICS is by construction the short-lived channel.

    Per-event and mass-weighted statistics are both drawn throughout, because
    for the BCG channel they disagree by an order of magnitude: it is fed by a
    very large number of tiny galaxies that merge almost immediately and a small
    number of massive ones that take several Gyr, and only the latter carry
    mass.  Events whose record was written as Type 0 have no usable
    TimeOfInfall (see _derive) and are excluded from every panel here.
    """
    print('Figure 2: infall-to-destruction timescales')
    ics, bcg, m = ev['to_ics'], ev['to_bcg'], ev['mstar']
    known = ev['infall_known']

    fig, axes = panel_grid(2, 2)

    # (a) the distribution itself, both weightings
    ax = axes[0, 0]
    tbins = np.arange(0.0, 9.01, 0.3)
    for dest, sel in ((DEST_ICS, ics), (DEST_BCG, bcg)):
        s_ = sel & known
        t, w = ev['t_res'][s_], m[s_]
        ax.hist(t, bins=tbins, weights=w / w.sum(), color=COLOUR[dest],
                alpha=0.55, zorder=Z_BAND, label=DEST_LABEL[dest] + ', mass-weighted')
        ax.hist(t, bins=tbins, weights=np.full(t.size, 1.0 / t.size),
                histtype='step', color=COLOUR[dest], lw=1.5, ls='--',
                zorder=Z_LINE, label='   ' + dest + ', per event')
        o = np.argsort(t)
        mw = t[o][np.searchsorted(np.cumsum(w[o]) / w.sum(), 0.5)]
        ax.axvline(mw, color=COLOUR[dest], lw=1.6, zorder=Z_LINE + 2)
        ax.annotate(f'{mw:.2f} Gyr', (mw, 0.62), xycoords=('data', 'axes fraction'),
                    textcoords='offset points', xytext=(4, 0), va='center',
                    fontsize=7, color=COLOUR[dest])
    ax.set_xlabel(r'$t_{\rm destruction} - t_{\rm infall}$ [Gyr]')
    ax.set_ylabel(r'fraction per bin')
    ax.set_xlim(0, tbins[-1])
    legend(ax, loc='upper right')  # the distributions pile up against t = 0

    # (b) and (c): against the satellite, and against its host.  Solid line and
    # band are the per-event median and 16--84 spread; the dashed line is the
    # mass-weighted median in the same bin.
    for ax, x, bins, xlabel, loc in (
            (axes[0, 1], ev['log_mstar'], MSTAR_BINS,
             r'$\log_{10}(M_\star/{\rm M}_\odot)$ of the satellite', 'upper left'),
            (axes[1, 0], ev['log_host_mvir'], MVIR_BINS,
             r'$\log_{10}(M_{\rm vir}/{\rm M}_\odot)$ of the host', 'upper left')):
        for dest, sel in ((DEST_ICS, ics), (DEST_BCG, bcg)):
            s_ = sel & known
            plot_median_band(ax, x[s_], ev['t_res'][s_], bins, color=COLOUR[dest],
                             label=DEST_LABEL[dest] + ', per event', min_count=50)
            c, mw = binned_weighted_median(x[s_], ev['t_res'][s_], m[s_], bins,
                                           min_count=50)
            ok = np.isfinite(mw)
            ax.plot(c[ok], mw[ok], color=COLOUR[dest], lw=1.8, ls='--',
                    zorder=Z_LINE + 1, label='   ' + dest + ', mass-weighted')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r'$t_{\rm destruction} - t_{\rm infall}$ [Gyr]')
        ax.set_xlim(bins[0], bins[-1])
        ax.set_ylim(0, None)
        legend(ax, loc=loc)

    # (d) against when the satellite fell in.  The merging-time estimate scales
    # with the host dynamical time, so a later infall should buy a longer clock;
    # the hard ceiling is that a satellite cannot outlive the run.
    ax = axes[1, 1]
    abins = np.arange(0.0, AGE_NOW + 0.01, 0.5)
    for dest, sel in ((DEST_ICS, ics), (DEST_BCG, bcg)):
        s_ = sel & known
        plot_median_band(ax, ev['t_infall'][s_], ev['t_res'][s_], abins,
                         color=COLOUR[dest], label=DEST_LABEL[dest] + ', per event',
                         min_count=50)
        c, mw = binned_weighted_median(ev['t_infall'][s_], ev['t_res'][s_], m[s_],
                                       abins, min_count=50)
        ok = np.isfinite(mw)
        ax.plot(c[ok], mw[ok], color=COLOUR[dest], lw=1.8, ls='--',
                zorder=Z_LINE + 1, label='   ' + dest + ', mass-weighted')
    ax.plot(abins, AGE_NOW - abins, color='0.5', lw=1.2, ls=':', zorder=1,
            label=r'time remaining to $z=0$')
    ax.set_xlabel(r'$t_{\rm infall}$ [Gyr]')
    ax.set_ylabel(r'$t_{\rm destruction} - t_{\rm infall}$ [Gyr]')
    time_axis(ax)
    ax.set_ylim(0, None)
    legend(ax, loc='upper left')
    _z_axis(ax)

    fig.tight_layout()
    save_figure(fig, 'fig2_satellite_timescales')


# ========================== FIGURES 3 AND 4: ASSEMBLY ==========================

def _assembly_rate(t_dest, mass, snaps=None):
    """
    Mass deposited per snapshot interval, as a density rate.

    Returns (t_mid, rate) with rate in Msun/yr/Mpc^3, the same units a cosmic
    star-formation history is drawn in, so the two can be read against
    each other.
    """
    edges = 0.5 * (AGE_AT_SNAP[:-1] + AGE_AT_SNAP[1:])
    edges = np.concatenate(([AGE_AT_SNAP[0] - (edges[0] - AGE_AT_SNAP[0])],
                            edges,
                            [AGE_AT_SNAP[-1] + (AGE_AT_SNAP[-1] - edges[-1])]))
    summed, _ = np.histogram(t_dest, bins=edges, weights=mass)
    dt_yr = np.diff(edges) * 1e9
    return AGE_AT_SNAP, summed / dt_yr / VOLUME


def _plot_assembly(ev, sel, dest, filename, what):
    """
    When a reservoir is built out of destroyed satellites, and out of how much.

    Left: the deposition rate density against cosmic time, in total and split by
    the halo mass of the host receiving it, so the cluster-scale history can be
    separated from the field.
    Middle: the cumulative mass, absolute on the left axis and as a fraction on
    the right, with the half-mass epoch marked.
    Right: the mass-weighted mean deposition redshift against host halo mass --
    one number per halo mass, saying how early that reservoir was laid down.
    """
    m = ev['mstar']
    colour = COLOUR[dest]

    fig, axes = panel_grid(1, 3)

    # --- rate history ---
    ax = axes[0]
    t, rate = _assembly_rate(ev['t_dest'][sel], m[sel])
    ok = rate > 0
    ax.plot(t[ok], rate[ok], color=colour, lw=3.0, zorder=Z_LINE, label='all hosts')
    for (lo, hi, lbl), c in zip(HOST_BINS, HOST_BIN_COLOURS):
        s = sel & (ev['log_host_mvir'] >= lo) & (ev['log_host_mvir'] < hi)
        if s.sum() < MIN_COUNT:
            continue
        t, r = _assembly_rate(ev['t_dest'][s], m[s])
        ok = r > 0
        ax.plot(t[ok], r[ok], color=c, lw=1.8, zorder=Z_LINE - 1,
                label=r'$\log M_{\rm vir} = $ ' + lbl)
    ax.set_yscale('log')
    ax.set_xlabel('cosmic time [Gyr]')
    ax.set_ylabel(what + r' rate $[{\rm M}_\odot\,{\rm yr}^{-1}\,{\rm Mpc}^{-3}]$')
    time_axis(ax)
    legend(ax, loc='lower left')
    _z_axis(ax)

    # --- cumulative ---
    # Read right to left, with time running backwards: the curve rises from
    # nothing at early times to the final reservoir at the present day.
    ax = axes[1]
    o = np.argsort(ev['t_dest'][sel])
    t_s, m_s = ev['t_dest'][sel][o], m[sel][o]
    total = m_s.sum() / VOLUME
    # Fix the y unit to a round power of ten rather than let matplotlib park an
    # offset "x10^n" in the top-left corner, where the redshift axis already is.
    exp = int(np.floor(np.log10(total)))
    ax.plot(t_s, np.cumsum(m_s) / VOLUME / 10.0 ** exp, color=colour, lw=3.0,
            zorder=Z_LINE)
    ax.set_xlabel('cosmic time [Gyr]')
    ax.set_ylabel(what + rf' from satellites $[10^{{{exp}}}\,{{\rm M}}_\odot\,{{\rm Mpc}}^{{-3}}]$')
    time_axis(ax)
    ax.set_ylim(0, None)

    frac = ax.twinx()
    frac.set_ylim(0, 1)
    frac.set_ylabel('fraction of the final mass')
    cumfrac = np.cumsum(m_s) / m_s.sum()
    for q, ls in ((0.5, '--'), (0.9, ':')):
        tq = t_s[np.searchsorted(cumfrac, q)]
        frac.axvline(tq, color=colour, ls=ls, lw=1.4, zorder=1)
        # time runs backwards, so a marker at high t sits near the left spine
        # and its label has to be placed on the inside
        left_edge = tq > 0.75 * AGE_NOW
        frac.annotate(rf'${q:.0%}$'.replace('%', r'\%') + f': {tq:.1f} Gyr, ' +
                      rf'$z={_z_at_age(tq):.2f}$',
                      (tq, q), textcoords='offset points',
                      xytext=(6 if left_edge else -6, 0),
                      ha='left' if left_edge else 'right', va='center',
                      fontsize=7, color=colour)
    # top right is empty: the curve is at zero there
    ax.annotate(rf'total $= {total:.3g}\ {{\rm M}}_\odot\,{{\rm Mpc}}^{{-3}}$',
                (0.97, 0.94), xycoords='axes fraction', va='top', ha='right',
                fontsize=8, color=colour)
    _z_axis(ax)

    # --- mean deposition epoch vs host mass ---
    ax = axes[2]
    centres = 0.5 * (MVIR_BINS[:-1] + MVIR_BINS[1:])
    mean_t = np.full(len(centres), np.nan)
    half_t = np.full(len(centres), np.nan)
    x = ev['log_host_mvir']
    for i in range(len(centres)):
        s = sel & (x >= MVIR_BINS[i]) & (x < MVIR_BINS[i + 1])
        if s.sum() >= MIN_COUNT and m[s].sum() > 0:
            mean_t[i] = np.average(ev['t_dest'][s], weights=m[s])
            half_t[i] = _half_mass_time(ev['t_dest'][s], m[s])
    ok = np.isfinite(mean_t)
    ax.plot(centres[ok], mean_t[ok], 'o-', color=colour, lw=2.6, ms=5,
            zorder=Z_LINE, label='mass-weighted mean epoch')
    ax.plot(centres[ok], half_t[ok], 's--', color=colour, lw=1.6, ms=4,
            alpha=0.7, zorder=Z_LINE, label='half-mass epoch')
    ax.set_xlabel(r'$\log_{10}(M_{\rm vir}/{\rm M}_\odot)$ of the host')
    ax.set_ylabel(what + ' deposition epoch [Gyr]')
    ax.set_xlim(MVIR_BINS[0], MVIR_BINS[-1])
    ax.set_ylim(0, AGE_NOW)
    legend(ax, loc='lower left')

    right = ax.twinx()
    right.set_ylim(ax.get_ylim())
    zt = [0, 0.2, 0.5, 1, 2, 3, 5]
    right.set_yticks([cosmic_time_gyr(z) for z in zt])
    right.set_yticklabels([f'{z:g}' for z in zt])
    right.set_ylabel(r'$z$', labelpad=2)

    fig.tight_layout()
    save_figure(fig, filename)


def plot_3_ics_assembly(ev):
    """When the ICS is built out of disrupted satellites, and out of how much mass."""
    print('Figure 3: ICS assembly from disrupted satellites')
    _plot_assembly(ev, ev['to_ics'], DEST_ICS, 'fig3_ICS_assembly', 'ICS')
    _crosscheck_ics_total(ev)


def plot_4_bcg_assembly(ev):
    """When the BCG's accreted component is built, and out of how much satellite mass."""
    print('Figure 4: BCG assembly from merged satellites')
    _plot_assembly(ev, ev['to_bcg'], DEST_BCG, 'fig4_BCG_assembly', 'BCG accreted')


def _crosscheck_ics_total(ev):
    """
    Check the event sum against SAGE's own two ICS accounting routes.

    SAGE writes ICS_disrupt (stars stripped straight into this reservoir) and
    ICS_sum_mt (the mass-weighted sum of m*t at deposition, in code time), with
    the invariant ICS_disrupt + ICS_accrete == IntraClusterStars.  The summed
    mergeType == 4 events should reproduce the total, and the mean assembly
    lookback derived from ICS_sum_mt should match the catalogue's mean epoch.
    Any disagreement means the catalogue is missing events.
    """
    if not HDR['track_ics_assembly']:
        print('  TrackICSAssembly is off in this run; skipping the cross-check.')
        return

    d = _read_snap(Z0_SNAP, ['IntraClusterStars', 'ICS_disrupt', 'ICS_accrete',
                             'ICS_sum_mt'])
    if not d:
        return
    ics_tot = d['IntraClusterStars'].sum()
    disrupt = d['ICS_disrupt'].sum()
    accrete = d['ICS_accrete'].sum()
    events = ev['mstar'][ev['to_ics'] & (ev['snap_dest'] <= Z0_SNAP)].sum()

    # ICS_sum_mt is [code mass] * [code time], so the mass in the denominator
    # has to be code mass too.  Code time is Myr/h -- core_init.c builds Age as
    # time_to_present/Hubble with Hubble = H0/h -- so a lookback in Gyr is
    # code_time * UnitLength/UnitVelocity / h.
    unit_time_gyr = (HDR['unit_length_in_cm'] / HDR['unit_velocity_in_cms']
                     / _SEC_PER_GYR / HUBBLE_H)
    mean_lookback = (d['ICS_sum_mt'].sum() / (ics_tot / MASS_CONVERT)) * unit_time_gyr

    sel = ev['to_ics'] & (ev['snap_dest'] <= Z0_SNAP)
    cat_lookback = AGE_NOW - np.average(ev['t_dest'][sel], weights=ev['mstar'][sel])

    # ICS_disrupt is NOT the number to compare the event sum against: when a
    # central that already holds ICS is itself destroyed, its ICS_disrupt is
    # re-booked as ICS_accrete on the new host (model_mergers.c), so the in-situ
    # column only records the stars stripped into the reservoir that still owns
    # them.  The conserved total is ICS_disrupt + ICS_accrete == IntraClusterStars,
    # and that is what summing every mergeType == 4 event should reproduce.
    print(f'  cross-check at Snap_{Z0_SNAP}:')
    print(f'    total ICS                {ics_tot:.4g} Msun'
          f'   (in-situ {disrupt/ics_tot*100:.1f} / carried in {accrete/ics_tot*100:.1f} per cent)')
    print(f'    summed mergeType==4      {events:.4g} Msun'
          f'   -> event sum / total ICS = {events/ics_tot:.4f}')
    print(f'    mean assembly lookback   {mean_lookback:.2f} Gyr from ICS_sum_mt, '
          f'{cat_lookback:.2f} Gyr from the event catalogue')


def _read_snap(snap, props, files=None, mass_convert=None):
    """
    Read *props* at *snap* across a run's model files, mass fields converted.

    Defaults to the primary run; pass *files* and *mass_convert* from
    run_context() to read one of the comparison runs instead.
    """
    files = MODEL_FILES if files is None else files
    mass_convert = MASS_CONVERT if mass_convert is None else mass_convert
    mass_props = {'StellarMass', 'Mvir', 'IntraClusterStars', 'ICS_disrupt',
                  'ICS_accrete', 'MetalsIntraClusterStars', 'BulgeMass',
                  'infallStellarMass', 'CentralMvir'}
    chunks = {p: [] for p in props}
    for fp in files:
        with h5.File(fp, 'r') as f:
            key = f'Snap_{snap}'
            if key not in f:
                continue
            for p in props:
                if p in f[key]:
                    chunks[p].append(np.array(f[key][p]))
    out = {}
    for p in props:
        if chunks[p]:
            a = np.concatenate(chunks[p])
            out[p] = a * mass_convert if p in mass_props else a
    return out


# ========================== FIGURE 5: f_ICS vs REDSHIFT ==========================

# Host-mass selections, chosen to match the two scales the compilation splits
# into.  Note what a 100 Mpc/h box can carry here: it holds only ~16 haloes
# above 10^14 at z = 0 and essentially none past z ~ 1.2, so the cluster cloud
# is genuinely sparse and thins out to nothing well before the high-redshift
# measurements.  Read it as a handful of individual objects, not a population.
# A larger box is the only way to populate the cluster selection properly.
# Host-mass selections, chosen to match the two scales the compilation splits
# into, and drawn differently because the box supports very different
# statistics at each.
#
#   cluster  one marker per halo, styled like an observed point, because a
#            100 Mpc/h box holds only ~16 haloes above 10^14 at z = 0 and 178
#            across all snapshots.  That is a handful of individual objects,
#            not a population, and a median line would misrepresent it.
#            The cloud thins to nothing by z ~ 0.75, well short of the
#            high-redshift measurements; a larger box is the only fix.
#   group    a median line, since the selection holds thousands of haloes per
#            snapshot and individual points would swamp the figure.
FICS_SELECTIONS = (
    ('cluster', 14.0, 16.0, r'$\log_{10}M_{\rm vir} > 14$',
     'scatter', '#08519c', '#08306b'),
    ('group',   12.5, 13.5, r'$12.5 < \log_{10}M_{\rm vir} < 13.5$',
     'median',  '#f16913', '#a63603'),
)


def run_context(directory):
    """
    Everything needed to read one run: its own files, mass conversion and
    snapshot table.

    Each run carries its own header, so the mass conversion and redshift table
    are taken per directory rather than inherited from the primary run.  Here
    all three runs are the same microUchuu box and the tables match, but
    reading them per run is what keeps the comparison honest if that changes.
    """
    hdr = read_sim_header(directory)
    if hdr is None:
        return None
    return dict(
        files=hdr['files'],
        mass_convert=hdr['unit_mass_in_g'] / _MSUN_CGS / hdr['hubble_h'],
        redshifts=hdr['redshifts'],
        snaps=hdr['output_snaps'],
        last_snap=hdr['last_snap_nr'],
        merger_time_factor=hdr['merger_time_factor'],
    )


def halo_fics_by_snapshot(verbose=True, ctx=None):
    """
    Per-FOF-halo f_ICS at every snapshot.

    f_ICS = M_ICS / (M_ICS + sum of all galaxy stellar mass in the halo), with
    haloes defined by CentralGalaxyIndex exactly as SAGE groups them.  At output
    time infall_recipe has already pooled every satellite's ICS onto the FOF
    central, so summing IntraClusterStars over the group is the halo total.

    The final snapshot is skipped: consistent-trees collapses the FOF grouping
    there (see Z0_SNAP), which would corrupt every group sum.

    Returns a list of dicts with 'z', 'log_mvir' and 'fics', one per snapshot.
    """
    snaps = SNAPS if ctx is None else ctx['snaps']
    last = LAST_SNAP if ctx is None else ctx['last_snap']
    zz = REDSHIFTS if ctx is None else ctx['redshifts']
    kw = {} if ctx is None else dict(files=ctx['files'],
                                     mass_convert=ctx['mass_convert'])
    out = []
    for snap in snaps:
        if snap >= last:
            continue
        d = _read_snap(snap, ['StellarMass', 'IntraClusterStars', 'Mvir',
                              'Type', 'CentralGalaxyIndex'], **kw)
        if not d or 'CentralGalaxyIndex' not in d:
            continue
        _, idx = np.unique(d['CentralGalaxyIndex'].astype(np.int64),
                           return_inverse=True)
        n = idx.max() + 1
        stars = np.bincount(idx, weights=d['StellarMass'], minlength=n)
        ics = np.bincount(idx, weights=d['IntraClusterStars'], minlength=n)

        cen = d['Type'] == 0
        mvir = np.full(n, np.nan)
        mvir[idx[cen]] = d['Mvir'][cen]

        total = stars + ics
        keep = np.isfinite(mvir) & (mvir > 0) & (total > 0)
        out.append(dict(z=float(zz[snap]), snap=snap,
                        log_mvir=np.log10(mvir[keep]),
                        fics=(ics / total)[keep]))
    if verbose:
        print(f'  f_ICS computed for {len(out)} snapshots')
    return out


def plot_5_fics_vs_redshift(ev=None):
    """
    f_ICS against redshift, against the observed ICL-fraction compilation.

    One marker per halo, not a median line: every observed point is a single
    cluster, so the like-for-like comparison is scatter against scatter.  A
    median would compress the model into a line and hide the fact that the
    halo-to-halo spread at fixed redshift is comparable to the spread between
    published measurements.  The model cloud is subsampled to
    MAX_SCATTER_POINTS per selection with a fixed seed so the PDF stays light;
    the percentiles printed at run time are computed on the full sample.

    The model applies no surface-brightness cut, counts intracluster stars all
    the way to the virial radius, and includes satellite light in the
    denominator; most of the measurements do none of those things.  The model
    cloud is therefore expected to sit above the points, and the comparison is
    about the redshift trend and the spread rather than the normalisation.
    """
    print('Figure 5: f_ICS vs redshift, against the observed compilation')
    snaps = halo_fics_by_snapshot()
    rng = np.random.default_rng(SCATTER_SEED)

    fig, ax = plt.subplots(figsize=(PANEL_W * 1.6, PANEL_H * 1.25))

    for scale, lo, hi, mlabel, style, colour, obs_colour in FICS_SELECTIONS:
        zs, fs, med_z, med_f = [], [], [], []
        for s_ in snaps:
            w = (s_['log_mvir'] >= lo) & (s_['log_mvir'] < hi)
            if not w.any():
                continue
            zs.append(np.full(int(w.sum()), s_['z']))
            fs.append(s_['fics'][w])
            if w.sum() >= MIN_COUNT:
                med_z.append(s_['z'])
                med_f.append(np.median(s_['fics'][w]))
        if not zs:
            print(f'  no haloes in the {scale} selection')
            continue
        zs, fs = np.concatenate(zs), np.concatenate(fs)

        near = zs < 0.3
        p16, p50, p84 = np.percentile(fs[near], (16, 50, 84))
        print(f'  {scale:8s} {zs.size:,} haloes over {len(snaps)} snapshots;  '
              f'at z < 0.3: f_ICS = {p50:.3f} (16--84: {p16:.3f}--{p84:.3f}), '
              f'{near.sum():,} haloes')

        if style == 'median':
            o = np.argsort(med_z)
            ax.plot(np.asarray(med_z)[o], np.asarray(med_f)[o], '-',
                    color=colour, lw=3.0, zorder=Z_LINE,
                    label=f'SAGE26 {scale} median, ' + mlabel)
        else:
            # Snapshots are discrete, so the raw cloud is a set of vertical
            # stripes.  Spread each snapshot's haloes across the redshift
            # interval that snapshot stands for.  This moves markers along the
            # z axis only; no f_ICS value is altered.
            width = np.interp(zs, _SNAP_Z_SORTED, _SNAP_DZ_SORTED)
            zs_plot = zs + rng.uniform(-0.4, 0.4, zs.size) * width
            show = np.arange(zs.size)
            if zs.size > MAX_SCATTER_POINTS:
                show = rng.choice(zs.size, MAX_SCATTER_POINTS, replace=False)
            # Same shape and size as an observed point, but solid-filled: fill
            # is what separates model from measurement throughout this figure.
            ax.plot(zs_plot[show], fs[show], 'o', ms=OBS_MS, mfc=colour,
                    mec='white', mew=0.6, alpha=0.85, ls='none', zorder=Z_LINE,
                    label=f'SAGE26 {scale} haloes, ' + mlabel)

        # Observations take the colour of the scale they belong to, so a reader
        # can tell at a glance which model curve a point should be read against.
        for i, o_ in enumerate(load_icl_fraction_observations(scale)):
            ax.plot(o_['z'], o_['f'], _OBS_MARKERS[i % len(_OBS_MARKERS)],
                    ms=OBS_MS, mfc='white', mew=1.3, mec=obs_colour,
                    color=obs_colour, ls='none', zorder=Z_OBS,
                    label=f"{o_['label']} ({scale})")

    ax.set_xlabel(r'redshift $z$')
    ax.set_ylabel(r'$f_{\rm ICS} = M_{\rm ICS}/M_{\star,\rm halo}$')
    ax.set_xlim(-0.05, 2.2)
    ax.set_ylim(0, 1.0)
    legend(ax, loc='upper left', ncol=2, fontsize=6.5,
           columnspacing=1.0, handletextpad=0.4)
    fig.tight_layout()
    save_figure(fig, 'fig5_fICS_vs_redshift')


# ============ FIGURE 6: CLUSTER f_ICS FOR THE ROUTING EXTREMES ============
#
# Two runs that bracket the routing question, identical to the default in every
# other respect.  MergerTimeFactor scales the dynamical-friction clock, and the
# clock's sign at the moment a satellite is destroyed is the only thing that
# decides destination (core_build_model.c: MergTime > 0 -> ICS, else BCG).
#
#   alpha = 0     the clock is zero from the start, so it has always expired:
#                 every destroyed satellite merges onto the central.  Exactly
#                 0.00 per cent of accreted stellar mass reaches the ICS.
#   alpha = 1000  the clock never runs out inside a Hubble time, so essentially
#                 everything disrupts.  99.19 per cent, not 100: two paths
#                 bypass the factor entirely and always merge -- a satellite
#                 subhalo below MinNumPartSatHalo = 10 particles, for which
#                 estimate_merging_time returns -1, and a galaxy dropping
#                 straight from Type 0 to orphan, which is handed MergTime = 0
#                 in core_build_model.c.
#
# The default run is drawn as well, faintly: it is what the two extremes
# bracket, and without it the figure has no reference point.
# alpha = 0 and alpha = 1000 are the two limits; 0.25/0.5/1.0 fill the interior
# so the figure shows whether any physical value of the parameter reaches the
# observed cluster fractions.  Greens and reds mark the two extremes, the blue
# ramp is the physical range, and 2.0 is the published value.
ROUTING_RUNS = (
    (0.0,    './output/microuchuu_allBCG/',    '#4daf4a', 's'),
    (0.25,   './output/microuchuu_mtf_0.25/',  '#a6bddb', 'v'),
    (0.5,    './output/microuchuu_mtf_0.5/',   '#67a9cf', '<'),
    (1.0,    './output/microuchuu_mtf_1.0/',   '#3182bd', '>'),
    (2.0,    MODEL_DIR,                        '#08519c', 'o'),
    (1000.0, './output/microuchuu_allICS/',    '#e41a1c', '^'),
)
ROUTING_FIDUCIAL = 2.0


def report_routing_budget():
    """
    Where the accreted stellar mass goes in each run, and whether the total is
    even conserved.

    The three columns that matter are the last three.  Re-routing satellites
    ought to move mass between M_ICS and the galaxies without changing their
    sum, and it does not: the disruption gate tests
    currentMvir / (M* + ColdGas), so mass kept on a galaxy instead of being
    dumped into the ICS raises that galaxy's own baryon count, trips the gate
    earlier, and changes when satellites die.  f_ICS therefore responds to
    alpha through the denominator as well as the numerator, and the runs are
    not a clean re-routing of a fixed budget.
    """
    print('Routing budget and mass closure, clusters above 10^14 at '
          f'Snap_{Z0_SNAP}:')
    print('  alpha      N      M_ICS      M*,gal      total     f_ICS')
    for alpha, directory, _, _ in ROUTING_RUNS:
        ctx = run_context(directory)
        if ctx is None:
            print(f'  {alpha:<7g} {directory} missing')
            continue
        d = _read_snap(Z0_SNAP, ['StellarMass', 'IntraClusterStars', 'Mvir',
                                 'Type', 'CentralGalaxyIndex'],
                       files=ctx['files'], mass_convert=ctx['mass_convert'])
        _, idx = np.unique(d['CentralGalaxyIndex'].astype(np.int64),
                           return_inverse=True)
        n = idx.max() + 1
        stars = np.bincount(idx, weights=d['StellarMass'], minlength=n)
        ics = np.bincount(idx, weights=d['IntraClusterStars'], minlength=n)
        cen = d['Type'] == 0
        mvir = np.full(n, np.nan)
        mvir[idx[cen]] = d['Mvir'][cen]
        w = np.isfinite(mvir) & (mvir > 1e14)
        tot = ics[w].sum() + stars[w].sum()
        print(f'  {alpha:<7g} {int(w.sum()):4d}  {ics[w].sum():.4e}  '
              f'{stars[w].sum():.4e}  {tot:.4e}  {ics[w].sum()/tot:7.3f}')
    print()


def plot_6_routing_fics_vs_redshift(ev=None):
    """
    Cluster-scale f_ICS against redshift for the two routing extremes.

    Same construction as figure 5 -- one marker per halo above 10^14, styled
    like an observed point, solid-filled to separate model from measurement --
    but groups are dropped and the model content is the alpha = 0 and
    alpha = 1000 runs instead of the default alone.

    The point of the figure is that the observed cluster points do not sit at
    either extreme, so the ICL fraction is not a question of whether satellites
    disrupt but of how long the clock runs before they do.
    """
    print('Figure 6: cluster f_ICS vs redshift across the MergerTimeFactor sweep')
    report_routing_budget()
    lo, hi = 14.0, 16.0
    rng = np.random.default_rng(SCATTER_SEED)

    fig, ax = plt.subplots(figsize=(PANEL_W * 1.6, PANEL_H * 1.25))

    for alpha, directory, colour, marker in ROUTING_RUNS:
        ctx = run_context(directory)
        if ctx is None:
            print(f'  {directory} missing, skipping alpha = {alpha:g}')
            continue
        is_fid = alpha == ROUTING_FIDUCIAL
        snaps = halo_fics_by_snapshot(verbose=False, ctx=ctx)

        zs, fs, med_z, med_f = [], [], [], []
        for s_ in snaps:
            w = (s_['log_mvir'] >= lo) & (s_['log_mvir'] < hi)
            if not w.any():
                continue
            zs.append(np.full(int(w.sum()), s_['z']))
            fs.append(s_['fics'][w])
            if w.sum() >= MIN_COUNT:
                med_z.append(s_['z'])
                med_f.append(np.median(s_['fics'][w]))
        if not zs:
            print(f'  no haloes above 10^14 at alpha = {alpha:g}')
            continue
        zs, fs = np.concatenate(zs), np.concatenate(fs)

        near = zs < 0.3
        p16, p50, p84 = np.percentile(fs[near], (16, 50, 84))
        print(f'  alpha = {ctx["merger_time_factor"]:<7g} '
              f'{zs.size:4,d} haloes;  z < 0.3: f_ICS = {p50:.3f} '
              f'(16--84: {p16:.3f}--{p84:.3f})')

        width = np.interp(zs, _SNAP_Z_SORTED, _SNAP_DZ_SORTED)
        zs_plot = zs + rng.uniform(-0.4, 0.4, zs.size) * width
        show = np.arange(zs.size)
        if zs.size > MAX_SCATTER_POINTS:
            show = rng.choice(zs.size, MAX_SCATTER_POINTS, replace=False)
        ax.plot(zs_plot[show], fs[show], marker, ms=OBS_MS * 0.8, mfc=colour,
                mec='white', mew=0.4, ls='none', alpha=0.45,
                zorder=Z_LINE + (1 if is_fid else 0),
                label=rf'$\alpha = {alpha:g}$' +
                      (' (published)' if is_fid else ''))
        # Six clouds overlap heavily; a thin median guide makes the ordering in
        # alpha readable without hiding the halo-to-halo scatter underneath.
        if med_z:
            o = np.argsort(med_z)
            ax.plot(np.asarray(med_z)[o], np.asarray(med_f)[o], '-',
                    color=colour, lw=2.6 if is_fid else 1.8,
                    zorder=Z_LINE + 5 + (1 if is_fid else 0),
                    solid_capstyle='round')

    for i, o_ in enumerate(load_icl_fraction_observations('cluster')):
        ax.plot(o_['z'], o_['f'], _OBS_MARKERS[i % len(_OBS_MARKERS)],
                ms=OBS_MS, mfc='white', mew=1.3, mec='k', color='k',
                ls='none', zorder=Z_OBS, label=f"{o_['label']}")

    ax.set_xlabel(r'redshift $z$')
    ax.set_ylabel(r'$f_{\rm ICS} = M_{\rm ICS}/M_{\star,\rm halo}$')
    ax.set_xlim(-0.05, 2.2)
    ax.set_ylim(0, 1.0)
    # the top-right quadrant is empty: no cluster survives past z ~ 1.2
    legend(ax, loc='upper right', ncol=2, fontsize=6.5,
           columnspacing=1.0, handletextpad=0.4,
           title=r'SAGE26 MergerTimeFactor $\alpha$ / observed clusters')
    fig.tight_layout()
    save_figure(fig, 'fig6_fICS_routing_extremes')


# ============ FIGURE 7: BCG STELLAR MASS -- HALO MASS ============

def load_bcg_halo_observations():
    """
    Kravtsov+18 central stellar mass against halo mass.

    Three digitised files, column 0 = log10 Mhalo, column 1 = log10 Mcen.

    Two things about this dataset decide how it must be compared, and both cut
    the same way:

      * Kravtsov+18 assume a Chabrier (2003) IMF, the same scale SAGE26's
        RecycleFraction puts the model on, so NO IMF shift is applied and none
        is needed.  (An earlier version of this module listed the IMF as "not
        established"; it is Chabrier.)
      * their BCG masses are Sersic fits integrated over many hundred kpc and
        extrapolated to infinity, so they INCLUDE the intracluster light.  The
        like-for-like model quantity is therefore M*,BCG + M_ICS, not the
        central's stellar mass alone.  Comparing these points against SAGE26's
        bare central understates the model by ~0.7 dex at 10^14 and is simply
        the wrong comparison.

    Returns None if the files are absent.
    """
    mvir, mstar = [], []
    for fname in ('morphology/ETGs_Kravtsov18.dat',
                  'morphology/LTGs_Kravtsov18.dat',
                  'morphology/SatKinsAndClusters_Kravtsov18.dat'):
        path = os.path.join(OBS_DIR, fname)
        if os.path.exists(path):
            d = np.atleast_2d(np.loadtxt(path))
            mvir.append(d[:, 0])
            mstar.append(d[:, 1])
    if not mvir:
        return None
    return dict(mvir=np.concatenate(mvir), mstar=np.concatenate(mstar))


def _bcg_table(ctx):
    """Central stellar mass, ICS and Mvir per FOF halo at Z0_SNAP for one run."""
    d = _read_snap(Z0_SNAP, ['StellarMass', 'IntraClusterStars', 'Mvir',
                             'Type', 'CentralGalaxyIndex'],
                   files=ctx['files'], mass_convert=ctx['mass_convert'])
    _, idx = np.unique(d['CentralGalaxyIndex'].astype(np.int64),
                       return_inverse=True)
    n = idx.max() + 1
    ics = np.bincount(idx, weights=d['IntraClusterStars'], minlength=n)
    cen = d['Type'] == 0
    mvir = np.full(n, np.nan)
    bcg = np.zeros(n)
    mvir[idx[cen]] = d['Mvir'][cen]
    bcg[idx[cen]] = d['StellarMass'][cen]
    keep = np.isfinite(mvir) & (mvir > 0) & (bcg > 0)
    return dict(mvir=mvir[keep], bcg=bcg[keep], ics=ics[keep])


def plot_7_bcg_halo_mass(ev=None):
    """
    BCG stellar mass against halo mass across the MergerTimeFactor sweep.

    The test that matters for figure 6: lowering alpha to reach the observed
    cluster ICL fractions also moves stellar mass out of the ICS and onto the
    central, so it cannot be judged on f_ICS alone.

    Left panel is the central's own stellar mass, which is what SAGE calls the
    BCG and excludes the ICS.  Right panel adds the halo's entire ICS to the
    central.

    THE RIGHT PANEL IS THE LIKE-FOR-LIKE COMPARISON.  Kravtsov+18's masses are
    extrapolated Sersic fits that already contain the intracluster light (see
    load_bcg_halo_observations), so the bare central in the left panel is not
    the quantity they measured.  The left panel is kept because it is what
    changes with alpha and because a surface-brightness-limited measurement
    would sit between the two, but a deficit read off the left panel alone is
    an artefact of comparing different things.

    Both model and data are on a Chabrier IMF, so no shift is applied.
    """
    print('Figure 7: BCG stellar mass -- halo mass across the sweep')
    obs = load_bcg_halo_observations()

    fig, axes = panel_grid(1, 2)
    bins = np.arange(11.0, 15.01, 0.25)

    for alpha, directory, colour, marker in ROUTING_RUNS:
        ctx = run_context(directory)
        if ctx is None:
            print(f'  {directory} missing, skipping alpha = {alpha:g}')
            continue
        t = _bcg_table(ctx)
        lm = np.log10(t['mvir'])
        is_fid = alpha == ROUTING_FIDUCIAL
        lw = 3.0 if is_fid else 1.9
        label = rf'$\alpha = {alpha:g}$' + (' (published)' if is_fid else '')

        for ax, y in ((axes[0], t['bcg']), (axes[1], t['bcg'] + t['ics'])):
            c, pct = binned_median(lm, np.log10(y), bins, min_count=MIN_COUNT)
            ok = np.isfinite(pct[1])
            ax.plot(c[ok], pct[1][ok], '-', color=colour, lw=lw,
                    zorder=Z_LINE + (1 if is_fid else 0),
                    label=label if ax is axes[0] else None)

        # one number to quote: the cluster end
        w = lm >= 14.0
        if w.sum() >= 5:
            print(f'  alpha = {alpha:<7g} log M*,BCG at M_vir > 1e14: '
                  f'{np.median(np.log10(t["bcg"][w])):.2f}   '
                  f'BCG+ICS: {np.median(np.log10((t["bcg"] + t["ics"])[w])):.2f}')

    for ax, title in ((axes[0], r'central only ($M_{\star,\rm BCG}$)'),
                      (axes[1], r'central $+$ all halo ICS')):
        if obs is not None:
            ax.plot(obs['mvir'], obs['mstar'], 'o', ms=5, mfc='white',
                    mec='k', mew=1.1, ls='none', zorder=Z_OBS,
                    label='Kravtsov+18' if ax is axes[1] else None)
        ax.set_xlabel(r'$\log_{10}(M_{\rm vir}/{\rm M}_\odot)$')
        ax.set_xlim(11.0, 15.2)
        ax.set_ylim(9.0, 13.0)
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel(r'$\log_{10}(M_\star/{\rm M}_\odot)$ of the central')
    legend(axes[0], loc='upper left')
    legend(axes[1], loc='upper left')

    fig.tight_layout()
    save_figure(fig, 'fig7_BCG_halo_mass')


# ============ FIGURE 8: THE ICS / BCG TRADE-OFF ============
#
# MergerTimeFactor moves one budget between two reservoirs, so neither f_ICS
# nor the BCG mass constrains it on its own -- figures 6 and 7 pull in opposite
# directions.  These two panels are the pair that does constrain it:
#
#   left   M_ICS / M_BCG, the ratio.  Every alpha that raises the ICS lowers
#          the BCG by the same stars, so the ratio moves roughly twice as fast
#          as either quantity and is the sharpest discriminator available.
#   right  M_BCG + M_ICS, the sum.  Invariant to the routing by construction,
#          so it tests whether the model has the right total stellar mass in
#          cluster centres independently of where the boundary is drawn.
#
# A model that matches the right panel but misses the left has a boundary
# problem; one that misses both has a mass problem.

# Indicative literature range, NOT a digitised measurement.  Drawn as a labelled
# band so the model can be read against roughly the right place; it must be
# replaced with digitised data before publication, and report_indicative_range()
# says so at run time.
_ICL_BCG_RATIO_RANGE = dict(
    lo=1.0, hi=3.0,
    source=r'Kluge+21; Montes 2022 (review)',
    note='cluster-scale ICL-to-BCG mass ratio; strongly definition-dependent')


def report_indicative_range():
    """Flag the one comparison band that is an eyeballed range, not data."""
    r = _ICL_BCG_RATIO_RANGE
    print(f'  NOTE: the M_ICS/M_BCG band ({r["lo"]}--{r["hi"]}) is an indicative '
          f'literature range')
    print(f'        [{r["source"]}], not digitised data -- {r["note"]}')


def plot_8_ics_bcg_tradeoff(ev=None):
    """
    The ICS-to-BCG ratio and the ICS-plus-BCG sum, across the alpha sweep.

    Left: M_ICS / M_BCG per halo, median per bin, against the indicative
    cluster range of 1--3.  Right: the summed central-plus-ICS stellar mass
    against Kravtsov+18, whose aperture already includes the ICL and so is the
    correct target for the sum rather than for the central alone.

    The sum is NOT invariant to alpha -- it spans 11.96 to 12.27 dex at
    M_vir > 10^14 across the sweep, a factor of two -- so it constrains the
    model, it does not merely absorb the routing.  It is simply far less
    sensitive than the split, which moves by a factor of seven in ratio over
    the same range.

    Both model and data are on a Chabrier IMF, so no shift is applied.
    """
    print('Figure 8: the ICS/BCG trade-off across the sweep')
    report_indicative_range()
    obs = load_bcg_halo_observations()

    fig, axes = panel_grid(1, 2)
    bins = np.arange(11.0, 15.01, 0.25)

    print('  alpha    median M_ICS/M_BCG (>1e14)   median log(M_BCG+M_ICS) (>1e14)')
    for alpha, directory, colour, marker in ROUTING_RUNS:
        ctx = run_context(directory)
        if ctx is None:
            print(f'  {directory} missing, skipping alpha = {alpha:g}')
            continue
        t = _bcg_table(ctx)
        lm = np.log10(t['mvir'])
        is_fid = alpha == ROUTING_FIDUCIAL
        lw = 3.0 if is_fid else 1.9
        label = rf'$\alpha = {alpha:g}$' + (' (published)' if is_fid else '')
        z = Z_LINE + (1 if is_fid else 0)

        # left: the ratio.  alpha = 0 puts no stars in the ICS at all, so the
        # ratio is identically zero and cannot be drawn on a log axis; it is
        # reported in the printout instead of being silently dropped.
        ratio = np.where(t['bcg'] > 0, t['ics'] / t['bcg'], np.nan)
        good = np.isfinite(ratio) & (ratio > 0)
        if good.sum() >= MIN_COUNT:
            c, pct = binned_median(lm[good], np.log10(ratio[good]), bins,
                                   min_count=MIN_COUNT)
            ok = np.isfinite(pct[1])
            axes[0].plot(c[ok], 10.0 ** pct[1][ok], '-', color=colour, lw=lw,
                         zorder=z, label=label)
        else:
            axes[0].plot([], [], '-', color=colour, lw=lw, label=label)

        # right: the sum
        c, pct = binned_median(lm, np.log10(t['bcg'] + t['ics']), bins,
                               min_count=MIN_COUNT)
        ok = np.isfinite(pct[1])
        axes[1].plot(c[ok], pct[1][ok], '-', color=colour, lw=lw, zorder=z)

        w = lm >= 14.0
        if w.sum() >= 5:
            r = np.median(ratio[w & good]) if (w & good).sum() >= 5 else 0.0
            print(f'  {alpha:<7g}  {r:>22.2f}   '
                  f'{np.median(np.log10((t["bcg"] + t["ics"])[w])):>28.2f}')

    r = _ICL_BCG_RATIO_RANGE
    axes[0].axhspan(r['lo'], r['hi'], color='0.4', alpha=0.16, lw=0, zorder=1,
                    label=_tex_safe(r['source']) + ' (indicative)')
    axes[0].set_yscale('log')
    axes[0].set_ylim(0.02, 30)
    axes[0].set_ylabel(r'$M_{\rm ICS}/M_{\star,\rm BCG}$')
    axes[0].set_title('the boundary: ICS-to-BCG ratio', fontsize=9)
    legend(axes[0], loc='upper left', ncol=2)

    if obs is not None:
        axes[1].plot(obs['mvir'], obs['mstar'], 'o', ms=5, mfc='white',
                     mec='k', mew=1.1, ls='none', zorder=Z_OBS,
                     label='Kravtsov+18')
    axes[1].set_ylim(9.0, 13.0)
    axes[1].set_ylabel(r'$\log_{10}[(M_{\star,\rm BCG} + M_{\rm ICS})/{\rm M}_\odot]$')
    axes[1].set_title('the total: BCG $+$ ICS', fontsize=9)
    legend(axes[1], loc='upper left')

    for ax in axes:
        ax.set_xlabel(r'$\log_{10}(M_{\rm vir}/{\rm M}_\odot)$')
        ax.set_xlim(11.0, 15.2)

    fig.tight_layout()
    save_figure(fig, 'fig8_ICS_BCG_tradeoff')


# ============ FIGURE 9: THE TIMESCALES THAT BUILD THE ICS ============
#
# External benchmarks.  Both are theoretical, from hydrodynamical simulations,
# because there is no direct observational measurement of how long a satellite
# survives before its stars become intracluster -- the observable is the light
# that is already there, not the clock that put it there.
#
# Brown et al. 2024 (Horizon-AGN, arXiv:2409.10607) is the sharpest available
# comparison because it reports the *infall* stellar masses of ICL progenitors,
# which is exactly what infallStellarMass records here:
#   -- half the stacked, not-pre-processed ICL comes from progenitors with
#      infall stellar mass within half a dex of log M* = 10.94 (+0.13 -0.07)
#   -- ~90 per cent comes from galaxies infalling above 10^9 Msun
# They also note the standard theoretical expectation, that the vast majority
# of z = 0 ICL was still bound in galaxies at z ~ 1.
HZAGN = dict(
    label='Horizon-AGN (Brown+24)',
    log_peak=10.94, log_peak_err=(0.07, 0.13), half_dex=0.5,
    log_90pc=9.0, z_mostly_in_galaxies=1.0)


def plot_9_ics_timescales(ev):
    """
    The clocks that set when and out of what the ICS is built.

    (a) The three timescales an ICS star passes through, mass-weighted: how
        long its galaxy survived after infall, how long ago it was deposited,
        and the total time since its galaxy first fell in.  These are not
        independent -- the first two sum to the third -- but plotting them
        together shows which dominates.
    (b) The residence time divided by the host's dynamical time, Rvir/Vvir at
        the moment of destruction.  This is the internal benchmark: a pure
        crossing-time process would sit at 1 with no mass dependence.
    (c) Residence time against infall stellar mass, with the Horizon-AGN band
        that dominates ICL production marked, so the timescale can be read at
        the mass that actually matters.
    (d) Cumulative ICS mass against progenitor infall stellar mass, directly
        against the two Horizon-AGN numbers.
    """
    print('Figure 9: ICS timescales and the mass that carries them')
    ics = ev['to_ics'] & ev['infall_known']
    m = ev['mstar']

    fig, axes = panel_grid(2, 2)

    # --- (a) the three clocks ---
    ax = axes[0, 0]
    clocks = (
        (ev['t_res'], r'$t_{\rm destr} - t_{\rm infall}$ (survival)', '#08519c'),
        (AGE_NOW - ev['t_dest'], r'$t_{\rm now} - t_{\rm destr}$ (since deposition)', '#e41a1c'),
        (AGE_NOW - ev['t_infall'], r'$t_{\rm now} - t_{\rm infall}$ (since infall)', '#4daf4a'),
    )
    bins = np.arange(0, 13.01, 0.4)
    for vals, label, colour in clocks:
        v, w = vals[ics], m[ics]
        ok = np.isfinite(v)
        o = np.argsort(v[ok])
        med = v[ok][o][np.searchsorted(np.cumsum(w[ok][o]) / w[ok].sum(), 0.5)]
        ax.hist(v[ok], bins=bins, weights=w[ok] / w[ok].sum(), histtype='step',
                color=colour, lw=2.0, zorder=Z_LINE,
                label=label + rf'  (med ${med:.2f}$ Gyr)')
        ax.axvline(med, color=colour, ls=':', lw=1.2, zorder=Z_BAND)
    ax.set_xlabel('timescale [Gyr]')
    ax.set_ylabel(r'fraction of ICS mass per bin')
    ax.set_xlim(0, bins[-1])
    legend(ax, loc='upper right')

    # --- (b) in units of the host dynamical time ---
    ax = axes[1, 0]
    r = ev['t_res_over_tdyn']
    ok = ics & np.isfinite(r) & (r > 0)
    rb = np.logspace(-1, 1.2, 40)
    ax.hist(r[ok], bins=rb, weights=m[ok] / m[ok].sum(), color='#08519c',
            alpha=0.6, zorder=Z_LINE, label='mass-weighted')
    ax.hist(r[ok], bins=rb, weights=np.full(int(ok.sum()), 1.0 / ok.sum()),
            histtype='step', color='0.3', lw=1.6, zorder=Z_LINE + 1,
            label='per event')
    o = np.argsort(r[ok])
    mw = r[ok][o][np.searchsorted(np.cumsum(m[ok][o]) / m[ok].sum(), 0.5)]
    ax.axvline(1.0, color='0.4', ls='--', lw=1.4, zorder=1,
               label=r'one crossing time')
    ax.axvline(mw, color='#08519c', ls='-', lw=1.8, zorder=Z_LINE + 2,
               label=rf'mass-weighted median $= {mw:.2f}\,t_{{\rm dyn}}$')
    ax.set_xscale('log')
    ax.set_xlabel(r'$(t_{\rm destr}-t_{\rm infall})\,/\,t_{\rm dyn}$,'
                  r'  $t_{\rm dyn}=R_{\rm vir}/V_{\rm vir}$')
    ax.set_ylabel(r'fraction per bin')
    legend(ax, loc='upper left')
    print(f'  residence time / host dynamical time: mass-weighted median '
          f'{mw:.2f}, per-event median {np.median(r[ok]):.2f}')

    # --- (c) residence time vs infall mass ---
    ax = axes[0, 1]
    mb = np.arange(6.0, 12.01, 0.25)
    x = ev['log_infall_mstar']
    sel = ics & np.isfinite(x)
    plot_median_band(ax, x[sel], ev['t_res'][sel], mb, color='#08519c',
                     label='per event', min_count=50)
    c, mwm = binned_weighted_median(x[sel], ev['t_res'][sel], m[sel], mb,
                                    min_count=50)
    good = np.isfinite(mwm)
    ax.plot(c[good], mwm[good], '--', color='#08519c', lw=1.8,
            zorder=Z_LINE + 1, label='mass-weighted')
    lo = HZAGN['log_peak'] - HZAGN['half_dex']
    hi = HZAGN['log_peak'] + HZAGN['half_dex']
    ax.axvspan(lo, hi, color='#e41a1c', alpha=0.13, lw=0, zorder=1,
               label=HZAGN['label'] + '\nhalf the ICL comes from here')
    ax.axvline(HZAGN['log_peak'], color='#e41a1c', ls=':', lw=1.4, zorder=2)
    ax.set_xlabel(r'$\log_{10}(M_\star/{\rm M}_\odot)$ at infall')
    ax.set_ylabel(r'$t_{\rm destr} - t_{\rm infall}$ [Gyr]')
    ax.set_xlim(mb[0], mb[-1])
    ax.set_ylim(0, None)
    legend(ax, loc='upper left')

    # --- (d) where the ICS mass comes from, vs Horizon-AGN ---
    ax = axes[1, 1]
    xi, wi = x[sel], m[sel]
    o = np.argsort(xi)
    cum = np.cumsum(wi[o]) / wi.sum()
    ax.plot(xi[o], cum, '-', color='#08519c', lw=3.0, zorder=Z_LINE,
            label='SAGE26 cumulative ICS mass')
    ax.axvspan(lo, hi, color='#e41a1c', alpha=0.13, lw=0, zorder=1,
               label=HZAGN['label'] + r': half the ICL from $\pm0.5$ dex here')
    ax.axvline(HZAGN['log_peak'], color='#e41a1c', ls=':', lw=1.4, zorder=2)
    ax.axvline(HZAGN['log_90pc'], color='#e41a1c', ls='--', lw=1.4, zorder=2,
               label=r'Horizon-AGN: $90$ per cent above $10^9$')
    f_band = float(np.interp(hi, xi[o], cum) - np.interp(lo, xi[o], cum))
    f_above9 = float(1.0 - np.interp(HZAGN['log_90pc'], xi[o], cum))
    med = float(xi[o][np.searchsorted(cum, 0.5)])
    for y, txt in ((0.42, rf'SAGE26 median $= {med:.2f}$'),
                   (0.30, rf'in band: ${f_band*100:.0f}$ per cent (HZ-AGN $50$)'),
                   (0.18, rf'above $10^9$: ${f_above9*100:.0f}$ per cent (HZ-AGN $90$)')):
        ax.annotate(txt, (0.97, y), xycoords=('axes fraction', 'data'),
                    fontsize=7.5, color='#08519c', va='center', ha='right')
    ax.axhline(0.5, color='0.6', lw=1.0, ls=':', zorder=1)
    ax.set_xlabel(r'$\log_{10}(M_\star/{\rm M}_\odot)$ at infall')
    ax.set_ylabel('cumulative fraction of ICS mass')
    ax.set_xlim(mb[0], mb[-1])
    ax.set_ylim(0, 1)
    legend(ax, loc='upper left')   # the curve and the annotations own the rest

    print(f'  ICS progenitor infall mass: median log M* = {med:.2f};  '
          f'{f_band*100:.0f} per cent from the Horizon-AGN band '
          f'({lo:.2f}--{hi:.2f}), {f_above9*100:.0f} per cent above 10^9')
    print(f'  Horizon-AGN (Brown+24) finds 50 and 90 per cent respectively')

    fig.tight_layout()
    save_figure(fig, 'fig9_ICS_timescales')


# ============ FIGURE 10: ICS BUILD-UP TIMESCALES vs MergerTimeFactor ============

def plot_10_ics_timescale_sweep(ev=None):
    """
    The two timescales of ICS build-up, across the MergerTimeFactor sweep.

    Left: how long an ICS progenitor survives between infall and disruption.
    This is the clock alpha scales directly, so the curves should march to the
    right with alpha -- and where they stop is the model's answer to "how long
    does a satellite last".

    Right: when the resulting ICS mass is actually laid down, as a fraction of
    each run's own z = 0 total.  Normalising each run to itself is deliberate:
    the runs differ enormously in how much ICS they make (alpha = 0 makes none
    at all and is omitted from both panels), and the question here is timing,
    not amount.

    Everything is mass-weighted, since the ICS budget is carried by a small
    number of massive progenitors and the per-event view says something quite
    different (see figure 2).
    """
    print('Figure 10: ICS build-up timescales across the sweep')
    fig, axes = panel_grid(1, 2)
    tbins = np.arange(0, 8.01, 0.3)

    print('  alpha   survival: mass-wtd / per-event [Gyr]   half-mass epoch')
    for alpha, directory, colour, marker in ROUTING_RUNS:
        if alpha == 0:
            continue                      # no ICS at all: nothing to time
        e = load_events(verbose=False, directory=directory)
        if e is None:
            print(f'  {directory} missing, skipping alpha = {alpha:g}')
            continue
        ics = e['to_ics']
        m = e['mstar']
        is_fid = alpha == ROUTING_FIDUCIAL
        lw = 3.0 if is_fid else 1.9
        label = rf'$\alpha = {alpha:g}$' + (' (published)' if is_fid else '')
        z = Z_LINE + (1 if is_fid else 0)

        # left: survival time, mass-weighted
        sel = ics & e['infall_known']
        t, w = e['t_res'][sel], m[sel]
        axes[0].hist(t, bins=tbins, weights=w / w.sum(), histtype='step',
                     color=colour, lw=lw, zorder=z, label=label)
        o = np.argsort(t)
        med = t[o][np.searchsorted(np.cumsum(w[o]) / w.sum(), 0.5)]

        # right: cumulative ICS mass, each run normalised to its own total
        td, wd = e['t_dest'][ics], m[ics]
        o2 = np.argsort(td)
        cum = np.cumsum(wd[o2]) / wd.sum()
        axes[1].plot(td[o2], cum, '-', color=colour, lw=lw, zorder=z, label=label)
        t50 = td[o2][np.searchsorted(cum, 0.5)]
        print(f'  {alpha:<7g} {med:>14.2f} {np.median(t):>9.2f}     '
              f'{t50:>8.2f} / z = {_z_at_age(t50):.2f}   '
              f'(ICS = {wd.sum()/ (m[e["to_ics"]].sum() + m[e["to_bcg"]].sum()) * 100:.1f} per cent of accreted)')

    axes[0].set_xlabel(r'$t_{\rm destruction} - t_{\rm infall}$ [Gyr]')
    axes[0].set_ylabel('fraction of ICS mass per bin')
    axes[0].set_xlim(0, tbins[-1])
    axes[0].set_title('how long the progenitor survives', fontsize=9)
    legend(axes[0], loc='upper right')

    axes[1].axhline(0.5, color='0.6', lw=1.0, ls=':', zorder=1)
    axes[1].set_xlabel('cosmic time [Gyr]')
    axes[1].set_ylabel('cumulative ICS mass / its own $z=0$ total')
    time_axis(axes[1])
    axes[1].set_ylim(0, 1)
    axes[1].set_title('when the ICS is laid down', fontsize=9)
    legend(axes[1], loc='lower left')   # the curve occupies the top left
    _z_axis(axes[1])

    fig.tight_layout()
    save_figure(fig, 'fig10_ICS_timescales_vs_alpha')


# ============ FIGURE 11: THE ThresholdSatDisruption SWEEP ============
#
# The test of the diagnosis in figure 10 and the routing analysis: the ICS is
# bloated not because alpha is wrong but because the disruption gate fires
# ~5x sooner than the dynamical-friction clock it is racing.  The gate is
#
#     currentMvir / (M* + ColdGas) <= ThresholdSatDisruption
#
# so LOWERING the threshold makes the test harder to pass and satellites
# survive longer.  If the diagnosis is right, lowering it should do four things
# at once, and all four are drawn here:
#   1  satellites live longer                      (panel a)
#   2  more of them survive to z = 0               (panel b)
#   3  f_ICS falls and the BCG grows               (panels b, c)
#   4  the BCG moves towards Kravtsov+18           (panel d)
# If instead only f_ICS moves and the satellite fraction does not, the gate is
# not the lever and the diagnosis is wrong.
#
# All runs are the default microUchuu with MergerTimeFactor fixed at 2.0.
THRESHOLD_RUNS = (
    (0.01, './output/microuchuu_tsd_0.01/', '#fee0d2', 'v'),
    (0.1,  './output/microuchuu_tsd_0.1/',  '#fc9272', '<'),
    (0.3,  './output/microuchuu_tsd_0.3/',  '#ef3b2c', '>'),
    (1.0,  MODEL_DIR,                       '#08519c', 'o'),
    (3.0,  './output/microuchuu_tsd_3.0/',  '#54278f', '^'),
)
THRESHOLD_FIDUCIAL = 1.0


def _cluster_budget(ctx, mvir_min=1e14):
    """BCG / satellite / ICS stellar mass summed over cluster haloes at Z0_SNAP."""
    d = _read_snap(Z0_SNAP, ['StellarMass', 'IntraClusterStars', 'Mvir',
                             'Type', 'CentralGalaxyIndex'],
                   files=ctx['files'], mass_convert=ctx['mass_convert'])
    _, idx = np.unique(d['CentralGalaxyIndex'].astype(np.int64),
                       return_inverse=True)
    n = idx.max() + 1
    cen = d['Type'] == 0
    mvir = np.full(n, np.nan)
    mvir[idx[cen]] = d['Mvir'][cen]
    big = np.isfinite(mvir) & (mvir > mvir_min)
    inbig = big[idx]
    ics = np.bincount(idx, weights=d['IntraClusterStars'], minlength=n)[big].sum()
    return dict(
        nhalo=int(big.sum()),
        bcg=float(d['StellarMass'][inbig & cen].sum()),
        sat=float(d['StellarMass'][inbig & ~cen].sum()),
        ics=float(ics))


def plot_11_threshold_sweep(ev=None):
    """
    ThresholdSatDisruption at fixed MergerTimeFactor = 2: does the gate move
    what alpha could not?
    """
    print('Figure 11: the ThresholdSatDisruption sweep at fixed alpha = 2')
    fig, axes = panel_grid(2, 2)
    tbins = np.arange(0, 8.01, 0.3)
    mb = np.arange(11.0, 15.01, 0.25)

    rows = []
    for thr, directory, colour, marker in THRESHOLD_RUNS:
        ctx = run_context(directory)
        if ctx is None:
            print(f'  {directory} missing, skipping threshold = {thr:g}')
            continue
        is_fid = thr == THRESHOLD_FIDUCIAL
        lw = 3.0 if is_fid else 1.9
        label = rf'$\theta = {thr:g}$' + (' (published)' if is_fid else '')
        z = Z_LINE + (1 if is_fid else 0)

        e = load_events(verbose=False, directory=directory)
        b = _cluster_budget(ctx)
        tot = b['bcg'] + b['sat'] + b['ics']

        # (a) survival time of ICS progenitors
        sel = e['to_ics'] & e['infall_known']
        t, w = e['t_res'][sel], e['mstar'][sel]
        if t.size:
            axes[0, 0].hist(t, bins=tbins, weights=w / w.sum(), histtype='step',
                            color=colour, lw=lw, zorder=z, label=label)
            o = np.argsort(t)
            t50 = t[o][np.searchsorted(np.cumsum(w[o]) / w.sum(), 0.5)]
        else:
            t50 = np.nan

        # (d) BCG -- halo mass
        tb = _bcg_table(ctx)
        c, pct = binned_median(np.log10(tb['mvir']), np.log10(tb['bcg']), mb,
                               min_count=MIN_COUNT)
        ok = np.isfinite(pct[1])
        axes[1, 1].plot(c[ok], pct[1][ok], '-', color=colour, lw=lw, zorder=z,
                        label=label)

        im = e['mstar'][e['to_ics']].sum()
        bm = e['mstar'][e['to_bcg']].sum()
        rows.append(dict(thr=thr, colour=colour, marker=marker, t50=t50,
                         fbcg=b['bcg'] / tot, fsat=b['sat'] / tot,
                         fics=b['ics'] / tot,
                         ratio=b['ics'] / b['bcg'] if b['bcg'] > 0 else np.nan,
                         routed=im / (im + bm), tot=tot, **b))

    if not rows:
        print('  no threshold runs found')
        plt.close(fig)
        return

    thr = np.array([r['thr'] for r in rows])
    print('  theta   routed->ICS  survival[Gyr]  f_BCG  f_sat  f_ICS  ICS/BCG  '
          'log M*_BCG_tot')
    for r in rows:
        print(f'  {r["thr"]:<7g} {r["routed"]*100:9.1f}%  {r["t50"]:11.2f}  '
              f'{r["fbcg"]:6.3f} {r["fsat"]:6.3f} {r["fics"]:6.3f} '
              f'{r["ratio"]:8.2f}  {np.log10(r["bcg"] * 1):.3f}')

    axes[0, 0].set_xlabel(r'$t_{\rm destruction} - t_{\rm infall}$ [Gyr]')
    axes[0, 0].set_ylabel('fraction of ICS mass per bin')
    axes[0, 0].set_xlim(0, tbins[-1])
    axes[0, 0].set_title('1. do satellites live longer?', fontsize=9)
    legend(axes[0, 0], loc='upper right')

    # (b) the three-way budget
    ax = axes[0, 1]
    for key, lbl, c_, mk in (('fbcg', 'BCG', '#08519c', 'o'),
                             ('fsat', 'satellites', '#4daf4a', 's'),
                             ('fics', 'ICS', '#e41a1c', '^')):
        ax.plot(thr, [r[key] for r in rows], mk + '-', color=c_, lw=2.2, ms=6,
                zorder=Z_LINE, label=lbl)
    ax.axvline(THRESHOLD_FIDUCIAL, color='0.5', ls=':', lw=1.2, zorder=1)
    ax.set_xscale('log')
    ax.set_xlabel(r'ThresholdSatDisruption $\theta$')
    ax.set_ylabel(r'share of cluster stellar mass ($M_{\rm vir}>10^{14}$)')
    ax.set_ylim(0, 1)
    ax.set_title('2--3. where the cluster mass sits', fontsize=9)
    legend(ax, loc='upper left')

    # (c) f_ICS and the ICS/BCG ratio against their comparisons
    ax = axes[1, 0]
    ax.plot(thr, [r['fics'] for r in rows], 'o-', color='#e41a1c', lw=2.4,
            ms=6, zorder=Z_LINE, label=r'$f_{\rm ICS}$ (stacked)')
    f = np.array([o['f'] for o in load_icl_fraction_observations('cluster')
                  for o in [o]], dtype=object)
    allf = np.concatenate([o['f'][o['z'] <= 0.3]
                           for o in load_icl_fraction_observations('cluster')])
    allf = allf[allf > 0]
    if allf.size:
        ax.axhspan(allf.min(), allf.max(), color='0.4', alpha=0.16, lw=0,
                   zorder=1, label=r'observed clusters ($z<0.3$)')
    ax2 = ax.twinx()
    ax2.plot(thr, [r['ratio'] for r in rows], 's--', color='#08519c', lw=2.0,
             ms=5, zorder=Z_LINE, label=r'$M_{\rm ICS}/M_{\rm BCG}$')
    ax2.axhspan(1.0, 3.0, color='#08519c', alpha=0.10, lw=0, zorder=1)
    ax2.set_ylabel(r'$M_{\rm ICS}/M_{\rm BCG}$  (indicative $1$--$3$ shaded)',
                   color='#08519c')
    ax2.set_yscale('log')
    ax.axvline(THRESHOLD_FIDUCIAL, color='0.5', ls=':', lw=1.2, zorder=1)
    ax.set_xscale('log')
    ax.set_xlabel(r'ThresholdSatDisruption $\theta$')
    ax.set_ylabel(r'$f_{\rm ICS}$', color='#e41a1c')
    ax.set_ylim(0, 0.8)
    ax.set_title(r'3. $f_{\rm ICS}$ and the ICS-to-BCG ratio', fontsize=9)
    legend(ax, loc='lower right')

    obs = load_bcg_halo_observations()
    if obs is not None:
        axes[1, 1].plot(obs['mvir'], obs['mstar'], 'o', ms=5, mfc='white',
                        mec='k', mew=1.1, ls='none', zorder=Z_OBS,
                        label='Kravtsov+18')
    axes[1, 1].set_xlabel(r'$\log_{10}(M_{\rm vir}/{\rm M}_\odot)$')
    axes[1, 1].set_ylabel(r'$\log_{10}(M_{\star,\rm BCG}/{\rm M}_\odot)$')
    axes[1, 1].set_xlim(11.0, 15.2)
    axes[1, 1].set_ylim(9.0, 13.0)
    axes[1, 1].set_title('4. does the BCG reach Kravtsov+18?', fontsize=9)
    legend(axes[1, 1], loc='upper left')

    fig.tight_layout()
    save_figure(fig, 'fig11_threshold_sweep')


# ========================== MAIN ==========================

PLOTS = {
    1: plot_1_satellite_population,
    2: plot_2_satellite_timescales,
    3: plot_3_ics_assembly,
    4: plot_4_bcg_assembly,
    5: plot_5_fics_vs_redshift,
    6: plot_6_routing_fics_vs_redshift,
    7: plot_7_bcg_halo_mass,
    8: plot_8_ics_bcg_tradeoff,
    9: plot_9_ics_timescales,
    10: plot_10_ics_timescale_sweep,
    11: plot_11_threshold_sweep,
}


def main():
    np.random.seed(2222)
    setup_style()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    refresh = '--refresh' in sys.argv

    ev = load_events(refresh=refresh)
    print()
    report_catalogue(ev)
    report_by_satellite_mass(ev)
    report_by_host_mass(ev)
    report_assembly(ev)

    nums = [int(a) for a in args] if args else sorted(PLOTS)
    for n in nums:
        if n in PLOTS:
            PLOTS[n](ev)
        else:
            print(f'Warning: figure {n} is not defined, skipping.')
        print()
    print('Done.')


if __name__ == '__main__':
    main()

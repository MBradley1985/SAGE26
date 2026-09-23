#!/usr/bin/env python
"""
SAGE26 FFB efficiency comparison: two runs differing only in FFBMaxEfficiency
(alpha_eff = 0.2 fiducial, 1.0 the theoretical maximum), in three redshift bins.
Nothing is recalibrated between them, so the 1.0 run is not a re-tuned model.

Two figures in <fid>/plots/:

    FFB_Efficiency_SHMR.pdf               m*/Mvir against halo mass, axes
                                          sized to the data, with the stellar
                                          baryon fraction on the right

    FFB_Efficiency_SHMR_ScreenshotAxes.pdf  the same curves, fixed to the
                                          x=5-300, y=7e-3-0.04 range read off
                                          the NIRISS WFSS proposal figure's SHMR
                                          panel -- an axis match only, NOT a
                                          comparison: that panel's curves are
                                          UniverseMachine (Behroozi+19), an
                                          empirical model fit to reproduce
                                          observed statistics, not a physical
                                          semi-analytic model like SAGE26.  Its
                                          smoothness and turnover shape are
                                          calibration outcomes, not predictions
                                          SAGE26 is expected to match.

    FFB_Efficiency_Panels.pdf             row 1 (wide)  number counts vs m*
                                          row 2  SHMR | SFR-m* | MZR | quiescent

Colour is the redshift bin, line style the efficiency (solid 0.2, dashed 1.0).
Self-contained: reads the SAGE HDF5 output directly, across every MPI rank's
model_*.hdf5, and imports nothing from the rest of the plotting code.

Binned by STELLAR mass, not halo mass
-------------------------------------
Galaxies are selected above a stellar mass limit (MIN_MSTAR, the survey's
spectroscopic limit) and binned in stellar mass -- including the SHMR, whose
points are placed at the median halo mass of each stellar mass bin.  This
mirrors how the relation is measured observationally: a flux-limited sample is
split by stellar mass and the host halo mass inferred from clustering.

Binning by halo mass instead would bias the result.  A stellar mass limit cuts
into a halo mass bin from the bottom, so a bin at Mvir keeps only galaxies with
m*/Mvir >= MIN_MSTAR/Mvir; that floor rises as Mvir falls and the curve turns
upward at low halo mass whatever the relation does.  Binning on the quantity
that was selected on avoids this entirely: the limit truncates the curve at
its left end and leaves every remaining point unchanged (verified -- with and
without the cut, every shared bin agrees exactly).

    python plotting/ffb_efficiency.py
    python plotting/ffb_efficiency.py --fid output/miniUchuu --full output/miniUchuu_ffb100
    python plotting/ffb_efficiency.py --figure shmr           # SHMR pair only, skip panels
    python plotting/ffb_efficiency.py --min-mstar 0           # extend below the limit
    python plotting/ffb_efficiency.py --dilute 500000         # thin a big box
    python plotting/ffb_efficiency.py --nproc 8               # parallel read

The reduced sample is cached beside each run as .ffb_efficiency_cache_*.npz
and reused until a model file is newer, so re-plotting is instant (--no-cache
to disable).
"""

import argparse, glob, os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
STYLE = os.path.join(HERE, 'kieren_cohare_palatino_sty.mplstyle')
MSUN_G = 1.989e33

FID_DIR = './output/microuchuu/'
FULL_DIR = './output/microuchuu_karl/'

Z_BINS = [(1.75, 2.4, 'black'), (2.5, 3.5, '#D62728'), (3.5, 4.5, '#1F45C0')]

MIN_MSTAR = 10**9.5                     # survey limit, Msun
MSTAR_BINS = np.arange(8.0, 13.5, 0.1)  # log10(m*/Msun); shared by every panel
MIN_COUNT = 50                         # galaxies needed in a bin
DILUTE = None                           # cap galaxies per sample (--dilute)
SEED = 2222                             # so a diluted figure is repeatable

SSFR_CUT = 1e-11                        # /yr, quiescent below this
Z_SUN = 0.02                            # for 12 + log10(O/H), solar = 9.0

FID_STYLE = dict(ls='-', lw=1.5)
FULL_STYLE = dict(ls=(0, (5, 1.8)), lw=1.0)
FID_LABEL = r'SAGE26 $\alpha_{\rm eff} = 0.2$ (fiducial)'
FULL_LABEL = r'SAGE26 $\alpha_{\rm eff} = 1.0$'

# --- SHMR figure axes ---
XLIM = (7.0, 1000.0)
XTICKS = [10, 20, 50, 100, 200, 500, 1000]
YPAD = 1.3                              # y range follows the data, padded
YTICKS = [(1e-3, r'$10^{-3}$'), (2e-3, r'$2\times10^{-3}$'),
          (3e-3, r'$3\times10^{-3}$'), (5e-3, r'$5\times10^{-3}$'),
          (7e-3, r'$7\times10^{-3}$'), (1e-2, '0.01'), (1.5e-2, '0.015'),
          (2e-2, '0.02'), (3e-2, '0.03'), (5e-2, '0.05')]
FBAR_TICKS = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5]

# Fixed axes read off the NIRISS WFSS proposal figure (plotting/Screenshot
# 2026-09-23 at 11.25.36 am.png): x runs ~5-300 in 10^10 h^-1 Msun with major
# ticks at 10/20/50/100/200; the left y-axis runs ~1e-3 to ~0.05-0.06 with
# ticks at 10^-3, 2e-3, 5e-3, 0.01, 0.05.
#
# This is an AXIS match only.  The three curves drawn in that figure are
# UniverseMachine (Behroozi, Wechsler, Hearin & Conroy 2019), not a physical
# model -- it is an empirical framework fit so that its SHMR reproduces the
# observed stellar mass function, SFR distributions, quenched fractions and
# clustering at each epoch.  It has no AGN feedback, no FFB, no cooling flows;
# its smoothness and turnover mass are properties of the fit and the (large,
# effectively noiseless) calibration data, not a physical prediction.  SAGE26
# is a forward semi-analytic model: its SHMR is an output of its recipes, on a
# box small enough to show real sample-size noise at the massive end.
# Differences from UniverseMachine's curve are not evidence SAGE26 is wrong --
# they are the two models' different physical assumptions, cosmology, and
# (for the shape) very different statistical footing.
SCREENSHOT_XLIM = (5.0, 300.0)
SCREENSHOT_XTICKS = [10, 20, 50, 100, 200]
# Tight to where the curves actually sit (data span ~0.010-0.028 for these
# runs) rather than the screenshot's own full range, which reaches far lower
# because their sample extends to much lower stellar mass than MIN_MSTAR here.
SCREENSHOT_YLIM = (7e-3, 0.04)

PROPS = ['StellarMass', 'Mvir', 'Type', 'SfrDisk', 'SfrBulge',
         'ColdGas', 'MetalsColdGas']
MASS_PROPS = {'StellarMass', 'Mvir', 'ColdGas', 'MetalsColdGas'}

MLABEL = r'$\log_{10}\ m_{*}\ [M_{\odot}]$'


# ================================ data ================================

def model_files(directory):
    """Every MPI rank's output file, in order."""
    files = sorted(glob.glob(os.path.join(directory, 'model_*.hdf5')))
    if not files:
        raise SystemExit(f'No model_*.hdf5 in {directory}')
    return files


def header(directory):
    """Cosmology, units and the snapshot table, from the run's own output."""
    with h5py.File(model_files(directory)[0], 'r') as f:
        sim, runtime = f['Header/Simulation'], f['Header/Runtime']
        return {
            'hubble_h': float(sim.attrs['hubble_h']),
            'baryon_frac': float(runtime.attrs['BaryonFrac']),
            # SAGE writes masses in units of UnitMass_in_g/h.
            'to_msun': float(runtime.attrs['UnitMass_in_g']) / MSUN_G
                       / float(sim.attrs['hubble_h']),
            'z': np.array(f['Header/snapshot_redshifts'][:]),
            'output_snaps': list(f['Header/output_snapshots'][:]),
        }


def _read_one_file(task):
    """Worker: one file, a set of snapshots, returning only kept rows.

    Opens the file once and walks its snapshots, rather than reopening it per
    snapshot -- the difference is one open per (file, chunk) instead of one per
    (file, snapshot), which dominates on a network filesystem.
    """
    path, snaps, props, to_msun, min_mstar = task
    out = {}
    with h5py.File(path, 'r') as f:
        for snap in snaps:
            grp = f.get(f'Snap_{snap}')
            if grp is None:
                continue
            # Select on two fields first, then read the rest.  Reading the
            # remaining fields in full and masking beats h5py point-selection
            # here (measured): the datasets are uncompressed and chunked, so a
            # contiguous read is bandwidth-bound while a scattered one is not.
            ms = np.asarray(grp['StellarMass']) * to_msun
            keep = (np.asarray(grp['Type']) == 0) & (ms > 0)
            if min_mstar:
                keep &= ms >= min_mstar
            mv = np.asarray(grp['Mvir']) * to_msun
            keep &= mv > 0
            idx = np.flatnonzero(keep)
            if idx.size == 0:
                continue
            d = {'StellarMass': ms[idx], 'Mvir': mv[idx]}
            for prop in props:
                if prop in d or prop == 'Type':
                    continue
                arr = np.asarray(grp[prop])[idx]
                d[prop] = arr * to_msun if prop in MASS_PROPS else arr
            out[snap] = d
    return out


def read_run(directory, props, min_mstar, nproc=None):
    """Every snapshot any redshift bin needs, from one run, keyed by snapshot.

    One pass over the files; optionally in parallel, since the files are
    independent.
    """
    hdr = header(directory)
    snaps = sorted({s for lo, hi, _ in Z_BINS
                    for s in hdr['output_snaps'] if lo <= hdr['z'][s] <= hi})
    files = model_files(directory)

    # Tasks are (file, snapshot chunk), not whole files, so parallelism does
    # not depend on how many files the run happened to be written to -- a run
    # in one big file parallelises over its snapshots just as well.  Chunks are
    # kept as large as possible so each still opens its file only once.
    if nproc and nproc > 1:
        per_file = max(1, -(-len(snaps) // max(1, -(-nproc // len(files)))))
    else:
        per_file = len(snaps)
    chunks = [snaps[i:i + per_file] for i in range(0, len(snaps), per_file)]
    tasks = [(p, c, props, hdr['to_msun'], min_mstar) for p in files for c in chunks]

    if nproc and nproc > 1 and len(tasks) > 1:
        import multiprocessing as mp
        with mp.Pool(min(nproc, len(tasks))) as pool:
            results = pool.map(_read_one_file, tasks)
    else:
        results = [_read_one_file(t) for t in tasks]

    merged = {}
    for part in results:
        for snap, d in part.items():
            merged.setdefault(snap, []).append(d)
    return hdr, {snap: {k: np.concatenate([d[k] for d in v]) for k in v[0]}
                 for snap, v in merged.items()}


def sample(hdr, per_snap, z_lo, z_hi, dilute=None):
    """Stack the snapshots in a redshift bin and derive SFR, sSFR and O/H.

    Both runs stack the same snapshots and the same haloes, so comparing them
    is unaffected.
    """
    snaps = [s for s in sorted(per_snap) if z_lo <= hdr['z'][s] <= z_hi]
    keys = per_snap[snaps[0]].keys()
    g = {k: np.concatenate([per_snap[s][k] for s in snaps]) for k in keys}

    # Random thinning, for speed on a big box.  A fair subsample of the same
    # population, so the medians are unchanged in expectation -- it costs
    # precision, not accuracy, and bins near MIN_COUNT may drop out.  Seeded,
    # and both runs are thinned the same way.
    n = len(g['StellarMass'])
    if dilute and n > dilute:
        pick = np.random.default_rng(SEED).choice(n, dilute, replace=False)
        g = {k: v[pick] for k, v in g.items()}

    if 'SfrDisk' in g:
        g['sfr'] = g['SfrDisk'] + g['SfrBulge']
        g['ssfr'] = g['sfr'] / g['StellarMass']
        with np.errstate(divide='ignore', invalid='ignore'):
            g['oh'] = np.log10((g['MetalsColdGas'] / g['ColdGas']) / Z_SUN) + 9.0
    return g


def cache_path(directory, props, min_mstar):
    """One cache file per (run, field set, mass cut), beside the run."""
    tag = f"{len(props)}p_{'all' if not min_mstar else f'{np.log10(min_mstar):.2f}'}"
    return os.path.join(directory, f'.ffb_efficiency_cache_{tag}.npz')


def load_run(directory, props, min_mstar, nproc=None, use_cache=True):
    """read_run(), memoised to an .npz beside the run's output.

    The reduced sample is a few MB against hundreds of GB of raw output, so
    re-plotting is instant.  The cache is invalidated whenever any model file
    is newer than it.
    """
    path = cache_path(directory, props, min_mstar)
    if use_cache and os.path.exists(path):
        newest = max(os.path.getmtime(p) for p in model_files(directory))
        if os.path.getmtime(path) >= newest:
            z = np.load(path)
            hdr = header(directory)
            per_snap = {}
            for key in z.files:
                snap, prop = key.split('|')
                per_snap.setdefault(int(snap), {})[prop] = z[key]
            print(f'  {directory}: cache hit ({os.path.basename(path)})')
            return hdr, per_snap

    hdr, per_snap = read_run(directory, props, min_mstar, nproc)
    if use_cache:
        try:
            # Uncompressed: the cache is small next to the raw output and
            # compressing it costs more time than the read it saves.
            np.savez(path, **{f'{s}|{k}': v
                                         for s, d in per_snap.items()
                                         for k, v in d.items()})
        except OSError as e:
            print(f'  could not write cache: {e}')
    return hdr, per_snap


def by_mstar(g, value, how='median'):
    """Bin by stellar mass; return (bin centre in log m*, statistic).

    how: 'median' of `value`, or 'fraction' when `value` is a boolean.
    """
    which = np.digitize(np.log10(g['StellarMass']), MSTAR_BINS) - 1
    centers = 0.5 * (MSTAR_BINS[:-1] + MSTAR_BINS[1:])
    x, y = [], []
    for i in range(len(centers)):
        in_bin = which == i
        if int(in_bin.sum()) < MIN_COUNT:
            continue
        v = value[in_bin]
        v = v[np.isfinite(v)]
        if len(v) < MIN_COUNT:
            continue
        x.append(centers[i])
        y.append(v.mean() if how == 'fraction' else np.median(v))
    return np.array(x), np.array(y)


def shmr(g, hubble_h):
    """Per stellar mass bin: median halo mass (10^10 h^-1 Msun), median m*/Mvir."""
    which = np.digitize(np.log10(g['StellarMass']), MSTAR_BINS) - 1
    x, y = [], []
    for i in range(len(MSTAR_BINS) - 1):
        in_bin = which == i
        if np.count_nonzero(in_bin) < MIN_COUNT:
            continue
        x.append(np.median(g['Mvir'][in_bin]) * hubble_h / 1e10)
        y.append(np.median(g['StellarMass'][in_bin] / g['Mvir'][in_bin]))
    return np.array(x), np.array(y)


# =============================== figures ===============================

def legend(ax, loc, handles, **kwargs):
    leg = ax.legend(handles, [h.get_label() for h in handles], loc=loc,
                    frameon=False, numpoints=1, labelspacing=0.1, **kwargs)
    for h in leg.legend_handles:
        h.set_alpha(1)
    return leg


def legend_handles():
    epochs = [Line2D([], [], color=c, lw=1.5, label=rf'${lo:g} < z < {hi:g}$')
              for lo, hi, c in Z_BINS]
    effs = [Line2D([], [], color='0.35', **FID_STYLE, label=FID_LABEL),
            Line2D([], [], color='0.35', **FULL_STYLE, label=FULL_LABEL)]
    return epochs, effs


def mass_cut_note(ax, min_mstar, x, y, **kwargs):
    if min_mstar:
        ax.text(x, y, rf'$m_* > 10^{{{np.log10(min_mstar):g}}}\ M_\odot$',
                transform=ax.transAxes, fontsize=10, **kwargs)


def figure_shmr(samples, f_b, min_mstar, outdir, fixed_axes=False):
    """m*/Mvir against halo mass, on the proposal figure's axes.

    fixed_axes=True locks the panel to SCREENSHOT_XLIM/YLIM -- the range and
    major ticks read off the proposal figure itself -- instead of sizing the
    axes to whatever the data spans.  The curves and the right-hand axis are
    unchanged either way: the right axis is still this run's own baryon
    fraction (from its HDF5 header), not a value borrowed from the source
    figure, since forcing an exact right-axis match would mean assuming a
    cosmic baryon fraction the model was not actually run with.

    The source figure's curves are UniverseMachine, an empirical model, not a
    forward physical prediction -- see the SCREENSHOT_XLIM comment above.
    fixed_axes gives ScreenshotAxes.pdf the same window to sit in, nothing more.
    """
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    xlim = SCREENSHOT_XLIM if fixed_axes else XLIM
    drawn = []

    for z_lo, z_hi, color in Z_BINS:
        for full, style in ((False, FID_STYLE), (True, FULL_STYLE)):
            g, hubble_h = samples[(z_lo, z_hi, full)]
            x, y = shmr(g, hubble_h)
            ax.plot(x, y, color=color, zorder=3, **style)
            drawn.append(y[(x >= xlim[0]) & (x <= xlim[1])])

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(*xlim)
    if fixed_axes:
        ylim = SCREENSHOT_YLIM
    else:
        shown = np.concatenate(drawn)
        shown = shown[np.isfinite(shown) & (shown > 0)]
        ylim = (shown.min() / YPAD, shown.max() * YPAD)
    ax.set_ylim(*ylim)
    xticks = SCREENSHOT_XTICKS if fixed_axes else XTICKS
    ax.set_xticks(xticks, [str(t) for t in xticks])
    ax.set_xticks([], minor=True)
    keep = [(v, t) for v, t in YTICKS if ylim[0] <= v <= ylim[1]]
    ax.set_yticks([v for v, _ in keep], [t for _, t in keep])
    ax.set_yticks([], minor=True)
    ax.set_xlabel(r'Halo Mass ($10^{10}\ h^{-1}\ M_{\odot}$)')
    ax.set_ylabel('Central Stellar Mass / Halo Mass')

    right = ax.secondary_yaxis('right', functions=(lambda v: v / f_b,
                                                   lambda v: v * f_b))
    right.set_ylabel('Central Stellar Baryon Fraction')
    fb_keep = [t for t in FBAR_TICKS if ylim[0] / f_b <= t <= ylim[1] / f_b]
    right.set_yticks(fb_keep, [str(t) for t in fb_keep])
    right.set_yticks([], minor=True)

    epochs, effs = legend_handles()
    ax.add_artist(legend(ax, 'upper left', epochs, fontsize=11))
    legend(ax, 'lower right', effs, fontsize=11)
    mass_cut_note(ax, min_mstar, 0.97, 0.97, ha='right', va='top')

    fig.tight_layout()
    name = 'FFB_Efficiency_SHMR_ScreenshotAxes.pdf' if fixed_axes \
           else 'FFB_Efficiency_SHMR.pdf'
    save(fig, outdir, name)


def figure_panels(samples, min_mstar, outdir):
    """Number counts across the top; SHMR, SFR, MZR and quiescent beneath."""
    # constrained layout, not tight_layout: the wide top panel spans the
    # gridspec and tight_layout cannot place it without overlapping row 2.
    fig = plt.figure(figsize=(16.0, 9.0), layout='constrained')
    gs = fig.add_gridspec(2, 4, height_ratios=[0.95, 1.0])
    ax_counts = fig.add_subplot(gs[0, :])
    ax_shmr = fig.add_subplot(gs[1, 0])
    ax_sfr = fig.add_subplot(gs[1, 1])
    ax_mzr = fig.add_subplot(gs[1, 2])
    ax_q = fig.add_subplot(gs[1, 3])

    ax_counts.set_yscale('log')

    for z_lo, z_hi, color in Z_BINS:
        for full, style in ((False, FID_STYLE), (True, FULL_STYLE)):
            g, hubble_h = samples[(z_lo, z_hi, full)]

            # Counts at the bin centres, on the shared MSTAR_BINS grid.
            counts, _ = np.histogram(np.log10(g['StellarMass']), bins=MSTAR_BINS)
            centers = 0.5 * (MSTAR_BINS[:-1] + MSTAR_BINS[1:])
            drawn = counts > 0
            ax_counts.plot(centers[drawn], counts[drawn], color=color, **style)

            x, y = shmr(g, hubble_h)
            ax_shmr.plot(x, y, color=color, **style)

            x, y = by_mstar(g, g['sfr'])
            ax_sfr.plot(x, y, color=color, **style)

            x, y = by_mstar(g, g['oh'])
            ax_mzr.plot(x, y, color=color, **style)

            x, y = by_mstar(g, (g['ssfr'] < SSFR_CUT).astype(float), 'fraction')
            ax_q.plot(x, y, color=color, **style)

    ax_counts.set_xlabel(MLABEL)
    ax_counts.set_ylabel('Number of galaxies')

    ax_shmr.set_xscale('log')
    ax_shmr.set_yscale('log')
    ax_shmr.set_xlabel(r'$M_{\mathrm{vir}}\ (10^{10}\ h^{-1}\ M_{\odot})$')
    ax_shmr.set_ylabel(r'$m_{*} / M_{\mathrm{vir}}$')

    ax_sfr.set_yscale('log')
    ax_sfr.set_xlabel(MLABEL)
    ax_sfr.set_ylabel(r'SFR $[M_{\odot}\ \mathrm{yr}^{-1}]$')

    ax_mzr.set_xlabel(MLABEL)
    ax_mzr.set_ylabel(r'$12 + \log_{10}(\mathrm{O/H})$')

    ax_q.set_xlabel(MLABEL)
    ax_q.set_ylabel('Quiescent fraction')
    ax_q.set_ylim(-0.03, 1.03)
    ax_q.text(0.04, 0.95, r'sSFR $< 10^{-11}\ \mathrm{yr}^{-1}$',
              transform=ax_q.transAxes, va='top', fontsize=9, color='0.35')

    epochs, effs = legend_handles()
    ax_counts.add_artist(legend(ax_counts, 'upper right', epochs, fontsize=11))
    # Stacked under the redshift key rather than in another corner, so the two
    # halves of the encoding are read together.
    legend(ax_counts, 'upper right', effs, fontsize=11,
           bbox_to_anchor=(1.0, 0.74))
    mass_cut_note(ax_counts, min_mstar, 0.5, 1.02, ha='center', va='bottom',
                  color='0.35')

    save(fig, outdir, 'FFB_Efficiency_Panels.pdf')


def save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    fig.savefig(path)
    plt.close(fig)
    print(f'  Saved: {path}')


# ================================ main ================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fid', default=FID_DIR, help='alpha_eff = 0.2 output')
    ap.add_argument('--full', default=FULL_DIR, help='alpha_eff = 1.0 output')
    ap.add_argument('--figure', default='both', choices=('both', 'shmr', 'panels'),
                    help="'shmr' also produces the fixed-axis "
                         "FFB_Efficiency_SHMR_ScreenshotAxes.pdf alongside it")
    ap.add_argument('--min-mstar', type=float, default=MIN_MSTAR, metavar='MSUN')
    ap.add_argument('--dilute', type=int, default=DILUTE, metavar='N',
                    help='randomly thin each sample to at most N galaxies')
    ap.add_argument('--nproc', type=int, default=0, metavar='N',
                    help='read on N processes, split over files and snapshots')
    ap.add_argument('--no-cache', action='store_true',
                    help='ignore and do not write the reduced-sample cache')
    ap.add_argument('--outdir', default=None, help='default: <fid>/plots/')
    args = ap.parse_args()

    min_mstar = args.min_mstar if args.min_mstar > 0 else None
    if os.path.exists(STYLE):
        plt.style.use(STYLE)

    # Only the fields the requested figures actually need.
    props = ['StellarMass', 'Mvir'] if args.figure == 'shmr' else list(PROPS)

    # Each run read once, in one pass over its files, then sliced per bin.
    samples = {}
    for full, directory in ((False, args.fid), (True, args.full)):
        hdr, per_snap = load_run(directory, props, min_mstar,
                                 args.nproc, not args.no_cache)
        for z_lo, z_hi, _ in Z_BINS:
            g = sample(hdr, per_snap, z_lo, z_hi, args.dilute)
            samples[(z_lo, z_hi, full)] = (g, hdr['hubble_h'])
            print(f'  {z_lo:g} < z < {z_hi:g}  {directory}: '
                  f'{len(g["StellarMass"])} centrals')

    outdir = args.outdir or os.path.join(args.fid, 'plots/')
    f_b = header(args.fid)['baryon_frac']
    # 'screenshot' is a fixed-axis view of the same SHMR curves as 'shmr', so
    # it is drawn alongside it by default rather than needing its own flag.
    if args.figure in ('both', 'shmr', 'screenshot'):
        figure_shmr(samples, f_b, min_mstar, outdir)
        figure_shmr(samples, f_b, min_mstar, outdir, fixed_axes=True)
    if args.figure in ('both', 'panels'):
        figure_panels(samples, min_mstar, outdir)


if __name__ == '__main__':
    main()

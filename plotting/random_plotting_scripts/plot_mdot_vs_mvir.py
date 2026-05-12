#!/usr/bin/env python
"""
Plot mdot_cool and mdot_stream as a function of Mvir.

Multi-panel figure with one column, each row a different redshift.
Uses the same data I/O infrastructure as paper_plots.py.

Usage:
    python plot_mdot_vs_mvir.py
"""

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import os

import warnings
warnings.filterwarnings("ignore")

# ========================== CONFIGURATION ==========================

PRIMARY_DIR = './output/millennium/'
MODEL_FILE = 'model_0.hdf5'
OUTPUT_DIR = './output/millennium/plots/'
OUTPUT_FORMAT = '.pdf'

HUBBLE_H = 0.73
MASS_CONVERT = 1.0e10 / HUBBLE_H

# Mass properties that need MASS_CONVERT
_MASS_PROPS = frozenset({
    'CentralMvir', 'Mvir', 'StellarMass', 'BulgeMass', 'BlackHoleMass',
    'MetalsStellarMass', 'MetalsColdGas', 'MetalsEjectedMass',
    'MetalsHotGas', 'MetalsCGMgas', 'ColdGas', 'HotGas', 'CGMgas',
    'EjectedMass', 'H2gas', 'H1gas', 'IntraClusterStars',
    'MergerBulgeMass', 'InstabilityBulgeMass', 'Vvir'
})

# Redshift array (snap 0 -> snap 63)
REDSHIFTS = [
    127.000, 79.998, 50.000, 30.000, 19.916, 18.244, 16.725, 15.343,
     14.086, 12.941, 11.897, 10.944, 10.073,  9.278,  8.550,  7.883,
      7.272,  6.712,  6.197,  5.724,  5.289,  4.888,  4.520,  4.179,
      3.866,  3.576,  3.308,  3.060,  2.831,  2.619,  2.422,  2.239,
      2.070,  1.913,  1.766,  1.630,  1.504,  1.386,  1.276,  1.173,
      1.078,  0.989,  0.905,  0.828,  0.755,  0.687,  0.624,  0.564,
      0.509,  0.457,  0.408,  0.362,  0.320,  0.280,  0.242,  0.208,
      0.175,  0.144,  0.116,  0.089,  0.064,  0.041,  0.020,  0.000,
]

# Snapshots to plot (snap_number, label)
SNAP_PANELS = [
    (63, r'$z = 0$'),
    (39, r'$z \approx 1$'),
    (32, r'$z \approx 2$'),
    (27, r'$z \approx 3$'),
    (23, r'$z \approx 4$'),
    (20, r'$z \approx 5$'),
]

# Properties to load per snapshot
PROPERTIES = ['Mvir', 'mdot_cool', 'mdot_stream', 'Type', 'Vvir']


# ========================== PLOTTING STYLE ==========================

def setup_style():
    """Configure matplotlib for publication-quality white-background plots."""
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["font.size"] = 14
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.major.size'] = 6
    plt.rcParams['ytick.major.size'] = 6
    plt.rcParams['xtick.minor.size'] = 3
    plt.rcParams['ytick.minor.size'] = 3
    plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['ytick.minor.visible'] = True
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['axes.labelcolor'] = 'black'
    plt.rcParams['text.color'] = 'black'
    plt.rcParams['legend.facecolor'] = 'white'
    plt.rcParams['legend.edgecolor'] = 'none'
    plt.rcParams['legend.framealpha'] = 0.8
    plt.rcParams['legend.fontsize'] = 'medium'
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Palatino'
    plt.rcParams['text.usetex'] = True


# ========================== DATA I/O ==========================

def load_snapshots(directory, snaps, properties, filename=MODEL_FILE):
    """
    Load multiple snapshots from a single HDF5 file.

    Returns
    -------
    dict : {snap_num: {prop_name: numpy array}}
    """
    filepath = os.path.join(directory, filename)
    snapdata = {}

    with h5.File(filepath, 'r') as f:
        for snap in snaps:
            snap_key = f'Snap_{snap}'
            if snap_key not in f:
                print(f"  Warning: {snap_key} not found, skipping.")
                continue
            grp = f[snap_key]
            data = {}
            for prop in properties:
                if prop in grp:
                    arr = np.array(grp[prop])
                    if prop in _MASS_PROPS:
                        arr *= MASS_CONVERT
                    data[prop] = arr
                else:
                    print(f"  Warning: '{prop}' not in {snap_key}")
            snapdata[snap] = data

    return snapdata


# ========================== BINNING UTILITY ==========================

def binned_median(x, y, bins, min_count=5):
    """Binned median with 16th/84th percentiles."""
    centers = 0.5 * (bins[:-1] + bins[1:])
    n = len(bins) - 1
    med = np.full(n, np.nan)
    p16 = np.full(n, np.nan)
    p84 = np.full(n, np.nan)

    for i in range(n):
        mask = (x >= bins[i]) & (x < bins[i + 1])
        count = np.sum(mask)
        if count >= min_count:
            vals = y[mask]
            med[i] = np.median(vals)
            p16[i] = np.percentile(vals, 16)
            p84[i] = np.percentile(vals, 84)

    return centers, med, p16, p84


# ========================== MAIN PLOT ==========================

def plot_mdot_vs_mvir():
    """Create multi-panel figure of mdot_cool and mdot_stream vs Mvir."""

    setup_style()

    snap_nums = [s for s, _ in SNAP_PANELS]
    snapdata = load_snapshots(PRIMARY_DIR, snap_nums, PROPERTIES)

    nrows = len(SNAP_PANELS)
    fig, axes = plt.subplots(nrows, 1, figsize=(7, 3.5 * nrows),
                             sharex=True)
    if nrows == 1:
        axes = [axes]

    mvir_bins = np.arange(9.5, 14.5, 0.2)

    for idx, (snap, zlabel) in enumerate(SNAP_PANELS):
        ax = axes[idx]

        if snap not in snapdata:
            ax.text(0.5, 0.5, f'{zlabel}: no data', transform=ax.transAxes,
                    ha='center', va='center')
            continue

        d = snapdata[snap]
        mvir = d['Mvir']
        mdot_cool = d.get('mdot_cool')
        mdot_stream = d.get('mdot_stream')

        # Only centrals (Type == 0) with positive Mvir
        central = (d.get('Type', np.zeros_like(mvir)) == 0) & (mvir > 0)
        log_mvir = np.log10(mvir[central])

        # --- mdot_cool ---
        if mdot_cool is not None:
            mc = mdot_cool[central]
            pos = mc > 0
            if np.sum(pos) > 0:
                log_mc = np.log10(mc[pos])
                c, med, p16, p84 = binned_median(log_mvir[pos], log_mc,
                                                  mvir_bins)
                valid = np.isfinite(med)
                ax.plot(c[valid], med[valid], color='C3', lw=2.2,
                        label=r'$\dot{M}_{\rm cool}$')
                ax.fill_between(c[valid], p16[valid], p84[valid],
                                color='C3', alpha=0.2)

        # --- mdot_stream ---
        if mdot_stream is not None:
            ms = mdot_stream[central]
            pos = ms > 0
            if np.sum(pos) > 0:
                log_ms = np.log10(ms[pos])
                c, med, p16, p84 = binned_median(log_mvir[pos], log_ms,
                                                  mvir_bins)
                valid = np.isfinite(med)
                ax.plot(c[valid], med[valid], color='C0', lw=2.2,
                        label=r'$\dot{M}_{\rm stream}$')
                ax.fill_between(c[valid], p16[valid], p84[valid],
                                color='C0', alpha=0.2)

        ax.set_ylabel(r'$\log_{10}\,\dot{M}\;[{\rm M}_\odot\,{\rm yr}^{-1}]$')
        ax.set_xlim(9.5, 14.5)
        ax.text(0.05, 0.92, zlabel, transform=ax.transAxes,
                fontsize=15, va='top', fontweight='bold')

        if idx == 0:
            ax.legend(loc='lower right', fontsize=12)

    axes[-1].set_xlabel(r'$\log_{10}\,(M_{\rm vir}\;/\;{\rm M}_\odot)$')

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.05)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    outpath = os.path.join(OUTPUT_DIR, f'mdot_vs_mvir{OUTPUT_FORMAT}')
    fig.savefig(outpath, bbox_inches='tight')
    print(f"Saved: {outpath}")
    plt.close(fig)

def plot_mdot_vs_vvir():
    """Create multi-panel figure of mdot_cool and mdot_stream vs Mvir."""

    setup_style()

    snap_nums = [s for s, _ in SNAP_PANELS]
    snapdata = load_snapshots(PRIMARY_DIR, snap_nums, PROPERTIES)

    nrows = len(SNAP_PANELS)
    fig, axes = plt.subplots(nrows, 1, figsize=(7, 3.5 * nrows),
                             sharex=True)
    if nrows == 1:
        axes = [axes]

    mvir_bins = np.arange(1.5, 3.0, 0.1)

    for idx, (snap, zlabel) in enumerate(SNAP_PANELS):
        ax = axes[idx]

        if snap not in snapdata:
            ax.text(0.5, 0.5, f'{zlabel}: no data', transform=ax.transAxes,
                    ha='center', va='center')
            continue

        d = snapdata[snap]
        mvir = d['Vvir']
        mdot_cool = d.get('mdot_cool')
        mdot_stream = d.get('mdot_stream')

        # Only centrals (Type == 0) with positive Mvir
        central = (d.get('Type', np.zeros_like(mvir)) == 0) & (mvir > 0)
        log_mvir = np.log10(mvir[central])

        # --- mdot_cool ---
        if mdot_cool is not None:
            mc = mdot_cool[central]
            pos = mc > 0
            if np.sum(pos) > 0:
                log_mc = np.log10(mc[pos])
                c, med, p16, p84 = binned_median(log_mvir[pos], log_mc,
                                                  mvir_bins)
                valid = np.isfinite(med)
                ax.plot(c[valid], med[valid], color='C3', lw=2.2,
                        label=r'$\dot{M}_{\rm cool}$')
                ax.fill_between(c[valid], p16[valid], p84[valid],
                                color='C3', alpha=0.2)

        # --- mdot_stream ---
        if mdot_stream is not None:
            ms = mdot_stream[central]
            pos = ms > 0
            if np.sum(pos) > 0:
                log_ms = np.log10(ms[pos])
                c, med, p16, p84 = binned_median(log_mvir[pos], log_ms,
                                                  mvir_bins)
                valid = np.isfinite(med)
                ax.plot(c[valid], med[valid], color='C0', lw=2.2,
                        label=r'$\dot{M}_{\rm stream}$')
                ax.fill_between(c[valid], p16[valid], p84[valid],
                                color='C0', alpha=0.2)

        ax.set_ylabel(r'$\log_{10}\,\dot{M}\;[{\rm M}_\odot\,{\rm yr}^{-1}]$')
        # ax.set_xlim(9.5, 14.5)
        ax.text(0.05, 0.92, zlabel, transform=ax.transAxes,
                fontsize=15, va='top', fontweight='bold')

        if idx == 0:
            ax.legend(loc='lower right', fontsize=12)

    axes[-1].set_xlabel(r'$\log_{10}\,(V_{\rm vir}\;/\;{\rm km\,s}^{-1})$')

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.05)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    outpath = os.path.join(OUTPUT_DIR, f'mdot_vs_vvir{OUTPUT_FORMAT}')
    fig.savefig(outpath, bbox_inches='tight')
    print(f"Saved: {outpath}")
    plt.close(fig)


if __name__ == '__main__':
    plot_mdot_vs_mvir()
    plot_mdot_vs_vvir()

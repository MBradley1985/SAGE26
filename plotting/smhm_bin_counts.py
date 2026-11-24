#!/usr/bin/env python
"""
smhm_bin_counts.py -- stellar mass vs halo mass as a hexbin of galaxy counts.

One panel per SAGE output directory: every galaxy with Mvir > 0 and M* above
--min-stellar is binned in log Mvir -- log M* hexagons, coloured by the number of
galaxies in each hexagon (log colour scale). The number of galaxies per halo-mass
bin is also printed as a table.

By default a galaxy is binned by its own Mvir. With --host each galaxy is binned by
its host's CentralMvir instead.

Usage:
    python plotting/smhm_bin_counts.py                                   # ./output/millennium/
    python plotting/smhm_bin_counts.py output/millennium_vanilla/ output/millennium/ \
        --labels "Vanilla" "SAGE26" -o smhm_vanilla_vs_sage26.png

Reads:  <OutputDir>/model_*.hdf5
Writes: <first OutputDir>/plots/smhm_bin_counts_snapNNN.png (or -o)
"""

import argparse
import glob
import os

import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


def read_snapshot(files, snap, fields):
    """Concatenate the requested fields of one snapshot across all model files."""
    out = {name: [] for name in fields}
    for path in files:
        with h5.File(path, 'r') as f:
            if snap not in f:
                continue
            for name in fields:
                out[name].append(np.array(f[snap][name]))
    return {name: np.concatenate(arrs) for name, arrs in out.items()}


def load(outdir, snap_num, halo_field, min_stellar):
    """Return (Mvir, M*, redshift, snap_num) in Msun for one output directory."""
    files = sorted(glob.glob(os.path.join(outdir, 'model_*.hdf5')))
    if not files:
        raise SystemExit(f'No model_*.hdf5 files in {outdir}')

    with h5.File(files[0], 'r') as f:
        hubble_h = float(f['Header/Simulation'].attrs['hubble_h'])
        redshifts = np.array(f['Header/snapshot_redshifts'])
        snaps = sorted(int(k[5:]) for k in f if k.startswith('Snap_'))
    if snap_num is None:
        snap_num = snaps[-1]

    data = read_snapshot(files, f'Snap_{snap_num}', [halo_field, 'StellarMass'])
    mhalo = data[halo_field] * 1.0e10 / hubble_h
    mstar = data['StellarMass'] * 1.0e10 / hubble_h
    keep = (mhalo > 0) & (mstar > min_stellar)
    return mhalo[keep], mstar[keep], redshifts[snap_num], snap_num


def print_counts(label, mhalo, bin_width):
    """Table of galaxy counts per log Mvir bin."""
    log_mh = np.log10(mhalo)
    edges = np.arange(np.floor(log_mh.min() / bin_width) * bin_width,
                      log_mh.max() + bin_width, bin_width)
    counts, _ = np.histogram(log_mh, bins=edges)
    print(f'\n{label}: {len(mhalo)} galaxies')
    print(f'{"log Mvir bin":>15} {"N":>9}')
    for lo, hi, n in zip(edges[:-1], edges[1:], counts):
        if n > 0:
            print(f'{lo:6.2f} - {hi:5.2f} {n:9d}')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('outdirs', nargs='*', default=['./output/millennium/'])
    parser.add_argument('--labels', nargs='+', default=None,
                        help='panel titles, one per directory (default: directory names)')
    parser.add_argument('-s', '--snap', type=int, default=None,
                        help='snapshot number (default: latest in each file)')
    parser.add_argument('--host', action='store_true',
                        help='bin by host CentralMvir rather than the galaxy\'s own Mvir')
    parser.add_argument('--min-stellar', type=float, default=1.0e6,
                        help='stellar mass cut in Msun (default 1e6)')
    parser.add_argument('--gridsize', type=int, default=80, help='hexagons across the x-axis')
    parser.add_argument('--bin-width', type=float, default=0.5,
                        help='log Mvir bin width for the printed count table')
    parser.add_argument('-o', '--output', default=None, help='output file path')
    args = parser.parse_args()

    labels = args.labels or [os.path.basename(os.path.normpath(d)) for d in args.outdirs]
    if len(labels) != len(args.outdirs):
        raise SystemExit('Give one --labels entry per output directory')

    halo_field = 'CentralMvir' if args.host else 'Mvir'
    runs = [load(d, args.snap, halo_field, args.min_stellar) for d in args.outdirs]

    # Shared axis limits so the panels compare directly
    xlim = (min(r[0].min() for r in runs) / 2, max(r[0].max() for r in runs) * 2)
    ylim = (args.min_stellar / 2, max(r[1].max() for r in runs) * 2)
    extent = np.log10([xlim[0], xlim[1], ylim[0], ylim[1]])

    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 5.5), squeeze=False)
    xlabel = r'Host Halo Mass ($M_\odot$)' if args.host else r'Halo Mass ($M_\odot$)'
    for ax, label, (mhalo, mstar, z, snap_num) in zip(axes[0], labels, runs):
        hb = ax.hexbin(mhalo, mstar, xscale='log', yscale='log', gridsize=args.gridsize,
                       extent=extent, mincnt=1, norm=LogNorm(), cmap='viridis')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r'Stellar Mass ($M_\odot$)')
        ax.set_title(label)
        fig.colorbar(hb, ax=ax, label='Bin Count')
        print_counts(f'{label} (Snap_{snap_num}, z = {z:.2f}, binned by {halo_field})',
                     mhalo, args.bin_width)

    fig.tight_layout()
    if args.output:
        outfile = args.output
    else:
        plot_dir = os.path.join(args.outdirs[0], 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        suffix = '_host' if args.host else ''
        outfile = os.path.join(plot_dir, f'smhm_bin_counts_snap{runs[0][3]:03d}{suffix}.png')
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f'\nSaved {outfile}')


if __name__ == '__main__':
    main()

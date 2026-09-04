#!/usr/bin/env python3
"""
plot_super_eddington_halo_mass.py
==================================
Plot the halo-mass distribution of super-Eddington galaxy records (as
selected by find_galaxy.py --edd-ratio-min) for one or more scenarios,
overlaid for comparison.

With --by-redshift, instead of one panel pooling every snapshot together,
makes a grid with one panel per snapshot actually present in the data
(labelled by its redshift), so you can see how the halo-mass distribution
of super-Eddington events evolves over cosmic time.

Usage
-----
  python3 plot_super_eddington_halo_mass.py \
      output/millennium/heavy_disknoedd_edd_10t/super_eddington_events.csv \
      output/millennium/heavy_mergernoedd_edd_10t/super_eddington_events.csv \
      --labels "disk, no Edd cap" "merger, no Edd cap" \
      -o output/millennium/super_eddington_halo_mass.png

  python3 plot_super_eddington_halo_mass.py \
      output/millennium/heavy_disknoedd_edd_10t/super_eddington_events.csv \
      output/millennium/heavy_mergernoedd_edd_10t/super_eddington_events.csv \
      --labels "disk, no Edd cap" "merger, no Edd cap" \
      --by-redshift \
      -o output/millennium/super_eddington_halo_mass_by_z.png
"""

import argparse
import math
import os

import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ['#1976D2', '#C62828', '#00897B', '#8E24AA']


def default_label(csv_path):
    return os.path.basename(os.path.dirname(os.path.abspath(csv_path))) or csv_path


def read_redshifts(csv_path):
    """Header/snapshot_redshifts from the model_0.hdf5 next to this CSV."""
    model_path = os.path.join(os.path.dirname(os.path.abspath(csv_path)), 'model_0.hdf5')
    if not os.path.isfile(model_path):
        return None
    with h5py.File(model_path, 'r') as f:
        if 'Header/snapshot_redshifts' in f:
            return np.array(f['Header/snapshot_redshifts'])
    return None


def load_dataset(csv_path, field):
    df = pd.read_csv(csv_path)
    valid = np.isfinite(df[field].to_numpy()) & (df[field].to_numpy() > 0)
    return df.loc[valid].copy()


def log_halo(df, field):
    return np.log10(df[field].to_numpy())


def plot_overlay(ax, datasets, field, bins, show_legend=True):
    for i, (label, sub) in enumerate(datasets):
        color = COLORS[i % len(COLORS)]
        vals = log_halo(sub, field)
        if len(vals) == 0:
            continue
        ax.hist(vals, bins=bins, density=True, histtype='step', lw=2,
                color=color, label=f'{label} (N={len(vals)})')
        ax.axvline(np.median(vals), color=color, ls='--', lw=1, alpha=0.8)
    ax.grid(True, alpha=0.3)
    if show_legend:
        ax.legend(fontsize=9)


def main():
    p = argparse.ArgumentParser(
        description="Overlay halo-mass distributions of super-Eddington galaxy "
                    "records from one or more find_galaxy.py CSV outputs.")
    p.add_argument('csv', nargs='+', help='super_eddington_events.csv file(s).')
    p.add_argument('--labels', nargs='+', default=None,
                   help='Legend label per CSV (default: parent directory name).')
    p.add_argument('--field', choices=['CentralMvir', 'Mvir'], default='CentralMvir',
                   help="Halo mass field to plot (default: CentralMvir, the host "
                        "FoF halo mass -- meaningful for centrals and satellites "
                        "alike; Mvir is the galaxy's own subhalo mass, which is "
                        "stripped down for satellites).")
    p.add_argument('--bins', type=int, default=40)
    p.add_argument('--by-redshift', action='store_true',
                   help='Grid of one panel per snapshot present in the data '
                        '(labelled by redshift) instead of one pooled panel.')
    p.add_argument('--exclude-snap', type=int, nargs='+', default=None,
                   help='Snapshot(s) to drop from the --by-redshift grid.')
    p.add_argument('-o', '--output', default=None,
                   help='Output plot path (default: next to the first CSV).')
    args = p.parse_args()

    labels = args.labels if args.labels else [default_label(c) for c in args.csv]
    if len(labels) != len(args.csv):
        raise SystemExit(f"--labels needs {len(args.csv)} entries, got {len(labels)}")

    frames = [load_dataset(c, args.field) for c in args.csv]
    for label, df in zip(labels, frames):
        vals = log_halo(df, args.field)
        print(f"{label}: N={len(vals)}  "
              f"log10({args.field})  median={np.median(vals):.2f}  "
              f"[{np.percentile(vals, 16):.2f}, {np.percentile(vals, 84):.2f}] (16-84%)")

    lo = min(log_halo(df, args.field).min() for df in frames)
    hi = max(log_halo(df, args.field).max() for df in frames)
    bins = np.linspace(lo, hi, args.bins + 1)

    if not args.by_redshift:
        fig, ax = plt.subplots(figsize=(8.34, 6.25))
        plot_overlay(ax, list(zip(labels, frames)), args.field, bins)
        ax.set_xlabel(rf'$\log_{{10}}(M_{{\rm {args.field}}} / M_\odot)$', fontsize=14)
        ax.set_ylabel('Normalized density', fontsize=14)
        ax.set_title('Halo mass distribution of super-Eddington galaxy records', fontsize=13)
        plt.tight_layout()
        out = args.output or os.path.join(
            os.path.dirname(os.path.abspath(args.csv[0])),
            'super_eddington_halo_mass.png')
        fig.savefig(out, dpi=140, bbox_inches='tight')
        plt.close(fig)
        print(f"[ok] plot -> {out}")
        return

    redshifts = read_redshifts(args.csv[0])
    if redshifts is None:
        raise SystemExit("Error: could not find Header/snapshot_redshifts in "
                          f"model_0.hdf5 next to {args.csv[0]}.")

    snaps = sorted(set().union(*[set(df['Snapshot'].unique()) for df in frames]))
    if args.exclude_snap:
        snaps = [s for s in snaps if s not in args.exclude_snap]
    n = len(snaps)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.6 * nrows),
                             squeeze=False)

    for idx, snap in enumerate(snaps):
        ax = axes[idx // ncols][idx % ncols]
        panel_datasets = [(label, df[df['Snapshot'] == snap]) for label, df in zip(labels, frames)]
        plot_overlay(ax, panel_datasets, args.field, bins, show_legend=(idx == 0))
        z = redshifts[int(snap)] if int(snap) < len(redshifts) else None
        title = f'snap {int(snap)}' + (f'  (z = {z:.2f})' if z is not None else '')
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(rf'$\log_{{10}}(M_{{\rm {args.field}}} / M_\odot)$', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis('off')

    fig.suptitle('Halo mass distribution of super-Eddington galaxy records, by redshift',
                 fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = args.output or os.path.join(
        os.path.dirname(os.path.abspath(args.csv[0])),
        'super_eddington_halo_mass_by_z.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"[ok] plot -> {out}")


if __name__ == "__main__":
    main()

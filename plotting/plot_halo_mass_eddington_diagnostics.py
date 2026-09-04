#!/usr/bin/env python3
"""
plot_halo_mass_eddington_diagnostics.py
========================================
Two diagnostics, computed directly from SAGE26 HDF5 output (not from a
find_galaxy.py CSV, so the *entire* galaxy population is used -- not just
the pre-selected super-Eddington subset):

  1. Halo mass function: number density of central galaxies (unique FOF
     haloes) per log10(Mvir/Msun) dex per Mpc^3, at a single representative
     snapshot ("total") and split into one panel per snapshot ("by
     redshift"). Fixes the blank HaloMassFunctionEvolution.pdf panels from
     allresults-history.py, which come out empty for sparse-output runs:
     that script only plots a redshift bin if it can compute a std-error
     across >=2 snapshots landing in the bin, but sparse runs (like this
     one, with output at only 7 of 64 snapshots) have at most one snapshot
     per bin, so every point's std-error is exactly zero and gets filtered
     out.
  2. Eddington ratio (BHMaxaccretionRate / BHEddingtonRateLimit) vs. host
     halo mass, for every actively-accreting galaxy (both centrals and
     satellites, since a satellite's *host* halo mass is what CentralMvir
     gives you) -- pooled across all snapshots ("total") and split into one
     panel per snapshot ("by redshift"). Shows whether the tendency to go
     super-Eddington (ratio >= 1, marked with a reference line) depends on
     halo mass, using the full dynamic range rather than only the
     already-super-Eddington subsample.

Usage
-----
  python3 plot_halo_mass_eddington_diagnostics.py \
      "output/millennium/heavy_edd_10t/model_*.hdf5" \
      -o output/millennium/heavy_edd_10t/

  python3 plot_halo_mass_eddington_diagnostics.py \
      "output/millennium/heavy_disknoedd_edd_10t/model_*.hdf5" \
      "output/millennium/heavy_mergernoedd_edd_10t/model_*.hdf5" \
      --labels "disk, no Edd cap" "merger, no Edd cap" \
      --exclude-snap 27 \
      -o output/millennium/
"""

import argparse
import glob
import math
import os

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ['#1976D2', '#C62828', '#00897B', '#8E24AA']
SEC_PER_YEAR = 365.25 * 24 * 3600
UNIT_LENGTH_IN_CM_DEFAULT = 3.08568e24
UNIT_VELOCITY_IN_CM_PER_S_DEFAULT = 1.0e5


def default_label(pattern):
    first_match = sorted(glob.glob(pattern))
    if not first_match:
        return pattern
    return os.path.basename(os.path.dirname(os.path.abspath(first_match[0])))


def read_sim_header(filepath):
    with h5py.File(filepath, 'r') as f:
        sim = f['Header/Simulation'].attrs
        hubble_h = float(sim['hubble_h'])
        box_size = float(sim['box_size'])
        runtime = f['Header/Runtime'].attrs
        volume_fraction = float(runtime.get('frac_volume_processed', 1.0))
        unit_time_in_s = (float(runtime.get('UnitLength_in_cm', UNIT_LENGTH_IN_CM_DEFAULT)) /
                          float(runtime.get('UnitVelocity_in_cm_per_s', UNIT_VELOCITY_IN_CM_PER_S_DEFAULT)))
        redshifts = np.array(f['Header/snapshot_redshifts'])
        snaps = sorted(int(k.split('_')[1]) for k in f.keys() if k.startswith('Snap_'))
    return {
        'hubble_h': hubble_h, 'box_size': box_size,
        'volume_fraction': volume_fraction, 'unit_time_in_s': unit_time_in_s,
        'redshifts': redshifts, 'snapshots': snaps,
    }


def read_snapshot(file_list, snap, hubble_h, unit_time_in_s):
    """Per-snapshot fields for every galaxy. BHMaxaccretionRate and
    BHEddingtonRateLimit are stored as [ngal, SimMaxSnaps] history arrays;
    collapse to the column for this snapshot the same way find_galaxy.py
    does (col = min(max(snap-1, 0), ncols-1) -- SAGE only finishes writing
    column N once processing has moved on to snapshot N+1)."""
    conv = 1.0e10 / hubble_h
    rate_conv = conv * SEC_PER_YEAR / unit_time_in_s
    parts = {k: [] for k in ('Mvir', 'CentralMvir', 'Type', 'BHMaxaccretionRate', 'BHEddingtonRateLimit')}
    key = f'Snap_{snap}'
    for fpath in file_list:
        with h5py.File(fpath, 'r') as f:
            if key not in f:
                continue
            grp = f[key]
            n = grp['Mvir'].shape[0] if 'Mvir' in grp else 0
            if n == 0:
                continue
            for field in parts:
                arr = np.array(grp[field])
                if arr.ndim == 2:
                    col = min(max(snap - 1, 0), arr.shape[1] - 1)
                    arr = arr[:, col]
                parts[field].append(arr)
    if not parts['Mvir']:
        return None
    out = {field: np.concatenate(arrs) for field, arrs in parts.items()}
    out['Mvir'] = out['Mvir'] * conv
    out['CentralMvir'] = out['CentralMvir'] * conv
    out['BHMaxaccretionRate'] = out['BHMaxaccretionRate'] * rate_conv
    out['BHEddingtonRateLimit'] = out['BHEddingtonRateLimit'] * rate_conv
    return out


def compute_hmf(data, mass_bins, volume):
    centrals = (data['Type'] == 0) & (data['Mvir'] > 0)
    log_mvir = np.log10(data['Mvir'][centrals])
    bin_width = mass_bins[1] - mass_bins[0]
    counts, _ = np.histogram(log_mvir, bins=mass_bins)
    phi = counts / (volume * bin_width)
    return phi


def compute_ratio_vs_mass(data, field):
    edd = data['BHEddingtonRateLimit']
    acc = data['BHMaxaccretionRate']
    halo = data[field]
    valid = (edd > 0) & (acc > 0) & (halo > 0) & np.isfinite(edd) & np.isfinite(acc)
    log_halo = np.log10(halo[valid])
    log_ratio = np.log10(acc[valid] / edd[valid])
    return log_halo, log_ratio


def binned_median(x, y, bins):
    centers, med, lo16, hi84, n = [], [], [], [], []
    for i in range(len(bins) - 1):
        m = (x >= bins[i]) & (x < bins[i + 1])
        if m.sum() < 5:
            continue
        centers.append(0.5 * (bins[i] + bins[i + 1]))
        med.append(np.median(y[m]))
        lo16.append(np.percentile(y[m], 16))
        hi84.append(np.percentile(y[m], 84))
        n.append(int(m.sum()))
    return np.array(centers), np.array(med), np.array(lo16), np.array(hi84), n


def main():
    p = argparse.ArgumentParser(
        description="Halo mass function and Eddington-ratio-vs-halo-mass "
                    "diagnostics, computed from the full galaxy population "
                    "in SAGE26 HDF5 output (not a find_galaxy.py subset).")
    p.add_argument('input_pattern', nargs='+',
                   help='Glob(s) for model HDF5 files, one per scenario.')
    p.add_argument('--labels', nargs='+', default=None,
                   help='Legend label per scenario (default: parent directory name).')
    p.add_argument('--field', choices=['CentralMvir', 'Mvir'], default='CentralMvir',
                   help='Halo mass field for the Eddington-ratio plot '
                        '(default: CentralMvir -- the host FoF halo, valid '
                        'for centrals and satellites alike).')
    p.add_argument('--exclude-snap', type=int, nargs='+', default=None,
                   help='Snapshot(s) to drop from the by-redshift grids.')
    p.add_argument('-o', '--output-dir', default='.',
                   help='Output directory for the 4 plots.')
    args = p.parse_args()

    patterns = args.input_pattern
    labels = args.labels if args.labels else [default_label(pat) for pat in patterns]
    if len(labels) != len(patterns):
        raise SystemExit(f"--labels needs {len(patterns)} entries, got {len(labels)}")

    os.makedirs(args.output_dir, exist_ok=True)

    scenarios = []
    for pattern, label in zip(patterns, labels):
        file_list = sorted(glob.glob(pattern))
        if not file_list:
            raise SystemExit(f"Error: no files match {pattern}")
        header = read_sim_header(file_list[0])
        volume_fraction = sum(read_sim_header(f)['volume_fraction'] for f in file_list)
        volume = (header['box_size'] / header['hubble_h']) ** 3.0 * volume_fraction
        snaps = header['snapshots']
        if args.exclude_snap:
            snaps_for_grid = [s for s in snaps if s not in args.exclude_snap]
        else:
            snaps_for_grid = snaps
        print(f"{label}: {len(file_list)} file(s), volume={volume:.3e} Mpc^3, "
              f"snapshots={snaps}")
        scenarios.append({
            'label': label, 'file_list': file_list, 'header': header,
            'volume': volume, 'snaps': snaps, 'snaps_for_grid': snaps_for_grid,
        })

    mass_bins = np.arange(10.0, 15.0, 0.2)

    # ---- Halo mass function: total (latest snapshot per scenario) ----
    fig, ax = plt.subplots(figsize=(8.34, 6.25))
    for i, sc in enumerate(scenarios):
        snap = sc['snaps'][-1]
        data = read_snapshot(sc['file_list'], snap, sc['header']['hubble_h'], sc['header']['unit_time_in_s'])
        phi = compute_hmf(data, mass_bins, sc['volume'])
        z = sc['header']['redshifts'][snap]
        centers = mass_bins[:-1] + 0.5 * (mass_bins[1] - mass_bins[0])
        valid = phi > 0
        color = COLORS[i % len(COLORS)]
        ax.plot(centers[valid], np.log10(phi[valid]), color=color, lw=2,
                label=f"{sc['label']} (snap {snap}, z={z:.2f})")
    ax.set_xlabel(r'$\log_{10}(M_{\rm vir} / M_\odot)$', fontsize=14)
    ax.set_ylabel(r'$\log_{10}\,\phi$  [Mpc$^{-3}$ dex$^{-1}$]', fontsize=14)
    ax.set_title('Halo mass function (centrals only)', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(args.output_dir, 'halo_mass_function.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"[ok] plot -> {out}")

    # ---- Halo mass function: by redshift ----
    all_grid_snaps = sorted(set().union(*[set(sc['snaps_for_grid']) for sc in scenarios]))
    n = len(all_grid_snaps)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.6 * nrows), squeeze=False)
    centers = mass_bins[:-1] + 0.5 * (mass_bins[1] - mass_bins[0])
    for idx, snap in enumerate(all_grid_snaps):
        ax = axes[idx // ncols][idx % ncols]
        z_label = None
        for i, sc in enumerate(scenarios):
            if snap not in sc['snaps']:
                continue
            data = read_snapshot(sc['file_list'], snap, sc['header']['hubble_h'], sc['header']['unit_time_in_s'])
            phi = compute_hmf(data, mass_bins, sc['volume'])
            valid = phi > 0
            color = COLORS[i % len(COLORS)]
            ax.plot(centers[valid], np.log10(phi[valid]), color=color, lw=2, label=sc['label'])
            if z_label is None:
                z_label = sc['header']['redshifts'][snap]
        title = f'snap {snap}' + (f'  (z = {z_label:.2f})' if z_label is not None else '')
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(r'$\log_{10}(M_{\rm vir}/M_\odot)$', fontsize=10)
        ax.set_ylabel(r'$\log_{10}\,\phi$', fontsize=10)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8)
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis('off')
    fig.suptitle('Halo mass function by redshift (centrals only)', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(args.output_dir, 'halo_mass_function_by_z.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"[ok] plot -> {out}")

    # ---- Eddington ratio vs halo mass: total (pooled across snapshots) ----
    ratio_bins = np.linspace(10.0, 14.5, 26)
    fig, ax = plt.subplots(figsize=(8.34, 6.25))
    for i, sc in enumerate(scenarios):
        log_halo_all, log_ratio_all = [], []
        for snap in sc['snaps']:
            data = read_snapshot(sc['file_list'], snap, sc['header']['hubble_h'], sc['header']['unit_time_in_s'])
            lh, lr = compute_ratio_vs_mass(data, args.field)
            log_halo_all.append(lh)
            log_ratio_all.append(lr)
        log_halo_all = np.concatenate(log_halo_all)
        log_ratio_all = np.concatenate(log_ratio_all)
        n_super = int(np.sum(log_ratio_all >= 0))
        print(f"{sc['label']}: {len(log_ratio_all)} accreting galaxies pooled, "
              f"{n_super} ({100 * n_super / len(log_ratio_all):.2f}%) super-Eddington")
        centers, med, lo16, hi84, counts = binned_median(log_halo_all, log_ratio_all, ratio_bins)
        color = COLORS[i % len(COLORS)]
        ax.plot(centers, med, color=color, lw=2, label=f"{sc['label']} (N={len(log_ratio_all)})")
        ax.fill_between(centers, lo16, hi84, color=color, alpha=0.18, linewidth=0)
    ax.axhline(0.0, color='grey', ls='--', lw=1.2, alpha=0.8, label='Eddington limit')
    ax.set_xlabel(rf'$\log_{{10}}(M_{{\rm {args.field}}} / M_\odot)$', fontsize=14)
    ax.set_ylabel(r'$\log_{10}(\dot{M}_{\rm BH}/\dot{M}_{\rm Edd})$', fontsize=14)
    ax.set_title(r'$\dot{M}_{\rm BH}/\dot{M}_{\rm Edd}$ vs. host halo mass (all accreting galaxies)', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(args.output_dir, 'halo_mass_vs_eddington_ratio.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"[ok] plot -> {out}")

    # ---- Eddington ratio vs halo mass: by redshift ----
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.6 * nrows), squeeze=False)
    for idx, snap in enumerate(all_grid_snaps):
        ax = axes[idx // ncols][idx % ncols]
        z_label = None
        for i, sc in enumerate(scenarios):
            if snap not in sc['snaps']:
                continue
            data = read_snapshot(sc['file_list'], snap, sc['header']['hubble_h'], sc['header']['unit_time_in_s'])
            lh, lr = compute_ratio_vs_mass(data, args.field)
            if z_label is None:
                z_label = sc['header']['redshifts'][snap]
            color = COLORS[i % len(COLORS)]
            centers, med, lo16, hi84, counts = binned_median(lh, lr, ratio_bins)
            if len(centers) == 0:
                continue
            ax.plot(centers, med, color=color, lw=2, label=sc['label'])
            ax.fill_between(centers, lo16, hi84, color=color, alpha=0.18, linewidth=0)
        ax.axhline(0.0, color='grey', ls='--', lw=1, alpha=0.8)
        title = f'snap {snap}' + (f'  (z = {z_label:.2f})' if z_label is not None else '')
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(rf'$\log_{{10}}(M_{{\rm {args.field}}}/M_\odot)$', fontsize=10)
        ax.set_ylabel(r'$\log_{10}(\dot{M}_{\rm BH}/\dot{M}_{\rm Edd})$', fontsize=10)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8)
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis('off')
    fig.suptitle(r'$\dot{M}_{\rm BH}/\dot{M}_{\rm Edd}$ vs. host halo mass, by redshift', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(args.output_dir, 'halo_mass_vs_eddington_ratio_by_z.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"[ok] plot -> {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
bh_bulge_ratio_histogram.py
============================
Histogram of M_BH / M_bulge for SAGE26 HDF5 output.

Reads BlackHoleMass and BulgeMass at a given snapshot, applies a minimum mass
floor to avoid noise from near-zero bulges/BHs, and plots a histogram of
log10(M_BH / M_bulge).

Usage
-----
  python3 bh_bulge_ratio_histogram.py -i "output/millennium/model_*.hdf5"
  python3 bh_bulge_ratio_histogram.py -i "..." -s 63
"""

import argparse
import glob
import os
import sys

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HUBBLE_H_DEFAULT = 0.73
MIN_BULGE_MASS = 1.0e8   # Msun
MIN_BH_MASS = 1.0e5      # Msun


def read_simulation_params(filepath):
    params = {'Hubble_h': HUBBLE_H_DEFAULT, 'available_snapshots': [], 'latest_snapshot': None}
    with h5py.File(filepath, 'r') as f:
        if 'Header/Simulation' in f:
            sim = f['Header/Simulation'].attrs
            params['Hubble_h'] = float(sim.get('hubble_h', sim.get('HubbleParam', HUBBLE_H_DEFAULT)))
        snap_groups = [k for k in f.keys() if k.startswith('Snap_')]
        snaps = sorted(int(s.split('_')[1]) for s in snap_groups)
        params['available_snapshots'] = snaps
        params['latest_snapshot'] = max(snaps) if snaps else None
    return params


def read_field(file_list, snap_num, field, conv=1.0):
    out = []
    key = f"Snap_{snap_num}"
    for f in file_list:
        with h5py.File(f, 'r') as hf:
            if key in hf and field in hf[key]:
                arr = np.array(hf[key][field])
                if arr.size:
                    out.append(arr)
    if not out:
        return np.array([])
    return np.concatenate(out) * conv


def main():
    p = argparse.ArgumentParser(description="Histogram of M_BH / M_bulge from SAGE26 HDF5 output.")
    p.add_argument('-i', '--input-pattern', default='./output/millennium/model_*.hdf5',
                   help='Glob for the model HDF5 files.')
    p.add_argument('-s', '--snapshot', type=int, default=None,
                   help='Snapshot to use (default: latest available).')
    p.add_argument('-o', '--output-dir', default=None,
                   help='Output directory (default: <input_dir>/plots).')
    p.add_argument('--min-bulge-mass', type=float, default=MIN_BULGE_MASS,
                   help=f'Minimum bulge mass in Msun (default: {MIN_BULGE_MASS:.0e}).')
    p.add_argument('--min-bh-mass', type=float, default=MIN_BH_MASS,
                   help=f'Minimum BH mass in Msun (default: {MIN_BH_MASS:.0e}).')
    p.add_argument('--bins', type=int, default=40, help='Number of histogram bins.')
    args = p.parse_args()

    file_list = sorted(glob.glob(args.input_pattern))
    if not file_list:
        print(f"Error: no files match {args.input_pattern}")
        sys.exit(1)

    sim = read_simulation_params(file_list[0])
    hubble_h = sim['Hubble_h']
    snap_num = args.snapshot if args.snapshot is not None else sim['latest_snapshot']
    if snap_num is None:
        print("Error: could not determine a snapshot to use.")
        sys.exit(1)

    conv = 1.0e10 / hubble_h
    bulge_mass = read_field(file_list, snap_num, 'BulgeMass', conv)
    bh_mass = read_field(file_list, snap_num, 'BlackHoleMass', conv)

    if len(bulge_mass) == 0 or len(bh_mass) == 0:
        print(f"Error: no BulgeMass/BlackHoleMass data at snapshot {snap_num}.")
        sys.exit(1)

    mask = (bulge_mass > args.min_bulge_mass) & (bh_mass > args.min_bh_mass)
    ratio = bh_mass[mask] / bulge_mass[mask]
    log_ratio = np.log10(ratio)

    if len(log_ratio) == 0:
        print("Error: no galaxies pass the mass cuts.")
        sys.exit(1)

    input_dir = os.path.dirname(os.path.abspath(file_list[0]))
    output_dir = args.output_dir or os.path.join(input_dir, 'plots')
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.34, 6.25))
    ax.hist(log_ratio, bins=args.bins, color='#1976D2', alpha=0.75,
            edgecolor='black', linewidth=0.5)
    ax.axvline(np.median(log_ratio), color='k', ls='--', lw=1.3,
               label=f'median = {np.median(log_ratio):.2f}')
    ax.set_xlabel(r'$\log_{10}(M_{\rm BH} / M_{\rm bulge})$', fontsize=14)
    ax.set_ylabel('Count', fontsize=13)
    ax.set_title(f'BH/bulge mass ratio (snap {snap_num}, N={len(log_ratio):,})', fontsize=13)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(output_dir, 'bh_bulge_mass_ratio_histogram.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)

    print(f"[ok] BH/bulge ratio histogram ({len(log_ratio):,} galaxies) -> {out}")
    print(f"     median log10(M_BH/M_bulge) = {np.median(log_ratio):.3f}")
    print(f"     mean   log10(M_BH/M_bulge) = {np.mean(log_ratio):.3f}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
compare_bulge_bh_mass.py
========================
Compare matched galaxies between two SAGE26 model_0.hdf5 outputs.

Selects galaxies from the first model with log10(BulgeMass/Msun) close to a
target value (default 10.5), looks up those exact galaxies (by GalaxyIndex/
GalaxyID) in the second model, and plots the change in BlackHoleMass against
the change in BulgeMass (both in dex, second run minus first) between the
two runs.

Usage
-----
  python3 compare_bulge_bh_mass.py dir1 dir2
  python3 compare_bulge_bh_mass.py dir1 dir2 -s 63 --window 0.1 -o plots/
"""

import argparse
import os
import sys

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HUBBLE_H_DEFAULT = 0.73
TARGET_LOG_BULGE_MASS = 10.5
DEFAULT_WINDOW = 0.25
POSSIBLE_ID_FIELDS = ['GalaxyIndex', 'GalaxyID', 'ID', 'galaxy_id', 'id', 'GalID']


def find_model_file(directory):
    path = os.path.join(directory, 'model_0.hdf5')
    if not os.path.isfile(path):
        print(f"Error: {path} not found.")
        sys.exit(1)
    return path


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


def find_id_field(filepath, snap_key):
    with h5py.File(filepath, 'r') as hf:
        if snap_key not in hf:
            return None
        for c in POSSIBLE_ID_FIELDS:
            if c in hf[snap_key]:
                return c
    return None


def read_fields(filepath, snap_key, fields):
    out = {}
    with h5py.File(filepath, 'r') as hf:
        if snap_key not in hf:
            print(f"Error: {snap_key} not found in {filepath}.")
            sys.exit(1)
        grp = hf[snap_key]
        for fld in fields:
            if fld not in grp:
                print(f"Error: field '{fld}' not found in {filepath}:{snap_key}.")
                sys.exit(1)
            out[fld] = np.array(grp[fld])
    return out


def main():
    p = argparse.ArgumentParser(
        description="Compare matched galaxies' BulgeMass/BlackHoleMass between two SAGE26 runs.")
    p.add_argument('dir1', help="First model directory (contains model_0.hdf5) - baseline for galaxy selection.")
    p.add_argument('dir2', help="Second model directory (contains model_0.hdf5) - compared against dir1.")
    p.add_argument('-s', '--snapshot', type=int, default=None,
                   help="Snapshot number to compare (default: latest available in dir1).")
    p.add_argument('--target-log-bulge-mass', type=float, default=TARGET_LOG_BULGE_MASS,
                   help=f"Target log10(BulgeMass/Msun) to select around (default: {TARGET_LOG_BULGE_MASS}).")
    p.add_argument('--window', type=float, default=DEFAULT_WINDOW,
                   help=f"Half-width of the selection window in dex (default: {DEFAULT_WINDOW} -> "
                        f"+/-{DEFAULT_WINDOW} dex).")
    p.add_argument('-o', '--output-dir', default=None,
                   help="Output directory for the plot (default: current directory).")
    args = p.parse_args()

    file1 = find_model_file(args.dir1)
    file2 = find_model_file(args.dir2)

    sim1 = read_simulation_params(file1)
    hubble_h = sim1['Hubble_h']

    snap_num = args.snapshot if args.snapshot is not None else sim1['latest_snapshot']
    if snap_num is None:
        print(f"Error: no Snap_* groups found in {file1}.")
        sys.exit(1)
    snap_key = f"Snap_{snap_num}"

    id_field = find_id_field(file1, snap_key)
    if id_field is None:
        print(f"Error: no galaxy-ID field ({', '.join(POSSIBLE_ID_FIELDS)}) found in {file1}:{snap_key}.")
        sys.exit(1)

    conv = 1.0e10 / hubble_h

    data1 = read_fields(file1, snap_key, [id_field, 'BulgeMass', 'BlackHoleMass'])
    bulge1 = data1['BulgeMass'] * conv
    bh1 = data1['BlackHoleMass'] * conv
    gid1 = data1[id_field]

    valid1 = (bulge1 > 0) & np.isfinite(bulge1)
    log_bulge1 = np.full_like(bulge1, np.nan)
    log_bulge1[valid1] = np.log10(bulge1[valid1])

    lo = args.target_log_bulge_mass - args.window
    hi = args.target_log_bulge_mass + args.window
    sel = valid1 & (log_bulge1 >= lo) & (log_bulge1 <= hi)

    n_sel = int(np.sum(sel))
    print(f"Selected {n_sel} galaxies in {file1} with log10(BulgeMass) in [{lo:.2f}, {hi:.2f}].")
    if n_sel == 0:
        print("Error: no galaxies pass the selection window.")
        sys.exit(1)

    target_ids = gid1[sel]
    target_bulge1 = bulge1[sel]
    target_bh1 = bh1[sel]

    data2 = read_fields(file2, snap_key, [id_field, 'BulgeMass', 'BlackHoleMass'])
    bulge2 = data2['BulgeMass'] * conv
    bh2 = data2['BlackHoleMass'] * conv
    gid2 = data2[id_field]

    order = np.argsort(gid2)
    gid2_sorted = gid2[order]
    pos = np.searchsorted(gid2_sorted, target_ids)
    pos = np.clip(pos, 0, len(gid2_sorted) - 1)
    found = gid2_sorted[pos] == target_ids
    n_matched = int(np.sum(found))
    print(f"Matched {n_matched}/{n_sel} of those galaxies by {id_field} in {file2}.")
    if n_matched == 0:
        print("Error: none of the selected galaxies were found in the second file.")
        sys.exit(1)

    idx2 = order[pos[found]]
    bulge1_m = target_bulge1[found]
    bh1_m = target_bh1[found]
    bulge2_m = bulge2[idx2]
    bh2_m = bh2[idx2]

    mask = (bulge1_m > 0) & (bulge2_m > 0) & (bh1_m > 0) & (bh2_m > 0)
    n_plot = int(np.sum(mask))
    print(f"{n_plot}/{n_matched} matched galaxies have positive masses in both files (plotted).")
    if n_plot == 0:
        print("Error: no matched galaxies have positive BulgeMass/BlackHoleMass in both files.")
        sys.exit(1)

    d_bulge = np.log10(bulge2_m[mask]) - np.log10(bulge1_m[mask])
    d_bh = np.log10(bh2_m[mask]) - np.log10(bh1_m[mask])

    output_dir = args.output_dir or '.'
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.34, 6.25))
    ax.axhline(0, color='grey', lw=1, ls='--', alpha=0.7)
    ax.axvline(0, color='grey', lw=1, ls='--', alpha=0.7)
    ax.scatter(d_bulge, d_bh, s=14, c='#1976D2', alpha=0.7, edgecolor='k', linewidth=0.3)
    ax.set_xlabel(r'$\Delta\log_{10}(M_{\rm bulge})$  [dex]', fontsize=14)
    ax.set_ylabel(r'$\Delta\log_{10}(M_{\rm BH})$  [dex]', fontsize=14)
    ax.set_title(
        rf'Matched galaxies near $\log_{{10}}M_{{\rm bulge}}\sim{args.target_log_bulge_mass:.1f}$'
        f' (snap {snap_num}, N={n_plot})', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(output_dir, 'bulge_bh_mass_diff.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)

    print(f"[ok] plot -> {out}")
    print(f"     median dBulge = {np.median(d_bulge):+.3f} dex   "
          f"median dBH = {np.median(d_bh):+.3f} dex")


if __name__ == "__main__":
    main()

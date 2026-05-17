#!/usr/bin/env python3
"""
Plot the spatial distribution of galaxies and dark matter halos at z~3.
Produces six figures: XY, XZ, and YZ projections for galaxies and for halos.

Usage:
    python spatial_distribution.py <output_folder>
    python spatial_distribution.py output/millennium/
    python spatial_distribution.py output/millennium/ --redshift 3.0
    python spatial_distribution.py output/millennium/ --snapshot 27
    python spatial_distribution.py output/millennium/ --tree-file input/millennium/trees/trees_STC.hdf5
"""

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
import glob
import os
import sys
from scipy.integrate import quad


def truncated_cmap(cmap, minval=0.0, maxval=1.0, n=256):
    return mcolors.LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(minval, maxval, n))
    )


def read_header(filepath):
    """Extract all required parameters from the HDF5 header."""
    params = {}
    with h5.File(filepath, 'r') as f:
        sim = f['Header/Simulation']
        params['hubble_h']    = float(sim.attrs['hubble_h'])
        params['box_size']    = float(sim.attrs['box_size'])    # Mpc/h
        params['particle_mass'] = float(sim.attrs['particle_mass'])  # 10^10 Msun/h

        params['snapshot_redshifts'] = np.array(f['Header/snapshot_redshifts'])
        params['save_full_sfh']      = int(f['Header/Runtime'].attrs.get('SaveFullSFH', 0))
        params['omega_matter']       = float(sim.attrs['omega_matter'])
        params['omega_lambda']       = float(sim.attrs['omega_lambda'])

        snap_groups = [k for k in f.keys() if k.startswith('Snap_')]
        params['available_snapshots'] = sorted(int(s.replace('Snap_', '')) for s in snap_groups)

    return params


def snapshot_times_gyr(redshifts, omega_m, omega_l, hubble_h):
    """Return cosmic time in Gyr for each entry in the redshifts array."""
    H0_inv_gyr = 9.778 / hubble_h  # 1/H0 in Gyr
    def integrand(a):
        return 1.0 / (a * np.sqrt(omega_m / a**3 + omega_l))
    times = np.empty(len(redshifts))
    for i, z in enumerate(redshifts):
        t, _ = quad(integrand, 0.0, 1.0 / (1.0 + z))
        times[i] = H0_inv_gyr * t
    return times


def find_snapshot(params, target_z=3.0, snap_override=None):
    """Return the snapshot number closest to target_z, or snap_override if given."""
    if snap_override is not None:
        return snap_override, params['snapshot_redshifts'][snap_override]

    zs = params['snapshot_redshifts']
    available = np.array(params['available_snapshots'])
    dz = np.abs(zs[available] - target_z)
    idx = np.argmin(dz)
    snap = int(available[idx])
    return snap, float(zs[snap])


def load_galaxies(filepaths, snap_num, particle_mass, min_len, quench_snaps, snap_times):
    """
    Read Posx/y/z, StellarMass, quiescence flag, and mass-weighted stellar age.

    Quiescence: SFR=0 at snap_num, plus zero SFH mass over the preceding
    (quench_snaps-1) bins (requires SaveFullSFH=1).

    Stellar age: mass-weighted mean age from SFHMassDisk+SFHMassBulge using
    snap_times (Gyr) for the formation time of each bin. NaN when SFH unavailable.
    """
    px, py, pz, sm, mvir, gtype, quiescent, ages = [], [], [], [], [], [], [], []
    snap_key = f'Snap_{snap_num}'
    t_now = snap_times[snap_num]

    for fp in filepaths:
        with h5.File(fp, 'r') as f:
            if snap_key not in f:
                continue
            g = f[snap_key]
            if 'StellarMass' not in g or 'Posx' not in g:
                continue
            stellar = np.array(g['StellarMass'], dtype=np.float64)
            length  = np.array(g['Len'], dtype=np.int32)
            mask = (stellar >= particle_mass) & (length >= min_len)
            if mask.sum() == 0:
                continue

            sfr = (np.array(g['SfrDisk'],  dtype=np.float64)[mask] +
                   np.array(g['SfrBulge'], dtype=np.float64)[mask])
            is_quiescent = sfr == 0.0

            if 'SFHMassDisk' in g:
                sfhd = np.array(g['SFHMassDisk'], dtype=np.float64)[mask]   # (N, n_bins)
                sfhb = np.array(g['SFHMassBulge'], dtype=np.float64)[mask]
                sfh  = sfhd + sfhb  # (N, n_bins)

                if quench_snaps > 1:
                    bin_lo = max(0, snap_num - quench_snaps + 1)
                    is_quiescent &= sfh[:, bin_lo:snap_num].sum(axis=1) == 0.0

                # Mass-weighted stellar age: bins 0..snap_num-1 are populated
                # bin i = mass formed in interval ending at snap i
                bin_ages  = t_now - snap_times[:snap_num]       # (snap_num,)  age in Gyr
                sfh_pop   = sfh[:, :snap_num]                   # (N, snap_num)
                total_sfh = sfh_pop.sum(axis=1)
                mw_age = np.where(
                    total_sfh > 0,
                    (sfh_pop * bin_ages[np.newaxis, :]).sum(axis=1) / total_sfh,
                    np.nan
                )
            else:
                mw_age = np.full(mask.sum(), np.nan)

            px.append(np.array(g['Posx'])[mask])
            py.append(np.array(g['Posy'])[mask])
            pz.append(np.array(g['Posz'])[mask])
            sm.append(stellar[mask])
            mvir.append(np.array(g['Mvir'], dtype=np.float64)[mask])
            gtype.append(np.array(g['Type'], dtype=np.int32)[mask])
            quiescent.append(is_quiescent)
            ages.append(mw_age)

    if not px:
        return None, None, None, None, None, None, None, None

    return (np.concatenate(px), np.concatenate(py),
            np.concatenate(pz), np.concatenate(sm),
            np.concatenate(mvir), np.concatenate(gtype),
            np.concatenate(quiescent), np.concatenate(ages))


def save_quiescent_csv(filepaths, snap_num, particle_mass, min_len, quench_snaps,
                       snap_times, hubble_h, out_path):
    """
    Write all scalar galaxy fields for quiescent galaxies at snap_num to a CSV.
    Applies the same selection and quiescence logic as load_galaxies.
    Adds derived columns: Posx_Mpc, Posy_Mpc, Posz_Mpc, log_StellarMass_Msun,
    log_Mvir_Msun, mw_stellar_age_Gyr, quiescent_class.
    """
    import csv

    SKIP_FIELDS = {'SFHMassDisk', 'SFHMassBulge'}   # 2-D arrays, not suitable for flat CSV

    snap_key = f'Snap_{snap_num}'
    t_now    = snap_times[snap_num]

    # collect field names from the first file that has the snapshot
    field_names = None
    for fp in filepaths:
        with h5.File(fp, 'r') as f:
            if snap_key in f:
                field_names = [k for k in sorted(f[snap_key].keys())
                               if k not in SKIP_FIELDS and f[snap_key][k].ndim == 1]
                break
    if field_names is None:
        print('  No snapshot found for CSV export.')
        return

    derived = ['Posx_Mpc', 'Posy_Mpc', 'Posz_Mpc',
               'log_StellarMass_Msun', 'log_Mvir_Msun',
               'mw_stellar_age_Gyr', 'quiescent_class']

    rows = []
    for fp in filepaths:
        with h5.File(fp, 'r') as f:
            if snap_key not in f:
                continue
            g = f[snap_key]
            stellar = np.array(g['StellarMass'], dtype=np.float64)
            length  = np.array(g['Len'],         dtype=np.int32)
            mask = (stellar >= particle_mass) & (length >= min_len)
            if mask.sum() == 0:
                continue

            sfr = (np.array(g['SfrDisk'],  dtype=np.float64)[mask] +
                   np.array(g['SfrBulge'], dtype=np.float64)[mask])
            is_q = sfr == 0.0

            if 'SFHMassDisk' in g:
                sfhd = np.array(g['SFHMassDisk'], dtype=np.float64)[mask]
                sfhb = np.array(g['SFHMassBulge'], dtype=np.float64)[mask]
                sfh  = sfhd + sfhb
                if quench_snaps > 1:
                    bin_lo = max(0, snap_num - quench_snaps + 1)
                    is_q &= sfh[:, bin_lo:snap_num].sum(axis=1) == 0.0
                bin_ages = t_now - snap_times[:snap_num]
                sfh_pop  = sfh[:, :snap_num]
                total    = sfh_pop.sum(axis=1)
                mw_age   = np.where(total > 0,
                                    (sfh_pop * bin_ages).sum(axis=1) / total,
                                    np.nan)
            else:
                mw_age = np.full(mask.sum(), np.nan)

            q_idx = np.where(mask)[0][is_q]
            if len(q_idx) == 0:
                continue

            # load all scalar fields for quiescent galaxies
            raw = {k: np.array(g[k])[q_idx] for k in field_names}
            mw_age_q = mw_age[is_q]

            sm_q   = np.array(g['StellarMass'], dtype=np.float64)[q_idx]
            mvir_q = np.array(g['Mvir'],        dtype=np.float64)[q_idx]
            px_q   = np.array(g['Posx'])[q_idx] / hubble_h
            py_q   = np.array(g['Posy'])[q_idx] / hubble_h
            pz_q   = np.array(g['Posz'])[q_idx] / hubble_h
            log_sm  = np.where(sm_q   > 0, np.log10(sm_q   * 1e10 / hubble_h), np.nan)
            log_mv  = np.where(mvir_q > 0, np.log10(mvir_q * 1e10 / hubble_h), np.nan)
            q_class = np.where(mw_age_q > 1.0, 'old_>1Gyr', 'young_<1Gyr')

            for i in range(len(q_idx)):
                row = {k: raw[k][i] for k in field_names}
                row['Posx_Mpc']            = px_q[i]
                row['Posy_Mpc']            = py_q[i]
                row['Posz_Mpc']            = pz_q[i]
                row['log_StellarMass_Msun'] = log_sm[i]
                row['log_Mvir_Msun']        = log_mv[i]
                row['mw_stellar_age_Gyr']   = mw_age_q[i]
                row['quiescent_class']      = q_class[i]
                rows.append(row)

    if not rows:
        print('  No quiescent galaxies to write.')
        return

    columns = field_names + derived
    with open(out_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved quiescent catalogue ({len(rows):,} galaxies): {out_path}')


def print_quiescent_table(px, py, pz, sm, mvir, gtype, quiescent, ages, hubble_h):
    """Print a summary table of quiescent galaxy properties to stdout."""
    if quiescent.sum() == 0:
        return
    q = quiescent
    type_label = {0: 'Central', 1: 'Satellite', 2: 'Orphan'}
    log_sm = np.log10(sm[q]   * 1e10 / hubble_h)
    log_mv = np.log10(mvir[q] * 1e10 / hubble_h)
    age    = ages[q]
    cls    = np.where(age > 1.0, 'old >1 Gyr', 'young <1 Gyr')
    gt     = gtype[q]
    gx, gy, gz = px[q] / hubble_h, py[q] / hubble_h, pz[q] / hubble_h

    hdrs   = ['#', 'Type', 'log M* [Msun]', 'log Mvir [Msun]', 'Age [Gyr]', 'Class', 'X [Mpc]', 'Y [Mpc]', 'Z [Mpc]']
    widths = [4,   10,     14,              15,                 10,          13,      9,          9,          9]
    sep    = '+-' + '-+-'.join('-' * w for w in widths) + '-+'
    hdr    = '| ' + ' | '.join(f'{h:^{w}}' for h, w in zip(hdrs, widths)) + ' |'

    print(f'\nQuiescent galaxies ({q.sum():,} total):')
    print(sep); print(hdr); print(sep)
    for i in range(q.sum()):
        tname   = type_label.get(int(gt[i]), str(int(gt[i])))
        age_str = f'{age[i]:.2f}' if np.isfinite(age[i]) else 'N/A'
        vals    = [str(i+1), tname, f'{log_sm[i]:.3f}', f'{log_mv[i]:.3f}',
                   age_str, cls[i], f'{gx[i]:.2f}', f'{gy[i]:.2f}', f'{gz[i]:.2f}']
        print('| ' + ' | '.join(f'{v:>{w}}' for v, w in zip(vals, widths)) + ' |')
    print(sep)


def plot_quiescent_sfh(filepaths, snap_num, particle_mass, min_len, quench_snaps,
                       snap_times, hubble_h, out_path, z_snap, run_label):
    """
    Plot the star formation history of quiescent galaxies vs lookback time.

    SFH bin j spans [snap_times[j-1], snap_times[j]] (or [0, snap_times[0]] for j=0).
    SFR [M☉/yr] = sfh[j] × 1e10/h / (bin_width[j] × 1e9).
    """
    snap_key = f'Snap_{snap_num}'
    t_now    = snap_times[snap_num]

    # bin widths: bin j ends at snap_times[j], starts at snap_times[j-1]
    t_edges    = np.concatenate([[0.0], snap_times[:snap_num]])   # length snap_num+1
    bin_widths = np.diff(t_edges)                                  # length snap_num, in Gyr
    lookback   = t_now - snap_times[:snap_num]                     # Gyr, decreasing

    sfh_list, sm_list, mv_list, age_list = [], [], [], []

    for fp in filepaths:
        with h5.File(fp, 'r') as f:
            if snap_key not in f or 'SFHMassDisk' not in f[snap_key]:
                continue
            g = f[snap_key]
            stellar = np.array(g['StellarMass'], dtype=np.float64)
            length  = np.array(g['Len'],         dtype=np.int32)
            mask = (stellar >= particle_mass) & (length >= min_len)
            if mask.sum() == 0:
                continue

            sfr = (np.array(g['SfrDisk'],  dtype=np.float64)[mask] +
                   np.array(g['SfrBulge'], dtype=np.float64)[mask])
            is_q = sfr == 0.0

            sfhd = np.array(g['SFHMassDisk'],  dtype=np.float64)[mask]
            sfhb = np.array(g['SFHMassBulge'], dtype=np.float64)[mask]
            sfh  = sfhd + sfhb

            if quench_snaps > 1:
                bin_lo = max(0, snap_num - quench_snaps + 1)
                is_q &= sfh[:, bin_lo:snap_num].sum(axis=1) == 0.0

            if is_q.sum() == 0:
                continue

            sfh_q = sfh[is_q, :snap_num]          # (Nq, snap_num), 10^10 Msun/h
            sm_q  = stellar[mask][is_q]
            mv_q  = np.array(g['Mvir'], dtype=np.float64)[mask][is_q]

            bin_ages  = t_now - snap_times[:snap_num]
            sfh_pop   = sfh_q
            total_sfh = sfh_pop.sum(axis=1)
            mw_age = np.where(total_sfh > 0,
                               (sfh_pop * bin_ages).sum(axis=1) / total_sfh,
                               np.nan)

            sfh_list.append(sfh_q)
            sm_list.append(sm_q)
            mv_list.append(mv_q)
            age_list.append(mw_age)

    if not sfh_list:
        print('  No SFH data for quiescent galaxies — skipping SFH plot.')
        return

    all_sfh = np.concatenate(sfh_list, axis=0)   # (Nq_total, snap_num)
    all_sm  = np.concatenate(sm_list)
    all_mv  = np.concatenate(mv_list)
    all_age = np.concatenate(age_list)

    # SFR in M☉/yr: mass [10^10 Msun/h] / bin_width [Gyr] → Msun/yr
    sfr_arr = all_sfh * 1e10 / hubble_h / (bin_widths * 1e9)   # (Nq, snap_num)

    log_sm_arr = np.log10(all_sm * 1e10 / hubble_h)
    sm_norm    = mcolors.Normalize(vmin=log_sm_arr.min(), vmax=log_sm_arr.max())
    cmap       = plt.cm.viridis

    fig, ax = plt.subplots(figsize=(8, 5))

    for i in range(len(all_sfh)):
        color = cmap(sm_norm(log_sm_arr[i]))
        ax.plot(lookback, sfr_arr[i], color=color, alpha=0.7, lw=1.2)

    if len(all_sfh) > 1:
        median_sfr = np.nanmedian(sfr_arr, axis=0)
        ax.plot(lookback, median_sfr, color='white', lw=2.5, ls='--', zorder=10, label='Median')
        ax.legend(fontsize=10)

    ax.set_xlabel('Lookback time [Gyr]', fontsize=13)
    ax.set_ylabel(r'SFR [$M_\odot\,\mathrm{yr}^{-1}$]', fontsize=13)
    ax.set_title(f'{run_label}  |  Quiescent galaxy SFH  |  z = {z_snap:.2f}', fontsize=12)
    ax.set_xlim(lookback.max(), 0)   # past on the left
    ax.set_ylim(bottom=0)

    sm_sm = plt.cm.ScalarMappable(cmap=cmap, norm=sm_norm)
    sm_sm.set_array([])
    cbar = fig.colorbar(sm_sm, ax=ax, pad=0.02)
    cbar.set_label(r'$\log_{10}(M_*\,/\,M_\odot)$', fontsize=12)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def load_halos_from_trees(tree_file, snap_num, hubble_h, min_len=0):
    """
    Load FoF central dark matter halo positions and M_Crit200 from an STC HDF5 tree file.

    Centrals: halos where FirstHaloInFOFGroup[i] == i (they are their own FoF first subhalo).
    Positions: kpc/h in file, returned in Mpc/h.
    Masses: stored in 10^10 Msun/h (same as SAGE Mvir) → log10(Msun).
    """
    px_list, py_list, pz_list, mv_list = [], [], [], []

    with h5.File(tree_file, 'r') as f:
        tree_keys = [k for k in f.keys() if k.startswith('Tree')]
        for tkey in tree_keys:
            t = f[tkey]
            snaps = np.array(t['SnapNum'], dtype=np.int32)
            all_idx = np.where(snaps == snap_num)[0]
            if len(all_idx) == 0:
                continue

            fhifog = np.array(t['FirstHaloInFOFGroup'], dtype=np.int32)
            central_mask = fhifog[all_idx] == all_idx
            idx = all_idx[central_mask]

            if min_len > 0:
                lengths = np.array(t['SubhaloLen'], dtype=np.int32)
                idx = idx[lengths[idx] >= min_len]

            if len(idx) == 0:
                continue

            pos = np.array(t['SubhaloPos'])          # (N_tree, 3) kpc/h
            mass = np.array(t['Group_M_Crit200'], dtype=np.float64)

            px_list.append(pos[idx, 0])
            py_list.append(pos[idx, 1])
            pz_list.append(pos[idx, 2])
            mv_list.append(mass[idx])

    if not px_list:
        return None, None, None, None

    px = np.concatenate(px_list) / 1000.0   # kpc/h → Mpc/h
    py = np.concatenate(py_list) / 1000.0
    pz = np.concatenate(pz_list) / 1000.0
    mv = np.concatenate(mv_list)

    # 10^10 Msun/h → Msun (same unit as SAGE Mvir)
    mv_msun = mv * 1e10 / hubble_h
    valid = mv_msun > 0
    log_mv = np.where(valid, np.log10(np.where(valid, mv_msun, 1.0)), np.nan)

    return px, py, pz, log_mv


def make_projection(ax, x, y, color_val, quiescent, ages, xlabel, ylabel, box_mpc, cmap, norm, dot_size):
    sf = ~quiescent
    sc = ax.scatter(x[sf], y[sf], c=color_val[sf], cmap=cmap, norm=norm,
                    s=dot_size, alpha=0.8, linewidths=0, rasterized=True)
    if quiescent.sum() > 0:
        old  = quiescent & (ages > 1.0)   # gold: mean stellar age > 1 Gyr
        young = quiescent & ~old           # red:  younger or no age info
        if young.sum() > 0:
            ax.scatter(x[young], y[young], c='red', marker='*',
                       s=dot_size * 30, alpha=0.9, linewidths=0,
                       zorder=5, rasterized=True, label='Quiescent (<1 Gyr)')
        if old.sum() > 0:
            ax.scatter(x[old], y[old], c='gold', marker='*',
                       s=dot_size * 30, alpha=0.9, linewidths=0,
                       zorder=6, rasterized=True, label='Quiescent (>1 Gyr)')
    ax.set_xlim(0, box_mpc)
    ax.set_ylim(0, box_mpc)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_aspect('equal')
    return sc


def make_halo_projection(ax, x, y, color_val, xlabel, ylabel, box_mpc, cmap, norm, dot_size,
                         gal_x=None, gal_y=None, quiescent=None, ages=None):
    sc = ax.scatter(x, y, c=color_val, cmap=cmap, norm=norm,
                    s=dot_size, alpha=0.8, linewidths=0, rasterized=True)
    if gal_x is not None and quiescent is not None and quiescent.sum() > 0:
        old   = quiescent & (ages > 1.0)
        young = quiescent & ~old
        if young.sum() > 0:
            ax.scatter(gal_x[young], gal_y[young], c='red', marker='*',
                       s=dot_size * 30, alpha=0.9, linewidths=0,
                       zorder=5, rasterized=True, label='Quiescent (<1 Gyr)')
        if old.sum() > 0:
            ax.scatter(gal_x[old], gal_y[old], c='gold', marker='*',
                       s=dot_size * 30, alpha=0.9, linewidths=0,
                       zorder=6, rasterized=True, label='Quiescent (>1 Gyr)')
    ax.set_xlim(0, box_mpc)
    ax.set_ylim(0, box_mpc)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_aspect('equal')
    return sc


def parse_args():
    p = argparse.ArgumentParser(description='Spatial distribution of galaxies at z~3')
    p.add_argument('output_folder',
                   help='Folder containing model_*.hdf5 files (e.g. output/millennium/)')
    p.add_argument('--redshift', type=float, default=3.0,
                   help='Target redshift (default: 3.0)')
    p.add_argument('--snapshot', type=int, default=None,
                   help='Override: use this snapshot number directly')
    p.add_argument('--dot-size', type=float, default=1.5,
                   help='Scatter point size (default: 1.5)')
    p.add_argument('--min-len', type=int, default=50,
                   help='Minimum halo particle count (default: 50)')
    p.add_argument('--quench-snaps', type=int, default=1,
                   help='Number of consecutive snapshots with SFR=0 to classify as quiescent (default: 1)')
    p.add_argument('--output-dir', type=str, default=None,
                   help='Where to save figures (default: <output_folder>/plots/)')
    p.add_argument('--tree-file', type=str, default=None,
                   help='STC HDF5 tree file for halo plots (e.g. input/millennium/trees/trees_STC.hdf5); '
                        'skips halo plots if not found')
    return p.parse_args()


def main():
    args = parse_args()

    # --- find files ---
    pattern = os.path.join(args.output_folder, 'model_*.hdf5')
    filepaths = sorted(glob.glob(pattern))
    if not filepaths:
        sys.exit(f'No model_*.hdf5 files found in {args.output_folder}')

    print(f'Found {len(filepaths)} HDF5 file(s) in {args.output_folder}')

    # --- read header from first file ---
    params = read_header(filepaths[0])
    h = params['hubble_h']
    box_mpc_h = params['box_size']        # Mpc/h
    box_mpc   = box_mpc_h / h             # physical Mpc

    # --- choose snapshot ---
    snap, z_snap = find_snapshot(params, target_z=args.redshift,
                                 snap_override=args.snapshot)
    print(f'Using Snap_{snap}  (z = {z_snap:.4f},'
          f' requested z ~ {args.redshift if args.snapshot is None else "override"})')

    particle_mass = params['particle_mass']
    print(f'Particle mass: {particle_mass:.4f} x10^10 Msun/h'
          f'  (floor = {particle_mass*1e10/h:.3e} Msun)')

    has_sfh = bool(params['save_full_sfh'])
    effective_quench = args.quench_snaps
    if args.quench_snaps > 1 and not has_sfh:
        print(f'  Warning: SaveFullSFH=0 — SFH arrays unavailable, falling back to quench_snaps=1')
        effective_quench = 1
    print(f'Cuts: Len >= {args.min_len},  quiescence over {effective_quench} snapshot(s)')

    # --- precompute cosmic time at every snapshot ---
    snap_times = snapshot_times_gyr(
        params['snapshot_redshifts'],
        params['omega_matter'], params['omega_lambda'], h
    )
    print(f'Cosmic time at selected snapshot: {snap_times[snap]:.3f} Gyr')

    # --- load galaxies ---
    px, py, pz, sm, mvir, gtype, quiescent, ages = load_galaxies(
        filepaths, snap, particle_mass, args.min_len, args.quench_snaps, snap_times)
    if px is None:
        sys.exit(f'No galaxies with StellarMass > 0 found in Snap_{snap}.')

    # Convert positions: Mpc/h → Mpc
    px /= h;  py /= h;  pz /= h

    # Stellar mass: 10^10 M_sun/h → log10(M_sun)
    log_sm = np.log10(sm * 1e10 / h)

    n_sf   = (~quiescent).sum()
    n_q    = quiescent.sum()
    n_gold = (quiescent & (ages > 1.0)).sum()
    n_red  = n_q - n_gold
    print(f'Loaded {len(px):,} galaxies  '
          f'(log M* range: {log_sm.min():.2f} – {log_sm.max():.2f})')
    print(f'  Star-forming: {n_sf:,}   Quiescent: {n_q:,}'
          f'  (red <1Gyr: {n_red:,}, gold >1Gyr: {n_gold:,})')

    # --- colormap & normalisation ---
    sf_mask = ~quiescent
    base_arr = log_sm[sf_mask] if sf_mask.any() else log_sm
    vmin, vmax = np.percentile(base_arr, [2, 98])
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.viridis

    # --- output directory ---
    out_dir = args.output_dir or os.path.join(args.output_folder, 'plots')
    os.makedirs(out_dir, exist_ok=True)

    # --- derive a run label from the folder name ---
    run_label = os.path.basename(os.path.normpath(args.output_folder))

    projections = [
        ('Posx', 'Posy', px, py, 'X [Mpc]',   'Y [Mpc]',   'XY'),
        ('Posx', 'Posz', px, pz, 'X [Mpc]',   'Z [Mpc]',   'XZ'),
        ('Posy', 'Posz', py, pz, 'Y [Mpc]',   'Z [Mpc]',   'YZ'),
    ]

    for _, _, xa, ya, xlabel, ylabel, tag in projections:
        fig, ax = plt.subplots(figsize=(7, 6.5))
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        sc = make_projection(ax, xa, ya, log_sm, quiescent, ages, xlabel, ylabel,
                             box_mpc, cmap, norm, args.dot_size)

        ax.set_title(f'{run_label}  |  {tag} projection  |  z = {z_snap:.2f}',
                     fontsize=12, color='white')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')

        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label(r'$\log_{10}(M_*\,/\,M_\odot)$', fontsize=12, color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

        if quiescent.sum() > 0:
            ax.legend(loc='upper right', fontsize=9, framealpha=0.3,
                            facecolor='black', edgecolor='white', labelcolor='white')


        fig.tight_layout()
        fname = os.path.join(out_dir, f'spatial_{tag}_z{z_snap:.2f}_{run_label}.pdf')
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {fname}')

    # --- quiescent galaxy table + CSV + SFH plot ---
    if quiescent.sum() > 0:
        print_quiescent_table(px, py, pz, sm, mvir, gtype, quiescent, ages, h)

        csv_path = os.path.join(out_dir, f'quiescent_galaxies_z{z_snap:.2f}_{run_label}.csv')
        save_quiescent_csv(filepaths, snap, particle_mass, args.min_len, args.quench_snaps,
                           snap_times, h, csv_path)

        sfh_path = os.path.join(out_dir, f'quiescent_sfh_z{z_snap:.2f}_{run_label}.pdf')
        plot_quiescent_sfh(filepaths, snap, particle_mass, args.min_len, args.quench_snaps,
                           snap_times, h, sfh_path, z_snap, run_label)

    # --- halo projections from merger trees ---
    tree_file = args.tree_file
    if tree_file is None:
        # try to guess: same structure as output but under input/
        folder_name = os.path.basename(os.path.normpath(args.output_folder))
        candidate = os.path.join('input', folder_name, 'trees', 'trees_STC.hdf5')
        if os.path.isfile(candidate):
            tree_file = candidate

    if tree_file and os.path.isfile(tree_file):
        print(f'\nLoading halos from: {tree_file}')
        hpx, hpy, hpz, hlog_mv = load_halos_from_trees(
            tree_file, snap, h, min_len=args.min_len)

        if hpx is None:
            print(f'No FoF central halos found at Snap_{snap} in tree file.')
        else:
            # convert Mpc/h → Mpc
            hpx /= h;  hpy /= h;  hpz /= h

            valid = np.isfinite(hlog_mv)
            print(f'Loaded {len(hpx):,} FoF central halos  '
                  f'(log M_Crit200 range: {hlog_mv[valid].min():.2f} – {hlog_mv[valid].max():.2f})')

            hv_min, hv_max = np.percentile(hlog_mv[valid], [2, 98])
            hnorm = mcolors.Normalize(vmin=hv_min, vmax=hv_max)
            hcmap = plt.cm.viridis

            halo_projections = [
                (hpx, hpy, px, py, 'X [Mpc]', 'Y [Mpc]', 'XY'),
                (hpx, hpz, px, pz, 'X [Mpc]', 'Z [Mpc]', 'XZ'),
                (hpy, hpz, py, pz, 'Y [Mpc]', 'Z [Mpc]', 'YZ'),
            ]

            for hxa, hya, gxa, gya, xlabel, ylabel, tag in halo_projections:
                fig, ax = plt.subplots(figsize=(7, 6.5))
                fig.patch.set_facecolor('black')
                ax.set_facecolor('black')
                sc = make_halo_projection(ax, hxa, hya, hlog_mv, xlabel, ylabel,
                                          box_mpc, hcmap, hnorm, args.dot_size,
                                          gal_x=gxa, gal_y=gya,
                                          quiescent=quiescent, ages=ages)

                ax.set_title(f'{run_label}  |  Halos {tag}  |  z = {z_snap:.2f}',
                             fontsize=12, color='white')
                ax.tick_params(colors='white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                for spine in ax.spines.values():
                    spine.set_edgecolor('white')

                cbar = fig.colorbar(sc, ax=ax, pad=0.02)
                cbar.set_label(r'$\log_{10}(M_\mathrm{Crit200}\,/\,M_\odot)$',
                               fontsize=12, color='white')
                cbar.ax.yaxis.set_tick_params(color='white')
                plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

                if quiescent.sum() > 0:
                    ax.legend(loc='upper right', fontsize=9, framealpha=0.3,
                                    facecolor='black', edgecolor='white', labelcolor='white')

                fig.tight_layout()
                fname = os.path.join(out_dir, f'spatial_halos_{tag}_z{z_snap:.2f}_{run_label}.pdf')
                fig.savefig(fname, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f'Saved: {fname}')
    else:
        if args.tree_file:
            print(f'\nWarning: tree file not found: {args.tree_file}  — skipping halo plots.')
        else:
            print('\nNo tree file found — skipping halo plots. Use --tree-file to specify one.')

    print('Done.')


if __name__ == '__main__':
    main()

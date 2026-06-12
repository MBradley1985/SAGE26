#!/usr/bin/env python
"""
Central/satellite fraction of the most massive quiescent galaxies vs redshift,
overlaid for three SAGE26 models.

For each snapshot in each model, select the top 10% of galaxies by stellar mass,
flag those with sSFR < 0.2 * H(z) as quiescent, then plot the central and
satellite fractions of that quiescent sample as a function of redshift.

Usage:
    python plotting/quiescent_centsat_three_models.py \
        --model1 ./output/millennium/ \
        --model2 ./output/millennium_noCGM/ \
        --model3 ./output/millennium_oldCGMAGN/ \
        --label1 "fiducial" --label2 "no CGM" --label3 "old CGM AGN"
"""

import argparse
import glob
import os
import sys
import warnings

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings('ignore')

_STYLE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      'kieren_cohare_palatino_sty.mplstyle')
if os.path.exists(_STYLE):
    plt.style.use(_STYLE)
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'

OMEGA_M_DEFAULT = 0.25
OMEGA_L_DEFAULT = 0.75

MODEL_COLORS = ['#175cdb', '#d83a21', '#2ca02c']


def parse_arguments():
    p = argparse.ArgumentParser(
        description='Plot central/satellite fraction of most massive quiescent '
                    'galaxies vs redshift, for three SAGE26 models.')
    p.add_argument('--model1', required=True, help='Path to model 1 (directory or glob)')
    p.add_argument('--model2', required=True, help='Path to model 2 (directory or glob)')
    p.add_argument('--model3', required=True, help='Path to model 3 (directory or glob)')
    p.add_argument('--label1', default='Model 1')
    p.add_argument('--label2', default='Model 2')
    p.add_argument('--label3', default='Model 3')
    p.add_argument('--first-snap', type=int, default=None)
    p.add_argument('--last-snap', type=int, default=None)
    p.add_argument('-o', '--output-dir', default='./output/quiescent_evolution/')
    p.add_argument('--format', default='.pdf')
    p.add_argument('--top-percent', type=float, default=10.0,
                   help='Stellar-mass percentile cut for "most massive" (default 10). '
                        'Ignored when --absolute-mass-cut is set.')
    p.add_argument('--absolute-mass-cut', type=float, default=None,
                   metavar='LOG_MSTAR',
                   help='Use an absolute log10(M*/Msun) cut instead of a percentile. '
                        'Same threshold applied at every snapshot and every model.')
    p.add_argument('--ssfr-factor', type=float, default=0.2,
                   help='sSFR threshold factor: sSFR < factor * H(z) (default 0.2)')
    p.add_argument('-q', '--quiet', action='store_true',
                   help='Suppress per-snapshot diagnostic table')
    return p.parse_args()


def resolve_files(path):
    """Accept either a directory or a glob pattern; return sorted list of files."""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, 'model_*.hdf5')))
    else:
        files = sorted(glob.glob(path))
    return files


def read_header(files):
    with h5.File(files[0], 'r') as f:
        sim = f['Header/Simulation']
        hubble_h = float(sim.attrs['hubble_h'])
        omega_m = float(sim.attrs.get('omega_matter', sim.attrs.get('omega_m', OMEGA_M_DEFAULT)))
        omega_l = float(sim.attrs.get('omega_lambda', sim.attrs.get('omega_l', OMEGA_L_DEFAULT)))
        redshifts = np.array(f['Header/snapshot_redshifts'])

    snap_set = set()
    for fp in files:
        with h5.File(fp, 'r') as f:
            for k in f.keys():
                if k.startswith('Snap_'):
                    snap_set.add(int(k.replace('Snap_', '')))
    return {
        'hubble_h': hubble_h,
        'omega_m': omega_m,
        'omega_l': omega_l,
        'redshifts': redshifts,
        'snaps': sorted(snap_set),
    }


def _process_snapshot(files, snap, hubble_h, omega_m, omega_l, z,
                      top_percent, ssfr_factor, abs_log_mass=None):
    """
    Two-pass per snapshot:
      1. read StellarMass from each file (cheap, one array per file).
      2. compute global mass threshold, then for each file partial-read
         SfrDisk/SfrBulge/Type only at indices above threshold.

    If abs_log_mass is set the threshold is fixed at 10**abs_log_mass instead of
    the per-snapshot percentile.

    Returns a diagnostics dict, or None if the snapshot is unusable.
    """
    key = f'Snap_{snap}'
    inv_H0_yr = (9.778 / hubble_h) * 1.0e9
    mass_conv = 1.0e10 / hubble_h

    # Pass 1 -- StellarMass per file (raw code units).
    per_file_mass_raw = []
    for fp in files:
        with h5.File(fp, 'r') as f:
            if key in f and 'StellarMass' in f[key]:
                per_file_mass_raw.append(np.array(f[key]['StellarMass']))
            else:
                per_file_mass_raw.append(np.array([]))

    total_n = sum(arr.size for arr in per_file_mass_raw)
    if total_n == 0:
        return None

    n_pos = sum(int(np.sum(arr > 0.0)) for arr in per_file_mass_raw if arr.size)
    if n_pos < 10:
        return None

    if abs_log_mass is not None:
        mass_thresh = 10.0 ** abs_log_mass
    else:
        # Global mass threshold over positive-mass galaxies.
        thresh_pct = 100.0 - top_percent
        pos_chunks = [arr[arr > 0.0] for arr in per_file_mass_raw if arr.size]
        pos_concat = np.concatenate(pos_chunks) * mass_conv
        mass_thresh = float(np.percentile(pos_concat, thresh_pct))
        del pos_concat, pos_chunks

    E_z = np.sqrt(omega_m * (1.0 + z)**3 + omega_l)
    ssfr_thresh = ssfr_factor * E_z / inv_H0_yr

    # Pass 2 -- for each file, partial-read SFR/Type at massive indices only.
    n_massive = n_massive_cen = n_massive_sat = 0
    n_cen = n_sat = n_q = 0
    for fp, mass_raw in zip(files, per_file_mass_raw):
        if mass_raw.size == 0:
            continue
        mass_phys = mass_raw * mass_conv
        local_idx = np.where(mass_phys >= mass_thresh)[0]
        if local_idx.size == 0:
            continue
        n_massive += int(local_idx.size)
        mass_sel = mass_phys[local_idx]
        with h5.File(fp, 'r') as f:
            grp = f[key]
            # h5py fancy indexing needs sorted unique indices -- np.where gives that.
            sfr_disk = grp['SfrDisk'][local_idx]
            sfr_bulge = grp['SfrBulge'][local_idx]
            types_sel = grp['Type'][local_idx]
        n_massive_cen += int(np.sum(types_sel == 0))
        n_massive_sat += int(np.sum(types_sel == 1))
        sSFR = (sfr_disk + sfr_bulge) / mass_sel
        q_mask = sSFR < ssfr_thresh
        if not q_mask.any():
            continue
        q_types = types_sel[q_mask]
        n_q += int(q_mask.sum())
        n_cen += int(np.sum(q_types == 0))
        n_sat += int(np.sum(q_types == 1))

    return {
        'total_n': int(total_n),
        'n_pos': int(n_pos),
        'mass_thresh': float(mass_thresh),
        'ssfr_thresh': float(ssfr_thresh),
        'n_massive': n_massive,
        'n_massive_cen': n_massive_cen,
        'n_massive_sat': n_massive_sat,
        'n_q': n_q,
        'n_cen': n_cen,
        'n_sat': n_sat,
        'n_other': n_q - n_cen - n_sat,
    }


def compute_fractions(files, header, first_snap, last_snap, top_percent,
                      ssfr_factor, abs_log_mass=None, verbose=True):
    """Return dict with z/cen/sat arrays plus per-snapshot diagnostic arrays."""
    hubble_h = header['hubble_h']
    omega_m = header['omega_m']
    omega_l = header['omega_l']
    redshifts = header['redshifts']
    snap_set = set(header['snaps'])

    rows = []  # one (snap, z, diag) per usable snapshot

    if verbose:
        print(f'  {"snap":>4}  {"z":>6}  {"N_tot":>10}  '
              f'{"logM_thr":>9}  {"N_mas":>6}  {"N_cen":>6}  {"N_sat":>6}  '
              f'{"N_q":>6}  {"f_cen":>6}  {"f_sat":>6}  '
              f'{"fq_cen":>6}  {"fq_sat":>6}  {"note":<8}')

    for snap in range(first_snap, last_snap + 1):
        if snap not in snap_set:
            continue
        z = float(redshifts[snap])
        diag = _process_snapshot(
            files, snap, hubble_h, omega_m, omega_l, z,
            top_percent, ssfr_factor, abs_log_mass=abs_log_mass,
        )
        if diag is None:
            if verbose:
                print(f'  {snap:>4}  {z:6.2f}  {"-":>10}  {"-":>9}  '
                      f'{"-":>6}  {"-":>6}  {"-":>6}  {"-":>6}  '
                      f'{"-":>6}  {"-":>6}  {"-":>6}  {"-":>6}  empty')
            continue

        n_q = diag['n_q']
        n_mas = diag['n_massive']
        n_mas_c = diag['n_massive_cen']
        n_mas_s = diag['n_massive_sat']
        log_mthr = np.log10(diag['mass_thresh']) if diag['mass_thresh'] > 0 else np.nan

        f_cen = diag['n_cen'] / n_q if n_q > 0 else np.nan
        f_sat = diag['n_sat'] / n_q if n_q > 0 else np.nan
        fq_cen = diag['n_cen'] / n_mas_c if n_mas_c > 0 else np.nan
        fq_sat = diag['n_sat'] / n_mas_s if n_mas_s > 0 else np.nan

        note = 'no quies.' if n_q == 0 else ('low-N' if n_q < 20 else '')
        if verbose:
            def _f(x):
                return f'{x:6.3f}' if np.isfinite(x) else f'{"-":>6}'
            print(f'  {snap:>4}  {z:6.2f}  {diag["total_n"]:>10d}  '
                  f'{log_mthr:>9.2f}  {n_mas:>6d}  {n_mas_c:>6d}  {n_mas_s:>6d}  '
                  f'{n_q:>6d}  {_f(f_cen)}  {_f(f_sat)}  '
                  f'{_f(fq_cen)}  {_f(fq_sat)}  {note:<8}')

        if n_q == 0:
            continue
        rows.append((snap, z, f_cen, f_sat, fq_cen, fq_sat, diag))

    if not rows:
        empty = np.array([])
        return {
            'z': empty, 'cen': empty, 'sat': empty,
            'fq_cen': empty, 'fq_sat': empty,
            'snaps': np.array([], dtype=int),
            'n_q': np.array([], dtype=int),
            'n_massive': np.array([], dtype=int),
            'n_massive_cen': np.array([], dtype=int),
            'n_massive_sat': np.array([], dtype=int),
            'log_mthr': empty,
        }

    rows.sort(key=lambda r: r[1])
    z = np.array([r[1] for r in rows])
    cen = np.array([r[2] for r in rows])
    sat = np.array([r[3] for r in rows])
    fq_cen = np.array([r[4] for r in rows])
    fq_sat = np.array([r[5] for r in rows])
    snaps = np.array([r[0] for r in rows], dtype=int)
    n_q = np.array([r[6]['n_q'] for r in rows], dtype=int)
    n_massive = np.array([r[6]['n_massive'] for r in rows], dtype=int)
    n_mas_cen = np.array([r[6]['n_massive_cen'] for r in rows], dtype=int)
    n_mas_sat = np.array([r[6]['n_massive_sat'] for r in rows], dtype=int)
    log_mthr = np.array([np.log10(r[6]['mass_thresh']) if r[6]['mass_thresh'] > 0 else np.nan
                         for r in rows])
    return {
        'z': z, 'cen': cen, 'sat': sat,
        'fq_cen': fq_cen, 'fq_sat': fq_sat,
        'snaps': snaps, 'n_q': n_q, 'n_massive': n_massive,
        'n_massive_cen': n_mas_cen, 'n_massive_sat': n_mas_sat,
        'log_mthr': log_mthr,
    }


def _crossing_redshift(z, cen, sat):
    """Lowest z at which centrals = satellites (linear interp), or None."""
    diff = cen - sat
    # Walk from low z up; find first sign change.
    order = np.argsort(z)
    z_s, d_s = z[order], diff[order]
    for i in range(len(z_s) - 1):
        if d_s[i] == 0:
            return float(z_s[i])
        if d_s[i] * d_s[i + 1] < 0:
            # Linear interp where d crosses zero.
            t = d_s[i] / (d_s[i] - d_s[i + 1])
            return float(z_s[i] + t * (z_s[i + 1] - z_s[i]))
    return None


def _interp_at(z, y, z_target):
    """Return y interpolated to z_target, or None if outside the range."""
    if len(z) == 0:
        return None
    order = np.argsort(z)
    z_s, y_s = z[order], y[order]
    if z_target < z_s[0] or z_target > z_s[-1]:
        return None
    return float(np.interp(z_target, z_s, y_s))


def _print_model_summary(label, result):
    z, cen, sat, n_q = result['z'], result['cen'], result['sat'], result['n_q']
    fq_cen, fq_sat = result['fq_cen'], result['fq_sat']
    print(f'\n  {label} summary')
    print(f'    snapshots used     : {len(z)}')
    if len(z) == 0:
        print('    (no usable snapshots)')
        return
    print(f'    redshift range     : {z.min():.3f} -- {z.max():.3f}')
    print(f'    quiescent / snap   : min={n_q.min()}  max={n_q.max()}  '
          f'mean={n_q.mean():.1f}  total={n_q.sum()}')
    print(f'    massive / snap     : min={result["n_massive"].min()}  '
          f'max={result["n_massive"].max()}  '
          f'mean={result["n_massive"].mean():.1f}')
    print(f'    massive cen / sat  : '
          f'cen mean={result["n_massive_cen"].mean():.1f}  '
          f'sat mean={result["n_massive_sat"].mean():.1f}')
    print(f'    central fraction   : min={cen.min():.3f}  max={cen.max():.3f}  '
          f'mean={cen.mean():.3f}  median={np.median(cen):.3f}')
    print(f'    satellite fraction : min={sat.min():.3f}  max={sat.max():.3f}  '
          f'mean={sat.mean():.3f}  median={np.median(sat):.3f}')

    def _stats(arr, name):
        valid = arr[np.isfinite(arr)]
        if valid.size == 0:
            print(f'    {name:<18} : (no valid snapshots)')
        else:
            print(f'    {name:<18} : min={valid.min():.3f}  max={valid.max():.3f}  '
                  f'mean={valid.mean():.3f}  median={np.median(valid):.3f}')
    _stats(fq_cen, 'fq_cen (cen->q)')
    _stats(fq_sat, 'fq_sat (sat->q)')

    z_eq = _crossing_redshift(z, cen, sat)
    if z_eq is None:
        which = 'centrals dominate' if cen.mean() > sat.mean() else 'satellites dominate'
        print(f'    cen=sat crossing   : not crossed ({which})')
    else:
        print(f'    cen=sat crossing   : z = {z_eq:.3f}')


def _print_comparison_table(results):
    print('\nCross-model comparison -- quiescent composition (f_cen = N_q_cen / N_q)')
    header = f'  {"model":<22}  {"N":>3}  {"z_eq":>6}  {"f_cen(z~0)":>10}  ' \
             f'{"f_cen(z=2)":>10}  {"f_cen(z=5)":>10}'
    print(header)
    print('  ' + '-' * (len(header) - 2))
    for label, res in results:
        z, cen, sat = res['z'], res['cen'], res['sat']
        if len(z) == 0:
            print(f'  {label:<22}  {"-":>3}  {"-":>6}  {"-":>10}  {"-":>10}  {"-":>10}')
            continue
        z_eq = _crossing_redshift(z, cen, sat)
        z_eq_s = f'{z_eq:.2f}' if z_eq is not None else '-'

        def fmt_at(arr, z_target):
            v = _interp_at(z, arr, z_target)
            return f'{v:.3f}' if v is not None else '-'

        print(f'  {label:<22}  {len(z):>3d}  {z_eq_s:>6}  '
              f'{fmt_at(cen, 0.0):>10}  {fmt_at(cen, 2.0):>10}  {fmt_at(cen, 5.0):>10}')

    print('\nCross-model comparison -- per-population quiescent fractions')
    header2 = f'  {"model":<22}  {"fq_cen(z~0)":>11}  {"fq_cen(z=2)":>11}  ' \
              f'{"fq_cen(z=5)":>11}  {"fq_sat(z~0)":>11}  {"fq_sat(z=2)":>11}  ' \
              f'{"fq_sat(z=5)":>11}'
    print(header2)
    print('  ' + '-' * (len(header2) - 2))
    for label, res in results:
        z = res['z']
        if len(z) == 0:
            dash = f'{"-":>11}'
            print(f'  {label:<22}  {dash}  {dash}  {dash}  {dash}  {dash}  {dash}')
            continue
        fq_cen, fq_sat = res['fq_cen'], res['fq_sat']

        def fmt_at(arr, z_target):
            # Only interpolate where the array is finite.
            mask = np.isfinite(arr)
            if mask.sum() < 2:
                return '-'
            v = _interp_at(z[mask], arr[mask], z_target)
            return f'{v:.3f}' if v is not None else '-'

        print(f'  {label:<22}  '
              f'{fmt_at(fq_cen, 0.0):>11}  {fmt_at(fq_cen, 2.0):>11}  '
              f'{fmt_at(fq_cen, 5.0):>11}  {fmt_at(fq_sat, 0.0):>11}  '
              f'{fmt_at(fq_sat, 2.0):>11}  {fmt_at(fq_sat, 5.0):>11}')


def main():
    args = parse_arguments()

    models = [
        (args.model1, args.label1),
        (args.model2, args.label2),
        (args.model3, args.label3),
    ]

    if args.absolute_mass_cut is not None:
        selection_label = (f'log10(M*/Msun) >= {args.absolute_mass_cut:g}, '
                           f'sSFR < {args.ssfr_factor:g} * H(z)')
        mass_panel_label = (rf'$\log_{{10}}(M_\star/M_\odot) \geq '
                            rf'{args.absolute_mass_cut:g}$')
        file_tag = f'absM{args.absolute_mass_cut:g}'.replace('.', 'p')
    else:
        selection_label = (f'top {args.top_percent:g}% by M_star, '
                           f'sSFR < {args.ssfr_factor:g} * H(z)')
        mass_panel_label = rf'top {args.top_percent:g}% by $M_\star$'
        file_tag = f'top{args.top_percent:g}pct'.replace('.', 'p')
    print(f'Selection: {selection_label}\n')

    results = []
    for path, label in models:
        files = resolve_files(path)
        if not files:
            print(f'Error: no HDF5 files found for {label} at {path}')
            sys.exit(1)
        print(f'{label}: {len(files)} files from {path}')
        header = read_header(files)
        print(f'  cosmology: h={header["hubble_h"]:.4f}  '
              f'Omega_m={header["omega_m"]:.4f}  Omega_L={header["omega_l"]:.4f}')

        first_snap = args.first_snap if args.first_snap is not None else min(header['snaps'])
        last_snap = args.last_snap if args.last_snap is not None else max(header['snaps'])

        print(f'  reading snapshots {first_snap}..{last_snap}')
        result = compute_fractions(
            files, header, first_snap, last_snap,
            args.top_percent, args.ssfr_factor,
            abs_log_mass=args.absolute_mass_cut,
            verbose=not args.quiet,
        )
        if result['z'].size == 0:
            print(f'  warning: no usable snapshots for {label}')
        _print_model_summary(label, result)
        results.append((label, result))

    _print_comparison_table(results)

    os.makedirs(args.output_dir, exist_ok=True)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(15, 6))

    z_max = 12.0
    for (label, res), color in zip(results, MODEL_COLORS):
        z = res['z']
        if z.size == 0:
            continue
        z_max = max(z_max, float(z.max()))

        # Left panel: composition of the quiescent sample.
        ax_left.plot(z, res['cen'], color=color, lw=2.5, ls='-')
        ax_left.plot(z, res['sat'], color=color, lw=2.5, ls='--')

        # Right panel: per-population quiescent fraction.
        fq_cen, fq_sat = res['fq_cen'], res['fq_sat']
        m_cen = np.isfinite(fq_cen)
        m_sat = np.isfinite(fq_sat)
        if m_cen.any():
            ax_right.plot(z[m_cen], fq_cen[m_cen], color=color, lw=2.5, ls='-')
        if m_sat.any():
            ax_right.plot(z[m_sat], fq_sat[m_sat], color=color, lw=2.5, ls='--')

    for ax in (ax_left, ax_right):
        ax.set_xlim(0, z_max)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel(r'Redshift $z$', fontsize=14)
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))

    ax_left.set_ylabel(rf'Fraction of quiescent ({mass_panel_label})', fontsize=13)
    ax_left.set_title('Composition of quiescent sample', fontsize=13)
    ax_right.set_ylabel(rf'Quiescent fraction within population ({mass_panel_label})',
                        fontsize=13)
    ax_right.set_title('Per-population quiescent fraction', fontsize=13)

    model_handles = [Line2D([0], [0], color=c, lw=2.5)
                     for c, _ in zip(MODEL_COLORS, results)]
    model_labels = [label for label, _ in results]
    style_handles = [Line2D([0], [0], color='black', lw=2.5, ls='-'),
                     Line2D([0], [0], color='black', lw=2.5, ls='--')]

    leg_left = ax_left.legend(model_handles, model_labels, loc='center left',
                              fontsize=11, frameon=False)
    ax_left.add_artist(leg_left)
    ax_left.legend(style_handles, ['Centrals', 'Satellites'], loc='center right',
                   fontsize=11, frameon=False)

    leg_right = ax_right.legend(model_handles, model_labels, loc='upper left',
                                fontsize=11, frameon=False)
    ax_right.add_artist(leg_right)
    ax_right.legend(style_handles, [r'$f_{q}^{\rm cen}$', r'$f_{q}^{\rm sat}$'],
                    loc='lower right', fontsize=11, frameon=False)

    plt.tight_layout()
    out_file = os.path.join(
        args.output_dir,
        f'CentralSatelliteFraction_MassiveQuiescent_3models_{file_tag}{args.format}',
    )
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f'\nSaved file to {out_file}')
    plt.close()


if __name__ == '__main__':
    main()

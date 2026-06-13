#!/usr/bin/env python
"""
Distribution of r_heat / Rvir for CGM-regime quiescent galaxies in the new
model at high redshift.

CGM regime  := Regime == 0
Quiescent   := sSFR < 0.2 H(z),    sSFR = (SfrDisk + SfrBulge) / StellarMass

Three side-by-side panels at the snapshots nearest z = 2, 3, 4 (or the values
passed via --redshifts).  r_heat and Rvir are both output in Mpc/h so the
ratio is dimensionless.

Usage:
    python plotting/quiescent_rheat_distribution.py
    python plotting/quiescent_rheat_distribution.py --redshifts 2 3 4 --model ./output/millennium/
"""

import argparse
import glob
import os
import sys
import warnings

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

_STYLE = './plotting/kieren_cohare_palatino_sty.mplstyle'
if os.path.exists(_STYLE):
    plt.style.use(_STYLE)
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor']   = 'white'
plt.rcParams['axes.edgecolor']   = 'black'


DEFAULT_MODEL      = './output/millennium/'
DEFAULT_LABEL      = 'SAGE26 (millennium)'
DEFAULT_REDSHIFTS  = [2.0, 3.0, 4.0]
DEFAULT_OUTPUT_DIR = './output/quiescent_evolution/'
DEFAULT_FORMAT     = '.pdf'
SSFR_FACTOR        = 0.2

MODEL_COLOR        = '#175cdb'

PROPS = ['StellarMass', 'SfrDisk', 'SfrBulge', 'Regime', 'r_heat', 'Rvir']

OMEGA_M_DEFAULT = 0.25
OMEGA_L_DEFAULT = 0.75


def _hubble_per_yr(z, omega_m, omega_l, hubble_h):
    inv_H0_yr = (9.778 / hubble_h) * 1.0e9
    return np.sqrt(omega_m * (1.0 + z)**3 + omega_l) / inv_H0_yr


def _ssfr_threshold(z, omega_m, omega_l, hubble_h):
    return SSFR_FACTOR * _hubble_per_yr(z, omega_m, omega_l, hubble_h)


def _find_files(directory):
    files = sorted(glob.glob(os.path.join(directory, 'model_*.hdf5')))
    if not files:
        single = os.path.join(directory, 'model_0.hdf5')
        if os.path.exists(single):
            files = [single]
    return files


def _read_header(directory):
    files = _find_files(directory)
    if not files:
        return None
    with h5.File(files[0], 'r') as f:
        sim     = f['Header/Simulation']
        runtime = f['Header/Runtime']
        hdr = {
            'hubble_h':       float(sim.attrs['hubble_h']),
            'omega_m':        float(sim.attrs.get('omega_m', OMEGA_M_DEFAULT)),
            'omega_l':        float(sim.attrs.get('omega_l', OMEGA_L_DEFAULT)),
            'unit_mass_in_g': float(runtime.attrs['UnitMass_in_g']),
            'redshifts':      list(f['Header/snapshot_redshifts'][:]),
        }
    snap_set = set()
    for fp in files:
        with h5.File(fp, 'r') as f:
            for k in f.keys():
                if k.startswith('Snap_'):
                    snap_set.add(int(k.replace('Snap_', '')))
    hdr['output_snaps'] = sorted(snap_set)
    hdr['files']        = files
    hdr['mass_conv']    = hdr['unit_mass_in_g'] / 1.989e33 / hdr['hubble_h']
    return hdr


def _load_snap(files, snap_num):
    """Return dict {prop: array} concatenated across files for Snap_<snap_num>.
    StellarMass is mass-converted; r_heat and Rvir are length [Mpc/h] kept native.
    """
    snap_key = f'Snap_{snap_num}'
    chunks = {p: [] for p in PROPS}
    found  = False
    missing = set()
    for fp in files:
        with h5.File(fp, 'r') as f:
            if snap_key not in f:
                continue
            grp = f[snap_key]
            snap_len = None
            for p in PROPS:
                if p in grp:
                    snap_len = int(grp[p].shape[0])
                    break
            if snap_len is None or snap_len == 0:
                continue
            found = True
            for p in PROPS:
                if p in grp:
                    chunks[p].append(np.array(grp[p]))
                else:
                    missing.add(p)
                    chunks[p].append(np.zeros(snap_len, dtype=np.float32))
    if not found:
        return {}, missing
    out = {}
    for p in PROPS:
        if chunks[p]:
            out[p] = np.concatenate(chunks[p])
    return out, missing


def _snap_nearest_z(redshifts, z_target, available):
    arr = np.array(redshifts)
    return min(available, key=lambda s: abs(arr[s] - z_target))


def _select_ratio(d, z, omega_m, omega_l, hubble_h, mass_conv):
    """Return r_heat/Rvir for CGM-regime quiescent galaxies in this snap."""
    sm_code = d['StellarMass']                 # 1e10 Msun/h
    sm_msun = sm_code * mass_conv              # Msun
    sfr     = d['SfrDisk'] + d['SfrBulge']     # Msun/yr
    regime  = d['Regime']
    rheat   = d['r_heat']                      # Mpc/h
    rvir    = d['Rvir']                        # Mpc/h

    ssfr = np.where(sm_msun > 0, sfr / np.maximum(sm_msun, 1e-30), np.inf)
    thr  = _ssfr_threshold(z, omega_m, omega_l, hubble_h)

    sel = (sm_msun > 0) & (regime == 0) & (rvir > 0) & np.isfinite(rheat) & (ssfr < thr)
    n_cgm        = int(((sm_msun > 0) & (regime == 0)).sum())
    n_cgm_quiesc = int(sel.sum())
    ratio = rheat[sel] / rvir[sel]
    return ratio, n_cgm, n_cgm_quiesc, thr


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',      default=DEFAULT_MODEL,
                        help='Output directory of the new model.')
    parser.add_argument('--label',      default=DEFAULT_LABEL)
    parser.add_argument('--redshifts',  type=float, nargs='+', default=DEFAULT_REDSHIFTS,
                        help='Target redshifts (one panel each).')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--format',     default=DEFAULT_FORMAT)
    parser.add_argument('--n-bins',     type=int, default=30)
    args = parser.parse_args(argv)

    hdr = _read_header(args.model)
    if hdr is None:
        sys.exit(f'No model files in {args.model}')

    redshifts = hdr['redshifts']
    omega_m   = hdr['omega_m']
    omega_l   = hdr['omega_l']
    hubble_h  = hdr['hubble_h']
    mass_conv = hdr['mass_conv']

    print('=' * 70)
    print(f'r_heat / Rvir distribution for CGM-regime quiescent galaxies')
    print('=' * 70)
    print(f'  model: {args.model}  ({args.label})')
    print(f'  target z: {args.redshifts}')
    print(f'  available snapshots: {len(hdr["output_snaps"])}')

    panels = []
    for z_t in args.redshifts:
        snap = _snap_nearest_z(redshifts, z_t, hdr['output_snaps'])
        z    = redshifts[snap]
        d, missing = _load_snap(hdr['files'], snap)
        if not d:
            print(f'  z~{z_t}: snap {snap} empty -- skipping.')
            continue
        if 'r_heat' in missing:
            sys.exit(
                'r_heat field missing from output -- the simulation predates the '
                'HDF5 writer update that adds r_heat. Re-run SAGE on this dataset.')
        ratio, n_cgm, n_cgm_q, thr = _select_ratio(
            d, z, omega_m, omega_l, hubble_h, mass_conv)
        if ratio.size == 0:
            print(f'  z={z:.2f} (snap {snap}): no CGM-regime quiescent galaxies.')
        else:
            print(f'  z={z:.2f} (snap {snap}): '
                  f'N_CGM={n_cgm}, N_CGM_quiescent={n_cgm_q}, '
                  f'sSFR_thr={thr:.2e}/yr, '
                  f'median(r_heat/Rvir)={np.median(ratio):.3f}, '
                  f'frac at cap (>=0.999)={(ratio >= 0.999).mean():.3f}')
        panels.append((z, snap, ratio, n_cgm_q))

    if not panels:
        sys.exit('No usable redshifts.')

    os.makedirs(args.output_dir, exist_ok=True)

    ncols = len(panels)
    fig, axes = plt.subplots(1, ncols, figsize=(4.6 * ncols, 4.0),
                             squeeze=False, sharey=False)
    bins = np.linspace(0.0, 1.0, args.n_bins + 1)

    for ai, (z, snap, ratio, n_q) in enumerate(panels):
        ax = axes[0, ai]
        if ratio.size == 0:
            ax.text(0.5, 0.5, 'no CGM-regime\nquiescent galaxies',
                    transform=ax.transAxes, ha='center', va='center')
        else:
            clipped = np.clip(ratio, 0.0, 1.0)
            ax.hist(clipped, bins=bins, color=MODEL_COLOR, alpha=0.7,
                    histtype='stepfilled', edgecolor=MODEL_COLOR, lw=1.4)
            med = np.median(clipped)
            ax.axvline(med, color='black', lw=1.0, ls='--',
                       label=f'median = {med:.3f}')
            ax.legend(loc='upper left', frameon=False, fontsize=9)
        ax.set_xlim(0.0, 1.05)
        ax.set_xlabel(r'$r_\mathrm{heat}\,/\,R_\mathrm{vir}$')
        ax.set_ylabel(r'$N$ galaxies')
        ax.set_title(f'z = {z:.2f}  (snap {snap}, N = {n_q})', fontsize=11)
        ax.tick_params(which='both', direction='in', top=True, right=True)

    fig.suptitle(f'CGM-regime quiescent galaxies, {args.label}', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    z_tag = '_'.join(f'{int(round(z))}' for z, _snap, _r, _n in panels)
    out_path = os.path.join(
        args.output_dir,
        f'cgm_quiescent_rheat_over_rvir_z{z_tag}{args.format}')
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


if __name__ == '__main__':
    main()

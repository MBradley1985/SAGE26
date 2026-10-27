#!/usr/bin/env python3
"""
compare_sage.py
===============
Where does SAGE26 sit relative to the SHAM benchmarks?  For each snapshot present in
BOTH a SAGE26 HDF5 output and SHAM/output, it compares

  row 1  SMF   : SAGE26 vs the benchmark Schechter curve (what SHAM reproduces)
  row 2  BHMF  : SAGE26 vs the TRINITY benchmark curve
  row 3  M_BH-M_* : SAGE26 median and 16-84% vs the SHAM median (+ observed LRD/AGN points)

and prints, per snapshot, the density of f_BH >= 3% galaxies in the observed-LRD M_* window
for SAGE26, next to the SHAM-allowed ceiling (see lrd_abundance.py), plus the rank
correlation (Spearman) between M_BH and M_* for SAGE26 and SHAM in that window.
If SAGE26's marginals match the benchmarks, any offset in the M_BH-M_* row is purely a
pairing effect.

Usage:
    python SHAM/compare_sage.py                                   # output/millennium/model_0.hdf5
    python SHAM/compare_sage.py -i output/millennium/heavy_edd_10t/model_0.hdf5 --snaps 24 18 15
"""

import argparse
import os
import sys

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(REPO, 'plotting'))
sys.path.insert(0, HERE)
import bh_lrd_analysis as lrd
from lrd_literature_data import PANG26, JONES25_LRD_SOURCES
from lrd_abundance import obs_window, ceiling
from plot_sham_fbh import percentile_track, SHAM_COLOR

SAGE_COLOR = '#C62828'
DLOG = 0.2


def read_sage(path, snap):
    """`path` may be a glob (MPI runs write one model_<rank>.hdf5 per rank): arrays are
    concatenated and the volume fractions added."""
    import glob
    ms, mb, seed, vol = [], [], [], 0.0
    for fp in sorted(glob.glob(path)) or [path]:
        with h5py.File(fp, 'r') as f:
            sim = f['Header/Simulation'].attrs
            h = float(sim['hubble_h'])
            vol += (float(sim['box_size']) / h) ** 3 * float(f['Header/Runtime'].attrs['frac_volume_processed'])
            g = f[f'Snap_{snap}']
            conv = 1e10 / h
            ms.append(np.asarray(g['StellarMass']) * conv)
            mb.append(np.asarray(g['BlackHoleMass']) * conv)
            seed.append(np.asarray(g['BHSeedMass']) * conv)
    return np.concatenate(ms), np.concatenate(mb), np.concatenate(seed), vol


def mass_function(logm, vol, edges):
    n, _ = np.histogram(logm, edges)
    return 0.5 * (edges[1:] + edges[:-1]), n / vol / np.diff(edges)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-i', '--input', default=os.path.join(REPO, 'output/millennium/model_0.hdf5'))
    ap.add_argument('--sham-dir', default=os.path.join(HERE, 'output'))
    ap.add_argument('--snaps', type=int, nargs='+', default=[63, 32, 24, 18, 15])
    ap.add_argument('--out', default=os.path.join(HERE, 'plots', 'sage_vs_sham.png'))
    ap.add_argument('--fbh', type=float, default=lrd.LRD_FBHM_THRESH)
    a = ap.parse_args()

    n = len(a.snaps)
    fig, axs = plt.subplots(3, n, figsize=(3.4 * n, 9.6), squeeze=False)
    print(f'SAGE26 file: {a.input}\n  f_BH >= {a.fbh:g}; window = 10-90th pct of observed M*; densities cMpc^-3')
    print(f"{'snap':>4} {'z':>5} {'window':>12} {'SAGE n(f>=3%)':>14} {'SAGE (no seeds)':>15} {'ceiling':>10} "
          f"{'rho SAGE':>9} {'rho SHAM':>9} {'dBHMF6-7.5':>10} {'dBHMF>8':>8} {'dSMFwin':>8} {'N_hit':>6}")
    for c, snap in enumerate(a.snaps):
        d = np.load(os.path.join(a.sham_dir, f'sham_snap{snap}.npz'))
        z = float(d['z'])
        ms, mb, seed, vol = read_sage(a.input, snap)
        good = ms > 0
        lms = np.log10(ms[good])
        bh = mb[good]; sd = seed[good]
        edges = np.arange(6.0, 12.6, DLOG)

        # --- row 1: SMF
        ax = axs[0, c]
        x, p = mass_function(lms, vol, edges)
        ax.plot(d['smf_logm'], d['smf_phi'], color='k', lw=1.5, label='benchmark')
        ax.plot(x[p > 0], p[p > 0], color=SAGE_COLOR, lw=1.8, label='SAGE26')
        ax.set_yscale('log'); ax.set_xlim(6.5, 12.2); ax.set_ylim(1e-7, 1e0)
        ax.set_title(rf'$z={z:.1f}$ (snap {snap})', fontsize=11)
        if c == 0:
            ax.set_ylabel(r'SMF  [Mpc$^{-3}$ dex$^{-1}$]'); ax.legend(fontsize=8, loc='lower left')
        ax.set_xlabel(r'$\log M_\star$')

        # --- row 2: BHMF
        ax = axs[1, c]
        has = bh > 0
        x, p = mass_function(np.log10(bh[has]), vol, np.arange(3.0, 10.6, DLOG))
        ax.plot(d['bhmf_logm'], d['bhmf_phi'], color='k', lw=1.5)
        _, lo = bm_curve(z, 'lo', d['bhmf_logm'])
        _, hi = bm_curve(z, 'hi', d['bhmf_logm'])
        ax.fill_between(d['bhmf_logm'], lo, hi, color='k', alpha=0.12, lw=0)
        ax.plot(x[p > 0], p[p > 0], color=SAGE_COLOR, lw=1.8)
        bref = np.interp(x, d['bhmf_logm'], d['bhmf_phi'])
        def _lr(lo_, hi_):
            m = (x >= lo_) & (x < hi_) & (p > 0)
            return np.log10(p[m].sum() / bref[m].sum()) if m.any() else np.nan
        bh_ratio = (_lr(6, 7.5), _lr(8, 11))
        ax.set_yscale('log'); ax.set_xlim(3.5, 10.2); ax.set_ylim(1e-7, 1e-0)
        if c == 0:
            ax.set_ylabel(r'BHMF  [Mpc$^{-3}$ dex$^{-1}$]')
        ax.set_xlabel(r'$\log M_{\rm BH}$')

        # --- row 3: M_BH-M_* (galaxies with a BH)
        ax = axs[2, c]
        sel = bh > 0
        sx, sy = lms[sel], np.log10(bh[sel])
        xc, q = percentile_track(sx, sy, np.arange(6.5, 12.0, 0.25))
        if len(sx) > 20000:
            i = np.random.default_rng(1).choice(len(sx), 20000, replace=False)
            ax.scatter(sx[i], sy[i], s=1.5, color=SAGE_COLOR, alpha=0.15, lw=0, rasterized=True)
        ax.plot(xc, q[:, 1], color=SAGE_COLOR, lw=2, label='SAGE26 median')
        ax.fill_between(xc, q[:, 0], q[:, 2], color=SAGE_COLOR, alpha=0.15, lw=0)
        ok = np.isfinite(d['logmstar']) & np.isfinite(d['logmbh'])
        xs, qs = percentile_track(d['logmstar'][ok], d['logmbh'][ok], np.arange(6.5, 12.0, 0.25))
        ax.plot(xs, qs[:, 1], color=SHAM_COLOR, lw=2, label='SHAM median')
        ax.fill_between(xs, qs[:, 0], qs[:, 2], color=SHAM_COLOR, alpha=0.15, lw=0)
        xr = np.linspace(6.5, 12, 50)
        ax.plot(xr, xr + np.log10(a.fbh), 'k--', lw=0.8)
        pm = lrd.lit_z_mask(PANG26['z'], z)
        lrd.plot_lit_points(ax, 'Pang+26', PANG26['log_mstar'][pm], PANG26['log_mbh'][pm])
        jm = lrd.jones25_family_masks(JONES25_LRD_SOURCES, z)
        lrd.plot_jones25_family(ax, JONES25_LRD_SOURCES, jm, kind='mbh_mstar')
        ax.set_xlim(6.5, 11.8); ax.set_ylim(3, 10)
        if c == 0:
            ax.set_ylabel(r'$\log M_{\rm BH}$'); ax.legend(fontsize=8, loc='upper left')
        ax.set_xlabel(r'$\log M_\star$')

        # --- abundance table
        w = obs_window(z)
        if w is not None:
            win = (w[0], w[1])
            inw = (lms >= win[0]) & (lms <= win[1])
            fb = np.where(bh > 0, np.log10(np.maximum(bh, 1)) - lms, -99)
            n_hit = np.sum(inw & (fb >= np.log10(a.fbh))) / vol
            N_hit = int(np.sum(inw & (fb >= np.log10(a.fbh))))
            dsmf = np.log10(max(inw.sum() / vol, 1e-30) / np.sum(d['smf_phi'][(d['smf_logm'] >= win[0]) & (d['smf_logm'] <= win[1])] * (d['smf_logm'][1] - d['smf_logm'][0])))
            grown = lrd.mask_ungrown_seeds(bh, sd)
            n_hit_ng = np.sum(inw & (fb >= np.log10(a.fbh)) & grown) / vol
            ce = ceiling(d['smf_logm'], d['smf_phi'], d['bhmf_logm'], d['bhmf_phi'], win, a.fbh)
            si = inw & (bh > 0)
            rs = spearmanr(lms[si], np.log10(bh[si]))[0] if si.sum() > 30 else np.nan
            wm = (d['logmstar'] >= win[0]) & (d['logmstar'] <= win[1]) & ok
            rh = spearmanr(d['logmstar'][wm][:200000], d['logmbh'][wm][:200000])[0] if wm.sum() > 30 else np.nan
            print(f'{snap:4d} {z:5.2f} {win[0]:5.2f}-{win[1]:5.2f}  {n_hit:14.2e} {n_hit_ng:15.2e} {ce:10.2e} {rs:9.2f} {rh:9.2f} {bh_ratio[0]:10.2f} {bh_ratio[1]:8.2f} {dsmf:8.2f} {N_hit:6d}')
        else:
            print(f'{snap:4d} {z:5.2f}   (no observed points near this z)')

    plt.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    plt.savefig(a.out, dpi=130, bbox_inches='tight')
    print(f'saved {a.out}')


def bm_curve(z, variant, grid):
    import sham_benchmarks as bm
    g, phi, _ = bm.bhmf_benchmark(z, variant, grid=grid)
    return g, phi


if __name__ == '__main__':
    main()

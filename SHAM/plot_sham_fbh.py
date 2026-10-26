#!/usr/bin/env python3
"""
plot_sham_fbh.py
================
Plots the abundance-matched ("theoretical") black holes from run_sham.py on the
f_BH = M_BH/M_* vs M_BH plane -- panel (b) of plotting/bh_lrd_analysis.py --
together with the LRD / broad-line AGN observations, and (optionally) the SAGE26
galaxies underneath, to test whether the SHAM black holes can even reach the
observed LRD f_BH.

Also writes a closure/validation figure (model SMF and BHMF vs the benchmark
curves they were matched to) and a text summary of the LRD-region statistics.

Usage:
    python SHAM/plot_sham_fbh.py                       # every SHAM/output/sham_snap*.npz
    python SHAM/plot_sham_fbh.py --snaps 19            # one snapshot
    python SHAM/plot_sham_fbh.py --no-sage             # SHAM only
    python SHAM/plot_sham_fbh.py -i './output/millennium/model_*.hdf5'
"""

import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(REPO, 'plotting'))
import bh_lrd_analysis as lrd          # panel-b machinery, constants, literature overlay
from lrd_literature_data import PANG26, JONES25_LRD_SOURCES

SHAM_COLOR = '#1565C0'
SAGE_COLOR = '#999999'
LRD_FBH = lrd.LRD_FBHM_THRESH           # 0.03


def percentile_track(x, y, edges, qs=(16, 50, 84), nmin=20):
    out = np.full((len(edges) - 1, len(qs)), np.nan)
    for i in range(len(edges) - 1):
        m = (x >= edges[i]) & (x < edges[i + 1])
        if m.sum() >= nmin:
            out[i] = np.percentile(y[m], qs)
    return 0.5 * (edges[1:] + edges[:-1]), out


def summarise(d, mbh, mstar, extrap):
    """LRD-region statistics of the SHAM sample; returns a list of text lines."""
    vol = float(d['volume_mpc3'])
    fbh = mbh - mstar
    lines = [f"z = {float(d['z']):.2f} (snap {int(d['snap'])}), V = {vol:.3e} Mpc^3 "
             f"({int(d['n_files'])} tree files), property = {str(d['property'])}, "
             f"sigma_* = {float(d['sigma_star'])}, sigma_BH = {float(d['sigma_bh'])}, "
             f"rho = {float(d['rho'])}, BHMF = TRINITY {str(d['bhmf_variant'])}",
             f"SHAM black holes: {len(mbh):,}   max f_BH = {10**fbh.max():.3g}   "
             f"[{int((fbh >= np.log10(LRD_FBH)).sum())} with f_BH >= {LRD_FBH:.0%}]",
             "  M_BH bin      N(all)   N(f_BH>=3%)   frac     phi_all[Mpc^-3 dex^-1]   "
             "phi(f_BH>=3%)   max f_BH   (M* below data range: " 
             f"{extrap.sum():,} BHs)"]
    for lo, hi in [(5, 6), (6, 7), (7, 8), (8, 9), (9, 10)]:
        m = (mbh >= lo) & (mbh < hi)
        n = int(m.sum())
        k = int((m & (fbh >= np.log10(LRD_FBH))).sum())
        mx = 10 ** fbh[m].max() if n else np.nan
        lines.append(f"  [{lo},{hi})  {n:9,d}  {k:11,d}   {k / max(n, 1):7.2e}   "
                     f"{n / vol:12.3e}        {k / vol:12.3e}   {mx:8.3g}")
    return lines


def plot_fbh(d, files, out, show_lit=True, n_scatter=40_000, mask_seeds=True):
    z = float(d['z'])
    snap = int(d['snap'])
    ms, mb = d['logmstar'], d['logmbh']
    ok = np.isfinite(ms) & np.isfinite(mb)
    ms, mb = ms[ok], mb[ok]
    ms_i, mb_i = d['logmstar_int'][ok], d['logmbh_int'][ok]
    extrap = ms_i < float(d['smf_data_min'])     # M* below the SMF data -> Schechter extrapolation
    lfbh = mb - ms

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.minorticks_on()
    x_lo, x_hi = lrd.lock_axis_range(*lrd.PANEL_B_XLIM, axis_name='x')
    y_lo, y_hi = lrd.lock_axis_range(*lrd.PANEL_B_YLIM, axis_name='y')

    # LRD shaded region + threshold lines (as panel b)
    ax.fill_between([x_lo, x_hi], np.log10(LRD_FBH), y_hi, color='#D32F2F', alpha=0.08, zorder=0)
    ax.axhline(np.log10(0.1), color='#F57C00', lw=1.8, zorder=4)
    ax.axhline(np.log10(LRD_FBH), color='#C62828', lw=1.8, zorder=4)
    ax.text(x_hi - 0.2, np.log10(LRD_FBH) + 0.10, r'$f_{\rm BH} = 3\%$',
            fontsize=13, color='#C62828', ha='right')
    ax.text(x_hi - 0.2, np.log10(0.1) + 0.10, r'$M_{\rm BH}/M_\star = 0.1$',
            fontsize=13, color='#F57C00', ha='right')

    handles = []

    # ---- SAGE26 galaxies (grey), same selection as panel b ------------------
    if files:
        try:
            h_h = lrd.read_sim_params(files[0])
            sd = lrd.read_epoch(files, snap, h_h)
            valid = ((sd['bh_mass'] > 0) & (sd['stellar_mass'] > 0) & (sd['mdot_msun_yr'] > 0))
            if mask_seeds:
                valid &= lrd.mask_ungrown_seeds(sd['bh_mass'], sd['seed_mass'])
            sx = np.log10(sd['bh_mass'][valid])
            sy = np.log10(sd['bh_mass'][valid] / sd['stellar_mass'][valid])
            if len(sx) > 30_000:
                i = np.random.default_rng(42).choice(len(sx), 30_000, replace=False)
                ax.scatter(sx[i], sy[i], s=4, color=SAGE_COLOR, alpha=0.2, lw=0,
                           rasterized=True, zorder=1)
            else:
                ax.scatter(sx, sy, s=4, color=SAGE_COLOR, alpha=0.2, lw=0, rasterized=True, zorder=1)
            lrd.kde_contour(ax, sx, sy, x_lo, x_hi, y_lo, y_hi, '#000000', zorder=2)
            handles.append(Line2D([0], [0], marker='o', ls='none', markerfacecolor=SAGE_COLOR,
                                  markeredgecolor='none', ms=7, label=f'SAGE26 ({len(sx):,} BH events)'))
        except Exception as e:
            print(f'  (SAGE26 overlay skipped: {e})')

    # ---- SHAM black holes ---------------------------------------------------
    rng = np.random.default_rng(7)
    sel = rng.choice(len(mb), min(n_scatter, len(mb)), replace=False)
    ex = extrap[sel]
    ax.scatter(mb[sel][~ex], lfbh[sel][~ex], s=5, color=SHAM_COLOR, alpha=0.30, lw=0,
               rasterized=True, zorder=3)
    if ex.any():
        ax.scatter(mb[sel][ex], lfbh[sel][ex], s=5, facecolors='none', edgecolors=SHAM_COLOR,
                   alpha=0.25, lw=0.3, rasterized=True, zorder=3)
    lrd.kde_contour(ax, mb, lfbh, x_lo, x_hi, y_lo, y_hi, SHAM_COLOR, zorder=5)
    # zero-scatter locus and scattered percentile track
    o = np.argsort(mb_i)
    ax.plot(mb_i[o], (mb_i - ms_i)[o], color='k', lw=1.5, ls='-', zorder=6)
    xc, q = percentile_track(mb, lfbh, np.arange(x_lo, x_hi + 0.01, 0.25))
    ax.plot(xc, q[:, 1], color=SHAM_COLOR, lw=2.2, zorder=7)
    ax.fill_between(xc, q[:, 0], q[:, 2], color=SHAM_COLOR, alpha=0.15, zorder=2)
    handles += [Line2D([0], [0], marker='o', ls='none', markerfacecolor=SHAM_COLOR,
                       markeredgecolor='none', ms=7,
                       label=f'SHAM BHs ({len(mb):,})'),
                Line2D([0], [0], color=SHAM_COLOR, lw=2.2, label='SHAM median, 16-84%'),
                Line2D([0], [0], color='k', lw=1.5, label='SHAM, zero scatter')]

    # ---- observations -------------------------------------------------------
    if show_lit:
        pm = lrd.lit_z_mask(PANG26['z'], z)
        labels = []
        if pm.any():
            t = PANG26['log_mbh'][pm] - PANG26['log_mstar'][pm]
            te = np.sqrt(PANG26['log_mbh_err'][pm] ** 2 + PANG26['log_mstar_err'][pm] ** 2)
            lrd.plot_lit_points(ax, 'Pang+26', PANG26['log_mbh'][pm], t,
                                xerr=PANG26['log_mbh_err'][pm], yerr=te)
            labels.append('Pang+26')
        jm = lrd.jones25_family_masks(JONES25_LRD_SOURCES, z)
        labels += lrd.plot_jones25_family(ax, JONES25_LRD_SOURCES, jm, kind='fbh')
        handles += lrd.lit_legend_handles(labels)

    ax.legend(handles=handles, loc='upper left', fontsize=11, ncol=1)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel(r'$M_{\rm BH}\ [M_\odot]$', fontsize=18)
    ax.set_ylabel(r'$f_{\rm BH} = M_{\rm BH}/M_\star$', fontsize=18)
    ax.set_xticks(np.arange(int(np.ceil(x_lo)), int(np.floor(x_hi)) + 1, 2))
    ax.text(0.97, 0.04, rf'$z = {z:.1f}$', transform=ax.transAxes, ha='right', va='bottom', fontsize=16)
    plt.tight_layout()
    plt.savefig(out, dpi=140, bbox_inches='tight')
    plt.close()
    print(f'  saved {out}')
    return mb, ms, extrap


def plot_closure(d, out):
    """Model SMF / BHMF (after scatter) vs the benchmark curves they were matched to."""
    vol = float(d['volume_mpc3'])
    z = float(d['z'])
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, kind in zip(axs, ('smf', 'bhmf')):
        g, phi, phi_i = d[f'{kind}_logm'], d[f'{kind}_phi'], d[f'{kind}_phi_int']
        m = d['logmstar' if kind == 'smf' else 'logmbh']
        m = m[np.isfinite(m)]
        edges = np.arange(g[0], g[-1] + 0.01, 0.2)
        n, _ = np.histogram(m, edges)
        xc = 0.5 * (edges[1:] + edges[:-1])
        ph = n / vol / 0.2
        ok = n > 0
        ax.plot(g, phi, 'k-', lw=2, label='benchmark')
        ax.plot(g, phi_i, 'k--', lw=1.2, label='de-scattered (matched)')
        ax.errorbar(xc[ok], ph[ok], yerr=ph[ok] / np.sqrt(n[ok]), fmt='o', color=SHAM_COLOR,
                    ms=5, label='SHAM + scatter')
        if kind == 'smf':
            ax.errorbar(d['smf_obs_x'], 10 ** d['smf_obs_y'], fmt='.', color='#C62828', ms=4,
                        alpha=0.5, label='SMF data (compilation)')
            ax.axvline(float(d['smf_data_min']), color='#C62828', ls=':', lw=1)
            lim = np.nanmin(d['logmstar_int'])
            ax.axvline(lim, color=SHAM_COLOR, ls=':', lw=1.2)
            ax.text(lim + 0.05, 2e-7, 'halo resolution limit', rotation=90, color=SHAM_COLOR,
                    fontsize=10, va='bottom')
            ax.set_xlabel(r'$\log_{10}\,M_\star\ [M_\odot]$', fontsize=16)
            ax.set_xlim(6.5, 12.3); ax.set_ylim(1e-7, 1e0)
        else:
            ax.set_xlabel(r'$\log_{10}\,M_{\rm BH}\ [M_\odot]$', fontsize=16)
            ax.set_xlim(4, 10.5); ax.set_ylim(1e-9, 1e-1)
        ax.set_yscale('log')
        ax.set_ylabel(r'$\phi\ [{\rm Mpc^{-3}\,dex^{-1}}]$', fontsize=16)
        ax.legend(fontsize=11)
    axs[0].set_title(f'SMF closure, z = {z:.2f}')
    axs[1].set_title(f'BHMF closure (TRINITY {str(d["bhmf_variant"])}), z = {z:.2f}')
    plt.tight_layout()
    plt.savefig(out, dpi=140, bbox_inches='tight')
    plt.close()
    print(f'  saved {out}')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sham-dir', default=os.path.join(HERE, 'output'))
    p.add_argument('--snaps', type=int, nargs='+', default=None,
                   help='snapshots to plot (default: every sham_snap*.npz found)')
    p.add_argument('-i', '--input-pattern', default=os.path.join(REPO, 'output/millennium/model_*.hdf5'),
                   help='SAGE26 HDF5 files drawn in grey underneath (skipped if none match)')
    p.add_argument('--no-sage', action='store_true')
    p.add_argument('--no-lit', action='store_true')
    p.add_argument('--no-mask-seeds', action='store_true')
    p.add_argument('--out-dir', default=os.path.join(HERE, 'plots'))
    a = p.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    found = sorted(glob.glob(os.path.join(a.sham_dir, 'sham_snap*.npz')))
    if a.snaps:
        found = [f for f in found if int(os.path.basename(f)[9:11]) in a.snaps]
    if not found:
        sys.exit(f'No sham_snap*.npz in {a.sham_dir}; run SHAM/run_sham.py first.')
    files = [] if a.no_sage else sorted(glob.glob(a.input_pattern))
    if not files and not a.no_sage:
        print('  NOTE: no SAGE26 files matched; plotting SHAM only')

    for f in found:
        d = np.load(f, allow_pickle=False)
        s = int(d['snap'])
        print(f'Snapshot {s} (z = {float(d["z"]):.2f})')
        mb, ms, extrap = plot_fbh(d, files, os.path.join(a.out_dir, f'sham_fbh_mbh_snap{s:02d}.png'),
                                  show_lit=not a.no_lit, mask_seeds=not a.no_mask_seeds)
        plot_closure(d, os.path.join(a.out_dir, f'sham_closure_snap{s:02d}.png'))
        txt = summarise(d, mb, ms, extrap)
        with open(os.path.join(a.out_dir, f'sham_summary_snap{s:02d}.txt'), 'w') as fh:
            fh.write('\n'.join(txt) + '\n')
        print('\n'.join(txt))


if __name__ == '__main__':
    main()

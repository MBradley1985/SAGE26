#!/usr/bin/env python3
"""
plot_sham_mbh_mstar.py
======================
M_BH - M_star relation of the abundance-matched ("theoretical") galaxies, in the
2x3 multi-redshift grid look of plotting/bh_lrd_analysis_multiz.py (panel c), with
the same observational overlays (Pang+26, Jones+25 family, Furtak+23), the
Kormendy & Ho (2013) relation and the M_BH/M_star = 0.1, 0.01, 0.001 lines.

Reads only the local SHAM/output/sham_snap*.npz files (no SAGE26 catalogues).

Usage:
    python SHAM/plot_sham_mbh_mstar.py                      # z ~ 0,1,2,4,6,8
    python SHAM/plot_sham_mbh_mstar.py --grid-z 3 5 7 9
    python SHAM/plot_sham_mbh_mstar.py --sham-dir SHAM/output_hi --out-dir SHAM/plots_hi
    python SHAM/plot_sham_mbh_mstar.py --no-lit
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
sys.path.insert(0, HERE)
import bh_lrd_analysis as lrd
from lrd_literature_data import PANG26, FURTAK23, JONES25_LRD_SOURCES
from plot_sham_fbh import GRID_RC, DEFAULT_GRID_Z, SHAM_COLOR, percentile_track

MULTIZ_C_XLIM = lrd.PANEL_C_XLIM          # (6, 11)   log M*
MULTIZ_C_YLIM = lrd.PANEL_C_YLIM          # (3, 9)    log M_BH
KEYS = ('snap', 'z', 'smf_data_min', 'logmstar', 'logmbh', 'logmstar_int', 'logmbh_int')


def load_snaps(sham_dir, snaps=None):
    out = []
    for f in sorted(glob.glob(os.path.join(sham_dir, 'sham_snap*.npz'))):
        npz = np.load(f)
        if snaps and int(npz['snap']) not in snaps:
            continue
        out.append({k: npz[k] for k in KEYS})          # npz is lazy: other arrays never read
    return out


def lit_for_z(z):
    pm = lrd.lit_z_mask(PANG26['z'], z)
    fm = bool(np.any(lrd.lit_z_mask(FURTAK23['z'], z)))
    jm = lrd.jones25_family_masks(JONES25_LRD_SOURCES, z)
    return pm, fm, jm


def draw_mbh_mstar(ax, d, xlim, ylim, show_lit=True, n_scatter=12_000):
    """One redshift's M_BH-M_* panel (SHAM galaxies + obs). Returns legend handles."""
    z = float(d['z'])
    ms, mb = d['logmstar'], d['logmbh']
    ok = np.isfinite(ms) & np.isfinite(mb)
    ms, mb = ms[ok], mb[ok]
    ms_i, mb_i = d['logmstar_int'][ok], d['logmbh_int'][ok]
    extrap = ms_i < float(d['smf_data_min'])
    x_lo, x_hi = xlim
    y_lo, y_hi = ylim
    ax.minorticks_on()

    # Kormendy & Ho 2013 + mass-ratio lines (as multiz panel c)
    xr = np.linspace(x_lo, x_hi, 200)
    ykh = lrd.kormendy_ho_mbh(xr)
    ax.fill_between(xr, ykh - lrd.KORMENDY_HO_SCATTER, ykh + lrd.KORMENDY_HO_SCATTER,
                    color='#999999', alpha=0.35, zorder=0)
    kh, = ax.plot(xr, ykh, color='black', lw=1.4, zorder=2, label='Kormendy & Ho 2013')
    rot = lrd._line_rotation_deg(1.0, x_lo, x_hi, y_lo, y_hi)
    for ratio in lrd.MASS_RATIO_LINES:
        ax.plot(xr, xr + np.log10(ratio), color='black', lw=0.8, ls='--', zorder=2)
        xl = x_lo + 0.12 * (x_hi - x_lo)
        yl = xl + np.log10(ratio)
        if y_lo < yl < y_hi:
            ax.text(xl, yl - 0.12, rf'$M_{{\rm BH}}/M_\star = {ratio:g}$', fontsize=6.5,
                    ha='left', va='top', rotation=rot, rotation_mode='anchor')

    # SHAM galaxies: filled = M* inside the SMF data range, open = Schechter extrapolation
    rng = np.random.default_rng(7)
    sel = rng.choice(len(mb), min(n_scatter, len(mb)), replace=False)
    ex = extrap[sel]
    ax.scatter(ms[sel][~ex], mb[sel][~ex], s=2.5, color=SHAM_COLOR, alpha=0.30, lw=0,
               rasterized=True, zorder=3)
    if ex.any():
        ax.scatter(ms[sel][ex], mb[sel][ex], s=2.5, facecolors='none', edgecolors=SHAM_COLOR,
                   alpha=0.25, lw=0.3, rasterized=True, zorder=3)
    big = rng.choice(len(mb), min(200_000, len(mb)), replace=False)
    lrd.kde_contour(ax, ms[big], mb[big], x_lo, x_hi, y_lo, y_hi, SHAM_COLOR, zorder=5)
    o = np.argsort(ms_i)
    ax.plot(ms_i[o], mb_i[o], color='k', lw=1.0, zorder=6)
    xc, q = percentile_track(ms, mb, np.arange(x_lo, x_hi + 0.01, 0.25))
    ax.plot(xc, q[:, 1], color=SHAM_COLOR, lw=1.5, zorder=7)
    ax.fill_between(xc, q[:, 0], q[:, 2], color=SHAM_COLOR, alpha=0.15, zorder=2)
    handles = [kh,
               Line2D([0], [0], marker='o', ls='none', markerfacecolor=SHAM_COLOR,
                      markeredgecolor='none', ms=7, label='SHAM'),
               Line2D([0], [0], color=SHAM_COLOR, lw=1.5, label='SHAM median, 16-84%'),
               Line2D([0], [0], color='k', lw=1.0, label='SHAM, zero scatter')]

    # observations
    if show_lit:
        pm, fm, jm = lit_for_z(z)
        labels = []
        if pm.any():
            lrd.plot_lit_points(ax, 'Pang+26', PANG26['log_mstar'][pm], PANG26['log_mbh'][pm],
                                xerr=PANG26['log_mstar_err'][pm], yerr=PANG26['log_mbh_err'][pm])
            labels.append('Pang+26')
        labels += lrd.plot_jones25_family(ax, JONES25_LRD_SOURCES, jm, kind='mbh_mstar')
        if fm:
            lrd.plot_lit_points(ax, 'Furtak+23', [FURTAK23['log_mstar_upper_limit']],
                                [FURTAK23['log_mbh']], xerr=[[0.4], [0.0]],
                                yerr=[[FURTAK23['log_mbh_err_lo']], [FURTAK23['log_mbh_err_hi']]],
                                xuplims=True)
            labels.append('Furtak+23')
        handles += lrd.lit_legend_handles(labels)

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.text(0.95, 0.05, rf'$z \approx {z:.1f}$', transform=ax.transAxes, ha='right',
            va='bottom', fontsize=10)
    return handles


def plot_grid(dlist, out, targets, ncols=3, show_lit=True):
    zs = np.array([float(d['z']) for d in dlist])
    chosen = []
    for zt in targets:
        i = int(np.argmin(np.abs(zs - zt)))
        ok = abs(zs[i] - zt) <= max(0.5, 0.15 * (1 + zt))
        chosen.append(dlist[i] if ok else None)
        print(f'  grid: z~{zt:g} -> ' + (f'snap {int(dlist[i]["snap"])} (z={zs[i]:.2f})' if ok
                                          else 'no SHAM output close enough (blank panel)'))
    n = len(targets)
    nrows = int(np.ceil(n / ncols))

    # shared range: multiz default, widened only if observations would be clipped
    xl, yl = [], []
    if show_lit:
        for d in chosen:
            if d is None:
                continue
            pm, fm, jm = lit_for_z(float(d['z']))
            xl.append(PANG26['log_mstar'][pm]); yl.append(PANG26['log_mbh'][pm])
            mx, my = lrd.jones25_family_mstar_mbh(JONES25_LRD_SOURCES, jm)
            xl.append(mx); yl.append(my)
            if fm:
                xl.append([FURTAK23['log_mstar_upper_limit']]); yl.append([FURTAK23['log_mbh']])
    x_lo, x_hi = lrd.lock_axis_range(*MULTIZ_C_XLIM, must_include=xl, axis_name='grid x')
    y_lo, y_hi = lrd.lock_axis_range(*MULTIZ_C_YLIM, must_include=yl, axis_name='grid y')

    with plt.rc_context(GRID_RC):
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.4 * nrows), squeeze=False)
        flat = axes.flatten()
        legend_by_label = {}
        for i, (zt, d) in enumerate(zip(targets, chosen)):
            ax = flat[i]
            if d is None:
                ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
                ax.text(0.5, 0.5, 'no SHAM output', transform=ax.transAxes, ha='center')
                ax.text(0.95, 0.05, rf'$z \approx {zt:g}$', transform=ax.transAxes, ha='right',
                        va='bottom', fontsize=10)
            else:
                for hh in draw_mbh_mstar(ax, d, (x_lo, x_hi), (y_lo, y_hi), show_lit):
                    legend_by_label.setdefault(hh.get_label(), hh)
            ax.set_xticks(np.arange(int(np.floor(x_lo)) + 1, int(np.ceil(x_hi))))
            ax.set_yticks(np.arange(int(np.floor(y_lo)) + 1, int(np.ceil(y_hi))))
            ax.minorticks_on()
            row, col = divmod(i, ncols)
            if col != 0:
                ax.tick_params(labelleft=False)
            if row != nrows - 1:
                ax.tick_params(labelbottom=False)
        for ax in flat[n:]:
            ax.axis('off')

        left, right, top, bottom = 0.06, 0.995, 0.995, 0.065
        fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom, wspace=0.0, hspace=0.0)
        bigax = fig.add_axes([left, bottom, right - left, top - bottom], frameon=False)
        bigax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        bigax.set_xlabel(r'$M_\star\ [M_\odot]$', fontsize=13, labelpad=9)
        bigax.set_ylabel(r'$M_{\rm BH}\ [M_\odot]$', fontsize=13, labelpad=14)
        if legend_by_label:
            flat[0].legend(handles=list(legend_by_label.values()), loc='upper left', fontsize=8,
                           labelspacing=0.3, handletextpad=0.4)
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
    print(f'  saved {out}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sham-dir', default=os.path.join(HERE, 'output'))
    ap.add_argument('--out-dir', default=os.path.join(HERE, 'plots'))
    ap.add_argument('--snaps', type=int, nargs='*', default=None, help='restrict to these snapshots')
    ap.add_argument('--grid-z', type=float, nargs='+', default=DEFAULT_GRID_Z)
    ap.add_argument('--ncols', type=int, default=3)
    ap.add_argument('--no-lit', action='store_true')
    a = ap.parse_args()

    dlist = load_snaps(a.sham_dir, a.snaps)
    if not dlist:
        sys.exit(f'no sham_snap*.npz found in {a.sham_dir}')
    os.makedirs(a.out_dir, exist_ok=True)
    plot_grid(dlist, os.path.join(a.out_dir, 'sham_mbh_mstar_grid.png'), a.grid_z,
              ncols=a.ncols, show_lit=not a.no_lit)


if __name__ == '__main__':
    main()

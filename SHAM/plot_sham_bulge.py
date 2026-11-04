#!/usr/bin/env python3
"""
plot_sham_bulge.py
==================
z ~ 0 SHAM M_BH against (a) total stellar mass and (b) bulge mass, with M_bulge obtained from M_*
through an ASSUMED bulge-to-total relation B/T(M_*), compared with Kormendy & Ho (2013) (and
Haering & Rix 2004). Reads only SHAM/output/sham_snap63.npz.

B/T(M_*) here is an illustrative power law that rises with mass, B/T = min(0.9, A (M_*/1e11)^0.5)
with A = 0.55 (fiducial), 0.28 and 0.9 (low / high); it is a rough stand-in for the observed trend
(roughly 0.1-0.2 near 1e10 and ~0.5 near 1e11, as in SDSS decompositions such as Mendel et al. 2014;
values quoted from memory, not a fit). Because B/T rises monotonically, M_bulge ranks galaxies in
the same order as M_*, so this is equivalent to matching a bulge mass function defined by that relation.

Usage:
    python SHAM/plot_sham_bulge.py
    python SHAM/plot_sham_bulge.py --snap 63 --out SHAM/plots/sham_mbh_mbulge_z0.png
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'plotting'))
sys.path.insert(0, HERE)
import bh_lrd_analysis as lrd
from plot_sham_fbh import percentile_track, SHAM_COLOR

VARIANTS = {'fiducial (A=0.55)': 0.55, 'low B/T (A=0.28)': 0.28, 'high B/T (A=0.9)': 0.9}


def bulge_fraction(log_mstar, A):
    return np.minimum(0.9, A * 10 ** (0.5 * (log_mstar - 11.0)))


def hr04(log_mbulge):
    """Haering & Rix (2004): 8.20 + 1.12 log(M_bulge/1e11) (coefficients from memory)."""
    return 8.20 + 1.12 * (log_mbulge - 11.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sham-dir', default=os.path.join(HERE, 'output'))
    ap.add_argument('--snap', type=int, default=63)
    ap.add_argument('--out', default=os.path.join(HERE, 'plots', 'sham_mbh_mbulge_z0.png'))
    a = ap.parse_args()

    d = np.load(os.path.join(a.sham_dir, f'sham_snap{a.snap}.npz'))
    ms, mb = d['logmstar'], d['logmbh']
    ok = np.isfinite(ms) & np.isfinite(mb)
    ms, mb = ms[ok], mb[ok]
    print(f'z = {float(d["z"]):.2f}, {len(ms):,} SHAM galaxies, M* range {ms.min():.2f}-{ms.max():.2f}')

    fig, axs = plt.subplots(1, 2, figsize=(13, 5.6))
    rng = np.random.default_rng(1)
    sel = rng.choice(len(ms), min(15000, len(ms)), replace=False)

    # left: total M*
    ax = axs[0]
    xr = np.linspace(7, 12.2, 100)
    ax.fill_between(xr, lrd.kormendy_ho_mbh(xr) - 0.29, lrd.kormendy_ho_mbh(xr) + 0.29, color='#999', alpha=0.3)
    ax.plot(xr, lrd.kormendy_ho_mbh(xr), 'k', lw=1.5, label='Kormendy & Ho 2013 (applied to total $M_\\star$)')
    ax.scatter(ms[sel], mb[sel], s=2, color=SHAM_COLOR, alpha=0.25, lw=0, rasterized=True)
    xc, q = percentile_track(ms, mb, np.arange(8.4, 12.1, 0.25))
    ax.plot(xc, q[:, 1], color=SHAM_COLOR, lw=2.2, label='SHAM median')
    ax.fill_between(xc, q[:, 0], q[:, 2], color=SHAM_COLOR, alpha=0.15)
    ax.set_xlabel(r'$\log M_\star$ (total)'); ax.set_ylabel(r'$\log M_{\rm BH}$')
    ax.set_xlim(7.5, 12.2); ax.set_ylim(3, 10); ax.legend(loc='upper left', fontsize=9)

    # right: bulge mass
    ax = axs[1]
    ax.fill_between(xr, lrd.kormendy_ho_mbh(xr) - 0.29, lrd.kormendy_ho_mbh(xr) + 0.29, color='#999', alpha=0.3)
    ax.plot(xr, lrd.kormendy_ho_mbh(xr), 'k', lw=1.5, label='Kormendy & Ho 2013')
    ax.plot(xr, hr04(xr), 'k--', lw=1.0, label='Haering & Rix 2004')
    cols = ['#1565C0', '#2E7D32', '#E65100']
    print('\nmedian M_BH - KH13(M_bulge)  [dex]; columns = bulge-mass bins')
    edges = np.array([8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5])
    print(' ' * 22 + ''.join(f'{e:6.1f}' for e in edges[:-1]) + '   (bin lower edge, log M_bulge)')
    for (lab, A), c in zip(VARIANTS.items(), cols):
        lmb = ms + np.log10(bulge_fraction(ms, A))
        xc, q = percentile_track(lmb, mb, np.arange(7.0, 11.9, 0.25))
        ax.plot(xc, q[:, 1], color=c, lw=2.2, label=f'SHAM, {lab}')
        if A == 0.55:
            ax.scatter(lmb[sel], mb[sel], s=2, color=c, alpha=0.2, lw=0, rasterized=True)
        ax.fill_between(xc, q[:, 0], q[:, 2], color=c, alpha=0.08)
        res = []
        for lo in edges[:-1]:
            m = (lmb >= lo) & (lmb < lo + 0.5)
            res.append(np.median(mb[m]) - np.median(lrd.kormendy_ho_mbh(lmb[m])) if m.sum() > 50 else np.nan)
        print(f'{lab:22s}' + ''.join(f'{r:6.2f}' for r in res))
    ax.set_xlabel(r'$\log M_{\rm bulge}$ (from $M_\star$ via assumed B/T)'); ax.set_ylabel(r'$\log M_{\rm BH}$')
    ax.set_xlim(7, 12.2); ax.set_ylim(3, 10); ax.legend(loc='upper left', fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    plt.savefig(a.out, dpi=140, bbox_inches='tight')
    print(f'\nsaved {a.out}')


if __name__ == '__main__':
    main()

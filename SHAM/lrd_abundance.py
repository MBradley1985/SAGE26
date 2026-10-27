#!/usr/bin/env python3
"""
lrd_abundance.py
================
Abundance consistency test for the LRD f_BH tail, using only the benchmark SMF/BHMF
curves stored in SHAM/output/sham_snap*.npz (no halo trees needed).

For each snapshot it takes the M_* window spanned by the observed high-z AGN/LRD
points (Pang+26, Jones+25 family) and asks:

  1. CEILING  -- the largest comoving density of galaxies in that M_* window with
     f_BH = M_BH/M_* >= 3% that is allowed by BOTH benchmark marginals (SMF in the window,
     BHMF above the needed M_BH), whatever the pairing. Solved exactly by a greedy
     matching (smallest sufficient BH to each galaxy, lowest M_* first).
  2. SHAM     -- the density in the same window for the rank-matched model with
     log-normal scatter sigma_BH (benchmarks re-deconvolved at each sigma, so BHMF and
     SMF stay exact). Shows how much scatter, if it is the only ingredient, is needed.

Compare the printed densities with the observed LRD abundance (--n-lrd, cMpc^-3).

Usage:
    python SHAM/lrd_abundance.py
    python SHAM/lrd_abundance.py --n-lrd 1e-5 --sigmas 0.3 0.5 0.8 1.2
    python SHAM/lrd_abundance.py --fbh 0.1
"""

import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'plotting'))
sys.path.insert(0, HERE)
import bh_lrd_analysis as lrd
from lrd_literature_data import PANG26, JONES25_LRD_SOURCES
import sham_benchmarks as bm


def obs_window(z, lo=10, hi=90):
    ms, mb = [], []
    pm = lrd.lit_z_mask(PANG26['z'], z)
    ms.append(PANG26['log_mstar'][pm]); mb.append(PANG26['log_mbh'][pm])
    jm = lrd.jones25_family_masks(JONES25_LRD_SOURCES, z)
    jx, jy = lrd.jones25_family_mstar_mbh(JONES25_LRD_SOURCES, jm)
    ms.append(np.asarray(jx)); mb.append(np.asarray(jy))
    ms, mb = np.concatenate(ms), np.concatenate(mb)
    ok = np.isfinite(ms) & np.isfinite(mb)
    ms, mb = ms[ok], mb[ok]
    if len(ms) < 3:
        return None
    return (np.percentile(ms, lo), np.percentile(ms, hi), len(ms),
            np.percentile(mb, 50), np.percentile(ms, 50))


def ceiling(lm_s, phi_s, lm_b, phi_b, win, fbh):
    """Greedy max density of (M* in win, M_BH >= fbh*M*) pairs given both marginals."""
    ds, db = lm_s[1] - lm_s[0], lm_b[1] - lm_b[0]
    supply = phi_b * db
    j, tot = 0, 0.0
    for i in np.where((lm_s >= win[0]) & (lm_s <= win[1]))[0]:
        need = phi_s[i] * ds
        j = max(j, int(np.searchsorted(lm_b, lm_s[i] + np.log10(fbh))))
        while need > 0 and j < len(lm_b):
            take = min(need, supply[j])
            supply[j] -= take; need -= take; tot += take
            if supply[j] <= 0:
                j += 1
    return tot


def intrinsic_locus(lm_s, phis_int, lm_b, phib_int):
    """M_BH_int(M*_int) from equal cumulative number densities (zero-scatter SHAM)."""
    Ns = bm.cumulative_above(lm_s, phis_int)
    Nb = bm.cumulative_above(lm_b, phib_int)
    return np.interp(Ns, Nb[::-1], lm_b[::-1], left=np.nan, right=np.nan)


def sham_abundance(lm_s, phi_s, lm_b, phi_b, sig_s, sig_b, win, fbh, n=1_500_000, seed=3):
    """Monte Carlo density of (M*_obs in win, f_BH >= fbh) for scattered, rank-matched SHAM."""
    phis_i = bm.deconvolve(lm_s, phi_s, sig_s)
    phib_i = bm.deconvolve(lm_b, phi_b, sig_b)
    locus = intrinsic_locus(lm_s, phis_i, lm_b, phib_i)
    ds = lm_s[1] - lm_s[0]
    rng = np.random.default_rng(seed)
    # draw intrinsic M* in a band around the window (scatter reaches ~4 sigma)
    band = (lm_s >= win[0] - 4 * sig_s) & (lm_s <= win[1] + 4 * sig_s) & np.isfinite(locus)
    w = phis_i[band] * ds
    tot = w.sum()
    idx = rng.choice(np.where(band)[0], n, p=w / tot)
    mi = lm_s[idx] + rng.uniform(-ds / 2, ds / 2, n)
    bi = locus[idx]
    mo = mi + rng.normal(0, sig_s, n)
    bo = bi + rng.normal(0, sig_b, n)
    sel = (mo >= win[0]) & (mo <= win[1])
    hit = sel & ((bo - mo) >= np.log10(fbh))
    return tot * hit.sum() / n, tot * sel.sum() / n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sham-dir', default=os.path.join(HERE, 'output'))
    ap.add_argument('--fbh', type=float, default=lrd.LRD_FBHM_THRESH)
    ap.add_argument('--sigmas', type=float, nargs='+', default=[0.3, 0.5, 0.8, 1.2])
    ap.add_argument('--n-lrd', type=float, default=None, help='observed LRD density [cMpc^-3] to compare with')
    ap.add_argument('--bhmf-variant', choices=['best', 'lo', 'hi'], default='best',
                    help='TRINITY BHMF percentile to use instead of the stored curve')
    ap.add_argument('--zmin', type=float, default=3.0)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.sham_dir, 'sham_snap*.npz')))
    print(f'f_BH >= {a.fbh:g}; densities in cMpc^-3 (window = 10-90th pct of observed M*)')
    hdr = f"{'z':>5} {'M* window':>13} {'n_obs':>5} {'n(M* win)':>10} {'ceiling':>10} {'ceil/n_win':>10}"
    hdr += ''.join(f'  sig_bh={s:<4g}' for s in a.sigmas)
    print(hdr)
    for f in files:
        d = np.load(f)
        z = float(d['z'])
        if z < a.zmin:
            continue
        w = obs_window(z)
        if w is None:
            continue
        win = (w[0], w[1])
        lm_s, phi_s = d['smf_logm'], d['smf_phi']
        lm_b, phi_b = d['bhmf_logm'], d['bhmf_phi']
        if a.bhmf_variant != 'best':
            phi_b = bm.bhmf_benchmark(z, a.bhmf_variant, grid=lm_b)[1]
        sig_s = float(d['sigma_star'])
        n_win = np.sum(phi_s[(lm_s >= win[0]) & (lm_s <= win[1])]) * (lm_s[1] - lm_s[0])
        ceil = ceiling(lm_s, phi_s, lm_b, phi_b, win, a.fbh)
        row = f'{z:5.2f} {win[0]:5.2f}-{win[1]:5.2f}   {w[2]:5d} {n_win:10.2e} {ceil:10.2e} {ceil / n_win:10.2f}'
        for sb in a.sigmas:
            try:
                n_hit, _ = sham_abundance(lm_s, phi_s, lm_b, phi_b, sig_s, sb, win, a.fbh)
                row += f'  {n_hit:11.2e}'
            except Exception as e:
                row += f'  {"fail":>11}'
        print(row)
    if a.n_lrd:
        print(f'\nobserved n_LRD = {a.n_lrd:.1e}: compare with the columns above.')


if __name__ == '__main__':
    main()

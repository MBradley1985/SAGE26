#!/usr/bin/env python3
"""
sham_benchmarks.py
==================
The two target ("benchmark") curves that the abundance matching reproduces,
taken from the SAME sources the SAGE26 plotting scripts compare against:

  * SMF  : the observational compilation of plotting/paper_plots.py
           (_load_smf_grid_observations: Baldry+08, Muzzin+13, Santini+12,
           Thorne+21, Wright+18, COSMOS-Web, Song+16, Stefanon+21, Weibel+24,
           ...), already put on a common h and IMF there. The compilation is
           points, so at a target redshift the points within +/- z_tol (default 0.15 (1+z), at least 0.3) are fit
           with a Schechter (z >= 3) or double Schechter (z < 3) function to
           give a smooth, monotonic curve that can be integrated.
  * BHMF : TRINITY (Zhang et al. 2023, arXiv:2309.07210) tables in data/bh/
           (the curves drawn by allresults-history.py / allresults-blackholes.py),
           linearly interpolated in log(phi) between the tabulated redshifts.

Both are returned as phi(logM) in Mpc^-3 dex^-1 on a regular logM grid.

deconvolve() undoes a Gaussian scatter so that  intrinsic (x) scatter = benchmark;
the matching is done on the intrinsic curve and the scatter is added afterwards,
so the scattered model function reproduces the benchmark by construction.
"""

import glob
import os
import sys
import numpy as np
from scipy.optimize import least_squares

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LN10 = np.log(10.0)

TRINITY_REDSHIFTS = [0.1, 1.0, 2.0, 4.0, 6.0, 8.0]


# ---------------------------------------------------------------------------
# SMF
# ---------------------------------------------------------------------------
def load_smf_compilation():
    """Observational SMF points from paper_plots.py (run from the repo root)."""
    sys.path.insert(0, os.path.join(REPO_ROOT, 'plotting'))
    import matplotlib
    matplotlib.use('Agg')
    cwd = os.getcwd()
    os.chdir(REPO_ROOT)            # paper_plots.py uses './data/...' paths
    try:
        import paper_plots as pp
        obs = pp._load_smf_grid_observations()
    finally:
        os.chdir(cwd)
    return obs


def _schechter_logphi(logm, p, double):
    """log10 phi [dex^-1] for a (double) Schechter function with shared M*."""
    logms, lphi1, a1 = p[0], p[1], p[2]
    x = 10.0 ** (logm - logms)
    phi = LN10 * (10.0 ** lphi1) * x ** (a1 + 1.0)
    if double:
        phi = phi + LN10 * (10.0 ** p[3]) * x ** (p[4] + 1.0)
    return np.log10(phi * np.exp(-x) + 1e-300)


def fit_smf(z, obs, z_tol=None, double=None, err_floor=0.1):
    """
    Schechter fit to the compilation points within |z_obs - z| <= z_tol.
    Weighted least squares in log phi; per-point errors are floored at
    `err_floor` dex so that one dataset with tiny quoted errors cannot dominate.
    Returns (params, double, info dict with the points used).
    """
    if z_tol is None:
        z_tol = max(0.3, 0.15 * (1.0 + z))   # data bins are coarser at high z
    pts = [o for o in obs if abs(o['z'] - z) <= z_tol]
    if not pts:
        raise ValueError(f'no SMF observations within dz={z_tol} of z={z:.2f}')
    x = np.concatenate([o['log_mass'] for o in pts])
    y = np.concatenate([o['log_phi'] for o in pts])
    def _err(o):
        lo, hi = o['err_lo'], o['err_hi']
        if lo is None or hi is None:                 # some datasets quote no errors
            return np.zeros(len(o['log_mass']))
        return 0.5 * (np.nan_to_num(np.asarray(lo, float)) + np.nan_to_num(np.asarray(hi, float)))
    e = np.concatenate([_err(o) for o in pts])
    e = np.maximum(e, err_floor)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, e = x[ok], y[ok], e[ok]
    if double is None:
        double = z < 3.0

    def resid(p):
        return (_schechter_logphi(x, p, double) - y) / e

    best = None
    for logms0 in (10.6, 11.0, 11.4):
        for a0 in (-1.2, -1.6, -2.0):
            p0 = [logms0, -3.0, a0] + ([-3.5, -0.5] if double else [])
            lo = [9.5, -8.0, -2.8] + ([-9.0, -2.0] if double else [])
            hi = [12.5, 0.0, 0.0] + ([0.0, 0.5] if double else [])
            try:
                r = least_squares(resid, p0, bounds=(lo, hi))
            except Exception:
                continue
            if best is None or r.cost < best.cost:
                best = r
    info = dict(n_points=len(x), datasets=sorted({o['label'] for o in pts}),
                z_obs=sorted({round(o['z'], 2) for o in pts}),
                logm_min=float(x.min()), logm_max=float(x.max()),
                chi2_red=float(2 * best.cost / max(len(x) - len(best.x), 1)),
                x=x, y=y, e=e)
    return best.x, double, info


def smf_benchmark(z, obs=None, grid=None, z_tol=None):
    """phi(logM) [Mpc^-3 dex^-1] on `grid` plus fit info (incl. data mass range)."""
    obs = load_smf_compilation() if obs is None else obs
    grid = np.arange(6.0, 12.8, 0.02) if grid is None else grid
    p, dbl, info = fit_smf(z, obs, z_tol=z_tol)
    phi = 10.0 ** _schechter_logphi(grid, p, dbl)
    info['params'] = p
    info['double'] = dbl
    return grid, phi, info


# ---------------------------------------------------------------------------
# BHMF
# ---------------------------------------------------------------------------
def load_trinity_bhmf(data_dir=None):
    """{z: (logM, phi_best, phi_16, phi_84)} from data/bh/fig4_bhmf_z*.txt."""
    data_dir = data_dir or os.path.join(REPO_ROOT, 'data', 'bh')
    tabs = {}
    for z in TRINITY_REDSHIFTS:
        f = os.path.join(data_dir, f'fig4_bhmf_z{z:.1f}.txt')
        if os.path.exists(f):
            d = np.loadtxt(f)
            tabs[z] = (d[:, 0], d[:, 1], d[:, 2], d[:, 3])
    if not tabs:
        raise FileNotFoundError(f'no TRINITY BHMF tables in {data_dir}')
    return tabs


def bhmf_benchmark(z, variant='best', tabs=None, grid=None):
    """
    TRINITY BHMF at redshift z (variant: 'best', 'lo' = 16th, 'hi' = 84th pct),
    log-linearly interpolated in z between tables (clamped at the ends).
    """
    tabs = load_trinity_bhmf() if tabs is None else tabs
    col = {'best': 1, 'lo': 2, 'hi': 3}[variant]
    zs = np.array(sorted(tabs))
    zc = float(np.clip(z, zs[0], zs[-1]))
    if zc != z:
        print(f'  NOTE: z={z:.2f} outside TRINITY range, clamped to z={zc:.2f}')
    i = int(np.clip(np.searchsorted(zs, zc) - 1, 0, len(zs) - 2))
    z0, z1 = zs[i], zs[i + 1]
    t0, t1 = tabs[z0], tabs[z1]
    grid = t0[0] if grid is None else grid
    l0 = np.log10(np.clip(np.interp(grid, t0[0], t0[col]), 1e-30, None))
    l1 = np.log10(np.clip(np.interp(grid, t1[0], t1[col]), 1e-30, None))
    w = (zc - z0) / (z1 - z0)
    return grid, 10.0 ** ((1 - w) * l0 + w * l1), dict(z_tables=(z0, z1), weight=w)


# ---------------------------------------------------------------------------
# Scatter handling
# ---------------------------------------------------------------------------
def _gauss_kernel(dx, sigma):
    k = np.exp(-0.5 * (np.arange(-int(np.ceil(5 * sigma / dx)),
                                 int(np.ceil(5 * sigma / dx)) + 1) * dx / sigma) ** 2)
    return k / k.sum()


def deconvolve(logm, phi, sigma, n_iter=200):
    """
    Richardson-Lucy deconvolution of a Gaussian log-mass scatter `sigma` (dex)
    from phi(logM) on a regular grid, so that  convolve(result, sigma) ~ phi.
    The ends are padded by extending the local log-slope, so the convolution
    does not see an artificial cliff at the grid edge. sigma <= 0 is a no-op.
    """
    if sigma <= 0:
        return phi.copy()
    dx = logm[1] - logm[0]
    npad = int(np.ceil(5 * sigma / dx))
    k = _gauss_kernel(dx, sigma)
    lp = np.log(np.clip(phi, 1e-300, None))
    nfit = min(10, len(lp) // 2)
    sl_lo = np.polyfit(np.arange(nfit), lp[:nfit], 1)[0]
    sl_hi = np.polyfit(np.arange(nfit), lp[-nfit:], 1)[0]
    pad_lo = lp[0] + sl_lo * np.arange(-npad, 0)
    pad_hi = lp[-1] + sl_hi * np.arange(1, npad + 1)
    d = np.exp(np.concatenate([pad_lo, lp, pad_hi]))
    est = d.copy()
    for _ in range(n_iter):
        conv = np.convolve(est, k, mode='same')
        est = est * np.convolve(d / np.clip(conv, 1e-300, None), k[::-1], mode='same')
    return est[npad:-npad]


def convolve_scatter(logm, phi, sigma):
    """Forward model: phi(logM) after Gaussian scatter sigma (dex), edge-padded."""
    if sigma <= 0:
        return phi.copy()
    dx = logm[1] - logm[0]
    npad = int(np.ceil(5 * sigma / dx))
    k = _gauss_kernel(dx, sigma)
    lp = np.log(np.clip(phi, 1e-300, None))
    nfit = min(10, len(lp) // 2)
    sl_lo = np.polyfit(np.arange(nfit), lp[:nfit], 1)[0]
    sl_hi = np.polyfit(np.arange(nfit), lp[-nfit:], 1)[0]
    d = np.exp(np.concatenate([lp[0] + sl_lo * np.arange(-npad, 0), lp,
                               lp[-1] + sl_hi * np.arange(1, npad + 1)]))
    return np.convolve(d, k, mode='same')[npad:-npad]


def cumulative_above(logm, phi):
    """N(>logM) [Mpc^-3] = integral of phi dlogM from logM to the top of the grid."""
    dx = logm[1] - logm[0]
    seg = 0.5 * (phi[1:] + phi[:-1]) * dx
    return np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])

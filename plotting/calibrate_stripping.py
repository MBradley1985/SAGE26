#!/usr/bin/env python
"""
calibrate_stripping.py -- calibrate the satellite-stripping timescale factor
(StrippingTimescaleFactor, f) against observed satellite HI content.

The stripping timescale t_strip = f * t_dyn(host) controls how fast satellites
lose their hot halo, and hence -- via the cooling supply -- their cold/HI gas.
Satellite HI-to-stellar mass ratio is therefore a direct observational handle.
This script overlays the model satellite HI ratio for a sweep of f against the
xGASS satellites-only measurement (Stevens et al. 2019 / Catinella et al. 2018).

Expects the f-sweep runs produced from input/millennium_f{05,10,20,40}.par:
    output/mill_f05, mill_f10, mill_f20, mill_f40   (f = 0.5, 1.0, 2.0, 4.0)
Runs that don't exist yet are skipped, so it can be used incrementally.

    python3 plotting/calibrate_stripping.py
"""
import os
# Allow reading HDF5 written by SAGE without a shared file lock (avoids the
# "unable to lock file" error on some filesystems / while a run is finishing).
os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')
import importlib.util

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

spec = importlib.util.spec_from_file_location('cs', 'plotting/compare_stripping.py')
cs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cs)
cs.SATELLITE_SUBSET = 'all'

# f value -> output directory
SWEEP = [(0.5, 'output/mill_f05'), (1.0, 'output/mill_f10'),
         (2.0, 'output/mill_f20'), (4.0, 'output/mill_f40')]
XGASS = 'data/Gas/HIMstar_xGASS_SatellitesOnly.dat'
BINW = 0.3


def model_retain_fraction(run, numer):
    """Fraction of satellites with numer>0 (e.g. a surviving hot halo) vs log10(M*)."""
    m = cs.satellite_mask(run); sm = run['StellarMass']; q = run[numer]
    sel = m & (sm > 0)
    logsm = np.log10(sm[sel]); has = q[sel] > 0
    edges = np.arange(9.0, 11.5, BINW)
    xc, fr = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        inb = (logsm >= a) & (logsm < b)
        if np.count_nonzero(inb) >= 10:
            xc.append(0.5 * (a + b)); fr.append(np.count_nonzero(has[inb]) / np.count_nonzero(inb))
    return np.array(xc), np.array(fr)


def model_median_ratio(run, numer, detected_only=True):
    """Median log10(numer/M*) for satellites vs log10(M*)."""
    m = cs.satellite_mask(run)
    sm, q = run['StellarMass'], run[numer]
    sel = m & (sm > 0) & (q > 0 if detected_only else np.ones_like(sm, bool))
    logsm = np.log10(sm[sel])
    with np.errstate(divide='ignore'):
        logratio = np.log10(np.where(q[sel] > 0, q[sel], np.nan) / sm[sel])
    edges = np.arange(9.0, 11.5, BINW)
    xc, med = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        inb = (logsm >= a) & (logsm < b)
        if np.count_nonzero(inb) >= 10:
            xc.append(0.5 * (a + b))
            med.append(np.nanmedian(logratio[inb]))
    return np.array(xc), np.array(med)


def load_xgass():
    d = np.loadtxt(XGASS)
    logm, med, e16, e84 = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
    lo = np.clip(med - e16, 0, None)   # med - 16th percentile value
    hi = np.clip(e84 - med, 0, None)   # 84th - med
    return logm, med, lo, hi


def main():
    import glob
    fig, (axH, axO) = plt.subplots(1, 2, figsize=(14, 6.2))

    colours = plt.cm.viridis(np.linspace(0.15, 0.85, len(SWEEP)))
    any_run = False
    for (f, d), col in zip(SWEEP, colours):
        pat = f'{d}/model_*.hdf5'
        if not (os.path.isdir(d) and glob.glob(pat)):
            print(f'  (skip f={f}: no runs in {d})'); continue
        try:
            r = cs.load_run(pat, None)
        except (OSError, BlockingIOError) as e:
            print(f'  (skip f={f}: {d} not readable [{e.__class__.__name__}])'); continue
        any_run = True
        lab = f'f = {f}  (t_strip = {f}·t_dyn)'
        # Panel A -- hot halo (what stripping acts on directly)
        xh, yh = model_retain_fraction(r, 'CGMgas')
        axH.plot(xh, yh, '-o', color=col, ms=4, lw=2, label=lab)
        # Panel B -- observable HI
        xo, yo = model_median_ratio(r, 'H1gas')
        axO.plot(xo, yo, '-o', color=col, ms=4, lw=2, label=lab)
        print(f'  f={f}: {np.count_nonzero(cs.satellite_mask(r))} satellites')

    if not any_run:
        print('\nNo f-sweep runs found. Run: ./sage input/millennium_f{05,10,20,40}.par')
        return

    axH.set(xlabel=r'$\log_{10}(M_*/M_\odot)$',
            ylabel='fraction of satellites retaining a hot halo (CGM>0)',
            xlim=(9.0, 11.3), ylim=(0, 1))
    axH.set_title('A. Hot halo (stripping acts here directly)\nf SEPARATES the curves')
    axH.legend(fontsize=9); axH.grid(alpha=0.3)

    xm, ym, ylo, yhi = load_xgass()
    axO.errorbar(xm, ym, yerr=[ylo, yhi], fmt='ks', ms=6, capsize=3,
                 label='xGASS satellites (Stevens+19)', zorder=5)
    axO.set(xlabel=r'$\log_{10}(M_*/M_\odot)$',
            ylabel=r'$\log_{10}(M_{\rm HI}/M_*)$  (satellites, observable)',
            xlim=(9.0, 11.3), ylim=(-2.5, 0.5))
    axO.set_title('B. Observable HI\nf curves OVERLAP -- observable is insensitive')
    axO.legend(fontsize=9); axO.grid(alpha=0.3)

    fig.suptitle('Stripping-timescale factor f moves the hot halo, not the observable cold gas',
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig('stripping_calibration.png', dpi=150, bbox_inches='tight')
    print('Wrote stripping_calibration.png')


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""
compare_runs.py -- diff two SAGE26 runs across the diagnostics the paper plots.

Computes the statistics behind each paper figure (mass functions, binned median
scaling relations, cosmic histories, regime fractions) for a reference run and a
test run, and reports how far the test run moves.  Intended for verifying that a
model change does not disturb diagnostics outside the ones already inspected.

Deviations are reported in the natural units of each diagnostic: dex for mass
functions and log-log relations, absolute for fractions, dex for medians of
logged quantities.

Usage:
    python plotting/compare_runs.py REF=dir TEST=dir [--tol 0.05] [--out report.txt]
"""

import argparse
import glob
import os

import h5py as h5
import numpy as np


# ----------------------------------------------------------------------

def run_meta(directory):
    files = sorted(glob.glob(os.path.join(directory, 'model_*.hdf5')))
    if not files:
        raise SystemExit(f'no model files in {directory}')
    fvp = 0.0
    for fn in files:
        with h5.File(fn, 'r') as f:
            fvp += float(f['Header/Runtime'].attrs['frac_volume_processed'])
    with h5.File(files[0], 'r') as f:
        sim = dict(f['Header/Simulation'].attrs)
        zz = np.array(f['Header/snapshot_redshifts'])
        snaps = sorted(int(k.split('_')[1]) for k in f.keys() if k.startswith('Snap_'))
        fields = set(f[f'Snap_{snaps[-1]}'].keys())
    h = float(sim['hubble_h'])
    return dict(files=files, h=h, zz=zz, snaps=snaps, fields=fields,
                volume=(float(sim['box_size']) / h) ** 3 * fvp)


def load(meta, snap, keys):
    keys = [k for k in keys if k in meta['fields']]
    out = {}
    for fn in meta['files']:
        with h5.File(fn, 'r') as f:
            g = f[f'Snap_{snap}']
            if g['StellarMass'].shape[0] == 0:
                continue
            for k in keys:
                out.setdefault(k, []).append(np.array(g[k]))
    if not out:
        return None
    return {k: np.concatenate(v) for k, v in out.items()}


def nearest(meta, z):
    return min([s for s in meta['snaps'] if s < len(meta['zz'])],
               key=lambda s: abs(meta['zz'][s] - z))


# ---------------------------------------------------------------- metrics

MIN_COUNT_PER_BIN = 10   # below this a bin is Poisson noise, not a model difference


def massfunc(values_msun, volume, bins):
    """log10 phi per bin, with sparse bins masked to NaN so Poisson scatter in
    the extreme tails is not mistaken for a systematic shift."""
    v = values_msun[values_msun > 0]
    n, _ = np.histogram(np.log10(v), bins=bins)
    phi = n / volume / (bins[1] - bins[0])
    with np.errstate(divide='ignore'):
        out = np.where(phi > 0, np.log10(phi), np.nan)
    return np.where(n >= MIN_COUNT_PER_BIN, out, np.nan)


def binned_median(x, y, bins, minc=20):
    out = np.full(len(bins) - 1, np.nan)
    for i in range(len(bins) - 1):
        m = (x >= bins[i]) & (x < bins[i + 1])
        if m.sum() >= minc:
            out[i] = np.median(y[m])
    return out


def binned_fraction(x, flag, bins, minc=20):
    out = np.full(len(bins) - 1, np.nan)
    for i in range(len(bins) - 1):
        m = (x >= bins[i]) & (x < bins[i + 1])
        if m.sum() >= minc:
            out[i] = np.mean(flag[m])
    return out


# ---------------------------------------------------------------- suite

MSTAR_BINS = np.arange(7.0, 12.5, 0.25)
MVIR_BINS = np.arange(10.0, 15.0, 0.25)


def diagnostics(meta):
    """Return {name: (array, unit)} for every diagnostic we can compute."""
    d = {}
    MC = 1e10 / meta['h']
    V = meta['volume']

    KEYS = ['StellarMass', 'ColdGas', 'MetalsColdGas', 'HotGas', 'CGMgas',
            'EjectedMass', 'BlackHoleMass', 'BulgeMass', 'Mvir', 'Vvir',
            'SfrDisk', 'SfrBulge', 'Type', 'H1gas', 'H2gas', 'MassLoading',
            'Regime', 'FFBRegime', 'tcool_over_tff', 'DiskRadius', 'BulgeRadius',
            'IntraClusterStars', 'MetalsStellarMass', 'Rvir']

    # ---- z = 0 diagnostics ----
    g = load(meta, nearest(meta, 0.0), KEYS)
    sm = g['StellarMass'] * MC
    mv = g['Mvir'] * MC
    cg = g['ColdGas'] * MC
    sfr = g['SfrDisk'] + g['SfrBulge']
    pos = sm > 0
    lsm = np.log10(np.where(pos, sm, 1.0))
    with np.errstate(divide='ignore', invalid='ignore'):
        ssfr = np.where(pos, sfr / sm, 0.0)
    quiescent = (ssfr < 1e-11).astype(float)

    d['SMF z=0'] = (massfunc(sm, V, MSTAR_BINS), 'dex')
    d['SMF z=0 star-forming'] = (massfunc(sm[ssfr >= 1e-11], V, MSTAR_BINS), 'dex')
    d['SMF z=0 quiescent'] = (massfunc(sm[(ssfr < 1e-11) & pos], V, MSTAR_BINS), 'dex')
    d['quiescent fraction vs m*'] = (binned_fraction(lsm[pos], quiescent[pos], MSTAR_BINS), 'abs')

    if 'H1gas' in g:
        d['HI mass function z=0'] = (massfunc(g['H1gas'] * MC, V, MSTAR_BINS), 'dex')
    if 'H2gas' in g:
        d['H2 mass function z=0'] = (massfunc(g['H2gas'] * MC, V, MSTAR_BINS), 'dex')
    d['cold gas mass function z=0'] = (massfunc(cg, V, MSTAR_BINS), 'dex')

    sel = pos & (cg > 0) & (g['MetalsColdGas'] > 0) & (cg / (sm + cg) > 0.1)
    Z = np.log10((g['MetalsColdGas'][sel] / g['ColdGas'][sel]) / 0.02) + 9.0
    d['MZR z=0'] = (binned_median(lsm[sel], Z, MSTAR_BINS), 'dex')

    with np.errstate(divide='ignore', invalid='ignore'):
        d['cold gas fraction vs m*'] = (
            binned_median(lsm[pos], (cg / (cg + sm))[pos], MSTAR_BINS), 'abs')
        # Median over galaxies with a detected reservoir only.  Flooring zeros at
        # 1e-6 instead would make the median jump by orders of magnitude whenever
        # the zero-reservoir fraction crosses 50 per cent, which is a property of
        # the metric rather than of the model.
        for lbl, key in [('HI', 'H1gas'), ('H2', 'H2gas')]:
            r = g[key] * MC / np.where(pos, sm, 1.0)
            hsel = pos & (r > 0)
            d[f'{lbl}/m* vs m*'] = (
                binned_median(lsm[hsel], np.log10(r[hsel]), MSTAR_BINS), 'dex')

    sfsel = pos & (sfr > 0)
    d['main sequence z=0'] = (
        binned_median(lsm[sfsel], np.log10(sfr[sfsel]), MSTAR_BINS), 'dex')

    bh = g['BlackHoleMass'] * MC
    bul = g['BulgeMass'] * MC
    bsel = (bh > 0) & (bul > 1e8)
    d['BH-bulge relation'] = (
        binned_median(np.log10(bul[bsel]), np.log10(bh[bsel]), MSTAR_BINS), 'dex')

    cen = (g['Type'] == 0) & (mv > 0) & pos
    d['stellar-halo mass relation'] = (
        binned_median(np.log10(mv[cen]), np.log10(sm[cen] / mv[cen]), MVIR_BINS), 'dex')
    d['baryon fraction vs Mvir'] = (
        binned_median(np.log10(mv[cen]),
                      ((g['StellarMass'] + g['ColdGas'] + g['HotGas'] + g['CGMgas']
                        + g['EjectedMass'] + g['IntraClusterStars']) / g['Mvir'])[cen],
                      MVIR_BINS), 'abs')
    for name, key in [('hot gas', 'HotGas'), ('CGM gas', 'CGMgas'),
                      ('ejected', 'EjectedMass'), ('cold gas', 'ColdGas')]:
        d[f'{name} fraction vs Mvir'] = (
            binned_median(np.log10(mv[cen]), (g[key] / g['Mvir'])[cen], MVIR_BINS), 'abs')

    if 'DiskRadius' in g:
        rsel = pos & (g['DiskRadius'] > 0)
        d['disk size vs m*'] = (
            binned_median(lsm[rsel], np.log10(g['DiskRadius'][rsel] * 1e3 / meta['h']),
                          MSTAR_BINS), 'dex')
    if 'BulgeRadius' in g:
        bsel2 = (bul > 1e8) & (g['BulgeRadius'] > 0)
        d['bulge size vs bulge mass'] = (
            binned_median(np.log10(bul[bsel2]),
                          np.log10(g['BulgeRadius'][bsel2] * 1e3 / meta['h']), MSTAR_BINS), 'dex')
    if 'Regime' in g:
        d['CGM-regime fraction vs Mvir'] = (
            binned_fraction(np.log10(np.maximum(mv, 1.0))[cen],
                            (g['Regime'][cen] == 0).astype(float), MVIR_BINS), 'abs')
    if 'tcool_over_tff' in g:
        t = g['tcool_over_tff'][cen]
        ok = t > 0
        d['tcool/tff vs Mvir'] = (
            binned_median(np.log10(mv[cen])[ok], np.log10(t[ok]), MVIR_BINS), 'dex')
    if 'MassLoading' in g:
        ml = g['MassLoading']
        msel = (ml > 0) & (sfr > 0) & (g['Vvir'] > 0)
        d['mass loading vs Vvir z=0'] = (
            binned_median(np.log10(g['Vvir'][msel]), np.log10(ml[msel]),
                          np.arange(1.2, 2.8, 0.1)), 'dex')

    # ---- higher-redshift diagnostics ----
    for zt in [1.0, 2.0, 3.0, 4.0, 6.0]:
        gz = load(meta, nearest(meta, zt), ['StellarMass', 'SfrDisk', 'SfrBulge',
                                            'ColdGas', 'H1gas', 'H2gas', 'FFBRegime'])
        if gz is None:
            continue
        smz = gz['StellarMass'] * MC
        d[f'SMF z={zt:.0f}'] = (massfunc(smz, V, MSTAR_BINS), 'dex')
        sz = smz > 0
        sfz = gz['SfrDisk'] + gz['SfrBulge']
        msel = sz & (sfz > 0)
        d[f'main sequence z={zt:.0f}'] = (
            binned_median(np.log10(smz[msel]), np.log10(sfz[msel]), MSTAR_BINS), 'dex')
        if 'FFBRegime' in gz:
            d[f'FFB fraction z={zt:.0f}'] = (
                binned_fraction(np.log10(np.where(sz, smz, 1.0))[sz],
                                (gz['FFBRegime'][sz] == 1).astype(float), MSTAR_BINS), 'abs')

    # ---- cosmic histories ----
    z, sfrd, smd, ejd, hid = [], [], [], [], []
    for s in sorted(meta['snaps']):
        if s >= len(meta['zz']):
            continue
        gs = load(meta, s, ['StellarMass', 'SfrDisk', 'SfrBulge', 'EjectedMass', 'H1gas'])
        if gs is None or gs['StellarMass'].size < 20:
            continue
        z.append(meta['zz'][s])
        sfrd.append((gs['SfrDisk'] + gs['SfrBulge']).sum() / V)
        smd.append(gs['StellarMass'].sum() * MC / V)
        ejd.append(gs['EjectedMass'].sum() * MC / V)
        hid.append(gs['H1gas'].sum() * MC / V if 'H1gas' in gs else np.nan)
    o = np.argsort(z)
    lg = lambda a: np.log10(np.maximum(np.array(a)[o], 1e-30))
    d['SFRD history'] = (lg(sfrd), 'dex')
    d['stellar mass density history'] = (lg(smd), 'dex')
    d['ejected mass density history'] = (lg(ejd), 'dex')
    d['HI density history'] = (lg(hid), 'dex')
    d['_zgrid'] = (np.array(z)[o], 'z')
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs=2, help='REF=dir TEST=dir')
    ap.add_argument('--tol', type=float, default=0.05,
                    help='flag threshold (dex for log quantities, absolute for fractions)')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    labels, metas = [], []
    for spec in args.runs:
        lab, _, dd = spec.partition('=')
        labels.append(lab)
        metas.append(run_meta(dd))

    a = diagnostics(metas[0])
    b = diagnostics(metas[1])
    zgrid = a.pop('_zgrid')[0]
    b.pop('_zgrid', None)

    lines = []
    P = lines.append
    P(f"Diagnostic comparison: {labels[1]} relative to {labels[0]}")
    P(f"  volume {metas[0]['volume']:.4e} Mpc^3, flag threshold {args.tol}")
    P("")
    P(f"{'diagnostic':>34} {'unit':>5} {'nbins':>6} {'mean|d|':>9} {'max|d|':>9} {'where':>10}  flag")
    P("-" * 92)
    flagged = []
    for name in a:
        x, unit = a[name]
        if name not in b:
            continue
        y = b[name][0]
        n = min(len(x), len(y))
        x, y = x[:n], y[:n]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() == 0:
            P(f"{name:>34} {unit:>5} {0:>6}      (no overlapping bins)")
            continue
        dd = np.abs(y[ok] - x[ok])
        i = int(np.argmax(dd))
        idx = np.where(ok)[0][i]
        if 'history' in name:
            where = f"z={zgrid[idx]:.2f}" if idx < len(zgrid) else f"bin {idx}"
        elif 'Mvir' in name:
            where = f"logMh={MVIR_BINS[idx] + 0.125:.2f}"
        elif 'Vvir' in name:
            where = f"logV={1.2 + 0.1 * idx + 0.05:.2f}"
        else:
            where = f"logM={MSTAR_BINS[idx] + 0.125:.2f}"
        if 'history' in name and len(zgrid) >= ok.sum():
            zsub = zgrid[:n][ok]
            lowz = zsub <= 3.0
            extra = f" | z<=3: {dd[lowz].max():.4f}" if lowz.any() else ""
        else:
            extra = ""
        flag = 'FLAG' if dd.max() > args.tol else ''
        if flag:
            flagged.append((name, dd.max(), where))
        P(f"{name:>34} {unit:>5} {ok.sum():>6} {dd.mean():9.4f} {dd.max():9.4f} {where:>10}  {flag}{extra}")
    P("")
    if flagged:
        P(f"{len(flagged)} diagnostic(s) exceed the threshold:")
        for nm, v, w in sorted(flagged, key=lambda t: -t[1]):
            P(f"    {nm:>34}  max |delta| = {v:.4f}  at {w}")
    else:
        P(f"No diagnostic exceeds |delta| = {args.tol}.")

    text = "\n".join(lines)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w') as f:
            f.write(text + "\n")
        print(f"\n  Saved: {args.out}")


if __name__ == '__main__':
    main()

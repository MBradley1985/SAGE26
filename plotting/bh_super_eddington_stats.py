#!/usr/bin/env python3
"""
bh_super_eddington_stats.py
===========================

Statistics of super-Eddington black-hole accretion events (and of the
LRD-selected subsets of them) for one or many SAGE26 scenario runs.

Each run is a directory holding model_*.hdf5 (one file, or the output split
over many -- they are streamed one after another and combined). For every run
the script writes

    <out-dir>/<run>/super_eddington_stats.json   machine-readable summary
    <out-dir>/<run>/summary.md                   human-readable summary

and with --combine it also builds <out-dir>/README.md: a cross-run comparison
table plus every per-run summary.

DEFINITIONS (match plotting/bh_lrd_analysis.py)
-----------------------------------------------
An "event" is one (galaxy, snapshot-column) cell of the per-galaxy accretion
history of the final catalogue with BHMaxaccretionRate > 0:

    Mdot_BH   = BHMaxaccretionRate[:, c]   (the rate actually applied)
    Mdot_Edd  = BHEddingtonRateLimit[:, c] (plain Eddington rate, eta = 0.1)
    M_BH      = BHMassatAccretion[:, c]    (BH mass at that episode)
    lambda    = Mdot_BH / Mdot_Edd
    channel   = BHAccretionType[:, c]      0 Radio, 1 Merger, 2 Disk instability

Super-Eddington event : lambda > 1 (+ --edd-tol, so events pinned exactly at
                        the Eddington cap by an Eddington-limited run are not
                        counted as "super").
LRD (full)    : Mdot_BH >= BHAR floor (0.1 Msun/yr)  AND  Mdot_BH >= Mdot_Edd
                AND  f_BH = M_BH/M_star >= 0.03        (Chen & Mo 2026; "red")
LRD (partial) : same, but f_BH < 0.03                  ("blue")
The same two selections are repeated for the alternative floor 0.05 Msun/yr.
Events recorded while the BH is still at its seed mass (< 1% grown) are
dropped, as everywhere else in the plotting code (--keep-seeds disables this).

CAVEAT -- host properties
-------------------------
Stellar mass, halo mass and central/satellite status are read from the FINAL
catalogue (the highest populated Snap_N, normally z = 0), not at the epoch of
the event: the per-epoch host is not stored with the accretion history. They
describe the galaxy the black hole ended up in. f_BH uses M_BH at the event
divided by that final stellar mass, exactly as bh_lrd_analysis.py does.

USAGE
-----
    # one run (a directory, or a quoted glob)
    python plotting/bh_super_eddington_stats.py /fred/oz004/jwillingham/miniuchuu_vanilla

    # every run under a root
    python plotting/bh_super_eddington_stats.py --root /fred/oz004/jwillingham --out-dir super_edd

    # rebuild the combined README from the JSON already written
    python plotting/bh_super_eddington_stats.py --combine super_edd

Memory is bounded: galaxies are streamed in chunks and quantities are
accumulated in fine histograms (0.01-0.02 dex), so percentiles are accurate
to that resolution however many events there are.
"""

import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np
import h5py

# ----------------------------------------------------------------------------
# constants (same values as bh_lrd_analysis.py)
# ----------------------------------------------------------------------------
SEC_PER_YEAR = 365.25 * 24 * 3600
UNIT_TIME_IN_S = 3.086e19             # SAGE code time unit
LRD_FBH_THRESH = 0.03
BHAR_FLOORS = (0.1, 0.05)             # Msun/yr; first one is the default
SEED_GROWTH_THRESHOLD = 0.01
ACC_NAMES = {0: 'Radio', 1: 'Merger', 2: 'Disk instability'}
HOST_NAMES = {0: 'Central', 1: 'Satellite', 2: 'Orphan'}
Z_BINS = [(0, 1), (1, 2), (2, 4), (4, 6), (6, 8), (8, np.inf)]
CHUNK_ROWS = 50_000
TOP_K = 10

# name -> (lo, hi, n_bins) of the accumulation histograms
HIST_SPECS = {
    'log_mbh':        (0.0, 12.0, 600),    # log10 M_BH at the event [Msun]
    'log_mbh_final':  (0.0, 12.0, 600),    # log10 M_BH of the final catalogue BH
    'log_lambda':     (-0.2, 6.0, 620),    # log10 Mdot/Mdot_Edd
    'log_mdot':       (-8.0, 8.0, 800),    # log10 Mdot [Msun/yr]
    'log_mstar':      (4.0, 13.0, 450),    # log10 host M_star (final catalogue)
    'log_mvir':       (6.0, 16.0, 500),    # log10 Mvir of the host subhalo
    'log_mhost':      (6.0, 16.0, 500),    # log10 CentralMvir (parent halo)
    'log_fbh':        (-8.0, 3.0, 550),    # log10 M_BH,event / M_star
}
QUANTITY_LABELS = {
    'log_mbh': 'log10 M_BH at event [Msun]',
    'log_mbh_final': 'log10 M_BH of final BH [Msun]',
    'log_lambda': 'log10 Eddington ratio (Mdot/Mdot_Edd)',
    'log_mdot': 'log10 Mdot_BH [Msun/yr]',
    'log_mstar': 'log10 host M_star [Msun]',
    'log_mvir': 'log10 host subhalo Mvir [Msun]',
    'log_mhost': 'log10 parent halo Mvir [Msun]',
    'log_fbh': 'log10 f_BH = M_BH/M_star',
}


def natural_key(path):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', path)]


# ----------------------------------------------------------------------------
# accumulation
# ----------------------------------------------------------------------------
class Hist:
    """Fixed-bin histogram with exact count / min / max / sum of finite values."""

    def __init__(self, lo, hi, nb):
        self.lo, self.hi, self.nb = lo, hi, nb
        self.counts = np.zeros(nb, dtype=np.int64)
        self.n = 0
        self.vmin = np.inf
        self.vmax = -np.inf
        self.total = 0.0

    def add(self, v):
        v = v[np.isfinite(v)]
        if v.size == 0:
            return
        self.n += v.size
        self.vmin = min(self.vmin, float(v.min()))
        self.vmax = max(self.vmax, float(v.max()))
        self.total += float(v.sum(dtype=np.float64))
        h, _ = np.histogram(np.clip(v, self.lo, self.hi - 1e-9),
                            bins=self.nb, range=(self.lo, self.hi))
        self.counts += h

    def quantile(self, q):
        if self.n == 0:
            return None
        c = np.cumsum(self.counts)
        t = q * self.n
        i = min(int(np.searchsorted(c, t)), self.nb - 1)
        prev = c[i - 1] if i > 0 else 0
        frac = (t - prev) / max(int(self.counts[i]), 1)
        w = (self.hi - self.lo) / self.nb
        val = self.lo + (i + min(max(frac, 0.0), 1.0)) * w
        return float(min(max(val, self.vmin), self.vmax))

    def summary(self):
        if self.n == 0:
            return {'n': 0}
        return {'n': int(self.n), 'min': self.vmin, 'p16': self.quantile(0.16),
                'median': self.quantile(0.5), 'p84': self.quantile(0.84),
                'max': self.vmax, 'mean': self.total / self.n}


class Accumulator:
    """Everything we report about one group of events."""

    def __init__(self, name, snap_z):
        self.name = name
        self.snap_z = snap_z
        self.n_events = 0
        self.n_galaxies = 0
        self.hists = {k: Hist(*v) for k, v in HIST_SPECS.items()}
        self.acc_counts = {k: 0 for k in (0, 1, 2, -1)}
        self.host_counts = {k: 0 for k in (0, 1, 2, -1)}
        self.snap_counts = np.zeros(len(snap_z), dtype=np.int64)
        self.extremes = []     # list of dicts, top by lambda

    def add(self, ev, rows):
        n = len(ev['snap'])
        if n == 0:
            return
        self.n_events += n
        self.n_galaxies += int(np.unique(rows).size)
        for k, h in self.hists.items():
            h.add(ev[k])
        for code in (0, 1, 2):
            self.acc_counts[code] += int(np.count_nonzero(ev['acc'] == code))
            self.host_counts[code] += int(np.count_nonzero(ev['host'] == code))
        self.acc_counts[-1] += n - int(np.isin(ev['acc'], (0, 1, 2)).sum())
        self.host_counts[-1] += n - int(np.isin(ev['host'], (0, 1, 2)).sum())
        np.add.at(self.snap_counts, ev['snap'], 1)

        k = min(TOP_K, n)
        idx = np.argpartition(-ev['lambda'], k - 1)[:k]
        for i in idx:
            self.extremes.append({
                'galaxy_index': int(ev['gidx'][i]), 'snapshot': int(ev['snap'][i]),
                'z': float(self.snap_z[ev['snap'][i]]),
                'log_mbh': float(ev['log_mbh'][i]), 'lambda': float(ev['lambda'][i]),
                'mdot_msun_yr': float(10 ** ev['log_mdot'][i]),
                'log_mstar': _f(ev['log_mstar'][i]), 'log_mvir': _f(ev['log_mvir'][i]),
                'channel': ACC_NAMES.get(int(ev['acc'][i]), 'unknown'),
                'host': HOST_NAMES.get(int(ev['host'][i]), 'unknown'),
            })
        self.extremes.sort(key=lambda d: -d['lambda'])
        del self.extremes[TOP_K:]

    def summary(self):
        out = {'name': self.name, 'n_events': self.n_events,
               'n_galaxies': self.n_galaxies,
               'quantities': {k: h.summary() for k, h in self.hists.items()},
               'channel_counts': {ACC_NAMES.get(k, 'unknown'): v
                                  for k, v in self.acc_counts.items() if k in ACC_NAMES or v},
               'host_counts': {HOST_NAMES.get(k, 'unknown'): v
                               for k, v in self.host_counts.items() if k in HOST_NAMES or v},
               'redshift_bin_counts': {}, 'top_events': self.extremes}
        for lo, hi in Z_BINS:
            sel = (self.snap_z >= lo) & (self.snap_z < hi)
            out['redshift_bin_counts'][_zlabel(lo, hi)] = int(self.snap_counts[sel].sum())
        return out


def _f(x):
    x = float(x)
    return x if np.isfinite(x) else None


def _zlabel(lo, hi):
    return f'z>={lo:g}' if not np.isfinite(hi) else f'{lo:g}<=z<{hi:g}'


# ----------------------------------------------------------------------------
# reading
# ----------------------------------------------------------------------------
def find_files(path):
    """model_*.hdf5 of a run directory / glob / single file, natural-sorted."""
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, 'model_*.hdf5'))
        if not files and os.path.exists(os.path.join(path, 'model.hdf5')):
            files = [os.path.join(path, 'model.hdf5')]
    elif glob.has_magic(path):
        files = glob.glob(path)
    else:
        files = [path] if os.path.exists(path) else []
    return sorted(files, key=natural_key)


def _attr(v):
    if isinstance(v, bytes):
        return v.decode(errors='replace')
    if isinstance(v, np.generic):
        return v.item()
    return v


def read_header(f):
    hdr = {'hubble_h': 0.73, 'box_size': None, 'redshifts': None,
           'frac_volume': 1.0, 'runtime': {}}
    with h5py.File(f, 'r') as hf:
        for grp in ('Header/Simulation', 'Header'):
            if grp in hf:
                a = hf[grp].attrs
                for key in ('hubble_h', 'HubbleParam', 'Hubble_h'):
                    if key in a:
                        hdr['hubble_h'] = float(a[key])
                if 'box_size' in a:
                    hdr['box_size'] = float(a['box_size'])
        if 'Header/snapshot_redshifts' in hf:
            hdr['redshifts'] = np.array(hf['Header/snapshot_redshifts'])
        if 'Header/Runtime' in hf:
            for k, v in hf['Header/Runtime'].attrs.items():
                if re.search(r'Edd|AGN|BH|Seed|Accretion|Scheme', k, re.I):
                    hdr['runtime'][k] = _attr(v)
    return hdr


def frac_volume(f):
    try:
        with h5py.File(f, 'r') as hf:
            if 'Header/Runtime' in hf and 'frac_volume_processed' in hf['Header/Runtime'].attrs:
                return float(hf['Header/Runtime'].attrs['frac_volume_processed'])
    except OSError:
        pass
    return 1.0


def pick_catalogue(hf):
    snaps = sorted([k for k in hf.keys() if k.startswith('Snap_')],
                   key=lambda s: int(s.split('_')[1]))
    for s in reversed(snaps):
        try:
            if hf[s]['BlackHoleMass'].shape[0] > 0 and 'BHMaxaccretionRate' in hf[s]:
                return s
        except KeyError:
            continue
    return None


class Run:
    """All accumulators + counters of one run."""

    def __init__(self, name, hdr, snap_z, keep_seeds, edd_tol):
        self.name = name
        self.hdr = hdr
        self.snap_z = snap_z
        self.keep_seeds = keep_seeds
        self.edd_tol = edd_tol
        h = hdr['hubble_h']
        self.mass_conv = 1e10 / h
        self.rate_conv = self.mass_conv / (UNIT_TIME_IN_S / SEC_PER_YEAR)
        A = lambda n: Accumulator(n, snap_z)
        self.groups = {'super_edd': A('Super-Eddington events (lambda > 1)')}
        for ch in (0, 1, 2):
            self.groups[f'super_edd/ch={ch}'] = A(f'Super-Eddington / {ACC_NAMES[ch]}')
        for lo, hi in Z_BINS:
            self.groups[f'super_edd/z={_zlabel(lo, hi)}'] = A(f'Super-Eddington / {_zlabel(lo, hi)}')
        for fl in BHAR_FLOORS:
            self.groups[f'lrd_full@{fl}'] = A(f'LRD full (f_BH>=3%, Mdot>={fl})')
            self.groups[f'lrd_partial@{fl}'] = A(f'LRD partial (f_BH<3%, Mdot>={fl})')
        self.n_files = 0
        self.n_galaxies = 0
        self.n_galaxies_bh = 0
        self.n_events_all = 0
        self.n_events_seed = 0
        self.n_events_at_cap = 0
        self.n_gal_any_event = 0
        self.n_gal_super = 0
        self.skipped = []
        self.catalogues = set()
        self.volume_h3 = 0.0

    # ------------------------------------------------------------------
    def process_file(self, f):
        try:
            hf = h5py.File(f, 'r')
        except OSError as e:
            self.skipped.append((f, str(e)))
            print(f'  ! skipping unreadable {f}: {e}', flush=True)
            return
        with hf:
            cat = pick_catalogue(hf)
            if cat is None:
                self.skipped.append((f, 'no populated catalogue with BH histories'))
                print(f'  ! {f}: no populated catalogue with BH histories', flush=True)
                return
            g = hf[cat]
            need = ['BHMaxaccretionRate', 'BHEddingtonRateLimit', 'BHMassatAccretion',
                    'BHAccretionType', 'BlackHoleMass', 'StellarMass']
            miss = [k for k in need if k not in g]
            if miss:
                self.skipped.append((f, f'missing {miss}'))
                print(f'  ! {f}: missing {miss}', flush=True)
                return
            self.catalogues.add(cat)
            self.n_files += 1
            ngal = g['BlackHoleMass'].shape[0]
            opt = {k: (g[k] if k in g else None) for k in
                   ('Mvir', 'CentralMvir', 'Type', 'BHSeedMass', 'GalaxyIndex')}
            for r0 in range(0, ngal, CHUNK_ROWS):
                r1 = min(r0 + CHUNK_ROWS, ngal)
                self._chunk(g, opt, r0, r1)

    def _chunk(self, g, opt, r0, r1):
        n = r1 - r0
        mdot = g['BHMaxaccretionRate'][r0:r1]
        edd = g['BHEddingtonRateLimit'][r0:r1]
        if mdot.ndim != 2:
            raise ValueError('BHMaxaccretionRate is not a 2D [Ngal, snapshots] history array')
        mbh_ev = g['BHMassatAccretion'][r0:r1]
        acc = g['BHAccretionType'][r0:r1]
        bh_final = g['BlackHoleMass'][r0:r1].astype(np.float64)
        star = g['StellarMass'][r0:r1].astype(np.float64)
        seed = (opt['BHSeedMass'][r0:r1].astype(np.float64)
                if opt['BHSeedMass'] is not None else np.zeros(n))
        mvir = (opt['Mvir'][r0:r1] if opt['Mvir'] is not None else np.full(n, np.nan))
        cmvir = (opt['CentralMvir'][r0:r1] if opt['CentralMvir'] is not None
                 else np.full(n, np.nan))
        gtype = (opt['Type'][r0:r1] if opt['Type'] is not None else np.full(n, -1))
        gidx = (opt['GalaxyIndex'][r0:r1] if opt['GalaxyIndex'] is not None
                else np.arange(r0, r1))

        self.n_galaxies += n
        self.n_galaxies_bh += int(np.count_nonzero(bh_final > 0))

        ev = mdot > 0
        edd_ok = edd > 0
        grown = np.ones_like(ev) if self.keep_seeds else \
            (mbh_ev > (seed[:, None] * (1.0 + SEED_GROWTH_THRESHOLD)))
        self.n_events_all += int(np.count_nonzero(ev))
        self.n_events_seed += int(np.count_nonzero(ev & ~grown))
        ev &= grown
        self.n_gal_any_event += int(np.count_nonzero(ev.any(axis=1)))

        # candidates: everything that can matter downstream (mdot >= Mdot_Edd)
        cand = ev & edd_ok & (mdot >= edd)
        rows, cols = np.nonzero(cand)
        if rows.size == 0:
            return
        m = mdot[rows, cols].astype(np.float64)
        e = edd[rows, cols].astype(np.float64)
        lam = m / e
        self.n_events_at_cap += int(np.count_nonzero(np.abs(lam - 1.0) <= self.edd_tol))

        mbh = mbh_ev[rows, cols].astype(np.float64) * self.mass_conv
        mdot_phys = m * self.rate_conv
        st = star[rows] * self.mass_conv
        with np.errstate(divide='ignore', invalid='ignore'):
            fbh = mbh / np.where(st > 0, st, np.inf)
            log_mstar = np.where(st > 0, np.log10(st), np.nan)
            log_fbh = np.where(st > 0, np.log10(fbh), np.nan)
            log_mvir = np.where(mvir[rows] > 0, np.log10(mvir[rows].astype(np.float64) * self.mass_conv), np.nan)
            log_mhost = np.where(cmvir[rows] > 0, np.log10(cmvir[rows].astype(np.float64) * self.mass_conv), np.nan)
            log_fin = np.where(bh_final[rows] > 0, np.log10(bh_final[rows] * self.mass_conv), np.nan)
            event = {
                'log_mbh': np.log10(mbh), 'log_mbh_final': log_fin,
                'log_lambda': np.log10(lam), 'log_mdot': np.log10(mdot_phys),
                'log_mstar': log_mstar, 'log_mvir': log_mvir, 'log_mhost': log_mhost,
                'log_fbh': log_fbh, 'lambda': lam, 'snap': cols,
                'acc': np.rint(acc[rows, cols]).astype(int),
                'host': gtype[rows].astype(int), 'gidx': gidx[rows],
            }

        def take(mask):
            return {k: v[mask] for k, v in event.items()}

        sup = lam > 1.0 + self.edd_tol
        self.n_gal_super += int(np.unique(rows[sup]).size)
        self.groups['super_edd'].add(take(sup), rows[sup])
        for ch in (0, 1, 2):
            s = sup & (event['acc'] == ch)
            self.groups[f'super_edd/ch={ch}'].add(take(s), rows[s])
        z_ev = self.snap_z[cols]
        for lo, hi in Z_BINS:
            s = sup & (z_ev >= lo) & (z_ev < hi)
            self.groups[f'super_edd/z={_zlabel(lo, hi)}'].add(take(s), rows[s])

        # LRD selections (mdot >= Mdot_Edd already holds for every candidate)
        big_f = (st > 0) & (fbh >= LRD_FBH_THRESH)
        for fl in BHAR_FLOORS:
            base = mdot_phys >= fl
            for key, s in ((f'lrd_full@{fl}', base & big_f),
                           (f'lrd_partial@{fl}', base & ~big_f)):
                self.groups[key].add(take(s), rows[s])

    # ------------------------------------------------------------------
    def summary(self):
        h = self.hdr['hubble_h']
        vol_phys = (self.hdr['box_size'] / h) ** 3 * self.volume_h3 \
            if self.hdr['box_size'] else None
        s = {
            'run': self.name,
            'files_read': self.n_files,
            'files_skipped': [{'file': f, 'reason': r} for f, r in self.skipped],
            'catalogue_groups': sorted(self.catalogues),
            'hubble_h': h, 'box_size_mpc_h': self.hdr['box_size'],
            'volume_mpc3': vol_phys,
            'runtime_parameters': self.hdr['runtime'],
            'definitions': {
                'super_eddington': f'Mdot/Mdot_Edd > 1 + {self.edd_tol}',
                'bhar_floors_msun_yr': list(BHAR_FLOORS),
                'fbh_threshold': LRD_FBH_THRESH,
                'seed_events_dropped': not self.keep_seeds,
            },
            'counts': {
                'galaxies': self.n_galaxies, 'galaxies_with_bh': self.n_galaxies_bh,
                'galaxies_with_any_event': self.n_gal_any_event,
                'galaxies_with_super_edd_event': self.n_gal_super,
                'events_all': self.n_events_all,
                'events_dropped_seed_only': self.n_events_seed,
                'events_at_eddington_cap': self.n_events_at_cap,
            },
            'groups': {k: a.summary() for k, a in self.groups.items()},
        }
        if vol_phys:
            s['number_density_per_mpc3'] = {
                k: self.groups[k].n_events / vol_phys for k in
                ('super_edd',) + tuple(f'lrd_full@{f}' for f in BHAR_FLOORS)
                + tuple(f'lrd_partial@{f}' for f in BHAR_FLOORS)}
        return s


# ----------------------------------------------------------------------------
# markdown rendering
# ----------------------------------------------------------------------------
def _n(x):
    return f'{x:,}'


def _v(x, nd=2):
    return '-' if x is None else f'{x:.{nd}f}'


def _pct(a, b):
    return '-' if not b else f'{100.0 * a / b:.1f}%'


def _qtable(g, keys):
    rows = ['| quantity | N | min | 16% | median | 84% | max | mean |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
    for k in keys:
        q = g['quantities'][k]
        if q.get('n', 0) == 0:
            rows.append(f'| {QUANTITY_LABELS[k]} | 0 | - | - | - | - | - | - |')
            continue
        rows.append(f"| {QUANTITY_LABELS[k]} | {_n(q['n'])} | {_v(q['min'])} | {_v(q['p16'])} "
                    f"| {_v(q['median'])} | {_v(q['p84'])} | {_v(q['max'])} | {_v(q['mean'])} |")
    return rows


def _group_block(g, title, level='###'):
    L = [f'{level} {title}', '']
    if g['n_events'] == 0:
        return L + ['_None._', '']
    L += [f"- events: **{_n(g['n_events'])}** in **{_n(g['n_galaxies'])}** galaxies", '']
    L += _qtable(g, ['log_mbh', 'log_lambda', 'log_mdot', 'log_mstar', 'log_mvir',
                     'log_mhost', 'log_fbh', 'log_mbh_final'])
    tot = g['n_events']
    ch = g['channel_counts']
    ho = g['host_counts']
    L += ['', '| growth channel | events | share | | host (final cat.) | events | share |',
          '|---|---:|---:|---|---|---:|---:|']
    chk, hok = list(ch), list(ho)
    for i in range(max(len(chk), len(hok))):
        a = (chk[i], _n(ch[chk[i]]), _pct(ch[chk[i]], tot)) if i < len(chk) else ('', '', '')
        b = (hok[i], _n(ho[hok[i]]), _pct(ho[hok[i]], tot)) if i < len(hok) else ('', '', '')
        L.append(f'| {a[0]} | {a[1]} | {a[2]} | | {b[0]} | {b[1]} | {b[2]} |')
    L += ['', '| redshift | events | share |', '|---|---:|---:|']
    for k, v in g['redshift_bin_counts'].items():
        L.append(f'| {k} | {_n(v)} | {_pct(v, tot)} |')
    if g['top_events']:
        L += ['', f'Most extreme events (highest Eddington ratio):', '',
              '| galaxy | z | log M_BH | lambda | Mdot [Msun/yr] | log M_star | log Mvir | channel | host |',
              '|---:|---:|---:|---:|---:|---:|---:|---|---|']
        for t in g['top_events']:
            L.append(f"| {t['galaxy_index']} | {t['z']:.2f} | {t['log_mbh']:.2f} | {t['lambda']:.3g} "
                     f"| {t['mdot_msun_yr']:.3g} | {_v(t['log_mstar'])} | {_v(t['log_mvir'])} "
                     f"| {t['channel']} | {t['host']} |")
    return L + ['']


def render_run_md(s, level='##'):
    sub = level + '#'
    G = s['groups']
    sup = G['super_edd']
    c = s['counts']
    L = [f"{level} {s['run']}", '']
    L += [f"- files read: {s['files_read']}"
          + (f" (**skipped {len(s['files_skipped'])}**)" if s['files_skipped'] else ''),
          f"- catalogue(s) used: {', '.join(s['catalogue_groups']) or '-'};  h = {s['hubble_h']};  "
          f"box = {s['box_size_mpc_h']} Mpc/h" + (f";  volume = {s['volume_mpc3']:.3e} Mpc^3"
                                                  if s.get('volume_mpc3') else ''),
          f"- galaxies: {_n(c['galaxies'])}  (with a BH: {_n(c['galaxies_with_bh'])})",
          f"- accretion events (Mdot > 0, seed-only dropped): "
          f"{_n(c['events_all'] - c['events_dropped_seed_only'])}"
          f"  (seed-only events dropped: {_n(c['events_dropped_seed_only'])})",
          f"- events pinned at the Eddington cap (|lambda-1| <= tol): {_n(c['events_at_eddington_cap'])}",
          f"- **super-Eddington events: {_n(sup['n_events'])}** "
          f"({_pct(sup['n_events'], c['events_all'] - c['events_dropped_seed_only'])} of events) "
          f"in {_n(c['galaxies_with_super_edd_event'])} galaxies "
          f"({_pct(c['galaxies_with_super_edd_event'], c['galaxies_with_bh'])} of BH-hosting galaxies)"]
    nd = s.get('number_density_per_mpc3')
    if nd:
        L.append('- number density of events per Mpc^3 (whole run): '
                 + ', '.join(f"{k}: {v:.3g}" for k, v in nd.items()))
    for f in s['files_skipped']:
        L.append(f"  - skipped `{os.path.basename(f['file'])}`: {f['reason']}")
    if s['runtime_parameters']:
        L.append('- relevant run parameters: '
                 + ', '.join(f'{k}={v}' for k, v in sorted(s['runtime_parameters'].items())))
    L.append('')
    L += _group_block(sup, 'Super-Eddington events (all)', sub)
    L += [f'{sub} Super-Eddington events by growth channel', '',
          '| channel | events | share | median log M_BH | median log lambda | median log M_star |',
          '|---|---:|---:|---:|---:|---:|']
    for ch in (0, 1, 2):
        g = G[f'super_edd/ch={ch}']
        q = g['quantities']
        L.append(f"| {ACC_NAMES[ch]} | {_n(g['n_events'])} | {_pct(g['n_events'], sup['n_events'])} "
                 f"| {_v(q['log_mbh'].get('median'))} | {_v(q['log_lambda'].get('median'))} "
                 f"| {_v(q['log_mstar'].get('median'))} |")
    L += ['', f'{sub} Super-Eddington events by redshift', '',
          '| redshift | events | share | median log M_BH | median log lambda | median log M_star | median log f_BH |',
          '|---|---:|---:|---:|---:|---:|---:|']
    for lo, hi in Z_BINS:
        g = G[f'super_edd/z={_zlabel(lo, hi)}']
        q = g['quantities']
        L.append(f"| {_zlabel(lo, hi)} | {_n(g['n_events'])} | {_pct(g['n_events'], sup['n_events'])} "
                 f"| {_v(q['log_mbh'].get('median'))} | {_v(q['log_lambda'].get('median'))} "
                 f"| {_v(q['log_mstar'].get('median'))} | {_v(q['log_fbh'].get('median'))} |")
    L += ['']
    for fl in BHAR_FLOORS:
        tag = ' (default floor)' if fl == BHAR_FLOORS[0] else ' (alternative floor)'
        L += _group_block(G[f'lrd_full@{fl}'],
                          f'LRD full: f_BH >= 3%, Mdot >= {fl} Msun/yr, lambda >= 1{tag}', sub)
        L += _group_block(G[f'lrd_partial@{fl}'],
                          f'LRD partial: f_BH < 3%, Mdot >= {fl} Msun/yr, lambda >= 1{tag}', sub)
    return L


NOTES = """\
## Definitions and caveats

- **Event**: one (galaxy, snapshot) cell of a galaxy's black-hole accretion history with
  `BHMaxaccretionRate > 0`. `Mdot` is `BHMaxaccretionRate`, `Mdot_Edd` is
  `BHEddingtonRateLimit` (plain Eddington rate, eta = 0.1), `M_BH` is `BHMassatAccretion`
  (BH mass at that episode) and the growth channel is `BHAccretionType`
  (0 Radio, 1 Merger, 2 Disk instability).
- **Super-Eddington**: `Mdot/Mdot_Edd > 1 + tol` (tol = 1e-3 by default), so events that are merely
  pinned at the Eddington cap in an Eddington-limited run are not counted.
- **LRD (full / partial)**: the two selections of `plotting/bh_lrd_analysis.py` (Chen & Mo 2026):
  `Mdot >= floor` and `Mdot >= Mdot_Edd`, split at `f_BH = M_BH/M_star = 3%`
  (full: >= 3%, partial: < 3%). Shown for floors 0.1 (default) and 0.05 Msun/yr.
  LRD selection uses `Mdot >= Mdot_Edd`, so it also includes events with lambda exactly 1.
- Events at the BH seed mass (< 1% grown) are dropped, as in the plotting scripts.
- **Host properties are from the final catalogue** (highest populated `Snap_N`, normally z = 0),
  not from the epoch of the event: they describe the galaxy the BH ended up in.
  "Central/Satellite/Orphan" is that galaxy's `Type` at the end of the run.
  `f_BH` divides the BH mass at the event by that final stellar mass (as in `bh_lrd_analysis.py`).
- Percentiles come from fine histograms (0.01-0.02 dex bins); min, max, mean and counts are exact.
- "Events" count every snapshot of an episode separately; one galaxy can contribute many events.
"""


def build_readme(out_dir):
    runs = []
    for p in sorted(glob.glob(os.path.join(out_dir, '*', 'super_eddington_stats.json'))):
        with open(p) as fh:
            runs.append(json.load(fh))
    L = ['# Super-Eddington accretion and LRD statistics', '',
         f'Generated by `plotting/bh_super_eddington_stats.py` for {len(runs)} runs.', '']
    L += NOTES.splitlines() + ['']
    L += ['## Comparison', '',
          '| run | files | super-Edd events | galaxies | median log M_BH | median log lambda | '
          'max lambda | merger | disk inst. | radio | central | satellite | LRD full | LRD partial |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in runs:
        G = s['groups']
        sup = G['super_edd']
        q = sup['quantities']
        d = f"{BHAR_FLOORS[0]}"
        ch, ho, tot = sup['channel_counts'], sup['host_counts'], sup['n_events']
        L.append(
            f"| [{s['run']}](#{_anchor(s['run'])}) | {s['files_read']} | {_n(tot)} | {_n(sup['n_galaxies'])} "
            f"| {_v(q['log_mbh'].get('median'))} | {_v(q['log_lambda'].get('median'))} "
            f"| {_maxlam(q)} "
            f"| {_pct(ch.get('Merger', 0), tot)} | {_pct(ch.get('Disk instability', 0), tot)} "
            f"| {_pct(ch.get('Radio', 0), tot)} | {_pct(ho.get('Central', 0), tot)} "
            f"| {_pct(ho.get('Satellite', 0), tot)} | {_n(G[f'lrd_full@{d}']['n_events'])} "
            f"| {_n(G[f'lrd_partial@{d}']['n_events'])} |")
    L += ['', '(LRD columns use the default 0.1 Msun/yr floor; shares are of super-Eddington events.)', '']
    for s in runs:
        L += render_run_md(s)
    path = os.path.join(out_dir, 'README.md')
    with open(path, 'w') as fh:
        fh.write('\n'.join(L) + '\n')
    print(f'wrote {path}')


def _maxlam(q):
    m = q['log_lambda'].get('max')
    return '-' if m is None else f'{10 ** m:.3g}'


def _anchor(name):
    return re.sub(r'[^a-z0-9_-]', '', name.lower().replace(' ', '-'))


# ----------------------------------------------------------------------------
def analyse_run(path, out_dir, keep_seeds, edd_tol, name=None):
    files = find_files(path)
    name = name or os.path.basename(os.path.normpath(path))
    if not files:
        print(f'[{name}] no model_*.hdf5 found in {path}', flush=True)
        return None
    print(f'[{name}] {len(files)} file(s)', flush=True)
    hdr = None
    for f in files:
        try:
            hdr = read_header(f)
            break
        except OSError:
            continue
    if hdr is None:
        print(f'[{name}] no readable file', flush=True)
        return None
    snap_z = hdr['redshifts'] if hdr['redshifts'] is not None else np.arange(64) * np.nan
    run = Run(name, hdr, np.asarray(snap_z, dtype=float), keep_seeds, edd_tol)
    t0 = time.time()
    for i, f in enumerate(files):
        run.process_file(f)
        run.volume_h3 += frac_volume(f) if f not in [s[0] for s in run.skipped] else 0.0
        print(f'[{name}] {i + 1}/{len(files)} {os.path.basename(f)}  '
              f'super-Edd so far: {run.groups["super_edd"].n_events:,}  ({time.time() - t0:.0f}s)',
              flush=True)
    if run.n_files == 0:
        print(f'[{name}] none of the files could be used', flush=True)
        return None
    s = run.summary()
    d = os.path.join(out_dir, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'super_eddington_stats.json'), 'w') as fh:
        json.dump(s, fh, indent=1)
    with open(os.path.join(d, 'summary.md'), 'w') as fh:
        fh.write('\n'.join(render_run_md(s, '#')) + '\n')
    print(f'[{name}] wrote {d}/summary.md and super_eddington_stats.json', flush=True)
    return s


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('runs', nargs='*',
                   help='Run directories (each holding model_*.hdf5), globs, or files.')
    p.add_argument('--root', help='Analyse every subdirectory of ROOT that holds model_*.hdf5.')
    p.add_argument('--only', nargs='*', default=None,
                   help='With --root: restrict to these run directory names (substring match).')
    p.add_argument('-o', '--out-dir', default='super_eddington_stats',
                   help='Output directory (default: ./super_eddington_stats).')
    p.add_argument('--combine', metavar='DIR', default=None,
                   help='Only rebuild DIR/README.md from the per-run JSON already written.')
    p.add_argument('--no-readme', action='store_true',
                   help='Do not rebuild README.md after analysing.')
    p.add_argument('--keep-seeds', action='store_true',
                   help='Keep events recorded while the BH is still at its seed mass.')
    p.add_argument('--edd-tol', type=float, default=1e-3,
                   help='lambda must exceed 1 + tol to count as super-Eddington (default 1e-3).')
    args = p.parse_args()

    if args.combine:
        build_readme(args.combine)
        return

    targets = list(args.runs)
    if args.root:
        for d in sorted(glob.glob(os.path.join(args.root, '*/'))):
            if not find_files(d):
                continue
            nm = os.path.basename(os.path.normpath(d))
            if args.only and not any(o in nm for o in args.only):
                continue
            targets.append(d)
    if not targets:
        p.error('give run directories or --root')

    os.makedirs(args.out_dir, exist_ok=True)
    done = 0
    for t in targets:
        try:
            if analyse_run(t, args.out_dir, args.keep_seeds, args.edd_tol) is not None:
                done += 1
        except Exception as e:                       # one bad run must not stop the rest
            print(f'[{os.path.basename(os.path.normpath(t))}] FAILED: {type(e).__name__}: {e}',
                  flush=True)
    if done and not args.no_readme:
        build_readme(args.out_dir)


if __name__ == '__main__':
    main()

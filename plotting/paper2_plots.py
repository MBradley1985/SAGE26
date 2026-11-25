#!/usr/bin/env python
"""
paper2_plots.py -- Paper II (FFB vs MBK25): Figures 1-8 and the numbers the text quotes.

Figures (same content and style as the paper_plots.py functions named):
  1  Selection thresholds in the M-z plane        plot_36_selection_thresholds_mz
  2  SMF grid, 5 < z < 11                          plot_19c_smf_ffb_grid_mbk25
  3  SFRD and SMD histories                        plot_14c_density_evolution_mbk25
  4  eps_SFE vs Mvir at z ~ 10, Mvir >= 20 particles  plot_10_sfe_ffb
  5  SFR and Z* histograms at z ~ 10               plot_11d_ffb_histograms_combined
  6  Tracked SFHs, FFB and MBK25 vs no-FFB         plot_12h_sfh_ffb_transitions_sfr
  7  m* at n = 1e-5 Mpc^-3 vs z                    plot_G_mstar_vs_z_ndensity
  8  Selected fraction vs z                        plot_23c_ffb_fraction_bk25

Built for the 500 Mpc/h box.  Every model file is read once, in parallel
(one task per file), keeping only the columns needed and only reductions:
histograms, sums, per-octant top-k lists, the selected haloes and the
candidates for halo matching.  A second, much smaller pass follows the tracked
haloes for Fig 6.  Two things differ from paper_plots.py only in how they are
computed, not in what they estimate:
  * SMF bands: the 100 bootstrap resamples are drawn from the binned counts
    (multinomial), which has the same distribution as resampling galaxies.
  * SFRD/SMD bands: the bootstrap of a sum of n values is normal to high
    accuracy at these n, so the 16-84% band is sum +- sqrt(n var).
The reductions are saved (paper2_reductions.pkl), so the figures can be
re-drawn anywhere with --replot without the model files.

Usage (from anywhere; paths are taken relative to the SAGE26 root):
    python plotting/paper2_plots.py                      # mini-Millennium runs in output/
    python plotting/paper2_plots.py --box big -j 32      # 500 Mpc/h runs in output/testmodels/
    python plotting/paper2_plots.py --replot --out DIR   # re-draw from DIR/paper2_reductions.pkl
    python plotting/paper2_plots.py --figs 2 3 --run NO=/path/to/run/

Runs that are missing are skipped (FFB is required).
"""

import os
import sys
import time
import glob
import pickle
import argparse

os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')   # Lustre/GPFS on clusters

import numpy as np
import h5py as h5

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ========================== RUNS ==========================

RUN_SETS = {
    'mini': {
        'FFB':    'output/millennium/',
        'MBK':    'output/millennium_mbk_smooth/',
        'NO':     'output/millennium_noffb/',
        'FFB100': 'output/millennium_ffb100/',
        'MBK100': 'output/millennium_mbk100/',
        'UCHUU':  'output/microuchuu/',
    },
    'big': {
        'FFB':    'output/testmodels/Millennium_FFB/',
        'MBK':    'output/testmodels/Millennium_FFB_MBK25_smooth/',
        'NO':     'output/testmodels/Millennium_noFFB/',
        'FFB100': 'output/testmodels/Millennium_FFB100/',
        'MBK100': 'output/testmodels/Millennium_MBK_FFB100/',
        # Section 4.2: SN reheating + ejection x 1/(1+eta), and no SN feedback, in FFB mode
        'FFBML2': 'output/testmodels/Millennium_FFB_reducedfb_both/',
        'FFBML3': 'output/testmodels/Millennium_FFB_fboff/',
        'UCHUU':  'output/testmodels/miniUchuu_FFB/',
    },
}

# ========================== ANALYSIS CHOICES (as paper_plots.py) ==========================

MIN_PARTICLES = 20
SMF_ZBINS = [(5.0, 6.0), (6.0, 7.0), (7.0, 9.0), (9.0, 11.0)]          # Fig 2
SMF_BIN = 0.2
SMF_RANGE = (0.0, 14.0)            # integer-aligned, so the bins match mass_function()
SMF_MIN_COUNT = 20                 # bins compared in the numbers need this many in both runs
N_BOOT = 100
DENS_Z = (4.8, 20.0)               # Fig 3
FIG1_Z = (4.0, 16.0)               # Fig 1
ND_TARGETS = [1e-6, 1e-5, 1e-4, 1e-3]  # Fig 7 (only 1e-5 drawn)
ND_PLOT = 1e-5
ND_MIN_COUNT = 10
ND_Z = (4.0, 30.0)
TRACK_N = 10                       # Fig 6: the 10 most massive ...
TRACK_SELECT_Z = 10.1              # ... selected FFB centrals at the snapshot nearest z = 10.1
TRACK_FIRST_SNAP = 8               # tracks start at z ~ 14
TRACK_Z_MIN = 3.0                  # and end at z = 3
FIG4_BIN = 0.15
DILUTE = 7500
SEED = 2222
SSFR_QUENCHED = -11.0
Z10_FLOOR_DEX = 1.5                # keep z~10 centrals down to this far below the lower threshold

_MASS_FIELDS = {'StellarMass', 'Mvir', 'MetalsStellarMass', 'BlackHoleMass'}


# ========================== I/O ==========================

def model_files(d):
    files = sorted(glob.glob(os.path.join(d, 'model_*.hdf5')))
    if not files and os.path.exists(os.path.join(d, 'model_0.hdf5')):
        files = [os.path.join(d, 'model_0.hdf5')]
    return files


def read_header(d):
    files = model_files(d)
    if not files:
        return None
    with h5.File(files[0], 'r') as f:
        sim, rt = f['Header/Simulation'].attrs, f['Header/Runtime'].attrs
        hdr = dict(
            h=float(sim['hubble_h']), box=float(sim['box_size']),
            om=float(sim['omega_matter']), ol=float(sim['omega_lambda']),
            unit_mass=float(rt['UnitMass_in_g']),
            unit_len=float(rt['UnitLength_in_cm']),
            unit_vel=float(rt['UnitVelocity_in_cm_per_s']),
            fb=float(rt.get('BaryonFrac', 0.17)),
            redshifts=np.array(f['Header/snapshot_redshifts'][:], dtype=float),
            mode=int(rt.get('EnhancedStarFormationOn', rt.get('FeedbackFreeModeOn', -1))),
            alpha=float(rt.get('EnhancedSFEfficiency', rt.get('FFBMaxEfficiency', np.nan))),
            regime_fixed=int(rt.get('RegimeDrawFixed', 0)),
            ffb_fixed=int(rt.get('FFBDrawFixed', 0)),
            particle_mass=float(sim.get('particle_mass', np.nan)),
        )
    hdr['vfrac'] = 0.0
    for fp in files:
        with h5.File(fp, 'r') as f:
            hdr['vfrac'] += float(f['Header/Runtime'].attrs['frac_volume_processed'])
    hdr['volume'] = (hdr['box'] / hdr['h']) ** 3 * hdr['vfrac']           # Mpc^3
    hdr['mass_convert'] = hdr['unit_mass'] / 1.989e33 / hdr['h']           # code -> Msun
    hdr['mp_msun'] = hdr['particle_mass'] * hdr['mass_convert']
    hdr['G_code'] = 6.674e-8 * hdr['unit_mass'] / (hdr['unit_len'] * hdr['unit_vel'] ** 2)
    hdr['files'] = files
    return hdr


class _Snap:
    """Lazy column reader for one snapshot group of one file, rows cut to Len >= min_len."""

    def __init__(self, grp, mass_convert, min_len=MIN_PARTICLES):
        self.g, self.mc = grp, mass_convert
        self.keep = grp['Len'][()] >= min_len if min_len > 0 else None
        self.cache = {}

    def __call__(self, name, raw=False):
        key = (name, raw)
        if key not in self.cache:
            a = self.g[name][()]                       # whole column: fast and contiguous
            if self.keep is not None:
                a = a[self.keep]
            if name in _MASS_FIELDS and not raw:
                a = a.astype(np.float64) * self.mc
            self.cache[key] = a
        return self.cache[key]

    def __contains__(self, name):
        return name in self.g


# ========================== PASS 1: ONE TASK PER FILE ==========================

def _top_k(a, k):
    return a if a.size <= k else np.partition(a, a.size - k)[-k:]


def reduce_file(task):
    """All per-snapshot reductions of one model file.  Runs in a worker process."""
    path, plan = task
    out = {}
    with h5.File(path, 'r') as f:
        for s in plan['snaps']:
            key = f'Snap_{s}'
            if key not in f:
                continue
            c = _Snap(f[key], plan['mass_convert'])
            if not c.keep.any():
                continue
            r = {}
            ms = c('StellarMass')
            if s in plan['smf']:
                lm = np.log10(ms[ms > 0])
                r['smf'] = np.histogram(lm, bins=int(round((SMF_RANGE[1] - SMF_RANGE[0]) / SMF_BIN)),
                                        range=SMF_RANGE)[0]
            if s in plan['dens']:
                sfr = c('SfrDisk').astype(np.float64) + c('SfrBulge')
                r['dens'] = np.array([ms.size, sfr.sum(), (sfr ** 2).sum(),
                                      ms.sum(), (ms ** 2).sum()])
            if s in plan['nd']:
                box = plan['box']
                pos = ms > 0
                oid = ((c('Posx') % box >= box / 2).astype(np.int8)
                       + 2 * (c('Posy') % box >= box / 2).astype(np.int8)
                       + 4 * (c('Posz') % box >= box / 2).astype(np.int8))
                r['nd'] = ([_top_k(ms[pos & (oid == o)], plan['nd_k']) for o in range(8)],
                           int(pos.sum()))
            if plan['ffb'] and (s in plan['frac'] or s in plan['sel'] or s in plan['fig1']):
                sel = (c('Type') == 0) & (c('FFBRegime') == 1)
                if s in plan['frac']:
                    r['frac'] = np.array([int((c('Type') == 0).sum()), int(sel.sum())])
                if s in plan['sel']:
                    r['sel'] = c('GalaxyIndex')[sel].astype(np.int64)
                if s in plan['fig1']:
                    w = sel & (c('Mvir') > 0)
                    conc = np.empty(0)
                    if plan['conc'] and w.any():
                        g_vir = plan['G_code'] * c('Mvir', raw=True)[w] / c('Rvir')[w].astype(np.float64) ** 2
                        conc = conc_from_ratio(c('g_max')[w] / g_vir)
                    r['fig1'] = (c('Mvir')[w], conc)
            if s == plan['z10']:
                w = (c('Type') == 0) & (c('Mvir') >= plan['z10_floor'])
                if plan['ffb']:
                    w |= (c('Type') == 0) & (c('FFBRegime') == 1)
                r['z10'] = {p: c(p)[w] for p in ('Mvir', 'StellarMass', 'MetalsStellarMass', 'Posx',
                                                 'Posy', 'Posz', 'Type', 'GalaxyIndex')}
                r['z10']['SFR'] = (c('SfrDisk') + c('SfrBulge'))[w]
                r['z10']['FFBRegime'] = (c('FFBRegime')[w] if 'FFBRegime' in c
                                         else np.zeros(int(w.sum()), dtype=np.int32))
            if r:
                out[s] = r
    return plan['run'], out


_C_GRID = np.logspace(0.0, np.log10(500.0), 4000)
_H_GRID = _C_GRID ** 2 / (2.0 * (np.log1p(_C_GRID) - _C_GRID / (1.0 + _C_GRID)))


def conc_from_ratio(ratio):
    """
    Concentration from g_max / g_vir = c^2 / (2 mu(c)), monotonic in c for c >= 1
    (the model floors c at 1).  Tabulated inversion; paper_plots solves it per point.
    """
    return np.where(ratio > _H_GRID[0], np.interp(ratio, _H_GRID, _C_GRID), 1.0)


def merge_file_results(results):
    """{run: {snap: merged reductions}} from the per-file outputs."""
    R = {}
    for run, out in results:
        Rr = R.setdefault(run, {})
        for s, r in out.items():
            m = Rr.setdefault(s, {})
            for k, v in r.items():
                if k in ('smf', 'dens', 'frac'):
                    m[k] = m[k] + v if k in m else v
                elif k == 'nd':
                    tops, n = v
                    if k in m:
                        kk = max(max(a.size, b.size) for a, b in zip(m[k][0], tops))
                        tops = [_top_k(np.concatenate([a, b]), kk) for a, b in zip(m[k][0], tops)]
                        n += m[k][1]
                    m[k] = (tops, n)
                elif k in ('sel',):
                    m.setdefault(k, []).append(v)
                elif k == 'fig1':
                    m.setdefault(k, []).append(v)
                elif k == 'z10':
                    m.setdefault(k, []).append(v)
    for Rr in R.values():
        for m in Rr.values():
            if 'sel' in m:
                m['sel'] = np.concatenate(m['sel'])
            if 'fig1' in m:
                m['fig1'] = (np.concatenate([a for a, _ in m['fig1']]),
                             np.concatenate([b for _, b in m['fig1']]))
            if 'z10' in m:
                m['z10'] = {p: np.concatenate([d[p] for d in m['z10']]) for p in m['z10'][0]}
    return R


# ========================== PASS 2: TRACKED HALOES (FIG 6) AND SURVIVORS ==========================

_TRACK_FIELDS = ('GalaxyIndex', 'Type', 'Mvir', 'Posx', 'Posy', 'Posz', 'StellarMass',
                 'BlackHoleMass', 'FFBRegime')


def track_file(task):
    """
    kind 'gid':  rows (Len >= 20) whose GalaxyIndex is in gids[s], per snapshot.
    kind 'near': rows (Len >= 20, Type <= 1) whose Mvir is within 1e-3 of targets[s].
    kind 'final': rows (any Len) of the final snapshot whose GalaxyIndex is in gids.
    """
    path, run, kind, spec, mass_convert = task
    out = {}
    with h5.File(path, 'r') as f:
        for s, want in spec.items():
            key = f'Snap_{s}'
            if key not in f or len(want) == 0:
                continue
            c = _Snap(f[key], mass_convert, min_len=0 if kind == 'final' else MIN_PARTICLES)
            if c.keep is not None and not c.keep.any():
                continue
            if kind in ('gid', 'final'):
                w = np.isin(c('GalaxyIndex'), want)
            else:
                lm_req = np.sort(np.log10(want))
                tol = 1.01 * np.log10(1.0 + 1e-3)
                lm = np.log10(np.maximum(c('Mvir'), 1e-30))
                i = np.searchsorted(lm_req, lm)
                near = np.minimum(np.abs(lm - lm_req[np.clip(i - 1, 0, lm_req.size - 1)]),
                                  np.abs(lm - lm_req[np.clip(i, 0, lm_req.size - 1)])) <= tol
                w = near & (c('Type') <= 1)
            if not w.any():
                continue
            fields = (('GalaxyIndex', 'Type', 'StellarMass') if kind == 'final'
                      else [p for p in _TRACK_FIELDS if p in c])
            d = {p: c(p)[w] for p in fields}
            d['SFR'] = (c('SfrDisk') + c('SfrBulge'))[w]
            out[s] = d
    return run, kind, out


def _cat(dicts):
    if not dicts:
        return None
    return {p: np.concatenate([d[p] for d in dicts]) for p in dicts[0]}


def merge_track_results(results):
    T = {}
    for run, kind, out in results:
        for s, d in out.items():
            T.setdefault((run, kind), {}).setdefault(s, []).append(d)
    return {k: {s: _cat(v) for s, v in per.items()} for k, per in T.items()}


# ========================== PIPELINE ==========================

def _pool_map(fn, tasks, workers):
    if workers <= 1 or len(tasks) <= 1:
        return [fn(t) for t in tasks]
    import multiprocessing as mp
    ctx = mp.get_context('fork' if sys.platform.startswith('linux') else 'spawn')
    with ctx.Pool(min(workers, len(tasks))) as pool:
        return pool.map(fn, tasks, chunksize=1)


def snap_nearest(Z, z):
    return int(np.argmin(np.abs(np.asarray(Z) - z)))


def snap_in_bin(Z, lo, hi):
    Z = np.asarray(Z)
    inb = np.where((Z >= lo) & (Z <= hi))[0]
    return int(inb[np.argmin(np.abs(Z[inb] - 0.5 * (lo + hi)))]) if inb.size else None


def compute(runs, hdrs, workers, pp, track_z=TRACK_SELECT_Z):
    """Both passes.  Returns the reductions dictionary that every figure and number uses."""
    t0 = time.time()
    ref = hdrs['FFB']
    Z = ref['redshifts']
    nsnap = len(Z)
    allsnaps = set(range(nsnap))
    snap_z10 = pp._snap_for_z(list(Z), 10.0)          # as SNAP_Z10 in paper_plots
    track_sel = snap_nearest(Z, track_z)
    lthr = min(np.log10(pp.ffb_threshold_mass_msun(Z[snap_z10])),
               np.log10(pp.mbk25_threshold_median_c(Z[snap_z10])[0]))
    z10_floor = 10 ** (lthr - Z10_FLOOR_DEX)

    smf_snaps = {s for s in (snap_in_bin(Z, lo, hi) for lo, hi in SMF_ZBINS) if s is not None}
    dens = {s for s in allsnaps if DENS_Z[0] <= Z[s] <= DENS_Z[1]}
    nd = {s for s in allsnaps if ND_Z[0] <= Z[s] <= ND_Z[1]}
    below = [s for s in allsnaps if Z[s] < ND_Z[0]]
    if below:
        nd.add(max(below, key=lambda s: Z[s]))
    fig1 = {s for s in allsnaps if FIG1_Z[0] <= Z[s] <= FIG1_Z[1]}
    z10s = {snap_z10, track_sel}

    roles = {
        'FFB':    dict(smf=smf_snaps, dens=dens, nd=nd, frac=allsnaps, sel=allsnaps, fig1=fig1, ffb=True),
        'MBK':    dict(smf=smf_snaps, dens=dens, nd=nd, frac=allsnaps, sel=allsnaps, fig1=fig1, ffb=True,
                       conc=True),
        'NO':     dict(smf=smf_snaps, dens=dens, nd=nd),
        'FFB100': dict(smf=smf_snaps, dens=dens),
        'MBK100': dict(smf=smf_snaps, dens=dens),
        'FFBML2': dict(ffb=True),
        'FFBML3': dict(ffb=True),
    }

    tasks = []
    for run, rl in roles.items():
        h = hdrs.get(run)
        if h is None:
            continue
        plan = dict(run=run, mass_convert=h['mass_convert'], box=h['box'], G_code=h['G_code'],
                    smf=rl.get('smf', set()), dens=rl.get('dens', set()), nd=rl.get('nd', set()),
                    frac=rl.get('frac', set()), sel=rl.get('sel', set()), fig1=rl.get('fig1', set()),
                    ffb=rl.get('ffb', False), conc=rl.get('conc', False),
                    z10=snap_z10 if run not in ('FFB100', 'MBK100') else -1, z10_floor=z10_floor,
                    nd_k=int(np.ceil(max(ND_TARGETS) * h['volume'])) + 2)
        # the tracked sample is chosen at track_sel in FFB, MBK and NO
        plan['snaps'] = sorted(set().union(plan['smf'], plan['dens'], plan['nd'], plan['frac'],
                                           plan['sel'], plan['fig1'],
                                           {plan['z10']} - {-1}))
        tasks += [(fp, plan) for fp in h['files']]
        if run in ('FFB', 'MBK', 'NO') and track_sel != snap_z10:
            p2 = dict(plan, snaps=[track_sel], smf=set(), dens=set(), nd=set(), frac=set(), sel=set(),
                      fig1=set(), z10=track_sel, run=run + '@track')
            tasks += [(fp, p2) for fp in h['files']]
    hu = hdrs.get('UCHUU')
    if hu is not None:
        Zu = hu['redshifts']
        su = {s for s in (snap_in_bin(Zu, lo, hi) for lo, hi in SMF_ZBINS) if s is not None}
        plan = dict(run='UCHUU', mass_convert=hu['mass_convert'], box=hu['box'], G_code=hu['G_code'],
                    smf=su, dens=set(), nd=set(), frac=set(), sel=set(), fig1=set(), ffb=False,
                    conc=False, z10=-1, z10_floor=0.0, nd_k=1, snaps=sorted(su))
        tasks += [(fp, plan) for fp in hu['files']]

    print(f'Pass 1: {len(tasks)} file tasks on {workers} worker(s)', flush=True)
    R = merge_file_results(_pool_map(reduce_file, tasks, workers))
    for run in list(R):
        if run.endswith('@track'):
            base = run.split('@')[0]
            extra = R.pop(run).get(track_sel)
            if extra and base in R:
                R[base].setdefault(track_sel, {}).update(extra)
    for run, Rr in R.items():
        for s, m in Rr.items():
            if 'nd' in m:       # per-octant lists -> {n: ((log m*, sigma) or None, reason)}; keeps memory small
                m['nd'] = {n: mstar_at_n(m['nd'][0], m['nd'][1], hdrs[run]['volume'], n) for n in ND_TARGETS}
    print(f'  done in {time.time() - t0:.1f} s', flush=True)

    # --- tracked sample (Fig 6): most massive selected FFB centrals with a same-halo central in MBK and NO
    track = None
    if all(k in R and track_sel in R[k] and 'z10' in R[k][track_sel] for k in ('FFB', 'MBK', 'NO')):
        d0 = R['FFB'][track_sel]['z10']
        cand = np.where((d0['FFBRegime'] == 1) & (d0['StellarMass'] > 0))[0]
        n_pool = cand.size
        cand = cand[np.argsort(d0['StellarMass'][cand])[::-1]][:2 * TRACK_N]
        for k in ('MBK', 'NO'):
            cand = cand[pp.same_halo_rows(d0, cand, R[k][track_sel]['z10'], box_size=ref['box']) >= 0]
        cand = cand[np.argsort(d0['StellarMass'][cand])[::-1]][:TRACK_N]
        track = dict(gids=d0['GalaxyIndex'][cand].astype(np.int64),
                     mstar0=np.log10(d0['StellarMass'][cand]), n_pool=n_pool, select_snap=track_sel,
                     snaps=[s for s in range(TRACK_FIRST_SNAP, nsnap) if Z[s] >= TRACK_Z_MIN])

    # --- gids selected below z = 4, for the survivors at the final snapshot (Section 5.3)
    s_last = int(np.argmin(Z))
    low_gids = {}
    for k in ('FFB', 'MBK'):
        if k in R:
            g = [R[k][s]['sel'] for s in R[k] if Z[s] < 4.0 and 'sel' in R[k][s]]
            low_gids[k] = np.unique(np.concatenate(g)) if g else np.empty(0, dtype=np.int64)

    t1 = time.time()
    tasks = []
    if track is not None:
        tasks += [(fp, 'FFB', 'gid', {s: track['gids'] for s in track['snaps']}, ref['mass_convert'])
                  for fp in ref['files']]
    for k, g in low_gids.items():
        if g.size:
            tasks += [(fp, k, 'final', {s_last: g}, hdrs[k]['mass_convert']) for fp in hdrs[k]['files']]
    print(f'Pass 2a: {len(tasks)} file tasks (tracked haloes in FFB, survivors)', flush=True)
    T = merge_track_results(_pool_map(track_file, tasks, workers))

    if track is not None:
        F = T.get(('FFB', 'gid'), {})
        targets = {s: F[s]['Mvir'][F[s]['Type'] <= 1] for s in track['snaps'] if s in F}
        tasks = [(fp, k, 'near', targets, hdrs[k]['mass_convert'])
                 for k in ('MBK', 'NO') for fp in hdrs[k]['files']]
        print(f'Pass 2b: {len(tasks)} file tasks (same haloes in MBK25 and no-FFB)', flush=True)
        T.update(merge_track_results(_pool_map(track_file, tasks, workers)))
        track['vals'] = assemble_tracks(track, T, ref['box'], pp)
    print(f'  done in {time.time() - t1:.1f} s', flush=True)

    prune_z10(R, snap_z10, track_sel)
    survivors = {k: T.get((k, 'final'), {}).get(s_last) for k in low_gids}
    return dict(R=R, track=track, survivors=survivors, low_gids=low_gids,
                snap_z10=snap_z10, s_last=s_last,
                hdrs={k: {kk: vv for kk, vv in h.items() if kk != 'files'} for k, h in hdrs.items() if h},
                runs=runs)


def prune_z10(R, *snaps):
    """
    Keep only the z~10 rows that can take part in halo matching: every selected central,
    and any central whose Mvir is within 1e-3 of a selected one (a same-halo partner must be).
    """
    tol = 1.01 * np.log10(1.0 + 1e-3)
    for s in set(snaps):
        cats = {k: R[k][s]['z10'] for k in R if s in R[k] and 'z10' in R[k][s]}
        sel = [np.log10(d['Mvir'][d['FFBRegime'] == 1]) for k, d in cats.items() if k in ('FFB', 'MBK')]
        lm_req = np.sort(np.concatenate(sel)) if sel else np.empty(0)
        for k, d in cats.items():
            keep = d['FFBRegime'] == 1
            if lm_req.size:
                lm = np.log10(np.maximum(d['Mvir'], 1e-30))
                i = np.searchsorted(lm_req, lm)
                keep |= np.minimum(np.abs(lm - lm_req[np.clip(i - 1, 0, lm_req.size - 1)]),
                                   np.abs(lm - lm_req[np.clip(i, 0, lm_req.size - 1)])) <= tol
            R[k][s]['z10'] = {p: a[keep] for p, a in d.items()}


def assemble_tracks(track, T, box, pp):
    """(n_gal, n_snap) arrays of sfr, mvir, regime, BH mass, m* per run; NaN = absent."""
    gids, snaps = track['gids'], track['snaps']
    fields = ('sfr', 'mvir', 'regime', 'bh', 'mstar')
    vals = {run: {f: np.full((gids.size, len(snaps)), np.nan) for f in fields}
            for run in ('FFB', 'MBK25', 'noFFB')}

    def store(run, j, d, rows, which):
        vals[run]['sfr'][which, j] = d['SFR'][rows]
        vals[run]['mvir'][which, j] = d['Mvir'][rows]
        vals[run]['regime'][which, j] = d['FFBRegime'][rows] if 'FFBRegime' in d else 0
        vals[run]['bh'][which, j] = d['BlackHoleMass'][rows]
        vals[run]['mstar'][which, j] = d['StellarMass'][rows]

    F = T.get(('FFB', 'gid'), {})
    for j, s in enumerate(snaps):
        d = F.get(s)
        if d is None:
            continue
        u, first = np.unique(d['GalaxyIndex'], return_index=True)
        pos = np.clip(np.searchsorted(u, gids), 0, max(u.size - 1, 0))
        present = u[pos] == gids
        rows = first[pos[present]]
        store('FFB', j, d, rows, present)
        has_halo = present.copy()
        has_halo[present] = d['Type'][rows] <= 1
        idx = np.where(has_halo)[0]
        if idx.size == 0:
            continue
        ref_rows = first[pos[idx]]
        for run, key in (('MBK25', 'MBK'), ('noFFB', 'NO')):
            other = T.get((key, 'near'), {}).get(s)
            if other is None:
                continue
            m = pp.same_halo_rows(d, ref_rows, other, box_size=box, types=(0, 1))
            ok = m >= 0
            store(run, j, other, m[ok], idx[ok])
    return vals


# ========================== SHARED DERIVED QUANTITIES ==========================

def smf_curve(counts, volume, rng):
    """x, log phi, 16th, 84th percentiles (bootstrap from counts), as mass_function_bootstrap()."""
    edges = np.linspace(SMF_RANGE[0], SMF_RANGE[1], counts.size + 1)
    x = edges[:-1] + 0.5 * SMF_BIN
    n = int(counts.sum())
    with np.errstate(divide='ignore'):
        phi = np.log10(counts / volume / SMF_BIN)
    phi[~np.isfinite(phi)] = np.nan
    if n == 0:
        return x, phi, phi, phi
    boots = rng.multinomial(n, counts / n, size=N_BOOT).astype(float)
    with np.errstate(divide='ignore'):
        bp = np.log10(boots / volume / SMF_BIN)
    bp[~np.isfinite(bp)] = np.nan
    with np.errstate(invalid='ignore'):
        lo = np.nanpercentile(bp, 16, axis=0)
        hi = np.nanpercentile(bp, 84, axis=0)
    return x, phi, lo, hi


def density_series(Rr, Z, volume, col):
    """z, log density, lo, hi for SFRD (col 1) or SMD (col 3) from (n, S, S2) sums."""
    zs, med, lo, hi = [], [], [], []
    for s in sorted(Rr):
        if 'dens' not in Rr[s]:
            continue
        n, S, S2 = Rr[s]['dens'][0], Rr[s]['dens'][col], Rr[s]['dens'][col + 1]
        sig = np.sqrt(max(S2 - S * S / n, 0.0)) if n > 0 else 0.0     # sqrt(n var)
        lg = lambda v: np.log10(v / volume) if v > 0 else np.nan
        zs.append(Z[s]); med.append(lg(S)); lo.append(lg(S - sig)); hi.append(lg(S + sig))
    o = np.argsort(zs)
    return tuple(np.array(a)[o] for a in (zs, med, lo, hi))


def mstar_at_n(tops, n_pos, volume, n):
    """(log m*, sigma) at cumulative density n from per-octant top-k lists, or None + reason."""
    def at(masses, v, nn):
        k = nn * v
        srt = np.sort(masses)[::-1]
        if k < 1 or k > srt.size:
            return np.nan
        return float(np.interp(np.log10(k), np.log10(np.arange(1, srt.size + 1)), np.log10(srt)))
    k = n * volume
    if k < ND_MIN_COUNT:
        return None, f'needs {k:.0f} galaxies, below the {ND_MIN_COUNT}-object floor'
    if k > n_pos:
        return None, f'needs rank {k:.0f} but only {n_pos} galaxies survive the {MIN_PARTICLES}-particle cut'
    centre = at(np.concatenate(tops), volume, n)
    sub = np.array([at(np.concatenate([tops[j] for j in range(8) if j != o]), volume * 7 / 8, n)
                    for o in range(8)])
    sig = np.sqrt(7 / 8 * np.sum((sub - sub.mean()) ** 2)) if np.all(np.isfinite(sub)) else np.nan
    return (centre, sig), ''


def nd_curves(D):
    """{(run, n): (z, logm, lo, hi)} and the truncation reasons, as plot_G."""
    Z = D['hdrs']['FFB']['redshifts']
    curves, why = {}, {}
    for run in ('MBK', 'FFB', 'NO'):
        Rr = D['R'].get(run)
        if Rr is None:
            continue
        per = {n: [] for n in ND_TARGETS}
        for s in sorted(Rr):
            if 'nd' not in Rr[s]:
                continue
            for n in ND_TARGETS:
                v, reason = Rr[s]['nd'][n]
                if v is None:
                    why[(run, s, n)] = reason
                else:
                    per[n].append((Z[s], v[0], v[0] - v[1], v[0] + v[1]))
        for n, rows in per.items():
            if len(rows) >= 3:
                curves[(run, n)] = tuple(np.array(c) for c in zip(*rows))
    return curves, why


def z10_samples(D, pp):
    """Fig 4/5 samples at z ~ 10: FFB and MBK25 selected centrals, and the same haloes in NO."""
    s = D['snap_z10']
    get = lambda k: D['R'].get(k, {}).get(s, {}).get('z10')
    F, M, NO = get('FFB'), get('MBK'), get('NO')
    box = D['hdrs']['FFB']['box']
    out = {}
    for name, d in (('FFB', F), ('MBK25', M), ('FFBML2', get('FFBML2')), ('FFBML3', get('FFBML3'))):
        if d is not None:
            out[name] = (d, np.where((d['FFBRegime'] == 1) & (d['Type'] == 0))[0])
    if NO is not None:
        rows = []
        for d in (F, M):
            if d is None:
                continue
            r = np.where((d['FFBRegime'] == 1) & (d['Type'] == 0))[0]
            m = pp.same_halo_rows(d, r, NO, box_size=box)
            if np.any(m < 0):
                print(f'  z~10: {int(np.sum(m < 0))} of {r.size} selected centrals have no same-halo '
                      f'central in the no-FFB run')
            rows.append(m[m >= 0])
        out['NO'] = (NO, np.unique(np.concatenate(rows)) if rows else np.empty(0, dtype=np.int64))
    return out


def episodes(Rr, snaps_order):
    """Lengths (in snapshots) and start snapshots of every selection episode (consecutive selected snaps)."""
    pairs = [(Rr[s]['sel'], np.full(Rr[s]['sel'].size, i)) for i, s in enumerate(snaps_order)
             if s in Rr and 'sel' in Rr[s] and Rr[s]['sel'].size]
    if not pairs:
        return np.empty(0, int), np.empty(0, int)
    g = np.concatenate([p[0] for p in pairs])
    t = np.concatenate([p[1] for p in pairs])
    o = np.lexsort((t, g))
    g, t = g[o], t[o]
    brk = np.r_[True, (g[1:] != g[:-1]) | (t[1:] != t[:-1] + 1)]
    starts = np.where(brk)[0]
    lengths = np.diff(np.r_[starts, g.size])
    return lengths, np.asarray(snaps_order)[t[starts]]


# ========================== FIGURES ==========================

def _style(pp):
    import matplotlib
    matplotlib.use('Agg')
    pp.setup_style()
    import matplotlib.pyplot as plt
    return plt


def fig1(D, pp, out):
    plt = _style(pp)
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    Z = D['hdrs']['FFB']['redshifts']
    rng = np.random.default_rng(SEED)

    def collect(run):
        zs, mv, cc = [], [], []
        for s, r in sorted(D['R'].get(run, {}).items()):
            if 'fig1' in r and r['fig1'][0].size:
                zs.append(np.full(r['fig1'][0].size, Z[s])); mv.append(r['fig1'][0]); cc.append(r['fig1'][1])
        if not mv:
            return np.empty(0), np.empty(0), np.empty(0)
        return np.concatenate(zs), np.concatenate(mv), np.concatenate(cc)

    z_m, m_m, c_m = collect('MBK')
    z_f, m_f, _ = collect('FFB')
    print(f'  Fig 1: MBK25 galaxies {m_m.size}, FFB galaxies {m_f.size}')
    if c_m.size:
        print('  MBK25 selection c (min/5/50/95 pct): %.2f / %.2f / %.2f / %.2f'
              % ((c_m.min(),) + tuple(np.percentile(c_m, [5, 50, 95]))))
        c_lo, c_hi = c_m.min(), np.percentile(c_m, 95)
    if m_m.size > DILUTE:
        i = rng.choice(m_m.size, DILUTE, replace=False); z_m, m_m, c_m = z_m[i], m_m[i], c_m[i]
    if m_f.size > DILUTE:
        i = rng.choice(m_f.size, DILUTE, replace=False); z_f, m_f = z_f[i], m_f[i]

    z_grid = np.linspace(4.0, 16.0, 200)
    fig, (ax_f, ax_m) = plt.subplots(1, 2, figsize=(10.5, 5.0), sharey=True, constrained_layout=True)
    C_MED = 3.2
    cmap = ListedColormap(plt.get_cmap('plasma')(np.linspace(0.0, 0.85, 256)))
    if m_m.size:
        sc = ax_m.scatter(z_m, np.log10(m_m), s=6, c=c_m, cmap=cmap, vmin=c_lo, vmax=c_hi,
                          edgecolors='none', rasterized=True, zorder=2)
        cb = fig.colorbar(sc, ax=ax_m, location='right', pad=0.02, aspect=30, extend='max')
        cb.set_label('Halo concentration', fontsize='small')
    ax_m.plot(z_grid, np.log10(pp.mbk25_threshold_mass_msun(z_grid, C_MED)), lw=2.2, color='mediumpurple',
              zorder=3)
    ax_m.legend(handles=[Line2D([], [], marker='o', ls='none', color=cmap(0.5), markersize=4,
                                label='MBK25 galaxies'),
                         Line2D([], [], color='mediumpurple', lw=2.2,
                                label=fr'$M_{{\rm MBK25}}(z)$, $c={C_MED:g}$')],
                loc='upper right', frameon=False, fontsize=8)
    if m_f.size:
        ax_f.scatter(z_f, np.log10(m_f), s=6, alpha=0.45, color='0.45', edgecolors='none',
                     rasterized=True, zorder=1)
    ax_f.plot(z_grid, np.log10([pp.ffb_threshold_mass_msun(z) for z in z_grid]), lw=2.2,
              color='firebrick', zorder=3)
    ax_f.legend(handles=[Line2D([], [], marker='o', ls='none', color='0.35', markersize=4,
                                label='FFB galaxies'),
                         Line2D([], [], color='firebrick', lw=2.2, label=r'$M_{\rm FFB}(z)$')],
                loc='upper right', frameon=False, fontsize=8)
    ax_f.set_ylabel(r'$\log_{10}\ M_{\rm vir}\ [M_\odot]$')
    for ax in (ax_f, ax_m):
        ax.set_xlabel(r'Redshift $z$')
        ax.set_xlim(z_grid.min(), z_grid.max())
        ax.set_ylim(9.5, 13.5)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.xaxis.set_minor_locator(plt.NullLocator())
    pp.save_figure(fig, os.path.join(out, 'Selection_Thresholds_Mz.pdf'))


_SMF_MODELS = [  # (run, label, colour, ls, lw)
    ('NO', r'no-FFB/MBK25 model', 'firebrick', '-', 3.0),
    ('FFB', r'$\alpha_{\rm FFB}=0.2$', 'black', '-', 3.5),
    ('MBK', r'$\alpha_{\rm MBK25}=0.2$', 'mediumpurple', '-', 3.0),
    ('FFB100', r'$\alpha_{\rm FFB}=1.0$', 'steelblue', '--', 3.0),
    ('MBK100', r'$\alpha_{\rm MBK25}=1.0$', 'magenta', '--', 3.0),
]


def fig2(D, pp, out):
    plt = _style(pp)
    rng = np.random.default_rng(SEED)
    Z = D['hdrs']['FFB']['redshifts']
    obs = pp._load_smf_grid_observations()
    used = set()
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharex=True, sharey=True)
    fig.set_tight_layout(False)
    axf = axes.flatten()
    for i, (lo, hi) in enumerate(SMF_ZBINS):
        ax = axf[i]
        s = snap_in_bin(Z, lo, hi)
        for run, lab, col, ls, lw in _SMF_MODELS:
            r = D['R'].get(run, {}).get(s)
            if r is None or 'smf' not in r:
                continue
            x, phi, plo, phi_hi = smf_curve(r['smf'], D['hdrs'][run]['volume'], rng)
            v = np.isfinite(phi)
            ax.plot(x[v], phi[v], lw=lw, color=col, ls=ls, label=lab if i == 0 else None)
            b = np.isfinite(plo) & np.isfinite(phi_hi)
            if b.any():
                ax.fill_between(x[b], plo[b], phi_hi[b], color=col, alpha=0.2, linewidth=0)
        for od in obs:
            zo = od['z']
            inb = (lo <= zo <= hi) if i == len(SMF_ZBINS) - 1 else (lo <= zo < hi)
            if not inb:
                continue
            lbl = od['label'] if od['label'] not in used else None
            used.add(od['label'])
            yerr = [od['err_lo'], od['err_hi']] if od['err_lo'] is not None and od['err_hi'] is not None else None
            ax.errorbar(od['log_mass'], od['log_phi'], yerr=yerr, fmt=od['marker'], color='grey', ms=od['ms'],
                        markeredgecolor='k', markeredgewidth=0.8, markerfacecolor='gray', alpha=0.6, lw=1.0,
                        label=lbl, zorder=1)
        ax.text(0.95, 0.95, rf'${lo:.0f} < z < {hi:.0f}$', transform=ax.transAxes, ha='right', va='top')
    axf[0].set_xlim(9, 12.3)
    axf[0].set_ylim(-6, -1.5)
    for i, ax in enumerate(axf):
        row, col = divmod(i, 2)
        if col == 0:
            ax.set_ylabel(r'$\log_{10}\ \phi\ [\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1}]$')
        if row == 1:
            ax.set_xlabel(r'$\log_{10}\ m_{\mathrm{*}}\ [M_{\odot}]$')
        ax.tick_params(axis='both', which='both', direction='in', top=True, bottom=True, left=True, right=True)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1.0)); ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2)); ax.yaxis.set_minor_locator(plt.MultipleLocator(0.2))
    handles, labels = [], []
    for ax in axf:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h); labels.append(l)
    model_labels = {m[1] for m in _SMF_MODELS}
    sim = [(h, l) for h, l in zip(handles, labels) if l in model_labels]
    obs_hl = [(h, l) for h, l in zip(handles, labels) if l not in model_labels]
    if sim:
        leg = axf[0].legend(*zip(*sim), loc='lower left', frameon=False, title='SAGE26')
        leg.get_title().set_fontweight('bold')
        axf[0].add_artist(leg)
    if obs_hl:
        axf[-1].legend(*zip(*obs_hl), loc='upper right', frameon=False, ncol=2, bbox_to_anchor=(1.0, 0.92))
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.001, wspace=0.001)
    pp.save_figure(fig, os.path.join(out, 'SMF_FFB_Grid_MBK25.pdf'))


def fig3(D, pp, out):
    plt = _style(pp)
    Z = D['hdrs']['FFB']['redshifts']
    fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    for col, ax in ((1, axes[0]), (3, axes[1])):
        for run, lab, c, ls, lw in _SMF_MODELS:
            if run not in D['R']:
                continue
            z, m, lo, hi = density_series(D['R'][run], Z, D['hdrs'][run]['volume'], col)
            v = np.isfinite(m)
            if v.sum() > 1:
                ax.plot(z[v], m[v], ls, color=c, linewidth=lw, label=lab)
                b = v & np.isfinite(lo) & np.isfinite(hi)
                if b.sum() > 1:
                    ax.fill_between(z[b], lo[b], hi[b], color=c, alpha=0.2)
    kw = dict(color='black', markersize=8, alpha=0.6, markeredgecolor='k', markeredgewidth=0.8,
              markerfacecolor='gray', linewidth=1.0, zorder=5)
    for ax, loaders in ((axes[0], [(pp.load_madau_dickinson_2014_data, 'o', pp._tex_safe(r'Madau \& Dickinson 14')),
                                   (pp.load_oesch_sfrd_2018_data, '*', 'Oesch+18'),
                                   (pp.load_mcleod_rho_sfr_2024_data, 'v', 'McLeod+23'),
                                   (pp.load_harikane_sfr_density_2023_data, 'D', 'Harikane+23')]),
                        (axes[1], [(pp.load_madau_dickinson_smd_2014_data, 'o', pp._tex_safe(r'Madau \& Dickinson 14')),
                                   (pp.load_kikuchihara_smd_2020_data, 'd', 'Kikuchihara+20'),
                                   (pp.load_papovich_smd_2023_data, 's', 'Papovich+23')])):
        for fn, mk, lab in loaders:
            zo, vo, ep, em = fn()
            if zo is None:
                continue
            w = (zo >= 5) & (zo <= 16)
            if w.any():
                ax.errorbar(zo[w], vo[w], yerr=[em[w], ep[w]], fmt=mk, label=lab, **kw)
    is_sim = lambda l: 'SAGE26' in l or 'FFB' in l or 'MBK25' in l
    for i, ax in enumerate(axes):
        h, l = ax.get_legend_handles_labels()
        sim = [(a, b) for a, b in zip(h, l) if is_sim(b)]
        ob = [(a, b) for a, b in zip(h, l) if not is_sim(b)]
        if i == 0 and sim:
            ax.add_artist(pp._standard_legend(ax, loc='upper right', handles=[a for a, _ in sim],
                                              labels=[b for _, b in sim]))
        if ob:
            pp._standard_legend(ax, loc='lower left', handles=[a for a, _ in ob], labels=[b for _, b in ob])
    axes[0].set_ylabel(r'$\log_{10} \rho_{\mathrm{SFR}}\ (M_\odot\,\mathrm{yr}^{-1}\,\mathrm{Mpc}^{-3})$')
    axes[1].set_ylabel(r'$\log_{10} \rho_\star\ [M_\odot\,\mathrm{Mpc}^{-3}]$')
    axes[1].set_xlabel(r'Redshift')
    for ax, yl, yt in ((axes[0], (-5, -1), [-4, -3, -2, -1]), (axes[1], (3, 8), [3, 4, 5, 6, 7])):
        ax.set_xlim(5, 16)
        ax.xaxis.set_major_locator(plt.MultipleLocator(2.0)); ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.2))
        ax.set_yticks(yt)
        ax.set_ylim(*yl)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.0)
    pp.save_figure(fig, os.path.join(out, 'FFB_Density_Evolution_MBK25.pdf'))


def mbk25_half_mass(pp, z):
    from scipy.optimize import brentq
    f = lambda m: pp.ffb_fraction_mbk25(np.array([m]), z)[0] - 0.5
    return brentq(f, 1e8, 1e13, xtol=1e6, rtol=1e-4) if f(1e8) * f(1e13) < 0 else None


def fig4_curves(D, pp):
    """Binned eps_SFE medians and 16-84% for the Fig 4 samples (haloes >= 20 particles in Mvir)."""
    S = z10_samples(D, pp)
    h = D['hdrs']['FFB']
    mvir_min = MIN_PARTICLES * h['mp_msun']
    use = {}
    for name in ('NO', 'FFB', 'MBK25', 'FFBML2', 'FFBML3'):
        if name not in S:
            continue
        d, rows = S[name]
        rows = rows[(d['StellarMass'][rows] > 0) & (d['Mvir'][rows] >= mvir_min)]
        use[name] = (np.log10(d['Mvir'][rows]), d['StellarMass'][rows] / (h['fb'] * d['Mvir'][rows]))
    allm = np.concatenate([u[0] for k, u in use.items() if k in ('NO', 'FFB', 'MBK25')] or [np.empty(0)])
    if allm.size == 0:
        return use, {}
    lo = np.floor(allm.min() / FIG4_BIN) * FIG4_BIN
    edges = np.arange(lo, allm.max() + FIG4_BIN, FIG4_BIN)
    cur = {}
    for name, (lm, eps) in use.items():
        xs, med, l16, h84 = [], [], [], []
        for i in range(edges.size - 1):
            w = (lm >= edges[i]) & (lm < edges[i + 1])
            if w.sum() >= 5:
                xs.append(0.5 * (edges[i] + edges[i + 1])); med.append(np.median(eps[w]))
                l16.append(np.percentile(eps[w], 16)); h84.append(np.percentile(eps[w], 84))
        cur[name] = tuple(np.array(a) for a in (xs, med, l16, h84))
    return use, cur


def fig4(D, pp, out):
    plt = _style(pp)
    Z = D['hdrs']['FFB']['redshifts']
    z = Z[D['snap_z10']]
    use, cur = fig4_curves(D, pp)
    print('  Fig 4: ' + ', '.join(f'{k} {v[0].size}' for k, v in use.items()) + ' haloes (>= 20 particles)')
    fig, ax = plt.subplots()
    for name, lc, bc, lab, zo in (('NO', 'firebrick', 'darkred', pp.NOFFB_SAME_HALO_LABEL, 3),
                                  ('FFB', 'black', 'grey', 'FFB galaxies', 5),
                                  ('MBK25', 'mediumpurple', 'mediumpurple', 'MBK25 galaxies', 6)):
        if name in cur and cur[name][0].size:
            x, m, l16, h84 = cur[name]
            ax.plot(x, m, color=lc, lw=2, label=lab, zorder=zo)
            ax.fill_between(x, l16, h84, color=bc, alpha=0.2, zorder=zo - 1)
    M_f = pp.ffb_threshold_mass_msun(z)
    ax.axvline(np.log10(M_f), color='goldenrod', ls=':', lw=2, alpha=0.6,
               label=fr'Li+24: $M_{{\rm vir, FFB}}$ = {M_f:.1e} $M_\odot$')
    M_m = mbk25_half_mass(pp, z)
    if M_m is not None:
        ax.axvline(np.log10(M_m), color='goldenrod', ls='--', lw=2, alpha=0.7,
                   label=fr'MBK25: $M_{{\rm vir, MBK25}}$ = {M_m:.1e} $M_\odot$')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\log_{10}(M_{\rm vir}/M_{\odot})$')
    ax.set_ylabel(r'$\varepsilon_{\mathrm{SFE}} \equiv m_*/(\,f_b \, M_{\rm vir})$')
    ax.set_ylim(1e-4, 2.0)
    ax.text(0.98, 0.04, f'$z = {z:.1f}$', transform=ax.transAxes, ha='right', va='bottom', fontsize=20, zorder=10)
    pp._standard_legend(ax, loc='upper left')
    fig.tight_layout()
    pp.save_figure(fig, os.path.join(out, 'SFE_FFB.pdf'))


def fig5_values(D, pp):
    S = z10_samples(D, pp)
    vals = {}
    for name in ('NO', 'FFB', 'MBK25'):
        if name not in S:
            continue
        d, w = S[name]
        sfr = d['SFR'][w]
        with np.errstate(divide='ignore', invalid='ignore'):
            zr = d['MetalsStellarMass'][w] / d['StellarMass'][w] / pp.Z_SUN
        vals[name] = (np.log10(sfr[sfr > 0]), np.log10(zr[zr > 0]))
    return vals


def fig5(D, pp, out):
    plt = _style(pp)
    Z = D['hdrs']['FFB']['redshifts']
    V = fig5_values(D, pp)
    kw = {'NO': dict(alpha=0.5, color='firebrick', edgecolor='darkred', label=pp.NOFFB_SAME_HALO_LABEL),
          'FFB': dict(alpha=0.7, color='black', edgecolor='black', label='FFB galaxies'),
          'MBK25': dict(alpha=0.7, color='mediumpurple', edgecolor='indigo', label='MBK25 galaxies')}
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for i, (ax, xl, xlim) in enumerate(((axes[0], r'$\log_{10}(\mathrm{SFR}\ [M_{\odot}\,\mathrm{yr}^{-1}])$', (-2, 2.7)),
                                        (axes[1], r'$\log_{10}(Z_*/Z_{\odot})$', (-2, -0.5)))):
        arrs = [V[k][i] for k in ('NO', 'FFB', 'MBK25') if k in V and V[k][i].size]
        if arrs:
            allv = np.concatenate(arrs)
            edges = np.linspace(allv.min(), allv.max(), 31) if allv.min() < allv.max() else 30
            for k in ('NO', 'FFB', 'MBK25'):
                if k in V and V[k][i].size:
                    ax.hist(V[k][i], bins=edges, density=True, linewidth=1.2, **kw[k])
        ax.set_xlabel(xl)
        ax.set_ylabel('Normalized Count')
        ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
        ax.set_xlim(*xlim)
    axes[1].text(0.95, 0.95, f'z={Z[D["snap_z10"]]:.1f}', transform=axes[1].transAxes, ha='right', va='top')
    pp._standard_legend(axes[0], loc='upper left')
    fig.tight_layout()
    pp.save_figure(fig, os.path.join(out, 'FFBPropertiesHistograms_Combined.pdf'))


def _tracked_sample(D):
    t = D['track']
    if t is None or 'vals' not in t:
        return None
    return {'snaps': t['snaps'], 'n_gal': t['gids'].size, 'mstar0': t['mstar0'], 'vals': t['vals']}


def fig6(D, pp, out):
    """As plot_12h: plasma colour per halo, grey no-FFB lines, Delta log SFR panels."""
    plt = _style(pp)
    sample = _tracked_sample(D)
    if sample is None:
        print('  Fig 6: no tracked sample; skipped')
        return
    Z = D['hdrs']['FFB']['redshifts']
    z_min, z_hi = TRACK_Z_MIN, Z[sample['snaps'][0]]
    keep = lambda v: z_min <= v <= z_hi
    colors = pp._plasma_colors(sample['n_gal'])
    cmp_c, cmp_lw = '0.6', 0.9
    fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex=True, sharey='row',
                             gridspec_kw={'height_ratios': [3, 1.4]})
    for i, (run, tag) in enumerate((('FFB', 'FFB'), ('MBK25', 'MBK25'))):
        ax_s, ax_r = axes[0, i], axes[1, i]
        pp._draw_tracked(ax_s, sample, run, colors, x='z', y='sfr', x_keep=keep,
                         comparison_color=cmp_c, comparison_lw=cmp_lw, track_lw=cmp_lw)
        pp._draw_tracked_ratio(ax_r, sample, run, colors, x_keep=keep, track_lw=cmp_lw)
        ax_s.set_yscale('log')
        ax_s.set_ylim(1.01e-3, 1.99e4)
        ax_s.text(0.98, 0.95, tag, transform=ax_s.transAxes, ha='right', va='top', fontsize=20, zorder=10)
        if i == 0:
            hd = pp._tracked_legend_handles('FFB/MBK25', '0.3', comparison_color=cmp_c,
                                            comparison_lw=cmp_lw, track_lw=cmp_lw)
            pp._standard_legend(ax_s, loc='lower center', handles=hd, labels=[x.get_label() for x in hd],
                                fontsize='small')
        ax_r.set_ylim(-1.5, 3.0)
        ax_r.set_yticks([-1, 0, 1, 2])
        ax_r.set_xlabel('Redshift')
    axes[0, 0].set_ylabel(r'$\mathrm{SFR}\;[M_{\odot}\,\mathrm{yr}^{-1}]$')
    axes[1, 0].set_ylabel(r'$\Delta\log\,\mathrm{SFR}$' + '\n' + r'(FFB/MBK25 $-$ no-FFB/MBK25)', fontsize='small')
    axes[0, 0].set_xlim(z_min, z_hi)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.0, wspace=0.0)
    pp.save_figure(fig, os.path.join(out, 'SFH_FFB_transitions_sfr_stacked.pdf'))


def fig7(D, pp, out):
    plt = _style(pp)
    import pandas as pd
    import matplotlib.lines as mlines
    curves, _ = nd_curves(D)
    n = ND_PLOT if any(k[1] == ND_PLOT for k in curves) else min((k[1] for k in curves), default=None)
    if n is None:
        print('  Fig 7: no reachable number density; skipped')
        return
    models = [('MBK', 'MBK25 model', 'mediumpurple', '-', 3.4, 5), ('FFB', 'FFB model', 'black', '--', 1.9, 6),
              ('NO', 'no-FFB/MBK25 model', 'firebrick', '-', 2.4, 5)]
    zr = (4.0, 15.0)
    fig, ax = plt.subplots()
    for run, lab, c, ls, lw, zo in models:
        if (run, n) in curves:
            z, m, lo, hi = curves[(run, n)]
            o = np.argsort(z)
            ax.plot(z[o], m[o], color=c, lw=lw, ls=ls, zorder=zo + 1)
            ax.fill_between(z[o], lo[o], hi[o], color=c, alpha=0.2, lw=0, zorder=zo)
    ep = pp._load_epochs()
    if ep is not None:
        ep = ep[(ep['zbest'] >= zr[0]) & (ep['zbest'] <= zr[1])]
        ax.scatter(ep['zbest'], ep['stellar_mass_pipes_zgauss'], s=4, color='#656262', marker='o', alpha=0.2,
                   linewidths=0.3, edgecolors='k', zorder=7)
    p = os.path.join(pp.SIZE_OBS_DIR, 'baggen_disk_2023.ecsv')
    if os.path.exists(p):
        b = pd.read_csv(p, comment='#', sep=r'\s+')
        b = b[(b['z_phot'] >= zr[0]) & (b['z_phot'] <= zr[1])]
        ax.scatter(b['z_phot'], b['log_M_star'], s=18, color='darkorange', marker='D', alpha=0.85,
                   linewidths=0.5, edgecolors='k', zorder=8)
    p = os.path.join(pp.SIZE_OBS_DIR, 'casey_disk_2024.ecsv')
    if os.path.exists(p):
        c = pd.read_csv(p, comment='#', sep=r'\s+')
        c = c[(c['z_phot_BAGPIPES'] >= zr[0]) & (c['z_phot_BAGPIPES'] <= zr[1])]
        lm = np.log10(c['M_star'])
        ax.errorbar(c['z_phot_BAGPIPES'], lm, yerr=np.log10(c['M_star'] + c['M_star_err_up']) - lm, fmt='s',
                    color='dodgerblue', markersize=5, markeredgecolor='k', markeredgewidth=0.5,
                    ecolor='dodgerblue', elinewidth=1.0, capsize=2, zorder=8)
    p = os.path.join(pp.SIZE_OBS_DIR, 'sun_disk_2024.ecsv')
    if os.path.exists(p):
        s = pd.read_csv(p, comment='#', sep=r'\s+')
        s = s[(s['z'] >= zr[0]) & (s['z'] <= zr[1])]
        ax.errorbar(s['z'], s['log_M_star'], yerr=s['log_M_star_err'], fmt='^', color='seagreen', markersize=6,
                    markeredgecolor='k', markeredgewidth=0.5, ecolor='seagreen', elinewidth=1.0, capsize=2,
                    zorder=8)
    for name, z, lm, err in pp._JWST_SPEC:
        ax.errorbar(z, lm, yerr=err, fmt='*', color='gold', markersize=11, markeredgecolor='k',
                    markeredgewidth=0.6, ecolor='gold', elinewidth=1.2, capsize=2, zorder=8)
        dy, va = (8, 'bottom') if name in {"Maisie's", 'GS-z11-0'} else (-8, 'top')
        ax.annotate(name, xy=(z, lm), xytext=(0, dy), textcoords='offset points', fontsize=10,
                    fontweight='bold', ha='center', va=va, color='#0E0C0C', zorder=9)
    hd = [mlines.Line2D([], [], color=c, lw=lw, ls=ls, label=lab) for run, lab, c, ls, lw, _ in models
          if (run, n) in curves]
    pp._standard_legend(ax, loc='upper left', handles=hd, labels=[h.get_label() for h in hd])
    ax.set_xlabel(r'Redshift')
    ax.set_ylabel(r'$\log_{10}\,m_\star\ [M_\odot]$')
    ax.set_xlim(*zr)
    ax.set_ylim(6.5, 12.5)
    fig.tight_layout()
    pp.save_figure(fig, os.path.join(out, 'G_mstar_vs_z_ndensity.pdf'))


def frac_series(D, run):
    Z = D['hdrs']['FFB']['redshifts']
    Rr = D['R'].get(run, {})
    f = np.full(len(Z), np.nan)
    for s, r in Rr.items():
        if 'frac' in r and r['frac'][0] > 0:
            f[s] = r['frac'][1] / r['frac'][0]
    return Z, f


def fig8(D, pp, out):
    plt = _style(pp)
    fig = plt.figure()
    ax = plt.subplot(111)
    for run, lab, col in (('FFB', 'FFB', 'black'), ('MBK', 'MBK25', 'mediumpurple')):
        if run not in D['R']:
            continue
        Z, f = frac_series(D, run)
        m = Z <= 15
        zf, ff = Z[m], f[m]
        edges = np.r_[15.0, 0.5 * (zf[:-1] + zf[1:]), 0.0]
        ax.stairs(ff, np.log10(1 + edges), fill=False, label=lab, edgecolor=col, linewidth=2)
    ax.set_yscale('log')
    ax.set_ylabel('FFB/MBK25 fraction')
    ax.legend(loc='upper left', frameon=False)
    pp._ffb_histogram_x_axes(ax)
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    pp.save_figure(fig, os.path.join(out, 'FFB_Fraction_Li24_vs_MBK25.pdf'))


FIGS = {1: fig1, 2: fig2, 3: fig3, 4: fig4, 5: fig5, 6: fig6, 7: fig7, 8: fig8}


# ========================== NUMBERS ==========================

def numbers(D, pp, path):
    lines = []

    def out(s=''):
        print(s, flush=True)
        lines.append(s)

    H = D['hdrs']
    ref = H['FFB']
    Z = ref['redshifts']
    R = D['R']
    rng = np.random.default_rng(SEED)

    out('A. RUNS')
    for k, d in D['runs'].items():
        h = H.get(k)
        if h is None:
            out(f'  {k:7s} MISSING  {d}')
        else:
            out(f'  {k:7s} mode={h["mode"]} alpha_eff={h["alpha"]:.2f} RegimeDrawFixed={h["regime_fixed"]} '
                f'FFBDrawFixed={h["ffb_fixed"]} box={h["box"]:.1f} V={h["volume"]:.3e} Mpc^3  {d}')

    # ---- analytic (Sections 3, 5.2, 5.4) ----
    from scipy.optimize import brentq
    out('\nB. THRESHOLDS (analytic, this cosmology)')
    zg = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, Z[D['snap_z10']], 11, 12, 14])
    lmF = np.log10([pp.ffb_threshold_mass_msun(z) for z in zg])
    mc = np.array([pp.mbk25_threshold_median_c(z) for z in zg])
    for z, a, (M, c) in zip(zg, lmF, mc):
        out(f'  z={z:5.2f}  log M_FFB={a:6.3f}  log M_MBK25(median c)={np.log10(M):6.3f}  c_med={c:4.2f}  '
            f'diff={np.log10(M) - a:+.3f} dex')
    lmM = np.log10(mc[:, 0])
    for a, b in ((6, 12), (0, 2)):
        s = (zg >= a) & (zg <= b)
        out(f'  MBK25 slope d log M / d log(1+z), {a}<=z<={b}: {np.polyfit(np.log10(1 + zg[s]), lmM[s], 1)[0]:.2f} '
            f'(Sec 3.2: -6.06; Sec 5.3: -6.1 -> -3.4)')
    out(f'  c at the median-c threshold over 6<z<12: {mc[(zg >= 6) & (zg <= 12), 1].min():.2f}-'
        f'{mc[(zg >= 6) & (zg <= 12), 1].max():.2f} (Sec 3.3: 3.19-3.27)')
    for z in (3.0, 2.0, 1.0):
        out(f'  z={z}: MBK25 (median c) minus FFB threshold = '
            f'{np.log10(pp.mbk25_threshold_median_c(z)[0]) - np.log10(pp.ffb_threshold_mass_msun(z)):+.2f} dex '
            f'(Sec 5.3: MBK25 below eq. 1 at z <= 3)')

    def width(fn, z):
        g = lambda lm, t: float(np.atleast_1d(fn(10 ** lm, z))[0]) - t
        return brentq(g, 7, 16, args=(0.84,)) - brentq(g, 7, 16, args=(0.16,))
    out(f'  16-84% width at z=9: FFB sigmoid {width(pp.ffb_fraction, 9):.2f} dex, MBK25 sigma_ln c=0.2 '
        f'{width(pp.ffb_fraction_mbk25, 9):.2f} dex (Sec 3.3/5.2: 0.50, 0.47)')
    for sg in (0.1, 0.3, 0.41, 0.5):
        out(f'    sigma_ln c={sg}: width {width(lambda M, z, s=sg: pp.ffb_fraction_mbk25(M, z, sigma_c=s), 9):.2f} dex '
            f'(Sec 5.2: 0.23, 0.70, 0.96, >1)')
    rho_c = lambda z: 2.775e-7 * ref['h'] ** 2 * (ref['om'] * (1 + z) ** 3 + ref['ol'])   # Msun pc^-3
    gv = lambda M, z: M / (3 * M / (4 * np.pi * 200 * rho_c(z))) ** (2 / 3)                # Msun pc^-2
    out(f'  g_vir/G at 10^10.8 Msun, z=9: {gv(10 ** 10.8, 9.0):.0f} Msun/pc^2 (Sec 5.2: 393)')
    shape = lambda c: c ** 2 / (2 * (np.log1p(c) - c / (1 + c)))
    for M, z in ((1e11, 10.0), (10 ** 10.5, 10.0), (1e12, 0.0), (1e15, 0.0)):
        c = float(pp._c_ishiyama21(np.array([M]), z)[0])
        out(f'  g_max/G for 10^{np.log10(M):.1f} Msun at z={z:g} (median c={c:.1f}): {gv(M, z) * shape(c):.0f} Msun/pc^2'
            f' = {gv(M, z) * shape(c) / 3100:.2f} g_crit (Sec 3.2)')
    for zq in (9.0, Z[D['snap_z10']]):
        out(f'  z={zq:.2f}: M_FFB = {pp.ffb_threshold_mass_msun(zq):.2e}, M_MBK25(f=0.5) = '
            f'{mbk25_half_mass(pp, zq):.2e} Msun')
    mp = ref['mp_msun']
    out(f'  particle mass {mp:.3e} Msun; 20 particles = 10^{np.log10(20 * mp):.2f} Msun')
    out(f'  M_FFB at z={Z[D["snap_z10"]]:.2f} = {pp.ffb_threshold_mass_msun(Z[D["snap_z10"]]) / mp:.0f} particles '
        f'(Sec 6: ~30)')
    out(f'  M_FFB falls below 20 particles at z = {brentq(lambda z: pp.ffb_threshold_mass_msun(z) - 20 * mp, 0, 30):.2f}'
        f' (Sec 4.1: 10.7)')
    th = 977.8 / (100 * ref['h'])
    age = lambda z: 2 * th / (3 * np.sqrt(ref['ol'])) * np.arcsinh(np.sqrt(ref['ol'] / ref['om']) * (1 + z) ** -1.5)
    i10 = D['snap_z10']
    out(f'  snapshot spacing at z={Z[i10]:.2f}: {1e3 * (age(Z[i10 + 1]) - age(Z[i10])):.0f} Myr (Sec 4.3: 57)')
    s1020 = np.where((Z >= 10) & (Z <= 20))[0]
    sp = 1e3 * np.diff(age(Z[s1020]))
    out(f'  {s1020.size} snapshots in 10<=z<=20, median spacing {np.median(sp):.0f} Myr (Sec 5.4: 9, 36)')

    # ---- Fig 2 ----
    out(f'\nC. SMF (Fig 2): log phi differences, m* >= 1e9 Msun bins with >= {SMF_MIN_COUNT} galaxies in both runs')
    x = np.linspace(SMF_RANGE[0], SMF_RANGE[1], int(round((SMF_RANGE[1] - SMF_RANGE[0]) / SMF_BIN)) + 1)
    cen = x[:-1] + 0.5 * SMF_BIN
    smf_maxdiff_FM = []
    for lo, hi in SMF_ZBINS:
        s = snap_in_bin(Z, lo, hi)
        for a, b in (('FFB', 'MBK'), ('FFB100', 'MBK100'), ('FFB', 'NO'), ('MBK', 'NO'), ('FFB100', 'FFB')):
            if s not in R.get(a, {}) or s not in R.get(b, {}):
                continue
            ha, hb = R[a][s]['smf'], R[b][s]['smf']
            ok = (ha >= SMF_MIN_COUNT) & (hb >= SMF_MIN_COUNT) & (cen >= 9.0)
            if not ok.any():
                out(f'  {lo:.0f}<z<{hi:.0f} (z={Z[s]:.2f}) {a:>6s}-{b:<6s}: no bin above 1e9 Msun with '
                    f'>= {SMF_MIN_COUNT} galaxies in both')
                continue
            d = np.log10(ha[ok] / H[a]['volume']) - np.log10(hb[ok] / H[b]['volume'])
            if (a, b) == ('FFB', 'MBK') and Z[s] >= 5:
                smf_maxdiff_FM.append(np.abs(d).max())
            out(f'  {lo:.0f}<z<{hi:.0f} (z={Z[s]:.2f}) {a:>6s}-{b:<6s}: range {d.min():+.2f} to {d.max():+.2f}, '
                f'median {np.median(d):+.3f} dex [log m* {cen[ok][0]:.1f}-{cen[ok][-1]:.1f}]')

    # ---- Fig 3 ----
    out('\nD. SFRD / SMD (Fig 3), dex')
    ser = {(k, c): density_series(R[k], Z, H[k]['volume'], c) for k in ('FFB', 'MBK', 'NO', 'FFB100', 'MBK100')
           if k in R for c in (1, 3)}
    def at(k, c, z):
        if (k, c) not in ser:
            return np.nan
        zz, m, _, _ = ser[(k, c)]
        i = np.argmin(np.abs(zz - z))
        return m[i] if abs(zz[i] - z) < 0.6 else np.nan
    out('   z   | SFRD F-NO  M-NO  F-M    F100-F  M100-M | SMD F-NO  M-NO  F-M')
    for z in (5, 6, 7, 8, 9, 10, 11, 12, 13, 14):
        out(f'  {z:4.1f} | {at("FFB",1,z)-at("NO",1,z):+.2f} {at("MBK",1,z)-at("NO",1,z):+.2f} '
            f'{at("FFB",1,z)-at("MBK",1,z):+.3f} {at("FFB100",1,z)-at("FFB",1,z):+.2f}  '
            f'{at("MBK100",1,z)-at("MBK",1,z):+.2f}  | {at("FFB",3,z)-at("NO",3,z):+.2f} '
            f'{at("MBK",3,z)-at("NO",3,z):+.2f} {at("FFB",3,z)-at("MBK",3,z):+.3f}')
    dens_max_FM = max([np.nanmax(np.abs(ser[('FFB', c)][1][ser[('FFB', c)][0] >= 5]
                                        - np.interp(ser[('FFB', c)][0][ser[('FFB', c)][0] >= 5],
                                                    ser[('MBK', c)][0], ser[('MBK', c)][1])))
                       for c in (1, 3) if ('FFB', c) in ser and ('MBK', c) in ser] or [np.nan])

    def vs_obs(fn, c, zlo, zhi, label):
        zo, vo, _, _ = fn()
        if zo is None:
            return
        w = (zo >= zlo) & (zo <= zhi)
        cells = []
        for k in ('NO', 'FFB', 'MBK', 'FFB100'):
            if (k, c) not in ser:
                continue
            zz, m, _, _ = ser[(k, c)]
            ok = w & (zo >= zz.min()) & (zo <= zz.max())
            if ok.any():
                d = np.interp(zo[ok], zz, m) - vo[ok]
                cells.append(f'{k} median {np.median(d):+.2f} [{d.min():+.2f},{d.max():+.2f}]')
        out(f'  model - {label} ({zlo}<=z<={zhi}): ' + '; '.join(cells))
    vs_obs(pp.load_madau_dickinson_2014_data, 1, 5, 8, 'MD14 SFRD (Sec 4.1: 0.1-0.3)')
    vs_obs(pp.load_oesch_sfrd_2018_data, 1, 5, 16, 'Oesch+18 SFRD')
    vs_obs(pp.load_mcleod_rho_sfr_2024_data, 1, 8, 12, 'McLeod+23 SFRD')
    vs_obs(pp.load_harikane_sfr_density_2023_data, 1, 5, 16, 'Harikane+23 SFRD')
    vs_obs(pp.load_madau_dickinson_smd_2014_data, 3, 5, 16, 'MD14 SMD (Sec 4.1: +0.13)')
    vs_obs(pp.load_papovich_smd_2023_data, 3, 5, 16, 'Papovich+23 SMD (Sec 4.1: -0.06, NO -0.28)')
    vs_obs(pp.load_kikuchihara_smd_2020_data, 3, 8, 9, 'Kikuchihara+20 SMD (Sec 4.1: -0.4 to -0.9)')

    # ---- Fig 4 / 5 ----
    zs = Z[D['snap_z10']]
    out(f'\nE. z = {zs:.2f} (Figs 4, 5)')
    use, cur = fig4_curves(D, pp)
    for k, (x, m, l16, h84) in cur.items():
        if x.size:
            out(f'  Fig 4 {k:7s}: N={use[k][0].size:6d}  binned median eps {m.min():.4f}-{m.max():.4f} '
                f'over log Mvir {x.min():.2f}-{x.max():.2f}; 16-84% {np.nanmin(l16):.4f}-{np.nanmax(h84):.4f}')
    if 'FFB' in cur and 'NO' in cur:
        common = np.intersect1d(np.round(cur['FFB'][0], 3), np.round(cur['NO'][0], 3))
        r = [cur['FFB'][1][np.isclose(cur['FFB'][0], c)][0] / cur['NO'][1][np.isclose(cur['NO'][0], c)][0]
             for c in common]
        out(f'  Fig 4 FFB/NO median eps per bin: {np.round(r, 1).tolist()}')
    thr = pp.ffb_threshold_mass_msun(zs)
    base = None
    for k in ('FFB', 'FFBML2', 'FFBML3'):
        if k in use:
            above = use[k][0] >= np.log10(thr)
            if above.sum():
                med = np.median(use[k][1][above])
                base = med if k == 'FFB' else base
                extra = f' ({np.log10(med / base):+.2f} dex vs FFB)' if base and k != 'FFB' else ''
                out(f'  median eps above M_FFB, {k}: {med:.3f} (N={above.sum()}){extra} (Sec 4.2: +0.4, +0.6 dex, to ~0.2)')
    V = fig5_values(D, pp)
    for i, q in enumerate(('log SFR', 'log Z*/Zsun')):
        cells = [f'{k} {np.median(V[k][i]):+.3f} (N={V[k][i].size})' for k in ('NO', 'FFB', 'MBK25') if k in V]
        out(f'  Fig 5 median {q}: ' + ', '.join(cells))
    if all(k in V for k in ('NO', 'FFB', 'MBK25')):
        out(f'  Fig 5 median SFR [Msun/yr]: NO {10 ** np.median(V["NO"][0]):.2f}, FFB {10 ** np.median(V["FFB"][0]):.2f}, '
            f'MBK25 {10 ** np.median(V["MBK25"][0]):.2f}; ratios FFB/NO {10 ** (np.median(V["FFB"][0]) - np.median(V["NO"][0])):.1f}x, '
            f'MBK25/NO {10 ** (np.median(V["MBK25"][0]) - np.median(V["NO"][0])):.1f}x (Sec 4.3: 1.1, 5.9, 6.6; Sec 6: ~5.5x)')
        out(f'  Fig 5 median Z*/Zsun: NO {10 ** np.median(V["NO"][1]):.3f}, FFB {10 ** np.median(V["FFB"][1]):.3f}, '
            f'MBK25 {10 ** np.median(V["MBK25"][1]):.3f} (Sec 4.3: 0.051, 0.061, 0.063)')

    # ---- Fig 6 ----
    sample = _tracked_sample(D)
    out('\nF. TRACKED HALOES (Fig 6)')
    if sample is not None:
        t = D['track']
        out(f'  the {sample["n_gal"]} most massive of {t["n_pool"]} FFB-selected centrals at '
            f'z={Z[t["select_snap"]]:.2f}; tracked z={Z[t["snaps"][0]]:.2f} to {Z[t["snaps"][-1]]:.2f}')
        zt = Z[np.array(sample['snaps'])]
        for run in ('FFB', 'MBK25', 'noFFB'):
            n = int(np.sum(~np.isnan(sample['vals'][run]['sfr'])))
            out(f'    {run}: {n} galaxy-snapshots')
        for run in ('FFB', 'MBK25'):
            reg = sample['vals'][run]['regime']
            n_tr = [int(np.sum(np.diff(r[~np.isnan(r)]) != 0)) for r in reg]
            out(f'  {run}: of {len(n_tr)} haloes, {sum(n > 1 for n in n_tr)} switch more than once, '
                f'{sum(n == 1 for n in n_tr)} once, {sum(n == 0 for n in n_tr)} never (Sec 4.3: half / a third)')
            a, b = sample['vals'][run]['sfr'], sample['vals']['noFFB']['sfr']
            with np.errstate(invalid='ignore', divide='ignore'):
                dl = np.log10(a / b)
            ok = (a > 0) & (b > 0)
            for lo, hi in ((12.0, 99.0), (7.0, 10.5), (3.0, 7.0)):
                w = ok & (zt[None, :] >= lo) & (zt[None, :] < hi)
                if w.any():
                    out(f'    {lo:g}<=z<{hi:g}: median Delta log SFR {np.median(dl[w]):+.2f} dex (N={w.sum()}) '
                        f'(Sec 4.3: 0.6, 0.35, 0.25)')
            act = ok & (sample['vals'][run]['regime'] == 1)
            if act.any():
                out(f'    in the efficient regime only: median Delta log SFR {np.median(dl[act]):+.2f} dex')
        j = len(sample['snaps']) - 1
        bh = sample['vals']
        with np.errstate(invalid='ignore', divide='ignore'):
            dbh = np.log10(bh['noFFB']['bh'][:, j] / bh['FFB']['bh'][:, j])
            dsf = np.log10(bh['noFFB']['sfr'][:, j] / bh['FFB']['sfr'][:, j])
        dbh[~np.isfinite(dbh)] = np.nan
        dsf[~np.isfinite(dsf)] = np.nan
        f = np.isfinite(dbh)
        if f.any():
            out(f'  at z={zt[j]:.2f} ({f.sum()} haloes with a BH in both): log BH(noFFB) - log BH(FFB) = '
                f'{np.nanmin(dbh):+.2f} to {np.nanmax(dbh):+.2f} '
                f'(median {np.nanmedian(dbh):+.2f}; Sec 4.3: 0.6-1.2 for the most massive); '
                f'log SFR(noFFB) - log SFR(FFB) = {np.nanmin(dsf):+.2f} to {np.nanmax(dsf):+.2f}')
            still = np.isfinite(bh['FFB']['regime'][:, j]) & (bh['FFB']['regime'][:, j] == 1)
            out(f'  haloes still in the FFB regime at z={zt[j]:.2f}: {int(still.sum())}')

    # ---- Fig 7 ----
    curves, why = nd_curves(D)
    n_plot = ND_PLOT if any(k[1] == ND_PLOT for k in curves) else min((k[1] for k in curves), default=ND_PLOT)
    out(f'\nG. log m* at n = {n_plot:.0e} Mpc^-3 (Fig 7)' + ('' if n_plot == ND_PLOT else
        f'; {ND_PLOT:.0e} is out of reach in this volume'))
    get = lambda k: curves.get((k, n_plot))
    if get('FFB') is not None:
        zF, mF = get('FFB')[0], get('FFB')[1]
        nd_max_FM = np.nan
        for z, m in sorted(zip(zF, mF), reverse=True):
            row = f'  z={z:5.2f}  FFB {m:6.2f}'
            for k in ('MBK', 'NO'):
                c = get(k)
                if c is not None and np.any(np.isclose(c[0], z)):
                    v = c[1][np.isclose(c[0], z)][0]
                    row += f'  {k}-FFB {v - m:+.3f}'
                    if k == 'MBK' and z >= 5:
                        nd_max_FM = np.nanmax([nd_max_FM, abs(v - m)])
            out(row)
        c = get('NO')
        if c is not None:
            w = (zF > 8.5) & (zF < 13)
            d = [m - c[1][np.isclose(c[0], z)][0] for z, m in zip(zF[w], mF[w]) if np.any(np.isclose(c[0], z))]
            if d:
                out(f'  FFB-NO over 8.5<z<13: {min(d):+.2f} to {max(d):+.2f} dex (Sec 4.4: 0.6-0.8)')
        for k in ('FFB', 'MBK', 'NO'):
            r = [(Z[s], txt) for (kk, s, n), txt in why.items() if kk == k and n == n_plot and Z[s] > 4]
            if r:
                zz, txt = min(r)
                out(f'  {k} curve stops below z={zz:.2f}: {txt}')
        out(f'\n  max |FFB - MBK25| at z >= 5: SMF {max(smf_maxdiff_FM or [np.nan]):.3f}, SFRD/SMD '
            f'{dens_max_FM:.3f}, m*(n) {nd_max_FM:.3f} dex (abstract / Sec 4.4: ~0.05)')

    # ---- Fig 8 and Section 5.3 ----
    out('\nH. SELECTED FRACTION of resolved centrals (Fig 8, Sec 5.3)')
    fr = {k: frac_series(D, k)[1] for k in ('FFB', 'MBK') if k in R}
    for zt_ in (14, 12, 10, 8, 6, 4, 3, 2, 1, 0.5, 0):
        s = snap_nearest(Z, zt_)
        out(f'  z={Z[s]:5.2f}: ' + '  '.join(f'{k} {fr[k][s]:.2e} ({R[k][s]["frac"][1] if s in R[k] else 0})'
                                             for k in fr))
    for k in fr:
        nz = [s for s in range(len(Z)) if s in R[k] and R[k][s]['frac'][1] > 0]
        out(f'  {k}: last snapshot with a selected central z={Z[max(nz)]:.2f}' if nz else f'  {k}: none selected')

    out('\nI. SELECTION EPISODES (consecutive snapshots in the regime, by GalaxyIndex in its own run)')
    order = sorted(range(len(Z)), key=lambda s: -Z[s])
    for k in ('FFB', 'MBK'):
        if k not in R:
            continue
        L, s0 = episodes(R[k], order)
        for lo, hi in ((4.0, 99.0), (0.0, 4.0)):
            w = (Z[s0] >= lo) & (Z[s0] < hi)
            if w.any():
                out(f'  {k} episodes starting {lo:g}<=z<{hi:g}: N={w.sum()}, mean {L[w].mean():.2f}, median '
                    f'{np.median(L[w]):.0f} snapshots; one snapshot {100 * np.mean(L[w] == 1):.0f}%, '
                    f'<=2 {100 * np.mean(L[w] <= 2):.0f}% (Sec 5.3: one or two)')
    for k, g0 in D['survivors'].items():
        n = D['low_gids'][k].size
        if g0 is None or n == 0:
            out(f'  {k}: no galaxies selected below z=4' if n == 0 else f'  {k}: no survivors')
            continue
        with np.errstate(divide='ignore', invalid='ignore'):
            ss = np.where(g0['SFR'] > 0, np.log10(g0['SFR'] / g0['StellarMass']), -99.0)
        q = ss < SSFR_QUENCHED
        out(f'  {k}: {n} galaxies selected at z<4; {g0["GalaxyIndex"].size} survive to z={Z[D["s_last"]]:.2f}; '
            f'{100 * q.mean():.0f}% quenched (log sSFR < {SSFR_QUENCHED:g}) (Sec 5.3: ~90%)')

    # ---- miniUchuu (Section 5.4) ----
    hu = H.get('UCHUU')
    if hu is not None and 'UCHUU' in R:
        out('\nJ. SMF, miniUchuu (FFB) minus Millennium (FFB), m* > 1e9 Msun (Sec 5.4: -0.1 to -0.2 at z~6-10)')
        Zu = hu['redshifts']
        for lo, hi in SMF_ZBINS:
            s, su = snap_in_bin(Z, lo, hi), snap_in_bin(Zu, lo, hi)
            if su not in R['UCHUU'] or s not in R['FFB']:
                continue
            a, b = R['UCHUU'][su]['smf'], R['FFB'][s]['smf']
            ok = (a >= SMF_MIN_COUNT) & (b >= SMF_MIN_COUNT) & (cen >= 9.0)
            if ok.any():
                d = np.log10(a[ok] / hu['volume']) - np.log10(b[ok] / ref['volume'])
                out(f'  {lo:.0f}<z<{hi:.0f} (Uchuu z={Zu[su]:.2f}, Mill z={Z[s]:.2f}): median {np.median(d):+.2f}, '
                    f'range {d.min():+.2f} to {d.max():+.2f}')

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nNumbers written to {path}')


# ========================== MAIN ==========================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--box', choices=sorted(RUN_SETS), default='mini', help='run directory preset')
    ap.add_argument('--run', action='append', default=[], metavar='KEY=DIR',
                    help='override a run directory, e.g. --run NO=/scratch/noFFB/')
    ap.add_argument('--figs', type=int, nargs='*', default=sorted(FIGS), help='figures to draw (default all)')
    ap.add_argument('--no-numbers', action='store_true', help='skip the numbers report')
    ap.add_argument('-j', '--workers', type=int,
                    default=int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1)))
    ap.add_argument('--out', default=None, help='output directory (default <FFB run>/paper2/)')
    ap.add_argument('--track-z', type=float, default=TRACK_SELECT_Z,
                    help='Fig 6: select the tracked haloes at the snapshot nearest this z (default 10.1)')
    ap.add_argument('--replot', action='store_true', help='draw from OUT/paper2_reductions.pkl, read no models')
    args = ap.parse_args()

    runs = dict(RUN_SETS[args.box])
    for kv in args.run:
        k, v = kv.split('=', 1)
        runs[k] = v
    runs = {k: os.path.abspath(v) if os.path.isabs(v) or os.path.exists(v) else os.path.join(ROOT, v)
            for k, v in runs.items()}
    out = os.path.abspath(args.out) if args.out else os.path.join(runs['FFB'], 'paper2')

    os.chdir(ROOT)                      # paper_plots reads ./data and ./output relative to the root
    sys.path.insert(0, os.path.join(ROOT, 'plotting'))
    import paper_plots as pp

    cache = os.path.join(out, 'paper2_reductions.pkl')
    t0 = time.time()
    if args.replot:
        with open(cache, 'rb') as f:
            D = pickle.load(f)
        print(f'Loaded reductions from {cache}')
    else:
        hdrs = {k: read_header(d) for k, d in runs.items()}
        for k, h in hdrs.items():
            print(f'  {k:7s} {"MISSING" if h is None else str(len(h["files"])) + " file(s)"}  {runs[k]}')
        if hdrs.get('FFB') is None:
            sys.exit('FFB run missing')
        _set_cosmology(pp, hdrs['FFB'])
        D = compute(runs, {k: h for k, h in hdrs.items() if h is not None}, args.workers, pp, args.track_z)
        os.makedirs(out, exist_ok=True)
        with open(cache, 'wb') as f:
            pickle.dump(D, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Reductions saved to {cache} ({os.path.getsize(cache) / 1e6:.1f} MB)')
    _set_cosmology(pp, D['hdrs']['FFB'])

    np.random.seed(SEED)
    for n in args.figs:
        print(f'Figure {n}')
        FIGS[n](D, pp, out)
    if not args.no_numbers:
        numbers(D, pp, os.path.join(out, 'paper2_numbers.txt'))
    print(f'Total {time.time() - t0:.1f} s')


def _set_cosmology(pp, h):
    """paper_plots' threshold and tracking helpers read these module globals."""
    pp.HUBBLE_H, pp.OMEGA_M, pp.OMEGA_L, pp.BOX_SIZE = h['h'], h['om'], h['ol'], h['box']
    pp.REDSHIFTS = list(h['redshifts'])


if __name__ == '__main__':
    main()

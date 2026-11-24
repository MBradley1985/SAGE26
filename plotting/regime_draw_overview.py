"""
regime_draw_overview.py -- RegimeDrawFixed and FFBDrawFixed on mini-Millennium.

Reads output/millennium_draw{00,10,01,11}/model_0.hdf5 (RegimeDrawFixed,
FFBDrawFixed; Li+24 FFB) and output/millennium_draw{bk0,bk1} (BK25 FFB,
FFBDrawFixed 0/1), all full outputs, and writes to cgm_depletion_plots/:

  03_regime_draw_transitions.png
    row 1  CGM/hot: regime changes per snapshot step against Mvir; hot
           fraction against the 0.1 dex sigmoid; changes per main branch of
           z = 0 centrals; hot centrals with r_heat >= Rvir at z = 0
    row 2  Li+24 FFB: FFB changes per snapshot step against Mvir / M_ffb(z);
           FFB fraction against Li+24 eq. 3; changes per main branch at
           z >= 5; length of FFB episodes
    row 3  main-branch histories of example haloes in both draws
  04_regime_draw_galaxies.png
    SMFs at z = 0, 2, 5, 8 with the paper's data, the change of each run
    against 00, the cosmic SFR density, the z = 0 quiescent fraction and SHMR
  05_regime_draw_bk25.png
    the BK25 pair: FFB changes per step, FFB fraction, changes per main
    branch, SMF changes at z = 5 and 8

Run from the repository root:  python3 plotting/regime_draw_overview.py
"""

import os
import sys

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_plots as pp  # noqa: E402  (observation loaders only)

H = 0.73
BOX = 62.5
VOLUME = (BOX / H) ** 3
MSHOCK = 6e11            # Msun, MShockMsun
W_REGIME = 0.1           # dex, determine_and_store_regime()
W_FFB = 0.15             # dex, Li+24 eq. 3
MIN_COUNT = 20
SSFR_CUT = 1e-11
OUTDIR = './cgm_depletion_plots'
RUNS = {'00': ('both per-snapshot (v0.98)', '#222222', '-'),
        '10': ('RegimeDrawFixed 1', '#d6452a', '-'),
        '01': ('FFBDrawFixed 1', '#2a78d6', '-'),
        '11': ('both fixed', '#1f9e5a', '--')}
BK = {'bk0': ('BK25, per-snapshot draw', '#222222', '-'), 'bk1': ('BK25, FFBDrawFixed 1', '#2a78d6', '-')}


def path(n):
    return f'./output/millennium_draw{n}/model_0.hdf5'


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def log_mffb(z):
    return 10.8 - 6.2 * np.log10((1.0 + z) / 10.0)


def steps(n):
    """Every central snapshot step on a main branch: GalaxyIndex, z, log Mvir, regime and FFB before/after."""
    f = h5py.File(path(n), 'r')
    zs = f['Header/snapshot_redshifts'][:]
    out = {k: [] for k in ('gi', 'z', 'lmv', 'r0', 'r1', 'f0', 'f1', 'snap')}
    prev = None
    for s in range(64):
        g = f[f'Snap_{s}']
        if len(g['GalaxyIndex']) == 0:
            continue
        cen = g['Type'][:] == 0
        cur = dict(gi=g['GalaxyIndex'][:][cen], mv=g['Mvir'][:][cen] * 1e10 / H,
                   r=g['Regime'][:][cen], ffb=g['FFBRegime'][:][cen])
        if prev is not None and len(prev['gi']):
            o = np.argsort(prev['gi'])
            p = np.clip(np.searchsorted(prev['gi'][o], cur['gi']), 0, len(o) - 1)
            ok = (prev['gi'][o][p] == cur['gi']) & (cur['mv'] > 0)
            j = o[p][ok]
            out['gi'].append(cur['gi'][ok])
            out['z'].append(np.full(ok.sum(), zs[s]))
            out['snap'].append(np.full(ok.sum(), s))
            out['lmv'].append(np.log10(cur['mv'][ok]))
            out['r0'].append(prev['r'][j])
            out['r1'].append(cur['r'][ok])
            out['f0'].append(prev['ffb'][j])
            out['f1'].append(cur['ffb'][ok])
        prev = cur
    return {k: np.concatenate(v) for k, v in out.items()}


def binned(x, y, m, e, stat=np.mean):
    out = np.full(len(e) - 1, np.nan)
    for j in range(len(out)):
        s = m & (x >= e[j]) & (x < e[j + 1])
        if s.sum() >= MIN_COUNT:
            out[j] = stat(y[s])
    return out


def per_branch_changes(st, changed, mask):
    gi, inv = np.unique(st['gi'][mask], return_inverse=True)
    return np.bincount(inv, weights=changed[mask].astype(float))


def episodes(st, mask):
    """Lengths (in snapshots) of consecutive FFB runs on each main branch."""
    lens = []
    order = np.lexsort((st['snap'], st['gi']))
    gi, f1 = st['gi'][order][mask[order]], st['f1'][order][mask[order]]
    run = 0
    for k in range(len(gi)):
        if k > 0 and gi[k] != gi[k - 1]:
            if run:
                lens.append(run)
            run = 0
        if f1[k] == 1:
            run += 1
        elif run:
            lens.append(run)
            run = 0
    if run:
        lens.append(run)
    return np.array(lens)


def history(n, ids, field):
    f = h5py.File(path(n), 'r')
    out = np.full((64, len(ids)), np.nan)
    mv = np.full((64, len(ids)), np.nan)
    for s in range(64):
        g = f[f'Snap_{s}']
        if len(g['GalaxyIndex']) == 0:
            continue
        gi = g['GalaxyIndex'][:]
        o = np.argsort(gi)
        p = np.clip(np.searchsorted(gi[o], ids), 0, len(gi) - 1)
        ok = (gi[o][p] == ids) & (g['Type'][:][o[p]] == 0)
        out[s, ok] = g[field][:][o[p][ok]]
        mv[s, ok] = g['Mvir'][:][o[p][ok]] * 1e10 / H
    return out, mv


def fig_transitions(st):
    zs = h5py.File(path('00'), 'r')['Header/snapshot_redshifts'][:]
    fig, axes = plt.subplots(3, 4, figsize=(22, 15))
    e = np.arange(11.0, 13.01, 0.1)
    x = 0.5 * (e[1:] + e[:-1])
    xs = x - np.log10(MSHOCK)

    ax = axes[0, 0]
    for n in ('00', '10'):
        s = st[n]
        ax.plot(x, binned(s['lmv'], (s['r0'] != s['r1']).astype(float), np.ones_like(s['lmv'], bool), e),
                color=RUNS[n][1], lw=2, label=RUNS[n][0])
    ax.plot(x, 2 * sigmoid(xs / W_REGIME) * (1 - sigmoid(xs / W_REGIME)), ':', color='#888',
            label='independent draws 2P(1-P)')
    ax.set(xlabel=r'$\log_{10} M_\mathrm{vir}\ [M_\odot]$', ylabel='fraction of snapshot steps', ylim=(0, 0.55),
           title='(a) CGM/hot regime changes per snapshot step')
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for n in ('00', '10'):
        s = st[n]
        ax.plot(x, binned(s['lmv'], s['r1'].astype(float), s['r1'] >= 0, e), color=RUNS[n][1], lw=2, label=RUNS[n][0])
    ax.plot(x, sigmoid(xs / W_REGIME), ':', color='#888', lw=2, label='sigmoid, 0.1 dex')
    ax.set(xlabel=r'$\log_{10} M_\mathrm{vir}\ [M_\odot]$', ylabel='hot fraction', ylim=(0, 1.02),
           title='(b) Hot fraction of centrals (all snapshots)')
    ax.legend(fontsize=8)

    ax = axes[0, 2]
    bins = np.arange(-0.5, 15.5, 1)
    for n in ('00', '10'):
        s = st[n]
        g0 = h5py.File(path(n), 'r')['Snap_63']
        keep = g0['GalaxyIndex'][:][(g0['Type'][:] == 0) & (g0['Mvir'][:] * 1e10 / H > 10 ** 11.8)]
        m = np.isin(s['gi'], keep)
        c = per_branch_changes(s, s['r0'] != s['r1'], m)
        ax.hist(c, bins=bins, histtype='step', color=RUNS[n][1], lw=2, density=True,
                label=f'{RUNS[n][0]}: median {np.median(c):.0f}, mean {np.mean(c):.1f}')
    ax.set(xlabel='regime changes along the main branch', ylabel='fraction of galaxies',
           title=r'(c) z = 0 centrals above $10^{11.8} M_\odot$')
    ax.legend(fontsize=8)

    ax = axes[0, 3]
    for n in ('00', '10'):
        g = h5py.File(path(n), 'r')['Snap_63']
        hc = (g['Type'][:] == 0) & (g['Regime'][:] == 1) & (g['Len'][:] >= 20)
        lmz = np.log10(np.maximum(g['Mvir'][:] * 1e10 / H, 1))
        ax.plot(x, binned(lmz, (g['r_heat'][:] >= 0.999 * g['Rvir'][:]).astype(float), hc, e),
                color=RUNS[n][1], lw=2, label=RUNS[n][0])
    ax.set(xlabel=r'$\log_{10} M_\mathrm{vir}\ [M_\odot]$', ylabel='fraction', ylim=(0, 1.02),
           title=r'(d) Hot centrals with $r_\mathrm{heat} \geq R_\mathrm{vir}$, z = 0')
    ax.legend(fontsize=8)

    ef = np.arange(-1.0, 1.01, 0.1)
    xf = 0.5 * (ef[1:] + ef[:-1])
    ax = axes[1, 0]
    for n in ('00', '01'):
        s = st[n]
        d = s['lmv'] - log_mffb(s['z'])
        ax.plot(xf, binned(d, (s['f0'] != s['f1']).astype(float), s['z'] >= 5, ef), color=RUNS[n][1], lw=2,
                label=RUNS[n][0])
    ax.plot(xf, 2 * sigmoid(xf / W_FFB) * (1 - sigmoid(xf / W_FFB)), ':', color='#888',
            label='independent draws 2P(1-P)')
    ax.set(xlabel=r'$\log_{10} (M_\mathrm{vir} / M_\mathrm{v,ffb}(z))$', ylabel='fraction of snapshot steps',
           ylim=(0, 0.55), title='(e) Li+24 FFB changes per snapshot step, z >= 5')
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for n in ('00', '01'):
        s = st[n]
        d = s['lmv'] - log_mffb(s['z'])
        ax.plot(xf, binned(d, s['f1'].astype(float), s['z'] >= 5, ef), color=RUNS[n][1], lw=2, label=RUNS[n][0])
    ax.plot(xf, sigmoid(xf / W_FFB), ':', color='#888', lw=2, label='Li+24 eq. 3')
    ax.set(xlabel=r'$\log_{10} (M_\mathrm{vir} / M_\mathrm{v,ffb}(z))$', ylabel='FFB fraction', ylim=(0, 1.02),
           title='(f) FFB fraction of centrals, z >= 5')
    ax.legend(fontsize=8)

    ax = axes[1, 2]
    for n in ('00', '01'):
        s = st[n]
        m = s['z'] >= 5
        c = per_branch_changes(s, s['f0'] != s['f1'], m)
        c = c[c > 0]
        ax.hist(c, bins=np.arange(0.5, 15.5, 1), histtype='step', color=RUNS[n][1], lw=2, density=True,
                label=f'{RUNS[n][0]}: median {np.median(c):.0f}, N = {len(c)}')
    ax.set(xlabel='FFB on/off changes along the main branch (z >= 5)', ylabel='fraction of branches',
           title='(g) Branches that change FFB state at least once')
    ax.legend(fontsize=8)

    ax = axes[1, 3]
    for n in ('00', '01'):
        s = st[n]
        L = episodes(s, s['z'] >= 5)
        ax.hist(L, bins=np.arange(0.5, 20.5, 1), histtype='step', color=RUNS[n][1], lw=2, density=True,
                label=f'{RUNS[n][0]}: {len(L)} episodes, mean {np.mean(L):.1f}')
    ax.set(xlabel='consecutive FFB snapshots (z >= 5)', ylabel='fraction of episodes', yscale='log',
           title='(h) Length of FFB episodes')
    ax.legend(fontsize=8)

    # examples: regime (first two) and FFB (last two)
    a = 1.0 / (1.0 + zs)
    g = h5py.File(path('00'), 'r')['Snap_63']
    lmz = np.log10(np.maximum(g['Mvir'][:] * 1e10 / H, 1))
    cand = g['GalaxyIndex'][:][(g['Type'][:] == 0) & (lmz > 11.9) & (lmz < 12.2)]
    r0, mv = history('00', cand, 'Regime')
    nfl = np.nansum(np.abs(np.diff(np.nan_to_num(r0, nan=-1), axis=0)) == 1, axis=0)
    pick = cand[np.where(nfl >= 4)[0][:2]]
    for c, gi in enumerate(pick):
        ax = axes[2, c]
        for n, off in (('00', 0.0), ('10', 0.04)):
            r, m = history(n, np.array([gi]), 'Regime')
            ax.step(a, r[:, 0] + off, where='mid', color=RUNS[n][1], lw=1.5, label=RUNS[n][0])
        ax2 = ax.twinx()
        ax2.plot(a, np.log10(m[:, 0]), color='#888', lw=1)
        ax2.axhline(np.log10(MSHOCK), color='#888', ls='--', lw=0.8)
        ax2.set(ylim=(10.8, 12.6), ylabel=r'$\log_{10} M_\mathrm{vir}$ (grey)')
        ax.set(xlim=(0.15, 1), ylim=(-0.2, 1.4), yticks=[0, 1], yticklabels=['CGM', 'hot'], xlabel='scale factor',
               title=f'(i{c + 1}) z = 0 central: regime in both draws')
        ax.legend(fontsize=8, loc='upper left')
    g = h5py.File(path('00'), 'r')['Snap_21']
    cand = g['GalaxyIndex'][:][g['Type'][:] == 0]
    f00, mv = history('00', cand, 'FFBRegime')
    nfl = np.nansum(np.abs(np.diff(np.nan_to_num(f00[:22], nan=-1), axis=0)) == 1, axis=0)
    pick = cand[np.argsort(-nfl)[:2]]
    for c, gi in enumerate(pick):
        ax = axes[2, 2 + c]
        for n, off in (('00', 0.0), ('01', 0.04)):
            r, m = history(n, np.array([gi]), 'FFBRegime')
            ax.step(zs, r[:, 0] + off, where='mid', color=RUNS[n][1], lw=1.5, label=RUNS[n][0])
        ax2 = ax.twinx()
        ax2.plot(zs, np.log10(m[:, 0]) - log_mffb(zs), color='#888', lw=1)
        ax2.axhline(0, color='#888', ls='--', lw=0.8)
        ax2.set(ylim=(-1, 1), ylabel=r'$\log_{10} M_\mathrm{vir}/M_\mathrm{v,ffb}$ (grey)')
        ax.set(xlim=(16, 4.5), ylim=(-0.2, 1.4), yticks=[0, 1], yticklabels=['normal', 'FFB'], xlabel='redshift',
               title=f'(j{c + 1}) z = 4.9 central: FFB state in both draws')
        ax.legend(fontsize=8, loc='upper left')
    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/03_regime_draw_transitions.png', dpi=85)


def smf(f, s):
    m = f[f'Snap_{s}']['StellarMass'][:] * 1e10 / H
    e = np.arange(7.0, 12.01, 0.1)
    c = np.histogram(np.log10(m[m > 0]), e)[0]
    return 0.5 * (e[1:] + e[:-1]), c


def fig_galaxies():
    files = {n: h5py.File(path(n), 'r') for n in RUNS}
    zs = files['00']['Header/snapshot_redshifts'][:]
    snaps = [int(np.argmin(abs(zs - z))) for z in (0.0, 2.0, 5.0, 8.0)]
    obs = pp._load_smf_grid_observations()
    fig, axes = plt.subplots(3, 4, figsize=(22, 15))
    for c, s in enumerate(snaps):
        z = zs[s]
        ax = axes[0, c]
        for o in obs:
            if abs(o['z'] - z) <= (0.1 if z < 0.2 else 0.4):
                ax.errorbar(o['log_mass'], o['log_phi'], fmt='o', ms=3, color='#8a8a86', mfc='white', lw=0.6)
        base = None
        for n in RUNS:
            x, cnt = smf(files[n], s)
            phi = np.where(cnt > 0, cnt / VOLUME / 0.1, np.nan)
            ax.plot(x, np.log10(phi), color=RUNS[n][1], ls=RUNS[n][2], lw=2, label=RUNS[n][0])
            if n == '00':
                base = cnt
            else:
                axes[1, c].plot(x, np.log10(np.where((cnt > 0) & (base > 0), cnt / np.maximum(base, 1), np.nan)),
                                color=RUNS[n][1], ls=RUNS[n][2], lw=1.8, label=RUNS[n][0])
        band = np.where(base > 0, np.log10(1 + 1 / np.sqrt(np.maximum(base, 1))), np.nan)
        axes[1, c].fill_between(x, -band, band, color='0.85', lw=0, label='Poisson 1-sigma (00)')
        axes[1, c].axhline(0, color='k', lw=0.8)
        ax.set(xlim=(7.5 if z > 4 else 8.5, 12), ylim=(-5.5, -1.0), title=f'SMF, z = {z:.2f}',
               xlabel=r'$\log_{10} M_*$', ylabel=r'$\log_{10}\Phi$')
        axes[1, c].set(xlim=(7.5 if z > 4 else 8.5, 11.8), ylim=(-0.4, 0.4), title=f'SMF change vs 00, z = {z:.2f}',
                       xlabel=r'$\log_{10} M_*$', ylabel=r'$\Delta\log_{10}\Phi$')
    axes[0, 0].legend(fontsize=8, loc='lower left')
    axes[1, 0].legend(fontsize=8, loc='lower left')

    ax = axes[2, 0]
    for n in RUNS:
        f = files[n]
        rho = [(f[f'Snap_{s}']['SfrDisk'][:].sum() + f[f'Snap_{s}']['SfrBulge'][:].sum()) / VOLUME
               if len(f[f'Snap_{s}']['SfrDisk']) else np.nan for s in range(64)]
        ax.plot(zs, np.log10(rho), color=RUNS[n][1], ls=RUNS[n][2], lw=2, label=RUNS[n][0])
    md = np.linspace(0, 12, 200)
    ax.plot(md, np.log10(0.015 * (1 + md) ** 2.7 / (1 + ((1 + md) / 2.9) ** 5.6) * 0.63), color='#8a8a86', lw=1,
            label='Madau & Dickinson 14 (Chabrier)')
    ax.set(xlim=(0, 12), ylim=(-3.5, -0.5), xlabel='redshift', ylabel=r'$\log_{10}\rho_\mathrm{SFR}$',
           title='Cosmic SFR density')
    ax.legend(fontsize=8)

    ax = axes[2, 1]
    e = np.arange(8.5, 11.81, 0.2)
    for n in RUNS:
        g = files[n]['Snap_63']
        ms = g['StellarMass'][:] * 1e10 / H
        q = (g['SfrDisk'][:] + g['SfrBulge'][:]) / np.maximum(ms, 1) < SSFR_CUT
        ax.plot(0.5 * (e[1:] + e[:-1]), binned(np.log10(np.maximum(ms, 1)), q.astype(float), ms > 0, e),
                color=RUNS[n][1], ls=RUNS[n][2], lw=2, label=RUNS[n][0])
    ax.set(ylim=(0, 1), xlabel=r'$\log_{10} M_*$', ylabel=r'fraction sSFR $< 10^{-11}$', title='Quiescent fraction, z = 0')

    ax = axes[2, 2]
    e = np.arange(10.5, 14.01, 0.2)
    for n in RUNS:
        g = files[n]['Snap_63']
        cen = (g['Type'][:] == 0) & (g['Len'][:] >= 20) & (g['StellarMass'][:] > 0)
        ax.plot(0.5 * (e[1:] + e[:-1]), binned(np.log10(np.maximum(g['Mvir'][:] * 1e10 / H, 1)),
                                               np.log10(np.maximum(g['StellarMass'][:] * 1e10 / H, 1)), cen, e, np.median),
                color=RUNS[n][1], ls=RUNS[n][2], lw=2, label=RUNS[n][0])
    ax.set(xlabel=r'$\log_{10} M_\mathrm{vir}$', ylabel=r'median $\log_{10} M_*$', ylim=(8, 12),
           title='Stellar-halo mass relation, centrals, z = 0')

    ax = axes[2, 3]
    e = np.arange(10.5, 14.01, 0.2)
    for n in RUNS:
        g = files[n]['Snap_63']
        cen = (g['Type'][:] == 0) & (g['Len'][:] >= 20) & (g['StellarMass'][:] > 0)
        ssfr = (g['SfrDisk'][:] + g['SfrBulge'][:]) / np.maximum(g['StellarMass'][:] * 1e10 / H, 1)
        ax.plot(0.5 * (e[1:] + e[:-1]), binned(np.log10(np.maximum(g['Mvir'][:] * 1e10 / H, 1)),
                                               (ssfr < SSFR_CUT).astype(float), cen, e),
                color=RUNS[n][1], ls=RUNS[n][2], lw=2, label=RUNS[n][0])
    ax.set(xlabel=r'$\log_{10} M_\mathrm{vir}$', ylabel='quiescent fraction', ylim=(0, 1),
           title='Quiescent fraction of centrals vs halo mass, z = 0')
    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/04_regime_draw_galaxies.png', dpi=85)


def fig_bk25():
    st = {n: steps_bk(n) for n in BK}
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    ef = np.arange(-1.0, 1.01, 0.1)
    xf = 0.5 * (ef[1:] + ef[:-1])
    for n in BK:
        s = st[n]
        d = s['lmv'] - log_mffb(s['z'])
        axes[0].plot(xf, binned(d, (s['f0'] != s['f1']).astype(float), s['z'] >= 5, ef), color=BK[n][1], lw=2,
                     label=BK[n][0])
        axes[1].plot(xf, binned(d, s['f1'].astype(float), s['z'] >= 5, ef), color=BK[n][1], lw=2, label=BK[n][0])
        c = per_branch_changes(s, s['f0'] != s['f1'], s['z'] >= 5)
        c = c[c > 0]
        axes[2].hist(c, bins=np.arange(0.5, 15.5, 1), histtype='step', color=BK[n][1], lw=2, density=True,
                     label=f'{BK[n][0]}: median {np.median(c):.0f}, N = {len(c)}')
    axes[0].set(xlabel=r'$\log_{10} (M_\mathrm{vir} / M_\mathrm{v,ffb}^\mathrm{Li+24}(z))$ (reference scale)',
                ylabel='fraction of snapshot steps', ylim=(0, 0.55), title='BK25: FFB changes per snapshot step, z >= 5')
    axes[1].set(xlabel=r'$\log_{10} (M_\mathrm{vir} / M_\mathrm{v,ffb}^\mathrm{Li+24}(z))$', ylabel='FFB fraction',
                ylim=(0, 1.02), title='BK25: FFB fraction of centrals, z >= 5')
    axes[2].set(xlabel='FFB on/off changes along the main branch (z >= 5)', ylabel='fraction of branches',
                title='BK25: branches that change FFB state')
    files = {n: h5py.File(path(n), 'r') for n in BK}
    zs = files['bk0']['Header/snapshot_redshifts'][:]
    for z, ls in ((5.0, '-'), (8.0, '--')):
        s = int(np.argmin(abs(zs - z)))
        x, c0 = smf(files['bk0'], s)
        _, c1 = smf(files['bk1'], s)
        axes[3].plot(x, np.log10(np.where((c0 > 0) & (c1 > 0), c1 / np.maximum(c0, 1), np.nan)), color=BK['bk1'][1],
                     ls=ls, lw=1.8, label=f'z = {zs[s]:.2f}')
    axes[3].axhline(0, color='k', lw=0.8)
    axes[3].set(xlim=(7.5, 11), ylim=(-0.4, 0.4), xlabel=r'$\log_{10} M_*$', ylabel=r'$\Delta\log_{10}\Phi$',
                title='BK25: SMF change, fixed - per-snapshot draw')
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/05_regime_draw_bk25.png', dpi=85)


def steps_bk(n):
    return steps(n)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    st = {n: steps(n) for n in RUNS}
    for n in RUNS:
        s = st[n]
        near = (s['lmv'] > 11.4) & (s['lmv'] < 12.3)
        print(f'{n}: regime changes per step at 10^11.4-12.3: {np.mean(s["r0"][near] != s["r1"][near]):.3f}; '
              f'FFB changes per step at z>=5, |log M/Mffb|<0.3: '
              f'{np.mean((s["f0"] != s["f1"])[(s["z"] >= 5) & (abs(s["lmv"] - log_mffb(s["z"])) < 0.3)]):.3f}')
    fig_transitions(st)
    fig_galaxies()
    fig_bk25()
    print('wrote', OUTDIR + '/03-05')


if __name__ == '__main__':
    main()

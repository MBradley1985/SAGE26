"""
streamfix_overview.py -- one multi-panel figure comparing StreamRheatFix 0 and 1 on mini-Millennium.

StreamRheatFix 0 (v0.98) passes the hot-regime Dekel & Birnboim (2006) cold
streams through the AGN step with the cooling flow; 1 adds them after it,
unsuppressed by r_heat.

Rows:
  1  stellar mass functions at the four output redshifts, with observations
  2  z = 0 scaling relations and the cosmic SFR density history (as
     paper_plots.plot_16_sfrd_history), with observations
  3  changes relative to StreamRheatFix 0, with the scatter of the baseline
     (Poisson or bootstrap) shaded
  4  the mechanism in hot-regime centrals: cooling shut-off at z = 2 and 0,
     the stream share of the cooling at z = 2, quenching vs halo mass

Mini-Millennium (solid) reads output/millennium_streamfix{0,1}/model_0.hdf5
(NumOutputs 4: snapshots 63, 37, 32, 27) and the SFR density histories in
rheat_hotonly_plots/sfrd_history_streamfix{0,1}.npz (summed SFR / volume at
all 64 snapshots with 100-sample bootstrap 16-84%, from full-output runs).
microUchuu (dashed) reads every model_*.hdf5 of output/microuchuu (v0.98, so
StreamRheatFix 0) and output/microuchuu_streamfix1, at the snapshots nearest
the Millennium redshifts; its SFR density history is computed from all 50
snapshots and cached in rheat_hotonly_plots/sfrd_history_uchuu_streamfix{0,1}.npz.
A missing run is skipped.  Observation loaders are from paper_plots.py.

Run from the repository root:  python3 plotting/streamfix_overview.py
"""

import os
import sys

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_plots as pp  # noqa: E402  (observation loaders only)

OUT = './rheat_hotonly_plots/06_streamfix_overview.png'
MSHOCK = 6e11                     # Msun, MShockMsun
MIN_LEN = 20                      # resolved haloes, as paper_plots.MIN_PARTICLES
SSFR_CUT = 1e-11                  # yr^-1
MIN_COUNT = 10                    # minimum galaxies per bin for a plotted statistic
MILL_SNAPS = (63, 37, 32, 27)

# Per simulation: run directories by StreamRheatFix, h, box (Mpc/h), line style
SIMS = {
    'mill':  dict(name='mini-Millennium', h=0.73, box=62.5, ls='-',
                  runs={0: './output/millennium_streamfix0', 1: './output/millennium_streamfix1',
                        2: './output/millennium_streamfix2'},
                  sfrd='./rheat_hotonly_plots/sfrd_history_streamfix{}.npz'),
    'uchuu': dict(name='microUchuu', h=0.6774, box=100.0, ls='--',
                  runs={0: './output/microuchuu', 1: './output/microuchuu_streamfix1',
                        2: './output/microuchuu_streamfix2'},
                  sfrd='./rheat_hotonly_plots/sfrd_history_uchuu_streamfix{}.npz'),
}
OPTIONS = {
    0: ('0  v0.98: streams suppressed by r_heat', '#222222'),
    1: ('1  streams bypass the AGN step',         '#eb6834'),
    2: ('2  as 1, streams <= f_stream x infall',  '#2a78d6'),
}
OBS_COLOR = '#8a8a86'

FIELDS = ['Type', 'Regime', 'Len', 'Mvir', 'StellarMass', 'BulgeMass', 'BlackHoleMass',
          'SfrDisk', 'SfrBulge', 'CoolingRate', 'CGMgas', 'mdot_stream', 'mdot_cool', 'r_heat', 'Rvir']


def run_files(sim, n):
    d = SIMS[sim]['runs'][n]
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.startswith('model_') and f.endswith('.hdf5')) \
        if os.path.isdir(d) else []


def load(sim, n, snap):
    """One snapshot of one run, all files concatenated, masses in Msun."""
    h = SIMS[sim]['h']
    parts = [h5py.File(f, 'r')[f'Snap_{snap}'] for f in run_files(sim, n)]
    d = {k: np.concatenate([g[k][:].astype(float) for g in parts]) for k in FIELDS}
    for k in ('Mvir', 'StellarMass', 'BulgeMass', 'BlackHoleMass', 'CGMgas'):
        d[k] *= 1e10 / h
    d['sfr'] = d['SfrDisk'] + d['SfrBulge']
    d['ssfr'] = d['sfr'] / np.maximum(d['StellarMass'], 1.0)
    d['lms'] = np.log10(np.maximum(d['StellarMass'], 1.0))
    d['lmv'] = np.log10(np.maximum(d['Mvir'], 1.0))
    d['cen'] = (d['Type'] == 0) & (d['Len'] >= MIN_LEN)
    return d


def volume(sim):
    return (SIMS[sim]['box'] / SIMS[sim]['h']) ** 3


def sfrd_history(sim, n):
    """Summed SFR / volume at every snapshot, bootstrap 16-84%; cached for microUchuu."""
    path = SIMS[sim]['sfrd'].format(n)
    if sim == 'uchuu' and not os.path.exists(path):
        files = [h5py.File(f, 'r') for f in run_files(sim, n)]
        zs = files[0]['Header/snapshot_redshifts'][:]
        dens, lo, hi = (np.zeros(len(zs)) for _ in range(3))
        np.random.seed(2222)
        for snap in range(len(zs)):
            if f'Snap_{snap}' not in files[0]:
                continue
            x = np.concatenate([f[f'Snap_{snap}']['SfrDisk'][:].astype(float) + f[f'Snap_{snap}']['SfrBulge'][:].astype(float)
                                for f in files])
            if len(x) == 0:
                continue
            dens[snap] = x.sum() / volume(sim)
            a, b = pp._bootstrap_sum_range(x, 100)
            lo[snap], hi[snap] = a / volume(sim), b / volume(sim)
        np.savez(path, z=zs, dens=dens, lo=lo, hi=hi)
    return np.load(path)


def binned(x, y, mask, edges, stat):
    """stat(y) in bins of x; NaN where a bin holds fewer than MIN_COUNT."""
    out = np.full(len(edges) - 1, np.nan)
    for j in range(len(out)):
        s = mask & (x >= edges[j]) & (x < edges[j + 1])
        if s.sum() >= MIN_COUNT:
            out[j] = stat(y[s])
    return out


def centres(edges):
    return 0.5 * (edges[1:] + edges[:-1])


def line(ax, x, y, key, **kw):
    sim, n = key
    ax.plot(x, y, color=OPTIONS[n][1], ls=SIMS[sim]['ls'], lw=2.4 if n == 0 else 1.7,
            zorder=10 if n == 0 else 5, **kw)


def obs_points(ax, x, y, yerr=None, xerr=None, marker='o', label=None, **kw):
    ax.errorbar(x, y, yerr=yerr, xerr=xerr, fmt=marker, ms=4, color=OBS_COLOR, mfc='white',
                mec=OBS_COLOR, ecolor=OBS_COLOR, elinewidth=0.8, capsize=0, zorder=2, label=label, **kw)


def main():
    keys = [(sim, n) for sim in SIMS for n in OPTIONS if run_files(sim, n)]
    sims = [sim for sim in SIMS if (sim, 0) in keys]
    zs = h5py.File(run_files('mill', 0)[0], 'r')['Header/snapshot_redshifts'][:]
    zslot = [zs[s] for s in MILL_SNAPS]
    snaps = {'mill': MILL_SNAPS}
    if 'uchuu' in sims:
        f = h5py.File(run_files('uchuu', 0)[0], 'r')
        zu = f['Header/snapshot_redshifts'][:]
        have = [i for i in range(len(zu)) if f'Snap_{i}' in f]
        snaps['uchuu'] = tuple(min(have, key=lambda i: abs(zu[i] - z)) for z in zslot)
        print('microUchuu snapshots', snaps['uchuu'], 'z =', [round(zu[i], 2) for i in snaps['uchuu']])
    data = {(k, i): load(k[0], k[1], snaps[k[0]][i]) for k in keys for i in range(4)}
    diffs = [k for k in keys if k[1] > 0]
    # SFR histories are cached, so a deleted run still has its line
    sfrd_keys = [(sim, n) for sim in SIMS for n in OPTIONS
                 if (sim, n) in keys or os.path.exists(SIMS[sim]['sfrd'].format(n))]

    fig, axes = plt.subplots(4, 4, figsize=(22, 21))
    smf_edges = np.arange(8.0, 12.01, 0.1)
    smf_x = centres(smf_edges)

    # ---------------- Row 1: stellar mass functions + observations ----------------
    smf_obs = pp._load_smf_grid_observations()
    markers = {}
    for i, z in enumerate(zslot):
        ax = axes[0, i]
        for o in smf_obs:
            if abs(o['z'] - z) <= (0.1 if z < 0.2 else 0.3):
                m = markers.setdefault(o['label'], 'osD^vP*Xh'[len(markers) % 9])
                yerr = None if o['err_lo'] is None else [o['err_lo'], o['err_hi']]
                obs_points(ax, o['log_mass'], o['log_phi'], yerr=yerr, marker=m, label=f"{o['label']} (z={o['z']:.2f})")
        for k in keys:
            cnt = np.histogram(data[k, i]['lms'], smf_edges)[0]
            phi = np.where(cnt > 0, cnt / volume(k[0]) / 0.1, np.nan)
            line(ax, smf_x, np.log10(phi), k)
        ax.set(xlim=(8.5, 12.0), ylim=(-5.2, -1.3), title=f'Stellar mass function, z = {z:.2f}',
               xlabel=r'$\log_{10} M_*\ [M_\odot]$', ylabel=r'$\log_{10}\Phi\ [\mathrm{Mpc^{-3}\,dex^{-1}}]$')
        ax.legend(loc='lower left', fontsize=7.5, frameon=False)

    # ---------------- Row 2: z = 0 relations + observations ----------------
    d0 = {k: data[k, 0] for k in keys}
    ms_edges = np.arange(8.5, 11.81, 0.2)

    # (a) quiescent fraction of all galaxies vs Baldry+12 red fraction
    ax = axes[1, 0]
    b12 = pp.load_baldry_blue_red()
    red = np.interp(b12['sf_mass'], b12['q_mass'], 10 ** b12['q_phi'])
    frac = red / (red + 10 ** b12['sf_phi'])
    obs_points(ax, b12['sf_mass'], frac, marker='o', label='Baldry+12 red fraction (colour)')
    for k in keys:
        d = d0[k]
        line(ax, centres(ms_edges), binned(d['lms'], (d['ssfr'] < SSFR_CUT).astype(float),
                                           d['StellarMass'] > 0, ms_edges, np.mean), k)
    ax.set(xlim=(8.5, 11.8), ylim=(0, 1), title='Quiescent fraction, all galaxies, z = 0',
           xlabel=r'$\log_{10} M_*\ [M_\odot]$', ylabel=r'fraction with sSFR $< 10^{-11}\,\mathrm{yr^{-1}}$')
    ax.legend(loc='upper left', fontsize=8, frameon=False)

    # (b) stellar-to-halo mass relation, centrals
    ax = axes[1, 1]
    shmr = pp.load_shmr_observations()
    ax.plot(shmr['moster']['mvir'], shmr['moster']['mstar'], color=OBS_COLOR, lw=2, ls=':', label='Moster+13')
    k = shmr['kravtsov']
    obs_points(ax, k['mvir'], k['mstar'], marker='s', label='Kravtsov+18')
    mv_edges = np.arange(10.4, 14.01, 0.2)
    for k in keys:
        d = d0[k]
        line(ax, centres(mv_edges), binned(d['lmv'], d['lms'], d['cen'] & (d['StellarMass'] > 0), mv_edges, np.median), k)
    ax.axvline(np.log10(MSHOCK), color=OBS_COLOR, lw=0.8, ls='--')
    ax.set(xlim=(10.4, 14.0), ylim=(7.5, 12.0), title='Stellar-halo mass relation, centrals, z = 0',
           xlabel=r'$\log_{10} M_\mathrm{vir}\ [M_\odot]$', ylabel=r'median $\log_{10} M_*\ [M_\odot]$')
    ax.legend(loc='upper left', fontsize=8, frameon=False)

    # (c) black hole - bulge mass
    ax = axes[1, 2]
    bh = pp.load_bh_bulge_observations()
    obs_points(ax, bh['log_M_sph'], bh['log_M_BH'], marker='o', label='observed (paper compilation)')
    ax.plot(bh['haring_rix_x'], bh['haring_rix_y'], color=OBS_COLOR, lw=2, ls=':', label='Haring & Rix 04')
    mb_edges = np.arange(8.5, 12.01, 0.2)
    for k in keys:
        d = d0[k]
        ok = (d['BulgeMass'] > 0) & (d['BlackHoleMass'] > 0)
        line(ax, centres(mb_edges), binned(np.log10(np.maximum(d['BulgeMass'], 1)),
                                           np.log10(np.maximum(d['BlackHoleMass'], 1)), ok, mb_edges, np.median), k)
    ax.set(xlim=(8.5, 12.0), ylim=(5.5, 10.0), title='Black hole - bulge mass, z = 0',
           xlabel=r'$\log_{10} M_\mathrm{bulge}\ [M_\odot]$', ylabel=r'median $\log_{10} M_\mathrm{BH}\ [M_\odot]$')
    ax.legend(loc='upper left', fontsize=8, frameon=False)

    # (d) cosmic SFR density history, as paper_plots.plot_16_sfrd_history
    ax = axes[1, 3]
    obs_style = dict(markerfacecolor='gray', markeredgecolor='k', markeredgewidth=1.0,
                     ecolor='k', color='k', ms=6, lw=1.0, alpha=0.6, ls='none', zorder=2)
    z_d18, sfrd_d18, err_d18, zlo_d18, zhi_d18 = pp.load_driver_sfrd_2018_data()
    if z_d18 is not None:
        ax.errorbar(z_d18, sfrd_d18, yerr=err_d18, xerr=[zlo_d18, zhi_d18], fmt='o', label='Driver+18', **obs_style)
    z_ds, sfrd_ds, elo_ds, ehi_ds, zlo_ds, zhi_ds = pp.load_dsilva_sfrd_2023_data()
    if z_ds is not None:
        ax.errorbar(z_ds, sfrd_ds, yerr=[elo_ds, ehi_ds], xerr=[zlo_ds, zhi_ds], fmt='D', label="D'Silva+23", **obs_style)
    for dataset, fmt in (('Novak+17', '^'), ('Bouwens+15', '<'), ('Finkelstein+15', '>')):
        z_t, sfrd_t, eu_t, el_t = pp.load_sfrd_table(dataset)
        if z_t is not None:
            ax.errorbar(z_t, sfrd_t, yerr=[el_t, eu_t], fmt=fmt, label=dataset, **obs_style)
    zf = np.linspace(0, 8, 200)
    md14 = np.log10(0.015 * (1 + zf)**2.7 / (1 + ((1 + zf) / 2.9)**5.6)) + pp.imf_shift('Madau+Dickinson 14')
    ax.plot(zf, md14, color='gray', lw=1.5, alpha=0.6, zorder=2, label='Madau & Dickinson 14')
    cw = './data/sfrd/CSFRD_inferred_from_SMD.ecsv'
    if pp.HAS_ASTROPY and os.path.exists(cw):
        t = pp.Table.read(cw, format='ascii.ecsv')
        zc = np.array(t['Redshift'])
        s50, s16, s84 = (np.log10(np.array(t[c])) for c in ('sfrd_50', 'sfrd_16', 'sfrd_84'))
        dz = np.median(np.diff(zc)) if len(zc) > 1 else 0.25
        sel = slice(None, None, max(1, int(round(0.25 / dz))) if dz > 0 else 1)
        ax.errorbar(zc[sel], s50[sel], yerr=[(s50 - s16)[sel], (s84 - s50)[sel]], fmt='s', label='COSMOS-Web', **obs_style)
    sfrd = {k: sfrd_history(*k) for k in sfrd_keys}
    for k in sfrd_keys:
        h = sfrd[k]
        ok = h['dens'] > 0
        line(ax, h['z'][ok], np.log10(h['dens'][ok]), k)
        okb = ok & (h['lo'] > 0)
        ax.fill_between(h['z'][okb], np.log10(h['lo'][okb]), np.log10(h['hi'][okb]),
                        color=OPTIONS[k[1]][1], alpha=0.2, lw=0, zorder=3)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.set(xlim=(0.0, 7.5), ylim=(-3.0, -0.5), title='Cosmic SFR density history',
           xlabel='Redshift', ylabel=r'$\log_{10}\rho_\mathrm{SFR}\ [M_\odot\,\mathrm{yr^{-1}\,Mpc^{-3}}]$')
    ax.legend(loc='lower left', fontsize=7.5, frameon=False, ncol=2)

    # ---------------- Row 3: changes relative to RheatFix 0 ----------------
    for col, i in enumerate((0, 2)):
        ax = axes[2, col]
        banded = False
        for sim, n in diffs:
            c0 = np.histogram(data[(sim, 0), i]['lms'], smf_edges)[0]
            ok0 = c0 >= MIN_COUNT
            if sim == 'mill' and not banded:
                banded = True
                band = 0.434 * np.sqrt(2.0 / np.maximum(c0, 1))      # Poisson scatter of a ratio of two such counts
                ax.fill_between(smf_x[ok0], -band[ok0], band[ok0], color=OBS_COLOR, alpha=0.25, lw=0,
                                label='Poisson 1-sigma (mini-Millennium)')
            c = np.histogram(data[(sim, n), i]['lms'], smf_edges)[0]
            ok = ok0 & (c >= MIN_COUNT)
            line(ax, smf_x[ok], np.log10(c[ok] / c0[ok]), (sim, n))
        s = 63 if i == 0 else 32
        ax.axhline(0, color='#222222', lw=1.2)
        ax.set(xlim=(8.5, 11.6), ylim=(-0.3, 0.3), title=f'SMF change vs v0.98, z = {zs[s]:.2f}',
               xlabel=r'$\log_{10} M_*\ [M_\odot]$', ylabel=r'$\Delta\log_{10}\Phi$')
        ax.legend(loc='lower left', fontsize=8, frameon=False)

    ax = axes[2, 2]
    banded = False
    for sim, n in [k for k in sfrd_keys if k[1] > 0 and (k[0], 0) in sfrd]:
        h0 = sfrd[sim, 0]
        ok0 = (h0['dens'] > 0) & (h0['lo'] > 0)
        if sim == 'mill' and not banded:
            banded = True
            ax.fill_between(h0['z'][ok0], np.log10(h0['lo'][ok0] / h0['dens'][ok0]), np.log10(h0['hi'][ok0] / h0['dens'][ok0]),
                            color=OBS_COLOR, alpha=0.25, lw=0, label='bootstrap 16-84% (mini-Millennium)')
        h = sfrd[sim, n]
        ok = ok0 & (h['dens'] > 0)
        line(ax, h['z'][ok], np.log10(h['dens'][ok] / h0['dens'][ok]), (sim, n))
    ax.axhline(0, color='#222222', lw=1.2)
    ax.set(xlim=(0.0, 7.5), ylim=(-0.3, 0.3), title='SFR density change vs v0.98',
           xlabel='Redshift', ylabel=r'$\Delta\log_{10}\rho_\mathrm{SFR}$')
    ax.legend(loc='lower left', fontsize=8, frameon=False)

    ax = axes[2, 3]
    for sim, n in diffs:
        a0 = d0[sim, 0]
        r0 = binned(a0['lmv'], a0['lms'], a0['cen'] & (a0['StellarMass'] > 0), mv_edges, np.median)
        d = d0[sim, n]
        r = binned(d['lmv'], d['lms'], d['cen'] & (d['StellarMass'] > 0), mv_edges, np.median)
        line(ax, centres(mv_edges), r - r0, (sim, n))
    ax.axhline(0, color='#222222', lw=1.2)
    ax.axvline(np.log10(MSHOCK), color=OBS_COLOR, lw=0.8, ls='--')
    ax.set(xlim=(10.4, 14.0), ylim=(-0.3, 0.3), title='SHMR change vs v0.98, centrals, z = 0',
           xlabel=r'$\log_{10} M_\mathrm{vir}\ [M_\odot]$', ylabel=r'$\Delta$ median $\log_{10} M_*$')

    # ---------------- Row 4: mechanism in hot-regime centrals ----------------
    hot_edges = np.arange(11.0, 13.61, 0.1)
    near_edges = np.arange(11.0, 13.01, 0.1)
    d2 = {k: data[k, 2] for k in keys}
    for n in keys:
        for col, d in ((0, d2[n]), (1, d0[n])):
            zero = (d['CoolingRate'] == 0).astype(float)
            line(axes[3, col], centres(hot_edges), binned(d['lmv'], zero, d['cen'] & (d['Regime'] == 1), hot_edges, np.mean), n)
        d = d2[n]
        tot = d['mdot_stream'] + d['mdot_cool']
        share = np.where(tot > 0, d['mdot_stream'] / np.where(tot > 0, tot, 1), np.nan)
        line(axes[3, 2], centres(hot_edges), binned(d['lmv'], share, d['cen'] & (d['Regime'] == 1) & (tot > 0), hot_edges, np.median), n)
        d = d0[n]
        line(axes[3, 3], centres(near_edges), binned(d['lmv'], (d['ssfr'] < SSFR_CUT).astype(float), d['cen'], near_edges, np.mean), n)
    axes[3, 0].set(title=f'Hot regime: centrals that do not cool, z = {zs[32]:.2f}', ylabel='fraction with zero cooling',
                   xlim=(11.0, 13.6), ylim=(0, 1))
    axes[3, 1].set(title='Hot regime: centrals that do not cool, z = 0', ylabel='fraction with zero cooling',
                   xlim=(11.0, 13.6), ylim=(0, 1))
    axes[3, 2].set(title=f'Stream share of the cooling, hot centrals, z = {zs[32]:.2f}',
                   ylabel=r'median $\dot m_\mathrm{stream} / (\dot m_\mathrm{stream} + \dot m_\mathrm{cool})$',
                   xlim=(11.0, 13.6), ylim=(0, 1))
    axes[3, 3].set(title='Quiescent fraction vs halo mass, centrals, z = 0', ylabel=r'fraction with sSFR $< 10^{-11}$',
                   xlim=(11.0, 13.0), ylim=(0, 1))
    for ax in axes[3]:
        ax.set_xlabel(r'$\log_{10} M_\mathrm{vir}\ [M_\odot]$')
        ax.axvline(np.log10(MSHOCK), color=OBS_COLOR, lw=0.8, ls='--')

    for ax in axes.ravel():
        ax.grid(alpha=0.2)

    handles = [Line2D([], [], color=c, lw=2.4 if n == 0 else 1.9, label=lab) for n, (lab, c) in OPTIONS.items()]
    handles += [Line2D([], [], color='#555555', ls=SIMS[sim]['ls'], lw=1.9, label=SIMS[sim]['name']) for sim in sims]
    fig.legend(handles=handles, loc='upper center', ncol=len(handles), fontsize=12, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle('StreamRheatFix on mini-Millennium (solid) and microUchuu (dashed): observations (rows 1-2), changes vs v0.98 (row 3), '
                 'hot-regime mechanism (row 4; dashed line $M_\\mathrm{shock}$)', y=0.975, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=90)
    print('wrote', OUT)


if __name__ == '__main__':
    main()

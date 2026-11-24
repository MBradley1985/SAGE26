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

Reads output/millennium_streamfix{0,1}/model_0.hdf5 (NumOutputs 4: snapshots
63, 37, 32, 27), the SFR density histories in
rheat_hotonly_plots/sfrd_history_streamfix{0,1}.npz (summed SFR / volume at
all 64 snapshots with 100-sample bootstrap 16-84%, from full-output runs) and
the observation loaders in paper_plots.py.

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

RUN_DIR = './output/millennium_streamfix{}/model_0.hdf5'
SFRD_FILE = './rheat_hotonly_plots/sfrd_history_streamfix{}.npz'
OUT = './rheat_hotonly_plots/06_streamfix_overview.png'
SNAPS = (63, 37, 32, 27)
H = 0.73
VOLUME = (62.5 / H) ** 3          # Mpc^3
MSHOCK = 6e11                     # Msun, MShockMsun
MIN_LEN = 20                      # resolved haloes, as paper_plots.MIN_PARTICLES
SSFR_CUT = 1e-11                  # yr^-1
MIN_COUNT = 10                    # minimum galaxies per bin for a plotted statistic

OPTIONS = {
    0: ('0  v0.98: streams suppressed by r_heat', '#222222', '-'),
    1: ('1  streams bypass the AGN step',         '#eb6834', '-'),
}
OBS_COLOR = '#8a8a86'

FIELDS = ['Type', 'Regime', 'Len', 'Mvir', 'StellarMass', 'BulgeMass', 'BlackHoleMass',
          'SfrDisk', 'SfrBulge', 'CoolingRate', 'CGMgas', 'mdot_stream', 'mdot_cool', 'r_heat', 'Rvir']


def load(n, snap):
    """One snapshot of one option, masses in Msun."""
    g = h5py.File(RUN_DIR.format(n), 'r')[f'Snap_{snap}']
    d = {k: g[k][:].astype(float) for k in FIELDS}
    for k in ('Mvir', 'StellarMass', 'BulgeMass', 'BlackHoleMass', 'CGMgas'):
        d[k] *= 1e10 / H
    d['sfr'] = d['SfrDisk'] + d['SfrBulge']
    d['ssfr'] = d['sfr'] / np.maximum(d['StellarMass'], 1.0)
    d['lms'] = np.log10(np.maximum(d['StellarMass'], 1.0))
    d['lmv'] = np.log10(np.maximum(d['Mvir'], 1.0))
    d['cen'] = (d['Type'] == 0) & (d['Len'] >= MIN_LEN)
    return d


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


def line(ax, x, y, n, **kw):
    label, color, ls = OPTIONS[n]
    ax.plot(x, y, color=color, ls=ls, lw=2.6 if n == 0 else 1.7, zorder=10 if n == 0 else 5, **kw)


def obs_points(ax, x, y, yerr=None, xerr=None, marker='o', label=None, **kw):
    ax.errorbar(x, y, yerr=yerr, xerr=xerr, fmt=marker, ms=4, color=OBS_COLOR, mfc='white',
                mec=OBS_COLOR, ecolor=OBS_COLOR, elinewidth=0.8, capsize=0, zorder=2, label=label, **kw)


def main():
    data = {(n, s): load(n, s) for n in OPTIONS for s in SNAPS}
    zs = h5py.File(RUN_DIR.format(0), 'r')['Header/snapshot_redshifts'][:]

    fig, axes = plt.subplots(4, 4, figsize=(22, 21))
    smf_edges = np.arange(8.0, 12.01, 0.1)
    smf_x = centres(smf_edges)

    # ---------------- Row 1: stellar mass functions + observations ----------------
    smf_obs = pp._load_smf_grid_observations()
    markers = {}
    for i, s in enumerate(SNAPS):
        ax = axes[0, i]
        z = zs[s]
        for o in smf_obs:
            if abs(o['z'] - z) <= (0.1 if z < 0.2 else 0.3):
                m = markers.setdefault(o['label'], 'osD^vP*Xh'[len(markers) % 9])
                yerr = None if o['err_lo'] is None else [o['err_lo'], o['err_hi']]
                obs_points(ax, o['log_mass'], o['log_phi'], yerr=yerr, marker=m, label=f"{o['label']} (z={o['z']:.2f})")
        for n in OPTIONS:
            cnt = np.histogram(data[n, s]['lms'], smf_edges)[0]
            phi = np.where(cnt > 0, cnt / VOLUME / 0.1, np.nan)
            line(ax, smf_x, np.log10(phi), n)
        ax.set(xlim=(8.5, 12.0), ylim=(-5.2, -1.3), title=f'Stellar mass function, z = {z:.2f}',
               xlabel=r'$\log_{10} M_*\ [M_\odot]$', ylabel=r'$\log_{10}\Phi\ [\mathrm{Mpc^{-3}\,dex^{-1}}]$')
        ax.legend(loc='lower left', fontsize=7.5, frameon=False)

    # ---------------- Row 2: z = 0 relations + observations ----------------
    d0 = {n: data[n, 63] for n in OPTIONS}
    ms_edges = np.arange(8.5, 11.81, 0.2)

    # (a) quiescent fraction of all galaxies vs Baldry+12 red fraction
    ax = axes[1, 0]
    b12 = pp.load_baldry_blue_red()
    red = np.interp(b12['sf_mass'], b12['q_mass'], 10 ** b12['q_phi'])
    frac = red / (red + 10 ** b12['sf_phi'])
    obs_points(ax, b12['sf_mass'], frac, marker='o', label='Baldry+12 red fraction (colour)')
    for n in OPTIONS:
        d = d0[n]
        line(ax, centres(ms_edges), binned(d['lms'], (d['ssfr'] < SSFR_CUT).astype(float),
                                           d['StellarMass'] > 0, ms_edges, np.mean), n)
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
    for n in OPTIONS:
        d = d0[n]
        line(ax, centres(mv_edges), binned(d['lmv'], d['lms'], d['cen'] & (d['StellarMass'] > 0), mv_edges, np.median), n)
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
    for n in OPTIONS:
        d = d0[n]
        ok = (d['BulgeMass'] > 0) & (d['BlackHoleMass'] > 0)
        line(ax, centres(mb_edges), binned(np.log10(np.maximum(d['BulgeMass'], 1)),
                                           np.log10(np.maximum(d['BlackHoleMass'], 1)), ok, mb_edges, np.median), n)
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
    sfrd = {n: np.load(SFRD_FILE.format(n)) for n in OPTIONS}
    for n in OPTIONS:
        h = sfrd[n]
        ok = h['dens'] > 0
        line(ax, h['z'][ok], np.log10(h['dens'][ok]), n)
        okb = ok & (h['lo'] > 0)
        ax.fill_between(h['z'][okb], np.log10(h['lo'][okb]), np.log10(h['hi'][okb]),
                        color=OPTIONS[n][1], alpha=0.2, lw=0, zorder=3)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.set(xlim=(0.0, 7.5), ylim=(-3.0, -0.5), title='Cosmic SFR density history',
           xlabel='Redshift', ylabel=r'$\log_{10}\rho_\mathrm{SFR}\ [M_\odot\,\mathrm{yr^{-1}\,Mpc^{-3}}]$')
    ax.legend(loc='lower left', fontsize=7.5, frameon=False, ncol=2)

    # ---------------- Row 3: changes relative to RheatFix 0 ----------------
    for i, s in enumerate((63, 32)):
        ax = axes[2, 0 if i == 0 else 1]
        c0 = np.histogram(data[0, s]['lms'], smf_edges)[0]
        ok0 = c0 >= MIN_COUNT
        band = 0.434 * np.sqrt(2.0 / np.maximum(c0, 1))      # Poisson scatter of a ratio of two such counts
        ax.fill_between(smf_x[ok0], -band[ok0], band[ok0], color=OBS_COLOR, alpha=0.25, lw=0, label='Poisson 1-sigma')
        for n in list(OPTIONS)[1:]:
            c = np.histogram(data[n, s]['lms'], smf_edges)[0]
            ok = ok0 & (c >= MIN_COUNT)
            line(ax, smf_x[ok], np.log10(c[ok] / c0[ok]), n)
        ax.axhline(0, color='#222222', lw=1.2)
        ax.set(xlim=(8.5, 11.6), ylim=(-0.3, 0.3), title=f'SMF change vs v0.98, z = {zs[s]:.2f}',
               xlabel=r'$\log_{10} M_*\ [M_\odot]$', ylabel=r'$\Delta\log_{10}\Phi$')
        ax.legend(loc='lower left', fontsize=8, frameon=False)

    ax = axes[2, 2]
    h0 = sfrd[0]
    ok0 = (h0['dens'] > 0) & (h0['lo'] > 0)
    ax.fill_between(h0['z'][ok0], np.log10(h0['lo'][ok0] / h0['dens'][ok0]), np.log10(h0['hi'][ok0] / h0['dens'][ok0]),
                    color=OBS_COLOR, alpha=0.25, lw=0, label='bootstrap 16-84%')
    for n in list(OPTIONS)[1:]:
        h = sfrd[n]
        ok = ok0 & (h['dens'] > 0)
        line(ax, h['z'][ok], np.log10(h['dens'][ok] / h0['dens'][ok]), n)
    ax.axhline(0, color='#222222', lw=1.2)
    ax.set(xlim=(0.0, 7.5), ylim=(-0.3, 0.3), title='SFR density change vs v0.98',
           xlabel='Redshift', ylabel=r'$\Delta\log_{10}\rho_\mathrm{SFR}$')
    ax.legend(loc='lower left', fontsize=8, frameon=False)

    ax = axes[2, 3]
    r0 = binned(d0[0]['lmv'], d0[0]['lms'], d0[0]['cen'] & (d0[0]['StellarMass'] > 0), mv_edges, np.median)
    for n in list(OPTIONS)[1:]:
        d = d0[n]
        r = binned(d['lmv'], d['lms'], d['cen'] & (d['StellarMass'] > 0), mv_edges, np.median)
        line(ax, centres(mv_edges), r - r0, n)
    ax.axhline(0, color='#222222', lw=1.2)
    ax.axvline(np.log10(MSHOCK), color=OBS_COLOR, lw=0.8, ls='--')
    ax.set(xlim=(10.4, 14.0), ylim=(-0.3, 0.3), title='SHMR change vs v0.98, centrals, z = 0',
           xlabel=r'$\log_{10} M_\mathrm{vir}\ [M_\odot]$', ylabel=r'$\Delta$ median $\log_{10} M_*$')

    # ---------------- Row 4: mechanism in hot-regime centrals ----------------
    hot_edges = np.arange(11.0, 13.61, 0.1)
    near_edges = np.arange(11.0, 13.01, 0.1)
    d2 = {n: data[n, 32] for n in OPTIONS}
    for n in OPTIONS:
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

    handles = [Line2D([], [], color=c, ls=ls, lw=2.6 if n == 0 else 1.9, label=lab) for n, (lab, c, ls) in OPTIONS.items()]
    fig.legend(handles=handles, loc='upper center', ncol=2, fontsize=12, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle('StreamRheatFix on mini-Millennium: observations (rows 1-2), changes vs v0.98 (row 3), '
                 'hot-regime mechanism (row 4; dashed line $M_\\mathrm{shock}$)', y=0.975, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=90)
    print('wrote', OUT)


if __name__ == '__main__':
    main()

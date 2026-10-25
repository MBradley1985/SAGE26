"""Test the proposed loop behind FixingGas 3's excess star formation:

  faster SF -> more metals -> reheated gas enriches hot/CGM gas -> metal-line
  cooling rises (and the Bondi radio-mode rate, which scales with T/Lambda, drops)
  -> more cooling in 10^11.5-10^12 haloes -> more SF, fewer quenched galaxies.

Uses the real runs (output/millennium_fixinggas{0,3}) and the diagnostic runs in
output/diag_loop/, made with a temporary patch that freezes the metallicity used
in the cooling function at log Z = -2.7 for the cooling rate (freeze_cool), the
Bondi rate (freeze_agn) or both (freeze_both).
"""
import os, h5py, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from style import BLUE, ORANGE, AQUA, MAGENTA, VIOLET, YELLOW, OBS

ROOT = '/Users/mbradley/Documents/PhD/SAGE26/output/'
PLOTS = ROOT + 'fixing_gas_plots/'
H, Y, R = 0.73, 0.025, 0.43
VOL = (62.5 / H) ** 3
CASES = [('real', '', 'metallicity-dependent (as run)', 'k'),
         ('both', 'diag_loop/fg{}_freeze_both', 'Z frozen in cooling + Bondi', BLUE),
         ('cool', 'diag_loop/fg{}_freeze_cool', 'Z frozen in cooling only', ORANGE),
         ('agn', 'diag_loop/fg{}_freeze_agn', 'Z frozen in Bondi only', AQUA)]
F = ['Type', 'Mvir', 'StellarMass', 'IntraClusterStars', 'SfrDisk', 'SfrBulge', 'HotGas', 'MetalsHotGas', 'CGMgas',
     'MetalsCGMgas', 'ColdGas', 'MetalsColdGas', 'EjectedMass', 'MetalsEjectedMass', 'MetalsStellarMass',
     'MetalsIntraClusterStars', 'Cooling', 'Heating', 'CentralGalaxyIndex', 'Regime']
plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.color': '#e6e5e0', 'legend.frameon': False,
                     'legend.fontsize': 7.5, 'lines.linewidth': 2})


def path(case, fg):
    sub = dict((c[0], c[1]) for c in CASES)[case]
    return ROOT + (f'millennium_fixinggas{fg}' if case == 'real' else sub.format(fg)) + '/model_0.hdf5'


_cache = {}
def load(case, fg, snap=63):
    key = (case, fg, snap)
    if key not in _cache:
        with h5py.File(path(case, fg), 'r') as f:
            g = f[f'Snap_{snap}']
            d = {k: g[k][:].astype('f8') for k in F}
            z = np.array(f['Header/snapshot_redshifts'])
        for k in F:
            if k not in ('Type', 'Cooling', 'Heating', 'SfrDisk', 'SfrBulge', 'CentralGalaxyIndex', 'Regime'):
                d[k] *= 1e10 / H
        d['sfr'] = d['SfrDisk'] + d['SfrBulge']
        d['lmv'] = np.log10(d['Mvir'] + 1); d['lm'] = np.log10(d['StellarMass'] + 1)
        d['cen'] = d['Type'] == 0
        d['q'] = d['sfr'] / np.maximum(d['StellarMass'], 1) < 1e-11
        d['z'] = z[snap]
        _cache[key] = d
    return _cache[key]


E = np.arange(10.75, 13.76, 0.25); C = 0.5 * (E[1:] + E[:-1])


def by_halo(d, y, stat=np.median, sel=None):
    s = d['cen'] if sel is None else d['cen'] & sel
    return np.array([stat(y[s & (d['lmv'] >= a) & (d['lmv'] < b)]) if (s & (d['lmv'] >= a) & (d['lmv'] < b)).sum() > 15 else np.nan
                     for a, b in zip(E[:-1], E[1:])])


def lg(x):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.log10(x)


def group_sum(d, k):
    u, inv = np.unique(d['CentralGalaxyIndex'], return_inverse=True)
    tot = np.bincount(inv, weights=d[k], minlength=len(u))
    return tot[inv]


# ---------------------------------------------------------------- Figure L1: links 1 and 2
fig, axs = plt.subplots(2, 2, figsize=(11, 8.4))
for fg, col, ls in ((0, VIOLET, '-'), (3, MAGENTA, '-')):
    d = load('real', fg)
    prod = Y * (group_sum(d, 'StellarMass') + group_sum(d, 'IntraClusterStars')) / (1 - R)
    halo_z = group_sum(d, 'MetalsHotGas') + group_sum(d, 'MetalsCGMgas')
    lab = f'FixingGas {fg}'
    axs[0, 0].plot(C, by_halo(d, lg(prod)), color=col, label=lab + ': metals produced (group)')
    axs[0, 0].plot(C, by_halo(d, lg(halo_z)), color=col, ls='--', label=lab + ': metals in hot + CGM gas')
    axs[0, 1].plot(C, by_halo(d, halo_z / prod, stat=np.mean), color=col, label=lab + ': hot + CGM')
    axs[0, 1].plot(C, by_halo(d, (group_sum(d, 'MetalsEjectedMass')) / prod, stat=np.mean), color=col, ls=':', label=lab + ': ejected')
    axs[0, 1].plot(C, by_halo(d, (group_sum(d, 'MetalsColdGas')) / prod, stat=np.mean), color=col, ls='--', label=lab + ': cold ISM')
    zh = (d['MetalsHotGas'] + d['MetalsCGMgas']) / np.maximum(d['HotGas'] + d['CGMgas'], 1e-30)
    axs[1, 0].plot(C, by_halo(d, lg(zh / 0.02), sel=(d['HotGas'] + d['CGMgas']) > 0), color=col, label=lab + ': hot + CGM gas')
    axs[1, 0].plot(C, by_halo(d, lg(d['MetalsColdGas'] / d['ColdGas'] / 0.02), sel=d['ColdGas'] > 0), color=col, ls='--', label=lab + ': cold ISM (source of reheated gas)')
    # evolution of halo-gas metallicity in 10^11.75-10^12.25 haloes
    zz, vh = [], []
    for snap in range(30, 64, 3):
        dd = load('real', fg, snap); s = dd['cen'] & (np.abs(dd['lmv'] - 12.0) < 0.25) & ((dd['HotGas'] + dd['CGMgas']) > 0)
        zz.append(dd['z']); vh.append(np.median(lg((dd['MetalsHotGas'] + dd['MetalsCGMgas'])[s] / (dd['HotGas'] + dd['CGMgas'])[s] / 0.02)))
    axs[1, 1].plot(zz, vh, 'o-', color=col, label=lab)
axs[0, 0].set_ylabel(r'log metal mass [M$_\odot$]'); axs[0, 0].set_title('(a) link 1: metals produced, and held in halo gas', fontsize=10)
axs[0, 1].set_ylabel('fraction of produced metals'); axs[0, 1].set_title('(b) link 2: where the produced metals sit', fontsize=10); axs[0, 1].set_ylim(0, 0.8)
axs[1, 0].set_ylabel(r'log $Z/0.02$'); axs[1, 0].set_title('(c) link 2: halo-gas vs ISM metallicity', fontsize=10)
axs[1, 1].set_xlabel('z'); axs[1, 1].set_ylabel(r'log $Z_{\rm hot+CGM}/0.02$'); axs[1, 1].set_title(r'(d) halo-gas metallicity, haloes $10^{11.75-12.25}$ M$_\odot$', fontsize=10)
for a in (axs[0, 0], axs[0, 1], axs[1, 0]):
    a.set_xlabel(r'log $M_{\rm vir}$ [M$_\odot$] (centrals)')
for a in axs.flat:
    a.legend(loc='best')
fig.suptitle('Loop links 1–2: more metals, and they end up in the halo gas (Millennium, z = 0)', fontsize=12)
fig.tight_layout(); fig.savefig(PLOTS + 'loop_1_metals_into_halo.png', dpi=130); plt.close(fig)

# ---------------------------------------------------------------- Figure L2: links 3 and 4
fig, axs = plt.subplots(2, 2, figsize=(11, 8.4))
for case, _, lab, col in CASES[:2]:
    for fg, ls in ((0, '--'), (3, '-')):
        d = load(case, fg)
        axs[0, 0].plot(C, by_halo(d, d['Cooling'], sel=d['Cooling'] != 0), color=col, ls=ls, label=f'FG{fg}, {lab}')
        axs[0, 1].plot(C, by_halo(d, d['Heating'], sel=d['Heating'] != 0), color=col, ls=ls, label=f'FG{fg}, {lab}')
        cool = np.where(d['Cooling'] != 0, 10 ** np.where(d['Cooling'] != 0, d['Cooling'] - 40, 0), 0)
        heat = np.where(d['Heating'] != 0, 10 ** np.where(d['Heating'] != 0, d['Heating'] - 40, 0), 0)
        axs[1, 0].plot(C, by_halo(d, (cool > 0).astype(float), stat=np.mean), color=col, ls=ls, label=f'FG{fg}, {lab}')
        axs[1, 1].plot(C, by_halo(d, d['q'].astype(float), stat=np.mean), color=col, ls=ls, label=f'FG{fg}, {lab}')
axs[0, 0].set_ylabel('log cooling luminosity [erg/s] (cooling haloes)'); axs[0, 0].set_title('(a) link 3: cooling', fontsize=10)
axs[0, 1].set_ylabel('log radio-mode heating [erg/s] (heated haloes)'); axs[0, 1].set_title('(b) link 4: radio-mode heating', fontsize=10)
axs[1, 0].set_ylabel('fraction of centrals still cooling'); axs[1, 0].set_title('(c) haloes where cooling is not shut off', fontsize=10); axs[1, 0].set_ylim(0, 1.05)
axs[1, 1].set_ylabel('quiescent fraction (centrals)'); axs[1, 1].set_title('(d) outcome: quenching', fontsize=10); axs[1, 1].set_ylim(0, 1.05)
for a in axs.flat:
    a.set_xlabel(r'log $M_{\rm vir}$ [M$_\odot$] (centrals)'); a.legend(loc='best')
fig.suptitle('Loop links 3–4: cooling and heating with real vs frozen metallicity (dashed FG0, solid FG3)', fontsize=12)
fig.tight_layout(); fig.savefig(PLOTS + 'loop_2_cooling_heating.png', dpi=130); plt.close(fig)

# ---------------------------------------------------------------- Figure L3: causal test
fig, axs = plt.subplots(2, 2, figsize=(11, 8.4))
bw = 0.1; e = np.arange(8, 12.4, bw); c = e[:-1] + bw / 2
rows = []
for case, _, lab, col in CASES:
    d0, d3 = load(case, 0), load(case, 3)
    smhm = by_halo(d3, d3['lm']) - by_halo(d0, d0['lm'])
    dq = by_halo(d3, d3['q'].astype(float), stat=np.mean) - by_halo(d0, d0['q'].astype(float), stat=np.mean)
    axs[0, 0].plot(C, smhm, color=col, label=lab)
    axs[0, 1].plot(C, dq, color=col, label=lab)
    n0, _ = np.histogram(d0['lm'][d0['lm'] > 0], bins=e); n3, _ = np.histogram(d3['lm'][d3['lm'] > 0], bins=e)
    axs[1, 0].plot(c, lg(n3 / n0), color=col, label=lab)
    zz, dr = [], []
    for snap in range(20, 64, 2):
        a, b = load(case, 0, snap), load(case, 3, snap)
        zz.append(a['z']); dr.append(np.log10(np.sum(b['sfr']) / np.sum(a['sfr'])))
    axs[1, 1].plot(np.log10(1 + np.array(zz)), dr, color=col, label=lab)
    i12 = np.argmin(np.abs(C - 12.0)); i115 = np.argmin(np.abs(C - 11.5))
    rows.append((lab, smhm[i115], smhm[i12], dq[i115], dq[i12], dr[-1],
                 np.log10(np.sum(d3['StellarMass']) / np.sum(d0['StellarMass']))))
for a in axs.flat:
    a.axhline(0, color='#52514e', lw=0.8); a.legend(loc='best')
axs[0, 0].set_xlabel(r'log $M_{\rm vir}$ [M$_\odot$] (centrals)'); axs[0, 0].set_ylabel(r'$\Delta$ median log $M_*$ (FG3 − FG0)')
axs[0, 0].set_title('(a) stellar mass at fixed halo mass', fontsize=10)
axs[0, 1].set_xlabel(r'log $M_{\rm vir}$ [M$_\odot$] (centrals)'); axs[0, 1].set_ylabel(r'$\Delta$ quiescent fraction (FG3 − FG0)')
axs[0, 1].set_title('(b) quiescent fraction at fixed halo mass', fontsize=10)
axs[1, 0].set_xlabel(r'log $M_*$ [M$_\odot$]'); axs[1, 0].set_ylabel(r'log $\Phi_{\rm FG3}/\Phi_{\rm FG0}$'); axs[1, 0].set_xlim(9, 11.8); axs[1, 0].set_ylim(-0.3, 0.8)
axs[1, 0].set_title('(c) stellar mass function ratio, z = 0', fontsize=10)
axs[1, 1].set_xlabel('log (1 + z)'); axs[1, 1].set_ylabel(r'log $\rho_{\rm SFR,FG3}/\rho_{\rm SFR,FG0}$')
axs[1, 1].set_title('(d) cosmic SFR density ratio', fontsize=10)
fig.suptitle('Causal test: how much of the FixingGas 3 excess survives when metallicity no longer affects cooling/AGN?', fontsize=12)
fig.tight_layout(); fig.savefig(PLOTS + 'loop_3_causal_test.png', dpi=130); plt.close(fig)

print(f"{'case':34s} dlogM*@11.5 dlogM*@12 dfQ@11.5 dfQ@12 dlogSFRD(z=0) dlog rho*(z=0)")
for r in rows:
    print(f"{r[0]:34s} {r[1]:+10.2f} {r[2]:+9.2f} {r[3]:+8.2f} {r[4]:+6.2f} {r[5]:+12.2f} {r[6]:+13.2f}")
for case, _, lab, col in CASES:
    for fg in (0, 3):
        d = load(case, fg); s = d['cen'] & (np.abs(d['lmv'] - 12.0) < 0.25)
        zh = (d['MetalsHotGas'] + d['MetalsCGMgas']) / np.maximum(d['HotGas'] + d['CGMgas'], 1e-30)
        print(f"{lab:34s} FG{fg} @Mvir~1e12: hot Z/0.02 {np.median(lg(zh[s & (zh > 0)] / 0.02)):+.2f}  log cool {np.median(d['Cooling'][s & (d['Cooling'] != 0)]):.2f}  log heat {np.median(d['Heating'][s & (d['Heating'] != 0)]):.2f}  frac cooling {(d['Cooling'][s] != 0).mean():.2f}  fQ {d['q'][s].mean():.2f}")

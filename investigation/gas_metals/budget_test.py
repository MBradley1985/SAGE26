"""Gas budget of central galaxies, FixingGas 0 vs 3 (Millennium).

Rates in Msun/yr, per central, from the snapshot outputs:
  cooling          2 L_cool / V_vir^2  (Cooling = 0.5 Mdot_cool V_vir^2, as accumulated in model_cooling_heating.c)
  reincorporation  (V_vir/V_crit - 1) M_ejected / (R_vir/V_vir), V_crit = 354.53 km/s x ReIncorporationFactor
                   (model_reincorporation.c), evaluated on the output state
  star formation   SfrDisk + SfrBulge
  SN reheating     OutflowRate
Reservoirs: cold gas, hot + CGM gas, ejected gas.
"""
import h5py, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from style import BLUE, ORANGE, AQUA, MAGENTA, VIOLET, YELLOW, OBS

ROOT = '/Users/mbradley/Documents/PhD/SAGE26/output/'
PLOTS = ROOT + 'fixing_gas_plots/'
H = 0.73
VCRIT = 354.53 * 0.19
KM_PER_MPC, SEC_PER_YR, MSUN = 3.0857e19, 3.15576e7, 1.989e33
RUNS = ((0, VIOLET, 'FixingGas 0'), (3, MAGENTA, 'FixingGas 3'))
plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.color': '#e6e5e0', 'legend.frameon': False,
                     'legend.fontsize': 7.5, 'lines.linewidth': 2})
FLOWS = (('cool', 'cooling', BLUE), ('reinc', 'reincorporation', ORANGE), ('sfr', 'star formation', AQUA), ('reheat', 'SN reheating', YELLOW))

_c = {}
def load(fg, snap):
    if (fg, snap) not in _c:
        with h5py.File(ROOT + f'millennium_fixinggas{fg}/model_0.hdf5', 'r') as f:
            g = f[f'Snap_{snap}']
            k = ['Type', 'Mvir', 'Rvir', 'Vvir', 'StellarMass', 'ColdGas', 'HotGas', 'CGMgas', 'EjectedMass', 'SfrDisk', 'SfrBulge', 'Cooling', 'OutflowRate']
            d = {x: g[x][:].astype('f8') for x in k}
            z = np.array(f['Header/snapshot_redshifts'])[snap]
        for x in ('Mvir', 'StellarMass', 'ColdGas', 'HotGas', 'CGMgas', 'EjectedMass'):
            d[x] *= 1e10 / H
        v = d['Vvir'] * 1e5
        d['cool'] = np.where(d['Cooling'] != 0, 2 * 10 ** np.where(d['Cooling'] != 0, d['Cooling'], 0) / np.maximum(v, 1) ** 2 / MSUN * SEC_PER_YR, 0.0)
        tdyn_yr = (d['Rvir'] / H * KM_PER_MPC) / np.maximum(d['Vvir'], 1e-3) / SEC_PER_YR
        d['reinc'] = np.where(d['Vvir'] > VCRIT, (d['Vvir'] / VCRIT - 1) * d['EjectedMass'] / tdyn_yr, 0.0)
        d['sfr'] = d['SfrDisk'] + d['SfrBulge']
        d['reheat'] = d['OutflowRate']
        d['halo'] = d['HotGas'] + d['CGMgas']
        d['lmv'] = np.log10(d['Mvir'] + 1); d['cen'] = d['Type'] == 0; d['z'] = z
        _c[(fg, snap)] = d
    return _c[(fg, snap)]


E = np.arange(10.75, 13.51, 0.25); C = 0.5 * (E[1:] + E[:-1])
def mean_by_halo(d, y):
    return np.array([np.mean(y[d['cen'] & (d['lmv'] >= a) & (d['lmv'] < b)]) if (d['cen'] & (d['lmv'] >= a) & (d['lmv'] < b)).sum() > 15 else np.nan
                     for a, b in zip(E[:-1], E[1:])])
lg = lambda x: np.log10(np.maximum(x, 1e-30))

# ------------------------------------------------------------ Figure B1: flows and reservoirs vs halo mass
for snap, ztag in ((63, 'z0'), (41, 'z1'), (32, 'z2')):
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.4))
    for i, (key, lab, col) in enumerate(FLOWS):
        ax = axs.flat[i]
        for fg, rc, rl in RUNS:
            d = load(fg, snap)
            ax.plot(C, lg(mean_by_halo(d, d[key])), color=rc, ls='-' if fg else '--', label=rl)
        ax.set_title(f'({"abcd"[i]}) {lab} rate', fontsize=10); ax.set_ylabel(r'log mean rate [M$_\odot$ yr$^{-1}$]')
        ax.legend(loc='upper left')
    ax = axs.flat[4]
    for fg, rc, rl in RUNS:
        d = load(fg, snap)
        for key, ls, nm in (('ColdGas', '-', 'cold'), ('halo', '--', 'hot + CGM'), ('EjectedMass', ':', 'ejected')):
            ax.plot(C, lg(mean_by_halo(d, d[key])), color=rc, ls=ls, label=f'{rl}: {nm}')
    ax.set_title('(e) gas reservoirs', fontsize=10); ax.set_ylabel(r'log mean mass [M$_\odot$]'); ax.legend(loc='upper left', ncol=1)
    ax = axs.flat[5]
    d0, d3 = load(0, snap), load(3, snap)
    for key, lab, col in FLOWS:
        ax.plot(C, lg(mean_by_halo(d3, d3[key])) - lg(mean_by_halo(d0, d0[key])), color=col, label=lab)
    ax.axhline(0, color='#52514e', lw=0.8)
    ax.set_title('(f) change in each flow, FG3 − FG0', fontsize=10); ax.set_ylabel(r'$\Delta$ log mean rate'); ax.legend(loc='upper left')
    for a in axs.flat:
        a.set_xlabel(r'log $M_{\rm vir}$ [M$_\odot$] (centrals)')
    fig.suptitle(f'Gas budget of centrals at z = {d0["z"]:.2f} (dashed FixingGas 0, solid FixingGas 3)', fontsize=12)
    fig.tight_layout(); fig.savefig(PLOTS + f'budget_1_vs_halo_{ztag}.png', dpi=130); plt.close(fig)

# ------------------------------------------------------------ Figure B2: flows vs redshift at fixed halo mass
fig, axs = plt.subplots(2, 3, figsize=(15, 8.4))
snaps = list(range(22, 64, 2))
for col_i, (mlo, mhi) in enumerate(((11.25, 11.75), (11.75, 12.25), (12.25, 12.75))):
    ax = axs[0, col_i]; axr = axs[1, col_i]
    for key, lab, col in FLOWS:
        for fg, rc, rl in RUNS:
            zz, v = [], []
            for s in snaps:
                d = load(fg, s); m = d['cen'] & (d['lmv'] >= mlo) & (d['lmv'] < mhi)
                zz.append(d['z']); v.append(np.mean(d[key][m]) if m.sum() > 15 else np.nan)
            ax.plot(np.log10(1 + np.array(zz)), lg(np.array(v)), color=col, ls='-' if fg else '--', label=f'{lab}' if fg else None)
    for key, ls, nm in (('ColdGas', '-', 'cold'), ('halo', '--', 'hot + CGM'), ('EjectedMass', ':', 'ejected'), ('StellarMass', '-.', 'stars')):
        zz, r = [], []
        for s in snaps:
            a, b = load(0, s), load(3, s)
            ma = a['cen'] & (a['lmv'] >= mlo) & (a['lmv'] < mhi); mb = b['cen'] & (b['lmv'] >= mlo) & (b['lmv'] < mhi)
            zz.append(a['z']); r.append(lg(np.mean(b[key][mb])) - lg(np.mean(a[key][ma])) if ma.sum() > 15 else np.nan)
        axr.plot(np.log10(1 + np.array(zz)), r, ls=ls, color='k', label=nm)
    ax.set_title(rf'flows, haloes $10^{{{mlo}}}$–$10^{{{mhi}}}$ M$_\odot$' + '\n(dashed FG0, solid FG3)', fontsize=10)
    ax.set_xlabel('log (1 + z)'); ax.set_ylabel(r'log mean rate [M$_\odot$ yr$^{-1}$]'); ax.legend(loc='lower left')
    axr.axhline(0, color='#52514e', lw=0.8)
    axr.set_title(rf'reservoirs, FG3 − FG0, haloes $10^{{{mlo}}}$–$10^{{{mhi}}}$', fontsize=10)
    axr.set_xlabel('log (1 + z)'); axr.set_ylabel(r'$\Delta$ log mean mass'); axr.legend(loc='upper left')
fig.suptitle('Gas budget of centrals through time at fixed halo mass (Millennium)', fontsize=12)
fig.tight_layout(); fig.savefig(PLOTS + 'budget_2_vs_redshift.png', dpi=130); plt.close(fig)

# ------------------------------------------------------------ numbers
for snap in (63, 41, 32):
    d0, d3 = load(0, snap), load(3, snap)
    print(f'z = {d0["z"]:.2f}')
    for lo in (11.5, 12.0):
        i = np.argmin(np.abs(C - lo))
        row = ' '.join(f"{k}: {lg(mean_by_halo(d0, d0[k]))[i]:.2f}->{lg(mean_by_halo(d3, d3[k]))[i]:.2f}" for k in ('cool', 'reinc', 'sfr', 'reheat', 'ColdGas', 'halo', 'EjectedMass'))
        print(f'  logMvir {lo}: {row}')

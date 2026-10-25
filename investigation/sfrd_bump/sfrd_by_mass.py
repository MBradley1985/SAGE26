"""Cosmic SFR density split by the stellar mass of the galaxies forming the stars,
FixingGas 0 vs the FixingGas 3 PSO best fit (Millennium)."""
import h5py, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = '/Users/mbradley/Documents/PhD/SAGE26/'
H = 0.73; VOL = (62.5 / H) ** 3
E = [0, 9.5, 10, 10.5, 11, 13]
LAB = [r'$<10^{9.5}$', r'$10^{9.5-10}$', r'$10^{10-10.5}$', r'$10^{10.5-11}$', r'$>10^{11}$']
COL = ['#2a78d6', '#1baf7a', '#eda100', '#eb6834', '#e34948']
fig, axs = plt.subplots(1, 3, figsize=(16, 5))
store = {}
for ax, (d, title) in zip(axs[:2], (('output/millennium_fixinggas0', 'FixingGas 0'), ('output/millennium_fixinggas3_best', 'FixingGas 3, PSO best fit'))):
    f = h5py.File(ROOT + d + '/model_0.hdf5', 'r'); z = np.array(f['Header/snapshot_redshifts']); snaps = range(25, 64)
    part = np.zeros((len(E) - 1, len(snaps)))
    for j, s in enumerate(snaps):
        g = f[f'Snap_{s}']; lm = np.log10(g['StellarMass'][:] * 1e10 / H + 1); sfr = g['SfrDisk'][:] + g['SfrBulge'][:]
        part[:, j] = [sfr[(lm >= a) & (lm < b)].sum() / VOL for a, b in zip(E[:-1], E[1:])]
    zz = np.array([z[s] for s in snaps])
    for i in range(len(E) - 1):
        ax.plot(zz, np.log10(np.maximum(part[i], 1e-6)), color=COL[i], label=LAB[i])
    ax.plot(zz, np.log10(part.sum(0)), 'k', lw=2.5, label='total')
    ax.set_title(title, fontsize=11); ax.set_xlim(0, 2.5); ax.set_ylim(-3.5, -0.9); ax.set_xlabel('z')
    ax.set_ylabel(r'log $\rho_{\rm SFR}$ [M$_\odot$ yr$^{-1}$ Mpc$^{-3}$]'); ax.grid(alpha=0.3); ax.legend(fontsize=8, title=r'$M_*$ [M$_\odot$]', frameon=False)
    store[title] = (zz, part)
ax = axs[2]
for i in range(len(E) - 1):
    for title, ls in (('FixingGas 0', '--'), ('FixingGas 3, PSO best fit', '-')):
        zz, part = store[title]; ax.plot(zz, part[i] / part.sum(0), color=COL[i], ls=ls, label=LAB[i] if ls == '-' else None)
ax.plot([], [], 'k--', label='FixingGas 0'); ax.plot([], [], 'k-', label='FixingGas 3 best fit')
ax.set_xlim(0, 2.5); ax.set_ylim(0, 0.6); ax.set_xlabel('z'); ax.set_ylabel('share of cosmic SFR'); ax.grid(alpha=0.3)
ax.set_title('share of star formation by stellar mass', fontsize=11); ax.legend(fontsize=8, frameon=False)
fig.suptitle('Where the late-time star formation happens (Millennium)', fontsize=12)
fig.tight_layout(); fig.savefig(ROOT + 'output/fixing_gas_plots/sfrd_3_by_stellar_mass.png', dpi=130)

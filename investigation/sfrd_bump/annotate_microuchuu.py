"""Annotated microUchuu cosmic SFR density: the late-time feature as I read it."""
import glob, h5py, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = '/Users/mbradley/Documents/PhD/SAGE26/'
fs = sorted(glob.glob(ROOT + 'output/microuchuu/model_*.hdf5'))
with h5py.File(fs[0], 'r') as f:
    z = np.array(f['Header/snapshot_redshifts']); h = f['Header/Simulation'].attrs['hubble_h']
vol = (100.0 / h) ** 3
snaps = range(10, 50)
sfrd = np.array([sum((lambda g: g['SfrDisk'][:].sum() + g['SfrBulge'][:].sum())(h5py.File(fn, 'r')[f'Snap_{s}']) for fn in fs) / vol for s in snaps])
zz = np.array([z[s] for s in snaps]); y = np.log10(sfrd)
# the steep decline just before z ~ 1: straight-line fit in z over 0.85 < z < 1.3, extrapolated to z = 0
fit = (zz > 0.85) & (zz < 1.3)
p = np.polyfit(zz[fit], y[fit], 1)
ze = np.linspace(0, 1.3, 100)
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(zz, y, color='#2a78d6', lw=3, label='microUchuu (as in History-SFR-density.png)')
ax.plot(ze, np.polyval(p, ze), 'k--', lw=1.5, label='steep decline over 0.85 < z < 1.3, extrapolated')
low = zz <= 0.9
ax.fill_between(zz[low], np.polyval(p, zz[low]), y[low], color='#eb6834', alpha=0.3, label='excess I think you mean')
ax.annotate(f'extra ≈ {y[-1] - np.polyval(p, 0):.2f} dex at z = 0', xy=(0.05, (y[-1] + np.polyval(p, 0)) / 2), xytext=(0.35, -1.15),
            arrowprops=dict(arrowstyle='->'), fontsize=10)
D = np.loadtxt(ROOT + 'data/sfrd/Driver18_sfr.dat'); zc = 0.5 * (D[:, 0] + D[:, 1])
ax.errorbar(zc, D[:, 2], yerr=np.hypot(D[:, 3], D[:, 5]), fmt='s', color='k', ms=4, capsize=2, lw=1, label='Driver+18')
ax.set_xlim(0, 3); ax.set_ylim(-2.2, -0.8); ax.set_xlabel('redshift'); ax.set_ylabel(r'log SFR density [M$_\odot$ yr$^{-1}$ Mpc$^{-3}$]')
ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=9, loc='lower right')
ax.set_title('microUchuu: the late-time flattening (shaded) vs the earlier decline')
fig.tight_layout(); fig.savefig(ROOT + 'output/fixing_gas_plots/sfrd_4_microuchuu_annotated.png', dpi=130)
print('z=0 model', y[-1].round(3), 'extrapolated', np.polyval(p, 0).round(3), 'slope per unit z', p[0].round(3))

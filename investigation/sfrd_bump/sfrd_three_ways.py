"""Cosmic SFR density of a SAGE26 Millennium run, measured three independent ways:
  A  sum of the snapshot SFR outputs (SfrDisk + SfrBulge, averaged over each snapshot interval)
  B  z = 0 star-formation histories (SFHMassDisk + SFHMassBulge) binned by formation snapshot,
     divided by (1 - R) and the bin's physical duration
  C  time derivative of the stellar mass density (+ ICS), divided by (1 - R)
"""
import sys, h5py, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = '/Users/mbradley/Documents/PhD/SAGE26/'
R = 0.43
runs = [(ROOT + 'output/millennium_fixinggas0/model_0.hdf5', 'FixingGas 0', '#6250d6'),
        (ROOT + 'output/millennium_fixinggas3/model_0.hdf5', 'FixingGas 3 (default feedback)', '#eb6834'),
        (ROOT + 'output/millennium_fixinggas3_best/model_0.hdf5', 'FixingGas 3, PSO best fit', '#1baf7a')]

def age_gyr(z, h, om):   # cosmic time, flat LCDM
    a = 1 / (1 + z); ol = 1 - om
    return 2 / (3 * 100 * h * np.sqrt(ol)) * np.arcsinh(np.sqrt(ol / om) * a ** 1.5) * 977.8

fig, axs = plt.subplots(1, 3, figsize=(16, 5))
out = {}
for path, lab, col in runs:
    with h5py.File(path, 'r') as f:
        h = f['Header/Simulation'].attrs['hubble_h']; om = f['Header/Simulation'].attrs['omega_matter']
        vol = (f['Header/Simulation'].attrs['box_size'] / h) ** 3
        z = np.array(f['Header/snapshot_redshifts']); t = age_gyr(z, h, om)
        snaps = [s for s in range(64) if f'Snap_{s}' in f]
        rA = np.array([np.sum(f[f'Snap_{s}/SfrDisk'][:] + f[f'Snap_{s}/SfrBulge'][:]) / vol for s in snaps])
        mst = np.array([(np.sum(f[f'Snap_{s}/StellarMass'][:].astype('f8')) + np.sum(f[f'Snap_{s}/IntraClusterStars'][:].astype('f8'))) * 1e10 / h / vol for s in snaps])
        sfh = (f['Snap_63/SFHMassDisk'][:].astype('f8') + f['Snap_63/SFHMassBulge'][:].astype('f8')).sum(0) * 1e10 / h / vol
    zs = z[snaps]
    # A: snapshot s holds the SFR averaged over (t[s-1], t[s]); plot at the interval midpoint
    tA = 0.5 * (t[np.array(snaps)] + t[np.array(snaps) - 1])
    # B: bin k = stars formed between snapshot k and k+1
    k = np.arange(63); dt = (t[k + 1] - t[k]) * 1e9
    rB = sfh[:63] / (1 - R) / dt; tB = 0.5 * (t[k] + t[k + 1])
    # C
    rC = np.diff(mst) / np.diff(t[np.array(snaps)] * 1e9) / (1 - R); tC = 0.5 * (t[np.array(snaps)][1:] + t[np.array(snaps)][:-1])
    for ax, (tt, rr, nm) in zip(axs, ((tA, rA, 'A: snapshot SFR outputs'), (tB, rB, 'B: z = 0 star-formation histories'), (tC, rC, 'C: d(stellar mass density)/dt'))):
        ax.plot(tt, np.log10(np.maximum(rr, 1e-6)), 'o-', ms=3, color=col, label=lab)
        ax.set_title(nm, fontsize=10)
    out[lab] = (tA, rA, tB, rB, z, t, snaps)
D = np.loadtxt(ROOT + 'data/sfrd/Driver18_sfr.dat')
zc = 0.5 * (D[:, 0] + D[:, 1])
for ax in axs:
    ax.errorbar(age_gyr(zc, 0.73, 0.25), D[:, 2], yerr=np.hypot(D[:, 3], D[:, 5]), fmt='s', color='k', ms=4, capsize=2, lw=1, label='Driver+18 (h = 0.7)')
    ax.set_xlabel('cosmic time [Gyr]'); ax.set_ylabel(r'log $\rho_{\rm SFR}$ [M$_\odot$ yr$^{-1}$ Mpc$^{-3}$]')
    ax.set_xlim(2, 14); ax.set_ylim(-2.3, -0.8); ax.grid(alpha=0.3)
    for zz in (0.5, 1.0, 2.0):
        ax.axvline(age_gyr(zz, 0.73, 0.25), color='grey', lw=0.6, ls=':'); ax.text(age_gyr(zz, 0.73, 0.25), -0.85, f'z={zz}', fontsize=7)
axs[0].legend(fontsize=8, loc='lower left')
fig.suptitle('Cosmic SFR density measured three ways (Millennium)', fontsize=12)
fig.tight_layout(); fig.savefig(ROOT + 'output/fixing_gas_plots/sfrd_1_three_ways.png', dpi=130)
lab = 'FixingGas 3, PSO best fit'; tA, rA, tB, rB, z, t, snaps = out[lab]
print('best fit, method A, z>=0 from z=1.5:')
for s, tt, r in zip(snaps, tA, rA):
    if z[s] < 1.6: print(f'  snap {s} z={z[s]:.2f} t={tt:.2f} logSFRD={np.log10(r):.3f}')

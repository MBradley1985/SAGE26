"""Which part of FixingGas fix 1 flattens the late-time SFR density?
Old law at higher efficiency (same z=0 depletion time) vs Tacconi+18 full / without the
main-sequence-offset term (C = 0) / without the redshift term (B = 0). Runs in output/diag_tdep
were made with a temporary, uncommitted patch reading the coefficients from the environment."""
import h5py, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = '/Users/mbradley/Documents/PhD/SAGE26/'
H = 0.73; VOL = (62.5 / H) ** 3
RUNS = [('output/millennium_fixinggas0', 'FG0: old law (eff 0.05)', '#6250d6', '--'),
        ('output/diag_tdep/fg0_eff0.14', 'old law, eff 0.14 (z=0 t_dep ≈ Tacconi)', '#6250d6', ':'),
        ('output/millennium_fixinggas1', 'FG1: Tacconi, full (with ΔMS term)', '#eb6834', '-'),
        ('output/diag_tdep/fg1_noC', 'FG1: Tacconi without ΔMS term (C = 0)', '#1baf7a', '-'),
        ('output/diag_tdep/fg1_noB', 'FG1: Tacconi without z term (B = 0)', '#eda100', '-')]
def speagle(lm, z):
    l = np.log10(1 + z); tc = 10 ** (1.143 - 1.026 * l - 0.599 * l ** 2 + 0.528 * l ** 3)
    return 10 ** ((-0.16 - 0.026 * tc) * (lm + 0.025) - (6.51 - 0.11 * tc) + 9) / 1e9
fig, axs = plt.subplots(2, 2, figsize=(11.5, 8.6))
E = [0, 9, 9.5, 10, 10.5, 11, 13]; C = [8.75, 9.25, 9.75, 10.25, 10.75, 11.25]
for d, lab, col, ls in RUNS:
    f = h5py.File(ROOT + d + '/model_0.hdf5', 'r'); z = np.array(f['Header/snapshot_redshifts']); snaps = range(25, 64)
    y = [np.log10((f[f'Snap_{s}/SfrDisk'][:].sum() + f[f'Snap_{s}/SfrBulge'][:].sum()) / VOL) for s in snaps]
    axs[0, 0].plot([z[s] for s in snaps], y, color=col, ls=ls, label=lab)
    r = {}
    for zt in (1.0, 0.0):
        g = f[f'Snap_{int(np.argmin(abs(z - zt)))}']; lm = np.log10(g['StellarMass'][:] * 1e10 / H + 1); sfr = g['SfrDisk'][:] + g['SfrBulge'][:]
        r[zt] = np.array([sfr[(lm >= a) & (lm < b)].sum() for a, b in zip(E[:-1], E[1:])])
    axs[0, 1].plot(C, np.log10(r[1.0] / r[0.0]), 'o-', color=col, ls=ls, label=lab)
    g = f['Snap_63']; ms = g['StellarMass'][:] * 1e10 / H; lm = np.log10(ms + 1); sfr = g['SfrDisk'][:] + g['SfrBulge'][:]
    sf = sfr / np.maximum(ms, 1) > 1e-11; e = np.arange(9, 11.61, 0.2); c = 0.5 * (e[1:] + e[:-1])
    axs[1, 0].plot(c, [np.median(np.log10(sfr[sf & (lm >= a) & (lm < b)] / ms[sf & (lm >= a) & (lm < b)])) if (sf & (lm >= a) & (lm < b)).sum() > 10 else np.nan
                       for a, b in zip(e[:-1], e[1:])], color=col, ls=ls, label=lab)
    q = ~sf
    axs[1, 1].plot(c, [q[(lm >= a) & (lm < b)].mean() if ((lm >= a) & (lm < b)).sum() > 10 else np.nan for a, b in zip(e[:-1], e[1:])], color=col, ls=ls, label=lab)
D = np.loadtxt(ROOT + 'data/sfrd/Driver18_sfr.dat'); zc = 0.5 * (D[:, 0] + D[:, 1])
axs[0, 0].errorbar(zc, D[:, 2], yerr=np.hypot(D[:, 3], D[:, 5]), fmt='s', color='k', ms=4, capsize=2, lw=1, label='Driver+18')
axs[0, 0].set_xlim(0, 2.5); axs[0, 0].set_xlabel('z'); axs[0, 0].set_ylabel(r'log $\rho_{\rm SFR}$ [M$_\odot$ yr$^{-1}$ Mpc$^{-3}$]'); axs[0, 0].set_title('(a) cosmic SFR density', fontsize=10)
axs[0, 1].axhline(0, color='grey', lw=0.8); axs[0, 1].set_xlabel(r'log $M_*$ [M$_\odot$]'); axs[0, 1].set_ylabel(r'log SFRD(z=1) / SFRD(z=0) in mass bin')
axs[0, 1].set_title('(b) late-time decline by stellar mass', fontsize=10)
lm_ = np.linspace(9, 11.5, 40)
axs[1, 0].plot(lm_, np.log10(speagle(lm_, 0.0)), 'k:', lw=1.5, label='Speagle+14 MS, z = 0')
axs[1, 0].set_xlabel(r'log $M_*$ [M$_\odot$]'); axs[1, 0].set_ylabel('median log sSFR [1/yr], star-forming'); axs[1, 0].set_ylim(-11, -9.3)
axs[1, 0].set_title('(c) star-forming sequence, z = 0', fontsize=10)
axs[1, 1].set_xlabel(r'log $M_*$ [M$_\odot$]'); axs[1, 1].set_ylabel('quiescent fraction'); axs[1, 1].set_ylim(0, 1); axs[1, 1].set_title('(d) quiescent fraction, z = 0', fontsize=10)
for a in axs.flat:
    a.grid(alpha=0.3); a.legend(fontsize=7, frameon=False, loc='best')
fig.suptitle('The late-time flattening comes from the ΔMS term of the Tacconi+18 depletion time (Millennium, flag 1, default feedback)', fontsize=11)
fig.tight_layout(); fig.savefig(ROOT + 'output/fixing_gas_plots/sfrd_5_tdep_terms.png', dpi=130)

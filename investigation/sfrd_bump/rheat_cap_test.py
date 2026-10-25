"""Does the Rvir cap on r_heat (hot-halo AGN path) stop haloes with rcool > Rvir from
quenching, and is that what flattens the late-time cosmic SFR density under FixingGas 3?
Compares the PSO best fit with and without the cap (temporary diagnostic patch)."""
import h5py, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = '/Users/mbradley/Documents/PhD/SAGE26/'
H = 0.73; VOL = (62.5 / H) ** 3
RUNS = [('output/millennium_fixinggas0', 'FixingGas 0', '#6250d6', '--'),
        ('output/millennium_fixinggas3_best', 'FixingGas 3 best fit (r_heat capped at Rvir)', '#eb6834', '-'),
        ('output/diag_rheat/best_nocap', 'FixingGas 3 best fit, cap lifted', '#1baf7a', '-')]
fig, axs = plt.subplots(2, 2, figsize=(11, 8.4))
for d, lab, col, ls in RUNS:
    f = h5py.File(ROOT + d + '/model_0.hdf5', 'r'); z = np.array(f['Header/snapshot_redshifts'])
    snaps = range(20, 64)
    sfrd = [np.log10((f[f'Snap_{s}/SfrDisk'][:].sum() + f[f'Snap_{s}/SfrBulge'][:].sum()) / VOL) for s in snaps]
    axs[0, 0].plot([z[s] for s in snaps], sfrd, color=col, ls=ls, label=lab)
    shut = []
    for s in snaps:
        g = f[f'Snap_{s}']; T = g['Type'][:]; mv = np.log10(g['Mvir'][:] * 1e10 / H + 1)
        m = (T == 0) & (mv >= 12) & (mv < 12.5); rc = g['RcoolToRvir'][:]; rh = g['r_heat'][:] / g['Rvir'][:]
        shut.append(np.mean(((rh >= 0.99 * rc) & (rc > 0))[m]) if m.sum() > 20 else np.nan)
    axs[0, 1].plot([z[s] for s in snaps], shut, color=col, ls=ls, label=lab)
    g = f['Snap_63']; ms = g['StellarMass'][:] * 1e10 / H; lm = np.log10(ms + 1)
    q = (g['SfrDisk'][:] + g['SfrBulge'][:]) / np.maximum(ms, 1) < 1e-11
    e = np.arange(8.5, 12.01, 0.2); c = 0.5 * (e[1:] + e[:-1])
    axs[1, 0].plot(c, [q[(lm >= a) & (lm < b)].mean() if ((lm >= a) & (lm < b)).sum() > 10 else np.nan for a, b in zip(e[:-1], e[1:])], color=col, ls=ls, label=lab)
    eb = np.arange(8, 12.4, 0.1); n, _ = np.histogram(lm[lm > 0], bins=eb)
    axs[1, 1].plot(eb[:-1] + 0.05, np.log10(np.maximum(n, 0.1) / VOL / 0.1), color=col, ls=ls, label=lab)
    i1, i0 = int(np.argmin(abs(z - 1))), 63
    print(f"{lab:46s} logSFRD z=1 {sfrd[list(snaps).index(i1)]:.2f}  z=0 {sfrd[-1]:.2f}  drop {sfrd[list(snaps).index(i1)] - sfrd[-1]:.2f}   "
          f"rho* z=0 {np.log10(ms.sum() / VOL):.3f}   fQ(10^10.5) {q[(lm >= 10.4) & (lm < 10.6)].mean():.2f}  fQ(10^11) {q[(lm >= 10.9) & (lm < 11.1)].mean():.2f}  shut-off 10^12 haloes z=0 {shut[-1]:.2f}")
D = np.loadtxt(ROOT + 'data/sfrd/Driver18_sfr.dat'); zc = 0.5 * (D[:, 0] + D[:, 1])
axs[0, 0].errorbar(zc, D[:, 2], yerr=np.hypot(D[:, 3], D[:, 5]), fmt='s', color='k', ms=4, capsize=2, lw=1, label='Driver+18')
axs[0, 0].set_xlim(0, 3); axs[0, 0].set_xlabel('z'); axs[0, 0].set_ylabel(r'log $\rho_{\rm SFR}$ [M$_\odot$ yr$^{-1}$ Mpc$^{-3}$]'); axs[0, 0].set_title('(a) cosmic SFR density', fontsize=10)
axs[0, 1].set_xlim(0, 3); axs[0, 1].set_xlabel('z'); axs[0, 1].set_ylabel('fraction with cooling shut off'); axs[0, 1].set_ylim(0, 1)
axs[0, 1].set_title(r'(b) centrals in $10^{12}$–$10^{12.5}$ M$_\odot$ haloes', fontsize=10)
axs[1, 0].set_xlabel(r'log $M_*$ [M$_\odot$]'); axs[1, 0].set_ylabel('quiescent fraction'); axs[1, 0].set_ylim(0, 1); axs[1, 0].set_title('(c) quiescent fraction, z = 0', fontsize=10)
gm = np.genfromtxt(ROOT + 'data/smf/gama_smf_morph.ecsv', comments='#', skip_header=1)
axs[1, 1].plot(gm[:, 0], np.log10(np.nansum(10 ** gm[:, [1, 3, 5, 7]], axis=1)), 'o', color='k', ms=4, label='GAMA morphological SMF, summed')
axs[1, 1].set_xlim(8.5, 12.2); axs[1, 1].set_ylim(-5.5, -1); axs[1, 1].set_xlabel(r'log $M_*$ [M$_\odot$]'); axs[1, 1].set_ylabel(r'log $\Phi$ [Mpc$^{-3}$ dex$^{-1}$]')
axs[1, 1].set_title('(d) stellar mass function, z = 0', fontsize=10)
for a in axs.flat:
    a.grid(alpha=0.3); a.legend(fontsize=7.5, loc='best', frameon=False)
fig.suptitle('Late-time SFR density: the Rvir cap on the AGN heating radius (Millennium)', fontsize=12)
fig.tight_layout(); fig.savefig(ROOT + 'output/fixing_gas_plots/sfrd_2_rheat_cap_test.png', dpi=130)

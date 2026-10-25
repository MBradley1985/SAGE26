"""Figures 1-2: where SAGE26 (microUchuu, Snap_48 z = 0.022, plus z = 1.2, 2.5) sits
against the observations stored in SAGE26/data. Diagnosis only; no model changes."""
import numpy as np
import matplotlib.pyplot as plt
from common import load, binned
from style import *

g = load('Snap_48')
x = g['logms']; sf = g['sf']; all_ = np.ones_like(sf)
sel9 = x > 8.9

# ---------------------------------------------------------------- Figure 1: gas
fig, axs = plt.subplots(2, 3, figsize=(13, 7.6))
ax = axs[0, 0]
band(ax, x, lg(g['mh2'] / g['ms']), sel9, BLUE, 'SAGE26, all galaxies')
band(ax, x, lg(1.36 * g['mh2'] / g['ms']), sel9, VIOLET, 'SAGE26, all, ×1.36 (with He)')
band(ax, x, lg(g['mh2'] / g['ms']), sel9 & sf, ORANGE, 'SAGE26, star-forming')
obs_lagos(ax, 'MolecularGasRatio_NonDetEQUpperLimits.dat', 'xCOLD GASS (Saintonge+17), all; non-det = limit')
ax.set_ylabel(r'log $M_{\rm H_2}/M_*$'); ax.set_ylim(-3, 0.5); ax.legend(loc='lower left')
ax.set_title('(a) molecular gas')

ax = axs[0, 1]
band(ax, x, lg(g['mhi'] / g['ms']), sel9, BLUE, 'SAGE26, all galaxies')
band(ax, x, lg(g['mhi'] / g['ms']), sel9 & sf, ORANGE, 'SAGE26, star-forming')
obs_lagos(ax, 'HIGasRatio_NonDetEQUpperLimits.dat', 'xGASS (Catinella+18), all; non-det = limit')
ax.set_ylabel(r'log $M_{\rm HI}/M_*$'); ax.set_ylim(-3, 1); ax.legend(loc='lower left')
ax.set_title('(b) atomic gas')

ax = axs[0, 2]
band(ax, x, lg(g['mh2'] / g['mhi']), sel9 & sf, ORANGE, 'SAGE26, star-forming')
band(ax, x, lg(g['mh2'] / g['mhi']), sel9, BLUE, 'SAGE26, all galaxies')
ax.set_ylabel(r'log $M_{\rm H_2}/M_{\rm HI}$'); ax.set_ylim(-2, 1); ax.legend(loc='lower right')
ax.set_title('(c) molecular-to-atomic ratio')

ax = axs[1, 0]
band(ax, x, lg(g['mh2'] / g['sfr']), sel9 & sf, ORANGE, r'SAGE26 $t_{\rm dep}(\rm H_2)$, H only')
band(ax, x, lg((g['mh2'] + g['mhi']) / g['sfr']), sel9 & sf, AQUA, r'SAGE26 $t_{\rm dep}(\rm H_2+HI)$')
ax.set_ylabel(r'log $t_{\rm dep}$ [yr], star-forming'); ax.set_ylim(8.5, 11); ax.legend(loc='upper left')
ax.set_title('(d) depletion times')

ax = axs[1, 1]
band(ax, x, lg(1.678 * g['rs_pc'] / 1e3), sel9 & sf, ORANGE, r'SAGE26 disk $r_e = 1.678\,r_s$, star-forming')
L = np.loadtxt(DATA + 'SizesAndAM/rdisk_L16.dat')
ax.fill(np.log10(L[:, 0]), np.log10(L[:, 1]), color=OBS, alpha=0.12, lw=0, label='Lange+16 disks (90% region)')
ax.set_ylabel(r'log $r_e$ [kpc]'); ax.set_ylim(-0.4, 1.5); ax.legend(loc='upper left')
ax.set_title('(e) disk sizes')

ax = axs[1, 2]
band(ax, x, lg(g['ssfr']), sel9 & sf, ORANGE, 'SAGE26, star-forming')
lm_ = np.linspace(9, 11.5, 50)
from fig_fixes import speagle_ms_ssfr
ax.plot(lm_, np.log10(speagle_ms_ssfr(lm_, 0.0224)), color=OBS, lw=1.5, ls=':', label='Speagle+14 MS (as in Tacconi+18), z = 0.02')
ax.plot(lm_, -0.344 * (lm_ - 9) - 9.822, color=OBS, lw=1.5, ls='--', label='xGASS SFMS (Catinella+18 Eq. 2)')
ax.set_ylabel(r'log sSFR [yr$^{-1}$]'); ax.set_ylim(-11.2, -9); ax.legend(loc='lower left')
ax.set_title('(f) star-forming sequence')
for ax in axs.flat:
    ax.set_xlabel(r'log $M_*$ [M$_\odot$]'); ax.set_xlim(8.9, 11.9)
fig.suptitle('Issue 2 diagnosis: SAGE26 gas content at z = 0.02 (microUchuu Snap_48)', fontsize=12)
fig.tight_layout(); fig.savefig('fig1_gas_diagnosis.png', dpi=130); plt.close(fig)

# ---------------------------------------------------------------- Figure 2: metals
fig, axs = plt.subplots(2, 2, figsize=(10.5, 8))
q = ~sf
ax = axs[0, 0]
band(ax, x, lg(g['zstar'] / g['ms'] / ZSUN_G05), sel9, BLUE, 'SAGE26, all (mass-weighted)')
band(ax, x, lg(g['zstar'] / g['ms'] / ZSUN_G05), sel9 & q, ORANGE, 'SAGE26, quiescent')
ax.plot(G05[:, 0], G05[:, 1], 'o', color=OBS, ms=5, label='Gallazzi+05 median (light-weighted)')
ax.fill_between(G05[:, 0], G05[:, 2], G05[:, 3], color=OBS, alpha=0.1, lw=0, label='Gallazzi+05 16–84%')
ax.set_ylabel(r'log $Z_*/Z_\odot$ ($Z_\odot = 0.02$)'); ax.set_ylim(-1.3, 0.6); ax.legend(loc='lower right')
ax.set_title('(a) stellar mass–metallicity')

ax = axs[0, 1]
# Asplund+09 solar: Z = 0.0134, 12 + log(O/H) = 8.69; assumes a solar abundance pattern
oh = 8.69 + lg(g['zcold'] / g['mcold'] / 0.0134)
band(ax, x, oh, sel9 & sf, ORANGE, 'SAGE26 cold gas, star-forming')
T = np.loadtxt(DATA + 'metallicity/Tremonti04.dat'); C = np.loadtxt(DATA + 'metallicity/Curti2020.dat')
ax.plot(T[:, 0] - 0.04, T[:, 1], color=OBS, lw=1.5, label='Tremonti+04 (Kroupa→Chabrier −0.04)')
ax.fill_between(T[:, 0] - 0.04, T[:, 2], T[:, 3], color=OBS, alpha=0.08, lw=0)
ax.plot(C[:, 0] - 0.04, C[:, 1], color=OBS, lw=1.5, ls='--', label='Curti+20 (Kroupa→Chabrier −0.04)')
ax.set_ylabel(r'12 + log(O/H)'); ax.set_ylim(8.0, 9.4); ax.legend(loc='lower right')
ax.set_title('(b) gas-phase mass–metallicity\n(model: Asplund+09 solar, $Z_\\odot$ = 0.0134 ↔ 8.69)', fontsize=9)

# metal budget, centrals, group totals relative to metals produced
y, R = 0.025, 0.43
cid = g['cid']; u, inv = np.unique(cid, return_inverse=True)
S = lambda k: np.bincount(inv, weights=g[k], minlength=len(u))
cen = g['type'] == 0
ic = np.searchsorted(u, cid[cen])
tot = {k: S(k)[ic] for k in ['ms', 'mics', 'zstar', 'zcold', 'zhot', 'zcgm', 'zej', 'zics']}
prod = y * (tot['ms'] + tot['mics']) / (1 - R)
xc = x[cen]
ax = axs[1, 0]
labs = [('zstar', 'stars', BLUE), ('zcold', 'cold gas (ISM)', ORANGE), ('zhot', 'hot halo', AQUA),
        ('zcgm', 'CGM (Regime 0)', YELLOW), ('zej', 'ejected', MAGENTA), ('zics', 'ICS', VIOLET)]
c = 0.5 * (EDGES[1:] + EDGES[:-1]); stack = np.zeros(len(c))
for k, lab, col in labs:
    f = np.array([np.mean(tot[k][(xc >= a) & (xc < b)] / prod[(xc >= a) & (xc < b)]) if ((xc >= a) & (xc < b)).sum() > 10 else np.nan
                  for a, b in zip(EDGES[:-1], EDGES[1:])])
    ax.fill_between(c, stack, stack + f, color=col, alpha=0.85, lw=0, label=lab, edgecolor='white')
    stack = stack + f
ax.set_ylabel('fraction of metals produced\n(central + its satellites)'); ax.set_ylim(0, 1.05)
ax.legend(loc='upper left', ncol=2); ax.set_title('(c) where the metals are (centrals)')

ax = axs[1, 1]
for snap, col in (('Snap_48', BLUE), ('Snap_30', ORANGE), ('Snap_20', AQUA)):
    gg = g if snap == 'Snap_48' else load(snap)
    zlab = {'Snap_48': 'z = 0.02', 'Snap_30': 'z = 1.2', 'Snap_20': 'z = 2.5'}[snap]
    band(ax, gg['logms'], lg(gg['zcold'] / gg['mcold'] / ZSUN_G05), gg['sf'] & (gg['logms'] > 8.9), col, f'cold gas, SF, {zlab}')
    c_, o = binned(gg['logms'], lg(gg['zstar'] / gg['ms'] / ZSUN_G05), EDGES, gg['logms'] > 8.9)
    ax.plot(c_, o[:, 0], color=col, ls=':', lw=2, label=f'stars, all, {zlab}')
ax.set_ylabel(r'log $Z/0.02$'); ax.set_ylim(-1.0, 0.5); ax.legend(loc='lower right', ncol=1)
ax.set_title('(d) gas vs stellar metallicity with redshift')
for ax in axs.flat:
    ax.set_xlabel(r'log $M_*$ [M$_\odot$]'); ax.set_xlim(8.9, 11.9)
fig.suptitle('Issue 1 diagnosis: SAGE26 metallicities (microUchuu)', fontsize=12)
fig.tight_layout(); fig.savefig('fig2_metals_diagnosis.png', dpi=130); plt.close(fig)
print('done')

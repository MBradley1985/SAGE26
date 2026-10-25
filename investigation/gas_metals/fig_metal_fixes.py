"""Figure 4: literature-based fixes for issue 1 (metallicities), post-processed.

- Yield: SAGE26 uses Y = 0.025 per unit mass of stars formed. Peeples+14 adopt a
  core-collapse yield of 0.030 per unit mass formed (range 0.0214-0.0408 across
  stellar models). Metals enter the model linearly in Y (pristine infall), so Z
  scales by Y_new/0.025 (ignoring the metal dependence of cooling).
- Gas: the Tacconi+18 depletion-time scenario of Figure 3 is run at six
  snapshots; each galaxy's Lilly+13 Eq. 27a factor is mapped onto the
  star-formation history of every z = 0 galaxy to get its stellar metallicity
  change (stars formed in bin k inherit the gas factor of a progenitor of the
  same cumulative mass at that epoch).
- Observations: Gallazzi+05 is light-weighted; Peeples+14 Eq. 7 (from Peeples &
  Somerville 2013) converts it: log Z_mass = 1.08 log Z_light - 0.16.
"""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import load, binned, SP
from style import band, lg, G05, BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET, OBS, EDGES, DATA
from fig_fixes import run_scenarios, regulator_factor, R

SNAPS = {'Snap_15': 3.314, 'Snap_20': 2.459, 'Snap_25': 1.771, 'Snap_30': 1.220, 'Snap_40': 0.425, 'Snap_48': 0.0224}
TAC = 'Tacconi+18 $t_{\\rm dep}$'
Y_PEEPLES, Y_LO, Y_HI = 0.030, 0.0214, 0.0408


def gas_factor_grids():
    """SFR-weighted mean Z_new/Z_old per stellar-mass bin, for each snapshot (Tacconi scenario)."""
    e = np.arange(7.0, 12.51, 0.25); c = 0.5 * (e[1:] + e[:-1])
    grids, store = {}, {}
    for snap, z in SNAPS.items():
        g = load(snap)
        sel, sc = run_scenarios(g, z)
        full, part = regulator_factor(g, sel, sc[TAC]['mcold'])
        F = np.ones(len(g['ms'])); P = np.ones(len(g['ms']))
        F[sel] = full; P[sel] = part
        w = g['sfr']; lm = g['logms']
        gf = np.array([np.sum((w * F)[(lm >= a) & (lm < b)]) / max(np.sum(w[(lm >= a) & (lm < b)]), 1e-30) for a, b in zip(e[:-1], e[1:])])
        gp = np.array([np.sum((w * P)[(lm >= a) & (lm < b)]) / max(np.sum(w[(lm >= a) & (lm < b)]), 1e-30) for a, b in zip(e[:-1], e[1:])])
        gf[gf == 0] = 1; gp[gp == 0] = 1
        grids[z] = (gf, gp)
        store[snap] = (g, sel, sc)
    return c, grids, store


def stellar_factors(c, grids):
    d = np.load(SP + 'micro_sfh48.npz')
    sfh, zsn = d['sfh'].astype('f8') * 1e10 / 0.6774, d['z']
    cum = np.cumsum(sfh, axis=1); lmk = np.log10(np.maximum(cum, 1.0))
    zs = np.array(sorted(grids)); lz = np.log10(1 + zs)
    out = []
    for which in (0, 1):
        tab = np.array([grids[z][which] for z in zs])          # (nz, nbins)
        f = np.ones_like(sfh)
        for k in range(sfh.shape[1]):
            lzk = np.clip(np.log10(1 + zsn[k]), lz.min(), lz.max())
            row = np.array([np.interp(lzk, lz, tab[:, j]) for j in range(tab.shape[1])])
            f[:, k] = np.interp(lmk[:, k], c, row)
        out.append(np.sum(sfh * f, 1) / np.maximum(np.sum(sfh, 1), 1e-30))
    return d['gid'], out[0], out[1]


if __name__ == '__main__':
    c, grids, store = gas_factor_grids()
    gid_s, sf_full, sf_part = stellar_factors(c, grids)
    g, sel, sc = store['Snap_48']
    x = g['logms']
    idx = np.searchsorted(gid_s, g['gid']) if np.all(np.diff(gid_s) > 0) else None
    order = np.argsort(gid_s); pos = np.searchsorted(gid_s[order], g['gid']); pos = np.clip(pos, 0, len(order) - 1)
    hit = gid_s[order][pos] == g['gid']
    Ff = np.ones(len(x)); Fp = np.ones(len(x))
    Ff[hit] = sf_full[order][pos[hit]]; Fp[hit] = sf_part[order][pos[hit]]
    zst = g['zstar'] / g['ms']
    mass = x > 8.9

    fig, axs = plt.subplots(2, 2, figsize=(11, 8.4))
    ax = axs[0, 0]
    ax.plot(G05[:, 0], G05[:, 1], 'o', color=OBS, ms=5, mfc='white', label='Gallazzi+05, light-weighted')
    ax.plot(G05[:, 0], 1.08 * G05[:, 1] - 0.16, 'o', color=OBS, ms=5, label='Gallazzi+05 → mass-weighted (Peeples+14 Eq. 7)')
    ax.fill_between(G05[:, 0], 1.08 * G05[:, 2] - 0.16, 1.08 * G05[:, 3] - 0.16, color=OBS, alpha=0.08, lw=0)
    def zline(v, col, ls, lab, sel=mass):
        c_, o = binned(x, lg(v / 0.02), EDGES, sel); ax.plot(c_, o[:, 0], color=col, ls=ls, label=lab); return c_, o
    zline(zst, VIOLET, '-', 'SAGE26 now (mass-weighted)')
    c_, lo_ = binned(x, lg(zst * Y_LO / 0.025 / 0.02), EDGES, mass); c_, hi_ = binned(x, lg(zst * Y_HI / 0.025 / 0.02), EDGES, mass)
    ax.fill_between(c_, lo_[:, 0], hi_[:, 0], color=ORANGE, alpha=0.15, lw=0, label='yield 0.021–0.041 (Peeples+14 range)')
    zline(zst * Y_PEEPLES / 0.025, ORANGE, '-', 'yield 0.030 (Peeples+14)')
    zline(zst * Ff, BLUE, '-', 'Tacconi+18 $t_{\\rm dep}$ gas fix')
    zline(zst * Ff * Y_PEEPLES / 0.025, MAGENTA, '--', 'gas fix + yield 0.030')
    ax.set_ylabel(r'log $Z_*/0.02$'); ax.set_ylim(-1.2, 0.5); ax.legend(loc='lower right', fontsize=7)
    ax.set_title('(a) stellar mass–metallicity', fontsize=10)

    ax = axs[0, 1]
    oh = lambda zg: 8.69 + lg(zg / 0.0134)                      # Asplund+09
    s = g['sf'] & mass
    zg = g['zcold'] / g['mcold']
    full, part = regulator_factor(g, sel, sc[TAC]['mcold'])
    zg_t = zg.copy(); zg_t[sel] = zg[sel] * full
    for v, col, ls, lab in ((zg, VIOLET, '-', 'SAGE26 now'), (zg_t, BLUE, '-', 'Tacconi $t_{\\rm dep}$ gas fix'),
                            (zg_t * Y_PEEPLES / 0.025, MAGENTA, '--', 'gas fix + yield 0.030')):
        c_, o = binned(x, oh(v), EDGES, s); ax.plot(c_, o[:, 0], color=col, ls=ls, label=lab)
    m = np.linspace(9, 11.5, 60)
    ax.plot(m, 8.82 - 0.28 / 3.43 * np.log10(1 + (10 ** m / 10 ** 10.16) ** -3.43), color=OBS, lw=5, alpha=0.25, label='Sanders+21 z ~ 0 (Curti-form fit)')
    C = np.loadtxt(DATA + 'metallicity/Curti2020.dat')
    ax.plot(C[:, 0] - 0.04, C[:, 1], color=OBS, ls='--', lw=1.2, label='Curti+20')
    ax.set_ylabel('12 + log(O/H)  (model: Asplund+09 solar)'); ax.set_ylim(8.1, 9.2)
    ax.set_title('(b) gas-phase MZR, z = 0.02', fontsize=10); ax.legend(loc='lower right', fontsize=7.5)

    ax = axs[1, 0]
    g2, sel2, sc2 = store['Snap_20']; x2 = g2['logms']; s2 = g2['sf'] & (x2 > 8.9)
    zg2 = g2['zcold'] / g2['mcold']; f2, p2 = regulator_factor(g2, sel2, sc2[TAC]['mcold'])
    zg2t = zg2.copy(); zg2t[sel2] = zg2[sel2] * f2
    for v, col, ls, lab in ((zg2, VIOLET, '-', 'SAGE26 now, z = 2.46'), (zg2t, BLUE, '-', 'Tacconi $t_{\\rm dep}$ gas fix'),
                            (zg2t * Y_PEEPLES / 0.025, MAGENTA, '--', 'gas fix + yield 0.030')):
        c_, o = binned(x2, oh(v), EDGES, s2); ax.plot(c_, o[:, 0], color=col, ls=ls, label=lab)
    m = np.linspace(9.0, 10.5, 20)
    ax.plot(m, 8.51 + 0.30 * (m - 10), color=OBS, lw=5, alpha=0.25, label='Sanders+21 z ~ 2.3 (fit range 9–10.5)')
    ax.plot(np.linspace(10.5, 11, 5), 8.51 + 0.30 * (np.linspace(10.5, 11, 5) - 10), color=OBS, lw=1, ls=':')
    ax.set_ylabel('12 + log(O/H)'); ax.set_ylim(7.9, 9.0)
    ax.set_title('(c) gas-phase MZR at cosmic noon (same calibration as b)', fontsize=10); ax.legend(loc='lower right', fontsize=7.5)

    ax = axs[1, 1]
    for snap, col in (('Snap_48', BLUE), ('Snap_20', ORANGE)):
        gg, ss, _ = store[snap]; xx = gg['logms']; sfm = gg['sf'] & (xx > 8.9)
        zl = {'Snap_48': 'z = 0.02', 'Snap_20': 'z = 2.46'}[snap]
        c_, o = binned(xx, gg['loading'] / (1 - R), EDGES, sfm); ax.plot(c_, o[:, 0], color=col, label=fr'$\lambda/(1-R)$, {zl}')
        c_, o = binned(xx, gg['mcold'] / gg['ms'], EDGES, sfm); ax.plot(c_, o[:, 0], color=col, ls='--', label=fr'$\mu = M_{{\rm gas}}/M_*$, {zl}')
    ax.axhline(1, color='#52514e', lw=1, ls=':', label='1')
    ax.set_yscale('log'); ax.set_ylabel('term in Lilly+13 denominator'); ax.set_ylim(0.1, 20)
    ax.set_title(r'(d) why gas cuts move Z little: $Z_{\rm eq}=Z_0+y/[1+\lambda/(1-R)+\mu]$', fontsize=10)
    ax.legend(loc='upper right', fontsize=7.5)
    for ax in axs.flat:
        ax.set_xlabel(r'log $M_*$ [M$_\odot$]'); ax.set_xlim(8.9, 11.9)
    fig.suptitle('Issue 1 fixes, post-processed (microUchuu)', fontsize=12)
    fig.tight_layout(); fig.savefig('fig4_metal_fixes.png', dpi=130); plt.close(fig)

    for lo in (10.0, 10.5, 11.0, 11.5):
        mm = mass & (x >= lo - 0.1) & (x < lo + 0.1)
        print(f"logM* {lo}: Z* now {np.log10(np.median(zst[mm])/0.02):+.2f}, gas-fix dlogZ* full {np.median(np.log10(Ff[mm])):.3f} part {np.median(np.log10(Fp[mm])):.3f}, G05 light {np.interp(lo, G05[:,0], G05[:,1]):+.2f} mass {1.08*np.interp(lo, G05[:,0], G05[:,1])-0.16:+.2f}")
    for lo in (9.5, 10.0, 10.5):
        mm = s2 & (x2 >= lo - 0.1) & (x2 < lo + 0.1)
        print(f"z=2.46 logM* {lo}: O/H now {np.nanmedian(oh(zg2[mm])):.2f}, gasfix {np.nanmedian(oh(zg2t[mm])):.2f}, Sanders {8.51+0.30*(lo-10):.2f}")
    for lo in (9.5, 10.0, 10.5):
        mm = s & (x >= lo - 0.1) & (x < lo + 0.1)
        print(f"z=0 logM* {lo}: O/H now {np.nanmedian(oh(zg[mm])):.2f}, Sanders z0 {8.82 - 0.28/3.43*np.log10(1+(10**lo/10**10.16)**-3.43):.2f}")
    print('hit fraction', hit[mass].mean())

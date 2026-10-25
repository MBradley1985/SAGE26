"""Figures 3-4: literature-based fixes for issues 1 and 2, applied by post-processing
SAGE26 (microUchuu) galaxies. Nothing here reruns the model.

Method (Lilly et al. 2013, ApJ 772, 119): in a gas-regulated disk the SFR is set
by supply and outflow; the star-formation law only sets the reservoir needed,
mu = eps^-1 sSFR (their Eq. 7). Each star-forming galaxy keeps its SFR, and we
solve for the cold gas each alternative recipe needs. The metallicity response
uses their ideal equilibrium Z_eq = Z0 + y / [1 + lambda/(1-R) + mu] (Eq. 27a).
"""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import load, h2_br06, tdyn_yr, solve_gas, binned, SP, X_H
from style import band, lg, G05, BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET, OBS, EDGES, DATA

R, Y_SAGE = 0.43, 0.025
HE = 1.36                                   # H2 -> H2+He, as in xCOLD GASS (Saintonge+17) and Tacconi+18
G_CGS, K_B, MSUNPC2 = 6.674e-8, 1.381e-16, 2.089e-4  # Msun/pc^2 -> g/cm^2


# ----------------------------------------------------------------- literature relations
def speagle_ms_ssfr(logm, z):
    """Speagle+14 MS as written in Tacconi+18 Eq. 1 [yr^-1] (Chabrier)."""
    l = np.log10(1 + z)
    tc = 10 ** (1.143 - 1.026 * l - 0.599 * l ** 2 + 0.528 * l ** 3)
    return 10 ** ((-0.16 - 0.026 * tc) * (logm + 0.025) - (6.51 - 0.11 * tc) + 9) / 1e9


def tacconi_tdep_yr(logm, ssfr, z):
    """Tacconi+18 Table 3a 'best' t_dep (H2 incl. He) in yr; size term set to zero."""
    dms = np.clip(ssfr / speagle_ms_ssfr(logm, z), 1e-2, 1e2)
    return 1e9 * 10 ** (0.09 - 0.62 * np.log10(1 + z) - 0.44 * np.log10(dms) + 0.09 * (logm - 10.7))


def lange16_re_kpc(mdisk):
    """Lange+16 combined z = 0 disc relation, R_e = 5.141 (M/1e10)^0.274 kpc (H0 = 70)."""
    return 5.141 * (mdisk / 1e10) ** 0.274


def h2_leroy08(mcold, mdisk_star, rs_pc, nbins=25, rmax=5.0):
    """Leroy+08: R_mol = (P_h/P0)^alpha with log P0/k = 4.23, alpha = 0.80 (Table 6, spirals
    combined), P_h = (pi/2) G Sg (Sg + (s_g/s_*z) S*) (Eq. 9), s_g = 11 km/s,
    s_*z = sqrt(2 pi G l*/7.3) S*^0.5 (Eq. B3)."""
    sg0 = mcold / (2 * np.pi * rs_pc ** 2); ss0 = np.where(mdisk_star > 0, mdisk_star / (2 * np.pi * rs_pc ** 2), 0.0)
    l_cm = rs_pc * 3.0857e18; dr = rmax * rs_pc / nbins; out = np.zeros_like(mcold)
    for i in range(nbins):
        r = (i + 0.5) * dr
        sg = sg0 * np.exp(-r / rs_pc) * MSUNPC2; ss = np.maximum(ss0 * np.exp(-r / rs_pc), 0.1) * MSUNPC2
        sz = np.sqrt(2 * np.pi * G_CGS * l_cm / 7.3 * ss)
        p = 0.5 * np.pi * G_CGS * sg * (sg + (1.1e6 / sz) * ss) / K_B
        rmol = (p / 10 ** 4.23) ** 0.80
        out += 2 * np.pi * r * (sg / MSUNPC2) * X_H * rmol / (1 + rmol) * dr
    return out


# ----------------------------------------------------------------- scenarios
def model_size_ratio(g, sel):
    """Per-galaxy factor taking the model's median disk size at its disk mass onto Lange+16."""
    md = g['ms'] - g['mb']; lmd = np.log10(np.maximum(md, 1.0))
    e = np.arange(8.5, 12.01, 0.1); c = 0.5 * (e[1:] + e[:-1])
    med = np.array([np.median(1.678 * g['rs_pc'][sel & (lmd >= a) & (lmd < b)] / 1e3) if (sel & (lmd >= a) & (lmd < b)).sum() > 10 else np.nan
                    for a, b in zip(e[:-1], e[1:])])
    ok = np.isfinite(med)
    return lange16_re_kpc(md) / np.interp(lmd, c[ok], med[ok])


def run_scenarios(g, z):
    sel = g['sf'] & (g['logms'] > 8.9) & (g['sfr_disk'] > 0) & (g['rs_pc'] > 0) & (g['mcold'] > 0)
    md = (g['ms'] - g['mb'])[sel]; rs = g['rs_pc'][sel]; vv = g['vvir'][sel]; target = g['sfr_disk'][sel]
    lo, hi = np.full(sel.sum(), 1e3), np.full(sel.sum(), 1e13)
    td_T = tacconi_tdep_yr(g['logms'][sel], g['ssfr'][sel], z) / HE        # H-only depletion time
    rs_L = rs * model_size_ratio(g, sel)[sel]
    law = {
        'SAGE26 now': lambda mg, r=rs: 0.05 * h2_br06(mg, md, r) / tdyn_yr(r, vv),
        'Tacconi+18 $t_{\\rm dep}$': lambda mg: h2_br06(mg, md, rs) / td_T,
        'Lange+16 disk sizes': lambda mg: 0.05 * h2_br06(mg, md, rs_L) / tdyn_yr(rs_L, vv),
        'Leroy+08 pressure law': lambda mg: 0.05 * h2_leroy08(mg, md, rs) / tdyn_yr(rs, vv),
        'Tacconi $t_{\\rm dep}$ + Lange sizes': lambda mg: h2_br06(mg, md, rs_L) / td_T,
    }
    h2f = {'Leroy+08 pressure law': lambda mg: h2_leroy08(mg, md, rs)}
    out = {}
    for name, f in law.items():
        mg = solve_gas(target, f, lo, hi)
        r_ = rs_L if 'Lange' in name else rs
        mh2 = h2f[name](mg) if name in h2f else h2_br06(mg, md, r_)
        new = {k: g[k].copy() for k in ('mcold', 'mh2', 'mhi')}
        new['mcold'][sel] = mg; new['mh2'][sel] = mh2; new['mhi'][sel] = X_H * mg - mh2
        out[name] = new
    return sel, out


def regulator_factor(g, sel, mcold_new):
    """Z_new/Z_old from Lilly+13 Eq. 27a with SFR, lambda fixed; full response (Z0 = 0)
    and partial response (inflow metallicity Z0 = Z_model - Z_eq,ideal held fixed)."""
    lam = g['loading'][sel]; ms = g['ms'][sel]
    d_old = 1 + lam / (1 - R) + g['mcold'][sel] / ms
    d_new = 1 + lam / (1 - R) + mcold_new[sel] / ms
    full = d_old / d_new
    z = g['zcold'][sel] / g['mcold'][sel]
    zeq = (Y_SAGE / (1 - R)) / d_old
    z0 = np.clip(z - zeq, 0, None)
    part = (z0 + (z - z0) * full) / z
    return full, part


STYLE = {'SAGE26 now': (VIOLET, '-'), 'Tacconi+18 $t_{\\rm dep}$': (BLUE, '-'), 'Lange+16 disk sizes': (ORANGE, '-'),
         'Leroy+08 pressure law': (AQUA, '-'), 'Tacconi $t_{\\rm dep}$ + Lange sizes': (MAGENTA, '--')}


def line(ax, x, y, sel, col, ls, lab, stat=np.median, edges=EDGES):
    c, o = binned(x, y, edges, sel, stat=stat)
    ax.plot(c, o[:, 0], color=col, ls=ls, label=lab)
    return c, o


if __name__ == '__main__':
    g = load('Snap_48'); z0 = 0.0224
    sel, sc = run_scenarios(g, z0)
    x = g['logms']
    # ------------------------------------------------------------- Figure 3: gas fixes
    fig, axs = plt.subplots(2, 2, figsize=(11, 8.4))
    ax = axs[0, 0]
    ms_sel = sel & (np.abs(np.log10(g['ssfr'] / speagle_ms_ssfr(x, z0))) < 0.4)
    logmean = lambda v: np.log10(np.mean(10 ** v))
    for name, new in sc.items():
        col, ls = STYLE[name]
        c, o = binned(x, lg(HE * new['mh2'] / g['ms']), EDGES, ms_sel, stat=logmean)
        lab = name + (' (H$_2$ = SAGE26 now)' if 'Leroy' in name else '')
        ax.plot(c, o[:, 0], color=col, ls=':' if name == 'SAGE26 now' else ls, lw=3 if name == 'SAGE26 now' else 2,
                label=lab, zorder=5 if name == 'SAGE26 now' else 2)
    s17 = np.array([[10.07, -0.89], [10.25, -1.05], [10.46, -0.99], [10.68, -1.10], [10.82, -1.26], [11.10, -1.41]])
    ax.plot(s17[:, 0], s17[:, 1], 's', color=OBS, ms=6, label='xCOLD GASS MS stacks (Saintonge+17 T5)')
    mu_t = lambda lm: 10 ** (-1.19 + 2.49 * 0 - 0.36 * (lm - 10.7))        # Tacconi+18 beta=0 form, z=0, dMS=1
    lm_ = np.linspace(9, 11.3, 50)
    ax.plot(lm_, np.log10(mu_t(lm_)), color=OBS, ls=':', lw=1.5, label=r'Tacconi+18 $\mu_{\rm gas}$, z = 0 MS')
    ax.set_ylabel(r'log $\langle M_{\rm H_2+He}/M_*\rangle$, main sequence'); ax.set_ylim(-2, 0.2)
    ax.set_title('(a) molecular gas, MS galaxies ($|\\Delta$MS$|<0.4$ dex, Speagle+14)', fontsize=10)
    ax.legend(loc='lower left', fontsize=7.5)

    ax = axs[0, 1]
    every = (x > 8.9)
    for name, new in sc.items():
        col, ls = STYLE[name]
        line(ax, x, lg(new['mhi'] / g['ms']), every, col, ls, name)
    d = np.loadtxt(DATA + 'Gas/HIGasRatio_NonDetEQUpperLimits.dat')
    ax.errorbar(d[:, 0], d[:, 1], yerr=[d[:, 2], d[:, 3]], fmt='o', color=OBS, ms=5, capsize=2, lw=1, label='xGASS medians (non-det = limit)')
    c18 = np.array([[10.07, -0.869], [10.34, -1.175], [10.65, -1.231], [10.95, -1.475], [11.20, -1.589]])
    ax.plot(c18[:, 0], c18[:, 1], 'D', color=OBS, mfc='white', ms=6, label='xGASS weighted means (Catinella+18 T1)')
    ax.set_ylabel(r'log $M_{\rm HI}/M_*$, all galaxies'); ax.set_ylim(-2.5, 0.6)
    ax.set_title('(b) atomic gas, all galaxies (only SF galaxies changed)', fontsize=10); ax.legend(loc='lower left', fontsize=7.5)

    ax = axs[1, 0]
    for name, new in sc.items():
        col, ls = STYLE[name]
        line(ax, x, lg(HE * new['mh2'] / g['sfr']), ms_sel, col, ls, name)
    ax.axhspan(8.98 - 0.27, 8.98 + 0.27, color=OBS, alpha=0.1, lw=0, label='Saintonge+17 MS, 8.98 ± 0.27')
    ax.axhline(np.log10(2.2e9), color=OBS, ls='--', lw=1, label='Leroy+13, 2.2 Gyr')
    ax.plot(lm_, np.log10(tacconi_tdep_yr(lm_, speagle_ms_ssfr(lm_, z0), z0)), color=OBS, ls=':', lw=1.5, label='Tacconi+18, z = 0 MS')
    ax.set_ylabel(r'log $t_{\rm dep}({\rm H_2+He})$ [yr], MS'); ax.set_ylim(8.4, 10.0)
    ax.set_title('(c) molecular depletion time', fontsize=10); ax.legend(loc='upper left', fontsize=7.5, ncol=2)

    ax = axs[1, 1]
    for name, new in sc.items():
        if name == 'SAGE26 now':
            continue
        col, ls = STYLE[name]
        full, part = regulator_factor(g, sel, new['mcold'])
        xs = x[sel]
        c, o = binned(xs, np.log10(full), EDGES); c, p = binned(xs, np.log10(part), EDGES)
        ax.plot(c, o[:, 0], color=col, ls=ls, label=name)
        ax.fill_between(c, p[:, 0], o[:, 0], color=col, alpha=0.15, lw=0)
    ax.axhline(0, color='#52514e', lw=0.8)
    ax.set_ylabel(r'$\Delta\log Z_{\rm gas}$ (Lilly+13 Eq. 27a)'); ax.set_ylim(-0.05, 0.25)
    ax.set_title('(d) gas-metallicity response, star-forming galaxies\nline: Z$_0$ = 0; band edge: enriched inflow held fixed', fontsize=9)
    ax.legend(loc='upper right', fontsize=7.5)
    for ax in axs.flat:
        ax.set_xlabel(r'log $M_*$ [M$_\odot$]'); ax.set_xlim(8.9, 11.6)
    fig.suptitle('Issue 2 fixes, post-processed at fixed SFR (microUchuu z = 0.02)', fontsize=12)
    fig.tight_layout(); fig.savefig('fig3_gas_fixes.png', dpi=130); plt.close(fig)

    # numbers for the write-up
    for name, new in sc.items():
        for lo in (9.25, 9.75, 10.25):
            m = every & (x >= lo - 0.25) & (x < lo + 0.25); mm = ms_sel & (x >= lo - 0.25) & (x < lo + 0.25)
            print(f"{name:40s} {lo}: HI/M* all {np.median(lg(new['mhi'][m] / g['ms'][m])):+.2f}  H2+He/M* MS {logmean(lg(HE * new['mh2'][mm] / g['ms'][mm])):+.2f}  tdep {np.median(lg(HE * new['mh2'][mm] / g['sfr'][mm])):.2f}")
    np.savez(SP + 'scen48.npz', sel=sel, **{f'{k}': v['mcold'] for k, v in sc.items()})

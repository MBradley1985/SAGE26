#!/usr/bin/env python
"""
energy_budget_diagnostic.py -- how much of the supernova energy budget the
FIRE ejection term actually spends.

The code sets E_FB = eps_halo * f_FIRE(V_vir, z) * 0.5 * m_* * eta_SN E_SN,
so the fraction of the total SN energy (m_* eta_SN E_SN) that is spent is

    E_FB / E_available = eps_halo * f_FIRE / 2

which is unbounded.  This script shows (a) where in the (V_vir, z) plane that
exceeds unity, and (b) how much of the star formation in a given run sits in
the over-budget regime.

Usage:
    python plotting/energy_budget_diagnostic.py label=dir [label=dir ...] \
        --out output/energy_cap_test
"""

import argparse
import glob
import os

import h5py as h5
import matplotlib.pyplot as plt
import numpy as np

V_CRIT = 60.0


def f_fire(v, z, alpha):
    v = np.maximum(np.asarray(v, dtype=float), 1.0)
    beta = np.where(v < V_CRIT, -3.2, -1.0)
    return (1.0 + z) ** alpha * (v / V_CRIT) ** beta


def run_series(directory):
    """SFR-weighted E_FB/E_available vs z, and the SFR fraction over budget."""
    files = sorted(glob.glob(os.path.join(directory, 'model_*.hdf5')))
    with h5.File(files[0], 'r') as f:
        r = dict(f['Header/Runtime'].attrs)
        zz = np.array(f['Header/snapshot_redshifts'])
        snaps = sorted(int(k.split('_')[1]) for k in f.keys() if k.startswith('Snap_'))
    eps_h = float(r['FeedbackEjectionEfficiency'])
    alpha = float(r['RedshiftPowerLawExponent'])
    z, mean_spend, frac2, frac1, worst = [], [], [], [], []
    for s in snaps:
        if s >= len(zz):
            continue
        sfr, vv = [], []
        for fn in files:
            with h5.File(fn, 'r') as f:
                g = f[f'Snap_{s}']
                if g['StellarMass'].shape[0] == 0:
                    continue
                sfr.append(np.array(g['SfrDisk']) + np.array(g['SfrBulge']))
                vv.append(np.array(g['Vvir']))
        if not sfr:
            continue
        sfr = np.concatenate(sfr)
        vv = np.concatenate(vv)
        w = (sfr > 0) & (vv > 0)
        if w.sum() < 50:
            continue
        spend = eps_h * f_fire(vv[w], zz[s], alpha) / 2.0   # E_FB / E_available
        sf = sfr[w]
        z.append(zz[s])
        mean_spend.append(np.sum(spend * sf) / np.sum(sf))
        frac2.append(np.sum(sf[spend > 1.0]) / np.sum(sf))   # over the eps_eff<=2 bound
        frac1.append(np.sum(sf[spend > 0.5]) / np.sum(sf))   # over the eps_eff<=1 bound
        worst.append(spend.max())
    o = np.argsort(z)
    return dict(z=np.array(z)[o], mean=np.array(mean_spend)[o],
                frac2=np.array(frac2)[o], frac1=np.array(frac1)[o],
                worst=np.array(worst)[o], eps_h=eps_h, alpha=alpha)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+', help='label=directory')
    ap.add_argument('--out', default='output/energy_cap_test')
    args = ap.parse_args()

    stylesheet = './plotting/kieren_cohare_palatino_sty.mplstyle'
    if os.path.exists(stylesheet):
        plt.style.use(stylesheet)

    series = []
    for spec in args.runs:
        label, _, d = spec.partition('=')
        series.append((label, run_series(d)))

    eps_h = series[0][1]['eps_h']
    alpha = series[0][1]['alpha']

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.9))

    # (a) analytic map
    ax = axes[0]
    vv = np.logspace(np.log10(10), np.log10(500), 400)
    cmap = plt.get_cmap('viridis')
    zs = [0, 1, 2, 4, 6, 8]
    for zt, c in zip(zs, [cmap(x) for x in np.linspace(0.05, 0.85, len(zs))]):
        ax.plot(vv, eps_h * f_fire(vv, zt, alpha) / 2.0, color=c, lw=2.2,
                label=rf'$z={zt}$')
    ax.axhline(1.0, color='crimson', ls='--', lw=1.6)
    ax.axhline(0.5, color='crimson', ls=':', lw=1.6)
    ax.text(11, 1.25, 'all of the SN energy', color='crimson', fontsize=10)
    ax.text(11, 0.30, 'half of the SN energy', color='crimson', fontsize=10)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(10, 500)
    ax.set_ylim(1e-2, 1e3)
    ax.set_xlabel(r'$V_{\rm vir}\ [{\rm km\,s^{-1}}]$')
    ax.set_ylabel(r'$E_{\rm FB} / (\dot{m}_{*}\,\eta_{\rm SN} E_{\rm SN})$')
    ax.set_title('(a) fraction of the SN budget spent', loc='left')
    ax.legend(frameon=False, fontsize=9, ncol=2, loc='upper right')

    # (b) SFR-weighted mean over cosmic time
    ax = axes[1]
    for (label, s), ls in zip(series, ['-', '--', '-.', ':']):
        ax.plot(s['z'], s['mean'], ls, lw=2.4, label=f'{label} (SFR-weighted mean)')
        ax.plot(s['z'], s['worst'], ':', lw=1.6, alpha=0.85,
                color=ax.lines[-1].get_color(), label=f'{label} (worst galaxy)')
    ax.axhline(1.0, color='crimson', ls='--', lw=1.6)
    ax.set_yscale('log')
    ax.set_xlim(0, 12)
    ax.set_xlabel('$z$')
    ax.set_ylabel(r'$E_{\rm FB} / (\dot{m}_{*}\,\eta_{\rm SN} E_{\rm SN})$')
    ax.set_title('(b) realised energy spend', loc='left')
    ax.legend(frameon=False, fontsize=8.5)

    # (c) SFR fraction over budget
    ax = axes[2]
    for (label, s), ls in zip(series, ['-', '--', '-.', ':']):
        p = ax.plot(s['z'], 100 * s['frac2'], ls, lw=2.4,
                    label=rf'{label}: over $\epsilon_{{\rm eff}}=2$')
        ax.plot(s['z'], 100 * s['frac1'], ':', lw=1.8, alpha=0.9,
                color=p[0].get_color(),
                label=rf'{label}: over $\epsilon_{{\rm eff}}=1$')
    ax.set_yscale('log')
    ax.set_xlim(0, 12)
    ax.set_ylim(1e-2, 150)
    ax.set_xlabel('$z$')
    ax.set_ylabel(r'per cent of $\rho_{\rm SFR}$ over budget')
    ax.set_title('(c) how much star formation is affected', loc='left')
    ax.legend(frameon=False, fontsize=8.5, loc='lower right')

    fig.tight_layout()
    os.makedirs(args.out, exist_ok=True)
    for ext in ('.pdf', '.png'):
        p = os.path.join(args.out, f'energy_budget_diagnostic{ext}')
        fig.savefig(p, dpi=110)
        print(f'  Saved: {p}')
    plt.close(fig)


if __name__ == '__main__':
    main()

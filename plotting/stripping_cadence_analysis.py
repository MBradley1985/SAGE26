#!/usr/bin/env python
"""
stripping_cadence_analysis.py -- why the satellite-stripping scheme matters
differently for Millennium vs microUchuu, and whether the physical scheme makes
the two simulations more consistent.

Produces a 4-panel figure:
  A. Mechanism: fraction stripped per snapshot vs dT/t_dyn (legacy is flat ~65%,
     physical is 1-exp(-dT/t_dyn)); the two sims sit at different dT/t_dyn.
  B. Fully CGM-stripped satellite fraction vs host halo mass, LEGACY (Mill vs Uch).
  C. Same, PHYSICAL -- curves should be closer together.
  D. Cross-simulation |Mill - Uch| discrepancy vs host mass, legacy vs physical.

Run from the repo root:
  python3 plotting/stripping_cadence_analysis.py
"""
import glob
import importlib.util

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# reuse the loaders/masks from compare_stripping.py
spec = importlib.util.spec_from_file_location('cs', 'plotting/compare_stripping.py')
cs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cs)
cs.SATELLITE_SUBSET = 'all'

RUNS = {
    'mill_m0': 'legacy_lowN', 'mill_m2': 'phys_once',
    'uch_m0': 'microuchuu_legacy', 'uch_m2': 'microuchuu_phys_once',
}
STEPS, MAX_STEPS = 10, 30


# ---------------------------------------------------------------------------
def age_gyr(z, h, Om, OL):
    return (2.0 / (3.0 * np.sqrt(OL))) * (9.778 / h) * \
        np.arcsinh(np.sqrt(OL / Om) * (1.0 + z) ** -1.5)


def cadence_ratio(pattern):
    """Median low-z dT/t_dyn for a run's host haloes."""
    f = sorted(glob.glob(pattern))[0]
    with h5py.File(f, 'r') as h:
        s = h['Header/Simulation'].attrs
        hub, Om, OL = float(s['hubble_h']), float(s['omega_matter']), float(s['omega_lambda'])
        zs = np.array(h['Header/snapshot_redshifts'])
        snaps = sorted(int(k.replace('Snap_', '')) for k in h.keys() if k.startswith('Snap_'))
        last = max(snaps)
        g = h[f'Snap_{last}']
        typ, Rvir, Vvir = np.array(g['Type']), np.array(g['Rvir']), np.array(g['Vvir'])
    cen = (typ == 0) & (Vvir > 0)
    tdyn = (Rvir[cen] / hub) / Vvir[cen] * 3.0857e19 / 3.1557e16          # Gyr
    ages = [age_gyr(zs[snaps[i]], hub, Om, OL) for i in range(-6, 0)]
    dT = np.median(np.diff(ages))
    return dT / np.median(tdyn), dT, np.median(tdyn)


def load_with_host(pattern):
    """cs.load_run + host halo mass (CentralMvir, physical Msun)."""
    run = cs.load_run(pattern, None)
    files = sorted(glob.glob(pattern))
    h = cs.read_simulation_params(files[0])['Hubble_h']
    snap = f"Snap_{run['snap_num']}"
    run['CentralMvir'] = cs.read_hdf(files, snap, 'CentralMvir') * 1.0e10 / h
    return run


def stripped_fraction_vs_host(run, edges):
    """Fully CGM-stripped satellite fraction in log10(host Mvir) bins."""
    m = cs.satellite_mask(run)
    lhm = np.log10(np.where(run['CentralMvir'] > 0, run['CentralMvir'], 1.0))
    xc, frac = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = m & (run['CentralMvir'] > 0) & (lhm >= a) & (lhm < b)
        n = np.count_nonzero(sel)
        if n >= 50:
            xc.append(0.5 * (a + b))
            frac.append(np.count_nonzero(run['CGMgas'][sel] == 0) / n)
        else:
            xc.append(0.5 * (a + b)); frac.append(np.nan)
    return np.array(xc), np.array(frac)


# ---------------------------------------------------------------------------
print('Loading runs...')
R = {k: load_with_host(f'output/{d}/model_*.hdf5') for k, d in RUNS.items()}
ratio_mill, dT_mill, td_mill = cadence_ratio('output/legacy_lowN/model_*.hdf5')
ratio_uch, dT_uch, td_uch = cadence_ratio('output/microuchuu_legacy/model_*.hdf5')
print(f'  Millennium dT/t_dyn = {ratio_mill:.3f}  (dT={dT_mill:.2f} Gyr, t_dyn={td_mill:.2f} Gyr)')
print(f'  microUchuu dT/t_dyn = {ratio_uch:.3f}  (dT={dT_uch:.2f} Gyr, t_dyn={td_uch:.2f} Gyr)')

# --- fully-stripped fraction vs SATELLITE stellar mass (isolates cadence) ---
def stripped_vs_smass(run, edges):
    m = cs.satellite_mask(run); sm = run['StellarMass']
    ls = np.log10(np.where(sm > 0, sm, 1.0))
    xc, fr = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = m & (sm > 0) & (ls >= a) & (ls < b)
        n = np.count_nonzero(sel)
        xc.append(0.5 * (a + b))
        fr.append(np.count_nonzero(run['CGMgas'][sel] == 0) / n if n >= 50 else np.nan)
    return np.array(xc), np.array(fr)

sedges = np.arange(9.0, 11.5, 0.5)
xs, fm0 = stripped_vs_smass(R['mill_m0'], sedges)
_, fm2 = stripped_vs_smass(R['mill_m2'], sedges)
_, fu0 = stripped_vs_smass(R['uch_m0'], sedges)
_, fu2 = stripped_vs_smass(R['uch_m2'], sedges)
gap_leg, gap_phys = np.abs(fm0 - fu0), np.abs(fm2 - fu2)

# --- 2D-matched cells (host x satellite mass): cadence-controlled scatter ---
def sfrac(r, ha, hb, sa, sb):
    m = cs.satellite_mask(r); sm = r['StellarMass']; cm = r['CentralMvir']
    lh = np.log10(np.where(cm > 0, cm, 1)); ls = np.log10(np.where(sm > 0, sm, 1))
    sel = m & (cm > 0) & (sm > 0) & (lh >= ha) & (lh < hb) & (ls >= sa) & (ls < sb)
    n = np.count_nonzero(sel)
    return (np.count_nonzero(r['CGMgas'][sel] == 0) / n, n) if n >= 50 else (np.nan, 0)

hedges = np.arange(11.5, 14.5, 0.5)
cell_leg, cell_phys, cell_w = [], [], []
for ha, hb in zip(hedges[:-1], hedges[1:]):
    for sa, sb in zip(sedges[:-1], sedges[1:]):
        v = {k: sfrac(R[k], ha, hb, sa, sb) for k in R}
        if min(v[k][1] for k in R) >= 50:
            cell_leg.append(abs(v['mill_m0'][0] - v['uch_m0'][0]))
            cell_phys.append(abs(v['mill_m2'][0] - v['uch_m2'][0]))
            cell_w.append(min(v[k][1] for k in R))
cell_leg, cell_phys, cell_w = map(np.array, (cell_leg, cell_phys, cell_w))
red2d = 100 * (1 - np.sum(cell_phys * cell_w) / np.sum(cell_leg * cell_w))
print(f'\n2D-matched (host x sat mass) cross-sim |diff|: '
      f'legacy={np.sum(cell_leg*cell_w)/np.sum(cell_w):.3f}  '
      f'physical={np.sum(cell_phys*cell_w)/np.sum(cell_w):.3f}  reduction={red2d:.0f}%')

# ---------------------------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(13, 11))
axA, axB, axC, axD = ax.flatten()

# Panel A -- mechanism
x = np.linspace(0.01, 1.3, 300)
def legacy_frac(r):                              # 1-(1-1/N)^N, N=clamp(ceil(10r),[10,30])
    N = np.clip(np.ceil(STEPS * r), STEPS, MAX_STEPS)
    return 1 - (1 - 1 / N) ** N
axA.plot(x, 1 - np.exp(-x), color='C2', lw=2.5, label=r'physical  $1-e^{-dT/t_{\rm dyn}}$')
axA.plot(x, legacy_frac(x), color='C3', lw=2.5, label='legacy  (fixed ~65%/snapshot)')
for r, dc, lab in [(ratio_mill, 'C0', 'Millennium'), (ratio_uch, 'C1', 'microUchuu')]:
    axA.axvline(r, color=dc, ls=':', lw=1.8)
    axA.plot([r], [1 - np.exp(-r)], 'o', color=dc, ms=9)
    axA.plot([r], [legacy_frac(r)], 's', color=dc, ms=9)
    axA.annotate(f'{lab}\ndT/t_dyn={r:.2f}', (r, 0.06), color=dc, fontsize=9,
                 ha='center', va='bottom')
axA.set(xlabel=r'$dT / t_{\rm dyn}$  (snapshot spacing / dynamical time)',
        ylabel='fraction of excess stripped per snapshot', ylim=(0, 1), xlim=(0, 1.3))
axA.set_title('A. Mechanism: cadence sets the per-snapshot stripping rate')
axA.legend(loc='center right', fontsize=9); axA.grid(alpha=0.3)

# Panel B -- fully-stripped fraction vs satellite mass, all 4 runs
axB.plot(xs, fm0, '-o', color='C0', label='Millennium legacy')
axB.plot(xs, fu0, '-o', color='C1', label='microUchuu legacy')
axB.plot(xs, fm2, '--s', color='C0', label='Millennium physical')
axB.plot(xs, fu2, '--s', color='C1', label='microUchuu physical')
axB.set(xlabel=r'$\log_{10}(M_*^{\rm sat}/M_\odot)$',
        ylabel='fully CGM-stripped fraction', ylim=(0, 1))
axB.set_title('B. At matched satellite mass (dashed=physical pulls sims together)')
axB.legend(fontsize=8); axB.grid(alpha=0.3)

# Panel C -- cross-sim discrepancy vs satellite mass
w = 0.18
axC.bar(xs - w/2, gap_leg, width=w, color='C3', label='legacy |Mill-Uch|')
axC.bar(xs + w/2, gap_phys, width=w, color='C2', label='physical |Mill-Uch|')
axC.set(xlabel=r'$\log_{10}(M_*^{\rm sat}/M_\odot)$', ylabel='cross-sim discrepancy')
axC.set_title('C. Physical shrinks the discrepancy (matched satellite mass)')
axC.legend(fontsize=9); axC.grid(alpha=0.3, axis='y')

# Panel D -- 2D-matched per-cell scatter: below the line = physical helps
mx = max(cell_leg.max(), cell_phys.max()) * 1.05
axD.plot([0, mx], [0, mx], 'k--', lw=1, alpha=0.6)
axD.scatter(cell_leg, cell_phys, s=cell_w/3, c='C4', alpha=0.7, edgecolor='k', lw=0.4)
axD.set(xlabel='legacy cross-sim |diff|  (per host x sat-mass cell)',
        ylabel='physical cross-sim |diff|', xlim=(0, mx), ylim=(0, mx))
axD.set_title(f'D. 2D-matched cells: {int(len(cell_leg))} cells, '
              f'{red2d:.0f}% mean reduction\n(below dashed line = physical brings sims closer)')
axD.grid(alpha=0.3)

fig.suptitle('Snapshot-cadence dependence of satellite stripping, and Mill-microUchuu consistency  (z=0)',
             fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.98))
fig.savefig('stripping_cadence_story.png', dpi=150, bbox_inches='tight')
print('Wrote stripping_cadence_story.png')

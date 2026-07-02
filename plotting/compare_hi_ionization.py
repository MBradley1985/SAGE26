#!/usr/bin/env python
"""
compare_hi_ionization.py -- sensitivity of the HI mass function to the
HIIonizationOn / SigmaHIcrit cut (Shark-style ionised outer-disk removal).

SAGE assigns all non-molecular cold hydrogen to HI. The optional ionisation cut
removes the diffuse low-column outer disk (below SigmaHIcrit, in Msun/pc^2) that
peer H2-based SAMs (Shark, DarkSage) treat as ionised. This overlays the z=0 HIMF
for a sweep of SigmaHIcrit against Zwaan+05 and Jones+18, and shows the SMF is
invariant (the cut only relabels HI; it never touches ColdGas, H2 or SF).

Expects the sweep runs:
    output/mill_ionOff (cut off), mill_ion05/10/20 (SigmaHIcrit = 0.5/1.0/2.0)

    python3 plotting/compare_hi_ionization.py
"""
import os
os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')
import glob
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SNAP = 'Snap_63'   # z = 0
BINW = 0.2
RUNS = [('mill_ionOff', 'cut OFF (all cold H -> HI)', None, 'k'),
        ('mill_ion05',  r'$\Sigma_{\rm crit}=0.5$', 0.5, 'C0'),
        ('mill_ion10',  r'$\Sigma_{\rm crit}=1.0$', 1.0, 'C1'),
        ('mill_ion20',  r'$\Sigma_{\rm crit}=2.0$', 2.0, 'C2')]


def load(name):
    fs = sorted(glob.glob(f'output/{name}/model_*.hdf5'))
    if not fs:
        return None
    cols = ('StellarMass', 'H1gas', 'ColdGas', 'H2gas', 'Type')
    acc = {k: [] for k in cols}
    with h5py.File(fs[0], 'r') as f:
        h = f['Header']['Simulation'].attrs['hubble_h']
        box = f['Header']['Simulation'].attrs['box_size']
        frac = f['Header']['Runtime'].attrs['frac_volume_processed']
    for fn in fs:
        with h5py.File(fn, 'r') as f:
            g = f[SNAP]
            for k in cols:
                acc[k].append(g[k][:])
    d = {k: np.concatenate(v) for k, v in acc.items()}
    d['h'] = float(h)
    d['vol'] = (float(box) / float(h)) ** 3 * float(frac)   # Mpc^3
    return d


def mass_function(mass_msun, vol):
    lg = np.log10(mass_msun[mass_msun > 0])
    edges = np.arange(7.0, 11.6, BINW)
    n, _ = np.histogram(lg, bins=edges)
    xc = 0.5 * (edges[:-1] + edges[1:])
    phi = n / (vol * BINW)
    return xc, phi


def main():
    fig, (axH, axS) = plt.subplots(1, 2, figsize=(14, 6))

    # observations (convert to h=0.73 not attempted; shown as published)
    z = np.loadtxt('data/Gas/HIMF_Zwaan2005.dat')
    axH.errorbar(z[:, 0], z[:, 1], yerr=[z[:, 3], z[:, 2]], fmt='ks', ms=4,
                 capsize=2, label='Zwaan+05 (HIPASS)', zorder=5)
    j = np.loadtxt('data/Gas/HIMF_Jones18.dat')
    axH.plot(j[:, 0], j[:, 1], 'v', color='0.4', ms=4, label='Jones+18 (ALFALFA)', zorder=4)

    for name, lab, sig, col in RUNS:
        r = load(name)
        if r is None:
            print(f'  (skip {name}: no output)'); continue
        h = r['h']
        # HIMF
        xh, phih = mass_function(r['H1gas'] * 1e10 / h, r['vol'])
        m = phih > 0
        axH.plot(xh[m], np.log10(phih[m]), '-', color=col, lw=2, label=lab)
        # SMF
        xs, phis = mass_function(r['StellarMass'] * 1e10 / h, r['vol'])
        ms = phis > 0
        axS.plot(xs[ms], np.log10(phis[ms]), '-', color=col, lw=2, label=lab)
        totHI = (r['H1gas'] * 1e10 / h).sum()
        print(f'  {name:12} total HI = {totHI:.3e} Msun')

    axH.set(xlabel=r'$\log_{10}(M_{\rm HI}/M_\odot)$',
            ylabel=r'$\log_{10}(\phi\,/\,{\rm Mpc^{-3}\,dex^{-1}})$',
            xlim=(7.5, 11.2), ylim=(-5.5, -0.5))
    axH.set_title('HI mass function -- ionisation cut trims the high-mass tail')
    axH.legend(fontsize=9); axH.grid(alpha=0.3)

    axS.set(xlabel=r'$\log_{10}(M_*/M_\odot)$',
            ylabel=r'$\log_{10}(\phi\,/\,{\rm Mpc^{-3}\,dex^{-1}})$',
            xlim=(8.0, 12.0), ylim=(-5.5, -0.5))
    axS.set_title('Stellar mass function -- invariant (cut only relabels HI)')
    axS.legend(fontsize=9); axS.grid(alpha=0.3)

    fig.suptitle('HIIonizationOn sensitivity: removing the ionised outer disk (Shark-style)', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig('hi_ionization_sweep.png', dpi=150, bbox_inches='tight')
    print('Wrote hi_ionization_sweep.png')


if __name__ == '__main__':
    main()

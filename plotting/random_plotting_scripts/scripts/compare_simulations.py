#!/usr/bin/env python
"""
Compare SAGE26 outputs from two different N-body simulations (e.g., miniMillennium vs microUchuu).
Produces diagnostic plots comparing galaxy population statistics at z~0.

Usage:
    python scripts/compare_simulations.py
    python scripts/compare_simulations.py --sim1 output/millennium/model_*.hdf5 --sim2 output/microuchuu/model_*.hdf5
"""

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import quad as _quad
import os
import glob
import argparse

import warnings
warnings.filterwarnings("ignore")

# Plot style
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 12
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'

OutputFormat = '.pdf'

# Minimum halo particle length for halo-based comparisons
HALO_LEN_MIN = 20

# sSFR cut for quiescent/star-forming classification
sSFRcut = -11.0


# ========== I/O helpers ==========

def read_simulation_params(filepath):
    params = {}
    with h5.File(filepath, 'r') as f:
        sim = f['Header/Simulation']
        params['Hubble_h'] = float(sim.attrs['hubble_h'])
        params['BoxSize'] = float(sim.attrs['box_size'])
        params['Omega'] = float(sim.attrs['omega_matter'])
        params['OmegaLambda'] = float(sim.attrs['omega_lambda'])
        params['PartMass'] = float(sim.attrs['particle_mass'])

        runtime = f['Header/Runtime']
        params['VolumeFraction'] = float(runtime.attrs['frac_volume_processed'])

        params['snapshot_redshifts'] = np.array(f['Header/snapshot_redshifts'])

        snap_groups = [key for key in f.keys() if key.startswith('Snap_')]
        snap_numbers = sorted([int(s.replace('Snap_', '')) for s in snap_groups])
        params['available_snapshots'] = snap_numbers
        params['latest_snapshot'] = max(snap_numbers) if snap_numbers else None

    return params


def read_hdf(filepaths, snap_num, param):
    data_list = []
    for filepath in filepaths:
        with h5.File(filepath, 'r') as f:
            if snap_num in f and param in f[snap_num]:
                data = np.array(f[snap_num][param])
                if data.size > 0:
                    data_list.append(data)
    if not data_list:
        return np.array([])
    return np.concatenate(data_list)


def load_sim(file_pattern, label):
    """Load a simulation's z~0 galaxy data. Returns dict with physical-unit arrays."""
    file_list = sorted(glob.glob(file_pattern))
    if not file_list:
        raise FileNotFoundError(f"No files matching: {file_pattern}")

    params = read_simulation_params(file_list[0])
    total_vf = sum(read_simulation_params(f)['VolumeFraction'] for f in file_list)
    params['VolumeFraction'] = total_vf

    h = params['Hubble_h']
    box = params['BoxSize']
    volume = (box / h) ** 3.0 * total_vf  # comoving Mpc^3

    snap = params['latest_snapshot']
    snap_key = f'Snap_{snap}'
    z = params['snapshot_redshifts'][snap]

    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"  Files: {len(file_list)}, Snapshot: {snap}, z = {z:.4f}")
    print(f"  Box = {box} h^-1 Mpc, h = {h}, PartMass = {params['PartMass']:.4f} (1e10 h^-1 Msun)")
    print(f"  Volume = {volume:.1f} Mpc^3")

    def read(param):
        return read_hdf(file_list, snap_key, param)

    def read_mass(param):
        return read(param) * 1.0e10 / h

    d = {}
    d['label'] = label
    d['params'] = params
    d['volume'] = volume
    d['h'] = h
    d['z'] = z
    d['mass_resolution'] = params['PartMass'] * 1.0e10 / h  # Msun per particle

    # Galaxy properties in physical (Msun, etc.)
    d['StellarMass'] = read_mass('StellarMass')
    d['BulgeMass'] = read_mass('BulgeMass')
    d['ColdGas'] = read_mass('ColdGas')
    d['HotGas'] = read_mass('HotGas')
    d['EjectedMass'] = read_mass('EjectedMass')
    d['CGMgas'] = read_mass('CGMgas')
    d['IntraClusterStars'] = read_mass('IntraClusterStars')
    d['BlackHoleMass'] = read_mass('BlackHoleMass')
    d['MetalsStellarMass'] = read_mass('MetalsStellarMass')
    d['MetalsColdGas'] = read_mass('MetalsColdGas')
    d['MetalsHotGas'] = read_mass('MetalsHotGas')
    d['Mvir'] = read_mass('Mvir')
    d['CentralMvir'] = read_mass('CentralMvir')
    d['H1gas'] = read_mass('H1gas')
    d['H2gas'] = read_mass('H2gas')

    d['SfrDisk'] = read('SfrDisk')
    d['SfrBulge'] = read('SfrBulge')
    d['Type'] = read('Type')
    d['Vvir'] = read('Vvir')
    d['Vmax'] = read('Vmax')
    d['Rvir'] = read('Rvir')
    d['Concentration'] = read('Concentration')
    d['Posx'] = read('Posx')
    d['Posy'] = read('Posy')
    d['Posz'] = read('Posz')
    d['GalaxyIndex'] = read('GalaxyIndex')
    d['CentralGalaxyIndex'] = read('CentralGalaxyIndex')
    d['Len'] = read('Len')
    d['Regime'] = read('Regime')
    d['FFBRegime'] = read('FFBRegime')
    d['Cooling'] = read('Cooling')
    d['Heating'] = read('Heating')
    d['RcoolToRvir'] = read('RcoolToRvir')
    d['tcool_over_tff'] = read('tcool_over_tff')
    d['QuasarModeBHaccretionMass'] = read_mass('QuasarModeBHaccretionMass')
    d['mdot_cool'] = read('mdot_cool')
    d['MetalsHotGas'] = read_mass('MetalsHotGas')
    d['MetalsCGMgas'] = read_mass('MetalsCGMgas')

    n = len(d['StellarMass'])
    print(f"  Galaxies at z~0: {n}")
    print(f"  Mass resolution: {d['mass_resolution']:.2e} Msun/particle")
    return d


# ========== Plotting functions ==========

def _mass_function(mass_array, volume, binwidth=0.1, mmin=6.0, mmax=13.0):
    """Return bin centres and log10(phi) for a mass function."""
    w = mass_array > 0
    logm = np.log10(mass_array[w])
    bins = np.arange(mmin, mmax + binwidth, binwidth)
    counts, edges = np.histogram(logm, bins=bins)
    centres = edges[:-1] + 0.5 * binwidth
    with np.errstate(divide='ignore'):
        phi = np.log10(counts / volume / binwidth)
    phi[~np.isfinite(phi)] = np.nan
    return centres, phi


def plot_smf(ax, sim1, sim2):
    """Stellar mass function comparison."""
    bw = 0.1
    for sim, color, ls in [(sim1, 'C0', '-'), (sim2, 'C1', '--')]:
        x, phi = _mass_function(sim['StellarMass'], sim['volume'], bw)
        ax.plot(x, phi, color=color, ls=ls, lw=2.5, label=sim['label'] + ' (all)')

        # Star-forming
        w = sim['StellarMass'] > 0
        sfr = sim['SfrDisk'][w] + sim['SfrBulge'][w]
        ssfr = np.log10(sfr / sim['StellarMass'][w])
        sf_mask = ssfr > sSFRcut
        x_sf, phi_sf = _mass_function(sim['StellarMass'][w][sf_mask], sim['volume'], bw)
        ax.plot(x_sf, phi_sf, color=color, ls=ls, lw=1.2, alpha=0.6)

        # Quiescent
        q_mask = ssfr < sSFRcut
        x_q, phi_q = _mass_function(sim['StellarMass'][w][q_mask], sim['volume'], bw)
        ax.plot(x_q, phi_q, color=color, ls=':', lw=1.2, alpha=0.6)

    ax.set_xlabel(r'$\log_{10}(M_\star\;/\;\mathrm{M_\odot})$')
    ax.set_ylabel(r'$\log_{10}(\Phi\;/\;\mathrm{Mpc^{-3}\,dex^{-1}})$')
    ax.set_title('Stellar Mass Function')
    ax.set_xlim(7, 12.5)
    ax.set_ylim(-6, 0)
    ax.legend(fontsize=9)


def plot_hmf(ax, sim1, sim2):
    """Halo mass function (centrals only)."""
    bw = 0.15
    for sim, color, ls in [(sim1, 'C0', '-'), (sim2, 'C1', '--')]:
        centrals = (sim['Type'] == 0) & (sim['Len'] >= HALO_LEN_MIN)
        x, phi = _mass_function(sim['Mvir'][centrals], sim['volume'], bw, mmin=9, mmax=15)
        ax.plot(x, phi, color=color, ls=ls, lw=2.5, label=sim['label'])
    ax.set_xlabel(r'$\log_{10}(M_\mathrm{vir}\;/\;\mathrm{M_\odot})$')
    ax.set_ylabel(r'$\log_{10}(\Phi\;/\;\mathrm{Mpc^{-3}\,dex^{-1}})$')
    ax.set_title('Halo Mass Function (centrals)')
    ax.set_xlim(9.5, 15)
    ax.set_ylim(-6, 0)
    ax.legend(fontsize=9)


def plot_smhm(ax, sim1, sim2):
    """Stellar-to-halo mass relation (centrals only)."""
    for sim, color, ls in [(sim1, 'C0', '-'), (sim2, 'C1', '--')]:
        centrals = (sim['Type'] == 0) & (sim['Len'] >= HALO_LEN_MIN) & (sim['Mvir'] > 0) & (sim['StellarMass'] > 0)
        logmh = np.log10(sim['Mvir'][centrals])
        ratio = sim['StellarMass'][centrals] / sim['Mvir'][centrals]

        bins = np.arange(10, 15, 0.2)
        idx = np.digitize(logmh, bins)
        medians = []
        p16, p84 = [], []
        centres = []
        for i in range(1, len(bins)):
            sel = idx == i
            if sel.sum() > 10:
                r = ratio[sel]
                medians.append(np.median(r))
                p16.append(np.percentile(r, 16))
                p84.append(np.percentile(r, 84))
                centres.append(0.5 * (bins[i - 1] + bins[i]))

        centres = np.array(centres)
        medians = np.array(medians)
        p16 = np.array(p16)
        p84 = np.array(p84)

        ax.plot(centres, np.log10(medians), color=color, ls=ls, lw=2.5, label=sim['label'])
        ax.fill_between(centres, np.log10(p16), np.log10(p84), color=color, alpha=0.15)

    ax.set_xlabel(r'$\log_{10}(M_\mathrm{vir}\;/\;\mathrm{M_\odot})$')
    ax.set_ylabel(r'$\log_{10}(M_\star / M_\mathrm{vir})$')
    ax.set_title('SMHM Relation (centrals)')
    ax.set_xlim(10, 14.5)
    ax.set_ylim(-3.5, -0.5)
    ax.legend(fontsize=9)


def plot_ssfr(ax, sim1, sim2):
    """Specific SFR vs stellar mass."""
    for sim, color in [(sim1, 'C0'), (sim2, 'C1')]:
        w = (sim['StellarMass'] > 1e7) & ((sim['SfrDisk'] + sim['SfrBulge']) > 0)
        sm = np.log10(sim['StellarMass'][w])
        ssfr = np.log10((sim['SfrDisk'][w] + sim['SfrBulge'][w]) / sim['StellarMass'][w])

        # Subsample for scatter
        n_samp = min(5000, w.sum())
        idx = np.random.choice(w.sum(), n_samp, replace=False)
        ax.scatter(sm[idx], ssfr[idx], s=1, alpha=0.15, color=color, rasterized=True)

        # Median trend
        bins = np.arange(7, 12.5, 0.3)
        dig = np.digitize(sm, bins)
        med_x, med_y = [], []
        for i in range(1, len(bins)):
            sel = dig == i
            if sel.sum() > 10:
                med_x.append(0.5 * (bins[i - 1] + bins[i]))
                med_y.append(np.median(ssfr[sel]))
        ax.plot(med_x, med_y, color=color, lw=2.5, label=sim['label'])

    ax.axhline(sSFRcut, color='grey', ls=':', lw=1, alpha=0.7)
    ax.set_xlabel(r'$\log_{10}(M_\star\;/\;\mathrm{M_\odot})$')
    ax.set_ylabel(r'$\log_{10}(\mathrm{sSFR}\;/\;\mathrm{yr^{-1}})$')
    ax.set_title('Specific SFR vs Stellar Mass')
    ax.set_xlim(7, 12.5)
    ax.set_ylim(-14, -8)
    ax.legend(fontsize=9)


def plot_quenched_fraction(ax, sim1, sim2):
    """Quenched fraction vs stellar mass."""
    for sim, color, ls in [(sim1, 'C0', '-'), (sim2, 'C1', '--')]:
        w = sim['StellarMass'] > 0
        sm = np.log10(sim['StellarMass'][w])
        sfr = sim['SfrDisk'][w] + sim['SfrBulge'][w]
        ssfr = np.log10(sfr / sim['StellarMass'][w])

        bins = np.arange(8, 12.5, 0.3)
        dig = np.digitize(sm, bins)
        frac_x, frac_y = [], []
        for i in range(1, len(bins)):
            sel = dig == i
            if sel.sum() > 20:
                frac_x.append(0.5 * (bins[i - 1] + bins[i]))
                frac_y.append((ssfr[sel] < sSFRcut).sum() / sel.sum())

        ax.plot(frac_x, frac_y, color=color, ls=ls, lw=2.5, marker='o', ms=4, label=sim['label'])

    ax.set_xlabel(r'$\log_{10}(M_\star\;/\;\mathrm{M_\odot})$')
    ax.set_ylabel('Quenched Fraction')
    ax.set_title('Quenched Fraction vs Stellar Mass')
    ax.set_xlim(8, 12.5)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)


def plot_gas_fraction(ax, sim1, sim2):
    """Cold gas fraction vs stellar mass."""
    for sim, color, ls in [(sim1, 'C0', '-'), (sim2, 'C1', '--')]:
        w = (sim['StellarMass'] > 1e7) & (sim['ColdGas'] > 0)
        sm = np.log10(sim['StellarMass'][w])
        fgas = sim['ColdGas'][w] / (sim['StellarMass'][w] + sim['ColdGas'][w])

        bins = np.arange(7, 12.5, 0.3)
        dig = np.digitize(sm, bins)
        med_x, med_y, p16, p84 = [], [], [], []
        for i in range(1, len(bins)):
            sel = dig == i
            if sel.sum() > 10:
                med_x.append(0.5 * (bins[i - 1] + bins[i]))
                med_y.append(np.median(fgas[sel]))
                p16.append(np.percentile(fgas[sel], 16))
                p84.append(np.percentile(fgas[sel], 84))

        med_x = np.array(med_x)
        med_y = np.array(med_y)
        ax.plot(med_x, med_y, color=color, ls=ls, lw=2.5, label=sim['label'])
        ax.fill_between(med_x, p16, p84, color=color, alpha=0.12)

    ax.set_xlabel(r'$\log_{10}(M_\star\;/\;\mathrm{M_\odot})$')
    ax.set_ylabel(r'$f_\mathrm{gas} = M_\mathrm{cold} / (M_\star + M_\mathrm{cold})$')
    ax.set_title('Cold Gas Fraction')
    ax.set_xlim(7, 12.5)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)


def plot_bh_bulge(ax, sim1, sim2):
    """Black hole mass vs bulge mass."""
    for sim, color in [(sim1, 'C0'), (sim2, 'C1')]:
        w = (sim['BlackHoleMass'] > 0) & (sim['BulgeMass'] > 1e7)
        bm = np.log10(sim['BulgeMass'][w])
        bh = np.log10(sim['BlackHoleMass'][w])

        n_samp = min(5000, w.sum())
        idx = np.random.choice(w.sum(), n_samp, replace=False)
        ax.scatter(bm[idx], bh[idx], s=1, alpha=0.15, color=color, rasterized=True)

        # Median trend
        bins = np.arange(7, 13, 0.3)
        dig = np.digitize(bm, bins)
        med_x, med_y = [], []
        for i in range(1, len(bins)):
            sel = dig == i
            if sel.sum() > 10:
                med_x.append(0.5 * (bins[i - 1] + bins[i]))
                med_y.append(np.median(bh[sel]))
        ax.plot(med_x, med_y, color=color, lw=2.5, label=sim['label'])

    ax.set_xlabel(r'$\log_{10}(M_\mathrm{bulge}\;/\;\mathrm{M_\odot})$')
    ax.set_ylabel(r'$\log_{10}(M_\mathrm{BH}\;/\;\mathrm{M_\odot})$')
    ax.set_title('BH–Bulge Relation')
    ax.set_xlim(7, 13)
    ax.set_ylim(4, 11)
    ax.legend(fontsize=9)


def plot_metallicity(ax, sim1, sim2):
    """Stellar mass–metallicity relation."""
    Zsun = 0.02
    for sim, color, ls in [(sim1, 'C0', '-'), (sim2, 'C1', '--')]:
        w = (sim['StellarMass'] > 1e7) & (sim['MetalsStellarMass'] > 0)
        sm = np.log10(sim['StellarMass'][w])
        Z = np.log10(sim['MetalsStellarMass'][w] / sim['StellarMass'][w] / Zsun)

        bins = np.arange(7, 12.5, 0.3)
        dig = np.digitize(sm, bins)
        med_x, med_y, p16, p84 = [], [], [], []
        for i in range(1, len(bins)):
            sel = dig == i
            if sel.sum() > 10:
                med_x.append(0.5 * (bins[i - 1] + bins[i]))
                med_y.append(np.median(Z[sel]))
                p16.append(np.percentile(Z[sel], 16))
                p84.append(np.percentile(Z[sel], 84))

        med_x = np.array(med_x)
        ax.plot(med_x, med_y, color=color, ls=ls, lw=2.5, label=sim['label'])
        ax.fill_between(med_x, p16, p84, color=color, alpha=0.12)

    ax.set_xlabel(r'$\log_{10}(M_\star\;/\;\mathrm{M_\odot})$')
    ax.set_ylabel(r'$\log_{10}(Z_\star / Z_\odot)$')
    ax.set_title('Mass–Metallicity Relation')
    ax.set_xlim(7, 12.5)
    ax.set_ylim(-2.5, 1)
    ax.legend(fontsize=9)


def plot_type_fractions(ax, sim1, sim2):
    """Central/satellite/orphan fractions vs stellar mass."""
    type_labels = {0: 'Central', 1: 'Satellite', 2: 'Orphan'}
    type_colors = {0: 'tab:blue', 1: 'tab:orange', 2: 'tab:green'}

    for sim, ls, marker in [(sim1, '-', 'o'), (sim2, '--', 's')]:
        w = sim['StellarMass'] > 0
        sm = np.log10(sim['StellarMass'][w])
        types = sim['Type'][w]

        bins = np.arange(8, 12.5, 0.3)
        dig = np.digitize(sm, bins)

        for t in [0, 1, 2]:
            frac_x, frac_y = [], []
            for i in range(1, len(bins)):
                sel = dig == i
                if sel.sum() > 20:
                    frac_x.append(0.5 * (bins[i - 1] + bins[i]))
                    frac_y.append((types[sel] == t).sum() / sel.sum())
            lbl = f"{sim['label']} {type_labels[t]}" if ls == '-' else None
            ax.plot(frac_x, frac_y, color=type_colors[t], ls=ls, lw=2,
                    marker=marker, ms=3, label=lbl)

    ax.set_xlabel(r'$\log_{10}(M_\star\;/\;\mathrm{M_\odot})$')
    ax.set_ylabel('Fraction')
    ax.set_title('Galaxy Type Fractions')
    ax.set_xlim(8, 12.5)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, ncol=2)


def plot_cold_gas_mf(ax, sim1, sim2):
    """Cold gas mass function."""
    bw = 0.15
    for sim, color, ls in [(sim1, 'C0', '-'), (sim2, 'C1', '--')]:
        x, phi = _mass_function(sim['ColdGas'], sim['volume'], bw, mmin=6, mmax=12)
        ax.plot(x, phi, color=color, ls=ls, lw=2.5, label=sim['label'])
    ax.set_xlabel(r'$\log_{10}(M_\mathrm{cold}\;/\;\mathrm{M_\odot})$')
    ax.set_ylabel(r'$\log_{10}(\Phi\;/\;\mathrm{Mpc^{-3}\,dex^{-1}})$')
    ax.set_title('Cold Gas Mass Function')
    ax.set_xlim(6.5, 11.5)
    ax.set_ylim(-6, 0)
    ax.legend(fontsize=9)


def plot_h1_mf(ax, sim1, sim2):
    """H I mass function.

    Uses the definition requested for this comparison run:
      M_HI = ColdGas - H2gas
    (and keeps only M_HI > 0).
    """
    bw = 0.2
    for sim, color, ls in [(sim1, 'C0', '-'), (sim2, 'C1', '--')]:
        mhi = sim['ColdGas'] - sim['H2gas']
        mhi = np.where(mhi > 0, mhi, 0.0)
        x, phi = _mass_function(mhi, sim['volume'], bw, mmin=6, mmax=12)
        ax.plot(x, phi, color=color, ls=ls, lw=2.5, label=sim['label'])
    ax.set_xlabel(r'$\log_{10}(M_\mathrm{H\,I}\;/\;\mathrm{M_\odot})$')
    ax.set_ylabel(r'$\log_{10}(\Phi\;/\;\mathrm{Mpc^{-3}\,dex^{-1}})$')
    ax.set_title(r'H I Mass Function (ColdGas$-$H$_2$)')
    ax.set_xlim(6.5, 11.8)
    ax.set_ylim(-6, 0)
    ax.legend(fontsize=9)


def plot_h2_mf(ax, sim1, sim2):
    """H2 mass function."""
    bw = 0.2
    for sim, color, ls in [(sim1, 'C0', '-'), (sim2, 'C1', '--')]:
        x, phi = _mass_function(sim['H2gas'], sim['volume'], bw, mmin=6, mmax=12)
        ax.plot(x, phi, color=color, ls=ls, lw=2.5, label=sim['label'])
    ax.set_xlabel(r'$\log_{10}(M_{\mathrm{H}_2}\;/\;\mathrm{M_\odot})$')
    ax.set_ylabel(r'$\log_{10}(\Phi\;/\;\mathrm{Mpc^{-3}\,dex^{-1}})$')
    ax.set_title(r'H$_2$ Mass Function')
    ax.set_xlim(6.5, 11.8)
    ax.set_ylim(-6, 0)
    ax.legend(fontsize=9)


def plot_bh_mass_function(ax, sim1, sim2):
    """Black hole mass function."""
    bw = 0.25
    for sim, color, ls in [(sim1, 'C0', '-'), (sim2, 'C1', '--')]:
        x, phi = _mass_function(sim['BlackHoleMass'], sim['volume'], bw, mmin=4.0, mmax=10.5)
        ax.plot(x, phi, color=color, ls=ls, lw=2.5, label=sim['label'])
    ax.set_xlabel(r'$\log_{10}(M_\mathrm{BH}\;/\;\mathrm{M_\odot})$')
    ax.set_ylabel(r'$\log_{10}(\Phi\;/\;\mathrm{Mpc^{-3}\,dex^{-1}})$')
    ax.set_title('Black Hole Mass Function')
    ax.set_xlim(4.5, 10.5)
    ax.set_ylim(-7, 0)
    ax.legend(fontsize=9)


def _periodic_delta(dx, boxsize):
    """Minimum-image convention for periodic boxes."""
    return dx - boxsize * np.round(dx / boxsize)


def _central_property_for_each_galaxy(sim, prop):
    """Return an array giving each galaxy its central's `prop`.

    Looks up the galaxy with `GalaxyIndex == CentralGalaxyIndex` and copies its
    property values to all members. Unmatched centrals return NaN.
    """
    gal_idx = sim['GalaxyIndex'].astype(np.int64)
    cen_idx = sim['CentralGalaxyIndex'].astype(np.int64)
    values = sim[prop]

    order = np.argsort(gal_idx)
    gal_sorted = gal_idx[order]
    val_sorted = values[order]

    loc = np.searchsorted(gal_sorted, cen_idx)
    valid = (loc >= 0) & (loc < gal_sorted.size) & (gal_sorted[loc] == cen_idx)
    out = np.full(gal_idx.size, np.nan, dtype=float)
    out[valid] = val_sorted[loc[valid]]
    return out


def _group_central_arrays(sim):
    """Return per-group arrays keyed by unique CentralGalaxyIndex.

    Output dict contains (all length = n_groups):
      - cen_values: the unique central GalaxyIndex IDs
      - mvir: central halo virial mass (Msun)
      - rvir: central Rvir (code units, typically Mpc/h)
      - mstar_bcg: central stellar mass (Msun)
      - mics: central intra-cluster stars mass (Msun)
      - n_members_gt: member counts above threshold (set by caller)
      - n_sat_gt: satellite counts above threshold (set by caller)
      - mstar_total: total member stellar mass (Msun)
    """
    gal_idx = sim['GalaxyIndex'].astype(np.int64)
    cen_idx = sim['CentralGalaxyIndex'].astype(np.int64)
    inv_cen_vals, inv = np.unique(cen_idx, return_inverse=True)

    # Total stellar mass of all members (including BCG)
    mstar_total = np.bincount(inv, weights=sim['StellarMass'], minlength=inv_cen_vals.size)

    # Identify BCG record (central galaxy)
    is_bcg = (sim['Type'] == 0) & (gal_idx == cen_idx) & (sim['Len'] >= HALO_LEN_MIN)
    g = inv[is_bcg]
    mvir = np.full(inv_cen_vals.size, np.nan)
    rvir = np.full(inv_cen_vals.size, np.nan)
    mstar_bcg = np.full(inv_cen_vals.size, np.nan)
    mics = np.full(inv_cen_vals.size, np.nan)
    if g.size > 0:
        mvir[g] = sim['Mvir'][is_bcg]
        rvir[g] = sim['Rvir'][is_bcg]
        mstar_bcg[g] = sim['StellarMass'][is_bcg]
        mics[g] = sim['IntraClusterStars'][is_bcg]

    return {
        'cen_values': inv_cen_vals,
        'inv': inv,
        'mvir': mvir,
        'rvir': rvir,
        'mstar_bcg': mstar_bcg,
        'mics': mics,
        'mstar_total': mstar_total,
    }


def plot_group_satellite_occupation(ax, sim1, sim2, mstar_thresh=1e9):
    """Mean satellite occupation vs halo mass (centrals/groups)."""
    bins = np.arange(11.5, 15.1, 0.25)
    for sim, color, ls in [(sim1, 'C0', '-'), (sim2, 'C1', '--')]:
        grp = _group_central_arrays(sim)
        inv = grp['inv']
        gal_idx = sim['GalaxyIndex'].astype(np.int64)
        cen_idx = sim['CentralGalaxyIndex'].astype(np.int64)

        is_sat = gal_idx != cen_idx
        sat_gt = is_sat & (sim['StellarMass'] >= mstar_thresh)
        n_sat = np.bincount(inv[sat_gt], minlength=grp['cen_values'].size)

        w = np.isfinite(grp['mvir'])
        logmh = np.log10(grp['mvir'][w])
        ns = n_sat[w].astype(float)

        dig = np.digitize(logmh, bins)
        cx, my, lo, hi = [], [], [], []
        for i in range(1, len(bins)):
            sel = dig == i
            if sel.sum() >= 10:
                cx.append(0.5 * (bins[i - 1] + bins[i]))
                my.append(np.mean(ns[sel]))
                lo.append(np.percentile(ns[sel], 16))
                hi.append(np.percentile(ns[sel], 84))

        cx = np.array(cx)
        ax.plot(cx, my, color=color, ls=ls, lw=2.5, label=sim['label'])
        ax.fill_between(cx, lo, hi, color=color, alpha=0.12)

    ax.set_xlabel(r'$\log_{10}(M_\mathrm{vir}\;/\;\mathrm{M_\odot})$')
    ax.set_ylabel(r'$\langle N_\mathrm{sat} \rangle\,(M_\star \geq 10^9\,\mathrm{M_\odot})$')
    ax.set_title('Satellite Occupation (groups/clusters)')
    ax.set_xlim(11.5, 15.0)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)


def plot_bcg_mass_vs_mvir(ax, sim1, sim2):
    """BCG stellar mass vs halo mass."""
    bins = np.arange(11.5, 15.1, 0.25)
    for sim, color, ls in [(sim1, 'C0', '-'), (sim2, 'C1', '--')]:
        grp = _group_central_arrays(sim)
        w = np.isfinite(grp['mvir']) & (grp['mstar_bcg'] > 0)
        logmh = np.log10(grp['mvir'][w])
        logmbcg = np.log10(grp['mstar_bcg'][w])
        cx, my, lo, hi = _binned_stat(logmh, logmbcg, bins, min_count=10)
        ax.plot(cx, my, color=color, ls=ls, lw=2.5, label=sim['label'])
        ax.fill_between(cx, lo, hi, color=color, alpha=0.12)

    ax.set_xlabel(r'$\log_{10}(M_\mathrm{vir}\;/\;\mathrm{M_\odot})$')
    ax.set_ylabel(r'$\log_{10}(M_{\star,\mathrm{BCG}}\;/\;\mathrm{M_\odot})$')
    ax.set_title('BCG Stellar Mass vs Halo Mass')
    ax.set_xlim(11.5, 15.0)
    ax.legend(fontsize=9)


def plot_bcg_fraction_of_group_stellar_mass(ax, sim1, sim2):
    """BCG fraction of total member stellar mass (by CentralGalaxyIndex membership)."""
    bins = np.arange(11.5, 15.1, 0.25)
    for sim, color, ls in [(sim1, 'C0', '-'), (sim2, 'C1', '--')]:
        grp = _group_central_arrays(sim)
        w = np.isfinite(grp['mvir']) & (grp['mstar_bcg'] > 0) & (grp['mstar_total'] > 0)
        logmh = np.log10(grp['mvir'][w])
        f = grp['mstar_bcg'][w] / grp['mstar_total'][w]
        cx, my, lo, hi = _binned_stat(logmh, f, bins, min_count=10)
        ax.plot(cx, my, color=color, ls=ls, lw=2.5, label=sim['label'])
        ax.fill_between(cx, lo, hi, color=color, alpha=0.12)

    ax.set_xlabel(r'$\log_{10}(M_\mathrm{vir}\;/\;\mathrm{M_\odot})$')
    ax.set_ylabel(r'$M_{\star,\mathrm{BCG}} / \sum M_{\star,\mathrm{members}}$')
    ax.set_title('BCG Stellar Mass Fraction (members)')
    ax.set_xlim(11.5, 15.0)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)


def plot_bcg_mass_gap(ax, sim1, sim2, mstar_min=1e9):
    """BCG-to-most-massive-satellite stellar mass gap."""
    bins = np.arange(11.5, 15.1, 0.25)
    for sim, color, ls in [(sim1, 'C0', '-'), (sim2, 'C1', '--')]:
        grp = _group_central_arrays(sim)
        inv = grp['inv']

        gal_idx = sim['GalaxyIndex'].astype(np.int64)
        cen_idx = sim['CentralGalaxyIndex'].astype(np.int64)
        is_sat = gal_idx != cen_idx
        w_sat = is_sat & (sim['StellarMass'] >= mstar_min)

        # Sort satellites within each group by stellar mass descending
        inv_s = inv[w_sat]
        ms_s = sim['StellarMass'][w_sat]
        if ms_s.size == 0:
            continue
        order = np.lexsort((-ms_s, inv_s))
        inv_sorted = inv_s[order]
        ms_sorted = ms_s[order]
        # first satellite (most massive) for each group
        first = np.r_[True, inv_sorted[1:] != inv_sorted[:-1]]
        g_ids = inv_sorted[first]
        m2 = ms_sorted[first]

        m2_by_group = np.full(grp['cen_values'].size, np.nan)
        m2_by_group[g_ids] = m2

        w = np.isfinite(grp['mvir']) & (grp['mstar_bcg'] > 0) & np.isfinite(m2_by_group)
        logmh = np.log10(grp['mvir'][w])
        gap = np.log10(grp['mstar_bcg'][w]) - np.log10(m2_by_group[w])
        cx, my, lo, hi = _binned_stat(logmh, gap, bins, min_count=10)
        ax.plot(cx, my, color=color, ls=ls, lw=2.5, label=sim['label'])
        ax.fill_between(cx, lo, hi, color=color, alpha=0.12)

    ax.set_xlabel(r'$\log_{10}(M_\mathrm{vir}\;/\;\mathrm{M_\odot})$')
    ax.set_ylabel(r'$\log_{10}(M_{\star,\mathrm{BCG}}) - \log_{10}(M_{\star,2})$')
    ax.set_title('BCG–2nd Member Mass Gap')
    ax.set_xlim(11.5, 15.0)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)


def plot_satellite_conditional_smf(ax, sim1, sim2, mvir_bins=((13.0, 14.0), (14.0, 15.0))):
    """Satellite conditional SMF: satellites per halo per dex in group/cluster mass bins."""
    bw = 0.2
    bins = np.arange(8.0, 12.5 + bw, bw)
    centres = bins[:-1] + 0.5 * bw

    for sim, color, base_ls in [(sim1, 'C0', '-'), (sim2, 'C1', '--')]:
        grp = _group_central_arrays(sim)
        inv = grp['inv']
        gal_idx = sim['GalaxyIndex'].astype(np.int64)
        cen_idx = sim['CentralGalaxyIndex'].astype(np.int64)
        is_sat = gal_idx != cen_idx

        # Central Mvir for each galaxy via its group index
        mvir_group = grp['mvir']
        mvir_for_gal = mvir_group[inv]

        for (mlo, mhi), ls in zip(mvir_bins, ['-', ':']):
            in_bin_group = np.isfinite(mvir_group) & (np.log10(mvir_group) >= mlo) & (np.log10(mvir_group) < mhi)
            n_halo = in_bin_group.sum()
            if n_halo < 10:
                continue

            w = is_sat & np.isfinite(mvir_for_gal) & (np.log10(mvir_for_gal) >= mlo) & (np.log10(mvir_for_gal) < mhi) & (sim['StellarMass'] > 0)
            logm = np.log10(sim['StellarMass'][w])
            counts, _ = np.histogram(logm, bins=bins)
            phi = counts / n_halo / bw  # satellites per halo per dex
            with np.errstate(divide='ignore'):
                y = np.log10(phi)
            y[~np.isfinite(y)] = np.nan
            ax.plot(centres, y, color=color, ls=base_ls if ls == '-' else (0, (1, 2)), lw=2.2,
                    label=f"{sim['label']}  {mlo:.0f}–{mhi:.0f}")

    ax.set_xlabel(r'$\log_{10}(M_{\star,\mathrm{sat}}\;/\;\mathrm{M_\odot})$')
    ax.set_ylabel(r'$\log_{10}(\mathrm{d}N_\mathrm{sat}/\mathrm{d}\log M_\star)$')
    ax.set_title('Satellite Conditional SMF (per halo)')
    ax.set_xlim(8.5, 12.2)
    ax.set_ylim(-3.5, 1.0)
    ax.legend(fontsize=8)


def plot_satellite_radial_profile(ax, sim1, sim2, mvir_range=(14.0, 15.0), rmax=2.0):
    """Satellite radial profile in clusters: satellites per halo per d(r/Rvir)."""
    rbins = np.arange(0.0, rmax + 0.1, 0.1)
    rcent = rbins[:-1] + 0.05

    for sim, color, ls in [(sim1, 'C0', '-'), (sim2, 'C1', '--')]:
        grp = _group_central_arrays(sim)
        inv = grp['inv']

        gal_idx = sim['GalaxyIndex'].astype(np.int64)
        cen_idx = sim['CentralGalaxyIndex'].astype(np.int64)
        is_sat = gal_idx != cen_idx

        # Cluster selection by central halo mass
        mvir_group = grp['mvir']
        in_cluster = np.isfinite(mvir_group) & (np.log10(mvir_group) >= mvir_range[0]) & (np.log10(mvir_group) < mvir_range[1])
        n_halo = in_cluster.sum()
        if n_halo < 10:
            continue

        # Central positions and Rvir for each galaxy
        cx = _central_property_for_each_galaxy(sim, 'Posx')
        cy = _central_property_for_each_galaxy(sim, 'Posy')
        cz = _central_property_for_each_galaxy(sim, 'Posz')
        crvir = _central_property_for_each_galaxy(sim, 'Rvir')

        box = sim['params']['BoxSize']  # (Mpc/h)
        dx = _periodic_delta(sim['Posx'] - cx, box)
        dy = _periodic_delta(sim['Posy'] - cy, box)
        dz = _periodic_delta(sim['Posz'] - cz, box)
        r = np.sqrt(dx * dx + dy * dy + dz * dz)

        # Cluster-members only
        w = is_sat & np.isfinite(crvir) & (crvir > 0) & in_cluster[inv]
        x = (r[w] / crvir[w])

        counts, _ = np.histogram(x, bins=rbins)
        prof = counts / n_halo / np.diff(rbins)  # satellites per halo per d(r/Rvir)
        ax.plot(rcent, prof, color=color, ls=ls, lw=2.5, label=sim['label'])

    ax.set_xlabel(r'$r / R_\mathrm{vir}$')
    ax.set_ylabel(r'$\mathrm{d}N_\mathrm{sat} / \mathrm{d}(r/R_\mathrm{vir})$ (per halo)')
    ax.set_title('Satellite Radial Profile (clusters)')
    ax.set_xlim(0, rmax)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)


def _stellar_mass_within_aperture_by_central(sim, aperture_kpc=300.0):
    """Compute total stellar mass within an aperture around each central.

    Uses `CentralGalaxyIndex` membership and positions. Returns:
      - central_logmvir: log10(Mvir) for each central
      - mstar_ap: total stellar mass within aperture (Msun)
      - mics: intra-cluster stars mass for each central (Msun)
    """
    gal_idx = sim['GalaxyIndex'].astype(np.int64)
    cen_idx = sim['CentralGalaxyIndex'].astype(np.int64)
    posx = sim['Posx']
    posy = sim['Posy']
    posz = sim['Posz']

    order = np.argsort(gal_idx)
    gal_sorted = gal_idx[order]
    posx_sorted = posx[order]
    posy_sorted = posy[order]
    posz_sorted = posz[order]

    # Map each galaxy to its central's position by index-join on GalaxyIndex.
    loc = np.searchsorted(gal_sorted, cen_idx)
    valid = (loc >= 0) & (loc < gal_sorted.size) & (gal_sorted[loc] == cen_idx)

    r_kpc = np.full(gal_idx.size, np.inf)
    if valid.any():
        box = sim['params']['BoxSize']  # (Mpc/h)
        dx = _periodic_delta(posx[valid] - posx_sorted[loc[valid]], box)
        dy = _periodic_delta(posy[valid] - posy_sorted[loc[valid]], box)
        dz = _periodic_delta(posz[valid] - posz_sorted[loc[valid]], box)
        r_mpc_over_h = np.sqrt(dx * dx + dy * dy + dz * dz)
        r_kpc[valid] = (r_mpc_over_h / sim['h']) * 1000.0

    in_ap = r_kpc <= aperture_kpc

    # Factorize CentralGalaxyIndex to dense group IDs.
    cen_vals, inv = np.unique(cen_idx, return_inverse=True)
    mstar_ap = np.bincount(inv[in_ap], weights=sim['StellarMass'][in_ap], minlength=cen_vals.size)

    # Pull central properties (Mvir, ICS) from the central galaxy record.
    is_central = (gal_idx == cen_idx) & (sim['Type'] == 0) & (sim['Len'] >= HALO_LEN_MIN)
    gcen = inv[is_central]
    mics = np.full(cen_vals.size, np.nan)
    mvir = np.full(cen_vals.size, np.nan)
    if gcen.size > 0:
        mics[gcen] = sim['IntraClusterStars'][is_central]
        mvir[gcen] = sim['Mvir'][is_central]

    w = np.isfinite(mvir)
    return np.log10(mvir[w]), mstar_ap[w], mics[w]


def plot_icl_fraction(ax, sim1, sim2, aperture_kpc=300.0):
    r"""ICL fraction

    $f_\mathrm{ICL} = M_\mathrm{ICS} / (M_{\star}(<R) + M_\mathrm{ICS})$ with $R=300$ kpc.
    Computed per central halo.
    """
    for sim, color, ls in [(sim1, 'C0', '-'), (sim2, 'C1', '--')]:
        logmh, mstar_ap, mics = _stellar_mass_within_aperture_by_central(sim, aperture_kpc=aperture_kpc)
        denom = mstar_ap + mics
        f_icl = np.where(denom > 0, mics / denom, np.nan)

        bins = np.arange(10.0, 15.0, 0.25)
        cx, my, lo, hi = _binned_stat(logmh, f_icl, bins, min_count=10)
        ax.plot(cx, my, color=color, ls=ls, lw=2.5, label=sim['label'])
        ax.fill_between(cx, lo, hi, color=color, alpha=0.12)

    ax.set_xlabel(r'$\log_{10}(M_\mathrm{vir}\;/\;\mathrm{M_\odot})$')
    ax.set_ylabel(r'$f_\mathrm{ICL}$')
    ax.set_title(r'$f_\mathrm{ICL}$ within 300 kpc (centrals)')
    ax.set_xlim(10, 14.8)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)


def plot_baryonic_tully_fisher(ax, sim1, sim2):
    """Baryonic Tully–Fisher relation using Vmax.

    Uses star-forming centrals and defines baryonic mass as:
      Mbar = M* + 1.36 * (M_HI + M_H2)
    """
    for sim, color, ls in [(sim1, 'C0', '-'), (sim2, 'C1', '--')]:
        centrals = (sim['Type'] == 0) & (sim['Len'] >= HALO_LEN_MIN)
        sfr = sim['SfrDisk'] + sim['SfrBulge']
        w = centrals & (sim['StellarMass'] > 1e8) & (sim['Vmax'] > 0) & (sfr > 0)
        ssfr = np.full(sim['StellarMass'].shape, -np.inf)
        ssfr[w] = np.log10(sfr[w] / sim['StellarMass'][w])
        w &= ssfr > sSFRcut

        mbar = sim['StellarMass'][w] + 1.36 * (sim['H1gas'][w] + sim['H2gas'][w])
        ok = mbar > 0
        logv = np.log10(sim['Vmax'][w][ok])
        logmbar = np.log10(mbar[ok])

        n_samp = min(5000, logv.size)
        if n_samp > 0:
            idx = np.random.choice(logv.size, n_samp, replace=False)
            ax.scatter(logv[idx], logmbar[idx], s=2, alpha=0.15, color=color, rasterized=True)

        bins = np.arange(1.2, 2.8, 0.05)
        cx, my, lo, hi = _binned_stat(logv, logmbar, bins, min_count=20)
        ax.plot(cx, my, color=color, ls=ls, lw=2.5, label=sim['label'])
        ax.fill_between(cx, lo, hi, color=color, alpha=0.12)

    ax.set_xlabel(r'$\log_{10}(V_\mathrm{max}\;/\;\mathrm{km\,s^{-1}})$')
    ax.set_ylabel(r'$\log_{10}(M_\mathrm{bar}\;/\;\mathrm{M_\odot})$')
    ax.set_title('Baryonic Tully–Fisher (star-forming centrals)')
    ax.set_xlim(1.2, 2.7)
    ax.set_ylim(7.5, 12.5)
    ax.legend(fontsize=9)


def plot_baryon_budget(ax, sim1, sim2):
    """Baryon mass budget in bins of halo mass (centrals)."""
    components = [
        ('StellarMass', 'Stars', 'tab:blue'),
        ('ColdGas', 'Cold Gas', 'tab:cyan'),
        ('HotGas', 'Hot Gas', 'tab:red'),
        ('EjectedMass', 'Ejected', 'tab:orange'),
        ('IntraClusterStars', 'ICS', 'tab:purple'),
        ('CGMgas', 'CGM', 'tab:green'),
    ]

    bins = np.arange(10, 14.5, 0.3)
    width = 0.12  # bar width

    for s_idx, (sim, offset) in enumerate([(sim1, -0.06), (sim2, 0.06)]):
        centrals = (sim['Type'] == 0) & (sim['Len'] >= HALO_LEN_MIN) & (sim['Mvir'] > 0)
        logmh = np.log10(sim['Mvir'][centrals])
        dig = np.digitize(logmh, bins)

        bottom = np.zeros(len(bins) - 1)
        for comp_name, comp_label, comp_color in components:
            vals = sim[comp_name][centrals]
            means = []
            centres = []
            for i in range(1, len(bins)):
                sel = dig == i
                if sel.sum() > 5:
                    means.append(np.log10(np.mean(vals[sel])) if np.mean(vals[sel]) > 0 else 0)
                    centres.append(0.5 * (bins[i - 1] + bins[i]))
                else:
                    means.append(0)
                    centres.append(0.5 * (bins[i - 1] + bins[i]))
            # Use line plot instead of stacked bars for clarity
            # Just plot median component mass vs halo mass
            pass

    # Simpler: plot ratio of each component to Mvir
    for sim, ls in [(sim1, '-'), (sim2, '--')]:
        centrals = (sim['Type'] == 0) & (sim['Len'] >= HALO_LEN_MIN) & (sim['Mvir'] > 1e10)
        logmh = np.log10(sim['Mvir'][centrals])
        dig = np.digitize(logmh, bins)

        for comp_name, comp_label, comp_color in components:
            vals = sim[comp_name][centrals]
            mvir = sim['Mvir'][centrals]
            med_x, med_y = [], []
            for i in range(1, len(bins)):
                sel = dig == i
                if sel.sum() > 10:
                    ratio = vals[sel] / mvir[sel]
                    med_x.append(0.5 * (bins[i - 1] + bins[i]))
                    med_y.append(np.median(ratio))

            lbl = comp_label if ls == '-' else None
            ax.plot(med_x, np.log10(np.array(med_y) + 1e-20), color=comp_color,
                    ls=ls, lw=2, label=lbl)

    ax.set_xlabel(r'$\log_{10}(M_\mathrm{vir}\;/\;\mathrm{M_\odot})$')
    ax.set_ylabel(r'$\log_{10}(\mathrm{component} / M_\mathrm{vir})$')
    ax.set_title('Baryon Budget (centrals)')
    ax.set_xlim(10.5, 14.5)
    ax.set_ylim(-5, -0.5)
    ax.legend(fontsize=7, ncol=2, loc='lower left')


def plot_concentration(ax, sim1, sim2):
    """Concentration vs halo mass (centrals)."""
    for sim, color, ls in [(sim1, 'C0', '-'), (sim2, 'C1', '--')]:
        centrals = (sim['Type'] == 0) & (sim['Len'] >= HALO_LEN_MIN) & (sim['Mvir'] > 0) & (sim['Concentration'] > 0)
        logmh = np.log10(sim['Mvir'][centrals])
        c = sim['Concentration'][centrals]

        bins = np.arange(10, 15, 0.2)
        dig = np.digitize(logmh, bins)
        med_x, med_y, p16, p84 = [], [], [], []
        for i in range(1, len(bins)):
            sel = dig == i
            if sel.sum() > 10:
                med_x.append(0.5 * (bins[i - 1] + bins[i]))
                med_y.append(np.median(c[sel]))
                p16.append(np.percentile(c[sel], 16))
                p84.append(np.percentile(c[sel], 84))

        med_x = np.array(med_x)
        ax.plot(med_x, med_y, color=color, ls=ls, lw=2.5, label=sim['label'])
        ax.fill_between(med_x, p16, p84, color=color, alpha=0.12)

    ax.set_xlabel(r'$\log_{10}(M_\mathrm{vir}\;/\;\mathrm{M_\odot})$')
    ax.set_ylabel('Concentration')
    ax.set_title('Concentration–Mass (centrals)')
    ax.set_xlim(10, 14.5)
    ax.legend(fontsize=9)


def plot_regime_fractions(ax, sim1, sim2):
    """FFB regime fraction vs halo mass."""
    for sim, ls, marker in [(sim1, '-', 'o'), (sim2, '--', 's')]:
        centrals = (sim['Type'] == 0) & (sim['Len'] >= HALO_LEN_MIN) & (sim['Mvir'] > 0)
        logmh = np.log10(sim['Mvir'][centrals])
        regime = sim['FFBRegime'][centrals]

        bins = np.arange(10, 14.5, 0.3)
        dig = np.digitize(logmh, bins)

        regime_labels = {0: 'No FFB', 1: 'FFB'}
        regime_colors = {0: 'tab:blue', 1: 'tab:red'}

        for r_val, r_label in regime_labels.items():
            frac_x, frac_y = [], []
            for i in range(1, len(bins)):
                sel = dig == i
                if sel.sum() > 20:
                    frac_x.append(0.5 * (bins[i - 1] + bins[i]))
                    frac_y.append((regime[sel] == r_val).sum() / sel.sum())
            lbl = f"{sim['label']} {r_label}" if ls == '-' else None
            ax.plot(frac_x, frac_y, color=regime_colors[r_val], ls=ls, lw=2,
                    marker=marker, ms=3, label=lbl)

    ax.set_xlabel(r'$\log_{10}(M_\mathrm{vir}\;/\;\mathrm{M_\odot})$')
    ax.set_ylabel('Fraction')
    ax.set_title('FFB Regime Fractions (centrals)')
    ax.set_xlim(10, 14.5)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, ncol=2)


# ========== Quenching diagnostic plots ==========

def _binned_stat(x, y, bins, min_count=10):
    """Return bin centres, medians, 16th and 84th percentiles."""
    dig = np.digitize(x, bins)
    cx, my, lo, hi = [], [], [], []
    for i in range(1, len(bins)):
        sel = dig == i
        if sel.sum() >= min_count:
            cx.append(0.5 * (bins[i - 1] + bins[i]))
            my.append(np.median(y[sel]))
            lo.append(np.percentile(y[sel], 16))
            hi.append(np.percentile(y[sel], 84))
    return np.array(cx), np.array(my), np.array(lo), np.array(hi)


def plot_quenched_central_vs_satellite(ax, sim1, sim2):
    """Quenched fraction split by centrals and satellites."""
    for sim, color, ls in [(sim1, 'C0', '-'), (sim2, 'C1', '--')]:
        w = sim['StellarMass'] > 0
        sm = np.log10(sim['StellarMass'][w])
        sfr = sim['SfrDisk'][w] + sim['SfrBulge'][w]
        ssfr = np.log10(sfr / sim['StellarMass'][w])
        types = sim['Type'][w]

        bins = np.arange(8, 12.5, 0.3)
        dig = np.digitize(sm, bins)

        for t, tname, tcolor in [(0, 'Central', 'tab:blue'), (1, 'Satellite', 'tab:orange')]:
            frac_x, frac_y = [], []
            for i in range(1, len(bins)):
                sel = (dig == i) & (types == t)
                if sel.sum() > 20:
                    frac_x.append(0.5 * (bins[i - 1] + bins[i]))
                    frac_y.append((ssfr[sel] < sSFRcut).sum() / sel.sum())
            lbl = f"{sim['label']} {tname}" if ls == '-' else None
            ax.plot(frac_x, frac_y, color=tcolor, ls=ls, lw=2.5,
                    marker='o' if ls == '-' else 's', ms=4, label=lbl)

    ax.set_xlabel(r'$\log_{10}(M_\star\;/\;\mathrm{M_\odot})$')
    ax.set_ylabel('Quenched Fraction')
    ax.set_title('Quenched Fraction: Centrals vs Satellites')
    ax.set_xlim(8, 12.5)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, ncol=2)


def plot_heating_vs_cooling(ax, sim1, sim2):
    """Heating vs cooling luminosity for centrals, coloured by sim."""
    for sim, color in [(sim1, 'C0'), (sim2, 'C1')]:
        centrals = (sim['Type'] == 0) & (sim['Len'] >= HALO_LEN_MIN) & (sim['Cooling'] > 0) & (sim['Heating'] > 0)
        cool = np.log10(sim['Cooling'][centrals])
        heat = np.log10(sim['Heating'][centrals])

        n_samp = min(5000, centrals.sum())
        idx = np.random.choice(centrals.sum(), n_samp, replace=False)
        ax.scatter(cool[idx], heat[idx], s=1, alpha=0.15, color=color, rasterized=True)

        # Median trend
        bins = np.arange(38, 46, 0.5)
        cx, my, _, _ = _binned_stat(cool, heat, bins)
        ax.plot(cx, my, color=color, lw=2.5, label=sim['label'])

    ax.plot([38, 46], [38, 46], 'k:', lw=1, alpha=0.5)
    ax.set_xlabel(r'$\log_{10}(\mathrm{Cooling}\;/\;\mathrm{erg\,s^{-1}})$')
    ax.set_ylabel(r'$\log_{10}(\mathrm{Heating}\;/\;\mathrm{erg\,s^{-1}})$')
    ax.set_title('AGN Heating vs Cooling (centrals)')
    ax.set_xlim(38, 46)
    ax.set_ylim(38, 46)
    ax.legend(fontsize=9)


def plot_heating_cooling_ratio_vs_mvir(ax, sim1, sim2):
    """Heating/Cooling ratio vs halo mass for centrals."""
    for sim, color, ls in [(sim1, 'C0', '-'), (sim2, 'C1', '--')]:
        centrals = (sim['Type'] == 0) & (sim['Len'] >= HALO_LEN_MIN) & (sim['Mvir'] > 0) & (sim['Cooling'] > 0)
        logmh = np.log10(sim['Mvir'][centrals])
        ratio = sim['Heating'][centrals] / sim['Cooling'][centrals]
        logratio = np.log10(ratio + 1e-10)

        bins = np.arange(10, 14.5, 0.2)
        cx, my, lo, hi = _binned_stat(logmh, logratio, bins)

        ax.plot(cx, my, color=color, ls=ls, lw=2.5, label=sim['label'])
        ax.fill_between(cx, lo, hi, color=color, alpha=0.12)

    ax.axhline(0, color='grey', ls=':', lw=1, alpha=0.7)
    ax.set_xlabel(r'$\log_{10}(M_\mathrm{vir}\;/\;\mathrm{M_\odot})$')
    ax.set_ylabel(r'$\log_{10}(\mathrm{Heating} / \mathrm{Cooling})$')
    ax.set_title('Heating/Cooling Ratio vs Halo Mass (centrals)')
    ax.set_xlim(10, 14.5)
    ax.set_ylim(-2, 1.5)
    ax.legend(fontsize=9)


def plot_hot_regime_fraction(ax, sim1, sim2):
    """Hot halo regime fraction vs halo mass (centrals)."""
    for sim, color, ls in [(sim1, 'C0', '-'), (sim2, 'C1', '--')]:
        centrals = (sim['Type'] == 0) & (sim['Len'] >= HALO_LEN_MIN) & (sim['Mvir'] > 0)
        logmh = np.log10(sim['Mvir'][centrals])
        regime = sim['Regime'][centrals]

        bins = np.arange(10, 14.5, 0.2)
        dig = np.digitize(logmh, bins)
        frac_x, frac_hot = [], []
        for i in range(1, len(bins)):
            sel = dig == i
            if sel.sum() > 20:
                frac_x.append(0.5 * (bins[i - 1] + bins[i]))
                frac_hot.append((regime[sel] == 1).sum() / sel.sum())

        ax.plot(frac_x, frac_hot, color=color, ls=ls, lw=2.5,
                marker='o', ms=4, label=sim['label'])

    ax.set_xlabel(r'$\log_{10}(M_\mathrm{vir}\;/\;\mathrm{M_\odot})$')
    ax.set_ylabel('Hot Halo Fraction (Regime = 1)')
    ax.set_title('Hot Halo Regime vs Halo Mass (centrals)')
    ax.set_xlim(10, 14.5)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)


def plot_bh_mass_vs_mvir(ax, sim1, sim2):
    """BH mass vs halo mass for centrals — directly controls radio mode heating."""
    for sim, color, ls in [(sim1, 'C0', '-'), (sim2, 'C1', '--')]:
        centrals = (sim['Type'] == 0) & (sim['Len'] >= HALO_LEN_MIN) & (sim['Mvir'] > 0) & (sim['BlackHoleMass'] > 0)
        logmh = np.log10(sim['Mvir'][centrals])
        logbh = np.log10(sim['BlackHoleMass'][centrals])

        bins = np.arange(10, 14.5, 0.2)
        cx, my, lo, hi = _binned_stat(logmh, logbh, bins)

        ax.plot(cx, my, color=color, ls=ls, lw=2.5, label=sim['label'])
        ax.fill_between(cx, lo, hi, color=color, alpha=0.12)

    ax.set_xlabel(r'$\log_{10}(M_\mathrm{vir}\;/\;\mathrm{M_\odot})$')
    ax.set_ylabel(r'$\log_{10}(M_\mathrm{BH}\;/\;\mathrm{M_\odot})$')
    ax.set_title('BH Mass vs Halo Mass (centrals)')
    ax.set_xlim(10, 14.5)
    ax.legend(fontsize=9)


def plot_ssfr_vs_mvir_centrals(ax, sim1, sim2):
    """sSFR vs halo mass for centrals only — isolates central quenching from satellite mixing."""
    for sim, color, ls in [(sim1, 'C0', '-'), (sim2, 'C1', '--')]:
        centrals = (sim['Type'] == 0) & (sim['Len'] >= HALO_LEN_MIN) & (sim['StellarMass'] > 1e7)
        sfr = sim['SfrDisk'][centrals] + sim['SfrBulge'][centrals]
        w_sf = sfr > 0
        logmh = np.log10(sim['Mvir'][centrals])
        ssfr = np.log10(sfr / sim['StellarMass'][centrals])

        # Scatter (star-forming only for visibility)
        logmh_sf = logmh[w_sf]
        ssfr_sf = ssfr[w_sf]
        n_samp = min(5000, w_sf.sum())
        idx = np.random.choice(w_sf.sum(), n_samp, replace=False)
        ax.scatter(logmh_sf[idx], ssfr_sf[idx], s=1, alpha=0.12, color=color, rasterized=True)

        # Median of ALL centrals (including quenched as sSFR = -14)
        ssfr_all = np.where(sfr > 0, ssfr, -14.0)
        bins = np.arange(10, 14.5, 0.3)
        cx, my, _, _ = _binned_stat(logmh, ssfr_all, bins)
        ax.plot(cx, my, color=color, ls=ls, lw=2.5, label=sim['label'])

    ax.axhline(sSFRcut, color='grey', ls=':', lw=1, alpha=0.7)
    ax.set_xlabel(r'$\log_{10}(M_\mathrm{vir}\;/\;\mathrm{M_\odot})$')
    ax.set_ylabel(r'$\log_{10}(\mathrm{sSFR}\;/\;\mathrm{yr^{-1}})$')
    ax.set_title('sSFR vs Halo Mass (centrals only)')
    ax.set_xlim(10, 14.5)
    ax.set_ylim(-14, -8)
    ax.legend(fontsize=9)


# ========== Gas supply / cooling diagnostics ==========

def plot_hotgas_vs_mvir(ax, sim1, sim2):
    """Hot gas mass vs halo mass for centrals."""
    for sim, color, ls in [(sim1, 'C0', '-'), (sim2, 'C1', '--')]:
        centrals = (sim['Type'] == 0) & (sim['Len'] >= HALO_LEN_MIN) & (sim['Mvir'] > 0) & (sim['HotGas'] > 0)
        logmh = np.log10(sim['Mvir'][centrals])
        logmhot = np.log10(sim['HotGas'][centrals])

        bins = np.arange(10, 14.5, 0.2)
        cx, my, lo, hi = _binned_stat(logmh, logmhot, bins)
        ax.plot(cx, my, color=color, ls=ls, lw=2.5, label=sim['label'])
        ax.fill_between(cx, lo, hi, color=color, alpha=0.12)

    # Plot cosmic baryon fraction line
    fb = 0.17  # approximate
    mh_arr = np.logspace(10, 14.5, 50)
    ax.plot(np.log10(mh_arr), np.log10(fb * mh_arr), 'k:', lw=1, alpha=0.5, label=r'$f_b \times M_\mathrm{vir}$')

    ax.set_xlabel(r'$\log_{10}(M_\mathrm{vir}\;/\;\mathrm{M_\odot})$')
    ax.set_ylabel(r'$\log_{10}(M_\mathrm{hot}\;/\;\mathrm{M_\odot})$')
    ax.set_title('Hot Gas Mass vs Halo Mass (centrals)')
    ax.set_xlim(10, 14.5)
    ax.set_ylim(8, 14)
    ax.legend(fontsize=9)


def plot_cgmgas_vs_mvir(ax, sim1, sim2):
    """CGM gas mass vs halo mass for centrals."""
    for sim, color, ls in [(sim1, 'C0', '-'), (sim2, 'C1', '--')]:
        centrals = (sim['Type'] == 0) & (sim['Len'] >= HALO_LEN_MIN) & (sim['Mvir'] > 0) & (sim['CGMgas'] > 0)
        logmh = np.log10(sim['Mvir'][centrals])
        logcgm = np.log10(sim['CGMgas'][centrals])

        bins = np.arange(10, 14.5, 0.2)
        cx, my, lo, hi = _binned_stat(logmh, logcgm, bins)
        ax.plot(cx, my, color=color, ls=ls, lw=2.5, label=sim['label'])
        ax.fill_between(cx, lo, hi, color=color, alpha=0.12)

    ax.set_xlabel(r'$\log_{10}(M_\mathrm{vir}\;/\;\mathrm{M_\odot})$')
    ax.set_ylabel(r'$\log_{10}(M_\mathrm{CGM}\;/\;\mathrm{M_\odot})$')
    ax.set_title('CGM Gas Mass vs Halo Mass (centrals)')
    ax.set_xlim(10, 14.5)
    ax.legend(fontsize=9)


def plot_coldgas_vs_mvir(ax, sim1, sim2):
    """Cold gas mass vs halo mass for centrals."""
    for sim, color, ls in [(sim1, 'C0', '-'), (sim2, 'C1', '--')]:
        centrals = (sim['Type'] == 0) & (sim['Len'] >= HALO_LEN_MIN) & (sim['Mvir'] > 0)
        logmh = np.log10(sim['Mvir'][centrals])
        coldgas = sim['ColdGas'][centrals]

        # Fraction with zero cold gas
        bins = np.arange(10, 14.5, 0.3)
        dig = np.digitize(logmh, bins)
        frac_x, frac_zero = [], []
        for i in range(1, len(bins)):
            sel = dig == i
            if sel.sum() > 20:
                frac_x.append(0.5 * (bins[i - 1] + bins[i]))
                frac_zero.append((coldgas[sel] == 0).sum() / sel.sum())

        # Median cold gas for those with gas
        w_gas = coldgas > 0
        logcold = np.log10(coldgas[w_gas])
        logmh_gas = logmh[w_gas]
        cx, my, lo, hi = _binned_stat(logmh_gas, logcold, np.arange(10, 14.5, 0.2))
        ax.plot(cx, my, color=color, ls=ls, lw=2.5, label=sim['label'])
        ax.fill_between(cx, lo, hi, color=color, alpha=0.12)

    ax.set_xlabel(r'$\log_{10}(M_\mathrm{vir}\;/\;\mathrm{M_\odot})$')
    ax.set_ylabel(r'$\log_{10}(M_\mathrm{cold}\;/\;\mathrm{M_\odot})$')
    ax.set_title('Cold Gas vs Halo Mass (centrals, gas>0)')
    ax.set_xlim(10, 14.5)
    ax.legend(fontsize=9)


def plot_zero_coldgas_fraction(ax, sim1, sim2):
    """Fraction of centrals with zero cold gas or zero SFR vs halo mass."""
    for sim, color, ls in [(sim1, 'C0', '-'), (sim2, 'C1', '--')]:
        centrals = (sim['Type'] == 0) & (sim['Len'] >= HALO_LEN_MIN) & (sim['Mvir'] > 0)
        logmh = np.log10(sim['Mvir'][centrals])
        coldgas = sim['ColdGas'][centrals]
        sfr = sim['SfrDisk'][centrals] + sim['SfrBulge'][centrals]

        bins = np.arange(10, 14.5, 0.25)
        dig = np.digitize(logmh, bins)

        frac_x, frac_nogas, frac_nosfr = [], [], []
        for i in range(1, len(bins)):
            sel = dig == i
            if sel.sum() > 20:
                frac_x.append(0.5 * (bins[i - 1] + bins[i]))
                frac_nogas.append((coldgas[sel] == 0).sum() / sel.sum())
                frac_nosfr.append((sfr[sel] == 0).sum() / sel.sum())

        ax.plot(frac_x, frac_nosfr, color=color, ls=ls, lw=2.5,
                marker='o' if ls == '-' else 's', ms=4, label=f'{sim["label"]} SFR=0')
        ax.plot(frac_x, frac_nogas, color=color, ls=':', lw=1.5, alpha=0.7,
                label=f'{sim["label"]} ColdGas=0' if ls == '-' else None)

    ax.set_xlabel(r'$\log_{10}(M_\mathrm{vir}\;/\;\mathrm{M_\odot})$')
    ax.set_ylabel('Fraction')
    ax.set_title('Fraction with Zero SFR / Zero Cold Gas (centrals)')
    ax.set_xlim(10, 14.5)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, ncol=2)


def plot_rcool_vs_mvir(ax, sim1, sim2):
    """Rcool/Rvir vs halo mass for centrals."""
    for sim, color, ls in [(sim1, 'C0', '-'), (sim2, 'C1', '--')]:
        centrals = (sim['Type'] == 0) & (sim['Len'] >= HALO_LEN_MIN) & (sim['Mvir'] > 0) & (sim['RcoolToRvir'] > 0)
        logmh = np.log10(sim['Mvir'][centrals])
        rcool = sim['RcoolToRvir'][centrals]

        bins = np.arange(10, 14.5, 0.2)
        cx, my, lo, hi = _binned_stat(logmh, rcool, bins)
        ax.plot(cx, my, color=color, ls=ls, lw=2.5, label=sim['label'])
        ax.fill_between(cx, lo, hi, color=color, alpha=0.12)

    ax.axhline(1.0, color='grey', ls=':', lw=1, alpha=0.7)
    ax.set_xlabel(r'$\log_{10}(M_\mathrm{vir}\;/\;\mathrm{M_\odot})$')
    ax.set_ylabel(r'$r_\mathrm{cool} / R_\mathrm{vir}$')
    ax.set_title(r'$r_\mathrm{cool}/R_\mathrm{vir}$ vs Halo Mass (centrals)')
    ax.set_xlim(10, 14.5)
    ax.set_ylim(0, 2)
    ax.legend(fontsize=9)


def plot_mdot_cool_vs_mvir(ax, sim1, sim2):
    """Cooling rate (mdot_cool) vs halo mass for centrals."""
    for sim, color, ls in [(sim1, 'C0', '-'), (sim2, 'C1', '--')]:
        centrals = (sim['Type'] == 0) & (sim['Len'] >= HALO_LEN_MIN) & (sim['Mvir'] > 0) & (sim['mdot_cool'] > 0)
        logmh = np.log10(sim['Mvir'][centrals])
        logmdot = np.log10(sim['mdot_cool'][centrals])

        bins = np.arange(10, 14.5, 0.2)
        cx, my, lo, hi = _binned_stat(logmh, logmdot, bins)
        ax.plot(cx, my, color=color, ls=ls, lw=2.5, label=sim['label'])
        ax.fill_between(cx, lo, hi, color=color, alpha=0.12)

    ax.set_xlabel(r'$\log_{10}(M_\mathrm{vir}\;/\;\mathrm{M_\odot})$')
    ax.set_ylabel(r'$\log_{10}(\dot{M}_\mathrm{cool}\;/\;\mathrm{code\,units})$')
    ax.set_title('Cooling Rate vs Halo Mass (centrals)')
    ax.set_xlim(10, 14.5)
    ax.legend(fontsize=9)


# ========== Concentration → cooling chain diagnostics ==========

def plot_rcool_vs_concentration(ax, sim1, sim2):
    """rcool/Rvir vs concentration at fixed halo mass bins — the key test."""
    mass_bins = [(11.5, 12.0), (12.0, 12.5), (12.5, 13.0)]
    markers = ['o', 's', 'D']

    for sim, color in [(sim1, 'C0'), (sim2, 'C1')]:
        centrals = (sim['Type'] == 0) & (sim['Len'] >= HALO_LEN_MIN) & (sim['Mvir'] > 0) & \
                   (sim['Concentration'] > 0) & (sim['RcoolToRvir'] > 0)
        logmh = np.log10(sim['Mvir'][centrals])
        conc = sim['Concentration'][centrals]
        rcool = sim['RcoolToRvir'][centrals]

        for (mlo, mhi), mk in zip(mass_bins, markers):
            in_bin = (logmh >= mlo) & (logmh < mhi)
            if in_bin.sum() < 20:
                continue
            c_bin = conc[in_bin]
            r_bin = rcool[in_bin]

            cbins = np.arange(2, 18, 1.5)
            cx, my, lo, hi = _binned_stat(c_bin, r_bin, cbins, min_count=5)

            lbl = f"{sim['label']} [{mlo:.1f},{mhi:.1f})" if color == 'C0' else None
            ax.plot(cx, my, color=color, marker=mk, ms=5, lw=2,
                    ls='-' if color == 'C0' else '--', label=lbl, alpha=0.85)

    ax.axhline(1.0, color='grey', ls=':', lw=1, alpha=0.5)
    ax.set_xlabel('Concentration')
    ax.set_ylabel(r'$r_\mathrm{cool} / R_\mathrm{vir}$')
    ax.set_title(r'$r_\mathrm{cool}/R_\mathrm{vir}$ vs Concentration (fixed $M_\mathrm{vir}$ bins)')
    ax.set_xlim(2, 16)
    ax.set_ylim(0, 1.5)
    ax.legend(fontsize=7, ncol=2)


def plot_concentration_distribution(ax, sim1, sim2):
    """Concentration distributions in narrow halo mass bins."""
    mass_bins = [(11.5, 12.0), (12.0, 12.5), (12.5, 13.0)]
    lss = ['-', '--', ':']

    for sim, color in [(sim1, 'C0'), (sim2, 'C1')]:
        centrals = (sim['Type'] == 0) & (sim['Len'] >= HALO_LEN_MIN) & (sim['Mvir'] > 0) & (sim['Concentration'] > 0)
        logmh = np.log10(sim['Mvir'][centrals])
        conc = sim['Concentration'][centrals]

        for (mlo, mhi), ls in zip(mass_bins, lss):
            in_bin = (logmh >= mlo) & (logmh < mhi)
            if in_bin.sum() < 10:
                continue
            c_bin = conc[in_bin]
            lbl = f"{sim['label']} [{mlo:.1f},{mhi:.1f})" if color == 'C0' else None
            ax.hist(c_bin, bins=np.arange(1, 20, 0.5), density=True,
                    histtype='step', color=color, ls=ls, lw=2, label=lbl)

    ax.set_xlabel('Concentration')
    ax.set_ylabel('Normalised PDF')
    ax.set_title('Concentration Distribution (fixed Mvir bins)')
    ax.set_xlim(1, 18)
    ax.legend(fontsize=7, ncol=2)


def plot_ssfr_vs_concentration(ax, sim1, sim2):
    """sSFR vs concentration for centrals in a narrow halo mass range."""
    mlo, mhi = 11.8, 12.5  # MW-to-group scale where quenching diverges
    for sim, color in [(sim1, 'C0'), (sim2, 'C1')]:
        centrals = (sim['Type'] == 0) & (sim['Len'] >= HALO_LEN_MIN) & (sim['Mvir'] > 0) & \
                   (sim['Concentration'] > 0) & (sim['StellarMass'] > 1e7)
        logmh = np.log10(sim['Mvir'][centrals])
        in_bin = (logmh >= mlo) & (logmh < mhi)

        conc = sim['Concentration'][centrals][in_bin]
        sfr = sim['SfrDisk'][centrals][in_bin] + sim['SfrBulge'][centrals][in_bin]
        sm = sim['StellarMass'][centrals][in_bin]
        ssfr = np.where(sfr > 0, np.log10(sfr / sm), -14.0)

        n_samp = min(3000, in_bin.sum())
        if n_samp > 0:
            idx = np.random.choice(in_bin.sum(), n_samp, replace=False)
            ax.scatter(conc[idx], ssfr[idx], s=3, alpha=0.2, color=color, rasterized=True)

        cbins = np.arange(2, 16, 1.0)
        cx, my, _, _ = _binned_stat(conc, ssfr, cbins, min_count=5)
        ax.plot(cx, my, color=color, lw=2.5, label=sim['label'])

    ax.axhline(sSFRcut, color='grey', ls=':', lw=1, alpha=0.5)
    ax.set_xlabel('Concentration')
    ax.set_ylabel(r'$\log_{10}(\mathrm{sSFR}\;/\;\mathrm{yr^{-1}})$')
    ax.set_title(f'sSFR vs Concentration (centrals, log Mvir=[{mlo},{mhi}])')
    ax.set_xlim(2, 16)
    ax.set_ylim(-14, -8)
    ax.legend(fontsize=9)


def plot_quenched_vs_concentration(ax, sim1, sim2):
    """Quenched fraction vs concentration in halo mass bins."""
    mass_bins = [(11.5, 12.0), (12.0, 12.5), (12.5, 13.0)]
    markers = ['o', 's', 'D']

    for sim, color in [(sim1, 'C0'), (sim2, 'C1')]:
        centrals = (sim['Type'] == 0) & (sim['Len'] >= HALO_LEN_MIN) & (sim['Mvir'] > 0) & \
                   (sim['Concentration'] > 0) & (sim['StellarMass'] > 1e7)
        logmh = np.log10(sim['Mvir'][centrals])
        conc = sim['Concentration'][centrals]
        sfr = sim['SfrDisk'][centrals] + sim['SfrBulge'][centrals]
        sm = sim['StellarMass'][centrals]
        ssfr = np.where(sfr > 0, np.log10(sfr / sm), -14.0)

        for (mlo, mhi), mk in zip(mass_bins, markers):
            in_bin = (logmh >= mlo) & (logmh < mhi)
            if in_bin.sum() < 20:
                continue
            c_bin = conc[in_bin]
            ssfr_bin = ssfr[in_bin]

            cbins = np.arange(2, 16, 1.5)
            dig = np.digitize(c_bin, cbins)
            frac_x, frac_q = [], []
            for i in range(1, len(cbins)):
                sel = dig == i
                if sel.sum() > 10:
                    frac_x.append(0.5 * (cbins[i - 1] + cbins[i]))
                    frac_q.append((ssfr_bin[sel] < sSFRcut).sum() / sel.sum())

            lbl = f"{sim['label']} [{mlo:.1f},{mhi:.1f})" if color == 'C0' else None
            ax.plot(frac_x, frac_q, color=color, marker=mk, ms=5, lw=2,
                    ls='-' if color == 'C0' else '--', label=lbl, alpha=0.85)

    ax.set_xlabel('Concentration')
    ax.set_ylabel('Quenched Fraction')
    ax.set_title('Quenched Fraction vs Concentration (fixed Mvir bins)')
    ax.set_xlim(2, 16)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, ncol=2)


def plot_mdot_vs_concentration(ax, sim1, sim2):
    """Cooling rate vs concentration at fixed halo mass."""
    mass_bins = [(11.5, 12.0), (12.0, 12.5), (12.5, 13.0)]
    markers = ['o', 's', 'D']

    for sim, color in [(sim1, 'C0'), (sim2, 'C1')]:
        centrals = (sim['Type'] == 0) & (sim['Len'] >= HALO_LEN_MIN) & (sim['Mvir'] > 0) & \
                   (sim['Concentration'] > 0) & (sim['mdot_cool'] > 0)
        logmh = np.log10(sim['Mvir'][centrals])
        conc = sim['Concentration'][centrals]
        mdot = np.log10(sim['mdot_cool'][centrals])

        for (mlo, mhi), mk in zip(mass_bins, markers):
            in_bin = (logmh >= mlo) & (logmh < mhi)
            if in_bin.sum() < 20:
                continue

            cbins = np.arange(2, 16, 1.5)
            cx, my, _, _ = _binned_stat(conc[in_bin], mdot[in_bin], cbins, min_count=5)
            lbl = f"{sim['label']} [{mlo:.1f},{mhi:.1f})" if color == 'C0' else None
            ax.plot(cx, my, color=color, marker=mk, ms=5, lw=2,
                    ls='-' if color == 'C0' else '--', label=lbl, alpha=0.85)

    ax.set_xlabel('Concentration')
    ax.set_ylabel(r'$\log_{10}(\dot{M}_\mathrm{cool})$')
    ax.set_title('Cooling Rate vs Concentration (fixed Mvir bins)')
    ax.set_xlim(2, 16)
    ax.legend(fontsize=7, ncol=2)


def plot_conc_residual_quenching(ax, sim1, sim2):
    """Do the two sims agree on quenched fraction if you match by concentration
    instead of halo mass? 2D binning in (Mvir, conc) space."""
    for sim, color, ls in [(sim1, 'C0', '-'), (sim2, 'C1', '--')]:
        centrals = (sim['Type'] == 0) & (sim['Len'] >= HALO_LEN_MIN) & (sim['Mvir'] > 0) & \
                   (sim['Concentration'] > 0) & (sim['StellarMass'] > 1e8)
        logmh = np.log10(sim['Mvir'][centrals])
        conc = sim['Concentration'][centrals]
        sfr = sim['SfrDisk'][centrals] + sim['SfrBulge'][centrals]
        sm = sim['StellarMass'][centrals]
        ssfr = np.where(sfr > 0, np.log10(sfr / sm), -14.0)
        quenched = ssfr < sSFRcut

        # Bin in Mvir, but weight by concentration rank within each bin
        # Show quenched fraction vs Mvir for HIGH-conc and LOW-conc haloes
        bins = np.arange(10.5, 14.5, 0.3)
        dig = np.digitize(logmh, bins)

        for clabel, cfunc, cmk in [('high-c', lambda c, m: c > np.median(c), '^'),
                                     ('low-c', lambda c, m: c <= np.median(c), 'v')]:
            frac_x, frac_q = [], []
            for i in range(1, len(bins)):
                sel = dig == i
                if sel.sum() > 30:
                    c_sel = conc[sel]
                    c_med = np.median(c_sel)
                    if clabel == 'high-c':
                        subsample = c_sel > c_med
                    else:
                        subsample = c_sel <= c_med
                    if subsample.sum() > 10:
                        frac_x.append(0.5 * (bins[i - 1] + bins[i]))
                        frac_q.append(quenched[sel][subsample].sum() / subsample.sum())

            lbl = f"{sim['label']} {clabel}" if ls == '-' else None
            ax.plot(frac_x, frac_q, color=color, ls=ls, lw=2,
                    marker=cmk, ms=5, label=lbl, alpha=0.85)

    ax.set_xlabel(r'$\log_{10}(M_\mathrm{vir}\;/\;\mathrm{M_\odot})$')
    ax.set_ylabel('Quenched Fraction')
    ax.set_title('Quenched Fraction: High-c vs Low-c haloes')
    ax.set_xlim(10.5, 14)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, ncol=2)


# ========== Hot gas metallicity diagnostics ==========

def plot_hotgas_metallicity_vs_mvir(ax, sim1, sim2):
    """Hot gas metallicity (MetalsHotGas/HotGas) vs halo mass for centrals."""
    Zsun = 0.02
    for sim, color, ls in [(sim1, 'C0', '-'), (sim2, 'C1', '--')]:
        centrals = (sim['Type'] == 0) & (sim['Len'] >= HALO_LEN_MIN) & (sim['Mvir'] > 0) & \
                   (sim['HotGas'] > 0) & (sim['MetalsHotGas'] > 0)
        logmh = np.log10(sim['Mvir'][centrals])
        logZ = np.log10(sim['MetalsHotGas'][centrals] / sim['HotGas'][centrals] / Zsun)

        bins = np.arange(10, 14.5, 0.2)
        cx, my, lo, hi = _binned_stat(logmh, logZ, bins)
        ax.plot(cx, my, color=color, ls=ls, lw=2.5, label=sim['label'])
        ax.fill_between(cx, lo, hi, color=color, alpha=0.12)

    ax.set_xlabel(r'$\log_{10}(M_\mathrm{vir}\;/\;\mathrm{M_\odot})$')
    ax.set_ylabel(r'$\log_{10}(Z_\mathrm{hot} / Z_\odot)$')
    ax.set_title('Hot Gas Metallicity vs Halo Mass (centrals)')
    ax.set_xlim(10, 14.5)
    ax.legend(fontsize=9)


def plot_cgm_metallicity_vs_mvir(ax, sim1, sim2):
    """CGM gas metallicity vs halo mass for centrals — this enters rcool for Regime=0."""
    Zsun = 0.02
    for sim, color, ls in [(sim1, 'C0', '-'), (sim2, 'C1', '--')]:
        centrals = (sim['Type'] == 0) & (sim['Len'] >= HALO_LEN_MIN) & (sim['Mvir'] > 0) & \
                   (sim['CGMgas'] > 0) & (sim['MetalsCGMgas'] > 0)
        logmh = np.log10(sim['Mvir'][centrals])
        logZ = np.log10(sim['MetalsCGMgas'][centrals] / sim['CGMgas'][centrals] / Zsun)

        bins = np.arange(10, 14.5, 0.2)
        cx, my, lo, hi = _binned_stat(logmh, logZ, bins)
        ax.plot(cx, my, color=color, ls=ls, lw=2.5, label=sim['label'])
        ax.fill_between(cx, lo, hi, color=color, alpha=0.12)

    ax.set_xlabel(r'$\log_{10}(M_\mathrm{vir}\;/\;\mathrm{M_\odot})$')
    ax.set_ylabel(r'$\log_{10}(Z_\mathrm{CGM} / Z_\odot)$')
    ax.set_title('CGM Gas Metallicity vs Halo Mass (centrals)')
    ax.set_xlim(10, 14.5)
    ax.legend(fontsize=9)


def plot_rcool_vs_hotgas_metallicity(ax, sim1, sim2):
    """rcool/Rvir vs hot gas metallicity at fixed halo mass — direct test."""
    mass_bins = [(11.5, 12.0), (12.0, 12.5), (12.5, 13.0)]
    markers = ['o', 's', 'D']

    for sim, color in [(sim1, 'C0'), (sim2, 'C1')]:
        centrals = (sim['Type'] == 0) & (sim['Len'] >= HALO_LEN_MIN) & (sim['Mvir'] > 0) & \
                   (sim['HotGas'] > 0) & (sim['MetalsHotGas'] > 0) & \
                   (sim['RcoolToRvir'] > 0)
        logmh = np.log10(sim['Mvir'][centrals])
        logZ = np.log10(sim['MetalsHotGas'][centrals] / sim['HotGas'][centrals])
        rcool = sim['RcoolToRvir'][centrals]

        for (mlo, mhi), mk in zip(mass_bins, markers):
            in_bin = (logmh >= mlo) & (logmh < mhi)
            if in_bin.sum() < 20:
                continue

            zbins = np.arange(-4, -0.5, 0.3)
            cx, my, _, _ = _binned_stat(logZ[in_bin], rcool[in_bin], zbins, min_count=5)
            lbl = f"{sim['label']} [{mlo:.1f},{mhi:.1f})" if color == 'C0' else None
            ax.plot(cx, my, color=color, marker=mk, ms=5, lw=2,
                    ls='-' if color == 'C0' else '--', label=lbl, alpha=0.85)

    ax.axhline(1.0, color='grey', ls=':', lw=1, alpha=0.5)
    ax.set_xlabel(r'$\log_{10}(Z_\mathrm{hot})$')
    ax.set_ylabel(r'$r_\mathrm{cool} / R_\mathrm{vir}$')
    ax.set_title(r'$r_\mathrm{cool}/R_\mathrm{vir}$ vs Hot Gas Metallicity (fixed $M_\mathrm{vir}$)')
    ax.set_xlim(-4, -1)
    ax.set_ylim(0, 1.5)
    ax.legend(fontsize=7, ncol=2)


def plot_hotgas_fraction_vs_mvir(ax, sim1, sim2):
    """Hot gas fraction (Mhot/Mvir) vs halo mass — normalised gas content."""
    for sim, color, ls in [(sim1, 'C0', '-'), (sim2, 'C1', '--')]:
        centrals = (sim['Type'] == 0) & (sim['Len'] >= HALO_LEN_MIN) & (sim['Mvir'] > 0) & (sim['HotGas'] > 0)
        logmh = np.log10(sim['Mvir'][centrals])
        fhot = sim['HotGas'][centrals] / sim['Mvir'][centrals]

        bins = np.arange(10, 14.5, 0.2)
        cx, my, lo, hi = _binned_stat(logmh, np.log10(fhot), bins)
        ax.plot(cx, my, color=color, ls=ls, lw=2.5, label=sim['label'])
        ax.fill_between(cx, lo, hi, color=color, alpha=0.12)

    fb = 0.17
    ax.axhline(np.log10(fb), color='grey', ls=':', lw=1, alpha=0.5, label=r'$f_b$')
    ax.set_xlabel(r'$\log_{10}(M_\mathrm{vir}\;/\;\mathrm{M_\odot})$')
    ax.set_ylabel(r'$\log_{10}(M_\mathrm{hot} / M_\mathrm{vir})$')
    ax.set_title('Hot Gas Fraction vs Halo Mass (centrals)')
    ax.set_xlim(10, 14.5)
    ax.set_ylim(-3, -0.3)
    ax.legend(fontsize=9)


def plot_rcool_vs_hotgas_fraction(ax, sim1, sim2):
    """rcool/Rvir vs hot gas fraction at fixed Mvir — tests if gas content drives rcool."""
    mass_bins = [(11.5, 12.0), (12.0, 12.5), (12.5, 13.0)]
    markers = ['o', 's', 'D']

    for sim, color in [(sim1, 'C0'), (sim2, 'C1')]:
        centrals = (sim['Type'] == 0) & (sim['Len'] >= HALO_LEN_MIN) & (sim['Mvir'] > 0) & \
                   (sim['HotGas'] > 0) & (sim['RcoolToRvir'] > 0)
        logmh = np.log10(sim['Mvir'][centrals])
        log_fhot = np.log10(sim['HotGas'][centrals] / sim['Mvir'][centrals])
        rcool = sim['RcoolToRvir'][centrals]

        for (mlo, mhi), mk in zip(mass_bins, markers):
            in_bin = (logmh >= mlo) & (logmh < mhi)
            if in_bin.sum() < 20:
                continue

            fbins = np.arange(-3, -0.5, 0.25)
            cx, my, _, _ = _binned_stat(log_fhot[in_bin], rcool[in_bin], fbins, min_count=5)
            lbl = f"{sim['label']} [{mlo:.1f},{mhi:.1f})" if color == 'C0' else None
            ax.plot(cx, my, color=color, marker=mk, ms=5, lw=2,
                    ls='-' if color == 'C0' else '--', label=lbl, alpha=0.85)

    ax.axhline(1.0, color='grey', ls=':', lw=1, alpha=0.5)
    ax.set_xlabel(r'$\log_{10}(M_\mathrm{hot} / M_\mathrm{vir})$')
    ax.set_ylabel(r'$r_\mathrm{cool} / R_\mathrm{vir}$')
    ax.set_title(r'$r_\mathrm{cool}/R_\mathrm{vir}$ vs Hot Gas Fraction (fixed $M_\mathrm{vir}$)')
    ax.set_xlim(-3, -0.5)
    ax.set_ylim(0, 1.5)
    ax.legend(fontsize=7, ncol=2)


def plot_tvir_vs_mvir(ax, sim1, sim2):
    """Virial temperature (from Vvir) vs halo mass — checks cosmology effect on Tvir."""
    for sim, color, ls in [(sim1, 'C0', '-'), (sim2, 'C1', '--')]:
        centrals = (sim['Type'] == 0) & (sim['Len'] >= HALO_LEN_MIN) & (sim['Mvir'] > 0) & (sim['Vvir'] > 0)
        logmh = np.log10(sim['Mvir'][centrals])
        logTvir = np.log10(35.9 * sim['Vvir'][centrals]**2)

        bins = np.arange(10, 14.5, 0.2)
        cx, my, lo, hi = _binned_stat(logmh, logTvir, bins)
        ax.plot(cx, my, color=color, ls=ls, lw=2.5, label=sim['label'])
        ax.fill_between(cx, lo, hi, color=color, alpha=0.12)

    ax.set_xlabel(r'$\log_{10}(M_\mathrm{vir}\;/\;\mathrm{M_\odot})$')
    ax.set_ylabel(r'$\log_{10}(T_\mathrm{vir}\;/\;\mathrm{K})$')
    ax.set_title('Virial Temperature vs Halo Mass (centrals)')
    ax.set_xlim(10, 14.5)
    ax.legend(fontsize=9)


# ========== Summary statistics ==========

def print_summary(sim1, sim2):
    """Print a table of summary statistics."""
    print(f"\n{'='*70}")
    print(f"  SUMMARY STATISTICS COMPARISON")
    print(f"{'='*70}")
    print(f"{'Property':<35} {'  ' + sim1['label']:>15} {'  ' + sim2['label']:>15}")
    print(f"{'-'*70}")

    def row(name, v1, v2, fmt='.2e'):
        print(f"  {name:<33} {v1:{fmt}!s:>15} {v2:{fmt}!s:>15}")

    n1, n2 = len(sim1['StellarMass']), len(sim2['StellarMass'])
    print(f"  {'Total galaxies':<33} {n1:>15,} {n2:>15,}")
    print(f"  {'Volume (Mpc^3)':<33} {sim1['volume']:>15.1f} {sim2['volume']:>15.1f}")
    print(f"  {'Number density (Mpc^-3)':<33} {n1/sim1['volume']:>15.4f} {n2/sim2['volume']:>15.4f}")

    for mcut in [1e8, 1e9, 1e10, 1e11]:
        n1c = (sim1['StellarMass'] > mcut).sum()
        n2c = (sim2['StellarMass'] > mcut).sum()
        d1 = n1c / sim1['volume']
        d2 = n2c / sim2['volume']
        label = f"n(M* > {mcut:.0e}) Mpc^-3"
        print(f"  {label:<33} {d1:>15.4e} {d2:>15.4e}")

    # Type fractions
    for t, tname in [(0, 'Central'), (1, 'Satellite'), (2, 'Orphan')]:
        f1 = (sim1['Type'] == t).sum() / n1
        f2 = (sim2['Type'] == t).sum() / n2
        print(f"  {f'{tname} fraction':<33} {f1:>15.3f} {f2:>15.3f}")

    # Quenched fraction (M* > 1e9)
    for sim, label in [(sim1, sim1['label']), (sim2, sim2['label'])]:
        w = sim['StellarMass'] > 1e9
        sfr = sim['SfrDisk'][w] + sim['SfrBulge'][w]
        ssfr = np.where(sim['StellarMass'][w] > 0, sfr / sim['StellarMass'][w], 0)
        fq = (np.log10(ssfr + 1e-20) < sSFRcut).sum() / w.sum()
        # just store to print
        pass

    w1 = sim1['StellarMass'] > 1e9
    sfr1 = sim1['SfrDisk'][w1] + sim1['SfrBulge'][w1]
    ssfr1 = np.where(sim1['StellarMass'][w1] > 0, sfr1 / sim1['StellarMass'][w1], 0)
    fq1 = (np.log10(ssfr1 + 1e-20) < sSFRcut).sum() / w1.sum()

    w2 = sim2['StellarMass'] > 1e9
    sfr2 = sim2['SfrDisk'][w2] + sim2['SfrBulge'][w2]
    ssfr2 = np.where(sim2['StellarMass'][w2] > 0, sfr2 / sim2['StellarMass'][w2], 0)
    fq2 = (np.log10(ssfr2 + 1e-20) < sSFRcut).sum() / w2.sum()
    print(f"  {'Quenched frac (M*>1e9)':<33} {fq1:>15.3f} {fq2:>15.3f}")

    print(f"{'='*70}\n")


# ========== Evolution (lookback time) plots ==========

def _lookback_time_scalar(z, H0, Omega_m, Omega_L):
    """Lookback time in Gyr for a single redshift."""
    if z <= 0:
        return 0.0
    H0_s = H0 * 3.2408e-20  # km/s/Mpc -> s^-1
    integrand = lambda zp: 1.0 / ((1 + zp) * np.sqrt(Omega_m * (1 + zp)**3 + Omega_L))
    result, _ = _quad(integrand, 0, z)
    return result / H0_s / 3.1557e16  # seconds -> Gyr


def _lookback_times(z_arr, H0, Omega_m, Omega_L):
    """Lookback times in Gyr for an array of redshifts."""
    return np.array([_lookback_time_scalar(z, H0, Omega_m, Omega_L) for z in z_arr])


def _add_redshift_axis(ax, H0, Omega_m, Omega_L):
    """Add a top x-axis showing redshift corresponding to lookback time on the bottom."""
    ax2 = ax.twiny()
    z_ticks = [0, 0.5, 1, 2, 3, 5, 8, 12, 20]
    t_ticks = _lookback_times(np.array(z_ticks, dtype=float), H0, Omega_m, Omega_L)
    xlim = ax.get_xlim()
    valid_z = [z for z, t in zip(z_ticks, t_ticks) if xlim[0] <= t <= xlim[1]]
    valid_t = [t for z, t in zip(z_ticks, t_ticks) if xlim[0] <= t <= xlim[1]]
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(valid_t)
    ax2.set_xticklabels([f'{z:g}' for z in valid_z], fontsize=9)
    ax2.set_xlabel('Redshift', fontsize=10)
    return ax2


def load_sim_evolution(file_pattern, label):
    """Load aggregate galaxy properties at all available snapshots for evolution plots."""
    file_list = sorted(glob.glob(file_pattern))
    if not file_list:
        raise FileNotFoundError(f"No files matching: {file_pattern}")

    params = read_simulation_params(file_list[0])
    total_vf = sum(read_simulation_params(f)['VolumeFraction'] for f in file_list)

    h = params['Hubble_h']
    box = params['BoxSize']
    volume = (box / h) ** 3.0 * total_vf

    H0 = 100.0 * h
    Omega_m = params['Omega']
    Omega_L = params['OmegaLambda']

    snapshots = sorted(params['available_snapshots'])
    snap_redshifts = params['snapshot_redshifts']

    keys = [
        'z', 't_lb',
        'sfrd', 'rho_stellar', 'rho_cold', 'rho_hot', 'rho_bh',
        'rho_ejected', 'rho_cgm', 'rho_ics',
        'fq_9', 'fq_10',
        'median_ssfr_9', 'median_ssfr_10',
        'f_central', 'f_satellite', 'f_orphan',
        'median_fgas_9', 'median_Z_9', 'smhm_12',
    ]
    data = {k: [] for k in keys}

    print(f"\n  Loading evolution for {label} ({len(snapshots)} snapshots)...")

    for snap in snapshots:
        snap_key = f'Snap_{snap}'
        z = snap_redshifts[snap]
        t_lb = _lookback_time_scalar(z, H0, Omega_m, Omega_L)

        def read(param):
            return read_hdf(file_list, snap_key, param)
        def read_mass(param):
            return read(param) * 1.0e10 / h

        stellar = read_mass('StellarMass')
        if stellar.size == 0:
            continue

        sfr = read('SfrDisk') + read('SfrBulge')
        coldgas = read_mass('ColdGas')
        hotgas = read_mass('HotGas')
        bh = read_mass('BlackHoleMass')
        ejected = read_mass('EjectedMass')
        cgm = read_mass('CGMgas')
        ics = read_mass('IntraClusterStars')
        metals_s = read_mass('MetalsStellarMass')
        gtype = read('Type')
        mvir = read_mass('Mvir')

        data['z'].append(z)
        data['t_lb'].append(t_lb)

        # Cosmic densities
        data['sfrd'].append(np.sum(sfr) / volume)
        data['rho_stellar'].append(np.sum(stellar) / volume)
        data['rho_cold'].append(np.sum(coldgas) / volume)
        data['rho_hot'].append(np.sum(hotgas) / volume)
        data['rho_bh'].append(np.sum(bh) / volume)
        data['rho_ejected'].append(np.sum(ejected) / volume)
        data['rho_cgm'].append(np.sum(cgm) / volume)
        data['rho_ics'].append(np.sum(ics) / volume)

        # Quenched fractions
        for mcut, key in [(1e9, 'fq_9'), (1e10, 'fq_10')]:
            w = stellar > mcut
            if w.sum() > 10:
                ssfr_vals = np.where(sfr[w] > 0, np.log10(sfr[w] / stellar[w]), -14.0)
                data[key].append((ssfr_vals < sSFRcut).sum() / w.sum())
            else:
                data[key].append(np.nan)

        # Median sSFR (star-forming only)
        for mcut, key in [(1e9, 'median_ssfr_9'), (1e10, 'median_ssfr_10')]:
            w = (stellar > mcut) & (sfr > 0)
            if w.sum() > 10:
                data[key].append(np.median(np.log10(sfr[w] / stellar[w])))
            else:
                data[key].append(np.nan)

        # Type fractions
        n = len(gtype)
        data['f_central'].append((gtype == 0).sum() / n)
        data['f_satellite'].append((gtype == 1).sum() / n)
        data['f_orphan'].append((gtype == 2).sum() / n)

        # Median gas fraction (M* > 1e9)
        w = (stellar > 1e9) & (coldgas > 0)
        if w.sum() > 10:
            data['median_fgas_9'].append(np.median(coldgas[w] / (stellar[w] + coldgas[w])))
        else:
            data['median_fgas_9'].append(np.nan)

        # Median metallicity (M* > 1e9)
        w = (stellar > 1e9) & (metals_s > 0)
        if w.sum() > 10:
            data['median_Z_9'].append(np.median(np.log10(metals_s[w] / stellar[w] / 0.02)))
        else:
            data['median_Z_9'].append(np.nan)

        # SMHM at log Mvir ~ 12
        centrals = (gtype == 0) & (mvir > 0) & (stellar > 0)
        if centrals.sum() > 0:
            logmh = np.log10(mvir[centrals])
            w_12 = (logmh > 11.75) & (logmh < 12.25)
            if w_12.sum() > 10:
                data['smhm_12'].append(np.median(
                    stellar[centrals][w_12] / mvir[centrals][w_12]))
            else:
                data['smhm_12'].append(np.nan)
        else:
            data['smhm_12'].append(np.nan)

    # Convert to arrays and sort by lookback time
    for k in keys:
        data[k] = np.array(data[k])
    order = np.argsort(data['t_lb'])
    for k in keys:
        data[k] = data[k][order]

    data['label'] = label
    data['H0'] = H0
    data['Omega_m'] = Omega_m
    data['Omega_L'] = Omega_L
    data['volume'] = volume

    print(f"  Done. {len(data['z'])} snapshots, z = {data['z'].min():.2f} -- {data['z'].max():.2f}")
    return data


def plot_sfrd_evolution(ax, evo1, evo2):
    """Cosmic star formation rate density vs lookback time."""
    for evo, color, ls in [(evo1, 'C0', '-'), (evo2, 'C1', '--')]:
        w = evo['sfrd'] > 0
        ax.plot(evo['t_lb'][w], np.log10(evo['sfrd'][w]), color=color, ls=ls, lw=2.5,
                marker='o', ms=3, label=evo['label'])
    ax.set_xlabel('Lookback Time (Gyr)')
    ax.set_ylabel(r'$\log_{10}(\dot{\rho}_\star\;/\;\mathrm{M_\odot\,yr^{-1}\,Mpc^{-3}})$')
    ax.set_title('Cosmic SFR Density')
    ax.legend(fontsize=9)


def plot_stellar_density_evolution(ax, evo1, evo2):
    """Stellar mass density vs lookback time."""
    for evo, color, ls in [(evo1, 'C0', '-'), (evo2, 'C1', '--')]:
        w = evo['rho_stellar'] > 0
        ax.plot(evo['t_lb'][w], np.log10(evo['rho_stellar'][w]), color=color, ls=ls, lw=2.5,
                marker='o', ms=3, label=evo['label'])
    ax.set_xlabel('Lookback Time (Gyr)')
    ax.set_ylabel(r'$\log_{10}(\rho_\star\;/\;\mathrm{M_\odot\,Mpc^{-3}})$')
    ax.set_title('Stellar Mass Density')
    ax.legend(fontsize=9)


def plot_coldgas_density_evolution(ax, evo1, evo2):
    """Cold gas density vs lookback time."""
    for evo, color, ls in [(evo1, 'C0', '-'), (evo2, 'C1', '--')]:
        w = evo['rho_cold'] > 0
        ax.plot(evo['t_lb'][w], np.log10(evo['rho_cold'][w]), color=color, ls=ls, lw=2.5,
                marker='o', ms=3, label=evo['label'])
    ax.set_xlabel('Lookback Time (Gyr)')
    ax.set_ylabel(r'$\log_{10}(\rho_\mathrm{cold}\;/\;\mathrm{M_\odot\,Mpc^{-3}})$')
    ax.set_title('Cold Gas Density')
    ax.legend(fontsize=9)


def plot_hotgas_density_evolution(ax, evo1, evo2):
    """Hot gas density vs lookback time."""
    for evo, color, ls in [(evo1, 'C0', '-'), (evo2, 'C1', '--')]:
        w = evo['rho_hot'] > 0
        ax.plot(evo['t_lb'][w], np.log10(evo['rho_hot'][w]), color=color, ls=ls, lw=2.5,
                marker='o', ms=3, label=evo['label'])
    ax.set_xlabel('Lookback Time (Gyr)')
    ax.set_ylabel(r'$\log_{10}(\rho_\mathrm{hot}\;/\;\mathrm{M_\odot\,Mpc^{-3}})$')
    ax.set_title('Hot Gas Density')
    ax.legend(fontsize=9)


def plot_bh_density_evolution(ax, evo1, evo2):
    """Black hole mass density vs lookback time."""
    for evo, color, ls in [(evo1, 'C0', '-'), (evo2, 'C1', '--')]:
        w = evo['rho_bh'] > 0
        ax.plot(evo['t_lb'][w], np.log10(evo['rho_bh'][w]), color=color, ls=ls, lw=2.5,
                marker='o', ms=3, label=evo['label'])
    ax.set_xlabel('Lookback Time (Gyr)')
    ax.set_ylabel(r'$\log_{10}(\rho_\mathrm{BH}\;/\;\mathrm{M_\odot\,Mpc^{-3}})$')
    ax.set_title('Black Hole Mass Density')
    ax.legend(fontsize=9)


def plot_baryon_budget_evolution(ax, evo1, evo2):
    """All baryon component densities vs lookback time."""
    components = [
        ('rho_stellar', 'Stars', 'tab:blue'),
        ('rho_cold', 'Cold Gas', 'tab:cyan'),
        ('rho_hot', 'Hot Gas', 'tab:red'),
        ('rho_ejected', 'Ejected', 'tab:orange'),
        ('rho_cgm', 'CGM', 'tab:green'),
        ('rho_ics', 'ICS', 'tab:purple'),
        ('rho_bh', 'BH', 'black'),
    ]
    for evo, ls in [(evo1, '-'), (evo2, '--')]:
        for key, clabel, color in components:
            w = evo[key] > 0
            if w.any():
                lbl = clabel if ls == '-' else None
                ax.plot(evo['t_lb'][w], np.log10(evo[key][w]), color=color, ls=ls,
                        lw=1.8, label=lbl, alpha=0.85)
    ax.set_xlabel('Lookback Time (Gyr)')
    ax.set_ylabel(r'$\log_{10}(\rho\;/\;\mathrm{M_\odot\,Mpc^{-3}})$')
    ax.set_title('Baryon Budget (solid/dashed = sim1/sim2)')
    ax.legend(fontsize=7, ncol=2, loc='lower left')


def plot_quenched_frac_evolution(ax, evo1, evo2):
    """Quenched fraction vs lookback time at two stellar mass cuts."""
    for evo, color, ls in [(evo1, 'C0', '-'), (evo2, 'C1', '--')]:
        w = np.isfinite(evo['fq_9'])
        ax.plot(evo['t_lb'][w], evo['fq_9'][w], color=color, ls=ls, lw=2.5,
                marker='o', ms=3,
                label=rf"{evo['label']} ($M_\star > 10^9$)")
        w10 = np.isfinite(evo['fq_10'])
        if w10.any():
            ax.plot(evo['t_lb'][w10], evo['fq_10'][w10], color=color, ls=':', lw=1.5,
                    marker='s', ms=2, alpha=0.7,
                    label=rf"{evo['label']} ($M_\star > 10^{{10}}$)")
    ax.set_xlabel('Lookback Time (Gyr)')
    ax.set_ylabel('Quenched Fraction')
    ax.set_title('Quenched Fraction Evolution')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, ncol=2)


def plot_ssfr_evolution(ax, evo1, evo2):
    """Median sSFR of star-forming galaxies vs lookback time."""
    for evo, color, ls in [(evo1, 'C0', '-'), (evo2, 'C1', '--')]:
        w = np.isfinite(evo['median_ssfr_9'])
        ax.plot(evo['t_lb'][w], evo['median_ssfr_9'][w], color=color, ls=ls, lw=2.5,
                marker='o', ms=3,
                label=rf"{evo['label']} ($M_\star > 10^9$)")
        w10 = np.isfinite(evo['median_ssfr_10'])
        if w10.any():
            ax.plot(evo['t_lb'][w10], evo['median_ssfr_10'][w10], color=color, ls=':',
                    lw=1.5, marker='s', ms=2, alpha=0.7)
    ax.axhline(sSFRcut, color='grey', ls=':', lw=1, alpha=0.7)
    ax.set_xlabel('Lookback Time (Gyr)')
    ax.set_ylabel(r'Median $\log_{10}(\mathrm{sSFR}\;/\;\mathrm{yr^{-1}})$')
    ax.set_title('Median sSFR (star-forming)')
    ax.legend(fontsize=8)


def plot_type_frac_evolution(ax, evo1, evo2):
    """Central / satellite / orphan fractions vs lookback time."""
    for evo, ls in [(evo1, '-'), (evo2, '--')]:
        ax.plot(evo['t_lb'], evo['f_central'], color='tab:blue', ls=ls, lw=2.5,
                label='Central' if ls == '-' else None)
        ax.plot(evo['t_lb'], evo['f_satellite'], color='tab:orange', ls=ls, lw=2.5,
                label='Satellite' if ls == '-' else None)
        ax.plot(evo['t_lb'], evo['f_orphan'], color='tab:green', ls=ls, lw=2.5,
                label='Orphan' if ls == '-' else None)
    ax.set_xlabel('Lookback Time (Gyr)')
    ax.set_ylabel('Fraction')
    ax.set_title('Galaxy Type Fractions (solid/dashed = sim1/sim2)')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)


def plot_gas_frac_evolution(ax, evo1, evo2):
    """Median cold gas fraction vs lookback time (M* > 1e9)."""
    for evo, color, ls in [(evo1, 'C0', '-'), (evo2, 'C1', '--')]:
        w = np.isfinite(evo['median_fgas_9'])
        ax.plot(evo['t_lb'][w], evo['median_fgas_9'][w], color=color, ls=ls, lw=2.5,
                marker='o', ms=3, label=evo['label'])
    ax.set_xlabel('Lookback Time (Gyr)')
    ax.set_ylabel(r'Median $f_\mathrm{gas}$')
    ax.set_title(r'Cold Gas Fraction ($M_\star > 10^9\,\mathrm{M_\odot}$)')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)


def plot_metallicity_evolution(ax, evo1, evo2):
    """Median stellar metallicity vs lookback time (M* > 1e9)."""
    for evo, color, ls in [(evo1, 'C0', '-'), (evo2, 'C1', '--')]:
        w = np.isfinite(evo['median_Z_9'])
        ax.plot(evo['t_lb'][w], evo['median_Z_9'][w], color=color, ls=ls, lw=2.5,
                marker='o', ms=3, label=evo['label'])
    ax.set_xlabel('Lookback Time (Gyr)')
    ax.set_ylabel(r'Median $\log_{10}(Z_\star / Z_\odot)$')
    ax.set_title(r'Stellar Metallicity ($M_\star > 10^9\,\mathrm{M_\odot}$)')
    ax.legend(fontsize=9)


def plot_smhm_evolution(ax, evo1, evo2):
    """SMHM ratio at log Mvir ~ 12 vs lookback time."""
    for evo, color, ls in [(evo1, 'C0', '-'), (evo2, 'C1', '--')]:
        w = np.isfinite(evo['smhm_12'])
        if w.any():
            ax.plot(evo['t_lb'][w], np.log10(evo['smhm_12'][w]), color=color, ls=ls,
                    lw=2.5, marker='o', ms=3, label=evo['label'])
    ax.set_xlabel('Lookback Time (Gyr)')
    ax.set_ylabel(r'$\log_{10}(M_\star / M_\mathrm{vir})$')
    ax.set_title(r'SMHM Ratio (centrals, $\log M_\mathrm{vir} \approx 12$)')
    ax.legend(fontsize=9)


# ========== Main ==========

def parse_args():
    parser = argparse.ArgumentParser(description='Compare SAGE26 outputs from two simulations')
    parser.add_argument('--sim1', default='output/millennium/model_*.hdf5',
                        help='File pattern for simulation 1')
    parser.add_argument('--sim2', default='output/microuchuu/model_*.hdf5',
                        help='File pattern for simulation 2')
    parser.add_argument('--label1', default='miniMillennium', help='Label for sim 1')
    parser.add_argument('--label2', default='microUchuu', help='Label for sim 2')
    parser.add_argument('-o', '--output-dir', default='plots/comparison/',
                        help='Output directory')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    outdir = args.output_dir
    os.makedirs(outdir, exist_ok=True)

    np.random.seed(42)

    sim1 = load_sim(args.sim1, args.label1)
    sim2 = load_sim(args.sim2, args.label2)

    print_summary(sim1, sim2)

    # ---- Page 1: Core diagnostics (3x2) ----
    fig, axes = plt.subplots(3, 2, figsize=(14, 18))
    fig.suptitle('SAGE26: miniMillennium vs microUchuu  (z = 0)', fontsize=16, y=0.995)

    plot_smf(axes[0, 0], sim1, sim2)
    plot_hmf(axes[0, 1], sim1, sim2)
    plot_smhm(axes[1, 0], sim1, sim2)
    plot_ssfr(axes[1, 1], sim1, sim2)
    plot_quenched_fraction(axes[2, 0], sim1, sim2)
    plot_gas_fraction(axes[2, 1], sim1, sim2)

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    outfile1 = os.path.join(outdir, 'comparison_core' + OutputFormat)
    fig.savefig(outfile1, dpi=150)
    print(f'Saved: {outfile1}')

    # ---- Page 2: Secondary diagnostics (3x2) ----
    fig2, axes2 = plt.subplots(3, 2, figsize=(14, 18))
    fig2.suptitle('SAGE26: miniMillennium vs microUchuu  (z = 0)', fontsize=16, y=0.995)

    plot_bh_bulge(axes2[0, 0], sim1, sim2)
    plot_metallicity(axes2[0, 1], sim1, sim2)
    plot_cold_gas_mf(axes2[1, 0], sim1, sim2)
    plot_type_fractions(axes2[1, 1], sim1, sim2)
    plot_baryon_budget(axes2[2, 0], sim1, sim2)
    plot_concentration(axes2[2, 1], sim1, sim2)

    fig2.tight_layout(rect=[0, 0, 1, 0.98])
    outfile2 = os.path.join(outdir, 'comparison_secondary' + OutputFormat)
    fig2.savefig(outfile2, dpi=150)
    print(f'Saved: {outfile2}')

    # ---- Page 3: Gas supply / cooling diagnostics (3x2) ----
    fig3, axes3 = plt.subplots(3, 2, figsize=(14, 18))
    fig3.suptitle('SAGE26: Gas Supply & Cooling Diagnostics  (z = 0)', fontsize=16, y=0.995)

    plot_hotgas_vs_mvir(axes3[0, 0], sim1, sim2)
    plot_cgmgas_vs_mvir(axes3[0, 1], sim1, sim2)
    plot_coldgas_vs_mvir(axes3[1, 0], sim1, sim2)
    plot_zero_coldgas_fraction(axes3[1, 1], sim1, sim2)
    plot_rcool_vs_mvir(axes3[2, 0], sim1, sim2)
    plot_mdot_cool_vs_mvir(axes3[2, 1], sim1, sim2)

    fig3.tight_layout(rect=[0, 0, 1, 0.98])
    outfile3 = os.path.join(outdir, 'comparison_gas_supply' + OutputFormat)
    fig3.savefig(outfile3, dpi=150)
    print(f'Saved: {outfile3}')

    # ---- Page 4: Quenching diagnostics (3x2) ----
    fig4, axes4 = plt.subplots(3, 2, figsize=(14, 18))
    fig4.suptitle('SAGE26: Quenching Diagnostics  (z = 0)', fontsize=16, y=0.995)

    plot_quenched_central_vs_satellite(axes4[0, 0], sim1, sim2)
    plot_heating_vs_cooling(axes4[0, 1], sim1, sim2)
    plot_heating_cooling_ratio_vs_mvir(axes4[1, 0], sim1, sim2)
    plot_hot_regime_fraction(axes4[1, 1], sim1, sim2)
    plot_bh_mass_vs_mvir(axes4[2, 0], sim1, sim2)
    plot_ssfr_vs_mvir_centrals(axes4[2, 1], sim1, sim2)

    fig4.tight_layout(rect=[0, 0, 1, 0.98])
    outfile4 = os.path.join(outdir, 'comparison_quenching' + OutputFormat)
    fig4.savefig(outfile4, dpi=150)
    print(f'Saved: {outfile4}')

    # ---- Page 5: Concentration → cooling → quenching chain (3x2) ----
    fig5, axes5 = plt.subplots(3, 2, figsize=(14, 18))
    fig5.suptitle('SAGE26: Concentration → Cooling → Quenching  (z = 0)', fontsize=16, y=0.995)

    plot_concentration_distribution(axes5[0, 0], sim1, sim2)
    plot_rcool_vs_concentration(axes5[0, 1], sim1, sim2)
    plot_mdot_vs_concentration(axes5[1, 0], sim1, sim2)
    plot_ssfr_vs_concentration(axes5[1, 1], sim1, sim2)
    plot_quenched_vs_concentration(axes5[2, 0], sim1, sim2)
    plot_conc_residual_quenching(axes5[2, 1], sim1, sim2)

    fig5.tight_layout(rect=[0, 0, 1, 0.98])
    outfile5 = os.path.join(outdir, 'comparison_concentration_chain' + OutputFormat)
    fig5.savefig(outfile5, dpi=150)
    print(f'Saved: {outfile5}')

    # ---- Page 6: Metallicity & gas content → rcool (3x2) ----
    fig6, axes6 = plt.subplots(3, 2, figsize=(14, 18))
    fig6.suptitle('SAGE26: What drives the rcool difference?  (z = 0)', fontsize=16, y=0.995)

    plot_hotgas_metallicity_vs_mvir(axes6[0, 0], sim1, sim2)
    plot_cgm_metallicity_vs_mvir(axes6[0, 1], sim1, sim2)
    plot_rcool_vs_hotgas_metallicity(axes6[1, 0], sim1, sim2)
    plot_rcool_vs_hotgas_fraction(axes6[1, 1], sim1, sim2)
    plot_hotgas_fraction_vs_mvir(axes6[2, 0], sim1, sim2)
    plot_tvir_vs_mvir(axes6[2, 1], sim1, sim2)

    fig6.tight_layout(rect=[0, 0, 1, 0.98])
    outfile6 = os.path.join(outdir, 'comparison_metallicity_rcool' + OutputFormat)
    fig6.savefig(outfile6, dpi=150)
    print(f'Saved: {outfile6}')

    # ---- Page 7: Requested comparison grid (3x2) ----
    fig7, axes7 = plt.subplots(3, 2, figsize=(14, 18))
    fig7.suptitle('SAGE26: Requested Comparison Panels  (z = 0)', fontsize=16, y=0.995)

    plot_h1_mf(axes7[0, 0], sim1, sim2)
    plot_h2_mf(axes7[0, 1], sim1, sim2)
    plot_icl_fraction(axes7[1, 0], sim1, sim2, aperture_kpc=300.0)
    plot_baryonic_tully_fisher(axes7[1, 1], sim1, sim2)
    plot_bh_mass_function(axes7[2, 0], sim1, sim2)
    plot_metallicity(axes7[2, 1], sim1, sim2)

    fig7.tight_layout(rect=[0, 0, 1, 0.98])
    outfile7 = os.path.join(outdir, 'comparison_requested_grid' + OutputFormat)
    fig7.savefig(outfile7, dpi=150)
    print(f'Saved: {outfile7}')

    # ---- Page 8: Groups & clusters / members / BCG (3x2) ----
    fig8, axes8 = plt.subplots(3, 2, figsize=(14, 18))
    fig8.suptitle('SAGE26: Groups & Clusters (Members + BCG)  (z = 0)', fontsize=16, y=0.995)

    plot_group_satellite_occupation(axes8[0, 0], sim1, sim2, mstar_thresh=1e9)
    plot_satellite_conditional_smf(axes8[0, 1], sim1, sim2, mvir_bins=((13.0, 14.0), (14.0, 15.0)))
    plot_bcg_mass_vs_mvir(axes8[1, 0], sim1, sim2)
    plot_bcg_fraction_of_group_stellar_mass(axes8[1, 1], sim1, sim2)
    plot_bcg_mass_gap(axes8[2, 0], sim1, sim2, mstar_min=1e9)
    plot_satellite_radial_profile(axes8[2, 1], sim1, sim2, mvir_range=(14.0, 15.0), rmax=2.0)

    fig8.tight_layout(rect=[0, 0, 1, 0.98])
    outfile8 = os.path.join(outdir, 'comparison_groups_clusters' + OutputFormat)
    fig8.savefig(outfile8, dpi=150)
    print(f'Saved: {outfile8}')

    # ---- Evolution plots (vs lookback time) ----
    print("\nLoading evolution data...")
    evo1 = load_sim_evolution(args.sim1, args.label1)
    evo2 = load_sim_evolution(args.sim2, args.label2)

    # Cosmology for redshift axis (use sim1)
    cosmo = (evo1['H0'], evo1['Omega_m'], evo1['Omega_L'])

    # ---- Page 9: Cosmic density evolution (3x2) ----
    fig9, axes9 = plt.subplots(3, 2, figsize=(14, 18))
    fig9.suptitle(f"{evo1['label']} vs {evo2['label']} — Density Evolution",
                  fontsize=16, y=0.995)

    plot_sfrd_evolution(axes9[0, 0], evo1, evo2)
    plot_stellar_density_evolution(axes9[0, 1], evo1, evo2)
    plot_coldgas_density_evolution(axes9[1, 0], evo1, evo2)
    plot_hotgas_density_evolution(axes9[1, 1], evo1, evo2)
    plot_bh_density_evolution(axes9[2, 0], evo1, evo2)
    plot_baryon_budget_evolution(axes9[2, 1], evo1, evo2)

    for row in axes9:
        for ax in row:
            _add_redshift_axis(ax, *cosmo)

    fig9.subplots_adjust(hspace=0.45, wspace=0.3, top=0.93, bottom=0.05,
                         left=0.08, right=0.97)
    outfile9 = os.path.join(outdir, 'comparison_evolution_densities' + OutputFormat)
    fig9.savefig(outfile9, dpi=150)
    print(f'Saved: {outfile9}')

    # ---- Page 10: Population property evolution (3x2) ----
    fig10, axes10 = plt.subplots(3, 2, figsize=(14, 18))
    fig10.suptitle(f"{evo1['label']} vs {evo2['label']} — Population Evolution",
                   fontsize=16, y=0.995)

    plot_quenched_frac_evolution(axes10[0, 0], evo1, evo2)
    plot_ssfr_evolution(axes10[0, 1], evo1, evo2)
    plot_type_frac_evolution(axes10[1, 0], evo1, evo2)
    plot_gas_frac_evolution(axes10[1, 1], evo1, evo2)
    plot_metallicity_evolution(axes10[2, 0], evo1, evo2)
    plot_smhm_evolution(axes10[2, 1], evo1, evo2)

    for row in axes10:
        for ax in row:
            _add_redshift_axis(ax, *cosmo)

    fig10.subplots_adjust(hspace=0.45, wspace=0.3, top=0.93, bottom=0.05,
                          left=0.08, right=0.97)
    outfile10 = os.path.join(outdir, 'comparison_evolution_populations' + OutputFormat)
    fig10.savefig(outfile10, dpi=150)
    print(f'Saved: {outfile10}')

    plt.close('all')
    print('\nDone.')

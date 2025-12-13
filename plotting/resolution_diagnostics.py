"""
Resolution Diagnostics: Compare Millennium vs miniUchuu
========================================================
This script compares key properties at fixed stellar mass to diagnose
why higher-resolution simulations produce fewer quiescent massive galaxies.

Key comparisons:
1. Black hole mass at fixed stellar mass
2. Hot gas mass at fixed stellar mass
3. AGN heating rate at fixed stellar mass
4. Quiescent fraction vs stellar mass
5. sSFR distribution at high masses

Run this script on the HPC where miniUchuu output is available.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import os
from pathlib import Path

# ========================== CONFIGURATION ==========================

# Millennium configuration
MILLENNIUM_CONFIG = {
    'name': 'Millennium',
    'path': './output/millennium/',  # Adjust path as needed
    'snapshot': 'Snap_63',  # z=0
    'Hubble_h': 0.73,
    'BoxSize': 62.5,  # Mpc/h
    'PartMass': 0.0860657,  # 10^10 Msun/h
    'color': 'black',
    'linestyle': '-'
}

# miniUchuu configuration
MINIUCHUU_CONFIG = {
    'name': 'miniUchuu',
    'path': '/fred/oz004/mbradley/SAGE-GAS/sage-model/output/testmodels/miniUchuu_FFB/',
    'snapshot': 'Snap_49',  # z=0 for miniUchuu
    'Hubble_h': 0.6774,
    'BoxSize': 400.0,  # Mpc/h
    'PartMass': 0.0325,  # 10^10 Msun/h
    'color': '#1f77b4',
    'linestyle': '--'
}

# Output directory for diagnostic plots
OUTPUT_DIR = './output/millennium/plots/diagnostics/'

# ========================== DATA READING ==========================

def read_galaxy_property(directory, snapshot, param):
    """Read a galaxy property from all HDF5 files in directory"""
    data_list = []

    # Find all model files
    model_files = sorted([f for f in os.listdir(directory) if f.startswith('model_') and f.endswith('.hdf5') and f != 'model.hdf5'])

    if not model_files:
        print(f"No model files found in {directory}")
        return np.array([])

    for model_file in model_files:
        filepath = os.path.join(directory, model_file)
        try:
            with h5.File(filepath, 'r') as f:
                if snapshot in f and param in f[snapshot]:
                    data_list.append(f[snapshot][param][:])
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue

    if data_list:
        return np.concatenate(data_list)
    return np.array([])


def load_simulation_data(config):
    """Load all required properties for a simulation"""
    print(f"\nLoading {config['name']} data from {config['path']}...")

    directory = config['path']
    snapshot = config['snapshot']
    h = config['Hubble_h']

    data = {}

    # Essential properties
    properties = [
        'StellarMass', 'BlackHoleMass', 'HotGas', 'ColdGas', 'CGMgas',
        'SfrDisk', 'SfrBulge', 'Mvir', 'Vvir', 'Type', 'Heating', 'Cooling'
    ]

    for prop in properties:
        raw_data = read_galaxy_property(directory, snapshot, prop)
        if len(raw_data) > 0:
            # Convert masses to physical units (Msun)
            if prop in ['StellarMass', 'BlackHoleMass', 'HotGas', 'ColdGas', 'CGMgas', 'Mvir']:
                data[prop] = raw_data * 1e10 / h
            else:
                data[prop] = raw_data
            print(f"  {prop}: {len(data[prop])} galaxies")
        else:
            print(f"  {prop}: NOT FOUND or empty")
            data[prop] = np.array([])

    return data


# ========================== ANALYSIS FUNCTIONS ==========================

def calculate_quiescent_fraction(stellar_mass, sfr, mass_bins):
    """Calculate quiescent fraction in stellar mass bins"""
    # sSFR threshold for quiescence: log(sSFR) < -11
    log_ssfr = np.log10(sfr / stellar_mass + 1e-15)
    is_quiescent = log_ssfr < -11

    bin_centers = (mass_bins[:-1] + mass_bins[1:]) / 2
    quiescent_frac = []
    quiescent_frac_err = []
    n_galaxies = []

    for i in range(len(mass_bins) - 1):
        mask = (np.log10(stellar_mass) >= mass_bins[i]) & (np.log10(stellar_mass) < mass_bins[i+1])
        n_in_bin = np.sum(mask)
        n_galaxies.append(n_in_bin)

        if n_in_bin > 10:
            frac = np.sum(is_quiescent[mask]) / n_in_bin
            # Binomial error
            err = np.sqrt(frac * (1 - frac) / n_in_bin)
            quiescent_frac.append(frac)
            quiescent_frac_err.append(err)
        else:
            quiescent_frac.append(np.nan)
            quiescent_frac_err.append(np.nan)

    return bin_centers, np.array(quiescent_frac), np.array(quiescent_frac_err), np.array(n_galaxies)


def calculate_median_in_bins(x_data, y_data, x_bins, min_count=20):
    """Calculate median and scatter of y in bins of x"""
    log_x = np.log10(x_data + 1e-15)
    log_y = np.log10(y_data + 1e-15)

    bin_centers = (x_bins[:-1] + x_bins[1:]) / 2
    medians = []
    p16 = []
    p84 = []
    counts = []

    for i in range(len(x_bins) - 1):
        mask = (log_x >= x_bins[i]) & (log_x < x_bins[i+1]) & np.isfinite(log_y)
        n_in_bin = np.sum(mask)
        counts.append(n_in_bin)

        if n_in_bin >= min_count:
            medians.append(np.median(log_y[mask]))
            p16.append(np.percentile(log_y[mask], 16))
            p84.append(np.percentile(log_y[mask], 84))
        else:
            medians.append(np.nan)
            p16.append(np.nan)
            p84.append(np.nan)

    return bin_centers, np.array(medians), np.array(p16), np.array(p84), np.array(counts)


# ========================== PLOTTING FUNCTIONS ==========================

def plot_quiescent_fraction(mill_data, uchuu_data, output_dir):
    """Plot quiescent fraction vs stellar mass"""
    fig, ax = plt.subplots(figsize=(8, 6))

    mass_bins = np.arange(9.0, 12.5, 0.25)

    for config, data, label_suffix in [
        (MILLENNIUM_CONFIG, mill_data, ''),
        (MINIUCHUU_CONFIG, uchuu_data, '')
    ]:
        if len(data['StellarMass']) == 0:
            continue

        # Select central galaxies only
        centrals = data['Type'] == 0
        stellar_mass = data['StellarMass'][centrals]
        sfr = data['SfrDisk'][centrals] + data['SfrBulge'][centrals]

        bin_centers, q_frac, q_err, n_gal = calculate_quiescent_fraction(stellar_mass, sfr, mass_bins)

        valid = ~np.isnan(q_frac)
        ax.plot(bin_centers[valid], q_frac[valid],
                color=config['color'], linestyle=config['linestyle'], linewidth=2,
                label=f"{config['name']} (N={np.sum(n_gal[valid]):.0f})")
        ax.fill_between(bin_centers[valid],
                        q_frac[valid] - q_err[valid],
                        q_frac[valid] + q_err[valid],
                        color=config['color'], alpha=0.2)

    ax.set_xlabel(r'$\log_{10}(M_*/M_\odot)$', fontsize=14)
    ax.set_ylabel('Quiescent Fraction', fontsize=14)
    ax.set_xlim(9, 12.5)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=12)
    ax.set_title('Quiescent Fraction vs Stellar Mass (Centrals Only)', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'quiescent_fraction_comparison.png'), dpi=200)
    plt.close()
    print(f"Saved: quiescent_fraction_comparison.png")


def plot_bh_mass_vs_stellar_mass(mill_data, uchuu_data, output_dir):
    """Plot black hole mass vs stellar mass"""
    fig, ax = plt.subplots(figsize=(8, 6))

    mass_bins = np.arange(9.0, 12.5, 0.25)

    for config, data in [(MILLENNIUM_CONFIG, mill_data), (MINIUCHUU_CONFIG, uchuu_data)]:
        if len(data['StellarMass']) == 0:
            continue

        centrals = (data['Type'] == 0) & (data['BlackHoleMass'] > 0)
        stellar_mass = data['StellarMass'][centrals]
        bh_mass = data['BlackHoleMass'][centrals]

        bin_centers, medians, p16, p84, counts = calculate_median_in_bins(stellar_mass, bh_mass, mass_bins)

        valid = ~np.isnan(medians)
        ax.plot(bin_centers[valid], medians[valid],
                color=config['color'], linestyle=config['linestyle'], linewidth=2,
                label=config['name'])
        ax.fill_between(bin_centers[valid], p16[valid], p84[valid],
                        color=config['color'], alpha=0.2)

    # Add 1:1000 relation for reference
    x_ref = np.array([9, 12.5])
    ax.plot(x_ref, x_ref - 3, 'k:', alpha=0.5, label=r'$M_{BH}/M_* = 10^{-3}$')

    ax.set_xlabel(r'$\log_{10}(M_*/M_\odot)$', fontsize=14)
    ax.set_ylabel(r'$\log_{10}(M_{BH}/M_\odot)$', fontsize=14)
    ax.set_xlim(9, 12.5)
    ax.set_ylim(5, 10)
    ax.legend(fontsize=12)
    ax.set_title('Black Hole Mass vs Stellar Mass (Centrals)', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bh_mass_comparison.png'), dpi=200)
    plt.close()
    print(f"Saved: bh_mass_comparison.png")


def plot_hot_gas_vs_stellar_mass(mill_data, uchuu_data, output_dir):
    """Plot hot gas mass vs stellar mass"""
    fig, ax = plt.subplots(figsize=(8, 6))

    mass_bins = np.arange(9.0, 12.5, 0.25)

    for config, data in [(MILLENNIUM_CONFIG, mill_data), (MINIUCHUU_CONFIG, uchuu_data)]:
        if len(data['StellarMass']) == 0:
            continue

        centrals = (data['Type'] == 0) & (data['HotGas'] > 0)
        stellar_mass = data['StellarMass'][centrals]
        hot_gas = data['HotGas'][centrals]

        bin_centers, medians, p16, p84, counts = calculate_median_in_bins(stellar_mass, hot_gas, mass_bins)

        valid = ~np.isnan(medians)
        ax.plot(bin_centers[valid], medians[valid],
                color=config['color'], linestyle=config['linestyle'], linewidth=2,
                label=config['name'])
        ax.fill_between(bin_centers[valid], p16[valid], p84[valid],
                        color=config['color'], alpha=0.2)

    ax.set_xlabel(r'$\log_{10}(M_*/M_\odot)$', fontsize=14)
    ax.set_ylabel(r'$\log_{10}(M_{HotGas}/M_\odot)$', fontsize=14)
    ax.set_xlim(9, 12.5)
    ax.set_ylim(8, 13)
    ax.legend(fontsize=12)
    ax.set_title('Hot Gas Mass vs Stellar Mass (Centrals)', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hot_gas_comparison.png'), dpi=200)
    plt.close()
    print(f"Saved: hot_gas_comparison.png")


def plot_cgm_gas_vs_stellar_mass(mill_data, uchuu_data, output_dir):
    """Plot CGM gas mass vs stellar mass"""
    fig, ax = plt.subplots(figsize=(8, 6))

    mass_bins = np.arange(9.0, 12.5, 0.25)

    for config, data in [(MILLENNIUM_CONFIG, mill_data), (MINIUCHUU_CONFIG, uchuu_data)]:
        if len(data['StellarMass']) == 0:
            continue

        if 'CGMgas' not in data or len(data['CGMgas']) == 0:
            print(f"  {config['name']}: CGMgas not available")
            continue

        centrals = (data['Type'] == 0) & (data['CGMgas'] > 0)
        stellar_mass = data['StellarMass'][centrals]
        cgm_gas = data['CGMgas'][centrals]

        bin_centers, medians, p16, p84, counts = calculate_median_in_bins(stellar_mass, cgm_gas, mass_bins)

        valid = ~np.isnan(medians)
        ax.plot(bin_centers[valid], medians[valid],
                color=config['color'], linestyle=config['linestyle'], linewidth=2,
                label=config['name'])
        ax.fill_between(bin_centers[valid], p16[valid], p84[valid],
                        color=config['color'], alpha=0.2)

    ax.set_xlabel(r'$\log_{10}(M_*/M_\odot)$', fontsize=14)
    ax.set_ylabel(r'$\log_{10}(M_{CGM}/M_\odot)$', fontsize=14)
    ax.set_xlim(9, 12.5)
    ax.set_ylim(8, 13)
    ax.legend(fontsize=12)
    ax.set_title('CGM Gas Mass vs Stellar Mass (Centrals)', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cgm_gas_comparison.png'), dpi=200)
    plt.close()
    print(f"Saved: cgm_gas_comparison.png")


def plot_heating_vs_stellar_mass(mill_data, uchuu_data, output_dir):
    """Plot AGN heating rate vs stellar mass"""
    fig, ax = plt.subplots(figsize=(8, 6))

    mass_bins = np.arange(9.0, 12.5, 0.25)

    for config, data in [(MILLENNIUM_CONFIG, mill_data), (MINIUCHUU_CONFIG, uchuu_data)]:
        if len(data['StellarMass']) == 0:
            continue

        if 'Heating' not in data or len(data['Heating']) == 0:
            print(f"  {config['name']}: Heating not available")
            continue

        centrals = (data['Type'] == 0) & (data['Heating'] > 0)
        stellar_mass = data['StellarMass'][centrals]
        heating = data['Heating'][centrals]

        bin_centers, medians, p16, p84, counts = calculate_median_in_bins(stellar_mass, heating, mass_bins)

        valid = ~np.isnan(medians)
        ax.plot(bin_centers[valid], medians[valid],
                color=config['color'], linestyle=config['linestyle'], linewidth=2,
                label=config['name'])
        ax.fill_between(bin_centers[valid], p16[valid], p84[valid],
                        color=config['color'], alpha=0.2)

    ax.set_xlabel(r'$\log_{10}(M_*/M_\odot)$', fontsize=14)
    ax.set_ylabel(r'$\log_{10}(\mathrm{Heating})$', fontsize=14)
    ax.set_xlim(9, 12.5)
    ax.legend(fontsize=12)
    ax.set_title('AGN Heating vs Stellar Mass (Centrals)', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'agn_heating_comparison.png'), dpi=200)
    plt.close()
    print(f"Saved: agn_heating_comparison.png")


def plot_ssfr_distribution_high_mass(mill_data, uchuu_data, output_dir):
    """Plot sSFR distribution for high-mass galaxies"""
    fig, ax = plt.subplots(figsize=(8, 6))

    # High mass cut: M* > 10^10.5 Msun
    mass_cut = 10**10.5

    for config, data in [(MILLENNIUM_CONFIG, mill_data), (MINIUCHUU_CONFIG, uchuu_data)]:
        if len(data['StellarMass']) == 0:
            continue

        centrals = (data['Type'] == 0) & (data['StellarMass'] > mass_cut)
        stellar_mass = data['StellarMass'][centrals]
        sfr = data['SfrDisk'][centrals] + data['SfrBulge'][centrals]

        log_ssfr = np.log10(sfr / stellar_mass + 1e-15)
        log_ssfr = log_ssfr[np.isfinite(log_ssfr)]

        ax.hist(log_ssfr, bins=50, range=(-14, -8), density=True,
                histtype='step', color=config['color'], linestyle=config['linestyle'],
                linewidth=2, label=f"{config['name']} (N={len(log_ssfr)})")

    # Mark quiescent threshold
    ax.axvline(-11, color='red', linestyle=':', linewidth=2, label='Quiescent threshold')

    ax.set_xlabel(r'$\log_{10}(\mathrm{sSFR}/\mathrm{yr}^{-1})$', fontsize=14)
    ax.set_ylabel('Normalized density', fontsize=14)
    ax.set_xlim(-14, -8)
    ax.legend(fontsize=11)
    ax.set_title(r'sSFR Distribution for $M_* > 10^{10.5} M_\odot$ (Centrals)', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ssfr_distribution_high_mass.png'), dpi=200)
    plt.close()
    print(f"Saved: ssfr_distribution_high_mass.png")


def plot_bh_to_stellar_ratio(mill_data, uchuu_data, output_dir):
    """Plot M_BH / M_* ratio vs stellar mass"""
    fig, ax = plt.subplots(figsize=(8, 6))

    mass_bins = np.arange(9.0, 12.5, 0.25)

    for config, data in [(MILLENNIUM_CONFIG, mill_data), (MINIUCHUU_CONFIG, uchuu_data)]:
        if len(data['StellarMass']) == 0:
            continue

        centrals = (data['Type'] == 0) & (data['BlackHoleMass'] > 0) & (data['StellarMass'] > 0)
        stellar_mass = data['StellarMass'][centrals]
        bh_ratio = data['BlackHoleMass'][centrals] / stellar_mass

        bin_centers, medians, p16, p84, counts = calculate_median_in_bins(stellar_mass, bh_ratio, mass_bins)

        valid = ~np.isnan(medians)
        ax.plot(bin_centers[valid], medians[valid],
                color=config['color'], linestyle=config['linestyle'], linewidth=2,
                label=config['name'])
        ax.fill_between(bin_centers[valid], p16[valid], p84[valid],
                        color=config['color'], alpha=0.2)

    ax.axhline(-3, color='gray', linestyle=':', alpha=0.7, label=r'$M_{BH}/M_* = 10^{-3}$')

    ax.set_xlabel(r'$\log_{10}(M_*/M_\odot)$', fontsize=14)
    ax.set_ylabel(r'$\log_{10}(M_{BH}/M_*)$', fontsize=14)
    ax.set_xlim(9, 12.5)
    ax.set_ylim(-5, -1)
    ax.legend(fontsize=12)
    ax.set_title('BH-to-Stellar Mass Ratio (Centrals)', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bh_stellar_ratio_comparison.png'), dpi=200)
    plt.close()
    print(f"Saved: bh_stellar_ratio_comparison.png")


def plot_hot_gas_to_mvir_ratio(mill_data, uchuu_data, output_dir):
    """Plot HotGas / Mvir ratio vs stellar mass - key for radio mode AGN"""
    fig, ax = plt.subplots(figsize=(8, 6))

    mass_bins = np.arange(9.0, 12.5, 0.25)

    for config, data in [(MILLENNIUM_CONFIG, mill_data), (MINIUCHUU_CONFIG, uchuu_data)]:
        if len(data['StellarMass']) == 0:
            continue

        centrals = (data['Type'] == 0) & (data['HotGas'] > 0) & (data['Mvir'] > 0)
        stellar_mass = data['StellarMass'][centrals]
        hot_gas_ratio = data['HotGas'][centrals] / data['Mvir'][centrals]

        bin_centers, medians, p16, p84, counts = calculate_median_in_bins(stellar_mass, hot_gas_ratio, mass_bins)

        valid = ~np.isnan(medians)
        ax.plot(bin_centers[valid], medians[valid],
                color=config['color'], linestyle=config['linestyle'], linewidth=2,
                label=config['name'])
        ax.fill_between(bin_centers[valid], p16[valid], p84[valid],
                        color=config['color'], alpha=0.2)

    # Reference line at cosmic baryon fraction
    ax.axhline(np.log10(0.17), color='gray', linestyle=':', alpha=0.7, label=r'$f_b = 0.17$')

    ax.set_xlabel(r'$\log_{10}(M_*/M_\odot)$', fontsize=14)
    ax.set_ylabel(r'$\log_{10}(M_{HotGas}/M_{vir})$', fontsize=14)
    ax.set_xlim(9, 12.5)
    ax.set_ylim(-3, 0)
    ax.legend(fontsize=12)
    ax.set_title('Hot Gas Fraction vs Stellar Mass (Centrals)', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hot_gas_fraction_comparison.png'), dpi=200)
    plt.close()
    print(f"Saved: hot_gas_fraction_comparison.png")


def plot_cold_gas_vs_stellar_mass(mill_data, uchuu_data, output_dir):
    """Plot cold gas mass vs stellar mass"""
    fig, ax = plt.subplots(figsize=(8, 6))

    mass_bins = np.arange(9.0, 12.5, 0.25)

    for config, data in [(MILLENNIUM_CONFIG, mill_data), (MINIUCHUU_CONFIG, uchuu_data)]:
        if len(data['StellarMass']) == 0:
            continue

        centrals = (data['Type'] == 0) & (data['ColdGas'] > 0)
        stellar_mass = data['StellarMass'][centrals]
        cold_gas = data['ColdGas'][centrals]

        bin_centers, medians, p16, p84, counts = calculate_median_in_bins(stellar_mass, cold_gas, mass_bins)

        valid = ~np.isnan(medians)
        ax.plot(bin_centers[valid], medians[valid],
                color=config['color'], linestyle=config['linestyle'], linewidth=2,
                label=config['name'])
        ax.fill_between(bin_centers[valid], p16[valid], p84[valid],
                        color=config['color'], alpha=0.2)

    ax.set_xlabel(r'$\log_{10}(M_*/M_\odot)$', fontsize=14)
    ax.set_ylabel(r'$\log_{10}(M_{ColdGas}/M_\odot)$', fontsize=14)
    ax.set_xlim(9, 12.5)
    ax.set_ylim(7, 11)
    ax.legend(fontsize=12)
    ax.set_title('Cold Gas Mass vs Stellar Mass (Centrals)', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cold_gas_comparison.png'), dpi=200)
    plt.close()
    print(f"Saved: cold_gas_comparison.png")


def plot_baryonic_tully_fisher(mill_data, uchuu_data, output_dir):
    """Plot baryonic Tully-Fisher relation: M_baryon vs V_vir"""
    fig, ax = plt.subplots(figsize=(8, 6))

    vvir_bins = np.arange(1.6, 2.8, 0.1)  # log(Vvir) bins

    for config, data in [(MILLENNIUM_CONFIG, mill_data), (MINIUCHUU_CONFIG, uchuu_data)]:
        if len(data['StellarMass']) == 0:
            continue

        centrals = (data['Type'] == 0) & (data['Vvir'] > 0) & (data['StellarMass'] > 0)
        vvir = data['Vvir'][centrals]
        stellar_mass = data['StellarMass'][centrals]
        cold_gas = data['ColdGas'][centrals]

        # Baryonic mass = stars + cold gas
        m_baryon = stellar_mass + cold_gas

        log_vvir = np.log10(vvir)
        log_mbaryon = np.log10(m_baryon)

        bin_centers = (vvir_bins[:-1] + vvir_bins[1:]) / 2
        medians = []
        p16_list = []
        p84_list = []

        for i in range(len(vvir_bins) - 1):
            mask = (log_vvir >= vvir_bins[i]) & (log_vvir < vvir_bins[i+1])
            if np.sum(mask) >= 20:
                medians.append(np.median(log_mbaryon[mask]))
                p16_list.append(np.percentile(log_mbaryon[mask], 16))
                p84_list.append(np.percentile(log_mbaryon[mask], 84))
            else:
                medians.append(np.nan)
                p16_list.append(np.nan)
                p84_list.append(np.nan)

        medians = np.array(medians)
        p16_arr = np.array(p16_list)
        p84_arr = np.array(p84_list)

        valid = ~np.isnan(medians)
        ax.plot(bin_centers[valid], medians[valid],
                color=config['color'], linestyle=config['linestyle'], linewidth=2,
                label=config['name'])
        ax.fill_between(bin_centers[valid], p16_arr[valid], p84_arr[valid],
                        color=config['color'], alpha=0.2)

    # Add reference BTF slope (M_baryon ∝ V^4)
    v_ref = np.array([1.8, 2.6])
    m_ref = 4 * v_ref + 1.5  # Approximate normalization
    ax.plot(v_ref, m_ref, 'k:', alpha=0.5, label=r'$M_b \propto V^4$')

    ax.set_xlabel(r'$\log_{10}(V_{vir}/\mathrm{km\,s}^{-1})$', fontsize=14)
    ax.set_ylabel(r'$\log_{10}(M_{baryon}/M_\odot)$', fontsize=14)
    ax.set_xlim(1.6, 2.7)
    ax.set_ylim(8, 12)
    ax.legend(fontsize=12)
    ax.set_title('Baryonic Tully-Fisher Relation (Centrals)', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'baryonic_tully_fisher.png'), dpi=200)
    plt.close()
    print(f"Saved: baryonic_tully_fisher.png")


def plot_baryon_fraction_vs_mvir(mill_data, uchuu_data, output_dir):
    """Plot baryon fraction vs halo mass - key diagnostic for gas deficit"""
    fig, ax = plt.subplots(figsize=(8, 6))

    mvir_bins = np.arange(10.0, 14.0, 0.25)

    for config, data in [(MILLENNIUM_CONFIG, mill_data), (MINIUCHUU_CONFIG, uchuu_data)]:
        if len(data['StellarMass']) == 0:
            continue

        centrals = (data['Type'] == 0) & (data['Mvir'] > 0)
        mvir = data['Mvir'][centrals]
        stellar_mass = data['StellarMass'][centrals]
        cold_gas = data['ColdGas'][centrals]
        hot_gas = data['HotGas'][centrals]
        cgm_gas = data['CGMgas'][centrals] if 'CGMgas' in data and len(data['CGMgas']) > 0 else np.zeros_like(hot_gas)
        bh_mass = data['BlackHoleMass'][centrals]

        # Total baryon fraction
        total_baryons = stellar_mass + cold_gas + hot_gas + cgm_gas + bh_mass
        f_baryon = total_baryons / mvir

        log_mvir = np.log10(mvir)
        log_fbaryon = np.log10(f_baryon + 1e-15)

        bin_centers = (mvir_bins[:-1] + mvir_bins[1:]) / 2
        medians = []
        p16_list = []
        p84_list = []

        for i in range(len(mvir_bins) - 1):
            mask = (log_mvir >= mvir_bins[i]) & (log_mvir < mvir_bins[i+1])
            if np.sum(mask) >= 20:
                medians.append(np.median(log_fbaryon[mask]))
                p16_list.append(np.percentile(log_fbaryon[mask], 16))
                p84_list.append(np.percentile(log_fbaryon[mask], 84))
            else:
                medians.append(np.nan)
                p16_list.append(np.nan)
                p84_list.append(np.nan)

        medians = np.array(medians)
        p16_arr = np.array(p16_list)
        p84_arr = np.array(p84_list)

        valid = ~np.isnan(medians)
        ax.plot(bin_centers[valid], medians[valid],
                color=config['color'], linestyle=config['linestyle'], linewidth=2,
                label=config['name'])
        ax.fill_between(bin_centers[valid], p16_arr[valid], p84_arr[valid],
                        color=config['color'], alpha=0.2)

    # Add cosmic baryon fraction reference
    ax.axhline(np.log10(0.17), color='gray', linestyle=':', alpha=0.7, label=r'$f_b = 0.17$')
    ax.axhline(np.log10(0.15), color='gray', linestyle='--', alpha=0.5, label=r'$f_b = 0.15$')

    ax.set_xlabel(r'$\log_{10}(M_{vir}/M_\odot)$', fontsize=14)
    ax.set_ylabel(r'$\log_{10}(f_{baryon})$', fontsize=14)
    ax.set_xlim(10, 14)
    ax.set_ylim(-2, 0)
    ax.legend(fontsize=12, loc='lower right')
    ax.set_title('Baryon Fraction vs Halo Mass (Centrals)', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'baryon_fraction_vs_mvir.png'), dpi=200)
    plt.close()
    print(f"Saved: baryon_fraction_vs_mvir.png")


def print_statistics_table(mill_data, uchuu_data):
    """Print comparison statistics for key mass bins"""
    print("\n" + "="*80)
    print("COMPARISON STATISTICS AT FIXED STELLAR MASS (Centrals Only)")
    print("="*80)

    mass_bins = [(10.5, 11.0), (11.0, 11.5), (11.5, 12.0)]

    for m_low, m_high in mass_bins:
        print(f"\n--- Stellar Mass: {m_low} < log(M*/Msun) < {m_high} ---")

        for name, data in [('Millennium', mill_data), ('miniUchuu', uchuu_data)]:
            if len(data['StellarMass']) == 0:
                print(f"  {name}: No data")
                continue

            centrals = data['Type'] == 0
            log_mstar = np.log10(data['StellarMass'][centrals])
            mask = (log_mstar >= m_low) & (log_mstar < m_high)

            n_gal = np.sum(mask)
            if n_gal < 10:
                print(f"  {name}: Only {n_gal} galaxies (insufficient)")
                continue

            sfr = (data['SfrDisk'][centrals] + data['SfrBulge'][centrals])[mask]
            mstar = data['StellarMass'][centrals][mask]
            log_ssfr = np.log10(sfr / mstar + 1e-15)
            q_frac = np.sum(log_ssfr < -11) / n_gal

            bh_mass = data['BlackHoleMass'][centrals][mask]
            hot_gas = data['HotGas'][centrals][mask]
            cold_gas = data['ColdGas'][centrals][mask]
            cgm_gas = data['CGMgas'][centrals][mask] if 'CGMgas' in data and len(data['CGMgas']) > 0 else np.zeros_like(hot_gas)
            mvir = data['Mvir'][centrals][mask]

            # Total baryons
            total_baryons = mstar + cold_gas + hot_gas + cgm_gas + bh_mass
            baryon_frac = total_baryons / mvir

            print(f"\n  {name} (N={n_gal}):")
            print(f"    Quiescent fraction:     {q_frac:.3f}")
            print(f"    Median log(M_BH):       {np.median(np.log10(bh_mass[bh_mass>0])):.2f}")
            print(f"    Median log(M_HotGas):   {np.median(np.log10(hot_gas[hot_gas>0])):.2f}")
            print(f"    Median log(M_ColdGas):  {np.median(np.log10(cold_gas[cold_gas>0])):.2f}")
            print(f"    Median log(M_BH/M*):    {np.median(np.log10(bh_mass[bh_mass>0]/mstar[bh_mass>0])):.2f}")
            print(f"    Median M_Hot/M_vir:     {np.median(hot_gas[hot_gas>0]/mvir[hot_gas>0]):.4f}")
            print(f"    Median M_Cold/M*:       {np.median(cold_gas/mstar):.4f}")
            print(f"    Median f_baryon:        {np.median(baryon_frac):.4f}")


# ========================== MAIN ==========================

def main():
    """Main function to run all diagnostics"""

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("="*60)
    print("RESOLUTION DIAGNOSTICS: Millennium vs miniUchuu")
    print("="*60)

    # Print simulation parameters
    print("\nSimulation Parameters:")
    print(f"  Millennium:  PartMass = {MILLENNIUM_CONFIG['PartMass']:.4f} (10^10 Msun/h)")
    print(f"               Min halo mass = {MILLENNIUM_CONFIG['PartMass'] * 20:.3f} (10^10 Msun/h)")
    print(f"  miniUchuu:   PartMass = {MINIUCHUU_CONFIG['PartMass']:.4f} (10^10 Msun/h)")
    print(f"               Min halo mass = {MINIUCHUU_CONFIG['PartMass'] * 20:.3f} (10^10 Msun/h)")
    print(f"  Resolution ratio: {MILLENNIUM_CONFIG['PartMass'] / MINIUCHUU_CONFIG['PartMass']:.2f}x")

    # Load data
    mill_data = load_simulation_data(MILLENNIUM_CONFIG)
    uchuu_data = load_simulation_data(MINIUCHUU_CONFIG)

    # Generate all diagnostic plots
    print("\nGenerating diagnostic plots...")

    plot_quiescent_fraction(mill_data, uchuu_data, OUTPUT_DIR)
    plot_bh_mass_vs_stellar_mass(mill_data, uchuu_data, OUTPUT_DIR)
    plot_hot_gas_vs_stellar_mass(mill_data, uchuu_data, OUTPUT_DIR)
    plot_cgm_gas_vs_stellar_mass(mill_data, uchuu_data, OUTPUT_DIR)
    plot_heating_vs_stellar_mass(mill_data, uchuu_data, OUTPUT_DIR)
    plot_ssfr_distribution_high_mass(mill_data, uchuu_data, OUTPUT_DIR)
    plot_bh_to_stellar_ratio(mill_data, uchuu_data, OUTPUT_DIR)
    plot_hot_gas_to_mvir_ratio(mill_data, uchuu_data, OUTPUT_DIR)
    plot_cold_gas_vs_stellar_mass(mill_data, uchuu_data, OUTPUT_DIR)
    plot_baryonic_tully_fisher(mill_data, uchuu_data, OUTPUT_DIR)
    plot_baryon_fraction_vs_mvir(mill_data, uchuu_data, OUTPUT_DIR)

    # Print statistics
    print_statistics_table(mill_data, uchuu_data)

    print(f"\n{'='*60}")
    print(f"Diagnostic plots saved to: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

def plot_radius_vs_stellar_mass(target_z=0.0):
    """
    Plot disk radius vs. stellar mass for different H2 prescriptions and area options
    """
    mass_bins = np.arange(8.0, 12.5, 0.2)
    z_diff = [abs(z - target_z) for z in DEFAULT_REDSHIFTS]
    best_snap = z_diff.index(min(z_diff))
    actual_z = DEFAULT_REDSHIFTS[best_snap]
    snap_str = f'Snap_{best_snap}'
    print(f"\n{'='*60}")
    print(f"Creating disk radius vs. stellar mass plot for z ~ {target_z}")
    print(f"Using {snap_str} (z={actual_z:.2f})")
    print(f"{'='*60}")
    fig, ax = plt.subplots(figsize=(10, 7))
    for prescription in H2_PRESCRIPTIONS:
        color = H2_PRESCRIPTION_COLORS[prescription]
        for area_option in AREA_OPTIONS:
            directory = f'./output/millennium_{prescription.lower()}_{area_option}/'
            linestyle = AREA_OPTION_LINESTYLES[area_option]
            if not os.path.exists(directory):
                print(f"  Warning: Directory {directory} not found, skipping")
                continue
            print(f"  Processing {prescription} with {area_option}...")
            try:
                stellar_mass = read_hdf(directory, snap_num=snap_str, param='StellarMass')
                disk_radius = read_hdf(directory, snap_num=snap_str, param='DiskRadius') * 1.0e3  # Convert to kpc
                galaxy_type = read_hdf(directory, snap_num=snap_str, param='Type')
                if stellar_mass is None or disk_radius is None or galaxy_type is None:
                    continue
                stellar_mass = stellar_mass * 1.0e10 / MILLENNIUM_HUBBLE_H
                # Disk radius in kpc
                mask = (stellar_mass > 0) & (disk_radius > 0) & (galaxy_type == 0)
                stellar_mass_filtered = stellar_mass[mask]
                disk_radius_filtered = disk_radius[mask]
                if len(stellar_mass_filtered) == 0:
                    continue
                log_stellar_mass = np.log10(stellar_mass_filtered)
                # Bin by stellar mass
                bin_centers = 0.5 * (mass_bins[:-1] + mass_bins[1:])
                bin_medians = []
                valid_bins = []
                for i in range(len(mass_bins)-1):
                    bin_mask = (log_stellar_mass >= mass_bins[i]) & (log_stellar_mass < mass_bins[i+1])
                    if np.sum(bin_mask) >= 5:
                        bin_medians.append(np.median(disk_radius_filtered[bin_mask]))
                        valid_bins.append(bin_centers[i])
                if len(valid_bins) > 0:
                    label = f'{prescription} {AREA_OPTION_LABELS[area_option]}'
                    ax.plot(valid_bins, bin_medians, color=color, linestyle=linestyle,
                            linewidth=2, label=label, alpha=0.8)
            except Exception as e:
                print(f"    Error: {e}")
                continue
    ax.set_xlabel(r'$\log_{10}(M_\star / \mathrm{M}_\odot)$', fontsize=14)
    ax.set_ylabel(r'Disk Radius (kpc)', fontsize=14)
    ax.set_xlim(8.0, 12.0)
    # ax.set_yscale('log')
    ax.set_ylim(bottom=0)
    ax.legend(loc='best', fontsize=9, framealpha=0.9, ncol=2)
    os.makedirs(OutputDir, exist_ok=True)
    output_path = os.path.join(OutputDir, f'disk_radius_vs_stellar_mass_z{target_z:.1f}.pdf')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n  Saved: {output_path}")
    plt.close()
def plot_sfr_surface_density_vs_cold_gas_surface_density(target_z=0.0):
    """
    Plot SFR surface density vs. cold gas (HI+H2) surface density for different H2 prescriptions and area options
    """
    # Surface density bins (log10)
    sigma_bins = np.arange(0.0, 10.2, 0.2)
    z_diff = [abs(z - target_z) for z in DEFAULT_REDSHIFTS]
    best_snap = z_diff.index(min(z_diff))
    actual_z = DEFAULT_REDSHIFTS[best_snap]
    snap_str = f'Snap_{best_snap}'
    print(f"\n{'='*60}")
    print(f"Creating SFR surface density vs. cold gas surface density plot for z ~ {target_z}")
    print(f"Using {snap_str} (z={actual_z:.2f})")
    print(f"{'='*60}")
    fig, ax = plt.subplots(figsize=(10, 7))
    for prescription in H2_PRESCRIPTIONS:
        color = H2_PRESCRIPTION_COLORS[prescription]
        for area_option in AREA_OPTIONS:
            directory = f'./output/millennium_{prescription.lower()}_{area_option}/'
            linestyle = AREA_OPTION_LINESTYLES[area_option]
            if not os.path.exists(directory):
                print(f"  Warning: Directory {directory} not found, skipping")
                continue
            print(f"  Processing {prescription} with {area_option}...")
            try:
                sfr_disk = read_hdf(directory, snap_num=snap_str, param='SfrDisk')
                sfr_bulge = read_hdf(directory, snap_num=snap_str, param='SfrBulge')
                cold_gas = read_hdf(directory, snap_num=snap_str, param='ColdGas')
                galaxy_type = read_hdf(directory, snap_num=snap_str, param='Type')
                disk_radius = read_hdf(directory, snap_num=snap_str, param='DiskRadius')
                if sfr_disk is None or sfr_bulge is None or cold_gas is None or disk_radius is None:
                    print(f"    Debug: Missing data for {prescription} {area_option}")
                    continue
                sfr = sfr_disk + sfr_bulge  # SFR in M_sun/yr
                cold_gas = cold_gas * 1.0e10 / MILLENNIUM_HUBBLE_H
                disk_radius_pc = disk_radius * 1.0e6
                disk_area_pc2 = np.pi * disk_radius_pc**2
                sigma_cold_gas = cold_gas / disk_area_pc2  # M_sun/pc^2
                sigma_sfr = sfr / disk_area_pc2  # M_sun/yr/pc^2
                mask = (cold_gas > 0) & (disk_area_pc2 > 0) & (galaxy_type == 0)
                sigma_cold_gas = sigma_cold_gas[mask]
                sigma_sfr = sigma_sfr[mask]
                # Only keep galaxies with positive SFR and cold gas
                valid = (sigma_cold_gas > 0) & (sigma_sfr > 0)
                sigma_cold_gas = sigma_cold_gas[valid]
                sigma_sfr = sigma_sfr[valid]
                log_sigma_cold_gas = np.log10(sigma_cold_gas)
                log_sigma_sfr = np.log10(sigma_sfr)
                # Bin by cold gas surface density
                bin_centers = 0.5 * (sigma_bins[:-1] + sigma_bins[1:])
                bin_medians = []
                valid_bins = []
                for i in range(len(sigma_bins)-1):
                    bin_mask = (log_sigma_cold_gas >= sigma_bins[i]) & (log_sigma_cold_gas < sigma_bins[i+1])
                    if np.sum(bin_mask) >= 5:
                        bin_medians.append(np.median(log_sigma_sfr[bin_mask]))
                        valid_bins.append(bin_centers[i])
                if len(valid_bins) > 0:
                    label = f'{prescription} {AREA_OPTION_LABELS[area_option]}'
                    ax.plot(10**np.array(valid_bins), 10**np.array(bin_medians), color=color, linestyle=linestyle,
                            linewidth=2, label=label, alpha=0.8)
            except Exception as e:
                print(f"    Error: {e}")
                continue
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\Sigma_{\mathrm{HI+H_2}}$ [M$_\odot$/pc$^2$]', fontsize=14)
    ax.set_ylabel(r'$\Sigma_{\mathrm{SFR}}$ [M$_\odot$/yr/pc$^2$]', fontsize=14)
    # ax.set_xticks([1, 10, 100, 1000])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    # ax.set_xlim(1, 1000)
    ax.legend(loc='best', fontsize=9, framealpha=0.9, ncol=2)
    os.makedirs(OutputDir, exist_ok=True)
    output_path = os.path.join(OutputDir, f'sfr_surface_density_vs_cold_gas_surface_density_z{target_z:.1f}.pdf')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n  Saved: {output_path}")
    plt.close()
#!/usr/bin/env python3
"""
SAGE H2 Area Comparison Plots
==============================

This script creates comparison plots for different H2 disk area normalizations.
Each H2 prescription is shown with 3 different linestyles for the 3 area options.

H2DiskAreaOption values:
  0 = π*r_s²
  1 = π*(3*r_s)² (default)
  2 = 2π*r_s² (central Σ₀)

Author: Analysis script for SAGE H2 area tests
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import os
from scipy import stats

# Configuration parameters
DEFAULT_REDSHIFTS = [127.000, 79.998, 50.000, 30.000, 19.916, 18.244, 16.725, 15.343, 14.086, 12.941, 11.897, 10.944, 10.073, 
                     9.278, 8.550, 7.883, 7.272, 6.712, 6.197, 5.724, 5.289, 4.888, 4.520, 4.179, 3.866, 3.576, 3.308, 3.060, 
                     2.831, 2.619, 2.422, 2.239, 2.070, 1.913, 1.766, 1.630, 1.504, 1.386, 1.276, 1.173, 1.078, 0.989, 0.905, 
                     0.828, 0.755, 0.687, 0.624, 0.564, 0.509, 0.457, 0.408, 0.362, 0.320, 0.280, 0.242, 0.208, 0.175, 0.144, 
                     0.116, 0.089, 0.064, 0.041, 0.020, 0.000]

# Millennium simulation parameters
MILLENNIUM_BOXSIZE = 62.5
MILLENNIUM_HUBBLE_H = 0.73

# Define H2 prescription configurations
H2_PRESCRIPTIONS = ['BR06', 'KD12', 'KMT09', 'K13', 'GD14']
H2_PRESCRIPTION_COLORS = {
    'BR06': 'orange',
    'KD12': 'red',
    'KMT09': 'blue',
    'K13': 'green',
    'GD14': 'purple'
}

# Define area options
AREA_OPTIONS = ['pi_rs2', '9pi_rs2', '2pi_rs2']
AREA_OPTION_LABELS = {
    'pi_rs2': r'$\pi r_s^2$',
    '9pi_rs2': r'$\pi (3r_s)^2$',
    '2pi_rs2': r'$2\pi r_s^2$'
}
AREA_OPTION_LINESTYLES = {
    'pi_rs2': '--',
    '9pi_rs2': '-',
    '2pi_rs2': ':'
}

OutputDir = './output/millennium/plots/'
ObsDataDir = '/Users/mbradley/Documents/PhD/shark/data/Gas/'
plt.style.use('/Users/mbradley/Documents/cohare_palatino_sty.mplstyle')


def read_hdf(directory, snap_num=None, param=None):
    """Read data from one or more SAGE model files"""
    model_files = [f for f in os.listdir(directory) if f.startswith('model_') and f.endswith('.hdf5')]
    model_files.sort()
    
    combined_data = None
    
    for model_file in model_files:
        try:
            with h5.File(os.path.join(directory, model_file), 'r') as property_file:
                data = np.array(property_file[snap_num][param])
                
                if combined_data is None:
                    combined_data = data
                else:
                    combined_data = np.concatenate((combined_data, data))
        except Exception as e:
            print(f"Warning: Could not read {param} from {model_file} for {snap_num}: {e}")
            continue
            
    return combined_data


def bootstrap_median_error(data, n_bootstrap=1000):
    """Calculate bootstrap errors on the median"""
    if len(data) == 0:
        return np.nan, np.nan
    
    bootstrap_medians = np.zeros(n_bootstrap)
    n = len(data)
    
    for i in range(n_bootstrap):
        resample = np.random.choice(data, size=n, replace=True)
        bootstrap_medians[i] = np.median(resample)
    
    lower_error = np.percentile(bootstrap_medians, 16)
    upper_error = np.percentile(bootstrap_medians, 84)
    
    return lower_error, upper_error


def read_obs_data(filename):
    """Read observational data files"""
    filepath = os.path.join(ObsDataDir, filename)
    if not os.path.exists(filepath):
        print(f"  Warning: Observational data file {filename} not found")
        return None
    
    data = np.loadtxt(filepath)
    return data


def plot_hi_mass_fraction_comparison(target_z=0.0):
    """
    Plot HI/M_star comparison for different H2 prescriptions and area options
    """
    mass_bins = np.arange(8.0, 12.5, 0.2)
    
    # Find snapshot closest to target redshift
    z_diff = [abs(z - target_z) for z in DEFAULT_REDSHIFTS]
    best_snap = z_diff.index(min(z_diff))
    actual_z = DEFAULT_REDSHIFTS[best_snap]
    snap_str = f'Snap_{best_snap}'
    
    print(f"\n{'='*60}")
    print(f"Creating HI mass fraction comparison plot for z ~ {target_z}")
    print(f"Using {snap_str} (z={actual_z:.2f})")
    print(f"{'='*60}")
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Track plotted models for legend
    plotted_configs = []
    
    for prescription in H2_PRESCRIPTIONS:
        color = H2_PRESCRIPTION_COLORS[prescription]
        
        for area_option in AREA_OPTIONS:
            directory = f'./output/millennium_{prescription.lower()}_{area_option}/'
            linestyle = AREA_OPTION_LINESTYLES[area_option]
            
            if not os.path.exists(directory):
                print(f"  Warning: Directory {directory} not found, skipping")
                continue
            
            print(f"  Processing {prescription} with {area_option}...")
            
            try:
                stellar_mass = read_hdf(directory, snap_num=snap_str, param='StellarMass')
                h2_gas = read_hdf(directory, snap_num=snap_str, param='H2gas')
                cold_gas = read_hdf(directory, snap_num=snap_str, param='ColdGas')
                galaxy_type = read_hdf(directory, snap_num=snap_str, param='Type')
                
                if stellar_mass is None or h2_gas is None or cold_gas is None:
                    continue
                
                # Convert to solar masses
                stellar_mass = stellar_mass * 1.0e10 / MILLENNIUM_HUBBLE_H
                h2_gas = h2_gas * 1.0e10 / MILLENNIUM_HUBBLE_H
                cold_gas = cold_gas * 1.0e10 / MILLENNIUM_HUBBLE_H
                
                # Calculate HI
                hi_gas = cold_gas - h2_gas
                hi_gas[hi_gas < 0] = 0
                
                # Filter for central galaxies with HI
                mask = (stellar_mass > 0) & (hi_gas > 0) & (galaxy_type == 0)
                stellar_mass_filtered = stellar_mass[mask]
                hi_gas_filtered = hi_gas[mask]
                
                if len(stellar_mass_filtered) == 0:
                    continue
                
                log_stellar_mass = np.log10(stellar_mass_filtered)
                log_hi_fraction = np.log10(hi_gas_filtered / stellar_mass_filtered)
                
                # Calculate median in bins
                bin_centers = 0.5 * (mass_bins[:-1] + mass_bins[1:])
                bin_medians = []
                valid_bins = []
                
                for i in range(len(mass_bins)-1):
                    bin_mask = (log_stellar_mass >= mass_bins[i]) & (log_stellar_mass < mass_bins[i+1])
                    if np.sum(bin_mask) > 10:
                        bin_medians.append(np.median(log_hi_fraction[bin_mask]))
                        valid_bins.append(bin_centers[i])
                
                if len(valid_bins) > 0:
                    label = f'{prescription} {AREA_OPTION_LABELS[area_option]}'
                    ax.plot(valid_bins, bin_medians, color=color, linestyle=linestyle, 
                           linewidth=2, label=label, alpha=0.8)
                    plotted_configs.append((prescription, area_option))
                    
            except Exception as e:
                print(f"    Error: {e}")
                continue
    
    # Add observational data
    try:
        data = read_obs_data('RHI-Mstars_Brown15.dat')
        if data is not None:
            log_mstar = data[:, 0]
            hi_frac = np.log10(data[:, 1])
            err = data[:, 2] * data[:, 1] / np.log(10)
            ax.errorbar(log_mstar, hi_frac, yerr=err, fmt='s', color='black', 
                       markersize=6, capsize=3, label='Brown+15', zorder=10,
                       markerfacecolor='none', markeredgewidth=1.5)
        
        data = read_obs_data('HIGasRatio_NonDetEQZero.dat')
        if data is not None:
            log_mstar = data[:, 0]
            median = data[:, 1]
            p16 = data[:, 2]
            p84 = data[:, 3]
            mask = (median > -10) & (median < 2) & (p16 > -10) & (p84 > -10)
            yerr_lower = np.abs(median[mask] - p16[mask])
            yerr_upper = np.abs(p84[mask] - median[mask])
            ax.errorbar(log_mstar[mask], median[mask], yerr=[yerr_lower, yerr_upper], 
                       fmt='o', color='black', markersize=6, capsize=3, 
                       label='xGASS', zorder=10, markerfacecolor='none', markeredgewidth=1.5)
    except Exception as e:
        print(f"  Warning: Could not add observational data: {e}")
    
    ax.set_xlabel(r'$\log_{10}(M_\star / \mathrm{M}_\odot)$', fontsize=14)
    ax.set_ylabel(r'$\log_{10}(M_{\mathrm{HI}} / M_\star)$', fontsize=14)
    ax.set_xlim(8.0, 12.0)
    ax.set_ylim(-3, 2)
    ax.legend(loc='best', fontsize=9, framealpha=0.9, ncol=2)
    
    os.makedirs(OutputDir, exist_ok=True)
    output_path = os.path.join(OutputDir, f'hi_mass_fraction_area_comparison_z{target_z:.1f}.pdf')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n  Saved: {output_path}")
    plt.close()


def plot_h2_mass_fraction_comparison(target_z=0.0):
    """
    Plot H2/M_star comparison for different H2 prescriptions and area options
    """
    mass_bins = np.arange(8.0, 12.5, 0.2)
    
    # Find snapshot closest to target redshift
    z_diff = [abs(z - target_z) for z in DEFAULT_REDSHIFTS]
    best_snap = z_diff.index(min(z_diff))
    actual_z = DEFAULT_REDSHIFTS[best_snap]
    snap_str = f'Snap_{best_snap}'
    
    print(f"\n{'='*60}")
    print(f"Creating H2 mass fraction comparison plot for z ~ {target_z}")
    print(f"Using {snap_str} (z={actual_z:.2f})")
    print(f"{'='*60}")
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for prescription in H2_PRESCRIPTIONS:
        color = H2_PRESCRIPTION_COLORS[prescription]
        
        for area_option in AREA_OPTIONS:
            directory = f'./output/millennium_{prescription.lower()}_{area_option}/'
            linestyle = AREA_OPTION_LINESTYLES[area_option]
            
            if not os.path.exists(directory):
                print(f"  Warning: Directory {directory} not found, skipping")
                continue
            
            print(f"  Processing {prescription} with {area_option}...")
            
            try:
                stellar_mass = read_hdf(directory, snap_num=snap_str, param='StellarMass')
                h2_gas = read_hdf(directory, snap_num=snap_str, param='H2gas')
                galaxy_type = read_hdf(directory, snap_num=snap_str, param='Type')
                
                if stellar_mass is None or h2_gas is None:
                    continue
                
                # Convert to solar masses
                stellar_mass = stellar_mass * 1.0e10 / MILLENNIUM_HUBBLE_H
                h2_gas = h2_gas * 1.0e10 / MILLENNIUM_HUBBLE_H
                
                # Filter for central galaxies with H2
                mask = (stellar_mass > 0) & (h2_gas > 0) & (galaxy_type == 0)
                stellar_mass_filtered = stellar_mass[mask]
                h2_gas_filtered = h2_gas[mask]
                
                if len(stellar_mass_filtered) == 0:
                    continue
                
                log_stellar_mass = np.log10(stellar_mass_filtered)
                log_h2_fraction = np.log10(h2_gas_filtered / stellar_mass_filtered)
                
                # Calculate median in bins
                bin_centers = 0.5 * (mass_bins[:-1] + mass_bins[1:])
                bin_medians = []
                valid_bins = []
                
                for i in range(len(mass_bins)-1):
                    bin_mask = (log_stellar_mass >= mass_bins[i]) & (log_stellar_mass < mass_bins[i+1])
                    if np.sum(bin_mask) > 10:
                        bin_medians.append(np.median(log_h2_fraction[bin_mask]))
                        valid_bins.append(bin_centers[i])
                
                if len(valid_bins) > 0:
                    label = f'{prescription} {AREA_OPTION_LABELS[area_option]}'
                    ax.plot(valid_bins, bin_medians, color=color, linestyle=linestyle, 
                           linewidth=2, label=label, alpha=0.8)
                    
            except Exception as e:
                print(f"    Error: {e}")
                continue
    
    # Add observational data
    try:
        data = read_obs_data('MolecularGasRatio_NonDetEQZero.dat')
        if data is not None:
            log_mstar = data[:, 0]
            median = data[:, 1]
            p16 = data[:, 2]
            p84 = data[:, 3]
            mask = (median > -10) & (median < 2) & (p16 > -10) & (p84 > -10)
            yerr_lower = np.abs(median[mask] - p16[mask])
            yerr_upper = np.abs(p84[mask] - median[mask])
            ax.errorbar(log_mstar[mask], median[mask], yerr=[yerr_lower, yerr_upper], 
                       fmt='o', color='black', markersize=6, capsize=3, 
                       label='xCOLDGASS', zorder=10, markerfacecolor='none', markeredgewidth=1.5)
    except Exception as e:
        print(f"  Warning: Could not add observational data: {e}")
    
    ax.set_xlabel(r'$\log_{10}(M_\star / \mathrm{M}_\odot)$', fontsize=14)
    ax.set_ylabel(r'$\log_{10}(M_{\mathrm{H}_2} / M_\star)$', fontsize=14)
    ax.set_xlim(8.0, 12.0)
    ax.set_ylim(-3, 2)
    ax.legend(loc='best', fontsize=9, framealpha=0.9, ncol=2)
    
    os.makedirs(OutputDir, exist_ok=True)
    output_path = os.path.join(OutputDir, f'h2_mass_fraction_area_comparison_z{target_z:.1f}.pdf')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n  Saved: {output_path}")
    plt.close()


def plot_cold_gas_mass_fraction_comparison(target_z=0.0):
    """
    Plot (HI+H2)/M_star comparison for different H2 prescriptions and area options
    """
    mass_bins = np.arange(8.0, 12.5, 0.2)
    
    # Find snapshot closest to target redshift
    z_diff = [abs(z - target_z) for z in DEFAULT_REDSHIFTS]
    best_snap = z_diff.index(min(z_diff))
    actual_z = DEFAULT_REDSHIFTS[best_snap]
    snap_str = f'Snap_{best_snap}'
    
    print(f"\n{'='*60}")
    print(f"Creating cold gas mass fraction comparison plot for z ~ {target_z}")
    print(f"Using {snap_str} (z={actual_z:.2f})")
    print(f"{'='*60}")
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for prescription in H2_PRESCRIPTIONS:
        color = H2_PRESCRIPTION_COLORS[prescription]
        
        for area_option in AREA_OPTIONS:
            directory = f'./output/millennium_{prescription.lower()}_{area_option}/'
            linestyle = AREA_OPTION_LINESTYLES[area_option]
            
            if not os.path.exists(directory):
                print(f"  Warning: Directory {directory} not found, skipping")
                continue
            
            print(f"  Processing {prescription} with {area_option}...")
            
            try:
                stellar_mass = read_hdf(directory, snap_num=snap_str, param='StellarMass')
                cold_gas = read_hdf(directory, snap_num=snap_str, param='ColdGas')
                galaxy_type = read_hdf(directory, snap_num=snap_str, param='Type')
                
                if stellar_mass is None or cold_gas is None:
                    continue
                
                # Convert to solar masses
                stellar_mass = stellar_mass * 1.0e10 / MILLENNIUM_HUBBLE_H
                cold_gas = cold_gas * 1.0e10 / MILLENNIUM_HUBBLE_H
                
                # Filter for central galaxies with cold gas
                mask = (stellar_mass > 0) & (cold_gas > 0) & (galaxy_type == 0)
                stellar_mass_filtered = stellar_mass[mask]
                cold_gas_filtered = cold_gas[mask]
                
                if len(stellar_mass_filtered) == 0:
                    continue
                
                log_stellar_mass = np.log10(stellar_mass_filtered)
                log_cold_gas_fraction = np.log10(cold_gas_filtered / stellar_mass_filtered)
                
                # Calculate median in bins
                bin_centers = 0.5 * (mass_bins[:-1] + mass_bins[1:])
                bin_medians = []
                valid_bins = []
                
                for i in range(len(mass_bins)-1):
                    bin_mask = (log_stellar_mass >= mass_bins[i]) & (log_stellar_mass < mass_bins[i+1])
                    if np.sum(bin_mask) > 10:
                        bin_medians.append(np.median(log_cold_gas_fraction[bin_mask]))
                        valid_bins.append(bin_centers[i])
                
                if len(valid_bins) > 0:
                    label = f'{prescription} {AREA_OPTION_LABELS[area_option]}'
                    ax.plot(valid_bins, bin_medians, color=color, linestyle=linestyle, 
                           linewidth=2, label=label, alpha=0.8)
                    
            except Exception as e:
                print(f"    Error: {e}")
                continue
    
    # Add observational data
    try:
        data = read_obs_data('NeutralGasRatio_NonDetEQZero.dat')
        if data is not None:
            log_mstar = data[:, 0]
            median = data[:, 1]
            p16 = data[:, 2]
            p84 = data[:, 3]
            mask = (median > -10) & (median < 2) & (p16 > -10) & (p84 > -10)
            yerr_lower = np.abs(median[mask] - p16[mask])
            yerr_upper = np.abs(p84[mask] - median[mask])
            ax.errorbar(log_mstar[mask], median[mask], yerr=[yerr_lower, yerr_upper], 
                       fmt='o', color='black', markersize=6, capsize=3, 
                       label='xGASS', zorder=10, markerfacecolor='none', markeredgewidth=1.5)
    except Exception as e:
        print(f"  Warning: Could not add observational data: {e}")
    
    ax.set_xlabel(r'$\log_{10}(M_\star / \mathrm{M}_\odot)$', fontsize=14)
    ax.set_ylabel(r'$\log_{10}(M_{\mathrm{HI+H_2}} / M_\star)$', fontsize=14)
    ax.set_xlim(8.0, 12.0)
    ax.set_ylim(-3, 2)
    ax.legend(loc='best', fontsize=9, framealpha=0.9, ncol=2)
    
    os.makedirs(OutputDir, exist_ok=True)
    output_path = os.path.join(OutputDir, f'cold_gas_mass_fraction_area_comparison_z{target_z:.1f}.pdf')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n  Saved: {output_path}")
    plt.close()


def plot_metallicity_comparison(target_z=0.0):
    """
    Plot gas-phase metallicity as a function of stellar mass for different H2 prescriptions and area options
    
    Metallicity is calculated as: 12 + log10(O/H) = 12 + log10(Z/Z_sun) + log10(Z_sun)
    where Z_sun = 0.0134 (Asplund et al. 2009) gives log10(Z_sun) = -1.873
    So: 12 + log10(O/H) ≈ 8.69 + log10(Z/Z_sun)
    """
    mass_bins = np.arange(8.0, 12.5, 0.2)
    
    # Find snapshot closest to target redshift
    z_diff = [abs(z - target_z) for z in DEFAULT_REDSHIFTS]
    best_snap = z_diff.index(min(z_diff))
    actual_z = DEFAULT_REDSHIFTS[best_snap]
    snap_str = f'Snap_{best_snap}'
    
    print(f"\n{'='*60}")
    print(f"Creating metallicity comparison plot for z ~ {target_z}")
    print(f"Using {snap_str} (z={actual_z:.2f})")
    print(f"{'='*60}")
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for prescription in H2_PRESCRIPTIONS:
        color = H2_PRESCRIPTION_COLORS[prescription]
        
        for area_option in AREA_OPTIONS:
            directory = f'./output/millennium_{prescription.lower()}_{area_option}/'
            linestyle = AREA_OPTION_LINESTYLES[area_option]
            
            if not os.path.exists(directory):
                print(f"  Warning: Directory {directory} not found, skipping")
                continue
            
            print(f"  Processing {prescription} with {area_option}...")
            
            try:
                stellar_mass = read_hdf(directory, snap_num=snap_str, param='StellarMass')
                cold_gas = read_hdf(directory, snap_num=snap_str, param='ColdGas')
                metal_cold_gas = read_hdf(directory, snap_num=snap_str, param='MetalsColdGas')
                galaxy_type = read_hdf(directory, snap_num=snap_str, param='Type')
                
                if stellar_mass is None or cold_gas is None or metal_cold_gas is None:
                    continue
                
                # Convert to solar masses
                stellar_mass = stellar_mass * 1.0e10 / MILLENNIUM_HUBBLE_H
                cold_gas = cold_gas * 1.0e10 / MILLENNIUM_HUBBLE_H
                metal_cold_gas = metal_cold_gas * 1.0e10 / MILLENNIUM_HUBBLE_H
                
                # Calculate gas-phase metallicity
                # Avoid division by zero
                mask = (stellar_mass > 0) & (cold_gas > 1e-6) & (galaxy_type == 0)
                stellar_mass_filtered = stellar_mass[mask]
                cold_gas_filtered = cold_gas[mask]
                metal_cold_gas_filtered = metal_cold_gas[mask]
                
                # Metallicity Z = M_metals / M_gas
                metallicity = metal_cold_gas_filtered / cold_gas_filtered
                
                # Convert to 12 + log10(O/H) using solar metallicity Z_sun = 0.0134
                # 12 + log10(O/H) = 8.69 + log10(Z/Z_sun)
                Z_sun = 0.0134
                metallicity_12OH = 8.69 + np.log10(metallicity / Z_sun)
                
                # Filter out unphysical values
                valid_mask = np.isfinite(metallicity_12OH) & (metallicity_12OH > 7.0) & (metallicity_12OH < 10.0)
                stellar_mass_filtered = stellar_mass_filtered[valid_mask]
                metallicity_12OH = metallicity_12OH[valid_mask]
                
                if len(stellar_mass_filtered) == 0:
                    continue
                
                log_stellar_mass = np.log10(stellar_mass_filtered)
                
                # Calculate median in bins
                bin_centers = 0.5 * (mass_bins[:-1] + mass_bins[1:])
                bin_medians = []
                valid_bins = []
                
                for i in range(len(mass_bins)-1):
                    bin_mask = (log_stellar_mass >= mass_bins[i]) & (log_stellar_mass < mass_bins[i+1])
                    if np.sum(bin_mask) > 10:
                        bin_medians.append(np.median(metallicity_12OH[bin_mask]))
                        valid_bins.append(bin_centers[i])
                
                if len(valid_bins) > 0:
                    label = f'{prescription} {AREA_OPTION_LABELS[area_option]}'
                    ax.plot(valid_bins, bin_medians, color=color, linestyle=linestyle, 
                           linewidth=2, label=label, alpha=0.8)
                    
            except Exception as e:
                print(f"    Error: {e}")
                continue
    
    ax.set_xlabel(r'$\log_{10}(M_\star / \mathrm{M}_\odot)$', fontsize=14)
    ax.set_ylabel(r'$12 + \log_{10}(\mathrm{O/H})$', fontsize=14)
    ax.set_xlim(8.0, 12.0)
    ax.set_ylim(7.5, 9.5)
    ax.legend(loc='best', fontsize=9, framealpha=0.9, ncol=2)
    
    os.makedirs(OutputDir, exist_ok=True)
    output_path = os.path.join(OutputDir, f'metallicity_area_comparison_z{target_z:.1f}.pdf')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n  Saved: {output_path}")
    plt.close()


def plot_quiescent_fraction_comparison(target_z=0.0):
    """
    Plot quiescent fraction as a function of stellar mass for different H2 prescriptions and area options
    
    Quiescent galaxies are defined as those with sSFR < 10^-11 yr^-1
    """
    mass_bins = np.arange(9.0, 12.0, 0.2)
    
    # Find snapshot closest to target redshift
    z_diff = [abs(z - target_z) for z in DEFAULT_REDSHIFTS]
    best_snap = z_diff.index(min(z_diff))
    actual_z = DEFAULT_REDSHIFTS[best_snap]
    snap_str = f'Snap_{best_snap}'
    
    print(f"\n{'='*60}")
    print(f"Creating quiescent fraction comparison plot for z ~ {target_z}")
    print(f"Using {snap_str} (z={actual_z:.2f})")
    print(f"{'='*60}")
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # sSFR threshold for quiescence (yr^-1)
    ssfr_threshold = 1e-11
    
    for prescription in H2_PRESCRIPTIONS:
        color = H2_PRESCRIPTION_COLORS[prescription]
        
        for area_option in AREA_OPTIONS:
            directory = f'./output/millennium_{prescription.lower()}_{area_option}/'
            linestyle = AREA_OPTION_LINESTYLES[area_option]
            
            if not os.path.exists(directory):
                print(f"  Warning: Directory {directory} not found, skipping")
                continue
            
            print(f"  Processing {prescription} with {area_option}...")
            
            try:
                stellar_mass = read_hdf(directory, snap_num=snap_str, param='StellarMass')
                sfr_disk = read_hdf(directory, snap_num=snap_str, param='SfrDisk')
                sfr_bulge = read_hdf(directory, snap_num=snap_str, param='SfrBulge')
                galaxy_type = read_hdf(directory, snap_num=snap_str, param='Type')
                
                if stellar_mass is None or sfr_disk is None or sfr_bulge is None:
                    continue
                
                # Convert to solar masses
                stellar_mass = stellar_mass * 1.0e10 / MILLENNIUM_HUBBLE_H
                # SFR is already in M_sun/yr, combine disk and bulge
                sfr = sfr_disk + sfr_bulge
                
                # Filter for central galaxies with stellar mass
                mask = (stellar_mass > 1e8) & (galaxy_type == 0)
                stellar_mass_filtered = stellar_mass[mask]
                sfr_filtered = sfr[mask]
                
                if len(stellar_mass_filtered) == 0:
                    continue
                
                # Calculate specific SFR (yr^-1)
                ssfr = sfr_filtered / stellar_mass_filtered
                
                log_stellar_mass = np.log10(stellar_mass_filtered)
                
                # Calculate quiescent fraction in bins
                bin_centers = 0.5 * (mass_bins[:-1] + mass_bins[1:])
                quiescent_fractions = []
                valid_bins = []
                
                for i in range(len(mass_bins)-1):
                    bin_mask = (log_stellar_mass >= mass_bins[i]) & (log_stellar_mass < mass_bins[i+1])
                    if np.sum(bin_mask) > 10:
                        n_total = np.sum(bin_mask)
                        n_quiescent = np.sum(ssfr[bin_mask] < ssfr_threshold)
                        quiescent_fraction = n_quiescent / n_total
                        quiescent_fractions.append(quiescent_fraction)
                        valid_bins.append(bin_centers[i])
                
                if len(valid_bins) > 0:
                    label = f'{prescription} {AREA_OPTION_LABELS[area_option]}'
                    ax.plot(valid_bins, quiescent_fractions, color=color, linestyle=linestyle, 
                           linewidth=2, label=label, alpha=0.8)
                    
            except Exception as e:
                print(f"    Error: {e}")
                continue
    
    ax.set_xlabel(r'$\log_{10}(M_\star / \mathrm{M}_\odot)$', fontsize=14)
    ax.set_ylabel(r'Quiescent Fraction (sSFR $< 10^{-11}$ yr$^{-1}$)', fontsize=14)
    ax.set_xlim(9.0, 12.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc='best', fontsize=9, framealpha=0.9, ncol=2)
    
    os.makedirs(OutputDir, exist_ok=True)
    output_path = os.path.join(OutputDir, f'quiescent_fraction_area_comparison_z{target_z:.1f}.pdf')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n  Saved: {output_path}")
    plt.close()


def calculate_model_statistics(target_z=0.0):
    """
    Calculate goodness-of-fit statistics comparing models to observations
    Returns dictionary with RMSE values for HI, H2, and cold gas
    """
    mass_bins = np.arange(8.0, 12.5, 0.2)
    
    # Find snapshot
    z_diff = [abs(z - target_z) for z in DEFAULT_REDSHIFTS]
    best_snap = z_diff.index(min(z_diff))
    snap_str = f'Snap_{best_snap}'
    
    print(f"\n{'='*60}")
    print(f"Calculating model statistics vs observations at z ~ {target_z}")
    print(f"{'='*60}")
    
    # Load observational data
    obs_data = {}
    
    # HI observations (xGASS)
    try:
        data = read_obs_data('HIGasRatio_NonDetEQZero.dat')
        if data is not None:
            mask = (data[:, 1] > -10) & (data[:, 1] < 2)
            # Use p16 and p84 to estimate uncertainty (average of upper and lower errors)
            p16 = data[mask, 2]
            p84 = data[mask, 3]
            uncertainty = (np.abs(data[mask, 1] - p16) + np.abs(p84 - data[mask, 1])) / 2.0
            obs_data['HI'] = {'mass': data[mask, 0], 'ratio': data[mask, 1], 'error': uncertainty}
    except:
        pass
    
    # H2 observations (xCOLDGASS)
    try:
        data = read_obs_data('MolecularGasRatio_NonDetEQZero.dat')
        if data is not None:
            mask = (data[:, 1] > -10) & (data[:, 1] < 2)
            p16 = data[mask, 2]
            p84 = data[mask, 3]
            uncertainty = (np.abs(data[mask, 1] - p16) + np.abs(p84 - data[mask, 1])) / 2.0
            obs_data['H2'] = {'mass': data[mask, 0], 'ratio': data[mask, 1], 'error': uncertainty}
    except:
        pass
    
    # Cold gas observations (xGASS neutral)
    try:
        data = read_obs_data('NeutralGasRatio_NonDetEQZero.dat')
        if data is not None:
            mask = (data[:, 1] > -10) & (data[:, 1] < 2)
            p16 = data[mask, 2]
            p84 = data[mask, 3]
            uncertainty = (np.abs(data[mask, 1] - p16) + np.abs(p84 - data[mask, 1])) / 2.0
            obs_data['ColdGas'] = {'mass': data[mask, 0], 'ratio': data[mask, 1], 'error': uncertainty}
    except:
        pass
    
    # Calculate statistics for each model
    statistics = {}
    
    for prescription in H2_PRESCRIPTIONS:
        for area_option in AREA_OPTIONS:
            directory = f'./output/millennium_{prescription.lower()}_{area_option}/'
            
            if not os.path.exists(directory):
                continue
            
            model_key = f'{prescription}_{area_option}'
            statistics[model_key] = {}
            
            try:
                stellar_mass = read_hdf(directory, snap_num=snap_str, param='StellarMass')
                h2_gas = read_hdf(directory, snap_num=snap_str, param='H2gas')
                cold_gas = read_hdf(directory, snap_num=snap_str, param='ColdGas')
                galaxy_type = read_hdf(directory, snap_num=snap_str, param='Type')
                
                if stellar_mass is None or h2_gas is None or cold_gas is None:
                    continue
                
                # Convert to solar masses
                stellar_mass = stellar_mass * 1.0e10 / MILLENNIUM_HUBBLE_H
                h2_gas = h2_gas * 1.0e10 / MILLENNIUM_HUBBLE_H
                cold_gas = cold_gas * 1.0e10 / MILLENNIUM_HUBBLE_H
                hi_gas = cold_gas - h2_gas
                hi_gas[hi_gas < 0] = 0
                
                # Calculate HI statistics
                if 'HI' in obs_data:
                    mask = (stellar_mass > 0) & (hi_gas > 0) & (galaxy_type == 0)
                    if np.sum(mask) > 0:
                        log_mass = np.log10(stellar_mass[mask])
                        log_hi_ratio = np.log10(hi_gas[mask] / stellar_mass[mask])
                        
                        # Interpolate model to obs mass points
                        model_at_obs = []
                        for obs_mass in obs_data['HI']['mass']:
                            mass_mask = (log_mass >= obs_mass - 0.1) & (log_mass <= obs_mass + 0.1)
                            if np.sum(mass_mask) > 10:
                                model_at_obs.append(np.median(log_hi_ratio[mass_mask]))
                            else:
                                model_at_obs.append(np.nan)
                        
                        model_at_obs = np.array(model_at_obs)
                        valid = np.isfinite(model_at_obs)
                        if np.sum(valid) > 0:
                            rmse = np.sqrt(np.mean((model_at_obs[valid] - obs_data['HI']['ratio'][valid])**2))
                            bias = np.mean(model_at_obs[valid] - obs_data['HI']['ratio'][valid])
                            # Calculate reduced chi-squared
                            chi2 = np.sum(((model_at_obs[valid] - obs_data['HI']['ratio'][valid]) / obs_data['HI']['error'][valid])**2)
                            dof = np.sum(valid) - 1  # degrees of freedom
                            chi2_red = chi2 / dof if dof > 0 else np.nan
                            statistics[model_key]['HI_RMSE'] = rmse
                            statistics[model_key]['HI_bias'] = bias
                            statistics[model_key]['HI_chi2red'] = chi2_red
                
                # Calculate H2 statistics
                if 'H2' in obs_data:
                    mask = (stellar_mass > 0) & (h2_gas > 0) & (galaxy_type == 0)
                    if np.sum(mask) > 0:
                        log_mass = np.log10(stellar_mass[mask])
                        log_h2_ratio = np.log10(h2_gas[mask] / stellar_mass[mask])
                        
                        model_at_obs = []
                        for obs_mass in obs_data['H2']['mass']:
                            mass_mask = (log_mass >= obs_mass - 0.1) & (log_mass <= obs_mass + 0.1)
                            if np.sum(mass_mask) > 10:
                                model_at_obs.append(np.median(log_h2_ratio[mass_mask]))
                            else:
                                model_at_obs.append(np.nan)
                        
                        model_at_obs = np.array(model_at_obs)
                        valid = np.isfinite(model_at_obs)
                        if np.sum(valid) > 0:
                            rmse = np.sqrt(np.mean((model_at_obs[valid] - obs_data['H2']['ratio'][valid])**2))
                            bias = np.mean(model_at_obs[valid] - obs_data['H2']['ratio'][valid])
                            chi2 = np.sum(((model_at_obs[valid] - obs_data['H2']['ratio'][valid]) / obs_data['H2']['error'][valid])**2)
                            dof = np.sum(valid) - 1
                            chi2_red = chi2 / dof if dof > 0 else np.nan
                            statistics[model_key]['H2_RMSE'] = rmse
                            statistics[model_key]['H2_bias'] = bias
                            statistics[model_key]['H2_chi2red'] = chi2_red
                
                # Calculate cold gas statistics
                if 'ColdGas' in obs_data:
                    mask = (stellar_mass > 0) & (cold_gas > 0) & (galaxy_type == 0)
                    if np.sum(mask) > 0:
                        log_mass = np.log10(stellar_mass[mask])
                        log_cold_ratio = np.log10(cold_gas[mask] / stellar_mass[mask])
                        
                        model_at_obs = []
                        for obs_mass in obs_data['ColdGas']['mass']:
                            mass_mask = (log_mass >= obs_mass - 0.1) & (log_mass <= obs_mass + 0.1)
                            if np.sum(mass_mask) > 10:
                                model_at_obs.append(np.median(log_cold_ratio[mass_mask]))
                            else:
                                model_at_obs.append(np.nan)
                        
                        model_at_obs = np.array(model_at_obs)
                        valid = np.isfinite(model_at_obs)
                        if np.sum(valid) > 0:
                            rmse = np.sqrt(np.mean((model_at_obs[valid] - obs_data['ColdGas']['ratio'][valid])**2))
                            bias = np.mean(model_at_obs[valid] - obs_data['ColdGas']['ratio'][valid])
                            chi2 = np.sum(((model_at_obs[valid] - obs_data['ColdGas']['ratio'][valid]) / obs_data['ColdGas']['error'][valid])**2)
                            dof = np.sum(valid) - 1
                            chi2_red = chi2 / dof if dof > 0 else np.nan
                            statistics[model_key]['ColdGas_RMSE'] = rmse
                            statistics[model_key]['ColdGas_bias'] = bias
                            statistics[model_key]['ColdGas_chi2red'] = chi2_red
                
            except Exception as e:
                print(f"  Error processing {model_key}: {e}")
                continue
    
    return statistics


def plot_model_statistics(statistics):
    """
    Create a summary plot showing RMSE for each model configuration
    """
    if len(statistics) == 0:
        print("No statistics to plot")
        return
    
    print(f"\n{'='*60}")
    print("Creating model statistics comparison plot")
    print(f"{'='*60}")
    
    # Print statistics table
    print("\nModel Statistics (RMSE in dex):")
    print(f"{'Model':<20} {'HI RMSE':>10} {'H2 RMSE':>10} {'ColdGas RMSE':>10} {'Total RMSE':>12}")
    print("-" * 65)
    
    # Calculate total RMSE and chi-squared for all models to find the best
    total_rmse_dict = {}
    total_chi2_dict = {}
    for model_key in statistics.keys():
        stats = statistics[model_key]
        hi_rmse = stats.get('HI_RMSE', np.nan)
        h2_rmse = stats.get('H2_RMSE', np.nan)
        cold_rmse = stats.get('ColdGas_RMSE', np.nan)
        valid_rmse = [r for r in [hi_rmse, h2_rmse, cold_rmse] if np.isfinite(r)]
        total_rmse = np.sqrt(np.mean(np.array(valid_rmse)**2)) if len(valid_rmse) > 0 else np.nan
        total_rmse_dict[model_key] = total_rmse
        
        # Calculate average chi-squared
        hi_chi2 = stats.get('HI_chi2red', np.nan)
        h2_chi2 = stats.get('H2_chi2red', np.nan)
        cold_chi2 = stats.get('ColdGas_chi2red', np.nan)
        valid_chi2 = [c for c in [hi_chi2, h2_chi2, cold_chi2] if np.isfinite(c)]
        total_chi2 = np.mean(valid_chi2) if len(valid_chi2) > 0 else np.nan
        total_chi2_dict[model_key] = total_chi2
    
    # Find best models for RMSE
    best_total_model = min(total_rmse_dict.items(), key=lambda x: x[1] if np.isfinite(x[1]) else np.inf)[0]
    best_hi_model = None
    best_h2_model = None
    best_cold_model = None
    min_hi = np.inf
    min_h2 = np.inf
    min_cold = np.inf
    
    for model_key, stats in statistics.items():
        if np.isfinite(stats.get('HI_RMSE', np.nan)) and stats['HI_RMSE'] < min_hi:
            min_hi = stats['HI_RMSE']
            best_hi_model = model_key
        if np.isfinite(stats.get('H2_RMSE', np.nan)) and stats['H2_RMSE'] < min_h2:
            min_h2 = stats['H2_RMSE']
            best_h2_model = model_key
        if np.isfinite(stats.get('ColdGas_RMSE', np.nan)) and stats['ColdGas_RMSE'] < min_cold:
            min_cold = stats['ColdGas_RMSE']
            best_cold_model = model_key
    
    # Find best models for chi-squared (closest to 1)
    best_chi2_model = min(total_chi2_dict.items(), key=lambda x: abs(x[1] - 1.0) if np.isfinite(x[1]) else np.inf)[0]
    
    for model_key in sorted(statistics.keys()):
        stats = statistics[model_key]
        hi_rmse = stats.get('HI_RMSE', np.nan)
        h2_rmse = stats.get('H2_RMSE', np.nan)
        cold_rmse = stats.get('ColdGas_RMSE', np.nan)
        total_rmse = total_rmse_dict[model_key]
        
        # Highlight best model with asterisk
        marker = " ★" if model_key == best_total_model else "  "
        
        print(f"{model_key:<20} {hi_rmse:>10.3f} {h2_rmse:>10.3f} {cold_rmse:>10.3f} {total_rmse:>12.3f}{marker}")
    
    print("\n" + "="*65)
    print(f"★ Best overall (lowest total RMSE): {best_total_model} (RMSE = {total_rmse_dict[best_total_model]:.3f})")
    print(f"  Best for HI:       {best_hi_model} (RMSE = {min_hi:.3f})")
    print(f"  Best for H2:       {best_h2_model} (RMSE = {min_h2:.3f})")
    print(f"  Best for Cold Gas: {best_cold_model} (RMSE = {min_cold:.3f})")
    print("="*65)
    
    # Print reduced chi-squared table
    print("\nModel Statistics (Reduced χ² - closer to 1.0 is better):")
    print(f"{'Model':<20} {'HI χ²':>10} {'H2 χ²':>10} {'ColdGas χ²':>10} {'Avg χ²':>12}")
    print("-" * 65)
    
    for model_key in sorted(statistics.keys()):
        stats = statistics[model_key]
        hi_chi2 = stats.get('HI_chi2red', np.nan)
        h2_chi2 = stats.get('H2_chi2red', np.nan)
        cold_chi2 = stats.get('ColdGas_chi2red', np.nan)
        total_chi2 = total_chi2_dict[model_key]
        
        # Highlight best chi-squared model
        marker = " ★" if model_key == best_chi2_model else "  "
        
        print(f"{model_key:<20} {hi_chi2:>10.3f} {h2_chi2:>10.3f} {cold_chi2:>10.3f} {total_chi2:>12.3f}{marker}")
    
    print("\n" + "="*65)
    print(f"★ Best overall (χ² closest to 1): {best_chi2_model} (χ² = {total_chi2_dict[best_chi2_model]:.3f})")
    print("="*65)
    
    # Create plot with 2 rows: RMSE on top, chi-squared on bottom
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    gas_types = ['HI', 'H2', 'ColdGas']
    titles_rmse = [r'HI/M$_\star$ RMSE', r'H$_2$/M$_\star$ RMSE', r'Cold Gas/M$_\star$ RMSE']
    titles_chi2 = [r'HI/M$_\star$ Reduced χ²', r'H$_2$/M$_\star$ Reduced χ²', r'Cold Gas/M$_\star$ Reduced χ²']
    
    # Plot RMSE (top row)
    for idx, (gas_type, title) in enumerate(zip(gas_types, titles_rmse)):
        ax = axes[0, idx]
        
        # Organize data by prescription and area
        for prescription in H2_PRESCRIPTIONS:
            area_rmse = []
            area_labels = []
            
            for area_option in AREA_OPTIONS:
                model_key = f'{prescription}_{area_option}'
                if model_key in statistics:
                    rmse = statistics[model_key].get(f'{gas_type}_RMSE', np.nan)
                    if np.isfinite(rmse):
                        area_rmse.append(rmse)
                        area_labels.append(AREA_OPTION_LABELS[area_option])
            
            if len(area_rmse) > 0:
                x_pos = np.arange(len(area_rmse)) + H2_PRESCRIPTIONS.index(prescription) * 0.15
                ax.bar(x_pos, area_rmse, width=0.15, 
                      label=prescription, color=H2_PRESCRIPTION_COLORS[prescription], alpha=0.8)
        
        ax.set_ylabel('RMSE (dex)', fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.set_xticks(np.arange(len(AREA_OPTIONS)))
        ax.set_xticklabels([AREA_OPTION_LABELS[ao] for ao in AREA_OPTIONS], fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        if idx == 0:
            ax.legend(loc='best', fontsize=9)
    
    # Plot reduced chi-squared (bottom row)
    for idx, (gas_type, title) in enumerate(zip(gas_types, titles_chi2)):
        ax = axes[1, idx]
        
        # Organize data by prescription and area
        for prescription in H2_PRESCRIPTIONS:
            area_chi2 = []
            area_labels = []
            
            for area_option in AREA_OPTIONS:
                model_key = f'{prescription}_{area_option}'
                if model_key in statistics:
                    chi2 = statistics[model_key].get(f'{gas_type}_chi2red', np.nan)
                    if np.isfinite(chi2):
                        area_chi2.append(chi2)
                        area_labels.append(AREA_OPTION_LABELS[area_option])
            
            if len(area_chi2) > 0:
                x_pos = np.arange(len(area_chi2)) + H2_PRESCRIPTIONS.index(prescription) * 0.15
                ax.bar(x_pos, area_chi2, width=0.15, 
                      label=prescription, color=H2_PRESCRIPTION_COLORS[prescription], alpha=0.8)
        
        # Add horizontal line at chi^2 = 1 (perfect fit)
        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='χ²=1 (ideal)')
        
        ax.set_ylabel('Reduced χ²', fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.set_xticks(np.arange(len(AREA_OPTIONS)))
        ax.set_xticklabels([AREA_OPTION_LABELS[ao] for ao in AREA_OPTIONS], fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        if idx == 0:
            ax.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    os.makedirs(OutputDir, exist_ok=True)
    output_path = os.path.join(OutputDir, 'model_statistics_comparison.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n  Saved: {output_path}")
    plt.close()



def plot_ssfr_vs_stellar_mass(target_z=0.0):
    """
    Plot sSFR vs. stellar mass for different H2 prescriptions and area options
    """
    mass_bins = np.arange(8.0, 12.5, 0.2)
    # Find snapshot closest to target redshift
    z_diff = [abs(z - target_z) for z in DEFAULT_REDSHIFTS]
    best_snap = z_diff.index(min(z_diff))
    actual_z = DEFAULT_REDSHIFTS[best_snap]
    snap_str = f'Snap_{best_snap}'
    print(f"\n{'='*60}")
    print(f"Creating sSFR vs. stellar mass plot for z ~ {target_z}")
    print(f"Using {snap_str} (z={actual_z:.2f})")
    print(f"{'='*60}")
    fig, ax = plt.subplots(figsize=(10, 7))
    for prescription in H2_PRESCRIPTIONS:
        color = H2_PRESCRIPTION_COLORS[prescription]
        for area_option in AREA_OPTIONS:
            directory = f'./output/millennium_{prescription.lower()}_{area_option}/'
            linestyle = AREA_OPTION_LINESTYLES[area_option]
            if not os.path.exists(directory):
                print(f"  Warning: Directory {directory} not found, skipping")
                continue
            print(f"  Processing {prescription} with {area_option}...")
            try:
                stellar_mass = read_hdf(directory, snap_num=snap_str, param='StellarMass')
                sfr_disk = read_hdf(directory, snap_num=snap_str, param='SfrDisk')
                sfr_bulge = read_hdf(directory, snap_num=snap_str, param='SfrBulge')
                galaxy_type = read_hdf(directory, snap_num=snap_str, param='Type')
                if stellar_mass is None or sfr_disk is None or sfr_bulge is None:
                    continue
                # Convert to solar masses
                stellar_mass = stellar_mass * 1.0e10 / MILLENNIUM_HUBBLE_H
                sfr = sfr_disk + sfr_bulge
                # Filter for central galaxies with stellar mass
                mask = (stellar_mass > 1e8) & (galaxy_type == 0)
                stellar_mass_filtered = stellar_mass[mask]
                sfr_filtered = sfr[mask]
                if len(stellar_mass_filtered) == 0:
                    continue
                log_stellar_mass = np.log10(stellar_mass_filtered)
                # Calculate sSFR (yr^-1)
                ssfr = sfr_filtered / stellar_mass_filtered
                # Calculate median sSFR in bins
                bin_centers = 0.5 * (mass_bins[:-1] + mass_bins[1:])
                bin_medians = []
                valid_bins = []
                for i in range(len(mass_bins)-1):
                    bin_mask = (log_stellar_mass >= mass_bins[i]) & (log_stellar_mass < mass_bins[i+1])
                    if np.sum(bin_mask) > 10:
                        bin_medians.append(np.median(ssfr[bin_mask]))
                        valid_bins.append(bin_centers[i])
                if len(valid_bins) > 0:
                    label = f'{prescription} {AREA_OPTION_LABELS[area_option]}'
                    ax.plot(valid_bins, bin_medians, color=color, linestyle=linestyle,
                            linewidth=2, label=label, alpha=0.8)
            except Exception as e:
                print(f"    Error: {e}")
                continue
    ax.set_xlabel(r'$\log_{10}(M_\star / \mathrm{M}_\odot)$', fontsize=14)
    ax.set_ylabel(r'sSFR (yr$^{-1}$)', fontsize=14)
    ax.set_xlim(8.0, 12.0)
    ax.set_ylim(1e-14, 1e-8)
    ax.set_yscale('log')
    ax.legend(loc='best', fontsize=9, framealpha=0.9, ncol=2)
    os.makedirs(OutputDir, exist_ok=True)
    output_path = os.path.join(OutputDir, f'ssfr_vs_stellar_mass_area_comparison_z{target_z:.1f}.pdf')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n  Saved: {output_path}")
    plt.close()

def plot_hydrogen_fraction_vs_cold_gas_surface_density(target_z=0.0):
    """
    Plot hydrogen fraction (H2/(HI+H2)) as a function of cold gas surface density for different H2 prescriptions and area options
    """
    # Surface density bins (log10)
    sigma_bins = np.arange(0.0, 10.2, 0.2)  # Cover full log_sigma range
    z_diff = [abs(z - target_z) for z in DEFAULT_REDSHIFTS]
    best_snap = z_diff.index(min(z_diff))
    actual_z = DEFAULT_REDSHIFTS[best_snap]
    snap_str = f'Snap_{best_snap}'
    print(f"\n{'='*60}")
    print(f"Creating hydrogen fraction vs. cold gas surface density plot for z ~ {target_z}")
    print(f"Using {snap_str} (z={actual_z:.2f})")
    print(f"{'='*60}")
    fig, ax = plt.subplots(figsize=(10, 7))
    for prescription in H2_PRESCRIPTIONS:
        color = H2_PRESCRIPTION_COLORS[prescription]
        for area_option in AREA_OPTIONS:
            directory = f'./output/millennium_{prescription.lower()}_{area_option}/'
            linestyle = AREA_OPTION_LINESTYLES[area_option]
            if not os.path.exists(directory):
                print(f"  Warning: Directory {directory} not found, skipping")
                continue
            print(f"  Processing {prescription} with {area_option}...")
            try:
                h2_gas = read_hdf(directory, snap_num=snap_str, param='H2gas')
                cold_gas = read_hdf(directory, snap_num=snap_str, param='ColdGas')
                galaxy_type = read_hdf(directory, snap_num=snap_str, param='Type')
                disk_radius = read_hdf(directory, snap_num=snap_str, param='DiskRadius')
                if h2_gas is None or cold_gas is None or disk_radius is None:
                    print(f"    Debug: Missing data for {prescription} {area_option} (h2_gas: {h2_gas is None}, cold_gas: {cold_gas is None}, disk_radius: {disk_radius is None})")
                    continue
                h2_gas = h2_gas * 1.0e10 / MILLENNIUM_HUBBLE_H
                cold_gas = cold_gas * 1.0e10 / MILLENNIUM_HUBBLE_H
                hi_gas = cold_gas - h2_gas
                hi_gas[hi_gas < 0] = 0
                disk_radius_pc = disk_radius * 1.0e6
                disk_area_pc2 = np.pi * disk_radius_pc**2
                sigma_cold_gas = cold_gas / disk_area_pc2
                mask = (cold_gas > 0) & (disk_area_pc2 > 0) & (galaxy_type == 0)
                print(f"    Debug: {prescription} {area_option} - Total galaxies: {len(cold_gas)}, After mask: {np.sum(mask)}")
                if np.sum(mask) == 0:
                    print(f"    Debug: No galaxies pass mask for {prescription} {area_option}")
                    continue
                sigma_cold_gas = sigma_cold_gas[mask]
                h2_gas_filtered = h2_gas[mask]
                hi_gas_filtered = hi_gas[mask]
                total_hydrogen = hi_gas_filtered + h2_gas_filtered
                hydrogen_fraction = np.zeros_like(h2_gas_filtered)
                valid = total_hydrogen > 0
                print(f"    Debug: {prescription} {area_option} - Total hydrogen > 0: {np.sum(valid)}")
                if np.sum(valid) == 0:
                    print(f"    Debug: No valid hydrogen for {prescription} {area_option}")
                    continue
                hydrogen_fraction[valid] = h2_gas_filtered[valid] / total_hydrogen[valid]
                log_sigma = np.log10(sigma_cold_gas[valid])
                hydrogen_fraction = hydrogen_fraction[valid]
                print(f"    Debug: {prescription} {area_option} - log_sigma range: {log_sigma.min():.2f} to {log_sigma.max():.2f}, hydrogen_fraction range: {hydrogen_fraction.min():.2f} to {hydrogen_fraction.max():.2f}")
                bin_centers = 0.5 * (sigma_bins[:-1] + sigma_bins[1:])
                bin_medians = []
                valid_bins = []
                for i in range(len(sigma_bins)-1):
                    bin_mask = (log_sigma >= sigma_bins[i]) & (log_sigma < sigma_bins[i+1])
                    if np.sum(bin_mask) >= 3:
                        bin_medians.append(np.median(hydrogen_fraction[bin_mask]))
                        valid_bins.append(bin_centers[i])
                    else:
                        print(f"    Debug: Bin {i} ({sigma_bins[i]:.2f}-{sigma_bins[i+1]:.2f}) has {np.sum(bin_mask)} galaxies")
                if len(valid_bins) > 0:
                    label = f'{prescription} {AREA_OPTION_LABELS[area_option]}'
                    ax.plot(valid_bins, bin_medians, color=color, linestyle=linestyle,
                            linewidth=2, label=label, alpha=0.8)
                else:
                    print(f"    Debug: No valid bins for {prescription} {area_option}")
            except Exception as e:
                print(f"    Error: {e}")
                continue
    ax.set_xlabel(r'$\log_{10}(\Sigma_{\mathrm{cold\,gas}} / \mathrm{M}_\odot\,\mathrm{pc}^{-2})$', fontsize=14)
    ax.set_ylabel(r'Hydrogen Fraction (H$_2$/(HI+H$_2$))', fontsize=14)
    # ax.set_xlim(0, 3)
    # ax.set_ylim(0, 1)
    ax.legend(loc='best', fontsize=9, framealpha=0.9, ncol=2)
    os.makedirs(OutputDir, exist_ok=True)
    output_path = os.path.join(OutputDir, f'hydrogen_fraction_vs_cold_gas_surface_density_z{target_z:.1f}.pdf')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n  Saved: {output_path}")
    plt.close()
    
def plot_hi_mass_vs_stellar_mass(target_z=0.0):
    """
    Plot HI mass vs. stellar mass for different H2 prescriptions and area options
    """
    mass_bins = np.arange(8.0, 12.5, 0.2)
    z_diff = [abs(z - target_z) for z in DEFAULT_REDSHIFTS]
    best_snap = z_diff.index(min(z_diff))
    actual_z = DEFAULT_REDSHIFTS[best_snap]
    snap_str = f'Snap_{best_snap}'
    print(f"\n{'='*60}")
    print(f"Creating HI mass vs. stellar mass plot for z ~ {target_z}")
    print(f"Using {snap_str} (z={actual_z:.2f})")
    print(f"{'='*60}")
    fig, ax = plt.subplots(figsize=(10, 7))
    for prescription in H2_PRESCRIPTIONS:
        color = H2_PRESCRIPTION_COLORS[prescription]
        for area_option in AREA_OPTIONS:
            directory = f'./output/millennium_{prescription.lower()}_{area_option}/'
            linestyle = AREA_OPTION_LINESTYLES[area_option]
            if not os.path.exists(directory):
                print(f"  Warning: Directory {directory} not found, skipping")
                continue
            print(f"  Processing {prescription} with {area_option}...")
            try:
                stellar_mass = read_hdf(directory, snap_num=snap_str, param='StellarMass')
                h2_gas = read_hdf(directory, snap_num=snap_str, param='H2gas')
                cold_gas = read_hdf(directory, snap_num=snap_str, param='ColdGas')
                galaxy_type = read_hdf(directory, snap_num=snap_str, param='Type')
                if stellar_mass is None or h2_gas is None or cold_gas is None:
                    continue
                stellar_mass = stellar_mass * 1.0e10 / MILLENNIUM_HUBBLE_H
                h2_gas = h2_gas * 1.0e10 / MILLENNIUM_HUBBLE_H
                cold_gas = cold_gas * 1.0e10 / MILLENNIUM_HUBBLE_H
                hi_gas = cold_gas - h2_gas
                hi_gas[hi_gas < 0] = 0
                mask = (stellar_mass > 0) & (hi_gas > 0) & (galaxy_type == 0)
                stellar_mass_filtered = stellar_mass[mask]
                hi_gas_filtered = hi_gas[mask]
                if len(stellar_mass_filtered) == 0:
                    continue
                log_stellar_mass = np.log10(stellar_mass_filtered)
                log_hi_mass = np.log10(hi_gas_filtered)
                bin_centers = 0.5 * (mass_bins[:-1] + mass_bins[1:])
                bin_medians = []
                valid_bins = []
                for i in range(len(mass_bins)-1):
                    bin_mask = (log_stellar_mass >= mass_bins[i]) & (log_stellar_mass < mass_bins[i+1])
                    if np.sum(bin_mask) > 10:
                        bin_medians.append(np.median(log_hi_mass[bin_mask]))
                        valid_bins.append(bin_centers[i])
                if len(valid_bins) > 0:
                    label = f'{prescription} {AREA_OPTION_LABELS[area_option]}'
                    ax.plot(valid_bins, bin_medians, color=color, linestyle=linestyle,
                            linewidth=2, label=label, alpha=0.8)
            except Exception as e:
                print(f"    Error: {e}")
                continue
    ax.set_xlabel(r'$\log_{10}(M_\star / \mathrm{M}_\odot)$', fontsize=14)
    ax.set_ylabel(r'$\log_{10}(M_{\mathrm{HI}} / \mathrm{M}_\odot)$', fontsize=14)
    ax.set_xlim(8.0, 12.0)
    ax.set_ylim(7, 11)
    ax.legend(loc='best', fontsize=9, framealpha=0.9, ncol=2)
    os.makedirs(OutputDir, exist_ok=True)
    output_path = os.path.join(OutputDir, f'hi_mass_vs_stellar_mass_area_comparison_z{target_z:.1f}.pdf')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n  Saved: {output_path}")
    plt.close()

def plot_h2_mass_vs_stellar_mass(target_z=0.0):
    """
    Plot H2 mass vs. stellar mass for different H2 prescriptions and area options
    """
    mass_bins = np.arange(8.0, 12.5, 0.2)
    z_diff = [abs(z - target_z) for z in DEFAULT_REDSHIFTS]
    best_snap = z_diff.index(min(z_diff))
    actual_z = DEFAULT_REDSHIFTS[best_snap]
    snap_str = f'Snap_{best_snap}'
    print(f"\n{'='*60}")
    print(f"Creating H2 mass vs. stellar mass plot for z ~ {target_z}")
    print(f"Using {snap_str} (z={actual_z:.2f})")
    print(f"{'='*60}")
    fig, ax = plt.subplots(figsize=(10, 7))
    for prescription in H2_PRESCRIPTIONS:
        color = H2_PRESCRIPTION_COLORS[prescription]
        for area_option in AREA_OPTIONS:
            directory = f'./output/millennium_{prescription.lower()}_{area_option}/'
            linestyle = AREA_OPTION_LINESTYLES[area_option]
            if not os.path.exists(directory):
                print(f"  Warning: Directory {directory} not found, skipping")
                continue
            print(f"  Processing {prescription} with {area_option}...")
            try:
                stellar_mass = read_hdf(directory, snap_num=snap_str, param='StellarMass')
                h2_gas = read_hdf(directory, snap_num=snap_str, param='H2gas')
                galaxy_type = read_hdf(directory, snap_num=snap_str, param='Type')
                if stellar_mass is None or h2_gas is None:
                    continue
                stellar_mass = stellar_mass * 1.0e10 / MILLENNIUM_HUBBLE_H
                h2_gas = h2_gas * 1.0e10 / MILLENNIUM_HUBBLE_H
                mask = (stellar_mass > 0) & (h2_gas > 0) & (galaxy_type == 0)
                stellar_mass_filtered = stellar_mass[mask]
                h2_gas_filtered = h2_gas[mask]
                if len(stellar_mass_filtered) == 0:
                    continue
                log_stellar_mass = np.log10(stellar_mass_filtered)
                log_h2_mass = np.log10(h2_gas_filtered)
                bin_centers = 0.5 * (mass_bins[:-1] + mass_bins[1:])
                bin_medians = []
                valid_bins = []
                for i in range(len(mass_bins)-1):
                    bin_mask = (log_stellar_mass >= mass_bins[i]) & (log_stellar_mass < mass_bins[i+1])
                    if np.sum(bin_mask) > 10:
                        bin_medians.append(np.median(log_h2_mass[bin_mask]))
                        valid_bins.append(bin_centers[i])
                if len(valid_bins) > 0:
                    label = f'{prescription} {AREA_OPTION_LABELS[area_option]}'
                    ax.plot(valid_bins, bin_medians, color=color, linestyle=linestyle,
                            linewidth=2, label=label, alpha=0.8)
            except Exception as e:
                print(f"    Error: {e}")
                continue
    ax.set_xlabel(r'$\log_{10}(M_\star / \mathrm{M}_\odot)$', fontsize=14)
    ax.set_ylabel(r'$\log_{10}(M_{\mathrm{H}_2} / \mathrm{M}_\odot)$', fontsize=14)
    ax.set_xlim(8.0, 12.0)
    ax.set_ylim(7, 11)
    ax.legend(loc='best', fontsize=9, framealpha=0.9, ncol=2)
    os.makedirs(OutputDir, exist_ok=True)
    output_path = os.path.join(OutputDir, f'h2_mass_vs_stellar_mass_area_comparison_z{target_z:.1f}.pdf')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n  Saved: {output_path}")
    plt.close()

if __name__ == '__main__':
    print("="*60)
    print("SAGE H2 Area Comparison Plots")
    print("="*60)
    # Check which directories exist
    print("\nChecking available model runs:")
    found_runs = 0
    for prescription in H2_PRESCRIPTIONS:
        for area_option in AREA_OPTIONS:
            directory = f'./output/millennium_{prescription.lower()}_{area_option}/'
            if os.path.exists(directory):
                print(f"  ✓ {prescription} {area_option}")
                found_runs += 1
            else:
                print(f"  ✗ {prescription} {area_option}")
    if found_runs == 0:
        print("\nError: No model runs found!")
        print("Please run run_h2_area_tests.sh first")
        exit(1)
    print(f"\nFound {found_runs} model run(s)")
    # Create comparison plots
    print("\n" + "="*60)
    print("Creating HI mass fraction comparison plot...")
    print("="*60)
    plot_hi_mass_fraction_comparison(target_z=0.0)
    print("\n" + "="*60)
    print("Creating H2 mass fraction comparison plot...")
    print("="*60)
    plot_h2_mass_fraction_comparison(target_z=0.0)
    print("\n" + "="*60)
    print("Creating cold gas mass fraction comparison plot...")
    print("="*60)
    plot_cold_gas_mass_fraction_comparison(target_z=0.0)
    print("\n" + "="*60)
    print("Creating metallicity comparison plot...")
    print("="*60)
    plot_metallicity_comparison(target_z=0.0)
    print("\n" + "="*60)
    print("Creating quiescent fraction comparison plot...")
    print("="*60)
    plot_quiescent_fraction_comparison(target_z=0.0)
    print("\n" + "="*60)
    print("Creating sSFR vs. stellar mass plot...")
    print("="*60)
    plot_ssfr_vs_stellar_mass(target_z=0.0)
    print("\n" + "="*60)
    print("Creating HI mass vs. stellar mass plot...")
    print("="*60)
    plot_hi_mass_vs_stellar_mass(target_z=0.0)
    print("\n" + "="*60)
    print("Creating H2 mass vs. stellar mass plot...")
    print("="*60)
    plot_h2_mass_vs_stellar_mass(target_z=0.0)
    print("\n" + "="*60)
    print("Creating hydrogen fraction vs. cold gas surface density plot...")
    print("="*60)
    plot_hydrogen_fraction_vs_cold_gas_surface_density(target_z=0.0)
    print("\n" + "="*60)
    print("Creating SFR surface density vs. cold gas surface density plot...")
    print("="*60)
    plot_sfr_surface_density_vs_cold_gas_surface_density(target_z=0.0)
    print("\n" + "="*60)
    print("Creating disk radius vs. stellar mass plot...")
    print("="*60)
    plot_radius_vs_stellar_mass(target_z=0.0)
    # Calculate and plot statistics
    print("\n" + "="*60)
    print("Calculating model-observation statistics...")
    print("="*60)
    statistics = calculate_model_statistics(target_z=0.0)
    plot_model_statistics(statistics)
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)

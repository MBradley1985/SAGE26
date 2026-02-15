#!/usr/bin/env python3
"""
Plot DarkMode disk structure and angular momentum distributions.

This script visualizes the radially-resolved disk arrays from DarkSage-style
disk modeling, including:
- Gas and stellar surface density profiles
- Angular momentum distributions  
- Mass-weighted disk properties
- Comparison across galaxy stellar mass bins

Usage:
    python plotting/plot_darkmode_disks.py [--output PREFIX] [--snap SNAPNUM]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import h5py
from pathlib import Path

# Plot styling
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['figure.figsize'] = (12, 10)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10


def load_darkmode_data(filepath, snap=63):
    """Load DarkMode disk arrays from HDF5 output."""
    with h5py.File(filepath, 'r') as f:
        snap_grp = f[f'Snap_{snap}']
        
        data = {
            'StellarMass': snap_grp['StellarMass'][:],
            'ColdGas': snap_grp['ColdGas'][:],
            'MetalsColdGas': snap_grp['MetalsColdGas'][:],
            'Vvir': snap_grp['Vvir'][:],
            'DiskScaleRadius': snap_grp['DiskRadius'][:],  # DiskRadius in output
        }
        
        # Check for DarkMode arrays
        if 'DiscGas' in snap_grp:
            data['DiscGas'] = snap_grp['DiscGas'][:]
            data['DiscStars'] = snap_grp['DiscStars'][:]
            data['DiscGasMetals'] = snap_grp['DiscGasMetals'][:]
            data['DiscStarsMetals'] = snap_grp['DiscStarsMetals'][:]
            data['DiscRadii'] = snap_grp['DiscRadii'][:]
            data['has_darkmode'] = True
            
            # Load new DarkMode features (might not exist in older outputs)
            if 'DiscSFR' in snap_grp:
                data['DiscSFR'] = snap_grp['DiscSFR'][:]
            if 'DiscH2' in snap_grp:
                data['DiscH2'] = snap_grp['DiscH2'][:]
            if 'DiscHI' in snap_grp:
                data['DiscHI'] = snap_grp['DiscHI'][:]
        else:
            data['has_darkmode'] = False
            print("Warning: No DarkMode arrays found in output")
        
        # Load bulk properties for comparison
        if 'BulgeMass' in snap_grp:
            data['BulgeMass'] = snap_grp['BulgeMass'][:]
        if 'InstabilityBulgeMass' in snap_grp:
            data['InstabilityBulgeMass'] = snap_grp['InstabilityBulgeMass'][:]
        if 'H2gas' in snap_grp:
            data['H2gas'] = snap_grp['H2gas'][:]
        if 'H1gas' in snap_grp:
            data['H1gas'] = snap_grp['H1gas'][:]
        
        # Try to load spin vectors if available
        for spin in ['SpinGas', 'SpinStars']:
            if spin in snap_grp:
                data[spin] = snap_grp[spin][:]
                
    return data


def compute_surface_density(disc_mass, disc_radii):
    """
    Compute surface density from annular mass and radii.
    
    Parameters
    ----------
    disc_mass : array (ngal, N_BINS)
        Mass in each annulus
    disc_radii : array (ngal, N_BINS+1)
        Bin edges (inner and outer radii)
        
    Returns
    -------
    sigma : array (ngal, N_BINS)
        Surface density in each annulus
    r_mid : array (ngal, N_BINS)
        Mid-point radius of each annulus
    """
    # Calculate annulus areas: π(r_out² - r_in²)
    r_in = disc_radii[:, :-1]
    r_out = disc_radii[:, 1:]
    area = np.pi * (r_out**2 - r_in**2)
    
    # Avoid division by zero
    area = np.where(area > 0, area, np.nan)
    
    sigma = disc_mass / area
    r_mid = 0.5 * (r_in + r_out)
    
    return sigma, r_mid


def plot_individual_profiles(data, output_prefix, n_examples=6):
    """Plot individual galaxy disk profiles."""
    if not data['has_darkmode']:
        print("Skipping individual profiles - no DarkMode data")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()
    
    # Select galaxies with significant disk mass spanning mass range
    stellar_mass = data['StellarMass']
    disc_gas_sum = np.sum(data['DiscGas'], axis=1)
    disc_stars_sum = np.sum(data['DiscStars'], axis=1)
    disc_radii = data['DiscRadii']
    
    # Find galaxies with actual disk content and valid radii
    has_disk = (disc_gas_sum > 0.05) & (disc_stars_sum > 0.1) & (disc_radii[:, 1] > 0)
    valid_idx = np.where(has_disk)[0]
    
    if len(valid_idx) < n_examples:
        print(f"Warning: Only {len(valid_idx)} galaxies with adequate disk content")
        n_examples = max(1, len(valid_idx))
    
    if n_examples == 0:
        print("No galaxies with disk content to plot")
        return
    
    # Sample across stellar mass range
    mass_sorted = np.argsort(stellar_mass[valid_idx])
    sample_idx = valid_idx[mass_sorted[np.linspace(0, len(mass_sorted)-1, n_examples, dtype=int)]]
    
    for ax_idx, gal_idx in enumerate(sample_idx):
        ax = axes[ax_idx]
        
        # Get radii in kpc
        r_edges = data['DiscRadii'][gal_idx] * 1000  # Convert to kpc
        r_mid = 0.5 * (r_edges[:-1] + r_edges[1:])
        
        gas = data['DiscGas'][gal_idx]
        stars = data['DiscStars'][gal_idx]
        
        # Find last bin with significant content
        last_gas = np.where(gas > 1e-6)[0]
        last_stars = np.where(stars > 1e-6)[0]
        last_bin = max(last_gas.max() if len(last_gas) > 0 else 0,
                       last_stars.max() if len(last_stars) > 0 else 0)
        last_bin = min(last_bin + 2, len(gas))
        
        if last_bin < 2:
            last_bin = min(10, len(gas))
        
        # Plot as bar chart for clarity
        width = np.diff(r_edges[:last_bin+1])
        ax.bar(r_mid[:last_bin], gas[:last_bin], width=width*0.4, alpha=0.7,
               label='Gas', color='C0', align='center')
        ax.bar(r_mid[:last_bin] + width*0.2, stars[:last_bin], width=width*0.4, alpha=0.7,
               label='Stars', color='C1', align='center')
        
        ax.set_xlabel('Radius [kpc]')
        ax.set_ylabel('Mass [10¹⁰ M☉]')
        ax.set_title(f'M★={stellar_mass[gal_idx]:.2f}, Mgas={data["ColdGas"][gal_idx]:.2f}')
        ax.legend(loc='upper right', fontsize=9)
        
        # Set reasonable axis limits
        max_val = max(gas[:last_bin].max(), stars[:last_bin].max())
        if max_val > 0:
            ax.set_ylim(0, max_val * 1.2)
        ax.set_xlim(0, r_mid[last_bin-1] * 1.3 if last_bin > 0 else 10)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_individual_profiles.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_prefix}_individual_profiles.png")


def plot_stacked_profiles(data, output_prefix):
    """Plot stacked/median profiles in stellar mass bins."""
    if not data['has_darkmode']:
        print("Skipping stacked profiles - no DarkMode data")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    stellar_mass = data['StellarMass']
    
    # Define mass bins
    mass_bins = [0.01, 0.1, 1.0, 10.0, 100.0]
    mass_labels = ['0.01-0.1', '0.1-1', '1-10', '10-100']
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(mass_labels)))
    
    # Normalize radii by disk scale radius for stacking
    r_edges = data['DiscRadii']  # (ngal, N_BINS+1)
    disk_scale = data['DiskScaleRadius'][:, np.newaxis]
    disk_scale = np.where(disk_scale > 0, disk_scale, np.nan)
    
    # Normalized radius bins (0 to 5 scale radii)
    r_norm_bins = np.linspace(0, 5, 31)
    r_norm_mid = 0.5 * (r_norm_bins[:-1] + r_norm_bins[1:])
    
    for mass_idx, (m_lo, m_hi) in enumerate(zip(mass_bins[:-1], mass_bins[1:])):
        mask = (stellar_mass >= m_lo) & (stellar_mass < m_hi)
        n_gal = np.sum(mask)
        
        if n_gal < 5:
            continue
        
        # Get normalized mid-radii for each galaxy
        r_mid = 0.5 * (r_edges[mask, :-1] + r_edges[mask, 1:])
        r_norm = r_mid / disk_scale[mask]
        
        # Interpolate to common normalized radius grid
        gas_interp = np.zeros((n_gal, len(r_norm_mid)))
        stars_interp = np.zeros((n_gal, len(r_norm_mid)))
        
        for i in range(n_gal):
            valid = ~np.isnan(r_norm[i]) & (r_norm[i] > 0)
            if np.sum(valid) > 2:
                gas_interp[i] = np.interp(r_norm_mid, r_norm[i, valid], 
                                          data['DiscGas'][mask][i, valid], left=0, right=0)
                stars_interp[i] = np.interp(r_norm_mid, r_norm[i, valid],
                                            data['DiscStars'][mask][i, valid], left=0, right=0)
        
        # Compute median profiles
        gas_median = np.nanmedian(gas_interp, axis=0)
        stars_median = np.nanmedian(stars_interp, axis=0)
        gas_16 = np.nanpercentile(gas_interp, 16, axis=0)
        gas_84 = np.nanpercentile(gas_interp, 84, axis=0)
        stars_16 = np.nanpercentile(stars_interp, 16, axis=0)
        stars_84 = np.nanpercentile(stars_interp, 84, axis=0)
        
        # Plot gas profiles
        axes[0, 0].plot(r_norm_mid, gas_median, color=colors[mass_idx], 
                        label=f'{mass_labels[mass_idx]} (N={n_gal})', linewidth=2)
        axes[0, 0].fill_between(r_norm_mid, gas_16, gas_84, color=colors[mass_idx], alpha=0.2)
        
        # Plot stellar profiles
        axes[0, 1].plot(r_norm_mid, stars_median, color=colors[mass_idx],
                        label=f'{mass_labels[mass_idx]}', linewidth=2)
        axes[0, 1].fill_between(r_norm_mid, stars_16, stars_84, color=colors[mass_idx], alpha=0.2)
    
    axes[0, 0].set_xlabel('r / R_d')
    axes[0, 0].set_ylabel('Gas Mass [10¹⁰ M☉]')
    axes[0, 0].set_title('Stacked Gas Profiles by Stellar Mass')
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_ylim(1e-4, 1)
    
    axes[0, 1].set_xlabel('r / R_d')
    axes[0, 1].set_ylabel('Stellar Mass [10¹⁰ M☉]')
    axes[0, 1].set_title('Stacked Stellar Profiles by Stellar Mass')
    axes[0, 1].legend()
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_ylim(1e-4, 10)
    
    # Plot metallicity profiles
    for mass_idx, (m_lo, m_hi) in enumerate(zip(mass_bins[:-1], mass_bins[1:])):
        mask = (stellar_mass >= m_lo) & (stellar_mass < m_hi)
        n_gal = np.sum(mask)
        
        if n_gal < 5:
            continue
        
        r_mid = 0.5 * (r_edges[mask, :-1] + r_edges[mask, 1:])
        r_norm = r_mid / disk_scale[mask]
        
        # Gas metallicity = DiscGasMetals / DiscGas
        with np.errstate(divide='ignore', invalid='ignore'):
            gas_Z = data['DiscGasMetals'][mask] / data['DiscGas'][mask]
            gas_Z = np.where(np.isfinite(gas_Z) & (data['DiscGas'][mask] > 1e-6), gas_Z, np.nan)
        
        # Interpolate
        Z_interp = np.full((n_gal, len(r_norm_mid)), np.nan)
        for i in range(n_gal):
            valid = ~np.isnan(r_norm[i]) & ~np.isnan(gas_Z[i]) & (r_norm[i] > 0)
            if np.sum(valid) > 2:
                Z_interp[i] = np.interp(r_norm_mid, r_norm[i, valid], gas_Z[i, valid])
        
        Z_median = np.nanmedian(Z_interp, axis=0)
        axes[1, 0].plot(r_norm_mid, Z_median, color=colors[mass_idx], 
                        label=f'{mass_labels[mass_idx]}', linewidth=2)
    
    axes[1, 0].set_xlabel('r / R_d')
    axes[1, 0].set_ylabel('Gas Metallicity (Z)')
    axes[1, 0].set_title('Gas Metallicity Gradients')
    axes[1, 0].legend()
    axes[1, 0].axhline(0.02, color='gray', linestyle='--', alpha=0.5, label='Solar')
    
    # Plot gas fraction profiles
    for mass_idx, (m_lo, m_hi) in enumerate(zip(mass_bins[:-1], mass_bins[1:])):
        mask = (stellar_mass >= m_lo) & (stellar_mass < m_hi)
        n_gal = np.sum(mask)
        
        if n_gal < 5:
            continue
        
        r_mid = 0.5 * (r_edges[mask, :-1] + r_edges[mask, 1:])
        r_norm = r_mid / disk_scale[mask]
        
        # Gas fraction = DiscGas / (DiscGas + DiscStars)
        total = data['DiscGas'][mask] + data['DiscStars'][mask]
        with np.errstate(divide='ignore', invalid='ignore'):
            f_gas = data['DiscGas'][mask] / total
            f_gas = np.where(np.isfinite(f_gas) & (total > 1e-6), f_gas, np.nan)
        
        # Interpolate
        fgas_interp = np.full((n_gal, len(r_norm_mid)), np.nan)
        for i in range(n_gal):
            valid = ~np.isnan(r_norm[i]) & ~np.isnan(f_gas[i]) & (r_norm[i] > 0)
            if np.sum(valid) > 2:
                fgas_interp[i] = np.interp(r_norm_mid, r_norm[i, valid], f_gas[i, valid])
        
        fgas_median = np.nanmedian(fgas_interp, axis=0)
        axes[1, 1].plot(r_norm_mid, fgas_median, color=colors[mass_idx],
                        label=f'{mass_labels[mass_idx]}', linewidth=2)
    
    axes[1, 1].set_xlabel('r / R_d')
    axes[1, 1].set_ylabel('Gas Fraction')
    axes[1, 1].set_title('Gas Fraction Profiles')
    axes[1, 1].legend()
    axes[1, 1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_stacked_profiles.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_prefix}_stacked_profiles.png")


def plot_angular_momentum(data, output_prefix):
    """Plot angular momentum distributions."""
    if not data['has_darkmode']:
        print("Skipping angular momentum plots - no DarkMode data")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    stellar_mass = data['StellarMass']
    vvir = data['Vvir']
    
    # Compute specific angular momentum from disc radii
    # j = r × Vvir (approximate for disk in circular orbits)
    r_edges = data['DiscRadii']
    r_mid = 0.5 * (r_edges[:, :-1] + r_edges[:, 1:])
    
    # Mass-weighted mean radius
    gas_mass = data['DiscGas']
    stars_mass = data['DiscStars']
    
    with np.errstate(divide='ignore', invalid='ignore'):
        gas_r_mean = np.sum(r_mid * gas_mass, axis=1) / np.sum(gas_mass, axis=1)
        stars_r_mean = np.sum(r_mid * stars_mass, axis=1) / np.sum(stars_mass, axis=1)
    
    # Specific angular momentum j ~ r × Vvir
    j_gas = gas_r_mean * vvir
    j_stars = stars_r_mean * vvir
    
    # Valid data mask
    valid_gas = np.isfinite(j_gas) & (j_gas > 0) & (stellar_mass > 0.01)
    valid_stars = np.isfinite(j_stars) & (j_stars > 0) & (stellar_mass > 0.01)
    
    # j-M★ relation
    ax = axes[0, 0]
    ax.scatter(stellar_mass[valid_gas], j_gas[valid_gas], s=1, alpha=0.3, c='C0', label='Gas')
    ax.scatter(stellar_mass[valid_stars], j_stars[valid_stars], s=1, alpha=0.3, c='C1', label='Stars')
    
    # Binned medians
    mass_bins = np.logspace(-2, 2, 20)
    mass_mid = np.sqrt(mass_bins[:-1] * mass_bins[1:])
    
    for data_arr, valid, color, label in [(j_gas, valid_gas, 'C0', 'Gas'),
                                           (j_stars, valid_stars, 'C1', 'Stars')]:
        j_median = []
        for m_lo, m_hi in zip(mass_bins[:-1], mass_bins[1:]):
            mask = valid & (stellar_mass >= m_lo) & (stellar_mass < m_hi)
            if np.sum(mask) > 5:
                j_median.append(np.median(data_arr[mask]))
            else:
                j_median.append(np.nan)
        ax.plot(mass_mid, j_median, color=color, linewidth=3, label=f'{label} median')
    
    # Fall & Efstathiou relation: j ∝ M^(2/3)
    m_ref = np.logspace(-2, 2, 100)
    j_ref = 0.1 * m_ref**(2/3)
    ax.plot(m_ref, j_ref, 'k--', alpha=0.5, label='j ∝ M★^(2/3)')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Stellar Mass [10¹⁰ M☉]')
    ax.set_ylabel('Specific Angular Momentum [kpc km/s]')
    ax.set_title('j-M★ Relation')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_xlim(0.01, 100)
    ax.set_ylim(1e-3, 10)
    
    # j_stars vs j_gas
    ax = axes[0, 1]
    both_valid = valid_gas & valid_stars
    ax.scatter(j_gas[both_valid], j_stars[both_valid], s=5, alpha=0.3, 
               c=np.log10(stellar_mass[both_valid]), cmap='viridis')
    ax.plot([1e-4, 10], [1e-4, 10], 'k--', alpha=0.5, label='1:1')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('j_gas [kpc km/s]')
    ax.set_ylabel('j_stars [kpc km/s]')
    ax.set_title('Gas vs Stellar Angular Momentum')
    ax.set_xlim(1e-3, 10)
    ax.set_ylim(1e-3, 10)
    
    # Disk size vs stellar mass
    ax = axes[1, 0]
    disk_scale = data['DiskScaleRadius'] * 1000  # to kpc
    valid = (disk_scale > 0) & (stellar_mass > 0.01)
    ax.scatter(stellar_mass[valid], disk_scale[valid], s=1, alpha=0.3)
    
    # Binned median
    r_median = []
    for m_lo, m_hi in zip(mass_bins[:-1], mass_bins[1:]):
        mask = valid & (stellar_mass >= m_lo) & (stellar_mass < m_hi)
        if np.sum(mask) > 5:
            r_median.append(np.median(disk_scale[mask]))
        else:
            r_median.append(np.nan)
    ax.plot(mass_mid, r_median, 'C1', linewidth=3, label='Median')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Stellar Mass [10¹⁰ M☉]')
    ax.set_ylabel('Disk Scale Radius [kpc]')
    ax.set_title('Size-Mass Relation')
    ax.legend()
    
    # Spin parameter distribution
    ax = axes[1, 1]
    
    # λ = j / (√2 × Vvir × Rvir) - approximate spin parameter
    # For now, just plot j/Vvir distribution
    j_norm = j_stars[valid_stars] / vvir[valid_stars]
    ax.hist(np.log10(j_norm), bins=50, density=True, alpha=0.7, label='Stars')
    
    j_norm_gas = j_gas[valid_gas] / vvir[valid_gas]
    ax.hist(np.log10(j_norm_gas), bins=50, density=True, alpha=0.7, label='Gas')
    
    ax.set_xlabel('log₁₀(j / Vvir) [kpc]')
    ax.set_ylabel('PDF')
    ax.set_title('Specific Angular Momentum Distribution')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_angular_momentum.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_prefix}_angular_momentum.png")


def plot_consistency_check(data, output_prefix):
    """Plot consistency checks: Sum(Disc) vs bulk quantities."""
    if not data['has_darkmode']:
        print("Skipping consistency checks - no DarkMode data")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Sum disc arrays
    disc_gas_sum = np.sum(data['DiscGas'], axis=1)
    disc_stars_sum = np.sum(data['DiscStars'], axis=1)
    disc_gas_metals_sum = np.sum(data['DiscGasMetals'], axis=1)
    
    cold_gas = data['ColdGas']
    stellar_mass = data['StellarMass']
    metals_cold_gas = data['MetalsColdGas']
    
    # Gas consistency - use log scale for visibility
    ax = axes[0, 0]
    valid = cold_gas > 1e-4
    ax.scatter(cold_gas[valid], disc_gas_sum[valid], s=3, alpha=0.5, c='C0', edgecolors='none')
    lim = cold_gas[valid].max() * 1.2
    ax.plot([1e-4, lim], [1e-4, lim], 'r-', linewidth=2, label='1:1')
    ax.set_xlabel('ColdGas (bulk) [10¹⁰ M☉]')
    ax.set_ylabel('Sum(DiscGas) [10¹⁰ M☉]')
    ax.set_title(f'Gas Mass Consistency (N={np.sum(valid)})')
    ax.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1e-4, lim)
    ax.set_ylim(1e-4, lim)
    
    # Stellar consistency
    ax = axes[0, 1]
    valid = stellar_mass > 1e-4
    ax.scatter(stellar_mass[valid], disc_stars_sum[valid], s=3, alpha=0.5, c='C1', edgecolors='none')
    lim = stellar_mass[valid].max() * 1.2
    ax.plot([1e-4, lim], [1e-4, lim], 'r-', linewidth=2, label='1:1')
    ax.set_xlabel('StellarMass (bulk) [10¹⁰ M☉]')
    ax.set_ylabel('Sum(DiscStars) [10¹⁰ M☉]')
    ax.set_title(f'Stellar Mass Consistency (N={np.sum(valid)})')
    ax.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1e-4, lim)
    ax.set_ylim(1e-4, lim)
    
    # Metals consistency
    ax = axes[1, 0]
    valid = metals_cold_gas > 1e-6
    ax.scatter(metals_cold_gas[valid], disc_gas_metals_sum[valid], s=3, alpha=0.5, c='C2', edgecolors='none')
    lim = metals_cold_gas[valid].max() * 1.2
    ax.plot([1e-6, lim], [1e-6, lim], 'r-', linewidth=2, label='1:1')
    ax.set_xlabel('MetalsColdGas (bulk) [10¹⁰ M☉]')
    ax.set_ylabel('Sum(DiscGasMetals) [10¹⁰ M☉]')
    ax.set_title(f'Gas Metals Consistency (N={np.sum(valid)})')
    ax.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1e-6, lim)
    ax.set_ylim(1e-6, lim)
    
    # Relative differences histogram
    ax = axes[1, 1]
    
    valid_gas = cold_gas > 1e-6
    rel_diff_gas = (disc_gas_sum[valid_gas] - cold_gas[valid_gas]) / cold_gas[valid_gas]
    ax.hist(rel_diff_gas, bins=100, alpha=0.7, color='C0', label=f'Gas (med={np.median(rel_diff_gas):.2e})')
    
    valid_stars = stellar_mass > 1e-6
    rel_diff_stars = (disc_stars_sum[valid_stars] - stellar_mass[valid_stars]) / stellar_mass[valid_stars]
    ax.hist(rel_diff_stars, bins=100, alpha=0.7, color='C1', label=f'Stars (med={np.median(rel_diff_stars):.2e})')
    
    ax.axvline(0, color='k', linestyle='--', linewidth=2)
    ax.set_xlabel('Relative Difference (Sum - Bulk) / Bulk')
    ax.set_ylabel('Count')
    ax.set_title('Mass Conservation Check')
    ax.legend()
    # Auto-scale to show data, but focus on small differences
    max_diff = max(np.abs(rel_diff_gas).max(), np.abs(rel_diff_stars).max())
    ax.set_xlim(-max(0.01, max_diff*1.5), max(0.01, max_diff*1.5))
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_consistency.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_prefix}_consistency.png")


def plot_sfr_profiles(data, output_prefix):
    """Plot local SFR profiles showing star formation in each annulus."""
    if not data['has_darkmode']:
        print("Skipping SFR profiles - no DarkMode data")
        return
    
    if 'DiscSFR' not in data:
        print("Skipping SFR profiles - DiscSFR not in output")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()
    
    stellar_mass = data['StellarMass']
    disc_sfr = data['DiscSFR']
    disc_radii = data['DiscRadii']
    
    # Find galaxies with active SF in disk
    total_sfr = np.sum(disc_sfr, axis=1)
    has_sf = total_sfr > 1e-4
    valid_idx = np.where(has_sf)[0]
    
    if len(valid_idx) < 6:
        print(f"Warning: Only {len(valid_idx)} galaxies with disk SF")
        return
    
    # Sample across mass range
    mass_sorted = np.argsort(stellar_mass[valid_idx])
    sample_idx = valid_idx[mass_sorted[np.linspace(0, len(mass_sorted)-1, 6, dtype=int)]]
    
    for ax_idx, gal_idx in enumerate(sample_idx):
        ax = axes[ax_idx]
        
        r_edges = disc_radii[gal_idx] * 1000  # kpc
        r_mid = 0.5 * (r_edges[:-1] + r_edges[1:])
        sfr = disc_sfr[gal_idx]
        
        # Find last bin with SF
        last_bin = np.where(sfr > 1e-6)[0]
        if len(last_bin) > 0:
            last_bin = min(last_bin.max() + 2, len(sfr))
        else:
            last_bin = 10
        
        # Plot as bar chart
        width = np.diff(r_edges[:last_bin+1])
        ax.bar(r_mid[:last_bin], sfr[:last_bin], width=width*0.8, alpha=0.7,
               color='C2', align='center')
        
        ax.set_xlabel('Radius [kpc]')
        ax.set_ylabel('SFR [M☉/yr]')
        total = total_sfr[gal_idx]
        ax.set_title(f'M*={stellar_mass[gal_idx]:.2f}, SFR={total:.3f} M☉/yr')
        
        if r_mid[last_bin-1] > 0:
            ax.set_xlim(0, r_mid[last_bin-1] * 1.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_sfr_profiles.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_prefix}_sfr_profiles.png")


def plot_h2_hi_profiles(data, output_prefix):
    """Plot H2 and HI gas distributions in disk annuli."""
    if not data['has_darkmode']:
        print("Skipping H2/HI profiles - no DarkMode data")
        return
    
    if 'DiscH2' not in data or 'DiscHI' not in data:
        print("Skipping H2/HI profiles - DiscH2/DiscHI not in output")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()
    
    stellar_mass = data['StellarMass']
    disc_gas = data['DiscGas']
    disc_h2 = data['DiscH2']
    disc_hi = data['DiscHI']
    disc_radii = data['DiscRadii']
    
    # Find galaxies with gas
    total_gas = np.sum(disc_gas, axis=1)
    has_gas = total_gas > 0.1
    valid_idx = np.where(has_gas)[0]
    
    if len(valid_idx) < 6:
        print(f"Warning: Only {len(valid_idx)} galaxies with gas")
        return
    
    # Sample across mass range
    mass_sorted = np.argsort(stellar_mass[valid_idx])
    sample_idx = valid_idx[mass_sorted[np.linspace(0, len(mass_sorted)-1, 6, dtype=int)]]
    
    for ax_idx, gal_idx in enumerate(sample_idx):
        ax = axes[ax_idx]
        
        r_edges = disc_radii[gal_idx] * 1000  # kpc
        r_mid = 0.5 * (r_edges[:-1] + r_edges[1:])
        h2 = disc_h2[gal_idx]
        hi = disc_hi[gal_idx]
        
        # Find last bin with gas
        last_bin = np.where((h2 > 1e-6) | (hi > 1e-6))[0]
        if len(last_bin) > 0:
            last_bin = min(last_bin.max() + 2, len(h2))
        else:
            last_bin = 10
        
        # Plot as stacked bar chart
        width = np.diff(r_edges[:last_bin+1])
        ax.bar(r_mid[:last_bin], h2[:last_bin], width=width*0.8, alpha=0.7,
               label='H₂', color='C0', align='center')
        ax.bar(r_mid[:last_bin], hi[:last_bin], width=width*0.8, alpha=0.7,
               label='HI', color='C1', align='center', bottom=h2[:last_bin])
        
        ax.set_xlabel('Radius [kpc]')
        ax.set_ylabel('Mass [10¹⁰ M☉]')
        
        # Calculate H2 fraction
        total_h2 = np.sum(h2)
        total_hi = np.sum(hi)
        f_h2 = total_h2 / (total_h2 + total_hi) if (total_h2 + total_hi) > 0 else 0
        ax.set_title(f'M*={stellar_mass[gal_idx]:.2f}, f_H2={f_h2:.2f}')
        ax.legend(loc='upper right', fontsize=8)
        
        if last_bin > 0 and r_mid[last_bin-1] > 0:
            ax.set_xlim(0, r_mid[last_bin-1] * 1.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_h2_hi_profiles.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_prefix}_h2_hi_profiles.png")


def plot_toomre_q(data, output_prefix):
    """Compute and plot Toomre Q parameter for disk stability."""
    if not data['has_darkmode']:
        print("Skipping Toomre Q - no DarkMode data")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()
    
    stellar_mass = data['StellarMass']
    disc_gas = data['DiscGas']
    disc_stars = data['DiscStars']
    disc_radii = data['DiscRadii']
    vvir = data['Vvir']
    
    # Constants
    h = 0.73  # Hubble parameter
    G = 4.302e-9  # (km/s)^2 kpc / Msun
    
    # Find galaxies with significant disk
    total_disk = np.sum(disc_gas + disc_stars, axis=1)
    has_disk = (total_disk > 0.5) & (vvir > 0)
    valid_idx = np.where(has_disk)[0]
    
    if len(valid_idx) < 6:
        print(f"Warning: Only {len(valid_idx)} galaxies with disk")
        return
    
    # Sample across mass range
    mass_sorted = np.argsort(stellar_mass[valid_idx])
    sample_idx = valid_idx[mass_sorted[np.linspace(0, len(mass_sorted)-1, 6, dtype=int)]]
    
    for ax_idx, gal_idx in enumerate(sample_idx):
        ax = axes[ax_idx]
        
        r_edges = disc_radii[gal_idx] * 1000  # kpc
        r_mid = 0.5 * (r_edges[:-1] + r_edges[1:])
        
        # Surface densities [Msun/pc^2]
        area_pc2 = np.pi * (r_edges[1:]**2 - r_edges[:-1]**2) * 1e6
        sigma_gas = (disc_gas[gal_idx] * 1e10 / h) / area_pc2
        sigma_stars = (disc_stars[gal_idx] * 1e10 / h) / area_pc2
        sigma_total = sigma_gas + sigma_stars
        
        # Velocity dispersions
        sigma_vel_gas = 10.0  # km/s
        sigma_vel_stars = vvir[gal_idx] / 10.0
        sigma_vel_eff = (sigma_gas * sigma_vel_gas + sigma_stars * sigma_vel_stars) / (sigma_total + 1e-10)
        
        # Epicyclic frequency: κ ≈ Vcirc / r
        kappa = vvir[gal_idx] / (r_mid + 1e-10)
        
        # Toomre Q = (σ κ) / (π G Σ)
        # Convert Σ to kpc units
        sigma_total_kpc2 = sigma_total * 1e6
        Q = (sigma_vel_eff * kappa) / (np.pi * G * sigma_total_kpc2)
        
        # Find last bin with disk
        last_bin = np.where((disc_gas[gal_idx] > 1e-4) | (disc_stars[gal_idx] > 1e-4))[0]
        if len(last_bin) > 0:
            last_bin = min(last_bin.max() + 2, len(Q))
        else:
            last_bin = 10
        
        ax.plot(r_mid[:last_bin], Q[:last_bin], 'b-', linewidth=2, label='Q')
        ax.axhline(1, color='r', linestyle='--', linewidth=1.5, label='Q=1 (marginal)')
        ax.fill_between(r_mid[:last_bin], 0, 1, alpha=0.2, color='red', label='Unstable (Q<1)')
        
        ax.set_xlabel('Radius [kpc]')
        ax.set_ylabel('Toomre Q')
        ax.set_title(f'M*={stellar_mass[gal_idx]:.2f}')
        ax.set_ylim(0, min(5, np.nanmax(Q[:last_bin]) * 1.2))
        ax.legend(loc='upper right', fontsize=8)
        
        if last_bin > 0 and r_mid[last_bin-1] > 0:
            ax.set_xlim(0, r_mid[last_bin-1] * 1.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_toomre_q.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_prefix}_toomre_q.png")


def plot_instability_diagnostics(data, output_prefix):
    """Plot diagnostics for disk instabilities."""
    if not data['has_darkmode']:
        print("Skipping instability diagnostics - no DarkMode data")
        return
    
    if 'InstabilityBulgeMass' not in data:
        print("Skipping instability diagnostics - InstabilityBulgeMass not in output")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    stellar_mass = data['StellarMass']
    bulge_mass = data['BulgeMass']
    instability_bulge = data['InstabilityBulgeMass']
    
    # Plot 1: Instability bulge vs total stellar mass
    ax = axes[0, 0]
    valid = (stellar_mass > 0.01) & (instability_bulge > 0)
    if np.sum(valid) > 0:
        ax.scatter(np.log10(stellar_mass[valid]), np.log10(instability_bulge[valid]),
                   s=3, alpha=0.5, c='C0', edgecolors='none')
        ax.set_xlabel('log₁₀(M* / 10¹⁰ M☉)')
        ax.set_ylabel('log₁₀(Instability Bulge Mass / 10¹⁰ M☉)')
        ax.set_title(f'Instability Bulge (N={np.sum(valid)} galaxies)')
    
    # Plot 2: Fraction of bulge from instabilities
    ax = axes[0, 1]
    valid = (bulge_mass > 0) & (instability_bulge >= 0)
    if np.sum(valid) > 0:
        frac = instability_bulge[valid] / bulge_mass[valid]
        ax.hist(frac, bins=50, alpha=0.7, color='C2', edgecolor='black')
        ax.axvline(np.median(frac), color='r', linestyle='--', linewidth=2,
                   label=f'Median = {np.median(frac):.2f}')
        ax.set_xlabel('Instability Bulge / Total Bulge Mass')
        ax.set_ylabel('Count')
        ax.set_title('Fraction of Bulge from Instabilities')
        ax.legend()
    
    # Plot 3: Bulge-to-total ratio vs stellar mass
    ax = axes[1, 0]
    valid = stellar_mass > 0.01
    if np.sum(valid) > 0:
        bt_ratio = bulge_mass[valid] / stellar_mass[valid]
        mass_bins = np.logspace(np.log10(stellar_mass[valid].min()),
                                np.log10(stellar_mass[valid].max()), 20)
        mass_centers = 0.5 * (mass_bins[:-1] + mass_bins[1:])
        
        median_bt = []
        for i in range(len(mass_bins) - 1):
            in_bin = (stellar_mass[valid] >= mass_bins[i]) & (stellar_mass[valid] < mass_bins[i+1])
            if np.sum(in_bin) > 5:
                median_bt.append(np.median(bt_ratio[in_bin]))
            else:
                median_bt.append(np.nan)
        
        ax.scatter(np.log10(stellar_mass[valid]), bt_ratio, s=1, alpha=0.3, c='gray')
        ax.plot(np.log10(mass_centers), median_bt, 'r-', linewidth=2, label='Median')
        ax.set_xlabel('log₁₀(M* / 10¹⁰ M☉)')
        ax.set_ylabel('Bulge / Total Mass')
        ax.set_title('Bulge-to-Total Ratio')
        ax.set_ylim(0, 1)
        ax.legend()
    
    # Plot 4: Instability fraction vs stellar mass
    ax = axes[1, 1]
    valid = (stellar_mass > 0.01) & (bulge_mass > 0)
    if np.sum(valid) > 0:
        inst_frac = instability_bulge[valid] / bulge_mass[valid]
        mass_bins = np.logspace(np.log10(stellar_mass[valid].min()),
                                np.log10(stellar_mass[valid].max()), 20)
        mass_centers = 0.5 * (mass_bins[:-1] + mass_bins[1:])
        
        median_inst = []
        for i in range(len(mass_bins) - 1):
            in_bin = (stellar_mass[valid] >= mass_bins[i]) & (stellar_mass[valid] < mass_bins[i+1])
            if np.sum(in_bin) > 5:
                median_inst.append(np.median(inst_frac[in_bin]))
            else:
                median_inst.append(np.nan)
        
        ax.scatter(np.log10(stellar_mass[valid]), inst_frac, s=1, alpha=0.3, c='gray')
        ax.plot(np.log10(mass_centers), median_inst, 'r-', linewidth=2, label='Median')
        ax.set_xlabel('log₁₀(M* / 10¹⁰ M☉)')
        ax.set_ylabel('Instability Bulge / Total Bulge')
        ax.set_title('Instability Contribution to Bulge')
        ax.set_ylim(0, 1)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_instability_diagnostics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_prefix}_instability_diagnostics.png")


def plot_summary_statistics(data, output_prefix):
    """Print and plot summary statistics."""
    print("\n" + "="*60)
    print("DARKMODE DISK STATISTICS")
    print("="*60)
    
    ngal = len(data['StellarMass'])
    print(f"Total galaxies: {ngal}")
    
    if not data['has_darkmode']:
        print("No DarkMode arrays found!")
        return
    
    # Calculate statistics
    disc_gas_sum = np.sum(data['DiscGas'], axis=1)
    disc_stars_sum = np.sum(data['DiscStars'], axis=1)
    
    has_gas_disk = disc_gas_sum > 1e-6
    has_stellar_disk = disc_stars_sum > 1e-6
    
    print(f"Galaxies with gas disk: {np.sum(has_gas_disk)} ({100*np.mean(has_gas_disk):.1f}%)")
    print(f"Galaxies with stellar disk: {np.sum(has_stellar_disk)} ({100*np.mean(has_stellar_disk):.1f}%)")
    
    # Non-zero annuli statistics  
    n_bins = data['DiscGas'].shape[1]
    print(f"\nNumber of radial bins: {n_bins}")
    
    nonzero_gas = np.sum(data['DiscGas'] > 0, axis=1)
    nonzero_stars = np.sum(data['DiscStars'] > 0, axis=1)
    
    print(f"Mean populated gas bins per galaxy: {np.mean(nonzero_gas):.1f}")
    print(f"Mean populated stellar bins per galaxy: {np.mean(nonzero_stars):.1f}")
    
    # Mass conservation check
    valid_gas = data['ColdGas'] > 1e-6
    if np.any(valid_gas):
        rel_diff = (disc_gas_sum[valid_gas] - data['ColdGas'][valid_gas]) / data['ColdGas'][valid_gas]
        print(f"\nGas mass conservation:")
        print(f"  Mean relative diff: {np.mean(rel_diff):.2e}")
        print(f"  Max abs relative diff: {np.max(np.abs(rel_diff)):.2e}")
    
    valid_stars = data['StellarMass'] > 1e-6
    if np.any(valid_stars):
        rel_diff = (disc_stars_sum[valid_stars] - data['StellarMass'][valid_stars]) / data['StellarMass'][valid_stars]
        print(f"\nStellar mass conservation:")
        print(f"  Mean relative diff: {np.mean(rel_diff):.2e}")
        print(f"  Max abs relative diff: {np.max(np.abs(rel_diff)):.2e}")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Plot DarkMode disk structure')
    parser.add_argument('--input', default='output/millennium/model_0.hdf5',
                        help='Input HDF5 file')
    parser.add_argument('--output', default='plotting/darkmode',
                        help='Output prefix for plots')
    parser.add_argument('--snap', type=int, default=63,
                        help='Snapshot number to analyze')
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from {args.input}, snapshot {args.snap}...")
    data = load_darkmode_data(args.input, args.snap)
    
    plot_summary_statistics(data, args.output)
    plot_individual_profiles(data, args.output)
    plot_stacked_profiles(data, args.output)
    plot_angular_momentum(data, args.output)
    plot_consistency_check(data, args.output)
    
    # DarkSage physics diagnostic plots
    plot_sfr_profiles(data, args.output)
    plot_h2_hi_profiles(data, args.output)
    plot_toomre_q(data, args.output)
    plot_instability_diagnostics(data, args.output)
    
    print("\nAll plots completed!")


if __name__ == '__main__':
    main()

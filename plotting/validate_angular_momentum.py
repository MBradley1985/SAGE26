#!/usr/bin/env python3
"""
Validate Angular Momentum Outputs from SAGE26 DarkMode

This script checks that:
1. SpinGas and SpinStars are unit vectors (|J| = 1)
2. Gas and stellar disks are reasonably aligned
3. AM direction correlates with halo spin
4. Generates diagnostic plots
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path

# Use a nice style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.facecolor'] = '#1a1a2e'
plt.rcParams['axes.facecolor'] = '#16213e'
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['legend.facecolor'] = '#16213e'
plt.rcParams['legend.edgecolor'] = 'white'

def load_spin_data(hdf5_file, snap='Snap_63'):
    """Load spin vectors from HDF5 file."""
    with h5py.File(hdf5_file, 'r') as f:
        gals = f[snap]
        
        # Load all spin components
        data = {
            'SpinGas': np.column_stack([
                gals['SpinGasx'][:],
                gals['SpinGasy'][:],
                gals['SpinGasz'][:]
            ]),
            'SpinStars': np.column_stack([
                gals['SpinStarsx'][:],
                gals['SpinStarsy'][:],
                gals['SpinStarsz'][:]
            ]),
            'SpinHalo': np.column_stack([
                gals['Spinx'][:],
                gals['Spiny'][:],
                gals['Spinz'][:]
            ]),
            'StellarMass': gals['StellarMass'][:],
            'ColdGas': gals['ColdGas'][:],
            'Mvir': gals['Mvir'][:],
            'Type': gals['Type'][:],
        }
        
        # Load disc arrays if present
        if 'DiscGas' in gals:
            data['DiscGas'] = gals['DiscGas'][:]
            data['DiscStars'] = gals['DiscStars'][:]
            data['DiscRadii'] = gals['DiscRadii'][:]
            
    return data


def compute_spin_magnitudes(spin_vectors):
    """Compute magnitude of spin vectors."""
    return np.sqrt(np.sum(spin_vectors**2, axis=1))


def compute_alignment_angle(spin1, spin2):
    """Compute alignment angle between two spin vectors (in degrees)."""
    # Handle zero vectors
    mag1 = np.sqrt(np.sum(spin1**2, axis=1))
    mag2 = np.sqrt(np.sum(spin2**2, axis=1))
    
    valid = (mag1 > 0) & (mag2 > 0)
    
    # Normalized dot product
    cos_angle = np.zeros(len(spin1))
    cos_angle[valid] = np.sum(spin1[valid] * spin2[valid], axis=1) / (mag1[valid] * mag2[valid])
    cos_angle = np.clip(cos_angle, -1, 1)  # Handle numerical errors
    
    angles = np.degrees(np.arccos(cos_angle))
    angles[~valid] = np.nan
    
    return angles


def print_statistics(data):
    """Print validation statistics."""
    print("=" * 60)
    print("ANGULAR MOMENTUM VALIDATION STATISTICS")
    print("=" * 60)
    
    # 1. Check that spins are unit vectors
    gas_mag = compute_spin_magnitudes(data['SpinGas'])
    stars_mag = compute_spin_magnitudes(data['SpinStars'])
    halo_mag = compute_spin_magnitudes(data['SpinHalo'])
    
    print("\n1. SPIN VECTOR MAGNITUDES (should be ~1 for unit vectors)")
    print("-" * 40)
    
    # Filter galaxies with gas/stars
    has_gas = data['ColdGas'] > 0
    has_stars = data['StellarMass'] > 0
    
    print(f"   SpinGas (galaxies with gas, N={has_gas.sum()}):")
    print(f"      Mean:   {np.nanmean(gas_mag[has_gas]):.4f}")
    print(f"      Median: {np.nanmedian(gas_mag[has_gas]):.4f}")
    print(f"      Std:    {np.nanstd(gas_mag[has_gas]):.4f}")
    print(f"      Min:    {np.nanmin(gas_mag[has_gas]):.4f}")
    print(f"      Max:    {np.nanmax(gas_mag[has_gas]):.4f}")
    
    print(f"\n   SpinStars (galaxies with stars, N={has_stars.sum()}):")
    print(f"      Mean:   {np.nanmean(stars_mag[has_stars]):.4f}")
    print(f"      Median: {np.nanmedian(stars_mag[has_stars]):.4f}")
    print(f"      Std:    {np.nanstd(stars_mag[has_stars]):.4f}")
    print(f"      Min:    {np.nanmin(stars_mag[has_stars]):.4f}")
    print(f"      Max:    {np.nanmax(stars_mag[has_stars]):.4f}")
    
    print(f"\n   SpinHalo (all galaxies):")
    print(f"      Mean:   {np.nanmean(halo_mag):.4f}")
    print(f"      Median: {np.nanmedian(halo_mag):.4f}")
    
    # 2. Gas-Stellar alignment
    print("\n2. GAS-STELLAR DISK ALIGNMENT")
    print("-" * 40)
    
    both_valid = has_gas & has_stars
    gas_star_angle = compute_alignment_angle(data['SpinGas'], data['SpinStars'])
    
    print(f"   Galaxies with both gas and stars: {both_valid.sum()}")
    print(f"   Alignment angle (degrees):")
    print(f"      Mean:   {np.nanmean(gas_star_angle[both_valid]):.1f}°")
    print(f"      Median: {np.nanmedian(gas_star_angle[both_valid]):.1f}°")
    print(f"      Std:    {np.nanstd(gas_star_angle[both_valid]):.1f}°")
    
    aligned = gas_star_angle[both_valid] < 30
    print(f"      Fraction aligned (<30°): {aligned.sum()/len(aligned)*100:.1f}%")
    
    # 3. Disk-Halo alignment
    print("\n3. DISK-HALO ALIGNMENT")
    print("-" * 40)
    
    gas_halo_angle = compute_alignment_angle(data['SpinGas'], data['SpinHalo'])
    star_halo_angle = compute_alignment_angle(data['SpinStars'], data['SpinHalo'])
    
    print(f"   Gas disk - Halo:")
    print(f"      Mean angle: {np.nanmean(gas_halo_angle[has_gas]):.1f}°")
    print(f"      Median:     {np.nanmedian(gas_halo_angle[has_gas]):.1f}°")
    
    print(f"   Stellar disk - Halo:")
    print(f"      Mean angle: {np.nanmean(star_halo_angle[has_stars]):.1f}°")
    print(f"      Median:     {np.nanmedian(star_halo_angle[has_stars]):.1f}°")
    
    # 4. Central vs Satellite comparison
    print("\n4. CENTRAL vs SATELLITE GALAXIES")
    print("-" * 40)
    
    centrals = data['Type'] == 0
    satellites = data['Type'] > 0
    
    print(f"   Central galaxies: {centrals.sum()}")
    print(f"   Satellite galaxies: {satellites.sum()}")
    
    central_gas_halo = gas_halo_angle[centrals & has_gas]
    sat_gas_halo = gas_halo_angle[satellites & has_gas]
    
    print(f"\n   Gas-Halo alignment (centrals):  {np.nanmedian(central_gas_halo):.1f}° median")
    print(f"   Gas-Halo alignment (satellites): {np.nanmedian(sat_gas_halo):.1f}° median")
    
    return {
        'gas_mag': gas_mag,
        'stars_mag': stars_mag,
        'gas_star_angle': gas_star_angle,
        'gas_halo_angle': gas_halo_angle,
        'star_halo_angle': star_halo_angle,
        'has_gas': has_gas,
        'has_stars': has_stars,
        'centrals': centrals,
    }


def create_validation_plots(data, stats, output_dir):
    """Create validation plots for angular momentum."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Angular Momentum Validation (z=0)', fontsize=16, color='white')
    
    # 1. Spin magnitude distributions
    ax = axes[0, 0]
    bins = np.linspace(0, 1.5, 50)
    ax.hist(stats['gas_mag'][stats['has_gas']], bins=bins, alpha=0.7, 
            label='SpinGas', color='#00d4ff', density=True)
    ax.hist(stats['stars_mag'][stats['has_stars']], bins=bins, alpha=0.7, 
            label='SpinStars', color='#ff6b6b', density=True)
    ax.axvline(1.0, color='yellow', ls='--', lw=2, label='Unit vector')
    ax.set_xlabel('Spin Magnitude')
    ax.set_ylabel('Density')
    ax.set_title('Spin Vector Magnitudes')
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1.5)
    
    # 2. Gas-Star alignment distribution
    ax = axes[0, 1]
    valid = stats['has_gas'] & stats['has_stars']
    ax.hist(stats['gas_star_angle'][valid], bins=30, alpha=0.8, 
            color='#9d4edd', edgecolor='white')
    ax.axvline(np.nanmedian(stats['gas_star_angle'][valid]), color='yellow', 
               ls='--', lw=2, label=f"Median: {np.nanmedian(stats['gas_star_angle'][valid]):.1f}°")
    ax.set_xlabel('Gas-Stellar Alignment Angle (degrees)')
    ax.set_ylabel('Count')
    ax.set_title('Gas-Stellar Disk Alignment')
    ax.legend(fontsize=9)
    
    # 3. Gas-Halo alignment
    ax = axes[0, 2]
    ax.hist(stats['gas_halo_angle'][stats['has_gas']], bins=30, alpha=0.8, 
            color='#00d4ff', edgecolor='white')
    ax.axvline(np.nanmedian(stats['gas_halo_angle'][stats['has_gas']]), color='yellow', 
               ls='--', lw=2, label=f"Median: {np.nanmedian(stats['gas_halo_angle'][stats['has_gas']]):.1f}°")
    ax.set_xlabel('Gas-Halo Alignment Angle (degrees)')
    ax.set_ylabel('Count')
    ax.set_title('Gas Disk - Halo Spin Alignment')
    ax.legend(fontsize=9)
    
    # 4. Alignment vs Stellar Mass
    ax = axes[1, 0]
    mass_bins = np.logspace(8, 12, 20)
    mass_centers = np.sqrt(mass_bins[:-1] * mass_bins[1:])
    
    # Gas-star alignment vs mass
    valid = stats['has_gas'] & stats['has_stars'] & (data['StellarMass'] > 0)
    median_angles = []
    for i in range(len(mass_bins)-1):
        mask = valid & (data['StellarMass']*1e10 >= mass_bins[i]) & (data['StellarMass']*1e10 < mass_bins[i+1])
        if mask.sum() > 10:
            median_angles.append(np.nanmedian(stats['gas_star_angle'][mask]))
        else:
            median_angles.append(np.nan)
    
    ax.plot(mass_centers, median_angles, 'o-', color='#9d4edd', lw=2, markersize=8)
    ax.set_xscale('log')
    ax.set_xlabel('Stellar Mass [M$_\\odot$]')
    ax.set_ylabel('Median Gas-Star Alignment (°)')
    ax.set_title('Alignment vs Stellar Mass')
    ax.set_ylim(0, 90)
    
    # 5. Central vs Satellite alignment
    ax = axes[1, 1]
    
    central_angles = stats['gas_halo_angle'][stats['centrals'] & stats['has_gas']]
    sat_angles = stats['gas_halo_angle'][~stats['centrals'] & stats['has_gas']]
    
    bins = np.linspace(0, 180, 30)
    ax.hist(central_angles, bins=bins, alpha=0.7, label='Centrals', 
            color='#00d4ff', density=True)
    ax.hist(sat_angles, bins=bins, alpha=0.7, label='Satellites', 
            color='#ff6b6b', density=True)
    ax.set_xlabel('Gas-Halo Alignment (degrees)')
    ax.set_ylabel('Density')
    ax.set_title('Central vs Satellite Alignment')
    ax.legend(fontsize=9)
    
    # 6. 3D Spin direction distribution (Mollweide projection of SpinGas)
    ax = axes[1, 2]
    
    # Convert to spherical coordinates
    valid_gas = stats['has_gas'] & (stats['gas_mag'] > 0)
    spin_gas = data['SpinGas'][valid_gas]
    
    # Subsample for clarity
    if len(spin_gas) > 5000:
        idx = np.random.choice(len(spin_gas), 5000, replace=False)
        spin_gas = spin_gas[idx]
    
    # Convert to theta, phi
    r = np.sqrt(np.sum(spin_gas**2, axis=1))
    theta = np.arccos(spin_gas[:, 2] / r)  # polar angle from z-axis
    phi = np.arctan2(spin_gas[:, 1], spin_gas[:, 0])  # azimuthal angle
    
    # Aitoff-style scatter
    ax.scatter(phi * 180/np.pi, 90 - theta * 180/np.pi, s=1, alpha=0.3, c='#00d4ff')
    ax.set_xlabel('Azimuthal angle φ (degrees)')
    ax.set_ylabel('Polar angle (90-θ) (degrees)')
    ax.set_title('Gas Spin Direction Distribution')
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.axhline(0, color='white', alpha=0.3, ls='--')
    ax.axvline(0, color='white', alpha=0.3, ls='--')
    
    plt.tight_layout()
    
    output_path = output_dir / 'AngularMomentum_Validation.pdf'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print(f"\nPlot saved to: {output_path}")
    plt.close()
    
    # Additional plot: Specific angular momentum
    create_specific_am_plot(data, output_dir)
    
    return output_path


def create_specific_am_plot(data, output_dir):
    """Create plot of specific angular momentum vs mass."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Compute specific angular momentum from disc profiles if available
    if 'DiscGas' not in data:
        print("No disc arrays available for specific AM plot")
        return
    
    disc_gas = data['DiscGas']  # Shape: (Ngal, Nbins)
    disc_stars = data['DiscStars']
    disc_radii = data['DiscRadii']  # Shape: (Ngal, Nbins+1) - bin edges
    
    stellar_mass = data['StellarMass'] * 1e10  # Convert to Msun
    cold_gas = data['ColdGas'] * 1e10
    
    # Compute mass-weighted mean radius (proxy for j ~ r*v ~ r)
    # For exponential disc with flat rotation curve: j ~ 2 * R_d
    
    has_gas = cold_gas > 1e7
    has_stars = stellar_mass > 1e8
    
    # Compute bin centers from bin edges
    # disc_radii is (Ngal, Nbins+1), disc_gas is (Ngal, Nbins)
    n_bins = disc_gas.shape[1]
    
    # Gas specific AM proxy (mean radius)
    gas_total = np.sum(disc_gas, axis=1)
    gas_mean_r = np.zeros(len(disc_gas))
    for i in range(len(disc_gas)):
        if gas_total[i] > 0:
            # Compute bin centers for this galaxy
            bin_centers = 0.5 * (disc_radii[i, :n_bins] + disc_radii[i, 1:n_bins+1])
            # Mass-weighted mean radius
            gas_mean_r[i] = np.sum(disc_gas[i] * bin_centers) / gas_total[i]
    
    # Stellar specific AM proxy
    star_total = np.sum(disc_stars, axis=1)
    star_mean_r = np.zeros(len(disc_stars))
    for i in range(len(disc_stars)):
        if star_total[i] > 0:
            bin_centers = 0.5 * (disc_radii[i, :n_bins] + disc_radii[i, 1:n_bins+1])
            star_mean_r[i] = np.sum(disc_stars[i] * bin_centers) / star_total[i]
    
    # Plot gas j vs Mgas
    ax = axes[0]
    mask = has_gas & (gas_mean_r > 0)
    
    # 2D histogram
    h = ax.hist2d(np.log10(cold_gas[mask]), np.log10(gas_mean_r[mask] * 1000),  # Convert to kpc
                  bins=50, cmap='magma', norm=plt.matplotlib.colors.LogNorm())
    plt.colorbar(h[3], ax=ax, label='Count')
    
    # Add j ∝ M^0.6 scaling (Fall relation)
    mass_line = np.logspace(7, 12, 100)
    j_line = 0.1 * (mass_line / 1e10)**0.6  # Approximate Fall relation
    ax.plot(np.log10(mass_line), np.log10(j_line), 'w--', lw=2, label='j ∝ M$^{0.6}$')
    
    ax.set_xlabel('log$_{10}$(Cold Gas Mass / M$_\\odot$)')
    ax.set_ylabel('log$_{10}$(Mean Radius / kpc)')
    ax.set_title('Gas Disc Size-Mass Relation')
    ax.legend(loc='lower right', fontsize=9)
    
    # Plot stellar j vs Mstar
    ax = axes[1]
    mask = has_stars & (star_mean_r > 0)
    
    h = ax.hist2d(np.log10(stellar_mass[mask]), np.log10(star_mean_r[mask] * 1000),
                  bins=50, cmap='magma', norm=plt.matplotlib.colors.LogNorm())
    plt.colorbar(h[3], ax=ax, label='Count')
    
    ax.plot(np.log10(mass_line), np.log10(j_line), 'w--', lw=2, label='j ∝ M$^{0.6}$')
    
    ax.set_xlabel('log$_{10}$(Stellar Mass / M$_\\odot$)')
    ax.set_ylabel('log$_{10}$(Mean Radius / kpc)')
    ax.set_title('Stellar Disc Size-Mass Relation')
    ax.legend(loc='lower right', fontsize=9)
    
    plt.tight_layout()
    
    output_path = output_dir / 'SpecificAM_Validation.pdf'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print(f"Plot saved to: {output_path}")
    plt.close()


def main():
    """Main validation function."""
    
    # Find output file
    workspace = Path('/Users/mbradley/Documents/PhD/SAGE26')
    hdf5_file = workspace / 'output/millennium/model_0.hdf5'
    output_dir = workspace / 'output/millennium/plots'
    output_dir.mkdir(exist_ok=True)
    
    if not hdf5_file.exists():
        print(f"ERROR: HDF5 file not found: {hdf5_file}")
        return
    
    print(f"Loading data from: {hdf5_file}")
    print(f"Snapshot: Snap_63 (z=0)")
    
    # Load data
    data = load_spin_data(hdf5_file, snap='Snap_63')
    print(f"Loaded {len(data['StellarMass'])} galaxies")
    
    # Print statistics
    stats = print_statistics(data)
    
    # Create plots
    print("\nGenerating validation plots...")
    create_validation_plots(data, stats, output_dir)
    
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()

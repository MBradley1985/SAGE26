#!/usr/bin/env python3
"""
Validate Angular Momentum Outputs from SAGE26 DarkMode

This script checks that:
1. SpinGas and SpinStars are unit vectors (|J| = 1)
2. Gas and stellar disks are reasonably aligned
3. AM direction correlates with halo spin
4. Spin evolution through cosmic time
5. Generates diagnostic plots
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import LogNorm
import warnings
warnings.filterwarnings('ignore')

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
    
    # Additional analysis: multi-redshift spin evolution
    print("\nAnalyzing spin evolution across redshifts...")
    analyze_spin_evolution(hdf5_file, output_dir)
    
    # Detailed spin alignment analysis
    print("\nGenerating detailed alignment analysis...")
    create_detailed_alignment_plots(data, stats, output_dir)
    
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)


def load_multi_snapshot_spins(hdf5_file):
    """Load spin data from multiple snapshots for evolution analysis."""
    
    # Millennium snapshots with approximate redshifts
    snap_info = [
        ('Snap_63', 0.0),
        ('Snap_60', 0.25),
        ('Snap_54', 0.62),
        ('Snap_50', 1.0),
        ('Snap_45', 1.5),
        ('Snap_41', 2.0),
        ('Snap_36', 3.0),
        ('Snap_30', 4.5),
    ]
    
    results = []
    
    with h5py.File(hdf5_file, 'r') as f:
        for snap_name, z in snap_info:
            if snap_name not in f:
                continue
            
            gals = f[snap_name]
            
            # Skip if no galaxies
            if 'SpinGasx' not in gals or len(gals['SpinGasx']) == 0:
                continue
            
            spin_gas = np.column_stack([
                gals['SpinGasx'][:],
                gals['SpinGasy'][:],
                gals['SpinGasz'][:]
            ])
            spin_stars = np.column_stack([
                gals['SpinStarsx'][:],
                gals['SpinStarsy'][:],
                gals['SpinStarsz'][:]
            ])
            spin_halo = np.column_stack([
                gals['Spinx'][:],
                gals['Spiny'][:],
                gals['Spinz'][:]
            ])
            
            stellar_mass = gals['StellarMass'][:]
            cold_gas = gals['ColdGas'][:]
            galaxy_type = gals['Type'][:]
            
            # Compute alignment
            gas_star_angle = compute_alignment_angle(spin_gas, spin_stars)
            gas_halo_angle = compute_alignment_angle(spin_gas, spin_halo)
            
            has_gas = cold_gas > 0
            has_stars = stellar_mass > 0
            both = has_gas & has_stars
            
            results.append({
                'snap': snap_name,
                'z': z,
                'ngal': len(stellar_mass),
                'ngal_with_both': both.sum(),
                'mean_gas_star_angle': np.nanmean(gas_star_angle[both]) if both.sum() > 0 else np.nan,
                'median_gas_star_angle': np.nanmedian(gas_star_angle[both]) if both.sum() > 0 else np.nan,
                'std_gas_star_angle': np.nanstd(gas_star_angle[both]) if both.sum() > 0 else np.nan,
                'mean_gas_halo_angle': np.nanmean(gas_halo_angle[has_gas]) if has_gas.sum() > 0 else np.nan,
                'median_gas_halo_angle': np.nanmedian(gas_halo_angle[has_gas]) if has_gas.sum() > 0 else np.nan,
                'fraction_aligned_30': (gas_star_angle[both] < 30).sum() / max(1, both.sum()) if both.sum() > 0 else np.nan,
                'fraction_misaligned_60': (gas_star_angle[both] > 60).sum() / max(1, both.sum()) if both.sum() > 0 else np.nan,
            })
    
    return results


def analyze_spin_evolution(hdf5_file, output_dir):
    """Analyze how spin alignment evolves with redshift."""
    
    results = load_multi_snapshot_spins(hdf5_file)
    
    if len(results) == 0:
        print("  No multi-snapshot data available")
        return
    
    # Print evolution table
    print("\n" + "=" * 80)
    print("SPIN EVOLUTION WITH REDSHIFT")
    print("=" * 80)
    print(f"{'z':>6} {'N_gal':>8} {'Mean θ(gas-star)':>18} {'Std θ':>10} {'%<30°':>10} {'%>60°':>10}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['z']:6.2f} {r['ngal_with_both']:8d} {r['mean_gas_star_angle']:18.2f} {r['std_gas_star_angle']:10.2f} "
              f"{r['fraction_aligned_30']*100:10.1f} {r['fraction_misaligned_60']*100:10.1f}")
    
    # Create evolution plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Angular Momentum Evolution with Redshift', fontsize=14, color='white')
    
    zs = [r['z'] for r in results]
    
    # Plot 1: Gas-Star alignment evolution
    ax = axes[0]
    mean_angles = [r['mean_gas_star_angle'] for r in results]
    std_angles = [r['std_gas_star_angle'] for r in results]
    
    ax.errorbar(zs, mean_angles, yerr=std_angles, fmt='o-', color='#9d4edd', 
                capsize=5, markersize=8, label='Mean ± σ')
    ax.set_xlabel('Redshift')
    ax.set_ylabel('Gas-Stellar Alignment (°)')
    ax.set_title('Gas-Stellar Misalignment')
    ax.set_xlim(-0.2, max(zs) + 0.5)
    ax.set_ylim(0, 30)
    ax.legend()
    ax.invert_xaxis()
    
    # Plot 2: Fraction aligned/misaligned
    ax = axes[1]
    frac_aligned = [r['fraction_aligned_30'] * 100 for r in results]
    frac_misaligned = [r['fraction_misaligned_60'] * 100 for r in results]
    
    ax.plot(zs, frac_aligned, 'o-', color='#00ff88', markersize=8, label='< 30° (aligned)')
    ax.plot(zs, frac_misaligned, 's-', color='#ff6b6b', markersize=8, label='> 60° (misaligned)')
    ax.set_xlabel('Redshift')
    ax.set_ylabel('Fraction (%)')
    ax.set_title('Alignment Fractions')
    ax.set_xlim(-0.2, max(zs) + 0.5)
    ax.set_ylim(0, 105)
    ax.legend()
    ax.invert_xaxis()
    
    # Plot 3: Gas-Halo alignment
    ax = axes[2]
    gas_halo = [r['mean_gas_halo_angle'] for r in results]
    ax.plot(zs, gas_halo, 'o-', color='#00d4ff', markersize=8)
    ax.axhline(90, color='white', ls='--', alpha=0.5, label='Random (90°)')
    ax.set_xlabel('Redshift')
    ax.set_ylabel('Gas-Halo Alignment (°)')
    ax.set_title('Gas Disk - Halo Spin')
    ax.set_xlim(-0.2, max(zs) + 0.5)
    ax.set_ylim(0, 120)
    ax.legend()
    ax.invert_xaxis()
    
    plt.tight_layout()
    
    output_path = output_dir / 'SpinEvolution_Redshift.pdf'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print(f"\nPlot saved to: {output_path}")
    plt.close()


def create_detailed_alignment_plots(data, stats, output_dir):
    """Create detailed plots showing spin alignment depends on galaxy properties."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Spin Alignment Dependencies (z=0)', fontsize=14, color='white')
    
    stellar_mass = data['StellarMass'] * 1e10
    cold_gas = data['ColdGas'] * 1e10
    mvir = data['Mvir'] * 1e10
    gtype = data['Type']
    
    has_gas = stats['has_gas']
    has_stars = stats['has_stars']
    both = has_gas & has_stars
    
    gas_star_angle = stats['gas_star_angle']
    gas_halo_angle = stats['gas_halo_angle']
    
    # 1. Gas-Star alignment vs Stellar Mass (2D histogram)
    ax = axes[0, 0]
    mask = both & (stellar_mass > 1e7)
    if mask.sum() > 10:
        h = ax.hist2d(np.log10(stellar_mass[mask]), gas_star_angle[mask],
                      bins=[30, np.linspace(0, 45, 30)], cmap='magma', 
                      norm=LogNorm(vmin=1))
        plt.colorbar(h[3], ax=ax, label='Count')
    ax.set_xlabel('log$_{10}$(M$_*$ / M$_\\odot$)')
    ax.set_ylabel('Gas-Stellar Angle (°)')
    ax.set_title('Misalignment vs Stellar Mass')
    
    # 2. Gas-Star alignment vs Gas Fraction
    ax = axes[0, 1]
    gas_frac = cold_gas / (stellar_mass + cold_gas + 1e-10)
    mask = both & (gas_frac > 0.01) & (gas_frac < 0.99)
    if mask.sum() > 10:
        h = ax.hist2d(gas_frac[mask], gas_star_angle[mask],
                      bins=[np.linspace(0, 1, 30), np.linspace(0, 45, 30)], 
                      cmap='magma', norm=LogNorm(vmin=1))
        plt.colorbar(h[3], ax=ax, label='Count')
    ax.set_xlabel('Gas Fraction')
    ax.set_ylabel('Gas-Stellar Angle (°)')
    ax.set_title('Misalignment vs Gas Fraction')
    
    # 3. Gas-Star alignment vs Halo Mass
    ax = axes[0, 2]
    mask = both & (mvir > 1e9)
    if mask.sum() > 10:
        h = ax.hist2d(np.log10(mvir[mask]), gas_star_angle[mask],
                      bins=[30, np.linspace(0, 45, 30)], cmap='magma', 
                      norm=LogNorm(vmin=1))
        plt.colorbar(h[3], ax=ax, label='Count')
    ax.set_xlabel('log$_{10}$(M$_{vir}$ / M$_\\odot$)')
    ax.set_ylabel('Gas-Stellar Angle (°)')
    ax.set_title('Misalignment vs Halo Mass')
    
    # 4. Central vs Satellite comparison (violin plot style)
    ax = axes[1, 0]
    central_angles = gas_star_angle[both & (gtype == 0)]
    sat_angles = gas_star_angle[both & (gtype > 0)]
    
    # Box plots
    bp = ax.boxplot([central_angles[~np.isnan(central_angles)], 
                     sat_angles[~np.isnan(sat_angles)]], 
                    labels=['Centrals', 'Satellites'],
                    patch_artist=True)
    bp['boxes'][0].set_facecolor('#00d4ff')
    bp['boxes'][1].set_facecolor('#ff6b6b')
    for element in ['whiskers', 'caps', 'medians']:
        plt.setp(bp[element], color='white')
    ax.set_ylabel('Gas-Stellar Angle (°)')
    ax.set_title('Central vs Satellite Alignment')
    
    # Add text with median values
    med_c = np.nanmedian(central_angles)
    med_s = np.nanmedian(sat_angles)
    ax.text(0.05, 0.95, f'Central median: {med_c:.1f}°\nSatellite median: {med_s:.1f}°',
            transform=ax.transAxes, va='top', fontsize=10, color='white')
    
    # 5. Spin direction scatter (x-y projection)
    ax = axes[1, 1]
    spin_gas = data['SpinGas']
    valid = has_gas & (stats['gas_mag'] > 0.5)
    
    if valid.sum() > 100:
        # Subsample for clarity
        idx = np.random.choice(np.where(valid)[0], min(3000, valid.sum()), replace=False)
        
        # Color by stellar mass
        colors = np.log10(stellar_mass[idx] + 1e6)
        sc = ax.scatter(spin_gas[idx, 0], spin_gas[idx, 1], 
                       c=colors, cmap='viridis', s=2, alpha=0.5)
        plt.colorbar(sc, ax=ax, label='log M$_*$')
        ax.set_xlabel('SpinGas$_x$')
        ax.set_ylabel('SpinGas$_y$')
        ax.set_title('Gas Spin Direction (x-y plane)')
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        
        # Add unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), 'w--', alpha=0.3)
    
    # 6. Cumulative distribution of misalignment
    ax = axes[1, 2]
    
    mask_c = both & (gtype == 0)
    mask_s = both & (gtype > 0)
    
    angles_c = np.sort(gas_star_angle[mask_c][~np.isnan(gas_star_angle[mask_c])])
    angles_s = np.sort(gas_star_angle[mask_s][~np.isnan(gas_star_angle[mask_s])])
    
    cdf_c = np.arange(1, len(angles_c) + 1) / len(angles_c)
    cdf_s = np.arange(1, len(angles_s) + 1) / len(angles_s)
    
    ax.plot(angles_c, cdf_c, '-', color='#00d4ff', lw=2, label='Centrals')
    ax.plot(angles_s, cdf_s, '-', color='#ff6b6b', lw=2, label='Satellites')
    
    # Reference lines
    ax.axvline(10, color='yellow', ls='--', alpha=0.5, label='10°')
    ax.axvline(30, color='orange', ls='--', alpha=0.5, label='30°')
    
    ax.set_xlabel('Gas-Stellar Alignment (°)')
    ax.set_ylabel('Cumulative Fraction')
    ax.set_title('CDF of Misalignment')
    ax.set_xlim(0, 90)
    ax.legend(loc='lower right', fontsize=9)
    
    plt.tight_layout()
    
    output_path = output_dir / 'SpinAlignment_Detailed.pdf'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print(f"Plot saved to: {output_path}")
    plt.close()
    
    # Create additional summary statistics
    print_detailed_stats(data, stats)


def print_detailed_stats(data, stats):
    """Print detailed statistics about spin alignment."""
    
    stellar_mass = data['StellarMass'] * 1e10
    cold_gas = data['ColdGas'] * 1e10
    gtype = data['Type']
    
    both = stats['has_gas'] & stats['has_stars']
    gas_star_angle = stats['gas_star_angle']
    
    print("\n" + "=" * 70)
    print("DETAILED ALIGNMENT STATISTICS")
    print("=" * 70)
    
    # Mass bins
    mass_bins = [
        (1e7, 1e8, "Dwarf (10^7-10^8)"),
        (1e8, 1e9, "Small (10^8-10^9)"),
        (1e9, 1e10, "MW-like (10^9-10^10)"),
        (1e10, 1e11, "Massive (10^10-10^11)"),
        (1e11, 1e13, "BCG (>10^11)"),
    ]
    
    print("\n5. ALIGNMENT BY STELLAR MASS BIN")
    print("-" * 70)
    print(f"{'Mass Range':25} {'N':>8} {'Median θ':>12} {'Mean θ':>12} {'%<10°':>10}")
    
    for m_lo, m_hi, label in mass_bins:
        mask = both & (stellar_mass >= m_lo) & (stellar_mass < m_hi)
        if mask.sum() > 5:
            angles = gas_star_angle[mask]
            valid_angles = angles[~np.isnan(angles)]
            med = np.median(valid_angles)
            mean = np.mean(valid_angles)
            frac_10 = (valid_angles < 10).sum() / len(valid_angles) * 100
            print(f"{label:25} {mask.sum():8d} {med:12.2f}° {mean:12.2f}° {frac_10:10.1f}%")
    
    # Type breakdown
    print("\n6. ALIGNMENT BY GALAXY TYPE")
    print("-" * 70)
    
    for t, label in [(0, "Central"), (1, "Type 1 Satellite"), (2, "Type 2 Satellite")]:
        mask = both & (gtype == t)
        if mask.sum() > 5:
            angles = gas_star_angle[mask]
            valid_angles = angles[~np.isnan(angles)]
            med = np.median(valid_angles)
            std = np.std(valid_angles)
            frac_30 = (valid_angles < 30).sum() / len(valid_angles) * 100
            print(f"{label:20} N={mask.sum():6d}  Median={med:5.1f}° ± {std:5.1f}°  Aligned(<30°)={frac_30:5.1f}%")
    
    # Gas fraction bins
    print("\n7. ALIGNMENT BY GAS FRACTION")
    print("-" * 70)
    
    gas_frac = cold_gas / (stellar_mass + cold_gas + 1e-10)
    frac_bins = [
        (0.0, 0.1, "Gas poor (<10%)"),
        (0.1, 0.3, "Intermediate (10-30%)"),
        (0.3, 0.5, "Gas rich (30-50%)"),
        (0.5, 1.0, "Very gas rich (>50%)"),
    ]
    
    for f_lo, f_hi, label in frac_bins:
        mask = both & (gas_frac >= f_lo) & (gas_frac < f_hi)
        if mask.sum() > 5:
            angles = gas_star_angle[mask]
            valid_angles = angles[~np.isnan(angles)]
            med = np.median(valid_angles)
            frac_10 = (valid_angles < 10).sum() / len(valid_angles) * 100
            print(f"{label:25} N={mask.sum():6d}  Median={med:5.1f}°  %<10°={frac_10:5.1f}%")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Comprehensive DarkMode + Dust Analysis
Generates:
1. Radial profiles (gas, stars, metals, dust)
2. Dust-to-gas ratios vs metallicity
3. Stellar mass function comparison
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys

# Cosmology
h = 0.73
SOLAR_Z = 0.02  # Solar metallicity

def load_darkmode_data(filename, snap=63):
    """Load DarkMode galaxy data"""
    with h5py.File(filename, 'r') as f:
        snap_group = f[f'Snap_{snap}']
        
        data = {
            'StellarMass': snap_group['StellarMass'][:],
            'ColdGas': snap_group['ColdGas'][:],
            'ColdDust': snap_group['ColdDust'][:],
            'MetalsColdGas': snap_group['MetalsColdGas'][:],
            'MetalsStellarMass': snap_group['MetalsStellarMass'][:],
            'DiskRadius': snap_group['DiskRadius'][:],
            'Type': snap_group['Type'][:],
        }
        
        # DarkMode arrays
        if 'DiscGas' in snap_group:
            data['DiscGas'] = snap_group['DiscGas'][:, :]
            data['DiscStars'] = snap_group['DiscStars'][:, :]
            data['DiscGasMetals'] = snap_group['DiscGasMetals'][:, :]
            data['DiscStarsMetals'] = snap_group['DiscStarsMetals'][:, :]
            data['DiscDust'] = snap_group['DiscDust'][:, :]
            data['DiscH2'] = snap_group['DiscH2'][:, :]
            data['DiscHI'] = snap_group['DiscHI'][:, :]
            data['DiscRadii'] = snap_group['DiscRadii'][:, :]
            data['has_darkmode'] = True
        else:
            data['has_darkmode'] = False
            
    return data

def plot_radial_profiles(data, output='plotting/darkmode_radial_profiles.png'):
    """Plot stacked radial profiles for different mass bins"""
    
    if not data['has_darkmode']:
        print("❌ No DarkMode data - skipping radial profiles")
        return
    
    print("\n" + "="*70)
    print("1. RADIAL PROFILES")
    print("="*70)
    
    # Select central galaxies with significant gas
    mask = (data['Type'] == 0) & (data['ColdGas'] > 0.1) & (data['StellarMass'] > 1.0)
    n_gals = np.sum(mask)
    print(f"Analyzing {n_gals} central galaxies with M* > 10^10 M☉/h and gas > 10^9 M☉/h")
    
    # Define mass bins
    stellar_mass = data['StellarMass'][mask]
    mass_bins = [
        (1.0, 3.0, '10^{10} - 3×10^{10}'),
        (3.0, 10.0, '3×10^{10} - 10^{11}'),
        (10.0, 100.0, '10^{11} - 10^{12}'),
    ]
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for mass_idx, (m_min, m_max, label) in enumerate(mass_bins):
        mass_mask = (stellar_mass >= m_min) & (stellar_mass < m_max)
        n_in_bin = np.sum(mass_mask)
        print(f"  Mass bin {label}: {n_in_bin} galaxies")
        
        if n_in_bin < 5:
            continue
        
        # Get indices in original array
        full_indices = np.where(mask)[0][mass_mask]
        
        # Collect profiles
        all_radii = []
        all_gas = []
        all_stars = []
        all_metals_gas = []
        all_metals_stars = []
        all_dust = []
        all_h2 = []
        all_hi = []
        
        for idx in full_indices:
            radii = data['DiscRadii'][idx, :] * 1000 / h  # kpc physical
            r_mid = 0.5 * (radii[:-1] + radii[1:])  # Midpoint of bins
            
            # Only use bins within 50 kpc
            valid = (r_mid > 0) & (r_mid < 50)
            if np.sum(valid) < 3:
                continue
                
            areas = np.pi * (radii[1:]**2 - radii[:-1]**2)  # kpc²
            
            # Surface densities (M☉/pc²)
            gas_surf = (data['DiscGas'][idx, :] * 1e10 / h) / (areas * 1e6)
            stars_surf = (data['DiscStars'][idx, :] * 1e10 / h) / (areas * 1e6)
            dust_surf = (data['DiscDust'][idx, :] * 1e10 / h) / (areas * 1e6)
            h2_surf = (data['DiscH2'][idx, :] * 1e10 / h) / (areas * 1e6)
            hi_surf = (data['DiscHI'][idx, :] * 1e10 / h) / (areas * 1e6)
            
            # Metallicities (mass fraction)
            z_gas = np.where(data['DiscGas'][idx, :] > 1e-6,
                            data['DiscGasMetals'][idx, :] / data['DiscGas'][idx, :], 0)
            z_stars = np.where(data['DiscStars'][idx, :] > 1e-6,
                              data['DiscStarsMetals'][idx, :] / data['DiscStars'][idx, :], 0)
            
            all_radii.append(r_mid[valid])
            all_gas.append(gas_surf[valid])
            all_stars.append(stars_surf[valid])
            all_metals_gas.append(z_gas[valid] / SOLAR_Z)  # Solar units
            all_metals_stars.append(z_stars[valid] / SOLAR_Z)
            all_dust.append(dust_surf[valid])
            all_h2.append(h2_surf[valid])
            all_hi.append(hi_surf[valid])
        
        if len(all_radii) < 5:
            continue
        
        # Compute median profiles in radial bins
        r_bins = np.logspace(-0.5, 1.7, 15)  # 0.3 to 50 kpc
        r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
        
        def bin_profile(radii_list, values_list):
            medians = []
            p16 = []
            p84 = []
            for i in range(len(r_bins)-1):
                all_vals = []
                for r, v in zip(radii_list, values_list):
                    mask_r = (r >= r_bins[i]) & (r < r_bins[i+1])
                    if np.sum(mask_r) > 0:
                        all_vals.extend(v[mask_r])
                if len(all_vals) > 3:
                    medians.append(np.median(all_vals))
                    p16.append(np.percentile(all_vals, 16))
                    p84.append(np.percentile(all_vals, 84))
                else:
                    medians.append(np.nan)
                    p16.append(np.nan)
                    p84.append(np.nan)
            return np.array(medians), np.array(p16), np.array(p84)
        
        gas_med, gas_16, gas_84 = bin_profile(all_radii, all_gas)
        stars_med, stars_16, stars_84 = bin_profile(all_radii, all_stars)
        zgas_med, zgas_16, zgas_84 = bin_profile(all_radii, all_metals_gas)
        zstars_med, zstars_16, zstars_84 = bin_profile(all_radii, all_metals_stars)
        dust_med, dust_16, dust_84 = bin_profile(all_radii, all_dust)
        h2_med, h2_16, h2_84 = bin_profile(all_radii, all_h2)
        hi_med, hi_16, hi_84 = bin_profile(all_radii, all_hi)
        
        # Plot gas surface density
        ax = fig.add_subplot(gs[mass_idx, 0])
        ax.plot(r_centers, gas_med, '-', color=colors[mass_idx], lw=2, label='Total')
        ax.fill_between(r_centers, gas_16, gas_84, color=colors[mass_idx], alpha=0.2)
        ax.plot(r_centers, h2_med, '--', color=colors[mass_idx], lw=1.5, alpha=0.7, label='H₂')
        ax.plot(r_centers, hi_med, ':', color=colors[mass_idx], lw=1.5, alpha=0.7, label='HI')
        ax.set_xlabel('Radius [kpc]')
        ax.set_ylabel('Σ$_{gas}$ [M$_☉$ pc$^{-2}$]')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(0.5, 50)
        ax.set_ylim(1e-2, 1e4)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        if mass_idx == 0:
            ax.set_title('Gas Surface Density', fontsize=10, fontweight='bold')
        ax.text(0.05, 0.95, f'M* = {label} M$_☉$/h\nN = {len(all_radii)}',
                transform=ax.transAxes, va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot stellar surface density
        ax = fig.add_subplot(gs[mass_idx, 1])
        ax.plot(r_centers, stars_med, '-', color=colors[mass_idx], lw=2)
        ax.fill_between(r_centers, stars_16, stars_84, color=colors[mass_idx], alpha=0.2)
        ax.set_xlabel('Radius [kpc]')
        ax.set_ylabel('Σ$_{*}$ [M$_☉$ pc$^{-2}$]')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(0.5, 50)
        ax.set_ylim(1e-1, 1e4)
        ax.grid(alpha=0.3)
        if mass_idx == 0:
            ax.set_title('Stellar Surface Density', fontsize=10, fontweight='bold')
        
        # Plot metallicity
        ax = fig.add_subplot(gs[mass_idx, 2])
        ax.plot(r_centers, zgas_med, '-', color=colors[mass_idx], lw=2, label='Gas')
        ax.fill_between(r_centers, zgas_16, zgas_84, color=colors[mass_idx], alpha=0.2)
        ax.plot(r_centers, zstars_med, '--', color=colors[mass_idx], lw=2, alpha=0.7, label='Stars')
        ax.set_xlabel('Radius [kpc]')
        ax.set_ylabel('Z/Z$_☉$')
        ax.set_xscale('log')
        ax.set_xlim(0.5, 50)
        ax.set_ylim(0, 2)
        ax.axhline(1, color='k', ls=':', alpha=0.5, lw=1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        if mass_idx == 0:
            ax.set_title('Metallicity', fontsize=10, fontweight='bold')
        
        # Plot dust surface density
        ax = fig.add_subplot(gs[mass_idx, 3])
        ax.plot(r_centers, dust_med, '-', color=colors[mass_idx], lw=2)
        ax.fill_between(r_centers, dust_16, dust_84, color=colors[mass_idx], alpha=0.2)
        ax.set_xlabel('Radius [kpc]')
        ax.set_ylabel('Σ$_{dust}$ [M$_☉$ pc$^{-2}$]')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(0.5, 50)
        ax.set_ylim(1e-6, 1e1)
        ax.grid(alpha=0.3)
        if mass_idx == 0:
            ax.set_title('Dust Surface Density', fontsize=10, fontweight='bold')
    
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"\n📊 Saved: {output}")

def plot_dust_to_gas_vs_metallicity(data, output='plotting/darkmode_dtg_metallicity.png'):
    """Plot dust-to-gas ratio vs metallicity"""
    
    print("\n" + "="*70)
    print("2. DUST-TO-GAS RATIO vs METALLICITY")
    print("="*70)
    
    # Compute global D/G and Z for all galaxies
    mask = (data['ColdGas'] > 0.01) & (data['Type'] == 0)
    
    cold_gas = data['ColdGas'][mask]
    cold_dust = data['ColdDust'][mask]
    metals = data['MetalsColdGas'][mask]
    stellar_mass = data['StellarMass'][mask]
    
    # Compute ratios
    dtg = cold_dust / cold_gas
    metallicity = metals / cold_gas / SOLAR_Z  # Z/Z_sun
    
    print(f"Galaxies with gas: {len(cold_gas)}")
    print(f"D/G range: [{np.min(dtg[dtg>0]):.2e}, {np.max(dtg):.2e}]")
    print(f"Metallicity range: [{np.min(metallicity[metallicity>0]):.3f}, {np.max(metallicity):.3f}] Z/Z☉")
    
    # Also compute for resolved bins
    dtg_bins = []
    z_bins = []
    if data['has_darkmode']:
        for i in range(len(data['DiscGas'])):
            for j in range(data['DiscGas'].shape[1]):
                if data['DiscGas'][i, j] > 1e-4:
                    dtg_bin = data['DiscDust'][i, j] / data['DiscGas'][i, j]
                    z_bin = data['DiscGasMetals'][i, j] / data['DiscGas'][i, j] / SOLAR_Z
                    if dtg_bin > 0 and z_bin > 0:
                        dtg_bins.append(dtg_bin)
                        z_bins.append(z_bin)
        dtg_bins = np.array(dtg_bins)
        z_bins = np.array(z_bins)
        print(f"Resolved bins with dust & gas: {len(dtg_bins)}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel 1: Galaxy-integrated
    ax = axes[0]
    valid = (dtg > 0) & (metallicity > 0)
    
    # Color by stellar mass
    sc = ax.scatter(metallicity[valid], dtg[valid], 
                   c=np.log10(stellar_mass[valid]), 
                   s=20, alpha=0.5, cmap='viridis',
                   vmin=0, vmax=2, edgecolors='none')
    
    # Observed relation (Rémy-Ruyer+2014)
    z_model = np.logspace(-1.5, 0.5, 50)
    dtg_model = 0.006 * (z_model / 1.0)**1.0  # Simplified linear scaling
    ax.plot(z_model, dtg_model, 'r--', lw=2, label='Rémy-Ruyer+14 (approx)')
    
    ax.set_xlabel('12 + log(O/H) → Z/Z$_☉$', fontsize=12)
    ax.set_ylabel('Dust-to-Gas Ratio', fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.03, 3)
    ax.set_ylim(1e-5, 0.1)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title('Galaxy-Integrated D/G vs Metallicity', fontweight='bold')
    
    cbar = plt.colorbar(sc, ax=ax, label='log$_{10}$(M$_*$ / 10$^{10}$ M$_☉$/h)')
    
    # Panel 2: Resolved bins (if available)
    ax = axes[1]
    if data['has_darkmode'] and len(dtg_bins) > 100:
        valid_bins = (dtg_bins > 0) & (z_bins > 0)
        h = ax.hexbin(z_bins[valid_bins], dtg_bins[valid_bins],
                     gridsize=50, cmap='viridis', mincnt=1, 
                     xscale='log', yscale='log', bins='log')
        
        # Observed relation
        ax.plot(z_model, dtg_model, 'r--', lw=2, label='Rémy-Ruyer+14 (approx)')
        
        ax.set_xlabel('Z/Z$_☉$', fontsize=12)
        ax.set_ylabel('Dust-to-Gas Ratio', fontsize=12)
        ax.set_xlim(0.03, 3)
        ax.set_ylim(1e-5, 0.1)
        ax.grid(alpha=0.3)
        ax.legend()
        ax.set_title('Resolved Bins D/G vs Metallicity', fontweight='bold')
        
        plt.colorbar(h, ax=ax, label='log$_{10}$(N)')
    else:
        ax.text(0.5, 0.5, 'No DarkMode data\nfor resolved analysis',
               ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_xlabel('Z/Z$_☉$')
        ax.set_ylabel('D/G')
    
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"📊 Saved: {output}")

def plot_smf_comparison(darkmode_file, baseline_file=None, output='plotting/darkmode_smf_comparison.png'):
    """Compare stellar mass function with and without DarkMode"""
    
    print("\n" + "="*70)
    print("3. STELLAR MASS FUNCTION COMPARISON")
    print("="*70)
    
    # Load DarkMode data
    data_dm = load_darkmode_data(darkmode_file, snap=63)
    stellar_mass_dm = data_dm['StellarMass'][data_dm['Type'] == 0]  # Central galaxies only
    
    print(f"DarkMode run: {len(stellar_mass_dm)} central galaxies")
    
    # Try to load baseline
    if baseline_file is None:
        # Look for a baseline file
        import os
        baseline_candidates = [
            'output/millennium_nodarkmode/model_0.hdf5',
            'output/millennium_baseline/model_0.hdf5'
        ]
        for candidate in baseline_candidates:
            if os.path.exists(candidate):
                baseline_file = candidate
                break
    
    stellar_mass_base = None
    if baseline_file:
        try:
            data_base = load_darkmode_data(baseline_file, snap=63)
            stellar_mass_base = data_base['StellarMass'][data_base['Type'] == 0]
            print(f"Baseline run: {len(stellar_mass_base)} central galaxies")
        except:
            print(f"⚠️  Could not load baseline from {baseline_file}")
            stellar_mass_base = None
    else:
        print("⚠️  No baseline file found - showing DarkMode only")
    
    # Compute SMF
    mass_bins = np.logspace(8, 12.5, 30) / 1e10 / h  # Convert to 10^10 M_sun/h units
    
    # Volume (Millennium: 62.5^3 Mpc^3/h^3)
    volume = 62.5**3  # (Mpc/h)^3
    
    def compute_smf(masses, bins, vol):
        hist, edges = np.histogram(masses, bins=bins)
        bin_widths = np.diff(np.log10(edges * 1e10 * h))  # dex
        phi = hist / vol / bin_widths  # Mpc^-3 h^3 dex^-1
        bin_centers = 0.5 * (edges[:-1] + edges[1:])
        return bin_centers * 1e10 * h, phi  # M_sun, Mpc^-3 h^3 dex^-1
    
    mass_dm, phi_dm = compute_smf(stellar_mass_dm, mass_bins, volume)
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # DarkMode
    ax.plot(np.log10(mass_dm), np.log10(phi_dm), 'o-', lw=2, 
           color='#1f77b4', label=f'DarkMode (N={len(stellar_mass_dm)})',
           markersize=6)
    
    # Baseline (if available)
    if stellar_mass_base is not None:
        mass_base, phi_base = compute_smf(stellar_mass_base, mass_bins, volume)
        ax.plot(np.log10(mass_base), np.log10(phi_base), 's--', lw=2,
               color='#ff7f0e', label=f'Baseline (N={len(stellar_mass_base)})',
               markersize=6, alpha=0.7)
        
        # Difference
        valid = (phi_dm > 0) & (phi_base > 0)
        if np.sum(valid) > 0:
            diff = np.abs(np.log10(phi_dm[valid]) - np.log10(phi_base[valid]))
            mean_diff = np.mean(diff)
            max_diff = np.max(diff)
            print(f"\nSMF Comparison:")
            print(f"  Mean difference: {mean_diff:.3f} dex")
            print(f"  Max difference: {max_diff:.3f} dex")
            
            if mean_diff < 0.05:
                print("  ✅ SMFs are nearly identical (< 0.05 dex)")
            elif mean_diff < 0.1:
                print("  ✓ SMFs are very similar (< 0.1 dex)")
            else:
                print(f"  ⚠️  SMFs differ by {mean_diff:.2f} dex on average")
    
    # Observations (Baldry+2012 z~0)
    # Approximate values
    obs_mass = np.array([8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5])
    obs_phi = np.array([-1.5, -1.8, -2.2, -2.5, -3.0, -3.7, -4.5])
    ax.plot(obs_mass, obs_phi, 'k^-', lw=1.5, label='Baldry+12 (z~0)', 
           markersize=7, alpha=0.6)
    
    ax.set_xlabel('log$_{10}$(M$_*$ / M$_☉$)', fontsize=14)
    ax.set_ylabel('log$_{10}$(Φ / Mpc$^{-3}$ h$^3$ dex$^{-1}$)', fontsize=14)
    ax.set_xlim(8.5, 12)
    ax.set_ylim(-6, -1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_title('Stellar Mass Function at z=0', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"📊 Saved: {output}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='DarkMode + Dust Analysis')
    parser.add_argument('--darkmode', default='output/millennium/model_0.hdf5',
                       help='DarkMode output file')
    parser.add_argument('--baseline', default=None,
                       help='Baseline (no DarkMode) output file for comparison')
    parser.add_argument('--snap', type=int, default=63,
                       help='Snapshot number')
    args = parser.parse_args()
    
    print("="*70)
    print("DARKMODE + DUST COMPREHENSIVE ANALYSIS")
    print("="*70)
    print(f"Input: {args.darkmode}")
    print(f"Snapshot: {args.snap}")
    
    # Load data
    data = load_darkmode_data(args.darkmode, snap=args.snap)
    
    # Generate plots
    plot_radial_profiles(data)
    plot_dust_to_gas_vs_metallicity(data)
    plot_smf_comparison(args.darkmode, args.baseline)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nGenerated plots:")
    print("  1. plotting/darkmode_radial_profiles.png")
    print("  2. plotting/darkmode_dtg_metallicity.png")
    print("  3. plotting/darkmode_smf_comparison.png")

if __name__ == '__main__':
    main()

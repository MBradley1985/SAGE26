#!/usr/bin/env python3
"""
SAGE H2 Fraction Analysis
==========================

This script analyzes the H2 fraction as a function of stellar mass for different
H2 models in the SAGE semi-analytic model output.

H2 fraction is defined as: f_H2 = M_H2 / M_ColdGas

Note: In SAGE, ColdGas = Total cold gas (HI + H2), and H2gas = H2 only
Therefore: HI = ColdGas - H2gas

The script compares different H2 prescription models from the output folder.

Author: Analysis script for SAGE semi-analytic model
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
MILLENNIUM_BARYON_FRACTION = 0.17

# Define H2 model configurations
H2_MODEL_CONFIGS = [
    {
        'name': 'KMT09',
        'dir': './output/millennium_kmt09/',
        'color': 'blue',
        'linestyle': '-',
        'linewidth': 2,
        'alpha': 0.8,
        'boxsize': MILLENNIUM_BOXSIZE,
        'hubble_h': MILLENNIUM_HUBBLE_H,
        'redshifts': DEFAULT_REDSHIFTS
    },
    {
        'name': 'K13',
        'dir': './output/millennium_k13/',
        'color': 'green',
        'linestyle': '-',
        'linewidth': 2,
        'alpha': 0.8,
        'boxsize': MILLENNIUM_BOXSIZE,
        'hubble_h': MILLENNIUM_HUBBLE_H,
        'redshifts': DEFAULT_REDSHIFTS
    },
    {
        'name': 'KD12',
        'dir': './output/millennium_kd12/',
        'color': 'red',
        'linestyle': '-',
        'linewidth': 2,
        'alpha': 0.8,
        'boxsize': MILLENNIUM_BOXSIZE,
        'hubble_h': MILLENNIUM_HUBBLE_H,
        'redshifts': DEFAULT_REDSHIFTS
    },
    {
        'name': 'GD14',
        'dir': './output/millennium_gd14/',
        'color': 'purple',
        'linestyle': '-',
        'linewidth': 2,
        'alpha': 0.8,
        'boxsize': MILLENNIUM_BOXSIZE,
        'hubble_h': MILLENNIUM_HUBBLE_H,
        'redshifts': DEFAULT_REDSHIFTS
    },
    {
        'name': 'BR06',
        'dir': './output/millennium/',
        'color': 'orange',
        'linestyle': '-',
        'linewidth': 2,
        'alpha': 0.8,
        'boxsize': MILLENNIUM_BOXSIZE,
        'hubble_h': MILLENNIUM_HUBBLE_H,
        'redshifts': DEFAULT_REDSHIFTS
    }
]

OutputDir = './output/millennium/plots/'
plt.style.use('/Users/mbradley/Documents/cohare_palatino_sty.mplstyle')


def read_hdf(directory, snap_num=None, param=None):
    """Read data from one or more SAGE model files"""
    # Get list of all model files in directory
    model_files = [f for f in os.listdir(directory) if f.startswith('model_') and f.endswith('.hdf5')]
    model_files.sort()
    
    # Initialize empty array for combined data
    combined_data = None
    
    # Read and combine data from each model file
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


def get_redshift_from_snapshot(snap_num, model_config):
    """Convert snapshot number to redshift using the model-specific redshift list"""
    redshifts = model_config['redshifts']
    
    if isinstance(snap_num, str):
        # Extract number from string like 'Snap_042'
        snap_num = int(snap_num.split('_')[-1])
    
    if 0 <= snap_num < len(redshifts):
        return redshifts[snap_num]
    else:
        raise ValueError(f"Snapshot {snap_num} out of range for redshift list")


def calculate_binned_h2_fraction(stellar_mass, h2_gas, cold_gas, mass_bins):
    """
    Calculate median H2 fraction in stellar mass bins
    
    Parameters:
    -----------
    stellar_mass : array
        Stellar mass of galaxies in solar masses
    h2_gas : array
        H2 gas mass in solar masses
    cold_gas : array
        Total cold gas mass in solar masses (HI + H2)
    mass_bins : array
        Mass bin edges
        
    Returns:
    --------
    bin_centers : array
        Center of each mass bin
    median_h2_frac : array
        Median H2 fraction in each bin
    percentile_16 : array
        16th percentile of H2 fraction
    percentile_84 : array
        84th percentile of H2 fraction
    n_galaxies : array
        Number of galaxies in each bin
    
    Note:
    -----
    SAGE has a bug where H2gas is not updated when ColdGas is reduced by star 
    formation and feedback, causing H2gas > ColdGas. The clipping handles this.
    
    Bug location: src/model_starformation_and_feedback.c
    - H2gas is calculated from f_H2 * ColdGas (line ~216)
    - Then ColdGas is reduced by star formation (line ~758)
    - And by feedback reheating (line ~785)
    - But H2gas is never updated, leading to H2gas > ColdGas
    """
    bin_centers = 0.5 * (mass_bins[:-1] + mass_bins[1:])
    median_h2_frac = np.zeros(len(bin_centers))
    percentile_16 = np.zeros(len(bin_centers))
    percentile_84 = np.zeros(len(bin_centers))
    n_galaxies = np.zeros(len(bin_centers), dtype=int)
    
    for i in range(len(bin_centers)):
        # Find galaxies in this mass bin
        mask = (stellar_mass >= mass_bins[i]) & (stellar_mass < mass_bins[i+1])
        mask = mask & (cold_gas > 0)  # Only consider galaxies with cold gas
        
        if np.sum(mask) > 0:
            # Calculate H2 fraction, clipping to [0, 1] to handle the SAGE bug
            # where H2gas > ColdGas due to ColdGas being reduced after H2 calculation
            h2_fraction = np.clip(h2_gas[mask] / cold_gas[mask], 0.0, 1.0)
            median_h2_frac[i] = np.median(h2_fraction)
            percentile_16[i] = np.percentile(h2_fraction, 16)
            percentile_84[i] = np.percentile(h2_fraction, 84)
            n_galaxies[i] = np.sum(mask)
        else:
            median_h2_frac[i] = np.nan
            percentile_16[i] = np.nan
            percentile_84[i] = np.nan
            n_galaxies[i] = 0
    
    return bin_centers, median_h2_frac, percentile_16, percentile_84, n_galaxies


def plot_h2_fraction_vs_stellar_mass(redshifts_to_plot=[0.0, 0.5, 1.0, 2.0, 3.0, 4.0]):
    """
    Plot H2 fraction as a function of stellar mass for different H2 models
    
    Parameters:
    -----------
    redshifts_to_plot : list
        List of redshifts to create plots for
    """
    # Create mass bins (log scale)
    mass_bins = np.arange(8.0, 12.5, 0.2)
    
    for target_z in redshifts_to_plot:
        print(f"\n{'='*60}")
        print(f"Creating plot for z ~ {target_z}")
        print(f"{'='*60}")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Track which models have data for legend
        models_plotted = []
        
        for model_config in H2_MODEL_CONFIGS:
            model_name = model_config['name']
            directory = model_config['dir']
            color = model_config['color']
            linestyle = model_config['linestyle']
            linewidth = model_config['linewidth']
            alpha = model_config['alpha']
            hubble_h = model_config['hubble_h']
            
            # Check if directory exists
            if not os.path.exists(directory):
                print(f"  Warning: Directory {directory} does not exist, skipping {model_name}")
                continue
            
            # Find the snapshot closest to target redshift
            redshifts = model_config['redshifts']
            z_diff = [abs(z - target_z) for z in redshifts]
            best_snap = z_diff.index(min(z_diff))
            actual_z = redshifts[best_snap]
            
            snap_str = f'Snap_{best_snap}'
            print(f"  {model_name}: Using {snap_str} (z={actual_z:.2f})")
            
            # Read galaxy properties
            try:
                stellar_mass = read_hdf(directory, snap_num=snap_str, param='StellarMass')
                h2_gas = read_hdf(directory, snap_num=snap_str, param='H2gas')
                cold_gas = read_hdf(directory, snap_num=snap_str, param='ColdGas')
                galaxy_type = read_hdf(directory, snap_num=snap_str, param='Type')
                
                if stellar_mass is None or h2_gas is None or cold_gas is None:
                    print(f"    Could not read data for {model_name}, skipping")
                    continue
                
                # Convert to solar masses
                stellar_mass = stellar_mass * 1.0e10 / hubble_h
                h2_gas = h2_gas * 1.0e10 / hubble_h
                cold_gas = cold_gas * 1.0e10 / hubble_h
                
                # Filter for star-forming galaxies (Type 0 = central, Type 1 = satellite)
                # Only keep galaxies with stellar mass > 0 and cold gas > 0
                mask = (stellar_mass > 0) & (cold_gas > 0)
                
                stellar_mass = stellar_mass[mask]
                h2_gas = h2_gas[mask]
                cold_gas = cold_gas[mask]
                
                print(f"    Number of galaxies with cold gas: {len(stellar_mass)}")
                
                if len(stellar_mass) == 0:
                    print(f"    No valid galaxies found for {model_name}, skipping")
                    continue
                
                # Calculate log stellar mass
                log_stellar_mass = np.log10(stellar_mass)
                
                # Calculate binned H2 fraction
                bin_centers, median_h2_frac, percentile_16, percentile_84, n_galaxies = \
                    calculate_binned_h2_fraction(log_stellar_mass, h2_gas, cold_gas, mass_bins)
                
                # Only plot bins with at least 10 galaxies
                valid_mask = (n_galaxies >= 10) & np.isfinite(median_h2_frac)
                
                if np.sum(valid_mask) > 0:
                    # Plot median line
                    ax.plot(bin_centers[valid_mask], median_h2_frac[valid_mask],
                           color=color, linestyle=linestyle, linewidth=linewidth,
                           label=model_name, alpha=alpha)
                    
                    # Plot shaded region for 16th-84th percentile
                    ax.fill_between(bin_centers[valid_mask], 
                                   percentile_16[valid_mask], 
                                   percentile_84[valid_mask],
                                   color=color, alpha=0.2)
                    
                    models_plotted.append(model_name)
                    print(f"    Plotted {np.sum(valid_mask)} mass bins")
                else:
                    print(f"    No valid bins found for {model_name}")
                    
            except Exception as e:
                print(f"    Error processing {model_name}: {e}")
                continue
        
        # Format plot
        ax.set_xlabel(r'$\log_{10}(M_\star / \mathrm{M}_\odot)$', fontsize=14)
        ax.set_ylabel(r'$f_{\mathrm{H_2}} = M_{\mathrm{H_2}} / M_{\mathrm{cold}}$', fontsize=14)
        ax.set_title(f'H2 Fraction vs Stellar Mass at z $\\approx$ {target_z:.1f}', fontsize=16)
        
        ax.set_xlim(8.0, 12.0)
        ax.set_ylim(0.0, 1.0)
        
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        if len(models_plotted) > 0:
            ax.legend(loc='best', fontsize=11, framealpha=0.9)
        
        # Save figure
        os.makedirs(OutputDir, exist_ok=True)
        output_filename = f'h2_fraction_vs_stellar_mass_z{target_z:.1f}.pdf'
        output_path = os.path.join(OutputDir, output_filename)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n  Saved figure: {output_path}")
        plt.close()
        
        print(f"  Models plotted: {', '.join(models_plotted) if models_plotted else 'None'}")


def plot_h2_fraction_multi_redshift():
    """
    Create a multi-panel plot showing H2 fraction evolution across redshifts
    """
    redshifts_to_plot = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0]
    mass_bins = np.arange(8.0, 12.5, 0.2)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, target_z in enumerate(redshifts_to_plot):
        ax = axes[idx]
        print(f"\nProcessing panel {idx+1}/6: z ~ {target_z}")
        
        for model_config in H2_MODEL_CONFIGS:
            model_name = model_config['name']
            directory = model_config['dir']
            color = model_config['color']
            linestyle = model_config['linestyle']
            linewidth = model_config['linewidth']
            alpha = model_config['alpha']
            hubble_h = model_config['hubble_h']
            
            if not os.path.exists(directory):
                continue
            
            # Find closest snapshot
            redshifts = model_config['redshifts']
            z_diff = [abs(z - target_z) for z in redshifts]
            best_snap = z_diff.index(min(z_diff))
            actual_z = redshifts[best_snap]
            snap_str = f'Snap_{best_snap}'
            
            try:
                stellar_mass = read_hdf(directory, snap_num=snap_str, param='StellarMass')
                h2_gas = read_hdf(directory, snap_num=snap_str, param='H2gas')
                cold_gas = read_hdf(directory, snap_num=snap_str, param='ColdGas')
                
                if stellar_mass is None or h2_gas is None or cold_gas is None:
                    continue
                
                # Convert to solar masses
                stellar_mass = stellar_mass * 1.0e10 / hubble_h
                h2_gas = h2_gas * 1.0e10 / hubble_h
                cold_gas = cold_gas * 1.0e10 / hubble_h
                
                # Filter
                mask = (stellar_mass > 0) & (cold_gas > 0)
                stellar_mass = stellar_mass[mask]
                h2_gas = h2_gas[mask]
                cold_gas = cold_gas[mask]
                
                if len(stellar_mass) == 0:
                    continue
                
                log_stellar_mass = np.log10(stellar_mass)
                
                # Calculate binned H2 fraction
                bin_centers, median_h2_frac, percentile_16, percentile_84, n_galaxies = \
                    calculate_binned_h2_fraction(log_stellar_mass, h2_gas, cold_gas, mass_bins)
                
                valid_mask = (n_galaxies >= 10) & np.isfinite(median_h2_frac)
                
                if np.sum(valid_mask) > 0:
                    # Plot median line
                    ax.plot(bin_centers[valid_mask], median_h2_frac[valid_mask],
                           color=color, linestyle=linestyle, linewidth=linewidth,
                           label=model_name if idx == 0 else None, alpha=alpha)
                    
                    # Plot shaded region
                    ax.fill_between(bin_centers[valid_mask], 
                                   percentile_16[valid_mask], 
                                   percentile_84[valid_mask],
                                   color=color, alpha=0.15)
                    
            except Exception as e:
                print(f"  Error processing {model_name}: {e}")
                continue
        
        # Format panel
        ax.set_xlim(8.0, 12.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.text(0.95, 0.95, f'z $\\approx$ {target_z:.1f}', 
               transform=ax.transAxes, ha='right', va='top',
               fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add axis labels
        if idx >= 3:  # Bottom row
            ax.set_xlabel(r'$\log_{10}(M_\star / \mathrm{M}_\odot)$', fontsize=12)
        if idx % 3 == 0:  # Left column
            ax.set_ylabel(r'$f_{\mathrm{H_2}}$', fontsize=12)
    
    # Add legend to first panel
    axes[0].legend(loc='lower right', fontsize=10, framealpha=0.9)
    
    # Overall title
    fig.suptitle('H2 Fraction Evolution: Comparison of H2 Models', fontsize=16, y=0.995)
    
    # Save figure
    os.makedirs(OutputDir, exist_ok=True)
    output_path = os.path.join(OutputDir, 'h2_fraction_evolution_multi_redshift.pdf')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved multi-panel figure: {output_path}")
    plt.close()


if __name__ == '__main__':
    print("="*60)
    print("SAGE H2 Fraction Analysis")
    print("="*60)
    
    # Check which H2 model directories exist
    print("\nChecking available H2 model directories:")
    available_models = []
    for config in H2_MODEL_CONFIGS:
        if os.path.exists(config['dir']):
            print(f"  ✓ {config['name']}: {config['dir']}")
            available_models.append(config['name'])
        else:
            print(f"  ✗ {config['name']}: {config['dir']} (not found)")
    
    if len(available_models) == 0:
        print("\nError: No H2 model directories found!")
        print("Please update H2_MODEL_CONFIGS in the script to point to your model directories.")
        exit(1)
    
    print(f"\nFound {len(available_models)} H2 model(s): {', '.join(available_models)}")
    
    # Create individual plots for selected redshifts
    print("\n" + "="*60)
    print("Creating individual redshift plots...")
    print("="*60)
    plot_h2_fraction_vs_stellar_mass(redshifts_to_plot=[0.0, 0.5, 1.0, 2.0, 3.0, 4.0])
    
    # Create multi-panel evolution plot
    print("\n" + "="*60)
    print("Creating multi-redshift evolution plot...")
    print("="*60)
    plot_h2_fraction_multi_redshift()
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)

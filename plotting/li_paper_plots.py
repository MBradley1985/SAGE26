#!/usr/bin/env python

import h5py as h5
import numpy as np
from numpy.random import seed
import matplotlib.pyplot as plt
import os
import warnings
import sys
from astropy.table import Table

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# ========================== USER OPTIONS ==========================

# Snapshot to analyze (0 = z=0, highest number = highest redshift)
Snapshot = 'Snap_63'  # z=0 for Millennium

# File details
DirName = './output/millennium/'
FileName = 'model_0.hdf5'

# Configuration parameters - UPDATE THESE TO MATCH YOUR SETUP

# Default redshifts for Millennium simulation
DEFAULT_REDSHIFTS = [127.000, 79.998, 50.000, 30.000, 19.916, 18.244, 16.725, 15.343, 14.086, 12.941, 11.897, 10.944, 10.073, 
                     9.278, 8.550, 7.883, 7.272, 6.712, 6.197, 5.724, 5.289, 4.888, 4.520, 4.179, 3.866, 3.576, 3.308, 3.060, 
                     2.831, 2.619, 2.422, 2.239, 2.070, 1.913, 1.766, 1.630, 1.504, 1.386, 1.276, 1.173, 1.078, 0.989, 0.905, 
                     0.828, 0.755, 0.687, 0.624, 0.564, 0.509, 0.457, 0.408, 0.362, 0.320, 0.280, 0.242, 0.208, 0.175, 0.144, 
                     0.116, 0.089, 0.064, 0.041, 0.020, 0.000]

# Default Millennium simulation parameters
MILLENNIUM_BOXSIZE = 62.5
MILLENNIUM_HUBBLE_H = 0.73
MILLENNIUM_BARYON_FRACTION = 0.17

# Define multiple directories and their properties
MODEL_CONFIGS = [
    {
        'name': 'SAGE26',           # Display name for legend (Duplicate for High-Z/All-Z plots)
        'dir': './output/millennium/',  # Directory path
        'color': 'black',            # Color for plotting
        'linestyle': '-',            # Line style
        'linewidth': 3,              # Thick line for SAGE25
        'alpha': 0.8,                # Transparency
        'boxsize': MILLENNIUM_BOXSIZE,             # Box size in h^-1 Mpc for this model
        'volume_fraction': 1.0,      # Fraction of the full volume output by the model
        'use_for_residuals': False,  # NEW: Flag to indicate this is NOT the comparison model
        'hubble_h': MILLENNIUM_HUBBLE_H,            # Hubble parameter for this model
        'baryon_fraction': MILLENNIUM_BARYON_FRACTION,     # Baryon fraction for this model
        'redshifts': DEFAULT_REDSHIFTS  # Redshift of each snapshot for this model
    },
    {
        'name': 'SAGE C16',           # Display name for legend
        'dir': '../SAGE-VANILLA/sage-model/output/millennium/',  # Second directory path
        'color': 'blue',             # Color for plotting
        'linestyle': '--',           # Dashed line style
        'linewidth': 2,              # Thin line for Vanilla SAGE
        'alpha': 0.8,                # Transparency
        'boxsize': MILLENNIUM_BOXSIZE,             # Box size in h^-1 Mpc for this model
        'volume_fraction': 1.0,      # Fraction of the full volume output by the model
        'use_for_residuals': True,   # NEW: Flag to indicate this is the comparison model
        'hubble_h': MILLENNIUM_HUBBLE_H,            # Different Hubble parameter for this model
        'baryon_fraction': MILLENNIUM_BARYON_FRACTION,     # Baryon fraction for this model
        'redshifts': DEFAULT_REDSHIFTS  # Redshift of each snapshot for this model
    },
    # NEW: Add your comparison model here
    {
        'name': 'evilSAGE',   # UPDATE: Display name for your comparison model
        'dir': '/Users/mbradley/Documents/PhD/SAGE_BROKEN/sage-model/output/millennium/',  # UPDATE: Path to your comparison model directory
        'color': 'purple',            # Color for plotting
        'linestyle': ':',           # Dotted line style
        'linewidth': 2,             # Line width
        'alpha': 0.8,               # Transparency
        'boxsize': MILLENNIUM_BOXSIZE,            # Box size in h^-1 Mpc for this model
        'volume_fraction': 1.0,     # Fraction of the full volume output by the model
        'use_for_residuals': False,   # NEW: Flag to indicate this is the comparison model
        'hubble_h': MILLENNIUM_HUBBLE_H,            # Different Hubble parameter for this model
        'baryon_fraction': MILLENNIUM_BARYON_FRACTION,     # Baryon fraction for this model
        'redshifts': DEFAULT_REDSHIFTS  # Redshift of each snapshot for this model
    },
    # --- PLACEHOLDERS FOR NEW MODELS (Update 'dir' and other properties) ---
    {
        'name': 'C16 feedback',
        'dir': './output/millennium_c16feedback/', 
        'color': 'cyan', 
        'linestyle': '-', 'alpha': 0.8, 
        'boxsize': MILLENNIUM_BOXSIZE, 'volume_fraction': 1.0, 
        'hubble_h': MILLENNIUM_HUBBLE_H, 
        'baryon_fraction': MILLENNIUM_BARYON_FRACTION,
        'redshifts': DEFAULT_REDSHIFTS
    },
    {
        'name': 'no FFB', 'dir': './output/millennium_noffb/', 
        'color': 'green', 'linestyle': '-', 'alpha': 0.8, 
        'boxsize': MILLENNIUM_BOXSIZE, 'volume_fraction': 1.0, 
        'hubble_h': MILLENNIUM_HUBBLE_H, 
        'baryon_fraction': MILLENNIUM_BARYON_FRACTION,
        'redshifts': DEFAULT_REDSHIFTS
    },
    {
        'name': '30% FFB', 'dir': './output/millennium_FFB30/', 
        'color': 'purple', 'linestyle': '-', 'alpha': 0.8, 
        'boxsize': MILLENNIUM_BOXSIZE, 'volume_fraction': 1.0, 
        'hubble_h': MILLENNIUM_HUBBLE_H, 
        'baryon_fraction': MILLENNIUM_BARYON_FRACTION,
        'redshifts': DEFAULT_REDSHIFTS
    },
    {
        'name': '40% FFB', 'dir': './output/millennium_FFB40/', 
        'color': 'red', 'linestyle': '-', 'alpha': 0.8, 
        'boxsize': MILLENNIUM_BOXSIZE, 'volume_fraction': 1.0, 
        'hubble_h': MILLENNIUM_HUBBLE_H, 
        'baryon_fraction': MILLENNIUM_BARYON_FRACTION,
        'redshifts': DEFAULT_REDSHIFTS
    },
    {
        'name': '50% FFB', 'dir': './output/millennium_FFB50/', 
        'color': 'orange', 'linestyle': '-', 'alpha': 0.8, 
        'boxsize': MILLENNIUM_BOXSIZE, 'volume_fraction': 1.0, 
        'hubble_h': MILLENNIUM_HUBBLE_H, 
        'baryon_fraction': MILLENNIUM_BARYON_FRACTION,
        'redshifts': DEFAULT_REDSHIFTS
    },
    {
        'name': '80% FFB', 'dir': './output/millennium_FFB80/', 
        'color': 'yellow', 'linestyle': '-', 'alpha': 0.8, 
        'boxsize': MILLENNIUM_BOXSIZE, 'volume_fraction': 1.0, 
        'hubble_h': MILLENNIUM_HUBBLE_H, 
        'baryon_fraction': MILLENNIUM_BARYON_FRACTION,
        'redshifts': DEFAULT_REDSHIFTS
    },
    {
        'name': '100% FFB', 'dir': './output/millennium_FFB100/', 
        'color': 'purple', 'linestyle': '-', 'alpha': 0.8, 
        'boxsize': MILLENNIUM_BOXSIZE, 'volume_fraction': 1.0, 
        'hubble_h': MILLENNIUM_HUBBLE_H, 
        'baryon_fraction': MILLENNIUM_BARYON_FRACTION,
        'redshifts': DEFAULT_REDSHIFTS
    }
]

OutputFormat = '.pdf'
plt.rcParams["figure.figsize"] = (8, 6)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["font.size"] = 12

# Define plot models (used across all plots)
PLOT_MODELS = [
    {
        'name': 'No FFB',
        'dir': './output/millennium_noffb/',
        'color': 'gray',
        'linestyle': '-',
        'linewidth': 5,
        'hubble_h': MILLENNIUM_HUBBLE_H,
        'boxsize': MILLENNIUM_BOXSIZE,
        'volume_fraction': 1.0
    },
    {
        'name': 'FFB 20% (SAGE26)',
        'dir': './output/millennium/',
        'color': 'orange',
        'linestyle': '-',
        'linewidth': 5,
        'hubble_h': MILLENNIUM_HUBBLE_H,
        'boxsize': MILLENNIUM_BOXSIZE,
        'volume_fraction': 1.0
    },
    {
        'name': 'FFB 100%',
        'dir': './output/millennium_FFB100/',
        'color': 'royalblue',
        'linestyle': '-',
        'linewidth': 5,
        'hubble_h': MILLENNIUM_HUBBLE_H,
        'boxsize': MILLENNIUM_BOXSIZE,
        'volume_fraction': 1.0
    }
]

# ========================== DATA LOADING ==========================

def read_hdf(filename=None, snap_num=None, param=None):
    """Reads SAGE data from HDF5"""
    path = f"{DirName}{FileName}"
    try:
        with h5.File(path, 'r') as f:
            if snap_num not in f:
                print(f"Error: Snapshot {snap_num} not found in {path}")
                return np.zeros(1)
            
            # SAGE saves some params with units that need conversion
            data = np.array(f[snap_num][param])
            return data
    except Exception as e:
        print(f"Error reading {param}: {e}")
        return np.zeros(1)

def read_hdf_from_model(model_dir, filename, snap_num, param, hubble_h):
    """Reads SAGE data from HDF5 for a specific model"""
    path = f"{model_dir}{filename}"
    try:
        with h5.File(path, 'r') as f:
            if snap_num not in f:
                print(f"Error: Snapshot {snap_num} not found in {path}")
                return np.zeros(1)
            
            data = np.array(f[snap_num][param])
            return data
    except Exception as e:
        print(f"Error reading {param} from {path}: {e}")
        return np.zeros(1)

def calculate_smf(stellar_mass, sfr_disk, sfr_bulge, volume, hubble_h, binwidth=0.1, return_errors=False):
    """Calculate stellar mass function with optional Poisson errors
    
    Args:
        stellar_mass: Stellar masses
        sfr_disk: Disk star formation rates
        sfr_bulge: Bulge star formation rates
        volume: Volume in (h^-1 Mpc)^3
        hubble_h: Hubble parameter
        binwidth: Width of bins in dex
        return_errors: If True, return lower and upper error bounds
    
    Returns:
        If return_errors=False: (mass_bins, log_smf)
        If return_errors=True: (mass_bins, log_smf, log_smf_lower, log_smf_upper)
    """
    # Select galaxies with stellar mass > 0
    w = np.where(stellar_mass > 0.0)[0]
    if len(w) == 0:
        if return_errors:
            return np.array([]), np.array([]), np.array([]), np.array([])
        else:
            return np.array([]), np.array([])
    
    mass = np.log10(stellar_mass[w])
    
    mi = np.floor(min(mass)) - 2
    ma = np.floor(max(mass)) + 2
    NB = int((ma - mi) / binwidth)
    (counts, binedges) = np.histogram(mass, range=(mi, ma), bins=NB)
    xaxeshisto = binedges[:-1] + 0.5 * binwidth
    
    # Convert to number density (Mpc^-3 dex^-1)
    # counts are per bin, volume is in (h^-1 Mpc)^3, binwidth is in dex
    smf = counts / volume / binwidth
    
    if return_errors:
        # Poisson errors: N ± sqrt(N)
        counts_upper = counts + np.sqrt(counts)
        counts_lower = np.maximum(counts - np.sqrt(counts), 0.1)  # Avoid zero
        
        smf_upper = counts_upper / volume / binwidth
        smf_lower = counts_lower / volume / binwidth
        
        # Convert to log10
        smf_log = np.zeros(len(smf))
        smf_log_upper = np.zeros(len(smf))
        smf_log_lower = np.zeros(len(smf))
        
        w_nonzero = np.where(smf > 0)[0]
        smf_log[w_nonzero] = np.log10(smf[w_nonzero])
        smf_log_upper[w_nonzero] = np.log10(smf_upper[w_nonzero])
        smf_log_lower[w_nonzero] = np.log10(smf_lower[w_nonzero])
        
        smf_log[smf == 0] = -10  # Placeholder for zero values
        smf_log_upper[smf == 0] = -10
        smf_log_lower[smf == 0] = -10
        
        return xaxeshisto, smf_log, smf_log_lower, smf_log_upper
    else:
        # Convert to log10
        smf_log = np.zeros(len(smf))
        w_nonzero = np.where(smf > 0)[0]
        smf_log[w_nonzero] = np.log10(smf[w_nonzero])
        smf_log[smf == 0] = -10  # Placeholder for zero values
        
        return xaxeshisto, smf_log

def load_cosmos2020_smf(redshift):
    """Load COSMOS2020 (Farmer+) observational SMF data for a given redshift"""
    # Map redshift to appropriate file
    if redshift < 0.35:
        filename = './data/COSMOS2020/SMF_Farmer_v2.1_0.2z0.5_total.txt'
        z_label = 'COSMOS 0.2<z<0.5'
    elif redshift < 0.65:
        filename = './data/COSMOS2020/SMF_Farmer_v2.1_0.5z0.8_total.txt'
        z_label = 'COSMOS 0.5<z<0.8'
    elif redshift < 0.95:
        filename = './data/COSMOS2020/SMF_Farmer_v2.1_0.8z1.1_total.txt'
        z_label = 'COSMOS 0.8<z<1.1'
    elif redshift < 1.3:
        filename = './data/COSMOS2020/SMF_Farmer_v2.1_1.1z1.5_total.txt'
        z_label = 'COSMOS 1.1<z<1.5'
    elif redshift < 1.75:
        filename = './data/COSMOS2020/SMF_Farmer_v2.1_1.5z2.0_total.txt'
        z_label = 'COSMOS 1.5<z<2.0'
    elif redshift < 2.25:
        filename = './data/COSMOS2020/SMF_Farmer_v2.1_2.0z2.5_total.txt'
        z_label = 'COSMOS 2.0<z<2.5'
    elif redshift < 2.75:
        filename = './data/COSMOS2020/SMF_Farmer_v2.1_2.5z3.0_total.txt'
        z_label = 'COSMOS 2.5<z<3.0'
    elif redshift < 3.25:
        filename = './data/COSMOS2020/SMF_Farmer_v2.1_3.0z3.5_total.txt'
        z_label = 'COSMOS 3.0<z<3.5'
    elif redshift < 4.0:
        filename = './data/COSMOS2020/SMF_Farmer_v2.1_3.5z4.5_total.txt'
        z_label = 'COSMOS 3.5<z<4.5'
    elif redshift < 5.0:
        filename = './data/COSMOS2020/SMF_Farmer_v2.1_4.5z5.5_total.txt'
        z_label = 'COSMOS 4.5<z<5.5'
    elif redshift < 6.0:
        filename = './data/COSMOS2020/SMF_Farmer_v2.1_5.5z6.5_total.txt'
        z_label = 'COSMOS 5.5<z<6.5'
    elif redshift < 7.0:
        filename = './data/COSMOS2020/SMF_Farmer_v2.1_6.5z7.5_total.txt'
        z_label = 'COSMOS 6.5<z<7.5'
    else:
        return None, None, None, None
    
    if not os.path.exists(filename):
        return None, None, None, None
    
    # Load data: columns are log10(M*), bin_width, phi, phi_err_low, phi_err_high
    data = np.loadtxt(filename)
    log_mass = data[:, 0]
    phi = np.log10(data[:, 2])  # Convert to log scale
    phi_err_low = np.log10(data[:, 3])
    phi_err_high = np.log10(data[:, 4])
    
    return log_mass, phi, phi_err_low, phi_err_high

def load_bagpipes_smf(redshift):
    """Load Bagpipes (Harvey+24) observational SMF data for a given redshift"""
    filename = './data/FiducialBagpipesGSMF.ecsv'
    
    if not os.path.exists(filename):
        return None, None, None, None
    
    try:
        # Read the ECSV file
        table = Table.read(filename, format='ascii.ecsv')
        
        # Filter for redshift closest to target (within ±1.5)
        z_match = np.abs(table['z'] - redshift) < 1.5
        if not np.any(z_match):
            return None, None, None, None
        
        data = table[z_match]
        
        log_mass = np.array(data['log10Mstar'])
        phi = np.log10(np.array(data['phi']))
        phi_err_low = np.log10(np.array(data['phi_error_low']))
        phi_err_high = np.log10(np.array(data['phi_error_upp']))
        
        return log_mass, phi, phi_err_low, phi_err_high
    except Exception as e:
        print(f"  Warning: Could not load Bagpipes data: {e}")
        return None, None, None, None

def plot_smf_grid(models=None, redshift_range='high'):
    """Plot SMF grid for different redshifts comparing different FFB models
    
    Args:
        models: List of model dictionaries to plot. If None, uses PLOT_MODELS.
        redshift_range: 'low' for z=0-4 or 'high' for z=6-13
    """
    
    if models is None:
        models = PLOT_MODELS
    
    print('\n' + '='*60)
    print(f'Creating SMF Grid Plot ({redshift_range}-z)')
    print('='*60)
    
    # Define redshifts and corresponding snapshots for the grid
    # 2 rows x 3 columns = 6 plots
    target_redshifts = [6.0, 7.0, 8.0, 9.0, 10.5, 12.5]
    output_suffix = ''
    
    # Find closest snapshots to target redshifts
    snapshots = []
    actual_redshifts = []
    for target_z in target_redshifts:
        # Find closest redshift in DEFAULT_REDSHIFTS
        idx = np.argmin(np.abs(np.array(DEFAULT_REDSHIFTS) - target_z))
        snapshots.append(f'Snap_{idx}')
        actual_redshifts.append(DEFAULT_REDSHIFTS[idx])
        print(f'z={target_z:.1f} -> Snap_{idx} (z={DEFAULT_REDSHIFTS[idx]:.3f})')
    
    # Create figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    # fig.suptitle('Stellar Mass Function at High Redshift', fontsize=16, y=0.995)
    
    # Flatten axes for easier iteration
    axes_flat = axes.flatten()
    
    # Loop through each subplot
    for idx, (snapshot, z_actual, ax) in enumerate(zip(snapshots, actual_redshifts, axes_flat)):
        print(f'\nProcessing {snapshot} (z={z_actual:.3f})...')
        
        # Plot each model
        for model in models:
            model_dir = model['dir']
            filename = 'model_0.hdf5'
            
            # Check if file exists
            if not os.path.exists(model_dir + filename):
                print(f"  Warning: {model_dir + filename} not found, skipping {model['name']}")
                continue
            
            # Read data
            hubble_h = model['hubble_h']
            stellar_mass = read_hdf_from_model(model_dir, filename, snapshot, 'StellarMass', hubble_h) * 1.0e10 / hubble_h
            sfr_disk = read_hdf_from_model(model_dir, filename, snapshot, 'SfrDisk', hubble_h)
            sfr_bulge = read_hdf_from_model(model_dir, filename, snapshot, 'SfrBulge', hubble_h)
            
            # Calculate volume
            volume = (model['boxsize'] / hubble_h)**3.0 * model.get('volume_fraction', 1.0)
            
            # Calculate SMF with errors
            mass_bins, smf, smf_lower, smf_upper = calculate_smf(stellar_mass, sfr_disk, sfr_bulge, volume, hubble_h, return_errors=True)
            
            # Plot line
            valid = smf > -9  # Only plot non-zero bins
            ax.plot(mass_bins[valid], smf[valid], 
                   color=model['color'], 
                   linestyle=model['linestyle'],
                   linewidth=model['linewidth'],
                   label=model['name'],
                   alpha=0.8,
                   zorder=3)
            
            # Add error shading
            ax.fill_between(mass_bins[valid],
                           smf_lower[valid],
                           smf_upper[valid],
                           color=model['color'],
                           alpha=0.2,
                           zorder=2)
            
            print(f"  {model['name']}: {len(stellar_mass[stellar_mass>0])} galaxies")
        
        # Add observational data
        # COSMOS2020 (Farmer+)
        cosmos_mass, cosmos_phi, cosmos_err_low, cosmos_err_high = load_cosmos2020_smf(z_actual)
        if cosmos_mass is not None:
            # Filter out invalid values
            valid = np.isfinite(cosmos_phi) & (cosmos_phi > -9)
            if np.any(valid):
                # Calculate error bars (already in log space)
                yerr_low = np.abs(cosmos_phi[valid] - cosmos_err_low[valid])
                yerr_high = np.abs(cosmos_err_high[valid] - cosmos_phi[valid])
                ax.errorbar(cosmos_mass[valid], cosmos_phi[valid], 
                           yerr=[yerr_low, yerr_high],
                           fmt='s', color='black', markersize=10, alpha=1.0,
                           label='Farmer+ (COSMOS2020)' if idx == 0 else '', capsize=2, linewidth=1.5)
                print(f"  Farmer+ (COSMOS2020) data added")
        
        # Bagpipes (Harvey+24) - for high redshifts
        if z_actual >= 6.0:
            bagpipes_mass, bagpipes_phi, bagpipes_err_low, bagpipes_err_high = load_bagpipes_smf(z_actual)
            if bagpipes_mass is not None:
                valid = np.isfinite(bagpipes_phi) & (bagpipes_phi > -9)
                if np.any(valid):
                    # Calculate error bars (already in log space)
                    yerr_low = np.abs(bagpipes_phi[valid] - bagpipes_err_low[valid])
                    yerr_high = np.abs(bagpipes_err_high[valid] - bagpipes_phi[valid])
                    ax.errorbar(bagpipes_mass[valid], bagpipes_phi[valid],
                               yerr=[yerr_low, yerr_high],
                               fmt='D', color='black', markersize=10, alpha=1.0,
                               label='Harvey+24' if idx == 0 else '', capsize=2, linewidth=1.5)
                    print(f"  Harvey+24 (Bagpipes) data added")
        
        # Formatting
        ax.set_xlim(8, 12)
        ax.set_ylim(-7, -1)
        # ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add redshift text to upper right without box
        ax.text(0.95, 0.95, f'z = {z_actual:.2f}', 
               transform=ax.transAxes, 
               verticalalignment='top',
               horizontalalignment='right',
               fontsize=14)
        
        # Only show x-axis label for bottom row
        if idx >= 3:  # Bottom row (indices 3, 4, 5)
            ax.set_xlabel(r'$\log_{10}(M_* / M_{\odot})$', fontsize=18)
        else:  # Top row - hide x-axis tick labels
            ax.tick_params(axis='x', labelbottom=False)
        
        # Only show legend in first subplot
        if idx == 0:
            ax.legend(loc='lower left', fontsize=9, ncol=1, frameon=False)
    
    # Add common y-axis label before tight_layout
    fig.text(0.04, 0.5, r'$\log_{10}(\Phi / \mathrm{Mpc}^{-3} \, \mathrm{dex}^{-1})$', 
             va='center', rotation='vertical', fontsize=20)
    
    plt.tight_layout(rect=[0.05, 0, 1, 1])
    
    # Save figure
    OutputDir = DirName + 'plots/'
    if not os.path.exists(OutputDir):
        os.makedirs(OutputDir)
    
    output_path = OutputDir + 'smf_grid' + OutputFormat
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'\nPlot saved to: {output_path}')
    plt.close()
    
    print('='*60 + '\n')

def plot_smf_vs_redshift(models=None):
    """Plot stellar mass function vs redshift for two mass thresholds
    
    Args:
        models: List of model dictionaries to plot. If None, uses PLOT_MODELS.
    """
    
    if models is None:
        models = PLOT_MODELS
    
    print('\n' + '='*60)
    print('Creating SMF vs Redshift Plot')
    print('='*60)
    
    # Define redshift range
    target_redshifts = np.arange(5, 17, 1)  # z=5 to z=16
    
    # Find closest snapshots
    snapshots = []
    actual_redshifts = []
    for target_z in target_redshifts:
        idx = np.argmin(np.abs(np.array(DEFAULT_REDSHIFTS) - target_z))
        snapshots.append(f'Snap_{idx}')
        actual_redshifts.append(DEFAULT_REDSHIFTS[idx])
    
    print(f'Analyzing {len(snapshots)} redshift snapshots from z={min(actual_redshifts):.1f} to z={max(actual_redshifts):.1f}')
    
    # Mass thresholds
    mass_thresholds = [1e9, 1e10]
    
    # Create figure with 1x2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loop through each mass threshold
    for ax_idx, (ax, mass_threshold) in enumerate(zip(axes, mass_thresholds)):
        print(f'\nProcessing mass threshold > {mass_threshold:.0e} Msun...')
        
        # Store data for each model
        for model in models:
            model_dir = model['dir']
            filename = 'model_0.hdf5'
            hubble_h = model['hubble_h']
            
            # Check if file exists
            if not os.path.exists(model_dir + filename):
                print(f"  Warning: {model_dir + filename} not found, skipping {model['name']}")
                continue
            
            # Arrays to store number density at each redshift
            redshifts_data = []
            number_density = []
            number_density_lower = []
            number_density_upper = []
            
            # Loop through snapshots
            for snapshot, z_actual in zip(snapshots, actual_redshifts):
                # Read data
                stellar_mass = read_hdf_from_model(model_dir, filename, snapshot, 'StellarMass', hubble_h) * 1.0e10 / hubble_h
                
                # Count galaxies above threshold
                n_above_threshold = np.sum(stellar_mass > mass_threshold)
                
                # Calculate volume
                volume = (model['boxsize'] / hubble_h)**3.0 * model.get('volume_fraction', 1.0)
                
                # Calculate number density (Mpc^-3) with Poisson errors
                n_density = n_above_threshold / volume
                
                # Poisson errors: N ± sqrt(N)
                n_upper = (n_above_threshold + np.sqrt(n_above_threshold)) / volume
                n_lower = max((n_above_threshold - np.sqrt(n_above_threshold)), 1) / volume
                
                if n_density > 0:
                    redshifts_data.append(z_actual)
                    number_density.append(np.log10(n_density))
                    number_density_upper.append(np.log10(n_upper))
                    number_density_lower.append(np.log10(n_lower))
            
            # Plot line
            if len(redshifts_data) > 0:
                ax.plot(redshifts_data, number_density,
                       color=model['color'],
                       linestyle=model['linestyle'],
                       linewidth=model['linewidth'],
                       label=model['name'],
                       alpha=0.8,
                       zorder=3)
                # Add error shading
                ax.fill_between(redshifts_data,
                               number_density_lower,
                               number_density_upper,
                               color=model['color'],
                               alpha=0.2,
                               zorder=2)
                print(f"  {model['name']}: plotted")
        
        # Formatting
        ax.set_xlim(5, 16)
        ax.set_xlabel('Redshift z', fontsize=12)
        # Don't invert x-axis (lower redshift on the left, higher on right)
        
        # Add mass threshold text in upper right corner
        ax.text(0.95, 0.95, rf'$M_* > 10^{{{int(np.log10(mass_threshold))}}}$ $M_\odot$',
               transform=ax.transAxes,
               verticalalignment='top',
               horizontalalignment='right',
               fontsize=14)
        
        # Only show y-axis label on left plot
        if ax_idx == 0:
            ax.set_ylabel(r'$\log_{10}(n / \mathrm{Mpc}^{-3})$', fontsize=12)
        else:
            # Remove y-axis tick labels from right plot
            ax.tick_params(axis='y', labelleft=False)
        
        # Only show legend on first subplot in lower left
        if ax_idx == 0:
            ax.legend(loc='lower left', fontsize=10, frameon=False)
    
    plt.tight_layout()
    
    # Save figure
    OutputDir = DirName + 'plots/'
    if not os.path.exists(OutputDir):
        os.makedirs(OutputDir)
    
    output_path = OutputDir + 'smf_vs_redshift' + OutputFormat
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'\nPlot saved to: {output_path}')
    plt.close()
    
    print('='*60 + '\n')

def calculate_uv_luminosity_function(stellar_mass, sfr_disk, sfr_bulge, volume, hubble_h, binwidth=0.5):
    """Calculate UV luminosity function with Poisson errors
    
    Uses conversion from SFR to UV luminosity following Kennicutt (1998) and Madau & Dickinson (2014):
    L_UV(1500Å) = 1.4e-28 * SFR [erg/s/Hz] where SFR is in Msun/yr
    
    Converting to absolute magnitude:
    M_UV = 51.63 - 2.5*log10(L_UV[erg/s/Hz])
    
    Simplified relation: M_UV ≈ -2.5*log10(SFR[Msun/yr]) - 17.88
    
    Returns: M_UV bins, log(phi), log(phi_lower), log(phi_upper)
    """
    # Total star formation rate
    sfr_total = sfr_disk + sfr_bulge
    
    # Select galaxies with SFR > 0
    w = np.where(sfr_total > 0.0)[0]
    if len(w) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    # Convert SFR to UV magnitude at 1500Å
    # M_UV = -2.5*log10(SFR) - 17.88 (Kennicutt 1998, converted to AB mag)
    M_UV = -2.5 * np.log10(sfr_total[w]) - 17.88
    
    # Create histogram in UV magnitude space
    mi = np.floor(M_UV.min()) - 2
    ma = np.ceil(M_UV.max()) + 2
    NB = int((ma - mi) / binwidth)
    (counts, binedges) = np.histogram(M_UV, range=(mi, ma), bins=NB)
    xaxeshisto = binedges[:-1] + 0.5 * binwidth
    
    # Convert to number density (Mpc^-3 mag^-1)
    uvlf = counts / volume / binwidth
    
    # Calculate Poisson errors (1 sigma)
    # Upper error: N + sqrt(N)
    # Lower error: N - sqrt(N) (but at least 0)
    counts_upper = counts + np.sqrt(counts)
    counts_lower = np.maximum(counts - np.sqrt(counts), 0.1)  # Avoid zero
    
    uvlf_upper = counts_upper / volume / binwidth
    uvlf_lower = counts_lower / volume / binwidth
    
    # Convert to log10
    uvlf_log = np.zeros(len(uvlf))
    uvlf_log_upper = np.zeros(len(uvlf))
    uvlf_log_lower = np.zeros(len(uvlf))
    
    w_nonzero = np.where(uvlf > 0)[0]
    uvlf_log[w_nonzero] = np.log10(uvlf[w_nonzero])
    uvlf_log_upper[w_nonzero] = np.log10(uvlf_upper[w_nonzero])
    uvlf_log_lower[w_nonzero] = np.log10(uvlf_lower[w_nonzero])
    
    uvlf_log[uvlf == 0] = -10  # Placeholder for zero values
    uvlf_log_upper[uvlf == 0] = -10
    uvlf_log_lower[uvlf == 0] = -10
    
    return xaxeshisto, uvlf_log, uvlf_log_lower, uvlf_log_upper

def plot_uvlf_grid(models=None):
    """Plot UV Luminosity Function grid for different redshifts
    
    Args:
        models: List of model dictionaries to plot. If None, uses PLOT_MODELS.
    """
    
    if models is None:
        models = PLOT_MODELS
    
    print('\n' + '='*60)
    print('Creating UV Luminosity Function Grid Plot')
    print('='*60)
    
    # Define redshifts for the grid (2 rows x 3 columns = 6 plots)
    target_redshifts = [9.0, 10.0, 11.0, 12.0, 13.0, 16.0]
    
    # Find closest snapshots to target redshifts
    snapshots = []
    actual_redshifts = []
    for target_z in target_redshifts:
        idx = np.argmin(np.abs(np.array(DEFAULT_REDSHIFTS) - target_z))
        snapshots.append(f'Snap_{idx}')
        actual_redshifts.append(DEFAULT_REDSHIFTS[idx])
        print(f'z={target_z:.1f} -> Snap_{idx} (z={DEFAULT_REDSHIFTS[idx]:.3f})')
    
    # Create figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Flatten axes for easier iteration
    axes_flat = axes.flatten()
    
    # Loop through each subplot
    for idx, (snapshot, z_actual, ax) in enumerate(zip(snapshots, actual_redshifts, axes_flat)):
        print(f'\nProcessing {snapshot} (z={z_actual:.3f})...')
        
        # Plot each model
        for model in models:
            model_dir = model['dir']
            filename = 'model_0.hdf5'
            
            # Check if file exists
            if not os.path.exists(model_dir + filename):
                print(f"  Warning: {model_dir + filename} not found, skipping {model['name']}")
                continue
            
            # Read data
            hubble_h = model['hubble_h']
            stellar_mass = read_hdf_from_model(model_dir, filename, snapshot, 'StellarMass', hubble_h) * 1.0e10 / hubble_h
            sfr_disk = read_hdf_from_model(model_dir, filename, snapshot, 'SfrDisk', hubble_h)
            sfr_bulge = read_hdf_from_model(model_dir, filename, snapshot, 'SfrBulge', hubble_h)
            
            # Calculate volume
            volume = (model['boxsize'] / hubble_h)**3.0 * model.get('volume_fraction', 1.0)
            
            # Calculate UVLF with errors
            M_UV_bins, uvlf, uvlf_lower, uvlf_upper = calculate_uv_luminosity_function(stellar_mass, sfr_disk, sfr_bulge, volume, hubble_h)
            
            # Plot
            valid = uvlf > -9  # Only plot non-zero bins
            
            # Plot line
            ax.plot(M_UV_bins[valid], uvlf[valid], 
                   color=model['color'], 
                   linestyle=model['linestyle'],
                   linewidth=model['linewidth'],
                   label=model['name'],
                   alpha=0.8,
                   zorder=3)
            
            # Add error shading
            ax.fill_between(M_UV_bins[valid], 
                           uvlf_lower[valid], 
                           uvlf_upper[valid],
                           color=model['color'],
                           alpha=0.2,
                           zorder=2)
            
            print(f"  {model['name']}: {len(stellar_mass[stellar_mass>0])} galaxies")
        
        # Formatting
        ax.set_xlim(-24, -16)
        ax.invert_xaxis()  # Flip x-axis (brighter on right, fainter on left)
        ax.set_ylim(-8, -2)
        
        # Add redshift text to upper right without box
        ax.text(0.95, 0.95, f'z = {z_actual:.2f}', 
               transform=ax.transAxes, 
               verticalalignment='top',
               horizontalalignment='right',
               fontsize=14)
        
        # Only show x-axis label for bottom row
        if idx >= 3:  # Bottom row (indices 3, 4, 5)
            ax.set_xlabel(r'$M_{\mathrm{UV}}$', fontsize=11)
        else:  # Top row - hide x-axis tick labels
            ax.tick_params(axis='x', labelbottom=False)
        
        # Only show legend in first subplot
        if idx == 0:
            ax.legend(loc='lower left', fontsize=9, ncol=1)
    
    # Add common y-axis label before tight_layout
    fig.text(0.04, 0.5, r'$\log_{10}(\Phi / \mathrm{Mpc}^{-3} \, \mathrm{mag}^{-1})$', 
             va='center', rotation='vertical', fontsize=20)
    
    plt.tight_layout(rect=[0.05, 0, 1, 1])
    
    # Save figure
    OutputDir = DirName + 'plots/'
    if not os.path.exists(OutputDir):
        os.makedirs(OutputDir)
    
    output_path = OutputDir + 'uvlf_grid' + OutputFormat
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'\nPlot saved to: {output_path}')
    plt.close()
    
    print('='*60 + '\n')

def plot_uvlf_vs_redshift(models=None):
    """Plot UV luminosity function vs redshift for two M_UV thresholds
    
    Args:
        models: List of model dictionaries to plot. If None, uses PLOT_MODELS.
    """
    
    if models is None:
        models = PLOT_MODELS
    
    print('\n' + '='*60)
    print('Creating UVLF vs Redshift Plot')
    print('='*60)
    
    # Define redshift range
    target_redshifts = np.arange(5, 17, 1)  # z=5 to z=16
    
    # Find closest snapshots
    snapshots = []
    actual_redshifts = []
    for target_z in target_redshifts:
        idx = np.argmin(np.abs(np.array(DEFAULT_REDSHIFTS) - target_z))
        snapshots.append(f'Snap_{idx}')
        actual_redshifts.append(DEFAULT_REDSHIFTS[idx])
    
    print(f'Analyzing {len(snapshots)} redshift snapshots from z={min(actual_redshifts):.1f} to z={max(actual_redshifts):.1f}')
    
    # M_UV thresholds
    M_UV_thresholds = [-17, -20]
    
    # Create figure with 1x2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loop through each M_UV threshold
    for ax_idx, (ax, M_UV_threshold) in enumerate(zip(axes, M_UV_thresholds)):
        print(f'\nProcessing M_UV < {M_UV_threshold}...')
        
        # Store data for each model
        for model in models:
            model_dir = model['dir']
            filename = 'model_0.hdf5'
            hubble_h = model['hubble_h']
            
            # Check if file exists
            if not os.path.exists(model_dir + filename):
                print(f"  Warning: {model_dir + filename} not found, skipping {model['name']}")
                continue
            
            # Arrays to store data at each redshift
            redshifts_data = []
            phi_values = []
            phi_values_lower = []
            phi_values_upper = []
            
            # Loop through snapshots
            for snapshot, z_actual in zip(snapshots, actual_redshifts):
                # Read data
                stellar_mass = read_hdf_from_model(model_dir, filename, snapshot, 'StellarMass', hubble_h) * 1.0e10 / hubble_h
                sfr_disk = read_hdf_from_model(model_dir, filename, snapshot, 'SfrDisk', hubble_h)
                sfr_bulge = read_hdf_from_model(model_dir, filename, snapshot, 'SfrBulge', hubble_h)
                
                # Calculate volume
                volume = (model['boxsize'] / hubble_h)**3.0 * model.get('volume_fraction', 1.0)
                
                # Calculate total SFR and convert to M_UV
                sfr_total = sfr_disk + sfr_bulge
                w = np.where(sfr_total > 0.0)[0]
                if len(w) > 0:
                    # Convert SFR to UV magnitude using Kennicutt (1998)
                    M_UV = -2.5 * np.log10(sfr_total[w]) - 17.88
                    
                    # Count galaxies brighter than threshold (M_UV < threshold, i.e., more negative)
                    n_brighter = np.sum(M_UV < M_UV_threshold)
                    
                    # Calculate number density (Mpc^-3) with Poisson errors
                    n_density = n_brighter / volume
                    
                    # Poisson errors: N ± sqrt(N)
                    n_upper = (n_brighter + np.sqrt(n_brighter)) / volume
                    n_lower = max((n_brighter - np.sqrt(n_brighter)), 1) / volume
                    
                    if n_density > 0:
                        redshifts_data.append(z_actual)
                        phi_values.append(np.log10(n_density))
                        phi_values_upper.append(np.log10(n_upper))
                        phi_values_lower.append(np.log10(n_lower))
            
            # Plot line
            if len(redshifts_data) > 0:
                ax.plot(redshifts_data, phi_values,
                       color=model['color'],
                       linestyle=model['linestyle'],
                       linewidth=model['linewidth'],
                       label=model['name'],
                       alpha=0.8,
                       zorder=3)
                # Add error shading
                ax.fill_between(redshifts_data,
                               phi_values_lower,
                               phi_values_upper,
                               color=model['color'],
                               alpha=0.2,
                               zorder=2)
                print(f"  {model['name']}: plotted")
        
        # Formatting
        ax.set_xlim(5, 16)
        ax.set_ylim(-5, -1.5)
        
        ax.set_xlabel('Redshift z', fontsize=12)
        
        # Add M_UV threshold text in upper right corner
        ax.text(0.95, 0.95, rf'$M_{{\mathrm{{UV}}}} < {M_UV_threshold}$',
               transform=ax.transAxes,
               verticalalignment='top',
               horizontalalignment='right',
               fontsize=14)
        
        # Only show y-axis label on left plot
        if ax_idx == 0:
            ax.set_ylabel(r'$\log_{10}(n / \mathrm{Mpc}^{-3})$', fontsize=12)
        else:
            # Remove y-axis tick labels from right plot
            ax.tick_params(axis='y', labelleft=False)
        
        # Only show legend on first subplot in lower left
        if ax_idx == 0:
            ax.legend(loc='lower left', fontsize=10, frameon=False)
    
    plt.tight_layout()
    
    # Save figure
    OutputDir = DirName + 'plots/'
    if not os.path.exists(OutputDir):
        os.makedirs(OutputDir)
    
    output_path = OutputDir + 'uvlf_vs_redshift' + OutputFormat
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'\nPlot saved to: {output_path}')
    plt.close()
    
    print('='*60 + '\n')

def convert_MUV_to_mF277W(M_UV, redshift):
    """Convert absolute UV magnitude to apparent F277W magnitude
    
    Uses proper cosmological luminosity distance and K-correction.
    F277W filter (2.77 μm) at high-z samples rest-frame optical/UV light.
    
    At z~6-10, F277W observes rest-frame ~0.35-0.55 μm (optical/UV)
    
    Args:
        M_UV: Absolute UV magnitude at 1500Å rest-frame
        redshift: Redshift
    
    Returns:
        m_F277W: Apparent F277W magnitude
    """
    # Cosmological parameters (should match simulation)
    H0 = 73.0  # km/s/Mpc (MILLENNIUM_HUBBLE_H * 100)
    Om0 = 0.25  # Matter density
    OmL = 0.75  # Dark energy density
    c = 299792.458  # km/s
    
    # Calculate proper luminosity distance using Ω_m and Ω_Λ
    # For flat universe: d_L = (c/H0) * (1+z) * integral[dz'/E(z')]
    # E(z) = sqrt(Ω_m(1+z)^3 + Ω_Λ)
    # Approximate integral for high-z:
    
    z = redshift
    # Use integral approximation (good to ~5% for z<20)
    E_z = lambda zp: np.sqrt(Om0 * (1 + zp)**3 + OmL)
    
    # Trapezoidal rule integration
    n_steps = 100
    z_array = np.linspace(0, z, n_steps)
    integrand = 1.0 / E_z(z_array)
    d_c = (c / H0) * np.trapezoid(integrand, z_array)  # Comoving distance
    d_L = d_c * (1 + z)  # Luminosity distance
    
    # Distance modulus
    dist_mod = 5 * np.log10(d_L * 1e6) - 5  # d_L in pc
    
    # K-correction: F277W at observed frame samples different rest-frame wavelength
    # At redshift z, F277W (2.77 μm) observes rest-frame λ_rest = 2770 / (1+z) nm
    # For z=6-10: samples 395-252 nm (UV/optical boundary)
    # For z=10-15: samples 252-173 nm (deep UV)
    
    # Approximate K-correction based on stellar population SEDs
    # For star-forming galaxies, β ~ -2 (UV slope)
    # K-corr ≈ -2.5*(1+β)*log10(1+z) 
    beta_uv = -2.0  # Typical UV slope for star-forming galaxies
    K_corr = -2.5 * (1 + beta_uv) * np.log10(1 + z)
    
    # Additional color term: rest-frame UV to optical color
    # For young stellar populations: (UV - optical) ≈ -0.3 to 0.0 mag
    # Use simple approximation
    rest_lambda_obs = 2770.0 / (1 + z)  # nm, rest-frame wavelength observed by F277W
    
    # If observing rest-frame < 400 nm, use UV magnitudes directly
    # If observing rest-frame > 400 nm, apply small color correction
    if rest_lambda_obs < 400:
        color_term = 0.0  # Still in UV
    else:
        # Transition to optical: galaxies slightly redder
        color_term = 0.2 * (rest_lambda_obs - 400) / 100  # Gradual reddening
    
    # Convert to apparent magnitude
    m_F277W = M_UV + dist_mod + K_corr + color_term
    
    return m_F277W

def plot_cumulative_surface_density(models=None):
    """Plot cumulative surface density of galaxies vs redshift for three F277W thresholds
    
    Args:
        models: List of model dictionaries to plot. If None, uses PLOT_MODELS.
    """
    
    if models is None:
        models = PLOT_MODELS
    
    print('\n' + '='*60)
    print('Creating Cumulative Surface Density Plot')
    print('='*60)
    
    # Define redshift range
    target_redshifts = np.arange(5, 17, 1)  # z=5 to z=16
    
    # Find closest snapshots
    snapshots = []
    actual_redshifts = []
    for target_z in target_redshifts:
        idx = np.argmin(np.abs(np.array(DEFAULT_REDSHIFTS) - target_z))
        snapshots.append(f'Snap_{idx}')
        actual_redshifts.append(DEFAULT_REDSHIFTS[idx])
    
    print(f'Analyzing {len(snapshots)} redshift snapshots from z={min(actual_redshifts):.1f} to z={max(actual_redshifts):.1f}')
    
    # F277W magnitude thresholds
    m_F277W_thresholds = [28.5, 29.5, 31.5]
    
    # Create figure with 1x3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loop through each F277W threshold
    for ax_idx, (ax, m_threshold) in enumerate(zip(axes, m_F277W_thresholds)):
        print(f'\nProcessing m_F277W < {m_threshold}...')
        
        # Store data for each model
        for model in models:
            model_dir = model['dir']
            filename = 'model_0.hdf5'
            hubble_h = model['hubble_h']
            boxsize = model['boxsize']
            
            # Check if file exists
            if not os.path.exists(model_dir + filename):
                print(f"  Warning: {model_dir + filename} not found, skipping {model['name']}")
                continue
            
            # Arrays to store cumulative surface density at each redshift
            redshifts_data = []
            cumulative_surface_density = []
            
            # Loop through snapshots
            for snapshot, z_actual in zip(snapshots, actual_redshifts):
                # Read data
                stellar_mass = read_hdf_from_model(model_dir, filename, snapshot, 'StellarMass', hubble_h) * 1.0e10 / hubble_h
                sfr_disk = read_hdf_from_model(model_dir, filename, snapshot, 'SfrDisk', hubble_h)
                sfr_bulge = read_hdf_from_model(model_dir, filename, snapshot, 'SfrBulge', hubble_h)
                
                # Convert SFR to M_UV
                sfr_total = sfr_disk + sfr_bulge
                w = np.where(sfr_total > 0.0)[0]
                if len(w) > 0:
                    # Convert SFR to UV magnitude using Kennicutt (1998)
                    M_UV = -2.5 * np.log10(sfr_total[w]) - 17.88
                    
                    # Convert M_UV to m_F277W
                    m_F277W = convert_MUV_to_mF277W(M_UV, z_actual)
                    
                    # Count galaxies brighter than threshold
                    n_brighter = np.sum(m_F277W < m_threshold)
                    
                    # Calculate surface density per unit angular area
                    # Volume is in (h^-1 Mpc)^3, need to convert to surface density
                    # Surface density = N / (box_area in deg^2)
                    # Box area in comoving Mpc^2
                    box_area = (boxsize / hubble_h)**2
                    
                    # Convert to angular area (need angular diameter distance)
                    # Approximation: angular diameter distance d_A ≈ d_L / (1+z)^2
                    c = 299792.458  # km/s
                    H0 = hubble_h * 100
                    if z_actual < 5:
                        d_L = (c / H0) * z_actual * (1 + z_actual / 2)
                    else:
                        d_L = (c / H0) * z_actual * (1 + z_actual * 0.75 / 2)
                    d_A = d_L / (1 + z_actual)**2
                    
                    # Angular size in radians
                    angular_size = boxsize / hubble_h / d_A
                    # Convert to square degrees
                    angular_area_deg2 = (angular_size * 180 / np.pi)**2
                    
                    # Surface density (galaxies per deg^2)
                    surf_dens = n_brighter / angular_area_deg2
                    
                    if surf_dens > 0:
                        redshifts_data.append(z_actual)
                        cumulative_surface_density.append(np.log10(surf_dens))
            
            # Plot
            if len(redshifts_data) > 0:
                ax.plot(redshifts_data, cumulative_surface_density,
                       color=model['color'],
                       linestyle=model['linestyle'],
                       linewidth=model['linewidth'],
                       label=model['name'],
                       alpha=0.8)
                print(f"  {model['name']}: plotted")
        
        # Formatting
        ax.set_xlim(5, 16)
        ax.set_ylim(-2, 3)
        
        ax.set_xlabel('Redshift z', fontsize=12)
        
        # Add threshold text in upper right corner
        ax.text(0.95, 0.95, rf'$m_{{F277W}} < {m_threshold}$',
               transform=ax.transAxes,
               verticalalignment='top',
               horizontalalignment='right',
               fontsize=14)
        
        # Only show y-axis label on left plot
        if ax_idx == 0:
            ax.set_ylabel(r'$\log_{10}(\Sigma / \mathrm{deg}^{-2})$', fontsize=12)
            ax.legend(loc='lower left', fontsize=10, frameon=False)
        else:
            # Remove y-axis tick labels from middle and right plots
            ax.tick_params(axis='y', labelleft=False)
    
    plt.tight_layout()
    
    # Save figure
    OutputDir = DirName + 'plots/'
    if not os.path.exists(OutputDir):
        os.makedirs(OutputDir)
    
    output_path = OutputDir + 'cumulative_surface_density' + OutputFormat
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'\nPlot saved to: {output_path}')
    plt.close()
    
    print('='*60 + '\n')

def plot_density_evolution(models=None):
    """Plot stellar mass density, UV luminosity density, and SFR density vs redshift
    
    Three panels side by side:
    - Left: Stellar mass density for M_UV < -17.5
    - Middle: UV luminosity density for M_UV < -17
    - Right: SFR density for M_UV < -17
    
    All with 1-sigma Poisson error shading
    
    Args:
        models: List of model dictionaries to plot. If None, uses PLOT_MODELS.
    """
    
    if models is None:
        models = PLOT_MODELS
    
    print('\n' + '='*60)
    print('Creating Density Evolution Plot')
    print('='*60)
    
    # Define redshift range
    target_redshifts = np.arange(5, 17, 1)  # z=5 to z=16
    
    # Find closest snapshots
    snapshots = []
    actual_redshifts = []
    for target_z in target_redshifts:
        idx = np.argmin(np.abs(np.array(DEFAULT_REDSHIFTS) - target_z))
        snapshots.append(f'Snap_{idx}')
        actual_redshifts.append(DEFAULT_REDSHIFTS[idx])
    
    print(f'Analyzing {len(snapshots)} redshift snapshots from z={min(actual_redshifts):.1f} to z={max(actual_redshifts):.1f}')
    
    # M_UV thresholds for each panel
    M_UV_threshold_stellar = -17.5
    M_UV_threshold_uv = -17.0
    M_UV_threshold_sfr = -17.0
    
    # Create figure with 1x3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Panel titles
    panel_titles = ['Stellar Mass Density', 'UV Luminosity Density', 'SFR Density']
    
    # Store data for each model
    for model in models:
        model_dir = model['dir']
        filename = 'model_0.hdf5'
        hubble_h = model['hubble_h']
        
        # Check if file exists
        if not os.path.exists(model_dir + filename):
            print(f"  Warning: {model_dir + filename} not found, skipping {model['name']}")
            continue
        
        # Arrays to store density at each redshift - separate redshifts for each panel
        redshifts_stellar = []
        stellar_mass_density = []
        stellar_mass_density_lower = []
        stellar_mass_density_upper = []
        
        redshifts_uv = []
        uv_luminosity_density = []
        uv_luminosity_density_lower = []
        uv_luminosity_density_upper = []
        
        redshifts_sfr = []
        sfr_density = []
        sfr_density_lower = []
        sfr_density_upper = []
        
        # Loop through snapshots
        for snapshot, z_actual in zip(snapshots, actual_redshifts):
            # Read data
            stellar_mass = read_hdf_from_model(model_dir, filename, snapshot, 'StellarMass', hubble_h) * 1.0e10 / hubble_h
            sfr_disk = read_hdf_from_model(model_dir, filename, snapshot, 'SfrDisk', hubble_h)
            sfr_bulge = read_hdf_from_model(model_dir, filename, snapshot, 'SfrBulge', hubble_h)
            
            # Calculate volume
            volume = (model['boxsize'] / hubble_h)**3.0 * model.get('volume_fraction', 1.0)
            
            # Calculate total SFR and M_UV
            sfr_total = sfr_disk + sfr_bulge
            w = np.where(sfr_total > 0.0)[0]
            
            if len(w) > 0:
                # Convert SFR to UV magnitude
                M_UV = -2.5 * np.log10(sfr_total[w]) - 17.88
                
                # PANEL 1: Stellar Mass Density (M_UV < -17.5)
                w_bright = np.where(M_UV < M_UV_threshold_stellar)[0]
                if len(w_bright) > 0:
                    stellar_mass_selected = stellar_mass[w[w_bright]]
                    total_stellar_mass = np.sum(stellar_mass_selected)
                    rho_stellar = total_stellar_mass / volume  # Msun/Mpc^3
                    
                    # Poisson errors: sqrt(N) galaxies
                    N_gal = len(w_bright)
                    N_upper = N_gal + np.sqrt(N_gal)
                    N_lower = max(N_gal - np.sqrt(N_gal), 1)
                    
                    # Scale mass proportionally
                    rho_stellar_upper = rho_stellar * N_upper / N_gal
                    rho_stellar_lower = rho_stellar * N_lower / N_gal
                    
                    if rho_stellar > 0:
                        redshifts_stellar.append(z_actual)
                        stellar_mass_density.append(np.log10(rho_stellar))
                        stellar_mass_density_upper.append(np.log10(rho_stellar_upper))
                        stellar_mass_density_lower.append(np.log10(rho_stellar_lower))
                
                # PANEL 2: UV Luminosity Density (M_UV < -17)
                w_bright_uv = np.where(M_UV < M_UV_threshold_uv)[0]
                if len(w_bright_uv) > 0:
                    # L_UV = 10^((51.63 - M_UV) / 2.5) [erg/s/Hz]
                    L_UV = 10**((51.63 - M_UV[w_bright_uv]) / 2.5)
                    total_L_UV = np.sum(L_UV)
                    rho_UV = total_L_UV / volume  # erg/s/Hz/Mpc^3
                    
                    # Poisson errors
                    N_gal = len(w_bright_uv)
                    N_upper = N_gal + np.sqrt(N_gal)
                    N_lower = max(N_gal - np.sqrt(N_gal), 1)
                    
                    rho_UV_upper = rho_UV * N_upper / N_gal
                    rho_UV_lower = rho_UV * N_lower / N_gal
                    
                    if rho_UV > 0:
                        redshifts_uv.append(z_actual)
                        uv_luminosity_density.append(np.log10(rho_UV))
                        uv_luminosity_density_upper.append(np.log10(rho_UV_upper))
                        uv_luminosity_density_lower.append(np.log10(rho_UV_lower))
                
                # PANEL 3: SFR Density (M_UV < -17)
                w_bright_sfr = np.where(M_UV < M_UV_threshold_sfr)[0]
                if len(w_bright_sfr) > 0:
                    sfr_selected = sfr_total[w[w_bright_sfr]]
                    total_sfr = np.sum(sfr_selected)
                    rho_SFR = total_sfr / volume  # Msun/yr/Mpc^3
                    
                    # Poisson errors
                    N_gal = len(w_bright_sfr)
                    N_upper = N_gal + np.sqrt(N_gal)
                    N_lower = max(N_gal - np.sqrt(N_gal), 1)
                    
                    rho_SFR_upper = rho_SFR * N_upper / N_gal
                    rho_SFR_lower = rho_SFR * N_lower / N_gal
                    
                    if rho_SFR > 0:
                        redshifts_sfr.append(z_actual)
                        sfr_density.append(np.log10(rho_SFR))
                        sfr_density_upper.append(np.log10(rho_SFR_upper))
                        sfr_density_lower.append(np.log10(rho_SFR_lower))
        
        # Plot Panel 1: Stellar Mass Density
        if len(redshifts_stellar) > 0 and len(stellar_mass_density) > 0:
            axes[0].plot(redshifts_stellar, stellar_mass_density,
                        color=model['color'],
                        linestyle=model['linestyle'],
                        linewidth=model['linewidth'],
                        label=model['name'],
                        alpha=0.8,
                        zorder=3)
            axes[0].fill_between(redshifts_stellar,
                                stellar_mass_density_lower,
                                stellar_mass_density_upper,
                                color=model['color'],
                                alpha=0.2,
                                zorder=2)
            print(f"  {model['name']}: Stellar mass density plotted")
        
        # Plot Panel 2: UV Luminosity Density
        if len(redshifts_uv) > 0 and len(uv_luminosity_density) > 0:
            axes[1].plot(redshifts_uv, uv_luminosity_density,
                        color=model['color'],
                        linestyle=model['linestyle'],
                        linewidth=model['linewidth'],
                        label=model['name'],
                        alpha=0.8,
                        zorder=3)
            axes[1].fill_between(redshifts_uv,
                                uv_luminosity_density_lower,
                                uv_luminosity_density_upper,
                                color=model['color'],
                                alpha=0.2,
                                zorder=2)
            print(f"  {model['name']}: UV luminosity density plotted")
        
        # Plot Panel 3: SFR Density
        if len(redshifts_sfr) > 0 and len(sfr_density) > 0:
            axes[2].plot(redshifts_sfr, sfr_density,
                        color=model['color'],
                        linestyle=model['linestyle'],
                        linewidth=model['linewidth'],
                        label=model['name'],
                        alpha=0.8,
                        zorder=3)
            axes[2].fill_between(redshifts_sfr,
                                sfr_density_lower,
                                sfr_density_upper,
                                color=model['color'],
                                alpha=0.2,
                                zorder=2)
            print(f"  {model['name']}: SFR density plotted")
    
    # Formatting for each panel
    for idx, ax in enumerate(axes):
        ax.set_xlim(5, 16)
        ax.set_xlabel('Redshift z', fontsize=12)
        
        # Add threshold text in upper right
        if idx == 0:
            ax.text(0.95, 0.95, rf'$M_{{\mathrm{{UV}}}} < {M_UV_threshold_stellar}$',
                   transform=ax.transAxes,
                   verticalalignment='top',
                   horizontalalignment='right',
                   fontsize=12)
            ax.set_ylabel(r'$\log_{10}(\rho_* / M_{\odot} \, \mathrm{Mpc}^{-3})$', fontsize=12)
            ax.legend(loc='lower left', fontsize=10, frameon=False)
        elif idx == 1:
            ax.text(0.95, 0.95, rf'$M_{{\mathrm{{UV}}}} < {M_UV_threshold_uv}$',
                   transform=ax.transAxes,
                   verticalalignment='top',
                   horizontalalignment='right',
                   fontsize=12)
            ax.set_ylabel(r'$\log_{10}(\rho_{\mathrm{UV}} / \mathrm{erg} \, \mathrm{s}^{-1} \, \mathrm{Hz}^{-1} \, \mathrm{Mpc}^{-3})$', fontsize=12)
        else:
            ax.text(0.95, 0.95, rf'$M_{{\mathrm{{UV}}}} < {M_UV_threshold_sfr}$',
                   transform=ax.transAxes,
                   verticalalignment='top',
                   horizontalalignment='right',
                   fontsize=12)
            ax.set_ylabel(r'$\log_{10}(\rho_{\mathrm{SFR}} / M_{\odot} \, \mathrm{yr}^{-1} \, \mathrm{Mpc}^{-3})$', fontsize=12)
    
    plt.tight_layout()
    
    # Save figure
    OutputDir = DirName + 'plots/'
    if not os.path.exists(OutputDir):
        os.makedirs(OutputDir)
    
    output_path = OutputDir + 'density_evolution' + OutputFormat
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'\nPlot saved to: {output_path}')
    plt.close()
    
    print('='*60 + '\n')

def calculate_ffb_threshold_mass(z, hubble_h):
    """Calculate FFB threshold mass from Li et al. 2024
    
    M_v,ffb / 10^10.8 M_sun ~ ((1+z)/10)^-6.2
    
    Args:
        z: Redshift
        hubble_h: Hubble parameter (not used, kept for API compatibility)
    
    Returns:
        M_vir threshold in physical Msun
    """
    z_norm = (1.0 + z) / 10.0
    M_norm = 10.8
    z_exponent = -6.2
    
    # Calculate log10(M_v,ffb) in units of M_sun (physical)
    log_Mvir_ffb_Msun = M_norm + z_exponent * np.log10(z_norm)
    
    # Return in physical Msun
    return 10.0**log_Mvir_ffb_Msun

def plot_ffb_threshold_analysis(models=None):
    """Plot FFB threshold galaxy properties
    
    Left: Mvir vs redshift for disk galaxies at FFB threshold, colored by half-mass radius
    Right: Half-mass radius evolution for galaxies at FFB threshold (disk and bulge separately)
    
    Args:
        models: List of model dictionaries to plot. If None, uses FFB 50% model only.
    """
    
    print('\n' + '='*60)
    print('Creating FFB Threshold Analysis Plot')
    print('='*60)
    
    # Use only FFB 50% model
    ffb50_model = {
        'name': 'FFB 50%',
        'dir': './output/millennium_FFB50/',
        'color': 'orange',
        'linestyle': '-',
        'linewidth': 3,
        'hubble_h': MILLENNIUM_HUBBLE_H,
        'boxsize': MILLENNIUM_BOXSIZE,
        'volume_fraction': 1.0
    }
    
    # Define redshift range - use ALL snapshots in the z=5-16 range for smoother coverage
    snapshots = []
    actual_redshifts = []
    for idx, z in enumerate(DEFAULT_REDSHIFTS):
        if 4.5 <= z <= 16.5:  # Slightly wider range to ensure coverage
            snapshots.append(f'Snap_{idx}')
            actual_redshifts.append(z)
    
    print(f'Analyzing {len(snapshots)} redshift snapshots from z={min(actual_redshifts):.1f} to z={max(actual_redshifts):.1f}')
    
    model_dir = ffb50_model['dir']
    filename = 'model_0.hdf5'
    hubble_h = ffb50_model['hubble_h']
    
    # Check if file exists
    if not os.path.exists(model_dir + filename):
        print(f"Error: {model_dir + filename} not found!")
        print("This plot requires FFB 50% model output.")
        return
    
    # Create figure with 1x2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Arrays to collect data for left plot (scatter)
    all_mvir = []
    all_redshifts = []
    all_radii = []
    
    # Arrays for right plot (lines)
    redshifts_right = []
    median_disk_radius = []
    median_bulge_radius = []
    disk_radius_lower = []
    disk_radius_upper = []
    bulge_radius_lower = []
    bulge_radius_upper = []
    
    # Loop through snapshots
    for snapshot, z_actual in zip(snapshots, actual_redshifts):
        # Read data
        mvir = read_hdf_from_model(model_dir, filename, snapshot, 'Mvir', hubble_h) * 1.0e10 / hubble_h
        stellar_mass = read_hdf_from_model(model_dir, filename, snapshot, 'StellarMass', hubble_h) * 1.0e10 / hubble_h
        bulge_mass = read_hdf_from_model(model_dir, filename, snapshot, 'BulgeMass', hubble_h) * 1.0e10 / hubble_h
        disk_radius = read_hdf_from_model(model_dir, filename, snapshot, 'DiskRadius', hubble_h)  # Already in Mpc/h
        bulge_radius = read_hdf_from_model(model_dir, filename, snapshot, 'BulgeScaleRadius', hubble_h)  # Already in Mpc/h
        galaxy_type = read_hdf_from_model(model_dir, filename, snapshot, 'Type', hubble_h)
        
        # Calculate FFB threshold mass for this redshift
        M_ffb_threshold = calculate_ffb_threshold_mass(z_actual, hubble_h)
        
        # Left plot: ALL galaxies (using disk half-mass radius)
        # Right plot: Only disk galaxies near FFB threshold
        is_disk = galaxy_type == 0
        has_stellar_mass = stellar_mass > 0
        near_threshold = (mvir > M_ffb_threshold * 0.8) & (mvir < M_ffb_threshold * 1.2)
        
        # For left plot: all galaxies with stellar mass (not just disk galaxies)
        w_all = np.where(has_stellar_mass)[0]
        
        # For right plot: only disk galaxies near FFB threshold
        w = np.where(is_disk & has_stellar_mass & near_threshold)[0]
        
        # Left plot: collect ALL galaxies
        if len(w_all) > 0:
            all_mvir.extend(mvir[w_all])
            all_redshifts.extend([z_actual] * len(w_all))
            # Convert disk scale length to half-mass radius: r_half = 1.68 * r_d
            # Convert from comoving Mpc/h to physical kpc
            # Mpc/h -> kpc/h: *1000, kpc/h -> kpc: /h, comoving -> physical: /(1+z)
            radius_kpc = disk_radius[w_all] * 1.68 * 1000.0 / hubble_h / (1.0 + z_actual)
            all_radii.extend(np.log10(radius_kpc))
        
        # Right plot: calculate median radii for galaxies near FFB threshold
        if len(w) > 5:  # Need at least 5 galaxies
            # Right plot: calculate median radii
            # For disk radius - convert scale length to half-mass radius, then to physical kpc
            valid_disk = disk_radius[w] > 0
            if np.sum(valid_disk) > 0:
                disk_radius_kpc = disk_radius[w[valid_disk]] * 1.68 * 1000.0 / hubble_h / (1.0 + z_actual)
                median_disk = np.median(disk_radius_kpc)
                # Calculate 16th and 84th percentiles for 1-sigma errors
                disk_lower = np.percentile(disk_radius_kpc, 16)
                disk_upper = np.percentile(disk_radius_kpc, 84)
            else:
                median_disk = np.nan
                disk_lower = np.nan
                disk_upper = np.nan
            
            # For bulge radius - convert scale length to half-mass radius, then to physical kpc
            valid_bulge = bulge_radius[w] > 0
            if np.sum(valid_bulge) > 0:
                bulge_radius_kpc = bulge_radius[w[valid_bulge]] * 1.68 * 1000.0 / hubble_h / (1.0 + z_actual)
                median_bulge = np.median(bulge_radius_kpc)
                # Calculate 16th and 84th percentiles for 1-sigma errors
                bulge_lower = np.percentile(bulge_radius_kpc, 16)
                bulge_upper = np.percentile(bulge_radius_kpc, 84)
            else:
                median_bulge = np.nan
                bulge_lower = np.nan
                bulge_upper = np.nan
            
            if not np.isnan(median_disk):
                redshifts_right.append(z_actual)
                median_disk_radius.append(median_disk)
                median_bulge_radius.append(median_bulge if not np.isnan(median_bulge) else 0)
                disk_radius_lower.append(disk_lower)
                disk_radius_upper.append(disk_upper)
                bulge_radius_lower.append(bulge_lower if not np.isnan(bulge_lower) else 0)
                bulge_radius_upper.append(bulge_upper if not np.isnan(bulge_upper) else 0)
            
            print(f"  z={z_actual:.1f}: {len(w)} disk galaxies near FFB threshold (M_ffb={M_ffb_threshold:.2e})")
    
    # LEFT PLOT: 2D histogram heatmap of Mvir vs redshift colored by median disk radius
    if len(all_mvir) > 0:
        # Convert to arrays
        all_redshifts = np.array(all_redshifts)
        all_mvir = np.array(all_mvir)
        all_radii = np.array(all_radii)
        
        # Filter out invalid Mvir values (zeros or negatives)
        valid = all_mvir > 0
        all_redshifts = all_redshifts[valid]
        all_mvir = all_mvir[valid]
        all_radii = all_radii[valid]
        
        log_mvir = np.log10(all_mvir)
        
        # Define bins for 2D histogram
        z_bins = np.linspace(5, 16, 60)  # Finer z spacing for smoother look
        mass_bins = np.linspace(9, 13, 60)  # Finer mass spacing
        
        # Create 2D histogram - calculate median radius in each bin
        # We'll use scipy.stats.binned_statistic_2d
        from scipy.stats import binned_statistic_2d
        from scipy.ndimage import gaussian_filter
        
        # Calculate median log radius in each bin
        median_radius, z_edges, mass_edges, binnumber = binned_statistic_2d(
            all_redshifts, log_mvir, all_radii,
            statistic='median',
            bins=[z_bins, mass_bins]
        )
        
        # Apply Gaussian smoothing to reduce noise and create smooth gradients
        # Need to handle NaN values properly
        mask = ~np.isnan(median_radius)
        if np.sum(mask) > 0:
            # Fill NaN values with nearest non-NaN neighbor for smoothing
            from scipy.ndimage import distance_transform_edt
            ind = distance_transform_edt(~mask, return_distances=False, return_indices=True)
            median_radius_filled = median_radius[tuple(ind)]
            
            # Apply moderate Gaussian smoothing (sigma=1.0 for less aggressive smoothing)
            median_radius_smooth = gaussian_filter(median_radius_filled, sigma=1.0, mode='nearest')
            
            # Create a weight mask to fade out smoothing far from data
            # This prevents artifacts at the edges of the data region
            distances = distance_transform_edt(~mask)
            weight_mask = np.exp(-distances / 3.0)  # Exponential decay
            
            # Blend smoothed and original data based on distance from real data
            median_radius_smooth = (median_radius_smooth * weight_mask + 
                                   median_radius_filled * (1 - weight_mask))
            
            # Keep only regions close to actual data (within ~2 bins)
            median_radius_smooth[distances > 2.5] = np.nan
        else:
            median_radius_smooth = median_radius
        
        # Create meshgrid for plotting - use bin centers for smooth shading
        z_centers = (z_edges[:-1] + z_edges[1:]) / 2
        mass_centers = (mass_edges[:-1] + mass_edges[1:]) / 2
        Z_centers, M_centers = np.meshgrid(z_centers, mass_centers)
        
        # Plot as heatmap with smooth interpolation
        im = axes[0].pcolormesh(z_edges, mass_edges, median_radius_smooth.T,
                               cmap='coolwarm_r',
                               shading='auto',
                               vmin=-1.5, vmax=0.5,
                               rasterized=True)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[0])
        cbar.set_label(r'$\log_{10} R_{\mathrm{half-mass}}$ (kpc)', fontsize=11)
        
        # Plot FFB threshold line
        z_line = np.linspace(5, 16, 100)
        M_ffb_line = [calculate_ffb_threshold_mass(z, hubble_h) for z in z_line]
        log_M_ffb_line = np.log10(M_ffb_line)
        axes[0].plot(z_line, log_M_ffb_line, 'k-', linewidth=3, alpha=0.8, label='FFB threshold')
        
        axes[0].set_xlabel('Redshift z', fontsize=12)
        axes[0].set_ylabel(r'$\log_{10} M_{\mathrm{vir}} \, (M_{\odot} \, h^{-1})$', fontsize=12)
        axes[0].set_xlim(5, 20)
        axes[0].set_ylim(9, 13)
        axes[0].legend(loc='upper right', fontsize=10, frameon=False)
        axes[0].text(0.95, 0.88, 'All Disk Galaxies (FFB 50%)',
                    transform=axes[0].transAxes,
                    verticalalignment='top',
                    horizontalalignment='right',
                    fontsize=11,
                    color='black')
    
    # RIGHT PLOT: Half-mass radius vs redshift
    if len(redshifts_right) > 0:
        # Keep data in linear scale (kpc)
        # Plot disk radius with error band
        axes[1].plot(redshifts_right, median_disk_radius,
                    color='orange',
                    linestyle='-',
                    linewidth=3,
                    label='Disk Half-Mass Radius',
                    alpha=0.8,
                    marker='o',
                    markersize=6,
                    zorder=3)
        
        # Add error shading for disk
        axes[1].fill_between(redshifts_right, disk_radius_lower, disk_radius_upper,
                            color='orange',
                            alpha=0.2,
                            zorder=2)
        
        # Only plot bulge where we have valid data (bulge_radius > 0)
        # Filter to only redshifts with actual bulges
        bulge_mask = np.array([r > 0 for r in median_bulge_radius])
        if np.any(bulge_mask):
            redshifts_bulge = np.array(redshifts_right)[bulge_mask]
            median_bulge_valid = np.array(median_bulge_radius)[bulge_mask]
            bulge_lower_valid = np.array(bulge_radius_lower)[bulge_mask]
            bulge_upper_valid = np.array(bulge_radius_upper)[bulge_mask]
            
            axes[1].plot(redshifts_bulge, median_bulge_valid,
                        color='royalblue',
                        linestyle='-',
                        linewidth=3,
                        label='Bulge Half-Mass Radius',
                        alpha=0.8,
                        marker='s',
                        markersize=6,
                        zorder=3)
            
            # Add error shading for bulge (only where valid)
            axes[1].fill_between(redshifts_bulge, bulge_lower_valid, bulge_upper_valid,
                                color='royalblue',
                                alpha=0.2,
                                zorder=2)
        
        axes[1].set_xlabel('Redshift z', fontsize=12)
        axes[1].set_ylabel(r'$R_{\mathrm{half-mass}}$ (kpc)', fontsize=12)
        axes[1].set_xlim(5, 16)
        axes[1].set_yscale('log')
        axes[1].set_ylim(0.01, 3)
        axes[1].set_yticks([0.01, 0.1, 1.0])
        axes[1].set_yticklabels(['0.01', '0.1', '1.0'])
        axes[1].legend(loc='upper right', fontsize=10, frameon=False)
        axes[1].text(0.05, 0.95, 'Galaxies at FFB Threshold',
                    transform=axes[1].transAxes,
                    verticalalignment='top',
                    fontsize=11)
    
    plt.tight_layout()
    
    # Save figure
    OutputDir = DirName + 'plots/'
    if not os.path.exists(OutputDir):
        os.makedirs(OutputDir)
    
    output_path = OutputDir + 'ffb_threshold_analysis' + OutputFormat
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'\nPlot saved to: {output_path}')
    plt.close()
    
    print('='*60 + '\n')


def plot_ffb_threshold_analysis_empirical():
    """
    Create FFB threshold analysis plot.
    Left: Smoothed 2D heatmap of Mvir vs redshift colored by Physical SAGE Disk Radius.
    Right: Evolution of Half-mass radius (Physical) comparing SAGE vs Theory.
    """
    
    print('\n' + '='*60)
    print('Creating FFB Threshold Analysis Plot (Physical Units + Smoothing)')
    print('='*60)
    
    # Use only FFB 50% model
    target_model_name = 'SAGE26' 
    model_config = next((item for item in MODEL_CONFIGS if item["name"] == target_model_name), MODEL_CONFIGS[0])
    
    # Define redshift range
    snapshots = []
    actual_redshifts = []
    for idx, z in enumerate(DEFAULT_REDSHIFTS):
        if 4.5 <= z <= 20.0:
            snapshots.append(f'Snap_{idx}')
            actual_redshifts.append(z)
    
    model_dir = model_config['dir']
    filename = 'model_0.hdf5'
    hubble_h = model_config['hubble_h']
    
    if not os.path.exists(model_dir + filename):
        print(f"Error: {model_dir + filename} not found!")
        return

    # Arrays for Left Plot
    all_mvir = []
    all_redshifts = []
    all_radii = [] 
    
    # Arrays for Right Plot
    stats_z = []
    stats_disk_median = []
    stats_disk_low = []
    stats_disk_high = []
    stats_bulge_median = []
    stats_bulge_low = []
    stats_bulge_high = []
    
    for snapshot, z_actual in zip(snapshots, actual_redshifts):
        # Read Data
        mvir = read_hdf_from_model(model_dir, filename, snapshot, 'Mvir', hubble_h) * 1.0e10 / hubble_h
        stellar_mass = read_hdf_from_model(model_dir, filename, snapshot, 'StellarMass', hubble_h) * 1.0e10 / hubble_h
        bulge_mass = read_hdf_from_model(model_dir, filename, snapshot, 'BulgeMass', hubble_h) * 1.0e10 / hubble_h
        
        # SAGE Radius is Comoving Mpc/h
        sage_disk_radius = read_hdf_from_model(model_dir, filename, snapshot, 'DiskRadius', hubble_h) 
        sage_bulge_radius = read_hdf_from_model(model_dir, filename, snapshot, 'BulgeScaleRadius', hubble_h)
        
        # --- UNIT CONVERSION (Correcting to Physical kpc) ---
        # 1. Mpc/h -> kpc/h: * 1000
        # 2. kpc/h -> kpc:   / h
        # 3. Comoving -> Physical: / (1+z)
        # 4. Scale Length -> Half Mass: * 1.68
        phys_scale_factor = (1000.0 / hubble_h) / (1.0 + z_actual)
        
        r_disk_kpc = sage_disk_radius * 1.68 * phys_scale_factor
        r_bulge_kpc = sage_bulge_radius * 1.68 * phys_scale_factor
        
        # Filter for Heatmap
        galaxy_type = read_hdf_from_model(model_dir, filename, snapshot, 'Type', hubble_h)
        is_central = galaxy_type == 0
        has_mass = stellar_mass > 0
        
        # Collect data for heatmap
        valid_map = is_central & has_mass & (r_disk_kpc > 0)
        if np.sum(valid_map) > 0:
            all_mvir.extend(mvir[valid_map])
            all_redshifts.extend([z_actual] * np.sum(valid_map))
            all_radii.extend(np.log10(r_disk_kpc[valid_map]))

        # --- Statistics for Right Plot ---
        M_ffb = calculate_ffb_threshold_mass(z_actual, hubble_h)
        near_threshold = (mvir > M_ffb * 0.8) & (mvir < M_ffb * 1.2)
        subset = is_central & has_mass & near_threshold
        w_sub = np.where(subset)[0]
        
        if len(w_sub) > 5:
            stats_z.append(z_actual)
            
            # Disk
            radii = r_disk_kpc[w_sub]
            valid = radii > 0
            if np.sum(valid) > 0:
                stats_disk_median.append(np.median(radii[valid]))
                stats_disk_low.append(np.percentile(radii[valid], 16))
                stats_disk_high.append(np.percentile(radii[valid], 84))
            else:
                stats_disk_median.append(np.nan)
                stats_disk_low.append(np.nan)
                stats_disk_high.append(np.nan)
                
            # Bulge
            radii = r_bulge_kpc[w_sub]
            valid = (radii > 0) & (bulge_mass[w_sub] > 0)
            if np.sum(valid) > 0:
                stats_bulge_median.append(np.median(radii[valid]))
                stats_bulge_low.append(np.percentile(radii[valid], 16))
                stats_bulge_high.append(np.percentile(radii[valid], 84))
            else:
                stats_bulge_median.append(np.nan)
                stats_bulge_low.append(np.nan)
                stats_bulge_high.append(np.nan)

    # ================= PLOTTING =================
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # --- LEFT PANEL: Heatmap ---
    ax_left = axes[0]
    
    if len(all_mvir) > 0:
        a_mvir = np.array(all_mvir)
        a_z = np.array(all_redshifts)
        a_radii = np.array(all_radii) # already log10
        
        # Define bins
        z_bins = np.linspace(5, 20, 60)
        m_bins = np.linspace(9, 13, 60)
        
        # 1. Bin statistics
        from scipy.stats import binned_statistic_2d
        median_radius, z_edges, mvir_edges, _ = binned_statistic_2d(
            a_z, np.log10(a_mvir), a_radii,
            statistic='median', bins=[z_bins, m_bins]
        )
        
        # 2. YOUR SMOOTHING LOGIC (Restored)
        from scipy.ndimage import gaussian_filter, distance_transform_edt
        
        mask = ~np.isnan(median_radius)
        if np.sum(mask) > 0:
            # Fill NaN with nearest valid values
            ind = distance_transform_edt(~mask, return_distances=False, return_indices=True)
            median_radius_filled = median_radius[tuple(ind)]
            
            # Gaussian smoothing
            median_radius_smooth = gaussian_filter(median_radius_filled, sigma=1.0)
            
            # Fade out far from data
            distances = distance_transform_edt(~mask)
            weight_mask = np.exp(-distances / 3.0)
            median_radius_smooth = median_radius_smooth * weight_mask + median_radius_filled * (1 - weight_mask)
            
            # Cutoff
            median_radius_smooth[distances > 2.5] = np.nan
            heatmap = median_radius_smooth.T
        else:
            heatmap = median_radius.T
        
        # Plot
        im = ax_left.pcolormesh(z_edges, mvir_edges, heatmap, cmap='coolwarm_r', 
                                vmin=-1.5, vmax=0.5, shading='auto')
        cbar = plt.colorbar(im, ax=ax_left)
        cbar.set_label(r'$\log_{10} R_{\rm half, phys}$ (kpc)', fontsize=12)

    # FFB Threshold Line
    z_line = np.linspace(5, 20, 100)
    m_ffb = [calculate_ffb_threshold_mass(z, hubble_h) for z in z_line]
    ax_left.plot(z_line, np.log10(m_ffb), 'k-', lw=3, label='FFB Threshold')
    
    ax_left.set_xlabel('Redshift z', fontsize=14)
    ax_left.set_ylabel(r'$\log_{10} M_{\rm vir}$', fontsize=14)
    ax_left.set_xlim(5, 20)
    ax_left.set_ylim(9, 13)
    ax_left.legend()

    # --- RIGHT PANEL: Evolution ---
    ax_right = axes[1]
    
    # Theory Lines
    z_t = np.linspace(5, 20, 100)
    z_10 = (1 + z_t) / 10.0
    # Eq 31 (Shell): Re ~ 0.5 * R_sh
    r_th_shell = 0.5 * 0.56 * (z_10 ** -1.78) 
    # Eq 33 (Disk)
    r_th_disk = 0.31 * (z_10 ** -3.07)
    
    ax_right.plot(z_t, r_th_shell, color='royalblue', ls='--', lw=2, label='Theory (Shell)')
    ax_right.plot(z_t, r_th_disk, color='darkorange', ls='--', lw=2, label='Theory (Disk)')

    # Simulation Data
    if len(stats_z) > 0:
        ax_right.plot(stats_z, stats_disk_median, 'o-', color='darkorange', lw=2, label='SAGE Disk')
        ax_right.fill_between(stats_z, stats_disk_low, stats_disk_high, color='darkorange', alpha=0.2)
        
        valid_b = ~np.isnan(stats_bulge_median)
        if np.any(valid_b):
            ax_right.plot(np.array(stats_z)[valid_b], np.array(stats_bulge_median)[valid_b], 
                         's-', color='royalblue', lw=2, label='SAGE Bulge')
            ax_right.fill_between(np.array(stats_z)[valid_b], 
                                 np.array(stats_bulge_low)[valid_b], 
                                 np.array(stats_bulge_high)[valid_b], 
                                 color='royalblue', alpha=0.2)

    ax_right.set_yscale('log')
    ax_right.set_xlabel('Redshift z', fontsize=14)
    ax_right.set_ylabel(r'$R_{\rm half}$ (physical kpc)', fontsize=14)
    ax_right.set_xlim(5, 20)
    ax_right.set_ylim(0.01, 3.0)
    ax_right.yaxis.set_major_formatter(plt.ScalarFormatter())
    ax_right.grid(True, alpha=0.1)
    ax_right.legend(frameon=False)

    plt.tight_layout()
    out_path = DirName + 'plots/ffb_threshold_analysis_empirical' + OutputFormat
    plt.savefig(out_path, dpi=300)
    print(f'Plot saved to: {out_path}')
    plt.close()
    print('='*60 + '\n')


def plot_gas_fraction_evolution():
    """Plot gas fraction evolution at FFB threshold across all FFB models
    
    Shows cold gas fraction (f_gas = M_cold / (M_cold + M_star)) vs redshift
    for disk and bulge components separately, with shading showing spread
    across FFB models (SAGE26, FFB40, FFB50, FFB80, FFB100)
    """
    
    print('\n' + '='*60)
    print('Creating Gas Fraction Evolution Plot')
    print('='*60)
    
    # Define all FFB models
    ffb_models = [
        {'name': 'SAGE26 (FFB20%)', 'dir': './output/millennium/'},
        {'name': 'FFB 40%', 'dir': './output/millennium_FFB40/'},
        {'name': 'FFB 50%', 'dir': './output/millennium_FFB50/'},
        {'name': 'FFB 80%', 'dir': './output/millennium_FFB80/'},
        {'name': 'FFB 100%', 'dir': './output/millennium_FFB100/'}
    ]
    
    filename = 'model_0.hdf5'
    hubble_h = MILLENNIUM_HUBBLE_H
    
    # Define redshift range - all snapshots in z=5-20 range
    snapshots = []
    actual_redshifts = []
    for idx, z in enumerate(DEFAULT_REDSHIFTS):
        if 4.5 <= z <= 20.5:
            snapshots.append(f'Snap_{idx}')
            actual_redshifts.append(z)
    
    print(f'Analyzing {len(snapshots)} redshift snapshots from z={min(actual_redshifts):.1f} to z={max(actual_redshifts):.1f}')
    print(f'Using {len(ffb_models)} FFB models for uncertainty estimation')
    
    # Arrays to store data for each model
    all_model_data = []
    
    for model in ffb_models:
        model_dir = model['dir']
        
        # Check if file exists
        if not os.path.exists(model_dir + filename):
            print(f"Warning: {model['name']} data not found at {model_dir + filename}")
            continue
        
        print(f"\nProcessing {model['name']}...")
        
        redshifts_model = []
        disk_gas_frac = []
        bulge_gas_frac = []
        
        # Loop through snapshots
        for snapshot, z_actual in zip(snapshots, actual_redshifts):
            try:
                # Read data
                mvir = read_hdf_from_model(model_dir, filename, snapshot, 'Mvir', hubble_h) * 1.0e10 / hubble_h
                stellar_mass = read_hdf_from_model(model_dir, filename, snapshot, 'StellarMass', hubble_h) * 1.0e10 / hubble_h
                bulge_mass = read_hdf_from_model(model_dir, filename, snapshot, 'BulgeMass', hubble_h) * 1.0e10 / hubble_h
                cold_gas = read_hdf_from_model(model_dir, filename, snapshot, 'ColdGas', hubble_h) * 1.0e10 / hubble_h
                hot_gas = read_hdf_from_model(model_dir, filename, snapshot, 'HotGas', hubble_h) * 1.0e10 / hubble_h
                galaxy_type = read_hdf_from_model(model_dir, filename, snapshot, 'Type', hubble_h)
                
                # Calculate FFB threshold mass for this redshift
                M_ffb_threshold = calculate_ffb_threshold_mass(z_actual, hubble_h)
                
                # Select disk galaxies near FFB threshold
                is_disk = galaxy_type == 0
                has_stellar_mass = stellar_mass > 0
                near_threshold = (mvir > M_ffb_threshold * 0.8) & (mvir < M_ffb_threshold * 1.2)
                
                w = np.where(is_disk & has_stellar_mass & near_threshold)[0]
                
                if len(w) > 5:  # Need at least 5 galaxies
                    # Calculate disk mass (stellar - bulge)
                    disk_mass = stellar_mass[w] - bulge_mass[w]
                    disk_mass = np.maximum(disk_mass, 0)  # Ensure non-negative
                    
                    # Disk gas fraction: assume cold gas is in disk
                    # f_gas_disk = M_cold / (M_cold + M_disk_star)
                    valid_disk = (disk_mass > 0) & (cold_gas[w] > 0)
                    if np.sum(valid_disk) > 0:
                        f_disk = cold_gas[w[valid_disk]] / (cold_gas[w[valid_disk]] + disk_mass[valid_disk])
                        median_f_disk = np.median(f_disk)
                    else:
                        median_f_disk = np.nan
                    
                    # Bulge gas fraction: use hot gas for bulge component
                    # f_gas_bulge = M_hot / (M_hot + M_bulge_star)
                    valid_bulge = (bulge_mass[w] > 0) & (cold_gas[w] > 0)
                    n_valid_bulge = np.sum(valid_bulge)
                    if n_valid_bulge > 0:
                        f_bulge = cold_gas[w[valid_bulge]] / (cold_gas[w[valid_bulge]] + bulge_mass[w[valid_bulge]])
                        median_f_bulge = np.median(f_bulge)
                    else:
                        median_f_bulge = np.nan
                    
                    # Debug first snapshot
                    if z_actual > 11.5 and z_actual < 12.0:
                        print(f"      z={z_actual:.2f}: {len(w)} galaxies at threshold")
                        print(f"        Bulge mass > 0: {np.sum(bulge_mass[w] > 0)}")
                        print(f"        Hot gas > 0: {np.sum(hot_gas[w] > 0)}")
                        print(f"        Valid bulge: {n_valid_bulge}")
                        if n_valid_bulge > 0:
                            print(f"        f_bulge range: {np.min(f_bulge):.3f} to {np.max(f_bulge):.3f}")
                    
                    if not np.isnan(median_f_disk):
                        redshifts_model.append(z_actual)
                        disk_gas_frac.append(median_f_disk)
                        bulge_gas_frac.append(median_f_bulge)
                        
            except Exception as e:
                # Skip snapshots that don't exist or have issues
                continue
        
        if len(redshifts_model) > 0:
            all_model_data.append({
                'name': model['name'],
                'redshifts': np.array(redshifts_model),
                'disk_gas_frac': np.array(disk_gas_frac),
                'bulge_gas_frac': np.array(bulge_gas_frac)
            })
            print(f"  Processed {len(redshifts_model)} redshifts: z={np.min(redshifts_model):.1f}-{np.max(redshifts_model):.1f}")
            print(f"    Disk gas fractions: {disk_gas_frac[:3]}")
            print(f"    Bulge gas fractions: {bulge_gas_frac[:3]}")
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    if len(all_model_data) > 0:
        # Find common redshift grid
        all_z = np.unique(np.concatenate([d['redshifts'] for d in all_model_data]))
        all_z = np.sort(all_z)
        
        # Interpolate all models to common grid and calculate spread
        disk_frac_all = []
        bulge_frac_all = []
        
        for data in all_model_data:
            # Sort redshifts for interpolation (np.interp requires increasing x)
            sort_idx = np.argsort(data['redshifts'])
            z_sorted = data['redshifts'][sort_idx]
            fgas_disk_sorted = data['disk_gas_frac'][sort_idx]
            fgas_bulge_sorted = data['bulge_gas_frac'][sort_idx]
            
            # Interpolate to common grid
            disk_interp = np.interp(all_z, z_sorted, fgas_disk_sorted, 
                                   left=np.nan, right=np.nan)
            bulge_interp = np.interp(all_z, z_sorted, fgas_bulge_sorted, 
                                     left=np.nan, right=np.nan)
            disk_frac_all.append(disk_interp)
            bulge_frac_all.append(bulge_interp)
        
        # Convert to arrays (models x redshifts)
        disk_frac_all = np.array(disk_frac_all)
        bulge_frac_all = np.array(bulge_frac_all)
        
        # Calculate median and percentiles across models
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            disk_median = np.nanmedian(disk_frac_all, axis=0)
            disk_lower = np.nanpercentile(disk_frac_all, 16, axis=0)
            disk_upper = np.nanpercentile(disk_frac_all, 84, axis=0)
            
            bulge_median = np.nanmedian(bulge_frac_all, axis=0)
            bulge_lower = np.nanpercentile(bulge_frac_all, 16, axis=0)
            bulge_upper = np.nanpercentile(bulge_frac_all, 84, axis=0)
        
        # Debug bulge data
        print(f"\nBulge data summary:")
        print(f"  Bulge median values: {bulge_median}")
        print(f"  Non-NaN bulge values: {np.sum(~np.isnan(bulge_median))}")
        print(f"  Positive bulge values: {np.sum((~np.isnan(bulge_median)) & (bulge_median > 0))}")
        
        # Plot disk gas fraction with uncertainty band
        valid_disk = ~np.isnan(disk_median) & (disk_median > 0)
        ax.plot(all_z[valid_disk], np.log10(disk_median[valid_disk]),
               color='darkorange',
               linestyle='-',
               linewidth=3,
               label='Disk',
               alpha=0.8,
               zorder=3)
        
        ax.fill_between(all_z[valid_disk], 
                       np.log10(disk_lower[valid_disk]), 
                       np.log10(disk_upper[valid_disk]),
                       color='darkorange',
                       alpha=0.3,
                       zorder=2)
        
        # Plot bulge gas fraction with uncertainty band
        valid_bulge = ~np.isnan(bulge_median) & (bulge_median > 0)
        if np.sum(valid_bulge) > 0:
            ax.plot(all_z[valid_bulge], np.log10(bulge_median[valid_bulge]),
                   color='royalblue',
                   linestyle='-',
                   linewidth=3,
                   label='Bulge',
                   alpha=0.8,
                   zorder=3)
            
            ax.fill_between(all_z[valid_bulge], 
                           np.log10(bulge_lower[valid_bulge]), 
                           np.log10(bulge_upper[valid_bulge]),
                           color='royalblue',
                           alpha=0.3,
                           zorder=2)
        
        ax.set_xlabel('Redshift z', fontsize=12)
        ax.set_ylabel(r'$\log_{10} f_{\mathrm{gas}}$', fontsize=12)
        ax.set_xlim(5, 20)
        ax.set_ylim(-3, 0)
        ax.legend(loc='lower left', fontsize=10, frameon=False, title='FFB Model Spread (16-84%)', title_fontsize=9)
    # Save figure
    OutputDir = DirName + 'plots/'
    if not os.path.exists(OutputDir):
        os.makedirs(OutputDir)
    
    output_path = OutputDir + 'gas_fraction_evolution' + OutputFormat
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'\nPlot saved to: {output_path}')
    plt.close()
    
    print('='*60 + '\n')


def plot_ffb_metallicity_limit():
    """
    Plot the upper limit for metallicity in FFB star-forming clouds.
    Shows mixing between outflowing metals (Zsn = 1 Zsun) and inflowing gas
    (Zin = 0.1 Zsun or 0.02 Zsun) at the FFB threshold mass.
    Shading represents epsilon_max from 0.2 (bottom) to 1.0 (top).
    
    Based on equation (42) from the paper.
    Uses actual mass loading factors from SAGE galaxies at FFB threshold.
    """
    print('\nCreating FFB Metallicity Limit Plot')
    print('='*60)
    
    # Constants
    hubble_h = 0.73
    Z_sun = 0.0134  # Solar metallicity (mass fraction)
    Z_sn = 1.0 * Z_sun  # Supernova ejecta metallicity
    
    # Two scenarios for inflowing gas metallicity
    Z_in_high = 0.1 * Z_sun  # Blue line
    Z_in_low = 0.02 * Z_sun  # Orange line
    
    # Get snapshots and redshifts
    snapshots = [63, 59, 55, 51, 47, 44, 42, 40, 38, 37, 36, 35, 34, 33, 32, 31]
    redshift_table = {
        63: 5.724, 59: 6.197, 55: 6.712, 51: 7.280, 47: 7.906,
        44: 8.429, 42: 8.738, 40: 9.061, 38: 9.521, 37: 9.760,
        36: 10.007, 35: 10.263, 34: 10.530, 33: 10.808, 32: 11.098, 31: 11.400
    }
    actual_redshifts = [redshift_table[snap] for snap in snapshots if snap in redshift_table]
    
    print(f'Analyzing {len(snapshots)} redshift snapshots from z={min(actual_redshifts):.1f} to z={max(actual_redshifts):.1f}')
    
    # Define FFB models with different epsilon_max values
    ffb_models = [
        {'name': 'FFB 20%', 'dir': './output/millennium/', 'eps_max': 0.2},
        {'name': 'FFB 40%', 'dir': './output/millennium_FFB40/', 'eps_max': 0.4},
        {'name': 'FFB 50%', 'dir': './output/millennium_FFB50/', 'eps_max': 0.5},
        {'name': 'FFB 80%', 'dir': './output/millennium_FFB80/', 'eps_max': 0.8},
        {'name': 'FFB 100%', 'dir': './output/millennium_FFB100/', 'eps_max': 1.0},
    ]
    
    # Collect mass loading factors from each model
    all_model_data = []
    
    for model in ffb_models:
        model_dir = model['dir']
        filename = 'model_0.hdf5'
        
        # Check if file exists
        if not os.path.exists(model_dir + filename):
            print(f"Warning: {model['name']} data not found at {model_dir + filename}")
            continue
        
        print(f"\nProcessing {model['name']}...")
        
        redshifts_model = []
        mass_loading_model = []
        
        # Loop through snapshots
        for snapshot, z_actual in zip(snapshots, actual_redshifts):
            try:
                # Read data - convert snapshot to Snap_XX format
                snap_str = f'Snap_{snapshot}'
                mvir = read_hdf_from_model(model_dir, filename, snap_str, 'Mvir', hubble_h) * 1.0e10 / hubble_h
                sfr_disk = read_hdf_from_model(model_dir, filename, snap_str, 'SfrDisk', hubble_h)
                sfr_bulge = read_hdf_from_model(model_dir, filename, snap_str, 'SfrBulge', hubble_h)
                mass_loading = read_hdf_from_model(model_dir, filename, snap_str, 'MassLoading', hubble_h)
                galaxy_type = read_hdf_from_model(model_dir, filename, snap_str, 'Type', hubble_h)
                
                # Calculate FFB threshold mass for this redshift
                M_ffb_threshold = calculate_ffb_threshold_mass(z_actual, hubble_h)
                
                # Select disk galaxies near FFB threshold with active SF
                sfr_total = sfr_disk + sfr_bulge
                is_disk = galaxy_type == 0
                has_sfr = sfr_total > 0
                near_threshold = (mvir > M_ffb_threshold * 0.8) & (mvir < M_ffb_threshold * 1.2)
                
                w = np.where(is_disk & has_sfr & near_threshold)[0]
                
                if len(w) > 5:  # Need at least 5 galaxies
                    # Get median mass loading factor
                    valid_ml = mass_loading[w] > 0
                    if np.sum(valid_ml) > 0:
                        median_ml = np.median(mass_loading[w[valid_ml]])
                        redshifts_model.append(z_actual)
                        mass_loading_model.append(median_ml)
                        
            except Exception as e:
                # Skip snapshots that don't exist or have issues
                continue
        
        if len(redshifts_model) > 0:
            all_model_data.append({
                'name': model['name'],
                'eps_max': model['eps_max'],
                'redshifts': np.array(redshifts_model),
                'mass_loading': np.array(mass_loading_model)
            })
            print(f"  Processed {len(redshifts_model)} redshifts")
            print(f"    Mass loading range: {np.min(mass_loading_model):.2f} to {np.max(mass_loading_model):.2f}")
    
    # Create fine redshift grid for plotting
    z_grid = np.linspace(5, 20, 100)
    
    def calculate_metallicity_from_data(z_grid, model_data, Z_in):
        """
        Calculate metallicity using actual mass loading data.
        """
        # Interpolate mass loading to grid
        eta_interp = np.interp(z_grid, model_data['redshifts'], model_data['mass_loading'],
                              left=np.nan, right=np.nan)
        
        # Inflow rate relative to SFR
        # For high-z, small galaxies: strong cosmological accretion
        f_in = 5.0 * (1 + z_grid) / 11.0  # ~9 at z=20, ~5 at z=10, ~2.7 at z=5
        
        # Mixed metallicity from equation (42)
        Z_mix = (eta_interp * Z_sn + f_in * Z_in) / (eta_interp + f_in)
        
        return Z_mix / Z_sun  # Return in units of Z_sun
    
    # Calculate metallicity limits for both Z_in scenarios using actual data
    # Find models closest to eps_max = 0.2 and 1.0
    model_low_eps = all_model_data[0]  # FFB 20% (eps_max = 0.2)
    model_high_eps = all_model_data[-1]  # FFB 100% (eps_max = 1.0)
    
    # High Z_in (0.1 Zsun) - Blue
    Z_high_low_eps = calculate_metallicity_from_data(z_grid, model_low_eps, Z_in_high)
    Z_high_high_eps = calculate_metallicity_from_data(z_grid, model_high_eps, Z_in_high)
    
    # Low Z_in (0.02 Zsun) - Orange
    Z_low_low_eps = calculate_metallicity_from_data(z_grid, model_low_eps, Z_in_low)
    Z_low_high_eps = calculate_metallicity_from_data(z_grid, model_high_eps, Z_in_low)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Filter out NaN values for plotting
    valid_high = ~np.isnan(Z_high_low_eps) & ~np.isnan(Z_high_high_eps)
    valid_low = ~np.isnan(Z_low_low_eps) & ~np.isnan(Z_low_high_eps)
    
    # Plot high Z_in scenario (blue)
    if np.sum(valid_high) > 0:
        ax.fill_between(z_grid[valid_high], Z_high_low_eps[valid_high], Z_high_high_eps[valid_high],
                       color='royalblue', alpha=0.4,
                       label=r'$Z_{\rm in} = 0.1\,Z_{\odot}$')
        
        # Add median line
        Z_high_median = (Z_high_low_eps + Z_high_high_eps) / 2
        ax.plot(z_grid[valid_high], Z_high_median[valid_high],
               color='royalblue', linestyle='-', linewidth=2, alpha=0.8)
    
    # Plot low Z_in scenario (orange)
    if np.sum(valid_low) > 0:
        ax.fill_between(z_grid[valid_low], Z_low_low_eps[valid_low], Z_low_high_eps[valid_low],
                       color='darkorange', alpha=0.4,
                       label=r'$Z_{\rm in} = 0.02\,Z_{\odot}$')
        
        # Add median line
        Z_low_median = (Z_low_low_eps + Z_low_high_eps) / 2
        ax.plot(z_grid[valid_low], Z_low_median[valid_low],
               color='darkorange', linestyle='-', linewidth=2, alpha=0.8)
    
    # Formatting
    ax.set_xlabel('Redshift z', fontsize=12)
    ax.set_ylabel(r'$\log_{10}(Z_{\rm FFB} / Z_{\odot})$', fontsize=12)
    ax.set_xlim(5, 20)
    ax.set_yscale('log')
    
    # Add text annotation
    ax.text(0.05, 0.95, 'FFB Threshold Galaxies',
           transform=ax.transAxes,
           verticalalignment='top',
           horizontalalignment='left',
           fontsize=11)
    
    ax.text(0.05, 0.88, r'$\epsilon_{\rm max}$: 0.2 (bottom) to 1.0 (top)',
           transform=ax.transAxes,
           verticalalignment='top',
           horizontalalignment='left',
           fontsize=9,
           style='italic')
    
    ax.legend(loc='upper right', fontsize=10, frameon=False)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save figure
    OutputDir = DirName + 'plots/'
    if not os.path.exists(OutputDir):
        os.makedirs(OutputDir)
    
    output_path = OutputDir + 'ffb_metallicity_limit' + OutputFormat
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'\nPlot saved to: {output_path}')
    plt.close()
    
    print('='*60 + '\n')

# ==================================================================

def main():
    """Main function to run the plotting script"""
    
    print('Running li_paper_plots.py\n')
    
    # Set random seed for reproducibility
    seed(2222)
    
    # Set up simulation parameters
    BoxSize = MILLENNIUM_BOXSIZE
    Hubble_h = MILLENNIUM_HUBBLE_H
    VolumeFraction = 1.0
    
    # Calculate volume
    volume = (BoxSize/Hubble_h)**3.0 * VolumeFraction
    
    # Create output directory
    OutputDir = DirName + 'plots/'
    if not os.path.exists(OutputDir): 
        os.makedirs(OutputDir)
    
    # Check if file exists
    if not os.path.exists(DirName + FileName):
        print(f"Error: File {DirName + FileName} not found!")
        print(f"Please ensure the output file exists at the specified location.")
        return
    
    # Read redshift info
    try:
        with h5.File(DirName + FileName, 'r') as f:
            if Snapshot in f:
                if 'Redshift' in f[Snapshot].attrs:
                    TargetZ = f[Snapshot].attrs['Redshift']
                    print(f'Analyzing snapshot {Snapshot} at redshift z = {TargetZ:.3f}')
                else:
                    # Use default redshift from list
                    # Extract number from snapshot name (e.g., 'Snap_63' -> 63)
                    snap_idx = int(Snapshot.split('_')[-1])
                    if snap_idx < len(DEFAULT_REDSHIFTS):
                        TargetZ = DEFAULT_REDSHIFTS[snap_idx]
                        print(f'Analyzing snapshot {Snapshot} at redshift z = {TargetZ:.3f}')
                    else:
                        TargetZ = 0.0
                        print(f'Analyzing snapshot {Snapshot} (redshift unknown, assuming z=0)')
            else:
                print(f"Error: Snapshot {Snapshot} not found in file")
                print(f"Available snapshots: {list(f.keys())}")
                return
    except Exception as e:
        print(f"Error opening file: {e}")
        return

    print(f'Reading galaxy properties from {DirName}{FileName}')
    print(f'Box size: {BoxSize} h^-1 Mpc')
    print(f'Volume: {volume:.2f} (h^-1 Mpc)^3')
    print(f'Hubble h: {Hubble_h}\n')

    # Read galaxy properties
    print('Loading galaxy data...')
    
    CentralMvir = read_hdf(snap_num = Snapshot, param = 'CentralMvir') * 1.0e10 / Hubble_h
    Mvir = read_hdf(snap_num = Snapshot, param = 'Mvir') * 1.0e10 / Hubble_h
    StellarMass = read_hdf(snap_num = Snapshot, param = 'StellarMass') * 1.0e10 / Hubble_h
    MetalsStellarMass = read_hdf(snap_num = Snapshot, param = 'MetalsStellarMass') * 1.0e10 / Hubble_h
    BulgeMass = read_hdf(snap_num = Snapshot, param = 'BulgeMass') * 1.0e10 / Hubble_h
    BlackHoleMass = read_hdf(snap_num = Snapshot, param = 'BlackHoleMass') * 1.0e10 / Hubble_h
    ColdGas = read_hdf(snap_num = Snapshot, param = 'ColdGas') * 1.0e10 / Hubble_h
    MetalsColdGas = read_hdf(snap_num = Snapshot, param = 'MetalsColdGas') * 1.0e10 / Hubble_h
    MetalsEjectedMass = read_hdf(snap_num = Snapshot, param = 'MetalsEjectedMass') * 1.0e10 / Hubble_h
    HotGas = read_hdf(snap_num = Snapshot, param = 'HotGas') * 1.0e10 / Hubble_h
    MetalsHotGas = read_hdf(snap_num = Snapshot, param = 'MetalsHotGas') * 1.0e10 / Hubble_h
    EjectedMass = read_hdf(snap_num = Snapshot, param = 'EjectedMass') * 1.0e10 / Hubble_h
    CGMgas = read_hdf(snap_num = Snapshot, param = 'CGMgas') * 1.0e10 / Hubble_h
    MetalsCGMgas = read_hdf(snap_num = Snapshot, param = 'MetalsCGMgas') * 1.0e10 / Hubble_h
    
    IntraClusterStars = read_hdf(snap_num = Snapshot, param = 'IntraClusterStars') * 1.0e10 / Hubble_h
    DiskRadius = read_hdf(snap_num = Snapshot, param = 'DiskRadius')
    BulgeRadius = read_hdf(snap_num = Snapshot, param = 'BulgeScaleRadius')
    MergerBulgeRadius = read_hdf(snap_num = Snapshot, param = 'MergerBulgeRadius')
    InstabilityBulgeRadius = read_hdf(snap_num = Snapshot, param = 'InstabilityBulgeRadius')
    MergerBulgeMass = read_hdf(snap_num = Snapshot, param = 'MergerBulgeMass') * 1.0e10 / Hubble_h
    InstabilityBulgeMass = read_hdf(snap_num = Snapshot, param = 'InstabilityBulgeMass') * 1.0e10 / Hubble_h
    
    H2gas = read_hdf(snap_num = Snapshot, param = 'H2gas') * 1.0e10 / Hubble_h
    Vvir = read_hdf(snap_num = Snapshot, param = 'Vvir')
    Vmax = read_hdf(snap_num = Snapshot, param = 'Vmax')
    Rvir = read_hdf(snap_num = Snapshot, param = 'Rvir')
    SfrDisk = read_hdf(snap_num = Snapshot, param = 'SfrDisk')
    SfrBulge = read_hdf(snap_num = Snapshot, param = 'SfrBulge')
    
    CentralGalaxyIndex = read_hdf(snap_num = Snapshot, param = 'CentralGalaxyIndex')
    Type = read_hdf(snap_num = Snapshot, param = 'Type')
    Posx = read_hdf(snap_num = Snapshot, param = 'Posx')
    Posy = read_hdf(snap_num = Snapshot, param = 'Posy')
    Posz = read_hdf(snap_num = Snapshot, param = 'Posz')
    
    OutflowRate = read_hdf(snap_num = Snapshot, param = 'OutflowRate')
    MassLoading = read_hdf(snap_num = Snapshot, param = 'MassLoading')
    Cooling = read_hdf(snap_num = Snapshot, param = 'Cooling')
    TimeOfLastMajorMerger = read_hdf(snap_num = Snapshot, param = 'TimeOfLastMajorMerger')
    Regime = read_hdf(snap_num = Snapshot, param = 'Regime')
    
    # Derived quantities
    Tvir = 35.9 * (Vvir)**2  # in Kelvin
    Tmax = 2.5e5  # K, corresponds to Vvir ~52.7 km/s
    
    # Statistics
    w = np.where(StellarMass > 1.0e10)[0]
    print(f'Number of galaxies read: {len(StellarMass)}')
    print(f'Galaxies more massive than 10^10 h^-1 Msun: {len(w)}')
    
    # Sample some properties
    if len(StellarMass) > 0:
        print(f'\nSample statistics:')
        print(f'  Stellar mass range: {StellarMass[StellarMass>0].min():.2e} - {StellarMass.max():.2e} h^-1 Msun')
        print(f'  Halo mass range: {Mvir[Mvir>0].min():.2e} - {Mvir.max():.2e} h^-1 Msun')
    
    print('\n' + '='*60)
    print('Data loaded successfully!')
    print('='*60)
    
    # Generate plots
    plot_smf_grid()
    plot_smf_vs_redshift()
    plot_uvlf_grid()
    plot_uvlf_vs_redshift()
    plot_cumulative_surface_density()
    plot_density_evolution()
    plot_ffb_threshold_analysis()
    plot_ffb_threshold_analysis_empirical()
    plot_gas_fraction_evolution()
    plot_ffb_metallicity_limit()
    
    print('\nAll plots completed!')
    print(f'Plots saved to: {OutputDir}')
    print('='*60 + '\n')

# ==================================================================

if __name__ == '__main__':
    main()
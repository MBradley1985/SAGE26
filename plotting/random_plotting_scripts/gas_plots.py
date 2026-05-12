import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import os
from random import sample, seed

import warnings
warnings.filterwarnings("ignore")

# ========================== USER OPTIONS ==========================

# File details
DirName = './output/millennium/'
FileName = 'model_0.hdf5'
Snapshot = 'Snap_63'
ObsDataDir = './data/Gas/'

# Simulation details
Hubble_h = 0.73        # Hubble parameter
BoxSize = 62.5         # h-1 Mpc
VolumeFraction = 1.0   # Fraction of the full volume output by the model

# Plotting options
whichimf = 1        # 0=Slapeter; 1=Chabrier
dilute = 7000       # Number of galaxies to plot in scatter plots
sSFRcut = -11.0     # Divide quiescent from star forming galaxies

OutputFormat = '.pdf'
plt.rcParams["figure.figsize"] = (8.34,6.25)
plt.rcParams["figure.dpi"] = 96
plt.rcParams["font.size"] = 14

plt.rcParams['figure.facecolor'] = 'black'
plt.rcParams['axes.facecolor'] = 'black'
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['axes.titlecolor'] = 'white'
plt.rcParams['text.color'] = 'white'
plt.rcParams['legend.facecolor'] = 'black'
plt.rcParams['legend.edgecolor'] = 'white'

# ==================================================================

def read_hdf(filename = None, snap_num = None, param = None):

    if filename is None:
        filename = os.path.join(DirName, FileName)
    
    # The new SAGE output has each snapshot in a folder, and then broken into multiple files.
    # We need to find all the files for a given snapshot and stitch them together.
    
    snap_dir = os.path.join(DirName, snap_num)
    
    # Check if the snapshot directory exists.
    if not os.path.isdir(snap_dir):
        # If it doesn't, we're probably dealing with the old SAGE output format.
        # So we'll just read from the single file.
        with h5.File(filename, 'r') as property_file:
            return np.array(property_file[snap_num][param])

    # If the directory exists, we're dealing with the new SAGE output format.
    # We need to find all the files in the directory and stitch them together.
    
    files = [os.path.join(snap_dir, f) for f in os.listdir(snap_dir) if f.endswith('.hdf5')]
    
    if len(files) == 0:
        print(f"No HDF5 files found in {snap_dir}")
        return np.array([])
        
    all_data = []
    for f in files:
        with h5.File(f, 'r') as property_file:
            all_data.append(np.array(property_file[param]))
            
    return np.concatenate(all_data)

def read_obs_data(filename):
    """Read observational data files"""
    filepath = os.path.join(ObsDataDir, filename)
    if not os.path.exists(filepath):
        print(f"  Warning: Observational data file {filename} not found")
        return None
    
    data = np.loadtxt(filepath)
    return data

# ==================================================================

if __name__ == '__main__':
    
    OutputDir = os.path.join(DirName, 'plots')
    if not os.path.exists(OutputDir):
        os.makedirs(OutputDir)

# ======================= HI MASS RATIO COMPARISON PLOT =======================
    print("Creating H I mass ratio comparison plot...")

    # Create the plot
    plt.figure()

    models = [
        {'name': 'millennium', 'label': 'BR06', 'color': 'white'},
        {'name': 'millennium_gd14', 'label': 'GD14', 'color': 'goldenrod'},
        {'name': 'millennium_kd12', 'label': 'KD12', 'color': 'dodgerblue'},
        {'name': 'millennium_kmt09', 'label': 'KMT09', 'color': 'limegreen'},
        {'name': 'millennium_k13', 'label': 'K13', 'color': 'firebrick'}
    ]

    for i, model in enumerate(models):
        DirName = f'./output/{model["name"]}/'
        
        # Read galaxy properties
        StellarMass = read_hdf(snap_num=Snapshot, param='StellarMass') * 1.0e10 / Hubble_h
        H1gas = read_hdf(snap_num=Snapshot, param='H1gas') * 1.0e10 / Hubble_h

        # Filter out invalid data
        valid_indices = (StellarMass > 1e8) & (H1gas > 0)
        StellarMass = StellarMass[valid_indices]
        H1gas = H1gas[valid_indices]
        
        # Calculate the ratio
        Ratio = H1gas / StellarMass
        
        if i == 0:
            # Create a 2D histogram for the first model
            h, xedges, yedges = np.histogram2d(np.log10(StellarMass), np.log10(Ratio), bins=256)
            
            # Define contour levels using a logarithmic scale
            levels = np.logspace(np.log10(h[h>0].min()), np.log10(h.max()), 10)
            
            # Create the contour plot
            plt.contourf(xedges[:-1], yedges[:-1], h.T, cmap='plasma', levels=levels, alpha=0.7)

        # Calculate median line
        bin_width = 0.2
        mass_bins = np.arange(8.0, 12.0 + bin_width, bin_width)
        mass_centers = mass_bins[:-1] + bin_width / 2
        
        median_ratio = np.zeros_like(mass_centers)
        p16 = np.full_like(mass_centers, np.nan)
        p84 = np.full_like(mass_centers, np.nan)
        n_bootstrap = 1000
        rng = np.random.default_rng(42)
        for j in range(len(mass_bins) - 1):
            mask = (np.log10(StellarMass) >= mass_bins[j]) & (np.log10(StellarMass) < mass_bins[j+1])
            data = np.log10(Ratio[mask])
            if data.size > 0:
                median_ratio[j] = np.median(data)
                # Bootstrap
                boot_meds = np.array([
                    np.median(rng.choice(data, size=data.size, replace=True))
                    for _ in range(n_bootstrap)
                ])
                p16[j] = np.percentile(boot_meds, 16)
                p84[j] = np.percentile(boot_meds, 84)
        # Plot median line and bootstrap error band
        # if i == 0:
        #     plt.plot(mass_centers, median_ratio, label=model['label'], color=model['color'], zorder=10, linewidth=4)
        #     plt.fill_between(mass_centers, p16, p84, color=model['color'], alpha=0.2, zorder=9)
        # elif model['name'] == 'millennium_kmt09':
        #     plt.plot(mass_centers, median_ratio, label=model['label'], color=model['color'], zorder=15, linewidth=2)
        #     plt.fill_between(mass_centers, p16, p84, color=model['color'], alpha=0.2, zorder=14)
        # elif model['name'] == 'millennium_k13':
        #     plt.plot(mass_centers, median_ratio, label=model['label'], color=model['color'], zorder=5, linewidth=2)
        #     plt.fill_between(mass_centers, p16, p84, color=model['color'], alpha=0.2, zorder=4)
        # else:
        plt.plot(mass_centers, median_ratio, label=model['label'], color=model['color'], zorder=5)
        plt.fill_between(mass_centers, p16, p84, color=model['color'], alpha=0.2, zorder=4)
    
    # Plot observational data
    data = read_obs_data('HIGasRatio_NonDetEQZero.dat')
    if data is not None:
        log_mstar = data[:, 0]
        median = data[:, 1]
        p16 = data[:, 2]
        p84 = data[:, 3]
        mask = (median > -10) & (median < 2) & (p16 > -10) & (p84 > -10)
        yerr_lower = np.abs(median[mask] - p16[mask])
        yerr_upper = np.abs(p84[mask] - median[mask])
        plt.errorbar(log_mstar[mask], median[mask], yerr=[yerr_lower, yerr_upper], 
                    fmt='o', color='gray', markersize=6, capsize=3, 
                    label='xGASS', zorder=10, markerfacecolor='none', markeredgewidth=1.5)

        from matplotlib.ticker import MaxNLocator
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    
        plt.ylim(-3, 1)
    
    plt.xlabel(r'$\log_{10}\ M_{\star}\ (M_{\odot})$')
    plt.ylabel(r'$\log_{10}\ (M_{\mathrm{HI}} / M_{\star})$')
    
    # Add legend at the bottom
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=len(models)+1, fontsize=10)
    plt.tight_layout()

    outputFile = os.path.join(OutputDir, 'h1_mass_ratio_comparison' + OutputFormat)
    plt.savefig(outputFile)
    print(f"Plot saved to {outputFile}")
    plt.close()

# ======================= H2 MASS RATIO COMPARISON PLOT =======================
    print("Creating H2 mass ratio comparison plot...")

    # Create the plot
    plt.figure()

    models = [
        {'name': 'millennium', 'label': 'BR06', 'color': 'white'},
        {'name': 'millennium_gd14', 'label': 'GD14', 'color': 'goldenrod'},
        {'name': 'millennium_kd12', 'label': 'KD12', 'color': 'dodgerblue'},
        {'name': 'millennium_kmt09', 'label': 'KMT09', 'color': 'limegreen'},
        {'name': 'millennium_k13', 'label': 'K13', 'color': 'firebrick'}
    ]

    for i, model in enumerate(models):
        DirName = f'./output/{model["name"]}/'
        
        # Read galaxy properties
        StellarMass = read_hdf(snap_num=Snapshot, param='StellarMass') * 1.0e10 / Hubble_h
        H2gas = read_hdf(snap_num=Snapshot, param='H2gas') * 1.0e10 / Hubble_h

        # Filter out invalid data
        valid_indices = (StellarMass > 1e8) & (H2gas > 0)
        StellarMass = StellarMass[valid_indices]
        H2gas = H2gas[valid_indices]
        
        # Calculate the ratio
        Ratio = H2gas / StellarMass
        
        if i == 0:
            # Create a 2D histogram for the first model
            h, xedges, yedges = np.histogram2d(np.log10(StellarMass), np.log10(Ratio), bins=256)
            
            # Define contour levels using a logarithmic scale
            levels = np.logspace(np.log10(h[h>0].min()), np.log10(h.max()), 10)
            
            # Create the contour plot
            plt.contourf(xedges[:-1], yedges[:-1], h.T, cmap='plasma', levels=levels, alpha=0.7)

        # Calculate median line
        bin_width = 0.2
        mass_bins = np.arange(8.0, 12.0 + bin_width, bin_width)
        mass_centers = mass_bins[:-1] + bin_width / 2
        
        median_ratio = np.zeros_like(mass_centers)
        p16 = np.full_like(mass_centers, np.nan)
        p84 = np.full_like(mass_centers, np.nan)
        n_bootstrap = 1000
        rng = np.random.default_rng(42)
        for j in range(len(mass_bins) - 1):
            mask = (np.log10(StellarMass) >= mass_bins[j]) & (np.log10(StellarMass) < mass_bins[j+1])
            data = np.log10(Ratio[mask])
            if data.size > 0:
                median_ratio[j] = np.median(data)
                # Bootstrap
                boot_meds = np.array([
                    np.median(rng.choice(data, size=data.size, replace=True))
                    for _ in range(n_bootstrap)
                ])
                p16[j] = np.percentile(boot_meds, 16)
                p84[j] = np.percentile(boot_meds, 84)
        # Plot median line and bootstrap error band
        # if i == 0:
        #     plt.plot(mass_centers, median_ratio, label=model['label'], color=model['color'], zorder=10, linewidth=4)
        #     plt.fill_between(mass_centers, p16, p84, color=model['color'], alpha=0.2, zorder=9)
        # elif model['name'] == 'millennium_kmt09':
        #     plt.plot(mass_centers, median_ratio, label=model['label'], color=model['color'], zorder=15, linewidth=2)
        #     plt.fill_between(mass_centers, p16, p84, color=model['color'], alpha=0.2, zorder=14)
        # elif model['name'] == 'millennium_k13':
        #     plt.plot(mass_centers, median_ratio, label=model['label'], color=model['color'], zorder=5, linewidth=2)
        #     plt.fill_between(mass_centers, p16, p84, color=model['color'], alpha=0.2, zorder=4)
        # else:
        plt.plot(mass_centers, median_ratio, label=model['label'], color=model['color'], zorder=5)
        plt.fill_between(mass_centers, p16, p84, color=model['color'], alpha=0.2, zorder=4)
    
    # Plot observational data
    data = read_obs_data('MolecularGasRatio_NonDetEQZero.dat')
    if data is not None:
        log_mstar = data[:, 0]
        median = data[:, 1]
        p16 = data[:, 2]
        p84 = data[:, 3]
        mask = (median > -10) & (median < 2) & (p16 > -10) & (p84 > -10)
        yerr_lower = np.abs(median[mask] - p16[mask])
        yerr_upper = np.abs(p84[mask] - median[mask])
        plt.errorbar(log_mstar[mask], median[mask], yerr=[yerr_lower, yerr_upper], 
                    fmt='o', color='gray', markersize=6, capsize=3, 
                    label='xCOLDGASS', zorder=10, markerfacecolor='none', markeredgewidth=1.5)

        from matplotlib.ticker import MaxNLocator
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    
        plt.ylim(-3, 1)
    
    plt.xlabel(r'$\log_{10}\ M_{\star}\ (M_{\odot})$')
    plt.ylabel(r'$\log_{10}\ (M_{\mathrm{H2}} / M_{\star})$')
    
    # Add legend at the bottom
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=len(models)+1, fontsize=10)
    plt.tight_layout()

    outputFile = os.path.join(OutputDir, 'h2_mass_ratio_comparison' + OutputFormat)
    plt.savefig(outputFile)
    print(f"Plot saved to {outputFile}")
    plt.close()

    # ======================= Cold gas MASS RATIO COMPARISON PLOT =======================
    print("Creating Cold gas mass ratio comparison plot...")

    # Create the plot
    plt.figure()

    models = [
        {'name': 'millennium', 'label': 'BR06', 'color': 'white'},
        {'name': 'millennium_gd14', 'label': 'GD14', 'color': 'goldenrod'},
        {'name': 'millennium_kd12', 'label': 'KD12', 'color': 'dodgerblue'},
        {'name': 'millennium_kmt09', 'label': 'KMT09', 'color': 'limegreen'},
        {'name': 'millennium_k13', 'label': 'K13', 'color': 'firebrick'}
    ]

    for i, model in enumerate(models):
        DirName = f'./output/{model["name"]}/'
        
        # Read galaxy properties
        StellarMass = read_hdf(snap_num=Snapshot, param='StellarMass') * 1.0e10 / Hubble_h
        ColdGas = read_hdf(snap_num=Snapshot, param='ColdGas') * 1.0e10 / Hubble_h

        # Filter out invalid data
        valid_indices = (StellarMass > 1e8) & (ColdGas > 0)
        StellarMass = StellarMass[valid_indices]
        ColdGas = ColdGas[valid_indices]
        
        # Calculate the ratio
        Ratio = ColdGas / StellarMass
        
        if i == 0:
            # Create a 2D histogram for the first model
            h, xedges, yedges = np.histogram2d(np.log10(StellarMass), np.log10(Ratio), bins=256)
            
            # Define contour levels using a logarithmic scale
            levels = np.logspace(np.log10(h[h>0].min()), np.log10(h.max()), 10)
            
            # Create the contour plot
            plt.contourf(xedges[:-1], yedges[:-1], h.T, cmap='plasma', levels=levels, alpha=0.7)

        # Calculate median line
        bin_width = 0.2
        mass_bins = np.arange(8.0, 12.0 + bin_width, bin_width)
        mass_centers = mass_bins[:-1] + bin_width / 2
        
        median_ratio = np.zeros_like(mass_centers)
        p16 = np.full_like(mass_centers, np.nan)
        p84 = np.full_like(mass_centers, np.nan)
        n_bootstrap = 1000
        rng = np.random.default_rng(42)
        for j in range(len(mass_bins) - 1):
            mask = (np.log10(StellarMass) >= mass_bins[j]) & (np.log10(StellarMass) < mass_bins[j+1])
            data = np.log10(Ratio[mask])
            if data.size > 0:
                median_ratio[j] = np.median(data)
                # Bootstrap
                boot_meds = np.array([
                    np.median(rng.choice(data, size=data.size, replace=True))
                    for _ in range(n_bootstrap)
                ])
                p16[j] = np.percentile(boot_meds, 16)
                p84[j] = np.percentile(boot_meds, 84)
        # Plot median line and bootstrap error band
        # if i == 0:
        #     plt.plot(mass_centers, median_ratio, label=model['label'], color=model['color'], zorder=10, linewidth=4)
        #     plt.fill_between(mass_centers, p16, p84, color=model['color'], alpha=0.2, zorder=9)
        # elif model['name'] == 'millennium_kmt09':
        #     plt.plot(mass_centers, median_ratio, label=model['label'], color=model['color'], zorder=15, linewidth=2)
        #     plt.fill_between(mass_centers, p16, p84, color=model['color'], alpha=0.2, zorder=14)
        # elif model['name'] == 'millennium_k13':
        #     plt.plot(mass_centers, median_ratio, label=model['label'], color=model['color'], zorder=5, linewidth=2)
        #     plt.fill_between(mass_centers, p16, p84, color=model['color'], alpha=0.2, zorder=4)
        # else:
        plt.plot(mass_centers, median_ratio, label=model['label'], color=model['color'], zorder=5)
        plt.fill_between(mass_centers, p16, p84, color=model['color'], alpha=0.2, zorder=4)
    
    # Plot observational data
    data = read_obs_data('NeutralGasRatio_NonDetEQZero.dat')
    if data is not None:
        log_mstar = data[:, 0]
        median = data[:, 1]
        p16 = data[:, 2]
        p84 = data[:, 3]
        mask = (median > -10) & (median < 2) & (p16 > -10) & (p84 > -10)
        yerr_lower = np.abs(median[mask] - p16[mask])
        yerr_upper = np.abs(p84[mask] - median[mask])
        plt.errorbar(log_mstar[mask], median[mask], yerr=[yerr_lower, yerr_upper], 
                    fmt='o', color='gray', markersize=6, capsize=3, 
                    label='xGASS', zorder=10, markerfacecolor='none', markeredgewidth=1.5)

        from matplotlib.ticker import MaxNLocator
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    
        plt.ylim(-3, 1)
    
    plt.xlabel(r'$\log_{10}\ M_{\star}\ (M_{\odot})$')
    plt.ylabel(r'$\log_{10}\ (M_{\mathrm{cold\ gas}} / M_{\star})$')
    
    # Add legend at the bottom
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=len(models)+1, fontsize=10)
    plt.tight_layout()

    outputFile = os.path.join(OutputDir, 'cold_gas_mass_ratio_comparison' + OutputFormat)
    plt.savefig(outputFile)
    print(f"Plot saved to {outputFile}")
    plt.close()
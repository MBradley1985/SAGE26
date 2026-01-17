#!/usr/bin/env python

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import os

from scipy.ndimage import gaussian_filter

import warnings
warnings.filterwarnings("ignore")

# ========================== USER OPTIONS ==========================

# File details
DirName = './output/millennium/'
FileName = 'model_0.hdf5'

# Simulation details
Hubble_h = 0.73        # Hubble parameter
BoxSize = 62.5         # h-1 Mpc
VolumeFraction = 1.0   # Fraction of the full volume output by the model
FirstSnap = 0          # First snapshot to read
LastSnap = 63          # Last snapshot to read
BaryonFrac = 0.17      # Cosmic baryon fraction

redshifts = [127.000, 79.998, 50.000, 30.000, 19.916, 18.244, 16.725, 15.343, 14.086, 12.941, 11.897, 10.944, 10.073,
             9.278, 8.550, 7.883, 7.272, 6.712, 6.197, 5.724, 5.289, 4.888, 4.520, 4.179, 3.866, 3.576, 3.308, 3.060,
             2.831, 2.619, 2.422, 2.239, 2.070, 1.913, 1.766, 1.630, 1.504, 1.386, 1.276, 1.173, 1.078, 0.989, 0.905,
             0.828, 0.755, 0.687, 0.624, 0.564, 0.509, 0.457, 0.408, 0.362, 0.320, 0.280, 0.242, 0.208, 0.175, 0.144,
             0.116, 0.089, 0.064, 0.041, 0.020, 0.000]

# Snapshots to plot (8 redshifts for 2x4 grid)
# z ~ 0, 0.5, 1, 1.5, 2, 2.5, 3, 4
ICS_snaps = [63, 49, 40, 35, 32, 28, 25, 20]

OutputFormat = '.pdf'
plt.rcParams["figure.figsize"] = (10, 16)
plt.rcParams["figure.dpi"] = 96
plt.rcParams["font.size"] = 12

# ==================================================================

def read_hdf(snap_num=None, param=None):
    property = h5.File(DirName + FileName, 'r')
    return np.array(property[snap_num][param])

# ==================================================================

if __name__ == '__main__':

    print('Running ICS Fraction vs Mvir Grid Plot\n')

    OutputDir = DirName + 'plots/'
    if not os.path.exists(OutputDir):
        os.makedirs(OutputDir)

    # Create 2x4 grid of subplots
    fig, axes = plt.subplots(4, 2, figsize=(10, 16))
    axes = axes.flatten()

    for idx, snap in enumerate(ICS_snaps):
        ax = axes[idx]
        Snapshot = 'Snap_' + str(snap)
        z = redshifts[snap]

        print(f'Processing snapshot {snap} (z = {z:.2f})')

        # Read data
        Mvir = read_hdf(snap_num=Snapshot, param='Mvir') * 1.0e10 / Hubble_h
        ICS = read_hdf(snap_num=Snapshot, param='IntraClusterStars') * 1.0e10 / Hubble_h
        Type = read_hdf(snap_num=Snapshot, param='Type')

        # Select central galaxies with valid Mvir and ICS
        w = np.where((Type == 0) & (Mvir > 0) & (ICS > 0))[0]

        if len(w) == 0:
            ax.text(0.5, 0.5, f'z = {z:.2f}\nNo data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14)
            ax.set_xlim(11, 15)
            ax.set_ylim(0, 0.5)
            continue

        log_Mvir = np.log10(Mvir[w])
        ICS_fraction = ICS[w] / (BaryonFrac * Mvir[w])

        # Define bins for contour plot
        x_bins = np.linspace(11, 15, 50)
        y_bins = np.linspace(0, 0.15, 50)

        # Create 2D histogram for contour
        H, xedges, yedges = np.histogram2d(log_Mvir, ICS_fraction, bins=[x_bins, y_bins])

        # Transpose for correct orientation
        H = H.T

        # Get bin centers
        x_centers = (xedges[:-1] + xedges[1:]) / 2
        y_centers = (yedges[:-1] + yedges[1:]) / 2

        # Create meshgrid for contour
        X, Y = np.meshgrid(x_centers, y_centers)

        # Plot filled contours
        levels = np.linspace(1, H.max(), 10)
        if H.max() > 1:
            contour = ax.contourf(X, Y, H, levels=levels, cmap='plasma', extend='max')

        # Calculate median in bins
        bin_width = 0.25
        bin_edges = np.arange(11, 15.5, bin_width)
        bin_centers = bin_edges[:-1] + bin_width / 2

        median_values = []
        valid_bins = []

        for i in range(len(bin_edges) - 1):
            mask = (log_Mvir >= bin_edges[i]) & (log_Mvir < bin_edges[i + 1])
            if np.sum(mask) > 10:  # Only compute if enough galaxies
                median_values.append(np.median(ICS_fraction[mask]))
                valid_bins.append(bin_centers[i])

        # Plot median line
        if len(valid_bins) > 0:
            valid_bins = np.array(valid_bins)
            median_values = np.array(median_values)
            ax.plot(valid_bins, median_values, 'k-', lw=2)

        # Formatting
        ax.set_xlim(11, 14)
        ax.set_ylim(0, 0.10)
        ax.set_xticks([11, 12, 13, 14])
        ax.set_yticks([0.00, 0.05, 0.10])

        # Only add y-axis label for left column (idx 0, 2, 4, 6)
        if idx % 2 == 1:
            ax.set_yticklabels([])
        else:
            ax.set_yticklabels(['0.00', '0.05', '0.10'])

        # Add redshift label
        ax.text(0.05, 0.95, f'z = {z:.2f}', transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top', ha='left')

    # Add shared axis labels
    fig.supxlabel(r'$\log_{10} M_{\mathrm{vir}}\ [M_{\odot}]$', fontsize=14, y=0.02)
    fig.supylabel(r'$f_{\mathrm{ICS}} = M_{\mathrm{ICS}} / (f_b \times M_{\mathrm{vir}})$', fontsize=14, x=0.005)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.06, left=0.08)

    outputFile = OutputDir + 'ICS_fraction_vs_Mvir_redshift_grid' + OutputFormat
    plt.savefig(outputFile, dpi=300, bbox_inches='tight')
    print(f'\nSaved file to {outputFile}')
    plt.close()

    print('\nICS fraction vs Mvir plot completed!')

    # ==================================================================
    # ICS Mass vs Mvir (Halo Mass)
    # ==================================================================

    print('\nRunning ICS Mass vs Mvir Grid Plot\n')

    # Create 2x4 grid of subplots
    fig, axes = plt.subplots(4, 2, figsize=(10, 16))
    axes = axes.flatten()

    for idx, snap in enumerate(ICS_snaps):
        ax = axes[idx]
        Snapshot = 'Snap_' + str(snap)
        z = redshifts[snap]

        print(f'Processing snapshot {snap} (z = {z:.2f})')

        # Read data
        Mvir = read_hdf(snap_num=Snapshot, param='Mvir') * 1.0e10 / Hubble_h
        ICS = read_hdf(snap_num=Snapshot, param='IntraClusterStars') * 1.0e10 / Hubble_h
        Type = read_hdf(snap_num=Snapshot, param='Type')

        # Select central galaxies with valid Mvir and ICS
        w = np.where((Type == 0) & (Mvir > 0) & (ICS > 0))[0]

        if len(w) == 0:
            ax.text(0.5, 0.5, f'z = {z:.2f}\nNo data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14)
            ax.set_xlim(11, 15)
            ax.set_ylim(7, 12)
            continue

        log_Mvir = np.log10(Mvir[w])
        log_ICS = np.log10(ICS[w])

        # Define bins for contour plot
        x_bins = np.linspace(11, 15, 50)
        y_bins = np.linspace(7, 12, 50)

        # Create 2D histogram for contour
        H, xedges, yedges = np.histogram2d(log_Mvir, log_ICS, bins=[x_bins, y_bins])

        # Transpose for correct orientation
        H = H.T

        # Get bin centers
        x_centers = (xedges[:-1] + xedges[1:]) / 2
        y_centers = (yedges[:-1] + yedges[1:]) / 2

        # Create meshgrid for contour
        X, Y = np.meshgrid(x_centers, y_centers)

        # Plot filled contours
        levels = np.linspace(1, H.max(), 10)
        if H.max() > 1:
            contour = ax.contourf(X, Y, H, levels=levels, cmap='plasma', extend='max')

        # Calculate median in bins
        bin_width = 0.25
        bin_edges = np.arange(11, 15.5, bin_width)
        bin_centers = bin_edges[:-1] + bin_width / 2

        median_values = []
        valid_bins = []

        for i in range(len(bin_edges) - 1):
            mask = (log_Mvir >= bin_edges[i]) & (log_Mvir < bin_edges[i + 1])
            if np.sum(mask) > 10:
                median_values.append(np.median(log_ICS[mask]))
                valid_bins.append(bin_centers[i])

        # Plot median line
        if len(valid_bins) > 0:
            valid_bins = np.array(valid_bins)
            median_values = np.array(median_values)
            ax.plot(valid_bins, median_values, 'k-', lw=2)

        # Formatting
        ax.set_xlim(11, 15)
        ax.set_ylim(7, 12)
        ax.set_xticks([11, 12, 13, 14, 15])
        ax.set_yticks([7, 8, 9, 10, 11, 12])

        # Only show y-axis labels for left column
        if idx % 2 == 1:
            ax.set_yticklabels([])

        # Add redshift label
        ax.text(0.05, 0.95, f'z = {z:.2f}', transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top', ha='left')

    # Add shared axis labels
    fig.supxlabel(r'$\log_{10} M_{\mathrm{vir}}\ [M_{\odot}]$', fontsize=14, y=0.02)
    fig.supylabel(r'$\log_{10} M_{\mathrm{ICS}}\ [M_{\odot}]$', fontsize=14, x=0.005)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.06, left=0.08)

    outputFile = OutputDir + 'ICS_mass_vs_Mvir_redshift_grid' + OutputFormat
    plt.savefig(outputFile, dpi=300, bbox_inches='tight')
    print(f'\nSaved file to {outputFile}')
    plt.close()

    print('\nICS mass vs Mvir plot completed!')

    # ==================================================================
    # ICS Mass vs Stellar Mass
    # ==================================================================

    print('\nRunning ICS Mass vs Stellar Mass Grid Plot\n')

    # Create 2x4 grid of subplots
    fig, axes = plt.subplots(4, 2, figsize=(10, 16))
    axes = axes.flatten()

    for idx, snap in enumerate(ICS_snaps):
        ax = axes[idx]
        Snapshot = 'Snap_' + str(snap)
        z = redshifts[snap]

        print(f'Processing snapshot {snap} (z = {z:.2f})')

        # Read data
        StellarMass = read_hdf(snap_num=Snapshot, param='StellarMass') * 1.0e10 / Hubble_h
        ICS = read_hdf(snap_num=Snapshot, param='IntraClusterStars') * 1.0e10 / Hubble_h
        Type = read_hdf(snap_num=Snapshot, param='Type')

        # Select central galaxies with valid StellarMass and ICS
        w = np.where((Type == 0) & (StellarMass > 0) & (ICS > 0))[0]

        if len(w) == 0:
            ax.text(0.5, 0.5, f'z = {z:.2f}\nNo data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14)
            ax.set_xlim(8, 12)
            ax.set_ylim(7, 12)
            continue

        log_StellarMass = np.log10(StellarMass[w])
        log_ICS = np.log10(ICS[w])

        # Define bins for contour plot
        x_bins = np.linspace(8, 12, 50)
        y_bins = np.linspace(7, 12, 50)

        # Create 2D histogram for contour
        H, xedges, yedges = np.histogram2d(log_StellarMass, log_ICS, bins=[x_bins, y_bins])

        # Transpose for correct orientation
        H = H.T

        # Get bin centers
        x_centers = (xedges[:-1] + xedges[1:]) / 2
        y_centers = (yedges[:-1] + yedges[1:]) / 2

        # Create meshgrid for contour
        X, Y = np.meshgrid(x_centers, y_centers)

        # Plot filled contours
        levels = np.linspace(1, H.max(), 10)
        if H.max() > 1:
            contour = ax.contourf(X, Y, H, levels=levels, cmap='plasma', extend='max')

        # Calculate median in bins
        bin_width = 0.25
        bin_edges = np.arange(8, 12.5, bin_width)
        bin_centers = bin_edges[:-1] + bin_width / 2

        median_values = []
        valid_bins = []

        for i in range(len(bin_edges) - 1):
            mask = (log_StellarMass >= bin_edges[i]) & (log_StellarMass < bin_edges[i + 1])
            if np.sum(mask) > 10:
                median_values.append(np.median(log_ICS[mask]))
                valid_bins.append(bin_centers[i])

        # Plot median line
        if len(valid_bins) > 0:
            valid_bins = np.array(valid_bins)
            median_values = np.array(median_values)
            ax.plot(valid_bins, median_values, 'k-', lw=2)

        # Formatting
        ax.set_xlim(8, 12)
        ax.set_ylim(7, 12)
        ax.set_xticks([8, 9, 10, 11, 12])
        ax.set_yticks([7, 8, 9, 10, 11, 12])

        # Only show y-axis labels for left column
        if idx % 2 == 1:
            ax.set_yticklabels([])

        # Add redshift label
        ax.text(0.05, 0.95, f'z = {z:.2f}', transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top', ha='left')

    # Add shared axis labels
    fig.supxlabel(r'$\log_{10} M_{\star}\ [M_{\odot}]$', fontsize=14, y=0.02)
    fig.supylabel(r'$\log_{10} M_{\mathrm{ICS}}\ [M_{\odot}]$', fontsize=14, x=0.005)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.06, left=0.08)

    outputFile = OutputDir + 'ICS_mass_vs_StellarMass_redshift_grid' + OutputFormat
    plt.savefig(outputFile, dpi=300, bbox_inches='tight')
    print(f'\nSaved file to {outputFile}')
    plt.close()

    print('\nICS mass vs Stellar Mass plot completed!')

    # ==================================================================
    # ICS Fraction vs Stellar Mass
    # ==================================================================

    print('\nRunning ICS Fraction vs Stellar Mass Grid Plot\n')

    # Create 2x4 grid of subplots
    fig, axes = plt.subplots(4, 2, figsize=(10, 16))
    axes = axes.flatten()

    for idx, snap in enumerate(ICS_snaps):
        ax = axes[idx]
        Snapshot = 'Snap_' + str(snap)
        z = redshifts[snap]

        print(f'Processing snapshot {snap} (z = {z:.2f})')

        # Read data
        Mvir = read_hdf(snap_num=Snapshot, param='Mvir') * 1.0e10 / Hubble_h
        StellarMass = read_hdf(snap_num=Snapshot, param='StellarMass') * 1.0e10 / Hubble_h
        ICS = read_hdf(snap_num=Snapshot, param='IntraClusterStars') * 1.0e10 / Hubble_h
        Type = read_hdf(snap_num=Snapshot, param='Type')

        # Select central galaxies with valid StellarMass and ICS
        w = np.where((Type == 0) & (StellarMass > 0) & (Mvir > 0) & (ICS > 0))[0]

        if len(w) == 0:
            ax.text(0.5, 0.5, f'z = {z:.2f}\nNo data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14)
            ax.set_xlim(8, 12)
            ax.set_ylim(0, 0.15)
            continue

        log_StellarMass = np.log10(StellarMass[w])
        ICS_fraction = ICS[w] / (BaryonFrac * Mvir[w])

        # Define bins for contour plot
        x_bins = np.linspace(8, 12, 50)
        y_bins = np.linspace(0, 0.15, 50)

        # Create 2D histogram for contour
        H, xedges, yedges = np.histogram2d(log_StellarMass, ICS_fraction, bins=[x_bins, y_bins])

        # Transpose for correct orientation
        H = H.T

        # Get bin centers
        x_centers = (xedges[:-1] + xedges[1:]) / 2
        y_centers = (yedges[:-1] + yedges[1:]) / 2

        # Create meshgrid for contour
        X, Y = np.meshgrid(x_centers, y_centers)

        # Plot filled contours
        levels = np.linspace(1, H.max(), 10)
        if H.max() > 1:
            contour = ax.contourf(X, Y, H, levels=levels, cmap='plasma', extend='max')

        # Calculate median in bins
        bin_width = 0.25
        bin_edges = np.arange(8, 12.5, bin_width)
        bin_centers = bin_edges[:-1] + bin_width / 2

        median_values = []
        valid_bins = []

        for i in range(len(bin_edges) - 1):
            mask = (log_StellarMass >= bin_edges[i]) & (log_StellarMass < bin_edges[i + 1])
            if np.sum(mask) > 10:  # Only compute if enough galaxies
                median_values.append(np.median(ICS_fraction[mask]))
                valid_bins.append(bin_centers[i])

        # Plot median line
        if len(valid_bins) > 0:
            valid_bins = np.array(valid_bins)
            median_values = np.array(median_values)
            ax.plot(valid_bins, median_values, 'k-', lw=2)

        # Formatting
        ax.set_xlim(8, 12)
        ax.set_ylim(0, 0.10)
        ax.set_xticks([8, 9, 10, 11, 12])
        ax.set_yticks([0.00, 0.05, 0.10])

        # Only add y-axis label for left column (idx 0, 2, 4, 6)
        if idx % 2 == 1:
            ax.set_yticklabels([])
        else:
            ax.set_yticklabels(['0.00', '0.05', '0.10'])

        # Add redshift label
        ax.text(0.05, 0.95, f'z = {z:.2f}', transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top', ha='left')

    # Add shared axis labels
    fig.supxlabel(r'$\log_{10} M_{\star}\ [M_{\odot}]$', fontsize=14, y=0.02)
    fig.supylabel(r'$f_{\mathrm{ICS}} = M_{\mathrm{ICS}} / (f_b \times M_{\mathrm{vir}})$', fontsize=14, x=0.005)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.06, left=0.08)

    outputFile = OutputDir + 'ICS_fraction_vs_StellarMass_redshift_grid' + OutputFormat
    plt.savefig(outputFile, dpi=300, bbox_inches='tight')
    print(f'\nSaved file to {outputFile}')
    plt.close()

    print('\nICS fraction vs Stellar Mass plot completed!')

    # ==================================================================
    # ICS Fraction vs Redshift (z=0 to z=1.25)
    # ==================================================================

    print('\nRunning ICS Fraction vs Redshift Plot\n')

    # Find snapshots in redshift range z=0 to z=1.25
    z_snaps = [snap for snap in range(LastSnap + 1) if redshifts[snap] <= 1.25]

    # Collect all data
    all_redshifts = []
    all_ICS_fractions = []

    for snap in z_snaps:
        Snapshot = 'Snap_' + str(snap)
        z = redshifts[snap]

        print(f'Processing snapshot {snap} (z = {z:.2f})')

        # Read data
        Mvir = read_hdf(snap_num=Snapshot, param='Mvir') * 1.0e10 / Hubble_h
        ICS = read_hdf(snap_num=Snapshot, param='IntraClusterStars') * 1.0e10 / Hubble_h
        Type = read_hdf(snap_num=Snapshot, param='Type')

        # Select central galaxies with valid Mvir and ICS
        w = np.where((Type == 0) & (Mvir > 0) & (ICS > 0))[0]

        if len(w) > 0:
            ICS_fraction = ICS[w] / (BaryonFrac * Mvir[w])
            all_redshifts.extend([z] * len(w))
            all_ICS_fractions.extend(ICS_fraction)

    all_redshifts = np.array(all_redshifts)
    all_ICS_fractions = np.array(all_ICS_fractions)

    # Create single figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Define bins for contour plot
    x_bins = np.linspace(0, 1.25, 50)
    y_bins = np.linspace(0, 0.10, 50)

    # Create 2D histogram for contour
    H, xedges, yedges = np.histogram2d(all_redshifts, all_ICS_fractions, bins=[x_bins, y_bins])

    # Transpose for correct orientation
    H = H.T

    # Apply Gaussian smoothing to fill gaps between snapshots
    H = gaussian_filter(H, sigma=1.5)

    # Get bin centers
    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2

    # Create meshgrid for contour
    X, Y = np.meshgrid(x_centers, y_centers)

    # Plot filled contours
    levels = np.linspace(1, H.max(), 10)
    if H.max() > 1:
        contour = ax.contourf(X, Y, H, levels=levels, cmap='plasma', extend='max')

    # Calculate median and max in bins
    bin_width = 0.1
    bin_edges = np.arange(0, 1.35, bin_width)
    bin_centers_med = bin_edges[:-1] + bin_width / 2

    median_values = []
    max_values = []
    valid_bins = []

    for i in range(len(bin_edges) - 1):
        mask = (all_redshifts >= bin_edges[i]) & (all_redshifts < bin_edges[i + 1])
        if np.sum(mask) > 10:
            median_values.append(np.median(all_ICS_fractions[mask]))
            max_values.append(np.max(all_ICS_fractions[mask]))
            valid_bins.append(bin_centers_med[i])

    # Plot median and max lines
    if len(valid_bins) > 0:
        valid_bins = np.array(valid_bins)
        median_values = np.array(median_values)
        max_values = np.array(max_values)
        ax.plot(valid_bins, median_values, 'k-', lw=2, label='SAGE Median')
        ax.plot(valid_bins, max_values, 'k--', lw=2, label='SAGE Maximum')

    # Observational data (values in % converted to fraction)
    # Pentagons - Burke, Collins, Stott and Hilton - ICL @ z=1, 2012
    redshifts_1 = [0.9468354430379745, 0.8303797468354429, 0.7949367088607594, 0.8075949367088605, 1.2227848101265821]
    ihs_fraction_1 = np.array([1.415094339622641, 2.594339622641499, 3.7735849056603783, 1.5330188679245182, 2.3584905660377373]) / 100
    # Circles (gray) - Montes and Trujillo - ICL at the Frontier, 2018
    redshifts_2 = [0.5341772151898734, 0.5443037974683542, 0.36708860759493667, 0.39746835443037976, 0.3417721518987341,
                   0.30379746835443033, 0.04810126582278479]
    ihs_fraction_2 = np.array([1.5330188679245182, 0, 1.0613207547169807, 1.5330188679245182, 2.7122641509433976,
                      3.3018867924528266, 8.60849056603773]) / 100
    # Squares - Burke, Hilton and Collins, ICL and CLASH, 2015
    redshifts_3 = [0.4025316455696200, 0.3873417721518990, 0.39746835443038000, 0.33924050632911400, 0.3443037974683540,
                   0.3417721518987340, 0.2911392405063290, 0.2253164556962030, 0.2177215189873420, 0.21265822784810100,
                   0.19493670886075900, 0.17721518987341800]
    ihs_fraction_3 = np.array([2.594339622641500, 2.7122641509434000, 3.3018867924528300, 5.542452830188670, 6.014150943396220,
                      7.193396226415100, 12.971698113207500, 12.500000000000000, 16.27358490566040, 18.042452830188700,
                      16.863207547169800, 23.113207547169800]) / 100
    # Crosses - Furnell et al., Growth of ICL in XCS-HSC from 0.1<z<0.5, 2021
    redshifts_4 = [0.1443037974683540, 0.12658227848101300, 0.12151898734177200, 0.08101265822784800, 0.2253164556962030,
                   0.21518987341772100, 0.2556962025316460, 0.3063291139240510, 0.260759493670886, 0.2936708860759490, 0.3215189873417720,
                   0.3417721518987340, 0.3721518987341770, 0.3367088607594940, 0.37721518987341800, 0.3291139240506330, 0.4962025316455700,
                   0.42531645569620200, 0.10886075949367100]
    ihs_fraction_4 = np.array([38.561320754717000, 30.660377358490600, 31.014150943396200, 28.891509433962300, 26.533018867924500,
                      23.58490566037740, 28.5377358490566, 29.716981132075500, 32.54716981132080, 27.476415094339600, 27.594339622641500,
                      26.650943396226400, 19.81132075471700, 18.867924528301900, 15.448113207547200, 15.330188679245300,
                      11.320754716981100, 9.669811320754720, 31.603773584905700]) / 100
    # Down Triangles - Feldmeier et al., Deep CCD, 2004
    redshifts_6 = [0.16202531645569600, 0.16202531645569600, 0.16202531645569600, 0.18481012658227800]
    ihs_fraction_6 = np.array([15.212264150943400, 12.146226415094300, 10.259433962264100, 7.311320754716980]) / 100
    # Black Circles - Montes and Trujillo - ICL at the Frontier, 2018
    redshifts_7 = [0.30126582278481000, 0.38987341772151900, 0.3417721518987340, 0.5367088607594940,
                   0.5367088607594940, 0.36962025316455700, 0.043037974683544300]
    ihs_fraction_7 = np.array([7.665094339622630, 8.60849056603773, 13.089622641509400, 6.603773584905650,
                      5.778301886792450, 4.834905660377360, 10.849056603773600]) / 100
    # Star - Ko and Jee, Existence of ICL at z = 1.24, 2018
    redshifts_8 = [1.2379746835443037]
    ihs_fraction_8 = np.array([9.905660377358487]) / 100
    # Black Diamond - Kluge et al., ICL and host Cluster, 2021
    redshifts_9 = [0.030379746835442978]
    ihs_fraction_9 = np.array([17.924528301886788]) / 100
    # Plus - Zibetti et al., IGS in z=0.25 clusters, 2005
    redshifts_10 = [0.24303797468354427]
    ihs_fraction_10 = np.array([10.849056603773576]) / 100
    # Triangle - Presotto et al., ICL in CLASH-VLT cluster MACS J1206.2-0947, 2014
    redshifts_11 = [0.4354430379746834]
    ihs_fraction_11 = np.array([12.264150943396224]) / 100
    # Black Triangle - Presotto et al., 2014
    redshifts_12 = [0.43291139240506316]
    ihs_fraction_12 = np.array([5.542452830188672]) / 100
    # Black Side Triangle - Spavone et al., Fornax Deep Survey, 2020
    redshifts_13 = [0]
    ihs_fraction_13 = np.array([34.08018867924528]) / 100

    # Plot observational data (no legend)
    ax.scatter(redshifts_1, ihs_fraction_1, marker='p', color='gray', edgecolors='black', s=60, zorder=5)
    ax.scatter(redshifts_2, ihs_fraction_2, marker='o', color='gray', edgecolors='black', s=60, zorder=5)
    ax.scatter(redshifts_3, ihs_fraction_3, marker='s', color='gray', edgecolors='black', s=60, zorder=5)
    ax.scatter(redshifts_4, ihs_fraction_4, marker='X', color='gray', edgecolors='black', s=60, zorder=5)
    ax.scatter(redshifts_6, ihs_fraction_6, marker='v', color='gray', edgecolors='black', s=60, zorder=5)
    ax.scatter(redshifts_7, ihs_fraction_7, marker='o', color='k', edgecolors='gray', s=60, zorder=5)
    ax.scatter(redshifts_8, ihs_fraction_8, marker='*', color='gray', edgecolors='black', s=80, zorder=5)
    ax.scatter(redshifts_9, ihs_fraction_9, marker='d', color='k', edgecolors='gray', s=60, zorder=5)
    ax.scatter(redshifts_10, ihs_fraction_10, marker='P', color='gray', edgecolors='black', s=60, zorder=5)
    ax.scatter(redshifts_11, ihs_fraction_11, marker='^', color='gray', edgecolors='black', s=60, zorder=5)
    ax.scatter(redshifts_12, ihs_fraction_12, marker='^', color='k', edgecolors='gray', s=60, zorder=5)
    ax.scatter(redshifts_13, ihs_fraction_13, marker='<', color='k', edgecolors='gray', s=60, zorder=5)

    # Simulation lines (with legend)
    # Rudick, Mihos and McBride, Quantity of ICL, 2011
    redshifts_5 = [0.004011349760203840, 0.023056412555524700, 0.05248969142102060, 0.08192297028651650, 0.12001309587715800,
                   0.1581032214678000, 0.19696284454512100, 0.23774621133914200, 0.2758363369297840, 0.31392646252042500, 0.3546136421286110,
                   0.38880818669293700, 0.42681174381632700, 0.4662869648829930, 0.5043770904736340, 0.542467216064276, 0.5805573416549180,
                   0.6186474672455600, 0.6567375928362010, 0.6948277184268430, 0.7329178440174850, 0.7710079696081270, 0.8090980951987680,
                   0.84718822078941, 0.8852783463800520, 0.9233684719706940, 0.9614585975613350, 0.999548723151977, 1.0341761100525600,
                   1.0711119894131800, 1.1449837481344300, 1.183073873725070, 1.2211639993157100, 1.2592541249063500, 1.2817619263917300, 1.1083123425692700]
    ihs_fraction_5 = np.array([17.035633925701400, 15.255522799283600, 13.375138908022200, 11.340768199977500, 11.708336986061200, 12.328609312577500,
                      12.848617042445400, 12.722826835652300, 11.208443436987300, 11.164335182657300, 11.105524176883900, 11.044875327180100,
                      10.871015291362500, 10.55417099775830, 10.091034327292800, 9.55438389961052, 8.833949078886400, 8.635461934401190, 9.150058234918420,
                      8.554596801462770, 7.782702350686930, 7.51070144898496, 7.701837217748510, 7.643026211975110, 7.326917055943100, 6.790266628260850,
                      6.6579418652707000, 6.7755638768175, 7.223997795839650, 6.9544473527115800, 6.15069694047515, 5.900750165938210,
                      5.966912547433290, 6.047777680371710, 6.011020801763340, 5.882352941176470]) / 100

    # Tang et al., hydrodynamical simulation, 2018
    redshifts_14 = [0.1021897810218980, 0.12408759124087600, 0.13625304136253000,
                    0.15571776155717800, 0.1800486618004870, 0.2068126520681270,
                    0.22871046228710500, 0.25304136253041400, 0.2700729927007300,
                    0.291970802919708, 0.3187347931873480, 0.34063260340632600,
                    0.3673965936739660, 0.3917274939172750, 0.42092457420924600,
                    0.44282238442822400, 0.45985401459854000, 0.4841849148418490]
    ihs_fraction_14 = np.array([22.675736961451200, 21.995464852607700, 20.975056689342400,
                       19.954648526077100, 18.934240362811800, 18.140589569161000,
                       17.006802721088400, 16.099773242630400, 15.532879818594100,
                       14.625850340136000, 13.718820861678000, 13.151927437641700,
                       12.131519274376400, 10.997732426303800, 9.523809523809520,
                       8.73015873015872, 7.596371882086170, 6.68934240362811]) / 100

    ax.plot(redshifts_5, ihs_fraction_5, linestyle='--', color='plum', lw=2, label='Rudick et al. 2011')
    ax.plot(redshifts_14, ihs_fraction_14, linestyle='--', color='royalblue', lw=2, label='Tang et al. 2018')

    # Formatting
    ax.set_xlim(0, 1.25)
    ax.set_ylim(0, 0.40)
    ax.set_yticks([0.00, 0.10, 0.20, 0.30, 0.40])
    ax.set_yticklabels(['0.00', '0.10', '0.20', '0.30', '0.40'])

    ax.set_xlabel(r'$z$', fontsize=14)
    ax.set_ylabel(r'$f_{\mathrm{ICS}} = M_{\mathrm{ICS}} / (f_b \times M_{\mathrm{vir}})$', fontsize=14)

    # Legend
    ax.legend(loc='upper right', fontsize=10, frameon=False)

    plt.tight_layout()

    outputFile = OutputDir + 'ICS_fraction_vs_redshift' + OutputFormat
    plt.savefig(outputFile, dpi=300, bbox_inches='tight')
    print(f'\nSaved file to {outputFile}')
    plt.close()

    print('\nICS fraction vs redshift plot completed!')

    # ==================================================================
    # ICS Mass Function (split by halo mass bins)
    # ==================================================================

    print('\nRunning ICS Mass Function Plot\n')

    from matplotlib.lines import Line2D

    # Volume for number density calculation
    volume = (BoxSize / Hubble_h)**3.0 * VolumeFraction

    # Mass function parameters
    mi_mf = 7.5
    ma_mf = 12.75
    binwidth_mf = 0.25
    NB = int((ma_mf - mi_mf) / binwidth_mf)

    # Halo mass bin edges (in log10 Msun)
    halo_bin_edges = [10.5, 12, 13.5, 17]
    subset_colors = ['red', 'green', 'blue']
    legend_labels = [r'$10^{10.5} < M_{\mathrm{halo}} < 10^{12}$',
                     r'$10^{12} < M_{\mathrm{halo}} < 10^{13.5}$',
                     r'$10^{13.5} < M_{\mathrm{halo}}$']

    # Create 4x2 grid
    fig, axes = plt.subplots(4, 2, figsize=(12, 18))
    axes = axes.flatten()

    for idx, snap in enumerate(ICS_snaps):
        ax = axes[idx]
        Snapshot = 'Snap_' + str(snap)
        z = redshifts[snap]

        print(f'Processing snapshot {snap} (z = {z:.2f})')

        # Read data
        Mvir = read_hdf(snap_num=Snapshot, param='Mvir') * 1.0e10 / Hubble_h
        ICS = read_hdf(snap_num=Snapshot, param='IntraClusterStars') * 1.0e10 / Hubble_h
        Type = read_hdf(snap_num=Snapshot, param='Type')

        # Select central galaxies with valid ICS
        w = np.where((Type == 0) & (ICS > 0))[0]
        Mvir_sel = Mvir[w]
        ICS_sel = ICS[w]
        log_Mvir = np.log10(Mvir_sel)
        log_ICS = np.log10(ICS_sel)

        # Plot mass function for each halo mass bin
        for bin_idx in range(len(halo_bin_edges) - 1):
            lo = halo_bin_edges[bin_idx]
            hi = halo_bin_edges[bin_idx + 1]
            mask = (log_Mvir >= lo) & (log_Mvir < hi)
            ICS_in_bin = log_ICS[mask]

            if len(ICS_in_bin) > 0:
                counts, binedges = np.histogram(ICS_in_bin, range=(mi_mf, ma_mf), bins=NB)
                xaxeshisto = binedges[:-1] + 0.5 * binwidth_mf
                phi = counts / volume / binwidth_mf
                ax.plot(xaxeshisto, phi, color=subset_colors[bin_idx], alpha=0.7)
                ax.fill_between(xaxeshisto, 0, phi, color=subset_colors[bin_idx], alpha=0.1)

        # Plot overall mass function
        counts, binedges = np.histogram(log_ICS, range=(mi_mf, ma_mf), bins=NB)
        xaxeshisto = binedges[:-1] + 0.5 * binwidth_mf
        phi = counts / volume / binwidth_mf
        ax.plot(xaxeshisto, phi, color='black')

        # Formatting
        ax.set_yscale('log')
        ax.set_xlim(7.5, 12.0)
        ax.set_ylim(1e-8, 1e-1)
        ax.set_xticks([8, 9, 10, 11, 12])

        # Only show y-axis labels for left column
        if idx % 2 == 1:
            ax.set_yticklabels([])

        # Add redshift label
        ax.text(0.05, 0.95, f'z = {z:.2f}', transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top', ha='left')

    # Add shared axis labels
    fig.supxlabel(r'$\log_{10} M_{\mathrm{ICS}}\ [M_{\odot}]$', fontsize=14, y=0.03)
    fig.supylabel(r'$\phi\ (\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1})$', fontsize=14, x=0.005)

    # Create legend at bottom of figure
    custom_lines = [Line2D([0], [0], color=c, lw=2) for c in subset_colors]
    custom_lines.append(Line2D([0], [0], color='black', lw=2))
    fig.legend(custom_lines, legend_labels + ['Overall'],
               loc='lower center', ncol=4, fontsize=10, frameon=False)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.06, left=0.08)

    outputFile = OutputDir + 'ICS_mass_function' + OutputFormat
    plt.savefig(outputFile, dpi=300, bbox_inches='tight')
    print(f'\nSaved file to {outputFile}')
    plt.close()

    print('\nICS mass function plot completed!')

    # ==================================================================
    # ICS Metallicity vs Stellar Metallicity
    # ==================================================================

    print('\nRunning ICS Metallicity vs Stellar Metallicity Grid Plot\n')

    # Create 2x4 grid of subplots
    fig, axes = plt.subplots(4, 2, figsize=(10, 16))
    axes = axes.flatten()

    for idx, snap in enumerate(ICS_snaps):
        ax = axes[idx]
        Snapshot = 'Snap_' + str(snap)
        z = redshifts[snap]

        print(f'Processing snapshot {snap} (z = {z:.2f})')

        # Read data
        StellarMass = read_hdf(snap_num=Snapshot, param='StellarMass') * 1.0e10 / Hubble_h
        MetalsStellarMass = read_hdf(snap_num=Snapshot, param='MetalsStellarMass') * 1.0e10 / Hubble_h
        ICS = read_hdf(snap_num=Snapshot, param='IntraClusterStars') * 1.0e10 / Hubble_h
        MetalsICS = read_hdf(snap_num=Snapshot, param='MetalsIntraClusterStars') * 1.0e10 / Hubble_h
        Type = read_hdf(snap_num=Snapshot, param='Type')

        # Select central galaxies with valid masses and metals
        w = np.where((Type == 0) & (StellarMass > 0) & (ICS > 0) &
                     (MetalsStellarMass > 0) & (MetalsICS > 0))[0]

        if len(w) == 0:
            ax.text(0.5, 0.5, f'z = {z:.2f}\nNo data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14)
            ax.set_xlim(7, 10)
            ax.set_ylim(7, 10)
            continue

        # Convert to 12 + log10(O/H) scale
        Z_stellar = np.log10((MetalsStellarMass[w] / StellarMass[w]) / 0.02) + 9.0
        Z_ICS = np.log10((MetalsICS[w] / ICS[w]) / 0.02) + 9.0

        # Define bins for contour plot
        x_bins = np.linspace(7, 10, 50)
        y_bins = np.linspace(7, 10, 50)

        # Create 2D histogram for contour
        H, xedges, yedges = np.histogram2d(Z_stellar, Z_ICS, bins=[x_bins, y_bins])

        # Transpose for correct orientation
        H = H.T

        # Get bin centers
        x_centers = (xedges[:-1] + xedges[1:]) / 2
        y_centers = (yedges[:-1] + yedges[1:]) / 2

        # Create meshgrid for contour
        X, Y = np.meshgrid(x_centers, y_centers)

        # Plot filled contours
        levels = np.linspace(1, H.max(), 10)
        if H.max() > 1:
            contour = ax.contourf(X, Y, H, levels=levels, cmap='plasma', extend='max')

        # Calculate median in bins
        bin_width = 0.2
        bin_edges = np.arange(7, 10.5, bin_width)
        bin_centers = bin_edges[:-1] + bin_width / 2

        median_values = []
        valid_bins = []

        for i in range(len(bin_edges) - 1):
            mask = (Z_stellar >= bin_edges[i]) & (Z_stellar < bin_edges[i + 1])
            if np.sum(mask) > 10:
                median_values.append(np.median(Z_ICS[mask]))
                valid_bins.append(bin_centers[i])

        # Plot median line
        if len(valid_bins) > 0:
            valid_bins = np.array(valid_bins)
            median_values = np.array(median_values)
            ax.plot(valid_bins, median_values, 'k-', lw=2)

        # Plot 1:1 line
        ax.plot([7, 10], [7, 10], 'k--', lw=1, alpha=0.5)

        # Formatting
        ax.set_xlim(7, 10)
        ax.set_ylim(7, 10)
        ax.set_xticks([7, 8, 9, 10])
        ax.set_yticks([7, 8, 9, 10])

        # Only show y-axis labels for left column
        if idx % 2 == 1:
            ax.set_yticklabels([])

        # Add redshift label
        ax.text(0.05, 0.95, f'z = {z:.2f}', transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top', ha='left')

    # Add shared axis labels
    fig.supxlabel(r'$12 + \log_{10}[\mathrm{O/H}]_{\star}$', fontsize=14, y=0.02)
    fig.supylabel(r'$12 + \log_{10}[\mathrm{O/H}]_{\mathrm{ICS}}$', fontsize=14, x=0.005)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.06, left=0.08)

    outputFile = OutputDir + 'ICS_metallicity_vs_stellar_metallicity_grid' + OutputFormat
    plt.savefig(outputFile, dpi=300, bbox_inches='tight')
    print(f'\nSaved file to {outputFile}')
    plt.close()

    print('\nICS metallicity vs Stellar metallicity plot completed!')

    # ==================================================================
    # ICS to Stellar Mass Ratio vs Mvir
    # ==================================================================

    print('\nRunning ICS/Stellar Mass Ratio vs Mvir Grid Plot\n')

    # Create 2x4 grid of subplots
    fig, axes = plt.subplots(4, 2, figsize=(10, 16))
    axes = axes.flatten()

    for idx, snap in enumerate(ICS_snaps):
        ax = axes[idx]
        Snapshot = 'Snap_' + str(snap)
        z = redshifts[snap]

        print(f'Processing snapshot {snap} (z = {z:.2f})')

        # Read data
        Mvir = read_hdf(snap_num=Snapshot, param='Mvir') * 1.0e10 / Hubble_h
        StellarMass = read_hdf(snap_num=Snapshot, param='StellarMass') * 1.0e10 / Hubble_h
        ICS = read_hdf(snap_num=Snapshot, param='IntraClusterStars') * 1.0e10 / Hubble_h
        Type = read_hdf(snap_num=Snapshot, param='Type')

        # Select central galaxies with valid masses
        w = np.where((Type == 0) & (Mvir > 0) & (StellarMass > 0) & (ICS > 0))[0]

        if len(w) == 0:
            ax.text(0.5, 0.5, f'z = {z:.2f}\nNo data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14)
            ax.set_xlim(11, 15)
            ax.set_ylim(0, 2)
            continue

        log_Mvir = np.log10(Mvir[w])
        ICS_stellar_ratio = ICS[w] / StellarMass[w]

        # Define bins for contour plot
        x_bins = np.linspace(11, 15, 50)
        y_bins = np.linspace(0, 2, 50)

        # Create 2D histogram for contour
        H, xedges, yedges = np.histogram2d(log_Mvir, ICS_stellar_ratio, bins=[x_bins, y_bins])

        # Transpose for correct orientation
        H = H.T

        # Get bin centers
        x_centers = (xedges[:-1] + xedges[1:]) / 2
        y_centers = (yedges[:-1] + yedges[1:]) / 2

        # Create meshgrid for contour
        X, Y = np.meshgrid(x_centers, y_centers)

        # Plot filled contours
        levels = np.linspace(1, H.max(), 10)
        if H.max() > 1:
            contour = ax.contourf(X, Y, H, levels=levels, cmap='plasma', extend='max')

        # Calculate median in bins
        bin_width = 0.25
        bin_edges = np.arange(11, 15.5, bin_width)
        bin_centers = bin_edges[:-1] + bin_width / 2

        median_values = []
        valid_bins = []

        for i in range(len(bin_edges) - 1):
            mask = (log_Mvir >= bin_edges[i]) & (log_Mvir < bin_edges[i + 1])
            if np.sum(mask) > 10:
                median_values.append(np.median(ICS_stellar_ratio[mask]))
                valid_bins.append(bin_centers[i])

        # Plot median line
        if len(valid_bins) > 0:
            valid_bins = np.array(valid_bins)
            median_values = np.array(median_values)
            ax.plot(valid_bins, median_values, 'k-', lw=2)

        # Formatting
        ax.set_xlim(11, 15)
        ax.set_ylim(0, 2)
        ax.set_xticks([11, 12, 13, 14, 15])
        ax.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])

        # Only show y-axis labels for left column
        if idx % 2 == 1:
            ax.set_yticklabels([])

        # Add redshift label
        ax.text(0.05, 0.95, f'z = {z:.2f}', transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top', ha='left')

    # Add shared axis labels
    fig.supxlabel(r'$\log_{10} M_{\mathrm{vir}}\ [M_{\odot}]$', fontsize=14, y=0.02)
    fig.supylabel(r'$M_{\mathrm{ICS}} / M_{\star}$', fontsize=14, x=0.02)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.06, left=0.08)

    outputFile = OutputDir + 'ICS_stellar_ratio_vs_Mvir_grid' + OutputFormat
    plt.savefig(outputFile, dpi=300, bbox_inches='tight')
    print(f'\nSaved file to {outputFile}')
    plt.close()

    print('\nICS/Stellar Mass ratio vs Mvir plot completed!')

    # ==================================================================
    # ICS to Stellar Mass Ratio vs Stellar Mass
    # ==================================================================

    print('\nRunning ICS/Stellar Mass Ratio vs Stellar Mass Grid Plot\n')

    # Create 2x4 grid of subplots
    fig, axes = plt.subplots(4, 2, figsize=(10, 16))
    axes = axes.flatten()

    for idx, snap in enumerate(ICS_snaps):
        ax = axes[idx]
        Snapshot = 'Snap_' + str(snap)
        z = redshifts[snap]

        print(f'Processing snapshot {snap} (z = {z:.2f})')

        # Read data
        StellarMass = read_hdf(snap_num=Snapshot, param='StellarMass') * 1.0e10 / Hubble_h
        ICS = read_hdf(snap_num=Snapshot, param='IntraClusterStars') * 1.0e10 / Hubble_h
        Type = read_hdf(snap_num=Snapshot, param='Type')

        # Select central galaxies with valid masses
        w = np.where((Type == 0) & (StellarMass > 0) & (ICS > 0))[0]

        if len(w) == 0:
            ax.text(0.5, 0.5, f'z = {z:.2f}\nNo data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14)
            ax.set_xlim(8, 12)
            ax.set_ylim(0, 2)
            continue

        log_StellarMass = np.log10(StellarMass[w])
        ICS_stellar_ratio = ICS[w] / StellarMass[w]

        # Define bins for contour plot
        x_bins = np.linspace(8, 12, 50)
        y_bins = np.linspace(0, 2, 50)

        # Create 2D histogram for contour
        H, xedges, yedges = np.histogram2d(log_StellarMass, ICS_stellar_ratio, bins=[x_bins, y_bins])

        # Transpose for correct orientation
        H = H.T

        # Get bin centers
        x_centers = (xedges[:-1] + xedges[1:]) / 2
        y_centers = (yedges[:-1] + yedges[1:]) / 2

        # Create meshgrid for contour
        X, Y = np.meshgrid(x_centers, y_centers)

        # Plot filled contours
        levels = np.linspace(1, H.max(), 10)
        if H.max() > 1:
            contour = ax.contourf(X, Y, H, levels=levels, cmap='plasma', extend='max')

        # Calculate median in bins
        bin_width = 0.25
        bin_edges = np.arange(8, 12.5, bin_width)
        bin_centers = bin_edges[:-1] + bin_width / 2

        median_values = []
        valid_bins = []

        for i in range(len(bin_edges) - 1):
            mask = (log_StellarMass >= bin_edges[i]) & (log_StellarMass < bin_edges[i + 1])
            if np.sum(mask) > 10:
                median_values.append(np.median(ICS_stellar_ratio[mask]))
                valid_bins.append(bin_centers[i])

        # Plot median line
        if len(valid_bins) > 0:
            valid_bins = np.array(valid_bins)
            median_values = np.array(median_values)
            ax.plot(valid_bins, median_values, 'k-', lw=2)

        # Formatting
        ax.set_xlim(8, 12)
        ax.set_ylim(0, 2)
        ax.set_xticks([8, 9, 10, 11, 12])
        ax.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])

        # Only show y-axis labels for left column
        if idx % 2 == 1:
            ax.set_yticklabels([])

        # Add redshift label
        ax.text(0.05, 0.95, f'z = {z:.2f}', transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top', ha='left')

    # Add shared axis labels
    fig.supxlabel(r'$\log_{10} M_{\star}\ [M_{\odot}]$', fontsize=14, y=0.02)
    fig.supylabel(r'$M_{\mathrm{ICS}} / M_{\star}$', fontsize=14, x=0.02)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.06, left=0.08)

    outputFile = OutputDir + 'ICS_stellar_ratio_vs_StellarMass_grid' + OutputFormat
    plt.savefig(outputFile, dpi=300, bbox_inches='tight')
    print(f'\nSaved file to {outputFile}')
    plt.close()

    print('\nICS/Stellar Mass ratio vs Stellar Mass plot completed!')

    # ==================================================================
    # ICS Formation Channel Analysis
    # ==================================================================

    print('\nRunning ICS Formation Channel Analysis\n')

    # Track contributions by mergeType across snapshots
    z_for_plot = []
    disruption_mass = []  # mergeType == 4
    minor_merger_mass = []  # mergeType == 1
    major_merger_mass = []  # mergeType == 2

    # Use more snapshots for this analysis
    analysis_snaps = list(range(63, 19, -1))  # z=0 to z~5

    for snap in analysis_snaps:
        Snapshot = 'Snap_' + str(snap)
        z = redshifts[snap]
        z_for_plot.append(z)

        print(f'Processing snapshot {snap} (z = {z:.2f})')

        # Read data
        StellarMass = read_hdf(snap_num=Snapshot, param='StellarMass') * 1.0e10 / Hubble_h
        mergeType = read_hdf(snap_num=Snapshot, param='mergeType')

        # Sum stellar mass by mergeType (this is the mass that was processed)
        w_disruption = np.where(mergeType == 4)[0]
        w_minor = np.where(mergeType == 1)[0]
        w_major = np.where(mergeType == 2)[0]

        disruption_mass.append(np.sum(StellarMass[w_disruption]) if len(w_disruption) > 0 else 0)
        minor_merger_mass.append(np.sum(StellarMass[w_minor]) if len(w_minor) > 0 else 0)
        major_merger_mass.append(np.sum(StellarMass[w_major]) if len(w_major) > 0 else 0)

    # Convert to arrays
    z_for_plot = np.array(z_for_plot)
    disruption_mass = np.array(disruption_mass)
    minor_merger_mass = np.array(minor_merger_mass)
    major_merger_mass = np.array(major_merger_mass)

    # Plot 1: Cumulative mass processed by each channel
    fig, ax = plt.subplots(figsize=(8, 6))

    # Cumulative sum (going from high z to low z, so reverse)
    cum_disruption = np.cumsum(disruption_mass[::-1])[::-1]
    cum_minor = np.cumsum(minor_merger_mass[::-1])[::-1]
    cum_major = np.cumsum(major_merger_mass[::-1])[::-1]

    ax.plot(z_for_plot, np.log10(cum_disruption + 1), 'r-', lw=2, label='Disruption to ICS')
    ax.plot(z_for_plot, np.log10(cum_minor + 1), 'b-', lw=2, label='Minor Mergers')
    ax.plot(z_for_plot, np.log10(cum_major + 1), 'g-', lw=2, label='Major Mergers')

    ax.set_xlabel(r'$z$', fontsize=14)
    ax.set_ylabel(r'$\log_{10}$ Cumulative Stellar Mass Processed $[M_{\odot}]$', fontsize=14)
    ax.set_xlim(5, 0)  # Flipped: high z on left, z=0 on right
    ax.legend(loc='upper left', fontsize=10, frameon=False)

    plt.tight_layout()
    outputFile = OutputDir + 'ICS_formation_channels_cumulative' + OutputFormat
    plt.savefig(outputFile, dpi=300, bbox_inches='tight')
    print(f'\nSaved file to {outputFile}')
    plt.close()

    # Plot 2: Count disruptions and mergers across ALL snapshots by halo mass
    print('\nPlotting mergeType distribution vs halo mass (all snapshots)')

    fig, ax = plt.subplots(figsize=(8, 6))

    # Accumulate counts across all snapshots
    bin_edges = np.arange(11, 15.5, 0.5)
    bin_centers = bin_edges[:-1] + 0.25
    total_disruptions = np.zeros(len(bin_centers))
    total_mergers = np.zeros(len(bin_centers))

    for snap in analysis_snaps:
        Snapshot = 'Snap_' + str(snap)
        Mvir = read_hdf(snap_num=Snapshot, param='Mvir') * 1.0e10 / Hubble_h
        mergeType = read_hdf(snap_num=Snapshot, param='mergeType')

        log_Mvir = np.log10(np.maximum(Mvir, 1e-10))

        for i in range(len(bin_edges) - 1):
            mask = (log_Mvir >= bin_edges[i]) & (log_Mvir < bin_edges[i + 1])
            total_disruptions[i] += np.sum(mergeType[mask] == 4)
            total_mergers[i] += np.sum((mergeType[mask] == 1) | (mergeType[mask] == 2))

    # Create bar plot
    width = 0.2
    ax.bar(bin_centers - width/2, total_disruptions, width, label='Disruptions (→ICS)', color='red', alpha=0.7)
    ax.bar(bin_centers + width/2, total_mergers, width, label='Mergers', color='blue', alpha=0.7)

    ax.set_xlabel(r'$\log_{10} M_{\mathrm{vir}}\ [M_{\odot}]$', fontsize=14)
    ax.set_ylabel('Total Count (all snapshots)', fontsize=14)
    ax.legend(loc='upper right', fontsize=10, frameon=False)
    ax.set_xlim(11, 15)
    ax.set_yscale('log')

    plt.tight_layout()
    outputFile = OutputDir + 'ICS_formation_mergeType_distribution' + OutputFormat
    plt.savefig(outputFile, dpi=300, bbox_inches='tight')
    print(f'\nSaved file to {outputFile}')
    plt.close()

    # Plot 3: ICS fraction vs Mvir colored by number of disruptions received
    print('\nPlotting ICS fraction colored by disruption count')

    fig, ax = plt.subplots(figsize=(10, 8))
    Snapshot = 'Snap_63'

    Mvir = read_hdf(snap_num=Snapshot, param='Mvir') * 1.0e10 / Hubble_h
    ICS = read_hdf(snap_num=Snapshot, param='IntraClusterStars') * 1.0e10 / Hubble_h
    Type = read_hdf(snap_num=Snapshot, param='Type')
    CentralGalaxyIndex = read_hdf(snap_num=Snapshot, param='CentralGalaxyIndex')
    mergeType = read_hdf(snap_num=Snapshot, param='mergeType')
    GalaxyIndex = read_hdf(snap_num=Snapshot, param='GalaxyIndex')

    # Count disruptions per central across all snapshots
    # Build a dictionary of central galaxy index -> disruption count
    disruption_count = {}

    for snap in analysis_snaps:
        Snapshot_loop = 'Snap_' + str(snap)
        mergeType_snap = read_hdf(snap_num=Snapshot_loop, param='mergeType')
        CentralIdx_snap = read_hdf(snap_num=Snapshot_loop, param='CentralGalaxyIndex')

        # Find disrupted galaxies and their centrals
        disrupted = np.where(mergeType_snap == 4)[0]
        for idx in disrupted:
            central_idx = CentralIdx_snap[idx]
            if central_idx not in disruption_count:
                disruption_count[central_idx] = 0
            disruption_count[central_idx] += 1

    # Now plot for z=0 centrals
    w_central = np.where((Type == 0) & (Mvir > 0) & (ICS > 0))[0]
    log_Mvir_central = np.log10(Mvir[w_central])
    ICS_frac_central = ICS[w_central] / (BaryonFrac * Mvir[w_central])

    # Get disruption counts for each central
    n_disruptions = np.array([disruption_count.get(GalaxyIndex[i], 0) for i in w_central])

    # Create scatter plot colored by disruption count
    scatter = ax.scatter(log_Mvir_central, ICS_frac_central, c=n_disruptions,
                         cmap='plasma', s=10, alpha=0.7, vmin=0, vmax=np.percentile(n_disruptions, 95))
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Number of Disruptions', fontsize=12)

    # Add median line
    bin_width = 0.25
    bin_edges_med = np.arange(11, 15.5, bin_width)
    bin_centers_med = bin_edges_med[:-1] + bin_width / 2
    median_values = []
    valid_bins = []

    for i in range(len(bin_edges_med) - 1):
        mask = (log_Mvir_central >= bin_edges_med[i]) & (log_Mvir_central < bin_edges_med[i + 1])
        if np.sum(mask) > 10:
            median_values.append(np.median(ICS_frac_central[mask]))
            valid_bins.append(bin_centers_med[i])

    if len(valid_bins) > 0:
        ax.plot(valid_bins, median_values, 'k-', lw=2, label='Median')
        ax.legend(loc='upper left', fontsize=10, frameon=False)

    ax.set_xlabel(r'$\log_{10} M_{\mathrm{vir}}\ [M_{\odot}]$', fontsize=14)
    ax.set_ylabel(r'$f_{\mathrm{ICS}} = M_{\mathrm{ICS}} / (f_b \times M_{\mathrm{vir}})$', fontsize=14)
    ax.set_xlim(11, 15)
    ax.set_ylim(0, 0.15)

    plt.tight_layout()
    outputFile = OutputDir + 'ICS_fraction_by_disruption_count' + OutputFormat
    plt.savefig(outputFile, dpi=300, bbox_inches='tight')
    print(f'\nSaved file to {outputFile}')
    plt.close()

    print('\nICS formation channel analysis completed!')

#!/usr/bin/env python
"""
FFB vs No-FFB Comparison: 2x2 Grid Plot
Compares galaxy properties between FFB and non-FFB models across redshifts z=14 to z=5.

Plots (2x2 grid):
Row 1: SFR vs. stellar mass (medians) | sSFR vs. stellar mass (medians)
Row 2: Metallicity vs. stellar mass (medians) | Quiescent fraction vs. stellar mass

Both FFB and no-FFB (Mvir-matched) galaxies are overlaid on each panel.
Redshifts from z=14 to z=5 colored using plasma colormap.
FFB: solid lines, No-FFB: dashed lines.
"""

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import os
from random import sample, seed
import warnings
warnings.filterwarnings("ignore")

# ========================== USER OPTIONS ==========================

DirName_FFB = './output/millennium/'
DirName_noFFB = './output/millennium_noffb/'
FileName = 'model_0.hdf5'

# Snapshots from z~14 to z~5
Snapshots = ['Snap_8', 'Snap_9', 'Snap_10', 'Snap_11', 'Snap_12', 'Snap_13',
             'Snap_14', 'Snap_15', 'Snap_16', 'Snap_17', 'Snap_18', 'Snap_19', 'Snap_20']

# Simulation details
Hubble_h = 0.73
BoxSize = 62.5
VolumeFraction = 1.0

# Plotting options
dilute = 7500
sSFRcut = -11.0  # sSFR threshold for quiescent galaxies (log10(sSFR) < -11)

OutputFormat = '.pdf'
plt.rcParams["figure.figsize"] = (12, 10)
plt.rcParams["figure.dpi"] = 96
plt.rcParams["font.size"] = 12

# Redshift list for snapshots
redshifts = [127.000, 79.998, 50.000, 30.000, 19.916, 18.244, 16.725, 15.343, 14.086, 12.941, 11.897, 10.944, 10.073,
             9.278, 8.550, 7.883, 7.272, 6.712, 6.197, 5.724, 5.289, 4.888, 4.520, 4.179, 3.866, 3.576, 3.308, 3.060,
             2.831, 2.619, 2.422, 2.239, 2.070, 1.913, 1.766, 1.630, 1.504, 1.386, 1.276, 1.173, 1.078, 0.989, 0.905,
             0.828, 0.755, 0.687, 0.624, 0.564, 0.509, 0.457, 0.408, 0.362, 0.320, 0.280, 0.242, 0.208, 0.175, 0.144,
             0.116, 0.089, 0.064, 0.041, 0.020, 0.000]

# ==================================================================

def read_hdf(file_name=None, snap_num=None, param=None, DirName=None):
    """Read parameter from HDF5 file."""
    with h5.File(DirName + file_name, 'r') as f:
        return np.array(f[snap_num][param])

def load_data(DirName, Snapshot, file_name=FileName):
    """Load all relevant galaxy properties for a snapshot."""
    return {
        'Mvir': read_hdf(snap_num=Snapshot, param='Mvir', DirName=DirName, file_name=file_name) * 1.0e10 / Hubble_h,
        'StellarMass': read_hdf(snap_num=Snapshot, param='StellarMass', DirName=DirName, file_name=file_name) * 1.0e10 / Hubble_h,
        'ColdGas': read_hdf(snap_num=Snapshot, param='ColdGas', DirName=DirName, file_name=file_name) * 1.0e10 / Hubble_h,
        'MetalsColdGas': read_hdf(snap_num=Snapshot, param='MetalsColdGas', DirName=DirName, file_name=file_name) * 1.0e10 / Hubble_h,
        'SfrDisk': read_hdf(snap_num=Snapshot, param='SfrDisk', DirName=DirName, file_name=file_name),
        'SfrBulge': read_hdf(snap_num=Snapshot, param='SfrBulge', DirName=DirName, file_name=file_name),
        'Type': read_hdf(snap_num=Snapshot, param='Type', DirName=DirName, file_name=file_name),
        'FFBRegime': read_hdf(snap_num=Snapshot, param='FFBRegime', DirName=DirName, file_name=file_name),
        'BulgeMass': read_hdf(snap_num=Snapshot, param='BulgeMass', DirName=DirName, file_name=file_name) * 1.0e10 / Hubble_h,
        'DiskMass': read_hdf(snap_num=Snapshot, param='StellarMass', DirName=DirName, file_name=file_name) * 1.0e10 / Hubble_h - read_hdf(snap_num=Snapshot, param='BulgeMass', DirName=DirName, file_name=file_name) * 1.0e10 / Hubble_h,
        'DiskRadius': read_hdf(snap_num=Snapshot, param='DiskRadius', DirName=DirName, file_name=file_name) / Hubble_h * 1e3,  # kpc
        'BulgeRadius': read_hdf(snap_num=Snapshot, param='BulgeRadius', DirName=DirName, file_name=file_name) / Hubble_h * 1e3,  # kpc
    }

def match_by_mvir(mvir_ffb, mvir_noffb):
    """
    Match FFB galaxies to no-FFB galaxies by closest Mvir (without replacement).
    Returns indices into the no-FFB catalogue.
    """
    used = set()
    matched_indices = []
    for m in mvir_ffb:
        diffs = np.abs(mvir_noffb - m)
        for i in used:
            diffs[i] = np.inf
        min_idx = np.argmin(diffs)
        matched_indices.append(min_idx)
        used.add(min_idx)
    return np.array(matched_indices)

def compute_medians(x, y, bins, min_count=3):
    """
    Compute median of y in bins of x.
    Returns bin centers and medians (NaN where insufficient data).
    """
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    medians = np.full(len(bin_centers), np.nan)

    for i in range(len(bins) - 1):
        mask = (x >= bins[i]) & (x < bins[i+1])
        if np.sum(mask) >= min_count:
            medians[i] = np.median(y[mask])

    return bin_centers, medians

def compute_quiescent_fraction(stellar_mass, sfr, bins, ssfr_cut=-11.0, min_count=3):
    """
    Compute quiescent fraction in bins of stellar mass.
    Quiescent: log10(sSFR) < ssfr_cut
    """
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    fractions = np.full(len(bin_centers), np.nan)

    # Compute sSFR
    ssfr = np.log10(sfr / stellar_mass + 1e-15)

    for i in range(len(bins) - 1):
        mask = (np.log10(stellar_mass) >= bins[i]) & (np.log10(stellar_mass) < bins[i+1])
        count = np.sum(mask)
        if count >= min_count:
            quiescent = np.sum(ssfr[mask] < ssfr_cut)
            fractions[i] = quiescent / count

    return bin_centers, fractions

def compute_quiescent_fraction_mvir(mvir, stellar_mass, sfr, bins, ssfr_cut=-11.0, min_count=3):
    """
    Compute quiescent fraction in bins of Mvir.
    Quiescent: sSFR < 10^ssfr_cut (linear comparison)
    """
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    fractions = np.full(len(bin_centers), np.nan)

    # Compute sSFR (linear, not log)
    ssfr = sfr / stellar_mass

    for i in range(len(bins) - 1):
        mask = (np.log10(mvir) >= bins[i]) & (np.log10(mvir) < bins[i+1])
        count = np.sum(mask)
        if count >= min_count:
            quiescent = np.sum(ssfr[mask] < 10.0**ssfr_cut)
            fractions[i] = quiescent / count

    return bin_centers, fractions

def get_snapshot_redshift(snapshot):
    """Get redshift for a given snapshot string."""
    snapnum = int(snapshot.split('_')[1])
    return redshifts[snapnum]

def plot_ffb_comparison_grid():
    """Create 2x2 grid comparing FFB vs no-FFB galaxy properties."""

    seed(2222)

    OutputDir = DirName_FFB + 'plots/'
    if not os.path.exists(OutputDir):
        os.makedirs(OutputDir)

    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Color map: plasma from z=5 (dark) to z=14 (bright)
    cmap = cm.plasma
    z_min, z_max = 5.0, 14.5

    # Mvir bins for computing medians (log10 Mvir in Msun)
    mvir_bins = np.arange(9.5, 12.5, 0.3)

    # Collect data for all snapshots
    print("Loading data from all snapshots...")

    # Data storage for each snapshot
    all_data = []

    for Snapshot in Snapshots:
        snapnum = int(Snapshot.split('_')[1])
        z = redshifts[snapnum]
        print(f'Processing {Snapshot} (z = {z:.2f})')

        # Load data
        data_FFB = load_data(DirName_FFB, Snapshot)
        data_noFFB = load_data(DirName_noFFB, Snapshot)

        # Identify FFB galaxies
        ffb_mask = data_FFB['FFBRegime'] == 1
        n_ffb = np.sum(ffb_mask)

        if n_ffb == 0:
            print(f"  No FFB galaxies at {Snapshot}, skipping.")
            continue

        # Get Mvir of FFB galaxies and match to no-FFB catalogue
        mvir_ffb = data_FFB['Mvir'][ffb_mask]
        matched_indices = match_by_mvir(mvir_ffb, data_noFFB['Mvir'])

        print(f"  Matched {n_ffb} FFB galaxies to no-FFB catalogue by Mvir")

        # Extract properties for FFB galaxies
        stellar_mass_ffb = data_FFB['StellarMass'][ffb_mask]
        coldgas_ffb = data_FFB['ColdGas'][ffb_mask]
        metals_coldgas_ffb = data_FFB['MetalsColdGas'][ffb_mask]
        sfr_ffb = data_FFB['SfrDisk'][ffb_mask] + data_FFB['SfrBulge'][ffb_mask]
        type_ffb = data_FFB['Type'][ffb_mask]

        # Extract properties for Mvir-matched no-FFB galaxies
        stellar_mass_noffb = data_noFFB['StellarMass'][matched_indices]
        coldgas_noffb = data_noFFB['ColdGas'][matched_indices]
        metals_coldgas_noffb = data_noFFB['MetalsColdGas'][matched_indices]
        sfr_noffb = data_noFFB['SfrDisk'][matched_indices] + data_noFFB['SfrBulge'][matched_indices]
        type_noffb = data_noFFB['Type'][matched_indices]

        # Store for plotting
        all_data.append({
            'z': z,
            'snapshot': Snapshot,
            'ffb': {
                'stellar_mass': stellar_mass_ffb,
                'coldgas': coldgas_ffb,
                'metals_coldgas': metals_coldgas_ffb,
                'sfr': sfr_ffb,
                'type': type_ffb,
                'mvir': mvir_ffb
            },
            'noffb': {
                'stellar_mass': stellar_mass_noffb,
                'coldgas': coldgas_noffb,
                'metals_coldgas': metals_coldgas_noffb,
                'sfr': sfr_noffb,
                'type': type_noffb,
                'mvir': data_noFFB['Mvir'][matched_indices]
            }
        })

    # Now plot all data
    print("\nGenerating plots...")

    for data in all_data:
        z = data['z']
        color = cmap((z - z_min) / (z_max - z_min))

        # Apply selection: stellar mass cut only (all galaxy types)
        # Note: Gas fraction cut removed because FFB galaxies are gas-poor by nature
        mask_ffb = data['ffb']['stellar_mass'] > 1.0e8
        mask_noffb = data['noffb']['stellar_mass'] > 1.0e8

        if np.sum(mask_ffb) < 2 or np.sum(mask_noffb) < 2:
            continue

        # Extract masked data
        sm_ffb = data['ffb']['stellar_mass'][mask_ffb]
        sfr_ffb = data['ffb']['sfr'][mask_ffb]
        cg_ffb = data['ffb']['coldgas'][mask_ffb]
        mcg_ffb = data['ffb']['metals_coldgas'][mask_ffb]
        mvir_ffb = data['ffb']['mvir'][mask_ffb]

        sm_noffb = data['noffb']['stellar_mass'][mask_noffb]
        sfr_noffb = data['noffb']['sfr'][mask_noffb]
        cg_noffb = data['noffb']['coldgas'][mask_noffb]
        mcg_noffb = data['noffb']['metals_coldgas'][mask_noffb]
        mvir_noffb = data['noffb']['mvir'][mask_noffb]

        log_mvir_ffb = np.log10(mvir_ffb)
        log_mvir_noffb = np.log10(mvir_noffb)

        # ----- Panel (0,0): SFR vs Mvir -----
        log_sfr_ffb = np.log10(sfr_ffb + 1e-10)
        log_sfr_noffb = np.log10(sfr_noffb + 1e-10)

        bc_ffb, med_ffb = compute_medians(log_mvir_ffb, log_sfr_ffb, mvir_bins)
        bc_noffb, med_noffb = compute_medians(log_mvir_noffb, log_sfr_noffb, mvir_bins)

        valid_ffb = ~np.isnan(med_ffb)
        valid_noffb = ~np.isnan(med_noffb)

        if np.sum(valid_ffb) > 1:
            axes[0, 0].plot(bc_ffb[valid_ffb], med_ffb[valid_ffb], '-', color=color, linewidth=1.5)
        if np.sum(valid_noffb) > 1:
            axes[0, 0].plot(bc_noffb[valid_noffb], med_noffb[valid_noffb], '--', color=color, linewidth=1.5)

        # ----- Panel (0,1): sSFR vs Mvir -----
        log_ssfr_ffb = np.log10(sfr_ffb / sm_ffb + 1e-15)
        log_ssfr_noffb = np.log10(sfr_noffb / sm_noffb + 1e-15)

        bc_ffb, med_ffb = compute_medians(log_mvir_ffb, log_ssfr_ffb, mvir_bins)
        bc_noffb, med_noffb = compute_medians(log_mvir_noffb, log_ssfr_noffb, mvir_bins)

        valid_ffb = ~np.isnan(med_ffb)
        valid_noffb = ~np.isnan(med_noffb)

        if np.sum(valid_ffb) > 1:
            axes[0, 1].plot(bc_ffb[valid_ffb], med_ffb[valid_ffb], '-', color=color, linewidth=1.5)
        if np.sum(valid_noffb) > 1:
            axes[0, 1].plot(bc_noffb[valid_noffb], med_noffb[valid_noffb], '--', color=color, linewidth=1.5)

        # ----- Panel (1,0): Metallicity vs Mvir -----
        # Only require non-zero gas to compute metallicity
        valid_gas_ffb = (cg_ffb > 0) & (mcg_ffb > 0)
        valid_gas_noffb = (cg_noffb > 0) & (mcg_noffb > 0)

        if np.sum(valid_gas_ffb) >= 3:
            Z_ffb = np.log10((mcg_ffb[valid_gas_ffb] / cg_ffb[valid_gas_ffb]) / 0.02) + 9.0
            bc_ffb, med_ffb = compute_medians(log_mvir_ffb[valid_gas_ffb], Z_ffb, mvir_bins)
            valid = ~np.isnan(med_ffb)
            if np.sum(valid) > 1:
                axes[1, 0].plot(bc_ffb[valid], med_ffb[valid], '-', color=color, linewidth=1.5)

        if np.sum(valid_gas_noffb) >= 3:
            Z_noffb = np.log10((mcg_noffb[valid_gas_noffb] / cg_noffb[valid_gas_noffb]) / 0.02) + 9.0
            bc_noffb, med_noffb = compute_medians(log_mvir_noffb[valid_gas_noffb], Z_noffb, mvir_bins)
            valid = ~np.isnan(med_noffb)
            if np.sum(valid) > 1:
                axes[1, 0].plot(bc_noffb[valid], med_noffb[valid], '--', color=color, linewidth=1.5)

        # ----- Panel (1,1): Quiescent fraction vs Mvir -----
        bc_ffb, fq_ffb = compute_quiescent_fraction_mvir(mvir_ffb, sm_ffb, sfr_ffb, mvir_bins, ssfr_cut=sSFRcut)
        bc_noffb, fq_noffb = compute_quiescent_fraction_mvir(mvir_noffb, sm_noffb, sfr_noffb, mvir_bins, ssfr_cut=sSFRcut)

        valid_ffb = ~np.isnan(fq_ffb)
        valid_noffb = ~np.isnan(fq_noffb)

        if np.sum(valid_ffb) > 1:
            axes[1, 1].plot(bc_ffb[valid_ffb], fq_ffb[valid_ffb], '-', color=color, linewidth=1.5)
        if np.sum(valid_noffb) > 1:
            axes[1, 1].plot(bc_noffb[valid_noffb], fq_noffb[valid_noffb], '--', color=color, linewidth=1.5)

    # Configure axes
    # Panel (0,0): SFR
    axes[0, 0].set_xlabel(r'$\log_{10}(M_{\mathrm{vir}}\ [M_\odot])$')
    axes[0, 0].set_ylabel(r'$\log_{10}(\mathrm{SFR}\ [M_\odot/\mathrm{yr}])$')
    axes[0, 0].set_xlim(10, 12.5)
    axes[0, 0].set_ylim(-1, 3)
    axes[0, 0].set_title('SFR vs. Halo Mass')

    # Panel (0,1): sSFR
    axes[0, 1].set_xlabel(r'$\log_{10}(M_{\mathrm{vir}}\ [M_\odot])$')
    axes[0, 1].set_ylabel(r'$\log_{10}(\mathrm{sSFR}\ [\mathrm{yr}^{-1}])$')
    axes[0, 1].set_xlim(10, 12.5)
    axes[0, 1].set_ylim(-9, -7)
    axes[0, 1].axhline(y=sSFRcut, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    axes[0, 1].set_title('sSFR vs. Halo Mass')

    # Panel (1,0): Metallicity
    axes[1, 0].set_xlabel(r'$\log_{10}(M_{\mathrm{vir}}\ [M_\odot])$')
    axes[1, 0].set_ylabel(r'$12 + \log_{10}(\mathrm{O/H})$')
    axes[1, 0].set_xlim(10, 12.5)
    axes[1, 0].set_ylim(7.5, 9.5)
    axes[1, 0].set_title('Metallicity vs. Halo Mass')

    # Panel (1,1): Quiescent fraction
    axes[1, 1].set_xlabel(r'$\log_{10}(M_{\mathrm{vir}}\ [M_\odot])$')
    axes[1, 1].set_ylabel(r'$f_{\mathrm{quiescent}}$')
    axes[1, 1].set_xlim(10, 12.5)
    axes[1, 1].set_ylim(0.0, 0.1)
    axes[1, 1].set_title('Quiescent Fraction vs. Halo Mass')

    # Add legend for line styles
    legend_elements = [
        Line2D([0], [0], color='black', linestyle='-', linewidth=1.5, label='FFB'),
        Line2D([0], [0], color='black', linestyle='--', linewidth=1.5, label='No-FFB (Mvir-matched)')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.98), fontsize=12)

    plt.tight_layout(rect=[0, 0.02, 0.88, 0.95])

    # Add colorbar outside the plots on the right
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=z_min, vmax=z_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Redshift', fontsize=14)

    output_file = OutputDir + 'ffb_comparison_2x2_grid' + OutputFormat
    plt.savefig(output_file, bbox_inches='tight')
    print(f"\nSaved: {output_file}")
    plt.close()

def plot_ffb_comparison_grid_shmr():
    """Create 2x2 grid comparing FFB vs no-FFB galaxy properties, with SHMR instead of quiescent fraction."""

    seed(2222)

    OutputDir = DirName_FFB + 'plots/'
    if not os.path.exists(OutputDir):
        os.makedirs(OutputDir)

    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Color map: plasma from z=5 (dark) to z=14 (bright)
    cmap = cm.plasma
    z_min, z_max = 5.0, 14.5

    # Mvir bins for computing medians (log10 Mvir in Msun)
    mvir_bins = np.arange(9.5, 12.5, 0.3)

    # Collect data for all snapshots
    print("Loading data from all snapshots...")

    # Data storage for each snapshot
    all_data = []

    for Snapshot in Snapshots:
        snapnum = int(Snapshot.split('_')[1])
        z = redshifts[snapnum]
        print(f'Processing {Snapshot} (z = {z:.2f})')

        # Load data
        data_FFB = load_data(DirName_FFB, Snapshot)
        data_noFFB = load_data(DirName_noFFB, Snapshot)

        # Identify FFB galaxies
        ffb_mask = data_FFB['FFBRegime'] == 1
        n_ffb = np.sum(ffb_mask)

        if n_ffb == 0:
            print(f"  No FFB galaxies at {Snapshot}, skipping.")
            continue

        # Get Mvir of FFB galaxies and match to no-FFB catalogue
        mvir_ffb = data_FFB['Mvir'][ffb_mask]
        matched_indices = match_by_mvir(mvir_ffb, data_noFFB['Mvir'])

        print(f"  Matched {n_ffb} FFB galaxies to no-FFB catalogue by Mvir")

        # Extract properties for FFB galaxies
        stellar_mass_ffb = data_FFB['StellarMass'][ffb_mask]
        coldgas_ffb = data_FFB['ColdGas'][ffb_mask]
        metals_coldgas_ffb = data_FFB['MetalsColdGas'][ffb_mask]
        sfr_ffb = data_FFB['SfrDisk'][ffb_mask] + data_FFB['SfrBulge'][ffb_mask]
        type_ffb = data_FFB['Type'][ffb_mask]

        # Extract properties for Mvir-matched no-FFB galaxies
        stellar_mass_noffb = data_noFFB['StellarMass'][matched_indices]
        coldgas_noffb = data_noFFB['ColdGas'][matched_indices]
        metals_coldgas_noffb = data_noFFB['MetalsColdGas'][matched_indices]
        sfr_noffb = data_noFFB['SfrDisk'][matched_indices] + data_noFFB['SfrBulge'][matched_indices]
        type_noffb = data_noFFB['Type'][matched_indices]

        # Store for plotting
        all_data.append({
            'z': z,
            'snapshot': Snapshot,
            'ffb': {
                'stellar_mass': stellar_mass_ffb,
                'coldgas': coldgas_ffb,
                'metals_coldgas': metals_coldgas_ffb,
                'sfr': sfr_ffb,
                'type': type_ffb,
                'mvir': mvir_ffb
            },
            'noffb': {
                'stellar_mass': stellar_mass_noffb,
                'coldgas': coldgas_noffb,
                'metals_coldgas': metals_coldgas_noffb,
                'sfr': sfr_noffb,
                'type': type_noffb,
                'mvir': data_noFFB['Mvir'][matched_indices]
            }
        })

    # Now plot all data
    print("\nGenerating plots...")

    for data in all_data:
        z = data['z']
        color = cmap((z - z_min) / (z_max - z_min))

        # Apply selection: stellar mass cut only (all galaxy types)
        # Note: Gas fraction cut removed because FFB galaxies are gas-poor by nature
        mask_ffb = data['ffb']['stellar_mass'] > 1.0e8
        mask_noffb = data['noffb']['stellar_mass'] > 1.0e8

        if np.sum(mask_ffb) < 2 or np.sum(mask_noffb) < 2:
            continue

        # Extract masked data
        sm_ffb = data['ffb']['stellar_mass'][mask_ffb]
        sfr_ffb = data['ffb']['sfr'][mask_ffb]
        cg_ffb = data['ffb']['coldgas'][mask_ffb]
        mcg_ffb = data['ffb']['metals_coldgas'][mask_ffb]
        mvir_ffb = data['ffb']['mvir'][mask_ffb]

        sm_noffb = data['noffb']['stellar_mass'][mask_noffb]
        sfr_noffb = data['noffb']['sfr'][mask_noffb]
        cg_noffb = data['noffb']['coldgas'][mask_noffb]
        mcg_noffb = data['noffb']['metals_coldgas'][mask_noffb]
        mvir_noffb = data['noffb']['mvir'][mask_noffb]

        log_mvir_ffb = np.log10(mvir_ffb)
        log_mvir_noffb = np.log10(mvir_noffb)

        # ----- Panel (0,0): SFR vs Mvir -----
        log_sfr_ffb = np.log10(sfr_ffb + 1e-10)
        log_sfr_noffb = np.log10(sfr_noffb + 1e-10)

        bc_ffb, med_ffb = compute_medians(log_mvir_ffb, log_sfr_ffb, mvir_bins)
        bc_noffb, med_noffb = compute_medians(log_mvir_noffb, log_sfr_noffb, mvir_bins)

        valid_ffb = ~np.isnan(med_ffb)
        valid_noffb = ~np.isnan(med_noffb)

        if np.sum(valid_ffb) > 1:
            axes[0, 0].plot(bc_ffb[valid_ffb], med_ffb[valid_ffb], '-', color=color, linewidth=1.5)
        if np.sum(valid_noffb) > 1:
            axes[0, 0].plot(bc_noffb[valid_noffb], med_noffb[valid_noffb], '--', color=color, linewidth=1.5)

        # ----- Panel (0,1): sSFR vs Mvir -----
        log_ssfr_ffb = np.log10(sfr_ffb / sm_ffb + 1e-15)
        log_ssfr_noffb = np.log10(sfr_noffb / sm_noffb + 1e-15)

        bc_ffb, med_ffb = compute_medians(log_mvir_ffb, log_ssfr_ffb, mvir_bins)
        bc_noffb, med_noffb = compute_medians(log_mvir_noffb, log_ssfr_noffb, mvir_bins)

        valid_ffb = ~np.isnan(med_ffb)
        valid_noffb = ~np.isnan(med_noffb)

        if np.sum(valid_ffb) > 1:
            axes[0, 1].plot(bc_ffb[valid_ffb], med_ffb[valid_ffb], '-', color=color, linewidth=1.5)
        if np.sum(valid_noffb) > 1:
            axes[0, 1].plot(bc_noffb[valid_noffb], med_noffb[valid_noffb], '--', color=color, linewidth=1.5)

        # ----- Panel (1,0): Metallicity vs Mvir -----
        # Only require non-zero gas to compute metallicity
        valid_gas_ffb = (cg_ffb > 0) & (mcg_ffb > 0)
        valid_gas_noffb = (cg_noffb > 0) & (mcg_noffb > 0)

        if np.sum(valid_gas_ffb) >= 3:
            Z_ffb = np.log10((mcg_ffb[valid_gas_ffb] / cg_ffb[valid_gas_ffb]) / 0.02) + 9.0
            bc_ffb, med_ffb = compute_medians(log_mvir_ffb[valid_gas_ffb], Z_ffb, mvir_bins)
            valid = ~np.isnan(med_ffb)
            if np.sum(valid) > 1:
                axes[1, 0].plot(bc_ffb[valid], med_ffb[valid], '-', color=color, linewidth=1.5)

        if np.sum(valid_gas_noffb) >= 3:
            Z_noffb = np.log10((mcg_noffb[valid_gas_noffb] / cg_noffb[valid_gas_noffb]) / 0.02) + 9.0
            bc_noffb, med_noffb = compute_medians(log_mvir_noffb[valid_gas_noffb], Z_noffb, mvir_bins)
            valid = ~np.isnan(med_noffb)
            if np.sum(valid) > 1:
                axes[1, 0].plot(bc_noffb[valid], med_noffb[valid], '--', color=color, linewidth=1.5)

        # ----- Panel (1,1): Stellar-to-Halo Mass Relation (SHMR) -----
        log_sm_ffb = np.log10(sm_ffb)
        log_sm_noffb = np.log10(sm_noffb)

        bc_ffb, med_ffb = compute_medians(log_mvir_ffb, log_sm_ffb, mvir_bins)
        bc_noffb, med_noffb = compute_medians(log_mvir_noffb, log_sm_noffb, mvir_bins)

        valid_ffb = ~np.isnan(med_ffb)
        valid_noffb = ~np.isnan(med_noffb)

        if np.sum(valid_ffb) > 1:
            axes[1, 1].plot(bc_ffb[valid_ffb], med_ffb[valid_ffb], '-', color=color, linewidth=1.5)
        if np.sum(valid_noffb) > 1:
            axes[1, 1].plot(bc_noffb[valid_noffb], med_noffb[valid_noffb], '--', color=color, linewidth=1.5)

    # Configure axes
    # Panel (0,0): SFR
    axes[0, 0].set_xlabel(r'$\log_{10}(M_{\mathrm{vir}}\ [M_\odot])$')
    axes[0, 0].set_ylabel(r'$\log_{10}(\mathrm{SFR}\ [M_\odot/\mathrm{yr}])$')
    axes[0, 0].set_xlim(10, 12.5)
    axes[0, 0].set_ylim(-1, 3)
    axes[0, 0].set_title('SFR vs. Halo Mass')

    # Panel (0,1): sSFR
    axes[0, 1].set_xlabel(r'$\log_{10}(M_{\mathrm{vir}}\ [M_\odot])$')
    axes[0, 1].set_ylabel(r'$\log_{10}(\mathrm{sSFR}\ [\mathrm{yr}^{-1}])$')
    axes[0, 1].set_xlim(10, 12.5)
    axes[0, 1].set_ylim(-9, -7)
    axes[0, 1].axhline(y=sSFRcut, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    axes[0, 1].set_title('sSFR vs. Halo Mass')

    # Panel (1,0): Metallicity
    axes[1, 0].set_xlabel(r'$\log_{10}(M_{\mathrm{vir}}\ [M_\odot])$')
    axes[1, 0].set_ylabel(r'$12 + \log_{10}(\mathrm{O/H})$')
    axes[1, 0].set_xlim(10, 12.5)
    axes[1, 0].set_ylim(7.5, 9.5)
    axes[1, 0].set_title('Metallicity vs. Halo Mass')

    # Panel (1,1): Stellar-to-Halo Mass Relation
    axes[1, 1].set_xlabel(r'$\log_{10}(M_{\mathrm{vir}}\ [M_\odot])$')
    axes[1, 1].set_ylabel(r'$\log_{10}(M_\star\ [M_\odot])$')
    axes[1, 1].set_xlim(10, 12.5)
    axes[1, 1].set_ylim(7, 11)
    axes[1, 1].set_title('Stellar-Halo Mass Relation')

    # Add legend for line styles
    legend_elements = [
        Line2D([0], [0], color='black', linestyle='-', linewidth=1.5, label='FFB'),
        Line2D([0], [0], color='black', linestyle='--', linewidth=1.5, label='No-FFB (Mvir-matched)')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.98), fontsize=12)

    plt.tight_layout(rect=[0, 0.02, 0.88, 0.95])

    # Add colorbar outside the plots on the right
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=z_min, vmax=z_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Redshift', fontsize=14)

    output_file = OutputDir + 'ffb_comparison_2x2_grid_shmr' + OutputFormat
    plt.savefig(output_file, bbox_inches='tight')
    print(f"\nSaved: {output_file}")
    plt.close()


def plot_ffb_comparison_grid_6panel():
    """Create 2x3 grid comparing FFB vs no-FFB galaxy properties vs Mvir.

    Panels:
    Row 1: SFR vs Mvir | sSFR vs Mvir | Metallicity vs Mvir
    Row 2: Quiescent fraction vs Mvir | SHMR | Half-mass radius vs stellar mass
    """

    seed(2222)

    OutputDir = DirName_FFB + 'plots/'
    if not os.path.exists(OutputDir):
        os.makedirs(OutputDir)

    # Create 2x3 figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Color map: plasma from z=5 (dark) to z=14 (bright)
    cmap = cm.plasma
    z_min, z_max = 5.0, 14.5

    # Mvir bins for computing medians (log10 Mvir in Msun)
    mvir_bins = np.arange(9.5, 12.5, 0.3)
    # Stellar mass bins for half-mass radius panel
    sm_bins = np.arange(7.5, 11.5, 0.3)

    # Collect data for all snapshots
    print("Loading data from all snapshots...")

    # Data storage for each snapshot
    all_data = []

    for Snapshot in Snapshots:
        snapnum = int(Snapshot.split('_')[1])
        z = redshifts[snapnum]
        print(f'Processing {Snapshot} (z = {z:.2f})')

        # Load data
        data_FFB = load_data(DirName_FFB, Snapshot)
        data_noFFB = load_data(DirName_noFFB, Snapshot)

        # Identify FFB galaxies
        ffb_mask = data_FFB['FFBRegime'] == 1
        n_ffb = np.sum(ffb_mask)

        if n_ffb == 0:
            print(f"  No FFB galaxies at {Snapshot}, skipping.")
            continue

        # Get Mvir of FFB galaxies and match to no-FFB catalogue
        mvir_ffb = data_FFB['Mvir'][ffb_mask]
        matched_indices = match_by_mvir(mvir_ffb, data_noFFB['Mvir'])

        print(f"  Matched {n_ffb} FFB galaxies to no-FFB catalogue by Mvir")

        # Extract properties for FFB galaxies
        stellar_mass_ffb = data_FFB['StellarMass'][ffb_mask]
        coldgas_ffb = data_FFB['ColdGas'][ffb_mask]
        metals_coldgas_ffb = data_FFB['MetalsColdGas'][ffb_mask]
        sfr_ffb = data_FFB['SfrDisk'][ffb_mask] + data_FFB['SfrBulge'][ffb_mask]
        type_ffb = data_FFB['Type'][ffb_mask]
        disk_mass_ffb = data_FFB['DiskMass'][ffb_mask]
        bulge_mass_ffb = data_FFB['BulgeMass'][ffb_mask]
        disk_radius_ffb = data_FFB['DiskRadius'][ffb_mask]
        bulge_radius_ffb = data_FFB['BulgeRadius'][ffb_mask]

        # Extract properties for Mvir-matched no-FFB galaxies
        stellar_mass_noffb = data_noFFB['StellarMass'][matched_indices]
        coldgas_noffb = data_noFFB['ColdGas'][matched_indices]
        metals_coldgas_noffb = data_noFFB['MetalsColdGas'][matched_indices]
        sfr_noffb = data_noFFB['SfrDisk'][matched_indices] + data_noFFB['SfrBulge'][matched_indices]
        type_noffb = data_noFFB['Type'][matched_indices]
        disk_mass_noffb = data_noFFB['DiskMass'][matched_indices]
        bulge_mass_noffb = data_noFFB['BulgeMass'][matched_indices]
        disk_radius_noffb = data_noFFB['DiskRadius'][matched_indices]
        bulge_radius_noffb = data_noFFB['BulgeRadius'][matched_indices]

        # Store for plotting
        all_data.append({
            'z': z,
            'snapshot': Snapshot,
            'ffb': {
                'stellar_mass': stellar_mass_ffb,
                'coldgas': coldgas_ffb,
                'metals_coldgas': metals_coldgas_ffb,
                'sfr': sfr_ffb,
                'type': type_ffb,
                'mvir': mvir_ffb,
                'disk_mass': disk_mass_ffb,
                'bulge_mass': bulge_mass_ffb,
                'disk_radius': disk_radius_ffb,
                'bulge_radius': bulge_radius_ffb
            },
            'noffb': {
                'stellar_mass': stellar_mass_noffb,
                'coldgas': coldgas_noffb,
                'metals_coldgas': metals_coldgas_noffb,
                'sfr': sfr_noffb,
                'type': type_noffb,
                'mvir': data_noFFB['Mvir'][matched_indices],
                'disk_mass': disk_mass_noffb,
                'bulge_mass': bulge_mass_noffb,
                'disk_radius': disk_radius_noffb,
                'bulge_radius': bulge_radius_noffb
            }
        })

    # Now plot all data
    print("\nGenerating plots...")

    for data in all_data:
        z = data['z']
        color = cmap((z - z_min) / (z_max - z_min))

        # Apply selection: stellar mass cut only (all galaxy types)
        mask_ffb = data['ffb']['stellar_mass'] > 1.0e8
        mask_noffb = data['noffb']['stellar_mass'] > 1.0e8

        if np.sum(mask_ffb) < 2 or np.sum(mask_noffb) < 2:
            continue

        # Extract masked data
        sm_ffb = data['ffb']['stellar_mass'][mask_ffb]
        sfr_ffb = data['ffb']['sfr'][mask_ffb]
        cg_ffb = data['ffb']['coldgas'][mask_ffb]
        mcg_ffb = data['ffb']['metals_coldgas'][mask_ffb]
        mvir_ffb = data['ffb']['mvir'][mask_ffb]
        disk_mass_ffb = data['ffb']['disk_mass'][mask_ffb]
        bulge_mass_ffb = data['ffb']['bulge_mass'][mask_ffb]
        disk_radius_ffb = data['ffb']['disk_radius'][mask_ffb]
        bulge_radius_ffb = data['ffb']['bulge_radius'][mask_ffb]

        sm_noffb = data['noffb']['stellar_mass'][mask_noffb]
        sfr_noffb = data['noffb']['sfr'][mask_noffb]
        cg_noffb = data['noffb']['coldgas'][mask_noffb]
        mcg_noffb = data['noffb']['metals_coldgas'][mask_noffb]
        mvir_noffb = data['noffb']['mvir'][mask_noffb]
        disk_mass_noffb = data['noffb']['disk_mass'][mask_noffb]
        bulge_mass_noffb = data['noffb']['bulge_mass'][mask_noffb]
        disk_radius_noffb = data['noffb']['disk_radius'][mask_noffb]
        bulge_radius_noffb = data['noffb']['bulge_radius'][mask_noffb]

        log_mvir_ffb = np.log10(mvir_ffb)
        log_mvir_noffb = np.log10(mvir_noffb)
        log_sm_ffb = np.log10(sm_ffb)
        log_sm_noffb = np.log10(sm_noffb)

        # ----- Panel (0,0): SFR vs Mvir -----
        log_sfr_ffb = np.log10(sfr_ffb + 1e-10)
        log_sfr_noffb = np.log10(sfr_noffb + 1e-10)

        bc_ffb, med_ffb = compute_medians(log_mvir_ffb, log_sfr_ffb, mvir_bins)
        bc_noffb, med_noffb = compute_medians(log_mvir_noffb, log_sfr_noffb, mvir_bins)

        valid_ffb = ~np.isnan(med_ffb)
        valid_noffb = ~np.isnan(med_noffb)

        if np.sum(valid_ffb) > 1:
            axes[0, 0].plot(bc_ffb[valid_ffb], med_ffb[valid_ffb], '-', color=color, linewidth=1.5)
        if np.sum(valid_noffb) > 1:
            axes[0, 0].plot(bc_noffb[valid_noffb], med_noffb[valid_noffb], '--', color=color, linewidth=1.5)

        # ----- Panel (0,1): sSFR vs Mvir -----
        log_ssfr_ffb = np.log10(sfr_ffb / sm_ffb + 1e-15)
        log_ssfr_noffb = np.log10(sfr_noffb / sm_noffb + 1e-15)

        bc_ffb, med_ffb = compute_medians(log_mvir_ffb, log_ssfr_ffb, mvir_bins)
        bc_noffb, med_noffb = compute_medians(log_mvir_noffb, log_ssfr_noffb, mvir_bins)

        valid_ffb = ~np.isnan(med_ffb)
        valid_noffb = ~np.isnan(med_noffb)

        if np.sum(valid_ffb) > 1:
            axes[0, 1].plot(bc_ffb[valid_ffb], med_ffb[valid_ffb], '-', color=color, linewidth=1.5)
        if np.sum(valid_noffb) > 1:
            axes[0, 1].plot(bc_noffb[valid_noffb], med_noffb[valid_noffb], '--', color=color, linewidth=1.5)

        # ----- Panel (0,2): Metallicity vs Mvir -----
        valid_gas_ffb = (cg_ffb > 0) & (mcg_ffb > 0)
        valid_gas_noffb = (cg_noffb > 0) & (mcg_noffb > 0)

        if np.sum(valid_gas_ffb) >= 3:
            Z_ffb = np.log10((mcg_ffb[valid_gas_ffb] / cg_ffb[valid_gas_ffb]) / 0.02) + 9.0
            bc_ffb, med_ffb = compute_medians(log_mvir_ffb[valid_gas_ffb], Z_ffb, mvir_bins)
            valid = ~np.isnan(med_ffb)
            if np.sum(valid) > 1:
                axes[0, 2].plot(bc_ffb[valid], med_ffb[valid], '-', color=color, linewidth=1.5)

        if np.sum(valid_gas_noffb) >= 3:
            Z_noffb = np.log10((mcg_noffb[valid_gas_noffb] / cg_noffb[valid_gas_noffb]) / 0.02) + 9.0
            bc_noffb, med_noffb = compute_medians(log_mvir_noffb[valid_gas_noffb], Z_noffb, mvir_bins)
            valid = ~np.isnan(med_noffb)
            if np.sum(valid) > 1:
                axes[0, 2].plot(bc_noffb[valid], med_noffb[valid], '--', color=color, linewidth=1.5)

        # ----- Panel (1,0): Quiescent fraction vs Mvir -----
        bc_ffb, fq_ffb = compute_quiescent_fraction_mvir(mvir_ffb, sm_ffb, sfr_ffb, mvir_bins, ssfr_cut=sSFRcut)
        bc_noffb, fq_noffb = compute_quiescent_fraction_mvir(mvir_noffb, sm_noffb, sfr_noffb, mvir_bins, ssfr_cut=sSFRcut)

        valid_ffb = ~np.isnan(fq_ffb)
        valid_noffb = ~np.isnan(fq_noffb)

        if np.sum(valid_ffb) > 1:
            axes[1, 0].plot(bc_ffb[valid_ffb], fq_ffb[valid_ffb], '-', color=color, linewidth=1.5)
        if np.sum(valid_noffb) > 1:
            axes[1, 0].plot(bc_noffb[valid_noffb], fq_noffb[valid_noffb], '--', color=color, linewidth=1.5)

        # ----- Panel (1,1): SHMR (Stellar Mass vs Mvir) -----
        bc_ffb, med_ffb = compute_medians(log_mvir_ffb, log_sm_ffb, mvir_bins)
        bc_noffb, med_noffb = compute_medians(log_mvir_noffb, log_sm_noffb, mvir_bins)

        valid_ffb = ~np.isnan(med_ffb)
        valid_noffb = ~np.isnan(med_noffb)

        if np.sum(valid_ffb) > 1:
            axes[1, 1].plot(bc_ffb[valid_ffb], med_ffb[valid_ffb], '-', color=color, linewidth=1.5)
        if np.sum(valid_noffb) > 1:
            axes[1, 1].plot(bc_noffb[valid_noffb], med_noffb[valid_noffb], '--', color=color, linewidth=1.5)

        # ----- Panel (1,2): Half-mass radius vs stellar mass -----
        # Compute half-mass radius as mass-weighted combination of disk and bulge
        # For disk: half-mass radius ~ 1.68 * scale radius (for exponential disk)
        total_mass_ffb = disk_mass_ffb + bulge_mass_ffb
        total_mass_noffb = disk_mass_noffb + bulge_mass_noffb

        # Only compute for galaxies with valid radii
        valid_radius_ffb = (disk_radius_ffb > 0) & (total_mass_ffb > 0)
        valid_radius_noffb = (disk_radius_noffb > 0) & (total_mass_noffb > 0)

        if np.sum(valid_radius_ffb) >= 3:
            # Half-mass radius: mass-weighted average
            # Disk half-mass ~ 1.68 * disk scale radius, bulge assumed to be bulge radius
            disk_half_ffb = 1.68 * disk_radius_ffb[valid_radius_ffb]
            # bulge_half_ffb = bulge_radius_ffb[valid_radius_ffb]
            # For galaxies with no bulge, use disk only
            # bulge_half_ffb = np.where(bulge_half_ffb > 0, bulge_half_ffb, disk_half_ffb)

            half_mass_ffb = disk_half_ffb
            log_rhalf_ffb = np.log10(half_mass_ffb + 1e-10)

            bc_ffb, med_ffb = compute_medians(log_sm_ffb[valid_radius_ffb], log_rhalf_ffb, sm_bins)
            valid = ~np.isnan(med_ffb)
            if np.sum(valid) > 1:
                axes[1, 2].plot(bc_ffb[valid], med_ffb[valid], '-', color=color, linewidth=1.5)

        if np.sum(valid_radius_noffb) >= 3:
            disk_half_noffb = 1.68 * disk_radius_noffb[valid_radius_noffb]
            # bulge_half_noffb = bulge_radius_noffb[valid_radius_noffb]
            # bulge_half_noffb = np.where(bulge_half_noffb > 0, bulge_half_noffb, disk_half_noffb)

            half_mass_noffb = disk_half_noffb
            log_rhalf_noffb = np.log10(half_mass_noffb + 1e-10)

            bc_noffb, med_noffb = compute_medians(log_sm_noffb[valid_radius_noffb], log_rhalf_noffb, sm_bins)
            valid = ~np.isnan(med_noffb)
            if np.sum(valid) > 1:
                axes[1, 2].plot(bc_noffb[valid], med_noffb[valid], '--', color=color, linewidth=1.5)

    # Configure axes
    # Panel (0,0): SFR vs Mvir
    axes[0, 0].set_xlabel(r'$\log_{10}(M_{\mathrm{vir}}\ [M_\odot])$')
    axes[0, 0].set_ylabel(r'$\log_{10}(\mathrm{SFR}\ [M_\odot/\mathrm{yr}])$')
    axes[0, 0].set_xlim(10, 12.5)
    axes[0, 0].set_ylim(-1, 3)
    axes[0, 0].set_title('SFR vs. Halo Mass')

    # Panel (0,1): sSFR vs Mvir
    axes[0, 1].set_xlabel(r'$\log_{10}(M_{\mathrm{vir}}\ [M_\odot])$')
    axes[0, 1].set_ylabel(r'$\log_{10}(\mathrm{sSFR}\ [\mathrm{yr}^{-1}])$')
    axes[0, 1].set_xlim(10, 12.5)
    axes[0, 1].set_ylim(-9, -7)
    axes[0, 1].axhline(y=sSFRcut, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    axes[0, 1].set_title('sSFR vs. Halo Mass')

    # Panel (0,2): Metallicity vs Mvir
    axes[0, 2].set_xlabel(r'$\log_{10}(M_{\mathrm{vir}}\ [M_\odot])$')
    axes[0, 2].set_ylabel(r'$12 + \log_{10}(\mathrm{O/H})$')
    axes[0, 2].set_xlim(10, 12.5)
    axes[0, 2].set_ylim(7.5, 9.5)
    axes[0, 2].set_title('Metallicity vs. Halo Mass')

    # Panel (1,0): Quiescent fraction vs Mvir
    axes[1, 0].set_xlabel(r'$\log_{10}(M_{\mathrm{vir}}\ [M_\odot])$')
    axes[1, 0].set_ylabel(r'$f_{\mathrm{quiescent}}$')
    axes[1, 0].set_xlim(10, 12.5)
    axes[1, 0].set_ylim(0.0, 0.1)
    axes[1, 0].set_title('Quiescent Fraction vs. Halo Mass')

    # Panel (1,1): SHMR
    axes[1, 1].set_xlabel(r'$\log_{10}(M_{\mathrm{vir}}\ [M_\odot])$')
    axes[1, 1].set_ylabel(r'$\log_{10}(M_\star\ [M_\odot])$')
    axes[1, 1].set_xlim(10, 12.5)
    axes[1, 1].set_ylim(7, 11)
    axes[1, 1].set_title('Stellar-Halo Mass Relation')

    # Panel (1,2): Half-mass radius vs stellar mass
    axes[1, 2].set_xlabel(r'$\log_{10}(M_\star\ [M_\odot])$')
    axes[1, 2].set_ylabel(r'$\log_{10}(R_{1/2}\ [\mathrm{kpc}])$')
    axes[1, 2].set_xlim(8, 11)
    axes[1, 2].set_ylim(-1, 0.5)
    axes[1, 2].set_title('Size-Mass Relation')

    # Add legend for line styles
    legend_elements = [
        Line2D([0], [0], color='black', linestyle='-', linewidth=1.5, label='FFB'),
        Line2D([0], [0], color='black', linestyle='--', linewidth=1.5, label='No-FFB (Mvir-matched)')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.98), fontsize=12)

    plt.tight_layout(rect=[0, 0.02, 0.92, 0.95])

    # Add colorbar outside the plots on the right
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=z_min, vmax=z_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Redshift', fontsize=14)

    output_file = OutputDir + 'ffb_comparison_2x3_grid' + OutputFormat
    plt.savefig(output_file, bbox_inches='tight')
    print(f"\nSaved: {output_file}")
    plt.close()

def plot_sfr_mvir_contours_grid():
    """Create 2x3 grid of SFR vs Mvir contour plots at different redshifts.
    
    Each panel shows 1, 2, and 3 sigma contours for both FFB and no-FFB populations.
    Panels show increasing redshift from left to right, top to bottom.
    """
    from scipy.ndimage import gaussian_filter
    
    seed(2222)
    
    OutputDir = DirName_FFB + 'plots/'
    if not os.path.exists(OutputDir):
        os.makedirs(OutputDir)
    
    # Create 2x3 figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    # Select 6 snapshots with decreasing redshift (reversed order)
    # From the available snapshots (z~14 to z~5), select 6 evenly spaced
    selected_snapshots = ['Snap_18', 'Snap_16', 'Snap_14', 'Snap_12', 'Snap_10', 'Snap_8']
    
    print("Creating SFR vs Mvir contour plots...")
    
    for idx, Snapshot in enumerate(selected_snapshots):
        snapnum = int(Snapshot.split('_')[1])
        z = redshifts[snapnum]
        ax = axes[idx]
        
        print(f'Processing {Snapshot} (z = {z:.2f})')
        
        # Load data
        data_FFB = load_data(DirName_FFB, Snapshot)
        data_noFFB = load_data(DirName_noFFB, Snapshot)
        
        # Identify FFB galaxies
        ffb_mask = data_FFB['FFBRegime'] == 1
        n_ffb = np.sum(ffb_mask)
        
        if n_ffb == 0:
            print(f"  No FFB galaxies at {Snapshot}, skipping.")
            ax.text(0.5, 0.5, f'No FFB galaxies\nz = {z:.2f}', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Get Mvir of FFB galaxies and match to no-FFB catalogue
        mvir_ffb = data_FFB['Mvir'][ffb_mask]
        matched_indices = match_by_mvir(mvir_ffb, data_noFFB['Mvir'])
        
        print(f"  Matched {n_ffb} FFB galaxies to no-FFB catalogue by Mvir")
        
        # Extract properties for FFB galaxies
        sfr_ffb = data_FFB['SfrDisk'][ffb_mask] + data_FFB['SfrBulge'][ffb_mask]
        stellar_mass_ffb = data_FFB['StellarMass'][ffb_mask]
        
        # Extract properties for Mvir-matched no-FFB galaxies
        sfr_noffb = data_noFFB['SfrDisk'][matched_indices] + data_noFFB['SfrBulge'][matched_indices]
        stellar_mass_noffb = data_noFFB['StellarMass'][matched_indices]
        mvir_noffb = data_noFFB['Mvir'][matched_indices]
        
        # Apply selection: stellar mass cut
        mask_ffb = stellar_mass_ffb > 1.0e8
        mask_noffb = stellar_mass_noffb > 1.0e8
        
        if np.sum(mask_ffb) < 2 or np.sum(mask_noffb) < 2:
            ax.text(0.5, 0.5, f'Insufficient data\nz = {z:.2f}', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Extract masked data
        log_mvir_ffb = np.log10(mvir_ffb[mask_ffb])
        log_sfr_ffb = np.log10(sfr_ffb[mask_ffb] + 1e-10)
        
        log_mvir_noffb = np.log10(mvir_noffb[mask_noffb])
        log_sfr_noffb = np.log10(sfr_noffb[mask_noffb] + 1e-10)
        
        # Define grid for histograms
        mvir_range = [10, 13]
        sfr_range = [-2, 3.5]
        nbins = 40
        
        # Helper function to plot sigma contours
        def plot_sigma_contours(x, y, color, label):
            """Plot 1, 2, 3 sigma filled contours for given data."""
            try:
                # Create 2D histogram
                H, xedges, yedges = np.histogram2d(x, y, bins=nbins, 
                                                   range=[mvir_range, sfr_range])
                
                # Smooth the histogram
                H_smooth = gaussian_filter(H, sigma=1.5)
                
                # Sort the flattened histogram
                H_flat = H_smooth.flatten()
                inds = np.argsort(H_flat)[::-1]
                H_sorted = H_flat[inds]
                
                # Calculate cumulative sum
                H_cumsum = np.cumsum(H_sorted)
                H_sum = np.sum(H_sorted)
                
                # Find levels for 1, 2, 3 sigma (68%, 95%, 99.7%)
                sigma_levels = [0.68, 0.95, 0.997]
                contour_levels = []
                for level in sigma_levels:
                    idx = np.searchsorted(H_cumsum, level * H_sum)
                    if idx < len(H_sorted):
                        contour_levels.append(H_sorted[idx])
                
                if len(contour_levels) == 0:
                    return
                
                # Get bin centers
                xcenters = 0.5 * (xedges[1:] + xedges[:-1])
                ycenters = 0.5 * (yedges[1:] + yedges[:-1])
                
                # Plot filled contours with different alpha levels
                # 3-sigma (lightest), 2-sigma (medium), 1-sigma (darkest)
                alphas = [0.2, 0.4, 0.6]  # Alpha for 3, 2, 1 sigma
                
                # Sort contour levels in increasing order (required by matplotlib)
                contour_levels_sorted = sorted(contour_levels)
                
                # Plot from outermost (3-sigma) to innermost (1-sigma)
                # contour_levels_sorted[0] = 3-sigma, [1] = 2-sigma, [2] = 1-sigma
                for i in range(len(contour_levels_sorted)):
                    if i == 0:
                        # Outermost contour (3-sigma)
                        ax.contourf(xcenters, ycenters, H_smooth.T, 
                                   levels=[contour_levels_sorted[i], H_smooth.max()], 
                                   colors=[color], alpha=alphas[0])
                    else:
                        # Inner contours
                        ax.contourf(xcenters, ycenters, H_smooth.T, 
                                   levels=[contour_levels_sorted[i], H_smooth.max()], 
                                   colors=[color], alpha=alphas[i])
                
                # Add contour lines for clarity
                ax.contour(xcenters, ycenters, H_smooth.T, 
                          levels=contour_levels_sorted, colors=[color], 
                          linewidths=1.0, alpha=0.8)
                
                # Add dummy patch for legend
                from matplotlib.patches import Patch
                ax.plot([], [], color=color, linewidth=2.0, label=label)
                
            except Exception as e:
                print(f"  Error plotting contours: {e}")
        
        # Plot contours for both populations
        plot_sigma_contours(log_mvir_ffb, log_sfr_ffb, 'dodgerblue', 'FFB')
        plot_sigma_contours(log_mvir_noffb, log_sfr_noffb, 'firebrick', 'No-FFB')
        
        # Configure axes
        ax.set_xlabel(r'$\log_{10}(M_{\mathrm{vir}}\ [M_\odot])$')
        ax.set_ylabel(r'$\log_{10}(\mathrm{SFR}\ [M_\odot/\mathrm{yr}])$')
        ax.set_xlim(mvir_range)
        ax.set_ylim(sfr_range)
        ax.set_title(f'z = {z:.2f}')
        # ax.grid(True, alpha=0.3, linestyle=':')
        
        # Add legend only to first panel
        if idx == 0:
            ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    
    # Add overall title
    # fig.suptitle('SFR vs. Halo Mass: FFB vs. No-FFB Comparison (1, 2, 3σ contours)', 
    #              fontsize=16, y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    output_file = OutputDir + 'sfr_mvir_contours_grid' + OutputFormat
    plt.savefig(output_file, bbox_inches='tight', dpi=150)
    print(f"\nSaved: {output_file}")
    plt.close()

# ==================================================================

if __name__ == "__main__":
    plot_ffb_comparison_grid()
    plot_ffb_comparison_grid_shmr()
    plot_ffb_comparison_grid_6panel()
    plot_sfr_mvir_contours_grid()
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

try:
    from astropy.table import Table
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False
    print("Warning: astropy not available, observational data will not be loaded")

# ========================== USER OPTIONS ==========================

DirName_FFB = './output/millennium/'
DirName_noFFB = './output/millennium_noFFB/'
FileName = 'model_0.hdf5'

# Snapshots from z~16 to z~5
Snapshots = ['Snap_5', 'Snap_6', 'Snap_7', 'Snap_8', 'Snap_9', 'Snap_10', 'Snap_11', 'Snap_12', 'Snap_13',
             'Snap_14', 'Snap_15', 'Snap_16', 'Snap_17', 'Snap_18', 'Snap_19', 'Snap_20', 'Snap_21']

# Additional FFB models with different star formation efficiencies for density plots
FFB_Models = [
    {'name': 'FFB 30%', 'dir': './output/testmodels/Millennium_FFB30/', 'sfe': 0.30},
    {'name': 'FFB 40%', 'dir': './output/testmodels/Millennium_FFB40/', 'sfe': 0.40},
    {'name': 'FFB 50%', 'dir': './output/testmodels/Millennium_FFB50/', 'sfe': 0.50},
    {'name': 'FFB 80%', 'dir': './output/testmodels/Millennium_FFB80/', 'sfe': 0.80},
    {'name': 'FFB 100%', 'dir': './output/testmodels/Millennium_FFB100/', 'sfe': 1.00},
]

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

# ==================== OBSERVATIONAL DATA LOADING ====================

def load_madau_dickinson_2014_data():
    """Load Madau and Dickinson 2014 SFRD data."""
    if not HAS_ASTROPY:
        return None, None, None, None
    filename = './data/MandD_sfrd_2014.ecsv'
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None, None, None, None

    try:
        table = Table.read(filename, format='ascii.ecsv')
        z = table['z_min']
        re = table['log_psi']
        re_err_plus = table['e_log_psi_up']
        re_err_minus = table['e_log_psi_lo']
        return z, re, re_err_plus, re_err_minus
    except Exception as e:
        print(f"Error loading Madau and Dickinson 2014 SFRD data: {e}")
    return None, None, None, None

def load_madau_dickinson_smd_2014_data():
    """Load Madau and Dickinson 2014 SMD data."""
    if not HAS_ASTROPY:
        return None, None, None, None
    filename = './data/MandD_smd_2014.ecsv'
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None, None, None, None

    try:
        table = Table.read(filename, format='ascii.ecsv')
        z = table['z_min']
        re = table['log_rho']
        re_err_plus = table['e_log_rho_up']
        re_err_minus = table['e_log_rho_lo']
        return z, re, re_err_plus, re_err_minus
    except Exception as e:
        print(f"Error loading Madau and Dickinson 2014 SMD data: {e}")
    return None, None, None, None

def load_kikuchihara_smd_2020_data():
    """Load Kikuchihara et al. 2020 SMD data."""
    if not HAS_ASTROPY:
        return None, None, None, None
    filename = './data/kikuchihara_smd_2020.ecsv'
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None, None, None, None

    try:
        table = Table.read(filename, format='ascii.ecsv')
        z = table['z']
        re = table['log_rho_star']
        re_err_plus = table['e_log_rho_star_upper']
        re_err_minus = table['e_log_rho_star_lower']
        return z, re, re_err_plus, re_err_minus
    except Exception as e:
        print(f"Error loading Kikuchihara 2020 SMD data: {e}")
    return None, None, None, None

def load_papovich_smd_2023_data():
    """Load Papovich et al. 2023 SMD data."""
    if not HAS_ASTROPY:
        return None, None, None, None
    filename = './data/papovich_smd_2023.ecsv'
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None, None, None, None

    try:
        table = Table.read(filename, format='ascii.ecsv')
        z = table['z']
        re = table['log_rho_star']
        re_err_plus = table['e_log_rho_star_upper']
        re_err_minus = table['e_log_rho_star_lower']
        return z, re, re_err_plus, re_err_minus
    except Exception as e:
        print(f"Error loading Papovich 2023 SMD data: {e}")
    return None, None, None, None

def load_oesch_sfrd_2018_data():
    """Load Oesch et al. 2018 SFRD data."""
    if not HAS_ASTROPY:
        return None, None, None, None
    filename = './data/oesch_sfrd_2018.ecsv'
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None, None, None, None

    try:
        table = Table.read(filename, format='ascii.ecsv')
        z = table['z']
        re = table['log_rho_sfr']
        re_err_plus = table['e_log_rho_sfr_upper']
        re_err_minus = table['e_log_rho_sfr_lower']
        return z, re, re_err_plus, re_err_minus
    except Exception as e:
        print(f"Error loading Oesch 2018 SFRD data: {e}")
    return None, None, None, None

def load_mcleod_rho_sfr_2024_data():
    """Load McLeod et al. 2024 SFR density data."""
    if not HAS_ASTROPY:
        return None, None, None, None
    filename = './data/mcleod_rhouv_2024.ecsv'
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None, None, None, None

    try:
        table = Table.read(filename, format='ascii.ecsv')
        z = table['z']
        re = table['log_rho_sfr']
        re_err_plus = np.zeros_like(re)
        re_err_minus = np.zeros_like(re)
        return z, re, re_err_plus, re_err_minus
    except Exception as e:
        print(f"Error loading McLeod 2024 SFRD data: {e}")
    return None, None, None, None

def load_harikane_sfr_density_2023_data():
    """Load Harikane et al. 2023 SFR density data."""
    if not HAS_ASTROPY:
        return None, None, None, None
    filename = './data/harikane_density_2023.ecsv'
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None, None, None, None

    try:
        table = Table.read(filename, format='ascii.ecsv')
        z = table['z']
        re = table['log_rho_SFR_UV']
        re_err_plus = table['e_log_rho_SFR_UV_upper']
        re_err_minus = table['e_log_rho_SFR_UV_lower']
        return z, re, re_err_plus, re_err_minus
    except Exception as e:
        print(f"Error loading Harikane 2023 SFRD data: {e}")
    return None, None, None, None

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
        Line2D([0], [0], color='black', linestyle='-', linewidth=1.5, label='SAGE26'),
        Line2D([0], [0], color='black', linestyle='--', linewidth=1.5, label='SAGE26 (no FFB)')
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
        Line2D([0], [0], color='black', linestyle='-', linewidth=1.5, label='SAGE26'),
        Line2D([0], [0], color='black', linestyle='--', linewidth=1.5, label='SAGE26 (no FFB)')
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
    Row 1: SFR vs Mvir | sSFR vs Mvir | SFRD vs redshift
    Row 2: Quiescent fraction vs Mvir | SHMR | SMD vs redshift
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

    # Volume for density calculations
    volume = (BoxSize / Hubble_h)**3.0 * VolumeFraction  # Mpc^3

    # Collect data for all snapshots
    print("Loading data from all snapshots...")

    # Data storage for each snapshot
    all_data = []

    # Arrays to store density evolution data for panels (0,2) and (1,2)
    redshifts_density = []
    sfrd_ffb_list = []
    sfrd_noffb_list = []
    smd_ffb_list = []
    smd_noffb_list = []

    for Snapshot in Snapshots:
        snapnum = int(Snapshot.split('_')[1])
        z = redshifts[snapnum]
        print(f'Processing {Snapshot} (z = {z:.2f})')

        # Load data
        data_FFB = load_data(DirName_FFB, Snapshot)
        data_noFFB = load_data(DirName_noFFB, Snapshot)

        # Identify FFB galaxies in FFB model
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

        # Store for plotting (panels 0,0; 0,1; 1,0; 1,1)
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
            },
            'noffb': {
                'stellar_mass': stellar_mass_noffb,
                'coldgas': coldgas_noffb,
                'metals_coldgas': metals_coldgas_noffb,
                'sfr': sfr_noffb,
                'type': type_noffb,
                'mvir': data_noFFB['Mvir'][matched_indices],
            }
        })

        # Compute SFRD and SMD for density panels (0,2) and (1,2)
        # For FFB galaxies
        total_sfr_ffb = np.sum(sfr_ffb)
        total_sm_ffb = np.sum(stellar_mass_ffb)

        # For Mvir-matched no-FFB galaxies
        total_sfr_noffb = np.sum(sfr_noffb)
        total_sm_noffb = np.sum(stellar_mass_noffb)

        # Store density data
        redshifts_density.append(z)
        if total_sfr_ffb > 0:
            sfrd_ffb_list.append(np.log10(total_sfr_ffb / volume))
        else:
            sfrd_ffb_list.append(np.nan)
        if total_sfr_noffb > 0:
            sfrd_noffb_list.append(np.log10(total_sfr_noffb / volume))
        else:
            sfrd_noffb_list.append(np.nan)
        if total_sm_ffb > 0:
            smd_ffb_list.append(np.log10(total_sm_ffb / volume))
        else:
            smd_ffb_list.append(np.nan)
        if total_sm_noffb > 0:
            smd_noffb_list.append(np.log10(total_sm_noffb / volume))
        else:
            smd_noffb_list.append(np.nan)

    # Convert density lists to arrays
    redshifts_density = np.array(redshifts_density)
    sfrd_ffb_arr = np.array(sfrd_ffb_list)
    sfrd_noffb_arr = np.array(sfrd_noffb_list)
    smd_ffb_arr = np.array(smd_ffb_list)
    smd_noffb_arr = np.array(smd_noffb_list)

    # Now plot all data for panels (0,0), (0,1), (1,0), (1,1)
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
        mvir_ffb = data['ffb']['mvir'][mask_ffb]

        sm_noffb = data['noffb']['stellar_mass'][mask_noffb]
        sfr_noffb = data['noffb']['sfr'][mask_noffb]
        mvir_noffb = data['noffb']['mvir'][mask_noffb]

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

    # ----- Panel (0,2): SFRD vs Redshift -----
    # Sort by redshift for clean plotting
    sort_idx = np.argsort(redshifts_density)
    z_sorted = redshifts_density[sort_idx]
    sfrd_ffb_sorted = sfrd_ffb_arr[sort_idx]
    sfrd_noffb_sorted = sfrd_noffb_arr[sort_idx]

    # Plot SFRD for FFB and non-FFB galaxies (main model)
    valid_ffb = ~np.isnan(sfrd_ffb_sorted)
    valid_noffb = ~np.isnan(sfrd_noffb_sorted)

    if np.sum(valid_ffb) > 1:
        axes[0, 2].plot(z_sorted[valid_ffb], sfrd_ffb_sorted[valid_ffb], '-',
                       color='black', linewidth=2.5, label='SAGE26')
    if np.sum(valid_noffb) > 1:
        axes[0, 2].plot(z_sorted[valid_noffb], sfrd_noffb_sorted[valid_noffb], '--',
                       color='firebrick', linewidth=2.5, label='SAGE26 (no FFB)')

    # Add additional FFB models with different SFE values (jet_r colormap, no legend)
    cmap_sfe = cm.jet_r
    sfe_min, sfe_max = 0.2, 1.0
    for model in FFB_Models:
        model_dir = model['dir']
        sfe = model['sfe']
        model_color = cmap_sfe((sfe - sfe_min) / (sfe_max - sfe_min))

        # Check if model exists
        if not os.path.exists(model_dir + FileName):
            print(f"  Warning: {model_dir + FileName} not found, skipping {model['name']}")
            continue

        # Compute SFRD for this model
        model_redshifts = []
        model_sfrd = []

        for Snapshot in Snapshots:
            snapnum = int(Snapshot.split('_')[1])
            z = redshifts[snapnum]

            try:
                # Load data for this model
                with h5.File(model_dir + FileName, 'r') as f:
                    sfr_disk = np.array(f[Snapshot]['SfrDisk'])
                    sfr_bulge = np.array(f[Snapshot]['SfrBulge'])
                    ffb_regime = np.array(f[Snapshot]['FFBRegime'])

                # Select FFB galaxies
                ffb_mask = ffb_regime == 1
                if np.sum(ffb_mask) == 0:
                    continue

                sfr_total = sfr_disk[ffb_mask] + sfr_bulge[ffb_mask]
                total_sfr = np.sum(sfr_total)

                if total_sfr > 0:
                    model_redshifts.append(z)
                    model_sfrd.append(np.log10(total_sfr / volume))
            except Exception as e:
                continue

        # Plot this model (no label for legend)
        if len(model_redshifts) > 1:
            sort_idx_model = np.argsort(model_redshifts)
            axes[0, 2].plot(np.array(model_redshifts)[sort_idx_model],
                          np.array(model_sfrd)[sort_idx_model], '-',
                          color=model_color, linewidth=1.5, alpha=0.7)

    # Add SFRD observational data
    z_madau, re_madau, re_err_plus_madau, re_err_minus_madau = load_madau_dickinson_2014_data()
    if z_madau is not None:
        mask = (z_madau >= 5) & (z_madau <= 16)
        if np.sum(mask) > 0:
            axes[0, 2].errorbar(z_madau[mask], re_madau[mask],
                               yerr=[re_err_minus_madau[mask], re_err_plus_madau[mask]],
                               fmt='o', color='black', markersize=6, alpha=0.8,
                               label='Madau & Dickinson 14', capsize=2, linewidth=1.5, zorder=5)

    z_oesch, re_oesch, re_err_plus_oesch, re_err_minus_oesch = load_oesch_sfrd_2018_data()
    if z_oesch is not None:
        mask = (z_oesch >= 5) & (z_oesch <= 16)
        if np.sum(mask) > 0:
            axes[0, 2].errorbar(z_oesch[mask], re_oesch[mask],
                               yerr=[re_err_minus_oesch[mask], re_err_plus_oesch[mask]],
                               fmt='*', color='black', markersize=8, alpha=0.8,
                               label='Oesch+18', capsize=2, linewidth=1.5, zorder=5)

    z_mcleod, re_mcleod, re_err_plus_mcleod, re_err_minus_mcleod = load_mcleod_rho_sfr_2024_data()
    if z_mcleod is not None:
        mask = (z_mcleod >= 5) & (z_mcleod <= 16)
        if np.sum(mask) > 0:
            axes[0, 2].errorbar(z_mcleod[mask], re_mcleod[mask],
                               yerr=[re_err_minus_mcleod[mask], re_err_plus_mcleod[mask]],
                               fmt='v', color='black', markersize=6, alpha=0.8,
                               label='McLeod+24', capsize=2, linewidth=1.5, zorder=5)

    z_harikane, re_harikane, re_err_plus_harikane, re_err_minus_harikane = load_harikane_sfr_density_2023_data()
    if z_harikane is not None:
        mask = (z_harikane >= 5) & (z_harikane <= 16)
        if np.sum(mask) > 0:
            axes[0, 2].errorbar(z_harikane[mask], re_harikane[mask],
                               yerr=[re_err_minus_harikane[mask], re_err_plus_harikane[mask]],
                               fmt='D', color='black', markersize=6, alpha=0.8,
                               label='Harikane+23', capsize=2, linewidth=1.5, zorder=5)

    # ----- Panel (1,2): SMD vs Redshift -----
    smd_ffb_sorted = smd_ffb_arr[sort_idx]
    smd_noffb_sorted = smd_noffb_arr[sort_idx]

    # Plot SMD for FFB and non-FFB galaxies (main model)
    valid_ffb = ~np.isnan(smd_ffb_sorted)
    valid_noffb = ~np.isnan(smd_noffb_sorted)

    if np.sum(valid_ffb) > 1:
        axes[1, 2].plot(z_sorted[valid_ffb], smd_ffb_sorted[valid_ffb], '-',
                       color='black', linewidth=2.5, label='SAGE26')
    if np.sum(valid_noffb) > 1:
        axes[1, 2].plot(z_sorted[valid_noffb], smd_noffb_sorted[valid_noffb], '--',
                       color='firebrick', linewidth=2.5, label='SAGE26 (no FFB)')

    # Add additional FFB models with different SFE values (jet colormap, no legend)
    for model in FFB_Models:
        model_dir = model['dir']
        sfe = model['sfe']
        model_color = cmap_sfe((sfe - sfe_min) / (sfe_max - sfe_min))

        # Check if model exists
        if not os.path.exists(model_dir + FileName):
            continue

        # Compute SMD for this model
        model_redshifts = []
        model_smd = []

        for Snapshot in Snapshots:
            snapnum = int(Snapshot.split('_')[1])
            z = redshifts[snapnum]

            try:
                # Load data for this model
                with h5.File(model_dir + FileName, 'r') as f:
                    stellar_mass = np.array(f[Snapshot]['StellarMass']) * 1.0e10 / Hubble_h
                    ffb_regime = np.array(f[Snapshot]['FFBRegime'])

                # Select FFB galaxies
                ffb_mask = ffb_regime == 1
                if np.sum(ffb_mask) == 0:
                    continue

                total_sm = np.sum(stellar_mass[ffb_mask])

                if total_sm > 0:
                    model_redshifts.append(z)
                    model_smd.append(np.log10(total_sm / volume))
            except Exception as e:
                continue

        # Plot this model (no label for legend)
        if len(model_redshifts) > 1:
            sort_idx_model = np.argsort(model_redshifts)
            axes[1, 2].plot(np.array(model_redshifts)[sort_idx_model],
                          np.array(model_smd)[sort_idx_model], '-',
                          color=model_color, linewidth=1.5, alpha=0.7)

    # Add SMD observational data
    z_madau_smd, re_madau_smd, re_err_plus_madau_smd, re_err_minus_madau_smd = load_madau_dickinson_smd_2014_data()
    if z_madau_smd is not None:
        mask = (z_madau_smd >= 5) & (z_madau_smd <= 16)
        if np.sum(mask) > 0:
            axes[1, 2].errorbar(z_madau_smd[mask], re_madau_smd[mask],
                               yerr=[re_err_minus_madau_smd[mask], re_err_plus_madau_smd[mask]],
                               fmt='o', color='black', markersize=6, alpha=0.8,
                               label='Madau & Dickinson 14', capsize=2, linewidth=1.5, zorder=5)

    z_kiku, re_kiku, re_err_plus_kiku, re_err_minus_kiku = load_kikuchihara_smd_2020_data()
    if z_kiku is not None:
        mask = (z_kiku >= 5) & (z_kiku <= 16)
        if np.sum(mask) > 0:
            axes[1, 2].errorbar(z_kiku[mask], re_kiku[mask],
                               yerr=[re_err_minus_kiku[mask], re_err_plus_kiku[mask]],
                               fmt='d', color='black', markersize=6, alpha=0.8,
                               label='Kikuchihara+20', capsize=2, linewidth=1.5, zorder=5)

    z_papovich, re_papovich, re_err_plus_papovich, re_err_minus_papovich = load_papovich_smd_2023_data()
    if z_papovich is not None:
        mask = (z_papovich >= 5) & (z_papovich <= 16)
        if np.sum(mask) > 0:
            axes[1, 2].errorbar(z_papovich[mask], re_papovich[mask],
                               yerr=[re_err_minus_papovich[mask], re_err_plus_papovich[mask]],
                               fmt='s', color='black', markersize=6, alpha=0.8,
                               label='Papovich+23', capsize=2, linewidth=1.5, zorder=5)

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

    # Panel (0,2): SFRD vs Redshift
    axes[0, 2].set_xlabel(r'Redshift $z$')
    axes[0, 2].set_ylabel(r'$\log_{10}(\rho_{\mathrm{SFR}}\ [M_\odot\,\mathrm{yr}^{-1}\,\mathrm{Mpc}^{-3}])$')
    axes[0, 2].set_xlim(5, 16)
    axes[0, 2].set_ylim(-5, -1)
    axes[0, 2].set_title('SFR Density vs. Redshift')
    axes[0, 2].legend(loc='upper right', fontsize=8, frameon=False)

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

    # Panel (1,2): SMD vs Redshift
    axes[1, 2].set_xlabel(r'Redshift $z$')
    axes[1, 2].set_ylabel(r'$\log_{10}(\rho_\star\ [M_\odot\,\mathrm{Mpc}^{-3}])$')
    axes[1, 2].set_xlim(5, 16)
    axes[1, 2].set_ylim(3, 8)
    axes[1, 2].set_title('Stellar Mass Density vs. Redshift')
    axes[1, 2].legend(loc='upper right', fontsize=8, frameon=False)

    # Add legend for line styles (for panels with colormap)
    legend_elements = [
        Line2D([0], [0], color='black', linestyle='-', linewidth=1.5, label='SAGE26'),
        Line2D([0], [0], color='black', linestyle='--', linewidth=1.5, label='SAGE26 (no FFB)')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, bbox_to_anchor=(0.35, 0.98), fontsize=12)

    plt.tight_layout(rect=[0, 0.02, 1.0, 0.95])

    # Shift the 3rd column (right panels) to the right to make room for colorbar
    for ax in [axes[0, 2], axes[1, 2]]:
        pos = ax.get_position()
        ax.set_position([pos.x0 + 0.05, pos.y0, pos.width, pos.height])

    # Add colorbar between columns 1 and 2 (between middle and right panels)
    # Position: [left, bottom, width, height] in figure coordinates
    cbar_ax = fig.add_axes([0.68, 0.12, 0.012, 0.76])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=z_min, vmax=z_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Redshift', fontsize=14)

    output_file = OutputDir + 'ffb_comparison_2x3_grid' + OutputFormat
    plt.savefig(output_file, bbox_inches='tight')
    print(f"\nSaved: {output_file}")
    plt.close()


def plot_density_evolution():
    """Create 2x1 figure with SFRD and SMD vs redshift (stacked vertically).

    Top panel: SFRD vs redshift
    Bottom panel: SMD vs redshift

    Shows entire galaxy populations from FFB and no-FFB models, plus additional
    FFB models with different star formation efficiencies.
    """

    seed(2222)

    OutputDir = DirName_FFB + 'plots/'
    if not os.path.exists(OutputDir):
        os.makedirs(OutputDir)

    # Create 2x1 figure (stacked vertically)
    fig, axes = plt.subplots(2, 1, figsize=(8, 10))

    # Volume for density calculations
    volume = (BoxSize / Hubble_h)**3.0 * VolumeFraction  # Mpc^3

    # Colormap for additional FFB models
    cmap_sfe = cm.jet_r
    sfe_min, sfe_max = 0.2, 1.0

    # Collect data for all snapshots
    print("Loading data for density evolution plots...")

    # Arrays to store density evolution data
    redshifts_density = []
    sfrd_ffb_list = []
    sfrd_noffb_list = []
    smd_ffb_list = []
    smd_noffb_list = []

    for Snapshot in Snapshots:
        snapnum = int(Snapshot.split('_')[1])
        z = redshifts[snapnum]
        print(f'Processing {Snapshot} (z = {z:.2f})')

        # Load data
        data_FFB = load_data(DirName_FFB, Snapshot)
        data_noFFB = load_data(DirName_noFFB, Snapshot)

        # Use entire galaxy population from both models
        # Extract properties for ALL galaxies in FFB model
        stellar_mass_ffb = data_FFB['StellarMass']
        sfr_ffb = data_FFB['SfrDisk'] + data_FFB['SfrBulge']

        # Extract properties for ALL galaxies in no-FFB model
        stellar_mass_noffb = data_noFFB['StellarMass']
        sfr_noffb = data_noFFB['SfrDisk'] + data_noFFB['SfrBulge']

        # Compute SFRD and SMD for entire populations
        total_sfr_ffb = np.sum(sfr_ffb)
        total_sm_ffb = np.sum(stellar_mass_ffb)
        total_sfr_noffb = np.sum(sfr_noffb)
        total_sm_noffb = np.sum(stellar_mass_noffb)

        # Store density data
        redshifts_density.append(z)
        sfrd_ffb_list.append(np.log10(total_sfr_ffb / volume) if total_sfr_ffb > 0 else np.nan)
        sfrd_noffb_list.append(np.log10(total_sfr_noffb / volume) if total_sfr_noffb > 0 else np.nan)
        smd_ffb_list.append(np.log10(total_sm_ffb / volume) if total_sm_ffb > 0 else np.nan)
        smd_noffb_list.append(np.log10(total_sm_noffb / volume) if total_sm_noffb > 0 else np.nan)

    # Convert to arrays and sort by redshift
    redshifts_density = np.array(redshifts_density)
    sfrd_ffb_arr = np.array(sfrd_ffb_list)
    sfrd_noffb_arr = np.array(sfrd_noffb_list)
    smd_ffb_arr = np.array(smd_ffb_list)
    smd_noffb_arr = np.array(smd_noffb_list)

    sort_idx = np.argsort(redshifts_density)
    z_sorted = redshifts_density[sort_idx]
    sfrd_ffb_sorted = sfrd_ffb_arr[sort_idx]
    sfrd_noffb_sorted = sfrd_noffb_arr[sort_idx]
    smd_ffb_sorted = smd_ffb_arr[sort_idx]
    smd_noffb_sorted = smd_noffb_arr[sort_idx]

    print("\nGenerating density evolution plots...")

    # ----- Top Panel: SFRD vs Redshift -----
    valid_ffb = ~np.isnan(sfrd_ffb_sorted)
    valid_noffb = ~np.isnan(sfrd_noffb_sorted)

    if np.sum(valid_ffb) > 1:
        axes[0].plot(z_sorted[valid_ffb], sfrd_ffb_sorted[valid_ffb], '-',
                    color='black', linewidth=2.5, label='SAGE26')
    if np.sum(valid_noffb) > 1:
        axes[0].plot(z_sorted[valid_noffb], sfrd_noffb_sorted[valid_noffb], '--',
                    color='firebrick', linewidth=2.5, label='SAGE26 (no FFB)')

    # Add additional FFB models (jet_r colormap, no legend) - use entire population
    for model in FFB_Models:
        model_dir = model['dir']
        sfe = model['sfe']
        model_color = cmap_sfe((sfe - sfe_min) / (sfe_max - sfe_min))

        if not os.path.exists(model_dir + FileName):
            print(f"  Warning: {model_dir + FileName} not found, skipping {model['name']}")
            continue

        model_redshifts = []
        model_sfrd = []

        for Snapshot in Snapshots:
            snapnum = int(Snapshot.split('_')[1])
            z = redshifts[snapnum]

            try:
                with h5.File(model_dir + FileName, 'r') as f:
                    sfr_disk = np.array(f[Snapshot]['SfrDisk'])
                    sfr_bulge = np.array(f[Snapshot]['SfrBulge'])

                # Use all galaxies in the model
                total_sfr = np.sum(sfr_disk + sfr_bulge)
                if total_sfr > 0:
                    model_redshifts.append(z)
                    model_sfrd.append(np.log10(total_sfr / volume))
            except:
                continue

        if len(model_redshifts) > 1:
            sort_idx_model = np.argsort(model_redshifts)
            axes[0].plot(np.array(model_redshifts)[sort_idx_model],
                        np.array(model_sfrd)[sort_idx_model], '-',
                        color=model_color, linewidth=1.5, alpha=0.7)

    # Add SFRD observational data
    z_madau, re_madau, re_err_plus_madau, re_err_minus_madau = load_madau_dickinson_2014_data()
    if z_madau is not None:
        mask = (z_madau >= 5) & (z_madau <= 16)
        if np.sum(mask) > 0:
            axes[0].errorbar(z_madau[mask], re_madau[mask],
                            yerr=[re_err_minus_madau[mask], re_err_plus_madau[mask]],
                            fmt='o', color='black', markersize=6, alpha=0.8,
                            label='Madau & Dickinson 14', capsize=2, linewidth=1.5, zorder=5)

    z_oesch, re_oesch, re_err_plus_oesch, re_err_minus_oesch = load_oesch_sfrd_2018_data()
    if z_oesch is not None:
        mask = (z_oesch >= 5) & (z_oesch <= 16)
        if np.sum(mask) > 0:
            axes[0].errorbar(z_oesch[mask], re_oesch[mask],
                            yerr=[re_err_minus_oesch[mask], re_err_plus_oesch[mask]],
                            fmt='*', color='black', markersize=8, alpha=0.8,
                            label='Oesch+18', capsize=2, linewidth=1.5, zorder=5)

    z_mcleod, re_mcleod, re_err_plus_mcleod, re_err_minus_mcleod = load_mcleod_rho_sfr_2024_data()
    if z_mcleod is not None:
        mask = (z_mcleod >= 5) & (z_mcleod <= 16)
        if np.sum(mask) > 0:
            axes[0].errorbar(z_mcleod[mask], re_mcleod[mask],
                            yerr=[re_err_minus_mcleod[mask], re_err_plus_mcleod[mask]],
                            fmt='v', color='black', markersize=6, alpha=0.8,
                            label='McLeod+24', capsize=2, linewidth=1.5, zorder=5)

    z_harikane, re_harikane, re_err_plus_harikane, re_err_minus_harikane = load_harikane_sfr_density_2023_data()
    if z_harikane is not None:
        mask = (z_harikane >= 5) & (z_harikane <= 16)
        if np.sum(mask) > 0:
            axes[0].errorbar(z_harikane[mask], re_harikane[mask],
                            yerr=[re_err_minus_harikane[mask], re_err_plus_harikane[mask]],
                            fmt='D', color='black', markersize=6, alpha=0.8,
                            label='Harikane+23', capsize=2, linewidth=1.5, zorder=5)

    # ----- Bottom Panel: SMD vs Redshift -----
    valid_ffb = ~np.isnan(smd_ffb_sorted)
    valid_noffb = ~np.isnan(smd_noffb_sorted)

    if np.sum(valid_ffb) > 1:
        axes[1].plot(z_sorted[valid_ffb], smd_ffb_sorted[valid_ffb], '-',
                    color='black', linewidth=2.5, label='SAGE26')
    if np.sum(valid_noffb) > 1:
        axes[1].plot(z_sorted[valid_noffb], smd_noffb_sorted[valid_noffb], '--',
                    color='firebrick', linewidth=2.5, label='SAGE26 (no FFB)')

    # Add additional FFB models (jet_r colormap, no legend) - use entire population
    for model in FFB_Models:
        model_dir = model['dir']
        sfe = model['sfe']
        model_color = cmap_sfe((sfe - sfe_min) / (sfe_max - sfe_min))

        if not os.path.exists(model_dir + FileName):
            continue

        model_redshifts = []
        model_smd = []

        for Snapshot in Snapshots:
            snapnum = int(Snapshot.split('_')[1])
            z = redshifts[snapnum]

            try:
                with h5.File(model_dir + FileName, 'r') as f:
                    stellar_mass = np.array(f[Snapshot]['StellarMass']) * 1.0e10 / Hubble_h

                # Use all galaxies in the model
                total_sm = np.sum(stellar_mass)
                if total_sm > 0:
                    model_redshifts.append(z)
                    model_smd.append(np.log10(total_sm / volume))
            except:
                continue

        if len(model_redshifts) > 1:
            sort_idx_model = np.argsort(model_redshifts)
            axes[1].plot(np.array(model_redshifts)[sort_idx_model],
                        np.array(model_smd)[sort_idx_model], '-',
                        color=model_color, linewidth=1.5, alpha=0.7)

    # Add SMD observational data
    z_madau_smd, re_madau_smd, re_err_plus_madau_smd, re_err_minus_madau_smd = load_madau_dickinson_smd_2014_data()
    if z_madau_smd is not None:
        mask = (z_madau_smd >= 5) & (z_madau_smd <= 16)
        if np.sum(mask) > 0:
            axes[1].errorbar(z_madau_smd[mask], re_madau_smd[mask],
                            yerr=[re_err_minus_madau_smd[mask], re_err_plus_madau_smd[mask]],
                            fmt='o', color='black', markersize=6, alpha=0.8,
                            label='Madau & Dickinson 14', capsize=2, linewidth=1.5, zorder=5)

    z_kiku, re_kiku, re_err_plus_kiku, re_err_minus_kiku = load_kikuchihara_smd_2020_data()
    if z_kiku is not None:
        mask = (z_kiku >= 5) & (z_kiku <= 16)
        if np.sum(mask) > 0:
            axes[1].errorbar(z_kiku[mask], re_kiku[mask],
                            yerr=[re_err_minus_kiku[mask], re_err_plus_kiku[mask]],
                            fmt='d', color='black', markersize=6, alpha=0.8,
                            label='Kikuchihara+20', capsize=2, linewidth=1.5, zorder=5)

    z_papovich, re_papovich, re_err_plus_papovich, re_err_minus_papovich = load_papovich_smd_2023_data()
    if z_papovich is not None:
        mask = (z_papovich >= 5) & (z_papovich <= 16)
        if np.sum(mask) > 0:
            axes[1].errorbar(z_papovich[mask], re_papovich[mask],
                            yerr=[re_err_minus_papovich[mask], re_err_plus_papovich[mask]],
                            fmt='s', color='black', markersize=6, alpha=0.8,
                            label='Papovich+23', capsize=2, linewidth=1.5, zorder=5)

    # Configure axes
    # Top panel: SFRD
    axes[0].set_xlabel(r'Redshift')
    axes[0].set_ylabel(r'$\log_{10} \rho_{\mathrm{SFR}}\ (M_\odot\,\mathrm{yr}^{-1}\,\mathrm{Mpc}^{-3})$')
    axes[0].set_xlim(5, 16)
    axes[0].set_ylim(-5, -1)
    # axes[0].set_title('SFR Density vs. Redshift')
    axes[0].legend(loc='upper right', fontsize=9, frameon=False)

    # Bottom panel: SMD
    axes[1].set_xlabel(r'Redshift')
    axes[1].set_ylabel(r'$\log_{10} \rho_\star\ (M_\odot\,\mathrm{Mpc}^{-3})$')
    axes[1].set_xlim(5, 16)
    axes[1].set_ylim(3, 8)
    # axes[1].set_title('Stellar Mass Density vs. Redshift')
    axes[1].legend(loc='upper right', fontsize=9, frameon=False)

    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1))
    plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(1))

    plt.tight_layout()

    # Hide x-axis tick labels (but keep ticks) for the top plot
    axes[0].set_xlabel("")
    axes[0].set_xticklabels([""] * len(axes[0].get_xticks()))

    output_file = OutputDir + 'ffb_density_evolution' + OutputFormat
    plt.savefig(output_file, bbox_inches='tight')
    print(f"\nSaved: {output_file}")
    plt.close()


def plot_ffb_comparison_4panel():
    """Create 2x2 grid comparing FFB vs no-FFB galaxy properties vs Mvir.

    Panels:
    Row 1: SFR vs Mvir | sSFR vs Mvir
    Row 2: Quiescent fraction vs Mvir | SHMR

    Lines colored by redshift using plasma colormap.
    """

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
    mvir_bins = np.arange(9.5, 14.5, 0.2)

    # Collect data for all snapshots
    print("Loading data for 4-panel comparison plot...")

    all_data = []

    for Snapshot in Snapshots:
        snapnum = int(Snapshot.split('_')[1])
        z = redshifts[snapnum]
        print(f'Processing {Snapshot} (z = {z:.2f})')

        # Load data
        data_FFB = load_data(DirName_FFB, Snapshot)
        data_noFFB = load_data(DirName_noFFB, Snapshot)

        # Identify FFB galaxies in FFB model
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
        sfr_ffb = data_FFB['SfrDisk'][ffb_mask] + data_FFB['SfrBulge'][ffb_mask]

        # Extract properties for Mvir-matched no-FFB galaxies
        stellar_mass_noffb = data_noFFB['StellarMass'][matched_indices]
        sfr_noffb = data_noFFB['SfrDisk'][matched_indices] + data_noFFB['SfrBulge'][matched_indices]

        # Store for plotting
        all_data.append({
            'z': z,
            'snapshot': Snapshot,
            'ffb': {
                'stellar_mass': stellar_mass_ffb,
                'sfr': sfr_ffb,
                'mvir': mvir_ffb,
            },
            'noffb': {
                'stellar_mass': stellar_mass_noffb,
                'sfr': sfr_noffb,
                'mvir': data_noFFB['Mvir'][matched_indices],
            }
        })

    # Plot all data
    print("\nGenerating 4-panel comparison plot...")

    for data in all_data:
        z = data['z']
        color = cmap((z - z_min) / (z_max - z_min))

        # Apply selection: stellar mass cut only
        mask_ffb = data['ffb']['stellar_mass'] > 1.0e6
        mask_noffb = data['noffb']['stellar_mass'] > 1.0e6

        if np.sum(mask_ffb) < 2 or np.sum(mask_noffb) < 2:
            continue

        # Extract masked data
        sm_ffb = data['ffb']['stellar_mass'][mask_ffb]
        sfr_ffb = data['ffb']['sfr'][mask_ffb]
        mvir_ffb = data['ffb']['mvir'][mask_ffb]

        sm_noffb = data['noffb']['stellar_mass'][mask_noffb]
        sfr_noffb = data['noffb']['sfr'][mask_noffb]
        mvir_noffb = data['noffb']['mvir'][mask_noffb]

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

    # Configure axes
    # Panel (0,0): SFR vs Mvir
    axes[0, 0].set_xlabel(r'$\log_{10}(M_{\mathrm{vir}}\ [M_\odot])$')
    axes[0, 0].set_ylabel(r'$\log_{10}(\mathrm{SFR}\ [M_\odot/\mathrm{yr}])$')
    axes[0, 0].set_xlim(10, 12.5)
    axes[0, 0].set_ylim(-0.5, 2.5)
    # axes[0, 0].set_title('SFR vs. Halo Mass')

    # Panel (0,1): sSFR vs Mvir
    axes[0, 1].set_xlabel(r'$\log_{10}(M_{\mathrm{vir}}\ [M_\odot])$')
    axes[0, 1].set_ylabel(r'$\log_{10}(\mathrm{sSFR}\ [\mathrm{yr}^{-1}])$')
    axes[0, 1].set_xlim(10, 12.5)
    axes[0, 1].set_ylim(-8.5, -7.25)
    axes[0, 1].axhline(y=sSFRcut, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    # axes[0, 1].set_title('sSFR vs. Halo Mass')

    # Panel (1,0): Quiescent fraction vs Mvir
    axes[1, 0].set_xlabel(r'$\log_{10}(M_{\mathrm{vir}}\ [M_\odot])$')
    axes[1, 0].set_ylabel(r'$f_{\mathrm{quiescent}}$')
    axes[1, 0].set_xlim(10, 12.5)
    axes[1, 0].set_ylim(0.0, 0.01)
    # axes[1, 0].set_title('Quiescent Fraction vs. Halo Mass')

    # Panel (1,1): SHMR
    axes[1, 1].set_xlabel(r'$\log_{10}(M_{\mathrm{vir}}\ [M_\odot])$')
    axes[1, 1].set_ylabel(r'$\log_{10}(M_\star\ [M_\odot])$')
    axes[1, 1].set_xlim(10, 12.5)
    axes[1, 1].set_ylim(8, 10.5)
    # axes[1, 1].set_title('Stellar-Halo Mass Relation')

    # Add legend for line styles
    legend_elements = [
        Line2D([0], [0], color='black', linestyle='-', linewidth=1.5, label='SAGE26'),
        Line2D([0], [0], color='black', linestyle='--', linewidth=1.5, label='SAGE26 (no FFB)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=12)

    plt.tight_layout(rect=[0, 0.02, 0.88, 0.95])

    # Add colorbar outside the plots on the right
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=z_min, vmax=z_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Redshift', fontsize=14)

    output_file = OutputDir + 'ffb_comparison_2x2_grid_properties' + OutputFormat
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
        mvir_range = [9, 15]
        sfr_range = [-2, 4.5]
        nbins = 40
        
        # Helper function to plot sigma contours
        def plot_sigma_contours(x, y, color, label):
            """Plot 1, 2, 3 sigma filled contours for given data."""
            try:
                # Reduce bins for sparse data
                adaptive_nbins = nbins if len(x) > 5000 else 30 if len(x) > 1000 else 20

                # Create 2D histogram
                H, xedges, yedges = np.histogram2d(x, y, bins=adaptive_nbins,
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
        plot_sigma_contours(log_mvir_ffb, log_sfr_ffb, 'purple', 'SAGE 26')
        plot_sigma_contours(log_mvir_noffb, log_sfr_noffb, 'dodgerblue', 'SAGE26 (no-FFB)')
        
        # Configure axes with fixed limits across all panels
        ax.set_xlabel(r'$\log_{10} M_{\mathrm{vir}}\ (M_\odot)$', fontsize=14)
        ax.set_ylabel(r'$\log_{10} \mathrm{SFR}\ (M_\odot\mathrm{yr}^{-1})$', fontsize=14)
        ax.set_xlim(9, 13.5)
        ax.set_ylim(sfr_range)
        ax.set_title(f'z = {z:.2f}')
        # ax.grid(True, alpha=0.3, linestyle=':')
        
        # Add legend only to first panel
        if idx == 0:
            ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    
    # Add overall title
    # fig.suptitle('SFR vs. Halo Mass: FFB vs. No-FFB Comparison (1, 2, 3σ contours)', 
    #              fontsize=16, y=0.995)
    
    plt.tight_layout()

    # Hide x-axis labels and tick values for top row (first 3 axes)
    for ax in axes[:3]:
        ax.set_xlabel("")
        ax.set_xticklabels([""] * len(ax.get_xticks()))

    output_file = OutputDir + 'sfr_mvir_contours_grid' + OutputFormat
    plt.savefig(output_file, bbox_inches='tight', dpi=150)
    print(f"\nSaved: {output_file}")
    plt.close()

# ==================================================================

if __name__ == "__main__":
    # plot_ffb_comparison_grid()
    # plot_ffb_comparison_grid_shmr()
    # plot_ffb_comparison_grid_6panel()
    plot_sfr_mvir_contours_grid()
    # plot_density_evolution()
    # plot_ffb_comparison_4panel()
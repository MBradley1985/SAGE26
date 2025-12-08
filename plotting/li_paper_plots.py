#!/usr/bin/env python

import h5py as h5
import numpy as np
from numpy.random import seed
import matplotlib.pyplot as plt
import os
import warnings
import sys
from astropy.table import Table
from astropy import units as u
from scipy.interpolate import Akima1DInterpolator

# Import analytical functions from ffb_predict
import sys
sys.path.insert(0, os.path.dirname(__file__))
from ffb_predict import (
    # Cosmology setup
    cosmo, fb,
    # Mass functions and densities
    compute_dNdlgMs, compute_dNdMUV_Ms, 
    compute_rho_star, compute_rho_SFR, compute_rho_UV,
    compute_surface_density_obs,
    # FFB functions
    ffb_lgMh_crit, func_lgMs_med, func_MUV_lgMs,
    ffb_radius, func_SFE_instant, ffb_fgas, ffb_Mgas,
    # Halo mass function
    compute_dNdlgMh,
    # Set options
    set_option, get_option
)

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# ========================== UV MAGNITUDE CONVERSION ==========================

def func_MUV_sfr(SFR, z):
    """
    Convert stellar mass to UV absolute magnitude using:
    median{M_UV|M_s} = -2.3 * log10(M_s/1e9 Msun) - 20.5
    
    Parameters:
    -----------
    SFR : array-like or float
        Star formation rate in Msun/yr (ignored, kept for compatibility)
    z : float
        Redshift (ignored, kept for compatibility)
    
    Returns:
    --------
    MUV : array-like or float
        Absolute UV magnitude at 1500 Angstrom
    """
    # This function now expects SFR to be the stellar mass (Msun)
    Ms = np.asarray(SFR)  # SFR argument is now interpreted as stellar mass
    MUV = -2.3 * np.log10(Ms / 1e9) - 20.5
    return MUV

# ========================== FFB METALLICITY ANALYTICAL FUNCTIONS ==========================

def func_SFE_instant(lgMh, z, eps_max=None):
    """Instantaneous star formation efficiency (SFR / Mdot_baryon)
    
    If eps_max is provided, scales to that maximum value.
    Otherwise uses default FFB maximum of 1.0
    """
    if eps_max is None:
        eps_max = 1.0
    
    lgMh_crit = 10.8 - 6.2 * np.log10((1 + z) / 10)
    # Below threshold: low efficiency, above threshold: approach eps_max
    # Using sigmoid centered at threshold with smooth transition
    transition_width = 0.5  # in dex
    eps = eps_max * (1 / (1 + np.exp(-(lgMh - lgMh_crit) / transition_width)))
    return eps.clip(0.01, eps_max)

def ffb_str_coverage(lgMh, z, eps=None, eps_max=None):
    """Sky coverage fraction by cold streams (from ffb_predict.py)"""
    if eps is None:
        eps = func_SFE_instant(lgMh, z, eps_max=eps_max)
    eta = 5 / eps - 4
    
    Mz_dep = 10 ** ((lgMh - 10.8) * 0.3333) * ((1 + z) / 10) ** 0.5
    f_omega = 0.22 * eta**-0.5 * eps**-1 * Mz_dep
    return np.clip(f_omega, 0, 1)

def ffb_metal_analytical(lgMh, z, Zsn=1, Zin=0.1, eps=None, eps_max=None):
    """Analytical metallicity from ffb_predict.py mixing formula"""
    if eps is None:
        eps = func_SFE_instant(lgMh, z, eps_max=eps_max)
    f_omega = ffb_str_coverage(lgMh, z, eps=eps, eps_max=eps_max)
    
    Zmix = Zin + 0.2 * eps * f_omega / (1 + (1 - 0.8 * eps) * f_omega) * (Zsn - Zin)
    return Zmix

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

def plot_rvir_vs_redshift():
    """
    Plot theoretical Rvir vs redshift for several Mvir values using the formula:
    Rvir = 12.3 kpc * (Mvir/10^10.8 Msun)^(1/3) * ((1+z)/10)^(-1)
    and compare with SAGE Rvir vs redshift for all galaxies.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Theoretical lines for several Mvir values
    mvir_vals = [1e11, 1e12, 1e13]  # Msun
    z_arr = np.linspace(0, 16, 100)
    rvir_theory = []
    for mvir in mvir_vals:
        m10_8 = mvir / 10**10.8
        rvir = 12.3 * m10_8**(1/3) * ((1+z_arr)/10)**(-1)  # kpc
        rvir_theory.append(rvir)

    # SAGE data: Rvir vs redshift for all galaxies
    # We'll use the same snapshots/redshifts as other plots
    snapshots = []
    actual_redshifts = []
    for idx, z in enumerate(DEFAULT_REDSHIFTS):
        if 0 <= z <= 16.5:
            snapshots.append(f'Snap_{idx}')
            actual_redshifts.append(z)

    model_dir = './output/millennium/'
    filename = 'model_0.hdf5'
    hubble_h = MILLENNIUM_HUBBLE_H

    sage_rvir = []
    sage_z = []
    for snapshot, z_actual in zip(snapshots, actual_redshifts):
        try:
            rvir = read_hdf_from_model(model_dir, filename, snapshot, 'Rvir', hubble_h)  # in kpc/h
            if rvir is None or np.size(rvir) == 0:
                print(f"Warning: No Rvir data for {snapshot} at z={z_actual}")
                continue
            # Check if Rvir is in Mpc/h (values < 100 likely Mpc)
            if np.nanmax(rvir) < 100:
                rvir = rvir * 1000  # Convert Mpc/h to kpc/h
                print(f"Info: Converted Rvir from Mpc/h to kpc/h for {snapshot} at z={z_actual}")
            # Convert to comoving kpc
            rvir_comov = rvir / hubble_h
            valid = np.isfinite(rvir_comov) & (rvir_comov > 0)
            if np.sum(valid) == 0:
                print(f"Warning: No valid Rvir values for {snapshot} at z={z_actual}")
                continue
            # Print sample values for inspection
            print(f"Sample Rvir_phys for {snapshot} at z={z_actual}: {rvir_comov[valid][:5]}")
            # Multiply by (1+z) to get comoving Rvir
            rvir_comov = rvir_comov[valid]
            sage_rvir.extend(rvir_comov)
            sage_z.extend([z_actual] * np.sum(valid))
        except Exception as e:
            print(f"Error reading Rvir for {snapshot} at z={z_actual}: {e}")
            continue

    # Plot
    plt.figure(figsize=(8, 6))
    for i, mvir in enumerate(mvir_vals):
        plt.plot(z_arr, rvir_theory[i], label=f'Theory: $M_{{vir}}={mvir:.0e}$ $M_\odot$', lw=2)
    if len(sage_rvir) > 0:
        # Randomly sample up to 5000 galaxies for plotting
        Nplot = min(5000, len(sage_rvir))
        idx = np.random.choice(len(sage_rvir), Nplot, replace=False)
        plt.scatter(np.array(sage_z)[idx], np.array(sage_rvir)[idx], s=8, color='blue', alpha=0.6, label=f'SAGE Galaxies (N={Nplot})')
        print(f"Plotted {Nplot} SAGE galaxy Rvir points (random sample).")
    else:
        print("Warning: No SAGE galaxy Rvir data found for plotting.")
    plt.xlabel('Redshift $z$')
    plt.ylabel(r'$R_{\mathrm{vir}}$ (comoving kpc)')
    plt.yscale('log')
    plt.xlim(0, 16)
    plt.ylim(1, 500)
    plt.legend()
    plt.title('Virial Radius vs Redshift')
    plt.tight_layout()
    OutputDir = DirName + 'plots/'
    if not os.path.exists(OutputDir):
        os.makedirs(OutputDir)
    output_path = OutputDir + 'rvir_vs_redshift' + OutputFormat
    plt.savefig(output_path, dpi=300)
    print(f'Plot saved to: {output_path}')
    plt.close()
def plot_radius_evolution_all_galaxies():
    """
    Plot the evolution of disk and bulge half-mass radii for all galaxies (no FFB threshold cut).
    Replicates the right panel of the FFB threshold analysis, but for the full galaxy population.
    """
    print('\n' + '='*60)
    print('Creating Radius Evolution Plot for All Galaxies')
    print('='*60)

    # Use only FFB 50% model (as in threshold analysis)
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

    snapshots = []
    actual_redshifts = []
    for idx, z in enumerate(DEFAULT_REDSHIFTS):
        if 4.5 <= z <= 16.5:
            snapshots.append(f'Snap_{idx}')
            actual_redshifts.append(z)

    model_dir = ffb50_model['dir']
    filename = 'model_0.hdf5'
    hubble_h = ffb50_model['hubble_h']

    if not os.path.exists(model_dir + filename):
        print(f"Error: {model_dir + filename} not found!")
        print("This plot requires FFB 50% model output.")
        return

    redshifts = []
    median_disk_radius = []
    disk_radius_low = []
    disk_radius_high = []
    median_bulge_radius = []
    bulge_radius_low = []
    bulge_radius_high = []

    for snapshot, z_actual in zip(snapshots, actual_redshifts):
        mvir = read_hdf_from_model(model_dir, filename, snapshot, 'Mvir', hubble_h) * 1.0e10 / hubble_h
        stellar_mass = read_hdf_from_model(model_dir, filename, snapshot, 'StellarMass', hubble_h) * 1.0e10 / hubble_h
        bulge_mass = read_hdf_from_model(model_dir, filename, snapshot, 'BulgeMass', hubble_h) * 1.0e10 / hubble_h
        disk_radius = read_hdf_from_model(model_dir, filename, snapshot, 'DiskRadius', hubble_h) * (1+z_actual)
        bulge_radius_raw = read_hdf_from_model(model_dir, filename, snapshot, 'BulgeRadius', hubble_h) * (1+z_actual)

        if len(bulge_radius_raw) != len(mvir):
            bulge_radius = np.zeros_like(disk_radius)
        else:
            bulge_radius = bulge_radius_raw

        has_stellar_mass = stellar_mass > 0
        w = np.where(has_stellar_mass)[0]

        if len(w) > 5:
            redshifts.append(z_actual)
            # Disk: where disk_radius > 0
            valid_disk = disk_radius[w] > 0
            if np.sum(valid_disk) > 0:
                disk_radius_kpc = disk_radius[w[valid_disk]] * 1.68 * 1000.0 / hubble_h / (1.0 + z_actual)
                median_disk_radius.append(np.median(disk_radius_kpc))
                disk_radius_low.append(np.percentile(disk_radius_kpc, 16))
                disk_radius_high.append(np.percentile(disk_radius_kpc, 84))
            else:
                median_disk_radius.append(np.nan)
                disk_radius_low.append(np.nan)
                disk_radius_high.append(np.nan)

            # Bulge: where bulge_radius > 0 and bulge_mass > 0
            valid_bulge = (bulge_radius[w] > 0) & (bulge_mass[w] > 0)
            if np.sum(valid_bulge) > 0:
                bulge_radius_kpc = bulge_radius[w[valid_bulge]] * 1000.0 / hubble_h / (1.0 + z_actual)
                median_bulge_radius.append(np.median(bulge_radius_kpc))
                bulge_radius_low.append(np.percentile(bulge_radius_kpc, 16))
                bulge_radius_high.append(np.percentile(bulge_radius_kpc, 84))
            else:
                median_bulge_radius.append(np.nan)
                bulge_radius_low.append(np.nan)
                bulge_radius_high.append(np.nan)

    # Plotting
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))

    if len(redshifts) > 0:
        # Disk
        ax.plot(redshifts, median_disk_radius, color='orange', linestyle='-', linewidth=3, label='Disk Half-Mass Radius', alpha=0.8, marker='o', markersize=6, zorder=3)
        ax.fill_between(redshifts, disk_radius_low, disk_radius_high, color='orange', alpha=0.2, zorder=2)
        # Bulge
        bulge_mask = np.array([r > 0 for r in median_bulge_radius])
        if np.any(bulge_mask):
            redshifts_bulge = np.array(redshifts)[bulge_mask]
            median_bulge_valid = np.array(median_bulge_radius)[bulge_mask]
            bulge_low_valid = np.array(bulge_radius_low)[bulge_mask]
            bulge_high_valid = np.array(bulge_radius_high)[bulge_mask]
            ax.plot(redshifts_bulge, median_bulge_valid, color='royalblue', linestyle='-', linewidth=3, label='Bulge Half-Mass Radius', alpha=0.8, marker='s', markersize=6, zorder=3)
            ax.fill_between(redshifts_bulge, bulge_low_valid, bulge_high_valid, color='royalblue', alpha=0.2, zorder=2)

        ax.set_xlabel('Redshift z', fontsize=12)
        ax.set_ylabel(r'$R_{\mathrm{half-mass}}$ (kpc)', fontsize=12)
        ax.set_xlim(5, 16)
        ax.set_yscale('log')
        ax.set_ylim(0.01, 3)
        ax.set_yticks([0.01, 0.1, 1.0])
        ax.set_yticklabels(['0.01', '0.1', '1.0'])
        from matplotlib.ticker import MultipleLocator
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.2))
        ax.legend(loc='upper right', fontsize=10, frameon=False)
        ax.text(0.05, 0.95, 'All Galaxies', transform=ax.transAxes, verticalalignment='top', fontsize=11)
        ax.grid(True, alpha=0.1)

    plt.tight_layout()
    OutputDir = DirName + 'plots/'
    if not os.path.exists(OutputDir):
        os.makedirs(OutputDir)
    output_path = OutputDir + 'radius_evolution_all_galaxies' + OutputFormat
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'\nPlot saved to: {output_path}')
    plt.close()
    print('='*60 + '\n')

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
    """Load COSMOS2020 (Weaver+) observational SMF data for a given redshift"""
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
    elif redshift < 6.5:
        filename = './data/COSMOS2020/SMF_Farmer_v2.1_5.5z6.5_total.txt'
        z_label = 'COSMOS 5.5<z<6.5'
    elif redshift < 7.5:
        filename = './data/COSMOS2020/SMF_Farmer_v2.1_6.5z7.5_total.txt'
        z_label = 'COSMOS 6.5<z<7.5'
    else:
        return None, None, None, None
    
    if not os.path.exists(filename):
        return None, None, None, None
    
    # Load data: columns are log10(M*), bin_width, phi, phi_lower_bound, phi_upper_bound
    # Columns 4 and 5 are the actual lower/upper bound VALUES in linear scale
    data = np.loadtxt(filename)
    log_mass = data[:, 0]
    phi_linear = data[:, 2]  # SMF value in linear scale
    phi_lower_bound = data[:, 3]  # Lower bound VALUE in linear scale (can be negative)
    phi_upper_bound = data[:, 4]  # Upper bound VALUE in linear scale
    
    # Convert values to log scale
    phi = np.log10(phi_linear)
    phi_upper = np.log10(phi_upper_bound)
    
    # Handle negative/zero lower bounds by setting to NaN
    phi_lower = np.full_like(phi, np.nan)
    valid_lower = phi_lower_bound > 0
    phi_lower[valid_lower] = np.log10(phi_lower_bound[valid_lower])
    
    return log_mass, phi, phi_lower, phi_upper

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
        
        # Columns: phi_error_low and phi_error_upp are the LOWER and UPPER BOUND values (not deltas!)
        log_mass = np.array(data['log10Mstar'])
        phi_linear = np.array(data['phi'])
        phi_lower_linear = np.array(data['phi_error_low'])  # Lower bound value
        phi_upper_linear = np.array(data['phi_error_upp'])  # Upper bound value
        
        # Convert ALL values to log scale
        phi = np.log10(phi_linear)
        phi_lower = np.log10(phi_lower_linear)
        phi_upper = np.log10(phi_upper_linear)
        
        return log_mass, phi, phi_lower, phi_upper
    except Exception as e:
        print(f"  Warning: Could not load Bagpipes data: {e}")
        return None, None, None, None

def load_stefanon_smf(redshift):
    """Load Stefanon+2021 observational SMF data for a given redshift
    
    The phi values in the file are in units of 1e-4 Mpc-3 dex-1,
    so they are multiplied by 10^-4 before converting to log scale.
    """
    filename = './data/stefanon_smf_2021.ecsv'
    
    if not os.path.exists(filename):
        return None, None, None, None
    
    try:
        # Read the ECSV file
        table = Table.read(filename, format='ascii.ecsv')
        
        # Match redshift bin (bins are labeled 6, 7, 8, 9, 10)
        # Map continuous redshift to appropriate bin
        if redshift < 6.5:
            z_bin = 6
        elif redshift < 7.5:
            z_bin = 7
        elif redshift < 8.5:
            z_bin = 8
        elif redshift < 9.5:
            z_bin = 9
        elif redshift < 11:
            z_bin = 10
        else:
            return None, None, None, None
        
        # Filter for matching redshift bin
        z_match = table['redshift_bin'] == z_bin
        if not np.any(z_match):
            return None, None, None, None
        
        data = table[z_match]
        
        # Extract data - phi values are in units of 1e-4 Mpc-3 dex-1
        log_mass = np.array(data['log_M'])
        phi_linear = np.array(data['phi']) * 1e-4  # Multiply by 10^-4
        phi_err_up_linear = np.array(data['phi_err_up']) * 1e-4
        phi_err_low_linear = np.array(data['phi_err_low']) * 1e-4
        
        # Calculate upper and lower bounds
        phi_upper_linear = phi_linear + phi_err_up_linear
        phi_lower_linear = phi_linear - phi_err_low_linear
        
        # Convert to log scale, handling negative/zero values
        phi = np.log10(phi_linear)
        phi_upper = np.log10(phi_upper_linear)
        
        phi_lower = np.full_like(phi, np.nan)
        valid_lower = phi_lower_linear > 0
        phi_lower[valid_lower] = np.log10(phi_lower_linear[valid_lower])
        
        return log_mass, phi, phi_lower, phi_upper
    except Exception as e:
        print(f"  Warning: Could not load Stefanon data: {e}")
        return None, None, None, None

def load_navarro_carrera_smf(redshift):
    """Load Navarro-Carrera+2023 observational SMF data for a given redshift
    
    The phi values in the file are in units of 1e-4 Mpc-3 dex-1,
    so they are multiplied by 10^-4 before converting to log scale.
    """
    filename = './data/navarro_carrera_smf_2023.ecsv'
    
    if not os.path.exists(filename):
        return None, None, None, None
    
    try:
        # Read the ECSV file
        table = Table.read(filename, format='ascii.ecsv')
        
        # Match redshift bin (bins are labeled 6, 7, 8)
        # Map continuous redshift to appropriate bin
        if redshift < 6.5:
            z_bin = 6
        elif redshift < 7.5:
            z_bin = 7
        elif redshift < 8.5:
            z_bin = 8
        else:
            return None, None, None, None
        
        # Filter for matching redshift bin
        z_match = table['redshift_bin'] == z_bin
        if not np.any(z_match):
            return None, None, None, None
        
        data = table[z_match]
        
        # Extract data - phi values are in units of 1e-4 Mpc-3 dex-1
        log_mass = np.array(data['log_M'])
        phi_linear = np.array(data['phi']) * 1e-4  # Multiply by 10^-4
        phi_err_up_linear = np.array(data['phi_err_up']) * 1e-4
        phi_err_low_linear = np.array(data['phi_err_low']) * 1e-4
        
        # Calculate upper and lower bounds
        phi_upper_linear = phi_linear + phi_err_up_linear
        phi_lower_linear = phi_linear - phi_err_low_linear
        
        # Convert to log scale, handling negative/zero values
        phi = np.log10(phi_linear)
        phi_upper = np.log10(phi_upper_linear)
        
        phi_lower = np.full_like(phi, np.nan)
        valid_lower = phi_lower_linear > 0
        phi_lower[valid_lower] = np.log10(phi_lower_linear[valid_lower])
        
        return log_mass, phi, phi_lower, phi_upper
    except Exception as e:
        print(f"  Warning: Could not load Navarro-Carrera data: {e}")
        return None, None, None, None

def load_weibel_smf(redshift):
    """Load Weibel+2024 observational SMF data for a given redshift
    
    The data is already in log scale, no conversion needed.
    """
    filename = './data/weibel_smf_2024.ecsv'
    
    if not os.path.exists(filename):
        return None, None, None, None
    
    try:
        # Read the ECSV file
        table = Table.read(filename, format='ascii.ecsv')
        
        # Match redshift bin (bins are labeled 6, 7, 8, 9)
        # Map continuous redshift to appropriate bin
        if redshift < 6.5:
            z_bin = 6
        elif redshift < 7.5:
            z_bin = 7
        elif redshift < 8.5:
            z_bin = 8
        elif redshift < 9.5:
            z_bin = 9
        else:
            return None, None, None, None
        
        # Filter for matching redshift bin
        z_match = table['redshift_bin'] == z_bin
        if not np.any(z_match):
            return None, None, None, None
        
        data = table[z_match]
        
        # Extract data - already in log scale
        log_mass = np.array(data['log_M'])
        log_phi = np.array(data['log_phi'])
        log_phi_err_up = np.array(data['log_phi_err_up'])
        log_phi_err_low = np.array(data['log_phi_err_low'])
        
        # Calculate upper and lower bounds
        log_phi_upper = log_phi + log_phi_err_up
        log_phi_lower = log_phi - log_phi_err_low
        
        # Handle zero errors (converted from infinities) by setting to NaN
        log_phi_lower[log_phi_err_low == 0] = np.nan
        
        return log_mass, log_phi, log_phi_lower, log_phi_upper
    except Exception as e:
        print(f"  Warning: Could not load Weibel data: {e}")
        return None, None, None, None

def load_kikuchihara_smf(redshift):
    """Load Kikuchihara+2020 observational SMF data for a given redshift
    
    This dataset contains Schechter function parameters. Returns the raw
    parameters as a single point at M_star. The phi_star values are in units of
    1e-5 Mpc-3 dex-1, so they are multiplied by 10^-5.
    """
    filename = './data/kikuchihara_smf_2020.ecsv'
    
    if not os.path.exists(filename):
        return None, None, None, None
    
    try:
        # Read the ECSV file
        table = Table.read(filename, format='ascii.ecsv')
        
        # Match redshift (approximate bins are 6, 7, 8, 9)
        if redshift < 6.5:
            z_bin = 6
        elif redshift < 7.5:
            z_bin = 7
        elif redshift < 8.5:
            z_bin = 8
        elif redshift < 9.5:
            z_bin = 9
        else:
            return None, None, None, None
        
        # Filter for matching redshift
        z_match = table['redshift_approx'] == z_bin
        if not np.any(z_match):
            return None, None, None, None
        
        data = table[z_match]
        
        # Get all available data points for this redshift
        log_mass = np.array(data['log_M_star'])
        phi_star = np.array(data['phi_star']) * 1e-5  # Multiply by 10^-5
        phi_star_err_up = np.array(data['phi_star_err_up']) * 1e-5
        phi_star_err_low = np.array(data['phi_star_err_low']) * 1e-5
        
        # Convert to log scale
        log_phi = np.log10(phi_star)
        
        # Calculate upper and lower bounds
        phi_upper_linear = phi_star + phi_star_err_up
        phi_lower_linear = phi_star - phi_star_err_low
        
        log_phi_upper = np.log10(phi_upper_linear)
        
        log_phi_lower = np.full_like(log_phi, np.nan)
        valid_lower = phi_lower_linear > 0
        log_phi_lower[valid_lower] = np.log10(phi_lower_linear[valid_lower])
        
        return log_mass, log_phi, log_phi_lower, log_phi_upper
    except Exception as e:
        print(f"  Warning: Could not load Kikuchihara data: {e}")
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
        
        # Add analytical predictions from Li+2023 (ffb_predict.py)
        try:
            lgMs_analytical = np.linspace(8, 12, 100)
            
            # FFB model with eps_max=1.0 (default, upper bound)
            set_option(FFB_SFE_MAX=1.0)
            lgMs_bins, dNdlgMs_ffb1 = compute_dNdlgMs(z_actual, lgMs=lgMs_analytical)
            log_phi_ffb1 = np.log10(dNdlgMs_ffb1)
            valid = np.isfinite(log_phi_ffb1) & (log_phi_ffb1 > -9)
            ax.plot(lgMs_bins[valid], log_phi_ffb1[valid],
                   color='dodgerblue', linestyle='--', linewidth=2,
                   label='Li+ 2024' if idx == 0 else '',
                   alpha=0.7, zorder=4)
            
            # FFB model with eps_max=0.2 (lower bound)
            set_option(FFB_SFE_MAX=0.2)
            lgMs_bins, dNdlgMs_ffb02 = compute_dNdlgMs(z_actual, lgMs=lgMs_analytical)
            log_phi_ffb02 = np.log10(dNdlgMs_ffb02)
            valid = np.isfinite(log_phi_ffb02) & (log_phi_ffb02 > -9)
            ax.plot(lgMs_bins[valid], log_phi_ffb02[valid],
                   color='orange', linestyle='--', linewidth=2,
                   label='' if idx == 0 else '',
                   alpha=0.7, zorder=4)
            
            # Universe Machine model (no FFB)
            set_option(FFB_SFE_MAX=0.0)  # Disable FFB
            lgMs_bins, dNdlgMs_um = compute_dNdlgMs(z_actual, lgMs=lgMs_analytical)
            log_phi_um = np.log10(dNdlgMs_um)
            valid = np.isfinite(log_phi_um) & (log_phi_um > -9)
            ax.plot(lgMs_bins[valid], log_phi_um[valid],
                   color='gray', linestyle='--', linewidth=2,
                   label='' if idx == 0 else '',
                   alpha=0.7, zorder=4)
            
            # Reset to default
            set_option(FFB_SFE_MAX=1.0)
            print(f"  Li+2023 analytical predictions added (3 lines)")
        except Exception as e:
            print(f"  Warning: Could not compute analytical SMF: {e}")
        
        # Add observational data
        # COSMOS2020 (Weaver+)
        cosmos_mass, cosmos_phi, cosmos_phi_lower, cosmos_phi_upper = load_cosmos2020_smf(z_actual)
        if cosmos_mass is not None:
            # Filter out invalid values
            valid = np.isfinite(cosmos_phi) & (cosmos_phi > -9) & np.isfinite(cosmos_phi_upper)
            if np.any(valid):
                # Calculate error bar magnitudes from actual bound values
                # For lower errors, use phi_lower if available, otherwise use large value to show only upper cap
                yerr_low = np.where(np.isfinite(cosmos_phi_lower[valid]), 
                                   cosmos_phi[valid] - cosmos_phi_lower[valid],
                                   0)  # No lower error bar when bound is invalid
                yerr_high = cosmos_phi_upper[valid] - cosmos_phi[valid]
                ax.errorbar(cosmos_mass[valid], cosmos_phi[valid], 
                           yerr=[yerr_low, yerr_high],
                           fmt='s', color='black', markersize=8, alpha=1.0,
                           label='Weaver+23' if idx == 0 else '', capsize=2, linewidth=1.5)
                print(f"  Weaver+23(COSMOS2020) data added")
        
        # Bagpipes (Harvey+24) - for high redshifts
        if z_actual >= 6.0:
            bagpipes_mass, bagpipes_phi, bagpipes_phi_lower, bagpipes_phi_upper = load_bagpipes_smf(z_actual)
            if bagpipes_mass is not None:
                valid = np.isfinite(bagpipes_phi) & (bagpipes_phi > -9) & np.isfinite(bagpipes_phi_lower) & np.isfinite(bagpipes_phi_upper)
                if np.any(valid):
                    # Calculate error bar magnitudes from actual bound values, set negative errors to zero
                    yerr_low = np.maximum(0, bagpipes_phi[valid] - bagpipes_phi_lower[valid])
                    yerr_high = np.maximum(0, bagpipes_phi_upper[valid] - bagpipes_phi[valid])
                    ax.errorbar(bagpipes_mass[valid], bagpipes_phi[valid],
                               yerr=[yerr_low, yerr_high],
                               fmt='D', color='black', markersize=8, alpha=1.0,
                               label='Harvey+24' if idx == 0 else '', capsize=2, linewidth=1.5)
                    print(f"  Harvey+24 (Bagpipes) data added")
        
        # Stefanon+2021 - for high redshifts (z=6-10)
        if z_actual >= 6.0:
            stefanon_mass, stefanon_phi, stefanon_phi_lower, stefanon_phi_upper = load_stefanon_smf(z_actual)
            if stefanon_mass is not None:
                valid = np.isfinite(stefanon_phi) & (stefanon_phi > -9) & np.isfinite(stefanon_phi_upper)
                if np.any(valid):
                    # Calculate error bar magnitudes from actual bound values
                    yerr_low = np.where(np.isfinite(stefanon_phi_lower[valid]), 
                                       np.maximum(0, stefanon_phi[valid] - stefanon_phi_lower[valid]),
                                       0)  # No lower error bar when bound is invalid
                    yerr_high = np.maximum(0, stefanon_phi_upper[valid] - stefanon_phi[valid])
                    ax.errorbar(stefanon_mass[valid], stefanon_phi[valid],
                               yerr=[yerr_low, yerr_high],
                               fmt='o', color='black', markersize=8, alpha=0.8,
                               label='Stefanon+21' if idx == 0 else '', capsize=2, linewidth=1.5)
                    print(f"  Stefanon+2021 data added")
        
        # Navarro-Carrera+2023 - for high redshifts (z=6-8)
        if z_actual >= 6.0 and z_actual <= 8.5:
            nc_mass, nc_phi, nc_phi_lower, nc_phi_upper = load_navarro_carrera_smf(z_actual)
            if nc_mass is not None:
                valid = np.isfinite(nc_phi) & (nc_phi > -9) & np.isfinite(nc_phi_upper)
                if np.any(valid):
                    # Calculate error bar magnitudes from actual bound values
                    yerr_low = np.where(np.isfinite(nc_phi_lower[valid]), 
                                       np.maximum(0, nc_phi[valid] - nc_phi_lower[valid]),
                                       0)  # No lower error bar when bound is invalid
                    yerr_high = np.maximum(0, nc_phi_upper[valid] - nc_phi[valid])
                    ax.errorbar(nc_mass[valid], nc_phi[valid],
                               yerr=[yerr_low, yerr_high],
                               fmt='^', color='black', markersize=8, alpha=0.8,
                               label='Navarro-Carrera+23' if idx == 0 else '', capsize=2, linewidth=1.5)
                    print(f"  Navarro-Carrera+2023 data added")
        
        # Weibel+2024 - for high redshifts (z=6-9)
        if z_actual >= 6.0 and z_actual <= 9.5:
            weibel_mass, weibel_phi, weibel_phi_lower, weibel_phi_upper = load_weibel_smf(z_actual)
            if weibel_mass is not None:
                valid = np.isfinite(weibel_phi) & (weibel_phi > -9) & np.isfinite(weibel_phi_upper)
                if np.any(valid):
                    # Calculate error bar magnitudes from actual bound values
                    yerr_low = np.where(np.isfinite(weibel_phi_lower[valid]), 
                                       np.maximum(0, weibel_phi[valid] - weibel_phi_lower[valid]),
                                       0)  # No lower error bar when bound is invalid
                    yerr_high = np.maximum(0, weibel_phi_upper[valid] - weibel_phi[valid])
                    ax.errorbar(weibel_mass[valid], weibel_phi[valid],
                               yerr=[yerr_low, yerr_high],
                               fmt='v', color='black', markersize=8, alpha=0.8,
                               label='Weibel+24' if idx == 0 else '', capsize=2, linewidth=1.5)
                    print(f"  Weibel+2024 data added")
        
        # Kikuchihara+2020 - Schechter function fits (z=6-9)
        if z_actual >= 6.0 and z_actual <= 9.5:
            kiku_mass, kiku_phi, kiku_phi_lower, kiku_phi_upper = load_kikuchihara_smf(z_actual)
            if kiku_mass is not None:
                valid = np.isfinite(kiku_phi) & (kiku_phi > -9)
                if np.any(valid):
                    # Plot as markers
                    ax.plot(kiku_mass[valid], kiku_phi[valid],
                           marker='<', color='black', linestyle='', markersize=8, alpha=0.8,
                           label='Kikuchihara+20' if idx == 0 else '')
                    print(f"  Kikuchihara+2020 data added")
        
        # Formatting
        ax.set_xlim(8, 12)
        ax.set_ylim(-7, -1)
        # Set integer ticks with minor ticks at 20% intervals
        from matplotlib.ticker import MultipleLocator
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.2))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.2))
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

def load_song2016_cumulative(mass_threshold):
    """Load Song+2016 SMF data and compute cumulative number density above mass threshold
    
    Args:
        mass_threshold: Mass threshold in solar masses (e.g., 1e9 or 1e10)
    
    Returns:
        redshifts: Array of redshifts
        n_cumulative: Array of log10(n) in Mpc^-3
        n_cumulative_lower: Array of log10(n_lower) for error bars
        n_cumulative_upper: Array of log10(n_upper) for error bars
    """
    filename = './data/song_smf_2016.ecsv'
    
    if not os.path.exists(filename):
        print(f"  Warning: {filename} not found")
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    try:
        table = Table.read(filename, format='ascii.ecsv')
        
        log_mass_threshold = np.log10(mass_threshold)
        
        # Redshifts available in the data
        redshifts_available = [4, 5, 6, 7, 8]
        
        redshifts = []
        n_cumulative = []
        n_cumulative_lower = []
        n_cumulative_upper = []
        
        for z in redshifts_available:
            phi_col = f'phi_z{z}'
            phi_err_up_col = f'phi_z{z}_err_up'
            phi_err_lo_col = f'phi_z{z}_err_lo'
            
            # Get data for this redshift
            log_mass = table['log_M']
            phi_log = table[phi_col]  # This is already log(phi)
            phi_err_up = table[phi_err_up_col]
            phi_err_lo = table[phi_err_lo_col]
            
            # Filter for masses above threshold and valid phi values
            mask = (log_mass >= log_mass_threshold) & np.isfinite(phi_log)
            
            if np.sum(mask) == 0:
                continue
            
            # The data is in log space: log(phi) in units of log(dex^-1 Mpc^-3)
            # Convert to linear scale for integration
            log_mass_above = log_mass[mask]
            phi_linear = 10**phi_log[mask]  # Convert log(phi) to phi in Mpc^-3 dex^-1
            phi_linear_upper = 10**(phi_log[mask] + phi_err_up[mask])
            phi_linear_lower = 10**(phi_log[mask] - phi_err_lo[mask])
            
            # Integrate phi over d(log M) to get cumulative number density above threshold
            # n(>M_cut) = integral of phi(M) d(log M) = Σ phi(M_i) × Δ(log M_i)
            n_total = np.trapz(phi_linear, log_mass_above)
            n_total_upper = np.trapz(phi_linear_upper, log_mass_above)
            n_total_lower = np.trapz(phi_linear_lower, log_mass_above)
            
            # Only include if we have positive values
            if n_total > 0 and n_total_lower > 0 and n_total_upper > 0:
                redshifts.append(z)
                n_cumulative.append(np.log10(n_total))
                n_cumulative_lower.append(np.log10(n_total_lower))
                n_cumulative_upper.append(np.log10(n_total_upper))
        
        return np.array(redshifts), np.array(n_cumulative), np.array(n_cumulative_lower), np.array(n_cumulative_upper)
        
    except Exception as e:
        print(f"  Warning: Could not load Song+2016 data: {e}")
        return np.array([]), np.array([]), np.array([]), np.array([])

def load_stefanon2021_cumulative(mass_threshold):
    """Load Stefanon+2021 SMF evolution data and extract cumulative densities above mass threshold
    
    Args:
        mass_threshold: Mass threshold in solar masses (e.g., 1e9 or 1e10)
    
    Returns:
        redshifts: Array of redshifts
        n_cumulative: Array of log10(n) in Mpc^-3
        n_cumulative_lower: Array of log10(n_lower) for error bars
        n_cumulative_upper: Array of log10(n_upper) for error bars
    """
    filename = './data/stefanon_smfevol_2021.ecsv'
    
    if not os.path.exists(filename):
        print(f"  Warning: {filename} not found")
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    try:
        table = Table.read(filename, format='ascii.ecsv')
        
        log_mass_threshold = np.log10(mass_threshold)
        
        # Filter for points above mass threshold (include both data and limits)
        mask = table['log_M_star'] >= log_mass_threshold
        
        if np.sum(mask) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        # Extract the data directly from the table
        redshifts = np.array(table['redshift'][mask])
        n_cumulative = np.array(table['log_cum_den'][mask])
        n_cumulative_lower = np.array(table['log_cum_den'][mask] - table['log_cum_den_err_low'][mask])
        n_cumulative_upper = np.array(table['log_cum_den'][mask] + table['log_cum_den_err_up'][mask])
        
        return redshifts, n_cumulative, n_cumulative_lower, n_cumulative_upper
        
    except Exception as e:
        print(f"  Warning: Could not load Stefanon+2021 data: {e}")
        return np.array([]), np.array([]), np.array([]), np.array([])

def load_cosmos2020_cumulative(mass_threshold):
    """Load COSMOS2020 data and compute cumulative number density above mass threshold
    
    Args:
        mass_threshold: Mass threshold in solar masses (e.g., 1e9 or 1e10)
    
    Returns:
        redshifts: Array of mean redshifts for each bin
        n_cumulative: Array of log10(n) in Mpc^-3
        n_cumulative_lower: Array of log10(n_lower) for error bars
        n_cumulative_upper: Array of log10(n_upper) for error bars
    """
    # Define redshift bins and their mean values
    redshift_bins = [
        (0.2, 0.5, 0.35),
        (0.5, 0.8, 0.65),
        (0.8, 1.1, 0.95),
        (1.1, 1.5, 1.3),
        (1.5, 2.0, 1.75),
        (2.0, 2.5, 2.25),
        (2.5, 3.0, 2.75),
        (3.0, 3.5, 3.25),
        (3.5, 4.5, 4.0),
        (4.5, 5.5, 5.0),
        (5.5, 6.5, 6.0),
        (6.5, 7.5, 7.0)
    ]
    
    data_dir = './data/COSMOS2020/'
    log_mass_threshold = np.log10(mass_threshold)
    
    redshifts = []
    n_cumulative = []
    n_cumulative_lower = []
    n_cumulative_upper = []
    
    for z_min, z_max, z_mean in redshift_bins:
        # Construct filename
        filename = f'{data_dir}SMF_Farmer_v2.1_{z_min}z{z_max}_total.txt'
        
        if not os.path.exists(filename):
            continue
        
        # Load data: columns are log10(M*), bin_width, phi, phi_lower_bound, phi_upper_bound
        data = np.loadtxt(filename)
        log_mass = data[:, 0]
        phi_linear = data[:, 2]  # SMF value in linear scale (Mpc^-3 dex^-1)
        phi_lower_bound = data[:, 3]  # Lower bound VALUE in linear scale
        phi_upper_bound = data[:, 4]  # Upper bound VALUE in linear scale
        
        # Find bins above mass threshold
        mask = log_mass >= log_mass_threshold
        
        if np.sum(mask) == 0:
            continue
        
        # Integrate to get cumulative number density above threshold
        # phi is in Mpc^-3 dex^-1, so integrate over dex
        log_mass_above = log_mass[mask]
        phi_above = phi_linear[mask]
        phi_lower_above = phi_lower_bound[mask]
        phi_upper_above = phi_upper_bound[mask]
        
        # Use trapezoidal integration
        n_total = np.trapz(phi_above, log_mass_above)
        n_total_lower = np.trapz(phi_lower_above, log_mass_above)
        n_total_upper = np.trapz(phi_upper_above, log_mass_above)
        
        # Only include if we have positive values
        if n_total > 0 and n_total_lower > 0 and n_total_upper > 0:
            redshifts.append(z_mean)
            n_cumulative.append(np.log10(n_total))
            n_cumulative_lower.append(np.log10(n_total_lower))
            n_cumulative_upper.append(np.log10(n_total_upper))
    
    return np.array(redshifts), np.array(n_cumulative), np.array(n_cumulative_lower), np.array(n_cumulative_upper)

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
    
    # Define redshift range - extended down to z=0 for continuous data
    target_redshifts = np.arange(0, 17, 1)  # z=0 to z=16
    
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
        
        # Add analytical predictions from Li+2023 (three lines)
        try:
            # FFB eps_max=1.0
            set_option(FFB_SFE_MAX=1.0)
            redshifts_ffb1, n_density_ffb1 = [], []
            for z_actual in actual_redshifts:
                lgMs_bins, dNdlgMs = compute_dNdlgMs(z_actual)
                mask = lgMs_bins > np.log10(mass_threshold)
                if np.sum(mask) > 0:
                    n_cumulative = np.trapz(dNdlgMs[mask], lgMs_bins[mask])
                    if n_cumulative > 0:
                        redshifts_ffb1.append(z_actual)
                        n_density_ffb1.append(np.log10(n_cumulative))
            if len(redshifts_ffb1) > 0:
                ax.plot(redshifts_ffb1, n_density_ffb1,
                       color='dodgerblue', linestyle='--', linewidth=2,
                       label='Li+ 2024' if ax_idx == 0 else '',
                       alpha=0.7, zorder=4)
            
            # FFB eps_max=0.2
            set_option(FFB_SFE_MAX=0.2)
            redshifts_ffb02, n_density_ffb02 = [], []
            for z_actual in actual_redshifts:
                lgMs_bins, dNdlgMs = compute_dNdlgMs(z_actual)
                mask = lgMs_bins > np.log10(mass_threshold)
                if np.sum(mask) > 0:
                    n_cumulative = np.trapz(dNdlgMs[mask], lgMs_bins[mask])
                    if n_cumulative > 0:
                        redshifts_ffb02.append(z_actual)
                        n_density_ffb02.append(np.log10(n_cumulative))
            if len(redshifts_ffb02) > 0:
                ax.plot(redshifts_ffb02, n_density_ffb02,
                       color='orange', linestyle='--', linewidth=2,
                       label='',
                       alpha=0.7, zorder=4)
            
            # UM model (no FFB)
            set_option(FFB_SFE_MAX=0.0)
            redshifts_um, n_density_um = [], []
            for z_actual in actual_redshifts:
                lgMs_bins, dNdlgMs = compute_dNdlgMs(z_actual)
                mask = lgMs_bins > np.log10(mass_threshold)
                if np.sum(mask) > 0:
                    n_cumulative = np.trapz(dNdlgMs[mask], lgMs_bins[mask])
                    if n_cumulative > 0:
                        redshifts_um.append(z_actual)
                        n_density_um.append(np.log10(n_cumulative))
            if len(redshifts_um) > 0:
                ax.plot(redshifts_um, n_density_um,
                       color='gray', linestyle='--', linewidth=2,
                       label='',
                       alpha=0.7, zorder=4)
            
            # Reset to default
            set_option(FFB_SFE_MAX=1.0)
            print(f"  Li+2023 analytical predictions added (3 lines) for M* > {mass_threshold:.0e}")
        except Exception as e:
            print(f"  Warning: Could not compute analytical prediction: {e}")
        
        # Add COSMOS2020 observational data (only for 1e10 threshold on right plot)
        if mass_threshold == 1e10:
            try:
                print(f"\n  Loading COSMOS2020 observational data for M* > {mass_threshold:.0e}...")
                z_cosmos, n_cosmos, n_cosmos_lower, n_cosmos_upper = load_cosmos2020_cumulative(mass_threshold)
                
                if len(z_cosmos) > 0:
                    # Plot error bars with pre-JWST style (white face, light gray edge)
                    ax.errorbar(z_cosmos, n_cosmos,
                               yerr=[n_cosmos - n_cosmos_lower, n_cosmos_upper - n_cosmos],
                               fmt='s', markersize=8,
                               color='black', linewidth=3, capsize=2,
                               label='Weaver+23',
                               alpha=1.0, zorder=5)
                    print(f"  COSMOS2020: {len(z_cosmos)} redshift bins plotted (z={z_cosmos.min():.1f} to {z_cosmos.max():.1f})")
            except Exception as e:
                print(f"  Warning: Could not load COSMOS2020 data: {e}")
            
            # Add Stefanon+2021 observational data
            try:
                print(f"\n  Loading Stefanon+2021 observational data for M* > {mass_threshold:.0e}...")
                z_stefanon, n_stefanon, n_stefanon_lower, n_stefanon_upper = load_stefanon2021_cumulative(mass_threshold)
                
                if len(z_stefanon) > 0:
                    # Plot error bars with pre-JWST style (white face, light gray edge)
                    ax.errorbar(z_stefanon, n_stefanon,
                               yerr=[n_stefanon - n_stefanon_lower, n_stefanon_upper - n_stefanon],
                               fmt='o', markersize=8,
                               color='black', linewidth=3, capsize=2,
                               label='Stefanon+21',
                               alpha=1.0, zorder=5)
                    print(f"  Stefanon+2021: {len(z_stefanon)} redshift bins plotted (z={z_stefanon.min():.1f} to {z_stefanon.max():.1f})")
            except Exception as e:
                print(f"  Warning: Could not load Stefanon+2021 data: {e}")
            
            # Add Song+2016 observational data
            try:
                print(f"\n  Loading Song+2016 observational data for M* > {mass_threshold:.0e}...")
                z_song, n_song, n_song_lower, n_song_upper = load_song2016_cumulative(mass_threshold)
                
                # No filtering - show all available data
                if len(z_song) > 0:
                    # Plot error bars with pre-JWST style (white face, light gray edge)
                    ax.errorbar(z_song, n_song,
                               yerr=[n_song - n_song_lower, n_song_upper - n_song],
                               fmt='D', markersize=8,
                               color='black', linewidth=3, capsize=2,
                               label='Song+16',
                               alpha=1.0, zorder=5)
                    print(f"  Song+2016: {len(z_song)} redshift bins plotted (z={z_song.min():.1f} to {z_song.max():.1f})")
            except Exception as e:
                print(f"  Warning: Could not load Song+2016 data: {e}")
        
        # Formatting
        ax.set_xlim(3, 16)
        # Extend y-axis for right plot to show Song+2016 data (no mass threshold)
        if mass_threshold == 1e10:
            ax.set_ylim(-6.5, -2.5)
        else:
            ax.set_ylim(-6.5, -2.5)
        
        # Set integer ticks with minor ticks at 20% intervals
        from matplotlib.ticker import MultipleLocator
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.2))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.2))
        
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
        
        # Show legend on first subplot in lower left, and on right plot in upper right
        if ax_idx == 0:
            ax.legend(loc='lower left', fontsize=10, frameon=False)
        elif mass_threshold == 1e10:
            # For right plot with COSMOS data, show observations legend in upper right
            ax.legend(loc='upper right', fontsize=10, frameon=False, bbox_to_anchor=(0.95, 0.87))
    
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

def calculate_uv_luminosity_function(stellar_mass, sfr_disk, sfr_bulge, volume, hubble_h, redshift, binwidth=0.5, use_stellar_mass=True):
    """Calculate UV luminosity function with Poisson errors

    Args:
        stellar_mass: Stellar mass in Msun
        sfr_disk: Disk star formation rate in Msun/yr
        sfr_bulge: Bulge star formation rate in Msun/yr
        volume: Comoving volume in Mpc^3
        hubble_h: Hubble parameter h
        redshift: Redshift
        binwidth: Magnitude bin width (default 0.5)
        use_stellar_mass: If True, use Li+24 stellar mass-based M_UV relation (func_MUV_lgMs).
                         If False, use Behroozi+2020 SFR-based conversion.

    The stellar mass method matches the Li+24 analytical UVLF normalization better
    because it captures cumulative star formation rather than instantaneous SFR.

    Returns: M_UV bins, log(phi), log(phi_lower), log(phi_upper)
    """
    if use_stellar_mass:
        # Use Li+24 stellar mass-based M_UV relation (Yung+2023)
        # MUV = -2.3 * (log10(Ms) - 9) - 20.5
        # Select galaxies with stellar mass > 0
        w = np.where(stellar_mass > 0.0)[0]
        if len(w) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        lgMs = np.log10(stellar_mass[w])
        M_UV = func_MUV_lgMs(lgMs, redshift)
    else:
        # Original method: SFR to UV using Behroozi+2020 (eq B3)
        # Total star formation rate
        sfr_total = sfr_disk + sfr_bulge

        # Select galaxies with SFR > 0
        w = np.where(sfr_total > 0.0)[0]
        if len(w) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        # Convert SFR to UV magnitude at 1500Å using Behroozi+2020 formula
        M_UV = func_MUV_sfr(sfr_total[w], redshift)
    
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

def load_observational_uvlf_data():
    """Load and process all observational UV luminosity function data.
    
    Applies necessary scaling factors to phi values and errors:
    - adams_lf_2024: multiply phi by 10^-5
    - bouwens_lf_2021: raw values (already in Mpc^-3 mag^-1)
    - bouwens_lf_2023: raw values (already in Mpc^-3 mag^-1)
    - donnan_lf_2023: multiply phi by 10^-6
    - finkelstein_lf_2022: multiply phi by 10^-6
    - harikane_lf_2023: raw values (already in Mpc^-3 mag^-1)
    
    Returns:
        dict: Dictionary with keys as dataset names, values are dicts with:
              'redshifts', 'M_UV', 'phi', 'phi_err_up', 'phi_err_low'
    """
    # Data directory is in the main SAGE26 folder, not in output
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data') + '/'
    obs_data = {}
    
    # Load Adams et al. 2024 - multiply by 10^-5
    try:
        table = Table.read(data_dir + 'adams_lf_2024.ecsv', format='ascii.ecsv')
        # Parse redshift from label (e.g., 'z=8' -> 8.0)
        redshifts = []
        for label in table['redshift_label']:
            # Extract first number from label (handles 'z=8', 'z=10.5', 'z=8-NoNEP')
            z_str = label.split('=')[1].split('-')[0]
            redshifts.append(float(z_str))
        
        obs_data['adams_2024'] = {
            'redshifts': np.array(redshifts),
            'M_UV': np.array(table['M_UV']),
            'phi': np.array(table['phi']) * 1e-5,
            'phi_err_up': np.array(table['phi_err']) * 1e-5,
            'phi_err_low': np.array(table['phi_err']) * 1e-5,
        }
        print(f"Loaded Adams+2024: {len(table)} data points")
    except Exception as e:
        print(f"Warning: Could not load adams_lf_2024.ecsv: {e}")
    
    # Load Bouwens et al. 2021 - raw values
    try:
        table = Table.read(data_dir + 'bouwens_lf_2021.ecsv', format='ascii.ecsv')
        obs_data['bouwens_2021'] = {
            'redshifts': np.array(table['redshift_bin'], dtype=float),
            'M_UV': np.array(table['M_1600']),
            'phi': np.array(table['phi']),
            'phi_err_up': np.array(table['phi_err']),
            'phi_err_low': np.array(table['phi_err']),
        }
        print(f"Loaded Bouwens+2021: {len(table)} data points")
    except Exception as e:
        print(f"Warning: Could not load bouwens_lf_2021.ecsv: {e}")
    
    # Load Bouwens et al. 2023 - raw values
    try:
        table = Table.read(data_dir + 'bouwens_lf_2023.ecsv', format='ascii.ecsv')
        # Parse redshift from label (e.g., 'z~8' -> 8.0, 'z~8-9' -> 8.5)
        redshifts = []
        for label in table['redshift_label']:
            z_str = label.replace('z~', '').replace('z=', '')
            if '-' in z_str:
                # Average of range
                z_parts = z_str.split('-')
                z = (float(z_parts[0]) + float(z_parts[1])) / 2.0
            else:
                z = float(z_str)
            redshifts.append(z)
        
        obs_data['bouwens_2023'] = {
            'redshifts': np.array(redshifts),
            'M_UV': np.array(table['M_UV']),
            'phi': np.array(table['phi_star']),
            'phi_err_up': np.array(table['phi_star_err']),
            'phi_err_low': np.array(table['phi_star_err']),
        }
        print(f"Loaded Bouwens+2023: {len(table)} data points")
    except Exception as e:
        print(f"Warning: Could not load bouwens_lf_2023.ecsv: {e}")
    
    # Load Donnan et al. 2023 - multiply by 10^-6
    try:
        table = Table.read(data_dir + 'donnan_lf_2023.ecsv', format='ascii.ecsv')
        obs_data['donnan_2023'] = {
            'redshifts': np.array(table['redshift']),
            'M_UV': np.array(table['M_UV']),
            'phi': np.array(table['phi']) * 1e-6,
            'phi_err_up': np.array(table['phi_err_up']) * 1e-6,
            'phi_err_low': np.array(table['phi_err_low']) * 1e-6,
        }
        print(f"Loaded Donnan+2023: {len(table)} data points")
    except Exception as e:
        print(f"Warning: Could not load donnan_lf_2023.ecsv: {e}")
    
    # Load Finkelstein et al. 2022 - multiply by 10^-6
    try:
        table = Table.read(data_dir + 'finkelstein_lf_2022.ecsv', format='ascii.ecsv')
        obs_data['finkelstein_2022'] = {
            'redshifts': np.array(table['redshift']),
            'M_UV': np.array(table['mag_bin']),
            'phi': np.array(table['number_density']) * 1e-6,
            'phi_err_up': np.array(table['density_err_up']) * 1e-6,
            'phi_err_low': np.array(table['density_err_low']) * 1e-6,
        }
        print(f"Loaded Finkelstein+2022: {len(table)} data points")
    except Exception as e:
        print(f"Warning: Could not load finkelstein_lf_2022.ecsv: {e}")
    
    # Load Harikane et al. 2023 - raw values
    try:
        table = Table.read(data_dir + 'harikane_lf_2023.ecsv', format='ascii.ecsv')
        # Filter out upper limits (phi == 0.0)
        mask = table['phi'] > 0.0
        obs_data['harikane_2023'] = {
            'redshifts': np.array(table['redshift_approx'][mask], dtype=float),
            'M_UV': np.array(table['M_UV'][mask]),
            'phi': np.array(table['phi'][mask]),
            'phi_err_up': np.array(table['phi_err_up'][mask]),
            'phi_err_low': np.array(table['phi_err_low'][mask]),
        }
        print(f"Loaded Harikane+2023: {np.sum(mask)} data points (excluded upper limits)")
    except Exception as e:
        print(f"Warning: Could not load harikane_lf_2023.ecsv: {e}")
    
    # Load McLeod et al. 2024 - multiply by 10^-5
    try:
        table = Table.read(data_dir + 'mcleod_lf_2024.ecsv', format='ascii.ecsv')
        obs_data['mcleod_2024'] = {
            'redshifts': np.array(table['redshift']),
            'M_UV': np.array(table['M_1500']),
            'phi': np.array(table['phi']) * 1e-5,
            'phi_err_up': np.array(table['phi_err']) * 1e-5,
            'phi_err_low': np.array(table['phi_err']) * 1e-5,
        }
        print(f"Loaded McLeod+2024: {len(table)} data points")
    except Exception as e:
        print(f"Warning: Could not load mcleod_lf_2024.ecsv: {e}")
    
    # Load Morishita et al. 2018 - data already in log space, filter upper limits
    try:
        table = Table.read(data_dir + 'morishita_lf_2018.ecsv', format='ascii.ecsv')
        # Filter out upper limits
        mask = ~table['is_upper_limit']
        # Convert from log space to linear space
        phi = 10**np.array(table['log_phi'][mask])
        # Asymmetric errors in log space -> linear space
        log_phi_vals = np.array(table['log_phi'][mask])
        log_phi_err_up = np.array(table['log_phi_err_up'][mask])
        log_phi_err_low = np.array(table['log_phi_err_low'][mask])
        phi_upper = 10**(log_phi_vals + log_phi_err_up)
        phi_lower = 10**(log_phi_vals - log_phi_err_low)
        
        obs_data['morishita_2018'] = {
            'redshifts': np.array(table['redshift_approx'][mask], dtype=float),
            'M_UV': np.array(table['M_UV'][mask]),
            'phi': phi,
            'phi_err_up': phi_upper - phi,
            'phi_err_low': phi - phi_lower,
        }
        print(f"Loaded Morishita+2018: {np.sum(mask)} data points (excluded upper limits)")
    except Exception as e:
        print(f"Warning: Could not load morishita_lf_2018.ecsv: {e}")
    
    # Load Yan et al. 2023 - multiply by 10^-6, use only Avg sample
    try:
        table = Table.read(data_dir + 'yan_lf_2023.ecsv', format='ascii.ecsv')
        # Use only the averaged sample (Avg)
        mask = table['sample_type'] == 'Avg'
        obs_data['yan_2023'] = {
            'redshifts': np.array(table['redshift_approx'][mask]),
            'M_UV': np.array(table['M_UV'][mask]),
            'phi': np.array(table['phi'][mask]) * 1e-6,
            'phi_err_up': np.array(table['phi_err'][mask]) * 1e-6,
            'phi_err_low': np.array(table['phi_err'][mask]) * 1e-6,
        }
        print(f"Loaded Yan+2023: {np.sum(mask)} data points (Avg sample)")
    except Exception as e:
        print(f"Warning: Could not load yan_lf_2023.ecsv: {e}")
    
    # Load Harikane et al. 2024 - raw values, exclude upper limits
    try:
        table = Table.read(data_dir + 'harikane_lf_2024.ecsv', format='ascii.ecsv')
        # Exclude upper limits
        mask = table['limit_type'] != 'upper_limit'
        obs_data['harikane_2024'] = {
            'redshifts': np.array(table['redshift_approx'][mask], dtype=float),
            'M_UV': np.array(table['M_UV'][mask]),
            'phi': np.array(table['phi'][mask]),
            'phi_err_up': np.array(table['phi_err_up'][mask]),
            'phi_err_low': np.array(table['phi_err_low'][mask]),
        }
        print(f"Loaded Harikane+2024: {np.sum(mask)} data points (excluded upper limits)")
    except Exception as e:
        print(f"Warning: Could not load harikane_lf_2024.ecsv: {e}")
    
    # Load Oesch et al. 2018 - multiply by 10^-4, exclude upper limits (phi_star=0)
    try:
        table = Table.read(data_dir + 'oesch_lf_2018.ecsv', format='ascii.ecsv')
        mask = table['phi_star'] > 0
        obs_data['oesch_2018'] = {
            'redshifts': np.array(table['redshift'][mask]),
            'M_UV': np.array(table['M_UV'][mask]),
            'phi': np.array(table['phi_star'][mask]) * 1e-4,
            'phi_err_up': np.array(table['phi_star_err_up'][mask]) * 1e-4,
            'phi_err_low': np.array(table['phi_star_err_low'][mask]) * 1e-4,
        }
        print(f"Loaded Oesch+2018: {np.sum(mask)} data points")
    except Exception as e:
        print(f"Warning: Could not load oesch_lf_2018.ecsv: {e}")
    
    # Load Stefanon et al. 2019 - multiply by 10^-4, exclude upper limits (phi_star=0)
    try:
        table = Table.read(data_dir + 'stefanon_lf_2019.ecsv', format='ascii.ecsv')
        mask = table['phi_star'] > 0
        obs_data['stefanon_2019'] = {
            'redshifts': np.array(table['redshift'][mask]),
            'M_UV': np.array(table['M_UV'][mask]),
            'phi': np.array(table['phi_star'][mask]) * 1e-4,
            'phi_err_up': np.array(table['phi_star_err_up'][mask]) * 1e-4,
            'phi_err_low': np.array(table['phi_star_err_low'][mask]) * 1e-4,
        }
        print(f"Loaded Stefanon+2019: {np.sum(mask)} data points")
    except Exception as e:
        print(f"Warning: Could not load stefanon_lf_2019.ecsv: {e}")

    # Load Finkelstein et al. 2024 - multiply by 10^-4, exclude upper limits (phi_star=0)
    try:
        table = Table.read(data_dir + 'finkelstein_lf_2024.ecsv', format='ascii.ecsv')
        obs_data['finkelstein_2024'] = {
            'redshifts': np.array(table['z_bin']),
            'M_UV': np.array(table['M_UV']),
            'phi': np.array(table['Phi_10_5']) * 1e-5,
            'phi_err_up': np.array(table['e_Phi_upper']) * 1e-5,
            'phi_err_low': np.array(table['e_Phi_lower']) * 1e-5,
        }
        print(f"Loaded Finkelstein+2024: {np.sum(mask)} data points")
    except Exception as e:
        print(f"Warning: Could not load finkelstein_lf_2024.ecsv: {e}")
    
    return obs_data

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
    
    # Load observational data
    print('\nLoading observational data...')
    obs_data = load_observational_uvlf_data()
    
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
    
    # Track which datasets have been added to legend
    datasets_in_legend = set()
    
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
            M_UV_bins, uvlf, uvlf_lower, uvlf_upper = calculate_uv_luminosity_function(stellar_mass, sfr_disk, sfr_bulge, volume, hubble_h, z_actual)
            
            # Plot
            valid = uvlf > -9  # Only plot non-zero bins
            
            # Plot line
            # Only add label in first subplot (z=9.278)
            label = model['name'] if idx == 0 else ''
            ax.plot(M_UV_bins[valid], uvlf[valid], 
                   color=model['color'], 
                   linestyle=model['linestyle'],
                   linewidth=model['linewidth'],
                   label=label,
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
        
        # Add analytical predictions from Li+2023 (three lines)
        try:
            MUV_analytical = np.linspace(-24, -16, 100)
            
            # FFB eps_max=1.0
            set_option(FFB_SFE_MAX=1.0)
            MUV_bins, dNdMUV_ffb1 = compute_dNdMUV_Ms(z_actual, MUV=MUV_analytical, attenuation=None)
            log_phi_ffb1 = np.log10(dNdMUV_ffb1)
            valid = np.isfinite(log_phi_ffb1) & (log_phi_ffb1 > -9)
            ax.plot(MUV_bins[valid], log_phi_ffb1[valid],
                   color='dodgerblue', linestyle='--', linewidth=2,
                   label='Li+ 2024' if idx == 0 else '',
                   alpha=0.7, zorder=4)
            
            # FFB eps_max=0.2
            set_option(FFB_SFE_MAX=0.2)
            MUV_bins, dNdMUV_ffb02 = compute_dNdMUV_Ms(z_actual, MUV=MUV_analytical, attenuation=None)
            log_phi_ffb02 = np.log10(dNdMUV_ffb02)
            valid = np.isfinite(log_phi_ffb02) & (log_phi_ffb02 > -9)
            ax.plot(MUV_bins[valid], log_phi_ffb02[valid],
                   color='orange', linestyle='--', linewidth=2,
                   label='',
                   alpha=0.7, zorder=4)
            
            # UM model (no FFB)
            set_option(FFB_SFE_MAX=0.0)
            MUV_bins, dNdMUV_um = compute_dNdMUV_Ms(z_actual, MUV=MUV_analytical, attenuation=None)
            log_phi_um = np.log10(dNdMUV_um)
            valid = np.isfinite(log_phi_um) & (log_phi_um > -9)
            ax.plot(MUV_bins[valid], log_phi_um[valid],
                   color='gray', linestyle='--', linewidth=2,
                   label='',
                   alpha=0.7, zorder=4)
            
            # Reset to default
            set_option(FFB_SFE_MAX=1.0)
            print(f"  Li+2023 analytical UVLF predictions added (3 lines)")
        except Exception as e:
            print(f"  Warning: Could not compute analytical UVLF: {e}")
        
        # Plot observational data for this redshift
        # Match observations within ±1.5 redshift of current panel
        # Exception: Finkelstein data should only appear in closest panel
        z_tolerance = 1.5
        obs_count = 0
        # Dataset name mapping for cleaner legend labels and marker styles
        # Pre-JWST datasets have white face, light gray edge, thick lines
        pre_jwst_datasets = ['oesch_2018', 'morishita_2018', 'stefanon_2019', 'finkelstein_2022', 'bouwens_2021']
        
        dataset_styles = {
            'adams_2024': {'label': 'Adams+24', 'marker': 'o', 'facecolor': 'black', 'edgecolor': 'black', 'linewidth': 1.5},
            'bouwens_2021': {'label': '', 'marker': 's', 'facecolor': 'white', 'edgecolor': 'lightgray', 'linewidth': 3},
            'bouwens_2023': {'label': 'Bouwens+23', 'marker': 'D', 'facecolor': 'black', 'edgecolor': 'black', 'linewidth': 1.5},
            'donnan_2023': {'label': 'Donnan+23', 'marker': '^', 'facecolor': 'black', 'edgecolor': 'black', 'linewidth': 1.5},
            'finkelstein_2022': {'label': '', 'marker': 'v', 'facecolor': 'white', 'edgecolor': 'lightgray', 'linewidth': 3},
            'harikane_2023': {'label': 'Harikane+23', 'marker': 'p', 'facecolor': 'black', 'edgecolor': 'black', 'linewidth': 1.5},
            'harikane_2024': {'label': 'Harikane+24', 'marker': '+', 'facecolor': 'black', 'edgecolor': 'black', 'linewidth': 1.5},
            'mcleod_2024': {'label': 'McLeod+24', 'marker': '*', 'facecolor': 'black', 'edgecolor': 'black', 'linewidth': 1.5},
            'morishita_2018': {'label': '', 'marker': 'h', 'facecolor': 'white', 'edgecolor': 'lightgray', 'linewidth': 3},
            'oesch_2018': {'label': '', 'marker': '<', 'facecolor': 'white', 'edgecolor': 'lightgray', 'linewidth': 3},
            'stefanon_2019': {'label': '', 'marker': '>', 'facecolor': 'white', 'edgecolor': 'lightgray', 'linewidth': 3},
            'yan_2023': {'label': 'Yan+23', 'marker': 'X', 'facecolor': 'black', 'edgecolor': 'black', 'linewidth': 1.5},
            'finkelstein_2024': {'label': 'Finkelstein+24', 'marker': 'P', 'facecolor': 'black', 'edgecolor': 'black', 'linewidth': 1.5}
        }
        
        # Track if pre-JWST legend entry has been added
        pre_jwst_legend_added = False
        for dataset_name, data in obs_data.items():
            # Find data points within redshift tolerance
            # For Finkelstein data, use tighter tolerance to only show in closest panel
            if dataset_name == 'finkelstein_2022':
                z_mask = np.abs(data['redshifts'] - z_actual) < 0.5
            # For Yan data at z=17.3, show in z=15.343 panel
            elif dataset_name == 'yan_2023':
                z_mask = np.abs(data['redshifts'] - z_actual) < z_tolerance
                # Also include z=17.3 data in the z~15.3 panel
                if np.abs(z_actual - 15.343) < 0.1:
                    z_mask = z_mask | (np.abs(data['redshifts'] - 17.3) < 0.1)
            else:
                z_mask = np.abs(data['redshifts'] - z_actual) < z_tolerance
            if np.sum(z_mask) > 0:
                M_UV_obs = data['M_UV'][z_mask]
                phi_obs = data['phi'][z_mask]
                phi_err_up = data['phi_err_up'][z_mask]
                phi_err_low = data['phi_err_low'][z_mask]
                
                # Convert to log10 for plotting
                log_phi_obs = np.log10(phi_obs)
                log_phi_err_up = np.log10(phi_obs + phi_err_up)
                log_phi_err_low = np.log10(phi_obs - phi_err_low)
                
                # Get marker style for this dataset
                style = dataset_styles.get(dataset_name, {'label': dataset_name, 'marker': 'o', 'facecolor': 'black', 'edgecolor': 'black', 'linewidth': 1.5})
                
                # Plot with different markers for each dataset (matching SMF grid size)
                # Add label only the first time this dataset appears across all subplots
                label = style['label'] if dataset_name not in datasets_in_legend else ''
                
                # For pre-JWST datasets, add a single "pre JWST" legend entry with circle marker
                # Use circle marker for legend, but actual marker for plot
                # Only add legend in first subplot (z=9.278)
                if dataset_name in pre_jwst_datasets and not pre_jwst_legend_added and idx == 0:
                    # Plot with actual marker shape but no label
                    ax.errorbar(M_UV_obs, log_phi_obs,
                               yerr=[log_phi_obs - log_phi_err_low, log_phi_err_up - log_phi_obs],
                               fmt=style['marker'], markersize=8,
                               markerfacecolor=style['facecolor'], markeredgecolor=style['edgecolor'],
                               markeredgewidth=style['linewidth'],
                               ecolor=style['edgecolor'], elinewidth=style['linewidth'], capsize=2,
                               alpha=1.0, zorder=5, label='')
                    # Add circle marker for legend only in first subplot
                    ax.plot([], [], 'o', markersize=8, markerfacecolor='white', 
                           markeredgecolor='lightgray', markeredgewidth=3, 
                           label='pre JWST')
                    pre_jwst_legend_added = True
                    datasets_in_legend.add(dataset_name)
                elif dataset_name in pre_jwst_datasets:
                    # Already added legend, just plot without label
                    ax.errorbar(M_UV_obs, log_phi_obs,
                               yerr=[log_phi_obs - log_phi_err_low, log_phi_err_up - log_phi_obs],
                               fmt=style['marker'], markersize=8,
                               markerfacecolor=style['facecolor'], markeredgecolor=style['edgecolor'],
                               markeredgewidth=style['linewidth'],
                               ecolor=style['edgecolor'], elinewidth=style['linewidth'], capsize=2,
                               alpha=1.0, zorder=5, label='')
                    datasets_in_legend.add(dataset_name)
                else:
                    # Non-pre-JWST datasets - normal plotting
                    if label:
                        datasets_in_legend.add(dataset_name)
                    ax.errorbar(M_UV_obs, log_phi_obs,
                               yerr=[log_phi_obs - log_phi_err_low, log_phi_err_up - log_phi_obs],
                               fmt=style['marker'], markersize=8,
                               markerfacecolor=style['facecolor'], markeredgecolor=style['edgecolor'],
                               markeredgewidth=style['linewidth'],
                               ecolor=style['edgecolor'], elinewidth=style['linewidth'], capsize=2,
                               alpha=1.0, zorder=5, label=label)
                obs_count += np.sum(z_mask)
        
        if obs_count > 0:
            print(f"  Added {obs_count} observational data points")
        
        # Formatting
        ax.set_xlim(-24, -16)
        ax.invert_xaxis()  # Flip x-axis (brighter on right, fainter on left)
        ax.set_ylim(-8, -2)
        
        # Set integer ticks with minor ticks at 20% intervals
        from matplotlib.ticker import MultipleLocator
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.2))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.2))
        
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
        
        # Show legend in each subplot that has labeled data
        handles, labels = ax.get_legend_handles_labels()
        if len(labels) > 0:
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

def load_adams2024_cumulative(M_UV_threshold=-20, tolerance=0.5):
    """Load Adams+2024 UVLF data and compute cumulative number density
    
    Integrates the luminosity function to get n(< M_UV_threshold).
    Uses trapezoidal integration over available data points.
    
    Args:
        M_UV_threshold: Magnitude threshold (e.g., -20)
        tolerance: Tolerance in magnitudes for selecting data around threshold
    
    Returns:
        redshifts: Array of mean redshifts
        n_cumulative: Array of log10(n) in Mpc^-3
        n_cumulative_lower: Lower error bounds
        n_cumulative_upper: Upper error bounds
    """
    filename = './data/adams_lf_2024.ecsv'
    
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found")
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    try:
        # Load data
        from astropy.table import Table
        data = Table.read(filename, format='ascii.ecsv')
        
        # Parse redshift labels to get numeric redshifts
        # Labels are like "z=8", "z=9", "z=10.5", etc.
        redshift_map = {
            'z=8': 8.0,
            'z=8-NoNEP': 8.0,
            'z=9': 9.0,
            'z=10.5': 10.5,
            'z=12.5': 12.5
        }
        
        # Group data by redshift
        unique_labels = set(data['redshift_label'])
        
        redshifts = []
        n_cumulative = []
        n_cumulative_lower = []
        n_cumulative_upper = []
        
        for label in sorted(unique_labels, key=lambda x: redshift_map.get(x, 999)):
            if label not in redshift_map:
                continue
            
            z = redshift_map[label]
            
            # Get data for this redshift
            mask = data['redshift_label'] == label
            M_UV = data['M_UV'][mask]
            phi = data['phi'][mask] * 1e-5  # Convert from 1e-5 to Mpc^-3 mag^-1
            phi_err = data['phi_err'][mask] * 1e-5
            
            # Sort by M_UV for integration
            sort_idx = np.argsort(M_UV)
            M_UV = M_UV[sort_idx]
            phi = phi[sort_idx]
            phi_err = phi_err[sort_idx]
            
            # Select data points brighter than threshold (M_UV < threshold)
            bright_mask = M_UV < M_UV_threshold
            
            if np.sum(bright_mask) > 0:
                # Integrate using trapezoidal rule
                M_bright = M_UV[bright_mask]
                phi_bright = phi[bright_mask]
                phi_err_bright = phi_err[bright_mask]
                
                # Cumulative number density: integrate phi(M_UV) dM_UV
                n_cum = np.trapz(phi_bright, M_bright)
                
                # Error propagation for integration (sum in quadrature)
                # For trapezoidal rule: error is roughly sqrt(sum((err_i * dM)^2))
                dM = np.diff(M_bright)
                if len(dM) > 0:
                    # Average errors at bin edges
                    phi_err_avg = (phi_err_bright[:-1] + phi_err_bright[1:]) / 2
                    n_cum_err = np.sqrt(np.sum((phi_err_avg * dM)**2))
                else:
                    n_cum_err = phi_err_bright[0] * 0.5  # rough estimate
                
                if n_cum > 0:
                    redshifts.append(z)
                    n_cumulative.append(np.log10(n_cum))
                    n_cumulative_lower.append(np.log10(max(n_cum - n_cum_err, 1e-10)))
                    n_cumulative_upper.append(np.log10(n_cum + n_cum_err))
        
        return np.array(redshifts), np.array(n_cumulative), np.array(n_cumulative_lower), np.array(n_cumulative_upper)
        
    except Exception as e:
        print(f"Error loading Adams+2024 cumulative UVLF: {e}")
        return np.array([]), np.array([]), np.array([]), np.array([])

def load_mcleod2024_cumulative(M_UV_threshold=-20, tolerance=0.5):
    """Load McLeod+2024 UVLF data and compute cumulative number density
    
    Integrates the luminosity function to get n(< M_UV_threshold).
    Uses trapezoidal integration over available data points.
    
    Args:
        M_UV_threshold: Magnitude threshold (e.g., -20)
        tolerance: Tolerance in magnitudes for selecting data around threshold
    
    Returns:
        redshifts: Array of mean redshifts
        n_cumulative: Array of log10(n) in Mpc^-3
        n_cumulative_lower: Lower error bounds
        n_cumulative_upper: Upper error bounds
    """
    filename = './data/mcleod_lf_2024.ecsv'
    
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found")
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    try:
        # Load data
        from astropy.table import Table
        data = Table.read(filename, format='ascii.ecsv')
        
        # Group data by redshift
        unique_redshifts = np.unique(data['redshift'])
        
        redshifts = []
        n_cumulative = []
        n_cumulative_lower = []
        n_cumulative_upper = []
        
        for z in sorted(unique_redshifts):
            # Get data for this redshift
            mask = data['redshift'] == z
            M_UV = data['M_1500'][mask]
            phi = data['phi'][mask] * 1e-5  # Convert from 1e-5 to Mpc^-3 mag^-1
            phi_err = data['phi_err'][mask] * 1e-5
            
            # Sort by M_UV for integration
            sort_idx = np.argsort(M_UV)
            M_UV = M_UV[sort_idx]
            phi = phi[sort_idx]
            phi_err = phi_err[sort_idx]
            
            # Select data points brighter than threshold (M_UV < threshold)
            bright_mask = M_UV < M_UV_threshold
            
            if np.sum(bright_mask) > 0:
                # Integrate using trapezoidal rule
                M_bright = M_UV[bright_mask]
                phi_bright = phi[bright_mask]
                phi_err_bright = phi_err[bright_mask]
                
                # Cumulative number density: integrate phi(M_UV) dM_UV
                n_cum = np.trapz(phi_bright, M_bright)
                
                # Error propagation for integration (sum in quadrature)
                # For trapezoidal rule: error is roughly sqrt(sum((err_i * dM)^2))
                dM = np.diff(M_bright)
                if len(dM) > 0:
                    # Average errors at bin edges
                    phi_err_avg = (phi_err_bright[:-1] + phi_err_bright[1:]) / 2
                    n_cum_err = np.sqrt(np.sum((phi_err_avg * dM)**2))
                else:
                    n_cum_err = phi_err_bright[0] * 0.5  # rough estimate
                
                if n_cum > 0:
                    redshifts.append(z)
                    n_cumulative.append(np.log10(n_cum))
                    n_cumulative_lower.append(np.log10(max(n_cum - n_cum_err, 1e-10)))
                    n_cumulative_upper.append(np.log10(n_cum + n_cum_err))
        
        return np.array(redshifts), np.array(n_cumulative), np.array(n_cumulative_lower), np.array(n_cumulative_upper)
        
    except Exception as e:
        print(f"Error loading McLeod+2024 cumulative UVLF: {e}")
        return np.array([]), np.array([]), np.array([]), np.array([])

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
                    # Convert SFR to UV magnitude using Behroozi+2020 (correct formula from Li+2023 paper)
                    M_UV = func_MUV_sfr(stellar_mass[w], z_actual)
                    
                    # Count galaxies in bin centered on threshold (differential UVLF at M_UV = threshold)
                    bin_width = 0.5  # magnitude bin width
                    n_in_bin = np.sum((M_UV >= M_UV_threshold - bin_width/2) & (M_UV < M_UV_threshold + bin_width/2))

                    # Calculate number density (Mpc^-3 mag^-1) with Poisson errors
                    n_density = n_in_bin / volume / bin_width

                    # Poisson errors: N ± sqrt(N)
                    n_upper = (n_in_bin + np.sqrt(n_in_bin)) / volume / bin_width
                    n_lower = max((n_in_bin - np.sqrt(n_in_bin)), 1) / volume / bin_width
                    
                    if n_density > 0:
                        redshifts_data.append(z_actual)
                        phi_values.append(np.log10(n_density))
                        phi_values_upper.append(np.log10(n_upper))
                        phi_values_lower.append(np.log10(n_lower))
            
            # Plot line
            if len(redshifts_data) > 0:
                # Only show model labels on left plot (ax_idx == 0)
                model_label = model['name'] if ax_idx == 0 else ''
                ax.plot(redshifts_data, phi_values,
                       color=model['color'],
                       linestyle=model['linestyle'],
                       linewidth=model['linewidth'],
                       label=model_label,
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
        
        # Add analytical predictions from Li+2023 (three lines)
        # Interpolate to get differential UVLF at exactly M_UV = threshold
        try:
            from scipy.interpolate import interp1d

            # FFB eps_max=1.0
            set_option(FFB_SFE_MAX=1.0)
            redshifts_ffb1, phi_ffb1 = [], []
            for z_actual in actual_redshifts:
                MUV_bins, dNdMUV = compute_dNdMUV_Ms(z_actual, attenuation=None)
                # Interpolate to get value at exactly M_UV_threshold
                if M_UV_threshold >= MUV_bins.min() and M_UV_threshold <= MUV_bins.max():
                    interp_func = interp1d(MUV_bins, dNdMUV, kind='linear')
                    phi_at_threshold = interp_func(M_UV_threshold)
                    if phi_at_threshold > 0:
                        redshifts_ffb1.append(z_actual)
                        phi_ffb1.append(np.log10(phi_at_threshold))
            if len(redshifts_ffb1) > 0:
                ax.plot(redshifts_ffb1, phi_ffb1,
                       color='dodgerblue', linestyle='--', linewidth=2,
                       label='Li+ 2024' if ax_idx == 0 else '',
                       alpha=0.7, zorder=4)

            # FFB eps_max=0.2
            set_option(FFB_SFE_MAX=0.2)
            redshifts_ffb02, phi_ffb02 = [], []
            for z_actual in actual_redshifts:
                MUV_bins, dNdMUV = compute_dNdMUV_Ms(z_actual, attenuation=None)
                if M_UV_threshold >= MUV_bins.min() and M_UV_threshold <= MUV_bins.max():
                    interp_func = interp1d(MUV_bins, dNdMUV, kind='linear')
                    phi_at_threshold = interp_func(M_UV_threshold)
                    if phi_at_threshold > 0:
                        redshifts_ffb02.append(z_actual)
                        phi_ffb02.append(np.log10(phi_at_threshold))
            if len(redshifts_ffb02) > 0:
                ax.plot(redshifts_ffb02, phi_ffb02,
                       color='orange', linestyle='--', linewidth=2,
                       label='',
                       alpha=0.7, zorder=4)

            # UM model (no FFB)
            set_option(FFB_SFE_MAX=0.0)
            redshifts_um, phi_um = [], []
            for z_actual in actual_redshifts:
                MUV_bins, dNdMUV = compute_dNdMUV_Ms(z_actual, attenuation=None)
                if M_UV_threshold >= MUV_bins.min() and M_UV_threshold <= MUV_bins.max():
                    interp_func = interp1d(MUV_bins, dNdMUV, kind='linear')
                    phi_at_threshold = interp_func(M_UV_threshold)
                    if phi_at_threshold > 0:
                        redshifts_um.append(z_actual)
                        phi_um.append(np.log10(phi_at_threshold))
            if len(redshifts_um) > 0:
                ax.plot(redshifts_um, phi_um,
                       color='gray', linestyle='--', linewidth=2,
                       label='',
                       alpha=0.7, zorder=4)

            # Reset to default
            set_option(FFB_SFE_MAX=1.0)
            print(f"  Li+2023 analytical predictions added (3 lines) for M_UV = {M_UV_threshold}")
        except Exception as e:
            print(f"  Warning: Could not compute analytical UVLF prediction: {e}")
        
        # Add observational data for M_UV=-20 (right plot only)
        if ax_idx == 1 and M_UV_threshold == -20:
            try:
                z_adams, n_adams, n_adams_lower, n_adams_upper = load_adams2024_cumulative(M_UV_threshold=-20)
                if len(z_adams) > 0:
                    ax.errorbar(z_adams, n_adams,
                               yerr=[n_adams - n_adams_lower, n_adams_upper - n_adams],
                               fmt='o', color='black', markersize=8,
                               markeredgecolor='black', markeredgewidth=1.5,
                               ecolor='black', elinewidth=2, capsize=4, capthick=2,
                               label='Adams+ 2024', alpha=0.8, zorder=5)
                    print(f"  Adams+2024 observational data added for M_UV < {M_UV_threshold}")
            except Exception as e:
                print(f"  Warning: Could not load Adams+2024 data: {e}")
            
            try:
                z_mcleod, n_mcleod, n_mcleod_lower, n_mcleod_upper = load_mcleod2024_cumulative(M_UV_threshold=-20)
                if len(z_mcleod) > 0:
                    ax.errorbar(z_mcleod, n_mcleod,
                               yerr=[n_mcleod - n_mcleod_lower, n_mcleod_upper - n_mcleod],
                               fmt='s', color='black', markersize=8,
                               markeredgecolor='black', markeredgewidth=1.5,
                               ecolor='black', elinewidth=2, capsize=4, capthick=2,
                               label='McLeod+ 2024', alpha=0.8, zorder=5)
                    print(f"  McLeod+2024 observational data added for M_UV < {M_UV_threshold}")
            except Exception as e:
                print(f"  Warning: Could not load McLeod+2024 data: {e}")
        
        # Formatting
        ax.set_xlim(5, 15)
        
        # Set different y-axis limits for each plot
        if ax_idx == 0:
            ax.set_ylim(-6, -2)  # Left plot (M_UV=-17)
        else:
            ax.set_ylim(-6, -3)  # Right plot (M_UV=-20)
        
        # Set integer ticks with minor ticks at 20% intervals
        from matplotlib.ticker import MultipleLocator
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.2))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.2))
        
        ax.set_xlabel('Redshift z', fontsize=12)
        
        # Add M_UV threshold text in upper right corner
        ax.text(0.95, 0.95, rf'$M_{{\mathrm{{UV}}}} = {M_UV_threshold}$',
               transform=ax.transAxes,
               verticalalignment='top',
               horizontalalignment='right',
               fontsize=14)
        
        # Show y-axis label on left plot, legend on both plots
        if ax_idx == 0:
            ax.set_ylabel(r'$\log_{10}(n / \mathrm{Mpc}^{-3})$', fontsize=12)
        
        # Add legend to both plots in lower left corner
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
                    # Convert SFR to UV magnitude using Behroozi+2020 (correct formula from Li+2023 paper)
                    M_UV = func_MUV_sfr(stellar_mass[w], z_actual)
                    
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
        
        # Add analytical predictions from Li+ 2024
        try:
            # Compute analytical surface density by integrating UVLF and converting to observable F277W
            # We'll compute cumulative number density in 1 deg^2 for galaxies brighter than m_threshold
            
            # FFB eps_max=1.0 (upper bound)
            set_option(FFB_SFE_MAX=1.0)
            redshifts_ffb1, surf_dens_ffb1 = [], []
            for z_actual in actual_redshifts:
                # Get UVLF from analytical model
                MUV_analytical = np.linspace(-25, -10, 200)
                MUV_bins, dNdMUV = compute_dNdMUV_Ms(z_actual, MUV=MUV_analytical, attenuation=None)
                
                # Convert M_UV to m_F277W for each magnitude
                m_F277W_analytical = convert_MUV_to_mF277W(MUV_bins, z_actual)
                
                # Integrate UVLF for galaxies brighter than threshold
                # dNdMUV is in units of Mpc^-3 mag^-1
                mask = m_F277W_analytical < m_threshold
                if np.sum(mask) > 0:
                    # Number density of galaxies brighter than threshold (per Mpc^3)
                    n_density = np.abs(np.trapz(dNdMUV[mask], MUV_bins[mask]))
                    
                    # Convert to surface density per deg^2
                    # Use same angular distance calculation as SAGE
                    c = 299792.458  # km/s
                    H0 = 0.678 * 100  # Using default hubble_h
                    if z_actual < 5:
                        d_L = (c / H0) * z_actual * (1 + z_actual / 2)
                    else:
                        d_L = (c / H0) * z_actual * (1 + z_actual * 0.75 / 2)
                    d_A = d_L / (1 + z_actual)**2
                    
                    # For 1 deg^2 area
                    angular_size_rad = np.deg2rad(1.0)  # 1 degree in radians
                    comoving_size = angular_size_rad * d_A  # Mpc
                    comoving_area = comoving_size**2  # Mpc^2
                    
                    # Comoving depth for thin shell (approximate as small dz)
                    dz = 0.1
                    z1, z2 = z_actual - dz/2, z_actual + dz/2
                    if z1 < 5:
                        d_L1 = (c / H0) * z1 * (1 + z1 / 2)
                        d_L2 = (c / H0) * z2 * (1 + z2 / 2)
                    else:
                        d_L1 = (c / H0) * z1 * (1 + z1 * 0.75 / 2)
                        d_L2 = (c / H0) * z2 * (1 + z2 * 0.75 / 2)
                    comoving_depth = abs(d_L2 - d_L1) / (1 + z_actual)
                    
                    comoving_volume = comoving_area * comoving_depth  # Mpc^3
                    
                    # Surface density (galaxies per deg^2)
                    surf_dens = n_density * comoving_volume
                    
                    if surf_dens > 0:
                        redshifts_ffb1.append(z_actual)
                        surf_dens_ffb1.append(np.log10(surf_dens))
            
            if len(redshifts_ffb1) > 0:
                ax.plot(redshifts_ffb1, surf_dens_ffb1,
                       color='dodgerblue', linestyle='--', linewidth=2,
                       label='Li+ 2024' if ax_idx == 0 else '',
                       alpha=0.7, zorder=4)
            
            # FFB eps_max=0.2 (lower bound)
            set_option(FFB_SFE_MAX=0.2)
            redshifts_ffb02, surf_dens_ffb02 = [], []
            for z_actual in actual_redshifts:
                MUV_analytical = np.linspace(-25, -10, 200)
                MUV_bins, dNdMUV = compute_dNdMUV_Ms(z_actual, MUV=MUV_analytical, attenuation=None)
                m_F277W_analytical = convert_MUV_to_mF277W(MUV_bins, z_actual)
                mask = m_F277W_analytical < m_threshold
                if np.sum(mask) > 0:
                    n_density = np.abs(np.trapz(dNdMUV[mask], MUV_bins[mask]))
                    c = 299792.458
                    H0 = 0.678 * 100
                    if z_actual < 5:
                        d_L = (c / H0) * z_actual * (1 + z_actual / 2)
                    else:
                        d_L = (c / H0) * z_actual * (1 + z_actual * 0.75 / 2)
                    d_A = d_L / (1 + z_actual)**2
                    angular_size_rad = np.deg2rad(1.0)
                    comoving_size = angular_size_rad * d_A
                    comoving_area = comoving_size**2
                    dz = 0.1
                    z1, z2 = z_actual - dz/2, z_actual + dz/2
                    if z1 < 5:
                        d_L1 = (c / H0) * z1 * (1 + z1 / 2)
                        d_L2 = (c / H0) * z2 * (1 + z2 / 2)
                    else:
                        d_L1 = (c / H0) * z1 * (1 + z1 * 0.75 / 2)
                        d_L2 = (c / H0) * z2 * (1 + z2 * 0.75 / 2)
                    comoving_depth = abs(d_L2 - d_L1) / (1 + z_actual)
                    comoving_volume = comoving_area * comoving_depth
                    surf_dens = n_density * comoving_volume
                    if surf_dens > 0:
                        redshifts_ffb02.append(z_actual)
                        surf_dens_ffb02.append(np.log10(surf_dens))
            
            if len(redshifts_ffb02) > 0:
                ax.plot(redshifts_ffb02, surf_dens_ffb02,
                       color='orange', linestyle='--', linewidth=2,
                       label='',
                       alpha=0.7, zorder=4)
            
            # UM model (no FFB)
            set_option(FFB_SFE_MAX=0.0)
            redshifts_um, surf_dens_um = [], []
            for z_actual in actual_redshifts:
                MUV_analytical = np.linspace(-25, -10, 200)
                MUV_bins, dNdMUV = compute_dNdMUV_Ms(z_actual, MUV=MUV_analytical, attenuation=None)
                m_F277W_analytical = convert_MUV_to_mF277W(MUV_bins, z_actual)
                mask = m_F277W_analytical < m_threshold
                if np.sum(mask) > 0:
                    n_density = np.abs(np.trapz(dNdMUV[mask], MUV_bins[mask]))
                    c = 299792.458
                    H0 = 0.678 * 100
                    if z_actual < 5:
                        d_L = (c / H0) * z_actual * (1 + z_actual / 2)
                    else:
                        d_L = (c / H0) * z_actual * (1 + z_actual * 0.75 / 2)
                    d_A = d_L / (1 + z_actual)**2
                    angular_size_rad = np.deg2rad(1.0)
                    comoving_size = angular_size_rad * d_A
                    comoving_area = comoving_size**2
                    dz = 0.1
                    z1, z2 = z_actual - dz/2, z_actual + dz/2
                    if z1 < 5:
                        d_L1 = (c / H0) * z1 * (1 + z1 / 2)
                        d_L2 = (c / H0) * z2 * (1 + z2 / 2)
                    else:
                        d_L1 = (c / H0) * z1 * (1 + z1 * 0.75 / 2)
                        d_L2 = (c / H0) * z2 * (1 + z2 * 0.75 / 2)
                    comoving_depth = abs(d_L2 - d_L1) / (1 + z_actual)
                    comoving_volume = comoving_area * comoving_depth
                    surf_dens = n_density * comoving_volume
                    if surf_dens > 0:
                        redshifts_um.append(z_actual)
                        surf_dens_um.append(np.log10(surf_dens))
            
            if len(redshifts_um) > 0:
                ax.plot(redshifts_um, surf_dens_um,
                       color='gray', linestyle='--', linewidth=2,
                       label='',
                       alpha=0.7, zorder=4)
            
            # Reset to default
            set_option(FFB_SFE_MAX=1.0)
            
        except Exception as e:
            print(f"  Warning: Could not compute analytical surface density: {e}")
        
        # Formatting
        ax.set_xlim(5, 16)
        ax.set_ylim(-2, 3)
        
        # Set integer ticks with minor ticks at 20% intervals
        from matplotlib.ticker import MultipleLocator
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.2))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.2))
        
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
            ax.legend(loc='upper right', fontsize=10, frameon=False, bbox_to_anchor=(0.95, 0.88))
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
            volume = (model['boxsize'] / hubble_h)**3.0 * model.get('volume_fraction', 1.0)  # in Mpc^3
            
            # Calculate total SFR and M_UV
            sfr_total = sfr_disk + sfr_bulge
            w = np.where(sfr_total > 0.0)[0]
            
            if len(w) > 0:
                # Convert SFR to UV magnitude using Behroozi+2020 (correct formula from Li+2023 paper)
                M_UV = func_MUV_sfr(sfr_total[w], z_actual)
                
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
            # Madau & Dickinson 2014 plotting moved outside loop
            print(f"  {model['name']}: SMD density plotted")
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
            # Madau & Dickinson 2014 plotting moved outside loop
            print(f"  {model['name']}: SFR density plotted")
    
    # Add analytical predictions from Li+2023 to all three panels
    try:
        # Panel 1: Stellar Mass Density (three lines)
        # FFB eps_max=1.0
        set_option(FFB_SFE_MAX=1.0)
        redshifts_ffb1, rho_star_ffb1 = [], []
        for z_actual in actual_redshifts:
            rho = compute_rho_star(z_actual, M_UV_threshold_stellar)
            if rho > 0:
                redshifts_ffb1.append(z_actual)
                rho_star_ffb1.append(np.log10(rho))
        if len(redshifts_ffb1) > 0:
            axes[0].plot(redshifts_ffb1, rho_star_ffb1,
                        color='dodgerblue', linestyle='--', linewidth=2,
                        label='Li+ 2024', alpha=0.7, zorder=4)
        
        # FFB eps_max=0.2
        set_option(FFB_SFE_MAX=0.2)
        redshifts_ffb02, rho_star_ffb02 = [], []
        for z_actual in actual_redshifts:
            rho = compute_rho_star(z_actual, M_UV_threshold_stellar)
            if rho > 0:
                redshifts_ffb02.append(z_actual)
                rho_star_ffb02.append(np.log10(rho))
        if len(redshifts_ffb02) > 0:
            axes[0].plot(redshifts_ffb02, rho_star_ffb02,
                        color='orange', linestyle='--', linewidth=2,
                        label='', alpha=0.7, zorder=4)
        
        # UM model (no FFB)
        set_option(FFB_SFE_MAX=0.0)
        redshifts_um, rho_star_um = [], []
        for z_actual in actual_redshifts:
            rho = compute_rho_star(z_actual, M_UV_threshold_stellar)
            if rho > 0:
                redshifts_um.append(z_actual)
                rho_star_um.append(np.log10(rho))
        if len(redshifts_um) > 0:
            axes[0].plot(redshifts_um, rho_star_um,
                        color='gray', linestyle='--', linewidth=2,
                        label='', alpha=0.7, zorder=4)
        
        # Reset to default
        set_option(FFB_SFE_MAX=1.0)
        print(f"  Li+2023 analytical stellar mass density added (3 lines)")
    except Exception as e:
        print(f"  Warning: Could not compute analytical stellar mass density: {e}")
    
    try:
        # Panel 2: UV Luminosity Density (three lines)
        # FFB eps_max=1.0
        set_option(FFB_SFE_MAX=1.0)
        redshifts_ffb1, rho_UV_ffb1 = [], []
        for z_actual in actual_redshifts:
            rho = compute_rho_UV(z_actual, M_UV_threshold_uv, attenuation=None)
            if rho > 0:
                redshifts_ffb1.append(z_actual)
                rho_UV_ffb1.append(np.log10(rho))
        if len(redshifts_ffb1) > 0:
            axes[1].plot(redshifts_ffb1, rho_UV_ffb1,
                        color='dodgerblue', linestyle='--', linewidth=2,
                        label='', alpha=0.7, zorder=4)

        # FFB eps_max=0.2
        set_option(FFB_SFE_MAX=0.2)
        redshifts_ffb02, rho_UV_ffb02 = [], []
        for z_actual in actual_redshifts:
            rho = compute_rho_UV(z_actual, M_UV_threshold_uv, attenuation=None)
            if rho > 0:
                redshifts_ffb02.append(z_actual)
                rho_UV_ffb02.append(np.log10(rho))
        if len(redshifts_ffb02) > 0:
            axes[1].plot(redshifts_ffb02, rho_UV_ffb02,
                        color='orange', linestyle='--', linewidth=2,
                        label='', alpha=0.7, zorder=4)

        # UM model (no FFB)
        set_option(FFB_SFE_MAX=0.0)
        redshifts_um, rho_UV_um = [], []
        for z_actual in actual_redshifts:
            rho = compute_rho_UV(z_actual, M_UV_threshold_uv, attenuation=None)
            if rho > 0:
                redshifts_um.append(z_actual)
                rho_UV_um.append(np.log10(rho))
        if len(redshifts_um) > 0:
            axes[1].plot(redshifts_um, rho_UV_um,
                        color='gray', linestyle='--', linewidth=2,
                        label='', alpha=0.7, zorder=4)

        # Reset to default
        set_option(FFB_SFE_MAX=1.0)
        print(f"  Li+2023 analytical UV luminosity density added (3 lines)")

        # --- Add Finkelstein et al. 2024 rho_UV data ---
        # Redshifts
        z_fink = np.array([9.0, 11.0, 14.0])
        # log10(rho_UV) values [erg s^-1 Hz^-1 Mpc^-3]
        log_rho_uv_fink = np.array([25.4, 25.0, 24.9])
        # Errors: [Lower Error, Upper Error]
        # Lower: [0.1, 0.2, 0.7], Upper: [0.1, 0.2, 1.1]
        err_lower = np.array([0.1, 0.2, 0.7])
        err_upper = np.array([0.1, 0.2, 1.1])
        axes[1].errorbar(z_fink, log_rho_uv_fink, yerr=[err_lower, err_upper],
                        fmt='P', color='black', markersize=8,
                        label='Finkelstein+24', capsize=3, linewidth=2, zorder=10)
        print("  Finkelstein+2024 rho_UV data added to UV density panel")
    except Exception as e:
        print(f"  Warning: Could not compute analytical UV luminosity density: {e}")
    
    try:
        # Panel 3: SFR Density (three lines)
        # FFB eps_max=1.0
        set_option(FFB_SFE_MAX=1.0)
        redshifts_ffb1, rho_SFR_ffb1 = [], []
        for z_actual in actual_redshifts:
            rho = compute_rho_SFR(z_actual, M_UV_threshold_sfr)
            if rho > 0:
                redshifts_ffb1.append(z_actual)
                rho_SFR_ffb1.append(np.log10(rho))
        if len(redshifts_ffb1) > 0:
            axes[2].plot(redshifts_ffb1, rho_SFR_ffb1,
                        color='dodgerblue', linestyle='--', linewidth=2,
                        label='', alpha=0.7, zorder=4)
        
        # FFB eps_max=0.2
        set_option(FFB_SFE_MAX=0.2)
        redshifts_ffb02, rho_SFR_ffb02 = [], []
        for z_actual in actual_redshifts:
            rho = compute_rho_SFR(z_actual, M_UV_threshold_sfr)
            if rho > 0:
                redshifts_ffb02.append(z_actual)
                rho_SFR_ffb02.append(np.log10(rho))
        if len(redshifts_ffb02) > 0:
            axes[2].plot(redshifts_ffb02, rho_SFR_ffb02,
                        color='orange', linestyle='--', linewidth=2,
                        label='', alpha=0.7, zorder=4)
        
        # UM model (no FFB)
        set_option(FFB_SFE_MAX=0.0)
        redshifts_um, rho_SFR_um = [], []
        for z_actual in actual_redshifts:
            rho = compute_rho_SFR(z_actual, M_UV_threshold_sfr)
            if rho > 0:
                redshifts_um.append(z_actual)
                rho_SFR_um.append(np.log10(rho))
        if len(redshifts_um) > 0:
            axes[2].plot(redshifts_um, rho_SFR_um,
                        color='gray', linestyle='--', linewidth=2,
                        label='', alpha=0.7, zorder=4)
        
        # Reset to default
        set_option(FFB_SFE_MAX=1.0)
        print(f"  Li+2023 analytical SFR density added (3 lines)")
    except Exception as e:
        print(f"  Warning: Could not compute analytical SFR density: {e}")
    
    # Plot Madau & Dickinson 2014 observational data ONCE per relevant axis (after model loop)
    z_madau, re_madau, re_err_plus_madau, re_err_minus_madau = load_madau_dickinson_smd_2014_data()
    if z_madau is not None:
        axes[0].errorbar(z_madau, re_madau,
                        yerr=[re_err_minus_madau, re_err_plus_madau],
                        fmt='o', color='black', markersize=8, alpha=0.8,
                        label='Madau & Dickinson 2014', capsize=2, linewidth=1.5, zorder=5)
        
    # Plot Kikuchihara 2020 observational data ONCE per relevant axis (after model loop)
    z_kikuchihara, re_kikuchihara, re_err_plus_kikuchihara, re_err_minus_kikuchihara = load_kikuchihara_smd_2020_data()
    if z_kikuchihara is not None:
        axes[0].errorbar(z_kikuchihara, re_kikuchihara,
                        yerr=[re_err_minus_kikuchihara, re_err_plus_kikuchihara],
                        fmt='d', color='black', markersize=8, alpha=0.8,
                        label='Kikuchihara+20', capsize=2, linewidth=1.5, zorder=3)
        
    # Plot Papovich 2023 observational data ONCE per relevant axis (after model loop)
    z_papovich, re_papovich, re_err_plus_papovich, re_err_minus_papovich = load_papovich_smd_2023_data()
    if z_papovich is not None:
        axes[0].errorbar(z_papovich, re_papovich,
                        yerr=[re_err_minus_papovich, re_err_plus_papovich],
                        fmt='s', color='black', markersize=8, alpha=0.8,
                        label='Papovich+23', capsize=2, linewidth=1.5, zorder=3)
        
    z_mcleod, re_mcleod, re_err_plus_mcleod, re_err_minus_mcleod = load_mcleod_rho_uv_2016_data()
    if z_mcleod is not None:
        axes[1].errorbar(z_mcleod, re_mcleod,
                        yerr=[re_err_minus_mcleod, re_err_plus_mcleod],
                        fmt='^', color='black', markersize=8, alpha=0.8,
                        label='Mcleod+16', capsize=2, linewidth=1.5, zorder=5)
        
    z_mcleod2, re_mcleod2, re_err_plus_mcleod2, re_err_minus_mcleod2 = load_mcleod_rho_uv_2024_data()
    if z_mcleod2 is not None:
        axes[1].errorbar(z_mcleod2, re_mcleod2,
                        yerr=[re_err_minus_mcleod2, re_err_plus_mcleod2],
                        fmt='v', color='black', markersize=8, alpha=0.8,
                        label='Mcleod+24', capsize=2, linewidth=1.5, zorder=5)
        
    z_perez, re_perez, re_err_plus_perez, re_err_minus_perez = load_perez_rho_uv_2023_data()
    if z_perez is not None:
        axes[1].errorbar(z_perez, re_perez,
                        yerr=[re_err_minus_perez, re_err_plus_perez],
                        fmt='s', color='black', markersize=8, alpha=0.8,
                        label='Perez+23', capsize=2, linewidth=1.5, zorder=5)
        
    z_harikane, re_harikane, re_err_plus_harikane, re_err_minus_harikane = load_harikane_uv_density_2023_data()
    if z_harikane is not None:
        axes[1].errorbar(z_harikane, re_harikane,
                        yerr=[re_err_minus_harikane, re_err_plus_harikane],
                        fmt='D', color='black', markersize=8, alpha=0.8,
                        label='Harikane+23', capsize=2, linewidth=1.5, zorder=5)

    z_madau, re_madau, re_err_plus_madau, re_err_minus_madau = load_madau_dickinson_2014_data()
    if z_madau is not None:
        axes[2].errorbar(z_madau, re_madau,
                        yerr=[re_err_minus_madau, re_err_plus_madau],
                        fmt='o', color='black', markersize=8, alpha=0.8,
                        label='Madau & Dickinson 2014', capsize=2, linewidth=1.5, zorder=5)
        
    z_oesch, re_oesch, re_err_plus_oesch, re_err_minus_oesch = load_oesch_sfrd_2018_data()
    if z_oesch is not None:
        axes[2].errorbar(z_oesch, re_oesch,
                        yerr=[re_err_minus_oesch, re_err_plus_oesch],
                        fmt='*', color='black', markersize=8, alpha=0.8,
                        label='Oesch+18', capsize=2, linewidth=1.5, zorder=5)
        
    z_mcleod3, re_mcleod3, re_err_plus_mcleod3, re_err_minus_mcleod3 = load_mcleod_rho_sfr_2024_data()
    if z_mcleod3 is not None:
        axes[2].errorbar(z_mcleod3, re_mcleod3,
                        yerr=[re_err_minus_mcleod3, re_err_plus_mcleod3],
                        fmt='v', color='black', markersize=8, alpha=0.8,
                        label='Mcleod+24', capsize=2, linewidth=1.5, zorder=5)
        
    z_harikane2, re_harikane2, re_err_plus_harikane2, re_err_minus_harikane2 = load_harikane_sfr_density_2023_data()
    if z_harikane2 is not None:
        axes[2].errorbar(z_harikane2, re_harikane2,
                        yerr=[re_err_minus_harikane2, re_err_plus_harikane2],
                        fmt='D', color='black', markersize=8, alpha=0.8,
                        label='Harikane+23', capsize=2, linewidth=1.5, zorder=5)

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
            ax.set_ylim(3, 7.6)
            ax.legend(loc='lower left', fontsize=10, frameon=False)
        elif idx == 1:
            ax.text(0.95, 0.95, rf'$M_{{\mathrm{{UV}}}} < {M_UV_threshold_uv}$',
                   transform=ax.transAxes,
                   verticalalignment='top',
                   horizontalalignment='right',
                   fontsize=12)
            ax.legend(loc='lower left', fontsize=10, frameon=False)
            ax.set_ylabel(r'$\log_{10}(\rho_{\mathrm{UV}} / \mathrm{erg} \, \mathrm{s}^{-1} \, \mathrm{Hz}^{-1} \, \mathrm{Mpc}^{-3})$', fontsize=12)
        else:
            ax.text(0.95, 0.95, rf'$M_{{\mathrm{{UV}}}} < {M_UV_threshold_sfr}$',
                   transform=ax.transAxes,
                   verticalalignment='top',
                   horizontalalignment='right',
                   fontsize=12)
            ax.set_ylabel(r'$\log_{10}(\rho_{\mathrm{SFR}} / M_{\odot} \, \mathrm{yr}^{-1} \, \mathrm{Mpc}^{-3})$', fontsize=12)
            ax.set_ylim(-5, -2)
            ax.legend(loc='lower left', fontsize=10, frameon=False)

        # Set integer ticks with minor ticks at 20% intervals
        from matplotlib.ticker import MultipleLocator
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.2))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.2))
    
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

def calculate_ffb_threshold_mass(z, hubble_h, return_log=False):
    """Calculate FFB threshold mass from Li et al. 2024
    
    M_v,ffb / 10^10.8 M_sun ~ ((1+z)/10)^-6.2
    
    Args:
        z: Redshift
        hubble_h: Hubble parameter (not used, kept for API compatibility)
        return_log: If True, return log10(M_vir) instead of M_vir
    
    Returns:
        M_vir threshold in physical Msun (or log10(Msun) if return_log=True)
    """
    z_norm = (1.0 + z) / 10.0
    M_norm = 10.8
    z_exponent = -6.2
    
    # Calculate log10(M_v,ffb) in units of M_sun (physical)
    log_Mvir_ffb_Msun = M_norm + z_exponent * np.log10(z_norm)
    
    # Return log or linear based on flag
    if return_log:
        return log_Mvir_ffb_Msun
    else:
        return 10.0**log_Mvir_ffb_Msun

def plot_ffb_threshold_analysis(models=None):
    """Plot FFB threshold galaxy properties
    
    Left: Mvir vs redshift for disk galaxies at FFB threshold, colored by half-mass radius
    Right: Half-mass radius evolution for galaxies at FFB threshold (disk only)
    
    Args:
        models: List of model dictionaries to plot. If None, uses FFB 50% model only.
    """
    
    print('\n' + '='*60)
    print('Creating FFB Threshold Analysis Plot')
    print('='*60)
    
    # Use only FFB 50% model
    ffb50_model = {
        'name': 'SAGE26',
        'dir': './output/millennium/',
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
    disk_radius_lower = []
    disk_radius_upper = []
    
    # Loop through snapshots
    for snapshot, z_actual in zip(snapshots, actual_redshifts):
        # Read data
        mvir = read_hdf_from_model(model_dir, filename, snapshot, 'Mvir', hubble_h) * 1.0e10 / hubble_h
        stellar_mass = read_hdf_from_model(model_dir, filename, snapshot, 'StellarMass', hubble_h) * 1.0e10 / hubble_h
        disk_radius = read_hdf_from_model(model_dir, filename, snapshot, 'DiskRadius', hubble_h)  # Already in Mpc/h
        galaxy_type = read_hdf_from_model(model_dir, filename, snapshot, 'Type', hubble_h)
        
        # Calculate FFB threshold mass for this redshift
        M_ffb_threshold = calculate_ffb_threshold_mass(z_actual, hubble_h)
        
        # Left plot: ALL galaxies (using disk half-mass radius)
        is_disk = galaxy_type == 0
        has_stellar_mass = stellar_mass > 0
        near_threshold = (mvir > M_ffb_threshold * 0.8) & (mvir < M_ffb_threshold * 1.2)
        
        w_all = np.where(has_stellar_mass)[0]
        
        w = np.where(is_disk & has_stellar_mass & near_threshold)[0]
        
        # Left plot: collect ALL galaxies (not just disks)
        if len(w_all) > 0:
            all_mvir.extend(mvir[w_all])
            all_redshifts.extend([z_actual] * len(w_all))
            # Use disk half-mass radius for all galaxies, even if some are not disks
            radius_kpc_comov = disk_radius[w_all] * 1.68 * 1000.0 / hubble_h
            all_radii.extend(np.log10(radius_kpc_comov))
        
        # Right plot: calculate median radii for galaxies near FFB threshold
        if len(w) > 5:  # Need at least 5 galaxies
            # For disk radius - convert scale length to half-mass radius, then to physical kpc
            valid_disk = disk_radius[w] > 0
            if np.sum(valid_disk) > 0:
                disk_radius_kpc_comov = disk_radius[w[valid_disk]] * 1.68 * 1000.0 / hubble_h
                median_disk = np.median(disk_radius_kpc_comov)
                # Calculate 16th and 84th percentiles for 1-sigma errors
                disk_lower = np.percentile(disk_radius_kpc_comov, 16)
                disk_upper = np.percentile(disk_radius_kpc_comov, 84)
            else:
                median_disk = np.nan
                disk_lower = np.nan
                disk_upper = np.nan
            
            if not np.isnan(median_disk):
                redshifts_right.append(z_actual)
                median_disk_radius.append(median_disk)
                disk_radius_lower.append(disk_lower)
                disk_radius_upper.append(disk_upper)
            
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
        axes[0].set_xlim(5, 16)
        axes[0].set_ylim(9, 13)
        
        # Set integer ticks with minor ticks at 20% intervals
        from matplotlib.ticker import MultipleLocator
        axes[0].xaxis.set_major_locator(MultipleLocator(1))
        axes[0].xaxis.set_minor_locator(MultipleLocator(0.2))
        axes[0].yaxis.set_major_locator(MultipleLocator(1))
        axes[0].yaxis.set_minor_locator(MultipleLocator(0.2))
        
        axes[0].legend(loc='upper right', fontsize=10, frameon=False)
        axes[0].text(0.95, 0.88, 'All Disk Galaxies (FFB 50%)',
                    transform=axes[0].transAxes,
                    verticalalignment='top',
                    horizontalalignment='right',
                    fontsize=11,
                    color='black')
    
    # RIGHT PLOT: Half-mass radius vs redshift (disk only)
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
        
        axes[1].set_xlabel('Redshift z', fontsize=12)
        axes[1].set_ylabel(r'$R_{\mathrm{half-mass}}$ (comoving kpc)', fontsize=12)
        axes[1].set_xlim(5, 13)
        axes[1].set_yscale('log')
        axes[1].set_ylim(0.01, 3)
        axes[1].set_yticks([0.01, 0.1, 1.0])
        axes[1].set_yticklabels(['0.01', '0.1', '1.0'])
        
        # Set integer x-ticks with minor ticks at 20% intervals
        from matplotlib.ticker import MultipleLocator
        axes[1].xaxis.set_major_locator(MultipleLocator(1))
        axes[1].xaxis.set_minor_locator(MultipleLocator(0.2))
        
        axes[1].legend(loc='upper right', fontsize=10, frameon=False)
        axes[1].text(0.05, 0.95, 'Galaxies at FFB Threshold',
                    transform=axes[1].transAxes,
                    verticalalignment='top',
                    fontsize=11)
        
        # Add analytical predictions from Li+2023 (shell mode with 3 epsilon values)
        try:
            redshifts_analytical = np.linspace(5, 16, 50)
            
            # FFB eps_max=1.0 (shell)
            set_option(FFB_SFE_MAX=1.0)
            radius_ffb1_shell = []
            for z in redshifts_analytical:
                lgMh_crit = ffb_lgMh_crit(z)
                R_shell = ffb_radius(lgMh_crit, z, mode='shell', lambdas=0.025, eps=None)
                R_shell_kpc = R_shell * 1.68 * 1000.0 / hubble_h / (1.0 + z)
                radius_ffb1_shell.append(R_shell_kpc)
            axes[1].plot(redshifts_analytical, radius_ffb1_shell,
                        color='dodgerblue', linestyle='--', linewidth=2,
                        label='Li+ 2024', alpha=0.7, zorder=4)
            
            # FFB eps_max=0.2 (shell)
            set_option(FFB_SFE_MAX=0.2)
            radius_ffb02_shell = []
            for z in redshifts_analytical:
                lgMh_crit = ffb_lgMh_crit(z)
                R_shell = ffb_radius(lgMh_crit, z, mode='shell', lambdas=0.025, eps=None)
                R_shell_kpc = R_shell * 1.68 * 1000.0 / hubble_h / (1.0 + z)
                radius_ffb02_shell.append(R_shell_kpc)
            axes[1].plot(redshifts_analytical, radius_ffb02_shell,
                        color='orange', linestyle='--', linewidth=2,
                        label='', alpha=0.7, zorder=4)
            
            # UM model - use disc mode as approximation
            set_option(FFB_SFE_MAX=0.0)
            radius_um_disc = []
            for z in redshifts_analytical:
                lgMh_crit = ffb_lgMh_crit(z)
                R_disc = ffb_radius(lgMh_crit, z, mode='disc', lambdas=0.025, eps=None)
                R_disc_kpc = R_disc * 1.68 * 1000.0 / hubble_h / (1.0 + z)
                radius_um_disc.append(R_disc_kpc)
            axes[1].plot(redshifts_analytical, radius_um_disc,
                        color='gray', linestyle='--', linewidth=2,
                        label='', alpha=0.7, zorder=4)
            
            # Reset to default
            set_option(FFB_SFE_MAX=1.0)
            print(f"  Li+2023 analytical radius predictions added (3 lines)")
        except Exception as e:
            print(f"  Warning: Could not compute analytical radius: {e}")
    
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
    Right: Evolution of Half-mass radius (Physical) - SAGE disk data with rolling median and Poisson errors.
    """
    
    print('\n' + '='*60)
    print('Creating FFB Threshold Analysis Plot (Physical Units + Rolling Medians)')
    print('='*60)
    
    # Use only FFB 50% model
    target_model_name = 'FFB 50%' 
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
    
    # Arrays for Right Plot (individual data points)
    disk_z = []
    disk_r = []
    
    for snapshot, z_actual in zip(snapshots, actual_redshifts):
        # Read Data
        mvir = read_hdf_from_model(model_dir, filename, snapshot, 'Mvir', hubble_h) * 1.0e10 / hubble_h
        stellar_mass = read_hdf_from_model(model_dir, filename, snapshot, 'StellarMass', hubble_h) * 1.0e10 / hubble_h
        
        # SAGE Radius is Comoving Mpc/h
        sage_disk_radius = read_hdf_from_model(model_dir, filename, snapshot, 'DiskRadius', hubble_h) 

        # --- UNIT CONVERSION (Comoving kpc) ---
        # 1. Mpc/h -> kpc/h: * 1000
        # 2. kpc/h -> kpc:   / h
        # 3. Disk Scale Length -> Half Mass: * 1.68
        r_disk_kpc_comov = sage_disk_radius * 1.68 * 1000.0 / hubble_h
        
        # Filter for Heatmap
        galaxy_type = read_hdf_from_model(model_dir, filename, snapshot, 'Type', hubble_h)
        has_mass = stellar_mass > 0
        
        # Collect data for heatmap (all galaxies, not just disks)
        valid_map = has_mass & (r_disk_kpc_comov > 0)
        if np.sum(valid_map) > 0:
            all_mvir.extend(mvir[valid_map])
            all_redshifts.extend([z_actual] * np.sum(valid_map))
            all_radii.extend(np.log10(r_disk_kpc_comov[valid_map]))

        # --- Statistics for Right Plot ---
        M_ffb = calculate_ffb_threshold_mass(z_actual, hubble_h)
        near_threshold = (mvir > M_ffb * 0.8) & (mvir < M_ffb * 1.2)
        # Right plot: all galaxies near FFB threshold (with stellar mass)
        subset = has_mass & near_threshold
        w_sub = np.where(subset)[0]

        if len(w_sub) > 0:
            # Disk: where disk radius > 0
            disk_valid = r_disk_kpc_comov[w_sub] > 0
            if np.sum(disk_valid) > 0:
                disk_z.extend([z_actual] * np.sum(disk_valid))
                disk_r.extend(r_disk_kpc_comov[w_sub][disk_valid])

    # ================= PLOTTING =================
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # --- LEFT PANEL: Heatmap ---
    ax_left = axes[0]
    
    if len(all_mvir) > 0:
        a_mvir = np.array(all_mvir)
        a_z = np.array(all_redshifts)
        a_radii = np.array(all_radii)  # already log10
        
        # Define bins
        z_bins = np.linspace(5, 20, 60)
        m_bins = np.linspace(9, 13, 60)
        
        # 1. Bin statistics
        from scipy.stats import binned_statistic_2d
        median_radius, z_edges, mvir_edges, _ = binned_statistic_2d(
            a_z, np.log10(a_mvir), a_radii,
            statistic='median', bins=[z_bins, m_bins]
        )
        
        # 2. Smoothing
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
        cbar.set_label(r'$\log_{10} R_{\rm \frac{1}{2}}$ (kpc)', fontsize=12)

    # FFB Threshold Line
    z_line = np.linspace(5, 20, 100)
    m_ffb = [calculate_ffb_threshold_mass(z, hubble_h) for z in z_line]
    ax_left.plot(z_line, np.log10(m_ffb), 'k-', lw=3, label='FFB Threshold')
    
    ax_left.set_xlabel('Redshift z', fontsize=14)
    ax_left.set_ylabel(r'$\log_{10} M_{\rm vir}$', fontsize=14)
    ax_left.set_xlim(5, 16)
    ax_left.set_ylim(9, 13)
    
    # Set integer ticks with minor ticks at 20% intervals
    from matplotlib.ticker import MultipleLocator
    ax_left.xaxis.set_major_locator(MultipleLocator(1))
    ax_left.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax_left.yaxis.set_major_locator(MultipleLocator(1))
    ax_left.yaxis.set_minor_locator(MultipleLocator(0.2))
    ax_left.legend()

    # --- RIGHT PANEL: Rolling Median Evolution ---
    ax_right = axes[1]

    if len(disk_z) > 0:
        # Convert to arrays
        disk_z_arr = np.array(disk_z)
        disk_r_arr = np.array(disk_r)
        
        # Define redshift bins
        z_bin_edges = np.arange(5, 20.5, 0.5)  # Bins every 0.5 in redshift
        z_bin_centers = (z_bin_edges[:-1] + z_bin_edges[1:]) / 2
        
        # Calculate median in each bin
        z_plot = []
        r_median_plot = []
        r_lower_plot = []
        r_upper_plot = []
        
        for i in range(len(z_bin_edges) - 1):
            # Find galaxies in this redshift bin
            in_bin = (disk_z_arr >= z_bin_edges[i]) & (disk_z_arr < z_bin_edges[i+1])
            
            if np.sum(in_bin) > 10:  # Need at least 10 galaxies
                r_in_bin = disk_r_arr[in_bin]
                
                # Calculate median and percentiles to show scatter
                median_r = np.median(r_in_bin)
                lower_r = np.percentile(r_in_bin, 16)  # 16th percentile (1-sigma lower)
                upper_r = np.percentile(r_in_bin, 84)  # 84th percentile (1-sigma upper)
                
                z_plot.append(z_bin_centers[i])
                r_median_plot.append(median_r)
                r_lower_plot.append(lower_r)
                r_upper_plot.append(upper_r)
        
        # Convert to arrays
        z_plot = np.array(z_plot)
        r_median_plot = np.array(r_median_plot)
        r_lower_plot = np.array(r_lower_plot)
        r_upper_plot = np.array(r_upper_plot)
        
        # Plot median
        ax_right.plot(z_plot, r_median_plot,
                     color='darkorange',
                     linestyle='-',
                     linewidth=3,
                     label='SAGE Disk (Median)',
                     alpha=0.8,
                     marker='o',
                     markersize=5,
                     zorder=3)
        
        # Add error shading
        ax_right.fill_between(z_plot, r_lower_plot, r_upper_plot,
                             color='darkorange',
                             alpha=0.3,
                             label='16th-84th percentile',
                             zorder=2)

    # Theory lines
    z_t = np.linspace(5, 20, 100)
    z_10 = (1 + z_t) / 10.0
    r_th_disk = 0.31 * (z_10 ** -3.07)

    ax_right.plot(z_t, r_th_disk, color='darkorange', ls='--', lw=2, label='Li+24 (Disk)', zorder=4)

    ax_right.set_yscale('log')
    ax_right.set_xlabel('Redshift z', fontsize=14)
    ax_right.set_ylabel(r'$R_{\rm \frac{1}{2}}$ (kpc)', fontsize=14)
    ax_right.set_xlim(5, 13)
    
    # Set integer x-ticks with minor ticks at 20% intervals
    from matplotlib.ticker import MultipleLocator
    ax_right.xaxis.set_major_locator(MultipleLocator(1))
    ax_right.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax_right.yaxis.set_major_formatter(plt.ScalarFormatter())
    # ax_right.grid(True, alpha=0.1)
    ax_right.legend(frameon=False)

    plt.tight_layout()
    out_path = DirName + 'plots/ffb_threshold_analysis_empirical' + OutputFormat
    plt.savefig(out_path, dpi=300)
    print(f'Plot saved to: {out_path}')
    plt.close()
    print('='*60 + '\n')

def plot_ffb_threshold_analysis_empirical_all():
    """
    Create FFB threshold analysis plot for ALL FFB galaxies (above threshold).
    Left: Smoothed 2D heatmap of Mvir vs redshift colored by Physical SAGE Disk Radius.
    Right: Evolution of Half-mass radius (Physical) - SAGE disk data for all galaxies above FFB threshold.
    """
    
    print('\n' + '='*60)
    print('Creating FFB Analysis Plot - ALL Galaxies Above Threshold')
    print('='*60)
    
    # Use only FFB 50% model
    target_model_name = 'FFB 50%' 
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
    
    # Arrays for Right Plot (individual data points) - three populations
    disk_z_above = []  # All galaxies above threshold
    disk_r_above = []
    disk_z_below = []  # All galaxies below threshold
    disk_r_below = []
    disk_z_threshold = []  # Galaxies near threshold (0.8-1.2x)
    disk_r_threshold = []
    disk_z_all = []  # All galaxies
    disk_r_all = []

    # Statistics tracking
    stats_by_redshift = []

    for snapshot, z_actual in zip(snapshots, actual_redshifts):
        # Read Data
        mvir = read_hdf_from_model(model_dir, filename, snapshot, 'Mvir', hubble_h) * 1.0e10 / hubble_h
        stellar_mass = read_hdf_from_model(model_dir, filename, snapshot, 'StellarMass', hubble_h) * 1.0e10 / hubble_h
        bulge_mass = read_hdf_from_model(model_dir, filename, snapshot, 'BulgeMass', hubble_h) * 1.0e10 / hubble_h
        cold_gas = read_hdf_from_model(model_dir, filename, snapshot, 'ColdGas', hubble_h) * 1.0e10 / hubble_h
        hot_gas = read_hdf_from_model(model_dir, filename, snapshot, 'HotGas', hubble_h) * 1.0e10 / hubble_h

        # SAGE Radius is Comoving Mpc/h
        sage_disk_radius = read_hdf_from_model(model_dir, filename, snapshot, 'DiskRadius', hubble_h)
        bulge_radius = read_hdf_from_model(model_dir, filename, snapshot, 'BulgeRadius', hubble_h) 

        # --- UNIT CONVERSION (Comoving kpc) ---
        # 1. Mpc/h -> kpc/h: * 1000
        # 2. kpc/h -> kpc:   / h
        # 3. Disk Scale Length -> Half Mass: * 1.68
        r_disk_kpc_comov = sage_disk_radius * 1.68 * 1000.0 / hubble_h
        
        # Filter for Heatmap
        galaxy_type = read_hdf_from_model(model_dir, filename, snapshot, 'Type', hubble_h)
        has_mass = stellar_mass > 0
        
        # Collect data for heatmap (all galaxies, not just disks)
        valid_map = has_mass & (r_disk_kpc_comov > 0)
        if np.sum(valid_map) > 0:
            all_mvir.extend(mvir[valid_map])
            all_redshifts.extend([z_actual] * np.sum(valid_map))
            all_radii.extend(np.log10(r_disk_kpc_comov[valid_map]))

        # --- Statistics for Right Plot - THREE POPULATIONS ---
        M_ffb = calculate_ffb_threshold_mass(z_actual, hubble_h)

        # Read FFBRegime flag
        ffb_regime = read_hdf_from_model(model_dir, filename, snapshot, 'FFBRegime', hubble_h)

        # Population 1: Galaxies above FFB threshold mass
        above_threshold = mvir > M_ffb
        subset_above = has_mass & above_threshold
        w_above = np.where(subset_above)[0]
        if len(w_above) > 0:
            disk_valid = r_disk_kpc_comov[w_above] > 0
            if np.sum(disk_valid) > 0:
                disk_z_above.extend([z_actual] * np.sum(disk_valid))
                disk_r_above.extend(r_disk_kpc_comov[w_above][disk_valid])

        # Population 1b: Galaxies below FFB threshold mass
        below_threshold = mvir <= M_ffb
        subset_below = has_mass & below_threshold
        w_below = np.where(subset_below)[0]
        if len(w_below) > 0:
            disk_valid = r_disk_kpc_comov[w_below] > 0
            if np.sum(disk_valid) > 0:
                disk_z_below.extend([z_actual] * np.sum(disk_valid))
                disk_r_below.extend(r_disk_kpc_comov[w_below][disk_valid])

        # Population 2: Galaxies near FFB threshold mass (0.9-1.1x)
        near_threshold = (mvir > M_ffb * 0.9) & (mvir < M_ffb * 1.1)
        subset_threshold = has_mass & near_threshold
        w_threshold = np.where(subset_threshold)[0]
        if len(w_threshold) > 0:
            disk_valid = r_disk_kpc_comov[w_threshold] > 0
            if np.sum(disk_valid) > 0:
                disk_z_threshold.extend([z_actual] * np.sum(disk_valid))
                disk_r_threshold.extend(r_disk_kpc_comov[w_threshold][disk_valid])

        # Population 3: All galaxies (FFBRegime == 0 and FFBRegime == 1)
        all_galaxies = (ffb_regime == 0) | (ffb_regime == 1)
        subset_all = has_mass & all_galaxies
        w_all = np.where(subset_all)[0]
        if len(w_all) > 0:
            disk_valid = r_disk_kpc_comov[w_all] > 0
            if np.sum(disk_valid) > 0:
                disk_z_all.extend([z_actual] * np.sum(disk_valid))
                disk_r_all.extend(r_disk_kpc_comov[w_all][disk_valid])

        # --- Collect statistics for this redshift ---
        n_total = len(mvir)
        n_with_mass = np.sum(has_mass)
        n_ffb_regime_1 = np.sum(ffb_regime == 1)

        # --- DIAGNOSTIC: Disk size statistics by population ---
        # Population: Above FFB threshold
        above_mask = has_mass & above_threshold & (r_disk_kpc_comov > 0)
        r_above_this_z = r_disk_kpc_comov[above_mask] if np.sum(above_mask) > 0 else np.array([])
        mvir_above_this_z = mvir[above_mask] if np.sum(above_mask) > 0 else np.array([])

        # Population: Below FFB threshold
        below_threshold = mvir <= M_ffb
        below_mask = has_mass & below_threshold & (r_disk_kpc_comov > 0)
        r_below_this_z = r_disk_kpc_comov[below_mask] if np.sum(below_mask) > 0 else np.array([])
        mvir_below_this_z = mvir[below_mask] if np.sum(below_mask) > 0 else np.array([])

        # Population: All with valid disk
        all_valid_mask = has_mass & (r_disk_kpc_comov > 0)
        r_all_this_z = r_disk_kpc_comov[all_valid_mask] if np.sum(all_valid_mask) > 0 else np.array([])
        mvir_all_this_z = mvir[all_valid_mask] if np.sum(all_valid_mask) > 0 else np.array([])

        # Count galaxies with zero disk radius
        n_zero_disk_above = np.sum(has_mass & above_threshold & (r_disk_kpc_comov == 0))
        n_zero_disk_below = np.sum(has_mass & below_threshold & (r_disk_kpc_comov == 0))
        n_zero_disk_total = np.sum(has_mass & (r_disk_kpc_comov == 0))
        n_ffb_regime_0 = np.sum(ffb_regime == 0)
        n_ffb_regime_neg1 = np.sum(ffb_regime == -1)

        # --- DIAGNOSTIC: Bulge-to-total ratio ---
        # B/T = BulgeMass / StellarMass (where stellar mass > 0)
        bt_ratio = np.where(stellar_mass > 0, bulge_mass / stellar_mass, 0.0)
        bt_above = bt_ratio[above_mask] if np.sum(above_mask) > 0 else np.array([])
        bt_below = bt_ratio[below_mask] if np.sum(below_mask) > 0 else np.array([])

        # --- DIAGNOSTIC: Gas fractions ---
        # Cold gas fraction: f_cold = ColdGas / (ColdGas + StellarMass)
        total_baryons = cold_gas + stellar_mass
        f_cold = np.where(total_baryons > 0, cold_gas / total_baryons, 0.0)
        f_cold_above = f_cold[above_mask] if np.sum(above_mask) > 0 else np.array([])
        f_cold_below = f_cold[below_mask] if np.sum(below_mask) > 0 else np.array([])

        # --- DIAGNOSTIC: FFBRegime breakdown for above/below threshold ---
        n_above_in_ffb = np.sum(above_threshold & (ffb_regime == 1))
        n_above_not_ffb = np.sum(above_threshold & (ffb_regime == 0))
        n_below_in_ffb = np.sum(below_threshold & (ffb_regime == 1))
        n_below_not_ffb = np.sum(below_threshold & (ffb_regime == 0))

        # --- DIAGNOSTIC: Disk vs Bulge radius comparison ---
        r_bulge_kpc = bulge_radius * 1000.0 / hubble_h  # Convert to kpc (no 1.68 factor for bulge)
        r_bulge_above = r_bulge_kpc[above_mask] if np.sum(above_mask) > 0 else np.array([])
        r_bulge_below = r_bulge_kpc[below_mask] if np.sum(below_mask) > 0 else np.array([])

        # --- DIAGNOSTIC: Stellar mass comparison ---
        mstar_above = stellar_mass[above_mask] if np.sum(above_mask) > 0 else np.array([])
        mstar_below = stellar_mass[below_mask] if np.sum(below_mask) > 0 else np.array([])
        n_above_mass_threshold = np.sum(mvir > M_ffb)
        n_below_mass_threshold = np.sum(mvir <= M_ffb)
        n_at_threshold = np.sum(near_threshold)
        n_ffb_with_mass = np.sum(has_mass & (ffb_regime == 1))
        n_normal_with_mass = np.sum(has_mass & (ffb_regime == 0))

        stats_by_redshift.append({
            'z': z_actual,
            'M_ffb': M_ffb,
            'n_total': n_total,
            'n_with_mass': n_with_mass,
            'n_ffb_regime_1': n_ffb_regime_1,
            'n_ffb_regime_0': n_ffb_regime_0,
            'n_ffb_regime_neg1': n_ffb_regime_neg1,
            'n_above_mass_threshold': n_above_mass_threshold,
            'n_below_mass_threshold': n_below_mass_threshold,
            'n_at_threshold': n_at_threshold,
            'n_ffb_with_mass': n_ffb_with_mass,
            'n_normal_with_mass': n_normal_with_mass,
            # Diagnostic: disk size stats
            'n_above_valid_disk': len(r_above_this_z),
            'n_below_valid_disk': len(r_below_this_z),
            'n_all_valid_disk': len(r_all_this_z),
            'n_zero_disk_above': n_zero_disk_above,
            'n_zero_disk_below': n_zero_disk_below,
            'n_zero_disk_total': n_zero_disk_total,
            'median_r_above': np.median(r_above_this_z) if len(r_above_this_z) > 0 else np.nan,
            'median_r_below': np.median(r_below_this_z) if len(r_below_this_z) > 0 else np.nan,
            'median_r_all': np.median(r_all_this_z) if len(r_all_this_z) > 0 else np.nan,
            'median_mvir_above': np.median(mvir_above_this_z) if len(mvir_above_this_z) > 0 else np.nan,
            'median_mvir_below': np.median(mvir_below_this_z) if len(mvir_below_this_z) > 0 else np.nan,
            'median_mvir_all': np.median(mvir_all_this_z) if len(mvir_all_this_z) > 0 else np.nan,
            # Diagnostic: Bulge-to-total ratio
            'median_bt_above': np.median(bt_above) if len(bt_above) > 0 else np.nan,
            'median_bt_below': np.median(bt_below) if len(bt_below) > 0 else np.nan,
            # Diagnostic: Cold gas fraction
            'median_fcold_above': np.median(f_cold_above) if len(f_cold_above) > 0 else np.nan,
            'median_fcold_below': np.median(f_cold_below) if len(f_cold_below) > 0 else np.nan,
            # Diagnostic: FFBRegime breakdown
            'n_above_in_ffb': n_above_in_ffb,
            'n_above_not_ffb': n_above_not_ffb,
            'n_below_in_ffb': n_below_in_ffb,
            'n_below_not_ffb': n_below_not_ffb,
            # Diagnostic: Bulge radius
            'median_rbulge_above': np.median(r_bulge_above) if len(r_bulge_above) > 0 else np.nan,
            'median_rbulge_below': np.median(r_bulge_below) if len(r_bulge_below) > 0 else np.nan,
            # Diagnostic: Stellar mass
            'median_mstar_above': np.median(mstar_above) if len(mstar_above) > 0 else np.nan,
            'median_mstar_below': np.median(mstar_below) if len(mstar_below) > 0 else np.nan,
        })

    # ================= PLOTTING =================
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # --- LEFT PANEL: Heatmap ---
    ax_left = axes[0]
    
    if len(all_mvir) > 0:
        a_mvir = np.array(all_mvir)
        a_z = np.array(all_redshifts)
        a_radii = np.array(all_radii)  # already log10
        
        # Define bins
        z_bins = np.linspace(5, 20, 60)
        m_bins = np.linspace(9, 13, 60)
        
        # 1. Bin statistics
        from scipy.stats import binned_statistic_2d
        median_radius, z_edges, mvir_edges, _ = binned_statistic_2d(
            a_z, np.log10(a_mvir), a_radii,
            statistic='median', bins=[z_bins, m_bins]
        )
        
        # 2. Smoothing
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
        cbar.set_label(r'$\log_{10} R_{\rm \frac{1}{2}}$ (kpc)', fontsize=12)

    # FFB Threshold Line
    z_line = np.linspace(5, 20, 100)
    m_ffb = [calculate_ffb_threshold_mass(z, hubble_h) for z in z_line]
    ax_left.plot(z_line, np.log10(m_ffb), 'k-', lw=3, label='FFB Threshold')
    
    ax_left.set_xlabel('Redshift z', fontsize=14)
    ax_left.set_ylabel(r'$\log_{10} M_{\rm vir}$', fontsize=14)
    ax_left.set_xlim(5, 16)
    ax_left.set_ylim(9, 13)
    
    # Set integer ticks with minor ticks at 20% intervals
    from matplotlib.ticker import MultipleLocator
    ax_left.xaxis.set_major_locator(MultipleLocator(1))
    ax_left.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax_left.yaxis.set_major_locator(MultipleLocator(1))
    ax_left.yaxis.set_minor_locator(MultipleLocator(0.2))
    ax_left.legend()

    # --- RIGHT PANEL: Evolution ---
    ax_right = axes[1]

    # Helper function to calculate binned statistics
    def calculate_binned_stats(z_arr, r_arr, z_bin_edges):
        """Calculate median and percentiles in redshift bins"""
        z_bin_centers = (z_bin_edges[:-1] + z_bin_edges[1:]) / 2
        z_plot = []
        r_median_plot = []
        r_lower_plot = []
        r_upper_plot = []
        
        for i in range(len(z_bin_edges) - 1):
            in_bin = (z_arr >= z_bin_edges[i]) & (z_arr < z_bin_edges[i+1])
            
            if np.sum(in_bin) > 10:
                r_in_bin = r_arr[in_bin]
                median_r = np.median(r_in_bin)
                lower_r = np.percentile(r_in_bin, 16)
                upper_r = np.percentile(r_in_bin, 84)
                
                z_plot.append(z_bin_centers[i])
                r_median_plot.append(median_r)
                r_lower_plot.append(lower_r)
                r_upper_plot.append(upper_r)
        
        return (np.array(z_plot), np.array(r_median_plot), 
                np.array(r_lower_plot), np.array(r_upper_plot))
    
    # Define redshift bins (shared for all populations)
    z_bin_edges = np.arange(1, 20.5, 0.5)
    
    # Plot 1: All galaxies (gray, lightest)
    if len(disk_z_all) > 0:
        z_all, r_all, r_all_lower, r_all_upper = calculate_binned_stats(
            np.array(disk_z_all), np.array(disk_r_all), z_bin_edges)
        
        ax_right.plot(z_all, r_all,
                     color='gray',
                     linestyle='-',
                     linewidth=2,
                     label='All Galaxies',
                     alpha=0.8,
                     zorder=1)
        
        ax_right.fill_between(z_all, r_all_lower, r_all_upper,
                             color='gray',
                             alpha=0.2,
                             zorder=0)
    
    # Plot 2: Galaxies above FFB threshold mass (orange, medium)
    if len(disk_z_above) > 0:
        z_above, r_above, r_above_lower, r_above_upper = calculate_binned_stats(
            np.array(disk_z_above), np.array(disk_r_above), z_bin_edges)

        ax_right.plot(z_above, r_above,
                     color='darkorange',
                     linestyle='-',
                     linewidth=2.5,
                     label=r'$M_{\rm vir} > M_{\rm FFB}$',
                     alpha=0.6,
                     marker='o',
                     markersize=4,
                     zorder=2)
        
        ax_right.fill_between(z_above, r_above_lower, r_above_upper,
                             color='darkorange',
                             alpha=0.3,
                             zorder=1)
    
    # Plot 3: Galaxies at threshold (red, darkest/most prominent)
    if len(disk_z_threshold) > 0:
        z_thresh, r_thresh, r_thresh_lower, r_thresh_upper = calculate_binned_stats(
            np.array(disk_z_threshold), np.array(disk_r_threshold), z_bin_edges)
        
        ax_right.plot(z_thresh, r_thresh,
                     color='dodgerblue',
                     linestyle='-',
                     linewidth=3,
                     label='At FFB Threshold (±10%)',
                     alpha=0.7,
                     marker='s',
                     markersize=5,
                     zorder=3)
        
        ax_right.fill_between(z_thresh, r_thresh_lower, r_thresh_upper,
                             color='dodgerblue',
                             alpha=0.25,
                             zorder=2)

    # Theory lines
    z_t = np.linspace(5, 20, 100)
    z_10 = (1 + z_t) / 10.0
    r_th_disk = 0.31 * (z_10 ** -3.07)
    r_th_disk_2 = 2 * (0.31 * (z_10 ** -3.07))

    ax_right.plot(z_t, r_th_disk, color='darkorange', ls='--', lw=2, label='Li+24 (Disk)', zorder=4)
    ax_right.plot(z_t, r_th_disk_2, color='darkred', ls=':', lw=2, label='Li+24 (2*Disk)', zorder=4)

    ax_right.set_yscale('log')
    ax_right.set_xlabel('Redshift z', fontsize=14)
    ax_right.set_ylabel(r'$R_{\rm \frac{1}{2}}$ (kpc)', fontsize=14)
    ax_right.set_xlim(5, 13)
    
    # Set integer x-ticks with minor ticks at 20% intervals
    from matplotlib.ticker import MultipleLocator
    ax_right.xaxis.set_major_locator(MultipleLocator(1))
    ax_right.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax_right.yaxis.set_major_formatter(plt.ScalarFormatter())
    # ax_right.grid(True, alpha=0.1)
    ax_right.legend(frameon=False, loc='lower right')

    # --- Add Baggen+2023 observational data ---
    z_baggen, re_baggen, re_err_plus_baggen, re_err_minus_baggen = load_baggen2023_data()
    if z_baggen is not None:
        ax_right.errorbar(z_baggen, re_baggen, 
                         yerr=[re_err_minus_baggen, re_err_plus_baggen],
                         fmt='^', color='black', markersize=8, alpha=0.8,
                         label='Baggen+23', capsize=2, linewidth=1.5, zorder=5)

    # --- Add Casey+2024 observational data ---
    z_casey, re_casey, re_err_plus_casey, re_err_minus_casey = load_casey2024_data()
    if z_casey is not None:
        ax_right.errorbar(z_casey, re_casey, 
                         yerr=[re_err_minus_casey, re_err_plus_casey],
                         fmt='s', color='black', markersize=8, alpha=0.8,
                         label='Casey+24', capsize=2, linewidth=1.5, zorder=5)

    # --- Add Sun+2024 observational data ---
    z_sun, re_sun, re_err_plus_sun, re_err_minus_sun = load_sun2024_data()
    if z_sun is not None:
        ax_right.errorbar(z_sun, re_sun, 
                         yerr=[re_err_minus_sun, re_err_plus_sun],
                         fmt='d', color='black', markersize=8, alpha=0.8,
                         label='Sun+24', capsize=2, linewidth=1.5, zorder=5)
        
    # --- Add Finkelstein+2023 observational data ---
    z_finkelstein, re_finkelstein, re_err_plus_finkelstein, re_err_minus_finkelstein = load_finkelstein2023_data()
    if z_finkelstein is not None:
        ax_right.errorbar(z_finkelstein, re_finkelstein, 
                         yerr=[re_err_minus_finkelstein, re_err_plus_finkelstein],
                         fmt='o', color='black', markersize=4, alpha=0.8,
                         label='Finkelstein+23', capsize=2, linewidth=1.5, zorder=5)
        
    # ax_right.legend(frameon=False, loc='lower right')

    plt.tight_layout()
    out_path = DirName + 'plots/ffb_threshold_analysis_empirical_all' + OutputFormat
    plt.savefig(out_path, dpi=300)
    print(f'Plot saved to: {out_path}')
    plt.close()

    # ================= STATISTICS =================
    print('\n' + '-'*100)
    print('GALAXY STATISTICS BY REDSHIFT')
    print('-'*100)
    print(f'{"z":>6} {"M_ffb":>12} {"N_total":>10} {"N_mass>0":>10} {"FFB=1":>8} {"FFB=0":>8} {"FFB=-1":>8} {"M>M_ffb":>10} {"M<=M_ffb":>10} {"At thresh":>10} {"f_above":>8}')
    print('-'*100)

    totals = {
        'n_total': 0,
        'n_with_mass': 0,
        'n_ffb_regime_1': 0,
        'n_ffb_regime_0': 0,
        'n_ffb_regime_neg1': 0,
        'n_above_mass_threshold': 0,
        'n_below_mass_threshold': 0,
        'n_at_threshold': 0,
        'n_ffb_with_mass': 0,
        'n_normal_with_mass': 0,
    }

    for s in stats_by_redshift:
        # Calculate fraction of halos above FFB threshold
        f_above = 100.0 * s["n_above_mass_threshold"] / s["n_total"] if s["n_total"] > 0 else 0.0
        print(f'{s["z"]:>6.2f} {s["M_ffb"]:>12.2e} {s["n_total"]:>10} {s["n_with_mass"]:>10} '
              f'{s["n_ffb_regime_1"]:>8} {s["n_ffb_regime_0"]:>8} {s["n_ffb_regime_neg1"]:>8} '
              f'{s["n_above_mass_threshold"]:>10} {s["n_below_mass_threshold"]:>10} {s["n_at_threshold"]:>10} {f_above:>7.1f}%')
        for key in totals:
            totals[key] += s[key]

    print('-'*100)
    total_f_above = 100.0 * totals["n_above_mass_threshold"] / totals["n_total"] if totals["n_total"] > 0 else 0.0
    print(f'{"TOTAL":>6} {"":>12} {totals["n_total"]:>10} {totals["n_with_mass"]:>10} '
          f'{totals["n_ffb_regime_1"]:>8} {totals["n_ffb_regime_0"]:>8} {totals["n_ffb_regime_neg1"]:>8} '
          f'{totals["n_above_mass_threshold"]:>10} {totals["n_below_mass_threshold"]:>10} {totals["n_at_threshold"]:>10} {total_f_above:>7.1f}%')

    print('\n' + '-'*60)
    print('SUMMARY')
    print('-'*60)
    print(f'Total galaxies across all snapshots:        {totals["n_total"]:>10}')
    print(f'Galaxies with stellar mass > 0:             {totals["n_with_mass"]:>10}')
    print(f'FFB regime galaxies (FFBRegime=1):          {totals["n_ffb_regime_1"]:>10}')
    print(f'Normal regime galaxies (FFBRegime=0):       {totals["n_ffb_regime_0"]:>10}')
    print(f'Unassigned galaxies (FFBRegime=-1):         {totals["n_ffb_regime_neg1"]:>10}')
    print(f'Galaxies above M_ffb threshold:             {totals["n_above_mass_threshold"]:>10}')
    print(f'Fraction of halos above M_ffb threshold:    {total_f_above:>9.1f}%')
    print(f'Galaxies at threshold (0.9-1.1x M_ffb):     {totals["n_at_threshold"]:>10}')
    print(f'FFB galaxies with stellar mass:             {totals["n_ffb_with_mass"]:>10}')
    print(f'Normal galaxies with stellar mass:          {totals["n_normal_with_mass"]:>10}')

    if totals["n_with_mass"] > 0:
        ffb_frac = 100.0 * totals["n_ffb_with_mass"] / totals["n_with_mass"]
        print(f'\nFraction of galaxies (with mass) in FFB regime: {ffb_frac:.1f}%')

    print('='*60 + '\n')

    # ================= DIAGNOSTIC: Disk size comparison =================
    print('\n' + '='*120)
    print('DIAGNOSTIC: DISK SIZE COMPARISON BY POPULATION')
    print('='*120)
    print(f'{"z":>6} {"N_above":>10} {"N_below":>10} {"N_all":>10} | '
          f'{"R_med_above":>12} {"R_med_below":>12} {"R_med_all":>12} | '
          f'{"Mvir_above":>12} {"Mvir_below":>12} {"Mvir_all":>12} | '
          f'{"zero_above":>10} {"zero_below":>10}')
    print('-'*120)

    for s in stats_by_redshift:
        print(f'{s["z"]:>6.2f} {s["n_above_valid_disk"]:>10} {s["n_below_valid_disk"]:>10} {s["n_all_valid_disk"]:>10} | '
              f'{s["median_r_above"]:>12.4f} {s["median_r_below"]:>12.4f} {s["median_r_all"]:>12.4f} | '
              f'{s["median_mvir_above"]:>12.2e} {s["median_mvir_below"]:>12.2e} {s["median_mvir_all"]:>12.2e} | '
              f'{s["n_zero_disk_above"]:>10} {s["n_zero_disk_below"]:>10}')

    print('-'*120)
    print('\nKEY QUESTIONS:')
    print('  1. Is R_med_above > R_med_below? (Expected if disk size scales with halo mass)')
    print('  2. Is R_med_all closer to R_med_below? (Expected if most galaxies are below threshold)')
    print('  3. Are there many zero-disk galaxies being excluded?')
    print('  4. What fraction of below-threshold galaxies have valid disks?')
    print('='*120 + '\n')

    # ================= DIAGNOSTIC: Bulge-to-Total and Gas Fractions =================
    print('\n' + '='*140)
    print('DIAGNOSTIC: BULGE-TO-TOTAL RATIO AND COLD GAS FRACTION')
    print('='*140)
    print(f'{"z":>6} | {"B/T_above":>10} {"B/T_below":>10} | '
          f'{"f_cold_above":>12} {"f_cold_below":>12} | '
          f'{"R_bulge_above":>14} {"R_bulge_below":>14} | '
          f'{"M*_above":>12} {"M*_below":>12}')
    print('-'*140)

    for s in stats_by_redshift:
        bt_above_str = f'{s["median_bt_above"]:>10.3f}' if not np.isnan(s["median_bt_above"]) else f'{"nan":>10}'
        bt_below_str = f'{s["median_bt_below"]:>10.3f}' if not np.isnan(s["median_bt_below"]) else f'{"nan":>10}'
        fc_above_str = f'{s["median_fcold_above"]:>12.3f}' if not np.isnan(s["median_fcold_above"]) else f'{"nan":>12}'
        fc_below_str = f'{s["median_fcold_below"]:>12.3f}' if not np.isnan(s["median_fcold_below"]) else f'{"nan":>12}'
        rb_above_str = f'{s["median_rbulge_above"]:>14.4f}' if not np.isnan(s["median_rbulge_above"]) else f'{"nan":>14}'
        rb_below_str = f'{s["median_rbulge_below"]:>14.4f}' if not np.isnan(s["median_rbulge_below"]) else f'{"nan":>14}'
        ms_above_str = f'{s["median_mstar_above"]:>12.2e}' if not np.isnan(s["median_mstar_above"]) else f'{"nan":>12}'
        ms_below_str = f'{s["median_mstar_below"]:>12.2e}' if not np.isnan(s["median_mstar_below"]) else f'{"nan":>12}'

        print(f'{s["z"]:>6.2f} | {bt_above_str} {bt_below_str} | '
              f'{fc_above_str} {fc_below_str} | '
              f'{rb_above_str} {rb_below_str} | '
              f'{ms_above_str} {ms_below_str}')

    print('-'*140)
    print('\nINTERPRETATION:')
    print('  - B/T: Higher = more bulge-dominated (mergers, instabilities)')
    print('  - f_cold: Cold gas fraction. Lower = less gas available for disk growth')
    print('  - R_bulge: Bulge radius. Compare to disk radius.')
    print('  - M*: Stellar mass. Above-threshold should be more massive.')
    print('='*140 + '\n')

    # ================= DIAGNOSTIC: FFBRegime breakdown =================
    print('\n' + '='*100)
    print('DIAGNOSTIC: FFBRegime FLAG vs MASS THRESHOLD')
    print('='*100)
    print(f'{"z":>6} | {"Above threshold":>40} | {"Below threshold":>40}')
    print(f'{"":>6} | {"FFB=1":>12} {"FFB=0":>12} {"%FFB":>12} | {"FFB=1":>12} {"FFB=0":>12} {"%FFB":>12}')
    print('-'*100)

    for s in stats_by_redshift:
        n_above_total = s["n_above_in_ffb"] + s["n_above_not_ffb"]
        n_below_total = s["n_below_in_ffb"] + s["n_below_not_ffb"]
        pct_above_ffb = 100.0 * s["n_above_in_ffb"] / n_above_total if n_above_total > 0 else 0.0
        pct_below_ffb = 100.0 * s["n_below_in_ffb"] / n_below_total if n_below_total > 0 else 0.0

        print(f'{s["z"]:>6.2f} | {s["n_above_in_ffb"]:>12} {s["n_above_not_ffb"]:>12} {pct_above_ffb:>11.1f}% | '
              f'{s["n_below_in_ffb"]:>12} {s["n_below_not_ffb"]:>12} {pct_below_ffb:>11.1f}%')

    print('-'*100)
    print('\nINTERPRETATION:')
    print('  - "Above threshold" = halos with Mvir > M_ffb at this redshift')
    print('  - "Below threshold" = halos with Mvir <= M_ffb at this redshift')
    print('  - %FFB = fraction of that population flagged as FFBRegime=1')
    print('  - If %FFB is high for "below threshold", these halos were above threshold earlier')
    print('='*100 + '\n')

    # ================= PLOT: FFB vs All galaxy counts by redshift =================
    fig_hist, ax_hist = plt.subplots(figsize=(10, 6))

    # Extract data for plotting
    z_plot = np.array([s['z'] for s in stats_by_redshift])
    n_total_plot = np.array([s['n_total'] for s in stats_by_redshift])
    n_ffb_plot = np.array([s['n_ffb_regime_1'] for s in stats_by_redshift])

    # Bar width
    bar_width = 0.35

    # Create bar positions
    x_pos = np.arange(len(z_plot))

    # Plot bars
    bars_all = ax_hist.bar(x_pos - bar_width/2, n_total_plot, bar_width,
                           color='gray', alpha=0.7, label='All Halos')
    bars_ffb = ax_hist.bar(x_pos + bar_width/2, n_ffb_plot, bar_width,
                           color='darkorange', alpha=0.8, label='FFB Galaxies')

    # Add percentage labels on FFB bars
    for i, (n_ffb, n_total) in enumerate(zip(n_ffb_plot, n_total_plot)):
        if n_total > 0:
            pct = 100.0 * n_ffb / n_total
            ax_hist.text(x_pos[i] + bar_width/2, n_ffb + max(n_total_plot)*0.01,
                        f'{pct:.1f}%', ha='center', va='bottom', fontsize=8, rotation=90)

    ax_hist.set_xlabel('Redshift z', fontsize=14)
    ax_hist.set_ylabel('Number of Halos', fontsize=14)
    ax_hist.set_yscale('log')

    # Set x-tick labels to redshift values
    ax_hist.set_xticks(x_pos)
    ax_hist.set_xticklabels([f'{z:.1f}' for z in z_plot], rotation=45, ha='right')

    ax_hist.legend(frameon=False, loc='upper right', fontsize=12)
    ax_hist.set_title('FFB Galaxy Counts by Redshift', fontsize=14)

    plt.tight_layout()
    frac_out_path = DirName + 'plots/ffb_counts_by_redshift' + OutputFormat
    plt.savefig(frac_out_path, dpi=300)
    print(f'FFB counts plot saved to: {frac_out_path}')
    plt.close()

    # ================= PLOT: Disk sizes for above vs below threshold =================
    fig_disk, ax_disk = plt.subplots(figsize=(10, 7))

    # Helper function to calculate binned statistics (reuse from earlier)
    def calculate_binned_stats_local(z_arr, r_arr, z_bin_edges):
        """Calculate median and percentiles in redshift bins"""
        z_bin_centers = (z_bin_edges[:-1] + z_bin_edges[1:]) / 2
        z_plot_out = []
        r_median_plot = []
        r_lower_plot = []
        r_upper_plot = []

        for i in range(len(z_bin_edges) - 1):
            in_bin = (z_arr >= z_bin_edges[i]) & (z_arr < z_bin_edges[i+1])

            if np.sum(in_bin) > 5:  # Need at least 5 galaxies
                r_in_bin = r_arr[in_bin]
                median_r = np.median(r_in_bin)
                lower_r = np.percentile(r_in_bin, 16)
                upper_r = np.percentile(r_in_bin, 84)

                z_plot_out.append(z_bin_centers[i])
                r_median_plot.append(median_r)
                r_lower_plot.append(lower_r)
                r_upper_plot.append(upper_r)

        return (np.array(z_plot_out), np.array(r_median_plot),
                np.array(r_lower_plot), np.array(r_upper_plot))

    # Define redshift bins
    z_bin_edges = np.arange(4, 21, 0.5)

    # Plot above-threshold galaxies (orange)
    if len(disk_z_above) > 0:
        z_above_arr, r_above_arr, r_above_lower, r_above_upper = calculate_binned_stats_local(
            np.array(disk_z_above), np.array(disk_r_above), z_bin_edges)

        if len(z_above_arr) > 0:
            ax_disk.plot(z_above_arr, r_above_arr,
                        color='darkorange', linestyle='-', linewidth=2.5,
                        label=r'Above threshold ($M_{\rm vir} > M_{\rm FFB}$)',
                        marker='o', markersize=6, zorder=3)
            ax_disk.fill_between(z_above_arr, r_above_lower, r_above_upper,
                                color='darkorange', alpha=0.3, zorder=2)

    # Plot below-threshold galaxies (blue)
    if len(disk_z_below) > 0:
        z_below_arr, r_below_arr, r_below_lower, r_below_upper = calculate_binned_stats_local(
            np.array(disk_z_below), np.array(disk_r_below), z_bin_edges)

        if len(z_below_arr) > 0:
            ax_disk.plot(z_below_arr, r_below_arr,
                        color='dodgerblue', linestyle='-', linewidth=2.5,
                        label=r'Below threshold ($M_{\rm vir} \leq M_{\rm FFB}$)',
                        marker='s', markersize=6, zorder=3)
            ax_disk.fill_between(z_below_arr, r_below_lower, r_below_upper,
                                color='dodgerblue', alpha=0.3, zorder=2)

    # Plot all galaxies (gray, for reference)
    if len(disk_z_all) > 0:
        z_all_arr, r_all_arr, r_all_lower, r_all_upper = calculate_binned_stats_local(
            np.array(disk_z_all), np.array(disk_r_all), z_bin_edges)

        if len(z_all_arr) > 0:
            ax_disk.plot(z_all_arr, r_all_arr,
                        color='gray', linestyle='--', linewidth=2,
                        label='All galaxies', alpha=0.7, zorder=1)

    # Add theory line from Li+24
    z_t = np.linspace(5, 20, 100)
    z_10 = (1 + z_t) / 10.0
    r_th_disk = 0.31 * (z_10 ** -3.07)
    ax_disk.plot(z_t, r_th_disk, color='darkred', ls=':', lw=2, label='Li+24 (Disk)', zorder=4)

    ax_disk.set_yscale('log')
    ax_disk.set_xlabel('Redshift z', fontsize=14)
    ax_disk.set_ylabel(r'$R_{\rm 1/2}$ (kpc)', fontsize=14)
    ax_disk.set_xlim(5, 16)
    ax_disk.set_ylim(0.1, 10)

    # Set integer x-ticks
    from matplotlib.ticker import MultipleLocator
    ax_disk.xaxis.set_major_locator(MultipleLocator(1))
    ax_disk.xaxis.set_minor_locator(MultipleLocator(0.2))

    ax_disk.legend(frameon=False, loc='upper right', fontsize=11)
    ax_disk.set_title('Disk Size: Above vs Below FFB Threshold', fontsize=14)

    plt.tight_layout()
    disk_compare_path = DirName + 'plots/ffb_disk_size_above_vs_below' + OutputFormat
    plt.savefig(disk_compare_path, dpi=300)
    print(f'Disk size comparison plot saved to: {disk_compare_path}')
    plt.close()


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
        
        # Set integer ticks with minor ticks at 20% intervals
        from matplotlib.ticker import MultipleLocator
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.2))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.2))
        
        ax.legend(loc='lower left', fontsize=10, frameon=False, title='FFB Model Spread (16-84%)', title_fontsize=9)
        
        # Add analytical predictions from Li+2023 (shell mode with 3 epsilon values)
        try:
            redshifts_analytical = np.linspace(5, 20, 50)
            
            # FFB eps_max=1.0 (shell)
            set_option(FFB_SFE_MAX=1.0)
            fgas_ffb1_shell = []
            for z in redshifts_analytical:
                lgMh_crit = ffb_lgMh_crit(z)
                fgas = ffb_fgas(lgMh_crit, z, mode='shell', eps=None)
                fgas_ffb1_shell.append(np.log10(fgas) if fgas > 0 else np.nan)
            valid = ~np.isnan(fgas_ffb1_shell)
            if np.sum(valid) > 0:
                ax.plot(redshifts_analytical[valid], np.array(fgas_ffb1_shell)[valid],
                       color='dodgerblue', linestyle='--', linewidth=2,
                       label='Li+ 2024', alpha=0.7, zorder=4)
            
            # FFB eps_max=0.2 (shell)
            set_option(FFB_SFE_MAX=0.2)
            fgas_ffb02_shell = []
            for z in redshifts_analytical:
                lgMh_crit = ffb_lgMh_crit(z)
                fgas = ffb_fgas(lgMh_crit, z, mode='shell', eps=None)
                fgas_ffb02_shell.append(np.log10(fgas) if fgas > 0 else np.nan)
            valid = ~np.isnan(fgas_ffb02_shell)
            if np.sum(valid) > 0:
                ax.plot(redshifts_analytical[valid], np.array(fgas_ffb02_shell)[valid],
                       color='orange', linestyle='--', linewidth=2,
                       label='', alpha=0.7, zorder=4)
            
            # UM model - use disc mode
            set_option(FFB_SFE_MAX=0.0)
            fgas_um_disc = []
            for z in redshifts_analytical:
                lgMh_crit = ffb_lgMh_crit(z)
                fgas = ffb_fgas(lgMh_crit, z, mode='disc', eps=None)
                fgas_um_disc.append(np.log10(fgas) if fgas > 0 else np.nan)
            valid = ~np.isnan(fgas_um_disc)
            if np.sum(valid) > 0:
                ax.plot(redshifts_analytical[valid], np.array(fgas_um_disc)[valid],
                       color='gray', linestyle='--', linewidth=2,
                       label='', alpha=0.7, zorder=4)
            
            # Reset to default
            set_option(FFB_SFE_MAX=1.0)
            print(f"  Li+2023 analytical gas fraction predictions added (3 lines)")
        except Exception as e:
            print(f"  Warning: Could not compute analytical gas fraction: {e}")
    
    # Save figure
    OutputDir = DirName + 'plots/'
    if not os.path.exists(OutputDir):
        os.makedirs(OutputDir)
    
    output_path = OutputDir + 'gas_fraction_evolution' + OutputFormat
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'\nPlot saved to: {output_path}')
    plt.close()
    
    print('='*60 + '\n')


def load_baggen2023_data():
    """Load Baggen+2023 disk size data."""
    filename = './data/baggen_disk_2023.ecsv'
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None, None, None, None
    
    try:
        table = Table.read(filename, format='ascii.ecsv')
        z = table['z_phot']
        # Convert re from physical pc to comoving kpc
        re_kpc_phys = table['re'] / 1000.0
        re_err_plus_kpc_phys = table['re_err_plus'] / 1000.0
        re_err_minus_kpc_phys = table['re_err_minus'] / 1000.0
        
        return z, re_kpc_phys, re_err_plus_kpc_phys, re_err_minus_kpc_phys
    except Exception as e:
        print(f"Error loading Baggen+2023 data: {e}")
    return None, None, None, None

def load_casey2024_data():
    """Load Casey+2024 disk size data."""
    filename = './data/casey_disk_2024.ecsv'
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None, None, None, None
    
    try:
        table = Table.read(filename, format='ascii.ecsv')
        z = table['z_phot_BAGPIPES']
        # Convert re from physical pc to comoving kpc
        re_kpc_phys = table['Reff_GALFIT'] / 1000.0
        re_err_plus_kpc_phys = table['Reff_GALFIT_err'] / 1000.0
        re_err_minus_kpc_phys = table['Reff_GALFIT_err'] / 1000.0
        
        return z, re_kpc_phys, re_err_plus_kpc_phys, re_err_minus_kpc_phys
    except Exception as e:
        print(f"Error loading Casey+2024 data: {e}")
    return None, None, None, None

def load_sun2024_data():
    """Load Sun+2024 disk size data."""
    filename = './data/sun_disk_2024.ecsv'
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None, None, None, None
    
    try:
        table = Table.read(filename, format='ascii.ecsv')
        z = table['z']

        re_kpc_phys = table['Re']
        re_err_plus_kpc_phys = table['Re_err']
        re_err_minus_kpc_phys = table['Re_err']
        
        return z, re_kpc_phys, re_err_plus_kpc_phys, re_err_minus_kpc_phys
    except Exception as e:
        print(f"Error loading Sun+2024 data: {e}")
    return None, None, None, None

def load_finkelstein2023_data():
    """Load Finkelstein+2023 disk size data."""
    filename = './data/finkelstein_disk_2023.ecsv'
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None, None, None, None
    
    try:
        table = Table.read(filename, format='ascii.ecsv')
        z = table['z_phot']

        re_kpc_phys = table['Rh_phys_kpc']
        re_err_plus_kpc_phys = table['Rh_phys_err_kpc']
        re_err_minus_kpc_phys = table['Rh_phys_err_kpc']
        
        return z, re_kpc_phys, re_err_plus_kpc_phys, re_err_minus_kpc_phys
    except Exception as e:
        print(f"Error loading Finkelstein+2023 data: {e}")
    return None, None, None, None

def load_madau_dickinson_2014_data():
    """Load Madau and Dickinson 2014 SFRD data."""
    filename = './data/MandD_sfrd_2014.ecsv'
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None, None, None, None
    
    try:
        table = Table.read(filename, format='ascii.ecsv')
        z = table['z_min']

        re_kpc_phys = table['log_psi']
        re_err_plus_kpc_phys = table['e_log_psi_up']
        re_err_minus_kpc_phys = table['e_log_psi_lo']
        
        return z, re_kpc_phys, re_err_plus_kpc_phys, re_err_minus_kpc_phys
    except Exception as e:
        print(f"Error loading Madau and Dickinson 2014 data: {e}")
    return None, None, None, None

def load_madau_dickinson_smd_2014_data():
    """Load Madau and Dickinson 2014 SMD data."""
    filename = './data/MandD_smd_2014.ecsv'
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None, None, None, None
    
    try:
        table = Table.read(filename, format='ascii.ecsv')
        z = table['z_min']

        re_kpc_phys = table['log_rho']
        re_err_plus_kpc_phys = table['e_log_rho_up']
        re_err_minus_kpc_phys = table['e_log_rho_lo']
        
        return z, re_kpc_phys, re_err_plus_kpc_phys, re_err_minus_kpc_phys
    except Exception as e:
        print(f"Error loading Madau and Dickinson 2014 data: {e}")
    return None, None, None, None

def load_kikuchihara_smd_2020_data():
    """Load Kikuchihara et al. 2020 SMD data."""
    filename = './data/kikuchihara_smd_2020.ecsv'
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None, None, None, None
    
    try:
        table = Table.read(filename, format='ascii.ecsv')
        z = table['z']

        re_kpc_phys = table['log_rho_star']
        re_err_plus_kpc_phys = table['e_log_rho_star_upper']
        re_err_minus_kpc_phys = table['e_log_rho_star_lower']
        
        return z, re_kpc_phys, re_err_plus_kpc_phys, re_err_minus_kpc_phys
    except Exception as e:
        print(f"Error loading Madau and Dickinson 2014 data: {e}")
    return None, None, None, None

def load_mcleod_rho_uv_2016_data():
    """Load Mcleod et al. 2016 UV density data."""
    filename = './data/mcleod_rhouv_2016.ecsv'
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None, None, None, None
    
    try:
        table = Table.read(filename, format='ascii.ecsv')
        z = table['z']

        re_kpc_phys = table['rho_uv']
        re_err_plus_kpc_phys = table['rho_uv_err_up']
        re_err_minus_kpc_phys = table['rho_uv_err_low']
        
        return z, re_kpc_phys, re_err_plus_kpc_phys, re_err_minus_kpc_phys
    except Exception as e:
        print(f"Error loading Madau and Dickinson 2014 data: {e}")
    return None, None, None, None

def load_oesch_sfrd_2018_data():
    """Load Oesch et al. 2018 SFRD data."""
    filename = './data/oesch_sfrd_2018.ecsv'
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None, None, None, None
    
    try:
        table = Table.read(filename, format='ascii.ecsv')
        z = table['z']

        re_kpc_phys = table['log_rho_sfr']
        re_err_plus_kpc_phys = table['e_log_rho_sfr_upper']
        re_err_minus_kpc_phys = table['e_log_rho_sfr_lower']
        
        return z, re_kpc_phys, re_err_plus_kpc_phys, re_err_minus_kpc_phys
    except Exception as e:
        print(f"Error loading Madau and Dickinson 2014 data: {e}")
    return None, None, None, None

def load_papovich_smd_2023_data():
    """Load Papovich et al. 2023 SMDdata."""
    filename = './data/papovich_smd_2023.ecsv'
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None, None, None, None
    
    try:
        table = Table.read(filename, format='ascii.ecsv')
        z = table['z']

        re_kpc_phys = table['log_rho_star']
        re_err_plus_kpc_phys = table['e_log_rho_star_upper']
        re_err_minus_kpc_phys = table['e_log_rho_star_lower']
        
        return z, re_kpc_phys, re_err_plus_kpc_phys, re_err_minus_kpc_phys
    except Exception as e:
        print(f"Error loading Madau and Dickinson 2014 data: {e}")
    return None, None, None, None

def load_mcleod_rho_uv_2024_data():
    """Load Mcleod et al. 2024 UV density data."""
    filename = './data/mcleod_rhouv_2024.ecsv'
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None, None, None, None
    
    try:
        table = Table.read(filename, format='ascii.ecsv')
        z = table['z']

        re_kpc_phys = table['log_rho_uv']
        re_err_plus_kpc_phys = table['e_log_rho_uv_upper']
        re_err_minus_kpc_phys = table['e_log_rho_uv_lower']
        
        return z, re_kpc_phys, re_err_plus_kpc_phys, re_err_minus_kpc_phys
    except Exception as e:
        print(f"Error loading Madau and Dickinson 2014 data: {e}")
    return None, None, None, None

def load_mcleod_rho_sfr_2024_data():
    """Load Mcleod et al. 2024 SFR density data."""
    filename = './data/mcleod_rhouv_2024.ecsv'
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None, None, None, None
    
    try:
        table = Table.read(filename, format='ascii.ecsv')
        z = table['z']

        re_kpc_phys = table['log_rho_sfr']
        re_err_plus_kpc_phys = np.zeros_like(re_kpc_phys)
        re_err_minus_kpc_phys = np.zeros_like(re_kpc_phys)
        
        return z, re_kpc_phys, re_err_plus_kpc_phys, re_err_minus_kpc_phys
    except Exception as e:
        print(f"Error loading Madau and Dickinson 2014 data: {e}")
    return None, None, None, None

def load_perez_rho_uv_2023_data():
    """Load Perez et al. 2023 UV density data."""
    filename = './data/perez_rho_uv_2023.ecsv'
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None, None, None, None
    
    try:
        table = Table.read(filename, format='ascii.ecsv')
        z = table['z_bin']

        re_kpc_phys = table['rho_UV_10_25']
        re_err_plus_kpc_phys = np.zeros_like(re_kpc_phys)
        re_err_minus_kpc_phys = np.zeros_like(re_kpc_phys)

        re_kpc_phys = np.log10(re_kpc_phys*10e25)
        re_err_plus_kpc_phys = re_err_plus_kpc_phys / (re_kpc_phys * np.log(10))
        re_err_minus_kpc_phys = re_err_minus_kpc_phys / (re_kpc_phys * np.log(10))
        
        return z, re_kpc_phys, re_err_plus_kpc_phys, re_err_minus_kpc_phys
    except Exception as e:
        print(f"Error loading Madau and Dickinson 2014 data: {e}")
    return None, None, None, None

def load_harikane_uv_density_2023_data():
    """Load Harizane et al. 2023 density data."""
    filename = './data/harikane_density_2023.ecsv'
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None, None, None, None
    
    try:
        table = Table.read(filename, format='ascii.ecsv')
        z = table['z']

        re_kpc_phys = table['log_rho_UV']
        re_err_plus_kpc_phys = table['e_log_rho_UV_upper']
        re_err_minus_kpc_phys = table['e_log_rho_UV_lower']
        
        return z, re_kpc_phys, re_err_plus_kpc_phys, re_err_minus_kpc_phys
    except Exception as e:
        print(f"Error loading Madau and Dickinson 2014 data: {e}")
    return None, None, None, None

def load_harikane_sfr_density_2023_data():
    """Load Harizane et al. 2023 SFR density data."""
    filename = './data/harikane_density_2023.ecsv'
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None, None, None, None
    
    try:
        table = Table.read(filename, format='ascii.ecsv')
        z = table['z']

        re_kpc_phys = table['log_rho_SFR_UV']
        re_err_plus_kpc_phys = table['e_log_rho_SFR_UV_upper']
        re_err_minus_kpc_phys = table['e_log_rho_SFR_UV_lower']
        
        return z, re_kpc_phys, re_err_plus_kpc_phys, re_err_minus_kpc_phys
    except Exception as e:
        print(f"Error loading Madau and Dickinson 2014 data: {e}")
    return None, None, None, None

def plot_ffb_metallicity_limit(use_analytical=True):
    """
    Plot the upper limit for metallicity in FFB star-forming clouds.
    Shows mixing between outflowing metals (Zsn = 1 Zsun) and inflowing gas
    (Zin = 0.1 Zsun or 0.02 Zsun) at the FFB threshold mass.
    Shading represents epsilon_max from 0.2 (bottom) to 1.0 (top).
    
    Parameters:
    -----------
    use_analytical : bool
        If True, includes analytical predictions from ffb_predict.py methodology
        If False, only plots empirical SAGE data (original behavior)
    
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
                    # Read metallicity data
                    try:
                        metals_cold = read_hdf_from_model(model_dir, filename, snap_str, 'MetalsColdGas', hubble_h) * 1.0e10 / hubble_h
                        cold_gas = read_hdf_from_model(model_dir, filename, snap_str, 'ColdGas', hubble_h) * 1.0e10 / hubble_h
                    except:
                        metals_cold = None
                        cold_gas = None
                    
                    # Get median mass loading factor and count for Poisson errors
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
                'mass_loading': np.array(mass_loading_model),
                'dir': model_dir,
                'filename': filename
            })
            print(f"  Processed {len(redshifts_model)} redshifts")
            print(f"    Mass loading range: {np.min(mass_loading_model):.2f} to {np.max(mass_loading_model):.2f}")
    
    # Create fine redshift grid for plotting
    z_grid = np.linspace(5, 20, 100)
    
    def calculate_metallicity_from_data(z_grid, model_data, Z_in):
        """
        Calculate metallicity using analytical formula with eps_max from model.
        This ensures consistency between empirical and analytical approaches.
        """
        # Get FFB threshold mass at each redshift
        lgMh_ffb = np.array([calculate_ffb_threshold_mass(z, hubble_h, return_log=True) for z in z_grid])
        
        # Calculate metallicity using analytical formula with model's eps_max
        eps_max = model_data['eps_max']
        Z_mix = ffb_metal_analytical(lgMh_ffb, z_grid, 
                                      Zsn=1.0,  # Z_sn in solar units
                                      Zin=Z_in / Z_sun,  # Convert to solar units
                                      eps_max=eps_max)
        
        return Z_mix  # Already in units of Z_sun
    
    def calculate_metallicity_from_sage_data(z_grid, model_data, Z_in):
        """
        Calculate metallicity directly from SAGE cold gas metallicity.
        This uses the actual simulated metallicity values, not analytical formulas.
        The Z_in parameter is not used here - it's just for consistency with the function signature.
        """
        # We need to read the actual cold gas metallicity from SAGE at each redshift
        Z_sage_by_z = []
        
        for z_target in z_grid:
            # Find closest redshift in model data
            z_diffs = np.abs(model_data['redshifts'] - z_target)
            if np.min(z_diffs) < 0.5:  # Within 0.5 in redshift
                iz = np.argmin(z_diffs)
                z_actual = model_data['redshifts'][iz]
                
                # Read metallicity data for this redshift
                try:
                    snap_str = f'Snap_{snapshots[actual_redshifts.index(z_actual)]}'
                    metals_cold = read_hdf_from_model(model_data['dir'], model_data['filename'], snap_str, 'MetalsColdGas', hubble_h) * 1.0e10 / hubble_h
                    cold_gas = read_hdf_from_model(model_data['dir'], model_data['filename'], snap_str, 'ColdGas', hubble_h) * 1.0e10 / hubble_h
                    mvir = read_hdf_from_model(model_data['dir'], model_data['filename'], snap_str, 'Mvir', hubble_h) * 1.0e10 / hubble_h
                    sfr_disk = read_hdf_from_model(model_data['dir'], model_data['filename'], snap_str, 'SfrDisk', hubble_h)
                    sfr_bulge = read_hdf_from_model(model_data['dir'], model_data['filename'], snap_str, 'SfrBulge', hubble_h)
                    galaxy_type = read_hdf_from_model(model_data['dir'], model_data['filename'], snap_str, 'Type', hubble_h)
                    
                    M_ffb_threshold = calculate_ffb_threshold_mass(z_actual, hubble_h)
                    sfr_total = sfr_disk + sfr_bulge
                    is_disk = galaxy_type == 0
                    has_sfr = sfr_total > 0
                    near_threshold = (mvir > M_ffb_threshold * 0.8) & (mvir < M_ffb_threshold * 1.2)
                    w = np.where(is_disk & has_sfr & near_threshold & (cold_gas > 0))[0]
                    
                    if len(w) > 0:
                        Z_gas = metals_cold[w] / cold_gas[w] / Z_sun
                        Z_median = np.median(Z_gas)
                        Z_sage_by_z.append(Z_median)
                    else:
                        Z_sage_by_z.append(np.nan)
                except:
                    Z_sage_by_z.append(np.nan)
            else:
                Z_sage_by_z.append(np.nan)
        
        return np.array(Z_sage_by_z)
    
    # Calculate SAGE metallicity for each Z_in scenario (median across all models)
    print(f"\nCalculating SAGE metallicity predictions for {len(all_model_data)} models...")
    
    Z_sage_high_all = []  # High Z_in = 0.1 Z_sun
    Z_sage_low_all = []   # Low Z_in = 0.02 Z_sun
    
    for model_data in all_model_data:
        print(f"  Processing {model_data['name']} (eps_max={model_data['eps_max']})")
        
        # Calculate metallicity using SAGE data with analytical mixing formula
        Z_high = calculate_metallicity_from_sage_data(z_grid, model_data, Z_in_high)
        Z_low = calculate_metallicity_from_sage_data(z_grid, model_data, Z_in_low)
        
        Z_sage_high_all.append(Z_high)
        Z_sage_low_all.append(Z_low)
    
    # Convert to arrays: (n_models, n_redshifts)
    Z_sage_high_all = np.array(Z_sage_high_all)
    Z_sage_low_all = np.array(Z_sage_low_all)
    
    # Take median across all models for clean single lines
    Z_sage_high = np.nanmedian(Z_sage_high_all, axis=0)
    Z_sage_low = np.nanmedian(Z_sage_low_all, axis=0)
    
    # Calculate spread across models for uncertainty bands
    Z_sage_high_min = np.nanpercentile(Z_sage_high_all, 16, axis=0)
    Z_sage_high_max = np.nanpercentile(Z_sage_high_all, 84, axis=0)
    
    Z_sage_low_min = np.nanpercentile(Z_sage_low_all, 16, axis=0)
    Z_sage_low_max = np.nanpercentile(Z_sage_low_all, 84, axis=0)
    
    valid_high = ~np.isnan(Z_sage_high)
    valid_low = ~np.isnan(Z_sage_low)
    
    print(f"\n  High Z_in (0.1 Z_sun): {np.nanmin(Z_sage_high):.3f} - {np.nanmax(Z_sage_high):.3f} Z_sun")
    print(f"  Low Z_in (0.02 Z_sun): {np.nanmin(Z_sage_low):.3f} - {np.nanmax(Z_sage_low):.3f} Z_sun")
    print(f"  Valid points: High={np.sum(valid_high)}, Low={np.sum(valid_low)} out of {len(z_grid)}")
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Plot two SAGE metallicity predictions using analytical mixing formula
    print(f"\nPlotting SAGE metallicity predictions...")
    
    # Calculate min/max for shading
    Z_high_full_min = np.nanmin(Z_sage_high_all, axis=0)
    Z_high_full_max = np.nanmax(Z_sage_high_all, axis=0)
    Z_low_full_min = np.nanmin(Z_sage_low_all, axis=0)
    Z_low_full_max = np.nanmax(Z_sage_low_all, axis=0)
    
    valid_band_high = ~np.isnan(Z_high_full_min) & ~np.isnan(Z_high_full_max)
    valid_band_low = ~np.isnan(Z_low_full_min) & ~np.isnan(Z_low_full_max)
    
    print(f"  High Z_in shading: {np.sum(valid_band_high)} valid points, range {np.nanmin(Z_high_full_min):.4f} to {np.nanmax(Z_high_full_max):.4f}")
    print(f"  Low Z_in shading: {np.sum(valid_band_low)} valid points, range {np.nanmin(Z_low_full_min):.4f} to {np.nanmax(Z_low_full_max):.4f}")
    print(f"  Z_sage_high_all shape: {Z_sage_high_all.shape}, number of models: {len(all_model_data)}")
    
    # Check actual width of shading
    if np.sum(valid_band_high) > 0:
        width_high = Z_high_full_max[valid_band_high] - Z_high_full_min[valid_band_high]
        print(f"  High Z_in band width: mean={np.mean(width_high):.4f}, max={np.max(width_high):.4f}")
    if np.sum(valid_band_low) > 0:
        width_low = Z_low_full_max[valid_band_low] - Z_low_full_min[valid_band_low]
        print(f"  Low Z_in band width: mean={np.mean(width_low):.4f}, max={np.max(width_low):.4f}")
    
    # Draw shading FIRST (so it's in background)
    if np.sum(valid_band_high) > 0:
        print(f"  Drawing HIGH shading: z from {z_grid[valid_band_high].min():.1f} to {z_grid[valid_band_high].max():.1f}")
        ax.fill_between(z_grid[valid_band_high], 
                       Z_high_full_min[valid_band_high], 
                       Z_high_full_max[valid_band_high],
                       color='cornflowerblue', alpha=0.5, linewidth=0, zorder=1,
                       label=r'$\epsilon_{\rm max}$ variation (20-100%)')
    
    if np.sum(valid_band_low) > 0:
        print(f"  Drawing LOW shading: z from {z_grid[valid_band_low].min():.1f} to {z_grid[valid_band_low].max():.1f}")
        ax.fill_between(z_grid[valid_band_low], 
                       Z_low_full_min[valid_band_low], 
                       Z_low_full_max[valid_band_low],
                       color='orange', alpha=0.5, linewidth=0, zorder=1)
    
    # High Z_in = 0.1 Z_sun (blue) - individual model lines
    if np.sum(valid_high) > 0:
        for i, model_data in enumerate(all_model_data):
            Z_model = Z_sage_high_all[i]
            valid_model = ~np.isnan(Z_model)
            if np.sum(valid_model) > 0:
                label = f'{model_data["name"]}' if i == 0 else None
                ax.plot(z_grid[valid_model], Z_model[valid_model],
                       color='royalblue', linestyle='-', linewidth=0.8, alpha=0.6, zorder=2)
        
        # Median line (thick, on top)
        ax.plot(z_grid[valid_high], Z_sage_high[valid_high],
               color='darkblue', linestyle='-', linewidth=3.0, zorder=3,
               label=r'SAGE ($Z_{\rm in}=0.1\,Z_{\odot}$)')
    
    # Low Z_in = 0.02 Z_sun (orange) - individual model lines
    if np.sum(valid_low) > 0:
        for i, model_data in enumerate(all_model_data):
            Z_model = Z_sage_low_all[i]
            valid_model = ~np.isnan(Z_model)
            if np.sum(valid_model) > 0:
                ax.plot(z_grid[valid_model], Z_model[valid_model],
                       color='darkorange', linestyle='-', linewidth=0.8, alpha=0.6, zorder=2)
        
        # Median line (thick, on top)
        ax.plot(z_grid[valid_low], Z_sage_low[valid_low],
               color='saddlebrown', linestyle='-', linewidth=3.0, zorder=3,
               label=r'SAGE ($Z_{\rm in}=0.02\,Z_{\odot}$)')
    
    if np.sum(valid_high) == 0 and np.sum(valid_low) == 0:
        print("WARNING: No valid SAGE data to plot!")
    
    # Formatting
    ax.set_xlabel('Redshift z', fontsize=12)
    ax.set_ylabel(r'$\log_{10}(Z_{\rm FFB} / Z_{\odot})$', fontsize=12)
    ax.set_xlim(5, 20)
    
    # Set integer ticks with minor ticks at 20% intervals
    from matplotlib.ticker import MultipleLocator
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax.set_yscale('log')
    
    ax.legend(loc='upper right', fontsize=9, frameon=True, framealpha=0.9, ncol=1)
    
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
# SPIN PARAMETER ANALYSIS
# ==================================================================

def plot_spin_vs_mass(models=None):
    """Plot median spin parameter as a function of halo mass at different redshifts.

    Args:
        models: List of model dictionaries to plot. If None, uses first model in PLOT_MODELS.
    """
    print('\n' + '='*60)
    print('Creating Spin Parameter vs Halo Mass Plot')
    print('='*60)

    if models is None:
        models = [PLOT_MODELS[0]]

    model = models[0]
    model_dir = model['dir']
    filename = 'model_0.hdf5'
    hubble_h = model['hubble_h']

    # Check if file exists
    if not os.path.exists(model_dir + filename):
        print(f"Error: {model_dir + filename} not found")
        return

    # Select redshifts to plot
    target_redshifts = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    colors = plt.cm.plasma(np.linspace(0, 0.9, len(target_redshifts)))

    fig, ax = plt.subplots(figsize=(10, 7))

    for z_target, color in zip(target_redshifts, colors):
        # Find closest snapshot
        idx = np.argmin(np.abs(np.array(DEFAULT_REDSHIFTS) - z_target))
        snapshot = f'Snap_{idx}'
        z_actual = DEFAULT_REDSHIFTS[idx]

        # Read data
        mvir = read_hdf_from_model(model_dir, filename, snapshot, 'Mvir', hubble_h) * 1.0e10 / hubble_h
        spinx = read_hdf_from_model(model_dir, filename, snapshot, 'Spinx', hubble_h)
        spiny = read_hdf_from_model(model_dir, filename, snapshot, 'Spiny', hubble_h)
        spinz = read_hdf_from_model(model_dir, filename, snapshot, 'Spinz', hubble_h)
        rvir = read_hdf_from_model(model_dir, filename, snapshot, 'Rvir', hubble_h)  # Mpc/h
        vvir = read_hdf_from_model(model_dir, filename, snapshot, 'Vvir', hubble_h)  # km/s

        # Calculate angular momentum magnitude |J| in (Mpc/h)*(km/s)
        J_mag = np.sqrt(spinx**2 + spiny**2 + spinz**2)

        # Calculate dimensionless spin parameter: λ = |J| / (sqrt(2) * Vvir * Rvir)
        # Rvir is in Mpc/h, Vvir in km/s, J in (Mpc/h)*(km/s)
        spin = J_mag / (np.sqrt(2) * vvir * rvir)

        # Select central galaxies with valid mass and non-zero spin
        galaxy_type = read_hdf_from_model(model_dir, filename, snapshot, 'Type', hubble_h)
        w = np.where((mvir > 0) & (galaxy_type == 0) & (vvir > 0) & (rvir > 0))[0]

        if len(w) < 10:
            print(f"  z={z_actual:.2f}: insufficient data ({len(w)} galaxies)")
            continue

        mvir_sel = mvir[w]
        spin_sel = spin[w]

        # Bin by mass and compute median
        mass_bins = np.logspace(np.log10(mvir_sel.min()), np.log10(mvir_sel.max()), 15)
        mass_centers = 0.5 * (mass_bins[:-1] + mass_bins[1:])

        median_spin = np.zeros(len(mass_centers))
        spin_16 = np.zeros(len(mass_centers))
        spin_84 = np.zeros(len(mass_centers))
        valid_bins = np.zeros(len(mass_centers), dtype=bool)

        for i in range(len(mass_centers)):
            in_bin = (mvir_sel >= mass_bins[i]) & (mvir_sel < mass_bins[i+1])
            if np.sum(in_bin) >= 5:
                median_spin[i] = np.median(spin_sel[in_bin])
                spin_16[i] = np.percentile(spin_sel[in_bin], 16)
                spin_84[i] = np.percentile(spin_sel[in_bin], 84)
                valid_bins[i] = True

        # Plot
        ax.plot(mass_centers[valid_bins], median_spin[valid_bins],
                color=color, linewidth=2, label=f'z = {z_actual:.1f}')
        ax.fill_between(mass_centers[valid_bins], spin_16[valid_bins], spin_84[valid_bins],
                       color=color, alpha=0.2)

        print(f"  z={z_actual:.2f}: {len(w)} centrals, median λ = {np.median(spin_sel):.4f}")

    # Theoretical expectation (λ ~ 0.035 from N-body simulations)
    ax.axhline(0.05, color='gray', linestyle='--', linewidth=1.5, label='λ = 0.05 (theory)')
    ax.axhline(0.025, color='red', linestyle='--', linewidth=1.5, label='λ = 0.025 (Li+24)')

    ax.set_xscale('log')
    ax.set_xlabel(r'$M_{\rm vir}$ [$M_\odot$]', fontsize=14)
    ax.set_ylabel(r'Spin Parameter $\lambda$', fontsize=14)
    ax.set_ylim(0, 0.15)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_title('Median Spin Parameter vs Halo Mass', fontsize=14)

    plt.tight_layout()

    # Save
    OutputDir = DirName + 'plots/'
    if not os.path.exists(OutputDir):
        os.makedirs(OutputDir)
    output_path = OutputDir + 'spin_vs_mass' + OutputFormat
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'\nPlot saved to: {output_path}')
    plt.close()

    print('='*60 + '\n')


def plot_spin_vs_redshift(models=None):
    """Plot median spin parameter as a function of redshift for different mass bins.

    Args:
        models: List of model dictionaries to plot. If None, uses first model in PLOT_MODELS.
    """
    print('\n' + '='*60)
    print('Creating Spin Parameter vs Redshift Plot')
    print('='*60)

    if models is None:
        models = [PLOT_MODELS[0]]

    model = models[0]
    model_dir = model['dir']
    filename = 'model_0.hdf5'
    hubble_h = model['hubble_h']

    # Check if file exists
    if not os.path.exists(model_dir + filename):
        print(f"Error: {model_dir + filename} not found")
        return

    # Define mass bins
    mass_bins = [(1e10, 1e11, r'$10^{10} < M_{\rm vir} < 10^{11}$'),
                 (1e11, 1e12, r'$10^{11} < M_{\rm vir} < 10^{12}$'),
                 (1e12, 1e13, r'$10^{12} < M_{\rm vir} < 10^{13}$')]
    colors = ['C0', 'C1', 'C2']

    # Redshift range
    redshifts_to_plot = []
    snapshots_to_plot = []
    for i, z in enumerate(DEFAULT_REDSHIFTS):
        if z <= 15:
            redshifts_to_plot.append(z)
            snapshots_to_plot.append(f'Snap_{i}')

    fig, ax = plt.subplots(figsize=(10, 7))

    for (m_min, m_max, label), color in zip(mass_bins, colors):
        median_spins = []
        spin_16_list = []
        spin_84_list = []
        valid_redshifts = []

        for snapshot, z_actual in zip(snapshots_to_plot, redshifts_to_plot):
            # Read data
            mvir = read_hdf_from_model(model_dir, filename, snapshot, 'Mvir', hubble_h) * 1.0e10 / hubble_h
            spinx = read_hdf_from_model(model_dir, filename, snapshot, 'Spinx', hubble_h)
            spiny = read_hdf_from_model(model_dir, filename, snapshot, 'Spiny', hubble_h)
            spinz = read_hdf_from_model(model_dir, filename, snapshot, 'Spinz', hubble_h)
            rvir = read_hdf_from_model(model_dir, filename, snapshot, 'Rvir', hubble_h)  # Mpc/h
            vvir = read_hdf_from_model(model_dir, filename, snapshot, 'Vvir', hubble_h)  # km/s

            # Calculate angular momentum magnitude |J| in (Mpc/h)*(km/s)
            J_mag = np.sqrt(spinx**2 + spiny**2 + spinz**2)

            # Calculate dimensionless spin parameter: λ = |J| / (sqrt(2) * Vvir * Rvir)
            # Rvir is in Mpc/h, Vvir in km/s, J in (Mpc/h)*(km/s)
            spin = J_mag / (np.sqrt(2) * vvir * rvir)

            # Select central galaxies in mass bin
            galaxy_type = read_hdf_from_model(model_dir, filename, snapshot, 'Type', hubble_h)
            w = np.where((mvir >= m_min) & (mvir < m_max) & (galaxy_type == 0) & (vvir > 0) & (rvir > 0))[0]

            if len(w) >= 5:
                median_spins.append(np.median(spin[w]))
                spin_16_list.append(np.percentile(spin[w], 16))
                spin_84_list.append(np.percentile(spin[w], 84))
                valid_redshifts.append(z_actual)

        if len(valid_redshifts) > 0:
            ax.plot(valid_redshifts, median_spins, color=color, linewidth=2, label=label)
            ax.fill_between(valid_redshifts, spin_16_list, spin_84_list, color=color, alpha=0.2)
            print(f"  {label}: {len(valid_redshifts)} redshifts with data")

    # Theoretical expectation
    ax.axhline(0.05, color='gray', linestyle='--', linewidth=1.5, label='λ = 0.05 (theory)')
    ax.axhline(0.025, color='red', linestyle='--', linewidth=1.5, label='λ = 0.025 (Li+24)')

    ax.set_xlabel('Redshift', fontsize=14)
    ax.set_ylabel(r'Spin Parameter $\lambda$', fontsize=14)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 0.15)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_title('Median Spin Parameter vs Redshift', fontsize=14)

    plt.tight_layout()

    # Save
    OutputDir = DirName + 'plots/'
    if not os.path.exists(OutputDir):
        os.makedirs(OutputDir)
    output_path = OutputDir + 'spin_vs_redshift' + OutputFormat
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
    
    Mvir = read_hdf(snap_num = Snapshot, param = 'Mvir') * 1.0e10 / Hubble_h
    StellarMass = read_hdf(snap_num = Snapshot, param = 'StellarMass') * 1.0e10 / Hubble_h
    
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
    # plot_cumulative_surface_density()
    plot_density_evolution()
    plot_ffb_threshold_analysis()
    plot_ffb_threshold_analysis_empirical()
    plot_ffb_threshold_analysis_empirical_all()
    plot_gas_fraction_evolution()
    plot_radius_evolution_all_galaxies()
    plot_rvir_vs_redshift()

    plot_spin_vs_mass()
    plot_spin_vs_redshift()

    # plot_ffb_metallicity_limit(use_analytical=True)  # Disabled
    
    print('\nAll plots completed!')
    print(f'Plots saved to: {OutputDir}')
    print('='*60 + '\n')

# ==================================================================

if __name__ == '__main__':
    main()
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
    Convert SFR to UV absolute magnitude using Behroozi+2020, eq B3.
    This is the correct conversion from the Li+2023 paper (ffb_predict.py).
    
    Parameters:
    -----------
    SFR : array-like
        Star formation rate in Msun/yr
    z : float
        Redshift
    
    Returns:
    --------
    MUV : array-like
        Absolute UV magnitude at 1500 Angstrom
    """
    # Behroozi+2020, eq B3 - redshift-dependent conversion factor
    kappa_UV = 5.1e-29 * (1 + np.exp(-20.79 / (1 + z) + 0.98))
    lum = SFR / kappa_UV * (u.erg / u.s / u.Hz)
    dist_lum = 10 * u.pc
    flux = lum / (4 * np.pi * dist_lum**2)
    MUV = flux.to(u.ABmag).value
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
        # COSMOS2020 (Farmer+)
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
                           fmt='s', color='black', markersize=10, alpha=1.0,
                           label='Farmer+ (COSMOS2020)' if idx == 0 else '', capsize=2, linewidth=1.5)
                print(f"  Farmer+ (COSMOS2020) data added")
        
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
                               fmt='D', color='black', markersize=10, alpha=1.0,
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

def calculate_uv_luminosity_function(stellar_mass, sfr_disk, sfr_bulge, volume, hubble_h, redshift, binwidth=0.5):
    """Calculate UV luminosity function with Poisson errors
    
    Uses conversion from SFR to UV luminosity following Behroozi+2020 (eq B3):
    kappa_UV = 5.1e-29 * (1 + exp(-20.79 / (1 + z) + 0.98))
    
    This is the correct conversion from Li+2023 paper (ffb_predict.py).
    
    Returns: M_UV bins, log(phi), log(phi_lower), log(phi_upper)
    """
    # Total star formation rate
    sfr_total = sfr_disk + sfr_bulge
    
    # Select galaxies with SFR > 0
    w = np.where(sfr_total > 0.0)[0]
    if len(w) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    # Convert SFR to UV magnitude at 1500Å using Behroozi+2020 formula
    # This includes redshift-dependent conversion factor
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
        dataset_styles = {
            'adams_2024': {'label': 'Adams+24', 'marker': 'o', 'color': 'black'},
            'bouwens_2021': {'label': 'Bouwens+21', 'marker': 's', 'color': 'black'},
            'bouwens_2023': {'label': 'Bouwens+23', 'marker': 'D', 'color': 'black'},
            'donnan_2023': {'label': 'Donnan+23', 'marker': '^', 'color': 'black'},
            'finkelstein_2022': {'label': 'Finkelstein+22', 'marker': 'v', 'color': 'black'},
            'harikane_2023': {'label': 'Harikane+23', 'marker': 'p', 'color': 'black'},
            'mcleod_2024': {'label': 'McLeod+24', 'marker': '*', 'color': 'black'},
            'morishita_2018': {'label': 'Morishita+18', 'marker': 'h', 'color': 'black'}
        }
        for dataset_name, data in obs_data.items():
            # Find data points within redshift tolerance
            # For Finkelstein data, use tighter tolerance to only show in closest panel
            if dataset_name == 'finkelstein_2022':
                z_mask = np.abs(data['redshifts'] - z_actual) < 0.5
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
                style = dataset_styles.get(dataset_name, {'label': dataset_name, 'marker': 'o', 'color': 'black'})
                
                # Plot with different markers for each dataset (matching SMF grid size)
                # Add label only the first time this dataset appears across all subplots
                label = style['label'] if dataset_name not in datasets_in_legend else ''
                if label:
                    datasets_in_legend.add(dataset_name)
                
                ax.errorbar(M_UV_obs, log_phi_obs,
                           yerr=[log_phi_obs - log_phi_err_low, log_phi_err_up - log_phi_obs],
                           fmt=style['marker'], color=style['color'], markersize=10,
                           markerfacecolor=style['color'], markeredgecolor=style['color'],
                           ecolor=style['color'], elinewidth=1.5, capsize=2,
                           alpha=1.0, zorder=5, label=label)
                obs_count += np.sum(z_mask)
        
        if obs_count > 0:
            print(f"  Added {obs_count} observational data points")
        
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
                    M_UV = func_MUV_sfr(sfr_total[w], z_actual)
                    
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
        
        # Add analytical predictions from Li+2023 (three lines)
        try:
            # FFB eps_max=1.0
            set_option(FFB_SFE_MAX=1.0)
            redshifts_ffb1, phi_ffb1 = [], []
            for z_actual in actual_redshifts:
                MUV_bins, dNdMUV = compute_dNdMUV_Ms(z_actual, attenuation=None)
                mask = MUV_bins < M_UV_threshold
                if np.sum(mask) > 0:
                    n_cumulative = np.abs(np.trapz(dNdMUV[mask], MUV_bins[mask]))
                    if n_cumulative > 0:
                        redshifts_ffb1.append(z_actual)
                        phi_ffb1.append(np.log10(n_cumulative))
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
                mask = MUV_bins < M_UV_threshold
                if np.sum(mask) > 0:
                    n_cumulative = np.abs(np.trapz(dNdMUV[mask], MUV_bins[mask]))
                    if n_cumulative > 0:
                        redshifts_ffb02.append(z_actual)
                        phi_ffb02.append(np.log10(n_cumulative))
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
                mask = MUV_bins < M_UV_threshold
                if np.sum(mask) > 0:
                    n_cumulative = np.abs(np.trapz(dNdMUV[mask], MUV_bins[mask]))
                    if n_cumulative > 0:
                        redshifts_um.append(z_actual)
                        phi_um.append(np.log10(n_cumulative))
            if len(redshifts_um) > 0:
                ax.plot(redshifts_um, phi_um,
                       color='gray', linestyle='--', linewidth=2,
                       label='',
                       alpha=0.7, zorder=4)
            
            # Reset to default
            set_option(FFB_SFE_MAX=1.0)
            print(f"  Li+2023 analytical predictions added (3 lines) for M_UV < {M_UV_threshold}")
        except Exception as e:
            print(f"  Warning: Could not compute analytical UVLF prediction: {e}")
        
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
                    # Convert SFR to UV magnitude using Behroozi+2020 (correct formula from Li+2023 paper)
                    M_UV = func_MUV_sfr(sfr_total[w], z_actual)
                    
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
            volume = (model['boxsize'] / hubble_h)**3.0 * model.get('volume_fraction', 1.0)
            
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
        bulge_radius_raw = read_hdf_from_model(model_dir, filename, snapshot, 'BulgeRadius', hubble_h)  # Already in Mpc/h
        galaxy_type = read_hdf_from_model(model_dir, filename, snapshot, 'Type', hubble_h)
        
        # Check if BulgeRadius loaded correctly (should have same length as other arrays)
        if len(bulge_radius_raw) != len(mvir):
            # BulgeRadius doesn't exist, use BulgeScaleRadius instead or set to zeros
            bulge_radius = np.zeros_like(disk_radius)
        else:
            bulge_radius = bulge_radius_raw
        
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
        sage_bulge_radius = read_hdf_from_model(model_dir, filename, snapshot, 'BulgeRadius', hubble_h)
        
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
        valid_map = has_mass & (r_disk_kpc > 0)
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
    BulgeRadius = read_hdf(snap_num = Snapshot, param = 'BulgeRadius')
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
    # plot_ffb_metallicity_limit(use_analytical=True)  # Disabled
    
    print('\nAll plots completed!')
    print(f'Plots saved to: {OutputDir}')
    print('='*60 + '\n')

# ==================================================================

if __name__ == '__main__':
    main()
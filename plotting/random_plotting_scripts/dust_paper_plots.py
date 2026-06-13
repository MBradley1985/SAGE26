#!/usr/bin/env python
"""
SAGE26 Dust Paper Plots
========================
Publication-quality figures for dusty-SAGE / SAGE26 dust modeling paper.
Consolidated from dusty-sage analysis notebooks (Paper2_plotting.ipynb,
dust-result.ipynb, H2_H1_Mass.ipynb, BulgeDisk Mass.ipynb).

Usage:
    python plotting/dust_paper_plots.py              # Generate all plots
    python plotting/dust_paper_plots.py 1 3 5        # Generate specific plots
    python plotting/dust_paper_plots.py --list       # List available plots
    python plotting/dust_paper_plots.py --dir output/millennium_dust/

Plots:
    1.  Dust Mass Function (z=0)
    2.  Dust Mass Function Evolution (8-panel, z=0 to z~7)
    3.  Dust-Stellar Mass Relation (z=0)
    4.  Dust-Stellar Mass Evolution (8-panel)
    5.  Dust-to-Gas vs Metallicity (DtG-Z)
    6.  Dust-to-Metal vs Metallicity (DtM-Z)
    7.  Dust-to-Gas vs Stellar Mass
    8.  Dust-to-Metal vs Stellar Mass
    9.  Cosmic Dust Density Evolution
    10. Dust Rate Evolution (formation, growth, destruction)
    11. Dust Reservoir Breakdown vs Stellar Mass
    12. BTT-sSFR-Dust Classification
    13. Gas Mass Function (HI, H2, Cold)
    14. Stellar Mass Function
    15. All DTG Evolution (ISM, Hot, Ejected)
    16. Dust Mass Density Evolution (all reservoirs)
"""

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import LogNorm
import argparse
import os
import sys
import glob

# ========================== CONFIGURATION ==========================

# Default paths
DEFAULT_DIR = './output/millennium/'
OBS_DIR = './data/'
OUTPUT_FORMAT = '.pdf'

# Simulation parameters
HUBBLE_H = 0.73
BOX_SIZE = 62.5          # h^-1 Mpc
VOLUME_FRACTION = 1.0
VOLUME = (BOX_SIZE / HUBBLE_H)**3 * VOLUME_FRACTION  # Mpc^3
MASS_CONVERT = 1.0e10 / HUBBLE_H
HYDROGEN_MASS_FRAC = 0.76
Z_SUN = 0.02

# Physics thresholds
SSFR_CUT = -11.0  # log10(sSFR/yr^-1) - quiescent/star-forming divide

# Redshift array for Millennium (snaps 0-63)
REDSHIFTS = np.array([
    127.000, 79.998, 50.000, 30.000, 19.916, 18.244, 16.725, 15.343,
    14.086, 12.941, 11.897, 10.944, 10.073,  9.278,  8.550,  7.883,
     7.272,  6.712,  6.197,  5.724,  5.289,  4.888,  4.520,  4.179,
     3.866,  3.576,  3.308,  3.060,  2.831,  2.619,  2.422,  2.239,
     2.070,  1.913,  1.766,  1.630,  1.504,  1.386,  1.276,  1.173,
     1.078,  0.989,  0.905,  0.828,  0.755,  0.687,  0.624,  0.564,
     0.509,  0.457,  0.408,  0.362,  0.320,  0.280,  0.242,  0.208,
     0.175,  0.144,  0.116,  0.089,  0.064,  0.041,  0.020,  0.000])

# Snapshot aliases
SNAP_Z0 = 63   # z = 0.000
SNAP_Z1 = 37   # z = 1.386
SNAP_Z2 = 32   # z = 2.070
SNAP_Z3 = 27   # z = 3.060
SNAP_Z4 = 23   # z = 4.179
SNAP_Z5 = 20   # z = 5.289
SNAP_Z6 = 18   # z = 6.197
SNAP_Z7 = 16   # z = 7.272

# Key snapshots for evolution plots (matching dusty-sage notebooks)
EVOLUTION_SNAPS = [SNAP_Z0, SNAP_Z1, SNAP_Z2, SNAP_Z3, SNAP_Z4, SNAP_Z5, SNAP_Z6, SNAP_Z7]
EVOLUTION_REDSHIFTS = [0.0, 1.386, 2.070, 3.060, 4.179, 5.289, 6.197, 7.272]

# Mass properties that need unit conversion
MASS_PROPS = frozenset({
    'CentralMvir', 'Mvir', 'StellarMass', 'BulgeMass', 'BlackHoleMass',
    'MetalsStellarMass', 'MetalsColdGas', 'MetalsEjectedMass',
    'MetalsHotGas', 'MetalsCGMgas', 'ColdGas', 'HotGas', 'CGMgas',
    'EjectedMass', 'H2gas', 'H1gas', 'IntraClusterStars',
    'MergerBulgeMass', 'InstabilityBulgeMass',
    'ColdDust', 'HotDust', 'EjectedDust', 'CGMDust',
})

# Plot colors (from dusty-sage Paper2_plotting.ipynb)
COLORS = {
    'halo': '#7570b3',
    'ism': '#d95f02',
    'ejected': '#1b9e77',
    'model': 'dodgerblue',
    'obs': 'grey',
}

# Classification colors
CLASS_COLORS = ['#984ea3', '#4daf4a', '#377eb8', '#e41a1c']  # purple, green, blue, red


# ========================== PLOTTING STYLE ==========================

def setup_style(dark=False):
    """Configure matplotlib style."""
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["font.size"] = 12
    plt.rcParams['errorbar.capsize'] = 2
    
    if dark:
        plt.rcParams['figure.facecolor'] = 'black'
        plt.rcParams['axes.facecolor'] = 'black'
        plt.rcParams['axes.edgecolor'] = 'white'
        plt.rcParams['xtick.color'] = 'white'
        plt.rcParams['ytick.color'] = 'white'
        plt.rcParams['axes.labelcolor'] = 'white'
        plt.rcParams['text.color'] = 'white'
        plt.rcParams['legend.facecolor'] = 'black'
        plt.rcParams['legend.edgecolor'] = 'white'
    else:
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['axes.edgecolor'] = 'black'
        plt.rcParams['xtick.color'] = 'black'
        plt.rcParams['ytick.color'] = 'black'
        plt.rcParams['axes.labelcolor'] = 'black'
        plt.rcParams['text.color'] = 'black'
        plt.rcParams['legend.facecolor'] = 'white'
        plt.rcParams['legend.edgecolor'] = 'none'


# ========================== DATA I/O ==========================

def load_snapshot(directory, snap_num, properties=None):
    """Load galaxy data from all model_*.hdf5 files for a snapshot."""
    if properties is None:
        properties = [
            'StellarMass', 'BulgeMass', 'ColdGas', 'HotGas', 'CGMgas',
            'EjectedMass', 'H2gas', 'H1gas', 'MetalsColdGas', 'MetalsHotGas',
            'MetalsStellarMass', 'MetalsEjectedMass', 'MetalsCGMgas',
            'ColdDust', 'HotDust', 'EjectedDust', 'CGMDust',
            'SfrDisk', 'SfrBulge', 'Vvir', 'Type', 'OutflowRate',
        ]
    
    snap_key = f'Snap_{snap_num}'
    # Support both model_*.hdf5 and model_N.hdf5 patterns
    pattern = os.path.join(directory, 'model_*.hdf5')
    filelist = sorted(glob.glob(pattern))
    # Filter to only numbered files (model_0.hdf5, model_1.hdf5, etc.)
    # Exclude combined files like model.hdf5 or model_z0.000.hdf5
    filelist = [f for f in filelist if os.path.basename(f).replace('model_', '').replace('.hdf5', '').isdigit()
                or 'z' not in os.path.basename(f)]
    
    if not filelist:
        print(f"  Warning: No model files found in {directory}")
        return None
    
    data = {prop: [] for prop in properties}
    
    for filepath in filelist:
        try:
            with h5.File(filepath, 'r') as f:
                if snap_key not in f:
                    continue
                snap = f[snap_key]
                for prop in properties:
                    if prop in snap:
                        arr = np.array(snap[prop])
                        if prop in MASS_PROPS:
                            arr *= MASS_CONVERT
                        data[prop].append(arr)
        except Exception as e:
            print(f"  Warning: Error reading {filepath}: {e}")
            continue
    
    # Concatenate arrays
    for prop in properties:
        if data[prop]:
            data[prop] = np.concatenate(data[prop])
        else:
            data[prop] = np.array([])
    
    return data


def load_multiple_snapshots(directory, snap_nums, properties=None):
    """Load data from multiple snapshots."""
    all_data = {}
    for snap in snap_nums:
        print(f"  Loading snapshot {snap}...")
        all_data[snap] = load_snapshot(directory, snap, properties)
    return all_data


# ========================== OBSERVATIONAL DATA ==========================

def load_remy_ruyer_2014():
    """Load Rémy-Ruyer+2014 galaxy sample."""
    datafile = os.path.join(OBS_DIR, 'remy-ruyer.master.dat')
    if not os.path.exists(datafile):
        # Try analysis directory
        datafile = './plotting/data/remy-ruyer.master.dat'
    if not os.path.exists(datafile):
        print("  Warning: Rémy-Ruyer data not found")
        return None
    
    logMstar, logMdust, Z_12logOH, logSFR, logMgas = [], [], [], [], []
    
    with open(datafile, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or len(line) == 0:
                continue
            parts = line.split()
            if len(parts) < 11:
                continue
            try:
                HI = float(parts[1])
                Z = float(parts[3])
                H2_MW = float(parts[4])
                logMd = float(parts[8])
                logMs = float(parts[9])
                lSFR = float(parts[10])
            except (ValueError, IndexError):
                continue
            if logMs <= 0 or logMd <= 0 or Z <= 0:
                continue
            gas_mass = HI + H2_MW
            logMgas.append(np.log10(gas_mass) if gas_mass > 0 else np.nan)
            logMstar.append(logMs)
            logMdust.append(logMd)
            Z_12logOH.append(Z)
            logSFR.append(lSFR)
    
    if len(logMstar) == 0:
        return None
    
    return {
        'logMstar': np.array(logMstar),
        'logMdust': np.array(logMdust),
        'logSFR': np.array(logSFR),
        'Z_12logOH': np.array(Z_12logOH),
        'logMgas': np.array(logMgas),
    }


def load_dmf_observations():
    """Dust mass function observations at z=0."""
    # Vlahakis et al. 2005
    vlahakis = {
        'logMd': np.array([6.289, 6.536, 6.784, 7.032, 7.287, 7.528, 7.776, 8.024, 8.289, 8.536]),
        'logphi': np.array([-1.389, -1.389, -1.504, -1.601, -1.679, -1.873, -2.105, -2.414, -2.782, -3.248]),
        'errhi': np.array([0.117, 0.117, 0.117, 0.117, 0.117, 0.117, 0.117, 0.215, 0.293, 0.508]),
        'errlo': np.array([0.137, 0.098, 0.098, 0.098, 0.117, 0.137, 0.156, 0.235, 0.508, 0.781]),
    }
    
    # Clemens et al. 2013
    clemens = {
        'logMd': np.array([6.405, 6.653, 6.949, 7.238, 7.478, 7.735, 7.982, 8.271, 8.560, 9.096]),
        'logphi': np.array([-1.536, -1.679, -1.873, -1.815, -1.893, -2.185, -2.555, -3.081, -3.567, -4.757]),
        'errhi': np.array([0.098, 0.098, 0.098, 0.098, 0.098, 0.117, 0.156, 0.195, 0.273, 0.742]),
        'errlo': np.array([0.117, 0.137, 0.137, 0.117, 0.098, 0.137, 0.195, 0.332, 0.547, 1.445]),
    }
    
    # Dunne et al. 2011
    dunne = {
        'logMd': np.array([6.071, 6.321, 6.571, 6.821, 7.071, 7.321, 7.571, 7.821, 8.071, 8.160, 8.409]),
        'logphi': np.array([-1.199, -1.102, -1.258, -1.316, -1.473, -1.717, -2.059, -2.362, -2.921, -3.186, -3.733]),
        'errhi': np.array([0.098, 0.078, 0.059, 0.059, 0.059, 0.039, 0.039, 0.078, 0.098, 0.234, 0.469]),
        'errlo': np.array([0.098, 0.059, 0.059, 0.059, 0.059, 0.039, 0.059, 0.078, 0.117, 0.313, 0.781]),
    }
    
    return {'Vlahakis+05': vlahakis, 'Clemens+13': clemens, 'Dunne+11': dunne}


def load_cosmic_dust_density_obs():
    """Cosmic dust mass density observations."""
    # Dunne et al. 2011
    dunne = {
        'z': np.array([0.052, 0.155, 0.251, 0.36, 0.45]),
        'logrho': np.array([4.986, 5.186, 5.316, 5.488, 5.353]),
        'errhi': np.array([0.054, 0.066, 0.049, 0.128, 0.089]),
        'errlo': np.array([0.08, 0.066, 0.078, 0.148, 0.181]),
    }
    
    # Ménard & Fukugita 2012
    menard = {
        'z': np.array([0.661, 0.976, 1.300, 1.618, 1.937]),
        'logrho': np.array([5.749, 5.767, 5.743, 5.631, 5.613]),
        'errhi': np.array([0.114, 0.092, 0.074, 0.062, 0.116]),
        'errlo': np.array([0.114, 0.092, 0.074, 0.062, 0.116]),
    }
    
    return {'Dunne+11': dunne, 'Ménard+12': menard}


def load_dustpedia():
    """DustPedia/Nersesian+2019 binned data."""
    return {
        'logMstar': np.array([7.25, 7.75, 8.25, 8.75, 9.25, 9.75, 10.25, 10.75, 11.25]),
        'logMdust': np.array([4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.3, 7.5, 7.6]),
        'err': np.array([0.4, 0.4, 0.4, 0.3, 0.3, 0.3, 0.3, 0.3, 0.4]),
    }


def load_kingfish():
    """Load KINGFISH survey data."""
    datafile = os.path.join(OBS_DIR, 'kingfish.txt')
    if not os.path.exists(datafile):
        return None
    try:
        data = np.loadtxt(datafile, usecols=(3, 7, 9, 10, 11, 13, 14))
        return {
            'Z': data[:, 0],
            'logMdust': data[:, 1],
            'logMstar': data[:, 3],
            'SFR': data[:, 4],
            'morphology': data[:, 5],
            'BD_ratio': data[:, 6],
        }
    except:
        return None


def load_smf_observations():
    """Stellar mass function observations (Baldry et al. 2008)."""
    baldry = {
        'logMstar': np.array([7.05, 7.45, 7.85, 8.25, 8.65, 9.05, 9.45, 9.85, 10.25, 10.65, 11.05, 11.45]),
        'logphi': np.array([-0.87, -0.66, -0.81, -0.90, -0.96, -1.10, -1.24, -1.40, -1.67, -2.06, -2.66, -3.77]),
        'err': np.array([0.18, 0.17, 0.14, 0.10, 0.08, 0.06, 0.05, 0.04, 0.04, 0.04, 0.05, 0.14]),
    }
    return baldry


def load_gmf_observations():
    """Gas mass function observations."""
    # Zwaan et al. 2005 (HI)
    zwaan = {
        'logMHI': np.array([6.8, 7.2, 7.6, 8.0, 8.4, 8.8, 9.2, 9.6, 10.0, 10.4, 10.8]),
        'logphi': np.array([-0.73, -0.78, -0.89, -1.03, -1.14, -1.37, -1.67, -2.06, -2.60, -3.32, -4.14]),
        'err': np.array([0.19, 0.11, 0.08, 0.06, 0.05, 0.05, 0.04, 0.05, 0.06, 0.10, 0.23]),
    }
    
    # Obreschkow & Rawlings 2009 (H2)
    obreschkow = {
        'logMH2': np.array([7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0]),
        'logphi': np.array([-1.2, -1.3, -1.5, -1.8, -2.2, -2.7, -3.4, -4.3, -5.4]),
        'err': np.array([0.3, 0.2, 0.15, 0.12, 0.10, 0.12, 0.15, 0.25, 0.5]),
    }
    
    return {'Zwaan+05': zwaan, 'Obreschkow+09': obreschkow}


# ========================== COMPUTATION UTILITIES ==========================

def compute_mass_function(mass, volume, nbins=30, mass_range=None):
    """Compute mass function phi(M) = dN/dlogM/dV."""
    logmass = np.log10(mass[mass > 0])
    if len(logmass) == 0:
        return None, None, None
    
    if mass_range is None:
        mass_range = [logmass.min(), logmass.max()]
    
    counts, edges = np.histogram(logmass, bins=nbins, range=mass_range)
    binwidth = edges[1] - edges[0]
    bin_centers = edges[:-1] + binwidth / 2
    phi = counts / (binwidth * volume)
    
    # Poisson errors
    err = np.sqrt(counts) / (binwidth * volume)
    err[counts == 0] = np.nan
    
    return bin_centers, np.log10(phi + 1e-10), err / (phi + 1e-10) / np.log(10)


def compute_metallicity(metals_cold, cold_gas):
    """Compute 12 + log10(O/H) from metals and gas."""
    w = (cold_gas > 0) & (metals_cold > 0)
    Z = np.full(len(cold_gas), np.nan)
    Z[w] = metals_cold[w] / cold_gas[w]
    # Convert mass fraction to 12+log(O/H) assuming solar calibration
    # Z_sun = 0.02, 12+log(O/H)_sun = 8.69
    Z_12logOH = np.full(len(cold_gas), np.nan)
    Z_12logOH[w] = 12 + np.log10(Z[w] / Z_SUN) + 8.69 - 12
    return Z_12logOH


def compute_binned_median(x, y, bins=10, xrange=None):
    """Compute binned median and percentiles."""
    if xrange is None:
        xrange = [np.nanmin(x), np.nanmax(x)]
    
    bin_edges = np.linspace(xrange[0], xrange[1], bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    medians = np.full(bins, np.nan)
    p16 = np.full(bins, np.nan)
    p84 = np.full(bins, np.nan)
    
    for i in range(bins):
        mask = (x >= bin_edges[i]) & (x < bin_edges[i + 1]) & np.isfinite(y)
        if np.sum(mask) >= 5:
            medians[i] = np.median(y[mask])
            p16[i] = np.percentile(y[mask], 16)
            p84[i] = np.percentile(y[mask], 84)
    
    return bin_centers, medians, p16, p84


# ========================== PLOTTING FUNCTIONS ==========================

def plot_1_dmf_z0(data, output_dir):
    """Plot 1: Dust Mass Function at z=0."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Model
    cold_dust = data['ColdDust']
    x, y, yerr = compute_mass_function(cold_dust, VOLUME, nbins=25, mass_range=[5, 10])
    if x is not None:
        ax.plot(x, y, 'b-', lw=2, label='SAGE26')
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.3, color='blue')
    
    # Observations
    obs = load_dmf_observations()
    markers = {'Vlahakis+05': 's', 'Clemens+13': 'o', 'Dunne+11': '^'}
    colors = {'Vlahakis+05': 'grey', 'Clemens+13': 'red', 'Dunne+11': 'green'}
    
    for name, obs_data in obs.items():
        ax.errorbar(obs_data['logMd'], obs_data['logphi'],
                    yerr=[obs_data['errlo'], obs_data['errhi']],
                    fmt=markers[name], color=colors[name], ms=6, capsize=2,
                    label=name)
    
    ax.set_xlabel(r'$\log_{10}(M_{\rm dust}/M_\odot)$', fontsize=14)
    ax.set_ylabel(r'$\log_{10}(\phi / {\rm Mpc}^{-3}\,{\rm dex}^{-1})$', fontsize=14)
    ax.set_xlim(5.5, 9.5)
    ax.set_ylim(-5.5, -0.5)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_title('Dust Mass Function (z = 0)', fontsize=14)
    
    plt.tight_layout()
    outfile = os.path.join(output_dir, f'plot01_dmf_z0{OUTPUT_FORMAT}')
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"  Saved: {outfile}")


def plot_2_dmf_evolution(all_data, output_dir):
    """Plot 2: Dust Mass Function Evolution (8 panels)."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for i, (snap, z) in enumerate(zip(EVOLUTION_SNAPS, EVOLUTION_REDSHIFTS)):
        ax = axes[i]
        data = all_data.get(snap)
        
        if data is not None and len(data.get('ColdDust', [])) > 0:
            cold_dust = data['ColdDust']
            x, y, yerr = compute_mass_function(cold_dust, VOLUME, nbins=20, mass_range=[4, 10])
            if x is not None:
                ax.plot(x, y, 'b-', lw=2)
                ax.fill_between(x, y - yerr, y + yerr, alpha=0.3, color='blue')
        
        # Add observations for z~0 panel
        if i == 0:
            obs = load_dmf_observations()
            for name, obs_data in obs.items():
                ax.errorbar(obs_data['logMd'], obs_data['logphi'],
                            yerr=[obs_data['errlo'], obs_data['errhi']],
                            fmt='o', color='grey', ms=4, capsize=1, alpha=0.7)
        
        ax.text(0.95, 0.95, f'z = {z:.1f}', transform=ax.transAxes,
                ha='right', va='top', fontsize=12)
        ax.set_xlim(4.5, 9.5)
        ax.set_ylim(-6, -0.5)
    
    for ax in axes[4:]:
        ax.set_xlabel(r'$\log_{10}(M_{\rm dust}/M_\odot)$', fontsize=12)
    for ax in [axes[0], axes[4]]:
        ax.set_ylabel(r'$\log_{10}(\phi / {\rm Mpc}^{-3}\,{\rm dex}^{-1})$', fontsize=12)
    
    fig.suptitle('Dust Mass Function Evolution', fontsize=14, y=1.02)
    plt.tight_layout()
    outfile = os.path.join(output_dir, f'plot02_dmf_evolution{OUTPUT_FORMAT}')
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outfile}")


def plot_3_dust_stellar_z0(data, output_dir):
    """Plot 3: Dust Mass vs Stellar Mass at z=0."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    stellar = data['StellarMass']
    dust = data['ColdDust']
    
    w = (stellar > 1e6) & (dust > 0)
    log_stellar = np.log10(stellar[w])
    log_dust = np.log10(dust[w])
    
    # 2D histogram
    h = ax.hist2d(log_stellar, log_dust, bins=50, range=[[7, 12], [3, 9]],
                  norm=LogNorm(), cmap='viridis')
    plt.colorbar(h[3], ax=ax, label='Galaxy count')
    
    # Median line
    xbin, ymed, p16, p84 = compute_binned_median(log_stellar, log_dust, bins=15, xrange=[7.5, 11.5])
    ax.plot(xbin, ymed, 'r-', lw=2, label='Median')
    ax.fill_between(xbin, p16, p84, alpha=0.3, color='red')
    
    # Observations
    rr = load_remy_ruyer_2014()
    if rr is not None:
        ax.scatter(rr['logMstar'], rr['logMdust'], c='white', s=20, 
                   edgecolor='black', alpha=0.7, label='Rémy-Ruyer+14')
    
    ax.set_xlabel(r'$\log_{10}(M_\star/M_\odot)$', fontsize=14)
    ax.set_ylabel(r'$\log_{10}(M_{\rm dust}/M_\odot)$', fontsize=14)
    ax.set_xlim(7, 12)
    ax.set_ylim(3, 9)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_title('Dust-Stellar Mass Relation (z = 0)', fontsize=14)
    
    plt.tight_layout()
    outfile = os.path.join(output_dir, f'plot03_dust_stellar_z0{OUTPUT_FORMAT}')
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"  Saved: {outfile}")


def plot_4_dust_stellar_evolution(all_data, output_dir):
    """Plot 4: Dust-Stellar Mass Relation Evolution (8 panels)."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for i, (snap, z) in enumerate(zip(EVOLUTION_SNAPS, EVOLUTION_REDSHIFTS)):
        ax = axes[i]
        data = all_data.get(snap)
        
        if data is not None and len(data.get('ColdDust', [])) > 0:
            stellar = data['StellarMass']
            dust = data['ColdDust']
            
            w = (stellar > 1e6) & (dust > 0)
            if np.sum(w) > 10:
                log_stellar = np.log10(stellar[w])
                log_dust = np.log10(dust[w])
                
                ax.hist2d(log_stellar, log_dust, bins=30, range=[[7, 12], [3, 9]],
                          norm=LogNorm(vmin=1, vmax=1000), cmap='viridis')
                
                xbin, ymed, p16, p84 = compute_binned_median(log_stellar, log_dust, bins=12)
                ax.plot(xbin, ymed, 'r-', lw=2)
        
        ax.text(0.05, 0.95, f'z = {z:.1f}', transform=ax.transAxes,
                ha='left', va='top', fontsize=12, color='white')
        ax.set_xlim(7, 12)
        ax.set_ylim(3, 9)
    
    for ax in axes[4:]:
        ax.set_xlabel(r'$\log_{10}(M_\star/M_\odot)$', fontsize=12)
    for ax in [axes[0], axes[4]]:
        ax.set_ylabel(r'$\log_{10}(M_{\rm dust}/M_\odot)$', fontsize=12)
    
    fig.suptitle('Dust-Stellar Mass Evolution', fontsize=14, y=1.02)
    plt.tight_layout()
    outfile = os.path.join(output_dir, f'plot04_dust_stellar_evolution{OUTPUT_FORMAT}')
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outfile}")


def plot_5_dtg_metallicity(data, output_dir):
    """Plot 5: Dust-to-Gas Ratio vs Metallicity (DtG-Z relation)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    cold_gas = data['ColdGas']
    cold_dust = data['ColdDust']
    metals_cold = data['MetalsColdGas']
    
    Z_12logOH = compute_metallicity(metals_cold, cold_gas)
    
    w = (cold_gas > 1e6) & (cold_dust > 0) & np.isfinite(Z_12logOH)
    if np.sum(w) > 0:
        logDtG = np.log10(cold_dust[w] / cold_gas[w])
        Z = Z_12logOH[w]
        
        ax.hist2d(Z, logDtG, bins=40, range=[[7, 9.5], [-5, -0.5]],
                  norm=LogNorm(), cmap='viridis')
        
        xbin, ymed, p16, p84 = compute_binned_median(Z, logDtG, bins=12, xrange=[7.5, 9.2])
        ax.plot(xbin, ymed, 'r-', lw=2, label='Median')
        ax.fill_between(xbin, p16, p84, alpha=0.3, color='red')
    
    # Rémy-Ruyer+14 observations
    rr = load_remy_ruyer_2014()
    if rr is not None:
        w_rr = np.isfinite(rr['logMgas'])
        if np.sum(w_rr) > 0:
            dtg_obs = rr['logMdust'][w_rr] - rr['logMgas'][w_rr]
            ax.scatter(rr['Z_12logOH'][w_rr], dtg_obs, c='white', s=30,
                       edgecolor='black', alpha=0.7, label='Rémy-Ruyer+14')
    
    ax.set_xlabel(r'$12 + \log_{10}({\rm O/H})$', fontsize=14)
    ax.set_ylabel(r'$\log_{10}(M_{\rm dust}/M_{\rm gas})$', fontsize=14)
    ax.set_xlim(7, 9.5)
    ax.set_ylim(-5, -0.5)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_title('Dust-to-Gas Ratio vs Metallicity', fontsize=14)
    
    plt.tight_layout()
    outfile = os.path.join(output_dir, f'plot05_dtg_metallicity{OUTPUT_FORMAT}')
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"  Saved: {outfile}")


def plot_6_dtm_metallicity(data, output_dir):
    """Plot 6: Dust-to-Metal Ratio vs Metallicity (DtM-Z relation)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    cold_gas = data['ColdGas']
    cold_dust = data['ColdDust']
    metals_cold = data['MetalsColdGas']
    
    Z_12logOH = compute_metallicity(metals_cold, cold_gas)
    
    w = (metals_cold > 1e3) & (cold_dust > 0) & np.isfinite(Z_12logOH)
    if np.sum(w) > 0:
        logDtM = np.log10(cold_dust[w] / metals_cold[w])
        Z = Z_12logOH[w]
        
        ax.hist2d(Z, logDtM, bins=40, range=[[7, 9.5], [-3, 0.5]],
                  norm=LogNorm(), cmap='viridis')
        
        xbin, ymed, p16, p84 = compute_binned_median(Z, logDtM, bins=12, xrange=[7.5, 9.2])
        ax.plot(xbin, ymed, 'r-', lw=2, label='Median')
        ax.fill_between(xbin, p16, p84, alpha=0.3, color='red')
    
    # Rémy-Ruyer+14 (convert to DtM)
    rr = load_remy_ruyer_2014()
    if rr is not None:
        # DtM from DtG and Z: DtM = DtG / (Z/Z_sun) where Z is mass fraction
        # Z_12logOH -> Z_mass: Z = Z_sun * 10^(Z_12logOH - 8.69)
        Z_mass = Z_SUN * 10**(rr['Z_12logOH'] - 8.69)
        w_rr = np.isfinite(rr['logMgas']) & (Z_mass > 0)
        if np.sum(w_rr) > 0:
            dtg_obs = 10**(rr['logMdust'][w_rr] - rr['logMgas'][w_rr])
            dtm_obs = np.log10(dtg_obs / Z_mass[w_rr])
            ax.scatter(rr['Z_12logOH'][w_rr], dtm_obs, c='white', s=30,
                       edgecolor='black', alpha=0.7, label='Rémy-Ruyer+14')
    
    ax.axhline(0, color='grey', ls='--', alpha=0.5, label='DtM = 1')
    ax.set_xlabel(r'$12 + \log_{10}({\rm O/H})$', fontsize=14)
    ax.set_ylabel(r'$\log_{10}(M_{\rm dust}/M_{\rm metals})$', fontsize=14)
    ax.set_xlim(7, 9.5)
    ax.set_ylim(-3, 0.5)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_title('Dust-to-Metal Ratio vs Metallicity', fontsize=14)
    
    plt.tight_layout()
    outfile = os.path.join(output_dir, f'plot06_dtm_metallicity{OUTPUT_FORMAT}')
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"  Saved: {outfile}")


def plot_7_dtg_stellar_mass(data, output_dir):
    """Plot 7: Dust-to-Gas Ratio vs Stellar Mass."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    stellar = data['StellarMass']
    cold_gas = data['ColdGas']
    cold_dust = data['ColdDust']
    
    w = (stellar > 1e6) & (cold_gas > 1e5) & (cold_dust > 0)
    if np.sum(w) > 0:
        log_stellar = np.log10(stellar[w])
        logDtG = np.log10(cold_dust[w] / cold_gas[w])
        
        ax.hist2d(log_stellar, logDtG, bins=40, range=[[7, 12], [-5, -0.5]],
                  norm=LogNorm(), cmap='viridis')
        
        xbin, ymed, p16, p84 = compute_binned_median(log_stellar, logDtG, bins=15, xrange=[7.5, 11.5])
        ax.plot(xbin, ymed, 'r-', lw=2, label='Median')
        ax.fill_between(xbin, p16, p84, alpha=0.3, color='red')
    
    ax.set_xlabel(r'$\log_{10}(M_\star/M_\odot)$', fontsize=14)
    ax.set_ylabel(r'$\log_{10}(M_{\rm dust}/M_{\rm gas})$', fontsize=14)
    ax.set_xlim(7, 12)
    ax.set_ylim(-5, -0.5)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_title('Dust-to-Gas Ratio vs Stellar Mass', fontsize=14)
    
    plt.tight_layout()
    outfile = os.path.join(output_dir, f'plot07_dtg_stellar{OUTPUT_FORMAT}')
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"  Saved: {outfile}")


def plot_8_dtm_stellar_mass(data, output_dir):
    """Plot 8: Dust-to-Metal Ratio vs Stellar Mass."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    stellar = data['StellarMass']
    cold_dust = data['ColdDust']
    metals_cold = data['MetalsColdGas']
    
    w = (stellar > 1e6) & (metals_cold > 1e3) & (cold_dust > 0)
    if np.sum(w) > 0:
        log_stellar = np.log10(stellar[w])
        logDtM = np.log10(cold_dust[w] / metals_cold[w])
        
        ax.hist2d(log_stellar, logDtM, bins=40, range=[[7, 12], [-3, 0.5]],
                  norm=LogNorm(), cmap='viridis')
        
        xbin, ymed, p16, p84 = compute_binned_median(log_stellar, logDtM, bins=15, xrange=[7.5, 11.5])
        ax.plot(xbin, ymed, 'r-', lw=2, label='Median')
        ax.fill_between(xbin, p16, p84, alpha=0.3, color='red')
    
    ax.axhline(0, color='grey', ls='--', alpha=0.5)
    ax.set_xlabel(r'$\log_{10}(M_\star/M_\odot)$', fontsize=14)
    ax.set_ylabel(r'$\log_{10}(M_{\rm dust}/M_{\rm metals})$', fontsize=14)
    ax.set_xlim(7, 12)
    ax.set_ylim(-3, 0.5)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_title('Dust-to-Metal Ratio vs Stellar Mass', fontsize=14)
    
    plt.tight_layout()
    outfile = os.path.join(output_dir, f'plot08_dtm_stellar{OUTPUT_FORMAT}')
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"  Saved: {outfile}")


def plot_9_cosmic_dust_density(all_data, output_dir):
    """Plot 9: Cosmic Dust Mass Density Evolution."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    z_arr = []
    rho_ism = []
    rho_hot = []
    rho_ejected = []
    rho_total = []
    
    for snap in range(0, 64):
        data = all_data.get(snap)
        if data is None or len(data.get('ColdDust', [])) == 0:
            continue
        
        z = REDSHIFTS[snap]
        cold_dust = np.sum(data['ColdDust'])
        hot_dust = np.sum(data.get('HotDust', [0]))
        ejected_dust = np.sum(data.get('EjectedDust', [0]))
        
        rho_c = cold_dust / VOLUME
        rho_h = hot_dust / VOLUME
        rho_e = ejected_dust / VOLUME
        
        z_arr.append(z)
        rho_ism.append(rho_c)
        rho_hot.append(rho_h)
        rho_ejected.append(rho_e)
        rho_total.append(rho_c + rho_h + rho_e)
    
    z_arr = np.array(z_arr)
    
    ax.plot(z_arr, np.log10(np.array(rho_total) + 1e-10), 'k-', lw=2, label='Total')
    ax.plot(z_arr, np.log10(np.array(rho_ism) + 1e-10), '-', color=COLORS['ism'], lw=2, label='ISM')
    ax.plot(z_arr, np.log10(np.array(rho_hot) + 1e-10), '-', color=COLORS['halo'], lw=2, label='Hot halo')
    ax.plot(z_arr, np.log10(np.array(rho_ejected) + 1e-10), '-', color=COLORS['ejected'], lw=2, label='Ejected')
    
    # Observations
    obs = load_cosmic_dust_density_obs()
    for name, obs_data in obs.items():
        ax.errorbar(obs_data['z'], obs_data['logrho'],
                    yerr=[obs_data['errlo'], obs_data['errhi']],
                    fmt='o', ms=8, capsize=3, label=name)
    
    ax.set_xlabel('Redshift', fontsize=14)
    ax.set_ylabel(r'$\log_{10}(\rho_{\rm dust} / M_\odot\,{\rm Mpc}^{-3})$', fontsize=14)
    ax.set_xlim(0, 8)
    ax.set_ylim(3, 7)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_title('Cosmic Dust Density Evolution', fontsize=14)
    
    plt.tight_layout()
    outfile = os.path.join(output_dir, f'plot09_cosmic_dust_density{OUTPUT_FORMAT}')
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"  Saved: {outfile}")


def plot_10_dust_rates(all_data, output_dir):
    """Plot 10: Dust Rate Evolution (formation, growth, destruction)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # This requires DustDotForm, DustDotGrowth, DustDotDestruct fields
    # If not available, skip
    print("  Note: Dust rate evolution requires rate tracking fields - plot may be incomplete")
    
    ax.set_xlabel('Redshift', fontsize=14)
    ax.set_ylabel(r'$\dot{\rho}_{\rm dust} / M_\odot\,{\rm yr}^{-1}\,{\rm Mpc}^{-3}$', fontsize=14)
    ax.set_xlim(0, 8)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_title('Dust Rate Evolution', fontsize=14)
    
    plt.tight_layout()
    outfile = os.path.join(output_dir, f'plot10_dust_rates{OUTPUT_FORMAT}')
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"  Saved: {outfile}")


def plot_11_dust_reservoirs(data, output_dir):
    """Plot 11: Dust Reservoir Breakdown vs Stellar Mass."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    stellar = data['StellarMass']
    cold_dust = data['ColdDust']
    hot_dust = data.get('HotDust', np.zeros_like(cold_dust))
    ejected_dust = data.get('EjectedDust', np.zeros_like(cold_dust))
    
    total_dust = cold_dust + hot_dust + ejected_dust
    
    w = (stellar > 1e7) & (total_dust > 0)
    log_stellar = np.log10(stellar[w])
    
    f_ism = cold_dust[w] / total_dust[w]
    f_hot = hot_dust[w] / total_dust[w]
    f_ejected = ejected_dust[w] / total_dust[w]
    
    # Binned medians
    xbin, y_ism, _, _ = compute_binned_median(log_stellar, f_ism, bins=15, xrange=[7.5, 11.5])
    _, y_hot, _, _ = compute_binned_median(log_stellar, f_hot, bins=15, xrange=[7.5, 11.5])
    _, y_ejected, _, _ = compute_binned_median(log_stellar, f_ejected, bins=15, xrange=[7.5, 11.5])
    
    ax.plot(xbin, y_ism, '-', color=COLORS['ism'], lw=2, label='ISM (cold)')
    ax.plot(xbin, y_hot, '-', color=COLORS['halo'], lw=2, label='Hot halo')
    ax.plot(xbin, y_ejected, '-', color=COLORS['ejected'], lw=2, label='Ejected')
    
    ax.set_xlabel(r'$\log_{10}(M_\star/M_\odot)$', fontsize=14)
    ax.set_ylabel(r'$f_{\rm dust,X}$', fontsize=14)
    ax.set_xlim(7.5, 11.5)
    ax.set_ylim(0, 1)
    ax.legend(loc='best', fontsize=10)
    ax.set_title('Dust Reservoir Fractions', fontsize=14)
    
    plt.tight_layout()
    outfile = os.path.join(output_dir, f'plot11_dust_reservoirs{OUTPUT_FORMAT}')
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"  Saved: {outfile}")


def plot_12_btt_ssfr(data, output_dir):
    """Plot 12: BTT-sSFR Classification (from Paper2_plotting)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    stellar = data['StellarMass']
    bulge = data['BulgeMass']
    sfr_disk = data['SfrDisk']
    sfr_bulge = data['SfrBulge']
    cold_dust = data['ColdDust']
    metals_cold = data['MetalsColdGas']
    cold_gas = data['ColdGas']
    
    sfr = sfr_disk + sfr_bulge
    w = (stellar > 1e8) & (sfr > 0) & (cold_dust > 0)
    
    btt = np.zeros(len(stellar))
    btt[stellar > 0] = bulge[stellar > 0] / stellar[stellar > 0]
    log_ssfr = np.full(len(stellar), np.nan)
    log_ssfr[w] = np.log10(sfr[w] / stellar[w])
    
    Z_12logOH = compute_metallicity(metals_cold, cold_gas)
    
    # Panel 1: BTT vs sSFR colored by metallicity
    ax = axes[0]
    w_plot = w & np.isfinite(Z_12logOH)
    if np.sum(w_plot) > 100:
        sc = ax.scatter(log_ssfr[w_plot], btt[w_plot], c=Z_12logOH[w_plot],
                        s=5, cmap='viridis', alpha=0.5, vmin=7.5, vmax=9.2)
        plt.colorbar(sc, ax=ax, label=r'$12 + \log({\rm O/H})$')
    
    ax.axhline(0.5, color='grey', ls='--', alpha=0.5)
    ax.axvline(SSFR_CUT, color='grey', ls='--', alpha=0.5)
    ax.set_xlabel(r'$\log_{10}({\rm sSFR}/{\rm yr}^{-1})$', fontsize=14)
    ax.set_ylabel(r'$B/T$', fontsize=14)
    ax.set_xlim(-14, -8)
    ax.set_ylim(0, 1)
    ax.set_title('BTT-sSFR colored by metallicity', fontsize=12)
    
    # Panel 2: Classification
    ax = axes[1]
    
    # Classify galaxies
    star_forming = log_ssfr > SSFR_CUT
    disk_dom = btt < 0.5
    
    # 4 classes: SF disk, SF bulge, Q disk, Q bulge
    sf_disk = star_forming & disk_dom
    sf_bulge = star_forming & ~disk_dom
    q_disk = ~star_forming & disk_dom
    q_bulge = ~star_forming & ~disk_dom
    
    for i, (mask, label, color) in enumerate([
        (sf_disk, 'SF disk', CLASS_COLORS[0]),
        (sf_bulge, 'SF bulge', CLASS_COLORS[1]),
        (q_disk, 'Q disk', CLASS_COLORS[2]),
        (q_bulge, 'Q bulge', CLASS_COLORS[3]),
    ]):
        mask = mask & w
        if np.sum(mask) > 10:
            ax.scatter(log_ssfr[mask], btt[mask], c=color, s=5, alpha=0.5, label=label)
    
    ax.axhline(0.5, color='grey', ls='--', alpha=0.5)
    ax.axvline(SSFR_CUT, color='grey', ls='--', alpha=0.5)
    ax.set_xlabel(r'$\log_{10}({\rm sSFR}/{\rm yr}^{-1})$', fontsize=14)
    ax.set_ylabel(r'$B/T$', fontsize=14)
    ax.set_xlim(-14, -8)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper left', fontsize=10, markerscale=3)
    ax.set_title('Galaxy Classification', fontsize=12)
    
    plt.tight_layout()
    outfile = os.path.join(output_dir, f'plot12_btt_ssfr{OUTPUT_FORMAT}')
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"  Saved: {outfile}")


def plot_13_gmf(data, output_dir):
    """Plot 13: Gas Mass Function (HI, H2, Cold gas)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    cold_gas = data['ColdGas']
    h2_gas = data.get('H2gas', np.zeros_like(cold_gas))
    h1_gas = data.get('H1gas', np.zeros_like(cold_gas))
    
    # Cold gas MF
    x, y, yerr = compute_mass_function(cold_gas, VOLUME, nbins=25, mass_range=[7, 11.5])
    if x is not None:
        ax.plot(x, y, 'b-', lw=2, label='Cold gas')
    
    # H2 MF
    if np.sum(h2_gas > 0) > 100:
        x, y, yerr = compute_mass_function(h2_gas, VOLUME, nbins=25, mass_range=[6, 11])
        if x is not None:
            ax.plot(x, y, 'g-', lw=2, label=r'H$_2$')
    
    # HI MF
    if np.sum(h1_gas > 0) > 100:
        x, y, yerr = compute_mass_function(h1_gas, VOLUME, nbins=25, mass_range=[6, 11])
        if x is not None:
            ax.plot(x, y, 'r-', lw=2, label='HI')
    
    # Observations
    obs = load_gmf_observations()
    ax.errorbar(obs['Zwaan+05']['logMHI'], obs['Zwaan+05']['logphi'],
                yerr=obs['Zwaan+05']['err'], fmt='rs', ms=6, capsize=2,
                label='Zwaan+05 (HI)')
    ax.errorbar(obs['Obreschkow+09']['logMH2'], obs['Obreschkow+09']['logphi'],
                yerr=obs['Obreschkow+09']['err'], fmt='go', ms=6, capsize=2,
                label='Obreschkow+09 (H2)')
    
    ax.set_xlabel(r'$\log_{10}(M_{\rm gas}/M_\odot)$', fontsize=14)
    ax.set_ylabel(r'$\log_{10}(\phi / {\rm Mpc}^{-3}\,{\rm dex}^{-1})$', fontsize=14)
    ax.set_xlim(6, 11.5)
    ax.set_ylim(-5.5, -0.5)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_title('Gas Mass Function (z = 0)', fontsize=14)
    
    plt.tight_layout()
    outfile = os.path.join(output_dir, f'plot13_gmf{OUTPUT_FORMAT}')
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"  Saved: {outfile}")


def plot_14_smf(data, output_dir):
    """Plot 14: Stellar Mass Function."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    stellar = data['StellarMass']
    x, y, yerr = compute_mass_function(stellar, VOLUME, nbins=30, mass_range=[7, 12])
    if x is not None:
        ax.plot(x, y, 'b-', lw=2, label='SAGE26')
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.3, color='blue')
    
    # Observations
    obs = load_smf_observations()
    ax.errorbar(obs['logMstar'], obs['logphi'], yerr=obs['err'],
                fmt='ko', ms=6, capsize=2, label='Baldry+08')
    
    ax.set_xlabel(r'$\log_{10}(M_\star/M_\odot)$', fontsize=14)
    ax.set_ylabel(r'$\log_{10}(\phi / {\rm Mpc}^{-3}\,{\rm dex}^{-1})$', fontsize=14)
    ax.set_xlim(7, 12)
    ax.set_ylim(-5.5, -0.5)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_title('Stellar Mass Function (z = 0)', fontsize=14)
    
    plt.tight_layout()
    outfile = os.path.join(output_dir, f'plot14_smf{OUTPUT_FORMAT}')
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"  Saved: {outfile}")


def plot_15_all_dtg_evolution(all_data, output_dir):
    """Plot 15: DtG Evolution for all reservoirs."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for i, (snap, z) in enumerate(zip(EVOLUTION_SNAPS, EVOLUTION_REDSHIFTS)):
        ax = axes[i]
        data = all_data.get(snap)
        
        if data is not None and len(data.get('ColdDust', [])) > 0:
            stellar = data['StellarMass']
            cold_gas = data['ColdGas']
            cold_dust = data['ColdDust']
            
            w = (stellar > 1e7) & (cold_gas > 1e5) & (cold_dust > 0)
            if np.sum(w) > 10:
                log_stellar = np.log10(stellar[w])
                logDtG = np.log10(cold_dust[w] / cold_gas[w])
                
                xbin, ymed, p16, p84 = compute_binned_median(log_stellar, logDtG, bins=12, xrange=[7.5, 11.5])
                ax.plot(xbin, ymed, '-', color=COLORS['ism'], lw=2)
                ax.fill_between(xbin, p16, p84, alpha=0.3, color=COLORS['ism'])
        
        ax.text(0.95, 0.95, f'z = {z:.1f}', transform=ax.transAxes,
                ha='right', va='top', fontsize=12)
        ax.set_xlim(7.5, 11.5)
        ax.set_ylim(-5, -0.5)
    
    for ax in axes[4:]:
        ax.set_xlabel(r'$\log_{10}(M_\star/M_\odot)$', fontsize=12)
    for ax in [axes[0], axes[4]]:
        ax.set_ylabel(r'$\log_{10}({\rm DtG})$', fontsize=12)
    
    fig.suptitle('Dust-to-Gas Ratio Evolution', fontsize=14, y=1.02)
    plt.tight_layout()
    outfile = os.path.join(output_dir, f'plot15_dtg_evolution{OUTPUT_FORMAT}')
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outfile}")


def plot_16_dust_density_reservoirs(all_data, output_dir):
    """Plot 16: Dust Mass Density Evolution by reservoir."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    z_arr = []
    rho_ism = []
    rho_cgm = []
    rho_hot = []
    rho_ejected = []
    
    for snap in range(0, 64):
        data = all_data.get(snap)
        if data is None or len(data.get('ColdDust', [])) == 0:
            continue
        
        z = REDSHIFTS[snap]
        
        cold = np.sum(data['ColdDust'])
        hot = np.sum(data.get('HotDust', [0]))
        ejected = np.sum(data.get('EjectedDust', [0]))
        cgm = np.sum(data.get('CGMDust', [0]))
        
        z_arr.append(z)
        rho_ism.append(cold / VOLUME)
        rho_hot.append(hot / VOLUME)
        rho_ejected.append(ejected / VOLUME)
        rho_cgm.append(cgm / VOLUME)
    
    z_arr = np.array(z_arr)
    
    ax.plot(z_arr, np.log10(np.array(rho_ism) + 1e-10), '-', 
            color=COLORS['ism'], lw=2, label='ISM')
    ax.plot(z_arr, np.log10(np.array(rho_hot) + 1e-10), '-',
            color=COLORS['halo'], lw=2, label='Hot halo')
    ax.plot(z_arr, np.log10(np.array(rho_ejected) + 1e-10), '-',
            color=COLORS['ejected'], lw=2, label='Ejected')
    if np.sum(rho_cgm) > 0:
        ax.plot(z_arr, np.log10(np.array(rho_cgm) + 1e-10), '-',
                color='purple', lw=2, label='CGM')
    
    ax.set_xlabel('Redshift', fontsize=14)
    ax.set_ylabel(r'$\log_{10}(\rho_{\rm dust} / M_\odot\,{\rm Mpc}^{-3})$', fontsize=14)
    ax.set_xlim(0, 8)
    ax.set_ylim(2, 7)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_title('Dust Reservoir Density Evolution', fontsize=14)
    
    plt.tight_layout()
    outfile = os.path.join(output_dir, f'plot16_dust_density_reservoirs{OUTPUT_FORMAT}')
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"  Saved: {outfile}")


# ========================== MAIN ==========================

PLOT_FUNCTIONS = {
    1: ('Dust Mass Function (z=0)', plot_1_dmf_z0, False),
    2: ('Dust Mass Function Evolution', plot_2_dmf_evolution, True),
    3: ('Dust-Stellar Mass (z=0)', plot_3_dust_stellar_z0, False),
    4: ('Dust-Stellar Mass Evolution', plot_4_dust_stellar_evolution, True),
    5: ('DtG vs Metallicity', plot_5_dtg_metallicity, False),
    6: ('DtM vs Metallicity', plot_6_dtm_metallicity, False),
    7: ('DtG vs Stellar Mass', plot_7_dtg_stellar_mass, False),
    8: ('DtM vs Stellar Mass', plot_8_dtm_stellar_mass, False),
    9: ('Cosmic Dust Density', plot_9_cosmic_dust_density, True),
    10: ('Dust Rate Evolution', plot_10_dust_rates, True),
    11: ('Dust Reservoirs', plot_11_dust_reservoirs, False),
    12: ('BTT-sSFR Classification', plot_12_btt_ssfr, False),
    13: ('Gas Mass Function', plot_13_gmf, False),
    14: ('Stellar Mass Function', plot_14_smf, False),
    15: ('DtG Evolution', plot_15_all_dtg_evolution, True),
    16: ('Dust Density by Reservoir', plot_16_dust_density_reservoirs, True),
}


def main():
    parser = argparse.ArgumentParser(description='SAGE26 Dust Paper Plots')
    parser.add_argument('plots', nargs='*', type=int, help='Plot numbers to generate')
    parser.add_argument('--dir', default=DEFAULT_DIR, help='Model output directory')
    parser.add_argument('--list', action='store_true', help='List available plots')
    parser.add_argument('--dark', action='store_true', help='Use dark theme')
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable plots:")
        for num, (name, _, _) in sorted(PLOT_FUNCTIONS.items()):
            print(f"  {num:2d}. {name}")
        print("\nUsage: python dust_paper_plots.py [plot_numbers] --dir <output_dir>")
        return
    
    setup_style(dark=args.dark)
    
    # Determine which plots to generate
    if args.plots:
        plots_to_make = args.plots
    else:
        plots_to_make = list(PLOT_FUNCTIONS.keys())
    
    # Check which plots need evolution data (multiple snapshots)
    needs_evolution = any(PLOT_FUNCTIONS[p][2] for p in plots_to_make if p in PLOT_FUNCTIONS)
    needs_z0 = any(not PLOT_FUNCTIONS[p][2] for p in plots_to_make if p in PLOT_FUNCTIONS)
    
    # Setup output directory
    output_dir = os.path.join(args.dir, 'plots')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"\n{'='*60}")
    print("SAGE26 Dust Paper Plots")
    print(f"{'='*60}")
    print(f"Model directory: {args.dir}")
    print(f"Output directory: {output_dir}")
    
    # Load z=0 data if needed
    data_z0 = None
    if needs_z0:
        print(f"\nLoading z=0 data (Snap_{SNAP_Z0})...")
        data_z0 = load_snapshot(args.dir, SNAP_Z0)
        if data_z0 is None or len(data_z0.get('StellarMass', [])) == 0:
            print("  ERROR: Could not load z=0 data!")
            return
        print(f"  Loaded {len(data_z0['StellarMass']):,} galaxies")
    
    # Load evolution data if needed
    all_data = {}
    if needs_evolution:
        print("\nLoading evolution data...")
        # Load all snapshots for cosmic evolution plots
        for snap in range(0, 64, 4):  # Every 4th snapshot for efficiency
            print(f"  Loading snapshot {snap}...", end=' ')
            data = load_snapshot(args.dir, snap)
            if data is not None and len(data.get('StellarMass', [])) > 0:
                all_data[snap] = data
                print(f"{len(data['StellarMass']):,} galaxies")
            else:
                print("skipped")
        
        # Also load key snapshots
        for snap in EVOLUTION_SNAPS:
            if snap not in all_data:
                print(f"  Loading key snapshot {snap}...", end=' ')
                data = load_snapshot(args.dir, snap)
                if data is not None:
                    all_data[snap] = data
                    print(f"{len(data.get('StellarMass', [])):,} galaxies")
        
        if SNAP_Z0 in all_data and data_z0 is None:
            data_z0 = all_data[SNAP_Z0]
    
    # Generate plots
    print(f"\nGenerating {len(plots_to_make)} plots...")
    for plot_num in plots_to_make:
        if plot_num not in PLOT_FUNCTIONS:
            print(f"  Warning: Plot {plot_num} not found")
            continue
        
        name, func, needs_evol = PLOT_FUNCTIONS[plot_num]
        print(f"\n  Plot {plot_num}: {name}")
        
        try:
            if needs_evol:
                func(all_data, output_dir)
            else:
                func(data_z0, output_dir)
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

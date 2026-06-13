#!/usr/bin/env python
"""
SFH Downsizing and Mass-Weighted Age Analysis (MPI-enabled)
============================================================
Plots demonstrating downsizing using full star formation histories:

1. Mass-weighted ages vs stellar mass - when stars actually formed
2. Downsizing plot - formation timescales vs stellar mass
3. Formation vs Assembly histories by mass bin

Usage:
    Serial:   python sfh_downsizing.py
    Parallel: mpirun -np 4 python sfh_downsizing.py

Author: GitHub Copilot
"""

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import os

# MPI support (optional - falls back to serial if not available)
try:
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
    MPI_ENABLED = SIZE > 1
except ImportError:
    COMM = None
    RANK = 0
    SIZE = 1
    MPI_ENABLED = False

# ========================== CONFIGURATION ==========================

# File paths
PRIMARY_DIR = './output/millennium/'
MODEL_FILE = 'model_0.hdf5'
OUTPUT_DIR = './output/millennium/plots/'
OUTPUT_FORMAT = '.pdf'

# Simulation parameters
HUBBLE_H = 0.73
MASS_CONVERT = 1.0e10 / HUBBLE_H

# Galaxy selection
MIN_STELLAR_MASS = 1.0e8   # Solar masses
MAX_GALAXIES = 7500       # Max galaxies to analyze

# Mass bins for formation/assembly plot
MASS_BINS = [
    (1e8, 1e9, r'$10^8 - 10^9$ M$_\odot$', 'C0'),
    (1e9, 1e10, r'$10^9 - 10^{10}$ M$_\odot$', 'C1'),
    (1e10, 1e11, r'$10^{10} - 10^{11}$ M$_\odot$', 'C2'),
    (1e11, 1e12, r'$10^{11} - 10^{12}$ M$_\odot$', 'C3'),
]

# Cosmological parameters
OMEGA_M = 0.25
OMEGA_L = 0.75

# Number of SFH substeps per snapshot
STEPS = 10

# Redshift array (snap 0 -> snap 63)
REDSHIFTS = np.array([
    127.000, 79.998, 50.000, 30.000, 19.916, 18.244, 16.725, 15.343,
     14.086, 12.941, 11.897, 10.944, 10.073,  9.278,  8.550,  7.883,
      7.272,  6.712,  6.197,  5.724,  5.289,  4.888,  4.520,  4.179,
      3.866,  3.576,  3.308,  3.060,  2.831,  2.619,  2.422,  2.239,
      2.070,  1.913,  1.766,  1.630,  1.504,  1.386,  1.276,  1.173,
      1.078,  0.989,  0.905,  0.828,  0.755,  0.687,  0.624,  0.564,
      0.509,  0.457,  0.408,  0.362,  0.320,  0.280,  0.242,  0.208,
      0.175,  0.144,  0.116,  0.089,  0.064,  0.041,  0.020,  0.000,
])

# ========================== HELPER FUNCTIONS ==========================

def cosmic_time_gyr(z):
    """Age of the universe at redshift z, in Gyr."""
    t_H = 977.8 / (HUBBLE_H * 100)
    def integrand(zp):
        return 1.0 / ((1 + zp) * np.sqrt(OMEGA_M * (1 + zp)**3 + OMEGA_L))
    result, _ = quad(integrand, z, 1000.0)
    return t_H * result


def lookback_time_gyr(z):
    """Lookback time at redshift z, in Gyr."""
    t_now = cosmic_time_gyr(0.0)
    return t_now - cosmic_time_gyr(z)


# Pre-compute times
LOOKBACK_TIMES = np.array([lookback_time_gyr(z) for z in REDSHIFTS])
COSMIC_TIMES = np.array([cosmic_time_gyr(z) for z in REDSHIFTS])
T_NOW = cosmic_time_gyr(0.0)


def setup_style():
    """Configure matplotlib for publication-quality plots."""
    style_path = "./plotting/ciaran_ohare_palatino_sty.mplstyle"
    if os.path.exists(style_path):
        plt.style.use(style_path)
    else:
        plt.rcParams["figure.figsize"] = (8, 6)
        plt.rcParams["figure.dpi"] = 150
        plt.rcParams["font.size"] = 14


def trace_main_progenitor(f, galaxy_index, start_snap=63):
    """
    Trace main progenitor of a galaxy back through time.
    Returns dict with snap -> (array_index, GalaxyIndex).
    
    Only used for assembly history - formation now uses SFHMass arrays.
    """
    progenitor_chain = {}
    current_gal_idx = galaxy_index
    
    for snap in range(start_snap, -1, -1):
        snap_key = f'Snap_{snap}'
        if snap_key not in f:
            continue
            
        gal_indices = f[snap_key]['GalaxyIndex'][:]
        matches = np.where(gal_indices == current_gal_idx)[0]
        if len(matches) == 0:
            break
            
        arr_idx = matches[0]
        progenitor_chain[snap] = (arr_idx, current_gal_idx)
        
    return progenitor_chain


def get_sfh_from_arrays(f, arr_idx, snap_key='Snap_63'):
    """
    Extract full SFH for a galaxy using SFHMassDisk/SFHMassBulge arrays.
    
    These arrays track when ALL stars in the final galaxy were born,
    including stars brought in by mergers (ex-situ).
    
    Returns:
        cosmic_times: Array of cosmic times (Gyr) - when stars formed
        masses: Array of stellar mass formed at each snapshot (Msun)
        lookback_times: Array of lookback times (Gyr)
    """
    # Read SFH arrays - these give mass of stars born at each snapshot
    sfh_disk = f[snap_key]['SFHMassDisk'][arr_idx] * MASS_CONVERT
    sfh_bulge = f[snap_key]['SFHMassBulge'][arr_idx] * MASS_CONVERT
    sfh_total = sfh_disk + sfh_bulge
    
    # Map to times - each entry corresponds to a snapshot
    n_snaps = len(sfh_total)
    cosmic_times = COSMIC_TIMES[:n_snaps]
    lookback_times = LOOKBACK_TIMES[:n_snaps]
    
    return cosmic_times, sfh_total, lookback_times


def calculate_mass_weighted_age(cosmic_times, masses):
    """
    Calculate mass-weighted age from SFH mass arrays.
    
    Mass-weighted age = Σ(M_i * t_i) / Σ(M_i)
    where M_i is mass of stars formed at snapshot i, t_i is cosmic time.
    
    Uses SFHMassDisk + SFHMassBulge which track when ALL stars were born.
    
    Returns age in Gyr (cosmic time of formation).
    """
    if len(cosmic_times) < 2:
        return np.nan
    
    total_mass = np.sum(masses)
    if total_mass == 0:
        return np.nan
    
    # Mass-weighted mean cosmic time
    mean_formation_time = np.sum(masses * cosmic_times) / total_mass
    return mean_formation_time


def calculate_formation_times(lookback_times, masses):
    """
    Calculate formation timescales from SFH mass arrays.
    
    Uses SFHMassDisk + SFHMassBulge which track when ALL stars were born.
    
    Returns:
        t50: Lookback time when 50% of stars had formed
        t90: Lookback time when 90% of stars had formed
    """
    total_mass = np.sum(masses)
    if total_mass == 0:
        return np.nan, np.nan
    
    # Sort by lookback time (oldest first, i.e. highest lookback first)
    sort_idx = np.argsort(lookback_times)[::-1]
    lookback_sorted = lookback_times[sort_idx]
    masses_sorted = masses[sort_idx]
    
    # Calculate cumulative mass formed (from old to young)
    cumulative_mass = np.cumsum(masses_sorted)
    cumulative_frac = cumulative_mass / total_mass
    
    # Find t50 and t90 (lookback time when 50%/90% had formed)
    t50 = np.interp(0.5, cumulative_frac, lookback_sorted)
    t90 = np.interp(0.9, cumulative_frac, lookback_sorted)
    
    return t50, t90


def calculate_formation_assembly(f, arr_idx_z0, progenitor_chain, final_mstar):
    """
    Calculate formation and assembly histories for a galaxy.
    
    Formation: Uses SFHMassDisk + SFHMassBulge arrays (when ALL stars formed,
               including ex-situ stars brought in by mergers)
    Assembly: M*(z) / M*(z=0) (when stars were assembled into main progenitor)
    
    Returns:
        lookback_times: Array of lookback times
        formation_frac: Formation fraction at each time
        assembly_frac: Assembly fraction at each time
    """
    # Get formation history from SFH arrays at z=0
    sfh_disk = f['Snap_63']['SFHMassDisk'][arr_idx_z0] * MASS_CONVERT
    sfh_bulge = f['Snap_63']['SFHMassBulge'][arr_idx_z0] * MASS_CONVERT
    sfh_total = sfh_disk + sfh_bulge
    
    n_snaps = len(sfh_total)
    sfh_lookback = LOOKBACK_TIMES[:n_snaps]
    
    total_formed = np.sum(sfh_total)
    if total_formed == 0:
        return None, None, None
    
    # Sort by lookback time (oldest first)
    sort_idx = np.argsort(sfh_lookback)[::-1]
    sfh_lookback_sorted = sfh_lookback[sort_idx]
    sfh_sorted = sfh_total[sort_idx]
    
    # Cumulative formation fraction
    cumulative_formed = np.cumsum(sfh_sorted)
    formation_frac = cumulative_formed / total_formed
    
    # Get assembly history from progenitor chain
    sorted_snaps = sorted(progenitor_chain.keys())
    snap_lookback = []
    snap_mstar = []
    
    for snap in sorted_snaps:
        prog_arr_idx, _ = progenitor_chain[snap]
        snap_key = f'Snap_{snap}'
        mstar = f[snap_key]['StellarMass'][prog_arr_idx] * MASS_CONVERT
        snap_mstar.append(mstar)
        snap_lookback.append(LOOKBACK_TIMES[snap])
    
    if len(snap_lookback) < 10:
        return None, None, None
    
    snap_lookback = np.array(snap_lookback)
    snap_mstar = np.array(snap_mstar)
    
    # Interpolate assembly to formation time grid
    assembly_frac = np.interp(sfh_lookback_sorted, snap_lookback[::-1], snap_mstar[::-1]) / final_mstar
    
    return sfh_lookback_sorted, formation_frac, assembly_frac


def analyze_galaxy(f, arr_idx, gal_indices, mstar):
    """
    Analyze a single galaxy's SFH using SFHMassDisk/SFHMassBulge arrays.
    
    Formation history now uses the new SFH arrays which track when ALL stars
    were born (including ex-situ stars from mergers). Assembly still traces
    the main progenitor.
    
    Returns dict with galaxy data, or None if invalid.
    """
    gal_idx = gal_indices[arr_idx]
    final_mstar = mstar[arr_idx]
    
    # Get SFH from arrays at z=0 (tracks ALL stars, including ex-situ)
    cosmic_times, masses, lookback_times = get_sfh_from_arrays(f, arr_idx)
    
    if np.sum(masses) == 0:
        return None
    
    # Calculate mass-weighted age
    mw_age = calculate_mass_weighted_age(cosmic_times, masses)
    
    # Calculate formation times
    t50, t90 = calculate_formation_times(lookback_times, masses)
    
    # Trace progenitor for assembly history
    chain = trace_main_progenitor(f, gal_idx)
    
    if len(chain) < 20:
        return None
    
    # Calculate formation/assembly
    fa_times, formation, assembly = calculate_formation_assembly(f, arr_idx, chain, final_mstar)
    
    # Convert arrays to lists for MPI serialization
    return {
        'mstar': final_mstar,
        'mw_age': mw_age,
        't50': t50,
        't90': t90,
        'fa_times': fa_times.tolist() if fa_times is not None else None,
        'formation': formation.tolist() if formation is not None else None,
        'assembly': assembly.tolist() if assembly is not None else None
    }


def main():
    """Main function to create downsizing and mass-weighted age plots."""
    setup_style()
    
    model_path = os.path.join(PRIMARY_DIR, MODEL_FILE)
    
    # Only rank 0 prints header
    if RANK == 0:
        print("SFH Downsizing and Mass-Weighted Age Analysis")
        print("=" * 60)
        if MPI_ENABLED:
            print(f"  Running with MPI: {SIZE} processes")
        else:
            print("  Running in serial mode")
        print(f"  Model file: {model_path}")
        print(f"  Min stellar mass: {MIN_STELLAR_MASS:.1e} Msun")
        print()
    
    with h5.File(model_path, 'r') as f:
        # Select galaxies at z=0 (all ranks read this)
        snap_key = 'Snap_63'
        mstar = f[snap_key]['StellarMass'][:] * MASS_CONVERT
        gal_type = f[snap_key]['Type'][:]
        gal_indices = f[snap_key]['GalaxyIndex'][:]
        
        # Rank 0 selects galaxies and broadcasts
        if RANK == 0:
            mask = (mstar >= MIN_STELLAR_MASS) & (gal_type == 0)
            selected_indices = np.where(mask)[0]
            
            # Limit sample size
            if len(selected_indices) > MAX_GALAXIES:
                np.random.seed(42)
                selected_indices = np.random.choice(selected_indices, MAX_GALAXIES, replace=False)
            
            print(f"Selected {len(selected_indices)} centrals at z=0")
        else:
            selected_indices = None
        
        # Broadcast selected indices to all ranks
        if MPI_ENABLED:
            selected_indices = COMM.bcast(selected_indices, root=0)
        
        # Divide work among ranks
        n_total = len(selected_indices)
        n_per_rank = n_total // SIZE
        remainder = n_total % SIZE
        
        # Calculate start and end indices for this rank
        if RANK < remainder:
            start_idx = RANK * (n_per_rank + 1)
            end_idx = start_idx + n_per_rank + 1
        else:
            start_idx = RANK * n_per_rank + remainder
            end_idx = start_idx + n_per_rank
        
        my_indices = selected_indices[start_idx:end_idx]
        
        if RANK == 0:
            print(f"\nAnalyzing SFHs...")
            if MPI_ENABLED:
                print(f"  Each rank processing ~{n_per_rank} galaxies")
        
        # Each rank processes its subset
        local_galaxy_data = []
        
        for i, arr_idx in enumerate(my_indices):
            if RANK == 0 and (i + 1) % 500 == 0:
                print(f"  Rank 0: Processed {i + 1}/{len(my_indices)}")
            
            result = analyze_galaxy(f, arr_idx, gal_indices, mstar)
            if result is not None:
                local_galaxy_data.append(result)
        
        if RANK == 0:
            print(f"  Rank 0 analyzed {len(local_galaxy_data)} galaxies locally")
    
    # Gather results from all ranks to rank 0
    if MPI_ENABLED:
        all_galaxy_data = COMM.gather(local_galaxy_data, root=0)
        if RANK == 0:
            # Flatten the list of lists
            galaxy_data = []
            for rank_data in all_galaxy_data:
                galaxy_data.extend(rank_data)
            print(f"  Total gathered: {len(galaxy_data)} galaxies from {SIZE} ranks")
    else:
        galaxy_data = local_galaxy_data
    
    # Only rank 0 does the plotting
    if RANK != 0:
        return
    
    # Convert arrays back from lists (after MPI gather)
    for g in galaxy_data:
        if g['fa_times'] is not None:
            g['fa_times'] = np.array(g['fa_times'])
            g['formation'] = np.array(g['formation'])
            g['assembly'] = np.array(g['assembly'])
    
    print(f"\n  Analyzed {len(galaxy_data)} galaxies total")
    print()
    
    # Convert to arrays
    stellar_masses = np.array([g['mstar'] for g in galaxy_data])
    mw_ages = np.array([g['mw_age'] for g in galaxy_data])
    t50s = np.array([g['t50'] for g in galaxy_data])
    t90s = np.array([g['t90'] for g in galaxy_data])
    
    # Filter NaNs
    valid = np.isfinite(mw_ages) & np.isfinite(t50s)
    stellar_masses = stellar_masses[valid]
    mw_ages = mw_ages[valid]
    t50s = t50s[valid]
    t90s = t90s[valid]
    galaxy_data = [g for g, v in zip(galaxy_data, valid) if v]
    
    print(f"  {len(galaxy_data)} galaxies with valid SFH data")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # ==================== PLOT 1: Mass-Weighted Age vs Stellar Mass ====================
    print("\nCreating Plot 1: Mass-Weighted Age vs Stellar Mass...")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Scatter plot
    sc = ax.scatter(stellar_masses, mw_ages, c=t50s, cmap='viridis_r', 
                   alpha=0.6, s=15, edgecolors='none')
    
    # Add colorbar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('$t_{50}$ [Gyr lookback]')
    
    # Binned median
    log_mass = np.log10(stellar_masses)
    bins = np.linspace(9, 12, 13)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    median_age = []
    p16_age = []
    p84_age = []
    
    for i in range(len(bins) - 1):
        in_bin = (log_mass >= bins[i]) & (log_mass < bins[i+1])
        if np.sum(in_bin) > 5:
            median_age.append(np.median(mw_ages[in_bin]))
            p16_age.append(np.percentile(mw_ages[in_bin], 16))
            p84_age.append(np.percentile(mw_ages[in_bin], 84))
        else:
            median_age.append(np.nan)
            p16_age.append(np.nan)
            p84_age.append(np.nan)
    
    median_age = np.array(median_age)
    p16_age = np.array(p16_age)
    p84_age = np.array(p84_age)
    
    valid_bins = np.isfinite(median_age)
    ax.plot(10**bin_centers[valid_bins], median_age[valid_bins], 
           'k-', linewidth=2.5, label='Median')
    # ax.fill_between(10**bin_centers[valid_bins], p16_age[valid_bins], p84_age[valid_bins],
    #                color='red', alpha=0.2, label='16-84th percentile')
    
    ax.set_xscale('log')
    ax.set_xlabel(r'Stellar Mass [M$_\odot$]')
    ax.set_ylabel('Mass-Weighted Age [Gyr]')
    ax.set_title('Mass-Weighted Stellar Ages')
    ax.set_xlim(1e9, 1e12)
    ax.set_ylim(0, T_NOW)
    ax.legend(loc='lower right')
    
    # Add redshift axis
    ax2 = ax.twinx()
    z_for_age = [0.5, 1, 2, 3, 5]
    age_for_z = [cosmic_time_gyr(z) for z in z_for_age]
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(age_for_z)
    ax2.set_yticklabels([f'z={z}' for z in z_for_age])
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, f'mass_weighted_age{OUTPUT_FORMAT}')
    plt.savefig(output_path, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()
    
    # ==================== PLOT 2: Downsizing (t50, t90 vs Mass) ====================
    print("\nCreating Plot 2: Downsizing - Formation Timescales...")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Binned statistics
    median_t50 = []
    median_t90 = []
    p16_t50, p84_t50 = [], []
    p16_t90, p84_t90 = [], []
    
    for i in range(len(bins) - 1):
        in_bin = (log_mass >= bins[i]) & (log_mass < bins[i+1])
        if np.sum(in_bin) > 5:
            median_t50.append(np.median(t50s[in_bin]))
            median_t90.append(np.median(t90s[in_bin]))
            p16_t50.append(np.percentile(t50s[in_bin], 16))
            p84_t50.append(np.percentile(t50s[in_bin], 84))
            p16_t90.append(np.percentile(t90s[in_bin], 16))
            p84_t90.append(np.percentile(t90s[in_bin], 84))
        else:
            median_t50.append(np.nan)
            median_t90.append(np.nan)
            p16_t50.append(np.nan)
            p84_t50.append(np.nan)
            p16_t90.append(np.nan)
            p84_t90.append(np.nan)
    
    median_t50 = np.array(median_t50)
    median_t90 = np.array(median_t90)
    p16_t50 = np.array(p16_t50)
    p84_t50 = np.array(p84_t50)
    p16_t90 = np.array(p16_t90)
    p84_t90 = np.array(p84_t90)
    
    valid_bins = np.isfinite(median_t50)
    
    # Plot t50
    ax.plot(10**bin_centers[valid_bins], median_t50[valid_bins], 
           'C0-', linewidth=2.5, label=r'$t_{50}$ (50% formed)')
    ax.fill_between(10**bin_centers[valid_bins], p16_t50[valid_bins], p84_t50[valid_bins],
                   color='C0', alpha=0.2)
    
    # Plot t90
    ax.plot(10**bin_centers[valid_bins], median_t90[valid_bins], 
           'C1-', linewidth=2.5, label=r'$t_{90}$ (90% formed)')
    ax.fill_between(10**bin_centers[valid_bins], p16_t90[valid_bins], p84_t90[valid_bins],
                   color='C1', alpha=0.2)
    
    ax.set_xscale('log')
    ax.set_xlabel(r'Stellar Mass [M$_\odot$]')
    ax.set_ylabel('Lookback Time [Gyr]')
    ax.set_title('Downsizing: Formation Timescales vs Stellar Mass')
    ax.set_xlim(1e9, 1e12)
    ax.set_ylim(0, 13)
    ax.legend(loc='upper left')
    
    # Add redshift axis
    ax2 = ax.twinx()
    z_ticks = [0, 0.5, 1, 2, 3, 5]
    lookback_for_z = [lookback_time_gyr(z) for z in z_ticks]
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(lookback_for_z)
    ax2.set_yticklabels([f'z={z}' for z in z_ticks])
    
    # Add annotation
    ax.annotate('More massive galaxies\nformed earlier', 
               xy=(3e11, 9), fontsize=10, ha='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, f'downsizing{OUTPUT_FORMAT}')
    plt.savefig(output_path, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()
    
    # ==================== PLOT 3: Formation vs Assembly by Mass Bin ====================
    print("\nCreating Plot 3: Formation vs Assembly by Mass Bin...")
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for mass_min, mass_max, label, color in MASS_BINS:
        # Select galaxies in this mass bin
        in_bin = (stellar_masses >= mass_min) & (stellar_masses < mass_max)
        bin_data = [g for g, m in zip(galaxy_data, in_bin) if m and g['fa_times'] is not None]
        
        if len(bin_data) < 5:
            continue
        
        print(f"    {label}: {len(bin_data)} galaxies")
        
        # Common time grid
        time_grid = np.linspace(0, 13, 100)
        
        # Interpolate all galaxies to common grid
        formation_all = []
        assembly_all = []
        
        for g in bin_data:
            if g['fa_times'] is None:
                continue
            # Interpolate formation
            f_interp = np.interp(time_grid, g['fa_times'][::-1], g['formation'][::-1],
                                left=0, right=1)
            formation_all.append(f_interp)
            
            # Interpolate assembly
            a_interp = np.interp(time_grid, g['fa_times'][::-1], g['assembly'][::-1],
                                left=0, right=1)
            assembly_all.append(a_interp)
        
        formation_all = np.array(formation_all)
        assembly_all = np.array(assembly_all)
        
        # Calculate median and percentiles
        formation_median = np.median(formation_all, axis=0)
        assembly_median = np.median(assembly_all, axis=0)
        
        formation_p16 = np.percentile(formation_all, 16, axis=0)
        formation_p84 = np.percentile(formation_all, 84, axis=0)
        assembly_p16 = np.percentile(assembly_all, 16, axis=0)
        assembly_p84 = np.percentile(assembly_all, 84, axis=0)
        
        # Plot formation (dashed)
        ax.plot(time_grid, formation_median, '--', color=color, linewidth=2, 
               label=f'{label} (Formation)')
        # ax.fill_between(time_grid, formation_p16, formation_p84, color=color, alpha=0.1)
        
        # Plot assembly (solid)
        ax.plot(time_grid, assembly_median, '-', color=color, linewidth=2.5,
               label=f'{label} (Assembly)')
        # ax.fill_between(time_grid, assembly_p16, assembly_p84, color=color, alpha=0.2)
    
    ax.set_xlabel('Lookback Time [Gyr]')
    ax.set_ylabel('Fraction')
    ax.set_title('Formation vs Assembly Histories by Stellar Mass')
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower left', fontsize=9, ncol=2)
    
    # Add redshift axis on top
    ax2 = ax.twiny()
    z_ticks = [0, 0.5, 1, 2, 3, 5, 10]
    z_tick_pos = [lookback_time_gyr(z) for z in z_ticks]
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(z_tick_pos)
    ax2.set_xticklabels([f'{z}' for z in z_ticks])
    ax2.set_xlabel('Redshift')
    
    # Add annotation
    # ax.annotate('Dashed = Formation (when stars formed)\nSolid = Assembly (in main progenitor)', 
    #            xy=(10, 0.15), fontsize=9, ha='left',
    #            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, f'formation_vs_assembly{OUTPUT_FORMAT}')
    plt.savefig(output_path, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics:")
    print("=" * 60)
    
    for mass_min, mass_max, label, _ in MASS_BINS:
        in_bin = (stellar_masses >= mass_min) & (stellar_masses < mass_max)
        if np.sum(in_bin) > 0:
            print(f"\n{label}:")
            print(f"  N galaxies: {np.sum(in_bin)}")
            print(f"  Mass-weighted age: {np.median(mw_ages[in_bin]):.1f} Gyr "
                  f"({np.percentile(mw_ages[in_bin], 16):.1f} - {np.percentile(mw_ages[in_bin], 84):.1f})")
            print(f"  t50: {np.median(t50s[in_bin]):.1f} Gyr lookback "
                  f"({np.percentile(t50s[in_bin], 16):.1f} - {np.percentile(t50s[in_bin], 84):.1f})")
            print(f"  t90: {np.median(t90s[in_bin]):.1f} Gyr lookback "
                  f"({np.percentile(t90s[in_bin], 16):.1f} - {np.percentile(t90s[in_bin], 84):.1f})")
    
    print("\n\nAll plots completed!")


if __name__ == '__main__':
    main()

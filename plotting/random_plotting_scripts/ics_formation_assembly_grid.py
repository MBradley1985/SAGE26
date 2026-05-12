#!/usr/bin/env python
"""
ICS Assembly History Grid Plot
==============================
Grid of panels showing ICS assembly histories for different 
halo mass bins, from 10^10 to 10^14 Msun.

Assembly is tracked via two channels:
- ICS_disrupt: mass converted to ICS through satellite disruption
- ICS_accrete: pre-existing ICS accreted from satellites

Total Assembly: ICS(z) / ICS(z=0) - when ICS accumulated
Disruption: ICS_disrupt(z) / ICS_disrupt(z=0)  
Accretion: ICS_accrete(z) / ICS_accrete(z=0)
"""

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import os

# ========================== CONFIGURATION ==========================

# File paths
PRIMARY_DIR = './output/millennium/'
MODEL_FILE = 'model_0.hdf5'
OUTPUT_DIR = './output/millennium/plots/'
OUTPUT_FORMAT = '.pdf'

# Simulation parameters
HUBBLE_H = 0.73
MASS_CONVERT = 1.0e10 / HUBBLE_H

# Mass bins for the grid (min, max, label)
MASS_BINS = [
    (1e10, 1e11, r'$10^{10} < M_{\rm vir} < 10^{11}$'),
    (1e11, 1e12, r'$10^{11} < M_{\rm vir} < 10^{12}$'),
    (1e12, 1e13, r'$10^{12} < M_{\rm vir} < 10^{13}$'),
    (1e13, 1e14, r'$10^{13} < M_{\rm vir} < 10^{14}$'),
    (1e14, 1e15, r'$10^{14} < M_{\rm vir} < 10^{15}$'),
    (1e15, 1e17, r'$M_{\rm vir} > 10^{15}$'),
]

N_MAX_PER_BIN = 7500  # Maximum halos per mass bin

# Cosmological parameters
OMEGA_M = 0.25
OMEGA_L = 0.75

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


# Pre-compute lookback times
LOOKBACK_TIMES = np.array([lookback_time_gyr(z) for z in REDSHIFTS])


def setup_style():
    """Configure matplotlib for publication-quality plots."""
    style_path = "./plotting/ciaran_ohare_palatino_sty.mplstyle"
    if os.path.exists(style_path):
        plt.style.use(style_path)
    else:
        plt.rcParams["figure.figsize"] = (12, 10)
        plt.rcParams["figure.dpi"] = 150
        plt.rcParams["font.size"] = 11


def select_halos_in_mass_bin(f, mass_min, mass_max, snap=63, n_max=N_MAX_PER_BIN):
    """
    Select central galaxies in a given halo mass bin at z=0.
    
    Returns:
        gal_indices: GalaxyIndex values for selected halos
        arr_indices: Array indices at z=0
        n_selected: Number selected
    """
    snap_key = f'Snap_{snap}'
    
    mvir = f[snap_key]['Mvir'][:] * MASS_CONVERT
    galtype = f[snap_key]['Type'][:]
    gal_idx = f[snap_key]['GalaxyIndex'][:]
    ics = f[snap_key]['IntraClusterStars'][:] * MASS_CONVERT
    
    # Select centrals in mass bin with ICS > 0
    mask = (galtype == 0) & (mvir >= mass_min) & (mvir < mass_max) & (ics > 0)
    selected = np.where(mask)[0]
    
    if len(selected) == 0:
        return np.array([]), np.array([]), 0
    
    # Randomly sample if too many
    if len(selected) > n_max:
        np.random.seed(42)
        selected = np.random.choice(selected, n_max, replace=False)
    
    return gal_idx[selected], selected, len(selected)


def compute_ics_histories(f, gal_indices, arr_indices_z0, snaps=None):
    """
    Compute assembly histories for ICS via disruption and accretion channels.
    
    Assembly is tracked via two channels:
    - ICS_disrupt: mass converted to ICS through satellite disruption  
    - ICS_accrete: pre-existing ICS accreted from satellites
    
    Returns:
        lookback_times: Array of lookback times
        assembly: 2D array (n_halo, n_snaps) of total assembly fractions
        disrupt: 2D array (n_halo, n_snaps) of disruption channel fractions
        accrete: 2D array (n_halo, n_snaps) of accretion channel fractions
    """
    if snaps is None:
        snaps = list(range(1, 64))
    
    n_halo = len(gal_indices)
    n_snaps = len(snaps)
    sorted_snaps = sorted(snaps)
    
    # Get final values at z=0
    sorted_order = np.argsort(arr_indices_z0)
    sorted_indices = arr_indices_z0[sorted_order]
    
    ics_z0_sorted = f['Snap_63']['IntraClusterStars'][sorted_indices] * MASS_CONVERT
    disrupt_z0_sorted = f['Snap_63']['ICS_disrupt'][sorted_indices] * MASS_CONVERT
    accrete_z0_sorted = f['Snap_63']['ICS_accrete'][sorted_indices] * MASS_CONVERT
    
    # Restore original order
    ics_z0 = np.empty_like(ics_z0_sorted)
    ics_z0[sorted_order] = ics_z0_sorted
    
    disrupt_z0 = np.empty_like(disrupt_z0_sorted)
    disrupt_z0[sorted_order] = disrupt_z0_sorted
    
    accrete_z0 = np.empty_like(accrete_z0_sorted)
    accrete_z0[sorted_order] = accrete_z0_sorted
    
    # Avoid division by zero
    ics_z0_safe = np.maximum(ics_z0, 1e-10)
    disrupt_z0_safe = np.maximum(disrupt_z0, 1e-10)
    accrete_z0_safe = np.maximum(accrete_z0, 1e-10)
    
    # Initialize output
    assembly = np.full((n_halo, n_snaps), np.nan)
    disrupt = np.full((n_halo, n_snaps), np.nan)
    accrete = np.full((n_halo, n_snaps), np.nan)
    
    # Lookup table
    gal_to_idx = {gid: i for i, gid in enumerate(gal_indices)}
    
    for si, snap in enumerate(sorted_snaps):
        snap_key = f'Snap_{snap}'
        if snap_key not in f:
            continue
        
        gal_indices_snap = f[snap_key]['GalaxyIndex'][:]
        
        # Find halos present in this snapshot
        halo_mask = np.isin(gal_indices_snap, gal_indices)
        if not np.any(halo_mask):
            continue
        
        halo_arr_indices = np.where(halo_mask)[0]
        halo_gal_ids = gal_indices_snap[halo_arr_indices]
        halo_out_indices = np.array([gal_to_idx[gid] for gid in halo_gal_ids])
        
        # Total ICS assembly
        ics_snap = f[snap_key]['IntraClusterStars'][halo_arr_indices] * MASS_CONVERT
        assembly[halo_out_indices, si] = ics_snap / ics_z0_safe[halo_out_indices]
        
        # Disruption channel
        disrupt_snap = f[snap_key]['ICS_disrupt'][halo_arr_indices] * MASS_CONVERT
        disrupt[halo_out_indices, si] = disrupt_snap / disrupt_z0_safe[halo_out_indices]
        
        # Accretion channel
        accrete_snap = f[snap_key]['ICS_accrete'][halo_arr_indices] * MASS_CONVERT
        accrete[halo_out_indices, si] = accrete_snap / accrete_z0_safe[halo_out_indices]
    
    # Get lookback times for these snapshots
    lookback = np.array([LOOKBACK_TIMES[s] for s in sorted_snaps])
    
    return lookback, assembly, disrupt, accrete


def main():
    """Main function to create the grid plot."""
    setup_style()
    
    filepath = os.path.join(PRIMARY_DIR, MODEL_FILE)
    print(f"Loading data from {filepath}")
    print("=" * 60)
    
    f = h5.File(filepath, 'r')
    
    # Determine grid layout
    n_bins = len(MASS_BINS)
    n_cols = 3
    n_rows = (n_bins + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows), 
                             sharex=True, sharey=True)
    axes = axes.flatten()
    
    # Colors
    blue = '#2166AC'
    green = '#1B7837'
    
    # Process each mass bin
    for i, (mass_min, mass_max, label) in enumerate(MASS_BINS):
        ax = axes[i]
        
        print(f"\n{label}:")
        
        # Select halos
        gal_indices, arr_indices, n_sel = select_halos_in_mass_bin(
            f, mass_min, mass_max
        )
        
        if n_sel == 0:
            print("  No halos found")
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.set_title(label, fontsize=11)
            continue
        
        print(f"  Selected {n_sel} halos")
        
        # Compute histories
        lookback, assembly, disrupt, accrete = compute_ics_histories(
            f, gal_indices, arr_indices
        )
        
        # Compute percentiles
        disrupt_median = np.nanmedian(disrupt, axis=0)
        disrupt_15 = np.nanpercentile(disrupt, 15, axis=0)
        disrupt_85 = np.nanpercentile(disrupt, 85, axis=0)
        
        accrete_median = np.nanmedian(accrete, axis=0)
        accrete_15 = np.nanpercentile(accrete, 15, axis=0)
        accrete_85 = np.nanpercentile(accrete, 85, axis=0)
        
        # Plot disruption channel (solid green)
        ax.fill_between(lookback, disrupt_15, disrupt_85, 
                        color='#1B7837', alpha=0.3, edgecolor='none')
        ax.plot(lookback, disrupt_median, '--', color='#1B7837', lw=2, 
                label='Disruption')
        
        # Plot accretion channel (solid blue)
        ax.fill_between(lookback, accrete_15, accrete_85, 
                        color='#2166AC', alpha=0.3, edgecolor='none')
        ax.plot(lookback, accrete_median, '-', color='#2166AC', lw=2, 
                label='Accretion')
        
        # Panel label
        ax.set_title(label, fontsize=11)
        
        ax.set_xlim(0, 13)
        ax.set_ylim(0, 1.05)
    
    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    # Add common labels
    fig.text(0.5, 0.02, 'Lookback time [Gyr]', ha='center', fontsize=14)
    fig.text(0.02, 0.5, r'Fraction of $z=0$ ICS', va='center', 
             rotation='vertical', fontsize=14)
    
    # Add legend to first panel
    axes[0].legend(loc='upper right', frameon=False, fontsize=10)
    
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.97])
    
    # Save
    output_path = os.path.join(OUTPUT_DIR, f'ICS_assembly_history_grid{OUTPUT_FORMAT}')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    
    plt.close()
    f.close()
    
    print("\nDone!")


if __name__ == '__main__':
    main()

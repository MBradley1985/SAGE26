#!/usr/bin/env python
"""
BCG Formation and Assembly History Plot
=======================================
Plots the assembly (blue) and formation (green) histories of BCGs selected 
at redshift 0. Assembly history shows the fraction of final stellar mass 
that has been assembled in the main progenitor. Formation history shows the 
fraction of final stellar mass that has been formed.

Thick lines show the median, shaded regions show 15th to 85th percentiles.
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

# BCG selection criteria
MIN_HALO_MASS = 1.0e13  # Solar masses - select clusters
N_BCG_MAX = 10000  # Maximum number of BCGs to process

# Cosmological parameters
OMEGA_M = 0.25
OMEGA_L = 0.75

# Redshift array (snap 0 -> snap 63)
REDSHIFTS = [
    127.000, 79.998, 50.000, 30.000, 19.916, 18.244, 16.725, 15.343,
     14.086, 12.941, 11.897, 10.944, 10.073,  9.278,  8.550,  7.883,
      7.272,  6.712,  6.197,  5.724,  5.289,  4.888,  4.520,  4.179,
      3.866,  3.576,  3.308,  3.060,  2.831,  2.619,  2.422,  2.239,
      2.070,  1.913,  1.766,  1.630,  1.504,  1.386,  1.276,  1.173,
      1.078,  0.989,  0.905,  0.828,  0.755,  0.687,  0.624,  0.564,
      0.509,  0.457,  0.408,  0.362,  0.320,  0.280,  0.242,  0.208,
      0.175,  0.144,  0.116,  0.089,  0.064,  0.041,  0.020,  0.000,
]

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


# Pre-compute lookback times for all redshifts (avoid repeated calculation)
_LOOKBACK_TIMES = None
def get_lookback_times(redshifts):
    """Get lookback times, using cache for standard redshifts."""
    global _LOOKBACK_TIMES
    if _LOOKBACK_TIMES is None or len(_LOOKBACK_TIMES) != len(redshifts):
        _LOOKBACK_TIMES = np.array([lookback_time_gyr(z) for z in redshifts])
    return _LOOKBACK_TIMES


def setup_style():
    """Configure matplotlib for publication-quality plots."""
    style_path = "./plotting/ciaran_ohare_palatino_sty.mplstyle"
    if os.path.exists(style_path):
        plt.style.use(style_path)
    else:
        plt.rcParams["figure.figsize"] = (8.34, 6.25)
        plt.rcParams["figure.dpi"] = 150
        plt.rcParams["font.size"] = 14


def select_bcgs(f, snap=63, min_halo_mass=MIN_HALO_MASS, n_max=N_BCG_MAX):
    """
    Select BCGs based on halo mass at z=0.
    
    BCGs are defined as the central galaxy (Type=0) in massive halos.
    
    Returns:
        bcg_gal_idx: GalaxyIndex values for selected BCGs (for main progenitor tracking)
        selected: Array indices at z=0
    """
    snap_key = f'Snap_{snap}'
    
    mvir = f[snap_key]['Mvir'][:] * MASS_CONVERT
    galtype = f[snap_key]['Type'][:]
    gal_idx = f[snap_key]['GalaxyIndex'][:]
    mstar = f[snap_key]['StellarMass'][:] * MASS_CONVERT
    
    # Select central galaxies in massive halos
    centrals = np.where((galtype == 0) & (mvir >= min_halo_mass))[0]
    
    if len(centrals) == 0:
        print(f"No BCGs found with Mvir >= {min_halo_mass:.1e} Msun")
        return np.array([]), np.array([])
    
    # Sort by halo mass and take top N
    sorted_idx = centrals[np.argsort(mvir[centrals])[::-1]]
    selected = sorted_idx[:min(n_max, len(sorted_idx))]
    
    print(f"Selected {len(selected)} BCGs with Mvir >= {min_halo_mass:.1e} Msun")
    print(f"  Halo mass range: {mvir[selected].min():.2e} - {mvir[selected].max():.2e} Msun")
    print(f"  Stellar mass range: {mstar[selected].min():.2e} - {mstar[selected].max():.2e} Msun")
    
    return gal_idx[selected], selected


def compute_histories(f, bcg_gal_indices, bcg_indices_z0, snaps=None):
    """
    Compute assembly and formation histories for each BCG.
    
    Assembly history: Mstar(z) / Mstar(z=0) for the main progenitor only
    Formation history: Cumulative SFH from z=0 galaxy - when stars were born
    
    Uses the z=0 SFHMassDisk and SFHMassBulge arrays which contain the complete
    star formation history including all merged progenitors.
    
    Key difference:
    - Formation tracks when stars were BORN (anywhere - main progenitor or satellites)
    - Assembly tracks when mass arrived in the MAIN PROGENITOR
    
    Formation should be AHEAD of assembly at early times because stars form in
    satellites before they merge into the main progenitor.
    
    Returns:
        redshifts: Array of redshifts
        assembly: 2D array (n_bcg, n_snaps) of assembly fractions
        formation: 2D array (n_bcg, n_snaps) of formation fractions
    """
    if snaps is None:
        snaps = list(range(1, 64))  # Skip snap 0 (very early, often no data)
    
    n_bcg = len(bcg_gal_indices)
    n_snaps = len(snaps)
    sorted_snaps = sorted(snaps)
    
    # Get final stellar masses and SFH at z=0
    sorted_order = np.argsort(bcg_indices_z0)
    sorted_indices = bcg_indices_z0[sorted_order]
    
    mstar_z0_sorted = f['Snap_63']['StellarMass'][sorted_indices] * MASS_CONVERT
    mstar_z0 = np.empty_like(mstar_z0_sorted)
    mstar_z0[sorted_order] = mstar_z0_sorted
    
    # Avoid division by zero
    mstar_z0_safe = np.maximum(mstar_z0, 1e-10)
    
    # Get z=0 SFH arrays - these contain complete history including merged satellites
    sfh_disk_z0_sorted = f['Snap_63']['SFHMassDisk'][sorted_indices] * MASS_CONVERT
    sfh_bulge_z0_sorted = f['Snap_63']['SFHMassBulge'][sorted_indices] * MASS_CONVERT
    
    # Restore original order
    sfh_disk_z0 = np.empty_like(sfh_disk_z0_sorted)
    sfh_bulge_z0 = np.empty_like(sfh_bulge_z0_sorted)
    sfh_disk_z0[sorted_order] = sfh_disk_z0_sorted
    sfh_bulge_z0[sorted_order] = sfh_bulge_z0_sorted
    
    # Compute formation history from z=0 SFH (cumulative sum over snapshots)
    # This tells us when the stars that end up in the z=0 BCG were born
    sfh_total_z0 = sfh_disk_z0 + sfh_bulge_z0  # (n_bcg, n_snap_bins)
    cumulative_sfh = np.cumsum(sfh_total_z0, axis=1)  # Cumulative over snapshot bins
    
    # Initialize output arrays
    assembly = np.full((n_bcg, n_snaps), np.nan)
    formation = np.full((n_bcg, n_snaps), np.nan)
    
    # Formation: use z=0 cumulative SFH
    for si, snap in enumerate(sorted_snaps):
        # Formation fraction at this snapshot (from z=0 SFH)
        formation[:, si] = cumulative_sfh[:, snap] / mstar_z0_safe
    
    # Convert BCG galaxy indices for assembly lookup
    bcg_gal_to_idx = {gid: i for i, gid in enumerate(bcg_gal_indices)}
    
    print("Computing assembly history across snapshots...")
    for si, snap in enumerate(sorted_snaps):
        snap_key = f'Snap_{snap}'
        if snap_key not in f:
            continue
        
        # Load galaxy indices for this snapshot
        gal_indices = f[snap_key]['GalaxyIndex'][:]
        
        # Find which BCGs exist in this snapshot
        bcg_mask = np.isin(gal_indices, bcg_gal_indices)
        if not np.any(bcg_mask):
            continue
        
        # Get the array indices and galaxy IDs of BCGs present
        bcg_arr_indices = np.where(bcg_mask)[0]
        bcg_gal_ids = gal_indices[bcg_arr_indices]
        
        # Map back to BCG output indices
        bcg_out_indices = np.array([bcg_gal_to_idx[gid] for gid in bcg_gal_ids])
        
        # Load stellar mass for assembly
        mstar_snap = f[snap_key]['StellarMass'][bcg_arr_indices] * MASS_CONVERT
        assembly[bcg_out_indices, si] = mstar_snap / mstar_z0_safe[bcg_out_indices]
    
    # Convert snapshot indices to redshifts
    redshifts_out = np.array([REDSHIFTS[s] for s in sorted_snaps])
    
    return redshifts_out, assembly, formation


def plot_formation_assembly(redshifts, assembly, formation, output_path=None):
    """
    Plot the formation and assembly histories.
    
    Blue solid line: median assembly history
    Green solid line: median formation history
    Shaded regions: 15th to 85th percentiles
    """
    setup_style()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Convert redshifts to lookback time (cached)
    lookback_times = get_lookback_times(redshifts)
    
    # Compute percentiles
    assembly_median = np.nanmedian(assembly, axis=0)
    assembly_15 = np.nanpercentile(assembly, 15, axis=0)
    assembly_85 = np.nanpercentile(assembly, 85, axis=0)
    
    formation_median = np.nanmedian(formation, axis=0)
    formation_15 = np.nanpercentile(formation, 15, axis=0)
    formation_85 = np.nanpercentile(formation, 85, axis=0)
    
    # Colors
    blue = '#2166AC'
    green = '#1B7837'
    
    # Plot assembly history (blue)
    ax.fill_between(lookback_times, assembly_15, assembly_85, 
                    color=blue, alpha=0.3, edgecolor='none')
    ax.plot(lookback_times, assembly_median, '-', color=blue, lw=2.5, 
            label='Assembly')
    
    # Plot formation history (green)
    ax.fill_between(lookback_times, formation_15, formation_85, 
                    color=green, alpha=0.3, edgecolor='none')
    ax.plot(lookback_times, formation_median, '-', color=green, lw=2.5, 
            label='Formation')
    
    # Formatting
    ax.set_xlabel('Lookback time [Gyr]')
    ax.set_ylabel(r'Fraction of $z=0$ stellar mass')
    ax.set_xlim(0, 13)  # High lookback time (early) on left, present on right
    ax.set_ylim(0, 1.05)
    
    # Legend
    ax.legend(loc='lower left', frameon=False, fontsize=12)
    
    # Grid
    # ax.grid(True, alpha=0.3, linestyle='--')
    
    fig.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    
    plt.close()
    
    return fig, ax


def main():
    """Main function to generate the BCG formation/assembly plot."""
    # Open HDF5 file
    filepath = os.path.join(PRIMARY_DIR, MODEL_FILE)
    print(f"Loading data from {filepath}")
    
    f = h5.File(filepath, 'r')
    
    # Select BCGs at z=0
    bcg_gal_indices, bcg_indices_z0 = select_bcgs(f)
    
    if len(bcg_gal_indices) == 0:
        print("No BCGs found. Exiting.")
        f.close()
        return
    
    # Compute formation and assembly histories
    print("\nComputing formation and assembly histories...")
    redshifts, assembly, formation = compute_histories(
        f, bcg_gal_indices, bcg_indices_z0
    )
    
    # Print some statistics
    print("\nStatistics at key redshifts:")
    z_targets = [5.0, 1.0, 0.5]
    for z_target in z_targets:
        # Find closest snapshot to target redshift
        idx = np.argmin(np.abs(redshifts - z_target))
        z_actual = redshifts[idx]
        
        form_med = np.nanmedian(formation[:, idx])
        form_15 = np.nanpercentile(formation[:, idx], 15)
        form_85 = np.nanpercentile(formation[:, idx], 85)
        
        assem_med = np.nanmedian(assembly[:, idx])
        assem_15 = np.nanpercentile(assembly[:, idx], 15)
        assem_85 = np.nanpercentile(assembly[:, idx], 85)
        
        print(f"  z={z_actual:.2f}:")
        print(f"    Formation: {form_med:.0%} ({form_15:.0%} - {form_85:.0%})")
        print(f"    Assembly:  {assem_med:.0%} ({assem_15:.0%} - {assem_85:.0%})")
    
    # Plot
    output_path = os.path.join(OUTPUT_DIR, 'BCG_formation_assembly' + OUTPUT_FORMAT)
    plot_formation_assembly(redshifts, assembly, formation, output_path)
    
    f.close()
    print("\nDone!")


if __name__ == '__main__':
    main()

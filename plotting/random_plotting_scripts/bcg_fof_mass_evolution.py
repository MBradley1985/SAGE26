#!/usr/bin/env python
"""
BCG/FOF Mass Evolution Plot
===========================
Evolution of the dark matter mass of the FOF group containing the main 
progenitor of the BCG at each time (blue), of the stellar content of 
this FOF group (green), and of the mass of the main progenitor of the 
BCG (orange).

Thick lines show the median and shaded regions show the 15th to 85th 
percentile range.

Note: Stellar mass of FOF group is multiplied by 50, and stellar mass
of main progenitor is multiplied by 100 for visibility.

Author: GitHub Copilot
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
MIN_HALO_MASS = 1.0e14  # Solar masses - select clusters
N_BCG_MAX = 10000        # Maximum number of BCGs to process

# Scaling factors for visibility
SCALE_FOF_STELLAR = 50
SCALE_BCG_STELLAR = 100

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


# Pre-compute times
LOOKBACK_TIMES = np.array([lookback_time_gyr(z) for z in REDSHIFTS])
T_NOW = cosmic_time_gyr(0.0)


def setup_style():
    """Configure matplotlib for publication-quality plots."""
    style_path = "./plotting/ciaran_ohare_palatino_sty.mplstyle"
    if os.path.exists(style_path):
        plt.style.use(style_path)
    else:
        plt.rcParams["figure.figsize"] = (10, 7)
        plt.rcParams["figure.dpi"] = 150
        plt.rcParams["font.size"] = 14


def select_bcgs(f, snap=63, min_halo_mass=MIN_HALO_MASS, n_max=N_BCG_MAX):
    """
    Select BCGs based on halo mass at z=0.
    
    Returns:
        bcg_gal_idx: GalaxyIndex values for selected BCGs
        selected: Array indices at z=0
        central_gal_idx: CentralGalaxyIndex for FOF group identification
    """
    snap_key = f'Snap_{snap}'
    
    mvir = f[snap_key]['Mvir'][:] * MASS_CONVERT
    gal_type = f[snap_key]['Type'][:]
    gal_indices = f[snap_key]['GalaxyIndex'][:]
    central_gal_idx = f[snap_key]['CentralGalaxyIndex'][:]
    
    # BCGs are Type=0 (centrals) in massive halos
    mask = (gal_type == 0) & (mvir >= min_halo_mass)
    selected = np.where(mask)[0]
    
    if len(selected) > n_max:
        np.random.seed(42)
        selected = np.random.choice(selected, n_max, replace=False)
        selected = np.sort(selected)  # h5py requires sorted indices
    
    return gal_indices[selected], selected, central_gal_idx[selected]


def trace_bcg_and_fof(f, bcg_gal_idx, start_snap=63):
    """
    Trace the main progenitor of the BCG and its FOF group back through time.
    
    At each snapshot, returns:
    - Mvir of the FOF group (dark matter mass)
    - Total stellar mass of all galaxies in the FOF group
    - Stellar mass of the main progenitor
    
    Returns dict: snap -> (Mvir, M*_FOF, M*_BCG)
    """
    history = {}
    current_gal_idx = bcg_gal_idx
    
    for snap in range(start_snap, -1, -1):
        snap_key = f'Snap_{snap}'
        if snap_key not in f:
            continue
        
        gal_indices = f[snap_key]['GalaxyIndex'][:]
        
        # Find the main progenitor in this snapshot
        matches = np.where(gal_indices == current_gal_idx)[0]
        if len(matches) == 0:
            break
        
        arr_idx = matches[0]
        
        # Get properties of the main progenitor
        mvir = f[snap_key]['Mvir'][arr_idx] * MASS_CONVERT
        mstar_bcg = f[snap_key]['StellarMass'][arr_idx] * MASS_CONVERT
        gal_type = f[snap_key]['Type'][arr_idx]
        
        # Get the CentralGalaxyIndex to identify the FOF group
        central_gal_idx = f[snap_key]['CentralGalaxyIndex'][arr_idx]
        
        # Find all galaxies in the same FOF group
        all_central_idx = f[snap_key]['CentralGalaxyIndex'][:]
        fof_mask = all_central_idx == central_gal_idx
        
        # Total stellar mass of the FOF group
        all_mstar = f[snap_key]['StellarMass'][:] * MASS_CONVERT
        mstar_fof = np.sum(all_mstar[fof_mask])
        
        # Get Mvir from the central galaxy of the FOF
        if gal_type == 0:
            # This is the central - use its Mvir
            mvir_fof = mvir
        else:
            # This is a satellite - find the central
            central_match = np.where((gal_indices == central_gal_idx) & 
                                    (f[snap_key]['Type'][:] == 0))[0]
            if len(central_match) > 0:
                mvir_fof = f[snap_key]['Mvir'][central_match[0]] * MASS_CONVERT
            else:
                mvir_fof = mvir  # Fallback
        
        history[snap] = (mvir_fof, mstar_fof, mstar_bcg)
    
    return history


def main():
    """Main function to create BCG/FOF mass evolution plot."""
    setup_style()
    
    model_path = os.path.join(PRIMARY_DIR, MODEL_FILE)
    print("BCG/FOF Mass Evolution Analysis")
    print("=" * 60)
    print(f"  Model file: {model_path}")
    print(f"  Min halo mass: {MIN_HALO_MASS:.1e} Msun")
    print(f"  FOF stellar mass scaled by: {SCALE_FOF_STELLAR}x")
    print(f"  BCG stellar mass scaled by: {SCALE_BCG_STELLAR}x")
    print()
    
    with h5.File(model_path, 'r') as f:
        # Select BCGs at z=0
        bcg_gal_idx, selected, _ = select_bcgs(f)
        print(f"Selected {len(selected)} BCGs at z=0")
        
        # Get z=0 properties for reference
        snap_key = 'Snap_63'
        mvir_z0 = f[snap_key]['Mvir'][selected] * MASS_CONVERT
        mstar_z0 = f[snap_key]['StellarMass'][selected] * MASS_CONVERT
        
        print(f"  Halo mass range: {np.min(mvir_z0):.2e} - {np.max(mvir_z0):.2e} Msun")
        print(f"  BCG mass range: {np.min(mstar_z0):.2e} - {np.max(mstar_z0):.2e} Msun")
        print()
        
        # Storage for all histories
        n_snaps = 64
        all_mvir = np.full((len(selected), n_snaps), np.nan)
        all_mstar_fof = np.full((len(selected), n_snaps), np.nan)
        all_mstar_bcg = np.full((len(selected), n_snaps), np.nan)
        
        # Track current GalaxyIndex for each BCG
        current_gal_idx = bcg_gal_idx.copy()
        
        print("Tracing BCG progenitors and FOF groups...")
        
        # Process snapshot by snapshot (more efficient)
        for snap in range(63, -1, -1):
            snap_key = f'Snap_{snap}'
            if snap_key not in f:
                continue
            
            if snap % 10 == 0:
                print(f"  Processing snapshot {snap} (z={REDSHIFTS[snap]:.2f})...")
            
            # Load all data for this snapshot once
            gal_indices = f[snap_key]['GalaxyIndex'][:]
            central_gal_idx_arr = f[snap_key]['CentralGalaxyIndex'][:]
            mvir_arr = f[snap_key]['Mvir'][:] * MASS_CONVERT
            mstar_arr = f[snap_key]['StellarMass'][:] * MASS_CONVERT
            gal_type_arr = f[snap_key]['Type'][:]
            
            # For each BCG, find its progenitor in this snapshot
            for i, gal_idx in enumerate(current_gal_idx):
                if gal_idx < 0:  # Already lost track
                    continue
                
                matches = np.where(gal_indices == gal_idx)[0]
                if len(matches) == 0:
                    current_gal_idx[i] = -1  # Mark as lost
                    continue
                
                arr_idx = matches[0]
                
                # Get BCG progenitor properties
                mstar_bcg = mstar_arr[arr_idx]
                all_mstar_bcg[i, snap] = mstar_bcg
                
                # Get the FOF group (via CentralGalaxyIndex)
                central_idx = central_gal_idx_arr[arr_idx]
                fof_mask = central_gal_idx_arr == central_idx
                
                # Total stellar mass of FOF group
                all_mstar_fof[i, snap] = np.sum(mstar_arr[fof_mask])
                
                # Get Mvir from the central of this FOF
                if gal_type_arr[arr_idx] == 0:
                    # This progenitor is the central
                    all_mvir[i, snap] = mvir_arr[arr_idx]
                else:
                    # Find the central
                    central_match = np.where((gal_indices == central_idx) & 
                                           (gal_type_arr == 0))[0]
                    if len(central_match) > 0:
                        all_mvir[i, snap] = mvir_arr[central_match[0]]
                    else:
                        # Fallback: use CentralMvir if available
                        if 'CentralMvir' in f[snap_key]:
                            all_mvir[i, snap] = f[snap_key]['CentralMvir'][arr_idx] * MASS_CONVERT
        
        print()
        
        # Calculate statistics at each snapshot
        median_mvir = np.nanmedian(all_mvir, axis=0)
        p16_mvir = np.nanpercentile(all_mvir, 16, axis=0)
        p84_mvir = np.nanpercentile(all_mvir, 84, axis=0)
        
        median_mstar_fof = np.nanmedian(all_mstar_fof, axis=0)
        p16_mstar_fof = np.nanpercentile(all_mstar_fof, 16, axis=0)
        p84_mstar_fof = np.nanpercentile(all_mstar_fof, 84, axis=0)
        
        median_mstar_bcg = np.nanmedian(all_mstar_bcg, axis=0)
        p16_mstar_bcg = np.nanpercentile(all_mstar_bcg, 16, axis=0)
        p84_mstar_bcg = np.nanpercentile(all_mstar_bcg, 84, axis=0)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Valid snapshots (where we have data)
        valid = np.isfinite(median_mvir) & (median_mvir > 0)
        snaps = np.arange(n_snaps)
        lookback = LOOKBACK_TIMES[snaps]
        
        # Plot FOF DM mass (blue)
        ax.semilogy(lookback[valid], median_mvir[valid], 
                   'C0-', linewidth=2.5, label=r'$M_{\rm vir}$ (FOF dark matter)')
        ax.fill_between(lookback[valid], p16_mvir[valid], p84_mvir[valid],
                       color='C0', alpha=0.2)
        
        # Plot FOF stellar mass x50 (green)
        ax.semilogy(lookback[valid], median_mstar_fof[valid] * SCALE_FOF_STELLAR, 
                   'C2-', linewidth=2.5, 
                   label=rf'$M_*$ (FOF stellar) $\times$ {SCALE_FOF_STELLAR}')
        ax.fill_between(lookback[valid], 
                       p16_mstar_fof[valid] * SCALE_FOF_STELLAR, 
                       p84_mstar_fof[valid] * SCALE_FOF_STELLAR,
                       color='C2', alpha=0.2)
        
        # Plot BCG stellar mass x100 (orange)
        ax.semilogy(lookback[valid], median_mstar_bcg[valid] * SCALE_BCG_STELLAR, 
                   'C1-', linewidth=2.5, 
                   label=rf'$M_*$ (BCG main progenitor) $\times$ {SCALE_BCG_STELLAR}')
        ax.fill_between(lookback[valid], 
                       p16_mstar_bcg[valid] * SCALE_BCG_STELLAR, 
                       p84_mstar_bcg[valid] * SCALE_BCG_STELLAR,
                       color='C1', alpha=0.2)
        
        ax.set_xlabel('Lookback Time [Gyr]')
        ax.set_ylabel(r'Mass [M$_\odot$]')
        ax.set_title('Mass Evolution of BCGs and their Host FOF Groups')
        
        ax.set_xlim(0, 13)
        ax.set_ylim(1e12, 1e15)
        
        ax.legend(loc='lower left', fontsize=10)
        
        # Add redshift axis on top
        ax2 = ax.twiny()
        z_ticks = [0, 0.5, 1, 2, 3, 5, 10]
        z_tick_pos = [lookback_time_gyr(z) for z in z_ticks]
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(z_tick_pos)
        ax2.set_xticklabels([f'{z}' for z in z_ticks])
        ax2.set_xlabel('Redshift')
        
        plt.tight_layout()
        
        # Save
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, f'BCG_FOF_mass_evolution{OUTPUT_FORMAT}')
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
        
        # Print statistics
        print()
        print("=" * 60)
        print("Statistics at key redshifts:")
        print("=" * 60)
        
        for z_target in [0.0, 0.5, 1.0, 2.0]:
            snap = np.argmin(np.abs(REDSHIFTS - z_target))
            if np.isfinite(median_mvir[snap]):
                print(f"\nz = {REDSHIFTS[snap]:.2f}:")
                print(f"  M_vir (FOF):    {median_mvir[snap]:.2e} Msun "
                      f"({p16_mvir[snap]:.2e} - {p84_mvir[snap]:.2e})")
                print(f"  M* (FOF):       {median_mstar_fof[snap]:.2e} Msun "
                      f"({p16_mstar_fof[snap]:.2e} - {p84_mstar_fof[snap]:.2e})")
                print(f"  M* (BCG):       {median_mstar_bcg[snap]:.2e} Msun "
                      f"({p16_mstar_bcg[snap]:.2e} - {p84_mstar_bcg[snap]:.2e})")
                print(f"  BCG/FOF ratio:  {median_mstar_bcg[snap]/median_mstar_fof[snap]*100:.1f}%")
        
        print("\n\nDone!")


if __name__ == '__main__':
    main()

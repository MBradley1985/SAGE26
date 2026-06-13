#!/usr/bin/env python
"""
Rejuvenation SFH Plot
=====================
Plots SFH vs lookback time for quenched galaxies, highlighting those that
have experienced rejuvenation (renewed star formation after quenching).

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

# Galaxy selection
MIN_STELLAR_MASS = 1.0e10  # Solar masses - select massive galaxies
MAX_GALAXIES = 5000  # Max galaxies to analyze

# Quenching definition - use SFR=0 (truly stopped forming stars)
SFR_QUENCH_THRESHOLD = 1e-6  # Msun/yr - below this is "quenched" (effectively zero)
SFR_REJUV_THRESHOLD = 0.1    # Msun/yr - above this after quenching = "rejuvenated"
MIN_QUENCH_DURATION = 0.5    # Gyr - minimum time spent quenched before rejuvenation counts
MAX_QUENCH_LOOKBACK = 10.0   # Gyr - only count quenching events more recent than this

# Plot settings
N_QUENCHED_TO_PLOT = 5
N_REJUVENATED_TO_PLOT = 5  # Total = 10 examples

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
        plt.rcParams["figure.figsize"] = (10, 7)
        plt.rcParams["figure.dpi"] = 150
        plt.rcParams["font.size"] = 14


def trace_main_progenitor(f, galaxy_index, start_snap=63):
    """
    Trace main progenitor of a galaxy back through time.
    
    Returns dict with snap -> (array_index, GalaxyIndex) for main progenitor chain.
    """
    progenitor_chain = {}
    current_gal_idx = galaxy_index
    
    for snap in range(start_snap, -1, -1):
        snap_key = f'Snap_{snap}'
        if snap_key not in f:
            continue
            
        gal_indices = f[snap_key]['GalaxyIndex'][:]
        
        # Find this galaxy in current snapshot
        matches = np.where(gal_indices == current_gal_idx)[0]
        if len(matches) == 0:
            break
            
        arr_idx = matches[0]
        progenitor_chain[snap] = (arr_idx, current_gal_idx)
        
        # For next iteration, keep same GalaxyIndex (main progenitor tracking)
        # GalaxyIndex encodes the main progenitor lineage
        
    return progenitor_chain


def get_sfh_for_galaxy(f, progenitor_chain):
    """
    Extract full SFH for a galaxy given its progenitor chain.
    
    Returns:
        times: Array of lookback times (Gyr)
        sfrs: Array of total SFR values (Msun/yr)
        stellar_masses: Array of stellar mass at each snapshot
    """
    times = []
    sfrs = []
    stellar_masses = []
    
    sorted_snaps = sorted(progenitor_chain.keys())
    
    for snap in sorted_snaps:
        arr_idx, _ = progenitor_chain[snap]
        snap_key = f'Snap_{snap}'
        
        # Get stellar mass
        mstar = f[snap_key]['StellarMass'][arr_idx] * MASS_CONVERT
        stellar_masses.append(mstar)
        
        # Get SFH for this snapshot
        if 'SfrDiskSTEPS' in f[snap_key]:
            sfr_disk = f[snap_key]['SfrDiskSTEPS'][arr_idx]
            sfr_bulge = f[snap_key]['SfrBulgeSTEPS'][arr_idx]
            sfr_total = sfr_disk + sfr_bulge
            
            # Get timestep width (in Myr)
            dt = f[snap_key]['dT'][arr_idx]  # Myr
            dt_per_step = dt / STEPS
            
            # Calculate time bins for this snapshot
            # End time is at this snapshot's redshift
            t_end = LOOKBACK_TIMES[snap]
            
            # Each step goes backwards in time
            for i in range(STEPS):
                t = t_end + (STEPS - 1 - i) * dt_per_step / 1000.0  # Convert Myr to Gyr
                times.append(t)
                sfrs.append(sfr_total[i])
        else:
            # No SFH data, use instantaneous SFR
            sfr = f[snap_key]['SfrDisk'][arr_idx] + f[snap_key]['SfrBulge'][arr_idx]
            times.append(LOOKBACK_TIMES[snap])
            sfrs.append(sfr)
    
    return np.array(times), np.array(sfrs), np.array(stellar_masses)


def classify_sfh(times, sfrs, stellar_masses):
    """
    Classify a galaxy's SFH as quenched, star-forming, or rejuvenated.
    
    Uses SFR=0 (or effectively zero) as the quenching criterion, not sSFR.
    
    Rejuvenation requires:
    - A quenching event (SFR -> 0) at lookback time < MAX_QUENCH_LOOKBACK
    - Sustained quenching for at least MIN_QUENCH_DURATION
    - Subsequently rising above SFR_REJUV_THRESHOLD
    
    Returns:
        classification: 'star-forming', 'quenched', or 'rejuvenated'
        quench_time: Lookback time of quenching (if applicable)
        rejuv_time: Lookback time of rejuvenation (if applicable)
    """
    if len(sfrs) < 10 or len(stellar_masses) < 2:
        return 'unknown', None, None
    
    # Sort by time (oldest first, i.e., highest lookback time first)
    sort_idx = np.argsort(times)[::-1]
    times_sorted = times[sort_idx]
    sfrs_sorted = sfrs[sort_idx]
    
    # Require some star formation history
    if np.max(sfrs_sorted) < SFR_QUENCH_THRESHOLD:
        return 'always-quenched', None, None
    
    # Look for quenching events at lookback < MAX_QUENCH_LOOKBACK
    valid_times_mask = times_sorted < MAX_QUENCH_LOOKBACK
    
    # Track quenching state - need sustained quench (SFR=0)
    quench_start_time = None
    rejuv_time = None
    final_quench_time = None
    
    # Walk through time (from old to young, i.e., decreasing lookback time)
    for i in range(len(sfrs_sorted)):
        if not valid_times_mask[i]:
            continue
            
        if sfrs_sorted[i] < SFR_QUENCH_THRESHOLD:
            # Galaxy has stopped forming stars
            if quench_start_time is None:
                quench_start_time = times_sorted[i]
        else:
            if quench_start_time is not None:
                # Check if we were quenched long enough
                quench_duration = quench_start_time - times_sorted[i]
                if quench_duration >= MIN_QUENCH_DURATION:
                    # This is a valid quench event
                    final_quench_time = quench_start_time
                    
                    # Check if this is rejuvenation (SFR came back)
                    if sfrs_sorted[i] > SFR_REJUV_THRESHOLD:
                        rejuv_time = times_sorted[i]
                        return 'rejuvenated', final_quench_time, rejuv_time
                        
            quench_start_time = None
    
    # Check if currently quenched
    if quench_start_time is not None and quench_start_time < MAX_QUENCH_LOOKBACK:
        quench_duration = quench_start_time  # Duration to z=0
        if quench_duration >= MIN_QUENCH_DURATION:
            return 'quenched', quench_start_time, None
    
    if final_quench_time is not None:
        return 'quenched', final_quench_time, None
    
    return 'star-forming', None, None


def main():
    """Main function to identify and plot rejuvenated galaxies."""
    setup_style()
    
    model_path = os.path.join(PRIMARY_DIR, MODEL_FILE)
    print("Rejuvenation SFH Analysis")
    print("=" * 50)
    print(f"  Model file: {model_path}")
    print(f"  Min stellar mass: {MIN_STELLAR_MASS:.1e} Msun")
    print(f"  SFR quench threshold: {SFR_QUENCH_THRESHOLD:.1e} Msun/yr (i.e., SFR = 0)")
    print(f"  SFR rejuvenation threshold: {SFR_REJUV_THRESHOLD:.1e} Msun/yr")
    print()
    
    with h5.File(model_path, 'r') as f:
        # Select massive galaxies at z=0
        snap_key = 'Snap_63'
        mstar = f[snap_key]['StellarMass'][:] * MASS_CONVERT
        gal_type = f[snap_key]['Type'][:]
        gal_indices = f[snap_key]['GalaxyIndex'][:]
        
        # Select centrals above mass threshold
        mask = (mstar >= MIN_STELLAR_MASS) & (gal_type == 0)
        selected_indices = np.where(mask)[0]
        
        # Limit sample size
        if len(selected_indices) > MAX_GALAXIES:
            np.random.seed(42)
            selected_indices = np.random.choice(selected_indices, MAX_GALAXIES, replace=False)
        
        print(f"Selected {len(selected_indices)} massive centrals at z=0")
        print()
        
        # Classify all galaxies
        results = {
            'star-forming': [],
            'quenched': [],
            'rejuvenated': [],
            'always-quenched': [],
            'unknown': []
        }
        
        print("Analyzing SFHs...")
        for i, arr_idx in enumerate(selected_indices):
            if (i + 1) % 500 == 0:
                print(f"  Processed {i + 1}/{len(selected_indices)}")
            
            gal_idx = gal_indices[arr_idx]
            
            # Trace progenitor
            chain = trace_main_progenitor(f, gal_idx)
            
            if len(chain) < 20:  # Need reasonable history
                continue
            
            # Get SFH
            times, sfrs, mstars = get_sfh_for_galaxy(f, chain)
            
            # Classify
            classification, quench_t, rejuv_t = classify_sfh(times, sfrs, mstars)
            
            results[classification].append({
                'gal_idx': gal_idx,
                'arr_idx': arr_idx,
                'times': times,
                'sfrs': sfrs,
                'mstars': mstars,
                'quench_time': quench_t,
                'rejuv_time': rejuv_t,
                'mstar_final': mstar[arr_idx]
            })
        
        print()
        print("Classification results:")
        for cat, gals in results.items():
            print(f"  {cat}: {len(gals)}")
        print()
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Helper function for plotting SFH
        def plot_sfh_simple(times, sfrs, color, alpha, linewidth):
            """Plot SFH with given style."""
            # Sort by time (oldest first)
            sort_idx = np.argsort(times)[::-1]
            times_sorted = times[sort_idx]
            sfrs_sorted = sfrs[sort_idx]
            
            # Smooth for visualization
            if len(sfrs_sorted) > 20:
                kernel_size = 5
                sfrs_smooth = np.convolve(sfrs_sorted, np.ones(kernel_size)/kernel_size, mode='valid')
                times_smooth = times_sorted[kernel_size//2:-(kernel_size//2)]
            else:
                sfrs_smooth = sfrs_sorted
                times_smooth = times_sorted
            
            # Add small offset to avoid log(0)
            sfrs_plot = np.maximum(sfrs_smooth, 1e-4)
            
            ax.semilogy(times_smooth, sfrs_plot, color=color, alpha=alpha, 
                       linewidth=linewidth)
            
            return times_smooth, sfrs_plot
        
        # Plot quenched galaxies (gray lines)
        quenched = results['quenched']
        n_plot_q = min(N_QUENCHED_TO_PLOT, len(quenched))
        
        if n_plot_q > 0:
            quenched_sorted = sorted(quenched, key=lambda x: x['mstar_final'], reverse=True)
            for i, gal in enumerate(quenched_sorted[:n_plot_q]):
                plot_sfh_simple(gal['times'], gal['sfrs'], color='gray', alpha=0.7, linewidth=1.5)
        
        # Plot rejuvenated galaxies - gray before rejuvenation, blue after
        rejuvenated = results['rejuvenated']
        n_plot_r = min(N_REJUVENATED_TO_PLOT, len(rejuvenated))
        
        if n_plot_r > 0:
            rejuvenated_sorted = sorted(rejuvenated, key=lambda x: x['mstar_final'], reverse=True)
            
            for i, gal in enumerate(rejuvenated_sorted[:n_plot_r]):
                times = gal['times']
                sfrs = gal['sfrs']
                rejuv_time = gal['rejuv_time']
                
                # Split into before and after rejuvenation
                # Before rejuvenation (higher lookback time) = gray
                before_mask = times >= rejuv_time
                after_mask = times < rejuv_time
                
                if np.sum(before_mask) > 5:
                    plot_sfh_simple(times[before_mask], sfrs[before_mask], 
                                   color='gray', alpha=0.7, linewidth=1.5)
                
                # After rejuvenation (lower lookback time) = thick blue
                if np.sum(after_mask) > 5:
                    plot_sfh_simple(times[after_mask], sfrs[after_mask], 
                                   color='steelblue', alpha=0.9, linewidth=3.0)
        
        ax.set_xlabel('Lookback Time [Gyr]')
        ax.set_ylabel('Star Formation Rate [M$_\\odot$/yr]')
        ax.set_title('Star Formation Histories: Quenched vs Rejuvenated Galaxies')
        
        ax.set_xlim(0, 13)
        ax.set_ylim(1e-2, 5e2)
        
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
        output_path = os.path.join(OUTPUT_DIR, f'sfh_rejuvenation{OUTPUT_FORMAT}')
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved: {output_path}")
        
        # Print details of rejuvenated galaxies
        if len(rejuvenated) > 0:
            print()
            print("=" * 50)
            print("Rejuvenated Galaxy Details:")
            print("=" * 50)
            for gal in rejuvenated_sorted[:n_plot_r]:
                print(f"  GalaxyIndex: {gal['gal_idx']}")
                print(f"    M* (z=0): {gal['mstar_final']:.2e} Msun")
                if gal['quench_time'] is not None:
                    quench_z = REDSHIFTS[np.argmin(np.abs(LOOKBACK_TIMES - gal['quench_time']))]
                    print(f"    Quench: {gal['quench_time']:.1f} Gyr lookback (z~{quench_z:.1f})")
                if gal['rejuv_time'] is not None:
                    rejuv_z = REDSHIFTS[np.argmin(np.abs(LOOKBACK_TIMES - gal['rejuv_time']))]
                    print(f"    Rejuvenation: {gal['rejuv_time']:.1f} Gyr lookback (z~{rejuv_z:.1f})")
                print()


if __name__ == '__main__':
    main()

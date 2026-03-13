#!/usr/bin/env python3
"""
Compare model BHMF against TRINITY observations to diagnose discrepancies.
"""
import numpy as np
import h5py
import glob
import os

def read_hdf(file_list, snap, field):
    """Read a field from multiple HDF5 files."""
    data = []
    for f in file_list:
        with h5py.File(f, 'r') as hf:
            snap_key = f'Snap_{snap}'
            if snap_key in hf:
                data.append(hf[snap_key][field][:])
    return np.concatenate(data) if data else np.array([])

def main():
    # Configuration
    output_dir = '../output/millennium/'
    data_dir = '../data/'
    Hubble_h = 0.73
    BoxSize = 62.5

    # Find all model files and calculate volume
    file_list = sorted(glob.glob(os.path.join(output_dir, 'model_*.hdf5')))
    if not file_list:
        print(f"No HDF5 files found in {output_dir}")
        return

    # Read volume fraction
    total_vol_frac = 0.0
    with h5py.File(file_list[0], 'r') as f:
        Hubble_h = float(f['Header/Simulation'].attrs['hubble_h'])
        BoxSize = float(f['Header/Simulation'].attrs['box_size'])
    for f in file_list:
        with h5py.File(f, 'r') as hf:
            total_vol_frac += float(hf['Header/Runtime'].attrs['frac_volume_processed'])

    volume = (BoxSize / Hubble_h)**3.0 * total_vol_frac
    print(f"Volume: {volume:.2f} Mpc^3 (VolFrac={total_vol_frac:.3f})")

    # Read z=0 BH masses (snapshot 63)
    Snapshot = 63
    BlackHoleMass = read_hdf(file_list, Snapshot, 'BlackHoleMass') * 1.0e10 / Hubble_h

    print(f"\nTotal galaxies: {len(BlackHoleMass)}")
    bh_mask = BlackHoleMass > 0
    print(f"Galaxies with BH > 0: {np.sum(bh_mask)}")

    # Calculate model BHMF
    bh_bins = np.arange(4.0, 11.5, 0.2)
    bin_width = 0.2
    bh_centers = bh_bins[:-1] + bin_width/2

    log_bh = np.log10(BlackHoleMass[bh_mask])
    counts, _ = np.histogram(log_bh, bins=bh_bins)
    model_phi = counts / (volume * bin_width)

    # Load TRINITY z=0.1 observations
    obs_file = os.path.join(data_dir, 'fig4_bhmf_z0.1.txt')
    obs_data = np.loadtxt(obs_file)
    obs_mass = obs_data[:, 0]
    obs_phi = obs_data[:, 1]
    obs_phi_16 = obs_data[:, 2]
    obs_phi_84 = obs_data[:, 3]

    print("\n" + "="*70)
    print("BHMF COMPARISON: MODEL vs TRINITY (z~0)")
    print("="*70)
    print(f"{'log M_BH':>10} {'Model φ':>14} {'Obs φ':>14} {'Ratio':>10} {'Status':>12}")
    print("-"*70)

    # Compare at key mass bins
    for mass_target in [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
        # Find closest model bin
        model_idx = np.argmin(np.abs(bh_centers - mass_target))
        model_val = model_phi[model_idx] if model_phi[model_idx] > 0 else 1e-10

        # Find closest obs value
        obs_idx = np.argmin(np.abs(obs_mass - mass_target))
        obs_val = obs_phi[obs_idx]

        ratio = model_val / obs_val if obs_val > 0 else 0

        if ratio < 0.5:
            status = "TOO FEW"
        elif ratio > 2.0:
            status = "TOO MANY"
        else:
            status = "OK"

        print(f"{mass_target:>10.1f} {model_val:>14.2e} {obs_val:>14.2e} {ratio:>10.2f} {status:>12}")

    print("-"*70)

    # Detailed breakdown
    print("\n" + "="*70)
    print("DETAILED MODEL BHMF")
    print("="*70)
    print(f"{'log M_BH':>10} {'N_gal':>10} {'φ [Mpc^-3 dex^-1]':>20}")
    print("-"*50)
    for i, center in enumerate(bh_centers):
        if counts[i] > 0:
            print(f"{center:>10.1f} {counts[i]:>10d} {model_phi[i]:>20.4e}")

    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    # Where is the model deficient?
    print("\nKey issues:")

    # Low mass BHs (< 10^6)
    low_mass_model = np.sum(model_phi[bh_centers < 6.0])
    low_mass_obs_idx = obs_mass < 6.0
    low_mass_obs = np.mean(obs_phi[low_mass_obs_idx]) if np.any(low_mass_obs_idx) else 0
    print(f"  Low-mass BHs (M < 10^6): Model has {low_mass_model/low_mass_obs:.1%} of observed")

    # Intermediate mass (10^6 - 10^8)
    mid_mask_model = (bh_centers >= 6.0) & (bh_centers < 8.0)
    mid_mask_obs = (obs_mass >= 6.0) & (obs_mass < 8.0)
    mid_model = np.mean(model_phi[mid_mask_model]) if np.any(mid_mask_model) else 0
    mid_obs = np.mean(obs_phi[mid_mask_obs]) if np.any(mid_mask_obs) else 0
    print(f"  Mid-mass BHs (10^6 < M < 10^8): Model has {mid_model/mid_obs:.1%} of observed")

    # High mass (> 10^8)
    high_mask_model = bh_centers >= 8.0
    high_mask_obs = obs_mass >= 8.0
    high_model = np.mean(model_phi[high_mask_model][model_phi[high_mask_model] > 0]) if np.any(model_phi[high_mask_model] > 0) else 0
    high_obs = np.mean(obs_phi[high_mask_obs]) if np.any(high_mask_obs) else 0
    print(f"  High-mass BHs (M > 10^8): Model has {high_model/high_obs:.1%} of observed")

if __name__ == '__main__':
    main()

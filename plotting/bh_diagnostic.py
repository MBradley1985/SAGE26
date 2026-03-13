#!/usr/bin/env python3
"""
Quick diagnostic to analyze black hole properties from SAGE output.
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
    Hubble_h = 0.73
    Snapshot = 63  # z=0

    # Find all model files
    file_list = sorted(glob.glob(os.path.join(output_dir, 'model_*.hdf5')))
    if not file_list:
        print(f"No HDF5 files found in {output_dir}")
        return

    print(f"Found {len(file_list)} output files")

    # Read relevant data
    print("\nReading data...")
    BlackHoleMass = read_hdf(file_list, Snapshot, 'BlackHoleMass') * 1.0e10 / Hubble_h
    BulgeMass = read_hdf(file_list, Snapshot, 'BulgeMass') * 1.0e10 / Hubble_h
    StellarMass = read_hdf(file_list, Snapshot, 'StellarMass') * 1.0e10 / Hubble_h
    InstabilityBulgeMass = read_hdf(file_list, Snapshot, 'InstabilityBulgeMass') * 1.0e10 / Hubble_h
    MergerBulgeMass = read_hdf(file_list, Snapshot, 'MergerBulgeMass') * 1.0e10 / Hubble_h
    QuasarModeBHaccretionMass = read_hdf(file_list, Snapshot, 'QuasarModeBHaccretionMass') * 1.0e10 / Hubble_h

    print(f"Total galaxies: {len(BlackHoleMass)}")

    # Basic BH statistics
    print("\n=== BLACK HOLE STATISTICS ===")
    bh_mask = BlackHoleMass > 0
    print(f"Galaxies with BH > 0: {np.sum(bh_mask)} ({100*np.sum(bh_mask)/len(BlackHoleMass):.1f}%)")

    if np.sum(bh_mask) > 0:
        print(f"BH mass range: {np.log10(BlackHoleMass[bh_mask].min()):.2f} - {np.log10(BlackHoleMass[bh_mask].max()):.2f} log(M_sun)")
        print(f"Median BH mass: {np.log10(np.median(BlackHoleMass[bh_mask])):.2f} log(M_sun)")
        print(f"Total BH mass: {BlackHoleMass.sum():.2e} M_sun")

    # Bulge statistics - instability vs merger origin
    print("\n=== BULGE ORIGIN STATISTICS ===")
    bulge_mask = BulgeMass > 0
    print(f"Galaxies with bulge > 0: {np.sum(bulge_mask)} ({100*np.sum(bulge_mask)/len(BulgeMass):.1f}%)")

    total_bulge = BulgeMass.sum()
    total_instability_bulge = InstabilityBulgeMass.sum()
    total_merger_bulge = MergerBulgeMass.sum()

    print(f"Total bulge mass: {total_bulge:.2e} M_sun")
    print(f"  From instabilities: {total_instability_bulge:.2e} M_sun ({100*total_instability_bulge/total_bulge:.1f}%)")
    print(f"  From mergers: {total_merger_bulge:.2e} M_sun ({100*total_merger_bulge/total_bulge:.1f}%)")

    # How much of bulge is from instabilities in individual galaxies?
    instability_dominated = (InstabilityBulgeMass > MergerBulgeMass) & bulge_mask
    print(f"\nInstability-dominated bulges: {np.sum(instability_dominated)} ({100*np.sum(instability_dominated)/np.sum(bulge_mask):.1f}% of bulge galaxies)")

    # QuasarMode BH accretion (this includes both merger and instability-driven)
    print("\n=== QUASAR MODE BH ACCRETION ===")
    qm_mask = QuasarModeBHaccretionMass > 0
    print(f"Total QuasarModeBHaccretionMass: {QuasarModeBHaccretionMass.sum():.2e} M_sun")
    print(f"Galaxies with QuasarMode accretion: {np.sum(qm_mask)}")

    # BH mass function
    print("\n=== BLACK HOLE MASS FUNCTION ===")
    bh_bins = np.arange(5, 11, 0.5)  # 10^5 to 10^10.5 M_sun
    counts, _ = np.histogram(np.log10(BlackHoleMass[bh_mask]), bins=bh_bins)

    print("log(M_BH)    N_galaxies")
    for i, (low, high) in enumerate(zip(bh_bins[:-1], bh_bins[1:])):
        print(f"  {low:.1f}-{high:.1f}     {counts[i]}")

    # BH-Bulge relation check
    print("\n=== BH-BULGE MASS RELATION ===")
    good_mask = (BulgeMass > 1e8) & (BlackHoleMass > 1e5)
    if np.sum(good_mask) > 0:
        log_bh = np.log10(BlackHoleMass[good_mask])
        log_bulge = np.log10(BulgeMass[good_mask])

        # Fit a line
        slope, intercept = np.polyfit(log_bulge, log_bh, 1)
        print(f"Fitted relation: log(M_BH) = {slope:.2f} * log(M_bulge) + {intercept:.2f}")
        print(f"(Observations: slope ~ 1.12, normalization gives M_BH ~ 0.001 * M_bulge)")

        # Check normalization at M_bulge = 10^11
        predicted_bh_at_11 = slope * 11 + intercept
        print(f"At M_bulge = 10^11: M_BH = 10^{predicted_bh_at_11:.2f} (expect ~10^8)")

        # Ratio distribution
        ratio = BlackHoleMass[good_mask] / BulgeMass[good_mask]
        print(f"M_BH/M_bulge: median = {np.median(ratio):.4f}, mean = {np.mean(ratio):.4f}")
        print(f"(Observations: ~0.001-0.002)")

    # Check if disk instabilities are contributing
    print("\n=== DISK INSTABILITY ACTIVITY ===")
    # Galaxies where instability contributed to bulge
    inst_active = InstabilityBulgeMass > 1e6
    print(f"Galaxies with significant instability bulge (>10^6 M_sun): {np.sum(inst_active)}")

    # For these, how does their BH compare?
    if np.sum(inst_active) > 0:
        print(f"  Their median BH mass: {np.median(BlackHoleMass[inst_active]):.2e} M_sun")
        print(f"  Their median total bulge: {np.median(BulgeMass[inst_active]):.2e} M_sun")
        print(f"  Their median instability bulge: {np.median(InstabilityBulgeMass[inst_active]):.2e} M_sun")

if __name__ == '__main__':
    main()

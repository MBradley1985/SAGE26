#!/usr/bin/env python
"""Quick diagnostic to check halo mass and regime data at z=0"""

import h5py as h5
import numpy as np

# Update these to match your 500 Mpc box settings
DirName = './output/millennium/'
FileName = 'model_0.hdf5'
Hubble_h = 0.73

snap = 63  # z=0

print(f"Reading from: {DirName}{FileName}")
print(f"Snapshot: {snap} (z=0)\n")

with h5.File(DirName + FileName, 'r') as f:
    # Check what's in the file
    print("Available datasets in Snap_63:")
    print(list(f[f'Snap_{snap}'].keys()))
    print()

    # Read the data
    Mvir = np.array(f[f'Snap_{snap}']['Mvir']) * 1.0e10 / Hubble_h
    StellarMass = np.array(f[f'Snap_{snap}']['StellarMass']) * 1.0e10 / Hubble_h
    Regime = np.array(f[f'Snap_{snap}']['Regime'])

    print(f"Total galaxies: {len(Mvir)}")
    print(f"Galaxies with Mvir > 0: {np.sum(Mvir > 0)}")
    print(f"Galaxies with StellarMass > 0: {np.sum(StellarMass > 0)}")
    print()

    # Check halo mass range
    w_valid = (Mvir > 0) & (StellarMass > 0)
    if np.sum(w_valid) > 0:
        log_mvir = np.log10(Mvir[w_valid])
        print(f"Halo mass range (log10 Msun):")
        print(f"  Min: {log_mvir.min():.2f}")
        print(f"  Max: {log_mvir.max():.2f}")
        print(f"  Median: {np.median(log_mvir):.2f}")
        print()

        # Count by mass bins
        print("Halo counts by mass bin:")
        for threshold in [11, 12, 13, 14]:
            count = np.sum(log_mvir > threshold)
            print(f"  log10(Mvir) > {threshold}: {count}")
        print()

        # Check regime values
        regime_valid = Regime[w_valid]
        print(f"Regime value statistics:")
        print(f"  Unique values: {np.unique(regime_valid)}")
        print(f"  Regime == 0 (CGM): {np.sum(regime_valid == 0)}")
        print(f"  Regime == 1 (Hot): {np.sum(regime_valid == 1)}")
        print(f"  Other values: {np.sum((regime_valid != 0) & (regime_valid != 1))}")
        print()

        # Check regime for massive haloes specifically
        massive = log_mvir > 13
        if np.sum(massive) > 0:
            regime_massive = regime_valid[massive]
            print(f"For haloes with log10(Mvir) > 13:")
            print(f"  Total count: {np.sum(massive)}")
            print(f"  Regime == 0 (CGM): {np.sum(regime_massive == 0)}")
            print(f"  Regime == 1 (Hot): {np.sum(regime_massive == 1)}")
            print(f"  Unique regime values: {np.unique(regime_massive)}")
        else:
            print("No haloes with log10(Mvir) > 13 found!")
    else:
        print("No valid galaxies found with both Mvir > 0 and StellarMass > 0!")

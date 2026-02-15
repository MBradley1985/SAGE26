#!/usr/bin/env python3
"""Quick check of stellar mass function to verify DarkMode preserves calibration."""

import h5py
import numpy as np
import matplotlib.pyplot as plt

# Load z=0 data
with h5py.File('output/millennium/model_0.hdf5', 'r') as f:
    stellar_mass = f['Snap_63/StellarMass'][:]
    h = 0.73  # Hubble parameter
    
# Convert to log10(M*/Msun) - remove h dependence
stellar_mass_msun = stellar_mass * 1e10 / h
valid = stellar_mass_msun > 1e8  # Minimum mass
log_mass = np.log10(stellar_mass_msun[valid])

# Create SMF
mass_bins = np.arange(8.5, 12.5, 0.2)
counts, edges = np.histogram(log_mass, bins=mass_bins)

# Millennium volume
box_size = 62.5 / h  # Mpc
volume = box_size**3

# Number density per dex
bin_width = edges[1] - edges[0]
phi = counts / (volume * bin_width)
mass_centers = 0.5 * (edges[:-1] + edges[1:])

# Plot
plt.figure(figsize=(8, 6))
plt.semilogy(mass_centers, phi, 'o-', linewidth=2, markersize=6, label='DarkMode (z=0)')
plt.xlabel(r'log$_{10}$(M$_*$ / M$_{\odot}$)', fontsize=14)
plt.ylabel(r'$\Phi$ [Mpc$^{-3}$ dex$^{-1}$]', fontsize=14)
plt.title('Stellar Mass Function - DarkMode Test', fontsize=14)
plt.ylim(1e-6, 1e-1)
plt.xlim(8.5, 12)
plt.grid(alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('plotting/darkmode_smf_check.png', dpi=150)
print(f"\nSaved: plotting/darkmode_smf_check.png")
print(f"Total galaxies M* > 10^8 Msun: {len(log_mass)}")
print(f"Peak at log10(M*/Msun) = {mass_centers[np.argmax(phi)]:.1f}")
print(f"Number density at peak: {phi.max():.2e} Mpc^-3 dex^-1")

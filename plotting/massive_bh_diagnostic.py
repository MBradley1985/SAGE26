#!/usr/bin/env python3
"""
Diagnose what's limiting growth of the most massive BHs (>10^9 M_sun).
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
            if snap_key in hf and field in hf[snap_key]:
                data.append(hf[snap_key][field][:])
    return np.concatenate(data) if data else np.array([])

def main():
    output_dir = '../output/millennium/'
    Hubble_h = 0.73

    file_list = sorted(glob.glob(os.path.join(output_dir, 'model_*.hdf5')))
    if not file_list:
        print(f"No HDF5 files found")
        return

    # Read z=0 data
    Snapshot = 63
    BlackHoleMass = read_hdf(file_list, Snapshot, 'BlackHoleMass') * 1.0e10 / Hubble_h
    BulgeMass = read_hdf(file_list, Snapshot, 'BulgeMass') * 1.0e10 / Hubble_h
    StellarMass = read_hdf(file_list, Snapshot, 'StellarMass') * 1.0e10 / Hubble_h
    Mvir = read_hdf(file_list, Snapshot, 'Mvir') * 1.0e10 / Hubble_h
    HotGas = read_hdf(file_list, Snapshot, 'HotGas') * 1.0e10 / Hubble_h
    ColdGas = read_hdf(file_list, Snapshot, 'ColdGas') * 1.0e10 / Hubble_h
    Type = read_hdf(file_list, Snapshot, 'Type')
    QuasarModeBHaccretionMass = read_hdf(file_list, Snapshot, 'QuasarModeBHaccretionMass') * 1.0e10 / Hubble_h

    print("="*70)
    print("ANALYSIS OF MOST MASSIVE BLACK HOLES")
    print("="*70)

    # Find galaxies with massive BHs
    massive_bh_mask = BlackHoleMass > 1e8  # > 10^8 M_sun
    very_massive_bh_mask = BlackHoleMass > 1e9  # > 10^9 M_sun

    print(f"\nGalaxies with M_BH > 10^8 M_sun: {np.sum(massive_bh_mask)}")
    print(f"Galaxies with M_BH > 10^9 M_sun: {np.sum(very_massive_bh_mask)}")

    # Properties of massive BH hosts
    if np.sum(massive_bh_mask) > 0:
        print("\n--- Properties of M_BH > 10^8 hosts ---")
        print(f"Median halo mass: {np.median(Mvir[massive_bh_mask]):.2e} M_sun")
        print(f"Max halo mass: {np.max(Mvir[massive_bh_mask]):.2e} M_sun")
        print(f"Median stellar mass: {np.median(StellarMass[massive_bh_mask]):.2e} M_sun")
        print(f"Median bulge mass: {np.median(BulgeMass[massive_bh_mask]):.2e} M_sun")
        print(f"Median hot gas: {np.median(HotGas[massive_bh_mask]):.2e} M_sun")
        print(f"Median cold gas: {np.median(ColdGas[massive_bh_mask]):.2e} M_sun")
        print(f"Fraction that are centrals: {np.sum(Type[massive_bh_mask] == 0) / np.sum(massive_bh_mask):.1%}")

    # What are the most massive halos doing?
    print("\n" + "="*70)
    print("MOST MASSIVE HALOS AND THEIR BHS")
    print("="*70)

    # Central galaxies only
    central_mask = Type == 0

    # Sort by halo mass
    halo_order = np.argsort(Mvir[central_mask])[::-1]

    print(f"\nTop 20 most massive halos (centrals only):")
    print(f"{'Rank':>4} {'log Mvir':>10} {'log M_BH':>10} {'log M_*':>10} {'log M_bulge':>10} {'log HotGas':>10}")
    print("-"*70)

    central_indices = np.where(central_mask)[0]
    for i, idx in enumerate(halo_order[:20]):
        gal_idx = central_indices[idx]
        log_mvir = np.log10(Mvir[gal_idx]) if Mvir[gal_idx] > 0 else -99
        log_bh = np.log10(BlackHoleMass[gal_idx]) if BlackHoleMass[gal_idx] > 0 else -99
        log_stellar = np.log10(StellarMass[gal_idx]) if StellarMass[gal_idx] > 0 else -99
        log_bulge = np.log10(BulgeMass[gal_idx]) if BulgeMass[gal_idx] > 0 else -99
        log_hot = np.log10(HotGas[gal_idx]) if HotGas[gal_idx] > 0 else -99
        print(f"{i+1:>4} {log_mvir:>10.2f} {log_bh:>10.2f} {log_stellar:>10.2f} {log_bulge:>10.2f} {log_hot:>10.2f}")

    # Halo mass distribution
    print("\n" + "="*70)
    print("HALO MASS FUNCTION (centrals)")
    print("="*70)

    halo_bins = np.arange(11, 15.5, 0.5)
    log_mvir_centrals = np.log10(Mvir[central_mask])

    print(f"\n{'log Mvir':>12} {'N_halos':>10}")
    print("-"*30)
    for i in range(len(halo_bins)-1):
        mask = (log_mvir_centrals >= halo_bins[i]) & (log_mvir_centrals < halo_bins[i+1])
        print(f"{halo_bins[i]:>6.1f}-{halo_bins[i+1]:<5.1f} {np.sum(mask):>10}")

    # BH mass vs halo mass relation
    print("\n" + "="*70)
    print("BH MASS vs HALO MASS (centrals)")
    print("="*70)

    print(f"\n{'log Mvir':>12} {'Median log M_BH':>18} {'Max log M_BH':>15} {'N_gal':>8}")
    print("-"*60)
    for i in range(len(halo_bins)-1):
        mask = (log_mvir_centrals >= halo_bins[i]) & (log_mvir_centrals < halo_bins[i+1])
        if np.sum(mask) > 0:
            bh_masses = BlackHoleMass[central_mask][mask]
            bh_masses = bh_masses[bh_masses > 0]
            if len(bh_masses) > 0:
                median_bh = np.log10(np.median(bh_masses))
                max_bh = np.log10(np.max(bh_masses))
                print(f"{halo_bins[i]:>6.1f}-{halo_bins[i+1]:<5.1f} {median_bh:>18.2f} {max_bh:>15.2f} {np.sum(mask):>8}")

    # What's the theoretical max BH mass based on halo mass?
    print("\n" + "="*70)
    print("THEORETICAL vs ACTUAL BH MASSES")
    print("="*70)

    # M_BH - M_bulge relation: M_BH ~ 0.001 * M_bulge
    # M_bulge ~ 0.01-0.1 * M_* for massive galaxies
    # M_* ~ 0.01-0.02 * M_vir at the massive end
    # So M_BH ~ 0.001 * 0.05 * 0.015 * M_vir ~ 7.5e-7 * M_vir

    print("\nExpected M_BH based on scaling relations:")
    print("(Assuming M_BH ~ 0.001 * M_bulge, M_bulge ~ 0.3 * M_*, M_* ~ 0.02 * Mvir)")
    print(f"\n{'log Mvir':>10} {'Expected log M_BH':>20}")
    print("-"*35)
    for log_mvir in [12, 13, 14, 15]:
        expected_bh = 10**log_mvir * 0.02 * 0.3 * 0.001
        print(f"{log_mvir:>10} {np.log10(expected_bh):>20.2f}")

    print("\n" + "="*70)
    print("LIMITING FACTORS FOR BH GROWTH")
    print("="*70)

    # Check Eddington ratio for massive BHs
    # Radio mode rate ~ M_BH * efficiency * (hot gas properties)
    # Eddington rate ~ 2.2e-8 * M_BH per year (in M_sun/yr)

    if np.sum(massive_bh_mask) > 0:
        print("\nFor M_BH > 10^8 hosts:")

        # Hot gas availability
        hot_gas_ratio = HotGas[massive_bh_mask] / Mvir[massive_bh_mask]
        print(f"  Hot gas / Mvir: median = {np.median(hot_gas_ratio):.4f}")

        # Cold gas availability for quasar mode
        cold_gas_mass = ColdGas[massive_bh_mask]
        print(f"  Cold gas: median = {np.median(cold_gas_mass):.2e} M_sun")

        # Quasar mode contribution
        qm_frac = QuasarModeBHaccretionMass[massive_bh_mask] / BlackHoleMass[massive_bh_mask]
        print(f"  QuasarMode fraction of BH mass: median = {np.median(qm_frac):.4f}")

if __name__ == '__main__':
    main()

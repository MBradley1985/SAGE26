#!/usr/bin/env python3
"""
Investigate the dip in the BHMF around 10^7 M_sun.
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

def analyze_bh_population(file_list, Snapshot, Hubble_h, label):
    """Analyze BH population at a given snapshot."""

    BlackHoleMass = read_hdf(file_list, Snapshot, 'BlackHoleMass') * 1.0e10 / Hubble_h
    BulgeMass = read_hdf(file_list, Snapshot, 'BulgeMass') * 1.0e10 / Hubble_h
    StellarMass = read_hdf(file_list, Snapshot, 'StellarMass') * 1.0e10 / Hubble_h
    Mvir = read_hdf(file_list, Snapshot, 'Mvir') * 1.0e10 / Hubble_h
    HotGas = read_hdf(file_list, Snapshot, 'HotGas') * 1.0e10 / Hubble_h
    ColdGas = read_hdf(file_list, Snapshot, 'ColdGas') * 1.0e10 / Hubble_h
    Type = read_hdf(file_list, Snapshot, 'Type')
    QuasarModeBHaccretionMass = read_hdf(file_list, Snapshot, 'QuasarModeBHaccretionMass') * 1.0e10 / Hubble_h
    InstabilityBulgeMass = read_hdf(file_list, Snapshot, 'InstabilityBulgeMass') * 1.0e10 / Hubble_h
    MergerBulgeMass = read_hdf(file_list, Snapshot, 'MergerBulgeMass') * 1.0e10 / Hubble_h

    # SFR
    SfrDisk = read_hdf(file_list, Snapshot, 'SfrDisk')
    SfrBulge = read_hdf(file_list, Snapshot, 'SfrBulge')
    SFR = SfrDisk + SfrBulge

    bh_mask = BlackHoleMass > 0

    print(f"\n{'='*70}")
    print(f"BHMF DIP ANALYSIS: {label}")
    print(f"{'='*70}")

    # Define mass bins around the dip
    mass_ranges = [
        (5.5, 6.0, "Below dip (seed-dominated)"),
        (6.0, 6.5, "Transition zone 1"),
        (6.5, 7.0, "DIP region"),
        (7.0, 7.5, "Transition zone 2"),
        (7.5, 8.0, "Above dip (growth-dominated)"),
        (8.0, 8.5, "Massive BHs"),
    ]

    log_bh = np.log10(BlackHoleMass[bh_mask])

    print(f"\n{'Mass Range':>20} {'N_gal':>8} {'Central%':>10} {'Merger%':>10} {'Instab%':>10} {'QM_frac':>10}")
    print("-"*78)

    for low, high, desc in mass_ranges:
        mask = (log_bh >= low) & (log_bh < high)
        indices = np.where(bh_mask)[0][mask]
        n_gal = len(indices)

        if n_gal > 0:
            # Central fraction
            central_frac = np.sum(Type[indices] == 0) / n_gal

            # Merger vs instability bulge dominance
            has_bulge = BulgeMass[indices] > 0
            if np.sum(has_bulge) > 0:
                merger_dom = np.sum(MergerBulgeMass[indices][has_bulge] > InstabilityBulgeMass[indices][has_bulge]) / np.sum(has_bulge)
                instab_dom = np.sum(InstabilityBulgeMass[indices][has_bulge] > MergerBulgeMass[indices][has_bulge]) / np.sum(has_bulge)
            else:
                merger_dom = 0
                instab_dom = 0

            # Quasar mode fraction
            qm_frac = np.median(QuasarModeBHaccretionMass[indices] / BlackHoleMass[indices])

            print(f"{low:.1f}-{high:.1f} ({desc[:15]:>15}) {n_gal:>8} {central_frac:>10.1%} {merger_dom:>10.1%} {instab_dom:>10.1%} {qm_frac:>10.4f}")

    # Detailed analysis of the dip region
    print(f"\n{'='*70}")
    print("DETAILED DIP REGION ANALYSIS (6.5 < log M_BH < 7.0)")
    print(f"{'='*70}")

    dip_mask = (log_bh >= 6.5) & (log_bh < 7.0)
    dip_indices = np.where(bh_mask)[0][dip_mask]

    above_mask = (log_bh >= 7.5) & (log_bh < 8.0)
    above_indices = np.where(bh_mask)[0][above_mask]

    if len(dip_indices) > 0 and len(above_indices) > 0:
        print(f"\n{'Property':>25} {'Dip (6.5-7.0)':>18} {'Above (7.5-8.0)':>18}")
        print("-"*65)

        # Halo mass
        print(f"{'Median log Mvir':>25} {np.median(np.log10(Mvir[dip_indices])):>18.2f} {np.median(np.log10(Mvir[above_indices])):>18.2f}")

        # Stellar mass
        print(f"{'Median log M_*':>25} {np.median(np.log10(StellarMass[dip_indices])):>18.2f} {np.median(np.log10(StellarMass[above_indices])):>18.2f}")

        # Bulge mass
        bulge_dip = BulgeMass[dip_indices]
        bulge_above = BulgeMass[above_indices]
        print(f"{'Median log M_bulge':>25} {np.median(np.log10(bulge_dip[bulge_dip>0])):>18.2f} {np.median(np.log10(bulge_above[bulge_above>0])):>18.2f}")

        # Hot gas
        hot_dip = HotGas[dip_indices]
        hot_above = HotGas[above_indices]
        print(f"{'Median log HotGas':>25} {np.median(np.log10(hot_dip[hot_dip>0])):>18.2f} {np.median(np.log10(hot_above[hot_above>0])):>18.2f}")

        # Cold gas
        cold_dip = ColdGas[dip_indices]
        cold_above = ColdGas[above_indices]
        if np.sum(cold_dip > 0) > 0 and np.sum(cold_above > 0) > 0:
            print(f"{'Median log ColdGas':>25} {np.median(np.log10(cold_dip[cold_dip>0])):>18.2f} {np.median(np.log10(cold_above[cold_above>0])):>18.2f}")

        # SFR
        sfr_dip = SFR[dip_indices]
        sfr_above = SFR[above_indices]
        print(f"{'Median SFR':>25} {np.median(sfr_dip):>18.2f} {np.median(sfr_above):>18.2f}")

        # Central fraction
        print(f"{'Central fraction':>25} {np.sum(Type[dip_indices]==0)/len(dip_indices):>18.1%} {np.sum(Type[above_indices]==0)/len(above_indices):>18.1%}")

        # Quasar mode growth
        qm_dip = QuasarModeBHaccretionMass[dip_indices]
        qm_above = QuasarModeBHaccretionMass[above_indices]
        print(f"{'QuasarMode > 0 fraction':>25} {np.sum(qm_dip>0)/len(dip_indices):>18.1%} {np.sum(qm_above>0)/len(above_indices):>18.1%}")

        # BH to bulge ratio
        ratio_dip = BlackHoleMass[dip_indices] / BulgeMass[dip_indices]
        ratio_above = BlackHoleMass[above_indices] / BulgeMass[above_indices]
        ratio_dip = ratio_dip[np.isfinite(ratio_dip)]
        ratio_above = ratio_above[np.isfinite(ratio_above)]
        if len(ratio_dip) > 0 and len(ratio_above) > 0:
            print(f"{'Median M_BH/M_bulge':>25} {np.median(ratio_dip):>18.4f} {np.median(ratio_above):>18.4f}")

    # Check if it's a satellite vs central effect
    print(f"\n{'='*70}")
    print("CENTRAL vs SATELLITE BHMF")
    print(f"{'='*70}")

    bh_bins = np.arange(5.0, 9.5, 0.5)

    print(f"\n{'log M_BH':>10} {'N_central':>12} {'N_satellite':>12} {'Sat_frac':>12}")
    print("-"*50)

    for i in range(len(bh_bins)-1):
        mask = (log_bh >= bh_bins[i]) & (log_bh < bh_bins[i+1])
        indices = np.where(bh_mask)[0][mask]

        n_central = np.sum(Type[indices] == 0)
        n_satellite = np.sum(Type[indices] == 1)
        n_total = n_central + n_satellite

        if n_total > 0:
            sat_frac = n_satellite / n_total
            print(f"{bh_bins[i]:.1f}-{bh_bins[i+1]:.1f} {n_central:>12} {n_satellite:>12} {sat_frac:>12.1%}")

def main():
    output_dir = '../output/millennium/'
    Hubble_h = 0.73

    file_list = sorted(glob.glob(os.path.join(output_dir, 'model_*.hdf5')))
    if not file_list:
        print(f"No HDF5 files found")
        return

    # Get redshifts
    with h5py.File(file_list[0], 'r') as f:
        redshifts = np.array(f['Header/snapshot_redshifts'])

    # Analyze at z=0
    analyze_bh_population(file_list, 63, Hubble_h, "z = 0")

    # Also check at z=1 to see if dip persists
    z1_snap = np.argmin(np.abs(redshifts - 1.0))
    analyze_bh_population(file_list, z1_snap, Hubble_h, f"z ~ 1 (snap {z1_snap})")

if __name__ == '__main__':
    main()

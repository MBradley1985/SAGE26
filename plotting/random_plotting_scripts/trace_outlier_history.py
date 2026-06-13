#!/usr/bin/env python3
"""
Trace outlier galaxies through their merger tree history.
Identifies when and why BCG mass became pathologically low relative to ICS.
"""

import numpy as np
import h5py
import glob
import sys
from pathlib import Path


def read_hdf_single_file(filepath, snapshot, field):
    """Read a single field from a single HDF5 file."""
    with h5py.File(filepath, 'r') as f:
        if snapshot in f and field in f[snapshot]:
            return f[snapshot][field][:]
    return np.array([])


def read_hdf_all_files(file_list, snapshot, fields):
    """Read multiple fields from all HDF5 files for a snapshot."""
    data = {field: [] for field in fields}
    for filepath in file_list:
        with h5py.File(filepath, 'r') as f:
            if snapshot not in f:
                continue
            for field in fields:
                if field in f[snapshot]:
                    data[field].append(f[snapshot][field][:])

    # Concatenate arrays
    for field in fields:
        if data[field]:
            data[field] = np.concatenate(data[field])
        else:
            data[field] = np.array([])
    return data


def get_simulation_params(file_list):
    """Read simulation parameters from first HDF5 file."""
    with h5py.File(file_list[0], 'r') as f:
        hubble_h = f['Header/Simulation'].attrs['hubble_h']

        # Get redshifts from header
        snapshot_redshifts = f['Header/snapshot_redshifts'][:]

        # Get available snapshots
        available_snaps = []
        redshifts = {}
        for key in f.keys():
            if key.startswith('Snap_'):
                snap_num = int(key.split('_')[1])
                available_snaps.append(snap_num)
                redshifts[snap_num] = snapshot_redshifts[snap_num]

        available_snaps = sorted(available_snaps)

    return hubble_h, redshifts, available_snaps


def find_galaxy_by_index(file_list, snapshot, galaxy_index):
    """Find a galaxy by its GalaxyIndex and return its properties."""
    snapshot_str = f'Snap_{snapshot}'

    fields = ['GalaxyIndex', 'SAGETreeIndex', 'SAGEHaloIndex', 'Type',
              'Mvir', 'StellarMass', 'BulgeMass', 'IntraClusterStars',
              'ColdGas', 'HotGas', 'CentralGalaxyIndex', 'MergerBulgeMass',
              'TimeOfLastMajorMerger', 'TimeOfLastMinorMerger']

    data = read_hdf_all_files(file_list, snapshot_str, fields)

    if len(data['GalaxyIndex']) == 0:
        return None

    idx = np.where(data['GalaxyIndex'] == galaxy_index)[0]
    if len(idx) == 0:
        return None

    idx = idx[0]
    result = {field: data[field][idx] for field in fields}
    return result


def find_galaxies_in_tree(file_list, snapshot, tree_index):
    """Find all galaxies in a given tree at a snapshot."""
    snapshot_str = f'Snap_{snapshot}'

    fields = ['GalaxyIndex', 'SAGETreeIndex', 'SAGEHaloIndex', 'Type',
              'Mvir', 'StellarMass', 'BulgeMass', 'IntraClusterStars',
              'ColdGas', 'HotGas', 'CentralGalaxyIndex', 'MergerBulgeMass',
              'TimeOfLastMajorMerger', 'TimeOfLastMinorMerger',
              'mergeIntoID', 'mergeIntoSnapNum', 'mergeType']

    data = read_hdf_all_files(file_list, snapshot_str, fields)

    if len(data['GalaxyIndex']) == 0:
        return []

    mask = data['SAGETreeIndex'] == tree_index
    indices = np.where(mask)[0]

    galaxies = []
    for idx in indices:
        gal = {field: data[field][idx] for field in fields}
        galaxies.append(gal)

    return galaxies


def trace_galaxy_history(file_list, target_snap, galaxy_index, hubble_h, redshifts, available_snaps):
    """
    Trace a galaxy's history back through time.
    """
    print(f"\n{'='*80}")
    print(f"Tracing galaxy history: Snap {target_snap}, GalaxyIndex {galaxy_index}")
    print(f"{'='*80}")

    # Find the target galaxy
    target_gal = find_galaxy_by_index(file_list, target_snap, galaxy_index)
    if target_gal is None:
        print(f"ERROR: Could not find galaxy with GalaxyIndex {galaxy_index} at Snap {target_snap}")
        return

    tree_index = target_gal['SAGETreeIndex']
    print(f"\nTarget galaxy found in SAGETreeIndex: {tree_index}")
    print(f"Target properties at z={redshifts[target_snap]:.2f}:")
    print(f"  Type: {target_gal['Type']}")
    print(f"  log Mvir: {np.log10(target_gal['Mvir'] * 1e10 / hubble_h):.2f}")
    print(f"  log M_BCG: {np.log10(target_gal['StellarMass'] * 1e10 / hubble_h):.2f}")
    print(f"  log M_ICS: {np.log10(target_gal['IntraClusterStars'] * 1e10 / hubble_h):.2f}")
    print(f"  M_ICS/M_BCG ratio: {target_gal['IntraClusterStars'] / target_gal['StellarMass']:.2f}")

    # Get snapshots to trace (from target back to earliest)
    snaps_to_trace = sorted([s for s in available_snaps if s <= target_snap], reverse=True)

    print(f"\n{'='*80}")
    print(f"Evolution history (following most massive progenitor in tree)")
    print(f"{'='*80}")
    print(f"{'Snap':>6} {'z':>6} {'Type':>6} {'log Mvir':>10} {'log M_BCG':>10} {'log M_ICS':>10} "
          f"{'Ratio':>8} {'N_tree':>7} {'Event':>20}")
    print(f"{'-'*6} {'-'*6} {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*7} {'-'*20}")

    prev_stellar = None
    prev_ics = None
    prev_mvir = None
    prev_central_idx = None

    for snap in snaps_to_trace:
        z = redshifts[snap]

        # Find all galaxies in this tree at this snapshot
        galaxies = find_galaxies_in_tree(file_list, snap, tree_index)

        if not galaxies:
            print(f"{snap:>6} {z:>6.2f}   -- No galaxies in tree --")
            continue

        n_in_tree = len(galaxies)

        # Find the central (Type 0) or most massive galaxy
        centrals = [g for g in galaxies if g['Type'] == 0]
        if centrals:
            # If multiple centrals (shouldn't happen), take most massive
            main_gal = max(centrals, key=lambda g: g['Mvir'])
        else:
            # No central - take most massive satellite
            main_gal = max(galaxies, key=lambda g: g['StellarMass'])

        stellar = main_gal['StellarMass'] * 1e10 / hubble_h
        ics = main_gal['IntraClusterStars'] * 1e10 / hubble_h
        mvir = main_gal['Mvir'] * 1e10 / hubble_h
        gal_type = main_gal['Type']
        central_idx = main_gal['CentralGalaxyIndex']

        # Detect events
        event = ""
        if prev_stellar is not None:
            stellar_change = stellar / prev_stellar if prev_stellar > 0 else 0
            ics_change = ics / prev_ics if prev_ics > 0 else 0
            mvir_change = mvir / prev_mvir if prev_mvir > 0 else 0

            if stellar_change < 0.5:
                event = "BCG mass dropped!"
            elif stellar_change > 2.0:
                event = "BCG mass jumped"
            elif ics_change > 2.0:
                event = "ICS jumped"
            elif prev_central_idx is not None and central_idx != prev_central_idx:
                event = "Central changed"

        # Handle log of zero
        log_stellar = np.log10(stellar) if stellar > 0 else -99
        log_ics = np.log10(ics) if ics > 0 else -99
        log_mvir = np.log10(mvir) if mvir > 0 else -99
        ratio = ics / stellar if stellar > 0 else 0

        print(f"{snap:>6} {z:>6.2f} {gal_type:>6} {log_mvir:>10.2f} {log_stellar:>10.2f} "
              f"{log_ics:>10.2f} {ratio:>8.2f} {n_in_tree:>7} {event:>20}")

        prev_stellar = stellar
        prev_ics = ics
        prev_mvir = mvir
        prev_central_idx = central_idx

    # Also show all galaxies in the tree at the target snapshot
    print(f"\n{'='*80}")
    print(f"All galaxies in tree {tree_index} at Snap {target_snap} (z={redshifts[target_snap]:.2f})")
    print(f"{'='*80}")

    galaxies = find_galaxies_in_tree(file_list, target_snap, tree_index)
    galaxies_sorted = sorted(galaxies, key=lambda g: g['StellarMass'], reverse=True)

    # Find target galaxy in the list
    target_in_list = None
    for i, gal in enumerate(galaxies_sorted):
        if gal['GalaxyIndex'] == galaxy_index:
            target_in_list = (i, gal)
            break

    print(f"{'Type':>6} {'log Mvir':>10} {'log M_*':>10} {'log M_ICS':>10} {'Ratio':>8} {'GalaxyIndex':>20}")
    print(f"{'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*20}")

    # Show top 10 by stellar mass
    for gal in galaxies_sorted[:10]:
        stellar = gal['StellarMass'] * 1e10 / hubble_h
        ics = gal['IntraClusterStars'] * 1e10 / hubble_h
        mvir = gal['Mvir'] * 1e10 / hubble_h

        log_stellar = np.log10(stellar) if stellar > 0 else -99
        log_ics = np.log10(ics) if ics > 0 else -99
        log_mvir = np.log10(mvir) if mvir > 0 else -99
        ratio = ics / stellar if stellar > 0 else 0

        marker = " <-- TARGET" if gal['GalaxyIndex'] == galaxy_index else ""
        print(f"{gal['Type']:>6} {log_mvir:>10.2f} {log_stellar:>10.2f} {log_ics:>10.2f} "
              f"{ratio:>8.2f} {gal['GalaxyIndex']:>20}{marker}")

    # If target not in top 10, show it separately
    if target_in_list and target_in_list[0] >= 10:
        print(f"  ... ({target_in_list[0] - 10} galaxies omitted) ...")
        gal = target_in_list[1]
        stellar = gal['StellarMass'] * 1e10 / hubble_h
        ics = gal['IntraClusterStars'] * 1e10 / hubble_h
        mvir = gal['Mvir'] * 1e10 / hubble_h
        log_stellar = np.log10(stellar) if stellar > 0 else -99
        log_ics = np.log10(ics) if ics > 0 else -99
        log_mvir = np.log10(mvir) if mvir > 0 else -99
        ratio = ics / stellar if stellar > 0 else 0
        print(f"{gal['Type']:>6} {log_mvir:>10.2f} {log_stellar:>10.2f} {log_ics:>10.2f} "
              f"{ratio:>8.2f} {gal['GalaxyIndex']:>20} <-- TARGET")

    # Count Type 0s (centrals) - this should normally be 1 per FOF
    n_centrals = sum(1 for g in galaxies if g['Type'] == 0)
    if n_centrals > 1:
        print(f"\n  WARNING: {n_centrals} Type 0 (central) galaxies in this tree!")
        print(f"  This suggests multiple FOF halos merged into this tree.")

    # Now trace the specific outlier galaxy's history (not just the most massive)
    print(f"\n{'='*80}")
    print(f"Tracing the SPECIFIC outlier galaxy (GalaxyIndex {galaxy_index})")
    print(f"{'='*80}")
    print(f"{'Snap':>6} {'z':>6} {'Type':>6} {'log Mvir':>10} {'log M_BCG':>10} {'log M_ICS':>10} "
          f"{'Ratio':>8} {'CentralIdx':>15}")
    print(f"{'-'*6} {'-'*6} {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*15}")

    # Find this specific galaxy at each snapshot
    snaps_to_trace = sorted([s for s in available_snaps if s <= target_snap], reverse=True)

    # We need to follow this galaxy by its SAGEHaloIndex within the tree
    current_halo_idx = target_gal['SAGEHaloIndex']

    for snap in snaps_to_trace:
        z = redshifts[snap]
        galaxies = find_galaxies_in_tree(file_list, snap, tree_index)

        if not galaxies:
            print(f"{snap:>6} {z:>6.2f}   -- No galaxies in tree --")
            continue

        # Find galaxy with matching SAGEHaloIndex or closest match
        match = None
        for gal in galaxies:
            if gal['SAGEHaloIndex'] == current_halo_idx:
                match = gal
                break

        if match is None:
            # Try to find by GalaxyIndex pattern (same local index)
            local_idx = galaxy_index % 1000000000000  # Extract local index
            for gal in galaxies:
                if gal['GalaxyIndex'] % 1000000000000 == local_idx:
                    match = gal
                    break

        if match is None:
            print(f"{snap:>6} {z:>6.2f}   -- Galaxy not found (halo may not exist yet) --")
            continue

        stellar = match['StellarMass'] * 1e10 / hubble_h
        ics = match['IntraClusterStars'] * 1e10 / hubble_h
        mvir = match['Mvir'] * 1e10 / hubble_h

        log_stellar = np.log10(stellar) if stellar > 0 else -99
        log_ics = np.log10(ics) if ics > 0 else -99
        log_mvir = np.log10(mvir) if mvir > 0 else -99
        ratio = ics / stellar if stellar > 0 else 0

        print(f"{snap:>6} {z:>6.2f} {match['Type']:>6} {log_mvir:>10.2f} {log_stellar:>10.2f} "
              f"{log_ics:>10.2f} {ratio:>8.2f} {match['CentralGalaxyIndex']:>15}")


def main():
    # Default outliers from the diagnostic output
    default_outliers = [
        (48, 5000113000000000, 30.08),  # Worst outlier
        (63, 3000060000000107, 8.88),
        (48, 1000097000000007, 7.96),
        (48, 36000000000, 5.27),
        (63, 5000177000000007, 3.97),
    ]

    # Find HDF5 files
    output_dir = Path('output/millennium')
    file_list = sorted(glob.glob(str(output_dir / 'model_*.hdf5')))

    if not file_list:
        print("ERROR: No HDF5 files found in output/millennium/")
        sys.exit(1)

    print(f"Found {len(file_list)} HDF5 files")

    # Get simulation parameters
    hubble_h, redshifts, available_snaps = get_simulation_params(file_list)
    print(f"Hubble_h: {hubble_h}")
    print(f"Available snapshots: {available_snaps}")

    # Parse command line arguments or use defaults
    if len(sys.argv) >= 3:
        # User specified snap and galaxy_index
        snap = int(sys.argv[1])
        galaxy_index = int(sys.argv[2])
        outliers_to_trace = [(snap, galaxy_index, 0)]
    else:
        # Trace default outliers
        print("\nNo arguments provided. Tracing top 5 outliers by default.")
        print("Usage: python trace_outlier_history.py <snap> <galaxy_index>")
        outliers_to_trace = default_outliers

    # Trace each outlier
    for snap, galaxy_index, ratio in outliers_to_trace:
        trace_galaxy_history(file_list, snap, galaxy_index, hubble_h, redshifts, available_snaps)
        print("\n")


if __name__ == '__main__':
    main()

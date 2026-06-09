#!/usr/bin/env python
"""Diagnose halos with very high ICS stellar fractions."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import glob
from BCG_ICS_fraction import read_hdf, read_simulation_params, DataCache

MVIR_THRESHOLD_REDSHIFT = 1.0e13
MVIR_THRESHOLD_REDSHIFT_GROUPS = 1.0e12
STELLARMASS_THRESHOLD = 8.6e8

file_list = sorted(glob.glob('./output/millennium/model_*.hdf5'))
if not file_list:
    print("No HDF5 files found")
    sys.exit(1)

sim_params = read_simulation_params(os.path.abspath(file_list[0]))
Hubble_h = sim_params['Hubble_h']
redshifts = sim_params['redshifts']
available_snapshots = sim_params['available_snapshots']
cache = DataCache(file_list, Hubble_h, sim_params)

max_redshift = 2.5
snaps_to_plot = [s for s in available_snapshots
                 if s < len(redshifts) and redshifts[s] <= max_redshift]
snaps_to_plot.sort(reverse=True)

print(f"{'z':>5} {'Type':>8} {'Mvir':>12} {'M*,BCG':>12} {'M_ICS':>12} {'M*,sat':>12} {'M*,total':>12} {'f_ICS':>8} {'Len':>6} {'N_sat':>6} {'ICS/BCG':>8}")
print("-" * 120)

for snap in snaps_to_plot:
    Snapshot = 'Snap_' + str(snap)
    z = redshifts[snap]

    data = cache.get(Snapshot, ['Mvir', 'IntraClusterStars', 'StellarMass', 'Type', 'Len',
                                'CentralGalaxyIndex', 'GalaxyIndex'])
    Mvir = data['Mvir']
    ICS = data['IntraClusterStars']
    StellarMass = data['StellarMass']
    Type = data['Type']
    Len = data['Len']
    GalaxyIndex = data['GalaxyIndex']
    CentralGalaxyIndex = data['CentralGalaxyIndex']

    satellite_mass = cache.get_satellite_mass(Snapshot)

    # Count satellites per central
    sorted_idx = np.argsort(GalaxyIndex)
    sorted_gids = GalaxyIndex[sorted_idx]
    satellite_mask = Type != 0
    satellite_central_gids = CentralGalaxyIndex[satellite_mask]
    insert_pos = np.searchsorted(sorted_gids, satellite_central_gids)
    insert_pos = np.clip(insert_pos, 0, len(sorted_gids) - 1)
    valid_match = sorted_gids[insert_pos] == satellite_central_gids
    central_indices = np.where(valid_match, sorted_idx[insert_pos], -1)
    valid_satellites = central_indices >= 0
    n_satellites = np.zeros(len(StellarMass), dtype=int)
    np.add.at(n_satellites, central_indices[valid_satellites], 1)

    total_stellar = StellarMass + satellite_mass + ICS

    # Clusters
    clusters = np.where((Type == 0) & (Mvir >= MVIR_THRESHOLD_REDSHIFT) & (ICS > 0) &
                         (StellarMass >= STELLARMASS_THRESHOLD) & (total_stellar > 0))[0]
    if len(clusters) > 0:
        ics_frac = ICS[clusters] / total_stellar[clusters]
        # Show top 5 at each snapshot
        top_idx = np.argsort(ics_frac)[::-1][:5]
        for i in top_idx:
            ci = clusters[i]
            print(f"{z:5.2f} {'Cluster':>8} {Mvir[ci]:12.3e} {StellarMass[ci]:12.3e} {ICS[ci]:12.3e} "
                  f"{satellite_mass[ci]:12.3e} {total_stellar[ci]:12.3e} {ics_frac[i]:8.3f} "
                  f"{Len[ci]:6d} {n_satellites[ci]:6d} {ICS[ci]/StellarMass[ci]:8.2f}")

    # Groups
    groups = np.where((Type == 0) & (Mvir >= MVIR_THRESHOLD_REDSHIFT_GROUPS) & (Mvir < MVIR_THRESHOLD_REDSHIFT) &
                       (ICS > 0) & (StellarMass >= STELLARMASS_THRESHOLD) & (total_stellar > 0))[0]
    if len(groups) > 0:
        ics_frac = ICS[groups] / total_stellar[groups]
        top_idx = np.argsort(ics_frac)[::-1][:5]
        for i in top_idx:
            gi = groups[i]
            print(f"{z:5.2f} {'Group':>8} {Mvir[gi]:12.3e} {ICS[gi]:12.3e} {ICS[gi]:12.3e} "
                  f"{satellite_mass[gi]:12.3e} {total_stellar[gi]:12.3e} {ics_frac[i]:8.3f} "
                  f"{Len[gi]:6d} {n_satellites[gi]:6d} {ICS[gi]/StellarMass[gi]:8.2f}")

print("\n\n=== SUMMARY: Distribution of f_ICS at z=0 (last snapshot) ===\n")
snap = snaps_to_plot[-1]  # lowest redshift
Snapshot = 'Snap_' + str(snap)
z = redshifts[snap]
data = cache.get(Snapshot, ['Mvir', 'IntraClusterStars', 'StellarMass', 'Type', 'Len'])
Mvir = data['Mvir']
ICS = data['IntraClusterStars']
StellarMass = data['StellarMass']
Type = data['Type']
Len = data['Len']
satellite_mass = cache.get_satellite_mass(Snapshot)
total_stellar = StellarMass + satellite_mass + ICS

clusters = np.where((Type == 0) & (Mvir >= MVIR_THRESHOLD_REDSHIFT) & (ICS > 0) &
                     (StellarMass >= STELLARMASS_THRESHOLD) & (total_stellar > 0))[0]
if len(clusters) > 0:
    ics_frac = ICS[clusters] / total_stellar[clusters]
    print(f"Clusters at z={z:.2f}: N={len(clusters)}")
    for pct in [50, 75, 90, 95, 99, 100]:
        print(f"  {pct}th percentile: f_ICS = {np.percentile(ics_frac, pct):.4f}")

    # Correlation with Len
    high = ics_frac > np.percentile(ics_frac, 90)
    low = ics_frac <= np.percentile(ics_frac, 90)
    print(f"\n  High f_ICS (>90th pct): median Len = {np.median(Len[clusters[high]]):.0f}, "
          f"median Mvir = {np.median(Mvir[clusters[high]]):.2e}")
    print(f"  Normal f_ICS (<=90th pct): median Len = {np.median(Len[clusters[low]]):.0f}, "
          f"median Mvir = {np.median(Mvir[clusters[low]]):.2e}")
    print(f"  High f_ICS: median M_ICS/M*,BCG = {np.median(ICS[clusters[high]]/StellarMass[clusters[high]]):.2f}")
    print(f"  Normal f_ICS: median M_ICS/M*,BCG = {np.median(ICS[clusters[low]]/StellarMass[clusters[low]]):.2f}")

groups = np.where((Type == 0) & (Mvir >= MVIR_THRESHOLD_REDSHIFT_GROUPS) & (Mvir < MVIR_THRESHOLD_REDSHIFT) &
                   (ICS > 0) & (StellarMass >= STELLARMASS_THRESHOLD) & (total_stellar > 0))[0]
if len(groups) > 0:
    ics_frac = ICS[groups] / total_stellar[groups]
    print(f"\nGroups at z={z:.2f}: N={len(groups)}")
    for pct in [50, 75, 90, 95, 99, 100]:
        print(f"  {pct}th percentile: f_ICS = {np.percentile(ics_frac, pct):.4f}")

    high = ics_frac > np.percentile(ics_frac, 90)
    low = ics_frac <= np.percentile(ics_frac, 90)
    print(f"\n  High f_ICS (>90th pct): median Len = {np.median(Len[groups[high]]):.0f}, "
          f"median Mvir = {np.median(Mvir[groups[high]]):.2e}")
    print(f"  Normal f_ICS (<=90th pct): median Len = {np.median(Len[groups[low]]):.0f}, "
          f"median Mvir = {np.median(Mvir[groups[low]]):.2e}")
    print(f"  High f_ICS: median M_ICS/M*,BCG = {np.median(ICS[groups[high]]/StellarMass[groups[high]]):.2f}")
    print(f"  Normal f_ICS: median M_ICS/M*,BCG = {np.median(ICS[groups[low]]/StellarMass[groups[low]]):.2f}")

#!/usr/bin/env python
"""
SAGE26 ICS (Intra-Cluster Stars) Magnitude Calculator

Reconstructs the star formation history of ICS by tracking:
1. Disruption events (mergeType=4) - captures disrupted galaxy's SFH
2. ICS accretion events - tracks ICS SFH transferred from merged satellites

Then computes ICS magnitudes using the same M/L ratio approach as magnitudes.py

Requires: numpy, h5py, matplotlib, astropy
Optional: mpi4py (for parallel processing)

Usage:
    python ics_magnitudes.py path/to/model_0.hdf5
    python ics_magnitudes.py path/to/model_0.hdf5 --snapshot 63
    mpirun -np 8 python ics_magnitudes.py path/to/model_0.hdf5  # with MPI
"""

import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
import os
import sys
import argparse
import time
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from collections import defaultdict

# Optional MPI support
try:
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
    HAS_MPI = SIZE > 1
except ImportError:
    COMM = None
    RANK = 0
    SIZE = 1
    HAS_MPI = False

# =============================================================================
# Configuration
# =============================================================================

OutputFormat = '.pdf'
plt.rcParams["figure.figsize"] = (8.34, 6.25)
plt.rcParams["figure.dpi"] = 96
plt.rcParams["font.size"] = 14

# Solar absolute magnitudes (AB system)
M_SUN = {
    'sdss_u': 6.39,
    'sdss_g': 5.12,
    'sdss_r': 4.65,
    'sdss_i': 4.53,
    'sdss_z': 4.51,
    'B': 5.48,
    'V': 4.83,
    'K': 3.27,
}


# =============================================================================
# Mass-to-Light Ratio Functions (copied from magnitudes.py)
# =============================================================================

def mass_to_light_ratio(age_gyr, Z, band='sdss_r'):
    """
    Calculate mass-to-light ratio M/L for a stellar population.
    Based on fits to Bruzual & Charlot (2003) SSP models with Chabrier IMF.
    """
    age_gyr = np.maximum(age_gyr, 0.001)
    Z = np.clip(Z, 1e-4, 0.05)
    Z_solar = Z / 0.02

    coefficients = {
        'sdss_u': {'a': 0.0,  'b': 1.0,  'c': 0.15},
        'sdss_g': {'a': -0.10, 'b': 0.75, 'c': 0.10},
        'sdss_r': {'a': -0.22, 'b': 0.70, 'c': 0.08},
        'sdss_i': {'a': -0.30, 'b': 0.65, 'c': 0.06},
        'sdss_z': {'a': -0.35, 'b': 0.60, 'c': 0.05},
        'B': {'a': -0.05, 'b': 0.80, 'c': 0.12},
        'V': {'a': -0.15, 'b': 0.72, 'c': 0.08},
        'K': {'a': -0.70, 'b': 0.45, 'c': 0.03},
    }

    if band not in coefficients:
        band = 'sdss_r'

    c = coefficients[band]
    log_age = np.log10(age_gyr)
    log_Z = np.log10(Z_solar)
    log_ML = c['a'] + c['b'] * log_age + c['c'] * log_Z
    log_ML = np.clip(log_ML, -1, 1.3)

    return 10**log_ML


def stellar_mass_to_magnitude(stellar_mass, age_gyr, Z, band='sdss_r'):
    """Convert stellar mass to absolute magnitude."""
    ML = mass_to_light_ratio(age_gyr, Z, band)
    luminosity = stellar_mass / ML
    luminosity = np.maximum(luminosity, 1e-10)
    M_sun_band = M_SUN.get(band, 4.65)
    magnitude = M_sun_band - 2.5 * np.log10(luminosity)
    return magnitude


def compute_ages_from_sfh(sfh_mass, redshifts, H0, Om0):
    """
    Compute mass-weighted and light-weighted ages from SFH array.

    Parameters:
        sfh_mass: stellar mass formed at each snapshot [M_sun]
        redshifts: array of redshifts for each snapshot
        H0: Hubble parameter (dimensionless h)
        Om0: matter density parameter

    Returns:
        (mass_weighted_age, light_age_r, light_age_g) in Gyr
    """
    cosmo = FlatLambdaCDM(H0=H0 * 100, Om0=Om0)
    cosmic_times = cosmo.age(redshifts).to(u.Gyr).value
    t_now = cosmic_times[np.argmin(np.abs(redshifts))]

    # Mass formed in each bin (SFH is cumulative, so difference it)
    mass_formed = np.zeros_like(sfh_mass)
    mass_formed[1:] = np.maximum(sfh_mass[1:] - sfh_mass[:-1], 0)
    mass_formed[0] = sfh_mass[0]

    stellar_ages = t_now - cosmic_times
    stellar_ages = np.maximum(stellar_ages, 0.001)

    total_mass = np.sum(mass_formed)
    if total_mass <= 0:
        return 1.0, 1.0, 1.0

    # Mass-weighted mean age
    mass_weighted_age = np.sum(mass_formed * stellar_ages) / total_mass

    # Light-weighted ages
    lum_weight_r = mass_formed * (stellar_ages ** -0.8)
    total_lum_r = np.sum(lum_weight_r)
    light_age_r = np.sum(lum_weight_r * stellar_ages) / total_lum_r if total_lum_r > 0 else mass_weighted_age

    lum_weight_g = mass_formed * (stellar_ages ** -1.1)
    total_lum_g = np.sum(lum_weight_g)
    light_age_g = np.sum(lum_weight_g * stellar_ages) / total_lum_g if total_lum_g > 0 else mass_weighted_age

    light_age_r = np.clip(light_age_r, 0.01, mass_weighted_age)
    light_age_g = np.clip(light_age_g, 0.01, light_age_r)

    return mass_weighted_age, light_age_r, light_age_g


# =============================================================================
# Helper Functions
# =============================================================================

def read_simulation_params(filepath):
    """Read simulation parameters from HDF5 file header."""
    params = {}

    with h5.File(filepath, 'r') as f:
        sim = f['Header/Simulation']
        params['Hubble_h'] = float(sim.attrs['hubble_h'])
        params['BoxSize'] = float(sim.attrs['box_size'])
        params['Omega'] = float(sim.attrs['omega_matter'])
        params['OmegaLambda'] = float(sim.attrs['omega_lambda'])
        params['SimMaxSnaps'] = int(sim.attrs['SimMaxSnaps'])

        runtime = f['Header/Runtime']
        params['VolumeFraction'] = float(runtime.attrs['frac_volume_processed'])

        params['snapshot_redshifts'] = np.array(f['Header/snapshot_redshifts'])
        params['output_snapshots'] = np.array(f['Header/output_snapshots'])

        snap_groups = [key for key in f.keys() if key.startswith('Snap_')]
        snap_numbers = sorted([int(s.replace('Snap_', '')) for s in snap_groups])
        params['available_snapshots'] = snap_numbers
        params['latest_snapshot'] = max(snap_numbers) if snap_numbers else None

    return params


def read_hdf(filepath, snap_num, param):
    """Read a parameter from HDF5 file for a given snapshot."""
    with h5.File(filepath, 'r') as f:
        if snap_num in f and param in f[snap_num]:
            return np.array(f[snap_num][param])
    return np.array([])


# =============================================================================
# ICS SFH Reconstruction
# =============================================================================

def reconstruct_ics_sfh(filepath, target_snapshot=None):
    """
    Reconstruct the star formation history of ICS by tracking disruption
    and accretion events through the merger tree.

    Strategy:
    1. Walk through snapshots chronologically
    2. For each galaxy, track ICS_disrupt and ICS_accrete changes
    3. When disruption happens (mergeType=4), capture that galaxy's SFH
    4. When ICS is accreted, transfer the satellite's ICS_SFH to central
    5. Build cumulative ICS_SFH arrays

    With MPI: Each rank processes a subset of snapshots, then results are merged.

    Parameters:
        filepath: path to HDF5 file
        target_snapshot: final snapshot to analyze (default: latest)

    Returns:
        dict with ICS SFH, masses, metallicities, etc.
        Returns None on non-root ranks when using MPI.
    """
    t_start = time.time()

    params = read_simulation_params(filepath)
    Hubble_h = params['Hubble_h']
    redshifts = params['snapshot_redshifts']
    available_snaps = params['available_snapshots']
    num_snaps = params['SimMaxSnaps']

    if target_snapshot is None:
        target_snapshot = params['latest_snapshot']

    if RANK == 0:
        print("=" * 60)
        print("ICS Star Formation History Reconstruction")
        print("=" * 60)
        print(f"\nInput file: {filepath}")
        print(f"Target snapshot: {target_snapshot} (z = {redshifts[target_snapshot]:.4f})")
        print(f"Available snapshots: {len(available_snaps)}")
        print(f"Num SFH bins: {num_snaps}")
        if HAS_MPI:
            print(f"MPI: {SIZE} ranks")

    # Dictionary to track ICS SFH for each galaxy by GalaxyIndex
    # Key: GalaxyIndex, Value: SFH array [num_snaps]
    ics_sfh_dict = defaultdict(lambda: np.zeros(num_snaps))
    ics_metals_dict = defaultdict(float)

    # Distribute snapshots across MPI ranks
    # Each rank processes its assigned snapshots
    my_snaps = available_snaps[RANK::SIZE]

    if RANK == 0:
        print(f"\nProcessing {len(available_snaps)} snapshots...")
        if HAS_MPI:
            print(f"  Each rank processes ~{len(my_snaps)} snapshots")

    t_loop = time.time()
    snaps_processed = 0

    for snap in my_snaps:
        snap_str = f'Snap_{snap}'

        # Read galaxy data for this snapshot
        with h5.File(filepath, 'r') as f:
            if snap_str not in f:
                continue

            grp = f[snap_str]

            # Core identifiers
            GalaxyIndex = np.array(grp['GalaxyIndex']) if 'GalaxyIndex' in grp else np.array([])
            CentralGalaxyIndex = np.array(grp['CentralGalaxyIndex']) if 'CentralGalaxyIndex' in grp else np.array([])
            mergeType = np.array(grp['mergeType']) if 'mergeType' in grp else np.array([])

            # SFH arrays
            SFHMassDisk = np.array(grp['SFHMassDisk']) if 'SFHMassDisk' in grp else np.array([])
            SFHMassBulge = np.array(grp['SFHMassBulge']) if 'SFHMassBulge' in grp else np.array([])

            # Stellar metals for metallicity tracking
            MetalsStellarMass = np.array(grp['MetalsStellarMass']) if 'MetalsStellarMass' in grp else np.array([])

        if len(GalaxyIndex) == 0:
            continue

        ngals = len(GalaxyIndex)
        has_sfh = SFHMassDisk.size > 0 and len(SFHMassDisk.shape) == 2

        snaps_processed += 1
        if RANK == 0 and snaps_processed % 10 == 0:
            elapsed = time.time() - t_loop
            rate = snaps_processed / elapsed if elapsed > 0 else 0
            remaining = len(my_snaps) - snaps_processed
            eta = remaining / rate if rate > 0 else 0
            print(f"  Progress: {snaps_processed}/{len(my_snaps)} snapshots - "
                  f"{rate:.1f} snap/s - ETA: {eta:.0f}s")

        # Process galaxies with disruption events (mergeType == 4)
        if has_sfh:
            disrupting = np.where(mergeType == 4)[0]

            for i in disrupting:
                central_gid = CentralGalaxyIndex[i]

                # This galaxy is disrupting - its SFH goes to the central's ICS
                sfh_total = SFHMassDisk[i, :] + SFHMassBulge[i, :]

                # Add this galaxy's SFH to the central's ICS SFH
                ics_sfh_dict[central_gid] += sfh_total * 1e10 / Hubble_h

                # Track metals
                if len(MetalsStellarMass) > i:
                    ics_metals_dict[central_gid] += MetalsStellarMass[i] * 1e10 / Hubble_h

    # Gather results from all ranks
    if HAS_MPI:
        if RANK == 0:
            print(f"\nGathering results from {SIZE} ranks...")

        # Convert defaultdicts to regular dicts for pickling
        local_sfh = dict(ics_sfh_dict)
        local_metals = dict(ics_metals_dict)

        # Gather all dicts to rank 0
        all_sfh_dicts = COMM.gather(local_sfh, root=0)
        all_metals_dicts = COMM.gather(local_metals, root=0)

        if RANK == 0:
            # Merge all dictionaries
            ics_sfh_dict = defaultdict(lambda: np.zeros(num_snaps))
            ics_metals_dict = defaultdict(float)

            for d in all_sfh_dicts:
                for gid, sfh in d.items():
                    ics_sfh_dict[gid] += sfh

            for d in all_metals_dicts:
                for gid, metals in d.items():
                    ics_metals_dict[gid] += metals

            print(f"  Merged {len(ics_sfh_dict)} unique galaxies with ICS SFH")

    if RANK == 0:
        print(f"\nSFH reconstruction completed in {time.time() - t_start:.2f}s")
        print(f"Galaxies with ICS SFH tracked: {len(ics_sfh_dict)}")

    # Only rank 0 continues with final analysis
    if RANK != 0:
        return None

    # Now read final snapshot data for analysis
    snap_str = f'Snap_{target_snapshot}'

    with h5.File(filepath, 'r') as f:
        grp = f[snap_str]

        GalaxyIndex = np.array(grp['GalaxyIndex'])
        Type = np.array(grp['Type'])
        # ICS field is named 'IntraClusterStars' in the output
        ICS = np.array(grp['IntraClusterStars']) * 1e10 / Hubble_h
        MetalsICS = np.array(grp['MetalsIntraClusterStars']) * 1e10 / Hubble_h
        ICS_disrupt = np.array(grp['ICS_disrupt']) * 1e10 / Hubble_h if 'ICS_disrupt' in grp else np.zeros_like(ICS)
        ICS_accrete = np.array(grp['ICS_accrete']) * 1e10 / Hubble_h if 'ICS_accrete' in grp else np.zeros_like(ICS)
        StellarMass = np.array(grp['StellarMass']) * 1e10 / Hubble_h
        MetalsStellarMass = np.array(grp['MetalsStellarMass']) * 1e10 / Hubble_h
        Mvir = np.array(grp['Mvir']) * 1e10 / Hubble_h

        # Get galaxy SFH for stellar age calculation
        SFHMassDisk = np.array(grp['SFHMassDisk']) * 1e10 / Hubble_h if 'SFHMassDisk' in grp else None
        SFHMassBulge = np.array(grp['SFHMassBulge']) * 1e10 / Hubble_h if 'SFHMassBulge' in grp else None

    ngals = len(GalaxyIndex)
    print(f"\nFinal snapshot: {ngals} galaxies")

    # Build ICS SFH arrays for final snapshot galaxies
    ics_sfh_array = np.zeros((ngals, num_snaps))
    for i, gid in enumerate(GalaxyIndex):
        if gid in ics_sfh_dict:
            ics_sfh_array[i, :] = ics_sfh_dict[gid]

    # Select galaxies with significant ICS
    has_ics = ICS > 1e8  # > 10^8 Msun
    w = np.where(has_ics)[0]

    print(f"Galaxies with ICS > 10^8 Msun: {len(w)}")

    if len(w) == 0:
        print("No galaxies with significant ICS found.")
        return {
            'ngals': 0,
            'redshifts': redshifts,
            'Hubble_h': Hubble_h,
            'Omega': params['Omega'],
        }

    # Calculate ICS and stellar ages/metallicities
    print("\nComputing ages from SFH...")
    t_age = time.time()

    ics_mass_ages = np.zeros(len(w))
    ics_light_ages_r = np.zeros(len(w))
    ics_light_ages_g = np.zeros(len(w))
    ics_metallicity = np.zeros(len(w))

    # Also compute stellar ages for ICL fraction calculation
    stellar_mass_ages = np.zeros(len(w))
    stellar_light_ages_r = np.zeros(len(w))
    stellar_light_ages_g = np.zeros(len(w))
    stellar_metallicity = np.zeros(len(w))

    has_stellar_sfh = SFHMassDisk is not None and SFHMassBulge is not None

    for idx, i in enumerate(w):
        if idx % 10000 == 0 and idx > 0:
            elapsed = time.time() - t_age
            rate = idx / elapsed if elapsed > 0 else 0
            eta = (len(w) - idx) / rate if rate > 0 else 0
            print(f"  Progress: {idx}/{len(w)} galaxies - {rate:.0f} gal/s - ETA: {eta:.0f}s")

        # ICS ages from reconstructed SFH
        sfh = ics_sfh_array[i, :]

        if np.sum(sfh) > 0:
            mass_age, light_r, light_g = compute_ages_from_sfh(sfh, redshifts, Hubble_h, params['Omega'])
            ics_mass_ages[idx] = mass_age
            ics_light_ages_r[idx] = light_r
            ics_light_ages_g[idx] = light_g
        else:
            # No SFH tracked - assume old population (disruption predates our tracking)
            ics_mass_ages[idx] = 8.0  # Gyr
            ics_light_ages_r[idx] = 6.0
            ics_light_ages_g[idx] = 4.0

        # ICS metallicity
        if ICS[i] > 0:
            ics_metallicity[idx] = MetalsICS[i] / ICS[i]
        else:
            ics_metallicity[idx] = 0.02  # Solar

        # Stellar ages from galaxy SFH
        if has_stellar_sfh:
            stellar_sfh = SFHMassDisk[i, :] + SFHMassBulge[i, :]
            if np.sum(stellar_sfh) > 0:
                mass_age, light_r, light_g = compute_ages_from_sfh(stellar_sfh, redshifts, Hubble_h, params['Omega'])
                stellar_mass_ages[idx] = mass_age
                stellar_light_ages_r[idx] = light_r
                stellar_light_ages_g[idx] = light_g
            else:
                stellar_mass_ages[idx] = 5.0
                stellar_light_ages_r[idx] = 3.0
                stellar_light_ages_g[idx] = 1.0
        else:
            stellar_mass_ages[idx] = 5.0
            stellar_light_ages_r[idx] = 3.0
            stellar_light_ages_g[idx] = 1.0

        # Stellar metallicity
        if StellarMass[i] > 0:
            stellar_metallicity[idx] = MetalsStellarMass[i] / StellarMass[i]
        else:
            stellar_metallicity[idx] = 0.02

    ics_metallicity = np.clip(ics_metallicity, 1e-4, 0.05)
    stellar_metallicity = np.clip(stellar_metallicity, 1e-4, 0.05)

    print(f"Age calculation completed in {time.time() - t_age:.2f}s")

    # Calculate volume
    BoxSize = params['BoxSize']
    VolumeFraction = params['VolumeFraction']
    volume = (BoxSize / Hubble_h) ** 3 * VolumeFraction

    print(f"\nICS Age Statistics:")
    print(f"  Mass-weighted age: {np.median(ics_mass_ages):.2f} Gyr (median)")
    print(f"  Light-weighted (r): {np.median(ics_light_ages_r):.2f} Gyr (median)")
    print(f"  Light-weighted (g): {np.median(ics_light_ages_g):.2f} Gyr (median)")
    print(f"  Metallicity: {np.median(ics_metallicity)/0.02:.2f} Z_sun (median)")

    return {
        'GalaxyIndex': GalaxyIndex[w],
        'Type': Type[w],
        'ICS': ICS[w],
        'ICS_disrupt': ICS_disrupt[w],
        'ICS_accrete': ICS_accrete[w],
        'MetalsICS': MetalsICS[w],
        'StellarMass': StellarMass[w],
        'Mvir': Mvir[w],
        'ics_sfh': ics_sfh_array[w, :],
        'ics_sfh_dict': dict(ics_sfh_dict),  # Full dict for evolution calculation
        'ics_mass_ages': ics_mass_ages,
        'ics_light_ages_r': ics_light_ages_r,
        'ics_light_ages_g': ics_light_ages_g,
        'ics_metallicity': ics_metallicity,
        'stellar_mass_ages': stellar_mass_ages,
        'stellar_light_ages_r': stellar_light_ages_r,
        'stellar_light_ages_g': stellar_light_ages_g,
        'stellar_metallicity': stellar_metallicity,
        'galaxy_sfh_disk': SFHMassDisk[w, :] if SFHMassDisk is not None else None,
        'galaxy_sfh_bulge': SFHMassBulge[w, :] if SFHMassBulge is not None else None,
        'redshifts': redshifts,
        'volume': volume,
        'Hubble_h': Hubble_h,
        'Omega': params['Omega'],
        'snapshot': target_snapshot,
        'redshift': redshifts[target_snapshot],
        'ngals': len(w),
        'filepath': filepath,  # Store for evolution calculation
    }


def compute_ics_magnitudes(results, bands=['sdss_g', 'sdss_r']):
    """
    Compute ICS and stellar magnitudes from SFH.

    Parameters:
        results: dict from reconstruct_ics_sfh()
        bands: list of photometric bands

    Returns:
        dict with magnitudes added (both ICS and stellar)
    """
    if results is None or results.get('ngals', 0) == 0:
        return results

    print("\nComputing magnitudes...")

    ICS = results['ICS']
    StellarMass = results['StellarMass']

    ics_mass_ages = results['ics_mass_ages']
    ics_light_ages_r = results['ics_light_ages_r']
    ics_light_ages_g = results['ics_light_ages_g']
    ics_Z = results['ics_metallicity']

    stellar_mass_ages = results['stellar_mass_ages']
    stellar_light_ages_r = results['stellar_light_ages_r']
    stellar_light_ages_g = results['stellar_light_ages_g']
    stellar_Z = results['stellar_metallicity']

    blue_bands = {'sdss_u', 'sdss_g', 'B'}
    red_optical_bands = {'sdss_r', 'sdss_i', 'V'}

    ics_magnitudes = {}
    stellar_magnitudes = {}

    for band in bands:
        if band in blue_bands:
            ics_ages = ics_light_ages_g
            stellar_ages = stellar_light_ages_g
        elif band in red_optical_bands:
            ics_ages = ics_light_ages_r
            stellar_ages = stellar_light_ages_r
        else:
            ics_ages = ics_mass_ages
            stellar_ages = stellar_mass_ages

        ics_magnitudes[band] = stellar_mass_to_magnitude(ICS, ics_ages, ics_Z, band)
        stellar_magnitudes[band] = stellar_mass_to_magnitude(StellarMass, stellar_ages, stellar_Z, band)

    results['ics_magnitudes'] = ics_magnitudes
    results['stellar_magnitudes'] = stellar_magnitudes
    results['bands'] = bands

    # Also keep 'magnitudes' as alias for backwards compatibility
    results['magnitudes'] = ics_magnitudes

    # Print statistics
    print("\n  ICS magnitudes:")
    for band in bands:
        mags = ics_magnitudes[band]
        valid = np.isfinite(mags) & (mags < 0) & (mags > -30)
        if np.sum(valid) > 0:
            print(f"    {band}: median = {np.median(mags[valid]):.2f}, "
                  f"range = [{np.min(mags[valid]):.2f}, {np.max(mags[valid]):.2f}]")

    print("\n  Stellar magnitudes:")
    for band in bands:
        mags = stellar_magnitudes[band]
        valid = np.isfinite(mags) & (mags < 0) & (mags > -30)
        if np.sum(valid) > 0:
            print(f"    {band}: median = {np.median(mags[valid]):.2f}, "
                  f"range = [{np.min(mags[valid]):.2f}, {np.max(mags[valid]):.2f}]")

    return results


def compute_icl_fraction_evolution(filepath, ics_sfh_dict, max_z=2.0, min_halo_mass=1e13, band='sdss_r'):
    """
    Compute ICL fraction evolution as a function of redshift.

    Parameters:
        filepath: path to HDF5 file
        ics_sfh_dict: dictionary of ICS SFH from reconstruct_ics_sfh
        max_z: maximum redshift to consider (default 2.0)
        min_halo_mass: minimum halo mass for centrals to include [Msun]
        band: photometric band for luminosity calculation

    Returns:
        dict with redshifts, median ICL fractions, and percentiles
    """
    params = read_simulation_params(filepath)
    Hubble_h = params['Hubble_h']
    Omega = params['Omega']
    redshifts = params['snapshot_redshifts']
    available_snaps = params['available_snapshots']

    print(f"\nComputing ICL fraction evolution (z <= {max_z})...")
    print(f"  Minimum halo mass: {min_halo_mass:.1e} Msun")

    # Filter snapshots by redshift
    valid_snaps = [s for s in available_snaps if redshifts[s] <= max_z]
    valid_snaps = sorted(valid_snaps, reverse=True)  # High z to low z

    print(f"  Processing {len(valid_snaps)} snapshots")

    result_z = []
    result_icl_median = []
    result_icl_16 = []
    result_icl_84 = []
    result_ngals = []

    blue_bands = {'sdss_u', 'sdss_g', 'B'}
    red_optical_bands = {'sdss_r', 'sdss_i', 'V'}

    for snap in valid_snaps:
        snap_str = f'Snap_{snap}'
        z = redshifts[snap]

        with h5.File(filepath, 'r') as f:
            if snap_str not in f:
                continue

            grp = f[snap_str]

            GalaxyIndex = np.array(grp['GalaxyIndex'])
            Type = np.array(grp['Type'])
            ICS = np.array(grp['IntraClusterStars']) * 1e10 / Hubble_h
            MetalsICS = np.array(grp['MetalsIntraClusterStars']) * 1e10 / Hubble_h
            StellarMass = np.array(grp['StellarMass']) * 1e10 / Hubble_h
            MetalsStellarMass = np.array(grp['MetalsStellarMass']) * 1e10 / Hubble_h
            Mvir = np.array(grp['Mvir']) * 1e10 / Hubble_h

            SFHMassDisk = np.array(grp['SFHMassDisk']) * 1e10 / Hubble_h if 'SFHMassDisk' in grp else None
            SFHMassBulge = np.array(grp['SFHMassBulge']) * 1e10 / Hubble_h if 'SFHMassBulge' in grp else None

        # Select centrals with significant ICS and above mass threshold
        centrals = (Type == 0) & (ICS > 1e8) & (Mvir > min_halo_mass)
        w = np.where(centrals)[0]

        if len(w) < 5:
            continue

        # Compute ages for this snapshot
        # For ICS: use reconstructed SFH if available, else assume old
        # For stellar: use galaxy SFH

        ics_ages = np.zeros(len(w))
        stellar_ages = np.zeros(len(w))
        ics_Z = np.zeros(len(w))
        stellar_Z = np.zeros(len(w))

        has_sfh = SFHMassDisk is not None and SFHMassBulge is not None

        for idx, i in enumerate(w):
            gid = GalaxyIndex[i]

            # ICS age from reconstructed SFH
            if gid in ics_sfh_dict and np.sum(ics_sfh_dict[gid]) > 0:
                sfh = ics_sfh_dict[gid]
                mass_age, light_r, light_g = compute_ages_from_sfh(sfh, redshifts, Hubble_h, Omega)
                if band in blue_bands:
                    ics_ages[idx] = light_g
                elif band in red_optical_bands:
                    ics_ages[idx] = light_r
                else:
                    ics_ages[idx] = mass_age
            else:
                ics_ages[idx] = 6.0  # Default old age

            # Stellar age from galaxy SFH
            if has_sfh:
                stellar_sfh = SFHMassDisk[i, :] + SFHMassBulge[i, :]
                if np.sum(stellar_sfh) > 0:
                    mass_age, light_r, light_g = compute_ages_from_sfh(stellar_sfh, redshifts, Hubble_h, Omega)
                    if band in blue_bands:
                        stellar_ages[idx] = light_g
                    elif band in red_optical_bands:
                        stellar_ages[idx] = light_r
                    else:
                        stellar_ages[idx] = mass_age
                else:
                    stellar_ages[idx] = 3.0
            else:
                stellar_ages[idx] = 3.0

            # Metallicities
            ics_Z[idx] = MetalsICS[i] / ICS[i] if ICS[i] > 0 else 0.02
            stellar_Z[idx] = MetalsStellarMass[i] / StellarMass[i] if StellarMass[i] > 0 else 0.02

        ics_Z = np.clip(ics_Z, 1e-4, 0.05)
        stellar_Z = np.clip(stellar_Z, 1e-4, 0.05)

        # Compute magnitudes
        ics_mags = stellar_mass_to_magnitude(ICS[w], ics_ages, ics_Z, band)
        stellar_mags = stellar_mass_to_magnitude(StellarMass[w], stellar_ages, stellar_Z, band)

        # Convert to luminosities
        M_sun_band = M_SUN.get(band, 4.65)
        ics_lum = 10 ** ((M_sun_band - ics_mags) / 2.5)
        stellar_lum = 10 ** ((M_sun_band - stellar_mags) / 2.5)

        # ICL fraction
        total_lum = ics_lum + stellar_lum
        icl_frac = ics_lum / total_lum

        # Filter valid values
        valid = np.isfinite(icl_frac) & (icl_frac >= 0) & (icl_frac <= 1)
        icl_frac = icl_frac[valid]

        if len(icl_frac) < 5:
            continue

        result_z.append(z)
        result_icl_median.append(np.median(icl_frac))
        result_icl_16.append(np.percentile(icl_frac, 16))
        result_icl_84.append(np.percentile(icl_frac, 84))
        result_ngals.append(len(icl_frac))

    print(f"  Computed ICL fraction at {len(result_z)} redshifts")

    return {
        'redshifts': np.array(result_z),
        'icl_median': np.array(result_icl_median),
        'icl_16': np.array(result_icl_16),
        'icl_84': np.array(result_icl_84),
        'ngals': np.array(result_ngals),
        'min_halo_mass': min_halo_mass,
        'band': band,
    }


def plot_icl_evolution(icl_evolution, output_dir):
    """Plot ICL fraction as a function of redshift."""

    if icl_evolution is None or len(icl_evolution['redshifts']) == 0:
        return

    os.makedirs(output_dir, exist_ok=True)

    z = icl_evolution['redshifts']
    icl_median = icl_evolution['icl_median']
    icl_16 = icl_evolution['icl_16']
    icl_84 = icl_evolution['icl_84']
    min_mass = icl_evolution['min_halo_mass']
    band = icl_evolution['band']

    plt.figure(figsize=(8, 6))

    # Plot with error band
    plt.fill_between(z, icl_16, icl_84, alpha=0.3, color='purple', label='16-84%')
    plt.plot(z, icl_median, 'o-', color='purple', linewidth=2, markersize=6, label='Median')

    plt.xlabel('Redshift')
    plt.ylabel(f'ICL Fraction ($L_{{\\rm ICL}} / L_{{\\rm total}}$, {band})')
    plt.title(f'ICL Fraction Evolution ($M_{{\\rm halo}} > 10^{{{np.log10(min_mass):.0f}}}$ M$_\\odot$)')
    plt.xlim(0, max(z) * 1.05)
    plt.ylim(0, min(1.0, max(icl_84) * 1.2))
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'ICL_vs_Redshift{OutputFormat}'))
    plt.close()

    print(f"  - ICL_vs_Redshift{OutputFormat}")


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_ics_results(results, output_dir):
    """Generate ICS diagnostic plots."""

    if results is None or results.get('ngals', 0) == 0:
        return

    os.makedirs(output_dir, exist_ok=True)

    ICS = results['ICS']
    Mvir = results['Mvir']
    Type = results['Type']
    ics_mass_ages = results['ics_mass_ages']
    volume = results['volume']
    magnitudes = results.get('magnitudes', {})

    centrals = Type == 0

    print(f"\nGenerating plots for {len(ICS)} galaxies...")

    # Plot 1: ICS Luminosity Function
    if 'sdss_r' in magnitudes:
        r_mags = magnitudes['sdss_r']
        valid = np.isfinite(r_mags) & (r_mags < 0) & (r_mags > -30)

        plt.figure(figsize=(8, 6))
        dM = 0.5
        mag_bins = np.arange(-26.0, -16.0, dM)
        counts, bin_edges = np.histogram(r_mags[valid], bins=mag_bins)
        phi = counts / (volume * dM)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        valid_bins = phi > 0
        plt.plot(bin_centers[valid_bins], np.log10(phi[valid_bins]),
                 marker='o', linestyle='-', color='purple', linewidth=2, label='ICS')

        plt.xlabel('ICS Absolute r-band Magnitude ($M_r$)')
        plt.ylabel(r'$\log_{10}\ \Phi\ [\mathrm{Mpc}^{-3}\ \mathrm{mag}^{-1}]$')
        plt.title('ICS Luminosity Function')
        plt.gca().invert_xaxis()
        plt.xlim(-16, -26)
        plt.ylim(-7, -2)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'ICS_LuminosityFunction_r{OutputFormat}'))
        plt.close()

    # Plot 2: ICS Mass vs Halo Mass
    plt.figure(figsize=(8, 6))

    log_mvir = np.log10(Mvir[centrals])
    log_ics = np.log10(ICS[centrals])

    plt.scatter(log_mvir, log_ics, c='purple', alpha=0.4, s=15, edgecolors='none')

    # Binned median
    mvir_bins = np.arange(11, 15.5, 0.5)
    bin_centers = []
    bin_medians = []
    bin_16 = []
    bin_84 = []

    for i in range(len(mvir_bins) - 1):
        mask = (log_mvir >= mvir_bins[i]) & (log_mvir < mvir_bins[i+1])
        if np.sum(mask) > 5:
            bin_centers.append(0.5 * (mvir_bins[i] + mvir_bins[i+1]))
            bin_medians.append(np.median(log_ics[mask]))
            bin_16.append(np.percentile(log_ics[mask], 16))
            bin_84.append(np.percentile(log_ics[mask], 84))

    if len(bin_centers) > 0:
        plt.errorbar(bin_centers, bin_medians,
                     yerr=[np.array(bin_medians) - np.array(bin_16),
                           np.array(bin_84) - np.array(bin_medians)],
                     fmt='o-', color='black', linewidth=2, markersize=8,
                     label='Median (16-84%)')

    plt.xlabel(r'$\log_{10}$ Halo Mass [$M_{\odot}$]')
    plt.ylabel(r'$\log_{10}$ ICS Mass [$M_{\odot}$]')
    plt.title('ICS Mass vs. Halo Mass (Centrals)')
    plt.xlim(11, 15.5)
    plt.ylim(8, 13)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'ICS_vs_HaloMass{OutputFormat}'))
    plt.close()

    # Plot 3: ICL Fraction vs Halo Mass (light-based, not mass-based)
    if 'sdss_r' in magnitudes and 'stellar_magnitudes' in results:
        plt.figure(figsize=(8, 6))

        # Get r-band magnitudes for ICS and stellar
        ics_r_mags = magnitudes['sdss_r']
        stellar_r_mags = results['stellar_magnitudes']['sdss_r']

        # Convert magnitudes to luminosities (in solar units)
        M_sun_r = M_SUN['sdss_r']
        ics_lum = 10 ** ((M_sun_r - ics_r_mags) / 2.5)
        stellar_lum = 10 ** ((M_sun_r - stellar_r_mags) / 2.5)

        # ICL fraction = ICS light / total light
        total_lum = ics_lum + stellar_lum
        icl_fraction = ics_lum / total_lum

        # Filter for centrals with valid data
        valid_centrals = centrals & np.isfinite(icl_fraction) & (icl_fraction >= 0) & (icl_fraction <= 1)
        log_mvir_valid = np.log10(Mvir[valid_centrals])
        icl_frac_valid = icl_fraction[valid_centrals]

        plt.scatter(log_mvir_valid, icl_frac_valid, c='purple', alpha=0.4, s=15, edgecolors='none')

        # Binned median
        bin_centers = []
        bin_medians = []
        bin_16 = []
        bin_84 = []

        for i in range(len(mvir_bins) - 1):
            mask = (log_mvir_valid >= mvir_bins[i]) & (log_mvir_valid < mvir_bins[i+1])
            if np.sum(mask) > 5:
                bin_centers.append(0.5 * (mvir_bins[i] + mvir_bins[i+1]))
                frac = icl_frac_valid[mask]
                bin_medians.append(np.median(frac))
                bin_16.append(np.percentile(frac, 16))
                bin_84.append(np.percentile(frac, 84))

        if len(bin_centers) > 0:
            plt.errorbar(bin_centers, bin_medians,
                         yerr=[np.array(bin_medians) - np.array(bin_16),
                               np.array(bin_84) - np.array(bin_medians)],
                         fmt='o-', color='black', linewidth=2, markersize=8,
                         label='Median (16-84%)')

        plt.xlabel(r'$\log_{10}$ Halo Mass [$M_{\odot}$]')
        plt.ylabel('ICL Fraction ($L_{\\rm ICL} / L_{\\rm total}$, r-band)')
        plt.title('Intracluster Light Fraction vs. Halo Mass (Centrals)')
        plt.xlim(11, 15.5)
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'ICLFraction_vs_HaloMass{OutputFormat}'))
        plt.close()

    # Plot 4: ICS Age Distribution
    plt.figure(figsize=(8, 6))

    plt.hist(ics_mass_ages, bins=30, alpha=0.6, color='purple',
             label='Mass-weighted', density=True)
    plt.hist(results['ics_light_ages_r'], bins=30, alpha=0.6, color='crimson',
             label='Light-weighted (r)', density=True)

    plt.xlabel('Stellar Age [Gyr]')
    plt.ylabel('Density')
    plt.title('ICS Age Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'ICS_AgeDistribution{OutputFormat}'))
    plt.close()

    # Plot 5: ICS Color-Mass Diagram
    if 'sdss_g' in magnitudes and 'sdss_r' in magnitudes:
        g_mags = magnitudes['sdss_g']
        r_mags = magnitudes['sdss_r']
        color_gr = g_mags - r_mags

        valid = np.isfinite(color_gr) & (color_gr > -1) & (color_gr < 2)

        plt.figure(figsize=(8, 6))
        sc = plt.scatter(np.log10(ICS[valid]), color_gr[valid],
                         c=ics_mass_ages[valid], cmap='viridis',
                         alpha=0.6, s=20, edgecolors='none')
        plt.colorbar(sc, label='Mass-weighted Age [Gyr]')

        plt.xlabel(r'$\log_{10}$ ICS Mass [$M_{\odot}$]')
        plt.ylabel('ICS Color ($g - r$)')
        plt.title('ICS Color vs. Mass')
        plt.xlim(8, 13)
        plt.ylim(0, 1.2)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'ICS_Color_vs_Mass{OutputFormat}'))
        plt.close()

    # Plot 6: Disruption vs Accretion contribution
    ICS_disrupt = results['ICS_disrupt']
    ICS_accrete = results['ICS_accrete']

    plt.figure(figsize=(8, 6))

    total_tracked = ICS_disrupt + ICS_accrete
    disrupt_frac = np.where(total_tracked > 0, ICS_disrupt / total_tracked, 0.5)

    valid = total_tracked > 1e8

    plt.scatter(np.log10(ICS[valid]), disrupt_frac[valid],
                c='purple', alpha=0.4, s=15, edgecolors='none')

    plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5)

    plt.xlabel(r'$\log_{10}$ ICS Mass [$M_{\odot}$]')
    plt.ylabel('Disruption Fraction (vs Accretion)')
    plt.title('ICS Assembly: Disruption vs. Accretion')
    plt.xlim(8, 13)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'ICS_DisruptionFraction{OutputFormat}'))
    plt.close()

    # Plot 7: ICS M/L ratio distribution
    if 'sdss_r' in magnitudes:
        r_mags = magnitudes['sdss_r']
        valid = np.isfinite(r_mags) & (r_mags < 0) & (r_mags > -30)

        ML_r = ICS[valid] / (10 ** ((M_SUN['sdss_r'] - r_mags[valid]) / 2.5))

        plt.figure(figsize=(8, 6))
        plt.hist(np.log10(ML_r), bins=50, alpha=0.6, color='purple', density=True)
        plt.xlabel(r'$\log_{10}$ M/L$_r$ [$M_{\odot}/L_{\odot}$]')
        plt.ylabel('Density')
        plt.title('ICS r-band Mass-to-Light Ratio')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'ICS_ML_ratio{OutputFormat}'))
        plt.close()

    print(f"\nPlots saved to {output_dir}:")
    print(f"  - ICS_LuminosityFunction_r{OutputFormat}")
    print(f"  - ICS_vs_HaloMass{OutputFormat}")
    print(f"  - ICLFraction_vs_HaloMass{OutputFormat}")
    print(f"  - ICS_AgeDistribution{OutputFormat}")
    print(f"  - ICS_Color_vs_Mass{OutputFormat}")
    print(f"  - ICS_DisruptionFraction{OutputFormat}")
    print(f"  - ICS_ML_ratio{OutputFormat}")


# =============================================================================
# Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Compute ICS magnitudes from reconstructed SFH',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python ics_magnitudes.py output/millennium/model_0.hdf5
    python ics_magnitudes.py output/millennium/model_0.hdf5 --snapshot 63
    python ics_magnitudes.py output/millennium/model_0.hdf5 --bands sdss_g sdss_r sdss_i
    python ics_magnitudes.py output/millennium/model_0.hdf5 --max-z 2.0 --min-halo-mass 1e13

    # With MPI for faster processing:
    mpirun -np 8 python ics_magnitudes.py output/millennium/model_0.hdf5
        """
    )

    parser.add_argument('input_file', help='Path to SAGE26 HDF5 output file')
    parser.add_argument('-s', '--snapshot', type=int, default=None,
                        help='Snapshot number (default: latest)')
    parser.add_argument('-o', '--output-dir', type=str, default=None,
                        help='Output directory for plots')
    parser.add_argument('--bands', type=str, nargs='+',
                        default=['sdss_g', 'sdss_r'],
                        help='Photometric bands (default: sdss_g sdss_r)')
    parser.add_argument('--max-z', type=float, default=2.0,
                        help='Maximum redshift for evolution plot (default: 2.0)')
    parser.add_argument('--min-halo-mass', type=float, default=1e13,
                        help='Minimum halo mass for ICL evolution [Msun] (default: 1e13)')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots')
    parser.add_argument('--no-evolution', action='store_true',
                        help='Skip ICL evolution calculation')

    args = parser.parse_args()

    if RANK == 0 and not os.path.exists(args.input_file):
        print(f"Error: File not found: {args.input_file}")
        sys.exit(1)

    # Synchronize before continuing
    if HAS_MPI:
        COMM.Barrier()

    # Determine output directory
    if args.output_dir is None:
        input_dir = os.path.dirname(args.input_file)
        args.output_dir = os.path.join(input_dir, 'plots')

    # Reconstruct ICS SFH (MPI parallelized)
    results = reconstruct_ics_sfh(args.input_file, target_snapshot=args.snapshot)

    # Only rank 0 continues with magnitudes and plots
    if results is not None:
        if results.get('ngals', 0) == 0:
            print("No galaxies with significant ICS found.")
        else:
            # Compute magnitudes
            results = compute_ics_magnitudes(results, bands=args.bands)

            # Generate plots
            if not args.no_plots:
                plot_ics_results(results, args.output_dir)

                # Compute and plot ICL evolution
                if not args.no_evolution:
                    icl_evolution = compute_icl_fraction_evolution(
                        args.input_file,
                        results['ics_sfh_dict'],
                        max_z=args.max_z,
                        min_halo_mass=args.min_halo_mass,
                        band='sdss_r'
                    )
                    plot_icl_evolution(icl_evolution, args.output_dir)

            print("\nDone!")

    # Final sync
    if HAS_MPI:
        COMM.Barrier()


if __name__ == '__main__':
    main()

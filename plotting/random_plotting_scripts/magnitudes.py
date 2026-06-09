#!/usr/bin/env python
"""
SAGE26 Star Formation History to Magnitudes using M/L ratios

Converts SAGE26 stellar masses to absolute magnitudes using empirical
mass-to-light ratios based on stellar population age and metallicity.

Based on fits to Bruzual & Charlot (2003) SSP models with Chabrier IMF.

Requires: numpy, h5py, matplotlib, astropy
Optional: mpi4py (for parallel processing of large catalogs)

Usage:
    python magnitudes.py path/to/model_0.hdf5
    python magnitudes.py path/to/model_0.hdf5 --snapshot 63
    mpirun -np 8 python magnitudes.py path/to/model_0.hdf5  # with MPI
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

# sSFR cut to divide quiescent from star-forming (log10 sSFR in yr^-1)
sSFRcut = -11.0

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
# Mass-to-Light Ratio Functions
# =============================================================================

def mass_to_light_ratio(age_gyr, Z, band='sdss_r'):
    """
    Calculate mass-to-light ratio M/L for a stellar population.

    Based on fits to Bruzual & Charlot (2003) SSP models with Chabrier IMF.

    Parameters:
        age_gyr: mass-weighted stellar age in Gyr
        Z: stellar metallicity (mass fraction, solar ~ 0.02)
        band: photometric band

    Returns:
        M/L in solar units (M_sun / L_sun)
    """
    # Ensure valid inputs
    age_gyr = np.maximum(age_gyr, 0.001)  # Minimum 1 Myr
    Z = np.clip(Z, 1e-4, 0.05)  # Valid metallicity range

    # Normalize metallicity to solar (Z_sun = 0.02)
    Z_solar = Z / 0.02

    # Coefficients for log10(M/L) = a + b*log10(age) + c*log10(Z/Z_sun)
    # Derived from BC03 SSP models (Chabrier IMF), anchored at 1 Gyr and 10 Gyr
    # Reference: Bruzual & Charlot (2003), Bell et al. (2003)
    #
    # From BC03 at solar metallicity:
    #   Age (Gyr) | M/L_g | M/L_r | M/L_i | M/L_K
    #   1         | 0.8   | 0.6   | 0.5   | 0.2
    #   10        | 4.5   | 3.0   | 2.2   | 0.6
    #
    coefficients = {
        'sdss_u': {'a': 0.0,  'b': 1.0,  'c': 0.15},  # UV fades fastest
        'sdss_g': {'a': -0.10, 'b': 0.75, 'c': 0.10},  # 1 Gyr: M/L~0.8, 10 Gyr: M/L~4.5
        'sdss_r': {'a': -0.22, 'b': 0.70, 'c': 0.08},  # 1 Gyr: M/L~0.6, 10 Gyr: M/L~3.0
        'sdss_i': {'a': -0.30, 'b': 0.65, 'c': 0.06},  # 1 Gyr: M/L~0.5, 10 Gyr: M/L~2.2
        'sdss_z': {'a': -0.35, 'b': 0.60, 'c': 0.05},
        'B': {'a': -0.05, 'b': 0.80, 'c': 0.12},
        'V': {'a': -0.15, 'b': 0.72, 'c': 0.08},
        'K': {'a': -0.70, 'b': 0.45, 'c': 0.03},       # NIR is more stable
    }

    if band not in coefficients:
        print(f"Warning: Unknown band '{band}', using sdss_r")
        band = 'sdss_r'

    c = coefficients[band]
    log_age = np.log10(age_gyr)
    log_Z = np.log10(Z_solar)

    log_ML = c['a'] + c['b'] * log_age + c['c'] * log_Z

    # Clip to reasonable range (0.1 to 20 M_sun/L_sun)
    log_ML = np.clip(log_ML, -1, 1.3)

    return 10**log_ML


def stellar_mass_to_magnitude(stellar_mass, age_gyr, Z, band='sdss_r'):
    """
    Convert stellar mass to absolute magnitude.

    Parameters:
        stellar_mass: stellar mass in M_sun
        age_gyr: mass-weighted stellar age in Gyr
        Z: stellar metallicity (mass fraction)
        band: photometric band

    Returns:
        Absolute magnitude in the specified band
    """
    ML = mass_to_light_ratio(age_gyr, Z, band)
    luminosity = stellar_mass / ML  # in L_sun

    # Handle zero/negative luminosity
    luminosity = np.maximum(luminosity, 1e-10)

    M_sun_band = M_SUN.get(band, 4.65)
    magnitude = M_sun_band - 2.5 * np.log10(luminosity)

    return magnitude


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
        params['LastSnapshotNr'] = int(sim.attrs['LastSnapshotNr'])

        runtime = f['Header/Runtime']
        params['VolumeFraction'] = float(runtime.attrs['frac_volume_processed'])

        params['snapshot_redshifts'] = np.array(f['Header/snapshot_redshifts'])
        params['output_snapshots'] = np.array(f['Header/output_snapshots'])

        # Find available snapshots
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


def compute_stellar_ages(sfh_mass, redshifts, H0, Om0):
    """
    Compute mass-weighted and light-weighted stellar ages from star formation history.

    Returns three ages:
    - Mass-weighted: appropriate for total stellar mass, NIR bands
    - Light-weighted (r-band): moderate weighting, for red optical bands
    - Light-weighted (g-band): aggressive weighting, for blue optical bands

    Parameters:
        sfh_mass: cumulative stellar mass at each snapshot [M_sun]
                  shape: [num_snapshots]
        redshifts: array of redshifts for each snapshot
        H0: Hubble parameter (dimensionless h)
        Om0: matter density parameter

    Returns:
        (mass_weighted_age, light_age_r, light_age_g) in Gyr
    """
    # Get cosmic times for each snapshot
    cosmo = FlatLambdaCDM(H0=H0 * 100, Om0=Om0)
    cosmic_times = cosmo.age(redshifts).to(u.Gyr).value  # Age of universe at each z

    # Current time (z=0)
    t_now = cosmic_times[np.argmin(np.abs(redshifts))]

    # Calculate mass formed in each bin by differencing cumulative mass
    mass_formed = np.zeros_like(sfh_mass)
    mass_formed[1:] = np.maximum(sfh_mass[1:] - sfh_mass[:-1], 0)
    mass_formed[0] = sfh_mass[0]  # First bin

    # Age of stars formed at each snapshot = t_now - t_formation
    stellar_ages = t_now - cosmic_times
    stellar_ages = np.maximum(stellar_ages, 0.001)  # Minimum age 1 Myr

    total_mass = np.sum(mass_formed)
    if total_mass <= 0:
        return 1.0, 1.0, 1.0  # Default to 1 Gyr if no mass

    # Mass-weighted mean age
    mass_weighted_age = np.sum(mass_formed * stellar_ages) / total_mass

    # Light-weighted ages with band-dependent fading rates
    # From BC03: L/M fading is steeper in blue bands
    # Empirically tuned to match observed color distributions:
    # r-band: L/M ~ age^-0.8 (moderate - red light persists)
    # g-band: L/M ~ age^-1.1 (aggressive - blue light fades fast)

    # r-band light-weighted age (moderate weighting)
    lum_weight_r = mass_formed * (stellar_ages ** -0.8)
    total_lum_r = np.sum(lum_weight_r)
    if total_lum_r > 0:
        light_age_r = np.sum(lum_weight_r * stellar_ages) / total_lum_r
    else:
        light_age_r = mass_weighted_age

    # g-band light-weighted age (aggressive weighting - blue light fades faster)
    # Using -1.1 exponent: 10 Gyr old stars contribute 10^-1.1 = 0.08x vs 1 Gyr stars
    lum_weight_g = mass_formed * (stellar_ages ** -1.1)
    total_lum_g = np.sum(lum_weight_g)
    if total_lum_g > 0:
        light_age_g = np.sum(lum_weight_g * stellar_ages) / total_lum_g
    else:
        light_age_g = mass_weighted_age

    # Ensure physical: light ages should be <= mass age
    light_age_r = np.clip(light_age_r, 0.01, mass_weighted_age)
    light_age_g = np.clip(light_age_g, 0.01, light_age_r)  # g should be youngest

    return mass_weighted_age, light_age_r, light_age_g


# =============================================================================
# Main Processing
# =============================================================================

def process_galaxies(filepath, snapshot=None, bands=['sdss_g', 'sdss_r']):
    """
    Process SAGE26 output and compute magnitudes.

    Parameters:
        filepath: path to HDF5 file
        snapshot: snapshot number to process (default: latest)
        bands: list of photometric bands

    Returns:
        dict with magnitudes, stellar masses, colors, etc.
        Returns None on non-root ranks when using MPI.
    """
    t_start = time.time()

    # Read simulation parameters (all ranks read)
    params = read_simulation_params(filepath)
    Hubble_h = params['Hubble_h']
    BoxSize = params['BoxSize']
    Omega = params['Omega']
    VolumeFraction = params['VolumeFraction']
    redshifts = params['snapshot_redshifts']

    # Determine snapshot to use
    if snapshot is None:
        snapshot = params['latest_snapshot']
    snap_str = f'Snap_{snapshot}'

    if RANK == 0:
        print("=" * 60)
        print("SAGE26 Magnitude Calculator")
        print("=" * 60)
        print(f"\nInput file: {filepath}")
        print(f"Snapshot: {snapshot} (z = {redshifts[snapshot]:.4f})")
        print(f"\nCosmology:")
        print(f"  Hubble h     = {Hubble_h}")
        print(f"  Omega_m      = {Omega}")
        print(f"  Omega_Lambda = {1 - Omega:.2f}")
        print(f"  Box size     = {BoxSize} Mpc/h")
        print(f"  Volume frac  = {VolumeFraction:.4f}")
        print(f"\nBands: {', '.join(bands)}")
        if HAS_MPI:
            print(f"MPI: {SIZE} ranks")
        print()

    # Read galaxy properties (all ranks read)
    t_read = time.time()
    StellarMass = read_hdf(filepath, snap_str, 'StellarMass') * 1e10 / Hubble_h
    MetalsStellarMass = read_hdf(filepath, snap_str, 'MetalsStellarMass') * 1e10 / Hubble_h
    SfrDisk = read_hdf(filepath, snap_str, 'SfrDisk')
    SfrBulge = read_hdf(filepath, snap_str, 'SfrBulge')
    Type = read_hdf(filepath, snap_str, 'Type')

    # Read SFH arrays (2D: [ngalaxies, num_snapshots])
    SFHMassDisk = read_hdf(filepath, snap_str, 'SFHMassDisk') * 1e10 / Hubble_h
    SFHMassBulge = read_hdf(filepath, snap_str, 'SFHMassBulge') * 1e10 / Hubble_h

    has_sfh = SFHMassDisk.size > 0 and SFHMassBulge.size > 0

    if RANK == 0:
        print(f"Data read in {time.time() - t_read:.2f}s")
        print(f"Total galaxies in file: {len(StellarMass):,}")
        if has_sfh:
            print(f"SFH data: {SFHMassDisk.shape[1]} snapshots per galaxy")
        else:
            print("No SFH data - using sSFR-based age estimate")

    # Select galaxies with valid data
    valid = (StellarMass > 1e6)
    w = np.where(valid)[0]
    ngals = len(w)

    if RANK == 0:
        print(f"Galaxies with M* > 10^6 M_sun: {ngals:,} ({100*ngals/len(StellarMass):.1f}%)")
        print()

    # Calculate metallicity (fast, do on all ranks)
    Z = MetalsStellarMass[w] / (StellarMass[w] + 1e-10)
    Z = np.clip(Z, 1e-4, 0.05)

    # Calculate mass-weighted age for each galaxy
    t_age = time.time()
    if has_sfh:
        if RANK == 0:
            print("Computing mass-weighted ages from SFH...")

        # Distribute work across MPI ranks
        gals_per_rank = ngals // SIZE
        remainder = ngals % SIZE

        # Calculate start and end indices for this rank
        if RANK < remainder:
            local_start = RANK * (gals_per_rank + 1)
            local_end = local_start + gals_per_rank + 1
        else:
            local_start = RANK * gals_per_rank + remainder
            local_end = local_start + gals_per_rank

        local_ngals = local_end - local_start

        if RANK == 0 and HAS_MPI:
            print(f"  Distributing {ngals:,} galaxies across {SIZE} ranks (~{local_ngals:,} per rank)")

        # Process local subset - compute mass-weighted and band-specific light-weighted ages
        t_loop = time.time()
        local_mass_ages = np.zeros(local_ngals)
        local_light_ages_r = np.zeros(local_ngals)
        local_light_ages_g = np.zeros(local_ngals)

        for i in range(local_ngals):
            global_i = local_start + i
            gal_idx = w[global_i]

            if RANK == 0 and local_ngals > 1000 and i % (local_ngals // 10) == 0 and i > 0:
                elapsed = time.time() - t_loop
                rate = i / elapsed
                eta = (local_ngals - i) / rate if rate > 0 else 0
                print(f"  Progress: {100*i/local_ngals:.0f}% ({i:,}/{local_ngals:,}) - "
                      f"{rate:.0f} gal/s - ETA: {eta:.0f}s")

            sfh_total = SFHMassDisk[gal_idx, :] + SFHMassBulge[gal_idx, :]
            mass_age, light_age_r, light_age_g = compute_stellar_ages(sfh_total, redshifts, Hubble_h, Omega)
            local_mass_ages[i] = mass_age
            local_light_ages_r[i] = light_age_r
            local_light_ages_g[i] = light_age_g

        # Gather ages from all ranks
        if HAS_MPI:
            sendcounts = np.array(COMM.allgather(local_ngals))
            displacements = np.zeros(SIZE, dtype=int)
            displacements[1:] = np.cumsum(sendcounts[:-1])

            mass_ages = np.zeros(ngals) if RANK == 0 else None
            light_ages_r = np.zeros(ngals) if RANK == 0 else None
            light_ages_g = np.zeros(ngals) if RANK == 0 else None
            COMM.Gatherv(local_mass_ages, [mass_ages, sendcounts, displacements, MPI.DOUBLE], root=0)
            COMM.Gatherv(local_light_ages_r, [light_ages_r, sendcounts, displacements, MPI.DOUBLE], root=0)
            COMM.Gatherv(local_light_ages_g, [light_ages_g, sendcounts, displacements, MPI.DOUBLE], root=0)
        else:
            mass_ages = local_mass_ages
            light_ages_r = local_light_ages_r
            light_ages_g = local_light_ages_g

    else:
        # Estimate age from sSFR if no SFH available
        if RANK == 0:
            print("Estimating ages from sSFR...")
            total_sfr = SfrDisk[w] + SfrBulge[w]
            ssfr_temp = total_sfr / (StellarMass[w] + 1e-10)

            # Mass-weighted age estimate from sSFR
            mass_ages = np.where(ssfr_temp > 1e-12,
                                 np.clip(1.0 / (ssfr_temp * 1e9), 0.1, 13.0),
                                 10.0)
            # Light-weighted ages are younger for star-forming galaxies
            # r-band: moderate correction
            light_ages_r = np.where(ssfr_temp > 1e-11,
                                    np.clip(mass_ages * 0.5, 0.1, 8.0),
                                    mass_ages * 0.8)
            # g-band: stronger correction (blue light from young stars)
            light_ages_g = np.where(ssfr_temp > 1e-11,
                                    np.clip(mass_ages * 0.2, 0.05, 3.0),
                                    mass_ages * 0.6)
        else:
            mass_ages = None
            light_ages_r = None
            light_ages_g = None

    if RANK == 0:
        print(f"Age calculation completed in {time.time() - t_age:.2f}s")

    # Only rank 0 continues with magnitude calculation and output
    if RANK != 0:
        return None

    # Compute magnitudes for each band
    # Use band-appropriate light-weighted ages:
    # - Blue bands (u, g, B): use light_ages_g (youngest, most aggressive weighting)
    # - Red optical (r, i, V): use light_ages_r (moderate weighting)
    # - NIR (z, K): use mass_ages (traces stellar mass)
    print("\nComputing magnitudes...")
    t_mag = time.time()
    magnitudes = {}

    # Band-to-age mapping
    blue_bands = {'sdss_u', 'sdss_g', 'B'}
    red_optical_bands = {'sdss_r', 'sdss_i', 'V'}

    for band in bands:
        if band in blue_bands:
            mags = stellar_mass_to_magnitude(StellarMass[w], light_ages_g, Z, band)
        elif band in red_optical_bands:
            mags = stellar_mass_to_magnitude(StellarMass[w], light_ages_r, Z, band)
        else:
            mags = stellar_mass_to_magnitude(StellarMass[w], mass_ages, Z, band)
        magnitudes[band] = mags

    print(f"Magnitude calculation completed in {time.time() - t_mag:.2f}s")

    # Calculate sSFR
    total_sfr = SfrDisk[w] + SfrBulge[w]
    ssfr = np.log10(total_sfr / (StellarMass[w] + 1e-10) + 1e-14)

    # Calculate volume
    volume = (BoxSize / Hubble_h) ** 3 * VolumeFraction

    # Print detailed statistics
    print("\n" + "=" * 60)
    print("GALAXY POPULATION STATISTICS")
    print("=" * 60)

    # Mass statistics
    log_mass = np.log10(StellarMass[w])
    print(f"\nStellar Mass [log10 M_sun]:")
    print(f"  Min / Max     : {np.min(log_mass):.2f} / {np.max(log_mass):.2f}")
    print(f"  Mean / Median : {np.mean(log_mass):.2f} / {np.median(log_mass):.2f}")
    print(f"  16-84 %ile    : {np.percentile(log_mass, 16):.2f} - {np.percentile(log_mass, 84):.2f}")

    # Mass bins
    print(f"\n  Mass bins:")
    mass_bins = [(6, 8), (8, 9), (9, 10), (10, 11), (11, 13)]
    for lo, hi in mass_bins:
        n = np.sum((log_mass >= lo) & (log_mass < hi))
        print(f"    10^{lo}-10^{hi} M_sun: {n:>10,} ({100*n/ngals:>5.1f}%)")

    # Age statistics
    print(f"\nMass-weighted Age [Gyr]:")
    print(f"  Min / Max     : {np.min(mass_ages):.2f} / {np.max(mass_ages):.2f}")
    print(f"  Mean / Median : {np.mean(mass_ages):.2f} / {np.median(mass_ages):.2f}")
    print(f"  16-84 %ile    : {np.percentile(mass_ages, 16):.2f} - {np.percentile(mass_ages, 84):.2f}")

    print(f"\nLight-weighted Age (r-band) [Gyr]:")
    print(f"  Min / Max     : {np.min(light_ages_r):.2f} / {np.max(light_ages_r):.2f}")
    print(f"  Mean / Median : {np.mean(light_ages_r):.2f} / {np.median(light_ages_r):.2f}")
    print(f"  16-84 %ile    : {np.percentile(light_ages_r, 16):.2f} - {np.percentile(light_ages_r, 84):.2f}")

    print(f"\nLight-weighted Age (g-band) [Gyr]:")
    print(f"  Min / Max     : {np.min(light_ages_g):.2f} / {np.max(light_ages_g):.2f}")
    print(f"  Mean / Median : {np.mean(light_ages_g):.2f} / {np.median(light_ages_g):.2f}")
    print(f"  16-84 %ile    : {np.percentile(light_ages_g, 16):.2f} - {np.percentile(light_ages_g, 84):.2f}")

    # Metallicity statistics
    log_Z = np.log10(Z / 0.02)  # Relative to solar
    print(f"\nMetallicity [log10 Z/Z_sun]:")
    print(f"  Min / Max     : {np.min(log_Z):.2f} / {np.max(log_Z):.2f}")
    print(f"  Mean / Median : {np.mean(log_Z):.2f} / {np.median(log_Z):.2f}")
    print(f"  16-84 %ile    : {np.percentile(log_Z, 16):.2f} - {np.percentile(log_Z, 84):.2f}")

    # sSFR and quiescent fraction
    quiescent = ssfr < sSFRcut
    n_q = np.sum(quiescent)
    n_sf = ngals - n_q
    print(f"\nStar Formation:")
    print(f"  sSFR cut      : 10^{sSFRcut} yr^-1")
    print(f"  Star-forming  : {n_sf:,} ({100*n_sf/ngals:.1f}%)")
    print(f"  Quiescent     : {n_q:,} ({100*n_q/ngals:.1f}%)")
    print(f"  sSFR [log10 yr^-1]:")
    print(f"    Mean / Median : {np.mean(ssfr):.2f} / {np.median(ssfr):.2f}")

    # Galaxy type
    n_central = np.sum(Type[w] == 0)
    n_satellite = np.sum(Type[w] == 1)
    n_orphan = np.sum(Type[w] == 2)
    print(f"\nGalaxy Type:")
    print(f"  Centrals      : {n_central:,} ({100*n_central/ngals:.1f}%)")
    print(f"  Satellites    : {n_satellite:,} ({100*n_satellite/ngals:.1f}%)")
    print(f"  Orphans       : {n_orphan:,} ({100*n_orphan/ngals:.1f}%)")

    # Magnitude statistics
    print(f"\nMagnitudes:")
    for band in bands:
        mags = magnitudes[band]
        valid_mags = mags[np.isfinite(mags) & (mags < 0) & (mags > -30)]
        print(f"  {band}:")
        print(f"    Valid       : {len(valid_mags):,} ({100*len(valid_mags)/ngals:.1f}%)")
        if len(valid_mags) > 0:
            print(f"    Min / Max   : {np.min(valid_mags):.2f} / {np.max(valid_mags):.2f}")
            print(f"    Median      : {np.median(valid_mags):.2f}")
            print(f"    16-84 %ile  : {np.percentile(valid_mags, 16):.2f} - {np.percentile(valid_mags, 84):.2f}")

    # Color statistics (if g and r available)
    if 'sdss_g' in magnitudes and 'sdss_r' in magnitudes:
        color = magnitudes['sdss_g'] - magnitudes['sdss_r']
        valid_color = color[np.isfinite(color) & (color > -1) & (color < 2)]
        print(f"\nColor (g - r):")
        print(f"  Valid         : {len(valid_color):,}")
        if len(valid_color) > 0:
            print(f"  Min / Max     : {np.min(valid_color):.3f} / {np.max(valid_color):.3f}")
            print(f"  Mean / Median : {np.mean(valid_color):.3f} / {np.median(valid_color):.3f}")
            print(f"  16-84 %ile    : {np.percentile(valid_color, 16):.3f} - {np.percentile(valid_color, 84):.3f}")

            # Color by population
            q_color = color[quiescent & np.isfinite(color)]
            sf_color = color[~quiescent & np.isfinite(color)]
            if len(q_color) > 0:
                print(f"  Quiescent     : {np.median(q_color):.3f} (median)")
            if len(sf_color) > 0:
                print(f"  Star-forming  : {np.median(sf_color):.3f} (median)")

    # M/L ratio statistics
    if 'sdss_r' in magnitudes:
        r_mags = magnitudes['sdss_r']
        valid = np.isfinite(r_mags) & (r_mags < 0) & (r_mags > -30)
        ML_r = StellarMass[w][valid] / (10 ** ((M_SUN['sdss_r'] - r_mags[valid]) / 2.5))
        log_ML = np.log10(ML_r)
        print(f"\nMass-to-Light Ratio (r-band) [M_sun/L_sun]:")
        print(f"  Min / Max     : {np.min(ML_r):.2f} / {np.max(ML_r):.2f}")
        print(f"  Mean / Median : {np.mean(ML_r):.2f} / {np.median(ML_r):.2f}")
        print(f"  16-84 %ile    : {np.percentile(ML_r, 16):.2f} - {np.percentile(ML_r, 84):.2f}")

    # Volume and number density
    print(f"\nVolume and Density:")
    print(f"  Volume        : {volume:.2e} Mpc^3")
    print(f"  Galaxy density: {ngals/volume:.2e} Mpc^-3")

    # Timing summary
    t_total = time.time() - t_start
    print(f"\n" + "=" * 60)
    print(f"Total processing time: {t_total:.2f}s")
    print(f"Processing rate: {ngals/t_total:.0f} galaxies/s")
    print("=" * 60)

    return {
        'magnitudes': magnitudes,
        'bands': bands,
        'stellar_mass': StellarMass[w],
        'ssfr': ssfr,
        'mass_ages': mass_ages,
        'light_ages_r': light_ages_r,
        'light_ages_g': light_ages_g,
        'metallicity': Z,
        'type': Type[w],
        'volume': volume,
        'Hubble_h': Hubble_h,
        'snapshot': snapshot,
        'redshift': redshifts[snapshot],
        'ngals': ngals,
        'n_quiescent': n_q,
        'n_starforming': n_sf
    }


def save_results(results, output_dir, filename='magnitudes.hdf5'):
    """Save computed magnitudes and properties to HDF5 file."""

    if results is None:
        return

    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, filename)

    print(f"\nSaving results to {outpath}")

    with h5.File(outpath, 'w') as f:
        # Metadata
        f.attrs['snapshot'] = results['snapshot']
        f.attrs['redshift'] = results['redshift']
        f.attrs['Hubble_h'] = results['Hubble_h']
        f.attrs['volume_Mpc3'] = results['volume']
        f.attrs['ngals'] = results['ngals']
        f.attrs['n_quiescent'] = results['n_quiescent']
        f.attrs['n_starforming'] = results['n_starforming']
        f.attrs['sSFRcut'] = sSFRcut

        # Galaxy properties
        f.create_dataset('stellar_mass', data=results['stellar_mass'])
        f.create_dataset('mass_weighted_age', data=results['mass_ages'])
        f.create_dataset('light_weighted_age_r', data=results['light_ages_r'])
        f.create_dataset('light_weighted_age_g', data=results['light_ages_g'])
        f.create_dataset('metallicity', data=results['metallicity'])
        f.create_dataset('ssfr', data=results['ssfr'])
        f.create_dataset('type', data=results['type'])

        # Magnitudes
        mag_grp = f.create_group('magnitudes')
        for band, mags in results['magnitudes'].items():
            mag_grp.create_dataset(band, data=mags)

        # Add band list
        f.attrs['bands'] = ','.join(results['bands'])

    print(f"Saved {results['ngals']:,} galaxies with {len(results['bands'])} bands")


# Path to Driver+2012 GAMA LF data files
DRIVER12_LF_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'shark', 'data', 'lf'
)

# Mapping from band name to Driver+2012 data file
DRIVER12_LF_FILES = {
    'sdss_u': 'lfu_z0_driver12.data',
    'sdss_g': 'lfg_z0_driver12.data',
    'sdss_r': 'lfr_z0_driver12.data',
    'sdss_i': 'lfi_z0_driver12.data',
    'sdss_z': 'lfz_z0_driver12.data',
    'K': 'lfk_z0_driver12.data',
}


def _load_driver12_lf(band, Hubble_h):
    """
    Load Driver+2012 GAMA LF data for a given band.

    Data file format:
      col 1: M_AB - 5*log10(h)  (bin centre)
      col 2: phi in (Mpc/h)^-3 (0.5 mag)^-1
      col 3: 1-sigma error on phi
      col 4: number of galaxies

    Returns (mag, phi, phi_err) converted to physical units (Mpc^-3 mag^-1),
    or (None, None, None) if file not found.
    """
    if band not in DRIVER12_LF_FILES:
        return None, None, None

    filepath = os.path.join(DRIVER12_LF_DIR, DRIVER12_LF_FILES[band])
    if not os.path.exists(filepath):
        print(f"Warning: Driver+2012 data not found at {filepath}")
        return None, None, None

    data = np.loadtxt(filepath)
    mag_h = data[:, 0]       # M_AB - 5*log10(h)
    phi_h = data[:, 1]       # (Mpc/h)^-3 (0.5 mag)^-1
    phi_err_h = data[:, 2]   # error in same units

    # Filter out empty bins
    nonzero = phi_h > 0
    mag_h = mag_h[nonzero]
    phi_h = phi_h[nonzero]
    phi_err_h = phi_err_h[nonzero]

    # Convert magnitudes: M_AB = M_h + 5*log10(h)
    mag_phys = mag_h + 5.0 * np.log10(Hubble_h)

    # Convert densities: phi_phys = phi_h * h^3 (from (Mpc/h)^-3 to Mpc^-3)
    # and from per 0.5 mag to per mag: divide by 0.5 (i.e. multiply by 2)
    phi_phys = phi_h * (Hubble_h ** 3) / 0.5
    phi_err_phys = phi_err_h * (Hubble_h ** 3) / 0.5

    return mag_phys, phi_phys, phi_err_phys


def _plot_driver12_lf(band, Hubble_h, plt_module):
    """Plot Driver+2012 GAMA LF data points with error bars for a given band."""
    mag, phi, phi_err = _load_driver12_lf(band, Hubble_h)
    if mag is None:
        return

    log_phi = np.log10(phi)
    # Asymmetric errors in log space
    log_phi_upper = np.log10(phi + phi_err) - log_phi
    log_phi_lower = log_phi - np.log10(np.maximum(phi - phi_err, 1e-10))

    plt_module.errorbar(mag, log_phi,
                        yerr=[log_phi_lower, log_phi_upper],
                        fmt='s', color='blue', markersize=5, capsize=2,
                        label='GAMA (Driver+2012)')


def plot_results(results, output_dir):
    """Generate diagnostic plots."""

    if results is None:
        return

    magnitudes = results['magnitudes']
    masses = results['stellar_mass']
    ssfrs = results['ssfr']
    light_ages_g = results['light_ages_g']
    volume = results['volume']
    Hubble_h = results['Hubble_h']

    # Check we have g and r bands
    if 'sdss_g' not in magnitudes or 'sdss_r' not in magnitudes:
        print("Warning: Need sdss_g and sdss_r for color plots")
        return

    g_mags = magnitudes['sdss_g']
    r_mags = magnitudes['sdss_r']
    color_gr = g_mags - r_mags

    # Filter valid magnitudes
    valid = np.isfinite(r_mags) & np.isfinite(g_mags) & (r_mags < 0) & (r_mags > -30)
    g_mags = g_mags[valid]
    r_mags = r_mags[valid]
    color_gr = color_gr[valid]
    masses = masses[valid]
    ssfrs = ssfrs[valid]
    light_ages_g = light_ages_g[valid]

    if len(r_mags) == 0:
        print("No valid magnitudes to plot")
        return

    # Separate populations
    quiescent = ssfrs < sSFRcut
    starforming = ssfrs >= sSFRcut

    print(f"\nPlotting {len(r_mags)} galaxies")
    print(f"Quiescent: {np.sum(quiescent)}, Star-Forming: {np.sum(starforming)}")
    print(f"Light-weighted age (g-band) range: {np.min(light_ages_g):.2f} - {np.max(light_ages_g):.2f} Gyr")
    print(f"Color range: {np.min(color_gr):.2f} - {np.max(color_gr):.2f}")

    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Color-Magnitude Diagram
    plt.figure(figsize=(8, 6))
    plt.scatter(r_mags[starforming], color_gr[starforming],
                c='dodgerblue', alpha=0.4, label='Star-Forming', edgecolors='none', s=15)
    plt.scatter(r_mags[quiescent], color_gr[quiescent],
                c='crimson', alpha=0.4, label='Quiescent', edgecolors='none', s=15)

    plt.xlabel('Absolute r-band Magnitude ($M_r$)')
    plt.ylabel('Color ($g - r$)')
    plt.title('Color-Magnitude Diagram')
    plt.xlim(-14, -24)
    plt.ylim(-0.2, 1.2)
    plt.gca().invert_xaxis()
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'CMD_g_vs_gr{OutputFormat}'))
    plt.close()

    # Plot 2: Color vs Stellar Mass
    plt.figure(figsize=(8, 6))
    plt.scatter(np.log10(masses[starforming]), color_gr[starforming],
                c='dodgerblue', alpha=0.4, label='Star-Forming', edgecolors='none', s=15)
    plt.scatter(np.log10(masses[quiescent]), color_gr[quiescent],
                c='crimson', alpha=0.4, label='Quiescent', edgecolors='none', s=15)

    plt.xlabel(r'$\log_{10}$ Stellar Mass [$M_{\odot}$]')
    plt.ylabel('Color ($g - r$)')
    plt.title('Color vs. Stellar Mass')
    plt.xlim(6, 12)
    plt.ylim(-0.2, 1.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'Color_vs_Mass{OutputFormat}'))
    plt.close()

    # Plot 3: Color vs Age (diagnostic)
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(np.log10(light_ages_g), color_gr, c=np.log10(masses),
                     cmap='viridis', alpha=0.4, edgecolors='none', s=15)
    plt.colorbar(sc, label=r'$\log_{10}$ Stellar Mass [$M_{\odot}$]')
    plt.xlabel('log$_{10}$ Light-weighted Age (g-band) [Gyr]')
    plt.ylabel('Color ($g - r$)')
    plt.title('Color vs. Stellar Age (g-band Light-weighted)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'Color_vs_Age{OutputFormat}'))
    plt.close()

    # Plot 4: r-band Luminosity Function
    plt.figure(figsize=(8, 6))

    dM = 0.5
    mag_bins = np.arange(-25.0, -14.0, dM)

    counts, bin_edges = np.histogram(r_mags, bins=mag_bins)
    phi = counts / (volume * dM)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    valid_bins = phi > 0
    plt.plot(bin_centers[valid_bins], np.log10(phi[valid_bins]),
             marker='o', linestyle='-', color='black', linewidth=2, label='SAGE26')

    # Overlay Blanton+2003 SDSS Schechter function
    # Convert from h^3 Mpc^-3 to physical Mpc^-3 to match SAGE units
    M_star = -20.44 + 5 * np.log10(Hubble_h)  # h-corrected M*
    phi_star = 0.0149 * (Hubble_h ** 3)  # Convert h^3 Mpc^-3 to Mpc^-3
    alpha = -1.05

    M_plot = np.linspace(-15, -24, 100)
    M_diff = 0.4 * (M_star - M_plot)
    schechter_phi = 0.4 * np.log(10) * phi_star * (10 ** M_diff) ** (alpha + 1) * np.exp(-10 ** M_diff)

    plt.plot(M_plot, np.log10(schechter_phi),
             linestyle='--', color='red', linewidth=2, label='SDSS (Blanton+2003)')

    # Overlay Driver+2012 GAMA r-band LF data points
    _plot_driver12_lf('sdss_r', Hubble_h, plt)

    plt.xlabel('Absolute r-band Magnitude ($M_r$)')
    plt.ylabel(r'$\log_{10}\ \Phi\ [\mathrm{Mpc}^{-3}\ \mathrm{mag}^{-1}]$')
    plt.title('$r$-band Luminosity Function')
    plt.gca().invert_xaxis()
    plt.xlim(-15, -24)
    plt.ylim(-6, -1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'LuminosityFunction_r{OutputFormat}'))
    plt.close()

    # Plot 5: M/L ratio distribution
    ML_r = masses / (10 ** ((M_SUN['sdss_r'] - r_mags) / 2.5))

    plt.figure(figsize=(8, 6))
    plt.hist(np.log10(ML_r[starforming]), bins=50, alpha=0.6,
             color='dodgerblue', label='Star-Forming', density=True)
    plt.hist(np.log10(ML_r[quiescent]), bins=50, alpha=0.6,
             color='crimson', label='Quiescent', density=True)
    plt.xlabel(r'$\log_{10}$ M/L$_r$ [$M_{\odot}/L_{\odot}$]')
    plt.ylabel('Density')
    plt.title('r-band Mass-to-Light Ratio Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'ML_ratio_distribution{OutputFormat}'))
    plt.close()

    print(f"\nPlots saved to {output_dir}:")
    print(f"  - CMD_g_vs_gr{OutputFormat}")
    print(f"  - Color_vs_Mass{OutputFormat}")
    print(f"  - Color_vs_Age{OutputFormat}")
    print(f"  - LuminosityFunction_r{OutputFormat}")
    print(f"  - ML_ratio_distribution{OutputFormat}")


# =============================================================================
# Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Compute magnitudes from SAGE26 using M/L ratios',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python magnitudes.py output/millennium/model_0.hdf5
    python magnitudes.py output/millennium/model_0.hdf5 --snapshot 63
    python magnitudes.py output/millennium/model_0.hdf5 --bands sdss_u sdss_g sdss_r sdss_i

    # With MPI for large catalogs (recommended for >100k galaxies):
    mpirun -np 8 python magnitudes.py output/millennium/model_0.hdf5
        """
    )

    parser.add_argument('input_file', help='Path to SAGE26 HDF5 output file')
    parser.add_argument('-s', '--snapshot', type=int, default=None,
                        help='Snapshot number (default: latest)')
    parser.add_argument('-o', '--output-dir', type=str, default=None,
                        help='Output directory for plots and results')
    parser.add_argument('--bands', type=str, nargs='+',
                        default=['sdss_g', 'sdss_r'],
                        help='Photometric bands (default: sdss_g sdss_r)')
    parser.add_argument('--save', action='store_true',
                        help='Save results to HDF5 file')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots')

    args = parser.parse_args()

    if RANK == 0 and not os.path.exists(args.input_file):
        print(f"Error: File not found: {args.input_file}")
        sys.exit(1)

    # Synchronize before continuing (in case rank 0 exits)
    if HAS_MPI:
        COMM.Barrier()

    # Determine output directory
    if args.output_dir is None:
        input_dir = os.path.dirname(args.input_file)
        args.output_dir = os.path.join(input_dir, 'plots')

    # Process galaxies
    results = process_galaxies(
        args.input_file,
        snapshot=args.snapshot,
        bands=args.bands
    )

    # Save and plot results (only on rank 0)
    if results is not None:
        if args.save:
            save_results(results, args.output_dir)

        if not args.no_plots:
            plot_results(results, args.output_dir)

    # Final sync
    if HAS_MPI:
        COMM.Barrier()


if __name__ == '__main__':
    main()

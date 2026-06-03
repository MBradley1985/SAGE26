#!/usr/bin/env python

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
from scipy import interpolate
from scipy import stats
from scipy.integrate import quad
from scipy.ndimage import gaussian_filter
from random import sample, seed
import matplotlib.cm as cm
import pandas as pd

import warnings
warnings.filterwarnings("ignore")
try:
    from astropy.table import Table
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False
    print("Warning: astropy not available, observational data will not be loaded")

try:
    from colossus.cosmology import cosmology as _colossus_cosmology
    from colossus.halo import concentration as _colossus_conc
    _colossus_cosmology.setCosmology('custom_microuchuu', flat=True,
                                     H0=67.74, Om0=0.3089, Ob0=0.0486,
                                     sigma8=0.8159, ns=0.9667, relspecies=False)
    _HAS_COLOSSUS = True
except Exception:
    _HAS_COLOSSUS = False



# ========================== CONFIGURATION ==========================

# File paths
PRIMARY_DIR    = './output/millennium/'
SECONDARY_DIR  = './output/millennium_exprheat/'  # optional; set to None to disable
TERTIARY_DIR   = './output/millennium_cgmoff/'
QUATERNARY_DIR =  None # optional; set to None to disable 
MODEL_FILE = 'model_0.hdf5'
OBS_DIR = './data/'

# Plotting (analysis choices — not simulation parameters)
OUTPUT_FORMAT = '.pdf'
DILUTE = 7500
SEED = 2222

# Analysis thresholds (not simulation parameters)
MIN_PARTICLES = 20     # minimum DM particles for a resolved halo (applied at load time)
MIN_MVIR_PARTICLES = 20  # independent halo-mass cut: require Mvir >= MIN_MVIR_PARTICLES * m_p
SSFR_CUT = -11.0       # log10(sSFR/yr^-1) dividing quiescent from star-forming

# Solar metallicity (Asplund et al. 2009)
Z_SUN = 0.0134

# ------------------------------------------------------------------
# Model display names, colours, and line styles — edit here to change
# labels globally across all plots.
# ------------------------------------------------------------------
PRIMARY_LABEL    = 'miniMillennium (Rvir capped r_heat)'
PRIMARY_COLOR    = 'steelblue'
PRIMARY_LS       = ':'

SECONDARY_LABEL  = 'miniMillennium (exponential r_heat)'
SECONDARY_COLOR  = 'coral'
SECONDARY_LS     = '--'

TERTIARY_LABEL   = 'miniMillennium (no CGM-AGN interaction)'
TERTIARY_COLOR   = 'mediumseagreen'
TERTIARY_LS      = '-'

QUATERNARY_LABEL = 'miniMillennium (vanilla)'
QUATERNARY_COLOR = 'mediumpurple'
QUATERNARY_LS    = '-.'

# Solar mass in grams (for MASS_CONVERT derivation)
_MSUN_CGS = 1.989e33


# --------------- HDF5 header reader ---------------

import glob as _glob_early


def _find_model_files_early(directory):
    """Minimal file discovery used during module init (before full I/O helpers)."""
    pattern = os.path.join(directory, 'model_*.hdf5')
    files = sorted(_glob_early.glob(pattern))
    if not files:
        single = os.path.join(directory, MODEL_FILE)
        if os.path.exists(single):
            files = [single]
    return files


def _read_sim_header(directory):
    """
    Read simulation parameters from the HDF5 header of the first model
    file found in *directory*.

    Returns a dict of parameters, or ``None`` if no model files exist.
    The ``volume_fraction`` key is the *total* fraction across all MPI
    files (summed ``frac_volume_processed``).
    """
    files = _find_model_files_early(directory)
    if not files:
        return None

    try:
        with h5.File(files[0], 'r') as f:
            sim = f['Header/Simulation']
            runtime = f['Header/Runtime']

            header = {
                'hubble_h':       float(sim.attrs['hubble_h']),
                'box_size':       float(sim.attrs['box_size']),
                'omega_matter':   float(sim.attrs['omega_matter']),
                'omega_lambda':   float(sim.attrs['omega_lambda']),
                'last_snap_nr':   int(sim.attrs['LastSnapshotNr']),
                'unit_mass_in_g': float(runtime.attrs['UnitMass_in_g']),
                'baryon_frac':    float(runtime.attrs.get('BaryonFrac', 0.17)),
                'redshifts':      list(f['Header/snapshot_redshifts'][:]),
                'output_snaps':   list(f['Header/output_snapshots'][:]),
                'particle_mass':  float(sim.attrs['particle_mass']),
            }

        # Sum frac_volume_processed across all MPI files to get the total
        total_fvp = 0.0
        for fp in files:
            with h5.File(fp, 'r') as f:
                total_fvp += float(f['Header/Runtime'].attrs['frac_volume_processed'])
        header['volume_fraction'] = total_fvp
    except Exception as e:
        print(f"Warning: could not read header from {directory}: {e}")
        return None

    return header


def _snap_for_z(redshifts, target_z):
    """
    Return the snapshot index of the last output snapshot whose redshift
    is >= *target_z*.  This reproduces the standard convention of choosing
    the snapshot just above the target redshift (e.g. z=4.179 for target 4).
    """
    neg_z = -np.array(redshifts)          # make increasing for searchsorted
    idx = int(np.searchsorted(neg_z, -target_z, side='right')) - 1
    return max(idx, 0)


# --------------- Primary simulation parameters (from HDF5) ---------------

_primary_hdr = _read_sim_header(PRIMARY_DIR)
if _primary_hdr is not None:
    HUBBLE_H         = _primary_hdr['hubble_h']
    BOX_SIZE         = _primary_hdr['box_size']
    VOLUME_FRACTION  = _primary_hdr['volume_fraction']
    VOLUME           = (BOX_SIZE / HUBBLE_H)**3 * VOLUME_FRACTION  # Mpc^3
    MASS_CONVERT     = _primary_hdr['unit_mass_in_g'] / _MSUN_CGS / HUBBLE_H
    OMEGA_M          = _primary_hdr['omega_matter']
    OMEGA_L          = _primary_hdr['omega_lambda']
    BARYON_FRAC      = _primary_hdr['baryon_frac']
    OMEGA_B          = BARYON_FRAC * OMEGA_M
    SNAPSHOT         = f"Snap_{_primary_hdr['last_snap_nr']}"
    REDSHIFTS        = _primary_hdr['redshifts']
    PART_MASS         = _primary_hdr['particle_mass']
    OUTPUT_DIR       = os.path.join(PRIMARY_DIR, 'plots/')

    # Snapshot aliases for key redshifts (derived from the redshift table)
    SNAP_Z0  = _snap_for_z(REDSHIFTS, 0.0)
    SNAP_Z1  = _snap_for_z(REDSHIFTS, 1.0)
    SNAP_Z2  = _snap_for_z(REDSHIFTS, 2.0)
    SNAP_Z3  = _snap_for_z(REDSHIFTS, 3.0)
    SNAP_Z4  = _snap_for_z(REDSHIFTS, 4.0)
    SNAP_Z5  = _snap_for_z(REDSHIFTS, 5.0)
    SNAP_Z6  = _snap_for_z(REDSHIFTS, 6.0)
    SNAP_Z7  = _snap_for_z(REDSHIFTS, 7.0)
    SNAP_Z8  = _snap_for_z(REDSHIFTS, 8.0)
    SNAP_Z10 = _snap_for_z(REDSHIFTS, 10.0)
    SNAP_Z11 = _snap_for_z(REDSHIFTS, 11.0)
    SNAP_Z12 = _snap_for_z(REDSHIFTS, 12.0)
    SNAP_Z13 = _snap_for_z(REDSHIFTS, 13.0)
    SNAP_Z14 = _snap_for_z(REDSHIFTS, 14.0)
    SNAP_Z20 = _snap_for_z(REDSHIFTS, 20.0)
else:
    # Fallback if primary HDF5 files are not available
    print("Warning: could not read primary model header — using hardcoded defaults")
    HUBBLE_H         = 0.73
    BOX_SIZE         = 62.5
    VOLUME_FRACTION  = 1.0
    VOLUME           = (BOX_SIZE / HUBBLE_H)**3 * VOLUME_FRACTION
    MASS_CONVERT     = 1.0e10 / HUBBLE_H
    OMEGA_M          = 0.25
    OMEGA_L          = 0.75
    BARYON_FRAC      = 0.17
    OMEGA_B          = 0.045
    SNAPSHOT         = 'Snap_63'
    REDSHIFTS        = [
        127.000, 79.998, 50.000, 30.000, 19.916, 18.244, 16.725, 15.343,
         14.086, 12.941, 11.897, 10.944, 10.073,  9.278,  8.550,  7.883,
          7.272,  6.712,  6.197,  5.724,  5.289,  4.888,  4.520,  4.179,
          3.866,  3.576,  3.308,  3.060,  2.831,  2.619,  2.422,  2.239,
          2.070,  1.913,  1.766,  1.630,  1.504,  1.386,  1.276,  1.173,
          1.078,  0.989,  0.905,  0.828,  0.755,  0.687,  0.624,  0.564,
          0.509,  0.457,  0.408,  0.362,  0.320,  0.280,  0.242,  0.208,
          0.175,  0.144,  0.116,  0.089,  0.064,  0.041,  0.020,  0.000,
    ]
    PART_MASS         = 0.0860657 * 1.0e10 / HUBBLE_H
    OUTPUT_DIR = './output/millennium/plots/'
    SNAP_Z0  = 63
    SNAP_Z1  = 39
    SNAP_Z2  = 32
    SNAP_Z3  = 27
    SNAP_Z4  = 23
    SNAP_Z5  = 20
    SNAP_Z6  = 18
    SNAP_Z7  = 16
    SNAP_Z8  = 14
    SNAP_Z10 = 12
    SNAP_Z11 = 11
    SNAP_Z12 = 10
    SNAP_Z13 = 9
    SNAP_Z14 = 8
    SNAP_Z15 = 7
    SNAP_Z20 = 4




# --------------- Secondary model parameters (from HDF5) ---------------

_secondary_hdr = _read_sim_header(SECONDARY_DIR)
if _secondary_hdr is not None:
    SECONDARY_HUBBLE_H        = _secondary_hdr['hubble_h']
    SECONDARY_BOX_SIZE        = _secondary_hdr['box_size']
    SECONDARY_VOLUME_FRACTION = _secondary_hdr['volume_fraction']
    SECONDARY_VOLUME          = (SECONDARY_BOX_SIZE / SECONDARY_HUBBLE_H)**3 * SECONDARY_VOLUME_FRACTION
    SECONDARY_MASS_CONVERT    = _secondary_hdr['unit_mass_in_g'] / _MSUN_CGS / SECONDARY_HUBBLE_H
    SECONDARY_FIRST_SNAP      = min(_secondary_hdr['output_snaps'])
    SECONDARY_LAST_SNAP       = max(_secondary_hdr['output_snaps'])
    SECONDARY_REDSHIFTS       = _secondary_hdr['redshifts']
    SECONDARY_PART_MASS       = _secondary_hdr['particle_mass']
else:
    # Fallback — same Millennium cosmology as primary
    SECONDARY_HUBBLE_H        = HUBBLE_H
    SECONDARY_BOX_SIZE        = BOX_SIZE
    SECONDARY_VOLUME_FRACTION = VOLUME_FRACTION
    SECONDARY_VOLUME          = VOLUME
    SECONDARY_MASS_CONVERT    = MASS_CONVERT
    SECONDARY_FIRST_SNAP      = SNAP_Z10
    SECONDARY_LAST_SNAP       = SNAP_Z0
    SECONDARY_REDSHIFTS       = REDSHIFTS
    SECONDARY_PART_MASS       = PART_MASS


# --------------- Tertiary model parameters (from HDF5) ---------------

_tertiary_hdr = _read_sim_header(TERTIARY_DIR)
if _tertiary_hdr is not None:
    TERTIARY_HUBBLE_H        = _tertiary_hdr['hubble_h']
    TERTIARY_BOX_SIZE        = _tertiary_hdr['box_size']
    TERTIARY_VOLUME_FRACTION = _tertiary_hdr['volume_fraction']
    TERTIARY_VOLUME          = (TERTIARY_BOX_SIZE / TERTIARY_HUBBLE_H)**3 * TERTIARY_VOLUME_FRACTION
    TERTIARY_MASS_CONVERT    = _tertiary_hdr['unit_mass_in_g'] / _MSUN_CGS / TERTIARY_HUBBLE_H
    TERTIARY_FIRST_SNAP      = min(_tertiary_hdr['output_snaps'])
    TERTIARY_LAST_SNAP       = max(_tertiary_hdr['output_snaps'])
    TERTIARY_REDSHIFTS       = _tertiary_hdr['redshifts']
    TERTIARY_PART_MASS       = _tertiary_hdr['particle_mass']
else:
    # Fallback — same Millennium cosmology as primary
    TERTIARY_HUBBLE_H        = HUBBLE_H
    TERTIARY_BOX_SIZE        = BOX_SIZE
    TERTIARY_VOLUME_FRACTION = VOLUME_FRACTION
    TERTIARY_VOLUME          = VOLUME
    TERTIARY_MASS_CONVERT    = MASS_CONVERT
    TERTIARY_FIRST_SNAP      = SNAP_Z10
    TERTIARY_LAST_SNAP       = SNAP_Z0
    TERTIARY_REDSHIFTS       = REDSHIFTS
    TERTIARY_PART_MASS       = PART_MASS


# --------------- Quaternary model parameters (from HDF5) ---------------

_quaternary_hdr = _read_sim_header(QUATERNARY_DIR) if QUATERNARY_DIR else None
if _quaternary_hdr is not None:
    QUATERNARY_HUBBLE_H        = _quaternary_hdr['hubble_h']
    QUATERNARY_BOX_SIZE        = _quaternary_hdr['box_size']
    QUATERNARY_VOLUME_FRACTION = _quaternary_hdr['volume_fraction']
    QUATERNARY_VOLUME          = (QUATERNARY_BOX_SIZE / QUATERNARY_HUBBLE_H)**3 * QUATERNARY_VOLUME_FRACTION
    QUATERNARY_MASS_CONVERT    = _quaternary_hdr['unit_mass_in_g'] / _MSUN_CGS / QUATERNARY_HUBBLE_H
    QUATERNARY_FIRST_SNAP      = min(_quaternary_hdr['output_snaps'])
    QUATERNARY_LAST_SNAP       = max(_quaternary_hdr['output_snaps'])
    QUATERNARY_REDSHIFTS       = _quaternary_hdr['redshifts']
    QUATERNARY_PART_MASS       = _quaternary_hdr['particle_mass']
else:
    # Fallback — same Millennium cosmology as primary
    QUATERNARY_HUBBLE_H        = HUBBLE_H
    QUATERNARY_BOX_SIZE        = BOX_SIZE
    QUATERNARY_VOLUME_FRACTION = VOLUME_FRACTION
    QUATERNARY_VOLUME          = VOLUME
    QUATERNARY_MASS_CONVERT    = MASS_CONVERT
    QUATERNARY_FIRST_SNAP      = SNAP_Z10
    QUATERNARY_LAST_SNAP       = SNAP_Z0
    QUATERNARY_REDSHIFTS       = REDSHIFTS
    QUATERNARY_PART_MASS       = PART_MASS

# Properties stored in HDF5 mass units (need MASS_CONVERT)
_MASS_PROPS = frozenset({
    'CentralMvir', 'Mvir', 'StellarMass', 'BulgeMass', 'BlackHoleMass',
    'MetalsStellarMass', 'MetalsColdGas', 'MetalsEjectedMass',
    'MetalsHotGas', 'MetalsCGMgas', 'ColdGas', 'HotGas', 'CGMgas',
    'EjectedMass', 'H2gas', 'H1gas', 'IntraClusterStars',
    'MergerBulgeMass', 'InstabilityBulgeMass',
})

# Default properties to load for the primary model
_DEFAULT_PROPERTIES = [
    'StellarMass', 'BulgeMass', 'ColdGas', 'HotGas', 'CGMgas',
    'EjectedMass', 'H2gas', 'H1gas', 'BlackHoleMass',
    'IntraClusterStars', 'CentralMvir', 'Mvir',
    'MergerBulgeMass', 'InstabilityBulgeMass',
    'MetalsStellarMass', 'MetalsColdGas', 'MetalsHotGas',
    'MetalsEjectedMass', 'MetalsCGMgas',
    'SfrDisk', 'SfrBulge', 'Vvir', 'Vmax', 'Rvir',
    'DiskRadius', 'BulgeRadius',
    'Type', 'CentralGalaxyIndex',
    'Posx', 'Posy', 'Posz',
    'OutflowRate', 'MassLoading', 'Cooling', 'Regime',
]

# Properties to load for evolution (multi-snapshot) plots
_EVOLUTION_PROPERTIES = [
    'StellarMass', 'SfrDisk', 'SfrBulge', 'Mvir', 'Rvir',
    'CGMgas', 'HotGas', 'MetalsStellarMass', 'DiskRadius',
    'FFBRegime', 'Regime', 'tcool_over_tff', 'tdeplete', 'tff',
    'GalaxyIndex', 'Type',
    'BulgeMass', 'ColdGas', 'H2gas', 'BlackHoleMass',
    'MergerBulgeMass', 'InstabilityBulgeMass',
]


# ========================== PLOTTING STYLE ==========================

def setup_style():
    """Configure matplotlib for publication-quality white-background plots."""
    plt.style.use("./plotting/kieren_cohare_palatino_sty.mplstyle")

# ========================== DATA I/O ==========================


def find_model_files(directory):
    """
    Find all model_*.hdf5 files in *directory*.

    Returns a sorted list of absolute paths.  Falls back to the single
    ``model_0.hdf5`` if no files match (backward-compatible).
    """
    return _find_model_files_early(directory)


def model_files_exist(directory):
    """Return True if at least one model HDF5 file exists in *directory*."""
    return len(find_model_files(directory)) > 0


def read_snap_from_files(filepaths, snap_key, properties, mass_convert=MASS_CONVERT,
                         particle_mass=None, mvir_particles_min=None):
    """
    Read *properties* from *snap_key* across multiple HDF5 files and
    concatenate the results.

    Parameters
    ----------
    filepaths : list of str
        HDF5 file paths (e.g. from ``find_model_files``).
    snap_key : str
        Snapshot group name, e.g. ``'Snap_63'``.
    properties : list of str
        Dataset names to read.
    mass_convert : float
        Multiplicative factor applied to properties in ``_MASS_PROPS``.
    particle_mass : float, optional
        DM particle mass for the simulation.
    mvir_particles_min : int, optional
        If provided alongside *particle_mass*, also applies a halo-mass
        resolution cut ``Mvir >= mvir_particles_min * particle_mass`` (after
        unit conversion).

    Returns
    -------
    dict : property name -> numpy array (concatenated across files).
           Empty dict if no file contains *snap_key*.
    """
    caller_wants_len = 'Len' in properties
    caller_wants_mvir = 'Mvir' in properties

    load_props = list(properties)
    if not caller_wants_len:
        load_props.append('Len')

    # If we're asked to apply a particle-mass-derived resolution cut, we need Mvir.
    if (particle_mass is not None and mvir_particles_min is not None) and 'Mvir' not in load_props:
        load_props.append('Mvir')

    chunks = {prop: [] for prop in load_props}
    found_snap = False

    for fp in filepaths:
        try:
            with h5.File(fp, 'r') as f:
                if snap_key not in f:
                    continue
                found_snap = True
                grp = f[snap_key]
                for prop in load_props:
                    if prop in grp:
                        chunks[prop].append(np.array(grp[prop]))
        except Exception as e:
            print(f"  Warning: could not read {fp}: {e}")
            continue

    if not found_snap:
        return {}

    data = {}
    for prop in load_props:
        if chunks[prop]:
            arr = np.concatenate(chunks[prop])
            if prop in _MASS_PROPS:
                arr = arr * mass_convert
            data[prop] = arr

    if 'Len' in data:
        mask = data['Len'] >= MIN_PARTICLES
        if (particle_mass is not None and mvir_particles_min is not None) and 'Mvir' in data:
            min_mvir = int(mvir_particles_min) * float(particle_mass) * mass_convert
            mask &= data['Mvir'] >= min_mvir
        data = {p: arr[mask] for p, arr in data.items()}

    if not caller_wants_len:
        data.pop('Len', None)

    if (particle_mass is not None and mvir_particles_min is not None) and not caller_wants_mvir:
        data.pop('Mvir', None)

    return data


def load_model(directory, filename=None, snapshot=SNAPSHOT,
               properties=None, mass_convert=MASS_CONVERT, particle_mass=None,
               mvir_particles_min=None):
    """
    Load galaxy properties from one or more model HDF5 files.

    When SAGE is run with MPI each rank writes its own file
    (``model_0.hdf5``, ``model_1.hdf5``, …).  This function automatically
    discovers all such files and concatenates their datasets.

    Parameters
    ----------
    directory : str
        Path to the model output directory.
    filename : str, optional
        Kept for backward compatibility.  If given, only that single file
        is read; otherwise every ``model_*.hdf5`` in *directory* is used.
    snapshot : str
        Snapshot key (e.g. ``'Snap_63'``).
    properties : list of str, optional
        Properties to load.  If *None*, loads ``_DEFAULT_PROPERTIES``.

    Returns
    -------
    dict : property name -> numpy array (converted where applicable).
    """
    if properties is None:
        properties = _DEFAULT_PROPERTIES

    if filename is not None:
        filepaths = [os.path.join(directory, filename)]
    else:
        filepaths = find_model_files(directory)

    if not filepaths:
        print(f"  Warning: no model files found in {directory}")
        return {}

    data = read_snap_from_files(filepaths, snapshot, properties,
                                mass_convert=mass_convert,
                                particle_mass=particle_mass,
                                mvir_particles_min=mvir_particles_min)
    if not data:
        print(f"  Warning: {snapshot} not found in any file in {directory}")
    return data


def load_snapshots(directory, snaps, properties=None, filename=None,
                   mass_convert=MASS_CONVERT, particle_mass=None,
                   mvir_particles_min=None):
    """
    Load multiple snapshots from one or more HDF5 files.

    Parameters
    ----------
    directory : str
        Path to model output directory.
    snaps : list of int
        Snapshot numbers to load.
    properties : list of str, optional
        Properties to load.  Defaults to ``_EVOLUTION_PROPERTIES``.
    filename : str, optional
        If given, only that single file is read; otherwise every
        ``model_*.hdf5`` in *directory* is used.

    Returns
    -------
    dict : {snap_num: {prop_name: numpy array}}
    """
    if properties is None:
        properties = _EVOLUTION_PROPERTIES

    if filename is not None:
        filepaths = [os.path.join(directory, filename)]
    else:
        filepaths = find_model_files(directory)

    if not filepaths:
        print(f"  Warning: no model files found in {directory}")
        return {}

    snapdata = {}
    for snap in snaps:
        snap_key = f'Snap_{snap}'
        data = read_snap_from_files(filepaths, snap_key, properties,
                                    mass_convert=mass_convert,
                                    particle_mass=particle_mass,
                                    mvir_particles_min=mvir_particles_min)
        if data:
            snapdata[snap] = data
        else:
            print(f"  Warning: {snap_key} not found, skipping.")

    return snapdata


# ========================== COMPUTATION UTILITIES ==========================

def mass_function(log_masses, volume, binwidth=0.1, mass_range=None):
    """
    Compute a mass function (log10 number density per dex per Mpc^3).

    Parameters
    ----------
    log_masses : array
        log10 masses.
    volume : float
        Comoving volume in Mpc^3.
    binwidth : float
        Bin width in dex.
    mass_range : tuple of (float, float), optional
        (min, max) for histogram. Auto-determined if None.

    Returns
    -------
    centers : array
        Bin centres.
    phi : array
        log10(number density). NaN where counts == 0.
    mrange : tuple
        (min, max) used, so subsets can reuse the same bins.
    """
    if mass_range is None:
        mi = np.floor(np.min(log_masses)) - 2
        ma = np.floor(np.max(log_masses)) + 2
    else:
        mi, ma = mass_range

    nbins = int(round((ma - mi) / binwidth))
    counts, edges = np.histogram(log_masses, range=(mi, ma), bins=nbins)
    centers = edges[:-1] + 0.5 * binwidth

    with np.errstate(divide='ignore'):
        phi = np.log10(counts / volume / binwidth)
    phi[~np.isfinite(phi)] = np.nan

    return centers, phi, (mi, ma)



def metallicity_12logOH(metals_cold_gas, cold_gas):
    """
    Gas-phase metallicity in 12 + log10(O/H).

    Uses Z_cold = MetalsColdGas / ColdGas, solar reference `Z_SUN` defined
    at module level (Asplund+2009 value 0.0134), and
    12 + log10(O/H)_sun = 8.69 (Asplund et al. 2009).
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        Z_cold = metals_cold_gas / cold_gas
        return np.log10(Z_cold / Z_SUN) + 8.69


def stellar_metallicity(metals_stellar_mass, stellar_mass):
    """
    Stellar metallicity log10(Z/Z_sun).
    Uses Z_star = MetalsStellarMass / StellarMass and the module-level
    `Z_SUN` (Asplund+2009 = 0.0134) as the solar reference.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.log10((metals_stellar_mass / stellar_mass) / Z_SUN)


def log_ssfr(sfr_disk, sfr_bulge, stellar_mass):
    """Compute log10(sSFR / yr^-1)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.log10((sfr_disk + sfr_bulge) / stellar_mass)


def binned_median(x, y, bins, min_count=5):
    """Binned median with 25th/75th percentiles."""
    centers = 0.5 * (bins[:-1] + bins[1:])
    n = len(bins) - 1
    med = np.full(n, np.nan)
    p25 = np.full(n, np.nan)
    p75 = np.full(n, np.nan)

    for i in range(n):
        mask = (x >= bins[i]) & (x < bins[i + 1])
        count = np.sum(mask)
        if count >= min_count:
            vals = y[mask]
            med[i] = np.median(vals)
            p25[i] = np.percentile(vals, 25)
            p75[i] = np.percentile(vals, 75)

    return centers, med, p25, p75


def binned_percentiles(x, y, bins, percentiles=(16, 50, 84), min_count=20):
    """Compute binned percentiles of *y* as a function of *x*.

    Parameters
    ----------
    x, y : array-like
        Data arrays.
    bins : array-like
        Bin edges in x.
    percentiles : tuple
        Percentiles to compute (e.g. (16, 50, 84)).
    min_count : int
        Minimum number of points required in a bin.

    Returns
    -------
    centers : array
        Bin centers.
    pct : array, shape (len(percentiles), nbins)
        Percentiles per bin; NaN for bins with insufficient counts.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]

    centers = 0.5 * (bins[:-1] + bins[1:])
    nbins = len(bins) - 1
    pct = np.full((len(percentiles), nbins), np.nan)

    for i in range(nbins):
        m = (x >= bins[i]) & (x < bins[i + 1])
        if np.sum(m) >= min_count:
            pct[:, i] = np.percentile(y[m], percentiles)

    return centers, pct


def plot_binned_median_1sigma(
    ax,
    x,
    y,
    bins,
    *,
    color,
    label,
    alpha=0.25,
    lw=3.0,
    ls='-',
    min_count=20,
    zorder_fill=3,
    zorder_line=4,
):
    """Plot a median line with a 16--84% (1\u03c3) shaded band."""
    centers, pct = binned_percentiles(x, y, bins, percentiles=(16, 50, 84), min_count=min_count)
    p16, p50, p84 = pct
    valid = np.isfinite(p50) & np.isfinite(p16) & np.isfinite(p84)
    if not np.any(valid):
        return None

    ax.fill_between(centers[valid], p16[valid], p84[valid],
                    color=color, alpha=alpha, lw=0.0, zorder=zorder_fill)
    (line,) = ax.plot(centers[valid], p50[valid],
                      color=color, lw=lw, ls=ls, label=label, zorder=zorder_line)
    return line


def baryon_fractions_by_halo_mass(primary, halo_bins=None):
    """
    Compute mean baryon component fractions binned by halo mass.

    Uses np.bincount to sum components per halo in O(N), avoiding
    per-halo Python loops.

    Returns
    -------
    mass_centers : array
        Mean log10(Mvir) in each occupied bin.
    results : dict
        {component_name: {'mean': array, 'upper': array, 'lower': array}}
    """
    if halo_bins is None:
        halo_bins = np.arange(11.0, 16.1, 0.1)

    cgi = primary['CentralGalaxyIndex'].astype(np.int64)

    # Remap CentralGalaxyIndex IDs to compact 0-based group indices
    unique_ids, compact_idx = np.unique(cgi, return_inverse=True)
    ngroups = len(unique_ids)

    # Components to track
    comp_keys = ['StellarMass', 'ColdGas', 'HotGas', 'CGMgas',
                 'IntraClusterStars', 'BlackHoleMass', 'EjectedMass']

    # Sum each component by halo using bincount — O(N), fully vectorized
    halo_sums = {}
    for key in comp_keys:
        halo_sums[key] = np.bincount(compact_idx, weights=primary[key],
                                     minlength=ngroups)
    halo_sums['Total'] = sum(halo_sums[k] for k in comp_keys)

    # Central galaxies define halos
    central_mask = primary['Type'] == 0
    central_compact = compact_idx[central_mask]
    mvir = primary['Mvir'][central_mask]
    log_mvir = np.log10(mvir)

    # Fractions: component_sum / Mvir for each halo
    fractions = {}
    all_keys = ['Total'] + comp_keys
    for key in all_keys:
        fractions[key] = halo_sums[key][central_compact] / mvir

    # Bin by halo mass and compute mean +/- stderr
    bin_idx = np.digitize(log_mvir, halo_bins) - 1
    results = {k: {'mean': [], 'upper': [], 'lower': []} for k in all_keys}
    mass_centers = []

    for i in range(len(halo_bins) - 1):
        w = bin_idx == i
        n_halos = np.sum(w)
        if n_halos < 3:
            continue

        mass_centers.append(np.mean(log_mvir[w]))
        sqrt_n = np.sqrt(n_halos)

        for key in all_keys:
            vals = fractions[key][w]
            mean = np.mean(vals)
            err = np.std(vals) / sqrt_n
            results[key]['mean'].append(mean)
            results[key]['upper'].append(mean + err)
            results[key]['lower'].append(max(mean - err, 1e-6))

    # Convert to arrays
    mass_centers = np.array(mass_centers)
    for key in results:
        for stat in results[key]:
            results[key][stat] = np.array(results[key][stat])

    return mass_centers, results


def snap_to_redshift(snap):
    """Return the redshift for a given snapshot number."""
    return REDSHIFTS[snap]


def cosmic_time_gyr(z):
    """Age of the universe at redshift z, in Gyr."""
    t_H = 977.8 / (HUBBLE_H * 100)  # Hubble time in Gyr

    def integrand(zp):
        return 1.0 / ((1 + zp) * np.sqrt(OMEGA_M * (1 + zp)**3 + OMEGA_L))

    result, _ = quad(integrand, z, 1000.0)
    return t_H * result


# ========================== FIGURE UTILITIES ==========================

def save_figure(fig, filepath):
    """Save figure to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath)
    print(f'  Saved: {filepath}')
    plt.close(fig)


def _standard_legend(ax, loc='lower left', handles=None, labels=None, **kwargs):
    """Apply consistent legend formatting with fully opaque handles."""
    kwargs.setdefault('frameon', False)
    if handles is not None and labels is not None:
        leg = ax.legend(handles, labels, loc=loc, numpoints=1,
                        labelspacing=0.1, **kwargs)
    else:
        leg = ax.legend(loc=loc, numpoints=1, labelspacing=0.1, **kwargs)
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    return leg

# ========================== PLOTS ==========================

def plot_1_number_counts(primary, secondary, tertiary=None, quaternary=None):
    """2×5 panel number-count figure comparing primary, secondary, and optional tertiary models.

    Panels (requested):
    - Mvir centrals
    - Mvir satellites
    - Stellar mass centrals
    - Stellar mass satellites
    - Cold gas
    - H1
    - H2
    - Gas metallicity (12 + log(O/H))
    - Black hole mass
    - ICS mass

    Notes
    -----
    Resolution thresholds are applied at load time in `read_snap_from_files()`:
    - `Len >= MIN_PARTICLES`
    - `Mvir >= MIN_MVIR_PARTICLES * particle_mass` (per simulation)
    """
    print('Plot 1: Number counts (2x5)')

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    axes = axes.flatten()

    models = [
        (PRIMARY_LABEL,    primary,    PRIMARY_COLOR,    PRIMARY_LS),
        (SECONDARY_LABEL,  secondary,  SECONDARY_COLOR,  SECONDARY_LS),
    ]
    if tertiary:
        models.append((TERTIARY_LABEL,   tertiary,   TERTIARY_COLOR,   TERTIARY_LS))
    if quaternary:
        models.append((QUATERNARY_LABEL, quaternary, QUATERNARY_COLOR, QUATERNARY_LS))

    # Resolution mass limits in physical Msun (for optional reference lines).
    mmin_primary = MIN_MVIR_PARTICLES * PART_MASS * MASS_CONVERT
    mmin_mini = MIN_MVIR_PARTICLES * SECONDARY_PART_MASS * SECONDARY_MASS_CONVERT

    def _log10_positive(x):
        x = np.asarray(x, dtype=float)
        ok = np.isfinite(x) & (x > 0.0)
        return np.log10(x[ok])

    def _finite(x):
        x = np.asarray(x, dtype=float)
        return x[np.isfinite(x)]

    def _bins_for(series_list, binwidth, pad_bins=1):
        series_list = [s for s in series_list if s is not None and len(s) > 0]
        if not series_list:
            return None
        vals = np.concatenate(series_list)
        lo = np.nanmin(vals)
        hi = np.nanmax(vals)
        lo = np.floor(lo / binwidth) * binwidth - pad_bins * binwidth
        hi = np.ceil(hi / binwidth) * binwidth + pad_bins * binwidth
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return None
        return np.arange(lo, hi + binwidth, binwidth)

    def _step_hist(ax, values, bins, *, color, label, ls='-', lw=2.2):
        counts, edges = np.histogram(values, bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        counts = counts.astype(float)
        counts[counts <= 0] = np.nan
        ax.plot(centers, counts, drawstyle='steps-mid', color=color, ls=ls, lw=lw,
            label=label, alpha=0.85)

    def _panel(ax, title, xlabel, series, *, binwidth=0.2, legend=False, vlines=None):
        bins = _bins_for([v for (_, _, _, v) in series], binwidth)
        if bins is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return

        for label, color, ls, vals in series:
            vals = _finite(vals)
            if len(vals) == 0:
                continue
            _step_hist(ax, vals, bins, color=color, label=label, ls=ls)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('N')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.25)

        if vlines:
            for x, color, ls in vlines:
                ax.axvline(x, color=color, ls=ls, lw=1.6, alpha=0.85)

        if legend:
            ax.legend(loc='upper right', frameon=False, fontsize=9)

    vlines_mvir = [
        (np.log10(mmin_primary), PRIMARY_COLOR,   ':'),
        (np.log10(mmin_mini),    SECONDARY_COLOR, ':'),
    ]

    # 1) Mvir centrals
    _panel(
        axes[0],
        'Mvir (centrals)',
        r'$\log_{10}\, M_{\rm vir}\,[M_\odot]$',
        [(name, color, ls, _log10_positive(d['Mvir'][d['Type'] == 0])) for name, d, color, ls in models],
        legend=True,
        vlines=vlines_mvir,
    )

    # 2) Mvir satellites
    _panel(
        axes[1],
        'Mvir (satellites)',
        r'$\log_{10}\, M_{\rm vir}\,[M_\odot]$',
        [(name, color, ls, _log10_positive(d['Mvir'][d['Type'] != 0])) for name, d, color, ls in models],
        vlines=vlines_mvir,
    )

    # 3) Stellar mass centrals
    _panel(
        axes[2],
        'Stellar mass (centrals)',
        r'$\log_{10}\, m_{\ast}\,[M_\odot]$',
        [(name, color, ls, _log10_positive(d['StellarMass'][d['Type'] == 0])) for name, d, color, ls in models],
    )

    # 4) Stellar mass satellites
    _panel(
        axes[3],
        'Stellar mass (satellites)',
        r'$\log_{10}\, m_{\ast}\,[M_\odot]$',
        [(name, color, ls, _log10_positive(d['StellarMass'][d['Type'] != 0])) for name, d, color, ls in models],
    )

    # 5) Cold gas
    _panel(
        axes[4],
        'Cold gas',
        r'$\log_{10}\, m_{\rm cold}\,[M_\odot]$',
        [(name, color, ls, _log10_positive(d['ColdGas'])) for name, d, color, ls in models],
    )

    # 6) H1
    _panel(
        axes[5],
        'H1',
        r'$\log_{10}\, m_{\rm H1}\,[M_\odot]$',
        [(name, color, ls, _log10_positive(d['H1gas'])) for name, d, color, ls in models],
    )

    # 7) H2
    _panel(
        axes[6],
        'H2',
        r'$\log_{10}\, m_{\rm H2}\,[M_\odot]$',
        [(name, color, ls, _log10_positive(d['H2gas'])) for name, d, color, ls in models],
    )

    # 8) Metallicity
    _panel(
        axes[7],
        r'Gas metallicity',
        r'$12 + \log_{10}(\mathrm{O/H})$',
        [(name, color, ls, _finite(metallicity_12logOH(d['MetalsColdGas'], d['ColdGas']))) for name, d, color, ls in models],
        binwidth=0.05,
    )

    # 9) Black hole mass
    _panel(
        axes[8],
        'Black hole mass',
        r'$\log_{10}\, m_{\rm BH}\,[M_\odot]$',
        [(name, color, ls, _log10_positive(d['BlackHoleMass'])) for name, d, color, ls in models],
    )

    # 10) ICS
    _panel(
        axes[9],
        'ICS mass',
        r'$\log_{10}\, m_{\rm ICS}\,[M_\odot]$',
        [(name, color, ls, _log10_positive(d['IntraClusterStars'])) for name, d, color, ls in models],
    )

    # fig.suptitle(
    #     'Number counts — ' + ', '.join(
    #         f'{name}: {len(d["StellarMass"]):,}' for name, d, _ in models
    #     ),
    #     y=0.995,
    # )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    save_figure(fig, os.path.join(OUTPUT_DIR, 'NumberCounts_2x5' + OUTPUT_FORMAT))


def plot_2_diagnostics(primary, secondary, tertiary=None, quaternary=None):
    """2×5 panel diagnostics comparing primary, secondary, and optional tertiary models at z=0.

    Panels (requested):
    - Stellar Mass Function
    - Stellar mass vs Star Formation Rate
    - Baryonic Tully-Fisher Relation
    - Stellar to halo mass ratio
    - Stellar to halo mass relation
    - Stellar mass vs metallicity
    - Black hole mass vs bulge mass
    - Cold gas Mass Function
    - H2 Mass function
    - Stellar mass vs virial velocity
    """
    print('Plot 2: Diagnostics (2x5)')

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    axes = axes.flatten()

    models = [
        (PRIMARY_LABEL,    primary,    PRIMARY_COLOR,    VOLUME,            PRIMARY_LS),
        (SECONDARY_LABEL,  secondary,  SECONDARY_COLOR,  SECONDARY_VOLUME,  SECONDARY_LS),
    ]
    if tertiary:
        models.append((TERTIARY_LABEL,   tertiary,   TERTIARY_COLOR,   TERTIARY_VOLUME,   TERTIARY_LS))
    if quaternary:
        models.append((QUATERNARY_LABEL, quaternary, QUATERNARY_COLOR, QUATERNARY_VOLUME, QUATERNARY_LS))

    def _finite(x):
        x = np.asarray(x, dtype=float)
        return x[np.isfinite(x)]

    def _auto_bins(x_list, step, pad=1):
        xs = [np.asarray(x) for x in x_list if x is not None and len(x) > 0]
        if not xs:
            return None
        x = np.concatenate(xs)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return None
        lo = np.floor(np.min(x) / step) * step - pad * step
        hi = np.ceil(np.max(x) / step) * step + pad * step
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return None
        return np.arange(lo, hi + step, step)

    def _log10_pos(x):
        x = np.asarray(x, dtype=float)
        ok = np.isfinite(x) & (x > 0.0)
        return np.log10(x[ok])

    def _mass_function_panel(ax, prop, title, xlabel, *, binwidth=0.2, legend=False):
        series = []
        for name, d, color, vol, ls in models:
            vals = _log10_pos(d[prop]) if prop in d else np.array([])
            series.append((name, vals, color, vol, ls))

        all_vals = [v for (_, v, _, _, _) in series if v.size > 0]
        if not all_vals:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return

        combined = np.concatenate(all_vals)
        mi = np.floor(np.min(combined)) - 1
        ma = np.ceil(np.max(combined)) + 1
        mass_range = (mi, ma)

        for name, vals, color, vol, ls in series:
            if vals.size == 0:
                continue
            centers, phi, _ = mass_function(vals, vol, binwidth=binwidth, mass_range=mass_range)
            ax.plot(centers, phi, color=color, ls=ls, lw=2.6, alpha=0.9, label=name)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r'$\log_{10}\,\Phi\ [{\rm Mpc}^{-3}\,{\rm dex}^{-1}]$')
        ax.grid(True, alpha=0.25)
        ax.set_xlim(5, 12)
        if legend:
            ax.legend(loc='lower left', frameon=False, fontsize=9)

    def _relation_panel(ax, title, xlabel, ylabel, xys, *, xstep=0.2, legend=False, vlines=None):
        bins = _auto_bins([x for (_, _, _, x, _) in xys], xstep, pad=1)
        if bins is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return

        for name, color, ls, x, y in xys:
            if x.size == 0 or y.size == 0:
                continue
            plot_binned_median_1sigma(
                ax,
                x,
                y,
                bins,
                color=color,
                label=name,
                ls=ls,
                alpha=0.20,
                lw=2.8,
                min_count=30,
            )

        if vlines:
            for xv, color, ls in vlines:
                ax.axvline(xv, color=color, ls=ls, lw=1.6, alpha=0.85)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        if legend:
            ax.legend(loc='best', frameon=False, fontsize=9)

    # Precompute halo-mass resolution limits (physical Msun)
    mmin_primary = MIN_MVIR_PARTICLES * PART_MASS * MASS_CONVERT
    mmin_mini = MIN_MVIR_PARTICLES * SECONDARY_PART_MASS * SECONDARY_MASS_CONVERT
    vlines_mvir = [
        (np.log10(mmin_primary), PRIMARY_COLOR,   ':'),
        (np.log10(mmin_mini),    SECONDARY_COLOR, ':'),
    ]

    # 1) Stellar Mass Function
    _mass_function_panel(
        axes[0],
        'StellarMass',
        'Stellar Mass Function',
        r'$\log_{10}\, M_{\ast}\,[M_\odot]$',
        binwidth=0.2,
        legend=True,
    )

    # 2) Stellar mass vs SFR
    xys = []
    for name, d, color, _vol, ls in models:
        mstar = np.asarray(d['StellarMass'], dtype=float)
        sfr = np.asarray(d['SfrDisk'], dtype=float) + np.asarray(d['SfrBulge'], dtype=float)
        ok = np.isfinite(mstar) & (mstar > 0.0) & np.isfinite(sfr) & (sfr > 0.0)
        x = np.log10(mstar[ok])
        y = np.log10(sfr[ok])
        xys.append((name, color, ls, x, y))
    _relation_panel(
        axes[1],
        r'$M_{\ast}$ vs SFR',
        r'$\log_{10}\, M_{\ast}\,[M_\odot]$',
        r'$\log_{10}\, {\rm SFR}\,[M_\odot\,{\rm yr}^{-1}]$',
        xys,
        xstep=0.2,
        legend=False,
    )

    # 3) Baryonic Tully-Fisher Relation (Sb/c selection; median lines)
    xys = []
    for name, d, color, _vol, ls in models:
        if 'Vmax' not in d:
            xys.append((name, color, ls, np.array([]), np.array([])))
            continue

        # Sb/c selection (as in your example):
        # centrals, Mbar>0, 0.1 < B/T < 0.5 (with B/T ~ BulgeMass/StellarMass)
        t = np.asarray(d['Type'])
        mstar = np.asarray(d['StellarMass'], dtype=float)
        mcold = np.asarray(d['ColdGas'], dtype=float)
        mbulge = np.asarray(d['BulgeMass'], dtype=float)
        v = np.asarray(d['Vmax'], dtype=float)

        with np.errstate(divide='ignore', invalid='ignore'):
            bt = mbulge / mstar
        mbar = mstar + mcold

        ok = (
            (t == 0)
            & np.isfinite(v) & (v > 0.0)
            & np.isfinite(mbar) & (mbar > 0.0)
            & np.isfinite(bt) & (bt > 0.1) & (bt < 0.5)
        )
        x = np.log10(v[ok])
        y = np.log10(mbar[ok])
        xys.append((name, color, ls, x, y))

    # Overplot Stark, McGaugh & Swatters 2009 band in (logV, logMbar) space.
    ax = axes[2]
    xgrid = np.arange(1.4, 2.9 + 1e-6, 0.05)
    tf = 3.94 * xgrid + 1.79
    ax.fill_between(xgrid, tf - 0.26, tf + 0.26, color='blue', alpha=0.2, zorder=1)

    _relation_panel(
        ax,
        'Baryonic TF (Sb/c)',
        r'$\log_{10}\, V_{\max}\,[{\rm km\,s^{-1}}]$',
        r'$\log_{10}\, (M_{\ast}+M_{\rm cold})\,[M_\odot]$',
        xys,
        xstep=0.05,
        legend=True,
    )

    from matplotlib.ticker import MultipleLocator
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(0.25))
    ax.set_xlim(1.4, 2.9)
    ax.set_ylim(7.5, 12.0)

    # 4) Stellar-to-halo mass ratio (centrals)
    xys = []
    for name, d, color, _vol, ls in models:
        c = (d['Type'] == 0)
        mvir = np.asarray(d['Mvir'], dtype=float)[c]
        mstar = np.asarray(d['StellarMass'], dtype=float)[c]
        ok = np.isfinite(mvir) & (mvir > 0.0) & np.isfinite(mstar) & (mstar > 0.0)
        x = np.log10(mvir[ok])
        with np.errstate(divide='ignore', invalid='ignore'):
            y = np.log10(mstar[ok] / mvir[ok])
        xys.append((name, color, ls, x, y))
    _relation_panel(
        axes[3],
        r'Stellar-to-halo ratio',
        r'$\log_{10}\, M_{\rm vir}\,[M_\odot]$',
        r'$\log_{10}(M_{\ast}/M_{\rm vir})$',
        xys,
        xstep=0.2,
        legend=False,
        vlines=vlines_mvir,
    )

    # 5) Stellar-to-halo mass relation (centrals)
    xys = []
    for name, d, color, _vol, ls in models:
        c = (d['Type'] == 0)
        mvir = np.asarray(d['Mvir'], dtype=float)[c]
        mstar = np.asarray(d['StellarMass'], dtype=float)[c]
        ok = np.isfinite(mvir) & (mvir > 0.0) & np.isfinite(mstar) & (mstar > 0.0)
        x = np.log10(mvir[ok])
        y = np.log10(mstar[ok])
        xys.append((name, color, ls, x, y))
    _relation_panel(
        axes[4],
        r'Stellar-to-halo relation',
        r'$\log_{10}\, M_{\rm vir}\,[M_\odot]$',
        r'$\log_{10}\, M_{\ast}\,[M_\odot]$',
        xys,
        xstep=0.2,
        legend=False,
        vlines=vlines_mvir,
    )

    # 6) Stellar mass vs metallicity
    xys = []
    for name, d, color, _vol, ls in models:
        mstar = np.asarray(d['StellarMass'], dtype=float)
        z = metallicity_12logOH(d['MetalsColdGas'], d['ColdGas'])
        ok = np.isfinite(mstar) & (mstar > 0.0) & np.isfinite(z)
        x = np.log10(mstar[ok])
        y = np.asarray(z, dtype=float)[ok]
        xys.append((name, color, ls, x, y))
    _relation_panel(
        axes[5],
        r'$M_{\ast}$ vs metallicity',
        r'$\log_{10}\, M_{\ast}\,[M_\odot]$',
        r'$12 + \log_{10}(\mathrm{O/H})$',
        xys,
        xstep=0.2,
        legend=False,
    )

    # 7) Black hole mass vs bulge mass
    xys = []
    for name, d, color, _vol, ls in models:
        mbulge = np.asarray(d['BulgeMass'], dtype=float)
        mbh = np.asarray(d['BlackHoleMass'], dtype=float)
        ok = np.isfinite(mbulge) & (mbulge > 0.0) & np.isfinite(mbh) & (mbh > 0.0)
        x = np.log10(mbulge[ok])
        y = np.log10(mbh[ok])
        xys.append((name, color, ls, x, y))

    # Haring & Rix (2004) reference band (log10 Mbh vs log10 Mbulge)
    # Common parameterization:
    #   log10(M_BH/Msun) = 8.20 + 1.12 * (log10(M_bulge/Msun) - 11)
    # with ~0.3 dex scatter.
    ax = axes[6]
    hr_slope = 1.12
    hr_norm = 8.20
    hr_pivot = 11.0
    hr_scatter = 0.30
    xgrid = np.arange(8.0, 12.6 + 1e-6, 0.05)
    ygrid = hr_norm + hr_slope * (xgrid - hr_pivot)
    ax.fill_between(
        xgrid,
        ygrid - hr_scatter,
        ygrid + hr_scatter,
        color='0.3',
        alpha=0.15,
        zorder=1,
        label=r'Haring \& Rix (2004)',
    )

    _relation_panel(
        ax,
        r'$M_{\rm BH}$ vs $M_{\rm bulge}$',
        r'$\log_{10}\, M_{\rm bulge}\,[M_\odot]$',
        r'$\log_{10}\, M_{\rm BH}\,[M_\odot]$',
        xys,
        xstep=0.2,
        legend=True,
    )

    # 8) Cold gas Mass Function
    _mass_function_panel(
        axes[7],
        'ColdGas',
        'Cold gas MF',
        r'$\log_{10}\, M_{\rm cold}\,[M_\odot]$',
        binwidth=0.2,
        legend=False,
    )

    # 9) H2 Mass Function
    _mass_function_panel(
        axes[8],
        'H2gas',
        'H2 MF',
        r'$\log_{10}\, M_{\rm H2}\,[M_\odot]$',
        binwidth=0.2,
        legend=False,
    )

    # 10) Stellar mass vs virial velocity (Vvir)
    xys = []
    for name, d, color, _vol, ls in models:
        t = np.asarray(d['Type'])
        vvir = np.asarray(d['Vvir'], dtype=float)
        mstar = np.asarray(d['StellarMass'], dtype=float)
        ok = (t == 0) & np.isfinite(vvir) & (vvir > 0.0) & np.isfinite(mstar) & (mstar > 0.0)
        x = np.log10(vvir[ok])
        y = np.log10(mstar[ok])
        xys.append((name, color, ls, x, y))
    _relation_panel(
        axes[9],
        r'$M_{\ast}$ vs $V_{\rm vir}$',
        r'$\log_{10}\, V_{\rm vir}\,[{\rm km\,s^{-1}}]$',
        r'$\log_{10}\, M_{\ast}\,[M_\odot]$',
        xys,
        xstep=0.05,
        legend=False,
    )

    # fig.suptitle(
    #     f'Diagnostics',
    #     y=0.995,
    # )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    save_figure(fig, os.path.join(OUTPUT_DIR, 'Diagnostics_2x5' + OUTPUT_FORMAT))

# ========================== MAIN ==========================

# Registry of plot functions
# ========================== EVOLUTION UTILITIES ==========================

def _lookback_time_Gyr(z):
    """Lookback time in Gyr for scalar or array z."""
    inv_H0_Gyr = (3.0857e19 / (HUBBLE_H * 100)) / 3.1557e16
    scalar = np.ndim(z) == 0
    zs = np.atleast_1d(np.asarray(z, dtype=float))
    result = np.zeros_like(zs)
    for i, zi in enumerate(zs):
        result[i], _ = quad(
            lambda zp: 1.0 / ((1 + zp) * np.sqrt(OMEGA_M * (1 + zp)**3 + OMEGA_L)),
            0, zi,
        )
    result *= inv_H0_Gyr
    return float(result[0]) if scalar else result


def _redshift_axis(ax, age_universe, z_ticks=(0, 0.5, 1, 2, 3, 5, 7)):
    """Add a secondary top x-axis showing redshift ticks on a lookback-time axis."""
    ax2 = ax.twiny()
    tick_lbt = np.array([_lookback_time_Gyr(z) for z in z_ticks])
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(tick_lbt)
    ax2.set_xticklabels([str(z) for z in z_ticks])
    ax2.set_xlabel('Redshift')
    return ax2


def plot_3_evolution(snapdata_h2, snapdata_cold, snapdata_tertiary=None, snapdata_quaternary=None):
    """2×3 panel figure: volume-averaged properties vs lookback time for all starburst gas prescriptions."""
    print('Plot 3: Evolution comparison (2x3)')

    age_universe = _lookback_time_Gyr(1100)   # proxy for t=0 Gyr
    snaps_h2   = sorted(snapdata_h2.keys())
    snaps_cold = sorted(snapdata_cold.keys())

    # --- SFRD observational compilation (Croton+2006 subset) ---
    _obs_sfrd = np.array([
        [0.000, 0.0158, 0.0251, 0.0100],
        [0.150, 0.0174, 0.0182, 0.0166],
        [0.625, 0.0275, 0.0331, 0.0229],
        [0.825, 0.0550, 0.0776, 0.0389],
        [1.250, 0.0468, 0.0661, 0.0331],
        [1.750, 0.0562, 0.0398, 0.0794],
        [2.750, 0.0794, 0.0562, 0.1122],
        [4.000, 0.0309, 0.0490, 0.0195],
    ])  # cols: z, sfrd, +err, -err  (Msun/yr/Mpc^3)

    def _build_series(snapdata, snaps):
        """Return arrays of (lookback_time, quantity) for a set of snapshots."""
        lbt, sfrd, bulge_sfrd, fburst, smd, bmd, cgd, mbmd, ibmd = [], [], [], [], [], [], [], [], []
        for snap in snaps:
            d = snapdata[snap]
            if 'StellarMass' not in d or len(d['StellarMass']) == 0:
                continue
            z = REDSHIFTS[snap]
            t = _lookback_time_Gyr(z)

            sfr_total      = np.asarray(d['SfrDisk']) + np.asarray(d['SfrBulge'])
            sfr_bulge      = np.asarray(d['SfrBulge'])
            total_sfr      = np.sum(sfr_total)
            total_bsfr     = np.sum(sfr_bulge)
            total_stars    = np.sum(d['StellarMass'])
            total_bulge    = np.sum(d['BulgeMass'])
            total_cold     = np.sum(d['ColdGas'])
            total_mbulge   = np.sum(d.get('MergerBulgeMass',   np.zeros(1)))
            total_ibulge   = np.sum(d.get('InstabilityBulgeMass', np.zeros(1)))

            lbt.append(t)
            sfrd.append(total_sfr    / VOLUME if total_sfr    > 0 else np.nan)
            bulge_sfrd.append(total_bsfr  / VOLUME if total_bsfr  > 0 else np.nan)
            fburst.append(total_bsfr / total_sfr if total_sfr > 0 else np.nan)
            smd.append(total_stars   / VOLUME if total_stars  > 0 else np.nan)
            bmd.append(total_bulge   / VOLUME if total_bulge  > 0 else np.nan)
            cgd.append(total_cold    / VOLUME if total_cold   > 0 else np.nan)
            mbmd.append(total_mbulge / VOLUME if total_mbulge > 0 else np.nan)
            ibmd.append(total_ibulge / VOLUME if total_ibulge > 0 else np.nan)

        return (np.array(lbt), np.array(sfrd), np.array(bulge_sfrd),
                np.array(fburst), np.array(smd), np.array(bmd), np.array(cgd),
                np.array(mbmd), np.array(ibmd))

    lbt_h2,   sfrd_h2,   bsfrd_h2,   fb_h2,   smd_h2,   bmd_h2,   cgd_h2,   mbmd_h2,   ibmd_h2   = _build_series(snapdata_h2,   snaps_h2)
    lbt_cold, sfrd_cold, bsfrd_cold, fb_cold, smd_cold, bmd_cold, cgd_cold, mbmd_cold, ibmd_cold = _build_series(snapdata_cold, snaps_cold)

    has_tertiary = snapdata_tertiary is not None and len(snapdata_tertiary) > 0
    if has_tertiary:
        snaps_tert = sorted(snapdata_tertiary.keys())
        lbt_tert, sfrd_tert, bsfrd_tert, fb_tert, smd_tert, bmd_tert, cgd_tert, mbmd_tert, ibmd_tert = _build_series(snapdata_tertiary, snaps_tert)

    has_quaternary = snapdata_quaternary is not None and len(snapdata_quaternary) > 0
    if has_quaternary:
        snaps_quat = sorted(snapdata_quaternary.keys())
        lbt_quat, sfrd_quat, bsfrd_quat, fb_quat, smd_quat, bmd_quat, cgd_quat, mbmd_quat, ibmd_quat = _build_series(snapdata_quaternary, snaps_quat)

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))

    C_H2   = PRIMARY_COLOR
    C_COLD = SECONDARY_COLOR
    C_TERT = TERTIARY_COLOR
    C_QUAT = QUATERNARY_COLOR
    LW = 2.2

    xlim = (0, age_universe * 0.85)

    def _plot_panel(ax, ylabel, title, log=True, ylim=None, obs=None, **series):
        for (lbt, y), color, label, ls in series['runs']:
            mask = np.isfinite(y)
            yp = np.log10(y[mask]) if log else y[mask]
            ax.plot(lbt[mask], yp, color=color, lw=LW, label=label, ls=ls)
        if obs is not None:
            z_obs, val_obs, hi_obs, lo_obs = obs
            t_obs = _lookback_time_Gyr(z_obs)
            y_obs = np.log10(val_obs) if log else val_obs
            ye_lo = np.abs(np.log10(val_obs) - np.log10(lo_obs)) if log else val_obs - lo_obs
            ye_hi = np.abs(np.log10(hi_obs) - np.log10(val_obs)) if log else hi_obs - val_obs
            ax.errorbar(t_obs, y_obs, yerr=[ye_lo, ye_hi],
                        fmt='o', color='dimgray', ms=4, lw=1.2, alpha=0.6,
                        label='Observations', zorder=0)
        ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        ax.set_xlabel('Lookback time (Gyr)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        _redshift_axis(ax, age_universe)

    def _runs(ya, yb, ytert=None, yquat=None):
        runs = [
            ((lbt_h2,   ya), C_H2,   PRIMARY_LABEL,    PRIMARY_LS),
            ((lbt_cold, yb), C_COLD, SECONDARY_LABEL,  SECONDARY_LS),
        ]
        if has_tertiary and ytert is not None:
            runs.append(((lbt_tert, ytert), C_TERT, TERTIARY_LABEL,   TERTIARY_LS))
        if has_quaternary and yquat is not None:
            runs.append(((lbt_quat, yquat), C_QUAT, QUATERNARY_LABEL, QUATERNARY_LS))
        return {'runs': runs}

    obs_sfrd = (_obs_sfrd[:, 0], _obs_sfrd[:, 1], _obs_sfrd[:, 2], _obs_sfrd[:, 3])

    _plot_panel(axes[0, 0],
                r'$\log_{10}\,\dot{\rho}_*\ [M_\odot\,\mathrm{yr}^{-1}\,\mathrm{Mpc}^{-3}]$',
                'Total SFRD', ylim=(-3.0, -0.5), obs=obs_sfrd,
                **_runs(sfrd_h2, sfrd_cold,
                        sfrd_tert if has_tertiary else None,
                        sfrd_quat if has_quaternary else None))

    _plot_panel(axes[0, 1],
                r'$\log_{10}\,\dot{\rho}_{*,\mathrm{bulge}}\ [M_\odot\,\mathrm{yr}^{-1}\,\mathrm{Mpc}^{-3}]$',
                'Bulge SFRD', ylim=(-4.5, -1.0),
                **_runs(bsfrd_h2, bsfrd_cold,
                        bsfrd_tert if has_tertiary else None,
                        bsfrd_quat if has_quaternary else None))

    _plot_panel(axes[0, 2],
                r'$f_\mathrm{burst} = \dot{M}_{*,\mathrm{bulge}}\,/\,\dot{M}_{*,\mathrm{total}}$',
                'Burst SFR fraction', log=False, ylim=(0, 1),
                **_runs(fb_h2, fb_cold,
                        fb_tert if has_tertiary else None,
                        fb_quat if has_quaternary else None))

    _plot_panel(axes[0, 3],
                r'$\log_{10}\,\rho_*\ [M_\odot\,\mathrm{Mpc}^{-3}]$',
                'Stellar mass density', ylim=(6.0, 9.0),
                **_runs(smd_h2, smd_cold,
                        smd_tert if has_tertiary else None,
                        smd_quat if has_quaternary else None))

    _plot_panel(axes[1, 0],
                r'$\log_{10}\,\rho_\mathrm{bulge,\,merger}\ [M_\odot\,\mathrm{Mpc}^{-3}]$',
                'Merger bulge mass density', ylim=(4.0, 8.5),
                **_runs(mbmd_h2, mbmd_cold,
                        mbmd_tert if has_tertiary else None,
                        mbmd_quat if has_quaternary else None))

    _plot_panel(axes[1, 1],
                r'$\log_{10}\,\rho_\mathrm{bulge,\,instab}\ [M_\odot\,\mathrm{Mpc}^{-3}]$',
                'Instability bulge mass density', ylim=(4.0, 8.5),
                **_runs(ibmd_h2, ibmd_cold,
                        ibmd_tert if has_tertiary else None,
                        ibmd_quat if has_quaternary else None))

    _plot_panel(axes[1, 2],
                r'$\log_{10}\,\rho_\mathrm{bulge,\,total}\ [M_\odot\,\mathrm{Mpc}^{-3}]$',
                'Total bulge mass density', ylim=(5.0, 9.0),
                **_runs(bmd_h2, bmd_cold,
                        bmd_tert if has_tertiary else None,
                        bmd_quat if has_quaternary else None))

    _plot_panel(axes[1, 3],
                r'$\log_{10}\,\rho_\mathrm{cold}\ [M_\odot\,\mathrm{Mpc}^{-3}]$',
                'Cold gas density', ylim=(5.0, 9.5),
                **_runs(cgd_h2, cgd_cold,
                        cgd_tert if has_tertiary else None,
                        cgd_quat if has_quaternary else None))

    axes[0, 0].legend(fontsize=9, frameon=False)

    # fig.suptitle('Starburst gas prescription: H2 vs cold gas', fontsize=13, y=1.01)
    fig.tight_layout()

    outfile = os.path.join(OUTPUT_DIR, f'Evolution_starburst_comparison{OUTPUT_FORMAT}')
    fig.savefig(outfile, bbox_inches='tight')
    print(f'  Saved {outfile}')
    plt.close(fig)

def plot_4_smf_z10(snapdata_h2, snapdata_cold, snapdata_tertiary=None, snapdata_quaternary=None):
    """Stellar mass function at z=10 for all models.

    Expects `snapdata_*` to be dicts mapping snapshot index -> property dict
    as returned by `load_snapshots`.
    """

    print('Plot 4: Stellar mass function at z=10')

    snap_z10 = SNAP_Z10

    # Extract snapshot dictionaries (may be missing)
    d_h2 = snapdata_h2.get(snap_z10, {}) if snapdata_h2 else {}
    d_cold = snapdata_cold.get(snap_z10, {}) if snapdata_cold else {}
    d_tert = snapdata_tertiary.get(snap_z10, {}) if snapdata_tertiary else {}
    d_quat = snapdata_quaternary.get(snap_z10, {}) if snapdata_quaternary else {}

    # Row snapshots: z=10, z=5, z=1, z=0
    z_snaps = [SNAP_Z13, SNAP_Z12, SNAP_Z11, SNAP_Z10]
    z_labels = [13, 12, 11, 10]

    binwidth = 0.2

    # Create 4x3 grid: rows are redshifts, columns are (SMF, metallicity, SFR)
    fig, axes = plt.subplots(len(z_snaps), 3, figsize=(18, 4 * len(z_snaps)), sharex='col')

    for row, snap in enumerate(z_snaps):
        # Extract snapshot dictionaries (may be missing)
        d_h2 = snapdata_h2.get(snap, {}) if snapdata_h2 else {}
        d_cold = snapdata_cold.get(snap, {}) if snapdata_cold else {}
        d_tert = snapdata_tertiary.get(snap, {}) if snapdata_tertiary else {}
        d_quat = snapdata_quaternary.get(snap, {}) if snapdata_quaternary else {}

        models = [
            (PRIMARY_LABEL,    d_h2,   PRIMARY_COLOR,    VOLUME,            PRIMARY_LS),
            (SECONDARY_LABEL,  d_cold, SECONDARY_COLOR,  SECONDARY_VOLUME,  SECONDARY_LS),
        ]
        if d_tert:
            models.append((TERTIARY_LABEL,   d_tert, TERTIARY_COLOR,   TERTIARY_VOLUME,   TERTIARY_LS))
        if d_quat:
            models.append((QUATERNARY_LABEL, d_quat, QUATERNARY_COLOR, QUATERNARY_VOLUME, QUATERNARY_LS))

        def _log10_pos(x):
            x = np.asarray(x, dtype=float)
            ok = np.isfinite(x) & (x > 0.0)
            return np.log10(x[ok])

        # Collect series of log10(stellar mass)
        series = []
        for name, d, color, vol, ls in models:
            vals = _log10_pos(d.get('StellarMass', np.array([]))) if d else np.array([])
            series.append((name, vals, color, vol, ls))

        all_vals = [v for (_n, v, _c, _v, _ls) in series if v.size > 0]
        if not all_vals:
            print(f'  No stellar mass data at snap {snap}, skipping row.')
            continue

        combined = np.concatenate(all_vals)
        mi = np.floor(np.min(combined)) - 1
        ma = np.ceil(np.max(combined)) + 1
        mass_bins = np.arange(mi, ma + binwidth, binwidth)

        ax_smf = axes[row, 0]
        ax_met = axes[row, 1]
        ax_sfr = axes[row, 2]

        # SMF
        for name, vals, color, vol, ls in series:
            if vals.size == 0:
                continue
            centers, phi, _ = mass_function(vals, vol, binwidth=binwidth, mass_range=(mi, ma))
            ax_smf.plot(centers, phi, color=color, ls=ls, lw=2.6, alpha=0.95, label=name if row == 0 else None)

        # annotate row with redshift label on top-right of first panel
        z_val = REDSHIFTS[snap] if isinstance(REDSHIFTS, (list, np.ndarray)) and snap < len(REDSHIFTS) else z_labels[row]
        ax_smf.text(0.98, 0.9, f'z={z_val:.2f}', transform=ax_smf.transAxes, fontsize=10, va='top', ha='right')

        ax_smf.set_xlabel(r'$\log_{10}\, m_{\ast}\,[M_\odot]$' if row == len(z_snaps) - 1 else '')
        ax_smf.set_ylabel(r'$\log_{10}\,\Phi\ [\mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1}]$')
        ax_smf.set_xlim(6.5, ma)
        ax_smf.grid(True, alpha=0.25)
        if row == 0:
            ax_smf.legend(loc='lower right', frameon=False, fontsize=9)

        # Metallicity median vs stellar mass
        for name, d, color, vol, ls in models:
            if not d:
                continue
            mstar = np.asarray(d.get('StellarMass', np.array([])), dtype=float)
            if mstar.size == 0:
                continue

            metals_cold = d.get('MetalsColdGas', None)
            coldgas = d.get('ColdGas', None)
            plotted = False
            if metals_cold is not None and coldgas is not None:
                metals = np.asarray(metals_cold, dtype=float)
                cold = np.asarray(coldgas, dtype=float)
                n = min(len(mstar), len(metals), len(cold))
                if n > 0:
                    mstar_n = mstar[:n]
                    metals_n = metals[:n]
                    cold_n = cold[:n]
                    mask = np.isfinite(metals_n) & np.isfinite(cold_n) & (cold_n > 0.0) & (mstar_n > 0.0)
                    if np.any(mask):
                        x_plot = np.log10(mstar_n[mask])
                        y_plot = metallicity_12logOH(metals_n[mask], cold_n[mask])
                        plot_binned_median_1sigma(
                            ax_met,
                            x_plot,
                            y_plot,
                            mass_bins,
                            color=color,
                            label=name if row == 0 else None,
                            alpha=0.0,
                            lw=2.6,
                            ls=ls,
                            min_count=1,
                        )
                        plotted = True

            if not plotted:
                metals_stellar = d.get('MetalsStellarMass', None)
                if metals_stellar is None:
                    continue
                metals_stellar = np.asarray(metals_stellar, dtype=float)
                n = min(len(mstar), len(metals_stellar))
                if n == 0:
                    continue
                mstar_n = mstar[:n]
                metals_stellar_n = metals_stellar[:n]
                mask = np.isfinite(metals_stellar_n) & (mstar_n > 0.0)
                if not np.any(mask):
                    continue
                x_plot = np.log10(mstar_n[mask])
                met_logz = stellar_metallicity(metals_stellar_n[mask], mstar_n[mask])
                y_plot = met_logz + 8.69
                plot_binned_median_1sigma(
                    ax_met,
                    x_plot,
                    y_plot,
                    mass_bins,
                    color=color,
                    label=(name + ' (stellar)') if row == 0 else None,
                    alpha=0.0,
                    lw=2.6,
                    ls='--',
                    min_count=1,
                )

        ax_met.set_xlabel(r'$\log_{10}\, m_{\ast}\,[M_\odot]$' if row == len(z_snaps) - 1 else '')
        ax_met.set_ylabel(r'$12 + \log_{10}(\mathrm{O/H})$')
        ax_met.set_xlim(6.5, ma)
        ax_met.set_ylim(6.0, 9.0)
        ax_met.grid(True, alpha=0.25)

        # Median SFR vs stellar mass
        for name, d, color, vol, ls in models:
            if not d:
                continue
            mstar = np.asarray(d.get('StellarMass', np.array([])), dtype=float)
            if mstar.size == 0:
                continue
            sfr = np.asarray(d.get('SfrDisk', np.zeros_like(mstar)), dtype=float) + np.asarray(d.get('SfrBulge', np.zeros_like(mstar)), dtype=float)
            n = min(len(mstar), len(sfr))
            if n == 0:
                continue
            mstar_n = mstar[:n]
            sfr_n = sfr[:n]
            mask = np.isfinite(mstar_n) & (mstar_n > 0.0) & np.isfinite(sfr_n) & (sfr_n > 0.0)
            if not np.any(mask):
                continue
            x_plot = np.log10(mstar_n[mask])
            y_plot = np.log10(sfr_n[mask])
            plot_binned_median_1sigma(
                ax_sfr,
                x_plot,
                y_plot,
                mass_bins,
                color=color,
                label=name if row == 0 else None,
                alpha=0.0,
                lw=2.6,
                ls=ls,
                min_count=1,
            )

        ax_sfr.set_xlabel(r'$\log_{10}\, m_{\ast}\,[M_\odot]$' if row == len(z_snaps) - 1 else '')
        ax_sfr.set_ylabel(r'$\log_{10}\, \mathrm{SFR}\,[M_\odot\,\mathrm{yr}^{-1}]$')
        ax_sfr.set_xlim(6.5, ma)
        ax_sfr.set_ylim(-5.0, 3.0)
        ax_sfr.grid(True, alpha=0.25)

    # Remove subplot subtitles (titles were not set) and save
    fig.tight_layout()
    outfile = os.path.join(OUTPUT_DIR, f'SMF_z_comparison_rows{OUTPUT_FORMAT}')
    fig.savefig(outfile, bbox_inches='tight')
    print(f'  Saved {outfile}')
    plt.close(fig)

def plot_5_fH2_redshift(snapdata_h2, snapdata_cold, snapdata_tertiary=None, snapdata_quaternary=None):
    """Molecular hydrogen fraction (H2gas/ColdGas) vs stellar mass at z=0,1,2,3,4,5.

    Six panels arranged in a 3x2 grid, one per redshift.
    """

    print('Plot 5: Molecular hydrogen fraction vs stellar mass at increasing redshift')

    z_snaps  = [SNAP_Z0, SNAP_Z2, SNAP_Z3, SNAP_Z5, SNAP_Z7, SNAP_Z10, SNAP_Z12, SNAP_Z13]
    z_labels = [0, 2, 3, 5, 7, 10, 12, 13]

    nrows, ncols = 4, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 12), sharey=True)
    axes_flat = axes.flatten()

    binwidth = 0.1
    mass_range = (7.0, 12.5)
    mass_bins = np.arange(mass_range[0], mass_range[1] + binwidth, binwidth)

    for idx, (snap, z_label) in enumerate(zip(z_snaps, z_labels)):
        ax = axes_flat[idx]

        d_h2   = snapdata_h2.get(snap, {})        if snapdata_h2        else {}
        d_cold = snapdata_cold.get(snap, {})       if snapdata_cold       else {}
        d_tert = snapdata_tertiary.get(snap, {})   if snapdata_tertiary   else {}
        d_quat = snapdata_quaternary.get(snap, {}) if snapdata_quaternary else {}

        models = [
            (PRIMARY_LABEL,    d_h2,   PRIMARY_COLOR,    VOLUME,            PRIMARY_LS),
            (SECONDARY_LABEL,  d_cold, SECONDARY_COLOR,  SECONDARY_VOLUME,  SECONDARY_LS),
        ]
        if d_tert:
            models.append((TERTIARY_LABEL,   d_tert, TERTIARY_COLOR,   TERTIARY_VOLUME,   TERTIARY_LS))
        if d_quat:
            models.append((QUATERNARY_LABEL, d_quat, QUATERNARY_COLOR, QUATERNARY_VOLUME, QUATERNARY_LS))

        for name, d, color, _vol, ls in models:
            if not d:
                continue
            mstar  = np.asarray(d.get('StellarMass', np.array([])), dtype=float)
            h2gas  = np.asarray(d.get('H2gas',        np.array([])), dtype=float)
            coldgas = np.asarray(d.get('ColdGas',     np.array([])), dtype=float)
            n = min(len(mstar), len(h2gas), len(coldgas))
            if n == 0:
                continue
            mstar   = mstar[:n]
            h2gas   = h2gas[:n]
            coldgas = coldgas[:n]

            mask = (mstar > 0) & (coldgas > 0) & np.isfinite(mstar) & np.isfinite(h2gas) & np.isfinite(coldgas)
            if not np.any(mask):
                continue

            x = np.log10(mstar[mask])
            y = np.clip(h2gas[mask] / coldgas[mask], 0.0, 1.0)

            plot_binned_median_1sigma(
                ax, x, y, mass_bins,
                color=color,
                label=name if idx == 0 else None,
                alpha=0.15,
                lw=2.6,
                ls=ls,
                min_count=1,
            )

        z_val = REDSHIFTS[snap] if isinstance(REDSHIFTS, (list, np.ndarray)) and snap < len(REDSHIFTS) else z_label
        ax.text(0.97, 0.95, f'z = {z_val:.2f}', transform=ax.transAxes,
                fontsize=10, va='top', ha='right')

        is_bottom = (idx >= (nrows - 1) * ncols)
        is_left   = (idx % ncols == 0)

        ax.set_xlim(*mass_range)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel(r'$\log_{10}\, m_{\ast}\,[M_\odot]$' if is_bottom else '')
        ax.set_ylabel(r'$f_{\rm H_2} = M_{\rm H_2} / M_{\rm cold}$' if is_left else '')
        ax.grid(True, alpha=0.25)

        if idx == 0:
            ax.legend(loc='upper left', frameon=False, fontsize=8)

    fig.tight_layout()
    outfile = os.path.join(OUTPUT_DIR, f'fH2_redshift_panels{OUTPUT_FORMAT}')
    fig.savefig(outfile, bbox_inches='tight')
    print(f'  Saved {outfile}')
    plt.close(fig)


def plot_6_satellite_smf_redshift(snapdata_h2, snapdata_cold, snapdata_tertiary=None, snapdata_quaternary=None):
    """Satellite stellar mass function (Type>0 only) at 10 redshifts from z=0 to z~20.

    2×5 grid; each panel shows all available models. The SMF is
    log10(number density per dex per Mpc^3).
    """
    print('Plot 6: Satellite SMF at z=0–10 (2×5)')

    z_snaps  = [SNAP_Z0, SNAP_Z1, SNAP_Z2, SNAP_Z3, SNAP_Z4,
                SNAP_Z5, SNAP_Z6, SNAP_Z7, SNAP_Z8, SNAP_Z10]
    z_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]

    binwidth = 0.2
    nrows, ncols = 2, 5
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, 9))
    axes_flat = axes.flatten()

    for idx, (snap, z_label) in enumerate(zip(z_snaps, z_labels)):
        ax = axes_flat[idx]

        d_h2   = snapdata_h2.get(snap, {})        if snapdata_h2        else {}
        d_cold = snapdata_cold.get(snap, {})       if snapdata_cold       else {}
        d_tert = snapdata_tertiary.get(snap, {})   if snapdata_tertiary   else {}
        d_quat = snapdata_quaternary.get(snap, {}) if snapdata_quaternary else {}

        models = [
            (PRIMARY_LABEL,    d_h2,   PRIMARY_COLOR,    VOLUME,            PRIMARY_LS),
            (SECONDARY_LABEL,  d_cold, SECONDARY_COLOR,  SECONDARY_VOLUME,  SECONDARY_LS),
        ]
        if d_tert:
            models.append((TERTIARY_LABEL,   d_tert, TERTIARY_COLOR,   TERTIARY_VOLUME,   TERTIARY_LS))
        if d_quat:
            models.append((QUATERNARY_LABEL, d_quat, QUATERNARY_COLOR, QUATERNARY_VOLUME, QUATERNARY_LS))

        z_val = REDSHIFTS[snap] if isinstance(REDSHIFTS, (list, np.ndarray)) and snap < len(REDSHIFTS) else z_label
        is_bottom = idx >= (nrows - 1) * ncols
        is_left   = idx % ncols == 0

        series = []
        for name, d, color, vol, ls in models:
            if not d or 'StellarMass' not in d or 'Type' not in d:
                continue
            mstar = np.asarray(d['StellarMass'], dtype=float)
            gtype = np.asarray(d['Type'])
            sat   = mstar[gtype > 0]
            ok    = np.isfinite(sat) & (sat > 0.0)
            vals  = np.log10(sat[ok]) if ok.any() else np.array([])
            series.append((name, vals, color, vol, ls))

        all_vals = [v for (_, v, _, _, _) in series if v.size > 0]

        ax.text(0.97, 0.95, f'z = {z_val:.2f}', transform=ax.transAxes,
                fontsize=10, va='top', ha='right')
        ax.set_xlabel(r'$\log_{10}\, m_{\ast}\,[M_\odot]$' if is_bottom else '')
        ax.set_ylabel(r'$\log_{10}\,\Phi_{\rm sat}\ [\mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1}]$' if is_left else '')
        ax.grid(True, alpha=0.25)

        if not all_vals:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue

        combined = np.concatenate(all_vals)
        mi = max(np.floor(np.min(combined)) - 1, 4.0)
        ma = min(np.ceil(np.max(combined)) + 1, 13.0)

        for name, vals, color, vol, ls in series:
            if vals.size == 0:
                continue
            centers, phi, _ = mass_function(vals, vol, binwidth=binwidth, mass_range=(mi, ma))
            ax.plot(centers, phi, color=color, ls=ls, lw=2.4, alpha=0.9,
                    label=name if idx == 0 else None)

        ax.set_xlim(mi, ma)

        if idx == 0:
            ax.legend(loc='lower left', frameon=False, fontsize=9)

    fig.tight_layout()
    outfile = os.path.join(OUTPUT_DIR, f'Satellite_SMF_redshift{OUTPUT_FORMAT}')
    fig.savefig(outfile, bbox_inches='tight')
    print(f'  Saved {outfile}')
    plt.close(fig)


def plot_7_smf_centrals_satellites(snapdata_h2, snapdata_cold, snapdata_tertiary=None, snapdata_quaternary=None):
    """Stellar mass function split into centrals and satellites at z=0–10.

    2×5 grid of redshift panels. Each panel plots all models twice: once for
    centrals (solid) and once for satellites (dashed). No total line is drawn.
    Legend in the z=0 panel shows model colours and type line styles separately.
    """
    print('Plot 7: SMF centrals vs satellites at z=0–10 (2×5)')

    z_snaps  = [SNAP_Z0, SNAP_Z1, SNAP_Z2, SNAP_Z3, SNAP_Z4,
                SNAP_Z5, SNAP_Z6, SNAP_Z7, SNAP_Z8, SNAP_Z10]
    z_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]

    LS_CEN = '-'
    LS_SAT = '--'
    binwidth = 0.2
    nrows, ncols = 2, 5
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, 9))
    axes_flat = axes.flatten()

    for idx, (snap, z_label) in enumerate(zip(z_snaps, z_labels)):
        ax = axes_flat[idx]

        d_h2   = snapdata_h2.get(snap, {})        if snapdata_h2        else {}
        d_cold = snapdata_cold.get(snap, {})       if snapdata_cold       else {}
        d_tert = snapdata_tertiary.get(snap, {})   if snapdata_tertiary   else {}
        d_quat = snapdata_quaternary.get(snap, {}) if snapdata_quaternary else {}

        models = [
            (PRIMARY_LABEL,    d_h2,   PRIMARY_COLOR,    VOLUME),
            (SECONDARY_LABEL,  d_cold, SECONDARY_COLOR,  SECONDARY_VOLUME),
        ]
        if d_tert:
            models.append((TERTIARY_LABEL,   d_tert, TERTIARY_COLOR,   TERTIARY_VOLUME))
        if d_quat:
            models.append((QUATERNARY_LABEL, d_quat, QUATERNARY_COLOR, QUATERNARY_VOLUME))

        z_val     = REDSHIFTS[snap] if isinstance(REDSHIFTS, (list, np.ndarray)) and snap < len(REDSHIFTS) else z_label
        is_bottom = idx >= (nrows - 1) * ncols
        is_left   = idx % ncols == 0

        ax.text(0.97, 0.95, f'z = {z_val:.2f}', transform=ax.transAxes,
                fontsize=10, va='top', ha='right')
        ax.set_xlabel(r'$\log_{10}\, m_{\ast}\,[M_\odot]$' if is_bottom else '')
        ax.set_ylabel(r'$\log_{10}\,\Phi\ [\mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1}]$' if is_left else '')
        ax.grid(True, alpha=0.25)

        has_data = False
        all_vals = []

        for name, d, color, vol in models:
            if not d or 'StellarMass' not in d or 'Type' not in d:
                continue
            mstar = np.asarray(d['StellarMass'], dtype=float)
            gtype = np.asarray(d['Type'])

            for mask, ls in [(gtype == 0, LS_CEN), (gtype > 0, LS_SAT)]:
                sub = mstar[mask]
                ok  = np.isfinite(sub) & (sub > 0.0)
                if not ok.any():
                    continue
                vals = np.log10(sub[ok])
                all_vals.append(vals)

        if not all_vals:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue

        combined = np.concatenate(all_vals)
        mi = max(np.floor(np.min(combined)) - 1, 4.0)
        ma = min(np.ceil(np.max(combined)) + 1, 13.0)

        for name, d, color, vol in models:
            if not d or 'StellarMass' not in d or 'Type' not in d:
                continue
            mstar = np.asarray(d['StellarMass'], dtype=float)
            gtype = np.asarray(d['Type'])

            for mask, ls in [(gtype == 0, LS_CEN), (gtype > 0, LS_SAT)]:
                sub = mstar[mask]
                ok  = np.isfinite(sub) & (sub > 0.0)
                if not ok.any():
                    continue
                vals = np.log10(sub[ok])
                centers, phi, _ = mass_function(vals, vol, binwidth=binwidth, mass_range=(mi, ma))
                ax.plot(centers, phi, color=color, ls=ls, lw=2.4, alpha=0.9)

        ax.set_xlim(mi, ma)

        if idx == 0:
            from matplotlib.lines import Line2D
            model_handles = [Line2D([0], [0], color=color, lw=2.4, ls='-', label=name)
                             for name, _, color, _ in models]
            type_handles = [
                Line2D([0], [0], color='0.3', lw=2.4, ls=LS_CEN, label='Centrals'),
                Line2D([0], [0], color='0.3', lw=2.4, ls=LS_SAT, label='Satellites'),
            ]
            ax.legend(handles=model_handles + type_handles,
                      loc='lower left', frameon=False, fontsize=9)

    fig.tight_layout()
    outfile = os.path.join(OUTPUT_DIR, f'SMF_centrals_satellites{OUTPUT_FORMAT}')
    fig.savefig(outfile, bbox_inches='tight')
    print(f'  Saved {outfile}')
    plt.close(fig)


def plot_8_central_satellite_counts(snapdata_h2, snapdata_cold, snapdata_tertiary=None, snapdata_quaternary=None):
    """Raw number-count histograms split by galaxy type at many redshifts.

    2×6 grid: top row = centrals, bottom row = satellites, one column per
    redshift (z=0,1,2,3,5,7). All models are overlaid per panel.
    y-axis is raw count N (log scale); no volume normalisation.
    """
    print('Plot 8: Central/satellite number counts at many redshifts (2×6)')

    z_snaps  = [SNAP_Z0, SNAP_Z1, SNAP_Z2, SNAP_Z3, SNAP_Z5, SNAP_Z7]
    ncols    = len(z_snaps)
    binwidth = 0.2

    fig, axes = plt.subplots(2, ncols, figsize=(24, 9))

    _type_rows = [
        ('Centrals',   lambda t: t == 0),
        ('Satellites', lambda t: t > 0),
    ]

    for col, snap in enumerate(z_snaps):
        d_h2   = snapdata_h2.get(snap,   {}) if snapdata_h2        else {}
        d_cold = snapdata_cold.get(snap,  {}) if snapdata_cold       else {}
        d_tert = snapdata_tertiary.get(snap, {})  if snapdata_tertiary  else {}
        d_quat = snapdata_quaternary.get(snap, {}) if snapdata_quaternary else {}

        models = [
            (PRIMARY_LABEL,   d_h2,   PRIMARY_COLOR,   PRIMARY_LS),
            (SECONDARY_LABEL, d_cold, SECONDARY_COLOR, SECONDARY_LS),
        ]
        if d_tert:
            models.append((TERTIARY_LABEL,   d_tert, TERTIARY_COLOR,   TERTIARY_LS))
        if d_quat:
            models.append((QUATERNARY_LABEL, d_quat, QUATERNARY_COLOR, QUATERNARY_LS))

        z_val = REDSHIFTS[snap] if isinstance(REDSHIFTS, (list, np.ndarray)) and snap < len(REDSHIFTS) else col

        # Shared x-range across all models and both types for this redshift
        all_log_m = []
        for _n, d, _c, _ls in models:
            if not d or 'StellarMass' not in d:
                continue
            m = np.asarray(d['StellarMass'], dtype=float)
            ok = np.isfinite(m) & (m > 0.0)
            if ok.any():
                all_log_m.append(np.log10(m[ok]))

        if not all_log_m:
            for row in range(2):
                axes[row, col].text(0.5, 0.5, 'No data', ha='center', va='center',
                                    transform=axes[row, col].transAxes)
            continue

        combined = np.concatenate(all_log_m)
        mi = max(np.floor(np.min(combined)) - 1, 4.0)
        ma = min(np.ceil(np.max(combined)) + 1, 13.0)
        bins = np.arange(mi, ma + binwidth, binwidth)

        for row, (type_label, type_mask) in enumerate(_type_rows):
            ax = axes[row, col]
            is_left   = col == 0
            is_bottom = row == 1

            ax.text(0.97, 0.95, f'z = {z_val:.2f}',
                    transform=ax.transAxes, fontsize=10, va='top', ha='right')
            ax.set_xlabel(r'$\log_{10}\, m_{\ast}\ [M_\odot]$' if is_bottom else '')
            ax.set_ylabel(f'{type_label}\nN' if is_left else '')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.25)
            ax.set_xlim(mi, ma)

            for name, d, color, ls in models:
                if not d or 'StellarMass' not in d or 'Type' not in d:
                    continue
                mstar = np.asarray(d['StellarMass'], dtype=float)
                gtype = np.asarray(d['Type'])
                sub   = mstar[type_mask(gtype)]
                ok    = np.isfinite(sub) & (sub > 0.0)
                if not ok.any():
                    continue
                counts, edges = np.histogram(np.log10(sub[ok]), bins=bins)
                c = counts.astype(float)
                c[c <= 0] = np.nan
                ax.bar(edges[:-1], c, width=binwidth, align='edge',
                       color=color, alpha=0.4, edgecolor=color, linewidth=0.8,
                       label=name if (row == 0 and col == 0) else None)

    axes[0, 0].legend(loc='upper left', frameon=False, fontsize=9)

    fig.tight_layout()
    outfile = os.path.join(OUTPUT_DIR, f'Central_Satellite_Counts_redshift{OUTPUT_FORMAT}')
    fig.savefig(outfile, bbox_inches='tight')
    print(f'  Saved {outfile}')
    plt.close(fig)


def plot_9_quiescent_counts(snapdata_h2, snapdata_cold, snapdata_tertiary=None, snapdata_quaternary=None):
    """Quiescent-only number-count histograms split by central/satellite at many redshifts.

    Same layout as plot_8 (2x6 grid) but only galaxies with log10(sSFR) < SSFR_CUT
    are included. Quiescent defined as log10(sSFR/yr^-1) < SSFR_CUT (-11).
    Galaxies with SFR=0 are counted as quiescent.
    """
    print(f'Plot 9: Quiescent central/satellite counts at many redshifts (2x6, sSFR < {SSFR_CUT})')

    z_snaps  = [SNAP_Z0, SNAP_Z1, SNAP_Z2, SNAP_Z3, SNAP_Z5, SNAP_Z7]
    ncols    = len(z_snaps)
    binwidth = 0.2

    fig, axes = plt.subplots(2, ncols, figsize=(24, 9))

    _type_rows = [
        ('Centrals',   lambda t: t == 0),
        ('Satellites', lambda t: t > 0),
    ]

    for col, snap in enumerate(z_snaps):
        d_h2   = snapdata_h2.get(snap,        {}) if snapdata_h2        else {}
        d_cold = snapdata_cold.get(snap,       {}) if snapdata_cold      else {}
        d_tert = snapdata_tertiary.get(snap,   {}) if snapdata_tertiary  else {}
        d_quat = snapdata_quaternary.get(snap, {}) if snapdata_quaternary else {}

        models = [
            (PRIMARY_LABEL,   d_h2,   PRIMARY_COLOR,   PRIMARY_LS),
            (SECONDARY_LABEL, d_cold, SECONDARY_COLOR, SECONDARY_LS),
        ]
        if d_tert:
            models.append((TERTIARY_LABEL,   d_tert, TERTIARY_COLOR,   TERTIARY_LS))
        if d_quat:
            models.append((QUATERNARY_LABEL, d_quat, QUATERNARY_COLOR, QUATERNARY_LS))

        z_val = REDSHIFTS[snap] if isinstance(REDSHIFTS, (list, np.ndarray)) and snap < len(REDSHIFTS) else col

        # Quiescent mask: SFR=0 counts as quiescent; SFR>0 requires log10(sSFR) < SSFR_CUT
        def _quiescent_mask(d):
            mstar  = np.asarray(d['StellarMass'], dtype=float)
            sfrd   = np.asarray(d.get('SfrDisk',  np.zeros(len(mstar))), dtype=float)
            sfrb   = np.asarray(d.get('SfrBulge', np.zeros(len(mstar))), dtype=float)
            sfr    = sfrd + sfrb
            valid  = np.isfinite(mstar) & (mstar > 0.0)
            zero   = valid & (sfr <= 0.0)
            active = valid & (sfr > 0.0)
            lssfr  = np.full(len(mstar), np.nan)
            lssfr[active] = np.log10(sfr[active] / mstar[active])
            return zero | (active & (lssfr < SSFR_CUT))

        # Shared x-range from all quiescent stellar masses across all models
        all_log_m = []
        for _n, d, _c, _ls in models:
            if not d or 'StellarMass' not in d:
                continue
            q = _quiescent_mask(d)
            m = np.asarray(d['StellarMass'], dtype=float)[q]
            if m.size:
                all_log_m.append(np.log10(m))

        if not all_log_m:
            for row in range(2):
                axes[row, col].text(0.5, 0.5, 'No data', ha='center', va='center',
                                    transform=axes[row, col].transAxes)
            continue

        combined = np.concatenate(all_log_m)
        mi = max(np.floor(np.min(combined)) - 1, 4.0)
        ma = min(np.ceil(np.max(combined)) + 1, 13.0)
        bins = np.arange(mi, ma + binwidth, binwidth)

        for row, (type_label, type_mask) in enumerate(_type_rows):
            ax = axes[row, col]
            is_left   = col == 0
            is_bottom = row == 1

            ax.text(0.97, 0.95, f'z = {z_val:.2f}',
                    transform=ax.transAxes, fontsize=10, va='top', ha='right')
            ax.set_xlabel(r'$\log_{10}\, m_{\ast}\ [M_\odot]$' if is_bottom else '')
            ax.set_ylabel(f'{type_label}\nN (quiescent)' if is_left else '')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.25)
            ax.set_xlim(mi, ma)

            for name, d, color, ls in models:
                if not d or 'StellarMass' not in d or 'Type' not in d:
                    continue
                q     = _quiescent_mask(d)
                gtype = np.asarray(d['Type'])
                sel   = q & type_mask(gtype)
                m     = np.asarray(d['StellarMass'], dtype=float)[sel]
                ok    = np.isfinite(m) & (m > 0.0)
                if not ok.any():
                    continue
                counts, edges = np.histogram(np.log10(m[ok]), bins=bins)
                c = counts.astype(float)
                c[c <= 0] = np.nan
                ax.bar(edges[:-1], c, width=binwidth, align='edge',
                       color=color, alpha=0.4, edgecolor=color, linewidth=0.8,
                       label=name if (row == 0 and col == 0) else None)

    axes[0, 0].legend(loc='upper left', frameon=False, fontsize=9)

    fig.suptitle(r'Quiescent galaxies ($\log_{10}\,\mathrm{sSFR} < -11$)', y=1.01, fontsize=12)
    fig.tight_layout()
    outfile = os.path.join(OUTPUT_DIR, f'Quiescent_Counts_redshift{OUTPUT_FORMAT}')
    fig.savefig(outfile, bbox_inches='tight')
    print(f'  Saved {outfile}')
    plt.close(fig)


def plot_10_morphology(primary, secondary, tertiary=None, quaternary=None):
    """2×3 panel morphology diagnostics at z=0.

    Panels:
      [0,0] BulgeMass vs StellarMass
      [0,1] BulgeRadius vs StellarMass
      [0,2] DiskRadius vs StellarMass
      [1,0] DiskMass (StellarMass - BulgeMass) vs StellarMass
      [1,1] B/T (BulgeMass / StellarMass) vs StellarMass
      [1,2] SFE (StellarMass / 0.17*Mvir) vs StellarMass

    All binned-median with 16-84% shaded band. Radii shown in kpc/h.
    """
    print('Plot 10: Morphology diagnostics (2x3)')

    models = [
        (PRIMARY_LABEL,   primary,   PRIMARY_COLOR,   PRIMARY_LS),
        (SECONDARY_LABEL, secondary, SECONDARY_COLOR, SECONDARY_LS),
    ]
    if tertiary:
        models.append((TERTIARY_LABEL,   tertiary,   TERTIARY_COLOR,   TERTIARY_LS))
    if quaternary:
        models.append((QUATERNARY_LABEL, quaternary, QUATERNARY_COLOR, QUATERNARY_LS))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    _XBINS = np.arange(7.0, 12.5, 0.2)
    _XLABEL = r'$\log_{10}\, M_{\ast}\ [M_\odot]$'

    def _panel(ax, title, ylabel, xys, *, ylim=None, legend=False):
        for name, color, ls, x, y in xys:
            ok = np.isfinite(x) & np.isfinite(y)
            if not ok.any():
                continue
            plot_binned_median_1sigma(ax, x[ok], y[ok], _XBINS,
                                      color=color, label=name, ls=ls,
                                      alpha=0.20, lw=2.8, min_count=30)
        ax.set_title(title)
        ax.set_xlabel(_XLABEL)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        if ylim is not None:
            ax.set_ylim(ylim)
        if legend:
            ax.legend(loc='upper left', frameon=False, fontsize=9)

    # --- precompute per-model arrays ---
    def _log(x):
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(x > 0, np.log10(x), np.nan)

    # Panel [0,0]: BulgeMass vs StellarMass
    xys = []
    for name, d, color, ls in models:
        ms = np.asarray(d['StellarMass'], dtype=float)
        mb = np.asarray(d['BulgeMass'],   dtype=float)
        xys.append((name, color, ls, _log(ms), _log(mb)))
    _panel(axes[0, 0], 'Bulge Mass', r'$\log_{10}\, M_{\rm bulge}\ [M_\odot]$', xys, legend=True)

    # Panel [0,1]: BulgeRadius vs StellarMass  (Mpc/h × 1000 → kpc/h)
    xys = []
    for name, d, color, ls in models:
        ms = np.asarray(d['StellarMass'],  dtype=float)
        rb = np.asarray(d.get('BulgeRadius', np.zeros(len(ms))), dtype=float) * 1e3
        xys.append((name, color, ls, _log(ms), _log(rb)))
    _panel(axes[0, 1], 'Bulge Radius', r'$\log_{10}\, r_{\rm bulge}\ [{\rm kpc}\,h^{-1}]$', xys)

    # Panel [0,2]: DiskRadius vs StellarMass  (Mpc/h × 1000 → kpc/h)
    xys = []
    for name, d, color, ls in models:
        ms = np.asarray(d['StellarMass'],  dtype=float)
        rd = np.asarray(d.get('DiskRadius', np.zeros(len(ms))), dtype=float) * 1e3
        xys.append((name, color, ls, _log(ms), _log(rd)))
    _panel(axes[0, 2], 'Disk Radius', r'$\log_{10}\, r_{\rm disk}\ [{\rm kpc}\,h^{-1}]$', xys)

    # Panel [1,0]: DiskMass vs StellarMass
    xys = []
    for name, d, color, ls in models:
        ms = np.asarray(d['StellarMass'], dtype=float)
        md = ms - np.asarray(d['BulgeMass'], dtype=float)
        xys.append((name, color, ls, _log(ms), _log(md)))
    _panel(axes[1, 0], 'Disk Mass', r'$\log_{10}\, M_{\rm disk}\ [M_\odot]$', xys)

    # Panel [1,1]: B/T vs StellarMass  (linear y, 0-1)
    xys = []
    for name, d, color, ls in models:
        ms  = np.asarray(d['StellarMass'], dtype=float)
        bt  = np.asarray(d['BulgeMass'],   dtype=float) / ms
        ok  = np.isfinite(ms) & (ms > 0) & np.isfinite(bt)
        xys.append((name, color, ls, _log(ms[ok]), bt[ok]))
    _panel(axes[1, 1], 'Bulge-to-Total Ratio', r'$B/T$', xys, ylim=(0.0, 1.0))

    # Panel [1,2]: SFE = StellarMass / (0.17 * Mvir) vs StellarMass  (log y)
    xys = []
    for name, d, color, ls in models:
        ms   = np.asarray(d['StellarMass'], dtype=float)
        mv   = np.asarray(d['Mvir'],        dtype=float)
        sfe  = ms / (0.17 * mv)
        xys.append((name, color, ls, _log(ms), _log(sfe)))
    _panel(axes[1, 2], r'Star Formation Efficiency', r'$\log_{10}\,(M_{\ast} / 0.17\,M_{\rm vir})$', xys)

    fig.tight_layout()
    outfile = os.path.join(OUTPUT_DIR, f'Morphology_diagnostics{OUTPUT_FORMAT}')
    fig.savefig(outfile, bbox_inches='tight')
    print(f'  Saved {outfile}')
    plt.close(fig)


Z0_PLOTS = {
    1: plot_1_number_counts,
    2: plot_2_diagnostics,
    10: plot_10_morphology,
}

EVOLUTION_PLOTS = {
    3: plot_3_evolution,
    4: plot_4_smf_z10,
    5: plot_5_fH2_redshift,
    6: plot_6_satellite_smf_redshift,
    7: plot_7_smf_centrals_satellites,
    8: plot_8_central_satellite_counts,
    9: plot_9_quiescent_counts,
}

# Standalone plots (load their own data)
STANDALONE_PLOTS = {
   
}

ALL_PLOTS = {**Z0_PLOTS, **EVOLUTION_PLOTS, **STANDALONE_PLOTS}


def main():
    seed(SEED)
    np.random.seed(SEED)
    setup_style()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Determine which plots to generate
    if len(sys.argv) > 1:
        plot_nums = [int(x) for x in sys.argv[1:]]
    else:
        plot_nums = sorted(ALL_PLOTS.keys())

    need_z0 = any(n in Z0_PLOTS for n in plot_nums)
    need_evo = any(n in EVOLUTION_PLOTS for n in plot_nums)

    primary = secondary = tertiary = quaternary = None
    snapdata_h2 = snapdata_cold = snapdata_tertiary = snapdata_quaternary = None

    _z0_props = ['StellarMass', 'SfrDisk', 'SfrBulge',
                 'ColdGas', 'MetalsColdGas', 'H1gas', 'H2gas',
                 'BlackHoleMass', 'BulgeMass', 'MergerBulgeMass', 'InstabilityBulgeMass',
                 'Mvir', 'Vmax', 'Vvir', 'DiskRadius', 'BulgeRadius',
                 'Type', 'IntraClusterStars']

    # Load z=0 data only if needed
    if need_z0:
        print('Loading primary model from', PRIMARY_DIR)
        primary = load_model(PRIMARY_DIR,
                             particle_mass=PART_MASS,
                             mvir_particles_min=MIN_MVIR_PARTICLES)
        print(f'  {len(primary["StellarMass"]):,} galaxies loaded')

        print('Loading secondary model from', SECONDARY_DIR)
        secondary = load_model(SECONDARY_DIR,
                               snapshot=f'Snap_{SECONDARY_LAST_SNAP}',
                               properties=_z0_props,
                               mass_convert=SECONDARY_MASS_CONVERT,
                               particle_mass=SECONDARY_PART_MASS,
                               mvir_particles_min=MIN_MVIR_PARTICLES)
        print(f'  {len(secondary["StellarMass"]):,} galaxies loaded')

        if model_files_exist(TERTIARY_DIR):
            print('Loading tertiary model from', TERTIARY_DIR)
            tertiary = load_model(TERTIARY_DIR,
                                  snapshot=f'Snap_{TERTIARY_LAST_SNAP}',
                                  properties=_z0_props,
                                  mass_convert=TERTIARY_MASS_CONVERT,
                                  particle_mass=TERTIARY_PART_MASS,
                                  mvir_particles_min=MIN_MVIR_PARTICLES)
            print(f'  {len(tertiary["StellarMass"]):,} galaxies loaded')
        else:
            print(f'  Tertiary model not found at {TERTIARY_DIR}, skipping.')

        if QUATERNARY_DIR and model_files_exist(QUATERNARY_DIR):
            print('Loading quaternary model from', QUATERNARY_DIR)
            quaternary = load_model(QUATERNARY_DIR,
                                    snapshot=f'Snap_{QUATERNARY_LAST_SNAP}',
                                    properties=_z0_props,
                                    mass_convert=QUATERNARY_MASS_CONVERT,
                                    particle_mass=QUATERNARY_PART_MASS,
                                    mvir_particles_min=MIN_MVIR_PARTICLES)
            print(f'  {len(quaternary["StellarMass"]):,} galaxies loaded')
        print()

    # Load multi-snapshot data only if needed
    if need_evo:
        key_snaps = [SNAP_Z0, SNAP_Z1, SNAP_Z2, SNAP_Z3, SNAP_Z4, SNAP_Z5,
                     SNAP_Z6, SNAP_Z7, SNAP_Z8, SNAP_Z10, SNAP_Z20]
        last_snap = _primary_hdr['last_snap_nr'] if _primary_hdr else SNAP_Z0
        sfh_snaps = list(_primary_hdr['output_snaps']) if _primary_hdr else list(range(last_snap + 1))
        all_snaps = sorted(set(key_snaps + sfh_snaps))

        print(f'Loading {len(all_snaps)} snapshots from', PRIMARY_DIR)
        snapdata_h2 = load_snapshots(PRIMARY_DIR,
                                     all_snaps,
                                     particle_mass=PART_MASS,
                                     mvir_particles_min=MIN_MVIR_PARTICLES)
        print(f'  {len(snapdata_h2)} snapshots loaded')

        print(f'Loading {len(all_snaps)} snapshots from', SECONDARY_DIR)
        snapdata_cold = load_snapshots(SECONDARY_DIR,
                                       all_snaps,
                                       mass_convert=SECONDARY_MASS_CONVERT,
                                       particle_mass=SECONDARY_PART_MASS,
                                       mvir_particles_min=MIN_MVIR_PARTICLES)
        print(f'  {len(snapdata_cold)} snapshots loaded')

        if model_files_exist(TERTIARY_DIR):
            print(f'Loading {len(all_snaps)} snapshots from', TERTIARY_DIR)
            snapdata_tertiary = load_snapshots(TERTIARY_DIR,
                                               all_snaps,
                                               mass_convert=TERTIARY_MASS_CONVERT,
                                               particle_mass=TERTIARY_PART_MASS,
                                               mvir_particles_min=MIN_MVIR_PARTICLES)
            print(f'  {len(snapdata_tertiary)} snapshots loaded')
        else:
            print(f'  Tertiary model not found at {TERTIARY_DIR}, skipping.')

        if QUATERNARY_DIR and model_files_exist(QUATERNARY_DIR):
            print(f'Loading {len(all_snaps)} snapshots from', QUATERNARY_DIR)
            snapdata_quaternary = load_snapshots(QUATERNARY_DIR,
                                                 all_snaps,
                                                 mass_convert=QUATERNARY_MASS_CONVERT,
                                                 particle_mass=QUATERNARY_PART_MASS,
                                                 mvir_particles_min=MIN_MVIR_PARTICLES)
            print(f'  {len(snapdata_quaternary)} snapshots loaded')
        print()

    # Generate requested plots
    for num in plot_nums:
        if num in Z0_PLOTS:
            Z0_PLOTS[num](primary, secondary, tertiary, quaternary)
        elif num in EVOLUTION_PLOTS:
            EVOLUTION_PLOTS[num](snapdata_h2, snapdata_cold, snapdata_tertiary, snapdata_quaternary)
        elif num in STANDALONE_PLOTS:
            STANDALONE_PLOTS[num]()
        else:
            print(f'Warning: Plot {num} not defined, skipping.')
        print()

    print('Done.')


if __name__ == '__main__':
    main()

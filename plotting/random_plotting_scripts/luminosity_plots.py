#!/usr/bin/env python
"""
luminosity_plots.py
===================
Convert SAGE26 stellar masses to luminosities using the Bell et al. (2003)
empirical mass-to-light calibration, and plot the galaxy luminosity function.

B-V colour is estimated from the bulge-to-total ratio (a simple proxy for
galaxy age/morphology: disk-dominated = blue, bulge-dominated = red).
Observational Schechter-function fits are overlaid for comparison.

Usage:
    python plotting/luminosity_plots.py           # B + r + K bands
    python plotting/luminosity_plots.py B K       # specific bands
    python plotting/luminosity_plots.py --grid    # multi-panel grid only
"""

import glob
import os
import sys

import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

# ========================== CONFIGURATION ==========================

PRIMARY_DIR = './output/millennium/'
OUTPUT_DIR  = './output/millennium/plots/'
OUTPUT_FMT  = '.pdf'

# Bell et al. (2003) use a "diet Salpeter" IMF, which gives log(M/L) ~0.15 dex
# higher than Kroupa/Chabrier.  Set to -0.15 if SAGE masses are on the Kroupa
# scale (check your calibration).  Default 0: no correction applied.
IMF_OFFSET = 0.0

_MSUN_CGS = 1.989e33

# ========================== DATA I/O ==========================

def _find_files(directory):
    files = sorted(glob.glob(os.path.join(directory, 'model_*.hdf5')))
    if not files:
        single = os.path.join(directory, 'model_0.hdf5')
        if os.path.exists(single):
            files = [single]
    return files


def load_z0_data(directory):
    """
    Load z=0 galaxy properties needed for luminosity calculations.

    Returns (data_dict, volume_Mpc3, hubble_h).
    """
    files = _find_files(directory)
    if not files:
        raise RuntimeError(f'No model files found in {directory}')

    with h5.File(files[0], 'r') as f:
        sim  = f['Header/Simulation']
        rt   = f['Header/Runtime']
        h         = float(sim.attrs['hubble_h'])
        box       = float(sim.attrs['box_size'])           # h^-1 Mpc
        snap_nr   = int(sim.attrs['LastSnapshotNr'])
        unit_mass = float(rt.attrs['UnitMass_in_g'])

    mass_conv = unit_mass / _MSUN_CGS / h                  # → M_sun

    fvp = 0.0
    for fp in files:
        with h5.File(fp, 'r') as f:
            fvp += float(f['Header/Runtime'].attrs['frac_volume_processed'])
    volume = (box / h) ** 3 * fvp                          # Mpc^3

    snap_key  = f'Snap_{snap_nr}'
    _props     = ('StellarMass', 'BulgeMass', 'Type')
    _mass_props = {'StellarMass', 'BulgeMass'}
    chunks = {p: [] for p in _props}

    for fp in files:
        with h5.File(fp, 'r') as f:
            if snap_key not in f:
                continue
            grp = f[snap_key]
            for p in _props:
                if p in grp:
                    chunks[p].append(np.array(grp[p]))

    data = {}
    for p in _props:
        arr = np.concatenate(chunks[p]) if chunks[p] else np.array([])
        data[p] = arr * mass_conv if p in _mass_props else arr

    return data, volume, h


# ========================== MASS-TO-LIGHT ==========================

# Bell et al. (2003) Table 7: log10(M/L_λ) = a_λ + b_λ × (B-V)
# Solar absolute magnitudes from Willmer (2018).
BELL03 = {
    'U': {'a': -1.224, 'b': 2.738, 'M_sun': 6.34, 'sys': 'Johnson-Cousins'},
    'B': {'a': -0.942, 'b': 1.737, 'M_sun': 5.36, 'sys': 'Johnson-Cousins'},
    'V': {'a': -0.628, 'b': 1.305, 'M_sun': 4.82, 'sys': 'Johnson-Cousins'},
    'R': {'a': -0.520, 'b': 1.434, 'M_sun': 4.42, 'sys': 'Johnson-Cousins'},
    'I': {'a': -0.318, 'b': 1.145, 'M_sun': 4.08, 'sys': 'Johnson-Cousins'},
    'J': {'a': -0.261, 'b': 0.863, 'M_sun': 3.64, 'sys': '2MASS'},
    'H': {'a': -0.185, 'b': 0.779, 'M_sun': 3.32, 'sys': '2MASS'},
    'K': {'a': -0.206, 'b': 0.750, 'M_sun': 3.28, 'sys': '2MASS'},
    # SDSS bands (Bell et al. 2003 Table B1, SDSS photometry, approx B-V colour)
    'u': {'a': -1.019, 'b': 2.406, 'M_sun': 6.39, 'sys': 'SDSS'},
    'g': {'a': -0.630, 'b': 1.716, 'M_sun': 5.11, 'sys': 'SDSS'},
    'r': {'a': -0.499, 'b': 1.519, 'M_sun': 4.65, 'sys': 'SDSS'},
    'i': {'a': -0.399, 'b': 1.314, 'M_sun': 4.53, 'sys': 'SDSS'},
    'z': {'a': -0.367, 'b': 1.243, 'M_sun': 4.51, 'sys': 'SDSS'},
}


def estimate_bv(bulge_mass, stellar_mass):
    """
    Estimate B-V colour from the bulge-to-total ratio.

    Anchor points (Bell & de Jong 2001 style):
      B/T = 0  (pure late-type disk):  B-V ≈ 0.40
      B/T = 1  (pure elliptical):       B-V ≈ 0.85
    Linear interpolation between these extremes.
    """
    bt = np.where(stellar_mass > 0,
                  np.clip(bulge_mass / stellar_mass, 0.0, 1.0),
                  0.0)
    return 0.40 + 0.45 * bt


def mass_to_abs_mag(stellar_mass, bv, band, imf_offset=IMF_OFFSET):
    """
    Compute absolute magnitude from stellar mass using Bell et al. (2003).

    Parameters
    ----------
    stellar_mass : array_like, M_sun
    bv           : array_like, B-V colour (Vega)
    band         : str, key in BELL03 (e.g. 'B', 'r', 'K')
    imf_offset   : float, dex to *subtract* from log10(M/L).
                   Use -0.15 to convert diet-Salpeter → Kroupa/Chabrier.

    Returns
    -------
    abs_mag : ndarray, absolute magnitude (Vega)
    """
    c = BELL03[band]
    log_ml = c['a'] + c['b'] * np.asarray(bv) + imf_offset
    log_l  = np.log10(np.maximum(stellar_mass, 1e-6)) - log_ml
    return c['M_sun'] - 2.5 * log_l


def mass_to_luminosity(stellar_mass, bv, band, imf_offset=IMF_OFFSET):
    """
    Convert stellar mass to luminosity in solar units (L / L_sun).

    Parameters
    ----------
    stellar_mass : array_like, M_sun
    bv           : array_like, B-V colour (Vega)
    band         : str, key in BELL03
    imf_offset   : float, dex correction for IMF (see mass_to_abs_mag)

    Returns
    -------
    lum : ndarray, luminosity in units of L_sun
    """
    c = BELL03[band]
    log_ml = c['a'] + c['b'] * np.asarray(bv) + imf_offset
    log_l  = np.log10(np.maximum(stellar_mass, 1e-6)) - log_ml
    return 10.0 ** log_l


# ========================== LUMINOSITY FUNCTION ==========================

def luminosity_function(magnitudes, volume, binwidth=0.5):
    """
    Compute log10(Φ) in Mpc^-3 mag^-1.

    Parameters
    ----------
    magnitudes : array_like, absolute magnitudes
    volume     : float, comoving volume in Mpc^3
    binwidth   : float, magnitude bin width

    Returns
    -------
    centres  : ndarray, bin centres
    log10phi : ndarray, log10 number density; NaN where count == 0
    """
    mi = np.floor(np.nanmin(magnitudes) / binwidth) * binwidth - binwidth
    ma = np.ceil( np.nanmax(magnitudes) / binwidth) * binwidth + binwidth
    nbins  = int(round((ma - mi) / binwidth))
    counts, edges = np.histogram(magnitudes, range=(mi, ma), bins=nbins)
    centres = edges[:-1] + 0.5 * binwidth
    with np.errstate(divide='ignore'):
        log_phi = np.log10(counts / volume / binwidth)
    log_phi[~np.isfinite(log_phi)] = np.nan
    return centres, log_phi


def schechter(M, M_star, phi_star, alpha):
    """Schechter LF in linear Φ [Mpc^-3 mag^-1]."""
    x = 10.0 ** (0.4 * (M_star - M))
    return 0.4 * np.log(10.0) * phi_star * x ** (alpha + 1.0) * np.exp(-x)


# ========================== OBSERVATIONAL COMPARISONS ==========================

SHARK_LF_DIR = '/Users/mbradley/Documents/PhD/shark/data/lf/'

# Vega → AB offset for each band (add to Vega magnitude to get AB magnitude).
# SDSS bands are defined in AB so offset = 0.  Broadband values from Willmer (2018).
_VEGA_TO_AB = {
    'U': 0.79, 'B': -0.09, 'V': 0.00, 'R': 0.16,
    'I': 0.44, 'J': 0.91, 'H': 1.39, 'K': 1.85,
    'u': 0.0, 'g': 0.0, 'r': 0.0, 'i': 0.0, 'z': 0.0,
}

# Real data files from Driver et al. (2012) GAMA survey.
# Columns: M_AB - 5log10(h)  |  phi [(Mpc/h)^-3 per 0.5 mag]  |  1-sigma err  |  N_gal
# All magnitudes are in the AB system.
_OBS_FILES = {
    'r': dict(file=SHARK_LF_DIR + 'lfr_z0_driver12.data', label='Driver+12 ($r$)'),
    'K': dict(file=SHARK_LF_DIR + 'lfk_z0_driver12.data', label='Driver+12 ($K$)'),
    'g': dict(file=SHARK_LF_DIR + 'lfg_z0_driver12.data', label='Driver+12 ($g$)'),
    'i': dict(file=SHARK_LF_DIR + 'lfi_z0_driver12.data', label='Driver+12 ($i$)'),
    'u': dict(file=SHARK_LF_DIR + 'lfu_z0_driver12.data', label='Driver+12 ($u$)'),
    'z': dict(file=SHARK_LF_DIR + 'lfz_z0_driver12.data', label='Driver+12 ($z$)'),
    'J': dict(file=SHARK_LF_DIR + 'lfj_z0_driver12.data', label='Driver+12 ($J$)'),
    'H': dict(file=SHARK_LF_DIR + 'lfh_z0_driver12.data', label='Driver+12 ($H$)'),
}

# Schechter fallback for bands without a real data file.
# M_star quoted as M* - 5log10(h); phi_star in h^3 Mpc^-3.
_OBS_SCHECHTER = {
    'B': dict(M_star=-19.66, phi_star=1.61e-2, alpha=-1.21, label=r'Norberg+02 ($b_J$)'),
    'V': dict(M_star=-20.8,  phi_star=1.5e-2,  alpha=-1.07, label=r'Schechter ($V$)'),
}

# x-axis limits in (M - 5log10(h)) AB units
_BAND_XLIM = {
    'B': (-25, -15),
    'r': (-25, -15),
    'K': (-25, -15),   # K in AB (~1.85 mag fainter than Vega)
    'g': (-25, -15),
    'i': (-25, -15),
}


def load_obs_lf(filepath, h):
    """
    Load a Driver et al. (2012) style LF file and convert to plot-ready arrays.

    Input file columns:
        M_AB - 5log10(h)  |  phi [(Mpc/h)^-3 (0.5 mag)^-1]  |  1-sigma err  |  N_gal

    Returns (mag, log10_phi, log10_phi_err) with zero-phi rows removed.
    phi is converted to Mpc^-3 mag^-1 via:  phi_obs × h^3 × 0.5
    """
    try:
        data = np.loadtxt(filepath, comments='#')
    except Exception as e:
        print(f'  Warning: could not read {filepath}: {e}')
        return None, None, None

    mag      = data[:, 0]
    phi_raw  = data[:, 1]
    err_raw  = data[:, 2]

    # Drop bins with zero phi (unfilled bins at bright/faint extremes)
    good = phi_raw > 0
    mag, phi_raw, err_raw = mag[good], phi_raw[good], err_raw[good]

    # Convert (Mpc/h)^-3 (0.5 mag)^-1  →  Mpc^-3 mag^-1
    factor   = h ** 3 * 0.5
    phi      = phi_raw * factor
    phi_err  = err_raw * factor

    with np.errstate(divide='ignore', invalid='ignore'):
        log_phi     = np.log10(phi)
        # Asymmetric errors in log space: Δlog = Δphi / (phi × ln10)
        log_phi_err = phi_err / (phi * np.log(10.0))

    return mag, log_phi, log_phi_err


def _get_obs(band, h, xlim):
    """
    Return (mag, log10_phi, log10_phi_err, label) for the observational comparison.

    Tries a real data file first; falls back to a Schechter function curve
    (err=None in that case).  Returns all-None if no data exists for the band.
    """
    if band in _OBS_FILES:
        info = _OBS_FILES[band]
        mag, log_phi, log_phi_err = load_obs_lf(info['file'], h)
        if mag is not None:
            # Restrict to the plot window
            if xlim is not None:
                ok = (mag >= xlim[0]) & (mag <= xlim[1])
                mag, log_phi, log_phi_err = mag[ok], log_phi[ok], log_phi_err[ok]
            return mag, log_phi, log_phi_err, info['label']

    if band in _OBS_SCHECHTER:
        p = _OBS_SCHECHTER[band]
        phi_star = p['phi_star'] * h ** 3
        M_range  = xlim if xlim is not None else (-28, -10)
        M_arr    = np.linspace(M_range[0], M_range[1], 400)
        phi_arr  = schechter(M_arr, p['M_star'], phi_star, p['alpha'])
        with np.errstate(divide='ignore'):
            log_phi = np.log10(phi_arr)
        log_phi[~np.isfinite(log_phi)] = np.nan
        return M_arr, log_phi, None, p['label']

    return None, None, None, None


# ========================== BC03 / python-fsps SPS PHOTOMETRY ==========================
#
# Builds a lookup table: SSP absolute magnitude(age, logZ, band) for a 1 M_sun burst.
# Cached to disk so the ~5 min build only happens once.
# At runtime: load SFHMassDisk/Bulge from HDF5, convert snapshots → lookback times,
# then sum  Σ_j  M_j × 10^(-0.4 × M_SSP(age_j, Z))  to get total flux per galaxy.

BC03_CACHE  = os.path.join(PRIMARY_DIR, 'bc03_ssp_cache.npz')
BC03_BANDS  = ['buser_b', 'v', 'sdss_g', 'sdss_r', '2mass_ks']
# map fsps filter name → our band letter
_FSPS_TO_BAND = {
    'buser_b': 'B', 'v': 'V', 'sdss_g': 'g', 'sdss_r': 'r', '2mass_ks': 'K',
}
# Cosmological parameters matching Millennium
_COSMO_H0   = 73.0
_COSMO_OM0  = 0.25

# Age grid for lookup table (Gyr).  Coarse enough to build quickly.
_SSP_AGE_GRID = np.concatenate([
    np.linspace(0.05, 1.0, 12),   # young populations, finer spacing
    np.linspace(1.5,  14.0, 20),  # older populations
])


def _build_ssp_table(cache_path=BC03_CACHE, bands=BC03_BANDS):
    """
    Build and save SSP lookup table using python-fsps (Chabrier IMF, no dust).

    Grid axes: age_grid (Gyr) × metallicity (log Z/Z_sol) × band
    Magnitudes are for a 1 M_sun SSP burst.
    """
    try:
        import fsps
    except ImportError:
        raise ImportError('python-fsps not installed; run: pip install fsps')

    sp = fsps.StellarPopulation(zcontinuous=1, sfh=0, imf_type=1, dust_type=0)
    sp.params['dust2'] = 0.0
    sp.params['add_neb_emission'] = False

    logz_vals = np.log10(sp.zlegend / 0.0142)  # log Z/Zsol; Zsol=0.0142 (Asplund+09)
    age_grid  = _SSP_AGE_GRID
    n_age     = len(age_grid)
    n_met     = len(logz_vals)
    n_band    = len(bands)

    mags = np.full((n_age, n_met, n_band), np.nan)

    total = n_age * n_met
    done  = 0
    print(f'  Building SSP cache ({n_age} ages × {n_met} metallicities) ...')
    for j, logz in enumerate(logz_vals):
        sp.params['logzsol'] = float(logz)
        for i, age in enumerate(age_grid):
            m = sp.get_mags(tage=float(age), bands=bands)
            mags[i, j, :] = m
            done += 1
            if done % 50 == 0:
                print(f'    {done}/{total}', flush=True)

    np.savez(cache_path, age_grid=age_grid, logz_vals=logz_vals,
             bands=np.array(bands), mags=mags)
    print(f'  SSP cache saved → {cache_path}')
    return age_grid, logz_vals, mags


def load_ssp_table(cache_path=BC03_CACHE, bands=BC03_BANDS):
    """
    Load cached SSP lookup table, building it if it doesn't exist.

    Returns
    -------
    interpolators : dict  band_letter → RegularGridInterpolator(age, logZ) → mag
    """
    if not os.path.exists(cache_path):
        age_grid, logz_vals, mags = _build_ssp_table(cache_path, bands)
    else:
        d = np.load(cache_path, allow_pickle=False)
        age_grid  = d['age_grid']
        logz_vals = d['logz_vals']
        cached_bands = list(d['bands'])
        mags      = d['mags']
        # rebuild if bands changed
        if cached_bands != bands:
            print('  SSP cache band mismatch — rebuilding ...')
            age_grid, logz_vals, mags = _build_ssp_table(cache_path, bands)

    interps = {}
    for k, fname in enumerate(bands):
        letter = _FSPS_TO_BAND.get(fname, fname)
        interps[letter] = RegularGridInterpolator(
            (age_grid, logz_vals), mags[:, :, k],
            method='linear', bounds_error=False, fill_value=None,
        )
    return interps


def _snap_lookback_times(h5file):
    """
    Return lookback time in Gyr for each of the 64 SAGE snapshot bins.

    Uses snapshot_redshifts from the HDF5 header and Millennium cosmology.
    Snap 63 (z=0) → lookback time = 0.
    """
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=_COSMO_H0, Om0=_COSMO_OM0)

    with h5.File(h5file, 'r') as f:
        zs = np.array(f['Header/snapshot_redshifts'])  # shape (64,)

    age_universe = cosmo.age(0).value                  # Gyr at z=0
    ages_snap    = cosmo.age(zs).value                 # Gyr at each snapshot
    lookback     = age_universe - ages_snap            # 0 at z=0, large at high z
    return lookback                                    # shape (64,)


def _load_z0_sfh(directory):
    """
    Load SFHMassDisk, SFHMassBulge, MetalsStellarMass and StellarMass at z=0.

    Returns (sfh_disk, sfh_bulge, metallicity, stellar_mass) all in M_sun,
    and the path of the first file found (for snapshot redshifts).
    """
    files = _find_files(directory)
    if not files:
        raise RuntimeError(f'No model files in {directory}')

    with h5.File(files[0], 'r') as f:
        sim  = f['Header/Simulation']
        rt   = f['Header/Runtime']
        h         = float(sim.attrs['hubble_h'])
        snap_nr   = int(sim.attrs['LastSnapshotNr'])
        unit_mass = float(rt.attrs['UnitMass_in_g'])

    mass_conv = unit_mass / _MSUN_CGS / h
    snap_key  = f'Snap_{snap_nr}'

    sfhd_chunks, sfhb_chunks, met_chunks, sm_chunks = [], [], [], []
    for fp in files:
        with h5.File(fp, 'r') as f:
            if snap_key not in f:
                continue
            g = f[snap_key]
            sfhd_chunks.append(np.array(g['SFHMassDisk']))
            sfhb_chunks.append(np.array(g['SFHMassBulge']))
            met_chunks.append(np.array(g['MetalsStellarMass']))
            sm_chunks.append(np.array(g['StellarMass']))

    sfh_disk  = np.concatenate(sfhd_chunks) * mass_conv   # (N, 64) M_sun
    sfh_bulge = np.concatenate(sfhb_chunks) * mass_conv
    met_sm    = np.concatenate(met_chunks)  * mass_conv   # metals in M_sun
    stellar_m = np.concatenate(sm_chunks)   * mass_conv

    # Mass-weighted metallicity as Z (absolute, not log)
    with np.errstate(divide='ignore', invalid='ignore'):
        metallicity = np.where(stellar_m > 0, met_sm / stellar_m, 0.0142)
    metallicity = np.clip(metallicity, 4.5e-5, 0.045)     # within fsps grid

    return sfh_disk, sfh_bulge, metallicity, stellar_m, files[0]


def compute_bc03_magnitudes(bands=('B', 'r', 'K'), batch_size=500):
    """
    Compute absolute magnitudes for all z=0 galaxies using BC03 SPS.

    For each galaxy:
        M_λ = -2.5 log10( Σ_j  (M_disk_j + M_bulge_j) × 10^(-0.4 × SSP_λ(age_j, Z)) )

    The SSP magnitudes are for 1 M_sun bursts (Chabrier IMF); no dust.

    Returns dict: band_letter → ndarray of absolute magnitudes (AB, M - 5log10(h))
    """
    print('Loading SFH data ...')
    sfh_disk, sfh_bulge, metallicity, stellar_m, h5file = _load_z0_sfh(PRIMARY_DIR)

    # Map band letters to fsps filter names
    _band_to_fsps = {v: k for k, v in _FSPS_TO_BAND.items()}
    fsps_filters  = [_band_to_fsps[b] for b in bands if b in _band_to_fsps]
    band_letters  = [_FSPS_TO_BAND[f] for f in fsps_filters]

    print('Loading / building SSP lookup table ...')
    interps = load_ssp_table(bands=fsps_filters)

    print('Computing lookback times ...')
    lookback = _snap_lookback_times(h5file)   # shape (64,)

    # Clip lookback to valid age range (avoid age=0 at z=0 bin)
    age_min  = _SSP_AGE_GRID[0]
    lookback = np.clip(lookback, age_min, _SSP_AGE_GRID[-1])

    # Galaxy metallicities → log Z/Zsol
    logz_gal = np.log10(metallicity / 0.0142)   # shape (N,)
    logz_gal = np.clip(logz_gal, -4.0, 0.5)

    sfh_total = sfh_disk + sfh_bulge             # (N, 64) total mass formed per bin

    N       = sfh_total.shape[0]
    n_snap  = sfh_total.shape[1]
    results = {b: np.full(N, np.nan) for b in band_letters}

    print(f'Computing magnitudes for {N:,} galaxies (batch={batch_size}) ...')
    for start in range(0, N, batch_size):
        end   = min(start + batch_size, N)
        batch = sfh_total[start:end]         # (B, 64)
        logz  = logz_gal[start:end]          # (B,)

        for letter in band_letters:
            itp  = interps[letter]
            flux = np.zeros(end - start)
            for j in range(n_snap):
                mass_j = batch[:, j]         # (B,)
                has_sf = mass_j > 0
                if not has_sf.any():
                    continue
                pts       = np.column_stack([
                    np.full(has_sf.sum(), lookback[j]),
                    logz[has_sf],
                ])
                ssp_mags  = itp(pts)         # magnitude for 1 M_sun SSP
                # flux contribution: mass × 10^(-0.4 × M_ssp)
                flux[has_sf] += mass_j[has_sf] * 10.0 ** (-0.4 * ssp_mags)

            with np.errstate(divide='ignore', invalid='ignore'):
                abs_mag = np.where(flux > 0, -2.5 * np.log10(flux), np.nan)
            results[letter][start:end] = abs_mag

        if (start // batch_size) % 10 == 0:
            print(f'  {end}/{N}', flush=True)

    # Apply h-convention: M_plot = M_abs(AB) - 5log10(h)
    with h5.File(h5file, 'r') as f:
        hub = float(f['Header/Simulation'].attrs['hubble_h'])

    for letter in band_letters:
        vega_ab = _VEGA_TO_AB.get(letter, 0.0)
        results[letter] += vega_ab - 5.0 * np.log10(hub)

    return results, stellar_m, hub


# ========================== PLOT FUNCTIONS ==========================

def plot_luminosity_function(bands=('B', 'r', 'K')):
    """
    One figure per band: SAGE26 luminosity function + observational Schechter fit.
    """
    data, volume, h = _load_and_prep()

    w     = data['StellarMass'] > 0
    mass  = data['StellarMass'][w]
    bulge = data['BulgeMass'][w]
    bv    = estimate_bv(bulge, mass)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for band in bands:
        if band not in BELL03:
            print(f'  Unknown band "{band}" — available: {sorted(BELL03)}')
            continue

        print(f'  {band}-band ({BELL03[band]["sys"]}) ...')
        mags_vega = mass_to_abs_mag(mass, bv, band)
        # Convert Vega → AB, then to M - 5log10(h) convention
        mags = mags_vega + _VEGA_TO_AB.get(band, 0.0) - 5.0 * np.log10(h)
        finite = np.isfinite(mags)

        xlim = _BAND_XLIM.get(band)
        if xlim is not None:
            finite &= (mags >= xlim[0]) & (mags <= xlim[1])

        centres, log_phi = luminosity_function(mags[finite], volume)
        valid = np.isfinite(log_phi)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(centres[valid], log_phi[valid], color='C0', lw=2.0, label='SAGE26')

        obs_M, obs_phi, obs_err, obs_label = _get_obs(band, h, xlim)
        if obs_M is not None:
            obs_ok = np.isfinite(obs_phi)
            if obs_err is not None:
                ax.errorbar(obs_M[obs_ok], obs_phi[obs_ok],
                            yerr=obs_err[obs_ok],
                            fmt='o', ms=4, color='k', lw=1.2,
                            capsize=2, label=obs_label)
            else:
                ax.plot(obs_M[obs_ok], obs_phi[obs_ok], 'k--', lw=1.5,
                        label=obs_label)

        ax.set_xlabel(rf'$M_{{{band}}} - 5\log_{{10}}\,h$  (AB)', fontsize=12)
        ax.set_ylabel(
            r'$\log_{10}\,\Phi\ [\mathrm{Mpc}^{-3}\ \mathrm{mag}^{-1}]$',
            fontsize=12,
        )
        if xlim is not None:
            ax.set_xlim(xlim[0], xlim[1])
        ax.legend(frameon=False, fontsize=10)
        ax.set_ylim(-7, -1)
        fig.tight_layout()

        outfile = os.path.join(OUTPUT_DIR, f'lf_{band}{OUTPUT_FMT}')
        fig.savefig(outfile, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'    → {outfile}')


def plot_luminosity_function_grid(bands=('B', 'r', 'K')):
    """
    Multi-panel luminosity function: all bands in a single figure side-by-side.
    """
    data, volume, h = _load_and_prep()

    w     = data['StellarMass'] > 0
    mass  = data['StellarMass'][w]
    bulge = data['BulgeMass'][w]
    bv    = estimate_bv(bulge, mass)

    valid_bands = [b for b in bands if b in BELL03]
    n   = len(valid_bands)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, band in zip(axes, valid_bands):
        mags_vega = mass_to_abs_mag(mass, bv, band)
        mags = mags_vega + _VEGA_TO_AB.get(band, 0.0) - 5.0 * np.log10(h)
        finite = np.isfinite(mags)

        xlim = _BAND_XLIM.get(band)
        if xlim is not None:
            finite &= (mags >= xlim[0]) & (mags <= xlim[1])

        centres, log_phi = luminosity_function(mags[finite], volume)
        valid = np.isfinite(log_phi)

        ax.plot(centres[valid], log_phi[valid], color='C0', lw=2.0, label='SAGE26')

        obs_M, obs_phi, obs_err, obs_label = _get_obs(band, h, xlim)
        if obs_M is not None:
            obs_ok = np.isfinite(obs_phi)
            if obs_err is not None:
                ax.errorbar(obs_M[obs_ok], obs_phi[obs_ok],
                            yerr=obs_err[obs_ok],
                            fmt='o', ms=4, color='k', lw=1.2,
                            capsize=2, label=obs_label)
            else:
                ax.plot(obs_M[obs_ok], obs_phi[obs_ok], 'k--', lw=1.5,
                        label=obs_label)

        ax.set_xlabel(rf'$M_{{{band}}} - 5\log_{{10}}\,h$  (AB)', fontsize=12)
        if xlim is not None:
            ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(-7, -1)
        ax.legend(frameon=False, fontsize=9)

    axes[0].set_ylabel(
        r'$\log_{10}\,\Phi\ [\mathrm{Mpc}^{-3}\ \mathrm{mag}^{-1}]$',
        fontsize=12,
    )
    fig.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    outfile = os.path.join(OUTPUT_DIR, f'lf_grid{OUTPUT_FMT}')
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Grid plot → {outfile}')


# ========================== HELPERS ==========================

_loaded_cache = None   # simple cache so grid + individual plots share one load


def _load_and_prep():
    global _loaded_cache
    if _loaded_cache is None:
        print(f'Loading z=0 data from {PRIMARY_DIR} ...')
        data, volume, h = load_z0_data(PRIMARY_DIR)
        print(f'  {len(data["StellarMass"]):,} galaxies  |  '
              f'V = {volume:.0f} Mpc^3  |  h = {h}')
        _loaded_cache = (data, volume, h)
    return _loaded_cache


# ========================== B-V VS STELLAR MASS ==========================

def _bt_luminosity_B(bulge_mass, stellar_mass):
    """
    Bulge-to-total ratio in B-band luminosity.

    Uses fixed colours for each component:
      bulge component: B-V = 0.85  (old stellar population)
      disk  component: B-V = 0.40  (young/mixed stellar population)
    """
    BV_BULGE, BV_DISK = 0.85, 0.40
    c = BELL03['B']
    ml_bulge = 10.0 ** (c['a'] + c['b'] * BV_BULGE)
    ml_disk  = 10.0 ** (c['a'] + c['b'] * BV_DISK)
    m_disk   = np.maximum(stellar_mass - bulge_mass, 0.0)
    l_bulge  = np.where(bulge_mass > 0,  bulge_mass / ml_bulge, 0.0)
    l_disk   = np.where(m_disk     > 0,  m_disk     / ml_disk,  0.0)
    l_total  = l_bulge + l_disk
    return np.where(l_total > 0, l_bulge / l_total, 0.0)


def _binned_median_bv(log_mass, bv, mask, bins):
    """Return (bin_centres, median_BV) for galaxies in mask."""
    centres, medians = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        in_bin = mask & (log_mass >= lo) & (log_mass < hi)
        if in_bin.sum() >= 10:
            centres.append(0.5 * (lo + hi))
            medians.append(np.median(bv[in_bin]))
    return np.array(centres), np.array(medians)


def plot_bv_vs_mass(bt_threshold=0.4, dilute=5000, mass_min=1e8):
    """
    B-V colour vs stellar mass.

    Morphological type is determined by the B-band bulge-to-total luminosity
    ratio (B/T_B), computed using fixed component colours (bulge B-V=0.85,
    disk B-V=0.40).

    Early type  (B/T_B >= bt_threshold): red triangles
    Late  type  (B/T_B <  bt_threshold): blue circles

    Note: because the bulge M/L_B is ~6x higher than the disk, B/T_B is
    substantially lower than B/T_mass.  bt_threshold=0.4 corresponds to
    B/T_mass ~ 0.8 (strongly bulge-dominated in mass).
    """
    data, volume, h = _load_and_prep()

    w     = data['StellarMass'] >= mass_min
    mass  = data['StellarMass'][w]
    bulge = data['BulgeMass'][w]

    bv    = estimate_bv(bulge, mass)
    bt_l  = _bt_luminosity_B(bulge, mass)
    log_m = np.log10(mass)

    early = bt_l >= bt_threshold
    late  = ~early

    print(f'  Early type (B/T_B >= {bt_threshold}): {100*early.mean():.1f}%  '
          f'({early.sum():,} galaxies)')
    print(f'  Late  type (B/T_B <  {bt_threshold}): {100*late.mean():.1f}%  '
          f'({late.sum():,} galaxies)')

    np.random.seed(2222)

    def _sample(mask, n):
        idx = np.where(mask)[0]
        if len(idx) > n:
            idx = np.random.choice(idx, n, replace=False)
        return idx

    frac_early = early.mean()
    n_e = max(50,  min(early.sum(), int(dilute * frac_early)))
    n_l = max(200, min(late.sum(),  int(dilute * (1 - frac_early))))

    idx_e = _sample(early, n_e)
    idx_l = _sample(late,  n_l)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(log_m[idx_l], bv[idx_l],
               c='royalblue', marker='o', s=6, alpha=0.35, lw=0,
               label=f'Late type ($B/T_B < {bt_threshold}$)',
               rasterized=True)
    ax.scatter(log_m[idx_e], bv[idx_e],
               c='crimson', marker='^', s=14, alpha=0.6, lw=0,
               label=f'Early type ($B/T_B \\geq {bt_threshold}$)',
               rasterized=True)

    ax.set_xlabel(r'$\log_{10}(M_\star / M_\odot)$', fontsize=12)
    ax.set_ylabel(r'$B - V$',                         fontsize=12)
    ax.set_xlim(9.0, 12.2)
    ax.set_ylim(0.35, 0.95)
    ax.legend(frameon=False, fontsize=10, loc='upper left')
    fig.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    outfile = os.path.join(OUTPUT_DIR, f'bv_vs_mass{OUTPUT_FMT}')
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {outfile}')


# ========================== BC03 LUMINOSITY FUNCTION PLOT ==========================

def plot_lf_bc03(bands=('B', 'r', 'K')):
    """
    Luminosity function using BC03 SPS photometry instead of Bell+03 M/L ratios.
    Produces one PDF per band; overlays same Driver+12 / Schechter obs data.
    """
    valid_bands = [b for b in bands if b in _FSPS_TO_BAND.values()]
    if not valid_bands:
        print(f'  No BC03-supported bands in {bands}. Supported: {list(_FSPS_TO_BAND.values())}')
        return

    mag_dict, stellar_m, h = compute_bc03_magnitudes(bands=valid_bands)
    data, volume, _h       = _load_and_prep()
    volume_use = volume   # Mpc^3

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for band in valid_bands:
        mags   = mag_dict[band]
        finite = np.isfinite(mags) & (stellar_m > 0)

        xlim = _BAND_XLIM.get(band)
        if xlim is not None:
            finite &= (mags >= xlim[0]) & (mags <= xlim[1])

        centres, log_phi = luminosity_function(mags[finite], volume_use)
        valid = np.isfinite(log_phi)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(centres[valid], log_phi[valid], color='C1', lw=2.0,
                label='SAGE26 (BC03)')

        obs_M, obs_phi, obs_err, obs_label = _get_obs(band, h, xlim)
        if obs_M is not None:
            obs_ok = np.isfinite(obs_phi)
            if obs_err is not None:
                ax.errorbar(obs_M[obs_ok], obs_phi[obs_ok],
                            yerr=obs_err[obs_ok],
                            fmt='o', ms=4, color='k', lw=1.2,
                            capsize=2, label=obs_label)
            else:
                ax.plot(obs_M[obs_ok], obs_phi[obs_ok], 'k--', lw=1.5,
                        label=obs_label)

        ax.set_xlabel(rf'$M_{{{band}}} - 5\log_{{10}}\,h$  (AB)', fontsize=12)
        ax.set_ylabel(
            r'$\log_{10}\,\Phi\ [\mathrm{Mpc}^{-3}\ \mathrm{mag}^{-1}]$',
            fontsize=12,
        )
        if xlim is not None:
            ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(-7, -1)
        ax.legend(frameon=False, fontsize=10)
        fig.tight_layout()

        outfile = os.path.join(OUTPUT_DIR, f'lf_bc03_{band}{OUTPUT_FMT}')
        fig.savefig(outfile, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  → {outfile}')


# ========================== MAIN ==========================

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    args = sys.argv[1:]

    use_bc03  = '--bc03' in args
    grid_only = '--grid' in args
    args = [a for a in args if a not in ('--grid', '--bc03')]

    bands = tuple(args) if args else ('B', 'r', 'K')

    if use_bc03:
        print('BC03 luminosity functions ...')
        plot_lf_bc03(bands)
    else:
        if not grid_only:
            plot_luminosity_function(bands)
        plot_luminosity_function_grid(bands)

    print('B-V vs stellar mass ...')
    plot_bv_vs_mass()

    print('Done.')

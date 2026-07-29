#!/usr/bin/env python
"""
luminosity_function_z0.py
=========================
End-to-end pipeline: SAGE26 Millennium HDF5 output -> galaxy luminosities via
stellar population synthesis (python-fsps) -> z=0 luminosity function, overlaid
on the Driver et al. (2012) and Loveday et al. (2012) GAMA observations.

Outputs (in output/millennium/plots/):
  lf_z0_<band>.pdf     per-band total LF: SAGE26 + Driver+12 points + Loveday+12 curve
  lf_z0_grid.pdf       all bands, total LF
  lf_z0_grid_redblue.pdf   all bands, split into star-forming/quiescent (an
                       old/young "red/blue" proxy via log sSFR > -11), vs the
                       Loveday+12 blue/red Schechter fits.
The magnitude axis runs faint -> bright (left -> right).

Method (FSPS from star-formation histories)
-------------------------------------------
SAGE does not compute magnitudes; instead it saves a per-snapshot
star-formation history (``SFHMassDisk`` + ``SFHMassBulge``, one column per
snapshot) plus the total stellar metallicity (``MetalsStellarMass``).  For each
galaxy we co-add FSPS simple-stellar-population (SSP) fluxes over its SFH bins:

    F_band = Sum_j  M_formed(j) * 10^(-0.4 * M_SSP(age_j, Z))
    M_band = -2.5 * log10(F_band)

where age_j is the lookback time from z=0 to snapshot j and Z is the galaxy's
mass-weighted stellar metallicity (assumed constant across bins, since SAGE does
not store a metallicity history).

Recycling: SAGE's SFH bins store the *surviving* stellar mass (their sum equals
StellarMass), whereas an SSP magnitude is normalised per unit mass *formed*.  We
recover the formed mass self-consistently using FSPS's own IMF mass-loss,

    M_formed(j) = M_surviving(j) / f_surv(age_j, Z),

so the light is correctly anchored to the surviving mass SAGE reports, without
relying on SAGE's assumed recycled fraction.

IMF: Chabrier (2003), matching SAGE's default recycled fraction.

Dust: a simple ISM screen (De Lucia & Blaizot 2007 / Guo+2011 style) is applied
by default -- a per-galaxy V-band optical depth set by the cold-gas metallicity
and hydrogen column, a uniform-slab geometry (so A_V saturates), and a Calzetti
(2000) curve for the wavelength dependence.  Gas-poor (quiescent) galaxies come
out ~dust-free, so the attenuation acts mostly on the star-forming population.
The intrinsic (dust-free) curve is shown dotted for reference.

Usage
-----
    python plotting/luminosity_function_z0.py                # u g r i K, with dust
    python plotting/luminosity_function_z0.py r K            # chosen bands
    python plotting/luminosity_function_z0.py --nodust       # intrinsic only
    python plotting/luminosity_function_z0.py --dust0=0.7    # tune dust normalisation
    python plotting/luminosity_function_z0.py --rebuild      # force SSP cache rebuild
    python plotting/luminosity_function_z0.py --dir=./output/microuchuu/   # another sim
    python plotting/luminosity_function_z0.py \
        --dir=./output/millennium/ --compare=./output/millennium_vanilla/ \
        --labels=SAGE26,vanilla                       # overlay two runs

Cosmology (h, Omega_m), volume and snapshot redshifts are read per-simulation
from the HDF5 header, so the same script works on Millennium, microUchuu, etc.;
observations are h-scaled to match.  Plots are written to <dir>/plots/.
"""

import glob
import os
import sys

import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

# ============================ CONFIGURATION ============================

PRIMARY_DIR = './output/millennium/'
OUTPUT_DIR  = './output/millennium/plots/'
OBS_DIR     = './data/lf/'
CACHE_PATH  = './output/millennium/fsps_ssp_lf_cache.npz'
OUTPUT_FMT  = '.pdf'
STYLE_FILE  = os.path.join(os.path.dirname(__file__), 'kieren_cohare_palatino_sty.mplstyle')

_MSUN_CGS = 1.989e33
_ZSOL     = 0.0142           # FSPS solar_metallicity (MIST); used for logZ/Zsol

# band letter -> (FSPS filter name, Driver+12 GAMA obs file, axis label)
BANDS = {
    'u': ('sdss_u',   'lfu_z0_driver12.data', 'u'),
    'g': ('sdss_g',   'lfg_z0_driver12.data', 'g'),
    'r': ('sdss_r',   'lfr_z0_driver12.data', 'r'),
    'i': ('sdss_i',   'lfi_z0_driver12.data', 'i'),
    'K': ('2mass_ks', 'lfk_z0_driver12.data', 'K'),
}
DEFAULT_BANDS = ('u', 'g', 'r', 'i', 'K')

# SSP lookup grid: log-spaced ages (fine at young ages for the u/g bands),
# metallicity taken directly from the FSPS zlegend grid.
_AGE_GRID = np.logspace(np.log10(0.003), np.log10(13.7), 40)   # Gyr

BINWIDTH = 0.5               # magnitude bin width for the LF
XLIM     = (-25.0, -15.0)    # M - 5log10(h), AB
YLIM     = (-6.0, -1.0)


# ============================ DATA I/O ============================

def _find_files(directory):
    files = sorted(glob.glob(os.path.join(directory, 'model_*.hdf5')))
    if not files and os.path.exists(os.path.join(directory, 'model_0.hdf5')):
        files = [os.path.join(directory, 'model_0.hdf5')]
    if not files:
        raise RuntimeError(f'No model_*.hdf5 files found in {directory}')
    return files


def load_z0_sfh(directory):
    """
    Load the z=0 (last snapshot) data needed for SPS luminosities.

    Returns a dict with:
        sfh        (N, Nsnap)  surviving stellar mass formed per snapshot [M_sun]
        metals     (N,)        MetalsStellarMass [M_sun]
        stellar    (N,)        StellarMass [M_sun]
        redshifts  (Nsnap,)    redshift of each snapshot bin
        h          float       Hubble parameter
        volume     float       comoving volume [Mpc^3] (physical)
    """
    files = _find_files(directory)

    with h5.File(files[0], 'r') as f:
        sim = f['Header/Simulation']
        rt  = f['Header/Runtime']
        h         = float(sim.attrs['hubble_h'])
        box       = float(sim.attrs['box_size'])        # Mpc/h
        snap_nr   = int(sim.attrs['LastSnapshotNr'])
        omega_m   = float(sim.attrs['omega_matter'])
        unit_mass = float(rt.attrs['UnitMass_in_g'])
        redshifts = np.array(f['Header/snapshot_redshifts'])

    mass_conv = unit_mass / _MSUN_CGS / h                # -> M_sun
    snap_key  = f'Snap_{snap_nr}'

    fvp = 0.0
    sfh_disk, sfh_bulge, metals, stellar, sfr = [], [], [], [], []
    coldgas, metals_cold, disk = [], [], []
    for fp in files:
        with h5.File(fp, 'r') as f:
            fvp += float(f['Header/Runtime'].attrs['frac_volume_processed'])
            if snap_key not in f:
                continue
            g = f[snap_key]
            sfh_disk.append(np.array(g['SFHMassDisk']))
            sfh_bulge.append(np.array(g['SFHMassBulge']))
            metals.append(np.array(g['MetalsStellarMass']))
            stellar.append(np.array(g['StellarMass']))
            sfr.append(np.array(g['SfrDisk']) + np.array(g['SfrBulge']))  # M_sun/yr
            coldgas.append(np.array(g['ColdGas']))
            metals_cold.append(np.array(g['MetalsColdGas']))
            disk.append(np.array(g['DiskRadius']))       # Mpc/h (disk scale radius)

    sfh      = (np.concatenate(sfh_disk) + np.concatenate(sfh_bulge)) * mass_conv
    metals   = np.concatenate(metals)  * mass_conv
    stellar  = np.concatenate(stellar) * mass_conv
    sfr      = np.concatenate(sfr)                       # already M_sun/yr
    coldgas  = np.concatenate(coldgas)     * mass_conv   # M_sun
    metals_cold = np.concatenate(metals_cold) * mass_conv
    disk_pc  = np.concatenate(disk) / h * 1.0e6          # physical pc
    volume   = (box / h) ** 3 * fvp                      # physical Mpc^3

    return dict(sfh=sfh, metals=metals, stellar=stellar, sfr=sfr,
                coldgas=coldgas, metals_cold=metals_cold, disk_pc=disk_pc,
                redshifts=redshifts, h=h, omega_m=omega_m, volume=volume)


# ============================ FSPS SSP GRID ============================

def build_ssp_grid(bands, cache_path=CACHE_PATH, rebuild=False):
    """
    Build (and cache) SSP absolute-magnitude and surviving-mass-fraction grids.

    Grid axes: log10(age/Gyr) x log10(Z/Zsol).  Magnitudes (AB) are for a
    1 M_sun *formed* SSP burst; f_surv is the surviving stellar mass fraction.

    Returns (mag_interp, fsurv_interp) dicts keyed by band letter, each a
    RegularGridInterpolator over (log_age, log_z).
    """
    # Always build/cache the full band set so requesting a subset never rebuilds.
    all_bands   = list(BANDS)
    all_filters = [BANDS[b][0] for b in all_bands]

    need_build = rebuild or not os.path.exists(cache_path)
    if not need_build:
        d = np.load(cache_path, allow_pickle=False)
        if list(d['bands']) != list(all_filters):
            print('  SSP cache band set changed -> rebuilding.')
            need_build = True

    if need_build:
        try:
            import fsps
        except ImportError:
            raise ImportError('python-fsps not installed; run: pip install fsps '
                              '(and set $SPS_HOME).')

        print(f'  Building FSPS SSP grid ({len(_AGE_GRID)} ages) '
              '- Chabrier IMF, no dust. This runs once and is cached ...')
        sp = fsps.StellarPopulation(zcontinuous=1, sfh=0, imf_type=1, dust_type=0)
        sp.params['dust1'] = 0.0
        sp.params['dust2'] = 0.0
        sp.params['add_neb_emission'] = False

        logz_grid = np.log10(sp.zlegend / _ZSOL)          # from FSPS metallicity grid
        n_age, n_z, n_band = len(_AGE_GRID), len(logz_grid), len(all_filters)
        mags  = np.full((n_age, n_z, n_band), np.nan)
        fsurv = np.full((n_age, n_z), np.nan)

        for jz, logz in enumerate(logz_grid):
            sp.params['logzsol'] = float(logz)
            for ia, age in enumerate(_AGE_GRID):
                mags[ia, jz, :] = sp.get_mags(tage=float(age), bands=all_filters)
                fsurv[ia, jz]   = float(sp.stellar_mass)   # surviving mass / formed
            print(f'    metallicity {jz + 1}/{n_z} done', flush=True)

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, log_age=np.log10(_AGE_GRID), logz=logz_grid,
                 bands=np.array(all_filters), mags=mags, fsurv=fsurv)
        print(f'  SSP grid cached -> {cache_path}')
        d = dict(log_age=np.log10(_AGE_GRID), logz=logz_grid, mags=mags, fsurv=fsurv)

    log_age, logz = d['log_age'], d['logz']
    col = {b: all_bands.index(b) for b in bands}          # requested -> cache column
    mag_interp, fsurv_interp = {}, {}
    fsurv_i = RegularGridInterpolator((log_age, logz), d['fsurv'],
                                      bounds_error=False, fill_value=None)
    for b in bands:
        mag_interp[b] = RegularGridInterpolator(
            (log_age, logz), d['mags'][:, :, col[b]],
            bounds_error=False, fill_value=None)
        fsurv_interp[b] = fsurv_i
    return mag_interp, fsurv_interp


# ============================ LUMINOSITIES ============================

def _snapshot_ages(redshifts, h, omega_m):
    """Lookback time [Gyr] from z=0 to each snapshot, using the sim cosmology."""
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=100.0 * h, Om0=omega_m)
    t0 = cosmo.age(0).value
    return t0 - cosmo.age(np.asarray(redshifts)).value       # 0 at z=0


def compute_magnitudes(model, bands, mag_interp, fsurv_interp):
    """
    Absolute AB magnitude per galaxy for each band (M - 5log10(h) convention).

    Returns dict band -> (N,) magnitudes; NaN where a galaxy has no light.
    """
    sfh     = model['sfh']                                   # (N, Nsnap) surviving M_sun
    stellar = model['stellar']
    h       = model['h']

    lookback = _snapshot_ages(model['redshifts'], h, model['omega_m'])   # (Nsnap,)
    age_min  = _AGE_GRID[0]
    log_age  = np.log10(np.clip(lookback, age_min, _AGE_GRID[-1]))   # (Nsnap,)

    with np.errstate(divide='ignore', invalid='ignore'):
        Z = np.where(stellar > 0, model['metals'] / stellar, _ZSOL)
    Z = np.clip(Z, 4.0e-5, 0.0449)                           # FSPS grid range
    logz = np.clip(np.log10(Z / _ZSOL), -4.0, 0.5)           # (N,)

    N, nsnap = sfh.shape
    results = {b: np.full(N, np.nan) for b in bands}

    # Precompute (age, logz) evaluation points per snapshot, shared across bands.
    for b in bands:
        flux = np.zeros(N)
        for j in range(nsnap):
            m_surv = sfh[:, j]
            active = m_surv > 0
            if not active.any():
                continue
            pts = np.column_stack([np.full(active.sum(), log_age[j]), logz[active]])
            m_ssp   = mag_interp[b](pts)                     # AB mag / M_sun formed
            f_surv  = np.clip(fsurv_interp[b](pts), 1e-3, 1.0)
            m_formed = m_surv[active] / f_surv               # recycling correction
            flux[active] += m_formed * 10.0 ** (-0.4 * m_ssp)
        with np.errstate(divide='ignore', invalid='ignore'):
            mag = np.where(flux > 0, -2.5 * np.log10(flux), np.nan)
        results[b] = mag - 5.0 * np.log10(h)                 # M - 5log10(h)
    return results


# ---------------------------- DUST SCREEN ----------------------------
# Simple ISM dust following the De Lucia & Blaizot (2007) / Guo+2011 recipe:
# a per-galaxy V-band optical depth set by the cold-gas metallicity and
# hydrogen column, a uniform-slab geometry (so A_V saturates rather than
# running away for gas-rich disks), and a Calzetti (2000) curve for the
# wavelength dependence.  Gas-poor (quiescent) galaxies come out ~dust-free.

_BAND_LAM_UM = {'u': 0.3557, 'g': 0.4702, 'r': 0.6176, 'i': 0.7490, 'K': 2.19}
_LAM_V_UM    = 0.551
_RV_CALZ     = 4.05
_NH_REF      = 2.1e21          # cm^-2, Milky-Way reference column
_XH          = 0.75            # hydrogen mass fraction of cold gas
# 1 M_sun/pc^2 of hydrogen -> N_H [cm^-2]
_SIGMA_TO_NH = _XH * (1.989e33 / 1.673e-24) / (3.086e18) ** 2


def _calzetti_k(lam_um):
    """Calzetti (2000) attenuation k(lambda) = A(lambda)/E(B-V)."""
    x = 1.0 / lam_um
    if lam_um >= 0.63:
        k = 2.659 * (-1.857 + 1.040 * x) + _RV_CALZ
    else:
        k = 2.659 * (-2.156 + 1.509 * x - 0.198 * x ** 2 + 0.011 * x ** 3) + _RV_CALZ
    return max(k, 0.0)


def compute_dust_attenuation(model, bands, dust0=1.0):
    """
    Per-galaxy attenuation A_band [mag] for a slab ISM dust screen.

    dust0 sets the V-band optical depth of a Milky-Way-like galaxy
    (Z_cold = Z_sun, N_H = N_H_ref); default 1.0 -> A_V ~ 0.5 mag for such a
    galaxy.  Returns (A_dict, A_V) where A_dict[b] is an (N,) array.
    """
    cold = model['coldgas']
    rpc  = model['disk_pc']
    ok   = (cold > 0) & (rpc > 0)

    # central face-on gas surface density [M_sun/pc^2] -> hydrogen column [cm^-2]
    with np.errstate(divide='ignore', invalid='ignore'):
        sigma = np.where(ok, cold / (2.0 * np.pi * rpc ** 2), 0.0)
        z_cold = np.where(cold > 0, model['metals_cold'] / cold, 0.0) / _ZSOL
    nh = sigma * _SIGMA_TO_NH

    s = np.where(z_cold >= 1.0, 1.6, 1.35)               # metallicity slope
    tau_v = dust0 * np.clip(z_cold, 1e-3, None) ** s \
                  * np.clip(nh / _NH_REF, 1e-3, None) ** 0.7
    tau_v = np.where(ok, tau_v, 0.0)

    with np.errstate(divide='ignore', invalid='ignore'):
        A_V = np.where(tau_v > 1e-3,
                       -2.5 * np.log10((1.0 - np.exp(-tau_v)) / tau_v), 0.0)

    kV = _calzetti_k(_LAM_V_UM)
    A = {b: A_V * (_calzetti_k(_BAND_LAM_UM[b]) / kV) for b in bands}
    return A, A_V


# ============================ LF + OBSERVATIONS ============================

def luminosity_function(mags, volume, binwidth=BINWIDTH, mrange=XLIM):
    """log10(Phi) [Mpc^-3 mag^-1] on a fixed magnitude grid."""
    lo, hi = mrange
    nbins  = int(round((hi - lo) / binwidth))
    counts, edges = np.histogram(mags[np.isfinite(mags)], range=(lo, hi), bins=nbins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    with np.errstate(divide='ignore'):
        log_phi = np.log10(counts / volume / binwidth)
    log_phi[~np.isfinite(log_phi)] = np.nan
    return centres, log_phi


def load_obs(band, h):
    """
    Driver et al. (2012) GAMA LF.  File columns:
        M_AB - 5log10(h) | phi [h^3 Mpc^-3 mag^-1] | 1-sigma err | N_gal
    Converted to physical Mpc^-3 mag^-1 (x h^3) to match the model volume.
    Returns (mag, log10_phi, log10_phi_err_low, log10_phi_err_high) or Nones.
    """
    path = os.path.join(OBS_DIR, BANDS[band][1])
    if not os.path.exists(path):
        return (None,) * 4
    data = np.loadtxt(path, comments='#')
    mag, phi, err = data[:, 0], data[:, 1], data[:, 2]
    good = phi > 0
    mag, phi, err = mag[good], phi[good], err[good]

    phi     *= h ** 3
    err     *= h ** 3
    log_phi  = np.log10(phi)
    lo = log_phi - np.log10(np.clip(phi - err, 1e-30, None))
    hi = np.log10(phi + err) - log_phi
    return mag, log_phi, lo, hi


# Loveday et al. (2012), MNRAS 420, 1239 -- GAMA single-Schechter fits (Table 3),
# 0.1-bandpass, AB.  M* is (0.1 M* - 5 log10 h); phi* in units of 10^-2 h^3 Mpc^-3.
# Available for the SDSS ugriz bands, for the full ('all'), blue and red samples.
# No K-band (Driver+12 remains the K reference).
LOVEDAY12 = {   # band -> {sample: (alpha, Mstar, phistar[h^3 Mpc^-3])}
    'u': {'all': (-1.21, -18.02, 1.96e-2), 'blue': (-1.44, -18.27, 0.88e-2), 'red': (-0.40, -17.34, 1.29e-2)},
    'g': {'all': (-1.20, -19.71, 1.33e-2), 'blue': (-1.42, -19.58, 0.71e-2), 'red': (-0.47, -19.31, 1.06e-2)},
    'r': {'all': (-1.26, -20.73, 0.90e-2), 'blue': (-1.45, -20.28, 0.55e-2), 'red': (-0.53, -20.28, 0.98e-2)},
    'i': {'all': (-1.22, -21.13, 0.90e-2), 'blue': (-1.45, -20.68, 0.50e-2), 'red': (-0.46, -20.63, 1.04e-2)},
}


def schechter(M, alpha, Mstar, phistar):
    """Schechter LF [same phistar units], in magnitudes."""
    x = 10.0 ** (0.4 * (Mstar - M))
    return 0.4 * np.log(10.0) * phistar * x ** (1.0 + alpha) * np.exp(-x)


def loveday_curve(band, sample, h, mrange=XLIM):
    """log10(Phi) [physical Mpc^-3 mag^-1] for a Loveday+12 sample, or None."""
    if band not in LOVEDAY12 or sample not in LOVEDAY12[band]:
        return None, None
    alpha, Mstar, phistar = LOVEDAY12[band][sample]
    M = np.linspace(mrange[0], mrange[1], 400)
    phi = schechter(M, alpha, Mstar, phistar) * h ** 3   # h^3 Mpc^-3 -> physical
    with np.errstate(divide='ignore'):
        return M, np.log10(phi)


# ---------------------------- STELLAR MASS FUNCTION ----------------------------
# Baldry et al. (2012), MNRAS 421, 621 -- GAMA z<0.06 total GSMF, double Schechter
# (Chabrier IMF, h=0.7).  Same survey family as the Driver+12 K-band LF, so the
# two provide a like-for-like mass-vs-light comparison.
BALDRY12 = dict(logMstar=10.66, phi1=3.96e-3, a1=-0.35, phi2=0.79e-3, a2=-1.47)
SMF_MRANGE = (8.5, 12.4)


def double_schechter(logM, p):
    """Double-Schechter phi(logM) [Mpc^-3 dex^-1]."""
    x = 10.0 ** (logM - p['logMstar'])
    return (np.log(10.0) * np.exp(-x) * x
            * (p['phi1'] * x ** p['a1'] + p['phi2'] * x ** p['a2']))


def stellar_mass_function(logm, volume, binwidth=0.2, mrange=SMF_MRANGE):
    """log10(Phi) [Mpc^-3 dex^-1] vs log10(M*/Msun)."""
    nb = int(round((mrange[1] - mrange[0]) / binwidth))
    counts, edges = np.histogram(logm[np.isfinite(logm)], range=mrange, bins=nb)
    centres = 0.5 * (edges[:-1] + edges[1:])
    with np.errstate(divide='ignore'):
        phi = np.log10(counts / volume / binwidth)
    phi[~np.isfinite(phi)] = np.nan
    return centres, phi


# ============================ PLOTTING ============================

def _use_style():
    if os.path.exists(STYLE_FILE):
        try:
            plt.style.use(STYLE_FILE)
        except Exception:
            pass


_YLABEL = r'$\log_{10}\,\Phi\ [\mathrm{Mpc}^{-3}\,\mathrm{mag}^{-1}]$'


def _finish_axis(ax, band):
    label = BANDS[band][2]
    ax.set_xlabel(rf'$M_{{{label}}} - 5\log_{{10}}h$  (AB)')
    ax.set_xlim(XLIM[1], XLIM[0])          # flipped: faint -> bright (left -> right)
    ax.set_ylim(*YLIM)
    ax.legend(frameon=False, fontsize=8.5)


def _plot_total_panel(ax, band, mags_d, mags_i, vol, h):
    """Total LF: dusty (solid) + intrinsic (dotted) SAGE26 + Driver+12 + Loveday+12."""
    if mags_i is not None:
        ci, lpi = luminosity_function(mags_i[band], vol)
        ax.plot(ci, lpi, color='C0', ls=':', lw=1.3, alpha=0.8,
                label='SAGE26 (intrinsic)')
    cd, lpd = luminosity_function(mags_d[band], vol)
    dlabel = 'SAGE26 (FSPS + dust)' if mags_i is not None else 'SAGE26 (FSPS, dust-free)'
    ax.plot(cd, lpd, color='C0', lw=2.2, label=dlabel)

    om, ophi, olo, ohi = load_obs(band, h)
    if om is not None:
        ax.errorbar(om, ophi, yerr=[olo, ohi], fmt='o', ms=4, color='k',
                    lw=1.1, capsize=2, label='Driver+12 (GAMA)')
    lm, lphi = loveday_curve(band, 'all', h)
    if lm is not None:
        ax.plot(lm, lphi, color='0.45', ls='--', lw=1.6, label='Loveday+12 (GAMA)')
    _finish_axis(ax, band)


def _plot_split_panel(ax, band, mags, vol, h, ssf_mask):
    """Red/blue LF: model split by sSFR, vs Loveday+12 blue/red Schechter fits."""
    for mask, colour, mlabel, ocolour, osample in (
        (ssf_mask,  'royalblue', 'SAGE26 star-forming', 'blue', 'blue'),
        (~ssf_mask, 'crimson',   'SAGE26 quiescent',    'red',  'red'),
    ):
        c, lp = luminosity_function(mags[band][mask], vol)
        ax.plot(c, lp, color=colour, lw=2.2, label=mlabel)
        lm, lphi = loveday_curve(band, osample, h)
        if lm is not None:
            ax.plot(lm, lphi, color=ocolour, ls='--', lw=1.5,
                    label=f'Loveday+12 {osample}')
    _finish_axis(ax, band)


def _grid(bands, panel_fn, tag):
    n = len(bands)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, b in zip(axes, bands):
        panel_fn(ax, b)
    axes[0].set_ylabel(_YLABEL)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, f'lf_z0_{tag}{OUTPUT_FMT}')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> {out}')


def plot_all(model, bands, mags, dust=True, dust0=1.0, ssfr_cut=-11.0):
    _use_style()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    h, vol = model['h'], model['volume']

    # apply the dust screen (mags_d = luminosities we plot; mags_i = intrinsic ref)
    if dust:
        A, A_V = compute_dust_attenuation(model, bands, dust0=dust0)
        mags_d = {b: mags[b] + A[b] for b in bands}
        mags_i = mags
        sf = np.isfinite(A_V) & (A_V > 0)
        print(f'  dust screen on (dust0={dust0}): median A_V = '
              f'{np.median(A_V[sf]):.2f} (dusty galaxies), '
              f'A_r/A_V = {_calzetti_k(_BAND_LAM_UM["r"]) / _calzetti_k(_LAM_V_UM):.2f}')
    else:
        mags_d, mags_i = mags, None

    # sSFR-based star-forming / quiescent split (a young/old, "blue/red" proxy)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_ssfr = np.log10(model['sfr'] / model['stellar'])
    ssf_mask = log_ssfr > ssfr_cut
    print(f'  star-forming fraction (log sSFR > {ssfr_cut}): '
          f'{100 * np.mean(ssf_mask):.1f}%')

    # 1) total LF: individual panels + grid
    for b in bands:
        fig, ax = plt.subplots(figsize=(6, 5))
        _plot_total_panel(ax, b, mags_d, mags_i, vol, h)
        ax.set_ylabel(_YLABEL)
        fig.tight_layout()
        out = os.path.join(OUTPUT_DIR, f'lf_z0_{b}{OUTPUT_FMT}')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  -> {out}')
    _grid(bands, lambda ax, b: _plot_total_panel(ax, b, mags_d, mags_i, vol, h),
          'grid')

    # 2) red/blue split grid (dusty magnitudes)
    _grid(bands, lambda ax, b: _plot_split_panel(ax, b, mags_d, vol, h, ssf_mask),
          'grid_redblue')


def _prepare_mags(directory, bands, rebuild=False, dust=True, dust0=1.0):
    """Load a run and return (model, plotted-magnitude dict) with dust applied."""
    model = load_z0_sfh(directory)
    print(f'  {os.path.basename(directory.rstrip("/")):22s} '
          f'{model["stellar"].size:>8,} galaxies | V = {model["volume"]:.0f} Mpc^3')
    mag_interp, fsurv_interp = build_ssp_grid(bands, rebuild=rebuild)
    mags = compute_magnitudes(model, bands, mag_interp, fsurv_interp)
    if dust:
        A, _ = compute_dust_attenuation(model, bands, dust0=dust0)
        mags = {b: mags[b] + A[b] for b in bands}
    return model, mags


def plot_compare(entries, bands, out_tag):
    """Overlay the LFs of several runs per band, against the observations."""
    _use_style()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    h = entries[0]['model']['h']

    def _panel(ax, b):
        for e in entries:
            c, lp = luminosity_function(e['mags'][b], e['model']['volume'])
            ax.plot(c, lp, color=e['color'], ls=e['ls'], lw=2.2, label=e['label'])
        om, ophi, olo, ohi = load_obs(b, h)
        if om is not None:
            ax.errorbar(om, ophi, yerr=[olo, ohi], fmt='o', ms=4, color='k',
                        lw=1.1, capsize=2, label='Driver+12 (GAMA)')
        lm, lphi = loveday_curve(b, 'all', h)
        if lm is not None:
            ax.plot(lm, lphi, color='0.45', ls='--', lw=1.4, label='Loveday+12')
        _finish_axis(ax, b)

    # per-band
    for b in bands:
        fig, ax = plt.subplots(figsize=(6, 5))
        _panel(ax, b)
        ax.set_ylabel(_YLABEL)
        fig.tight_layout()
        out = os.path.join(OUTPUT_DIR, f'lf_z0_{out_tag}_{b}{OUTPUT_FMT}')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  -> {out}')
    # grid
    n = len(bands)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, b in zip(axes, bands):
        _panel(ax, b)
    axes[0].set_ylabel(_YLABEL)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, f'lf_z0_{out_tag}_grid{OUTPUT_FMT}')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> {out}')


def plot_k_vs_smf(entries, out_tag):
    """
    Mass-vs-light diagnostic: K-band LF (left) and stellar mass function (right),
    each model vs its GAMA observation (Driver+12 K / Baldry+12 GSMF).  If the
    massive-end model excess appears in BOTH panels it is a mass (feedback)
    problem; if only in K it is a light (M/L, SPS) problem.
    """
    _use_style()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    h = entries[0]['model']['h']
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 5.0))

    # --- left: K-band luminosity function ---
    for e in entries:
        c, lp = luminosity_function(e['mags']['K'], e['model']['volume'])
        axL.plot(c, lp, color=e['color'], ls=e['ls'], lw=2.2, label=e['label'])
    om, ophi, olo, ohi = load_obs('K', h)
    if om is not None:
        axL.errorbar(om, ophi, yerr=[olo, ohi], fmt='o', ms=4, color='k',
                     lw=1.1, capsize=2, label='Driver+12 ($K$)')
    axL.set_xlabel(r'$M_K - 5\log_{10}h$  (AB)')
    axL.set_ylabel(_YLABEL)
    axL.set_xlim(XLIM[1], XLIM[0])
    axL.set_ylim(*YLIM)
    axL.legend(frameon=False, fontsize=9)
    axL.set_title('Light (K-band LF)', fontsize=10)

    # --- right: stellar mass function ---
    grid = np.linspace(*SMF_MRANGE, 250)
    axR.plot(grid, np.log10(double_schechter(grid, BALDRY12)), color='k', ls='--',
             lw=1.6, label='Baldry+12 (GAMA)')
    for e in entries:
        m = e['model']['stellar']
        c, phi = stellar_mass_function(np.log10(m[m > 0]), e['model']['volume'])
        axR.plot(c, phi, color=e['color'], ls=e['ls'], lw=2.2, label=e['label'])
    axR.set_xlabel(r'$\log_{10}(M_\star / M_\odot)$')
    axR.set_xlim(SMF_MRANGE[0], 12.2)
    axR.set_ylim(-6.0, -1.0)
    axR.legend(frameon=False, fontsize=9)
    axR.set_title('Mass (stellar mass function)', fontsize=10)

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, f'lf_z0_{out_tag}{OUTPUT_FMT}')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> {out}')


# ============================ MAIN ============================

def main(argv):
    global OUTPUT_DIR
    rebuild = '--rebuild' in argv
    dust    = '--nodust' not in argv          # dust screen on by default
    smf     = '--smf' in argv                 # K-LF vs SMF mass/light diagnostic
    dust0   = 1.0
    directory = PRIMARY_DIR
    compare  = None
    labels   = None
    for a in argv:
        if a.startswith('--dust0='):
            dust0 = float(a.split('=', 1)[1])
        if a.startswith('--dir='):
            directory = a.split('=', 1)[1]
        if a.startswith('--compare='):
            compare = a.split('=', 1)[1]
        if a.startswith('--labels='):
            labels = a.split('=', 1)[1].split(',')
    argv = [a for a in argv if not a.startswith('--')]
    bands = tuple(b for b in argv if b in BANDS) or DEFAULT_BANDS

    # ---- comparison / diagnostic modes: overlay runs on shared axes ----
    if compare is not None or smf:
        OUTPUT_DIR = os.path.join(directory, 'plots')
        name_a = os.path.basename(directory.rstrip('/'))
        name_b = os.path.basename(compare.rstrip('/')) if compare else None
        use_bands = ('K',) if smf else bands       # SMF diagnostic only needs K
        lab_a = labels[0] if labels else name_a
        lab_b = (labels[1] if labels and len(labels) > 1 else name_b)
        mA, gA = _prepare_mags(directory, use_bands, rebuild, dust, dust0)
        entries = [dict(label=lab_a, model=mA, mags=gA, color='C0', ls='-')]
        if compare is not None:
            mB, gB = _prepare_mags(compare, use_bands, False, dust, dust0)
            entries.append(dict(label=lab_b, model=mB, mags=gB,
                                color='darkorange', ls='--'))
        if smf:
            tag = 'K_vs_smf' + (f'_{name_b}' if name_b else '')
            print(f'Mass-vs-light diagnostic (K LF vs SMF) ...')
            plot_k_vs_smf(entries, out_tag=tag)
        else:
            print(f'Comparing {name_a} vs {name_b} ...')
            plot_compare(entries, bands, out_tag=f'compare_{name_b}')
        print('Done.')
        return

    OUTPUT_DIR = os.path.join(directory, 'plots')
    print(f'Loading z=0 SFH data from {directory} ...')
    model = load_z0_sfh(directory)
    print(f'  {model["stellar"].size:,} galaxies | V = {model["volume"]:.0f} Mpc^3 '
          f'| h = {model["h"]}')

    print('Preparing FSPS SSP grid ...')
    mag_interp, fsurv_interp = build_ssp_grid(bands, rebuild=rebuild)

    print(f'Computing magnitudes for bands {bands} ...')
    mags = compute_magnitudes(model, bands, mag_interp, fsurv_interp)

    print('Plotting luminosity functions ...')
    plot_all(model, bands, mags, dust=dust, dust0=dust0)
    print('Done.')


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    main(sys.argv[1:])

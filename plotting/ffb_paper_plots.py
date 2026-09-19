#!/usr/bin/env python
"""
ffb_paper_plots.py
==================
Publication-quality figures for the FFB paper.

Usage:
    python ffb_paper_plots.py           # Generate all plots
    python ffb_paper_plots.py A         # Plot A: FFB fraction vs redshift (mass bins)
    python ffb_paper_plots.py B         # Plot B: f_FFB(M_halo, z) heatmap
    python ffb_paper_plots.py C         # Plot C: stellar mass vs redshift + JWST
    python ffb_paper_plots.py D         # Plot D: f_FFB vs log10(1+z) + residual
    python ffb_paper_plots.py E         # Plot E: FFB plane in (Mvir, z) space
    python ffb_paper_plots.py F         # Plot F: m* vs z, medians + JWST
    python ffb_paper_plots.py G         # Plot G: m* at fixed number density + JWST
    python ffb_paper_plots.py A B C D   # Multiple plots
"""

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import os
import sys
import glob
import warnings
warnings.filterwarnings("ignore")
import pandas as pd

from scipy.stats import norm as _snorm

# ========================== CONFIGURATION ==========================

LI24_DIR      = './output/millennium/'
MBK25_DIR     = './output/millennium_mbk_smooth/'
VANILLA_DIR   = './output/millennium_vanilla/'
NOFFB_DIR     = './output/millennium_noffb/'
OUTPUT_DIR    = './output/ffb_paper/plots/'
OBS_DIR       = './data/'
SMF_OBS_DIR   = os.path.join(OBS_DIR, 'smf')
SIZE_OBS_DIR  = os.path.join(OBS_DIR, 'SizesAndAM')
OUTPUT_FORMAT = '.pdf'

# Minimum number of dark matter particles for a halo to be considered resolved.
# Corresponds to the 'Len' field in the SAGE HDF5 output.
MIN_PARTICLES = 20

_MSUN_CGS = 1.989e33

_MASS_PROPS = frozenset({
    'CentralMvir', 'Mvir', 'StellarMass', 'BulgeMass', 'BlackHoleMass',
    'MetalsStellarMass', 'MetalsColdGas', 'MetalsEjectedMass',
    'MetalsHotGas', 'MetalsCGMgas', 'ColdGas', 'HotGas', 'CGMgas',
    'EjectedMass', 'H2gas', 'H1gas', 'IntraClusterStars',
    'MergerBulgeMass', 'InstabilityBulgeMass',
})

plt.style.use("./plotting/kieren_cohare_palatino_sty.mplstyle")

# ========================== SIMULATION HEADER ==========================

def _find_model_files(directory):
    files = sorted(glob.glob(os.path.join(directory, 'model_*.hdf5')))
    if not files:
        single = os.path.join(directory, 'model_0.hdf5')
        if os.path.exists(single):
            files = [single]
    return files


def _read_sim_header(directory):
    files = _find_model_files(directory)
    if not files:
        return None
    try:
        with h5.File(files[0], 'r') as f:
            header = {
                'hubble_h':       float(f['Header/Simulation'].attrs['hubble_h']),
                'omega_matter':   float(f['Header/Simulation'].attrs['omega_matter']),
                'omega_lambda':   float(f['Header/Simulation'].attrs['omega_lambda']),
                'unit_mass_in_g': float(f['Header/Runtime'].attrs['UnitMass_in_g']),
                'redshifts':      list(f['Header/snapshot_redshifts'][:]),
            }
        return header
    except Exception as e:
        print(f"Warning: could not read header from {directory}: {e}")
        return None


_hdr = _read_sim_header(LI24_DIR) or _read_sim_header(MBK25_DIR)
if _hdr:
    HUBBLE_H     = _hdr['hubble_h']
    OMEGA_M      = _hdr['omega_matter']
    OMEGA_L      = _hdr['omega_lambda']
    MASS_CONVERT = _hdr['unit_mass_in_g'] / _MSUN_CGS / HUBBLE_H
    REDSHIFTS    = _hdr['redshifts']
else:
    print("Warning: no model header found — using Millennium defaults")
    HUBBLE_H     = 0.73
    OMEGA_M      = 0.25
    OMEGA_L      = 0.75
    MASS_CONVERT = 1.0e10 / 0.73
    REDSHIFTS    = [
        127.000, 79.998, 50.000, 30.000, 19.916, 18.244, 16.725, 15.343,
         14.086, 12.941, 11.897, 10.944, 10.073,  9.278,  8.550,  7.883,
          7.272,  6.712,  6.197,  5.724,  5.289,  4.888,  4.520,  4.179,
          3.866,  3.576,  3.308,  3.060,  2.831,  2.619,  2.422,  2.239,
          2.070,  1.913,  1.766,  1.630,  1.504,  1.386,  1.276,  1.173,
          1.078,  0.989,  0.905,  0.828,  0.755,  0.687,  0.624,  0.564,
          0.509,  0.457,  0.408,  0.362,  0.320,  0.280,  0.242,  0.208,
          0.175,  0.144,  0.116,  0.089,  0.064,  0.041,  0.020,  0.000,
    ]

# ========================== DATA I/O ==========================

def read_snap(directory, snap, properties, min_particles=None):
    """
    Read properties for a single snapshot, concatenated across MPI files.
    Halos with Len < min_particles (default MIN_PARTICLES) are removed.
    """
    if min_particles is None:
        min_particles = MIN_PARTICLES
    # Always load Len for the resolution cut
    load_props = list(properties)
    caller_wants_len = 'Len' in load_props
    if not caller_wants_len:
        load_props.append('Len')

    files = _find_model_files(directory)
    if not files:
        return {}
    snap_key = f'Snap_{snap}'
    chunks = {p: [] for p in load_props}
    found = False
    for fp in files:
        try:
            with h5.File(fp, 'r') as f:
                if snap_key not in f:
                    continue
                found = True
                grp = f[snap_key]
                for p in load_props:
                    if p in grp:
                        chunks[p].append(np.array(grp[p]))
        except Exception as e:
            print(f"  Warning: {fp}: {e}")
    if not found:
        return {}

    # Concatenate all chunks
    data = {}
    for p in load_props:
        if chunks[p]:
            arr = np.concatenate(chunks[p])
            data[p] = arr * MASS_CONVERT if p in _MASS_PROPS else arr

    # Apply resolution cut
    if 'Len' in data:
        mask = data['Len'] >= min_particles
        data = {p: arr[mask] for p, arr in data.items()}

    # Drop Len if the caller didn't ask for it
    if not caller_wants_len:
        data.pop('Len', None)

    return data

# ========================== PHYSICS ==========================

def ffb_threshold_mass_msun(z):
    """Li+24 FFB threshold mass [M_sun] from Eq. 2.

    M_v,ffb / 10^10.8 M_sun = ((1+z)/10)^-6.2, i.e. the normalisation is pinned
    at z = 9 and the slope is -6.2.  This mirrors calculate_ffb_threshold_mass()
    in src/model_regimes.c, which works in code units of 10^10 M_sun/h: the h in
    log_M_code cancels against the 1e10/h conversion back to M_sun, so the
    threshold carries no residual h.
    """
    z_norm = (1.0 + np.asarray(z, dtype=float)) / 10.0
    log_M_code = 0.8 + np.log10(HUBBLE_H) - 6.2 * np.log10(z_norm)
    return 10.0**log_M_code * 1.0e10 / HUBBLE_H


def ffb_fraction_li24(Mvir_msun, z, delta_log_M=0.15):
    """Li+24 logistic-sigmoid FFB fraction (Eq. 3)."""
    M_thresh = ffb_threshold_mass_msun(z)
    x = np.log10(np.asarray(Mvir_msun) / M_thresh) / delta_log_M
    return 1.0 / (1.0 + np.exp(-x))


# --- MBK25 helpers ---

try:
    from colossus.cosmology import cosmology as _col_cosmo
    from colossus.halo import concentration as _col_conc
    _col_cosmo.setCosmology('custom_millennium', flat=True,
                            H0=73.0, Om0=OMEGA_M, Ob0=0.045,
                            sigma8=0.90, ns=1.0, relspecies=False)
    _HAS_COLOSSUS = True
except Exception:
    _HAS_COLOSSUS = False


def _delta_vir_bn98(z):
    """Halo overdensity relative to rho_crit(z), matching the model.

    SAGE defines R_vir with DELTA_VIRT = 200 rho_crit (model_halo_properties.c)
    and the Ishiyama+21 table it reads is mdef=200c.  The MBK25 criterion
    combines R_vir with c through c^2 / (2 mu(c)), so both must use the same
    definition; a Bryan & Norman overdensity here (the previous behaviour) mixed
    a BN98 R_vir with a 200c concentration and pushed the predicted threshold
    ~0.14 dex above the one the runs actually produce.  Name kept so existing
    callers are unaffected.
    """
    return 200.0 + 0.0 * np.asarray(z, dtype=float)


def _rvir_m(Mvir_msun, z):
    """Virial radius [m] from M_vir [M_sun] using Bryan & Norman overdensity."""
    H0_si = HUBBLE_H * 1.0e5 / 3.085678e22
    Ez    = np.sqrt(OMEGA_M * (1.0 + z)**3 + OMEGA_L)
    rho_c = 3.0 * (H0_si * Ez)**2 / (8.0 * np.pi * 6.674e-11)
    delta = _delta_vir_bn98(z)
    return (3.0 * np.asarray(Mvir_msun) * 1.989e30 / (4.0 * np.pi * delta * rho_c))**(1.0 / 3.0)


def _c_ishiyama21(Mvir_msun, z):
    """Ishiyama+21 mean concentration (falls back to Bullock+01 power law)."""
    M_h = np.asarray(Mvir_msun) * HUBBLE_H
    if _HAS_COLOSSUS:
        try:
            c = _col_conc.concentration(M_h, '200c', z, model='ishiyama21')
            return np.maximum(np.atleast_1d(np.asarray(c, dtype=float)), 1.0)
        except Exception:
            pass
    c = 9.0 / (1.0 + z) * (M_h / 1.0e12)**(-0.13)
    return np.maximum(c, 1.0)


# g_crit = G * 3100 M_sun / pc^2  (BK25 Table 1)
_G_CRIT_SI = 6.674e-11 * 3100.0 * 1.989e30 / (3.085678e16)**2


def ffb_fraction_mbk25(Mvir_msun, z, sigma_c=0.2):
    """
    MBK25 FFB fraction via log-normal concentration scatter (BK25 Eq. 4).

    f_FFB(M, z) = P(c > c_thresh) = norm.sf((ln c_thresh - ln c_mean) / sigma_c)

    c_thresh is defined implicitly by g_max(c_thresh) = g_crit, where
    g_max = G M_vir c^2 / (2 R_vir^2 mu(c)),  mu(c) = ln(1+c) - c/(1+c).
    """
    from scipy.optimize import brentq
    Mvir_msun = np.atleast_1d(np.asarray(Mvir_msun, dtype=float))
    c_mean = _c_ishiyama21(Mvir_msun, z)
    Rvir   = _rvir_m(Mvir_msun, z)
    g_vir  = 6.674e-11 * Mvir_msun * 1.989e30 / Rvir**2

    if sigma_c == 0.0:
        mu    = np.log(1.0 + c_mean) - c_mean / (1.0 + c_mean)
        g_max = g_vir * c_mean**2 / (2.0 * mu)
        return (g_max > _G_CRIT_SI).astype(float)

    f = np.zeros(len(Mvir_msun))
    for i in range(len(Mvir_msun)):
        gv = float(g_vir[i])

        def _obj(cv):
            mu = np.log(1.0 + cv) - cv / (1.0 + cv)
            return gv * cv**2 / (2.0 * mu) - _G_CRIT_SI

        if _obj(1.0) > 0.0:
            f[i] = 1.0
            continue
        if _obj(200.0) < 0.0:
            f[i] = 0.0
            continue
        try:
            c_thresh = brentq(_obj, 1.0, 200.0, xtol=1e-3, rtol=1e-4)
            f[i] = _snorm.sf((np.log(c_thresh) - np.log(float(c_mean[i]))) / sigma_c)
        except ValueError:
            f[i] = 0.0
    return f

# ========================== STYLE ==========================

def setup_style():
    try:
        plt.style.use('./plotting/kieren_cohare_palatino_sty.mplstyle')
    except Exception:
        pass


def save_figure(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches='tight')
    print(f'  Saved: {path}')
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

# ========================== SNAPSHOT SELECTION ==========================

_Z_RANGE = (4.0, 15.0)


def _ffb_snaps():
    """Snapshot indices covering _Z_RANGE, ordered high-z to low-z."""
    return [i for i, z in enumerate(REDSHIFTS) if _Z_RANGE[0] <= z <= _Z_RANGE[1]]

# ========================== PLOT A ==========================

# Three halo-mass bins used in Plot A.
_MASS_BINS_A = [
    (9.0,  10.0, r'$9 < \log M_{\rm vir} < 10$',  '#92c5de'),
    (10.0, 11.0, r'$10 < \log M_{\rm vir} < 11$', '#2166ac'),
    (11.0, 13.5, r'$11 < \log M_{\rm vir} < 13.5$', '#053061'),
]


def _wilson68(n, frac):
    """68% Wilson score interval half-widths (lo, hi)."""
    z_s   = 1.0
    denom = 1 + z_s**2 / n
    cw    = (frac + z_s**2 / (2 * n)) / denom
    margin = z_s * np.sqrt((frac * (1 - frac) + z_s**2 / (4 * n)) / n) / denom
    return max(0.0, frac - (cw - margin)), max(0.0, (cw + margin) - frac)


def plot_A_ffb_fraction_vs_redshift():
    """
    FFB fraction f_FFB = N_FFB / N_central vs redshift for three halo-mass
    bins.  Li+24 shown as solid lines, MBK25 as dashed.  Shading is the
    68% Wilson confidence interval.
    """
    print('Plot A: FFB fraction vs redshift')

    models = [
        {'label': 'Li+24',  'dir': LI24_DIR,  'ls': '-'},
        {'label': 'MBK25',  'dir': MBK25_DIR, 'ls': '--'},
    ]
    props = ['FFBRegime', 'Type', 'Mvir']
    snaps = _ffb_snaps()

    fig, ax = plt.subplots()

    for mlo, mhi, mlabel, color in _MASS_BINS_A:
        for model in models:
            if not _find_model_files(model['dir']):
                print(f"  Skipping {model['label']}: no files in {model['dir']}")
                continue

            z_vals, f_vals, f_lo, f_hi = [], [], [], []
            for snap in snaps:
                d = read_snap(model['dir'], snap, props)
                if not d or 'FFBRegime' not in d:
                    continue
                central  = d['Type'] == 0
                log_mvir = np.log10(np.maximum(d['Mvir'][central], 1e-30))
                in_bin   = (log_mvir >= mlo) & (log_mvir < mhi)
                n = int(np.sum(in_bin))
                if n < 10:
                    continue
                ffb  = d['FFBRegime'][central][in_bin].astype(float)
                frac = np.mean(ffb)
                lo, hi = _wilson68(n, frac)
                z_vals.append(REDSHIFTS[snap])
                f_vals.append(frac)
                f_lo.append(lo)
                f_hi.append(hi)

            if not z_vals:
                continue

            z_arr = np.array(z_vals)
            f_arr = np.array(f_vals)
            ax.plot(z_arr, f_arr, color=color, ls=model['ls'], lw=2)
            ax.fill_between(z_arr,
                            f_arr - np.array(f_lo),
                            f_arr + np.array(f_hi),
                            color=color, alpha=0.12)

    ax.set_xlabel(r'$z$')
    ax.set_ylabel(r'$f_{\rm FFB} = N_{\rm FFB}\,/\,N_{\rm central}$')
    ax.set_xlim(_Z_RANGE[1], _Z_RANGE[0])
    ax.set_ylim(0, 1)

    # Legend: mass bins (colour patches) + model line styles
    bin_handles = [mpatches.Patch(color=c, label=lbl)
                   for _, _, lbl, c in _MASS_BINS_A]
    style_handles = [
        mlines.Line2D([], [], color='k', ls='-',  lw=2, label='Li+24'),
        mlines.Line2D([], [], color='k', ls='--', lw=2, label='MBK25'),
    ]
    leg1 = ax.legend(handles=bin_handles,   loc='upper left',  frameon=False,
                     fontsize='small', title=r'$\log_{10}\,M_{\rm vir}\ [M_\odot]$',
                     title_fontsize='small')
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, loc='upper right', frameon=False, fontsize='small')

    fig.tight_layout()
    save_figure(fig, os.path.join(OUTPUT_DIR, 'A_ffb_fraction_vs_z' + OUTPUT_FORMAT))

# ========================== PLOT B ==========================

def _build_ffb_grid(directory, snaps, mass_bins):
    """
    Build a 2D array (n_snaps × n_mass_bins) of mean FFB fraction.
    Returns the grid; cells with fewer than MIN_N galaxies are NaN.
    """
    MIN_N = 5
    props = ['FFBRegime', 'Type', 'Mvir']
    grid  = np.full((len(snaps), len(mass_bins) - 1), np.nan)

    if not _find_model_files(directory):
        return grid

    for row, snap in enumerate(snaps):
        d = read_snap(directory, snap, props)
        if not d or 'FFBRegime' not in d:
            continue
        central  = d['Type'] == 0
        log_mvir = np.log10(np.maximum(d['Mvir'][central], 1e-30))
        ffb      = d['FFBRegime'][central].astype(float)
        for col in range(len(mass_bins) - 1):
            mask = (log_mvir >= mass_bins[col]) & (log_mvir < mass_bins[col + 1])
            if np.sum(mask) >= MIN_N:
                grid[row, col] = np.mean(ffb[mask])
    return grid


def _z_edges(snaps):
    """Build N+1 redshift bin edges for N snapshots (decreasing order)."""
    z_arr = np.array([REDSHIFTS[s] for s in snaps])
    dz    = np.abs(np.diff(z_arr))
    top   = z_arr[0]  + 0.5 * dz[0]
    bot   = z_arr[-1] - 0.5 * dz[-1]
    mid   = 0.5 * (z_arr[:-1] + z_arr[1:])
    return np.concatenate([[top], mid, [bot]])


def plot_B_ffb_heatmap():
    """
    2-D heatmap of f_FFB(M_halo, z) for Li+24 (left) and MBK25 (right).
    Colour shows the simulated FFB fraction in each (mass, redshift) bin.
    Dashed white contour marks f_FFB = 0.5 from the respective theoretical
    prediction (Li+24 sigmoid or MBK25 log-normal concentration scatter).
    """
    print('Plot B: f_FFB(M_halo, z) heatmap')

    mass_bins = np.linspace(8.5, 13.0, 32)   # log10(M_vir / M_sun)
    snaps     = _ffb_snaps()
    z_e       = _z_edges(snaps)

    models = [
        {'label': 'Li+24',  'dir': LI24_DIR,  'theory': ffb_fraction_li24},
        {'label': 'MBK25',  'dir': MBK25_DIR, 'theory': ffb_fraction_mbk25},
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    pcm_last = None

    for ax, m in zip(axes, models):
        grid = _build_ffb_grid(m['dir'], snaps, mass_bins)

        pcm = ax.pcolormesh(mass_bins, z_e, grid,
                            cmap='RdPu', vmin=0.0, vmax=1.0,
                            shading='flat')
        pcm_last = pcm

        # Theoretical f_FFB = 0.5 contour
        log_M_th  = np.linspace(8.5, 13.0, 200)
        M_th      = 10.0**log_M_th
        z_th      = np.linspace(_Z_RANGE[0], _Z_RANGE[1], 60)
        F_th = np.zeros((len(z_th), len(log_M_th)))
        for j, zz in enumerate(z_th):
            F_th[j, :] = m['theory'](M_th, zz)
        ax.contour(log_M_th, z_th, F_th, levels=[0.5],
                   colors='white', linewidths=1.8, linestyles='--')

        ax.set_xlabel(r'$\log_{10}\,M_{\rm vir}\ [M_\odot]$')
        ax.set_xlim(8.5, 13.0)
        ax.set_ylim(_Z_RANGE[1], _Z_RANGE[0])
        ax.set_title(m['label'])

    axes[0].set_ylabel(r'$z$')

    cbar = fig.colorbar(pcm_last, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label(r'$f_{\rm FFB}$')

    # Annotate the 50% contour
    for ax in axes:
        ax.annotate(r'$f_{\rm FFB}=0.5$', xy=(0.97, 0.05),
                    xycoords='axes fraction', ha='right', va='bottom',
                    color='white', fontsize='small')

    fig.tight_layout()
    save_figure(fig, os.path.join(OUTPUT_DIR, 'B_ffb_heatmap' + OUTPUT_FORMAT))

# ========================== PLOT C ==========================

def _load_epochs():
    """
    Load EPOCHS photometric catalog and return a filtered DataFrame.
    Keeps galaxies with certain_by_eye=True, z > 4, and a valid stellar mass.
    Columns used:
      zbest                       — photometric redshift
      stellar_mass_pipes_zgauss   — log10(M_star / M_sun) from Bagpipes
      stellar_mass_pipes_l1/u1_zgauss — 1-sigma lower/upper uncertainties
    """
    path = os.path.join(SMF_OBS_DIR, 'EPOCHS.csv')
    if not os.path.exists(path):
        print(f'  Warning: EPOCHS catalog not found at {path}')
        return None
    df = pd.read_csv(path)
    mask = (
        (df['certain_by_eye'] == True) &
        (df['zbest'] > 4.0) &
        df['stellar_mass_pipes_zgauss'].notna()
    )
    return df[mask].copy()


# --- Empirical EPOCHS stellar-mass completeness floor -----------------------
#
# EPOCHS is flux-limited, so its stellar-mass distribution is truncated from
# below by the detection limit rather than by any physical cut.  To compare
# the simulation against it on the same footing we derive an empirical floor
# directly from the catalogue: in each redshift bin, the Nth percentile of the
# observed log M* distribution.  This is a *derived* limit -- EPOCHS.csv ships
# no published completeness column -- so it should be quoted as such.
#
# 10th percentile is the default: the highest-z bins hold only ~20-40 galaxies,
# where the 5th percentile is set by one or two objects.
EPOCHS_FLOOR_PCT   = 10.0
EPOCHS_FLOOR_EDGES = [6.5, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 19.0]
EPOCHS_FLOOR_MIN_N = 15      # bins with fewer galaxies are dropped as too noisy

# Fallback floor if the catalogue is unavailable (the previous hard-coded value).
MSTAR_FLOOR_FALLBACK = 1e7   # M_sun


def _epochs_mass_floor(verbose=True):
    """
    Derive an empirical stellar-mass completeness floor from EPOCHS.

    Returns a callable z -> log10(M_star_floor / M_sun), linearly interpolated
    between bin centres and held flat outside the range spanned by the
    catalogue.  Returns None if EPOCHS is unavailable, in which case callers
    should fall back to MSTAR_FLOOR_FALLBACK.

    Note the flat extrapolation below the lowest EPOCHS bin: the catalogue
    starts at z ~ 6.5 while these plots span z = 4-15, so the floor over
    z = 4-6.5 is an assumption, not a measurement.
    """
    ep = _load_epochs()
    if ep is None or len(ep) == 0:
        return None

    z_all = ep['zbest'].to_numpy()
    m_all = ep['stellar_mass_pipes_zgauss'].to_numpy()

    z_cen, m_flr = [], []
    for lo, hi in zip(EPOCHS_FLOOR_EDGES[:-1], EPOCHS_FLOOR_EDGES[1:]):
        sel = (z_all >= lo) & (z_all < hi)
        n = int(sel.sum())
        if n < EPOCHS_FLOOR_MIN_N:
            if verbose and n > 0:
                print(f'    z=[{lo:4.1f},{hi:4.1f}) N={n:<4d} -- dropped (N < {EPOCHS_FLOOR_MIN_N})')
            continue
        floor = float(np.percentile(m_all[sel], EPOCHS_FLOOR_PCT))
        # Bin centre weighted by the galaxies actually in the bin, so a bin
        # whose objects pile up at one edge is not mis-placed.
        z_cen.append(float(np.median(z_all[sel])))
        m_flr.append(floor)
        if verbose:
            print(f'    z=[{lo:4.1f},{hi:4.1f}) N={n:<4d} '
                  f'z_med={z_cen[-1]:5.2f}  floor=10^{floor:.2f}')

    if len(z_cen) < 2:
        print('  Warning: too few EPOCHS bins to build a floor; using fallback.')
        return None

    z_cen = np.array(z_cen)
    m_flr = np.array(m_flr)

    def floor_at(z):
        # np.interp clamps to the end values outside the range, which is the
        # flat extrapolation documented above.
        return np.interp(z, z_cen, m_flr)

    return floor_at


def plot_C_mstar_vs_z():
    """
    Stellar mass vs redshift for three models (Li+24, MBK25, Vanilla).

    For each model the 95th-percentile stellar mass of central galaxies
    (M_star > 10^7 M_sun) is shown as the main line, with a shaded band
    spanning the 84th–99th percentile.  The 50th percentile is shown as
    a thin dotted reference line.

    EPOCHS photometric galaxies (certain_by_eye, z > 4) are overlaid as
    a grey scatter, giving the observed high-z galaxy population for direct
    comparison.  EPOCHS stellar masses are from Bagpipes SED fitting
    (stellar_mass_pipes_zgauss).
    """
    print('Plot C: stellar mass vs redshift + EPOCHS')

    models = [
        {'label': 'Li+24',   'dir': LI24_DIR,    'color': '#1f77b4', 'ls': '-'},
        {'label': 'MBK25',   'dir': MBK25_DIR,   'color': '#d62728', 'ls': '--'},
        {'label': 'Vanilla', 'dir': VANILLA_DIR, 'color': '#555555', 'ls': ':'},
    ]

    MSTAR_FLOOR = 1e7   # M_sun — exclude sub-resolution objects
    props = ['StellarMass', 'Type']
    snaps = _ffb_snaps()

    fig, ax = plt.subplots()

    for m in models:
        if not _find_model_files(m['dir']):
            print(f"  Skipping {m['label']}: no files in {m['dir']}")
            continue

        z_vals = []
        p50_vals, p84_vals, p95_vals, p99_vals = [], [], [], []

        for snap in snaps:
            d = read_snap(m['dir'], snap, props)
            if not d or 'StellarMass' not in d:
                continue
            central = d['Type'] == 0
            mstar   = d['StellarMass'][central]
            mstar   = mstar[mstar > MSTAR_FLOOR]
            if len(mstar) < 20:
                continue
            log_m = np.log10(mstar)
            z_vals.append(REDSHIFTS[snap])
            p50_vals.append(np.percentile(log_m, 50))
            p84_vals.append(np.percentile(log_m, 84))
            p95_vals.append(np.percentile(log_m, 95))
            p99_vals.append(np.percentile(log_m, 99))

        if not z_vals:
            continue

        z_arr  = np.array(z_vals)
        p50    = np.array(p50_vals)
        p84    = np.array(p84_vals)
        p95    = np.array(p95_vals)
        p99    = np.array(p99_vals)

        # 95th percentile — main comparison line
        ax.plot(z_arr, p95, color=m['color'], ls=m['ls'], lw=2.5,
                label=m['label'])
        # 84th–99th band around the high-mass end
        ax.fill_between(z_arr, p84, p99, color=m['color'], alpha=0.12)
        # 50th percentile — thin reference
        ax.plot(z_arr, p50, color=m['color'], ls=m['ls'], lw=0.8, alpha=0.5)

    # --- EPOCHS observational data ---
    epochs = _load_epochs()
    if epochs is not None:
        ax.scatter(
            epochs['zbest'],
            epochs['stellar_mass_pipes_zgauss'],
            s=6, color='#888888', alpha=0.45, linewidths=0,
            zorder=2, label='EPOCHS (phot-z)',
        )

    ax.set_xlabel(r'$z$')
    ax.set_ylabel(r'$\log_{10}\,M_\star\ [M_\odot]$')
    ax.set_xlim(_Z_RANGE[1], _Z_RANGE[0])
    ax.set_ylim(6.5, 12.0)

    # Annotate what the model lines represent
    ax.annotate('95th percentile\n(shaded: 84th–99th)',
                xy=(0.97, 0.06), xycoords='axes fraction',
                ha='right', va='bottom', fontsize='x-small',
                color='#333333')
    ax.annotate('thin lines: median',
                xy=(0.97, 0.13), xycoords='axes fraction',
                ha='right', va='bottom', fontsize='x-small',
                color='#333333')

    ax.legend(frameon=False, fontsize='small', loc='upper right')
    fig.tight_layout()
    save_figure(fig, os.path.join(OUTPUT_DIR, 'C_mstar_vs_z' + OUTPUT_FORMAT))

# ========================== PLOT D ==========================

def plot_D_ffb_fraction_vs_z():
    """
    f_FFB = N_FFB / N_total (central galaxies, Len >= MIN_PARTICLES) vs redshift.

    Main panel: solid lines + 1-sigma Wilson shading (simulation).
    Dashed firebrick: Li+24 analytic prediction, f_Li24(M, z) averaged over the
    actual Millennium HMF at each snapshot — validates the SAGE26 implementation.
    Dotted: f_FFB = 0.5 reference marking where FFB galaxies dominate.

    Residual: f_MBK25 − f_Li24 with 1-sigma shading; grey band = ±0.05.
    """
    print('Plot D: FFB fraction vs redshift')

    models = [
        {'label': 'Li+24',  'dir': LI24_DIR,  'color': 'firebrick'},
        {'label': 'MBK25',  'dir': MBK25_DIR, 'color': 'slateblue'},
    ]
    # Mvir needed for HMF-weighted analytic Li+24 prediction
    props = ['FFBRegime', 'Type', 'Mvir']
    snaps = _ffb_snaps()

    results = {}
    for m in models:
        if not _find_model_files(m['dir']):
            print(f"  Skipping {m['label']}: no files in {m['dir']}")
            continue

        z_vals, f_vals, f_lo, f_hi = [], [], [], []
        f_analytic_vals = []
        is_li24 = (m['label'] == 'Li+24')

        for snap in snaps:
            d = read_snap(m['dir'], snap, props)
            if not d or 'FFBRegime' not in d or 'Type' not in d:
                continue
            central = d['Type'] == 0
            n_total = int(np.sum(central))
            if n_total < 2:
                continue
            n_ffb = int(np.sum(d['FFBRegime'][central] == 1))
            frac  = n_ffb / n_total
            lo, hi = _wilson68(n_total, frac)
            z_vals.append(REDSHIFTS[snap])
            f_vals.append(frac)
            f_lo.append(lo)
            f_hi.append(hi)

            # Analytic Li+24 prediction: average f_Li24(M, z) over the resolved HMF
            if is_li24 and 'Mvir' in d:
                mvir_c = d['Mvir'][central]
                f_analytic_vals.append(
                    float(np.mean(ffb_fraction_li24(mvir_c, REDSHIFTS[snap])))
                )

        if z_vals:
            r = {
                'log1pz': np.log10(1.0 + np.array(z_vals)),
                'f':      np.array(f_vals),
                'lo':     np.array(f_lo),
                'hi':     np.array(f_hi),
                'color':  m['color'],
            }
            if f_analytic_vals:
                r['f_analytic'] = np.array(f_analytic_vals)
            results[m['label']] = r

    z_ticks = [4, 5, 6, 7, 8, 9, 10, 12, 15]
    xlim    = (np.log10(1 + _Z_RANGE[1]), np.log10(1 + _Z_RANGE[0]))

    fig, (ax, ax_res) = plt.subplots(
        2, 1, figsize=(6, 5),
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0},
        sharex=True,
    )

    # --- Main panel: simulated f_FFB ---
    for label, r in results.items():
        ax.plot(r['log1pz'], r['f'], color=r['color'], ls='-', lw=2, label=label)
        ax.fill_between(r['log1pz'],
                        r['f'] - r['lo'],
                        r['f'] + r['hi'],
                        color=r['color'], alpha=0.2)

    # Li+24 analytic prediction (HMF-weighted): validates the SAGE26 implementation.
    # With the Eq.-2 threshold (10^10.8 M_sun at z = 9, slope -6.2) it should
    # track the simulated Li+24 line closely; a systematic offset would point at
    # the regime gate or the resolution cut rather than at the sigmoid itself.
    if 'Li+24' in results and 'f_analytic' in results['Li+24']:
        r_li = results['Li+24']
        ax.plot(r_li['log1pz'], r_li['f_analytic'],
                color='firebrick', ls='--', lw=1.4, alpha=0.7,
                label='Li+24 (analytic)')

    # f_FFB = 0.5 dominance reference
    ax.axhline(0.5, color='k', ls=':', lw=0.9, alpha=0.45)
    ax.text(0.97, (0.5 + 0.025) / 1.1,
            r'$f_{\rm FFB} = 0.5$',
            transform=ax.transAxes, fontsize='x-small',
            va='bottom', ha='right', color='k', alpha=0.6)

    ax.legend(frameon=False, fontsize='small', loc='lower left')
    ax.set_ylabel(r'$f_{\rm FFB}$')
    ax.set_xlim(*xlim)
    ax.set_ylim(0, 1.1)
    ax.tick_params(labelbottom=False)

    # Top redshift axis
    ax_top = ax.twiny()
    ax_top.set_xlim(*xlim)
    ax_top.set_xticks([np.log10(1 + z) for z in z_ticks])
    ax_top.set_xticklabels([str(z) for z in z_ticks])
    ax_top.set_xlabel(r'$z$')

    # --- Residual panel: f_MBK25 − f_Li24 ---
    if 'Li+24' in results and 'MBK25' in results:
        r0 = results['Li+24']
        r1 = results['MBK25']
        # Interpolate MBK25 onto Li+24's x-grid (arrays are high-z first, so reversed)
        f1_interp  = np.interp(r0['log1pz'], r1['log1pz'][::-1], r1['f'][::-1])
        lo1_interp = np.interp(r0['log1pz'], r1['log1pz'][::-1], r1['lo'][::-1])
        hi1_interp = np.interp(r0['log1pz'], r1['log1pz'][::-1], r1['hi'][::-1])

        delta    = f1_interp - r0['f']           # f_MBK25 − f_Li24
        delta_lo = np.sqrt(r0['lo']**2 + lo1_interp**2)
        delta_hi = np.sqrt(r0['hi']**2 + hi1_interp**2)

        # ±0.05 agreement reference band
        ax_res.fill_between(r0['log1pz'], -0.05, 0.05,
                            color='k', alpha=0.07, zorder=0)
        ax_res.plot(r0['log1pz'], delta, color='k', lw=1.5)
        ax_res.fill_between(r0['log1pz'],
                            delta - delta_lo,
                            delta + delta_hi,
                            color='k', alpha=0.15)
        ax_res.axhline(0, color='k', ls='--', lw=0.8)

    ax_res.set_xlabel(r'$\log_{10}(1+z)$')
    ax_res.set_ylabel(r'$f_{\rm MBK25} - f_{\rm Li+24}$', fontsize='small')
    ax_res.set_xlim(*xlim)
    ax_res.set_ylim(-0.28, 0.28)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    save_figure(fig, os.path.join(OUTPUT_DIR, 'D_ffb_fraction_vs_z' + OUTPUT_FORMAT))

# ========================== PLOT E ==========================

# Millennium particle mass [M_sun] — Springel+05, h=0.73
_MILL_PART_MASS_MSUN = 8.61e8


def plot_E_ffb_plane():
    """
    FFB galaxies in (log10 Mvir, z) space.

    Background: grey log-density of all resolved central halos.
    Coloured contours: 2D density of FFB halos from Li+24 (firebrick) and
    MBK25 (slateblue) at levels 5%, 20%, 50%, 90% of peak.
    Solid curves: theoretical 50% threshold — Li+24 sigmoid (firebrick) and
    MBK25 g_max = g_crit (slateblue).
    Dotted line: simulation resolution limit (MIN_PARTICLES halos).
    """
    print('Plot E: FFB plane in (Mvir, z) space')

    from matplotlib.colors import LogNorm

    snaps  = _ffb_snaps()
    props  = ['FFBRegime', 'Type', 'Mvir']

    models = [
        {'label': 'Li+24',  'dir': LI24_DIR,  'color': 'firebrick'},
        {'label': 'MBK25',  'dir': MBK25_DIR, 'color': 'slateblue'},
    ]

    logm_edges = np.linspace(10.5, 14.5, 71)
    z_edges    = np.linspace(4.0,  15.0, 46)
    logm_c     = 0.5 * (logm_edges[:-1] + logm_edges[1:])
    z_c        = 0.5 * (z_edges[:-1]    + z_edges[1:])

    H_all     = np.zeros((len(logm_edges) - 1, len(z_edges) - 1), dtype=float)
    H_ffb     = {m['label']: np.zeros_like(H_all) for m in models}
    all_built = False

    for m in models:
        if not _find_model_files(m['dir']):
            print(f"  Skipping {m['label']}: no files in {m['dir']}")
            continue

        for snap in snaps:
            d = read_snap(m['dir'], snap, props)
            if not d or 'Mvir' not in d:
                continue
            central = d['Type'] == 0
            logm    = np.log10(np.maximum(d['Mvir'][central], 1e-30))
            ffb     = d['FFBRegime'][central]
            zz      = float(REDSHIFTS[snap])
            z_col   = np.full(int(central.sum()), zz)

            if not all_built:
                Htmp, _, _ = np.histogram2d(logm, z_col, bins=[logm_edges, z_edges])
                H_all += Htmp

            is_ffb = ffb == 1
            if is_ffb.any():
                Hf, _, _ = np.histogram2d(logm[is_ffb], z_col[is_ffb],
                                          bins=[logm_edges, z_edges])
                H_ffb[m['label']] += Hf

        all_built = True  # only collect all-halo histogram once (halos are the same)

    fig, ax = plt.subplots(figsize=(7, 6))

    # Background: all resolved halos (grey log-scale density)
    with np.errstate(invalid='ignore'):
        ax.pcolormesh(
            logm_edges, z_edges,
            np.where(H_all.T > 0, H_all.T, np.nan),
            cmap='Greys',
            norm=LogNorm(vmin=1, vmax=max(float(H_all.max()), 2.0)),
            zorder=0, alpha=0.7,
        )

    # FFB density contours for each model
    for m in models:
        H = H_ffb[m['label']]
        if H.max() == 0:
            continue
        vmax   = float(H.T.max())
        levels = vmax * np.array([0.05, 0.2, 0.5, 0.9])
        lws    = [0.8, 1.2, 1.6, 2.0]
        ax.contour(logm_c, z_c, H.T, levels=levels,
                   colors=m['color'], linewidths=lws[:len(levels)],
                   alpha=0.85, zorder=2)

    # ---- Theoretical threshold curves ----
    z_curve = np.linspace(4.0, 15.0, 300)

    # Li+24: M_FFB(z) = 3e11 * (1+z)^-1.5 / h  (sigmoid 50% point)
    log_M_li24 = np.log10(ffb_threshold_mass_msun(z_curve))
    ax.plot(log_M_li24, z_curve, color='firebrick', lw=2.5, ls='-', zorder=4)

    # MBK25: f_FFB = 0.5 contour (g_max = g_crit with log-normal scatter)
    print('  Computing MBK25 threshold contour (may take ~30 s)...')
    log_M_th = np.linspace(10.0, 15.5, 50)
    z_th     = np.linspace(4.0, 15.0, 35)
    F_mbk25  = np.zeros((len(z_th), len(log_M_th)))
    for j, zz in enumerate(z_th):
        F_mbk25[j, :] = ffb_fraction_mbk25(10.0**log_M_th, zz)
    ax.contour(log_M_th, z_th, F_mbk25, levels=[0.5],
               colors='slateblue', linewidths=2.5, zorder=4)

    # Resolution limit
    _log_M_res = np.log10(MIN_PARTICLES * _MILL_PART_MASS_MSUN)
    ax.axvline(_log_M_res, color='#888888', ls=':', lw=1.2, zorder=3)
    ax.text(_log_M_res + 0.05, 14.6,
            f'Res. limit\n({MIN_PARTICLES} ptcl)',
            color='#888888', fontsize='x-small', va='top', ha='left')

    # Legend (proxy artists)
    li_line  = mlines.Line2D([], [], color='firebrick', lw=2.5, label='Li+24 threshold')
    mbk_line = mlines.Line2D([], [], color='slateblue', lw=2.5, label='MBK25 threshold')
    ax.legend(handles=[li_line, mbk_line], frameon=False,
              fontsize='small', loc='upper right')

    ax.set_xlabel(r'$\log_{10}\,M_{\rm vir}\ [M_\odot]$')
    ax.set_ylabel(r'$z$')
    ax.set_xlim(10.5, 14.5)
    ax.set_ylim(4.0, 15.0)

    fig.tight_layout()
    save_figure(fig, os.path.join(OUTPUT_DIR, 'E_ffb_plane' + OUTPUT_FORMAT))

# ========================== PLOT F ==========================

# Spectroscopically confirmed JWST galaxies with reliable stellar mass estimates.
# Sources: Curtis-Lake+23 (JADES), Bunker+23 (GN-z11), Carniani+24 (GS-z14-0),
#          Finkelstein+23 (Maisie's), Harikane+22, Robertson+23.
_JWST_SPEC = [
    # (label,              z,     log10_Mstar, err_dex)
    ('GN-z11',            10.60,  9.1,        0.3),
    ('GS-z10-0',          10.38,  7.9,        0.3),
    ("Maisie's",          12.00,  8.5,        0.4),
    ('GS-z11-0',          11.70,  8.9,        0.3),
    ('GS-z12-0',          12.63,  8.4,        0.3),
    ('GS-z13-0',          13.20,  7.8,        0.4),
    ('GS-z14-0',          14.32,  8.6,        0.4),
]


def plot_F_mstar_ffb_scatter():
    """
    Stellar mass vs redshift for all resolved central galaxies
    (Len >= MIN_PARTICLES) above the EPOCHS empirical completeness floor.

    The mass floor applied to the models is derived per redshift bin from the
    EPOCHS catalogue itself (see _epochs_mass_floor) rather than being a fixed
    value, so the model medians are truncated at the observed completeness
    limit.  Falls back to a flat MSTAR_FLOOR_FALLBACK if EPOCHS is unavailable.
    The floor is applied to the model selection only; nothing about the plotted
    observations is altered.

    Non-FFB galaxies: diluted grey background.
    Li+24 FFB galaxies: red.
    MBK25 FFB galaxies: purple.

    Overlaid: EPOCHS photometric catalog and notable JWST spec-z discoveries.
    """
    print('Plot F: stellar mass vs redshift (FFB median lines)')

    props = ['StellarMass', 'FFBRegime', 'Type']
    snaps = _ffb_snaps()
    MIN_N = 1    # minimum galaxies per snapshot to plot a point

    print('  Deriving EPOCHS empirical mass floor '
          f'(p{EPOCHS_FLOOR_PCT:g} per redshift bin):')
    floor_at = _epochs_mass_floor()
    if floor_at is None:
        print(f'    EPOCHS unavailable -- falling back to a flat '
              f'{MSTAR_FLOOR_FALLBACK:.1e} M_sun floor.')
        def floor_at(z):
            # Array-safe so the same callable works for scalars and grids.
            return np.full_like(np.asarray(z, dtype=float),
                                np.log10(MSTAR_FLOOR_FALLBACK))

    def _percentiles(ms_arr):
        log_m = np.log10(ms_arr)
        return (np.percentile(log_m, 50),
                np.percentile(log_m, 16),
                np.percentile(log_m, 84))

    z_bg,  med_bg,  lo_bg,  hi_bg  = [], [], [], []
    z_li,  med_li,  lo_li,  hi_li  = [], [], [], []
    z_mbk, med_mbk, lo_mbk, hi_mbk = [], [], [], []

    for snap in snaps:
        zz = REDSHIFTS[snap]
        # Same floor for every model at this redshift.
        MSTAR_FLOOR = 10.0 ** float(floor_at(zz))

        d_li = read_snap(LI24_DIR, snap, props)
        if d_li and 'StellarMass' in d_li:
            c      = d_li['Type'] == 0
            ms     = d_li['StellarMass'][c]
            ffb    = d_li['FFBRegime'][c]
            ms_ffb = ms[(ms > MSTAR_FLOOR) & (ffb == 1)]
            if len(ms_ffb) >= MIN_N:
                med, lo, hi = _percentiles(ms_ffb)
                z_li.append(zz); med_li.append(med)
                lo_li.append(lo); hi_li.append(hi)

        d_noffb = read_snap(NOFFB_DIR, snap, ['StellarMass', 'Type'])
        if d_noffb and 'StellarMass' in d_noffb:
            c       = d_noffb['Type'] == 0
            ms_nffb = d_noffb['StellarMass'][c]
            ms_nffb = ms_nffb[ms_nffb > MSTAR_FLOOR]
            if len(ms_nffb) >= MIN_N:
                med, lo, hi = _percentiles(ms_nffb)
                z_bg.append(zz); med_bg.append(med)
                lo_bg.append(lo); hi_bg.append(hi)

        d_mbk = read_snap(MBK25_DIR, snap, props)
        if d_mbk and 'StellarMass' in d_mbk:
            c   = d_mbk['Type'] == 0
            ms  = d_mbk['StellarMass'][c]
            ffb = d_mbk['FFBRegime'][c]
            ok  = ms > MSTAR_FLOOR
            ms_ffb = ms[ok][ffb[ok] == 1]
            if len(ms_ffb) >= MIN_N:
                med, lo, hi = _percentiles(ms_ffb)
                z_mbk.append(zz); med_mbk.append(med)
                lo_mbk.append(lo); hi_mbk.append(hi)

    fig, ax = plt.subplots(figsize=(8, 6))

    def _plot_band(zs, meds, los, his, color, label, zorder):
        zs   = np.array(zs);   meds = np.array(meds)
        los  = np.array(los);  his  = np.array(his)
        ax.plot(zs, meds, color=color, lw=2, zorder=zorder, label=label)
        ax.fill_between(zs, los, his, color=color, alpha=0.2, zorder=zorder - 1)

    if z_bg:
        _plot_band(z_bg,  med_bg,  lo_bg,  hi_bg,  'firebrick',  'No FFB/MBK25 model',   zorder=2)
    if z_li:
        _plot_band(z_li,  med_li,  lo_li,  hi_li,  'black',  'FFB galaxies', zorder=4)
    if z_mbk:
        _plot_band(z_mbk, med_mbk, lo_mbk, hi_mbk, 'mediumpurple',     'MBK25 galaxies', zorder=6)

    # EPOCHS photometric catalog
    epochs = _load_epochs()
    if epochs is not None:
        in_range = (
            (epochs['zbest'] >= _Z_RANGE[0]) &
            (epochs['zbest'] <= _Z_RANGE[1])
        )
        ep = epochs[in_range]
        ax.scatter(ep['zbest'], ep['stellar_mass_pipes_zgauss'],
                   s=4, color="#656262", marker='o', alpha=0.2,
                   linewidths=0.3, edgecolors='k',
                   zorder=7)

    # Baggen+23 individual JWST disk galaxies (z=6.5–8.8)
    _baggen_path = os.path.join(SIZE_OBS_DIR, 'baggen_disk_2023.ecsv')
    if os.path.exists(_baggen_path):
        df_b = pd.read_csv(_baggen_path, comment='#', sep=r'\s+')
        in_range = (df_b['z_phot'] >= _Z_RANGE[0]) & (df_b['z_phot'] <= _Z_RANGE[1])
        df_b = df_b[in_range]
        if len(df_b) > 0:
            ax.scatter(df_b['z_phot'], df_b['log_M_star'],
                       s=18, color='darkorange', marker='D', alpha=0.85,
                       linewidths=0.5, edgecolors='k', zorder=8)

    # Casey+24 photometric galaxies (z=9.2–14.4, COSMOS-Web)
    _casey_path = os.path.join(SIZE_OBS_DIR, 'casey_disk_2024.ecsv')
    if os.path.exists(_casey_path):
        df_c = pd.read_csv(_casey_path, comment='#', sep=r'\s+')
        in_range = (df_c['z_phot_BAGPIPES'] >= _Z_RANGE[0]) & (df_c['z_phot_BAGPIPES'] <= _Z_RANGE[1])
        df_c = df_c[in_range]
        if len(df_c) > 0:
            log_m  = np.log10(df_c['M_star'])
            log_eu = np.log10(df_c['M_star'] + df_c['M_star_err_up']) - log_m
            ax.errorbar(df_c['z_phot_BAGPIPES'], log_m, yerr=log_eu,
                        fmt='s', color='dodgerblue', markersize=5,
                        markeredgecolor='k', markeredgewidth=0.5,
                        ecolor='dodgerblue', elinewidth=1.0, capsize=2,
                        zorder=8)

    # Sun+24 individual JWST galaxies (z=4.4–6.5)
    _sun_path = os.path.join(SIZE_OBS_DIR, 'sun_disk_2024.ecsv')
    if os.path.exists(_sun_path):
        df_s = pd.read_csv(_sun_path, comment='#', sep=r'\s+')
        in_range = (df_s['z'] >= _Z_RANGE[0]) & (df_s['z'] <= _Z_RANGE[1])
        df_s = df_s[in_range]
        if len(df_s) > 0:
            ax.errorbar(df_s['z'], df_s['log_M_star'], yerr=df_s['log_M_star_err'],
                        fmt='^', color='seagreen', markersize=6,
                        markeredgecolor='k', markeredgewidth=0.5,
                        ecolor='seagreen', elinewidth=1.0, capsize=2,
                        zorder=8)

    # Notable JWST spectroscopic galaxies
    _above = {"Maisie's", "GS-z11-0"}  # label those above the point, others below
    for name, z, logm, err in _JWST_SPEC:
        ax.errorbar(z, logm, yerr=err,
                    fmt='*', color='gold', markersize=11,
                    markeredgecolor='k', markeredgewidth=0.6,
                    ecolor='gold', elinewidth=1.2, capsize=2, zorder=8)
        if name in _above:
            ax.annotate(name, xy=(z, logm), xytext=(0, 8),
                        textcoords='offset points', fontsize=10, fontweight='bold',
                        ha='center', va='bottom', color="#0E0C0C", zorder=9)
        else:
            ax.annotate(name, xy=(z, logm), xytext=(0, -8),
                        textcoords='offset points', fontsize=10, fontweight='bold',
                        ha='center', va='top', color="#0E0C0C", zorder=9)

    _standard_legend(ax, loc='upper left', fontsize='small')

    ax.set_xlabel(r'Redshift')
    ax.set_ylabel(r'$\log_{10}\,m_\star\ [M_\odot]$')
    ax.set_xlim(_Z_RANGE[0], _Z_RANGE[1])
    ax.set_ylim(6.5, 12.5)

    fig.tight_layout()
    save_figure(fig, os.path.join(OUTPUT_DIR, 'F_mstar_ffb_scatter' + OUTPUT_FORMAT))


# ========================== PLOT G ==========================
#
# Number-density-matched alternative to Plot F.
#
# Plot F compares a model *median* against flux-limited JWST detections.  Those
# are not the same statistic: the observed points are the most massive objects
# recovered from a survey volume of 10^5-10^6 Mpc^3, while the model median is
# taken over a box of 6x10^5 Mpc^3 (mini-Millennium) or 3x10^8 Mpc^3
# (Millennium-500).  Worse, a median taken above a mass floor mostly measures
# the floor when the mass function is steep, which compresses the separation
# between the three models.
#
# The fix is to compare at fixed cumulative comoving number density: for each
# model, report the stellar mass m* such that n(> m*) equals a chosen value.
# This is the statistic Boylan-Kolchin (2025) uses in his Fig. 5, so it also
# makes the figure directly comparable to the paper being implemented.

# Cumulative comoving number densities, rarest first.  Only those reachable in
# the simulated volume (>= _ND_MIN_COUNT galaxies) are drawn.
_ND_TARGETS   = [1e-6, 1e-5, 1e-4, 1e-3]
_ND_MIN_COUNT = 10

# The single density drawn in the figure.  10^-5 Mpc^-3 is roughly what the
# deep JWST fields probe (see _report_survey_densities); if the box is too
# small to reach it, the rarest reachable density is used instead and the
# substitution is reported.  Every reachable density is still tabulated.
_ND_PLOT_TARGET = 1e-5

# Particle cut for this measurement only.  The global MIN_PARTICLES = 50 is too
# aggressive here: at z = 10 the FFB threshold sits near 40 particles in
# Millennium-500, so a 50-particle cut removes the haloes the figure is about
# and truncates the sample before rank n*V is reached, which stops the curves
# at z ~ 10 for no physical reason.
_ND_MIN_PARTICLES = 20

# (directory, snap, n) -> why that point could not be measured.
_ND_TRUNCATION = {}

# Approximate survey areas, used only to report the number density each survey
# probes.  Nothing in the figure depends on these.
_SURVEY_AREAS_ARCMIN2 = {
    'JADES-Deep':  45.0,
    'CEERS':      100.0,
    'COSMOS-Web': 1944.0,   # 0.54 deg^2
}

_BOX_VOLUME_CACHE = {}


def _box_geometry(directory):
    """
    (box side [Mpc/h], hubble_h, processed volume fraction) from the header.

    A run restricted to a subset of tree files covers only part of the box, so
    the number density n = k / V must divide by the volume actually processed,
    not by the full box.  SAGE records this as `frac_volume_processed`; if the
    attribute is missing, fall back to (LastFile - FirstFile + 1) / num files.
    """
    if directory in _BOX_VOLUME_CACHE:
        return _BOX_VOLUME_CACHE[directory]
    box = hh = None
    frac = 1.0
    files = _find_model_files(directory)
    if files:
        try:
            with h5.File(files[0], 'r') as f:
                sim = f['Header/Simulation'].attrs
                run = f['Header/Runtime'].attrs
                box = float(sim['box_size'])
                hh  = float(sim['hubble_h'])
                if 'frac_volume_processed' in run:
                    frac = float(run['frac_volume_processed'])
                elif all(k in run for k in ('FirstFile', 'LastFile')) \
                        and 'num_simulation_tree_files' in sim:
                    n_tot = float(sim['num_simulation_tree_files'])
                    if n_tot > 0:
                        frac = (float(run['LastFile']) -
                                float(run['FirstFile']) + 1.0) / n_tot
                if not (0.0 < frac <= 1.0):
                    print(f'  Warning: implausible volume fraction {frac} in '
                          f'{directory}; treating as 1.0')
                    frac = 1.0
        except Exception as e:
            print(f'  Warning: could not read box geometry from {directory}: {e}')
    _BOX_VOLUME_CACHE[directory] = (box, hh, frac)
    return box, hh, frac


def _box_size_mpc_h(directory):
    """Box side length [Mpc/h] from the HDF5 header, or None."""
    return _box_geometry(directory)[0]


def _box_volume_mpc3(directory):
    """Comoving volume actually processed by the run [Mpc^3]."""
    box, hh, frac = _box_geometry(directory)
    if not box or not hh:
        return None
    return (box / hh) ** 3 * frac


def _mstar_at_number_density(directory, snap, n_targets,
                             centrals_only=False, min_count=_ND_MIN_COUNT):
    """
    log10 m* at fixed cumulative comoving number density.

    Returns {n_target: (log10 m*, log10 m*_lo, log10 m*_hi)}.

    The uncertainty is a delete-one jackknife over the 8 octants of the box,
    which captures cosmic variance as well as shot noise.  Pure Poisson error
    on the cumulative rank is negligible here -- at n = 1e-5 in a 500 Mpc/h box
    the rank is ~3200, so sqrt(k)/k is under 2 per cent and the band would be
    invisible -- while the octant-to-octant scatter is what a survey of that
    volume would actually see.

    A target is omitted when it corresponds to fewer than `min_count` galaxies,
    or to more galaxies than the sample contains: in the latter case the
    density lies below where the particle cut truncates the sample, so it is
    not measurable rather than merely noisy.  `_ND_TRUNCATION` records why, so
    the caller can report it instead of the curve silently stopping.

    All galaxies are used, not only centrals and not only those in the
    efficient mode, because that is what a survey counts.
    """
    vol = _box_volume_mpc3(directory)
    if not vol:
        return {}
    props = ['StellarMass', 'Type', 'Posx', 'Posy', 'Posz']
    d = read_snap(directory, snap, props, min_particles=_ND_MIN_PARTICLES)
    if not d or 'StellarMass' not in d:
        return {}

    keep = d['StellarMass'] > 0
    if centrals_only and 'Type' in d:
        keep &= d['Type'] == 0
    ms = d['StellarMass'][keep]
    if ms.size < min_count:
        return {}

    # Octant label for the jackknife, if positions are available.  With a
    # partial-volume run the octants are unequal, so the jackknife is only
    # approximate there; the caller warns.
    oct_id = None
    if all(p in d for p in ('Posx', 'Posy', 'Posz')):
        box = _box_size_mpc_h(directory)
        if box:
            half = box / 2.0
            oct_id = ((d['Posx'][keep] % box >= half).astype(int) +
                      2 * (d['Posy'][keep] % box >= half).astype(int) +
                      4 * (d['Posz'][keep] % box >= half).astype(int))

    def _mstar_at(mass_arr, volume, k):
        """log10 m* at rank k in a sample of `mass_arr` occupying `volume`."""
        if mass_arr.size < 1 or k < 1 or k > mass_arr.size:
            return np.nan
        srt = np.sort(mass_arr)[::-1]
        rk  = np.arange(1, srt.size + 1)
        return float(np.interp(np.log10(k), np.log10(rk), np.log10(srt)))

    out = {}
    for nt in n_targets:
        k = nt * vol
        if k < min_count:
            _ND_TRUNCATION[(directory, snap, nt)] = (
                f'needs {k:.0f} galaxies, below the {min_count}-object floor')
            continue
        if k > ms.size:
            _ND_TRUNCATION[(directory, snap, nt)] = (
                f'needs rank {k:.0f} but only {ms.size} galaxies survive the '
                f'{_ND_MIN_PARTICLES}-particle cut')
            continue

        centre = _mstar_at(ms, vol, k)
        lo = hi = np.nan

        if oct_id is not None:
            sub = []
            for o in range(8):
                m_sub = ms[oct_id != o]
                v_sub = vol * 7.0 / 8.0
                v_val = _mstar_at(m_sub, v_sub, nt * v_sub)
                if np.isfinite(v_val):
                    sub.append(v_val)
            if len(sub) == 8:
                sub = np.array(sub)
                # delete-one jackknife: sigma^2 = (N-1)/N * sum (x_i - xbar)^2
                sigma = np.sqrt(7.0 / 8.0 * np.sum((sub - sub.mean()) ** 2))
                lo, hi = centre - sigma, centre + sigma

        if not np.isfinite(lo):      # no positions: fall back to Poisson
            sk = np.sqrt(k)
            lo = _mstar_at(ms, vol, min(k + sk, ms.size))
            hi = _mstar_at(ms, vol, max(k - sk, 1.0))

        out[nt] = (centre, lo, hi)
    return out


def _report_survey_densities():
    """Print the number density each survey probes; astropy optional."""
    try:
        from astropy.cosmology import Planck18 as _cos
        import astropy.units as _u
    except Exception:
        print('  (astropy unavailable -- skipping survey volume table)')
        return
    sr_per_arcmin2 = (1.0 / 60.0 * np.pi / 180.0) ** 2
    print('  Number density probed by each survey (one object per volume):')
    for z0, z1 in [(6.5, 7.5), (9.5, 10.5), (13.5, 14.5)]:
        dV = (_cos.comoving_volume(z1) - _cos.comoving_volume(z0)).to(_u.Mpc ** 3).value
        per_arcmin2 = dV * sr_per_arcmin2 / (4.0 * np.pi)
        row = f'    z = {z0:.1f}-{z1:.1f}: '
        row += '  '.join(
            f'{name} n = {1.0 / (per_arcmin2 * area):.1e}'
            for name, area in _SURVEY_AREAS_ARCMIN2.items())
        print(row)


def plot_G_mstar_vs_z_ndensity():
    """
    Stellar mass at fixed cumulative comoving number density vs redshift.

    Number-density-matched counterpart to Plot F.  One curve per model at the
    density _ND_PLOT_TARGET, shaded with the 1-sigma Poisson uncertainty on the
    cumulative count.  The observations are the same as in Plot F and are
    unchanged.

    Reading the figure: a survey covering volume V can find one object at
    n = 1/V, so the plotted density is chosen to match the rarity the JWST
    surveys actually probe (n ~ 10^-5 Mpc^-3 in the deep fields).  A model
    reproduces an observed galaxy when its curve at the matching density passes
    through that point -- not when its median does.
    """
    print('Plot G: stellar mass at fixed comoving number density')

    # MBK25 is drawn as a thick solid line and Li+24 dashed on top of it: the
    # two agree to ~0.01 dex at every redshift, so a single style would hide
    # one curve completely and read as a missing model.
    models = [
        {'label': 'MBK25 galaxies',     'dir': MBK25_DIR, 'color': 'mediumpurple',
         'ls': '-',  'lw': 3.4, 'z': 5},
        {'label': 'FFB galaxies',       'dir': LI24_DIR,  'color': 'black',
         'ls': '--', 'lw': 1.9, 'z': 6},
        {'label': 'No FFB/MBK25 model', 'dir': NOFFB_DIR, 'color': 'firebrick',
         'ls': '-',  'lw': 2.4, 'z': 5},
    ]
    snaps = _ffb_snaps()
    print(f'    particle cut for this figure: Len >= {_ND_MIN_PARTICLES}')

    for m in models:
        v = _box_volume_mpc3(m['dir'])
        if v:
            _, _, frac = _box_geometry(m['dir'])
            note = '' if frac >= 0.999 else f'  [{100 * frac:.1f}% of the box]'
            print(f"    {m['label']:20s} V = {v:.3e} Mpc^3  "
                  f"(1 object -> n = {1.0 / v:.2e} Mpc^-3){note}")
            if frac < 0.999:
                print('      partial volume: the octant jackknife is '
                      'approximate for this run')
    _report_survey_densities()

    # --- gather curves --------------------------------------------------
    # (model label, n) -> (z, logm, logm_lo, logm_hi)
    curves = {}
    for m in models:
        if not _find_model_files(m['dir']):
            print(f"  Skipping {m['label']}: no files in {m['dir']}")
            continue
        per_n = {nt: ([], [], [], []) for nt in _ND_TARGETS}
        for snap in snaps:
            res = _mstar_at_number_density(m['dir'], snap, _ND_TARGETS)
            for nt, (logm, lo, hi) in res.items():
                per_n[nt][0].append(REDSHIFTS[snap])
                per_n[nt][1].append(logm)
                per_n[nt][2].append(lo)
                per_n[nt][3].append(hi)
        for nt, cols in per_n.items():
            if len(cols[0]) >= 3:
                curves[(m['label'], nt)] = tuple(np.array(c) for c in cols)

    reachable = sorted({nt for (_, nt) in curves})
    if not reachable:
        print('  No number density is reachable in this volume -- nothing to plot.')
        return
    unreachable = [nt for nt in _ND_TARGETS if nt not in reachable]
    if unreachable:
        print('  Not reachable in this volume (fewer than '
              f'{_ND_MIN_COUNT} galaxies): '
              + ', '.join(f'{nt:.0e}' for nt in unreachable)
              + ' Mpc^-3 -- run the 500 Mpc/h box for these.')

    # Draw a single density: the JWST-matched one when the box reaches it,
    # otherwise the rarest it can.
    if _ND_PLOT_TARGET in reachable:
        plot_n = _ND_PLOT_TARGET
    else:
        plot_n = reachable[0]
        print(f'  n = {_ND_PLOT_TARGET:.0e} Mpc^-3 is out of reach here; '
              f'plotting n = {plot_n:.0e} Mpc^-3 instead.')

    # No figsize: inherit 8.34 x 6.25 from kieren_cohare_palatino_sty.mplstyle,
    # matching the single-panel figures in paper_plots.py.
    fig, ax = plt.subplots()

    for m in models:
        key = (m['label'], plot_n)
        if key not in curves:
            continue
        zs, lm, lo, hi = curves[key]
        o = np.argsort(zs)
        ax.plot(zs[o], lm[o], color=m['color'], lw=m['lw'], ls=m['ls'],
                zorder=m['z'] + 1)
        ax.fill_between(zs[o], lo[o], hi[o], color=m['color'],
                        alpha=0.2, lw=0, zorder=m['z'])

    # Why does each curve stop where it does?
    stops = []
    for m in models:
        key = (m['label'], plot_n)
        if key not in curves:
            continue
        z_hi = curves[key][0].max()
        reasons = [(REDSHIFTS[s], why) for (dd, s, nt), why
                   in _ND_TRUNCATION.items()
                   if dd == m['dir'] and nt == plot_n and REDSHIFTS[s] > z_hi]
        if reasons:
            z_next, why = min(reasons, key=lambda t: t[0])
            stops.append(f"    {m['label']:20s} stops at z = {z_hi:.2f}; "
                         f"at z = {z_next:.2f} it {why}")
    if stops:
        print(f'  Curve limits at n = {plot_n:.0e} Mpc^-3:')
        print('\n'.join(stops))

    # --- observations: identical to Plot F ------------------------------
    epochs = _load_epochs()
    if epochs is not None:
        in_range = ((epochs['zbest'] >= _Z_RANGE[0]) &
                    (epochs['zbest'] <= _Z_RANGE[1]))
        ep = epochs[in_range]
        ax.scatter(ep['zbest'], ep['stellar_mass_pipes_zgauss'],
                   s=4, color="#656262", marker='o', alpha=0.2,
                   linewidths=0.3, edgecolors='k', zorder=7)

    _baggen_path = os.path.join(SIZE_OBS_DIR, 'baggen_disk_2023.ecsv')
    if os.path.exists(_baggen_path):
        df_b = pd.read_csv(_baggen_path, comment='#', sep=r'\s+')
        df_b = df_b[(df_b['z_phot'] >= _Z_RANGE[0]) & (df_b['z_phot'] <= _Z_RANGE[1])]
        if len(df_b) > 0:
            ax.scatter(df_b['z_phot'], df_b['log_M_star'],
                       s=18, color='darkorange', marker='D', alpha=0.85,
                       linewidths=0.5, edgecolors='k', zorder=8)

    _casey_path = os.path.join(SIZE_OBS_DIR, 'casey_disk_2024.ecsv')
    if os.path.exists(_casey_path):
        df_c = pd.read_csv(_casey_path, comment='#', sep=r'\s+')
        df_c = df_c[(df_c['z_phot_BAGPIPES'] >= _Z_RANGE[0]) &
                    (df_c['z_phot_BAGPIPES'] <= _Z_RANGE[1])]
        if len(df_c) > 0:
            log_m  = np.log10(df_c['M_star'])
            log_eu = np.log10(df_c['M_star'] + df_c['M_star_err_up']) - log_m
            ax.errorbar(df_c['z_phot_BAGPIPES'], log_m, yerr=log_eu,
                        fmt='s', color='dodgerblue', markersize=5,
                        markeredgecolor='k', markeredgewidth=0.5,
                        ecolor='dodgerblue', elinewidth=1.0, capsize=2, zorder=8)

    _sun_path = os.path.join(SIZE_OBS_DIR, 'sun_disk_2024.ecsv')
    if os.path.exists(_sun_path):
        df_s = pd.read_csv(_sun_path, comment='#', sep=r'\s+')
        df_s = df_s[(df_s['z'] >= _Z_RANGE[0]) & (df_s['z'] <= _Z_RANGE[1])]
        if len(df_s) > 0:
            ax.errorbar(df_s['z'], df_s['log_M_star'], yerr=df_s['log_M_star_err'],
                        fmt='^', color='seagreen', markersize=6,
                        markeredgecolor='k', markeredgewidth=0.5,
                        ecolor='seagreen', elinewidth=1.0, capsize=2, zorder=8)

    _above = {"Maisie's", "GS-z11-0"}
    for name, z, logm, err in _JWST_SPEC:
        ax.errorbar(z, logm, yerr=err, fmt='*', color='gold', markersize=11,
                    markeredgecolor='k', markeredgewidth=0.6,
                    ecolor='gold', elinewidth=1.2, capsize=2, zorder=8)
        dy, va = (8, 'bottom') if name in _above else (-8, 'top')
        ax.annotate(name, xy=(z, logm), xytext=(0, dy),
                    textcoords='offset points', fontsize=10, fontweight='bold',
                    ha='center', va=va, color="#0E0C0C", zorder=9)

    # --- legend: one entry per model ------------------------------------
    handles = [mlines.Line2D([], [], color=m['color'], lw=m['lw'],
                             ls=m['ls'], label=m['label'])
               for m in models if (m['label'], plot_n) in curves]
    _standard_legend(ax, loc='upper left', handles=handles,
                     labels=[h.get_label() for h in handles])

    ax.set_xlabel(r'Redshift')
    ax.set_ylabel(r'$\log_{10}\,m_\star\ [M_\odot]$')
    ax.set_xlim(_Z_RANGE[0], _Z_RANGE[1])
    ax.set_ylim(6.5, 12.5)

    fig.tight_layout()
    save_figure(fig, os.path.join(OUTPUT_DIR, 'G_mstar_vs_z_ndensity' + OUTPUT_FORMAT))

    # --- quotable numbers ------------------------------------------------
    print('\n  log10 m* at fixed cumulative number density:')
    header = '    {:>6s}'.format('z') + ''.join(
        f'{m["label"][:9]:>11s}' for m in models)
    for nt in reachable:
        marker = '  <- plotted' if nt == plot_n else ''
        print(f'   n = {nt:.0e} Mpc^-3{marker}')
        print(header)
        zs_all = sorted({z for key, val in curves.items()
                         if key[1] == nt for z in val[0]}, reverse=True)
        for z in zs_all:
            row = f'    {z:6.2f}'
            vals = {}
            for m in models:
                key = (m['label'], nt)
                if key not in curves:
                    row += f'{"--":>11s}'
                    continue
                zs, lm, lo, hi = curves[key]
                i = np.argmin(np.abs(zs - z))
                if abs(zs[i] - z) > 0.05:
                    row += f'{"--":>11s}'
                    continue
                vals[m['label']] = lm[i]
                row += f'{lm[i]:11.2f}'
            if 'FFB galaxies' in vals and 'No FFB/MBK25 model' in vals:
                row += (f'   (FFB - noFFB = '
                        f'{vals["FFB galaxies"] - vals["No FFB/MBK25 model"]:+.2f} dex)')
            print(row)


# ========================== MAIN ==========================

ALL_PLOTS = {
    # 'A': plot_A_ffb_fraction_vs_redshift,
    # 'B': plot_B_ffb_heatmap,
    # 'C': plot_C_mstar_vs_z,
    # 'D': plot_D_ffb_fraction_vs_z,
    # 'E': plot_E_ffb_plane,
    'F': plot_F_mstar_ffb_scatter,
    'G': plot_G_mstar_vs_z_ndensity,
}


def main():
    setup_style()
    keys = [k.upper() for k in sys.argv[1:]] if len(sys.argv) > 1 else list(ALL_PLOTS)
    for key in keys:
        if key in ALL_PLOTS:
            ALL_PLOTS[key]()
        else:
            print(f"Unknown plot '{key}'. Available: {list(ALL_PLOTS)}")


if __name__ == '__main__':
    main()

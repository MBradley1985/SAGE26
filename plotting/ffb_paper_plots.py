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
MBK25_DIR     = './output/millennium_mbk/'
NOFFB_DIR     = './output/millennium_noffb/'
VANILLA_DIR   = './output/millennium_vanilla/'
OUTPUT_DIR    = './output/ffb_paper/plots/'
OBS_DIR       = './data/'
OUTPUT_FORMAT = '.pdf'

# Minimum number of dark matter particles for a halo to be considered resolved.
# Corresponds to the 'Len' field in the SAGE HDF5 output.
MIN_PARTICLES = 50

_MSUN_CGS = 1.989e33

_MASS_PROPS = frozenset({
    'CentralMvir', 'Mvir', 'StellarMass', 'BulgeMass', 'BlackHoleMass',
    'MetalsStellarMass', 'MetalsColdGas', 'MetalsEjectedMass',
    'MetalsHotGas', 'MetalsCGMgas', 'ColdGas', 'HotGas', 'CGMgas',
    'EjectedMass', 'H2gas', 'H1gas', 'IntraClusterStars',
    'MergerBulgeMass', 'InstabilityBulgeMass',
})

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

def read_snap(directory, snap, properties):
    """
    Read properties for a single snapshot, concatenated across MPI files.
    Halos with Len < MIN_PARTICLES are removed before returning.
    """
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
        mask = data['Len'] >= MIN_PARTICLES
        data = {p: arr[mask] for p, arr in data.items()}

    # Drop Len if the caller didn't ask for it
    if not caller_wants_len:
        data.pop('Len', None)

    return data

# ========================== PHYSICS ==========================

def ffb_threshold_mass_msun(z):
    """Li+24 FFB threshold mass [M_sun] from Eq. 2."""
    return 3e11 * (1.0 + z)**(-1.5) / HUBBLE_H


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
    Ez2 = OMEGA_M * (1.0 + z)**3 + OMEGA_L
    x   = OMEGA_M * (1.0 + z)**3 / Ez2 - 1.0
    return 18.0 * np.pi**2 + 82.0 * x - 39.0 * x**2


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
    path = os.path.join(OBS_DIR, 'EPOCHS.csv')
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
    # Sits at ~1 for all resolved Millennium halos since M_vir >> M_FFB(z);
    # any gap from the simulation line exposes the CGM-regime suppression.
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
    (Len >= MIN_PARTICLES, M_star > 10^7 M_sun).

    Non-FFB galaxies: diluted grey background.
    Li+24 FFB galaxies: red.
    MBK25 FFB galaxies: purple.

    Overlaid: EPOCHS photometric catalog and notable JWST spec-z discoveries.
    """
    print('Plot F: stellar mass vs redshift (FFB median lines)')

    props       = ['StellarMass', 'FFBRegime', 'Type']
    snaps       = _ffb_snaps()
    MSTAR_FLOOR = 1e7   # M_sun
    MIN_N       = 10    # minimum galaxies per snapshot to plot a point

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
        _plot_band(z_bg,  med_bg,  lo_bg,  hi_bg,  'firebrick',  'Non-FFB',   zorder=2)
    if z_li:
        _plot_band(z_li,  med_li,  lo_li,  hi_li,  'black',  'Li+24 FFB', zorder=4)
    if z_mbk:
        _plot_band(z_mbk, med_mbk, lo_mbk, hi_mbk, 'mediumpurple',     'MBK25 FFB', zorder=6)

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
    _baggen_path = os.path.join(OBS_DIR, 'baggen_disk_2023.ecsv')
    if os.path.exists(_baggen_path):
        df_b = pd.read_csv(_baggen_path, comment='#', sep=r'\s+')
        in_range = (df_b['z_phot'] >= _Z_RANGE[0]) & (df_b['z_phot'] <= _Z_RANGE[1])
        df_b = df_b[in_range]
        if len(df_b) > 0:
            ax.scatter(df_b['z_phot'], df_b['log_M_star'],
                       s=18, color='darkorange', marker='D', alpha=0.85,
                       linewidths=0.5, edgecolors='k', zorder=8)

    # Casey+24 photometric galaxies (z=9.2–14.4, COSMOS-Web)
    _casey_path = os.path.join(OBS_DIR, 'casey_disk_2024.ecsv')
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
    _sun_path = os.path.join(OBS_DIR, 'sun_disk_2024.ecsv')
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
    _above = {"Maisie's"}
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

    ax.legend(frameon=False, fontsize='small', loc='upper left')

    ax.set_xlabel(r'$z$')
    ax.set_ylabel(r'$\log_{10}\,M_\star\ [M_\odot]$')
    ax.set_xlim(_Z_RANGE[1], _Z_RANGE[0])
    ax.set_ylim(6.5, 12.5)

    fig.tight_layout()
    save_figure(fig, os.path.join(OUTPUT_DIR, 'F_mstar_ffb_scatter' + OUTPUT_FORMAT))


# ========================== MAIN ==========================

ALL_PLOTS = {
    'A': plot_A_ffb_fraction_vs_redshift,
    'B': plot_B_ffb_heatmap,
    'C': plot_C_mstar_vs_z,
    'D': plot_D_ffb_fraction_vs_z,
    'E': plot_E_ffb_plane,
    'F': plot_F_mstar_ffb_scatter,
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

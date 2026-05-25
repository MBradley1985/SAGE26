#!/usr/bin/env python
"""
ICS Paper Plots
===============
Publication-quality figures for the Intracluster Stars paper.

Usage:
    python plotting/ics_paper_plots.py          # all plots
    python plotting/ics_paper_plots.py 1        # plot 1 only
    python plotting/ics_paper_plots.py 1 2      # plots 1 and 2
"""

import glob
import os
import re
import sys
import warnings

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings('ignore')

# ── Style ──────────────────────────────────────────────────────────────────────
_STYLE = './plotting/kieren_cohare_palatino_sty.mplstyle'
if os.path.exists(_STYLE):
    plt.style.use(_STYLE)
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor']   = 'white'
plt.rcParams['axes.edgecolor']   = 'black'


# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_DIR     = './output/microuchuu/'
OUTPUT_FORMAT = '.pdf'
OUTPUT_DIR    = './output/ics_paper/plots/'
OBS_DIR       = './data/'
SIM_LABEL     = 'SAGE26'
SIM_COLOR     = '#2166ac'   # carries through all ICS paper plots

MIN_PARTICLES      = 50        # DM particle threshold for resolved halos
MVIR_CLUSTER       = 1.0e14   # M_sun — cluster selection floor
MIN_SATELLITES     = 2         # minimum satellite count per central
BCG_MVIR_RATIO_MIN = 10**-3.5  # pathological BCG filter: M_BCG/M_vir
DIST_300KPC        = 300.0     # kpc physical — aperture for panels 3 & 4
MAX_REDSHIFT       = 2.5       # maximum redshift to show
MIN_N_SNAP         = 1        # minimum objects per snapshot to plot

# Definition of “BCG” used in Plot 1 (selection/filtering):
# - 'central': Type==0 stellar mass
# - 'most_massive': max stellar mass among all members of the central's halo
BCG_DEFINITION = 'most_massive'

# Analytic proxy for aperture-limited masses in Plot 1 (panel row 2).
# Concentrations are taken from Ishiyama+21 lookup tables when available.
ICS_PROXY_OMEGA_M0 = 0.25
ICS_PROXY_OMEGA_L0 = 0.75
ICS_PROXY_C        = 4.0
BCG_PROXY_C        = 7.0
BCG_PROXY_C_TO_HALO = BCG_PROXY_C / ICS_PROXY_C

# ── Multi-model configuration (Plots 4–5) ──────────────────────────────────────
MODELS = [
    {
        'dir':   './output/millennium_fixed100/',
        'label': r'Fixed ($\alpha=1.0$)',
        'color': '#d6604d',
        'ls':    '--',
    },
    {
        'dir':   './output/millennium_fixed80/',
        'label': r'Fixed ($\alpha=0.8$)',
        'color': '#e08214',
        'ls':    '-.',
    },
    {
        'dir':   './output/microuchuu/',
        'label': 'Dynamic (fiducial)',
        'color': '#2166ac',
        'ls':    '-',
    },
]

# Target redshifts for the redshift-evolution plot (Plot 5)
_BCG_TARGET_Z = [0.0, 0.5, 1.0, 2.0]

# Mass properties requiring unit conversion (code units -> M_sun)
_MASS_PROPS = frozenset({
    'Mvir', 'CentralMvir', 'StellarMass', 'BulgeMass', 'BlackHoleMass',
    'MetalsStellarMass', 'MetalsColdGas', 'MetalsEjectedMass',
    'MetalsHotGas', 'MetalsCGMgas', 'ColdGas', 'HotGas', 'CGMgas',
    'EjectedMass', 'H2gas', 'H1gas', 'IntraClusterStars',
    'MergerBulgeMass', 'InstabilityBulgeMass',
})

_DEFAULT_LEN_CUT = object()

# plt.style.use("./plotting/kieren_cohare_palatino_sty.mplstyle")

# ── Data I/O ───────────────────────────────────────────────────────────────────
def _find_files(directory):
    files = sorted(glob.glob(os.path.join(directory, 'model_*.hdf5')))
    if not files:
        single = os.path.join(directory, 'model_0.hdf5')
        if os.path.exists(single):
            files = [single]
    return files


def _read_header(directory):
    """Read simulation header from first model file. Returns dict or None."""
    files = _find_files(directory)
    if not files:
        return None
    try:
        with h5.File(files[0], 'r') as f:
            sim     = f['Header/Simulation']
            runtime = f['Header/Runtime']
            h_val   = float(sim.attrs['hubble_h'])
            hdr = {
                'hubble_h':       h_val,
                'box_size':       float(sim.attrs['box_size']),   # Mpc/h comoving
                'baryon_frac':    float(runtime.attrs.get('BaryonFrac', 0.17)),
                'unit_mass_in_g': float(runtime.attrs['UnitMass_in_g']),
                # particle_mass is in code units (1e10 M_sun/h); convert to M_sun
                'part_mass_msun': float(sim.attrs['particle_mass']) * 1e10 / h_val,
                'redshifts':      list(f['Header/snapshot_redshifts'][:]),
                'output_snaps':   sorted(
                    int(k.replace('Snap_', ''))
                    for k in f.keys() if k.startswith('Snap_')),
            }
        fvp = 0.0
        for fp in files:
            with h5.File(fp, 'r') as f:
                fvp += float(f['Header/Runtime'].attrs['frac_volume_processed'])
        hdr['volume_fraction'] = fvp
    except Exception as e:
        print(f'  Warning: could not read header: {e}')
        return None
    return hdr


def _load_snap(files, snap_num, props, mass_conv, min_len=_DEFAULT_LEN_CUT):
    """
    Load *props* from Snap_<snap_num> across all HDF5 *files*.
    Applies mass conversion and (optionally) a Len>=MIN_PARTICLES filter.
    Returns dict {prop: array} or empty dict.
    """
    if min_len is _DEFAULT_LEN_CUT:
        min_len = MIN_PARTICLES
    snap_key  = f'Snap_{snap_num}'
    need_len  = 'Len' not in props
    all_props = list(props) + (['Len'] if need_len else [])

    chunks = {p: [] for p in all_props}
    found  = False
    for fp in files:
        try:
            with h5.File(fp, 'r') as f:
                if snap_key not in f:
                    continue
                found = True
                grp = f[snap_key]
                for p in all_props:
                    if p in grp:
                        chunks[p].append(np.array(grp[p]))
        except Exception as e:
            print(f'  Warning: could not read {fp}: {e}')

    if not found:
        return {}

    data = {}
    for p in all_props:
        if chunks[p]:
            arr = np.concatenate(chunks[p])
            if p in _MASS_PROPS:
                arr = arr * mass_conv
            data[p] = arr

    if 'Len' in data and min_len is not None:
        mask = data['Len'] >= min_len
        data = {p: a[mask] for p, a in data.items()}
        if need_len:
            data.pop('Len', None)

    return data


# ── Satellite helpers ───────────────────────────────────────────────────────────

def _sat_sums_all(Type, SM, GalIdx, CenGalIdx):
    """Satellite stellar-mass sum and count per galaxy (full halo)."""
    sort_ord = np.argsort(GalIdx)
    s_gids   = GalIdx[sort_ord]
    sat_mask = Type != 0
    cen_gids = CenGalIdx[sat_mask]
    sat_sm   = SM[sat_mask]
    ins      = np.searchsorted(s_gids, cen_gids)
    ins      = np.clip(ins, 0, len(s_gids) - 1)
    matched  = s_gids[ins] == cen_gids
    c_idx    = np.where(matched, sort_ord[ins], -1)
    ok       = c_idx >= 0
    n_sat    = np.zeros(len(Type), dtype=np.int32)
    sm_sat   = np.zeros(len(Type), dtype=float)
    np.add.at(n_sat,  c_idx[ok], 1)
    np.add.at(sm_sat, c_idx[ok], sat_sm[ok])
    return n_sat, sm_sat


def _sat_max_all(Type, SM, GalIdx, CenGalIdx):
    """Maximum satellite stellar mass per central galaxy (full halo)."""
    sort_ord = np.argsort(GalIdx)
    s_gids   = GalIdx[sort_ord]
    sat_mask = Type != 0
    cen_gids = CenGalIdx[sat_mask]
    sat_sm   = SM[sat_mask]
    ins      = np.searchsorted(s_gids, cen_gids)
    ins      = np.clip(ins, 0, len(s_gids) - 1)
    matched  = s_gids[ins] == cen_gids
    c_idx    = np.where(matched, sort_ord[ins], -1)
    ok       = c_idx >= 0
    max_sat  = np.zeros(len(Type), dtype=float)
    np.maximum.at(max_sat, c_idx[ok], sat_sm[ok])
    return max_sat


def _sat_sums_within_radius(Type, SM, GalIdx, CenGalIdx,
                             Px, Py, Pz, z, hubble_h, box_size,
                             threshold_kpc=DIST_300KPC):
    """
    Satellite stellar-mass sum and count per galaxy for satellites within
    *threshold_kpc* physical of the central.

    Positions are comoving Mpc/h; periodic boundaries applied.
    Physical distance [kpc] = comoving separation [Mpc/h] * 1000 / h / (1+z)
    """
    thr_mpch = threshold_kpc / 1000.0 * hubble_h * (1.0 + z)

    sort_ord = np.argsort(GalIdx)
    s_gids   = GalIdx[sort_ord]
    sat_mask = Type != 0
    cen_gids = CenGalIdx[sat_mask]
    sat_sm   = SM[sat_mask]
    spx, spy, spz = Px[sat_mask], Py[sat_mask], Pz[sat_mask]

    ins     = np.searchsorted(s_gids, cen_gids)
    ins     = np.clip(ins, 0, len(s_gids) - 1)
    matched = s_gids[ins] == cen_gids
    c_idx   = np.where(matched, sort_ord[ins], -1)
    ok      = c_idx >= 0

    safe_c = c_idx.clip(0)
    cpx = np.where(ok, Px[safe_c], 0.0)
    cpy = np.where(ok, Py[safe_c], 0.0)
    cpz = np.where(ok, Pz[safe_c], 0.0)

    dx = spx - cpx;  dx -= box_size * np.round(dx / box_size)
    dy = spy - cpy;  dy -= box_size * np.round(dy / box_size)
    dz = spz - cpz;  dz -= box_size * np.round(dz / box_size)
    dist   = np.sqrt(dx**2 + dy**2 + dz**2)
    within = ok & (dist <= thr_mpch)

    n_sat  = np.zeros(len(Type), dtype=np.int32)
    sm_sat = np.zeros(len(Type), dtype=float)
    np.add.at(n_sat,  c_idx[within], 1)
    np.add.at(sm_sat, c_idx[within], sat_sm[within])
    return n_sat, sm_sat


_ISHIYAMA_TABLE_CACHE = None


def _parse_c_1d_array(text, name):
    pat = rf'const double {name}\[[^\]]+\]\s*=\s*\{{(.*?)\}};'
    m = re.search(pat, text, flags=re.S)
    if m is None:
        raise ValueError(f'Could not find {name} in model_misc.c')
    vals = [float(v) for v in re.findall(r'-?\d+\.\d+', m.group(1))]
    return np.array(vals, dtype=float)


def _parse_c_2d_array(text, name, n_mass, n_z):
    pat = rf'const double {name}\[[^\]]+\]\[[^\]]+\]\s*=\s*\{{(.*?)\}};'
    m = re.search(pat, text, flags=re.S)
    if m is None:
        raise ValueError(f'Could not find {name} in model_misc.c')
    vals = [float(v) for v in re.findall(r'-?\d+\.\d+', m.group(1))]
    arr = np.array(vals, dtype=float)
    if arr.size != n_mass * n_z:
        raise ValueError(f'Unexpected shape for {name}: got {arr.size}, expected {n_mass * n_z}')
    return arr.reshape((n_mass, n_z))


def _load_ishiyama_tables_from_c():
    """Load the Ishiyama+21 lookup tables from src/model_misc.c (same as runtime C code)."""
    global _ISHIYAMA_TABLE_CACHE
    if _ISHIYAMA_TABLE_CACHE is not None:
        return _ISHIYAMA_TABLE_CACHE

    path = './src/model_misc.c'
    if not os.path.exists(path):
        _ISHIYAMA_TABLE_CACHE = None
        return None

    try:
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        logm = _parse_c_1d_array(text, 'cm_table_logmass')
        ztab = _parse_c_1d_array(text, 'cm_table_z')
        base = _parse_c_2d_array(text, 'cm_table', len(logm), len(ztab))
        uchuu = _parse_c_2d_array(text, 'cm_uchuu_table', len(logm), len(ztab))
        _ISHIYAMA_TABLE_CACHE = {'logm': logm, 'z': ztab, 'base': base, 'uchuu': uchuu}
    except Exception as e:
        print(f'  Warning: could not parse Ishiyama table from {path}: {e}')
        _ISHIYAMA_TABLE_CACHE = None

    return _ISHIYAMA_TABLE_CACHE


def _ishiyama_concentration_lookup(logm, z, omega_m0):
    """Bilinear-interpolate Ishiyama+21 c(M,z) table with same table selection rule as C code."""
    tables = _load_ishiyama_tables_from_c()
    if tables is None:
        return None

    grid_m = tables['logm']
    grid_z = tables['z']
    table = tables['uchuu'] if abs(omega_m0 - 0.3089) < 0.01 else tables['base']

    lm = np.clip(np.asarray(logm, dtype=float), grid_m[0], grid_m[-1])
    zz = np.clip(float(z), grid_z[0], grid_z[-1])

    im = np.searchsorted(grid_m, lm, side='right') - 1
    im = np.clip(im, 0, len(grid_m) - 2)
    iz = int(np.searchsorted(grid_z, zz, side='right') - 1)
    iz = max(0, min(iz, len(grid_z) - 2))

    fm = (lm - grid_m[im]) / (grid_m[im + 1] - grid_m[im])
    fz = (zz - grid_z[iz]) / (grid_z[iz + 1] - grid_z[iz])

    c00 = table[im, iz]
    c10 = table[im + 1, iz]
    c01 = table[im, iz + 1]
    c11 = table[im + 1, iz + 1]

    return (
        c00 * (1.0 - fm) * (1.0 - fz)
        + c10 * fm * (1.0 - fz)
        + c01 * (1.0 - fm) * fz
        + c11 * fm * fz
    )


def _halo_concentration_for_proxy(mvir, z, hubble_h,
                                  omega_m0=ICS_PROXY_OMEGA_M0):
    """Return halo concentration c(M,z) for proxy aperture masses."""
    mvir_msun_h = np.clip(mvir * hubble_h, 1e-30, None)
    c_halo = _ishiyama_concentration_lookup(np.log10(mvir_msun_h), z, omega_m0)
    if c_halo is None:
        c_halo = np.full_like(mvir, ICS_PROXY_C, dtype=float)
    return c_halo


def _mass_within_radius_nfw_proxy(total_mass, mvir, z,
                                  radius_kpc=DIST_300KPC,
                                  c_nfw=ICS_PROXY_C,
                                  omega_m0=ICS_PROXY_OMEGA_M0,
                                  omega_l0=ICS_PROXY_OMEGA_L0):
    """
    Approximate aperture-limited mass with an NFW cumulative profile.

    We map each halo mass to a physical R200 and use:
      M(<r) / M(<R200) = f(x) / f(c),  f(y)=ln(1+y)-y/(1+y), x=c*r/R200.
    """
    e_z = np.sqrt(omega_m0 * (1.0 + z)**3 + omega_l0)
    # R200 physical in kpc for M200 in M_sun (standard scaling).
    r200_kpc = 206.0 * np.cbrt(np.clip(mvir, 0.0, None) / 1.0e12) / np.cbrt(e_z**2)
    x = np.where(r200_kpc > 0.0, c_nfw * radius_kpc / r200_kpc, 0.0)

    f_x = np.log1p(x) - x / (1.0 + x)
    f_c = np.log1p(c_nfw) - c_nfw / (1.0 + c_nfw)

    frac = np.where(f_c > 0.0, f_x / f_c, 0.0)
    frac = np.clip(frac, 0.0, 1.0)
    return total_mass * frac


def _ics_mass_within_radius_proxy(ics_total, mvir, z, hubble_h,
                                  omega_m0=ICS_PROXY_OMEGA_M0,
                                  c_halo=None):
    """Approximate ICS mass within 300 kpc using Ishiyama+21 c(M,z) where available."""
    if c_halo is None:
        c_halo = _halo_concentration_for_proxy(mvir, z, hubble_h, omega_m0)
    return _mass_within_radius_nfw_proxy(ics_total, mvir, z, c_nfw=c_halo)


def _bcg_mass_within_radius_proxy(sm_total, mvir, z, hubble_h,
                                  omega_m0=ICS_PROXY_OMEGA_M0,
                                  c_halo=None):
    """Approximate central stellar mass within 300 kpc from Ishiyama+21 halo c(M,z)."""
    if c_halo is None:
        c_halo = _halo_concentration_for_proxy(mvir, z, hubble_h, omega_m0)
    c_bcg = np.maximum(1.0, BCG_PROXY_C_TO_HALO * c_halo)
    return _mass_within_radius_nfw_proxy(sm_total, mvir, z, c_nfw=c_bcg)


# ── Observational data ──────────────────────────────────────────────────────────

def _cluster_obs():
    """
    Observational ICS fractions for clusters in the stellar-fraction
    definition: M_ICS / M_star_total.  Same dataset used on all four panels.
    """
    return [
        {'label': 'Burke+12',
         'z': [0.947, 0.830, 0.795, 0.808, 1.223],
         'f': np.array([1.42, 2.59, 3.77, 1.53, 2.36]) / 100,
         'marker': 'p',  'color': '#666666'},
        {'label': r'Montes \& Trujillo 18',
         'z': [0.534, 0.544, 0.367, 0.397, 0.342, 0.304, 0.048,
               0.301, 0.390, 0.342, 0.537, 0.537, 0.370, 0.043],
         'f': np.array([1.53, 0.0, 1.06, 1.53, 2.71, 3.30, 8.61,
                        7.67, 8.61, 13.09, 6.60, 5.78, 4.83, 10.85]) / 100,
         'marker': 'o',  'color': '#777777'},
        {'label': 'Burke+15',
         'z': [0.403, 0.387, 0.397, 0.339, 0.344, 0.342, 0.291, 0.225,
               0.218, 0.213, 0.195, 0.177],
         'f': np.array([2.59, 2.71, 3.30, 5.54, 6.01, 7.19, 12.97, 12.50,
                        16.27, 18.04, 16.86, 23.11]) / 100,
         'marker': 's',  'color': '#666666'},
        {'label': 'Furnell+21',
         'z': [0.144, 0.127, 0.122, 0.081, 0.225, 0.215, 0.256, 0.306,
               0.261, 0.294, 0.322, 0.342, 0.372, 0.337, 0.377, 0.329,
               0.496, 0.425, 0.109],
         'f': np.array([38.56, 30.66, 31.01, 28.89, 26.53, 23.58, 28.54,
                        29.72, 32.55, 27.48, 27.59, 26.65, 19.81, 18.87,
                        15.45, 15.33, 11.32, 9.67, 31.60]) / 100,
         'marker': '^',  'color': '#777777'},
        {'label': 'Feldmeier+04',
         'z': [0.162, 0.162, 0.162, 0.185],
         'f': np.array([15.21, 12.15, 10.26, 7.31]) / 100,
         'marker': 'v',  'color': '#666666'},
        {'label': r'Ko \& Jee 18',
         'z': [1.238],
         'f': np.array([9.91]) / 100,
         'marker': '*',  'color': '#555555'},
        {'label': 'Kluge+21',
         'z': [0.030],
         'f': np.array([17.92]) / 100,
         'marker': 'D',  'color': '#666666'},
        {'label': 'Zibetti+05',
         'z': [0.243],
         'f': np.array([10.85]) / 100,
         'marker': 'P',  'color': '#777777'},
        {'label': 'Presotto+14',
         'z': [0.435, 0.433],
         'f': np.array([12.26, 5.54]) / 100,
         'marker': 'h',  'color': '#666666'},
        {'label': 'Spavone+20',
         'z': [0.0],
         'f': np.array([34.08]) / 100,
         'marker': '<',  'color': '#555555'},
        {'label': 'JWST XLSSC 122',
         'z': [1.98],
         'f': np.array([17.0]) / 100,
         'marker': '*',  'color': 'goldenrod'},
    ]


def _plot_obs(ax, obs_list, ms=6, alpha=0.75, zorder=4):
    """Scatter all observational data onto *ax*. Returns legend handles."""
    handles = []
    seen    = set()
    for ob in obs_list:
        zz   = np.array(ob['z'])
        ff   = np.array(ob['f'])
        good = ff > 0
        if not good.any():
            continue
        ax.scatter(zz[good], ff[good],
                   marker=ob['marker'], color=ob['color'],
                   s=ms**2, alpha=alpha, zorder=zorder,
                   linewidths=0.6, edgecolors='none')
        if ob['label'] not in seen:
            handles.append(Line2D([0], [0], marker=ob['marker'],
                                  color=ob['color'], ls='none',
                                  ms=ms, label=ob['label']))
            seen.add(ob['label'])
    return handles


# ── Plot 1: 1x3 ICS fraction vs redshift ───────────────────────────────────────

def plot_1_ics_fraction_vs_redshift():
    """
        1x3 row of cluster ICS fraction vs redshift, one panel per definition:
            (0) M_ICS / (f_b * M_vir)                               baryon-budget
            (1) M_ICS / M_star_tot                                   full-halo stellar
            (2) M_ICS(<300 kpc) / M_star_tot(<300 kpc)                    aperture stellar

    Cluster selection: M_vir >= 10^14 M_sun, >= 2 satellites, BCG filter.
    Line = median (50th percentile); shading = 16th-84th percentile.
    """
    print('\n' + '=' * 70)
    print('Plot 1: 1x3 ICS fraction vs redshift')
    print('=' * 70)
    print(f'  BCG definition: {BCG_DEFINITION}')

    files = _find_files(MODEL_DIR)
    if not files:
        print(f'  No model files found in {MODEL_DIR}')
        return

    hdr = _read_header(MODEL_DIR)
    if hdr is None:
        return

    h_val     = hdr['hubble_h']
    box       = hdr['box_size']       # Mpc/h comoving
    f_b       = hdr['baryon_frac']
    redshifts = hdr['redshifts']
    avail     = hdr['output_snaps']
    mass_conv = hdr['unit_mass_in_g'] / 1.989e33 / h_val   # code units -> M_sun
    min_stel  = hdr['part_mass_msun'] * f_b                 # 1 DM particle * f_b

    snaps = sorted(
        [s for s in avail
         if s < len(redshifts) and 0.0 <= redshifts[s] <= MAX_REDSHIFT],
        reverse=True)

    if not snaps:
        print(f'  No snapshots in z <= {MAX_REDSHIFT}')
        return

    print(f'  {len(snaps)} snapshots, '
          f'z = {redshifts[snaps[-1]]:.2f} to {redshifts[snaps[0]]:.2f}')

    # per-panel: list of (z, p50, p16, p84)
    panel_pts = [[] for _ in range(3)]

    for snap in snaps:
        z = redshifts[snap]

        # IMPORTANT: do NOT apply the Len cut globally here.
        # Orphan / poorly-resolved satellites can have Len~0 but still carry
        # non-negligible stellar mass and should contribute to m_{*,tot}.
        # We apply Len>=MIN_PARTICLES only when selecting the resolved *central*.
        d = _load_snap(
            files, snap,
            ['Mvir', 'IntraClusterStars', 'StellarMass',
             'Type', 'GalaxyIndex', 'CentralGalaxyIndex',
             'Posx', 'Posy', 'Posz', 'Len'],
            mass_conv,
            min_len=None,
        )
        if not d or d['Mvir'].size == 0:
            continue

        Mvir  = d['Mvir']
        ICS   = d['IntraClusterStars']
        SM    = d['StellarMass']
        Type  = d['Type']
        Len   = d['Len']
        Gidx  = d['GalaxyIndex']
        CGidx = d['CentralGalaxyIndex']
        Px, Py, Pz = d['Posx'], d['Posy'], d['Posz']

        n_sat_all, sm_sat_all = _sat_sums_all(Type, SM, Gidx, CGidx)
        n_sat_300, sm_sat_300 = _sat_sums_within_radius(
            Type, SM, Gidx, CGidx, Px, Py, Pz, z, h_val, box)
        c_halo = _halo_concentration_for_proxy(Mvir, z, h_val)
        c_bcg = np.maximum(1.0, BCG_PROXY_C_TO_HALO * c_halo)
        ics_300 = _ics_mass_within_radius_proxy(ICS, Mvir, z, h_val, c_halo=c_halo)
        sm_300 = _bcg_mass_within_radius_proxy(SM, Mvir, z, h_val, c_halo=c_halo)

        total_all = SM + sm_sat_all + ICS
        total_300 = sm_300 + sm_sat_300 + ics_300

        # Diagnostic: is the Type==0 central actually the most massive galaxy?
        max_sat_sm = _sat_max_all(Type, SM, Gidx, CGidx)

        if BCG_DEFINITION == 'central':
            sm_bcg = SM
        elif BCG_DEFINITION == 'most_massive':
            sm_bcg = np.maximum(SM, max_sat_sm)
        else:
            raise ValueError(f'Unknown BCG_DEFINITION={BCG_DEFINITION!r}')

        bcg_mvir = np.where(Mvir > 0, sm_bcg / Mvir, 0.0)

        sel = (
            (Type == 0) &
            (Len >= MIN_PARTICLES) &
            (Mvir >= MVIR_CLUSTER) &
            (ICS > 0) &
            (sm_bcg >= min_stel) &
            (n_sat_all >= MIN_SATELLITES) &
            (total_all > 0) &
            (bcg_mvir >= BCG_MVIR_RATIO_MIN)
        )
        idx = np.where(sel)[0]
        if len(idx) < MIN_N_SNAP:
            continue

        t3 = total_300[idx]
        fracs = [
            ICS[idx] / (f_b * Mvir[idx]),
            ICS[idx] / total_all[idx],
            np.where(t3 > 0, ics_300[idx] / t3, np.nan),
        ]

        for pi, fv in enumerate(fracs):
            fv = fv[np.isfinite(fv)]
            if len(fv) < MIN_N_SNAP:
                continue
            p16, p50, p84 = np.percentile(fv, [16, 50, 84])
            panel_pts[pi].append((z, p50, p16, p84))

        logm_med = np.nanmedian(np.log10(np.clip(Mvir[idx], 1e-30, None)))
        sat_share = np.nanmedian(np.where(t3 > 0, sm_sat_300[idx] / t3, np.nan))
        bcg_share = np.nanmedian(np.where(t3 > 0, sm_300[idx] / t3, np.nan))
        noncentral_bcg = np.mean(max_sat_sm[idx] > SM[idx])
        central_self = np.mean(CGidx[idx] == Gidx[idx])
        central_frac_all = np.nanmedian(SM[idx] / total_all[idx])
        bcg_frac_all = np.nanmedian(sm_bcg[idx] / total_all[idx])
        sat_frac_all = np.nanmedian(sm_sat_all[idx] / total_all[idx])
        ics_frac_all = np.nanmedian(ICS[idx] / total_all[idx])

        print(f'  snap={snap:2d}  z={z:.3f}  N={len(idx):3d}  '
              + '  '.join(f'p{i}={np.nanmedian(f):.4f}' for i, f in enumerate(fracs))
              + f'  c_halo={np.nanmedian(c_halo[idx]):.2f}'
              + f'  c_bcg={np.nanmedian(c_bcg[idx]):.2f}'
              + f'  logMvir={logm_med:.2f}'
              + f'  sat300={sat_share:.3f}'
              + f'  bcg300={bcg_share:.3f}'
              + f'  bcg_is_central={1.0 - noncentral_bcg:.3f}'
              + f'  central_self={central_self:.3f}'
              + f'  (central,BCG,sat,ICS)_all=({central_frac_all:.3f},{bcg_frac_all:.3f},{sat_frac_all:.3f},{ics_frac_all:.3f})')

    # ── Figure ────────────────────────────────────────────────────────────────
    panel_titles = [
        r'$f_{ICS}=m_\mathrm{ICS}\ /\ (f_b\,M_\mathrm{vir})$',
        r'$f_{ICS}=m_\mathrm{ICS}\ /\ m_{\star,\mathrm{tot}}$',
        r'$f_{ICS}=m_\mathrm{ICS}^{300\,\mathrm{kpc}}\ /\ m_{\star,\mathrm{tot}}^{\,300\,\mathrm{kpc}}$',
    ]

    obs       = _cluster_obs()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    axes_flat = np.atleast_1d(axes).flatten()

    x_ticks = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]

    # Display panels in the requested order with the removed panel dropped: 2, 3, 1.
    # (i.e. old indices 1,2,0 mapped onto axes_flat 0,1,2)
    panel_order = [1, 2, 0]

    for pi, ax in enumerate(axes_flat):
        src = panel_order[pi]
        pts = panel_pts[src]
        if pts:
            pts.sort(key=lambda t: t[0])
            zz  = np.array([t[0] for t in pts])
            med = np.array([t[1] for t in pts])
            lo  = np.array([t[2] for t in pts])
            hi  = np.array([t[3] for t in pts])
            ax.fill_between(zz, lo, hi, color=SIM_COLOR, alpha=0.2, lw=0.0)
            ax.plot(zz, med, color=SIM_COLOR, lw=2.5)

        _plot_obs(ax, obs)

        # ax.text(0.97, 0.97, panel_titles[src],
        #         transform=ax.transAxes, ha='right', va='top')
        
        # Make panel titles actual sub-plot titles, not legend entries
        ax.set_title(panel_titles[src], fontsize=16, pad=10)

        ax.set_xlim(0, MAX_REDSHIFT)
        ax.set_xticks(x_ticks)
        ax.set_ylim(0, 0.6)
        ax.tick_params(which='both', direction='in', top=True, right=True)

        ax.set_xlabel(r'Redshift')
        if pi != 0:
            ax.tick_params(labelleft=False)

    axes_flat[0].set_ylabel(r'ICS fraction')

    fig.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, 'ICS_fraction_vs_redshift' + OUTPUT_FORMAT)
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'\n  Saved: {out}')


# ── Plot 2: ICS mass function (4x2 redshift grid) ──────────────────────────────

# Target redshifts for the 4x2 grid panels
_MF_TARGET_Z = [0, 0.5, 1, 2, 3, 4, 5, 6]

# Halo mass bins for coloured sub-populations
_MF_HALO_BINS   = [10, 12.0, 14.0, 17.0]   # log10(M_sun) edges
_MF_HALO_COLORS = ['firebrick', 'green', 'slateblue']
_MF_HALO_LABELS = [
    r'$10^{10} < M_\mathrm{vir} < 10^{12}\ M_\odot$',
    r'$10^{12} < M_\mathrm{vir} < 10^{14}\ M_\odot$',
    r'$M_\mathrm{vir} > 10^{14}\ M_\odot$',
]


def _snap_nearest_z(redshifts, target_z, available):
    """Return the available snapshot number whose redshift is closest to target_z."""
    arr = np.array(redshifts)
    best = min(available, key=lambda s: abs(arr[s] - target_z))
    return best


def plot_2_ics_mass_function():
    """
    4x2 grid of ICS mass functions, one panel per target redshift.
    All resolved centrals with ICS > 0 (no halo mass threshold).
    Coloured lines = halo mass sub-populations; black = all resolved halos.
    """
    print('\n' + '=' * 70)
    print('Plot 2: ICS mass function')
    print('=' * 70)

    files = _find_files(MODEL_DIR)
    if not files:
        print(f'  No model files found in {MODEL_DIR}')
        return

    hdr = _read_header(MODEL_DIR)
    if hdr is None:
        return

    h_val     = hdr['hubble_h']
    redshifts = hdr['redshifts']
    avail     = hdr['output_snaps']
    mass_conv = hdr['unit_mass_in_g'] / 1.989e33 / h_val
    volume    = (hdr['box_size'] / h_val)**3 * hdr['volume_fraction']  # Mpc^3

    # Map each target redshift to its nearest available snapshot
    snaps = [_snap_nearest_z(redshifts, tz, avail) for tz in _MF_TARGET_Z]
    # Deduplicate while preserving order
    seen = set(); snaps_unique = []
    for s in snaps:
        if s not in seen:
            snaps_unique.append(s); seen.add(s)
    snaps = snaps_unique

    # Mass function binning
    log_ics_lo  = 4.5
    log_ics_hi  = 12.75
    bin_width   = 0.25
    n_bins      = int((log_ics_hi - log_ics_lo) / bin_width)
    bin_centers = np.linspace(log_ics_lo, log_ics_hi, n_bins, endpoint=False) + 0.5 * bin_width

    n_panels = len(snaps)
    ncols = 2
    nrows = (n_panels + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, nrows * 4.0), sharey=True)
    axes_flat  = axes.flatten()

    for pi, snap in enumerate(snaps):
        ax = axes_flat[pi]
        z  = redshifts[snap]

        d = _load_snap(files, snap,
                       ['Mvir', 'IntraClusterStars', 'StellarMass', 'Type'],
                       mass_conv)

        if not d or d['Mvir'].size == 0:
            ax.text(0.5, 0.5, f'z = {z:.2f}\nNo data',
                    transform=ax.transAxes, ha='center', va='center')
            _format_mf_ax(ax, pi, ncols, nrows)
            continue

        Mvir = d['Mvir']
        ICS  = d['IntraClusterStars']
        Type = d['Type']

        # All resolved centrals with ICS — no halo mass threshold
        sel  = (Type == 0) & (ICS > 0)
        Mvir_sel    = Mvir[sel]
        log_ICS_sel = np.log10(ICS[sel])
        log_Mvir    = np.log10(Mvir_sel)

        print(f'  snap={snap:2d}  z={z:.2f}  N={sel.sum():5d}')

        # Halo mass sub-populations
        for bi in range(len(_MF_HALO_BINS) - 1):
            lo = _MF_HALO_BINS[bi];  hi = _MF_HALO_BINS[bi + 1]
            mask = (log_Mvir >= lo) & (log_Mvir < hi)
            vals = log_ICS_sel[mask]
            if len(vals) == 0:
                continue
            counts, _ = np.histogram(vals, bins=n_bins,
                                     range=(log_ics_lo, log_ics_hi))
            phi = counts / volume / bin_width
            good = phi > 0
            ax.plot(bin_centers[good], phi[good],
                    color=_MF_HALO_COLORS[bi], lw=2.0, alpha=0.85)
            ax.fill_between(bin_centers[good], 1e-12, phi[good],
                            color=_MF_HALO_COLORS[bi], alpha=0.10)

        # Overall (all resolved halos)
        counts, _ = np.histogram(log_ICS_sel, bins=n_bins,
                                 range=(log_ics_lo, log_ics_hi))
        phi  = counts / volume / bin_width
        good = phi > 0
        ax.plot(bin_centers[good], phi[good], color='black', lw=2.0)

        ax.text(0.95, 0.95, f'$z = {z:.2f}$',
                transform=ax.transAxes, ha='right', va='top')

        _format_mf_ax(ax, pi, ncols, nrows)

    # Hide unused panels
    for pi in range(len(snaps), len(axes_flat)):
        axes_flat[pi].set_visible(False)

    # Legend below figure
    from matplotlib.lines import Line2D as _L2D
    handles = [_L2D([0], [0], color=c, lw=2) for c in _MF_HALO_COLORS]
    handles.append(_L2D([0], [0], color='black', lw=2))
    labels  = _MF_HALO_LABELS + ['All resolved haloes']
    fig.tight_layout()
    fig.legend(handles, labels, loc='lower center',
               bbox_to_anchor=(0.5, -0.02),
               ncol=4, frameon=False)
    fig.subplots_adjust(bottom=0.06)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, 'ICS_mass_function' + OUTPUT_FORMAT)
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'\n  Saved: {out}')


def _format_mf_ax(ax, pi, ncols, nrows):
    """Apply consistent axis formatting to a mass-function panel."""
    ax.set_yscale('log')
    ax.set_xlim(5.5, 12.5)
    ax.set_ylim(1e-6, 1e-2)
    ax.set_xticks([6, 7, 8, 9, 10, 11, 12])
    ax.tick_params(which='both', direction='in', top=True, right=True)

    col = pi % ncols
    row = pi // ncols

    y_ticks  = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    y_labels = ['-2', '-3', '-4', '-5', '-6']
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)   # shared axis: set labels on every panel
    if col == 0:
        ax.tick_params(labelleft=True)
        ax.set_ylabel(r'$\log_{10}(\phi\ [\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1}])$')
    else:
        ax.tick_params(labelleft=False)

    if row != nrows - 1:
        ax.tick_params(labelbottom=False)
    else:
        ax.set_xlabel(r'$\log_{10}(m_\mathrm{ICS}\ [M_\odot])$')


# ── Plot 3: ICS assembly history grid ──────────────────────────────────────────

_ASSEMBLY_BINS = [
    (1e10, 1e11,  r'$10^{10} < M_\mathrm{vir} < 10^{11}\ M_\odot$'),
    (1e11, 1e12,  r'$10^{11} < M_\mathrm{vir} < 10^{12}\ M_\odot$'),
    (1e12, 1e13,  r'$10^{12} < M_\mathrm{vir} < 10^{13}\ M_\odot$'),
    (1e13, 1e14,  r'$10^{13} < M_\mathrm{vir} < 10^{14}\ M_\odot$'),
    (1e14, 1e15,  r'$10^{14} < M_\mathrm{vir} < 10^{15}\ M_\odot$'),
    (1e15, 1e18,  r'$M_\mathrm{vir} > 10^{15}\ M_\odot$'),
]
_ASSEMBLY_N_MAX   = 7500   # max halos sampled per mass bin
_DISRUPT_COLOR    = '#1B7837'   # green  — disruption channel
_ACCRETE_COLOR    = '#762A83'   # purple — accretion channel


def _lookback_times_gyr(redshifts, omega_m, omega_l, h):
    """Lookback time in Gyr for each entry in *redshifts*."""
    from scipy.integrate import quad
    t_H = 9.778 / h

    def _t_age(z):
        val, _ = quad(
            lambda zp: 1.0 / ((1 + zp) * np.sqrt(omega_m * (1 + zp)**3 + omega_l)),
            z, 1000.0)
        return t_H * val

    t_now = _t_age(0.0)
    return np.array([t_now - _t_age(z) for z in redshifts])


def _read_field_snap(files, snap_key, field, mass_conv, is_mass=False):
    """Read a single field from one snapshot across all files, concatenated."""
    chunks = []
    for fp in files:
        try:
            with h5.File(fp, 'r') as f:
                if snap_key in f and field in f[snap_key]:
                    chunks.append(np.array(f[snap_key][field]))
        except Exception:
            pass
    if not chunks:
        return np.array([])
    arr = np.concatenate(chunks)
    return arr * mass_conv if is_mass else arr


def _ics_histories(files, gal_indices_z0, avail_snaps, mass_conv,
                   last_snap=None):
    """
    Trace ICS_disrupt and ICS_accrete through all snapshots for a set of
    galaxies identified by their z=0 GalaxyIndex values.

    *last_snap* is the snapshot used for z=0 normalisation (defaults to
    max of avail_snaps).

    Returns
    -------
    sorted_snaps : list of int
    disrupt : (n_gal, n_snaps) — ICS_disrupt(snap) / ICS_disrupt(z=0)
    accrete : (n_gal, n_snaps) — ICS_accrete(snap) / ICS_accrete(z=0)
    """
    n_gal        = len(gal_indices_z0)
    sorted_snaps = sorted(avail_snaps)
    n_snaps      = len(sorted_snaps)
    if last_snap is None:
        last_snap = max(avail_snaps)

    # z=0 normalisers — always from the final snapshot
    last_key   = f'Snap_{last_snap}'
    gidx_last  = _read_field_snap(files, last_key, 'GalaxyIndex', mass_conv)
    dis_z0_all = _read_field_snap(files, last_key, 'ICS_disrupt',  mass_conv, True)
    acc_z0_all = _read_field_snap(files, last_key, 'ICS_accrete',  mass_conv, True)

    if gidx_last.size == 0:
        return sorted_snaps, np.full((n_gal, n_snaps), np.nan), np.full((n_gal, n_snaps), np.nan)

    sort_ord = np.argsort(gidx_last)
    s_gids   = gidx_last[sort_ord]
    ins      = np.searchsorted(s_gids, gal_indices_z0)
    ins      = np.clip(ins, 0, len(s_gids) - 1)
    matched  = s_gids[ins] == gal_indices_z0
    row_map  = np.where(matched, sort_ord[ins], -1)

    dis_z0      = np.where(row_map >= 0, dis_z0_all[row_map.clip(0)], 0.0)
    acc_z0      = np.where(row_map >= 0, acc_z0_all[row_map.clip(0)], 0.0)
    dis_z0_safe = np.maximum(dis_z0, 1e-10)
    acc_z0_safe = np.maximum(acc_z0, 1e-10)

    disrupt = np.full((n_gal, n_snaps), np.nan)
    accrete = np.full((n_gal, n_snaps), np.nan)

    # Build lookup: GalaxyIndex -> row in output arrays
    gal_set = set(gal_indices_z0)
    gal_to_row = {gid: i for i, gid in enumerate(gal_indices_z0)}

    for si, snap in enumerate(sorted_snaps):
        snap_key = f'Snap_{snap}'
        gidx_s = _read_field_snap(files, snap_key, 'GalaxyIndex', mass_conv)
        if gidx_s.size == 0:
            continue
        dis_s = _read_field_snap(files, snap_key, 'ICS_disrupt', mass_conv, True)
        acc_s = _read_field_snap(files, snap_key, 'ICS_accrete', mass_conv, True)

        # Only process galaxies we care about
        present = np.isin(gidx_s, list(gal_set))
        for arr_i in np.where(present)[0]:
            gid = gidx_s[arr_i]
            row = gal_to_row.get(gid)
            if row is None:
                continue
            disrupt[row, si] = dis_s[arr_i] / dis_z0_safe[row]
            accrete[row, si] = acc_s[arr_i] / acc_z0_safe[row]

    return sorted_snaps, disrupt, accrete


def plot_3_ics_assembly_grid():
    """
    3x2 grid of ICS assembly history panels, one per halo mass bin.
    Each panel shows the cumulative build-up of the disruption channel
    (green dashed) and accretion channel (purple solid), both normalised
    to their z=0 value, as a function of lookback time.
    Shading = 15th–85th percentile.
    """
    print('\n' + '=' * 70)
    print('Plot 3: ICS assembly history grid')
    print('=' * 70)

    files = _find_files(MODEL_DIR)
    if not files:
        print(f'  No model files found in {MODEL_DIR}')
        return

    hdr = _read_header(MODEL_DIR)
    if hdr is None:
        return

    h_val     = hdr['hubble_h']
    redshifts = hdr['redshifts']
    avail     = hdr['output_snaps']
    mass_conv = hdr['unit_mass_in_g'] / 1.989e33 / h_val
    omega_m   = 0.25
    omega_l   = 0.75

    lb_times = _lookback_times_gyr(redshifts, omega_m, omega_l, h_val)

    # Load z=0 snapshot for mass-bin selection
    last_snap = max(avail)
    d0 = _load_snap(files, last_snap,
                    ['Mvir', 'IntraClusterStars', 'StellarMass',
                     'Type', 'GalaxyIndex'],
                    mass_conv)
    if not d0 or d0['Mvir'].size == 0:
        print('  Could not load z=0 snapshot.')
        return

    Mvir0 = d0['Mvir']
    ICS0  = d0['IntraClusterStars']
    Type0 = d0['Type']
    Gidx0 = d0['GalaxyIndex']

    n_bins = len(_ASSEMBLY_BINS)
    ncols  = 3
    nrows  = (n_bins + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 4.5),
                             sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for bi, (mlo, mhi, label) in enumerate(_ASSEMBLY_BINS):
        ax = axes_flat[bi]

        # Select z=0 centrals in mass bin with ICS > 0
        sel = np.where(
            (Type0 == 0) & (Mvir0 >= mlo) & (Mvir0 < mhi) & (ICS0 > 0)
        )[0]

        if len(sel) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            _format_assembly_ax(ax, bi, ncols, nrows, label)
            continue

        # Subsample if needed
        if len(sel) > _ASSEMBLY_N_MAX:
            rng = np.random.default_rng(42)
            sel = rng.choice(sel, _ASSEMBLY_N_MAX, replace=False)

        gal_ids = Gidx0[sel]
        print(f'  {label}: N={len(sel)}')

        sorted_snaps, disrupt, accrete = _ics_histories(
            files, gal_ids, avail, mass_conv, last_snap=last_snap)

        lb = np.array([lb_times[s] for s in sorted_snaps])

        for arr, color, ls, lbl in [
            (disrupt, _DISRUPT_COLOR, '--', 'Disruption'),
            (accrete, _ACCRETE_COLOR, '-',  'Accretion'),
        ]:
            med = np.nanmedian(arr, axis=0)
            p15 = np.nanpercentile(arr, 15, axis=0)
            p85 = np.nanpercentile(arr, 85, axis=0)
            valid = np.isfinite(med)
            ax.fill_between(lb[valid], p15[valid], p85[valid],
                            color=color, alpha=0.20, lw=0.0)
            ax.plot(lb[valid], med[valid],
                    color=color, ls=ls, lw=2.0, label=lbl)

        _format_assembly_ax(ax, bi, ncols, nrows, label)

    # Hide unused panels
    for bi in range(n_bins, len(axes_flat)):
        axes_flat[bi].set_visible(False)

    # Shared axis labels via fig.text
    fig.text(0.5, 0.01, 'Lookback time [Gyr]',
             ha='center')
    fig.text(0.001, 0.5, r'Fraction of $z\!=\!0$ ICS',
             va='center', rotation='vertical')

    fig.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, 'ICS_assembly_grid' + OUTPUT_FORMAT)
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'\n  Saved: {out}')


def _format_assembly_ax(ax, pi, ncols, nrows, label):
    """Consistent formatting for assembly history panels."""
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([0, 2, 4, 6, 8, 10, 12])

    ax.tick_params(which='both', direction='in', top=True, right=True)
    # ax.text(0.97, 0.97, label,
    #         transform=ax.transAxes, ha='right', va='top')

    # Make panel titles actual sub-plot titles, not legend entries
    ax.set_title(label, fontsize=16, pad=10)

    col = pi % ncols
    row = pi // ncols
    if col != 0:
        ax.tick_params(labelleft=False)
    if row != nrows - 1:
        ax.tick_params(labelbottom=False)


# ── Plot 4 & 5 helpers ─────────────────────────────────────────────────────────

def _binned_stats(x, y, bin_edges, min_count=10):
    """Return (bin_centres, median, p16, p84) for y binned by x."""
    cen = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    med = np.full(len(cen), np.nan)
    p16 = np.full(len(cen), np.nan)
    p84 = np.full(len(cen), np.nan)
    for i in range(len(bin_edges) - 1):
        m = (x >= bin_edges[i]) & (x < bin_edges[i + 1])
        if m.sum() >= min_count:
            med[i] = np.median(y[m])
            p16[i] = np.percentile(y[m], 16)
            p84[i] = np.percentile(y[m], 84)
    return cen, med, p16, p84


def _bcg_halo_obs():
    """
    Load Kravtsov+18 BCG stellar mass vs halo mass (cluster subset only).
    Returns dict with log10(Mvir) and log10(Mstar) arrays, or None.
    """
    path = os.path.join(OBS_DIR, 'SatKinsAndClusters_Kravtsov18.dat')
    if not os.path.exists(path):
        return None
    d = np.loadtxt(path)
    # Cluster rows are those with log10(Mhalo) >= 13.7
    mask = d[:, 0] >= 13.7
    return {'log_mvir': d[mask, 0], 'log_mstar': d[mask, 1]}


def _load_model_snap(model, snap_num, props):
    """Load a snapshot for one model entry from MODELS list."""
    files     = _find_files(model['dir'])
    hdr       = _read_header(model['dir'])
    if not files or hdr is None:
        return None, None
    mass_conv = hdr['unit_mass_in_g'] / 1.989e33 / hdr['hubble_h']
    data      = _load_snap(files, snap_num, props, mass_conv)
    return data, hdr


def _bcg_panel(ax, snap_num, bin_edges, show_obs=True, show_legend=True):
    """
    Plot log10 M_BCG vs log10 M_vir for all three models on *ax*.
    Kravtsov+18 clusters overlaid when show_obs=True.
    """
    props = {'Mvir', 'StellarMass', 'IntraClusterStars', 'Type'}

    for mdl in MODELS:
        data, hdr = _load_model_snap(mdl, snap_num, props)
        if data is None or not data:
            continue
        min_stellar = hdr['part_mass_msun'] * hdr['baryon_frac']
        cen = (
            (data['Type'] == 0) &
            (data['Mvir'] > 0) &
            (data['StellarMass'] >= min_stellar) &
            (data['StellarMass'] / data['Mvir'] >= BCG_MVIR_RATIO_MIN)
        )
        mvir = data['Mvir'][cen]
        sm   = data['StellarMass'][cen]
        if len(mvir) < MIN_N_SNAP:
            continue
        log_mv = np.log10(mvir)
        log_sm = np.log10(sm)
        bc, med, p16, p84 = _binned_stats(log_mv, log_sm, bin_edges)
        ok = ~np.isnan(med)
        ax.fill_between(bc[ok], p16[ok], p84[ok],
                        color=mdl['color'], alpha=0.20, lw=0)
        ax.plot(bc[ok], med[ok], ls=mdl['ls'], color=mdl['color'],
                lw=2.2, label=mdl['label'])

    if show_obs:
        obs = _bcg_halo_obs()
        if obs is not None:
            ax.scatter(obs['log_mvir'], obs['log_mstar'],
                       marker='D', color='#333333', s=30, zorder=5,
                       label='Kravtsov+18 clusters')

    ax.set_ylabel(r'$\log_{10}\, M_{\star,\mathrm{BCG}}\ [\mathrm{M}_{\odot}]$')
    ax.set_xlim(11.5, 15.3)
    ax.set_ylim(9.5, 13.0)
    ax.tick_params(which='both', direction='in', top=True, right=True)
    if show_legend:
        ax.legend(loc='upper left', frameon=False)


def _ratio_panel(ax, snap_num, bin_edges):
    """
    Plot M_ICS/M_BCG vs log10 M_vir for all three models on *ax*.
    """
    props = {'Mvir', 'StellarMass', 'IntraClusterStars', 'Type'}

    for mdl in MODELS:
        data, hdr = _load_model_snap(mdl, snap_num, props)
        if data is None or not data:
            continue
        min_stellar = hdr['part_mass_msun'] * hdr['baryon_frac']
        cen = (
            (data['Type'] == 0) &
            (data['Mvir'] > 0) &
            (data['StellarMass'] >= min_stellar) &
            (data['StellarMass'] / data['Mvir'] >= BCG_MVIR_RATIO_MIN) &
            (data['IntraClusterStars'] > 0)
        )
        mvir  = data['Mvir'][cen]
        sm    = data['StellarMass'][cen]
        ics   = data['IntraClusterStars'][cen]
        if len(mvir) < MIN_N_SNAP:
            continue
        log_mv = np.log10(mvir)
        ratio  = ics / sm
        bc, med, p16, p84 = _binned_stats(log_mv, ratio, bin_edges)
        ok = ~np.isnan(med)
        ax.fill_between(bc[ok], p16[ok], p84[ok],
                        color=mdl['color'], alpha=0.20, lw=0)
        ax.plot(bc[ok], med[ok], ls=mdl['ls'], color=mdl['color'],
                lw=2.2, label=mdl['label'])

    ax.set_ylabel(r'$M_{\mathrm{ICS}}\,/\,M_{\star,\mathrm{BCG}}$')
    ax.set_xlabel(r'$\log_{10}\, M_{\mathrm{vir}}\ [\mathrm{M}_{\odot}]$')
    ax.set_xlim(11.5, 15.3)
    ax.set_ylim(0, None)
    ax.tick_params(which='both', direction='in', top=True, right=True)


# ── Plot 4: BCG–halo mass relation (z=0, three models) ─────────────────────────

def plot_4_bcg_halo_mass():
    """
    2-panel vertical figure at z=0:
      Top:    log M_BCG vs log M_vir for three model variants + Kravtsov+18 clusters
      Bottom: M_ICS/M_BCG vs log M_vir for three model variants
    """
    print('\n' + '=' * 70)
    print('Plot 4: BCG–halo mass relation (z=0, three models)')
    print('=' * 70)

    # Use fiducial model header to find z=0 snapshot
    hdr = _read_header(MODELS[0]['dir'])
    if hdr is None:
        print('  Could not read header.')
        return
    avail  = hdr['output_snaps']
    snap0  = _snap_nearest_z(hdr['redshifts'], 0.0, avail)

    bin_edges = np.arange(11.0, 15.51, 0.25)

    fig, (ax_bcg, ax_rat) = plt.subplots(
        2, 1, figsize=(6, 9),
        sharex=True,
        gridspec_kw={'hspace': 0.05},
    )

    _bcg_panel(ax_bcg, snap0, bin_edges, show_obs=True, show_legend=True)
    _ratio_panel(ax_rat, snap0, bin_edges)

    ax_bcg.tick_params(labelbottom=False)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, f'BCG_halo_mass_z0{OUTPUT_FORMAT}')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')


# ── Plot 5: BCG–halo mass redshift evolution (three models) ────────────────────

def plot_5_bcg_halo_redshift():
    """
    2×N_z grid showing BCG–halo mass (top row) and M_ICS/M_BCG (bottom row)
    at z ≈ 0, 0.5, 1.0, 2.0, for all three model variants.
    """
    print('\n' + '=' * 70)
    print('Plot 5: BCG–halo mass redshift evolution (three models)')
    print('=' * 70)

    hdr = _read_header(MODELS[0]['dir'])
    if hdr is None:
        print('  Could not read header.')
        return
    avail     = hdr['output_snaps']
    redshifts = hdr['redshifts']

    snaps  = [_snap_nearest_z(redshifts, tz, avail) for tz in _BCG_TARGET_Z]
    z_vals = [redshifts[s] for s in snaps]
    n_z    = len(snaps)

    bin_edges = np.arange(11.0, 15.51, 0.25)

    fig, axes = plt.subplots(
        2, n_z, figsize=(4.0 * n_z, 8),
        sharex=True,
        gridspec_kw={'hspace': 0.05, 'wspace': 0.05},
    )

    for ci, (snap, z_act) in enumerate(zip(snaps, z_vals)):
        ax_top = axes[0, ci]
        ax_bot = axes[1, ci]

        _bcg_panel(ax_top, snap, bin_edges,
                   show_obs=(ci == 0), show_legend=(ci == 0))
        _ratio_panel(ax_bot, snap, bin_edges)

        # Remove redundant axis labels except left/bottom edges
        if ci != 0:
            ax_top.tick_params(labelleft=False)
            ax_bot.tick_params(labelleft=False)
            ax_top.set_ylabel('')
            ax_bot.set_ylabel('')
        ax_bot.set_xlabel(r'$\log_{10}\, M_{\mathrm{vir}}\ [\mathrm{M}_{\odot}]$')

        ax_top.tick_params(labelbottom=False)
        ax_top.set_xlabel('')

        ax_top.text(0.97, 0.05, fr'$z={z_act:.1f}$',
                    transform=ax_top.transAxes, ha='right', va='bottom')

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, f'BCG_halo_mass_evolution{OUTPUT_FORMAT}')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')


# ── Main ────────────────────────────────────────────────────────────────────────

_PLOTS = {
    1: plot_1_ics_fraction_vs_redshift,
    2: plot_2_ics_mass_function,
    3: plot_3_ics_assembly_grid,
    # 4: plot_4_bcg_halo_mass,
    # 5: plot_5_bcg_halo_redshift,
}


def main():
    if len(sys.argv) > 1:
        requested = sorted(int(a) for a in sys.argv[1:])
    else:
        requested = sorted(_PLOTS.keys())

    for num in requested:
        if num in _PLOTS:
            _PLOTS[num]()
        else:
            print(f'  Unknown plot number: {num}')


if __name__ == '__main__':
    main()

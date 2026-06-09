#!/usr/bin/env python
"""
ICS paper plots: sampling strategies, demographics, and f_ICS vs redshift.

Three halo sampling strategies (Random, Uniform-in-log-Mvir, Gaussian-in-log-Mvir)
applied to groups/clusters with ICS > 0 and a minimum satellite count.

Produces:
  1. A 3x3 demographic panel (rows = sampling strategy, cols = z / Mvir / mass-gap).
  2. A 3-panel f_ICS vs redshift figure with multiple sample-size median lines and
    optional Spearman statistics below each panel (enable with --stats-boxes).
"""

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr, t as t_dist, norm as norm_dist
import os
import glob
import argparse
import warnings

try:
    from colossus.halo import profile_nfw as _colossus_profile_nfw
    _HAVE_COLOSSUS = True
except Exception:
    _colossus_profile_nfw = None
    _HAVE_COLOSSUS = False

warnings.filterwarnings('ignore')

# If False (default), do not draw the statistics annotation text boxes
# (Spearman/p-values/etc.) on figures. Stats are still printed to the terminal.
SHOW_STATS_BOXES = False

# ── Thresholds ────────────────────────────────────────────────────────────────
N_HALOES = [177, 1000, 7000]       # sample sizes to compare
MIN_MVIR = 10**12.5                     # minimum halo virial mass (Msun)
CLUSTER_MVIR = 1e14                      # Mvir threshold used for 'clusters-only' cuts
Z_MIN = 0.0                            # minimum redshift
Z_MAX = 0.5                            # maximum redshift
MIN_SATELLITES = 2                      # minimum satellites (in addition to the central)
SEED = 42                               # reproducibility

# Gaussian sampling parameters
GAUSS_MU = 13.75                        # peak of Gaussian in log10(Mvir)
GAUSS_SIGMA = 0.25                       # width

# Uniform sampling: number of equal-width log-Mvir bins
N_UNIFORM_BINS = 20

# Distance correlation (dCor) settings
DCOR_MAX_N = 1000                       # max N used for dCor (subsample if larger)

# Solar metallicity (mass fraction)
Z_SUN = 0.02

# Observational aperture for the Gaussian+Selection panels
APERTURE_R_KPC = 300.0
APERTURE_R_MPC = APERTURE_R_KPC / 1000.0

# Disruption-fraction models: (label, subdirectory, colour)
DISRUPT_MODELS = [
    ('70%', 'millennium',            '#0C5DA5'),
    ('50%', 'millennium_disrupt50',  '#FF9500'),
    ('30%', 'millennium_disrupt30',  '#00B945'),
]

# ── Plotting defaults ────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 140,
    'font.size': 12,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'legend.frameon': False,
})


# ── Data I/O (reused patterns from the existing codebase) ────────────────────

def read_simulation_params(filepath):
    params = {}
    with h5.File(filepath, 'r') as f:
        sim = f['Header/Simulation']
        params['Hubble_h'] = float(sim.attrs['hubble_h'])
        params['omega_matter'] = float(sim.attrs['omega_matter'])
        params['omega_lambda'] = float(sim.attrs['omega_lambda'])
        if 'box_size' in sim.attrs:
            params['box_size'] = float(sim.attrs['box_size'])
        elif 'BoxSize' in sim.attrs:
            params['box_size'] = float(sim.attrs['BoxSize'])
        else:
            params['box_size'] = None
        params['PartMass'] = (float(sim.attrs['particle_mass']) * 1.0e10
                              / float(sim.attrs['hubble_h']))
        runtime = f['Header/Runtime']
        params['BaryonFrac'] = float(runtime.attrs.get('BaryonFrac', 0.17))
        params['UnitTime_in_Megayears'] = 978028.5 / params['Hubble_h']
        params['redshifts'] = np.array(f['Header/snapshot_redshifts'])
        snap_groups = [k for k in f.keys() if k.startswith('Snap_')]
        params['available_snapshots'] = sorted(
            int(s.replace('Snap_', '')) for s in snap_groups)
    return params


def _concat_field(file_list, snap_name, field, hubble_h, is_mass=False):
    pieces = []
    for fp in file_list:
        with h5.File(fp, 'r') as f:
            if snap_name not in f or field not in f[snap_name]:
                continue
            data = np.array(f[snap_name][field])
            if data.size > 0:
                if is_mass:
                    data = data * 1.0e10 / hubble_h
                pieces.append(data)
    if not pieces:
        return np.array([])
    return np.concatenate(pieces, axis=0)


def _load_snap_fields(file_list, snap_num, hubble_h, mass_fields,
                      other_fields):
    snap_name = f'Snap_{snap_num}'
    data = {}
    for field in mass_fields:
        data[field] = _concat_field(file_list, snap_name, field, hubble_h,
                                    is_mass=True)
    for field in other_fields:
        data[field] = _concat_field(file_list, snap_name, field, hubble_h,
                                    is_mass=False)
    return data


def _compute_satellite_sums(Type, StellarMass, GalaxyIndex,
                            CentralGalaxyIndex):
    """Satellite count and total stellar mass per central."""
    sorted_idx = np.argsort(GalaxyIndex)
    sorted_gids = GalaxyIndex[sorted_idx]
    sat_mask = Type != 0
    sat_central_gids = CentralGalaxyIndex[sat_mask]
    sat_sm = StellarMass[sat_mask]
    ins = np.searchsorted(sorted_gids, sat_central_gids)
    ins = np.clip(ins, 0, len(sorted_gids) - 1)
    valid = sorted_gids[ins] == sat_central_gids
    c_idx = np.where(valid, sorted_idx[ins], -1)
    ok = c_idx >= 0
    n_sat = np.zeros(len(Type), dtype=int)
    np.add.at(n_sat, c_idx[ok], 1)
    sm_sat = np.zeros(len(Type), dtype=float)
    np.add.at(sm_sat, c_idx[ok], sat_sm[ok])
    return n_sat, sm_sat


def _compute_max_satellite_mass(Type, StellarMass, GalaxyIndex,
                                CentralGalaxyIndex):
    """Stellar mass of the most massive satellite per central."""
    sorted_idx = np.argsort(GalaxyIndex)
    sorted_gids = GalaxyIndex[sorted_idx]
    sat_mask = Type != 0
    sat_central_gids = CentralGalaxyIndex[sat_mask]
    sat_sm = StellarMass[sat_mask]
    ins = np.searchsorted(sorted_gids, sat_central_gids)
    ins = np.clip(ins, 0, len(sorted_gids) - 1)
    valid = sorted_gids[ins] == sat_central_gids
    c_idx = np.where(valid, sorted_idx[ins], -1)
    ok = c_idx >= 0
    max_sat_sm = np.zeros(len(Type), dtype=float)
    np.maximum.at(max_sat_sm, c_idx[ok], sat_sm[ok])
    return max_sat_sm


def _minimum_image_delta(dx, box_size):
    """Apply minimum-image convention for a periodic box."""
    if box_size is None or not np.isfinite(box_size) or box_size <= 0:
        return dx
    return dx - box_size * np.rint(dx / box_size)


def _compute_satellite_sums_within_aperture(Type, StellarMass, GalaxyIndex,
                                            CentralGalaxyIndex,
                                            Posx, Posy, Posz,
                                            aperture_r, box_size):
    """Satellite count/stellar-mass sums within an aperture around centrals.

    Parameters
    ----------
    aperture_r : float
        Aperture radius in the same distance units as Posx/y/z.
    box_size : float or None
        Periodic box size in the same distance units as Posx/y/z.
    """
    sorted_idx = np.argsort(GalaxyIndex)
    sorted_gids = GalaxyIndex[sorted_idx]

    sat_mask = Type != 0
    sat_central_gids = CentralGalaxyIndex[sat_mask]
    sat_sm = StellarMass[sat_mask]
    sx = Posx[sat_mask]
    sy = Posy[sat_mask]
    sz = Posz[sat_mask]

    ins = np.searchsorted(sorted_gids, sat_central_gids)
    ins = np.clip(ins, 0, len(sorted_gids) - 1)
    valid = sorted_gids[ins] == sat_central_gids
    c_idx = np.where(valid, sorted_idx[ins], -1)
    ok = c_idx >= 0
    if not np.any(ok):
        n_sat = np.zeros(len(Type), dtype=int)
        sm_sat = np.zeros(len(Type), dtype=float)
        max_sat_sm = np.zeros(len(Type), dtype=float)
        return n_sat, sm_sat, max_sat_sm

    cx = Posx[c_idx[ok]]
    cy = Posy[c_idx[ok]]
    cz = Posz[c_idx[ok]]

    dx = _minimum_image_delta(sx[ok] - cx, box_size)
    dy = _minimum_image_delta(sy[ok] - cy, box_size)
    dz = _minimum_image_delta(sz[ok] - cz, box_size)
    r = np.sqrt(dx * dx + dy * dy + dz * dz)

    in_ap = r <= aperture_r
    c_in = c_idx[ok][in_ap]
    sm_in = sat_sm[ok][in_ap]

    n_sat = np.zeros(len(Type), dtype=int)
    sm_sat = np.zeros(len(Type), dtype=float)
    max_sat_sm = np.zeros(len(Type), dtype=float)
    if c_in.size:
        np.add.at(n_sat, c_in, 1)
        np.add.at(sm_sat, c_in, sm_in)
        np.maximum.at(max_sat_sm, c_in, sm_in)
    return n_sat, sm_sat, max_sat_sm


def _nfw_enclosed_mass_fraction(r, rvir, c):
    """Return M(<r)/M(<Rvir) for an NFW profile.

    Uses f(x) = ln(1+x) - x/(1+x), with x = c * r / Rvir.
    Clamps to [0, 1] and returns NaN for invalid inputs.
    """
    r, rvir, c = np.broadcast_arrays(
        np.asarray(r, dtype=float),
        np.asarray(rvir, dtype=float),
        np.asarray(c, dtype=float),
    )

    frac = np.full(r.shape, np.nan, dtype=float)
    ok = (
        np.isfinite(r) & np.isfinite(rvir) & np.isfinite(c)
        & (rvir > 0) & (c > 0) & (r >= 0)
    )
    if not np.any(ok):
        return frac

    x = c[ok] * (r[ok] / rvir[ok])

    # Prefer COLOSSUS when available.
    # NFWProfile.M(rhos, rs, x) = rhos * rs^3 * f(x), so in the ratio
    # M(1,1,x) / M(1,1,c) the rhos*rs^3 prefactors cancel, giving f(x)/f(c)
    # = M(<r) / M(<Rvir).  Using rhos=rs=1 avoids needing physical parameters.
    if _HAVE_COLOSSUS and _colossus_profile_nfw is not None:
        denom = _colossus_profile_nfw.NFWProfile.M(1.0, 1.0, c[ok])
        num = _colossus_profile_nfw.NFWProfile.M(1.0, 1.0, x)
        with np.errstate(divide='ignore', invalid='ignore'):
            frac_ok = num / denom
    else:
        def f(xx):
            return np.log1p(xx) - xx / (1.0 + xx)

        denom = f(c[ok])
        num = f(x)
        with np.errstate(divide='ignore', invalid='ignore'):
            frac_ok = num / denom
    frac_ok = np.clip(frac_ok, 0.0, 1.0)
    frac[ok] = frac_ok
    return frac


# ── Cosmology ────────────────────────────────────────────────────────────────

def lookback_time_gyr(z_arr, h=0.73, om=0.25, ol=0.75):
    """Lookback time in Gyr for an array of redshifts (flat LCDM)."""
    H0_inv_Gyr = 9.778 / h
    z_arr = np.atleast_1d(np.asarray(z_arr, dtype=float))
    z_fine = np.linspace(0, max(200.0, float(z_arr.max()) + 1.0), 20000)
    E = np.sqrt(om * (1.0 + z_fine)**3 + ol)
    integrand = 1.0 / ((1.0 + z_fine) * E)
    dz = np.diff(z_fine)
    mid = 0.5 * (integrand[:-1] + integrand[1:])
    cum = np.concatenate([[0.0], np.cumsum(mid * dz)])
    return H0_inv_Gyr * np.interp(z_arr, z_fine, cum)


# ── Build the full candidate pool ────────────────────────────────────────────

def _compute_t50_from_sfh(sfh_disk, sfh_bulge, sel, lb_gyr):
    """
    Lookback time (Gyr) when cumulative SFH first reached 50% of its
    current value.  Returns NaN where SFH data is missing or final mass is 0.
    """
    if sfh_disk.size == 0 or sfh_bulge.size == 0:
        return np.full(len(sel), np.nan)

    sfh_total = sfh_disk[sel] + sfh_bulge[sel]        # (n_sel, n_snaps)
    cumsfh = np.cumsum(sfh_total, axis=1)
    final_mass = cumsfh[:, -1]
    targets = 0.5 * final_mass
    t_50 = np.full(len(sel), np.nan)

    valid = final_mass > 0
    if not valid.any():
        return t_50

    met = cumsfh[valid] >= targets[valid, None]
    first_snap = np.argmax(met, axis=1)

    vi = np.where(valid)[0]
    for k, gi in enumerate(vi):
        s1 = first_snap[k]
        if s1 == 0:
            t_50[gi] = lb_gyr[0]
            continue
        s0 = s1 - 1
        c0, c1 = cumsfh[gi, s0], cumsfh[gi, s1]
        dc = c1 - c0
        frac = (targets[gi] - c0) / dc if dc > 0 else 0.0
        t_50[gi] = lb_gyr[s0] + frac * (lb_gyr[s1] - lb_gyr[s0])
    return t_50


def build_candidate_pool(sim_params, file_list, z_min=None, z_max=None):
    """
    Scan all snapshots in [z_min, z_max] and return arrays for every
    group/cluster central that passes the selection cuts.

    Includes t_asm (ICS assembly lookback, Gyr) for all snapshots and
    t_50 (lookback time when 50% of z=0 stellar mass formed, Gyr) for
    the z=0 snapshot only (NaN for others).
    """
    if z_min is None:
        z_min = Z_MIN
    if z_max is None:
        z_max = Z_MAX
    h_ = sim_params['Hubble_h']
    f_b = sim_params['BaryonFrac']
    box_size = sim_params.get('box_size', None)
    min_stellar = sim_params['PartMass'] * f_b
    redshifts = sim_params['redshifts']
    utm = sim_params['UnitTime_in_Megayears'] / 1000.0  # code time -> Gyr
    valid_snaps = [s for s in sim_params['available_snapshots']
                   if s < len(redshifts) and z_min <= redshifts[s] <= z_max]

    # Pre-compute lookback times for all snapshot indices (for t_50)
    lb_gyr_all = lookback_time_gyr(redshifts,
                                    h=h_,
                                    om=sim_params['omega_matter'],
                                    ol=sim_params['omega_lambda'])

    print(f"Scanning {len(valid_snaps)} snapshots in "
          f"{z_min} <= z <= {z_max} ...")

    pool = {k: [] for k in [
        'mvir', 'ics', 'bcg_sm', 'sat_sm_total', 'max_sat_sm',
        'sat_sm_300', 'max_sat_sm_300', 'ics_300',
        'z', 't_asm', 't_50',
        'metals_sm', 'metals_ics',
        'ics_disrupt', 'ics_accrete',
    ]}

    for snap in valid_snaps:
        z = redshifts[snap]
        d = _load_snap_fields(
            file_list, snap, h_,
            mass_fields=['Mvir', 'IntraClusterStars', 'StellarMass',
                         'MetalsStellarMass', 'MetalsIntraClusterStars',
                         'ICS_disrupt', 'ICS_accrete', 'ICS_sum_mt'],
            other_fields=['Type', 'GalaxyIndex', 'CentralGalaxyIndex',
                          'Posx', 'Posy', 'Posz', 'Rvir', 'Concentration'])
        if d['Mvir'].size == 0:
            continue

        n_sat, sm_sat = _compute_satellite_sums(
            d['Type'], d['StellarMass'],
            d['GalaxyIndex'], d['CentralGalaxyIndex'])
        max_sat = _compute_max_satellite_mass(
            d['Type'], d['StellarMass'],
            d['GalaxyIndex'], d['CentralGalaxyIndex'])

        # Satellite sums within 300 kpc proper. Positions are comoving Mpc/h.
        # Selecting within a proper radius R means comoving threshold
        # R_com = R_proper * (1+z), in the same units as positions.
        aperture_r_com = APERTURE_R_MPC * h_ * (1.0 + z)
        if d.get('Posx', np.array([])).size == d['Type'].size:
            _, sm_sat_300, max_sat_300 = _compute_satellite_sums_within_aperture(
                d['Type'], d['StellarMass'],
                d['GalaxyIndex'], d['CentralGalaxyIndex'],
                d['Posx'], d['Posy'], d['Posz'],
                aperture_r_com, box_size)
        else:
            sm_sat_300 = np.zeros_like(sm_sat)
            max_sat_300 = np.zeros_like(max_sat)

        mask = ((d['Type'] == 0)
                & (d['Mvir'] >= MIN_MVIR)
                & (d['IntraClusterStars'] > 0)
                & (n_sat >= MIN_SATELLITES)
                & (d['StellarMass'] >= min_stellar))
        sel = np.where(mask)[0]
        if len(sel) == 0:
            continue

        pool['mvir'].append(d['Mvir'][sel])
        pool['ics'].append(d['IntraClusterStars'][sel])
        pool['bcg_sm'].append(d['StellarMass'][sel])
        pool['sat_sm_total'].append(sm_sat[sel])
        pool['max_sat_sm'].append(max_sat[sel])
        pool['sat_sm_300'].append(sm_sat_300[sel])
        pool['max_sat_sm_300'].append(max_sat_300[sel])

        # Scale ICS to 300 kpc using an NFW enclosed-mass fraction.
        # Rvir is in physical Mpc/h in SAGE; use a proper 300 kpc radius.
        r_ap_phys = APERTURE_R_MPC * h_
        rvir_all = d.get('Rvir', np.array([]))
        conc_all = d.get('Concentration', np.array([]))
        if rvir_all.size != d['Type'].size:
            rvir_all = np.full_like(d['IntraClusterStars'], np.nan)
        if conc_all.size != d['Type'].size:
            conc_all = np.full_like(d['IntraClusterStars'], np.nan)
        frac_ics_300 = _nfw_enclosed_mass_fraction(
            r_ap_phys,
            rvir_all[sel],
            conc_all[sel],
        )
        pool['ics_300'].append(d['IntraClusterStars'][sel] * frac_ics_300)
        pool['metals_sm'].append(d['MetalsStellarMass'][sel])
        pool['metals_ics'].append(d['MetalsIntraClusterStars'][sel])
        pool['z'].append(np.full(len(sel), z))

        pool['ics_disrupt'].append(d['ICS_disrupt'][sel])
        pool['ics_accrete'].append(d['ICS_accrete'][sel])

        # t_asm: mass-weighted mean ICS assembly lookback (Gyr)
        denom = d['ICS_disrupt'][sel] + d['ICS_accrete'][sel]
        t_asm = np.full(len(sel), np.nan)
        ok = denom > 0
        t_asm[ok] = d['ICS_sum_mt'][sel][ok] / denom[ok] * utm
        pool['t_asm'].append(t_asm)

        # t_50 only at z=0 (50% of *final* stellar mass)
        pool['t_50'].append(np.full(len(sel), np.nan))

    # Concatenate
    for k in pool:
        pool[k] = np.concatenate(pool[k]) if pool[k] else np.array([])

    n = len(pool['mvir'])
    print(f"Total candidate haloes: {n}")
    if n == 0:
        return None

    # t_50: compute only for z=0 haloes using z=0 SFH
    snap_z0 = min(valid_snaps, key=lambda s: redshifts[s])
    z0_mask = pool['z'] == redshifts[snap_z0]
    z0_idx = np.where(z0_mask)[0]
    if len(z0_idx) > 0:
        snap_name = f'Snap_{snap_z0}'
        sfh_disk = _concat_field(file_list, snap_name, 'SFHMassDisk', h_,
                                 is_mass=True)
        sfh_bulge = _concat_field(file_list, snap_name, 'SFHMassBulge', h_,
                                  is_mass=True)
        # z0_idx maps into the pool; we need the original galaxy indices at
        # this snapshot.  Since we appended snapshots in order, we need the
        # selection indices used for snap_z0.  Recompute them here by
        # reloading; the SFH _concat_field already loaded all galaxies at
        # this snapshot, so sel maps directly into those arrays.
        d_z0 = _load_snap_fields(
            file_list, snap_z0, h_,
            mass_fields=['Mvir', 'IntraClusterStars', 'StellarMass'],
            other_fields=['Type', 'GalaxyIndex', 'CentralGalaxyIndex'])
        n_sat_z0, _ = _compute_satellite_sums(
            d_z0['Type'], d_z0['StellarMass'],
            d_z0['GalaxyIndex'], d_z0['CentralGalaxyIndex'])
        sel_z0 = np.where(
            (d_z0['Type'] == 0) & (d_z0['Mvir'] >= MIN_MVIR)
            & (d_z0['IntraClusterStars'] > 0)
            & (n_sat_z0 >= MIN_SATELLITES)
            & (d_z0['StellarMass'] >= min_stellar))[0]
        t_50_z0 = _compute_t50_from_sfh(sfh_disk, sfh_bulge, sel_z0,
                                        lb_gyr_all)
        pool['t_50'][z0_mask] = t_50_z0

    # Derived quantities
    pool['f_baryon'] = pool['ics'] / (f_b * pool['mvir'])
    total_stellar = pool['bcg_sm'] + pool['sat_sm_total'] + pool['ics']
    pool['f_stellar'] = pool['ics'] / total_stellar
    pool['mass_gap'] = np.log10(pool['bcg_sm']) - np.log10(pool['max_sat_sm'])

    pool['f_baryon_300'] = pool['ics_300'] / (f_b * pool['mvir'])
    total_stellar_300 = pool['bcg_sm'] + pool['sat_sm_300'] + pool['ics_300']
    pool['f_stellar_300'] = pool['ics_300'] / total_stellar_300
    with np.errstate(divide='ignore', invalid='ignore'):
        mg300 = np.log10(pool['bcg_sm']) - np.log10(pool['max_sat_sm_300'])
    pool['mass_gap_300'] = np.where(pool['max_sat_sm_300'] > 0, mg300, np.nan)
    pool['log_mvir'] = np.log10(pool['mvir'])
    pool['bcg_Z_solar'] = (pool['metals_sm'] / pool['bcg_sm']) / Z_SUN
    pool['ics_Z_solar'] = (pool['metals_ics'] / pool['ics']) / Z_SUN
    with np.errstate(divide='ignore', invalid='ignore'):
        pool['log_ics_disrupt'] = np.where(pool['ics_disrupt'] > 0,
                                           np.log10(pool['ics_disrupt']), np.nan)
        pool['log_ics_accrete'] = np.where(pool['ics_accrete'] > 0,
                                           np.log10(pool['ics_accrete']), np.nan)

    # Lookback time (Gyr) from redshift
    pool['lookback_time'] = lookback_time_gyr(
        pool['z'], h=h_, om=sim_params['omega_matter'],
        ol=sim_params['omega_lambda'])

    # BCG fractions
    pool['f_bcg_baryon'] = pool['bcg_sm'] / (f_b * pool['mvir'])
    pool['f_bcg_stellar'] = pool['bcg_sm'] / total_stellar

    # BCG + ICS combined fractions
    bcg_plus_ics = pool['bcg_sm'] + pool['ics']
    pool['f_bcg_ics_baryon'] = bcg_plus_ics / (f_b * pool['mvir'])
    pool['f_bcg_ics_stellar'] = bcg_plus_ics / total_stellar

    n_asm = np.sum(np.isfinite(pool['t_asm']))
    n_t50 = np.sum(np.isfinite(pool['t_50']))
    print(f"  Valid t_asm: {n_asm}, valid t_50: {n_t50}")

    return pool


# ── Sampling strategies ──────────────────────────────────────────────────────

def sample_random(pool, n, seed=SEED):
    """Uniform random draw (no weighting)."""
    rng = np.random.default_rng(seed)
    n_total = len(pool['mvir'])
    n_sel = min(n, n_total)
    idx = rng.choice(n_total, size=n_sel, replace=False)
    return idx


def sample_uniform(pool, n, seed=SEED):
    """
    Uniform sampling in log10(Mvir): divide into N_UNIFORM_BINS equal-width
    bins and draw the same number from each bin.
    """
    rng = np.random.default_rng(seed)
    log_m = np.log10(pool['mvir'])
    bin_edges = np.linspace(log_m.min(), log_m.max(), N_UNIFORM_BINS + 1)
    per_bin = max(1, n // N_UNIFORM_BINS)

    selected = []
    for i in range(N_UNIFORM_BINS):
        in_bin = np.where((log_m >= bin_edges[i])
                          & (log_m < bin_edges[i + 1]))[0]
        if len(in_bin) == 0:
            continue
        draw = min(per_bin, len(in_bin))
        selected.append(rng.choice(in_bin, size=draw, replace=False))

    if not selected:
        return np.array([], dtype=int)
    return np.concatenate(selected)


def sample_gaussian(pool, n, seed=SEED):
    """
    Gaussian-weighted draw in log10(Mvir), peaked at GAUSS_MU with width
    GAUSS_SIGMA.
    """
    rng = np.random.default_rng(seed)
    log_m = np.log10(pool['mvir'])
    n_total = len(log_m)
    if n_total == 0:
        return np.array([], dtype=int)

    weights = np.exp(-0.5 * ((log_m - GAUSS_MU) / GAUSS_SIGMA) ** 2)
    wsum = np.sum(weights)
    if (not np.isfinite(wsum)) or wsum <= 0:
        probs = np.full(n_total, 1.0 / n_total)
    else:
        probs = weights / wsum

    n_sel = min(n, n_total)
    # Important: never sample with replacement. If a cut-selected pool has fewer
    # than n entries, we just return all available entries (n_sel == n_total).
    return rng.choice(n_total, size=n_sel, replace=False, p=probs)


SAMPLERS = {
    'Random': sample_random,
    'Uniform': sample_uniform,
    'Gaussian': sample_gaussian,
}

SAMPLER_COLOURS = {
    'Random': 'steelblue',
    'Uniform': 'seagreen',
    'Gaussian': 'firebrick',
}


# ── Plot 1: 3x3 demographic histograms ──────────────────────────────────────

def plot_demographics(pool, output_dir, n_sample=None):
    """
    12-panel figure. Rows = Random / Uniform / Gaussian / Selection (4th-panel cuts).
    Cols = Redshift / log10(Mvir) / Mass-gap.

    The 4th row matches the additional selection criteria used in the 4th panels
    of other figures:
      - Redshift column: clusters-only (Mvir >= 1e14)
      - Halo mass column: small mass-gap (mass_gap < 1.2)
      - Mass-gap column: massive + higher-z (Mvir >= 1e13.5 and z > 0.16)
    """
    if n_sample is None:
        n_sample = min(N_HALOES)

    fig, axes = plt.subplots(4, 3, figsize=(14, 14))

    for i, (name, sampler) in enumerate(SAMPLERS.items()):
        idx = sampler(pool, n_sample)
        n_actual = len(idx)
        colour = SAMPLER_COLOURS[name]

        z_sel = pool['z'][idx]
        logm_sel = np.log10(pool['mvir'][idx])
        gap_sel = pool['mass_gap'][idx]

        # Column 0: Redshift
        ax = axes[i, 0]
        ax.hist(z_sel, bins=np.linspace(Z_MIN, Z_MAX, 16),
                color=colour, edgecolor='black', alpha=0.8)
        ax.set_xlabel('Redshift $z$')
        ax.set_ylabel('Count')
        ax.set_xlim(Z_MIN - 0.02, Z_MAX + 0.02)

        # Column 1: Halo mass
        ax = axes[i, 1]
        ax.hist(logm_sel, bins=20,
                color=colour, edgecolor='black', alpha=0.8)
        ax.set_xlabel(r'$\log_{10}\,M_{\mathrm{vir}}\ [\mathrm{M}_\odot]$')
        ax.set_ylabel('Count')

        # Column 2: Mass-gap
        ax = axes[i, 2]
        finite = gap_sel[np.isfinite(gap_sel)]
        ax.hist(finite, bins=20,
                color=colour, edgecolor='black', alpha=0.8)
        ax.set_xlabel(r'$\Delta M_\star = \log_{10}(M_{\mathrm{BCG}}) '
                      r'- \log_{10}(M_{\mathrm{sat,\,max}})$')
        ax.set_ylabel('Count')

        # Row label
        axes[i, 0].annotate(
            f'{name}\n(N = {n_actual})',
            xy=(0, 0.5), xycoords='axes fraction',
            xytext=(-55, 0), textcoords='offset points',
            ha='right', va='center', fontsize=12, fontweight='bold',
            rotation=90)

    # 4th row: selection-criteria subsamples used in other figures' 4th panels.
    # Use Gaussian sampling on each subsample for consistency with those panels.
    i = 3
    colour = '#FF9500'  # orange for selection row

    # Column 0 selection: clusters only (for f_ICS vs redshift 4th panel)
    pool_clusters = _filter_pool(pool, pool['mvir'] >= CLUSTER_MVIR)
    idx = sample_gaussian(pool_clusters, n_sample)
    n_actual = len(idx)
    z_sel = pool_clusters['z'][idx]
    ax = axes[i, 0]
    ax.hist(z_sel, bins=np.linspace(Z_MIN, Z_MAX, 16),
            color=colour, edgecolor='black', alpha=0.8)
    ax.set_xlabel('Redshift $z$')
    ax.set_ylabel('Count')
    ax.set_xlim(Z_MIN - 0.02, Z_MAX + 0.02)
    ax.text(0.02, 0.98, r'$M_{\rm vir} \geq 10^{14}$',
            transform=ax.transAxes, ha='left', va='top', fontsize=10)

    # Column 1 selection: small mass-gap (for f_ICS vs Mvir 4th panel)
    gap_key = 'mass_gap_300' if 'mass_gap_300' in pool else 'mass_gap'
    pool_smallgap = _filter_pool(pool, pool[gap_key] < 1.2)
    idx = sample_gaussian(pool_smallgap, n_sample)
    n_actual_m = len(idx)
    logm_sel = np.log10(pool_smallgap['mvir'][idx])
    ax = axes[i, 1]
    ax.hist(logm_sel, bins=20,
            color=colour, edgecolor='black', alpha=0.8)
    ax.set_xlabel(r'$\log_{10}\,M_{\mathrm{vir}}\ [\mathrm{M}_\odot]$')
    ax.set_ylabel('Count')
    cut_txt = (r'$\Delta M_{\star,300} < 1.2$' if gap_key == 'mass_gap_300'
               else r'$\Delta M_\star < 1.2$')
    ax.text(0.02, 0.98, cut_txt,
            transform=ax.transAxes, ha='left', va='top', fontsize=10)

    # Column 2 selection: massive + higher-z (for f_ICS vs mass-gap 4th panel)
    pool_massive_hiz = _filter_pool(
        pool,
        (pool['mvir'] >= 10**13.5) & (pool['z'] > 0.16) & (pool['z'] <= Z_MAX),
    )
    idx = sample_gaussian(pool_massive_hiz, n_sample)
    n_actual_g = len(idx)
    gap_sel = pool_massive_hiz['mass_gap'][idx]
    ax = axes[i, 2]
    finite = gap_sel[np.isfinite(gap_sel)]
    ax.hist(finite, bins=20,
            color=colour, edgecolor='black', alpha=0.8)
    ax.set_xlabel(r'$\Delta M_\star = \log_{10}(M_{\mathrm{BCG}}) '
                  r'- \log_{10}(M_{\mathrm{sat,\,max}})$')
    ax.set_ylabel('Count')
    ax.text(0.02, 0.98, r'$M_{\rm vir} \geq 10^{13.5},\ z>0.16$',
            transform=ax.transAxes, ha='left', va='top', fontsize=10)

    # Row label
    axes[i, 0].annotate(
        'Selection\n'
        f'$N_z={n_actual}$',
        xy=(0, 0.5), xycoords='axes fraction',
        xytext=(-55, 0), textcoords='offset points',
        ha='right', va='center', fontsize=12, fontweight='bold',
        rotation=90)

    # Column titles
    for j, title in enumerate(['Redshift', 'Halo Mass', 'Mass-gap']):
        axes[0, j].set_title(title, fontsize=13, fontweight='bold')

    plt.tight_layout(rect=[0.06, 0, 1, 0.96])
    # fig.suptitle(f'Halo Demographics by Sampling Strategy '
    #              f'(N = {n_sample})', fontsize=15, fontweight='bold')

    outfile = os.path.join(output_dir, 'ICS_demographics_3x3.pdf')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {outfile}')


# ── Helpers ───────────────────────────────────────────────────────────────────

def partial_spearman(x, y, covar):
    """
    Partial Spearman correlation between x and y, controlling for covar.
    Returns (r_partial, p_value).
    """
    r_xy, _ = spearmanr(x, y)
    r_xc, _ = spearmanr(x, covar)
    r_yc, _ = spearmanr(y, covar)

    den = np.sqrt((1 - r_xc**2) * (1 - r_yc**2))
    if den == 0:
        return 0.0, 1.0

    r_partial = np.clip((r_xy - r_xc * r_yc) / den, -1.0, 1.0)

    n = len(x)
    df = n - 3
    if abs(r_partial) == 1.0 or df <= 0:
        return r_partial, 0.0

    t_stat = r_partial * np.sqrt(df / (1 - r_partial**2))
    p_val = 2 * t_dist.sf(np.abs(t_stat), df)
    return r_partial, p_val


def corr_ci_fisher(r, n, alpha=0.05):
    """Approximate CI for a correlation coefficient using Fisher z.

    Works well for large n and keeps runtime small compared to bootstrap.
    Returns (lo, hi) or (nan, nan) if undefined.
    """
    if n is None or n < 4 or (not np.isfinite(r)):
        return np.nan, np.nan

    r = float(np.clip(r, -0.999999, 0.999999))
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    zcrit = norm_dist.ppf(1.0 - alpha / 2.0)
    lo = np.tanh(z - zcrit * se)
    hi = np.tanh(z + zcrit * se)
    return lo, hi


def format_effect_ci(r, n, alpha=0.05):
    lo, hi = corr_ci_fisher(r, n, alpha=alpha)
    if np.isfinite(lo) and np.isfinite(hi):
        return f'CI=[{lo:+.3f},{hi:+.3f}]'
    return 'CI=n/a'


def distance_correlation(x, y, *, seed=SEED, max_n=DCOR_MAX_N):
    """Distance correlation for 1D arrays.

    Uses the biased distance covariance estimator with double-centering.
    For speed/memory, if n > max_n, computes dCor on a random subset.

    Returns
    -------
    (dcor, n_used)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    n = x.size
    if n < 4:
        return np.nan, int(n)

    if max_n is not None and n > max_n:
        rng = np.random.default_rng(seed)
        ii = rng.choice(n, size=max_n, replace=False)
        x = x[ii]
        y = y[ii]
        n = x.size

    # Prefer the installed `dcor` package (faster + well-tested), but keep a
    # dependency-free fallback.
    try:
        import dcor as dcor_pkg  # type: ignore

        d = dcor_pkg.distance_correlation(x.reshape(-1, 1), y.reshape(-1, 1))
        return float(np.clip(d, 0.0, 1.0)), int(n)
    except Exception:
        # Pairwise absolute distances (n x n)
        a = np.abs(x[:, None] - x[None, :]).astype(float, copy=False)
        b = np.abs(y[:, None] - y[None, :]).astype(float, copy=False)

        # Double-centering
        a_row = a.mean(axis=1, keepdims=True)
        a_col = a.mean(axis=0, keepdims=True)
        a_mean = a.mean()
        A = a - a_row - a_col + a_mean

        b_row = b.mean(axis=1, keepdims=True)
        b_col = b.mean(axis=0, keepdims=True)
        b_mean = b.mean()
        B = b - b_row - b_col + b_mean

        dcov2 = np.mean(A * B)
        dvarx2 = np.mean(A * A)
        dvary2 = np.mean(B * B)

        if dcov2 <= 0 or dvarx2 <= 0 or dvary2 <= 0:
            return 0.0, int(n)

        dcor = np.sqrt(dcov2) / ((dvarx2 * dvary2) ** 0.25)
        return float(np.clip(dcor, 0.0, 1.0)), int(n)


def format_dcor(x, y, *, seed=SEED, max_n=DCOR_MAX_N):
    dcor, n_used = distance_correlation(x, y, seed=seed, max_n=max_n)
    if not np.isfinite(dcor):
        return 'dCor=n/a'
    return f'dCor={dcor:.3f}'


def _strip_latex(text):
    """Remove common LaTeX markup for clean terminal output."""
    import re
    s = text
    s = re.sub(r'\$([^$]*)\$', r'\1', s)      # strip $ delimiters
    s = s.replace(r'\rho_s', 'rho_s')
    s = s.replace(r'\mathrm{', '').replace(r'\,', ' ')
    s = s.replace(r'\log_{10}', 'log10')
    s = s.replace(r'\star', '*').replace(r'\odot', 'sun')
    s = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', s)  # \cmd{...} -> ...
    s = re.sub(r'[\\{}^_]', '', s)             # leftover backslashes/braces
    return s


def _print_plot_header(title):
    """Print a clear separator for a new plot."""
    width = max(len(title) + 4, 60)
    print(f'\n{"─" * width}')
    print(f'  {title}')
    print(f'{"─" * width}')


def _print_stats(panel_name, text):
    """Print stat text with proper indentation under a panel heading."""
    plain = _strip_latex(text)
    lines = plain.split('\n')
    print(f'  [{panel_name}]')
    for line in lines:
        print(f'    {line}')


# ── Plot 2: f_ICS vs Redshift  ──────────────────────────────────────────────

def _filter_pool(pool, mask):
    """Return a new pool dict containing only entries where mask is True."""
    sub = {}
    for k, v in pool.items():
        sub[k] = v[mask]
    return sub


def _filter_pool_finite_positive(pool, *, finite_keys, positive_keys=()):
    """Filter a pool to rows with finite values (and optionally > 0) in given keys.

    Parameters
    ----------
    finite_keys : iterable[str]
        Keys that must be finite.
    positive_keys : iterable[str]
        Keys that must be strictly > 0 (and finite).
    """
    if pool is None or len(pool) == 0:
        return pool

    # Determine row count from Mvir if present, else first array-like.
    if 'mvir' in pool:
        n = len(pool['mvir'])
    else:
        first = next(iter(pool.values()))
        n = len(first)

    mask = np.ones(n, dtype=bool)
    for k in finite_keys:
        if k not in pool:
            mask &= False
            continue
        mask &= np.isfinite(pool[k])
    for k in positive_keys:
        if k not in pool:
            mask &= False
            continue
        mask &= np.isfinite(pool[k]) & (pool[k] > 0)
    return _filter_pool(pool, mask)


def _median_in_bins(z, f, n_bins=12):
    """Return bin centres and medians for z-binned data."""
    edges = np.linspace(z.min(), z.max(), n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    medians = np.full(n_bins, np.nan)
    for k in range(n_bins):
        in_bin = (z >= edges[k]) & (z < edges[k + 1])
        if in_bin.sum() >= 3:
            medians[k] = np.median(f[in_bin])
    valid = np.isfinite(medians)
    return centres[valid], medians[valid]


def _build_z0_gc_tasm_t50_pool(sim_params, file_list, *, nsat_min=1):
    """Build a z~0 groups/clusters-only (centrals with >=nsat_min satellites) pool.

    Returns a dict with keys: 'mvir', 't_50', 't_asm'.
    """
    h_ = sim_params['Hubble_h']
    redshifts = sim_params['redshifts']
    utm = sim_params['UnitTime_in_Megayears'] / 1000.0  # code time -> Gyr
    min_stellar = sim_params['PartMass'] * sim_params.get('BaryonFrac', 0.17)

    snap_z0 = min(sim_params['available_snapshots'], key=lambda s: redshifts[s])
    snap_name = f'Snap_{snap_z0}'

    lb_gyr_all = lookback_time_gyr(
        redshifts,
        h=h_,
        om=sim_params['omega_matter'],
        ol=sim_params['omega_lambda'],
    )

    d = _load_snap_fields(
        file_list, snap_z0, h_,
        mass_fields=[
            'Mvir', 'IntraClusterStars', 'StellarMass',
            'ICS_disrupt', 'ICS_accrete', 'ICS_sum_mt',
        ],
        other_fields=['Type', 'GalaxyIndex', 'CentralGalaxyIndex'],
    )
    if d.get('Mvir', np.array([])).size == 0:
        return {'mvir': np.array([]), 't_50': np.array([]), 't_asm': np.array([])}

    sfh_disk = _concat_field(file_list, snap_name, 'SFHMassDisk', h_, is_mass=True)
    sfh_bulge = _concat_field(file_list, snap_name, 'SFHMassBulge', h_, is_mass=True)
    if sfh_disk.size == 0 or sfh_bulge.size == 0:
        return {'mvir': np.array([]), 't_50': np.array([]), 't_asm': np.array([])}

    n_sat, _ = _compute_satellite_sums(
        d['Type'], d['StellarMass'], d['GalaxyIndex'], d['CentralGalaxyIndex']
    )

    denom = d['ICS_disrupt'] + d['ICS_accrete']
    t_asm_all = np.full_like(denom, np.nan, dtype=float)
    ok = denom > 0
    t_asm_all[ok] = d['ICS_sum_mt'][ok] / denom[ok] * utm

    base_mask = (
        (d['Type'] == 0)
        & (d['Mvir'] > 0)
        & (d['StellarMass'] >= min_stellar)
        & (d['IntraClusterStars'] > 0)
        & np.isfinite(t_asm_all)
        & (t_asm_all > 0)
    )
    sel = np.where(base_mask)[0]
    if sel.size == 0:
        return {'mvir': np.array([]), 't_50': np.array([]), 't_asm': np.array([])}

    t50_sel = _compute_t50_from_sfh(sfh_disk, sfh_bulge, sel, lb_gyr_all)
    tasm_sel = t_asm_all[sel]
    nsat_sel = n_sat[sel]
    mvir_sel = d['Mvir'][sel]

    finite = np.isfinite(t50_sel) & np.isfinite(tasm_sel)
    t50_c = t50_sel[finite]
    tasm_c = tasm_sel[finite]
    nsat_c = nsat_sel[finite]
    mvir_c = mvir_sel[finite]

    gc = nsat_c >= nsat_min
    return {
        'mvir': mvir_c[gc],
        't_50': t50_c[gc],
        't_asm': tasm_c[gc],
    }


def _draw_fics_panel(ax, ax_stat, src_pool, sampler, x_key, frac_labels,
                     frac_colours, extra_spearman_fn=None, *,
                     stat_y_step_factor=1.0,
                     baryon_key='f_baryon',
                     stellar_key='f_stellar',
                     panel_name=None):
    """
    Shared logic for one panel of a f_ICS-vs-X plot.

    Parameters
    ----------
    x_key : str   Pool key for the x-axis variable (e.g. 'z', or a lambda).
    extra_spearman_fn : callable(idx) -> list[str]
        Optional extra stat lines per sample size (e.g. partial Spearman).
    """
    n_sizes = len(N_HALOES)
    stat_boxes = {j: [] for j in range(n_sizes)}
    n_draw = {}
    n_used_counts = {j: {} for j in range(n_sizes)}

    for j, n_target in enumerate(N_HALOES):
        idx = sampler(src_pool, n_target, seed=SEED + j)
        if len(idx) == 0:
            continue

        n_draw[j] = int(len(idx))

        x_sel = src_pool[x_key][idx]
        lw = 1.0 + 1.5 * (j / max(n_sizes - 1, 1))
        alpha = 0.4 + 0.6 * (j / max(n_sizes - 1, 1))

        for fkey in [baryon_key, stellar_key]:
            f_sel = src_pool[fkey][idx]
            finite = np.isfinite(x_sel) & np.isfinite(f_sel)
            n_used_local = int(np.sum(finite))
            n_used_counts[j][fkey] = n_used_local
            if n_used_local < 5:
                continue

            xc, med = _median_in_bins(x_sel[finite], f_sel[finite])
            if len(xc) == 0:
                continue
            label = (f'N={n_used_local} {frac_labels[fkey]}'
                     if j == n_sizes - 1 else
                     f'N={n_used_local}' if fkey == baryon_key else None)
            ax.plot(xc, med,
                    color=frac_colours[fkey],
                    lw=lw, alpha=alpha,
                    linestyle='-' if fkey == baryon_key else '--',
                    label=label)
            if fkey == baryon_key:
                nice = 'baryon(300)' if baryon_key != 'f_baryon' else 'baryon'
            else:
                nice = 'stellar(300)' if stellar_key != 'f_stellar' else 'stellar'

            rho, pval = spearmanr(x_sel[finite], f_sel[finite])
            ci_txt = format_effect_ci(rho, n_used_local, alpha=0.05)
            dcor_txt = format_dcor(x_sel[finite], f_sel[finite], seed=SEED + j)
            stat_boxes[j].append(
                f'{nice}: '
                r'$\rho_s$' f' = {rho:+.3f}, {ci_txt}, '
                f'p = {pval:.2e}, {dcor_txt}')

        if extra_spearman_fn is not None:
            extra = extra_spearman_fn(idx)
            if extra:
                stat_boxes[j].extend(extra)

    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8, loc='best')

    y_top = 0.95
    y_step = stat_y_step_factor / max(n_sizes, 1)
    for j in range(n_sizes):
        if not stat_boxes[j]:
            continue
        draw = n_draw.get(j, 0)
        nb = n_used_counts[j].get(baryon_key, 0)
        ns = n_used_counts[j].get(stellar_key, 0)
        if nb == ns:
            header = f'N_used={nb} (draw={draw})'
        else:
            header = f'N_used(b)={nb}, N_used(*)={ns} (draw={draw})'

        text = header + '\n' + '\n'.join(stat_boxes[j])
        if SHOW_STATS_BOXES:
            ax_stat.text(0.5, y_top - j * y_step, text,
                         transform=ax_stat.transAxes,
                         fontsize=7, ha='center', va='top',
                         bbox=dict(boxstyle='round,pad=0.3',
                                   facecolor='white', edgecolor='grey',
                                   alpha=0.9))

        # Terminal output
        pname = panel_name or getattr(sampler, '__name__', 'panel')
        _print_stats(pname, text)


def _draw_disruption_panel(ax, ax_stat, all_pools, x_key, y_configs):
    """
    Comparison panel overlaying multiple disruption models (Gaussian, max N).

    Parameters
    ----------
    all_pools : list of (label, pool, colour)
    x_key : str  Pool key for the x-axis variable.
    y_configs : list of (pool_key, linestyle, nice_name)
        e.g. [('f_baryon', '-', '$f_b$'), ('f_stellar', '--', '$f_\\star$')]
    """
    n_target = max(N_HALOES)
    # Use the same seed as the max-N Gaussian curves elsewhere (SEED + last j)
    # so the default-model line matches the corresponding Gaussian panel.
    seed = SEED + (len(N_HALOES) - 1)
    stat_lines = []

    for label, pool, colour in all_pools:
        idx = sample_gaussian(pool, n_target, seed=seed)
        if len(idx) == 0:
            continue
        x_sel = pool[x_key][idx]

        for y_key, ls, nice in y_configs:
            y_sel = pool[y_key][idx]
            finite = np.isfinite(x_sel) & np.isfinite(y_sel)
            xf, yf = x_sel[finite], y_sel[finite]
            if len(xf) < 5:
                continue

            xc, med = _median_in_bins(xf, yf)
            if len(xc) > 0:
                lbl = f'{label} {nice}' if len(y_configs) > 1 else label
                ax.plot(xc, med, color=colour, lw=2.0, linestyle=ls,
                        label=lbl)

            rho, pval = spearmanr(xf, yf)
            ci_txt = format_effect_ci(rho, len(xf), alpha=0.05)
            dcor_txt = format_dcor(xf, yf, seed=seed)
            stat_lines.append(
                f'{label} {nice}: '
                r'$\rho_s$' f' = {rho:+.3f}, {ci_txt}, p = {pval:.2e}, {dcor_txt}')

    ax.legend(fontsize=7, loc='best')

    if stat_lines:
        text = '\n'.join(stat_lines)
        if SHOW_STATS_BOXES:
            ax_stat.text(0.5, 0.95, text, transform=ax_stat.transAxes,
                         fontsize=7, ha='center', va='top',
                         bbox=dict(boxstyle='round,pad=0.3',
                                   facecolor='white', edgecolor='grey',
                                   alpha=0.9))
        # Terminal output
        _print_stats('Disruption Comparison', text)


# ── f_ICS y-configs (reused across several plots) ────────────────────────────
_FICS_Y_CONFIGS = [
    ('f_baryon',  '-',  r'$f_b$'),
    ('f_stellar', '--', r'$f_\star$'),
]

_FICS_Y_CONFIGS_300 = [
    ('f_baryon_300',  '-',  r'$f_{b,300}$'),
    ('f_stellar_300', '--', r'$f_{\star,300}$'),
]


def plot_fics_vs_redshift(pool, output_dir, all_pools=None, *, aperture_all=False):
    """
    4 (+1 comparison) panels: Random / Uniform / Gaussian / Gaussian (Mvir > 10^14)
    / Disruption Comparison.  f_ICS vs redshift.
    """
    _print_plot_header('f_ICS vs Redshift')
    frac_labels = {
        'f_baryon': r'$M_\mathrm{ICS}/(f_b\,M_\mathrm{vir})$',
        'f_stellar': r'$M_\mathrm{ICS}/M_{\star,\mathrm{tot}}$',
        'f_baryon_300': r'$M_{\mathrm{ICS},300}/(f_b\,M_\mathrm{vir})$',
        'f_stellar_300': r'$M_{\mathrm{ICS},300}/M_{\star,\mathrm{tot},300}$',
    }
    frac_colours = {
        'f_baryon': '#0C5DA5',
        'f_stellar': '#FF2C00',
        'f_baryon_300': '#0C5DA5',
        'f_stellar_300': '#FF2C00',
    }

    # 4th panel: Gaussian on clusters only
    cluster_mask = pool['mvir'] >= CLUSTER_MVIR
    pool_clusters = _filter_pool(pool, cluster_mask)

    panels = list(SAMPLERS.items()) + [
        (r'Gaussian ($M_{\rm vir}>10^{14}$)', sample_gaussian)]
    pools = [pool] * len(SAMPLERS) + [pool_clusters]
    has_comp = all_pools is not None and len(all_pools) > 1

    fig_h = 16 if SHOW_STATS_BOXES else 10
    fig = plt.figure(figsize=(18, fig_h))
    if SHOW_STATS_BOXES:
        outer_gs = gridspec.GridSpec(4, 1, height_ratios=[2, 1, 2, 1], hspace=0.35,
                                    figure=fig)
        ax_gs_r0 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[0],
                                                    wspace=0.30)
        stat_gs_r0 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[1],
                                                      wspace=0.30)
        ax_gs_r1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[2],
                                                    wspace=0.30)
        stat_gs_r1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[3],
                                                      wspace=0.30)
        ax_specs = [ax_gs_r0[0], ax_gs_r0[1], ax_gs_r0[2],
                    ax_gs_r1[0], ax_gs_r1[1]]
        stat_specs = [stat_gs_r0[0], stat_gs_r0[1], stat_gs_r0[2],
                      stat_gs_r1[0], stat_gs_r1[1]]
    else:
        outer_gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.35,
                                    figure=fig)
        ax_gs_r0 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[0],
                                                    wspace=0.30)
        ax_gs_r1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[1],
                                                    wspace=0.30)
        stat_gs_r1 = None
        ax_specs = [ax_gs_r0[0], ax_gs_r0[1], ax_gs_r0[2],
                    ax_gs_r1[0], ax_gs_r1[1]]
        stat_specs = [None] * len(ax_specs)
    all_handles, all_labels = [], []

    for i, ((name, sampler), src) in enumerate(zip(panels, pools)):
        ax = fig.add_subplot(ax_specs[i])
        ax_stat = fig.add_subplot(stat_specs[i]) if SHOW_STATS_BOXES else None
        if ax_stat is not None:
            ax_stat.axis('off')

        use_aper = aperture_all or (i == len(SAMPLERS))
        bkey = 'f_baryon_300' if use_aper else 'f_baryon'
        skey = 'f_stellar_300' if use_aper else 'f_stellar'
        _draw_fics_panel(ax, ax_stat, src, sampler, 'z',
                         frac_labels, frac_colours,
                         baryon_key=bkey, stellar_key=skey,
                         panel_name=name)

        if i == 0:
            h, l = ax.get_legend_handles_labels()
            all_handles.extend(h)
            all_labels.extend(l)
        leg = ax.get_legend()
        if leg:
            leg.remove()

        ax.set_xlabel('Redshift $z$')
        if i == 0:
            ax.set_ylabel(r'$f_\mathrm{ICS}$')
        ax.set_title(name, fontsize=13, fontweight='bold')
        ax.set_xlim(Z_MIN - 0.02, Z_MAX + 0.02)

    if has_comp:
        ic = len(panels)
        ax = fig.add_subplot(ax_specs[ic])
        ax_stat = fig.add_subplot(stat_specs[ic]) if SHOW_STATS_BOXES else None
        if ax_stat is not None:
            ax_stat.axis('off')
        # Match the "Gaussian + selection" population (clusters-only) used in
        # the 4th panel so the disruption comparison is like-for-like.
        all_pools_sel = []
        for label, p, colour in all_pools:
            p_sel = _filter_pool(p, p['mvir'] >= CLUSTER_MVIR)
            all_pools_sel.append((label, p_sel, colour))
        _draw_disruption_panel(ax, ax_stat, all_pools_sel, 'z', _FICS_Y_CONFIGS_300)
        h, l = ax.get_legend_handles_labels()
        all_handles.extend(h)
        all_labels.extend(l)
        leg = ax.get_legend()
        if leg:
            leg.remove()
        ax.set_xlabel('Redshift $z$')
        ax.set_title('Disruption Comparison', fontsize=13, fontweight='bold')
        ax.set_xlim(Z_MIN - 0.02, Z_MAX + 0.02)

    ax_leg = fig.add_subplot(ax_gs_r1[2])
    ax_leg.axis('off')
    if all_handles:
        ax_leg.legend(all_handles, all_labels, loc='center', fontsize=10,
                      frameon=True, fancybox=True, edgecolor='grey')
    if SHOW_STATS_BOXES and stat_gs_r1 is not None:
        stat_leg = fig.add_subplot(stat_gs_r1[2])
        stat_leg.axis('off')

    # fig.suptitle(r'$f_\mathrm{ICS}$ vs Redshift by Sampling Strategy',
    #              fontsize=15, fontweight='bold', y=0.98)

    outfile = os.path.join(output_dir, 'ICS_fics_vs_redshift.pdf')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {outfile}')


# ── Plot 3: f_ICS vs Mass-gap ────────────────────────────────────────────────

def plot_fics_vs_massgap(pool, output_dir, all_pools=None, *, aperture_all=False):
    """
    4 (+1 comparison) panels: Random / Uniform / Gaussian /
    Gaussian (Mvir>10^13.5, 0.16<z<0.5) / Disruption Comparison.
    """
    _print_plot_header('f_ICS vs Mass-gap')
    frac_labels = {
        'f_baryon': r'$M_\mathrm{ICS}/(f_b\,M_\mathrm{vir})$',
        'f_stellar': r'$M_\mathrm{ICS}/M_{\star,\mathrm{tot}}$',
        'f_baryon_300': r'$M_{\mathrm{ICS},300}/(f_b\,M_\mathrm{vir})$',
        'f_stellar_300': r'$M_{\mathrm{ICS},300}/M_{\star,\mathrm{tot},300}$',
    }
    frac_colours = {
        'f_baryon': '#0C5DA5',
        'f_stellar': '#FF2C00',
        'f_baryon_300': '#0C5DA5',
        'f_stellar_300': '#FF2C00',
    }

    # 4th panel: Gaussian on massive haloes at higher redshift
    sub_mask = ((pool['mvir'] >= 10**13.5)
                & (pool['z'] > 0.16) & (pool['z'] <= 0.5))
    pool_sub = _filter_pool(pool, sub_mask)

    panels = list(SAMPLERS.items()) + [
        (r'Gauss ($M>10^{13.5}$, $z>0.16$)', sample_gaussian)]
    pools = [pool] * len(SAMPLERS) + [pool_sub]
    has_comp = all_pools is not None and len(all_pools) > 1

    fig_h = 16 if SHOW_STATS_BOXES else 10
    fig = plt.figure(figsize=(18, fig_h))
    if SHOW_STATS_BOXES:
        outer_gs = gridspec.GridSpec(4, 1, height_ratios=[2, 1, 2, 1], hspace=0.35,
                                    figure=fig)
        ax_gs_r0 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[0],
                                                    wspace=0.30)
        stat_gs_r0 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[1],
                                                      wspace=0.30)
        ax_gs_r1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[2],
                                                    wspace=0.30)
        stat_gs_r1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[3],
                                                      wspace=0.30)
        ax_specs = [ax_gs_r0[0], ax_gs_r0[1], ax_gs_r0[2],
                    ax_gs_r1[0], ax_gs_r1[1]]
        stat_specs = [stat_gs_r0[0], stat_gs_r0[1], stat_gs_r0[2],
                      stat_gs_r1[0], stat_gs_r1[1]]
    else:
        outer_gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.35,
                                    figure=fig)
        ax_gs_r0 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[0],
                                                    wspace=0.30)
        ax_gs_r1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[1],
                                                    wspace=0.30)
        stat_gs_r1 = None
        ax_specs = [ax_gs_r0[0], ax_gs_r0[1], ax_gs_r0[2],
                    ax_gs_r1[0], ax_gs_r1[1]]
        stat_specs = [None] * len(ax_specs)
    all_handles, all_labels = [], []

    n_sizes = len(N_HALOES)

    for i, ((name, sampler), src) in enumerate(zip(panels, pools)):
        ax = fig.add_subplot(ax_specs[i])
        ax_stat = fig.add_subplot(stat_specs[i]) if SHOW_STATS_BOXES else None
        if ax_stat is not None:
            ax_stat.axis('off')

        use_aper = aperture_all or (i == len(SAMPLERS))
        gap_key = 'mass_gap_300' if (use_aper and 'mass_gap_300' in src) else 'mass_gap'
        bkey = 'f_baryon_300' if use_aper else 'f_baryon'
        skey = 'f_stellar_300' if use_aper else 'f_stellar'

        stat_boxes = {j: [] for j in range(n_sizes)}
        n_draw = {}
        n_used = {j: {} for j in range(n_sizes)}

        for j, n_target in enumerate(N_HALOES):
            idx = sampler(src, n_target, seed=SEED + j)
            if len(idx) == 0:
                continue

            n_draw[j] = int(len(idx))

            gap_all = src[gap_key][idx]
            f_baryon_all = src[bkey][idx]
            f_stellar_all = src[skey][idx]
            lw = 1.0 + 1.5 * (j / max(n_sizes - 1, 1))
            alpha = 0.4 + 0.6 * (j / max(n_sizes - 1, 1))

            for fkey, f_all in [(bkey, f_baryon_all),
                                (skey, f_stellar_all)]:
                finite = np.isfinite(gap_all) & np.isfinite(f_all)
                gap_sel = gap_all[finite]
                f_sel = f_all[finite]
                n_actual = len(gap_sel)
                n_used[j][fkey] = int(n_actual)
                if n_actual < 5:
                    continue

                xc, med = _median_in_bins(gap_sel, f_sel)
                if len(xc) == 0:
                    continue
                label = (f'N={n_actual} {frac_labels[fkey]}'
                         if j == n_sizes - 1 else
                         f'N={n_actual}' if fkey == bkey else None)
                ax.plot(xc, med,
                        color=frac_colours[fkey],
                        lw=lw, alpha=alpha,
                        linestyle='-' if fkey == bkey else '--',
                        label=label)

                rho, pval = spearmanr(gap_sel, f_sel)
                if fkey == bkey:
                    nice = 'baryon(300)' if bkey != 'f_baryon' else 'baryon'
                else:
                    nice = 'stellar(300)' if skey != 'f_stellar' else 'stellar'
                ci_txt = format_effect_ci(rho, n_actual, alpha=0.05)
                dcor_txt = format_dcor(gap_sel, f_sel, seed=SEED + j)
                stat_boxes[j].append(
                    f'{nice}: '
                    r'$\rho_s$' f' = {rho:+.3f}, {ci_txt}, '
                    f'p = {pval:.2e}, {dcor_txt}')

        ax.set_xlabel(r'$\Delta M_\star = \log_{10}(M_{\mathrm{BCG}}) '
                      r'- \log_{10}(M_{\mathrm{sat,\,max}})$')
        if i == 0:
            ax.set_ylabel(r'$f_\mathrm{ICS}$')
        ax.set_title(name, fontsize=13, fontweight='bold')

        if i == 0:
            h, l = ax.get_legend_handles_labels()
            all_handles.extend(h)
            all_labels.extend(l)

        y_top = 0.95
        y_step = 1.0 / max(n_sizes, 1)
        for j in range(n_sizes):
            if not stat_boxes[j]:
                continue
            draw = n_draw.get(j, 0)
            nb = n_used[j].get(bkey, 0)
            ns = n_used[j].get(skey, 0)
            if nb == ns:
                header = f'N_used={nb} (draw={draw})'
            else:
                header = f'N_used(b)={nb}, N_used(*)={ns} (draw={draw})'

            text = header + '\n' + '\n'.join(stat_boxes[j])
            if SHOW_STATS_BOXES:
                ax_stat.text(0.5, y_top - j * y_step, text,
                             transform=ax_stat.transAxes,
                             fontsize=7, ha='center', va='top',
                             bbox=dict(boxstyle='round,pad=0.3',
                                       facecolor='white', edgecolor='grey',
                                       alpha=0.9))
            _print_stats(name, text)

        if use_aper and gap_key == 'mass_gap_300':
            ax.set_xlabel(r'$\Delta M_{\star,300} = \log_{10}(M_{\mathrm{BCG}}) '
                          r'- \log_{10}(M_{\mathrm{sat,\,max},300})$')

    if has_comp:
        ic = len(panels)
        ax = fig.add_subplot(ax_specs[ic])
        ax_stat = fig.add_subplot(stat_specs[ic]) if SHOW_STATS_BOXES else None
        if ax_stat is not None:
            ax_stat.axis('off')
        # Match the "Gaussian + selection" population used in the 4th panel.
        all_pools_sel = []
        for label, p, colour in all_pools:
            p_sel = _filter_pool(
                p,
                (p['mvir'] >= 10**13.5) & (p['z'] > 0.16) & (p['z'] <= Z_MAX),
            )
            all_pools_sel.append((label, p_sel, colour))
        gap_key = ('mass_gap_300'
                   if all(('mass_gap_300' in psel) for _, psel, _ in all_pools_sel)
                   else 'mass_gap')
        _draw_disruption_panel(ax, ax_stat, all_pools_sel, gap_key,
                       _FICS_Y_CONFIGS_300)
        h, l = ax.get_legend_handles_labels()
        all_handles.extend(h)
        all_labels.extend(l)
        leg = ax.get_legend()
        if leg:
            leg.remove()
        if gap_key == 'mass_gap_300':
            ax.set_xlabel(r'$\Delta M_{\star,300} = \log_{10}(M_{\mathrm{BCG}}) '
                          r'- \log_{10}(M_{\mathrm{sat,\,max},300})$')
        else:
            ax.set_xlabel(r'$\Delta M_\star = \log_{10}(M_{\mathrm{BCG}}) '
                          r'- \log_{10}(M_{\mathrm{sat,\,max}})$')
        ax.set_title('Disruption Comparison', fontsize=13, fontweight='bold')

    ax_leg = fig.add_subplot(ax_gs_r1[2])
    ax_leg.axis('off')
    if all_handles:
        ax_leg.legend(all_handles, all_labels, loc='center', fontsize=10,
                      frameon=True, fancybox=True, edgecolor='grey')
    if SHOW_STATS_BOXES and stat_gs_r1 is not None:
        stat_leg = fig.add_subplot(stat_gs_r1[2])
        stat_leg.axis('off')

    # fig.suptitle(r'$f_\mathrm{ICS}$ vs Mass-gap by Sampling Strategy',
    #              fontsize=15, fontweight='bold', y=0.98)

    outfile = os.path.join(output_dir, 'ICS_fics_vs_massgap.pdf')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {outfile}')


def plot_fics_vs_massgap_selection_panel(pool, output_dir, *, aperture_all=False):
    """Selection panel from `ICS_fics_vs_massgap.pdf` as a standalone figure.

    This reproduces the 4th panel: Gaussian sampling of massive, higher-z haloes
    (Mvir > 10^13.5 and z > 0.16 within the default z-range), using the 300 kpc
    aperture-based f_ICS definitions where available.
    """
    _print_plot_header('f_ICS vs Mass-gap (Selection panel only)')
    if pool is None or len(pool.get('mvir', [])) == 0:
        print('  No haloes in pool; skipping selection-only f_ICS vs mass-gap plot.')
        return

    frac_labels = {
        'f_baryon': r'$M_\mathrm{ICS}/(f_b\,M_\mathrm{vir})$',
        'f_stellar': r'$M_\mathrm{ICS}/M_{\star,\mathrm{tot}}$',
        'f_baryon_300': r'$M_{\mathrm{ICS},300}/(f_b\,M_\mathrm{vir})$',
        'f_stellar_300': r'$M_{\mathrm{ICS},300}/M_{\star,\mathrm{tot},300}$',
    }
    frac_colours = {
        'f_baryon': '#0C5DA5',
        'f_stellar': '#FF2C00',
        'f_baryon_300': '#0C5DA5',
        'f_stellar_300': '#FF2C00',
    }

    # Match the selection cut used in the 4th panel of plot_fics_vs_massgap().
    sub_mask = (
        (pool['mvir'] >= 10**13.5)
        & (pool['z'] > 0.16)
        & (pool['z'] <= Z_MAX)
    )
    src = _filter_pool(pool, sub_mask)
    if len(src.get('mvir', [])) == 0:
        print('  Warning: no haloes pass Mvir/z selection; skipping plot.')
        return

    # Selection panel uses aperture-based quantities (300 kpc) when available.
    use_aper = True
    gap_key = 'mass_gap_300' if (use_aper and 'mass_gap_300' in src) else 'mass_gap'
    bkey = 'f_baryon_300' if use_aper else 'f_baryon'
    skey = 'f_stellar_300' if use_aper else 'f_stellar'

    name = r'Gauss ($M>10^{13.5}$, $z>0.16$)'

    if SHOW_STATS_BOXES:
        fig = plt.figure(figsize=(7.2, 8.0))
        outer_gs = gridspec.GridSpec(3, 1, height_ratios=[2.0, 1.0, 0.55], hspace=0.30, figure=fig)
        ax = fig.add_subplot(outer_gs[0])
        ax_stat = fig.add_subplot(outer_gs[1])
        ax_leg = fig.add_subplot(outer_gs[2])
        ax_stat.axis('off')
        ax_leg.axis('off')
    else:
        fig = plt.figure(figsize=(7.2, 6.4))
        outer_gs = gridspec.GridSpec(2, 1, height_ratios=[1.0, 0.23], hspace=0.05, figure=fig)
        ax = fig.add_subplot(outer_gs[0])
        ax_leg = fig.add_subplot(outer_gs[1])
        ax_stat = None
        ax_leg.axis('off')

    n_sizes = len(N_HALOES)
    stat_boxes = {j: [] for j in range(n_sizes)}
    n_draw = {}
    n_used = {j: {} for j in range(n_sizes)}

    for j, n_target in enumerate(N_HALOES):
        idx = sample_gaussian(src, n_target, seed=SEED + j)
        if len(idx) == 0:
            continue

        n_draw[j] = int(len(idx))

        gap_all = src[gap_key][idx]
        f_baryon_all = src[bkey][idx]
        f_stellar_all = src[skey][idx]
        lw = 1.0 + 1.5 * (j / max(n_sizes - 1, 1))
        alpha = 0.4 + 0.6 * (j / max(n_sizes - 1, 1))

        for fkey, f_all in [(bkey, f_baryon_all), (skey, f_stellar_all)]:
            finite = np.isfinite(gap_all) & np.isfinite(f_all)
            gap_sel = gap_all[finite]
            f_sel = f_all[finite]
            n_actual = int(len(gap_sel))
            n_used[j][fkey] = n_actual
            if n_actual < 5:
                continue

            xc, med = _median_in_bins(gap_sel, f_sel)
            if len(xc) == 0:
                continue

            label = (
                f'N={n_actual} {frac_labels[fkey]}' if j == n_sizes - 1 else
                f'N={n_actual}' if fkey == bkey else None
            )
            ax.plot(
                xc,
                med,
                color=frac_colours[fkey],
                lw=lw,
                alpha=alpha,
                linestyle='-' if fkey == bkey else '--',
                label=label,
            )

            rho, pval = spearmanr(gap_sel, f_sel)
            if fkey == bkey:
                nice = 'baryon(300)' if bkey != 'f_baryon' else 'baryon'
            else:
                nice = 'stellar(300)' if skey != 'f_stellar' else 'stellar'
            ci_txt = format_effect_ci(rho, n_actual, alpha=0.05)
            dcor_txt = format_dcor(gap_sel, f_sel, seed=SEED + j)
            stat_boxes[j].append(
                f'{nice}: '
                r'$\rho_s$' f' = {rho:+.3f}, {ci_txt}, '
                f'p = {pval:.2e}, {dcor_txt}')

    # Axis labels
    if use_aper and gap_key == 'mass_gap_300':
        ax.set_xlabel(r'$\Delta M_{\star,300} = \log_{10}(M_{\mathrm{BCG}}) '
                      r'- \log_{10}(M_{\mathrm{sat,\,max},300})$')
    else:
        ax.set_xlabel(r'$\Delta M_\star = \log_{10}(M_{\mathrm{BCG}}) '
                      r'- \log_{10}(M_{\mathrm{sat,\,max}})$')
    ax.set_ylabel(r'$f_\mathrm{ICS}$')

    # Stats box (optional)
    if SHOW_STATS_BOXES and ax_stat is not None:
        y_top = 0.95
        y_step = 1.0 / max(n_sizes, 1)
        for j in range(n_sizes):
            if not stat_boxes[j]:
                continue
            draw = n_draw.get(j, 0)
            nb = n_used[j].get(bkey, 0)
            ns = n_used[j].get(skey, 0)
            if nb == ns:
                header = f'N_used={nb} (draw={draw})'
            else:
                header = f'N_used(b)={nb}, N_used(*)={ns} (draw={draw})'
            text = header + '\n' + '\n'.join(stat_boxes[j])
            ax_stat.text(
                0.5, y_top - j * y_step, text,
                transform=ax_stat.transAxes,
                fontsize=7, ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='grey', alpha=0.9),
            )
            _print_stats(name, text)

    # Legend in its own strip, slightly low
    h, l = ax.get_legend_handles_labels()
    leg = ax.get_legend()
    if leg:
        leg.remove()
    if h:
        ax_leg.legend(
            h, l,
            loc='lower center',
            bbox_to_anchor=(0.5, 0.0),
            borderaxespad=0.0,
            fontsize=10,
            frameon=True,
            fancybox=True,
            edgecolor='grey',
            ncol=2,
        )

    # No title (to match the standalone Mvir selection panel style).
    outfile = os.path.join(output_dir, 'ICS_fics_vs_massgap_selection_panel.pdf')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {outfile}')


# ── Plot 4: f_ICS vs Mvir ────────────────────────────────────────────────────

def plot_fics_vs_mvir(pool, output_dir, all_pools=None, *, aperture_all=False):
    """
    4 (+1 comparison) panels: Random / Uniform / Gaussian /
    Gaussian (mass-gap < 1.2) / Disruption Comparison.
    """
    _print_plot_header('f_ICS vs Mvir')
    frac_labels = {
        'f_baryon': r'$M_\mathrm{ICS}/(f_b\,M_\mathrm{vir})$',
        'f_stellar': r'$M_\mathrm{ICS}/M_{\star,\mathrm{tot}}$',
        'f_baryon_300': r'$M_{\mathrm{ICS},300}/(f_b\,M_\mathrm{vir})$',
        'f_stellar_300': r'$M_{\mathrm{ICS},300}/M_{\star,\mathrm{tot},300}$',
    }
    frac_colours = {
        'f_baryon': '#0C5DA5',
        'f_stellar': '#FF2C00',
        'f_baryon_300': '#0C5DA5',
        'f_stellar_300': '#FF2C00',
    }

    # 4th panel: Gaussian on haloes with small mass-gap
    gap_key = 'mass_gap_300' if 'mass_gap_300' in pool else 'mass_gap'
    gap_mask = pool[gap_key] < 1.2
    pool_smallgap = _filter_pool(pool, gap_mask)

    panels = list(SAMPLERS.items()) + [
        (r'Gauss (gap $< 1.2$)', sample_gaussian)]
    pools = [pool] * len(SAMPLERS) + [pool_smallgap]
    has_comp = all_pools is not None and len(all_pools) > 1

    fig_h = 16 if SHOW_STATS_BOXES else 10
    fig = plt.figure(figsize=(18, fig_h))
    if SHOW_STATS_BOXES:
        outer_gs = gridspec.GridSpec(4, 1, height_ratios=[2, 1, 2, 1], hspace=0.35,
                                    figure=fig)
        ax_gs_r0 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[0],
                                                    wspace=0.30)
        stat_gs_r0 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[1],
                                                      wspace=0.30)
        ax_gs_r1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[2],
                                                    wspace=0.30)
        stat_gs_r1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[3],
                                                      wspace=0.30)
        ax_specs = [ax_gs_r0[0], ax_gs_r0[1], ax_gs_r0[2],
                    ax_gs_r1[0], ax_gs_r1[1]]
        stat_specs = [stat_gs_r0[0], stat_gs_r0[1], stat_gs_r0[2],
                      stat_gs_r1[0], stat_gs_r1[1]]
    else:
        outer_gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.35,
                                    figure=fig)
        ax_gs_r0 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[0],
                                                    wspace=0.30)
        ax_gs_r1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[1],
                                                    wspace=0.30)
        stat_gs_r1 = None
        ax_specs = [ax_gs_r0[0], ax_gs_r0[1], ax_gs_r0[2],
                    ax_gs_r1[0], ax_gs_r1[1]]
        stat_specs = [None] * len(ax_specs)
    all_handles, all_labels = [], []

    for i, ((name, sampler), src) in enumerate(zip(panels, pools)):
        ax = fig.add_subplot(ax_specs[i])
        ax_stat = fig.add_subplot(stat_specs[i]) if SHOW_STATS_BOXES else None
        if ax_stat is not None:
            ax_stat.axis('off')

        use_aper = aperture_all or (i == len(SAMPLERS))
        bkey = 'f_baryon_300' if use_aper else 'f_baryon'
        skey = 'f_stellar_300' if use_aper else 'f_stellar'

        def _partial_spearman_lines(idx, _src=src, _bkey=bkey, _skey=skey):
            """Partial Spearman controlling for redshift."""
            lines = []
            logm_sel = np.log10(_src['mvir'][idx])
            z_sel = _src['z'][idx]
            for fkey in [_bkey, _skey]:
                f_sel = _src[fkey][idx]
                finite = np.isfinite(logm_sel) & np.isfinite(f_sel) & np.isfinite(z_sel)
                if np.sum(finite) < 5:
                    continue
                rho_p, pval_p = partial_spearman(logm_sel[finite], f_sel[finite], z_sel[finite])
                if fkey == _bkey:
                    nice = 'baryon(300)' if _bkey != 'f_baryon' else 'baryon'
                else:
                    nice = 'stellar(300)' if _skey != 'f_stellar' else 'stellar'
                ci_txt = format_effect_ci(rho_p, int(np.sum(finite)), alpha=0.05)
                dcor_txt = format_dcor(logm_sel[finite], f_sel[finite], seed=SEED)
                lines.append(
                    f'{nice}: '
                    r'$\rho_{s|z}$' f' = {rho_p:+.3f}, {ci_txt}, '
                    f'p = {pval_p:.2e}, {dcor_txt}')
            return lines

        _draw_fics_panel(ax, ax_stat, src, sampler, 'log_mvir',
                         frac_labels, frac_colours,
                         extra_spearman_fn=_partial_spearman_lines,
                         stat_y_step_factor=1.15, panel_name=name,
                         baryon_key=bkey, stellar_key=skey,
                         )

        if i == 0:
            h, l = ax.get_legend_handles_labels()
            all_handles.extend(h)
            all_labels.extend(l)
        leg = ax.get_legend()
        if leg:
            leg.remove()

        ax.set_xlabel(r'$\log_{10}\,M_{\mathrm{vir}}\ [\mathrm{M}_\odot]$')
        if i == 0:
            ax.set_ylabel(r'$f_\mathrm{ICS}$')
        ax.set_title(name, fontsize=13, fontweight='bold')

    if has_comp:
        ic = len(panels)
        ax = fig.add_subplot(ax_specs[ic])
        ax_stat = fig.add_subplot(stat_specs[ic]) if SHOW_STATS_BOXES else None
        if ax_stat is not None:
            ax_stat.axis('off')
        # Match the "Gaussian + selection" population used in the 4th panel.
        all_pools_sel = []
        for label, p, colour in all_pools:
            gk = 'mass_gap_300' if 'mass_gap_300' in p else 'mass_gap'
            p_sel = _filter_pool(p, p[gk] < 1.2)
            all_pools_sel.append((label, p_sel, colour))
        _draw_disruption_panel(ax, ax_stat, all_pools_sel, 'log_mvir',
                       _FICS_Y_CONFIGS_300)
        h, l = ax.get_legend_handles_labels()
        all_handles.extend(h)
        all_labels.extend(l)
        leg = ax.get_legend()
        if leg:
            leg.remove()
        ax.set_xlabel(r'$\log_{10}\,M_{\mathrm{vir}}\ [\mathrm{M}_\odot]$')
        ax.set_title('Disruption Comparison', fontsize=13, fontweight='bold')

    ax_leg = fig.add_subplot(ax_gs_r1[2])
    ax_leg.axis('off')
    if all_handles:
        ax_leg.legend(all_handles, all_labels, loc='center', fontsize=10,
                      frameon=True, fancybox=True, edgecolor='grey')
    if SHOW_STATS_BOXES and stat_gs_r1 is not None:
        stat_leg = fig.add_subplot(stat_gs_r1[2])
        stat_leg.axis('off')

    # fig.suptitle(r'$f_\mathrm{ICS}$ vs Halo Mass by Sampling Strategy',
    #              fontsize=15, fontweight='bold', y=0.98)

    outfile = os.path.join(output_dir, 'ICS_fics_vs_mvir.pdf')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {outfile}')


def plot_fics_vs_mvir_selection_panel(pool, output_dir, *, aperture_all=False):
    """Selection panel from `ICS_fics_vs_mvir.pdf` as a standalone figure.

    This reproduces the 4th panel: Gaussian sampling of haloes with a small
    mass-gap (gap < 1.2), using the 300 kpc aperture-based f_ICS definitions.
    """
    _print_plot_header('f_ICS vs Mvir (Selection panel only)')
    if pool is None or len(pool.get('mvir', [])) == 0:
        print('  No haloes in pool; skipping selection-only f_ICS vs Mvir plot.')
        return

    frac_labels = {
        'f_baryon': r'$M_\mathrm{ICS}/(f_b\,M_\mathrm{vir})$',
        'f_stellar': r'$M_\mathrm{ICS}/M_{\star,\mathrm{tot}}$',
        'f_baryon_300': r'$M_{\mathrm{ICS},300}/(f_b\,M_\mathrm{vir})$',
        'f_stellar_300': r'$M_{\mathrm{ICS},300}/M_{\star,\mathrm{tot},300}$',
    }
    frac_colours = {
        'f_baryon': '#0C5DA5',
        'f_stellar': '#FF2C00',
        'f_baryon_300': '#0C5DA5',
        'f_stellar_300': '#FF2C00',
    }

    # Match the selection cut used in the 4th panel.
    gap_key = 'mass_gap_300' if 'mass_gap_300' in pool else 'mass_gap'
    pool_smallgap = _filter_pool(pool, pool[gap_key] < 1.2)
    if len(pool_smallgap.get('mvir', [])) == 0:
        print('  Warning: no haloes pass gap < 1.2; skipping plot.')
        return

    # Selection panel uses aperture-based fractions by default.
    use_aper = True if not aperture_all else True
    bkey = 'f_baryon_300' if use_aper else 'f_baryon'
    skey = 'f_stellar_300' if use_aper else 'f_stellar'

    name = r'Gauss (gap $< 1.2$)'

    def _partial_spearman_lines(idx, _src=pool_smallgap, _bkey=bkey, _skey=skey):
        """Partial Spearman controlling for redshift."""
        lines = []
        logm_sel = np.log10(_src['mvir'][idx])
        z_sel = _src['z'][idx]
        for fkey in [_bkey, _skey]:
            f_sel = _src[fkey][idx]
            finite = np.isfinite(logm_sel) & np.isfinite(f_sel) & np.isfinite(z_sel)
            if np.sum(finite) < 5:
                continue
            rho_p, pval_p = partial_spearman(logm_sel[finite], f_sel[finite], z_sel[finite])
            if fkey == _bkey:
                nice = 'baryon(300)' if _bkey != 'f_baryon' else 'baryon'
            else:
                nice = 'stellar(300)' if _skey != 'f_stellar' else 'stellar'
            ci_txt = format_effect_ci(rho_p, int(np.sum(finite)), alpha=0.05)
            dcor_txt = format_dcor(logm_sel[finite], f_sel[finite], seed=SEED)
            lines.append(
                f'{nice}: '
                r'$\rho_{s|z}$' f' = {rho_p:+.3f}, {ci_txt}, '
                f'p = {pval_p:.2e}, {dcor_txt}')
        return lines

    if SHOW_STATS_BOXES:
        fig = plt.figure(figsize=(7.2, 8.0))
        outer_gs = gridspec.GridSpec(3, 1, height_ratios=[2.0, 1.0, 0.55], hspace=0.30, figure=fig)
        ax = fig.add_subplot(outer_gs[0])
        ax_stat = fig.add_subplot(outer_gs[1])
        ax_leg = fig.add_subplot(outer_gs[2])
        ax_stat.axis('off')
        ax_leg.axis('off')
    else:
        fig = plt.figure(figsize=(7.2, 6.4))
        outer_gs = gridspec.GridSpec(2, 1, height_ratios=[1.0, 0.23], hspace=0.05, figure=fig)
        ax = fig.add_subplot(outer_gs[0])
        ax_leg = fig.add_subplot(outer_gs[1])
        ax_stat = None
        ax_leg.axis('off')

    _draw_fics_panel(
        ax,
        ax_stat,
        pool_smallgap,
        sample_gaussian,
        'log_mvir',
        frac_labels,
        frac_colours,
        extra_spearman_fn=_partial_spearman_lines,
        stat_y_step_factor=1.15,
        panel_name=name,
        baryon_key=bkey,
        stellar_key=skey,
    )

    h, l = ax.get_legend_handles_labels()
    leg = ax.get_legend()
    if leg:
        leg.remove()
    if h:
        ax_leg.legend(
            h, l,
            loc='lower center',
            bbox_to_anchor=(0.5, 0.0),
            borderaxespad=0.0,
            fontsize=10,
            frameon=True,
            fancybox=True,
            edgecolor='grey',
            ncol=2,
        )

    ax.set_xlabel(r'$\log_{10}\,M_{\mathrm{vir}}\ [\mathrm{M}_\odot]$')
    ax.set_ylabel(r'$f_\mathrm{ICS}$')

    outfile = os.path.join(output_dir, 'ICS_fics_vs_mvir_selection_panel.pdf')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {outfile}')


# ── Plot 5: f_ICS vs BCG metallicity ──────────────────────────────────────────

def plot_fics_vs_bcg_metallicity(pool, output_dir, all_pools=None):
    """
    3 (+1 comparison) panels: Random / Uniform / Gaussian /
    Disruption Comparison.  f_ICS vs BCG metallicity (Z/Z_sun).
    """
    _print_plot_header('f_ICS vs BCG Metallicity')
    frac_labels = {
        'f_baryon': r'$M_\mathrm{ICS}/(f_b\,M_\mathrm{vir})$',
        'f_stellar': r'$M_\mathrm{ICS}/M_{\star,\mathrm{tot}}$',
    }
    frac_colours = {'f_baryon': '#0C5DA5', 'f_stellar': '#FF2C00'}

    panels = list(SAMPLERS.items())
    pools = [pool] * len(panels)
    has_comp = all_pools is not None and len(all_pools) > 1
    n_panels = len(panels) + (1 if has_comp else 0)

    fig_h = 8 if SHOW_STATS_BOXES else 5.5
    fig = plt.figure(figsize=(6 * n_panels, fig_h))
    if SHOW_STATS_BOXES:
        outer_gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.35,
                                    figure=fig)
        ax_gs = gridspec.GridSpecFromSubplotSpec(1, n_panels, subplot_spec=outer_gs[0],
                                                wspace=0.30)
        stat_gs = gridspec.GridSpecFromSubplotSpec(1, n_panels, subplot_spec=outer_gs[1],
                                                  wspace=0.30)
    else:
        outer_gs = gridspec.GridSpec(1, 1, hspace=0.35, figure=fig)
        ax_gs = gridspec.GridSpecFromSubplotSpec(1, n_panels, subplot_spec=outer_gs[0],
                                                wspace=0.30)
        stat_gs = None

    for i, ((name, sampler), src) in enumerate(zip(panels, pools)):
        ax = fig.add_subplot(ax_gs[i])
        ax_stat = fig.add_subplot(stat_gs[i]) if SHOW_STATS_BOXES else None
        if ax_stat is not None:
            ax_stat.axis('off')

        _draw_fics_panel(ax, ax_stat, src, sampler, 'bcg_Z_solar',
                         frac_labels, frac_colours,
                         panel_name=name)

        ax.set_xlabel(r'$Z_\mathrm{BCG}\,/\,Z_\odot$')
        if i == 0:
            ax.set_ylabel(r'$f_\mathrm{ICS}$')
        ax.set_title(name, fontsize=13, fontweight='bold')

    if has_comp:
        ic = len(panels)
        ax = fig.add_subplot(ax_gs[ic])
        ax_stat = fig.add_subplot(stat_gs[ic]) if SHOW_STATS_BOXES else None
        if ax_stat is not None:
            ax_stat.axis('off')
        _draw_disruption_panel(ax, ax_stat, all_pools, 'bcg_Z_solar',
                       _FICS_Y_CONFIGS)
        ax.set_xlabel(r'$Z_\mathrm{BCG}\,/\,Z_\odot$')
        ax.set_title('Disruption Comparison', fontsize=13, fontweight='bold')

    # fig.suptitle(r'$f_\mathrm{ICS}$ vs BCG Metallicity by Sampling Strategy',
    #              fontsize=15, fontweight='bold', y=0.98)

    outfile = os.path.join(output_dir, 'ICS_fics_vs_bcg_metallicity.pdf')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {outfile}')


# ── Plot 6: ICS metallicity vs BCG metallicity ───────────────────────────────

def plot_ics_vs_bcg_metallicity(pool, output_dir, all_pools=None):
    """
    3 (+1 comparison) panels: Random / Uniform / Gaussian /
    Disruption Comparison.  Z_ICS vs Z_BCG, both in Z/Z_sun.
    """
    _print_plot_header('Z_ICS vs Z_BCG Metallicity')
    panels = list(SAMPLERS.items())
    pools = [pool] * len(panels)
    has_comp = all_pools is not None and len(all_pools) > 1
    n_panels = len(panels) + (1 if has_comp else 0)
    n_sizes = len(N_HALOES)

    fig_h = 8 if SHOW_STATS_BOXES else 5.5
    fig = plt.figure(figsize=(6 * n_panels, fig_h))
    if SHOW_STATS_BOXES:
        outer_gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.35,
                                    figure=fig)
        ax_gs = gridspec.GridSpecFromSubplotSpec(1, n_panels, subplot_spec=outer_gs[0],
                                                wspace=0.30)
        stat_gs = gridspec.GridSpecFromSubplotSpec(1, n_panels, subplot_spec=outer_gs[1],
                                                  wspace=0.30)
    else:
        outer_gs = gridspec.GridSpec(1, 1, hspace=0.35, figure=fig)
        ax_gs = gridspec.GridSpecFromSubplotSpec(1, n_panels, subplot_spec=outer_gs[0],
                                                wspace=0.30)
        stat_gs = None

    for i, ((name, sampler), src) in enumerate(zip(panels, pools)):
        ax = fig.add_subplot(ax_gs[i])
        ax_stat = fig.add_subplot(stat_gs[i]) if SHOW_STATS_BOXES else None
        if ax_stat is not None:
            ax_stat.axis('off')

        stat_boxes = {j: [] for j in range(n_sizes)}

        for j, n_target in enumerate(N_HALOES):
            idx = sampler(src, n_target, seed=SEED + j)
            if len(idx) == 0:
                continue

            x_sel = src['bcg_Z_solar'][idx]
            y_sel = src['ics_Z_solar'][idx]
            c_sel = src['log_mvir'][idx]

            finite = np.isfinite(x_sel) & np.isfinite(y_sel)
            xf, yf, cf = x_sel[finite], y_sel[finite], c_sel[finite]
            n_actual = len(xf)
            if n_actual < 5:
                continue

            lw = 1.0 + 1.5 * (j / max(n_sizes - 1, 1))
            alpha = 0.4 + 0.6 * (j / max(n_sizes - 1, 1))

            # Scatter colored by log10(Mvir) — only for the largest sample
            if j == n_sizes - 1:
                ax.scatter(xf, yf, c=cf, cmap='plasma', s=8,
                           alpha=0.5, edgecolors='none', zorder=1)

            xc, med = _median_in_bins(xf, yf)
            if len(xc) > 0:
                label = f'N={n_actual}'
                ax.plot(xc, med, color='k', lw=lw, alpha=alpha,
                        label=label, zorder=3)
            rho, pval = spearmanr(xf, yf)
            ci_txt = format_effect_ci(rho, n_actual, alpha=0.05)
            dcor_txt = format_dcor(xf, yf, seed=SEED + j)
            stat_boxes[j].append(
                r'$\rho_s$' f' = {rho:+.3f}, {ci_txt}, p = {pval:.2e}, {dcor_txt}')

        # 1:1 line
        lims = ax.get_xlim()
        ax.plot(lims, lims, 'k--', lw=0.8, alpha=0.5, label='1:1', zorder=2)
        ax.set_xlim(lims)

        ax.set_xlabel(r'$Z_\mathrm{BCG}\,/\,Z_\odot$')
        if i == 0:
            ax.set_ylabel(r'$Z_\mathrm{ICS}\,/\,Z_\odot$')
        ax.set_title(name, fontsize=13, fontweight='bold')

        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8, loc='best')

        y_top = 0.95
        y_step = 1.0 / max(n_sizes, 1)
        for j in range(n_sizes):
            if not stat_boxes[j]:
                continue
            header = f'N = {len(sampler(src, N_HALOES[j], seed=SEED + j))}'
            text = header + '\n' + '\n'.join(stat_boxes[j])
            if SHOW_STATS_BOXES:
                ax_stat.text(0.5, y_top - j * y_step, text,
                             transform=ax_stat.transAxes,
                             fontsize=7, ha='center', va='top',
                             bbox=dict(boxstyle='round,pad=0.3',
                                       facecolor='white', edgecolor='grey',
                                       alpha=0.9))
            _print_stats(name, text)

    if has_comp:
        ic = len(panels)
        ax = fig.add_subplot(ax_gs[ic])
        ax_stat = fig.add_subplot(stat_gs[ic]) if SHOW_STATS_BOXES else None
        if ax_stat is not None:
            ax_stat.axis('off')
        _draw_disruption_panel(
            ax,
            ax_stat,
            all_pools,
            'bcg_Z_solar',
            [('ics_Z_solar', '-', r'$Z_\mathrm{ICS}/Z_\odot$')],
        )
        lims = ax.get_xlim()
        ax.plot(lims, lims, 'k--', lw=0.8, alpha=0.5, label='1:1')
        ax.set_xlim(lims)
        ax.set_xlabel(r'$Z_\mathrm{BCG}\,/\,Z_\odot$')
        ax.set_ylabel(r'$Z_\mathrm{ICS}\,/\,Z_\odot$')
        ax.set_title('Disruption Comparison', fontsize=13, fontweight='bold')
        ax.legend(fontsize=7, loc='best')

    # fig.suptitle(r'ICS vs BCG Metallicity by Sampling Strategy',
    #              fontsize=15, fontweight='bold', y=0.98)

    outfile = os.path.join(output_dir, 'ICS_ics_vs_bcg_metallicity.pdf')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {outfile}')


# ── Plot 7: f_ICS vs t_asm ───────────────────────────────────────────────────

def plot_fics_vs_tasm(pool, output_dir, all_pools=None):
    """
    3 (+1 comparison) panels: Random / Uniform / Gaussian /
    Disruption Comparison.  f_ICS vs assembly-weighted lookback time t_asm.

    Pre-filters to finite t_asm so samplers draw only usable entries.
    """
    _print_plot_header('f_ICS vs t_asm')
    # Pre-filter to entries with valid t_asm
    tasm_mask = np.isfinite(pool['t_asm'])
    src_pool = _filter_pool(pool, tasm_mask)
    if len(src_pool.get('t_asm', [])) == 0:
        print('  Warning: no finite t_asm values; skipping f_ICS vs t_asm plot.')
        return

    # Pre-filter disruption pools too
    if all_pools is not None and len(all_pools) > 1:
        all_pools_filt = []
        for label, p, colour in all_pools:
            p_filt = _filter_pool(p, np.isfinite(p['t_asm']))
            if len(p_filt.get('t_asm', [])) > 0:
                all_pools_filt.append((label, p_filt, colour))
        has_comp = len(all_pools_filt) > 1
    else:
        all_pools_filt = []
        has_comp = False

    frac_labels = {
        'f_baryon': r'$M_\mathrm{ICS}/(f_b\,M_\mathrm{vir})$',
        'f_stellar': r'$M_\mathrm{ICS}/M_{\star,\mathrm{tot}}$',
    }
    frac_colours = {'f_baryon': '#0C5DA5', 'f_stellar': '#FF2C00'}

    panels = list(SAMPLERS.items())
    n_panels = len(panels) + (1 if has_comp else 0)

    fig_h = 8 if SHOW_STATS_BOXES else 5.5
    fig = plt.figure(figsize=(6 * n_panels, fig_h))
    if SHOW_STATS_BOXES:
        outer_gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.35,
                                    figure=fig)
        ax_gs = gridspec.GridSpecFromSubplotSpec(1, n_panels, subplot_spec=outer_gs[0],
                                                wspace=0.30)
        stat_gs = gridspec.GridSpecFromSubplotSpec(1, n_panels, subplot_spec=outer_gs[1],
                                                  wspace=0.30)
    else:
        outer_gs = gridspec.GridSpec(1, 1, hspace=0.35, figure=fig)
        ax_gs = gridspec.GridSpecFromSubplotSpec(1, n_panels, subplot_spec=outer_gs[0],
                                                wspace=0.30)
        stat_gs = None

    for i, (name, sampler) in enumerate(panels):
        ax = fig.add_subplot(ax_gs[i])
        ax_stat = fig.add_subplot(stat_gs[i]) if SHOW_STATS_BOXES else None
        if ax_stat is not None:
            ax_stat.axis('off')

        _draw_fics_panel(ax, ax_stat, src_pool, sampler, 't_asm',
                         frac_labels, frac_colours, panel_name=name)

        ax.set_xlabel(r'$t_\mathrm{asm}$ [Gyr lookback]')
        if i == 0:
            ax.set_ylabel(r'$f_\mathrm{ICS}$')
        ax.set_title(name, fontsize=13, fontweight='bold')

    if has_comp:
        ic = len(panels)
        ax = fig.add_subplot(ax_gs[ic])
        ax_stat = fig.add_subplot(stat_gs[ic]) if SHOW_STATS_BOXES else None
        if ax_stat is not None:
            ax_stat.axis('off')
        _draw_disruption_panel(ax, ax_stat, all_pools_filt, 't_asm',
                       _FICS_Y_CONFIGS)
        ax.set_xlabel(r'$t_\mathrm{asm}$ [Gyr lookback]')
        ax.set_title('Disruption Comparison', fontsize=13, fontweight='bold')

    # fig.suptitle(r'$f_\mathrm{ICS}$ vs $t_\mathrm{asm}$ by Sampling Strategy',
    #              fontsize=15, fontweight='bold', y=0.98)

    outfile = os.path.join(output_dir, 'ICS_fics_vs_tasm.pdf')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {outfile}')


# ── Plot 8: f_ICS vs t_50 ────────────────────────────────────────────────────

def plot_fics_vs_t50(pool, output_dir, all_pools=None):
    """
    3 (+1 comparison) panels: Random / Uniform / Gaussian /
    Disruption Comparison.  f_ICS vs t_50 (lookback time when cumulative SFH
    reaches 50% of z=0 stellar mass).

    t_50 is only valid at z=0, so the pool is pre-filtered to finite t_50.
    """
    _print_plot_header('f_ICS vs t_50')
    # Pre-filter to z~0 entries with valid t_50
    t50_mask = np.isfinite(pool['t_50'])
    src_pool = _filter_pool(pool, t50_mask)
    if len(src_pool.get('t_50', [])) == 0:
        print('  Warning: no finite t_50 values; skipping f_ICS vs t_50 plot.')
        return

    # Pre-filter disruption pools too
    if all_pools is not None and len(all_pools) > 1:
        all_pools_filt = []
        for label, p, colour in all_pools:
            p_filt = _filter_pool(p, np.isfinite(p['t_50']))
            if len(p_filt.get('t_50', [])) > 0:
                all_pools_filt.append((label, p_filt, colour))
        has_comp = len(all_pools_filt) > 1
    else:
        all_pools_filt = []
        has_comp = False

    frac_labels = {
        'f_baryon': r'$M_\mathrm{ICS}/(f_b\,M_\mathrm{vir})$',
        'f_stellar': r'$M_\mathrm{ICS}/M_{\star,\mathrm{tot}}$',
    }
    frac_colours = {'f_baryon': '#0C5DA5', 'f_stellar': '#FF2C00'}

    panels = list(SAMPLERS.items())
    n_panels = len(panels) + (1 if has_comp else 0)

    fig_h = 8 if SHOW_STATS_BOXES else 5.5
    fig = plt.figure(figsize=(6 * n_panels, fig_h))
    if SHOW_STATS_BOXES:
        outer_gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.35,
                                    figure=fig)
        ax_gs = gridspec.GridSpecFromSubplotSpec(1, n_panels, subplot_spec=outer_gs[0],
                                                wspace=0.30)
        stat_gs = gridspec.GridSpecFromSubplotSpec(1, n_panels, subplot_spec=outer_gs[1],
                                                  wspace=0.30)
    else:
        outer_gs = gridspec.GridSpec(1, 1, hspace=0.35, figure=fig)
        ax_gs = gridspec.GridSpecFromSubplotSpec(1, n_panels, subplot_spec=outer_gs[0],
                                                wspace=0.30)
        stat_gs = None

    for i, (name, sampler) in enumerate(panels):
        ax = fig.add_subplot(ax_gs[i])
        ax_stat = fig.add_subplot(stat_gs[i]) if SHOW_STATS_BOXES else None
        if ax_stat is not None:
            ax_stat.axis('off')

        _draw_fics_panel(ax, ax_stat, src_pool, sampler, 't_50',
                         frac_labels, frac_colours, panel_name=name)

        ax.set_xlabel(r'$t_{50}$ [Gyr lookback]')
        if i == 0:
            ax.set_ylabel(r'$f_\mathrm{ICS}$')
        ax.set_title(name, fontsize=13, fontweight='bold')

    if has_comp:
        ic = len(panels)
        ax = fig.add_subplot(ax_gs[ic])
        ax_stat = fig.add_subplot(stat_gs[ic]) if SHOW_STATS_BOXES else None
        if ax_stat is not None:
            ax_stat.axis('off')
        _draw_disruption_panel(ax, ax_stat, all_pools_filt, 't_50',
                       _FICS_Y_CONFIGS)
        ax.set_xlabel(r'$t_{50}$ [Gyr lookback]')
        ax.set_title('Disruption Comparison', fontsize=13, fontweight='bold')

    # fig.suptitle(r'$f_\mathrm{ICS}$ vs $t_{50}$ by Sampling Strategy',
    #              fontsize=15, fontweight='bold', y=0.98)

    outfile = os.path.join(output_dir, 'ICS_fics_vs_t50.pdf')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {outfile}')


# ── Plot 9: f_ICS vs ICS_disrupt ─────────────────────────────────────────────

def plot_fics_vs_ics_disrupt(pool, output_dir, all_pools=None):
    """
    3 (+1 comparison) panels: Random / Uniform / Gaussian /
    Disruption Comparison.  f_ICS vs cumulative disrupted stellar mass.
    """
    _print_plot_header('f_ICS vs ICS Disrupted Mass')
    frac_labels = {
        'f_baryon': r'$M_\mathrm{ICS}/(f_b\,M_\mathrm{vir})$',
        'f_stellar': r'$M_\mathrm{ICS}/M_{\star,\mathrm{tot}}$',
    }
    frac_colours = {'f_baryon': '#0C5DA5', 'f_stellar': '#FF2C00'}

    panels = list(SAMPLERS.items())
    has_comp = all_pools is not None and len(all_pools) > 1
    n_panels = len(panels) + (1 if has_comp else 0)

    fig_h = 8 if SHOW_STATS_BOXES else 5.5
    fig = plt.figure(figsize=(6 * n_panels, fig_h))
    if SHOW_STATS_BOXES:
        outer_gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.35,
                                    figure=fig)
        ax_gs = gridspec.GridSpecFromSubplotSpec(1, n_panels, subplot_spec=outer_gs[0],
                                                wspace=0.30)
        stat_gs = gridspec.GridSpecFromSubplotSpec(1, n_panels, subplot_spec=outer_gs[1],
                                                  wspace=0.30)
    else:
        outer_gs = gridspec.GridSpec(1, 1, hspace=0.35, figure=fig)
        ax_gs = gridspec.GridSpecFromSubplotSpec(1, n_panels, subplot_spec=outer_gs[0],
                                                wspace=0.30)
        stat_gs = None

    for i, (name, sampler) in enumerate(panels):
        ax = fig.add_subplot(ax_gs[i])
        ax_stat = fig.add_subplot(stat_gs[i]) if SHOW_STATS_BOXES else None
        if ax_stat is not None:
            ax_stat.axis('off')

        _draw_fics_panel(ax, ax_stat, pool, sampler, 'log_ics_disrupt',
                         frac_labels, frac_colours, panel_name=name)

        ax.set_xlabel(r'$\log_{10}\,M_\mathrm{ICS,\,disrupt}\ [\mathrm{M}_\odot]$')
        if i == 0:
            ax.set_ylabel(r'$f_\mathrm{ICS}$')
        ax.set_title(name, fontsize=13, fontweight='bold')

    if has_comp:
        ic = len(panels)
        ax = fig.add_subplot(ax_gs[ic])
        ax_stat = fig.add_subplot(stat_gs[ic]) if SHOW_STATS_BOXES else None
        if ax_stat is not None:
            ax_stat.axis('off')
        _draw_disruption_panel(ax, ax_stat, all_pools, 'log_ics_disrupt',
                       _FICS_Y_CONFIGS)
        ax.set_xlabel(r'$\log_{10}\,M_\mathrm{ICS,\,disrupt}\ [\mathrm{M}_\odot]$')
        ax.set_title('Disruption Comparison', fontsize=13, fontweight='bold')

    # fig.suptitle(r'$f_\mathrm{ICS}$ vs Cumulative Disrupted Mass',
    #              fontsize=15, fontweight='bold', y=0.98)

    outfile = os.path.join(output_dir, 'ICS_fics_vs_ics_disrupt.pdf')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {outfile}')


# ── Plot 10: f_ICS vs ICS_accrete ────────────────────────────────────────────

def plot_fics_vs_ics_accrete(pool, output_dir, all_pools=None):
    """
    3 (+1 comparison) panels: Random / Uniform / Gaussian /
    Disruption Comparison.  f_ICS vs cumulative accreted ICS mass.
    """
    _print_plot_header('f_ICS vs ICS Accreted Mass')
    frac_labels = {
        'f_baryon': r'$M_\mathrm{ICS}/(f_b\,M_\mathrm{vir})$',
        'f_stellar': r'$M_\mathrm{ICS}/M_{\star,\mathrm{tot}}$',
    }
    frac_colours = {'f_baryon': '#0C5DA5', 'f_stellar': '#FF2C00'}

    panels = list(SAMPLERS.items())
    has_comp = all_pools is not None and len(all_pools) > 1
    n_panels = len(panels) + (1 if has_comp else 0)

    fig_h = 8 if SHOW_STATS_BOXES else 5.5
    fig = plt.figure(figsize=(6 * n_panels, fig_h))
    if SHOW_STATS_BOXES:
        outer_gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.35,
                                    figure=fig)
        ax_gs = gridspec.GridSpecFromSubplotSpec(1, n_panels, subplot_spec=outer_gs[0],
                                                wspace=0.30)
        stat_gs = gridspec.GridSpecFromSubplotSpec(1, n_panels, subplot_spec=outer_gs[1],
                                                  wspace=0.30)
    else:
        outer_gs = gridspec.GridSpec(1, 1, hspace=0.35, figure=fig)
        ax_gs = gridspec.GridSpecFromSubplotSpec(1, n_panels, subplot_spec=outer_gs[0],
                                                wspace=0.30)
        stat_gs = None

    for i, (name, sampler) in enumerate(panels):
        ax = fig.add_subplot(ax_gs[i])
        ax_stat = fig.add_subplot(stat_gs[i]) if SHOW_STATS_BOXES else None
        if ax_stat is not None:
            ax_stat.axis('off')

        _draw_fics_panel(ax, ax_stat, pool, sampler, 'log_ics_accrete',
                         frac_labels, frac_colours, panel_name=name)

        ax.set_xlabel(r'$\log_{10}\,M_\mathrm{ICS,\,accrete}\ [\mathrm{M}_\odot]$')
        if i == 0:
            ax.set_ylabel(r'$f_\mathrm{ICS}$')
        ax.set_title(name, fontsize=13, fontweight='bold')

    if has_comp:
        ic = len(panels)
        ax = fig.add_subplot(ax_gs[ic])
        ax_stat = fig.add_subplot(stat_gs[ic]) if SHOW_STATS_BOXES else None
        if ax_stat is not None:
            ax_stat.axis('off')
        _draw_disruption_panel(ax, ax_stat, all_pools, 'log_ics_accrete',
                       _FICS_Y_CONFIGS)
        ax.set_xlabel(r'$\log_{10}\,M_\mathrm{ICS,\,accrete}\ [\mathrm{M}_\odot]$')
        ax.set_title('Disruption Comparison', fontsize=13, fontweight='bold')

    # fig.suptitle(r'$f_\mathrm{ICS}$ vs Cumulative Accreted ICS Mass',
    #              fontsize=15, fontweight='bold', y=0.98)

    outfile = os.path.join(output_dir, 'ICS_fics_vs_ics_accrete.pdf')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {outfile}')


# ── Plot 11: ICS_accrete vs ICS_disrupt ──────────────────────────────────────

def plot_ics_accrete_vs_disrupt(pool, output_dir, all_pools=None):
    """
    3 (+1 comparison) panels: Random / Uniform / Gaussian /
    Disruption Comparison.  ICS_accrete vs ICS_disrupt (both in M_sun).
    """
    _print_plot_header('ICS Accreted vs Disrupted Mass')
    panels = list(SAMPLERS.items())
    has_comp = all_pools is not None and len(all_pools) > 1
    n_panels = len(panels) + (1 if has_comp else 0)
    n_sizes = len(N_HALOES)

    fig_h = 8 if SHOW_STATS_BOXES else 5.5
    fig = plt.figure(figsize=(6 * n_panels, fig_h))
    if SHOW_STATS_BOXES:
        outer_gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.35,
                                    figure=fig)
        ax_gs = gridspec.GridSpecFromSubplotSpec(1, n_panels, subplot_spec=outer_gs[0],
                                                wspace=0.30)
        stat_gs = gridspec.GridSpecFromSubplotSpec(1, n_panels, subplot_spec=outer_gs[1],
                                                  wspace=0.30)
    else:
        outer_gs = gridspec.GridSpec(1, 1, hspace=0.35, figure=fig)
        ax_gs = gridspec.GridSpecFromSubplotSpec(1, n_panels, subplot_spec=outer_gs[0],
                                                wspace=0.30)
        stat_gs = None

    scatter_axes = []
    last_sc = None

    for i, (name, sampler) in enumerate(panels):
        ax = fig.add_subplot(ax_gs[i])
        ax_stat = fig.add_subplot(stat_gs[i]) if SHOW_STATS_BOXES else None
        if ax_stat is not None:
            ax_stat.axis('off')
        scatter_axes.append(ax)

        stat_boxes = {j: [] for j in range(n_sizes)}

        for j, n_target in enumerate(N_HALOES):
            idx = sampler(pool, n_target, seed=SEED + j)
            if len(idx) == 0:
                continue

            x_sel = pool['log_ics_disrupt'][idx]
            y_sel = pool['log_ics_accrete'][idx]
            c_sel = pool['log_mvir'][idx]

            finite = np.isfinite(x_sel) & np.isfinite(y_sel)
            xf, yf, cf = x_sel[finite], y_sel[finite], c_sel[finite]
            n_actual = len(xf)
            if n_actual < 5:
                continue

            lw = 1.0 + 1.5 * (j / max(n_sizes - 1, 1))
            alpha = 0.4 + 0.6 * (j / max(n_sizes - 1, 1))

            # Scatter colored by log10(Mvir) — only for the largest sample
            if j == n_sizes - 1:
                last_sc = ax.scatter(xf, yf, c=cf, cmap='plasma', s=8,
                                     alpha=0.5, edgecolors='none', zorder=1)

            xc, med = _median_in_bins(xf, yf)
            if len(xc) > 0:
                label = f'N={n_actual}'
                ax.plot(xc, med, color='k', lw=lw, alpha=alpha,
                        label=label, zorder=3)

            rho, pval = spearmanr(xf, yf)
            ci_txt = format_effect_ci(rho, n_actual, alpha=0.05)
            dcor_txt = format_dcor(xf, yf, seed=SEED + j)
            stat_boxes[j].append(
                r'$\rho_s$' f' = {rho:+.3f}, {ci_txt}, p = {pval:.2e}, {dcor_txt}')

        # 1:1 line
        lims = ax.get_xlim()
        ax.plot(lims, lims, 'k--', lw=0.8, alpha=0.5, label='1:1', zorder=2)
        ax.set_xlim(lims)

        ax.set_xlabel(r'$\log_{10}\,M_\mathrm{ICS,\,disrupt}\ [\mathrm{M}_\odot]$')
        if i == 0:
            ax.set_ylabel(r'$\log_{10}\,M_\mathrm{ICS,\,accrete}\ [\mathrm{M}_\odot]$')
        ax.set_title(name, fontsize=13, fontweight='bold')

        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8, loc='best')

        y_top = 0.95
        y_step = 1.0 / max(n_sizes, 1)
        for j in range(n_sizes):
            if not stat_boxes[j]:
                continue
            header = f'N = {len(sampler(pool, N_HALOES[j], seed=SEED + j))}'
            text = header + '\n' + '\n'.join(stat_boxes[j])
            if SHOW_STATS_BOXES:
                ax_stat.text(0.5, y_top - j * y_step, text,
                             transform=ax_stat.transAxes,
                             fontsize=7, ha='center', va='top',
                             bbox=dict(boxstyle='round,pad=0.3',
                                       facecolor='white', edgecolor='grey',
                                       alpha=0.9))
            _print_stats(name, text)

    if has_comp:
        ic = len(panels)
        ax = fig.add_subplot(ax_gs[ic])
        ax_stat = fig.add_subplot(stat_gs[ic]) if SHOW_STATS_BOXES else None
        if ax_stat is not None:
            ax_stat.axis('off')
        _draw_disruption_panel(
            ax, ax_stat, all_pools, 'log_ics_disrupt',
            [('log_ics_accrete', '-', r'$\log_{10}\,M_\mathrm{ICS,\,accrete}$')],
        )
        lims = ax.get_xlim()
        ax.plot(lims, lims, 'k--', lw=0.8, alpha=0.5, label='1:1')
        ax.set_xlim(lims)
        ax.set_xlabel(r'$\log_{10}\,M_\mathrm{ICS,\,disrupt}\ [\mathrm{M}_\odot]$')
        ax.set_ylabel(r'$\log_{10}\,M_\mathrm{ICS,\,accrete}\ [\mathrm{M}_\odot]$')
        ax.set_title('Disruption Comparison', fontsize=13, fontweight='bold')
        ax.legend(fontsize=7, loc='best')

    # fig.suptitle(r'ICS Accreted vs Disrupted Mass',
    #              fontsize=15, fontweight='bold', y=0.98)

    outfile = os.path.join(output_dir, 'ICS_accrete_vs_disrupt.pdf')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {outfile}')


def plot_ics_accrete_vs_disrupt_z0_colmvir(pool, output_dir, *, n_points=None):
    """Single-panel z~0 population: log10(ICS_accrete) vs log10(ICS_disrupt).

    Points are colored by log10(Mvir). Uses the full (unsampled) z=0 pool.
    Axes are hard-set to [7, 13] in both x and y.
    """
    _print_plot_header('ICS Accreted vs Disrupted Mass (z=0, coloured by Mvir)')
    if pool is None or len(pool.get('mvir', [])) == 0:
        print('  No haloes in pool; skipping z=0 accrete-vs-disrupt plot.')
        return

    if n_points is None:
        n_points = int(max(N_HALOES))

    z0 = np.nanmin(pool['z'])
    z0_mask = np.isfinite(pool['z']) & np.isclose(pool['z'], z0, rtol=0.0, atol=1e-8)
    pz0 = _filter_pool(pool, z0_mask)

    # Apply consistent filtering: drop NaNs and zeros in the underlying masses.
    # (Zeros produce NaNs in the log arrays.)
    pz0 = _filter_pool_finite_positive(
        pz0,
        finite_keys=('mvir',),
        positive_keys=('mvir', 'ics_disrupt', 'ics_accrete'),
    )

    # Uniform-in-log(Mvir) subsample (like the Uniform panels elsewhere).
    idx = sample_uniform(pz0, int(n_points), seed=SEED)
    if len(idx) == 0:
        print('  Uniform sampler returned no indices; skipping plot.')
        return

    x = pz0.get('log_ics_disrupt', np.array([]))[idx]
    y = pz0.get('log_ics_accrete', np.array([]))[idx]
    c = pz0.get('log_mvir', np.array([]))[idx]
    n = len(x)

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6.0))
    sc = ax.scatter(x, y, c=c, cmap='plasma', s=8,
                    alpha=0.6, edgecolors='none')

    ax.plot([7, 13], [7, 13], 'k--', lw=1.0, alpha=0.7, zorder=2)
    ax.set_xlim(7, 13)
    ax.set_ylim(7, 13)

    ax.set_xlabel(r'$\log_{10}\,M_\mathrm{ICS,\,disrupt}\ [\mathrm{M}_\odot]$')
    ax.set_ylabel(r'$\log_{10}\,M_\mathrm{ICS,\,accrete}\ [\mathrm{M}_\odot]$')

    ax.text(
        0.03, 0.97,
        f'z = {z0:.2f}',
        transform=ax.transAxes,
        ha='left', va='top', fontsize=11,
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                  edgecolor='0.6', alpha=0.9),
    )

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(r'$\log_{10}\,M_{\mathrm{vir}}\ [\mathrm{M}_\odot]$')

    outfile = os.path.join(output_dir, 'ICS_accrete_vs_disrupt_z0_colMvir.pdf')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {outfile}')


def plot_tasm_vs_t50_all_gc_colmvir(
    pool,
    output_dir,
    *,
    sim_params=None,
    file_list=None,
    nsat_min=1,
    n_points=None,
):
    """Single-panel t_asm vs t_50 for the full (unsampled) groups+clusters population.

    Style matches the *Uniform* panel in the multi-panel `plot_tasm_vs_t50()`:
    scatter colored by log10(Mvir) + a black median line in fixed t50 bins,
    plus a 1:1 dashed line.

    Notes
    -----
    In this script, t_50 is defined relative to the z~0 stellar mass and is
    therefore only available for the z~0 population. When `sim_params` and
    `file_list` are provided we rebuild the z~0 GC sample directly from the
    snapshot fields to match the multi-panel selection logic.
    """
    _print_plot_header('t_asm vs t_50 (all GCs, coloured by Mvir)')
    if pool is None or len(pool.get('mvir', [])) == 0:
        print('  No haloes in pool; skipping all-GC t_asm-vs-t_50 plot.')
        return

    if n_points is None:
        n_points = int(max(N_HALOES))

    # Match the multi-panel selection if possible.
    if sim_params is not None and file_list is not None:
        src_pool = _build_z0_gc_tasm_t50_pool(sim_params, file_list, nsat_min=nsat_min)
    else:
        base_mask = np.isfinite(pool.get('t_50', np.array([]))) & np.isfinite(pool.get('t_asm', np.array([])))
        src_pool = _filter_pool(pool, base_mask)

    if len(src_pool.get('t_50', [])) == 0:
        print('  Warning: no finite t_50/t_asm values; skipping all-GC t_asm vs t_50 plot.')
        return

    # Apply consistent filtering: drop NaNs and zeros before sampling.
    src_pool = _filter_pool_finite_positive(
        src_pool,
        finite_keys=('mvir', 't_50', 't_asm'),
        positive_keys=('mvir', 't_50', 't_asm'),
    )
    if len(src_pool.get('t_50', [])) == 0:
        print('  Warning: no finite positive t_50/t_asm values; skipping plot.')
        return

    # Uniform-in-log(Mvir) subsample, like the Uniform panel.
    idx = sample_uniform(src_pool, int(n_points), seed=SEED)
    if len(idx) == 0:
        print('  Uniform sampler returned no indices; skipping plot.')
        return

    x = np.asarray(src_pool['t_50'][idx], dtype=float)
    y = np.asarray(src_pool['t_asm'][idx], dtype=float)
    mvir = np.asarray(src_pool['mvir'][idx], dtype=float)
    c = np.log10(mvir)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)
    x, y, c = x[finite], y[finite], c[finite]
    n = int(len(x))
    if n == 0:
        print('  No finite t_50/t_asm/Mvir values; skipping plot.')
        return

    # Fixed binning and limits consistent with plot_tasm_vs_t50().
    t_lo, t_hi = 0.0, 13.8
    t50_edges = np.linspace(t_lo, t_hi, 13)
    t50_centres = 0.5 * (t50_edges[:-1] + t50_edges[1:])

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6.0))
    sc = ax.scatter(x, y, c=c, cmap='plasma', s=8,
                    alpha=0.5, edgecolors='none', zorder=1)

    ax.plot([t_lo, t_hi], [t_lo, t_hi], 'k--', lw=1.0, alpha=0.4, zorder=2)
    ax.set_xlim(t_lo, t_hi)
    ax.set_ylim(t_lo, t_hi)

    ax.set_xlabel(r'$t_{50}$ [Gyr lookback]')
    ax.set_ylabel(r'$t_{\mathrm{asm}}$ [Gyr lookback]')

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(r'$\log_{10}\,M_{\mathrm{vir}}\ [\mathrm{M}_\odot]$')

    outfile = os.path.join(output_dir, 'ICS_tasm_vs_t50_allGC_colMvir.pdf')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {outfile}')


# ── Plot 12: f_BCG vs lookback time ──────────────────────────────────────────

def plot_fbcg_vs_lookback(pool, output_dir, all_pools=None):
    """
    3 (+1 comparison) panels: Random / Uniform / Gaussian /
    Disruption Comparison.  f_BCG vs lookback time.
    """
    _print_plot_header('f_BCG vs Lookback Time')
    frac_labels = {
        'f_bcg_baryon': r'$M_\mathrm{BCG}/(f_b\,M_\mathrm{vir})$',
        'f_bcg_stellar': r'$M_\mathrm{BCG}/M_{\star,\mathrm{tot}}$',
    }
    frac_colours = {'f_bcg_baryon': '#0C5DA5', 'f_bcg_stellar': '#FF2C00'}

    panels = list(SAMPLERS.items())
    has_comp = all_pools is not None and len(all_pools) > 1
    n_panels = len(panels) + (1 if has_comp else 0)

    fig_h = 8 if SHOW_STATS_BOXES else 5.5
    fig = plt.figure(figsize=(6 * n_panels, fig_h))
    if SHOW_STATS_BOXES:
        outer_gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.35,
                                    figure=fig)
        ax_gs = gridspec.GridSpecFromSubplotSpec(1, n_panels, subplot_spec=outer_gs[0],
                                                wspace=0.30)
        stat_gs = gridspec.GridSpecFromSubplotSpec(1, n_panels, subplot_spec=outer_gs[1],
                                                  wspace=0.30)
    else:
        outer_gs = gridspec.GridSpec(1, 1, hspace=0.35, figure=fig)
        ax_gs = gridspec.GridSpecFromSubplotSpec(1, n_panels, subplot_spec=outer_gs[0],
                                                wspace=0.30)
        stat_gs = None

    for i, (name, sampler) in enumerate(panels):
        ax = fig.add_subplot(ax_gs[i])
        ax_stat = fig.add_subplot(stat_gs[i]) if SHOW_STATS_BOXES else None
        if ax_stat is not None:
            ax_stat.axis('off')

        _draw_fics_panel(ax, ax_stat, pool, sampler, 'lookback_time',
                         frac_labels, frac_colours,
                         baryon_key='f_bcg_baryon',
                         stellar_key='f_bcg_stellar',
                         panel_name=name)

        ax.set_xlabel(r'Lookback time [Gyr]')
        if i == 0:
            ax.set_ylabel(r'$f_\mathrm{BCG}$')
        ax.set_title(name, fontsize=13, fontweight='bold')

    if has_comp:
        ic = len(panels)
        ax = fig.add_subplot(ax_gs[ic])
        ax_stat = fig.add_subplot(stat_gs[ic]) if SHOW_STATS_BOXES else None
        if ax_stat is not None:
            ax_stat.axis('off')
        _draw_disruption_panel(ax, ax_stat, all_pools, 'lookback_time',
                       [('f_bcg_baryon', '-', r'$f_{b}$'),
                        ('f_bcg_stellar', '--', r'$f_{\star}$')])
        ax.set_xlabel(r'Lookback time [Gyr]')
        ax.set_title('Disruption Comparison', fontsize=13, fontweight='bold')

    # fig.suptitle(r'$f_\mathrm{BCG}$ vs Lookback Time by Sampling Strategy',
    #              fontsize=15, fontweight='bold', y=0.98)

    outfile = os.path.join(output_dir, 'ICS_fbcg_vs_lookback.pdf')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {outfile}')


# ── Plot 13: f_BCG + f_ICS vs lookback time ──────────────────────────────────

def plot_fbcg_fics_vs_lookback(pool, output_dir, all_pools=None):
    """
    3 (+1 comparison) panels: Random / Uniform / Gaussian /
    Disruption Comparison.  (f_BCG + f_ICS) vs lookback time.
    """
    _print_plot_header('f_BCG + f_ICS vs Lookback Time')
    frac_labels = {
        'f_bcg_ics_baryon': r'$(M_\mathrm{BCG}+M_\mathrm{ICS})/(f_b\,M_\mathrm{vir})$',
        'f_bcg_ics_stellar': r'$(M_\mathrm{BCG}+M_\mathrm{ICS})/M_{\star,\mathrm{tot}}$',
    }
    frac_colours = {'f_bcg_ics_baryon': '#0C5DA5', 'f_bcg_ics_stellar': '#FF2C00'}

    panels = list(SAMPLERS.items())
    has_comp = all_pools is not None and len(all_pools) > 1
    n_panels = len(panels) + (1 if has_comp else 0)

    fig_h = 8 if SHOW_STATS_BOXES else 5.5
    fig = plt.figure(figsize=(6 * n_panels, fig_h))
    if SHOW_STATS_BOXES:
        outer_gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.35,
                                    figure=fig)
        ax_gs = gridspec.GridSpecFromSubplotSpec(1, n_panels, subplot_spec=outer_gs[0],
                                                wspace=0.30)
        stat_gs = gridspec.GridSpecFromSubplotSpec(1, n_panels, subplot_spec=outer_gs[1],
                                                  wspace=0.30)
    else:
        outer_gs = gridspec.GridSpec(1, 1, hspace=0.35, figure=fig)
        ax_gs = gridspec.GridSpecFromSubplotSpec(1, n_panels, subplot_spec=outer_gs[0],
                                                wspace=0.30)
        stat_gs = None

    for i, (name, sampler) in enumerate(panels):
        ax = fig.add_subplot(ax_gs[i])
        ax_stat = fig.add_subplot(stat_gs[i]) if SHOW_STATS_BOXES else None
        if ax_stat is not None:
            ax_stat.axis('off')

        _draw_fics_panel(ax, ax_stat, pool, sampler, 'lookback_time',
                         frac_labels, frac_colours,
                         baryon_key='f_bcg_ics_baryon',
                         stellar_key='f_bcg_ics_stellar',
                         panel_name=name)

        ax.set_xlabel(r'Lookback time [Gyr]')
        if i == 0:
            ax.set_ylabel(r'$f_\mathrm{BCG} + f_\mathrm{ICS}$')
        ax.set_title(name, fontsize=13, fontweight='bold')

    if has_comp:
        ic = len(panels)
        ax = fig.add_subplot(ax_gs[ic])
        ax_stat = fig.add_subplot(stat_gs[ic]) if SHOW_STATS_BOXES else None
        if ax_stat is not None:
            ax_stat.axis('off')
        _draw_disruption_panel(ax, ax_stat, all_pools, 'lookback_time',
                       [('f_bcg_ics_baryon', '-', r'$f_{b}$'),
                        ('f_bcg_ics_stellar', '--', r'$f_{\star}$')])
        ax.set_xlabel(r'Lookback time [Gyr]')
        ax.set_title('Disruption Comparison', fontsize=13, fontweight='bold')

    # fig.suptitle(r'$f_\mathrm{BCG} + f_\mathrm{ICS}$ vs Lookback Time by Sampling Strategy',
    #              fontsize=15, fontweight='bold', y=0.98)

    outfile = os.path.join(output_dir, 'ICS_fbcg_fics_vs_lookback.pdf')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {outfile}')


# ── Plot 14: t_asm vs t_50 ───────────────────────────────────────────────────

def plot_tasm_vs_t50(
    pool,
    output_dir,
    all_pools=None,
    *,
    sim_params=None,
    file_list=None,
    all_model_files=None,
):
    """
    Multi-panel (Random / Uniform / Gaussian) plot of t_asm vs t_50, using
    *only* groups+clusters at z~0 (n_satellites >= 1) and no contouring.

    When sim_params+file_list are provided, this function rebuilds the z~0
    sample directly from snapshot fields (so it matches the diagnostics
    selection logic). Otherwise it falls back to the pre-built candidate pool.
    """
    _print_plot_header('t_asm vs t_50')
    # Build the source pool to sample from.
    # Preferred path: rebuild the z~0 GC sample from raw snapshot fields.
    if sim_params is not None and file_list is not None:
        src_pool = _build_z0_gc_tasm_t50_pool(sim_params, file_list, nsat_min=1)
    else:
        # Legacy fallback: use the pre-built pool and just restrict to finite t50/tasm.
        base_mask = np.isfinite(pool['t_50']) & np.isfinite(pool['t_asm'])
        src_pool = _filter_pool(pool, base_mask)

    if len(src_pool.get('t_50', [])) == 0:
        print('  Warning: no finite t_50/t_asm values; skipping t_asm vs t_50 plot.')
        return

    has_comp = all_pools is not None and len(all_pools) > 1
    n_panels = len(SAMPLERS) + (1 if has_comp else 0)
    fig_h = 8 if SHOW_STATS_BOXES else 5.5
    fig = plt.figure(figsize=(6 * n_panels, fig_h))

    if SHOW_STATS_BOXES:
        outer_gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.35,
                                    figure=fig)
        ax_gs = gridspec.GridSpecFromSubplotSpec(1, n_panels, subplot_spec=outer_gs[0],
                                                wspace=0.30)
        stat_gs = gridspec.GridSpecFromSubplotSpec(1, n_panels, subplot_spec=outer_gs[1],
                                                  wspace=0.30)
    else:
        outer_gs = gridspec.GridSpec(1, 1, hspace=0.35, figure=fig)
        ax_gs = gridspec.GridSpecFromSubplotSpec(1, n_panels, subplot_spec=outer_gs[0],
                                                wspace=0.30)
        stat_gs = None

    n_sizes = len(N_HALOES)

    # Fixed t50 binning (consistent across sampling strategies and sample sizes)
    # Match the lookback-time range used elsewhere in the diagnostics.
    t_lo, t_hi = 0.0, 13.8
    t50_edges = np.linspace(t_lo, t_hi, 13)
    t50_centres = 0.5 * (t50_edges[:-1] + t50_edges[1:])

    scatter_axes = []
    last_sc = None

    for i, (name, sampler) in enumerate(SAMPLERS.items()):
        ax = fig.add_subplot(ax_gs[i])
        ax_stat = fig.add_subplot(stat_gs[i]) if SHOW_STATS_BOXES else None
        if ax_stat is not None:
            ax_stat.axis('off')
        scatter_axes.append(ax)

        stat_boxes = {j: [] for j in range(n_sizes)}

        for j, n_target in enumerate(N_HALOES):
            idx = sampler(src_pool, n_target, seed=SEED + j)
            if len(idx) == 0:
                continue

            t50_sel = src_pool['t_50'][idx]
            tasm_sel = src_pool['t_asm'][idx]
            mvir_sel = np.log10(src_pool['mvir'][idx])

            finite = np.isfinite(t50_sel) & np.isfinite(tasm_sel)
            t50_f = t50_sel[finite]
            tasm_f = tasm_sel[finite]
            mvir_f = mvir_sel[finite]
            n_actual = len(t50_f)
            if n_actual < 5:
                continue

            lw = 1.0 + 1.5 * (j / max(n_sizes - 1, 1))
            alpha = 0.4 + 0.6 * (j / max(n_sizes - 1, 1))

            # Scatter colored by log10(Mvir) — only for the largest sample
            if j == n_sizes - 1:
                last_sc = ax.scatter(t50_f, tasm_f, c=mvir_f, cmap='plasma',
                                     s=8, alpha=0.5, edgecolors='none',
                                     zorder=1)

            # Median line in fixed t50 bins
            med = np.full_like(t50_centres, np.nan, dtype=float)
            for bi in range(len(t50_edges) - 1):
                in_bin = (t50_f >= t50_edges[bi]) & (t50_f < t50_edges[bi + 1])
                if np.sum(in_bin) >= 3:
                    med[bi] = np.median(tasm_f[in_bin])
            valid = np.isfinite(med)
            if np.any(valid):
                ax.plot(t50_centres[valid], med[valid],
                        color='k', lw=lw, alpha=alpha,
                        label=f'N={n_actual}', zorder=3)

            rho, pval = spearmanr(t50_f, tasm_f)
            ci_txt = format_effect_ci(rho, n_actual, alpha=0.05)
            dcor_txt = format_dcor(t50_f, tasm_f, seed=SEED + j)
            stat_boxes[j].append(
                r'$\rho_s$' f' = {rho:+.3f}, {ci_txt}, '
                f'p = {pval:.2e}, {dcor_txt}, N = {n_actual}')

        # 1:1 line and consistent limits
        ax.plot([t_lo, t_hi], [t_lo, t_hi], 'k--', lw=0.8, alpha=0.4, zorder=2)
        ax.set_xlim(t_lo, t_hi)
        ax.set_ylim(t_lo, t_hi)

        ax.set_xlabel(r'$t_{50}$ [Gyr lookback]')
        if i == 0:
            ax.set_ylabel(r'$t_{\mathrm{asm}}$ [Gyr lookback]')
        ax.set_title(name, fontsize=13, fontweight='bold')

        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8, loc='best')

        y_top = 0.95
        y_step = 1.0 / max(n_sizes, 1)
        for j in range(n_sizes):
            if not stat_boxes[j]:
                continue
            header = f'N = {len(sampler(src_pool, N_HALOES[j], seed=SEED + j))}'
            text = header + '\n' + '\n'.join(stat_boxes[j])
            if SHOW_STATS_BOXES:
                ax_stat.text(0.5, y_top - j * y_step, text,
                             transform=ax_stat.transAxes,
                             fontsize=7, ha='center', va='top',
                             bbox=dict(boxstyle='round,pad=0.3',
                                       facecolor='white', edgecolor='grey',
                                       alpha=0.9))
            _print_stats(name, text)

    if has_comp:
        ic = len(SAMPLERS)
        ax = fig.add_subplot(ax_gs[ic])
        ax_stat = fig.add_subplot(stat_gs[ic]) if SHOW_STATS_BOXES else None
        if ax_stat is not None:
            ax_stat.axis('off')

        # Build comparison pools using the *same* z~0 GC selection as the main panels.
        # This ensures the default model in the comparison panel matches the first
        # three panels when using the same sampler/seed.
        if all_model_files is not None and sim_params is not None:
            cmp_models = []
            for label, flist, colour in all_model_files:
                p = _build_z0_gc_tasm_t50_pool(sim_params, flist, nsat_min=1)
                if len(p.get('t_50', [])) == 0:
                    continue
                cmp_models.append((label, p, colour))

            stat_boxes = {j: [] for j in range(n_sizes)}
            for j, n_target in enumerate(N_HALOES):
                lw = 1.0 + 1.5 * (j / max(n_sizes - 1, 1))
                alpha = 0.4 + 0.6 * (j / max(n_sizes - 1, 1))

                for label, p, colour in cmp_models:
                    idx = sample_gaussian(p, n_target, seed=SEED + j)
                    if len(idx) == 0:
                        continue

                    t50_sel = p['t_50'][idx]
                    tasm_sel = p['t_asm'][idx]
                    finite = np.isfinite(t50_sel) & np.isfinite(tasm_sel)
                    t50_f = t50_sel[finite]
                    tasm_f = tasm_sel[finite]
                    n_actual = len(t50_f)
                    if n_actual < 5:
                        continue

                    med = np.full_like(t50_centres, np.nan, dtype=float)
                    for bi in range(len(t50_edges) - 1):
                        in_bin = (t50_f >= t50_edges[bi]) & (t50_f < t50_edges[bi + 1])
                        if np.sum(in_bin) >= 3:
                            med[bi] = np.median(tasm_f[in_bin])
                    valid = np.isfinite(med)
                    if np.any(valid):
                        lbl = f'{label}' if j == n_sizes - 1 else None
                        ax.plot(t50_centres[valid], med[valid],
                                color=colour, lw=lw, alpha=alpha, label=lbl)

                    rho, pval = spearmanr(t50_f, tasm_f)
                    ci_txt = format_effect_ci(rho, n_actual, alpha=0.05)
                    dcor_txt = format_dcor(t50_f, tasm_f, seed=SEED + j)
                    stat_boxes[j].append(
                        f'{label}: '
                        r'$\rho_s$' f' = {rho:+.3f}, {ci_txt}, '
                        f'p = {pval:.2e}, {dcor_txt}, N = {n_actual}')

            y_top = 0.95
            y_step = 1.0 / max(n_sizes, 1)
            for j in range(n_sizes):
                if not stat_boxes[j]:
                    continue
                header = f'N_target = {N_HALOES[j]}'
                text = header + '\n' + '\n'.join(stat_boxes[j])
                if SHOW_STATS_BOXES:
                    ax_stat.text(0.5, y_top - j * y_step, text,
                                 transform=ax_stat.transAxes,
                                 fontsize=7, ha='center', va='top',
                                 bbox=dict(boxstyle='round,pad=0.3',
                                           facecolor='white', edgecolor='grey',
                                           alpha=0.9))
                _print_stats('Disruption Comparison', text)

            handles, _ = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=7, loc='best')
        else:
            # Fallback (kept for safety): use the prebuilt candidate pools.
            all_pools_z0 = []
            for label, p, colour in all_pools:
                mk = np.isfinite(p.get('t_50', np.array([]))) & np.isfinite(p.get('t_asm', np.array([])))
                all_pools_z0.append((label, _filter_pool(p, mk), colour))
            _draw_disruption_panel(
                ax, ax_stat, all_pools_z0, 't_50',
                [('t_asm', '-', r'$t_{\mathrm{asm}}$')]
            )

        ax.plot([t_lo, t_hi], [t_lo, t_hi], 'k--', lw=0.8, alpha=0.4, zorder=0)
        ax.set_xlim(t_lo, t_hi)
        ax.set_ylim(t_lo, t_hi)
        ax.set_xlabel(r'$t_{50}$ [Gyr lookback]')
        ax.set_ylabel(r'$t_{\mathrm{asm}}$ [Gyr lookback]')
        ax.set_title('Disruption Comparison', fontsize=13, fontweight='bold')

    # fig.suptitle(r'ICS Assembly Time vs Stellar Formation Time (z=0)',
    #              fontsize=15, fontweight='bold', y=0.98)

    outfile = os.path.join(output_dir, 'ICS_tasm_vs_t50.pdf')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {outfile}')


def plot_sample_cuts_correlation_grid(pool, output_dir, n_sample=None):
    """Grid showing sample cuts (dashed) and induced correlations.

    Layout: 2 rows x 3 columns.
      - Left column: redshift sub-sample (used for f_ICS vs z)
      - Middle column: halo mass sub-sample (used for f_ICS vs Mvir)
      - Right column: relaxation sub-sample (used for f_ICS vs mass-gap)

    Each column applies its own selection cuts (shown as dashed lines where
    applicable). All candidates are plotted as grey crosses; the Gaussian-
    selected sub-sample (size ~ n_sample) is plotted as large filled circles.
    Spearman rho and p-value can be annotated per panel on the selected sample
    (enable with --stats-boxes).
    """
    if n_sample is None:
        n_sample = min(N_HALOES)

    gap_key = 'mass_gap_300' if 'mass_gap_300' in pool else 'mass_gap'
    gap_label = (r'$\Delta M_{\star,300}$'
                 if gap_key == 'mass_gap_300' else r'$\Delta M_\star$')

    # Baseline ("all candidates") shown as a single I77-style Gaussian sample
    # to reduce clutter and match the point density of the selected sub-samples.
    base_idx = sample_gaussian(pool, n_sample, seed=SEED)
    base = {
        'z': pool['z'][base_idx],
        'logm': np.log10(pool['mvir'][base_idx]),
        'gap': pool[gap_key][base_idx],
    }

    # Define three sub-samples ("Table 1" style cuts used elsewhere)
    sub_z = {
        'name': 'Redshift sub-sample',
        'pool': _filter_pool(pool, pool['mvir'] >= CLUSTER_MVIR),
        'note': r'$M_{\rm vir} \geq 10^{14}$',
    }
    sub_m = {
        'name': 'Halo mass sub-sample',
        'pool': _filter_pool(pool, pool[gap_key] < 1.2),
        'note': (r'$\Delta M_{\star,300} < 1.2$'
                 if gap_key == 'mass_gap_300' else r'$\Delta M_\star < 1.2$'),
    }
    sub_r = {
        'name': 'Relaxation sub-sample',
        'pool': _filter_pool(
            pool,
            (pool['mvir'] >= 10**13.5) & (pool['z'] > 0.16) & (pool['z'] <= Z_MAX),
        ),
        'note': r'$M_{\rm vir} \geq 10^{13.5},\ z>0.16$',
    }
    subs = [sub_z, sub_m, sub_r]

    # Style
    all_style = dict(marker='x', s=60, color='0.55', alpha=0.35, linewidths=0.8)
    sel_style = dict(marker='o', s=60, facecolor='dodgerblue', edgecolor='k',
                     alpha=0.9, linewidths=0.4)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex='col')

    def _annotate_spearman(
        ax,
        x,
        y,
        extra_note=None,
        xy=(0.02, 0.98),
        ha='left',
        va='top',
        *,
        ci_alpha=0.05,
        n_boot=500,
    ):
        finite = np.isfinite(x) & np.isfinite(y)
        xf, yf = x[finite], y[finite]

        txt = 'N < 5'
        if xf.size >= 5:
            rho, pval = spearmanr(xf, yf)

            # Bootstrap CI for Spearman rho (effect size).
            ci_txt = 'CI: n/a'
            if np.isfinite(rho) and xf.size >= 8 and n_boot > 0:
                rng = np.random.default_rng(SEED)
                n = xf.size
                rhos = []
                for _ in range(n_boot):
                    ii = rng.integers(0, n, size=n)
                    rb, _ = spearmanr(xf[ii], yf[ii])
                    if np.isfinite(rb):
                        rhos.append(rb)

                if len(rhos) >= max(50, n_boot // 5):
                    lo, hi = np.percentile(
                        rhos,
                        [100 * ci_alpha / 2, 100 * (1 - ci_alpha / 2)],
                    )
                    ci_txt = f'{int((1 - ci_alpha) * 100)}% CI=[{lo:+.3f},{hi:+.3f}]'

            txt = '\n'.join([
                r'$\rho_s$' + f'={rho:+.3f}',
                ci_txt,
                f'p={pval:.2e}',
                format_dcor(xf, yf, seed=SEED),
                f'N={xf.size}',
            ])

        if extra_note:
            txt = txt + '\n' + extra_note

        if not SHOW_STATS_BOXES:
            return

        ax.text(
            xy[0],
            xy[1],
            txt,
            transform=ax.transAxes,
            ha=ha,
            va=va,
            fontsize=10,
            bbox=dict(
                boxstyle='round,pad=0.25',
                facecolor='white',
                edgecolor='0.7',
                alpha=0.9,
            ),
        )

    def _scatter(ax, x_all, y_all, x_sel, y_sel):
        m_all = np.isfinite(x_all) & np.isfinite(y_all)
        ax.scatter(x_all[m_all], y_all[m_all], **all_style)
        if x_sel.size and y_sel.size:
            m_sel = np.isfinite(x_sel) & np.isfinite(y_sel)
            ax.scatter(x_sel[m_sel], y_sel[m_sel], **sel_style)

    # Build selected samples per column using Gaussian sampling
    sel_samples = []
    for s in subs:
        p = s['pool']
        if len(p.get('mvir', [])) == 0:
            sel_samples.append({'z': np.array([]), 'logm': np.array([]), 'gap': np.array([])})
            continue
        idx = sample_gaussian(p, n_sample, seed=SEED)
        sel_samples.append({
            'z': p['z'][idx],
            'logm': np.log10(p['mvir'][idx]),
            'gap': p[gap_key][idx],
        })

    # Column 0: z-sub-sample -> x = z
    s0 = sel_samples[0]
    _scatter(axes[0, 0], base['z'], base['logm'], s0['z'], s0['logm'])
    axes[0, 0].axhline(14.0, color='k', ls='--', lw=1.0, alpha=0.7)
    _annotate_spearman(
        axes[0, 0], s0['z'], s0['logm'], extra_note=sub_z['note'],
        xy=(0.98, 0.02), ha='right', va='bottom'
    )
    axes[0, 0].set_ylabel(r'$\log_{10}\,M_{\mathrm{vir}}\ [\mathrm{M}_\odot]$')
    axes[0, 0].set_title(sub_z['name'], fontsize=12, fontweight='bold')

    _scatter(axes[1, 0], base['z'], base['gap'], s0['z'], s0['gap'])
    _annotate_spearman(
        axes[1, 0], s0['z'], s0['gap'], extra_note=sub_z['note'],
        xy=(0.98, 0.98), ha='right', va='top'
    )
    axes[1, 0].set_xlabel('Redshift $z$')
    axes[1, 0].set_ylabel(gap_label)

    # Column 1: Mvir-sub-sample -> x = logMvir
    s1 = sel_samples[1]
    _scatter(axes[0, 1], base['logm'], base['z'], s1['logm'], s1['z'])
    _annotate_spearman(
        axes[0, 1], s1['logm'], s1['z'],
        extra_note=sub_m['note'] + '\n' + '(residual $z$ corr.)'
    )
    axes[0, 1].set_ylabel('Redshift $z$')
    axes[0, 1].set_title(sub_m['name'], fontsize=12, fontweight='bold')

    _scatter(axes[1, 1], base['logm'], base['gap'], s1['logm'], s1['gap'])
    axes[1, 1].axhline(1.2, color='k', ls='--', lw=1.0, alpha=0.7)
    _annotate_spearman(axes[1, 1], s1['logm'], s1['gap'], extra_note=sub_m['note'])
    axes[1, 1].set_xlabel(r'$\log_{10}\,M_{\mathrm{vir}}\ [\mathrm{M}_\odot]$')
    axes[1, 1].set_ylabel(gap_label)

    # Column 2: relaxation-sub-sample -> x = mass-gap
    s2 = sel_samples[2]
    _scatter(axes[0, 2], base['gap'], base['z'], s2['gap'], s2['z'])
    axes[0, 2].axhline(0.16, color='k', ls='--', lw=1.0, alpha=0.7)
    _annotate_spearman(
        axes[0, 2], s2['gap'], s2['z'], extra_note=sub_r['note'],
        xy=(0.98, 0.02), ha='right', va='bottom'
    )
    axes[0, 2].set_ylabel('Redshift $z$')
    axes[0, 2].set_title(sub_r['name'], fontsize=12, fontweight='bold')

    _scatter(axes[1, 2], base['gap'], base['logm'], s2['gap'], s2['logm'])
    axes[1, 2].axhline(13.5, color='k', ls='--', lw=1.0, alpha=0.7)
    _annotate_spearman(
        axes[1, 2], s2['gap'], s2['logm'], extra_note=sub_r['note'],
        xy=(0.98, 0.98), ha='right', va='top'
    )
    axes[1, 2].set_xlabel(gap_label)
    axes[1, 2].set_ylabel(r'$\log_{10}\,M_{\mathrm{vir}}\ [\mathrm{M}_\odot]$')

    # fig.suptitle('Sample cuts (dashed) and induced correlations',
    #              fontsize=15, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    outfile = os.path.join(output_dir, 'ICS_sample_cuts_correlation_grid.pdf')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {outfile}')


# ── Plot: f_ICS Observations vs Simulations side-by-side ─────────────────────

def plot_fics_obs_vs_sim(pool, output_dir):
    """
    Side-by-side f_ICS (both definitions) vs redshift (z=0 to 2.5).
    Left panel: observational data.  Right panel: simulation lines.
    """
    from matplotlib.lines import Line2D

    _print_plot_header('f_ICS Observations vs Simulations')

    fig, (ax_obs, ax_sim) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    max_redshift = 2.5

    # ── Observational data ───────────────────────────────────────────────
    # Cluster observations
    # Burke, Collins, Stott and Hilton 2012
    redshifts_1 = [0.9468354430379745, 0.8303797468354429, 0.7949367088607594, 0.8075949367088605, 1.2227848101265821]
    ihs_fraction_1 = np.array([1.415094339622641, 2.594339622641499, 3.7735849056603783, 1.5330188679245182, 2.3584905660377373]) / 100

    # Montes and Trujillo 2018 (gray circles)
    redshifts_2 = [0.5341772151898734, 0.5443037974683542, 0.36708860759493667, 0.39746835443037976, 0.3417721518987341,
                   0.30379746835443033, 0.04810126582278479]
    ihs_fraction_2 = np.array([1.5330188679245182, 0, 1.0613207547169807, 1.5330188679245182, 2.7122641509433976,
                               3.3018867924528266, 8.60849056603773]) / 100

    # Burke, Hilton and Collins 2015
    redshifts_3 = [0.4025316455696200, 0.3873417721518990, 0.39746835443038000, 0.33924050632911400, 0.3443037974683540,
                   0.3417721518987340, 0.2911392405063290, 0.2253164556962030, 0.2177215189873420, 0.21265822784810100,
                   0.19493670886075900, 0.17721518987341800]
    ihs_fraction_3 = np.array([2.594339622641500, 2.7122641509434000, 3.3018867924528300, 5.542452830188670, 6.014150943396220,
                               7.193396226415100, 12.971698113207500, 12.500000000000000, 16.27358490566040, 18.042452830188700,
                               16.863207547169800, 23.113207547169800]) / 100

    # Furnell et al. 2021
    redshifts_4 = [0.1443037974683540, 0.12658227848101300, 0.12151898734177200, 0.08101265822784800, 0.2253164556962030,
                   0.21518987341772100, 0.2556962025316460, 0.3063291139240510, 0.260759493670886, 0.2936708860759490, 0.3215189873417720,
                   0.3417721518987340, 0.3721518987341770, 0.3367088607594940, 0.37721518987341800, 0.3291139240506330, 0.4962025316455700,
                   0.42531645569620200, 0.10886075949367100]
    ihs_fraction_4 = np.array([38.561320754717000, 30.660377358490600, 31.014150943396200, 28.891509433962300, 26.533018867924500,
                               23.58490566037740, 28.5377358490566, 29.716981132075500, 32.54716981132080, 27.476415094339600, 27.594339622641500,
                               26.650943396226400, 19.81132075471700, 18.867924528301900, 15.448113207547200, 15.330188679245300,
                               11.320754716981100, 9.669811320754720, 31.603773584905700]) / 100

    # Feldmeier et al. 2004
    redshifts_6 = [0.16202531645569600, 0.16202531645569600, 0.16202531645569600, 0.18481012658227800]
    ihs_fraction_6 = np.array([15.212264150943400, 12.146226415094300, 10.259433962264100, 7.311320754716980]) / 100

    # Montes and Trujillo 2018 (black circles)
    redshifts_7 = [0.30126582278481000, 0.38987341772151900, 0.3417721518987340, 0.5367088607594940,
                   0.5367088607594940, 0.36962025316455700, 0.043037974683544300]
    ihs_fraction_7 = np.array([7.665094339622630, 8.60849056603773, 13.089622641509400, 6.603773584905650,
                               5.778301886792450, 4.834905660377360, 10.849056603773600]) / 100

    # Ko and Jee 2018
    redshifts_8 = [1.2379746835443037]
    ihs_fraction_8 = np.array([9.905660377358487]) / 100

    # Kluge et al. 2021
    redshifts_9 = [0.030379746835442978]
    ihs_fraction_9 = np.array([17.924528301886788]) / 100

    # Zibetti et al. 2005
    redshifts_10 = [0.24303797468354427]
    ihs_fraction_10 = np.array([10.849056603773576]) / 100

    # Presotto et al. 2014
    redshifts_11 = [0.4354430379746834]
    ihs_fraction_11 = np.array([12.264150943396224]) / 100

    # Presotto et al. 2014 (black triangle)
    redshifts_12 = [0.43291139240506316]
    ihs_fraction_12 = np.array([5.542452830188672]) / 100

    # Spavone et al. 2020 (Fornax Deep Survey)
    redshifts_13 = [0]
    ihs_fraction_13 = np.array([34.08018867924528]) / 100

    # JWST XLSSC 122 at z=1.98
    redshifts_16 = [1.98]
    ihs_fraction_16 = np.array([17.0]) / 100

    # Ragusa et al. 2023, VEGAS Antlia cluster
    redshifts_18 = [0.05]
    ihs_fraction_18 = np.array([0.35])

    # Group observations
    # Ragusa et al. 2023, VEGAS groups
    redshift_17 = [0.05] * 16
    ihs_fraction_17 = np.array([0.16, 0.05, 0.05, 0.17, 0.05, 0.27, 0.34, 0.17, 0.08, 0.35, 0.18, 0.07, 0.20, 0.22, 0.28, 0.30])

    # Ahad et al. 2025 KIDS+GAMA groups
    redshifts_19 = [0.12, 0.12, 0.12, 0.18, 0.18, 0.18, 0.24, 0.24, 0.24]
    ihs_fraction_19 = np.array([0.16, 0.10, 0.04, 0.15, 0.12, 0.08, 0.13, 0.15, 0.05])

    # ── Left panel: Observations ─────────────────────────────────────────
    # Clusters (crosses)
    for rz, fr in [(redshifts_1, ihs_fraction_1), (redshifts_2, ihs_fraction_2),
                   (redshifts_3, ihs_fraction_3), (redshifts_4, ihs_fraction_4),
                   (redshifts_6, ihs_fraction_6), (redshifts_7, ihs_fraction_7),
                   (redshifts_8, ihs_fraction_8), (redshifts_9, ihs_fraction_9),
                   (redshifts_10, ihs_fraction_10), (redshifts_11, ihs_fraction_11),
                   (redshifts_12, ihs_fraction_12), (redshifts_13, ihs_fraction_13),
                   (redshifts_18, ihs_fraction_18)]:
        ax_obs.scatter(rz, fr, marker='x', color='k', s=60, zorder=5, alpha=0.8)

    # JWST (star)
    ax_obs.scatter(redshifts_16, ihs_fraction_16, marker='*', color='yellow',
                   edgecolors='black', s=100, zorder=5)

    # Groups (plus)
    ax_obs.scatter(redshift_17, ihs_fraction_17, marker='+', color='k', s=60, zorder=5, alpha=0.8)
    ax_obs.scatter(redshifts_19, ihs_fraction_19, marker='+', color='k', s=60, zorder=5, alpha=0.8)

    obs_marker_handles = [
        Line2D([], [], marker='x', color='k', linestyle='None', markersize=7, label='Clusters'),
        Line2D([], [], marker='+', color='k', linestyle='None', markersize=7, label='Groups'),
        Line2D([], [], marker='*', color='yellow', markeredgecolor='black',
               linestyle='None', markersize=10, label='JWST XLSSC 122 (z=1.98)'),
    ]

    # ── SAGE median lines with bootstrap errors (drawn on both panels) ───
    z_vals = pool['z']
    unique_z = np.sort(np.unique(z_vals))
    n_boot = 1000
    rng = np.random.default_rng(SEED)

    sage_configs = [
        ('f_baryon', '#0C5DA5', '--',
         r'SAGE $M_\mathrm{ICS}/(f_b\,M_\mathrm{vir})$'),
        ('f_stellar', '#FF2C00', '--',
         r'SAGE $M_\mathrm{ICS}/M_{\star,\mathrm{tot}}$'),
    ]

    for fkey, color, ls, label in sage_configs:
        meds, lo, hi, z_plot = [], [], [], []
        for z in unique_z:
            mask = z_vals == z
            vals = pool[fkey][mask]
            finite = np.isfinite(vals)
            if finite.sum() < 3:
                continue
            v = vals[finite]
            z_plot.append(z)
            meds.append(np.median(v))
            boot_medians = np.array([
                np.median(rng.choice(v, size=len(v), replace=True))
                for _ in range(n_boot)])
            lo.append(np.percentile(boot_medians, 16))
            hi.append(np.percentile(boot_medians, 84))

        z_plot = np.array(z_plot)
        meds = np.array(meds)
        lo = np.array(lo)
        hi = np.array(hi)

        for ax in (ax_obs, ax_sim):
            ax.plot(z_plot, meds, color=color, ls=ls, lw=2, label=label)
            ax.fill_between(z_plot, lo, hi, color=color, alpha=0.15)

    # Extra SAGE line: clusters-only, aperture-based f_ICS within 300 kpc
    # (same definition as the red line, but using the 300 kpc aperture quantities).
    fkey = 'f_stellar_300'
    color = '#00B945'
    ls = '--'
    label = r'SAGE (clusters) $M_\mathrm{ICS}(<300\,\mathrm{kpc})/M_{\star,\mathrm{tot}}(<300\,\mathrm{kpc})$'
    meds, lo, hi, z_plot = [], [], [], []
    for z in unique_z:
        mask = (z_vals == z) & (pool['mvir'] >= CLUSTER_MVIR)
        vals = pool[fkey][mask]
        finite = np.isfinite(vals)
        if finite.sum() < 3:
            continue
        v = vals[finite]
        z_plot.append(z)
        meds.append(np.median(v))
        boot_medians = np.array([
            np.median(rng.choice(v, size=len(v), replace=True))
            for _ in range(n_boot)])
        lo.append(np.percentile(boot_medians, 16))
        hi.append(np.percentile(boot_medians, 84))

    z_plot = np.array(z_plot)
    meds = np.array(meds)
    lo = np.array(lo)
    hi = np.array(hi)

    for ax in (ax_obs, ax_sim):
        ax.plot(z_plot, meds, color=color, ls=ls, lw=2, label=label)
        ax.fill_between(z_plot, lo, hi, color=color, alpha=0.15)

    # Extra SAGE line: clusters-only, baryonic f_ICS within 300 kpc
    # f_ICS = M_ICS(<300 kpc) / (f_b M_vir)
    fkey = 'f_baryon_300'
    color = 'lightskyblue'
    ls = '--'
    label = r'SAGE (clusters) $M_\mathrm{ICS}(<300\,\mathrm{kpc})/(f_b\,M_\mathrm{vir})$'
    meds, lo, hi, z_plot = [], [], [], []
    for z in unique_z:
        mask = (z_vals == z) & (pool['mvir'] >= CLUSTER_MVIR)
        vals = pool[fkey][mask]
        finite = np.isfinite(vals)
        if finite.sum() < 3:
            continue
        v = vals[finite]
        z_plot.append(z)
        meds.append(np.median(v))
        boot_medians = np.array([
            np.median(rng.choice(v, size=len(v), replace=True))
            for _ in range(n_boot)])
        lo.append(np.percentile(boot_medians, 16))
        hi.append(np.percentile(boot_medians, 84))

    z_plot = np.array(z_plot)
    meds = np.array(meds)
    lo = np.array(lo)
    hi = np.array(hi)

    for ax in (ax_obs, ax_sim):
        ax.plot(z_plot, meds, color=color, ls=ls, lw=2, label=label)
        ax.fill_between(z_plot, lo, hi, color=color, alpha=0.15)

    # ── Right panel: Literature simulation lines ─────────────────────────
    # Rudick, Mihos and McBride 2011
    redshifts_r = [0.004011349760203840, 0.023056412555524700, 0.05248969142102060, 0.08192297028651650, 0.12001309587715800,
                   0.1581032214678000, 0.19696284454512100, 0.23774621133914200, 0.2758363369297840, 0.31392646252042500, 0.3546136421286110,
                   0.38880818669293700, 0.42681174381632700, 0.4662869648829930, 0.5043770904736340, 0.542467216064276, 0.5805573416549180,
                   0.6186474672455600, 0.6567375928362010, 0.6948277184268430, 0.7329178440174850, 0.7710079696081270, 0.8090980951987680,
                   0.84718822078941, 0.8852783463800520, 0.9233684719706940, 0.9614585975613350, 0.999548723151977, 1.0341761100525600,
                   1.0711119894131800, 1.1449837481344300, 1.183073873725070, 1.2211639993157100, 1.2592541249063500, 1.2817619263917300, 1.1083123425692700]
    ihs_fraction_r = np.array([17.035633925701400, 15.255522799283600, 13.375138908022200, 11.340768199977500, 11.708336986061200, 12.328609312577500,
                               12.848617042445400, 12.722826835652300, 11.208443436987300, 11.164335182657300, 11.105524176883900, 11.044875327180100,
                               10.871015291362500, 10.55417099775830, 10.091034327292800, 9.55438389961052, 8.833949078886400, 8.635461934401190, 9.150058234918420,
                               8.554596801462770, 7.782702350686930, 7.51070144898496, 7.701837217748510, 7.643026211975110, 7.326917055943100, 6.790266628260850,
                               6.6579418652707000, 6.7755638768175, 7.223997795839650, 6.9544473527115800, 6.15069694047515, 5.900750165938210,
                               5.966912547433290, 6.047777680371710, 6.011020801763340, 5.882352941176470]) / 100
    ax_sim.plot(redshifts_r, ihs_fraction_r, linestyle='-', color='plum', lw=2, label='Rudick et al. 2011')

    # Tang et al. 2018
    redshifts_t = [0.1021897810218980, 0.12408759124087600, 0.13625304136253000,
                   0.15571776155717800, 0.1800486618004870, 0.2068126520681270,
                   0.22871046228710500, 0.25304136253041400, 0.2700729927007300,
                   0.291970802919708, 0.3187347931873480, 0.34063260340632600,
                   0.3673965936739660, 0.3917274939172750, 0.42092457420924600,
                   0.44282238442822400, 0.45985401459854000, 0.4841849148418490]
    ihs_fraction_t = np.array([22.675736961451200, 21.995464852607700, 20.975056689342400,
                               19.954648526077100, 18.934240362811800, 18.140589569161000,
                               17.006802721088400, 16.099773242630400, 15.532879818594100,
                               14.625850340136000, 13.718820861678000, 13.151927437641700,
                               12.131519274376400, 10.997732426303800, 9.523809523809520,
                               8.73015873015872, 7.596371882086170, 6.68934240362811]) / 100
    ax_sim.plot(redshifts_t, ihs_fraction_t, linestyle='-', color='magenta', lw=2, label='Tang et al. 2018')

    # Brown et al. 2024 Horizon-AGN Simulation
    redshifts_b = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ihs_fraction_b = np.array([0.130, 0.142, 0.129, 0.168, 0.115, 0.129, 0.139, 0.112, 0.101, 0.128, 0.112])
    ax_sim.scatter(redshifts_b, ihs_fraction_b, marker='D', color='dodgerblue', edgecolors='black', s=80, label='Brown et al. 2024')

    # Yoo et al. 2026 C-EAGLE Simulation CDM
    _lb_new = np.array([
        12.305194998069327, 11.951525232423442, 11.581014049365848, 11.210502866308255,
        10.839991683250663, 10.469480500193068, 10.098969317135474, 9.728458134077883,
        9.357946951020288, 8.987435767962694, 8.616924584905101, 8.246413401847509,
        7.875902218789914, 7.5053910357323215, 7.134879852674727, 6.764368669617134,
        6.39385748655954, 6.023346303501947, 5.652835120444353, 5.28232393738676,
        4.911812754329167, 4.541301571271573, 4.17079038821398, 3.800279205156386,
        3.429768022098793, 3.0592568390411987, 2.688745655983606, 2.3182344729260134,
        1.947723289868419, 1.5772121068108262, 1.2067009237532318, 0.8361897406956391,
        0.46567855763804467, 0.11200879199216018])
    _fics_new = np.array([
        0.2396360504597388, 0.1623798269192207, 0.16653167026782223, 0.18665214188027635,
        0.19495582857747962, 0.1825002985316747, 0.16748978796365344, 0.16653167026782223,
        0.18281967109695174, 0.19751080909969598, 0.191123357794155, 0.18090343570528944,
        0.18473590648861404, 0.1872908870108304, 0.1872908870108304, 0.1872908870108304,
        0.1939977108816484, 0.19942704449135829, 0.2013432798830206, 0.20421763297051398,
        0.20389826040523695, 0.19814955423025016, 0.19719143653441895, 0.1984689267955272,
        0.2013432798830206, 0.2013432798830206, 0.20102390731774356, 0.19495582857747962,
        0.1872908870108304, 0.19240084805526325, 0.19431708344692555, 0.19335896575109435,
        0.1872908870108304, 0.18202123968375916])
    _z_grid = np.linspace(0, 20, 20000)
    _lb_grid = lookback_time_gyr(_z_grid)
    redshifts_new = np.interp(_lb_new, _lb_grid, _z_grid)
    ax_sim.plot(redshifts_new, _fics_new, linestyle='-', color='orange', lw=2, label='Yoo et al. 2026')

    # Collect SAGE line entries for a figure-level legend (below both panels).
    handles_obs, labels_obs = ax_obs.get_legend_handles_labels()
    seen_sage = set()
    sage_handles, sage_labels = [], []
    for h, l in zip(handles_obs, labels_obs):
        if not l.startswith('SAGE'):
            continue
        if l in seen_sage:
            continue
        seen_sage.add(l)
        sage_handles.append(h)
        sage_labels.append(l)

    # Sim panel legend: literature simulations only (exclude SAGE curves)
    handles_sim, labels_sim = ax_sim.get_legend_handles_labels()
    seen = set()
    unique_h, unique_l = [], []
    for h, l in zip(handles_sim, labels_sim):
        if l.startswith('SAGE'):
            continue
        if l not in seen:
            seen.add(l)
            unique_h.append(h)
            unique_l.append(l)
    ax_sim.legend(handles=unique_h, labels=unique_l, loc='upper right', fontsize=9, frameon=False)

    # Obs panel legend: markers only
    ax_obs.legend(handles=obs_marker_handles, loc='upper right',
                  fontsize=9, frameon=False)

    # ── Formatting ───────────────────────────────────────────────────────
    for ax in (ax_obs, ax_sim):
        ax.set_xlabel('Redshift $z$', fontsize=14)
        ax.set_xlim(0, max_redshift)
        ax.set_ylim(0, None)

    ax_obs.set_ylabel(r'$f_\mathrm{ICS}$', fontsize=14)
    ax_obs.set_title('Observations', fontsize=14, fontweight='bold')
    ax_sim.set_title('Simulations', fontsize=14, fontweight='bold')

    # Leave room for the SAGE legend below the x-axis labels.
    plt.tight_layout(rect=[0, 0.085, 1, 1])
    if sage_handles:
        fig.legend(
            handles=sage_handles,
            labels=sage_labels,
            loc='lower center',
            ncol=4,
            fontsize=9,
            frameon=False,
            bbox_to_anchor=(0.5, 0.055),
            borderaxespad=0.0,
        )
    outfile = os.path.join(output_dir, 'ICS_fics_obs_vs_sim.pdf')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {outfile}')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='ICS paper plots: demographics and f_ICS vs redshift '
                    'for three sampling strategies.')
    parser.add_argument('input_pattern', nargs='?',
                        default='./output/millennium/model_*.hdf5',
                        help='Glob pattern to model HDF5 files')
    parser.add_argument('-o', '--output-dir', type=str, default='./plots',
                        help='Output directory')
    parser.add_argument('--stats-boxes', action='store_true',
                        help='Draw statistics annotation text boxes on figures')
    parser.add_argument('--aperture-all', action='store_true',
                        help='Use 300 kpc aperture-based f_ICS (and mass-gap where applicable) '
                             'in all sampling panels, not just the selection panel(s)')
    args = parser.parse_args()

    global SHOW_STATS_BOXES
    SHOW_STATS_BOXES = bool(args.stats_boxes)

    file_list = sorted(glob.glob(args.input_pattern))
    if not file_list:
        print(f'Error: no files match {args.input_pattern}')
        return

    if _HAVE_COLOSSUS:
        print('NFW enclosed fractions: using COLOSSUS')
    else:
        print('NFW enclosed fractions: COLOSSUS not available (analytic fallback)')

    os.makedirs(args.output_dir, exist_ok=True)
    sim_params = read_simulation_params(file_list[0])

    pool = build_candidate_pool(sim_params, file_list)
    if pool is None:
        return

    # ── Load alternate disruption-fraction models for comparison panel ────
    input_dir = os.path.dirname(args.input_pattern)
    parent_dir = os.path.dirname(input_dir)
    filename_pat = os.path.basename(args.input_pattern)

    all_pools = [(DISRUPT_MODELS[0][0], pool, DISRUPT_MODELS[0][2])]
    all_model_files = [(DISRUPT_MODELS[0][0], file_list, DISRUPT_MODELS[0][2])]
    for label, subdir, colour in DISRUPT_MODELS[1:]:
        alt_pattern = os.path.join(parent_dir, subdir, filename_pat)
        alt_files = sorted(glob.glob(alt_pattern))
        if not alt_files:
            print(f'  Warning: no files for {label} model ({alt_pattern})')
            continue
        print(f'\nLoading {label} disruption model ...')
        alt_pool = build_candidate_pool(sim_params, alt_files)
        if alt_pool is not None:
            all_pools.append((label, alt_pool, colour))
            all_model_files.append((label, alt_files, colour))

    if len(all_pools) < 2:
        all_pools = None  # no comparison panel if only the default exists

    # ── Build full-history pools (all z) for lookback-time plots ──────
    z_max_full = max(sim_params['redshifts'])
    print(f'\nBuilding full-history pool (z_max={z_max_full:.1f}) for '
          'lookback-time plots ...')
    pool_full = build_candidate_pool(sim_params, file_list,
                                     z_max=z_max_full)
    all_pools_full = None
    if all_pools is not None:
        all_pools_full = [(all_pools[0][0], pool_full, all_pools[0][2])]
        for label, _, colour in all_pools[1:]:
            alt_files = None
            for ml, mf, mc in all_model_files:
                if ml == label:
                    alt_files = mf
                    break
            if alt_files is not None:
                alt_pool_full = build_candidate_pool(sim_params, alt_files,
                                                     z_max=z_max_full)
                if alt_pool_full is not None:
                    all_pools_full.append((label, alt_pool_full, colour))
        if len(all_pools_full) < 2:
            all_pools_full = None

    # ── Generate plots ───────────────────────────────────────────────────
    plot_demographics(pool, args.output_dir)
    plot_fics_vs_redshift(pool, args.output_dir, all_pools,
                          aperture_all=bool(args.aperture_all))
    plot_fics_vs_massgap(pool, args.output_dir, all_pools,
                         aperture_all=bool(args.aperture_all))
    plot_fics_vs_massgap_selection_panel(pool, args.output_dir,
                                         aperture_all=bool(args.aperture_all))
    plot_fics_vs_mvir(pool, args.output_dir, all_pools,
                      aperture_all=bool(args.aperture_all))
    plot_fics_vs_mvir_selection_panel(pool, args.output_dir,
                                      aperture_all=bool(args.aperture_all))
    plot_fics_vs_bcg_metallicity(pool, args.output_dir, all_pools)
    plot_ics_vs_bcg_metallicity(pool, args.output_dir, all_pools)
    plot_fics_vs_tasm(pool, args.output_dir, all_pools)
    plot_fics_vs_t50(pool, args.output_dir, all_pools)
    plot_fics_vs_ics_disrupt(pool, args.output_dir, all_pools)
    plot_fics_vs_ics_accrete(pool, args.output_dir, all_pools)
    plot_ics_accrete_vs_disrupt(pool, args.output_dir, all_pools)
    plot_ics_accrete_vs_disrupt_z0_colmvir(pool, args.output_dir)
    plot_fbcg_vs_lookback(pool_full, args.output_dir, all_pools_full)
    plot_fbcg_fics_vs_lookback(pool_full, args.output_dir, all_pools_full)
    plot_tasm_vs_t50(pool, args.output_dir, all_pools,
                     sim_params=sim_params, file_list=file_list,
                     all_model_files=all_model_files)
    plot_tasm_vs_t50_all_gc_colmvir(pool, args.output_dir,
                                   sim_params=sim_params, file_list=file_list,
                                   nsat_min=1)
    plot_sample_cuts_correlation_grid(pool, args.output_dir)
    plot_fics_obs_vs_sim(pool_full, args.output_dir)


if __name__ == '__main__':
    main()

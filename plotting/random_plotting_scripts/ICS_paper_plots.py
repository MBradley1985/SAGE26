#!/usr/bin/env python

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats as spstats
import os
import sys
import argparse
import glob
import warnings

warnings.filterwarnings('ignore')

OutputFormat = '.pdf'

_style_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'kieren_cohare_palatino_sty.mplstyle')
if os.path.exists(_style_path):
    plt.style.use(_style_path)

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'

# ── Thresholds ────────────────────────────────────────────────────────────────
MVIR_CLUSTER = 1.0e13                 # M_vir >= 10^13 for clusters
MVIR_GROUP_LO = 10**12.5              # 10^12.5 <= M_vir < 10^13 for groups
MIN_SATELLITES = 2                     # minimum satellite count
BCG_MVIR_RATIO_THRESHOLD = 10**-3.5   # pathological BCG filter
FILTER_PATHOLOGICAL_BCGS = True
MIN_N_PLOT = 10                        # min objects per snapshot to plot
MAX_N_SPEARMAN = 1500                  # subsample cap for Spearman on individuals
MAX_N_SPEARMAN_REDSHIFT = 1500         # smaller cap for redshift plot (strong trend)


# ── Cosmology ─────────────────────────────────────────────────────────────────

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


# ── Data I/O ──────────────────────────────────────────────────────────────────

def read_simulation_params(filepath):
    """Read simulation parameters from the first model HDF5 file."""
    params = {}
    with h5.File(filepath, 'r') as f:
        sim = f['Header/Simulation']
        params['Hubble_h'] = float(sim.attrs['hubble_h'])
        params['omega_matter'] = float(sim.attrs['omega_matter'])
        params['omega_lambda'] = float(sim.attrs['omega_lambda'])
        params['PartMass'] = (float(sim.attrs['particle_mass']) * 1.0e10
                              / float(sim.attrs['hubble_h']))

        runtime = f['Header/Runtime']
        params['BaryonFrac'] = float(runtime.attrs.get('BaryonFrac', 0.17))
        params['UnitTime_in_Megayears'] = 978028.5 / params['Hubble_h']

        params['redshifts'] = np.array(f['Header/snapshot_redshifts'])
        params['output_snapshots'] = np.array(f['Header/output_snapshots'])

        snap_groups = [k for k in f.keys() if k.startswith('Snap_')]
        params['available_snapshots'] = sorted(
            int(s.replace('Snap_', '')) for s in snap_groups)
        params['last_snapshot'] = (max(params['available_snapshots'])
                                   if snap_groups else 0)
    return params


def _concat_field(file_list, snap_name, field, hubble_h, is_mass=False):
    """Concatenate a single field from all HDF5 files for one snapshot."""
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
    """Load a set of fields from one snapshot across all files."""
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
    """Compute satellite count and stellar-mass sum per central."""
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
    """Return the stellar mass of the most massive satellite for each galaxy."""
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


def _compute_halo_t50(file_list, target_gids, target_mvir_z0, sim_params):
    """
    For each target central, trace Mvir backward through all snapshots and
    return (t50_gyr, z50) — the lookback time and redshift at which the halo
    first reached 50% of its z=0 virial mass.  Uses linear interpolation
    between the two bracketing snapshots.
    """
    h_ = sim_params['Hubble_h']
    redshifts = sim_params['redshifts']
    avail_snaps = sorted(sim_params['available_snapshots'])

    n_gal = len(target_gids)
    n_snap = len(avail_snaps)
    mvir_history = np.zeros((n_gal, n_snap), dtype=float)

    # Pre-sort target GIDs for fast matching
    sort_order = np.argsort(target_gids)
    sorted_tgids = target_gids[sort_order]

    print(f'  Tracing Mvir through {n_snap} snapshots for {n_gal} centrals...')
    for i_snap, snap_num in enumerate(avail_snaps):
        d = _load_snap_fields(file_list, snap_num, h_,
                              mass_fields=['Mvir'],
                              other_fields=['GalaxyIndex'])
        if d['Mvir'].size == 0:
            continue

        gids = d['GalaxyIndex']
        mvir = d['Mvir']

        ins = np.searchsorted(sorted_tgids, gids)
        ins = np.clip(ins, 0, len(sorted_tgids) - 1)
        matched = sorted_tgids[ins] == gids

        target_pos = sort_order[ins[matched]]
        mvir_history[target_pos, i_snap] = mvir[matched]

    # Lookback times and redshifts for each available snapshot
    snap_z = np.array([redshifts[s] for s in avail_snaps])
    lb_gyr = lookback_time_gyr(snap_z, h=h_,
                                om=sim_params['omega_matter'],
                                ol=sim_params['omega_lambda'])

    # Find first snapshot where Mvir >= 50% of z=0 value
    targets = 0.5 * target_mvir_z0
    met = mvir_history >= targets[:, None]
    first_idx = np.argmax(met, axis=1)
    valid = met.any(axis=1)

    t50 = np.full(n_gal, np.nan)
    z50 = np.full(n_gal, np.nan)

    vi = np.where(valid)[0]
    fs = first_idx[vi]

    # Galaxies that cross at the very first snapshot — no earlier bracket
    can_interp = fs > 0
    snap_only = vi[~can_interp]
    t50[snap_only] = lb_gyr[fs[~can_interp]]
    z50[snap_only] = snap_z[fs[~can_interp]]

    # Galaxies with a bracket: interpolate between snapshots
    idx = vi[can_interp]
    s1 = fs[can_interp]          # first snap >= target
    s0 = s1 - 1                  # last snap < target
    c0 = mvir_history[idx, s0]
    c1 = mvir_history[idx, s1]
    tgt = targets[idx]
    dc = c1 - c0
    frac_step = np.where(dc > 0, (tgt - c0) / dc, 0.0)

    t50[idx] = lb_gyr[s0] + frac_step * (lb_gyr[s1] - lb_gyr[s0])
    z50[idx] = snap_z[s0] + frac_step * (snap_z[s1] - snap_z[s0])

    n_ok = np.sum(~np.isnan(t50))
    print(f'  t50 computed for {n_ok}/{n_gal} centrals '
          f'(range: {np.nanmin(t50):.2f} – {np.nanmax(t50):.2f} Gyr)')
    return t50, z50


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def bootstrap_mean_ci(data, n_boot=1000, ci=68.27, seed=42):
    """Bootstrap 1-sigma confidence interval on the mean."""
    n = len(data)
    if n == 0:
        return np.nan, np.nan, np.nan
    if n < 3:
        m = np.mean(data)
        return m, m, m
    rng = np.random.default_rng(seed)
    boot = rng.choice(data, size=(n_boot, n), replace=True)
    means = boot.mean(axis=1)
    lo = np.percentile(means, (100 - ci) / 2)
    hi = np.percentile(means, 100 - (100 - ci) / 2)
    return np.mean(data), lo, hi


def subsampled_spearman(x, y, max_n=None, seed=42):
    """Spearman correlation, subsampling to max_n if needed."""
    if max_n is None:
        max_n = MAX_N_SPEARMAN
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) < 5:
        return np.nan, np.nan
    if len(x) > max_n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(x), max_n, replace=False)
        x, y = x[idx], y[idx]
    return spstats.spearmanr(x, y)


# ── Observational data ────────────────────────────────────────────────────────

def get_cluster_observations():
    """
    Observational f_ICS for clusters (stellar-fraction definition).
    Returns list of dicts with keys: label, z, f, marker, color.
    """
    obs = []

    obs.append({
        'label': 'Burke+12',
        'z': [0.947, 0.830, 0.795, 0.808, 1.223],
        'f': np.array([1.42, 2.59, 3.77, 1.53, 2.36]) / 100,
        'marker': 'p', 'color': '#666666',
    })
    obs.append({
        'label': r'Montes \& Trujillo 18',
        'z': [0.534, 0.544, 0.367, 0.397, 0.342, 0.304, 0.048,
              0.301, 0.390, 0.342, 0.537, 0.537, 0.370, 0.043],
        'f': np.array([1.53, 0.0, 1.06, 1.53, 2.71, 3.30, 8.61,
                       7.67, 8.61, 13.09, 6.60, 5.78, 4.83, 10.85]) / 100,
        'marker': 'o', 'color': '#777777',
    })
    obs.append({
        'label': 'Burke+15',
        'z': [0.403, 0.387, 0.397, 0.339, 0.344, 0.342, 0.291, 0.225,
              0.218, 0.213, 0.195, 0.177],
        'f': np.array([2.59, 2.71, 3.30, 5.54, 6.01, 7.19, 12.97, 12.50,
                       16.27, 18.04, 16.86, 23.11]) / 100,
        'marker': 's', 'color': '#666666',
    })
    obs.append({
        'label': 'Furnell+21',
        'z': [0.144, 0.127, 0.122, 0.081, 0.225, 0.215, 0.256, 0.306,
              0.261, 0.294, 0.322, 0.342, 0.372, 0.337, 0.377, 0.329,
              0.496, 0.425, 0.109],
        'f': np.array([38.56, 30.66, 31.01, 28.89, 26.53, 23.58, 28.54,
                       29.72, 32.55, 27.48, 27.59, 26.65, 19.81, 18.87,
                       15.45, 15.33, 11.32, 9.67, 31.60]) / 100,
        'marker': '^', 'color': '#777777',
    })
    obs.append({
        'label': 'Feldmeier+04',
        'z': [0.162, 0.162, 0.162, 0.185],
        'f': np.array([15.21, 12.15, 10.26, 7.31]) / 100,
        'marker': 'v', 'color': '#666666',
    })
    obs.append({
        'label': r'Ko \& Jee 18',
        'z': [1.238],
        'f': np.array([9.91]) / 100,
        'marker': '*', 'color': '#555555',
    })
    obs.append({
        'label': 'Kluge+21',
        'z': [0.030],
        'f': np.array([17.92]) / 100,
        'marker': 'D', 'color': '#666666',
    })
    obs.append({
        'label': 'Zibetti+05',
        'z': [0.243],
        'f': np.array([10.85]) / 100,
        'marker': 'P', 'color': '#777777',
    })
    obs.append({
        'label': 'Presotto+14',
        'z': [0.435, 0.433],
        'f': np.array([12.26, 5.54]) / 100,
        'marker': 'h', 'color': '#666666',
    })
    obs.append({
        'label': 'Spavone+20',
        'z': [0.0],
        'f': np.array([34.08]) / 100,
        'marker': '<', 'color': '#555555',
    })
    obs.append({
        'label': 'JWST XLSSC 122',
        'z': [1.98],
        'f': np.array([17.0]) / 100,
        'marker': '*', 'color': 'goldenrod',
    })
    return obs


def get_group_observations():
    """Observational f_ICS for groups (stellar-fraction definition)."""
    obs = []

    obs.append({
        'label': 'Ragusa+23',
        'z': [0.05] * 16 + [0.05],
        'f': np.array([0.16, 0.05, 0.05, 0.17, 0.05, 0.27, 0.34, 0.17,
                       0.08, 0.35, 0.18, 0.07, 0.20, 0.22, 0.28, 0.30,
                       0.35]),
        'marker': 'o', 'color': '#999999',
    })
    obs.append({
        'label': 'Ahad+25',
        'z': [0.12, 0.12, 0.12, 0.18, 0.18, 0.18, 0.24, 0.24, 0.24],
        'f': np.array([0.16, 0.10, 0.04, 0.15, 0.12, 0.08, 0.13, 0.15,
                       0.05]),
        'marker': 's', 'color': '#999999',
    })
    return obs


# ── Plot 1: f_ICS vs Redshift ────────────────────────────────────────────────

def plot_fics_vs_redshift(file_list, sim_params, output_dir,
                          max_redshift=2.5):
    """
    Both f_ICS definitions on one panel, groups and clusters, with
    bootstrap mean lines, observational compilation, and statistics
    placed below the x-axis.
    """
    print(f'\n{"="*70}')
    print(f'Plot 1: f_ICS vs redshift (z <= {max_redshift})')
    print(f'{"="*70}')

    h_ = sim_params['Hubble_h']
    redshifts = sim_params['redshifts']
    avail = sim_params['available_snapshots']
    f_b = sim_params['BaryonFrac']
    min_stellar = sim_params['PartMass'] * sim_params['BaryonFrac']

    snaps_to_plot = sorted(
        [s for s in avail
         if s < len(redshifts) and redshifts[s] <= max_redshift],
        reverse=True)
    if not snaps_to_plot:
        print('  No snapshots in range.')
        return

    # ── Collect per-snapshot statistics ────────────────────────────────────
    # Keys: 'cl_baryon', 'cl_stellar', 'gr_baryon', 'gr_stellar'
    # Each entry: list of (z, mean, lo, hi, raw_values)
    results = {k: [] for k in ['cl_baryon', 'cl_stellar',
                                'gr_baryon', 'gr_stellar']}
    # Also collect all raw (z, f) pairs for Spearman
    raw_pairs = {k: ([], []) for k in results}

    for snap in snaps_to_plot:
        z = redshifts[snap]
        d = _load_snap_fields(
            file_list, snap, h_,
            mass_fields=['Mvir', 'IntraClusterStars', 'StellarMass'],
            other_fields=['Type', 'GalaxyIndex', 'CentralGalaxyIndex'])
        if d['Mvir'].size == 0:
            continue

        Mvir = d['Mvir']
        ICS = d['IntraClusterStars']
        SM = d['StellarMass']
        Type = d['Type']
        n_sat, sm_sat = _compute_satellite_sums(
            Type, SM, d['GalaxyIndex'], d['CentralGalaxyIndex'])
        total_stellar = SM + sm_sat + ICS

        # BCG/Mvir ratio for pathological BCG filter
        bcg_mvir_ratio = np.where(Mvir > 0, SM / Mvir, 0)

        for label, key_prefix, mvir_lo, mvir_hi in [
                ('Clusters', 'cl', MVIR_CLUSTER, 1e17),
                ('Groups', 'gr', MVIR_GROUP_LO, MVIR_CLUSTER)]:

            base = ((Type == 0) & (Mvir >= mvir_lo) & (Mvir < mvir_hi)
                    & (ICS > 0) & (SM >= min_stellar)
                    & (n_sat >= MIN_SATELLITES) & (total_stellar > 0))
            if FILTER_PATHOLOGICAL_BCGS:
                base = base & (bcg_mvir_ratio >= BCG_MVIR_RATIO_THRESHOLD)
            sel = np.where(base)[0]
            if len(sel) == 0:
                continue

            # Baryon-budget definition
            fb_vals = ICS[sel] / (f_b * Mvir[sel])
            mean_fb, lo_fb, hi_fb = bootstrap_mean_ci(fb_vals)
            results[f'{key_prefix}_baryon'].append(
                (z, mean_fb, lo_fb, hi_fb, fb_vals))
            raw_pairs[f'{key_prefix}_baryon'][0].extend([z] * len(fb_vals))
            raw_pairs[f'{key_prefix}_baryon'][1].extend(fb_vals)

            # Stellar-fraction definition
            fs_vals = ICS[sel] / total_stellar[sel]
            mean_fs, lo_fs, hi_fs = bootstrap_mean_ci(fs_vals)
            results[f'{key_prefix}_stellar'].append(
                (z, mean_fs, lo_fs, hi_fs, fs_vals))
            raw_pairs[f'{key_prefix}_stellar'][0].extend([z] * len(fs_vals))
            raw_pairs[f'{key_prefix}_stellar'][1].extend(fs_vals)

    # Check we have data
    has_data = any(len(v) > 0 for v in results.values())
    if not has_data:
        print('  No data to plot.')
        return

    # ── Compute summary statistics ────────────────────────────────────────
    stat_lines = []  # lines of text for the stats panel

    curve_meta = {
        'cl_baryon': ('Clusters, baryon-budget', '-', 'C0'),
        'cl_stellar': ('Clusters, stellar-fraction', '-', 'C3'),
        'gr_baryon': ('Groups, baryon-budget', '--', 'C0'),
        'gr_stellar': ('Groups, stellar-fraction', '--', 'C3'),
    }

    # Per-snapshot N diagnostic
    for key in ['cl_stellar', 'gr_stellar']:
        nice = curve_meta[key][0]
        for entry in sorted(results[key], key=lambda t: t[0]):
            n = len(entry[4])
            flag = ' *' if n < MIN_N_PLOT else ''
            print(f'    {nice}: z={entry[0]:.2f}  N={n}{flag}')

    for key in ['cl_baryon', 'cl_stellar', 'gr_baryon', 'gr_stellar']:
        if not results[key]:
            continue
        nice_name = curve_meta[key][0]

        # z=0 stats (closest snapshot to z=0)
        z0_entry = min(results[key], key=lambda t: t[0])
        z0_vals = z0_entry[4]
        z0_z = z0_entry[0]
        n0 = len(z0_vals)

        # Spearman on individual galaxies, subsampled to MAX_N_SPEARMAN_REDSHIFT
        rp_z = np.array(raw_pairs[key][0])
        rp_f = np.array(raw_pairs[key][1])
        rho, pval = subsampled_spearman(rp_z, rp_f, max_n=MAX_N_SPEARMAN_REDSHIFT)

        line = (f'{nice_name} (z={z0_z:.2f}): '
                f'N={n0}, '
                f'mean={np.mean(z0_vals):.4f}, '
                f'med={np.median(z0_vals):.4f}, '
                f'std={np.std(z0_vals):.4f}, '
                f'min={np.min(z0_vals):.4f}, '
                f'max={np.max(z0_vals):.4f}; '
                f'Spearman(z): rho={rho:+.3f}, p={pval:.2e}')
        stat_lines.append(line)
        print(f'  {line}')

    # ── Spearman p-value vs redshift range diagnostic ─────────────────────
    print(f'\n  Spearman p-value by cumulative redshift range '
          f'(subsampled N={MAX_N_SPEARMAN_REDSHIFT}):')
    z_upper_limits = np.arange(0.5, max_redshift + 0.01, 0.5)
    for key in ['cl_stellar', 'cl_baryon', 'gr_stellar', 'gr_baryon']:
        if not raw_pairs[key][0]:
            continue
        nice_name = curve_meta[key][0]
        rp_z = np.array(raw_pairs[key][0])
        rp_f = np.array(raw_pairs[key][1])
        parts = []
        for z_max in z_upper_limits:
            mask = rp_z <= z_max
            n_in = int(np.sum(mask))
            if n_in >= 10:
                rho, pval = subsampled_spearman(
                    rp_z[mask], rp_f[mask],
                    max_n=MAX_N_SPEARMAN_REDSHIFT)
                parts.append(f'z<={z_max:.1f}: rho={rho:+.3f}, '
                             f'p={pval:.2e}, N={n_in}')
            else:
                parts.append(f'z<={z_max:.1f}: N={n_in} (too few)')
        print(f'    {nice_name}:')
        for p in parts:
            print(f'      {p}')

    # ── Figure ────────────────────────────────────────────────────────────
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(12, 10.5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1.1], hspace=0.30)
    ax = fig.add_subplot(gs[0])
    ax_leg = fig.add_subplot(gs[1])
    ax_leg.axis('off')

    # Color scheme: blue for baryon-budget, red for stellar-fraction
    colors = {
        'cl_baryon':  '#0C5DA5',   # blue
        'gr_baryon':  '#0C5DA5',
        'cl_stellar': '#FF2C00',   # red
        'gr_stellar': '#FF2C00',
    }
    linestyles = {
        'cl_baryon': '-', 'cl_stellar': '-',
        'gr_baryon': '--', 'gr_stellar': '--',
    }
    labels = {
        'cl_baryon':  r'Clusters: $M_\mathrm{ICS}/(f_b M_\mathrm{vir})$',
        'cl_stellar': r'Clusters: $M_\mathrm{ICS}/M_{\star,\mathrm{tot}}$',
        'gr_baryon':  r'Groups: $M_\mathrm{ICS}/(f_b M_\mathrm{vir})$',
        'gr_stellar': r'Groups: $M_\mathrm{ICS}/M_{\star,\mathrm{tot}}$',
    }
    fill_alpha = {'cl_baryon': 0.15, 'cl_stellar': 0.15,
                  'gr_baryon': 0.10, 'gr_stellar': 0.10}

    # Plot SAGE mean lines with bootstrap CI (skip snapshots with N < MIN_N_PLOT)
    for key in ['cl_baryon', 'cl_stellar', 'gr_baryon', 'gr_stellar']:
        if not results[key]:
            continue
        # Filter to snapshots with enough objects
        plot_entries = [t for t in results[key] if len(t[4]) >= MIN_N_PLOT]
        if not plot_entries:
            continue
        zz = np.array([t[0] for t in plot_entries])
        mm = np.array([t[1] for t in plot_entries])
        lo = np.array([t[2] for t in plot_entries])
        hi = np.array([t[3] for t in plot_entries])
        order = np.argsort(zz)
        zz, mm, lo, hi = zz[order], mm[order], lo[order], hi[order]

        ax.plot(zz, mm, linestyle=linestyles[key], color=colors[key],
                lw=2.5)
        ax.fill_between(zz, lo, hi, color=colors[key],
                        alpha=fill_alpha[key])

    # Plot observations (stellar-fraction definition) — consistent
    # markers: 'x' for cluster obs, '+' for group obs
    cluster_obs = get_cluster_observations()
    group_obs = get_group_observations()

    all_cl_z, all_cl_f = [], []
    for ob in cluster_obs:
        all_cl_z.extend(ob['z'])
        all_cl_f.extend(ob['f'])
    all_gr_z, all_gr_f = [], []
    for ob in group_obs:
        all_gr_z.extend(ob['z'])
        all_gr_f.extend(ob['f'])

    if all_cl_z:
        ax.scatter(all_cl_z, all_cl_f, marker='x', color='#555555',
                   s=45, alpha=0.7, zorder=5, linewidths=1.0)
    if all_gr_z:
        ax.scatter(all_gr_z, all_gr_f, marker='+', color='#888888',
                   s=55, alpha=0.7, zorder=5, linewidths=1.2)

    ax.set_xlim(0, max_redshift)
    ax.set_ylim(0, None)
    ax.set_xlabel(r'Redshift $z$')
    ax.set_ylabel(r'$f_\mathrm{ICS}$')

    # ── Legend + references + stats in lower panel ────────────────────────
    handles = []
    for key in ['cl_baryon', 'cl_stellar', 'gr_baryon', 'gr_stellar']:
        if results[key]:
            handles.append(Line2D(
                [0], [0], color=colors[key], ls=linestyles[key],
                lw=2.5, label=labels[key]))
    handles.append(Line2D(
        [0], [0], marker='x', color='#555555', ls='none', ms=7,
        markeredgewidth=1.0,
        label=r'Cluster obs.\ (stellar frac.)'))
    handles.append(Line2D(
        [0], [0], marker='+', color='#888888', ls='none', ms=8,
        markeredgewidth=1.2,
        label=r'Group obs.\ (stellar frac.)'))

    ax_leg.legend(handles=handles, loc='upper center',
                  bbox_to_anchor=(0.5, 1.02),
                  ncol=3, fontsize=11, frameon=False,
                  handlelength=2.5, columnspacing=1.5,
                  handletextpad=0.8)

    # Statistics table (N, mean, std, Spearman rho, p-value)
    col_labels = ['N', 'Mean', r'$\sigma$', r'$\rho_s(z)$', r'$p$-value']
    row_labels = []
    table_data = []
    row_colors = []
    color_map = {
        'cl_baryon': '#d6e5f3', 'cl_stellar': '#fcd5cc',
        'gr_baryon': '#e8eff7', 'gr_stellar': '#fde8e3',
    }

    for key in ['cl_stellar', 'cl_baryon', 'gr_stellar', 'gr_baryon']:
        if not results[key]:
            continue
        nice = curve_meta[key][0]
        z0_entry = min(results[key], key=lambda t: t[0])
        v = z0_entry[4]
        rp_z = np.array(raw_pairs[key][0])
        rp_f = np.array(raw_pairs[key][1])
        rho, pval = subsampled_spearman(rp_z, rp_f, max_n=MAX_N_SPEARMAN_REDSHIFT)
        row_labels.append(nice)
        table_data.append([
            f'{len(v)}',
            f'{np.mean(v):.4f}',
            f'{np.std(v):.4f}',
            f'{rho:+.3f}',
            f'{pval:.1e}',
        ])
        row_colors.append(color_map[key])

    # z<=0.5 restricted Spearman rows
    for key in ['cl_stellar', 'cl_baryon', 'gr_stellar', 'gr_baryon']:
        if not results[key]:
            continue
        nice = curve_meta[key][0]
        rp_z = np.array(raw_pairs[key][0])
        rp_f = np.array(raw_pairs[key][1])
        mask05 = rp_z <= 0.5
        n05 = int(np.sum(mask05))
        if n05 >= 10:
            rho05, pval05 = subsampled_spearman(
                rp_z[mask05], rp_f[mask05],
                max_n=MAX_N_SPEARMAN_REDSHIFT)
            row_labels.append(f'{nice} ($z\\leq0.5$)')
            table_data.append([
                f'{n05}',
                '--', '--',
                f'{rho05:+.3f}',
                f'{pval05:.1e}',
            ])
            row_colors.append('#f5f5dc')

    if table_data:
        tbl = ax_leg.table(
            cellText=table_data,
            rowLabels=row_labels,
            colLabels=col_labels,
            loc='center',
            bbox=[0.10, -0.15, 0.80, 0.75],
            cellLoc='center',
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)

        # Style the table
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor('#cccccc')
            cell.set_linewidth(0.5)
            if r == 0:
                cell.set_facecolor('#e0e0e0')
                cell.set_text_props(weight='bold', fontsize=10)
            elif r > 0 and c == -1:
                cell.set_text_props(fontsize=9, ha='right')
                cell.set_facecolor(row_colors[r - 1])
            elif r > 0:
                cell.set_facecolor(row_colors[r - 1])

    outfile = os.path.join(output_dir,
                           f'P1_fICS_vs_redshift{OutputFormat}')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


# ── Observational data: f_ICS vs Mvir ─────────────────────────────────────────

def get_fics_vs_mvir_observations():
    """
    Observational f_ICS (stellar-fraction) vs log10(Mvir) at z~0.
    Compilation following Contini (2021).
    Returns list of dicts with keys: label, logMvir, f, f_err, marker, color.
    """
    obs = []
    obs.append({
        'label': 'Zibetti+05',
        'logMvir': [14.30], 'f': [0.110], 'f_err': [0.030],
        'marker': 'o', 'color': 'firebrick',
    })
    obs.append({
        'label': r'Krick \& Bernstein 07',
        'logMvir': [14.20], 'f': [0.120], 'f_err': [0.050],
        'marker': 's', 'color': 'darkorange',
    })
    obs.append({
        'label': 'Presotto+14',
        'logMvir': [15.10], 'f': [0.120], 'f_err': [0.020],
        'marker': 'v', 'color': 'seagreen',
    })
    obs.append({
        'label': 'Burke+15',
        'logMvir': [14.90], 'f': [0.030], 'f_err': [0.010],
        'marker': '^', 'color': 'navy',
    })
    obs.append({
        'label': r'Montes \& Trujillo 18',
        'logMvir': [14.95], 'f': [0.150], 'f_err': [0.070],
        'marker': 'D', 'color': 'purple',
    })
    obs.append({
        'label': 'Furnell+21',
        'logMvir': [14.70], 'f': [0.240], 'f_err': [0.090],
        'marker': 'P', 'color': 'teal',
    })
    obs.append({
        'label': 'Kluge+21',
        'logMvir': [14.50], 'f': [0.300], 'f_err': [0.100],
        'marker': 'X', 'color': 'saddlebrown',
    })
    return obs


# ── Plot 2: f_ICS vs Mvir at z=0 ─────────────────────────────────────────────

def plot_fics_vs_mvir(file_list, sim_params, output_dir):
    """
    Both f_ICS definitions vs log10(Mvir) at z=0, binned mean lines
    with bootstrap CI, Contini (2021) observations, and statistics table.
    """
    print(f'\n{"="*70}')
    print('Plot 2: f_ICS vs Mvir at z=0')
    print(f'{"="*70}')

    h_ = sim_params['Hubble_h']
    redshifts = sim_params['redshifts']
    f_b = sim_params['BaryonFrac']
    min_stellar = sim_params['PartMass'] * sim_params['BaryonFrac']
    last_snap = sim_params['last_snapshot']
    z_snap = redshifts[last_snap]

    d = _load_snap_fields(
        file_list, last_snap, h_,
        mass_fields=['Mvir', 'IntraClusterStars', 'StellarMass'],
        other_fields=['Type', 'GalaxyIndex', 'CentralGalaxyIndex'])
    if d['Mvir'].size == 0:
        print('  No data at z=0.')
        return

    Mvir = d['Mvir']
    ICS = d['IntraClusterStars']
    SM = d['StellarMass']
    Type = d['Type']
    n_sat, sm_sat = _compute_satellite_sums(
        Type, SM, d['GalaxyIndex'], d['CentralGalaxyIndex'])
    total_stellar = SM + sm_sat + ICS

    bcg_mvir_ratio = np.where(Mvir > 0, SM / Mvir, 0)

    base = ((Type == 0) & (Mvir >= MVIR_GROUP_LO) & (ICS > 0)
            & (SM >= min_stellar) & (n_sat >= MIN_SATELLITES)
            & (total_stellar > 0))
    if FILTER_PATHOLOGICAL_BCGS:
        base = base & (bcg_mvir_ratio >= BCG_MVIR_RATIO_THRESHOLD)
    sel = np.where(base)[0]

    if len(sel) == 0:
        print('  No qualifying centrals.')
        return

    logMvir = np.log10(Mvir[sel])
    f_baryon = ICS[sel] / (f_b * Mvir[sel])
    f_stellar = ICS[sel] / total_stellar[sel]

    # ── Binned statistics ─────────────────────────────────────────────────
    is_cl = logMvir >= np.log10(MVIR_CLUSTER)
    is_gr = (logMvir >= np.log10(MVIR_GROUP_LO)) & (logMvir < np.log10(MVIR_CLUSTER))

    bin_edges = np.arange(
        np.floor(logMvir.min() * 4) / 4,
        np.ceil(logMvir.max() * 4) / 4 + 0.25, 0.25)
    bin_cen = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    def _binned_bootstrap(vals, logM, edges):
        means = np.full(len(edges) - 1, np.nan)
        lo = np.full(len(edges) - 1, np.nan)
        hi = np.full(len(edges) - 1, np.nan)
        counts = np.zeros(len(edges) - 1, dtype=int)
        for i in range(len(edges) - 1):
            m = (logM >= edges[i]) & (logM < edges[i + 1])
            counts[i] = int(np.sum(m))
            if counts[i] >= MIN_N_PLOT:
                means[i], lo[i], hi[i] = bootstrap_mean_ci(vals[m])
        return means, lo, hi, counts

    bar_mean, bar_lo, bar_hi, _ = _binned_bootstrap(
        f_baryon, logMvir, bin_edges)
    ste_mean, ste_lo, ste_hi, _ = _binned_bootstrap(
        f_stellar, logMvir, bin_edges)

    # ── Summary statistics ────────────────────────────────────────────────
    curve_meta = {
        'cl_stellar': 'Clusters, stellar-fraction',
        'cl_baryon':  'Clusters, baryon-budget',
        'gr_stellar': 'Groups, stellar-fraction',
        'gr_baryon':  'Groups, baryon-budget',
    }
    stat_data = {}
    for key, mask, vals in [
            ('cl_stellar', is_cl, f_stellar), ('cl_baryon', is_cl, f_baryon),
            ('gr_stellar', is_gr, f_stellar), ('gr_baryon', is_gr, f_baryon)]:
        v = vals[mask]
        lm = logMvir[mask]
        rho, pval = subsampled_spearman(lm, v)
        stat_data[key] = {
            'n': len(v),
            'mean': np.mean(v) if len(v) else np.nan,
            'std': np.std(v) if len(v) else np.nan,
            'rho': rho, 'pval': pval,
        }
        print(f'  {curve_meta[key]}: N={len(v)}, '
              f'mean={np.mean(v):.4f}, std={np.std(v):.4f}, '
              f'Spearman(Mvir): rho={rho:+.3f}, p={pval:.2e}')

    # ── Figure ────────────────────────────────────────────────────────────
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(12, 10.5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1.1], hspace=0.30)
    ax = fig.add_subplot(gs[0])
    ax_leg = fig.add_subplot(gs[1])
    ax_leg.axis('off')

    col_bar = '#0C5DA5'
    col_ste = '#FF2C00'

    # Baryon-budget
    v = ~np.isnan(bar_mean)
    if v.any():
        ax.plot(bin_cen[v], bar_mean[v], '-', color=col_bar, lw=2.5)
        ax.fill_between(bin_cen[v], bar_lo[v], bar_hi[v],
                        color=col_bar, alpha=0.15)

    # Stellar-fraction
    v = ~np.isnan(ste_mean)
    if v.any():
        ax.plot(bin_cen[v], ste_mean[v], '-', color=col_ste, lw=2.5)
        ax.fill_between(bin_cen[v], ste_lo[v], ste_hi[v],
                        color=col_ste, alpha=0.15)

    # Group / cluster boundary
    ax.axvline(np.log10(MVIR_CLUSTER), color='grey', ls=':', lw=1.0,
               alpha=0.6)
    ax.text(np.log10(MVIR_CLUSTER) - 0.05, ax.get_ylim()[1] * 0.95,
            'groups', ha='right', va='top', fontsize=10, color='grey')
    ax.text(np.log10(MVIR_CLUSTER) + 0.05, ax.get_ylim()[1] * 0.95,
            'clusters', ha='left', va='top', fontsize=10, color='grey')

    ax.set_xlabel(
        r'$\log_{10}\, M_{\mathrm{vir}}\ [\mathrm{M}_{\odot}]$')
    ax.set_ylabel(r'$f_\mathrm{ICS}$')
    ax.set_ylim(0, None)

    # ── Legend + stats in lower panel ─────────────────────────────────────
    handles = [
        Line2D([0], [0], color=col_bar, lw=2.5,
               label=r'$M_\mathrm{ICS}/(f_b M_\mathrm{vir})$'),
        Line2D([0], [0], color=col_ste, lw=2.5,
               label=r'$M_\mathrm{ICS}/M_{\star,\mathrm{tot}}$'),
    ]

    ax_leg.legend(handles=handles, loc='upper center',
                  bbox_to_anchor=(0.5, 1.02),
                  ncol=2, fontsize=11, frameon=False,
                  handlelength=2.5, columnspacing=2.0,
                  handletextpad=0.8)

    # Statistics table
    col_labels = ['N', 'Mean', r'$\sigma$',
                  r'$\rho_s(M_\mathrm{vir})$', r'$p$-value']
    row_labels = []
    table_data = []
    row_colors = []
    color_map = {
        'cl_baryon': '#d6e5f3', 'cl_stellar': '#fcd5cc',
        'gr_baryon': '#e8eff7', 'gr_stellar': '#fde8e3',
    }

    for key in ['cl_stellar', 'cl_baryon', 'gr_stellar', 'gr_baryon']:
        sd = stat_data[key]
        row_labels.append(curve_meta[key])
        table_data.append([
            f'{sd["n"]}',
            f'{sd["mean"]:.4f}',
            f'{sd["std"]:.4f}',
            f'{sd["rho"]:+.3f}',
            f'{sd["pval"]:.1e}',
        ])
        row_colors.append(color_map[key])

    tbl = ax_leg.table(
        cellText=table_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc='center',
        bbox=[0.15, 0.05, 0.70, 0.55],
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#cccccc')
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor('#e0e0e0')
            cell.set_text_props(weight='bold', fontsize=10)
        elif r > 0 and c == -1:
            cell.set_text_props(fontsize=9, ha='right')
            cell.set_facecolor(row_colors[r - 1])
        elif r > 0:
            cell.set_facecolor(row_colors[r - 1])

    outfile = os.path.join(output_dir,
                           f'P2_fICS_vs_Mvir{OutputFormat}')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


# ── Plot 3: f_ICS vs BCG stellar mass at z=0 ─────────────────────────────────

def plot_fics_vs_bcg_mass(file_list, sim_params, output_dir):
    """
    Both f_ICS definitions vs log10(M_BCG) at z=0, binned mean lines
    with bootstrap CI, and statistics table below.
    """
    print(f'\n{"="*70}')
    print('Plot 3: f_ICS vs BCG stellar mass at z=0')
    print(f'{"="*70}')

    h_ = sim_params['Hubble_h']
    redshifts = sim_params['redshifts']
    f_b = sim_params['BaryonFrac']
    min_stellar = sim_params['PartMass'] * sim_params['BaryonFrac']
    last_snap = sim_params['last_snapshot']
    z_snap = redshifts[last_snap]

    d = _load_snap_fields(
        file_list, last_snap, h_,
        mass_fields=['Mvir', 'IntraClusterStars', 'StellarMass'],
        other_fields=['Type', 'GalaxyIndex', 'CentralGalaxyIndex'])
    if d['Mvir'].size == 0:
        print('  No data at z=0.')
        return

    Mvir = d['Mvir']
    ICS = d['IntraClusterStars']
    SM = d['StellarMass']
    Type = d['Type']
    n_sat, sm_sat = _compute_satellite_sums(
        Type, SM, d['GalaxyIndex'], d['CentralGalaxyIndex'])
    total_stellar = SM + sm_sat + ICS

    bcg_mvir_ratio = np.where(Mvir > 0, SM / Mvir, 0)

    base = ((Type == 0) & (Mvir >= MVIR_GROUP_LO) & (ICS > 0)
            & (SM >= min_stellar) & (n_sat >= MIN_SATELLITES)
            & (total_stellar > 0))
    if FILTER_PATHOLOGICAL_BCGS:
        base = base & (bcg_mvir_ratio >= BCG_MVIR_RATIO_THRESHOLD)
    sel = np.where(base)[0]

    if len(sel) == 0:
        print('  No qualifying centrals.')
        return

    logMbcg = np.log10(SM[sel])
    logMvir = np.log10(Mvir[sel])
    f_baryon = ICS[sel] / (f_b * Mvir[sel])
    f_stellar = ICS[sel] / total_stellar[sel]

    # ── Binned statistics ─────────────────────────────────────────────────
    is_cl = logMvir >= np.log10(MVIR_CLUSTER)
    is_gr = (logMvir >= np.log10(MVIR_GROUP_LO)) & (logMvir < np.log10(MVIR_CLUSTER))

    bin_edges = np.arange(
        np.floor(logMbcg.min() * 4) / 4,
        np.ceil(logMbcg.max() * 4) / 4 + 0.25, 0.25)
    bin_cen = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    def _binned_bootstrap(vals, logx, edges):
        means = np.full(len(edges) - 1, np.nan)
        lo = np.full(len(edges) - 1, np.nan)
        hi = np.full(len(edges) - 1, np.nan)
        counts = np.zeros(len(edges) - 1, dtype=int)
        for i in range(len(edges) - 1):
            m = (logx >= edges[i]) & (logx < edges[i + 1])
            counts[i] = int(np.sum(m))
            if counts[i] >= MIN_N_PLOT:
                means[i], lo[i], hi[i] = bootstrap_mean_ci(vals[m])
        return means, lo, hi, counts

    bar_mean, bar_lo, bar_hi, _ = _binned_bootstrap(
        f_baryon, logMbcg, bin_edges)
    ste_mean, ste_lo, ste_hi, _ = _binned_bootstrap(
        f_stellar, logMbcg, bin_edges)

    # ── Summary statistics ────────────────────────────────────────────────
    curve_meta = {
        'cl_stellar': 'Clusters, stellar-fraction',
        'cl_baryon':  'Clusters, baryon-budget',
        'gr_stellar': 'Groups, stellar-fraction',
        'gr_baryon':  'Groups, baryon-budget',
    }
    stat_data = {}
    for key, mask, vals in [
            ('cl_stellar', is_cl, f_stellar), ('cl_baryon', is_cl, f_baryon),
            ('gr_stellar', is_gr, f_stellar), ('gr_baryon', is_gr, f_baryon)]:
        v = vals[mask]
        lm = logMbcg[mask]
        rho, pval = subsampled_spearman(lm, v)
        stat_data[key] = {
            'n': len(v),
            'mean': np.mean(v) if len(v) else np.nan,
            'std': np.std(v) if len(v) else np.nan,
            'rho': rho, 'pval': pval,
        }
        print(f'  {curve_meta[key]}: N={len(v)}, '
              f'mean={np.mean(v):.4f}, std={np.std(v):.4f}, '
              f'Spearman(M_BCG): rho={rho:+.3f}, p={pval:.2e}')

    # ── Figure ────────────────────────────────────────────────────────────
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(12, 10.5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1.1], hspace=0.30)
    ax = fig.add_subplot(gs[0])
    ax_leg = fig.add_subplot(gs[1])
    ax_leg.axis('off')

    col_bar = '#0C5DA5'
    col_ste = '#FF2C00'

    # Baryon-budget
    v = ~np.isnan(bar_mean)
    if v.any():
        ax.plot(bin_cen[v], bar_mean[v], '-', color=col_bar, lw=2.5)
        ax.fill_between(bin_cen[v], bar_lo[v], bar_hi[v],
                        color=col_bar, alpha=0.15)

    # Stellar-fraction
    v = ~np.isnan(ste_mean)
    if v.any():
        ax.plot(bin_cen[v], ste_mean[v], '-', color=col_ste, lw=2.5)
        ax.fill_between(bin_cen[v], ste_lo[v], ste_hi[v],
                        color=col_ste, alpha=0.15)

    ax.set_xlabel(
        r'$\log_{10}\, M_{\star,\mathrm{BCG}}\ [\mathrm{M}_{\odot}]$')
    ax.set_ylabel(r'$f_\mathrm{ICS}$')
    ax.set_ylim(0, None)

    # ── Legend + stats in lower panel ─────────────────────────────────────
    handles = [
        Line2D([0], [0], color=col_bar, lw=2.5,
               label=r'$M_\mathrm{ICS}/(f_b M_\mathrm{vir})$'),
        Line2D([0], [0], color=col_ste, lw=2.5,
               label=r'$M_\mathrm{ICS}/M_{\star,\mathrm{tot}}$'),
    ]

    ax_leg.legend(handles=handles, loc='upper center',
                  bbox_to_anchor=(0.5, 1.02),
                  ncol=2, fontsize=11, frameon=False,
                  handlelength=2.5, columnspacing=2.0,
                  handletextpad=0.8)

    # Statistics table
    col_labels = ['N', 'Mean', r'$\sigma$',
                  r'$\rho_s(M_\mathrm{BCG})$', r'$p$-value']
    row_labels = []
    table_data = []
    row_colors = []
    color_map = {
        'cl_baryon': '#d6e5f3', 'cl_stellar': '#fcd5cc',
        'gr_baryon': '#e8eff7', 'gr_stellar': '#fde8e3',
    }

    for key in ['cl_stellar', 'cl_baryon', 'gr_stellar', 'gr_baryon']:
        sd = stat_data[key]
        row_labels.append(curve_meta[key])
        table_data.append([
            f'{sd["n"]}',
            f'{sd["mean"]:.4f}',
            f'{sd["std"]:.4f}',
            f'{sd["rho"]:+.3f}',
            f'{sd["pval"]:.1e}',
        ])
        row_colors.append(color_map[key])

    tbl = ax_leg.table(
        cellText=table_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc='center',
        bbox=[0.15, 0.05, 0.70, 0.55],
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#cccccc')
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor('#e0e0e0')
            cell.set_text_props(weight='bold', fontsize=10)
        elif r > 0 and c == -1:
            cell.set_text_props(fontsize=9, ha='right')
            cell.set_facecolor(row_colors[r - 1])
        elif r > 0:
            cell.set_facecolor(row_colors[r - 1])

    outfile = os.path.join(output_dir,
                           f'P3_fICS_vs_BCGmass{OutputFormat}')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


# ── Plot 4: Metallicity triptych ──────────────────────────────────────────────

def _make_stats_table(ax, row_labels, row_data, col_labels, row_colors,
                      bbox=None):
    """Draw a small stats table on an invisible axes."""
    if bbox is None:
        bbox = [0.02, 0.0, 0.96, 0.85]
    tbl = ax.table(
        cellText=row_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc='center',
        bbox=bbox,
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#cccccc')
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor('#e0e0e0')
            cell.set_text_props(weight='bold', fontsize=8)
        elif r > 0 and c == -1:
            cell.set_text_props(fontsize=7.5, ha='right')
            cell.set_facecolor(row_colors[r - 1])
        elif r > 0:
            cell.set_facecolor(row_colors[r - 1])


def plot_metallicity_triptych(file_list, sim_params, output_dir):
    """
    Three side-by-side panels at z=0:
      Left:   f_ICS (both defs) vs Z_BCG
      Middle: Z_ICS vs log10(Mvir)
      Right:  Z_ICS vs Z_BCG
    Metallicities are raw mass fractions Z = M_metals / M_component.
    """
    print(f'\n{"="*70}')
    print('Plot 4: Metallicity triptych at z=0')
    print(f'{"="*70}')

    h_ = sim_params['Hubble_h']
    redshifts = sim_params['redshifts']
    f_b = sim_params['BaryonFrac']
    min_stellar = sim_params['PartMass'] * sim_params['BaryonFrac']
    last_snap = sim_params['last_snapshot']
    z_snap = redshifts[last_snap]

    d = _load_snap_fields(
        file_list, last_snap, h_,
        mass_fields=['Mvir', 'IntraClusterStars', 'StellarMass',
                     'MetalsStellarMass', 'MetalsIntraClusterStars'],
        other_fields=['Type', 'GalaxyIndex', 'CentralGalaxyIndex'])
    if d['Mvir'].size == 0:
        print('  No data at z=0.')
        return

    Mvir = d['Mvir']
    ICS = d['IntraClusterStars']
    SM = d['StellarMass']
    MetSM = d['MetalsStellarMass']
    MetICS = d['MetalsIntraClusterStars']
    Type = d['Type']
    n_sat, sm_sat = _compute_satellite_sums(
        Type, SM, d['GalaxyIndex'], d['CentralGalaxyIndex'])
    total_stellar = SM + sm_sat + ICS

    bcg_mvir_ratio = np.where(Mvir > 0, SM / Mvir, 0)

    base = ((Type == 0) & (Mvir >= MVIR_GROUP_LO) & (ICS > 0)
            & (SM >= min_stellar) & (n_sat >= MIN_SATELLITES)
            & (total_stellar > 0) & (MetSM > 0) & (MetICS > 0))
    if FILTER_PATHOLOGICAL_BCGS:
        base = base & (bcg_mvir_ratio >= BCG_MVIR_RATIO_THRESHOLD)
    sel = np.where(base)[0]

    if len(sel) == 0:
        print('  No qualifying centrals.')
        return

    logMvir = np.log10(Mvir[sel])
    f_baryon = ICS[sel] / (f_b * Mvir[sel])
    f_stellar = ICS[sel] / total_stellar[sel]
    Z_sun = 0.02
    Z_BCG = np.log10((MetSM[sel] / SM[sel]) / Z_sun)
    Z_ICS = np.log10((MetICS[sel] / ICS[sel]) / Z_sun)

    is_cl = logMvir >= np.log10(MVIR_CLUSTER)
    is_gr = (logMvir >= np.log10(MVIR_GROUP_LO)) & (logMvir < np.log10(MVIR_CLUSTER))

    # ── Helper: binned bootstrap ──────────────────────────────────────────
    def _binned_boot(xvals, yvals, n_bins=15):
        edges = np.linspace(np.nanpercentile(xvals, 1),
                            np.nanpercentile(xvals, 99), n_bins + 1)
        cen = 0.5 * (edges[:-1] + edges[1:])
        means = np.full(len(cen), np.nan)
        lo = np.full(len(cen), np.nan)
        hi = np.full(len(cen), np.nan)
        for i in range(len(edges) - 1):
            m = (xvals >= edges[i]) & (xvals < edges[i + 1])
            if np.sum(m) >= MIN_N_PLOT:
                means[i], lo[i], hi[i] = bootstrap_mean_ci(yvals[m])
        return cen, means, lo, hi

    # ── Figure ────────────────────────────────────────────────────────────
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(20, 9))
    gs = gridspec.GridSpec(2, 3, height_ratios=[3, 1.2], hspace=0.35,
                           wspace=0.30)
    ax_L = fig.add_subplot(gs[0, 0])
    ax_M = fig.add_subplot(gs[0, 1])
    ax_R = fig.add_subplot(gs[0, 2])
    ax_sL = fig.add_subplot(gs[1, 0]); ax_sL.axis('off')
    ax_sM = fig.add_subplot(gs[1, 1]); ax_sM.axis('off')
    ax_sR = fig.add_subplot(gs[1, 2]); ax_sR.axis('off')

    col_bar = '#0C5DA5'
    col_ste = '#FF2C00'
    col_all = '#474747'
    row_color_map = {'cl': '#fcd5cc', 'gr': '#d6e5f3'}

    # ════════════════════════════════════════════════════════════════════
    # LEFT: f_ICS (both defs) vs Z_BCG — groups & clusters
    # ════════════════════════════════════════════════════════════════════
    for regime, mask, ls, alpha in [('cl', is_cl, '-', 0.15),
                                     ('gr', is_gr, '--', 0.10)]:
        xsub = Z_BCG[mask]
        if len(xsub) < MIN_N_PLOT:
            continue
        for yvals, col in [(f_baryon[mask], col_bar),
                           (f_stellar[mask], col_ste)]:
            cen, mm, lo, hi = _binned_boot(xsub, yvals)
            v = ~np.isnan(mm)
            if v.any():
                ax_L.plot(cen[v], mm[v], ls, color=col, lw=2.5)
                ax_L.fill_between(cen[v], lo[v], hi[v], color=col, alpha=alpha)

    ax_L.set_xlabel(r'$\log_{10}(Z_\mathrm{BCG}/Z_\odot)$')
    ax_L.set_ylabel(r'$f_\mathrm{ICS}$')
    ax_L.set_ylim(0, None)

    # Legend on left panel
    handles_L = [
        Line2D([0], [0], color=col_bar, ls='-', lw=2.5,
               label=r'Cl: $M_\mathrm{ICS}/(f_b M_\mathrm{vir})$'),
        Line2D([0], [0], color=col_ste, ls='-', lw=2.5,
               label=r'Cl: $M_\mathrm{ICS}/M_{\star,\mathrm{tot}}$'),
        Line2D([0], [0], color=col_bar, ls='--', lw=2.5,
               label=r'Gr: $M_\mathrm{ICS}/(f_b M_\mathrm{vir})$'),
        Line2D([0], [0], color=col_ste, ls='--', lw=2.5,
               label=r'Gr: $M_\mathrm{ICS}/M_{\star,\mathrm{tot}}$'),
    ]
    ax_L.legend(handles=handles_L, fontsize=7, loc='best', frameon=False)

    # Stats for left panel
    L_rows, L_data, L_colors = [], [], []
    for lbl, mask, rc in [('Clusters', is_cl, 'cl'), ('Groups', is_gr, 'gr')]:
        for yv, defn in [(f_stellar, 'stellar'), (f_baryon, 'baryon')]:
            v = yv[mask]; xv = Z_BCG[mask]
            rho, pv = subsampled_spearman(xv, v)
            L_rows.append(f'{lbl}, {defn}')
            L_data.append([f'{len(v)}', f'{np.mean(v):.4f}',
                           f'{np.std(v):.4f}',
                           f'{rho:+.3f}', f'{pv:.1e}'])
            L_colors.append(row_color_map[rc])
    _make_stats_table(ax_sL, L_rows, L_data,
                      ['N', 'Mean', r'$\sigma$', r'$\rho_s$', r'$p$'],
                      L_colors)

    # ════════════════════════════════════════════════════════════════════
    # MIDDLE: Z_ICS vs log10(Mvir)
    # ════════════════════════════════════════════════════════════════════
    cen, mm, lo, hi = _binned_boot(logMvir, Z_ICS)
    v = ~np.isnan(mm)
    if v.any():
        ax_M.plot(cen[v], mm[v], '-', color=col_all, lw=2.5)
        ax_M.fill_between(cen[v], lo[v], hi[v], color=col_all, alpha=0.15)

    ax_M.axvline(np.log10(MVIR_CLUSTER), color='grey', ls=':', lw=1.0,
                 alpha=0.6)
    ax_M.set_xlabel(
        r'$\log_{10}\, M_{\mathrm{vir}}\ [\mathrm{M}_{\odot}]$')
    ax_M.set_ylabel(r'$\log_{10}(Z_\mathrm{ICS}/Z_\odot)$')

    # Stats for middle panel
    M_rows, M_data, M_colors = [], [], []
    for lbl, mask, rc in [('Clusters', is_cl, 'cl'), ('Groups', is_gr, 'gr')]:
        vi = Z_ICS[mask]; xv = logMvir[mask]
        rho, pv = subsampled_spearman(xv, vi)
        M_rows.append(lbl)
        M_data.append([f'{len(vi)}', f'{np.mean(vi):.2f}',
                       f'{np.std(vi):.2f}',
                       f'{rho:+.3f}', f'{pv:.1e}'])
        M_colors.append(row_color_map[rc])
    _make_stats_table(ax_sM, M_rows, M_data,
                      ['N', 'Mean', r'$\sigma$', r'$\rho_s$', r'$p$'],
                      M_colors)

    # ════════════════════════════════════════════════════════════════════
    # RIGHT: Z_ICS vs Z_BCG — groups & clusters
    # ════════════════════════════════════════════════════════════════════
    for regime, mask, ls, alpha in [('cl', is_cl, '-', 0.15),
                                     ('gr', is_gr, '--', 0.10)]:
        xsub = Z_BCG[mask]
        if len(xsub) < MIN_N_PLOT:
            continue
        cen, mm, lo, hi = _binned_boot(xsub, Z_ICS[mask])
        v = ~np.isnan(mm)
        if v.any():
            ax_R.plot(cen[v], mm[v], ls, color=col_all, lw=2.5)
            ax_R.fill_between(cen[v], lo[v], hi[v], color=col_all, alpha=alpha)

    # 1:1 line
    zlo = min(Z_BCG.min(), Z_ICS.min())
    zhi = max(Z_BCG.max(), Z_ICS.max())
    pad = 0.1 * (zhi - zlo)
    zlims = [zlo - pad, zhi + pad]
    ax_R.plot(zlims, zlims, '--', color='grey', lw=1.0, alpha=0.5,
              label='1:1')
    ax_R.set_xlabel(r'$\log_{10}(Z_\mathrm{BCG}/Z_\odot)$')
    ax_R.set_ylabel(r'$\log_{10}(Z_\mathrm{ICS}/Z_\odot)$')
    ax_R.legend(handles=[
        Line2D([0], [0], color=col_all, ls='-', lw=2.5, label='Clusters'),
        Line2D([0], [0], color=col_all, ls='--', lw=2.5, label='Groups'),
        Line2D([0], [0], color='grey', ls='--', lw=1.0, alpha=0.5, label='1:1'),
    ], fontsize=8, loc='best', frameon=False)

    # Stats for right panel
    R_rows, R_data, R_colors = [], [], []
    for lbl, mask, rc in [('Clusters', is_cl, 'cl'), ('Groups', is_gr, 'gr')]:
        vi = Z_ICS[mask]; xv = Z_BCG[mask]
        rho, pv = subsampled_spearman(xv, vi)
        R_rows.append(lbl)
        R_data.append([f'{len(vi)}', f'{np.mean(vi):.2f}',
                       f'{np.std(vi):.2f}',
                       f'{rho:+.3f}', f'{pv:.1e}'])
        R_colors.append(row_color_map[rc])
    _make_stats_table(ax_sR, R_rows, R_data,
                      ['N', 'Mean', r'$\sigma$', r'$\rho_s$', r'$p$'],
                      R_colors)

    outfile = os.path.join(output_dir,
                           f'P4_metallicity_triptych{OutputFormat}')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


# ── Plot 5: f_ICS vs mass-step function ───────────────────────────────────────

def plot_fics_vs_mass_gap(file_list, sim_params, output_dir):
    """
    Both f_ICS definitions vs the mass-step function
    DeltaM*,12 = log10(M*,central) - log10(M*,satellite1)
    at z=0, binned mean lines with bootstrap CI, and statistics table below.
    """
    print(f'\n{"="*70}')
    print('Plot 5: f_ICS vs mass-step function at z=0')
    print(f'{"="*70}')

    h_ = sim_params['Hubble_h']
    redshifts = sim_params['redshifts']
    f_b = sim_params['BaryonFrac']
    min_stellar = sim_params['PartMass'] * sim_params['BaryonFrac']
    last_snap = sim_params['last_snapshot']
    z_snap = redshifts[last_snap]

    d = _load_snap_fields(
        file_list, last_snap, h_,
        mass_fields=['Mvir', 'IntraClusterStars', 'StellarMass'],
        other_fields=['Type', 'GalaxyIndex', 'CentralGalaxyIndex'])
    if d['Mvir'].size == 0:
        print('  No data at z=0.')
        return

    Mvir = d['Mvir']
    ICS = d['IntraClusterStars']
    SM = d['StellarMass']
    Type = d['Type']
    n_sat, sm_sat = _compute_satellite_sums(
        Type, SM, d['GalaxyIndex'], d['CentralGalaxyIndex'])
    max_sat_sm = _compute_max_satellite_mass(
        Type, SM, d['GalaxyIndex'], d['CentralGalaxyIndex'])
    total_stellar = SM + sm_sat + ICS

    bcg_mvir_ratio = np.where(Mvir > 0, SM / Mvir, 0)

    # Selection: centrals in group/cluster range with at least one massive satellite
    base = ((Type == 0) & (Mvir >= MVIR_GROUP_LO) & (ICS > 0)
            & (SM >= min_stellar) & (n_sat >= MIN_SATELLITES)
            & (total_stellar > 0) & (max_sat_sm > 0))
    if FILTER_PATHOLOGICAL_BCGS:
        base = base & (bcg_mvir_ratio >= BCG_MVIR_RATIO_THRESHOLD)
    sel = np.where(base)[0]

    if len(sel) == 0:
        print('  No qualifying centrals.')
        return

    logMvir = np.log10(Mvir[sel])
    f_baryon = ICS[sel] / (f_b * Mvir[sel])
    f_stellar = ICS[sel] / total_stellar[sel]
    delta_m12 = np.log10(SM[sel]) - np.log10(max_sat_sm[sel])

    print(f'  N qualifying centrals: {len(sel)}')
    print(f'  DeltaM*,12 range: [{delta_m12.min():.2f}, {delta_m12.max():.2f}]')

    # ── Binned statistics (per regime) ────────────────────────────────────
    is_cl = logMvir >= np.log10(MVIR_CLUSTER)
    is_gr = (logMvir >= np.log10(MVIR_GROUP_LO)) & (logMvir < np.log10(MVIR_CLUSTER))

    def _binned_bootstrap(vals, xvar, edges):
        means = np.full(len(edges) - 1, np.nan)
        lo = np.full(len(edges) - 1, np.nan)
        hi = np.full(len(edges) - 1, np.nan)
        counts = np.zeros(len(edges) - 1, dtype=int)
        for i in range(len(edges) - 1):
            m = (xvar >= edges[i]) & (xvar < edges[i + 1])
            counts[i] = int(np.sum(m))
            if counts[i] >= MIN_N_PLOT:
                means[i], lo[i], hi[i] = bootstrap_mean_ci(vals[m])
        return means, lo, hi, counts

    binned = {}
    for regime, mask in [('cl', is_cl), ('gr', is_gr)]:
        xsub = delta_m12[mask]
        if len(xsub) < MIN_N_PLOT:
            continue
        edges = np.linspace(np.nanpercentile(xsub, 1),
                            np.nanpercentile(xsub, 99), 16)
        cen = 0.5 * (edges[:-1] + edges[1:])
        for defn, vals in [('baryon', f_baryon[mask]),
                           ('stellar', f_stellar[mask])]:
            mm, lo, hi, _ = _binned_bootstrap(vals, xsub, edges)
            binned[f'{regime}_{defn}'] = (cen, mm, lo, hi)

    # ── Summary statistics ────────────────────────────────────────────────
    curve_meta = {
        'cl_stellar': 'Clusters, stellar-fraction',
        'cl_baryon':  'Clusters, baryon-budget',
        'gr_stellar': 'Groups, stellar-fraction',
        'gr_baryon':  'Groups, baryon-budget',
    }
    stat_data = {}
    for key, mask, vals in [
            ('cl_stellar', is_cl, f_stellar), ('cl_baryon', is_cl, f_baryon),
            ('gr_stellar', is_gr, f_stellar), ('gr_baryon', is_gr, f_baryon)]:
        v = vals[mask]
        dm = delta_m12[mask]
        rho, pval = subsampled_spearman(dm, v)
        stat_data[key] = {
            'n': len(v),
            'mean': np.mean(v) if len(v) else np.nan,
            'std': np.std(v) if len(v) else np.nan,
            'rho': rho, 'pval': pval,
        }
        print(f'  {curve_meta[key]}: N={len(v)}, '
              f'mean={np.mean(v):.4f}, std={np.std(v):.4f}, '
              f'Spearman(DM12): rho={rho:+.3f}, p={pval:.2e}')

    # ── Figure ────────────────────────────────────────────────────────────
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(12, 10.5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1.1], hspace=0.30)
    ax = fig.add_subplot(gs[0])
    ax_leg = fig.add_subplot(gs[1])
    ax_leg.axis('off')

    col_bar = '#0C5DA5'
    col_ste = '#FF2C00'

    style_map = {
        'cl_baryon': ('-', col_bar, 0.15), 'cl_stellar': ('-', col_ste, 0.15),
        'gr_baryon': ('--', col_bar, 0.10), 'gr_stellar': ('--', col_ste, 0.10),
    }
    for key, (ls, col, alpha) in style_map.items():
        if key not in binned:
            continue
        cen, mm, lo, hi = binned[key]
        v = ~np.isnan(mm)
        if v.any():
            ax.plot(cen[v], mm[v], ls, color=col, lw=2.5)
            ax.fill_between(cen[v], lo[v], hi[v], color=col, alpha=alpha)

    ax.set_xlabel(
        r'$\Delta M_{\star,12} = \log_{10}(M_{\star,\mathrm{BCG}})'
        r' - \log_{10}(M_{\star,\mathrm{sat1}})$')
    ax.set_ylabel(r'$f_\mathrm{ICS}$')
    ax.set_ylim(0, None)

    # ── Legend + stats in lower panel ─────────────────────────────────────
    handles = [
        Line2D([0], [0], color=col_bar, ls='-', lw=2.5,
               label=r'Clusters: $M_\mathrm{ICS}/(f_b M_\mathrm{vir})$'),
        Line2D([0], [0], color=col_ste, ls='-', lw=2.5,
               label=r'Clusters: $M_\mathrm{ICS}/M_{\star,\mathrm{tot}}$'),
        Line2D([0], [0], color=col_bar, ls='--', lw=2.5,
               label=r'Groups: $M_\mathrm{ICS}/(f_b M_\mathrm{vir})$'),
        Line2D([0], [0], color=col_ste, ls='--', lw=2.5,
               label=r'Groups: $M_\mathrm{ICS}/M_{\star,\mathrm{tot}}$'),
    ]

    ax_leg.legend(handles=handles, loc='upper center',
                  bbox_to_anchor=(0.5, 1.02),
                  ncol=2, fontsize=11, frameon=False,
                  handlelength=2.5, columnspacing=2.0,
                  handletextpad=0.8)

    # Statistics table
    col_labels = [r'$N$', 'Mean', r'$\sigma$',
                  r'$\rho_s(\Delta M_{\star,12})$', r'$p$-value']
    row_labels = []
    table_data = []
    row_colors = []
    color_map = {
        'cl_baryon': '#d6e5f3', 'cl_stellar': '#fcd5cc',
        'gr_baryon': '#e8eff7', 'gr_stellar': '#fde8e3',
    }

    for key in ['cl_stellar', 'cl_baryon', 'gr_stellar', 'gr_baryon']:
        sd = stat_data[key]
        row_labels.append(curve_meta[key])
        table_data.append([
            f'{sd["n"]}',
            f'{sd["mean"]:.4f}',
            f'{sd["std"]:.4f}',
            f'{sd["rho"]:+.3f}',
            f'{sd["pval"]:.1e}',
        ])
        row_colors.append(color_map[key])

    tbl = ax_leg.table(
        cellText=table_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc='center',
        bbox=[0.15, 0.05, 0.70, 0.55],
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#cccccc')
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor('#e0e0e0')
            cell.set_text_props(weight='bold', fontsize=10)
        elif r > 0 and c == -1:
            cell.set_text_props(fontsize=9, ha='right')
            cell.set_facecolor(row_colors[r - 1])
        elif r > 0:
            cell.set_facecolor(row_colors[r - 1])

    outfile = os.path.join(output_dir,
                           f'P5_fICS_vs_mass_gap{OutputFormat}')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


# ── Plot 6: f_ICS vs halo assembly time t_50 ────────────��────────────────────

def plot_fics_vs_t50(file_list, sim_params, output_dir):
    """
    Both f_ICS definitions vs t_50 (lookback time at which the halo first
    reached 50% of its z=0 virial mass), binned mean lines with bootstrap CI,
    and statistics table below.
    """
    print(f'\n{"="*70}')
    print('Plot 6: f_ICS vs halo assembly time t_50 at z=0')
    print(f'{"="*70}')

    h_ = sim_params['Hubble_h']
    f_b = sim_params['BaryonFrac']
    min_stellar = sim_params['PartMass'] * sim_params['BaryonFrac']
    last_snap = sim_params['last_snapshot']

    d = _load_snap_fields(
        file_list, last_snap, h_,
        mass_fields=['Mvir', 'IntraClusterStars', 'StellarMass'],
        other_fields=['Type', 'GalaxyIndex', 'CentralGalaxyIndex'])
    if d['Mvir'].size == 0:
        print('  No data at z=0.')
        return

    Mvir = d['Mvir']
    ICS = d['IntraClusterStars']
    SM = d['StellarMass']
    Type = d['Type']
    GalIdx = d['GalaxyIndex']
    n_sat, sm_sat = _compute_satellite_sums(
        Type, SM, GalIdx, d['CentralGalaxyIndex'])
    total_stellar = SM + sm_sat + ICS

    bcg_mvir_ratio = np.where(Mvir > 0, SM / Mvir, 0)

    base = ((Type == 0) & (Mvir >= MVIR_GROUP_LO) & (ICS > 0)
            & (SM >= min_stellar) & (n_sat >= MIN_SATELLITES)
            & (total_stellar > 0))
    if FILTER_PATHOLOGICAL_BCGS:
        base = base & (bcg_mvir_ratio >= BCG_MVIR_RATIO_THRESHOLD)
    sel = np.where(base)[0]

    if len(sel) == 0:
        print('  No qualifying centrals.')
        return

    # Compute halo t_50 by tracing Mvir backward through all snapshots
    t50, z50 = _compute_halo_t50(file_list, GalIdx[sel], Mvir[sel],
                                  sim_params)

    logMvir = np.log10(Mvir[sel])
    f_baryon = ICS[sel] / (f_b * Mvir[sel])
    f_stellar = ICS[sel] / total_stellar[sel]

    # Drop galaxies with no valid t_50
    ok = ~np.isnan(t50)
    if np.sum(ok) == 0:
        print('  No centrals with valid t_50.')
        return
    t50 = t50[ok]
    z50 = z50[ok]
    logMvir = logMvir[ok]
    f_baryon = f_baryon[ok]
    f_stellar = f_stellar[ok]

    print(f'  N with valid t_50: {len(t50)}')
    print(f'  z_50 range: [{z50.min():.2f}, {z50.max():.2f}]')

    # ── Binned statistics (per regime) ────────────────────────────────────
    is_cl = logMvir >= np.log10(MVIR_CLUSTER)
    is_gr = (logMvir >= np.log10(MVIR_GROUP_LO)) & (logMvir < np.log10(MVIR_CLUSTER))

    def _binned_bootstrap(vals, xvar, edges):
        means = np.full(len(edges) - 1, np.nan)
        lo = np.full(len(edges) - 1, np.nan)
        hi = np.full(len(edges) - 1, np.nan)
        counts = np.zeros(len(edges) - 1, dtype=int)
        for i in range(len(edges) - 1):
            m = (xvar >= edges[i]) & (xvar < edges[i + 1])
            counts[i] = int(np.sum(m))
            if counts[i] >= MIN_N_PLOT:
                means[i], lo[i], hi[i] = bootstrap_mean_ci(vals[m])
        return means, lo, hi, counts

    binned = {}
    for regime, mask in [('cl', is_cl), ('gr', is_gr)]:
        xsub = t50[mask]
        if len(xsub) < MIN_N_PLOT:
            continue
        edges = np.linspace(np.nanpercentile(xsub, 1),
                            np.nanpercentile(xsub, 99), 16)
        cen = 0.5 * (edges[:-1] + edges[1:])
        for defn, vals in [('baryon', f_baryon[mask]),
                           ('stellar', f_stellar[mask])]:
            mm, lo, hi, _ = _binned_bootstrap(vals, xsub, edges)
            binned[f'{regime}_{defn}'] = (cen, mm, lo, hi)

    # ── Summary statistics ────────────────────────────────────────────────
    curve_meta = {
        'cl_stellar': 'Clusters, stellar-fraction',
        'cl_baryon':  'Clusters, baryon-budget',
        'gr_stellar': 'Groups, stellar-fraction',
        'gr_baryon':  'Groups, baryon-budget',
    }
    stat_data = {}
    for key, mask, vals in [
            ('cl_stellar', is_cl, f_stellar), ('cl_baryon', is_cl, f_baryon),
            ('gr_stellar', is_gr, f_stellar), ('gr_baryon', is_gr, f_baryon)]:
        v = vals[mask]
        tv = t50[mask]
        rho, pval = subsampled_spearman(tv, v)
        stat_data[key] = {
            'n': len(v),
            'mean': np.mean(v) if len(v) else np.nan,
            'std': np.std(v) if len(v) else np.nan,
            'rho': rho, 'pval': pval,
        }
        print(f'  {curve_meta[key]}: N={len(v)}, '
              f'mean={np.mean(v):.4f}, std={np.std(v):.4f}, '
              f'Spearman(t50): rho={rho:+.3f}, p={pval:.2e}')

    # ── Figure ────────────────────────────────────────────────────────────
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(12, 10.5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1.1], hspace=0.30)
    ax = fig.add_subplot(gs[0])
    ax_leg = fig.add_subplot(gs[1])
    ax_leg.axis('off')

    col_bar = '#0C5DA5'
    col_ste = '#FF2C00'

    style_map = {
        'cl_baryon': ('-', col_bar, 0.15), 'cl_stellar': ('-', col_ste, 0.15),
        'gr_baryon': ('--', col_bar, 0.10), 'gr_stellar': ('--', col_ste, 0.10),
    }
    for key, (ls, col, alpha) in style_map.items():
        if key not in binned:
            continue
        cen, mm, lo, hi = binned[key]
        v = ~np.isnan(mm)
        if v.any():
            ax.plot(cen[v], mm[v], ls, color=col, lw=2.5)
            ax.fill_between(cen[v], lo[v], hi[v], color=col, alpha=alpha)

    ax.set_xlabel(r'$t_{50}$  halo assembly lookback [Gyr]')
    ax.set_ylabel(r'$f_\mathrm{ICS}$')
    ax.set_ylim(0, None)

    # ── Legend + stats in lower panel ─────────────────────────────────────
    handles = [
        Line2D([0], [0], color=col_bar, ls='-', lw=2.5,
               label=r'Clusters: $M_\mathrm{ICS}/(f_b M_\mathrm{vir})$'),
        Line2D([0], [0], color=col_ste, ls='-', lw=2.5,
               label=r'Clusters: $M_\mathrm{ICS}/M_{\star,\mathrm{tot}}$'),
        Line2D([0], [0], color=col_bar, ls='--', lw=2.5,
               label=r'Groups: $M_\mathrm{ICS}/(f_b M_\mathrm{vir})$'),
        Line2D([0], [0], color=col_ste, ls='--', lw=2.5,
               label=r'Groups: $M_\mathrm{ICS}/M_{\star,\mathrm{tot}}$'),
    ]

    ax_leg.legend(handles=handles, loc='upper center',
                  bbox_to_anchor=(0.5, 1.02),
                  ncol=2, fontsize=11, frameon=False,
                  handlelength=2.5, columnspacing=2.0,
                  handletextpad=0.8)

    # Statistics table
    col_labels = [r'$N$', 'Mean', r'$\sigma$',
                  r'$\rho_s(t_{50})$', r'$p$-value']
    row_labels = []
    table_data = []
    row_colors = []
    color_map = {
        'cl_baryon': '#d6e5f3', 'cl_stellar': '#fcd5cc',
        'gr_baryon': '#e8eff7', 'gr_stellar': '#fde8e3',
    }

    for key in ['cl_stellar', 'cl_baryon', 'gr_stellar', 'gr_baryon']:
        sd = stat_data[key]
        row_labels.append(curve_meta[key])
        table_data.append([
            f'{sd["n"]}',
            f'{sd["mean"]:.4f}',
            f'{sd["std"]:.4f}',
            f'{sd["rho"]:+.3f}',
            f'{sd["pval"]:.1e}',
        ])
        row_colors.append(color_map[key])

    tbl = ax_leg.table(
        cellText=table_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc='center',
        bbox=[0.15, 0.05, 0.70, 0.55],
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#cccccc')
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor('#e0e0e0')
            cell.set_text_props(weight='bold', fontsize=10)
        elif r > 0 and c == -1:
            cell.set_text_props(fontsize=9, ha='right')
            cell.set_facecolor(row_colors[r - 1])
        elif r > 0:
            cell.set_facecolor(row_colors[r - 1])

    outfile = os.path.join(output_dir,
                           f'P6_fICS_vs_t50{OutputFormat}')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


# ── Plot 7: f_ICS vs ICS assembly time t_asm ──────────────────────────────────

def plot_fics_vs_tasm(file_list, sim_params, output_dir):
    """
    Both f_ICS definitions vs t_asm (mass-weighted mean ICS deposition
    lookback time), binned mean lines with bootstrap CI, groups/clusters
    split, and statistics table below.
    """
    print(f'\n{"="*70}')
    print('Plot 7: f_ICS vs ICS assembly time t_asm at z=0')
    print(f'{"="*70}')

    h_ = sim_params['Hubble_h']
    f_b = sim_params['BaryonFrac']
    min_stellar = sim_params['PartMass'] * sim_params['BaryonFrac']
    last_snap = sim_params['last_snapshot']
    utm = sim_params['UnitTime_in_Megayears'] / 1000.0  # code time -> Gyr

    d = _load_snap_fields(
        file_list, last_snap, h_,
        mass_fields=['Mvir', 'IntraClusterStars', 'StellarMass',
                     'ICS_disrupt', 'ICS_accrete', 'ICS_sum_mt'],
        other_fields=['Type', 'GalaxyIndex', 'CentralGalaxyIndex'])
    if d['Mvir'].size == 0:
        print('  No data at z=0.')
        return

    Mvir = d['Mvir']
    ICS = d['IntraClusterStars']
    SM = d['StellarMass']
    Type = d['Type']
    n_sat, sm_sat = _compute_satellite_sums(
        Type, SM, d['GalaxyIndex'], d['CentralGalaxyIndex'])
    total_stellar = SM + sm_sat + ICS

    bcg_mvir_ratio = np.where(Mvir > 0, SM / Mvir, 0)

    # Compute t_asm
    denom = d['ICS_disrupt'] + d['ICS_accrete']
    t_asm_all = np.full(len(denom), np.nan)
    ok_asm = denom > 0
    t_asm_all[ok_asm] = d['ICS_sum_mt'][ok_asm] / denom[ok_asm] * utm

    base = ((Type == 0) & (Mvir >= MVIR_GROUP_LO) & (ICS > 0)
            & (SM >= min_stellar) & (n_sat >= MIN_SATELLITES)
            & (total_stellar > 0) & np.isfinite(t_asm_all) & (t_asm_all > 0))
    if FILTER_PATHOLOGICAL_BCGS:
        base = base & (bcg_mvir_ratio >= BCG_MVIR_RATIO_THRESHOLD)
    sel = np.where(base)[0]

    if len(sel) == 0:
        print('  No qualifying centrals.')
        return

    logMvir = np.log10(Mvir[sel])
    f_baryon = ICS[sel] / (f_b * Mvir[sel])
    f_stellar = ICS[sel] / total_stellar[sel]
    t_asm = t_asm_all[sel]

    print(f'  N: {len(sel)}')
    print(f'  t_asm range: [{t_asm.min():.2f}, {t_asm.max():.2f}] Gyr')

    # ── Binned statistics (per regime) ────────────────────────────────────
    is_cl = logMvir >= np.log10(MVIR_CLUSTER)
    is_gr = (logMvir >= np.log10(MVIR_GROUP_LO)) & (logMvir < np.log10(MVIR_CLUSTER))

    def _binned_bootstrap(vals, xvar, edges):
        means = np.full(len(edges) - 1, np.nan)
        lo = np.full(len(edges) - 1, np.nan)
        hi = np.full(len(edges) - 1, np.nan)
        for i in range(len(edges) - 1):
            m = (xvar >= edges[i]) & (xvar < edges[i + 1])
            if int(np.sum(m)) >= MIN_N_PLOT:
                means[i], lo[i], hi[i] = bootstrap_mean_ci(vals[m])
        return means, lo, hi

    binned = {}
    for regime, mask in [('cl', is_cl), ('gr', is_gr)]:
        xsub = t_asm[mask]
        if len(xsub) < MIN_N_PLOT:
            continue
        edges = np.linspace(np.nanpercentile(xsub, 1),
                            np.nanpercentile(xsub, 99), 16)
        cen = 0.5 * (edges[:-1] + edges[1:])
        for defn, vals in [('baryon', f_baryon[mask]),
                           ('stellar', f_stellar[mask])]:
            mm, lo, hi = _binned_bootstrap(vals, xsub, edges)
            binned[f'{regime}_{defn}'] = (cen, mm, lo, hi)

    # ── Summary statistics ────────────────────────────────────────────────
    curve_meta = {
        'cl_stellar': 'Clusters, stellar-fraction',
        'cl_baryon':  'Clusters, baryon-budget',
        'gr_stellar': 'Groups, stellar-fraction',
        'gr_baryon':  'Groups, baryon-budget',
    }
    stat_data = {}
    for key, mask, vals in [
            ('cl_stellar', is_cl, f_stellar), ('cl_baryon', is_cl, f_baryon),
            ('gr_stellar', is_gr, f_stellar), ('gr_baryon', is_gr, f_baryon)]:
        v = vals[mask]
        tv = t_asm[mask]
        rho, pval = subsampled_spearman(tv, v)
        stat_data[key] = {
            'n': len(v),
            'mean': np.mean(v) if len(v) else np.nan,
            'std': np.std(v) if len(v) else np.nan,
            'rho': rho, 'pval': pval,
        }
        print(f'  {curve_meta[key]}: N={len(v)}, '
              f'mean={np.mean(v):.4f}, std={np.std(v):.4f}, '
              f'Spearman(t_asm): rho={rho:+.3f}, p={pval:.2e}')

    # ── Figure ────────────────────────────────────────────────────────────
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(12, 10.5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1.1], hspace=0.30)
    ax = fig.add_subplot(gs[0])
    ax_leg = fig.add_subplot(gs[1])
    ax_leg.axis('off')

    col_bar = '#0C5DA5'
    col_ste = '#FF2C00'

    style_map = {
        'cl_baryon': ('-', col_bar, 0.15), 'cl_stellar': ('-', col_ste, 0.15),
        'gr_baryon': ('--', col_bar, 0.10), 'gr_stellar': ('--', col_ste, 0.10),
    }
    for key, (ls, col, alpha) in style_map.items():
        if key not in binned:
            continue
        cen, mm, lo, hi = binned[key]
        v = ~np.isnan(mm)
        if v.any():
            ax.plot(cen[v], mm[v], ls, color=col, lw=2.5)
            ax.fill_between(cen[v], lo[v], hi[v], color=col, alpha=alpha)

    ax.set_xlabel(r'$t_\mathrm{asm}$  ICS assembly lookback [Gyr]')
    ax.set_ylabel(r'$f_\mathrm{ICS}$')
    ax.set_ylim(0, None)

    # ── Legend + stats in lower panel ─────────────────────────────────────
    handles = [
        Line2D([0], [0], color=col_bar, ls='-', lw=2.5,
               label=r'Clusters: $M_\mathrm{ICS}/(f_b M_\mathrm{vir})$'),
        Line2D([0], [0], color=col_ste, ls='-', lw=2.5,
               label=r'Clusters: $M_\mathrm{ICS}/M_{\star,\mathrm{tot}}$'),
        Line2D([0], [0], color=col_bar, ls='--', lw=2.5,
               label=r'Groups: $M_\mathrm{ICS}/(f_b M_\mathrm{vir})$'),
        Line2D([0], [0], color=col_ste, ls='--', lw=2.5,
               label=r'Groups: $M_\mathrm{ICS}/M_{\star,\mathrm{tot}}$'),
    ]

    ax_leg.legend(handles=handles, loc='upper center',
                  bbox_to_anchor=(0.5, 1.02),
                  ncol=2, fontsize=11, frameon=False,
                  handlelength=2.5, columnspacing=2.0,
                  handletextpad=0.8)

    # Statistics table
    col_labels = [r'$N$', 'Mean', r'$\sigma$',
                  r'$\rho_s(t_\mathrm{asm})$', r'$p$-value']
    row_labels = []
    table_data = []
    row_colors = []
    color_map = {
        'cl_baryon': '#d6e5f3', 'cl_stellar': '#fcd5cc',
        'gr_baryon': '#e8eff7', 'gr_stellar': '#fde8e3',
    }

    for key in ['cl_stellar', 'cl_baryon', 'gr_stellar', 'gr_baryon']:
        sd = stat_data[key]
        row_labels.append(curve_meta[key])
        table_data.append([
            f'{sd["n"]}',
            f'{sd["mean"]:.4f}',
            f'{sd["std"]:.4f}',
            f'{sd["rho"]:+.3f}',
            f'{sd["pval"]:.1e}',
        ])
        row_colors.append(color_map[key])

    tbl = ax_leg.table(
        cellText=table_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc='center',
        bbox=[0.15, 0.05, 0.70, 0.55],
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#cccccc')
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor('#e0e0e0')
            cell.set_text_props(weight='bold', fontsize=10)
        elif r > 0 and c == -1:
            cell.set_text_props(fontsize=9, ha='right')
            cell.set_facecolor(row_colors[r - 1])
        elif r > 0:
            cell.set_facecolor(row_colors[r - 1])

    outfile = os.path.join(output_dir,
                           f'P7_fICS_vs_tasm{OutputFormat}')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


# ── Plot 8: t_asm vs t_50 ─────────────────────────────────────────────────────

def plot_tasm_vs_t50(file_list, sim_params, output_dir):
    """
    ICS assembly time t_asm vs halo assembly time t_50 at z=0,
    binned mean lines with bootstrap CI, groups/clusters split,
    and statistics table below.
    """
    print(f'\n{"="*70}')
    print('Plot 8: t_asm vs t_50 at z=0')
    print(f'{"="*70}')

    h_ = sim_params['Hubble_h']
    f_b = sim_params['BaryonFrac']
    min_stellar = sim_params['PartMass'] * sim_params['BaryonFrac']
    last_snap = sim_params['last_snapshot']
    utm = sim_params['UnitTime_in_Megayears'] / 1000.0

    d = _load_snap_fields(
        file_list, last_snap, h_,
        mass_fields=['Mvir', 'IntraClusterStars', 'StellarMass',
                     'ICS_disrupt', 'ICS_accrete', 'ICS_sum_mt'],
        other_fields=['Type', 'GalaxyIndex', 'CentralGalaxyIndex'])
    if d['Mvir'].size == 0:
        print('  No data at z=0.')
        return

    Mvir = d['Mvir']
    ICS = d['IntraClusterStars']
    SM = d['StellarMass']
    Type = d['Type']
    GalIdx = d['GalaxyIndex']
    n_sat, sm_sat = _compute_satellite_sums(
        Type, SM, GalIdx, d['CentralGalaxyIndex'])
    total_stellar = SM + sm_sat + ICS

    bcg_mvir_ratio = np.where(Mvir > 0, SM / Mvir, 0)

    # Compute t_asm
    denom = d['ICS_disrupt'] + d['ICS_accrete']
    t_asm_all = np.full(len(denom), np.nan)
    ok_asm = denom > 0
    t_asm_all[ok_asm] = d['ICS_sum_mt'][ok_asm] / denom[ok_asm] * utm

    base = ((Type == 0) & (Mvir >= MVIR_GROUP_LO) & (ICS > 0)
            & (SM >= min_stellar) & (n_sat >= MIN_SATELLITES)
            & (total_stellar > 0) & np.isfinite(t_asm_all) & (t_asm_all > 0))
    if FILTER_PATHOLOGICAL_BCGS:
        base = base & (bcg_mvir_ratio >= BCG_MVIR_RATIO_THRESHOLD)
    sel = np.where(base)[0]

    if len(sel) == 0:
        print('  No qualifying centrals.')
        return

    # Compute halo t_50
    t50, z50 = _compute_halo_t50(file_list, GalIdx[sel], Mvir[sel],
                                  sim_params)

    logMvir = np.log10(Mvir[sel])
    t_asm = t_asm_all[sel]

    # Drop galaxies with no valid t_50
    ok = ~np.isnan(t50)
    if np.sum(ok) == 0:
        print('  No centrals with valid t_50.')
        return
    t50 = t50[ok]
    t_asm = t_asm[ok]
    logMvir = logMvir[ok]

    print(f'  N with valid t_50 and t_asm: {len(t50)}')
    print(f'  t_asm range: [{t_asm.min():.2f}, {t_asm.max():.2f}] Gyr')
    print(f'  t_50  range: [{t50.min():.2f}, {t50.max():.2f}] Gyr')

    # ── Binned statistics (per regime) ────────────────────────────────────
    is_cl = logMvir >= np.log10(MVIR_CLUSTER)
    is_gr = (logMvir >= np.log10(MVIR_GROUP_LO)) & (logMvir < np.log10(MVIR_CLUSTER))

    def _binned_bootstrap(vals, xvar, edges):
        means = np.full(len(edges) - 1, np.nan)
        lo = np.full(len(edges) - 1, np.nan)
        hi = np.full(len(edges) - 1, np.nan)
        for i in range(len(edges) - 1):
            m = (xvar >= edges[i]) & (xvar < edges[i + 1])
            if int(np.sum(m)) >= MIN_N_PLOT:
                means[i], lo[i], hi[i] = bootstrap_mean_ci(vals[m])
        return means, lo, hi

    binned = {}
    for regime, mask in [('cl', is_cl), ('gr', is_gr)]:
        xsub = t50[mask]
        if len(xsub) < MIN_N_PLOT:
            continue
        edges = np.linspace(np.nanpercentile(xsub, 1),
                            np.nanpercentile(xsub, 99), 16)
        cen = 0.5 * (edges[:-1] + edges[1:])
        mm, lo, hi = _binned_bootstrap(t_asm[mask], xsub, edges)
        binned[regime] = (cen, mm, lo, hi)

    # ── Summary statistics ────────────────────────────────────────────────
    stat_data = {}
    for lbl, mask in [('Clusters', is_cl), ('Groups', is_gr)]:
        tv = t_asm[mask]
        xv = t50[mask]
        rho, pval = subsampled_spearman(xv, tv)
        stat_data[lbl] = {
            'n': len(tv),
            'mean_tasm': np.mean(tv) if len(tv) else np.nan,
            'std_tasm': np.std(tv) if len(tv) else np.nan,
            'mean_t50': np.mean(xv) if len(xv) else np.nan,
            'rho': rho, 'pval': pval,
        }
        print(f'  {lbl}: N={len(tv)}, '
              f'mean t_asm={np.mean(tv):.2f} Gyr, '
              f'mean t_50={np.mean(xv):.2f} Gyr, '
              f'Spearman(t_50, t_asm): rho={rho:+.3f}, p={pval:.2e}')

    # ── Figure ────────────────────────────────────────────────────────────
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(12, 10.5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1.1], hspace=0.30)
    ax = fig.add_subplot(gs[0])
    ax_leg = fig.add_subplot(gs[1])
    ax_leg.axis('off')

    col_all = '#474747'

    style_map = {
        'cl': ('-', col_all, 0.15),
        'gr': ('--', col_all, 0.10),
    }
    for regime, (ls, col, alpha) in style_map.items():
        if regime not in binned:
            continue
        cen, mm, lo, hi = binned[regime]
        v = ~np.isnan(mm)
        if v.any():
            ax.plot(cen[v], mm[v], ls, color=col, lw=2.5)
            ax.fill_between(cen[v], lo[v], hi[v], color=col, alpha=alpha)

    # 1:1 line
    tlo = min(t50.min(), t_asm.min())
    thi = max(t50.max(), t_asm.max())
    pad = 0.1 * (thi - tlo)
    tlims = [tlo - pad, thi + pad]
    ax.plot(tlims, tlims, '--', color='grey', lw=1.0, alpha=0.5)

    ax.set_xlabel(r'$t_{50}$  halo assembly lookback [Gyr]')
    ax.set_ylabel(r'$t_\mathrm{asm}$  ICS assembly lookback [Gyr]')

    # ── Legend + stats in lower panel ─────────────────────────────────────
    handles = [
        Line2D([0], [0], color=col_all, ls='-', lw=2.5, label='Clusters'),
        Line2D([0], [0], color=col_all, ls='--', lw=2.5, label='Groups'),
        Line2D([0], [0], color='grey', ls='--', lw=1.0, alpha=0.5,
               label='1:1'),
    ]

    ax_leg.legend(handles=handles, loc='upper center',
                  bbox_to_anchor=(0.5, 1.02),
                  ncol=3, fontsize=11, frameon=False,
                  handlelength=2.5, columnspacing=2.0,
                  handletextpad=0.8)

    # Statistics table
    col_labels = [r'$N$',
                  r'$\langle t_\mathrm{asm}\rangle$ [Gyr]',
                  r'$\sigma(t_\mathrm{asm})$',
                  r'$\rho_s(t_{50}, t_\mathrm{asm})$', r'$p$-value']
    row_labels = []
    table_data = []
    row_colors = []
    color_map = {'Clusters': '#fcd5cc', 'Groups': '#d6e5f3'}

    for lbl in ['Clusters', 'Groups']:
        sd = stat_data[lbl]
        row_labels.append(lbl)
        table_data.append([
            f'{sd["n"]}',
            f'{sd["mean_tasm"]:.2f}',
            f'{sd["std_tasm"]:.2f}',
            f'{sd["rho"]:+.3f}',
            f'{sd["pval"]:.1e}',
        ])
        row_colors.append(color_map[lbl])

    tbl = ax_leg.table(
        cellText=table_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc='center',
        bbox=[0.15, 0.15, 0.70, 0.45],
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#cccccc')
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor('#e0e0e0')
            cell.set_text_props(weight='bold', fontsize=10)
        elif r > 0 and c == -1:
            cell.set_text_props(fontsize=9, ha='right')
            cell.set_facecolor(row_colors[r - 1])
        elif r > 0:
            cell.set_facecolor(row_colors[r - 1])

    outfile = os.path.join(output_dir,
                           f'P8_tasm_vs_t50{OutputFormat}')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_arguments():
    p = argparse.ArgumentParser(
        description='ICS paper plotting suite (SAGE26)')
    p.add_argument('input_pattern', nargs='?',
                   default='./output/millennium/model_*.hdf5',
                   help='Path pattern to model HDF5 files')
    p.add_argument('-o', '--output-dir', type=str, default=None,
                   help='Output directory (default: <input_dir>/plots/)')
    p.add_argument('--max-redshift', type=float, default=2.5,
                   help='Maximum redshift for f_ICS plot (default: 2.5)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    file_list = sorted(glob.glob(args.input_pattern))
    if not file_list:
        print(f'Error: no files match {args.input_pattern}')
        sys.exit(1)
    print(f'Found {len(file_list)} model files.')

    sim_params = read_simulation_params(file_list[0])

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(file_list[0])), 'plots')
    os.makedirs(output_dir, exist_ok=True)

    print(f'Baryon fraction: {sim_params["BaryonFrac"]:.3f}')
    print(f'Particle mass:   {sim_params["PartMass"]:.4e} Msun')
    print(f'Stellar threshold: '
          f'{sim_params["PartMass"]*sim_params["BaryonFrac"]:.4e} Msun')
    print(f'Output directory: {output_dir}')

    plot_fics_vs_redshift(file_list, sim_params, output_dir,
                          max_redshift=args.max_redshift)
    plot_fics_vs_mvir(file_list, sim_params, output_dir)
    plot_fics_vs_bcg_mass(file_list, sim_params, output_dir)
    plot_metallicity_triptych(file_list, sim_params, output_dir)
    plot_fics_vs_mass_gap(file_list, sim_params, output_dir)
    plot_fics_vs_t50(file_list, sim_params, output_dir)
    plot_fics_vs_tasm(file_list, sim_params, output_dir)
    plot_tasm_vs_t50(file_list, sim_params, output_dir)

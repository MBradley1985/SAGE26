#!/usr/bin/env python
"""
Plot mass ratio vs lookback time for SAGE merger/disruption events,
overlaid with Boylan-Kolchin et al. (2008) dynamical friction timescales.

Reproduces the style of plot showing which satellites merge, which disrupt
to ICS, and which (according to BK08) should not merge within a Hubble time.
"""

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import os
import glob
from scipy.integrate import quad
from scipy.optimize import brentq
from random import sample, seed as rseed

# ========================== CONFIGURATION ==========================

PRIMARY_DIR = './output/millennium/'
OUTPUT_DIR = './output/millennium/plots/'
SNAP_MIN = 5  # skip earliest snapshots (very high z, few galaxies)
MIN_CENTRAL_MVIR = 10**14  # Msun (after unit conversion)

DILUTE = 10000  # max points per population for readability
SEED = 42

MIN_LEN = 20  # minimum halo particle count (resolution limit)

PROPERTIES = [
    'Type', 'mergeType', 'CentralMvir', 'infallMvir',
    'StellarMass', 'infallStellarMass',
    'GalaxyIndex', 'CentralGalaxyIndex',
    'TimeOfInfall', 'Len',
]

MASS_PROPS = frozenset({
    'CentralMvir', 'Mvir', 'StellarMass', 'infallStellarMass', 'infallMvir',
})

# ========================== HEADER / COSMOLOGY ==========================

def read_header(directory):
    pattern = os.path.join(directory, 'model_*.hdf5')
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No model files in {directory}")
    with h5.File(files[0], 'r') as f:
        sim = f['Header/Simulation']
        runtime = f['Header/Runtime']
        header = {
            'hubble_h':     float(sim.attrs['hubble_h']),
            'omega_matter': float(sim.attrs['omega_matter']),
            'omega_lambda': float(sim.attrs['omega_lambda']),
            'redshifts':    list(f['Header/snapshot_redshifts'][:]),
            'output_snaps': sorted(int(s) for s in f['Header/output_snapshots'][:]),
            'mass_convert': float(runtime.attrs['UnitMass_in_g']) / 1.989e33
                            / float(sim.attrs['hubble_h']),
            'FractionDisruptedToICS': float(runtime.attrs.get('FractionDisruptedToICS', 0.8)),
            'DynamicDisruptionSplit': int(runtime.attrs.get('DynamicDisruptionSplit', 0)),
            'DisruptionSplitAlpha':   float(runtime.attrs.get('DisruptionSplitAlpha', 0.4)),
        }
    return header, files


def make_cosmology(header):
    h = header['hubble_h']
    Om = header['omega_matter']
    Ol = header['omega_lambda']
    t_H = 977.8 / (h * 100)  # Hubble time in Gyr

    def cosmic_time(z):
        integrand = lambda zp: 1.0 / ((1 + zp) * np.sqrt(Om * (1 + zp)**3 + Ol))
        result, _ = quad(integrand, z, 1000.0)
        return t_H * result

    age_now = cosmic_time(0.0)

    def lookback(z):
        return age_now - cosmic_time(z)

    def z_from_lookback(t_lb):
        if t_lb <= 0:
            return 0.0
        return brentq(lambda z: lookback(z) - t_lb, 0, 30)

    def H_gyr(z):
        """Hubble parameter in Gyr^-1."""
        return (h * 100 / 977.8) * np.sqrt(Om * (1 + z)**3 + Ol)

    return lookback, z_from_lookback, H_gyr, age_now


# ========================== BK08 MERGER TIMESCALE ==========================

def BK08_boundary(t_lb_array, z_from_lookback, H_gyr, eta=0.5, factor=1.0):
    """
    For each lookback time, find the mass ratio M_host/M_sat at which
    the BK08 merger timescale equals the available cosmic time (= t_lb).

    BK08 formula (Boylan-Kolchin, Ma & Quataert 2008, MNRAS 383, 93):
        T_merge / t_dyn = A * (M/m)^b * exp(c * eta) / ln(1 + M/m)

    where t_dyn = sqrt(2 / Delta_vir) / H(z), Delta_vir ~ 200.

    Returns log10(M_host / M_sat) for each t_lb.
    """
    A, b, c = 0.216, 1.3, 1.9
    Delta_vir = 200.0

    log_ratios = np.full_like(t_lb_array, np.nan)
    for i, t_lb in enumerate(t_lb_array):
        if t_lb <= 0:
            continue
        z = z_from_lookback(t_lb)
        t_dyn = np.sqrt(2.0 / Delta_vir) / H_gyr(z)  # Gyr

        # Solve: A * r^b * exp(c*eta) / ln(1+r) * t_dyn * factor = t_lb
        # for r = M_host / M_sat
        target = t_lb / (factor * t_dyn)

        def objective(log_r):
            r = 10**log_r
            return A * r**b * np.exp(c * eta) / np.log(1 + r) - target

        try:
            log_ratios[i] = brentq(objective, 0.0, 6.0)
        except ValueError:
            pass

    return log_ratios


# ========================== DATA I/O ==========================

def read_snap(filepaths, snap_key, properties, mass_convert):
    chunks = {p: [] for p in properties}
    for fp in filepaths:
        with h5.File(fp, 'r') as f:
            if snap_key not in f:
                continue
            grp = f[snap_key]
            for p in properties:
                if p in grp:
                    chunks[p].append(np.array(grp[p]))
    data = {}
    for p in properties:
        if chunks[p]:
            arr = np.concatenate(chunks[p])
            if p in MASS_PROPS:
                arr *= mass_convert
            data[p] = arr
    return data


def collect_events(model_files, header, lookback_fn):
    """Iterate over snapshots, collect merger and disruption events."""
    redshifts = header['redshifts']
    output_snaps = header['output_snaps']
    mc = header['mass_convert']

    # Pre-compute lookback times for all snapshots
    snap_lookback = {}
    for s in range(len(redshifts)):
        snap_lookback[s] = lookback_fn(redshifts[s])

    # First pass: load z=0 centrals to get final BCG stellar masses
    print("  Loading z=0 centrals for BCG mass normalisation...")
    snap_z0 = output_snaps[-1]
    z0_data = read_snap(model_files, f'Snap_{snap_z0}',
                        ['Type', 'StellarMass', 'GalaxyIndex'], mc)
    z0_centrals = z0_data['Type'] == 0
    z0_lookup = dict(zip(z0_data['GalaxyIndex'][z0_centrals],
                         z0_data['StellarMass'][z0_centrals]))

    all_ratio, all_tlb, all_frac, all_smass = [], [], [], []
    all_evtype, all_t_since_infall, all_t_lb_infall, all_len = [], [], [], []

    for snap in output_snaps:
        if snap < SNAP_MIN:
            continue

        z = redshifts[snap]
        t_lb = lookback_fn(z)
        data = read_snap(model_files, f'Snap_{snap}', PROPERTIES, mc)

        if not data or 'mergeType' not in data:
            continue

        mtype = data['mergeType']
        cmvir = data['CentralMvir']
        gtype = data['Type']
        infall_mvir = data['infallMvir']
        smass = data['StellarMass']
        infall_smass = data['infallStellarMass']
        gidx = data['GalaxyIndex']
        cgidx = data['CentralGalaxyIndex']
        toi = data.get('TimeOfInfall')
        halo_len = data.get('Len')

        valid = (infall_mvir > 0) & (cmvir >= MIN_CENTRAL_MVIR)
        if halo_len is not None:
            valid &= (halo_len >= MIN_LEN)

        # Disruption split parameters from the model run
        dynamic_split = header['DynamicDisruptionSplit']
        alpha = header['DisruptionSplitAlpha']
        fixed_frac_ics = header['FractionDisruptedToICS']

        for mtype_sel, evtype_val in [
            ((mtype == 1) | (mtype == 2), 0),   # merger
            (mtype == 4, 1),                      # disruption
        ]:
            sel = valid & mtype_sel
            if not np.any(sel):
                continue

            n_sel = np.sum(sel)
            ratio = cmvir[sel] / infall_mvir[sel]
            sat_smass = infall_smass[sel]

            # Compute per-event fraction going to BCG
            if evtype_val == 0:
                # Mergers: all stellar mass goes to BCG
                frac_to_BCG = np.ones(n_sel)
            elif dynamic_split == 1:
                # Dynamic split: f_ICL = 1 - (Msub/Mhost)^alpha
                mass_ratio = np.clip(infall_mvir[sel] / cmvir[sel], 0, 1)
                f_ICL = 1.0 - mass_ratio**alpha
                frac_to_BCG = 1.0 - f_ICL  # = (Msub/Mhost)^alpha
            else:
                # Fixed fraction
                frac_to_BCG = np.full(n_sel, 1.0 - fixed_frac_ics)

            # Use z=0 BCG stellar mass as denominator where available,
            # fall back to instantaneous central stellar mass
            centrals_mask = gtype == 0
            inst_lookup = dict(zip(gidx[centrals_mask], smass[centrals_mask]))
            central_sm = np.array([
                z0_lookup.get(cg, inst_lookup.get(cg, np.nan))
                for cg in cgidx[sel]
            ])
            frac = np.where(central_sm > 0,
                            frac_to_BCG * sat_smass / central_sm, 0.0)

            # Time since infall (TimeOfInfall is stored as snapshot number)
            t_since = np.full(n_sel, np.nan)
            t_lb_inf = np.full(n_sel, np.nan)
            if toi is not None:
                toi_sel = toi[sel]
                for j in range(n_sel):
                    si = int(toi_sel[j])
                    if 0 <= si < len(redshifts):
                        t_lb_inf[j] = snap_lookback[si]
                        t_since[j] = t_lb_inf[j] - t_lb

            all_ratio.append(np.log10(ratio))
            all_tlb.append(np.full(n_sel, t_lb))
            all_frac.append(frac)
            all_smass.append(sat_smass)
            all_evtype.append(np.full(n_sel, evtype_val, dtype=int))
            all_t_since_infall.append(t_since)
            all_t_lb_infall.append(t_lb_inf)
            if halo_len is not None:
                all_len.append(halo_len[sel])
            else:
                all_len.append(np.full(n_sel, np.nan))

        n_m = np.sum(valid & ((mtype == 1) | (mtype == 2)))
        n_d = np.sum(valid & (mtype == 4))
        print(f"  Snap {snap:2d}  z={z:.2f}  t_lb={t_lb:.1f} Gyr  "
              f"mergers={n_m:,}  disruptions={n_d:,}")

    return {
        'ratio': np.concatenate(all_ratio),
        'tlb': np.concatenate(all_tlb),
        'frac': np.concatenate(all_frac),
        'smass': np.concatenate(all_smass),
        'evtype': np.concatenate(all_evtype),
        't_since_infall': np.concatenate(all_t_since_infall),
        'len': np.concatenate(all_len),
        't_lb_infall': np.concatenate(all_t_lb_infall),
    }


# ========================== PLOTTING ==========================

def dilute(n, *arrays):
    if n <= DILUTE:
        return arrays
    rseed(SEED)
    idx = sorted(sample(range(n), DILUTE))
    return tuple(a[idx] for a in arrays)


def setup_style():
    # Test if LaTeX + Palatino is available
    use_tex = True
    try:
        import subprocess
        result = subprocess.run(['kpsewhich', 'pplr7t.tfm'],
                                capture_output=True, timeout=5)
        if result.returncode != 0:
            use_tex = False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        use_tex = False

    style = {
        'figure.dpi': 150,
        'font.size': 14,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.2,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.minor.size': 3,
        'ytick.minor.size': 3,
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,
        'xtick.top': True,
        'ytick.right': True,
    }

    plt.rcParams.update(style)


def scatter_by_type(ax, x, y, frac, smass, evtype, norm, cmap):
    """Scatter points split by event type: circles for mergers, squares for disruptions."""
    sm_98 = np.nanpercentile(smass, 98) if len(smass) > 0 else 1.0
    sizes = np.clip(3 + 250 * (smass / sm_98), 3, 300)

    # Sort by fraction so high-fraction points are drawn on top
    sort_idx = np.argsort(frac)
    x, y, frac, evtype, sizes = x[sort_idx], y[sort_idx], frac[sort_idx], evtype[sort_idx], sizes[sort_idx]

    is_zero = frac < 1e-3
    nz = ~is_zero
    im = None

    for evtype_val, marker in [(0, 'o'), (1, 's')]:
        pop = evtype == evtype_val

        # Open markers: near-zero contribution
        mask = pop & is_zero
        if np.any(mask):
            ax.scatter(x[mask], y[mask],
                       s=sizes[mask], marker=marker,
                       facecolors='none', edgecolors='grey', linewidths=0.5,
                       rasterized=True, zorder=2)

        # Filled markers: colored by fraction
        mask = pop & nz
        if np.any(mask):
            sc = ax.scatter(x[mask], y[mask],
                           s=sizes[mask], alpha=0.5, marker=marker,
                           edgecolors='none', rasterized=True, zorder=3,
                           c=frac[mask], cmap=cmap, norm=norm)
            if im is None:
                im = sc

    return im


def plot_bk08_comparison(ev, lookback_fn, z_from_lb, H_gyr, age_now):
    """Plot 1: BK08 comparison (lookback time x-axis)."""
    # ---- BK08 curves ----
    t_grid = np.linspace(0.5, age_now - 0.5, 200)

    bk_median = BK08_boundary(t_grid, z_from_lb, H_gyr, eta=0.5)
    bk_lo = BK08_boundary(t_grid, z_from_lb, H_gyr, eta=0.3)
    bk_hi = BK08_boundary(t_grid, z_from_lb, H_gyr, eta=0.7)
    bk_2x = BK08_boundary(t_grid, z_from_lb, H_gyr, eta=0.5, factor=2.0)

    fig, ax = plt.subplots(figsize=(10, 7))

    # BK08 shaded region (15-85th percentile of orbital parameters)
    valid = np.isfinite(bk_lo) & np.isfinite(bk_hi)
    ax.fill_between(t_grid[valid], bk_lo[valid], bk_hi[valid],
                     color='steelblue', alpha=0.15, zorder=10)
    ax.plot(t_grid[valid], bk_median[valid], color='steelblue', lw=2, zorder=11)

    valid_2x = np.isfinite(bk_2x)
    ax.plot(t_grid[valid_2x], bk_2x[valid_2x], color='darkorange', lw=2, ls='--', zorder=11)

    # Crossing time reference lines
    A_bk, b_bk, c_bk, eta_ref = 0.216, 1.3, 1.9, 0.5
    tcross_lines = [
        (1,  ':',  r'$1\,t_{\rm cross}$'),
        (3,  '-.',  r'$3\,t_{\rm cross}$'),
        (10, '--', r'$10\,t_{\rm cross}$'),
    ]
    for N, ls, label in tcross_lines:
        def _obj(log_r, target=N):
            r = 10**log_r
            return A_bk * r**b_bk * np.exp(c_bk * eta_ref) / np.log(1 + r) - target
        try:
            log_r = brentq(_obj, -0.5, 5.0)
            ax.axhline(log_r, color='firebrick', ls=ls, lw=1.5, alpha=0.7, zorder=9)
        except ValueError:
            pass

    # Scatter points
    norm = mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=0.5)
    im = scatter_by_type(ax, ev['tlb'], ev['ratio'], ev['frac'],
                         ev['smass'], ev['evtype'], norm, 'viridis')

    if im is not None:
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label(r'Fraction of $z{=}0$ BCG $M_*$ contributed')

    # Legend with proxy artists
    legend_elements = [
        Patch(facecolor='steelblue', alpha=0.15, edgecolor='steelblue',
              label=r'BK08 15--85\% $\eta$'),
        Line2D([0], [0], color='steelblue', lw=2, label=r'BK08 ($\eta = 0.5$)'),
        Line2D([0], [0], color='darkorange', lw=2, ls='--', label=r'$2 \times$ BK08'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='grey',
               markeredgecolor='grey', markersize=8, linestyle='None', label='Merger'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='grey',
               markeredgecolor='grey', markersize=8, linestyle='None', label='Disruption'),
        Line2D([0], [0], color='firebrick', ls=':', lw=1.5, alpha=0.7,
               label=r'$1\,t_{\rm cross}$'),
        Line2D([0], [0], color='firebrick', ls='-.', lw=1.5, alpha=0.7,
               label=r'$3\,t_{\rm cross}$'),
        Line2D([0], [0], color='firebrick', ls='--', lw=1.5, alpha=0.7,
               label=r'$10\,t_{\rm cross}$'),
    ]
    ax.set_xlabel(r'Lookback time [Gyr]')
    ax.set_ylabel(r'$\log_{10}(M_{\rm central} \,/\, M_{\rm sat,\,infall})$')
    ax.set_xlim(0, age_now)
    ax.set_ylim(3.0, -0.5)

    # Redshift axis on top
    ax2 = ax.twiny()
    z_ticks = [0, 0.5, 1, 2, 3, 5, 8]
    t_ticks = [lookback_fn(z) for z in z_ticks]
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(t_ticks)
    ax2.set_xticklabels([f'{z}' for z in z_ticks])
    ax2.set_xlabel(r'$z$')

    fig.tight_layout()
    ax.legend(handles=legend_elements, loc='upper center',
              bbox_to_anchor=(0.45, -0.12), ncol=6, fontsize=9,
              framealpha=0.9, columnspacing=1.2, handletextpad=0.5)
    return fig


def plot_time_since_infall(ev, H_gyr):
    """Plot 2: Mass ratio vs time since infall with crossing time reference."""
    # Filter to valid t_since_infall
    valid = np.isfinite(ev['t_since_infall']) & (ev['t_since_infall'] > 0)
    if not np.any(valid):
        print("  No valid time-since-infall data; skipping plot.")
        return None

    ratio = ev['ratio'][valid]
    t_si = ev['t_since_infall'][valid]
    frac = ev['frac'][valid]
    evtype = ev['evtype'][valid]
    smass = ev['smass'][valid]

    fig, ax = plt.subplots(figsize=(10, 7))

    # Crossing time band (z=2 to z=0)
    Delta_vir = 200.0
    t_cross_z0 = np.sqrt(2.0 / Delta_vir) / H_gyr(0.0)
    t_cross_z1 = np.sqrt(2.0 / Delta_vir) / H_gyr(1.0)
    t_cross_z2 = np.sqrt(2.0 / Delta_vir) / H_gyr(2.0)
    ax.axvspan(0, t_cross_z2, color='firebrick', alpha=0.08, zorder=0)
    ax.axvline(t_cross_z0, color='firebrick', ls=':', lw=1.5, alpha=0.7, zorder=9)
    ax.axvline(t_cross_z1, color='firebrick', ls='--', lw=1.5, alpha=0.7, zorder=9)

    # BK08 predicted merger timescale curves (T_merge vs mass ratio)
    A, b, c, eta = 0.216, 1.3, 1.9, 0.5
    log_r = np.linspace(-0.5, 3.5, 500)
    r = 10**log_r
    bk_factor = A * r**b * np.exp(c * eta) / np.log(1 + r)

    t_dyn_z0 = np.sqrt(2.0 / Delta_vir) / H_gyr(0.0)
    t_dyn_z1 = np.sqrt(2.0 / Delta_vir) / H_gyr(1.0)

    ax.plot(bk_factor * t_dyn_z0, log_r, color='steelblue', lw=2, zorder=11)
    ax.plot(bk_factor * t_dyn_z1, log_r, color='steelblue', lw=2, ls='--', zorder=11)

    # Scatter points
    norm = mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=0.5)
    im = scatter_by_type(ax, t_si, ratio, frac, smass, evtype, norm, 'viridis')

    if im is not None:
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label(r'Fraction of $z{=}0$ BCG $M_*$ contributed')

    # Legend
    legend_elements = [
        Line2D([0], [0], color='steelblue', lw=2,
               label=r'BK08 $T_{\rm merge}$ ($z=0$)'),
        Line2D([0], [0], color='steelblue', lw=2, ls='--',
               label=r'BK08 $T_{\rm merge}$ ($z=1$)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='grey',
               markeredgecolor='grey', markersize=8, linestyle='None', label='Merger'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='grey',
               markeredgecolor='grey', markersize=8, linestyle='None', label='Disruption'),
        Patch(facecolor='firebrick', alpha=0.08,
              label=r'$< t_{\rm cross}(z{=}2)$'),
        Line2D([0], [0], color='firebrick', ls=':', lw=1.5, alpha=0.7,
               label=r'$t_{\rm cross}(z{=}0) = %.1f$ Gyr' % t_cross_z0),
        Line2D([0], [0], color='firebrick', ls='--', lw=1.5, alpha=0.7,
               label=r'$t_{\rm cross}(z{=}1) = %.1f$ Gyr' % t_cross_z1),
    ]
    x_max = min(np.nanpercentile(t_si, 99) + 1, 12)
    ax.set_xlabel(r'Time since infall [Gyr]')
    ax.set_ylabel(r'$\log_{10}(M_{\rm central} \,/\, M_{\rm sat,\,infall})$')
    ax.set_xlim(0, x_max)
    ax.set_ylim(3.0, -0.5)

    fig.tight_layout()
    ax.legend(handles=legend_elements, loc='upper center',
              bbox_to_anchor=(0.45, -0.12), ncol=6, fontsize=9,
              framealpha=0.9, columnspacing=1.2, handletextpad=0.5)
    return fig


def plot_lookback_of_infall(ev, lookback_fn, z_from_lb, H_gyr, age_now):
    """Plot 3: Mass ratio vs lookback time of infall."""
    valid = np.isfinite(ev['t_lb_infall']) & (ev['t_lb_infall'] > 0)
    if not np.any(valid):
        print("  No valid lookback-of-infall data; skipping plot.")
        return None

    ratio = ev['ratio'][valid]
    t_lb_inf = ev['t_lb_infall'][valid]
    frac = ev['frac'][valid]
    evtype = ev['evtype'][valid]
    smass = ev['smass'][valid]

    # ---- BK08 curves (same as plot 1, but now x-axis is infall lookback) ----
    t_grid = np.linspace(0.5, age_now - 0.5, 200)
    bk_median = BK08_boundary(t_grid, z_from_lb, H_gyr, eta=0.5)
    bk_lo = BK08_boundary(t_grid, z_from_lb, H_gyr, eta=0.3)
    bk_hi = BK08_boundary(t_grid, z_from_lb, H_gyr, eta=0.7)
    bk_2x = BK08_boundary(t_grid, z_from_lb, H_gyr, eta=0.5, factor=2.0)

    fig, ax = plt.subplots(figsize=(10, 7))

    # BK08 shaded region
    bk_valid = np.isfinite(bk_lo) & np.isfinite(bk_hi)
    ax.fill_between(t_grid[bk_valid], bk_lo[bk_valid], bk_hi[bk_valid],
                     color='steelblue', alpha=0.15, zorder=10)
    ax.plot(t_grid[bk_valid], bk_median[bk_valid], color='steelblue', lw=2, zorder=11)

    valid_2x = np.isfinite(bk_2x)
    ax.plot(t_grid[valid_2x], bk_2x[valid_2x], color='darkorange', lw=2, ls='--', zorder=11)

    # Crossing time reference lines
    A_bk, b_bk, c_bk, eta_ref = 0.216, 1.3, 1.9, 0.5
    tcross_lines = [
        (1,  ':',  r'$1\,t_{\rm cross}$'),
        (3,  '-.',  r'$3\,t_{\rm cross}$'),
        (10, '--', r'$10\,t_{\rm cross}$'),
    ]
    for N_tc, ls, label in tcross_lines:
        def _obj(log_r, target=N_tc):
            r = 10**log_r
            return A_bk * r**b_bk * np.exp(c_bk * eta_ref) / np.log(1 + r) - target
        try:
            log_r_val = brentq(_obj, -0.5, 5.0)
            ax.axhline(log_r_val, color='firebrick', ls=ls, lw=1.5, alpha=0.7, zorder=9)
        except ValueError:
            pass

    # Scatter points
    norm = mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=0.5)
    im = scatter_by_type(ax, t_lb_inf, ratio, frac, smass, evtype, norm, 'viridis')

    if im is not None:
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label(r'Fraction of $z{=}0$ BCG $M_*$ contributed')

    # Legend
    legend_elements = [
        Patch(facecolor='steelblue', alpha=0.15, edgecolor='steelblue',
              label=r'BK08 15--85\% $\eta$'),
        Line2D([0], [0], color='steelblue', lw=2, label=r'BK08 ($\eta = 0.5$)'),
        Line2D([0], [0], color='darkorange', lw=2, ls='--', label=r'$2 \times$ BK08'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='grey',
               markeredgecolor='grey', markersize=8, linestyle='None', label='Merger'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='grey',
               markeredgecolor='grey', markersize=8, linestyle='None', label='Disruption'),
        Line2D([0], [0], color='firebrick', ls=':', lw=1.5, alpha=0.7,
               label=r'$1\,t_{\rm cross}$'),
        Line2D([0], [0], color='firebrick', ls='-.', lw=1.5, alpha=0.7,
               label=r'$3\,t_{\rm cross}$'),
        Line2D([0], [0], color='firebrick', ls='--', lw=1.5, alpha=0.7,
               label=r'$10\,t_{\rm cross}$'),
    ]
    ax.set_xlabel(r'Lookback time of infall [Gyr]')
    ax.set_ylabel(r'$\log_{10}(M_{\rm central} \,/\, M_{\rm sat,\,infall})$')
    ax.set_xlim(0, age_now)
    ax.set_ylim(3.0, -0.5)

    # Redshift axis on top
    ax2 = ax.twiny()
    z_ticks = [0, 0.5, 1, 2, 3, 5, 8]
    t_ticks = [lookback_fn(z) for z in z_ticks]
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(t_ticks)
    ax2.set_xticklabels([f'{z}' for z in z_ticks])
    ax2.set_xlabel(r'$z_{\rm infall}$')

    fig.tight_layout()
    ax.legend(handles=legend_elements, loc='upper center',
              bbox_to_anchor=(0.45, -0.12), ncol=6, fontsize=9,
              framealpha=0.9, columnspacing=1.2, handletextpad=0.5)
    return fig


def print_diagnostics(ev, H_gyr):
    """Print diagnostic summary of merger/disruption events to terminal."""
    is_merger = ev['evtype'] == 0
    is_disrupt = ev['evtype'] == 1
    t_si = ev['t_since_infall']
    ratio = ev['ratio']  # log10
    halo_len = ev['len']
    valid_tsi = np.isfinite(t_si) & (t_si > 0)

    Delta_vir = 200.0
    t_cross_z0 = np.sqrt(2.0 / Delta_vir) / H_gyr(0.0)
    t_cross_z1 = np.sqrt(2.0 / Delta_vir) / H_gyr(1.0)
    t_cross_z2 = np.sqrt(2.0 / Delta_vir) / H_gyr(2.0)

    n_m = np.sum(is_merger)
    n_d = np.sum(is_disrupt)

    print("\n" + "=" * 70)
    print("  DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print(f"  Mergers:     {n_m:>6,}")
    print(f"  Disruptions: {n_d:>6,}")
    print(f"  Total:       {n_m + n_d:>6,}")

    # --- Halo length (resolution) ---
    valid_len = np.isfinite(halo_len)
    if np.any(valid_len):
        print(f"\n  --- Halo Len at event (resolution) ---")
        for label, mask in [("All", np.ones(len(halo_len), dtype=bool)),
                            ("Mergers", is_merger), ("Disruptions", is_disrupt)]:
            m = mask & valid_len
            if np.any(m):
                vals = halo_len[m]
                pcts = np.percentile(vals, [5, 25, 50, 75, 95])
                print(f"  {label:12s}  min={vals.min():.0f}  "
                      f"p5/25/50/75/95 = {pcts[0]:.0f}/{pcts[1]:.0f}/"
                      f"{pcts[2]:.0f}/{pcts[3]:.0f}/{pcts[4]:.0f}  "
                      f"max={vals.max():.0f}")
                n_low = np.sum(vals < 2 * MIN_LEN)
                print(f"  {'':12s}  Len < {2*MIN_LEN}: {n_low:,} ({100*n_low/len(vals):.1f}%)")

    # --- Time since infall ---
    print(f"\n  --- Time since infall [Gyr] ---")
    print(f"  t_cross(z=0) = {t_cross_z0:.2f} Gyr,  "
          f"t_cross(z=1) = {t_cross_z1:.2f} Gyr,  "
          f"t_cross(z=2) = {t_cross_z2:.2f} Gyr")
    for label, mask in [("Mergers", is_merger), ("Disruptions", is_disrupt)]:
        m = mask & valid_tsi
        if not np.any(m):
            print(f"  {label:12s}  no valid data")
            continue
        vals = t_si[m]
        pcts = np.percentile(vals, [5, 25, 50, 75, 95])
        print(f"  {label:12s}  min={vals.min():.2f}  "
              f"p5/25/50/75/95 = {pcts[0]:.1f}/{pcts[1]:.1f}/"
              f"{pcts[2]:.1f}/{pcts[3]:.1f}/{pcts[4]:.1f}  "
              f"max={vals.max():.1f}")
        n_fast_z0 = np.sum(vals < t_cross_z0)
        n_fast_z1 = np.sum(vals < t_cross_z1)
        n_total = len(vals)
        print(f"  {'':12s}  < t_cross(z=0): {n_fast_z0:,} ({100*n_fast_z0/n_total:.1f}%)  "
              f"< t_cross(z=1): {n_fast_z1:,} ({100*n_fast_z1/n_total:.1f}%)")

    # --- Mass ratio distribution ---
    print(f"\n  --- log10(M_central / M_sat,infall) ---")
    for label, mask in [("Mergers", is_merger), ("Disruptions", is_disrupt)]:
        m = mask & np.isfinite(ratio)
        if not np.any(m):
            continue
        vals = ratio[m]
        pcts = np.percentile(vals, [5, 25, 50, 75, 95])
        print(f"  {label:12s}  median={pcts[2]:.2f}  "
              f"p5/95 = {pcts[0]:.2f}/{pcts[4]:.2f}")

    # --- Cross-tab: fast disruptions by Len ---
    if np.any(valid_len):
        fast_disrupt = is_disrupt & valid_tsi & (t_si < t_cross_z0) & valid_len
        slow_disrupt = is_disrupt & valid_tsi & (t_si >= t_cross_z0) & valid_len
        if np.any(fast_disrupt) or np.any(slow_disrupt):
            print(f"\n  --- Disruption Len breakdown (t_cross(z=0) = {t_cross_z0:.2f} Gyr) ---")
            for label, mask in [("< t_cross", fast_disrupt), (">= t_cross", slow_disrupt)]:
                if np.any(mask):
                    vals = halo_len[mask]
                    print(f"  {label:12s}  N={np.sum(mask):>5,}  "
                          f"Len median={np.median(vals):.0f}  "
                          f"min={vals.min():.0f}  max={vals.max():.0f}  "
                          f"log10(M/m) median={np.median(ratio[mask]):.2f}")

    print("=" * 70 + "\n")


def main():
    setup_style()

    header, model_files = read_header(PRIMARY_DIR)
    lookback_fn, z_from_lb, H_gyr, age_now = make_cosmology(header)

    print("Collecting merger/disruption events...")
    ev = collect_events(model_files, header, lookback_fn)
    N = len(ev['ratio'])
    print(f"\nTotal events: {N:,}")

    # ---- Diagnostic summary ----
    print_diagnostics(ev, H_gyr)

    # Dilute for plotting (don't dilute 'len' — only needed for diagnostics above)
    arrays = dilute(N, ev['ratio'], ev['tlb'], ev['frac'], ev['smass'],
                    ev['evtype'], ev['t_since_infall'], ev['t_lb_infall'])
    ev['ratio'], ev['tlb'], ev['frac'], ev['smass'], ev['evtype'], ev['t_since_infall'], ev['t_lb_infall'] = arrays
    print(f"After dilution: {len(ev['ratio']):,}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Plot 1: BK08 comparison
    fig1 = plot_bk08_comparison(ev, lookback_fn, z_from_lb, H_gyr, age_now)
    outpath1 = os.path.join(OUTPUT_DIR, 'BK08_comparison.pdf')
    fig1.savefig(outpath1, dpi=200, bbox_inches='tight')
    print(f"\nSaved: {outpath1}")
    plt.close(fig1)

    # Plot 2: Time since infall
    fig2 = plot_time_since_infall(ev, H_gyr)
    if fig2 is not None:
        outpath2 = os.path.join(OUTPUT_DIR, 'time_since_infall.pdf')
        fig2.savefig(outpath2, dpi=200, bbox_inches='tight')
        print(f"Saved: {outpath2}")
        plt.close(fig2)

    # Plot 3: Lookback time of infall
    fig3 = plot_lookback_of_infall(ev, lookback_fn, z_from_lb, H_gyr, age_now)
    if fig3 is not None:
        outpath3 = os.path.join(OUTPUT_DIR, 'lookback_of_infall.pdf')
        fig3.savefig(outpath3, dpi=200, bbox_inches='tight')
        print(f"Saved: {outpath3}")
        plt.close(fig3)


if __name__ == '__main__':
    main()

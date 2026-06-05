#!/usr/bin/env python3
"""
sSFR vs cosmic time for galaxies selected by spatial_distribution.py.

Reads a CSV produced by spatial_distribution.py, locates the galaxies in the
model HDF5 files, and plots sSFR(t) for each galaxy as a separate coloured line.

Usage:
    python galaxy_history.py quiescent_galaxies_z3.00_millennium_loose_massive.csv output/millennium/
    python galaxy_history.py <csv> <folder> --output-dir plots/history/
    python galaxy_history.py <csv> <folder> --snapshot 27
"""

import argparse
import glob
import os
import re
import sys
import time

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator
from scipy.spatial import cKDTree

QUIESCENT_MODES = {
    'strict', 'loose', 'loose_massive', 'massive_bt', 'loose_massive_bt',
    'karls_galaxy', 'hubble', 'hubble_central',
}


# ---------------------------------------------------------------------------
# Cosmological utilities
# ---------------------------------------------------------------------------

def snapshot_times_gyr(redshifts, omega_m, omega_l, hubble_h):
    H0_inv_gyr = 9.778 / hubble_h
    def integrand(a):
        return 1.0 / (a * np.sqrt(omega_m / a**3 + omega_l))
    times = np.empty(len(redshifts))
    for i, z in enumerate(redshifts):
        t, _ = quad(integrand, 0.0, 1.0 / (1.0 + z))
        times[i] = H0_inv_gyr * t
    return times


# ---------------------------------------------------------------------------
# Header loading
# ---------------------------------------------------------------------------

def read_header(filepath):
    params = {}
    with h5.File(filepath, 'r') as f:
        sim = f['Header/Simulation']
        params['hubble_h']           = float(sim.attrs['hubble_h'])
        params['omega_matter']       = float(sim.attrs['omega_matter'])
        params['omega_lambda']       = float(sim.attrs['omega_lambda'])
        params['particle_mass']      = float(sim.attrs['particle_mass'])
        params['box_size']           = float(sim.attrs.get('box_size', 0.0))  # Mpc/h
        params['snapshot_redshifts'] = np.array(f['Header/snapshot_redshifts'])
        params['save_full_sfh']      = int(f['Header/Runtime'].attrs.get('SaveFullSFH', 0))
        snap_groups = [k for k in f.keys() if k.startswith('Snap_')]
        params['available_snapshots'] = sorted(
            int(s.replace('Snap_', '')) for s in snap_groups)
    return params


def find_snap_for_redshift(params, target_z):
    zs        = params['snapshot_redshifts']
    available = np.array(params['available_snapshots'])
    snap      = int(available[np.argmin(np.abs(zs[available] - target_z))])
    return snap, float(zs[snap])


# ---------------------------------------------------------------------------
# CSV metadata
# ---------------------------------------------------------------------------

def parse_csv_metadata(csv_path):
    base = os.path.splitext(os.path.basename(csv_path))[0]
    m = re.search(r'_z([\d.]+)_', base)
    if not m:
        return None, None, None
    z_sel   = float(m.group(1))
    rest    = base[m.end():]
    qmode, run_label = None, rest
    for candidate in sorted(QUIESCENT_MODES, key=len, reverse=True):
        if rest.endswith('_' + candidate):
            qmode     = candidate
            run_label = rest[:-(len(candidate) + 1)]
            break
    return z_sel, run_label, qmode


def load_csv(csv_path):
    import csv as _csv
    with open(csv_path, newline='') as fh:
        reader = _csv.DictReader(fh)
        rows = list(reader)
    if not rows:
        return {}
    data = {}
    for key in rows[0]:
        vals = [r[key] for r in rows]
        try:
            # Try int64 first — preserves large GalaxyIndex values that float64 rounds
            data[key] = np.array(vals, dtype=np.int64)
        except ValueError:
            try:
                data[key] = np.array(vals, dtype=np.float64)
            except ValueError:
                data[key] = np.array(vals)
    return data


# ---------------------------------------------------------------------------
# Galaxy SFH loading
# ---------------------------------------------------------------------------

def load_galaxy_sfh(filepaths, snap_num, galaxy_ids, hubble_h):
    """Load SFH arrays and stellar mass for a set of GalaxyIndex values."""
    id_set   = set(int(x) for x in galaxy_ids)
    snap_key = f'Snap_{snap_num}'
    results  = {}

    for fp in filepaths:
        fname = os.path.basename(fp)
        with h5.File(fp, 'r') as f:
            if snap_key not in f:
                print(f'  [sfh] {fname}: {snap_key} NOT FOUND — skipping')
                continue
            if 'GalaxyIndex' not in f[snap_key]:
                print(f'  [sfh] {fname}: GalaxyIndex NOT FOUND in {snap_key} — skipping')
                continue
            g    = f[snap_key]
            gids = np.array(g['GalaxyIndex'], dtype=np.int64)
            hit  = np.where(np.isin(gids, list(id_set)))[0]
            has_sfh = 'SFHMassDisk' in g and 'SFHMassBulge' in g
            print(f'  [sfh] {fname}: {len(gids):>7} gals in {snap_key}, '
                  f'{len(hit)} hits, SFH={"yes" if has_sfh else "NO"}')
            if len(hit) == 0:
                continue

            sm   = np.array(g['StellarMass'], dtype=np.float64)[hit]
            gt   = np.array(g['Type'], dtype=np.int32)[hit] if 'Type' in g else np.zeros(len(hit), dtype=np.int32)
            sfhd = np.array(g['SFHMassDisk'],  dtype=np.float64)[hit] if 'SFHMassDisk'  in g else None
            sfhb = np.array(g['SFHMassBulge'], dtype=np.float64)[hit] if 'SFHMassBulge' in g else None

            for i, raw in enumerate(hit):
                gid = int(gids[raw])
                results[gid] = {
                    'sm':    float(sm[i]),
                    'gtype': int(gt[i]),
                    'sfh':   (sfhd[i] + sfhb[i]) if sfhd is not None else None,
                }
    return results


# ---------------------------------------------------------------------------
# sSFR history computation
# ---------------------------------------------------------------------------

def compute_ssfr_history(sfh, snap_times, snap_num, hubble_h):
    """Return (cosmic_times, log10_ssfr) arrays of length snap_num + 1.

    SFH bin i covers the interval ending at snap_times[i], so the last bin
    ends at snap_times[snap_num-1].  We append snap_times[snap_num] (the
    selection epoch) carrying the final sSFR value so lines reach z_sel.
    """
    t_edges    = np.concatenate([[0.0], snap_times[:snap_num]])
    bin_widths = np.diff(t_edges)                               # Gyr
    sfh_msun   = sfh[:snap_num] * 1e10 / hubble_h              # Msun per bin

    sfr       = sfh_msun / (bin_widths * 1e9)                  # Msun/yr
    mstar_cum = np.cumsum(sfh_msun)                            # Msun (formed)

    with np.errstate(divide='ignore', invalid='ignore'):
        ssfr     = np.where(mstar_cum > 0, sfr / mstar_cum, np.nan)
        log_ssfr = np.where(ssfr > 0, np.log10(np.where(ssfr > 0, ssfr, 1.0)), np.nan)

    # Extend to the selection snapshot so lines reach z_sel exactly
    t         = np.append(snap_times[:snap_num], snap_times[snap_num])
    log_ssfr  = np.append(log_ssfr, log_ssfr[-1])
    return t, log_ssfr


def compute_smh(sfh, snap_times, snap_num, hubble_h):
    """Return (cosmic_times, log10_mstar_formed) arrays of length snap_num + 1."""
    sfh_msun  = sfh[:snap_num] * 1e10 / hubble_h
    mstar_cum = np.cumsum(sfh_msun)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_mstar = np.where(mstar_cum > 0,
                             np.log10(np.where(mstar_cum > 0, mstar_cum, 1.0)),
                             np.nan)
    t         = np.append(snap_times[:snap_num], snap_times[snap_num])
    log_mstar = np.append(log_mstar, log_mstar[-1])
    return t, log_mstar


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_ssfr_vs_cosmic_time(histories, snap_times, snap_num, hubble_h,
                              z_snap, run_label, qmode, out_path):
    """One line per galaxy, coloured by log M*, sSFR vs cosmic time."""
    # Filter to galaxies with SFH data
    gals = [(gid, d) for gid, d in histories.items() if d['sfh'] is not None]
    if not gals:
        print('  No SFH data — skipping plot.')
        return

    n = len(gals)
    log_sm_arr = np.array([
        np.log10(d['sm'] * 1e10 / hubble_h) if d['sm'] > 0 else np.nan
        for _, d in gals
    ])

    # Colour by log M* if range is meaningful, else use tab cycle
    sm_range = np.nanmax(log_sm_arr) - np.nanmin(log_sm_arr)
    if n <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, max(n, 1)))
        use_cbar = False
    elif sm_range > 0.3:
        norm   = plt.Normalize(vmin=np.nanmin(log_sm_arr),
                               vmax=np.nanmax(log_sm_arr))
        cmap   = plt.cm.viridis
        colors = [cmap(norm(lsm)) for lsm in log_sm_arr]
        use_cbar = True
    else:
        colors = plt.cm.viridis(np.linspace(0.15, 0.85, n))
        use_cbar = False

    t_snap = snap_times[snap_num]   # cosmic time at selection redshift

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Collect trajectory stats while we plot
    final_log_ssfr = []
    peak_log_ssfr  = []
    quench_t       = []     # cosmic time when log sSFR first dropped below -11
    for i, (_, d) in enumerate(gals):
        t, log_ssfr = compute_ssfr_history(d['sfh'], snap_times, snap_num, hubble_h)
        ax.plot(t, log_ssfr, color=colors[i], lw=1.4, alpha=0.85)
        fin = log_ssfr[np.isfinite(log_ssfr)]
        if fin.size:
            final_log_ssfr.append(log_ssfr[-1] if np.isfinite(log_ssfr[-1])
                                  else np.nan)
            peak_log_ssfr.append(np.nanmax(log_ssfr))
            below = np.where(np.isfinite(log_ssfr) & (log_ssfr < -11.0))[0]
            quench_t.append(t[below[0]] if below.size else np.nan)
        else:
            final_log_ssfr.append(np.nan)
            peak_log_ssfr.append(np.nan)
            quench_t.append(np.nan)

    print(f'  ── sSFR-history plot data ──')
    print(f'    N galaxies        : {len(gals)}')
    print(f'    log M*            : {_array_stats(log_sm_arr)}')
    print(f'    log sSFR @ z_sel  : {_array_stats(np.array(final_log_ssfr))}')
    print(f'    log sSFR peak     : {_array_stats(np.array(peak_log_ssfr))}')
    qt = np.array(quench_t)
    qfin = qt[np.isfinite(qt)]
    n_quench = int(qfin.size)
    if n_quench:
        print(f'    crossed sSFR<−11  : {n_quench}/{len(gals)}  '
              f'median t = {np.median(qfin):.2f} Gyr  '
              f'range [{qfin.min():.2f}, {qfin.max():.2f}] Gyr')
    else:
        print(f'    crossed sSFR<−11  : 0/{len(gals)} galaxies')

    # Mark the quiescent threshold and the selection epoch
    ax.axhline(-11, color='grey', ls='--', lw=1.0, alpha=0.7)
    ax.axvline(t_snap, color='white', ls=':', lw=1.0, alpha=0.5)

    ax.set_xlabel('Cosmic time [Gyr]', fontsize=13)
    ax.set_ylabel(r'$\log_{10}(\mathrm{sSFR}\;[\mathrm{yr}^{-1}])$', fontsize=13)
    ax.set_xlim(t[0], t_snap)

    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')

    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    if use_cbar:
        sm_sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
        sm_sm.set_array([])
        cbar = fig.colorbar(sm_sm, ax=ax, pad=0.02)
        cbar.set_label(r'$\log_{10}(M_*\,/\,\mathrm{M}_\odot)$', fontsize=11, color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.outline.set_edgecolor('white')
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


# ---------------------------------------------------------------------------
# Plot — stellar mass history
# ---------------------------------------------------------------------------

def plot_smh_vs_cosmic_time(histories, snap_times, snap_num, hubble_h,
                             z_snap, run_label, qmode, out_path):
    """One line per galaxy, coloured by log M*, stellar mass formed vs cosmic time."""
    gals = [(gid, d) for gid, d in histories.items() if d['sfh'] is not None]
    if not gals:
        print('  No SFH data — skipping plot.')
        return

    n = len(gals)
    log_sm_arr = np.array([
        np.log10(d['sm'] * 1e10 / hubble_h) if d['sm'] > 0 else np.nan
        for _, d in gals
    ])

    sm_range = np.nanmax(log_sm_arr) - np.nanmin(log_sm_arr)
    if n <= 10:
        colors   = plt.cm.tab10(np.linspace(0, 1, max(n, 1)))
        use_cbar = False
    elif sm_range > 0.3:
        norm     = plt.Normalize(vmin=np.nanmin(log_sm_arr),
                                 vmax=np.nanmax(log_sm_arr))
        cmap     = plt.cm.viridis
        colors   = [cmap(norm(lsm)) for lsm in log_sm_arr]
        use_cbar = True
    else:
        colors   = plt.cm.viridis(np.linspace(0.15, 0.85, n))
        use_cbar = False

    t_snap = snap_times[snap_num]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    final_log_mstar = []
    first_log_mstar = []
    growth_log_dex  = []
    for i, (_, d) in enumerate(gals):
        t, log_mstar = compute_smh(d['sfh'], snap_times, snap_num, hubble_h)
        ax.plot(t, log_mstar, color=colors[i], lw=1.4, alpha=0.85)
        fin = log_mstar[np.isfinite(log_mstar)]
        if fin.size:
            final_log_mstar.append(fin[-1])
            first_log_mstar.append(fin[0])
            growth_log_dex.append(fin[-1] - fin[0])
        else:
            final_log_mstar.append(np.nan)
            first_log_mstar.append(np.nan)
            growth_log_dex.append(np.nan)

    print(f'  ── SMH-history plot data ──')
    print(f'    N galaxies        : {len(gals)}')
    print(f'    log M*_formed @ z_sel : '
          f'{_array_stats(np.array(final_log_mstar))}')
    print(f'    log M*_formed @ first snap : '
          f'{_array_stats(np.array(first_log_mstar))}')
    print(f'    growth dex (final-first)   : '
          f'{_array_stats(np.array(growth_log_dex))}')

    ax.axvline(t_snap, color='white', ls=':', lw=1.0, alpha=0.5)

    ax.set_xlabel('Cosmic time [Gyr]', fontsize=13)
    ax.set_ylabel(r'$\log_{10}(M_*^{\rm formed}\;/\;\mathrm{M}_\odot)$', fontsize=13)
    ax.set_xlim(t[0], t_snap)

    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')

    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    if use_cbar:
        sm_sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
        sm_sm.set_array([])
        cbar = fig.colorbar(sm_sm, ax=ax, pad=0.02)
        cbar.set_label(r'$\log_{10}(M_*\,/\,\mathrm{M}_\odot)$', fontsize=11, color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.outline.set_edgecolor('white')
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


# ---------------------------------------------------------------------------
# Main-branch stellar mass history
# ---------------------------------------------------------------------------

def load_main_branch(filepaths, snap_sel, target_gid, snap_times, hubble_h):
    """Return [(t_gyr, log10_sm)] for target_gid at each snapshot 0..snap_sel."""
    branch = []
    for s in range(snap_sel + 1):
        snap_key = f'Snap_{s}'
        t        = snap_times[s]
        found    = False
        for fp in filepaths:
            with h5.File(fp, 'r') as f:
                if snap_key not in f or 'GalaxyIndex' not in f[snap_key]:
                    continue
                g    = f[snap_key]
                gids = np.array(g['GalaxyIndex'], dtype=np.int64)
                idx  = np.where(gids == target_gid)[0]
                if len(idx) > 0:
                    sm_msun = (float(np.array(g['StellarMass'],
                               dtype=np.float64)[idx[0]]) * 1e10 / hubble_h)
                    branch.append(
                        (t, np.log10(sm_msun) if sm_msun > 0 else np.nan))
                    found = True
                    break
            if found:
                break
    return branch


def load_most_massive_merger(filepaths, snap_sel, target_gid, snap_times, hubble_h,
                              n=None):
    """Return GalaxyIndex list of all (or top-n) mergers into target_gid by peak mass.

    Scans all snapshots for satellites (CentralGalaxyIndex == target_gid) that
    disappear before snap_sel (i.e. merged).  Returns them ranked by peak stellar
    mass, all of them when n=None.
    """
    sat_tracks    = {}   # gid -> [(t, log_sm), ...]
    sat_last_snap = {}   # gid -> last snapshot index it was seen

    for s in range(snap_sel + 1):
        snap_key = f'Snap_{s}'
        t        = snap_times[s]
        for fp in filepaths:
            with h5.File(fp, 'r') as f:
                if snap_key not in f or 'GalaxyIndex' not in f[snap_key]:
                    continue
                g = f[snap_key]
                if 'CentralGalaxyIndex' not in g:
                    continue
                gids   = np.array(g['GalaxyIndex'],        dtype=np.int64)
                cgi    = np.array(g['CentralGalaxyIndex'], dtype=np.int64)
                sm_arr = np.array(g['StellarMass'],        dtype=np.float64)

                sat_idx = np.where((cgi == target_gid) & (gids != target_gid))[0]
                for si in sat_idx:
                    sgid    = int(gids[si])
                    sm_msun = float(sm_arr[si]) * 1e10 / hubble_h
                    if sgid not in sat_tracks:
                        sat_tracks[sgid] = []
                    sat_tracks[sgid].append(
                        (t, np.log10(sm_msun) if sm_msun > 0 else np.nan))
                    sat_last_snap[sgid] = s

    # Satellites that disappeared before snap_sel have merged
    merged = {gid: s for gid, s in sat_last_snap.items() if s < snap_sel}
    if not merged:
        print('  No mergers found before selection snapshot.')
        return None

    def _peak(gid):
        vals = [sm for _, sm in sat_tracks[gid] if not np.isnan(sm)]
        return max(vals) if vals else -np.inf

    ranked = sorted(merged, key=_peak, reverse=True)[:n]  # n=None → all
    for gid in ranked:
        print(f'  Merger: GalaxyIndex={gid}  '
              f'last seen Snap_{merged[gid]}  '
              f'peak log M*={_peak(gid):.2f}')
    return ranked


def _smooth_line(ax, track, color, lw, alpha=0.95, label=None):
    """Parametric PCHIP smooth through a [(t, log_sm)] track."""
    ts_  = np.array([p[0] for p in track], dtype=float)
    sms_ = np.array([p[1] for p in track], dtype=float)
    mask = ~np.isnan(sms_)
    ts_, sms_ = ts_[mask], sms_[mask]
    if len(ts_) < 2:
        return
    if len(ts_) >= 3:
        param  = np.arange(len(ts_), dtype=float)
        p_fine = np.linspace(0, param[-1], 300)
        t_out  = PchipInterpolator(param, ts_)(p_fine)
        sm_out = PchipInterpolator(param, sms_)(p_fine)
        ax.plot(t_out, sm_out, color=color, lw=lw, alpha=alpha, label=label)
    else:
        ax.plot(ts_, sms_, color=color, lw=lw, alpha=alpha, label=label)


def plot_main_branch_history(branch, snap_times, snap_sel, z_snap,
                              run_label, qmode, out_path,
                              galaxy_id=None, merger_branches=None):
    """Stellar mass vs cosmic time — main galaxy plus optional merger branches."""
    if not branch:
        print('  No data — skipping main-branch history.')
        return

    t_snap       = snap_times[snap_sel]
    t_min        = branch[0][0]
    main_sm_at_t = {t: sm for t, sm in branch}

    # --- Main-branch trajectory milestones ---
    bt_arr = np.array([(t, sm) for t, sm in branch], dtype=float)
    fin    = np.isfinite(bt_arr[:, 1])
    if fin.any():
        first_t, first_sm = bt_arr[fin, 0][0], bt_arr[fin, 1][0]
        last_t,  last_sm  = bt_arr[fin, 0][-1], bt_arr[fin, 1][-1]
        peak_i = int(np.argmax(bt_arr[:, 1]))
        peak_t, peak_sm = bt_arr[peak_i, 0], bt_arr[peak_i, 1]
        print(f'  ── Main-branch trajectory '
              f'(GID {galaxy_id}) ──')
        print(f'    first detected   : t = {first_t:.2f} Gyr  '
              f'log M* = {first_sm:.2f}')
        print(f'    peak log M*      : t = {peak_t:.2f} Gyr  '
              f'log M* = {peak_sm:.2f}')
        print(f'    at selection snap: t = {last_t:.2f} Gyr  '
              f'log M* = {last_sm:.2f}')
        print(f'    growth (last-first): {last_sm - first_sm:+.2f} dex over '
              f'{last_t - first_t:.2f} Gyr')
    if merger_branches:
        peak_logsm = [max((sm for _, sm in mb if np.isfinite(sm)),
                          default=np.nan)
                      for mb in merger_branches]
        valid_peaks = [p for p in peak_logsm if np.isfinite(p)]
        if valid_peaks:
            print(f'    mergers ({len(valid_peaks)} traced): '
                  f'peak log M* range [{min(valid_peaks):.2f}, '
                  f'{max(valid_peaks):.2f}], '
                  f'median {np.median(valid_peaks):.2f}')

    # Colour each merger by its peak stellar mass
    cmap = plt.cm.plasma_r
    peak_sms = []
    for mb in (merger_branches or []):
        vals = [sm for _, sm in mb if mb and not np.isnan(sm)]
        peak_sms.append(max(vals) if vals else np.nan)
    valid = [v for v in peak_sms if not np.isnan(v)]
    if len(valid) > 1:
        norm     = plt.Normalize(vmin=min(valid), vmax=max(valid))
        colours  = [cmap(norm(p)) if not np.isnan(p) else 'grey' for p in peak_sms]
        use_cbar = True
    else:
        colours  = [cmap(0.5)] * len(peak_sms)
        use_cbar = False
        norm     = None

    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    main_label = f'GalaxyIndex {galaxy_id}' if galaxy_id is not None else 'Most massive galaxy'
    _smooth_line(ax, branch, color='white', lw=1.2, label=main_label)

    for i, mb in enumerate(merger_branches or []):
        if not mb:
            continue
        col = colours[i]
        _smooth_line(ax, mb, color=col, lw=0.4,
                     alpha=0.85)
        t_min = min(t_min, mb[0][0])

        # Vertical dashed line from merger galaxy's last point up to main branch
        t_end, sm_end = mb[-1]
        main_sm = main_sm_at_t.get(t_end)
        if main_sm is not None and not np.isnan(sm_end) and not np.isnan(main_sm):
            ax.plot([t_end, t_end], [sm_end, main_sm],
                    color=col, lw=0.4, ls=':', alpha=0.7)

    ax.axvline(t_snap, color='white', ls=':', lw=1.0, alpha=0.5)
    ax.set_xlim(t_min, t_snap)

    ax.set_xlabel('Cosmic time [Gyr]', fontsize=13)
    ax.set_ylabel(r'$\log_{10}(M_*\,/\,\mathrm{M}_\odot)$', fontsize=13)


    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')

    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    handles, _ = ax.get_legend_handles_labels()
    if handles:
        leg = ax.legend(fontsize=9, loc='upper left')
        for text in leg.get_texts():
            text.set_color('white')
        leg.get_frame().set_facecolor('black')
        leg.get_frame().set_edgecolor('white')

    if use_cbar and norm is not None:
        sm_sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm_sm.set_array([])
        cbar = fig.colorbar(sm_sm, ax=ax, pad=0.02)
        cbar.set_label(r'$\log_{10}(M_*^{\rm merger,\,peak}\,/\,\mathrm{M}_\odot)$',
                       fontsize=10, color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.outline.set_edgecolor('white')
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


# ---------------------------------------------------------------------------
# Shared stats helper
# ---------------------------------------------------------------------------

def _print_panel_stats(label, y, width=22):
    """Print median / mean / std / [min, max] for a panel's y-axis data."""
    if y is None:
        print(f'  {label:<{width}}: no data')
        return
    ok = np.isfinite(y)
    n  = ok.sum()
    if n == 0:
        print(f'  {label:<{width}}: all NaN / no finite values')
        return
    v = y[ok]
    print(f'  {label:<{width}}: N={n:>5}  '
          f'median={np.median(v):>8.3f}  mean={np.mean(v):>8.3f}  '
          f'std={np.std(v):>7.3f}  [{np.min(v):.3f}, {np.max(v):.3f}]')


# ---------------------------------------------------------------------------
# Scaling relations grid
# ---------------------------------------------------------------------------

def plot_scaling_relations(csv_data, hubble_h, z_snap, run_label, qmode, out_path):
    """4×2 grid of galaxy scaling relations from the quiescent catalogue."""
    def _col(key):
        v = csv_data.get(key)
        return v if v is not None and len(v) > 0 else None

    log_sm = _col('log_StellarMass_Msun')
    if log_sm is None:
        print('  No stellar mass data — skipping scaling relations.')
        return

    # Index of the most massive target — overlaid as a gold star per panel
    finite_sm = np.isfinite(log_sm)
    if finite_sm.any():
        idx_max = int(np.argmax(np.where(finite_sm, log_sm, -np.inf)))
    else:
        idx_max = None

    log_mv = _col('log_Mvir_Msun')
    log_bm = _col('log_BulgeMass_Msun')
    log_bh = _col('log_BlackHoleMass_Msun')
    bt     = _col('bulge_to_total')
    vvir   = _col('Vvir')
    age    = _col('mw_stellar_age_Gyr')

    sfrd = _col('SfrDisk')
    sfrb = _col('SfrBulge')
    sfr  = (sfrd + sfrb) if (sfrd is not None and sfrb is not None) else sfrd

    sm_msun = 10.0 ** log_sm
    with np.errstate(divide='ignore', invalid='ignore'):
        log_sfr  = (np.where(sfr  > 0, np.log10(np.where(sfr  > 0, sfr,  1.0)), np.nan)
                    if sfr is not None else None)
        ssfr_arr = (np.where(sm_msun > 0, sfr / sm_msun, np.nan)
                    if sfr is not None else None)
        log_ssfr = (np.where(ssfr_arr > 0,
                              np.log10(np.where(ssfr_arr > 0, ssfr_arr, 1.0)), np.nan)
                    if ssfr_arr is not None else None)
        log_vvir = (np.where(vvir > 0, np.log10(np.where(vvir > 0, vvir, 1.0)), np.nan)
                    if vvir is not None else None)

    metals_cg = _col('MetalsColdGas')
    cg        = _col('ColdGas')
    if metals_cg is not None and cg is not None:
        # 12 + log(O/H): assume solar abundance pattern (f_O = 0.426, Asplund+09)
        # O/H = (f_O * Z / A_O) / X_H,  A_O=16,  X_H≈0.76
        # Solar: 12 + log(O/H)_sun = 8.69
        f_O = 0.426
        X_H = 0.76
        with np.errstate(divide='ignore', invalid='ignore'):
            Z_frac   = np.where(cg > 0, metals_cg / cg, np.nan)
            OH       = np.where(Z_frac > 0, f_O * Z_frac / (16.0 * X_H), np.nan)
            log_Z    = np.where(OH > 0, 12.0 + np.log10(OH), np.nan)
    else:
        log_Z = None

    panels = [
        ('Stellar mass [log Msun]',  log_mv, log_sm,
         r'$\log_{10}(M_{\rm vir}\,/\,\mathrm{M}_\odot)$',
         r'$\log_{10}(m_*\,/\,\mathrm{M}_\odot)$'),
        ('SFR [log Msun/yr]',       log_sm, log_sfr,
         r'$\log_{10}(m_*\,/\,\mathrm{M}_\odot)$',
         r'$\log_{10}(\mathrm{SFR}\;[\mathrm{M}_\odot\,\mathrm{yr}^{-1}])$'),
        ('sSFR [log /yr]',          log_sm, log_ssfr,
         r'$\log_{10}(m_*\,/\,\mathrm{M}_\odot)$',
         r'$\log_{10}(\mathrm{sSFR}\;[\mathrm{yr}^{-1}])$'),
        ('Cold gas [12+log(O/H)]',  log_sm, log_Z,
         r'$\log_{10}(m_*\,/\,\mathrm{M}_\odot)$',
         r'$12 + \log_{10}(\mathrm{O/H})$'),
        ('Bulge mass [log Msun]',   log_sm, log_bm,
         r'$\log_{10}(m_*\,/\,\mathrm{M}_\odot)$',
         r'$\log_{10}(m_{\rm bulge}\,/\,\mathrm{M}_\odot)$'),
        ('B/T',                     log_sm, bt,
         r'$\log_{10}(m_*\,/\,\mathrm{M}_\odot)$',
         r'$B/T$'),
        ('BH mass [log Msun]',      log_sm, log_bh,
         r'$\log_{10}(m_*\,/\,\mathrm{M}_\odot)$',
         r'$\log_{10}(m_{\rm BH}\,/\,\mathrm{M}_\odot)$'),
        ('Vvir [log km/s]',         log_sm, log_vvir,
         r'$\log_{10}(m_*\,/\,\mathrm{M}_\odot)$',
         r'$\log_{10}(V_{\rm vir}\;[\mathrm{km\,s}^{-1}])$'),
    ]

    # Terminal statistics — stats per panel (stellar mass is printed once
    # by the panel loop since it appears as the first panel's y-axis)
    n_gals = len(log_sm)
    print(f'\n=== Scaling relations  z={z_snap:.2f}  {run_label}  [{qmode}]  N={n_gals} ===')
    for label, _x, y, *_ in panels:
        _print_panel_stats(label, y)

    # Colour points by mass-weighted stellar age when available
    use_cbar = False
    norm_cb  = None
    cmap_cb  = None
    c_all    = 'cyan'
    if age is not None:
        age_fin = np.where(np.isfinite(age), age, np.nan)
        finite  = age_fin[np.isfinite(age_fin)]
        if len(finite) > 1:
            norm_cb  = plt.Normalize(vmin=finite.min(), vmax=finite.max())
            cmap_cb  = plt.cm.plasma
            c_mapped = np.where(np.isfinite(age_fin), norm_cb(age_fin), 0.5)
            c_all    = cmap_cb(c_mapped)   # (N, 4) RGBA
            use_cbar = True

    n_gals = len(log_sm)
    ms = max(4, min(50, 200 // max(1, n_gals)))

    fig, axes = plt.subplots(4, 2, figsize=(10, 14))
    fig.patch.set_facecolor('black')

    for ax, (_lbl, x, y, xl, yl) in zip(axes.flat, panels):
        ax.set_facecolor('black')
        ax.tick_params(colors='white', labelsize=8)
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
        ax.set_xlabel(xl, fontsize=9)
        ax.set_ylabel(yl, fontsize=9)

        if x is None or y is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    color='grey', fontsize=9, transform=ax.transAxes)
            continue

        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() == 0:
            ax.text(0.5, 0.5, 'No finite data', ha='center', va='center',
                    color='grey', fontsize=9, transform=ax.transAxes)
            continue

        sc_c = c_all[ok] if isinstance(c_all, np.ndarray) else c_all
        ax.scatter(x[ok], y[ok], c=sc_c, s=ms, alpha=0.8, linewidths=0)

        # Gold star for the most massive target in the catalogue
        if (idx_max is not None and ok[idx_max]):
            ax.scatter(x[idx_max], y[idx_max], s=180, marker='*',
                       facecolor='gold', edgecolor='black',
                       linewidths=0.8, zorder=5)

    if use_cbar:
        fig.tight_layout(rect=[0, 0.05, 1, 1])
        cax = fig.add_axes([0.15, 0.01, 0.70, 0.018])
        sm_cb = plt.cm.ScalarMappable(cmap=cmap_cb, norm=norm_cb)
        sm_cb.set_array([])
        cbar = fig.colorbar(sm_cb, cax=cax, orientation='horizontal')
        cbar.set_label('Mass-weighted stellar age [Gyr]', fontsize=10, color='white')
        cbar.ax.xaxis.set_tick_params(color='white')
        cbar.outline.set_edgecolor('white')
        plt.setp(cbar.ax.xaxis.get_ticklabels(), color='white')
    else:
        fig.tight_layout()

    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


# ---------------------------------------------------------------------------
# Structural and kinematic properties grid
# ---------------------------------------------------------------------------

def plot_structural_kinematics(csv_data, hubble_h, z_snap, run_label, qmode, out_path):
    """2×2 grid: disk radius, bulge radius, velocity dispersion, circular velocity."""
    def _col(key):
        v = csv_data.get(key)
        return v if v is not None and len(v) > 0 else None

    log_sm = _col('log_StellarMass_Msun')
    # Index of the most massive target — overlaid as a gold star per panel
    if log_sm is not None:
        finite_sm = np.isfinite(log_sm)
        idx_max = (int(np.argmax(np.where(finite_sm, log_sm, -np.inf)))
                   if finite_sm.any() else None)
    else:
        idx_max = None
    if log_sm is None:
        print('  No stellar mass data — skipping structural/kinematic plot.')
        return

    def _to_log_kpc(raw):
        if raw is None:
            return None
        kpc = raw * 1e3 / hubble_h
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(kpc > 0, np.log10(kpc), np.nan)

    log_disk  = _to_log_kpc(_col('DiskRadius'))
    log_bulge = _to_log_kpc(_col('BulgeRadius'))

    vdisp = _col('VelDisp')
    with np.errstate(divide='ignore', invalid='ignore'):
        log_vdisp = (np.where(vdisp > 0, np.log10(vdisp), np.nan)
                     if vdisp is not None else None)

    # Baryonic circular velocity at the disk scale radius:
    #   V_c = sqrt(G * M_baryons / R_disk)
    #   G = 4.302e-6 kpc (km/s)^2 / Msun
    sm_raw   = _col('StellarMass')   # 10^10 Msun/h
    cg_raw   = _col('ColdGas')       # 10^10 Msun/h
    dr_raw   = _col('DiskRadius')    # Mpc/h
    log_vc = None
    if sm_raw is not None and dr_raw is not None:
        m_bar_msun = (sm_raw + (cg_raw if cg_raw is not None else 0.0)) * 1e10 / hubble_h
        r_disk_kpc = dr_raw * 1e3 / hubble_h
        G_kpc = 4.302e-6  # kpc (km/s)^2 / Msun
        with np.errstate(divide='ignore', invalid='ignore'):
            vc2 = np.where((m_bar_msun > 0) & (r_disk_kpc > 0),
                           G_kpc * m_bar_msun / r_disk_kpc, np.nan)
            vc  = np.where(vc2 > 0, np.sqrt(vc2), np.nan)
            log_vc = np.where(vc > 0, np.log10(vc), np.nan)

    age = _col('mw_stellar_age_Gyr')

    panels = [
        ('Disk radius [log kpc]',      log_sm, log_disk,
         r'$\log_{10}(m_*\,/\,\mathrm{M}_\odot)$',
         r'$\log_{10}(R_{\rm disk}\;[\mathrm{kpc}])$'),
        ('Bulge radius [log kpc]',     log_sm, log_bulge,
         r'$\log_{10}(m_*\,/\,\mathrm{M}_\odot)$',
         r'$\log_{10}(R_{\rm bulge}\;[\mathrm{kpc}])$'),
        ('Vel. dispersion [log km/s]', log_sm, log_vdisp,
         r'$\log_{10}(m_*\,/\,\mathrm{M}_\odot)$',
         r'$\log_{10}(\sigma_v\;[\mathrm{km\,s}^{-1}])$'),
        ('Baryonic V_c [log km/s]',    log_sm, log_vc,
         r'$\log_{10}(m_*\,/\,\mathrm{M}_\odot)$',
         r'$\log_{10}(V_c^{\rm bar}\;[\mathrm{km\,s}^{-1}])$'),
    ]

    # Terminal statistics
    n_gals = len(log_sm)
    sm_fin = log_sm[np.isfinite(log_sm)]
    print(f'\n=== Structural & kinematics  z={z_snap:.2f}  {run_label}  [{qmode}]  N={n_gals} ===')
    print(f'  {"Stellar mass [log Msun]":<28}: N={len(sm_fin):>5}  '
          f'median={np.median(sm_fin):>8.3f}  mean={np.mean(sm_fin):>8.3f}  '
          f'std={np.std(sm_fin):>7.3f}  [{np.min(sm_fin):.3f}, {np.max(sm_fin):.3f}]')
    for label, _x, y, *_ in panels:
        _print_panel_stats(label, y, width=28)

    # Physical (non-log) disk and bulge radii for convenience
    if log_disk is not None:
        r_kpc = 10.0 ** log_disk[np.isfinite(log_disk)]
        if len(r_kpc):
            print(f'  {"  Disk radius [kpc, linear]":<28}: '
                  f'median={np.median(r_kpc):.2f}  '
                  f'[{np.min(r_kpc):.2f}, {np.max(r_kpc):.2f}]')
    if log_bulge is not None:
        r_kpc = 10.0 ** log_bulge[np.isfinite(log_bulge)]
        if len(r_kpc):
            print(f'  {"  Bulge radius [kpc, linear]":<28}: '
                  f'median={np.median(r_kpc):.2f}  '
                  f'[{np.min(r_kpc):.2f}, {np.max(r_kpc):.2f}]')

    use_cbar = False
    norm_cb  = None
    cmap_cb  = None
    c_all    = 'cyan'
    if age is not None:
        age_fin = np.where(np.isfinite(age), age, np.nan)
        finite  = age_fin[np.isfinite(age_fin)]
        if len(finite) > 1:
            norm_cb  = plt.Normalize(vmin=finite.min(), vmax=finite.max())
            cmap_cb  = plt.cm.plasma
            c_mapped = np.where(np.isfinite(age_fin), norm_cb(age_fin), 0.5)
            c_all    = cmap_cb(c_mapped)
            use_cbar = True

    n_gals = len(log_sm)
    ms = max(4, min(50, 200 // max(1, n_gals)))

    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    fig.patch.set_facecolor('black')

    for ax, (_lbl, x, y, xl, yl) in zip(axes.flat, panels):
        ax.set_facecolor('black')
        ax.tick_params(colors='white', labelsize=9)
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
        ax.set_xlabel(xl, fontsize=10)
        ax.set_ylabel(yl, fontsize=10)

        if x is None or y is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    color='grey', fontsize=10, transform=ax.transAxes)
            continue

        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() == 0:
            ax.text(0.5, 0.5, 'No finite data', ha='center', va='center',
                    color='grey', fontsize=10, transform=ax.transAxes)
            continue

        sc_c = c_all[ok] if isinstance(c_all, np.ndarray) else c_all
        ax.scatter(x[ok], y[ok], c=sc_c, s=ms, alpha=0.8, linewidths=0)

        # Gold star for the most massive target in the catalogue
        if (idx_max is not None and ok[idx_max]):
            ax.scatter(x[idx_max], y[idx_max], s=180, marker='*',
                       facecolor='gold', edgecolor='black',
                       linewidths=0.8, zorder=5)

    if use_cbar:
        fig.tight_layout(rect=[0, 0.08, 1, 1])
        cax = fig.add_axes([0.15, 0.02, 0.70, 0.025])
        sm_cb = plt.cm.ScalarMappable(cmap=cmap_cb, norm=norm_cb)
        sm_cb.set_array([])
        cbar = fig.colorbar(sm_cb, cax=cax, orientation='horizontal')
        cbar.set_label('Mass-weighted stellar age [Gyr]', fontsize=10, color='white')
        cbar.ax.xaxis.set_tick_params(color='white')
        cbar.outline.set_edgecolor('white')
        plt.setp(cbar.ax.xaxis.get_ticklabels(), color='white')
    else:
        fig.tight_layout()

    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


# ---------------------------------------------------------------------------
# Environment analysis
# ---------------------------------------------------------------------------

def load_snapshot_all_galaxies(filepaths, snap_num, particle_mass, hubble_h):
    """Load minimal fields for every galaxy at snap_num (for environment analysis).

    Positions returned in Mpc, masses in Msun.
    """
    snap_key = f'Snap_{snap_num}'
    bufs = {k: [] for k in ('gid', 'cgi', 'gtype',
                              'posx', 'posy', 'posz',
                              'sm', 'mvir', 'cmvir', 'bm', 'sfr')}
    n_files = len(filepaths)
    progress_step = max(1, n_files // 10)
    t0 = time.time()
    total_gals = 0

    for fi, fp in enumerate(filepaths):
        with h5.File(fp, 'r') as f:
            if snap_key not in f or 'StellarMass' not in f[snap_key]:
                continue
            g    = f[snap_key]
            sm_r = np.array(g['StellarMass'], dtype=np.float64)
            mask = sm_r >= particle_mass
            if mask.sum() == 0:
                continue
            n_kept = int(mask.sum())
            total_gals += n_kept

            def _l(field, dtype=np.float64, fill=0.0):
                return (np.array(g[field], dtype=dtype)[mask]
                        if field in g else np.full(mask.sum(), fill, dtype=dtype))

            bufs['gid'].append(  _l('GalaxyIndex',        np.int64,  -1))
            bufs['cgi'].append(  _l('CentralGalaxyIndex', np.int64,  -1))
            bufs['gtype'].append(_l('Type',               np.int32,   0))
            bufs['posx'].append( _l('Posx') / hubble_h)          # Mpc
            bufs['posy'].append( _l('Posy') / hubble_h)
            bufs['posz'].append( _l('Posz') / hubble_h)
            bufs['sm'].append(   sm_r[mask] * 1e10 / hubble_h)   # Msun
            bufs['mvir'].append( _l('Mvir')        * 1e10 / hubble_h)
            bufs['cmvir'].append(_l('CentralMvir') * 1e10 / hubble_h)
            bufs['bm'].append(   _l('BulgeMass')   * 1e10 / hubble_h)
            bufs['sfr'].append(  _l('SfrDisk') + _l('SfrBulge'))

        if ((fi + 1) % progress_step == 0) or (fi + 1 == n_files):
            print(f'    file {fi+1:>4d}/{n_files}  '
                  f'cumulative galaxies={total_gals:,}  '
                  f'[{_fmt_dt(time.time()-t0)}]')

    if not bufs['gid']:
        return None
    return {k: np.concatenate(v) for k, v in bufs.items()}


def _env_class(cmvir_msun):
    log_m = np.log10(cmvir_msun) if cmvir_msun > 0 else 0.0
    if log_m >= 14.0:
        return 'Cluster'
    elif log_m >= 12.5:
        return 'Group'
    else:
        return 'Isolated'


SM_DENSITY_THRESHOLD = 1e10   # Msun — fixed mass cut for overdensity counts
SEARCH_RADII_MPC     = (1.0, 5.0)  # inner and outer aperture radii
KERNEL_SIGMAS_MPC    = (3.0, 5.0)  # Gaussian kernel scales (cMpc) for mass-weighted overdensity
HALO_MASS_FLOOR_MSUN = 10.0 ** 10.5  # resolution floor — halos below this are excluded from
                                      # the kernel field (under-resolved in typical N-body setups)


def compute_gaussian_kernel_overdensity(target_pos, halo_pos, halo_mass,
                                         sigma_mpc, box_size_mpc=None,
                                         truncation_sigma=4.0,
                                         target_self_mass=None):
    """Mass-weighted 3D Gaussian kernel overdensity at each target position.

    For each target i:
        rho(x_i) = (2 pi sigma^2)^(-3/2) * sum_j m_j * exp(-|x_i - x_j|^2 / (2 sigma^2))
    where the sum runs over halos within ``truncation_sigma * sigma`` of the
    target.  If the target's own halo is part of the field its self mass
    contributes a delta-function spike at d=0, so ``target_self_mass[i]`` is
    subtracted from rho before normalising.

    Returns delta = rho / rho_mean - 1, with rho_mean = sum(m_j) / V_box.  If
    the box size is unknown rho_mean is undefined and NaN is returned.
    """
    n = len(target_pos)
    if n == 0 or len(halo_pos) == 0:
        return np.full(n, np.nan)

    halo_pos  = np.asarray(halo_pos,  dtype=np.float64)
    halo_mass = np.asarray(halo_mass, dtype=np.float64)
    norm_3d   = 1.0 / (2.0 * np.pi * sigma_mpc ** 2) ** 1.5
    search_r  = truncation_sigma * sigma_mpc

    if box_size_mpc:
        pts       = np.mod(halo_pos,  box_size_mpc)
        query_pts = np.mod(target_pos, box_size_mpc)
        tree      = cKDTree(pts, boxsize=box_size_mpc)
        rho_mean  = float(np.sum(halo_mass)) / box_size_mpc ** 3
    else:
        tree      = cKDTree(halo_pos)
        query_pts = np.asarray(target_pos, dtype=np.float64)
        rho_mean  = None

    deltas = np.full(n, np.nan)
    for i in range(n):
        idx = tree.query_ball_point(query_pts[i], search_r)
        if not idx:
            if rho_mean is not None and rho_mean > 0:
                deltas[i] = -1.0
            continue
        dx = halo_pos[idx] - target_pos[i]
        if box_size_mpc:
            dx -= box_size_mpc * np.round(dx / box_size_mpc)
        d2        = np.sum(dx * dx, axis=1)
        weights   = np.exp(-d2 / (2.0 * sigma_mpc * sigma_mpc))
        rho_local = norm_3d * float(np.sum(halo_mass[idx] * weights))
        if target_self_mass is not None:
            rho_local -= norm_3d * float(target_self_mass[i])
        if rho_mean is not None and rho_mean > 0:
            deltas[i] = rho_local / rho_mean - 1.0
    return deltas


def compute_environment(target_gids, all_gals, box_size_mpc=None):
    """FoF group membership + aperture overdensity at two radii for each target.

    Overdensity uses only galaxies above SM_DENSITY_THRESHOLD so the mean
    field density is independent of the particle-mass floor.
    Returns a list of result dicts, one per target GalaxyIndex.
    """
    # Mask for galaxies above the fixed mass threshold (for density counts)
    dense_mask = all_gals['sm'] >= SM_DENSITY_THRESHOLD
    n_dense     = int(dense_mask.sum())

    # Pre-compute mean field densities for each radius
    if box_size_mpc and n_dense > 0:
        mean_density = n_dense / box_size_mpc ** 3   # gal / Mpc³
    else:
        mean_density = None

    def _overdensity(n_local, radius):
        vol = (4.0 / 3.0) * np.pi * radius ** 3
        if mean_density and mean_density > 0:
            return (n_local / vol) / mean_density - 1.0
        return np.nan

    gid_to_idx = {int(g): i for i, g in enumerate(all_gals['gid'])}
    n_all      = len(all_gals['gid'])

    # --- Mass-weighted Gaussian kernel overdensity ---
    # Field: ALL halos (centrals + satellites + orphans) above the resolution
    # floor, weighted by their own Mvir.
    halo_mask = all_gals['mvir'] >= HALO_MASS_FLOOR_MSUN
    halo_pos  = np.column_stack([all_gals['posx'][halo_mask],
                                  all_gals['posy'][halo_mask],
                                  all_gals['posz'][halo_mask]])
    halo_mass = all_gals['mvir'][halo_mask]

    valid_tgids, tgt_pos, tgt_self_m = [], [], []
    for tgid in target_gids:
        tgid_i = int(tgid)
        if tgid_i not in gid_to_idx:
            continue
        idx = gid_to_idx[tgid_i]
        valid_tgids.append(tgid_i)
        tgt_pos.append([float(all_gals['posx'][idx]),
                        float(all_gals['posy'][idx]),
                        float(all_gals['posz'][idx])])
        # Self mass only when the target's Mvir clears the resolution floor
        # (every such halo is in the field — see HALO_MASS_FLOOR_MSUN).
        mvir_i = float(all_gals['mvir'][idx])
        tgt_self_m.append(mvir_i if mvir_i >= HALO_MASS_FLOOR_MSUN else 0.0)
    tgt_pos    = np.asarray(tgt_pos,    dtype=np.float64)
    tgt_self_m = np.asarray(tgt_self_m, dtype=np.float64)

    kernel_deltas = {}
    for sigma in KERNEL_SIGMAS_MPC:
        d = compute_gaussian_kernel_overdensity(
            tgt_pos, halo_pos, halo_mass, sigma,
            box_size_mpc=box_size_mpc, target_self_mass=tgt_self_m)
        kernel_deltas[sigma] = dict(zip(valid_tgids, d))

    results = []
    n_targets = len(target_gids)
    progress_step = max(1, n_targets // 10)
    t_loop = time.time()

    for k, tgid in enumerate(target_gids):
        tgid = int(tgid)
        if tgid not in gid_to_idx:
            results.append({'gid': tgid, 'found': False})
            continue

        if k > 0 and (k % progress_step == 0):
            print(f'    target {k:>5d}/{n_targets}  '
                  f'({100.0*k/n_targets:.0f}%)  '
                  f'[{_fmt_dt(time.time()-t_loop)} elapsed]')

        idx   = gid_to_idx[tgid]
        tx    = float(all_gals['posx'][idx])
        ty    = float(all_gals['posy'][idx])
        tz    = float(all_gals['posz'][idx])
        cgi   = int(all_gals['cgi'][idx])
        cmvir = float(all_gals['cmvir'][idx])

        # --- FoF group members (same CentralGalaxyIndex) ---
        grp_mask = all_gals['cgi'] == cgi
        grp_idx  = np.where(grp_mask)[0]
        members  = []
        for mi in grp_idx:
            sm_i  = float(all_gals['sm'][mi])
            bm_i  = float(all_gals['bm'][mi])
            sfr_i = float(all_gals['sfr'][mi])
            with np.errstate(divide='ignore', invalid='ignore'):
                ssfr_i = sfr_i / sm_i if sm_i > 0 else np.nan
                bt_i   = bm_i  / sm_i if sm_i > 0 else np.nan
            members.append({
                'gid':       int(all_gals['gid'][mi]),
                'is_target': int(all_gals['gid'][mi]) == tgid,
                'type':      int(all_gals['gtype'][mi]),
                'sm_msun':   sm_i,
                'sfr':       sfr_i,
                'ssfr':      ssfr_i,
                'bt':        bt_i,
                'posx':      float(all_gals['posx'][mi]),
                'posy':      float(all_gals['posy'][mi]),
                'posz':      float(all_gals['posz'][mi]),
            })

        # --- Distances with periodic boundary ---
        dx = all_gals['posx'] - tx
        dy = all_gals['posy'] - ty
        dz = all_gals['posz'] - tz
        if box_size_mpc:
            dx -= box_size_mpc * np.round(dx / box_size_mpc)
            dy -= box_size_mpc * np.round(dy / box_size_mpc)
            dz -= box_size_mpc * np.round(dz / box_size_mpc)
        dist = np.sqrt(dx**2 + dy**2 + dz**2)

        # Nearest non-group neighbour
        non_grp_dist   = np.where(~grp_mask, dist, np.inf)
        nearest_nongrp = float(np.min(non_grp_dist))

        # Overdensity at each radius — count only above mass threshold,
        # exclude self
        arange     = np.arange(n_all)
        self_mask  = arange != idx
        count_base = dense_mask & self_mask

        n_inner = int((count_base & (dist < SEARCH_RADII_MPC[0])).sum())
        n_outer = int((count_base & (dist < SEARCH_RADII_MPC[1])).sum())

        # Sphere indices for plotting (all galaxies, not just above threshold)
        outer_sphere_idx = np.where((dist < SEARCH_RADII_MPC[1]) & self_mask)[0]

        n_members  = len(grp_idx)
        mass_class = _env_class(cmvir)
        if mass_class == 'Isolated' and n_members == 2:
            env_class = 'Pair'
        elif mass_class == 'Isolated' and n_members > 2:
            env_class = 'Field group'
        else:
            env_class = mass_class

        results.append({
            'found':             True,
            'gid':               tgid,
            'env_class':         env_class,
            'host_log_mvir':     np.log10(cmvir) if cmvir > 0 else np.nan,
            'n_group_members':   n_members,
            'group_members':     members,
            'nearest_nongrp_mpc': nearest_nongrp if not np.isinf(nearest_nongrp) else np.nan,
            'n_inner':           n_inner,
            'n_outer':           n_outer,
            'r_inner':           SEARCH_RADII_MPC[0],
            'r_outer':           SEARCH_RADII_MPC[1],
            'delta_inner':       _overdensity(n_inner, SEARCH_RADII_MPC[0]),
            'delta_outer':       _overdensity(n_outer, SEARCH_RADII_MPC[1]),
            'sm_threshold_msun': SM_DENSITY_THRESHOLD,
            'kernel_sigmas':     list(KERNEL_SIGMAS_MPC),
            'kernel_deltas':     [float(kernel_deltas[s].get(tgid, np.nan))
                                  for s in KERNEL_SIGMAS_MPC],
            'pos':               (tx, ty, tz),
            'sphere_idx':        outer_sphere_idx,
        })

    return results


def print_environment_report(env_results, z_snap, run_label, qmode):
    """Print per-galaxy environment summary and group member details."""
    found = [r for r in env_results if r.get('found', False)]
    if not found:
        print('  No environment data available.')
        return

    r0      = found[0]
    r_in    = r0['r_inner']
    r_out   = r0['r_outer']
    sm_thr  = r0['sm_threshold_msun']
    sig_lo, sig_hi = r0.get('kernel_sigmas', list(KERNEL_SIGMAS_MPC))
    print(f'\n=== Environment analysis  z={z_snap:.2f}  {run_label}  '
          f'[{qmode}]  N={len(found)}  '
          f'R=({r_in:.1f}, {r_out:.1f}) Mpc  '
          f'sigma_K=({sig_lo:.0f}, {sig_hi:.0f}) cMpc  '
          f'M*_thr=10^{np.log10(sm_thr):.0f} Msun ===')

    classes = [r['env_class'] for r in found]
    for cls in ('Isolated', 'Pair', 'Field group', 'Group', 'Cluster'):
        n = classes.count(cls)
        pct = 100 * n / len(found)
        print(f'  {cls:<10}: {n:>3}  ({pct:.0f}%)')

    print()
    hdr = (f'  {"GID":>12}  {"Class":<10}  {"log M_host":>10}  '
           f'{"N_grp":>5}  {"d_near [Mpc]":>12}  '
           f'{"N(<{:.1f})".format(r_in):>8}  {"δ({:.1f})".format(r_in):>8}  '
           f'{"N(<{:.1f})".format(r_out):>8}  {"δ({:.1f})".format(r_out):>8}  '
           f'{"δK({:.0f})".format(sig_lo):>8}  {"δK({:.0f})".format(sig_hi):>8}')
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))
    for r in found:
        mv_s   = f'{r["host_log_mvir"]:.3f}' if np.isfinite(r['host_log_mvir']) else '   N/A'
        dn_s   = (f'{r["nearest_nongrp_mpc"]:.3f}'
                  if np.isfinite(r.get('nearest_nongrp_mpc', np.nan)) else '         N/A')
        di_s   = (f'{r["delta_inner"]:.2f}'
                  if np.isfinite(r.get('delta_inner', np.nan)) else '     N/A')
        do_s   = (f'{r["delta_outer"]:.2f}'
                  if np.isfinite(r.get('delta_outer', np.nan)) else '     N/A')
        kd     = r.get('kernel_deltas', [np.nan, np.nan])
        kdi_s  = f'{kd[0]:.2f}' if np.isfinite(kd[0]) else '     N/A'
        kdo_s  = f'{kd[1]:.2f}' if np.isfinite(kd[1]) else '     N/A'
        print(f'  {r["gid"]:>12}  {r["env_class"]:<10}  {mv_s:>10}  '
              f'{r["n_group_members"]:>5}  {dn_s:>12}  '
              f'{r["n_inner"]:>8}  {di_s:>8}  '
              f'{r["n_outer"]:>8}  {do_s:>8}  '
              f'{kdi_s:>8}  {kdo_s:>8}')

    # Group member detail per target
    type_names = {0: 'Central', 1: 'Satellite', 2: 'Orphan'}
    for r in found:
        if r['n_group_members'] <= 1:
            continue
        print(f'\n  GID {r["gid"]}  [{r["env_class"]},  '
              f'log M_host={r["host_log_mvir"]:.2f},  '
              f'N_members={r["n_group_members"]}]')
        mhdr = (f'    {"GID":>12}  {"Role":<10}  '
                f'{"log m*":>7}  {"log sSFR":>9}  {"B/T":>5}')
        print(mhdr)
        print('    ' + '-' * (len(mhdr) - 4))
        for m in sorted(r['group_members'], key=lambda x: -x['sm_msun']):
            tag    = ' <-- you' if m['is_target'] else ''
            sm_s   = f'{np.log10(m["sm_msun"]):.2f}' if m['sm_msun'] > 0 else '  N/A'
            ss_val = m.get('ssfr', np.nan)
            ss_s   = (f'{np.log10(ss_val):.2f}'
                      if ss_val is not None and np.isfinite(ss_val) and ss_val > 0
                      else '  -inf' if (ss_val == 0 or ss_val is not None and ss_val == 0)
                      else '   N/A')
            bt_val = m.get('bt', np.nan)
            bt_s   = f'{bt_val:.3f}' if bt_val is not None and np.isfinite(bt_val) else '  N/A'
            role   = type_names.get(m['type'], '?')
            print(f'    {m["gid"]:>12}  {role:<10}  '
                  f'{sm_s:>7}  {ss_s:>9}  {bt_s:>5}{tag}')


def plot_environment_summary(env_results, _z_snap, _run_label, _qmode, out_path):
    """1×3 summary: environment class counts, host halo mass, overdensity."""
    found = [r for r in env_results if r.get('found', False)]
    if not found:
        return

    classes      = [r['env_class'] for r in found]
    log_mvir     = np.array([r['host_log_mvir'] for r in found])
    delta_inner  = np.array([r.get('delta_inner', np.nan) for r in found])
    delta_outer  = np.array([r.get('delta_outer', np.nan) for r in found])
    kernel_arr   = np.array([r.get('kernel_deltas', [np.nan, np.nan]) for r in found])
    kernel_lo    = kernel_arr[:, 0]
    kernel_hi    = kernel_arr[:, 1]
    sig_lo, sig_hi = found[0].get('kernel_sigmas', list(KERNEL_SIGMAS_MPC))
    n_members    = np.array([r['n_group_members'] for r in found])
    r_in         = found[0]['r_inner']
    r_out        = found[0]['r_outer']

    cls_colors = {'Isolated': '#4CAF50', 'Pair': '#8BC34A',
                  'Field group': '#CDDC39', 'Group': '#FF9800', 'Cluster': '#F44336'}
    cls_names  = [c for c in ['Isolated', 'Pair', 'Field group', 'Group', 'Cluster']
                  if c in classes]

    fig, axes2d = plt.subplots(2, 2, figsize=(9.5, 8))
    axes = axes2d.flatten()
    fig.patch.set_facecolor('black')

    # Panel 1 — environment class bar chart
    ax = axes[0]
    ax.set_facecolor('black')
    counts = [classes.count(c) for c in cls_names]
    bars = ax.barh(cls_names, counts,
                   color=[cls_colors[c] for c in cls_names], alpha=0.85)
    for bar, val in zip(bars, counts):
        if val > 0:
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                    str(val), va='center', color='white', fontsize=10)
    ax.set_xlabel('Count', fontsize=11)
    ax.set_xlim(0, max(counts) * 1.25 + 0.5)

    # Panel 2 — host halo mass histogram
    ax = axes[1]
    ax.set_facecolor('black')
    fin = log_mvir[np.isfinite(log_mvir)]
    if len(fin):
        lo, hi = fin.min() - 0.3, fin.max() + 0.3
        bins = np.linspace(lo, hi, max(8, len(fin) // 3 + 3))
        ax.hist(fin, bins=bins, color='cyan', alpha=0.75,
                edgecolor='white', linewidth=0.5)
        ax.axvline(12.5, color='orange', ls='--', lw=1.0, alpha=0.9,
                   label='Group floor')
        ax.axvline(14.0, color='red',    ls='--', lw=1.0, alpha=0.9,
                   label='Cluster floor')
        leg = ax.legend(fontsize=8)
        leg.get_frame().set_facecolor('black')
        leg.get_frame().set_edgecolor('white')
        for t in leg.get_texts():
            t.set_color('white')
    ax.set_xlabel(r'$\log_{10}(M_{\rm host}\,/\,\mathrm{M}_\odot)$', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)

    # Panel 3 — overdensity at both radii, or N-members fallback
    ax = axes[2]
    ax.set_facecolor('black')
    di_fin = delta_inner[np.isfinite(delta_inner)]
    do_fin = delta_outer[np.isfinite(delta_outer)]
    if len(di_fin) or len(do_fin):
        all_od = np.concatenate([di_fin, do_fin])
        lo, hi = all_od.min() - 1, all_od.max() + 1
        bins = np.linspace(lo, hi, max(8, len(all_od) // 4 + 3))
        if len(di_fin):
            ax.hist(di_fin, bins=bins, color='magenta', alpha=0.6,
                    edgecolor='white', linewidth=0.5,
                    label=fr'$\delta$({r_in:.0f} Mpc)')
        if len(do_fin):
            ax.hist(do_fin, bins=bins, color='cyan', alpha=0.5,
                    edgecolor='white', linewidth=0.5,
                    label=fr'$\delta$({r_out:.0f} Mpc)')
        ax.axvline(0, color='white', ls=':', lw=1.0, alpha=0.5)
        leg = ax.legend(fontsize=8)
        leg.get_frame().set_facecolor('black')
        leg.get_frame().set_edgecolor('white')
        for t in leg.get_texts():
            t.set_color('white')
        ax.set_xlabel(r'Overdensity $\delta$', fontsize=11)
    else:
        bins = np.arange(0, n_members.max() + 2) - 0.5
        ax.hist(n_members, bins=bins, color='magenta', alpha=0.75,
                edgecolor='white', linewidth=0.5)
        ax.set_xlabel('N FoF group members', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)

    # Panel 4 — mass-weighted Gaussian-kernel overdensity (halo field)
    ax = axes[3]
    ax.set_facecolor('black')
    kl_fin = kernel_lo[np.isfinite(kernel_lo)]
    kh_fin = kernel_hi[np.isfinite(kernel_hi)]
    if len(kl_fin) or len(kh_fin):
        all_k = np.concatenate([kl_fin, kh_fin])
        lo, hi = all_k.min() - 0.5, all_k.max() + 0.5
        bins = np.linspace(lo, hi, max(8, len(all_k) // 4 + 3))
        if len(kl_fin):
            ax.hist(kl_fin, bins=bins, color='gold', alpha=0.6,
                    edgecolor='white', linewidth=0.5,
                    label=fr'$\delta_K(\sigma={sig_lo:.0f}\,\mathrm{{cMpc}})$')
        if len(kh_fin):
            ax.hist(kh_fin, bins=bins, color='tomato', alpha=0.5,
                    edgecolor='white', linewidth=0.5,
                    label=fr'$\delta_K(\sigma={sig_hi:.0f}\,\mathrm{{cMpc}})$')
        ax.axvline(0, color='white', ls=':', lw=1.0, alpha=0.5)
        leg = ax.legend(fontsize=8)
        leg.get_frame().set_facecolor('black')
        leg.get_frame().set_edgecolor('white')
        for t in leg.get_texts():
            t.set_color('white')
    else:
        ax.text(0.5, 0.5, 'No kernel data', ha='center', va='center',
                color='grey', fontsize=10, transform=ax.transAxes)
    ax.set_xlabel(r'Kernel overdensity $\delta_K$ (M$_{\rm vir}$-weighted)',
                  fontsize=11)
    ax.set_ylabel('Count', fontsize=11)

    for ax in axes:
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def plot_group_maps(env_results, all_gals, _z_snap, _run_label, _qmode,
                    out_path, box_size_mpc=None, max_panels=16,
                    min_overdensity=None):
    """Grid of projected XY maps, one panel per target galaxy.

    Background grey points: all galaxies within the search sphere.
    Coloured circles: FoF group members (size ∝ log m*, colour = log m*).
    Red filled circle: target galaxy.

    min_overdensity: if set, only include targets where max(delta_inner, delta_outer) >= this value.
    """
    found = [r for r in env_results if r.get('found', False)]
    if min_overdensity is not None:
        found = [r for r in found
                 if r.get('delta_outer', -np.inf) >= min_overdensity]
        # Deduplicate: one panel per unique FoF group
        seen, deduped = set(), []
        for r in found:
            key = frozenset(m['gid'] for m in r['group_members'])
            if key not in seen:
                seen.add(key)
                deduped.append(r)
        found = deduped
    if not found:
        if min_overdensity is not None:
            print(f'  No targets with δ(5 Mpc) >= {min_overdensity:.0f} — skipping overdense maps.')
        return

    n = min(len(found), max_panels)
    if len(found) > max_panels:
        print(f'  Note: {len(found)} targets — showing first {max_panels} in group maps.')

    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows),
                              squeeze=False)
    fig.patch.set_facecolor('black')

    # Global log-m* range across all group members for a shared colormap
    all_log_sm = []
    for r in found[:n]:
        for m in r['group_members']:
            if m['sm_msun'] > 0:
                all_log_sm.append(np.log10(m['sm_msun']))
    if all_log_sm:
        sm_min, sm_max = np.nanmin(all_log_sm), np.nanmax(all_log_sm)
    else:
        sm_min, sm_max = 8.0, 12.0
    norm = plt.Normalize(vmin=sm_min, vmax=sm_max)
    cmap = plt.cm.plasma

    R = found[0]['r_outer']   # outer aperture radius — sets map extent

    for i, r in enumerate(found[:n]):
        row, col = divmod(i, ncols)
        ax = axes[row][col]
        ax.set_facecolor('black')
        ax.set_aspect('equal')

        tx, ty, _ = r['pos']

        # Background: all galaxies in the search sphere
        if len(r['sphere_idx']) > 0:
            sx = all_gals['posx'][r['sphere_idx']] - tx
            sy = all_gals['posy'][r['sphere_idx']] - ty
            if box_size_mpc:
                sx -= box_size_mpc * np.round(sx / box_size_mpc)
                sy -= box_size_mpc * np.round(sy / box_size_mpc)
            ax.scatter(sx, sy, s=3, color='#3a3a3a', zorder=1, linewidths=0)

        # FoF group members coloured by log m*
        grp_dx, grp_dy = [], []
        for m in r['group_members']:
            if m['sm_msun'] <= 0:
                continue
            log_sm_m = np.log10(m['sm_msun'])
            color    = cmap(norm(log_sm_m))
            mx = m['posx'] - tx
            my = m['posy'] - ty
            if box_size_mpc:
                mx -= box_size_mpc * np.round(mx / box_size_mpc)
                my -= box_size_mpc * np.round(my / box_size_mpc)
            grp_dx.append(mx)
            grp_dy.append(my)
            if m['is_target']:
                ax.scatter(mx, my, s=20, color='red', zorder=5,
                           linewidths=0)
            else:
                ax.scatter(mx, my, s=20, color=color, zorder=3,
                           linewidths=0, alpha=0.9)

        # Red circle enclosing the FoF group
        if len(grp_dx) > 1:
            cx = np.mean(grp_dx)
            cy = np.mean(grp_dy)
            r_grp = np.hypot(np.array(grp_dx) - cx,
                             np.array(grp_dy) - cy).max() * 1.25
            theta = np.linspace(0, 2 * np.pi, 200)
            ax.plot(cx + r_grp * np.cos(theta), cy + r_grp * np.sin(theta),
                    color='red', lw=1.0, ls='--', alpha=0.75, zorder=4)

        # Search-radius reference circle
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(R * np.cos(theta), R * np.sin(theta),
                color='white', lw=0.5, ls=':', alpha=0.35, zorder=2)

        ax.set_xlim(-R * 1.08, R * 1.08)
        ax.set_ylim(-R * 1.08, R * 1.08)
        ax.set_xlabel('ΔX [Mpc]', fontsize=8, color='white')
        ax.set_ylabel('ΔY [Mpc]', fontsize=8, color='white')
        ax.tick_params(colors='white', labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor('white')

        di_s = (f'δ({found[0]["r_inner"]:.0f})={r["delta_inner"]:.1f}'
                if np.isfinite(r.get('delta_inner', np.nan)) else '')
        do_s = (f'  δ({found[0]["r_outer"]:.0f})={r["delta_outer"]:.1f}'
                if np.isfinite(r.get('delta_outer', np.nan)) else '')
        ax.set_title(
            f'GID {r["gid"]}  —  {r["env_class"]}\n'
            f'log M_host={r["host_log_mvir"]:.2f}  N_grp={r["n_group_members"]}\n'
            f'{di_s}{do_s}',
            fontsize=7, color='white', pad=4)

    for i in range(n, nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row][col].set_visible(False)

    sm_cb = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm_cb.set_array([])
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    cax = fig.add_axes([0.15, 0.02, 0.70, 0.025])
    cbar = fig.colorbar(sm_cb, cax=cax, orientation='horizontal')
    cbar.set_label(r'$\log_{10}(m_*\,/\,\mathrm{M}_\odot)$ — FoF group members',
                   fontsize=10, color='white')
    cbar.ax.xaxis.set_tick_params(color='white')
    cbar.outline.set_edgecolor('white')
    plt.setp(cbar.ax.xaxis.get_ticklabels(), color='white')

    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def plot_kernel_density_maps(env_results, all_gals, _z_snap, _run_label, _qmode,
                              out_path, box_size_mpc=None, max_panels=16,
                              sigma_mpc=None, extent_mpc=None,
                              z_slab_mpc=None, grid_n=1024,
                              min_overdensity=None):
    """Per-target projected mass-weighted Gaussian kernel maps.

    For each target the field of central halos (weighted by Mvir) is summed
    with a 2D Gaussian of width ``sigma_mpc`` over an XY grid centred on the
    target, restricting to halos inside a z-slab of half-thickness
    ``z_slab_mpc``.  Halo positions are overlaid as cyan dots; the target sits
    at the origin as a red star.  Dashed/dotted circles mark the kernel
    scales used in ``compute_environment``.  All panels share a log colour
    scale so densities can be compared between targets.
    """
    from matplotlib.colors import LogNorm, TwoSlopeNorm

    found = [r for r in env_results if r.get('found', False)]
    if min_overdensity is not None:
        found = [r for r in found
                 if max(r.get('kernel_deltas', [-np.inf, -np.inf]))
                    >= min_overdensity]
        # Deduplicate by FoF host so overdense panels do not repeat
        seen, deduped = set(), []
        for r in found:
            key = frozenset(m['gid'] for m in r['group_members'])
            if key not in seen:
                seen.add(key)
                deduped.append(r)
        found = deduped
    if not found:
        if min_overdensity is not None:
            print(f'  No targets with delta_K >= {min_overdensity:.1f} — '
                  'skipping kernel density maps.')
        return

    sigmas = found[0].get('kernel_sigmas', list(KERNEL_SIGMAS_MPC))
    sigma  = float(sigma_mpc  if sigma_mpc  is not None else max(sigmas))
    R      = float(extent_mpc if extent_mpc is not None else 3.0 * max(sigmas))
    zslab  = float(z_slab_mpc if z_slab_mpc is not None else 2.0 * sigma)

    # Field: ALL halos above the resolution floor, weighted by Mvir.
    halo_mask = all_gals['mvir'] >= HALO_MASS_FLOOR_MSUN
    cx = all_gals['posx'][halo_mask]
    cy = all_gals['posy'][halo_mask]
    cz = all_gals['posz'][halo_mask]
    cm = all_gals['mvir'][halo_mask]

    # sSFR colour normalisation for the target star — diverging, centred at
    # the conventional quiescent threshold log10(sSFR) = -11.
    ssfr_norm = TwoSlopeNorm(vmin=-12.0, vcenter=-11.0, vmax=-9.0)
    ssfr_cmap = plt.cm.bwr_r

    n = min(len(found), max_panels)
    if len(found) > max_panels:
        print(f'  Note: {len(found)} targets — showing first {max_panels} '
              'kernel density maps.')

    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows),
                              squeeze=False)
    fig.patch.set_facecolor('black')

    norm_2d    = 1.0 / (2.0 * np.pi * sigma ** 2)
    sigma2_inv = 1.0 / (2.0 * sigma ** 2)
    grid_axis  = np.linspace(-R, R, grid_n)
    gx, gy     = np.meshgrid(grid_axis, grid_axis)

    images, extras = [], []
    rho_max_all    = 0.0

    for r in found[:n]:
        tx, ty, tz = r['pos']

        dx = cx - tx
        dy = cy - ty
        dz = cz - tz
        if box_size_mpc:
            dx -= box_size_mpc * np.round(dx / box_size_mpc)
            dy -= box_size_mpc * np.round(dy / box_size_mpc)
            dz -= box_size_mpc * np.round(dz / box_size_mpc)

        # z-slab + XY pre-filter slightly larger than R so kernel tails are kept
        keep = ((np.abs(dz) <= zslab) &
                (np.abs(dx) <= R + 3.0 * sigma) &
                (np.abs(dy) <= R + 3.0 * sigma))
        hdx = dx[keep]
        hdy = dy[keep]
        hm  = cm[keep]

        rho = np.zeros((grid_n, grid_n), dtype=np.float64)
        if len(hm):
            # Chunk halos to keep peak memory bounded (grid_n^2 * chunk floats);
            # chunk=32 keeps (1024² × 32 × 8B) ≈ 256 MB per iteration
            chunk = 32
            for k0 in range(0, len(hm), chunk):
                hx = hdx[k0:k0 + chunk]
                hy = hdy[k0:k0 + chunk]
                hM = hm[k0:k0 + chunk]
                rr = (gx[..., None] - hx) ** 2 + (gy[..., None] - hy) ** 2
                rho += norm_2d * np.sum(hM * np.exp(-rr * sigma2_inv), axis=2)

        # Marker overlay — ALL galaxies (no Mvir cut), inside slab + window
        gdx_all = all_gals['posx'] - tx
        gdy_all = all_gals['posy'] - ty
        gdz_all = all_gals['posz'] - tz
        if box_size_mpc:
            gdx_all -= box_size_mpc * np.round(gdx_all / box_size_mpc)
            gdy_all -= box_size_mpc * np.round(gdy_all / box_size_mpc)
            gdz_all -= box_size_mpc * np.round(gdz_all / box_size_mpc)
        g_keep = ((np.abs(gdz_all) <= zslab) &
                  (np.abs(gdx_all) <= R) &
                  (np.abs(gdy_all) <= R))
        g_dx = gdx_all[g_keep]
        g_dy = gdy_all[g_keep]
        g_sm = all_gals['sm'][g_keep]

        images.append(rho)
        extras.append((g_dx, g_dy, g_sm))
        rho_max_all = max(rho_max_all, float(rho.max()) if rho.size else 0.0)

    # --- Summary table of what each panel actually shows ---
    print(f'  ── Kernel density map data ──')
    print(f'    panels rendered : {len(images)}  (max requested {max_panels})')
    print(f'    smoothing σ     : {sigma:.2f} cMpc   '
          f'extent ±{R:.1f} cMpc   slab ±{zslab:.1f} cMpc')
    delta_in  = []
    delta_out = []
    rho_peaks = []
    for r, rho in zip(found[:n], images):
        kd = r.get('kernel_deltas', [np.nan, np.nan])
        delta_in.append(kd[0])
        delta_out.append(kd[1])
        rho_peaks.append(float(rho.max()) if rho.size else np.nan)
    print(f'    panel Σ_max     : {_array_stats(np.array(rho_peaks), "{:.2e}")}')
    print(f'    δ_K(3 cMpc)     : {_array_stats(np.array(delta_in))}')
    print(f'    δ_K(5 cMpc)     : {_array_stats(np.array(delta_out))}')

    # Stretch the colour scale: 1st-percentile floor of the positive density
    # values across all panels (brings out low-density structure), capped at
    # vmax/1e7 to keep a sane dynamic range.  vmax = peak across all panels.
    norm_img = None
    if rho_max_all > 0:
        nonzero = [im[im > 0] for im in images]
        nonzero = [v for v in nonzero if v.size]
        if nonzero:
            flat = np.concatenate(nonzero)
            vmin = max(float(np.percentile(flat, 1.0)), rho_max_all / 1e7)
        else:
            vmin = rho_max_all / 1e7
        norm_img = LogNorm(vmin=vmin, vmax=rho_max_all)
    cmap = plt.cm.inferno

    for i, (r, rho, (gxx, gyy, gsm)) in enumerate(
            zip(found[:n], images, extras)):
        row, col = divmod(i, ncols)
        ax = axes[row][col]
        ax.set_facecolor('black')
        ax.set_aspect('equal')

        ax.imshow(rho, origin='lower', extent=[-R, R, -R, R],
                  cmap=cmap, norm=norm_img,
                  interpolation='bilinear', zorder=1)

        # All galaxies as markers (no Mvir cut), sized by log stellar mass
        if len(gsm):
            log_sm = np.log10(np.maximum(gsm, 1.0))
            sizes  = 4.0 + 3.0 * np.clip(log_sm - 9.0, 0.0, 3.5)
            ax.scatter(gxx, gyy, s=sizes,
                       facecolor='white', edgecolor='black',
                       linewidths=0.4, alpha=0.85, zorder=2)

        # Kernel sigma reference circles — slateblue
        theta = np.linspace(0, 2.0 * np.pi, 200)
        for ss, ls in zip(sigmas, ['--', ':']):
            ax.plot(ss * np.cos(theta), ss * np.sin(theta),
                    color='slateblue', lw=1.4, ls=ls, alpha=0.9, zorder=3,
                    label=fr'$\sigma={ss:.0f}\,\mathrm{{cMpc}}$' if i == 0 else None)

        # Target marker — coloured by its own sSFR via bwr_r/TwoSlopeNorm,
        # with a black outline.
        target_ssfr = next((m['ssfr'] for m in r['group_members']
                            if m.get('is_target')), None)
        if (target_ssfr is not None and np.isfinite(target_ssfr)
                and target_ssfr > 0):
            log_t_ssfr = float(np.log10(target_ssfr))
        else:
            log_t_ssfr = -16.0   # quiescent / undefined → red end
        t_color = ssfr_cmap(ssfr_norm(log_t_ssfr))
        ax.scatter([0], [0], s=220, marker='*', zorder=4,
                   facecolor=t_color, edgecolor='black', linewidths=0.9)

        ax.set_xlim(-R, R)
        ax.set_ylim(-R, R)
        ax.set_xlabel(r'$\Delta X$ [cMpc]', fontsize=8, color='white')
        ax.set_ylabel(r'$\Delta Y$ [cMpc]', fontsize=8, color='white')
        ax.tick_params(colors='white', labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor('white')

        kd = r.get('kernel_deltas', [np.nan, np.nan])
        kd_s = '  '.join(
            fr'$\delta_K({s:.0f})={d:.2f}$'
            for s, d in zip(sigmas, kd) if np.isfinite(d))
        ax.set_title(
            f'GID {r["gid"]}  —  {r["env_class"]}\n'
            f'log M_host={r["host_log_mvir"]:.2f}\n'
            f'{kd_s}',
            fontsize=7, color='white', pad=4)

        if i == 0:
            leg = ax.legend(fontsize=6, loc='lower right')
            leg.get_frame().set_facecolor('black')
            leg.get_frame().set_edgecolor('white')
            for t in leg.get_texts():
                t.set_color('white')

    for i in range(n, nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row][col].set_visible(False)

    fig.tight_layout(rect=[0, 0.10, 1, 1])

    log_floor = np.log10(HALO_MASS_FLOOR_MSUN)

    if norm_img is not None:
        sm_cb = plt.cm.ScalarMappable(cmap=cmap, norm=norm_img)
        sm_cb.set_array([])
        cax = fig.add_axes([0.07, 0.045, 0.42, 0.022])
        cbar = fig.colorbar(sm_cb, cax=cax, orientation='horizontal')
        cbar.set_label(
            fr'$\Sigma_{{M_{{\rm vir}}}}\;[\mathrm{{M}}_\odot\,\mathrm{{cMpc}}^{{-2}}]$  '
            fr'($\sigma={sigma:.1f}\,\mathrm{{cMpc}}$, '
            fr'$|\Delta z|<{zslab:.1f}\,\mathrm{{cMpc}}$, '
            fr'$\log M_{{\rm vir}}\geq{log_floor:.1f}$)',
            fontsize=9, color='white')
        cbar.ax.xaxis.set_tick_params(color='white')
        cbar.outline.set_edgecolor('white')
        plt.setp(cbar.ax.xaxis.get_ticklabels(), color='white')

    # sSFR colorbar for the halo markers
    sm_cb2 = plt.cm.ScalarMappable(cmap=ssfr_cmap, norm=ssfr_norm)
    sm_cb2.set_array([])
    cax2 = fig.add_axes([0.55, 0.045, 0.40, 0.022])
    cbar2 = fig.colorbar(sm_cb2, cax=cax2, orientation='horizontal',
                          extend='both')
    cbar2.set_label(
        r'$\log_{10}(\mathrm{sSFR}\;[\mathrm{yr}^{-1}])$ — target star colour',
        fontsize=9, color='white')
    cbar2.ax.xaxis.set_tick_params(color='white')
    cbar2.outline.set_edgecolor('white')
    plt.setp(cbar2.ax.xaxis.get_ticklabels(), color='white')

    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def plot_kernel_density_projections(env_results, all_gals, _z_snap, _run_label,
                                     _qmode, out_path, box_size_mpc=None,
                                     target_gid=None,
                                     sigma_mpc=None, extent_mpc=None,
                                     z_slab_mpc=None, grid_n=1024):
    """Three orthogonal kernel-density projections (XY, XZ, YZ) for one target.

    Defaults to the most massive target (by target stellar mass) in
    ``env_results``.  Each panel sums Mvir-weighted 2D Gaussians on a grid
    over halos inside a slab perpendicular to the view; halo dots, sigma
    rings and the sSFR-coloured target star follow the same conventions as
    ``plot_kernel_density_maps``.
    """
    from matplotlib.colors import LogNorm, TwoSlopeNorm

    found = [r for r in env_results if r.get('found', False)]
    if not found:
        return

    def _target_sm(r):
        for m in r['group_members']:
            if m.get('is_target'):
                return m.get('sm_msun', 0.0)
        return 0.0

    if target_gid is None:
        best       = max(found, key=_target_sm)
        target_sm  = _target_sm(best)
        sm_log_str = (f'{np.log10(target_sm):.2f}' if target_sm > 0 else 'N/A')
        print(f'  Most massive target for projections: '
              f'GID={best["gid"]}  log M*={sm_log_str}')
    else:
        match = [r for r in found if r['gid'] == int(target_gid)]
        if not match:
            print(f'  Target {target_gid} not in env_results — '
                  'skipping projections.')
            return
        best = match[0]

    sigmas = best.get('kernel_sigmas', list(KERNEL_SIGMAS_MPC))
    sigma  = float(sigma_mpc  if sigma_mpc  is not None else max(sigmas))
    R      = float(extent_mpc if extent_mpc is not None else 3.0 * max(sigmas))
    zslab  = float(z_slab_mpc if z_slab_mpc is not None else 2.0 * sigma)

    # Field: ALL halos above the resolution floor, weighted by Mvir.
    halo_mask = all_gals['mvir'] >= HALO_MASS_FLOOR_MSUN
    cx = all_gals['posx'][halo_mask]
    cy = all_gals['posy'][halo_mask]
    cz = all_gals['posz'][halo_mask]
    cm = all_gals['mvir'][halo_mask]

    ssfr_norm = TwoSlopeNorm(vmin=-12.0, vcenter=-11.0, vmax=-9.0)
    ssfr_cmap = plt.cm.bwr_r

    tx, ty, tz = best['pos']
    dx = cx - tx
    dy = cy - ty
    dz = cz - tz
    if box_size_mpc:
        dx -= box_size_mpc * np.round(dx / box_size_mpc)
        dy -= box_size_mpc * np.round(dy / box_size_mpc)
        dz -= box_size_mpc * np.round(dz / box_size_mpc)

    # All-galaxy offsets for the marker overlay (no Mvir cut)
    gdx_all = all_gals['posx'] - tx
    gdy_all = all_gals['posy'] - ty
    gdz_all = all_gals['posz'] - tz
    if box_size_mpc:
        gdx_all -= box_size_mpc * np.round(gdx_all / box_size_mpc)
        gdy_all -= box_size_mpc * np.round(gdy_all / box_size_mpc)
        gdz_all -= box_size_mpc * np.round(gdz_all / box_size_mpc)
    g_sm_all = all_gals['sm']

    grid_axis  = np.linspace(-R, R, grid_n)
    gh, gv     = np.meshgrid(grid_axis, grid_axis)
    norm_2d    = 1.0 / (2.0 * np.pi * sigma ** 2)
    sigma2_inv = 1.0 / (2.0 * sigma ** 2)

    def _kernel_grid(h_off, v_off, perp_off):
        keep = ((np.abs(perp_off) <= zslab) &
                (np.abs(h_off)    <= R + 3.0 * sigma) &
                (np.abs(v_off)    <= R + 3.0 * sigma))
        hh = h_off[keep]
        vv = v_off[keep]
        mm = cm[keep]
        rho = np.zeros((grid_n, grid_n), dtype=np.float64)
        if len(mm):
            chunk = 32
            for k0 in range(0, len(mm), chunk):
                hx = hh[k0:k0 + chunk]
                hy = vv[k0:k0 + chunk]
                hM = mm[k0:k0 + chunk]
                rr = (gh[..., None] - hx) ** 2 + (gv[..., None] - hy) ** 2
                rho += norm_2d * np.sum(hM * np.exp(-rr * sigma2_inv), axis=2)
        return rho

    def _galaxy_markers(gh_off, gv_off, gperp_off):
        g_keep = ((np.abs(gperp_off) <= zslab) &
                  (np.abs(gh_off)    <= R) &
                  (np.abs(gv_off)    <= R))
        return gh_off[g_keep], gv_off[g_keep], g_sm_all[g_keep]

    # (h_label, v_label, halo_h, halo_v, halo_perp, gal_h, gal_v, gal_perp)
    projections = [
        ('X', 'Y', dx, dy, dz, gdx_all, gdy_all, gdz_all),
        ('X', 'Z', dx, dz, dy, gdx_all, gdz_all, gdy_all),
        ('Y', 'Z', dy, dz, dx, gdy_all, gdz_all, gdx_all),
    ]

    panels      = []
    rho_max_all = 0.0
    for hl, vl, h, v, sl, gh_off, gv_off, gsl_off in projections:
        rho = _kernel_grid(h, v, sl)
        gxx, gyy, gsm = _galaxy_markers(gh_off, gv_off, gsl_off)
        panels.append((hl, vl, rho, gxx, gyy, gsm))
        rho_max_all = max(rho_max_all,
                          float(rho.max()) if rho.size else 0.0)

    # --- Summary per projection ---
    print(f'  ── Kernel density projections data ──')
    print(f'    target GID  : {best["gid"]}  ({best["env_class"]})')
    print(f'    σ = {sigma:.2f} cMpc   extent ±{R:.1f}   slab ±{zslab:.1f}')
    for hl, vl, rho, gxx, _gyy, gsm in panels:
        n_gal_in_window = int(len(gsm))
        rho_peak = float(rho.max()) if rho.size else 0.0
        n_above_floor = int(((rho > 0) & np.isfinite(rho)).sum())
        print(f'    {hl}{vl}  Σ_max={rho_peak:.2e} M⊙/cMpc²  '
              f'galaxies in window={n_gal_in_window:>5d}  '
              f'pixels with ρ>0={n_above_floor}/{rho.size}')

    norm_img = None
    if rho_max_all > 0:
        nonzero = [p[2][p[2] > 0] for p in panels]
        nonzero = [v for v in nonzero if v.size]
        if nonzero:
            flat = np.concatenate(nonzero)
            vmin = max(float(np.percentile(flat, 1.0)), rho_max_all / 1e7)
        else:
            vmin = rho_max_all / 1e7
        norm_img = LogNorm(vmin=vmin, vmax=rho_max_all)
    cmap = plt.cm.inferno

    # Target star colour from its own sSFR
    target_ssfr = next((m['ssfr'] for m in best['group_members']
                        if m.get('is_target')), None)
    if (target_ssfr is not None and np.isfinite(target_ssfr)
            and target_ssfr > 0):
        log_t_ssfr = float(np.log10(target_ssfr))
    else:
        log_t_ssfr = -16.0
    t_color = ssfr_cmap(ssfr_norm(log_t_ssfr))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5))
    fig.patch.set_facecolor('black')

    for ax_idx, (hl, vl, rho, gxx, gyy, gsm) in enumerate(panels):
        ax = axes[ax_idx]
        ax.set_facecolor('black')
        ax.set_aspect('equal')

        ax.imshow(rho, origin='lower', extent=[-R, R, -R, R],
                  cmap=cmap, norm=norm_img,
                  interpolation='bilinear', zorder=1)

        # All galaxies as markers (no Mvir cut), sized by log stellar mass
        if len(gsm):
            log_sm = np.log10(np.maximum(gsm, 1.0))
            sizes  = 4.0 + 3.0 * np.clip(log_sm - 9.0, 0.0, 3.5)
            ax.scatter(gxx, gyy, s=sizes,
                       facecolor='white', edgecolor='black',
                       linewidths=0.4, alpha=0.85, zorder=2)

        theta = np.linspace(0, 2.0 * np.pi, 200)
        for ss, ls in zip(sigmas, ['--', ':']):
            ax.plot(ss * np.cos(theta), ss * np.sin(theta),
                    color='slateblue', lw=1.4, ls=ls, alpha=0.9, zorder=3,
                    label=(fr'$\sigma={ss:.0f}\,\mathrm{{cMpc}}$'
                           if ax_idx == 0 else None))

        ax.scatter([0], [0], s=220, marker='*', zorder=4,
                   facecolor=t_color, edgecolor='black', linewidths=0.9)

        ax.set_xlim(-R, R)
        ax.set_ylim(-R, R)
        ax.set_xlabel(fr'$\Delta {hl}$ [cMpc]', fontsize=10, color='white')
        ax.set_ylabel(fr'$\Delta {vl}$ [cMpc]', fontsize=10, color='white')
        ax.tick_params(colors='white', labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
        ax.set_title(f'{hl}{vl} projection', fontsize=11, color='white')

    handles, _ = axes[0].get_legend_handles_labels()
    if handles:
        leg = axes[0].legend(fontsize=8, loc='lower right')
        leg.get_frame().set_facecolor('black')
        leg.get_frame().set_edgecolor('white')
        for t in leg.get_texts():
            t.set_color('white')

    kd     = best.get('kernel_deltas', [np.nan, np.nan])
    kd_s   = '  '.join(fr'$\delta_K({s:.0f})={d:.2f}$'
                       for s, d in zip(sigmas, kd) if np.isfinite(d))
    target_sm = _target_sm(best)
    sm_part   = (fr'log $M_*$ = {np.log10(target_sm):.2f}    '
                 if target_sm > 0 else '')
    suptitle = (f'GID {best["gid"]}  —  {best["env_class"]}    '
                f'{sm_part}log $M_{{\\rm host}}$ = '
                f'{best["host_log_mvir"]:.2f}    {kd_s}')
    fig.suptitle(suptitle, fontsize=11, color='white', y=0.97)

    fig.tight_layout(rect=[0, 0.14, 1, 0.93])

    log_floor = np.log10(HALO_MASS_FLOOR_MSUN)
    if norm_img is not None:
        sm_cb = plt.cm.ScalarMappable(cmap=cmap, norm=norm_img)
        sm_cb.set_array([])
        cax = fig.add_axes([0.07, 0.05, 0.42, 0.022])
        cbar = fig.colorbar(sm_cb, cax=cax, orientation='horizontal')
        cbar.set_label(
            fr'$\Sigma_{{M_{{\rm vir}}}}\;[\mathrm{{M}}_\odot\,\mathrm{{cMpc}}^{{-2}}]$  '
            fr'($\sigma={sigma:.1f}\,\mathrm{{cMpc}}$, '
            fr'$|\Delta_\perp|<{zslab:.1f}\,\mathrm{{cMpc}}$, '
            fr'$\log M_{{\rm vir}}\geq{log_floor:.1f}$)',
            fontsize=9, color='white')
        cbar.ax.xaxis.set_tick_params(color='white')
        cbar.outline.set_edgecolor('white')
        plt.setp(cbar.ax.xaxis.get_ticklabels(), color='white')

    sm_cb2 = plt.cm.ScalarMappable(cmap=ssfr_cmap, norm=ssfr_norm)
    sm_cb2.set_array([])
    cax2 = fig.add_axes([0.55, 0.05, 0.40, 0.022])
    cbar2 = fig.colorbar(sm_cb2, cax=cax2, orientation='horizontal',
                          extend='both')
    cbar2.set_label(
        r'$\log_{10}(\mathrm{sSFR}\;[\mathrm{yr}^{-1}])$ — target star colour',
        fontsize=9, color='white')
    cbar2.ax.xaxis.set_tick_params(color='white')
    cbar2.outline.set_edgecolor('white')
    plt.setp(cbar2.ax.xaxis.get_ticklabels(), color='white')

    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def compute_radial_skew(target_pos, halo_pos, halo_mass, r_max_mpc,
                          box_size_mpc=None):
    """Weighted radial skew (Chittenden & Tojeiro 2023) for one target.

    Returns the third standardised weighted moment
        mu_3 = sum(w_j * ((r_j - mu_1) / sqrt(mu_2))^3) / sum(w_j)
    where r_j is the radial distance from the target to halo j inside an
    aperture of ``r_max_mpc``, and w_j = Mvir_j.  Self (r=0) is excluded.
    Returns NaN when fewer than three halos fall in the aperture.
    """
    dx = halo_pos[:, 0] - target_pos[0]
    dy = halo_pos[:, 1] - target_pos[1]
    dz = halo_pos[:, 2] - target_pos[2]
    if box_size_mpc:
        dx -= box_size_mpc * np.round(dx / box_size_mpc)
        dy -= box_size_mpc * np.round(dy / box_size_mpc)
        dz -= box_size_mpc * np.round(dz / box_size_mpc)
    r = np.sqrt(dx * dx + dy * dy + dz * dz)

    keep = (r > 0.0) & (r <= r_max_mpc)
    if keep.sum() < 3:
        return np.nan
    r_in = r[keep].astype(np.float64)
    w_in = halo_mass[keep].astype(np.float64)
    W = float(w_in.sum())
    if W <= 0.0:
        return np.nan
    mu1 = float((w_in * r_in).sum() / W)
    mu2 = float((w_in * (r_in - mu1) ** 2).sum() / W)
    if mu2 <= 0.0:
        return np.nan
    z = (r_in - mu1) / np.sqrt(mu2)
    return float((w_in * z ** 3).sum() / W)


def plot_kernel_density_skew_panels(env_results, all_gals, _z_snap,
                                     _run_label, _qmode, out_path,
                                     box_size_mpc=None,
                                     skew_targets=(-3.93, -0.88, 0.50),
                                     skew_aperture_mpc=5.0,
                                     sigma_mpc=None, extent_mpc=None,
                                     z_slab_mpc=None, grid_n=1024):
    """XY kernel-density panels for three targets matching given skew values.

    For each value in ``skew_targets`` the env_results target whose
    Mvir-weighted radial skew (Chittenden & Tojeiro 2023) is closest is
    selected (no repeats).  Each chosen target is rendered as an XY
    projection in the same wide style as ``plot_kernel_density_projections``,
    with the computed skew printed above the panel.
    """
    from matplotlib.colors import LogNorm, TwoSlopeNorm

    found = [r for r in env_results if r.get('found', False)]
    if not found:
        return

    sigmas = list(KERNEL_SIGMAS_MPC)
    sigma  = float(sigma_mpc  if sigma_mpc  is not None else max(sigmas))
    R      = float(extent_mpc if extent_mpc is not None else 3.0 * max(sigmas))
    zslab  = float(z_slab_mpc if z_slab_mpc is not None else 2.0 * sigma)

    # Field: ALL halos above the resolution floor, weighted by Mvir.
    halo_mask = all_gals['mvir'] >= HALO_MASS_FLOOR_MSUN
    cx = all_gals['posx'][halo_mask]
    cy = all_gals['posy'][halo_mask]
    cz = all_gals['posz'][halo_mask]
    cm = all_gals['mvir'][halo_mask]
    halo_pos_full = np.column_stack([cx, cy, cz])

    ssfr_norm = TwoSlopeNorm(vmin=-12.0, vcenter=-11.0, vmax=-9.0)
    ssfr_cmap = plt.cm.bwr_r

    # --- Compute weighted radial skew per target ---
    skews = np.full(len(found), np.nan)
    for i, r in enumerate(found):
        skews[i] = compute_radial_skew(
            r['pos'], halo_pos_full, cm, skew_aperture_mpc,
            box_size_mpc=box_size_mpc)

    finite_mask = np.isfinite(skews)
    n_fin = int(finite_mask.sum())
    if n_fin == 0:
        print('  No targets have a finite radial skew — '
              'skipping skew panels.')
        return
    print(f'  Skew computed for {n_fin}/{len(found)} targets  '
          f'(aperture = {skew_aperture_mpc:.1f} cMpc)')
    finite_vals = skews[finite_mask]
    print(f'  Skew range: [{finite_vals.min():.2f}, '
          f'{finite_vals.max():.2f}]  median={np.median(finite_vals):.2f}')

    # Pick the closest match for each requested skew, no repeats.
    picks = []
    used  = set()
    for target_sk in skew_targets:
        best_i  = -1
        best_d  = np.inf
        for i in range(len(found)):
            if i in used or not finite_mask[i]:
                continue
            d = abs(skews[i] - target_sk)
            if d < best_d:
                best_d = d
                best_i = i
        if best_i < 0:
            print(f'  No remaining target close to skew={target_sk:.2f} — '
                  'panel will be skipped.')
            continue
        picks.append((found[best_i], float(skews[best_i]), float(target_sk)))
        used.add(best_i)
        print(f'    requested mu_3 = {target_sk:>+6.2f}  →  '
              f'GID {found[best_i]["gid"]}  '
              f'(actual mu_3 = {skews[best_i]:+.2f}, '
              f'env={found[best_i]["env_class"]})')

    if not picks:
        print('  No skew matches — skipping figure.')
        return

    grid_axis  = np.linspace(-R, R, grid_n)
    gh, gv     = np.meshgrid(grid_axis, grid_axis)
    norm_2d    = 1.0 / (2.0 * np.pi * sigma ** 2)
    sigma2_inv = 1.0 / (2.0 * sigma ** 2)

    # XY projection per picked target — same kernel as the wide projection
    panels = []
    rho_max_all = 0.0
    for r, actual_sk, target_sk in picks:
        tx, ty, tz = r['pos']
        dx = cx - tx
        dy = cy - ty
        dz = cz - tz
        if box_size_mpc:
            dx -= box_size_mpc * np.round(dx / box_size_mpc)
            dy -= box_size_mpc * np.round(dy / box_size_mpc)
            dz -= box_size_mpc * np.round(dz / box_size_mpc)

        keep = ((np.abs(dz) <= zslab) &
                (np.abs(dx) <= R + 3.0 * sigma) &
                (np.abs(dy) <= R + 3.0 * sigma))
        hdx = dx[keep]
        hdy = dy[keep]
        hm  = cm[keep]
        rho = np.zeros((grid_n, grid_n), dtype=np.float64)
        if len(hm):
            chunk = 32
            for k0 in range(0, len(hm), chunk):
                hx = hdx[k0:k0 + chunk]
                hy = hdy[k0:k0 + chunk]
                hM = hm[k0:k0 + chunk]
                rr = (gh[..., None] - hx) ** 2 + (gv[..., None] - hy) ** 2
                rho += norm_2d * np.sum(hM * np.exp(-rr * sigma2_inv), axis=2)

        # Marker overlay: ALL galaxies (no Mvir cut), inside the slab + window
        gdx = all_gals['posx'] - tx
        gdy = all_gals['posy'] - ty
        gdz = all_gals['posz'] - tz
        if box_size_mpc:
            gdx -= box_size_mpc * np.round(gdx / box_size_mpc)
            gdy -= box_size_mpc * np.round(gdy / box_size_mpc)
            gdz -= box_size_mpc * np.round(gdz / box_size_mpc)
        g_keep = ((np.abs(gdz) <= zslab) &
                  (np.abs(gdx) <= R) &
                  (np.abs(gdy) <= R))
        g_dx = gdx[g_keep]
        g_dy = gdy[g_keep]
        g_sm = all_gals['sm'][g_keep]   # Msun

        panels.append((r, actual_sk, target_sk, rho, g_dx, g_dy, g_sm))
        rho_max_all = max(rho_max_all,
                          float(rho.max()) if rho.size else 0.0)

    # Shared colour scale (same logic as plot_kernel_density_projections)
    norm_img = None
    if rho_max_all > 0:
        nonzero = [p[3][p[3] > 0] for p in panels]
        nonzero = [v for v in nonzero if v.size]
        if nonzero:
            flat = np.concatenate(nonzero)
            vmin = max(float(np.percentile(flat, 1.0)), rho_max_all / 1e7)
        else:
            vmin = rho_max_all / 1e7
        norm_img = LogNorm(vmin=vmin, vmax=rho_max_all)
    cmap = plt.cm.inferno

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.5))
    fig.patch.set_facecolor('black')

    # Fill only the panels we have picks for; hide any remainders
    for ax_idx, (r, actual_sk, target_sk, rho, gxx, gyy, gsm) in enumerate(panels):
        ax = axes[ax_idx]
        ax.set_facecolor('black')
        ax.set_aspect('equal')

        ax.imshow(rho, origin='lower', extent=[-R, R, -R, R],
                  cmap=cmap, norm=norm_img,
                  interpolation='bilinear', zorder=1)

        # All galaxies as markers (no Mvir cut), sized by log stellar mass
        if len(gsm):
            log_sm = np.log10(np.maximum(gsm, 1.0))
            sizes  = 4.0 + 3.0 * np.clip(log_sm - 9.0, 0.0, 3.5)
            ax.scatter(gxx, gyy, s=sizes,
                       facecolor='white', edgecolor='black',
                       linewidths=0.4, alpha=0.85, zorder=2)

        theta = np.linspace(0, 2.0 * np.pi, 200)
        for ss, ls in zip(sigmas, ['--', ':']):
            ax.plot(ss * np.cos(theta), ss * np.sin(theta),
                    color='slateblue', lw=1.4, ls=ls, alpha=0.9, zorder=3,
                    label=(fr'$\sigma={ss:.0f}\,\mathrm{{cMpc}}$'
                           if ax_idx == 0 else None))

        target_ssfr = next((m['ssfr'] for m in r['group_members']
                            if m.get('is_target')), None)
        if (target_ssfr is not None and np.isfinite(target_ssfr)
                and target_ssfr > 0):
            log_t_ssfr = float(np.log10(target_ssfr))
        else:
            log_t_ssfr = -16.0
        t_color = ssfr_cmap(ssfr_norm(log_t_ssfr))
        ax.scatter([0], [0], s=220, marker='*', zorder=4,
                   facecolor=t_color, edgecolor='black', linewidths=0.9)

        ax.set_xlim(-R, R)
        ax.set_ylim(-R, R)
        ax.set_xlabel(r'$\Delta X$ [cMpc]', fontsize=10, color='white')
        ax.set_ylabel(r'$\Delta Y$ [cMpc]', fontsize=10, color='white')
        ax.tick_params(colors='white', labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor('white')

        kd     = r.get('kernel_deltas', [np.nan, np.nan])
        kd_s   = '  '.join(fr'$\delta_K({s:.0f})={d:.2f}$'
                           for s, d in zip(sigmas, kd) if np.isfinite(d))
        ax.set_title(
            fr'$\mu_3 = {actual_sk:+.2f}$  '
            fr'(target ${target_sk:+.2f}$)' + '\n' +
            f'GID {r["gid"]}  —  {r["env_class"]}    '
            f'log $M_{{\\rm host}}$ = {r["host_log_mvir"]:.2f}\n'
            f'{kd_s}',
            fontsize=9, color='white', pad=6)

    for j in range(len(panels), len(axes)):
        axes[j].set_visible(False)

    handles, _ = axes[0].get_legend_handles_labels()
    if handles:
        leg = axes[0].legend(fontsize=8, loc='lower right')
        leg.get_frame().set_facecolor('black')
        leg.get_frame().set_edgecolor('white')
        for t in leg.get_texts():
            t.set_color('white')

    target_sks_str = ',  '.join(f'{s:+.2f}' for s in skew_targets)
    suptitle = (fr'Radial skew triplet — requested $\mu_3 \in \{{ {target_sks_str} \}}$    '
                fr'(aperture = {skew_aperture_mpc:.1f} cMpc, Mvir-weighted)')
    fig.suptitle(suptitle, fontsize=11, color='white', y=0.97)

    fig.tight_layout(rect=[0, 0.14, 1, 0.92])

    log_floor = np.log10(HALO_MASS_FLOOR_MSUN)
    if norm_img is not None:
        sm_cb = plt.cm.ScalarMappable(cmap=cmap, norm=norm_img)
        sm_cb.set_array([])
        cax = fig.add_axes([0.07, 0.05, 0.42, 0.022])
        cbar = fig.colorbar(sm_cb, cax=cax, orientation='horizontal')
        cbar.set_label(
            fr'$\Sigma_{{M_{{\rm vir}}}}\;[\mathrm{{M}}_\odot\,\mathrm{{cMpc}}^{{-2}}]$  '
            fr'($\sigma={sigma:.1f}\,\mathrm{{cMpc}}$, '
            fr'$|\Delta z|<{zslab:.1f}\,\mathrm{{cMpc}}$, '
            fr'$\log M_{{\rm vir}}\geq{log_floor:.1f}$)',
            fontsize=9, color='white')
        cbar.ax.xaxis.set_tick_params(color='white')
        cbar.outline.set_edgecolor('white')
        plt.setp(cbar.ax.xaxis.get_ticklabels(), color='white')

    sm_cb2 = plt.cm.ScalarMappable(cmap=ssfr_cmap, norm=ssfr_norm)
    sm_cb2.set_array([])
    cax2 = fig.add_axes([0.55, 0.05, 0.40, 0.022])
    cbar2 = fig.colorbar(sm_cb2, cax=cax2, orientation='horizontal',
                          extend='both')
    cbar2.set_label(
        r'$\log_{10}(\mathrm{sSFR}\;[\mathrm{yr}^{-1}])$ — target star colour',
        fontsize=9, color='white')
    cbar2.ax.xaxis.set_tick_params(color='white')
    cbar2.outline.set_edgecolor('white')
    plt.setp(cbar2.ax.xaxis.get_ticklabels(), color='white')

    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def plot_target_redshift_evolution(env_results, filepaths, snap_times,
                                    snapshot_redshifts, hubble_h, box_size_mpc,
                                    out_path,
                                    target_gid=None,
                                    target_redshifts=(7.0, 5.0, 2.95),
                                    sigma_mpc=None, extent_mpc=None,
                                    z_slab_mpc=None, grid_n=1024,
                                    apertures_mpc=(1.0, 3.0, 5.0)):
    """Kernel-density XY views of one target tracked across three redshifts.

    Picks the most massive target in ``env_results`` (by stellar mass at the
    selection snap) or uses ``target_gid`` if given.  For each value in
    ``target_redshifts`` the closest available snapshot is selected; at that
    snap every model file is opened, all halos above the resolution floor
    are loaded, the target's main-branch position is located by
    ``GalaxyIndex``, and the Mvir-weighted Gaussian kernel field is built
    around it.  The Chittenden & Tojeiro radial skew μ₃ is evaluated at
    every aperture in ``apertures_mpc``.

    The colour scale is shared across all three panels so densities are
    directly comparable across cosmic time.
    """
    from matplotlib.colors import LogNorm, TwoSlopeNorm

    found = [r for r in env_results if r.get('found', False)]
    if not found:
        print('  No env_results — skipping redshift evolution figure.')
        return

    def _target_sm(r):
        for m in r['group_members']:
            if m.get('is_target'):
                return m.get('sm_msun', 0.0)
        return 0.0

    if target_gid is None:
        best = max(found, key=_target_sm)
    else:
        match = [r for r in found if r['gid'] == int(target_gid)]
        if not match:
            print(f'  Target {target_gid} not in env_results — '
                  'skipping redshift evolution.')
            return
        best = match[0]

    tgid = int(best['gid'])
    sm0  = _target_sm(best)
    sm0_str = f'{np.log10(sm0):.2f}' if sm0 > 0 else 'N/A'
    print(f'  Tracking GID={tgid}  log M*(z_sel)={sm0_str}  '
          f'env={best["env_class"]}')

    # Map requested redshifts → nearest available snapshots
    rs_arr = np.asarray(snapshot_redshifts)
    snap_picks = []
    for z_req in target_redshifts:
        s = int(np.argmin(np.abs(rs_arr - z_req)))
        snap_picks.append((float(z_req), s,
                            float(rs_arr[s]), float(snap_times[s])))
    # Order by increasing snap (oldest to youngest, left to right)
    snap_picks.sort(key=lambda x: x[1])
    print(f'  Requested z: {", ".join(f"{p[0]:.2f}" for p in snap_picks)}')
    print(f'  Matched snaps: '
          f'{", ".join(f"{p[1]}(z={p[2]:.2f})" for p in snap_picks)}')

    sigmas = best.get('kernel_sigmas', list(KERNEL_SIGMAS_MPC))
    sigma  = float(sigma_mpc  if sigma_mpc  is not None else max(sigmas))
    R      = float(extent_mpc if extent_mpc is not None else 3.0 * max(sigmas))
    zslab  = float(z_slab_mpc if z_slab_mpc is not None else 2.0 * sigma)
    apertures = tuple(float(a) for a in apertures_mpc)

    ssfr_norm = TwoSlopeNorm(vmin=-12.0, vcenter=-11.0, vmax=-9.0)
    ssfr_cmap = plt.cm.bwr_r

    # Kernel-density grid setup (Gaussian smoothing, same as the wide map)
    grid_axis  = np.linspace(-R, R, grid_n)
    gh, gv     = np.meshgrid(grid_axis, grid_axis)
    norm_2d    = 1.0 / (2.0 * np.pi * sigma ** 2)
    sigma2_inv = 1.0 / (2.0 * sigma ** 2)

    panels = []        # one dict per snap_pick
    rho_max_all = 0.0

    for (z_req, snap_i, z_actual, t_gyr) in snap_picks:
        t_load = time.time()
        snap_key = f'Snap_{snap_i}'

        halo_xs, halo_ys, halo_zs, halo_ms = [], [], [], []
        target_pos        = None
        target_sm_msun    = np.nan
        target_sfr        = np.nan
        target_gtype      = -1
        target_mvir_msun  = np.nan

        for fp in filepaths:
            with h5.File(fp, 'r') as f:
                if snap_key not in f:
                    continue
                g = f[snap_key]
                if 'Mvir' not in g or 'Posx' not in g:
                    continue

                mvir_msun = (np.array(g['Mvir'], dtype=np.float64)
                             * 1e10 / hubble_h)
                posx = np.array(g['Posx'], dtype=np.float64) / hubble_h
                posy = np.array(g['Posy'], dtype=np.float64) / hubble_h
                posz = np.array(g['Posz'], dtype=np.float64) / hubble_h

                hmask = mvir_msun >= HALO_MASS_FLOOR_MSUN
                if hmask.any():
                    halo_xs.append(posx[hmask])
                    halo_ys.append(posy[hmask])
                    halo_zs.append(posz[hmask])
                    halo_ms.append(mvir_msun[hmask])

                if target_pos is None and 'GalaxyIndex' in g:
                    gids = np.array(g['GalaxyIndex'], dtype=np.int64)
                    idx = np.where(gids == tgid)[0]
                    if len(idx) > 0:
                        i = int(idx[0])
                        target_pos = (float(posx[i]), float(posy[i]),
                                      float(posz[i]))
                        target_mvir_msun = float(mvir_msun[i])
                        if 'StellarMass' in g:
                            sm_v = float(np.array(g['StellarMass'],
                                                  dtype=np.float64)[i])
                            target_sm_msun = sm_v * 1e10 / hubble_h
                        if 'SfrDisk' in g and 'SfrBulge' in g:
                            target_sfr = (
                                float(np.array(g['SfrDisk'],
                                               dtype=np.float64)[i])
                                + float(np.array(g['SfrBulge'],
                                                 dtype=np.float64)[i]))
                        if 'Type' in g:
                            target_gtype = int(np.array(g['Type'],
                                                        dtype=np.int32)[i])

        if not halo_xs or target_pos is None:
            print(f'    [snap {snap_i}] target not found — skipping panel.')
            panels.append({'snap': snap_i, 'z': z_actual, 't': t_gyr,
                            'rho': None})
            continue

        cx = np.concatenate(halo_xs)
        cy = np.concatenate(halo_ys)
        cz = np.concatenate(halo_zs)
        cm = np.concatenate(halo_ms)
        halo_pos = np.column_stack([cx, cy, cz])

        tx, ty, tz_pos = target_pos
        dx = cx - tx
        dy = cy - ty
        dz = cz - tz_pos
        if box_size_mpc:
            dx -= box_size_mpc * np.round(dx / box_size_mpc)
            dy -= box_size_mpc * np.round(dy / box_size_mpc)
            dz -= box_size_mpc * np.round(dz / box_size_mpc)

        # Filter to slab + slightly wider than R so kernel tails are kept
        keep = ((np.abs(dz) <= zslab) &
                (np.abs(dx) <= R + 3.0 * sigma) &
                (np.abs(dy) <= R + 3.0 * sigma))
        hdx = dx[keep]
        hdy = dy[keep]
        hm  = cm[keep]

        # Kernel density grid — Mvir-weighted Gaussian sum on the grid.
        # Chunk = 32 keeps peak (grid, grid, chunk) memory ~256 MB at
        # grid_n=1024 (1024² × 32 × 8 bytes).
        rho = np.zeros((grid_n, grid_n), dtype=np.float64)
        if len(hm):
            chunk = 32
            for k0 in range(0, len(hm), chunk):
                hx = hdx[k0:k0 + chunk]
                hy = hdy[k0:k0 + chunk]
                hM = hm[k0:k0 + chunk]
                rr = (gh[..., None] - hx) ** 2 + (gv[..., None] - hy) ** 2
                rho += norm_2d * np.sum(hM * np.exp(-rr * sigma2_inv), axis=2)

        # Skew at each requested aperture (full loaded halo field)
        sk_by_ap = {ap: compute_radial_skew(target_pos, halo_pos, cm, ap,
                                            box_size_mpc=box_size_mpc)
                    for ap in apertures}

        # Target sSFR for star colour
        with np.errstate(divide='ignore', invalid='ignore'):
            if (np.isfinite(target_sfr) and target_sfr > 0
                    and np.isfinite(target_sm_msun) and target_sm_msun > 0):
                log_target_ssfr = float(np.log10(target_sfr
                                                  / target_sm_msun))
            else:
                log_target_ssfr = -16.0

        panels.append({
            'snap': snap_i, 'z': z_actual, 't': t_gyr,
            'rho': rho, 'hdx': hdx, 'hdy': hdy, 'hm': hm,
            'target_sm': target_sm_msun, 'target_mvir': target_mvir_msun,
            'target_gtype': target_gtype,
            'log_ssfr':   log_target_ssfr,
            'sk_by_ap':   sk_by_ap,
            'n_halos':    int(len(hm)),
            'total_halos': int(cm.size),
        })
        rho_max_all = max(rho_max_all,
                          float(rho.max()) if rho.size else 0.0)

        sk_str = '  '.join(f'μ₃({ap:.0f})={sk_by_ap[ap]:+.2f}'
                            for ap in apertures
                            if np.isfinite(sk_by_ap[ap]))
        type_lbl = {0: 'cent', 1: 'sat', 2: 'orph'}.get(target_gtype, '?')
        print(f'    [snap {snap_i:>3d}  z={z_actual:.2f}  t={t_gyr:.2f} Gyr]  '
              f'halos_loaded={cm.size:,}  in_slab={len(hm):,}  '
              f'logM*={np.log10(target_sm_msun):.2f}  '
              f'type={type_lbl}  {sk_str}  '
              f'[load {_fmt_dt(time.time() - t_load)}]')

    # --- Shared LogNorm across panels for the kernel density heatmap ---
    norm_img = None
    if rho_max_all > 0:
        rho_pos = []
        for p in panels:
            r = p.get('rho')
            if r is not None and r.size:
                rho_pos.append(r[r > 0])
        rho_pos = [v for v in rho_pos if v.size]
        if rho_pos:
            flat = np.concatenate(rho_pos)
            vmin = max(float(np.percentile(flat, 1.0)), rho_max_all / 1e7)
        else:
            vmin = rho_max_all / 1e7
        norm_img = LogNorm(vmin=vmin, vmax=rho_max_all)
    cmap = plt.cm.inferno

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels,
                              figsize=(5.0 * n_panels, 5.6),
                              sharey=False)
    fig.patch.set_facecolor('black')
    if n_panels == 1:
        axes = [axes]

    for ax_idx, p in enumerate(panels):
        ax = axes[ax_idx]
        ax.set_facecolor('black')
        ax.set_aspect('equal')

        # Missing-target panel
        if p.get('rho') is None:
            ax.text(0.5, 0.5, f'GID {tgid}\nnot in Snap_{p["snap"]}',
                    ha='center', va='center', color='grey',
                    transform=ax.transAxes, fontsize=10)
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_edgecolor('white')
            ax.set_title(fr'$z={p["z"]:.2f}$  '
                         fr'($t={p["t"]:.2f}$ Gyr)',
                         fontsize=11, color='white')
            continue

        # Mvir-weighted Gaussian kernel density field
        ax.imshow(p['rho'], origin='lower', extent=[-R, R, -R, R],
                  cmap=cmap, norm=norm_img,
                  interpolation='bilinear', zorder=1)

        # Halo position markers — white fill, black edge, sized by log Mvir
        if p['hm'].size:
            log_m = np.log10(np.maximum(p['hm'], 1.0))
            sizes_dot = 6.0 + 4.0 * np.clip(log_m - 10.5, 0.0, 4.5)
            ax.scatter(p['hdx'], p['hdy'], s=sizes_dot,
                       facecolor='white', edgecolor='black',
                       linewidths=0.4, alpha=0.85, zorder=2)

        # Reference σ_K rings (only those fitting in the window)
        theta = np.linspace(0.0, 2.0 * np.pi, 200)
        for ss, ls in zip(sigmas, ['--', ':']):
            if ss > R * 1.42:
                continue
            ax.plot(ss * np.cos(theta), ss * np.sin(theta),
                    color='slateblue', lw=1.4, ls=ls, alpha=0.9, zorder=5,
                    label=(fr'$\sigma={ss:.0f}\,\mathrm{{cMpc}}$'
                           if ax_idx == 0 else None))

        # Target star — sSFR-coloured, black outline
        t_color = ssfr_cmap(ssfr_norm(p['log_ssfr']))
        ax.scatter([0], [0], s=240, marker='*', zorder=6,
                   facecolor=t_color, edgecolor='black', linewidths=0.9)

        ax.set_xlim(-R, R)
        ax.set_ylim(-R, R)
        ax.set_xlabel(r'$\Delta X$ [cMpc]', fontsize=10, color='white')
        if ax_idx == 0:
            ax.set_ylabel(r'$\Delta Y$ [cMpc]', fontsize=10, color='white')
        ax.tick_params(colors='white', labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor('white')

        # Title — z, t, log M*, target type
        type_lbl = {0: 'cent', 1: 'sat', 2: 'orph'}.get(
            p['target_gtype'], '?')
        log_sm_str = (f'{np.log10(p["target_sm"]):.2f}'
                      if p['target_sm'] > 0 else 'N/A')
        log_mv_str = (f'{np.log10(p["target_mvir"]):.2f}'
                      if p['target_mvir'] > 0 else 'N/A')
        title_top = (fr'$z = {p["z"]:.2f}$  '
                     fr'($t={p["t"]:.2f}$ Gyr)' + '\n'
                     fr'log $M_* = {log_sm_str}$  '
                     fr'log $M_{{\rm vir}} = {log_mv_str}$  '
                     fr'[{type_lbl}]')
        ax.set_title(title_top, fontsize=9, color='white', pad=4)

        # Skew annotation in panel — bottom left
        sk_lines = []
        for ap in apertures:
            sk_v = p['sk_by_ap'][ap]
            if np.isfinite(sk_v):
                sk_lines.append(fr'$\mu_3({ap:.0f})={sk_v:+.2f}$')
            else:
                sk_lines.append(fr'$\mu_3({ap:.0f})=$ N/A')
        ax.text(0.03, 0.03, '\n'.join(sk_lines),
                transform=ax.transAxes, fontsize=9, color='white',
                ha='left', va='bottom',
                bbox=dict(facecolor='black', edgecolor='white',
                          alpha=0.75, boxstyle='round,pad=0.35',
                          linewidth=0.5))

    # Suptitle
    fig.suptitle(fr'GID {tgid} — kernel density evolution'
                 fr'   ($\sigma={sigma:.1f}$ cMpc, '
                 fr'$|\Delta z|<{zslab:.1f}$ cMpc, '
                 fr'$\log M_{{\rm vir}}\geq{np.log10(HALO_MASS_FLOOR_MSUN):.1f}$)',
                 fontsize=11, color='white', y=0.99)

    fig.tight_layout(rect=[0, 0.07, 1, 0.94])

    if norm_img is not None:
        sm_cb = plt.cm.ScalarMappable(cmap=cmap, norm=norm_img)
        sm_cb.set_array([])
        cax = fig.add_axes([0.07, 0.025, 0.42, 0.022])
        cbar = fig.colorbar(sm_cb, cax=cax, orientation='horizontal')
        cbar.set_label(
            r'$\Sigma_{M_{\rm vir}}\;[\mathrm{M}_\odot\,\mathrm{cMpc}^{-2}]$',
            fontsize=9, color='white')
        cbar.ax.xaxis.set_tick_params(color='white')
        cbar.outline.set_edgecolor('white')
        plt.setp(cbar.ax.xaxis.get_ticklabels(), color='white')

    sm_cb2 = plt.cm.ScalarMappable(cmap=ssfr_cmap, norm=ssfr_norm)
    sm_cb2.set_array([])
    cax2 = fig.add_axes([0.55, 0.025, 0.40, 0.022])
    cbar2 = fig.colorbar(sm_cb2, cax=cax2, orientation='horizontal',
                          extend='both')
    cbar2.set_label(
        r'$\log_{10}(\mathrm{sSFR}\;[\mathrm{yr}^{-1}])$ — target star colour',
        fontsize=9, color='white')
    cbar2.ax.xaxis.set_tick_params(color='white')
    cbar2.outline.set_edgecolor('white')
    plt.setp(cbar2.ax.xaxis.get_ticklabels(), color='white')

    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def compute_skew_history(filepaths, snap_sel, target_gids, snap_times,
                          hubble_h, box_size_mpc,
                          apertures_mpc=(1.0, 3.0, 5.0),
                          progress_every=1):
    """Mvir-weighted radial skew per target at each snapshot 0..snap_sel,
    evaluated at every aperture in ``apertures_mpc`` in a single pass.

    For each snapshot the full halo field above HALO_MASS_FLOOR_MSUN is
    loaded across all model files; targets are located by GalaxyIndex (so
    each target is tracked along its own main branch through cosmic time)
    and skew is evaluated at every aperture from that snapshot's position.

    Returns ``{aperture_mpc: {int(gid): [(t_gyr, mu_3), ...]}}`` — one inner
    dict per aperture, with one entry per snapshot where the target was
    located and the skew is finite.
    """
    target_gids_set = set(int(g) for g in target_gids)
    apertures = tuple(float(a) for a in apertures_mpc)
    history = {a: {tg: [] for tg in target_gids_set} for a in apertures}

    ap_str = ', '.join(f'{a:.1f}' for a in apertures)
    print(f'  Computing skew history for {len(target_gids_set)} targets '
          f'over snapshots 0..{snap_sel}  '
          f'(apertures = [{ap_str}] cMpc)')
    print(f'  files to scan per snap : {len(filepaths)}')

    t_start = time.time()

    for s in range(snap_sel + 1):
        t_snap_start = time.time()
        snap_key = f'Snap_{s}'
        t_gyr    = float(snap_times[s])

        halo_xs, halo_ys, halo_zs, halo_ms = [], [], [], []
        target_pos = {}

        for fp in filepaths:
            with h5.File(fp, 'r') as f:
                if snap_key not in f:
                    continue
                g = f[snap_key]
                if 'Mvir' not in g or 'Posx' not in g:
                    continue

                mvir_msun = (np.array(g['Mvir'],  dtype=np.float64)
                             * 1e10 / hubble_h)
                posx_mpc  = np.array(g['Posx'],  dtype=np.float64) / hubble_h
                posy_mpc  = np.array(g['Posy'],  dtype=np.float64) / hubble_h
                posz_mpc  = np.array(g['Posz'],  dtype=np.float64) / hubble_h

                halo_mask = mvir_msun >= HALO_MASS_FLOOR_MSUN
                if halo_mask.any():
                    halo_xs.append(posx_mpc[halo_mask])
                    halo_ys.append(posy_mpc[halo_mask])
                    halo_zs.append(posz_mpc[halo_mask])
                    halo_ms.append(mvir_msun[halo_mask])

                if 'GalaxyIndex' in g and target_gids_set:
                    gids = np.array(g['GalaxyIndex'], dtype=np.int64)
                    remaining = [tg for tg in target_gids_set
                                 if tg not in target_pos]
                    if remaining:
                        rem_arr = np.array(remaining, dtype=np.int64)
                        hit_mask = np.isin(gids, rem_arr)
                        if hit_mask.any():
                            hit_idx = np.where(hit_mask)[0]
                            for i in hit_idx:
                                tg = int(gids[i])
                                if tg in target_gids_set and tg not in target_pos:
                                    target_pos[tg] = (float(posx_mpc[i]),
                                                       float(posy_mpc[i]),
                                                       float(posz_mpc[i]))

        if not halo_xs or not target_pos:
            continue

        cx = np.concatenate(halo_xs)
        cy = np.concatenate(halo_ys)
        cz = np.concatenate(halo_zs)
        cm = np.concatenate(halo_ms)
        halo_pos = np.column_stack([cx, cy, cz])

        # Evaluate at every aperture from the same loaded snapshot field.
        for tg, tpos in target_pos.items():
            for ap in apertures:
                sk = compute_radial_skew(tpos, halo_pos, cm, ap,
                                          box_size_mpc=box_size_mpc)
                if np.isfinite(sk):
                    history[ap][tg].append((t_gyr, sk))

        if progress_every and (s % progress_every == 0):
            # Targets that have at least one finite history point in any
            # aperture so far
            seen = set()
            for ap_hist in history.values():
                seen.update(tg for tg, h in ap_hist.items() if h)
            n_halos = sum(arr.size for arr in halo_xs) if halo_xs else 0
            elapsed = time.time() - t_start
            snap_dt = time.time() - t_snap_start
            remaining = snap_sel - s
            eta = (elapsed / max(s, 1)) * remaining if s else None
            eta_str = f'  ETA {_fmt_dt(eta)}' if eta else ''
            print(f'    snap {s:>3d}  t={t_gyr:6.2f} Gyr  '
                  f'halos={n_halos:>7,}  '
                  f'located={len(target_pos):>4d}/{len(target_gids_set)}  '
                  f'cumulative={len(seen):>4d}  '
                  f'[snap {_fmt_dt(snap_dt)}, total {_fmt_dt(elapsed)}'
                  f'{eta_str}]')

    return history


def plot_skew_history(history_by_ap, snap_times, snap_sel, _run_label,
                       _qmode, skew_targets, out_path,
                       picks_aperture_mpc=5.0):
    """Skew vs cosmic time, one panel per aperture.

    ``history_by_ap`` is the dict-of-dicts returned by
    ``compute_skew_history`` keyed by aperture (cMpc).  Picks are made *once*
    using the targets' skews at ``picks_aperture_mpc`` (so the highlighted
    triplet matches ``plot_kernel_density_skew_panels``); the same three GIDs
    are then drawn as solid coloured lines in every panel, with everything
    else as thin dashed grey.
    """
    if not history_by_ap:
        print('  No skew history to plot.')
        return

    apertures = sorted(history_by_ap.keys())

    # Triplet picked from the picks_aperture history at the selection snap
    pick_ap = float(picks_aperture_mpc)
    if pick_ap not in history_by_ap:
        # Fall back to the closest available aperture
        pick_ap = min(apertures, key=lambda a: abs(a - picks_aperture_mpc))
    pick_hist   = history_by_ap[pick_ap]
    final_skews = {gid: h[-1][1] for gid, h in pick_hist.items() if h}

    picks = []
    used  = set()
    for target_sk in skew_targets:
        best_gid, best_d = None, np.inf
        for gid, fsk in final_skews.items():
            if gid in used or not np.isfinite(fsk):
                continue
            d = abs(fsk - target_sk)
            if d < best_d:
                best_d, best_gid = d, gid
        if best_gid is None:
            continue
        picks.append((best_gid, float(target_sk), float(final_skews[best_gid])))
        used.add(best_gid)

    picked_gids = [p[0] for p in picks]
    pick_colors = ['gold', 'tomato', 'cyan']

    n_panels = len(apertures)
    fig, axes = plt.subplots(1, n_panels,
                              figsize=(5.0 * n_panels, 5.0),
                              sharey=False)
    fig.patch.set_facecolor('black')
    if n_panels == 1:
        axes = [axes]

    t_snap = float(snap_times[snap_sel])

    for ap_idx, ap in enumerate(apertures):
        ax  = axes[ap_idx]
        ax.set_facecolor('black')

        ap_hist = history_by_ap[ap]

        # Dashed grey for non-picked targets
        n_dashed = 0
        for gid, hist in ap_hist.items():
            if gid in picked_gids or not hist:
                continue
            t  = [p[0] for p in hist]
            sk = [p[1] for p in hist]
            ax.plot(t, sk, color='lightgrey', ls='--', lw=0.7, alpha=0.45,
                    zorder=1)
            n_dashed += 1

        # Solid coloured lines for the matched triplet
        for i, gid in enumerate(picked_gids):
            hist = ap_hist.get(gid, [])
            if not hist:
                continue
            t  = [p[0] for p in hist]
            sk = [p[1] for p in hist]
            col = pick_colors[i % len(pick_colors)]
            actual_now = hist[-1][1]
            label = None
            if ap_idx == 0:
                req_sk = picks[i][1]
                label = (fr'GID {gid}  ($\mu_3^{{\rm req}}={req_sk:+.2f}$, '
                         fr'$\mu_3^{{\rm now}}={actual_now:+.2f}$)')
            ax.plot(t, sk, color=col, lw=2.2, alpha=1.0, zorder=5,
                    label=label)

        ax.axvline(t_snap, color='white', ls=':', lw=1.0, alpha=0.5)
        ax.axhline(0.0,    color='white', ls='-', lw=0.4, alpha=0.3)

        ax.set_xlabel('Cosmic time [Gyr]', fontsize=11, color='white')
        if ap_idx == 0:
            ax.set_ylabel(r'Radial skew $\mu_3$', fontsize=11, color='white')
        ax.set_title(fr'aperture = {ap:.1f} cMpc',
                     fontsize=11, color='white')
        ax.tick_params(colors='white', labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor('white')

        # X-range bounded to where we actually have data
        first_t = min((h[0][0] for h in ap_hist.values() if h),
                      default=t_snap)
        ax.set_xlim(first_t, t_snap)

        if ap_idx == 0 and picks:
            leg = ax.legend(fontsize=8, loc='best', framealpha=0.85)
            leg.get_frame().set_facecolor('black')
            leg.get_frame().set_edgecolor('white')
            for txt in leg.get_texts():
                txt.set_color('white')

        # Per-aperture summary: distribution of final μ₃ across all targets
        all_finals = np.array([h[-1][1] for h in ap_hist.values() if h],
                               dtype=float)
        print(f'  aperture={ap:.1f} cMpc: {n_dashed} dashed + '
              f'{len(picked_gids)} highlighted picks')
        print(f'    μ₃ @ z_sel across all targets : '
              f'{_array_stats(all_finals)}')
        # Per-pick trajectory milestones at this aperture
        for i, gid in enumerate(picked_gids):
            hist = ap_hist.get(gid, [])
            if not hist:
                print(f'    pick {i+1} GID {gid}: no data at this aperture')
                continue
            arr_sk = np.array([p[1] for p in hist], dtype=float)
            sk_first, sk_last = float(arr_sk[0]), float(arr_sk[-1])
            sk_peak  = float(np.nanmax(arr_sk))
            sk_trough = float(np.nanmin(arr_sk))
            trend = sk_last - sk_first
            print(f'    pick {i+1} GID {gid}: '
                  f'μ₃ first={sk_first:+.2f}  last={sk_last:+.2f}  '
                  f'min={sk_trough:+.2f}  max={sk_peak:+.2f}  '
                  f'Δ(last−first)={trend:+.2f}  '
                  f'N points={len(hist)}')

    fig.suptitle(fr'Radial skew history — picks from {pick_ap:.1f} cMpc '
                  fr'aperture',
                  fontsize=11, color='white', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


# ---------------------------------------------------------------------------
# Terminal output helpers
# ---------------------------------------------------------------------------

def _fmt_or_na(value, fmt='{:.2f}', na='   N/A'):
    """Format a finite numeric value, or return the NA placeholder."""
    if value is None:
        return na
    try:
        v = float(value)
    except (TypeError, ValueError):
        return na
    if not np.isfinite(v):
        return na
    return fmt.format(v)


def _array_stats(arr, fmt='{:.2f}'):
    """Return a compact 'med=X  mean=X  std=X  [min, max]  N=N' summary."""
    if arr is None:
        return 'no data'
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 'all NaN'
    return (f'median={fmt.format(np.median(a))}  '
            f'mean={fmt.format(np.mean(a))}  '
            f'std={fmt.format(np.std(a))}  '
            f'[{fmt.format(a.min())}, {fmt.format(a.max())}]  '
            f'N={a.size}')


def print_target_catalogue(csv_data):
    """Pre-flight table of every target galaxy's key properties."""
    gids = csv_data.get('GalaxyIndex')
    if gids is None or len(gids) == 0:
        return
    n = len(gids)

    type_names = {0: 'cent', 1: 'sat', 2: 'orph'}
    log_sm    = csv_data.get('log_StellarMass_Msun')
    log_mv    = csv_data.get('log_Mvir_Msun')
    log_mbh   = csv_data.get('log_BlackHoleMass_Msun')
    bt        = csv_data.get('bulge_to_total')
    age       = csv_data.get('mw_stellar_age_Gyr')
    gtype_arr = csv_data.get('Type')
    sfrd      = csv_data.get('SfrDisk')
    sfrb      = csv_data.get('SfrBulge')
    sfr_total = ((sfrd + sfrb) if (sfrd is not None and sfrb is not None)
                 else sfrd)

    print('\n--- TARGET CATALOGUE ---')
    hdr = (f'  {"GID":>14}  {"type":<5}  {"logM*":>6}  {"logSFR":>7}  '
           f'{"logsSFR":>8}  {"B/T":>5}  {"logMvir":>7}  {"logMBH":>7}  '
           f'{"age[Gyr]":>9}')
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))

    for i in range(n):
        gid = int(gids[i])
        ty  = (type_names.get(int(gtype_arr[i]), '?')
               if gtype_arr is not None else '?')

        # Derived SFR / sSFR (log)
        log_sfr_str  = '   N/A'
        log_ssfr_str = '   N/A'
        if sfr_total is not None and log_sm is not None:
            sfr_v = float(sfr_total[i])
            sm_v  = (10.0 ** float(log_sm[i])
                     if np.isfinite(log_sm[i]) else 0.0)
            if np.isfinite(sfr_v) and sfr_v > 0:
                log_sfr_str = f'{np.log10(sfr_v):.2f}'
                if sm_v > 0:
                    log_ssfr_str = f'{np.log10(sfr_v / sm_v):.2f}'

        print(f'  {gid:>14}  {ty:<5}  '
              f'{_fmt_or_na(log_sm[i]  if log_sm  is not None else None, "{:.2f}"):>6}  '
              f'{log_sfr_str:>7}  '
              f'{log_ssfr_str:>8}  '
              f'{_fmt_or_na(bt[i]      if bt      is not None else None, "{:.3f}"):>5}  '
              f'{_fmt_or_na(log_mv[i]  if log_mv  is not None else None, "{:.2f}"):>7}  '
              f'{_fmt_or_na(log_mbh[i] if log_mbh is not None else None, "{:.2f}"):>7}  '
              f'{_fmt_or_na(age[i]     if age     is not None else None, "{:.2f}"):>9}')

    # Aggregate stats
    print()
    if log_sm is not None:
        print(f'  log M*  : {_array_stats(log_sm)}')
    if log_mv is not None:
        print(f'  log Mvir: {_array_stats(log_mv)}')
    if log_mbh is not None:
        print(f'  log MBH : {_array_stats(log_mbh)}')
    if bt is not None:
        print(f'  B/T     : {_array_stats(bt, "{:.3f}")}')


def _banner(title, char='='):
    """Big section divider — prints a blank line, the title, and a rule."""
    rule = char * max(60, len(title) + 4)
    print(f'\n{rule}\n  {title}\n{rule}')


def _step(label):
    """Mid-level step indicator."""
    print(f'\n>>> {label}')


def _fmt_dt(dt):
    if dt < 1.0:
        return f'{dt*1000:.0f} ms'
    if dt < 60.0:
        return f'{dt:.1f} s'
    m, s = divmod(dt, 60.0)
    return f'{int(m)}m {s:.0f}s'


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='sSFR vs cosmic time for galaxies from spatial_distribution.py')
    p.add_argument('csv_file',       help='CSV from spatial_distribution.py')
    p.add_argument('output_folder',  help='Folder with model_*.hdf5 files')
    p.add_argument('--snapshot',     type=int,  default=None,
                   help='Override snapshot number')
    p.add_argument('--galaxy-id',    type=int,  default=None,
                   help='GalaxyIndex to follow (must be in the CSV); '
                        'defaults to the most massive galaxy')
    p.add_argument('--output-dir',   type=str,  default=None,
                   help='Output directory (default: <output_folder>/plots/history/)')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_total = time.time()
    args = parse_args()

    _banner('CATALOGUE & PARAMETERS')
    z_sel, run_label, qmode = parse_csv_metadata(args.csv_file)
    if z_sel is None:
        sys.exit('Could not parse redshift from CSV filename.')
    print(f'  CSV file       : {args.csv_file}')
    print(f'  z_sel          : {z_sel:.2f}')
    print(f'  run label      : {run_label}')
    print(f'  mode           : {qmode}')

    t0 = time.time()
    csv_data = load_csv(args.csv_file)
    if not csv_data or 'GalaxyIndex' not in csv_data:
        sys.exit('CSV is empty or missing GalaxyIndex column.')
    galaxy_ids = csv_data['GalaxyIndex'].astype(np.int64)
    print(f'  catalogue rows : {len(galaxy_ids)}  [load {_fmt_dt(time.time()-t0)}]')

    filepaths = sorted(glob.glob(os.path.join(args.output_folder, 'model_*.hdf5')))
    if not filepaths:
        sys.exit(f'No model_*.hdf5 files found in {args.output_folder}')
    print(f'  HDF5 files     : {len(filepaths)}')

    t0 = time.time()
    params = read_header(filepaths[0])
    h      = params['hubble_h']
    print(f'  header read    : h={h:.4f}  '
          f'Om={params["omega_matter"]:.3f}  '
          f'OL={params["omega_lambda"]:.3f}  '
          f'box={params["box_size"]:.1f} Mpc/h  '
          f'm_p={params["particle_mass"]:.2e}  '
          f'[{_fmt_dt(time.time()-t0)}]')
    print(f'  available snaps: {len(params["available_snapshots"])}')

    if args.snapshot is not None:
        snap   = args.snapshot
        z_snap = float(params['snapshot_redshifts'][snap])
    else:
        snap, z_snap = find_snap_for_redshift(params, z_sel)
    print(f'  selection      : Snap_{snap}  z = {z_snap:.4f}')

    if not params['save_full_sfh']:
        sys.exit('SaveFullSFH=0 — no SFH arrays available.')

    t0 = time.time()
    snap_times = snapshot_times_gyr(
        params['snapshot_redshifts'],
        params['omega_matter'], params['omega_lambda'], h)
    print(f'  cosmic times   : t(z={z_snap:.2f})={snap_times[snap]:.2f} Gyr  '
          f'[{_fmt_dt(time.time()-t0)}]')

    _banner('GALAXY SFH LOAD')
    t0 = time.time()
    gal_data = load_galaxy_sfh(filepaths, snap, galaxy_ids, h)
    print(f'  matched {len(gal_data)} / {len(galaxy_ids)} targets  '
          f'[{_fmt_dt(time.time()-t0)}]')
    missing = [int(g) for g in galaxy_ids if int(g) not in gal_data]
    if missing:
        print(f'  Missing GIDs ({len(missing)}):')
        for gid in missing:
            print(f'    {gid}')

    out_dir = args.output_dir or os.path.join(args.output_folder, 'plots', 'history')
    os.makedirs(out_dir, exist_ok=True)
    tag = f'z{z_snap:.2f}_{run_label}_{qmode}'
    print(f'  output dir     : {out_dir}')
    print(f'  filename tag   : {tag}')

    print_target_catalogue(csv_data)

    _banner('BASIC HISTORY FIGURES')
    _step('sSFR vs cosmic time')
    t0 = time.time()
    plot_ssfr_vs_cosmic_time(
        gal_data, snap_times, snap, h,
        z_snap, run_label, qmode,
        os.path.join(out_dir, f'ssfr_history_{tag}.pdf'))
    print(f'    rendered in {_fmt_dt(time.time()-t0)}')

    _step('Stellar mass vs cosmic time')
    t0 = time.time()
    plot_smh_vs_cosmic_time(
        gal_data, snap_times, snap, h,
        z_snap, run_label, qmode,
        os.path.join(out_dir, f'smh_history_{tag}.pdf'))
    print(f'    rendered in {_fmt_dt(time.time()-t0)}')

    # Stellar mass history — target galaxy (specified or most massive)
    if gal_data:
        _step('Main-branch + mergers history')
        if args.galaxy_id is not None:
            if args.galaxy_id not in gal_data:
                sys.exit(f'--galaxy-id {args.galaxy_id} not found in CSV/snapshot. '
                         f'Available IDs: {sorted(gal_data.keys())}')
            target_gid = args.galaxy_id
        else:
            target_gid = max(gal_data, key=lambda gid: gal_data[gid]['sm'])
        target_sm  = gal_data[target_gid]['sm'] * 1e10 / h
        print(f'    target GID={target_gid}  M*={target_sm:.3e} Msun')
        t0 = time.time()
        branch       = load_main_branch(filepaths, snap, target_gid, snap_times, h)
        print(f'    main branch  : {len(branch)} snapshots  '
              f'[{_fmt_dt(time.time()-t0)}]')
        t0 = time.time()
        merger_gids  = load_most_massive_merger(filepaths, snap, target_gid, snap_times, h)
        print(f'    merger scan  : {len(merger_gids) if merger_gids else 0} mergers '
              f'found  [{_fmt_dt(time.time()-t0)}]')
        t0 = time.time()
        merger_branches = [
            load_main_branch(filepaths, snap, gid, snap_times, h)
            for gid in merger_gids
        ]
        print(f'    merger branches loaded  [{_fmt_dt(time.time()-t0)}]')
        t0 = time.time()
        plot_main_branch_history(
            branch, snap_times, snap, z_snap,
            run_label, qmode,
            os.path.join(out_dir, f'main_branch_history_{tag}.pdf'),
            galaxy_id=target_gid,
            merger_branches=merger_branches)
        print(f'    rendered in {_fmt_dt(time.time()-t0)}')

    _banner('SCALING & STRUCTURAL FIGURES')
    _step('Scaling relations grid')
    t0 = time.time()
    plot_scaling_relations(
        csv_data, h, z_snap, run_label, qmode,
        os.path.join(out_dir, f'scaling_relations_{tag}.pdf'))
    print(f'    rendered in {_fmt_dt(time.time()-t0)}')

    _step('Structural & kinematic grid')
    t0 = time.time()
    plot_structural_kinematics(
        csv_data, h, z_snap, run_label, qmode,
        os.path.join(out_dir, f'structural_kinematics_{tag}.pdf'))
    print(f'    rendered in {_fmt_dt(time.time()-t0)}')

    _banner('ENVIRONMENT ANALYSIS')
    _step('Loading all galaxies at selection snapshot')
    t0 = time.time()
    all_gals = load_snapshot_all_galaxies(
        filepaths, snap, params['particle_mass'], h)
    print(f'    load complete in {_fmt_dt(time.time()-t0)}')
    if all_gals is not None:
        box_size_mpc = params['box_size'] / h if params['box_size'] > 0 else None
        box_str = f'box={box_size_mpc:.1f} Mpc' if box_size_mpc else 'box size unknown'
        print(f'  Loaded {len(all_gals["gid"]):,} galaxies  ({box_str})')

        _step('Compute per-target environment (FoF + aperture + kernel δ_K)')
        t0 = time.time()
        env_results = compute_environment(
            galaxy_ids, all_gals,
            box_size_mpc=box_size_mpc)
        n_found = sum(1 for r in env_results if r.get('found', False))
        print(f'    {n_found}/{len(galaxy_ids)} targets located  '
              f'[{_fmt_dt(time.time()-t0)}]')

        print_environment_report(env_results, z_snap, run_label, qmode)

        _step('Environment summary figure')
        t0 = time.time()
        plot_environment_summary(
            env_results, z_snap, run_label, qmode,
            os.path.join(out_dir, f'environment_summary_{tag}.pdf'))
        print(f'    rendered in {_fmt_dt(time.time()-t0)}')

        _step('Group maps (all targets)')
        t0 = time.time()
        plot_group_maps(
            env_results, all_gals, z_snap, run_label, qmode,
            os.path.join(out_dir, f'group_maps_{tag}.pdf'),
            box_size_mpc=box_size_mpc)
        print(f'    rendered in {_fmt_dt(time.time()-t0)}')

        _step('Group maps (overdense subset)')
        t0 = time.time()
        plot_group_maps(
            env_results, all_gals, z_snap, run_label, qmode,
            os.path.join(out_dir, f'group_maps_overdense_{tag}.pdf'),
            box_size_mpc=box_size_mpc, min_overdensity=15)
        print(f'    rendered in {_fmt_dt(time.time()-t0)}')

        _banner('KERNEL DENSITY FIGURES')
        _step('Per-target Σ_Mvir maps')
        t0 = time.time()
        plot_kernel_density_maps(
            env_results, all_gals, z_snap, run_label, qmode,
            os.path.join(out_dir, f'kernel_density_maps_{tag}.pdf'),
            box_size_mpc=box_size_mpc, sigma_mpc=2.0)
        print(f'    rendered in {_fmt_dt(time.time()-t0)}')

        _step('Per-target Σ_Mvir maps (overdense subset, δ_K ≥ 5)')
        t0 = time.time()
        plot_kernel_density_maps(
            env_results, all_gals, z_snap, run_label, qmode,
            os.path.join(out_dir, f'kernel_density_maps_overdense_{tag}.pdf'),
            box_size_mpc=box_size_mpc, min_overdensity=5.0, sigma_mpc=2.0)
        print(f'    rendered in {_fmt_dt(time.time()-t0)}')

        _step('XY/XZ/YZ projections for most massive target')
        t0 = time.time()
        plot_kernel_density_projections(
            env_results, all_gals, z_snap, run_label, qmode,
            os.path.join(out_dir, f'kernel_density_projections_{tag}.pdf'),
            box_size_mpc=box_size_mpc, sigma_mpc=2.0)
        print(f'    rendered in {_fmt_dt(time.time()-t0)}')

        _step('Skew-triplet density panels')
        t0 = time.time()
        plot_kernel_density_skew_panels(
            env_results, all_gals, z_snap, run_label, qmode,
            os.path.join(out_dir, f'kernel_density_skew_panels_{tag}.pdf'),
            box_size_mpc=box_size_mpc, sigma_mpc=2.0)
        print(f'    rendered in {_fmt_dt(time.time()-t0)}')

        _step('Most-massive target — kernel density at z = 7, 5, 2.95')
        t0 = time.time()
        plot_target_redshift_evolution(
            env_results, filepaths, snap_times,
            params['snapshot_redshifts'], h, box_size_mpc,
            os.path.join(out_dir, f'target_redshift_evolution_{tag}.pdf'),
            target_redshifts=(7.0, 5.0, 2.95),
            sigma_mpc=2.0)
        print(f'    rendered in {_fmt_dt(time.time()-t0)}')

        _banner('SKEW HISTORY (multi-aperture)')
        _step('Compute μ_3(t) for every target at 1, 3, 5 cMpc')
        t0 = time.time()
        skew_history = compute_skew_history(
            filepaths, snap, galaxy_ids, snap_times, h,
            box_size_mpc, apertures_mpc=(1.0, 3.0, 5.0))
        print(f'    skew history complete in {_fmt_dt(time.time()-t0)}')

        _step('Render skew-history figure')
        t0 = time.time()
        plot_skew_history(
            skew_history, snap_times, snap, run_label, qmode,
            skew_targets=(-3.93, -0.88, 0.50),
            out_path=os.path.join(out_dir, f'skew_history_{tag}.pdf'),
            picks_aperture_mpc=5.0)
        print(f'    skew-history figure rendered in {_fmt_dt(time.time()-t0)}')
    else:
        print('  No galaxies loaded — skipping environment analysis.')

    _banner('DONE')
    print(f'  total elapsed: {_fmt_dt(time.time() - t_total)}')
    print(f'  outputs in   : {out_dir}')


if __name__ == '__main__':
    main()

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

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator

QUIESCENT_MODES = {
    'strict', 'loose', 'loose_massive', 'massive_bt', 'loose_massive_bt', 'karls_galaxy',
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

    for i, (_, d) in enumerate(gals):
        t, log_ssfr = compute_ssfr_history(d['sfh'], snap_times, snap_num, hubble_h)
        ax.plot(t, log_ssfr, color=colors[i], lw=1.4, alpha=0.85)

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

    for i, (_, d) in enumerate(gals):
        t, log_mstar = compute_smh(d['sfh'], snap_times, snap_num, hubble_h)
        ax.plot(t, log_mstar, color=colors[i], lw=1.4, alpha=0.85)

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

    # Terminal statistics
    n_gals = len(log_sm)
    sm_fin = log_sm[np.isfinite(log_sm)]
    print(f'\n=== Scaling relations  z={z_snap:.2f}  {run_label}  [{qmode}]  N={n_gals} ===')
    print(f'  {"Stellar mass [log Msun]":<22}: N={len(sm_fin):>5}  '
          f'median={np.median(sm_fin):>8.3f}  mean={np.mean(sm_fin):>8.3f}  '
          f'std={np.std(sm_fin):>7.3f}  [{np.min(sm_fin):.3f}, {np.max(sm_fin):.3f}]')
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
    for fp in filepaths:
        with h5.File(fp, 'r') as f:
            if snap_key not in f or 'StellarMass' not in f[snap_key]:
                continue
            g    = f[snap_key]
            sm_r = np.array(g['StellarMass'], dtype=np.float64)
            mask = sm_r >= particle_mass
            if mask.sum() == 0:
                continue

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
    results    = []

    for tgid in target_gids:
        tgid = int(tgid)
        if tgid not in gid_to_idx:
            results.append({'gid': tgid, 'found': False})
            continue

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
    print(f'\n=== Environment analysis  z={z_snap:.2f}  {run_label}  '
          f'[{qmode}]  N={len(found)}  '
          f'R=({r_in:.1f}, {r_out:.1f}) Mpc  '
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
           f'{"N(<{:.1f})".format(r_out):>8}  {"δ({:.1f})".format(r_out):>8}')
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
        print(f'  {r["gid"]:>12}  {r["env_class"]:<10}  {mv_s:>10}  '
              f'{r["n_group_members"]:>5}  {dn_s:>12}  '
              f'{r["n_inner"]:>8}  {di_s:>8}  '
              f'{r["n_outer"]:>8}  {do_s:>8}')

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
    n_members    = np.array([r['n_group_members'] for r in found])
    r_in         = found[0]['r_inner']
    r_out        = found[0]['r_outer']

    cls_colors = {'Isolated': '#4CAF50', 'Pair': '#8BC34A',
                  'Field group': '#CDDC39', 'Group': '#FF9800', 'Cluster': '#F44336'}
    cls_names  = [c for c in ['Isolated', 'Pair', 'Field group', 'Group', 'Cluster']
                  if c in classes]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
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
    args = parse_args()

    z_sel, run_label, qmode = parse_csv_metadata(args.csv_file)
    if z_sel is None:
        sys.exit('Could not parse redshift from CSV filename.')
    print(f'CSV: z={z_sel:.2f}  run={run_label}  mode={qmode}')

    csv_data = load_csv(args.csv_file)
    if not csv_data or 'GalaxyIndex' not in csv_data:
        sys.exit('CSV is empty or missing GalaxyIndex column.')
    galaxy_ids = csv_data['GalaxyIndex'].astype(np.int64)
    print(f'Galaxies: {len(galaxy_ids)}')

    filepaths = sorted(glob.glob(os.path.join(args.output_folder, 'model_*.hdf5')))
    if not filepaths:
        sys.exit(f'No model_*.hdf5 files found in {args.output_folder}')
    print(f'HDF5 files: {len(filepaths)}')
    params = read_header(filepaths[0])
    h      = params['hubble_h']

    if args.snapshot is not None:
        snap   = args.snapshot
        z_snap = float(params['snapshot_redshifts'][snap])
    else:
        snap, z_snap = find_snap_for_redshift(params, z_sel)
    print(f'Snap_{snap}  z = {z_snap:.4f}')

    if not params['save_full_sfh']:
        sys.exit('SaveFullSFH=0 — no SFH arrays available.')

    snap_times = snapshot_times_gyr(
        params['snapshot_redshifts'],
        params['omega_matter'], params['omega_lambda'], h)

    gal_data = load_galaxy_sfh(filepaths, snap, galaxy_ids, h)
    print(f'Matched: {len(gal_data)} / {len(galaxy_ids)}')
    missing = [int(g) for g in galaxy_ids if int(g) not in gal_data]
    if missing:
        print(f'  Missing GIDs ({len(missing)}):')
        for gid in missing:
            print(f'    {gid}')

    out_dir = args.output_dir or os.path.join(args.output_folder, 'plots', 'history')
    os.makedirs(out_dir, exist_ok=True)
    tag = f'z{z_snap:.2f}_{run_label}_{qmode}'

    plot_ssfr_vs_cosmic_time(
        gal_data, snap_times, snap, h,
        z_snap, run_label, qmode,
        os.path.join(out_dir, f'ssfr_history_{tag}.pdf'))

    plot_smh_vs_cosmic_time(
        gal_data, snap_times, snap, h,
        z_snap, run_label, qmode,
        os.path.join(out_dir, f'smh_history_{tag}.pdf'))

    # Stellar mass history — target galaxy (specified or most massive)
    if gal_data:
        if args.galaxy_id is not None:
            if args.galaxy_id not in gal_data:
                sys.exit(f'--galaxy-id {args.galaxy_id} not found in CSV/snapshot. '
                         f'Available IDs: {sorted(gal_data.keys())}')
            target_gid = args.galaxy_id
        else:
            target_gid = max(gal_data, key=lambda gid: gal_data[gid]['sm'])
        target_sm  = gal_data[target_gid]['sm'] * 1e10 / h
        print(f'Main branch: GalaxyIndex={target_gid}  M* = {target_sm:.3e} Msun')
        branch       = load_main_branch(filepaths, snap, target_gid, snap_times, h)
        merger_gids  = load_most_massive_merger(filepaths, snap, target_gid, snap_times, h)
        merger_branches = [
            load_main_branch(filepaths, snap, gid, snap_times, h)
            for gid in merger_gids
        ]
        print(f'  Main branch: {len(branch)} snapshots')
        plot_main_branch_history(
            branch, snap_times, snap, z_snap,
            run_label, qmode,
            os.path.join(out_dir, f'main_branch_history_{tag}.pdf'),
            galaxy_id=target_gid,
            merger_branches=merger_branches)

    plot_scaling_relations(
        csv_data, h, z_snap, run_label, qmode,
        os.path.join(out_dir, f'scaling_relations_{tag}.pdf'))

    plot_structural_kinematics(
        csv_data, h, z_snap, run_label, qmode,
        os.path.join(out_dir, f'structural_kinematics_{tag}.pdf'))

    # Environment analysis — loads all galaxies at the snapshot
    print('Loading all galaxies for environment analysis...')
    all_gals = load_snapshot_all_galaxies(
        filepaths, snap, params['particle_mass'], h)
    if all_gals is not None:
        box_size_mpc = params['box_size'] / h if params['box_size'] > 0 else None
        box_str = f'box={box_size_mpc:.1f} Mpc' if box_size_mpc else 'box size unknown'
        print(f'  Loaded {len(all_gals["gid"]):,} galaxies  ({box_str})')
        env_results = compute_environment(
            galaxy_ids, all_gals,
            box_size_mpc=box_size_mpc)
        print_environment_report(env_results, z_snap, run_label, qmode)
        plot_environment_summary(
            env_results, z_snap, run_label, qmode,
            os.path.join(out_dir, f'environment_summary_{tag}.pdf'))
        plot_group_maps(
            env_results, all_gals, z_snap, run_label, qmode,
            os.path.join(out_dir, f'group_maps_{tag}.pdf'),
            box_size_mpc=box_size_mpc)
        plot_group_maps(
            env_results, all_gals, z_snap, run_label, qmode,
            os.path.join(out_dir, f'group_maps_overdense_{tag}.pdf'),
            box_size_mpc=box_size_mpc, min_overdensity=15)
    else:
        print('  No galaxies loaded — skipping environment analysis.')

    print('Done.')


if __name__ == '__main__':
    main()

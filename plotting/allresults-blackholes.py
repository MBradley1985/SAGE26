#!/usr/bin/env python3
"""
allresults-blackholes.py
========================
One-stop black-hole diagnostics for SAGE26 HDF5 output. Combines, in a single
streamlined script, the diagnostics that previously lived in separate files:

  1. BH growth tracking per channel        (from bh_growth_median_halos_ID.py)
  2. Accretion rate function dN/dlog10(lambda) split by channel
                                            (from bh_eddington_analysis.py)
  3. BH seed-mass histogram by method       (from bh_seed_distribution.py /
                                             bh_init_mass_cont.py)
  4. Simplified timescale overlay           (from plot_timescales.py /
                                             timescale_hists.py)
  5. Black-hole - bulge mass relation       (from allresults-local.py)
  6. Black-hole mass function at fixed z     (from allresults-history.py)

Design notes
------------
* Every panel is wrapped so that, if the developer fields it needs are not yet
  present in the HDF5 file (e.g. the per-channel accretion arrays you have not
  merged into main yet), it prints "[skip] <plot>: missing <field>" and moves
  on instead of crashing.
* --snapshot lets you build the snapshot-dependent plots from any output snap.
* --bin-mode {none,stellar,redshift} turns the accretion rate function into a
  2x3 panel grid:
      stellar  -> one panel per stellar-mass bin (at --snapshot)
      redshift -> one panel per redshift (instantaneous accretion column)

Units / conventions (shared across SAGE26)
------------------------------------------
* Masses stored in 1e10 Msun/h, converted with 1e10 / Hubble_h.
* Radii in Mpc/h.  h read from Header/Simulation:hubble_h (fallback 0.73).
* HDF5 layout: Snap_<N> groups, per-galaxy struct-of-arrays.
* Per-channel 2D arrays dimensioned [ngal, SimMaxSnaps]; column index == snap.
* Accretion channels: 0 = Radio Mode, 1 = Merger, 2 = Disk Instability.
* Accretion rate function volume uses the comoving box in (Mpc/h)^3  (h^3 units).
  BHMF volume uses (box/h)^3 in physical Mpc^3, matching allresults-history.py.

Examples
--------
  python allresults-blackholes.py -i "output/millennium/model_*.hdf5"
  python allresults-blackholes.py -i "output/millennium/model_*.hdf5" -s 40
  python allresults-blackholes.py -i "..." --bin-mode stellar
  python allresults-blackholes.py -i "..." --bin-mode redshift --edd-limited
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, ScalarFormatter

import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# MATPLOTLIB STYLE  (matches the BH analysis scripts)
# ============================================================================
plt.rcParams['figure.figsize']    = (8.34, 6.25)
plt.rcParams['figure.dpi']        = 140
plt.rcParams['figure.autolayout'] = False
plt.rcParams['font.family']       = 'serif'
plt.rcParams['font.size']         = 14.0
plt.rcParams['axes.linewidth']    = 1.5
plt.rcParams['xtick.major.size']  = 7.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['xtick.minor.size']  = 5.5
plt.rcParams['xtick.minor.width'] = 0.5
plt.rcParams['xtick.direction']   = 'in'
plt.rcParams['xtick.top']         = True
plt.rcParams['xtick.labelsize']   = 13
plt.rcParams['ytick.major.size']  = 7.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['ytick.minor.size']  = 5.5
plt.rcParams['ytick.minor.width'] = 0.5
plt.rcParams['ytick.direction']   = 'in'
plt.rcParams['ytick.right']       = True
plt.rcParams['ytick.labelsize']   = 13
plt.rcParams['legend.frameon']    = False
plt.rcParams['legend.fontsize']   = 12

OutputFormat = '.png'

# ============================================================================
# CONSTANTS
# ============================================================================
HUBBLE_H_DEFAULT     = 0.73
MILLENNIUM_BOX_MPC_H = 62.5            # comoving box side (Mpc/h); override w/ header

MIN_STELLAR_MASS_LOG = 8.5
MIN_HALO_MASS_LOG    = 11.0
MIN_Z0_BH_MASS       = 1.0e4           # Msun, for accretion-rate selection

# Unit-time conversion for timescale dumps (code units -> Myr)
UNIT_LENGTH_CM     = 3.08568e24
UNIT_VELOCITY_CM_S = 1.0e5
SEC_PER_MYR        = 3.155e13
UNIT_TIME_IN_MYR   = (UNIT_LENGTH_CM / UNIT_VELOCITY_CM_S) / SEC_PER_MYR  # ~978029

# Accretion channel codes
ACC_RADIO, ACC_MERGER, ACC_INSTAB = 0, 1, 2

POSSIBLE_ID_FIELDS = ['GalaxyIndex', 'GalaxyID', 'ID', 'galaxy_id', 'id', 'GalID']

# Fallback Millennium snapshot -> redshift map (used only if the header lacks it)
MILLENNIUM_SNAP_TO_Z = {
    0: 127.0, 1: 65.74, 2: 40.0, 3: 26.66, 4: 19.36, 5: 14.78, 6: 11.66,
    7: 9.44, 8: 7.64, 9: 6.44, 10: 5.48, 11: 4.73, 12: 4.19, 13: 3.72,
    14: 3.33, 15: 3.0, 16: 2.73, 17: 2.48, 18: 2.27, 19: 2.07, 20: 1.90,
    21: 1.75, 22: 1.61, 23: 1.48, 24: 1.37, 25: 1.27, 26: 1.18, 27: 1.10,
    28: 1.02, 29: 0.96, 30: 0.90, 31: 0.85, 32: 0.81, 33: 0.77, 34: 0.73,
    35: 0.70, 36: 0.67, 37: 0.63, 38: 0.60, 39: 0.57, 40: 0.54, 41: 0.51,
    42: 0.49, 43: 0.46, 44: 0.43, 45: 0.41, 46: 0.39, 47: 0.37, 48: 0.36,
    49: 0.34, 50: 0.32, 51: 0.31, 52: 0.29, 53: 0.28, 54: 0.27, 55: 0.26,
    56: 0.25, 57: 0.24, 58: 0.23, 59: 0.21, 60: 0.20, 61: 0.18, 62: 0.0
}

# Panel defaults
DEFAULT_STELLAR_EDGES = [8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 12.0]   # -> 6 bins
DEFAULT_PANEL_Z       = [0.0, 0.5, 1.0, 2.0, 4.0, 6.0]            # -> 6 redshifts


# ============================================================================
# HDF5 HELPERS
# ============================================================================
def read_simulation_params(filepath):
    """Robust header read: h, box size, volume fraction, redshift table, snaps."""
    params = {
        'Hubble_h': HUBBLE_H_DEFAULT,
        'BoxSize': MILLENNIUM_BOX_MPC_H,
        'VolumeFraction': 1.0,
        'redshifts': None,
        'available_snapshots': [],
        'latest_snapshot': None,
    }
    try:
        with h5py.File(filepath, 'r') as f:
            if 'Header/Simulation' in f:
                sim = f['Header/Simulation'].attrs
                params['Hubble_h'] = float(sim.get('hubble_h',
                                            sim.get('HubbleParam', HUBBLE_H_DEFAULT)))
                params['BoxSize'] = float(sim.get('box_size', MILLENNIUM_BOX_MPC_H))
            elif 'Header' in f:
                hdr = f['Header'].attrs
                params['Hubble_h'] = float(hdr.get('hubble_h',
                                            hdr.get('HubbleParam', HUBBLE_H_DEFAULT)))
                params['BoxSize'] = float(hdr.get('box_size', MILLENNIUM_BOX_MPC_H))

            if 'Header/Runtime' in f and 'frac_volume_processed' in f['Header/Runtime'].attrs:
                params['VolumeFraction'] = float(
                    f['Header/Runtime'].attrs['frac_volume_processed'])

            if 'Header/snapshot_redshifts' in f:
                params['redshifts'] = np.array(f['Header/snapshot_redshifts'])

            snap_groups = [k for k in f.keys() if k.startswith('Snap_')]
            snaps = sorted(int(s.split('_')[1]) for s in snap_groups)
            params['available_snapshots'] = snaps
            params['latest_snapshot'] = max(snaps) if snaps else None
    except Exception as e:
        print(f"  ! could not fully read header of {filepath}: {e}")
    return params


def total_volume_fraction(file_list):
    tot = 0.0
    for f in file_list:
        tot += read_simulation_params(f)['VolumeFraction']
    return tot if tot > 0 else 1.0


def find_id_field(file_list, snap_num):
    for f in file_list:
        with h5py.File(f, 'r') as hf:
            key = f"Snap_{snap_num}"
            if key in hf:
                for c in POSSIBLE_ID_FIELDS:
                    if c in hf[key]:
                        return c
    return None


def get_redshift_from_snapshot(snap_num, redshifts=None):
    """Prefer the header redshift table; fall back to the Millennium map."""
    if redshifts is not None and 0 <= snap_num < len(redshifts):
        return float(redshifts[snap_num])
    if snap_num in MILLENNIUM_SNAP_TO_Z:
        return MILLENNIUM_SNAP_TO_Z[snap_num]
    return 0.0


def snapshot_for_redshift(target_z, redshifts, available):
    """Closest available snapshot to a target redshift."""
    if not available:
        return None
    zs = np.array([get_redshift_from_snapshot(s, redshifts) for s in available])
    return int(available[int(np.argmin(np.abs(zs - target_z)))])


def field_present(file_list, snap_num, field):
    for f in file_list:
        with h5py.File(f, 'r') as hf:
            key = f"Snap_{snap_num}"
            if key in hf and field in hf[key]:
                return True
    return False


def read_hdf(file_list, snap_num, field, conv=1.0):
    """Plain read + concatenate of a 1D per-galaxy field across files."""
    out = []
    key = f"Snap_{snap_num}"
    for f in file_list:
        with h5py.File(f, 'r') as hf:
            if key in hf and field in hf[key]:
                arr = np.array(hf[key][field])
                if arr.size:
                    out.append(arr)
    if not out:
        return np.array([])
    return np.concatenate(out) * conv


def read_hdf_cumulative(file_list, snap_num, field, ref_field='BlackHoleMass'):
    """
    Read a per-channel growth field and return the cumulative accreted mass up to
    and including `snap_num`.  Handles clean 2D [ngal, maxsnaps] arrays as well as
    the flattened 1D case.  (Adapted from bh_growth_median_halos_ID.py.)
    """
    data = []
    key = f"Snap_{snap_num}"
    for f in file_list:
        with h5py.File(f, 'r') as hf:
            if key not in hf or field not in hf[key]:
                continue
            val = np.array(hf[key][field])
            ref_len = len(hf[key][ref_field]) if ref_field in hf[key] else len(val)
            if val.ndim == 2:
                val = np.nansum(val[:, :int(snap_num) + 1], axis=1)
            elif val.ndim == 1 and ref_len > 0 and len(val) > ref_len:
                max_snaps = len(val) // ref_len
                val = np.nansum(val.reshape(ref_len, max_snaps)[:, :int(snap_num) + 1], axis=1)
            # else: 1D length == ngal -> use as-is (may be the C 1D-output case)
            data.append(val)
    return np.concatenate(data) if data else np.array([])


def _reshape_history(raw, ngal):
    """Coerce a raw array into (ngal, maxsnaps)."""
    if raw.ndim == 2:
        return raw
    if len(raw) == ngal:
        return raw.reshape(-1, 1)
    maxsnaps = len(raw) // ngal
    return raw.reshape(ngal, maxsnaps)


def read_bh_histories(file_list, snap_num, hubble_h, fields):
    """
    Read the requested per-galaxy history fields at `snap_num` and return a dict
    of 2D arrays (ngal, maxsnaps) plus 1D BlackHoleMass / StellarMass / Mvir for
    masking.  Returns None if no galaxies are found.  Missing 2D fields come back
    as None so callers can decide whether to skip.
    """
    conv = 1.0e10 / hubble_h
    key = f"Snap_{snap_num}"
    blocks = {fld: [] for fld in fields}
    bh, sm, mv = [], [], []

    for f in file_list:
        with h5py.File(f, 'r') as hf:
            if key not in hf:
                continue
            grp = hf[key]
            if 'BlackHoleMass' not in grp:
                continue
            ngal = len(grp['BlackHoleMass'])
            bh.append(np.array(grp['BlackHoleMass']) * conv)
            sm.append(np.array(grp['StellarMass']) * conv if 'StellarMass' in grp
                      else np.zeros(ngal))
            mv.append(np.array(grp['Mvir']) * conv if 'Mvir' in grp
                      else np.zeros(ngal))
            for fld in fields:
                if fld in grp:
                    c = 1.0 if fld == 'BHAccretionType' else conv
                    blocks[fld].append(_reshape_history(np.array(grp[fld]), ngal) * c)
                else:
                    blocks[fld].append(None)

    if not bh:
        return None

    out = {'BlackHoleMass': np.concatenate(bh),
           'StellarMass': np.concatenate(sm),
           'Mvir': np.concatenate(mv)}
    for fld in fields:
        parts = blocks[fld]
        if any(p is None for p in parts):
            out[fld] = None
        else:
            # pad to common width before concatenating
            w = max(p.shape[1] for p in parts)
            padded = []
            for p in parts:
                if p.shape[1] < w:
                    pad = np.zeros((p.shape[0], w - p.shape[1]))
                    if fld == 'BHAccretionType':
                        pad -= 1.0
                    p = np.hstack([p, pad])
                padded.append(p)
            out[fld] = np.concatenate(padded, axis=0)
    return out


def selection_mask(bh, sm, mv, no_cuts):
    if no_cuts:
        return bh > 0
    return (bh > MIN_Z0_BH_MASS) & (sm > 10 ** MIN_STELLAR_MASS_LOG) \
        & (mv > 10 ** MIN_HALO_MASS_LOG)


# ============================================================================
# 1. BH GROWTH TRACKING PER CHANNEL
# ============================================================================
def plot_bh_growth_channels(file_list, snap_num, hubble_h, redshifts,
                            available, output_dir):
    name = "BH growth channels"
    channel_fields = {
        'md': 'MergerDrivenBHaccretionMass',
        'id': 'InstabilityDrivenBHaccretionMass',
        'rm': 'RadioModeBHaccretionMass',
        'bm': 'BHMergerMass',
    }
    have = {k: field_present(file_list, snap_num, v) for k, v in channel_fields.items()}
    if not any(have.values()):
        print(f"[skip] {name}: none of "
              f"{', '.join(channel_fields.values())} present.")
        return
    missing = [channel_fields[k] for k, v in have.items() if not v]
    if missing:
        print(f"  ({name}: missing {', '.join(missing)} - those channels blank.)")

    id_field = find_id_field(file_list, snap_num)
    if id_field is None:
        print(f"[skip] {name}: no galaxy-ID field for cross-snapshot tracking.")
        return

    # z=0 (selected snapshot) baseline + halo-mass bin membership by fixed ID
    bh0 = read_hdf(file_list, snap_num, 'BlackHoleMass', 1.0e10 / hubble_h)
    sm0 = read_hdf(file_list, snap_num, 'StellarMass',  1.0e10 / hubble_h)
    mv0 = read_hdf(file_list, snap_num, 'Mvir',         1.0e10 / hubble_h)
    if len(bh0) == 0:
        print(f"[skip] {name}: no galaxies at snapshot {snap_num}.")
        return

    base = (bh0 > 0) & (sm0 > 10 ** MIN_STELLAR_MASS_LOG) & (mv0 > 10 ** MIN_HALO_MASS_LOG)
    log_mv0 = np.log10(mv0 + 1e-10)

    halo_bins = [(11.5, 12.5), (12.5, 13.5), (13.5, 14.5), (14.5, 15.5)]
    bin_labels = [r"$\log_{10}(M_{h,0}) \sim 12\,M_\odot$",
                  r"$\log_{10}(M_{h,0}) \sim 13\,M_\odot$",
                  r"$\log_{10}(M_{h,0}) \sim 14\,M_\odot$",
                  r"$\log_{10}(M_{h,0}) \sim 15\,M_\odot$"]

    gid0 = read_hdf(file_list, snap_num, id_field)
    tracked_ids = []
    for (mlo, mhi) in halo_bins:
        m = base & (log_mv0 >= mlo) & (log_mv0 < mhi)
        tracked_ids.append(gid0[m] if len(gid0) == len(base) else np.array([]))

    def lookup(target, snap_ids):
        if len(target) == 0 or snap_ids is None or len(snap_ids) == 0:
            return np.array([], dtype=int)
        order = np.argsort(snap_ids)
        s = snap_ids[order]
        pos = np.clip(np.searchsorted(s, target), 0, len(s) - 1)
        ok = s[pos] == target
        idx = np.full(len(target), -1, dtype=int)
        idx[ok] = order[pos[ok]]
        return idx[idx >= 0]

    results = [[] for _ in halo_bins]
    for sn in available:
        z = get_redshift_from_snapshot(sn, redshifts)
        if z is None or z > 7.5:
            continue
        bh = read_hdf_cumulative(file_list, sn, 'BlackHoleMass') * 1.0e10 / hubble_h
        if len(bh) == 0:
            continue

        def chan(key):
            if not have[key]:
                return np.zeros(len(bh))
            a = read_hdf_cumulative(file_list, sn, channel_fields[key])
            return a * 1.0e10 / hubble_h if len(a) else np.zeros(len(bh))

        ch = {k: chan(k) for k in channel_fields}
        gid_sn = read_hdf(file_list, sn, id_field)

        for i, _ in enumerate(halo_bins):
            idx = lookup(tracked_ids[i], gid_sn if len(gid_sn) else None)
            if len(idx) == 0:
                continue

            def pct(x):
                v = x[idx]
                v = v[v > 1e-6]
                return np.percentile(v, [16, 50, 84]) if len(v) > 2 else [np.nan] * 3

            results[i].append({'z': z, **{k: pct(ch[k]) for k in channel_fields}})

    # ---- plot 1x4 ----
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
    channels = [('Merger-driven', 'md', '#2196F3'),
                ('Instability-driven', 'id', '#FF9800'),
                ('Radio mode', 'rm', '#9C27B0'),
                ('BH-BH mergers', 'bm', '#4CAF50')]

    for i, ax in enumerate(axes):
        ax.set_title(bin_labels[i])
        ax.set_xlabel(r'Redshift ($z$)')
        if i == 0:
            ax.set_ylabel(r'$\log_{10}(M_{\rm BH}\,[M_\odot])$')
        res = sorted(results[i], key=lambda r: r['z'], reverse=True)
        if not res:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
            continue
        zarr = np.array([r['z'] for r in res])
        for label, key, color in channels:
            if not have[key]:
                continue
            p16 = np.array([r[key][0] for r in res])
            p50 = np.array([r[key][1] for r in res])
            p84 = np.array([r[key][2] for r in res])
            ok = ~np.isnan(p50) & (p50 > 0)
            if np.sum(ok) > 1:
                ax.plot(zarr[ok], np.log10(p50[ok]), color=color, lw=1.8, label=label)
                ax.fill_between(zarr[ok], np.log10(p16[ok]), np.log10(p84[ok]),
                                color=color, alpha=0.15)
        ax.set_xlim(max(zarr.min(), 0), 7.0)
        ax.set_ylim(-2.5, 10)
        ax.grid(True, alpha=0.3)
    handles = [plt.Line2D([0], [0], color=c, lw=2) for _, _, c in channels]
    axes[-1].legend(handles, [l for l, _, _ in channels], fontsize=10)

    plt.tight_layout()
    out = os.path.join(output_dir, f"bh_growth_channels{OutputFormat}")
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"[ok]   {name} -> {out}")


# ============================================================================
# 2. ACCRETION RATE FUNCTION  (single or 2x3 panel)
# ============================================================================
def _draw_rate_function(ax, accr, edd, acc_type, volume_h3, edd_limited,
                        n_bins=40, show_legend=True, show_xlabel=True,
                        show_ylabel=True):
    """Draw dN/dlog10(lambda) split by channel onto a single axis."""
    accr = np.ravel(accr); edd = np.ravel(edd); acc_type = np.ravel(acc_type)
    valid = (accr > 0) & (edd > 0) & np.isfinite(accr) & np.isfinite(edd)
    accr, edd, acc_type = accr[valid], edd[valid], acc_type[valid]
    if len(accr) == 0:
        ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                ha='center', va='center', color='grey')
        return 0

    lam = accr / edd
    if edd_limited:
        lam = np.minimum(lam, 1.0)
    log_lam = np.log10(lam)

    lo = np.floor(log_lam.min() * 2) / 2
    if edd_limited:
        bins = np.linspace(lo, 0.0, n_bins + 1)
        centre_mask_cap = 0.0
    else:
        hi = np.ceil(log_lam.max() * 2) / 2
        bins = np.linspace(lo, hi, n_bins + 1)
        centre_mask_cap = None
    bw = bins[1] - bins[0]
    centres = 0.5 * (bins[:-1] + bins[1:])
    pmask = centres <= centre_mask_cap if centre_mask_cap is not None \
        else np.ones(len(centres), bool)

    cats = [
        (log_lam,                              'Total',            'k',       2.4, 0.10),
        (log_lam[acc_type == ACC_MERGER],      'Merger',           '#1976D2', 1.9, 0.15),
        (log_lam[acc_type == ACC_RADIO],       'Radio Mode',       '#D32F2F', 1.9, 0.15),
        (log_lam[acc_type == ACC_INSTAB],      'Disk Instability', '#388E3C', 1.9, 0.15),
    ]

    gmin = np.inf
    store = []
    for data, _, _, _, _ in cats:
        counts, _ = np.histogram(data, bins=bins)
        y = counts / (bw * volume_h3) if volume_h3 else counts / bw
        pos = y > 0
        logy = np.full_like(y, np.nan, dtype=float)
        logy[pos] = np.log10(y[pos])
        if np.any(pos):
            gmin = min(gmin, np.nanmin(logy[pos]))
        store.append((logy, pos))
    floor = gmin - 1.5 if np.isfinite(gmin) else -10.0

    allv = []
    for (data, label, color, lw, alpha), (logy, pos) in zip(cats, store):
        ax.step(centres[pmask], logy[pmask], where='mid', lw=lw, color=color,
                label=label)
        ax.fill_between(centres[pmask], logy[pmask], floor, step='mid',
                        alpha=alpha, color=color)
        allv.extend(logy[pos & pmask])

    ax.axvline(0.0, color='k', ls='--', lw=1.3, alpha=0.7)
    #ax.set_xlim(lo, 0.2 if edd_limited else bins[-1])
    ax.set_xlim(-10, 5)
    #ax.set_ylim(-7,0)
    if allv:
        ax.set_ylim(floor + 1.0, np.nanmax(allv) + 0.8)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    if show_xlabel:
        ax.set_xlabel(r'$\log_{10}(\dot{M}_{\rm BH,max}/\dot{M}_{\rm Edd})$',
                      fontsize=14)
    if show_ylabel:
        yl = (r'$\log_{10}(\mathrm{d}N/\mathrm{d}\log_{10}\lambda\,/\,'
              r'\mathrm{Mpc}^{-3}h^{3})$') if volume_h3 else \
             r'$\log_{10}(\mathrm{d}N/\mathrm{d}\log_{10}\lambda)$'
        ax.set_ylabel(yl, fontsize=13)
    if show_legend:
        ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.25, ls=':', lw=0.6)
    return len(log_lam)


def plot_accretion_rate_function(file_list, snap_num, hubble_h, redshifts,
                                 available, output_dir, bin_mode, stellar_edges,
                                 panel_z, edd_limited, volume_h3, no_cuts):
    name = "accretion rate function"
    need = ['BHMaxaccretionRate', 'BHEddingtonRateLimit', 'BHAccretionType']
    for fld in need:
        if not field_present(file_list, snap_num, fld):
            print(f"[skip] {name}: missing {fld}.")
            return

    data = read_bh_histories(file_list, snap_num, hubble_h, need)
    if data is None:
        print(f"[skip] {name}: no galaxies at snapshot {snap_num}.")
        return
    accr, edd, typ = (data['BHMaxaccretionRate'],
                      data['BHEddingtonRateLimit'], data['BHAccretionType'])
    if accr is None or edd is None or typ is None:
        print(f"[skip] {name}: required arrays absent after read.")
        return

    mask = selection_mask(data['BlackHoleMass'], data['StellarMass'],
                          data['Mvir'], no_cuts)

    # ---- single panel ----
    if bin_mode == 'none':
        fig, ax = plt.subplots(figsize=(8.34, 6.25))
        n = _draw_rate_function(ax, accr[mask], edd[mask], typ[mask],
                                volume_h3, edd_limited)
        z = get_redshift_from_snapshot(snap_num, redshifts)
        ax.set_title(f"snap {snap_num}  (z = {z:.2f})", fontsize=13)
        plt.tight_layout()
        out = os.path.join(output_dir, f"bh_accretion_rate_function{OutputFormat}")
        fig.savefig(out, dpi=140, bbox_inches='tight')
        plt.close(fig)
        print(f"[ok]   {name} ({n:,} events) -> {out}")
        return

    # ---- 2x3 stellar-mass panels ----
    if bin_mode == 'stellar':
        edges = stellar_edges
        pairs = list(zip(edges[:-1], edges[1:]))[:6]
        log_sm = np.log10(np.where(data['StellarMass'] > 0,
                                   data['StellarMass'], np.nan))
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.ravel()
        bh_cut = (data['BlackHoleMass'] > 0) if no_cuts else \
                 (data['BlackHoleMass'] > MIN_Z0_BH_MASS) & \
                 (data['Mvir'] > 10 ** MIN_HALO_MASS_LOG)
        for k, (lo, hi) in enumerate(pairs):
            sel = bh_cut & (log_sm >= lo) & (log_sm < hi)
            _draw_rate_function(axes[k], accr[sel], edd[sel], typ[sel],
                                volume_h3, edd_limited,
                                show_legend=(k == 0),
                                show_xlabel=(k >= 3),
                                show_ylabel=(k % 3 == 0))
            axes[k].set_title(rf"${lo:.1f}\leq\log M_\star<{hi:.1f}$  "
                              rf"($N_{{gal}}={int(np.sum(sel))}$)", fontsize=12)
        for j in range(len(pairs), 6):
            axes[j].axis('off')
        z = get_redshift_from_snapshot(snap_num, redshifts)
        fig.suptitle(f"Accretion rate function by stellar mass "
                     f"(snap {snap_num}, z = {z:.2f})", fontsize=15)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        out = os.path.join(output_dir,
                           f"bh_accretion_rate_function_stellar_panels{OutputFormat}")
        fig.savefig(out, dpi=140, bbox_inches='tight')
        plt.close(fig)
        print(f"[ok]   {name} (stellar 2x3) -> {out}")
        return

    # ---- 2x3 redshift panels (instantaneous accretion column per snapshot) ----
    if bin_mode == 'redshift':
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.ravel()
        for k, tz in enumerate(panel_z[:6]):
            sn = snapshot_for_redshift(tz, redshifts, available)
            if sn is None:
                axes[k].text(0.5, 0.5, 'no snap', transform=axes[k].transAxes,
                             ha='center'); axes[k].set_title(f"z~{tz:.1f}")
                continue
            d = read_bh_histories(file_list, sn, hubble_h, need)
            if d is None or d['BHMaxaccretionRate'] is None:
                axes[k].text(0.5, 0.5, 'no data', transform=axes[k].transAxes,
                             ha='center')
                axes[k].set_title(f"snap {sn}")
                continue
            m = selection_mask(d['BlackHoleMass'], d['StellarMass'],
                               d['Mvir'], no_cuts)
            a, e, t = d['BHMaxaccretionRate'], d['BHEddingtonRateLimit'], d['BHAccretionType']
            col = min(sn, a.shape[1] - 1)
            _draw_rate_function(axes[k], a[m][:, col], e[m][:, col], t[m][:, col],
                                volume_h3, edd_limited,
                                show_legend=(k == 0),
                                show_xlabel=(k >= 3),
                                show_ylabel=(k % 3 == 0))
            zz = get_redshift_from_snapshot(sn, redshifts)
            axes[k].set_title(f"snap {sn}  (z = {zz:.2f})", fontsize=12)
        fig.suptitle("Accretion rate function vs redshift (instantaneous)",
                     fontsize=15)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        out = os.path.join(output_dir,
                           f"bh_accretion_rate_function_redshift_panels{OutputFormat}")
        fig.savefig(out, dpi=140, bbox_inches='tight')
        plt.close(fig)
        print(f"[ok]   {name} (redshift 2x3) -> {out}")
        return


# ============================================================================
# 3. SEED-MASS HISTOGRAM BY METHOD
# ============================================================================
def classify_seeding_method(seed_mass, heavy_threshold=1.0e4):
    """0 = other/intermediate, 1 = light (30-100), 2 = heavy (>= 1e4)."""
    c = np.zeros(len(seed_mass), dtype=int)
    c[(seed_mass >= 30) & (seed_mass <= 100)] = 1
    c[seed_mass >= heavy_threshold] = 2
    return c


def plot_seed_histogram(file_list, snap_num, hubble_h, output_dir):
    name = "seed-mass histogram"
    if not field_present(file_list, snap_num, 'BHSeedMass'):
        print(f"[skip] {name}: missing BHSeedMass.")
        return
    seeds = read_hdf(file_list, snap_num, 'BHSeedMass', 1.0e10 / hubble_h)
    seeds = seeds[seeds > 0]
    if len(seeds) == 0:
        print(f"[skip] {name}: no positive seed masses.")
        return

    cls = classify_seeding_method(seeds)
    light, heavy, other = seeds[cls == 1], seeds[cls == 2], seeds[cls == 0]
    bins = np.logspace(np.log10(seeds.min()), np.log10(seeds.max()), 50)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.minorticks_on()
    ax.hist(light, bins=bins, color='#2196F3', alpha=0.7, edgecolor='black',
            linewidth=0.5, label=fr'Light seeds (30-100 $M_\odot$, n={len(light)})')
    ax.hist(heavy, bins=bins, color='#FF9800', alpha=0.7, edgecolor='black',
            linewidth=0.5, label=fr'Heavy seeds ($\geq 10^4\,M_\odot$, n={len(heavy)})')
    if len(other):
        ax.hist(other, bins=bins, color='#9E9E9E', alpha=0.5, histtype='step',
                linewidth=1.5, label=f'Other (n={len(other)})')
    ax.set_xscale('log')
    ax.set_xlabel(r'Seed Mass ($M_\odot$)', fontsize=14)
    ax.set_ylabel('Count', fontsize=13)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3, which='both')

    z = get_redshift_from_snapshot(snap_num)
    ax.set_title(f"BH seed mass by method (snap {snap_num})", fontsize=13)
    plt.tight_layout()
    out = os.path.join(output_dir, f"bh_seed_by_method{OutputFormat}")
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"[ok]   {name} ({len(seeds):,} seeds: "
          f"{len(light)} light / {len(heavy)} heavy / {len(other)} other) -> {out}")


# ============================================================================
# 4. SIMPLIFIED TIMESCALE OVERLAY
# ============================================================================
TS_KEYS    = ["dt", "tdyn_B_end", "tdyn_disk", "hubbletime", "halotime"]
TS_COLOURS = {"dt": "#4477AA", "tdyn_B_end": "#EE6677", "tdyn_disk": "#AA3377",
              "hubbletime": "#228833", "halotime": "#CCBB44"}
TS_LABELS  = {"dt": r"$\Delta t$ (snapshot step)",
              "tdyn_B_end": r"$t_{\rm dyn}$ (bulge)",
              "tdyn_disk": r"$t_{\rm dyn}$ (disk)",
              "hubbletime": r"$0.1\,t_H(z)=1/H(z)$",
              "halotime": r"$t_{\rm halo}=R_{\rm vir}/V_{\rm vir}$"}


def load_dt_from_hdf(file_list, snap_num):
    dt = read_hdf(file_list, snap_num, 'dt')

    if len(dt) == 0:
        return None

    dt = dt[np.isfinite(dt) & (dt > 0)]

    # convert code units if necessary
    dt *= UNIT_TIME_IN_MYR

    return dt if len(dt) else None


def plot_timescales_overlay(file_list, snap_num,
                            ts_dir, units, output_dir):
    name = "timescale overlay"

    datasets = {}

    dt = read_hdf(file_list, snap_num, 'tacc')
    dt = dt[np.isfinite(dt) & (dt > 0)]

    # convert code units → Myr
    dt = dt * UNIT_TIME_IN_MYR

    present = {'dt': dt}

    if dt is None or len(dt) == 0:
        print(f"[skip] {name}: no valid dt data in HDF5.")
        return

    allv = dt
    bins = np.logspace(np.log10(allv.min()), np.log10(allv.max()), 80)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    c = TS_COLOURS.get('dt', 'black')
    label = TS_LABELS.get('dt', 'dt')

    ax.hist(dt, bins=bins, color=c, alpha=0.45, histtype='stepfilled',
            label=label + rf"  (med {np.median(dt):.2g} Myr)")
    ax.hist(dt, bins=bins, color=c, histtype='step', linewidth=1.5)
    ax.axvline(np.median(dt), color=c, ls=':', lw=1.4, alpha=0.9)

    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.get_major_formatter().set_scientific(False)
    ax.set_xlabel("Time (Myr)")
    ax.set_ylabel("Count")
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    out = os.path.join(output_dir, f"bh_timescales_overlay{OutputFormat}")
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)

    print(f"[ok]   {name} -> {out}")


# ============================================================================
# 5. BLACK HOLE - BULGE MASS RELATION   (matches allresults-local.py styling)
# ============================================================================
def plot_bh_bulge_relation(file_list, snap_num, hubble_h, output_dir, dilute=7500):
    from random import sample, seed as rseed
    name = "BH-bulge relation"
    for fld in ('BulgeMass', 'BlackHoleMass'):
        if not field_present(file_list, snap_num, fld):
            print(f"[skip] {name}: missing {fld}.")
            return

    conv = 1.0e10 / hubble_h
    BulgeMass = read_hdf(file_list, snap_num, 'BulgeMass', conv)
    BlackHoleMass = read_hdf(file_list, snap_num, 'BlackHoleMass', conv)
    if len(BulgeMass) == 0:
        print(f"[skip] {name}: no galaxies at snapshot {snap_num}.")
        return

    rseed(2222)
    fig, ax = plt.subplots(figsize=(8.34, 6.25))
    w = np.where((BulgeMass > 1.0e8) & (BlackHoleMass > 1.0e6))[0]
    if len(w) > dilute:
        w = sample(list(w), dilute)
    if len(w) == 0:
        print(f"[skip] {name}: no galaxies pass the bulge/BH mass floor.")
        plt.close(fig)
        return

    ax.scatter(np.log10(BulgeMass[w]), np.log10(BlackHoleMass[w]), marker='x',
               s=1, c='k', alpha=0.9, label='Model galaxies', zorder=10)

    # Haring & Rix 2004
    ww = 10. ** np.arange(20)
    BHdata = 10. ** (8.2 + 1.12 * np.log10(ww / 1.0e11))
    ax.plot(np.log10(ww), np.log10(BHdata), 'b-', label=r"Haring \& Rix 2004")

    # Scott et al. 2013 (S13) compilation
    hf = (0.7 / hubble_h) ** 2
    M_BH_obs = hf * 1e8 * np.array([39, 11, 0.45, 25, 24, 0.044, 1.4, 0.73, 9.0, 58, 0.10, 8.3, 0.39, 0.42, 0.084, 0.66, 0.73, 15, 4.7, 0.083, 0.14, 0.15, 0.4, 0.12, 1.7, 0.024, 8.8, 0.14, 2.0, 0.073, 0.77, 4.0, 0.17, 0.34, 2.4, 0.058, 3.1, 1.3, 2.0, 97, 8.1, 1.8, 0.65, 0.39, 5.0, 3.3, 4.5, 0.075, 0.68, 1.2, 0.13, 4.7, 0.59, 6.4, 0.79, 3.9, 47, 1.8, 0.06, 0.016, 210, 0.014, 7.4, 1.6, 6.8, 2.6, 11, 37, 5.9, 0.31, 0.10, 3.7, 0.55, 13, 0.11])
    M_BH_hi = hf * 1e8 * np.array([4, 2, 0.17, 7, 10, 0.044, 0.9, 0.0, 0.9, 3.5, 0.10, 2.7, 0.26, 0.04, 0.003, 0.03, 0.69, 2, 0.6, 0.004, 0.02, 0.09, 0.04, 0.005, 0.2, 0.024, 10, 0.1, 0.5, 0.015, 0.04, 1.0, 0.01, 0.02, 0.3, 0.008, 1.4, 0.5, 1.1, 30, 2.0, 0.6, 0.07, 0.01, 1.0, 0.9, 2.3, 0.002, 0.13, 0.4, 0.08, 0.5, 0.03, 0.4, 0.38, 0.4, 10, 0.2, 0.014, 0.004, 160, 0.014, 4.7, 0.3, 0.7, 0.4, 1, 18, 2.0, 0.004, 0.001, 2.6, 0.26, 5, 0.005])
    M_BH_lo = hf * 1e8 * np.array([5, 2, 0.10, 7, 10, 0.022, 0.3, 0.0, 0.8, 3.5, 0.05, 1.3, 0.09, 0.04, 0.003, 0.03, 0.35, 2, 0.6, 0.004, 0.13, 0.1, 0.05, 0.005, 0.2, 0.012, 2.7, 0.06, 0.5, 0.015, 0.06, 1.0, 0.02, 0.02, 0.3, 0.008, 0.6, 0.5, 0.6, 26, 1.9, 0.3, 0.07, 0.01, 1.0, 2.5, 1.5, 0.002, 0.13, 0.9, 0.08, 0.5, 0.09, 0.4, 0.33, 0.4, 10, 0.1, 0.014, 0.004, 160, 0.007, 3.0, 0.4, 0.7, 1.5, 1, 11, 2.0, 0.004, 0.001, 1.5, 0.19, 4, 0.005])
    M_sph_obs = hf * 1e10 * np.array([69, 37, 1.4, 55, 27, 2.4, 0.46, 1.0, 19, 23, 0.61, 4.6, 11, 1.9, 4.5, 1.4, 0.66, 4.7, 26, 2.0, 0.39, 0.35, 0.30, 3.5, 6.7, 0.88, 1.9, 0.93, 1.24, 0.86, 2.0, 5.4, 1.2, 4.9, 2.0, 0.66, 5.1, 2.6, 3.2, 100, 1.4, 0.88, 1.3, 0.56, 29, 6.1, 0.65, 3.3, 2.0, 6.9, 1.4, 7.7, 0.9, 3.9, 1.8, 8.4, 27, 6.0, 0.43, 1.0, 122, 0.30, 29, 11, 20, 2.8, 24, 78, 96, 3.6, 2.6, 55, 1.4, 64, 1.2])
    M_sph_hi = hf * 1e10 * np.array([59, 32, 2.0, 80, 23, 3.5, 0.68, 1.5, 16, 19, 0.89, 6.6, 9, 2.7, 6.6, 2.1, 0.91, 6.9, 22, 2.9, 0.57, 0.52, 0.45, 5.1, 5.7, 1.28, 2.7, 1.37, 1.8, 1.26, 1.7, 4.7, 1.7, 7.1, 2.9, 0.97, 7.4, 3.8, 2.7, 86, 2.1, 1.30, 1.9, 0.82, 25, 5.2, 0.96, 4.9, 3.0, 5.9, 1.2, 6.6, 1.3, 5.7, 2.7, 7.2, 23, 5.2, 0.64, 1.5, 105, 0.45, 25, 10, 17, 2.4, 20, 67, 83, 5.2, 3.8, 48, 2.0, 55, 1.8])
    M_sph_lo = hf * 1e10 * np.array([32, 17, 0.8, 33, 12, 1.4, 0.28, 0.6, 9, 10, 0.39, 2.7, 5, 1.1, 2.7, 0.8, 0.40, 2.8, 12, 1.2, 0.23, 0.21, 0.18, 2.1, 3.1, 0.52, 1.1, 0.56, 0.7, 0.51, 0.9, 2.5, 0.7, 2.9, 1.2, 0.40, 3.0, 1.5, 1.5, 46, 0.9, 0.53, 0.8, 0.34, 13, 2.8, 0.39, 2.0, 1.2, 3.2, 0.6, 3.6, 0.5, 2.3, 1.1, 3.9, 12, 2.8, 0.26, 0.6, 57, 0.18, 13, 5, 9, 1.3, 11, 36, 44, 2.1, 1.5, 26, 0.8, 30, 0.7])
    core = np.array([1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0])
    yerr2 = np.log10((M_BH_obs + M_BH_hi) / M_BH_obs)
    yerr1 = -np.log10((M_BH_obs - M_BH_lo) / M_BH_obs)
    xerr2 = np.log10((M_sph_obs + M_sph_hi) / M_sph_obs)
    xerr1 = -np.log10((M_sph_obs - M_sph_lo) / M_sph_obs)
    ax.errorbar(np.log10(M_sph_obs[core == 0]), np.log10(M_BH_obs[core == 0]),
                yerr=[yerr1[core == 0], yerr2[core == 0]],
                xerr=[xerr1[core == 0], xerr2[core == 0]], color='orange',
                alpha=0.6, label=r'S13 core', ls='none', lw=2, ms=0)
    ax.errorbar(np.log10(M_sph_obs[core == 1]), np.log10(M_BH_obs[core == 1]),
                yerr=[yerr1[core == 1], yerr2[core == 1]],
                xerr=[xerr1[core == 1], xerr2[core == 1]], color='c',
                alpha=0.6, label=r'S13 Sersic', ls='none', lw=2, ms=0)

    ax.set_ylabel(r'$\log\ M_{\mathrm{BH}}\ (M_{\odot})$')
    ax.set_xlabel(r'$\log\ M_{\mathrm{bulge}}\ (M_{\odot})$')
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.25))
    ax.axis([8.0, 12.0, 6.0, 10.0])
    leg = ax.legend(loc='upper left')
    leg.draw_frame(False)
    for t in leg.get_texts():
        t.set_fontsize('medium')

    plt.tight_layout()
    out = os.path.join(output_dir, f"BlackHoleBulgeRelationship{OutputFormat}")
    fig.savefig(out)
    plt.close(fig)
    print(f"[ok]   {name} -> {out}")


# ============================================================================
# 6. BLACK HOLE MASS FUNCTION   (matches allresults-history.py styling)
# ============================================================================
def plot_bh_mass_function(file_list, hubble_h, volume_phys, redshifts,
                          available, output_dir, data_dir):
    name = "black hole mass function"
    if not any(field_present(file_list, s, 'BlackHoleMass') for s in available):
        print(f"[skip] {name}: BlackHoleMass not present in any snapshot.")
        return
    if redshifts is None:
        redshifts = np.array([get_redshift_from_snapshot(s) for s in available])
    if volume_phys is None or volume_phys <= 0:
        print(f"[skip] {name}: no simulation volume available.")
        return

    bhmf_redshifts = [0.1, 1.0, 2.0, 4.0, 6.0, 8.0]
    snaps, actual_z = [], []
    for tz in bhmf_redshifts:
        s = snapshot_for_redshift(tz, redshifts, available)
        snaps.append(s)
        actual_z.append(get_redshift_from_snapshot(s, redshifts) if s is not None else tz)

    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(bhmf_redshifts)))
    mass_bins = np.arange(6.0, 11.5, 0.1)
    centres = mass_bins[:-1] + 0.05
    bw = mass_bins[1] - mass_bins[0]

    fig, ax = plt.subplots(figsize=(10, 8))
    plotted = 0
    for i, (sn, az) in enumerate(zip(snaps, actual_z)):
        if sn is None:
            continue
        bh = read_hdf(file_list, sn, 'BlackHoleMass', 1.0e10 / hubble_h)
        w = np.where(bh > 0.0)[0]
        if len(w) == 0:
            continue
        counts, _ = np.histogram(np.log10(bh[w]), bins=mass_bins)
        phi = counts / (volume_phys * bw)
        ok = phi > 0
        if np.any(ok):
            ax.plot(centres[ok], phi[ok], color=colors[i], lw=2,
                    label=f'z = {az:.1f} (SAGE)')
            plotted += 1

    if plotted == 0:
        print(f"[skip] {name}: no snapshots yielded black holes.")
        plt.close(fig)
        return

    # optional observational overlays (skip silently per-file if absent)
    obs_files = {0.1: 'fig4_bhmf_z0.1.txt', 1.0: 'fig4_bhmf_z1.0.txt',
                 2.0: 'fig4_bhmf_z2.0.txt', 4.0: 'fig4_bhmf_z4.0.txt',
                 6.0: 'fig4_bhmf_z6.0.txt', 8.0: 'fig4_bhmf_z8.0.txt'}
    if data_dir:
        for i, tz in enumerate(bhmf_redshifts):
            fp = os.path.join(data_dir, obs_files[tz])
            if not os.path.isfile(fp):
                continue
            try:
                od = np.loadtxt(fp)
                ax.plot(od[:, 0], od[:, 1], color=colors[i], lw=2, ls='--',
                        alpha=0.8, label=f'z = {tz:.1f} (Obs)')
                if od.shape[1] >= 4:
                    ax.fill_between(od[:, 0], od[:, 2], od[:, 3],
                                    color=colors[i], alpha=0.2)
            except Exception as e:
                print(f"  (could not load {obs_files[tz]}: {e})")

    ax.set_yscale('log')
    ax.set_xlim(6.0, 11.0)
    ax.set_ylim(1e-5, 1e-1)
    ax.set_xlabel(r'$\log_{10} M_{\rm BH} [M_\odot]$', fontsize=14)
    ax.set_ylabel(r'$\phi$ [Mpc$^{-3}$ dex$^{-1}$]', fontsize=14)
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    leg = ax.legend(loc='upper right', fontsize=9, frameon=False, ncol=2)
    for t in leg.get_texts():
        t.set_fontsize(9)
    plt.tight_layout()
    out = os.path.join(output_dir, f"BlackHoleMassFunction{OutputFormat}")
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[ok]   {name} -> {out}")


# ============================================================================
# MAIN
# ============================================================================
def _parse_edges(s, default):
    if not s:
        return default
    try:
        return [float(x) for x in s.split(',')]
    except ValueError:
        print(f"  ! could not parse edges '{s}', using defaults.")
        return default


def main():
    p = argparse.ArgumentParser(
        description="Combined SAGE26 black-hole diagnostics.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-i', '--input-pattern',
                   default='./output/millennium/model_*.hdf5',
                   help='Glob for the model HDF5 files.')
    p.add_argument('-s', '--snapshot', type=int, default=None,
                   help='Snapshot for snapshot-dependent plots (default: latest).')
    p.add_argument('-o', '--output-dir', default=None,
                   help='Output directory (default: <input_dir>/plots).')
    p.add_argument('--data-dir', default='./data/bh/',
                   help='Directory with BHMF observational fig4_bhmf_z*.txt files.')
    p.add_argument('--timescale-dir', default=None,
                   help='Directory with timescale .txt dumps '
                        '(default: <input_dir>/timescales then ./timescales).')
    p.add_argument('--units', choices=['code', 'myr'], default='code',
                   help='Units of the timescale dumps (default: code -> Myr).')
    p.add_argument('--bin-mode', choices=['none', 'stellar', 'redshift'],
                   default='none',
                   help='Accretion rate function panelling (default: none).')
    p.add_argument('--stellar-bin-edges', default=None,
                   help='Comma list of 7 log10(M*) edges -> 6 stellar panels.')
    p.add_argument('--panel-redshifts', default=None,
                   help='Comma list of up to 6 redshifts for redshift panels.')
    p.add_argument('--edd-limited', action='store_true',
                   help='Clamp lambda=min(lambda,1) in the rate function.')
    p.add_argument('--sim-volume', type=float, default=None,
                   help='Override comoving volume in (Mpc/h)^3 for the rate function.')
    p.add_argument('--no-cuts', action='store_true',
                   help='Use all galaxies with BH>0 (skip mass cuts).')
    # per-plot switches
    p.add_argument('--no-growth', action='store_true')
    p.add_argument('--no-ratefunc', action='store_true')
    p.add_argument('--no-seeds', action='store_true')
    p.add_argument('--no-timescales', action='store_true')
    p.add_argument('--no-bhbulge', action='store_true')
    p.add_argument('--no-bhmf', action='store_true')
    args = p.parse_args()

    file_list = sorted(glob.glob(args.input_pattern))
    if not file_list:
        print(f"Error: no files match {args.input_pattern}")
        sys.exit(1)

    sim = read_simulation_params(file_list[0])
    hubble_h = sim['Hubble_h']
    redshifts = sim['redshifts']
    available = sim['available_snapshots']
    fracvol = total_volume_fraction(file_list)

    snap_num = args.snapshot if args.snapshot is not None else sim['latest_snapshot']
    if snap_num is None:
        print("Error: could not determine a snapshot to use.")
        sys.exit(1)
    if available and snap_num not in available:
        print(f"Warning: snapshot {snap_num} not available; "
              f"using latest ({sim['latest_snapshot']}).")
        snap_num = sim['latest_snapshot']

    # volumes
    volume_h3 = args.sim_volume if args.sim_volume is not None \
        else (sim['BoxSize'] ** 3) * fracvol          # (Mpc/h)^3
    volume_phys = (sim['BoxSize'] / hubble_h) ** 3 * fracvol   # Mpc^3

    # output dir
    input_dir = os.path.dirname(os.path.abspath(file_list[0]))
    output_dir = args.output_dir or os.path.join(input_dir, 'plots')
    os.makedirs(output_dir, exist_ok=True)

    # timescale dir resolution
    if args.timescale_dir:
        ts_dir = args.timescale_dir
    else:
        cand = os.path.join(input_dir, 'timescales')
        ts_dir = cand if os.path.isdir(cand) else './timescales'

    stellar_edges = _parse_edges(args.stellar_bin_edges, DEFAULT_STELLAR_EDGES)
    panel_z = _parse_edges(args.panel_redshifts, DEFAULT_PANEL_Z)

    z = get_redshift_from_snapshot(snap_num, redshifts)
    print("=" * 70)
    print("SAGE26 black-hole diagnostics")
    print("=" * 70)
    print(f"  files            : {len(file_list)}")
    print(f"  snapshot         : {snap_num}  (z = {z:.3f})")
    print(f"  Hubble_h         : {hubble_h}")
    print(f"  box size         : {sim['BoxSize']} Mpc/h   frac_vol = {fracvol:.4f}")
    print(f"  volume (rate fn) : {volume_h3:.4e} (Mpc/h)^3")
    print(f"  volume (BHMF)    : {volume_phys:.4e} Mpc^3")
    print(f"  bin-mode         : {args.bin_mode}")
    print(f"  output dir       : {output_dir}")
    print("=" * 70)

    if not args.no_growth:
        plot_bh_growth_channels(file_list, snap_num, hubble_h, redshifts,
                                available, output_dir)
    if not args.no_ratefunc:
        plot_accretion_rate_function(file_list, snap_num, hubble_h, redshifts,
                                     available, output_dir, args.bin_mode,
                                     stellar_edges, panel_z, args.edd_limited,
                                     volume_h3, args.no_cuts)
    if not args.no_seeds:
        plot_seed_histogram(file_list, snap_num, hubble_h, output_dir)
    if not args.no_timescales:
        plot_timescales_overlay(file_list, snap_num, ts_dir, args.units, output_dir)
    if not args.no_bhbulge:
        plot_bh_bulge_relation(file_list, snap_num, hubble_h, output_dir)
    if not args.no_bhmf:
        plot_bh_mass_function(file_list, hubble_h, volume_phys, redshifts,
                              available, output_dir, args.data_dir)

    print("=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
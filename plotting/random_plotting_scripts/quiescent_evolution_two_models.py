#!/usr/bin/env python
"""
Quiescent galaxy evolution between two models.
===============================================
Pick the 500 most massive quiescent galaxies in *model 1* at a user-supplied
redshift, match them by GalaxyIndex into *model 2*, then trace each one back
in *model 1* to the earliest snapshot at which it was quiescent.  Record
matched properties at z_target and at the per-galaxy quench snapshot, and
plot side-by-side distributions.

Quiescence: sSFR < 0.2 / t_Hubble(z) = 0.2 * H(z).

Usage:
    python plotting/quiescent_evolution_two_models.py --redshift 2.0
    python plotting/quiescent_evolution_two_models.py --redshift 1.0 --n-top 500 \\
        --model1 ./output/millennium/ --model2 ./output/millennium_oldCGMAGN/
"""

import argparse
import glob
import os
import sys
import warnings

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

_STYLE = './plotting/kieren_cohare_palatino_sty.mplstyle'
if os.path.exists(_STYLE):
    plt.style.use(_STYLE)
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor']   = 'white'
plt.rcParams['axes.edgecolor']   = 'black'


# ----- Configuration -----------------------------------------------------------

DEFAULT_MODEL1       = './output/millennium/'
DEFAULT_MODEL2       = './output/millennium_oldCGMAGN/'
DEFAULT_MODEL1_LABEL = 'SAGE26 (millennium)'
DEFAULT_MODEL2_LABEL = 'SAGE26 (old CGM AGN)'
DEFAULT_REDSHIFT     = 3.0
DEFAULT_N_TOP        = 1000
DEFAULT_OUTPUT_DIR   = './output/quiescent_evolution/'
DEFAULT_FORMAT       = '.pdf'
SSFR_FACTOR          = 0.2

MODEL1_COLOR = '#175cdb'
MODEL2_COLOR = '#d83a21'

# Default cosmology (overridden by header when available).
OMEGA_M_DEFAULT = 0.25
OMEGA_L_DEFAULT = 0.75

PROPS_TO_LOAD = [
    'GalaxyIndex', 'Type', 'StellarMass', 'Mvir', 'BlackHoleMass',
    'SfrDisk', 'SfrBulge', 'ColdGas', 'HotGas', 'CGMgas', 'BulgeMass',
    'IntraClusterStars',
]

_MASS_PROPS = frozenset({
    'Mvir', 'CentralMvir', 'StellarMass', 'BulgeMass', 'BlackHoleMass',
    'ColdGas', 'HotGas', 'CGMgas', 'EjectedMass', 'H2gas', 'H1gas',
    'IntraClusterStars',
})

PANEL_KEYS = ['StellarMass', 'Mvir', 'BlackHoleMass', 'sSFR', 'SFR',
              'ColdGas', 'HotGas', 'CGMgas', 'BoverT', 'ICS']
PANEL_LABELS = {
    'StellarMass':   r'$\log_{10}(M_\star\,/\,M_\odot)$',
    'Mvir':          r'$\log_{10}(M_\mathrm{vir}\,/\,M_\odot)$',
    'BlackHoleMass': r'$\log_{10}(M_\mathrm{BH}\,/\,M_\odot)$',
    'sSFR':          r'$\log_{10}(\mathrm{sSFR}\,/\,\mathrm{yr}^{-1})$',
    'SFR':           r'$\log_{10}(\mathrm{SFR}\,/\,M_\odot\,\mathrm{yr}^{-1})$',
    'ColdGas':       r'$\log_{10}(M_\mathrm{cold}\,/\,M_\odot)$',
    'HotGas':        r'$\log_{10}(M_\mathrm{hot}\,/\,M_\odot)$',
    'CGMgas':        r'$\log_{10}(M_\mathrm{CGM}\,/\,M_\odot)$',
    'BoverT':        r'$B/T$',
    'ICS':           r'$\log_{10}(M_\mathrm{ICS}\,/\,M_\odot)$',
}


# ----- Cosmology helpers -------------------------------------------------------

def _hubble_per_yr(z, omega_m, omega_l, hubble_h):
    """H(z) in 1/yr."""
    inv_H0_yr = (9.778 / hubble_h) * 1.0e9
    return np.sqrt(omega_m * (1.0 + z)**3 + omega_l) / inv_H0_yr


def _ssfr_threshold(z, omega_m, omega_l, hubble_h, factor=SSFR_FACTOR):
    """sSFR threshold (1/yr) below which the galaxy is quiescent."""
    return factor * _hubble_per_yr(z, omega_m, omega_l, hubble_h)


# ----- File I/O ---------------------------------------------------------------

def _find_files(directory):
    files = sorted(glob.glob(os.path.join(directory, 'model_*.hdf5')))
    if not files:
        single = os.path.join(directory, 'model_0.hdf5')
        if os.path.exists(single):
            files = [single]
    return files


def _read_header(directory):
    """
    Read simulation header from the first model file in *directory*, but
    aggregate the available-snapshot set across ALL model_*.hdf5 files.
    """
    files = _find_files(directory)
    if not files:
        return None
    with h5.File(files[0], 'r') as f:
        sim     = f['Header/Simulation']
        runtime = f['Header/Runtime']
        h_val   = float(sim.attrs['hubble_h'])
        hdr = {
            'hubble_h':       h_val,
            'omega_m':        float(sim.attrs.get('omega_m', OMEGA_M_DEFAULT)),
            'omega_l':        float(sim.attrs.get('omega_l', OMEGA_L_DEFAULT)),
            'unit_mass_in_g': float(runtime.attrs['UnitMass_in_g']),
            'redshifts':      list(f['Header/snapshot_redshifts'][:]),
        }
    # Aggregate available snapshots across every file — different files may
    # contain different subsets of the snapshot list in principle.
    snap_set = set()
    for fp in files:
        with h5.File(fp, 'r') as f:
            for k in f.keys():
                if k.startswith('Snap_'):
                    snap_set.add(int(k.replace('Snap_', '')))
    hdr['output_snaps'] = sorted(snap_set)
    hdr['n_files']      = len(files)
    hdr['files']        = files
    hdr['mass_conv']    = hdr['unit_mass_in_g'] / 1.989e33 / hdr['hubble_h']
    return hdr


def _load_snap(files, snap_num, props, mass_conv):
    """
    Return dict {prop: array} concatenated across files for Snap_<snap_num>.

    To stay safe under multi-file output, every requested property is padded
    with zeros to the per-file group length when missing, so the concatenated
    arrays remain row-aligned with GalaxyIndex.  A warning is emitted whenever
    such padding is needed.
    """
    snap_key = f'Snap_{snap_num}'
    chunks = {p: [] for p in props}
    found  = False
    missing = set()
    for fp in files:
        with h5.File(fp, 'r') as f:
            if snap_key not in f:
                continue
            grp = f[snap_key]
            snap_len = None
            for p in props:
                if p in grp:
                    snap_len = int(grp[p].shape[0])
                    break
            if snap_len is None or snap_len == 0:
                continue
            found = True
            for p in props:
                if p in grp:
                    chunks[p].append(np.array(grp[p]))
                else:
                    missing.add((fp, p))
                    chunks[p].append(np.zeros(snap_len))
    if missing:
        for fp, p in sorted(missing):
            print(f'  Warning: field "{p}" missing in {fp}/{snap_key}; padded with zeros.')
    if not found:
        return {}
    out = {}
    for p in props:
        if chunks[p]:
            arr = np.concatenate(chunks[p])
            if p in _MASS_PROPS:
                arr = arr * mass_conv
            out[p] = arr
    return out


def _snap_nearest_z(redshifts, z_target, available):
    arr = np.array(redshifts)
    return min(available, key=lambda s: abs(arr[s] - z_target))


def _build_lookup(galid):
    """GalaxyIndex -> row index."""
    return {int(g): i for i, g in enumerate(galid.astype(np.int64))}


# ----- Property derivations ---------------------------------------------------

def _safe_log10(x):
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan)
    pos = x > 0
    out[pos] = np.log10(x[pos])
    return out


def _compute_panel_array(arrs, key):
    """Return the panel quantity for a dict of property arrays."""
    if key == 'StellarMass':
        return _safe_log10(arrs['StellarMass'])
    if key == 'Mvir':
        return _safe_log10(arrs['Mvir'])
    if key == 'BlackHoleMass':
        return _safe_log10(arrs['BlackHoleMass'])
    if key == 'sSFR':
        sm  = arrs['StellarMass']
        sfr = arrs['SfrDisk'] + arrs['SfrBulge']
        ssfr = np.where(sm > 0, sfr / np.maximum(sm, 1e-30), np.nan)
        return _safe_log10(np.where(np.isfinite(ssfr) & (ssfr > 0), ssfr, np.nan))
    if key == 'SFR':
        return _safe_log10(arrs['SfrDisk'] + arrs['SfrBulge'])
    if key == 'ColdGas':
        return _safe_log10(arrs['ColdGas'])
    if key == 'HotGas':
        return _safe_log10(arrs['HotGas'])
    if key == 'CGMgas':
        return _safe_log10(arrs['CGMgas'])
    if key == 'BoverT':
        sm = arrs['StellarMass']
        return np.where(sm > 0, arrs['BulgeMass'] / np.maximum(sm, 1e-30), np.nan)
    if key == 'ICS':
        return _safe_log10(arrs['IntraClusterStars'])
    raise KeyError(f'Unknown panel key: {key}')


# ----- Main pipeline ----------------------------------------------------------

def _select_top_quiescent(d_target, z_target, n_top, omega_m, omega_l, hubble_h):
    """Return indices of the n_top most massive quiescent galaxies at z_target."""
    sm  = d_target['StellarMass']
    sfr = d_target['SfrDisk'] + d_target['SfrBulge']
    ssfr = np.where(sm > 0, sfr / np.maximum(sm, 1e-30), 0.0)
    thr  = _ssfr_threshold(z_target, omega_m, omega_l, hubble_h)
    quiescent = (sm > 0) & (ssfr < thr)
    n_q = int(quiescent.sum())
    print(f'  sSFR threshold = {thr:.3e} /yr  ->  N_quiescent = {n_q}')
    if n_q == 0:
        return np.array([], dtype=np.int64), thr
    qidx  = np.where(quiescent)[0]
    order = np.argsort(-sm[qidx])
    top   = qidx[order[:n_top]]
    return top


def _count_quiescent(d, ssfr_threshold):
    """Number of galaxies in d with sSFR < ssfr_threshold (and StellarMass > 0)."""
    sm  = d['StellarMass']
    sfr = d['SfrDisk'] + d['SfrBulge']
    ssfr = np.where(sm > 0, sfr / np.maximum(sm, 1e-30), np.inf)
    return int(((sm > 0) & (ssfr < ssfr_threshold)).sum())


def _quiescent_logmvir(d, z, omega_m, omega_l, hubble_h):
    """log10(Mvir/Msun) for every quiescent galaxy in d at redshift z."""
    sm  = d['StellarMass']
    sfr = d['SfrDisk'] + d['SfrBulge']
    ssfr = np.where(sm > 0, sfr / np.maximum(sm, 1e-30), np.inf)
    thr  = _ssfr_threshold(z, omega_m, omega_l, hubble_h)
    sel  = (sm > 0) & (ssfr < thr)
    return _safe_log10(d['Mvir'][sel])


def _match_in_model(galids, d_other):
    """Return rows in d_other whose GalaxyIndex matches galids. -1 where absent."""
    lookup = _build_lookup(d_other['GalaxyIndex'])
    return np.array([lookup.get(int(g), -1) for g in galids], dtype=np.int64)


def _earliest_quench_snap(galids, snap_target, snaps1_data, redshifts,
                          omega_m, omega_l, hubble_h):
    """
    For each galaxy, find the earliest output snapshot s <= snap_target at which
    sSFR < threshold(z_s).  Returns array of snap numbers.
    """
    earliest = np.full(len(galids), snap_target, dtype=int)
    found    = np.zeros(len(galids), dtype=bool)
    snap_order = sorted(s for s in snaps1_data.keys() if s <= snap_target)

    for s in snap_order:
        d = snaps1_data[s]
        lookup = _build_lookup(d['GalaxyIndex'])
        sm  = d['StellarMass']
        sfr = d['SfrDisk'] + d['SfrBulge']
        thr_s = _ssfr_threshold(redshifts[s], omega_m, omega_l, hubble_h)
        for gi, gid in enumerate(galids):
            if found[gi]:
                continue
            i = lookup.get(int(gid), -1)
            if i < 0 or sm[i] <= 0:
                continue
            ssfr_g = sfr[i] / sm[i]
            if ssfr_g < thr_s:
                earliest[gi] = s
                found[gi]    = True
        if found.all():
            break
    return earliest


def _gather_at_quench(galids, quench_snaps, snaps_data):
    """Return dict prop -> array of property values at each galaxy's quench snap."""
    out = {p: np.full(len(galids), np.nan, dtype=float) for p in PROPS_TO_LOAD}
    out['GalaxyIndex'] = np.full(len(galids), -1, dtype=np.int64)
    for gi, (gid, qs) in enumerate(zip(galids, quench_snaps)):
        if qs not in snaps_data:
            continue
        d = snaps_data[qs]
        lookup = _build_lookup(d['GalaxyIndex'])
        i = lookup.get(int(gid), -1)
        if i < 0:
            continue
        for p in PROPS_TO_LOAD:
            if p == 'GalaxyIndex':
                out[p][gi] = int(d[p][i])
            else:
                out[p][gi] = d[p][i]
    return out


def _load_all_snaps_upto(directory, props, snap_max):
    """Load all output snaps with snap_num <= snap_max from directory."""
    files = _find_files(directory)
    hdr   = _read_header(directory)
    if hdr is None or not files:
        return None, {}
    data = {}
    for s in hdr['output_snaps']:
        if s > snap_max:
            continue
        d = _load_snap(files, s, props, hdr['mass_conv'])
        if d:
            data[s] = d
    return hdr, data


# ----- Statistics --------------------------------------------------------------

def _summary_stats(v):
    """Return (n, median, p16, p84, mean, std) for finite values of v."""
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0, np.nan, np.nan, np.nan, np.nan, np.nan
    p16, p50, p84 = np.percentile(v, [16, 50, 84])
    return v.size, p50, p16, p84, float(np.mean(v)), float(np.std(v))


def _print_population_stats(d1, d2, label1, label2, header, redshifts=None,
                            quench_snaps=None):
    """Print median / p16 / p84 / mean / std for each PANEL_KEY in both populations."""
    print('\n' + '-' * 70)
    print(header)
    print('-' * 70)

    if quench_snaps is not None and redshifts is not None:
        zq = np.array([redshifts[s] for s in quench_snaps])
        print(f'  Quench-snap redshift: median={np.median(zq):.3f}  '
              f'p16={np.percentile(zq, 16):.3f}  p84={np.percentile(zq, 84):.3f}  '
              f'min={zq.min():.3f}  max={zq.max():.3f}')
        print(f'  Snap numbers used: {sorted(set(int(s) for s in quench_snaps))}')

    col1 = label1[:24]
    col2 = label2[:24]
    print(f'\n  {"Quantity":<15} {"N1/N2":>10}  '
          f'{col1:>28}  {col2:>28}  {"d(med)":>8}')
    print('  ' + '-' * 96)
    for key in PANEL_KEYS:
        v1 = _compute_panel_array(d1, key)
        v2 = _compute_panel_array(d2, key)
        n1, m1, lo1, hi1, mu1, sd1 = _summary_stats(v1)
        n2, m2, lo2, hi2, mu2, sd2 = _summary_stats(v2)
        s1 = f'{m1:6.2f} [{lo1:5.2f}, {hi1:5.2f}]' if n1 else '---'
        s2 = f'{m2:6.2f} [{lo2:5.2f}, {hi2:5.2f}]' if n2 else '---'
        dmed = (m1 - m2) if (n1 and n2) else np.nan
        dstr = f'{dmed:+6.2f}' if np.isfinite(dmed) else '   ---'
        print(f'  {key:<15} {n1:>4d}/{n2:<5d}  {s1:>28}  {s2:>28}  {dstr:>8}')

    print('  ' + '-' * 96)
    print(f'  Format: median [p16, p84].  d(med) = {label1} - {label2}.')


# ----- Plotting ---------------------------------------------------------------

def _plot_distributions(d1, d2, label1, label2, title, out_path,
                        n_bins=30, weight_label='N'):
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    axes_flat = axes.flatten()

    for pi, key in enumerate(PANEL_KEYS):
        ax = axes_flat[pi]
        v1 = _compute_panel_array(d1, key)
        v2 = _compute_panel_array(d2, key)
        v1 = v1[np.isfinite(v1)]
        v2 = v2[np.isfinite(v2)]
        if len(v1) == 0 and len(v2) == 0:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center')
            ax.set_xlabel(PANEL_LABELS[key])
            continue

        lo = np.inf
        hi = -np.inf
        if len(v1):
            lo = min(lo, v1.min());  hi = max(hi, v1.max())
        if len(v2):
            lo = min(lo, v2.min());  hi = max(hi, v2.max())
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        bins = np.linspace(lo, hi, n_bins + 1)

        if len(v1):
            ax.hist(v1, bins=bins, color=MODEL1_COLOR, alpha=0.55,
                    label=f'{label1} (N={len(v1)})',
                    histtype='stepfilled', edgecolor=MODEL1_COLOR, lw=1.4)
        if len(v2):
            ax.hist(v2, bins=bins, color=MODEL2_COLOR, alpha=0.55,
                    label=f'{label2} (N={len(v2)})',
                    histtype='stepfilled', edgecolor=MODEL2_COLOR, lw=1.4)

        ax.set_xlabel(PANEL_LABELS[key])
        ax.set_ylabel(weight_label)
        ax.tick_params(which='both', direction='in', top=True, right=True)
        if pi == 0:
            ax.legend(loc='best', frameon=False, fontsize=10)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


def _save_quench_snap_histogram(quench_snaps, redshifts, out_path):
    """Optional diagnostic: histogram of redshifts at which galaxies first quenched."""
    z_q = np.array([redshifts[s] for s in quench_snaps])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(z_q.min() - 0.05, z_q.max() + 0.05, 25)
    ax.hist(z_q, bins=bins, color=MODEL1_COLOR, alpha=0.7,
            histtype='stepfilled', edgecolor=MODEL1_COLOR, lw=1.4)
    ax.set_xlabel(r'Redshift of first quiescence')
    ax.set_ylabel(r'$N$ galaxies')
    ax.tick_params(which='both', direction='in', top=True, right=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


# ----- Redshift-sweep mode ----------------------------------------------------

# Thresholds at which the new and old models are declared to "deviate" at a
# given snap. We require BOTH to trip simultaneously so single-bin noise spikes
# in either statistic don't fire the flag.
KS_PVALUE_THRESHOLD     = 1.0e-2
MEDIAN_SHIFT_DEX        = 0.1
MIN_SAMPLE_PER_MODEL    = 10


def _redshift_sweep(args, hdr, snaps1, snaps2):
    """
    For every snapshot common to both models, build the log10(Mvir) distribution
    of all quiescent galaxies (sSFR < 0.2 H(z)) independently in each model,
    compare with a 2-sample KS test and the median shift, and report the first
    redshift (walking high z -> low z) at which both metrics flag a deviation.
    """
    try:
        from scipy import stats as _stats
    except ImportError:
        sys.exit('redshift-sweep mode requires scipy (scipy.stats.ks_2samp).')

    redshifts = hdr['redshifts']
    omega_m   = hdr['omega_m']
    omega_l   = hdr['omega_l']
    hubble_h  = hdr['hubble_h']

    common = sorted(set(snaps1) & set(snaps2), key=lambda s: -redshifts[s])
    if not common:
        sys.exit('No snapshots common to both models.')

    print('\n' + '=' * 88)
    print('REDSHIFT SWEEP: quiescent log10(Mvir) distribution, model1 vs model2')
    print('=' * 88)
    print(f'  Walking high z -> low z; deviation = (KS p < {KS_PVALUE_THRESHOLD:g}) '
          f'AND (|dmed| > {MEDIAN_SHIFT_DEX} dex).')
    print(f'  {"snap":>4} {"z":>6}  {"N1":>6} {"N2":>6}  '
          f'{"med1":>6} {"med2":>6} {"dmed":>7}  {"KS":>6} {"p":>9}  flag')
    print('  ' + '-' * 78)

    rows = []
    first_div_snap = None
    first_div_z    = None
    last_agree_z   = None

    for s in common:
        z  = redshifts[s]
        v1 = _quiescent_logmvir(snaps1[s], z, omega_m, omega_l, hubble_h)
        v2 = _quiescent_logmvir(snaps2[s], z, omega_m, omega_l, hubble_h)
        v1 = v1[np.isfinite(v1)]
        v2 = v2[np.isfinite(v2)]
        n1, n2 = v1.size, v2.size
        med1 = float(np.median(v1)) if n1 else np.nan
        med2 = float(np.median(v2)) if n2 else np.nan
        dmed = (med1 - med2) if (n1 and n2) else np.nan
        if n1 < MIN_SAMPLE_PER_MODEL or n2 < MIN_SAMPLE_PER_MODEL:
            ks_stat, pval, flag = np.nan, np.nan, 'n/a'
        else:
            ks = _stats.ks_2samp(v1, v2)
            ks_stat = float(ks.statistic)
            pval    = float(ks.pvalue)
            diverges = (pval < KS_PVALUE_THRESHOLD) and (abs(dmed) > MEDIAN_SHIFT_DEX)
            flag = 'DIVERGE' if diverges else 'agree'
            if diverges and first_div_snap is None:
                first_div_snap = s
                first_div_z    = z
            elif not diverges and first_div_snap is None:
                last_agree_z = z
        rows.append((s, z, n1, n2, med1, med2, dmed, ks_stat, pval, flag))
        med1_s = f'{med1:>6.2f}' if np.isfinite(med1) else '   ---'
        med2_s = f'{med2:>6.2f}' if np.isfinite(med2) else '   ---'
        dmed_s = f'{dmed:>+7.2f}' if np.isfinite(dmed) else '    ---'
        ks_s   = f'{ks_stat:>6.3f}' if np.isfinite(ks_stat) else '   ---'
        p_s    = f'{pval:>9.2e}'    if np.isfinite(pval)    else '      ---'
        print(f'  {s:>4d} {z:>6.2f}  {n1:>6d} {n2:>6d}  '
              f'{med1_s} {med2_s} {dmed_s}  {ks_s} {p_s}  {flag}')
    print('  ' + '-' * 78)
    if first_div_z is not None:
        print(f'  First divergence (walking high z -> low z): '
              f'snap {first_div_snap}, z = {first_div_z:.3f}.')
        if last_agree_z is not None:
            print(f'  Last agreeing snap before that: z = {last_agree_z:.3f}.')
    else:
        print('  No snapshot trips both thresholds; distributions never formally diverge.')

    # ---- Per-redshift histogram grid -----------------------------------------
    n_panel = len(common)
    ncols   = min(4, n_panel)
    nrows   = (n_panel + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.3 * nrows),
                             squeeze=False)
    for ai, s in enumerate(common):
        ax = axes.flat[ai]
        z  = redshifts[s]
        v1 = _quiescent_logmvir(snaps1[s], z, omega_m, omega_l, hubble_h)
        v2 = _quiescent_logmvir(snaps2[s], z, omega_m, omega_l, hubble_h)
        v1 = v1[np.isfinite(v1)]
        v2 = v2[np.isfinite(v2)]
        if v1.size == 0 and v2.size == 0:
            ax.text(0.5, 0.5, 'no quiescent', transform=ax.transAxes,
                    ha='center', va='center')
            ax.set_xlabel(PANEL_LABELS['Mvir'])
            ax.set_title(f'z = {z:.2f} (snap {s})', fontsize=10)
            continue
        lo = min((v1.min() if v1.size else np.inf),
                 (v2.min() if v2.size else np.inf))
        hi = max((v1.max() if v1.size else -np.inf),
                 (v2.max() if v2.size else -np.inf))
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        bins = np.linspace(lo, hi, 30)
        if v1.size:
            ax.hist(v1, bins=bins, color=MODEL1_COLOR, alpha=0.55,
                    histtype='stepfilled', edgecolor=MODEL1_COLOR, lw=1.4,
                    label=f'{args.model1_label} (N={v1.size})')
        if v2.size:
            ax.hist(v2, bins=bins, color=MODEL2_COLOR, alpha=0.55,
                    histtype='stepfilled', edgecolor=MODEL2_COLOR, lw=1.4,
                    label=f'{args.model2_label} (N={v2.size})')
        title_marker = ''
        if first_div_snap is not None and s == first_div_snap:
            title_marker = '  [first DIVERGE]'
        ax.set_title(f'z = {z:.2f} (snap {s}){title_marker}', fontsize=10)
        ax.set_xlabel(PANEL_LABELS['Mvir'])
        ax.set_ylabel('N')
        ax.tick_params(which='both', direction='in', top=True, right=True)
        if ai == 0:
            ax.legend(loc='best', frameon=False, fontsize=9)
    for ai in range(n_panel, nrows * ncols):
        axes.flat[ai].axis('off')
    fig.suptitle('Quiescent halo mass distribution per redshift', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    grid_path = os.path.join(args.output_dir,
                             f'quiescent_mvir_per_redshift{args.format}')
    fig.savefig(grid_path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {grid_path}')

    # ---- KS stat and median-shift vs z ---------------------------------------
    finite = [r for r in rows if np.isfinite(r[7])]
    if finite:
        z_arr  = np.array([r[1] for r in finite])
        dm_arr = np.array([r[6] for r in finite])
        ks_arr = np.array([r[7] for r in finite])
        p_arr  = np.array([r[8] for r in finite])
        order  = np.argsort(z_arr)
        z_arr  = z_arr[order];  dm_arr = dm_arr[order]
        ks_arr = ks_arr[order]; p_arr  = p_arr[order]

        fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.2))
        axA.plot(z_arr, ks_arr, 'o-', color='black', lw=1.4)
        axA.set_xlabel('z'); axA.set_ylabel('KS statistic')
        axA.tick_params(which='both', direction='in', top=True, right=True)
        axA2 = axA.twinx()
        axA2.semilogy(z_arr, np.maximum(p_arr, 1e-300), 's--',
                      color='gray', lw=1.0, ms=4)
        axA2.axhline(KS_PVALUE_THRESHOLD, color='red', lw=0.6, ls=':')
        axA2.set_ylabel('KS p-value', color='gray')
        axA2.tick_params(axis='y', colors='gray')

        axB.plot(z_arr, dm_arr, 'o-', color='black', lw=1.4)
        axB.axhline(0,                  color='gray', lw=0.5)
        axB.axhline(+MEDIAN_SHIFT_DEX,  color='red',  lw=0.6, ls='--')
        axB.axhline(-MEDIAN_SHIFT_DEX,  color='red',  lw=0.6, ls='--')
        axB.set_xlabel('z')
        axB.set_ylabel(r'$\Delta\,\mathrm{median}\;\log_{10}M_\mathrm{vir}\;'
                       r'(\mathrm{m1}-\mathrm{m2})$')
        axB.tick_params(which='both', direction='in', top=True, right=True)

        if first_div_z is not None:
            for ax in (axA, axB):
                ax.axvline(first_div_z, color='purple', lw=1.0, ls=':',
                           label=f'first DIVERGE: z={first_div_z:.2f}')
            axA.legend(loc='best', frameon=False, fontsize=9)

        fig.suptitle('Quiescent Mvir distribution: divergence vs redshift',
                     fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        div_path = os.path.join(
            args.output_dir, f'quiescent_mvir_divergence_vs_z{args.format}')
        fig.savefig(div_path, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {div_path}')


# ----- Driver -----------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',         default='track',
                        choices=['track', 'redshift-sweep'],
                        help=('"track" (default): existing behaviour - pick top quiescent '
                              'at z_target in model 1, match to model 2, trace back. '
                              '"redshift-sweep": compare the full quiescent Mvir '
                              'distribution at every snapshot to find where the '
                              'two models first deviate.'))
    parser.add_argument('--redshift',     type=float, default=DEFAULT_REDSHIFT,
                        help='Target redshift at which to select the quiescent sample (track mode).')
    parser.add_argument('--n-top',        type=int,   default=DEFAULT_N_TOP,
                        help='Number of most massive quiescent galaxies to track.')
    parser.add_argument('--model1',       default=DEFAULT_MODEL1)
    parser.add_argument('--model2',       default=DEFAULT_MODEL2)
    parser.add_argument('--model1-label', default=DEFAULT_MODEL1_LABEL)
    parser.add_argument('--model2-label', default=DEFAULT_MODEL2_LABEL)
    parser.add_argument('--output-dir',   default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--format',       default=DEFAULT_FORMAT)
    args = parser.parse_args(argv)

    print('=' * 70)
    print(f'Quiescent evolution comparison')
    print('=' * 70)
    print(f'  model1: {args.model1}  ({args.model1_label})')
    print(f'  model2: {args.model2}  ({args.model2_label})')
    print(f'  z_target: {args.redshift}    n_top: {args.n_top}')

    hdr1 = _read_header(args.model1)
    hdr2 = _read_header(args.model2)
    if hdr1 is None:
        sys.exit(f'No model files in {args.model1}')
    if hdr2 is None:
        sys.exit(f'No model files in {args.model2}')

    redshifts = hdr1['redshifts']
    omega_m   = hdr1['omega_m']
    omega_l   = hdr1['omega_l']
    hubble_h  = hdr1['hubble_h']

    print(f'  model1 files: {hdr1["n_files"]} ({len(hdr1["output_snaps"])} snapshots)')
    print(f'  model2 files: {hdr2["n_files"]} ({len(hdr2["output_snaps"])} snapshots)')

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == 'redshift-sweep':
        snap_max = max(max(hdr1['output_snaps']), max(hdr2['output_snaps']))
        print('\nLoading model 1 history (all snapshots)...')
        _, snaps1 = _load_all_snaps_upto(args.model1, PROPS_TO_LOAD, snap_max)
        print(f'  Loaded {len(snaps1)} snapshots from model 1')
        print('Loading model 2 history (all snapshots)...')
        _, snaps2 = _load_all_snaps_upto(args.model2, PROPS_TO_LOAD, snap_max)
        print(f'  Loaded {len(snaps2)} snapshots from model 2')
        _redshift_sweep(args, hdr1, snaps1, snaps2)
        print('\nDone.')
        return

    snap_target = _snap_nearest_z(redshifts, args.redshift, hdr1['output_snaps'])
    z_target    = redshifts[snap_target]
    print(f'  Nearest available snap: {snap_target} (z = {z_target:.3f})')

    if snap_target not in hdr2['output_snaps']:
        sys.exit(f'Model 2 lacks Snap_{snap_target}')

    print('\nLoading model 1 history (snap 0 .. snap_target)...')
    _, snaps1 = _load_all_snaps_upto(args.model1, PROPS_TO_LOAD, snap_target)
    print(f'  Loaded {len(snaps1)} snapshots from model 1')

    print('Loading model 2 history (snap 0 .. snap_target)...')
    _, snaps2 = _load_all_snaps_upto(args.model2, PROPS_TO_LOAD, snap_target)
    print(f'  Loaded {len(snaps2)} snapshots from model 2')

    if snap_target not in snaps1 or snap_target not in snaps2:
        sys.exit('Target snap not loaded for one of the models.')

    # Sanity: GalaxyIndex must be globally unique across all files for the
    # dict-based lookups (and the per-galaxy traceback) to be unambiguous.
    for label, d in [('model1', snaps1[snap_target]), ('model2', snaps2[snap_target])]:
        gid = d['GalaxyIndex'].astype(np.int64)
        n_dup = gid.size - np.unique(gid).size
        if n_dup:
            print(f'  WARNING: {label} snap {snap_target} has {n_dup} duplicate '
                  f'GalaxyIndex values out of {gid.size}. Multi-file matching '
                  f'will be ambiguous.')

    # Step 1: 500 most massive quiescent in model 1 at z_target
    print(f'\nSelecting quiescent sample in model 1 at snap {snap_target}...')
    top_rows = _select_top_quiescent(
        snaps1[snap_target], z_target, args.n_top, omega_m, omega_l, hubble_h)
    if top_rows.size == 0:
        sys.exit('No quiescent galaxies at target snap.')
    galids = snaps1[snap_target]['GalaxyIndex'][top_rows].astype(np.int64)
    print(f'  Selected {len(galids)} galaxies (top-mass quiescent in model 1).')

    # Total quiescent population in each model at z_target (for plot titles).
    ssfr_thr_target = _ssfr_threshold(z_target, omega_m, omega_l, hubble_h)
    n_q_m1 = _count_quiescent(snaps1[snap_target], ssfr_thr_target)
    n_q_m2 = _count_quiescent(snaps2[snap_target], ssfr_thr_target)
    print(f'  Total quiescent at z_target: model1 = {n_q_m1},  model2 = {n_q_m2}')

    # Step 2: match into model 2 at same snap
    print('\nMatching by GalaxyIndex into model 2 at z_target...')
    rows2 = _match_in_model(galids, snaps2[snap_target])
    matched = rows2 >= 0
    print(f'  Matched {matched.sum()} / {len(galids)} in model 2.')
    galids   = galids[matched]
    top_rows = top_rows[matched]
    rows2    = rows2[matched]

    # Properties at z_target for both models
    target_d1 = {k: v[top_rows] for k, v in snaps1[snap_target].items()}
    target_d2 = {k: v[rows2]    for k, v in snaps2[snap_target].items()}

    # Step 3: trace back in model 1 to find earliest quiescent snap per galaxy
    print('\nWalking model 1 history to find earliest quiescent snap per galaxy...')
    quench_snaps = _earliest_quench_snap(
        galids, snap_target, snaps1, redshifts, omega_m, omega_l, hubble_h)
    z_quench = np.array([redshifts[s] for s in quench_snaps])
    print(f'  Quench-snap redshift range: '
          f'z = [{z_quench.min():.2f}, {z_quench.max():.2f}], '
          f'median z = {np.median(z_quench):.2f}')

    # Step 4: gather model 1 and model 2 properties at each quench snap
    print('\nCollecting properties at quench snap...')
    qd1 = _gather_at_quench(galids, quench_snaps, snaps1)
    qd2 = _gather_at_quench(galids, quench_snaps, snaps2)
    n_match2_q = int(np.isfinite(qd2['StellarMass']).sum())
    print(f'  Matched in model 2 at quench snap: {n_match2_q} / {len(galids)}')

    # Step 5: report stats and plot
    tag = f'z{z_target:.2f}'.replace('.', 'p')

    _print_population_stats(
        target_d1, target_d2, args.model1_label, args.model2_label,
        header=f'STATS: matched galaxies at z_target = {z_target:.3f} '
               f'(snap {snap_target})')

    _print_population_stats(
        qd1, qd2, args.model1_label, args.model2_label,
        header='STATS: matched galaxies at each galaxy\'s earliest-quiescent snap',
        redshifts=redshifts, quench_snaps=quench_snaps)

    print('\nPlotting...')
    pop_tag = (f'$N_\\mathrm{{q}}$: {args.model1_label} = {n_q_m1},  '
               f'{args.model2_label} = {n_q_m2}')

    _plot_distributions(
        target_d1, target_d2,
        args.model1_label, args.model2_label,
        title=(f'Quiescent galaxies at z = {z_target:.2f}  '
               f'(top {len(galids)} by $M_\\star$ in model 1; '
               f'sSFR $< {SSFR_FACTOR}/t_H$)\n{pop_tag}'),
        out_path=os.path.join(
            args.output_dir, f'quiescent_distributions_at_target_{tag}{args.format}'))

    _plot_distributions(
        qd1, qd2,
        args.model1_label, args.model2_label,
        title=(f'Properties at each galaxy\'s earliest-quiescent snapshot '
               f'(sample selected at z = {z_target:.2f})\n{pop_tag}'),
        out_path=os.path.join(
            args.output_dir, f'quench_snap_distributions_{tag}{args.format}'))

    _save_quench_snap_histogram(
        quench_snaps, redshifts,
        out_path=os.path.join(args.output_dir, f'quench_redshift_histogram_{tag}{args.format}'))

    print('\nDone.')


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""
ICS assembly-history diagnostics for SAGE26 output.

Uses per-galaxy SFH (SFHMassDisk, SFHMassBulge), ICS assembly tracking
(ICS_disrupt, ICS_accrete), and merger-channel tracking (MergerBulgeMass,
InstabilityBulgeMass, TimeOfLastMajorMerger) to probe the connection between
halo assembly history and the intracluster-stars reservoir.

Plots produced:
  1. t50 / t90 formation-time distributions for GC vs isolated centrals
  2. ICS_disrupt / ICS_total fraction vs M_vir (3-panel)
  3. ICS(Mvir) residual vs t50 formation time (3-panel)
  4. log10(ICS) vs log10(cumulative SFHMassBulge) (3-panel)
  5. ICS(Mvir) residual vs MergerBulgeMass(Mvir) residual (3-panel)
  6. 12+log(Z_ICS/0.02) vs t50 (3-panel)
  7. f_ICL = M_ICS/(M_ICS + M_BCG + M_sats) vs M_vir, with Contini (2021) obs
  8. log10(Z_ICS/Z_*,BCG) offset vs M_vir and t50 — assembly-ordering diagnostic
  9. log10(M_ICS/M_MergerBulge) vs M_vir and t50 — ICS:BCG partition dependence
 10. ICS mass-metallicity relation, split by t50 quartiles
 11. Mass-weighted ICS assembly lookback time (from ICS_sum_mt) vs M_vir,
     vs t50, and delta = t50 - t_assembly vs M_vir
 11b. ICS MZR, split by t_asm quartiles (complements the t50 split)
 12. f_ICL vs M_vir, scatter colored by t_assembly
 13. dZ = log(Z_ICS/Z_BCG) vs t_assembly (chemical-timing coupling)
 14. Multi-redshift t_asm(logMvir) overlay (single figure, all requested z)
 15. Halo concentration vs t_asm at fixed M_vir (halo-assembly tracer test)
 16. t_asm vs TimeOfLast(Major/Minor)Merger (BCG-ICS coevolution)
 17. BCG residual vs ICS residual at fixed M_vir (f_disrupt partition test)
 18. log(M_ICS/M*_BCG) vs t_asm (differential assembly)
 19. t_asm vs t_recent=min(TLMM,TLMm) (merger-timing proxy)
 20. f_disrupt grid scan: post-processing rescaling of f_ICL and M_BCG for a
     range of FractionDisruptedToICS values (exact, using ICS_disrupt /
     ICS_accrete accumulators), to explore ICL-fraction vs BCG-growth trade-off
 21. f_ICS vs redshift (side-by-side): baryon-budget and stellar-fraction
     definitions with observational data overlay (multi-snapshot)
 22. ICS mass function grid: 4x2 redshift panels (z=0–6), split by halo mass
 23. ICS assembly history grid: fractional build-up of disruption (blue) and
     accretion (green) channels vs lookback time, by halo mass bin
"""

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
import glob
import warnings
warnings.filterwarnings('ignore')

OutputFormat = '.pdf'

plt.rcParams["figure.figsize"] = (8.34, 6.25)
plt.rcParams["figure.dpi"] = 140
plt.rcParams["font.size"] = 14
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['legend.frameon'] = False

_style_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'kieren_cohare_palatino_sty.mplstyle')
if os.path.exists(_style_path):
    plt.style.use(_style_path)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _int_levels(H, n=15):
    """Integer contour levels from 1 to max(H)."""
    if not np.any(~np.isnan(H)):
        return np.array([1, 2], dtype=float)
    mx = int(np.nanmax(H))
    if mx < 2:
        return np.array([1, 2], dtype=float)
    if mx <= n:
        return np.arange(1, mx + 1, dtype=float)
    return np.unique(np.round(np.linspace(1, mx, n)).astype(int)).astype(float)


def lookback_time_gyr(z_arr, h=0.73, om=0.25, ol=0.75):
    """Lookback time in Gyr for an array of redshifts (flat LCDM)."""
    H0_inv_Gyr = 9.778 / h   # Hubble time in Gyr for H0=100h km/s/Mpc
    z_arr = np.atleast_1d(np.asarray(z_arr, dtype=float))
    z_fine = np.linspace(0, max(200.0, float(z_arr.max()) + 1.0), 20000)
    E = np.sqrt(om * (1.0 + z_fine)**3 + ol)
    integrand = 1.0 / ((1.0 + z_fine) * E)
    # Cumulative integral by trapezoid
    dz = np.diff(z_fine)
    mid = 0.5 * (integrand[:-1] + integrand[1:])
    cum = np.concatenate([[0.0], np.cumsum(mid * dz)])
    return H0_inv_Gyr * np.interp(z_arr, z_fine, cum)


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def read_simulation_params(filepath):
    params = {}
    with h5.File(filepath, 'r') as f:
        sim = f['Header/Simulation']
        params['Hubble_h'] = float(sim.attrs['hubble_h'])
        params['omega_matter'] = float(sim.attrs['omega_matter'])
        params['omega_lambda'] = float(sim.attrs['omega_lambda'])
        params['PartMass'] = float(sim.attrs['particle_mass']) * 1.0e10 / float(sim.attrs['hubble_h'])

        runtime = f['Header/Runtime']
        params['BaryonFrac'] = float(runtime.attrs.get('BaryonFrac', 0.17))
        params['FractionDisruptedToICS'] = float(runtime.attrs.get('FractionDisruptedToICS', 1.0))
        params['UnitTime_in_Megayears'] = 978028.5 / params['Hubble_h']  # standard SAGE conversion

        params['redshifts'] = np.array(f['Header/snapshot_redshifts'])
        params['output_snapshots'] = np.array(f['Header/output_snapshots'])
        params['BoxSize'] = float(sim.attrs.get('box_size', 62.5))
        params['VolumeFraction'] = float(sim.attrs.get('volume_fraction', 1.0))

        snap_groups = [k for k in f.keys() if k.startswith('Snap_')]
        params['available_snapshots'] = sorted(int(s.replace('Snap_', '')) for s in snap_groups)
        params['last_snapshot'] = max(params['available_snapshots']) if snap_groups else 0

    # Pre-compute lookback times (Gyr) for all snapshots
    params['lookback_gyr'] = lookback_time_gyr(
        params['redshifts'], h=params['Hubble_h'],
        om=params['omega_matter'], ol=params['omega_lambda'])
    return params


def _concat_field(file_list, snap_name, field, hubble_h, is_mass=False, is_2d=False):
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
    return np.concatenate(pieces, axis=0) if not is_2d else np.concatenate(pieces, axis=0)


def load_assembly_data(file_list, sim_params, snap_num):
    """Load snapshot fields needed for assembly diagnostics."""
    h_ = sim_params['Hubble_h']
    snap_name = f'Snap_{snap_num}'

    mass_fields_1d = ['Mvir', 'IntraClusterStars', 'MetalsIntraClusterStars',
                      'StellarMass', 'MetalsStellarMass',
                      'BulgeMass', 'MergerBulgeMass', 'InstabilityBulgeMass',
                      'ICS_disrupt', 'ICS_accrete', 'ICS_sum_mt']
    other_fields_1d = ['Concentration', 'TimeOfLastMajorMerger',
                       'TimeOfLastMinorMerger',
                       'Type', 'GalaxyIndex', 'CentralGalaxyIndex']
    mass_fields_2d = ['SFHMassDisk', 'SFHMassBulge']

    data = {}
    for field in mass_fields_1d:
        data[field] = _concat_field(file_list, snap_name, field, h_, is_mass=True, is_2d=False)
    for field in other_fields_1d:
        data[field] = _concat_field(file_list, snap_name, field, h_, is_mass=False, is_2d=False)
    for field in mass_fields_2d:
        data[field] = _concat_field(file_list, snap_name, field, h_, is_mass=True, is_2d=True)

    if data['Mvir'].size == 0:
        return None

    # Alias to friendlier names used downstream
    data['ICS'] = data['IntraClusterStars']
    data['MetalsICS'] = data['MetalsIntraClusterStars']

    # Mean ICS assembly lookback time (Gyr) from mass-weighted accumulator.
    # ICS_sum_mt = Sum(dm * t_lookback) at each deposition (code units).
    # After the 1e10/h mass conversion on both numerator and denominator,
    # the ratio is in code-time units; multiply by UnitTime_in_Megayears/1000
    # (= 977.8/h) to convert to Gyr.
    utm = sim_params['UnitTime_in_Megayears'] / 1000.0   # code time -> Gyr
    if data['ICS_sum_mt'].size == data['ICS'].size:
        denom = data['ICS_disrupt'] + data['ICS_accrete']
        t_asm = np.full_like(denom, np.nan, dtype=float)
        ok = denom > 0
        t_asm[ok] = data['ICS_sum_mt'][ok] / denom[ok] * utm
        data['ICS_t_assembly_Gyr'] = t_asm
    else:
        data['ICS_t_assembly_Gyr'] = np.full(data['ICS'].size, np.nan)

    # TimeOfLastMajor/Minor Merger are saved to HDF5 as code_time *
    # UnitTime_in_Megayears (i.e. Myr lookback from z=0). Convert to Gyr and
    # flag sentinel values (<=0 means "no merger recorded").
    tlmm_myr = data['TimeOfLastMajorMerger']
    tlmm_gyr = np.where(tlmm_myr > 0, tlmm_myr / 1000.0, np.nan)
    data['TimeOfLastMajorMerger_Gyr'] = tlmm_gyr

    if data.get('TimeOfLastMinorMerger', np.array([])).size == data['ICS'].size:
        tlmm_minor = data['TimeOfLastMinorMerger']
        data['TimeOfLastMinorMerger_Gyr'] = np.where(
            tlmm_minor > 0, tlmm_minor / 1000.0, np.nan)
    else:
        data['TimeOfLastMinorMerger_Gyr'] = np.full(data['ICS'].size, np.nan)

    # Count current satellites per central (searchsorted pattern)
    Type = data['Type']
    GalaxyIndex = data['GalaxyIndex']
    CentralGalaxyIndex = data['CentralGalaxyIndex']

    sorted_idx = np.argsort(GalaxyIndex)
    sorted_gids = GalaxyIndex[sorted_idx]
    sat_mask = Type != 0
    sat_central_gids = CentralGalaxyIndex[sat_mask]
    insert_pos = np.searchsorted(sorted_gids, sat_central_gids)
    insert_pos = np.clip(insert_pos, 0, len(sorted_gids) - 1)
    valid_match = sorted_gids[insert_pos] == sat_central_gids
    central_indices = np.where(valid_match, sorted_idx[insert_pos], -1)
    valid_sats = central_indices >= 0

    n_satellites = np.zeros(len(Type), dtype=int)
    np.add.at(n_satellites, central_indices[valid_sats], 1)
    data['n_satellites'] = n_satellites

    # SFH cumulative total (each galaxy's cumulative stellar mass ever formed)
    # Shape of SFHMassDisk: (N_gal, N_snaps). Snap index = snapshot number.
    sfh_total = data['SFHMassDisk'] + data['SFHMassBulge']      # per-bin mass formed
    data['SFH_total'] = sfh_total
    data['SFH_cumsum'] = np.cumsum(sfh_total, axis=1)           # cumulative by snap
    data['SFHBulge_cumsum'] = np.cumsum(data['SFHMassBulge'], axis=1)

    # Redshifts + lookback times for SFH interpretation
    data['redshifts'] = sim_params['redshifts']
    data['lookback_gyr'] = sim_params['lookback_gyr']
    data['snap_num'] = snap_num
    data['redshift'] = sim_params['redshifts'][snap_num]

    return data


def compute_formation_lookback(cumsfh, lookback_gyr, fraction=0.5):
    """
    For each galaxy, compute the lookback time (Gyr) at which its cumulative
    SFH first reached `fraction` of its current (final) value. Returns NaN
    where the cumulative total is <= 0.

    Uses linear interpolation between the two bracketing snapshots so
    the returned times are continuous (not quantized to snapshot times).
    """
    n_gal, _ = cumsfh.shape
    totals = cumsfh[:, -1]
    out = np.full(n_gal, np.nan)
    has_stars = totals > 0
    targets = np.where(has_stars, fraction * totals, np.inf)
    # snap index where cumulative first meets target
    met = cumsfh >= targets[:, None]                # (n_gal, n_snap) boolean
    first_snap = np.argmax(met, axis=1)            # first True per row
    valid = has_stars & met.any(axis=1)

    vi = np.where(valid)[0]
    fs = first_snap[vi]
    # Interpolate between snap (fs-1) and snap (fs) where the threshold is crossed
    can_interp = fs > 0
    # Galaxies that cross at the very first snapshot — no earlier bracket
    snap_only = vi[~can_interp]
    out[snap_only] = lookback_gyr[fs[~can_interp]]
    # Galaxies with a bracket: interpolate
    idx = vi[can_interp]
    s1 = fs[can_interp]          # first snap >= target
    s0 = s1 - 1                  # last snap < target
    c0 = cumsfh[idx, s0]
    c1 = cumsfh[idx, s1]
    tgt = targets[idx]
    dc = c1 - c0
    frac_step = np.where(dc > 0, (tgt - c0) / dc, 0.0)
    t0 = lookback_gyr[s0]
    t1 = lookback_gyr[s1]
    out[idx] = t0 + frac_step * (t1 - t0)
    return out


# -----------------------------------------------------------------------------
# Shared plotting helpers
# -----------------------------------------------------------------------------

def _contour_panel(ax, xarr, yarr, x_bins, y_bins, cmap='RdPu_r'):
    H, xe, ye = np.histogram2d(xarr, yarr, bins=[x_bins, y_bins])
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    X, Y = np.meshgrid(xc, yc)
    H = H.T
    Hm = np.where(H > 0, H, np.nan)
    cf = ax.contourf(X, Y, Hm, levels=_int_levels(Hm), cmap=cmap)
    return cf


def _running_median(xarr, yarr, bins):
    bcen = 0.5 * (bins[:-1] + bins[1:])
    xs, ys = [], []
    for i in range(len(bins) - 1):
        m = (xarr >= bins[i]) & (xarr < bins[i+1])
        if np.sum(m) >= 5:
            xs.append(bcen[i])
            ys.append(np.median(yarr[m]))
    return np.array(xs), np.array(ys)


def _fit_log_residual(logm, logy):
    slope, intercept = np.polyfit(logm, logy, 1)
    resid = logy - (slope * logm + intercept)
    return slope, intercept, resid


# -----------------------------------------------------------------------------
# Plot 1: t50 / t90 distributions
# -----------------------------------------------------------------------------

def plot_formation_time_distributions(d, sim_params, output_dir):
    redshift = d['redshift']
    min_stellar = sim_params['PartMass'] * sim_params['BaryonFrac']
    print(f'\nFormation-time distributions: snap {d["snap_num"]} (z={redshift:.2f})...')

    cumsfh = d['SFH_cumsum']
    t50 = compute_formation_lookback(cumsfh, d['lookback_gyr'], 0.5)
    t90 = compute_formation_lookback(cumsfh, d['lookback_gyr'], 0.9)

    base_mask = ((d['Type'] == 0) & (d['ICS'] > 0) & (d['Mvir'] > 0)
                 & (d['StellarMass'] >= min_stellar)
                 & np.isfinite(t50) & np.isfinite(t90))
    gc_mask = base_mask & (d['n_satellites'] >= 1)
    iso_mask = base_mask & (d['n_satellites'] == 0)

    print(f'  N_gc={int(np.sum(gc_mask))}, N_iso={int(np.sum(iso_mask))}')
    for label, mk in [('GC', gc_mask), ('Iso', iso_mask)]:
        print(f'  [{label}] t50 med={np.median(t50[mk]):.2f} Gyr (16/84={np.percentile(t50[mk],16):.2f}/{np.percentile(t50[mk],84):.2f}), '
              f't90 med={np.median(t90[mk]):.2f} Gyr')

    fig, (ax_t50, ax_t90) = plt.subplots(1, 2, figsize=(14, 5.5))
    bins = np.linspace(0, 13.8, 40)

    ax_t50.hist(t50[gc_mask], bins=bins, histtype='step', lw=2.25, color='firebrick',
                label=r'Groups/Clusters ($\geq 1$ sat.)', density=True)
    ax_t50.hist(t50[iso_mask], bins=bins, histtype='step', lw=2.25, color='steelblue',
                ls='-.', label='Isolated (no sat.)', density=True)
    ax_t50.axvline(np.median(t50[gc_mask]), color='firebrick', ls=':', lw=1.5)
    ax_t50.axvline(np.median(t50[iso_mask]), color='steelblue', ls=':', lw=1.5)
    ax_t50.set_xlabel(r'$t_{50}$ formation lookback [Gyr]')
    ax_t50.set_ylabel('Density')
    ax_t50.set_xlim(0, 13.8)
    ax_t50.legend(loc='upper right', fontsize=10)

    ax_t90.hist(t90[gc_mask], bins=bins, histtype='step', lw=2.25, color='firebrick',
                label=r'Groups/Clusters ($\geq 1$ sat.)', density=True)
    ax_t90.hist(t90[iso_mask], bins=bins, histtype='step', lw=2.25, color='steelblue',
                ls='-.', label='Isolated (no sat.)', density=True)
    ax_t90.axvline(np.median(t90[gc_mask]), color='firebrick', ls=':', lw=1.5)
    ax_t90.axvline(np.median(t90[iso_mask]), color='steelblue', ls=':', lw=1.5)
    ax_t90.set_xlabel(r'$t_{90}$ formation lookback [Gyr]')
    ax_t90.set_xlim(0, 13.8)
    ax_t90.legend(loc='upper right', fontsize=10)

    fig.tight_layout()
    outfile = os.path.join(output_dir, f'ICS_FormationTime_distributions_z{redshift:.1f}{OutputFormat}')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')

    return t50, t90


# -----------------------------------------------------------------------------
# Plot 2: ICS channel decomposition vs Mvir
# -----------------------------------------------------------------------------

def plot_ics_channel_fraction_vs_mvir(d, sim_params, output_dir):
    redshift = d['redshift']
    min_stellar = sim_params['PartMass'] * sim_params['BaryonFrac']
    print(f'\nICS channel decomposition vs Mvir: z={redshift:.2f}...')

    base_mask = ((d['Type'] == 0) & (d['ICS'] > 0) & (d['Mvir'] > 0)
                 & (d['StellarMass'] >= min_stellar))
    # Require sum of channels > 0 (tracking enabled)
    ch_tot = d['ICS_disrupt'] + d['ICS_accrete']
    base_mask &= (ch_tot > 0)
    gc_mask = base_mask & (d['n_satellites'] >= 1)
    iso_mask = base_mask & (d['n_satellites'] == 0)

    if np.sum(base_mask) < 50:
        print('  Too few haloes.')
        return

    f_disrupt = d['ICS_disrupt'][base_mask] / ch_tot[base_mask]
    logM_all = np.log10(d['Mvir'][base_mask])

    f_disrupt_gc = d['ICS_disrupt'][gc_mask] / ch_tot[gc_mask]
    logM_gc = np.log10(d['Mvir'][gc_mask])
    f_disrupt_iso = d['ICS_disrupt'][iso_mask] / ch_tot[iso_mask]
    logM_iso = np.log10(d['Mvir'][iso_mask])

    def _stats(lbl, f, lm):
        if f.size < 10:
            return
        print(f'  [{lbl}] N={f.size}: median f_disrupt={np.median(f):.3f}, mean={np.mean(f):.3f}')
        for ml in [11.5, 12.0, 12.5, 13.0, 13.5, 14.0]:
            m = (lm >= ml - 0.25) & (lm < ml + 0.25)
            if np.sum(m) >= 5:
                print(f'    logMvir~{ml:.1f} (N={int(np.sum(m))}): median f_disrupt={np.median(f[m]):.3f}')

    _stats('Combined', f_disrupt, logM_all)
    _stats('Groups/Clusters', f_disrupt_gc, logM_gc)
    _stats('Isolated', f_disrupt_iso, logM_iso)

    x_lo = np.floor(logM_all.min() * 2) / 2
    x_hi = np.ceil(logM_all.max() * 2) / 2
    y_lo, y_hi = -0.02, 1.02

    x_bins = np.linspace(x_lo, x_hi, 70)
    y_bins = np.linspace(y_lo, y_hi, 60)

    fig, (ax_gc, ax_iso, ax_comb) = plt.subplots(1, 3, sharey=True, figsize=(24, 6.25))

    xlabel = r'$\log_{10}\, M_{\mathrm{vir}}\ [\mathrm{M}_{\odot}]$'
    ylabel = r'$M_{\mathrm{ICS}}^{\mathrm{disrupt}}\,/\,(M_{\mathrm{ICS}}^{\mathrm{disrupt}} + M_{\mathrm{ICS}}^{\mathrm{accrete}})$'

    bw = max(0.1, (x_hi - x_lo) / 20)
    mb = np.arange(x_lo, x_hi + bw, bw)

    cf_gc = _contour_panel(ax_gc, logM_gc, f_disrupt_gc, x_bins, y_bins)
    xg_, yg_ = _running_median(logM_gc, f_disrupt_gc, mb)
    if xg_.size:
        ax_gc.plot(xg_, yg_, color='gold', ls='-', lw=2.25,
                   label=r'Groups/Clusters ($\geq 1$ sat.)')
    ax_gc.axhline(0.5, color='grey', ls=':', lw=1.5, alpha=0.7)
    ax_gc.legend(loc='lower left', fontsize=10)
    ax_gc.set_xlim(x_lo, x_hi)
    ax_gc.set_ylim(y_lo, y_hi)
    ax_gc.set_xlabel(xlabel)
    ax_gc.set_ylabel(ylabel)

    cf_iso = _contour_panel(ax_iso, logM_iso, f_disrupt_iso, x_bins, y_bins)
    xi_, yi_ = _running_median(logM_iso, f_disrupt_iso, mb)
    if xi_.size:
        ax_iso.plot(xi_, yi_, color='gold', ls='-.', lw=2.25, label='Isolated (no sat.)')
        ax_iso.legend(loc='lower left', fontsize=10)
    ax_iso.axhline(0.5, color='grey', ls=':', lw=1.5, alpha=0.7)
    ax_iso.set_xlim(x_lo, x_hi)
    ax_iso.set_xlabel(xlabel)

    cf_comb = _contour_panel(ax_comb, logM_all, f_disrupt, x_bins, y_bins)
    if xg_.size:
        ax_comb.plot(xg_, yg_, color='gold', ls='-', lw=2.25, label='Groups/Clusters Median')
    if xi_.size:
        ax_comb.plot(xi_, yi_, color='gold', ls='-.', lw=2.25, label='Isolated Median')
    ax_comb.axhline(0.5, color='grey', ls=':', lw=1.5, alpha=0.7)
    ax_comb.legend(loc='lower left', fontsize=10)
    ax_comb.set_xlim(x_lo, x_hi)
    ax_comb.set_xlabel(xlabel)

    fig.colorbar(cf_gc, ax=ax_gc).set_label(r'Number of haloes')
    fig.colorbar(cf_iso, ax=ax_iso).set_label(r'Number of haloes')
    fig.colorbar(cf_comb, ax=ax_comb).set_label(r'Number of haloes')

    fig.tight_layout()
    outfile = os.path.join(output_dir, f'ICS_ChannelFraction_vs_Mvir_z{redshift:.1f}{OutputFormat}')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


# -----------------------------------------------------------------------------
# Plot 3: ICS(Mvir) residual vs t50
# -----------------------------------------------------------------------------

def plot_ics_residual_vs_t50(d, sim_params, output_dir, t50=None):
    redshift = d['redshift']
    min_stellar = sim_params['PartMass'] * sim_params['BaryonFrac']
    print(f'\nICS residual vs t50: z={redshift:.2f}...')

    if t50 is None:
        t50 = compute_formation_lookback(d['SFH_cumsum'], d['lookback_gyr'], 0.5)

    base_mask = ((d['Type'] == 0) & (d['ICS'] > 0) & (d['Mvir'] > 0)
                 & (d['StellarMass'] >= min_stellar) & np.isfinite(t50))
    gc_mask = base_mask & (d['n_satellites'] >= 1)
    iso_mask = base_mask & (d['n_satellites'] == 0)

    if np.sum(base_mask) < 50:
        print('  Too few haloes.')
        return

    logM = np.log10(d['Mvir'][base_mask])
    logIcs = np.log10(d['ICS'][base_mask])
    slope, intercept, _ = _fit_log_residual(logM, logIcs)
    print(f'  Fit: log(M_ICS) = {slope:.3f}*log(M_vir) + {intercept:.3f}')

    def _residual(mask):
        lm = np.log10(d['Mvir'][mask])
        return np.log10(d['ICS'][mask]) - (slope * lm + intercept)

    res_gc = _residual(gc_mask)
    res_iso = _residual(iso_mask)
    res_all = _residual(base_mask)

    t50_gc = t50[gc_mask]
    t50_iso = t50[iso_mask]
    t50_all = t50[base_mask]

    for lbl, tt, rr in [('Groups/Clusters', t50_gc, res_gc),
                        ('Isolated', t50_iso, res_iso),
                        ('Combined', t50_all, res_all)]:
        if rr.size < 10:
            continue
        sl, _ = np.polyfit(tt, rr, 1)
        rho = np.corrcoef(tt, rr)[0, 1]
        print(f'  [{lbl}] N={rr.size}: Pearson r={rho:+.3f}, d(res)/d(t50)={sl:+.3f} dex/Gyr, '
              f'sigma(res)={np.std(rr):.3f}')
        q = np.quantile(tt, [0.25, 0.75])
        low_m = tt <= q[0]
        hi_m = tt >= q[1]
        if np.any(low_m) and np.any(hi_m):
            print(f'    low-t50 quartile (t50<={q[0]:.2f}): median res={np.median(rr[low_m]):+.3f}, '
                  f'high-t50 quartile (t50>={q[1]:.2f}): median res={np.median(rr[hi_m]):+.3f}, '
                  f'delta={np.median(rr[hi_m])-np.median(rr[low_m]):+.3f}')

    x_lo = max(0.0, t50_all.min() - 0.2)
    x_hi = min(13.8, t50_all.max() + 0.2)
    y_lo = min(res_gc.min() if res_gc.size else 0, res_iso.min() if res_iso.size else 0) - 0.1
    y_hi = max(res_gc.max() if res_gc.size else 0, res_iso.max() if res_iso.size else 0) + 0.1

    x_bins = np.linspace(x_lo, x_hi, 70)
    y_bins = np.linspace(y_lo, y_hi, 80)

    fig, (ax_gc, ax_iso, ax_comb) = plt.subplots(1, 3, sharey=True, figsize=(24, 6.25))

    xlabel = r'$t_{50}$ formation lookback [Gyr]'
    ylabel = r'$\log_{10}\, M_{\mathrm{ICS}}\ -\ \log_{10}\, M_{\mathrm{ICS}}^{\mathrm{fit}}(M_{\mathrm{vir}})$'

    bw = max(0.25, (x_hi - x_lo) / 20)
    mb = np.arange(x_lo, x_hi + bw, bw)

    cf_gc = _contour_panel(ax_gc, t50_gc, res_gc, x_bins, y_bins)
    xg_, yg_ = _running_median(t50_gc, res_gc, mb)
    if xg_.size:
        ax_gc.plot(xg_, yg_, color='gold', ls='-', lw=2.25,
                   label=r'Groups/Clusters ($\geq 1$ sat.)')
    ax_gc.axhline(0, color='grey', ls=':', lw=1.5, alpha=0.7)
    ax_gc.legend(loc='lower right', fontsize=10)
    ax_gc.set_xlim(x_lo, x_hi)
    ax_gc.set_ylim(y_lo, y_hi)
    ax_gc.set_xlabel(xlabel)
    ax_gc.set_ylabel(ylabel)

    cf_iso = _contour_panel(ax_iso, t50_iso, res_iso, x_bins, y_bins)
    xi_, yi_ = _running_median(t50_iso, res_iso, mb)
    if xi_.size:
        ax_iso.plot(xi_, yi_, color='gold', ls='-.', lw=2.25, label='Isolated (no sat.)')
        ax_iso.legend(loc='lower right', fontsize=10)
    ax_iso.axhline(0, color='grey', ls=':', lw=1.5, alpha=0.7)
    ax_iso.set_xlim(x_lo, x_hi)
    ax_iso.set_xlabel(xlabel)

    cf_comb = _contour_panel(ax_comb, t50_all, res_all, x_bins, y_bins)
    if xg_.size:
        ax_comb.plot(xg_, yg_, color='gold', ls='-', lw=2.25, label='Groups/Clusters Median')
    if xi_.size:
        ax_comb.plot(xi_, yi_, color='gold', ls='-.', lw=2.25, label='Isolated Median')
    ax_comb.axhline(0, color='grey', ls=':', lw=1.5, alpha=0.7)
    ax_comb.legend(loc='lower right', fontsize=10)
    ax_comb.set_xlim(x_lo, x_hi)
    ax_comb.set_xlabel(xlabel)

    fig.colorbar(cf_gc, ax=ax_gc).set_label(r'Number of haloes')
    fig.colorbar(cf_iso, ax=ax_iso).set_label(r'Number of haloes')
    fig.colorbar(cf_comb, ax=ax_comb).set_label(r'Number of haloes')

    fig.tight_layout()
    outfile = os.path.join(output_dir, f'ICS_Residual_vs_t50_z{redshift:.1f}{OutputFormat}')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


# -----------------------------------------------------------------------------
# Plot 4: log(ICS) vs log(cumulative SFHMassBulge)
# -----------------------------------------------------------------------------

def plot_ics_vs_sfh_bulge(d, sim_params, output_dir):
    redshift = d['redshift']
    min_stellar = sim_params['PartMass'] * sim_params['BaryonFrac']
    print(f'\nlog(ICS) vs log(SFHBulge cum): z={redshift:.2f}...')

    burst_total = d['SFHBulge_cumsum'][:, -1]
    base_mask = ((d['Type'] == 0) & (d['ICS'] > 0) & (d['Mvir'] > 0)
                 & (d['StellarMass'] >= min_stellar) & (burst_total > 0))
    gc_mask = base_mask & (d['n_satellites'] >= 1)
    iso_mask = base_mask & (d['n_satellites'] == 0)

    if np.sum(base_mask) < 50:
        print('  Too few haloes.')
        return

    log_ics_all = np.log10(d['ICS'][base_mask])
    log_burst_all = np.log10(burst_total[base_mask])
    log_ics_gc = np.log10(d['ICS'][gc_mask])
    log_burst_gc = np.log10(burst_total[gc_mask])
    log_ics_iso = np.log10(d['ICS'][iso_mask])
    log_burst_iso = np.log10(burst_total[iso_mask])

    for lbl, x, y in [('Groups/Clusters', log_burst_gc, log_ics_gc),
                      ('Isolated', log_burst_iso, log_ics_iso),
                      ('Combined', log_burst_all, log_ics_all)]:
        if y.size < 10:
            continue
        sl, ic = np.polyfit(x, y, 1)
        rho = np.corrcoef(x, y)[0, 1]
        print(f'  [{lbl}] N={y.size}: Pearson r={rho:+.3f}, slope={sl:+.3f}, intercept={ic:+.3f}')

    x_lo = np.floor(log_burst_all.min() * 2) / 2
    x_hi = np.ceil(log_burst_all.max() * 2) / 2
    y_lo = np.floor(log_ics_all.min() * 2) / 2
    y_hi = np.ceil(log_ics_all.max() * 2) / 2

    x_bins = np.linspace(x_lo, x_hi, 80)
    y_bins = np.linspace(y_lo, y_hi, 80)

    fig, (ax_gc, ax_iso, ax_comb) = plt.subplots(1, 3, sharey=True, figsize=(24, 6.25))

    xlabel = r'$\log_{10}\, \Sigma\, \mathrm{SFHMassBulge}\ [\mathrm{M}_{\odot}]$'
    ylabel = r'$\log_{10}\, M_{\mathrm{ICS}}\ [\mathrm{M}_{\odot}]$'

    bw = max(0.1, (x_hi - x_lo) / 20)
    mb = np.arange(x_lo, x_hi + bw, bw)

    cf_gc = _contour_panel(ax_gc, log_burst_gc, log_ics_gc, x_bins, y_bins)
    xg_, yg_ = _running_median(log_burst_gc, log_ics_gc, mb)
    if xg_.size:
        ax_gc.plot(xg_, yg_, color='gold', ls='-', lw=2.25,
                   label=r'Groups/Clusters ($\geq 1$ sat.)')
    # 1:1 line for reference
    one2one = np.array([x_lo, x_hi])
    ax_gc.plot(one2one, one2one, color='grey', ls=':', lw=1.5, alpha=0.7)
    ax_gc.legend(loc='lower right', fontsize=10)
    ax_gc.set_xlim(x_lo, x_hi)
    ax_gc.set_ylim(y_lo, y_hi)
    ax_gc.set_xlabel(xlabel)
    ax_gc.set_ylabel(ylabel)

    cf_iso = _contour_panel(ax_iso, log_burst_iso, log_ics_iso, x_bins, y_bins)
    xi_, yi_ = _running_median(log_burst_iso, log_ics_iso, mb)
    if xi_.size:
        ax_iso.plot(xi_, yi_, color='gold', ls='-.', lw=2.25, label='Isolated (no sat.)')
        ax_iso.legend(loc='lower right', fontsize=10)
    ax_iso.plot(one2one, one2one, color='grey', ls=':', lw=1.5, alpha=0.7)
    ax_iso.set_xlim(x_lo, x_hi)
    ax_iso.set_xlabel(xlabel)

    cf_comb = _contour_panel(ax_comb, log_burst_all, log_ics_all, x_bins, y_bins)
    if xg_.size:
        ax_comb.plot(xg_, yg_, color='gold', ls='-', lw=2.25, label='Groups/Clusters Median')
    if xi_.size:
        ax_comb.plot(xi_, yi_, color='gold', ls='-.', lw=2.25, label='Isolated Median')
    ax_comb.plot(one2one, one2one, color='grey', ls=':', lw=1.5, alpha=0.7)
    ax_comb.legend(loc='lower right', fontsize=10)
    ax_comb.set_xlim(x_lo, x_hi)
    ax_comb.set_xlabel(xlabel)

    fig.colorbar(cf_gc, ax=ax_gc).set_label(r'Number of haloes')
    fig.colorbar(cf_iso, ax=ax_iso).set_label(r'Number of haloes')
    fig.colorbar(cf_comb, ax=ax_comb).set_label(r'Number of haloes')

    fig.tight_layout()
    outfile = os.path.join(output_dir, f'ICS_vs_SFHBulge_z{redshift:.1f}{OutputFormat}')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


# -----------------------------------------------------------------------------
# Plot 5: ICS residual vs MergerBulgeMass residual (both off Mvir)
# -----------------------------------------------------------------------------

def plot_ics_residual_vs_merger_bulge_residual(d, sim_params, output_dir):
    redshift = d['redshift']
    min_stellar = sim_params['PartMass'] * sim_params['BaryonFrac']
    print(f'\nICS residual vs MergerBulge residual (both off M_vir): z={redshift:.2f}...')

    base_mask = ((d['Type'] == 0) & (d['ICS'] > 0) & (d['Mvir'] > 0)
                 & (d['StellarMass'] >= min_stellar) & (d['MergerBulgeMass'] > 0))
    gc_mask = base_mask & (d['n_satellites'] >= 1)
    iso_mask = base_mask & (d['n_satellites'] == 0)

    if np.sum(base_mask) < 50:
        print('  Too few haloes.')
        return

    logM = np.log10(d['Mvir'][base_mask])
    logIcs = np.log10(d['ICS'][base_mask])
    logMbm = np.log10(d['MergerBulgeMass'][base_mask])

    sl_i, ic_i = np.polyfit(logM, logIcs, 1)
    sl_b, ic_b = np.polyfit(logM, logMbm, 1)
    print(f'  Fit ICS: log(M_ICS) = {sl_i:.3f}*log(M_vir) + {ic_i:.3f}')
    print(f'  Fit MBM: log(M_MBM) = {sl_b:.3f}*log(M_vir) + {ic_b:.3f}')

    def _res(mask):
        lm = np.log10(d['Mvir'][mask])
        ri = np.log10(d['ICS'][mask]) - (sl_i * lm + ic_i)
        rb = np.log10(d['MergerBulgeMass'][mask]) - (sl_b * lm + ic_b)
        return ri, rb

    ri_gc, rb_gc = _res(gc_mask)
    ri_iso, rb_iso = _res(iso_mask)
    ri_all, rb_all = _res(base_mask)

    for lbl, rb, ri in [('Groups/Clusters', rb_gc, ri_gc),
                        ('Isolated', rb_iso, ri_iso),
                        ('Combined', rb_all, ri_all)]:
        if ri.size < 10:
            continue
        sl, _ = np.polyfit(rb, ri, 1)
        rho = np.corrcoef(rb, ri)[0, 1]
        print(f'  [{lbl}] N={ri.size}: Pearson r={rho:+.3f}, slope={sl:+.3f}, '
              f'sigma(ICS_res)={np.std(ri):.3f}, sigma(MBM_res)={np.std(rb):.3f}')

    x_lo = min(rb_gc.min() if rb_gc.size else 0, rb_iso.min() if rb_iso.size else 0) - 0.1
    x_hi = max(rb_gc.max() if rb_gc.size else 0, rb_iso.max() if rb_iso.size else 0) + 0.1
    y_lo = min(ri_gc.min() if ri_gc.size else 0, ri_iso.min() if ri_iso.size else 0) - 0.1
    y_hi = max(ri_gc.max() if ri_gc.size else 0, ri_iso.max() if ri_iso.size else 0) + 0.1

    x_bins = np.linspace(x_lo, x_hi, 80)
    y_bins = np.linspace(y_lo, y_hi, 80)

    fig, (ax_gc, ax_iso, ax_comb) = plt.subplots(1, 3, sharey=True, figsize=(24, 6.25))

    xlabel = r'$\log_{10}\, M_{\mathrm{MergerBulge}}\ -\ \log_{10}\, M_{\mathrm{MergerBulge}}^{\mathrm{fit}}(M_{\mathrm{vir}})$'
    ylabel = r'$\log_{10}\, M_{\mathrm{ICS}}\ -\ \log_{10}\, M_{\mathrm{ICS}}^{\mathrm{fit}}(M_{\mathrm{vir}})$'

    bw = max(0.1, (x_hi - x_lo) / 20)
    mb = np.arange(x_lo, x_hi + bw, bw)

    cf_gc = _contour_panel(ax_gc, rb_gc, ri_gc, x_bins, y_bins)
    xg_, yg_ = _running_median(rb_gc, ri_gc, mb)
    if xg_.size:
        ax_gc.plot(xg_, yg_, color='gold', ls='-', lw=2.25,
                   label=r'Groups/Clusters ($\geq 1$ sat.)')
    ax_gc.axhline(0, color='grey', ls=':', lw=1.5, alpha=0.7)
    ax_gc.axvline(0, color='grey', ls=':', lw=1.5, alpha=0.7)
    ax_gc.legend(loc='lower right', fontsize=10)
    ax_gc.set_xlim(x_lo, x_hi)
    ax_gc.set_ylim(y_lo, y_hi)
    ax_gc.set_xlabel(xlabel)
    ax_gc.set_ylabel(ylabel)

    cf_iso = _contour_panel(ax_iso, rb_iso, ri_iso, x_bins, y_bins)
    xi_, yi_ = _running_median(rb_iso, ri_iso, mb)
    if xi_.size:
        ax_iso.plot(xi_, yi_, color='gold', ls='-.', lw=2.25, label='Isolated (no sat.)')
        ax_iso.legend(loc='lower right', fontsize=10)
    ax_iso.axhline(0, color='grey', ls=':', lw=1.5, alpha=0.7)
    ax_iso.axvline(0, color='grey', ls=':', lw=1.5, alpha=0.7)
    ax_iso.set_xlim(x_lo, x_hi)
    ax_iso.set_xlabel(xlabel)

    cf_comb = _contour_panel(ax_comb, rb_all, ri_all, x_bins, y_bins)
    if xg_.size:
        ax_comb.plot(xg_, yg_, color='gold', ls='-', lw=2.25, label='Groups/Clusters Median')
    if xi_.size:
        ax_comb.plot(xi_, yi_, color='gold', ls='-.', lw=2.25, label='Isolated Median')
    ax_comb.axhline(0, color='grey', ls=':', lw=1.5, alpha=0.7)
    ax_comb.axvline(0, color='grey', ls=':', lw=1.5, alpha=0.7)
    ax_comb.legend(loc='lower right', fontsize=10)
    ax_comb.set_xlim(x_lo, x_hi)
    ax_comb.set_xlabel(xlabel)

    fig.colorbar(cf_gc, ax=ax_gc).set_label(r'Number of haloes')
    fig.colorbar(cf_iso, ax=ax_iso).set_label(r'Number of haloes')
    fig.colorbar(cf_comb, ax=ax_comb).set_label(r'Number of haloes')

    fig.tight_layout()
    outfile = os.path.join(output_dir, f'ICS_Residual_vs_MergerBulge_Residual_z{redshift:.1f}{OutputFormat}')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


# -----------------------------------------------------------------------------
# Plot 6: 12+log(Z_ICS/0.02) vs t50
# -----------------------------------------------------------------------------

def plot_ics_metallicity_vs_t50(d, sim_params, output_dir, t50=None):
    redshift = d['redshift']
    min_stellar = sim_params['PartMass'] * sim_params['BaryonFrac']
    print(f'\n12+log(Z_ICS/0.02) vs t50: z={redshift:.2f}...')

    if t50 is None:
        t50 = compute_formation_lookback(d['SFH_cumsum'], d['lookback_gyr'], 0.5)

    base_mask = ((d['Type'] == 0) & (d['ICS'] > 0) & (d['Mvir'] > 0)
                 & (d['StellarMass'] >= min_stellar)
                 & (d['MetalsICS'] > 0) & np.isfinite(t50))
    gc_mask = base_mask & (d['n_satellites'] >= 1)
    iso_mask = base_mask & (d['n_satellites'] == 0)

    if np.sum(base_mask) < 50:
        print('  Too few haloes.')
        return

    # Metallicity in 12 + log10(Z / 0.02) convention, matching allresults-local.py
    Z_ICS = np.log10(d['MetalsICS'][base_mask] / d['ICS'][base_mask] / 0.02) + 9.0
    Z_ICS_gc = np.log10(d['MetalsICS'][gc_mask] / d['ICS'][gc_mask] / 0.02) + 9.0
    Z_ICS_iso = np.log10(d['MetalsICS'][iso_mask] / d['ICS'][iso_mask] / 0.02) + 9.0

    t50_all = t50[base_mask]
    t50_gc = t50[gc_mask]
    t50_iso = t50[iso_mask]

    for lbl, tt, zz in [('Groups/Clusters', t50_gc, Z_ICS_gc),
                        ('Isolated', t50_iso, Z_ICS_iso),
                        ('Combined', t50_all, Z_ICS)]:
        if zz.size < 10:
            continue
        sl, _ = np.polyfit(tt, zz, 1)
        rho = np.corrcoef(tt, zz)[0, 1]
        print(f'  [{lbl}] N={zz.size}: Pearson r={rho:+.3f}, d(Z_ICS)/d(t50)={sl:+.3f} dex/Gyr, '
              f'med Z={np.median(zz):.3f}')

    x_lo = max(0.0, t50_all.min() - 0.2)
    x_hi = min(13.8, t50_all.max() + 0.2)
    y_lo = np.floor(Z_ICS.min() * 2) / 2
    y_hi = np.ceil(Z_ICS.max() * 2) / 2

    x_bins = np.linspace(x_lo, x_hi, 70)
    y_bins = np.linspace(y_lo, y_hi, 80)

    fig, (ax_gc, ax_iso, ax_comb) = plt.subplots(1, 3, sharey=True, figsize=(24, 6.25))

    xlabel = r'$t_{50}$ formation lookback [Gyr]'
    ylabel = r'$12 + \log_{10}(Z_{\mathrm{ICS}}/0.02)$'

    bw = max(0.25, (x_hi - x_lo) / 20)
    mb = np.arange(x_lo, x_hi + bw, bw)

    cf_gc = _contour_panel(ax_gc, t50_gc, Z_ICS_gc, x_bins, y_bins)
    xg_, yg_ = _running_median(t50_gc, Z_ICS_gc, mb)
    if xg_.size:
        ax_gc.plot(xg_, yg_, color='gold', ls='-', lw=2.25,
                   label=r'Groups/Clusters ($\geq 1$ sat.)')
    ax_gc.legend(loc='lower left', fontsize=10)
    ax_gc.set_xlim(x_lo, x_hi)
    ax_gc.set_ylim(y_lo, y_hi)
    ax_gc.set_xlabel(xlabel)
    ax_gc.set_ylabel(ylabel)

    cf_iso = _contour_panel(ax_iso, t50_iso, Z_ICS_iso, x_bins, y_bins)
    xi_, yi_ = _running_median(t50_iso, Z_ICS_iso, mb)
    if xi_.size:
        ax_iso.plot(xi_, yi_, color='gold', ls='-.', lw=2.25, label='Isolated (no sat.)')
        ax_iso.legend(loc='lower left', fontsize=10)
    ax_iso.set_xlim(x_lo, x_hi)
    ax_iso.set_xlabel(xlabel)

    cf_comb = _contour_panel(ax_comb, t50_all, Z_ICS, x_bins, y_bins)
    if xg_.size:
        ax_comb.plot(xg_, yg_, color='gold', ls='-', lw=2.25, label='Groups/Clusters Median')
    if xi_.size:
        ax_comb.plot(xi_, yi_, color='gold', ls='-.', lw=2.25, label='Isolated Median')
    ax_comb.legend(loc='lower left', fontsize=10)
    ax_comb.set_xlim(x_lo, x_hi)
    ax_comb.set_xlabel(xlabel)

    fig.colorbar(cf_gc, ax=ax_gc).set_label(r'Number of haloes')
    fig.colorbar(cf_iso, ax=ax_iso).set_label(r'Number of haloes')
    fig.colorbar(cf_comb, ax=ax_comb).set_label(r'Number of haloes')

    fig.tight_layout()
    outfile = os.path.join(output_dir, f'ICS_Metallicity12_vs_t50_z{redshift:.1f}{OutputFormat}')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


# -----------------------------------------------------------------------------
# Plot 7: ICS fraction vs M_vir, compared to Contini (2021) obs compilation
# -----------------------------------------------------------------------------

def plot_ics_fraction_vs_contini_obs(d, sim_params, output_dir):
    """
    Plot M_ICS / (M_ICS + M_star,halo) vs log M_vir for SAGE26 centrals and
    compare against a curated set of observational ICL-fraction measurements
    (e.g. Zibetti+05, Krick & Bernstein 2007, Burke+15, Presotto+14,
    Montes & Trujillo 2018, Furnell+21, Kluge+21). M_star,halo sums the
    central's StellarMass and the total StellarMass of its satellites; thus
    the quantity is M_ICL / (M_BCG + M_sats + M_ICL), matching the usual
    definition in Contini (2021) and related reviews.
    """
    redshift = d['redshift']
    min_stellar = sim_params['PartMass'] * sim_params['BaryonFrac']
    print(f'\nICS fraction vs observations: z={redshift:.2f}...')

    # Sum satellite StellarMass per central
    Type = d['Type']
    GalaxyIndex = d['GalaxyIndex']
    CentralGalaxyIndex = d['CentralGalaxyIndex']
    StellarMass = d['StellarMass']

    sorted_idx = np.argsort(GalaxyIndex)
    sorted_gids = GalaxyIndex[sorted_idx]
    sat_mask_field = Type != 0
    sat_central_gids = CentralGalaxyIndex[sat_mask_field]
    sat_sm = StellarMass[sat_mask_field]
    insert_pos = np.searchsorted(sorted_gids, sat_central_gids)
    insert_pos = np.clip(insert_pos, 0, len(sorted_gids) - 1)
    valid_match = sorted_gids[insert_pos] == sat_central_gids
    central_idx = np.where(valid_match, sorted_idx[insert_pos], -1)
    valid_sats = central_idx >= 0

    sat_sm_sum = np.zeros(len(Type), dtype=float)
    np.add.at(sat_sm_sum, central_idx[valid_sats], sat_sm[valid_sats])

    M_ICS = d['ICS']
    Mvir = d['Mvir']
    M_halo_stars = StellarMass + sat_sm_sum   # BCG + satellites

    base_mask = ((Type == 0) & (Mvir > 0) & (M_ICS > 0)
                 & (StellarMass >= min_stellar)
                 & (M_halo_stars > 0))

    if np.sum(base_mask) < 20:
        print('  Too few haloes.')
        return

    logMvir = np.log10(Mvir[base_mask])
    f_ICL = M_ICS[base_mask] / (M_ICS[base_mask] + M_halo_stars[base_mask])

    # Binned median + percentiles
    bin_edges = np.arange(11.0, 15.26, 0.25)
    bin_cen = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    med = np.full(len(bin_cen), np.nan)
    p16 = np.full(len(bin_cen), np.nan)
    p84 = np.full(len(bin_cen), np.nan)
    p5 = np.full(len(bin_cen), np.nan)
    p95 = np.full(len(bin_cen), np.nan)
    counts = np.zeros(len(bin_cen), dtype=int)
    for i in range(len(bin_edges) - 1):
        m = (logMvir >= bin_edges[i]) & (logMvir < bin_edges[i+1])
        counts[i] = int(np.sum(m))
        if counts[i] >= 10:
            med[i] = np.median(f_ICL[m])
            p16[i] = np.percentile(f_ICL[m], 16)
            p84[i] = np.percentile(f_ICL[m], 84)
            p5[i] = np.percentile(f_ICL[m], 5)
            p95[i] = np.percentile(f_ICL[m], 95)

    print('  Model f_ICL = M_ICS / (M_ICS + M_BCG + M_sats):')
    for i, bc in enumerate(bin_cen):
        if not np.isnan(med[i]):
            print(f'    logMvir={bc:.2f} (N={counts[i]}): '
                  f'median={med[i]:.3f}, 16/84={p16[i]:.3f}/{p84[i]:.3f}')

    # --- Observational compilation (representative set near z~0) ---
    # Format: (label, log10 M_vir/M200 [Msun], f_ICL, err_f, marker, colour)
    # Values extracted from compilation in Contini (2021) Galaxies 9, 60 and
    # cited primary references. Where authors report only BCG+ICL fractions
    # or ICL-only fractions we have taken the ICL-only value.
    obs = [
        ('Zibetti+05',             14.30, 0.110, 0.030, 'o', 'firebrick'),
        ('Krick \\& Bernstein 07', 14.20, 0.120, 0.050, 's', 'darkorange'),
        ('Presotto+14',            15.10, 0.120, 0.020, 'v', 'seagreen'),
        ('Burke+15',               14.90, 0.030, 0.010, '^', 'navy'),
        ('Montes \\& Trujillo 18', 14.95, 0.150, 0.070, 'D', 'purple'),
        ('Furnell+21',             14.70, 0.240, 0.090, 'P', 'teal'),
        ('Kluge+21',               14.50, 0.300, 0.100, 'X', 'saddlebrown'),
    ]

    fig, ax = plt.subplots(1, 1, figsize=(9.5, 7.0))

    valid = ~np.isnan(med)
    ax.fill_between(bin_cen[valid], p5[valid], p95[valid],
                    color='steelblue', alpha=0.12,
                    label=r'SAGE26 5$-$95$^{\mathrm{th}}$')
    ax.fill_between(bin_cen[valid], p16[valid], p84[valid],
                    color='steelblue', alpha=0.28,
                    label=r'SAGE26 16$-$84$^{\mathrm{th}}$')
    ax.plot(bin_cen[valid], med[valid], color='steelblue', lw=2.5,
            label='SAGE26 median')

    for (lbl, logM, f, err, mk, cl) in obs:
        ax.errorbar([logM], [f], yerr=[[err], [err]], marker=mk, ms=9,
                    color=cl, mfc='white', mec=cl, mew=1.6, capsize=3,
                    elinewidth=1.3, label=lbl, ls='none')

    f_dis = sim_params.get('FractionDisruptedToICS', 1.0)
    ax.text(0.03, 0.97, f'$f_{{\\mathrm{{disrupt}}}} = {f_dis:.2f}$',
            transform=ax.transAxes, va='top', ha='left', fontsize=12,
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                      edgecolor='gold', alpha=0.8))

    ax.set_xlabel(r'$\log_{10}\, M_{\mathrm{vir}}\ [\mathrm{M}_{\odot}]$')
    ax.set_ylabel(r'$M_{\mathrm{ICS}}\,/\,(M_{\mathrm{ICS}} + M_{\star,\mathrm{BCG}} + M_{\star,\mathrm{sats}})$')
    ax.set_xlim(11.5, 15.3)
    ax.set_ylim(0.0, 0.75)
    ax.legend(loc='upper left', fontsize=8, ncol=2, framealpha=0.9)

    fig.tight_layout()
    outfile = os.path.join(output_dir, f'ICS_fraction_vs_Contini_obs_z{redshift:.1f}{OutputFormat}')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


# -----------------------------------------------------------------------------
# Plot 8: Z_ICS - Z_BCG offset vs M_vir and vs t50
# -----------------------------------------------------------------------------

def plot_zics_minus_zbcg(d, sim_params, output_dir, t50=None):
    """
    Assembly-ordering diagnostic. If the ICS is built from tidally stripped
    dwarf satellites, Z_ICS < Z_BCG (negative offset). If the ICS grew along
    with the BCG from major mergers of similarly-massive progenitors the
    offset is ~0. The sign and its dependence on halo mass / t50 tells us
    whether the ICS is progenitor-biased toward low-mass (dwarfs) or
    high-mass (BCG-building majors) satellites.
    """
    redshift = d['redshift']
    min_stellar = sim_params['PartMass'] * sim_params['BaryonFrac']
    print(f'\nZ_ICS - Z_BCG offset vs Mvir & t50: z={redshift:.2f}...')

    if t50 is None:
        t50 = compute_formation_lookback(d['SFH_cumsum'], d['lookback_gyr'], 0.5)

    base_mask = ((d['Type'] == 0) & (d['ICS'] > 0) & (d['Mvir'] > 0)
                 & (d['StellarMass'] >= min_stellar)
                 & (d['MetalsICS'] > 0) & (d['MetalsStellarMass'] > 0)
                 & np.isfinite(t50))

    if np.sum(base_mask) < 50:
        print('  Too few haloes.')
        return

    # log10(Z) — the Z_sun normalisation cancels in the difference
    logZ_ICS = np.log10(d['MetalsICS'][base_mask] / d['ICS'][base_mask])
    logZ_BCG = np.log10(d['MetalsStellarMass'][base_mask] / d['StellarMass'][base_mask])
    dZ = logZ_ICS - logZ_BCG
    logMvir = np.log10(d['Mvir'][base_mask])
    t50_c = t50[base_mask]
    nsat = d['n_satellites'][base_mask]
    gc = nsat >= 1
    iso = nsat == 0

    for lbl, m in [('Groups/Clusters', gc), ('Isolated', iso),
                   ('Combined', np.ones(dZ.size, dtype=bool))]:
        if np.sum(m) < 20:
            continue
        rho_M = np.corrcoef(logMvir[m], dZ[m])[0, 1]
        rho_t = np.corrcoef(t50_c[m], dZ[m])[0, 1]
        print(f'  [{lbl}] N={int(np.sum(m))}: median dZ={np.median(dZ[m]):+.3f} dex, '
              f'16/84={np.percentile(dZ[m],16):+.3f}/{np.percentile(dZ[m],84):+.3f}')
        print(f'    r(dZ, logMvir)={rho_M:+.3f}, r(dZ, t50)={rho_t:+.3f}')

    # --- 2-panel figure ---
    fig, (ax_M, ax_t) = plt.subplots(1, 2, figsize=(16, 6.25), sharey=True)

    # x bins
    xM_lo = np.floor(logMvir.min() * 2) / 2
    xM_hi = np.ceil(logMvir.max() * 2) / 2
    xt_lo = max(0.0, t50_c.min() - 0.2)
    xt_hi = min(13.8, t50_c.max() + 0.2)
    y_lo = np.floor(np.percentile(dZ, 1) * 2) / 2
    y_hi = np.ceil(np.percentile(dZ, 99) * 2) / 2

    xM_bins = np.linspace(xM_lo, xM_hi, 70)
    xt_bins = np.linspace(xt_lo, xt_hi, 70)
    y_bins = np.linspace(y_lo, y_hi, 70)

    _contour_panel(ax_M, logMvir, dZ, xM_bins, y_bins)
    bwM = max(0.1, (xM_hi - xM_lo) / 20)
    mbM = np.arange(xM_lo, xM_hi + bwM, bwM)
    xg_, yg_ = _running_median(logMvir[gc], dZ[gc], mbM)
    xi_, yi_ = _running_median(logMvir[iso], dZ[iso], mbM)
    if xg_.size:
        ax_M.plot(xg_, yg_, color='gold', ls='-', lw=2.25,
                  label=r'GC median ($\geq 1$ sat.)')
    if xi_.size:
        ax_M.plot(xi_, yi_, color='gold', ls='-.', lw=2.25, label='Iso median')
    ax_M.axhline(0, color='grey', ls=':', lw=1.5, alpha=0.7)
    ax_M.set_xlim(xM_lo, xM_hi)
    ax_M.set_ylim(y_lo, y_hi)
    ax_M.set_xlabel(r'$\log_{10}\, M_{\mathrm{vir}}\ [\mathrm{M}_{\odot}]$')
    ax_M.set_ylabel(r'$\log_{10}(Z_{\mathrm{ICS}}/Z_{\star,\mathrm{BCG}})$  [dex]')
    ax_M.legend(loc='lower right', fontsize=10)

    cf = _contour_panel(ax_t, t50_c, dZ, xt_bins, y_bins)
    bwt = max(0.25, (xt_hi - xt_lo) / 20)
    mbt = np.arange(xt_lo, xt_hi + bwt, bwt)
    xg_, yg_ = _running_median(t50_c[gc], dZ[gc], mbt)
    xi_, yi_ = _running_median(t50_c[iso], dZ[iso], mbt)
    if xg_.size:
        ax_t.plot(xg_, yg_, color='gold', ls='-', lw=2.25,
                  label=r'GC median ($\geq 1$ sat.)')
    if xi_.size:
        ax_t.plot(xi_, yi_, color='gold', ls='-.', lw=2.25, label='Iso median')
    ax_t.axhline(0, color='grey', ls=':', lw=1.5, alpha=0.7)
    ax_t.set_xlim(xt_lo, xt_hi)
    ax_t.set_xlabel(r'$t_{50}$ formation lookback [Gyr]')
    ax_t.legend(loc='lower right', fontsize=10)

    fig.colorbar(cf, ax=ax_t).set_label(r'Number of haloes')

    fig.tight_layout()
    outfile = os.path.join(output_dir, f'ICS_ZoffsetBCG_vs_Mvir_t50_z{redshift:.1f}{OutputFormat}')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


# -----------------------------------------------------------------------------
# Plot 9: log(M_ICS / M_MergerBulge) vs M_vir and vs t50
# -----------------------------------------------------------------------------

def plot_ics_over_mergerbulge(d, sim_params, output_dir, t50=None):
    """
    Per unit of merger-driven BCG bulge growth, how much mass ends up as ICS?
    If this ratio is halo-mass / t50 independent, the ICS:BCG partition is a
    pure stochastic split of merging-satellite mass. Any residual trend
    implicates an extra physical dependence (e.g. satellite-progenitor mass
    mix varies with halo assembly).
    """
    redshift = d['redshift']
    min_stellar = sim_params['PartMass'] * sim_params['BaryonFrac']
    print(f'\nlog(M_ICS / M_MergerBulge) vs Mvir & t50: z={redshift:.2f}...')

    if t50 is None:
        t50 = compute_formation_lookback(d['SFH_cumsum'], d['lookback_gyr'], 0.5)

    base_mask = ((d['Type'] == 0) & (d['ICS'] > 0) & (d['Mvir'] > 0)
                 & (d['StellarMass'] >= min_stellar)
                 & (d['MergerBulgeMass'] > 0) & np.isfinite(t50))

    if np.sum(base_mask) < 50:
        print('  Too few haloes.')
        return

    log_ratio = np.log10(d['ICS'][base_mask] / d['MergerBulgeMass'][base_mask])
    logMvir = np.log10(d['Mvir'][base_mask])
    t50_c = t50[base_mask]
    nsat = d['n_satellites'][base_mask]
    gc = nsat >= 1
    iso = nsat == 0

    for lbl, m in [('Groups/Clusters', gc), ('Isolated', iso),
                   ('Combined', np.ones(log_ratio.size, dtype=bool))]:
        if np.sum(m) < 20:
            continue
        rho_M = np.corrcoef(logMvir[m], log_ratio[m])[0, 1]
        rho_t = np.corrcoef(t50_c[m], log_ratio[m])[0, 1]
        print(f'  [{lbl}] N={int(np.sum(m))}: median log(ICS/MBM)={np.median(log_ratio[m]):+.3f}, '
              f'16/84={np.percentile(log_ratio[m],16):+.3f}/{np.percentile(log_ratio[m],84):+.3f}')
        print(f'    r(log(ICS/MBM), logMvir)={rho_M:+.3f}, r(log(ICS/MBM), t50)={rho_t:+.3f}')

    # reference lines
    f_dis = sim_params.get('FractionDisruptedToICS', 0.7)
    ref_pure_disrupt = np.log10(f_dis / (1.0 - f_dis)) if f_dis < 1.0 else None

    fig, (ax_M, ax_t) = plt.subplots(1, 2, figsize=(16, 6.25), sharey=True)

    xM_lo = np.floor(logMvir.min() * 2) / 2
    xM_hi = np.ceil(logMvir.max() * 2) / 2
    xt_lo = max(0.0, t50_c.min() - 0.2)
    xt_hi = min(13.8, t50_c.max() + 0.2)
    y_lo = np.floor(np.percentile(log_ratio, 1) * 2) / 2
    y_hi = np.ceil(np.percentile(log_ratio, 99) * 2) / 2

    xM_bins = np.linspace(xM_lo, xM_hi, 70)
    xt_bins = np.linspace(xt_lo, xt_hi, 70)
    y_bins = np.linspace(y_lo, y_hi, 70)

    _contour_panel(ax_M, logMvir, log_ratio, xM_bins, y_bins)
    bwM = max(0.1, (xM_hi - xM_lo) / 20)
    mbM = np.arange(xM_lo, xM_hi + bwM, bwM)
    xg_, yg_ = _running_median(logMvir[gc], log_ratio[gc], mbM)
    xi_, yi_ = _running_median(logMvir[iso], log_ratio[iso], mbM)
    if xg_.size:
        ax_M.plot(xg_, yg_, color='gold', ls='-', lw=2.25,
                  label=r'GC median ($\geq 1$ sat.)')
    if xi_.size:
        ax_M.plot(xi_, yi_, color='gold', ls='-.', lw=2.25, label='Iso median')
    if ref_pure_disrupt is not None:
        ax_M.axhline(ref_pure_disrupt, color='firebrick', ls='--', lw=1.5,
                     alpha=0.8, label=fr'pure disrupt-split ($f={f_dis:.2f}$)')
    ax_M.axhline(0, color='grey', ls=':', lw=1.5, alpha=0.7)
    ax_M.set_xlim(xM_lo, xM_hi)
    ax_M.set_ylim(y_lo, y_hi)
    ax_M.set_xlabel(r'$\log_{10}\, M_{\mathrm{vir}}\ [\mathrm{M}_{\odot}]$')
    ax_M.set_ylabel(r'$\log_{10}(M_{\mathrm{ICS}}\,/\,M_{\mathrm{MergerBulge}})$')
    ax_M.legend(loc='upper left', fontsize=9)

    cf = _contour_panel(ax_t, t50_c, log_ratio, xt_bins, y_bins)
    bwt = max(0.25, (xt_hi - xt_lo) / 20)
    mbt = np.arange(xt_lo, xt_hi + bwt, bwt)
    xg_, yg_ = _running_median(t50_c[gc], log_ratio[gc], mbt)
    xi_, yi_ = _running_median(t50_c[iso], log_ratio[iso], mbt)
    if xg_.size:
        ax_t.plot(xg_, yg_, color='gold', ls='-', lw=2.25,
                  label=r'GC median ($\geq 1$ sat.)')
    if xi_.size:
        ax_t.plot(xi_, yi_, color='gold', ls='-.', lw=2.25, label='Iso median')
    if ref_pure_disrupt is not None:
        ax_t.axhline(ref_pure_disrupt, color='firebrick', ls='--', lw=1.5, alpha=0.8)
    ax_t.axhline(0, color='grey', ls=':', lw=1.5, alpha=0.7)
    ax_t.set_xlim(xt_lo, xt_hi)
    ax_t.set_xlabel(r'$t_{50}$ formation lookback [Gyr]')
    ax_t.legend(loc='upper left', fontsize=9)

    fig.colorbar(cf, ax=ax_t).set_label(r'Number of haloes')

    fig.tight_layout()
    outfile = os.path.join(output_dir, f'ICS_over_MergerBulge_vs_Mvir_t50_z{redshift:.1f}{OutputFormat}')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


# -----------------------------------------------------------------------------
# Plot 10: ICS mass-metallicity relation, colored by t50
# -----------------------------------------------------------------------------

def plot_ics_mzr(d, sim_params, output_dir, t50=None):
    """
    Mass-metallicity relation for the ICS: 12+log(Z_ICS/0.02) vs log(M_ICS),
    with points binned and median curves computed in t50-quartiles. If a
    positive MZR exists, Z_ICS is driven by the typical progenitor mass mix
    (more massive progenitors = higher Z). Splitting by t50 separates the
    halo-age contribution from the ICS-mass contribution.
    """
    redshift = d['redshift']
    min_stellar = sim_params['PartMass'] * sim_params['BaryonFrac']
    print(f'\nICS MZR, t50-quartile split: z={redshift:.2f}...')

    if t50 is None:
        t50 = compute_formation_lookback(d['SFH_cumsum'], d['lookback_gyr'], 0.5)

    base_mask = ((d['Type'] == 0) & (d['ICS'] > 0) & (d['Mvir'] > 0)
                 & (d['StellarMass'] >= min_stellar)
                 & (d['MetalsICS'] > 0) & np.isfinite(t50))

    if np.sum(base_mask) < 50:
        print('  Too few haloes.')
        return

    logM_ICS = np.log10(d['ICS'][base_mask])
    Z_ICS = np.log10(d['MetalsICS'][base_mask] / d['ICS'][base_mask] / 0.02) + 9.0
    t50_c = t50[base_mask]

    # Global MZR fit
    slope, intercept = np.polyfit(logM_ICS, Z_ICS, 1)
    rho = np.corrcoef(logM_ICS, Z_ICS)[0, 1]
    print(f'  Global MZR: slope={slope:+.3f} dex per dex, intercept={intercept:+.3f}, '
          f'Pearson r={rho:+.3f}, N={logM_ICS.size}')

    # t50 quartiles
    q = np.percentile(t50_c, [0, 25, 50, 75, 100])
    quartiles = [('Q1 (young)',  q[0], q[1]),
                 ('Q2',          q[1], q[2]),
                 ('Q3',          q[2], q[3]),
                 ('Q4 (old)',    q[3], q[4])]
    q_colors = plt.get_cmap('plasma')(np.linspace(0.1, 0.85, 4))

    # per-quartile stats
    for (lbl, lo, hi), col in zip(quartiles, q_colors):
        qm = (t50_c >= lo) & (t50_c <= hi)
        if np.sum(qm) >= 20:
            s, _ = np.polyfit(logM_ICS[qm], Z_ICS[qm], 1)
            r_q = np.corrcoef(logM_ICS[qm], Z_ICS[qm])[0, 1]
            print(f'  {lbl} (t50 {lo:.2f}-{hi:.2f} Gyr, N={int(np.sum(qm))}): '
                  f'slope={s:+.3f}, r={r_q:+.3f}, med Z={np.median(Z_ICS[qm]):.3f}')

    # --- plot ---
    x_lo = np.floor(logM_ICS.min() * 2) / 2
    x_hi = np.ceil(logM_ICS.max() * 2) / 2
    y_lo = np.floor(np.percentile(Z_ICS, 1) * 2) / 2
    y_hi = np.ceil(np.percentile(Z_ICS, 99) * 2) / 2

    fig, ax = plt.subplots(1, 1, figsize=(9.5, 7.0))

    # background 2D histogram (grey) for overall distribution
    x_bins = np.linspace(x_lo, x_hi, 80)
    y_bins = np.linspace(y_lo, y_hi, 80)
    H, xe, ye = np.histogram2d(logM_ICS, Z_ICS, bins=[x_bins, y_bins])
    Xc = 0.5 * (xe[:-1] + xe[1:])
    Yc = 0.5 * (ye[:-1] + ye[1:])
    XX, YY = np.meshgrid(Xc, Yc)
    Hm = np.where(H.T > 0, H.T, np.nan)
    ax.contourf(XX, YY, Hm, levels=_int_levels(Hm), cmap='Greys', alpha=0.5)

    bw = max(0.15, (x_hi - x_lo) / 20)
    mb = np.arange(x_lo, x_hi + bw, bw)

    for (lbl, lo, hi), col in zip(quartiles, q_colors):
        qm = (t50_c >= lo) & (t50_c <= hi)
        if np.sum(qm) < 20:
            continue
        xq, yq = _running_median(logM_ICS[qm], Z_ICS[qm], mb)
        if xq.size:
            ax.plot(xq, yq, color=col, lw=2.5,
                    label=fr'{lbl}  $t_{{50}}\in[{lo:.1f},{hi:.1f}]$ Gyr')

    # global fit line
    xx = np.linspace(x_lo, x_hi, 30)
    ax.plot(xx, slope * xx + intercept, color='black', ls='--', lw=1.75,
            label=fr'global fit: slope={slope:+.2f}, r={rho:+.2f}')

    ax.set_xlabel(r'$\log_{10}\, M_{\mathrm{ICS}}\ [\mathrm{M}_{\odot}]$')
    ax.set_ylabel(r'$12 + \log_{10}(Z_{\mathrm{ICS}}/0.02)$')
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)

    fig.tight_layout()
    outfile = os.path.join(output_dir, f'ICS_MZR_by_t50_z{redshift:.1f}{OutputFormat}')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


def plot_ics_mzr_by_tasm(d, sim_params, output_dir):
    """
    ICS MZR split by quartiles of t_assembly (mass-weighted ICS deposition
    lookback time). Complements the t50 split: if the MZR Q1/Q4 normalization
    offset is driven by *stellar-age*, the t50 split should show it; if it is
    driven by *deposition-time*, the t_asm split should show it; if both, the
    two splits should be qualitatively similar.
    """
    redshift = d['redshift']
    min_stellar = sim_params['PartMass'] * sim_params['BaryonFrac']
    print(f'\nICS MZR, t_asm-quartile split: z={redshift:.2f}...')

    t_asm = d.get('ICS_t_assembly_Gyr', None)
    if t_asm is None or np.all(np.isnan(t_asm)):
        print('  ICS_sum_mt not available; skipping.')
        return

    base_mask = ((d['Type'] == 0) & (d['ICS'] > 0) & (d['Mvir'] > 0)
                 & (d['StellarMass'] >= min_stellar)
                 & (d['MetalsICS'] > 0)
                 & np.isfinite(t_asm) & (t_asm > 0))

    if np.sum(base_mask) < 50:
        print('  Too few haloes.')
        return

    logM_ICS = np.log10(d['ICS'][base_mask])
    Z_ICS = np.log10(d['MetalsICS'][base_mask] / d['ICS'][base_mask] / 0.02) + 9.0
    t_asm_c = t_asm[base_mask]

    # Global MZR fit
    slope, intercept = np.polyfit(logM_ICS, Z_ICS, 1)
    rho = np.corrcoef(logM_ICS, Z_ICS)[0, 1]
    print(f'  Global MZR: slope={slope:+.3f} dex per dex, intercept={intercept:+.3f}, '
          f'Pearson r={rho:+.3f}, N={logM_ICS.size}')

    # t_asm quartiles
    q = np.percentile(t_asm_c, [0, 25, 50, 75, 100])
    quartiles = [('Q1 (recent)', q[0], q[1]),
                 ('Q2',          q[1], q[2]),
                 ('Q3',          q[2], q[3]),
                 ('Q4 (ancient)', q[3], q[4])]
    q_colors = plt.get_cmap('plasma')(np.linspace(0.1, 0.85, 4))

    for (lbl, lo, hi), _ in zip(quartiles, q_colors):
        qm = (t_asm_c >= lo) & (t_asm_c <= hi)
        if np.sum(qm) >= 20:
            s, _ = np.polyfit(logM_ICS[qm], Z_ICS[qm], 1)
            r_q = np.corrcoef(logM_ICS[qm], Z_ICS[qm])[0, 1]
            print(f'  {lbl} (t_asm {lo:.2f}-{hi:.2f} Gyr, N={int(np.sum(qm))}): '
                  f'slope={s:+.3f}, r={r_q:+.3f}, med Z={np.median(Z_ICS[qm]):.3f}')

    # --- plot ---
    x_lo = np.floor(logM_ICS.min() * 2) / 2
    x_hi = np.ceil(logM_ICS.max() * 2) / 2
    y_lo = np.floor(np.percentile(Z_ICS, 1) * 2) / 2
    y_hi = np.ceil(np.percentile(Z_ICS, 99) * 2) / 2

    fig, ax = plt.subplots(1, 1, figsize=(9.5, 7.0))

    x_bins = np.linspace(x_lo, x_hi, 80)
    y_bins = np.linspace(y_lo, y_hi, 80)
    H, xe, ye = np.histogram2d(logM_ICS, Z_ICS, bins=[x_bins, y_bins])
    Xc = 0.5 * (xe[:-1] + xe[1:])
    Yc = 0.5 * (ye[:-1] + ye[1:])
    XX, YY = np.meshgrid(Xc, Yc)
    Hm = np.where(H.T > 0, H.T, np.nan)
    ax.contourf(XX, YY, Hm, levels=_int_levels(Hm), cmap='Greys', alpha=0.5)

    bw = max(0.15, (x_hi - x_lo) / 20)
    mb = np.arange(x_lo, x_hi + bw, bw)

    for (lbl, lo, hi), col in zip(quartiles, q_colors):
        qm = (t_asm_c >= lo) & (t_asm_c <= hi)
        if np.sum(qm) < 20:
            continue
        xq, yq = _running_median(logM_ICS[qm], Z_ICS[qm], mb)
        if xq.size:
            ax.plot(xq, yq, color=col, lw=2.5,
                    label=fr'{lbl}  $t_{{\mathrm{{asm}}}}\in[{lo:.1f},{hi:.1f}]$ Gyr')

    xx = np.linspace(x_lo, x_hi, 30)
    ax.plot(xx, slope * xx + intercept, color='black', ls='--', lw=1.75,
            label=fr'global fit: slope={slope:+.2f}, r={rho:+.2f}')

    ax.set_xlabel(r'$\log_{10}\, M_{\mathrm{ICS}}\ [\mathrm{M}_{\odot}]$')
    ax.set_ylabel(r'$12 + \log_{10}(Z_{\mathrm{ICS}}/0.02)$')
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)

    fig.tight_layout()
    outfile = os.path.join(output_dir, f'ICS_MZR_by_tasm_z{redshift:.1f}{OutputFormat}')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


# -----------------------------------------------------------------------------
# Plot 11: ICS assembly lookback time vs halo properties
# -----------------------------------------------------------------------------

def plot_ics_assembly_time(d, sim_params, output_dir, t50=None):
    """
    Mass-weighted ICS assembly lookback time from the ICS_sum_mt accumulator.

    t_assembly = sum(dm_deposited * t_lookback_at_deposit) / (ICS_disrupt +
    ICS_accrete) gives the mean lookback time at which ICS mass was
    deposited, weighted by the amount deposited.

    Compared to t50 (SFH-based stellar-formation time) this tells us:
      - whether ICS is a "fossil" reservoir (t_assembly large, assembled
        long ago) or a "recent" one (t_assembly small);
      - how much time elapsed between star formation in progenitors and
        deposition into the ICS (delta = t50 - t_assembly; a positive
        delta means stars formed earlier than they were deposited).

    3-panel figure:
      A. t_assembly vs log Mvir
      B. t_assembly vs t50
      C. delta (t50 - t_assembly) vs log Mvir, split by GC / Iso
    """
    redshift = d['redshift']
    min_stellar = sim_params['PartMass'] * sim_params['BaryonFrac']
    print(f'\nICS assembly time vs Mvir & t50: z={redshift:.2f}...')

    t_asm = d.get('ICS_t_assembly_Gyr', None)
    if t_asm is None or np.all(np.isnan(t_asm)):
        print('  ICS_sum_mt not available; skipping.')
        return

    if t50 is None:
        t50 = compute_formation_lookback(d['SFH_cumsum'], d['lookback_gyr'], 0.5)

    base_mask = ((d['Type'] == 0) & (d['Mvir'] > 0)
                 & (d['StellarMass'] >= min_stellar)
                 & (d['ICS'] > 0) & np.isfinite(t_asm) & (t_asm > 0)
                 & np.isfinite(t50))

    if np.sum(base_mask) < 50:
        print('  Too few haloes.')
        return

    t_asm_c = t_asm[base_mask]
    logMvir = np.log10(d['Mvir'][base_mask])
    t50_c = t50[base_mask]
    delta = t50_c - t_asm_c
    nsat = d['n_satellites'][base_mask]
    gc = nsat >= 1
    iso = nsat == 0

    # --- stats ---
    for lbl, m in [('Groups/Clusters', gc), ('Isolated', iso),
                   ('Combined', np.ones(t_asm_c.size, dtype=bool))]:
        if np.sum(m) < 20:
            continue
        rho_M = np.corrcoef(logMvir[m], t_asm_c[m])[0, 1]
        rho_t = np.corrcoef(t50_c[m], t_asm_c[m])[0, 1]
        print(f'  [{lbl}] N={int(np.sum(m))}: median t_assembly={np.median(t_asm_c[m]):.2f} Gyr, '
              f'16/84={np.percentile(t_asm_c[m],16):.2f}/{np.percentile(t_asm_c[m],84):.2f}')
        print(f'    r(t_asm, logMvir)={rho_M:+.3f}, r(t_asm, t50)={rho_t:+.3f}, '
              f'median delta(t50-t_asm)={np.median(delta[m]):+.2f} Gyr')

    # quartile-split stats by t50
    q = np.percentile(t50_c, [25, 50, 75])
    q1 = t50_c <= q[0]
    q4 = t50_c >= q[2]
    if q1.any() and q4.any():
        print(f'  t50-quartile split: Q1 (young, t50<={q[0]:.2f}): '
              f'median t_asm={np.median(t_asm_c[q1]):.2f} Gyr; '
              f'Q4 (old, t50>={q[2]:.2f}): median t_asm={np.median(t_asm_c[q4]):.2f} Gyr; '
              f'delta={np.median(t_asm_c[q4]) - np.median(t_asm_c[q1]):+.2f} Gyr')

    # --- 3-panel figure ---
    fig, (ax_M, ax_t, ax_d) = plt.subplots(1, 3, figsize=(22, 6.25))

    # shared y range for Panels A, B
    y_lo = max(0.0, np.floor(np.percentile(t_asm_c, 1) * 2) / 2)
    y_hi = np.ceil(np.percentile(t_asm_c, 99) * 2) / 2

    # --- Panel A: t_asm vs logMvir ---
    xM_lo = np.floor(logMvir.min() * 2) / 2
    xM_hi = np.ceil(logMvir.max() * 2) / 2
    xM_bins = np.linspace(xM_lo, xM_hi, 70)
    y_bins = np.linspace(y_lo, y_hi, 70)

    _contour_panel(ax_M, logMvir, t_asm_c, xM_bins, y_bins)
    bwM = max(0.1, (xM_hi - xM_lo) / 20)
    mbM = np.arange(xM_lo, xM_hi + bwM, bwM)
    xg_, yg_ = _running_median(logMvir[gc], t_asm_c[gc], mbM)
    xi_, yi_ = _running_median(logMvir[iso], t_asm_c[iso], mbM)
    if xg_.size:
        ax_M.plot(xg_, yg_, color='gold', ls='-', lw=2.25,
                  label=r'GC median ($\geq 1$ sat.)')
    if xi_.size:
        ax_M.plot(xi_, yi_, color='gold', ls='-.', lw=2.25, label='Iso median')
    ax_M.set_xlim(xM_lo, xM_hi)
    ax_M.set_ylim(y_lo, y_hi)
    ax_M.set_xlabel(r'$\log_{10}\, M_{\mathrm{vir}}\ [\mathrm{M}_{\odot}]$')
    ax_M.set_ylabel(r'$\langle t_{\mathrm{assembly,ICS}} \rangle$  [Gyr lookback]')
    ax_M.legend(loc='upper left', fontsize=10)

    # --- Panel B: t_asm vs t50 ---
    xt_lo = max(0.0, t50_c.min() - 0.2)
    xt_hi = min(13.8, t50_c.max() + 0.2)
    xt_bins = np.linspace(xt_lo, xt_hi, 70)
    cf = _contour_panel(ax_t, t50_c, t_asm_c, xt_bins, y_bins)
    bwt = max(0.25, (xt_hi - xt_lo) / 20)
    mbt = np.arange(xt_lo, xt_hi + bwt, bwt)
    xg_, yg_ = _running_median(t50_c[gc], t_asm_c[gc], mbt)
    xi_, yi_ = _running_median(t50_c[iso], t_asm_c[iso], mbt)
    if xg_.size:
        ax_t.plot(xg_, yg_, color='gold', ls='-', lw=2.25,
                  label=r'GC median ($\geq 1$ sat.)')
    if xi_.size:
        ax_t.plot(xi_, yi_, color='gold', ls='-.', lw=2.25, label='Iso median')
    # 1:1 reference
    lo_ref = min(xt_lo, y_lo)
    hi_ref = max(xt_hi, y_hi)
    ax_t.plot([lo_ref, hi_ref], [lo_ref, hi_ref], color='grey', ls=':', lw=1.5,
              alpha=0.7, label=r'$t_{\mathrm{asm}} = t_{50}$')
    ax_t.set_xlim(xt_lo, xt_hi)
    ax_t.set_ylim(y_lo, y_hi)
    ax_t.set_xlabel(r'$t_{50}$ formation lookback [Gyr]')
    ax_t.set_ylabel(r'$\langle t_{\mathrm{assembly,ICS}} \rangle$  [Gyr lookback]')
    ax_t.legend(loc='upper left', fontsize=10)

    # --- Panel C: delta = t50 - t_asm vs logMvir ---
    d_lo = np.floor(np.percentile(delta, 1) * 2) / 2
    d_hi = np.ceil(np.percentile(delta, 99) * 2) / 2
    d_bins = np.linspace(d_lo, d_hi, 70)
    _contour_panel(ax_d, logMvir, delta, xM_bins, d_bins)
    xg_, yg_ = _running_median(logMvir[gc], delta[gc], mbM)
    xi_, yi_ = _running_median(logMvir[iso], delta[iso], mbM)
    if xg_.size:
        ax_d.plot(xg_, yg_, color='gold', ls='-', lw=2.25,
                  label=r'GC median ($\geq 1$ sat.)')
    if xi_.size:
        ax_d.plot(xi_, yi_, color='gold', ls='-.', lw=2.25, label='Iso median')
    ax_d.axhline(0, color='grey', ls=':', lw=1.5, alpha=0.7)
    ax_d.set_xlim(xM_lo, xM_hi)
    ax_d.set_ylim(d_lo, d_hi)
    ax_d.set_xlabel(r'$\log_{10}\, M_{\mathrm{vir}}\ [\mathrm{M}_{\odot}]$')
    ax_d.set_ylabel(r'$t_{50} - \langle t_{\mathrm{asm,ICS}} \rangle$  [Gyr]')
    ax_d.legend(loc='upper right', fontsize=10)

    fig.colorbar(cf, ax=ax_t).set_label(r'Number of haloes')

    fig.tight_layout()
    outfile = os.path.join(output_dir,
                           f'ICS_assembly_time_vs_Mvir_t50_z{redshift:.1f}{OutputFormat}')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


# -----------------------------------------------------------------------------
# Plot 12: f_ICL vs M_vir, colored by t_asm
# -----------------------------------------------------------------------------

def plot_ficl_colored_by_tasm(d, sim_params, output_dir):
    """
    f_ICL = M_ICS/(M_ICS + M_BCG + M_sats) vs log M_vir with per-halo colour
    encoding the mean ICS-deposition lookback time (t_asm). Tests whether
    high-f_ICL haloes are recent-big-merger systems (small t_asm) or
    gradual-build-up systems (large t_asm).
    """
    redshift = d['redshift']
    min_stellar = sim_params['PartMass'] * sim_params['BaryonFrac']
    print(f'\nf_ICL vs M_vir colored by t_asm: z={redshift:.2f}...')

    t_asm = d.get('ICS_t_assembly_Gyr', None)
    if t_asm is None or np.all(np.isnan(t_asm)):
        print('  ICS_sum_mt not available; skipping.')
        return

    # Sum satellite StellarMass per central (same pattern as Contini plot)
    Type = d['Type']
    GalaxyIndex = d['GalaxyIndex']
    CentralGalaxyIndex = d['CentralGalaxyIndex']
    StellarMass = d['StellarMass']

    sorted_idx = np.argsort(GalaxyIndex)
    sorted_gids = GalaxyIndex[sorted_idx]
    sat_mask_field = Type != 0
    sat_central_gids = CentralGalaxyIndex[sat_mask_field]
    sat_sm = StellarMass[sat_mask_field]
    insert_pos = np.searchsorted(sorted_gids, sat_central_gids)
    insert_pos = np.clip(insert_pos, 0, len(sorted_gids) - 1)
    valid_match = sorted_gids[insert_pos] == sat_central_gids
    central_idx = np.where(valid_match, sorted_idx[insert_pos], -1)
    valid_sats = central_idx >= 0
    sat_sm_sum = np.zeros(len(Type), dtype=float)
    np.add.at(sat_sm_sum, central_idx[valid_sats], sat_sm[valid_sats])

    M_ICS = d['ICS']
    Mvir = d['Mvir']
    M_halo_stars = StellarMass + sat_sm_sum

    base_mask = ((Type == 0) & (Mvir > 0) & (M_ICS > 0)
                 & (StellarMass >= min_stellar) & (M_halo_stars > 0)
                 & np.isfinite(t_asm) & (t_asm > 0))

    if np.sum(base_mask) < 20:
        print('  Too few haloes.')
        return

    logMvir = np.log10(Mvir[base_mask])
    f_ICL = M_ICS[base_mask] / (M_ICS[base_mask] + M_halo_stars[base_mask])
    t_asm_c = t_asm[base_mask]

    # --- stats: at fixed Mvir bins, r(f_ICL, t_asm) ---
    print('  r(f_ICL, t_asm) at fixed M_vir:')
    for lm_lo, lm_hi in [(11.5, 12.0), (12.0, 12.5), (12.5, 13.0),
                         (13.0, 13.5), (13.5, 14.5)]:
        m = (logMvir >= lm_lo) & (logMvir < lm_hi)
        if np.sum(m) >= 20:
            r_lm = np.corrcoef(np.log10(f_ICL[m]), t_asm_c[m])[0, 1]
            print(f'    logMvir in [{lm_lo:.1f},{lm_hi:.1f}) '
                  f'(N={int(np.sum(m))}): r={r_lm:+.3f}, '
                  f'median f_ICL={np.median(f_ICL[m]):.3f}, '
                  f'median t_asm={np.median(t_asm_c[m]):.2f} Gyr')

    # --- figure ---
    fig, ax = plt.subplots(1, 1, figsize=(10.0, 7.0))

    # downsample for readability
    idx = np.arange(f_ICL.size)
    if f_ICL.size > 8000:
        rng = np.random.default_rng(42)
        idx = rng.choice(f_ICL.size, size=8000, replace=False)

    sc = ax.scatter(logMvir[idx], f_ICL[idx], c=t_asm_c[idx], s=8, cmap='viridis',
                    vmin=np.percentile(t_asm_c, 2),
                    vmax=np.percentile(t_asm_c, 98), alpha=0.7, lw=0)

    # overlay binned median
    bin_edges = np.arange(11.0, 15.26, 0.25)
    bin_cen = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    med = np.full(len(bin_cen), np.nan)
    for i in range(len(bin_edges) - 1):
        m = (logMvir >= bin_edges[i]) & (logMvir < bin_edges[i+1])
        if np.sum(m) >= 10:
            med[i] = np.median(f_ICL[m])
    v = ~np.isnan(med)
    ax.plot(bin_cen[v], med[v], color='black', lw=2.5, label='median $f_{\\mathrm{ICL}}$')

    ax.set_xlim(11.5, 15.3)
    ax.set_ylim(0.0, 0.75)
    ax.set_xlabel(r'$\log_{10}\, M_{\mathrm{vir}}\ [\mathrm{M}_{\odot}]$')
    ax.set_ylabel(r'$M_{\mathrm{ICS}}\,/\,(M_{\mathrm{ICS}} + M_{\star,\mathrm{BCG}} + M_{\star,\mathrm{sats}})$')
    ax.legend(loc='upper left', fontsize=10)
    fig.colorbar(sc, ax=ax).set_label(r'$\langle t_{\mathrm{asm,ICS}} \rangle$  [Gyr lookback]')

    fig.tight_layout()
    outfile = os.path.join(output_dir,
                           f'ICS_fraction_vs_Mvir_colored_by_tasm_z{redshift:.1f}{OutputFormat}')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


# -----------------------------------------------------------------------------
# Plot 13: dZ = log(Z_ICS/Z_BCG) vs t_assembly
# -----------------------------------------------------------------------------

def plot_dz_vs_tasm(d, sim_params, output_dir):
    """
    Chemical-timing coupling. dZ = log(Z_ICS/Z_*,BCG) vs the mass-weighted
    ICS assembly lookback time (t_asm). If the ICS built up early from
    low-mass, metal-poor satellites, dZ should be more negative at large
    t_asm. Conversely if later deposition is by more equal-mass (metal-
    richer) mergers, dZ should be less negative at small t_asm.

    2-panel figure:
      A. dZ vs t_asm (contour + GC/Iso running medians)
      B. dZ vs log Mvir, scatter colored by t_asm
    """
    redshift = d['redshift']
    min_stellar = sim_params['PartMass'] * sim_params['BaryonFrac']
    print(f'\ndZ = log(Z_ICS/Z_BCG) vs t_assembly: z={redshift:.2f}...')

    t_asm = d.get('ICS_t_assembly_Gyr', None)
    if t_asm is None or np.all(np.isnan(t_asm)):
        print('  ICS_sum_mt not available; skipping.')
        return

    base_mask = ((d['Type'] == 0) & (d['Mvir'] > 0)
                 & (d['StellarMass'] >= min_stellar)
                 & (d['ICS'] > 0) & (d['MetalsICS'] > 0)
                 & (d['MetalsStellarMass'] > 0)
                 & np.isfinite(t_asm) & (t_asm > 0))

    if np.sum(base_mask) < 50:
        print('  Too few haloes.')
        return

    logZ_ICS = np.log10(d['MetalsICS'][base_mask] / d['ICS'][base_mask])
    logZ_BCG = np.log10(d['MetalsStellarMass'][base_mask] / d['StellarMass'][base_mask])
    dZ = logZ_ICS - logZ_BCG
    logMvir = np.log10(d['Mvir'][base_mask])
    t_asm_c = t_asm[base_mask]
    nsat = d['n_satellites'][base_mask]
    gc = nsat >= 1
    iso = nsat == 0

    # --- stats ---
    for lbl, m in [('Groups/Clusters', gc), ('Isolated', iso),
                   ('Combined', np.ones(dZ.size, dtype=bool))]:
        if np.sum(m) < 20:
            continue
        rho = np.corrcoef(t_asm_c[m], dZ[m])[0, 1]
        print(f'  [{lbl}] N={int(np.sum(m))}: r(dZ, t_asm)={rho:+.3f}')
        # quartile split by t_asm
        if np.sum(m) >= 80:
            q = np.percentile(t_asm_c[m], [25, 75])
            early = m & (t_asm_c >= q[1])    # larger lookback = earlier
            late = m & (t_asm_c <= q[0])     # smaller lookback = later
            if early.any() and late.any():
                dd = np.median(dZ[early]) - np.median(dZ[late])
                print(f'    early-assembly Q4 (t_asm>={q[1]:.2f}): med dZ={np.median(dZ[early]):+.3f}; '
                      f'late Q1 (t_asm<={q[0]:.2f}): med dZ={np.median(dZ[late]):+.3f}; '
                      f'delta={dd:+.3f} dex')

    # --- 2-panel figure ---
    fig, (ax_t, ax_M) = plt.subplots(1, 2, figsize=(16, 6.25))

    # panel A: contour + running medians
    t_lo = max(0.0, t_asm_c.min() - 0.2)
    t_hi = min(13.8, t_asm_c.max() + 0.2)
    y_lo = np.floor(np.percentile(dZ, 1) * 2) / 2
    y_hi = np.ceil(np.percentile(dZ, 99) * 2) / 2
    t_bins = np.linspace(t_lo, t_hi, 70)
    y_bins = np.linspace(y_lo, y_hi, 70)

    cf = _contour_panel(ax_t, t_asm_c, dZ, t_bins, y_bins)
    bwt = max(0.25, (t_hi - t_lo) / 20)
    mbt = np.arange(t_lo, t_hi + bwt, bwt)
    xg_, yg_ = _running_median(t_asm_c[gc], dZ[gc], mbt)
    xi_, yi_ = _running_median(t_asm_c[iso], dZ[iso], mbt)
    if xg_.size:
        ax_t.plot(xg_, yg_, color='gold', ls='-', lw=2.25,
                  label=r'GC median ($\geq 1$ sat.)')
    if xi_.size:
        ax_t.plot(xi_, yi_, color='gold', ls='-.', lw=2.25, label='Iso median')
    ax_t.axhline(0, color='grey', ls=':', lw=1.5, alpha=0.7)
    ax_t.set_xlim(t_lo, t_hi)
    ax_t.set_ylim(y_lo, y_hi)
    ax_t.set_xlabel(r'$\langle t_{\mathrm{assembly,ICS}} \rangle$  [Gyr lookback]')
    ax_t.set_ylabel(r'$\log_{10}(Z_{\mathrm{ICS}}/Z_{\star,\mathrm{BCG}})$  [dex]')
    ax_t.legend(loc='lower right', fontsize=10)
    fig.colorbar(cf, ax=ax_t).set_label(r'Number of haloes')

    # panel B: dZ vs logMvir, colored by t_asm
    xM_lo = np.floor(logMvir.min() * 2) / 2
    xM_hi = np.ceil(logMvir.max() * 2) / 2
    # downsample scatter for readability if very large
    idx = np.arange(dZ.size)
    if dZ.size > 6000:
        rng = np.random.default_rng(42)
        idx = rng.choice(dZ.size, size=6000, replace=False)
    sc = ax_M.scatter(logMvir[idx], dZ[idx], c=t_asm_c[idx], s=6, cmap='viridis',
                      vmin=np.percentile(t_asm_c, 2),
                      vmax=np.percentile(t_asm_c, 98), alpha=0.7, lw=0)
    ax_M.axhline(0, color='grey', ls=':', lw=1.5, alpha=0.7)
    bwM = max(0.1, (xM_hi - xM_lo) / 20)
    mbM = np.arange(xM_lo, xM_hi + bwM, bwM)
    xg_, yg_ = _running_median(logMvir[gc], dZ[gc], mbM)
    xi_, yi_ = _running_median(logMvir[iso], dZ[iso], mbM)
    if xg_.size:
        ax_M.plot(xg_, yg_, color='black', ls='-', lw=2.5,
                  label=r'GC median ($\geq 1$ sat.)')
    if xi_.size:
        ax_M.plot(xi_, yi_, color='black', ls='-.', lw=2.5, label='Iso median')
    ax_M.set_xlim(xM_lo, xM_hi)
    ax_M.set_ylim(y_lo, y_hi)
    ax_M.set_xlabel(r'$\log_{10}\, M_{\mathrm{vir}}\ [\mathrm{M}_{\odot}]$')
    ax_M.set_ylabel(r'$\log_{10}(Z_{\mathrm{ICS}}/Z_{\star,\mathrm{BCG}})$  [dex]')
    ax_M.legend(loc='lower right', fontsize=10)
    fig.colorbar(sc, ax=ax_M).set_label(r'$\langle t_{\mathrm{asm,ICS}} \rangle$  [Gyr]')

    fig.tight_layout()
    outfile = os.path.join(output_dir,
                           f'ICS_dZ_vs_tasm_z{redshift:.1f}{OutputFormat}')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


# -----------------------------------------------------------------------------
# Plot 15: Halo concentration vs t_asm at fixed Mvir (halo-assembly tracer)
# -----------------------------------------------------------------------------

def plot_concentration_vs_tasm(d, sim_params, output_dir):
    """
    Halo concentration correlates monotonically with halo assembly time
    (Wechsler+02). A direct test of "is t_asm tracing halo assembly?": at
    fixed M_vir, does Concentration correlate with t_asm?

        positive r(c, t_asm) at fixed Mvir  =>  ICS timing tracks halo timing
        ~zero correlation                   =>  ICS timing is galaxy-physics
                                                driven, not halo-driven

    3-panel figure, one per M_vir bin (group, group-cluster, cluster).
    """
    redshift = d['redshift']
    min_stellar = sim_params['PartMass'] * sim_params['BaryonFrac']
    print(f'\nConcentration vs t_asm at fixed Mvir: z={redshift:.2f}...')

    t_asm = d.get('ICS_t_assembly_Gyr', None)
    if t_asm is None or np.all(np.isnan(t_asm)):
        print('  ICS_sum_mt not available; skipping.')
        return

    c = d.get('Concentration', None)
    if c is None or c.size == 0:
        print('  Concentration field missing; skipping.')
        return

    base_mask = ((d['Type'] == 0) & (d['Mvir'] > 0) & (d['ICS'] > 0)
                 & (d['StellarMass'] >= min_stellar)
                 & np.isfinite(t_asm) & (t_asm > 0) & np.isfinite(c) & (c > 0))
    if np.sum(base_mask) < 50:
        print('  Too few haloes.')
        return

    logMvir = np.log10(d['Mvir'][base_mask])
    c_c = c[base_mask]
    t_asm_c = t_asm[base_mask]

    bins = [(11.5, 12.5, 'group'),
            (12.5, 13.5, 'group/cluster'),
            (13.5, 15.5, 'cluster')]

    print('  r(Concentration, t_asm) at fixed log Mvir:')
    for lo, hi, lbl in bins:
        m = (logMvir >= lo) & (logMvir < hi)
        if np.sum(m) >= 20:
            rho = np.corrcoef(c_c[m], t_asm_c[m])[0, 1]
            s, _ = np.polyfit(c_c[m], t_asm_c[m], 1)
            print(f'    logMvir in [{lo:.1f},{hi:.1f}) '
                  f'({lbl}, N={int(np.sum(m))}): r={rho:+.3f}, '
                  f'd(t_asm)/dc={s:+.3f} Gyr, median c={np.median(c_c[m]):.2f}, '
                  f'median t_asm={np.median(t_asm_c[m]):.2f} Gyr')

    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(22, 6.25))

    c_lo = max(0.5, np.floor(np.percentile(c_c, 1)))
    c_hi = min(25.0, np.ceil(np.percentile(c_c, 99)))
    t_lo = max(0.0, np.floor(np.percentile(t_asm_c, 1) * 2) / 2)
    t_hi = np.ceil(np.percentile(t_asm_c, 99) * 2) / 2

    x_bins = np.linspace(c_lo, c_hi, 60)
    y_bins = np.linspace(t_lo, t_hi, 60)

    xlabel = r'Halo concentration $c$'
    ylabel = r'$\langle t_{\mathrm{asm,ICS}} \rangle$  [Gyr lookback]'

    cf_last = None
    for ax, (lo, hi, lbl) in zip(axes, bins):
        m = (logMvir >= lo) & (logMvir < hi)
        n = int(np.sum(m))
        if n < 20:
            ax.text(0.5, 0.5, f'N={n} (too few)', ha='center', va='center',
                    transform=ax.transAxes, fontsize=11)
            ax.set_xlim(c_lo, c_hi)
            ax.set_ylim(t_lo, t_hi)
            ax.set_xlabel(xlabel)
            continue
        cf_last = _contour_panel(ax, c_c[m], t_asm_c[m], x_bins, y_bins)
        # binned median in c
        bw = max(0.5, (c_hi - c_lo) / 12)
        mb = np.arange(c_lo, c_hi + bw, bw)
        x_, y_ = _running_median(c_c[m], t_asm_c[m], mb)
        if x_.size:
            ax.plot(x_, y_, color='gold', lw=2.25, label='median')
        rho = np.corrcoef(c_c[m], t_asm_c[m])[0, 1]
        ax.set_xlim(c_lo, c_hi)
        ax.set_ylim(t_lo, t_hi)
        ax.set_xlabel(xlabel)
        ax.set_title(rf'$\log_{{10}} M_{{\mathrm{{vir}}}} \in [{lo:.1f},{hi:.1f})$ '
                     rf'— $r={rho:+.2f}$  (N={n})', fontsize=11)
        ax.legend(loc='upper left', fontsize=10)

    axes[0].set_ylabel(ylabel)
    if cf_last is not None:
        fig.colorbar(cf_last, ax=axes[-1]).set_label('Number of haloes')

    fig.tight_layout()
    outfile = os.path.join(output_dir,
                           f'ICS_concentration_vs_tasm_z{redshift:.1f}{OutputFormat}')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


# -----------------------------------------------------------------------------
# Plot 16: TimeOfLast(Major/Minor)Merger vs t_asm (BCG–ICS coevolution)
# -----------------------------------------------------------------------------

def plot_tlmm_vs_tasm(d, sim_params, output_dir):
    """
    BCG–ICS coevolution test. TimeOfLastMajorMerger marks an event that grew
    the merger-driven bulge; t_asm marks when (mass-weighted) ICS was
    deposited. If ICS is dominated by the last major merger, points should
    cluster on the 1:1 line. Systematic t_asm > TLMM means ICS assembly
    leads the last major-merger epoch (dominant contribution from earlier,
    smaller events or from pre-formed ICS accretion).

    Two panels:
      A. t_asm vs TimeOfLastMajorMerger  (1:1 line, GC/Iso medians)
      B. t_asm vs TimeOfLastMinorMerger  (1:1 line, GC/Iso medians)
    """
    redshift = d['redshift']
    min_stellar = sim_params['PartMass'] * sim_params['BaryonFrac']
    print(f'\nTLMM / TLMm vs t_asm: z={redshift:.2f}...')

    t_asm = d.get('ICS_t_assembly_Gyr', None)
    tlmm = d.get('TimeOfLastMajorMerger_Gyr', None)
    tlmnr = d.get('TimeOfLastMinorMerger_Gyr', None)
    if t_asm is None or tlmm is None or tlmnr is None:
        print('  required fields missing; skipping.')
        return

    base_mask = ((d['Type'] == 0) & (d['Mvir'] > 0) & (d['ICS'] > 0)
                 & (d['StellarMass'] >= min_stellar)
                 & np.isfinite(t_asm) & (t_asm > 0))
    if np.sum(base_mask) < 50:
        print('  Too few haloes.')
        return

    nsat = d['n_satellites'][base_mask]
    gc = nsat >= 1
    iso = nsat == 0
    t_asm_c = t_asm[base_mask]
    tlmm_c = tlmm[base_mask]
    tlmnr_c = tlmnr[base_mask]

    def _stats(name, ttt):
        ok = np.isfinite(ttt)
        print(f'  {name}: N_with_event={int(np.sum(ok))} / {ttt.size}')
        for lbl, msel in [('Combined', np.ones_like(ok)),
                          ('GC', gc), ('Iso', iso)]:
            m = ok & msel
            if np.sum(m) < 20:
                continue
            delta = t_asm_c[m] - ttt[m]
            rho = np.corrcoef(ttt[m], t_asm_c[m])[0, 1]
            print(f'    [{lbl}] N={int(np.sum(m))}: '
                  f'r(t_asm, {name})={rho:+.3f}, '
                  f'median delta(t_asm-{name})={np.median(delta):+.2f} Gyr, '
                  f'|delta|<1 Gyr frac={np.mean(np.abs(delta) < 1.0):.2f}')

    _stats('TLMM', tlmm_c)
    _stats('TLMm', tlmnr_c)

    fig, (ax_M, ax_m) = plt.subplots(1, 2, sharey=True, figsize=(16.5, 6.25))

    t_lo = max(0.0, np.floor(np.percentile(t_asm_c, 1) * 2) / 2)
    t_hi = min(13.8, np.ceil(np.percentile(t_asm_c, 99) * 2) / 2)

    def _one_panel(ax, ttt, title_name):
        ok = np.isfinite(ttt)
        if np.sum(ok) < 20:
            ax.text(0.5, 0.5, 'no events', ha='center', va='center',
                    transform=ax.transAxes)
            return None
        x_lo_p = max(0.0, np.floor(np.percentile(ttt[ok], 1) * 2) / 2)
        x_hi_p = min(13.8, np.ceil(np.percentile(ttt[ok], 99) * 2) / 2)
        x_bins = np.linspace(x_lo_p, x_hi_p, 70)
        y_bins = np.linspace(t_lo, t_hi, 70)
        cf = _contour_panel(ax, ttt[ok], t_asm_c[ok], x_bins, y_bins)
        # GC/Iso running medians
        bw = max(0.5, (x_hi_p - x_lo_p) / 18)
        mb = np.arange(x_lo_p, x_hi_p + bw, bw)
        xg_, yg_ = _running_median(ttt[ok & gc], t_asm_c[ok & gc], mb)
        xi_, yi_ = _running_median(ttt[ok & iso], t_asm_c[ok & iso], mb)
        if xg_.size:
            ax.plot(xg_, yg_, color='gold', ls='-', lw=2.25,
                    label=r'GC median ($\geq 1$ sat.)')
        if xi_.size:
            ax.plot(xi_, yi_, color='gold', ls='-.', lw=2.25, label='Iso median')
        lo_ref = min(x_lo_p, t_lo)
        hi_ref = max(x_hi_p, t_hi)
        ax.plot([lo_ref, hi_ref], [lo_ref, hi_ref], color='grey', ls=':', lw=1.5,
                alpha=0.7, label=r'$t_{\mathrm{asm}} = t_{\mathrm{event}}$')
        ax.set_xlim(x_lo_p, x_hi_p)
        ax.set_ylim(t_lo, t_hi)
        ax.set_xlabel(rf'{title_name} lookback [Gyr]')
        ax.legend(loc='upper left', fontsize=10)
        return cf

    cf1 = _one_panel(ax_M, tlmm_c, r'$t_{\mathrm{last\ major\ merger}}$')
    cf2 = _one_panel(ax_m, tlmnr_c, r'$t_{\mathrm{last\ minor\ merger}}$')
    ax_M.set_ylabel(r'$\langle t_{\mathrm{asm,ICS}} \rangle$  [Gyr lookback]')

    cf_use = cf2 if cf2 is not None else cf1
    if cf_use is not None:
        fig.colorbar(cf_use, ax=ax_m).set_label('Number of haloes')

    fig.tight_layout()
    outfile = os.path.join(output_dir,
                           f'ICS_tasm_vs_TLMM_TLMm_z{redshift:.1f}{OutputFormat}')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


# -----------------------------------------------------------------------------
# Plot 17: BCG–ICS residual correlation at fixed Mvir (partition-model test)
# -----------------------------------------------------------------------------

def plot_bcg_ics_residual_correlation(d, sim_params, output_dir):
    """
    The f_disrupt partition couples BCG growth and ICS growth at every
    disruption event, so at fixed M_vir the scatter in BCG and ICS should
    be CORRELATED (not independent). Stripped a lot -> ICS high AND BCG
    high; stripped little -> both low.

    3-panel figure (GC / Iso / Combined). For each, compute
        Delta log BCG = log M*_BCG - <log M*_BCG | log Mvir>
        Delta log ICS = log M_ICS  - <log M_ICS  | log Mvir>
    using a linear fit to the full base sample. Plot scatter and measure
    Pearson r — a strong positive r is the expected signature of the
    f_disrupt-driven co-deposition.
    """
    redshift = d['redshift']
    min_stellar = sim_params['PartMass'] * sim_params['BaryonFrac']
    print(f'\nBCG-ICS residual correlation at fixed Mvir: z={redshift:.2f}...')

    base_mask = ((d['Type'] == 0) & (d['Mvir'] > 0) & (d['ICS'] > 0)
                 & (d['StellarMass'] >= min_stellar))
    if np.sum(base_mask) < 50:
        print('  Too few haloes.')
        return

    logMvir = np.log10(d['Mvir'][base_mask])
    logBcg = np.log10(d['StellarMass'][base_mask])
    logIcs = np.log10(d['ICS'][base_mask])
    nsat = d['n_satellites'][base_mask]
    gc = nsat >= 1
    iso = nsat == 0

    # Linear fit on full sample (both fits)
    sB, iB, dB = _fit_log_residual(logMvir, logBcg)
    sI, iI, dI = _fit_log_residual(logMvir, logIcs)
    print(f'  Fit: log M*_BCG = {sB:.3f}*logMvir + {iB:.3f}')
    print(f'  Fit: log M_ICS  = {sI:.3f}*logMvir + {iI:.3f}')

    for lbl, m in [('Combined', np.ones_like(dB, dtype=bool)),
                   ('Groups/Clusters', gc), ('Isolated', iso)]:
        if np.sum(m) < 20:
            continue
        rho = np.corrcoef(dB[m], dI[m])[0, 1]
        s, _ = np.polyfit(dB[m], dI[m], 1)
        print(f'  [{lbl}] N={int(np.sum(m))}: Pearson r(dBCG,dICS)={rho:+.3f}, '
              f'slope d(dICS)/d(dBCG)={s:+.3f}, '
              f'sigma(dBCG)={np.std(dB[m]):.3f}, sigma(dICS)={np.std(dI[m]):.3f}')

    fig, (ax_gc, ax_iso, ax_all) = plt.subplots(1, 3, sharex=True, sharey=True,
                                                figsize=(22, 6.25))

    x_lo = np.floor(np.percentile(dB, 1) * 2) / 2
    x_hi = np.ceil(np.percentile(dB, 99) * 2) / 2
    y_lo = np.floor(np.percentile(dI, 1) * 2) / 2
    y_hi = np.ceil(np.percentile(dI, 99) * 2) / 2
    x_bins = np.linspace(x_lo, x_hi, 70)
    y_bins = np.linspace(y_lo, y_hi, 70)

    bw = max(0.1, (x_hi - x_lo) / 20)
    mb = np.arange(x_lo, x_hi + bw, bw)

    def _one(ax, m, lbl, ls):
        if np.sum(m) < 20:
            return None
        cf = _contour_panel(ax, dB[m], dI[m], x_bins, y_bins)
        xm_, ym_ = _running_median(dB[m], dI[m], mb)
        if xm_.size:
            ax.plot(xm_, ym_, color='gold', lw=2.25, ls=ls, label=f'{lbl} median')
        # diagonal r-reference
        r_ = np.corrcoef(dB[m], dI[m])[0, 1]
        s_, ii_ = np.polyfit(dB[m], dI[m], 1)
        xrange = np.array([x_lo, x_hi])
        ax.plot(xrange, s_ * xrange + ii_, color='crimson', ls='--', lw=1.5,
                label=rf'fit ($r={r_:+.2f}$)')
        ax.axhline(0, color='grey', ls=':', lw=1.2, alpha=0.7)
        ax.axvline(0, color='grey', ls=':', lw=1.2, alpha=0.7)
        ax.legend(loc='upper left', fontsize=10)
        return cf

    cf_gc = _one(ax_gc, gc, 'Groups/Clusters', '-')
    cf_iso = _one(ax_iso, iso, 'Isolated', '-.')
    cf_all = _one(ax_all, np.ones_like(dB, dtype=bool), 'Combined', '-')

    xlabel = r'$\Delta \log_{10} M_{\star,\mathrm{BCG}}\ |\ M_{\mathrm{vir}}$'
    ylabel = r'$\Delta \log_{10} M_{\mathrm{ICS}}\ |\ M_{\mathrm{vir}}$'
    for ax in (ax_gc, ax_iso, ax_all):
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel(xlabel)
    ax_gc.set_ylabel(ylabel)
    ax_gc.set_title('Groups/Clusters', fontsize=11)
    ax_iso.set_title('Isolated', fontsize=11)
    ax_all.set_title('Combined', fontsize=11)

    for cf, ax in [(cf_gc, ax_gc), (cf_iso, ax_iso), (cf_all, ax_all)]:
        if cf is not None:
            fig.colorbar(cf, ax=ax).set_label('Number of haloes')

    fig.tight_layout()
    outfile = os.path.join(output_dir,
                           f'ICS_BCG_residual_correlation_z{redshift:.1f}{OutputFormat}')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


# -----------------------------------------------------------------------------
# Plot 18: log(M_ICS / M*_BCG) vs t_asm (differential assembly)
# -----------------------------------------------------------------------------

def plot_ics_bcg_ratio_vs_tasm(d, sim_params, output_dir):
    """
    The ICS/BCG mass ratio grows when disruption dominates over in-situ
    BCG star formation. If t_asm is truly an assembly tracer, haloes with
    larger t_asm (older ICS deposition) should have systematically
    different ICS/BCG ratios than haloes with recent deposition.

    3-panel figure (GC / Iso / Combined).
    """
    redshift = d['redshift']
    min_stellar = sim_params['PartMass'] * sim_params['BaryonFrac']
    print(f'\nlog(M_ICS/M*_BCG) vs t_asm: z={redshift:.2f}...')

    t_asm = d.get('ICS_t_assembly_Gyr', None)
    if t_asm is None or np.all(np.isnan(t_asm)):
        print('  ICS_sum_mt not available; skipping.')
        return

    base_mask = ((d['Type'] == 0) & (d['Mvir'] > 0) & (d['ICS'] > 0)
                 & (d['StellarMass'] >= min_stellar)
                 & np.isfinite(t_asm) & (t_asm > 0))
    if np.sum(base_mask) < 50:
        print('  Too few haloes.')
        return

    logRatio = np.log10(d['ICS'][base_mask] / d['StellarMass'][base_mask])
    t_asm_c = t_asm[base_mask]
    nsat = d['n_satellites'][base_mask]
    gc = nsat >= 1
    iso = nsat == 0

    for lbl, m in [('Combined', np.ones_like(gc, dtype=bool)),
                   ('Groups/Clusters', gc), ('Isolated', iso)]:
        if np.sum(m) < 20:
            continue
        rho = np.corrcoef(t_asm_c[m], logRatio[m])[0, 1]
        s, _ = np.polyfit(t_asm_c[m], logRatio[m], 1)
        print(f'  [{lbl}] N={int(np.sum(m))}: '
              f'r(log(ICS/BCG), t_asm)={rho:+.3f}, '
              f'slope={s:+.3f} dex/Gyr, '
              f'median log(ICS/BCG)={np.median(logRatio[m]):+.2f}')
        q = np.percentile(t_asm_c[m], [25, 75])
        low = t_asm_c[m] <= q[0]
        hi = t_asm_c[m] >= q[1]
        if np.any(low) and np.any(hi):
            print(f'    low-t_asm quartile (t<={q[0]:.2f}): '
                  f'median log(ICS/BCG)={np.median(logRatio[m][low]):+.2f}; '
                  f'high-t_asm (t>={q[1]:.2f}): '
                  f'median={np.median(logRatio[m][hi]):+.2f}; '
                  f'delta={np.median(logRatio[m][hi]) - np.median(logRatio[m][low]):+.2f} dex')

    fig, (ax_gc, ax_iso, ax_all) = plt.subplots(1, 3, sharey=True, figsize=(22, 6.25))

    x_lo = max(0.0, np.floor(np.percentile(t_asm_c, 1) * 2) / 2)
    x_hi = min(13.8, np.ceil(np.percentile(t_asm_c, 99) * 2) / 2)
    y_lo = np.floor(np.percentile(logRatio, 1) * 2) / 2
    y_hi = np.ceil(np.percentile(logRatio, 99) * 2) / 2
    x_bins = np.linspace(x_lo, x_hi, 70)
    y_bins = np.linspace(y_lo, y_hi, 70)
    bw = max(0.25, (x_hi - x_lo) / 20)
    mb = np.arange(x_lo, x_hi + bw, bw)

    def _one(ax, m, lbl, ls):
        if np.sum(m) < 20:
            return None
        cf = _contour_panel(ax, t_asm_c[m], logRatio[m], x_bins, y_bins)
        xm_, ym_ = _running_median(t_asm_c[m], logRatio[m], mb)
        if xm_.size:
            ax.plot(xm_, ym_, color='gold', lw=2.25, ls=ls, label=f'{lbl} median')
        ax.axhline(0, color='grey', ls=':', lw=1.2, alpha=0.7)
        ax.legend(loc='upper left', fontsize=10)
        return cf

    cf_gc = _one(ax_gc, gc, 'Groups/Clusters', '-')
    cf_iso = _one(ax_iso, iso, 'Isolated', '-.')
    cf_all = _one(ax_all, np.ones_like(gc, dtype=bool), 'Combined', '-')

    xlabel = r'$\langle t_{\mathrm{asm,ICS}} \rangle$  [Gyr lookback]'
    ylabel = r'$\log_{10}\, (M_{\mathrm{ICS}}\,/\,M_{\star,\mathrm{BCG}})$'
    for ax in (ax_gc, ax_iso, ax_all):
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel(xlabel)
    ax_gc.set_ylabel(ylabel)
    ax_gc.set_title('Groups/Clusters', fontsize=11)
    ax_iso.set_title('Isolated', fontsize=11)
    ax_all.set_title('Combined', fontsize=11)

    for cf, ax in [(cf_gc, ax_gc), (cf_iso, ax_iso), (cf_all, ax_all)]:
        if cf is not None:
            fig.colorbar(cf, ax=ax).set_label('Number of haloes')

    fig.tight_layout()
    outfile = os.path.join(output_dir,
                           f'ICS_over_BCG_vs_tasm_z{redshift:.1f}{OutputFormat}')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


# -----------------------------------------------------------------------------
# Plot 19: Recent-merger lookback vs t_asm (merger-timing proxy)
# -----------------------------------------------------------------------------

def plot_recent_merger_vs_tasm(d, sim_params, output_dir):
    """
    SAGE does not record a total merger count, but
        t_recent = min(TLMM, TLMm)
    is the lookback time to the most recent merger of either kind and is
    a proxy for merger-rate timing. At fixed M_vir, if t_asm tracks
    merger-driven deposition, then t_recent and t_asm should co-vary.

    Also visualise in the (t_recent, t_asm) plane with points coloured by
    which event won (major vs minor), since minor mergers are more
    frequent and tend to lead t_asm to smaller values.
    """
    redshift = d['redshift']
    min_stellar = sim_params['PartMass'] * sim_params['BaryonFrac']
    print(f'\nt_recent vs t_asm: z={redshift:.2f}...')

    t_asm = d.get('ICS_t_assembly_Gyr', None)
    tlmm = d.get('TimeOfLastMajorMerger_Gyr', None)
    tlmnr = d.get('TimeOfLastMinorMerger_Gyr', None)
    if t_asm is None or tlmm is None or tlmnr is None:
        print('  required fields missing; skipping.')
        return

    base_mask = ((d['Type'] == 0) & (d['Mvir'] > 0) & (d['ICS'] > 0)
                 & (d['StellarMass'] >= min_stellar)
                 & np.isfinite(t_asm) & (t_asm > 0))
    if np.sum(base_mask) < 50:
        print('  Too few haloes.')
        return

    t_asm_c = t_asm[base_mask]
    tlmm_c = tlmm[base_mask]
    tlmnr_c = tlmnr[base_mask]
    logMvir = np.log10(d['Mvir'][base_mask])

    # t_recent = min of the two where both finite; else the finite one
    t_recent = np.full(t_asm_c.size, np.nan)
    have_both = np.isfinite(tlmm_c) & np.isfinite(tlmnr_c)
    only_M = np.isfinite(tlmm_c) & ~np.isfinite(tlmnr_c)
    only_m = ~np.isfinite(tlmm_c) & np.isfinite(tlmnr_c)
    t_recent[have_both] = np.minimum(tlmm_c[have_both], tlmnr_c[have_both])
    t_recent[only_M] = tlmm_c[only_M]
    t_recent[only_m] = tlmnr_c[only_m]

    # which merger type was most recent?
    minor_most_recent = np.zeros(t_asm_c.size, dtype=bool)
    minor_most_recent[have_both] = tlmnr_c[have_both] < tlmm_c[have_both]
    minor_most_recent[only_m] = True

    ok = np.isfinite(t_recent)
    print(f'  N with >=1 recorded merger: {int(np.sum(ok))} / {ok.size}')
    print(f'  N minor-most-recent: {int(np.sum(minor_most_recent & ok))} '
          f'({100.0*np.mean(minor_most_recent[ok]):.1f}% of those with events)')

    print('  r(t_recent, t_asm) at fixed log Mvir:')
    for lo, hi in [(11.5, 12.5), (12.5, 13.5), (13.5, 15.5)]:
        mm = ok & (logMvir >= lo) & (logMvir < hi)
        if np.sum(mm) >= 20:
            rho = np.corrcoef(t_recent[mm], t_asm_c[mm])[0, 1]
            delta = t_asm_c[mm] - t_recent[mm]
            print(f'    logMvir in [{lo:.1f},{hi:.1f}) '
                  f'(N={int(np.sum(mm))}): r={rho:+.3f}, '
                  f'median t_recent={np.median(t_recent[mm]):.2f} Gyr, '
                  f'median delta(t_asm-t_recent)={np.median(delta):+.2f} Gyr')

    fig, (ax_s, ax_h) = plt.subplots(1, 2, figsize=(16.0, 6.25))

    t_lo = max(0.0, np.floor(np.percentile(t_asm_c, 1) * 2) / 2)
    t_hi = min(13.8, np.ceil(np.percentile(t_asm_c, 99) * 2) / 2)
    tr = t_recent[ok]
    r_lo_p = max(0.0, np.floor(np.percentile(tr, 1) * 2) / 2)
    r_hi_p = min(13.8, np.ceil(np.percentile(tr, 99) * 2) / 2)

    # --- Panel A: overlaid contours for minor/major most-recent merger ---
    m_minor = ok & minor_most_recent
    m_major = ok & ~minor_most_recent

    x_bins = np.linspace(r_lo_p, r_hi_p, 50)
    y_bins = np.linspace(t_lo, t_hi, 50)

    for mask, cmap, ls, lbl in [
            (m_minor, 'Blues', '-', 'minor most recent'),
            (m_major, 'Reds', '--', 'major most recent')]:
        if np.sum(mask) < 20:
            continue
        H, xe, ye = np.histogram2d(t_recent[mask], t_asm_c[mask],
                                   bins=[x_bins, y_bins])
        xc = 0.5 * (xe[:-1] + xe[1:])
        yc = 0.5 * (ye[:-1] + ye[1:])
        X, Y = np.meshgrid(xc, yc)
        Hm = np.where(H.T > 0, H.T, np.nan)
        ax_s.contourf(X, Y, Hm, levels=_int_levels(Hm), cmap=cmap, alpha=0.6)
        # running median per population
        bw_pop = max(0.5, (r_hi_p - r_lo_p) / 18)
        mb_pop = np.arange(r_lo_p, r_hi_p + bw_pop, bw_pop)
        xm_, ym_ = _running_median(t_recent[mask], t_asm_c[mask], mb_pop)
        if xm_.size:
            ax_s.plot(xm_, ym_, color='black', ls=ls, lw=2.5,
                      label=f'{lbl} (N={int(np.sum(mask))})')

    # 1:1 line
    ref_lo = max(r_lo_p, t_lo)
    ref_hi = min(r_hi_p, t_hi)
    ax_s.plot([ref_lo, ref_hi], [ref_lo, ref_hi], color='grey', ls=':',
              lw=1.5, alpha=0.7, label=r'$t_{\mathrm{asm}}=t_{\mathrm{recent}}$')
    ax_s.set_xlim(r_lo_p, r_hi_p)
    ax_s.set_ylim(t_lo, t_hi)
    ax_s.set_xlabel(r'$t_{\mathrm{recent}}=\min(t_{\mathrm{TLMM}},t_{\mathrm{TLMm}})$  [Gyr]')
    ax_s.set_ylabel(r'$\langle t_{\mathrm{asm,ICS}} \rangle$  [Gyr lookback]')
    ax_s.legend(loc='upper left', fontsize=10)

    # --- Panel B: histogram of delta = t_asm - t_recent, per Mvir bin ---
    delta_all = t_asm_c[ok] - t_recent[ok]
    lm_bins = [(11.5, 12.5, 'group'),
               (12.5, 13.5, 'group/cluster'),
               (13.5, 15.5, 'cluster')]
    colors = ['steelblue', 'darkorange', 'firebrick']
    hbins = np.linspace(np.floor(np.percentile(delta_all, 1)),
                        np.ceil(np.percentile(delta_all, 99)), 40)
    for (lo, hi, lbl), col in zip(lm_bins, colors):
        mm = ok & (logMvir >= lo) & (logMvir < hi)
        if np.sum(mm) >= 20:
            ax_h.hist(t_asm_c[mm] - t_recent[mm], bins=hbins, histtype='step',
                      lw=2.25, color=col, density=True,
                      label=rf'$\log M_{{\mathrm{{vir}}}}\in[{lo:.1f},{hi:.1f})$ ({lbl}, N={int(np.sum(mm))})')
            ax_h.axvline(np.median(t_asm_c[mm] - t_recent[mm]),
                         color=col, ls=':', lw=1.5, alpha=0.7)
    ax_h.axvline(0, color='grey', ls='--', lw=1.0, alpha=0.7)
    ax_h.set_xlabel(r'$\Delta = t_{\mathrm{asm,ICS}} - t_{\mathrm{recent}}$  [Gyr]')
    ax_h.set_ylabel('Density')
    ax_h.legend(loc='upper right', fontsize=9)

    fig.tight_layout()
    outfile = os.path.join(output_dir,
                           f'ICS_tasm_vs_t_recent_merger_z{redshift:.1f}{OutputFormat}')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


# -----------------------------------------------------------------------------
# Plot 13: f_disrupt grid scan — f_ICL and M_BCG vs M_vir
# -----------------------------------------------------------------------------

def _rescale_to_fdisrupt(d, f_new, f_run):
    """
    Post-processing rescaling of the disruption-channel mass between ICS
    and BCG for a hypothetical FractionDisruptedToICS = f_new, given that
    the simulation was run with FractionDisruptedToICS = f_run.

    We back out the cumulative disrupted mass per central from
        ICS_disrupt = f_run * M_disrupt_cum
    and redistribute it:
        ICS_new         = f_new * M_disrupt_cum + ICS_accrete
                        = (f_new / f_run) * ICS_disrupt + ICS_accrete
        StellarMass_new = StellarMass + (f_run - f_new) / f_run * ICS_disrupt

    This is first-order: it holds the total disrupted stellar mass fixed and
    only swaps its destination between ICS and BCG. It does NOT feed the
    change back into cooling, star formation, or subsequent merger history.
    """
    ICS_disrupt = d['ICS_disrupt']
    ICS_accrete = d['ICS_accrete']
    StellarMass = d['StellarMass']

    ICS_new = (f_new / f_run) * ICS_disrupt + ICS_accrete
    StellarMass_new = StellarMass + ((f_run - f_new) / f_run) * ICS_disrupt
    # Guard against a negative BCG mass (should not happen unless f_new > f_run
    # AND ICS_disrupt > StellarMass, which is non-physical)
    StellarMass_new = np.maximum(StellarMass_new, 0.0)
    return ICS_new, StellarMass_new


def _binned_median_percentiles(x, y, bin_edges, min_count=10):
    bin_cen = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    med = np.full(len(bin_cen), np.nan)
    p16 = np.full(len(bin_cen), np.nan)
    p84 = np.full(len(bin_cen), np.nan)
    counts = np.zeros(len(bin_cen), dtype=int)
    for i in range(len(bin_edges) - 1):
        m = (x >= bin_edges[i]) & (x < bin_edges[i+1])
        counts[i] = int(np.sum(m))
        if counts[i] >= min_count:
            med[i] = np.median(y[m])
            p16[i] = np.percentile(y[m], 16)
            p84[i] = np.percentile(y[m], 84)
    return bin_cen, med, p16, p84, counts


def plot_ics_fraction_fdisrupt_grid(d, sim_params, output_dir,
                                     f_grid=(0.3, 0.5, 0.7, 1.0)):
    """
    2-panel plot scanning FractionDisruptedToICS via post-processing rescaling
    of ICS_disrupt / ICS_accrete:
      Left : f_ICL = M_ICS / (M_ICS + M_BCG + M_sats) vs log M_vir, model curves
             for each f_disrupt value, overlaid with Contini (2021) obs.
      Right: M_BCG vs log M_vir for the same f_disrupt grid, to expose the
             BCG over-growth trade-off.
    """
    redshift = d['redshift']
    min_stellar = sim_params['PartMass'] * sim_params['BaryonFrac']
    f_run = sim_params.get('FractionDisruptedToICS', 0.7)
    print(f'\nf_disrupt grid scan vs observations: z={redshift:.2f}, '
          f'f_run={f_run:.2f}...')

    # --- Sum satellite StellarMass per central (invariant under f_disrupt) ---
    Type = d['Type']
    GalaxyIndex = d['GalaxyIndex']
    CentralGalaxyIndex = d['CentralGalaxyIndex']
    StellarMass = d['StellarMass']
    Mvir = d['Mvir']

    sorted_idx = np.argsort(GalaxyIndex)
    sorted_gids = GalaxyIndex[sorted_idx]
    sat_mask_field = Type != 0
    sat_central_gids = CentralGalaxyIndex[sat_mask_field]
    sat_sm = StellarMass[sat_mask_field]
    insert_pos = np.searchsorted(sorted_gids, sat_central_gids)
    insert_pos = np.clip(insert_pos, 0, len(sorted_gids) - 1)
    valid_match = sorted_gids[insert_pos] == sat_central_gids
    central_idx = np.where(valid_match, sorted_idx[insert_pos], -1)
    valid_sats = central_idx >= 0
    sat_sm_sum = np.zeros(len(Type), dtype=float)
    np.add.at(sat_sm_sum, central_idx[valid_sats], sat_sm[valid_sats])

    base_mask = ((Type == 0) & (Mvir > 0) & (StellarMass >= min_stellar))
    if np.sum(base_mask) < 20:
        print('  Too few haloes.')
        return

    logMvir_all = np.log10(Mvir[base_mask])
    sat_sm_sum_c = sat_sm_sum[base_mask]

    bin_edges = np.arange(11.0, 15.26, 0.25)

    # Observational ICL-fraction compilation (same set as plot 7)
    obs = [
        ('Zibetti+05',             14.30, 0.110, 0.030, 'o', 'firebrick'),
        ('Krick \\& Bernstein 07', 14.20, 0.120, 0.050, 's', 'darkorange'),
        ('Presotto+14',            15.10, 0.120, 0.020, 'v', 'seagreen'),
        ('Burke+15',               14.90, 0.030, 0.010, '^', 'navy'),
        ('Montes \\& Trujillo 18', 14.95, 0.150, 0.070, 'D', 'purple'),
        ('Furnell+21',             14.70, 0.240, 0.090, 'P', 'teal'),
        ('Kluge+21',               14.50, 0.300, 0.100, 'X', 'saddlebrown'),
    ]

    # Colour per f_disrupt (ordered low->high)
    cmap = plt.get_cmap('viridis')
    grid_colors = [cmap(i / max(1, len(f_grid) - 1)) for i in range(len(f_grid))]

    fig, (ax_f, ax_b) = plt.subplots(1, 2, figsize=(16, 7.0))

    for f_new, col in zip(f_grid, grid_colors):
        ICS_new, SM_new = _rescale_to_fdisrupt(d, f_new, f_run)

        SM_new_c = SM_new[base_mask]
        ICS_new_c = ICS_new[base_mask]
        M_halo_new = SM_new_c + sat_sm_sum_c
        denom = ICS_new_c + M_halo_new

        valid = (denom > 0) & (ICS_new_c >= 0)
        f_ICL_new = np.full_like(ICS_new_c, np.nan)
        f_ICL_new[valid] = ICS_new_c[valid] / denom[valid]

        v_mask = valid & np.isfinite(f_ICL_new)
        # --- f_ICL panel ---
        bc, med, p16, p84, _ = _binned_median_percentiles(
            logMvir_all[v_mask], f_ICL_new[v_mask], bin_edges)
        vmed = ~np.isnan(med)
        ax_f.fill_between(bc[vmed], p16[vmed], p84[vmed], color=col, alpha=0.12)
        ax_f.plot(bc[vmed], med[vmed], color=col, lw=2.4,
                  label=fr'$f_{{\mathrm{{disrupt}}}} = {f_new:.2f}$')

        # --- BCG stellar mass panel ---
        bcg_mask = (SM_new_c > 0)
        bc2, med2, p162, p842, _ = _binned_median_percentiles(
            logMvir_all[bcg_mask], np.log10(SM_new_c[bcg_mask]), bin_edges)
        v2 = ~np.isnan(med2)
        ax_b.fill_between(bc2[v2], p162[v2], p842[v2], color=col, alpha=0.12)
        ax_b.plot(bc2[v2], med2[v2], color=col, lw=2.4,
                  label=fr'$f_{{\mathrm{{disrupt}}}} = {f_new:.2f}$')

        # --- stats ---
        print(f'  f_disrupt = {f_new:.2f}:')
        for lm_target in [12.0, 13.0, 13.5, 14.0, 14.5]:
            m = (logMvir_all >= lm_target - 0.125) & (logMvir_all < lm_target + 0.125) & v_mask
            if np.sum(m) >= 10:
                print(f'    logMvir~{lm_target:.1f} (N={int(np.sum(m))}): '
                      f'median f_ICL={np.median(f_ICL_new[m]):.3f}, '
                      f'median log M_BCG={np.median(np.log10(SM_new_c[m])):.3f}')

    # --- overlay observations on f_ICL panel ---
    for (lbl, logM, f, err, mk, cl) in obs:
        ax_f.errorbar([logM], [f], yerr=[[err], [err]], marker=mk, ms=9,
                      color=cl, mfc='white', mec=cl, mew=1.6, capsize=3,
                      elinewidth=1.3, label=lbl, ls='none')

    ax_f.set_xlabel(r'$\log_{10}\, M_{\mathrm{vir}}\ [\mathrm{M}_{\odot}]$')
    ax_f.set_ylabel(r'$M_{\mathrm{ICS}}\,/\,(M_{\mathrm{ICS}} + M_{\star,\mathrm{BCG}} + M_{\star,\mathrm{sats}})$')
    ax_f.set_xlim(11.5, 15.3)
    ax_f.set_ylim(0.0, 0.75)
    ax_f.legend(loc='upper left', fontsize=8, ncol=2, framealpha=0.9)
    ax_f.text(0.97, 0.03,
              fr'run: $f_{{\mathrm{{disrupt}}}}^{{\mathrm{{run}}}} = {f_run:.2f}$',
              transform=ax_f.transAxes, va='bottom', ha='right', fontsize=10,
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor='grey', alpha=0.8))

    ax_b.set_xlabel(r'$\log_{10}\, M_{\mathrm{vir}}\ [\mathrm{M}_{\odot}]$')
    ax_b.set_ylabel(r'$\log_{10}\, M_{\star,\mathrm{BCG}}\ [\mathrm{M}_{\odot}]$')
    ax_b.set_xlim(11.5, 15.3)
    ax_b.legend(loc='upper left', fontsize=9, framealpha=0.9)

    fig.tight_layout()
    outfile = os.path.join(output_dir,
                           f'ICS_fraction_fdisrupt_grid_z{redshift:.1f}{OutputFormat}')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


# -----------------------------------------------------------------------------
# Multi-snapshot helpers
# -----------------------------------------------------------------------------

def _load_snap_fields(file_list, snap_num, hubble_h, mass_fields, other_fields):
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


def _compute_satellite_sums(Type, StellarMass, GalaxyIndex, CentralGalaxyIndex):
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


def _observational_ics_data():
    """Return observational ICL fraction data points (from multiple surveys)."""
    obs = []
    # Burke, Collins, Stott and Hilton 2012
    obs.append(([0.947, 0.830, 0.795, 0.808, 1.223],
                [1.42, 2.59, 3.77, 1.53, 2.36]))
    # Montes & Trujillo 2018 (gray circles)
    obs.append(([0.534, 0.544, 0.367, 0.397, 0.342, 0.304, 0.048],
                [1.53, 0.0, 1.06, 1.53, 2.71, 3.30, 8.61]))
    # Burke, Hilton & Collins 2015
    obs.append(([0.403, 0.387, 0.397, 0.339, 0.344, 0.342, 0.291, 0.225,
                 0.218, 0.213, 0.195, 0.177],
                [2.59, 2.71, 3.30, 5.54, 6.01, 7.19, 12.97, 12.50,
                 16.27, 18.04, 16.86, 23.11]))
    # Furnell et al. 2021
    obs.append(([0.144, 0.127, 0.122, 0.081, 0.225, 0.215, 0.256, 0.306,
                 0.261, 0.294, 0.322, 0.342, 0.372, 0.337, 0.377, 0.329,
                 0.496, 0.425, 0.109],
                [38.56, 30.66, 31.01, 28.89, 26.53, 23.58, 28.54, 29.72,
                 32.55, 27.48, 27.59, 26.65, 19.81, 18.87, 15.45, 15.33,
                 11.32, 9.67, 31.60]))
    # Feldmeier et al. 2004
    obs.append(([0.162, 0.162, 0.162, 0.185],
                [15.21, 12.15, 10.26, 7.31]))
    # Montes & Trujillo 2018 (black circles)
    obs.append(([0.301, 0.390, 0.342, 0.537, 0.537, 0.370, 0.043],
                [7.67, 8.61, 13.09, 6.60, 5.78, 4.83, 10.85]))
    # Ko & Jee 2018
    obs.append(([1.238], [9.91]))
    # Kluge et al. 2021
    obs.append(([0.030], [17.92]))
    # Zibetti et al. 2005
    obs.append(([0.243], [10.85]))
    # Presotto et al. 2014 (two estimates)
    obs.append(([0.435, 0.433], [12.26, 5.54]))
    # Spavone et al. 2020
    obs.append(([0.0], [34.08]))
    # JWST XLSSC 122 z=1.98
    obs.append(([1.98], [17.0]))
    # Ragusa et al. 2023 (VEGAS groups)
    obs.append(([0.05]*16,
                [16, 5, 5, 17, 5, 27, 34, 17, 8, 35, 18, 7, 20, 22, 28, 30]))
    # Ahad et al. 2025 KiDS+GAMA groups
    obs.append(([0.12, 0.12, 0.12, 0.18, 0.18, 0.18, 0.24, 0.24, 0.24],
                [16, 10, 4, 15, 12, 8, 13, 15, 5]))
    # Collect as arrays (fractions 0-1)
    all_z, all_f = [], []
    for zz, ff in obs:
        all_z.extend(zz)
        all_f.extend([v / 100.0 if v > 1.0 else v for v in ff])
    return np.array(all_z), np.array(all_f)


# -----------------------------------------------------------------------------
# Plot 21: f_ICS vs redshift — two definitions side-by-side
# -----------------------------------------------------------------------------

def plot_ics_fraction_vs_redshift(file_list, sim_params, output_dir,
                                  max_redshift=2.5):
    """
    Side-by-side panels:
      Left:  f_ICS = M_ICS / (f_b * M_vir)  — baryon-budget definition
      Right: f_ICS = M_ICS / (M_BCG + M_sats + M_ICS) — stellar-fraction

    Median + 16/84 bands for clusters (Mvir >= 1e13) and groups
    (10^12.5 <= Mvir < 1e13), each requiring >= 2 satellites.
    Observational data overlaid from multiple surveys.
    """
    print(f'\nf_ICS vs redshift (two definitions): z <= {max_redshift}...')
    h_ = sim_params['Hubble_h']
    redshifts = sim_params['redshifts']
    avail = sim_params['available_snapshots']
    f_b = sim_params['BaryonFrac']
    min_stellar = sim_params['PartMass'] * sim_params['BaryonFrac']

    snaps_to_plot = sorted([s for s in avail
                            if s < len(redshifts) and redshifts[s] <= max_redshift],
                           reverse=True)
    if not snaps_to_plot:
        print('  No snapshots in range.')
        return

    # Thresholds (matching BCG_ICS_fraction.py)
    MVIR_CL = 1.0e13
    MVIR_GR = 10**12.5
    MIN_SAT = 2

    # Accumulators keyed by 'cl' (cluster) / 'gr' (group)
    # Each definition stored separately
    cl_z, cl_baryon, cl_stellar = [], [], []
    gr_z, gr_baryon, gr_stellar = [], [], []

    for snap in snaps_to_plot:
        z = redshifts[snap]
        d = _load_snap_fields(file_list, snap, h_,
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

        for _label, z_list, bar_list, stel_list, mvir_lo, mvir_hi in [
                ('cl', cl_z, cl_baryon, cl_stellar, MVIR_CL, 1e17),
                ('gr', gr_z, gr_baryon, gr_stellar, MVIR_GR, MVIR_CL)]:
            mask = ((Type == 0) & (Mvir >= mvir_lo) & (Mvir < mvir_hi)
                    & (ICS > 0) & (SM >= min_stellar) & (n_sat >= MIN_SAT))
            sel = np.where(mask)[0]
            if len(sel) == 0:
                continue
            z_list.append(z)
            fb = ICS[sel] / (f_b * Mvir[sel])
            total_stellar = SM[sel] + sm_sat[sel] + ICS[sel]
            fs = ICS[sel] / total_stellar
            bar_list.append((np.median(fb),
                             np.percentile(fb, 16), np.percentile(fb, 84)))
            stel_list.append((np.median(fs),
                              np.percentile(fs, 16), np.percentile(fs, 84)))

    # --- Figure ---
    fig, (ax_b, ax_s) = plt.subplots(1, 2, figsize=(18, 6.5))
    obs_z, obs_f = _observational_ics_data()

    for ax, data_cl, data_gr, ylabel in [
            (ax_b, cl_baryon, gr_baryon,
             r'$f_{\mathrm{ICS}} = M_{\mathrm{ICS}}\,/\,(f_b \times M_{\mathrm{vir}})$'),
            (ax_s, cl_stellar, gr_stellar,
             r'$f_{\mathrm{ICS}} = M_{\mathrm{ICS}}\,/\,(M_{\star,\mathrm{BCG}} + M_{\star,\mathrm{sat}} + M_{\mathrm{ICS}})$')]:

        # Observations
        ax.scatter(obs_z, obs_f, marker='x', color='k', s=40, alpha=0.6,
                   zorder=5, label='Observations (compilation)')

        # Clusters
        if data_cl:
            zz = np.array(cl_z)
            med = np.array([v[0] for v in data_cl])
            lo = np.array([v[1] for v in data_cl])
            hi = np.array([v[2] for v in data_cl])
            ax.plot(zz, med, '-', color='dodgerblue', lw=2,
                    label=r'Clusters ($M_{\mathrm{vir}} \geq 10^{13}$)')
            ax.fill_between(zz, lo, hi, color='dodgerblue', alpha=0.15)
        # Groups
        if data_gr:
            zz = np.array(gr_z)
            med = np.array([v[0] for v in data_gr])
            lo = np.array([v[1] for v in data_gr])
            hi = np.array([v[2] for v in data_gr])
            ax.plot(zz, med, '--', color='green', lw=2,
                    label=r'Groups ($10^{12.5} \leq M_{\mathrm{vir}} < 10^{13}$)')
            ax.fill_between(zz, lo, hi, color='green', alpha=0.15)

        ax.set_xlim(0, max_redshift)
        ax.set_ylim(0, None)
        ax.set_xlabel(r'Redshift $z$')
        ax.set_ylabel(ylabel)
        ax.legend(loc='upper right', fontsize=9)

    fig.tight_layout()
    outfile = os.path.join(output_dir,
                           f'ICS_fraction_vs_redshift_sidebyside{OutputFormat}')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


# -----------------------------------------------------------------------------
# Plot 22: ICS mass function grid (multi-redshift)
# -----------------------------------------------------------------------------

def plot_ics_mass_function_grid(file_list, sim_params, output_dir):
    """
    4x2 grid of ICS mass functions at target redshifts (z = 0, 0.5, 1, 2,
    3, 4, 5, 6), each panel split by halo mass bin.
    """
    print('\nICS mass function grid...')
    h_ = sim_params['Hubble_h']
    redshifts = sim_params['redshifts']
    avail = sim_params['available_snapshots']
    box = sim_params['BoxSize']
    volfrac = sim_params['VolumeFraction']
    volume = (box / h_)**3.0 * volfrac
    min_stellar = sim_params['PartMass'] * sim_params['BaryonFrac']

    target_z = [0, 0.5, 1, 2, 3, 4, 5, 6]
    snap_list = []
    for zt in target_z:
        diffs = [(abs(redshifts[s] - zt), s) for s in avail
                 if s < len(redshifts)]
        if diffs:
            snap_list.append(min(diffs)[1])
    snap_list = list(dict.fromkeys(snap_list))  # remove dupes, keep order

    # Halo mass bin edges (log10 Msun)
    halo_bins = [(10.5, 12.0, 'firebrick'),
                 (12.0, 13.5, 'green'),
                 (13.5, 17.0, 'dodgerblue')]
    halo_labels = [r'$10^{10.5} < M_{\mathrm{vir}} < 10^{12}$',
                   r'$10^{12} < M_{\mathrm{vir}} < 10^{13.5}$',
                   r'$M_{\mathrm{vir}} > 10^{13.5}$']

    mi, ma, bw = 4.5, 12.75, 0.25
    NB = int((ma - mi) / bw)

    n_panels = min(len(snap_list), 8)
    fig, axes = plt.subplots(4, 2, figsize=(12, 18))
    axes = axes.flatten()

    for idx in range(n_panels):
        snap = snap_list[idx]
        z = redshifts[snap]
        ax = axes[idx]

        d = _load_snap_fields(file_list, snap, h_,
                              mass_fields=['Mvir', 'IntraClusterStars', 'StellarMass'],
                              other_fields=['Type', 'GalaxyIndex', 'CentralGalaxyIndex'])
        if d['Mvir'].size == 0:
            ax.text(0.5, 0.5, f'z = {z:.2f}\nNo data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14)
            continue

        n_sat, _ = _compute_satellite_sums(
            d['Type'], d['StellarMass'], d['GalaxyIndex'], d['CentralGalaxyIndex'])
        w = ((d['Type'] == 0) & (d['IntraClusterStars'] > 0) & (d['Mvir'] >= 1e10)
             & (d['StellarMass'] >= min_stellar) & (n_sat >= 2))
        logMvir = np.log10(d['Mvir'][w])
        logICS = np.log10(d['IntraClusterStars'][w])

        for (lo, hi, col) in halo_bins:
            m = (logMvir >= lo) & (logMvir < hi)
            if np.sum(m) > 0:
                counts, edges = np.histogram(logICS[m], range=(mi, ma), bins=NB)
                xc = edges[:-1] + 0.5 * bw
                phi = counts / volume / bw
                ax.plot(xc, phi, color=col, alpha=0.7)
                ax.fill_between(xc, 0, phi, color=col, alpha=0.1)

        # Overall
        counts, edges = np.histogram(logICS, range=(mi, ma), bins=NB)
        xc = edges[:-1] + 0.5 * bw
        phi = counts / volume / bw
        ax.plot(xc, phi, color='black')

        ax.set_yscale('log')
        ax.set_xlim(4.5, 12.0)
        ax.set_ylim(1e-8, 1e-1)
        ax.text(0.95, 0.95, f'z = {z:.2f}', transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top', ha='right')
        if idx % 2 == 0:
            ax.set_ylabel(r'$\phi\ (\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1})$',
                          fontsize=11)

    for idx in range(n_panels, 8):
        axes[idx].set_visible(False)
    for idx in [6, 7]:
        if idx < n_panels:
            axes[idx].set_xlabel(r'$\log_{10}\, M_{\mathrm{ICS}}\ [\mathrm{M}_{\odot}]$',
                                 fontsize=12)

    from matplotlib.lines import Line2D
    custom = [Line2D([0], [0], color=c, lw=2) for _, _, c in halo_bins]
    custom.append(Line2D([0], [0], color='black', lw=2))
    fig.legend(custom, halo_labels + ['Overall'], loc='lower center',
               ncol=4, fontsize=10, frameon=False)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.05)
    outfile = os.path.join(output_dir, f'ICS_mass_function_grid{OutputFormat}')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


# -----------------------------------------------------------------------------
# Plot 23: ICS assembly history grid (disruption blue, accretion green)
# -----------------------------------------------------------------------------

def plot_ics_assembly_history_grid(file_list, sim_params, output_dir,
                                   n_max_per_bin=7500):
    """
    Grid of panels (one per halo-mass bin at z=0) showing the build-up of
    ICS as a fraction of its z=0 value, split into the two channels:
      - ICS_disrupt (blue) — stripped from satellites in-halo
      - ICS_accrete (green) — pre-formed ICS accreted with infalling groups

    Each galaxy's ICS_disrupt(z)/ICS_disrupt(z=0) and
    ICS_accrete(z)/ICS_accrete(z=0) are tracked across all snapshots using
    GalaxyIndex matching. Median + 15/85 percentile bands shown.
    """
    print('\nICS assembly history grid...')
    h_ = sim_params['Hubble_h']
    MASS_CONVERT = 1.0e10 / h_

    mass_bins = [
        (1e10, 1e11, r'$10^{10} < M_{\mathrm{vir}} < 10^{11}$'),
        (1e11, 1e12, r'$10^{11} < M_{\mathrm{vir}} < 10^{12}$'),
        (1e12, 1e13, r'$10^{12} < M_{\mathrm{vir}} < 10^{13}$'),
        (1e13, 1e14, r'$10^{13} < M_{\mathrm{vir}} < 10^{14}$'),
        (1e14, 1e15, r'$10^{14} < M_{\mathrm{vir}} < 10^{15}$'),
        (1e15, 1e17, r'$M_{\mathrm{vir}} > 10^{15}$'),
    ]

    filepath = file_list[0]
    f = h5.File(filepath, 'r')

    last_snap = sim_params['last_snapshot']
    snap_key_z0 = f'Snap_{last_snap}'

    # Select z=0 halos per mass bin
    mvir_z0 = f[snap_key_z0]['Mvir'][:] * MASS_CONVERT
    type_z0 = f[snap_key_z0]['Type'][:]
    gal_idx_z0 = f[snap_key_z0]['GalaxyIndex'][:]
    ics_z0_raw = f[snap_key_z0]['IntraClusterStars'][:] * MASS_CONVERT
    disrupt_z0_raw = f[snap_key_z0]['ICS_disrupt'][:] * MASS_CONVERT
    accrete_z0_raw = f[snap_key_z0]['ICS_accrete'][:] * MASS_CONVERT

    n_cols = 3
    n_rows = (len(mass_bins) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows),
                             sharex=True, sharey=True)
    axes = axes.flatten()

    snaps_to_track = [s for s in range(1, last_snap + 1)]
    lookback_all = sim_params['lookback_gyr']

    for i, (m_lo, m_hi, label) in enumerate(mass_bins):
        ax = axes[i]
        mask = ((type_z0 == 0) & (mvir_z0 >= m_lo) & (mvir_z0 < m_hi)
                & (ics_z0_raw > 0))
        sel = np.where(mask)[0]
        if len(sel) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)
            ax.set_title(label, fontsize=11)
            continue

        rng = np.random.default_rng(42)
        if len(sel) > n_max_per_bin:
            sel = rng.choice(sel, size=n_max_per_bin, replace=False)
        n_sel = len(sel)
        print(f'  {label}: N={n_sel}')

        gids = gal_idx_z0[sel]
        disrupt_z0_safe = np.maximum(disrupt_z0_raw[sel], 1e-10)
        accrete_z0_safe = np.maximum(accrete_z0_raw[sel], 1e-10)
        gid_to_idx = {gid: j for j, gid in enumerate(gids)}

        n_snaps = len(snaps_to_track)
        disrupt_frac = np.full((n_sel, n_snaps), np.nan)
        accrete_frac = np.full((n_sel, n_snaps), np.nan)

        for si, snap in enumerate(snaps_to_track):
            sk = f'Snap_{snap}'
            if sk not in f:
                continue
            gids_snap = f[sk]['GalaxyIndex'][:]
            hmask = np.isin(gids_snap, gids)
            if not np.any(hmask):
                continue
            h_arr = np.where(hmask)[0]
            h_gids = gids_snap[h_arr]
            h_out = np.array([gid_to_idx[g] for g in h_gids])

            ds = f[sk]['ICS_disrupt'][h_arr] * MASS_CONVERT
            ac = f[sk]['ICS_accrete'][h_arr] * MASS_CONVERT
            disrupt_frac[h_out, si] = ds / disrupt_z0_safe[h_out]
            accrete_frac[h_out, si] = ac / accrete_z0_safe[h_out]

        lb = np.array([lookback_all[s] for s in snaps_to_track])

        d_med = np.nanmedian(disrupt_frac, axis=0)
        d_15 = np.nanpercentile(disrupt_frac, 15, axis=0)
        d_85 = np.nanpercentile(disrupt_frac, 85, axis=0)
        a_med = np.nanmedian(accrete_frac, axis=0)
        a_15 = np.nanpercentile(accrete_frac, 15, axis=0)
        a_85 = np.nanpercentile(accrete_frac, 85, axis=0)

        ax.fill_between(lb, d_15, d_85, color='#2166AC', alpha=0.2,
                        edgecolor='none')
        ax.plot(lb, d_med, '-', color='#2166AC', lw=2, label='Disruption')
        ax.fill_between(lb, a_15, a_85, color='#1B7837', alpha=0.2,
                        edgecolor='none')
        ax.plot(lb, a_med, '--', color='#1B7837', lw=2, label='Accretion')

        ax.set_title(label, fontsize=11)
        ax.set_xlim(0, 13)
        ax.set_ylim(0, 1.05)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    axes[0].legend(loc='upper right', frameon=False, fontsize=10)
    fig.text(0.5, 0.02, 'Lookback time [Gyr]', ha='center', fontsize=14)
    fig.text(0.02, 0.5, r'Fraction of $z=0$ ICS', va='center',
             rotation='vertical', fontsize=14)

    plt.tight_layout(rect=[0.03, 0.03, 1, 0.97])
    outfile = os.path.join(output_dir,
                           f'ICS_assembly_history_grid{OutputFormat}')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    f.close()
    print(f'  Saved: {outfile}')


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_arguments():
    p = argparse.ArgumentParser(description='ICS assembly-history diagnostics from SAGE26 output')
    p.add_argument('input_pattern', nargs='?',
                   default='./output/millennium/model_*.hdf5',
                   help='Path pattern to model HDF5 files')
    p.add_argument('-s', '--snapshot', type=int, default=None,
                   help='Single snapshot number (default: latest available)')
    p.add_argument('-z', '--redshifts', type=float, nargs='+', default=None,
                   help='Target redshifts (e.g. -z 0 0.5 1 2); nearest available '
                        'snapshot is used for each. Overrides --snapshot.')
    p.add_argument('-o', '--output-dir', type=str, default=None,
                   help='Output directory for plots (default: <input_dir>/plots/)')
    return p.parse_args()


def _snap_for_redshift(sim_params, z_target):
    """Return snapshot number whose redshift is closest to z_target."""
    redshifts = sim_params['redshifts']
    out_snaps = sim_params['output_snapshots']
    if len(out_snaps) > 0:
        # restrict to snapshots that were actually written
        candidates = np.asarray(out_snaps, dtype=int)
    else:
        candidates = np.arange(len(redshifts))
    idx = int(candidates[np.argmin(np.abs(redshifts[candidates] - z_target))])
    return idx


def extract_tasm_summary(d, sim_params):
    """
    Extract (logMvir, t_asm_Gyr) arrays for central galaxies with ICS > 0,
    for use in multi-redshift overlays. Returns None if no t_asm info.
    """
    min_stellar = sim_params['PartMass'] * sim_params['BaryonFrac']
    t_asm = d.get('ICS_t_assembly_Gyr', None)
    if t_asm is None or np.all(np.isnan(t_asm)):
        return None
    m = ((d['Type'] == 0) & (d['Mvir'] > 0) & (d['ICS'] > 0)
         & (d['StellarMass'] >= min_stellar)
         & np.isfinite(t_asm) & (t_asm > 0))
    if np.sum(m) < 20:
        return None
    return {
        'logMvir': np.log10(d['Mvir'][m]),
        't_asm': t_asm[m],
        'redshift': d['redshift'],
    }


def plot_ics_assembly_time_multiz(snap_summaries, output_dir):
    """
    Overlay running medians of t_asm(logMvir) for each snapshot (redshift)
    on a single figure. Probes how the ICS assembly clock shifts with
    cosmic time.
    """
    if not snap_summaries:
        print('\nMulti-z t_asm overlay: no data to plot.')
        return

    z_strs_local = ['{:.2f}'.format(s['redshift']) for s in snap_summaries]
    print(f'\nMulti-z t_asm(logMvir) overlay: z = [{", ".join(z_strs_local)}]')

    fig, ax = plt.subplots(1, 1, figsize=(10.0, 7.0))

    # Sort by ascending redshift so colour runs z=0 -> high-z
    ordered = sorted(snap_summaries, key=lambda s: s['redshift'])
    cmap = plt.get_cmap('viridis')
    cols = cmap(np.linspace(0.15, 0.85, max(len(ordered), 2)))

    all_logM = np.concatenate([s['logMvir'] for s in ordered])
    all_t = np.concatenate([s['t_asm'] for s in ordered])
    xM_lo = np.floor(all_logM.min() * 2) / 2
    xM_hi = np.ceil(all_logM.max() * 2) / 2
    y_lo = max(0.0, np.floor(np.percentile(all_t, 1) * 2) / 2)
    y_hi = np.ceil(np.percentile(all_t, 99) * 2) / 2

    bw = max(0.1, (xM_hi - xM_lo) / 20)
    mb = np.arange(xM_lo, xM_hi + bw, bw)

    for s, col in zip(ordered, cols):
        bcen = 0.5 * (mb[:-1] + mb[1:])
        med = np.full(len(bcen), np.nan)
        p16 = np.full(len(bcen), np.nan)
        p84 = np.full(len(bcen), np.nan)
        for i in range(len(mb) - 1):
            m = (s['logMvir'] >= mb[i]) & (s['logMvir'] < mb[i+1])
            if np.sum(m) >= 10:
                med[i] = np.median(s['t_asm'][m])
                p16[i] = np.percentile(s['t_asm'][m], 16)
                p84[i] = np.percentile(s['t_asm'][m], 84)
        v = ~np.isnan(med)
        if v.any():
            ax.fill_between(bcen[v], p16[v], p84[v], color=col, alpha=0.15)
            ax.plot(bcen[v], med[v], color=col, lw=2.6,
                    label=fr'$z = {s["redshift"]:.2f}$')

            # print summary
            n_tot = s['logMvir'].size
            med_all = np.median(s['t_asm'])
            print(f'    z={s["redshift"]:.2f}: N={n_tot}, median t_asm={med_all:.2f} Gyr, '
                  f'range={np.percentile(s["t_asm"],16):.2f}-{np.percentile(s["t_asm"],84):.2f} Gyr (16-84)')

    ax.set_xlim(xM_lo, xM_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel(r'$\log_{10}\, M_{\mathrm{vir}}\ [\mathrm{M}_{\odot}]$')
    ax.set_ylabel(r'$\langle t_{\mathrm{assembly,ICS}} \rangle$  [Gyr lookback]')
    ax.legend(loc='upper left', fontsize=11, title='snapshot')
    fig.tight_layout()
    outfile = os.path.join(output_dir, f'ICS_assembly_time_vs_Mvir_multiz{OutputFormat}')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outfile}')


def run_all_plots(d, sim_params, output_dir):
    """Run every diagnostic plot for a single loaded snapshot dataset."""
    t50, _ = plot_formation_time_distributions(d, sim_params, output_dir)
    plot_ics_residual_vs_t50(d, sim_params, output_dir, t50=t50)
    plot_ics_vs_sfh_bulge(d, sim_params, output_dir)
    plot_ics_residual_vs_merger_bulge_residual(d, sim_params, output_dir)
    plot_ics_metallicity_vs_t50(d, sim_params, output_dir, t50=t50)
    plot_ics_fraction_vs_contini_obs(d, sim_params, output_dir)
    plot_zics_minus_zbcg(d, sim_params, output_dir, t50=t50)
    plot_ics_mzr(d, sim_params, output_dir, t50=t50)
    plot_ics_assembly_time(d, sim_params, output_dir, t50=t50)
    plot_ics_mzr_by_tasm(d, sim_params, output_dir)
    plot_ficl_colored_by_tasm(d, sim_params, output_dir)
    plot_dz_vs_tasm(d, sim_params, output_dir)
    plot_concentration_vs_tasm(d, sim_params, output_dir)
    plot_tlmm_vs_tasm(d, sim_params, output_dir)
    plot_bcg_ics_residual_correlation(d, sim_params, output_dir)
    plot_ics_bcg_ratio_vs_tasm(d, sim_params, output_dir)
    plot_recent_merger_vs_tasm(d, sim_params, output_dir)


if __name__ == '__main__':
    args = parse_arguments()

    file_list = sorted(glob.glob(args.input_pattern))
    if not file_list:
        print(f'Error: no files match {args.input_pattern}')
        sys.exit(1)
    print(f'Found {len(file_list)} model files.')

    sim_params = read_simulation_params(file_list[0])

    # Resolve list of snapshots to process
    if args.redshifts is not None:
        snap_list = sorted({_snap_for_redshift(sim_params, z) for z in args.redshifts})
    elif args.snapshot is not None:
        snap_list = [args.snapshot]
    else:
        snap_list = [sim_params['last_snapshot']]

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(file_list[0])), 'plots')
    os.makedirs(output_dir, exist_ok=True)

    print(f'Particle mass: {sim_params["PartMass"]:.4e} Msun')
    print(f'Stellar threshold: {sim_params["PartMass"]*sim_params["BaryonFrac"]:.4e} Msun')
    print(f'FractionDisruptedToICS: {sim_params["FractionDisruptedToICS"]:.3f}')
    print(f'Output directory: {output_dir}')
    z_list = [sim_params['redshifts'][s] for s in snap_list]
    z_strs = ['{:.2f}'.format(z) for z in z_list]
    print(f'Snapshots to process: {snap_list}  (z = [{", ".join(z_strs)}])')

    tasm_summaries = []
    for snap_num in snap_list:
        z_here = sim_params['redshifts'][snap_num]
        print('\n' + '=' * 78)
        print(f'=== Processing snapshot {snap_num}  (z = {z_here:.3f}) ===')
        print('=' * 78)

        d = load_assembly_data(file_list, sim_params, snap_num)
        if d is None:
            print(f'  No data loaded for snap {snap_num}. Skipping.')
            continue
        run_all_plots(d, sim_params, output_dir)
        # collect t_asm summary for multi-z overlay before discarding d
        summary = extract_tasm_summary(d, sim_params)
        if summary is not None:
            tasm_summaries.append(summary)
        # release memory before next snapshot
        del d

    if len(tasm_summaries) >= 2:
        print('\n' + '=' * 78)
        print('=== Multi-redshift overlay ===')
        print('=' * 78)
        plot_ics_assembly_time_multiz(tasm_summaries, output_dir)

    # --- Multi-snapshot plots (need all files + sim_params) ---
    print('\n' + '=' * 78)
    print('=== Multi-snapshot plots ===')
    print('=' * 78)

    print('\n--- Plot 21: f_ICS vs redshift (side-by-side) ---')
    plot_ics_fraction_vs_redshift(file_list, sim_params, output_dir)

    print('\n--- Plot 22: ICS mass function grid ---')
    plot_ics_mass_function_grid(file_list, sim_params, output_dir)

    print('\n--- Plot 23: ICS assembly history grid ---')
    plot_ics_assembly_history_grid(file_list, sim_params, output_dir)

    print('\nDone.')

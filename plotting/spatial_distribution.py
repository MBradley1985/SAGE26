#!/usr/bin/env python3
"""
Plot the spatial distribution of galaxies and dark matter halos at z~3.
Produces six figures: XY, XZ, and YZ projections for galaxies and for halos.

Usage:
    python spatial_distribution.py <output_folder>
    python spatial_distribution.py output/millennium/
    python spatial_distribution.py output/millennium/ --redshift 3.0
    python spatial_distribution.py output/millennium/ --snapshot 27
    python spatial_distribution.py output/millennium/ --tree-file input/millennium/trees/trees_STC.hdf5
"""

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
import glob
import os
import sys
from scipy.integrate import quad


def truncated_cmap(cmap, minval=0.0, maxval=1.0, n=256):
    return mcolors.LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(minval, maxval, n))
    )


# Quiescence definitions used throughout the script.
#   'strict'        -> SFR == 0 exactly (no current star formation at all);
#                      can be tightened with quench_snaps > 1 to require zero SFH
#                      in the preceding (quench_snaps-1) bins.
#   'loose'         -> sSFR < 1e-11 /yr (instantaneous; standard SAGE/observational
#                      passive cut). quench_snaps does not apply.
#   'loose_massive' -> 'loose' AND M* > 1e10 Msun. Useful for isolating the
#                      observationally-comparable high-z quenched-massive population.
#   'massive_bt'         -> SFR == 0 AND M* >= 1e11 Msun AND 0.75 <= B/T <= 0.90.
#   'loose_massive_bt'   -> sSFR < 1e-11 /yr AND M* >= 1e11 Msun AND 0.75 <= B/T <= 0.90.
#   'karls_galaxy'       -> sSFR < 1e-11 /yr AND M* >= 1e11 Msun AND 0.79 <= B/T <= 0.81 AND central only.
QUIESCENT_MODES = {
    'strict':           'SFR == 0',
    'loose':            'sSFR < 1e-11 /yr',
    'loose_massive':    'sSFR < 1e-11 /yr AND M* > 1e10 Msun',
    'hubble':           'sSFR < 0.2 / t_H(z) AND M* > 1e10 Msun AND 0.75 <= B/T <= 0.90  (snapshot Hubble time)',
    'hubble_central':   'hubble AND central only (Type==0)',
    'massive_bt':       'M* >= 1e11 Msun, SFR == 0, 0.75 <= B/T <= 0.90',
    'loose_massive_bt': 'sSFR < 1e-11 /yr AND M* >= 1e11 Msun AND 0.75 <= B/T <= 0.90',
    'karls_galaxy':     'sSFR < 1e-11 /yr AND M* >= 1e11 Msun AND 0.79 <= B/T <= 0.81 AND central',
}

MASSIVE_FLOOR_MSUN = 1.0e10       # used by 'loose_massive'
MASSIVE_BT_FLOOR_MSUN = 1.0e10    # used by 'massive_bt'
LOOSE_MASSIVE_BT_FLOOR_MSUN = 1.0e10  # used by 'loose_massive_bt'
MASSIVE_BT_MIN = 0.750            # B/T lower bound for 'massive_bt'
MASSIVE_BT_MAX = 0.900            # B/T upper bound for 'massive_bt'
KARLS_GALAXY_SM_FLOOR  = 1.0e10   # M* > 1e11 Msun
KARLS_GALAXY_BT_MIN    = 0.750
KARLS_GALAXY_BT_MAX    = 0.900

# Galaxy count above which per-point alpha is scaled with log10(M*) rather than
# using a flat alpha.  microUchuu (local test run) sits well below this.
LARGE_SIM_GALAXY_THRESHOLD = 500_000

# Quiescence threshold factor for the 'hubble' mode:
#   quiescent if sSFR < HUBBLE_QUIESCENCE_FACTOR / t_H(z)
# where t_H(z) = 1/H(z) is the Hubble time at the snapshot redshift.
HUBBLE_QUIESCENCE_FACTOR = 0.2


def hubble_time_gyr(redshift, omega_m, omega_l, hubble_h):
    """Return the Hubble time t_H(z)=1/H(z) in Gyr for a flat (Omega_m, Omega_L) cosmology."""
    # E(z) = H(z)/H0
    ez = np.sqrt(omega_m * (1.0 + redshift) ** 3 + omega_l)
    # 1/H0 in Gyr
    t_h0_gyr = 9.778 / hubble_h
    return t_h0_gyr / ez


def compute_quiescent_mask(sfr, sm, hubble_h, mode, sfh=None, snap_num=None, quench_snaps=1,
                           bulge_mass=None, gtype=None, ssfr_cut_yr=None):
    """Return a boolean array marking quiescent galaxies under the given mode.

    sfr        : (N,)        total SFR in Msun/yr
    sm         : (N,)        StellarMass in 10^10 Msun/h
    sfh        : (N, nbins)  optional total SFH mass in 10^10 Msun/h (only used by 'strict' with quench_snaps>1)
    bulge_mass : (N,)        BulgeMass in 10^10 Msun/h (required for 'massive_bt'/'karls_galaxy' modes)
    gtype      : (N,)        Type int array (0=Central; required for 'karls_galaxy' mode)
    """
    if mode == 'strict':
        q = sfr == 0.0
        if quench_snaps > 1 and sfh is not None and snap_num is not None:
            bin_lo = max(0, snap_num - quench_snaps + 1)
            q &= sfh[:, bin_lo:snap_num].sum(axis=1) == 0.0
        return q
    if mode in ('loose', 'loose_massive'):
        sm_msun = sm * 1e10 / hubble_h
        with np.errstate(divide='ignore', invalid='ignore'):
            ssfr = np.where(sm_msun > 0, sfr / sm_msun, 0.0)
        q = ssfr < 1e-11
        if mode == 'loose_massive':
            q &= sm_msun > MASSIVE_FLOOR_MSUN
        return q
    if mode in ('hubble', 'hubble_central'):
        if ssfr_cut_yr is None:
            raise ValueError(f"mode={mode!r} requires ssfr_cut_yr")
        sm_msun = sm * 1e10 / hubble_h
        with np.errstate(divide='ignore', invalid='ignore'):
            ssfr = np.where(sm_msun > 0, sfr / sm_msun, 0.0)
        q = ssfr < ssfr_cut_yr
        q &= sm_msun > MASSIVE_FLOOR_MSUN
        if bulge_mass is not None:
            bt = np.where(sm > 0, bulge_mass / sm, 0.0)
            q &= (bt >= MASSIVE_BT_MIN) & (bt <= MASSIVE_BT_MAX)
        if mode == 'hubble_central' and gtype is not None:
            q &= (gtype == 0)
        return q
    if mode == 'massive_bt':
        sm_msun = sm * 1e10 / hubble_h
        q = (sfr == 0.0) & (sm_msun >= MASSIVE_BT_FLOOR_MSUN)
        if bulge_mass is not None:
            bt = np.where(sm > 0, bulge_mass / sm, 0.0)
            q &= (bt >= MASSIVE_BT_MIN) & (bt <= MASSIVE_BT_MAX)
        return q
    if mode == 'loose_massive_bt':
        sm_msun = sm * 1e10 / hubble_h
        with np.errstate(divide='ignore', invalid='ignore'):
            ssfr = np.where(sm_msun > 0, sfr / sm_msun, 0.0)
        q = (ssfr < 1e-11) & (sm_msun >= LOOSE_MASSIVE_BT_FLOOR_MSUN)
        if bulge_mass is not None:
            bt = np.where(sm > 0, bulge_mass / sm, 0.0)
            q &= (bt >= MASSIVE_BT_MIN) & (bt <= MASSIVE_BT_MAX)
        return q
    if mode == 'karls_galaxy':
        sm_msun = sm * 1e10 / hubble_h
        with np.errstate(divide='ignore', invalid='ignore'):
            ssfr = np.where(sm_msun > 0, sfr / sm_msun, 0.0)
        q = (ssfr < 1e-11) & (sm_msun >= KARLS_GALAXY_SM_FLOOR)
        if bulge_mass is not None:
            bt = np.where(sm > 0, bulge_mass / sm, 0.0)
            q &= (bt >= KARLS_GALAXY_BT_MIN) & (bt <= KARLS_GALAXY_BT_MAX)
        if gtype is not None:
            q &= (gtype == 0)
        return q
    raise ValueError(f"Unknown quiescent mode: {mode!r}")


def read_header(filepath):
    """Extract all required parameters from the HDF5 header."""
    params = {}
    with h5.File(filepath, 'r') as f:
        sim = f['Header/Simulation']
        params['hubble_h']    = float(sim.attrs['hubble_h'])
        params['box_size']    = float(sim.attrs['box_size'])    # Mpc/h
        params['particle_mass'] = float(sim.attrs['particle_mass'])  # 10^10 Msun/h

        params['snapshot_redshifts'] = np.array(f['Header/snapshot_redshifts'])
        params['save_full_sfh']      = int(f['Header/Runtime'].attrs.get('SaveFullSFH', 0))
        params['omega_matter']       = float(sim.attrs['omega_matter'])
        params['omega_lambda']       = float(sim.attrs['omega_lambda'])

        snap_groups = [k for k in f.keys() if k.startswith('Snap_')]
        params['available_snapshots'] = sorted(int(s.replace('Snap_', '')) for s in snap_groups)

    return params


def snapshot_times_gyr(redshifts, omega_m, omega_l, hubble_h):
    """Return cosmic time in Gyr for each entry in the redshifts array."""
    H0_inv_gyr = 9.778 / hubble_h  # 1/H0 in Gyr
    def integrand(a):
        return 1.0 / (a * np.sqrt(omega_m / a**3 + omega_l))
    times = np.empty(len(redshifts))
    for i, z in enumerate(redshifts):
        t, _ = quad(integrand, 0.0, 1.0 / (1.0 + z))
        times[i] = H0_inv_gyr * t
    return times


def find_snapshot(params, target_z=3.0, snap_override=None):
    """Return the snapshot number closest to target_z, or snap_override if given."""
    if snap_override is not None:
        return snap_override, params['snapshot_redshifts'][snap_override]

    zs = params['snapshot_redshifts']
    available = np.array(params['available_snapshots'])
    dz = np.abs(zs[available] - target_z)
    idx = np.argmin(dz)
    snap = int(available[idx])
    return snap, float(zs[snap])


def load_galaxies(filepaths, snap_num, particle_mass, min_len, quench_snaps, snap_times,
                  hubble_h, qmode='strict', ssfr_cut_yr=None):
    """
    Read Posx/y/z, StellarMass, quiescence flag, and mass-weighted stellar age.

    Quiescence flag is computed using `compute_quiescent_mask` with the given
    qmode ('strict' or 'loose'). See QUIESCENT_MODES for definitions.

    Stellar age: mass-weighted mean age from SFHMassDisk+SFHMassBulge using
    snap_times (Gyr) for the formation time of each bin. NaN when SFH unavailable.
    """
    px, py, pz, sm, mvir, gtype, quiescent, ages = [], [], [], [], [], [], [], []
    galaxy_ids, sfrs, cold_gas_list, bulge_mass_list, bh_mass_list, vvir_list, vmax_list = (
        [], [], [], [], [], [], []
    )
    snap_key = f'Snap_{snap_num}'
    t_now = snap_times[snap_num]

    for fp in filepaths:
        with h5.File(fp, 'r') as f:
            if snap_key not in f:
                continue
            g = f[snap_key]
            if 'StellarMass' not in g or 'Posx' not in g:
                continue
            stellar = np.array(g['StellarMass'], dtype=np.float64)
            length  = np.array(g['Len'], dtype=np.int32)
            mask = (stellar >= particle_mass) & (length >= min_len)
            if mask.sum() == 0:
                continue

            n_masked = int(mask.sum())
            sfr = (np.array(g['SfrDisk'],  dtype=np.float64)[mask] +
                   np.array(g['SfrBulge'], dtype=np.float64)[mask])
            sm_m = stellar[mask]

            if 'SFHMassDisk' in g:
                sfhd = np.array(g['SFHMassDisk'], dtype=np.float64)[mask]   # (N, n_bins)
                sfhb = np.array(g['SFHMassBulge'], dtype=np.float64)[mask]
                sfh  = sfhd + sfhb  # (N, n_bins)

                # Mass-weighted stellar age: bins 0..snap_num-1 are populated
                # bin i = mass formed in interval ending at snap i
                bin_ages  = t_now - snap_times[:snap_num]       # (snap_num,)  age in Gyr
                sfh_pop   = sfh[:, :snap_num]                   # (N, snap_num)
                total_sfh = sfh_pop.sum(axis=1)
                mw_age = np.where(
                    total_sfh > 0,
                    (sfh_pop * bin_ages[np.newaxis, :]).sum(axis=1) / total_sfh,
                    np.nan
                )
            else:
                sfh = None
                mw_age = np.full(n_masked, np.nan)

            bm_raw = (np.array(g['BulgeMass'], dtype=np.float64)[mask]
                      if 'BulgeMass' in g else np.zeros(n_masked))
            gtype_arr = (np.array(g['Type'], dtype=np.int32)[mask]
                         if 'Type' in g else None)
            is_quiescent = compute_quiescent_mask(
                sfr, sm_m, hubble_h, qmode,
                sfh=sfh, snap_num=snap_num, quench_snaps=quench_snaps,
                bulge_mass=bm_raw, gtype=gtype_arr,
                ssfr_cut_yr=ssfr_cut_yr,
            )

            def _load_field(field, dtype=np.float64, fill=0.0):
                if field in g:
                    return np.array(g[field], dtype=dtype)[mask]
                return np.full(n_masked, fill, dtype=dtype)

            px.append(np.array(g['Posx'])[mask])
            py.append(np.array(g['Posy'])[mask])
            pz.append(np.array(g['Posz'])[mask])
            sm.append(stellar[mask])
            mvir.append(np.array(g['Mvir'], dtype=np.float64)[mask])
            gtype.append(np.array(g['Type'], dtype=np.int32)[mask])
            quiescent.append(is_quiescent)
            ages.append(mw_age)

            galaxy_ids.append(_load_field('GalaxyIndex', dtype=np.int64, fill=-1))
            sfrs.append(sfr)
            cold_gas_list.append(_load_field('ColdGas'))
            bulge_mass_list.append(bm_raw)
            bh_mass_list.append(_load_field('BlackHoleMass'))
            vvir_list.append(_load_field('Vvir'))
            vmax_list.append(_load_field('Vmax'))

    if not px:
        return None, None, None, None, None, None, None, None, None

    extras = {
        'galaxy_id':  np.concatenate(galaxy_ids),
        'sfr':        np.concatenate(sfrs),
        'cold_gas':   np.concatenate(cold_gas_list),
        'bulge_mass': np.concatenate(bulge_mass_list),
        'bh_mass':    np.concatenate(bh_mass_list),
        'vvir':       np.concatenate(vvir_list),
        'vmax':       np.concatenate(vmax_list),
    }
    return (np.concatenate(px), np.concatenate(py),
            np.concatenate(pz), np.concatenate(sm),
            np.concatenate(mvir), np.concatenate(gtype),
            np.concatenate(quiescent), np.concatenate(ages), extras)


def save_quiescent_csv(filepaths, snap_num, particle_mass, min_len, quench_snaps,
                       snap_times, hubble_h, out_path, qmode='strict', ssfr_cut_yr=None):
    """
    Write all scalar galaxy fields for quiescent galaxies at snap_num to a CSV.
    Applies the same selection and quiescence logic as load_galaxies under the
    given qmode ('strict' or 'loose'; see QUIESCENT_MODES).
    Adds derived columns: Posx_Mpc, Posy_Mpc, Posz_Mpc, log_StellarMass_Msun,
    log_Mvir_Msun, mw_stellar_age_Gyr, quiescent_class, log_ColdGas_Msun,
    log_BulgeMass_Msun, log_BlackHoleMass_Msun, SFR_total_Msun_yr, bulge_to_total.
    """
    import csv

    SKIP_FIELDS = {'SFHMassDisk', 'SFHMassBulge'}   # 2-D arrays, not suitable for flat CSV

    snap_key = f'Snap_{snap_num}'
    t_now    = snap_times[snap_num]

    # collect field names from the first file that has the snapshot
    field_names = None
    for fp in filepaths:
        with h5.File(fp, 'r') as f:
            if snap_key in f:
                field_names = [k for k in sorted(f[snap_key].keys())
                               if k not in SKIP_FIELDS and f[snap_key][k].ndim == 1]
                break
    if field_names is None:
        print('  No snapshot found for CSV export.')
        return

    derived = ['Posx_Mpc', 'Posy_Mpc', 'Posz_Mpc',
               'log_StellarMass_Msun', 'log_Mvir_Msun',
               'mw_stellar_age_Gyr', 'quiescent_class',
               'log_BulgeMass_Msun', 'log_BlackHoleMass_Msun', 'bulge_to_total']

    rows = []
    for fp in filepaths:
        with h5.File(fp, 'r') as f:
            if snap_key not in f:
                continue
            g = f[snap_key]
            stellar = np.array(g['StellarMass'], dtype=np.float64)
            length  = np.array(g['Len'],         dtype=np.int32)
            mask = (stellar >= particle_mass) & (length >= min_len)
            if mask.sum() == 0:
                continue

            sfr = (np.array(g['SfrDisk'],  dtype=np.float64)[mask] +
                   np.array(g['SfrBulge'], dtype=np.float64)[mask])
            sm_m = stellar[mask]

            if 'SFHMassDisk' in g:
                sfhd = np.array(g['SFHMassDisk'], dtype=np.float64)[mask]
                sfhb = np.array(g['SFHMassBulge'], dtype=np.float64)[mask]
                sfh  = sfhd + sfhb
                bin_ages = t_now - snap_times[:snap_num]
                sfh_pop  = sfh[:, :snap_num]
                total    = sfh_pop.sum(axis=1)
                mw_age   = np.where(total > 0,
                                    (sfh_pop * bin_ages).sum(axis=1) / total,
                                    np.nan)
            else:
                sfh = None
                mw_age = np.full(mask.sum(), np.nan)

            bm_csv = (np.array(g['BulgeMass'], dtype=np.float64)[mask]
                      if 'BulgeMass' in g else np.zeros(mask.sum()))
            gtype_csv = (np.array(g['Type'], dtype=np.int32)[mask]
                         if 'Type' in g else None)
            is_q = compute_quiescent_mask(
                sfr, sm_m, hubble_h, qmode,
                sfh=sfh, snap_num=snap_num, quench_snaps=quench_snaps,
                bulge_mass=bm_csv, gtype=gtype_csv,
                ssfr_cut_yr=ssfr_cut_yr,
            )

            q_idx = np.where(mask)[0][is_q]
            if len(q_idx) == 0:
                continue

            # load all scalar fields for quiescent galaxies
            raw = {k: np.array(g[k])[q_idx] for k in field_names}
            mw_age_q = mw_age[is_q]

            sm_q   = np.array(g['StellarMass'], dtype=np.float64)[q_idx]
            mvir_q = np.array(g['Mvir'],        dtype=np.float64)[q_idx]
            px_q   = np.array(g['Posx'])[q_idx] / hubble_h
            py_q   = np.array(g['Posy'])[q_idx] / hubble_h
            pz_q   = np.array(g['Posz'])[q_idx] / hubble_h
            log_sm  = np.where(sm_q   > 0, np.log10(sm_q   * 1e10 / hubble_h), np.nan)
            log_mv  = np.where(mvir_q > 0, np.log10(mvir_q * 1e10 / hubble_h), np.nan)
            q_class = np.where(mw_age_q > 1.0, 'old_>1Gyr', 'young_<1Gyr')

            def _qfield(field):
                if field in g:
                    return np.array(g[field], dtype=np.float64)[q_idx]
                return np.zeros(len(q_idx))

            bm_q   = _qfield('BulgeMass')
            bh_q   = _qfield('BlackHoleMass')
            log_bm = np.where(bm_q > 0, np.log10(bm_q * 1e10 / hubble_h), np.nan)
            log_bh = np.where(bh_q > 0, np.log10(bh_q * 1e10 / hubble_h), np.nan)
            bt_q   = np.where(sm_q > 0, bm_q / sm_q, np.nan)

            for i in range(len(q_idx)):
                row = {k: raw[k][i] for k in field_names}
                row['Posx_Mpc']               = px_q[i]
                row['Posy_Mpc']               = py_q[i]
                row['Posz_Mpc']               = pz_q[i]
                row['log_StellarMass_Msun']   = log_sm[i]
                row['log_Mvir_Msun']          = log_mv[i]
                row['mw_stellar_age_Gyr']     = mw_age_q[i]
                row['quiescent_class']        = q_class[i]
                row['log_BulgeMass_Msun']     = log_bm[i]
                row['log_BlackHoleMass_Msun'] = log_bh[i]
                row['bulge_to_total']         = bt_q[i]
                rows.append(row)

    if not rows:
        print('  No quiescent galaxies to write.')
        return

    columns = field_names + derived
    with open(out_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved quiescent catalogue ({len(rows):,} galaxies): {out_path}')


def print_quiescent_table(px, py, pz, sm, mvir, gtype, quiescent, ages, hubble_h):
    """Print a summary table of quiescent galaxy properties to stdout."""
    if quiescent.sum() == 0:
        return
    q = quiescent
    type_label = {0: 'Central', 1: 'Satellite', 2: 'Orphan'}
    log_sm = np.log10(sm[q]   * 1e10 / hubble_h)
    log_mv = np.log10(mvir[q] * 1e10 / hubble_h)
    age    = ages[q]
    cls    = np.where(age > 1.0, 'old >1 Gyr', 'young <1 Gyr')
    gt     = gtype[q]
    gx, gy, gz = px[q] / hubble_h, py[q] / hubble_h, pz[q] / hubble_h

    hdrs   = ['#', 'Type', 'log M* [Msun]', 'log Mvir [Msun]', 'Age [Gyr]', 'Class', 'X [Mpc]', 'Y [Mpc]', 'Z [Mpc]']
    widths = [4,   10,     14,              15,                 10,          13,      9,          9,          9]
    sep    = '+-' + '-+-'.join('-' * w for w in widths) + '-+'
    hdr    = '| ' + ' | '.join(f'{h:^{w}}' for h, w in zip(hdrs, widths)) + ' |'

    print(f'\nQuiescent galaxies ({q.sum():,} total):')
    print(sep); print(hdr); print(sep)
    for i in range(q.sum()):
        tname   = type_label.get(int(gt[i]), str(int(gt[i])))
        age_str = f'{age[i]:.2f}' if np.isfinite(age[i]) else 'N/A'
        vals    = [str(i+1), tname, f'{log_sm[i]:.3f}', f'{log_mv[i]:.3f}',
                   age_str, cls[i], f'{gx[i]:.2f}', f'{gy[i]:.2f}', f'{gz[i]:.2f}']
        print('| ' + ' | '.join(f'{v:>{w}}' for v, w in zip(vals, widths)) + ' |')
    print(sep)


def print_quiescent_extended_table(sm, mvir, quiescent, extras, hubble_h):
    """Print a second table of extended properties for quiescent galaxies."""
    if quiescent.sum() == 0:
        return
    q = quiescent

    gal_id = extras['galaxy_id'][q]
    bm     = extras['bulge_mass'][q]
    bh     = extras['bh_mass'][q]
    vvir   = extras['vvir'][q]
    vmax   = extras['vmax'][q]
    sm_q   = sm[q]
    mv_q   = mvir[q]

    log_sm = np.where(sm_q > 0, np.log10(sm_q * 1e10 / hubble_h), np.nan)
    log_mv = np.where(mv_q > 0, np.log10(mv_q * 1e10 / hubble_h), np.nan)
    log_bm = np.where(bm   > 0, np.log10(bm   * 1e10 / hubble_h), np.nan)
    log_bh = np.where(bh   > 0, np.log10(bh   * 1e10 / hubble_h), np.nan)
    bt     = np.where(sm_q > 0, bm / sm_q,                        np.nan)

    hdrs   = ['#', 'GalaxyIndex', 'log M* [Msun]', 'log Mvir [Msun]', 'log BulgeMass',
              'log BH Mass', 'Vvir [km/s]', 'Vmax [km/s]', 'B/T']
    widths = [4,   11,             13,               15,                 13,
              12,             12,             12,             6]
    sep  = '+-' + '-+-'.join('-' * w for w in widths) + '-+'
    hdr  = '| ' + ' | '.join(f'{h:^{w}}' for h, w in zip(hdrs, widths)) + ' |'

    print('\nQuiescent galaxies — extended properties:')
    print(sep); print(hdr); print(sep)
    for i in range(int(q.sum())):
        def _fmt(val, spec):
            return f'{val:{spec}}' if np.isfinite(val) else 'N/A'
        vals = [
            str(i + 1),
            str(int(gal_id[i])),
            _fmt(log_sm[i], '.3f'),
            _fmt(log_mv[i], '.3f'),
            _fmt(log_bm[i], '.3f'),
            _fmt(log_bh[i], '.3f'),
            _fmt(vvir[i],   '.1f'),
            _fmt(vmax[i],   '.1f'),
            _fmt(bt[i],     '.3f'),
        ]
        print('| ' + ' | '.join(f'{v:>{w}}' for v, w in zip(vals, widths)) + ' |')
    print(sep)


def plot_quiescent_sfh(filepaths, snap_num, particle_mass, min_len, quench_snaps,
                       snap_times, hubble_h, out_path, z_snap, run_label, qmode='strict',
                       ssfr_cut_yr=None):
    """
    Plot the star formation history of quiescent galaxies vs lookback time.

    SFH bin j spans [snap_times[j-1], snap_times[j]] (or [0, snap_times[0]] for j=0).
    SFR [M☉/yr] = sfh[j] × 1e10/h / (bin_width[j] × 1e9).

    Quiescence selection follows `compute_quiescent_mask(qmode)`.
    """
    snap_key = f'Snap_{snap_num}'
    t_now    = snap_times[snap_num]

    # bin widths: bin j ends at snap_times[j], starts at snap_times[j-1]
    t_edges    = np.concatenate([[0.0], snap_times[:snap_num]])   # length snap_num+1
    bin_widths = np.diff(t_edges)                                  # length snap_num, in Gyr
    lookback   = t_now - snap_times[:snap_num]                     # Gyr, decreasing

    sfh_list, sm_list, mv_list, age_list = [], [], [], []

    for fp in filepaths:
        with h5.File(fp, 'r') as f:
            if snap_key not in f or 'SFHMassDisk' not in f[snap_key]:
                continue
            g = f[snap_key]
            stellar = np.array(g['StellarMass'], dtype=np.float64)
            length  = np.array(g['Len'],         dtype=np.int32)
            mask = (stellar >= particle_mass) & (length >= min_len)
            if mask.sum() == 0:
                continue

            sfr = (np.array(g['SfrDisk'],  dtype=np.float64)[mask] +
                   np.array(g['SfrBulge'], dtype=np.float64)[mask])

            sfhd = np.array(g['SFHMassDisk'],  dtype=np.float64)[mask]
            sfhb = np.array(g['SFHMassBulge'], dtype=np.float64)[mask]
            sfh  = sfhd + sfhb

            bm_sfh = (np.array(g['BulgeMass'], dtype=np.float64)[mask]
                      if 'BulgeMass' in g else np.zeros(mask.sum()))
            gtype_sfh = (np.array(g['Type'], dtype=np.int32)[mask]
                         if 'Type' in g else None)
            is_q = compute_quiescent_mask(
                sfr, stellar[mask], hubble_h, qmode,
                sfh=sfh, snap_num=snap_num, quench_snaps=quench_snaps,
                bulge_mass=bm_sfh, gtype=gtype_sfh,
                ssfr_cut_yr=ssfr_cut_yr,
            )

            if is_q.sum() == 0:
                continue

            sfh_q = sfh[is_q, :snap_num]          # (Nq, snap_num), 10^10 Msun/h
            sm_q  = stellar[mask][is_q]
            mv_q  = np.array(g['Mvir'], dtype=np.float64)[mask][is_q]

            bin_ages  = t_now - snap_times[:snap_num]
            sfh_pop   = sfh_q
            total_sfh = sfh_pop.sum(axis=1)
            mw_age = np.where(total_sfh > 0,
                               (sfh_pop * bin_ages).sum(axis=1) / total_sfh,
                               np.nan)

            sfh_list.append(sfh_q)
            sm_list.append(sm_q)
            mv_list.append(mv_q)
            age_list.append(mw_age)

    if not sfh_list:
        print('  No SFH data for quiescent galaxies — skipping SFH plot.')
        return

    all_sfh = np.concatenate(sfh_list, axis=0)   # (Nq_total, snap_num)
    all_sm  = np.concatenate(sm_list)
    all_mv  = np.concatenate(mv_list)
    all_age = np.concatenate(age_list)

    # SFR in M☉/yr: mass [10^10 Msun/h] / bin_width [Gyr] → Msun/yr
    sfr_arr = all_sfh * 1e10 / hubble_h / (bin_widths * 1e9)   # (Nq, snap_num)

    log_sm_arr = np.log10(all_sm * 1e10 / hubble_h)
    sm_norm    = mcolors.Normalize(vmin=log_sm_arr.min(), vmax=log_sm_arr.max())
    cmap       = plt.cm.bwr

    fig, ax = plt.subplots(figsize=(8, 5))

    for i in range(len(all_sfh)):
        color = cmap(sm_norm(log_sm_arr[i]))
        ax.plot(lookback, sfr_arr[i], color=color, alpha=0.7, lw=1.2)

    if len(all_sfh) > 1:
        median_sfr = np.nanmedian(sfr_arr, axis=0)
        ax.plot(lookback, median_sfr, color='white', lw=2.5, ls='--', zorder=10, label='Median')
        ax.legend(fontsize=10)

    ax.set_xlabel('Lookback time [Gyr]', fontsize=13)
    ax.set_ylabel(r'SFR [$M_\odot\,\mathrm{yr}^{-1}$]', fontsize=13)
    ax.set_title(f'{run_label}  |  Quiescent galaxy SFH  |  z = {z_snap:.2f}', fontsize=12)
    ax.set_xlim(lookback.max(), 0)   # past on the left
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def load_halos_from_trees(tree_files, snap_num, hubble_h, min_len=0):
    """
    Load FoF central dark matter halo positions and M_Crit200 from STC HDF5 tree file(s).

    Centrals: halos where FirstHaloInFOFGroup[i] == i (they are their own FoF first subhalo).
    Positions: kpc/h in file, returned in Mpc/h.
    Masses: stored in 10^10 Msun/h (same as SAGE Mvir) → log10(Msun).
    """
    px_list, py_list, pz_list, mv_list = [], [], [], []

    for tree_file in tree_files:
        with h5.File(tree_file, 'r') as f:
            tree_keys = [k for k in f.keys() if k.startswith('Tree')]
            for tkey in tree_keys:
                t = f[tkey]
                snaps = np.array(t['SnapNum'], dtype=np.int32)
                all_idx = np.where(snaps == snap_num)[0]
                if len(all_idx) == 0:
                    continue

                fhifog = np.array(t['FirstHaloInFOFGroup'], dtype=np.int32)
                central_mask = fhifog[all_idx] == all_idx
                idx = all_idx[central_mask]

                if min_len > 0:
                    lengths = np.array(t['SubhaloLen'], dtype=np.int32)
                    idx = idx[lengths[idx] >= min_len]

                if len(idx) == 0:
                    continue

                pos = np.array(t['SubhaloPos'])          # (N_tree, 3) kpc/h
                mass = np.array(t['Group_M_Crit200'], dtype=np.float64)

                px_list.append(pos[idx, 0])
                py_list.append(pos[idx, 1])
                pz_list.append(pos[idx, 2])
                mv_list.append(mass[idx])

    if not px_list:
        return None, None, None, None

    px = np.concatenate(px_list) / 1000.0   # kpc/h → Mpc/h
    py = np.concatenate(py_list) / 1000.0
    pz = np.concatenate(pz_list) / 1000.0
    mv = np.concatenate(mv_list)

    # 10^10 Msun/h → Msun (same unit as SAGE Mvir)
    mv_msun = mv * 1e10 / hubble_h
    valid = mv_msun > 0
    log_mv = np.where(valid, np.log10(np.where(valid, mv_msun, 1.0)), np.nan)

    return px, py, pz, log_mv


def _annotate_mass(ax, x, y, logm, color, zorder=10):
    """Bold log10(M*) text just above (x, y), coloured to match the circle."""
    ax.annotate(f'$\mathrm{{log}}_{{10}}\ {logm:.2f}\ \mathrm{{m}}_\odot$', xy=(x, y), xytext=(0, 12),
                textcoords='offset points', ha='center', va='bottom',
                fontsize=14, fontweight='bold', color=color, zorder=zorder)


def _draw_population_overlays(ax, mmc_pos=None, mmc_logm=None, mms_pos=None, mms_logm=None):
    """Draw circle + bold mass label for the most-massive selected central (gold) and satellite (white)."""
    if mmc_pos is not None:
        ax.plot(mmc_pos[0], mmc_pos[1], 'o', mfc='none', mec='gold',
                mew=2.0, markersize=18, zorder=9)
    if mms_pos is not None:
        ax.plot(mms_pos[0], mms_pos[1], 'o', mfc='none', mec='white',
                mew=2.0, markersize=18, zorder=9)
    if mmc_pos is not None and mmc_logm is not None:
        _annotate_mass(ax, mmc_pos[0], mmc_pos[1], mmc_logm, 'gold', zorder=11)
    if mms_pos is not None and mms_logm is not None:
        _annotate_mass(ax, mms_pos[0], mms_pos[1], mms_logm, 'white', zorder=11)


def make_projection(ax, x, y, color_val, quiescent, ages, xlabel, ylabel, box_mpc, cmap, norm, dot_size,
                    mmc_pos=None, mmc_logm=None, mms_pos=None, mms_logm=None):
    sf = ~quiescent
    if sf.sum() > LARGE_SIM_GALAXY_THRESHOLD:
        # Per-point alpha scaled with log10(M*): low-mass galaxies fade out,
        # high-mass galaxies remain visible in dense large-volume simulations.
        cv       = color_val[sf]
        cv_range = cv.max() - cv.min()
        alpha_norm = (cv - cv.min()) / (cv_range if cv_range > 0 else 1.0)
        rgba        = cmap(norm(cv)).copy()           # (N, 4)
        rgba[:, 3]  = 0.01 + alpha_norm ** 2.5 * 0.80  # power-law: [0.01, 0.81]
        ax.scatter(x[sf], y[sf], c=rgba, s=dot_size, linewidths=0, rasterized=True)
        sc = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sc.set_array(cv)
    else:
        sc = ax.scatter(x[sf], y[sf], c=color_val[sf], cmap=cmap, norm=norm,
                        s=dot_size, alpha=0.8, linewidths=0, rasterized=True)
    if quiescent.sum() > 0:
        old  = quiescent & (ages > 1.0)   # gold: mean stellar age > 1 Gyr
        young = quiescent & ~old           # red:  younger or no age info
        if young.sum() > 0:
            ax.scatter(x[young], y[young], c='red', marker='*',
                       s=dot_size * 30, alpha=0.9, linewidths=0,
                       zorder=5, rasterized=True, label='Quiescent (<1 Gyr)')
        if old.sum() > 0:
            ax.scatter(x[old], y[old], c='gold', marker='*',
                       s=dot_size * 30, alpha=0.9, linewidths=0,
                       zorder=6, rasterized=True, label='Quiescent (>1 Gyr)')
    _draw_population_overlays(ax, mmc_pos=mmc_pos, mmc_logm=mmc_logm,
                              mms_pos=mms_pos, mms_logm=mms_logm)
    ax.set_xlim(0, box_mpc)
    ax.set_ylim(0, box_mpc)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_aspect('equal')
    return sc


def make_halo_projection(ax, x, y, color_val, xlabel, ylabel, box_mpc, cmap, norm, dot_size,
                         gal_x=None, gal_y=None, quiescent=None, ages=None,
                         mmc_pos=None, mmc_logm=None, mms_pos=None, mms_logm=None):
    if len(x) > LARGE_SIM_GALAXY_THRESHOLD:
        cv       = color_val
        cv_range = cv.max() - cv.min()
        alpha_norm = (cv - cv.min()) / (cv_range if cv_range > 0 else 1.0)
        rgba        = cmap(norm(cv)).copy()
        rgba[:, 3]  = 0.01 + alpha_norm ** 2.5 * 0.80  # power-law: [0.01, 0.81]
        ax.scatter(x, y, c=rgba, s=dot_size, linewidths=0, rasterized=True)
        sc = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sc.set_array(cv)
    else:
        sc = ax.scatter(x, y, c=color_val, cmap=cmap, norm=norm,
                        s=dot_size, alpha=0.8, linewidths=0, rasterized=True)
    if gal_x is not None and quiescent is not None and quiescent.sum() > 0:
        old   = quiescent & (ages > 1.0)
        young = quiescent & ~old
        if young.sum() > 0:
            ax.scatter(gal_x[young], gal_y[young], c='red', marker='*',
                       s=dot_size * 30, alpha=0.9, linewidths=0,
                       zorder=5, rasterized=True, label='Quiescent (<1 Gyr)')
        if old.sum() > 0:
            ax.scatter(gal_x[old], gal_y[old], c='gold', marker='*',
                       s=dot_size * 30, alpha=0.9, linewidths=0,
                       zorder=6, rasterized=True, label='Quiescent (>1 Gyr)')
    _draw_population_overlays(ax, mmc_pos=mmc_pos, mmc_logm=mmc_logm,
                              mms_pos=mms_pos, mms_logm=mms_logm)
    ax.set_xlim(0, box_mpc)
    ax.set_ylim(0, box_mpc)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_aspect('equal')
    return sc


def parse_args():
    p = argparse.ArgumentParser(description='Spatial distribution of galaxies at z~3')
    p.add_argument('output_folder',
                   help='Folder containing model_*.hdf5 files (e.g. output/millennium/)')
    p.add_argument('--redshift', type=float, default=3.0,
                   help='Target redshift (default: 3.0)')
    p.add_argument('--snapshot', type=int, default=None,
                   help='Override: use this snapshot number directly')
    p.add_argument('--dot-size', type=float, default=1.5,
                   help='Scatter point size (default: 1.5)')
    p.add_argument('--min-len', type=int, default=0,
                   help='Minimum halo particle count (default: 20)')
    p.add_argument('--quench-snaps', type=int, default=1,
                   help='Number of consecutive snapshots with SFR=0 to classify as quiescent '
                        '(default: 1; applies only to the strict mode)')
    p.add_argument('--modes', nargs='+', default=['strict', 'loose', 'loose_massive'],
                   choices=list(QUIESCENT_MODES.keys()),
                   help='Quiescence definitions to produce outputs for. '
                        'strict = SFR==0; loose = sSFR<1e-11/yr; '
                        'loose_massive = sSFR<1e-11/yr AND M* > 1e10 Msun; '
                        'hubble = sSFR < 0.2/t_H(z) AND M* > 1e10 Msun AND 0.75<=B/T<=0.90 using the snapshot redshift; '
                        'hubble_central = hubble AND central only (Type==0); '
                        'massive_bt = M*>=1e11 Msun AND SFR==0 AND 0.75<=B/T<=0.90. '
                        'Default: strict, loose, loose_massive. '
                        'Each mode produces its own plots/tables/CSV with a _<mode> suffix.')
    p.add_argument('--output-dir', type=str, default=None,
                   help='Where to save figures (default: <output_folder>/plots/)')
    p.add_argument('--tree-file', type=str, nargs='+', default=None,
                   help='STC HDF5 tree file(s) for halo plots (e.g. input/millennium/trees/trees_STC.hdf5); '
                        'accepts multiple files; skips halo plots if none found')
    return p.parse_args()


def run_workflow_for_mode(filepaths, snap, z_snap, params, particle_mass, snap_times,
                          h, args, out_dir, run_label, halos, qmode):
    """Produce projections + tables + CSV + SFH plot for a single quiescence mode.

    Output files are tagged with `_<qmode>` so the two modes don't overwrite each
    other. Tables are also labelled in stdout.
    """
    mode_desc = QUIESCENT_MODES[qmode]
    suffix = f'_{qmode}'
    box_mpc = params['box_size'] / h

    print()
    print('=' * 70)
    print(f'  Quiescence mode: {qmode}  ({mode_desc})')
    print('=' * 70)

    ssfr_cut_yr = None
    if qmode in ('hubble', 'hubble_central'):
        t_h_gyr = hubble_time_gyr(z_snap, params['omega_matter'], params['omega_lambda'], h)
        ssfr_cut_yr = HUBBLE_QUIESCENCE_FACTOR / (t_h_gyr * 1.0e9)
        print(f"  Hubble time at z={z_snap:.3f}: t_H = {t_h_gyr:.3f} Gyr")
        print(f"  Hubble-quiescent cut: sSFR < {HUBBLE_QUIESCENCE_FACTOR}/t_H = {ssfr_cut_yr:.3e} /yr")
        print(f"  Stellar-mass cut: M* > {MASSIVE_FLOOR_MSUN:.2e} Msun")
        print(f"  Morphology cut: {MASSIVE_BT_MIN:.3f} <= B/T <= {MASSIVE_BT_MAX:.3f}")
        if qmode == 'hubble_central':
            print("  Type cut: central only (Type==0)")

    # --- load galaxies for this mode ---
    px, py, pz, sm, mvir, gtype, quiescent, ages, extras = load_galaxies(
        filepaths, snap, particle_mass, args.min_len, args.quench_snaps,
        snap_times, h, qmode=qmode, ssfr_cut_yr=ssfr_cut_yr)
    if px is None:
        print(f'No galaxies with StellarMass > 0 found in Snap_{snap}; skipping mode {qmode}.')
        return

    px = px / h;  py = py / h;  pz = pz / h
    log_sm = np.log10(sm * 1e10 / h)

    n_sf   = (~quiescent).sum()
    n_q    = int(quiescent.sum())
    n_gold = int((quiescent & (ages > 1.0)).sum())
    n_red  = n_q - n_gold
    print(f'Loaded {len(px):,} galaxies  '
          f'(log M* range: {log_sm.min():.2f} – {log_sm.max():.2f})')
    print(f'  Star-forming: {n_sf:,}   Quiescent: {n_q:,}'
          f'  (red <1Gyr: {n_red:,}, gold >1Gyr: {n_gold:,})')

    # --- colormap & normalisation (mode-specific because SF/quiescent split changes) ---
    sf_mask = ~quiescent
    base_arr = log_sm[sf_mask] if sf_mask.any() else log_sm
    vmin, vmax = np.percentile(base_arr, [2, 98])
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.bwr

    q_cen_sm = np.where((gtype == 0) & quiescent, sm, 0.0)
    q_sat_sm = np.where((gtype == 1) & quiescent, sm, 0.0)
    mmc_idx = int(np.argmax(q_cen_sm)) if q_cen_sm.max() > 0 else None
    mms_idx = int(np.argmax(q_sat_sm)) if q_sat_sm.max() > 0 else None
    if n_q > 0:
        print(f'  Most-massive selected central / satellite circled (gold / white).')

    projections = [
        ('Posx', 'Posy', px, py, 'X [Mpc]', 'Y [Mpc]', 'XY'),
        ('Posx', 'Posz', px, pz, 'X [Mpc]', 'Z [Mpc]', 'XZ'),
        ('Posy', 'Posz', py, pz, 'Y [Mpc]', 'Z [Mpc]', 'YZ'),
    ]

    for _, _, xa, ya, xlabel, ylabel, tag in projections:
        mmc_pos = (xa[mmc_idx], ya[mmc_idx]) if mmc_idx is not None else None
        mms_pos = (xa[mms_idx], ya[mms_idx]) if mms_idx is not None else None
        mmc_logm = float(log_sm[mmc_idx]) if mmc_idx is not None else None
        mms_logm = float(log_sm[mms_idx]) if mms_idx is not None else None

        fig, ax = plt.subplots(figsize=(7, 6.5))
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        sc = make_projection(ax, xa, ya, log_sm, quiescent, ages, xlabel, ylabel,
                             box_mpc, cmap, norm, args.dot_size,
                             mmc_pos=mmc_pos, mms_pos=mms_pos,
                             mmc_logm=mmc_logm, mms_logm=mms_logm)

        ax.set_title(f'{run_label}  |  {tag} projection  |  z = {z_snap:.2f}  |  {qmode}: {mode_desc}',
                     fontsize=11, color='white')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')

        fig.tight_layout()
        fname = os.path.join(out_dir, f'spatial_{tag}_z{z_snap:.2f}_{run_label}{suffix}.pdf')
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {fname}')

    # --- quiescent galaxy table + CSV + SFH plot ---
    if n_q > 0:
        print(f'\n[{qmode}: {mode_desc}]')
        print_quiescent_table(px, py, pz, sm, mvir, gtype, quiescent, ages, h)
        print_quiescent_extended_table(sm, mvir, quiescent, extras, h)

        csv_path = os.path.join(out_dir, f'quiescent_galaxies_z{z_snap:.2f}_{run_label}{suffix}.csv')
        save_quiescent_csv(filepaths, snap, particle_mass, args.min_len, args.quench_snaps,
                           snap_times, h, csv_path, qmode=qmode, ssfr_cut_yr=ssfr_cut_yr)

        sfh_path = os.path.join(out_dir, f'quiescent_sfh_z{z_snap:.2f}_{run_label}{suffix}.pdf')
        plot_quiescent_sfh(filepaths, snap, particle_mass, args.min_len, args.quench_snaps,
                   snap_times, h, sfh_path, z_snap, run_label, qmode=qmode,
                   ssfr_cut_yr=ssfr_cut_yr)

    # --- halo projections (re-use the pre-loaded halo data; overlay this mode's quiescent) ---
    if halos is not None:
        hpx, hpy, hpz, hlog_mv = halos
        valid = np.isfinite(hlog_mv)
        hv_min, hv_max = np.percentile(hlog_mv[valid], [2, 98])
        hnorm = mcolors.Normalize(vmin=hv_min, vmax=hv_max)
        hcmap = plt.cm.bwr

        halo_projections = [
            (hpx, hpy, px, py, 'X [Mpc]', 'Y [Mpc]', 'XY'),
            (hpx, hpz, px, pz, 'X [Mpc]', 'Z [Mpc]', 'XZ'),
            (hpy, hpz, py, pz, 'Y [Mpc]', 'Z [Mpc]', 'YZ'),
        ]

        for hxa, hya, gxa, gya, xlabel, ylabel, tag in halo_projections:
            mmc_pos = (gxa[mmc_idx], gya[mmc_idx]) if mmc_idx is not None else None
            mms_pos = (gxa[mms_idx], gya[mms_idx]) if mms_idx is not None else None
            mmc_logm = float(log_sm[mmc_idx]) if mmc_idx is not None else None
            mms_logm = float(log_sm[mms_idx]) if mms_idx is not None else None

            fig, ax = plt.subplots(figsize=(7, 6.5))
            fig.patch.set_facecolor('black')
            ax.set_facecolor('black')
            sc = make_halo_projection(ax, hxa, hya, hlog_mv, xlabel, ylabel,
                                      box_mpc, hcmap, hnorm, args.dot_size,
                                      gal_x=gxa, gal_y=gya,
                                      quiescent=quiescent, ages=ages,
                                      mmc_pos=mmc_pos, mms_pos=mms_pos,
                                      mmc_logm=mmc_logm, mms_logm=mms_logm)

            ax.set_title(f'{run_label}  |  Halos {tag}  |  z = {z_snap:.2f}  |  {qmode}: {mode_desc}',
                         fontsize=11, color='white')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            for spine in ax.spines.values():
                spine.set_edgecolor('white')

            fig.tight_layout()
            fname = os.path.join(out_dir, f'spatial_halos_{tag}_z{z_snap:.2f}_{run_label}{suffix}.pdf')
            fig.savefig(fname, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f'Saved: {fname}')


def main():
    args = parse_args()

    # --- find files ---
    pattern = os.path.join(args.output_folder, 'model_*.hdf5')
    filepaths = sorted(glob.glob(pattern))
    if not filepaths:
        sys.exit(f'No model_*.hdf5 files found in {args.output_folder}')

    print(f'Found {len(filepaths)} HDF5 file(s) in {args.output_folder}')

    # --- read header from first file ---
    params = read_header(filepaths[0])
    h = params['hubble_h']

    # --- choose snapshot ---
    snap, z_snap = find_snapshot(params, target_z=args.redshift,
                                 snap_override=args.snapshot)
    print(f'Using Snap_{snap}  (z = {z_snap:.4f},'
          f' requested z ~ {args.redshift if args.snapshot is None else "override"})')

    particle_mass = params['particle_mass']
    print(f'Particle mass: {particle_mass:.4f} x10^10 Msun/h'
          f'  (floor = {particle_mass*1e10/h:.3e} Msun)')

    has_sfh = bool(params['save_full_sfh'])
    effective_quench = args.quench_snaps
    if args.quench_snaps > 1 and not has_sfh:
        print('  Warning: SaveFullSFH=0 — SFH arrays unavailable, falling back to quench_snaps=1')
        effective_quench = 1
    print(f'Cuts: Len >= {args.min_len},  '
          f'strict-mode quench window = {effective_quench} snapshot(s)')
    print(f'Quiescence modes to run: {", ".join(args.modes)}')

    # --- precompute cosmic time at every snapshot ---
    snap_times = snapshot_times_gyr(
        params['snapshot_redshifts'],
        params['omega_matter'], params['omega_lambda'], h
    )
    print(f'Cosmic time at selected snapshot: {snap_times[snap]:.3f} Gyr')

    # --- output directory ---
    out_dir = args.output_dir or os.path.join(args.output_folder, 'plots')
    os.makedirs(out_dir, exist_ok=True)

    # --- derive a run label from the folder name ---
    run_label = os.path.basename(os.path.normpath(args.output_folder))

    # --- resolve tree files and load halos once (mode-independent) ---
    tree_files = args.tree_file
    if tree_files is None:
        folder_name = os.path.basename(os.path.normpath(args.output_folder))
        pattern = os.path.join('input', folder_name, 'trees', 'trees_STC*.hdf5')
        found = sorted(glob.glob(pattern))
        if found:
            tree_files = found

    if tree_files:
        expanded = []
        for tf in tree_files:
            if os.path.isdir(tf):
                found = sorted(glob.glob(os.path.join(tf, '*.hdf5')))
                if found:
                    expanded.extend(found)
                else:
                    print(f'\nWarning: no HDF5 files found in directory: {tf}')
            else:
                expanded.append(tf)
        tree_files = expanded

        missing = [tf for tf in tree_files if not os.path.isfile(tf)]
        for tf in missing:
            print(f'\nWarning: tree file not found: {tf}')
        tree_files = [tf for tf in tree_files if os.path.isfile(tf)]

    halos = None
    if tree_files:
        print(f'\nLoading halos from {len(tree_files)} tree file(s)...')
        hpx, hpy, hpz, hlog_mv = load_halos_from_trees(
            tree_files, snap, h, min_len=args.min_len)
        if hpx is None:
            print(f'No FoF central halos found at Snap_{snap} in tree file.')
        else:
            hpx /= h;  hpy /= h;  hpz /= h
            valid = np.isfinite(hlog_mv)
            print(f'Loaded {len(hpx):,} FoF central halos  '
                  f'(log M_Crit200 range: {hlog_mv[valid].min():.2f} – {hlog_mv[valid].max():.2f})')
            halos = (hpx, hpy, hpz, hlog_mv)
    else:
        if args.tree_file:
            print('\nWarning: no valid tree files found — skipping halo plots.')
        else:
            print('\nNo tree file found — skipping halo plots. Use --tree-file to specify one.')

    # --- run per-mode workflow ---
    for qmode in args.modes:
        run_workflow_for_mode(filepaths, snap, z_snap, params, particle_mass,
                              snap_times, h, args, out_dir, run_label, halos, qmode)

    print('\nDone.')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
find_galaxy.py
==============
Look up individual galaxies in SAGE26 HDF5 output, either by galaxy ID or by
a set of halo / stellar / black-hole mass ranges, and dump every field SAGE
stores for the matches to a file.

Two selection modes (mutually exclusive):
  --galaxy-id ID [ID ...]   exact match on the galaxy ID field (GalaxyIndex by
                             default). With --all-snapshots, follows that
                             galaxy across its whole history.
  --halo-mass-min/-max,
  --stellar-mass-min/-max,
  --bh-mass-min/-max,
  --acc-rate-min/-max,
  --edd-rate-min/-max,
  --edd-ratio-min,
  --bh-frac-min/-max         any combination of range cuts (Msun, or Msun/yr
                             for the two rate options, dimensionless ratios
                             for --edd-ratio-min and --bh-frac-min/-max,
                             AND'ed together), applied at the chosen
                             snapshot(s). --bh-frac-min/-max filters on
                             BlackHoleMass / StellarMass.

Units / conventions (shared across SAGE26)
------------------------------------------
* Masses stored on disk in 1e10 Msun/h; this script converts the fields in
  MASS_FIELDS to Msun before filtering/reporting (conv = 1e10 / hubble_h).
* BHMaxaccretionRate and BHEddingtonRateLimit are mass-per-code-time; they are
  converted to Msun/yr using conv / (UnitTime_in_s / SEC_PER_YEAR), where
  UnitTime_in_s = UnitLength_in_cm / UnitVelocity_in_cm_per_s is read from
  Header/Runtime (falls back to the Millennium defaults).
* Per-snapshot history fields (shape [ngal, SimMaxSnaps], e.g. per-channel BH
  accretion, SFH, BHMaxaccretionRate/BHEddingtonRateLimit) are reduced to a
  single column per galaxy: col = min(max(snapshot - 1, 0), ncols-1). SAGE
  only finishes writing a history column for snapshot N once processing has
  moved on to N+1, so the column matching N itself is always empty -- N-1 is
  the latest one actually populated at output time for Snap_N.

Examples
--------
  python find_galaxy.py -i "output/millennium_mini/model_*.hdf5" --galaxy-id 12345
  python find_galaxy.py -i "..." --galaxy-id 12345 --all-snapshots -o history.csv
  python find_galaxy.py -i "..." --bh-mass-min 1e6 --stellar-mass-min 1e10 -s 63
  python find_galaxy.py -i "..." --acc-rate-min 1.0 -s 63
"""

import argparse
import glob
import os
import sys

import numpy as np
import h5py
import pandas as pd

HUBBLE_H_DEFAULT = 0.73
POSSIBLE_ID_FIELDS = ['GalaxyIndex', 'GalaxyID', 'ID', 'galaxy_id', 'id', 'GalID']

SEC_PER_YEAR = 365.25 * 24 * 3600
UNIT_LENGTH_IN_CM_DEFAULT = 3.08568e24        # SAGE Millennium default
UNIT_VELOCITY_IN_CM_PER_S_DEFAULT = 1.0e5     # SAGE Millennium default

# Fields stored in 1e10 Msun/h that get converted to Msun for filtering/output.
MASS_FIELDS = {
    'BHMergerMass', 'BHSeedMass', 'BlackHoleMass', 'BulgeMass', 'CGMgas',
    'CentralMvir', 'ColdGas', 'EjectedMass', 'H1gas', 'H2gas', 'HotGas',
    'ICS_accrete', 'ICS_disrupt', 'ICS_sum_mt', 'InstabilityBulgeMass',
    'InstabilityDrivenBHaccretionMass', 'IntraClusterStars', 'MergerBulgeMass',
    'MergerDrivenBHaccretionMass', 'MetalsBulgeMass', 'MetalsCGMgas',
    'MetalsColdGas', 'MetalsEjectedMass', 'MetalsHotGas',
    'MetalsIntraClusterStars', 'MetalsStellarMass', 'Mvir',
    'QuasarModeBHaccretionMass', 'RadioModeBHaccretionMass', 'SFHMassBulge',
    'SFHMassDisk', 'StellarMass', 'infallMvir', 'infallStellarMass',
}

# Fields stored in 1e10 Msun/h per code-time-unit; converted to Msun/yr.
RATE_FIELDS = {'BHMaxaccretionRate', 'BHEddingtonRateLimit'}


def read_simulation_params(filepath):
    params = {
        'Hubble_h': HUBBLE_H_DEFAULT,
        'UnitTime_in_s': UNIT_LENGTH_IN_CM_DEFAULT / UNIT_VELOCITY_IN_CM_PER_S_DEFAULT,
        'redshifts': None,
        'available_snapshots': [],
        'latest_snapshot': None,
    }
    with h5py.File(filepath, 'r') as f:
        if 'Header/Simulation' in f:
            sim = f['Header/Simulation'].attrs
            params['Hubble_h'] = float(sim.get('hubble_h',
                                        sim.get('HubbleParam', HUBBLE_H_DEFAULT)))
        elif 'Header' in f:
            hdr = f['Header'].attrs
            params['Hubble_h'] = float(hdr.get('hubble_h',
                                        hdr.get('HubbleParam', HUBBLE_H_DEFAULT)))
        if 'Header/Runtime' in f:
            rt = f['Header/Runtime'].attrs
            unit_length = float(rt.get('UnitLength_in_cm', UNIT_LENGTH_IN_CM_DEFAULT))
            unit_velocity = float(rt.get('UnitVelocity_in_cm_per_s',
                                         UNIT_VELOCITY_IN_CM_PER_S_DEFAULT))
            params['UnitTime_in_s'] = unit_length / unit_velocity
        if 'Header/snapshot_redshifts' in f:
            params['redshifts'] = np.array(f['Header/snapshot_redshifts'])
        snaps = sorted(int(k.split('_')[1]) for k in f.keys() if k.startswith('Snap_'))
        params['available_snapshots'] = snaps
        params['latest_snapshot'] = max(snaps) if snaps else None
    return params


def get_redshift(snap_num, redshifts):
    if redshifts is not None and 0 <= snap_num < len(redshifts):
        return float(redshifts[snap_num])
    return None


def find_id_field(file_list, snap_num, override=None):
    if override:
        return override
    for f in file_list:
        with h5py.File(f, 'r') as hf:
            key = f"Snap_{snap_num}"
            if key in hf:
                for c in POSSIBLE_ID_FIELDS:
                    if c in hf[key]:
                        return c
    return None


def read_snapshot_frame(filepath, snap_num, hubble_h, unit_time_in_s, source_label):
    """Read every field of one Snap_<N> group into a flat per-galaxy DataFrame.

    2D history fields [ngal, maxsnaps] are collapsed to the column for this
    snapshot; MASS_FIELDS are converted from 1e10 Msun/h to Msun; RATE_FIELDS
    are converted from 1e10 Msun/h/(code time) to Msun/yr.
    """
    conv = 1.0e10 / hubble_h
    rate_conv = conv * SEC_PER_YEAR / unit_time_in_s
    key = f"Snap_{snap_num}"
    with h5py.File(filepath, 'r') as hf:
        if key not in hf:
            return None
        grp = hf[key]
        if len(grp.keys()) == 0:
            return None
        ngal = None
        cols = {}
        for field in grp.keys():
            arr = np.array(grp[field])
            if arr.ndim == 2:
                col = min(max(snap_num - 1, 0), arr.shape[1] - 1)
                arr = arr[:, col]
            if ngal is None:
                ngal = len(arr)
            if field in MASS_FIELDS:
                arr = arr * conv
            elif field in RATE_FIELDS:
                arr = arr * rate_conv
            cols[field] = arr
        if not ngal:
            return None
        cols['Snapshot'] = np.full(ngal, snap_num, dtype=int)
        cols['SourceFile'] = [source_label] * ngal
        return pd.DataFrame(cols)


def collect_matches(file_list, snap_list, hubble_h, unit_time_in_s, id_field,
                    galaxy_ids, mass_cuts, edd_ratio_min=None,
                    bh_frac_min=None, bh_frac_max=None):
    frames = []
    for snap_num in snap_list:
        for fpath in file_list:
            df = read_snapshot_frame(fpath, snap_num, hubble_h, unit_time_in_s,
                                     os.path.basename(fpath))
            if df is None:
                continue

            if galaxy_ids is not None:
                if id_field not in df.columns:
                    continue
                mask = df[id_field].isin(galaxy_ids)
            else:
                mask = pd.Series(True, index=df.index)
                for cut_key, field in (('halo', 'Mvir'),
                                       ('stellar', 'StellarMass'),
                                       ('bh', 'BlackHoleMass'),
                                       ('acc_rate', 'BHMaxaccretionRate'),
                                       ('edd_rate', 'BHEddingtonRateLimit')):
                    lo, hi = mass_cuts[cut_key]
                    if lo is not None:
                        mask &= df.get(field, 0) >= lo
                    if hi is not None:
                        mask &= df.get(field, 0) <= hi

                if edd_ratio_min is not None:
                    if 'BHMaxaccretionRate' in df.columns and 'BHEddingtonRateLimit' in df.columns:
                        edd = df['BHEddingtonRateLimit']
                        ratio = df['BHMaxaccretionRate'] / edd.replace(0, np.nan)
                        mask &= (edd > 0) & (ratio >= edd_ratio_min)
                        df = df.assign(EddRatio=ratio)
                    else:
                        mask &= False

                if bh_frac_min is not None or bh_frac_max is not None:
                    if 'BlackHoleMass' in df.columns and 'StellarMass' in df.columns:
                        stellar = df['StellarMass']
                        bh_frac = df['BlackHoleMass'] / stellar.replace(0, np.nan)
                        frac_mask = stellar > 0
                        if bh_frac_min is not None:
                            frac_mask &= bh_frac >= bh_frac_min
                        if bh_frac_max is not None:
                            frac_mask &= bh_frac <= bh_frac_max
                        mask &= frac_mask
                        df = df.assign(BHFrac=bh_frac)
                    else:
                        mask &= False

            if mask.any():
                frames.append(df[mask])

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def write_txt_report(df, redshifts, id_field, out_path):
    with open(out_path, 'w') as fh:
        fh.write(f"Found {len(df)} matching galaxy record(s)\n")
        fh.write("=" * 70 + "\n\n")
        for i, row in df.iterrows():
            z = get_redshift(int(row['Snapshot']), redshifts)
            header = f"Record {i + 1}  |  Snapshot {int(row['Snapshot'])}"
            if z is not None:
                header += f"  (z = {z:.3f})"
            header += f"  |  {row['SourceFile']}"
            if id_field in row:
                header += f"  |  {id_field} = {row[id_field]}"
            fh.write(header + "\n")
            fh.write("-" * 70 + "\n")
            for field in df.columns:
                if field in ('Snapshot', 'SourceFile'):
                    continue
                fh.write(f"  {field:<35} {row[field]}\n")
            fh.write("\n")


def main():
    p = argparse.ArgumentParser(
        description="Find SAGE26 galaxies by ID or mass range and dump all "
                    "their fields to a file.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-i', '--input-pattern',
                   default='./output/millennium/model_*.hdf5',
                   help='Glob for the model HDF5 files.')
    p.add_argument('-s', '--snapshot', type=int, default=None,
                   help='Snapshot to search (default: latest available).')
    p.add_argument('--all-snapshots', action='store_true',
                   help='Search every available snapshot instead of just one '
                        '(e.g. to pull a full history for --galaxy-id).')
    p.add_argument('--galaxy-id', type=int, nargs='+', default=None,
                   help='One or more galaxy IDs to match exactly.')
    p.add_argument('--id-field', default=None,
                   help='Override auto-detected ID field (default: '
                        f'first of {POSSIBLE_ID_FIELDS} found in the file).')
    p.add_argument('--halo-mass-min', type=float, default=None)
    p.add_argument('--halo-mass-max', type=float, default=None)
    p.add_argument('--stellar-mass-min', type=float, default=None)
    p.add_argument('--stellar-mass-max', type=float, default=None)
    p.add_argument('--bh-mass-min', type=float, default=None)
    p.add_argument('--bh-mass-max', type=float, default=None)
    p.add_argument('--acc-rate-min', type=float, default=None,
                   help='Min BHMaxaccretionRate, in Msun/yr.')
    p.add_argument('--acc-rate-max', type=float, default=None,
                   help='Max BHMaxaccretionRate, in Msun/yr.')
    p.add_argument('--edd-rate-min', type=float, default=None,
                   help='Min BHEddingtonRateLimit, in Msun/yr.')
    p.add_argument('--edd-rate-max', type=float, default=None,
                   help='Max BHEddingtonRateLimit, in Msun/yr.')
    p.add_argument('--edd-ratio-min', type=float, default=None,
                   help='Min BHMaxaccretionRate / BHEddingtonRateLimit '
                        '(e.g. 1.0 selects super-Eddington galaxies, i.e. '
                        'max accretion rate exceeds the Eddington rate).')
    p.add_argument('--bh-frac-min', type=float, default=None,
                   help='Min BlackHoleMass / StellarMass ratio.')
    p.add_argument('--bh-frac-max', type=float, default=None,
                   help='Max BlackHoleMass / StellarMass ratio.')
    p.add_argument('-o', '--output', default=None,
                   help='Output file path (default: galaxy_lookup_results.txt '
                        'or .csv next to the first input file).')
    p.add_argument('--format', choices=['txt', 'csv'], default='txt',
                   help='Output format (default: txt).')
    args = p.parse_args()

    mass_filters_given = any(v is not None for v in (
        args.halo_mass_min, args.halo_mass_max, args.stellar_mass_min,
        args.stellar_mass_max, args.bh_mass_min, args.bh_mass_max,
        args.acc_rate_min, args.acc_rate_max, args.edd_rate_min,
        args.edd_rate_max, args.edd_ratio_min, args.bh_frac_min,
        args.bh_frac_max))

    if args.galaxy_id is None and not mass_filters_given:
        print("Error: supply either --galaxy-id or at least one range "
              "option (--halo-mass-min/-max, --stellar-mass-min/-max, "
              "--bh-mass-min/-max, --acc-rate-min/-max, --edd-rate-min/-max).")
        sys.exit(1)
    if args.galaxy_id is not None and mass_filters_given:
        print("Error: --galaxy-id and range filters are mutually "
              "exclusive; use one selection mode at a time.")
        sys.exit(1)

    file_list = sorted(glob.glob(args.input_pattern))
    if not file_list:
        print(f"Error: no files match {args.input_pattern}")
        sys.exit(1)

    sim = read_simulation_params(file_list[0])
    hubble_h = sim['Hubble_h']
    redshifts = sim['redshifts']
    available = sim['available_snapshots']
    if not available:
        print("Error: no Snap_<N> groups found in the input files.")
        sys.exit(1)

    if args.all_snapshots:
        snap_list = available
    else:
        snap_num = args.snapshot if args.snapshot is not None else sim['latest_snapshot']
        if snap_num not in available:
            print(f"Error: snapshot {snap_num} not available. "
                  f"Available: {available[0]}-{available[-1]}")
            sys.exit(1)
        snap_list = [snap_num]

    id_field = find_id_field(file_list, snap_list[0], args.id_field)
    if args.galaxy_id is not None and id_field is None:
        print("Error: no galaxy-ID field found in the input files "
              f"(looked for {POSSIBLE_ID_FIELDS}).")
        sys.exit(1)

    mass_cuts = {
        'halo': (args.halo_mass_min, args.halo_mass_max),
        'stellar': (args.stellar_mass_min, args.stellar_mass_max),
        'bh': (args.bh_mass_min, args.bh_mass_max),
        'acc_rate': (args.acc_rate_min, args.acc_rate_max),
        'edd_rate': (args.edd_rate_min, args.edd_rate_max),
    }

    print("=" * 70)
    print("SAGE26 galaxy lookup")
    print("=" * 70)
    print(f"  files       : {len(file_list)}")
    print(f"  snapshots   : {'all (' + str(len(snap_list)) + ')' if args.all_snapshots else snap_list[0]}")
    print(f"  Hubble_h    : {hubble_h}")
    if args.galaxy_id is not None:
        print(f"  mode        : galaxy ID ({id_field}) in {args.galaxy_id}")
    else:
        print(f"  mode        : range cuts (Msun; Msun/yr for rates) -> {mass_cuts}")
        if args.edd_ratio_min is not None:
            print(f"  edd ratio   : BHMaxaccretionRate / BHEddingtonRateLimit >= {args.edd_ratio_min}")
        if args.bh_frac_min is not None or args.bh_frac_max is not None:
            print(f"  bh fraction : BlackHoleMass / StellarMass in "
                  f"[{args.bh_frac_min}, {args.bh_frac_max}]")
    print("=" * 70)

    matches = collect_matches(file_list, snap_list, hubble_h,
                              sim['UnitTime_in_s'], id_field,
                              set(args.galaxy_id) if args.galaxy_id else None,
                              mass_cuts, args.edd_ratio_min,
                              args.bh_frac_min, args.bh_frac_max)

    if matches.empty:
        print("No matching galaxies found.")
        sys.exit(0)

    input_dir = os.path.dirname(os.path.abspath(file_list[0]))
    if args.output:
        out_path = args.output
    else:
        ext = 'csv' if args.format == 'csv' else 'txt'
        out_path = os.path.join(input_dir, f"galaxy_lookup_results.{ext}")

    if args.format == 'csv':
        matches.to_csv(out_path, index=False)
    else:
        write_txt_report(matches, redshifts, id_field, out_path)

    print(f"Found {len(matches)} matching galaxy record(s).")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()

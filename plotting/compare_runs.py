#!/usr/bin/env python3
"""
compare_runs.py
===============
Overlay black-hole diagnostic plots for two (or more) SAGE26 runs on the
same axes -- run A solid/full-color, run B (and beyond) dashed and
lightened -- instead of re-running the single-run plotting scripts once
per run into separate folders.

`--plots blackholes` overlays plotting/allresults-blackholes.py's 5
diagnostics (growth channels, accretion rate function, seed density,
BH-bulge relation, BH mass function) -- run A solid/full-color, run B
dashed/lightened.

`--plots lrd` / `lrd_multiz` overlay bh_lrd_analysis.py's / bh_lrd_analysis
_multiz.py's panels a-f. Panels a-e are scatter+KDE-contour plots of
individual accretion events, so in compare mode they drop the raw
background scatter and the LRD red/blue selection dots (unreadable with
two+ full point clouds) and show one KDE density contour per run instead,
colored per run. Panel f (a luminosity function) keeps its category split
and uses linestyle per run, like the blackholes plots. Single-run mode in
both scripts is completely unaffected.

Usage
-----
    python3 plotting/compare_runs.py \\
        --run millennium/heavy_mergernoedd_edd_10t:Merger \\
        --run millennium/heavy_disknoedd_edd_10t:Disk \\
        --plots blackholes,lrd,lrd_multiz

Each --run value is a path (relative to the repo root, or to `output/` if
it doesn't already start with `output/` or an absolute path -- matching
the convention `output/millennium/<run_name>/model_*.hdf5`), optionally
followed by `:LABEL` (defaults to the last path component).
"""

import argparse
import glob
import importlib.util
import os
import sys

# allresults-blackholes.py's filename has a hyphen, so it can't be imported
# with a normal `import` statement.
_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "allresults_blackholes", os.path.join(_HERE, "allresults-blackholes.py"))
ab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ab)

import bh_lrd_analysis as la
import bh_lrd_analysis_multiz as lam
from run_style import style_for_index, contour_style_for_index

PLOT_CHOICES = ['blackholes', 'lrd', 'lrd_multiz']


def resolve_run_path(path):
    """Match the output/<sim>/<run_name>/model_*.hdf5 convention: prepend
    'output/' unless the path is already absolute or starts with it."""
    if os.path.isabs(path) or path.startswith('output/') or path.startswith('output' + os.sep):
        return path
    return os.path.join('output', path)


def parse_run_arg(raw):
    """'PATH[:LABEL]' -> (resolved_dir, label)."""
    if ':' in raw and not os.path.isabs(raw.split(':', 1)[0]):
        path, label = raw.split(':', 1)
    else:
        path, label = raw, None
    resolved = resolve_run_path(path)
    if not label:
        label = os.path.basename(os.path.normpath(path))
    return resolved, label


def load_run(raw, snapshot_override=None):
    """Resolve one --run argument into the dict shape the *_compare()
    functions in allresults-blackholes.py expect."""
    run_dir, label = parse_run_arg(raw)
    file_list = sorted(glob.glob(os.path.join(run_dir, 'model_*.hdf5')))
    if not file_list:
        sys.exit(f"ERROR: no 'model_*.hdf5' files found under {run_dir!r} "
                 f"(from --run {raw!r}).")

    sim = ab.read_simulation_params(file_list[0])
    hubble_h = sim['Hubble_h']
    redshifts = sim['redshifts']
    available = sim['available_snapshots']
    fracvol = ab.total_volume_fraction(file_list)

    snap_num = snapshot_override if snapshot_override is not None else sim['latest_snapshot']
    if snap_num is None:
        sys.exit(f"ERROR: could not determine a snapshot to use for {run_dir!r}.")
    if available and snap_num not in available:
        print(f"  [warn] {label}: snapshot {snap_num} not available; "
              f"using latest ({sim['latest_snapshot']}).")
        snap_num = sim['latest_snapshot']

    volume_h3 = (sim['BoxSize'] ** 3) * fracvol
    volume_phys = (sim['BoxSize'] / hubble_h) ** 3 * fracvol

    return {
        'label': label,
        'run_dir': run_dir,
        'file_list': file_list,
        'hubble_h': hubble_h,
        'redshifts': redshifts,
        'available': available,
        'snap_num': snap_num,
        'volume_h3': volume_h3,
        'volume_phys': volume_phys,
    }


def run_lrd_compare(runs, output_dir, args):
    """--plots lrd: overlay bh_lrd_analysis.py's panels a-f."""
    snap_col = args.snapshot if args.snapshot is not None else 40
    la_runs, la_runs_f, z_label = [], [], None
    for i, run in enumerate(runs):
        data = la.read_epoch(run['file_list'], snap_col, run['hubble_h'],
                             catalogue=args.catalogue, window=args.window)
        # panels a-e distinguish runs by contour color; panel f (a luminosity
        # function, not a scatter+KDE plot) distinguishes them by linestyle.
        la_runs.append({'data': data, 'style': contour_style_for_index(i, run['label'])})
        la_runs_f.append({'data': data, 'style': style_for_index(i, run['label'])})
        if z_label is None:
            z_label = la.snap_to_z(snap_col, run['redshifts'])

    kw = dict(bhar_floor=args.bhar_floor, z_override=z_label,
             show_lit=(not args.no_lit), mask_seeds=(not args.no_mask_seeds))
    d = output_dir
    la.plot_panel_a_compare(la_runs, snap_col, os.path.join(d, 'lrd_bh_accretion_scatter_compare.png'), **kw)
    la.plot_panel_b_compare(la_runs, snap_col, os.path.join(d, 'lrd_fbh_scatter_compare.png'), **kw)
    la.plot_panel_c_compare(la_runs, snap_col, os.path.join(d, 'lrd_mbh_mstar_scatter_compare.png'), **kw)
    la.plot_panel_d_compare(la_runs, snap_col, os.path.join(d, 'lrd_lbol_mbh_scatter_compare.png'), **kw)
    la.plot_panel_e_compare(la_runs, snap_col, os.path.join(d, 'lrd_m1450_mbh_scatter_compare.png'), **kw)
    volume_h3 = la.read_box_volume_h3(runs[0]['file_list'])
    la.plot_panel_f_compare(la_runs_f, snap_col, os.path.join(d, 'lrd_bolometric_luminosity_function_compare.png'),
                            volume_h3, n_bins=args.lf_bins, h_h=runs[0]['hubble_h'], **kw)


def run_lrd_multiz_compare(runs, output_dir, args):
    """--plots lrd_multiz: overlay bh_lrd_analysis_multiz.py's panels a-f."""
    redshift_bins = lam.DEFAULT_REDSHIFTS
    snap_data_by_run, snap_data_by_run_f = {}, {}
    for zb in redshift_bins:
        entries, entries_f = [], []
        for i, run in enumerate(runs):
            file_list, h_h, redshifts = run['file_list'], run['hubble_h'], run['redshifts']
            if isinstance(zb, tuple):
                lo, hi = zb
                cols = lam.snaps_in_range(lo, hi, redshifts)
                if not cols:
                    cols = [lam.nearest_snap_for_z(0.5 * (lo + hi), redshifts)]
                data = la.read_epoch(file_list, cols[0], h_h, catalogue=args.catalogue, cols=cols)
            else:
                snap_col = lam.nearest_snap_for_z(zb, redshifts)
                # default to that snapshot's own catalogue -- see
                # bh_lrd_analysis_multiz.py's module docstring.
                data = la.read_epoch(file_list, snap_col, h_h,
                                     catalogue=args.catalogue or f'Snap_{snap_col}',
                                     window=args.window)
            # panels a-e distinguish runs by contour color; panel f (a
            # luminosity function) distinguishes them by linestyle.
            entries.append({'data': data, 'style': contour_style_for_index(i, run['label'])})
            entries_f.append({'data': data, 'style': style_for_index(i, run['label'])})
        snap_data_by_run[zb] = entries
        snap_data_by_run_f[zb] = entries_f

    kw = dict(bhar_floor=args.bhar_floor, show_lit=(not args.no_lit),
             mask_seeds=(not args.no_mask_seeds))
    d = output_dir
    lam.make_grid('a', redshift_bins, snap_data_by_run,
                 os.path.join(d, 'lrd_bh_accretion_scatter_multiz_compare.png'),
                 lam.draw_panel_a_compare, **kw)
    lam.make_grid('b', redshift_bins, snap_data_by_run,
                 os.path.join(d, 'lrd_fbh_scatter_multiz_compare.png'),
                 lam.draw_panel_b_compare, **kw)
    lam.make_grid('c', redshift_bins, snap_data_by_run,
                 os.path.join(d, 'lrd_mbh_mstar_scatter_multiz_compare.png'),
                 lam.draw_panel_c_compare, **kw)
    lam.make_grid('d', redshift_bins, snap_data_by_run,
                 os.path.join(d, 'lrd_lbol_mbh_scatter_multiz_compare.png'),
                 lam.draw_panel_d_compare, **kw)
    lam.make_grid('e', redshift_bins, snap_data_by_run,
                 os.path.join(d, 'lrd_m1450_mbh_scatter_multiz_compare.png'),
                 lam.draw_panel_e_compare, x_reversed=True, **kw)
    volume_h3 = la.read_box_volume_h3(runs[0]['file_list'])
    lam.make_grid('f', redshift_bins, snap_data_by_run_f,
                 os.path.join(d, 'lrd_bolometric_luminosity_function_multiz_compare.png'),
                 lam.draw_panel_f_compare, volume_h3=volume_h3, n_bins=args.lf_bins,
                 h_h=runs[0]['hubble_h'], **kw)


def main():
    p = argparse.ArgumentParser(
        description="Overlay black-hole diagnostic plots for two or more SAGE26 runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', action='append', dest='runs', required=True,
                   metavar='PATH[:LABEL]',
                   help="A run directory (e.g. millennium/heavy_mergernoedd_edd_10t), "
                        "optionally followed by ':LABEL'. Repeat for each run "
                        "(at least 2 expected).")
    p.add_argument('-s', '--snapshot', type=int, default=None,
                   help='Snapshot to use for every run (default: each run\'s own latest).')
    p.add_argument('--plots', default='blackholes',
                   help=f"Comma list of {{{','.join(PLOT_CHOICES)}}} (default: blackholes). "
                        "Only 'blackholes' is implemented so far.")
    p.add_argument('--outdir', default=None,
                   help='Output directory (default: output/compare/<label_a>_vs_<label_b>/).')
    p.add_argument('--data-dir', default='./data/bh/',
                   help='Directory with BHMF observational fig4_bhmf_z*.txt files.')
    p.add_argument('--bin-mode', choices=['none', 'stellar', 'redshift'], default='none',
                   help="Accretion rate function panelling for --plots blackholes (default: none).")
    p.add_argument('--edd-limited', action='store_true')
    p.add_argument('--no-cuts', action='store_true')
    p.add_argument('--seed-density-zmax', type=float, default=None)
    p.add_argument('--window', type=int, default=0,
                   help='For --plots lrd/lrd_multiz: stack +/- this many snapshot '
                        'columns to fight sparsity at high z.')
    p.add_argument('--catalogue', default=None,
                   help='For --plots lrd/lrd_multiz: force a specific Snap_N catalogue group.')
    p.add_argument('--bhar-floor', type=float, default=la.LRD_BHAR_DEFAULT,
                   help='For --plots lrd/lrd_multiz: BHAR floor in M_sun/yr for LRD selection.')
    p.add_argument('--no-lit', action='store_true',
                   help='For --plots lrd/lrd_multiz: skip the literature overlay.')
    p.add_argument('--no-mask-seeds', action='store_true',
                   help='For --plots lrd/lrd_multiz: do not mask still-ungrown BH seeds.')
    p.add_argument('--lf-bins', type=int, default=40,
                   help='For --plots lrd/lrd_multiz: number of log10(L_bol) bins for panel f.')
    args = p.parse_args()

    if len(args.runs) < 2:
        sys.exit("ERROR: need at least 2 --run arguments to compare.")

    requested_plots = [s.strip() for s in args.plots.split(',') if s.strip()]
    for rp in requested_plots:
        if rp not in PLOT_CHOICES:
            sys.exit(f"ERROR: unknown --plots entry {rp!r}; choose from {PLOT_CHOICES}.")

    runs = [load_run(raw, args.snapshot) for raw in args.runs]
    for i, run in enumerate(runs):
        run['style'] = style_for_index(i, run['label'])

    if args.outdir:
        output_dir = args.outdir
    else:
        tag = '_vs_'.join(r['label'] for r in runs)
        common_parent = os.path.dirname(runs[0]['run_dir'])
        output_dir = os.path.join(common_parent, 'compare', tag)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("SAGE26 multi-run comparison")
    print("=" * 70)
    for run in runs:
        print(f"  run   : {run['label']:<12} {run['run_dir']}  "
              f"(snap {run['snap_num']}, {len(run['file_list'])} files)")
    print(f"  plots      : {', '.join(requested_plots)}")
    print(f"  output dir : {output_dir}")
    print("=" * 70)

    for rp in requested_plots:
        if rp == 'blackholes':
            ab.plot_bh_growth_channels_compare(runs, output_dir)
            ab.plot_accretion_rate_function_compare(
                runs, output_dir, bin_mode=args.bin_mode,
                edd_limited=args.edd_limited, no_cuts=args.no_cuts)
            ab.plot_bh_seed_density_compare(runs, output_dir, zmax=args.seed_density_zmax)
            ab.plot_bh_bulge_relation_compare(runs, output_dir)
            ab.plot_bh_mass_function_compare(runs, output_dir, data_dir=args.data_dir)
        elif rp == 'lrd':
            run_lrd_compare(runs, output_dir, args)
        elif rp == 'lrd_multiz':
            run_lrd_multiz_compare(runs, output_dir, args)

    print("=" * 70)
    print("Done.")


if __name__ == '__main__':
    main()

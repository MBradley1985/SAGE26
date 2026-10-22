#!/usr/bin/env python3
"""
paper_plots_bhseeds.py
=======================
Publication figures for the BH-seed / Eddington-limiting growth-channel
story (see bh_investigation/GROWTH_STORIES.md). Unlike compare_runs.py,
which overlays runs on shared axes, figures here stack one run per panel
so each run's own shape is easy to read off directly.

Usage:
    python plotting/paper_plots_bhseeds.py

Reuses allresults-blackholes.py's data I/O and rate-function binning
(loaded via importlib since its filename has a hyphen), same as
compare_runs.py does.
"""

import glob
import importlib.util
import os
import shutil
import string

import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "allresults_blackholes", os.path.join(_HERE, "allresults-blackholes.py"))
ab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ab)

MILLENNIUM_DIR = os.path.join(_HERE, '..', 'output', 'millennium')
OUTPUT_DIR = os.path.join(MILLENNIUM_DIR, 'plots', 'paper')
OUTPUT_FORMAT = '.pdf'

# Panels for plot_accretion_density_function_seedmodels(), top to bottom.
# These are mini-Millennium (10t) placeholder runs standing in for the full
# scenario runs (baseline_heavy_eddlim / unlimited_light / unlimited_heavy,
# see run_sage_scenarios.sh) -- swap the run_name column once those finish.
ACCRETION_DENSITY_RUNS = [
    ('heavy_edd_10t',   'Heavy seed, Eddington-limited'),
    ('light_noedd_10t',   'Light seed, unlimited'),
    ('heavy_noedd_10t', 'Heavy seed, unlimited'),
]


def setup_style():
    """Configure matplotlib for publication-quality white-background plots.

    The shared style hardcodes text.usetex=True, which needs a `latex`
    binary on PATH (available via `ml` on the cluster, not installed on
    every dev machine) -- fall back to mathtext so local runs still work."""
    plt.style.use(os.path.join(_HERE, 'kieren_cohare_palatino_sty.mplstyle'))
    if shutil.which('latex') is None:
        print('  [note] latex not found on PATH; falling back to mathtext '
              '(labels will render slightly differently than on the cluster).')
        plt.rcParams['text.usetex'] = False


def save_figure(fig, filepath):
    """Save figure to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath)
    print(f'  Saved: {filepath}')
    plt.close(fig)


def _infer_edd_limited(run_name):
    """Whether to cap log10(lambda) at 0 when binning -- inferred from the
    run's directory name (SAGE26 scenario naming: 'noedd'/'unlimited' means
    the sim didn't enforce an Eddington cap)."""
    name = run_name.lower()
    if 'noedd' in name or 'unlimited' in name or 'unlim' in name:
        return False
    return True


def _load_run_rate_function(run_name, snapshot=None, no_cuts=False, n_bins=40):
    """Read one run's (accr, edd, acc_type) at its latest snapshot and bin
    them into a rate-function result dict (see allresults-blackholes.py's
    _rate_function_series). Returns None if the run has no model files or
    the accretion fields aren't present there."""
    run_dir = os.path.join(MILLENNIUM_DIR, run_name)
    file_list = sorted(glob.glob(os.path.join(run_dir, 'model_*.hdf5')))
    if not file_list:
        print(f'  [skip] {run_name}: no model_*.hdf5 files found under {run_dir}')
        return None

    sim = ab.read_simulation_params(file_list[0])
    hubble_h = sim['Hubble_h']
    snap_num = snapshot if snapshot is not None else sim['latest_snapshot']
    if snap_num is None:
        print(f'  [skip] {run_name}: could not determine a snapshot to use.')
        return None

    inputs = ab._read_rate_function_inputs(file_list, snap_num, hubble_h, no_cuts)
    if inputs is None:
        print(f'  [skip] {run_name}: accretion-rate fields not present at snap {snap_num}.')
        return None
    accr, edd, typ, mask, _data = inputs

    fracvol = ab.total_volume_fraction(file_list)
    volume_h3 = (sim['BoxSize'] ** 3) * fracvol

    edd_limited = _infer_edd_limited(run_name)
    result = ab._rate_function_series(accr[mask], edd[mask], typ[mask], volume_h3,
                                       edd_limited, n_bins=n_bins)
    if result is None:
        print(f'  [skip] {run_name}: no valid (accr>0, edd>0) events at snap {snap_num}.')
        return None
    result['edd_limited'] = edd_limited
    result['snap_num'] = snap_num
    return result


def plot_accretion_density_function_seedmodels(runs=ACCRETION_DENSITY_RUNS,
                                                output_dir=OUTPUT_DIR):
    """
    Stack the BH accretion density function (dN/dlog10(lambda)/Mpc^3) for
    several runs, one panel per run (not overlaid), sharing the x-axis.
    Row order follows `runs` (default: heavy/Eddington-limited on top,
    light and heavy unlimited below -- see ACCRETION_DENSITY_RUNS).
    """
    print('Accretion density function (seed-model panels):')
    loaded = [(run_name, label, _load_run_rate_function(run_name)) for run_name, label in runs]

    n = len(loaded)
    fig, axes = plt.subplots(n, 1, figsize=(7, 4.2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for i, (ax, (run_name, label, result)) in enumerate(zip(axes, loaded)):
        panel_tag = f'{string.ascii_lowercase[i]})'
        if result is None:
            ax.text(0.5, 0.5, f'{panel_tag} (no data)', transform=ax.transAxes,
                    ha='center', va='center', color='grey')
            ax.set_xlim(-10, 2.5)
            continue
        ab._draw_rate_series(ax, result)
        ax.axvline(0.0, color='k', ls='--', lw=1.3, alpha=0.7)
        ax.set_xlim(-10, 2.5)
        ax.set_ylim(result['floor'] + 1.0, result['ymax'] + 0.8)
        ax.xaxis.set_minor_locator(ab.AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(ab.AutoMinorLocator(5))
        ax.grid(True, alpha=0.25, ls=':', lw=0.6)
        ax.text(0.03, 0.92, panel_tag, transform=ax.transAxes,
                ha='left', va='top', fontsize=12)
        if i == 0:
            ax.legend(loc='upper right', fontsize=10)
        if i > 0:
            ab._blank_edge_ticklabel(ax, 'y', ax.get_ylim()[1])

    axes[-1].set_xlabel(r'$\log_{10}(\dot{M}_{\rm BH}/\dot{M}_{\rm Edd})$', fontsize=14)
    fig.text(0.02, 0.5,
              r'$\log_{10}(\mathrm{d}N/\mathrm{d}\log_{10}\lambda\,/\,\mathrm{Mpc}^{-3}h^{3})$',
              va='center', rotation='vertical', fontsize=13)
    fig.tight_layout(rect=(0.04, 0, 1, 1))
    fig.subplots_adjust(hspace=0)
    save_figure(fig, os.path.join(output_dir,
                                  'bh_accretion_density_function_seedmodels' + OUTPUT_FORMAT))


PLOTS = {
    'accretion_density_seedmodels': plot_accretion_density_function_seedmodels,
}


def main():
    setup_style()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_accretion_density_function_seedmodels()
    print('Done.')


if __name__ == '__main__':
    main()

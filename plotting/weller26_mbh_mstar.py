#!/usr/bin/env python3
"""
weller26_mbh_mstar.py
======================
M_BH-M_star maps for a SAGE26 run, styled after Figs. 3 & 4 of Weller,
Natarajan, Burke & Dattathri (2026, arXiv:2607.07793) -- "Black Hole and
Galaxy Growth Since Cosmic Noon" -- which track ASTRID and TNG300 BHs
(central, satellite, and off-nuclear wandering) from z=2 to z=0.5 in the
M_BH-M_star plane, colored by merger-driven growth fraction (their Fig. 3)
and by sSFR (their Fig. 4).

SAGE26 doesn't produce off-nuclear "wandering" BHs in its output catalogue
(a stripped satellite's galaxy -- and its BH -- either survives as Type 1+
or is merged into its central) -- so this script reproduces the
central/satellite split only, as two rows of one grid, in place of the
paper's ASTRID/TNG simulation rows.

Two figures are produced, one per color scheme:

  weller26_mbh_mstar_merger_fraction.png  (Fig. 3 style)
      log10(dM_BH,merger / dM_BH,tot): the fraction of a BH's mass growth
      between two epochs that came from direct BH-BH coalescence
      (BHMergerMass) rather than gas accretion (radio mode + disk
      instability + merger-triggered quasar mode). Two columns: z=1 (delta
      since z=2) and z=0.5 (delta since z=1) -- matching the paper's own
      two merger-fraction panels. dM_BH,merger == 0 is set to a ratio of
      1e-5 for visualization, exactly as in the paper's caption.

  weller26_mbh_mstar_ssfr.png  (Fig. 4 style)
      log10(sSFR), sSFR = (SfrDisk+SfrBulge)/StellarMass at that epoch.
      Three columns: z=2, z=1, z=0.5. sSFR == 0 is set to 1e-15, again
      matching the paper's caption.

Both are binned 2D MEDIAN maps (scipy.stats.binned_statistic_2d), not
scatter -- matching the paper's own binned-map style, not the KDE-contour
style used elsewhere in this codebase (bh_lrd_analysis.py). A Kormendy &
Ho (2013) M_BH-M_star relation + scatter band (the same reference used
throughout bh_lrd_analysis.py) is drawn on every panel as a common,
redshift-independent anchor. No BH-mass/stellar-mass "resolution cut"
reference lines are drawn (the paper's red lines) -- those mark ASTRID/
TNG's numerical seed-mass floors, which have no analogue in SAGE's
semi-analytic BH seeding; the sample cut applied instead is the same
selection_mask() (BH/stellar/halo mass floors) used throughout
allresults-blackholes.py, so this plot's sample matches the rest of the
BH diagnostics for this run.

Usage
-----
    python3 plotting/weller26_mbh_mstar.py \\
        -i './output/millennium/heavy_edd_10t/model_*.hdf5'
"""

import argparse
import glob
import importlib.util
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.stats import binned_statistic_2d

# allresults-blackholes.py's filename has a hyphen, so it can't be imported
# with a normal `import` statement (see compare_runs.py for the same trick).
_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "allresults_blackholes", os.path.join(_HERE, "allresults-blackholes.py"))
ab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ab)

from bh_lrd_analysis import kormendy_ho_mbh, KORMENDY_HO_SCATTER

# ============================================================================
# MATPLOTLIB STYLE  (matching bh_lrd_analysis.py)
# ============================================================================
plt.rcParams.update({
    'figure.dpi': 140,
    'font.family': 'serif',
    'font.size': 13.0,
    'axes.linewidth': 1.3,
    'xtick.direction': 'in', 'xtick.top': True, 'xtick.labelsize': 11,
    'ytick.direction': 'in', 'ytick.right': True, 'ytick.labelsize': 11,
    'legend.frameon': False, 'legend.fontsize': 10,
})

# ============================================================================
# CONSTANTS
# ============================================================================
# Growth channels summed for delta_total -- SAME set as GROWTH_CHANNEL_FIELDS
# in allresults-blackholes.py. QuasarModeBHaccretionMass is a scalar (not a
# per-snapshot array) so, like that module's own accounting, it is left out.
GROWTH_CHANNEL_FIELDS_FOR_TOTAL = [
    'MergerDrivenBHaccretionMass', 'InstabilityDrivenBHaccretionMass',
    'RadioModeBHaccretionMass', 'BHMergerMass',
]

MERGER_FLOOR_LOG = -5.0                 # dM_merger == 0 -> this log10 ratio
SSFR_FLOOR_LOG   = -15.0                # SFR == 0       -> this log10 sSFR
MERGER_CLIM      = (-5.0, 0.0)
SSFR_CLIM_DEFAULT = (-15.0, -8.0)

MSTAR_RANGE_DEFAULT = (8.0, 12.5)       # log10(M_star [Msun])
MBH_RANGE_DEFAULT   = (4.0, 10.0)       # log10(M_BH   [Msun])

ROW_SPECS = [('Central', lambda t: t == 0), ('Satellite', lambda t: t >= 1)]


# ============================================================================
# I/O
# ============================================================================

def read_epoch_catalogue(file_list, snap_hi, hubble_h, snap_lo=None):
    """
    Read one epoch's M_BH / M_star / M_vir / sSFR / Type catalogue directly
    from the Snap_{snap_hi} group -- the full snapshot population, unlike
    bh_lrd_analysis.py's read_epoch() which only keeps galaxies with a
    recorded accretion event.

    If snap_lo is given, also computes -- from the SAME group's per-
    snapshot growth-channel history arrays (2D [Ngal, MAXSNAPS], one
    column per snapshot) -- each galaxy's BH mass gained strictly after
    snap_lo through snap_hi (inclusive):
        delta_merger  BHMergerMass channel only (direct BH-BH coalescence)
        delta_total   all four GROWTH_CHANNEL_FIELDS_FOR_TOTAL summed
    matching Weller+26's "total mass gain since the previous snapshot".
    """
    mass_conv = 1.0e10 / hubble_h
    key = f'Snap_{snap_hi}'
    need_delta = snap_lo is not None
    fields = ('bh_mass', 'stellar_mass', 'halo_mass', 'sfr', 'gal_type',
              'delta_merger', 'delta_total')
    cols = {k: [] for k in fields}

    for fpath in file_list:
        with h5py.File(fpath, 'r') as hf:
            if key not in hf or 'BlackHoleMass' not in hf[key]:
                continue
            grp = hf[key]
            bh = np.array(grp['BlackHoleMass']) * mass_conv
            if bh.size == 0:
                continue
            cols['bh_mass'].append(bh)
            cols['stellar_mass'].append(np.array(grp['StellarMass']) * mass_conv)
            cols['halo_mass'].append(np.array(grp['Mvir']) * mass_conv)
            cols['sfr'].append(np.array(grp['SfrDisk']) + np.array(grp['SfrBulge']))
            cols['gal_type'].append(np.array(grp['Type']))

            if need_delta:
                totals = np.zeros_like(bh)
                merger = np.zeros_like(bh)
                lo = max(snap_lo, -1)
                for fld in GROWTH_CHANNEL_FIELDS_FOR_TOTAL:
                    arr = np.array(grp[fld]) * mass_conv
                    hi = min(snap_hi, arr.shape[1] - 1)
                    chan = (np.nansum(arr[:, lo + 1: hi + 1], axis=1) if hi >= lo + 1
                            else np.zeros(arr.shape[0]))
                    totals += chan
                    if fld == 'BHMergerMass':
                        merger = chan
                cols['delta_merger'].append(merger)
                cols['delta_total'].append(totals)

    if not cols['bh_mass']:
        return {k: np.array([]) for k in fields}

    out = {k: np.concatenate(v) for k, v in cols.items() if v}
    if not need_delta:
        n = len(out['bh_mass'])
        out['delta_merger'] = np.full(n, np.nan)
        out['delta_total'] = np.full(n, np.nan)
    return out


# ============================================================================
# COLOR-QUANTITY EXTRACTORS  (catalogue dict, boolean mask) -> (x, y, color)
# ============================================================================

def merger_frac_value_fn(cat, mask):
    dm, dt = cat['delta_merger'][mask], cat['delta_total'][mask]
    sm, bh = cat['stellar_mass'][mask], cat['bh_mass'][mask]
    valid = dt > 0
    dm, dt, sm, bh = dm[valid], dt[valid], sm[valid], bh[valid]
    ratio = np.where(dm > 0, dm / dt, 10 ** MERGER_FLOOR_LOG)
    return np.log10(sm), np.log10(bh), np.log10(ratio)


def ssfr_value_fn(cat, mask):
    sm, sfr, bh = cat['stellar_mass'][mask], cat['sfr'][mask], cat['bh_mass'][mask]
    valid = sm > 0
    sm, sfr, bh = sm[valid], sfr[valid], bh[valid]
    ssfr = np.where(sfr > 0, sfr / sm, 10 ** SSFR_FLOOR_LOG)
    return np.log10(sm), np.log10(bh), np.log10(ssfr)


# ============================================================================
# PLOTTING
# ============================================================================

def binned_median_map(ax, log_mstar, log_mbh, values, xlim, ylim, nbins,
                       min_count, vmin, vmax):
    """Draw one panel's binned-median pcolormesh; return the QuadMesh (for
    the figure's shared colorbar), plus the Kormendy & Ho (2013) reference
    line/band as a common anchor across every panel."""
    x_edges = np.linspace(xlim[0], xlim[1], nbins + 1)
    y_edges = np.linspace(ylim[0], ylim[1], nbins + 1)
    mesh = None
    if log_mstar.size:
        stat, xe, ye, _ = binned_statistic_2d(log_mstar, log_mbh, values,
                                              statistic='median', bins=[x_edges, y_edges])
        counts, _, _, _ = binned_statistic_2d(log_mstar, log_mbh, values,
                                              statistic='count', bins=[x_edges, y_edges])
        stat = np.where(counts >= min_count, stat, np.nan)
        mesh = ax.pcolormesh(xe, ye, stat.T, cmap='viridis', vmin=vmin, vmax=vmax,
                             shading='flat', zorder=1)

    x_ref = np.linspace(xlim[0], xlim[1], 200)
    y_kh = kormendy_ho_mbh(x_ref)
    outline = [pe.withStroke(linewidth=3.0, foreground='white')]
    ax.fill_between(x_ref, y_kh - KORMENDY_HO_SCATTER, y_kh + KORMENDY_HO_SCATTER,
                    color='#F57C00', alpha=0.25, zorder=2, lw=0)
    ax.plot(x_ref, y_kh, color='#F57C00', lw=1.8, zorder=3, path_effects=outline)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(alpha=0.15, zorder=0)
    return mesh


def plot_type_z_grid(cats, col_order, col_labels, value_fn, cbar_label, clim,
                      xlim, ylim, nbins, min_count, no_cuts, output_path, suptitle):
    """One grid: rows = Central/Satellite (Type), columns = col_order
    (snapshot numbers, already resolved to nearest-available per target z).
    `cats` maps snap_num -> catalogue dict from read_epoch_catalogue()."""
    ncols = len(col_order)
    fig, axes = plt.subplots(2, ncols, figsize=(4.3 * ncols, 8.4),
                             sharex=True, sharey=True, squeeze=False)
    mesh = None

    for col, snap_num in enumerate(col_order):
        cat = cats[snap_num]
        base_mask = ab.selection_mask(cat['bh_mass'], cat['stellar_mass'],
                                      cat['halo_mass'], no_cuts) if cat['bh_mass'].size \
            else np.array([], dtype=bool)
        for row, (row_label, type_pred) in enumerate(ROW_SPECS):
            ax = axes[row, col]
            m = base_mask & type_pred(cat['gal_type']) if base_mask.size else base_mask
            log_mstar, log_mbh, values = value_fn(cat, m) if m.size else \
                (np.array([]), np.array([]), np.array([]))
            if log_mstar.size == 0:
                ax.set_xlim(*xlim); ax.set_ylim(*ylim)
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                       ha='center', va='center', fontsize=11, color='#888888')
            else:
                m_out = binned_median_map(ax, log_mstar, log_mbh, values, xlim, ylim,
                                          nbins, min_count, *clim)
                mesh = m_out if m_out is not None else mesh
            ax.text(0.04, 0.94, f'{row_label}\n{col_labels[snap_num]}',
                   transform=ax.transAxes, ha='left', va='top', fontsize=10)

    fig.supxlabel(r'$\log_{10}(M_\star\,[M_\odot])$', fontsize=13)
    fig.supylabel(r'$\log_{10}(M_{\rm BH}\,[M_\odot])$', fontsize=13)
    fig.suptitle(suptitle, fontsize=13, y=1.0)
    fig.tight_layout(rect=(0.0, 0.0, 0.9, 1.0))
    if mesh is not None:
        # explicit colorbar axis to the right of the whole grid -- fig.colorbar's
        # own ax=axes placement badly overlaps the last column under sharex/sharey
        cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(mesh, cax=cbar_ax)
        cbar.set_label(cbar_label, fontsize=12)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[ok]   {suptitle} -> {output_path}')


# ============================================================================
# CLI
# ============================================================================

def main():
    p = argparse.ArgumentParser(
        description="M_BH-M_star maps styled after Weller, Natarajan, Burke & "
                    "Dattathri (2026, arXiv:2607.07793) Figs. 3 & 4 -- colored by "
                    "merger-driven growth fraction and by sSFR, split into "
                    "central/satellite rows.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-i', '--input-pattern', default='./output/millennium/model_*.hdf5')
    p.add_argument('--outdir', default=None,
                   help='Output directory (default: <input dir>/plots).')
    p.add_argument('--nbins', type=int, default=45,
                   help='2D bins per axis for the median-color maps (default 45).')
    p.add_argument('--min-count', type=int, default=1,
                   help='Minimum galaxies per bin to plot (default 1).')
    p.add_argument('--mstar-range', type=float, nargs=2, default=MSTAR_RANGE_DEFAULT,
                   metavar=('LO', 'HI'), help='log10(M_star) axis range.')
    p.add_argument('--mbh-range', type=float, nargs=2, default=MBH_RANGE_DEFAULT,
                   metavar=('LO', 'HI'), help='log10(M_BH) axis range.')
    p.add_argument('--ssfr-clim', type=float, nargs=2, default=SSFR_CLIM_DEFAULT,
                   metavar=('LO', 'HI'), help='log10(sSFR) colorbar range.')
    p.add_argument('--no-cuts', action='store_true',
                   help='Skip the BH/stellar/halo mass sample cuts (allresults-blackholes.py selection_mask).')
    p.add_argument('--no-merger-panel', action='store_true')
    p.add_argument('--no-ssfr-panel', action='store_true')
    args = p.parse_args()

    files = sorted(glob.glob(args.input_pattern))
    if not files:
        sys.exit(f"ERROR: no files matched {args.input_pattern!r}")

    sim = ab.read_simulation_params(files[0])
    hubble_h, redshifts, available = sim['Hubble_h'], sim['redshifts'], sim['available_snapshots']
    if not available:
        sys.exit("ERROR: no Snap_N groups found in the input files.")

    snap = {z: ab.snapshot_for_redshift(z, redshifts, available) for z in (2.0, 1.0, 0.5)}
    actual_z = {z: ab.get_redshift_from_snapshot(snap[z], redshifts) for z in snap}
    print("Target z  ->  nearest available snapshot:")
    for z in (2.0, 1.0, 0.5):
        print(f"  z = {z:g}  ->  snap {snap[z]}  (z = {actual_z[z]:.3f})")

    outdir = Path(args.outdir) if args.outdir else Path(files[0]).parent / 'plots'
    outdir.mkdir(parents=True, exist_ok=True)
    xlim, ylim = tuple(args.mstar_range), tuple(args.mbh_range)

    if not args.no_ssfr_panel:
        col_order = [snap[z] for z in (2.0, 1.0, 0.5)]
        col_labels = {snap[z]: rf'$z \approx {actual_z[z]:.2g}$' for z in (2.0, 1.0, 0.5)}
        cats = {s: read_epoch_catalogue(files, s, hubble_h) for s in set(col_order)}
        plot_type_z_grid(cats, col_order, col_labels, ssfr_value_fn,
                         r'$\log_{10}({\rm sSFR}\,[{\rm yr}^{-1}])$', tuple(args.ssfr_clim),
                         xlim, ylim, args.nbins, args.min_count, args.no_cuts,
                         outdir / 'weller26_mbh_mstar_ssfr.png',
                         'sSFR (styled after Weller+26 Fig. 4)')

    if not args.no_merger_panel:
        cat_z1 = read_epoch_catalogue(files, snap[1.0], hubble_h, snap_lo=snap[2.0])
        cat_z05 = read_epoch_catalogue(files, snap[0.5], hubble_h, snap_lo=snap[1.0])
        cats = {snap[1.0]: cat_z1, snap[0.5]: cat_z05}
        col_order = [snap[1.0], snap[0.5]]
        col_labels = {
            snap[1.0]: rf'$z \approx {actual_z[1.0]:.2g}$ (since $z\approx{actual_z[2.0]:.2g}$)',
            snap[0.5]: rf'$z \approx {actual_z[0.5]:.2g}$ (since $z\approx{actual_z[1.0]:.2g}$)',
        }
        plot_type_z_grid(cats, col_order, col_labels, merger_frac_value_fn,
                         r'$\log_{10}(\Delta M_{\rm BH,merger}/\Delta M_{\rm BH,tot})$',
                         MERGER_CLIM, xlim, ylim, args.nbins, args.min_count, args.no_cuts,
                         outdir / 'weller26_mbh_mstar_merger_fraction.png',
                         'Merger-driven growth fraction (styled after Weller+26 Fig. 3)')

    print("Done.")


if __name__ == '__main__':
    main()

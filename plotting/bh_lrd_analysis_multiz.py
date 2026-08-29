#!/usr/bin/env python3
"""
bh_lrd_analysis_multiz.py
==========================
Multi-redshift companion to bh_lrd_analysis.py.

Instead of one output file per panel per redshift, this script makes ONE
grid figure PER PANEL (a-f), with each subplot showing that panel at a
different, fixed target redshift:

    z ~ 0, 1, 2, 4, 6, 8     (2 rows x 3 cols)

Each target maps to its single closest available Millennium snapshot (via
nearest_snap_for_z()), and reads that snapshot's OWN catalogue directly
(catalogue=f'Snap_{{snap_col}}') -- so a panel shows the population actually
observed at that redshift, not just the subset of galaxies that happen to
survive to a later, more-complete catalogue. This requires accretion
history written with the current (fixed) SnapNum indexing; on older output
a snapshot's own most-recent column is always blank. Pass --redshifts to
override the targets, --catalogue to force one group for every panel, or
--window to stack neighbouring columns around each target if a given
redshift is too sparse on its own. All data reading, unit conversions,
LRD selection,
axis locking, and physical relations are imported unchanged from
bh_lrd_analysis.py so the two scripts can never drift out of sync; only the
per-panel drawing (onto a supplied `ax` instead of a standalone figure) and
the grid assembly are new here.

Usage
-----
    python3 plotting/bh_lrd_analysis_multiz.py
    python3 plotting/bh_lrd_analysis_multiz.py --window 1
    python3 plotting/bh_lrd_analysis_multiz.py --no-panel-c --no-panel-f
    python3 plotting/bh_lrd_analysis_multiz.py --redshifts 0 1 2 4 6
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter
from scipy.stats import gaussian_kde

from bh_lrd_analysis import (
    # I/O
    read_sim_params, read_actual_redshifts, read_box_volume_h3, read_epoch,
    # physics / selection helpers
    compute_selection, lock_axis_range, kormendy_ho_mbh, mask_ungrown_seeds,
    eddington_luminosity, eddington_mdot, lbol_from_mdot, mdot_from_lbol,
    m1450_from_lbol, bh_mass_min_from_lbol, shen20_bolometric_qlf_logphi,
    SHEN20_MODEL_LOGLBOL_MIN,
    # plotting helpers
    plot_lit_points, lit_legend_handles, lit_z_mask, _line_rotation_deg,
    draw_contours_multirun,
    # constants
    snap_to_z, MILLENNIUM_SNAP_TO_Z, LIT_Z_TOL,
    LRD_BHAR_DEFAULT, LRD_BHAR_ALT, LRD_FBHM_THRESH, SEED_GROWTH_THRESHOLD,
    MASS_RATIO_LINES, EDDINGTON_RATIO_LINES, KORMENDY_HO_SCATTER,
    PANEL_A_XLIM, PANEL_A_YLIM, PANEL_B_XLIM, PANEL_B_YLIM,
    PANEL_C_XLIM, PANEL_C_YLIM, PANEL_D_XLIM, PANEL_D_YLIM,
    PANEL_E_XLIM, PANEL_E_YLIM, PANEL_F_XLIM, PANEL_F_YLIM,
    # literature tables
    PANG26, MATHEE24, LABBE25, FURTAK23, LIN25,
)
from run_style import contour_style_for_index, style_for_index, lighten_color

# ============================================================================
# MATPLOTLIB STYLE  (compact variant of bh_lrd_analysis.py's style, sized for
# a 3x3 grid rather than a single 7.5x7.5 panel)
# ============================================================================
plt.rcParams.update({
    'figure.dpi': 120,
    # bh_lrd_analysis.py sets this True at import time (for its own single-
    # panel tight_layout() calls); it's global rcParams state, so it would
    # otherwise silently override this script's manual wspace/hspace=0 and
    # margins every time a figure renders, re-introducing gaps.
    'figure.autolayout': False,
    'font.family': 'serif',
    'font.size': 11.0,
    'axes.linewidth': 1.0,
    'xtick.major.size': 4.5, 'xtick.major.width': 1.0,
    'xtick.minor.size': 3.0, 'xtick.minor.width': 0.5,
    'xtick.direction': 'in', 'xtick.top': True, 'xtick.labelsize': 9,
    'ytick.major.size': 4.5, 'ytick.major.width': 1.0,
    'ytick.minor.size': 3.0, 'ytick.minor.width': 0.5,
    'ytick.direction': 'in', 'ytick.right': True, 'ytick.labelsize': 9,
    'legend.frameon': False, 'legend.fontsize': 8,
})

DEFAULT_REDSHIFTS = [0.0, 1.0, 2.0, 4.0, 6.0, 8.0]

# ── Per-panel fixed axis ranges for the multi-z GRID (separate from the
# single-panel PANEL_*_XLIM/YLIM imported above, since a shared 3x3 grid
# axis may need a different fixed range than a standalone 7.5x7.5 panel --
# tune these independently if the grid ever needs to diverge from the
# single-panel defaults again; for now they're the same values, carried
# back into bh_lrd_analysis.py's own PANEL_*_XLIM/YLIM after tuning them
# here interactively). ──
MULTIZ_PANEL_A_XLIM = PANEL_A_XLIM
MULTIZ_PANEL_A_YLIM = PANEL_A_YLIM
MULTIZ_PANEL_B_XLIM = PANEL_B_XLIM
MULTIZ_PANEL_B_YLIM = PANEL_B_YLIM
MULTIZ_PANEL_C_XLIM = PANEL_C_XLIM
MULTIZ_PANEL_C_YLIM = PANEL_C_YLIM
MULTIZ_PANEL_D_XLIM = PANEL_D_XLIM
MULTIZ_PANEL_D_YLIM = PANEL_D_YLIM
MULTIZ_PANEL_E_XLIM = PANEL_E_XLIM
MULTIZ_PANEL_E_YLIM = PANEL_E_YLIM
MULTIZ_PANEL_F_XLIM = PANEL_F_XLIM
MULTIZ_PANEL_F_YLIM = PANEL_F_YLIM

# Reduced point budgets relative to bh_lrd_analysis.py: a single run of this
# script computes up to 6 panels x 9 redshifts = 54 KDEs, so each one is
# capped more aggressively to keep total runtime reasonable.
N_SCATTER_GRID = 12_000
N_KDE_GRID     = 20_000
KDE_GRID_RES   = 150


def nearest_snap_for_z(z_target, redshifts=None):
    """Snapshot index whose redshift is closest to z_target.

    Prefers the file's own Header/snapshot_redshifts table (`redshifts`, from
    read_actual_redshifts) over MILLENNIUM_SNAP_TO_Z, since the hardcoded
    table assumes the stock Millennium snapshot spacing and silently picks
    the wrong snapshot for any run whose output snapshots don't follow that
    exact spacing.
    """
    if redshifts is not None and len(redshifts):
        return int(np.argmin(np.abs(np.asarray(redshifts) - z_target)))
    return min(MILLENNIUM_SNAP_TO_Z, key=lambda s: abs(MILLENNIUM_SNAP_TO_Z[s] - z_target))


def snaps_in_range(z_lo, z_hi, redshifts=None):
    """Snapshot indices whose redshift falls in [z_lo, z_hi] (inclusive),
    for stacking into one RANGE bin (e.g. z = 2-4) -- the range analogue of
    nearest_snap_for_z(), preferring the file's own snapshot_redshifts table
    over MILLENNIUM_SNAP_TO_Z for the same reason (see nearest_snap_for_z).
    """
    if redshifts is not None and len(redshifts):
        idx = np.where((np.asarray(redshifts) >= z_lo) & (np.asarray(redshifts) <= z_hi))[0]
        return sorted(int(i) for i in idx)
    return sorted(s for s, z in MILLENNIUM_SNAP_TO_Z.items() if z_lo <= z <= z_hi)


def _z_numeric(z_target):
    """Single representative redshift for a bin -- itself for a scalar
    target, the midpoint for a (lo, hi) range bin -- for use anywhere a
    single number is needed (e.g. the Shen+20 model curve in panel f)."""
    return 0.5 * (z_target[0] + z_target[1]) if isinstance(z_target, tuple) else z_target


def _fmt_z(z_target):
    """Display label for a redshift bin: '$z \\approx 0.5$' for a scalar
    target, '$z = 2-4$' for a (lo, hi) range bin."""
    if isinstance(z_target, tuple):
        lo, hi = z_target
        return rf'$z = {lo:g}-{hi:g}$'
    return rf'$z \approx {z_target:g}$'


def _lit_z_mask(source_z, z_target, tol=LIT_Z_TOL):
    """lit_z_mask() extended to a (lo, hi) range bin: a point counts if it
    falls within `tol` of either edge, i.e. anywhere in [lo-tol, hi+tol]."""
    if isinstance(z_target, tuple):
        z_lo, z_hi = z_target
        source_z = np.asarray(source_z, dtype=float)
        return (source_z >= z_lo - tol) & (source_z <= z_hi + tol)
    return lit_z_mask(source_z, z_target, tol)


def _bg_scatter_and_contours(ax, x_bg, y_bg, x_lo, x_hi, y_lo, y_hi,
                             contour_color='#333333', seed=42):
    """Grey background scatter + 68/95/99.7% KDE contours, shared by every
    panel (mirrors the per-panel logic in bh_lrd_analysis.py at reduced
    point budgets suited to a 3x3 grid -- see N_SCATTER_GRID/N_KDE_GRID)."""
    if len(x_bg) > N_SCATTER_GRID:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(x_bg), N_SCATTER_GRID, replace=False)
        x_sc, y_sc = x_bg[idx], y_bg[idx]
    else:
        x_sc, y_sc = x_bg, y_bg
    ax.scatter(x_sc, y_sc, s=2, color='#999999', alpha=0.18,
               linewidths=0, rasterized=True, zorder=1)

    if len(x_bg) < 50:
        return
    try:
        if len(x_bg) > N_KDE_GRID:
            rng = np.random.default_rng(seed + 35)
            idx = rng.choice(len(x_bg), N_KDE_GRID, replace=False)
            xk, yk = x_bg[idx], y_bg[idx]
        else:
            xk, yk = x_bg, y_bg

        kde = gaussian_kde(np.vstack([xk, yk]), bw_method='scott')
        xi = np.linspace(min(x_lo, x_hi), max(x_lo, x_hi), KDE_GRID_RES)
        yi = np.linspace(min(y_lo, y_hi), max(y_lo, y_hi), KDE_GRID_RES)
        Xi, Yi = np.meshgrid(xi, yi)
        Zi = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)

        z_sort = np.sort(Zi.ravel())[::-1]
        z_cum = np.cumsum(z_sort) / z_sort.sum()

        def lvl(frac):
            return z_sort[min(np.searchsorted(z_cum, frac), len(z_sort) - 1)]

        levels = sorted([lvl(0.683), lvl(0.954), lvl(0.997)])
        ax.contour(Xi, Yi, Zi, levels=levels, colors=contour_color,
                   linewidths=[0.5, 0.8, 1.1], linestyles=[':', '--', '-'],
                   zorder=2)
    except Exception as e:
        print(f'    WARNING: KDE contours skipped ({e})')


def _no_data_panel(ax, xlim, ylim, z_target):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
            ha='center', va='center', fontsize=10, color='#999999')
    ax.text(0.95, 0.05, _fmt_z(z_target), transform=ax.transAxes,
            ha='right', va='bottom', fontsize=10)


def _valid_mask(*arrs):
    m = np.ones(len(arrs[0]), dtype=bool)
    for a in arrs:
        m &= np.isfinite(a) & (a > 0)
    return m

# ============================================================================
# PER-PANEL DRAWING  (each draws onto a supplied ax; returns legend handles)
# ============================================================================

def draw_panel_a(ax, data, z_target, show_lrd=True, use_fbh=True,
                 bhar_floor=LRD_BHAR_DEFAULT, show_lit=True,
                 xlim=None, ylim=None, range_only=False, mask_seeds=True):
    bh_mass, mdot, medd, star, seed = (data['bh_mass'], data['mdot_msun_yr'],
                                        data['mdot_edd'], data['stellar_mass'],
                                        data['seed_mass'])
    if len(bh_mass) == 0:
        if not range_only:
            _no_data_panel(ax, xlim or MULTIZ_PANEL_A_XLIM, ylim or MULTIZ_PANEL_A_YLIM, z_target)
        return None, (xlim or MULTIZ_PANEL_A_XLIM), (ylim or MULTIZ_PANEL_A_YLIM)

    valid = _valid_mask(bh_mass, mdot)
    if mask_seeds:
        valid &= mask_ungrown_seeds(bh_mass, seed)
    bh_mass, mdot, medd, star = bh_mass[valid], mdot[valid], medd[valid], star[valid]
    log_mbh, log_mdot = np.log10(bh_mass), np.log10(mdot)

    if use_fbh:
        _, lrd_red, lrd_blue = compute_selection(bh_mass, mdot, medd, star, bhar_floor)
    else:
        bhar_pass = mdot >= bhar_floor
        edd_pass = mdot >= np.where(medd > 0, medd, np.inf)
        lrd_red = bhar_pass & edd_pass
        lrd_blue = np.zeros(len(log_mbh), dtype=bool)

    pang_m   = _lit_z_mask(PANG26['z'], z_target)
    mathee_m = _lit_z_mask(MATHEE24['z'], z_target)
    lin_m    = _lit_z_mask(LIN25['z'], z_target)

    lit_log_mbh, lit_log_mdot = [], []
    if show_lit:
        t_mdot = PANG26['lambda_edd'][pang_m] * 10**eddington_mdot(PANG26['log_mbh'][pang_m])
        lit_log_mbh.append(PANG26['log_mbh'][pang_m]); lit_log_mdot.append(np.log10(t_mdot))
        m_mdot = mdot_from_lbol(np.log10(MATHEE24['lbol_1e44'][mathee_m]) + 44.0)
        lit_log_mbh.append(MATHEE24['log_mbh'][mathee_m]); lit_log_mdot.append(np.log10(m_mdot))
        l_mdot = mdot_from_lbol(LIN25['log_lbol'][lin_m])
        lit_log_mbh.append(LIN25['log_mbh'][lin_m]); lit_log_mdot.append(np.log10(l_mdot))

    selected = (lrd_red | lrd_blue) if show_lrd else np.zeros(len(log_mbh), dtype=bool)
    x_lo, x_hi = lock_axis_range(*MULTIZ_PANEL_A_XLIM,
                                 must_include=[log_mbh[selected], *lit_log_mbh],
                                 axis_name='panel a x-axis')
    y_lo, y_hi = lock_axis_range(*MULTIZ_PANEL_A_YLIM,
                                 must_include=[log_mdot[selected], *lit_log_mdot],
                                 axis_name='panel a y-axis')
    if xlim is not None:
        x_lo, x_hi = xlim
    if ylim is not None:
        y_lo, y_hi = ylim
    if range_only:
        return None, (x_lo, x_hi), (y_lo, y_hi)

    if show_lrd:
        x_fill = np.linspace(x_lo, x_hi, 200)
        y_lower = np.maximum(eddington_mdot(x_fill), np.log10(bhar_floor))
        ax.fill_between(x_fill, y_lower, y_hi, color='#D32F2F', alpha=0.08, zorder=0)

    bg = ~(lrd_red | lrd_blue) if show_lrd else np.ones(len(log_mbh), dtype=bool)
    _bg_scatter_and_contours(ax, log_mbh[bg], log_mdot[bg], x_lo, x_hi, y_lo, y_hi)

    if show_lrd:
        if lrd_blue.sum() > 0:
            ax.scatter(log_mbh[lrd_blue], log_mdot[lrd_blue], s=10,
                       color='#F57C00', edgecolors='white', linewidths=0.2, zorder=5)
        if lrd_red.sum() > 0:
            ax.scatter(log_mbh[lrd_red], log_mdot[lrd_red], s=10,
                       color='#C62828', edgecolors='white', linewidths=0.2, zorder=6)

    x_ref = np.linspace(x_lo, x_hi, 200)
    y_edd = eddington_mdot(x_ref)
    ax.plot(x_ref, y_edd, color='#C62828', lw=1.1, zorder=4)
    ax.plot(x_ref, y_edd + 1.0, color='#E65100', lw=1.1, zorder=4)
    ax.axhline(np.log10(LRD_BHAR_DEFAULT), color='#C62828', lw=0.8, ls='--', zorder=3, alpha=0.85)
    ax.axhline(np.log10(LRD_BHAR_ALT), color='#C62828', lw=0.6, ls=':', zorder=3, alpha=0.70)

    lit_labels = []
    if show_lit:
        if pang_m.any():
            t_mdot = PANG26['lambda_edd'][pang_m] * 10**eddington_mdot(PANG26['log_mbh'][pang_m])
            t_mdot_lo = (PANG26['lambda_edd'][pang_m] - PANG26['lambda_edd_err'][pang_m]) * \
                10**eddington_mdot(PANG26['log_mbh'][pang_m])
            plot_lit_points(ax, 'Pang+26', PANG26['log_mbh'][pang_m], np.log10(t_mdot),
                            xerr=PANG26['log_mbh_err'][pang_m],
                            yerr=[np.log10(t_mdot) - np.log10(np.maximum(t_mdot_lo, t_mdot * 1e-3)),
                                  np.full_like(t_mdot, 0.15)])
            lit_labels.append('Pang+26')
        if mathee_m.any():
            m_log_mdot = np.log10(mdot_from_lbol(np.log10(MATHEE24['lbol_1e44'][mathee_m]) + 44.0))
            plot_lit_points(ax, 'Mathee+24', MATHEE24['log_mbh'][mathee_m], m_log_mdot,
                            xerr=MATHEE24['log_mbh_err'][mathee_m],
                            yerr=MATHEE24['lbol_1e44_err'][mathee_m] / (MATHEE24['lbol_1e44'][mathee_m] * np.log(10)))
            lit_labels.append('Mathee+24')
        if lin_m.any():
            l_log_mdot = np.log10(mdot_from_lbol(LIN25['log_lbol'][lin_m]))
            plot_lit_points(ax, 'Lin+25', LIN25['log_mbh'][lin_m], l_log_mdot,
                            xerr=LIN25['log_mbh_err'][lin_m], yerr=LIN25['log_lbol_err'][lin_m])
            lit_labels.append('Lin+25')

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xticks(np.arange(int(np.ceil(x_lo)), int(np.floor(x_hi)) + 1, 2))
    ax.text(0.95, 0.05, _fmt_z(z_target), transform=ax.transAxes,
            ha='right', va='bottom', fontsize=10)

    handles = [
        Line2D([0], [0], color='#C62828', lw=1.6, label=r'$\dot{M}_{\rm BH} = \dot{M}_{\rm Edd}$'),
        Line2D([0], [0], color='#E65100', lw=1.6, label=r'$\dot{M}_{\rm BH} = 10\,\dot{M}_{\rm Edd}$'),
    ]
    if show_lrd:
        handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='#C62828',
                              markersize=7, label=(r'LRD ($f_{\rm BH}\geq 3\%$)' if use_fbh else 'LRD')))
        if use_fbh:
            handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='#F57C00',
                                  markersize=7, label=r'LRD ($f_{\rm BH}<3\%$)'))
    handles += lit_legend_handles(lit_labels)
    return handles, (x_lo, x_hi), (y_lo, y_hi)


def draw_panel_b(ax, data, z_target, show_lrd=True, bhar_floor=LRD_BHAR_DEFAULT, show_lit=True,
                 xlim=None, ylim=None, range_only=False, mask_seeds=True):
    bh_mass, mdot, medd, star, seed = (data['bh_mass'], data['mdot_msun_yr'],
                                        data['mdot_edd'], data['stellar_mass'],
                                        data['seed_mass'])
    if len(bh_mass) == 0:
        if not range_only:
            _no_data_panel(ax, xlim or MULTIZ_PANEL_B_XLIM, ylim or MULTIZ_PANEL_B_YLIM, z_target)
        return None, (xlim or MULTIZ_PANEL_B_XLIM), (ylim or MULTIZ_PANEL_B_YLIM)

    valid = _valid_mask(bh_mass, mdot, star)
    if mask_seeds:
        valid &= mask_ungrown_seeds(bh_mass, seed)
    bh_mass, mdot, medd, star = bh_mass[valid], mdot[valid], medd[valid], star[valid]
    log_mbh = np.log10(bh_mass)

    f_bh, lrd_red, lrd_blue = compute_selection(bh_mass, mdot, medd, star, bhar_floor)
    log_fbh = np.log10(f_bh)

    pang_m = _lit_z_mask(PANG26['z'], z_target)
    lit_log_fbh = (PANG26['log_mbh'][pang_m] - PANG26['log_mstar'][pang_m]) if show_lit else []

    selected = (lrd_red | lrd_blue) if show_lrd else np.zeros(len(log_mbh), dtype=bool)
    x_lo, x_hi = lock_axis_range(*MULTIZ_PANEL_B_XLIM,
                                 must_include=[log_mbh[selected], PANG26['log_mbh'][pang_m] if show_lit else []],
                                 axis_name='panel b x-axis')
    y_lo, y_hi = lock_axis_range(*MULTIZ_PANEL_B_YLIM,
                                 must_include=[log_fbh[selected], lit_log_fbh],
                                 axis_name='panel b y-axis')
    if xlim is not None:
        x_lo, x_hi = xlim
    if ylim is not None:
        y_lo, y_hi = ylim
    if range_only:
        return None, (x_lo, x_hi), (y_lo, y_hi)

    log_fbh_thresh = np.log10(LRD_FBHM_THRESH)
    if show_lrd:
        ax.fill_between([x_lo, x_hi], log_fbh_thresh, y_hi, color='#D32F2F', alpha=0.08, zorder=0)

    bg = ~(lrd_red | lrd_blue) if show_lrd else np.ones(len(log_mbh), dtype=bool)
    _bg_scatter_and_contours(ax, log_mbh[bg], log_fbh[bg], x_lo, x_hi, y_lo, y_hi, contour_color='#000000')

    if show_lrd:
        if lrd_blue.sum() > 0:
            ax.scatter(log_mbh[lrd_blue], log_fbh[lrd_blue], s=10,
                       color='#F57C00', edgecolors='white', linewidths=0.2, zorder=5)
        if lrd_red.sum() > 0:
            ax.scatter(log_mbh[lrd_red], log_fbh[lrd_red], s=10,
                       color='#C62828', edgecolors='white', linewidths=0.2, zorder=6)

    ax.axhline(np.log10(0.1), color='#F57C00', lw=1.4, zorder=4)
    ax.axhline(log_fbh_thresh, color='#C62828', lw=1.4, zorder=4)

    handles = []
    if show_lit and pang_m.any():
        t_log_fbh = PANG26['log_mbh'][pang_m] - PANG26['log_mstar'][pang_m]
        t_fbh_err = np.sqrt(PANG26['log_mbh_err'][pang_m]**2 + PANG26['log_mstar_err'][pang_m]**2)
        plot_lit_points(ax, 'Pang+26', PANG26['log_mbh'][pang_m], t_log_fbh,
                        xerr=PANG26['log_mbh_err'][pang_m], yerr=t_fbh_err)
        handles += lit_legend_handles(['Pang+26'])

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xticks(np.arange(int(np.ceil(x_lo)), int(np.floor(x_hi)) + 1, 2))
    ax.text(0.95, 0.05, _fmt_z(z_target), transform=ax.transAxes,
            ha='right', va='bottom', fontsize=10)
    return handles, (x_lo, x_hi), (y_lo, y_hi)


def draw_panel_c(ax, data, z_target, show_lrd=True, bhar_floor=LRD_BHAR_DEFAULT, show_lit=True,
                 xlim=None, ylim=None, range_only=False, mask_seeds=True):
    bh_mass, mdot, medd, star, seed = (data['bh_mass'], data['mdot_msun_yr'],
                                        data['mdot_edd'], data['stellar_mass'],
                                        data['seed_mass'])
    if len(bh_mass) == 0:
        if not range_only:
            _no_data_panel(ax, xlim or MULTIZ_PANEL_C_XLIM, ylim or MULTIZ_PANEL_C_YLIM, z_target)
        return None, (xlim or MULTIZ_PANEL_C_XLIM), (ylim or MULTIZ_PANEL_C_YLIM)

    valid = _valid_mask(bh_mass, mdot, star)
    if mask_seeds:
        valid &= mask_ungrown_seeds(bh_mass, seed)
    bh_mass, mdot, medd, star = bh_mass[valid], mdot[valid], medd[valid], star[valid]
    log_mstar, log_mbh = np.log10(star), np.log10(bh_mass)

    _, lrd_red, lrd_blue = compute_selection(bh_mass, mdot, medd, star, bhar_floor)

    selected = (lrd_red | lrd_blue) if show_lrd else np.zeros(len(log_mbh), dtype=bool)
    pang_m   = _lit_z_mask(PANG26['z'], z_target)
    furtak_m = _lit_z_mask(FURTAK23['z'], z_target)
    lit_mstar, lit_mbh = [], []
    if show_lit:
        lit_mstar.append(PANG26['log_mstar'][pang_m])
        lit_mbh.append(PANG26['log_mbh'][pang_m])
        if furtak_m:
            lit_mstar.append([FURTAK23['log_mstar_upper_limit']])
            lit_mbh.append([FURTAK23['log_mbh']])
    x_lo, x_hi = lock_axis_range(*MULTIZ_PANEL_C_XLIM,
                                 must_include=[log_mstar[selected], *lit_mstar],
                                 axis_name='panel c x-axis')
    y_lo, y_hi = lock_axis_range(*MULTIZ_PANEL_C_YLIM,
                                 must_include=[log_mbh[selected], *lit_mbh],
                                 axis_name='panel c y-axis')
    if xlim is not None:
        x_lo, x_hi = xlim
    if ylim is not None:
        y_lo, y_hi = ylim
    if range_only:
        return None, (x_lo, x_hi), (y_lo, y_hi)

    x_ref = np.linspace(x_lo, x_hi, 200)
    y_kh = kormendy_ho_mbh(x_ref)
    ax.fill_between(x_ref, y_kh - KORMENDY_HO_SCATTER, y_kh + KORMENDY_HO_SCATTER,
                    color='#999999', alpha=0.35, zorder=0)
    kh_line, = ax.plot(x_ref, y_kh, color='black', lw=1.4, zorder=2, label='Kormendy & Ho 2013')

    ratio_rot = _line_rotation_deg(1.0, x_lo, x_hi, y_lo, y_hi)
    for ratio in MASS_RATIO_LINES:
        y_ratio = x_ref + np.log10(ratio)
        ax.plot(x_ref, y_ratio, color='black', lw=0.8, ls='--', zorder=2)
        x_lab = x_lo + 0.12 * (x_hi - x_lo)
        y_lab = x_lab + np.log10(ratio)
        if y_lo < y_lab < y_hi:
            ax.text(x_lab, y_lab - 0.12, rf'$M_{{\rm BH}}/M_\star = {ratio:g}$',
                    fontsize=6.5, color='black', ha='left', va='top',
                    rotation=ratio_rot, rotation_mode='anchor')

    bg = ~(lrd_red | lrd_blue) if show_lrd else np.ones(len(log_mbh), dtype=bool)
    _bg_scatter_and_contours(ax, log_mstar[bg], log_mbh[bg], x_lo, x_hi, y_lo, y_hi, contour_color='#4A4A8A')

    if show_lrd:
        if lrd_blue.sum() > 0:
            ax.scatter(log_mstar[lrd_blue], log_mbh[lrd_blue], s=10,
                       color='#F57C00', edgecolors='white', linewidths=0.2, zorder=5)
        if lrd_red.sum() > 0:
            ax.scatter(log_mstar[lrd_red], log_mbh[lrd_red], s=10,
                       color='#C62828', edgecolors='white', linewidths=0.2, zorder=6)

    lit_labels = []
    if show_lit:
        if pang_m.any():
            plot_lit_points(ax, 'Pang+26', PANG26['log_mstar'][pang_m], PANG26['log_mbh'][pang_m],
                            xerr=PANG26['log_mstar_err'][pang_m], yerr=PANG26['log_mbh_err'][pang_m])
            lit_labels.append('Pang+26')
        if furtak_m:
            plot_lit_points(ax, 'Furtak+23', [FURTAK23['log_mstar_upper_limit']], [FURTAK23['log_mbh']],
                            xerr=[[0.4], [0.0]],
                            yerr=[[FURTAK23['log_mbh_err_lo']], [FURTAK23['log_mbh_err_hi']]],
                            xuplims=True)
            lit_labels.append('Furtak+23')

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.text(0.95, 0.05, _fmt_z(z_target), transform=ax.transAxes,
            ha='right', va='bottom', fontsize=10)

    handles = [kh_line]
    if show_lrd:
        handles += [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#C62828',
                   markersize=7, label=r'LRD ($f_{\rm BH}\geq 3\%$)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#F57C00',
                   markersize=7, label=r'LRD ($f_{\rm BH}<3\%$)'),
        ]
    handles += lit_legend_handles(lit_labels)
    return handles, (x_lo, x_hi), (y_lo, y_hi)


def draw_panel_d(ax, data, z_target, show_lrd=True, bhar_floor=LRD_BHAR_DEFAULT, show_lit=True,
                 xlim=None, ylim=None, range_only=False, mask_seeds=True):
    bh_mass, mdot, medd, star, seed = (data['bh_mass'], data['mdot_msun_yr'],
                                        data['mdot_edd'], data['stellar_mass'],
                                        data['seed_mass'])
    if len(bh_mass) == 0:
        if not range_only:
            _no_data_panel(ax, xlim or MULTIZ_PANEL_D_XLIM, ylim or MULTIZ_PANEL_D_YLIM, z_target)
        return None, (xlim or MULTIZ_PANEL_D_XLIM), (ylim or MULTIZ_PANEL_D_YLIM)

    valid = _valid_mask(bh_mass, mdot, star)
    if mask_seeds:
        valid &= mask_ungrown_seeds(bh_mass, seed)
    bh_mass, mdot, medd, star = bh_mass[valid], mdot[valid], medd[valid], star[valid]
    log_mbh = np.log10(bh_mass)
    log_lbol = lbol_from_mdot(mdot)

    _, lrd_red, lrd_blue = compute_selection(bh_mass, mdot, medd, star, bhar_floor)

    pang_m   = _lit_z_mask(PANG26['z'], z_target)
    mathee_m = _lit_z_mask(MATHEE24['z'], z_target)
    lin_m    = _lit_z_mask(LIN25['z'], z_target)
    furtak_m = _lit_z_mask(FURTAK23['z'], z_target)

    lit_log_mbh, lit_log_lbol = [], []
    if show_lit:
        lit_log_mbh.append(PANG26['log_mbh'][pang_m])
        lit_log_lbol.append(np.log10(PANG26['lambda_edd'][pang_m]) + eddington_luminosity(PANG26['log_mbh'][pang_m]))
        lit_log_mbh.append(MATHEE24['log_mbh'][mathee_m])
        lit_log_lbol.append(np.log10(MATHEE24['lbol_1e44'][mathee_m]) + 44.0)
        lit_log_mbh.append(LIN25['log_mbh'][lin_m])
        lit_log_lbol.append(LIN25['log_lbol'][lin_m])
        if furtak_m:
            lit_log_mbh.append([FURTAK23['log_mbh']])
            lit_log_lbol.append([FURTAK23['log_lbol']])

    selected = (lrd_red | lrd_blue) if show_lrd else np.zeros(len(log_mbh), dtype=bool)
    x_lo, x_hi = lock_axis_range(*MULTIZ_PANEL_D_XLIM,
                                 must_include=[log_mbh[selected], *lit_log_mbh],
                                 axis_name='panel d x-axis')
    y_lo, y_hi = lock_axis_range(*MULTIZ_PANEL_D_YLIM,
                                 must_include=[log_lbol[selected], *lit_log_lbol],
                                 axis_name='panel d y-axis')
    if xlim is not None:
        x_lo, x_hi = xlim
    if ylim is not None:
        y_lo, y_hi = ylim
    if range_only:
        return None, (x_lo, x_hi), (y_lo, y_hi)

    x_ref = np.linspace(x_lo, x_hi, 200)
    line_rot = _line_rotation_deg(1.0, x_lo, x_hi, y_lo, y_hi)
    for lam in EDDINGTON_RATIO_LINES:
        y_line = eddington_luminosity(x_ref) + np.log10(lam)
        ax.plot(x_ref, y_line, color='#777777', lw=0.9, ls='--', zorder=2)
        x_lab = x_lo + 0.60 * (x_hi - x_lo)
        y_lab = eddington_luminosity(x_lab) + np.log10(lam)
        if y_lo < y_lab < y_hi:
            ax.text(x_lab, y_lab - 0.12, rf'$\lambda_{{\rm Edd}} = {lam:g}$',
                    fontsize=6.5, color='#555555', ha='left', va='top',
                    rotation=line_rot, rotation_mode='anchor')

    bg = ~(lrd_red | lrd_blue) if show_lrd else np.ones(len(log_mbh), dtype=bool)
    _bg_scatter_and_contours(ax, log_mbh[bg], log_lbol[bg], x_lo, x_hi, y_lo, y_hi, contour_color='#4A4A8A')

    if show_lrd:
        if lrd_blue.sum() > 0:
            ax.scatter(log_mbh[lrd_blue], log_lbol[lrd_blue], s=10,
                       color='#F57C00', edgecolors='white', linewidths=0.2, zorder=5)
        if lrd_red.sum() > 0:
            ax.scatter(log_mbh[lrd_red], log_lbol[lrd_red], s=10,
                       color='#C62828', edgecolors='white', linewidths=0.2, zorder=6)

    lit_labels = []
    if show_lit:
        if pang_m.any():
            t_log_lbol = np.log10(PANG26['lambda_edd'][pang_m]) + eddington_luminosity(PANG26['log_mbh'][pang_m])
            t_lbol_err = PANG26['lambda_edd_err'][pang_m] / (PANG26['lambda_edd'][pang_m] * np.log(10))
            plot_lit_points(ax, 'Pang+26', PANG26['log_mbh'][pang_m], t_log_lbol,
                            xerr=PANG26['log_mbh_err'][pang_m], yerr=t_lbol_err)
            lit_labels.append('Pang+26')

        if mathee_m.any():
            m_log_lbol = np.log10(MATHEE24['lbol_1e44'][mathee_m]) + 44.0
            m_lbol_err = MATHEE24['lbol_1e44_err'][mathee_m] / (MATHEE24['lbol_1e44'][mathee_m] * np.log(10))
            plot_lit_points(ax, 'Mathee+24', MATHEE24['log_mbh'][mathee_m], m_log_lbol,
                            xerr=MATHEE24['log_mbh_err'][mathee_m], yerr=m_lbol_err)
            lit_labels.append('Mathee+24')

        if lin_m.any():
            plot_lit_points(ax, 'Lin+25', LIN25['log_mbh'][lin_m], LIN25['log_lbol'][lin_m],
                            xerr=LIN25['log_mbh_err'][lin_m], yerr=LIN25['log_lbol_err'][lin_m])
            lit_labels.append('Lin+25')

        if furtak_m:
            plot_lit_points(ax, 'Furtak+23', [FURTAK23['log_mbh']], [FURTAK23['log_lbol']],
                            xerr=[[FURTAK23['log_mbh_err_lo']], [FURTAK23['log_mbh_err_hi']]],
                            yerr=[[FURTAK23['log_lbol_err']], [FURTAK23['log_lbol_err']]])
            lit_labels.append('Furtak+23')

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.text(0.95, 0.05, _fmt_z(z_target), transform=ax.transAxes,
            ha='right', va='bottom', fontsize=10)

    handles = []
    if show_lrd:
        handles += [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#C62828',
                   markersize=7, label=r'LRD ($f_{\rm BH}\geq 3\%$)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#F57C00',
                   markersize=7, label=r'LRD ($f_{\rm BH}<3\%$)'),
        ]
    handles += lit_legend_handles(lit_labels)
    return handles, (x_lo, x_hi), (y_lo, y_hi)


def draw_panel_e(ax, data, z_target, show_lrd=True, bhar_floor=LRD_BHAR_DEFAULT, show_lit=True,
                 xlim=None, ylim=None, range_only=False, mask_seeds=True):
    bh_mass, mdot, medd, star, seed = (data['bh_mass'], data['mdot_msun_yr'],
                                        data['mdot_edd'], data['stellar_mass'],
                                        data['seed_mass'])
    if len(bh_mass) == 0:
        if not range_only:
            _no_data_panel(ax, xlim or MULTIZ_PANEL_E_XLIM, ylim or MULTIZ_PANEL_E_YLIM, z_target)
        return None, (xlim or MULTIZ_PANEL_E_XLIM), (ylim or MULTIZ_PANEL_E_YLIM)

    valid = _valid_mask(bh_mass, mdot, star)
    if mask_seeds:
        valid &= mask_ungrown_seeds(bh_mass, seed)
    bh_mass, mdot, medd, star = bh_mass[valid], mdot[valid], medd[valid], star[valid]
    log_mbh = np.log10(bh_mass)
    m1450 = m1450_from_lbol(lbol_from_mdot(mdot))

    _, lrd_red, lrd_blue = compute_selection(bh_mass, mdot, medd, star, bhar_floor)

    selected = (lrd_red | lrd_blue) if show_lrd else np.zeros(len(log_mbh), dtype=bool)

    mathee_m = _lit_z_mask(MATHEE24['z'], z_target)
    labbe_m  = _lit_z_mask(LABBE25['z'], z_target)
    furtak_m = _lit_z_mask(FURTAK23['z'], z_target)

    lit_muv, lit_mbh = [], []
    labbe_mbh_min = None
    if show_lit:
        labbe_mbh_min = bh_mass_min_from_lbol(LABBE25['log_lbol'][labbe_m])
        lit_muv += [LABBE25['m1450'][labbe_m], MATHEE24['muv'][mathee_m]]
        lit_mbh += [np.log10(labbe_mbh_min), MATHEE24['log_mbh'][mathee_m]]
        if furtak_m:
            lit_muv.append([FURTAK23['muv']])
            lit_mbh.append([FURTAK23['log_mbh']])

    x_faint, x_bright = lock_axis_range(*MULTIZ_PANEL_E_XLIM,
                                       must_include=[m1450[selected], *lit_muv],
                                       axis_name='panel e x-axis')
    y_lo, y_hi = lock_axis_range(*MULTIZ_PANEL_E_YLIM,
                                 must_include=[log_mbh[selected], *lit_mbh],
                                 axis_name='panel e y-axis')
    if xlim is not None:
        x_faint, x_bright = xlim
    if ylim is not None:
        y_lo, y_hi = ylim
    if range_only:
        return None, (x_faint, x_bright), (y_lo, y_hi)

    bg = ~(lrd_red | lrd_blue) if show_lrd else np.ones(len(log_mbh), dtype=bool)
    _bg_scatter_and_contours(ax, m1450[bg], log_mbh[bg], x_faint, x_bright, y_lo, y_hi, contour_color='#4A4A8A')

    if show_lrd:
        if lrd_blue.sum() > 0:
            ax.scatter(m1450[lrd_blue], log_mbh[lrd_blue], s=10,
                       color='#F57C00', edgecolors='white', linewidths=0.2, zorder=5)
        if lrd_red.sum() > 0:
            ax.scatter(m1450[lrd_red], log_mbh[lrd_red], s=10,
                       color='#C62828', edgecolors='white', linewidths=0.2, zorder=6)

    lit_labels = []
    if show_lit:
        if mathee_m.any():
            plot_lit_points(ax, 'Mathee+24', MATHEE24['muv'][mathee_m], MATHEE24['log_mbh'][mathee_m],
                            xerr=MATHEE24['muv_err'][mathee_m], yerr=MATHEE24['log_mbh_err'][mathee_m])
            lit_labels.append('Mathee+24')

        if furtak_m:
            plot_lit_points(ax, 'Furtak+23', [FURTAK23['muv']], [FURTAK23['log_mbh']],
                            xerr=FURTAK23['muv_err'],
                            yerr=[[FURTAK23['log_mbh_err_lo']], [FURTAK23['log_mbh_err_hi']]])
            lit_labels.append('Furtak+23')

        if labbe_m.any():
            plot_lit_points(ax, 'Labbe+25', LABBE25['m1450'][labbe_m], np.log10(labbe_mbh_min),
                            yerr=0.3, lolims=True)
            lit_labels.append('Labbe+25')

    ax.set_xlim(x_faint, x_bright)
    ax.set_ylim(y_lo, y_hi)
    ax.text(0.95, 0.05, _fmt_z(z_target), transform=ax.transAxes,
            ha='right', va='bottom', fontsize=10)

    handles = []
    if show_lrd:
        handles += [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#C62828',
                   markersize=7, label=r'LRD ($f_{\rm BH}\geq 3\%$)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#F57C00',
                   markersize=7, label=r'LRD ($f_{\rm BH}<3\%$)'),
        ]
    handles += lit_legend_handles(lit_labels)
    return handles, (x_faint, x_bright), (y_lo, y_hi)


def draw_panel_f(ax, data, z_target, volume_h3, show_lrd=True,
                 bhar_floor=LRD_BHAR_DEFAULT, n_bins=30, h_h=None, show_lit=True,
                 xlim=None, ylim=None, range_only=False, mask_seeds=True):
    bh_mass, mdot, medd, star, seed = (data['bh_mass'], data['mdot_msun_yr'],
                                        data['mdot_edd'], data['stellar_mass'],
                                        data['seed_mass'])
    if len(bh_mass) == 0:
        if not range_only:
            _no_data_panel(ax, xlim or MULTIZ_PANEL_F_XLIM, ylim or MULTIZ_PANEL_F_YLIM, z_target)
        return None, (xlim or MULTIZ_PANEL_F_XLIM), (ylim or MULTIZ_PANEL_F_YLIM)

    valid = _valid_mask(bh_mass, mdot, star)
    if mask_seeds:
        valid &= mask_ungrown_seeds(bh_mass, seed)
    bh_mass, mdot, medd, star = bh_mass[valid], mdot[valid], medd[valid], star[valid]
    log_lbol = lbol_from_mdot(mdot)
    _, lrd_red, lrd_blue = compute_selection(bh_mass, mdot, medd, star, bhar_floor)

    selected = (lrd_red | lrd_blue) if show_lrd else np.zeros(len(log_lbol), dtype=bool)
    lo, hi = lock_axis_range(*MULTIZ_PANEL_F_XLIM, must_include=[log_lbol[selected]],
                             axis_name='panel f x-axis')
    if xlim is not None:
        lo, hi = xlim

    bins = np.linspace(lo, hi, n_bins + 1)
    bw = bins[1] - bins[0]
    centres = 0.5 * (bins[:-1] + bins[1:])

    cats = [(log_lbol, 'Total', 'k', 'o')]
    if show_lrd:
        cats.append((log_lbol[lrd_red], r'LRD ($f_{\rm BH}\geq 3\%$)', '#C62828', 'D'))
        cats.append((log_lbol[lrd_blue], r'LRD ($f_{\rm BH}<3\%$)', '#F57C00', 's'))

    allv = []
    plot_data = []
    for values, label, color, marker in cats:
        counts, _ = np.histogram(values, bins=bins)
        pos = counts > 0
        y = counts / (bw * volume_h3) if volume_h3 else counts / bw
        logy = np.log10(y[pos])
        logy_err = 1.0 / (np.sqrt(counts[pos]) * np.log(10))
        allv.extend(logy)
        plot_data.append((centres[pos], logy, logy_err, label, color, marker))

    y_lo, y_hi = lock_axis_range(*MULTIZ_PANEL_F_YLIM, must_include=[np.array(allv)],
                                 axis_name='panel f y-axis')
    if ylim is not None:
        y_lo, y_hi = ylim
    if range_only:
        return None, (lo, hi), (y_lo, y_hi)

    for centres_pos, logy, logy_err, label, color, marker in plot_data:
        ax.errorbar(centres_pos, logy, yerr=logy_err, fmt=marker, color=color,
                    mec='black', mew=0.4, ms=4, capsize=1.5, elinewidth=0.7,
                    ls='none', label=label, zorder=5)

    # Shen et al. (2020) bolometric QLF model (global fit A), z = 1-7 only
    # -- see the single-panel plot_panel_f() for why the curve is clipped
    # to SHEN20_MODEL_LOGLBOL_MIN and kept out of the axis-lock above.
    z_num = _z_numeric(z_target)
    show_shen20 = show_lit and (1.0 <= z_num <= 7.0)
    if show_shen20:
        l_ref = np.linspace(max(lo, SHEN20_MODEL_LOGLBOL_MIN), hi, 200)
        shen20_logphi = shen20_bolometric_qlf_logphi(l_ref, z_num, h_h=h_h)
        ax.plot(l_ref, shen20_logphi, color='#000000', lw=1.2, ls='--', zorder=4)

    ax.set_xlim(lo, hi)
    ax.set_ylim(y_lo, y_hi)
    ax.text(0.95, 0.90, _fmt_z(z_target), transform=ax.transAxes,
            ha='right', va='top', fontsize=10)

    handles = [Line2D([0], [0], marker=m, color='w', markerfacecolor=c,
                      markeredgecolor='black', markersize=6, linestyle='none', label=l)
               for _, l, c, m in cats]
    if show_shen20:
        handles.append(Line2D([0], [0], color='#000000', lw=1.2, ls='--', label='Shen+20'))
    return (handles,
            (lo, hi), (y_lo, y_hi))

# ============================================================================
# PER-PANEL DRAWING -- COMPARE MODE (multi-run overlay)
# ============================================================================
# Same contours-only-for-scatter-panels / linestyle-for-panel-f design as
# bh_lrd_analysis.py's own compare functions (see the comment above those).
# `datasets` here is a list of {'data', 'style'} dicts -- one read_epoch()
# catalogue + a run_style style per run, for the SAME redshift bin.
# make_grid() doesn't care whether snap_data[z] is a single data dict or one
# of these lists -- it just forwards it to draw_fn -- so these compare
# functions plug into the existing make_grid() unchanged; no separate grid
# assembly function is needed.

def draw_panel_a_compare(ax, datasets, z_target, show_lit=True, bhar_floor=LRD_BHAR_DEFAULT,
                         xlim=None, ylim=None, range_only=False, mask_seeds=True):
    prepped = []
    for ds in datasets:
        data, style = ds['data'], ds['style']
        bh_mass, mdot, seed = data['bh_mass'], data['mdot_msun_yr'], data['seed_mass']
        if len(bh_mass) == 0:
            continue
        valid = _valid_mask(bh_mass, mdot)
        if mask_seeds:
            valid &= mask_ungrown_seeds(bh_mass, seed)
        if valid.sum() == 0:
            continue
        prepped.append((np.log10(bh_mass[valid]), np.log10(mdot[valid]), style))

    pang_m = _lit_z_mask(PANG26['z'], z_target)
    mathee_m = _lit_z_mask(MATHEE24['z'], z_target)
    lin_m = _lit_z_mask(LIN25['z'], z_target)
    lit_log_mbh, lit_log_mdot = [], []
    if show_lit:
        t_mdot = PANG26['lambda_edd'][pang_m] * 10**eddington_mdot(PANG26['log_mbh'][pang_m])
        lit_log_mbh.append(PANG26['log_mbh'][pang_m]); lit_log_mdot.append(np.log10(t_mdot))
        m_mdot = mdot_from_lbol(np.log10(MATHEE24['lbol_1e44'][mathee_m]) + 44.0)
        lit_log_mbh.append(MATHEE24['log_mbh'][mathee_m]); lit_log_mdot.append(np.log10(m_mdot))
        l_mdot = mdot_from_lbol(LIN25['log_lbol'][lin_m])
        lit_log_mbh.append(LIN25['log_mbh'][lin_m]); lit_log_mdot.append(np.log10(l_mdot))

    x_lo, x_hi = lock_axis_range(*MULTIZ_PANEL_A_XLIM, must_include=lit_log_mbh,
                                 axis_name='panel a x-axis (compare)')
    y_lo, y_hi = lock_axis_range(*MULTIZ_PANEL_A_YLIM, must_include=lit_log_mdot,
                                 axis_name='panel a y-axis (compare)')
    if xlim is not None:
        x_lo, x_hi = xlim
    if ylim is not None:
        y_lo, y_hi = ylim
    if range_only:
        return None, (x_lo, x_hi), (y_lo, y_hi)
    if not prepped:
        _no_data_panel(ax, (x_lo, x_hi), (y_lo, y_hi), z_target)
        return None, (x_lo, x_hi), (y_lo, y_hi)

    x_ref = np.linspace(x_lo, x_hi, 200)
    y_edd = eddington_mdot(x_ref)
    ax.plot(x_ref, y_edd, color='#C62828', lw=1.1, zorder=4)
    ax.plot(x_ref, y_edd + 1.0, color='#E65100', lw=1.1, zorder=4)
    ax.axhline(np.log10(LRD_BHAR_DEFAULT), color='#C62828', lw=0.8, ls='--', zorder=3, alpha=0.85)
    ax.axhline(np.log10(LRD_BHAR_ALT), color='#C62828', lw=0.6, ls=':', zorder=3, alpha=0.70)

    run_handles = draw_contours_multirun(ax, prepped, x_lo, x_hi, y_lo, y_hi,
                                        linewidths=(0.5, 0.8, 1.1), n_kde=N_KDE_GRID)

    lit_labels = []
    if show_lit:
        if pang_m.any():
            t_mdot = PANG26['lambda_edd'][pang_m] * 10**eddington_mdot(PANG26['log_mbh'][pang_m])
            t_mdot_lo = (PANG26['lambda_edd'][pang_m] - PANG26['lambda_edd_err'][pang_m]) * \
                10**eddington_mdot(PANG26['log_mbh'][pang_m])
            plot_lit_points(ax, 'Pang+26', PANG26['log_mbh'][pang_m], np.log10(t_mdot),
                            xerr=PANG26['log_mbh_err'][pang_m],
                            yerr=[np.log10(t_mdot) - np.log10(np.maximum(t_mdot_lo, t_mdot * 1e-3)),
                                  np.full_like(t_mdot, 0.15)])
            lit_labels.append('Pang+26')
        if mathee_m.any():
            m_log_mdot = np.log10(mdot_from_lbol(np.log10(MATHEE24['lbol_1e44'][mathee_m]) + 44.0))
            plot_lit_points(ax, 'Mathee+24', MATHEE24['log_mbh'][mathee_m], m_log_mdot,
                            xerr=MATHEE24['log_mbh_err'][mathee_m],
                            yerr=MATHEE24['lbol_1e44_err'][mathee_m] / (MATHEE24['lbol_1e44'][mathee_m] * np.log(10)))
            lit_labels.append('Mathee+24')
        if lin_m.any():
            l_log_mdot = np.log10(mdot_from_lbol(LIN25['log_lbol'][lin_m]))
            plot_lit_points(ax, 'Lin+25', LIN25['log_mbh'][lin_m], l_log_mdot,
                            xerr=LIN25['log_mbh_err'][lin_m], yerr=LIN25['log_lbol_err'][lin_m])
            lit_labels.append('Lin+25')

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xticks(np.arange(int(np.ceil(x_lo)), int(np.floor(x_hi)) + 1, 2))
    ax.text(0.95, 0.05, _fmt_z(z_target), transform=ax.transAxes, ha='right', va='bottom', fontsize=10)

    handles = [
        Line2D([0], [0], color='#C62828', lw=1.6, label=r'$\dot{M}_{\rm BH} = \dot{M}_{\rm Edd}$'),
        Line2D([0], [0], color='#E65100', lw=1.6, label=r'$\dot{M}_{\rm BH} = 10\,\dot{M}_{\rm Edd}$'),
    ] + run_handles + lit_legend_handles(lit_labels)
    return handles, (x_lo, x_hi), (y_lo, y_hi)


def draw_panel_b_compare(ax, datasets, z_target, show_lit=True, bhar_floor=LRD_BHAR_DEFAULT,
                         xlim=None, ylim=None, range_only=False, mask_seeds=True):
    prepped = []
    for ds in datasets:
        data, style = ds['data'], ds['style']
        bh_mass, mdot, star, seed = (data['bh_mass'], data['mdot_msun_yr'],
                                     data['stellar_mass'], data['seed_mass'])
        if len(bh_mass) == 0:
            continue
        valid = _valid_mask(bh_mass, mdot, star)
        if mask_seeds:
            valid &= mask_ungrown_seeds(bh_mass, seed)
        if valid.sum() == 0:
            continue
        f_bh = bh_mass[valid] / star[valid]
        prepped.append((np.log10(bh_mass[valid]), np.log10(f_bh), style))

    pang_m = _lit_z_mask(PANG26['z'], z_target)
    lit_log_mbh = PANG26['log_mbh'][pang_m] if show_lit else []
    lit_log_fbh = (PANG26['log_mbh'][pang_m] - PANG26['log_mstar'][pang_m]) if show_lit else []

    x_lo, x_hi = lock_axis_range(*MULTIZ_PANEL_B_XLIM, must_include=[lit_log_mbh],
                                 axis_name='panel b x-axis (compare)')
    y_lo, y_hi = lock_axis_range(*MULTIZ_PANEL_B_YLIM, must_include=[lit_log_fbh],
                                 axis_name='panel b y-axis (compare)')
    if xlim is not None:
        x_lo, x_hi = xlim
    if ylim is not None:
        y_lo, y_hi = ylim
    if range_only:
        return None, (x_lo, x_hi), (y_lo, y_hi)
    if not prepped:
        _no_data_panel(ax, (x_lo, x_hi), (y_lo, y_hi), z_target)
        return None, (x_lo, x_hi), (y_lo, y_hi)

    log_fbh_thresh = np.log10(LRD_FBHM_THRESH)
    ax.axhline(np.log10(0.1), color='#F57C00', lw=1.4, zorder=4)
    ax.axhline(log_fbh_thresh, color='#C62828', lw=1.4, zorder=4)

    run_handles = draw_contours_multirun(ax, prepped, x_lo, x_hi, y_lo, y_hi,
                                        linewidths=(0.5, 0.8, 1.1), n_kde=N_KDE_GRID)

    lit_labels = []
    if show_lit and pang_m.any():
        t_log_fbh = PANG26['log_mbh'][pang_m] - PANG26['log_mstar'][pang_m]
        t_fbh_err = np.sqrt(PANG26['log_mbh_err'][pang_m]**2 + PANG26['log_mstar_err'][pang_m]**2)
        plot_lit_points(ax, 'Pang+26', PANG26['log_mbh'][pang_m], t_log_fbh,
                        xerr=PANG26['log_mbh_err'][pang_m], yerr=t_fbh_err)
        lit_labels.append('Pang+26')

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xticks(np.arange(int(np.ceil(x_lo)), int(np.floor(x_hi)) + 1, 2))
    ax.text(0.95, 0.05, _fmt_z(z_target), transform=ax.transAxes, ha='right', va='bottom', fontsize=10)
    return run_handles + lit_legend_handles(lit_labels), (x_lo, x_hi), (y_lo, y_hi)


def draw_panel_c_compare(ax, datasets, z_target, show_lit=True, bhar_floor=LRD_BHAR_DEFAULT,
                         xlim=None, ylim=None, range_only=False, mask_seeds=True):
    prepped = []
    for ds in datasets:
        data, style = ds['data'], ds['style']
        bh_mass, mdot, star, seed = (data['bh_mass'], data['mdot_msun_yr'],
                                     data['stellar_mass'], data['seed_mass'])
        if len(bh_mass) == 0:
            continue
        valid = _valid_mask(bh_mass, mdot, star)
        if mask_seeds:
            valid &= mask_ungrown_seeds(bh_mass, seed)
        if valid.sum() == 0:
            continue
        prepped.append((np.log10(star[valid]), np.log10(bh_mass[valid]), style))

    pang_m = _lit_z_mask(PANG26['z'], z_target)
    furtak_m = _lit_z_mask(FURTAK23['z'], z_target)
    lit_mstar, lit_mbh = [], []
    if show_lit:
        lit_mstar.append(PANG26['log_mstar'][pang_m])
        lit_mbh.append(PANG26['log_mbh'][pang_m])
        if furtak_m:
            lit_mstar.append([FURTAK23['log_mstar_upper_limit']])
            lit_mbh.append([FURTAK23['log_mbh']])

    x_lo, x_hi = lock_axis_range(*MULTIZ_PANEL_C_XLIM, must_include=lit_mstar,
                                 axis_name='panel c x-axis (compare)')
    y_lo, y_hi = lock_axis_range(*MULTIZ_PANEL_C_YLIM, must_include=lit_mbh,
                                 axis_name='panel c y-axis (compare)')
    if xlim is not None:
        x_lo, x_hi = xlim
    if ylim is not None:
        y_lo, y_hi = ylim
    if range_only:
        return None, (x_lo, x_hi), (y_lo, y_hi)
    if not prepped:
        _no_data_panel(ax, (x_lo, x_hi), (y_lo, y_hi), z_target)
        return None, (x_lo, x_hi), (y_lo, y_hi)

    x_ref = np.linspace(x_lo, x_hi, 200)
    y_kh = kormendy_ho_mbh(x_ref)
    ax.fill_between(x_ref, y_kh - KORMENDY_HO_SCATTER, y_kh + KORMENDY_HO_SCATTER,
                    color='#999999', alpha=0.35, zorder=0)
    kh_line, = ax.plot(x_ref, y_kh, color='black', lw=1.4, zorder=2, label='Kormendy & Ho 2013')

    ratio_rot = _line_rotation_deg(1.0, x_lo, x_hi, y_lo, y_hi)
    for ratio in MASS_RATIO_LINES:
        y_ratio = x_ref + np.log10(ratio)
        ax.plot(x_ref, y_ratio, color='black', lw=0.8, ls='--', zorder=2)
        x_lab = x_lo + 0.12 * (x_hi - x_lo)
        y_lab = x_lab + np.log10(ratio)
        if y_lo < y_lab < y_hi:
            ax.text(x_lab, y_lab - 0.12, rf'$M_{{\rm BH}}/M_\star = {ratio:g}$',
                    fontsize=6.5, color='black', ha='left', va='top',
                    rotation=ratio_rot, rotation_mode='anchor')

    run_handles = draw_contours_multirun(ax, prepped, x_lo, x_hi, y_lo, y_hi,
                                        linewidths=(0.5, 0.8, 1.1), n_kde=N_KDE_GRID)

    lit_labels = []
    if show_lit:
        if pang_m.any():
            plot_lit_points(ax, 'Pang+26', PANG26['log_mstar'][pang_m], PANG26['log_mbh'][pang_m],
                            xerr=PANG26['log_mstar_err'][pang_m], yerr=PANG26['log_mbh_err'][pang_m])
            lit_labels.append('Pang+26')
        if furtak_m:
            plot_lit_points(ax, 'Furtak+23', [FURTAK23['log_mstar_upper_limit']], [FURTAK23['log_mbh']],
                            xerr=[[0.4], [0.0]],
                            yerr=[[FURTAK23['log_mbh_err_lo']], [FURTAK23['log_mbh_err_hi']]],
                            xuplims=True)
            lit_labels.append('Furtak+23')

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.text(0.95, 0.05, _fmt_z(z_target), transform=ax.transAxes, ha='right', va='bottom', fontsize=10)
    handles = [kh_line] + run_handles + lit_legend_handles(lit_labels)
    return handles, (x_lo, x_hi), (y_lo, y_hi)


def draw_panel_d_compare(ax, datasets, z_target, show_lit=True, bhar_floor=LRD_BHAR_DEFAULT,
                         xlim=None, ylim=None, range_only=False, mask_seeds=True):
    prepped = []
    for ds in datasets:
        data, style = ds['data'], ds['style']
        bh_mass, mdot, star, seed = (data['bh_mass'], data['mdot_msun_yr'],
                                     data['stellar_mass'], data['seed_mass'])
        if len(bh_mass) == 0:
            continue
        valid = _valid_mask(bh_mass, mdot, star)
        if mask_seeds:
            valid &= mask_ungrown_seeds(bh_mass, seed)
        if valid.sum() == 0:
            continue
        prepped.append((np.log10(bh_mass[valid]), lbol_from_mdot(mdot[valid]), style))

    pang_m = _lit_z_mask(PANG26['z'], z_target)
    mathee_m = _lit_z_mask(MATHEE24['z'], z_target)
    lin_m = _lit_z_mask(LIN25['z'], z_target)
    furtak_m = _lit_z_mask(FURTAK23['z'], z_target)
    lit_log_mbh, lit_log_lbol = [], []
    if show_lit:
        lit_log_mbh.append(PANG26['log_mbh'][pang_m])
        lit_log_lbol.append(np.log10(PANG26['lambda_edd'][pang_m]) + eddington_luminosity(PANG26['log_mbh'][pang_m]))
        lit_log_mbh.append(MATHEE24['log_mbh'][mathee_m])
        lit_log_lbol.append(np.log10(MATHEE24['lbol_1e44'][mathee_m]) + 44.0)
        lit_log_mbh.append(LIN25['log_mbh'][lin_m])
        lit_log_lbol.append(LIN25['log_lbol'][lin_m])
        if furtak_m:
            lit_log_mbh.append([FURTAK23['log_mbh']])
            lit_log_lbol.append([FURTAK23['log_lbol']])

    x_lo, x_hi = lock_axis_range(*MULTIZ_PANEL_D_XLIM, must_include=lit_log_mbh,
                                 axis_name='panel d x-axis (compare)')
    y_lo, y_hi = lock_axis_range(*MULTIZ_PANEL_D_YLIM, must_include=lit_log_lbol,
                                 axis_name='panel d y-axis (compare)')
    if xlim is not None:
        x_lo, x_hi = xlim
    if ylim is not None:
        y_lo, y_hi = ylim
    if range_only:
        return None, (x_lo, x_hi), (y_lo, y_hi)
    if not prepped:
        _no_data_panel(ax, (x_lo, x_hi), (y_lo, y_hi), z_target)
        return None, (x_lo, x_hi), (y_lo, y_hi)

    x_ref = np.linspace(x_lo, x_hi, 200)
    line_rot = _line_rotation_deg(1.0, x_lo, x_hi, y_lo, y_hi)
    for lam in EDDINGTON_RATIO_LINES:
        y_line = eddington_luminosity(x_ref) + np.log10(lam)
        ax.plot(x_ref, y_line, color='#777777', lw=0.9, ls='--', zorder=2)
        x_lab = x_lo + 0.60 * (x_hi - x_lo)
        y_lab = eddington_luminosity(x_lab) + np.log10(lam)
        if y_lo < y_lab < y_hi:
            ax.text(x_lab, y_lab - 0.12, rf'$\lambda_{{\rm Edd}} = {lam:g}$',
                    fontsize=6.5, color='#555555', ha='left', va='top',
                    rotation=line_rot, rotation_mode='anchor')

    run_handles = draw_contours_multirun(ax, prepped, x_lo, x_hi, y_lo, y_hi,
                                        linewidths=(0.5, 0.8, 1.1), n_kde=N_KDE_GRID)

    lit_labels = []
    if show_lit:
        if pang_m.any():
            t_log_lbol = np.log10(PANG26['lambda_edd'][pang_m]) + eddington_luminosity(PANG26['log_mbh'][pang_m])
            t_lbol_err = PANG26['lambda_edd_err'][pang_m] / (PANG26['lambda_edd'][pang_m] * np.log(10))
            plot_lit_points(ax, 'Pang+26', PANG26['log_mbh'][pang_m], t_log_lbol,
                            xerr=PANG26['log_mbh_err'][pang_m], yerr=t_lbol_err)
            lit_labels.append('Pang+26')
        if mathee_m.any():
            m_log_lbol = np.log10(MATHEE24['lbol_1e44'][mathee_m]) + 44.0
            m_lbol_err = MATHEE24['lbol_1e44_err'][mathee_m] / (MATHEE24['lbol_1e44'][mathee_m] * np.log(10))
            plot_lit_points(ax, 'Mathee+24', MATHEE24['log_mbh'][mathee_m], m_log_lbol,
                            xerr=MATHEE24['log_mbh_err'][mathee_m], yerr=m_lbol_err)
            lit_labels.append('Mathee+24')
        if lin_m.any():
            plot_lit_points(ax, 'Lin+25', LIN25['log_mbh'][lin_m], LIN25['log_lbol'][lin_m],
                            xerr=LIN25['log_mbh_err'][lin_m], yerr=LIN25['log_lbol_err'][lin_m])
            lit_labels.append('Lin+25')
        if furtak_m:
            plot_lit_points(ax, 'Furtak+23', [FURTAK23['log_mbh']], [FURTAK23['log_lbol']],
                            xerr=[[FURTAK23['log_mbh_err_lo']], [FURTAK23['log_mbh_err_hi']]],
                            yerr=[[FURTAK23['log_lbol_err']], [FURTAK23['log_lbol_err']]])
            lit_labels.append('Furtak+23')

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.text(0.95, 0.05, _fmt_z(z_target), transform=ax.transAxes, ha='right', va='bottom', fontsize=10)
    handles = run_handles + lit_legend_handles(lit_labels)
    return handles, (x_lo, x_hi), (y_lo, y_hi)


def draw_panel_e_compare(ax, datasets, z_target, show_lit=True, bhar_floor=LRD_BHAR_DEFAULT,
                         xlim=None, ylim=None, range_only=False, mask_seeds=True):
    prepped = []
    for ds in datasets:
        data, style = ds['data'], ds['style']
        bh_mass, mdot, star, seed = (data['bh_mass'], data['mdot_msun_yr'],
                                     data['stellar_mass'], data['seed_mass'])
        if len(bh_mass) == 0:
            continue
        valid = _valid_mask(bh_mass, mdot, star)
        if mask_seeds:
            valid &= mask_ungrown_seeds(bh_mass, seed)
        if valid.sum() == 0:
            continue
        m1450 = m1450_from_lbol(lbol_from_mdot(mdot[valid]))
        prepped.append((m1450, np.log10(bh_mass[valid]), style))

    mathee_m = _lit_z_mask(MATHEE24['z'], z_target)
    labbe_m = _lit_z_mask(LABBE25['z'], z_target)
    furtak_m = _lit_z_mask(FURTAK23['z'], z_target)
    lit_muv, lit_mbh = [], []
    labbe_mbh_min = None
    if show_lit:
        labbe_mbh_min = bh_mass_min_from_lbol(LABBE25['log_lbol'][labbe_m])
        lit_muv += [LABBE25['m1450'][labbe_m], MATHEE24['muv'][mathee_m]]
        lit_mbh += [np.log10(labbe_mbh_min), MATHEE24['log_mbh'][mathee_m]]
        if furtak_m:
            lit_muv.append([FURTAK23['muv']])
            lit_mbh.append([FURTAK23['log_mbh']])

    x_faint, x_bright = lock_axis_range(*MULTIZ_PANEL_E_XLIM, must_include=lit_muv,
                                        axis_name='panel e x-axis (compare)')
    y_lo, y_hi = lock_axis_range(*MULTIZ_PANEL_E_YLIM, must_include=lit_mbh,
                                 axis_name='panel e y-axis (compare)')
    if xlim is not None:
        x_faint, x_bright = xlim
    if ylim is not None:
        y_lo, y_hi = ylim
    if range_only:
        return None, (x_faint, x_bright), (y_lo, y_hi)
    if not prepped:
        _no_data_panel(ax, (x_faint, x_bright), (y_lo, y_hi), z_target)
        return None, (x_faint, x_bright), (y_lo, y_hi)

    run_handles = draw_contours_multirun(ax, prepped, x_faint, x_bright, y_lo, y_hi,
                                        linewidths=(0.5, 0.8, 1.1), n_kde=N_KDE_GRID)

    lit_labels = []
    if show_lit:
        if mathee_m.any():
            plot_lit_points(ax, 'Mathee+24', MATHEE24['muv'][mathee_m], MATHEE24['log_mbh'][mathee_m],
                            xerr=MATHEE24['muv_err'][mathee_m], yerr=MATHEE24['log_mbh_err'][mathee_m])
            lit_labels.append('Mathee+24')
        if furtak_m:
            plot_lit_points(ax, 'Furtak+23', [FURTAK23['muv']], [FURTAK23['log_mbh']],
                            xerr=FURTAK23['muv_err'],
                            yerr=[[FURTAK23['log_mbh_err_lo']], [FURTAK23['log_mbh_err_hi']]])
            lit_labels.append('Furtak+23')
        if labbe_m.any():
            plot_lit_points(ax, 'Labbe+25', LABBE25['m1450'][labbe_m], np.log10(labbe_mbh_min),
                            yerr=0.3, lolims=True)
            lit_labels.append('Labbe+25')

    ax.set_xlim(x_faint, x_bright)
    ax.set_ylim(y_lo, y_hi)
    ax.text(0.95, 0.05, _fmt_z(z_target), transform=ax.transAxes, ha='right', va='bottom', fontsize=10)
    handles = run_handles + lit_legend_handles(lit_labels)
    return handles, (x_faint, x_bright), (y_lo, y_hi)


_PANEL_F_CAT_SPECS = [('Total', 'k', 'o'),
                     (r'LRD ($f_{\rm BH}\geq 3\%$)', '#C62828', 'D'),
                     (r'LRD ($f_{\rm BH}<3\%$)', '#F57C00', 's')]


def draw_panel_f_compare(ax, datasets, z_target, volume_h3, show_lit=True,
                         bhar_floor=LRD_BHAR_DEFAULT, n_bins=30, h_h=None,
                         xlim=None, ylim=None, range_only=False, mask_seeds=True):
    """Panel f is a luminosity function (histogram/errorbar), not a scatter
    plot -- unlike panels a-e it keeps its Total/LRD-red/LRD-blue category
    split in compare mode, distinguished by linestyle per run (like
    allresults-blackholes.py's compare functions), not by contour color."""
    per_run = []
    for ds in datasets:
        data, style = ds['data'], ds['style']
        bh_mass, mdot, medd, star, seed = (data['bh_mass'], data['mdot_msun_yr'],
                                           data['mdot_edd'], data['stellar_mass'],
                                           data['seed_mass'])
        if len(bh_mass) == 0:
            per_run.append((None, style)); continue
        valid = _valid_mask(bh_mass, mdot, star)
        if mask_seeds:
            valid &= mask_ungrown_seeds(bh_mass, seed)
        if valid.sum() == 0:
            per_run.append((None, style)); continue
        bh_mass, mdot, medd, star = bh_mass[valid], mdot[valid], medd[valid], star[valid]
        log_lbol = lbol_from_mdot(mdot)
        _, lrd_red, lrd_blue = compute_selection(bh_mass, mdot, medd, star, bhar_floor)
        per_run.append(({'log_lbol': log_lbol, 'lrd_red': lrd_red, 'lrd_blue': lrd_blue}, style))

    all_selected = [d['log_lbol'][d['lrd_red'] | d['lrd_blue']] for d, _ in per_run if d is not None]
    lo, hi = lock_axis_range(*MULTIZ_PANEL_F_XLIM, must_include=all_selected,
                             axis_name='panel f x-axis (compare)')
    if xlim is not None:
        lo, hi = xlim

    bins = np.linspace(lo, hi, n_bins + 1)
    bw = bins[1] - bins[0]
    centres = 0.5 * (bins[:-1] + bins[1:])

    allv = []
    plot_data = []
    for i, (d, style) in enumerate(per_run):
        if d is None:
            continue
        cats = [(d['log_lbol'], *_PANEL_F_CAT_SPECS[0]),
                (d['log_lbol'][d['lrd_red']],  *_PANEL_F_CAT_SPECS[1]),
                (d['log_lbol'][d['lrd_blue']], *_PANEL_F_CAT_SPECS[2])]
        for values, label, color, marker in cats:
            counts, _ = np.histogram(values, bins=bins)
            pos = counts > 0
            if not np.any(pos):
                continue
            y = counts / (bw * volume_h3) if volume_h3 else counts / bw
            logy = np.log10(y[pos])
            logy_err = 1.0 / (np.sqrt(counts[pos]) * np.log(10))
            allv.extend(logy)
            plot_data.append((centres[pos], logy, logy_err, label, color, marker, style, i))

    y_lo, y_hi = lock_axis_range(*MULTIZ_PANEL_F_YLIM,
                                 must_include=[np.array(allv)] if allv else [],
                                 axis_name='panel f y-axis (compare)')
    if ylim is not None:
        y_lo, y_hi = ylim
    if range_only:
        return None, (lo, hi), (y_lo, y_hi)
    if not plot_data:
        _no_data_panel(ax, (lo, hi), (y_lo, y_hi), z_target)
        return None, (lo, hi), (y_lo, y_hi)

    run_handles, seen_runs = [], set()
    for centres_pos, logy, logy_err, label, color, marker, style, i in plot_data:
        draw_color = lighten_color(color, style['lighten'])
        ax.errorbar(centres_pos, logy, yerr=logy_err, fmt=marker, color=draw_color,
                    mec='black', mew=0.4, ms=4, capsize=1.5, elinewidth=0.7,
                    ls=style['linestyle'], label=None, zorder=5)
        if i not in seen_runs:
            seen_runs.add(i)
            run_handles.append(Line2D([0], [0], color='black', lw=1.4,
                                      ls=style['linestyle'], label=style['label']))

    z_num = _z_numeric(z_target)
    show_shen20 = show_lit and (1.0 <= z_num <= 7.0)
    if show_shen20:
        l_ref = np.linspace(max(lo, SHEN20_MODEL_LOGLBOL_MIN), hi, 200)
        shen20_logphi = shen20_bolometric_qlf_logphi(l_ref, z_num, h_h=h_h)
        ax.plot(l_ref, shen20_logphi, color='#000000', lw=1.2, ls='--', zorder=4)

    ax.set_xlim(lo, hi)
    ax.set_ylim(y_lo, y_hi)
    ax.text(0.95, 0.90, _fmt_z(z_target), transform=ax.transAxes, ha='right', va='top', fontsize=10)

    cat_handles = [Line2D([0], [0], marker=m, color='w', markerfacecolor=c, markeredgecolor='black',
                          markersize=6, linestyle='none', label=l) for l, c, m in _PANEL_F_CAT_SPECS]
    handles = cat_handles + run_handles
    if show_shen20:
        handles.append(Line2D([0], [0], color='#000000', lw=1.2, ls='--', label='Shen+20'))
    return handles, (lo, hi), (y_lo, y_hi)


# ============================================================================
# GRID ASSEMBLY
# ============================================================================

PANEL_SPECS = {
    'a': dict(title=r'$\dot{M}_{\rm BH}$ vs $M_{\rm BH}$',
              xlabel=r'$\log\,M_{\rm BH}\ [M_\odot]$',
              ylabel=r'$\log\,\dot{M}_{\rm BH}\ [M_\odot\,\mathrm{yr}^{-1}]$',
              fname='lrd_bh_accretion_scatter_multiz.png'),
    'b': dict(title=r'$f_{\rm BH}$ vs $M_{\rm BH}$',
              xlabel=r'$\log\,M_{\rm BH}\ [M_\odot]$',
              ylabel=r'$f_{\rm BH} = M_{\rm BH}/M_\star$',
              fname='lrd_fbh_scatter_multiz.png'),
    'c': dict(title=r'$M_{\rm BH}$ vs $M_\star$',
              xlabel=r'$\log\,M_\star\ [M_\odot]$',
              ylabel=r'$\log\,M_{\rm BH}\ [M_\odot]$',
              fname='lrd_mbh_mstar_scatter_multiz.png'),
    'd': dict(title=r'$L_{\rm bol}$ vs $M_{\rm BH}$',
              xlabel=r'$\log\,M_{\rm BH}\ [M_\odot]$',
              ylabel=r'$\log\,L_{\rm bol}\ [{\rm erg\,s^{-1}}]$',
              fname='lrd_lbol_mbh_scatter_multiz.png'),
    'e': dict(title=r'$M_{1450}$ vs $M_{\rm BH}$',
              xlabel=r'$M_{\rm UV,1450}$',
              ylabel=r'$\log\,M_{\rm BH}\ [M_\odot]$',
              fname='lrd_m1450_mbh_scatter_multiz.png'),
    'f': dict(title='Bolometric AGN luminosity function',
              xlabel=r'$\log\,L_{\rm bol}\ [{\rm erg\,s^{-1}}]$',
              ylabel=r'$\log\,({\rm d}N/{\rm d}\log L_{\rm bol}\ /\ {\rm Mpc^{-3}}\,h^3)$',
              fname='lrd_bolometric_luminosity_function_multiz.png'),
}


def _blank_edge_ticklabel(ax, axis, target):
    """Blank the tick label nearest `target` on the given axis ('x' or
    'y'), used to remove one side of a duplicate label pair that collides
    at a touching-subplot seam (see the call site in make_grid).

    Installs a FixedLocator/FixedFormatter pinned to the CURRENT tick
    positions with just that one label blanked -- mutating the existing
    tick Text objects directly doesn't survive the formatter re-running
    at draw/save time, which would silently restore the label.
    """
    axis_obj = ax.xaxis if axis == 'x' else ax.yaxis
    ticks = np.asarray(axis_obj.get_majorticklocs())
    if len(ticks) == 0:
        return
    lo, hi = (ax.get_xlim() if axis == 'x' else ax.get_ylim())
    span = abs(hi - lo) or 1.0
    idx = int(np.argmin(np.abs(ticks - target)))
    if abs(ticks[idx] - target) > 0.02 * span:
        return
    labels = [f'{t:g}' for t in ticks]
    labels[idx] = ''
    axis_obj.set_major_locator(FixedLocator(ticks))
    axis_obj.set_major_formatter(FixedFormatter(labels))


def _union_lim(ranges, reversed_axis=False):
    """Union of a list of (lo, hi) ranges, respecting a reversed axis
    (e.g. panel e's faint->bright M_1450 axis, where lo > hi)."""
    los = [r[0] for r in ranges]
    his = [r[1] for r in ranges]
    if reversed_axis:
        return max(los), min(his)
    return min(los), max(his)


def make_grid(panel_key, redshifts, snap_data, output_file, draw_fn,
              x_reversed=False, **draw_kwargs):
    """Assemble one grid figure for a given panel, one subplot per bin in
    `redshifts` (any length -- laid out row-major, 3 columns wide, e.g. 6
    bins give 2 rows x 3 cols), using pre-loaded `snap_data[z]` catalogues.
    Each bin is either a scalar target z (see DEFAULT_REDSHIFTS, the default)
    or a (lo, hi) range (see snaps_in_range(), for callers that still want
    to stack a redshift window) and is used verbatim as the `snap_data` key
    and as the `z_target` passed to `draw_fn`.

    Subplots are touching (no gaps) with tick labels shown only on the
    outer left column / bottom row of the grid, like a standard corner plot.
    For that to look right every subplot must share the SAME x/y range, but
    a given bin's own data (its LRD selection) can force lock_axis_range()
    to widen past the MULTIZ_PANEL_*_XLIM/YLIM default. So before drawing
    anything for real, every bin is run once in `range_only` mode (cheap --
    stops right after computing its natural range, before any scatter/KDE
    work) to find the union range needed across all of them, and every
    subplot is then drawn on that identical union range -- this is also why
    panel f's y-range pass reruns with the x union already applied, since
    its bins (and therefore its y-range) depend on the x-range.
    """
    spec = PANEL_SPECS[panel_key]
    n = len(redshifts)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    scratch_fig = plt.figure()
    scratch_ax = scratch_fig.add_subplot(111)
    x_ranges = []
    for z in redshifts:
        scratch_ax.cla()
        _, xr, _ = draw_fn(scratch_ax, snap_data[z], z, range_only=True, **draw_kwargs)
        x_ranges.append(xr)
    x_union = _union_lim(x_ranges, reversed_axis=x_reversed)

    y_ranges = []
    for z in redshifts:
        scratch_ax.cla()
        _, _, yr = draw_fn(scratch_ax, snap_data[z], z, xlim=x_union, range_only=True, **draw_kwargs)
        y_ranges.append(yr)
    y_union = _union_lim(y_ranges, reversed_axis=False)
    plt.close(scratch_fig)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.4 * nrows),
                             squeeze=False)
    axes_flat = axes.flatten()

    # Merge handles across ALL subplots (by label, first-seen order), not
    # just the first one -- some overlays only appear on a subset of
    # redshifts (e.g. the Shen+20 model only draws for z = 1-7), so taking
    # only the first subplot's handles could silently drop those from the
    # legend.
    legend_by_label = {}
    for i, z in enumerate(redshifts):
        ax = axes_flat[i]
        ax.minorticks_on()
        data = snap_data[z]
        handles, _, _ = draw_fn(ax, data, z, xlim=x_union, ylim=y_union, **draw_kwargs)
        for h in (handles or []):
            legend_by_label.setdefault(h.get_label(), h)
    legend_handles = list(legend_by_label.values())

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis('off')

    # last populated row in each column (bottom row isn't always full if
    # n isn't a multiple of ncols)
    last_row_in_col = {}
    for i in range(n):
        row, col = divmod(i, ncols)
        last_row_in_col[col] = max(last_row_in_col.get(col, 0), row)

    for i in range(n):
        ax = axes_flat[i]
        row, col = divmod(i, ncols)
        if col != 0:
            ax.tick_params(labelleft=False)
        if row != last_row_in_col[col]:
            ax.tick_params(labelbottom=False)

    left, right, top, bottom = 0.06, 0.995, 0.95, 0.065
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom,
                        wspace=0.0, hspace=0.0)
    # Finalize the layout BEFORE reading tick positions below: each
    # subplot's AutoLocator picks its tick spacing based on the axes' final
    # pixel size, which subplots_adjust only just fixed -- querying ticks
    # any earlier reads a stale, pre-layout locator result that doesn't
    # match what actually renders.
    fig.canvas.draw()

    # Since every subplot shares the identical axis range, each internal
    # seam has the SAME tick position from both sides but a DIFFERENT tick
    # value (e.g. one subplot's right edge is 13, its touching neighbour's
    # left edge is 6) -- so their labels land on top of each other. Blank
    # the one nearest the shared edge on the side that isn't the outer
    # edge of the whole grid, leaving a single readable label per seam.
    for i in range(n):
        ax = axes_flat[i]
        row, col = divmod(i, ncols)
        if row == last_row_in_col[col] and col != ncols - 1:
            _blank_edge_ticklabel(ax, 'x', ax.get_xlim()[1])
        if col == 0 and row != last_row_in_col[0]:
            _blank_edge_ticklabel(ax, 'y', ax.get_ylim()[0])

    # ONE shared x/y-axis label for the whole grid (not repeated per
    # subplot): an invisible axes whose bounding box exactly matches the
    # real subplot grid's (so its own bottom/left edge coincides with
    # where the real tick labels start), with a labelpad just big enough
    # to clear that tick-label text -- not a large hand-tuned gap.
    big_ax = fig.add_axes([left, bottom, right - left, top - bottom], frameon=False)
    big_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    big_ax.set_xlabel(spec['xlabel'], fontsize=13, labelpad=18)
    big_ax.set_ylabel(spec['ylabel'], fontsize=13, labelpad=24)

    fig.suptitle(spec['title'], fontsize=15, y=0.995)
    if legend_handles:
        # Position the legend below the xlabel's ACTUAL rendered extent
        # (rather than a hand-tuned fraction) so it doesn't collide with it
        # regardless of legend row count/label length.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        xlabel_bbox = big_ax.xaxis.label.get_window_extent(renderer=renderer)
        xlabel_bottom_fig = xlabel_bbox.transformed(fig.transFigure.inverted()).y0
        legend_y = xlabel_bottom_fig - 0.025
        fig.legend(handles=legend_handles, loc='upper center',
                  bbox_to_anchor=(0.55, legend_y), ncol=min(len(legend_handles), 5),
                  fontsize=10, frameon=False)

    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'✓  Saved  →  {output_file}')

# ============================================================================
# CLI
# ============================================================================

def main():
    p = argparse.ArgumentParser(
        description='Multi-redshift grid version of bh_lrd_analysis.py: '
                    'one figure per panel (a-f), one redshift-bin subplot each.'
    )
    p.add_argument('-i', '--input-pattern', default='./output/millennium/model_*.hdf5')
    p.add_argument('--redshifts', type=float, nargs='+', default=None,
                   help='Target redshifts, one subplot each '
                        f'(default: {DEFAULT_REDSHIFTS}). Each maps to its '
                        'single nearest snapshot column -- see --window to '
                        'stack neighbours if a target is too sparse.')
    p.add_argument('--window', type=int, default=0,
                   help='Stack columns [s-window, s+window] per redshift to fight sparsity at high z.')
    p.add_argument('--catalogue', default=None,
                   help='Force a specific Snap_N catalogue group (default: auto-select most complete).')
    p.add_argument('--no-lrd', action='store_true', help='Skip LRD selection overlay.')
    p.add_argument('--no-fbh', action='store_true',
                   help='Disable the f_BH red/blue split on panel a (all selected = red).')
    p.add_argument('--no-lit', action='store_true',
                   help='Skip the literature overlay on panels a-e.')
    p.add_argument('--no-mask-seeds', action='store_true',
                   help='Do not mask BH accretion events still essentially at '
                        f'their own BHSeedMass (< {SEED_GROWTH_THRESHOLD:.0%} grown); '
                        'by default these are excluded everywhere since they '
                        'pile up as a spurious cluster at log M_BH ~ 2.')
    p.add_argument('--bhar-floor', type=float, default=LRD_BHAR_DEFAULT,
                   help=f'BHAR floor in M_sun/yr (default {LRD_BHAR_DEFAULT}).')
    p.add_argument('--outdir', default=None,
                   help='Output directory (default: <input dir>/plots).')
    p.add_argument('--no-panel-a', action='store_true')
    p.add_argument('--no-panel-b', action='store_true')
    p.add_argument('--no-panel-c', action='store_true')
    p.add_argument('--no-panel-d', action='store_true')
    p.add_argument('--no-panel-e', action='store_true')
    p.add_argument('--no-panel-f', action='store_true')
    p.add_argument('--sim-volume', type=float, default=None,
                   help='Override comoving volume in (Mpc/h)^3 for panel f.')
    p.add_argument('--lf-bins', type=int, default=30,
                   help='Number of log10(L_bol) bins for panel f (default 30).')
    args = p.parse_args()

    files = sorted(glob.glob(args.input_pattern))
    if not files:
        print(f'ERROR: no files matched "{args.input_pattern}"'); sys.exit(1)

    h_h = read_sim_params(files[0])
    redshifts = read_actual_redshifts(files[0])
    redshift_bins = args.redshifts if args.redshifts is not None else DEFAULT_REDSHIFTS
    print(f'Files:        {len(files)}')
    print(f'Hubble_h:     {h_h}')
    print(f'BHAR floor:   {args.bhar_floor} M_sun/yr')
    print(f'Redshifts:    {redshift_bins}')

    # ── resolve each bin to one or more snapshot columns, then read once and
    # reuse across every panel (a-f all draw from the same catalogues). A
    # scalar bin maps to its nearest single snapshot (+/- --window); a
    # (lo, hi) range bin stacks every snapshot whose redshift falls inside
    # it, to fight LRD sparsity at high z. ─────────────────────────────────
    snap_data = {}
    for zb in redshift_bins:
        if isinstance(zb, tuple):
            lo, hi = zb
            cols = snaps_in_range(lo, hi, redshifts)
            if not cols:
                cols = [nearest_snap_for_z(0.5 * (lo + hi), redshifts)]
                print(f'  z ~ {lo:g}-{hi:g}  ->  no snapshots in range, '
                      f'falling back to nearest single snapshot {cols[0]}')
            else:
                z_list = ', '.join(f'{snap_to_z(c, redshifts):.2f}' for c in cols)
                print(f'  z ~ {lo:g}-{hi:g}  ->  snapshots {cols} (z = {z_list})')
            data = read_epoch(files, cols[0], h_h, catalogue=args.catalogue, cols=cols)
        else:
            snap_col = nearest_snap_for_z(zb, redshifts)
            actual_z = snap_to_z(snap_col, redshifts)
            print(f'  z ~ {zb:g}  ->  snapshot {snap_col} (z = {actual_z:.3f})'
                  + (f'  +/- {args.window}' if args.window else ''))
            # default to that snapshot's own catalogue -- the population
            # actually observed at this redshift, not just the subset of
            # galaxies surviving to whichever catalogue is most complete.
            # --catalogue still overrides this for every panel if forced.
            data = read_epoch(files, snap_col, h_h,
                              catalogue=args.catalogue or f'Snap_{snap_col}',
                              window=args.window)
        n_events = len(data['bh_mass'])
        print(f'    catalogue={data["cat_group"]}  events={n_events:,}')
        snap_data[zb] = data

    outdir = Path(args.outdir) if args.outdir else Path(files[0]).parent / 'plots'
    outdir.mkdir(exist_ok=True, parents=True)

    if not args.no_panel_a:
        print('Building panel a grid...')
        make_grid('a', redshift_bins, snap_data, outdir / PANEL_SPECS['a']['fname'],
                  draw_panel_a, show_lrd=(not args.no_lrd), use_fbh=(not args.no_fbh),
                  bhar_floor=args.bhar_floor, show_lit=(not args.no_lit),
                  mask_seeds=(not args.no_mask_seeds))

    if not args.no_panel_b:
        print('Building panel b grid...')
        make_grid('b', redshift_bins, snap_data, outdir / PANEL_SPECS['b']['fname'],
                  draw_panel_b, show_lrd=(not args.no_lrd),
                  bhar_floor=args.bhar_floor, show_lit=(not args.no_lit),
                  mask_seeds=(not args.no_mask_seeds))

    if not args.no_panel_c:
        print('Building panel c grid...')
        make_grid('c', redshift_bins, snap_data, outdir / PANEL_SPECS['c']['fname'],
                  draw_panel_c, show_lrd=(not args.no_lrd),
                  bhar_floor=args.bhar_floor, show_lit=(not args.no_lit),
                  mask_seeds=(not args.no_mask_seeds))

    if not args.no_panel_d:
        print('Building panel d grid...')
        make_grid('d', redshift_bins, snap_data, outdir / PANEL_SPECS['d']['fname'],
                  draw_panel_d, show_lrd=(not args.no_lrd),
                  bhar_floor=args.bhar_floor, show_lit=(not args.no_lit),
                  mask_seeds=(not args.no_mask_seeds))

    if not args.no_panel_e:
        print('Building panel e grid...')
        make_grid('e', redshift_bins, snap_data, outdir / PANEL_SPECS['e']['fname'],
                  draw_panel_e, x_reversed=True, show_lrd=(not args.no_lrd),
                  bhar_floor=args.bhar_floor, show_lit=(not args.no_lit),
                  mask_seeds=(not args.no_mask_seeds))

    if not args.no_panel_f:
        volume_h3 = args.sim_volume if args.sim_volume is not None else read_box_volume_h3(files)
        print(f'Volume (panel f): {volume_h3:.4e} (Mpc/h)^3')
        print('Building panel f grid...')
        make_grid('f', redshift_bins, snap_data, outdir / PANEL_SPECS['f']['fname'],
                  draw_panel_f, volume_h3=volume_h3, show_lrd=(not args.no_lrd),
                  bhar_floor=args.bhar_floor, n_bins=args.lf_bins, h_h=h_h,
                  show_lit=(not args.no_lit),
                  mask_seeds=(not args.no_mask_seeds))


if __name__ == '__main__':
    main()

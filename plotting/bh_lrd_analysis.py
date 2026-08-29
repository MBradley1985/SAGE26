#!/usr/bin/env python3
"""
bh_lrd_analysis.py
==================
Recreates panels (a)-(e), all at a chosen redshift and all sharing the same
LRD selection/colour scheme, since each is just a different projection of
the same underlying accretion events:

    panel a:  log10(Mdot_BH  [M_sun/yr])       vs  log10(M_BH  [M_sun])
              (Chen & Mo 2026, arXiv:2605.31077, Fig. 1a)
    panel b:  f_BH = log10(M_BH / M_star)      vs  log10(M_BH  [M_sun])
              (Chen & Mo 2026, Fig. 1b)
    panel c:  log10(M_BH  [M_sun])             vs  log10(M_star [M_sun])
              (styled after Kocevski et al. 2023, ApJL 954, L4, Fig. 7 --
              Kormendy & Ho 2013 relation + constant-ratio lines, no obs. data)
    panel d:  log10(L_bol  [erg/s])            vs  log10(M_BH  [M_sun])
              (styled after Kocevski et al. 2023, Fig. 6 -- lambda_Edd lines)
    panel e:  M_1450 (rest-frame UV absolute mag) vs log10(M_BH [M_sun])
              (styled after the M_UV-M_BH comparison plots common in the
              JWST LRD literature)

Panels d and e derive L_bol and M_1450 from the simulated BH accretion rate
via a chain of standard scaling relations (radiative efficiency -> L_bol,
bolometric correction -> L_1450, AB magnitude definition -> M_1450). See
BH_LUMINOSITY_CONVERSIONS.md for the full derivation, constants, and
citations for every step.

HOW THE DATA IS STORED (important!)
-----------------------------------
In the SAGE26 HDF5 output, each `Snap_N` group is a *galaxy catalogue* at that
output time.  The high-N catalogues (e.g. Snap_62 / Snap_63) hold every
surviving galaxy together with that galaxy's FULL black-hole accretion history
in `[Ngal, ABSOLUTEMAXSNAPS]` arrays:

    BHMaxaccretionRate[:, c]    Mdot_BH recorded at snapshot c
    BHEddingtonRateLimit[:, c]  Mdot_Edd at snapshot c
    BHMassatAccretion[:, c]     M_BH at the time of that accretion episode
    BHAccretionType[:, c]       0=Radio, 1=Merger, 2=Disk Instability

So to reproduce the z=5 plane we read the LATE catalogue (most complete) and
slice the history COLUMN closest to z=5 (snapshot 10, z=5.48 in Millennium).
We do NOT read the `Snap_10` group directly — at z=5 the catalogue is nearly
empty and the histories there are blank.

The x-axis uses BHMassatAccretion (M_BH at that epoch), NOT the z=0
BlackHoleMass, so the plot shows the true (M_BH, Mdot_BH) plane at that redshift.

LRD selection criteria (Chen & Mo 2026, §II.2 / Fig. 1):
    Red dots  (full LRD):  Mdot_BH >= 0.1 M_sun/yr  AND  Mdot_BH >= Mdot_Edd
                           AND  f_BH = M_BH/M_star >= 0.03
    Blue dots (partial):   Mdot_BH >= 0.1 M_sun/yr  AND  Mdot_BH >= Mdot_Edd
                           AND  f_BH < 0.03
Reference lines:
    red  solid  -> Mdot_BH = Mdot_Edd
    orange solid-> Mdot_BH = 10 * Mdot_Edd
    red  dashed -> Mdot_BH = 0.1  M_sun/yr   (default BHAR threshold)
    red  dotted -> Mdot_BH = 0.05 M_sun/yr   (alternative threshold)
Contours enclose 68 / 95 / 99.7 % of the plotted BHs.

NOTE ON f_BH:  the per-epoch stellar mass is not stored alongside the accretion
history, so f_BH is computed from the catalogue-level StellarMass.  Pass
--no-fbh to disable the f_BH split (all selected BHs drawn red) if you'd rather
not mix epochs.  See the comment in compute_selection() for details.

Usage
-----
    python3 plotting/bh_lrd_analysis.py                  # writes panels a-e
    python3 plotting/bh_lrd_analysis.py -s 10            # z = 5.48 column
    python3 plotting/bh_lrd_analysis.py -s 10 --window 1 # stack snaps 9-11
    python3 plotting/bh_lrd_analysis.py --catalogue Snap_63
    python3 plotting/bh_lrd_analysis.py --bhar-floor 0.05
    python3 plotting/bh_lrd_analysis.py --no-panel-b --no-panel-c --no-panel-d --no-panel-e
"""

import argparse
import glob
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

from lrd_literature_data import (
    PANG26, MATHEE24, LABBE25, FURTAK23, LIN25, LIT_STYLE,
    SHEN20_ZREF, SHEN20_GAMMA1_A, SHEN20_GAMMA2_A, SHEN20_LSTAR_A, SHEN20_PHISTAR_A,
)
from run_style import contour_style_for_index, style_for_index, lighten_color

# ============================================================================
# MATPLOTLIB STYLE  (matching bh_eddington_analysis.py)
# ============================================================================
plt.rcParams.update({
    'figure.dpi': 140,
    'figure.autolayout': True,
    'font.family': 'serif',
    'font.size': 20.0,
    'axes.linewidth': 1.5,
    'xtick.major.size': 7.5, 'xtick.major.width': 1.5,
    'xtick.minor.size': 5.5, 'xtick.minor.width': 0.5,
    'xtick.direction': 'in', 'xtick.top': True, 'xtick.labelsize': 16,
    'xtick.major.pad': 9,
    'ytick.major.size': 7.5, 'ytick.major.width': 1.5,
    'ytick.minor.size': 5.5, 'ytick.minor.width': 0.5,
    'ytick.direction': 'in', 'ytick.right': True, 'ytick.labelsize': 16,
    'legend.frameon': False, 'legend.fontsize': 12,
})

# ============================================================================
# CONSTANTS & UNIT CONVERSIONS
# ============================================================================
SEC_PER_YEAR   = 365.25 * 24 * 3600    # s / yr
UNIT_TIME_IN_S = 3.086e19              # SAGE Millennium code time unit (~1 Gyr)

# ── Fundamental constants (cgs) ────────────────────────────────────────────
C_LIGHT_CGS = 2.99792458e10    # cm / s
SOLAR_MASS_G = 1.989e33        # g
SOLAR_LUM_ERG_S = 3.828e33     # erg / s (IAU nominal value)
PC_TO_CM     = 3.0856775814913673e18   # cm

# Eddington luminosity per unit mass (Rybicki & Lightman 1979, eq. 1.4.9) and
# the standard thin-disk radiative efficiency -- SAME values as
# model_enhanced_bhphysics.c (EDDINGTON_LUM_PER_MSUN_CGS, AGN_RADIATIVE_EFFICIENCY),
# kept in sync so panels a/c/d/e all use one Eddington relation.
EDDINGTON_LUM_PER_MSUN_CGS = 1.3e38   # erg/s per Msun
AGN_RADIATIVE_EFFICIENCY   = 0.1      # eta (Shakura & Sunyaev 1973)

# Salpeter e-folding time, derived (not hardcoded) from the two constants
# above so it can't drift out of sync with them:
#   T_Sal = eta * (M c^2) / L_Edd(M)   [independent of M]
# For eta=0.1 this evaluates to ~4.4e7 yr, the standard textbook value
# (NOT 4.5e8 yr -- an earlier version of this constant was off by 10x).
T_SALPETER_YR = (AGN_RADIATIVE_EFFICIENCY * SOLAR_MASS_G * C_LIGHT_CGS**2
                  / EDDINGTON_LUM_PER_MSUN_CGS) / SEC_PER_YEAR

# ── LRD selection (Chen & Mo 2026) ─────────────────────────────────────────
LRD_BHAR_DEFAULT = 0.1     # M_sun/yr  (red dashed line)
LRD_BHAR_ALT     = 0.05    # M_sun/yr  (red dotted line)
LRD_FBHM_THRESH  = 0.03    # M_BH / M_star >= 3% -> red, else blue

# A literature point only overlays a panel if the panel's own redshift is
# within this much of that point's redshift -- otherwise the observation
# isn't actually relevant to what's being plotted at that epoch.
LIT_Z_TOL = 1.0

# BH must exceed its own BHSeedMass by at least this fraction to count as
# "grown" -- see mask_ungrown_seeds(). Below this, BHMassatAccretion is
# essentially still the seed value (a few x 1e-3 relative growth is within
# float/timestep noise of zero), and these events pile up as a spurious
# cluster at log M_BH ~ 2 (the ~100 Msun light-seed mass) in every panel
# plotting M_BH, rather than reflecting any real accretion history.
SEED_GROWTH_THRESHOLD = 0.01

# ── M_BH-M_star relation (Kormendy & Ho 2013, ARA&A 51, 511, eq. 10) ──────
# log(M_BH/Msun) = alpha + beta * log(M_bulge / 1e11 Msun), intrinsic scatter eps (dex)
KORMENDY_HO_ALPHA   = 8.69
KORMENDY_HO_BETA    = 1.16
KORMENDY_HO_SCATTER = 0.29   # dex

# M_BH/M_star ratio lines shown in panel c (Kocevski et al. 2023, Fig. 7)
MASS_RATIO_LINES = [0.1, 0.01, 1.0e-3]

# Eddington ratio lines shown in panel d (Kocevski et al. 2023, Fig. 6)
EDDINGTON_RATIO_LINES = [1.0, 0.1, 0.01]

# ── BHAccretionType coding (see module docstring) and the colour scheme used
# to split the panel-a background scatter by accretion channel; matches the
# convention in allresults-blackholes.py (ACC_RADIO/ACC_MERGER/ACC_INSTAB). ──
ACC_RADIO, ACC_MERGER, ACC_INSTAB = 0, 1, 2
ACC_TYPE_COLORS = {
    ACC_MERGER: ('#1976D2', 'Merger-driven'),
    ACC_INSTAB: ('#FBC02D', 'Disk instability'),
}

# ── Bolometric-to-UV correction (Runnoe, Brotherton & Shang 2012, MNRAS 422,
# 478; coefficients from the Dec 2012 erratum, MNRAS 427, 1800) ───────────
# L_iso = BC_1450 * lambda*L_lambda(1450 Angstrom), constant-ratio form.
BC_1450 = 4.20
WAVELENGTH_1450_CM = 1450.0e-8   # cm

# Oke & Gunn (1983) AB magnitude zero point
AB_ZEROPOINT = 48.60

# Comoving box side (Mpc/h); fallback when the HDF5 header lacks box_size
# (matches allresults-blackholes.py's MILLENNIUM_BOX_MPC_H).
MILLENNIUM_BOX_MPC_H = 62.5

# ── Fixed axis ranges ───────────────────────────────────────────────────────
# Every panel plots these ranges by default (NOT derived from the current
# run's own percentiles), so different model runs/parameter choices land on
# an identical grid and can be compared side by side. A panel only widens
# past its default if an LRD-selected or literature point would otherwise be
# clipped off-axis -- and prints a NOTE when that happens, since it means
# that particular run's plot is no longer on the standard shared scale.
PANEL_A_XLIM = (2.0, 10.5)     # log M_BH
PANEL_A_YLIM = (-13.5, 3.0)    # log Mdot_BH
PANEL_B_XLIM = (3.0, 10.0)     # log M_BH
PANEL_B_YLIM = (-9.0, 0.5)     # log f_BH
PANEL_C_XLIM = (6.0, 11.0)     # log M_star
PANEL_C_YLIM = (3.0, 9.0)      # log M_BH
PANEL_D_XLIM = (3.0, 9.0)      # log M_BH
PANEL_D_YLIM = (35.0, 47.0)    # log L_bol
PANEL_E_XLIM = (-5.0, -25.0)   # M_UV,1450 (faint -> bright, deliberately reversed)
PANEL_E_YLIM = (3.0, 10.0)     # log M_BH
PANEL_F_XLIM = (30.0, 48.0)    # log L_bol
PANEL_F_YLIM = (-8.0, 0.0)     # log(dN/dlogL / Volume)

# Lower log L_bol bound for the Shen+20 QLF model overlay on panel f -- see
# shen20_bolometric_qlf_logphi and its call site for why this is needed.
SHEN20_MODEL_LOGLBOL_MIN = 42.0

# ── Millennium snapshot -> redshift ────────────────────────────────────────
MILLENNIUM_SNAP_TO_Z = {
    0: 127.0, 1: 65.74, 2: 40.0,  3: 26.66, 4: 19.36, 5: 14.78, 6: 11.66,
    7: 9.44,  8: 7.64,  9: 6.44,  10: 5.48, 11: 4.73, 12: 4.19, 13: 3.72,
    14: 3.33, 15: 3.0,  16: 2.73, 17: 2.48, 18: 2.27, 19: 2.07, 20: 1.90,
    21: 1.75, 22: 1.61, 23: 1.48, 24: 1.37, 25: 1.27, 26: 1.18, 27: 1.10,
    28: 1.02, 29: 0.96, 30: 0.90, 31: 0.85, 32: 0.81, 33: 0.77, 34: 0.73,
    35: 0.70, 36: 0.67, 37: 0.63, 38: 0.60, 39: 0.57, 40: 0.54, 41: 0.51,
    42: 0.49, 43: 0.46, 44: 0.43, 45: 0.41, 46: 0.39, 47: 0.37, 48: 0.36,
    49: 0.34, 50: 0.32, 51: 0.31, 52: 0.29, 53: 0.28, 54: 0.27, 55: 0.26,
    56: 0.25, 57: 0.24, 58: 0.23, 59: 0.21, 60: 0.20, 61: 0.18, 62: 0.0,
}

def snap_to_z(snap, redshifts=None):
    """Redshift for a given snapshot. Prefers the file's own
    Header/snapshot_redshifts table (`redshifts`, from read_actual_redshifts)
    since MILLENNIUM_SNAP_TO_Z assumes the stock Millennium snapshot spacing
    and silently gives the wrong z for any run whose output snapshots don't
    follow that exact spacing.
    """
    if redshifts is not None and 0 <= snap < len(redshifts):
        return float(redshifts[snap])
    return MILLENNIUM_SNAP_TO_Z.get(snap, 0.0)

# ============================================================================
# I/O
# ============================================================================

def read_sim_params(filepath):
    """Extract Hubble_h from HDF5 header."""
    try:
        with h5py.File(filepath, 'r') as hf:
            for grp in ('Header/Simulation', 'Header', 'Parameters'):
                if grp in hf:
                    attrs = hf[grp].attrs
                    for key in ('hubble_h', 'HubbleParam', 'Hubble_h'):
                        if key in attrs:
                            return float(attrs[key])
    except Exception:
        pass
    return 0.73   # Millennium default


def read_actual_redshifts(filepath):
    """Header/snapshot_redshifts array, or None if the file doesn't have one."""
    try:
        with h5py.File(filepath, 'r') as hf:
            if 'Header/snapshot_redshifts' in hf:
                return np.array(hf['Header/snapshot_redshifts'])
    except Exception:
        pass
    return None


def read_box_volume_h3(file_list):
    """
    Total comoving simulation volume in (Mpc/h)^3, matching the convention
    used for the accretion rate function in allresults-blackholes.py:
        volume_h3 = box_size^3 * sum(frac_volume_processed across files)
    box_size is read from Header/Simulation:box_size (Mpc/h); each file's
    frac_volume_processed (Header/Runtime) defaults to 1.0 if absent, and
    multiple input files are assumed to be non-overlapping shares of one box.
    """
    box_size = None
    frac_total = 0.0
    for fpath in file_list:
        try:
            with h5py.File(fpath, 'r') as hf:
                for grp in ('Header/Simulation', 'Header'):
                    if grp in hf and 'box_size' in hf[grp].attrs:
                        box_size = float(hf[grp].attrs['box_size'])
                        break
                if 'Header/Runtime' in hf and 'frac_volume_processed' in hf['Header/Runtime'].attrs:
                    frac_total += float(hf['Header/Runtime'].attrs['frac_volume_processed'])
                else:
                    frac_total += 1.0
        except Exception:
            frac_total += 1.0
    if box_size is None:
        box_size = MILLENNIUM_BOX_MPC_H
    return box_size**3 * frac_total


def pick_catalogue_group(hf, requested=None):
    """
    Choose which Snap_N catalogue group to read the histories from.
    Default: the highest-numbered group that actually has galaxies
    (the most complete catalogue, carrying full accretion histories).
    """
    snaps = sorted(
        [k for k in hf.keys() if k.startswith('Snap_')],
        key=lambda s: int(s.split('_')[1]),
    )
    if requested is not None:
        if requested in hf:
            return requested
        print(f'  WARNING: requested catalogue "{requested}" not found; '
              f'falling back to auto-select.')
    # auto: walk from the top until we find a populated group
    for s in reversed(snaps):
        try:
            if hf[s]['BlackHoleMass'].shape[0] > 0:
                return s
        except Exception:
            continue
    return snaps[-1]


def _history_column(arr2d, col):
    """Return column `col` of a [Ngal, MAXSNAPS] array, clamped to bounds."""
    c = min(max(col, 0), arr2d.shape[1] - 1)
    return arr2d[:, c]


def read_epoch(file_list, snap_col, h_h, catalogue=None, window=0, cols=None):
    """
    Read the (M_BH, Mdot_BH) plane at the history column `snap_col`
    (= Millennium snapshot index) from the most complete catalogue group.

    window > 0 stacks columns [snap_col-window, snap_col+window] to fight
    sparsity at very high z (each event still becomes its own point).

    cols, if given, overrides both snap_col and window with an explicit list
    of history columns to stack -- used for a redshift RANGE bin (e.g.
    z = 2-4) where the columns to stack aren't a symmetric window around one
    snapshot; see bh_lrd_analysis_multiz.py's range bins.

    Returns dict of physical-unit arrays:
        bh_mass        [M_sun]   (M_BH at the accretion epoch)
        mdot_msun_yr   [M_sun/yr]
        mdot_edd       [M_sun/yr]
        acc_type       {0,1,2,-1}
        stellar_mass   [M_sun]   (catalogue-level; see f_BH caveat)
        seed_mass      [M_sun]   (catalogue-level BHSeedMass, for mask_ungrown_seeds())
        cat_group      str       (which group was read)
    """
    mass_conv = 1e10 / h_h
    rate_conv = mass_conv / (UNIT_TIME_IN_S / SEC_PER_YEAR)

    out_bh, out_mdot, out_edd, out_type, out_star, out_seed = [], [], [], [], [], []
    cat_used = None
    cols = cols if cols is not None else list(range(snap_col - window, snap_col + window + 1))

    for fpath in file_list:
        with h5py.File(fpath, 'r') as hf:
            cat = pick_catalogue_group(hf, catalogue)
            cat_used = cat
            grp = hf[cat]

            mdot_h = grp['BHMaxaccretionRate'][:]      # [Ngal, MAXSNAPS]
            edd_h  = grp['BHEddingtonRateLimit'][:]
            mass_h = (grp['BHMassatAccretion'][:]
                      if 'BHMassatAccretion' in grp else None)
            type_h = (grp['BHAccretionType'][:]
                      if 'BHAccretionType' in grp else None)
            star   = grp['StellarMass'][:]              # [Ngal] catalogue-level
            seed   = (grp['BHSeedMass'][:] if 'BHSeedMass' in grp
                      else np.zeros(star.shape[0]))       # [Ngal] catalogue-level

            for c in cols:
                if c < 0 or c >= mdot_h.shape[1]:
                    continue
                mdot_c = _history_column(mdot_h, c)
                edd_c  = _history_column(edd_h,  c)

                if mass_h is not None:
                    bh_c = _history_column(mass_h, c)
                else:
                    # fallback: no per-epoch mass -> derive from Eddington rate
                    bh_c = edd_c * rate_conv * T_SALPETER_YR / mass_conv  # code units, to match bh_c * mass_conv below

                type_c = (_history_column(type_h, c)
                          if type_h is not None
                          else np.full_like(mdot_c, -1.0))

                # only keep galaxies that actually have an event in this column
                ev = mdot_c > 0
                if not np.any(ev):
                    continue

                out_bh.append(bh_c[ev]   * mass_conv)
                out_mdot.append(mdot_c[ev] * rate_conv)
                out_edd.append(edd_c[ev]  * rate_conv)
                out_type.append(type_c[ev])
                out_star.append(star[ev]  * mass_conv)
                out_seed.append(seed[ev]  * mass_conv)

    if not out_bh:
        return {
            'bh_mass': np.array([]), 'mdot_msun_yr': np.array([]),
            'mdot_edd': np.array([]), 'acc_type': np.array([]),
            'stellar_mass': np.array([]), 'seed_mass': np.array([]),
            'cat_group': cat_used,
        }

    return {
        'bh_mass'      : np.concatenate(out_bh),
        'mdot_msun_yr' : np.concatenate(out_mdot),
        'mdot_edd'     : np.concatenate(out_edd),
        'acc_type'     : np.concatenate(out_type),
        'stellar_mass' : np.concatenate(out_star),
        'seed_mass'    : np.concatenate(out_seed),
        'cat_group'    : cat_used,
    }

# ============================================================================
# HELPERS
# ============================================================================

def eddington_mdot(log_mbh):
    """log10(Mdot_Edd [M_sun/yr]) from log10(M_BH [M_sun])  (eta=0.1)."""
    return log_mbh - np.log10(T_SALPETER_YR)


def kde_contour(ax, x, y, x_lo, x_hi, y_lo, y_hi, color,
                 linewidths=(0.9, 1.2, 1.6), linestyles=(':', '--', '-'),
                 n_kde=60_000, seed=77, zorder=2, min_points=50):
    """
    Overlay 68/95/99.7% KDE contours of (x, y) in a single colour.
    Returns False (no-op) if there are too few points to fit a KDE.
    """
    if len(x) < min_points:
        return False
    try:
        if len(x) > n_kde:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(x), n_kde, replace=False)
            xk, yk = x[idx], y[idx]
        else:
            xk, yk = x, y

        kde = gaussian_kde(np.vstack([xk, yk]), bw_method='scott')
        xi = np.linspace(x_lo, x_hi, 250)
        yi = np.linspace(y_lo, y_hi, 250)
        Xi, Yi = np.meshgrid(xi, yi)
        Zi = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)

        z_sort = np.sort(Zi.ravel())[::-1]
        z_cum = np.cumsum(z_sort) / z_sort.sum()
        def lvl(frac):
            return z_sort[min(np.searchsorted(z_cum, frac), len(z_sort) - 1)]
        levels = sorted([lvl(0.683), lvl(0.954), lvl(0.997)])

        ax.contour(Xi, Yi, Zi, levels=levels, colors=color,
                   linewidths=linewidths, linestyles=linestyles, zorder=zorder)
        return True
    except Exception as e:
        print(f'  WARNING: KDE contours skipped ({e})')
        return False


def draw_contours_multirun(ax, xy_per_run, x_lo, x_hi, y_lo, y_hi, **kde_kwargs):
    """
    Draw one KDE contour set per run -- no raw scatter cloud, no per-point
    LRD red/blue selection dots -- the shared drawing primitive for every
    scatter+KDE panel's compare mode (panels a-e here, and their
    bh_lrd_analysis_multiz.py equivalents, which reuse this function
    directly). Single-run mode is unaffected; that only ever calls
    kde_contour() itself, once, with no `style` involved.

    `xy_per_run` is a list of (x, y, style) tuples, `style` a
    run_style.contour_style_for_index() dict (color + label). Any extra
    `kde_kwargs` (e.g. linewidths, n_kde) are forwarded to kde_contour() --
    bh_lrd_analysis_multiz.py uses this to match its own smaller point
    budget / thinner linewidths for compact grid subplots.

    Returns one legend handle per run that had enough points to contour.
    """
    handles = []
    for x, y, style in xy_per_run:
        ok = kde_contour(ax, x, y, x_lo, x_hi, y_lo, y_hi, style['color'], **kde_kwargs)
        if ok:
            handles.append(Line2D([0], [0], color=style['color'], lw=1.6, label=style['label']))
        else:
            print(f"  (too few points for {style['label']}'s KDE contour)")
    return handles


def lock_axis_range(default_lo, default_hi, must_include=(), axis_name='axis', pad=0.3):
    """
    Returns (lo, hi) for a panel's axis. Normally just default_lo/default_hi,
    unchanged, so different runs share an identical fixed scale. Only widens
    past the default if a value in must_include (LRD-selected or literature
    points -- data that must stay visible) would otherwise be clipped off,
    printing a NOTE so it's clear this run's axis is non-standard.
    Handles both ascending (lo < hi) and reversed (lo > hi, e.g. panel e's
    magnitude axis) defaults.
    """
    lo, hi = default_lo, default_hi
    vals = [np.atleast_1d(v) for v in must_include if v is not None]
    vals = [v[np.isfinite(v)] for v in vals if len(v)]
    if not vals:
        return lo, hi
    data_min = min(v.min() for v in vals) - pad
    data_max = max(v.max() for v in vals) + pad

    axis_min, axis_max = min(lo, hi), max(lo, hi)
    new_min = min(axis_min, np.floor(data_min))
    new_max = max(axis_max, np.ceil(data_max))
    if new_min == axis_min and new_max == axis_max:
        return lo, hi

    print(f'  NOTE: widened {axis_name} from ({axis_min:g}, {axis_max:g}) to '
          f'({new_min:g}, {new_max:g}) to keep all plotted points visible '
          f'(this run is off the standard shared scale).')
    return (new_min, new_max) if lo < hi else (new_max, new_min)


def compute_selection(bh_mass, mdot, medd, star, bhar_floor):
    """
    LRD selection masks shared by panels a and b (Chen & Mo 2026, Sec II.2):
        bhar_pass  Mdot_BH >= bhar_floor
        edd_pass   Mdot_BH >= Mdot_Edd
        f_bh       M_BH / M_star
    Returns f_bh, lrd_red (bhar & edd & f_bh>=3%), lrd_blue (bhar & edd & f_bh<3%).
    """
    bhar_pass = mdot >= bhar_floor
    edd_pass  = mdot >= np.where(medd > 0, medd, np.inf)
    f_bh      = bh_mass / np.where(star > 0, star, np.inf)
    fbh_pass  = f_bh >= LRD_FBHM_THRESH
    lrd_red   = bhar_pass & edd_pass & fbh_pass
    lrd_blue  = bhar_pass & edd_pass & ~fbh_pass
    return f_bh, lrd_red, lrd_blue


def mask_ungrown_seeds(bh_mass, seed_mass, growth_threshold=SEED_GROWTH_THRESHOLD):
    """
    True where a BH has grown at least `growth_threshold` beyond its own
    BHSeedMass; False for accretion "events" recorded while the BH is still
    essentially at its seed value. Those show up as a spurious pileup at
    log M_BH ~ 2 in every M_BH panel (see SEED_GROWTH_THRESHOLD) rather than
    reflecting real accretion, so they're masked out by default everywhere
    (toggle with --no-mask-seeds).
    """
    return bh_mass > seed_mass * (1.0 + growth_threshold)


def kormendy_ho_mbh(log_mstar):
    """log10(M_BH [M_sun]) on the Kormendy & Ho (2013) M_BH-M_star relation."""
    return KORMENDY_HO_ALPHA + KORMENDY_HO_BETA * (log_mstar - 11.0)


def eddington_luminosity(log_mbh):
    """log10(L_Edd [erg/s]) from log10(M_BH [M_sun]) (Rybicki & Lightman 1979)."""
    return log_mbh + np.log10(EDDINGTON_LUM_PER_MSUN_CGS)


def lbol_from_mdot(mdot_msun_yr):
    """
    log10(L_bol [erg/s]) from Mdot_BH [M_sun/yr], assuming a constant
    radiative efficiency eta (thin-disk accretion, Shakura & Sunyaev 1973):
        L_bol = eta * Mdot_BH * c^2
    See BH_LUMINOSITY_CONVERSIONS.md for the full derivation and caveats.
    """
    mdot_g_s = np.asarray(mdot_msun_yr, dtype=np.float64) * SOLAR_MASS_G / SEC_PER_YEAR
    l_bol = AGN_RADIATIVE_EFFICIENCY * mdot_g_s * C_LIGHT_CGS**2
    return np.log10(l_bol)


def mdot_from_lbol(log_lbol):
    """
    Inverse of lbol_from_mdot(): Mdot_BH [M_sun/yr] from log10(L_bol [erg/s]),
    for converting literature L_bol measurements onto the same Mdot_BH axis
    as panel a, under the same eta=0.1 assumption.
    """
    l_bol = 10.0**np.asarray(log_lbol, dtype=np.float64)
    mdot_g_s = l_bol / (AGN_RADIATIVE_EFFICIENCY * C_LIGHT_CGS**2)
    return mdot_g_s * SEC_PER_YEAR / SOLAR_MASS_G


def bh_mass_min_from_lbol(log_lbol):
    """
    Minimum M_BH [M_sun] implied by L_bol under an Eddington-limited
    (lambda_Edd <= 1) assumption: M_BH >= L_bol / L_Edd_per_Msun. Used only
    for literature sources that give L_bol but no direct M_BH measurement.
    """
    log_l_edd_per_msun = np.log10(EDDINGTON_LUM_PER_MSUN_CGS)
    return 10.0**(np.asarray(log_lbol, dtype=np.float64) - log_l_edd_per_msun)


def shen20_bolometric_qlf_logphi(log_lbol, z, h_h=None):
    """
    log10(phi_bol [dex^-1 (Mpc/h)^-3]) from Shen et al. (2020, MNRAS 495,
    3252) global fit A -- their Eq. 11 (double power-law dn/dlogL) with the
    redshift-dependent parameters of their Eq. 14 (Table 4), evaluated at
    bolometric luminosity log_lbol [log10(erg/s)] and redshift z.

    Shen+20 tabulate log L_star in L_sun, so log_lbol is converted from
    erg/s before forming the L/L_star ratio. Their phi_star is in
    dex^-1 cMpc^-3 for their fixed H0 = 70 km/s/Mpc cosmology (NOT h-scaled
    like this module's own panel f, which bins into dex^-1 (Mpc/h)^-3 for
    whatever h the simulation being plotted actually used) -- so if h_h is
    given, the result is converted via phi[(Mpc/h)^-3] = phi[Mpc^-3] / h_h^3
    to sit on the same axis as this module's own luminosity function.
    """
    zp1 = 1.0 + z
    x = zp1 / (1.0 + SHEN20_ZREF)

    a0, a1, a2 = SHEN20_GAMMA1_A
    gamma1 = a0 + a1 * zp1 + a2 * (2.0 * zp1**2 - 1.0)

    b0, b1, b2 = SHEN20_GAMMA2_A
    gamma2 = 2.0 * b0 / (x**b1 + x**b2)

    c0, c1, c2 = SHEN20_LSTAR_A
    log_lstar_lsun = 2.0 * c0 / (x**c1 + x**c2)

    d0, d1 = SHEN20_PHISTAR_A
    log_phistar = d0 + d1 * zp1

    log_lbol_lsun = np.asarray(log_lbol, dtype=np.float64) - np.log10(SOLAR_LUM_ERG_S)
    ratio = 10.0**(log_lbol_lsun - log_lstar_lsun)
    log_phi = log_phistar - np.log10(ratio**gamma1 + ratio**gamma2)

    if h_h:
        log_phi = log_phi - 3.0 * np.log10(h_h)
    return log_phi


def m1450_from_lbol(log_lbol):
    """
    M_1450 (rest-frame AB absolute magnitude) from log10(L_bol [erg/s]).

    L_bol -> L_1450 = lambda*L_lambda(1450A) via the constant bolometric
    correction of Runnoe, Brotherton & Shang (2012); L_1450 -> L_nu(1450A)
    -> f_nu at the AB-magnitude reference distance of 10 pc -> M_1450 via
    the Oke & Gunn (1983) AB zero point. See BH_LUMINOSITY_CONVERSIONS.md.
    """
    l_1450 = 10**log_lbol / BC_1450                      # lambda*L_lambda [erg/s]
    l_nu   = l_1450 * WAVELENGTH_1450_CM / C_LIGHT_CGS    # L_nu [erg/s/Hz]
    d_10pc_cm = 10.0 * PC_TO_CM
    f_nu = l_nu / (4.0 * np.pi * d_10pc_cm**2)            # erg/s/cm^2/Hz
    return -2.5 * np.log10(f_nu) - AB_ZEROPOINT


def lit_z_mask(source_z, plot_z, tol=LIT_Z_TOL):
    """Boolean mask selecting entries of a literature source whose redshift
    is within `tol` of the panel's own redshift (`plot_z`)."""
    return np.abs(np.asarray(source_z, dtype=float) - plot_z) <= tol


def plot_lit_points(ax, label, x, y, xerr=None, yerr=None,
                    xlolims=False, xuplims=False, lolims=False, uplims=False):
    """Overlay one literature source's points with its shared marker/colour
    (see LIT_STYLE in lrd_literature_data.py), adding a single legend entry."""
    style = LIT_STYLE[label]
    ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt=style['marker'],
                color=style['color'], mec='black', mew=0.6, ms=style['ms'],
                capsize=3, elinewidth=1.0, zorder=8, ls='none', label=label,
                xlolims=xlolims, xuplims=xuplims, lolims=lolims, uplims=uplims)


def lit_legend_handles(labels):
    """Legend proxies for plot_lit_points() sources (errorbar handles don't
    reproduce cleanly when a legend is built from an explicit handles list)."""
    return [Line2D([0], [0], marker=LIT_STYLE[l]['marker'], color='w',
                   markerfacecolor=LIT_STYLE[l]['color'], markeredgecolor='black',
                   markersize=LIT_STYLE[l]['ms'] - 1, linestyle='none', label=l)
            for l in labels]

# ============================================================================
# MAIN PLOT
# ============================================================================

def plot_panel_a(data, snap_col, output_file,
                 show_lrd=True, use_fbh=True,
                 bhar_floor=LRD_BHAR_DEFAULT, z_override=None, show_lit=True,
                 mask_seeds=True, color_by_acctype=False):

    bh_mass = data['bh_mass']
    mdot    = data['mdot_msun_yr']
    medd    = data['mdot_edd']
    star    = data['stellar_mass']
    seed    = data['seed_mass']
    acc_type = data.get('acc_type', np.array([]))

    if len(bh_mass) == 0:
        print('  ERROR: no accretion events found in the selected column(s).')
        print('         Try a later column (-s 15), a wider --window, '
              'or a different --catalogue.')
        return

    # quality: positive, finite mass & rate
    valid = (
        (bh_mass > 0) & (mdot > 0) & np.isfinite(bh_mass) & np.isfinite(mdot)
    )
    if mask_seeds:
        valid &= mask_ungrown_seeds(bh_mass, seed)
    bh_mass = bh_mass[valid]; mdot = mdot[valid]
    medd    = medd[valid];    star = star[valid]
    acc_type = acc_type[valid] if len(acc_type) == len(valid) else \
        np.full(valid.sum(), -1.0)

    log_mbh  = np.log10(bh_mass)
    log_mdot = np.log10(mdot)

    print(f'  Accretion events plotted: {len(log_mbh):,}')

    # ── selection masks ───────────────────────────────────────────────────
    if use_fbh:
        _, lrd_red, lrd_blue = compute_selection(bh_mass, mdot, medd, star, bhar_floor)
    else:
        bhar_pass = mdot >= bhar_floor
        edd_pass  = mdot >= np.where(medd > 0, medd, np.inf)
        lrd_red   = bhar_pass & edd_pass
        lrd_blue  = np.zeros(len(log_mbh), dtype=bool)

    print(f'  LRD red  (full):    {lrd_red.sum():,}')
    if use_fbh:
        print(f'  LRD blue (f_BH<3%): {lrd_blue.sum():,}')

    redshift = z_override if z_override is not None else snap_to_z(snap_col)

    # literature Mdot_BH, derived once and reused for both the axis lock and
    # the overlay further down -- each source is restricted to the points
    # within LIT_Z_TOL of this panel's own redshift
    lit_log_mbh, lit_log_mdot = [], []
    pang_m  = lit_z_mask(PANG26['z'], redshift)
    mathee_m = lit_z_mask(MATHEE24['z'], redshift)
    lin_m   = lit_z_mask(LIN25['z'], redshift)
    if show_lit:
        t_mdot = PANG26['lambda_edd'][pang_m] * 10**eddington_mdot(PANG26['log_mbh'][pang_m])
        lit_log_mbh.append(PANG26['log_mbh'][pang_m]); lit_log_mdot.append(np.log10(t_mdot))
        m_mdot = mdot_from_lbol(np.log10(MATHEE24['lbol_1e44'][mathee_m]) + 44.0)
        lit_log_mbh.append(MATHEE24['log_mbh'][mathee_m]); lit_log_mdot.append(np.log10(m_mdot))
        l_mdot = mdot_from_lbol(LIN25['log_lbol'][lin_m])
        lit_log_mbh.append(LIN25['log_mbh'][lin_m]); lit_log_mdot.append(np.log10(l_mdot))

    # ── figure ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.minorticks_on()

    selected = (lrd_red | lrd_blue) if show_lrd else np.zeros(len(log_mbh), dtype=bool)
    x_lo, x_hi = lock_axis_range(*PANEL_A_XLIM,
                                 must_include=[log_mbh[selected], *lit_log_mbh],
                                 axis_name='panel a x-axis')
    y_lo, y_hi = lock_axis_range(*PANEL_A_YLIM,
                                 must_include=[log_mdot[selected], *lit_log_mdot],
                                 axis_name='panel a y-axis')

    # ── LRD shaded region ─────────────────────────────────────────────────
    if show_lrd:
        x_fill  = np.linspace(x_lo, x_hi, 400)
        y_lower = np.maximum(eddington_mdot(x_fill), np.log10(bhar_floor))
        ax.fill_between(x_fill, y_lower, y_hi,
                        color='#D32F2F', alpha=0.08, zorder=0)

    # ── grey/coloured background scatter ────────────────────────────────────
    bg = ~(lrd_red | lrd_blue) if show_lrd else np.ones(len(log_mbh), dtype=bool)
    x_bg, y_bg, type_bg = log_mbh[bg], log_mdot[bg], acc_type[bg]

    N_SCATTER = 30_000
    if len(x_bg) > N_SCATTER:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(x_bg), N_SCATTER, replace=False)
        x_sc, y_sc, type_sc = x_bg[idx], y_bg[idx], type_bg[idx]
    else:
        x_sc, y_sc, type_sc = x_bg, y_bg, type_bg

    if color_by_acctype:
        is_other = np.ones(len(x_sc), dtype=bool)
        for t, (color, _label) in ACC_TYPE_COLORS.items():
            m = type_sc == t
            is_other &= ~m
            ax.scatter(x_sc[m], y_sc[m], s=4, color=color, alpha=0.35,
                       linewidths=0, rasterized=True, zorder=1.1)
        ax.scatter(x_sc[is_other], y_sc[is_other], s=4, color='#999999',
                   alpha=0.20, linewidths=0, rasterized=True, zorder=1)
    else:
        ax.scatter(x_sc, y_sc, s=4, color='#999999', alpha=0.20,
                   linewidths=0, rasterized=True, zorder=1)

    # ── KDE contours (68/95/99.7%) ────────────────────────────────────────
    if color_by_acctype:
        any_contour = False
        matched = np.zeros(len(x_bg), dtype=bool)
        for t, (color, _label) in ACC_TYPE_COLORS.items():
            m = type_bg == t
            matched |= m
            if kde_contour(ax, x_bg[m], y_bg[m], x_lo, x_hi, y_lo, y_hi, color):
                any_contour = True
        if kde_contour(ax, x_bg[~matched], y_bg[~matched], x_lo, x_hi, y_lo, y_hi,
                       '#333333'):
            any_contour = True
        if not any_contour:
            print('  (too few background points for KDE contours)')
    else:
        if not kde_contour(ax, x_bg, y_bg, x_lo, x_hi, y_lo, y_hi, '#333333'):
            print('  (too few background points for KDE contours)')

    # ── LRD coloured dots ─────────────────────────────────────────────────
    if show_lrd:
        if lrd_blue.sum() > 0:
            ax.scatter(log_mbh[lrd_blue], log_mdot[lrd_blue],
                       s=35, color='#F57C00', edgecolors='white',
                       linewidths=0.3, zorder=5)
        if lrd_red.sum() > 0:
            ax.scatter(log_mbh[lrd_red], log_mdot[lrd_red],
                       s=35, color='#C62828', edgecolors='white',
                       linewidths=0.3, zorder=6)

    # ── reference lines ───────────────────────────────────────────────────
    x_ref = np.linspace(x_lo, x_hi, 400)
    y_edd = eddington_mdot(x_ref)
    ax.plot(x_ref, y_edd, color='#C62828', lw=1.8, ls='-', zorder=4)
    ax.plot(x_ref, y_edd + 1.0, color='#E65100', lw=1.8, ls='-', zorder=4)

    ax.axhline(np.log10(LRD_BHAR_DEFAULT), color='#C62828', lw=1.3,
               ls='--', zorder=3, alpha=0.85)
    ax.axhline(np.log10(LRD_BHAR_ALT), color='#C62828', lw=1.0,
               ls=':', zorder=3, alpha=0.70)

    ax.annotate(
        rf'$\dot{{M}}_{{\rm BH}} = {LRD_BHAR_DEFAULT}\,M_\odot\,\mathrm{{yr}}^{{-1}}$',
        xy=(x_hi - 0.2, np.log10(LRD_BHAR_DEFAULT) + 0.15),
        fontsize=12, color='#C62828', ha='right',
    )

    if show_lrd:
        ax.text(x_lo + 3.3, y_hi - 1.45, 'LRD selection',
                fontsize=13, color='#C62828')

    # ── literature overlay (Mdot_BH derived from L_bol or lambda_Edd) ──────
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

    # ── axes & decorations ────────────────────────────────────────────────
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel(r'$M_{\rm BH}\ [M_\odot]$', fontsize=18)
    ax.set_ylabel(r'$\dot{M}_{\rm BH}\ [M_\odot\,\mathrm{yr}^{-1}]$', fontsize=18)
    ax.set_xticks(np.arange(int(np.ceil(x_lo)), int(np.floor(x_hi)) + 1, 2))

    ax.text(0.97, 0.04, rf'$z = {redshift:.1f}$',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=16)
   # ax.text(0.03, 0.97, 'a', transform=ax.transAxes, ha='left', va='top',
    #        fontsize=18, fontweight='bold')

    # ── legend ────────────────────────────────────────────────────────────
    handles = [
        Line2D([0], [0], color='#C62828', lw=1.8,
               label=r'$\dot{M}_{\rm BH} = \dot{M}_{\rm Edd}$'),
        Line2D([0], [0], color='#E65100', lw=1.8,
               label=r'$\dot{M}_{\rm BH} = 10\,\dot{M}_{\rm Edd}$'),
    ]
    if color_by_acctype:
        for _t, (color, label) in ACC_TYPE_COLORS.items():
            handles.append(
                Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                       markersize=7, label=label))
        handles.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#999999',
                   markersize=7, label='Radio mode / unknown'))
    if show_lrd:
        handles.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#C62828',
                   markersize=7,
                   label=(r'LRD ($f_{\rm BH}\geq 3\%$)' if use_fbh else 'LRD')))
        if use_fbh:
            handles.append(
                Line2D([0], [0], marker='o', color='w',
                       markerfacecolor='#F57C00', markersize=7,
                       label=r'LRD ($f_{\rm BH}<3\%$)'))
    handles += lit_legend_handles(lit_labels)
    ax.legend(handles=handles, loc='upper left', fontsize=12,
              handlelength=1.6, handletextpad=0.5)

    plt.tight_layout()
    plt.savefig(output_file, dpi=140, bbox_inches='tight')
    plt.close()
    print(f'✓  Saved  →  {output_file}')


def plot_panel_b(data, snap_col, output_file,
                 show_lrd=True, bhar_floor=LRD_BHAR_DEFAULT, z_override=None,
                 show_lit=True, mask_seeds=True):
    """
    Panel (b): f_BH = log10(M_BH / M_star)  vs  log10(M_BH [M_sun]).
    Always splits the LRD selection by f_BH (that split is the entire point
    of this panel), regardless of the --no-fbh flag used for panel a.
    """
    bh_mass = data['bh_mass']
    mdot    = data['mdot_msun_yr']
    medd    = data['mdot_edd']
    star    = data['stellar_mass']
    seed    = data['seed_mass']

    if len(bh_mass) == 0:
        print('  ERROR: no accretion events found in the selected column(s).')
        return

    # quality: positive, finite mass & rate, and a usable stellar mass for f_BH
    valid = (
        (bh_mass > 0) & (mdot > 0) & (star > 0) &
        np.isfinite(bh_mass) & np.isfinite(mdot) & np.isfinite(star)
    )
    if mask_seeds:
        valid &= mask_ungrown_seeds(bh_mass, seed)
    bh_mass = bh_mass[valid]; mdot = mdot[valid]
    medd    = medd[valid];    star = star[valid]

    log_mbh = np.log10(bh_mass)

    f_bh, lrd_red, lrd_blue = compute_selection(bh_mass, mdot, medd, star, bhar_floor)
    log_fbh = np.log10(f_bh)

    print(f'  Panel b events plotted: {len(log_mbh):,}')
    print(f'  LRD red  (full):    {lrd_red.sum():,}')
    print(f'  LRD blue (f_BH<3%): {lrd_blue.sum():,}')

    redshift = z_override if z_override is not None else snap_to_z(snap_col)
    pang_m = lit_z_mask(PANG26['z'], redshift)
    lit_log_fbh = (PANG26['log_mbh'][pang_m] - PANG26['log_mstar'][pang_m]) if show_lit else []

    # ── figure ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.minorticks_on()

    selected = (lrd_red | lrd_blue) if show_lrd else np.zeros(len(log_mbh), dtype=bool)
    x_lo, x_hi = lock_axis_range(*PANEL_B_XLIM,
                                 must_include=[log_mbh[selected],
                                               PANG26['log_mbh'][pang_m] if show_lit else []],
                                 axis_name='panel b x-axis')
    y_lo, y_hi = lock_axis_range(*PANEL_B_YLIM,
                                 must_include=[log_fbh[selected], lit_log_fbh],
                                 axis_name='panel b y-axis')

    log_fbh_thresh = np.log10(LRD_FBHM_THRESH)

    # ── LRD shaded region (flat cut at f_BH = 3%) ─────────────────────────
    if show_lrd:
        ax.fill_between([x_lo, x_hi], log_fbh_thresh, y_hi,
                        color='#D32F2F', alpha=0.08, zorder=0)

    # ── grey background scatter ───────────────────────────────────────────
    bg = ~(lrd_red | lrd_blue) if show_lrd else np.ones(len(log_mbh), dtype=bool)
    x_bg, y_bg = log_mbh[bg], log_fbh[bg]

    N_SCATTER = 30_000
    if len(x_bg) > N_SCATTER:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(x_bg), N_SCATTER, replace=False)
        x_sc, y_sc = x_bg[idx], y_bg[idx]
    else:
        x_sc, y_sc = x_bg, y_bg
    ax.scatter(x_sc, y_sc, s=4, color='#999999', alpha=0.20,
               linewidths=0, rasterized=True, zorder=1)

    # ── KDE contours (68/95/99.7%) ────────────────────────────────────────
    if len(x_bg) >= 50:
        try:
            N_KDE = 60_000
            if len(x_bg) > N_KDE:
                rng = np.random.default_rng(77)
                idx = rng.choice(len(x_bg), N_KDE, replace=False)
                xk, yk = x_bg[idx], y_bg[idx]
            else:
                xk, yk = x_bg, y_bg

            kde  = gaussian_kde(np.vstack([xk, yk]), bw_method='scott')
            xi   = np.linspace(x_lo, x_hi, 250)
            yi   = np.linspace(y_lo, y_hi, 250)
            Xi, Yi = np.meshgrid(xi, yi)
            Zi   = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)

            z_sort = np.sort(Zi.ravel())[::-1]
            z_cum  = np.cumsum(z_sort) / z_sort.sum()
            def lvl(frac):
                return z_sort[min(np.searchsorted(z_cum, frac), len(z_sort) - 1)]
            levels = sorted([lvl(0.683), lvl(0.954), lvl(0.997)])

            ax.contour(Xi, Yi, Zi, levels=levels,
                       colors='#000000',
                       linewidths=[0.9, 1.2, 1.6],
                       linestyles=[':', '--', '-'], zorder=2)
        except Exception as e:
            print(f'  WARNING: KDE contours skipped ({e})')
    else:
        print('  (too few background points for KDE contours)')

    # ── LRD coloured dots ─────────────────────────────────────────────────
    if show_lrd:
        if lrd_blue.sum() > 0:
            ax.scatter(log_mbh[lrd_blue], log_fbh[lrd_blue],
                       s=35, color='#F57C00', edgecolors='white',
                       linewidths=0.3, zorder=5)
        if lrd_red.sum() > 0:
            ax.scatter(log_mbh[lrd_red], log_fbh[lrd_red],
                       s=35, color='#C62828', edgecolors='white',
                       linewidths=0.3, zorder=6)

    # ── reference lines ───────────────────────────────────────────────────
    ax.axhline(np.log10(0.1), color='#F57C00', lw=1.8, zorder=4)
    ax.axhline(log_fbh_thresh, color='#C62828', lw=1.8, zorder=4)

    ax.annotate(
        r'$M_{\rm BH}/M_\star = 0.1$',
        xy=(x_hi - 4.3, np.log10(0.1)), xytext=(x_hi - 3.6, np.log10(0.1) + 0.55),
        fontsize=13, color='#F57C00', ha='left',
        arrowprops=dict(arrowstyle='->', color='#F57C00', lw=1.2),
    )
    ax.text(x_hi - 0.2, log_fbh_thresh + 0.10, r'$f_{\rm BH} = 3\%$',
            fontsize=13, color='#C62828', ha='right')

    if show_lrd:
        ax.text(x_hi - 3.3, y_hi - 0.5, 'LRD selection',
                fontsize=13, color='#C62828')

    # ── literature overlay (only Pang+26 gives both M_BH and M_star) ────
    if show_lit and pang_m.any():
        t_log_fbh = PANG26['log_mbh'][pang_m] - PANG26['log_mstar'][pang_m]
        t_fbh_err = np.sqrt(PANG26['log_mbh_err'][pang_m]**2 + PANG26['log_mstar_err'][pang_m]**2)
        plot_lit_points(ax, 'Pang+26', PANG26['log_mbh'][pang_m], t_log_fbh,
                        xerr=PANG26['log_mbh_err'][pang_m], yerr=t_fbh_err)
        ax.legend(handles=lit_legend_handles(['Pang+26']),
                  loc='upper left', fontsize=12)

    # ── axes & decorations ────────────────────────────────────────────────
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel(r'$M_{\rm BH}\ [M_\odot]$', fontsize=18)
    ax.set_ylabel(r'$f_{\rm BH} = M_{\rm BH}/M_\star$', fontsize=18)
    ax.set_xticks(np.arange(int(np.ceil(x_lo)), int(np.floor(x_hi)) + 1, 2))

    ax.text(0.97, 0.04, rf'$z = {redshift:.1f}$',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=16)

    plt.tight_layout()
    plt.savefig(output_file, dpi=140, bbox_inches='tight')
    plt.close()
    print(f'✓  Saved  →  {output_file}')


def _line_rotation_deg(slope, x_lo, x_hi, y_lo, y_hi):
    """Approximate on-screen rotation (degrees) of a data-space line of the
    given slope, for a square axes box -- so text can follow the line."""
    return np.degrees(np.arctan(slope * (x_hi - x_lo) / (y_hi - y_lo)))


def plot_panel_c(data, snap_col, output_file,
                 show_lrd=True, bhar_floor=LRD_BHAR_DEFAULT, z_override=None,
                 show_lit=True, mask_seeds=True):
    """
    Panel (c): log10(M_BH)  vs  log10(M_star), styled after Fig. 7 of
    Kocevski et al. (2023).  Shows the Kormendy & Ho (2013) M_BH-M_star
    relation and constant M_BH/M_star ratio lines instead of the paper's
    observational comparison samples, with our own LRD/partial-LRD selection
    (from panels a/b) marked on top.
    """
    bh_mass = data['bh_mass']
    mdot    = data['mdot_msun_yr']
    medd    = data['mdot_edd']
    star    = data['stellar_mass']
    seed    = data['seed_mass']

    if len(bh_mass) == 0:
        print('  ERROR: no accretion events found in the selected column(s).')
        return

    valid = (
        (bh_mass > 0) & (mdot > 0) & (star > 0) &
        np.isfinite(bh_mass) & np.isfinite(mdot) & np.isfinite(star)
    )
    if mask_seeds:
        valid &= mask_ungrown_seeds(bh_mass, seed)
    bh_mass = bh_mass[valid]; mdot = mdot[valid]
    medd    = medd[valid];    star = star[valid]

    log_mstar = np.log10(star)
    log_mbh   = np.log10(bh_mass)

    _, lrd_red, lrd_blue = compute_selection(bh_mass, mdot, medd, star, bhar_floor)

    print(f'  Panel c events plotted: {len(log_mbh):,}')
    print(f'  LRD red  (full):    {lrd_red.sum():,}')
    print(f'  LRD blue (f_BH<3%): {lrd_blue.sum():,}')

    redshift = z_override if z_override is not None else snap_to_z(snap_col)
    pang_m   = lit_z_mask(PANG26['z'], redshift)
    furtak_m = lit_z_mask(FURTAK23['z'], redshift)

    # ── figure ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.minorticks_on()

    selected = (lrd_red | lrd_blue) if show_lrd else np.zeros(len(log_mbh), dtype=bool)
    lit_mstar, lit_mbh = [], []
    if show_lit:
        lit_mstar.append(PANG26['log_mstar'][pang_m])
        lit_mbh.append(PANG26['log_mbh'][pang_m])
        if furtak_m:
            lit_mstar.append([FURTAK23['log_mstar_upper_limit']])
            lit_mbh.append([FURTAK23['log_mbh']])
    x_lo, x_hi = lock_axis_range(*PANEL_C_XLIM,
                                 must_include=[log_mstar[selected], *lit_mstar],
                                 axis_name='panel c x-axis')
    y_lo, y_hi = lock_axis_range(*PANEL_C_YLIM,
                                 must_include=[log_mbh[selected], *lit_mbh],
                                 axis_name='panel c y-axis')

    x_ref = np.linspace(x_lo, x_hi, 400)

    # ── Kormendy & Ho (2013) relation + intrinsic scatter band ────────────
    y_kh = kormendy_ho_mbh(x_ref)
    ax.fill_between(x_ref, y_kh - KORMENDY_HO_SCATTER, y_kh + KORMENDY_HO_SCATTER,
                    color='#999999', alpha=0.35, zorder=0)
    kh_line, = ax.plot(x_ref, y_kh, color='black', lw=1.8, zorder=2,
                       label='Kormendy & Ho 2013')

    # ── constant M_BH/M_star ratio lines ───────────────────────────────────
    ratio_rot = _line_rotation_deg(1.0, x_lo, x_hi, y_lo, y_hi)
    for ratio in MASS_RATIO_LINES:
        y_ratio = x_ref + np.log10(ratio)
        ax.plot(x_ref, y_ratio, color='black', lw=1.0, ls='--', zorder=2)
        x_lab = x_lo + 0.12 * (x_hi - x_lo)
        y_lab = x_lab + np.log10(ratio)
        if y_lo < y_lab < y_hi:
            label = rf'$M_{{\rm BH}}/M_\star = {ratio:g}$'
            ax.text(x_lab, y_lab - 0.12, label, fontsize=10.5, color='black',
                    ha='left', va='top', rotation=ratio_rot, rotation_mode='anchor')

    # ── grey background scatter + KDE contours (our own model, not obs.) ──
    bg = ~(lrd_red | lrd_blue) if show_lrd else np.ones(len(log_mbh), dtype=bool)
    x_bg, y_bg = log_mstar[bg], log_mbh[bg]

    N_SCATTER = 30_000
    if len(x_bg) > N_SCATTER:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(x_bg), N_SCATTER, replace=False)
        x_sc, y_sc = x_bg[idx], y_bg[idx]
    else:
        x_sc, y_sc = x_bg, y_bg
    ax.scatter(x_sc, y_sc, s=4, color='#999999', alpha=0.20,
               linewidths=0, rasterized=True, zorder=1)

    if len(x_bg) >= 50:
        try:
            N_KDE = 60_000
            if len(x_bg) > N_KDE:
                rng = np.random.default_rng(77)
                idx = rng.choice(len(x_bg), N_KDE, replace=False)
                xk, yk = x_bg[idx], y_bg[idx]
            else:
                xk, yk = x_bg, y_bg

            kde  = gaussian_kde(np.vstack([xk, yk]), bw_method='scott')
            xi   = np.linspace(x_lo, x_hi, 250)
            yi   = np.linspace(y_lo, y_hi, 250)
            Xi, Yi = np.meshgrid(xi, yi)
            Zi   = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)

            z_sort = np.sort(Zi.ravel())[::-1]
            z_cum  = np.cumsum(z_sort) / z_sort.sum()
            def lvl(frac):
                return z_sort[min(np.searchsorted(z_cum, frac), len(z_sort) - 1)]
            levels = sorted([lvl(0.683), lvl(0.954), lvl(0.997)])

            ax.contour(Xi, Yi, Zi, levels=levels,
                       colors='#4A4A8A',
                       linewidths=[0.9, 1.2, 1.6],
                       linestyles=[':', '--', '-'], zorder=1.5)
        except Exception as e:
            print(f'  WARNING: KDE contours skipped ({e})')
    else:
        print('  (too few background points for KDE contours)')

    # ── LRD/partial-LRD points (identified in panels a/b) ─────────────────
    if show_lrd:
        if lrd_blue.sum() > 0:
            ax.scatter(log_mstar[lrd_blue], log_mbh[lrd_blue],
                       s=35, color='#F57C00', edgecolors='white',
                       linewidths=0.3, zorder=5)
        if lrd_red.sum() > 0:
            ax.scatter(log_mstar[lrd_red], log_mbh[lrd_red],
                       s=35, color='#C62828', edgecolors='white',
                       linewidths=0.3, zorder=6)

    # ── literature overlay ─────────────────────────────────────────────────
    lit_labels = []
    if show_lit:
        if pang_m.any():
            plot_lit_points(ax, 'Pang+26', PANG26['log_mstar'][pang_m], PANG26['log_mbh'][pang_m],
                            xerr=PANG26['log_mstar_err'][pang_m], yerr=PANG26['log_mbh_err'][pang_m])
            lit_labels.append('Pang+26')

        # Furtak+23: single lensed z=7.04 point; M_star is an upper limit (left arrow)
        if furtak_m:
            plot_lit_points(ax, 'Furtak+23', [FURTAK23['log_mstar_upper_limit']],
                            [FURTAK23['log_mbh']],
                            xerr=[[0.4], [0.0]],
                            yerr=[[FURTAK23['log_mbh_err_lo']], [FURTAK23['log_mbh_err_hi']]],
                            xuplims=True)
            lit_labels.append('Furtak+23')

    # ── axes & decorations ────────────────────────────────────────────────
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel(r'$\log\,M_\star\ [M_\odot]$', fontsize=18)
    ax.set_ylabel(r'$\log\,M_{\rm BH}\ [M_\odot]$', fontsize=18)

    ax.text(0.97, 0.04, rf'$z = {redshift:.1f}$',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=16)

    # ── legend ────────────────────────────────────────────────────────────
    handles = [kh_line]
    if show_lrd:
        handles += [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#C62828',
                   markersize=7, label=r'LRD ($f_{\rm BH}\geq 3\%$)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#F57C00',
                   markersize=7, label=r'LRD ($f_{\rm BH}<3\%$)'),
        ]
    handles += lit_legend_handles(lit_labels)
    ax.legend(handles=handles, loc='upper left', fontsize=12,
              handlelength=1.6, handletextpad=0.5)

    plt.tight_layout()
    plt.savefig(output_file, dpi=140, bbox_inches='tight')
    plt.close()
    print(f'✓  Saved  →  {output_file}')


def plot_panel_d(data, snap_col, output_file,
                 show_lrd=True, bhar_floor=LRD_BHAR_DEFAULT, z_override=None,
                 show_lit=True, mask_seeds=True):
    """
    Panel (d): log10(L_bol [erg/s])  vs  log10(M_BH [M_sun]), styled after
    Fig. 6 of Kocevski et al. (2023).  L_bol is derived from Mdot_BH assuming
    a constant radiative efficiency (see lbol_from_mdot / BH_LUMINOSITY_
    CONVERSIONS.md).  Shows lambda_Edd = 1, 0.1, 0.01 reference lines instead
    of the paper's observational comparison samples.
    """
    bh_mass = data['bh_mass']
    mdot    = data['mdot_msun_yr']
    medd    = data['mdot_edd']
    star    = data['stellar_mass']
    seed    = data['seed_mass']

    if len(bh_mass) == 0:
        print('  ERROR: no accretion events found in the selected column(s).')
        return

    valid = (
        (bh_mass > 0) & (mdot > 0) & (star > 0) &
        np.isfinite(bh_mass) & np.isfinite(mdot) & np.isfinite(star)
    )
    if mask_seeds:
        valid &= mask_ungrown_seeds(bh_mass, seed)
    bh_mass = bh_mass[valid]; mdot = mdot[valid]
    medd    = medd[valid];    star = star[valid]

    log_mbh  = np.log10(bh_mass)
    log_lbol = lbol_from_mdot(mdot)

    _, lrd_red, lrd_blue = compute_selection(bh_mass, mdot, medd, star, bhar_floor)

    print(f'  Panel d events plotted: {len(log_mbh):,}')
    print(f'  LRD red  (full):    {lrd_red.sum():,}')
    print(f'  LRD blue (f_BH<3%): {lrd_blue.sum():,}')

    redshift = z_override if z_override is not None else snap_to_z(snap_col)
    pang_m   = lit_z_mask(PANG26['z'], redshift)
    mathee_m = lit_z_mask(MATHEE24['z'], redshift)
    lin_m    = lit_z_mask(LIN25['z'], redshift)
    furtak_m = lit_z_mask(FURTAK23['z'], redshift)

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

    # ── figure ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.minorticks_on()

    selected = (lrd_red | lrd_blue) if show_lrd else np.zeros(len(log_mbh), dtype=bool)
    x_lo, x_hi = lock_axis_range(*PANEL_D_XLIM,
                                 must_include=[log_mbh[selected], *lit_log_mbh],
                                 axis_name='panel d x-axis')
    y_lo, y_hi = lock_axis_range(*PANEL_D_YLIM,
                                 must_include=[log_lbol[selected], *lit_log_lbol],
                                 axis_name='panel d y-axis')

    x_ref = np.linspace(x_lo, x_hi, 400)

    # ── Eddington ratio reference lines ───────────────────────────────────
    line_rot = _line_rotation_deg(1.0, x_lo, x_hi, y_lo, y_hi)
    for lam in EDDINGTON_RATIO_LINES:
        y_line = eddington_luminosity(x_ref) + np.log10(lam)
        ax.plot(x_ref, y_line, color='#777777', lw=1.2, ls='--', zorder=2)
        x_lab = x_lo + 0.60 * (x_hi - x_lo)
        y_lab = eddington_luminosity(x_lab) + np.log10(lam)
        if y_lo < y_lab < y_hi:
            label = rf'$\lambda_{{\rm Edd}} = {lam:g}$'
            ax.text(x_lab, y_lab - 0.12, label, fontsize=10.5, color='#555555',
                    ha='left', va='top', rotation=line_rot, rotation_mode='anchor')

    # ── grey background scatter + KDE contours ────────────────────────────
    bg = ~(lrd_red | lrd_blue) if show_lrd else np.ones(len(log_mbh), dtype=bool)
    x_bg, y_bg = log_mbh[bg], log_lbol[bg]

    N_SCATTER = 30_000
    if len(x_bg) > N_SCATTER:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(x_bg), N_SCATTER, replace=False)
        x_sc, y_sc = x_bg[idx], y_bg[idx]
    else:
        x_sc, y_sc = x_bg, y_bg
    ax.scatter(x_sc, y_sc, s=4, color='#999999', alpha=0.20,
               linewidths=0, rasterized=True, zorder=1)

    if len(x_bg) >= 50:
        try:
            N_KDE = 60_000
            if len(x_bg) > N_KDE:
                rng = np.random.default_rng(77)
                idx = rng.choice(len(x_bg), N_KDE, replace=False)
                xk, yk = x_bg[idx], y_bg[idx]
            else:
                xk, yk = x_bg, y_bg

            kde  = gaussian_kde(np.vstack([xk, yk]), bw_method='scott')
            xi   = np.linspace(x_lo, x_hi, 250)
            yi   = np.linspace(y_lo, y_hi, 250)
            Xi, Yi = np.meshgrid(xi, yi)
            Zi   = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)

            z_sort = np.sort(Zi.ravel())[::-1]
            z_cum  = np.cumsum(z_sort) / z_sort.sum()
            def lvl(frac):
                return z_sort[min(np.searchsorted(z_cum, frac), len(z_sort) - 1)]
            levels = sorted([lvl(0.683), lvl(0.954), lvl(0.997)])

            ax.contour(Xi, Yi, Zi, levels=levels,
                       colors='#4A4A8A',
                       linewidths=[0.9, 1.2, 1.6],
                       linestyles=[':', '--', '-'], zorder=1.5)
        except Exception as e:
            print(f'  WARNING: KDE contours skipped ({e})')
    else:
        print('  (too few background points for KDE contours)')

    # ── LRD/partial-LRD points ──────────────────────────────────────────────
    if show_lrd:
        if lrd_blue.sum() > 0:
            ax.scatter(log_mbh[lrd_blue], log_lbol[lrd_blue],
                       s=35, color='#F57C00', edgecolors='white',
                       linewidths=0.3, zorder=5)
        if lrd_red.sum() > 0:
            ax.scatter(log_mbh[lrd_red], log_lbol[lrd_red],
                       s=35, color='#C62828', edgecolors='white',
                       linewidths=0.3, zorder=6)

    # ── literature overlay ─────────────────────────────────────────────────
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

    # ── axes & decorations ────────────────────────────────────────────────
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel(r'$\log\,M_{\rm BH}\ [M_\odot]$', fontsize=18)
    ax.set_ylabel(r'$\log\,L_{\rm bol}\ [{\rm erg\,s^{-1}}]$', fontsize=18)

    ax.text(0.97, 0.04, rf'$z = {redshift:.1f}$',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=16)

    handles = []
    if show_lrd:
        handles += [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#C62828',
                   markersize=7, label=r'LRD ($f_{\rm BH}\geq 3\%$)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#F57C00',
                   markersize=7, label=r'LRD ($f_{\rm BH}<3\%$)'),
        ]
    handles += lit_legend_handles(lit_labels)
    if handles:
        ax.legend(handles=handles, loc='upper left', fontsize=12,
                  handlelength=1.6, handletextpad=0.5)

    plt.tight_layout()
    plt.savefig(output_file, dpi=140, bbox_inches='tight')
    plt.close()
    print(f'✓  Saved  →  {output_file}')


def plot_panel_e(data, snap_col, output_file,
                 show_lrd=True, bhar_floor=LRD_BHAR_DEFAULT, z_override=None,
                 show_lit=True, mask_seeds=True):
    """
    Panel (e): M_1450 (rest-frame UV absolute magnitude)  vs  log10(M_BH),
    styled after the M_UV-M_BH comparison plots common in the JWST LRD
    literature (e.g. Bogdan et al. 2024; Kokorev et al. 2023).  M_1450 is
    derived from Mdot_BH via L_bol and a bolometric correction -- see
    lbol_from_mdot / m1450_from_lbol / BH_LUMINOSITY_CONVERSIONS.md.
    """
    bh_mass = data['bh_mass']
    mdot    = data['mdot_msun_yr']
    medd    = data['mdot_edd']
    star    = data['stellar_mass']
    seed    = data['seed_mass']

    if len(bh_mass) == 0:
        print('  ERROR: no accretion events found in the selected column(s).')
        return

    valid = (
        (bh_mass > 0) & (mdot > 0) & (star > 0) &
        np.isfinite(bh_mass) & np.isfinite(mdot) & np.isfinite(star)
    )
    if mask_seeds:
        valid &= mask_ungrown_seeds(bh_mass, seed)
    bh_mass = bh_mass[valid]; mdot = mdot[valid]
    medd    = medd[valid];    star = star[valid]

    log_mbh = np.log10(bh_mass)
    m1450   = m1450_from_lbol(lbol_from_mdot(mdot))

    _, lrd_red, lrd_blue = compute_selection(bh_mass, mdot, medd, star, bhar_floor)

    print(f'  Panel e events plotted: {len(log_mbh):,}')
    print(f'  LRD red  (full):    {lrd_red.sum():,}')
    print(f'  LRD blue (f_BH<3%): {lrd_blue.sum():,}')

    redshift = z_override if z_override is not None else snap_to_z(snap_col)
    mathee_m = lit_z_mask(MATHEE24['z'], redshift)
    labbe_m  = lit_z_mask(LABBE25['z'], redshift)
    furtak_m = lit_z_mask(FURTAK23['z'], redshift)

    # ── figure ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.minorticks_on()

    selected = (lrd_red | lrd_blue) if show_lrd else np.zeros(len(log_mbh), dtype=bool)

    lit_muv, lit_mbh = [], []
    if show_lit:
        labbe_mbh_min = bh_mass_min_from_lbol(LABBE25['log_lbol'][labbe_m])
        lit_muv += [LABBE25['m1450'][labbe_m], MATHEE24['muv'][mathee_m]]
        lit_mbh += [np.log10(labbe_mbh_min), MATHEE24['log_mbh'][mathee_m]]
        if furtak_m:
            lit_muv.append([FURTAK23['muv']])
            lit_mbh.append([FURTAK23['log_mbh']])

    x_faint, x_bright = lock_axis_range(*PANEL_E_XLIM,
                                       must_include=[m1450[selected], *lit_muv],
                                       axis_name='panel e x-axis')
    y_lo, y_hi = lock_axis_range(*PANEL_E_YLIM,
                                 must_include=[log_mbh[selected], *lit_mbh],
                                 axis_name='panel e y-axis')

    # ── grey background scatter + KDE contours ────────────────────────────
    bg = ~(lrd_red | lrd_blue) if show_lrd else np.ones(len(log_mbh), dtype=bool)
    x_bg, y_bg = m1450[bg], log_mbh[bg]

    N_SCATTER = 30_000
    if len(x_bg) > N_SCATTER:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(x_bg), N_SCATTER, replace=False)
        x_sc, y_sc = x_bg[idx], y_bg[idx]
    else:
        x_sc, y_sc = x_bg, y_bg
    ax.scatter(x_sc, y_sc, s=4, color='#999999', alpha=0.20,
               linewidths=0, rasterized=True, zorder=1)

    if len(x_bg) >= 50:
        try:
            N_KDE = 60_000
            if len(x_bg) > N_KDE:
                rng = np.random.default_rng(77)
                idx = rng.choice(len(x_bg), N_KDE, replace=False)
                xk, yk = x_bg[idx], y_bg[idx]
            else:
                xk, yk = x_bg, y_bg

            kde  = gaussian_kde(np.vstack([xk, yk]), bw_method='scott')
            xi   = np.linspace(min(x_faint, x_bright), max(x_faint, x_bright), 250)
            yi   = np.linspace(y_lo, y_hi, 250)
            Xi, Yi = np.meshgrid(xi, yi)
            Zi   = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)

            z_sort = np.sort(Zi.ravel())[::-1]
            z_cum  = np.cumsum(z_sort) / z_sort.sum()
            def lvl(frac):
                return z_sort[min(np.searchsorted(z_cum, frac), len(z_sort) - 1)]
            levels = sorted([lvl(0.683), lvl(0.954), lvl(0.997)])

            ax.contour(Xi, Yi, Zi, levels=levels,
                       colors='#4A4A8A',
                       linewidths=[0.9, 1.2, 1.6],
                       linestyles=[':', '--', '-'], zorder=1.5)
        except Exception as e:
            print(f'  WARNING: KDE contours skipped ({e})')
    else:
        print('  (too few background points for KDE contours)')

    # ── LRD/partial-LRD points ──────────────────────────────────────────────
    if show_lrd:
        if lrd_blue.sum() > 0:
            ax.scatter(m1450[lrd_blue], log_mbh[lrd_blue],
                       s=35, color='#F57C00', edgecolors='white',
                       linewidths=0.3, zorder=5)
        if lrd_red.sum() > 0:
            ax.scatter(m1450[lrd_red], log_mbh[lrd_red],
                       s=35, color='#C62828', edgecolors='white',
                       linewidths=0.3, zorder=6)

    # ── literature overlay ─────────────────────────────────────────────────
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

        # Labbe+25 has no M_BH -- plot as an Eddington-limited LOWER LIMIT
        # (upward arrow) derived from L_bol, for their SED-classified AGN rows only.
        if labbe_m.any():
            plot_lit_points(ax, 'Labbe+25', LABBE25['m1450'][labbe_m], np.log10(labbe_mbh_min),
                            yerr=0.3, lolims=True)
            lit_labels.append('Labbe+25')

    # ── axes & decorations (x-axis runs faint -> bright, left to right) ────
    ax.set_xlim(x_faint, x_bright)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel(r'$M_{\rm UV,1450}$', fontsize=18)
    ax.set_ylabel(r'$\log\,M_{\rm BH}\ [M_\odot]$', fontsize=18)

    ax.text(0.97, 0.04, rf'$z = {redshift:.1f}$',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=16)

    handles = []
    if show_lrd:
        handles += [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#C62828',
                   markersize=7, label=r'LRD ($f_{\rm BH}\geq 3\%$)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#F57C00',
                   markersize=7, label=r'LRD ($f_{\rm BH}<3\%$)'),
        ]
    handles += lit_legend_handles(lit_labels)
    if handles:
        ax.legend(handles=handles, loc='upper left', fontsize=12,
                  handlelength=1.6, handletextpad=0.5)

    plt.tight_layout()
    plt.savefig(output_file, dpi=140, bbox_inches='tight')
    plt.close()
    print(f'✓  Saved  →  {output_file}')


def plot_panel_f(data, snap_col, output_file, volume_h3,
                 show_lrd=True, bhar_floor=LRD_BHAR_DEFAULT, z_override=None,
                 n_bins=40, h_h=None, show_lit=True, mask_seeds=True):
    """
    Panel (f): bolometric AGN luminosity function at the chosen redshift,

        phi(L_bol) = dN/dlog10(L_bol) / Volume   [Mpc^-3 h^3 dex^-1]

    using the same comoving-volume convention (box_size^3 * frac_volume_
    processed, in (Mpc/h)^3) as the accretion rate function in
    allresults-blackholes.py.  L_bol is derived from Mdot_BH via
    lbol_from_mdot().
    """
    bh_mass = data['bh_mass']
    mdot    = data['mdot_msun_yr']
    medd    = data['mdot_edd']
    star    = data['stellar_mass']
    seed    = data['seed_mass']

    if len(bh_mass) == 0:
        print('  ERROR: no accretion events found in the selected column(s).')
        return

    valid = (
        (bh_mass > 0) & (mdot > 0) & (star > 0) &
        np.isfinite(bh_mass) & np.isfinite(mdot) & np.isfinite(star)
    )
    if mask_seeds:
        valid &= mask_ungrown_seeds(bh_mass, seed)
    bh_mass = bh_mass[valid]; mdot = mdot[valid]
    medd    = medd[valid];    star = star[valid]

    log_lbol = lbol_from_mdot(mdot)
    _, lrd_red, lrd_blue = compute_selection(bh_mass, mdot, medd, star, bhar_floor)

    print(f'  Panel f events plotted: {len(log_lbol):,}')
    print(f'  LRD red  (full):    {lrd_red.sum():,}')
    print(f'  LRD blue (f_BH<3%): {lrd_blue.sum():,}')
    if not volume_h3:
        print('  WARNING: no simulation volume available; plotting raw counts, not phi.')

    # ── binning ─────────────────────────────────────────────────────────
    selected = (lrd_red | lrd_blue) if show_lrd else np.zeros(len(log_lbol), dtype=bool)
    lo, hi = lock_axis_range(*PANEL_F_XLIM, must_include=[log_lbol[selected]],
                             axis_name='panel f x-axis')

    bins = np.linspace(lo, hi, n_bins + 1)
    bw = bins[1] - bins[0]
    centres = 0.5 * (bins[:-1] + bins[1:])

    cats = [(log_lbol, 'Total', 'k', 'o')]
    if show_lrd:
        cats.append((log_lbol[lrd_red],  r'LRD ($f_{\rm BH}\geq 3\%$)', '#C62828', 'D'))
        cats.append((log_lbol[lrd_blue], r'LRD ($f_{\rm BH}<3\%$)',     '#F57C00', 's'))

    redshift = z_override if z_override is not None else snap_to_z(snap_col)

    # ── figure ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.minorticks_on()

    allv = []
    for values, label, color, marker in cats:
        counts, _ = np.histogram(values, bins=bins)
        pos = counts > 0
        y = counts / (bw * volume_h3) if volume_h3 else counts / bw
        logy = np.log10(y[pos])
        # Poisson counting error, propagated into log10 space:
        # sigma_log10(N) = sigma_N / (N * ln10) = 1 / (sqrt(N) * ln10)
        logy_err = 1.0 / (np.sqrt(counts[pos]) * np.log(10))

        ax.errorbar(centres[pos], logy, yerr=logy_err, fmt=marker, color=color,
                    mec='black', mew=0.5, ms=6, capsize=2.5, elinewidth=1.0,
                    ls='none', label=label, zorder=5)
        allv.extend(logy)

    # ── Shen et al. (2020) bolometric QLF model (global fit A) ─────────────
    # Only valid/requested for z = 1-7; the fit is unconstrained/extrapolated
    # below that range (see shen20_bolometric_qlf_logphi docstring). The
    # faint-end slope also makes phi formally diverge as L -> 0 (worse at
    # higher z -- an actual property of their double power-law fit, not a
    # bug here), so the curve is only drawn from SHEN20_MODEL_LOGLBOL_MIN
    # up -- roughly where Shen+20 themselves plot it (their Fig. 5 starts
    # at log L_bol ~ 43) -- and is kept OUT of the axis-lock's must_include
    # so that divergent tail can't blow up this panel's y-range.
    show_shen20 = show_lit and (1.0 <= redshift <= 7.0)
    if show_shen20:
        l_ref = np.linspace(max(lo, SHEN20_MODEL_LOGLBOL_MIN), hi, 200)
        shen20_logphi = shen20_bolometric_qlf_logphi(l_ref, redshift, h_h=h_h)

    y_lo, y_hi = lock_axis_range(*PANEL_F_YLIM, must_include=[np.array(allv)],
                                 axis_name='panel f y-axis')

    if show_shen20:
        ax.plot(l_ref, shen20_logphi, color='#000000', lw=1.6, ls='--',
                zorder=4, label='Shen+20')

    ax.set_xlim(lo, hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel(r'$\log\,L_{\rm bol}\ [{\rm erg\,s^{-1}}]$', fontsize=18)
    ylabel = (r'$\log\,({\rm d}N/{\rm d}\log L_{\rm bol}\ /\ {\rm Mpc^{-3}}\,h^3)$'
              if volume_h3 else r'$\log\,({\rm d}N/{\rm d}\log L_{\rm bol})$')
    ax.set_ylabel(ylabel, fontsize=16)

    ax.text(0.97, 0.04, rf'$z = {redshift:.1f}$',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=16)
    ax.legend(loc='upper right', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_file, dpi=140, bbox_inches='tight')
    plt.close()
    print(f'✓  Saved  →  {output_file}')

# ============================================================================
# COMPARE MODE (multi-run overlay, contours only for the scatter+KDE panels)
# ============================================================================
# Single-run mode above (plot_panel_a..f) is completely unaffected by
# everything below. In compare mode, panels a-e drop the raw background
# scatter cloud and the LRD red/blue selection dots -- with two or more
# full point clouds on one axis those become an unreadable smear -- and
# show only each run's KDE density contour, colored per run (see
# run_style.contour_style_for_index). Panel f (a luminosity function, not
# a scatter plot) keeps its category split and is instead distinguished by
# linestyle per run (run_style.style_for_index), the same convention used
# throughout allresults-blackholes.py's compare functions.
#
# `runs` in every function below is a list of {'data', 'style'} dicts:
# 'data' is one read_epoch() dict for that run, 'style' from
# contour_style_for_index() (panels a-e) or style_for_index() (panel f).

def plot_panel_a_compare(runs, snap_col, output_file, bhar_floor=LRD_BHAR_DEFAULT,
                         z_override=None, show_lit=True, mask_seeds=True):
    name = "panel a (compare)"
    redshift = z_override if z_override is not None else snap_to_z(snap_col)

    prepped = []
    for run in runs:
        data, style = run['data'], run['style']
        bh_mass, mdot, seed = data['bh_mass'], data['mdot_msun_yr'], data['seed_mass']
        if len(bh_mass) == 0:
            print(f"  [skip] {style['label']}: no accretion events.")
            continue
        valid = (bh_mass > 0) & (mdot > 0) & np.isfinite(bh_mass) & np.isfinite(mdot)
        if mask_seeds:
            valid &= mask_ungrown_seeds(bh_mass, seed)
        if valid.sum() == 0:
            print(f"  [skip] {style['label']}: no valid events after masking.")
            continue
        prepped.append((np.log10(bh_mass[valid]), np.log10(mdot[valid]), style))

    if not prepped:
        print(f"[skip] {name}: no data for any run.")
        return

    pang_m = lit_z_mask(PANG26['z'], redshift)
    mathee_m = lit_z_mask(MATHEE24['z'], redshift)
    lin_m = lit_z_mask(LIN25['z'], redshift)
    lit_log_mbh, lit_log_mdot = [], []
    if show_lit:
        t_mdot = PANG26['lambda_edd'][pang_m] * 10**eddington_mdot(PANG26['log_mbh'][pang_m])
        lit_log_mbh.append(PANG26['log_mbh'][pang_m]); lit_log_mdot.append(np.log10(t_mdot))
        m_mdot = mdot_from_lbol(np.log10(MATHEE24['lbol_1e44'][mathee_m]) + 44.0)
        lit_log_mbh.append(MATHEE24['log_mbh'][mathee_m]); lit_log_mdot.append(np.log10(m_mdot))
        l_mdot = mdot_from_lbol(LIN25['log_lbol'][lin_m])
        lit_log_mbh.append(LIN25['log_mbh'][lin_m]); lit_log_mdot.append(np.log10(l_mdot))

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.minorticks_on()

    x_lo, x_hi = lock_axis_range(*PANEL_A_XLIM, must_include=lit_log_mbh,
                                 axis_name='panel a x-axis (compare)')
    y_lo, y_hi = lock_axis_range(*PANEL_A_YLIM, must_include=lit_log_mdot,
                                 axis_name='panel a y-axis (compare)')

    x_ref = np.linspace(x_lo, x_hi, 400)
    y_edd = eddington_mdot(x_ref)
    ax.plot(x_ref, y_edd, color='#C62828', lw=1.8, ls='-', zorder=4)
    ax.plot(x_ref, y_edd + 1.0, color='#E65100', lw=1.8, ls='-', zorder=4)
    ax.axhline(np.log10(LRD_BHAR_DEFAULT), color='#C62828', lw=1.3, ls='--', zorder=3, alpha=0.85)
    ax.axhline(np.log10(LRD_BHAR_ALT), color='#C62828', lw=1.0, ls=':', zorder=3, alpha=0.70)
    ax.annotate(
        rf'$\dot{{M}}_{{\rm BH}} = {LRD_BHAR_DEFAULT}\,M_\odot\,\mathrm{{yr}}^{{-1}}$',
        xy=(x_hi - 0.2, np.log10(LRD_BHAR_DEFAULT) + 0.15),
        fontsize=12, color='#C62828', ha='right')

    run_handles = draw_contours_multirun(ax, prepped, x_lo, x_hi, y_lo, y_hi)

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
    ax.set_xlabel(r'$M_{\rm BH}\ [M_\odot]$', fontsize=18)
    ax.set_ylabel(r'$\dot{M}_{\rm BH}\ [M_\odot\,\mathrm{yr}^{-1}]$', fontsize=18)
    ax.set_xticks(np.arange(int(np.ceil(x_lo)), int(np.floor(x_hi)) + 1, 2))
    ax.text(0.97, 0.04, rf'$z = {redshift:.1f}$', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=16)

    handles = [
        Line2D([0], [0], color='#C62828', lw=1.8, label=r'$\dot{M}_{\rm BH} = \dot{M}_{\rm Edd}$'),
        Line2D([0], [0], color='#E65100', lw=1.8, label=r'$\dot{M}_{\rm BH} = 10\,\dot{M}_{\rm Edd}$'),
    ] + run_handles + lit_legend_handles(lit_labels)
    ax.legend(handles=handles, loc='upper left', fontsize=12, handlelength=1.6, handletextpad=0.5)

    plt.tight_layout()
    plt.savefig(output_file, dpi=140, bbox_inches='tight')
    plt.close()
    print(f'✓  Saved  →  {output_file}')


def plot_panel_b_compare(runs, snap_col, output_file, bhar_floor=LRD_BHAR_DEFAULT,
                         z_override=None, show_lit=True, mask_seeds=True):
    name = "panel b (compare)"
    redshift = z_override if z_override is not None else snap_to_z(snap_col)

    prepped = []
    for run in runs:
        data, style = run['data'], run['style']
        bh_mass, mdot, star, seed = (data['bh_mass'], data['mdot_msun_yr'],
                                     data['stellar_mass'], data['seed_mass'])
        if len(bh_mass) == 0:
            print(f"  [skip] {style['label']}: no accretion events.")
            continue
        valid = ((bh_mass > 0) & (mdot > 0) & (star > 0) &
                np.isfinite(bh_mass) & np.isfinite(mdot) & np.isfinite(star))
        if mask_seeds:
            valid &= mask_ungrown_seeds(bh_mass, seed)
        if valid.sum() == 0:
            print(f"  [skip] {style['label']}: no valid events after masking.")
            continue
        f_bh = bh_mass[valid] / star[valid]
        prepped.append((np.log10(bh_mass[valid]), np.log10(f_bh), style))

    if not prepped:
        print(f"[skip] {name}: no data for any run.")
        return

    pang_m = lit_z_mask(PANG26['z'], redshift)
    lit_log_fbh = (PANG26['log_mbh'][pang_m] - PANG26['log_mstar'][pang_m]) if show_lit else []
    lit_log_mbh = PANG26['log_mbh'][pang_m] if show_lit else []

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.minorticks_on()

    x_lo, x_hi = lock_axis_range(*PANEL_B_XLIM, must_include=[lit_log_mbh],
                                 axis_name='panel b x-axis (compare)')
    y_lo, y_hi = lock_axis_range(*PANEL_B_YLIM, must_include=[lit_log_fbh],
                                 axis_name='panel b y-axis (compare)')

    log_fbh_thresh = np.log10(LRD_FBHM_THRESH)
    ax.axhline(np.log10(0.1), color='#F57C00', lw=1.8, zorder=4)
    ax.axhline(log_fbh_thresh, color='#C62828', lw=1.8, zorder=4)
    ax.annotate(
        r'$M_{\rm BH}/M_\star = 0.1$',
        xy=(x_hi - 4.3, np.log10(0.1)), xytext=(x_hi - 3.6, np.log10(0.1) + 0.55),
        fontsize=13, color='#F57C00', ha='left',
        arrowprops=dict(arrowstyle='->', color='#F57C00', lw=1.2))
    ax.text(x_hi - 0.2, log_fbh_thresh + 0.10, r'$f_{\rm BH} = 3\%$',
            fontsize=13, color='#C62828', ha='right')

    run_handles = draw_contours_multirun(ax, prepped, x_lo, x_hi, y_lo, y_hi)

    lit_labels = []
    if show_lit and pang_m.any():
        t_log_fbh = PANG26['log_mbh'][pang_m] - PANG26['log_mstar'][pang_m]
        t_fbh_err = np.sqrt(PANG26['log_mbh_err'][pang_m]**2 + PANG26['log_mstar_err'][pang_m]**2)
        plot_lit_points(ax, 'Pang+26', PANG26['log_mbh'][pang_m], t_log_fbh,
                        xerr=PANG26['log_mbh_err'][pang_m], yerr=t_fbh_err)
        lit_labels.append('Pang+26')

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel(r'$M_{\rm BH}\ [M_\odot]$', fontsize=18)
    ax.set_ylabel(r'$f_{\rm BH} = M_{\rm BH}/M_\star$', fontsize=18)
    ax.set_xticks(np.arange(int(np.ceil(x_lo)), int(np.floor(x_hi)) + 1, 2))
    ax.text(0.97, 0.04, rf'$z = {redshift:.1f}$', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=16)

    handles = run_handles + lit_legend_handles(lit_labels)
    if handles:
        ax.legend(handles=handles, loc='upper left', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_file, dpi=140, bbox_inches='tight')
    plt.close()
    print(f'✓  Saved  →  {output_file}')


def plot_panel_c_compare(runs, snap_col, output_file, bhar_floor=LRD_BHAR_DEFAULT,
                         z_override=None, show_lit=True, mask_seeds=True):
    name = "panel c (compare)"
    redshift = z_override if z_override is not None else snap_to_z(snap_col)

    prepped = []
    for run in runs:
        data, style = run['data'], run['style']
        bh_mass, mdot, star, seed = (data['bh_mass'], data['mdot_msun_yr'],
                                     data['stellar_mass'], data['seed_mass'])
        if len(bh_mass) == 0:
            print(f"  [skip] {style['label']}: no accretion events.")
            continue
        valid = ((bh_mass > 0) & (mdot > 0) & (star > 0) &
                np.isfinite(bh_mass) & np.isfinite(mdot) & np.isfinite(star))
        if mask_seeds:
            valid &= mask_ungrown_seeds(bh_mass, seed)
        if valid.sum() == 0:
            print(f"  [skip] {style['label']}: no valid events after masking.")
            continue
        prepped.append((np.log10(star[valid]), np.log10(bh_mass[valid]), style))

    if not prepped:
        print(f"[skip] {name}: no data for any run.")
        return

    pang_m = lit_z_mask(PANG26['z'], redshift)
    furtak_m = lit_z_mask(FURTAK23['z'], redshift)
    lit_mstar, lit_mbh = [], []
    if show_lit:
        lit_mstar.append(PANG26['log_mstar'][pang_m])
        lit_mbh.append(PANG26['log_mbh'][pang_m])
        if furtak_m:
            lit_mstar.append([FURTAK23['log_mstar_upper_limit']])
            lit_mbh.append([FURTAK23['log_mbh']])

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.minorticks_on()

    x_lo, x_hi = lock_axis_range(*PANEL_C_XLIM, must_include=lit_mstar,
                                 axis_name='panel c x-axis (compare)')
    y_lo, y_hi = lock_axis_range(*PANEL_C_YLIM, must_include=lit_mbh,
                                 axis_name='panel c y-axis (compare)')

    x_ref = np.linspace(x_lo, x_hi, 400)
    y_kh = kormendy_ho_mbh(x_ref)
    ax.fill_between(x_ref, y_kh - KORMENDY_HO_SCATTER, y_kh + KORMENDY_HO_SCATTER,
                    color='#999999', alpha=0.35, zorder=0)
    kh_line, = ax.plot(x_ref, y_kh, color='black', lw=1.8, zorder=2, label='Kormendy & Ho 2013')

    ratio_rot = _line_rotation_deg(1.0, x_lo, x_hi, y_lo, y_hi)
    for ratio in MASS_RATIO_LINES:
        y_ratio = x_ref + np.log10(ratio)
        ax.plot(x_ref, y_ratio, color='black', lw=1.0, ls='--', zorder=2)
        x_lab = x_lo + 0.12 * (x_hi - x_lo)
        y_lab = x_lab + np.log10(ratio)
        if y_lo < y_lab < y_hi:
            ax.text(x_lab, y_lab - 0.12, rf'$M_{{\rm BH}}/M_\star = {ratio:g}$',
                    fontsize=10.5, color='black', ha='left', va='top',
                    rotation=ratio_rot, rotation_mode='anchor')

    run_handles = draw_contours_multirun(ax, prepped, x_lo, x_hi, y_lo, y_hi)

    lit_labels = []
    if show_lit:
        if pang_m.any():
            plot_lit_points(ax, 'Pang+26', PANG26['log_mstar'][pang_m], PANG26['log_mbh'][pang_m],
                            xerr=PANG26['log_mstar_err'][pang_m], yerr=PANG26['log_mbh_err'][pang_m])
            lit_labels.append('Pang+26')
        if furtak_m:
            plot_lit_points(ax, 'Furtak+23', [FURTAK23['log_mstar_upper_limit']],
                            [FURTAK23['log_mbh']],
                            xerr=[[0.4], [0.0]],
                            yerr=[[FURTAK23['log_mbh_err_lo']], [FURTAK23['log_mbh_err_hi']]],
                            xuplims=True)
            lit_labels.append('Furtak+23')

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel(r'$\log\,M_\star\ [M_\odot]$', fontsize=18)
    ax.set_ylabel(r'$\log\,M_{\rm BH}\ [M_\odot]$', fontsize=18)
    ax.text(0.97, 0.04, rf'$z = {redshift:.1f}$', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=16)

    handles = [kh_line] + run_handles + lit_legend_handles(lit_labels)
    ax.legend(handles=handles, loc='upper left', fontsize=12, handlelength=1.6, handletextpad=0.5)

    plt.tight_layout()
    plt.savefig(output_file, dpi=140, bbox_inches='tight')
    plt.close()
    print(f'✓  Saved  →  {output_file}')


def plot_panel_d_compare(runs, snap_col, output_file, bhar_floor=LRD_BHAR_DEFAULT,
                         z_override=None, show_lit=True, mask_seeds=True):
    name = "panel d (compare)"
    redshift = z_override if z_override is not None else snap_to_z(snap_col)

    prepped = []
    for run in runs:
        data, style = run['data'], run['style']
        bh_mass, mdot, star, seed = (data['bh_mass'], data['mdot_msun_yr'],
                                     data['stellar_mass'], data['seed_mass'])
        if len(bh_mass) == 0:
            print(f"  [skip] {style['label']}: no accretion events.")
            continue
        valid = ((bh_mass > 0) & (mdot > 0) & (star > 0) &
                np.isfinite(bh_mass) & np.isfinite(mdot) & np.isfinite(star))
        if mask_seeds:
            valid &= mask_ungrown_seeds(bh_mass, seed)
        if valid.sum() == 0:
            print(f"  [skip] {style['label']}: no valid events after masking.")
            continue
        prepped.append((np.log10(bh_mass[valid]), lbol_from_mdot(mdot[valid]), style))

    if not prepped:
        print(f"[skip] {name}: no data for any run.")
        return

    pang_m = lit_z_mask(PANG26['z'], redshift)
    mathee_m = lit_z_mask(MATHEE24['z'], redshift)
    lin_m = lit_z_mask(LIN25['z'], redshift)
    furtak_m = lit_z_mask(FURTAK23['z'], redshift)
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

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.minorticks_on()

    x_lo, x_hi = lock_axis_range(*PANEL_D_XLIM, must_include=lit_log_mbh,
                                 axis_name='panel d x-axis (compare)')
    y_lo, y_hi = lock_axis_range(*PANEL_D_YLIM, must_include=lit_log_lbol,
                                 axis_name='panel d y-axis (compare)')

    x_ref = np.linspace(x_lo, x_hi, 400)
    line_rot = _line_rotation_deg(1.0, x_lo, x_hi, y_lo, y_hi)
    for lam in EDDINGTON_RATIO_LINES:
        y_line = eddington_luminosity(x_ref) + np.log10(lam)
        ax.plot(x_ref, y_line, color='#777777', lw=1.2, ls='--', zorder=2)
        x_lab = x_lo + 0.60 * (x_hi - x_lo)
        y_lab = eddington_luminosity(x_lab) + np.log10(lam)
        if y_lo < y_lab < y_hi:
            ax.text(x_lab, y_lab - 0.12, rf'$\lambda_{{\rm Edd}} = {lam:g}$',
                    fontsize=10.5, color='#555555', ha='left', va='top',
                    rotation=line_rot, rotation_mode='anchor')

    run_handles = draw_contours_multirun(ax, prepped, x_lo, x_hi, y_lo, y_hi)

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
    ax.set_xlabel(r'$\log\,M_{\rm BH}\ [M_\odot]$', fontsize=18)
    ax.set_ylabel(r'$\log\,L_{\rm bol}\ [{\rm erg\,s^{-1}}]$', fontsize=18)
    ax.text(0.97, 0.04, rf'$z = {redshift:.1f}$', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=16)

    handles = run_handles + lit_legend_handles(lit_labels)
    if handles:
        ax.legend(handles=handles, loc='upper left', fontsize=12, handlelength=1.6, handletextpad=0.5)

    plt.tight_layout()
    plt.savefig(output_file, dpi=140, bbox_inches='tight')
    plt.close()
    print(f'✓  Saved  →  {output_file}')


def plot_panel_e_compare(runs, snap_col, output_file, bhar_floor=LRD_BHAR_DEFAULT,
                         z_override=None, show_lit=True, mask_seeds=True):
    name = "panel e (compare)"
    redshift = z_override if z_override is not None else snap_to_z(snap_col)

    prepped = []
    for run in runs:
        data, style = run['data'], run['style']
        bh_mass, mdot, star, seed = (data['bh_mass'], data['mdot_msun_yr'],
                                     data['stellar_mass'], data['seed_mass'])
        if len(bh_mass) == 0:
            print(f"  [skip] {style['label']}: no accretion events.")
            continue
        valid = ((bh_mass > 0) & (mdot > 0) & (star > 0) &
                np.isfinite(bh_mass) & np.isfinite(mdot) & np.isfinite(star))
        if mask_seeds:
            valid &= mask_ungrown_seeds(bh_mass, seed)
        if valid.sum() == 0:
            print(f"  [skip] {style['label']}: no valid events after masking.")
            continue
        m1450 = m1450_from_lbol(lbol_from_mdot(mdot[valid]))
        prepped.append((m1450, np.log10(bh_mass[valid]), style))

    if not prepped:
        print(f"[skip] {name}: no data for any run.")
        return

    mathee_m = lit_z_mask(MATHEE24['z'], redshift)
    labbe_m = lit_z_mask(LABBE25['z'], redshift)
    furtak_m = lit_z_mask(FURTAK23['z'], redshift)
    lit_muv, lit_mbh = [], []
    labbe_mbh_min = None
    if show_lit:
        labbe_mbh_min = bh_mass_min_from_lbol(LABBE25['log_lbol'][labbe_m])
        lit_muv += [LABBE25['m1450'][labbe_m], MATHEE24['muv'][mathee_m]]
        lit_mbh += [np.log10(labbe_mbh_min), MATHEE24['log_mbh'][mathee_m]]
        if furtak_m:
            lit_muv.append([FURTAK23['muv']])
            lit_mbh.append([FURTAK23['log_mbh']])

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.minorticks_on()

    x_faint, x_bright = lock_axis_range(*PANEL_E_XLIM, must_include=lit_muv,
                                        axis_name='panel e x-axis (compare)')
    y_lo, y_hi = lock_axis_range(*PANEL_E_YLIM, must_include=lit_mbh,
                                 axis_name='panel e y-axis (compare)')

    run_handles = draw_contours_multirun(ax, prepped, x_faint, x_bright, y_lo, y_hi)

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
    ax.set_xlabel(r'$M_{\rm UV,1450}$', fontsize=18)
    ax.set_ylabel(r'$\log\,M_{\rm BH}\ [M_\odot]$', fontsize=18)
    ax.text(0.97, 0.04, rf'$z = {redshift:.1f}$', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=16)

    handles = run_handles + lit_legend_handles(lit_labels)
    if handles:
        ax.legend(handles=handles, loc='upper left', fontsize=12, handlelength=1.6, handletextpad=0.5)

    plt.tight_layout()
    plt.savefig(output_file, dpi=140, bbox_inches='tight')
    plt.close()
    print(f'✓  Saved  →  {output_file}')


def plot_panel_f_compare(runs, snap_col, output_file, volume_h3, bhar_floor=LRD_BHAR_DEFAULT,
                         z_override=None, n_bins=40, h_h=None, show_lit=True, mask_seeds=True):
    """
    Panel f is a luminosity function (histogram/errorbar), not a scatter
    plot -- unlike panels a-e it keeps its Total/LRD-red/LRD-blue category
    split in compare mode, distinguished by linestyle per run (like
    allresults-blackholes.py's compare functions), not by contour color.
    """
    name = "panel f (compare)"
    redshift = z_override if z_override is not None else snap_to_z(snap_col)

    per_run = []
    for run in runs:
        data, style = run['data'], run['style']
        bh_mass, mdot, medd, star, seed = (data['bh_mass'], data['mdot_msun_yr'],
                                           data['mdot_edd'], data['stellar_mass'],
                                           data['seed_mass'])
        if len(bh_mass) == 0:
            print(f"  [skip] {style['label']}: no accretion events.")
            per_run.append((None, style)); continue
        valid = ((bh_mass > 0) & (mdot > 0) & (star > 0) &
                np.isfinite(bh_mass) & np.isfinite(mdot) & np.isfinite(star))
        if mask_seeds:
            valid &= mask_ungrown_seeds(bh_mass, seed)
        if valid.sum() == 0:
            print(f"  [skip] {style['label']}: no valid events after masking.")
            per_run.append((None, style)); continue
        bh_mass, mdot, medd, star = bh_mass[valid], mdot[valid], medd[valid], star[valid]
        log_lbol = lbol_from_mdot(mdot)
        _, lrd_red, lrd_blue = compute_selection(bh_mass, mdot, medd, star, bhar_floor)
        per_run.append(({'log_lbol': log_lbol, 'lrd_red': lrd_red, 'lrd_blue': lrd_blue}, style))

    if not any(d is not None for d, _ in per_run):
        print(f"[skip] {name}: no data for any run.")
        return

    all_selected = [d['log_lbol'][d['lrd_red'] | d['lrd_blue']] for d, _ in per_run if d is not None]
    lo, hi = lock_axis_range(*PANEL_F_XLIM, must_include=all_selected, axis_name='panel f x-axis (compare)')
    bins = np.linspace(lo, hi, n_bins + 1)
    bw = bins[1] - bins[0]
    centres = 0.5 * (bins[:-1] + bins[1:])

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.minorticks_on()

    allv = []
    run_handles = []
    for i, (d, style) in enumerate(per_run):
        if d is None:
            continue
        cats = [(d['log_lbol'], 'Total', 'k', 'o'),
                (d['log_lbol'][d['lrd_red']],  r'LRD ($f_{\rm BH}\geq 3\%$)', '#C62828', 'D'),
                (d['log_lbol'][d['lrd_blue']], r'LRD ($f_{\rm BH}<3\%$)',     '#F57C00', 's')]
        for values, label, color, marker in cats:
            counts, _ = np.histogram(values, bins=bins)
            pos = counts > 0
            if not np.any(pos):
                continue
            y = counts / (bw * volume_h3) if volume_h3 else counts / bw
            logy = np.log10(y[pos])
            logy_err = 1.0 / (np.sqrt(counts[pos]) * np.log(10))
            draw_color = lighten_color(color, style['lighten'])
            ax.errorbar(centres[pos], logy, yerr=logy_err, fmt=marker, color=draw_color,
                        mec='black', mew=0.5, ms=6, capsize=2.5, elinewidth=1.0, ls=style['linestyle'],
                        label=(label if i == 0 else None), zorder=5)
            allv.extend(logy)
        run_handles.append(Line2D([0], [0], color='black', lw=1.8, ls=style['linestyle'], label=style['label']))

    show_shen20 = show_lit and (1.0 <= redshift <= 7.0)
    if show_shen20:
        l_ref = np.linspace(max(lo, SHEN20_MODEL_LOGLBOL_MIN), hi, 200)
        shen20_logphi = shen20_bolometric_qlf_logphi(l_ref, redshift, h_h=h_h)

    y_lo, y_hi = lock_axis_range(*PANEL_F_YLIM, must_include=[np.array(allv)],
                                 axis_name='panel f y-axis (compare)')

    if show_shen20:
        ax.plot(l_ref, shen20_logphi, color='#000000', lw=1.6, ls='--', zorder=4, label='Shen+20')

    ax.set_xlim(lo, hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel(r'$\log\,L_{\rm bol}\ [{\rm erg\,s^{-1}}]$', fontsize=18)
    ylabel = (r'$\log\,({\rm d}N/{\rm d}\log L_{\rm bol}\ /\ {\rm Mpc^{-3}}\,h^3)$'
             if volume_h3 else r'$\log\,({\rm d}N/{\rm d}\log L_{\rm bol})$')
    ax.set_ylabel(ylabel, fontsize=16)
    ax.text(0.97, 0.04, rf'$z = {redshift:.1f}$', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=16)

    cat_handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=cat_handles + run_handles, loc='upper right', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_file, dpi=140, bbox_inches='tight')
    plt.close()
    print(f'✓  Saved  →  {output_file}')


# ============================================================================
# CLI
# ============================================================================

def main():
    p = argparse.ArgumentParser(
        description='Recreate panels (a) and (b) of Chen & Mo 2026 (arXiv:2605.31077), '
                    'plus a panel (c) styled after Fig. 7 of Kocevski et al. (2023).'
    )
    p.add_argument('-i', '--input-pattern',
                   default='./output/millennium/model_*.hdf5')
    p.add_argument('-s', '--snapshot', type=int, default=27,
                   help='History COLUMN (= Millennium snapshot) to slice. '
                        'Default 27 -> z~3.')
    p.add_argument('--window', type=int, default=0,
                   help='Stack columns [s-window, s+window] to fight sparsity '
                        'at high z (default 0 = single column).')
    p.add_argument('--catalogue', default=None,
                   help='Force a specific Snap_N catalogue group to read '
                        'histories from (default: most complete, auto-selected).')
    p.add_argument('--no-lrd', action='store_true',
                   help='Skip LRD selection overlay.')
    p.add_argument('--no-fbh', action='store_true',
                   help='Disable the f_BH red/blue split (all selected = red). '
                        'Use if mixing the per-epoch M_BH with catalogue-level '
                        'M_star is a concern.')
    p.add_argument('--no-lit', action='store_true',
                   help='Skip the literature overlay (Pang+26, Mathee+24, '
                        'Labbe+25, Furtak+23, Lin+25) on panels a-e.')
    p.add_argument('--no-mask-seeds', action='store_true',
                   help='Do not mask BH accretion events still essentially at '
                        f'their own BHSeedMass (< {SEED_GROWTH_THRESHOLD:.0%} grown); '
                        'by default these are excluded everywhere since they '
                        'pile up as a spurious cluster at log M_BH ~ 2.')
    p.add_argument('--bhar-floor', type=float, default=LRD_BHAR_DEFAULT,
                   help=f'BHAR floor in M_sun/yr (default {LRD_BHAR_DEFAULT}; '
                        f'paper alternative {LRD_BHAR_ALT}).')
    p.add_argument('--output', default=None,
                   help='Output path for panel a (Mdot_BH vs M_BH).')
    p.add_argument('--output-a-acctype', default=None,
                   help='Output path for the accretion-type-coloured version '
                        'of panel a (background BHs split into merger-driven '
                        '/ disk-instability-driven / radio-mode-or-unknown).')
    p.add_argument('--output-b', default=None,
                   help='Output path for panel b (f_BH vs M_BH).')
    p.add_argument('--output-c', default=None,
                   help='Output path for panel c (M_BH vs M_star).')
    p.add_argument('--output-d', default=None,
                   help='Output path for panel d (L_bol vs M_BH).')
    p.add_argument('--output-e', default=None,
                   help='Output path for panel e (M_1450 vs M_BH).')
    p.add_argument('--output-f', default=None,
                   help='Output path for panel f (bolometric luminosity function).')
    p.add_argument('--no-panel-a', action='store_true',
                   help='Skip panel a (Mdot_BH vs M_BH).')
    p.add_argument('--no-panel-b', action='store_true',
                   help='Skip panel b (f_BH vs M_BH).')
    p.add_argument('--no-panel-c', action='store_true',
                   help='Skip panel c (M_BH vs M_star).')
    p.add_argument('--no-panel-d', action='store_true',
                   help='Skip panel d (L_bol vs M_BH).')
    p.add_argument('--no-panel-e', action='store_true',
                   help='Skip panel e (M_1450 vs M_BH).')
    p.add_argument('--no-panel-f', action='store_true',
                   help='Skip panel f (bolometric luminosity function).')
    p.add_argument('--sim-volume', type=float, default=None,
                   help='Override comoving volume in (Mpc/h)^3 for panel f '
                        '(default: auto from box_size * frac_volume_processed).')
    p.add_argument('--lf-bins', type=int, default=40,
                   help='Number of log10(L_bol) bins for panel f (default 40).')
    p.add_argument('--z', type=float, default=None,
                   help='Override redshift label on the plot.')
    args = p.parse_args()

    files = sorted(glob.glob(args.input_pattern))
    if not files:
        print(f'ERROR: no files matched "{args.input_pattern}"'); sys.exit(1)

    h_h  = read_sim_params(files[0])
    redshifts = read_actual_redshifts(files[0])
    z    = args.z if args.z is not None else snap_to_z(args.snapshot, redshifts)

    print(f'Files:       {len(files)}')
    print(f'History col: {args.snapshot}  ->  z ~ {z:.3f}'
          + (f'  (+/- {args.window})' if args.window else ''))
    print(f'Hubble_h:    {h_h}')
    print(f'BHAR floor:  {args.bhar_floor} M_sun/yr')
    print(f'Mask seeds:  {not args.no_mask_seeds}')
    print('Reading data...')

    data = read_epoch(files, args.snapshot, h_h,
                      catalogue=args.catalogue, window=args.window)
    print(f'Catalogue read: {data["cat_group"]}')

    d = Path(files[0]).parent / 'plots'
    d.mkdir(exist_ok=True)

    if not args.no_panel_a:
        out_a = Path(args.output) if args.output else \
            d / f'lrd_bh_accretion_scatter_snap{args.snapshot:02d}.png'
        print('Plotting panel a...')
        plot_panel_a(data, args.snapshot, out_a,
                     show_lrd=(not args.no_lrd),
                     use_fbh=(not args.no_fbh),
                     bhar_floor=args.bhar_floor,
                     z_override=z,
                     show_lit=(not args.no_lit),
                     mask_seeds=(not args.no_mask_seeds))

        out_a_acctype = Path(args.output_a_acctype) if args.output_a_acctype else \
            d / f'lrd_bh_accretion_scatter_by_acctype_snap{args.snapshot:02d}.png'
        print('Plotting panel a (coloured by accretion type)...')
        plot_panel_a(data, args.snapshot, out_a_acctype,
                     show_lrd=(not args.no_lrd),
                     use_fbh=(not args.no_fbh),
                     bhar_floor=args.bhar_floor,
                     z_override=z,
                     show_lit=(not args.no_lit),
                     mask_seeds=(not args.no_mask_seeds),
                     color_by_acctype=True)

    if not args.no_panel_b:
        out_b = Path(args.output_b) if args.output_b else \
            d / f'lrd_fbh_scatter_snap{args.snapshot:02d}.png'
        print('Plotting panel b...')
        plot_panel_b(data, args.snapshot, out_b,
                     show_lrd=(not args.no_lrd),
                     bhar_floor=args.bhar_floor,
                     z_override=z,
                     show_lit=(not args.no_lit),
                     mask_seeds=(not args.no_mask_seeds))

    if not args.no_panel_c:
        out_c = Path(args.output_c) if args.output_c else \
            d / f'lrd_mbh_mstar_scatter_snap{args.snapshot:02d}.png'
        print('Plotting panel c...')
        plot_panel_c(data, args.snapshot, out_c,
                     show_lrd=(not args.no_lrd),
                     bhar_floor=args.bhar_floor,
                     z_override=z,
                     show_lit=(not args.no_lit),
                     mask_seeds=(not args.no_mask_seeds))

    if not args.no_panel_d:
        out_d = Path(args.output_d) if args.output_d else \
            d / f'lrd_lbol_mbh_scatter_snap{args.snapshot:02d}.png'
        print('Plotting panel d...')
        plot_panel_d(data, args.snapshot, out_d,
                     show_lrd=(not args.no_lrd),
                     bhar_floor=args.bhar_floor,
                     z_override=z,
                     show_lit=(not args.no_lit),
                     mask_seeds=(not args.no_mask_seeds))

    if not args.no_panel_e:
        out_e = Path(args.output_e) if args.output_e else \
            d / f'lrd_m1450_mbh_scatter_snap{args.snapshot:02d}.png'
        print('Plotting panel e...')
        plot_panel_e(data, args.snapshot, out_e,
                     show_lrd=(not args.no_lrd),
                     bhar_floor=args.bhar_floor,
                     z_override=z,
                     show_lit=(not args.no_lit),
                     mask_seeds=(not args.no_mask_seeds))

    if not args.no_panel_f:
        volume_h3 = args.sim_volume if args.sim_volume is not None \
            else read_box_volume_h3(files)
        print(f'Volume (panel f): {volume_h3:.4e} (Mpc/h)^3')
        out_f = Path(args.output_f) if args.output_f else \
            d / f'lrd_bolometric_luminosity_function_snap{args.snapshot:02d}.png'
        print('Plotting panel f...')
        plot_panel_f(data, args.snapshot, out_f, volume_h3,
                     show_lrd=(not args.no_lrd),
                     bhar_floor=args.bhar_floor,
                     z_override=z,
                     n_bins=args.lf_bins,
                     h_h=h_h,
                     show_lit=(not args.no_lit),
                     mask_seeds=(not args.no_mask_seeds))


if __name__ == '__main__':
    main()
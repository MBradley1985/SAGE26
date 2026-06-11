#!/usr/bin/env python
"""
Clustering bias of quiescent galaxies vs Tinker+10 halo bias.
============================================================

For each model:
  1. Select all quiescent galaxies in model 1 at a user-supplied redshift
     (sSFR < 0.2 / t_H), then take the top N by stellar mass.
  2. Match by GalaxyIndex into model 2 at the same snap.
  3. Compute Tinker et al. (2010) large-scale halo bias b(M_vir, z) for each
     matched galaxy from M_vir and the simulation cosmology.  sigma(M) is
     evaluated from the Eisenstein & Hu (1998) no-wiggle power spectrum,
     normalised to sigma_8, with the linear growth factor of Carroll+92.
  4. Trace the matched sample through model 1's snapshot history and compute
     the per-snap bias of every progenitor that still exists (in BOTH models).

Three panels:
    (a) b vs log10(M_vir) at z_target -- scatter for both models with the
        Tinker+10 curve at z_target overlaid.
    (b) <b>(z) for the matched-progenitor sample, model 1 vs model 2, with
        the Tinker+10 prediction evaluated at the median M_vir at each z.
    (c) Distribution of b at z_target for the two models.

Usage:
    python plotting/quiescent_bias.py --redshift 2.0
    python plotting/quiescent_bias.py --redshift 1.0 --n-top 500 \\
        --sigma-8 0.9 --omega-b 0.045 --n-s 1.0
"""

import argparse
import os
import sys
import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

warnings.filterwarnings('ignore')

# Reuse I/O + cosmology + selection helpers from the companion script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quiescent_evolution_two_models import (  # noqa: E402
    _find_files, _read_header, _load_snap, _snap_nearest_z,
    _build_lookup, _ssfr_threshold, _count_quiescent,
    PROPS_TO_LOAD, SSFR_FACTOR, MODEL1_COLOR, MODEL2_COLOR,
)

_STYLE = './plotting/kieren_cohare_palatino_sty.mplstyle'
if os.path.exists(_STYLE):
    plt.style.use(_STYLE)
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor']   = 'white'
plt.rcParams['axes.edgecolor']   = 'black'


# ----- Defaults ---------------------------------------------------------------

DEFAULT_MODEL1       = './output/millennium/'
DEFAULT_MODEL2       = './output/millennium_oldCGMAGN/'
DEFAULT_MODEL1_LABEL = 'SAGE26 (millennium)'
DEFAULT_MODEL2_LABEL = 'SAGE26 (old CGM AGN)'
DEFAULT_REDSHIFT     = 0.0
DEFAULT_N_TOP        = 1000
DEFAULT_OUTPUT_DIR   = './output/quiescent_evolution/'
DEFAULT_FORMAT       = '.pdf'

# Millennium-1 reference cosmology; override via CLI when needed.
DEFAULT_SIGMA_8      = 0.9
DEFAULT_N_S          = 1.0
DEFAULT_OMEGA_B      = 0.045
DEFAULT_DELTA        = 200.0   # halo overdensity for Tinker+10 parameters

DELTA_C              = 1.686


# ----- Cosmology: transfer function + growth factor ---------------------------

def _eh98_no_wiggle_T(k_h, omega_m, omega_b, h, theta_cmb=2.728/2.7):
    """
    Eisenstein & Hu 1998 no-wiggle transfer function T(k).
    k_h is in h/Mpc; returns dimensionless T.
    """
    k = k_h * h                      # Mpc^-1
    omh2 = omega_m * h**2
    obh2 = omega_b * h**2
    f_b  = omega_b / omega_m

    s = 44.5 * np.log(9.83 / omh2) / np.sqrt(1.0 + 10.0 * obh2**0.75)
    alpha_g = (1.0
               - 0.328 * np.log(431.0 * omh2) * f_b
               + 0.38  * np.log(22.3  * omh2) * f_b**2)
    Gamma = omega_m * h * (alpha_g + (1.0 - alpha_g) / (1.0 + (0.43 * k * s)**4))
    q = k * theta_cmb**2 / Gamma
    L0 = np.log(2.0 * np.e + 1.8 * q)
    C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
    return L0 / (L0 + C0 * q**2)


def _growth_factor(z, omega_m, omega_l):
    """Linear growth factor D(z) normalised to D(0)=1 (Carroll+92 fit)."""
    a = 1.0 / (1.0 + z)
    om_z = omega_m / (omega_m + omega_l * a**3)
    ol_z = 1.0 - om_z

    def _g(om, ol):
        return 2.5 * om / (om**(4./7.) - ol + (1.0 + om/2.0) * (1.0 + ol/70.0))

    return (_g(om_z, ol_z) / _g(omega_m, omega_l)) * a


# ----- sigma(M) table at z=0 --------------------------------------------------

def _build_sigma_table(omega_m, omega_b, h, sigma_8, n_s,
                      logm_lo=8.0, logm_hi=16.0, n_pts=200):
    """
    Build a table of log10(M) -> sigma(M, z=0), normalised so that
    sigma(R = 8 Mpc/h, z=0) = sigma_8.

    M is in M_sun; sigma is dimensionless.
    """
    rho_crit_0 = 2.775e11 * h**2          # M_sun / Mpc^3 (physical)
    rho_m_0    = omega_m * rho_crit_0     # mean matter density today

    def _w_th(x):
        # Top-hat window in Fourier space; safe near x=0.
        out = np.where(x < 1e-3,
                       1.0 - x**2 / 10.0,
                       3.0 * (np.sin(x) - x * np.cos(x)) / np.maximum(x, 1e-30)**3)
        return out

    def _sigma2_R(R_h):
        """sigma^2 for top-hat radius R_h (Mpc/h), unnormalised shape."""
        def _integrand(k_h):
            T = _eh98_no_wiggle_T(k_h, omega_m, omega_b, h)
            P = (k_h ** n_s) * T**2
            return P * _w_th(k_h * R_h)**2 * k_h**2
        val, _ = quad(_integrand, 1e-5, 5.0e2, limit=400)
        return val / (2.0 * np.pi**2)

    # Normalisation: enforce sigma(8 Mpc/h) = sigma_8 at z=0.
    sigma2_8 = _sigma2_R(8.0)
    norm     = sigma_8**2 / sigma2_8

    logm   = np.linspace(logm_lo, logm_hi, n_pts)
    M      = 10.0 ** logm
    R_phys = (3.0 * M / (4.0 * np.pi * rho_m_0))**(1.0/3.0)   # Mpc
    R_h    = R_phys * h                                       # Mpc/h
    sig    = np.array([np.sqrt(norm * _sigma2_R(r)) for r in R_h])
    return {'logm': logm, 'sigma0': sig}


def _sigma_of_M(mvir, sigma_table):
    """Lookup sigma(M, z=0) from the precomputed table; mvir in M_sun."""
    logm = np.log10(np.clip(mvir, 1e3, 1e20))
    return np.interp(logm, sigma_table['logm'], sigma_table['sigma0'])


# ----- Tinker+10 bias ---------------------------------------------------------

def _tinker10_params(delta):
    """Closed-form Tinker+10 b(nu) parameters (their eqs. 6 + Table 2)."""
    y = np.log10(delta)
    A = 1.0 + 0.24 * y * np.exp(-(4.0/y)**4)
    a = 0.44 * y - 0.88
    B = 0.183
    b = 1.5
    C = 0.019 + 0.107 * y + 0.19 * np.exp(-(4.0/y)**4)
    c = 2.4
    return A, a, B, b, C, c


def _tinker10_bias(nu, delta=DEFAULT_DELTA):
    """Tinker+10 large-scale halo bias b(nu)."""
    A, a, B, b, C, c = _tinker10_params(delta)
    return 1.0 - A * nu**a / (nu**a + DELTA_C**a) + B * nu**b + C * nu**c


def _bias_for_galaxies(mvir, z, sigma_table, omega_m, omega_l, delta=DEFAULT_DELTA):
    """Tinker+10 b(M_vir, z) for arrays of mvir at a single redshift z."""
    sigma0 = _sigma_of_M(mvir, sigma_table)
    D_z    = _growth_factor(z, omega_m, omega_l)
    sigma_z = sigma0 * D_z
    nu = DELTA_C / np.maximum(sigma_z, 1e-30)
    return _tinker10_bias(nu, delta=delta)


# ----- Sample selection (shares logic with companion script) ------------------

def _select_top_quiescent(d_target, z_target, n_top, omega_m, omega_l, hubble_h):
    sm  = d_target['StellarMass']
    sfr = d_target['SfrDisk'] + d_target['SfrBulge']
    ssfr = np.where(sm > 0, sfr / np.maximum(sm, 1e-30), 0.0)
    thr  = _ssfr_threshold(z_target, omega_m, omega_l, hubble_h)
    quiescent = (sm > 0) & (ssfr < thr)
    qidx  = np.where(quiescent)[0]
    if qidx.size == 0:
        return np.array([], dtype=np.int64)
    order = np.argsort(-sm[qidx])
    return qidx[order[:n_top]]


def _load_all_snaps_upto(directory, props, snap_max):
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


# ----- Plotting ---------------------------------------------------------------

def _plot_bias_trajectory(mvir_curve, sigma_table, hdr, args,
                          label1, label2, z_samp,
                          logmv_med1, bz_med1,
                          logmv_med2, bz_med2,
                          tinker_z_lines, title, out_path):
    """
    Standalone version of the (M_vir, b) trajectory plot:
        x = log10(M_vir),  y = b
        - Tinker+10 curves at fixed z (colour = z)
        - sample medians per snap as markers (shape = model, colour = z)
    """
    fig, ax = plt.subplots(figsize=(9, 7))

    log_mvir_grid = np.log10(mvir_curve)
    z_norm = plt.Normalize(vmin=0.0, vmax=max(tinker_z_lines))
    cmap   = plt.get_cmap('viridis')

    for z_line in tinker_z_lines:
        b_curve = _bias_for_galaxies(
            mvir_curve, z_line, sigma_table,
            hdr['omega_m'], hdr['omega_l'], delta=args.delta)
        ax.plot(log_mvir_grid, b_curve,
                color=cmap(z_norm(z_line)), lw=2.0,
                label=rf'Tinker+10, $z={z_line:.1f}$')

    ok1 = np.isfinite(bz_med1) & np.isfinite(logmv_med1)
    ok2 = np.isfinite(bz_med2) & np.isfinite(logmv_med2)
    # Model 1: filled circles (s=120).  Model 2: larger hollow squares (s=260,
    # facecolor='none') so they remain visible even when the per-snap (Mvir, b)
    # medians of the two samples coincide.
    if ok1.any():
        ax.scatter(logmv_med1[ok1], bz_med1[ok1],
                   c=z_samp[ok1], cmap=cmap, norm=z_norm,
                   marker='o', s=120, edgecolors='black', linewidths=1.0,
                   zorder=5)
    if ok2.any():
        ax.scatter(logmv_med2[ok2], bz_med2[ok2],
                   facecolors='none',
                   edgecolors=cmap(z_norm(z_samp[ok2])),
                   marker='s', s=260, linewidths=2.0,
                   zorder=6)

    from matplotlib.lines import Line2D as _L2D
    marker_handles = [
        _L2D([0], [0], marker='o', color='lightgray', markeredgecolor='black',
             markersize=11, lw=0, label=label1),
        _L2D([0], [0], marker='s', markerfacecolor='none',
             markeredgecolor='black', markersize=14, lw=0, label=label2),
    ]
    leg_markers = ax.legend(handles=marker_handles, loc='upper left',
                            frameon=False, fontsize=11,
                            title='Sample median (per snap)')
    ax.add_artist(leg_markers)
    ax.legend(loc='lower right', frameon=False, fontsize=10, ncol=1)

    ax.set_xlabel(r'$\log_{10}(M_\mathrm{vir}\,/\,M_\odot)$')
    ax.set_ylabel(r'$b(M_\mathrm{vir},\,z)$')
    ax.set_xlim(log_mvir_grid.min(), log_mvir_grid.max())
    ax.set_ylim(0.0, 10.0)
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.set_title(title, fontsize=13)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


def _plot_bias(d1, d2, b1, b2, mvir_curve, b_curve_target,
               z_target, sigma_table, hdr, label1, label2,
               z_samp,
               logmv_med1, bz_med1,
               logmv_med2, bz_med2,
               tinker_z_lines,
               n_q_m1, n_q_m2, title_extra, out_path,
               args):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # (a) b vs log10(M_vir) at z_target
    ax = axes[0]
    log_mv1 = np.log10(np.maximum(d1['Mvir'], 1e3))
    log_mv2 = np.log10(np.maximum(d2['Mvir'], 1e3))
    ax.scatter(log_mv1, b1, s=10, color=MODEL1_COLOR, alpha=0.45,
               edgecolors='none', label=f'{label1} (N={len(b1)})')
    ax.scatter(log_mv2, b2, s=10, color=MODEL2_COLOR, alpha=0.45,
               edgecolors='none', label=f'{label2} (N={len(b2)})')
    ax.plot(np.log10(mvir_curve), b_curve_target,
            color='black', lw=2.2, ls='-',
            label=f'Tinker+10 ($z={z_target:.2f}$, $\\Delta={int(DEFAULT_DELTA)}$)')
    ax.set_xlabel(r'$\log_{10}(M_\mathrm{vir}\,/\,M_\odot)$')
    ax.set_ylabel(r'$b(M_\mathrm{vir},\,z)$')
    ax.set_yscale('log')
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.legend(loc='best', frameon=False, fontsize=9)
    ax.set_title('(a) bias vs $M_\\mathrm{vir}$ at $z_\\mathrm{target}$')

    # (b) Tinker+10 b(M_vir) at fixed redshifts, with model sample medians
    #     plotted as markers at (med log10 Mvir, med b) per snap, colored by z.
    ax = axes[1]
    log_mvir_grid = np.log10(mvir_curve)
    z_norm = plt.Normalize(vmin=0.0, vmax=max(tinker_z_lines))
    cmap = plt.get_cmap('viridis')
    for z_line in tinker_z_lines:
        b_curve = _bias_for_galaxies(
            mvir_curve, z_line, sigma_table,
            hdr['omega_m'], hdr['omega_l'], delta=args.delta)
        ax.plot(log_mvir_grid, b_curve,
                color=cmap(z_norm(z_line)), lw=1.6,
                label=rf'Tinker+10, $z={z_line:.1f}$')

    ok1 = np.isfinite(bz_med1) & np.isfinite(logmv_med1)
    ok2 = np.isfinite(bz_med2) & np.isfinite(logmv_med2)
    if ok1.any():
        ax.scatter(logmv_med1[ok1], bz_med1[ok1],
                   c=z_samp[ok1], cmap=cmap, norm=z_norm,
                   marker='o', s=85, edgecolors='black', linewidths=1.0,
                   zorder=5)
    if ok2.any():
        ax.scatter(logmv_med2[ok2], bz_med2[ok2],
                   c=z_samp[ok2], cmap=cmap, norm=z_norm,
                   marker='s', s=85, edgecolors='black', linewidths=1.0,
                   zorder=5)

    # Marker-only legend (so the user can distinguish the two models).
    from matplotlib.lines import Line2D as _L2D
    marker_handles = [
        _L2D([0], [0], marker='o', color='lightgray', markeredgecolor='black',
             markersize=9, lw=0, label=label1),
        _L2D([0], [0], marker='s', color='lightgray', markeredgecolor='black',
             markersize=9, lw=0, label=label2),
    ]
    leg_markers = ax.legend(handles=marker_handles, loc='upper left',
                            frameon=False, fontsize=9, title='Sample median')
    ax.add_artist(leg_markers)
    ax.legend(loc='lower right', frameon=False, fontsize=8, ncol=1)

    ax.set_xlabel(r'$\log_{10}(M_\mathrm{vir}\,/\,M_\odot)$')
    ax.set_ylabel(r'$b(M_\mathrm{vir},\,z)$')
    ax.set_yscale('log')
    ax.set_xlim(log_mvir_grid.min(), log_mvir_grid.max())
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.set_title('(b) Tinker+10 $b(M_\\mathrm{vir})$ at fixed $z$; markers = sample medians')

    # (c) Bias distribution at z_target
    ax = axes[2]
    lo = min(np.min(b1) if len(b1) else np.inf,
             np.min(b2) if len(b2) else np.inf)
    hi = max(np.max(b1) if len(b1) else -np.inf,
             np.max(b2) if len(b2) else -np.inf)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = 0.5, 5.0
    bins = np.linspace(lo, hi, 30)
    if len(b1):
        ax.hist(b1, bins=bins, color=MODEL1_COLOR, alpha=0.55,
                histtype='stepfilled', edgecolor=MODEL1_COLOR, lw=1.4,
                label=f'{label1} (median = {np.median(b1):.2f})')
    if len(b2):
        ax.hist(b2, bins=bins, color=MODEL2_COLOR, alpha=0.55,
                histtype='stepfilled', edgecolor=MODEL2_COLOR, lw=1.4,
                label=f'{label2} (median = {np.median(b2):.2f})')
    ax.set_xlabel(r'$b(M_\mathrm{vir},\,z_\mathrm{target})$')
    ax.set_ylabel(r'$N$')
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.legend(loc='best', frameon=False, fontsize=9)
    ax.set_title('(c) bias distribution at $z_\\mathrm{target}$')

    fig.suptitle(
        f'Tinker+10 halo bias of quiescent galaxies at z = {z_target:.2f}  '
        f'(top {len(b1)} matched; sSFR $< {SSFR_FACTOR}/t_H$)\n'
        f'$N_\\mathrm{{q}}$: {label1} = {n_q_m1},  {label2} = {n_q_m2}.  '
        f'{title_extra}',
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


# ----- Driver -----------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--redshift',     type=float, default=DEFAULT_REDSHIFT)
    p.add_argument('--n-top',        type=int,   default=DEFAULT_N_TOP)
    p.add_argument('--model1',       default=DEFAULT_MODEL1)
    p.add_argument('--model2',       default=DEFAULT_MODEL2)
    p.add_argument('--model1-label', default=DEFAULT_MODEL1_LABEL)
    p.add_argument('--model2-label', default=DEFAULT_MODEL2_LABEL)
    p.add_argument('--output-dir',   default=DEFAULT_OUTPUT_DIR)
    p.add_argument('--format',       default=DEFAULT_FORMAT)
    p.add_argument('--sigma-8',      type=float, default=DEFAULT_SIGMA_8,
                   help='sigma_8 normalisation of the linear matter power spectrum.')
    p.add_argument('--n-s',          type=float, default=DEFAULT_N_S,
                   help='Primordial scalar tilt n_s.')
    p.add_argument('--omega-b',      type=float, default=DEFAULT_OMEGA_B,
                   help='Baryon density parameter Omega_b.')
    p.add_argument('--delta',        type=float, default=DEFAULT_DELTA,
                   help='Halo overdensity Delta for Tinker+10 parameters.')
    args = p.parse_args(argv)

    print('=' * 70)
    print('Quiescent galaxy bias (Tinker+10)')
    print('=' * 70)

    hdr1 = _read_header(args.model1)
    hdr2 = _read_header(args.model2)
    if hdr1 is None: sys.exit(f'No model files in {args.model1}')
    if hdr2 is None: sys.exit(f'No model files in {args.model2}')

    redshifts = hdr1['redshifts']
    omega_m   = hdr1['omega_m']
    omega_l   = hdr1['omega_l']
    hubble_h  = hdr1['hubble_h']
    print(f'  cosmo: Omega_m={omega_m}, Omega_L={omega_l}, h={hubble_h}, '
          f'Omega_b={args.omega_b}, sigma_8={args.sigma_8}, n_s={args.n_s}, '
          f'Delta={args.delta}')
    print(f'  model1 files: {hdr1["n_files"]}    model2 files: {hdr2["n_files"]}')

    snap_target = _snap_nearest_z(redshifts, args.redshift, hdr1['output_snaps'])
    z_target    = redshifts[snap_target]
    print(f'  Target redshift z={args.redshift}  ->  snap {snap_target} (z={z_target:.3f})')

    if snap_target not in hdr2['output_snaps']:
        sys.exit(f'Model 2 lacks Snap_{snap_target}')

    print('\nBuilding sigma(M) table from Eisenstein-Hu 1998 (no-wiggle)...')
    sigma_table = _build_sigma_table(
        omega_m=omega_m, omega_b=args.omega_b, h=hubble_h,
        sigma_8=args.sigma_8, n_s=args.n_s)
    # Sanity check: sigma at the mass enclosed in a top-hat of R = 8 Mpc/h
    # should equal sigma_8.
    rho_crit_0 = 2.775e11 * hubble_h**2
    rho_m_0    = omega_m * rho_crit_0
    R_phys     = 8.0 / hubble_h
    M_8        = (4.0/3.0) * np.pi * rho_m_0 * R_phys**3
    sig_check  = float(_sigma_of_M(np.array([M_8]), sigma_table)[0])
    print(f'  sigma(R=8 Mpc/h, z=0) = {sig_check:.3f}  '
          f'(should match sigma_8 = {args.sigma_8})')

    # Diagnostic: sigma(M, z=0) and Tinker b(M, z=z_target) at reference masses.
    print('  Diagnostic sigma(M) / b(M,z) at reference masses:')
    print(f'    {"log10(M)":>10}  {"sigma_0":>8}  {"D(z_t)":>8}  '
          f'{"sigma_z":>8}  {"nu":>6}  {"b(M,z_t)":>9}')
    D_t = _growth_factor(z_target, omega_m, omega_l)
    for logm in [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]:
        M = 10.0**logm
        s0 = float(_sigma_of_M(np.array([M]), sigma_table)[0])
        sz = s0 * D_t
        nu = DELTA_C / max(sz, 1e-30)
        bb = float(_bias_for_galaxies(np.array([M]), z_target, sigma_table,
                                      omega_m, omega_l, delta=args.delta)[0])
        print(f'    {logm:>10.2f}  {s0:>8.3f}  {D_t:>8.3f}  '
              f'{sz:>8.3f}  {nu:>6.3f}  {bb:>9.3f}')

    print('\nLoading model histories (snap 0 .. snap_target)...')
    _, snaps1 = _load_all_snaps_upto(args.model1, PROPS_TO_LOAD, snap_target)
    _, snaps2 = _load_all_snaps_upto(args.model2, PROPS_TO_LOAD, snap_target)
    print(f'  model1 loaded {len(snaps1)} snapshots;  model2 loaded {len(snaps2)} snapshots.')

    if snap_target not in snaps1 or snap_target not in snaps2:
        sys.exit('Target snap not loaded for one of the models.')

    # Independent selection in each model — the two recipes give different
    # host-Mvir distributions for the quiescent population, so bias differs.
    # (A matched-sample selection would be degenerate: same GalaxyIndex => same
    # Mvir => same Tinker bias by construction.)
    print(f'\nSelecting top {args.n_top} quiescent independently in each model '
          f'at z={z_target:.2f}...')
    rows1 = _select_top_quiescent(
        snaps1[snap_target], z_target, args.n_top, omega_m, omega_l, hubble_h)
    rows2 = _select_top_quiescent(
        snaps2[snap_target], z_target, args.n_top, omega_m, omega_l, hubble_h)
    if rows1.size == 0 or rows2.size == 0:
        sys.exit('No quiescent galaxies at target snap in one of the models.')
    print(f'  Selected: model1 N={len(rows1)};  model2 N={len(rows2)}')

    galids1 = snaps1[snap_target]['GalaxyIndex'][rows1].astype(np.int64)
    galids2 = snaps2[snap_target]['GalaxyIndex'][rows2].astype(np.int64)

    # Bias at z_target — per-galaxy
    d1_t = {k: v[rows1] for k, v in snaps1[snap_target].items()}
    d2_t = {k: v[rows2] for k, v in snaps2[snap_target].items()}
    b1   = _bias_for_galaxies(d1_t['Mvir'], z_target, sigma_table,
                              omega_m, omega_l, delta=args.delta)
    b2   = _bias_for_galaxies(d2_t['Mvir'], z_target, sigma_table,
                              omega_m, omega_l, delta=args.delta)
    print(f'  <b> at z_target: {args.model1_label}={np.median(b1):.2f} '
          f'(p16,p84)=({np.percentile(b1,16):.2f},{np.percentile(b1,84):.2f});  '
          f'{args.model2_label}={np.median(b2):.2f} '
          f'(p16,p84)=({np.percentile(b2,16):.2f},{np.percentile(b2,84):.2f})')

    # Tinker+10 curve at z_target across a range of M_vir
    mvir_curve = np.logspace(10.0, 15.5, 200)
    b_curve_t  = _bias_for_galaxies(mvir_curve, z_target, sigma_table,
                                    omega_m, omega_l, delta=args.delta)

    # Population counts for the title
    thr_t = _ssfr_threshold(z_target, omega_m, omega_l, hubble_h)
    n_q_m1 = _count_quiescent(snaps1[snap_target], thr_t)
    n_q_m2 = _count_quiescent(snaps2[snap_target], thr_t)
    print(f'  Total quiescent at z_target: model1={n_q_m1}, model2={n_q_m2}')

    # Trace each sample through its own history; collect median (log10 Mvir, b)
    # per snap so the middle panel can plot trajectories in the (Mvir, b) plane.
    print('\nTracing each model\'s sample through its own history...')
    snap_list  = sorted(s for s in snaps1.keys() if s in snaps2)
    z_samp     = np.array([redshifts[s] for s in snap_list])
    logmv_med1 = np.full(len(snap_list), np.nan)
    logmv_med2 = np.full(len(snap_list), np.nan)
    bz_med1    = np.full(len(snap_list), np.nan)
    bz_p16_1   = np.full(len(snap_list), np.nan)
    bz_p84_1   = np.full(len(snap_list), np.nan)
    bz_med2    = np.full(len(snap_list), np.nan)
    bz_p16_2   = np.full(len(snap_list), np.nan)
    bz_p84_2   = np.full(len(snap_list), np.nan)

    for si, s in enumerate(snap_list):
        d1 = snaps1[s]; d2 = snaps2[s]
        l1 = _build_lookup(d1['GalaxyIndex'])
        l2 = _build_lookup(d2['GalaxyIndex'])
        i1 = np.array([l1.get(int(g), -1) for g in galids1], dtype=np.int64)
        i2 = np.array([l2.get(int(g), -1) for g in galids2], dtype=np.int64)
        mv1 = d1['Mvir'][i1[i1 >= 0]] if (i1 >= 0).any() else np.array([])
        mv2 = d2['Mvir'][i2[i2 >= 0]] if (i2 >= 0).any() else np.array([])
        mv1 = mv1[mv1 > 0]; mv2 = mv2[mv2 > 0]
        z_s = redshifts[s]
        if mv1.size:
            bb1 = _bias_for_galaxies(mv1, z_s, sigma_table, omega_m, omega_l,
                                     delta=args.delta)
            logmv_med1[si] = np.log10(np.median(mv1))
            bz_med1[si], bz_p16_1[si], bz_p84_1[si] = (
                np.median(bb1), np.percentile(bb1, 16), np.percentile(bb1, 84))
        if mv2.size:
            bb2 = _bias_for_galaxies(mv2, z_s, sigma_table, omega_m, omega_l,
                                     delta=args.delta)
            logmv_med2[si] = np.log10(np.median(mv2))
            bz_med2[si], bz_p16_2[si], bz_p84_2[si] = (
                np.median(bb2), np.percentile(bb2, 16), np.percentile(bb2, 84))

    print('  Snap-by-snap trace:')
    print(f'    {"snap":>4} {"z":>6}  '
          f'{"log10<Mvir>(M1)":>16} {"<b>(M1)":>10}     '
          f'{"log10<Mvir>(M2)":>16} {"<b>(M2)":>10}')
    for si, s in enumerate(snap_list):
        if not (np.isfinite(bz_med1[si]) or np.isfinite(bz_med2[si])):
            continue
        m1_lm = f'{logmv_med1[si]:>16.2f}' if np.isfinite(logmv_med1[si]) else f'{"---":>16}'
        m1_b  = f'{bz_med1[si]:>10.2f}'    if np.isfinite(bz_med1[si])    else f'{"---":>10}'
        m2_lm = f'{logmv_med2[si]:>16.2f}' if np.isfinite(logmv_med2[si]) else f'{"---":>16}'
        m2_b  = f'{bz_med2[si]:>10.2f}'    if np.isfinite(bz_med2[si])    else f'{"---":>10}'
        print(f'    {s:>4d} {z_samp[si]:>6.2f}  {m1_lm} {m1_b}     {m2_lm} {m2_b}')

    # Plot
    os.makedirs(args.output_dir, exist_ok=True)
    tag = f'z{z_target:.2f}'.replace('.', 'p')
    out_path = os.path.join(args.output_dir, f'quiescent_bias_{tag}{args.format}')
    print('\nPlotting...')
    title_extra = (f'cosmo: $\\Omega_m={omega_m}$, $\\Omega_\\Lambda={omega_l}$, '
                   f'$h={hubble_h}$, $\\sigma_8={args.sigma_8}$, '
                   f'$\\Omega_b={args.omega_b}$, $n_s={args.n_s}$')
    # Extend Tinker curves to cover the full z-range of the traced sample so
    # the high-z markers have a curve to reference against.
    z_max_trace = float(np.nanmax(z_samp[np.isfinite(bz_med1) | np.isfinite(bz_med2)]))
    tinker_z_lines = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0]
    tinker_z_lines = [z for z in tinker_z_lines if z <= z_max_trace + 1.0]
    if max(tinker_z_lines) < z_max_trace:
        tinker_z_lines.append(round(z_max_trace, 1))
    _plot_bias(
        d1_t, d2_t, b1, b2, mvir_curve, b_curve_t,
        z_target, sigma_table, hdr1,
        args.model1_label, args.model2_label,
        z_samp,
        logmv_med1, bz_med1,
        logmv_med2, bz_med2,
        tinker_z_lines,
        n_q_m1, n_q_m2, title_extra, out_path, args)

    # Standalone trajectory figure (just the middle panel).
    out_traj = os.path.join(
        args.output_dir, f'quiescent_bias_trajectory_{tag}{args.format}')
    traj_title = (f'Tinker+10 $b(M_\\mathrm{{vir}})$ at fixed $z$  '
                  f'with quiescent-sample medians per snap\n'
                  f'(selected at z = {z_target:.2f}; '
                  f'$N_\\mathrm{{q}}$: {args.model1_label} = {n_q_m1}, '
                  f'{args.model2_label} = {n_q_m2})')
    _plot_bias_trajectory(
        mvir_curve, sigma_table, hdr1, args,
        args.model1_label, args.model2_label, z_samp,
        logmv_med1, bz_med1,
        logmv_med2, bz_med2,
        tinker_z_lines, traj_title, out_traj)

    print('\nDone.')


if __name__ == '__main__':
    main()

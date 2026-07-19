#!/usr/bin/env python
"""
Clustering bias of massive quiescent galaxies (MQGs) through cosmic time.
=========================================================================

Pipeline
--------
1. Identify MQGs at each requested redshift using the *Donnari-style evolving
   sSFR floor*.  A galaxy is quiescent if

       sSFR < sSFR_floor(z) = ssfr0 * E(z),
       E(z) = H(z)/H0 = sqrt(Omega_m (1+z)^3 + Omega_L),

   anchored at z=0 to ``ssfr0`` (default 1e-11 /yr -- the canonical Donnari
   et al. 2019 fixed cut) and evolving with the Hubble rate, so the boundary
   tracks the declining characteristic sSFR of the star-forming population
   without requiring a per-snapshot main-sequence fit.  "Massive" means
   log10(M*/Msun) > ``min_logmstar``.

2. Measure the real-space clustering bias of the MQG sample directly from the
   galaxy positions (Posx/y/z, comoving Mpc/h) in the periodic box:

       b(z) = sqrt( < xi_gg(r) / xi_mm(r, z) > ),   r in [rmin_fit, rmax_fit].

   xi_gg is the galaxy autocorrelation (Corrfunc periodic estimator, with a
   scipy.cKDTree natural-estimator fallback); xi_mm is the *linear* matter
   correlation function from colossus, evaluated at the sample redshift.
   Uncertainties come from a 3D sub-cube (delete-one) jackknife.

3. Compare to the Tinker et al. (2010) large-scale halo bias b(Mvir, z),
   evaluated via colossus at the MQG host-halo masses.

Figures
-------
  (A) b vs log10(Mvir): one measured point per redshift (colour = z, jackknife
      error bars) tracing the population through time, with Tinker+10 curves at
      each redshift overlaid.                    -> "bias as a function of Mvir"
  (B) b vs z: measured MQG clustering bias and the Tinker+10 prediction at the
      sample median Mvir, side by side.          -> "bias through time"
  (C) diagnostic: xi_gg(r) with the best-fit b^2 * xi_mm(r) at each redshift.

Usage
-----
    python plotting/random_plotting_scripts/mqg_clustering_bias.py
    python plotting/random_plotting_scripts/mqg_clustering_bias.py \\
        --model output/microuchuu --redshifts 0 0.5 1 2 3 \\
        --min-logmstar 10.5 --sigma-8 0.8159 --n-s 0.9667

Notes
-----
* Defaults target microUchuu (100 Mpc/h box, Planck-like cosmology).  For a
  different simulation pass the matching --sigma-8 / --n-s (they are not stored
  in the SAGE header); Omega_m, Omega_L, h and the box size ARE read from the
  header.
* MQGs are selected independently at each redshift (population bias evolution),
  not by tracing z=0 progenitors.
"""

import argparse
import glob
import os
import shutil
import sys
import warnings

import numpy as np
import h5py as h5
import matplotlib
matplotlib.use('Agg')          # headless: write figures without a display (HPC-safe)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings('ignore')

from colossus.cosmology import cosmology
from colossus.lss import bias as colossus_bias

try:
    from Corrfunc.theory import xi as _corrfunc_xi
    HAVE_CORRFUNC = True
except Exception:
    HAVE_CORRFUNC = False


# ----- Style ------------------------------------------------------------------

_STYLE = './plotting/kieren_cohare_palatino_sty.mplstyle'
if os.path.exists(_STYLE):
    plt.style.use(_STYLE)
# The style enables LaTeX text; fall back to mathtext on nodes without a TeX
# install (compute nodes usually lack one) so figures render instead of crashing.
if shutil.which('latex') is None:
    plt.rcParams['text.usetex'] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'


# ----- Defaults ---------------------------------------------------------------

DEFAULT_MODEL       = './output/microuchuu/'
DEFAULT_REDSHIFTS   = [0.0, 0.99, 1.91, 2.83, 3.87, 4.89]
DEFAULT_MIN_LOGMSTAR = 10.5
# Recommended default selection: a constant comoving abundance of the most
# massive quiescent galaxies.  This picks the massive, central-dominated host
# population (host Mvir ~= CentralMvir) that traces the clean declining
# bias-vs-mass trend, with enough objects for a robust xi_gg on a big box.
# On miniUchuu (400 Mpc/h) 1e-4 is N ~ 6400; on Uchuu it scales with the
# volume.  Go lower (~1.5e-5, N ~ 1000 on miniUchuu) for the most extreme
# massive-central trend; below ~1e-5 the sample becomes shot-noise limited.
DEFAULT_NUMBER_DENSITY = 1.0e-4      # (Mpc/h)^-3
DEFAULT_SSFR0       = 1.0e-11        # z=0 quiescence boundary (Donnari fixed cut)
DEFAULT_OUTPUT_DIR  = './output/mqg_clustering/'
DEFAULT_FORMAT      = '.pdf'
DEFAULT_OBS_FILE    = './data/clustering/quiescent_bias_obs.dat'

# Linear power-spectrum normalisation.  NOT stored in the SAGE header, so these
# default to the Uchuu / Planck-2015 values; override for other simulations.
DEFAULT_SIGMA_8     = 0.8159
DEFAULT_N_S         = 0.9667

DEFAULT_MDEF        = 'vir'           # SAGE Mvir is the (Bryan-Norman) virial mass

# Correlation-function binning (comoving Mpc/h).
DEFAULT_RMIN        = 0.5
DEFAULT_RMAX        = 30.0
DEFAULT_NBINS       = 16
DEFAULT_RMIN_FIT    = 5.0
DEFAULT_RMAX_FIT    = 25.0
DEFAULT_NJACK       = 3               # njack^3 sub-cubes
DEFAULT_SCAN_NLIST  = [50, 100, 200, 500, 1000, 5000]   # top-N values for the number scan
DEFAULT_MASS_BINS   = [10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5]   # log10 Msun (physical) bin edges

PROPS_TO_LOAD = ['StellarMass', 'Mvir', 'CentralMvir', 'SfrDisk', 'SfrBulge',
                 'Posx', 'Posy', 'Posz', 'Type']
_MASS_PROPS = frozenset({'StellarMass', 'Mvir', 'CentralMvir'})


# ----- File I/O ---------------------------------------------------------------

def find_files(directory):
    files = sorted(glob.glob(os.path.join(directory, 'model_*.hdf5')))
    if not files:
        single = os.path.join(directory, 'model_0.hdf5')
        if os.path.exists(single):
            files = [single]
    return files


def read_header(directory):
    """Read cosmology, box size and the snapshot->redshift map from the header."""
    files = find_files(directory)
    if not files:
        return None
    with h5.File(files[0], 'r') as f:
        sim = f['Header/Simulation']
        rt = f['Header/Runtime']
        h_val = float(sim.attrs['hubble_h'])
        hdr = {
            'hubble_h': h_val,
            'omega_m':  float(sim.attrs['omega_matter']),
            'omega_l':  float(sim.attrs['omega_lambda']),
            'box':      float(sim.attrs['box_size']),          # Mpc/h, comoving
            'redshifts': np.array(f['Header/snapshot_redshifts'][:], dtype=float),
            'unit_mass_g': float(rt.attrs['UnitMass_in_g']),
            'baryon_frac': float(rt.attrs.get('BaryonFrac',
                                sim.attrs.get('BaryonFrac', 0.17))),
        }
    snap_set = set()
    for fp in files:
        with h5.File(fp, 'r') as f:
            for k in f.keys():
                if k.startswith('Snap_'):
                    snap_set.add(int(k.replace('Snap_', '')))
    hdr['output_snaps'] = sorted(snap_set)
    hdr['n_files'] = len(files)
    hdr['files'] = files
    hdr['mass_conv'] = hdr['unit_mass_g'] / 1.989e33 / hdr['hubble_h']
    return hdr


def load_snap(files, snap_num, props, mass_conv):
    """Return {prop: array} concatenated across files for Snap_<snap_num>."""
    snap_key = f'Snap_{snap_num}'
    chunks = {p: [] for p in props}
    found = False
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
            if not snap_len:
                continue
            found = True
            for p in props:
                chunks[p].append(np.array(grp[p]) if p in grp
                                 else np.zeros(snap_len))
    if not found:
        return {}
    out = {}
    for p in props:
        arr = np.concatenate(chunks[p])
        if p in _MASS_PROPS:
            arr = arr * mass_conv
        out[p] = arr
    return out


def snap_nearest_z(redshifts, z_target, available):
    return min(available, key=lambda s: abs(redshifts[s] - z_target))


def load_obs_bias(path):
    """Read the observational bias compilation.

    Columns (whitespace-separated, '#' comments):
        z  logMhalo  logMhalo_err  bias  bias_err  reference
    Use 'nan' for any value not reported (that error bar is then omitted).
    """
    if not path or not os.path.exists(path):
        return None

    def _f(x):
        try:
            return float(x)
        except ValueError:
            return np.nan

    z, lmh, lmh_e, b, b_e, ref = [], [], [], [], [], []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            z.append(_f(parts[0]))
            lmh.append(_f(parts[1]))
            lmh_e.append(_f(parts[2]))
            b.append(_f(parts[3]))
            b_e.append(_f(parts[4]))
            ref.append(parts[5])
    if not z:
        return None
    return {'z': np.array(z), 'logMhalo': np.array(lmh),
            'logMhalo_err': np.array(lmh_e), 'bias': np.array(b),
            'bias_err': np.array(b_e), 'ref': ref}


# ----- MQG selection (Donnari-style evolving sSFR floor) ----------------------

def ssfr_floor(z, omega_m, omega_l, ssfr0):
    """Evolving sSFR quiescence boundary (1/yr): ssfr0 * E(z)."""
    e_z = np.sqrt(omega_m * (1.0 + z) ** 3 + omega_l)
    return ssfr0 * e_z


def select_mqg(d, z, hdr, ssfr0, min_logmstar=None, number_density=None,
               top_percent=None, top_n=None):
    """Boolean mask of massive quiescent galaxies; also returns the floor used.

    Quiescence is always the Donnari-style evolving sSFR floor.  "Massive" is
    defined by exactly one of the following (checked in this order):
      * top_n (explicit count)    : the top_n most massive quiescent galaxies
        (used by --match-highz-count: same count at every redshift).
      * number_density (Mpc/h)^-3 : the N = n*box^3 most massive quiescent galaxies.
      * min_logmstar              : fixed cut, log10(M*) > min_logmstar.
      * top_percent               : the most massive top_percent% of ALL galaxies
        at this snapshot (per-snapshot M* percentile threshold), then quiescent.
    Default (all None) is top_percent = 10.
    """
    sm = d['StellarMass']
    sfr = d['SfrDisk'] + d['SfrBulge']
    ssfr = np.where(sm > 0, sfr / np.maximum(sm, 1e-30), np.inf)
    floor = ssfr_floor(z, hdr['omega_m'], hdr['omega_l'], ssfr0)
    quiescent = (ssfr < floor) & (sm > 0)
    # count-based selection (explicit top_n takes priority over number_density)
    n_target = None
    if top_n is not None:
        n_target = int(top_n)
    elif number_density is not None:
        # value >= 1 is treated as a plain count N; < 1 as a density (Mpc/h)^-3.
        n_target = (int(round(number_density)) if number_density >= 1.0
                    else int(round(number_density * hdr['box'] ** 3)))
    if n_target is not None:
        qidx = np.where(quiescent)[0]
        mask = np.zeros(sm.shape, dtype=bool)
        if qidx.size and n_target > 0:
            mask[qidx[np.argsort(-sm[qidx])[:n_target]]] = True
        return mask, floor
    if min_logmstar is not None:
        return ((sm > 10.0 ** min_logmstar) & quiescent), floor
    pct = 10.0 if top_percent is None else top_percent
    pos = sm[sm > 0]
    thresh = np.percentile(pos, 100.0 - pct) if pos.size else np.inf
    return ((sm >= thresh) & quiescent), floor


def count_quiescent(d, z, hdr, ssfr0):
    """Number of Donnari-quiescent galaxies (M* > 0) at this snapshot."""
    sm = d['StellarMass']
    ssfr = np.where(sm > 0, (d['SfrDisk'] + d['SfrBulge']) / np.maximum(sm, 1e-30), np.inf)
    floor = ssfr_floor(z, hdr['omega_m'], hdr['omega_l'], ssfr0)
    return int(np.sum((ssfr < floor) & (sm > 0)))


# ----- Correlation function ---------------------------------------------------

def xi_gg(pos, box, r_edges, nthreads=2):
    """Galaxy autocorrelation xi(r) in a periodic box.

    pos : (N, 3) comoving Mpc/h.  Returns (xi, r_mid, npairs).
    """
    pos = np.mod(np.ascontiguousarray(pos, dtype=np.float64), box)
    x, y, zc = pos[:, 0].copy(), pos[:, 1].copy(), pos[:, 2].copy()
    if HAVE_CORRFUNC:
        res = _corrfunc_xi(box, nthreads, r_edges, x, y, zc, output_ravg=True)
        r_mid = np.where(res['ravg'] > 0, res['ravg'],
                         0.5 * (r_edges[:-1] + r_edges[1:]))
        return np.asarray(res['xi']), r_mid, np.asarray(res['npairs'])
    return _xi_scipy(pos, box, r_edges)


def _xi_scipy(pos, box, r_edges):
    """Natural-estimator xi(r) via periodic cKDTree pair counts (fallback)."""
    from scipy.spatial import cKDTree
    n = len(pos)
    tree = cKDTree(pos, boxsize=box)
    cum = tree.count_neighbors(tree, r_edges)          # ordered pairs incl. self
    dd = np.diff(cum).astype(float)                    # ordered pairs per shell
    v_shell = (4.0 / 3.0) * np.pi * (r_edges[1:] ** 3 - r_edges[:-1] ** 3)
    nbar = n / box ** 3
    rr = n * nbar * v_shell                            # ~ N^2 * Vshell / Vbox
    xi = np.where(rr > 0, dd / rr - 1.0, np.nan)
    r_mid = 0.5 * (r_edges[:-1] + r_edges[1:])
    return xi, r_mid, dd


def bias_from_ratio(xi_g, r_mid, z, cosmo, rmin_fit, rmax_fit):
    """b = sqrt(<xi_gg/xi_mm>) over the linear fitting range."""
    xi_m = cosmo.correlationFunction(r_mid, z)
    sel = ((r_mid >= rmin_fit) & (r_mid <= rmax_fit)
           & (xi_g > 0) & (xi_m > 0) & np.isfinite(xi_g))
    if not sel.any():
        return np.nan, xi_m, sel
    ratio = xi_g[sel] / xi_m[sel]
    return float(np.sqrt(np.mean(ratio))), xi_m, sel


def jackknife_bias(pos, box, r_edges, z, cosmo,
                   rmin_fit, rmax_fit, njack, nthreads):
    """Delete-one sub-cube jackknife error on the clustering bias."""
    if njack < 2:
        return np.nan
    cell = np.floor(np.mod(pos, box) / box * njack).astype(int)
    cell = np.clip(cell, 0, njack - 1)
    cid = (cell[:, 0] * njack + cell[:, 1]) * njack + cell[:, 2]
    b_samples = []
    for k in range(njack ** 3):
        keep = cid != k
        if keep.sum() < 10:
            continue
        xi_g, rm, _ = xi_gg(pos[keep], box, r_edges, nthreads)
        b_k, _, _ = bias_from_ratio(xi_g, rm, z, cosmo, rmin_fit, rmax_fit)
        if np.isfinite(b_k):
            b_samples.append(b_k)
    b_samples = np.array(b_samples)
    if b_samples.size < 2:
        return np.nan
    n = b_samples.size
    return float(np.sqrt((n - 1) / n * np.sum((b_samples - b_samples.mean()) ** 2)))


# ----- Tinker+2010 halo bias --------------------------------------------------

def tinker_bias(mvir_msun, z, cosmo, mdef):
    """Tinker+10 large-scale halo bias for Mvir given in Msun (physical)."""
    m_h = np.atleast_1d(mvir_msun) * cosmo.h           # -> Msun/h for colossus
    return colossus_bias.haloBias(m_h, z=z, mdef=mdef, model='tinker10')


# ----- Plotting ---------------------------------------------------------------

def _obs_refidx(obs):
    if obs is None:
        return None
    uniq = []
    for r in obs['ref']:
        if r not in uniq:
            uniq.append(r)
    return {r: i + 1 for i, r in enumerate(uniq)}


def _bias_mvir_panel(ax, results, tinker_zs, obs, cosmo, args, mkey,
                     z_norm, cmap, title, refidx):
    """Draw one bias-vs-halo-mass panel for the halo-mass field named by mkey."""
    xlo, xhi = args.mvir_lim
    mvir_grid = np.logspace(xlo, xhi, 200)
    for z_line in sorted(tinker_zs):
        ax.plot(np.log10(mvir_grid),
                tinker_bias(mvir_grid, z_line, cosmo, args.mdef),
                color=cmap(z_norm(z_line)), lw=1.6, alpha=0.9)

    zs = np.array([r['z'] for r in results])
    logmv = np.array([r[mkey] for r in results])
    b_meas = np.array([r['b'] for r in results])
    b_err = np.array([r['b_err'] for r in results])
    ax.errorbar(logmv, b_meas, yerr=b_err, fmt='none',
                ecolor='0.35', elinewidth=1.3, capsize=3, zorder=4)
    sc = ax.scatter(logmv, b_meas, c=zs, cmap=cmap, norm=z_norm,
                    s=140, marker='o', edgecolors='black', linewidths=1.2, zorder=5)

    # Observations (same host/effective halo masses on both panels, so the
    # observed reference stays fixed while the model points shift definition).
    if obs is not None and refidx is not None and np.isfinite(obs['bias']).any():
        xerr = np.where(np.isfinite(obs['logMhalo_err']) & (obs['logMhalo_err'] > 0),
                        obs['logMhalo_err'], np.nan)
        yerr = np.where(np.isfinite(obs['bias_err']) & (obs['bias_err'] > 0),
                        obs['bias_err'], np.nan)
        ax.errorbar(obs['logMhalo'], obs['bias'], xerr=xerr, yerr=yerr,
                    fmt='none', ecolor='0.55', elinewidth=1.0, capsize=2, zorder=6)
        ax.scatter(obs['logMhalo'], obs['bias'], c=obs['z'], cmap=cmap, norm=z_norm,
                   s=170, marker='*', edgecolors='black', linewidths=1.0, zorder=7)
        for xi, yi, r in zip(obs['logMhalo'], obs['bias'], obs['ref']):
            if np.isfinite(xi) and np.isfinite(yi):
                ax.annotate(f'[{refidx[r]}]', (xi, yi), textcoords='offset points',
                            xytext=(6, 4), fontsize=7, zorder=8)

    ax.set_xlim(xlo, xhi)
    ax.set_ylim(*args.bias_lim)
    ax.set_xlabel(r'$\log_{10}(M\,/\,M_\odot)$  (sample median)')
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.set_title(title, fontsize=12)
    return sc


def plot_bias_vs_mvir(results, tinker_zs, obs, cosmo, args, out_path,
                      scan_results=None, mass_bin_rows=None):
    """2x2: (a) host CentralMvir, (b) subhalo Mvir, (c) Mvir-binned median,
    (d) number-of-galaxies scan.  Panels a/b/c are z-coloured; d is N-coloured."""
    from matplotlib.colors import LogNorm
    from matplotlib.cm import ScalarMappable
    obs_zmax = float(np.nanmax(obs['z'])) if obs is not None else 0.0
    z_max = max(max(tinker_zs, default=0.0),
                max((r['z'] for r in results), default=0.0), obs_zmax, 1e-3)
    z_norm = plt.Normalize(vmin=0.0, vmax=z_max)
    cmap = plt.get_cmap('viridis')
    refidx = _obs_refidx(obs)
    xlo, xhi = args.mvir_lim
    mgrid = np.logspace(xlo, xhi, 200)

    fig, axes = plt.subplots(2, 2, figsize=(16.0, 13.0), sharey=True,
                             constrained_layout=True)

    sc = _bias_mvir_panel(
        axes[0, 0], results, tinker_zs, obs, cosmo, args, 'logmv_cen',
        z_norm, cmap,
        r'(a) host halo mass  $M_\mathrm{vir}^\mathrm{host}$ (CentralMvir)', refidx)
    _bias_mvir_panel(
        axes[0, 1], results, tinker_zs, obs, cosmo, args, 'logmv_sub',
        z_norm, cmap,
        r'(b) subhalo mass  $M_\mathrm{vir}$ (stripped for satellites)', refidx)

    # (c) binned by Mvir: one median track per redshift (no count/density knob)
    ax = axes[1, 0]
    for zl in sorted(tinker_zs):
        ax.plot(np.log10(mgrid), tinker_bias(mgrid, zl, cosmo, args.mdef),
                color=cmap(z_norm(zl)), lw=1.6, alpha=0.9)
    if mass_bin_rows:
        for zl in sorted(set(r['z'] for r in mass_bin_rows)):
            rr = sorted((r for r in mass_bin_rows
                         if r['z'] == zl and np.isfinite(r['b'])),
                        key=lambda x: x['med'])
            if rr:
                ax.errorbar([r['med'] for r in rr], [r['b'] for r in rr],
                            yerr=[r['be'] for r in rr], fmt='o-',
                            color=cmap(z_norm(zl)), ms=7, lw=1.6, mec='black',
                            mew=0.7, capsize=3, zorder=5)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(*args.bias_lim)
    ax.set_xlabel(r'$\log_{10}(M_\mathrm{vir}\,/\,M_\odot)$  (bin median)')
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.set_title(r'(c) binned by $M_\mathrm{vir}$ (median per bin; no count/density)',
                 fontsize=12)

    # (d) number scan: one track per top-N (subhalo Mvir), coloured by N
    ax = axes[1, 1]
    for zl in sorted(tinker_zs):
        ax.plot(np.log10(mgrid), tinker_bias(mgrid, zl, cosmo, args.mdef),
                color='0.85', lw=1.0, zorder=1)
    smN = None
    n_list = sorted(scan_results) if scan_results else []
    if n_list:
        nnorm = LogNorm(vmin=min(n_list), vmax=max(n_list))
        pcmap = plt.get_cmap('plasma')
        for N in n_list:
            rows = [r for r in scan_results[N]
                    if np.isfinite(r['b']) and np.isfinite(r['sub'])]
            if not rows:
                continue
            o = np.argsort([r['z'] for r in rows])
            ax.plot(np.array([rows[i]['sub'] for i in o]),
                    np.array([rows[i]['b'] for i in o]), '-',
                    color=pcmap(nnorm(N)), lw=1.8, marker='o', ms=6,
                    mec='black', mew=0.5, zorder=4)
        smN = ScalarMappable(norm=nnorm, cmap=pcmap)
        smN.set_array([])
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(*args.bias_lim)
    ax.set_xlabel(r'$\log_{10}(M_\mathrm{vir}\,/\,M_\odot)$  (sample median)')
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.set_title(r'(d) subhalo $M_\mathrm{vir}$ vs number of galaxies (top-$N$)',
                 fontsize=12)

    axes[0, 0].set_ylabel(r'large-scale bias $b$')
    axes[1, 0].set_ylabel(r'large-scale bias $b$')

    # z colorbar for the top row (a, b; also the scheme for c); N colorbar bottom
    cb = fig.colorbar(sc, ax=[axes[0, 0], axes[0, 1]], location='right',
                      pad=0.02, shrink=0.9)
    cb.set_label(r'redshift $z$')
    if smN is not None:
        cbN = fig.colorbar(smN, ax=[axes[1, 0], axes[1, 1]], location='right',
                           pad=0.02, shrink=0.9)
        cbN.set_label(r'$N$ (top-$N$ most massive quiescent)')

    meas_handle = Line2D([0], [0], marker='o', color='0.6', markeredgecolor='black',
                         markersize=11, lw=0, label='Measured MQG (this work)')
    tinker_handle = Line2D([0], [0], color='0.4', lw=1.7,
                           label=r'Tinker+10 $b(M,z)$')
    handles = [meas_handle]
    if obs is not None:
        handles.append(Line2D([0], [0], marker='*', color='0.6',
                              markeredgecolor='black', markersize=15, lw=0,
                              label='Observed quiescent/passive'))
    handles.append(tinker_handle)
    axes[0, 0].legend(handles=handles, loc='upper left', frameon=False, fontsize=9)

    if refidx is not None:
        caption = 'Obs: ' + ';  '.join(f'[{i}] {r}' for r, i in refidx.items())
        fig.text(0.5, -0.03, caption, ha='center', va='top', fontsize=7.5)
    fig.suptitle('MQG clustering bias vs halo mass — (a) host, (b) subhalo, '
                 '(c) Mvir-binned median, (d) vs number of galaxies', fontsize=13)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


def _bias_z_panel(ax, results, obs, tink_key, title):
    zs = np.array([r['z'] for r in results])
    b_meas = np.array([r['b'] for r in results])
    b_err = np.array([r['b_err'] for r in results])
    b_tink = np.array([r[tink_key] for r in results])
    order = np.argsort(zs)
    ax.errorbar(zs[order], b_meas[order], yerr=b_err[order], fmt='o-',
                color='#175cdb', lw=1.8, capsize=3, markersize=8,
                label='Measured MQG clustering bias')
    ax.plot(zs[order], b_tink[order], 's--', color='#d83a21', lw=1.8,
            markersize=8, label=r'Tinker+10 at median halo mass')
    if obs is not None and np.isfinite(obs['bias']).any():
        yerr = np.where(np.isfinite(obs['bias_err']) & (obs['bias_err'] > 0),
                        obs['bias_err'], np.nan)
        ax.errorbar(obs['z'], obs['bias'], yerr=yerr, fmt='*', color='0.25',
                    markersize=13, markeredgecolor='black', markeredgewidth=0.8,
                    capsize=2, lw=0, label='Observed quiescent/passive', zorder=6)
    ax.set_xlabel(r'redshift $z$')
    ax.set_ylim(bottom=0.0)
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.set_title(title, fontsize=12)


def plot_bias_vs_z(results, obs, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.0), sharey=True)
    _bias_z_panel(axes[0], results, obs, 'b_tink_cen',
                  r'(a) Tinker+10 at median host mass (CentralMvir)')
    _bias_z_panel(axes[1], results, obs, 'b_tink_sub',
                  r'(b) Tinker+10 at median subhalo mass (Mvir)')
    axes[0].set_ylabel(r'large-scale bias $b$')
    axes[0].legend(loc='upper left', frameon=False, fontsize=10)
    fig.suptitle('MQG clustering bias through cosmic time — '
                 'measured (identical) vs Tinker+10 for each halo-mass definition',
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


def plot_xi_diagnostic(results, cosmo, out_path):
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    zs = np.array([r['z'] for r in results])
    z_norm = plt.Normalize(vmin=0.0, vmax=max(zs.max(), 1e-3))
    cmap = plt.get_cmap('viridis')
    for r in results:
        col = cmap(z_norm(r['z']))
        rm = r['r_mid']
        ok = r['xi_gg'] > 0
        ax.plot(rm[ok], r['xi_gg'][ok], 'o', color=col, markersize=5,
                label=rf"$z={r['z']:.2f}$ ($b={r['b']:.2f}$)")
        xi_fit = r['b'] ** 2 * cosmo.correlationFunction(rm, r['z'])
        ax.plot(rm, xi_fit, '-', color=col, lw=1.4, alpha=0.8)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$r$  [$h^{-1}\,\mathrm{Mpc}$]')
    ax.set_ylabel(r'$\xi(r)$')
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.legend(loc='best', frameon=False, fontsize=9,
              title=r'points: $\xi_\mathrm{gg}$;  lines: $b^2\,\xi_\mathrm{mm}$')
    ax.set_title('MQG autocorrelation vs linear-matter fit')
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


# ----- Number scan (default plot) ---------------------------------------------

def accumulate_number_scan(scan_results, d, z, hdr, cosmo, args, r_edges):
    """For the snapshot in `d`, measure bias + median host/subhalo mass of the
    top-N most massive quiescent for each N in args.scan_n_list, appending to
    scan_results[N].  The quiescent list is sorted once and sliced per N."""
    sm = d['StellarMass']
    ssfr = np.where(sm > 0, (d['SfrDisk'] + d['SfrBulge']) / np.maximum(sm, 1e-30), np.inf)
    floor = ssfr_floor(z, hdr['omega_m'], hdr['omega_l'], args.ssfr0)
    order = np.where((ssfr < floor) & (sm > 0))[0]
    order = order[np.argsort(-sm[order])]
    pos_all = np.column_stack([d['Posx'], d['Posy'], d['Posz']])
    cmv = d['CentralMvir']                                # physical Msun
    mv = d['Mvir']
    _lm = lambda x: float(np.log10(np.median(x[x > 0]))) if np.any(x > 0) else np.nan
    for N in args.scan_n_list:
        sel = order[:N]
        n = int(sel.size)
        row = {'z': z, 'n': n, 'host': np.nan, 'sub': np.nan, 'b': np.nan}
        if n >= 2:
            xg, rm, _ = xi_gg(pos_all[sel], hdr['box'], r_edges, args.nthreads)
            b, _, _ = bias_from_ratio(xg, rm, z, cosmo, args.rmin_fit, args.rmax_fit)
            row.update(host=_lm(cmv[sel]), sub=_lm(mv[sel]), b=float(b))
        scan_results.setdefault(int(N), []).append(row)


def plot_number_scan(scan_results, tinker_zs, cosmo, args, out_path):
    """Bias vs host | subhalo mass, one track per top-N, colour-coded by N."""
    from matplotlib.colors import LogNorm
    from matplotlib.cm import ScalarMappable
    n_list = sorted(scan_results)
    if not n_list:
        return
    norm = LogNorm(vmin=min(n_list), vmax=max(n_list))
    cmap = plt.get_cmap('plasma')
    mgrid = np.logspace(10, 15, 200)
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.6), sharey=True,
                             constrained_layout=True)

    def panel(ax, key, title):
        for zl in sorted(tinker_zs):
            ax.plot(np.log10(mgrid), tinker_bias(mgrid, zl, cosmo, args.mdef),
                    color='0.85', lw=1.0, zorder=1)
        for N in n_list:
            rows = [r for r in scan_results[N]
                    if np.isfinite(r['b']) and np.isfinite(r[key])]
            if not rows:
                continue
            o = np.argsort([r['z'] for r in rows])
            x = np.array([rows[i][key] for i in o])
            b = np.array([rows[i]['b'] for i in o])
            ax.plot(x, b, '-', color=cmap(norm(N)), lw=1.8, marker='o', ms=6,
                    mec='black', mew=0.5, zorder=4)
        ax.set_xlim(10.0, 15.0)
        ax.set_ylim(0.0, 8.0)
        ax.set_xlabel(r'$\log_{10}(M\,/\,M_\odot)$  (sample median)')
        ax.tick_params(which='both', direction='in', top=True, right=True)
        ax.set_title(title, fontsize=12)

    panel(axes[0], 'host', r'(a) host halo mass $M_\mathrm{vir}^\mathrm{host}$ (CentralMvir)')
    panel(axes[1], 'sub', r'(b) subhalo mass $M_\mathrm{vir}$ (plain)')
    axes[0].set_ylabel(r'measured large-scale bias $b$')
    sm_ = ScalarMappable(norm=norm, cmap=cmap)
    sm_.set_array([])
    cb = fig.colorbar(sm_, ax=axes, location='right', pad=0.02, shrink=0.9)
    cb.set_label(r'$N$ (top-$N$ most massive quiescent)')
    fig.suptitle('MQG bias vs halo mass: number scan (each track a fixed top-N; '
                 'grey=Tinker+10). Small N: massive, declining L;  large N: flat J.',
                 fontsize=12)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


# ----- Mass-binned validation mode (trees-free) -------------------------------

def accumulate_mass_bins(rows, d, z, hdr, cosmo, args, r_edges, edges):
    """For the loaded snapshot, bin quiescent galaxies by their own Mvir and
    measure each bin's autocorrelation bias, appending (z, bin) rows.  Feeds the
    default b(Mvir,z) validation figure -- no count/density, no trees."""
    sm = d['StellarMass']
    ssfr = np.where(sm > 0, (d['SfrDisk'] + d['SfrBulge']) / np.maximum(sm, 1e-30), np.inf)
    floor = ssfr_floor(z, hdr['omega_m'], hdr['omega_l'], args.ssfr0)
    q = (ssfr < floor) & (sm > 0)
    mv = d['Mvir']                                        # physical Msun
    pos_all = np.column_stack([d['Posx'], d['Posy'], d['Posz']])
    logmv = np.log10(np.where(mv > 0, mv, np.nan))
    print(f'  mass-bins: {int(q.sum())} quiescent')
    for i in range(edges.size - 1):
        lo, hi = edges[i], edges[i + 1]
        inbin = q & (logmv >= lo) & (logmv < hi)
        n = int(inbin.sum())
        if n < args.min_per_bin:
            print(f'    [{lo:.1f},{hi:.1f})  n={n:<5d}  (below --min-per-bin; skipped)')
            continue
        pos = pos_all[inbin]
        xi_g, rm, _ = xi_gg(pos, hdr['box'], r_edges, args.nthreads)
        b, _, _ = bias_from_ratio(xi_g, rm, z, cosmo, args.rmin_fit, args.rmax_fit)
        be = jackknife_bias(pos, hdr['box'], r_edges, z, cosmo,
                            args.rmin_fit, args.rmax_fit, args.njack, args.nthreads)
        med = float(np.median(logmv[inbin]))
        rows.append(dict(z=z, lo=lo, hi=hi, med=med, n=n, b=b, be=be))
        print(f'    [{lo:.1f},{hi:.1f})  n={n:<5d}  med={med:.2f}  b={b:.2f}+/-{be:.2f}')


def plot_binned_auto(rows, tinker_zs, obs, cosmo, args, out_path):
    """b(Mvir,z) binned (one track per z) vs Tinker+10 and observations.
    Axis is physical Msun (matches the obs file); no h-shift needed."""
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    zmodel = list(tinker_zs)
    zall = zmodel + ([float(zz) for zz in obs['z'] if np.isfinite(zz)]
                     if obs is not None else [])
    znorm = Normalize(vmin=min(zall), vmax=max(zall))
    cmap = plt.get_cmap('viridis')
    xlo, xhi = args.mvir_lim
    mgrid = np.logspace(xlo, xhi, 200)
    fig, ax = plt.subplots(figsize=(10.0, 7.2), constrained_layout=True)
    for zl in sorted(tinker_zs):
        ax.plot(np.log10(mgrid), tinker_bias(mgrid, zl, cosmo, args.mdef),
                color=cmap(znorm(zl)), lw=1.4, alpha=0.85, zorder=1)
    for zl in sorted(set(r['z'] for r in rows)):
        rr = sorted((r for r in rows if r['z'] == zl and np.isfinite(r['b'])),
                    key=lambda x: x['med'])
        if not rr:
            continue
        ax.errorbar([r['med'] for r in rr], [r['b'] for r in rr],
                    yerr=[r['be'] for r in rr], fmt='o-', color=cmap(znorm(zl)),
                    ms=8, lw=1.6, mec='black', mew=0.8, capsize=3, zorder=5,
                    label=f'model $z={zl:.2f}$')
    if obs is not None and np.isfinite(obs['bias']).any():
        refidx = _obs_refidx(obs)
        xerr = np.where(np.isfinite(obs['logMhalo_err']) & (obs['logMhalo_err'] > 0),
                        obs['logMhalo_err'], np.nan)
        yerr = np.where(np.isfinite(obs['bias_err']) & (obs['bias_err'] > 0),
                        obs['bias_err'], np.nan)
        ax.errorbar(obs['logMhalo'], obs['bias'], xerr=xerr, yerr=yerr, fmt='none',
                    ecolor='0.5', elinewidth=1.0, capsize=2, zorder=6)
        ax.scatter(obs['logMhalo'], obs['bias'], c=obs['z'], cmap=cmap, norm=znorm,
                   s=180, marker='*', edgecolors='black', linewidths=1.0, zorder=7)
        for xi, yi, rf in zip(obs['logMhalo'], obs['bias'], obs['ref']):
            if np.isfinite(xi) and np.isfinite(yi):
                ax.annotate(f'[{refidx[rf]}]', (xi, yi), textcoords='offset points',
                            xytext=(5, 4), fontsize=7, zorder=8)
        cap = '  '.join(f'[{i}] {r}' for r, i in refidx.items())
        fig.text(0.01, -0.02, 'obs: ' + cap, fontsize=7, ha='left', va='top')
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(*args.bias_lim)
    ax.set_xlabel(r'$\log_{10}(M_\mathrm{vir}\,/\,M_\odot)$  (bin median)')
    ax.set_ylabel(r'large-scale bias $b$')
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.legend(loc='upper left', frameon=False, fontsize=8, ncol=2)
    cb = fig.colorbar(ScalarMappable(norm=znorm, cmap=cmap), ax=ax, pad=0.02)
    cb.set_label(r'redshift $z$')
    ax.set_title('Model validation: quiescent-galaxy bias vs Mvir '
                 '(lines=Tinker+10, circles=model binned by Mvir, stars=obs)',
                 fontsize=11)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


# ----- Driver -----------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default=DEFAULT_MODEL)
    p.add_argument('--redshifts', type=float, nargs='+', default=DEFAULT_REDSHIFTS)
    p.add_argument('--top-percent', type=float, default=None,
                   help='Alternative selection: keep the most massive TOP-PERCENT%% of '
                        'all galaxies at each snapshot (per-snapshot M* percentile), '
                        'then quiescent. Overrides the default number density.')
    p.add_argument('--min-logmstar', type=float, default=None,
                   help='Alternative selection: fixed stellar-mass cut log10(M*/Msun). '
                        'Overrides the default number density.')
    p.add_argument('--number-density', type=float, default=None,
                   help=f'Select the N most massive quiescent galaxies at a constant '
                        f'comoving abundance. A value >=1 is a plain COUNT (e.g. 100, '
                        f'500); a value <1 is a density (Mpc/h)^-3, N=n*box^3 '
                        f'(e.g. 1e-5 -> ~10^14 hosts). RECOMMENDED DEFAULT when no '
                        f'selection flag is given: {DEFAULT_NUMBER_DENSITY:.1e} (Mpc/h)^-3.')
    p.add_argument('--match-highz-count', action='store_true',
                   help='Count the quiescent galaxies at the HIGHEST requested redshift '
                        'and select that same number (most massive quiescent) at every '
                        'redshift -- abundance-matched to the high-z quiescent count. '
                        'Overrides all other selection options.')
    p.add_argument('--ssfr0', type=float, default=DEFAULT_SSFR0,
                   help='z=0 sSFR quiescence boundary in 1/yr (evolves as E(z)).')
    p.add_argument('--sigma-8', type=float, default=DEFAULT_SIGMA_8)
    p.add_argument('--n-s', type=float, default=DEFAULT_N_S)
    p.add_argument('--omega-b', type=float, default=None,
                   help='Omega_b; default = BaryonFrac * Omega_m from header.')
    p.add_argument('--mdef', default=DEFAULT_MDEF,
                   help="Halo mass definition for Tinker+10 (e.g. 'vir','200c','200m').")
    p.add_argument('--rmin', type=float, default=DEFAULT_RMIN)
    p.add_argument('--rmax', type=float, default=DEFAULT_RMAX)
    p.add_argument('--nbins', type=int, default=DEFAULT_NBINS)
    p.add_argument('--rmin-fit', type=float, default=DEFAULT_RMIN_FIT)
    p.add_argument('--rmax-fit', type=float, default=DEFAULT_RMAX_FIT)
    p.add_argument('--njack', type=int, default=DEFAULT_NJACK)
    p.add_argument('--scan-n-list', type=int, nargs='+', default=DEFAULT_SCAN_NLIST,
                   help='top-N values for the default number-scan plot.')
    p.add_argument('--no-number-scan', action='store_true',
                   help='skip the default number-scan plot.')
    p.add_argument('--nthreads', type=int, default=2)
    p.add_argument('--mvir-lim', type=float, nargs=2, default=[10.0, 15.0],
                   metavar=('LOGMIN', 'LOGMAX'),
                   help='x-axis range for the bias-vs-Mvir figure (log10 Msun).')
    p.add_argument('--bias-lim', type=float, nargs=2, default=[0.0, 8.5],
                   metavar=('BMIN', 'BMAX'),
                   help='y-axis (bias) range for the bias-vs-Mvir figure.')
    p.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    p.add_argument('--format', default=DEFAULT_FORMAT)
    p.add_argument('--obs-file', default=DEFAULT_OBS_FILE,
                   help='Observational bias compilation to overlay '
                        '(z logMhalo logMhalo_err bias bias_err reference). '
                        'Pass "none" to disable.')
    p.add_argument('--mass-bins', type=float, nargs='+', default=DEFAULT_MASS_BINS,
                   metavar='LOGM',
                   help='Mvir bin edges (log10 Msun, physical) for the DEFAULT '
                        'b(Mvir,z) validation figure -- ALL quiescent galaxies '
                        'binned by their own Mvir, per-bin autocorrelation bias, '
                        'no count/density. Override the edges here.')
    p.add_argument('--no-mass-bins', action='store_true',
                   help='skip the default b(Mvir,z) mass-binned validation figure.')
    p.add_argument('--min-per-bin', type=int, default=5,
                   help='skip an Mvir bin with fewer quiescent galaxies than this.')
    args = p.parse_args(argv)

    # Recommended default: constant-abundance selection.  Only applied when the
    # user gave no explicit selection flag (an explicit flag always wins).
    if (args.number_density is None and args.min_logmstar is None
            and args.top_percent is None and not args.match_highz_count):
        args.number_density = DEFAULT_NUMBER_DENSITY

    print('=' * 72)
    print('MQG clustering bias through time (measured xi_gg/xi_mm vs Tinker+10)')
    print('=' * 72)
    print(f'  Corrfunc available: {HAVE_CORRFUNC}'
          f'{"" if HAVE_CORRFUNC else "  (using scipy cKDTree fallback)"}')

    hdr = read_header(args.model)
    if hdr is None:
        sys.exit(f'No model_*.hdf5 files in {args.model}')
    redshifts = hdr['redshifts']
    omega_b = (args.omega_b if args.omega_b is not None
               else hdr['baryon_frac'] * hdr['omega_m'])
    print(f'  model: {args.model}  ({hdr["n_files"]} file(s), '
          f'box = {hdr["box"]:.1f} Mpc/h)')
    print(f'  cosmo: Om={hdr["omega_m"]}, OL={hdr["omega_l"]}, h={hdr["hubble_h"]}, '
          f'Ob={omega_b:.4f}, sigma8={args.sigma_8}, ns={args.n_s}')
    match_n = None
    if args.match_highz_count:
        z_top_req = max(args.redshifts)
        snap_top = snap_nearest_z(redshifts, z_top_req, hdr['output_snaps'])
        z_top = float(redshifts[snap_top])
        d_top = load_snap(hdr['files'], snap_top,
                          ['StellarMass', 'SfrDisk', 'SfrBulge'], hdr['mass_conv'])
        match_n = count_quiescent(d_top, z_top, hdr, args.ssfr0) if d_top else 0
        print(f'  match-highz-count: {match_n} quiescent at z={z_top:.2f} '
              f'(snap {snap_top}); using top-{match_n} most massive quiescent at every z')

    if match_n is not None:
        sel = f'top-{match_n} most massive quiescent (matched to z={z_top:.2f} count)'
    elif args.number_density is not None:
        if args.number_density >= 1.0:
            sel = f'top-{int(round(args.number_density))} most massive quiescent (fixed count)'
        else:
            n_box = int(round(args.number_density * hdr['box'] ** 3))
            sel = f'top-{n_box} most massive quiescent (n={args.number_density:.1e} (Mpc/h)^-3)'
    elif args.min_logmstar is not None:
        sel = f'log10(M*/Msun) > {args.min_logmstar}'
    else:
        sel = f'most massive {args.top_percent:g}% by M* per snapshot'
    print(f'  MQG: {sel}, sSFR < {args.ssfr0:.2e} * E(z) /yr')

    cosmo = cosmology.setCosmology('sage', {
        'flat': True, 'H0': hdr['hubble_h'] * 100.0, 'Om0': hdr['omega_m'],
        'Ob0': omega_b, 'sigma8': args.sigma_8, 'ns': args.n_s})

    obs = None if str(args.obs_file).lower() == 'none' else load_obs_bias(args.obs_file)
    if obs is not None:
        print(f'  obs overlay: {len(obs["z"])} points from {args.obs_file}')
    elif str(args.obs_file).lower() != 'none':
        print(f'  obs overlay: file not found ({args.obs_file}); skipping.')

    r_edges = np.logspace(np.log10(args.rmin), np.log10(args.rmax),
                          args.nbins + 1)
    mass_edges = np.array(sorted(args.mass_bins), dtype=float)

    results = []
    tinker_zs = []
    scan_results = {}
    mass_bin_rows = []
    for z_req in args.redshifts:
        snap = snap_nearest_z(redshifts, z_req, hdr['output_snaps'])
        z = float(redshifts[snap])
        if z not in tinker_zs:
            tinker_zs.append(z)
        print(f'\n--- z_req={z_req} -> Snap_{snap} (z={z:.3f}) ---')
        d = load_snap(hdr['files'], snap, PROPS_TO_LOAD, hdr['mass_conv'])
        if not d:
            print('  no data; skipping.')
            continue

        if not args.no_number_scan:
            accumulate_number_scan(scan_results, d, z, hdr, cosmo, args, r_edges)
        if not args.no_mass_bins:
            accumulate_mass_bins(mass_bin_rows, d, z, hdr, cosmo, args, r_edges, mass_edges)

        mask, floor = select_mqg(d, z, hdr, args.ssfr0,
                                 min_logmstar=args.min_logmstar,
                                 number_density=args.number_density,
                                 top_percent=args.top_percent,
                                 top_n=match_n)
        n_mqg = int(mask.sum())
        print(f'  sSFR floor = {floor:.3e} /yr;  N_MQG = {n_mqg}')
        if n_mqg < 2:
            print(f'  only {n_mqg} MQGs -- cannot measure clustering; skipping.')
            continue
        if n_mqg < 50:
            print(f'  WARNING: only {n_mqg} MQGs -- clustering/jackknife will be noisy.')

        pos = np.column_stack([d['Posx'][mask], d['Posy'][mask], d['Posz'][mask]])
        cen = d['CentralMvir'][mask]
        cen = cen[cen > 0]
        sub = d['Mvir'][mask]
        sub = sub[sub > 0]
        logmv_cen = float(np.log10(np.median(cen)))
        logmv_sub = float(np.log10(np.median(sub)))
        f_sat = float(np.mean(d['Type'][mask] > 0))

        xi_g, r_mid, _ = xi_gg(pos, hdr['box'], r_edges, args.nthreads)
        b, _, _ = bias_from_ratio(
            xi_g, r_mid, z, cosmo, args.rmin_fit, args.rmax_fit)
        b_err = jackknife_bias(
            pos, hdr['box'], r_edges, z, cosmo,
            args.rmin_fit, args.rmax_fit, args.njack, args.nthreads)
        b_tink_cen = float(tinker_bias(np.median(cen), z, cosmo, args.mdef)[0])
        b_tink_sub = float(tinker_bias(np.median(sub), z, cosmo, args.mdef)[0])

        print(f'  f_sat = {f_sat:.2f};  '
              f'<log10 Mh> host={logmv_cen:.2f} / sub={logmv_sub:.2f};  '
              f'b_meas = {b:.2f} +/- {b_err:.2f};  '
              f'b_Tinker host={b_tink_cen:.2f} / sub={b_tink_sub:.2f}')

        results.append({
            'z': z, 'snap': snap, 'n_mqg': n_mqg, 'f_sat': f_sat,
            'logmv_cen': logmv_cen, 'logmv_sub': logmv_sub,
            'b': b, 'b_err': b_err,
            'b_tink_cen': b_tink_cen, 'b_tink_sub': b_tink_sub,
            'r_mid': r_mid, 'xi_gg': xi_g,
        })

    if not results:
        sys.exit('No redshift produced a usable MQG sample.')

    print('\nSummary  (Mh_host = CentralMvir, Mh_sub = own Mvir):')
    print(f'  {"z":>6} {"snap":>5} {"N_MQG":>7} {"f_sat":>6} '
          f'{"logMh_host":>10} {"logMh_sub":>10} '
          f'{"b_meas":>8} {"b_err":>7} {"bT_host":>8} {"bT_sub":>8}')
    for r in sorted(results, key=lambda x: x['z']):
        print(f'  {r["z"]:>6.2f} {r["snap"]:>5d} {r["n_mqg"]:>7d} '
              f'{r["f_sat"]:>6.2f} {r["logmv_cen"]:>10.2f} {r["logmv_sub"]:>10.2f} '
              f'{r["b"]:>8.2f} {r["b_err"]:>7.2f} '
              f'{r["b_tink_cen"]:>8.2f} {r["b_tink_sub"]:>8.2f}')

    os.makedirs(args.output_dir, exist_ok=True)
    fmt = args.format
    plot_bias_vs_mvir(results, tinker_zs, obs, cosmo, args,
                      os.path.join(args.output_dir, f'mqg_bias_vs_mvir{fmt}'),
                      scan_results=scan_results, mass_bin_rows=mass_bin_rows)
    plot_bias_vs_z(results, obs,
                   os.path.join(args.output_dir, f'mqg_bias_vs_z{fmt}'))
    plot_xi_diagnostic(results, cosmo,
                       os.path.join(args.output_dir, f'mqg_xi_diagnostic{fmt}'))
    if not args.no_number_scan and scan_results:
        plot_number_scan(scan_results, tinker_zs, cosmo, args,
                         os.path.join(args.output_dir, f'mqg_number_scan{fmt}'))
    if not args.no_mass_bins and mass_bin_rows:
        plot_binned_auto(mass_bin_rows, tinker_zs, obs, cosmo, args,
                         os.path.join(args.output_dir, f'mqg_bias_massbins{fmt}'))
    print('\nDone.')


if __name__ == '__main__':
    main()

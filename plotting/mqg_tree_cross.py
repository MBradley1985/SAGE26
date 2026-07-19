#!/usr/bin/env python
"""
Robust MQG clustering bias via cross-correlation with the halo-tree field.
=========================================================================

The MQG autocorrelation bias (see ``mqg_clustering_bias.py``) is shot-noise
limited when the sample is sparse -- exactly the high-z / rare-selection regime
where a fixed count or high abundance over-reaches the massive-quiescent supply.
This script makes the measurement robust by cross-correlating the (sparse) MQG
sample against the *dense* halo field taken straight from the merger trees:

    xi_gh = b_g b_h xi_mm ,   xi_hh = b_h^2 xi_mm
    =>  b_g = xi_gh / sqrt( xi_hh * xi_mm )          (per bin, averaged over the
                                                       linear fitting range)

Because the cross term pairs the few MQGs against ~10^6 halos, its noise is set
by the dense sample, not by the MQG count.  The halo autocorrelation also gives
a *measured* halo-bias baseline b_h(z), an apples-to-apples check on the
analytic Tinker+10 curve (same box, cosmology, estimator, mass definition).

It also compares the three spherical-overdensity mass definitions the trees
carry directly -- M_TopHat (~virial), M_Crit200 (200c), M_Mean200 (200m) --
so the Tinker mass-definition systematic is explicit.

Halo field: the LHaloTree *binary* files (fast: one np.fromfile per file).  The
struct layout is read from ``src/core_simulation.h`` (struct halo_data, 104 B).

Usage
-----
    python plotting/mqg_tree_cross.py \
        --model output/miniuchuu \
        --tree-dir /path/to/trees --tree-name miniUchuu_STC --tree-nfiles 8 \
        --redshifts 0 0.99 1.91 2.83 3.87 --number-density 1e-5 --nthreads 8

Runs the same MQG selection as mqg_clustering_bias (imported as a module), so
--number-density / --top-percent / --min-logmstar / --match-highz-count all work.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mqg_clustering_bias as m  # noqa: E402  (Agg + LaTeX fallback + style + I/O)
from colossus.cosmology import cosmology  # noqa: E402
from colossus.halo import mass_defs  # noqa: E402
from colossus.lss import bias as colossus_bias  # noqa: E402

try:
    from Corrfunc.theory.DD import DD as _corrfunc_DD
    HAVE_CORRFUNC_DD = True
except Exception:
    HAVE_CORRFUNC_DD = False


# LHaloTree binary struct halo_data (see src/core_simulation.h): 104 bytes,
# naturally packed (MostBoundID 8-byte aligned at offset 80).  Masses are in
# 10^10 Msun/h; Pos in comoving Mpc/h.  The union {Mvir; M200c} is M_Crit200
# for these catalogues.
HALO_DTYPE = np.dtype([
    ('Descendant', 'i4'), ('FirstProgenitor', 'i4'), ('NextProgenitor', 'i4'),
    ('FirstHaloInFOFgroup', 'i4'), ('NextHaloInFOFgroup', 'i4'),
    ('Len', 'i4'), ('M_Mean200', 'f4'), ('M200c', 'f4'), ('M_TopHat', 'f4'),
    ('Pos', 'f4', 3), ('Vel', 'f4', 3), ('VelDisp', 'f4'), ('Vmax', 'f4'),
    ('Spin', 'f4', 3), ('MostBoundID', 'i8'),
    ('SnapNum', 'i4'), ('FileNr', 'i4'), ('SubhaloIndex', 'i4'),
    ('SubHalfMass', 'f4'),
])
assert HALO_DTYPE.itemsize == 104, HALO_DTYPE.itemsize

MDEFS = [('vir', 'M_TopHat', r'$M_\mathrm{vir}$ (tophat)'),
         ('200c', 'M200c', r'$M_\mathrm{200c}$'),
         ('200m', 'M_Mean200', r'$M_\mathrm{200m}$')]

# Default host-halo-mass bin edges (log10 Msun/h) for the --mass-bins mode.
DEFAULT_MASS_BINS = [12.0, 12.5, 13.0, 13.5, 14.0, 14.5]


# ----- Tree halo field --------------------------------------------------------

def read_tree_halos(tree_dir, tree_name, nfiles, snap, min_len):
    """All halos at snapshot ``snap`` from the LHaloTree binary files.

    Returns dict with Pos (N,3, Mpc/h) and the three mass fields in Msun/h,
    concatenated across files and cut at Len >= min_len.
    """
    pos, mvir, m200c, m200m = [], [], [], []
    for i in range(nfiles):
        fp = os.path.join(tree_dir, f'{tree_name}.{i}')
        if not os.path.exists(fp):
            print(f'  [tree] missing {fp}; skipping')
            continue
        with open(fp, 'rb') as fh:
            ntrees = int(np.fromfile(fh, np.int32, 1)[0])
            tot = int(np.fromfile(fh, np.int32, 1)[0])
            np.fromfile(fh, np.int32, ntrees)               # per-tree NHalos
            h = np.fromfile(fh, HALO_DTYPE, tot)
        sel = (h['SnapNum'] == snap) & (h['Len'] >= min_len)
        hs = h[sel]
        pos.append(hs['Pos'].astype(np.float64))
        mvir.append(hs['M_TopHat'].astype(np.float64) * 1e10)
        m200c.append(hs['M200c'].astype(np.float64) * 1e10)
        m200m.append(hs['M_Mean200'].astype(np.float64) * 1e10)
    if not pos:
        return None
    return {'Pos': np.concatenate(pos), 'vir': np.concatenate(mvir),
            '200c': np.concatenate(m200c), '200m': np.concatenate(m200m)}


# ----- Cross-correlation ------------------------------------------------------

def xi_cross(p1, p2, box, r_edges, nthreads=2):
    """Cross-correlation xi_12(r) between two point sets in a periodic box.

    Analytic randoms: RR = N1*N2 * Vshell / Vbox.  Returns (xi, r_mid).
    """
    p1 = np.mod(np.ascontiguousarray(p1, np.float64), box)
    p2 = np.mod(np.ascontiguousarray(p2, np.float64), box)
    r_mid = 0.5 * (r_edges[:-1] + r_edges[1:])
    v_shell = (4.0 / 3.0) * np.pi * (r_edges[1:] ** 3 - r_edges[:-1] ** 3)
    n1, n2 = len(p1), len(p2)
    rr = n1 * n2 * v_shell / box ** 3
    if HAVE_CORRFUNC_DD:
        res = _corrfunc_DD(0, nthreads, r_edges,
                           p1[:, 0].copy(), p1[:, 1].copy(), p1[:, 2].copy(),
                           X2=p2[:, 0].copy(), Y2=p2[:, 1].copy(), Z2=p2[:, 2].copy(),
                           periodic=True, boxsize=box)
        dd = np.asarray(res['npairs'], dtype=float)
    else:
        from scipy.spatial import cKDTree
        cum = cKDTree(p1, boxsize=box).count_neighbors(
            cKDTree(p2, boxsize=box), r_edges)
        dd = np.diff(cum).astype(float)
    xi = np.where(rr > 0, dd / rr - 1.0, np.nan)
    return xi, r_mid


def cross_bias(xi_gh, xi_hh, r_mid, z, cosmo, rmin_fit, rmax_fit):
    """b_g from the cross term, and the measured halo bias b_h.

    b_h = sqrt(<xi_hh/xi_mm>);  b_g = <xi_gh/xi_mm> / b_h,  both averaged over
    the linear fitting range (bins with positive xi_hh and xi_mm).
    """
    xi_m = cosmo.correlationFunction(r_mid, z)
    sel = ((r_mid >= rmin_fit) & (r_mid <= rmax_fit)
           & (xi_hh > 0) & (xi_m > 0) & np.isfinite(xi_gh))
    if not sel.any():
        return np.nan, np.nan
    b_h = float(np.sqrt(np.mean(xi_hh[sel] / xi_m[sel])))
    if not np.isfinite(b_h) or b_h <= 0:
        return np.nan, b_h
    b_g = float(np.mean(xi_gh[sel] / xi_m[sel]) / b_h)
    return b_g, b_h


def jackknife_cross(pos_g, pos_h, box, r_edges, z, cosmo,
                    rmin_fit, rmax_fit, njack, nthreads):
    """Delete-one sub-cube jackknife error on the cross-correlation bias b_g."""
    if njack < 2:
        return np.nan
    cg = np.clip(np.floor(np.mod(pos_g, box) / box * njack).astype(int), 0, njack - 1)
    ch = np.clip(np.floor(np.mod(pos_h, box) / box * njack).astype(int), 0, njack - 1)
    idg = (cg[:, 0] * njack + cg[:, 1]) * njack + cg[:, 2]
    idh = (ch[:, 0] * njack + ch[:, 1]) * njack + ch[:, 2]
    samples = []
    for k in range(njack ** 3):
        kg, kh = pos_g[idg != k], pos_h[idh != k]
        if len(kg) < 5 or len(kh) < 50:
            continue
        xgh, rm = xi_cross(kg, kh, box, r_edges, nthreads)
        xhh, _, _ = m.xi_gg(kh, box, r_edges, nthreads)
        b_g, _ = cross_bias(xgh, xhh, rm, z, cosmo, rmin_fit, rmax_fit)
        if np.isfinite(b_g):
            samples.append(b_g)
    s = np.array(samples)
    if s.size < 2:
        return np.nan
    n = s.size
    return float(np.sqrt((n - 1) / n * np.sum((s - s.mean()) ** 2)))


# ----- Mass-definition conversion for the MQG sample --------------------------

def mqg_masses_by_def(mvir_h, conc, z):
    """Median MQG halo mass (Msun/h) in each of vir / 200c / 200m.

    mvir_h : SAGE Mvir in Msun/h (tophat virial).  conc : SAGE concentration
    (c_vir).  Uses colossus changeMassDefinition per galaxy; invalid c fall back
    to the sample median c.
    """
    out = {'vir': np.nan, '200c': np.nan, '200m': np.nan}
    good = np.isfinite(mvir_h) & (mvir_h > 0)
    if not good.any():
        return out
    mv, c = mvir_h[good], conc[good]
    c = np.where(np.isfinite(c) & (c > 1.0) & (c < 40.0), c,
                 np.nanmedian(c[(c > 1.0) & (c < 40.0)]) if np.any((c > 1.0) & (c < 40.0)) else 6.0)
    out['vir'] = float(np.log10(np.median(mv)))
    m200c, _, _ = mass_defs.changeMassDefinition(mv, c, z, 'vir', '200c')
    m200m, _, _ = mass_defs.changeMassDefinition(mv, c, z, 'vir', '200m')
    out['200c'] = float(np.log10(np.median(m200c)))
    out['200m'] = float(np.log10(np.median(m200m)))
    return out


# ----- Plots ------------------------------------------------------------------

def plot_robustness(results, tinker_zs, cosmo, args, out_path):
    """(a) MQG bias auto vs cross vs z;  (b) measured halo bias vs Tinker."""
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2), constrained_layout=True)
    zs = np.array([r['z'] for r in results])
    ax = axes[0]
    ax.errorbar(zs, [r['b_auto'] for r in results], yerr=[r['b_auto_err'] for r in results],
                fmt='o-', color='#d62728', capsize=3, label='autocorrelation (MQG x MQG)')
    ax.errorbar(zs, [r['b_cross'] for r in results], yerr=[r['b_cross_err'] for r in results],
                fmt='s-', color='#1f77b4', capsize=3, label='cross (MQG x halo field)')
    ax.set_xlabel('redshift $z$')
    ax.set_ylabel(r'MQG large-scale bias $b_g$')
    ax.set_title('(a) sparse-sample robustness: cross vs auto', fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    ax.tick_params(which='both', direction='in', top=True, right=True)

    ax = axes[1]
    b_h = np.array([r['b_h'] for r in results])
    b_tink = np.array([r['b_tink_field'] for r in results])
    ax.plot(zs, b_h, 'D-', color='black', label='measured halo bias (xi_hh)')
    ax.plot(zs, b_tink, 'x--', color='0.5',
            label='Tinker+10, abundance-weighted over field')
    ax.set_xlabel('redshift $z$')
    ax.set_ylabel(r'halo bias $b_h$')
    ax.set_title('(b) measured halo-bias baseline vs Tinker+10', fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    ax.tick_params(which='both', direction='in', top=True, right=True)
    fig.suptitle('MQG clustering bias: cross-correlation with the halo-tree field',
                 fontsize=12.5)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


def plot_massdef(results, tinker_zs, cosmo, args, out_path):
    """MQG cross-bias vs halo mass in vir | 200c | 200m, each vs its Tinker curve."""
    from matplotlib import cm
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    zs = np.array([r['z'] for r in results])
    znorm = Normalize(vmin=float(min(tinker_zs)), vmax=float(max(tinker_zs)))
    cmap = cm.get_cmap('viridis')
    mgrid = np.logspace(11, 15, 200)
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 6.2), sharey=True,
                             constrained_layout=True)
    for ax, (mdef, _fld, lab) in zip(axes, MDEFS):
        for zl in sorted(tinker_zs):
            ax.plot(np.log10(mgrid),
                    m.tinker_bias(mgrid / cosmo.h, zl, cosmo, mdef),
                    color=cmap(znorm(zl)), lw=1.4, alpha=0.85)
        x = np.array([r['mqg_mass'][mdef] for r in results])
        y = np.array([r['b_cross'] for r in results])
        ye = np.array([r['b_cross_err'] for r in results])
        ax.errorbar(x, y, yerr=ye, fmt='none', ecolor='0.35', elinewidth=1.2,
                    capsize=3, zorder=4)
        ax.scatter(x, y, c=zs, cmap=cmap, norm=znorm, s=140, marker='o',
                   edgecolors='black', linewidths=1.1, zorder=5)
        ax.set_xlim(11.0, 15.0)
        ax.set_ylim(*args.bias_lim)
        ax.set_xlabel(r'$\log_{10}(M\,/\,[M_\odot/h])$  (MQG median)')
        ax.set_title(lab, fontsize=12)
        ax.tick_params(which='both', direction='in', top=True, right=True)
    axes[0].set_ylabel(r'MQG cross-correlation bias $b_g$')
    cb = fig.colorbar(ScalarMappable(norm=znorm, cmap=cmap), ax=axes,
                      location='right', pad=0.02, shrink=0.9)
    cb.set_label('redshift $z$')
    fig.suptitle('MQG bias vs halo mass under three mass definitions '
                 '(coloured curves = Tinker+10 at each z for that definition)',
                 fontsize=12.5)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


# ----- Halo-mass-binned mode (no number/density knob) -------------------------

def galaxy_jackknife_cross(pos_g, pos_h, xi_hh, box, r_edges, z, cosmo,
                           rmin_fit, rmax_fit, njack, nthreads):
    """Error on b_g by jackknifing the GALAXY sample only, holding the (dense)
    halo field fixed -- its variance is negligible, and the galaxy side
    dominates the error for a sparse bin.  Cheap: no halo autocorr recompute."""
    if njack < 2 or len(pos_g) < njack:
        return np.nan
    cg = np.clip(np.floor(np.mod(pos_g, box) / box * njack).astype(int), 0, njack - 1)
    idg = (cg[:, 0] * njack + cg[:, 1]) * njack + cg[:, 2]
    samples = []
    for k in range(njack ** 3):
        kb = pos_g[idg != k]
        if len(kb) < 3:
            continue
        xgh, rm = xi_cross(kb, pos_h, box, r_edges, nthreads)
        b_g, _ = cross_bias(xgh, xi_hh, rm, z, cosmo, rmin_fit, rmax_fit)
        if np.isfinite(b_g):
            samples.append(b_g)
    s = np.array(samples)
    if s.size < 2:
        return np.nan
    n = s.size
    return float(np.sqrt((n - 1) / n * np.sum((s - s.mean()) ** 2)))


def run_binned(args, hdr, cosmo, redshifts, r_edges):
    """Bin the quiescent galaxies by HOST halo mass and measure the cross-
    correlation bias per bin -- the mass scale is set by the bin, not by a
    chosen count/density.  Quiescence is the only selection (Donnari floor)."""
    hub = hdr['hubble_h']
    edges = np.array(sorted(args.mass_bins) if args.mass_bins else DEFAULT_MASS_BINS,
                     dtype=float)
    print('  mass-bins mode: host-mass edges (log10 Msun/h) = '
          + ', '.join(f'{e:.2f}' for e in edges))
    need = ['StellarMass', 'CentralMvir', 'SfrDisk', 'SfrBulge',
            'Type', 'Posx', 'Posy', 'Posz']
    tinker_zs, rows = [], []
    for z_req in args.redshifts:
        snap = m.snap_nearest_z(redshifts, z_req, hdr['output_snaps'])
        z = float(redshifts[snap])
        if z not in tinker_zs:
            tinker_zs.append(z)
        print(f'\n--- z_req={z_req} -> Snap_{snap} (z={z:.3f}) ---')
        d = m.load_snap(hdr['files'], snap, need, hdr['mass_conv'])
        if not d:
            print('  no galaxy data; skipping.')
            continue
        sm = d['StellarMass']
        ssfr = np.where(sm > 0, (d['SfrDisk'] + d['SfrBulge']) / np.maximum(sm, 1e-30), np.inf)
        floor = m.ssfr_floor(z, hdr['omega_m'], hdr['omega_l'], args.ssfr0)
        q = (ssfr < floor) & (sm > 0)
        pos_q = np.column_stack([d['Posx'][q], d['Posy'][q], d['Posz'][q]])
        host_q = d['CentralMvir'][q] * hub                    # Msun/h
        loghost = np.log10(np.where(host_q > 0, host_q, np.nan))

        halos = read_tree_halos(args.tree_dir, args.tree_name, args.tree_nfiles,
                                snap, args.min_len)
        if halos is None or halos['Pos'].shape[0] < 100:
            print('  no/too-few tree halos; skipping.')
            continue
        n_h_total = halos['Pos'].shape[0]
        if 0 < args.halo_subsample < n_h_total:
            rng = np.random.default_rng(12345 + snap)
            pick = rng.choice(n_h_total, args.halo_subsample, replace=False)
            halos['Pos'] = halos['Pos'][pick]
        pos_h = halos['Pos']
        xi_hh, rm, _ = m.xi_gg(pos_h, hdr['box'], r_edges, args.nthreads)
        _, b_h = cross_bias(np.zeros_like(xi_hh), xi_hh, rm, z, cosmo,
                            args.rmin_fit, args.rmax_fit)
        print(f'  quiescent={int(q.sum())}  N_halo={pos_h.shape[0]} (of {n_h_total})  b_h={b_h:.2f}')

        for i in range(edges.size - 1):
            lo, hi = edges[i], edges[i + 1]
            inbin = (loghost >= lo) & (loghost < hi)
            n = int(inbin.sum())
            if n < args.min_per_bin:
                print(f'    [{lo:.1f},{hi:.1f})  n={n:<5d}  (below --min-per-bin; skipped)')
                continue
            pos_b = pos_q[inbin]
            xgh, _ = xi_cross(pos_b, pos_h, hdr['box'], r_edges, args.nthreads)
            b_g, _ = cross_bias(xgh, xi_hh, rm, z, cosmo, args.rmin_fit, args.rmax_fit)
            be = galaxy_jackknife_cross(pos_b, pos_h, xi_hh, hdr['box'], r_edges, z,
                                        cosmo, args.rmin_fit, args.rmax_fit,
                                        args.njack, args.nthreads)
            med = float(np.median(loghost[inbin]))
            rows.append(dict(z=z, lo=lo, hi=hi, med=med, n=n, b=b_g, be=be, b_h=b_h))
            print(f'    [{lo:.1f},{hi:.1f})  n={n:<5d}  med={med:.2f}  '
                  f'b_g={b_g:.2f}+/-{be:.2f}')

    if not rows:
        sys.exit('No usable bins.')
    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, f'mqg_tree_cross_massbins{args.format}')
    plot_binned(rows, tinker_zs, cosmo, args, out)
    print('\nDone.')


def plot_binned(rows, tinker_zs, cosmo, args, out_path):
    """Cross-correlation bias vs host halo mass, binned; one track per z."""
    from matplotlib import cm
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    znorm = Normalize(vmin=float(min(tinker_zs)), vmax=float(max(tinker_zs)))
    cmap = cm.get_cmap('viridis')
    mgrid = np.logspace(11, 15, 200)
    fig, ax = plt.subplots(figsize=(9.5, 7.0), constrained_layout=True)
    for zl in sorted(tinker_zs):
        ax.plot(np.log10(mgrid), m.tinker_bias(mgrid / cosmo.h, zl, cosmo, 'vir'),
                color=cmap(znorm(zl)), lw=1.4, alpha=0.85, zorder=1)
    for zl in sorted(set(r['z'] for r in rows)):
        rr = sorted((r for r in rows if r['z'] == zl and np.isfinite(r['b'])),
                    key=lambda x: x['med'])
        if not rr:
            continue
        x = [r['med'] for r in rr]
        y = [r['b'] for r in rr]
        ye = [r['be'] for r in rr]
        ax.errorbar(x, y, yerr=ye, fmt='o-', color=cmap(znorm(zl)),
                    ms=8, lw=1.6, mec='black', mew=0.8, capsize=3, zorder=5)
    ax.set_xlim(11.5, 15.0)
    ax.set_ylim(*args.bias_lim)
    ax.set_xlabel(r'$\log_{10}(M_\mathrm{vir}^\mathrm{host}\,/\,[M_\odot/h])$  (bin median)')
    ax.set_ylabel(r'MQG cross-correlation bias $b_g$')
    ax.tick_params(which='both', direction='in', top=True, right=True)
    cb = fig.colorbar(ScalarMappable(norm=znorm, cmap=cmap), ax=ax, pad=0.02)
    cb.set_label('redshift $z$')
    ax.set_title('Quiescent-galaxy bias binned by host halo mass '
                 '(no count/density; curves = Tinker+10)', fontsize=11.5)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


# ----- Driver -----------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default=m.DEFAULT_MODEL)
    p.add_argument('--tree-dir', required=True,
                   help='directory holding the LHaloTree binary files.')
    p.add_argument('--tree-name', required=True,
                   help='tree file basename (files are <name>.0, <name>.1, ...).')
    p.add_argument('--tree-nfiles', type=int, required=True)
    p.add_argument('--redshifts', type=float, nargs='+', default=m.DEFAULT_REDSHIFTS)
    p.add_argument('--min-len', type=int, default=20,
                   help='minimum particle count for a halo to enter the field.')
    p.add_argument('--halo-subsample', type=int, default=100000,
                   help='randomly subsample the halo field to this many objects '
                        'for the correlation measurements (0 = use all). A dense '
                        'subsample still vastly outnumbers the MQGs, so it beats '
                        'their shot noise while keeping pair counts tractable.')
    p.add_argument('--mass-bins', type=float, nargs='*', default=None,
                   metavar='LOGM',
                   help='HALO-MASS-BINNED mode: bin ALL quiescent galaxies by host '
                        'halo mass (log10 Msun/h edges) and measure the cross bias '
                        'per bin -- NO count/density needed. Pass edges, or give the '
                        f'flag alone for defaults {DEFAULT_MASS_BINS}.')
    p.add_argument('--min-per-bin', type=int, default=5,
                   help='skip a host-mass bin with fewer quiescent galaxies than this.')
    # selection (mirrors mqg_clustering_bias)
    p.add_argument('--top-percent', type=float, default=None)
    p.add_argument('--min-logmstar', type=float, default=None)
    p.add_argument('--number-density', type=float, default=None)
    p.add_argument('--match-highz-count', action='store_true')
    p.add_argument('--ssfr0', type=float, default=m.DEFAULT_SSFR0)
    p.add_argument('--sigma-8', type=float, default=m.DEFAULT_SIGMA_8)
    p.add_argument('--n-s', type=float, default=m.DEFAULT_N_S)
    p.add_argument('--omega-b', type=float, default=None)
    p.add_argument('--rmin', type=float, default=m.DEFAULT_RMIN)
    p.add_argument('--rmax', type=float, default=m.DEFAULT_RMAX)
    p.add_argument('--nbins', type=int, default=m.DEFAULT_NBINS)
    p.add_argument('--rmin-fit', type=float, default=m.DEFAULT_RMIN_FIT)
    p.add_argument('--rmax-fit', type=float, default=m.DEFAULT_RMAX_FIT)
    p.add_argument('--njack', type=int, default=m.DEFAULT_NJACK)
    p.add_argument('--nthreads', type=int, default=2)
    p.add_argument('--bias-lim', type=float, nargs=2, default=[0.0, 8.5])
    p.add_argument('--output-dir', default=m.DEFAULT_OUTPUT_DIR)
    p.add_argument('--format', default='.png')
    args = p.parse_args(argv)
    if (args.mass_bins is None and args.number_density is None
            and args.min_logmstar is None and args.top_percent is None
            and not args.match_highz_count):
        args.number_density = m.DEFAULT_NUMBER_DENSITY

    print('=' * 72)
    print('MQG clustering bias -- cross-correlation with the halo-tree field')
    print('=' * 72)
    print(f'  Corrfunc cross available: {HAVE_CORRFUNC_DD}')
    hdr = m.read_header(args.model)
    if hdr is None:
        sys.exit(f'No model_*.hdf5 files in {args.model}')
    redshifts = hdr['redshifts']
    omega_b = (args.omega_b if args.omega_b is not None
               else hdr['baryon_frac'] * hdr['omega_m'])
    cosmo = cosmology.setCosmology('sage', {
        'flat': True, 'H0': hdr['hubble_h'] * 100.0, 'Om0': hdr['omega_m'],
        'Ob0': omega_b, 'sigma8': args.sigma_8, 'ns': args.n_s})
    hub = hdr['hubble_h']
    r_edges = np.logspace(np.log10(args.rmin), np.log10(args.rmax), args.nbins + 1)

    if args.mass_bins is not None:
        run_binned(args, hdr, cosmo, redshifts, r_edges)
        return

    match_n = None
    if args.match_highz_count:
        z_top = float(redshifts[m.snap_nearest_z(redshifts, max(args.redshifts),
                                                 hdr['output_snaps'])])
        snap_top = m.snap_nearest_z(redshifts, max(args.redshifts), hdr['output_snaps'])
        d_top = m.load_snap(hdr['files'], snap_top,
                            ['StellarMass', 'SfrDisk', 'SfrBulge'], hdr['mass_conv'])
        match_n = m.count_quiescent(d_top, z_top, hdr, args.ssfr0) if d_top else 0
        print(f'  match-highz-count -> top-{match_n} at every z')

    need = ['StellarMass', 'Mvir', 'CentralMvir', 'SfrDisk', 'SfrBulge',
            'Type', 'Concentration', 'Posx', 'Posy', 'Posz']
    tinker_zs, results = [], []
    for z_req in args.redshifts:
        snap = m.snap_nearest_z(redshifts, z_req, hdr['output_snaps'])
        z = float(redshifts[snap])
        if z not in tinker_zs:
            tinker_zs.append(z)
        print(f'\n--- z_req={z_req} -> Snap_{snap} (z={z:.3f}) ---')
        d = m.load_snap(hdr['files'], snap, need, hdr['mass_conv'])
        if not d:
            print('  no galaxy data; skipping.')
            continue
        mask, _ = m.select_mqg(d, z, hdr, args.ssfr0,
                               min_logmstar=args.min_logmstar,
                               number_density=args.number_density,
                               top_percent=args.top_percent, top_n=match_n)
        n_mqg = int(mask.sum())
        if n_mqg < 2:
            print(f'  only {n_mqg} MQGs; skipping.')
            continue
        pos_g = np.column_stack([d['Posx'][mask], d['Posy'][mask], d['Posz'][mask]])
        mvir_h = d['Mvir'][mask] * hub                       # Msun/h
        conc = d['Concentration'][mask]

        halos = read_tree_halos(args.tree_dir, args.tree_name, args.tree_nfiles,
                                snap, args.min_len)
        if halos is None or halos['Pos'].shape[0] < 100:
            print('  no/too-few tree halos at this snap; skipping.')
            continue
        n_h_total = halos['Pos'].shape[0]
        if 0 < args.halo_subsample < n_h_total:
            rng = np.random.default_rng(12345 + snap)     # reproducible per snap
            pick = rng.choice(n_h_total, args.halo_subsample, replace=False)
            for k in ('Pos', 'vir', '200c', '200m'):
                halos[k] = halos[k][pick]
        pos_h = halos['Pos']
        n_h = pos_h.shape[0]

        xi_gg, rm, _ = m.xi_gg(pos_g, hdr['box'], r_edges, args.nthreads)
        xi_hh, _, _ = m.xi_gg(pos_h, hdr['box'], r_edges, args.nthreads)
        xi_gh, _ = xi_cross(pos_g, pos_h, hdr['box'], r_edges, args.nthreads)

        b_auto, _, _ = m.bias_from_ratio(xi_gg, rm, z, cosmo, args.rmin_fit, args.rmax_fit)
        b_cross, b_h = cross_bias(xi_gh, xi_hh, rm, z, cosmo, args.rmin_fit, args.rmax_fit)
        b_auto_err = m.jackknife_bias(pos_g, hdr['box'], r_edges, z, cosmo,
                                      args.rmin_fit, args.rmax_fit, args.njack, args.nthreads)
        b_cross_err = jackknife_cross(pos_g, pos_h, hdr['box'], r_edges, z, cosmo,
                                      args.rmin_fit, args.rmax_fit, args.njack, args.nthreads)
        mqg_mass = mqg_masses_by_def(mvir_h, conc, z)
        fmass = halos['vir'][halos['vir'] > 0]
        logMh_field = float(np.log10(np.median(fmass)))
        # abundance-weighted mean Tinker bias over the actual field masses -- the
        # fair baseline for b_h (bias is steep in M, so <b(M)> != b(<M>)).
        b_tink_field = float(np.mean(m.tinker_bias(fmass / hub, z, cosmo, 'vir')))

        results.append(dict(z=z, snap=snap, n_mqg=n_mqg, n_halo=n_h,
                            b_auto=b_auto, b_auto_err=b_auto_err,
                            b_cross=b_cross, b_cross_err=b_cross_err, b_h=b_h,
                            logMh_field=logMh_field, b_tink_field=b_tink_field,
                            mqg_mass=mqg_mass))
        print(f'  N_MQG={n_mqg}  N_halo={n_h} (of {n_h_total})  '
              f'b_auto={b_auto:.2f}+/-{b_auto_err:.2f}  '
              f'b_cross={b_cross:.2f}+/-{b_cross_err:.2f}  b_h={b_h:.2f}')
        print(f'  MQG median mass  vir={mqg_mass["vir"]:.2f}  '
              f'200c={mqg_mass["200c"]:.2f}  200m={mqg_mass["200m"]:.2f}')

    if not results:
        sys.exit('No usable redshifts.')
    os.makedirs(args.output_dir, exist_ok=True)
    fmt = args.format
    print('\nSummary:')
    print(f'  {"z":>5} {"N_MQG":>7} {"N_halo":>9} {"b_auto":>10} {"b_cross":>10} '
          f'{"b_h":>6} {"vir":>6} {"200c":>6} {"200m":>6}')
    for r in sorted(results, key=lambda x: x['z']):
        print(f'  {r["z"]:>5.2f} {r["n_mqg"]:>7d} {r["n_halo"]:>9d} '
              f'{r["b_auto"]:>5.2f}+/-{r["b_auto_err"]:<4.2f} '
              f'{r["b_cross"]:>5.2f}+/-{r["b_cross_err"]:<4.2f} {r["b_h"]:>6.2f} '
              f'{r["mqg_mass"]["vir"]:>6.2f} {r["mqg_mass"]["200c"]:>6.2f} '
              f'{r["mqg_mass"]["200m"]:>6.2f}')
    plot_robustness(results, tinker_zs, cosmo, args,
                    os.path.join(args.output_dir, f'mqg_tree_cross_robustness{fmt}'))
    plot_massdef(results, tinker_zs, cosmo, args,
                 os.path.join(args.output_dir, f'mqg_tree_cross_massdef{fmt}'))
    print('\nDone.')


if __name__ == '__main__':
    main()

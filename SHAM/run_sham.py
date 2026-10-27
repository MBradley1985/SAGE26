#!/usr/bin/env python3
"""
run_sham.py
===========
Subhalo abundance matching (SHAM) on the Millennium merger trees.

For each requested snapshot:
  1. read every trees_063.* file and build each (sub)halo's Vpeak
     (see sham_trees.py);
  2. rank all (sub)haloes by Vpeak (highest first) -> cumulative number density
     n(>Vpeak) = rank / V_sim;
  3. assign M_* and M_BH by matching n(>Vpeak) to the cumulative number density
     of the benchmark SMF and BHMF (see sham_benchmarks.py). The benchmark is
     first DE-convolved with the chosen log-normal scatter so that, after the
     scatter is added in step 4, the model SMF/BHMF reproduce the benchmarks;
  4. add independent (or correlated, --rho) Gaussian scatter in log M_* and
     log M_BH at fixed Vpeak.
Results go to SHAM/output/sham_snapNN.npz for plot_sham_fbh.py.

Typical use (full box, 8 processes):
    python SHAM/run_sham.py --z 3 5 6 --nproc 8
Quick test on a few files:
    python SHAM/run_sham.py --z 5 --max-files 2
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sham_trees as st
import sham_benchmarks as bm

REPO = bm.REPO_ROOT


def get_catalogs(files, snaps, cache_dir, n_files_total, nproc, rebuild=False):
    """Halo catalogues per snapshot, cached on disk (the tree sweep is the slow part)."""
    os.makedirs(cache_dir, exist_ok=True)
    cats, todo = {}, []
    for s in snaps:
        f = os.path.join(cache_dir, f'halos_snap{s:02d}.npz')
        if os.path.exists(f) and not rebuild:
            d = np.load(f)
            if int(d['n_files']) == len(files):
                cats[s] = {k: d[k] for k in ('vpeak', 'lpeak', 'vmax', 'length', 'central')}
                continue
        todo.append(s)
    if todo:
        print(f'Reading {len(files)} tree files for snapshots {todo} ...')
        t0 = time.time()
        new = st.build_halo_catalogs(files, todo, nproc=nproc)
        print(f'  done in {time.time() - t0:.0f}s')
        for s in todo:
            np.savez(os.path.join(cache_dir, f'halos_snap{s:02d}.npz'),
                     n_files=len(files), **new[s])
            cats[s] = new[s]
    return cats


def completeness_limit(prop, lpeak, lmin, frac=0.9, dlog=0.02):
    """
    Lowest property value above which >= `frac` of the catalogued haloes have
    a peak particle count >= lmin (walk down from the top, stop at the first
    bin that falls short). Below it the halo list is resolution-incomplete.
    """
    lp = np.log10(prop[prop > 0])
    ll = lpeak[prop > 0]
    edges = np.arange(np.floor(lp.min() / dlog) * dlog, lp.max() + dlog, dlog)
    tot, _ = np.histogram(lp, edges)
    good, _ = np.histogram(lp[ll >= lmin], edges)
    lim = edges[0]
    for i in range(len(tot) - 1, -1, -1):
        if tot[i] > 0 and good[i] / tot[i] < frac:
            lim = edges[i + 1]
            break
    return 10.0 ** lim


def match(n_halo, logm, phi_int, logm_floor):
    """
    Map halo number densities n(>prop) to log mass by inverting the cumulative
    benchmark N(>M). Returns NaN for haloes whose n lies beyond the benchmark's
    reliable range (below logm_floor, or above the grid top).
    """
    N = bm.cumulative_above(logm, phi_int)
    out = np.interp(n_halo, N[::-1], logm[::-1], left=np.nan, right=np.nan)
    out[out < logm_floor] = np.nan
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--trees-dir', default=os.path.join(REPO, '/fred/oz004/simulations/Millennium/'))
    p.add_argument('--tree-name', default='trees_063')
    p.add_argument('--a-list', default=os.path.join(REPO, '/fred/oz004/simulations/Millennium/millennium.alist'))
    p.add_argument('--n-files-total', type=int, default=512,
                   help='number of files the full simulation is split into (volume = '
                        'box^3 * n_found / n_total); default 512')
    p.add_argument('--max-files', type=int, default=None, help='only use the first N files (testing)')
    p.add_argument('--box-size', type=float, default=500.0, help='Mpc/h')
    p.add_argument('--hubble', type=float, default=0.73)
    p.add_argument('--snaps', type=int, nargs='+', default=[27],
                   help='snapshot(s) (Millennium numbering; 27 -> z=3.06, 19 -> z=5.72, '
                        '17 -> z=6.71); default 27, same as bh_lrd_analysis.py')
    p.add_argument('--z', type=float, nargs='+', default=None,
                   help='redshift(s); the nearest snapshot to each is used instead of --snaps')
    p.add_argument('--property', choices=['vpeak', 'vmax', 'lpeak'], default='vpeak',
                   help='ranking property (vpeak = default; vmax = current; lpeak = '
                        'peak particle count, a Mpeak proxy)')
    p.add_argument('--sigma-star', type=float, default=0.15, help='scatter in log M* [dex]')
    p.add_argument('--sigma-bh', type=float, default=0.30, help='scatter in log M_BH [dex]')
    p.add_argument('--rho', type=float, default=0.0,
                   help='correlation of the two scatter terms (default 0 = independent)')
    p.add_argument('--lmin', type=int, default=50,
                   help='min peak particle count defining the resolution limit (default 50)')
    p.add_argument('--bhmf-variant', choices=['best', 'lo', 'hi'], default='best',
                   help="TRINITY BHMF: 'best', or its 16th ('lo') / 84th ('hi') percentile")
    p.add_argument('--z-tol', type=float, default=None,
                   help='|dz| of SMF data pooled for the fit (default max(0.3, 0.15(1+z)))')
    p.add_argument('--nproc', type=int, default=1)
    p.add_argument('--rebuild-cache', action='store_true')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--out-dir', default=os.path.join(REPO, 'SHAM/output'))
    a = p.parse_args()

    files = st.list_tree_files(a.trees_dir, a.tree_name)
    if not files:
        sys.exit(f'No {a.tree_name}.* files in {a.trees_dir}')
    if a.max_files:
        files = files[:a.max_files]
    zs = st.read_snapshot_redshifts(a.a_list)
    if a.z:
        a.snaps = sorted({int(np.argmin(np.abs(zs - zz))) for zz in a.z})
        print('Nearest snapshots:', ', '.join(f'{s} (z={zs[s]:.2f})' for s in a.snaps))
    vol = (a.box_size / a.hubble) ** 3 * len(files) / a.n_files_total   # Mpc^3
    print(f'{len(files)}/{a.n_files_total} tree files -> V = {vol:.4e} Mpc^3 '
          f'(box {a.box_size} Mpc/h, h={a.hubble})')

    cache_dir = os.path.join(a.out_dir, 'cache' if not a.max_files else f'cache_first{a.max_files}')
    cats = get_catalogs(files, a.snaps, cache_dir, a.n_files_total, a.nproc, a.rebuild_cache)

    obs = bm.load_smf_compilation()
    tabs = bm.load_trinity_bhmf()
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)

    for s in a.snaps:
        z = float(zs[s])
        c = cats[s]
        prop = {'vpeak': c['vpeak'], 'vmax': c['vmax'], 'lpeak': c['lpeak'].astype(float)}[a.property]
        print(f'\n=== snapshot {s}  z = {z:.3f}  ({len(prop):,} (sub)haloes) ===')

        # --- resolution-complete sample, ranked by the chosen property
        p_lim = completeness_limit(prop, c['lpeak'], a.lmin)
        sel = np.flatnonzero(prop >= p_lim)
        jitter = prop[sel] * (1 + 1e-6 * rng.standard_normal(sel.size))   # break ties
        order = np.argsort(-jitter)
        idx = sel[order]                                    # highest property first
        n_halo = (np.arange(idx.size) + 0.5) / vol          # n(>prop) [Mpc^-3]
        print(f'  complete above {a.property} = {p_lim:.4g}  -> {idx.size:,} haloes '
              f'(n = {n_halo[-1]:.3e} Mpc^-3)')

        # --- benchmarks, de-scattered
        g_s, phi_s, info = bm.smf_benchmark(z, obs=obs, z_tol=a.z_tol)
        phi_s_int = bm.deconvolve(g_s, phi_s, a.sigma_star)
        g_b, phi_b, binfo = bm.bhmf_benchmark(z, variant=a.bhmf_variant, tabs=tabs,
                                              grid=np.arange(4.0, 10.5 + 1e-9, 0.02))
        phi_b_int = bm.deconvolve(g_b, phi_b, a.sigma_bh)
        floor_s = g_s[0]                       # Schechter extrapolated below the data
        data_min = info['logm_min']
        floor_b = g_b[0]
        print(f'  SMF fit: {info["n_points"]} points from {", ".join(info["datasets"])} '
              f'(z_obs {info["z_obs"]}), {"double" if info["double"] else "single"} Schechter, '
              f'chi2_red = {info["chi2_red"]:.2f}; data reach down to log M* = {data_min:.2f} (fit extrapolated below)')
        print(f'  BHMF: TRINITY z-tables {binfo["z_tables"]}, w = {binfo["weight"]:.2f}; '
              f'variant = {a.bhmf_variant}')

        ms_int = match(n_halo, g_s, phi_s_int, floor_s)
        mb_int = match(n_halo, g_b, phi_b_int, floor_b)
        z1 = rng.standard_normal(idx.size)
        z2 = a.rho * z1 + np.sqrt(1 - a.rho ** 2) * rng.standard_normal(idx.size)
        ms = ms_int + a.sigma_star * z1
        mb = mb_int + a.sigma_bh * z2

        np.savez(os.path.join(a.out_dir, f'sham_snap{s:02d}.npz'),
                 snap=s, z=z, volume_mpc3=vol, n_files=len(files), property=a.property,
                 p_lim=p_lim, sigma_star=a.sigma_star, sigma_bh=a.sigma_bh, rho=a.rho,
                 bhmf_variant=a.bhmf_variant,
                 prop=prop[idx], central=c['central'][idx], n_halo=n_halo,
                 logmstar=ms, logmbh=mb, logmstar_int=ms_int, logmbh_int=mb_int,
                 smf_logm=g_s, smf_phi=phi_s, smf_phi_int=phi_s_int, smf_floor=floor_s, smf_data_min=data_min,
                 bhmf_logm=g_b, bhmf_phi=phi_b, bhmf_phi_int=phi_b_int, bhmf_floor=floor_b,
                 smf_obs_x=info['x'], smf_obs_y=info['y'], smf_obs_e=info['e'])
        both = np.isfinite(ms) & np.isfinite(mb)
        print(f'  matched: {np.isfinite(ms_int).sum():,} with M*, {np.isfinite(mb_int).sum():,} '
              f'with M_BH, {both.sum():,} with both')
        print(f'  saved {os.path.join(a.out_dir, f"sham_snap{s:02d}.npz")}')


if __name__ == '__main__':
    main()

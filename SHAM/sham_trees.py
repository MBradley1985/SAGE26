#!/usr/bin/env python3
"""
sham_trees.py
=============
Reads the Millennium LHalo binary merger trees (trees_063.*) and builds, for
every (sub)halo at the requested snapshots, the abundance-matching property

    Vpeak  = max over the halo's MAIN-PROGENITOR BRANCH (up to and including
             that snapshot) of Vmax,
    Lpeak  = same thing for the particle count Len (a mass-proxy "Mpeak"
             used only as a resolution diagnostic / alternative property).

The main branch is followed through FirstProgenitor, which in LHalo trees is by
construction the progenitor with the most massive history (De Lucia & Blaizot
2007), i.e. the standard main-progenitor definition.

Each tree file is independent, so files are processed in parallel and the
per-snapshot halo lists are concatenated afterwards (see build_halo_catalogs).

LHalo on-disk layout (same struct as src/core_simulation.h, 104 bytes/halo):
    int32 Ntrees; int32 TotNHalos; int32 TreeNHalos[Ntrees]; halo_data[TotNHalos]
"""

import glob
import os
import re
from multiprocessing import Pool

import numpy as np

HALO_DTYPE = np.dtype([
    ('Descendant', '<i4'), ('FirstProgenitor', '<i4'), ('NextProgenitor', '<i4'),
    ('FirstHaloInFOFgroup', '<i4'), ('NextHaloInFOFgroup', '<i4'),
    ('Len', '<i4'), ('M_Mean200', '<f4'), ('Mvir', '<f4'), ('M_TopHat', '<f4'),
    ('Pos', '<f4', 3), ('Vel', '<f4', 3), ('VelDisp', '<f4'), ('Vmax', '<f4'),
    ('Spin', '<f4', 3), ('MostBoundID', '<i8'),
    ('SnapNum', '<i4'), ('FileNr', '<i4'), ('SubhaloIndex', '<i4'),
    ('SubHalfMass', '<f4'),
])
assert HALO_DTYPE.itemsize == 104, 'LHalo struct size mismatch'


def list_tree_files(trees_dir, tree_name):
    """Tree files trees_063.<n>, sorted numerically."""
    files = glob.glob(os.path.join(trees_dir, f'{tree_name}.*'))
    files = [f for f in files if re.search(r'\.\d+$', f)]
    return sorted(files, key=lambda f: int(f.rsplit('.', 1)[1]))


def read_snapshot_redshifts(a_list_file):
    """z = 1/a - 1 for every snapshot, from the Millennium expansion-factor list."""
    return 1.0 / np.loadtxt(a_list_file) - 1.0


def read_tree_file(path):
    """Return the full halo array of one tree file plus the per-tree halo counts."""
    with open(path, 'rb') as f:
        ntrees, tot = np.fromfile(f, dtype='<i4', count=2)
        tree_n = np.fromfile(f, dtype='<i4', count=ntrees)
        halos = np.fromfile(f, dtype=HALO_DTYPE, count=tot)
    if halos.shape[0] != tot:
        raise IOError(f'{path}: expected {tot} halos, read {halos.shape[0]}')
    return halos, tree_n


def compute_peaks(halos, tree_n):
    """
    Vpeak and Lpeak for every halo in one file.

    Progenitors always sit at strictly lower SnapNum than their descendant, so a
    single sweep over snapshots in increasing order is enough:
        peak[h] = max(own value, peak[FirstProgenitor[h]]).
    FirstProgenitor is tree-local, so it is shifted by each tree's start index.
    """
    n = halos.shape[0]
    starts = np.concatenate([[0], np.cumsum(tree_n)[:-1]]).astype(np.int64)
    fp = halos['FirstProgenitor'].astype(np.int64)
    fp = np.where(fp >= 0, fp + np.repeat(starts, tree_n), -1)

    vmax = halos['Vmax'].astype(np.float32)
    length = halos['Len'].astype(np.int32)
    vpeak = vmax.copy()
    lpeak = length.copy()

    snap = halos['SnapNum']
    order = np.argsort(snap, kind='stable')
    bounds = np.searchsorted(snap[order], np.arange(snap.max() + 2))
    for s in range(snap.max() + 1):
        idx = order[bounds[s]:bounds[s + 1]]
        if idx.size == 0:
            continue
        p = fp[idx]
        has = p >= 0
        ii, pp = idx[has], p[has]
        vpeak[ii] = np.maximum(vpeak[ii], vpeak[pp])
        lpeak[ii] = np.maximum(lpeak[ii], lpeak[pp])
    return vpeak, lpeak


def _process_file(args):
    path, snaps = args
    halos, tree_n = read_tree_file(path)
    vpeak, lpeak = compute_peaks(halos, tree_n)
    starts = np.repeat(np.concatenate([[0], np.cumsum(tree_n)[:-1]]), tree_n)
    local = np.arange(halos.shape[0]) - starts
    is_central = halos['FirstHaloInFOFgroup'] == local

    out = {}
    for s in snaps:
        m = halos['SnapNum'] == s
        out[s] = dict(vpeak=vpeak[m], lpeak=lpeak[m], vmax=halos['Vmax'][m],
                      length=halos['Len'][m], central=is_central[m])
    return out


def build_halo_catalogs(files, snaps, nproc=1):
    """
    Concatenate the per-file (sub)halo lists at each snapshot in `snaps`.
    Returns {snap: dict(vpeak, lpeak, vmax, length, central)} (numpy arrays).
    """
    jobs = [(f, list(snaps)) for f in files]
    if nproc > 1:
        with Pool(nproc) as pool:
            parts = []
            for i, r in enumerate(pool.imap(_process_file, jobs), 1):
                parts.append(r)
                print(f'  trees: {i}/{len(jobs)} files', end='\r', flush=True)
    else:
        parts = []
        for i, j in enumerate(jobs, 1):
            parts.append(_process_file(j))
            print(f'  trees: {i}/{len(jobs)} files', end='\r', flush=True)
    print()
    cat = {}
    for s in snaps:
        cat[s] = {k: np.concatenate([p[s][k] for p in parts])
                  for k in ('vpeak', 'lpeak', 'vmax', 'length', 'central')}
    return cat

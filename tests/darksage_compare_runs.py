#!/usr/bin/env python
"""Field-by-field comparison of two SAGE26 HDF5 runs at a few snapshots.

Usage: darksage_compare_runs.py DIR_A DIR_B [SNAP ...]
"""
import glob
import sys

import h5py
import numpy as np


def main():
    da, db = sys.argv[1], sys.argv[2]
    snaps = [int(s) for s in sys.argv[3:]] or [63, 40, 20]
    fa, fb = sorted(glob.glob(f"{da}/model_*.hdf5")), sorted(glob.glob(f"{db}/model_*.hdf5"))
    n_diff = 0
    for snap in snaps:
        key = f"Snap_{snap}"
        fields = sorted(set(h5py.File(fa[0], "r")[key].keys()) & set(h5py.File(fb[0], "r")[key].keys()))
        for fld in fields:
            a = np.concatenate([h5py.File(f, "r")[f"{key}/{fld}"][:] for f in fa])
            b = np.concatenate([h5py.File(f, "r")[f"{key}/{fld}"][:] for f in fb])
            if a.shape != b.shape or not np.array_equal(a, b, equal_nan=True):
                n_diff += 1
                print(f"{key}/{fld}: differs")
    only = set(h5py.File(fa[0], "r")["Snap_63"].keys()) ^ set(h5py.File(fb[0], "r")["Snap_63"].keys())
    print(f"fields in one run only: {sorted(only)}")
    print("IDENTICAL" if n_diff == 0 else f"{n_diff} field(s) differ")


if __name__ == "__main__":
    main()

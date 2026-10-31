#!/usr/bin/env python
"""Annuli against SAGE26's bulk quantities in a DarkSAGEMode run.

Checks, at each snapshot given: sum(DiscGas) = ColdGas, sum(DiscGasMetals) = MetalsColdGas,
sum(DiscStars) = StellarMass - BulgeMass, sum(DiscStarsMetals) = MetalsStellarMass -
MetalsBulgeMass, sum(DiscSFR) = SfrDisk.  Prints the worst relative mismatch (relative to
the galaxy's ColdGas + StellarMass, or to SfrDisk for the rates) and the fraction of
galaxies above 1e-4.

Usage: darksage_annuli_sums.py DIR [SNAP ...]
"""
import glob
import sys

import h5py
import numpy as np


def main():
    d = sys.argv[1]
    snaps = [int(s) for s in sys.argv[2:]] or [63, 40, 20]
    files = sorted(glob.glob(f"{d}/model_*.hdf5"))
    for snap in snaps:
        def get(f):
            return np.concatenate([h5py.File(fn, "r")[f"Snap_{snap}/{f}"][:] for fn in files]).astype(float)
        n = len(get("ColdGas"))
        if n == 0:
            continue
        scale = get("ColdGas") + get("StellarMass")
        scale[scale <= 0] = 1.0
        checks = {
            "gas": (get("DiscGas").reshape(n, -1).sum(1), get("ColdGas"), scale),
            "gas metals": (get("DiscGasMetals").reshape(n, -1).sum(1), get("MetalsColdGas"), scale),
            "disc stars": (get("DiscStars").reshape(n, -1).sum(1), get("StellarMass") - get("BulgeMass"), scale),
            "disc star metals": (get("DiscStarsMetals").reshape(n, -1).sum(1),
                                 get("MetalsStellarMass") - get("MetalsBulgeMass"), scale),
        }
        sfr = get("SfrDisk")
        checks["SFR"] = (get("DiscSFR").reshape(n, -1).sum(1), sfr, np.where(sfr > 0, sfr, 1.0))
        line = [f"Snap_{snap} (N={n}):"]
        for name, (ann, bulk, sc) in checks.items():
            rel = np.abs(ann - bulk) / sc
            line.append(f"{name} max {rel.max():.1e} (>1e-4: {np.mean(rel > 1e-4):.1%})")
        print("  ".join(line))


if __name__ == "__main__":
    main()

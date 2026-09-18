"""Reduce raw orphan-event dumps to the summary statistics used by the figures.

Input : data/orphan_events_<run>.csv  (written by the instrumented SAGE build,
        see scripts/core_build_model.instrumented.c -- one row per orphan
        merger/disruption event)
Output: data/orphan_analysis.json

Run from anywhere: python3 scripts/analyse_orphan_events.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")
ALIST = "/Users/mbradley/Documents/PhD/SAGE26/input/millennium/trees/millennium.a_list"

HUBBLE_H = 0.73
# code time unit -> Gyr:  UnitLength_in_cm / UnitVelocity_in_cm_per_s, /h
UNIT_T_GYR = 3.08568e24 / 1e5 / 3.15576e16 / HUBBLE_H  # ~1339 Gyr

RUNS = [("fiducial", "orphan_events_fiducial.csv"),
        ("vanilla", "orphan_events_vanilla.csv")]


def reduce_run(path, zsnap):
    d = np.genfromtxt(path, delimiter=",", names=True)
    snap = d["snap"].astype(int)
    origin = d["origin"].astype(int)   # 0 = was Type 0 last snapshot, 1 = was Type 1
    fate = d["fate"].astype(int)       # 1 = merger, 0 = disrupted to ICS
    ms = d["mstar"] * 1e10 / HUBBLE_H
    cold = d["coldgas"] * 1e10 / HUBBLE_H
    hot = (d["hotgas"] + d["cgmgas"]) * 1e10 / HUBBLE_H
    # MergTime left on the clock at the moment the subhalo was lost
    tclock = (d["mergtime"] + d["dT"]) * UNIT_T_GYR

    M, D = fate == 1, fate == 0
    r = dict(
        n=len(d), n_merged=int(M.sum()), n_disr=int(D.sum()),
        t0_merged=int((M & (origin == 0)).sum()), t0_disr=int((D & (origin == 0)).sum()),
        t1_merged=int((M & (origin == 1)).sum()), t1_disr=int((D & (origin == 1)).sum()),
        mstar_merged=float(ms[M].sum()), mstar_disr=float(ms[D].sum()),
        cold_merged=float(cold[M].sum()), cold_disr=float(cold[D].sum()),
        hot_merged=float(hot[M].sum()), hot_disr=float(hot[D].sum()),
        med_ms_merged=float(np.median(ms[M & (ms > 0)])),
        med_ms_disr=float(np.median(ms[D & (ms > 0)])),
    )

    bins = np.arange(5.6, 11.9, 0.3)
    r["mass_bins"] = bins.tolist()
    for key, sel in (("mh_merged", M), ("mh_disr", D)):
        v = ms[sel & (ms > 0)]
        r[key] = np.histogram(np.log10(v), bins=bins)[0].tolist()

    r["per_snap"] = [
        dict(z=round(float(zsnap[s]), 3),
             merged=int((( snap == s) & M).sum()), disr=int(((snap == s) & D).sum()),
             ms_merged=float(ms[(snap == s) & M].sum()),
             ms_disr=float(ms[(snap == s) & D].sum()))
        for s in sorted(set(snap.tolist()))
    ]

    tb = np.array([0, .25, .5, 1, 2, 3, 4, 6, 8, 10, 13.8, 1e9])
    r["tclock_bins"] = tb[:-1].tolist()
    r["tclock_hist"] = np.histogram(tclock[D], bins=tb)[0].tolist()
    r["tclock_pct"] = [float(np.percentile(tclock[D], p)) for p in (10, 25, 50, 75, 90)]

    lm = np.log10(np.where(ms > 0, ms, 1e-30))
    mb = np.arange(6.0, 11.5, 0.5)
    r["fate_vs_mass"] = []
    for lo, hi in zip(mb[:-1], mb[1:]):
        sel = (lm >= lo) & (lm < hi) & (ms > 0)
        n = int(sel.sum())
        r["fate_vs_mass"].append(
            dict(lo=float(lo), n=n, fdisr=float((sel & D).sum() / n) if n > 20 else None))
    return r


def main():
    alist = np.loadtxt(ALIST)
    zsnap = {i: 1.0 / a - 1.0 for i, a in enumerate(alist)}

    out = {"unit_t_gyr": UNIT_T_GYR}
    for tag, fname in RUNS:
        path = os.path.join(DATA, fname)
        if not os.path.exists(path):
            print(f"skipping {tag}: {fname} not found")
            continue
        out[tag] = reduce_run(path, zsnap)
        print(f"{tag}: {out[tag]['n']} events, "
              f"{out[tag]['n_merged']} merged / {out[tag]['n_disr']} disrupted")

    dest = os.path.join(DATA, "orphan_analysis.json")
    json.dump(out, open(dest, "w"), indent=1)
    print("wrote", dest)


if __name__ == "__main__":
    main()

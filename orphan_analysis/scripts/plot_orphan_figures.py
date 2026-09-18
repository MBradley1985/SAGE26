"""Figures for the SAGE26 orphan (Type 2) galaxy analysis.

Input : data/orphan_analysis.json   (from analyse_orphan_events.py)
Output: figures/fig1_orphan_fates.png
        figures/fig2_orphan_timing.png

Run from anywhere: python3 scripts/plot_orphan_figures.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")
FIGS = os.path.join(HERE, "figures")

C_MERGE = "#2a6fb0"   # deal_with_galaxy_merger
C_DISR = "#d1622a"    # disrupt_satellite_to_ICS

plt.rcParams.update({
    "font.size": 9, "axes.linewidth": 0.8,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "figure.dpi": 130, "savefig.dpi": 130,
    "axes.labelsize": 10, "legend.frameon": False,
})


def fig_fates(f):
    fig, ax = plt.subplots(2, 2, figsize=(10, 7.2))

    a = ax[0, 0]
    m = [f["t0_merged"], f["t1_merged"]]
    dd = [f["t0_disr"], f["t1_disr"]]
    y = np.arange(2)
    a.barh(y, m, color=C_MERGE, height=0.55, label="Merger (MergTime $\\leq$ 0)")
    a.barh(y, dd, left=m, color=C_DISR, height=0.55, label="Disrupted to ICS (MergTime > 0)")
    for i in range(2):
        if m[i] > 1500:
            a.text(m[i] / 2, y[i], f"{m[i]:,}", ha="center", va="center", color="w")
        else:
            a.text(m[i] + 400, y[i] + 0.33, f"{m[i]:,}", ha="left", va="center",
                   color=C_MERGE, fontsize=8.5)
        if dd[i] > 1500:
            a.text(m[i] + dd[i] / 2, y[i], f"{dd[i]:,}", ha="center", va="center", color="w")
    a.set_yticks(y)
    a.set_yticklabels(["Type 0 $\\rightarrow$ 2\n(central, halo lost)",
                       "Type 1 $\\rightarrow$ 2\n(satellite, subhalo lost)"])
    a.invert_yaxis()
    a.set_xlabel("number of orphan events")
    a.legend(loc="lower right", fontsize=8.5)
    a.set_title("(a) How an orphan is made vs. how it dies", loc="left", fontsize=10.5)

    b = ax[0, 1]
    x = np.arange(2)
    w = 0.38
    nums = np.array([f["n_merged"], f["n_disr"]]) / f["n"] * 100
    mass = np.array([f["mstar_merged"], f["mstar_disr"]])
    mtot = mass.sum()
    mass = mass / mtot * 100
    b.bar(x - w / 2, [nums[0], mass[0]], w, color=C_MERGE, label="Merger")
    b.bar(x + w / 2, [nums[1], mass[1]], w, color=C_DISR, label="ICS disruption")
    for xx, vv in zip([-w / 2, 1 - w / 2], [nums[0], mass[0]]):
        b.text(xx, vv + 1.5, f"{vv:.0f}%", ha="center", color=C_MERGE)
    for xx, vv in zip([w / 2, 1 + w / 2], [nums[1], mass[1]]):
        b.text(xx, vv + 1.5, f"{vv:.0f}%", ha="center", color=C_DISR)
    b.set_xticks(x)
    b.set_xticklabels([f"by event count\n({f['n']:,} events)",
                       f"by stellar mass\n({mtot/1e13:.1f}$\\times$10$^{{13}}$ M$_\\odot$)"])
    b.set_ylabel("share of orphan channel [%]")
    b.set_ylim(0, 92)
    b.legend(fontsize=8.5)
    b.set_title("(b) Orphan stars mostly end up in the ICS, not in remnants",
                loc="left", fontsize=10.5)

    c = ax[1, 0]
    bins = np.array(f["mass_bins"])
    ctr = 0.5 * (bins[1:] + bins[:-1])
    for h, col, lab in ((f["mh_merged"], C_MERGE, "Merger"),
                        (f["mh_disr"], C_DISR, "ICS disruption")):
        c.step(ctr, h, where="mid", color=col, lw=1.8, label=lab)
        c.fill_between(ctr, h, step="mid", color=col, alpha=0.15)
    c.axvline(np.log10(f["med_ms_merged"]), color=C_MERGE, ls=":", lw=1.2)
    c.axvline(np.log10(f["med_ms_disr"]), color=C_DISR, ls=":", lw=1.2)
    c.set_xlabel(r"log$_{10}$ ( M$_\star$ / M$_\odot$ ) at the moment of death")
    c.set_ylabel("orphan events per 0.3 dex")
    c.set_xlim(5.6, 11.3)
    c.legend(fontsize=8.5)
    c.text(np.log10(f["med_ms_disr"]) + 0.08, 0.93 * max(f["mh_disr"]),
           "medians\n%.1f$\\times$10$^{7}$ vs %.1f$\\times$10$^{8}$ M$_\\odot$"
           % (f["med_ms_merged"] / 1e7, f["med_ms_disr"] / 1e8),
           fontsize=8, color="0.3")
    c.set_title("(c) Disrupted orphans are ~15$\\times$ more massive", loc="left", fontsize=10.5)

    e = ax[1, 1]
    pts = [p for p in f["fate_vs_mass"] if p["fdisr"] is not None]
    lo = np.array([p["lo"] for p in pts])
    fr = np.array([p["fdisr"] for p in pts])
    nn = np.array([p["n"] for p in pts])
    e.errorbar(lo + 0.25, fr * 100, yerr=np.sqrt(fr * (1 - fr) / nn) * 100,
               color=C_DISR, marker="o", ms=5, lw=1.8, capsize=2.5)
    e.axhline(50, color="0.6", lw=0.8, ls="--")
    e.set_xlabel(r"log$_{10}$ ( M$_\star$ / M$_\odot$ )")
    e.set_ylabel("fraction disrupted to ICS [%]")
    e.set_ylim(0, 100)
    e.set_title("(d) Above 10$^{8}$ M$_\\odot$, orphans are shredded, not merged",
                loc="left", fontsize=10.5)

    fig.suptitle("SAGE26 orphan (Type 2) galaxies — mini-Millennium, fiducial run "
                 f"({f['n']:,} orphan events, 64 snapshots)", fontsize=11, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out = os.path.join(FIGS, "fig1_orphan_fates.png")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


def fig_timing(f, v):
    fig, ax = plt.subplots(2, 2, figsize=(10, 7.4))

    a = ax[0, 0]
    z = np.array([p["z"] for p in f["per_snap"]])
    nm = np.array([p["merged"] for p in f["per_snap"]])
    nd = np.array([p["disr"] for p in f["per_snap"]])
    a.plot(z, nm, color=C_MERGE, lw=1.8, label="Merger")
    a.plot(z, nd, color=C_DISR, lw=1.8, label="ICS disruption")
    a.set_xscale("log")
    a.set_xlim(0.018, 11)
    a.invert_xaxis()
    a.set_xticks([0.02, 0.1, 0.5, 1, 2, 5, 10])
    a.set_xticklabels(["0.02", "0.1", "0.5", "1", "2", "5", "10"])
    a.set_xlabel("redshift  z")
    a.set_ylabel("orphan events per snapshot")
    a.set_ylim(0, 1150)
    a.legend(fontsize=8.5, loc="upper left")
    a.axvline(1.28, color="0.6", lw=0.8, ls="--")
    a.text(1.22, 120, "z $\\approx$ 1.3", fontsize=8, color="0.35", rotation=90, va="bottom")
    a.set_title("(a) Disruption overtakes merging below z $\\approx$ 1.3",
                loc="left", fontsize=10.5)

    b = ax[0, 1]
    lab = ["0–0.25", "0.25–0.5", "0.5–1", "1–2", "2–3", "3–4",
           "4–6", "6–8", "8–10", "10–13.8", ">13.8"]
    h = f["tclock_hist"]
    xx = np.arange(len(h))
    b.bar(xx, h, color=[C_DISR] * 10 + ["#7a3410"], width=0.78)
    b.set_xticks(xx)
    b.set_xticklabels(lab, fontsize=7.2, rotation=45, ha="right")
    b.set_xlabel("DF merger time still on the clock when subhalo was lost  [Gyr]", fontsize=9)
    b.set_ylabel("disrupted orphans")
    b.set_ylim(0, 1.28 * max(h))
    b.text(0.03, 0.95,
           "median %.1f Gyr still to run\n%.0f%% (last bin) would never\nhave merged at all"
           % (f["tclock_pct"][2], 100 * h[-1] / sum(h)),
           transform=b.transAxes, ha="left", va="top", fontsize=8.5, color="0.25")
    b.set_title("(b) ICS disruption fires long before the merger would have",
                loc="left", fontsize=10.5)

    c = ax[1, 0]
    mm = np.array([f["mstar_merged"], f["cold_merged"], f["hot_merged"]]) / 1e12
    dd = np.array([f["mstar_disr"], f["cold_disr"], f["hot_disr"]]) / 1e12
    x = np.arange(3)
    w = 0.38
    c.bar(x - w / 2, mm, w, color=C_MERGE, label="via merger")
    c.bar(x + w / 2, dd, w, color=C_DISR, label="via ICS disruption")
    for x_, vv in zip(x - w / 2, mm):
        c.text(x_, vv + 1.5, f"{vv:.1f}", ha="center", color=C_MERGE, fontsize=8.5)
    for x_, vv in zip(x + w / 2, dd):
        c.text(x_, vv + 1.5, f"{vv:.1f}", ha="center", color=C_DISR, fontsize=8.5)
    c.set_xticks(x)
    c.set_xticklabels(["stars  M$_\\star$", "cold gas", "hot + CGM"])
    c.set_ylabel("total mass moved to the central  [10$^{12}$ M$_\\odot$]")
    c.set_ylim(0, 1.2 * dd.max())
    c.legend(fontsize=8.5)
    c.set_title("(c) Everything an orphan owns is handed over in one snapshot",
                loc="left", fontsize=10.5)

    e = ax[1, 1]
    fm = [f["n_merged"], v["n_merged"]]
    fd = [f["n_disr"], v["n_disr"]]
    x = np.arange(2)
    e.bar(x, fm, 0.5, color=C_MERGE, label="Merger")
    e.bar(x, fd, 0.5, bottom=fm, color=C_DISR, label="ICS disruption")
    for i in range(2):
        tot = fm[i] + fd[i]
        e.text(x[i], fm[i] / 2, f"{fm[i]:,}\n({fm[i]/tot*100:.1f}%)",
               ha="center", va="center", color="w", fontsize=8.5)
        e.text(x[i], fm[i] + fd[i] / 2, f"{fd[i]:,}\n({fd[i]/tot*100:.1f}%)",
               ha="center", va="center", color="w", fontsize=8.5)
    e.set_xticks(x)
    e.set_xticklabels(["fiducial\n(CGM + FFB on)", "vanilla\n(CGM + FFB off)"])
    e.set_ylabel("orphan events")
    e.set_ylim(0, 1.3 * max(np.array(fm) + np.array(fd)))
    e.legend(fontsize=8.5, loc="upper center", ncol=2)
    e.set_title("(d) The split is set by the trees, not by the baryon physics",
                loc="left", fontsize=10.5)

    fig.suptitle("SAGE26 orphans — timing, mass transfer, and robustness", fontsize=11, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.962])
    out = os.path.join(FIGS, "fig2_orphan_timing.png")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


def main():
    os.makedirs(FIGS, exist_ok=True)
    d = json.load(open(os.path.join(DATA, "orphan_analysis.json")))
    fig_fates(d["fiducial"])
    fig_timing(d["fiducial"], d["vanilla"])


if __name__ == "__main__":
    main()

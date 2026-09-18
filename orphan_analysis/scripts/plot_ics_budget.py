"""Figure 3: does the orphan channel explain the bloated ICS reservoir?

Input : data/ics_budget.json      (z=0 ICS budget, three ThresholdSatDisruption runs)
        data/orphan_analysis.json (orphan event history)
Output: figures/fig3_ics_budget.png
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

C_MERGE = "#2a6fb0"
C_DISR = "#d1622a"
C_OBS = "#5c6b73"

plt.rcParams.update({
    "font.size": 9, "axes.linewidth": 0.8,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "figure.dpi": 130, "savefig.dpi": 130,
    "axes.labelsize": 10, "legend.frameon": False,
})

b = json.load(open(os.path.join(DATA, "ics_budget.json")))
o = json.load(open(os.path.join(DATA, "orphan_analysis.json")))["fiducial"]

fig, ax = plt.subplots(2, 2, figsize=(10, 7.2))

# (a) f_ICS vs halo mass, against the observed ICL range
a = ax[0, 0]
p = b["fid"]["profile"]
lm = np.array([q["lm"] for q in p])
a.fill_between([10.8, 14.6], 10, 40, color=C_OBS, alpha=0.18, lw=0)
a.text(11.0, 25, "observed ICL fraction\n(~10–40%, cluster studies)",
       fontsize=8, color=C_OBS, va="center")
a.plot(lm, [q["f"] * 100 for q in p], color=C_DISR, lw=2, marker="o", ms=5,
       label="SAGE26 fiducial (mass-weighted)")
a.plot(lm, [q["fmed"] * 100 for q in p], color=C_DISR, lw=1.4, ls="--",
       label="SAGE26 fiducial (median)")
a.set_xlim(10.8, 14.6)
a.set_ylim(0, 80)
a.set_xlabel(r"log$_{10}$ ( M$_{\rm vir}$ / M$_\odot$ )")
a.set_ylabel(r"f$_{\rm ICS}$ = ICS / (ICS + M$_\star$)  [%]")
a.legend(fontsize=8.5, loc="upper left")
a.set_title("(a) The ICS overshoots badly in groups", loc="left", fontsize=10.5)

# (b) where the ICS came from
c = ax[0, 1]
dis, acc = b["fid"]["ics_disrupt"], b["fid"]["ics_accrete"]
tot = dis + acc
c.barh([0], [dis / tot * 100], color=C_DISR, height=0.45,
       label="ICS_disrupt — stars shredded in this halo")
c.barh([0], [acc / tot * 100], left=[dis / tot * 100], color="#8c4a20", height=0.45,
       label="ICS_accrete — ICS inherited from satellites")
c.text(dis / tot * 50, 0, f"{dis/tot*100:.0f}%", ha="center", va="center", color="w")
c.text(dis / tot * 100 + acc / tot * 50, 0, f"{acc/tot*100:.0f}%",
       ha="center", va="center", color="w")
c.set_yticks([])
c.set_ylim(-0.6, 1.1)
c.set_xlim(0, 100)
c.set_xlabel("share of the z = 0 ICS reservoir [%]")
c.legend(fontsize=8.5, loc="upper center")
c.text(50, -0.45,
       "both channels originate in disrupt_satellite_to_ICS(),\n"
       "and 99.7% of those calls act on orphans",
       ha="center", va="center", fontsize=8.5, color="0.25")
c.set_title("(b) The ICS has exactly one source", loc="left", fontsize=10.5)

# (c) ICS growth history, split by channel
e = ax[1, 0]
z = np.array([q["z"] for q in o["per_snap"]])
cum_d = np.cumsum([q["ms_disr"] for q in o["per_snap"]])[::-1] / 1e13
cum_m = np.cumsum([q["ms_merged"] for q in o["per_snap"]])[::-1] / 1e13
zz = z[::-1]
e.plot(zz, cum_d, color=C_DISR, lw=2, label="shredded to ICS")
e.plot(zz, cum_m, color=C_MERGE, lw=2, label="merged into remnants")
e.set_xscale("log")
e.set_xlim(0.018, 11)
e.invert_xaxis()
e.set_xticks([0.02, 0.1, 0.5, 1, 2, 5, 10])
e.set_xticklabels(["0.02", "0.1", "0.5", "1", "2", "5", "10"])
e.set_xlabel("redshift  z")
e.set_ylabel(r"cumulative orphan M$_\star$  [10$^{13}$ M$_\odot$]")
e.legend(fontsize=8.5, loc="upper right")
e.set_title("(c) The ICS is built late, from orphans", loc="left", fontsize=10.5)

# (d) threshold sensitivity
d = ax[1, 1]
tags = ["th0", "fid", "th10"]
labs = ["0.0\n(nominally\n'never disrupt')", "1.0\n(fiducial)", "10.0"]
fg = [b[t]["fglob"] * 100 for t in tags]
x = np.arange(3)
d.bar(x, fg, 0.5, color=[C_DISR] * 3)
for xx, vv in zip(x, fg):
    d.text(xx, vv + 1, f"{vv:.1f}%", ha="center", color=C_DISR)
d.axhspan(10, 40, color=C_OBS, alpha=0.18, lw=0)
d.set_xticks(x)
d.set_xticklabels(labs, fontsize=8.5)
d.set_xlabel("ThresholdSatDisruption")
d.set_ylabel(r"global f$_{\rm ICS}$  [%]")
d.set_ylim(0, 46)
d.text(0.02, 0.96, "the knob cannot switch the channel off:\n"
                   "currentMvir reaches exactly 0 on the final\n"
                   "substep, so the test always passes",
       transform=d.transAxes, va="top", fontsize=8.5, color="0.25")
d.set_title("(d) ThresholdSatDisruption is inert for orphans", loc="left", fontsize=10.5)

fig.suptitle("Why the SAGE26 ICS reservoir is bloated — mini-Millennium, z = 0",
             fontsize=11, y=0.985)
fig.tight_layout(rect=[0, 0, 1, 0.962])
out = os.path.join(FIGS, "fig3_ics_budget.png")
fig.savefig(out, bbox_inches="tight")
print("wrote", out)

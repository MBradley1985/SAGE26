"""Figure 4: SAGE26's ICS against the two published prescriptions.

Henriques & Thomas (2010), MNRAS 403, 768  -- tidal stripping of orphans
Contini et al. (2014), MNRAS 437, 3787     -- three ICL prescriptions compared

Input : data/ics_budget.json, data/orphan_events_fiducial.csv
Output: figures/fig4_literature_comparison.png
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

C_SAGE = "#d1622a"
C_HT10 = "#2a6fb0"
C_CON = "#1b7f5e"
C_OBS = "#5c6b73"
HUBBLE_H = 0.73

plt.rcParams.update({
    "font.size": 9, "axes.linewidth": 0.8,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "figure.dpi": 130, "savefig.dpi": 130,
    "axes.labelsize": 10, "legend.frameon": False,
})

b = json.load(open(os.path.join(DATA, "ics_budget.json")))
ev = np.genfromtxt(os.path.join(DATA, "orphan_events_fiducial.csv"),
                   delimiter=",", names=True)

fig, ax = plt.subplots(2, 2, figsize=(10, 7.2))

# (a) f_ICS vs halo mass against both papers
a = ax[0, 0]
p = b["fid"]["profile"]
a.fill_between([10.8, 14.6], 10, 40, color=C_OBS, alpha=0.16, lw=0)
a.text(10.95, 34, "observed ICL, 10–40%", fontsize=8, color=C_OBS)
a.plot([q["lm"] for q in p], [q["f"] * 100 for q in p],
       color=C_SAGE, lw=2.2, marker="o", ms=5, label="SAGE26 (this work)")
# Henriques & Thomas 2010: ~7% at 1e12, ~18% mean above 1e13
a.plot([12.0, 13.0, 14.0], [7, 18, 18], color=C_HT10, lw=2, ls="--",
       marker="s", ms=6, label="Henriques & Thomas (2010)")
# Contini+14: 20-40%, no halo-mass trend
a.fill_between([13.0, 14.6], 20, 40, color=C_CON, alpha=0.22, lw=0)
a.plot([13.0, 14.6], [27, 27], color=C_CON, lw=2, ls="-.",
       label="Contini et al. (2014), flat")
a.set_xlim(10.8, 14.6)
a.set_ylim(0, 80)
a.set_xlabel(r"log$_{10}$ ( M$_{\rm vir}$ / M$_\odot$ )")
a.set_ylabel(r"f$_{\rm ICS}$ = ICS / (ICS + M$_\star$)  [%]")
a.legend(fontsize=8, loc="upper left")
a.set_title("(a) SAGE26 is 3–4$\\times$ high, and has the wrong slope",
            loc="left", fontsize=10.5)

# (b) which donors build the ICS
c = ax[0, 1]
ms = ev["mstar"] * 1e10 / HUBBLE_H
v = ms[(ev["fate"].astype(int) == 0) & (ms > 0)]
lm = np.log10(v)
bins = np.arange(6, 12.1, 0.5)
share = np.array([v[(lm >= lo) & (lm < lo + 0.5)].sum() for lo in bins[:-1]])
share = share / share.sum() * 100
c.bar(bins[:-1] + 0.25, share, 0.45, color=C_SAGE, label="SAGE26 ICS donors")
c.axvspan(10, 11, color=C_CON, alpha=0.22, lw=0)
c.text(7.6, 27, "Contini+14: bulk of ICL\nfrom M$_\\star$ ~ 10$^{10-11}$ M$_\\odot$",
       ha="left", fontsize=8, color=C_CON)
c.set_xlabel(r"donor log$_{10}$ ( M$_\star$ / M$_\odot$ )")
c.set_ylabel("share of ICS mass [%]")
c.set_xlim(6.5, 11.8)
c.set_ylim(0, 34)
c.set_title("(b) The donor spectrum is right — 60% from M$_\\star$ > 10$^{10}$",
            loc="left", fontsize=10.5)

# (c) what a tidal radius would actually strip
e = ax[1, 0]
x = np.linspace(0.01, 6, 400)
f_disc = (1 + x) * np.exp(-x) * 100                 # exponential disc outside R_t/R_d
f_bulge = 1 / (1 + (x / 0.56) ** 2) * 100           # Hernquist-like bulge, a = 0.56 R_b
e.plot(x, f_disc, color=C_HT10, lw=2, label="exponential disc (H&T10 eq. 6)")
e.plot(x, f_bulge, color=C_CON, lw=2, ls="--", label="bulge (H&T10 eq. 9)")
e.axhline(100, color=C_SAGE, lw=2.2, ls=":")
e.text(5.9, 104, "SAGE26: always 100%", color=C_SAGE, fontsize=9, ha="right")
e.set_xlim(0, 6)
e.set_ylim(0, 122)
e.set_xlabel(r"tidal radius  R$_t$ / R$_{\rm scale}$")
e.set_ylabel("fraction of stellar mass stripped [%]")
e.legend(fontsize=8.5, loc="center right")
e.set_title("(c) The fix: strip outside R$_t$, don't delete the galaxy",
            loc="left", fontsize=10.5)

# (d) the merger channel SAGE26 is missing
d = ax[1, 1]
merged_ms = ms[(ev["fate"].astype(int) == 1) & (ms > 0)].sum()
ics_now = b["fid"]["ics"]
mstar_now = b["fid"]["mstar"]
labels = ["SAGE26 now", "+ Contini+14\nmerger channel\n(20% unbound)"]
vals = [ics_now / (ics_now + mstar_now) * 100,
        (ics_now + 0.2 * merged_ms) / (ics_now + mstar_now) * 100]
d.bar([0, 1], vals, 0.5, color=[C_SAGE, "#8c4a20"])
for xx, vv in zip([0, 1], vals):
    d.text(xx, vv + 0.6, f"{vv:.1f}%", ha="center", color="0.25")
d.axhspan(10, 40, color=C_OBS, alpha=0.16, lw=0)
d.set_xticks([0, 1])
d.set_xticklabels(labels, fontsize=8.5)
d.set_ylabel(r"global f$_{\rm ICS}$  [%]")
d.set_ylim(0, 34)
d.text(0.03, 0.96,
       "SAGE26 puts 100% of a merged satellite's\n"
       "stars into the bulge. Adding the merger\n"
       "channel makes the overshoot worse — the\n"
       "disruption channel has to shrink first.",
       transform=d.transAxes, va="top", fontsize=8.2, color="0.25")
d.set_title("(d) The other missing channel points the same way",
            loc="left", fontsize=10.5)

fig.suptitle("SAGE26 ICS vs. Henriques & Thomas (2010) and Contini et al. (2014)",
             fontsize=11, y=0.985)
fig.tight_layout(rect=[0, 0, 1, 0.962])
out = os.path.join(FIGS, "fig4_literature_comparison.png")
fig.savefig(out, bbox_inches="tight")
print("wrote", out)

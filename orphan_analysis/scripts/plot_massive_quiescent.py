"""Figure 5: central / satellite split of the massive quiescent population.

Replicates the CentralSatelliteFraction_MassiveQuiescent diagnostic from
plotting/allresults-history.py (top 10 per cent by stellar mass, sSFR below
0.2/t_H) for the DisruptionGate = 0 and = 1 runs. Satellites are Type >= 1, so
orphans are included where they exist -- matching the script, and the reason the
two curves sum to 1 again.

Output: figures/fig5_massive_quiescent.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import h5py

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(HERE, "figures")

HUBBLE_H = 0.73
OMEGA_M = 0.25
OMEGA_L = 0.75
INV_H0_YR = (9.778 / HUBBLE_H) * 1.0e9
ALIST = "/Users/mbradley/Documents/PhD/SAGE26/input/millennium/trees/millennium.a_list"

C_CEN = "#2a6fb0"
C_SAT = "#b03a2a"

plt.rcParams.update({
    "font.size": 9, "axes.linewidth": 0.8,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "figure.dpi": 130, "savefig.dpi": 130,
    "axes.labelsize": 10, "legend.frameon": False,
})


def curve(fn, alist):
    """Central / satellite / orphan fractions of the massive quiescent sample."""
    f = h5py.File(fn, "r")
    rows = []
    for snap in range(len(alist)):
        key = "Snap_%d" % snap
        if key not in f:
            continue
        g = f[key]
        mstar = g["StellarMass"][:] * 1.0e10 / HUBBLE_H
        types = g["Type"][:]
        sfr = g["SfrDisk"][:] + g["SfrBulge"][:]

        w = np.where(mstar > 0.0)[0]
        if len(w) < 10:
            continue
        massive = w[mstar[w] >= np.percentile(mstar[w], 90)]
        if len(massive) == 0:
            continue

        z = 1.0 / alist[snap] - 1.0
        e_z = np.sqrt(OMEGA_M * (1.0 + z) ** 3 + OMEGA_L)
        q = massive[(sfr[massive] / mstar[massive]) < 0.2 * e_z / INV_H0_YR]
        if len(q) == 0:
            continue

        n_q = float(len(q))
        rows.append((z,
                     np.sum(types[q] == 0) / n_q,
                     np.sum(types[q] >= 1) / n_q))
    return np.array(sorted(rows))


def main():
    os.makedirs(FIGS, exist_ok=True)
    alist = np.loadtxt(ALIST)

    runs = [("DisruptionGate = 0 (default)",
             "/Users/mbradley/Documents/PhD/SAGE26/output/millennium/model_0.hdf5"),
            ("DisruptionGate = 1",
             os.environ["GATE_RUN"])]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, (label, fn) in zip(axes, runs):
        d = curve(fn, alist)
        ax.plot(d[:, 0], d[:, 1], color=C_CEN, lw=2, label="Centrals")
        ax.plot(d[:, 0], d[:, 2], color=C_SAT, lw=2, label="Satellites")
        ax.plot(d[:, 0], d[:, 1] + d[:, 2], color="0.6", lw=0.9, ls=":", label="sum")
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 1.08)
        ax.set_xlabel(r"redshift  $z$")
        ax.set_title(label, loc="left", fontsize=10.5)
    axes[0].set_ylabel(r"fraction of massive quiescent galaxies")
    axes[0].legend(fontsize=8.5, loc="upper left")

    fig.suptitle("Massive quiescent galaxies by type (top 10% $M_\\star$, sSFR < 0.2/$t_H$)",
                 fontsize=11, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(FIGS, "fig5_massive_quiescent.png")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()

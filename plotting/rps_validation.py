"""
Science validation for the ram-pressure ISM stripping toggle
(RamPressureStrippingOn, model_ram_pressure.c).

Compares matched microUchuu runs (identical except the toggle / epsilon)
against the observational targets from the validation plan:

  (a) quenched Type-1 satellite fraction vs host halo mass, against the
      SDSS group-catalogue trend of Wetzel et al. (2012);
  (b) median satellite HI-to-stellar ratio vs host halo mass at fixed
      stellar mass (the HI-deficiency trend of Brown et al. 2017);
  (c) the global HI mass function against the ALFALFA Schechter fit of
      Jones et al. (2018) -- the calibration-safety check;
  (d) median satellite HI retention (on/off) vs time since infall, split
      by host mass: the cumulative-exposure signature of the mechanism.

Run directories are produced by sage from par files that differ only in
RamPressureStrippingOn / RamPressureEpsilon (see output/rps_demo_uchuu/).
Runs listed in EPS_RUNS that do not exist yet are skipped, so the script
works before an epsilon sweep has finished.

Usage: python3 plotting/rps_validation.py
"""
import os
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/Users/mbradley/Documents/PhD/SAGE26/output/rps_demo_uchuu"
OUTDIR = BASE
SNAP = "Snap_49"          # z = 0
H = 0.6774
BOX = 100.0               # Mpc/h
OMEGA_M, OMEGA_L = 0.3089, 0.6911

# run label -> subdirectory; "off" is the reference every ratio uses
EPS_RUNS = [("off", "off"), ("eps=0.3", "eps03"), ("eps=1", "on"), ("eps=3", "eps30")]

# satellite stellar-mass window for the observational comparisons [Msun];
# matches the intermediate-mass bins where Wetzel+12 and Brown+17 are best
# constrained and SAGE is comfortably above the resolution floor
LOGMSTAR_LO, LOGMSTAR_HI = 9.5, 10.5

# Wetzel, Tinker & Conroy (2012, MNRAS 424, 232) fig. 5, satellites with
# log M* ~ 9.7-10.1: approximate by-eye digitisation, +/-0.05 band
WETZEL12_LOGMH = np.array([12.25, 12.75, 13.25, 13.75, 14.25, 14.75])
WETZEL12_FQ    = np.array([0.32,  0.38,  0.44,  0.52,  0.60,  0.66])
WETZEL12_ERR = 0.05

# Jones et al. (2018, MNRAS 477, 2) ALFALFA 100% HIMF Schechter fit
# (h70 = H0/70 units): phi* [1e-3 Mpc^-3 dex^-1], log10 M* [Msun], alpha
J18_PHISTAR, J18_LOGMSTAR, J18_ALPHA = 4.5e-3, 9.94, -1.25


def age_gyr(z):
    """Flat-LCDM age of the universe at redshift z [Gyr] (analytic)."""
    H0 = 100.0 * H * 1.02271e-3   # km/s/Mpc -> 1/Gyr
    return (2.0 / (3.0 * H0 * np.sqrt(OMEGA_L))) * np.arcsinh(
        np.sqrt(OMEGA_L / OMEGA_M) * (1.0 + np.asarray(z)) ** -1.5)


def load(subdir):
    out = {}
    path = f"{BASE}/{subdir}/model.hdf5"
    with h5py.File(path, "r") as f:
        s = f[f"Core_0/{SNAP}"]
        for k in ["Type", "GalaxyIndex", "CentralGalaxyIndex", "StellarMass",
                  "ColdGas", "H1gas", "H2gas", "CentralMvir",
                  "SfrDisk", "SfrBulge", "TimeOfInfall"]:
            out[k] = s[k][:]
        out["snap_z"] = f["Header/snapshot_redshifts"][:]
    return out


def match(ref, other):
    """Row-align `other` to `ref` on GalaxyIndex; returns aligned dicts."""
    common, i_r, i_o = np.intersect1d(ref["GalaxyIndex"], other["GalaxyIndex"],
                                      return_indices=True)
    r = {k: v[i_r] for k, v in ref.items() if k != "snap_z"}
    o = {k: v[i_o] for k, v in other.items() if k != "snap_z"}
    return r, o


def quenched(d):
    ssfr = (d["SfrDisk"] + d["SfrBulge"]) / np.maximum(d["StellarMass"] * 1e10 / H, 1.0)
    return ssfr < 1e-11


runs = {}
for label, sub in EPS_RUNS:
    if os.path.exists(f"{BASE}/{sub}/model.hdf5"):
        runs[label] = load(sub)
    else:
        print(f"[skip] {label}: {BASE}/{sub}/model.hdf5 not found")
off = runs["off"]
snap_z = off["snap_z"]

colors = {"off": "steelblue", "eps=0.3": "seagreen", "eps=1": "crimson", "eps=3": "purple"}

fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.5))
fig.suptitle("Ram-pressure ISM stripping: science validation (microUchuu z=0)", fontsize=13)

# ---------------------------------------------------------------- (a)
axa = axes[0, 0]
hbins = np.arange(12.0, 15.01, 0.5)
hmid = 0.5 * (hbins[1:] + hbins[:-1])
for label, d in runs.items():
    ref, o = match(off, d)
    lmstar = np.log10(np.maximum(o["StellarMass"] * 1e10 / H, 1.0))
    lmh = np.log10(np.maximum(o["CentralMvir"] * 1e10 / H, 1.0))
    sel = (o["Type"] == 1) & (lmstar > LOGMSTAR_LO) & (lmstar < LOGMSTAR_HI)
    q = quenched(o)
    fq = [q[sel & (lmh >= lo) & (lmh < hi)].mean()
          if (sel & (lmh >= lo) & (lmh < hi)).sum() > 19 else np.nan
          for lo, hi in zip(hbins[:-1], hbins[1:])]
    axa.plot(hmid, fq, "o-", color=colors[label], lw=2, label=label)
axa.fill_between(WETZEL12_LOGMH, WETZEL12_FQ - WETZEL12_ERR, WETZEL12_FQ + WETZEL12_ERR,
                 color="gray", alpha=0.35, label="Wetzel+12 (SDSS, approx.)")
axa.set(xlabel=r"$\log_{10} M_{\rm host}\ [M_\odot]$",
        ylabel="quenched satellite fraction",
        title=rf"Quenched Type-1 satellites, $10^{{{LOGMSTAR_LO}}}$--$10^{{{LOGMSTAR_HI}}}\,M_\odot$",
        ylim=(0, 1))
axa.legend(fontsize=8, loc="upper left")

# ---------------------------------------------------------------- (b)
axb = axes[0, 1]
for label, d in runs.items():
    ref, o = match(off, d)
    lmstar = np.log10(np.maximum(o["StellarMass"] * 1e10 / H, 1.0))
    lmh = np.log10(np.maximum(o["CentralMvir"] * 1e10 / H, 1.0))
    sel = (o["Type"] == 1) & (lmstar > LOGMSTAR_LO) & (lmstar < LOGMSTAR_HI) & (o["H1gas"] > 0)
    with np.errstate(divide="ignore"):
        fhi = np.log10(o["H1gas"] / np.maximum(o["StellarMass"], 1e-12))
    med = [np.median(fhi[sel & (lmh >= lo) & (lmh < hi)])
           if (sel & (lmh >= lo) & (lmh < hi)).sum() > 19 else np.nan
           for lo, hi in zip(hbins[:-1], hbins[1:])]
    axb.plot(hmid, med, "s-", color=colors[label], lw=2, label=label)
axb.annotate("Brown+17 (xGASS): mean satellite gas fraction\nsuppressed by ~0.3-0.6 dex from group to\ncluster scale at fixed $M_*$",
             xy=(0.03, 0.05), xycoords="axes fraction", fontsize=8, style="italic")
axb.set(xlabel=r"$\log_{10} M_{\rm host}\ [M_\odot]$",
        ylabel=r"median $\log_{10}(M_{\rm HI}/M_*)$ (HI-detected)",
        title="Satellite HI content vs environment")
axb.legend(fontsize=8, loc="upper right")

# ---------------------------------------------------------------- (c)
axc = axes[1, 0]
vol = (BOX / H) ** 3          # Mpc^3
mb = np.arange(7.5, 11.01, 0.25)
mbm = 0.5 * (mb[1:] + mb[:-1])
for label, d in runs.items():
    hi = d["H1gas"] * 1e10 / H
    counts, _ = np.histogram(np.log10(hi[hi > 0]), bins=mb)
    phi = counts / vol / 0.25
    axc.plot(mbm, np.log10(np.maximum(phi, 1e-12)), "-", color=colors[label], lw=2, label=label)
h70 = H / 0.7
mgrid = 10 ** (mbm - (J18_LOGMSTAR - 2.0 * np.log10(h70)))   # M / M*
schechter = np.log(10) * (J18_PHISTAR * h70 ** 3) * mgrid ** (J18_ALPHA + 1) * np.exp(-mgrid)
axc.plot(mbm, np.log10(schechter), "k--", lw=1.5, label="ALFALFA (Jones+18)")
axc.set(xlabel=r"$\log_{10} M_{\rm HI}\ [M_\odot]$",
        ylabel=r"$\log_{10} \phi\ [{\rm Mpc^{-3}\,dex^{-1}}]$",
        title="Global HI mass function (calibration check)", ylim=(-5.5, 0))
axc.legend(fontsize=8, loc="lower left")

# ---------------------------------------------------------------- (d)
axd = axes[1, 1]
if "eps=1" in runs:
    ref, o = match(off, runs["eps=1"])
    lmh = np.log10(np.maximum(ref["CentralMvir"] * 1e10 / H, 1.0))
    sel0 = (ref["Type"] == 1) & (ref["StellarMass"] > 1e-3) & (ref["H1gas"] > 0) \
           & (ref["TimeOfInfall"] >= 0)
    snap_inf = np.clip(ref["TimeOfInfall"].astype(int), 0, len(snap_z) - 1)
    t_since = age_gyr(0.0) - age_gyr(snap_z[snap_inf])
    tbins = np.arange(0, 12.1, 1.5)
    tmid = 0.5 * (tbins[1:] + tbins[:-1])
    for lo_h, hi_h, c, lab in [(12.0, 13.0, "seagreen", r"$10^{12-13}$ hosts"),
                               (13.0, 14.0, "darkorange", r"$10^{13-14}$ hosts"),
                               (14.0, 15.5, "crimson", r"$>10^{14}$ hosts")]:
        m0 = sel0 & (lmh >= lo_h) & (lmh < hi_h)
        ret = o["H1gas"][m0] / ref["H1gas"][m0]
        tt = t_since[m0]
        med = [np.median(ret[(tt >= lo) & (tt < hi)])
               if ((tt >= lo) & (tt < hi)).sum() > 19 else np.nan
               for lo, hi in zip(tbins[:-1], tbins[1:])]
        axd.plot(tmid, med, "o-", color=c, lw=2, label=lab)
    axd.axhline(1.0, color="k", ls=":", lw=1)
    axd.set(xlabel="time since infall [Gyr]",
            ylabel=r"median HI$_{\rm on}$ / HI$_{\rm off}$ (eps = 1)",
            title="Stripping vs environmental exposure time", ylim=(0, 1.15))
    axd.legend(fontsize=8, loc="lower left")

fig.tight_layout()
figpath = f"{OUTDIR}/rps_validation.png"
fig.savefig(figpath, dpi=110)
print(f"figure: {figpath}")

# ------------------------------------------------------------ numbers
print("\nsatellite totals (Type 1, M* > 1e7 Msun/h), relative to off:")
sel = (off["Type"] == 1) & (off["StellarMass"] > 1e-3)
for label, d in runs.items():
    if label == "off":
        continue
    ref, o = match(off, d)
    s = (ref["Type"] == 1) & (ref["StellarMass"] > 1e-3)
    for k in ("ColdGas", "H1gas", "H2gas"):
        r = o[k][s].sum() / ref[k][s].sum()
        print(f"  {label:8s} {k:8s} {100 * (r - 1):+6.1f}%")

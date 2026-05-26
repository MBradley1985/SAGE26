#!/usr/bin/env python
"""
H2/HI validation diagnostic — multi-model comparison.

Compares H2/M*, HI/M*, and H2 depletion time across models against
COLD GASS (Saintonge+2011/2017) and xGASS (Catinella+18).
Also prints the BR06 surface density / f_mol table for the first model.

Run from the SAGE26 root:
    python plotting/random_plotting_scripts/h2_validation.py
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import h5py

# ---------------------------------------------------------------------------
# Config — add/remove models here
# ---------------------------------------------------------------------------
SNAP_KEY = 'Snap_49'   # z=0 for MicroUchuu

MODELS = [
    {'dir': './output/microuchuu/',               'label': 'FFB radial integration',   'color': 'steelblue'},
    {'dir': './output/microuchuu_pir2/',          'label': 'FFB $\pi r^2$',  'color': 'tomato'},
    {'dir': './output/microuchuu_pi3r2/',         'label': 'FFB $\pi 3r^2$',     'color': 'forestgreen'},
    {'dir': './output/microuchuu_h2/',            'label': 'FFBs H2',   'color': 'mediumpurple'},
]

# ---------------------------------------------------------------------------
# Observational reference data
# ---------------------------------------------------------------------------

# COLD GASS / xCOLD GASS (Saintonge+2011, 2017): log(M_H2/M*) per bin
COLDGASS_LOGMSTAR       = np.array([9.0,  9.5,  10.0,  10.5,  11.0])
COLDGASS_LOG_H2_FRAC_MED = np.array([-0.80, -1.00, -1.30, -1.70, -2.10])
COLDGASS_LOG_H2_FRAC_LO  = np.array([-1.00, -1.30, -1.70, -2.10, -2.50])
COLDGASS_LOG_H2_FRAC_HI  = np.array([-0.50, -0.70, -1.00, -1.30, -1.70])

# xGASS (Catinella+18): log(M_HI/M*) per bin
XGASS_LOGMSTAR          = np.array([9.0,  9.5,  10.0,  10.5,  11.0])
XGASS_LOG_HI_FRAC_MED   = np.array([ 0.10, -0.15, -0.50, -0.90, -1.40])
XGASS_LOG_HI_FRAC_LO    = np.array([-0.20, -0.50, -0.90, -1.30, -1.80])
XGASS_LOG_HI_FRAC_HI    = np.array([ 0.40,  0.20, -0.10, -0.50, -1.00])

TDEP_OBS_GYR = 2.0   # Saintonge+2011, Bigiel+08

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def find_model_files(directory):
    import glob
    files = sorted(glob.glob(os.path.join(directory, 'model_*.hdf5')))
    if not files:
        single = os.path.join(directory, 'model_0.hdf5')
        if os.path.exists(single):
            files = [single]
    return files


def load_z0(directory, snap_key):
    files = find_model_files(directory)
    if not files:
        print(f"  WARNING: no model_*.hdf5 in {directory} — skipping")
        return None, None, None

    with h5py.File(files[0], 'r') as f:
        h   = float(f['Header/Simulation'].attrs['hubble_h'])
        umg = float(f['Header/Runtime'].attrs['UnitMass_in_g'])

    mass_convert = umg / 1.989e33 / h

    fields = ['Type', 'StellarMass', 'BulgeMass', 'ColdGas',
              'H2gas', 'H1gas', 'SfrDisk', 'SfrBulge', 'DiskRadius', 'Mvir']
    arrays = {k: [] for k in fields}

    for fp in files:
        with h5py.File(fp, 'r') as f:
            if snap_key not in f:
                continue
            grp = f[snap_key]
            for k in fields:
                if k in grp:
                    arrays[k].append(grp[k][:])

    if not any(arrays[k] for k in fields):
        print(f"  WARNING: snap {snap_key} not found in {directory} — skipping")
        return None, None, None

    data = {k: np.concatenate(v) if v else np.array([]) for k, v in arrays.items()}
    return data, mass_convert, h


def prepare(data, mc, h):
    """Convert to physical units and apply central-galaxy selection."""
    mstar  = data['StellarMass'] * mc
    mbulge = data['BulgeMass']   * mc
    mh2    = data['H2gas']       * mc
    mhi    = data['H1gas']       * mc
    mcold  = data['ColdGas']     * mc
    sfr    = data['SfrDisk'] + data['SfrBulge']   # already M_sun/yr
    dr     = data['DiskRadius'] * 1.0e6 / h       # Mpc/h → physical pc

    sel = (data['Type'] == 0) & (mstar > 1e8) & (mcold > 0) & np.isfinite(mstar)
    return {'mstar': mstar[sel], 'mbulge': mbulge[sel],
            'mh2': mh2[sel], 'mhi': mhi[sel], 'mcold': mcold[sel],
            'sfr': sfr[sel], 'dr': dr[sel]}


def bin_stat(x, y, bins, min_n=20):
    mids, meds, lo, hi = [], [], [], []
    for i in range(len(bins)-1):
        ok = (x >= bins[i]) & (x < bins[i+1]) & np.isfinite(y) & (y > 0)
        if ok.sum() >= min_n:
            mids.append(0.5*(bins[i]+bins[i+1]))
            meds.append(np.median(y[ok]))
            lo.append(np.percentile(y[ok], 16))
            hi.append(np.percentile(y[ok], 84))
    return map(np.array, (mids, meds, lo, hi))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    bins = np.arange(8.5, 12.0, 0.5)

    loaded = []
    for m in MODELS:
        print(f"Loading {m['dir']} ...")
        data, mc, h = load_z0(m['dir'], SNAP_KEY)
        if data is None:
            continue
        gal = prepare(data, mc, h)
        loaded.append((m['label'], m['color'], gal, mc, h))
        print(f"  {gal['mstar'].size:,} central galaxies with ColdGas>0")

    if not loaded:
        sys.exit("No models loaded.")

    # -----------------------------------------------------------------------
    # Surface density table (first available model only)
    # -----------------------------------------------------------------------
    label0, _, gal0, _, _ = loaded[0]
    print(f"\n--- BR06 surface density / f_mol table  [{label0}] ---")
    print(f"{'logM*':>7}  {'N':>6}  {'Σ_gas':>8}  {'Σ_star':>8}  {'h*(pc)':>7}  "
          f"{'P':>10}  {'f_mol_stored':>12}  {'H2/M*_sim':>10}")
    logm0 = np.log10(gal0['mstar'])
    for i in range(len(bins)-1):
        m = (logm0 >= bins[i]) & (logm0 < bins[i+1])
        if m.sum() < 10:
            continue
        rs_pc  = gal0['dr'][m] / 3.0
        area   = np.pi * (3.0*rs_pc)**2
        sg_gas  = gal0['mcold'][m] / area
        sg_star = (gal0['mstar'][m] - gal0['mbulge'][m]) / area
        log_h   = -0.23 + 0.8*np.log10(np.clip(rs_pc, 1.0, None))
        h_star  = 10.0**log_h
        P = 272.0 * 8.0 * sg_gas * np.sqrt(np.clip(sg_star, 0.1, None)) / np.sqrt(h_star)
        f_stored = gal0['mh2'][m] / np.clip(gal0['mcold'][m]*0.74, 1e-30, None)
        print(f"  {0.5*(bins[i]+bins[i+1]):5.2f}  {m.sum():6d}  "
              f"{np.median(sg_gas):8.1f}  {np.median(sg_star):8.1f}  "
              f"{np.median(h_star):7.0f}  {np.median(P):10.2e}  "
              f"{np.median(f_stored):12.4f}  "
              f"{np.median(gal0['mh2'][m]/gal0['mstar'][m]):10.4f}")

    # -----------------------------------------------------------------------
    # Depletion times
    # -----------------------------------------------------------------------
    print(f"\n--- H2 depletion time (9.5 < log M* < 11, SFR>0) ---")
    for label, _, gal, _, _ in loaded:
        logm = np.log10(gal['mstar'])
        sel  = (gal['sfr'] > 0) & (gal['mh2'] > 0) & (logm > 9.5) & (logm < 11.0)
        if sel.sum() > 10:
            td = gal['mh2'][sel] / gal['sfr'][sel] / 1.0e9
            print(f"  {label:30s}  median {np.median(td):.2f} Gyr  "
                  f"(16th–84th: {np.percentile(td,16):.2f}–{np.percentile(td,84):.2f})")
    print(f"  {'Observed (Saintonge+11)':30s}  ~{TDEP_OBS_GYR:.1f} Gyr")

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'H2/HI validation  ({SNAP_KEY}, centrals)', fontsize=11)

    # Observational shading
    for ax, lo, hi, med, xs, ylabel, ylim in [
        (axes[0], COLDGASS_LOG_H2_FRAC_LO, COLDGASS_LOG_H2_FRAC_HI,
         COLDGASS_LOG_H2_FRAC_MED, COLDGASS_LOGMSTAR,
         r'$\log(M_{\rm H2}/M_*)$', (-3.5, 0.5)),
        (axes[1], XGASS_LOG_HI_FRAC_LO, XGASS_LOG_HI_FRAC_HI,
         XGASS_LOG_HI_FRAC_MED, XGASS_LOGMSTAR,
         r'$\log(M_{\rm HI}/M_*)$', (-3.5, 1.0)),
    ]:
        ax.fill_between(xs, lo, hi, color='grey', alpha=0.25, label='Observed')
        ax.plot(xs, med, 'k--', lw=1.5)

    # Simulation curves
    for label, color, gal, _, _ in loaded:
        logm = np.log10(gal['mstar'])

        # H2/M*
        bm, med, lo, hi = bin_stat(logm, gal['mh2']/gal['mstar'], bins)
        if len(bm):
            axes[0].plot(bm, np.log10(med), '-o', color=color, lw=2, ms=5, label=label)
            axes[0].fill_between(bm, np.log10(np.clip(lo,1e-5,None)),
                                 np.log10(np.clip(hi,1e-5,None)), color=color, alpha=0.15)

        # HI/M*
        bm, med, lo, hi = bin_stat(logm, gal['mhi']/gal['mstar'], bins)
        if len(bm):
            axes[1].plot(bm, np.log10(np.clip(med,1e-5,None)), '-o',
                         color=color, lw=2, ms=5, label=label)
            axes[1].fill_between(bm,
                                 np.log10(np.clip(lo,1e-5,None)),
                                 np.log10(np.clip(hi,1e-5,None)), color=color, alpha=0.15)

        # Depletion time
        sel2 = (gal['sfr'] > 0) & (gal['mh2'] > 0)
        bm, med, lo, hi = bin_stat(logm[sel2], gal['mh2'][sel2]/gal['sfr'][sel2]/1e9, bins)
        if len(bm):
            axes[2].plot(bm, med, '-o', color=color, lw=2, ms=5, label=label)
            axes[2].fill_between(bm, lo, hi, color=color, alpha=0.15)

    for ax, ylabel, ylim in [
        (axes[0], r'$\log(M_{\rm H2}/M_*)$',  (-3.5, 0.5)),
        (axes[1], r'$\log(M_{\rm HI}/M_*)$',  (-3.5, 1.0)),
    ]:
        ax.set_xlabel(r'$\log(M_*/M_\odot)$');  ax.set_ylabel(ylabel)
        ax.set_xlim(8.5, 11.5);  ax.set_ylim(*ylim)
        ax.legend(fontsize=8);   ax.grid(True, alpha=0.3)

    axes[2].axhline(TDEP_OBS_GYR, color='k', ls='--', lw=1.5, label='Observed ~2 Gyr')
    axes[2].axhspan(1.0, 3.0, color='grey', alpha=0.15)
    axes[2].set_xlabel(r'$\log(M_*/M_\odot)$')
    axes[2].set_ylabel(r'$\tau_{\rm dep} = M_{\rm H2}/{\rm SFR}$ [Gyr]')
    axes[2].set_xlim(8.5, 11.5);  axes[2].set_ylim(0, 12)
    axes[2].legend(fontsize=8);   axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = './output/h2_validation_comparison.pdf'
    os.makedirs('./output', exist_ok=True)
    plt.savefig(outpath, bbox_inches='tight')
    print(f"\nPlot saved → {outpath}")
    plt.close()


if __name__ == '__main__':
    main()

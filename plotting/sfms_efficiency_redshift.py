#!/usr/bin/env python
"""
SAGE26 Star-Forming Main Sequence — coloured by SFR efficiency (H2SFRMode=2)
=============================================================================
8 redshift panels of the SFMS (SFR_disk vs M_star), scatter points coloured
by log10(ε_eff = SFR × τ_dyn / M_H2). Helps diagnose why ε declines at high z:
is it the sequence normalisation rising, disk sizes shrinking, or both?

Same starburst filter and efficiency definition as sfr_efficiency_redshift.py.

Usage:
    python plotting/sfms_efficiency_redshift.py
    python plotting/sfms_efficiency_redshift.py ./output/myrun/
"""

import sys, os, glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings("ignore")
import h5py as h5

# ─── Configuration ────────────────────────────────────────────────────────────

MODEL_DIR        = sys.argv[1] if len(sys.argv) > 1 else './output/microuchuu/'
OUT_FILE         = os.path.join(MODEL_DIR, 'plots', 'sfms_efficiency_redshift.pdf')
TARGET_REDSHIFTS = [0, 0.5, 1, 2, 4, 6, 8, 12]
MIN_PARTICLES    = 20
DILUTE           = 7500
SEED             = 42

_MSUN_CGS      = 1.989e33
_MPC_KMS_TO_YR = 3.0857e19 / 3.1557e7   # Mpc/(km/s) → yr

EFF_VMIN = np.log10(0.5)   # 0.5%
EFF_VMAX = 2.0             # 100%

# ─── HDF5 helpers ─────────────────────────────────────────────────────────────

_MASS_PROPS = frozenset({'StellarMass', 'H2gas', 'ColdGas'})

def _find_files(directory):
    files = sorted(glob.glob(os.path.join(directory, 'model_*.hdf5')))
    if not files:
        single = os.path.join(directory, 'model_0.hdf5')
        if os.path.exists(single):
            files = [single]
    return files

def _read_header(directory):
    files = _find_files(directory)
    if not files:
        return None
    with h5.File(files[0], 'r') as f:
        sim     = f['Header/Simulation']
        runtime = f['Header/Runtime']
        return {
            'hubble_h':       float(sim.attrs['hubble_h']),
            'unit_mass_in_g': float(runtime.attrs['UnitMass_in_g']),
            'redshifts':      np.array(f['Header/snapshot_redshifts'][:]),
        }

def load_snap(files, snap_key, props, mass_convert):
    chunks = {p: [] for p in props + ['Len']}
    for fp in files:
        try:
            with h5.File(fp, 'r') as f:
                if snap_key not in f:
                    continue
                grp = f[snap_key]
                for p in props + ['Len']:
                    if p in grp:
                        chunks[p].append(np.array(grp[p]))
        except Exception as e:
            print(f"  Warning reading {fp}: {e}")

    data = {}
    for p in props + ['Len']:
        if chunks[p]:
            arr = np.concatenate(chunks[p])
            data[p] = arr * mass_convert if p in _MASS_PROPS else arr

    if 'Len' in data:
        mask = data['Len'] >= MIN_PARTICLES
        data = {p: a[mask] for p, a in data.items()}
        data.pop('Len', None)
    return data

# ─── Load header ──────────────────────────────────────────────────────────────

hdr = _read_header(MODEL_DIR)
if hdr is None:
    sys.exit(f"ERROR: no model files found in {MODEL_DIR}")

HUBBLE_H     = hdr['hubble_h']
redshifts    = hdr['redshifts']
mass_convert = hdr['unit_mass_in_g'] / _MSUN_CGS / HUBBLE_H
files        = _find_files(MODEL_DIR)

props = ['StellarMass', 'H2gas', 'ColdGas', 'SfrDisk', 'SfrBulge', 'DiskRadius', 'Vvir']

# ─── Figure ───────────────────────────────────────────────────────────────────

plt.rcParams['font.size']        = 11
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor']   = 'white'
plt.rcParams['axes.edgecolor']   = 'black'
plt.rcParams['xtick.color']      = 'black'
plt.rcParams['ytick.color']      = 'black'
plt.rcParams['axes.labelcolor']  = 'black'

cmap = plt.cm.plasma
norm = mcolors.Normalize(vmin=EFF_VMIN, vmax=EFF_VMAX)

fig, axes = plt.subplots(2, 4, figsize=(16, 8),
                         sharex=True, sharey=True,
                         gridspec_kw={'hspace': 0.12, 'wspace': 0.06})
axes_flat = axes.flatten()

rng = np.random.default_rng(SEED)

# ─── Panels ───────────────────────────────────────────────────────────────────

for i, z_target in enumerate(TARGET_REDSHIFTS):
    ax = axes_flat[i]

    snap_nr  = int(np.argmin(np.abs(redshifts - z_target)))
    snap_z   = redshifts[snap_nr]
    snap_key = f'Snap_{snap_nr}'

    print(f"Panel {i+1}/{len(TARGET_REDSHIFTS)}: z_target={z_target},"
          f"  snap={snap_nr},  z_actual={snap_z:.3f}", flush=True)

    d = load_snap(files, snap_key, props, mass_convert)
    if not d or 'StellarMass' not in d:
        ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                ha='center', va='center', color='gray')
        ax.set_title(f'z = {snap_z:.2f}')
        continue

    Mstar    = d['StellarMass']
    H2       = np.maximum(d.get('H2gas',   np.zeros(len(Mstar))), 0.0)
    ColdGas  = np.maximum(d.get('ColdGas', np.zeros(len(Mstar))), 0.0)
    gas      = np.where(H2 > 0, H2, ColdGas)
    SfrDisk  = np.maximum(d['SfrDisk'],  0.0)
    SfrBulge = np.maximum(d['SfrBulge'], 0.0)
    SFR_tot  = SfrDisk + SfrBulge

    DiskR   = np.maximum(d['DiskRadius'], 0.0)
    Vvir_v  = np.maximum(d['Vvir'], 1.0)
    tau_dyn = 3.0 * DiskR / HUBBLE_H * _MPC_KMS_TO_YR / Vvir_v  # yr

    with np.errstate(divide='ignore', invalid='ignore'):
        eff = np.where(
            (gas > 0) & (SFR_tot > 0) & (Mstar > 0) & (DiskR > 0),
            SFR_tot * tau_dyn / gas * 100.0,
            np.nan
        )

    valid = (Mstar > 0) & (SFR_tot > 0) & np.isfinite(eff) & (eff > 0)
    lM    = np.log10(Mstar[valid])
    lSFR  = np.log10(SFR_tot[valid])
    lEff  = np.log10(eff[valid])
    n     = len(lM)

    # Dilute
    if n > DILUTE:
        idx  = rng.choice(n, DILUTE, replace=False)
        lM_p, lSFR_p, lEff_p = lM[idx], lSFR[idx], lEff[idx]
    else:
        lM_p, lSFR_p, lEff_p = lM, lSFR, lEff

    sc = ax.scatter(lM_p, lSFR_p, c=lEff_p, s=2, alpha=0.6,
                    cmap=cmap, norm=norm, rasterized=True, linewidths=0)

    # Running median
    if n > 100:
        bin_edges = np.linspace(np.percentile(lM, 2), np.percentile(lM, 98), 20)
        bin_idx   = np.digitize(lM, bin_edges)
        med_x, med_y = [], []
        for b in range(1, len(bin_edges)):
            sel = bin_idx == b
            if sel.sum() >= 10:
                med_x.append(0.5 * (bin_edges[b - 1] + bin_edges[b]))
                med_y.append(np.median(lSFR[sel]))
        if len(med_x) > 2:
            ax.plot(med_x, med_y, color='white', lw=2.5, zorder=5,
                    solid_capstyle='round')
            ax.plot(med_x, med_y, color='black',  lw=1.5, zorder=6,
                    solid_capstyle='round')

    ax.set_title(f'z = {snap_z:.2f}', fontsize=11, pad=4)
    print(f"  N = {n:,}  median log SFR = {np.nanmedian(lSFR):.2f}"
          f"  median ε_dyn = {10**np.nanmedian(lEff):.2f}%")

# ─── Axes formatting ──────────────────────────────────────────────────────────

for ax in axes_flat:
    ax.set_xlim(7.5, 12.5)
    ax.set_ylim(-5, 3)
    ax.grid(False)

for ax in axes[1]:
    ax.set_xlabel(r'log$_{10}$(M$_\star$/M$_\odot$)', fontsize=13)

fig.supylabel(r'log$_{10}$(SFR / M$_\odot$ yr$^{-1}$)',
              x=0.06, fontsize=12)

# ─── Shared colourbar ─────────────────────────────────────────────────────────

cbar_ax = fig.add_axes([0.92, 0.12, 0.015, 0.76])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label(r'$\varepsilon_\mathrm{eff} = \tau_\mathrm{dyn}\,/\,\tau_\mathrm{dep}$', fontsize=11)
cbar.set_ticks([np.log10(0.5), np.log10(1), np.log10(3), np.log10(10), np.log10(100)])
cbar.set_ticklabels(['0.5%', '1%', '3%', '10%', '100%'])

os.makedirs(os.path.join(MODEL_DIR, 'plots'), exist_ok=True)
fig.savefig(OUT_FILE, bbox_inches='tight', dpi=150)
print(f"\nSaved → {OUT_FILE}")
plt.close()

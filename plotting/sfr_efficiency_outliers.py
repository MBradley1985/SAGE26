#!/usr/bin/env python
"""
Diagnostic: ε > 20% non-FFB outliers at z=0
=============================================
Compares the high-efficiency non-FFB population against normal SF galaxies
to understand what drives their anomalously high ε_eff at low redshift.

Panels:
  1. Galaxy Type fraction (central / satellite / orphan)
  2. Halo mass (Mvir) distribution
  3. Stellar mass distribution
  4. DiskRadius vs StellarMass — spot anomalous disk sizes
  5. SFR vs M_H2 — K-S plane, coloured by population
  6. τ_dyn vs τ_dep — the two timescales driving ε

Usage:
    python plotting/sfr_efficiency_outliers.py
    python plotting/sfr_efficiency_outliers.py ./output/myrun/
"""

import sys, os, glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import warnings
warnings.filterwarnings("ignore")
import h5py as h5

# ─── Configuration ────────────────────────────────────────────────────────────

MODEL_DIR   = sys.argv[1] if len(sys.argv) > 1 else './output/microuchuu/'
OUT_FILE    = os.path.join(MODEL_DIR, 'plots', 'sfr_efficiency_outliers.pdf')
TARGET_Z    = 0.0
EFF_THRESH  = 20.0     # % — threshold defining "outlier"
MIN_PARTICLES = 20

_MSUN_CGS      = 1.989e33
_MPC_KMS_TO_YR = 9.779e11

# ─── HDF5 helpers ─────────────────────────────────────────────────────────────

_MASS_PROPS = frozenset({'StellarMass', 'BulgeMass', 'H2gas', 'ColdGas', 'Mvir'})

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

def load_snap(directory, snap_key, props):
    files        = _find_files(directory)
    hdr          = _read_header(directory)
    mass_convert = hdr['unit_mass_in_g'] / _MSUN_CGS / hdr['hubble_h']

    chunks = {p: [] for p in props + ['Len']}
    for fp in files:
        with h5.File(fp, 'r') as f:
            if snap_key not in f:
                continue
            grp = f[snap_key]
            for p in props + ['Len']:
                if p in grp:
                    chunks[p].append(np.array(grp[p]))

    data = {}
    for p in props + ['Len']:
        if chunks[p]:
            arr = np.concatenate(chunks[p])
            data[p] = arr * mass_convert if p in _MASS_PROPS else arr

    if 'Len' in data:
        mask = data['Len'] >= MIN_PARTICLES
        data = {p: a[mask] for p, a in data.items()}
        data.pop('Len', None)
    return data, hdr

# ─── Load z=0 snapshot ────────────────────────────────────────────────────────

hdr      = _read_header(MODEL_DIR)
redshifts = hdr['redshifts']
snap_nr  = int(np.argmin(np.abs(redshifts - TARGET_Z)))
snap_z   = redshifts[snap_nr]
snap_key = f'Snap_{snap_nr}'
print(f"Loading {snap_key}  (z = {snap_z:.3f})")

props = ['StellarMass', 'BulgeMass', 'H2gas', 'ColdGas', 'Mvir',
         'SfrDisk', 'SfrBulge', 'DiskRadius', 'Vvir', 'Type', 'FFBRegime']
d, hdr = load_snap(MODEL_DIR, snap_key, props)
HUBBLE_H = hdr['hubble_h']

# ─── Derived quantities ───────────────────────────────────────────────────────

Mstar    = d['StellarMass']
H2       = np.maximum(d['H2gas'], 0.0)
SfrDisk  = np.maximum(d['SfrDisk'], 0.0)
SfrBulge = np.maximum(d['SfrBulge'], 0.0)
SFR_tot  = SfrDisk + SfrBulge
Mvir     = d['Mvir']
Type     = d['Type']
r_disk_phys = d['DiskRadius'] / HUBBLE_H   # Mpc

with np.errstate(divide='ignore', invalid='ignore'):
    tau_dyn_yr = np.where(
        (d['Vvir'] > 0) & (r_disk_phys > 0),
        3.0 * r_disk_phys * _MPC_KMS_TO_YR / d['Vvir'],
        np.nan
    )
    eff = np.where(
        (H2 > 0) & (SFR_tot > 0) & (Mstar > 0) & np.isfinite(tau_dyn_yr),
        SFR_tot * tau_dyn_yr / H2 * 100.0,
        np.nan
    )
    tau_dep_yr = np.where(SFR_tot > 0, H2 / SFR_tot, np.nan)

# all SF galaxies — no starburst filter, that's what we're diagnosing
sf = (Mstar > 0) & (SFR_tot > 0) & (H2 > 0) & np.isfinite(eff) & (eff > 0)

ffb_flag  = d['FFBRegime'][sf] > 0
high_eff  = eff[sf] > EFF_THRESH

# Three populations
norm_mask  = sf & ~(eff > EFF_THRESH)
outl_mask  = sf & (eff > EFF_THRESH) & ~(d['FFBRegime'] > 0)
ffbr_mask  = sf & (eff > EFF_THRESH) &  (d['FFBRegime'] > 0)

def pop(mask, key):
    return d[key][mask]

print(f"  Normal SF:         {norm_mask.sum():>7,}")
print(f"  ε>20% + FFBRegime: {ffbr_mask.sum():>7,}")
print(f"  ε>20%, non-FFB:    {outl_mask.sum():>7,}  ← outliers")

# ─── Stats printout ───────────────────────────────────────────────────────────

def pct(arr, label, fmt='.2f', unit=''):
    p16, p50, p84 = np.nanpercentile(arr, [16, 50, 84])
    print(f"    {label:<28s}  {p16:{fmt}}  {p50:{fmt}}  {p84:{fmt}}  {unit}")

r_disk_kpc    = d['DiskRadius'] / HUBBLE_H * 1e3   # kpc
tau_dyn_Myr   = tau_dyn_yr / 1e6
tau_dep_Gyr   = tau_dep_yr / 1e9
disk_to_mvir  = r_disk_kpc / (d['Mvir'] ** (1/3))  # size–mass offset proxy

for mask, name in [(norm_mask, 'Normal SF'), (outl_mask, 'ε>20% non-FFB')]:
    print(f"\n{'─'*60}")
    print(f"  {name}  (n={mask.sum():,})")
    print(f"  {'Property':<28s}  {'p16':>8}  {'p50':>8}  {'p84':>8}  unit")
    print(f"  {'─'*62}")

    types = d['Type'][mask]
    t0 = (types == 0).sum() / mask.sum() * 100
    t1 = (types == 1).sum() / mask.sum() * 100
    t2 = (types == 2).sum() / mask.sum() * 100
    print(f"    {'Type 0/1/2 fractions':<28s}  {t0:.1f}%  {t1:.1f}%  {t2:.1f}%")

    pct(np.log10(d['Mvir'][mask]),           'log Mvir [Msun]',      unit='dex')
    pct(np.log10(d['StellarMass'][mask]),     'log Mstar [Msun]',     unit='dex')
    pct(np.log10(d['H2gas'][mask]),           'log M_H2 [Msun]',      unit='dex')
    pct(np.log10(SFR_tot[mask]),              'log SFR [Msun/yr]',    unit='dex')
    pct(r_disk_kpc[mask],                     'DiskRadius [kpc]',     fmt='.1f', unit='kpc')
    pct(d['Vvir'][mask],                      'Vvir [km/s]',          fmt='.1f', unit='km/s')
    pct(tau_dyn_Myr[mask],                    'τ_dyn [Myr]',          fmt='.0f', unit='Myr')
    pct(tau_dep_Gyr[mask],                    'τ_dep [Gyr]',          fmt='.2f', unit='Gyr')
    pct(eff[mask],                            'ε_eff [%]',            fmt='.1f', unit='%')
    pct(d['BulgeMass'][mask]/d['StellarMass'][mask], 'B/T ratio',     fmt='.2f')

print(f"\n{'─'*60}")

# ─── Figure ───────────────────────────────────────────────────────────────────

plt.rcParams['font.size']        = 11
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor']   = 'white'
plt.rcParams['axes.edgecolor']   = 'black'

fig, axes = plt.subplots(2, 3, figsize=(14, 8),
                         gridspec_kw={'hspace': 0.35, 'wspace': 0.30})
axes_flat = axes.flatten()

c_norm = 'steelblue'
c_out  = 'goldenrod'
c_ffb  = 'firebrick'
kw_norm = dict(histtype='step', density=True, lw=1.5, color=c_norm)
kw_out  = dict(histtype='stepfilled', density=True, lw=1.5,
               color=c_out, alpha=0.6, edgecolor='darkorange')

# ── Panel 1: Galaxy Type bar chart ────────────────────────────────────────────
ax = axes_flat[0]
type_labels = ['Central\n(Type 0)', 'Satellite\n(Type 1)', 'Orphan\n(Type 2)']
x = np.arange(3)
width = 0.35

for i, (mask, label, color) in enumerate([
    (norm_mask, 'Normal SF', c_norm),
    (outl_mask, f'ε>{EFF_THRESH:.0f}%, non-FFB', c_out),
]):
    types = d['Type'][mask]
    counts = np.array([(types == t).sum() for t in [0, 1, 2]], dtype=float)
    fracs  = counts / counts.sum() * 100
    ax.bar(x + (i - 0.5) * width, fracs, width, label=label, color=color,
           edgecolor='k', linewidth=0.5, alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(type_labels)
ax.set_ylabel('Fraction (%)')
ax.set_title('Galaxy type')
ax.legend(fontsize=8, frameon=False)

# ── Panel 2: Halo mass Mvir ───────────────────────────────────────────────────
ax = axes_flat[1]
bins = np.linspace(8, 15, 40)
ax.hist(np.log10(pop(norm_mask, 'Mvir')), bins=bins, **kw_norm)
ax.hist(np.log10(pop(outl_mask, 'Mvir')), bins=bins, **kw_out)
ax.set_xlabel(r'log$_{10}$(M$_\mathrm{vir}$/M$_\odot$)')
ax.set_ylabel('Density')
ax.set_title('Halo mass')

# ── Panel 3: Stellar mass ─────────────────────────────────────────────────────
ax = axes_flat[2]
bins = np.linspace(6, 13, 40)
ax.hist(np.log10(pop(norm_mask, 'StellarMass')), bins=bins, **kw_norm)
ax.hist(np.log10(pop(outl_mask, 'StellarMass')), bins=bins, **kw_out)
ax.set_xlabel(r'log$_{10}$(M$_\star$/M$_\odot$)')
ax.set_ylabel('Density')
ax.set_title('Stellar mass')

# ── Panel 4: DiskRadius vs StellarMass ────────────────────────────────────────
ax = axes_flat[3]
# sample normal for visibility
rng = np.random.default_rng(42)
n_norm_tot = norm_mask.sum()
idx = rng.choice(n_norm_tot, min(5000, n_norm_tot), replace=False)
lM_norm  = np.log10(d['StellarMass'][norm_mask][idx])
lR_norm  = np.log10(d['DiskRadius'][norm_mask][idx] / HUBBLE_H * 1e3)  # kpc
lM_out   = np.log10(d['StellarMass'][outl_mask])
lR_out   = np.log10(d['DiskRadius'][outl_mask] / HUBBLE_H * 1e3)

ax.scatter(lM_norm, lR_norm, s=1, alpha=0.2, color=c_norm,
           rasterized=True, linewidths=0)
ax.scatter(lM_out,  lR_out,  s=8, alpha=0.7, color=c_out,
           rasterized=True, linewidths=0, zorder=3)
ax.set_xlabel(r'log$_{10}$(M$_\star$/M$_\odot$)')
ax.set_ylabel(r'log$_{10}$(R$_\mathrm{disk}$ / kpc)')
ax.set_title('Disk scale radius')

# ── Panel 5: SFR vs M_H2 (K-S plane) ─────────────────────────────────────────
ax = axes_flat[4]
idx2 = rng.choice(n_norm_tot, min(5000, n_norm_tot), replace=False)
ax.scatter(np.log10(d['H2gas'][norm_mask][idx2]),
           np.log10(SFR_tot[norm_mask][idx2]),
           s=1, alpha=0.2, color=c_norm, rasterized=True, linewidths=0)
ax.scatter(np.log10(d['H2gas'][outl_mask]),
           np.log10(SFR_tot[outl_mask]),
           s=8, alpha=0.7, color=c_out, rasterized=True, linewidths=0, zorder=3)
ax.set_xlabel(r'log$_{10}$(M$_{\mathrm{H_2}}$/M$_\odot$)')
ax.set_ylabel(r'log$_{10}$(SFR / M$_\odot$ yr$^{-1}$)')
ax.set_title('SFR vs H₂ mass')

# ── Panel 6: τ_dyn vs τ_dep ───────────────────────────────────────────────────
ax = axes_flat[5]
lTdyn_norm = np.log10(tau_dyn_yr[norm_mask][idx2] / 1e6)   # Myr
lTdep_norm = np.log10(tau_dep_yr[norm_mask][idx2] / 1e9)   # Gyr
lTdyn_out  = np.log10(tau_dyn_yr[outl_mask] / 1e6)
lTdep_out  = np.log10(tau_dep_yr[outl_mask] / 1e9)

ax.scatter(lTdep_norm, lTdyn_norm, s=1, alpha=0.2, color=c_norm,
           rasterized=True, linewidths=0)
ax.scatter(lTdep_out,  lTdyn_out,  s=8, alpha=0.7, color=c_out,
           rasterized=True, linewidths=0, zorder=3)

# Diagonal lines of constant ε
for eff_val, ls in [(3, '--'), (20, '-')]:
    # log10(τ_dyn [Myr]) = log10(ε/100 × τ_dep [Gyr] × 1e9/1e6)
    # = log10(ε/100) + log10(τ_dep [Gyr]) + 3
    x_line = np.array([-2, 2])
    y_line = np.log10(eff_val / 100) + x_line + 3
    ax.plot(x_line, y_line, ls=ls, lw=1, color='gray', alpha=0.6,
            label=f'ε={eff_val}%')

ax.set_xlabel(r'log$_{10}$($\tau_\mathrm{dep}$ / Gyr)')
ax.set_ylabel(r'log$_{10}$($\tau_\mathrm{dyn}$ / Myr)')
ax.set_title(r'Depletion vs dynamical time')
ax.legend(fontsize=8, frameon=False)

# ─── Legend patches ───────────────────────────────────────────────────────────
from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor=c_norm, edgecolor='k', lw=0.5, label='Normal SF'),
    Patch(facecolor=c_out,  edgecolor='darkorange', lw=0.5,
          label=f'ε > {EFF_THRESH:.0f}%, non-FFB  (n={outl_mask.sum():,})'),
]
fig.legend(handles=legend_handles, loc='upper center', ncol=2,
           fontsize=10, frameon=False, bbox_to_anchor=(0.5, 1.01))

os.makedirs(os.path.join(MODEL_DIR, 'plots'), exist_ok=True)
fig.savefig(OUT_FILE, bbox_inches='tight', dpi=150)
print(f"\nSaved → {OUT_FILE}")
plt.close()

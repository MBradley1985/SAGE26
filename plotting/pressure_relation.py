#!/usr/bin/env python
"""
SAGE26 Pressure Relation
========================
R_mol = Sigma_H2 / Sigma_HI as a function of midplane pressure P_ext/k,
following the Blitz & Rosolowsky (2006) framework.

Midplane pressure:
    P_ext/k = (pi G / 2k) * Sigma_gas * (Sigma_gas + vel_ratio * Sigma_disk_star)
            ~ 33.1 * Sigma_gas * (Sigma_gas + 0.1 * Sigma_disk_star)   [K cm^-3]
    where Sigma values are in M_sun/pc^2.

Usage:
    python plotting/pressure_relation.py
    python plotting/pressure_relation.py ./output/myrun/
"""

import sys, os, glob
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import h5py as h5

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MODEL_DIR       = sys.argv[1] if len(sys.argv) > 1 else './output/millennium/'
OUT_FILE        = os.path.join(MODEL_DIR, 'plots', 'pressure_relation.pdf')

TARGET_REDSHIFT = 0.0
MIN_PARTICLES   = 20
DILUTE          = 8000
SEED            = 42
_MSUN_CGS       = 1.989e33

# Blitz & Rosolowsky (2006) parameters
BR06_P0_K  = 4.3e4   # K cm^-3, characteristic pressure
BR06_ALPHA = 0.92    # power-law slope  R_mol = (P/P0)^alpha
VEL_RATIO  = 0.1     # sigma_gas / sigma_star

# Pressure conversion factor: (pi G / 2k) * (M_sun/pc^2 -> g/cm^2)^2
# = pi * 6.674e-8 / (2 * 1.381e-16) * (1.989e33 / 3.086e18^2)^2 = 33.1
_P_FACTOR  = 33.1    # [(K cm^-3) / (M_sun/pc^2)^2]

# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────

_MASS_PROPS = frozenset({
    'StellarMass', 'BulgeMass', 'ColdGas', 'H2gas', 'H1gas',
})

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
    try:
        with h5.File(files[0], 'r') as f:
            sim     = f['Header/Simulation']
            runtime = f['Header/Runtime']
            hdr = {
                'hubble_h':       float(sim.attrs['hubble_h']),
                'unit_mass_in_g': float(runtime.attrs['UnitMass_in_g']),
                'last_snap':      int(sim.attrs['LastSnapshotNr']),
                'redshifts':      list(f['Header/snapshot_redshifts'][:]),
            }
        vol_frac = sum(
            float(h5.File(fp, 'r')['Header/Runtime'].attrs['frac_volume_processed'])
            for fp in files
        )
        hdr['volume_fraction'] = vol_frac
        return hdr
    except Exception as e:
        print(f"  Header read failed: {e}")
        return None

def load_snap(directory, snap_key, props):
    files = _find_files(directory)
    if not files:
        return {}
    hdr = _read_header(directory)
    mass_convert = hdr['unit_mass_in_g'] / _MSUN_CGS / hdr['hubble_h'] if hdr else 1e10
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

# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────

hdr = _read_header(MODEL_DIR)
if hdr is None:
    sys.exit(f"ERROR: no model files in {MODEL_DIR}")

HUBBLE_H  = hdr['hubble_h']
redshifts = np.array(hdr['redshifts'])
SNAP_NR   = int(np.argmin(np.abs(redshifts - TARGET_REDSHIFT)))
SNAP_KEY  = f'Snap_{SNAP_NR}'
SNAP_Z    = redshifts[SNAP_NR]

print(f"Model:  {MODEL_DIR}")
print(f"Snap:   {SNAP_KEY}  (z ≈ {SNAP_Z:.3f},  requested z={TARGET_REDSHIFT})")

props = ['StellarMass', 'BulgeMass', 'ColdGas', 'H2gas', 'H1gas', 'DiskRadius']
d = load_snap(MODEL_DIR, SNAP_KEY, props)
if not d:
    sys.exit(f"ERROR: snapshot {SNAP_KEY} not found")
print(f"Loaded: {len(d['StellarMass']):,} galaxies after resolution cut")

# ─────────────────────────────────────────────────────────────────────────────
# Derived quantities
# ─────────────────────────────────────────────────────────────────────────────

Mstar  = d['StellarMass']
Mbulge = d['BulgeMass']
Mgas   = d['ColdGas']
H2     = np.maximum(d['H2gas'], 0.0)
H1     = np.maximum(d['H1gas'], 0.0)

# DiskRadius = exponential scale length r_s in Mpc/h (from paper_plots.py convention)
r_s_pc = d['DiskRadius'] / HUBBLE_H * 1e6   # Mpc/h → pc

# Disk stellar mass (exclude bulge)
M_disk_star = np.maximum(Mstar - Mbulge, 0.0)

# Central surface density convention: Sigma_0 = M / (2*pi*r_s^2)
# consistent with SAGE's internal radial integration (exponential disk)
area = 2.0 * np.pi * r_s_pc**2   # pc^2

# Valid galaxies: resolved disk, gas and both HI/H2 present
good = (r_s_pc > 0) & (Mgas > 0) & (H2 > 0) & (H1 > 0)

Sigma_gas  = np.where(good, Mgas / area,        np.nan)   # M_sun/pc^2
Sigma_star = np.where(good, M_disk_star / area, np.nan)
Sigma_H2   = np.where(good, H2 / area,          np.nan)
Sigma_HI   = np.where(good, H1 / area,          np.nan)

# Midplane pressure  [K cm^-3]
P_over_k = np.where(good,
    _P_FACTOR * Sigma_gas * (Sigma_gas + VEL_RATIO * Sigma_star),
    np.nan)

R_mol = np.where(good & (H1 > 0), H2 / H1, np.nan)     # Sigma_H2 / Sigma_HI
f_mol = np.where(good, H2 / (H2 + H1),       np.nan)

# log quantities
def lg(x):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(x > 0, np.log10(x), np.nan)

lP    = lg(P_over_k)
lRmol = lg(R_mol)
lfmol = lg(f_mol)
lSg   = lg(Sigma_gas)
lSh2  = lg(Sigma_H2)
lShi  = lg(Sigma_HI)
lMs   = lg(Mstar)

# Diagnostics
valid = np.isfinite(lP) & np.isfinite(lRmol)
print(f"\n  Valid galaxies (Σ and R_mol finite): {valid.sum():,}")
for label, arr in [("log(P/k)",      lP),
                   ("log(R_mol)",    lRmol),
                   ("log(Σ_gas)",    lSg),
                   ("log(Σ_H2)",     lSh2)]:
    fin = arr[np.isfinite(arr)]
    p = np.percentile(fin, [2, 16, 50, 84, 98])
    print(f"  {label:<14s}  p[2,16,50,84,98] = "
          f"[{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}, {p[3]:.2f}, {p[4]:.2f}]")

# ─────────────────────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────────────────────

try:
    plt.style.use('./plotting/kieren_cohare_palatino_sty.mplstyle')
except:
    plt.rcParams.update({'font.size': 9, 'axes.labelsize': 10,
                         'legend.fontsize': 8, 'lines.linewidth': 1.5})

MODEL_COLOR = '#2166ac'
MODEL_LABEL = os.path.basename(os.path.normpath(MODEL_DIR))

def scatter_dilute(ax, x, y, c=None, cmap='plasma_r', **kw):
    ok = np.isfinite(x) & np.isfinite(y)
    if c is not None:
        ok &= np.isfinite(c)
    xi, yi = x[ok], y[ok]
    ci = c[ok] if c is not None else None
    np.random.seed(SEED)
    if len(xi) > DILUTE:
        idx = np.random.choice(len(xi), DILUTE, replace=False)
        xi, yi = xi[idx], yi[idx]
        if ci is not None:
            ci = ci[idx]
    if ci is not None:
        return ax.scatter(xi, yi, c=ci, s=2, alpha=0.35, cmap=cmap,
                          rasterized=True, zorder=2, **kw)
    ax.scatter(xi, yi, s=2, alpha=0.25, color=MODEL_COLOR,
               rasterized=True, zorder=2)
    return None

def running_median(x, y, bins):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    xc, lo, med, hi = [], [], [], []
    for i in range(len(bins)-1):
        m = (x >= bins[i]) & (x < bins[i+1])
        if m.sum() >= 10:
            p = np.percentile(y[m], [16, 50, 84])
            xc.append(0.5*(bins[i]+bins[i+1]))
            lo.append(p[0]); med.append(p[1]); hi.append(p[2])
    return np.array(xc), np.array(lo), np.array(med), np.array(hi)

def add_colorbar(fig, ax, sc, label):
    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.04)
    cb.set_label(label, fontsize=7)
    cb.ax.tick_params(labelsize=6)

# BR06 reference curves
P_ref  = np.logspace(2, 7.5, 300)
R_ref  = (P_ref / BR06_P0_K) ** BR06_ALPHA
f_ref  = R_ref / (1.0 + R_ref)

bins_p  = np.arange(2.0, 8.0, 0.3)
bins_sg = np.arange(-2.0, 4.0, 0.25)

# ─────────────────────────────────────────────────────────────────────────────
# Figure  (2 × 2)
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(10, 9))
fig.subplots_adjust(hspace=0.38, wspace=0.35)

# ── Panel 1: R_mol vs P/k  ───────────────────────────────────────────────────
ax = axes[0, 0]
sc = scatter_dilute(ax, lP, lRmol, c=lMs)
if sc is not None:
    add_colorbar(fig, ax, sc, r'$\log M_\star$')

xc, lo, med, hi = running_median(lP, lRmol, bins_p)
ax.plot(xc, med, '-', color=MODEL_COLOR, lw=2, label=MODEL_LABEL, zorder=5)
ax.fill_between(xc, lo, hi, alpha=0.25, color=MODEL_COLOR, zorder=4)
ax.plot(np.log10(P_ref), np.log10(R_ref), 'k--', lw=1.5,
        label=fr'BR06  $\alpha={BR06_ALPHA}$,  $P_0/k={BR06_P0_K:.0e}$', zorder=6)
ax.axhline(0, color='grey', lw=0.7, ls=':')   # R_mol = 1

ax.set_xlabel(r'$\log_{10}(P_\mathrm{ext}/k\ [\mathrm{K\,cm}^{-3}])$')
ax.set_ylabel(r'$\log_{10}(\Sigma_\mathrm{H_2}/\Sigma_\mathrm{HI})$')
ax.set_xlim(2, 7.5); ax.set_ylim(-3, 2)
ax.legend(fontsize=7)
ax.set_title(r'$R_\mathrm{mol}$ vs midplane pressure')

# ── Panel 2: f_mol vs P/k  ───────────────────────────────────────────────────
ax = axes[0, 1]
sc = scatter_dilute(ax, lP, lfmol, c=lMs)
if sc is not None:
    add_colorbar(fig, ax, sc, r'$\log M_\star$')

xc, lo, med, hi = running_median(lP, lfmol, bins_p)
ax.plot(xc, med, '-', color=MODEL_COLOR, lw=2, label=MODEL_LABEL, zorder=5)
ax.fill_between(xc, lo, hi, alpha=0.25, color=MODEL_COLOR, zorder=4)
ax.plot(np.log10(P_ref), np.log10(f_ref), 'k--', lw=1.5, label='BR06', zorder=6)
ax.axhline(0, color='grey', lw=0.7, ls=':')   # f_mol = 1

ax.set_xlabel(r'$\log_{10}(P_\mathrm{ext}/k\ [\mathrm{K\,cm}^{-3}])$')
ax.set_ylabel(r'$\log_{10}(f_\mathrm{mol})\ [H_2/(H_2+\mathrm{HI})]$')
ax.set_xlim(2, 7.5); ax.set_ylim(-4, 0.2)
ax.legend(fontsize=7)
ax.set_title(r'$f_\mathrm{mol}$ vs midplane pressure')

# ── Panel 3: R_mol vs Σ_gas  ─────────────────────────────────────────────────
ax = axes[1, 0]
sc = scatter_dilute(ax, lSg, lRmol, c=lMs)
if sc is not None:
    add_colorbar(fig, ax, sc, r'$\log M_\star$')

xc, lo, med, hi = running_median(lSg, lRmol, bins_sg)
ax.plot(xc, med, '-', color=MODEL_COLOR, lw=2, label=MODEL_LABEL, zorder=5)
ax.fill_between(xc, lo, hi, alpha=0.25, color=MODEL_COLOR, zorder=4)
ax.axhline(0, color='grey', lw=0.7, ls=':')

ax.set_xlabel(r'$\log_{10}(\Sigma_\mathrm{gas}\ [M_\odot\,\mathrm{pc}^{-2}])$')
ax.set_ylabel(r'$\log_{10}(\Sigma_\mathrm{H_2}/\Sigma_\mathrm{HI})$')
ax.set_xlim(-2, 4); ax.set_ylim(-3, 2)
ax.legend(fontsize=7)
ax.set_title(r'$R_\mathrm{mol}$ vs gas surface density')

# ── Panel 4: Σ_H2 vs Σ_HI  ───────────────────────────────────────────────────
ax = axes[1, 1]
sc = scatter_dilute(ax, lShi, lSh2, c=lMs)
if sc is not None:
    add_colorbar(fig, ax, sc, r'$\log M_\star$')

xc, lo, med, hi = running_median(lShi, lSh2, bins_sg)
ax.plot(xc, med, '-', color=MODEL_COLOR, lw=2, label=MODEL_LABEL, zorder=5)
ax.fill_between(xc, lo, hi, alpha=0.25, color=MODEL_COLOR, zorder=4)
xl = np.array([-2, 4])
ax.plot(xl, xl, 'k:', lw=1, label=r'$\Sigma_\mathrm{H_2} = \Sigma_\mathrm{HI}$')

ax.set_xlabel(r'$\log_{10}(\Sigma_\mathrm{HI}\ [M_\odot\,\mathrm{pc}^{-2}])$')
ax.set_ylabel(r'$\log_{10}(\Sigma_\mathrm{H_2}\ [M_\odot\,\mathrm{pc}^{-2}])$')
ax.set_xlim(-2, 4); ax.set_ylim(-4, 4)
ax.legend(fontsize=7)
ax.set_title(r'$\Sigma_\mathrm{H_2}$ vs $\Sigma_\mathrm{HI}$')

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────

fig.suptitle(
    f'{MODEL_LABEL}   —   {SNAP_KEY}  (z = {SNAP_Z:.3f})',
    fontsize=11, y=1.01
)
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
fig.savefig(OUT_FILE, dpi=150, bbox_inches='tight')
print(f"\nSaved → {OUT_FILE}")
plt.close(fig)

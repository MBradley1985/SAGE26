#!/usr/bin/env python
"""
SAGE26 Gas Diagnostics
======================
Quick-look plots for HI, H2 and cold gas to check the model is behaving
after changes to the H2 prescription.

Usage:
    python plotting/gas_diagnostics.py                     # primary dir
    python plotting/gas_diagnostics.py ./output/myrun/     # custom dir
"""

import sys, os, glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

import h5py as h5

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MODEL_DIR       = sys.argv[1] if len(sys.argv) > 1 else './output/millennium/'
OBS_DIR         = './data/'
OUT_FILE        = os.path.join(MODEL_DIR, 'plots', 'gas_diagnostics.pdf')

TARGET_REDSHIFT = 0.0    # desired redshift; nearest available snapshot is chosen
MIN_PARTICLES   = 20     # minimum halo length (number of DM particles)
DILUTE          = 7500   # max scatter points per panel
SEED            = 42
_MSUN_CGS       = 1.989e33

# ─────────────────────────────────────────────────────────────────────────────
# HDF5 I/O  (mirrors paper_plots.py pattern)
# ─────────────────────────────────────────────────────────────────────────────

_MASS_PROPS = frozenset({
    'StellarMass', 'BulgeMass', 'ColdGas', 'HotGas', 'H2gas', 'H1gas',
    'EjectedMass', 'MetalsColdGas', 'MetalsStellarMass',
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
                'box_size':       float(sim.attrs['box_size']),
                'unit_mass_in_g': float(runtime.attrs['UnitMass_in_g']),
                'last_snap':      int(sim.attrs['LastSnapshotNr']),
                'redshifts':      list(f['Header/snapshot_redshifts'][:]),
                'part_mass':      float(sim.attrs['particle_mass']),
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
        print(f"  No files in {directory}")
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
# Load model
# ─────────────────────────────────────────────────────────────────────────────

hdr = _read_header(MODEL_DIR)
if hdr is None:
    sys.exit(f"ERROR: no model files found in {MODEL_DIR}")

HUBBLE_H      = hdr['hubble_h']
BOX_SIZE      = hdr['box_size']
VOL_FRAC      = hdr['volume_fraction']
VOLUME        = (BOX_SIZE / HUBBLE_H)**3 * VOL_FRAC   # Mpc^3
MASS_CONVERT  = hdr['unit_mass_in_g'] / _MSUN_CGS / HUBBLE_H
redshifts     = np.array(hdr['redshifts'])
SNAP_NR       = int(np.argmin(np.abs(redshifts - TARGET_REDSHIFT)))
SNAP_KEY      = f'Snap_{SNAP_NR}'
SNAP_Z        = redshifts[SNAP_NR]

PART_MASS_MSUN = hdr['part_mass'] * MASS_CONVERT   # Msun per DM particle

print(f"Model:   {MODEL_DIR}")
print(f"Snap:    {SNAP_KEY}  (z ≈ {SNAP_Z:.3f},  requested z={TARGET_REDSHIFT})")
print(f"Volume:  {VOLUME:.1f} Mpc³  (frac={VOL_FRAC:.3f})")
print(f"Cut:     Len ≥ {MIN_PARTICLES} particles  (Mvir ≥ {MIN_PARTICLES * PART_MASS_MSUN:.3e} Msun)")

props = ['StellarMass', 'BulgeMass', 'ColdGas', 'H2gas', 'H1gas',
         'SfrDisk', 'SfrBulge', 'Vvir', 'DiskRadius', 'Type']
d = load_snap(MODEL_DIR, SNAP_KEY, props)
if not d:
    sys.exit(f"ERROR: snapshot {SNAP_KEY} (z={SNAP_Z:.3f}) not found in {MODEL_DIR}")
print(f"Loaded:  {len(d['StellarMass']):,} galaxies after resolution cut")

# ─────────────────────────────────────────────────────────────────────────────
# Derived quantities
# ─────────────────────────────────────────────────────────────────────────────

Mstar   = d['StellarMass']
ColdGas = d['ColdGas']
H2      = d['H2gas']
H1      = d['H1gas']
SFR     = d['SfrDisk'] + d['SfrBulge']

# Clamp negative values from float precision
H2  = np.maximum(H2,  0.0)
H1  = np.maximum(H1,  0.0)
SFR = np.maximum(SFR, 0.0)

# Molecular fraction and depletion time
# SFR is stored in M_sun/yr in the HDF5 (pre-converted, not code units)
# H2 is in M_sun (after mass_convert) → H2/SFR is in years
f_mol     = np.where(H2 + H1 > 0, H2 / (H2 + H1), 0.0)
t_dep_Gyr = np.where(SFR > 0, H2 / SFR / 1e9, np.nan)   # yr → Gyr

# sSFR in yr^-1 (SFR Msun/yr, Mstar Msun)
with np.errstate(divide='ignore', invalid='ignore'):
    sSFR = np.where((Mstar > 0) & (SFR > 0), SFR / Mstar, np.nan)

# log10 with safe guard
def lg(x):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(x > 0, np.log10(x), np.nan)

lMstar   = lg(Mstar)
lH2      = lg(H2)
lH1      = lg(H1)
lCold    = lg(ColdGas)
lSFR     = lg(SFR)
lsSFR    = lg(sSFR)

# Gas fractions (log)
lH1_Mstar   = lg(H1 / Mstar)
lH2_Mstar   = lg(H2 / Mstar)
lCold_Mstar = lg(ColdGas / Mstar)
lNeutral    = lg((H1 + H2) / Mstar)
lt_dep_Gyr  = lg(t_dep_Gyr)

# ─────────────────────────────────────────────────────────────────────────────
# Derived quantity diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def _stat(label, arr, unit=''):
    fin = arr[np.isfinite(arr)]
    if len(fin) == 0:
        print(f"  {label:<30s}  ALL NaN/Inf")
        return
    p = np.percentile(fin, [2, 16, 50, 84, 98])
    print(f"  {label:<30s}  N={len(fin):>8,}  "
          f"p[2,16,50,84,98] = [{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}, {p[3]:.2f}, {p[4]:.2f}]{unit}")

print("\n── Derived quantities (all galaxies after resolution cut) ──")
_stat("log10(Mstar/Msun)",        lMstar)
_stat("log10(H2/Msun)",           lH2)
_stat("log10(HI/Msun)",           lH1)
_stat("log10(ColdGas/Msun)",      lCold)
_stat("log10(SFR) [Msun/yr]",     lSFR)
_stat("log10(sSFR) [yr^-1]",      lsSFR)
_stat("log10(H2/Mstar)",          lH2_Mstar)
_stat("log10(HI/Mstar)",          lH1_Mstar)
_stat("log10(ColdGas/Mstar)",     lCold_Mstar)
_stat("log10(f_mol) H2/(H2+HI)", lg(f_mol))
_stat("log10(t_dep_H2) [Gyr]",   lt_dep_Gyr)

n_sfing = np.sum(SFR > 0)
n_h2    = np.sum(H2 > 0)
n_hi    = np.sum(H1 > 0)
n_cold  = np.sum(ColdGas > 0)
n_tot   = len(Mstar)
print(f"\n  SFR > 0:       {n_sfing:>8,}  ({100*n_sfing/n_tot:.1f}%)")
print(f"  H2  > 0:       {n_h2:>8,}  ({100*n_h2/n_tot:.1f}%)")
print(f"  HI  > 0:       {n_hi:>8,}  ({100*n_hi/n_tot:.1f}%)")
print(f"  ColdGas > 0:   {n_cold:>8,}  ({100*n_cold/n_tot:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# Mass function helper
# ─────────────────────────────────────────────────────────────────────────────

def mass_function(log_mass, bins):
    """Return bin centres and phi [Mpc^-3 dex^-1] for a log-mass array."""
    valid = np.isfinite(log_mass)
    counts, edges = np.histogram(log_mass[valid], bins=bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    dlogM   = edges[1] - edges[0]
    phi     = counts / (VOLUME * dlogM)
    # mask empty bins
    phi_masked = np.where(counts > 0, phi, np.nan)
    return centres, phi_masked


def running_median(x, y, bins, pct=(16, 50, 84)):
    """Percentiles of y in x bins. Returns (centres, low, med, hi)."""
    valid = np.isfinite(x) & np.isfinite(y)
    x, y  = x[valid], y[valid]
    centres, lo, med, hi = [], [], [], []
    for i in range(len(bins) - 1):
        mask = (x >= bins[i]) & (x < bins[i+1])
        if mask.sum() >= 5:
            p = np.percentile(y[mask], pct)
            centres.append(0.5 * (bins[i] + bins[i+1]))
            lo.append(p[0]); med.append(p[1]); hi.append(p[2])
    return (np.array(centres), np.array(lo),
            np.array(med), np.array(hi))


# ─────────────────────────────────────────────────────────────────────────────
# Load observational data  (all gracefully optional)
# ─────────────────────────────────────────────────────────────────────────────

def load_obs(path, **kwargs):
    full = os.path.join(OBS_DIR, path)
    if not os.path.exists(full):
        return None
    try:
        return np.loadtxt(full, comments='#', **kwargs)
    except Exception:
        return None

himf_jones   = load_obs('HIMF_Jones18.dat')         # log(MHI), log(phi), lo, hi
himf_zwaan   = load_obs('HIMF_Zwaan2005.dat')        # log(MHI), log(phi), err_up, err_dn
h2mf_dd      = load_obs('H2MF_Fletcher21_DetNonDet.dat')
h2mf_est     = load_obs('H2MF_Fletcher21_Estimated.dat')

rhi_brown15  = load_obs('Gas/RHI-Mstars_Brown15.dat')          # Mstar, MHI/Mstar (linear!), err
rhi_cal_ltg  = load_obs('Gas/RHI-Mstars_Callette18-LTGs.dat')  # Mstar, log(HI/Mstar), lo, hi
rhi_cal_etg  = load_obs('Gas/RHI-Mstars_Callette18-ETGs.dat')
rh2_cal_ltg  = load_obs('Gas/RH2-Mstars_Callette18-LTGs.dat')  # Mstar, log(H2/Mstar), lo, hi
rh2_cal_etg  = load_obs('Gas/RH2-Mstars_Callette18-ETGs.dat')
hi_xgass     = load_obs('Gas/HIGasRatio_NonDetEQUpperLimits.dat')  # Mstar, log(HI/Mstar), lo, hi
h2_xcg       = load_obs('Gas/MolecularGasRatio_NonDetEQUpperLimits.dat')
neutral_xcg  = load_obs('Gas/NeutralGasRatio_NonDetEQUpperLimits.dat')
hi_brown17   = load_obs('Gas/HIMstar_Brown17.dat')        # Mstar, log(HI/Mstar), lo, hi, flag
hissfr_b17   = load_obs('Gas/HISSFR_Brown17.dat')         # log(sSFR), log(HI/Mstar), lo, hi, flag

# ─────────────────────────────────────────────────────────────────────────────
# Style helpers
# ─────────────────────────────────────────────────────────────────────────────

try:
    plt.style.use('./plotting/kieren_cohare_palatino_sty.mplstyle')
except:
    plt.rcParams.update({'font.size': 9, 'axes.labelsize': 10,
                         'legend.fontsize': 8, 'lines.linewidth': 1.5})

MODEL_COLOR = '#2166ac'
MODEL_LABEL = os.path.basename(os.path.normpath(MODEL_DIR))

def obs_fill(ax, x, lo, hi, **kw):
    kw.setdefault('alpha', 0.25)
    kw.setdefault('color', 'grey')
    ax.fill_between(x, lo, hi, **kw)

def obs_line(ax, x, y, label, **kw):
    kw.setdefault('color', 'k')
    kw.setdefault('lw', 1.2)
    kw.setdefault('ls', '--')
    ax.plot(x, y, label=label, **kw)

def model_line(ax, xc, phi, label='Model'):
    ok = np.isfinite(phi) & (phi > 0)
    ax.semilogy(xc[ok], phi[ok], '-', color=MODEL_COLOR, lw=2, label=label, zorder=5)

def scatter(ax, x, y, c=None, **kw):
    """Diluted scatter with optional colour."""
    valid = np.isfinite(x) & np.isfinite(y)
    if c is not None:
        valid &= np.isfinite(c)
    xi, yi = x[valid], y[valid]
    ci = c[valid] if c is not None else None
    np.random.seed(SEED)
    if len(xi) > DILUTE:
        idx = np.random.choice(len(xi), DILUTE, replace=False)
        xi, yi = xi[idx], yi[idx]
        if ci is not None:
            ci = ci[idx]
    if ci is not None:
        sc = ax.scatter(xi, yi, c=ci, s=1, alpha=0.4, cmap='plasma_r',
                        rasterized=True, zorder=2, **kw)
        return sc
    ax.scatter(xi, yi, s=1, alpha=0.25, color=MODEL_COLOR,
               rasterized=True, zorder=2)


# ─────────────────────────────────────────────────────────────────────────────
# Figure layout: 4 rows × 3 cols = 12 panels
# ─────────────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(14, 18))
gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.42, wspace=0.32)
axes = [fig.add_subplot(gs[r, c]) for r in range(4) for c in range(3)]
(ax_himf, ax_h2mf, ax_coldmf,
 ax_hi_ms, ax_h2_ms, ax_neut_ms,
 ax_cold_ms, ax_fmol_ms, ax_hi_ssfr,
 ax_tdep, ax_h2hi, ax_fmol_hist) = axes

bins_mf  = np.arange(6.0, 12.5, 0.2)
bins_ms  = np.arange(8.0, 12.5, 0.25)
bins_ssfr = np.arange(-13, -8, 0.5)

def _panel(title):
    print(f"\n── {title}")

def _mf_info(label, log_mass, bins):
    fin = log_mass[np.isfinite(log_mass)]
    xc, phi = mass_function(log_mass, bins)
    ok = np.isfinite(phi) & (phi > 0)
    if ok.any():
        peak_x = xc[ok][np.argmax(phi[ok])]
        print(f"  {label}: N={len(fin):,}  log-range=[{fin.min():.1f},{fin.max():.1f}]  "
              f"peak phi at log(M)={peak_x:.1f}  phi_peak={phi[ok].max():.2e}")
    else:
        print(f"  {label}: N={len(fin):,}  no non-zero bins")

def _rel_info(label, x, y):
    fin = np.isfinite(x) & np.isfinite(y)
    if fin.sum() == 0:
        print(f"  {label}: no valid pairs"); return
    xc, lo, med, hi = running_median(x, y, bins_ms)
    if len(med) > 0:
        print(f"  {label}: N={fin.sum():,}  median range [{med.min():.2f}, {med.max():.2f}]  "
              f"(at Mstar bins {xc.min():.1f}–{xc.max():.1f})")
    else:
        print(f"  {label}: N={fin.sum():,}  no median bins with ≥5 galaxies")

# ── 1. HI Mass Function ──────────────────────────────────────────────────────
_panel("1. HI Mass Function")
_mf_info("HI", lH1, bins_mf)
ax = ax_himf
xc, phi = mass_function(lH1, bins_mf)
model_line(ax, xc, phi, MODEL_LABEL)

if himf_jones is not None:
    ax.fill_between(himf_jones[:,0], 10**himf_jones[:,2], 10**himf_jones[:,3],
                    alpha=0.2, color='tomato')
    obs_line(ax, himf_jones[:,0], 10**himf_jones[:,1], 'Jones+18', color='tomato', ls='-')
if himf_zwaan is not None:
    ax.errorbar(himf_zwaan[:,0], 10**himf_zwaan[:,1],
                yerr=[himf_zwaan[:,3], himf_zwaan[:,2]],
                fmt='s', ms=3, color='darkorange', label='Zwaan+05', capsize=2, zorder=3)

ax.set_xlabel(r'$\log_{10}(M_\mathrm{HI}/M_\odot)$')
ax.set_ylabel(r'$\phi$ [Mpc$^{-3}$ dex$^{-1}$]')
ax.set_xlim(6.5, 11.5);  ax.set_ylim(1e-6, 0.3)
ax.set_yscale('log');  ax.legend(fontsize=7)
ax.set_title('HI Mass Function')

# ── 2. H2 Mass Function ──────────────────────────────────────────────────────
_panel("2. H2 Mass Function")
_mf_info("H2", lH2, bins_mf)
ax = ax_h2mf
xc, phi = mass_function(lH2, bins_mf)
model_line(ax, xc, phi, MODEL_LABEL)

if h2mf_dd is not None:
    ax.fill_between(h2mf_dd[:,0], 10**h2mf_dd[:,2], 10**h2mf_dd[:,3],
                    alpha=0.2, color='steelblue')
    obs_line(ax, h2mf_dd[:,0], 10**h2mf_dd[:,1], 'Fletcher+21 (det+nondet)',
             color='steelblue', ls='-')
if h2mf_est is not None:
    obs_line(ax, h2mf_est[:,0], 10**h2mf_est[:,1], 'Fletcher+21 (estimated)',
             color='navy', ls=':')

ax.set_xlabel(r'$\log_{10}(M_\mathrm{H_2}/M_\odot)$')
ax.set_ylabel(r'$\phi$ [Mpc$^{-3}$ dex$^{-1}$]')
ax.set_xlim(6.5, 11.5);  ax.set_ylim(1e-6, 0.3)
ax.set_yscale('log');  ax.legend(fontsize=7)
ax.set_title('H$_2$ Mass Function')

# ── 3. Cold Gas Mass Function ────────────────────────────────────────────────
_panel("3. Cold/HI/H2 Mass Functions")
_mf_info("ColdGas", lCold, bins_mf)
ax = ax_coldmf
xc_h, phi_h = mass_function(lH1, bins_mf)
xc_m, phi_m = mass_function(lH2, bins_mf)
xc_c, phi_c = mass_function(lCold, bins_mf)

model_line(ax, xc_c, phi_c, 'Cold gas')
ok_h = np.isfinite(phi_h) & (phi_h > 0)
ok_m = np.isfinite(phi_m) & (phi_m > 0)
ax.semilogy(xc_h[ok_h], phi_h[ok_h], '--', color='steelblue', lw=1.5, label='HI')
ax.semilogy(xc_m[ok_m], phi_m[ok_m], ':', color='firebrick', lw=1.5, label='H$_2$')

ax.set_xlabel(r'$\log_{10}(M/M_\odot)$')
ax.set_ylabel(r'$\phi$ [Mpc$^{-3}$ dex$^{-1}$]')
ax.set_xlim(6.5, 12.0);  ax.set_ylim(1e-6, 0.3)
ax.set_yscale('log');  ax.legend(fontsize=7)
ax.set_title('Gas Mass Functions (model)')

# ── 4. HI/Mstar vs Mstar ─────────────────────────────────────────────────────
_panel("4. HI/Mstar vs Mstar")
_rel_info("HI/Mstar", lMstar, lH1_Mstar)
ax = ax_hi_ms
scatter(ax, lMstar, lH1_Mstar)
xc, lo, med, hi = running_median(lMstar, lH1_Mstar, bins_ms)
ax.plot(xc, med, '-', color=MODEL_COLOR, lw=2, label=MODEL_LABEL, zorder=5)
ax.fill_between(xc, lo, hi, alpha=0.2, color=MODEL_COLOR, zorder=4)

if rhi_brown15 is not None:
    ax.plot(rhi_brown15[:,0], np.log10(rhi_brown15[:,1]),
            'o', ms=5, color='tomato', label='Brown+15', zorder=6)
if rhi_cal_ltg is not None:
    obs_fill(ax, rhi_cal_ltg[:,0], rhi_cal_ltg[:,2], rhi_cal_ltg[:,3], color='steelblue')
    obs_line(ax, rhi_cal_ltg[:,0], rhi_cal_ltg[:,1], 'Calette+18 LTG', color='steelblue', ls='-')
if rhi_cal_etg is not None:
    obs_line(ax, rhi_cal_etg[:,0], rhi_cal_etg[:,1], 'Calette+18 ETG', color='navy', ls='--')
if hi_xgass is not None:
    obs_fill(ax, hi_xgass[:,0], hi_xgass[:,1]-hi_xgass[:,3],
             hi_xgass[:,1]+hi_xgass[:,2], color='orange', alpha=0.2)
    obs_line(ax, hi_xgass[:,0], hi_xgass[:,1], 'xGASS', color='darkorange', ls='-.')

ax.set_xlabel(r'$\log_{10}(M_\star/M_\odot)$')
ax.set_ylabel(r'$\log_{10}(M_\mathrm{HI}/M_\star)$')
ax.set_xlim(8, 12.5);  ax.set_ylim(-3.5, 1.5)
ax.legend(fontsize=6, ncol=2);  ax.set_title('HI fraction vs $M_\\star$')

# ── 5. H2/Mstar vs Mstar ─────────────────────────────────────────────────────
_panel("5. H2/Mstar vs Mstar")
_rel_info("H2/Mstar", lMstar, lH2_Mstar)
ax = ax_h2_ms
scatter(ax, lMstar, lH2_Mstar)
xc, lo, med, hi = running_median(lMstar, lH2_Mstar, bins_ms)
ax.plot(xc, med, '-', color=MODEL_COLOR, lw=2, label=MODEL_LABEL, zorder=5)
ax.fill_between(xc, lo, hi, alpha=0.2, color=MODEL_COLOR, zorder=4)

if rh2_cal_ltg is not None:
    obs_fill(ax, rh2_cal_ltg[:,0], rh2_cal_ltg[:,2], rh2_cal_ltg[:,3], color='steelblue')
    obs_line(ax, rh2_cal_ltg[:,0], rh2_cal_ltg[:,1], 'Calette+18 LTG', color='steelblue', ls='-')
if rh2_cal_etg is not None:
    obs_line(ax, rh2_cal_etg[:,0], rh2_cal_etg[:,1], 'Calette+18 ETG', color='navy', ls='--')
if h2_xcg is not None:
    obs_fill(ax, h2_xcg[:,0], h2_xcg[:,1]-h2_xcg[:,3],
             h2_xcg[:,1]+h2_xcg[:,2], color='orange', alpha=0.2)
    obs_line(ax, h2_xcg[:,0], h2_xcg[:,1], 'xCOLD GASS', color='darkorange', ls='-.')

ax.set_xlabel(r'$\log_{10}(M_\star/M_\odot)$')
ax.set_ylabel(r'$\log_{10}(M_\mathrm{H_2}/M_\star)$')
ax.set_xlim(8, 12.5);  ax.set_ylim(-3.5, 1.0)
ax.legend(fontsize=6, ncol=2);  ax.set_title('H$_2$ fraction vs $M_\\star$')

# ── 6. Neutral (HI+H2)/Mstar vs Mstar ───────────────────────────────────────
_panel("6. Neutral gas/Mstar vs Mstar")
_rel_info("(HI+H2)/Mstar", lMstar, lNeutral)
ax = ax_neut_ms
scatter(ax, lMstar, lNeutral)
xc, lo, med, hi = running_median(lMstar, lNeutral, bins_ms)
ax.plot(xc, med, '-', color=MODEL_COLOR, lw=2, label=MODEL_LABEL, zorder=5)
ax.fill_between(xc, lo, hi, alpha=0.2, color=MODEL_COLOR, zorder=4)

if neutral_xcg is not None:
    obs_fill(ax, neutral_xcg[:,0], neutral_xcg[:,1]-neutral_xcg[:,3],
             neutral_xcg[:,1]+neutral_xcg[:,2], color='orange', alpha=0.25)
    obs_line(ax, neutral_xcg[:,0], neutral_xcg[:,1], 'xGASS+xCGS', color='darkorange', ls='-.')

ax.set_xlabel(r'$\log_{10}(M_\star/M_\odot)$')
ax.set_ylabel(r'$\log_{10}((M_\mathrm{HI}+M_\mathrm{H_2})/M_\star)$')
ax.set_xlim(8, 12.5);  ax.set_ylim(-3.5, 1.5)
ax.legend(fontsize=7);  ax.set_title('Neutral gas fraction vs $M_\\star$')

# ── 7. Cold gas/Mstar vs Mstar ───────────────────────────────────────────────
_panel("7. ColdGas/Mstar vs Mstar")
_rel_info("ColdGas/Mstar", lMstar, lCold_Mstar)
ax = ax_cold_ms
scatter(ax, lMstar, lCold_Mstar)
xc, lo, med, hi = running_median(lMstar, lCold_Mstar, bins_ms)
ax.plot(xc, med, '-', color=MODEL_COLOR, lw=2, label=MODEL_LABEL, zorder=5)
ax.fill_between(xc, lo, hi, alpha=0.2, color=MODEL_COLOR, zorder=4)
ax.axhline(0, color='k', lw=0.8, ls=':')

ax.set_xlabel(r'$\log_{10}(M_\star/M_\odot)$')
ax.set_ylabel(r'$\log_{10}(M_\mathrm{cold}/M_\star)$')
ax.set_xlim(8, 12.5);  ax.set_ylim(-3.5, 1.5)
ax.legend(fontsize=7);  ax.set_title('Cold gas fraction vs $M_\\star$')

# ── 8. f_mol = H2/(HI+H2) vs Mstar ─────────────────────────────────────────
_panel("8. f_mol vs Mstar")
_rel_info("log(f_mol)", lMstar, lg(f_mol))
ax = ax_fmol_ms
lg_fmol = lg(f_mol)
scatter(ax, lMstar, lg_fmol)
xc, lo, med, hi = running_median(lMstar, lg_fmol, bins_ms)
ax.plot(xc, med, '-', color=MODEL_COLOR, lw=2, label=MODEL_LABEL, zorder=5)
ax.fill_between(xc, lo, hi, alpha=0.2, color=MODEL_COLOR, zorder=4)
ax.axhline(0, color='k', lw=0.8, ls=':')  # f_mol = 1 (all molecular)

ax.set_xlabel(r'$\log_{10}(M_\star/M_\odot)$')
ax.set_ylabel(r'$\log_{10}(f_\mathrm{mol})$  $[H_2/(H_2+HI)]$')
ax.set_xlim(8, 12.5);  ax.set_ylim(-4, 0.5)
ax.legend(fontsize=7);  ax.set_title('Molecular fraction vs $M_\\star$')

# ── 9. HI/Mstar vs sSFR ──────────────────────────────────────────────────────
_panel("9. HI/Mstar vs sSFR")
fin_ssfr = np.isfinite(lsSFR) & np.isfinite(lH1_Mstar)
print(f"  sSFR range: [{lsSFR[np.isfinite(lsSFR)].min():.1f}, {lsSFR[np.isfinite(lsSFR)].max():.1f}]  "
      f"N with both finite: {fin_ssfr.sum():,}")
ax = ax_hi_ssfr
scatter(ax, lsSFR, lH1_Mstar)
xc, lo, med, hi = running_median(lsSFR, lH1_Mstar, bins_ssfr)
ax.plot(xc, med, '-', color=MODEL_COLOR, lw=2, label=MODEL_LABEL, zorder=5)
ax.fill_between(xc, lo, hi, alpha=0.2, color=MODEL_COLOR, zorder=4)

if hissfr_b17 is not None:
    # col5 flag: 0=centrals, 1=satellite-dominated bins
    for flag, label, fmt in [(0, 'Brown+17 (all)', 'o'), (1, 'Brown+17 (sat.)', 's')]:
        m = hissfr_b17[:, 4] == flag
        if m.any():
            data = hissfr_b17[m]
            ax.errorbar(data[:,0], data[:,1],
                        yerr=[data[:,1]-data[:,2], data[:,3]-data[:,1]],
                        fmt=fmt, ms=4, label=label,
                        color='tomato' if flag==0 else 'salmon',
                        capsize=2, zorder=6)

ax.set_xlabel(r'$\log_{10}(\mathrm{sSFR}/\mathrm{yr}^{-1})$')
ax.set_ylabel(r'$\log_{10}(M_\mathrm{HI}/M_\star)$')
ax.set_xlim(-13, -8);  ax.set_ylim(-3.5, 1.5)
ax.legend(fontsize=7);  ax.set_title('HI fraction vs sSFR')

# ── 10. H2 depletion time vs Mstar ───────────────────────────────────────────
_panel("10. H2 depletion time vs Mstar")
sfr_mask = SFR > 0
_rel_info("log(t_dep/Gyr)", lMstar[sfr_mask], lt_dep_Gyr[sfr_mask])
fin_td = lt_dep_Gyr[sfr_mask & np.isfinite(lt_dep_Gyr)]
if len(fin_td):
    print(f"  t_dep range: [{fin_td.min():.2f}, {fin_td.max():.2f}] log10(Gyr)  "
          f"  median={np.median(fin_td):.2f}")
ax = ax_tdep
scatter(ax, lMstar[sfr_mask], lt_dep_Gyr[sfr_mask])
xc, lo, med, hi = running_median(lMstar[sfr_mask], lt_dep_Gyr[sfr_mask], bins_ms)
ax.plot(xc, med, '-', color=MODEL_COLOR, lw=2, label=MODEL_LABEL, zorder=5)
ax.fill_between(xc, lo, hi, alpha=0.2, color=MODEL_COLOR, zorder=4)

# Observational reference lines for typical depletion times
for t_ref, ls_ref, lbl in [(0.5, ':', '0.5 Gyr'), (2.0, '--', '2 Gyr'), (10.0, '-.', '10 Gyr')]:
    ax.axhline(np.log10(t_ref), color='grey', lw=0.9, ls=ls_ref, label=lbl)

ax.set_xlabel(r'$\log_{10}(M_\star/M_\odot)$')
ax.set_ylabel(r'$\log_{10}(\tau_\mathrm{dep,H_2}\ /\ \mathrm{Gyr})$')
ax.set_xlim(8, 12.5);  ax.set_ylim(-2, 2)
ax.legend(fontsize=7, ncol=2);  ax.set_title('H$_2$ depletion time vs $M_\\star$')

# ── 11. H2 vs HI ─────────────────────────────────────────────────────────────
_panel("11. H2 vs HI scatter")
both = np.isfinite(lH1) & np.isfinite(lH2)
if both.sum():
    above = np.sum(lH2[both] > lH1[both])
    below = np.sum(lH2[both] < lH1[both])
    print(f"  N with both: {both.sum():,}  H2>HI: {above:,} ({100*above/both.sum():.0f}%)  "
          f"HI>H2: {below:,} ({100*below/both.sum():.0f}%)")
ax = ax_h2hi
sc = scatter(ax, lH1, lH2, c=lMstar, vmin=8, vmax=12)
if sc is not None:
    cb = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.04)
    cb.set_label(r'$\log M_\star$', fontsize=7)
    cb.ax.tick_params(labelsize=6)
# 1:1 line
xl = np.array([6, 12])
ax.plot(xl, xl, 'k:', lw=1, label='HI = H$_2$')
ax.set_xlabel(r'$\log_{10}(M_\mathrm{HI}/M_\odot)$')
ax.set_ylabel(r'$\log_{10}(M_\mathrm{H_2}/M_\odot)$')
ax.set_xlim(6, 12);  ax.set_ylim(6, 12)
ax.legend(fontsize=7);  ax.set_title('H$_2$ vs HI')

# ── 12. f_mol distribution (histogram) ───────────────────────────────────────
_panel("12. f_mol histogram by Mstar bin")
sf_mask = SFR > 0
valid_all = f_mol[(f_mol > 0) & np.isfinite(f_mol)]
valid_sf  = f_mol[sf_mask & (f_mol > 0) & np.isfinite(f_mol)]
print(f"  All    f_mol>0: {len(valid_all):,}  "
      f"median={np.median(valid_all):.3f}  "
      f"f_mol>0.5: {np.mean(valid_all>0.5)*100:.1f}%")
print(f"  SF     f_mol>0: {len(valid_sf):,}  "
      f"median={np.median(valid_sf):.3f}  "
      f"f_mol>0.5: {np.mean(valid_sf>0.5)*100:.1f}%")

ax = ax_fmol_hist

m_bins = [(8.0, 9.5, '#abd9e9', r'$10^{8}$–$10^{9.5}$'),
          (9.5, 10.5, '#2c7bb6', r'$10^{9.5}$–$10^{10.5}$'),
          (10.5, 12.5, '#d73027', r'$>10^{10.5}$')]
fbins = np.linspace(0, 1, 40)

base_mask = (f_mol > 0) & np.isfinite(f_mol)
for mlo, mhi, col, lbl in m_bins:
    m_all = (lMstar >= mlo) & (lMstar < mhi) & base_mask
    m_sf  = m_all & sf_mask
    if m_all.sum() > 10:
        fm_all = f_mol[m_all]
        print(f"  {lbl:<22s}  all N={m_all.sum():>7,}  med={np.median(fm_all):.3f}  "
              f">0.5: {np.mean(fm_all>0.5)*100:.0f}%", end='')
        ax.hist(fm_all, bins=fbins, density=True, histtype='step',
                color=col, lw=1.5, ls='--', alpha=0.6)
    if m_sf.sum() > 10:
        fm_sf = f_mol[m_sf]
        print(f"    SF N={m_sf.sum():>7,}  med={np.median(fm_sf):.3f}  "
              f">0.5: {np.mean(fm_sf>0.5)*100:.0f}%")
        ax.hist(fm_sf, bins=fbins, density=True, histtype='step',
                color=col, lw=1.5, ls='-', label=lbl)
    elif m_all.sum() > 10:
        print()  # newline if no SF bin

# Legend proxy lines for line-style meaning
from matplotlib.lines import Line2D
handles, labels = ax.get_legend_handles_labels()
handles += [Line2D([0],[0], color='grey', lw=1.5, ls='-'),
            Line2D([0],[0], color='grey', lw=1.5, ls='--', alpha=0.6)]
labels  += ['SF only', 'All']
ax.axvline(0.5, color='grey', lw=0.8, ls=':')
ax.set_xlabel(r'$f_\mathrm{mol} = M_\mathrm{H_2}/(M_\mathrm{H_2}+M_\mathrm{HI})$')
ax.set_ylabel('PDF')
ax.set_xlim(0, 1)
ax.legend(handles, labels, fontsize=7, title=r'$\log M_\star$', title_fontsize=7)
ax.set_title('Molecular fraction distribution')

# ─────────────────────────────────────────────────────────────────────────────
# Global title and save
# ─────────────────────────────────────────────────────────────────────────────

fig.suptitle(
    f'{MODEL_LABEL}   —   {SNAP_KEY}  (z = {SNAP_Z:.3f})   '
    f'V = {VOLUME:.0f} Mpc³',
    fontsize=11, y=0.995
)

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
fig.savefig(OUT_FILE, dpi=150, bbox_inches='tight')
print(f"\nSaved → {OUT_FILE}")
plt.close(fig)

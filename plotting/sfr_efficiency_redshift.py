#!/usr/bin/env python
"""
SAGE26 SFR Efficiency vs Stellar Mass — Redshift Evolution (H2SFRMode=2)
=========================================================================
Plots the effective SFR efficiency as a function of stellar mass across
8 redshift panels (z=0 to z~12).

Definition:
    ε_eff = SFR × τ_dyn / M_H2   (× 100 for %)

where τ_dyn = 3 × DiskRadius_phys / Vvir, exactly mirroring the SAGE code.
This is the per-dynamical-time efficiency K13 implies — i.e. what
SfrEfficiency would need to be in mode 0 to reproduce the same SFR.

Usage:
    python plotting/sfr_efficiency_redshift.py
    python plotting/sfr_efficiency_redshift.py ./output/myrun/
"""

import sys, os, glob
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import h5py as h5

# ─── Configuration ────────────────────────────────────────────────────────────

MODEL_DIR   = sys.argv[1] if len(sys.argv) > 1 else './output/millennium/'
VANILLA_DIR = sys.argv[2] if len(sys.argv) > 2 else None
OUT_FILE    = os.path.join(MODEL_DIR, 'plots', 'sfr_efficiency_redshift.pdf')

TARGET_REDSHIFTS = [0, 0.5, 1, 2, 4, 6, 8, 12]

MIN_PARTICLES = 20
DILUTE        = 7500
SEED          = 42

_MSUN_CGS      = 1.989e33
_MPC_KMS_TO_YR = 3.0857e19 / 3.1557e7   # Mpc/(km/s) → yr

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
    try:
        with h5.File(files[0], 'r') as f:
            sim     = f['Header/Simulation']
            runtime = f['Header/Runtime']
            hdr = {
                'hubble_h':       float(sim.attrs['hubble_h']),
                'unit_mass_in_g': float(runtime.attrs['UnitMass_in_g']),
                'redshifts':      np.array(f['Header/snapshot_redshifts'][:]),
            }
        return hdr
    except Exception as e:
        print(f"  Header read failed: {e}")
        return None

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

if VANILLA_DIR:
    v_hdr = _read_header(VANILLA_DIR)
    if v_hdr:
        vanilla_files         = _find_files(VANILLA_DIR)
        vanilla_hubble_h      = v_hdr['hubble_h']
        vanilla_mass_convert  = v_hdr['unit_mass_in_g'] / _MSUN_CGS / vanilla_hubble_h
        vanilla_redshifts     = v_hdr['redshifts']
    else:
        vanilla_files = []
else:
    vanilla_files = []

props         = ['StellarMass', 'H2gas', 'ColdGas', 'SfrDisk', 'SfrBulge', 'FFBRegime', 'DiskRadius', 'Vvir']
vanilla_props = ['StellarMass', 'ColdGas', 'SfrDisk', 'SfrBulge', 'DiskRadius', 'Vvir']

# ─── Figure ───────────────────────────────────────────────────────────────────

plt.rcParams["font.size"]       = 11
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor']   = 'white'
plt.rcParams['axes.edgecolor']   = 'black'
plt.rcParams['xtick.color']      = 'black'
plt.rcParams['ytick.color']      = 'black'
plt.rcParams['axes.labelcolor']  = 'black'

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
    gas      = np.where(H2 > 0, H2, ColdGas)   # H2 if available, else cold gas
    SfrDisk  = np.maximum(d['SfrDisk'],  0.0)
    SfrBulge = np.maximum(d['SfrBulge'], 0.0)
    SFR_tot  = SfrDisk + SfrBulge

    disk_dominated = (SfrBulge < 0.1 * SFR_tot) | (SFR_tot == 0)

    DiskR   = np.maximum(d['DiskRadius'], 0.0)
    Vvir_v  = np.maximum(d['Vvir'], 1.0)
    tau_dyn = 3.0 * DiskR / HUBBLE_H * _MPC_KMS_TO_YR / Vvir_v  # yr

    with np.errstate(divide='ignore', invalid='ignore'):
        eff = np.where(
            (gas > 0) & (SFR_tot > 0) & (Mstar > 0) & (DiskR > 0),
            SFR_tot * tau_dyn / gas * 100.0,
            np.nan
        )

    valid    = (Mstar > 0) & (eff > 0) & np.isfinite(eff) & disk_dominated
    lM       = np.log10(Mstar[valid])
    lEff     = np.log10(eff[valid])
    ffb_flag = d['FFBRegime'][valid] > 0 if 'FFBRegime' in d else np.zeros(valid.sum(), dtype=bool)
    n        = len(lM)

    high_eff  = lEff > np.log10(20.0)
    confirmed = high_eff & ffb_flag

    # All non-FFB galaxies — diluted steelblue (includes high-eff non-FFB)
    norm_mask = ~confirmed
    lM_norm, lEff_norm = lM[norm_mask], lEff[norm_mask]
    n_norm = len(lM_norm)
    if n_norm > DILUTE:
        idx = rng.choice(n_norm, DILUTE, replace=False)
        ax.scatter(lM_norm[idx], lEff_norm[idx], s=1.5, alpha=0.35,
                   color='steelblue', rasterized=True, linewidths=0)
    elif n_norm > 0:
        ax.scatter(lM_norm, lEff_norm, s=1.5, alpha=0.35,
                   color='steelblue', rasterized=True, linewidths=0)

    # FFBRegime — firebrick
    if confirmed.any():
        ax.scatter(lM[confirmed], lEff[confirmed], s=10, alpha=0.7,
                   color='firebrick', rasterized=True, linewidths=0, zorder=4)

    # Running median (H2 run)
    if n > 100:
        bin_edges = np.linspace(np.percentile(lM, 2), np.percentile(lM, 98), 20)
        bin_idx   = np.digitize(lM, bin_edges)
        med_x, med_y = [], []
        for b in range(1, len(bin_edges)):
            sel = bin_idx == b
            if sel.sum() >= 10:
                med_x.append(0.5 * (bin_edges[b - 1] + bin_edges[b]))
                med_y.append(np.median(lEff[sel]))
        if len(med_x) > 2:
            ax.plot(med_x, med_y, color='navy', lw=2.0, zorder=5, solid_capstyle='round')

    # ── Vanilla SAGE overlay ──────────────────────────────────────────────────
    if vanilla_files:
        v_snap_nr  = int(np.argmin(np.abs(vanilla_redshifts - z_target)))
        v_snap_key = f'Snap_{v_snap_nr}'
        dv = load_snap(vanilla_files, v_snap_key, vanilla_props, vanilla_mass_convert)
        if dv and 'StellarMass' in dv and 'ColdGas' in dv:
            Mstar_v   = dv['StellarMass']
            ColdGas_v = np.maximum(dv['ColdGas'], 0.0)
            SfrDisk_v  = np.maximum(dv['SfrDisk'],  0.0)
            SfrBulge_v = np.maximum(dv['SfrBulge'], 0.0)
            SFRtot_v   = SfrDisk_v + SfrBulge_v
            DiskR_v    = np.maximum(dv['DiskRadius'], 0.0)
            Vvir_raw_v = np.maximum(dv['Vvir'], 0.0)
            Vvir_vv    = np.maximum(Vvir_raw_v, 1.0)
            dd_v       = (SfrBulge_v < 0.1 * SFRtot_v) | (SFRtot_v == 0)
            tau_dyn_v  = 3.0 * DiskR_v / vanilla_hubble_h * _MPC_KMS_TO_YR / Vvir_vv

            # cold_crit = 0.19 * Vvir * reff (code units: 10^10 Msun/h)
            # reff = 3 * DiskScaleRadius; DiskRadius in HDF5 is DiskScaleRadius (Mpc/h)
            cold_crit_v = 0.19 * Vvir_raw_v * (3.0 * DiskR_v) * vanilla_mass_convert
            gas_sf_v    = np.maximum(ColdGas_v - cold_crit_v, 0.0)

            with np.errstate(divide='ignore', invalid='ignore'):
                eff_v = np.where(
                    (gas_sf_v > 0) & (SFRtot_v > 0) & (Mstar_v > 0) & (DiskR_v > 0),
                    SFRtot_v * tau_dyn_v / gas_sf_v * 100.0,
                    np.nan
                )
            valid_v = (Mstar_v > 0) & (eff_v > 0) & np.isfinite(eff_v) & dd_v
            lM_v    = np.log10(Mstar_v[valid_v])
            lEff_v  = np.log10(eff_v[valid_v])
            nv      = len(lM_v)

            if nv > DILUTE:
                idx = rng.choice(nv, DILUTE, replace=False)
                ax.scatter(lM_v[idx], lEff_v[idx], s=1.5, alpha=0.25,
                           color='mediumpurple', rasterized=True, linewidths=0)
            elif nv > 0:
                ax.scatter(lM_v, lEff_v, s=1.5, alpha=0.25,
                           color='mediumpurple', rasterized=True, linewidths=0)

            if nv > 100:
                bin_edges = np.linspace(np.percentile(lM_v, 2), np.percentile(lM_v, 98), 20)
                bin_idx   = np.digitize(lM_v, bin_edges)
                med_x, med_y = [], []
                for b in range(1, len(bin_edges)):
                    sel = bin_idx == b
                    if sel.sum() >= 10:
                        med_x.append(0.5 * (bin_edges[b-1] + bin_edges[b]))
                        med_y.append(np.median(lEff_v[sel]))
                if len(med_x) > 2:
                    ax.plot(med_x, med_y, color='rebeccapurple', lw=2.0,
                            zorder=6, solid_capstyle='round')

    ax.axhline(np.log10(3.0), color='tomato',    lw=1.0, ls='--', alpha=0.7, zorder=3)
    ax.axhline(np.log10(5.0), color='darkorange', lw=1.0, ls='--', alpha=0.7, zorder=3)

    ax.set_title(f'z = {snap_z:.2f}', fontsize=11, pad=4)
    print(f"  N_SF = {n:,}  median ε = {10**np.nanmedian(lEff):.2f}%"
          f"  FFB: {confirmed.sum()}")

# ─── Axis formatting ──────────────────────────────────────────────────────────

for ax in axes_flat:
    ax.set_xlim(7.5, 12.5)
    ax.set_ylim(np.log10(0.5), 2)
    ax.grid(False)

ytick_vals   = [np.log10(0.5), np.log10(1), np.log10(3), np.log10(10), np.log10(100)]
ytick_labels = ['0.5%', '1%', '3%', '10%', '100%']
for ax in axes_flat:
    ax.set_yticks(ytick_vals)
    ax.set_yticklabels([])

for ax in axes[:, 0]:
    ax.set_yticklabels(ytick_labels)

xtick_vals = [8, 9, 10, 11, 12]
for ax in axes_flat:
    ax.set_xticks(xtick_vals)

fig.supxlabel(r'log$_{10}$(M$_\star$/M$_\odot$)', y=0.02, fontsize=13)
fig.supylabel(
    r'$\varepsilon_\mathrm{dyn} = \mathrm{SFR}\,\tau_\mathrm{dyn}\,/\,M_{\mathrm{H_2}}$',
    x=0.06, fontsize=12
)

legend_handles = [
    plt.Line2D([0], [0], color='navy',         lw=2,   label='H2 SFR mode (median)'),
    plt.scatter([], [], s=10, color='firebrick', alpha=0.7, label='FFBRegime'),
    plt.Line2D([0], [0], color='tomato',        lw=1, ls='--', label=r'$\varepsilon = 3\%$'),
    plt.Line2D([0], [0], color='darkorange',    lw=1, ls='--', label=r'$\varepsilon = 5\%$'),
]
if vanilla_files:
    legend_handles += [
        plt.Line2D([0], [0], color='rebeccapurple', lw=2, label='Vanilla SAGE (median)'),
    ]
axes_flat[-1].legend(handles=legend_handles, fontsize=9, loc='lower right', frameon=False)


os.makedirs(os.path.join(MODEL_DIR, 'plots'), exist_ok=True)
fig.savefig(OUT_FILE, bbox_inches='tight', dpi=150)
print(f"\nSaved → {OUT_FILE}")
plt.close()

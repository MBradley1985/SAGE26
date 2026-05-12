#!/usr/bin/env python
"""
SAGE26 Discussion Figures: Precipitation Physics & FFB Haloes
=============================================================

Generates figures for the discussion sections of the SAGE26 paper:

  A: t_cool/t_ff distribution for CGM-regime haloes (violin plot)
  B: Precipitation fraction vs t_cool/t_ff (theoretical model curve)
  C: CGM gas fractions and depletion timescales (2-panel)
  E: Star formation efficiency: FFB vs normal at z~10
  F: FFB galaxy properties at z~10 (3-panel: size-mass, MZR, sSFR)
  G: Star formation histories of FFB galaxies

Based on allresults-history.py
"""

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

from random import seed
from scipy.integrate import quad

import warnings
warnings.filterwarnings("ignore")

# ========================== USER OPTIONS ==========================

# File details
DirName = './output/millennium/'
FileName = 'model_0.hdf5'

# Simulation details
Hubble_h = 0.73
BoxSize = 62.5         # h^-1 Mpc
VolumeFraction = 1.0
FirstSnap = 0
LastSnap = 63

# Cosmology
Omega_m = 0.25
Omega_L = 0.75
Omega_b = 0.045
f_baryon = 0.17

redshifts = [127.000, 79.998, 50.000, 30.000, 19.916, 18.244, 16.725, 15.343,
             14.086, 12.941, 11.897, 10.944, 10.073, 9.278, 8.550, 7.883,
             7.272, 6.712, 6.197, 5.724, 5.289, 4.888, 4.520, 4.179,
             3.866, 3.576, 3.308, 3.060, 2.831, 2.619, 2.422, 2.239,
             2.070, 1.913, 1.766, 1.630, 1.504, 1.386, 1.276, 1.173,
             1.078, 0.989, 0.905, 0.828, 0.755, 0.687, 0.624, 0.564,
             0.509, 0.457, 0.408, 0.362, 0.320, 0.280, 0.242, 0.208,
             0.175, 0.144, 0.116, 0.089, 0.064, 0.041, 0.020, 0.000]

# Key snapshots for specific redshifts
snap_z0  = 63   # z = 0.000
snap_z1  = 39   # z = 1.173
snap_z2  = 32   # z = 2.070
snap_z4  = 23   # z = 4.179
snap_z10 = 12   # z = 10.073

# Solar metallicity (Asplund et al. 2009)
Z_sun = 0.0134

# Color Scheme (FFB vs Normal)
# FFB = Firebrick (Red), Normal = Dodgerblue (Blue)
c_ffb = 'firebrick'
c_ffb_edge = 'darkred'
c_norm = 'dodgerblue'
c_norm_edge = 'navy'

# Plotting options
dilute = 7500
OutputFormat = '.pdf'
plt.rcParams["figure.figsize"] = (8.34, 6.25)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 14

# Dark theme
plt.rcParams['figure.facecolor'] = 'black'
plt.rcParams['axes.facecolor'] = 'black'
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['axes.titlecolor'] = 'white'
plt.rcParams['text.color'] = 'white'
plt.rcParams['legend.facecolor'] = 'black'
plt.rcParams['legend.edgecolor'] = 'white'


# ==================================================================
# HELPER FUNCTIONS
# ==================================================================

def read_hdf(snap_num=None, param=None):
    """Read a single parameter from a snapshot."""
    property = h5.File(DirName + FileName, 'r')
    return np.array(property[snap_num][param])


def cosmic_time_gyr(z):
    """Age of the universe at redshift z, in Gyr."""
    t_H = 977.8 / (Hubble_h * 100)  # Hubble time in Gyr

    def integrand(zp):
        return 1.0 / ((1 + zp) * np.sqrt(Omega_m * (1 + zp)**3 + Omega_L))

    result, _ = quad(integrand, z, 1000.0)
    return t_H * result


def precipitation_fraction(tcool_over_tff):
    """Calculate precipitation fraction from the SAGE26 model."""
    threshold = 10.0
    width = 2.0

    x = np.atleast_1d(np.array(tcool_over_tff, dtype=float))
    f = np.zeros_like(x)

    # Unstable regime
    mask_unstable = x < threshold
    inst = np.minimum(threshold / x[mask_unstable], 3.0)
    f[mask_unstable] = np.tanh(inst / 2.0)

    # Transition regime
    mask_trans = (x >= threshold) & (x < threshold + width)
    xi = (x[mask_trans] - threshold) / width
    f[mask_trans] = 0.5 * (1.0 - np.tanh(xi))

    return f.squeeze()


def ffb_threshold_mass_msun(z):
    """FFB threshold mass from Li et al. (2024) Eq. 2."""
    z_norm = (1.0 + z) / 10.0
    log_M_code = 0.8 + np.log10(Hubble_h) - 6.2 * np.log10(z_norm)
    return 10.0**log_M_code * 1.0e10 / Hubble_h


def binned_median(x, y, bins, min_count=5):
    """Compute binned median with 25th/75th percentiles."""
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    n = len(bins) - 1
    medians = np.full(n, np.nan)
    p25 = np.full(n, np.nan)
    p75 = np.full(n, np.nan)

    for i in range(n):
        mask = (x >= bins[i]) & (x < bins[i + 1])
        if np.sum(mask) >= min_count:
            medians[i] = np.median(y[mask])
            p25[i] = np.percentile(y[mask], 25)
            p75[i] = np.percentile(y[mask], 75)

    return bin_centers, medians, p25, p75


def save_white_copy(fig, dark_path):
    """Save a white-background copy with specific style adjustments."""
    white_path = dark_path.replace(OutputFormat, '_white' + OutputFormat)

    # 1. Standard Inversions (Backgrounds, Text, Spines)
    fig.set_facecolor('white')

    for ax in fig.get_axes():
        ax.set_facecolor('white')
        ax.title.set_color('black')
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
        ax.tick_params(colors='black', which='both')
        for spine in ax.spines.values():
            spine.set_edgecolor('black')

        # Legend Handling
        leg = ax.get_legend()
        if leg is not None:
            # A) Text to black
            for t in leg.get_texts():
                t.set_color('black')
            
            # B) Legend Lines (Handles) to black if they were yellow/cyan/white
            for h in leg.legend_handles:
                try:
                    # For simple Line2D objects
                    if hasattr(h, 'get_color'):
                        c_hex = mcolors.to_hex(h.get_color())
                        # Cyan, Yellow, White -> Black
                        if c_hex in ['#00ffff', '#ffff00', '#ffffff']:
                            h.set_color('black')
                    # For collections (scatter plots in legend) - usually keep color
                except (ValueError, TypeError, AttributeError):
                    pass

        # Text objects on plot
        for t in ax.texts:
            try:
                h = mcolors.to_hex(t.get_color())
            except (ValueError, TypeError):
                continue
            if h == '#ffffff':
                t.set_color('black')
            elif h in ('#ffff00', '#ffd93d'):
                t.set_color('#b8860b') # Dark gold

        # 2. Line Adjustments (Plot lines)
        for line in ax.lines:
            try:
                c_hex = mcolors.to_hex(line.get_color())
            except (ValueError, TypeError):
                continue
            # Cyan -> Black (e.g., tcool/tff threshold)
            if c_hex == '#00ffff':
                line.set_color('black')
            # Yellow/White -> Black (Standard lines)
            elif c_hex in ['#ffff00', '#ffffff']:
                line.set_color('black')

        # 3. Collection Adjustments (Shading and Violin parts)
        for coll in ax.collections:
            # A) Check Facecolors (e.g. Shading)
            fcs = coll.get_facecolor()
            if len(fcs) > 0:
                # Check for Cyan shading (0, 1, 1)
                c = fcs[0]
                if len(c) >= 3 and c[0] == 0.0 and c[1] == 1.0 and c[2] == 1.0:
                    # Found cyan precipitation zone. Change to Grey.
                    new_c = list(c)
                    new_c[0] = 0.5
                    new_c[1] = 0.5
                    new_c[2] = 0.5
                    coll.set_facecolor(new_c)

            # B) Check Edgecolors (e.g. Violin bars/caps)
            ecs = coll.get_edgecolor()
            if len(ecs) > 0:
                # Convert White or Yellow edges to Black
                new_ecs = []
                changed = False
                for c in ecs:
                    try:
                        h = mcolors.to_hex(c)
                        if h in ['#ffffff', '#ffff00']:
                            new_ecs.append(mcolors.to_rgba('black', alpha=c[3] if len(c)>3 else 1))
                            changed = True
                        else:
                            new_ecs.append(c)
                    except:
                        new_ecs.append(c)
                
                if changed:
                    coll.set_edgecolor(new_ecs)

    fig.savefig(white_path, facecolor='white')
    print(f'  Saved white copy to {white_path}')


# ==================================================================
# MAIN SCRIPT
# ==================================================================

if __name__ == '__main__':

    print('Running SAGE26 Discussion Figures')
    print('=' * 60 + '\n')

    seed(2222)
    volume = (BoxSize / Hubble_h)**3.0 * VolumeFraction

    OutputDir = DirName + 'plots/'
    if not os.path.exists(OutputDir):
        os.makedirs(OutputDir)

    # ------------------------------------------------------------------
    # Read galaxy data
    # ------------------------------------------------------------------

    print('Reading galaxy properties from', DirName + FileName)

    key_snaps = [snap_z0, snap_z1, snap_z2, snap_z4, snap_z10]
    fig_g_snaps = list(range(8, LastSnap + 1))
    all_snaps = sorted(set(key_snaps + fig_g_snaps))

    StellarMass = {}
    SfrDisk = {}
    SfrBulge = {}
    Mvir = {}
    Rvir = {}
    CGMgas = {}
    HotGas = {}
    MetalsStellarMass = {}
    DiskRadius = {}
    FFBRegime_data = {}
    Regime_data = {}
    tcool_tff = {}
    tdeplete = {}
    tff = {}
    GalIdx = {}
    GalType = {}

    for snap in all_snaps:
        Snapshot = 'Snap_' + str(snap)
        StellarMass[snap] = read_hdf(snap_num=Snapshot, param='StellarMass') * 1.0e10 / Hubble_h
        SfrDisk[snap] = read_hdf(snap_num=Snapshot, param='SfrDisk')
        SfrBulge[snap] = read_hdf(snap_num=Snapshot, param='SfrBulge')
        Mvir[snap] = read_hdf(snap_num=Snapshot, param='Mvir') * 1.0e10 / Hubble_h
        Rvir[snap] = read_hdf(snap_num=Snapshot, param='Rvir') / Hubble_h
        CGMgas[snap] = read_hdf(snap_num=Snapshot, param='CGMgas') * 1.0e10 / Hubble_h
        HotGas[snap] = read_hdf(snap_num=Snapshot, param='HotGas') * 1.0e10 / Hubble_h
        MetalsStellarMass[snap] = read_hdf(snap_num=Snapshot, param='MetalsStellarMass') * 1.0e10 / Hubble_h
        DiskRadius[snap] = read_hdf(snap_num=Snapshot, param='DiskRadius') / Hubble_h
        FFBRegime_data[snap] = read_hdf(snap_num=Snapshot, param='FFBRegime')
        Regime_data[snap] = read_hdf(snap_num=Snapshot, param='Regime')
        tcool_tff[snap] = read_hdf(snap_num=Snapshot, param='tcool_over_tff')
        tdeplete[snap] = read_hdf(snap_num=Snapshot, param='tdeplete')
        tff[snap] = read_hdf(snap_num=Snapshot, param='tff')
        GalIdx[snap] = read_hdf(snap_num=Snapshot, param='GalaxyIndex')
        GalType[snap] = read_hdf(snap_num=Snapshot, param='Type')

    print(f'  Read {len(all_snaps)} snapshots\n')

    cgm_active = np.any(tcool_tff[snap_z0] > 0)
    if not cgm_active:
        print('WARNING: CGMrecipeOn appears disabled.')


    # ==================================================================
    # FIGURE A: t_cool/t_ff Distribution
    # ==================================================================

    if cgm_active:
        print('Plotting Figure A: t_cool/t_ff distribution')

        fig, ax = plt.subplots(figsize=(10, 6))

        snap_info = [
            (snap_z4, f'z = {redshifts[snap_z4]:.1f}'),
            (snap_z2, f'z = {redshifts[snap_z2]:.1f}'),
            (snap_z1, f'z = {redshifts[snap_z1]:.1f}'),
            (snap_z0, f'z = {redshifts[snap_z0]:.1f}'),
        ]
        colors_A = ['#d62728', '#2ca02c', '#ff7f0e', '#1f77b4']

        violin_data = []
        violin_positions = []
        violin_labels = []
        valid_colors = []

        for i, (snap, label) in enumerate(snap_info):
            ratio = tcool_tff[snap]
            w = np.where(
                (Regime_data[snap] == 0) &
                (ratio > 0) & np.isfinite(ratio) &
                (GalType[snap] == 0) &
                (Mvir[snap] > 1e10)
            )[0]

            if len(w) > 10:
                data = np.log10(ratio[w])
                data = data[np.isfinite(data)]
                data = np.clip(data, -2, 4)
                violin_data.append(data)
                violin_positions.append(i)
                violin_labels.append(label)
                valid_colors.append(colors_A[i])

        if len(violin_data) > 0:
            parts = ax.violinplot(violin_data, positions=violin_positions,
                                  showmedians=True, showextrema=True, widths=0.7)

            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(valid_colors[i])
                pc.set_edgecolor('white')
                pc.set_alpha(0.6)
            # Default colors for Dark Theme
            parts['cmedians'].set_color('yellow')
            parts['cmedians'].set_linewidth(2)
            parts['cmins'].set_color('white')
            parts['cmaxes'].set_color('white')
            parts['cbars'].set_color('white')

            # Precipitation threshold
            ax.axhline(y=np.log10(10), color='cyan', ls='--', lw=2,
                        label=r'$t_{\rm cool}/t_{\rm ff} = 10$ (precipitation threshold)')

            # Shaded precipitation zone
            ax.axhspan(np.log10(5), np.log10(20), alpha=0.12, color='cyan',
                        label='Precipitation zone (5-20)')

            ax.set_xticks(violin_positions)
            ax.set_xticklabels(violin_labels, fontsize=13)
            ax.set_ylabel(r'$\log_{10}(t_{\rm cool}/t_{\rm ff})$', fontsize=14)
            ax.set_ylim(-2.5, 4.5)

            # Legend frame off
            leg = ax.legend(loc='upper left', fontsize=11, frameon=False)

            plt.tight_layout()
            outputFile = OutputDir + 'FigA_tcool_tff_distribution' + OutputFormat
            plt.savefig(outputFile)
            save_white_copy(fig, outputFile)
            print('  Saved file to', outputFile, '\n')
        else:
            print('  No valid CGM-regime data found. Skipping.\n')
        plt.close()


    # ==================================================================
    # FIGURE B: Precipitation Fraction vs t_cool/t_ff
    # ==================================================================

    print('Plotting Figure B: Precipitation fraction model')

    fig, ax = plt.subplots(figsize=(10, 6))

    ratio_arr = np.logspace(np.log10(0.5), 2.5, 2000)
    f_precip = precipitation_fraction(ratio_arr)

    ax.plot(ratio_arr, f_precip, 'cyan', lw=3, label='SAGE26 precipitation model', zorder=5)

    ax.axvline(x=10, color='yellow', ls='--', lw=1.5, alpha=0.7,
               label=r'$t_{\rm cool}/t_{\rm ff} = 10$')
    ax.axvline(x=12, color='yellow', ls=':', lw=1.0, alpha=0.5)

    ax.axvspan(0.5, 10, alpha=0.06, color='red')
    ax.axvspan(10, 12, alpha=0.06, color='yellow')
    ax.axvspan(12, 300, alpha=0.06, color='dodgerblue')

    ax.text(3, 0.65, 'Thermally\nUnstable', fontsize=14, ha='center', va='center',
            color=c_ffb, fontweight='bold')
    ax.text(11, 0.50, 'Transition', fontsize=11, ha='center', va='center',
            color='#ffd93d', fontweight='bold', rotation=90)
    ax.text(50, 0.12, 'Thermally\nStable', fontsize=14, ha='center', va='center',
            color=c_norm, fontweight='bold')

    if cgm_active:
        markers = ['x', 'D']
        for (snap, label), mark in zip([(snap_z0, 'z=0'), (snap_z2, 'z=2')], markers):
            ratio = tcool_tff[snap]
            w = np.where(
                (Regime_data[snap] == 0) &
                (ratio > 0) & np.isfinite(ratio) &
                (GalType[snap] == 0) &
                (Mvir[snap] > 1e10) &
                (ratio < 200)
            )[0]
            if len(w) > 0:
                if len(w) > 500:
                    w = np.random.choice(w, 500, replace=False)
                r_vals = ratio[w]
                f_vals = precipitation_fraction(r_vals)
                # Scatter with marker
                sc = ax.scatter(r_vals, f_vals, s=40, alpha=0.7,
                                c=np.log10(Mvir[snap][w]), cmap='plasma',
                                marker=mark,
                                edgecolors='none', zorder=10, label=f'Galaxies ({label})')

        cbar = plt.colorbar(sc, ax=ax, pad=0.02, aspect=30)
        cbar.set_label(r'$\log_{10}(M_{\rm vir}/M_{\odot})$', fontsize=12, color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    ax.set_xscale('log')
    ax.set_xlim(0.5, 300)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(r'$t_{\rm cool}/t_{\rm ff}$', fontsize=14)
    ax.set_ylabel(r'$f_{\rm precip}$', fontsize=14)

    # Legend frame off
    leg = ax.legend(loc='upper right', fontsize=10, frameon=False)

    plt.tight_layout()
    outputFile = OutputDir + 'FigB_precipitation_fraction' + OutputFormat
    plt.savefig(outputFile)
    save_white_copy(fig, outputFile)
    print('  Saved file to', outputFile, '\n')
    plt.close()


    # ==================================================================
    # FIGURE C: CGM Gas Fractions and Depletion Timescales (2-panel)
    # ==================================================================

    print('Plotting Figure C: CGM gas fractions and depletion timescales')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    snap_list_C = [
        (snap_z0, f'z={redshifts[snap_z0]:.0f}', '#1f77b4'),
        (snap_z2, f'z={redshifts[snap_z2]:.1f}', '#2ca02c'),
        (snap_z4, f'z={redshifts[snap_z4]:.1f}', '#d62728'),
    ]

    mass_bins = np.arange(10.0, 15.0, 0.3)

    if cgm_active:
        gas_label = r'$M_{\rm CGM}/M_{\rm vir}$'
        for snap, label, color in snap_list_C:
            w_cgm = np.where(
                (Regime_data[snap] == 0) & (Mvir[snap] > 1e10) &
                (CGMgas[snap] > 0) & (GalType[snap] == 0)
            )[0]
            w_hot = np.where(
                (Regime_data[snap] == 1) & (Mvir[snap] > 1e10) &
                (HotGas[snap] > 0) & (GalType[snap] == 0)
            )[0]

            if len(w_cgm) > 0:
                log_mv = np.log10(Mvir[snap][w_cgm])
                frac = CGMgas[snap][w_cgm] / Mvir[snap][w_cgm]
                bc, med, _, _ = binned_median(log_mv, frac, mass_bins)
                valid = ~np.isnan(med)
                ax1.plot(bc[valid], med[valid], '-o', color=color, lw=2,
                         markersize=5, label=f'CGM ({label})')

            if len(w_hot) > 0:
                log_mv = np.log10(Mvir[snap][w_hot])
                frac = HotGas[snap][w_hot] / Mvir[snap][w_hot]
                bc, med, _, _ = binned_median(log_mv, frac, mass_bins)
                valid = ~np.isnan(med)
                ax1.plot(bc[valid], med[valid], '--s', color=color, lw=2, alpha=0.6,
                         markersize=5, label=f'Hot ({label})')
    else:
        gas_label = r'$M_{\rm hot}/M_{\rm vir}$'
        for snap, label, color in snap_list_C:
            w = np.where(
                (Mvir[snap] > 1e10) & (HotGas[snap] > 0) & (GalType[snap] == 0)
            )[0]
            if len(w) > 0:
                log_mv = np.log10(Mvir[snap][w])
                frac = HotGas[snap][w] / Mvir[snap][w]
                bc, med, _, _ = binned_median(log_mv, frac, mass_bins)
                valid = ~np.isnan(med)
                ax1.plot(bc[valid], med[valid], '-o', color=color, lw=2,
                         markersize=5, label=label)

    ax1.axhline(y=f_baryon, color='gray', ls=':', lw=1.5, alpha=0.5,
                label=r'$f_b = \Omega_b/\Omega_m$')
    ax1.set_yscale('log')
    ax1.set_xlabel(r'$\log_{10}(M_{\rm vir}/M_{\odot})$', fontsize=13)
    ax1.set_ylabel(gas_label, fontsize=13)
    ax1.set_xlim(10.2, 14.5)
    ax1.set_ylim(1e-4, 0.5)
    leg1 = ax1.legend(loc='lower right', fontsize=9, frameon=False)

    for snap, label, color in snap_list_C:
        w = np.where(
            (Mvir[snap] > 1e10) &
            (tdeplete[snap] > 0) & np.isfinite(tdeplete[snap]) &
            (GalType[snap] == 0)
        )[0]
        if len(w) > 0:
            log_mv = np.log10(Mvir[snap][w])
            
            # === UNIT CONVERSION ===
            # Convert tdeplete from Code Units to Gyr
            # 1 code unit time ~ 977.8/h Gyr
            td = tdeplete[snap][w] * (977.8 / Hubble_h)

            bc, med, p25, p75 = binned_median(log_mv, td, mass_bins)
            valid = ~np.isnan(med)
            ax2.plot(bc[valid], med[valid], '-o', color=color, lw=2,
                     markersize=5, label=label)
            ax2.fill_between(bc[valid], p25[valid], p75[valid],
                             color=color, alpha=0.12)

    ax2.set_yscale('log')
    ax2.set_xlabel(r'$\log_{10}(M_{\rm vir}/M_{\odot})$', fontsize=13)
    # Updated label
    ax2.set_ylabel(r'$t_{\rm deplete}$ [Gyr]', fontsize=13)
    ax2.set_xlim(10.2, 14.5)
    leg2 = ax2.legend(loc='upper right', fontsize=11, frameon=False)

    plt.tight_layout()
    outputFile = OutputDir + 'FigC_cgm_fractions_depletion' + OutputFormat
    plt.savefig(outputFile)
    save_white_copy(fig, outputFile)
    print('  Saved file to', outputFile, '\n')
    plt.close()


    # ==================================================================
    # FIGURE E: Star Formation Efficiency: FFB vs Normal
    # ==================================================================

    print('Plotting Figure E: Star formation efficiency at z~10')

    fig, ax = plt.subplots(figsize=(10, 6))

    snap = snap_z10
    z_snap = redshifts[snap]

    w_ffb = np.where(
        (StellarMass[snap] > 0) & (Mvir[snap] > 0) &
        (FFBRegime_data[snap] == 1) & (GalType[snap] == 0)
    )[0]
    w_normal = np.where(
        (StellarMass[snap] > 0) & (Mvir[snap] > 0) &
        (FFBRegime_data[snap] == 0) & (GalType[snap] == 0)
    )[0]

    if len(w_ffb) > 0:
        eps_ffb = StellarMass[snap][w_ffb] / (f_baryon * Mvir[snap][w_ffb])
        log_mvir_ffb = np.log10(Mvir[snap][w_ffb])
        ax.scatter(log_mvir_ffb, eps_ffb, s=50, c=c_ffb, alpha=0.8,
                   edgecolors=c_ffb_edge, linewidths=0.8,
                   label=f'FFB (N={len(w_ffb)})', zorder=3)

    if len(w_normal) > 0:
        eps_normal = StellarMass[snap][w_normal] / (f_baryon * Mvir[snap][w_normal])
        log_mvir_normal = np.log10(Mvir[snap][w_normal])
        ax.scatter(log_mvir_normal, eps_normal, s=50, c=c_norm, alpha=0.8,
                   edgecolors=c_norm_edge, linewidths=0.8,
                   label=f'Normal (N={len(w_normal)})', zorder=2)

    ax.axhline(y=0.2, color=c_ffb, ls='--', lw=1.5, alpha=0.5,
               label=r'$\varepsilon = 0.2$ (FFB expectation)')
    ax.axhline(y=0.03, color=c_norm, ls='--', lw=1.5, alpha=0.5,
               label=r'$\varepsilon = 0.03$ (normal)')

    M_ffb = ffb_threshold_mass_msun(z_snap)
    ax.axvline(x=np.log10(M_ffb), color='yellow', ls=':', lw=2, alpha=0.7,
               label=fr'$M_{{\rm ffb}}$ = {M_ffb:.1e} $M_\odot$')

    ax.set_yscale('log')
    ax.set_xlabel(r'$\log_{10}(M_{\rm vir}/M_{\odot})$', fontsize=14)
    ax.set_ylabel(r'$\varepsilon \equiv M_*/(\,f_b \, M_{\rm vir})$', fontsize=14)
    ax.set_ylim(1e-4, 2.0)

    leg = ax.legend(loc='lower right', fontsize=10, frameon=False)

    plt.tight_layout()
    outputFile = OutputDir + 'FigE_sfe_ffb_vs_normal' + OutputFormat
    plt.savefig(outputFile)
    save_white_copy(fig, outputFile)
    print('  Saved file to', outputFile, '\n')
    plt.close()


    # ==================================================================
    # FIGURE F: FFB Galaxy Properties at z ~ 10 (3-panel)
    # ==================================================================

    print('Plotting Figure F: FFB galaxy properties at z~10')

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    snap = snap_z10
    z_snap = redshifts[snap]

    w_ffb = np.where(
        (StellarMass[snap] > 0) & (FFBRegime_data[snap] == 1) & (GalType[snap] == 0)
    )[0]
    w_normal = np.where(
        (StellarMass[snap] > 0) & (FFBRegime_data[snap] == 0) & (GalType[snap] == 0)
    )[0]

    log_ms_ffb = np.log10(StellarMass[snap][w_ffb]) if len(w_ffb) > 0 else np.array([])
    log_ms_norm = np.log10(StellarMass[snap][w_normal]) if len(w_normal) > 0 else np.array([])

    # ----- Panel (a): Size-mass relation -----
    if len(w_ffb) > 0:
        Re_ffb = 1.678 * DiskRadius[snap][w_ffb] * 1e3
        ok = Re_ffb > 0
        if np.sum(ok) > 0:
            ax1.scatter(log_ms_ffb[ok], Re_ffb[ok], s=50, c=c_ffb, alpha=0.8,
                        edgecolors=c_ffb_edge, linewidths=0.8,
                        label=f'FFB (N={np.sum(ok)})', zorder=3)

    if len(w_normal) > 0:
        Re_norm = 1.678 * DiskRadius[snap][w_normal] * 1e3
        ok = Re_norm > 0
        if np.sum(ok) > 0:
            ax1.scatter(log_ms_norm[ok], Re_norm[ok], s=50, c=c_norm, alpha=0.8,
                        edgecolors=c_norm_edge, linewidths=0.8,
                        label=f'Normal (N={np.sum(ok)})', zorder=2)

    ax1.axhline(y=0.3, color='yellow', ls='--', lw=1.5, alpha=0.5, label='0.3 kpc (compact)')
    ax1.set_yscale('log')
    ax1.set_xlabel(r'$\log_{10}(M_*/M_{\odot})$', fontsize=13)
    ax1.set_ylabel(r'$R_e$ [kpc]', fontsize=13)
    leg1 = ax1.legend(loc='lower left', fontsize=9, frameon=False)

    # ----- Panel (b): Mass-metallicity relation -----
    if len(w_ffb) > 0:
        ms = StellarMass[snap][w_ffb]
        mz = MetalsStellarMass[snap][w_ffb]
        Z_ratio = (mz / ms) / Z_sun
        ok = Z_ratio > 0
        if np.sum(ok) > 0:
            ax2.scatter(log_ms_ffb[ok], np.log10(Z_ratio[ok]), s=50, c=c_ffb,
                        alpha=0.8, edgecolors=c_ffb_edge, linewidths=0.8,
                        label='FFB', zorder=3)

    if len(w_normal) > 0:
        ms = StellarMass[snap][w_normal]
        mz = MetalsStellarMass[snap][w_normal]
        Z_ratio = (mz / ms) / Z_sun
        ok = Z_ratio > 0
        if np.sum(ok) > 0:
            ax2.scatter(log_ms_norm[ok], np.log10(Z_ratio[ok]), s=50, c=c_norm,
                        alpha=0.8, edgecolors=c_norm_edge, linewidths=0.8,
                        label='Normal', zorder=2)

    ax2.axhline(y=np.log10(0.1), color='yellow', ls='--', lw=1.5, alpha=0.5,
                label=r'$0.1\,Z_{\odot}$')
    ax2.set_xlabel(r'$\log_{10}(M_*/M_{\odot})$', fontsize=13)
    ax2.set_ylabel(r'$\log_{10}(Z_*/Z_{\odot})$', fontsize=13)
    leg2 = ax2.legend(loc='lower right', fontsize=9, frameon=False)

    # ----- Panel (c): sSFR vs M_* -----
    if len(w_ffb) > 0:
        sfr = SfrDisk[snap][w_ffb] + SfrBulge[snap][w_ffb]
        ssfr = sfr / StellarMass[snap][w_ffb]
        ok = ssfr > 0
        if np.sum(ok) > 0:
            ax3.scatter(log_ms_ffb[ok], np.log10(ssfr[ok]), s=50, c=c_ffb,
                        alpha=0.8, edgecolors=c_ffb_edge, linewidths=0.8,
                        label='FFB', zorder=3)

    if len(w_normal) > 0:
        sfr = SfrDisk[snap][w_normal] + SfrBulge[snap][w_normal]
        ssfr = sfr / StellarMass[snap][w_normal]
        ok = ssfr > 0
        if np.sum(ok) > 0:
            ax3.scatter(log_ms_norm[ok], np.log10(ssfr[ok]), s=50, c=c_norm,
                        alpha=0.8, edgecolors=c_norm_edge, linewidths=0.8,
                        label='Normal', zorder=2)

    ax3.set_xlabel(r'$\log_{10}(M_*/M_{\odot})$', fontsize=13)
    ax3.set_ylabel(r'$\log_{10}(\mathrm{sSFR}\ [\mathrm{yr}^{-1}])$', fontsize=13)
    leg3 = ax3.legend(loc='lower left', fontsize=9, frameon=False)

    plt.tight_layout()
    outputFile = OutputDir + 'FigF_ffb_properties_z10' + OutputFormat
    plt.savefig(outputFile)
    save_white_copy(fig, outputFile)
    print('  Saved file to', outputFile, '\n')
    plt.close()


    # ==================================================================
    # FIGURE G: Star Formation Histories of FFB Galaxies
    # ==================================================================

    print('Plotting Figure G: Star formation histories of FFB galaxies')

    snap = snap_z10
    w_ffb_g = np.where(
        (StellarMass[snap] > 0) & (FFBRegime_data[snap] == 1) & (GalType[snap] == 0)
    )[0]
    w_normal_g = np.where(
        (StellarMass[snap] > 0) & (FFBRegime_data[snap] == 0) & (GalType[snap] == 0)
    )[0]

    if len(w_ffb_g) == 0:
        print('  No FFB galaxies found at z~10. Skipping Figure G.\n')
    else:
        N_track = min(10, len(w_ffb_g))
        mass_order = np.argsort(StellarMass[snap][w_ffb_g])[::-1]
        ffb_idx = w_ffb_g[mass_order[:N_track]]
        ffb_gal_ids = GalIdx[snap][ffb_idx]

        N_norm_track = min(5, len(w_normal_g))
        if N_norm_track > 0:
            mass_order_n = np.argsort(StellarMass[snap][w_normal_g])[::-1]
            norm_idx = w_normal_g[mass_order_n[:N_norm_track]]
            norm_gal_ids = GalIdx[snap][norm_idx]
        else:
            norm_gal_ids = np.array([], dtype=np.int64)

        cosmic_times = {s: cosmic_time_gyr(redshifts[s]) for s in fig_g_snaps}
        ffb_tracks = {int(gid): ([], []) for gid in ffb_gal_ids}
        norm_tracks = {int(gid): ([], []) for gid in norm_gal_ids}

        for s in fig_g_snaps:
            gids = GalIdx[s]
            sfr_total = SfrDisk[s] + SfrBulge[s]
            t = cosmic_times[s]

            for gid in ffb_gal_ids:
                match = np.where(gids == gid)[0]
                if len(match) > 0:
                    ffb_tracks[int(gid)][0].append(t)
                    ffb_tracks[int(gid)][1].append(sfr_total[match[0]])

            for gid in norm_gal_ids:
                match = np.where(gids == gid)[0]
                if len(match) > 0:
                    norm_tracks[int(gid)][0].append(t)
                    norm_tracks[int(gid)][1].append(sfr_total[match[0]])

        fig, ax = plt.subplots(figsize=(12, 6))

        for i, gid in enumerate(ffb_gal_ids):
            times, sfrs = ffb_tracks[int(gid)]
            if len(times) > 1:
                sfrs = np.array(sfrs)
                sfrs_plot = np.where(sfrs > 0, sfrs, 1e-5)
                lbl = 'FFB galaxies' if i == 0 else None
                ax.plot(times, sfrs_plot, '-', color=c_ffb, alpha=0.6, lw=1.5,
                        label=lbl, zorder=3)

        for i, gid in enumerate(norm_gal_ids):
            times, sfrs = norm_tracks[int(gid)]
            if len(times) > 1:
                sfrs = np.array(sfrs)
                sfrs_plot = np.where(sfrs > 0, sfrs, 1e-5)
                lbl = 'Normal galaxies' if i == 0 else None
                ax.plot(times, sfrs_plot, '--', color=c_norm, alpha=0.6, lw=1.5,
                        label=lbl, zorder=2)

        ax.set_yscale('log')
        ax.set_xlabel('Cosmic time [Gyr]', fontsize=14)
        ax.set_ylabel(r'SFR [$M_{\odot}\,\mathrm{yr}^{-1}$]', fontsize=14)
        ax.set_xlim(0, cosmic_time_gyr(0))

        ax_top = ax.twiny()
        z_ticks = [20, 10, 6, 4, 3, 2, 1, 0.5, 0]
        t_ticks = [cosmic_time_gyr(z) for z in z_ticks]
        ax_top.set_xlim(ax.get_xlim())
        ax_top.set_xticks(t_ticks)
        ax_top.set_xticklabels([str(z) for z in z_ticks])
        ax_top.set_xlabel('Redshift', fontsize=13)

        leg = ax.legend(loc='upper right', fontsize=11, frameon=False)

        plt.tight_layout()
        outputFile = OutputDir + 'FigG_sfh_ffb_galaxies' + OutputFormat
        plt.savefig(outputFile)
        save_white_copy(fig, outputFile)
        print('  Saved file to', outputFile, '\n')
        plt.close()


    # ==================================================================

    print('=' * 60)
    print('All discussion figures complete.')
#!/usr/bin/env python

import h5py as h5
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import argparse
import glob

import warnings
warnings.filterwarnings("ignore")

# ========================== USER OPTIONS ==========================
OutputFormat = '.pdf'
plt.rcParams["figure.figsize"] = (14, 12) # Adjusted for 2x2 grid
plt.rcParams["figure.dpi"] = 96
plt.rcParams["font.size"] = 14

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['axes.titlecolor'] = 'black'
plt.rcParams['text.color'] = 'black'
plt.rcParams['legend.facecolor'] = 'white'
plt.rcParams['legend.edgecolor'] = 'black'

# BK25 FFB Critical Threshold (Code units: G * 3100 Msun / pc^2)
G_CRIT_VAL = 3100.0  
# ==================================================================

def get_script_dir():
    return os.path.dirname(os.path.abspath(__file__))

def read_simulation_params(filepath):
    params = {}
    with h5.File(filepath, 'r') as f:
        sim = f['Header/Simulation']
        params['Hubble_h'] = float(sim.attrs['hubble_h'])
        params['BoxSize'] = float(sim.attrs['box_size'])
        
        runtime = f['Header/Runtime']
        params['VolumeFraction'] = float(runtime.attrs['frac_volume_processed'])
        
        params['snapshot_redshifts'] = np.array(f['Header/snapshot_redshifts'])
        params['output_snapshots'] = np.array(f['Header/output_snapshots'])

        snap_groups = [key for key in f.keys() if key.startswith('Snap_')]
        snap_numbers = sorted([int(s.replace('Snap_', '')) for s in snap_groups])
        params['available_snapshots'] = snap_numbers
        
    return params

def get_snapshot_redshift(params, snap_num):
    if snap_num < len(params['snapshot_redshifts']):
        return params['snapshot_redshifts'][snap_num]
    return None

def read_hdf(filepaths, snap_num, param):
    data_list = []
    for filepath in filepaths:
        with h5.File(filepath, 'r') as f:
            if snap_num in f and param in f[snap_num]:
                data = np.array(f[snap_num][param])
                if data.size > 0:
                    data_list.append(data)
    if not data_list:
        return np.array([])
    return np.concatenate(data_list)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Plot SAGE BK25 FFB Diagnostics')
    parser.add_argument('input_pattern', nargs='?', default='./output/millennium_ffb_bk25/model_*.hdf5')
    parser.add_argument('-o', '--output-dir', type=str, default=None)
    return parser.parse_args()

# ==================================================================
# BOOTSTRAP HELPER FUNCTIONS
# ==================================================================

def compute_binned_bootstrap(x, y, bins, n_boot=200):
    """Calculates median and 1-sigma (16th-84th percentile) bootstrap errors for mass bins."""
    bin_idx = np.digitize(x, bins) - 1
    medians = np.full(len(bins)-1, np.nan)
    lowers = np.full(len(bins)-1, np.nan)
    uppers = np.full(len(bins)-1, np.nan)
    
    for i in range(len(bins)-1):
        in_bin = y[bin_idx == i]
        in_bin = in_bin[~np.isnan(in_bin)]
        if len(in_bin) > 5:
            medians[i] = np.median(in_bin)
            boot = np.median(np.random.choice(in_bin, size=(n_boot, len(in_bin)), replace=True), axis=1)
            lowers[i] = np.percentile(boot, 16)
            uppers[i] = np.percentile(boot, 84)
        elif len(in_bin) > 0:
            medians[i] = np.median(in_bin)
            lowers[i] = medians[i]
            uppers[i] = medians[i]
            
    return medians, lowers, uppers

def compute_redshift_bootstrap(df_mask, target_col, is_fraction=False, n_boot=200):
    """Calculates median/mean and 1-sigma bootstrap errors across redshift snapshots."""
    z_bins = np.sort(df_mask['Redshift'].unique())
    stats = np.full(len(z_bins), np.nan)
    lowers = np.full(len(z_bins), np.nan)
    uppers = np.full(len(z_bins), np.nan)
    
    for i, z in enumerate(z_bins):
        in_bin = df_mask[df_mask['Redshift'] == z][target_col].values
        in_bin = in_bin[~np.isnan(in_bin)]
        if len(in_bin) > 5:
            if is_fraction:
                stats[i] = np.mean(in_bin)
                boot = np.mean(np.random.choice(in_bin, size=(n_boot, len(in_bin)), replace=True), axis=1)
            else:
                stats[i] = np.median(in_bin)
                boot = np.median(np.random.choice(in_bin, size=(n_boot, len(in_bin)), replace=True), axis=1)
            lowers[i] = np.percentile(boot, 16)
            uppers[i] = np.percentile(boot, 84)
        elif len(in_bin) > 0:
            stats[i] = np.mean(in_bin) if is_fraction else np.median(in_bin)
            lowers[i] = stats[i]
            uppers[i] = stats[i]
            
    return z_bins, stats, lowers, uppers

# ==================================================================
# DIAGNOSTIC PLOTTING FUNCTION
# ==================================================================

def plot_bk25_diagnostics(df, outdir):
    fig, axs = plt.subplots(2, 2)
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    unique_z = np.sort(df['Redshift'].unique())
    target_z = [0.0, 1.0, 2.0, 6.0, 8.0, 10.0] 
    redshifts_to_plot = [unique_z[np.argmin(np.abs(unique_z - tz))] for tz in target_z]
    redshifts_to_plot = sorted(list(set(redshifts_to_plot)))
    
    colors = plt.cm.plasma_r(np.linspace(0, 0.9, len(redshifts_to_plot)))
    bins = np.logspace(np.log10(1e9), np.log10(1e14), 20)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # --------------------------------------------------------------------------
    # Plot 1: Concentration vs Halo Mass
    # --------------------------------------------------------------------------
    ax = axs[0, 0]
    for z, color in zip(redshifts_to_plot, colors):
        mask = (np.abs(df['Redshift'] - z) < 0.01) & (df['Type'] == 0)
        if mask.sum() == 0: continue
        
        medians, lowers, uppers = compute_binned_bootstrap(
            df.loc[mask, 'Mvir'].values, df.loc[mask, 'Concentration'].values, bins)
        
        ax.plot(bin_centers, medians, label=f'z = {z:.1f}', color=color, lw=2)
        ax.fill_between(bin_centers, lowers, uppers, color=color, alpha=0.2, lw=0)

    ax.set_xscale('log')
    ax.set_xlabel(r'$\rm M_{vir}$ [$M_{\odot}\ h^{-1}$]')
    ax.set_ylabel(r'$\rm c$')
    ax.legend(frameon=False)

    # --------------------------------------------------------------------------
    # Plot 2: g_max vs Halo Mass (The BK25 Threshold)
    # --------------------------------------------------------------------------
    ax = axs[0, 1]
    for z, color in zip(redshifts_to_plot, colors):
        mask = (np.abs(df['Redshift'] - z) < 0.01) & (df['Type'] == 0)
        if mask.sum() == 0: continue
        
        medians, lowers, uppers = compute_binned_bootstrap(
            df.loc[mask, 'Mvir'].values, df.loc[mask, 'g_max'].values, bins)
        
        ax.plot(bin_centers, medians, label=f'z = {z:.1f}', color=color, lw=2)
        ax.fill_between(bin_centers, lowers, uppers, color=color, alpha=0.2, lw=0)

    ax.axhline(G_CRIT_VAL, color='red', linestyle='--', lw=2, label=r'Threshold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\rm M_{vir}$ [$M_{\odot}\ h^{-1}$]')
    ax.set_ylabel(r'$\rm g_{max}\ [M_{\odot}\ pc^{-2}]$')
    ax.legend(frameon=False)

    # --------------------------------------------------------------------------
    # Plot 3: g_max vs Redshift (Phase Space)
    # --------------------------------------------------------------------------
    ax = axs[1, 0]
    mass_bins = [(1e10, 1e11), (1e11, 1e12), (1e12, 1e13), (1e13, 1e14), (1e14, 1e15)]
    m_colors = ['blue', 'green', 'purple', 'orange', 'brown']
    
    for (m_low, m_high), color in zip(mass_bins, m_colors):
        mask = (df['Mvir'] >= m_low) & (df['Mvir'] < m_high) & (df['Type'] == 0)
        if mask.sum() == 0: continue
        
        z_bins, medians, lowers, uppers = compute_redshift_bootstrap(df[mask], 'g_max', is_fraction=False)
        
        ax.plot(z_bins, medians, color=color, lw=2, 
                label=f'$10^{{{int(np.log10(m_low))}}} - 10^{{{int(np.log10(m_high))}}} M_{{\odot}}$')
        ax.fill_between(z_bins, lowers, uppers, color=color, alpha=0.2, lw=0)

    ax.axhline(G_CRIT_VAL, color='red', linestyle='--', lw=2)
    ax.set_yscale('log')
    ax.set_xlabel('$z$')
    ax.set_ylabel(r'Median $g_{\rm max}\ [M_{\odot}\ pc^{-2}]$')
    ax.set_ylim(2e2, 5e4)
    ax.legend(frameon=False, ncol=2)

    # --------------------------------------------------------------------------
    # Plot 4: Fraction of FFB Galaxies vs Redshift
    # --------------------------------------------------------------------------
    ax = axs[1, 1]
    for (m_low, m_high), color in zip(mass_bins, m_colors):
        mask = (df['Mvir'] >= m_low) & (df['Mvir'] < m_high) & (df['Type'] == 0)
        if mask.sum() == 0: continue
        
        z_bins, fractions, lowers, uppers = compute_redshift_bootstrap(df[mask], 'FFBRegime', is_fraction=True)
            
        ax.plot(z_bins, fractions, color=color, lw=2, 
                label=f'$10^{{{int(np.log10(m_low))}}} - 10^{{{int(np.log10(m_high))}}} M_{{\odot}}$')
        ax.fill_between(z_bins, lowers, uppers, color=color, alpha=0.2, lw=0)

    ax.set_xlabel('$z$')
    ax.set_ylabel('Fraction of Halos in FFB Regime')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=False)

    plt.tight_layout()
    outpath = os.path.join(outdir, f'BK25_FFB_Diagnostics{OutputFormat}')
    plt.savefig(outpath)
    print(f"Plot saved successfully as '{outpath}'")

def plot_evolution_grid(df, outdir, dilute=7500, g_crit_val=3100.0):
    unique_z = np.sort(df['Redshift'].unique())
    target_z = [10.0, 6.0, 2.0, 0.0]
    
    redshifts_to_plot = []
    for tz in target_z:
        if len(unique_z) > 0:
            closest_z = unique_z[np.argmin(np.abs(unique_z - tz))]
            if closest_z not in redshifts_to_plot:
                redshifts_to_plot.append(closest_z)
                
    while len(redshifts_to_plot) < 4 and len(unique_z) > len(redshifts_to_plot):
        remaining = [z for z in unique_z if z not in redshifts_to_plot]
        redshifts_to_plot.append(remaining[0])
    redshifts_to_plot = sorted(redshifts_to_plot[:4], reverse=True)

    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    plt.subplots_adjust(hspace=0.25, wspace=0.25, right=0.88)
    
    c_min, c_max = df['Concentration'].quantile(0.01), df['Concentration'].quantile(0.99)
    if c_min <= 0: c_min = 1.0

    for ax, z in zip(axs.flatten(), redshifts_to_plot):
        mask = (np.abs(df['Redshift'] - z) < 0.01) & (df['Type'] == 0)
        df_z = df[mask]
        
        if len(df_z) == 0:
            ax.text(0.5, 0.5, f"No Data at z={z:.1f}", ha='center', va='center')
            continue
            
        if len(df_z) > dilute:
            df_plot = df_z.sample(n=dilute, random_state=42)
        else:
            df_plot = df_z
            
        ffb_mask = df_plot['g_max'] > g_crit_val
        df_non_ffb = df_plot[~ffb_mask]
        df_ffb = df_plot[ffb_mask]

        sc1 = ax.scatter(df_non_ffb['Mvir'], df_non_ffb['g_max'], 
                         c=df_non_ffb['Concentration'], cmap='plasma_r',
                         s=15, alpha=0.6, marker='o', 
                         vmin=c_min, vmax=c_max)
        
        if len(df_ffb) > 0:
            sc2 = ax.scatter(df_ffb['Mvir'], df_ffb['g_max'], 
                             c=df_ffb['Concentration'], cmap='plasma_r',
                             s=150, alpha=1.0, marker='*', edgecolor='k', linewidth=0.5,
                             vmin=c_min, vmax=c_max, label='FFB Galaxies')

        ax.axhline(g_crit_val, color='gray', linestyle='--', lw=1.0, zorder=10)
        
        if ax == axs.flatten()[0]:
            ax.text(0.05, 0.95, r"$\rm g_{crit}\ =\ 3100\ [M_{\odot} / pc^2]$", transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax.legend(frameon=False, loc='lower right')
            
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(1e10, 5e13)
        ax.set_ylim(1e1, 1e5) 
        
        ax.set_title(f"z = {z:.2f}", fontsize=14)
        ax.set_xlabel(r'$\rm M_{vir}$ $[M_{\odot}\ h^{-1}]$', fontsize=12)
        ax.set_ylabel(r'$\rm g_{max}\ [M_{\odot} / pc^2]$')

    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sc1, cax=cbar_ax)
    cbar.set_label('Halo Concentration', fontsize=14)

    outpath = os.path.join(outdir, f'BK25_Evolution_Grid{OutputFormat}')
    plt.savefig(outpath, bbox_inches='tight')
    print(f"Plot saved successfully as '{outpath}'")

def plot_bk25_figure1(df, outdir, g_crit_val=3100.0):
    c = df['Concentration'].clip(lower=1.0) 
    mu_c = np.log(1.0 + c) - (c / (1.0 + c))
    concentration_boost = (c * c / 2.0) / mu_c
    
    df['g_vir'] = df['g_max'] / concentration_boost
    
    df_clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['g_vir', 'g_max', 'Mvir'])
    df_clean = df_clean[(df_clean['g_vir'] > 0) & (df_clean['g_max'] > 0)]

    unique_z = np.sort(df_clean['Redshift'].unique())
    target_z = [0.0, 5.0, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0]
    
    redshifts_to_plot = []
    for tz in target_z:
        if len(unique_z) > 0:
            closest_z = unique_z[np.argmin(np.abs(unique_z - tz))]
            if closest_z not in redshifts_to_plot:
                redshifts_to_plot.append(closest_z)
    redshifts_to_plot = sorted(redshifts_to_plot)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = plt.cm.plasma(np.linspace(0, 0.9, len(redshifts_to_plot)))
    bins = np.logspace(np.log10(1e8), np.log10(1e13), 25)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # --------------------------------------------------------------------------
    # Panel 1: Virial Acceleration (g_vir / G)
    # --------------------------------------------------------------------------
    ax1 = axs[0]
    for z, color in zip(redshifts_to_plot, colors):
        mask = (np.abs(df_clean['Redshift'] - z) < 0.01) & (df_clean['Type'] == 0)
        if mask.sum() == 0: continue
        
        medians, lowers, uppers = compute_binned_bootstrap(
            df_clean.loc[mask, 'Mvir'].values, df_clean.loc[mask, 'g_vir'].values, bins)
        
        ax1.plot(bin_centers, medians, label=f'$z = {z:.1f}$', color=color, lw=2.5)
        ax1.fill_between(bin_centers, lowers, uppers, color=color, alpha=0.2, lw=0)

    ax1.axhline(g_crit_val, color='black', linestyle='--', lw=2, label=r'Threshold ($g_{\rm crit}/G$)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(1e10, 1e15)
    ax1.set_ylim(1e-1, 1e5)
    ax1.set_xlabel(r'$M_{\rm vir}$ [$M_{\odot}\ h^{-1}$]', fontsize=14)
    ax1.set_ylabel(r'$g_{\rm vir}/G$ [$M_{\odot} / {\rm pc}^2$]', fontsize=14)
    ax1.legend(loc='lower right', fontsize=12, frameon=False, ncol=3)

    # --------------------------------------------------------------------------
    # Panel 2: Maximum Acceleration (g_max / G)
    # --------------------------------------------------------------------------
    ax2 = axs[1]
    for z, color in zip(redshifts_to_plot, colors):
        mask = (np.abs(df_clean['Redshift'] - z) < 0.01) & (df_clean['Type'] == 0)
        if mask.sum() == 0: continue
        
        medians, lowers, uppers = compute_binned_bootstrap(
            df_clean.loc[mask, 'Mvir'].values, df_clean.loc[mask, 'g_max'].values, bins)
        
        ax2.plot(bin_centers, medians, color=color, lw=2.5)
        ax2.fill_between(bin_centers, lowers, uppers, color=color, alpha=0.2, lw=0)

    ax2.axhline(g_crit_val, color='black', linestyle='--', lw=2)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlim(1e10, 1e15)
    ax2.set_ylim(1e-1, 1e5)
    ax2.set_xlabel(r'$M_{\rm vir}$ [$M_{\odot}\ h^{-1}$]', fontsize=14)
    ax2.set_ylabel(r'$g_{\rm max}/G$ [$M_{\odot} / {\rm pc}^2$]', fontsize=14)

    ax2.fill_between([1e10, 1e15], g_crit_val, 1e5, color='red', alpha=0.05)
    ax2.text(1.5e10, g_crit_val * 4.5, 'Feedback-Free Burst Regime', 
             color='red', alpha=0.6, fontsize=12, weight='bold')

    plt.tight_layout()
    outpath = os.path.join(outdir, f'BK25_Figure1_Replication{OutputFormat}')
    plt.savefig(outpath, bbox_inches='tight')
    print(f"Plot saved successfully as '{outpath}'")

def plot_gvir_contours(df, outdir, g_crit_val=3100.0):
    """
    Plots the phase space of log10(Mvir) vs Redshift, highlighting FFB galaxies
    and theoretical constant acceleration contours.
    FFB galaxies are color-coded by their dark matter concentration.
    """
    fig, ax = plt.subplots(figsize=(11, 8)) # Slightly wider to accommodate colorbar
    
    # 1. Scatter Plot
    df_cen = df[df['Type'] == 0].copy()
    
    df_normal = df_cen[df_cen['FFBRegime'] == 0]
    if len(df_normal) > 50000:
        df_normal = df_normal.sample(50000, random_state=42)
        
    df_ffb = df_cen[df_cen['FFBRegime'] == 1]
    if len(df_ffb) > 20000:
        df_ffb = df_ffb.sample(20000, random_state=42)
        
    ax.scatter(df_normal['Redshift'], np.log10(df_normal['Mvir']), 
               c='gray', s=1.5, alpha=0.15, zorder=1)

    # Print concentration stats for FFB galaxies
    
    if len(df_ffb) > 0:
        print(f"FFB Galaxies: {len(df_ffb)}")
        print(f"  Concentration (c) - min: {df_ffb['Concentration'].min():.2f}, max: {df_ffb['Concentration'].max():.2f}, mean: {df_ffb['Concentration'].mean():.2f}")
               
    # Color code FFB galaxies by Concentration
    sc = ax.scatter(df_ffb['Redshift'], np.log10(df_ffb['Mvir']), 
                    c=df_ffb['Concentration'], cmap='plasma', s=10, alpha=0.8, zorder=2)
    
    # Add Colorbar
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label('FFB Halo Concentration ($c$)', fontsize=14)
               
    # 2. Theoretical Curves (BK25 Eq 8)
    z_vals = np.linspace(0, 15, 100)
    
    Om = 0.3158
    Ol = 0.6842
    E_z_sq = Om * (1 + z_vals)**3 + Ol
    Om_z = Om * (1 + z_vals)**3 / E_z_sq
    x = Om_z - 1.0
    Delta_c = 200
    Delta_m = Delta_c / Om_z 
    
    def calc_contour(g_target):
        # First principles calculation tailored to your exact simulation cosmology
        G_phys = 4.3009e-3  # pc M_sun^-1 (km/s)^2
        
        # H(z) in km/s/Mpc
        H_z = 100.0 * Hubble_h * np.sqrt(Om * (1 + z_vals)**3 + Ol)
        # Convert H(z) to km/s/pc
        H_z_pc = H_z / 1e6  
        
        # Calculate Mvir directly from g_vir, H(z), and SAGE's 200c overdensity
        Mvir = (g_target)**3 * ( (2.0 * G_phys) / (200.0 * H_z_pc**2) )**2
        return np.log10(Mvir)
        
    # ax.plot(z_vals, calc_contour(100), color='tab:blue', lw=2.5, label=r'$g_{\rm vir}/G = 100$', zorder=3)
    # ax.plot(z_vals, calc_contour(250), color='tab:orange', lw=2.5, label=r'$g_{\rm vir}/G = 250$', zorder=3)
    # ax.plot(z_vals, calc_contour(500), color='tab:green', lw=2.5, label=r'$g_{\rm vir}/G = 500$', zorder=3)
    # ax.plot(z_vals, calc_contour(1000), color='tab:red', lw=2.5, label=r'$g_{\rm vir}/G = 1000$', zorder=3)
    
    # 3. Theoretical g_max = g_crit curve (Fixed Concentration)
    c_fixed_1 = 3.0
    mu_c = np.log(1.0 + c_fixed_1) - (c_fixed_1 / (1.0 + c_fixed_1))
    boost_factor = (c_fixed_1**2 / 2.0) / mu_c
    
    g_vir_req = g_crit_val / boost_factor
    
    ax.plot(z_vals, calc_contour(g_vir_req), color='black', lw=3.5, 
            label=f'$g_{{\\rm max}} = g_{{\\rm crit}}$ ($c={c_fixed_1}$)', zorder=5)
    
    # 3. Theoretical g_max = g_crit curve (Fixed Concentration)
    c_fixed_2 = 4.0
    mu_c = np.log(1.0 + c_fixed_2) - (c_fixed_2 / (1.0 + c_fixed_2))
    boost_factor = (c_fixed_2**2 / 2.0) / mu_c
    
    g_vir_req = g_crit_val / boost_factor
    
    ax.plot(z_vals, calc_contour(g_vir_req), color='green', lw=3.5, 
            label=f'$g_{{\\rm max}} = g_{{\\rm crit}}$ ($c={c_fixed_2}$)', zorder=5)
    
    # 3. Theoretical g_max = g_crit curve (Fixed Concentration)
    c_fixed_3 = 10.0
    mu_c = np.log(1.0 + c_fixed_3) - (c_fixed_3 / (1.0 + c_fixed_3))
    boost_factor = (c_fixed_3**2 / 2.0) / mu_c
    
    g_vir_req = g_crit_val / boost_factor
    
    ax.plot(z_vals, calc_contour(g_vir_req), color='red', lw=3.5, 
            label=f'$g_{{\\rm max}} = g_{{\\rm crit}}$ ($c={c_fixed_3}$)', zorder=5)

    # 4. Li et al. 2024 (M_v, ffb) curve
    logM_Li24 = 10.8 - 6.2 * np.log10((1 + z_vals) / 10.0)
    ax.plot(z_vals, logM_Li24, color='mediumblue', linestyle='--', lw=3, label=r'Li+24 ($M_{\rm v, ffb}$)', zorder=4)

    # Formatting
    ax.set_xlim(-0.5, 15.5)
    ax.set_ylim(10.0, 15.0)
    
    ax.set_xlabel('Redshift', fontsize=16)
    ax.set_ylabel(r'$\log_{10}(M_{\rm vir} / M_{\odot})$', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    leg = ax.legend(loc='upper right', fontsize=12, frameon=False)
    for handle in leg.legend_handles:
        try:
            handle.set_alpha(1.0)
        except:
            pass 
            
    plt.tight_layout()
    outpath = os.path.join(outdir, f'BK25_Gvir_Contours{OutputFormat}')
    plt.savefig(outpath, bbox_inches='tight')
    print(f"Plot saved successfully as '{outpath}'")

# ==================================================================
if __name__ == '__main__':

    print('Running BK25 FFB Diagnostics')
    args = parse_arguments()
    
    file_list = glob.glob(args.input_pattern)
    file_list.sort()

    if not file_list:
        print(f"Error: No files found matching: {args.input_pattern}")
        sys.exit(1)

    first_file = os.path.abspath(file_list[0])
    sim_params = read_simulation_params(first_file)
    Hubble_h = sim_params['Hubble_h']

    OutputDir = args.output_dir if args.output_dir else os.path.join(os.path.dirname(first_file), 'plots')
    if not os.path.exists(OutputDir):
        os.makedirs(OutputDir)

    print(f"Found {len(file_list)} model files. Reading available snapshots...")

    all_data = []

    for snap_num in sim_params['available_snapshots']:
        Snapshot = f'Snap_{snap_num}'
        redshift = get_snapshot_redshift(sim_params, snap_num)
        
        Mvir = read_hdf(file_list, Snapshot, 'Mvir') 
        if len(Mvir) == 0:
            continue
            
        Mvir = Mvir * 1.0e10 / Hubble_h 
        Type = read_hdf(file_list, Snapshot, 'Type')
        Rvir_code = read_hdf(file_list, Snapshot, 'Rvir')

        M_phys = Mvir * 1.0e10 / Hubble_h       
        R_phys = Rvir_code * 1.0e6 / Hubble_h   

        valid = R_phys > 0
        g_vir_phys = np.zeros_like(M_phys)
        g_vir_phys[valid] = M_phys[valid] / (R_phys[valid]**2)
        
        g_max_sage = read_hdf(file_list, Snapshot, 'g_max')
        G_code = 43.0071 
        
        g_max = (g_max_sage / G_code) * 0.01 * Hubble_h
        Concentration = read_hdf(file_list, Snapshot, 'Concentration')
        FFBRegime = read_hdf(file_list, Snapshot, 'FFBRegime')
        
        if len(g_max) == 0 or len(Concentration) == 0 or len(FFBRegime) == 0:
            print(f"  Warning: Missing g_max, Concentration, or FFBRegime in {Snapshot}. Ensure core_save.c is writing them.")
            continue
            
        df_snap = pd.DataFrame({
            'Redshift': np.full(len(Mvir), redshift),
            'Type': Type,
            'Mvir': Mvir,
            'g_vir': g_vir_phys,
            'Concentration': Concentration,
            'g_max': g_max,
            'FFBRegime': FFBRegime
        })
        
        all_data.append(df_snap)
        print(f"  Read {Snapshot} (z={redshift:.2f}) - {len(Mvir)} galaxies")

    if not all_data:
        print("Error: No valid data extracted.")
        sys.exit(1)

    full_df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal galaxies processed: {len(full_df)}")

    print("Generating diagnostic plots...")
    
    plot_bk25_diagnostics(full_df, OutputDir)
    plot_evolution_grid(full_df, OutputDir)
    plot_bk25_figure1(full_df, OutputDir, g_crit_val=G_CRIT_VAL)
    plot_gvir_contours(full_df, OutputDir, g_crit_val=G_CRIT_VAL)
#!/usr/bin/env python
"""
Compare CGM Density Profiles
=============================
Comprehensive comparison of the three CGM density profile options:
  - Uniform (CGMDensityProfile = 0)
  - NFW (CGMDensityProfile = 1)
  - Beta profile (CGMDensityProfile = 2)

Usage:
    python compare_cgm_profiles.py
"""

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy import stats

import warnings
warnings.filterwarnings("ignore")

# ========================== CONFIGURATION ==========================

# Output directories for each profile
UNIFORM_DIR = './output/millennium_uniform/'
NFW_DIR = './output/millennium_NFW/'
BETA_DIR = './output/millennium_beta/'

MODEL_FILE = 'model_0.hdf5'
OUTPUT_DIR = './output/cgm_profile_comparison/'

# Plotting settings
OUTPUT_FORMAT = '.pdf'
DILUTE = 5000
SEED = 2222

# Solar mass in grams
_MSUN_CGS = 1.989e33

# Properties that need mass conversion
_MASS_PROPS = frozenset({
    'CentralMvir', 'Mvir', 'StellarMass', 'BulgeMass', 'BlackHoleMass',
    'MetalsStellarMass', 'MetalsColdGas', 'MetalsEjectedMass',
    'MetalsHotGas', 'MetalsCGMgas', 'ColdGas', 'HotGas', 'CGMgas',
    'EjectedMass', 'H2gas', 'H1gas', 'IntraClusterStars',
})

# Properties to load
PROPERTIES = [
    'StellarMass', 'BulgeMass', 'ColdGas', 'HotGas', 'CGMgas',
    'EjectedMass', 'H2gas', 'H1gas', 'BlackHoleMass',
    'IntraClusterStars', 'CentralMvir', 'Mvir',
    'MetalsStellarMass', 'MetalsColdGas', 'MetalsHotGas', 'MetalsCGMgas',
    'SfrDisk', 'SfrBulge', 'Vvir', 'Rvir',
    'Type', 'Regime',
    'tcool', 'tff', 'tcool_over_tff', 'tdeplete', 'RcoolToRvir',
]


# ========================== DATA I/O ==========================

def find_model_files(directory):
    """Find all model_*.hdf5 files in directory."""
    import glob
    pattern = os.path.join(directory, 'model_*.hdf5')
    files = sorted(glob.glob(pattern))
    if not files:
        single = os.path.join(directory, MODEL_FILE)
        if os.path.exists(single):
            files = [single]
    return files


def read_header(directory):
    """Read simulation parameters from HDF5 header."""
    files = find_model_files(directory)
    if not files:
        return None

    try:
        with h5.File(files[0], 'r') as f:
            sim = f['Header/Simulation']
            runtime = f['Header/Runtime']

            header = {
                'hubble_h': float(sim.attrs['hubble_h']),
                'box_size': float(sim.attrs['box_size']),
                'last_snap_nr': int(sim.attrs['LastSnapshotNr']),
                'unit_mass_in_g': float(runtime.attrs['UnitMass_in_g']),
                'redshifts': list(f['Header/snapshot_redshifts'][:]),
            }

        # Sum volume fraction across all files
        total_fvp = 0.0
        for fp in files:
            with h5.File(fp, 'r') as f:
                total_fvp += float(f['Header/Runtime'].attrs['frac_volume_processed'])
        header['volume_fraction'] = total_fvp

    except Exception as e:
        print(f"Warning: could not read header from {directory}: {e}")
        return None

    return header


def load_model(directory, snapshot, properties, mass_convert):
    """Load galaxy properties from model HDF5 files."""
    filepaths = find_model_files(directory)
    if not filepaths:
        print(f"  Warning: no model files found in {directory}")
        return {}

    chunks = {prop: [] for prop in properties}
    found_snap = False

    for fp in filepaths:
        try:
            with h5.File(fp, 'r') as f:
                if snapshot not in f:
                    continue
                found_snap = True
                grp = f[snapshot]
                for prop in properties:
                    if prop in grp:
                        chunks[prop].append(np.array(grp[prop]))
        except Exception as e:
            print(f"  Warning: could not read {fp}: {e}")
            continue

    if not found_snap:
        return {}

    data = {}
    for prop in properties:
        if chunks[prop]:
            arr = np.concatenate(chunks[prop])
            if prop in _MASS_PROPS:
                arr = arr * mass_convert
            data[prop] = arr

    return data


def setup_style():
    """Configure matplotlib style."""
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.figsize': (10, 8),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


# ========================== PLOTTING FUNCTIONS ==========================

def stellar_mass_function(ax, data, volume, color, label, linestyle='-'):
    """Plot stellar mass function."""
    stellar = data.get('StellarMass', np.array([]))
    if len(stellar) == 0:
        return

    stellar = stellar[stellar > 0]
    log_stellar = np.log10(stellar)

    bins = np.arange(6, 12.5, 0.2)
    counts, edges = np.histogram(log_stellar, bins=bins)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = edges[1] - edges[0]

    phi = counts / volume / bin_width
    phi[phi == 0] = np.nan

    ax.plot(bin_centers, np.log10(phi), color=color, label=label,
            linestyle=linestyle, linewidth=2)


def cold_gas_vs_stellar(ax, data, color, label, dilute=DILUTE):
    """Plot cold gas mass vs stellar mass."""
    stellar = data.get('StellarMass', np.array([]))
    cold = data.get('ColdGas', np.array([]))
    if len(stellar) == 0 or len(cold) == 0:
        return

    mask = (stellar > 1e7) & (cold > 0)
    stellar = stellar[mask]
    cold = cold[mask]

    if len(stellar) > dilute:
        np.random.seed(SEED)
        idx = np.random.choice(len(stellar), dilute, replace=False)
        stellar = stellar[idx]
        cold = cold[idx]

    ax.scatter(np.log10(stellar), np.log10(cold), s=1, alpha=0.3,
               color=color, label=label, rasterized=True)


def cgm_gas_vs_mvir(ax, data, color, label, dilute=DILUTE):
    """Plot CGM gas mass vs halo mass."""
    mvir = data.get('Mvir', np.array([]))
    cgm = data.get('CGMgas', np.array([]))
    regime = data.get('Regime', np.array([]))

    if len(mvir) == 0 or len(cgm) == 0:
        return

    # Only CGM regime galaxies (Regime == 0)
    if len(regime) > 0:
        mask = (mvir > 1e9) & (cgm > 0) & (regime == 0)
    else:
        mask = (mvir > 1e9) & (cgm > 0)

    mvir = mvir[mask]
    cgm = cgm[mask]

    if len(mvir) > dilute:
        np.random.seed(SEED)
        idx = np.random.choice(len(mvir), dilute, replace=False)
        mvir = mvir[idx]
        cgm = cgm[idx]

    ax.scatter(np.log10(mvir), np.log10(cgm), s=1, alpha=0.3,
               color=color, label=label, rasterized=True)


def tcool_tff_distribution(ax, data, color, label):
    """Plot histogram of tcool/tff ratio."""
    ratio = data.get('tcool_over_tff', np.array([]))
    regime = data.get('Regime', np.array([]))

    if len(ratio) == 0:
        return

    # Only CGM regime galaxies with valid ratios
    if len(regime) > 0:
        mask = (ratio > 0) & (ratio < 1000) & (regime == 0)
    else:
        mask = (ratio > 0) & (ratio < 1000)

    ratio = ratio[mask]
    if len(ratio) == 0:
        return

    log_ratio = np.log10(ratio)

    bins = np.linspace(-2, 3, 50)
    ax.hist(log_ratio, bins=bins, density=True, alpha=0.5,
            color=color, label=label, histtype='stepfilled')
    ax.hist(log_ratio, bins=bins, density=True,
            color=color, histtype='step', linewidth=1.5)


def rcool_rvir_distribution(ax, data, color, label):
    """Plot histogram of rcool/Rvir ratio."""
    ratio = data.get('RcoolToRvir', np.array([]))
    regime = data.get('Regime', np.array([]))

    if len(ratio) == 0:
        return

    # Only CGM regime galaxies with valid ratios
    if len(regime) > 0:
        mask = (ratio > 0) & (ratio <= 1.0) & (regime == 0)
    else:
        mask = (ratio > 0) & (ratio <= 1.0)

    ratio = ratio[mask]
    if len(ratio) == 0:
        return

    bins = np.linspace(0, 1, 50)
    ax.hist(ratio, bins=bins, density=True, alpha=0.5,
            color=color, label=label, histtype='stepfilled')
    ax.hist(ratio, bins=bins, density=True,
            color=color, histtype='step', linewidth=1.5)


def sfr_vs_stellar(ax, data, color, label, dilute=DILUTE):
    """Plot SFR vs stellar mass."""
    stellar = data.get('StellarMass', np.array([]))
    sfr_disk = data.get('SfrDisk', np.array([]))
    sfr_bulge = data.get('SfrBulge', np.array([]))

    if len(stellar) == 0:
        return

    # Sum SFR arrays if they're 2D (multiple steps)
    if sfr_disk.ndim > 1:
        sfr_disk = np.sum(sfr_disk, axis=1)
    if sfr_bulge.ndim > 1:
        sfr_bulge = np.sum(sfr_bulge, axis=1)

    sfr = sfr_disk + sfr_bulge

    mask = (stellar > 1e7) & (sfr > 1e-4)
    stellar = stellar[mask]
    sfr = sfr[mask]

    if len(stellar) > dilute:
        np.random.seed(SEED)
        idx = np.random.choice(len(stellar), dilute, replace=False)
        stellar = stellar[idx]
        sfr = sfr[idx]

    ax.scatter(np.log10(stellar), np.log10(sfr), s=1, alpha=0.3,
               color=color, label=label, rasterized=True)


def median_relation(ax, x, y, color, label, bins=15):
    """Plot median relation with scatter."""
    if len(x) == 0 or len(y) == 0:
        return

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 10:
        return

    bin_edges = np.percentile(x, np.linspace(0, 100, bins + 1))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    medians = []
    p16 = []
    p84 = []

    for i in range(len(bin_edges) - 1):
        in_bin = (x >= bin_edges[i]) & (x < bin_edges[i + 1])
        if np.sum(in_bin) > 5:
            medians.append(np.median(y[in_bin]))
            p16.append(np.percentile(y[in_bin], 16))
            p84.append(np.percentile(y[in_bin], 84))
        else:
            medians.append(np.nan)
            p16.append(np.nan)
            p84.append(np.nan)

    medians = np.array(medians)
    p16 = np.array(p16)
    p84 = np.array(p84)

    valid = np.isfinite(medians)
    ax.plot(bin_centers[valid], medians[valid], color=color, label=label, linewidth=2)
    ax.fill_between(bin_centers[valid], p16[valid], p84[valid],
                    color=color, alpha=0.2)


def rcool_vs_mvir(ax, data, color, label, dilute=DILUTE):
    """Plot rcool/Rvir vs halo mass."""
    mvir = data.get('Mvir', np.array([]))
    rcool = data.get('RcoolToRvir', np.array([]))
    regime = data.get('Regime', np.array([]))

    if len(mvir) == 0 or len(rcool) == 0:
        return

    if len(regime) > 0:
        mask = (mvir > 1e9) & (rcool > 0) & (rcool <= 1.0) & (regime == 0)
    else:
        mask = (mvir > 1e9) & (rcool > 0) & (rcool <= 1.0)

    mvir = mvir[mask]
    rcool = rcool[mask]

    log_mvir = np.log10(mvir)
    median_relation(ax, log_mvir, rcool, color, label)


def tcool_tff_vs_mvir(ax, data, color, label, dilute=DILUTE):
    """Plot tcool/tff vs halo mass."""
    mvir = data.get('Mvir', np.array([]))
    ratio = data.get('tcool_over_tff', np.array([]))
    regime = data.get('Regime', np.array([]))

    if len(mvir) == 0 or len(ratio) == 0:
        return

    if len(regime) > 0:
        mask = (mvir > 1e9) & (ratio > 0) & (ratio < 1000) & (regime == 0)
    else:
        mask = (mvir > 1e9) & (ratio > 0) & (ratio < 1000)

    mvir = mvir[mask]
    ratio = ratio[mask]

    log_mvir = np.log10(mvir)
    log_ratio = np.log10(ratio)
    median_relation(ax, log_mvir, log_ratio, color, label)


def baryon_fractions(ax, data, color, label):
    """Plot baryon fractions vs halo mass."""
    mvir = data.get('Mvir', np.array([]))
    stellar = data.get('StellarMass', np.array([]))
    cold = data.get('ColdGas', np.array([]))
    cgm = data.get('CGMgas', np.array([]))
    hot = data.get('HotGas', np.array([]))

    if len(mvir) == 0:
        return

    mask = mvir > 1e9
    mvir = mvir[mask]
    stellar = stellar[mask] if len(stellar) == len(mask) else np.zeros_like(mvir)
    cold = cold[mask] if len(cold) == len(mask) else np.zeros_like(mvir)
    cgm = cgm[mask] if len(cgm) == len(mask) else np.zeros_like(mvir)
    hot = hot[mask] if len(hot) == len(mask) else np.zeros_like(mvir)

    total_baryon = stellar + cold + cgm + hot
    f_baryon = total_baryon / mvir

    log_mvir = np.log10(mvir)
    median_relation(ax, log_mvir, f_baryon, color, label)


# ========================== MAIN PLOTTING ==========================

def create_comparison_plots():
    """Generate all comparison plots."""

    # Check if output directories exist
    dirs_exist = []
    for name, directory in [('Uniform', UNIFORM_DIR), ('NFW', NFW_DIR), ('Beta', BETA_DIR)]:
        if os.path.exists(directory) and find_model_files(directory):
            dirs_exist.append((name, directory))
            print(f"Found {name} model in {directory}")
        else:
            print(f"Warning: {name} model not found in {directory}")

    if len(dirs_exist) < 2:
        print("Need at least 2 models to compare. Exiting.")
        return

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Read header from first available directory
    header = read_header(dirs_exist[0][1])
    if header is None:
        print("Could not read simulation header. Exiting.")
        return

    hubble_h = header['hubble_h']
    box_size = header['box_size']
    volume_fraction = header['volume_fraction']
    volume = (box_size / hubble_h)**3 * volume_fraction
    mass_convert = header['unit_mass_in_g'] / _MSUN_CGS / hubble_h
    last_snap = header['last_snap_nr']
    snapshot = f'Snap_{last_snap}'

    print(f"\nSimulation: BoxSize={box_size} Mpc/h, h={hubble_h}")
    print(f"Volume fraction: {volume_fraction:.3f}, Effective volume: {volume:.1f} Mpc^3")
    print(f"Loading snapshot: {snapshot}")

    # Load data for each model
    models = {}
    colors = {'Uniform': 'C0', 'NFW': 'C1', 'Beta': 'C2'}

    for name, directory in dirs_exist:
        print(f"\nLoading {name} model...")
        models[name] = load_model(directory, snapshot, PROPERTIES, mass_convert)
        if models[name]:
            n_gals = len(models[name].get('StellarMass', []))
            print(f"  Loaded {n_gals} galaxies")
        else:
            print(f"  Warning: No data loaded for {name}")

    # Remove empty models
    models = {k: v for k, v in models.items() if v}

    if len(models) < 2:
        print("Not enough valid models to compare. Exiting.")
        return

    setup_style()

    # ========== Figure 1: Overview (2x3 grid) ==========
    print("\nGenerating Figure 1: Overview plots...")
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # 1a: Stellar Mass Function
    ax = axes[0, 0]
    for name, data in models.items():
        stellar_mass_function(ax, data, volume, colors[name], name)
    ax.set_xlabel(r'$\log_{10}(M_\star / M_\odot)$')
    ax.set_ylabel(r'$\log_{10}(\Phi / \mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1})$')
    ax.set_xlim(7, 12)
    ax.set_ylim(-6, -1)
    ax.legend(loc='upper right')
    ax.set_title('Stellar Mass Function')

    # 1b: Cold Gas vs Stellar Mass
    ax = axes[0, 1]
    for name, data in models.items():
        cold_gas_vs_stellar(ax, data, colors[name], name)
    ax.set_xlabel(r'$\log_{10}(M_\star / M_\odot)$')
    ax.set_ylabel(r'$\log_{10}(M_\mathrm{cold} / M_\odot)$')
    ax.set_xlim(7, 12)
    ax.set_ylim(6, 11)
    ax.legend(loc='upper left', markerscale=5)
    ax.set_title('Cold Gas vs Stellar Mass')

    # 1c: CGM Gas vs Halo Mass
    ax = axes[0, 2]
    for name, data in models.items():
        cgm_gas_vs_mvir(ax, data, colors[name], name)
    ax.set_xlabel(r'$\log_{10}(M_\mathrm{vir} / M_\odot)$')
    ax.set_ylabel(r'$\log_{10}(M_\mathrm{CGM} / M_\odot)$')
    ax.set_xlim(9, 13)
    ax.set_ylim(7, 12)
    ax.legend(loc='upper left', markerscale=5)
    ax.set_title('CGM Gas vs Halo Mass (CGM regime)')

    # 1d: SFR vs Stellar Mass
    ax = axes[1, 0]
    for name, data in models.items():
        sfr_vs_stellar(ax, data, colors[name], name)
    ax.set_xlabel(r'$\log_{10}(M_\star / M_\odot)$')
    ax.set_ylabel(r'$\log_{10}(\mathrm{SFR} / M_\odot\,\mathrm{yr}^{-1})$')
    ax.set_xlim(7, 12)
    ax.set_ylim(-4, 3)
    ax.legend(loc='upper left', markerscale=5)
    ax.set_title('Star Formation Rate')

    # 1e: tcool/tff Distribution
    ax = axes[1, 1]
    for name, data in models.items():
        tcool_tff_distribution(ax, data, colors[name], name)
    ax.axvline(1.0, color='k', linestyle='--', linewidth=1, label=r'$t_\mathrm{cool}/t_\mathrm{ff} = 10$')
    ax.set_xlabel(r'$\log_{10}(t_\mathrm{cool} / t_\mathrm{ff})$')
    ax.set_ylabel('Probability Density')
    ax.set_xlim(-2, 3)
    ax.legend(loc='upper right')
    ax.set_title(r'$t_\mathrm{cool}/t_\mathrm{ff}$ Distribution (CGM regime)')

    # 1f: rcool/Rvir Distribution
    ax = axes[1, 2]
    for name, data in models.items():
        rcool_rvir_distribution(ax, data, colors[name], name)
    ax.set_xlabel(r'$r_\mathrm{cool} / R_\mathrm{vir}$')
    ax.set_ylabel('Probability Density')
    ax.set_xlim(0, 1)
    ax.legend(loc='upper left')
    ax.set_title(r'$r_\mathrm{cool}/R_\mathrm{vir}$ Distribution (CGM regime)')

    plt.tight_layout()
    outfile = os.path.join(OUTPUT_DIR, f'cgm_profile_comparison_overview{OUTPUT_FORMAT}')
    plt.savefig(outfile)
    print(f"  Saved: {outfile}")
    plt.close()

    # ========== Figure 2: Median Relations (2x2 grid) ==========
    print("\nGenerating Figure 2: Median relations...")
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    # 2a: rcool/Rvir vs Mvir
    ax = axes[0, 0]
    for name, data in models.items():
        rcool_vs_mvir(ax, data, colors[name], name)
    ax.set_xlabel(r'$\log_{10}(M_\mathrm{vir} / M_\odot)$')
    ax.set_ylabel(r'$r_\mathrm{cool} / R_\mathrm{vir}$')
    ax.set_xlim(9.5, 12.5)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right')
    ax.set_title(r'Cooling Radius vs Halo Mass')

    # 2b: tcool/tff vs Mvir
    ax = axes[0, 1]
    for name, data in models.items():
        tcool_tff_vs_mvir(ax, data, colors[name], name)
    ax.axhline(1.0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel(r'$\log_{10}(M_\mathrm{vir} / M_\odot)$')
    ax.set_ylabel(r'$\log_{10}(t_\mathrm{cool} / t_\mathrm{ff})$')
    ax.set_xlim(9.5, 12.5)
    ax.set_ylim(-1, 3)
    ax.legend(loc='upper left')
    ax.set_title(r'$t_\mathrm{cool}/t_\mathrm{ff}$ vs Halo Mass')

    # 2c: Baryon fraction vs Mvir
    ax = axes[1, 0]
    for name, data in models.items():
        baryon_fractions(ax, data, colors[name], name)
    ax.axhline(0.17, color='k', linestyle='--', linewidth=1, alpha=0.5, label='Cosmic')
    ax.set_xlabel(r'$\log_{10}(M_\mathrm{vir} / M_\odot)$')
    ax.set_ylabel(r'$f_\mathrm{baryon}$')
    ax.set_xlim(9.5, 13)
    ax.set_ylim(0, 0.25)
    ax.legend(loc='upper right')
    ax.set_title('Baryon Fraction vs Halo Mass')

    # 2d: Summary statistics
    ax = axes[1, 1]
    ax.axis('off')

    # Create summary table
    summary_text = "Summary Statistics (CGM regime galaxies):\n\n"
    summary_text += f"{'Profile':<10} {'N_gal':<10} {'<r_cool/R_vir>':<15} {'<log(t_c/t_ff)>':<15}\n"
    summary_text += "-" * 50 + "\n"

    for name, data in models.items():
        regime = data.get('Regime', np.array([]))
        rcool = data.get('RcoolToRvir', np.array([]))
        ratio = data.get('tcool_over_tff', np.array([]))

        if len(regime) > 0:
            mask = (regime == 0) & (rcool > 0) & (rcool <= 1) & (ratio > 0) & (ratio < 1000)
        else:
            mask = (rcool > 0) & (rcool <= 1) & (ratio > 0) & (ratio < 1000)

        n_gal = np.sum(mask)
        mean_rcool = np.median(rcool[mask]) if n_gal > 0 else np.nan
        mean_ratio = np.median(np.log10(ratio[mask])) if n_gal > 0 else np.nan

        summary_text += f"{name:<10} {n_gal:<10} {mean_rcool:<15.3f} {mean_ratio:<15.2f}\n"

    ax.text(0.1, 0.7, summary_text, transform=ax.transAxes,
            fontsize=11, family='monospace', verticalalignment='top')
    ax.set_title('Summary')

    plt.tight_layout()
    outfile = os.path.join(OUTPUT_DIR, f'cgm_profile_comparison_medians{OUTPUT_FORMAT}')
    plt.savefig(outfile)
    print(f"  Saved: {outfile}")
    plt.close()

    # ========== Figure 3: Detailed rcool/Rvir analysis ==========
    print("\nGenerating Figure 3: Detailed rcool analysis...")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for i, (name, data) in enumerate(models.items()):
        ax = axes[i]

        mvir = data.get('Mvir', np.array([]))
        rcool = data.get('RcoolToRvir', np.array([]))
        regime = data.get('Regime', np.array([]))

        if len(mvir) == 0 or len(rcool) == 0:
            continue

        if len(regime) > 0:
            mask = (mvir > 1e9) & (rcool > 0) & (rcool <= 1.0) & (regime == 0)
        else:
            mask = (mvir > 1e9) & (rcool > 0) & (rcool <= 1.0)

        mvir_plot = mvir[mask]
        rcool_plot = rcool[mask]

        if len(mvir_plot) > 10000:
            np.random.seed(SEED)
            idx = np.random.choice(len(mvir_plot), 10000, replace=False)
            mvir_plot = mvir_plot[idx]
            rcool_plot = rcool_plot[idx]

        # 2D histogram
        h = ax.hist2d(np.log10(mvir_plot), rcool_plot,
                      bins=[30, 30], range=[[9.5, 12.5], [0, 1]],
                      cmap='Blues', cmin=1)
        plt.colorbar(h[3], ax=ax, label='Count')

        ax.set_xlabel(r'$\log_{10}(M_\mathrm{vir} / M_\odot)$')
        ax.set_ylabel(r'$r_\mathrm{cool} / R_\mathrm{vir}$')
        ax.set_title(f'{name} Profile')

    plt.tight_layout()
    outfile = os.path.join(OUTPUT_DIR, f'cgm_profile_rcool_detail{OUTPUT_FORMAT}')
    plt.savefig(outfile)
    print(f"  Saved: {outfile}")
    plt.close()

    # ========== Detailed Statistics ==========
    print("\n" + "=" * 80)
    print("DETAILED STATISTICS")
    print("=" * 80)

    for name, data in models.items():
        print(f"\n{'=' * 40}")
        print(f"  {name} PROFILE")
        print(f"{'=' * 40}")

        regime = data.get('Regime', np.array([]))
        mvir = data.get('Mvir', np.array([]))
        stellar = data.get('StellarMass', np.array([]))
        cgm = data.get('CGMgas', np.array([]))
        cold = data.get('ColdGas', np.array([]))
        hot = data.get('HotGas', np.array([]))
        rcool = data.get('RcoolToRvir', np.array([]))
        tcool = data.get('tcool', np.array([]))
        tff = data.get('tff', np.array([]))
        ratio = data.get('tcool_over_tff', np.array([]))
        sfr_disk = data.get('SfrDisk', np.array([]))
        sfr_bulge = data.get('SfrBulge', np.array([]))

        # Calculate total SFR
        if sfr_disk.ndim > 1:
            sfr_disk = np.sum(sfr_disk, axis=1)
        if sfr_bulge.ndim > 1:
            sfr_bulge = np.sum(sfr_bulge, axis=1)
        sfr = sfr_disk + sfr_bulge if len(sfr_disk) > 0 else np.array([])

        # Overall counts
        n_total = len(mvir) if len(mvir) > 0 else 0
        n_cgm_regime = np.sum(regime == 0) if len(regime) > 0 else 0
        n_hot_regime = np.sum(regime == 1) if len(regime) > 0 else 0

        print(f"\n  Galaxy Counts:")
        print(f"    Total galaxies:      {n_total:,}")
        print(f"    CGM regime (0):      {n_cgm_regime:,} ({100*n_cgm_regime/n_total:.1f}%)")
        print(f"    Hot regime (1):      {n_hot_regime:,} ({100*n_hot_regime/n_total:.1f}%)")

        # CGM regime statistics
        if len(regime) > 0:
            cgm_mask = (regime == 0)
        else:
            cgm_mask = np.ones(len(mvir), dtype=bool)

        if np.sum(cgm_mask) > 0:
            print(f"\n  CGM Regime Statistics:")

            # Halo mass
            mvir_cgm = mvir[cgm_mask]
            print(f"\n    Halo Mass (log10 Msun):")
            print(f"      Min:    {np.log10(np.min(mvir_cgm[mvir_cgm > 0])):.2f}")
            print(f"      Max:    {np.log10(np.max(mvir_cgm)):.2f}")
            print(f"      Median: {np.log10(np.median(mvir_cgm[mvir_cgm > 0])):.2f}")
            print(f"      Mean:   {np.log10(np.mean(mvir_cgm[mvir_cgm > 0])):.2f}")

            # Stellar mass
            stellar_cgm = stellar[cgm_mask]
            stellar_valid = stellar_cgm[stellar_cgm > 0]
            if len(stellar_valid) > 0:
                print(f"\n    Stellar Mass (log10 Msun):")
                print(f"      Min:    {np.log10(np.min(stellar_valid)):.2f}")
                print(f"      Max:    {np.log10(np.max(stellar_valid)):.2f}")
                print(f"      Median: {np.log10(np.median(stellar_valid)):.2f}")

            # CGM gas
            cgm_cgm = cgm[cgm_mask]
            cgm_valid = cgm_cgm[cgm_cgm > 0]
            if len(cgm_valid) > 0:
                print(f"\n    CGM Gas Mass (log10 Msun):")
                print(f"      Min:    {np.log10(np.min(cgm_valid)):.2f}")
                print(f"      Max:    {np.log10(np.max(cgm_valid)):.2f}")
                print(f"      Median: {np.log10(np.median(cgm_valid)):.2f}")
                print(f"      N with CGMgas > 0: {len(cgm_valid):,}")

            # r_cool / R_vir
            rcool_cgm = rcool[cgm_mask]
            rcool_valid = rcool_cgm[(rcool_cgm > 0) & (rcool_cgm <= 1)]
            if len(rcool_valid) > 0:
                print(f"\n    r_cool / R_vir:")
                print(f"      Min:    {np.min(rcool_valid):.3f}")
                print(f"      Max:    {np.max(rcool_valid):.3f}")
                print(f"      Median: {np.median(rcool_valid):.3f}")
                print(f"      Mean:   {np.mean(rcool_valid):.3f}")
                print(f"      Std:    {np.std(rcool_valid):.3f}")
                print(f"      10th %%: {np.percentile(rcool_valid, 10):.3f}")
                print(f"      90th %%: {np.percentile(rcool_valid, 90):.3f}")

            # t_cool / t_ff
            ratio_cgm = ratio[cgm_mask]
            ratio_valid = ratio_cgm[(ratio_cgm > 0) & (ratio_cgm < 1000)]
            if len(ratio_valid) > 0:
                log_ratio = np.log10(ratio_valid)
                print(f"\n    t_cool / t_ff:")
                print(f"      Min:    {np.min(ratio_valid):.3f} (log: {np.min(log_ratio):.2f})")
                print(f"      Max:    {np.max(ratio_valid):.3f} (log: {np.max(log_ratio):.2f})")
                print(f"      Median: {np.median(ratio_valid):.3f} (log: {np.median(log_ratio):.2f})")
                print(f"      Mean:   {np.mean(ratio_valid):.3f} (log: {np.mean(log_ratio):.2f})")
                print(f"      10th %%: {np.percentile(ratio_valid, 10):.3f}")
                print(f"      90th %%: {np.percentile(ratio_valid, 90):.3f}")

                # Precipitation regime breakdown
                n_precip = np.sum(ratio_valid < 10)
                n_stable = np.sum(ratio_valid >= 10)
                print(f"\n    Precipitation Regime (t_cool/t_ff < 10):")
                print(f"      Precipitating: {n_precip:,} ({100*n_precip/len(ratio_valid):.1f}%)")
                print(f"      Stable:        {n_stable:,} ({100*n_stable/len(ratio_valid):.1f}%)")

            # t_cool and t_ff separately
            tcool_cgm = tcool[cgm_mask]
            tff_cgm = tff[cgm_mask]
            tcool_valid = tcool_cgm[tcool_cgm > 0]
            tff_valid = tff_cgm[tff_cgm > 0]

            if len(tcool_valid) > 0:
                print(f"\n    t_cool (code units):")
                print(f"      Median: {np.median(tcool_valid):.4f}")
                print(f"      Mean:   {np.mean(tcool_valid):.4f}")

            if len(tff_valid) > 0:
                print(f"\n    t_ff (code units):")
                print(f"      Median: {np.median(tff_valid):.4f}")
                print(f"      Mean:   {np.mean(tff_valid):.4f}")

            # SFR
            if len(sfr) > 0:
                sfr_cgm = sfr[cgm_mask]
                sfr_valid = sfr_cgm[sfr_cgm > 0]
                if len(sfr_valid) > 0:
                    print(f"\n    SFR (Msun/yr):")
                    print(f"      Median: {np.median(sfr_valid):.3f}")
                    print(f"      Mean:   {np.mean(sfr_valid):.3f}")
                    print(f"      N with SFR > 0: {len(sfr_valid):,}")

            # Mass bins analysis
            print(f"\n    Statistics by Halo Mass Bin:")
            mass_bins = [(9, 10), (10, 11), (11, 12), (12, 13)]
            print(f"    {'Mass Bin':<12} {'N':<8} {'<r_cool/Rvir>':<14} {'<log(tc/tff)>':<14} {'<log(Mcgm)>':<12}")
            print(f"    {'-'*60}")

            for m_lo, m_hi in mass_bins:
                bin_mask = cgm_mask & (mvir > 10**m_lo) & (mvir <= 10**m_hi)
                n_bin = np.sum(bin_mask)
                if n_bin > 0:
                    rc_bin = rcool[bin_mask]
                    rc_valid = rc_bin[(rc_bin > 0) & (rc_bin <= 1)]
                    rat_bin = ratio[bin_mask]
                    rat_valid = rat_bin[(rat_bin > 0) & (rat_bin < 1000)]
                    cgm_bin = cgm[bin_mask]
                    cgm_bin_valid = cgm_bin[cgm_bin > 0]

                    med_rc = np.median(rc_valid) if len(rc_valid) > 0 else np.nan
                    med_rat = np.median(np.log10(rat_valid)) if len(rat_valid) > 0 else np.nan
                    med_cgm = np.median(np.log10(cgm_bin_valid)) if len(cgm_bin_valid) > 0 else np.nan

                    print(f"    {m_lo}-{m_hi:<10} {n_bin:<8} {med_rc:<14.3f} {med_rat:<14.2f} {med_cgm:<12.2f}")
                else:
                    print(f"    {m_lo}-{m_hi:<10} {0:<8} {'--':<14} {'--':<14} {'--':<12}")

    # Investigate stable vs precipitating haloes
    print("\n" + "=" * 80)
    print("STABLE vs PRECIPITATING HALOES ANALYSIS")
    print("=" * 80)

    for name, data in models.items():
        print(f"\n  {name} Profile:")

        regime = data.get('Regime', np.array([]))
        mvir = data.get('Mvir', np.array([]))
        cgm = data.get('CGMgas', np.array([]))
        stellar = data.get('StellarMass', np.array([]))
        rcool = data.get('RcoolToRvir', np.array([]))
        ratio = data.get('tcool_over_tff', np.array([]))
        tcool = data.get('tcool', np.array([]))
        tff = data.get('tff', np.array([]))

        # CGM regime with valid data
        if len(regime) > 0:
            base_mask = (regime == 0) & (ratio > 0) & (ratio < 1000) & (rcool > 0)
        else:
            base_mask = (ratio > 0) & (ratio < 1000) & (rcool > 0)

        precip_mask = base_mask & (ratio < 10)
        stable_mask = base_mask & (ratio >= 10)

        n_precip = np.sum(precip_mask)
        n_stable = np.sum(stable_mask)

        if n_stable > 0:
            print(f"\n    Stable haloes (tcool/tff >= 10): {n_stable}")
            print(f"    Precipitating haloes (tcool/tff < 10): {n_precip}")

            # Compare properties
            print(f"\n    {'Property':<20} {'Precipitating':<18} {'Stable':<18}")
            print(f"    {'-'*56}")

            # Halo mass
            mvir_precip = mvir[precip_mask]
            mvir_stable = mvir[stable_mask]
            print(f"    {'log(Mvir) median':<20} {np.median(np.log10(mvir_precip)):<18.2f} {np.median(np.log10(mvir_stable)):<18.2f}")

            # CGM gas
            cgm_precip = cgm[precip_mask]
            cgm_stable = cgm[stable_mask]
            cgm_precip_valid = cgm_precip[cgm_precip > 0]
            cgm_stable_valid = cgm_stable[cgm_stable > 0]
            if len(cgm_precip_valid) > 0 and len(cgm_stable_valid) > 0:
                print(f"    {'log(Mcgm) median':<20} {np.median(np.log10(cgm_precip_valid)):<18.2f} {np.median(np.log10(cgm_stable_valid)):<18.2f}")
                print(f"    {'% with CGM > 0':<20} {100*len(cgm_precip_valid)/n_precip:<18.1f} {100*len(cgm_stable_valid)/n_stable:<18.1f}")

            # Stellar mass
            stellar_precip = stellar[precip_mask]
            stellar_stable = stellar[stable_mask]
            stellar_precip_valid = stellar_precip[stellar_precip > 0]
            stellar_stable_valid = stellar_stable[stellar_stable > 0]
            if len(stellar_precip_valid) > 0 and len(stellar_stable_valid) > 0:
                print(f"    {'log(Mstar) median':<20} {np.median(np.log10(stellar_precip_valid)):<18.2f} {np.median(np.log10(stellar_stable_valid)):<18.2f}")

            # r_cool
            rcool_precip = rcool[precip_mask]
            rcool_stable = rcool[stable_mask]
            print(f"    {'r_cool/Rvir median':<20} {np.median(rcool_precip):<18.3f} {np.median(rcool_stable):<18.3f}")

            # tcool and tff separately
            tcool_precip = tcool[precip_mask]
            tcool_stable = tcool[stable_mask]
            tff_precip = tff[precip_mask]
            tff_stable = tff[stable_mask]

            tcool_precip_valid = tcool_precip[tcool_precip > 0]
            tcool_stable_valid = tcool_stable[tcool_stable > 0]
            tff_precip_valid = tff_precip[tff_precip > 0]
            tff_stable_valid = tff_stable[tff_stable > 0]

            if len(tcool_precip_valid) > 0 and len(tcool_stable_valid) > 0:
                print(f"    {'tcool median':<20} {np.median(tcool_precip_valid):<18.6f} {np.median(tcool_stable_valid):<18.6f}")
            if len(tff_precip_valid) > 0 and len(tff_stable_valid) > 0:
                print(f"    {'tff median':<20} {np.median(tff_precip_valid):<18.6f} {np.median(tff_stable_valid):<18.6f}")

            # Mass distribution of stable haloes
            print(f"\n    Stable haloes by mass bin:")
            for m_lo, m_hi in [(9, 10), (10, 11), (11, 12), (12, 13)]:
                n_in_bin = np.sum(stable_mask & (mvir > 10**m_lo) & (mvir <= 10**m_hi))
                n_total_bin = np.sum(base_mask & (mvir > 10**m_lo) & (mvir <= 10**m_hi))
                if n_total_bin > 0:
                    pct = 100 * n_in_bin / n_total_bin
                    print(f"      10^{m_lo}-10^{m_hi}: {n_in_bin:,} / {n_total_bin:,} ({pct:.1f}%)")
        else:
            print(f"    No stable haloes (all precipitating)")

    # Comparison summary
    print("\n" + "=" * 80)
    print("PROFILE COMPARISON SUMMARY")
    print("=" * 80)
    print(f"\n{'Profile':<10} {'N_CGM':<10} {'<r_cool>':<10} {'<tc/tff>':<10} {'<log(tc/tff)>':<14} {'%precip':<10}")
    print("-" * 64)

    for name, data in models.items():
        regime = data.get('Regime', np.array([]))
        rcool = data.get('RcoolToRvir', np.array([]))
        ratio = data.get('tcool_over_tff', np.array([]))

        if len(regime) > 0:
            mask = (regime == 0) & (rcool > 0) & (rcool <= 1) & (ratio > 0) & (ratio < 1000)
        else:
            mask = (rcool > 0) & (rcool <= 1) & (ratio > 0) & (ratio < 1000)

        n_gal = np.sum(mask)
        if n_gal > 0:
            med_rcool = np.median(rcool[mask])
            med_ratio = np.median(ratio[mask])
            med_log_ratio = np.median(np.log10(ratio[mask]))
            pct_precip = 100 * np.sum(ratio[mask] < 10) / n_gal
            print(f"{name:<10} {n_gal:<10} {med_rcool:<10.3f} {med_ratio:<10.3f} {med_log_ratio:<14.2f} {pct_precip:<10.1f}")

    print("=" * 80)
    print(f"\nAll plots saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    create_comparison_plots()

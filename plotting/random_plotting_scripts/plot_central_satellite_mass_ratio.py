#!/usr/bin/env python
"""
Plot CentralMvir / Mvir (mass ratio) vs lookback time for all satellite haloes.

Iterates over all snapshots, selects satellites (Type == 1), and produces a
2D density plot of the central-to-satellite virial mass ratio as a function
of lookback time.

Usage:
    python plot_central_satellite_mass_ratio.py
"""

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy.integrate import quad

from random import sample, seed as rseed

import warnings
warnings.filterwarnings("ignore")

DILUTE = 7500
SEED = 2222

# ========================== CONFIGURATION ==========================

PRIMARY_DIR = './output/millennium/'
OUTPUT_DIR = './output/millennium/plots/'
OUTPUT_FORMAT = '.pdf'

# Properties to load per snapshot
PROPERTIES = ['Mvir', 'CentralMvir', 'Type', 'StellarMass', 'mergeType',
              'infallStellarMass', 'infallMvir',
              'GalaxyIndex', 'CentralGalaxyIndex']

# Mass properties that need MASS_CONVERT (both are mass, so their ratio
# is dimensionless — but we still need them converted for any filtering)
_MASS_PROPS = frozenset({
    'CentralMvir', 'Mvir', 'StellarMass', 'BulgeMass', 'BlackHoleMass',
    'MetalsStellarMass', 'MetalsColdGas', 'MetalsEjectedMass',
    'MetalsHotGas', 'MetalsCGMgas', 'ColdGas', 'HotGas', 'CGMgas',
    'EjectedMass', 'H2gas', 'H1gas', 'IntraClusterStars',
    'MergerBulgeMass', 'InstabilityBulgeMass',
    'infallStellarMass', 'infallMvir',
})


# ========================== READ HEADER ==========================

def read_header(directory):
    """Read simulation parameters from the HDF5 header."""
    pattern = os.path.join(directory, 'model_*.hdf5')
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No model files in {directory}")

    with h5.File(files[0], 'r') as f:
        sim = f['Header/Simulation']
        runtime = f['Header/Runtime']
        header = {
            'hubble_h':     float(sim.attrs['hubble_h']),
            'omega_matter': float(sim.attrs['omega_matter']),
            'omega_lambda': float(sim.attrs['omega_lambda']),
            'redshifts':    list(f['Header/snapshot_redshifts'][:]),
            'output_snaps': [int(s) for s in f['Header/output_snapshots'][:]],
            'mass_convert': float(runtime.attrs['UnitMass_in_g']) / 1.989e33
                            / float(sim.attrs['hubble_h']),
        }
    return header, files


HEADER, MODEL_FILES = read_header(PRIMARY_DIR)
HUBBLE_H     = HEADER['hubble_h']
OMEGA_M      = HEADER['omega_matter']
OMEGA_L      = HEADER['omega_lambda']
MASS_CONVERT = HEADER['mass_convert']
REDSHIFTS    = HEADER['redshifts']
OUTPUT_SNAPS = sorted(HEADER['output_snaps'])


# ========================== COSMOLOGY ==========================

def cosmic_time_gyr(z):
    """Age of the universe at redshift z, in Gyr."""
    t_H = 977.8 / (HUBBLE_H * 100)

    def integrand(zp):
        return 1.0 / ((1 + zp) * np.sqrt(OMEGA_M * (1 + zp)**3 + OMEGA_L))

    result, _ = quad(integrand, z, 1000.0)
    return t_H * result

AGE_NOW = cosmic_time_gyr(0.0)

def lookback_time_gyr(z):
    """Lookback time to redshift z from z=0, in Gyr."""
    return AGE_NOW - cosmic_time_gyr(z)


# ========================== PLOTTING STYLE ==========================

def setup_style():
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["font.size"] = 14
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.major.size'] = 6
    plt.rcParams['ytick.major.size'] = 6
    plt.rcParams['xtick.minor.size'] = 3
    plt.rcParams['ytick.minor.size'] = 3
    plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['ytick.minor.visible'] = True
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['axes.labelcolor'] = 'black'
    plt.rcParams['text.color'] = 'black'
    plt.rcParams['legend.facecolor'] = 'white'
    plt.rcParams['legend.edgecolor'] = 'none'
    plt.rcParams['legend.framealpha'] = 0.8
    plt.rcParams['legend.fontsize'] = 'medium'
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Palatino'
    plt.rcParams['text.usetex'] = True


# ========================== DATA I/O ==========================

def read_snap_from_files(filepaths, snap_key, properties):
    """Read properties from a snapshot across multiple HDF5 files."""
    chunks = {prop: [] for prop in properties}
    found = False

    for fp in filepaths:
        try:
            with h5.File(fp, 'r') as f:
                if snap_key not in f:
                    continue
                found = True
                grp = f[snap_key]
                for prop in properties:
                    if prop in grp:
                        chunks[prop].append(np.array(grp[prop]))
        except Exception as e:
            print(f"  Warning: could not read {fp}: {e}")

    if not found:
        return {}

    data = {}
    for prop in properties:
        if chunks[prop]:
            arr = np.concatenate(chunks[prop])
            if prop in _MASS_PROPS:
                arr *= MASS_CONVERT
            data[prop] = arr
    return data


# ========================== MAIN ==========================

def plot_mass_ratio_vs_lookback():
    setup_style()

    # Collect satellite events across all snapshots:
    #   mergers (mergeType=1,2): stars go to central — filled, colored by fraction
    #   ICS disruptions (mergeType=4): stars go to ICL, zero to central — open circles
    merger_ratios = []
    merger_lookback = []
    merger_frac = []
    ics_ratios = []
    ics_lookback = []

    snap_min = 5

    for snap in OUTPUT_SNAPS:
        if snap < snap_min:
            continue

        z = REDSHIFTS[snap]
        t_lb = lookback_time_gyr(z)

        snap_key = f'Snap_{snap}'
        data = read_snap_from_files(MODEL_FILES, snap_key, PROPERTIES)

        if not data or 'mergeType' not in data:
            continue

        cmvir = data['CentralMvir']
        smass = data['StellarMass']
        mtype = data['mergeType']
        gtype = data['Type']
        infall_mvir = data['infallMvir']
        infall_smass = data['infallStellarMass']

        pos_mass = (gtype == 1) & (infall_mvir > 0) & (cmvir > 0)

        # Satellite mergers (stars go to central)
        gidx = data['GalaxyIndex']
        cgidx = data['CentralGalaxyIndex']
        centrals = gtype == 0
        central_lookup = dict(zip(gidx[centrals], smass[centrals]))

        sel_merger = pos_mass & ((mtype == 1) | (mtype == 2))
        n_merger = np.sum(sel_merger)
        if n_merger > 0:
            ratio = cmvir[sel_merger] / infall_mvir[sel_merger]
            sat_infall_sm = infall_smass[sel_merger]
            central_sm = np.array([central_lookup.get(cg, np.nan)
                                   for cg in cgidx[sel_merger]])
            combined = sat_infall_sm + central_sm
            valid = (combined > 0) & np.isfinite(combined)
            if np.sum(valid) > 0:
                frac = sat_infall_sm[valid] / combined[valid]
                merger_ratios.append(np.log10(ratio[valid]))
                merger_lookback.append(np.full(np.sum(valid), t_lb))
                merger_frac.append(frac)
                n_merger = np.sum(valid)

        # ICS disruptions (zero stellar contribution to central)
        sel_ics = pos_mass & (mtype == 4)
        n_ics = np.sum(sel_ics)
        if n_ics > 0:
            ratio_ics = cmvir[sel_ics] / infall_mvir[sel_ics]
            ics_ratios.append(np.log10(ratio_ics))
            ics_lookback.append(np.full(n_ics, t_lb))

        print(f"  Snap {snap:2d}  z={z:.2f}  t_lb={t_lb:.2f} Gyr  "
              f"N_mergers={n_merger:,}  N_ics={n_ics:,}")

    merger_ratios = np.concatenate(merger_ratios)
    merger_lookback = np.concatenate(merger_lookback)
    merger_frac = np.concatenate(merger_frac)
    ics_ratios = np.concatenate(ics_ratios) if ics_ratios else np.array([])
    ics_lookback = np.concatenate(ics_lookback) if ics_lookback else np.array([])

    print(f"\nSatellite mergers: {len(merger_ratios):,}")
    print(f"ICS disruptions:   {len(ics_ratios):,}")

    # --- Dilute each population to DILUTE points ---
    def dilute(n, *arrays):
        if n <= DILUTE:
            return arrays
        rseed(SEED)
        idx = sorted(sample(range(n), DILUTE))
        return tuple(a[idx] for a in arrays)

    merger_ratios, merger_lookback, merger_frac = dilute(
        len(merger_ratios), merger_ratios, merger_lookback, merger_frac)
    if len(ics_ratios) > 0:
        ics_ratios, ics_lookback = dilute(
            len(ics_ratios), ics_ratios, ics_lookback)

    print(f"After dilution: {len(merger_ratios)} mergers, {len(ics_ratios)} ICS")

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(9, 6))

    max_ratio = max(merger_ratios.max() if len(merger_ratios) else 0,
                    ics_ratios.max() if len(ics_ratios) else 0)

    # Open circles: ICS disruptions (zero stellar contribution to central)
    if len(ics_ratios) > 0:
        sizes_ics = 25 + 0.1 * 10**(max_ratio - ics_ratios)
        ax.scatter(ics_lookback, ics_ratios, s=sizes_ics, alpha=0.4,
                   marker='o', facecolors='none', edgecolors='grey',
                   linewidths=0.5, rasterized=True, label='ICS disruption (zero)')

    # Filled circles: mergers colored by fraction of BCG mass
    sizes_m = 25 + 0.1 * 10**(max_ratio - merger_ratios)
    im = ax.scatter(merger_lookback, merger_ratios, s=sizes_m, alpha=0.5,
                    marker='o', edgecolors='none', rasterized=True,
                    c=merger_frac, cmap='viridis', vmin=0, vmax=0.5,
                    label='Merger')
    ax.invert_yaxis()

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(r'$M_{*,\rm sat}^{\rm infall} \,/\, (M_{*,\rm sat}^{\rm infall} + M_{*,\rm central})$')

    ax.legend(loc='lower left', fontsize=11, framealpha=0.9)

    # Redshift ticks on top axis
    ax2 = ax.twiny()
    z_ticks = [0, 0.5, 1, 2, 3, 5, 8]
    t_ticks = [lookback_time_gyr(z) for z in z_ticks]
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(t_ticks)
    ax2.set_xticklabels([f'{z}' for z in z_ticks])
    ax2.set_xlabel(r'$z$')

    ax.set_xlabel(r'Lookback time [Gyr]')
    ax.set_ylabel(r'$\log_{10}\,(M_{\rm central} \,/\, M_{\rm sat,\,infall})$')

    fig.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    outpath = os.path.join(OUTPUT_DIR, f'central_satellite_mass_ratio{OUTPUT_FORMAT}')
    fig.savefig(outpath, dpi=200, bbox_inches='tight')
    print(f"\nSaved: {outpath}")
    plt.close()


if __name__ == '__main__':
    plot_mass_ratio_vs_lookback()
